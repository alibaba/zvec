// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This translation unit hosts the AVX512-accelerated kernels for the
// SymphonyQG query path. The build system compiles this .cc with an
// AVX512-capable -march when the toolchain/host supports it (see
// src/turbo/CMakeLists.txt). Every entry point runtime-gates on CpuFeatures
// because a binary built on an AVX512 host may run on a lower-arch machine.

#include <ailego/internal/cpu_features.h>
#include <turbo/distance/common/symphony_qg_kernels.h>
#include <zvec/ailego/internal/platform.h>

#if defined(__AVX512F__) && defined(__AVX512BW__)
#include <immintrin.h>

namespace zvec::turbo {
namespace {

bool symqg_avx512_available() {
  static const bool available =
      ailego::internal::CpuFeatures::static_flags_.AVX512F &&
      ailego::internal::CpuFeatures::static_flags_.AVX512BW;
  return available;
}

}  // namespace

bool symqg_lut_build(const float *__restrict q, size_t padded_dim, uint8_t *lut,
                     float *delta, float *sum_vl, float *k1xsumq) {
  if (!symqg_avx512_available() || (padded_dim & 0xF) != 0) return false;

  float vl = 0.0f, vr = 0.0f, sumq = 0.0f;
  const __m512 zero = _mm512_setzero_ps();
  const __m512i pair =
      _mm512_setr_epi32(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);
  const __m512i quad =
      _mm512_setr_epi32(2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13);
  for (size_t i = 0; i < padded_dim; i += 16) {
    const __m512 x = _mm512_loadu_ps(q + i);
    sumq += _mm512_reduce_add_ps(x);
    // Per-codebook (groups of 4) sums of negatives/positives bound the LUT's
    // subset sums: the 16 entries of a codebook are exactly all subsets.
    __m512 mn = _mm512_min_ps(x, zero);
    __m512 mx = _mm512_max_ps(x, zero);
    mn = _mm512_add_ps(mn, _mm512_permutexvar_ps(pair, mn));
    mn = _mm512_add_ps(mn, _mm512_permutexvar_ps(quad, mn));
    mx = _mm512_add_ps(mx, _mm512_permutexvar_ps(pair, mx));
    mx = _mm512_add_ps(mx, _mm512_permutexvar_ps(quad, mx));
    vl = vl < _mm512_reduce_min_ps(mn) ? vl : _mm512_reduce_min_ps(mn);
    vr = vr > _mm512_reduce_max_ps(mx) ? vr : _mm512_reduce_max_ps(mx);
  }
  *delta = (vr - vl) / 255.0f;
  *sum_vl = vl * static_cast<float>(padded_dim >> 2);
  *k1xsumq = sumq * -0.5f;

  const __m512 lo = _mm512_set1_ps(vl);
  const __m512 od = _mm512_set1_ps(1.0f / *delta);
  uint8_t *out = lut;
  for (size_t cb = 0; cb < (padded_dim >> 2); ++cb, q += 4, out += 16) {
    const __m512 va = _mm512_set1_ps(q[0]);
    const __m512 vb = _mm512_set1_ps(q[1]);
    const __m512 vc = _mm512_set1_ps(q[2]);
    const __m512 vd = _mm512_set1_ps(q[3]);
    // Lane j holds the subset sum of the four query values selected by j's
    // bits (bit 3 -> q[0] ... bit 0 -> q[3]), matching kPos in fastscan.
    __m512 s = _mm512_mask_blend_ps(0xAAAA, zero, vd);
    s = _mm512_mask_blend_ps(0xCCCC, s, _mm512_add_ps(s, vc));
    s = _mm512_mask_blend_ps(0xF0F0, s, _mm512_add_ps(s, vb));
    s = _mm512_mask_blend_ps(0xFF00, s, _mm512_add_ps(s, va));
    const __m512 t = _mm512_mul_ps(_mm512_sub_ps(s, lo), od);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out),
                     _mm512_cvtusepi32_epi8(_mm512_cvtps_epi32(t)));
  }
  return true;
}

bool symqg_hadamard(float *v, size_t padded_dim) {
  if (!symqg_avx512_available() || (padded_dim & 0xF) != 0) return false;

  for (size_t stride = 1; stride < padded_dim; stride *= 2) {
    if (stride >= 16) {
      for (size_t offset = 0; offset < padded_dim; offset += 2 * stride) {
        for (size_t i = 0; i < stride; i += 16) {
          const __m512 a = _mm512_loadu_ps(v + offset + i);
          const __m512 b = _mm512_loadu_ps(v + offset + i + stride);
          _mm512_storeu_ps(v + offset + i, _mm512_add_ps(a, b));
          _mm512_storeu_ps(v + offset + i + stride, _mm512_sub_ps(a, b));
        }
      }
      continue;
    }
    // A whole butterfly group fits in one vector: permute pairs lane i with
    // i XOR stride, then a masked subtract yields both (a+b, a-b) outputs in
    // a single pass, bit-identical to the scalar formulation.
    __m512i partner;
    __mmask16 high;
    switch (stride) {
      case 8:
        partner = _mm512_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4,
                                    5, 6, 7);
        high = 0xFF00;
        break;
      case 4:
        partner = _mm512_setr_epi32(4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8,
                                    9, 10, 11);
        high = 0xF0F0;
        break;
      case 2:
        partner = _mm512_setr_epi32(2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14,
                                    15, 12, 13);
        high = 0xCCCC;
        break;
      default:
        partner = _mm512_setr_epi32(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13,
                                    12, 15, 14);
        high = 0xAAAA;
        break;
    }
    for (size_t i = 0; i < padded_dim; i += 16) {
      const __m512 t = _mm512_loadu_ps(v + i);
      const __m512 u = _mm512_permutexvar_ps(partner, t);
      _mm512_storeu_ps(v + i,
                       _mm512_mask_sub_ps(_mm512_add_ps(t, u), high, u, t));
    }
  }
  return true;
}

bool symqg_mul(const float *factors, float *v, size_t n) {
  if (!symqg_avx512_available()) return false;
  for (size_t i = 0; i + 16 <= n; i += 16) {
    _mm512_storeu_ps(v + i, _mm512_mul_ps(_mm512_loadu_ps(v + i),
                                          _mm512_loadu_ps(factors + i)));
  }
  for (size_t i = n & ~size_t{15}; i < n; ++i) v[i] *= factors[i];
  return true;
}

bool symqg_scan(const uint8_t *codes, const uint8_t *lut, size_t padded_dim,
                const float *add, const float *scale, float center_distance,
                float delta, float sum_vl, float k1xsumq, float *distances) {
  if (!symqg_avx512_available() || padded_dim == 0 || (padded_dim & 15) != 0)
    return false;

  __m512i sums0 = _mm512_setzero_si512();
  __m512i sums1 = _mm512_setzero_si512();
  const __m512i nibble = _mm512_set1_epi8(15);
  // Same packed layout and shuffle accumulation as RaBitQ's FastScan. Keep
  // sums in registers through widening and score conversion, avoiding the
  // generic path's temporary uint16/uint32 arrays and per-lane conversion.
  for (size_t dim = 0; dim < padded_dim; dim += 1024) {
    const size_t slice = (padded_dim - dim < 1024) ? padded_dim - dim : 1024;
    __m512i a0 = _mm512_setzero_si512(), a1 = _mm512_setzero_si512();
    __m512i a2 = _mm512_setzero_si512(), a3 = _mm512_setzero_si512();
    for (size_t offset = dim * 4; offset < (dim + slice) * 4; offset += 64) {
      const __m512i c = _mm512_loadu_si512(codes + offset);
      const __m512i table = _mm512_loadu_si512(lut + offset);
      const __m512i lo =
          _mm512_shuffle_epi8(table, _mm512_and_si512(c, nibble));
      const __m512i hi = _mm512_shuffle_epi8(
          table, _mm512_and_si512(_mm512_srli_epi16(c, 4), nibble));
      a0 = _mm512_add_epi16(a0, lo);
      a1 = _mm512_add_epi16(a1, _mm512_srli_epi16(lo, 8));
      a2 = _mm512_add_epi16(a2, hi);
      a3 = _mm512_add_epi16(a3, _mm512_srli_epi16(hi, 8));
    }
    a0 = _mm512_sub_epi16(a0, _mm512_slli_epi16(a1, 8));
    a2 = _mm512_sub_epi16(a2, _mm512_slli_epi16(a3, 8));
    const __m512i r0 = _mm512_add_epi16(_mm512_mask_blend_epi64(0xF0, a0, a1),
                                        _mm512_shuffle_i64x2(a0, a1, 0x4E));
    const __m512i r1 = _mm512_add_epi16(_mm512_mask_blend_epi64(0xF0, a2, a3),
                                        _mm512_shuffle_i64x2(a2, a3, 0x4E));
    const __m512i sum = _mm512_add_epi16(_mm512_shuffle_i64x2(r0, r1, 0x88),
                                         _mm512_shuffle_i64x2(r0, r1, 0xDD));
    sums0 = _mm512_add_epi32(
        sums0, _mm512_cvtepu16_epi32(_mm512_castsi512_si256(sum)));
    sums1 = _mm512_add_epi32(
        sums1, _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(sum, 1)));
  }

  const __m512 center = _mm512_set1_ps(center_distance);
  const __m512 width = _mm512_set1_ps(delta);
  const __m512 vl = _mm512_set1_ps(sum_vl);
  const __m512 sq = _mm512_set1_ps(k1xsumq);
  for (size_t i = 0; i < 32; i += 16) {
    __m512 estimate =
        _mm512_mul_ps(width, _mm512_cvtepi32_ps(i == 0 ? sums0 : sums1));
    estimate = _mm512_add_ps(estimate, vl);
    estimate = _mm512_add_ps(estimate, sq);
    estimate = _mm512_mul_ps(_mm512_loadu_ps(scale + i), estimate);
    const __m512 base = _mm512_add_ps(_mm512_loadu_ps(add + i), center);
    _mm512_storeu_ps(distances + i, _mm512_add_ps(base, estimate));
  }
  return true;
}

}  // namespace zvec::turbo

#else

namespace zvec::turbo {

bool symqg_lut_build(const float *, size_t, uint8_t *, float *, float *,
                     float *) {
  return false;
}

bool symqg_hadamard(float *, size_t) {
  return false;
}

bool symqg_mul(const float *, float *, size_t) {
  return false;
}

bool symqg_scan(const uint8_t *, const uint8_t *, size_t, const float *,
                const float *, float, float, float, float, float *) {
  return false;
}

}  // namespace zvec::turbo

#endif  // __AVX512F__ && __AVX512BW__
