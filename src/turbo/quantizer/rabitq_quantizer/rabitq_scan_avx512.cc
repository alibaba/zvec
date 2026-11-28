// Copyright 2026-present the zvec project
// Licensed under the Apache License, Version 2.0.
#include "rabitq_scan_internal.h"

#if defined(__AVX512F__)
#include <immintrin.h>
#include <rabitqlib/fastscan/data_layout.hpp>

namespace zvec::turbo::rabitq_internal {
int PruneBlockAvx512(const void *block, size_t dim, size_t clusters, int bits,
                     bool inner_product, const float *scalars,
                     const rabitqlib::SplitBatchQuery<float> &query,
                     const int32_t *sums, float worst, DistanceEstimate *out,
                     uint32_t *candidates) {
  *candidates = 0;
  const auto *bytes = static_cast<const char *>(block);
  const auto limit = _mm512_set1_epi32(static_cast<int>(clusters));
  const __m512i ids[2] = {_mm512_loadu_si512(bytes),
                          _mm512_loadu_si512(bytes + 16 * sizeof(uint32_t))};
  // Validate every active model ID, including lanes that pruning rejects.
  if (_mm512_cmp_epu32_mask(ids[0], limit, _MM_CMPINT_NLT) ||
      _mm512_cmp_epu32_mask(ids[1], limit, _MM_CMPINT_NLT))
    return kErrInvalidArgument;
  const float *g_add = scalars + 4;
  const float *g_error = g_add + clusters;
  const __m512 delta = _mm512_set1_ps(query.delta());
  const __m512 sum_vl = _mm512_set1_ps(query.sum_vl_lut());
  const __m512 k1 = _mm512_set1_ps(query.k1xsumq());
  const __m512 threshold = _mm512_set1_ps(worst);
  // 0.7 exposes packed factor views instead of float pointers. SIMD loads
  // can still read the serialized bytes directly using its layout offsets.
  using Offsets = rabitqlib::detail::BatchCodeOffsets<float>;
  const char *factors = bytes + 32 * sizeof(uint32_t);
  for (size_t half = 0; half < 2; ++half) {
    const size_t first = half * 16;
    const __m512 dot = _mm512_fmadd_ps(
        delta, _mm512_cvtepi32_ps(_mm512_load_si512(sums + first)), sum_vl);
    const __m512 add = _mm512_i32gather_ps(ids[half], g_add, 4);
    __m512 score = _mm512_fmadd_ps(
        _mm512_loadu_ps(factors + Offsets::rescale(dim) +
                        first * sizeof(float)),
        _mm512_add_ps(dot, k1),
        _mm512_add_ps(_mm512_loadu_ps(factors + Offsets::add(dim) +
                                      first * sizeof(float)),
                      add));
    if (inner_product) score = _mm512_sub_ps(score, _mm512_set1_ps(1));
    const __m512 lower =
        bits == 1 ? score
                  : _mm512_fnmadd_ps(
                        _mm512_loadu_ps(factors + Offsets::error(dim) +
                                        first * sizeof(float)),
                        _mm512_i32gather_ps(ids[half], g_error, 4), score);
    uint32_t mask = _mm512_cmp_ps_mask(lower, threshold, _CMP_NGT_UQ);
    *candidates |= mask << first;
    if (!mask) continue;
    // Most blocks have no candidates. Avoid writing 32 AoS estimates and
    // gathering their bounds back just to discover an empty mask.
    alignas(64) float scores[16], bounds[16], dots[16];
    _mm512_store_ps(scores, score);
    _mm512_store_ps(bounds, lower);
    _mm512_store_ps(dots, dot);
    while (mask) {
      const size_t lane = ailego_ctz32(mask);
      mask &= mask - 1;
      out[first + lane] = {scores[lane], bounds[lane], dots[lane]};
    }
  }
  return 0;
}
}  // namespace zvec::turbo::rabitq_internal
#endif
