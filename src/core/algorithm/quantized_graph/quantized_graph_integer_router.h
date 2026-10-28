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
#pragma once
#include <immintrin.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>
#include "quantized_graph_integer_center.h"
#include "quantized_graph_router_format.h"

namespace zvec::core {
// Database-trained hierarchy. Scores only select a graph entry;
// they provide no recall guarantee and do not replace terminal refinement.
class QuantizedGraphIntegerRouter {
 public:
  static constexpr size_t kDim = 384;
  static constexpr uint32_t kVersion = 2, kTransform = 0x53514701;
  static constexpr float kFactor = 768.f / float(kDim);
  static constexpr size_t kWidth = 16, kMaxBeam = 4;
  static constexpr int32_t kInvalid = INT32_MIN;
  struct alignas(64) Group {
    std::vector<uint8_t> codes;
    float scale[kWidth]{}, offset[kWidth]{}, mean_dot[kWidth]{};
    int32_t target[kWidth]{};
    uint32_t count = 0;
  };
  void clear() {
    groups_.clear();
  }
  bool empty() const {
    return groups_.empty();
  }
  size_t group_count() const {
    return groups_.size();
  }
  const Group &group(size_t i) const {
    return groups_[i];
  }
  size_t dimension() const {
    return format_.dimension;
  }
  size_t projected_dimension() const {
    return format_.projected;
  }
  size_t code_stride() const {
    return format_.stride;
  }
  bool cosine() const {
    return format_.metric == 0;
  }
  static bool read_mean(const char *path, size_t docs, std::vector<float> &mean,
                        size_t dimension = 768, bool cosine = true) {
    std::string data;
    return ReadQuantizedGraphRouterFile(path, data) &&
           read_mean_data(data, docs, mean, dimension, cosine);
  }
  static bool read_mean_data(std::string_view data, size_t docs,
                             std::vector<float> &mean, size_t dimension = 768,
                             bool cosine = true) {
    QuantizedGraphRouterReader read{data};
    QuantizedGraphRouterFormat header;
    bool ok = header.read_bytes(read) && header.docs == docs &&
              header.dimension == dimension &&
              header.metric == (cosine ? 0u : 1u);
    std::vector<float> parsed(ok ? dimension : 0);
    ok = ok && read(parsed.data(), dimension * sizeof(float));
    for (float value : parsed) ok = ok && std::isfinite(value);
    if (ok) mean = std::move(parsed);
    return ok;
  }
  template <class Graph>
  bool load(const char *path, const std::vector<float> &mean,
            const Graph &entity, size_t docs,
            const std::vector<float> *rotated_mean = nullptr,
            bool cosine = true) {
    std::string data;
    if (!ReadQuantizedGraphRouterFile(path, data)) {
      clear();
      return false;
    }
    return load_data(data, mean, entity, docs, rotated_mean, cosine);
  }
  template <class Graph>
  bool load_data(std::string_view data, const std::vector<float> &mean,
                 const Graph &entity, size_t docs,
                 const std::vector<float> *rotated_mean = nullptr,
                 bool cosine = true) {
    clear();
    if (!__builtin_cpu_supports("avx512vnni")) return false;
    QuantizedGraphRouterReader read{data};
    QuantizedGraphRouterFormat header;
    bool ok = header.read_bytes(read) && header.docs == docs &&
              header.dimension == mean.size() &&
              header.metric == (cosine ? 0u : 1u);
    const uint64_t expected =
        read.offset +
        uint64_t(header.dimension + header.rotated) * sizeof(float) + 4 +
        uint64_t(header.groups) * (4 + 16 * 16 + uint64_t(16) * header.stride);
    ok = ok && data.size() == expected;
    std::vector<float> stored_mean(ok ? header.dimension : 0);
    ok = ok && read(stored_mean.data(), stored_mean.size() * sizeof(float)) &&
         std::memcmp(stored_mean.data(), mean.data(),
                     stored_mean.size() * sizeof(float)) == 0;
    uint32_t tag = 0;
    std::vector<float> projected_mean(ok ? header.rotated : 0);
    ok = ok && rotated_mean && rotated_mean->size() == header.rotated &&
         read(&tag, sizeof(tag)) && tag == kTransform &&
         read(projected_mean.data(), projected_mean.size() * sizeof(float)) &&
         std::memcmp(projected_mean.data(), rotated_mean->data(),
                     projected_mean.size() * sizeof(float)) == 0;
    for (float v : stored_mean) ok = ok && std::isfinite(v);
    for (float v : projected_mean) ok = ok && std::isfinite(v);
    std::vector<Group> parsed(ok ? header.groups : 0);
    std::vector<unsigned> parents(parsed.size(), 0), depth(parsed.size(), 0);
    for (size_t i = 0; ok && i < parsed.size(); ++i) {
      Group &g = parsed[i];
      g.codes.resize(kWidth * header.stride);
      ok = read(&g.count, sizeof(g.count)) && g.count > 0 &&
           g.count <= kWidth && read(g.target, sizeof(g.target)) &&
           read(g.scale, sizeof(g.scale)) && read(g.offset, sizeof(g.offset)) &&
           read(g.mean_dot, sizeof(g.mean_dot)) &&
           read(g.codes.data(), g.codes.size());
      for (size_t j = 0; ok && j < g.count; ++j) {
        ok = std::isfinite(g.scale[j]) && g.scale[j] >= 0 &&
             std::isfinite(g.offset[j]) && std::isfinite(g.mean_dot[j]);
        const int32_t t = g.target[j];
        if (t >= 0) {
          ok = ok && size_t(t) > i && size_t(t) < parsed.size();
          if (ok) {
            ok = ++parents[t] == 1;
            depth[t] = depth[i] + 1;
            ok = ok && depth[t] < 16;
          }
        } else {
          const uint32_t id = uint32_t(-int64_t(t) - 1);
          ok = ok && t != kInvalid && id < docs && entity.valid(id);
        }
      }
      if (ok && i > 0) ok = parents[i] == 1;
    }
    ok = ok && read.offset == data.size();
    if (ok) {
      format_ = header;
      projected_mean_ = std::move(projected_mean);
      groups_ = std::move(parsed);
    }
    return ok;
  }
  template <size_t FixedStride = 0>
  __attribute__((target("avx512f,avx512bw,avx512vnni,fma"))) void score_impl(
      const Group &g, const IntegerCenter::Query &q, float *scores,
      int32_t *products = nullptr) const {
    const size_t stride = FixedStride ? FixedStride : format_.stride;
    __m512i acc[16];
#pragma GCC unroll 16
    for (size_t i = 0; i < 16; ++i) acc[i] = _mm512_setzero_si512();
    for (size_t d = 0; d < stride; d += 64) {
      const __m512i query = _mm512_loadu_si512(q.codes.data() + d);
#pragma GCC unroll 16
      for (size_t i = 0; i < 16; ++i)
        acc[i] = _mm512_dpbusd_epi32(
            acc[i], _mm512_loadu_si512(g.codes.data() + i * stride + d), query);
    }
    alignas(64) int32_t dots[16];
#pragma GCC unroll 16
    for (size_t i = 0; i < 16; ++i) dots[i] = _mm512_reduce_add_epi32(acc[i]);
    if (products) std::memcpy(products, dots, sizeof(dots));
    const __m512 residual = _mm512_fmadd_ps(
        _mm512_loadu_ps(g.scale), _mm512_cvtepi32_ps(_mm512_load_si512(dots)),
        _mm512_mul_ps(_mm512_loadu_ps(g.offset),
                      _mm512_set1_ps(float(q.code_sum))));
    const float factor = float(format_.rotated) / format_.projected;
    const __m512 value = _mm512_fmadd_ps(
        _mm512_set1_ps(q.scale * factor * (cosine() ? 1.f : -2.f)), residual,
        _mm512_loadu_ps(g.mean_dot));
    _mm512_storeu_ps(
        scores, cosine() ? _mm512_sub_ps(_mm512_setzero_ps(), value) : value);
    for (size_t i = g.count; i < 16; ++i)
      scores[i] = std::numeric_limits<float>::infinity();
  }
  void score(const Group &g, const IntegerCenter::Query &q, float *scores,
             int32_t *products = nullptr) const {
    switch (format_.stride) {
      case 384:
        score_impl<384>(g, q, scores, products);
        break;
      case 512:
        score_impl<512>(g, q, scores, products);
        break;
      case 768:
        score_impl<768>(g, q, scores, products);
        break;
      case 1024:
        score_impl<1024>(g, q, scores, products);
        break;
      default:
        score_impl(g, q, scores, products);
        break;
    }
  }
  // Heuristic routing codes only: no terminal-refinement error bound is
  // implied.
  __attribute__((target("avx512f,avx512bw"))) bool prepare_projected(
      const float *rotated_query, IntegerCenter::Query &q) const {
    if (!rotated_query || empty()) return false;
    const size_t projected = format_.projected;
    const size_t vectorized = projected & ~size_t{15};
    q.codes.resize(format_.stride);
    std::fill(q.codes.begin() + vectorized, q.codes.end(), int8_t{0});
    __m512 maximum = _mm512_setzero_ps();
    for (size_t d = 0; d < vectorized; d += 16) {
      const __m512 r =
          _mm512_sub_ps(_mm512_loadu_ps(rotated_query + d),
                        _mm512_loadu_ps(projected_mean_.data() + d));
      if (_mm512_cmp_ps_mask(r, r, _CMP_UNORD_Q)) return false;
      maximum =
          _mm512_max_ps(maximum, _mm512_castsi512_ps(_mm512_andnot_si512(
                                     _mm512_set1_epi32(int32_t(0x80000000u)),
                                     _mm512_castps_si512(r))));
    }
    float max_abs = _mm512_reduce_max_ps(maximum);
    for (size_t d = vectorized; d < projected; ++d) {
      const float r = rotated_query[d] - projected_mean_[d];
      if (!std::isfinite(r)) return false;
      max_abs = std::max(max_abs, std::abs(r));
    }
    q.scale = max_abs / 127.f;
    if (!std::isfinite(q.scale)) return false;
    const __m512 inverse = _mm512_set1_ps(q.scale > 0 ? 1.f / q.scale : 0.f);
    __m512i sum = _mm512_setzero_si512();
    for (size_t d = 0; d < vectorized; d += 16) {
      const __m512 r =
          _mm512_sub_ps(_mm512_loadu_ps(rotated_query + d),
                        _mm512_loadu_ps(projected_mean_.data() + d));
      __m512i k = _mm512_cvt_roundps_epi32(
          _mm512_mul_ps(r, inverse),
          _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
      k = _mm512_max_epi32(_mm512_set1_epi32(-127),
                           _mm512_min_epi32(_mm512_set1_epi32(127), k));
      _mm_storeu_si128(reinterpret_cast<__m128i *>(q.codes.data() + d),
                       _mm512_cvtsepi32_epi8(k));
      sum = _mm512_add_epi32(sum, k);
    }
    q.code_sum = _mm512_reduce_add_epi32(sum);
    for (size_t d = vectorized; d < projected; ++d) {
      const float v = (rotated_query[d] - projected_mean_[d]) *
                      (q.scale > 0 ? 1.f / q.scale : 0.f);
      q.codes[d] = int8_t(std::clamp(std::nearbyint(v), -127.f, 127.f));
      q.code_sum += q.codes[d];
    }
    return true;
  }
  uint32_t choose(const IntegerCenter::Query &q, size_t beam,
                  size_t *groups_scored = nullptr) const {
    if (empty() || q.codes.size() != format_.stride) return UINT32_MAX;
    beam = std::clamp(beam, size_t{1}, kMaxBeam);
    std::array<uint32_t, kMaxBeam> current{};
    size_t count = 1, visits = 0;
    std::pair<float, uint32_t> best{std::numeric_limits<float>::infinity(),
                                    UINT32_MAX};
    for (size_t level = 0; level < 16 && count; ++level) {
      std::array<std::pair<float, uint32_t>, kMaxBeam> next;
      next.fill({std::numeric_limits<float>::infinity(), UINT32_MAX});
      for (size_t i = 0; i < count; ++i) {
        const Group &g = groups_[current[i]];
        alignas(64) float values[16];
        score(g, q, values);
        ++visits;
        for (size_t j = 0; j < g.count; ++j) {
          if (g.target[j] < 0) {
            best = std::min(
                best,
                std::make_pair(values[j], uint32_t(-int64_t(g.target[j]) - 1)));
          } else {
            const auto candidate =
                std::make_pair(values[j], uint32_t(g.target[j]));
            if (candidate >= next[beam - 1]) continue;
            size_t pos = beam - 1;
            while (pos && candidate < next[pos - 1]) {
              next[pos] = next[pos - 1];
              --pos;
            }
            next[pos] = candidate;
          }
        }
      }
      count = 0;
      for (size_t j = 0; j < beam && next[j].second != UINT32_MAX; ++j)
        current[count++] = next[j].second;
    }
    if (groups_scored) *groups_scored = visits;
    return best.second;
  }

 private:
  std::vector<Group> groups_;
  std::vector<float> projected_mean_;
  QuantizedGraphRouterFormat format_;
};
}  // namespace zvec::core
