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
    uint8_t codes[kWidth][kDim]{};
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
  static bool read_mean(const char *path, size_t docs,
                        std::vector<float> &mean) {
    FILE *f = std::fopen(path, "rb");
    if (!f) return false;
    uint32_t header[5]{};
    std::vector<float> parsed(768);
    bool ok = std::fread(header, sizeof(header), 1, f) == 1 &&
              header[0] == 0x51475259 && header[1] == kVersion &&
              header[2] == kDim && header[3] > 0 && header[3] <= 100000 &&
              header[4] == docs &&
              std::fread(parsed.data(), sizeof(float), 768, f) == 768;
    ok = std::fclose(f) == 0 && ok;
    for (float value : parsed) ok = ok && std::isfinite(value);
    if (ok) mean = std::move(parsed);
    return ok;
  }
  template <class Graph>
  bool load(const char *path, const std::vector<float> &mean,
            const Graph &entity, size_t docs,
            const std::vector<float> *rotated_mean = nullptr) {
    clear();
    if (!path) return false;
    if (mean.size() != 768 || !__builtin_cpu_supports("avx512vnni"))
      return false;
    FILE *f = std::fopen(path, "rb");
    if (!f) return false;
    auto read = [f](void *p, size_t bytes) {
      return std::fread(p, 1, bytes, f) == bytes;
    };
    uint32_t header[5]{};
    std::array<float, 768> stored_mean{};
    bool ok =
        read(header, sizeof(header)) && header[0] == 0x51475259 &&
        header[1] == kVersion && header[2] == kDim && header[3] > 0 &&
        header[3] <= 100000 && header[4] == docs &&
        read(stored_mean.data(), sizeof(stored_mean)) &&
        std::memcmp(stored_mean.data(), mean.data(), sizeof(stored_mean)) == 0;
    uint32_t tag = 0;
    std::array<float, 768> projected_mean{};
    ok = ok && rotated_mean && rotated_mean->size() == 768 &&
         read(&tag, sizeof(tag)) && tag == kTransform &&
         read(projected_mean.data(), sizeof(projected_mean)) &&
         std::memcmp(projected_mean.data(), rotated_mean->data(),
                     sizeof(projected_mean)) == 0;
    if (ok) projected_mean_ = projected_mean;
    std::vector<Group> parsed(ok ? header[3] : 0);
    std::vector<unsigned> parents(parsed.size(), 0), depth(parsed.size(), 0);
    for (size_t i = 0; ok && i < parsed.size(); ++i) {
      Group &g = parsed[i];
      ok = read(&g.count, sizeof(g.count)) && g.count > 0 &&
           g.count <= kWidth && read(g.target, sizeof(g.target)) &&
           read(g.scale, sizeof(g.scale)) && read(g.offset, sizeof(g.offset)) &&
           read(g.mean_dot, sizeof(g.mean_dot)) &&
           read(g.codes, sizeof(g.codes));
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
    ok = ok && std::fgetc(f) == EOF && !std::ferror(f);
    ok = std::fclose(f) == 0 && ok;
    if (ok) groups_ = std::move(parsed);
    return ok;
  }
  __attribute__((target("avx512f,avx512bw,avx512vnni,fma"))) static void score(
      const Group &g, const IntegerCenter::Query &q, float *scores,
      int32_t *products = nullptr) {
    __m512i acc[16];
#pragma GCC unroll 16
    for (size_t i = 0; i < 16; ++i) acc[i] = _mm512_setzero_si512();
    for (size_t d = 0; d < kDim; d += 64) {
      const __m512i query = _mm512_loadu_si512(q.codes.data() + d);
#pragma GCC unroll 16
      for (size_t i = 0; i < 16; ++i)
        acc[i] = _mm512_dpbusd_epi32(acc[i], _mm512_loadu_si512(g.codes[i] + d),
                                     query);
    }
    alignas(64) int32_t dots[16];
#pragma GCC unroll 16
    for (size_t i = 0; i < 16; ++i) dots[i] = _mm512_reduce_add_epi32(acc[i]);
    if (products) std::memcpy(products, dots, sizeof(dots));
    const __m512 residual = _mm512_fmadd_ps(
        _mm512_loadu_ps(g.scale), _mm512_cvtepi32_ps(_mm512_load_si512(dots)),
        _mm512_mul_ps(_mm512_loadu_ps(g.offset),
                      _mm512_set1_ps(float(q.code_sum))));
    const __m512 value = _mm512_fmadd_ps(_mm512_set1_ps(q.scale * kFactor),
                                         residual, _mm512_loadu_ps(g.mean_dot));
    _mm512_storeu_ps(scores, _mm512_sub_ps(_mm512_setzero_ps(), value));
    for (size_t i = g.count; i < 16; ++i)
      scores[i] = std::numeric_limits<float>::infinity();
  }
  // Heuristic routing codes only: no terminal-refinement error bound is
  // implied.
  __attribute__((target("avx512f,avx512bw"))) bool prepare_projected(
      const float *rotated_query, IntegerCenter::Query &q) const {
    if (!rotated_query || empty()) return false;
    q.codes.resize(kDim);
    __m512 maximum = _mm512_setzero_ps();
    for (size_t d = 0; d < kDim; d += 16) {
      const __m512 r =
          _mm512_sub_ps(_mm512_loadu_ps(rotated_query + d),
                        _mm512_loadu_ps(projected_mean_.data() + d));
      maximum =
          _mm512_max_ps(maximum, _mm512_castsi512_ps(_mm512_andnot_si512(
                                     _mm512_set1_epi32(int32_t(0x80000000u)),
                                     _mm512_castps_si512(r))));
    }
    q.scale = _mm512_reduce_max_ps(maximum) / 127.f;
    if (!std::isfinite(q.scale)) return false;
    const __m512 inverse = _mm512_set1_ps(q.scale > 0 ? 1.f / q.scale : 0.f);
    __m512i sum = _mm512_setzero_si512();
    for (size_t d = 0; d < kDim; d += 16) {
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
    return true;
  }
  uint32_t choose(const IntegerCenter::Query &q, size_t beam,
                  size_t *groups_scored = nullptr) const {
    if (empty() || q.codes.size() != kDim) return UINT32_MAX;
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
  std::array<float, 768> projected_mean_{};
};
}  // namespace zvec::core
