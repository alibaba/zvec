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

#include <cstddef>
#include <cstring>

namespace zvec::turbo::distance_internal {

// These helpers are compiled into TUs built with different /arch flags
// (scalar, AVX2, AVX512). They must be `static` so each TU keeps its own
// copy; a shared COMDAT copy could be the AVX512-encoded one, crashing
// AVX2/scalar callers on CPUs without AVX512.

struct RecordMeta {
  float scale;
  float bias;
  float sum;
  float squared_sum;
};

static inline RecordMeta load_record_meta(const void *record,
                                          size_t tail_offset) {
  RecordMeta meta;
  std::memcpy(&meta, static_cast<const char *>(record) + tail_offset,
              sizeof(meta));
  return meta;
}

static inline float record_minus_inner_product(const void *a, const void *b,
                                               size_t original_dim,
                                               size_t tail_offset,
                                               float raw_ip) {
  const RecordMeta m = load_record_meta(a, tail_offset);
  const RecordMeta q = load_record_meta(b, tail_offset);

  return -(m.scale * q.scale * raw_ip + m.bias * q.scale * q.sum +
           q.bias * m.scale * m.sum +
           static_cast<float>(original_dim) * q.bias * m.bias);
}

static inline float record_squared_euclidean(const void *a, const void *b,
                                             size_t original_dim,
                                             size_t tail_offset, float raw_ip) {
  const RecordMeta m = load_record_meta(a, tail_offset);
  const RecordMeta q = load_record_meta(b, tail_offset);
  const float query_sum = q.scale * q.sum;
  const float query_squared_sum = q.scale * q.scale * q.squared_sum;
  const float bias_diff = m.bias - q.bias;

  return m.scale * m.scale * m.squared_sum + query_squared_sum -
         2.0f * m.scale * q.scale * raw_ip +
         bias_diff * bias_diff * static_cast<float>(original_dim) +
         2.0f * bias_diff * (m.sum * m.scale - query_sum);
}

}  // namespace zvec::turbo::distance_internal
