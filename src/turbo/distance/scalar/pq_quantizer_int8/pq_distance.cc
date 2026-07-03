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

#include "scalar/pq_quantizer_int8/pq_distance.h"

namespace zvec::turbo::scalar {

void pq_adc_int8_distance(const uint8_t *pq_code, const float *lut,
                          size_t num_subquantizers, float *out) {
  constexpr size_t kNumCentroids = 256;
  float sum = 0.0f;
  for (size_t m = 0; m < num_subquantizers; ++m) {
    sum += lut[m * kNumCentroids + pq_code[m]];
  }
  *out = sum;
}

void pq_sdc_int8_distance(const uint8_t *a, const uint8_t *b,
                          const float *dist_table, size_t num_subquantizers,
                          float *out) {
  constexpr size_t kNumCentroids = 256;
  constexpr size_t kTablePerSub = kNumCentroids * kNumCentroids;  // 65536
  float sum = 0.0f;
  for (size_t m = 0; m < num_subquantizers; ++m) {
    size_t idx = m * kTablePerSub +
                 static_cast<size_t>(a[m]) * kNumCentroids +
                 static_cast<size_t>(b[m]);
    sum += dist_table[idx];
  }
  *out = sum;
}

}  // namespace zvec::turbo::scalar
