
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
#include "linear_seeker.h"

namespace zvec {
namespace core {

int LinearSeeker::seek(const void *query, size_t len, Document *out) {
  if (ailego_unlikely(!query || !out || meta_.element_size() != len)) {
    return IndexError_InvalidArgument;
  }

  float sel_score = std::numeric_limits<float>::max();
  uint32_t sel_column = 0;
  uint32_t total = static_cast<uint32_t>(features_->count());

  if (quantizer_) {
    constexpr uint32_t batch_size = 32;
    const void *vectors[batch_size];
    float scores[batch_size];
    for (uint32_t base = 0; base < total;) {
      uint32_t count = std::min(batch_size, total - base);
      for (uint32_t j = 0; j < count; ++j) {
        vectors[j] = features_->element(base + j);
      }
      quantizer_->calc_distance_dp_query_batch(vectors, count, query, scores);
      for (uint32_t j = 0; j < count; ++j) {
        if (scores[j] < sel_score) {
          sel_score = scores[j];
          sel_column = base + j;
        }
      }
      base += count;
    }
  } else {
    for (uint32_t i = 0; i < total; ++i) {
      float score = 0.0f;

      distance_func_(features_->element(i), query, meta_.dimension(), &score);
      if (score < sel_score) {
        sel_score = score;
        sel_column = i;
      }
    }
  }

  out->index = sel_column;
  out->score = sel_score;
  return 0;
}

}  // namespace core
}  // namespace zvec
