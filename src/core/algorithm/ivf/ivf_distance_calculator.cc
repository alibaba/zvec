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
#include "ivf_distance_calculator.h"
#include <iostream>

namespace zvec {
namespace core {

IVFDistanceCalculator::IVFDistanceCalculator(const IndexMeta &meta,
                                             const IndexMetric::Pointer &metric,
                                             uint32_t block_vec_cnt)
    : metric_ptr_(metric), block_vec_cnt_(block_vec_cnt) {
  row_distance_ = metric->distance();
  distance_xx1 = metric->distance_matrix(block_vec_cnt, 1);
  distances_.resize(33);
  for (size_t b = 32; b != 0; b /= 2) {
    distances_[b] = metric->distance_matrix(block_vec_cnt, b);
  }
  element_size_ = meta.element_size();
  dimension_ = meta.dimension();
  if (meta.major_order() == IndexMeta::MajorOrder::MO_COLUMN) {
    column_major_order_ = true;
  } else {
    column_major_order_ = false;
  }
}

IVFDistanceCalculator::~IVFDistanceCalculator() {
  row_distance_ = nullptr;
  distance_xx1 = nullptr;
  distances_.clear();
}

}  // namespace core
}  // namespace zvec
