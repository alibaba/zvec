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
  element_size_ = meta.element_size();
  dimension_ = meta.dimension();
  column_major_order_ = meta.major_order() == IndexMeta::MajorOrder::MO_COLUMN;
  // Turbo uses a query-bound DistanceImpl instead of a legacy metric.
  if (!metric) {
    return;
  }
  row_distance_ = metric->distance();
  distance_xx1 = metric->distance_matrix(block_vec_cnt, 1);
  if (column_major_order_ && meta.extra_meta_size() == 0 &&
      (meta.data_type() == IndexMeta::DT_FP32 ||
       meta.data_type() == IndexMeta::DT_FP16)) {
    const auto type = meta.data_type() == IndexMeta::DT_FP32
                          ? turbo::DataType::kFp32
                          : turbo::DataType::kFp16;
    const auto metric_type = meta.metric_name() == "InnerProduct"
                                 ? turbo::MetricType::kInnerProduct
                                 : turbo::MetricType::kSquaredEuclidean;
    using RawDistance = void (*)(const void *, const void *, size_t, float *);
    const auto raw =
        turbo::get_distance_func(metric_type, type, turbo::QuantizeType::kRaw);
    const auto expected = raw.target<RawDistance>();
    const auto actual = row_distance_.target<RawDistance>();
    // A caller-supplied query metric may have semantics beyond its meta name.
    const bool raw_metric = expected && actual && *expected == *actual;
    if (raw_metric && (meta.metric_name() == "InnerProduct" ||
                       meta.metric_name() == "SquaredEuclidean")) {
      column_many_distance_ = turbo::get_interleaved_many_to_many_distance_func(
          metric_type, type, block_vec_cnt);
    }
  }
  distances_.resize(33);
  for (size_t b = 32; b != 0; b /= 2) {
    distances_[b] = metric->distance_matrix(block_vec_cnt, b);
  }
}

IVFDistanceCalculator::~IVFDistanceCalculator() {
  row_distance_ = nullptr;
  distance_xx1 = nullptr;
  distances_.clear();
}

}  // namespace core
}  // namespace zvec
