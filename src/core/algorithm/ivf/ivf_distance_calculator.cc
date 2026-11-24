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
#include "ivf_params.h"

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
  column_single_distance_ = metric->distance_matrix(block_vec_cnt, 1);
  if (column_major_order_ && meta.extra_meta_size() == 0 &&
      (meta.data_type() == IndexMeta::DT_FP32 ||
       meta.data_type() == IndexMeta::DT_FP16) &&
      (meta.metric_name() == kIPMetricName ||
       meta.metric_name() == kL2MetricName) &&
      metric->is_matched(meta)) {
    // Ask the metric for its own kernel: identical functions can have
    // different addresses in independently linked libraries. Custom metrics
    // retain their original path unless they explicitly provide a kernel.
    if (auto *provider =
            dynamic_cast<const InterleavedDistanceProvider *>(metric.get())) {
      column_many_distance_ =
          provider->interleaved_many_distance(block_vec_cnt);
    }
  }
}

IVFDistanceCalculator::~IVFDistanceCalculator() = default;

}  // namespace core
}  // namespace zvec
