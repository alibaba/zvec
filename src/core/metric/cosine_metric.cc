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
#include <ailego/math/norm2_matrix.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include "turbo_metric.h"

namespace zvec {
namespace core {

class CosineMetric : public IndexMetric {
 public:
  //! Initialize Metric
  int init(const IndexMeta &meta, const ailego::Params &index_params) override {
    IndexMeta::DataType ft = meta.data_type();
    if (ft != IndexMeta::DataType::DT_FP16 &&
        ft != IndexMeta::DataType::DT_FP32) {
      return IndexError_Unsupported;
    }
    if (IndexMeta::UnitSizeof(ft) != meta.unit_size()) {
      return IndexError_Unsupported;
    }
    data_type_ = ft;
    params_ = index_params;

    return 0;
  }

  //! Cleanup Metric
  int cleanup() override {
    return 0;
  }

  //! Retrieve if it matched
  bool is_matched(const IndexMeta &meta) const override {
    return (meta.data_type() == data_type_ &&
            meta.unit_size() == IndexMeta::UnitSizeof(data_type_));
  }

  //! Retrieve if it matched
  bool is_matched(const IndexMeta &meta,
                  const IndexQueryMeta &qmeta) const override {
    return (qmeta.data_type() == data_type_ &&
            qmeta.unit_size() == IndexMeta::UnitSizeof(data_type_) &&
            qmeta.dimension() == meta.dimension());
  }

  //! Retrieve distance function for query
  MatrixDistance distance() const override {
    auto distance = RawKernels(turbo::MetricType::kCosine, data_type_).dist;
    const size_t extra = sizeof(float) / IndexMeta::UnitSizeof(data_type_);
    if (!distance) return nullptr;
    // CosineConverter appends a float norm to the normalized vector.
    return [=](const void *m, const void *q, size_t dim, float *out) {
      distance(m, q, dim - extra, out);
    };
  }

  //! Retrieve distance function for index features
  MatrixDistance distance_matrix(size_t m, size_t n) const override {
    if (m != 1 || n != 1) {
      return nullptr;
    }
    return distance();
  }

  //! Retrieve distance function for query
  MatrixBatchDistance batch_distance() const override {
    auto batch = RawKernels(turbo::MetricType::kCosine, data_type_).batch;
    const size_t extra = sizeof(float) / IndexMeta::UnitSizeof(data_type_);
    if (!batch) return nullptr;
    return [=](const void **m, const void *q, size_t count, size_t dim,
               float *out, const void **values) {
      batch(m, q, count, dim - extra, out, values);
    };
  }

  //! Retrieve params of Metric
  const ailego::Params &params() const override {
    return params_;
  }

  //! Retrieve query metric object of this index metric
  Pointer query_metric() const override {
    return nullptr;
  }

 private:
  IndexMeta::DataType data_type_{IndexMeta::DataType::DT_FP32};
  ailego::Params params_{};
};

INDEX_FACTORY_REGISTER_METRIC_ALIAS(Cosine, CosineMetric);

}  // namespace core
}  // namespace zvec
