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
#include <zvec/turbo/turbo.h>
#include "metric_params.h"
#include "turbo_metric.h"

namespace zvec {
namespace core {

/*! Index Metric for quantized integer by IntegerStreamingConverter
 */
class QuantizedIntegerMetric : public IndexMetric {
 public:
  //! Initialize Metric
  int init(const IndexMeta &meta, const ailego::Params &index_params) override {
    if (meta.data_type() != IndexMeta::DataType::DT_INT8 &&
        meta.data_type() != IndexMeta::DataType::DT_INT4) {
      LOG_ERROR("Unsupported type %d", meta.data_type());
      return IndexError_Unsupported;
    }
    std::string metric_name;
    ailego::Params metric_params;
    index_params.get(QUANTIZED_INTEGER_METRIC_ORIGIN_METRIC_NAME, &metric_name);
    index_params.get(QUANTIZED_INTEGER_METRIC_ORIGIN_METRIC_PARAMS,
                     &metric_params);
    if (metric_name.empty()) {
      LOG_ERROR("Param %s is required",
                QUANTIZED_INTEGER_METRIC_ORIGIN_METRIC_NAME.c_str());
      return IndexError_InvalidArgument;
    }
    if (metric_name == "SquaredEuclidean") {
      origin_metric_type_ = MetricType::kSquaredEuclidean;
    } else if (metric_name == "InnerProduct") {
      origin_metric_type_ = MetricType::kInnerProduct;
    } else if (metric_name == "MipsSquaredEuclidean") {
      origin_metric_type_ = MetricType::kMipsSquaredEuclidean;
    } else if (metric_name == "NormalizedCosine") {
      origin_metric_type_ = MetricType::kNormalizedCosine;
    } else if (metric_name == "Cosine") {
      origin_metric_type_ = MetricType::kCosine;
    } else {
      LOG_ERROR("Unsupported metric %s", metric_name.c_str());
      return IndexError_Unsupported;
    }
    meta_ = meta;
    params_ = index_params;

    return 0;
  }

  //! Cleanup Metric
  int cleanup() override {
    return 0;
  }

  //! Retrieve if it matched
  bool is_matched(const IndexMeta &meta) const override {
    return meta.data_type() == meta_.data_type() &&
           meta.unit_size() == meta_.unit_size();
  }

  //! Retrieve if it matched
  bool is_matched(const IndexMeta &meta,
                  const IndexQueryMeta &qmeta) const override {
    return qmeta.data_type() == meta_.data_type() &&
           qmeta.unit_size() == meta_.unit_size() &&
           qmeta.dimension() == meta.dimension();
  }

  //! Retrieve distance function for query
  MatrixDistance distance() const override {
    auto kernels = record_kernels();
    if (origin_metric_type_ != MetricType::kMipsSquaredEuclidean)
      return kernels.dist;
    const auto ip = kernels.dist;
    if (!ip) return nullptr;
    return [=](const void *m, const void *q, size_t dim, float *out) {
      float u2, v2;
      ip(m, m, dim, &u2);
      ip(q, q, dim, &v2);
      ip(m, q, dim, out);
      *out = 2.0f + 2.0f * *out / std::max(-u2, -v2);
    };
  }

  //! Retrieve matrix distance function for index features
  MatrixDistance distance_matrix(size_t m, size_t n) const override {
    return turbo::MakeDistanceMatrix(distance(), m, n,
                                     TurboDataType(meta_.data_type()));
  }

  //! Retrieve distance function for query
  MatrixBatchDistance batch_distance() const override {
    if (origin_metric_type_ == MetricType::kMipsSquaredEuclidean) {
      return BatchFromDistance(distance());
    }
    return record_kernels().batch;
  }

  //! Retrieve params of Metric
  const ailego::Params &params() const override {
    return params_;
  }

  //! Train the metric
  int train(const void * /*vec*/, size_t /*dim*/) override {
    return 0;
  }

  //! Retrieve if it supports training
  bool support_train() const override {
    // No global norm scaling => eta_ == 0 => no training.
    return false;
  }

  //! Normalize result
  void normalize(float *score) const override {
    if (origin_metric_type_ == MetricType::kInnerProduct) {
      *score = -(*score);
    } else if (origin_metric_type_ == MetricType::kNormalizedCosine) {
      *score = 1.0f + *score;
    } else if (origin_metric_type_ == MetricType::kCosine) {
      *score = 1.0f + *score;
    }
  }

  //! Retrieve if it supports normalization
  bool support_normalize() const override {
    return origin_metric_type_ == MetricType::kInnerProduct ||
           origin_metric_type_ == MetricType::kNormalizedCosine ||
           origin_metric_type_ == MetricType::kCosine;
  }

  //! Build-time distance offset to make the internal distance non-negative.
  //! For kCosine / kNormalizedCosine on the quantized int8 path, the internal
  //! distance is -cos(m,q) in [-1, 1]. Adding 1.0 maps it to [0, 2] (i.e.
  //! 1 - cos), which is what ratio-based pruning (Vamana RobustPrune) needs
  //! for a geometrically meaningful occlude_factor.
  float build_distance_offset() const override {
    if (origin_metric_type_ == MetricType::kCosine ||
        origin_metric_type_ == MetricType::kNormalizedCosine) {
      return 1.0f;
    }
    return 0.0f;
  }

  //! Retrieve query metric object of this index metric
  Pointer query_metric() const override {
    if (origin_metric_type_ == MetricType::kMipsSquaredEuclidean) {
      auto metric = IndexFactory::CreateMetric("QuantizedInteger");
      if (metric) {
        ailego::Params metric_params;
        metric_params.set(QUANTIZED_INTEGER_METRIC_ORIGIN_METRIC_NAME,
                          "InnerProduct");
        metric->init(meta_, metric_params);
      }
      return metric;
    }
    return nullptr;
  }

  DistanceBatchQueryPreprocessFunc get_query_preprocess_func() const override {
    return record_kernels().preprocess;
  }


 private:
  turbo::DistanceKernels record_kernels() const {
    auto metric = turbo::MetricType::kInnerProduct;
    if (origin_metric_type_ == MetricType::kSquaredEuclidean) {
      metric = turbo::MetricType::kSquaredEuclidean;
    } else if (origin_metric_type_ == MetricType::kCosine) {
      metric = turbo::MetricType::kCosine;
    }
    return turbo::get_distance_kernels(metric, TurboDataType(meta_.data_type()),
                                       turbo::QuantizeType::kRecord);
  }

  enum struct MetricType {
    kSquaredEuclidean = 0,
    kInnerProduct = 1,
    kMipsSquaredEuclidean = 2,
    kNormalizedCosine = 3,
    kCosine = 4
  };

  //! Members
  IndexMeta meta_{};
  ailego::Params params_{};
  MetricType origin_metric_type_{};
};

INDEX_FACTORY_REGISTER_METRIC_ALIAS(QuantizedInteger, QuantizedIntegerMetric);

}  // namespace core
}  // namespace zvec
