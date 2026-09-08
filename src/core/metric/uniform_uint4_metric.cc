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

#include <cstdint>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/turbo/turbo.h>
#include "metric_params.h"

namespace zvec {
namespace core {
namespace {

void UniformUint4SquaredEuclidean(const void *lhs, const void *rhs,
                                  size_t encoded_dimension, float *distance) {
  const auto *a = static_cast<const uint8_t *>(lhs);
  const auto *b = static_cast<const uint8_t *>(rhs);
  int64_t sum = 0;
  for (size_t i = 0; i < encoded_dimension; ++i) {
    const int low_delta =
        static_cast<int>(a[i] & 0x0fU) - static_cast<int>(b[i] & 0x0fU);
    const int high_delta = static_cast<int>((a[i] >> 4U) & 0x0fU) -
                           static_cast<int>((b[i] >> 4U) & 0x0fU);
    sum += low_delta * low_delta + high_delta * high_delta;
  }
  *distance = static_cast<float>(sum);
}

void UniformUint4SquaredEuclideanBatch(const void *const *vectors,
                                       const void *query, size_t count,
                                       size_t encoded_dimension,
                                       float *distances,
                                       const void *const * /*extra_values*/) {
  for (size_t i = 0; i < count; ++i) {
    UniformUint4SquaredEuclidean(vectors[i], query, encoded_dimension,
                                 distances + i);
  }
}

}  // namespace

class UniformUint4Metric : public IndexMetric {
 public:
  int init(const IndexMeta &meta, const ailego::Params &params) override {
    if (meta.data_type() != IndexMeta::DataType::DT_INT8 ||
        meta.dimension() == 0 || (meta.dimension() % 64U) != 0) {
      LOG_ERROR(
          "UniformUint4Metric: expected a non-empty packed DT_INT8 dimension "
          "aligned to 64 bytes, got type=%d dimension=%u",
          meta.data_type(), meta.dimension());
      return IndexError_Unsupported;
    }
    std::string origin_metric;
    params.get(UNIFORM_UINT4_METRIC_ORIGIN_METRIC_NAME, &origin_metric);
    if (origin_metric != "SquaredEuclidean") {
      LOG_ERROR("UniformUint4Metric: only SquaredEuclidean is supported");
      return IndexError_Unsupported;
    }
    meta_ = meta;
    params_ = params;
    return 0;
  }

  int cleanup(void) override {
    return 0;
  }
  bool is_matched(const IndexMeta &meta) const override {
    return meta.data_type() == meta_.data_type() &&
           meta.dimension() == meta_.dimension() &&
           meta.unit_size() == meta_.unit_size();
  }
  bool is_matched(const IndexMeta &meta,
                  const IndexQueryMeta &qmeta) const override {
    return is_matched(meta) && qmeta.data_type() == meta_.data_type() &&
           qmeta.dimension() == meta_.dimension() &&
           qmeta.unit_size() == meta_.unit_size();
  }

  MatrixDistance distance(void) const override {
    auto turbo_distance = turbo::get_distance_func(
        turbo::MetricType::kSquaredEuclidean, turbo::DataType::kInt4,
        turbo::QuantizeType::kUniformUint4);
    return turbo_distance ? turbo_distance : UniformUint4SquaredEuclidean;
  }
  MatrixDistance distance_matrix(size_t m, size_t n) const override {
    return m == 1 && n == 1 ? distance() : MatrixDistance{};
  }
  MatrixBatchDistance batch_distance(void) const override {
    auto turbo_distance = turbo::get_batch_distance_func(
        turbo::MetricType::kSquaredEuclidean, turbo::DataType::kInt4,
        turbo::QuantizeType::kUniformUint4);
    return turbo_distance ? turbo_distance : UniformUint4SquaredEuclideanBatch;
  }
  DistanceBatchQueryPreprocessFunc get_query_preprocess_func() const override {
    return nullptr;
  }
  const ailego::Params &params(void) const override {
    return params_;
  }
  int train(const void * /*vector*/, size_t /*dimension*/) override {
    return 0;
  }
  bool support_train(void) const override {
    return false;
  }
  void normalize(float * /*score*/) const override {}
  bool support_normalize(void) const override {
    return false;
  }
  Pointer query_metric(void) const override {
    return nullptr;
  }

 private:
  IndexMeta meta_{};
  ailego::Params params_{};
};

INDEX_FACTORY_REGISTER_METRIC_ALIAS(UniformUint4, UniformUint4Metric);

}  // namespace core
}  // namespace zvec
