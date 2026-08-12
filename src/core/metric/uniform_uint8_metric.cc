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
#include <cstring>
#include <limits>
#include <memory>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/interface/index_param.h>
#include <zvec/turbo/turbo.h>
#include "metric_params.h"

namespace zvec {
namespace core {

namespace {

constexpr size_t kTailBytes = sizeof(uint32_t);

size_t OriginalDimension(size_t encoded_dimension) {
  return encoded_dimension > kTailBytes ? encoded_dimension - kTailBytes : 0;
}

void UniformUint8QueryPreprocess(void *query, size_t encoded_dimension) {
  const size_t original_dimension = OriginalDimension(encoded_dimension);
  if (original_dimension == 0) {
    return;
  }

  auto *raw_query = static_cast<uint8_t *>(query);
  // Match the existing record-quantizer contract: transform() emits the
  // canonical shifted layout, and graph contexts preprocess their private
  // query copy exactly once before using the query-oriented distance kernels.
  uint64_t sum = 0;
  uint64_t sum_squared = 0;
  for (size_t i = 0; i < original_dimension; ++i) {
    raw_query[i] ^= uint8_t{0x80};
    const uint64_t value = raw_query[i];
    sum += value;
    sum_squared += value * value;
  }

  const int64_t correction =
      static_cast<int64_t>(sum_squared) - 256 * static_cast<int64_t>(sum);
  if (correction < (std::numeric_limits<int32_t>::min)() ||
      correction > (std::numeric_limits<int32_t>::max)()) {
    // Oversized externally supplied metadata uses the scalar squared-
    // difference path, which ignores the query tail.
    return;
  }
  const int32_t encoded_correction = static_cast<int32_t>(correction);
  std::memcpy(static_cast<uint8_t *>(query) + original_dimension,
              &encoded_correction, sizeof(encoded_correction));
}

IndexMetric::DistanceBatchQueryPreprocessFunc
UniformUint8QueryPreprocessFunc() {
  static const auto preprocess = []() {
    auto turbo_preprocess = turbo::get_query_preprocess_func(
        turbo::MetricType::kSquaredEuclidean, turbo::DataType::kInt8,
        turbo::QuantizeType::kUniformUint8);
    return turbo_preprocess ? turbo_preprocess : UniformUint8QueryPreprocess;
  }();
  return preprocess;
}

void UniformUint8StoredSquaredEuclidean(const void *lhs_data,
                                        const void *rhs_data,
                                        size_t encoded_dimension,
                                        float *distance) {
  const size_t original_dimension = OriginalDimension(encoded_dimension);
  const auto *lhs = static_cast<const int8_t *>(lhs_data);
  const auto *rhs = static_cast<const int8_t *>(rhs_data);
  int64_t sum = 0;
  for (size_t i = 0; i < original_dimension; ++i) {
    const int difference = static_cast<int>(lhs[i]) - static_cast<int>(rhs[i]);
    sum += static_cast<int64_t>(difference) * difference;
  }
  *distance = static_cast<float>(sum);
}

IndexMetric::MatrixDistance UniformUint8StoredDistance() {
  static const IndexMetric::MatrixDistance distance = []() {
    auto turbo_distance = turbo::get_distance_func(
        turbo::MetricType::kSquaredEuclidean, turbo::DataType::kInt8,
        turbo::QuantizeType::kUniformUint8);
    return turbo_distance ? turbo_distance : UniformUint8StoredSquaredEuclidean;
  }();
  return distance;
}

void UniformUint8StoredQuerySquaredEuclidean(const void *stored_data,
                                             const void *query_data,
                                             size_t encoded_dimension,
                                             float *distance) {
  const size_t original_dimension = OriginalDimension(encoded_dimension);
  const auto *stored = static_cast<const int8_t *>(stored_data);
  const auto *query = static_cast<const uint8_t *>(query_data);
  int64_t sum = 0;
  for (size_t i = 0; i < original_dimension; ++i) {
    const int difference =
        static_cast<int>(stored[i]) - (static_cast<int>(query[i]) - 128);
    sum += static_cast<int64_t>(difference) * difference;
  }
  *distance = static_cast<float>(sum);
}

void UniformUint8StoredQuerySquaredEuclideanBatch(const void *const *vectors,
                                                  const void *query,
                                                  size_t count,
                                                  size_t encoded_dimension,
                                                  float *distances) {
  for (size_t i = 0; i < count; ++i) {
    UniformUint8StoredQuerySquaredEuclidean(vectors[i], query,
                                            encoded_dimension, distances + i);
  }
}

}  // namespace

class UniformUint8QueryMetric : public IndexMetric {
 public:
  UniformUint8QueryMetric() = default;
  UniformUint8QueryMetric(const IndexMeta &meta, const ailego::Params &params)
      : meta_(meta), params_(params) {}

  int init(const IndexMeta &meta, const ailego::Params &params) override {
    if (meta.data_type() != IndexMeta::DataType::DT_INT8) {
      LOG_ERROR("UniformUint8Metric: unsupported type %d", meta.data_type());
      return IndexError_Unsupported;
    }
    if (meta.dimension() <= kTailBytes) {
      LOG_ERROR(
          "UniformUint8Metric: encoded dimension=%u must include a non-empty "
          "vector and a %zu-byte tail",
          meta.dimension(), kTailBytes);
      return IndexError_InvalidArgument;
    }

    std::string metric_name;
    params.get(UNIFORM_UINT8_METRIC_ORIGIN_METRIC_NAME, &metric_name);
    if (metric_name.empty()) {
      LOG_ERROR("UniformUint8Metric: param %s is required",
                UNIFORM_UINT8_METRIC_ORIGIN_METRIC_NAME.c_str());
      return IndexError_InvalidArgument;
    }
    if (metric_name != "SquaredEuclidean") {
      LOG_ERROR("UniformUint8Metric: only SquaredEuclidean supported, got %s",
                metric_name.c_str());
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
           meta.unit_size() == meta_.unit_size() &&
           meta.dimension() == meta_.dimension();
  }

  bool is_matched(const IndexMeta &meta,
                  const IndexQueryMeta &query_meta) const override {
    return is_matched(meta) && query_meta.data_type() == meta_.data_type() &&
           query_meta.unit_size() == meta_.unit_size() &&
           query_meta.dimension() == meta_.dimension();
  }

  MatrixDistance distance(void) const override {
    return UniformUint8StoredQuerySquaredEuclidean;
  }

  // FlatSearcher scans canonical query/record encodings through
  // distance_matrix(), just like the existing record quantizer.
  // Query-oriented distance() and batch_distance() are reserved for the
  // once-preprocessed graph path.
  MatrixDistance distance_matrix(size_t rows, size_t columns) const override {
    return rows == 1 && columns == 1 ? UniformUint8StoredDistance() : nullptr;
  }

  MatrixBatchDistance batch_distance(void) const override {
    const size_t original_dimension = OriginalDimension(meta_.dimension());
    // The VNNI kernel reduces its signed dot product in int32 lanes. The
    // public quantizer dimension bound guarantees that reduction is exact;
    // preserve the scalar int64 path for larger externally supplied metadata.
    if (original_dimension > 0 && original_dimension <= MAX_DIMENSION) {
      auto turbo_distance = turbo::get_batch_distance_func(
          turbo::MetricType::kSquaredEuclidean, turbo::DataType::kInt8,
          turbo::QuantizeType::kUniformUint8);
      if (turbo_distance) {
        return turbo_distance;
      }
    }
    return UniformUint8StoredQuerySquaredEuclideanBatch;
  }

  DistanceBatchQueryPreprocessFunc get_query_preprocess_func() const override {
    return UniformUint8QueryPreprocessFunc();
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

 protected:
  IndexMeta meta_{};
  ailego::Params params_{};
};

class UniformUint8Metric : public UniformUint8QueryMetric {
 public:
  MatrixDistance distance(void) const override {
    return UniformUint8StoredDistance();
  }

  MatrixDistance distance_matrix(size_t rows, size_t columns) const override {
    return rows == 1 && columns == 1 ? UniformUint8StoredDistance() : nullptr;
  }

  // Deliberately inherit the query-oriented batch_distance(). Graph builders
  // preprocess their private build-query copy before batch comparisons, while
  // pairwise pruning and Flat use the stored-stored functions above.

  Pointer query_metric(void) const override {
    return std::make_shared<UniformUint8QueryMetric>(meta_, params_);
  }
};

INDEX_FACTORY_REGISTER_METRIC_ALIAS(UniformUint8, UniformUint8Metric);

}  // namespace core
}  // namespace zvec
