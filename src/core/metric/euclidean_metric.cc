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
#include <ailego/math/sparse_distance.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_metric.h>
#include <zvec/turbo/turbo.h>
#include "turbo_metric.h"

namespace zvec {
namespace core {

class SquaredEuclideanMetric : public IndexMetric {
 public:
  //! Initialize Metric
  int init(const IndexMeta &meta, const ailego::Params &index_params) override {
    IndexMeta::DataType dt = meta.data_type();
    if (dt != IndexMeta::DataType::DT_FP16 &&
        dt != IndexMeta::DataType::DT_FP32 &&
        dt != IndexMeta::DataType::DT_INT8 &&
        dt != IndexMeta::DataType::DT_INT4 &&
        dt != IndexMeta::DataType::DT_UINT8) {
      return IndexError_Unsupported;
    }
    if (IndexMeta::UnitSizeof(dt) != meta.unit_size()) {
      return IndexError_Unsupported;
    }
    data_type_ = dt;
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
    return RawKernels(turbo::MetricType::kSquaredEuclidean, data_type_).dist;
  }

  //! Retrieve sparse distance function for query
  MatrixSparseDistance sparse_distance() const override {
    return reinterpret_cast<MatrixSparseDistanceHandle>(
        ailego::SquaredEuclideanSparseDistanceMatrix<float>::Compute);
  }

  //! Retrieve distance function for index features
  MatrixDistance distance_matrix(size_t m, size_t n) const override {
    if (data_type_ == IndexMeta::DT_UINT8 && (m != 1 || n != 1)) return nullptr;
    return turbo::MakeDistanceMatrix(distance(), m, n,
                                     TurboDataType(data_type_));
  }

  //! Retrieve distance function for query
  MatrixBatchDistance batch_distance() const override {
    return RawKernels(turbo::MetricType::kSquaredEuclidean, data_type_).batch;
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

/*! Euclidean Distance Metric
 */
class EuclideanMetric : public IndexMetric {
 public:
  //! Initialize Metric
  int init(const IndexMeta &meta, const ailego::Params &index_params) override {
    IndexMeta::DataType dt = meta.data_type();
    if (dt != IndexMeta::DataType::DT_FP16 &&
        dt != IndexMeta::DataType::DT_FP32 &&
        dt != IndexMeta::DataType::DT_INT8 &&
        dt != IndexMeta::DataType::DT_INT4) {
      return IndexError_Unsupported;
    }
    if (IndexMeta::UnitSizeof(dt) != meta.unit_size()) {
      return IndexError_Unsupported;
    }
    data_type_ = dt;
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
    auto distance =
        RawKernels(turbo::MetricType::kSquaredEuclidean, data_type_).dist;
    if (!distance) return nullptr;
    return [=](const void *m, const void *q, size_t dim, float *out) {
      distance(m, q, dim, out);
      *out = std::sqrt(*out);
    };
  }

  //! Retrieve distance function for index features
  MatrixDistance distance_matrix(size_t m, size_t n) const override {
    return turbo::MakeDistanceMatrix(distance(), m, n,
                                     TurboDataType(data_type_));
  }

  //! Retrieve distance function for query
  MatrixBatchDistance batch_distance() const override {
    auto batch =
        RawKernels(turbo::MetricType::kSquaredEuclidean, data_type_).batch;
    if (!batch) return nullptr;
    return [=](const void **m, const void *q, size_t count, size_t dim,
               float *out, const void **extra) {
      batch(m, q, count, dim, out, extra);
      for (size_t i = 0; i < count; ++i) out[i] = std::sqrt(out[i]);
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

/*! Squared Euclidean Sparse Metric
 */
class SquaredEuclideanSparseMetric : public IndexMetric {
 public:
  //! Initialize Metric
  int init(const IndexMeta &meta, const ailego::Params &index_params) override {
    IndexMeta::DataType data_type = meta.data_type();
    if (data_type != IndexMeta::DataType::DT_FP16 &&
        data_type != IndexMeta::DataType::DT_FP32) {
      return IndexError_Unsupported;
    }

    if (IndexMeta::UnitSizeof(data_type) != meta.unit_size()) {
      return IndexError_Unsupported;
    }

    data_type_ = data_type;
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
            qmeta.data_type() == meta.data_type() &&
            qmeta.unit_size() == IndexMeta::UnitSizeof(data_type_) &&
            qmeta.unit_size() == meta.unit_size());
  }

  //! Retrieve sparse distance function for query
  MatrixSparseDistance sparse_distance() const override {
    return reinterpret_cast<MatrixSparseDistanceHandle>(
        ailego::SquaredEuclideanSparseDistanceMatrix<float>::Compute);
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

INDEX_FACTORY_REGISTER_METRIC_ALIAS(SquaredEuclidean, SquaredEuclideanMetric);
INDEX_FACTORY_REGISTER_METRIC_ALIAS(Euclidean, EuclideanMetric);

INDEX_FACTORY_REGISTER_METRIC_ALIAS(SquaredEuclideanSparse,
                                    SquaredEuclideanSparseMetric);

}  // namespace core
}  // namespace zvec
