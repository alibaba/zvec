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
#include <ailego/math/sparse_distance.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_metric.h>
#include "turbo_metric.h"

namespace zvec {
namespace core {

class InnerProductMetric : public IndexMetric {
 public:
  //! Initialize Metric
  int init(const IndexMeta &meta, const ailego::Params &index_params) override {
    IndexMeta::MetaType mt = meta.meta_type();
    if (mt != IndexMeta::MetaType::MT_DENSE) {
      return IndexError_Unsupported;
    }

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

    meta_type_ = mt;
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
    return RawKernels(turbo::MetricType::kInnerProduct, data_type_).dist;
  }

  //! Retrieve sparse distance function for query
  MatrixSparseDistance sparse_distance() const override {
    return reinterpret_cast<MatrixSparseDistanceHandle>(
        ailego::MinusInnerProductSparseMatrix<float>::Compute);
  }

  //! Retrieve distance function for index features
  MatrixDistance distance_matrix(size_t m, size_t n) const override {
    return turbo::MakeDistanceMatrix(distance(), m, n,
                                     TurboDataType(data_type_));
  }

  //! Retrieve distance function for query
  MatrixBatchDistance batch_distance() const override {
    return RawKernels(turbo::MetricType::kInnerProduct, data_type_).batch;
  }

  //! Normalize result
  void normalize(float *score) const override {
    *score = -(*score);
  }

  //! Denormalize threshold
  void denormalize(float *score) const override {
    *score = -(*score);
  }

  //! Retrieve if it supports normalization
  bool support_normalize() const override {
    return true;
  }

  //! Retrieve params of Metric
  const ailego::Params &params() const override {
    return params_;
  }

  //! Retrieve query measure object of this index measure
  Pointer query_metric() const override {
    return nullptr;
  }

 private:
  IndexMeta::MetaType meta_type_{IndexMeta::MetaType::MT_DENSE};
  IndexMeta::DataType data_type_{IndexMeta::DataType::DT_FP32};
  ailego::Params params_{};
};

/*! Normalized Cosine Metric
 */
class NormalizedCosineMetric : public InnerProductMetric {
 public:
  //! Initialize Metric
  int init(const IndexMeta &meta, const ailego::Params &index_params) override {
    IndexMeta::DataType dt = meta.data_type();
    if (dt != IndexMeta::DataType::DT_FP16 &&
        dt != IndexMeta::DataType::DT_FP32) {
      return IndexError_Unsupported;
    }

    InnerProductMetric::init(meta, index_params);

    return 0;
  }

  //! Normalize result
  void normalize(float *score) const override {
    *score = 1 + (*score);
  }

  //! Denormalize threshold

  void denormalize(float *score) const override {
    *score -= 1;
  }
};

/*! Inner Product Sparse Metric
 */
class InnerProductSparseMetric : public IndexMetric {
 public:
  //! Initialize Metric
  int init(const IndexMeta &meta, const ailego::Params &index_params) override {
    IndexMeta::DataType dt = meta.data_type();
    if (dt != IndexMeta::DataType::DT_FP16 &&
        dt != IndexMeta::DataType::DT_FP32) {
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
            qmeta.data_type() == meta.data_type() &&
            qmeta.unit_size() == IndexMeta::UnitSizeof(data_type_) &&
            qmeta.unit_size() == meta.unit_size());
  }

  //! Retrieve distance function for query
  MatrixDistance distance() const override {
    return nullptr;
  }

  //! Retrieve sparse distance function for query
  MatrixSparseDistance sparse_distance() const override {
    switch (data_type_) {
      case IndexMeta::DataType::DT_FP16:
        return reinterpret_cast<MatrixSparseDistanceHandle>(
            ailego::MinusInnerProductSparseMatrix<ailego::Float16>::Compute);
      case IndexMeta::DataType::DT_FP32:
        return reinterpret_cast<MatrixSparseDistanceHandle>(
            ailego::MinusInnerProductSparseMatrix<float>::Compute);
      default:
        return nullptr;
    }
  }

  //! Normalize result
  void normalize(float *score) const override {
    *score = -(*score);
  }

  //! Denormalize threshold
  void denormalize(float *score) const override {
    *score = -(*score);
  }

  //! Retrieve if it supports normalization
  bool support_normalize() const override {
    return true;
  }

  //! Retrieve params of Metric
  const ailego::Params &params() const override {
    return params_;
  }

  //! Retrieve query measure object of this index measure
  Pointer query_metric() const override {
    return nullptr;
  }

 private:
  IndexMeta::DataType data_type_{IndexMeta::DataType::DT_FP32};
  ailego::Params params_{};
};

INDEX_FACTORY_REGISTER_METRIC_ALIAS(InnerProduct, InnerProductMetric);
INDEX_FACTORY_REGISTER_METRIC_ALIAS(NormalizedCosine, NormalizedCosineMetric);

INDEX_FACTORY_REGISTER_METRIC_ALIAS(InnerProductSparse,
                                    InnerProductSparseMetric);

}  // namespace core
}  // namespace zvec
