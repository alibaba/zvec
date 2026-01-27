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
#include <ailego/math/cosine_distance_matrix.h>
#include <ailego/math/norm2_matrix.h>
#include <ailego/math_batch/distance_batch.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>

namespace zvec {
namespace core {

//! Retrieve distance function for index features
inline IndexMetric::MatrixDistanceHandle CosineDistanceMatrixFp32(size_t m,
                                                                  size_t n) {
  static const IndexMetric::MatrixDistanceHandle distance_table[6][6] = {
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::CosineDistanceMatrix<float, 1, 1>::Compute),
       nullptr, nullptr, nullptr, nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::CosineDistanceMatrix<float, 2, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::CosineDistanceMatrix<float, 2, 2>::Compute),
       nullptr, nullptr, nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::CosineDistanceMatrix<float, 4, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::CosineDistanceMatrix<float, 4, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::CosineDistanceMatrix<float, 4, 4>::Compute),
       nullptr, nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::CosineDistanceMatrix<float, 8, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::CosineDistanceMatrix<float, 8, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::CosineDistanceMatrix<float, 8, 4>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::CosineDistanceMatrix<float, 8, 8>::Compute),
       nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::CosineDistanceMatrix<float, 16, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::CosineDistanceMatrix<float, 16, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::CosineDistanceMatrix<float, 16, 4>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::CosineDistanceMatrix<float, 16, 8>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::CosineDistanceMatrix<float, 16, 16>::Compute),
       nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::CosineDistanceMatrix<float, 32, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::CosineDistanceMatrix<float, 32, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::CosineDistanceMatrix<float, 32, 4>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::CosineDistanceMatrix<float, 32, 8>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::CosineDistanceMatrix<float, 32, 16>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::CosineDistanceMatrix<float, 32, 32>::Compute)},
  };
  if (m > 32 || n > 32 || ailego_popcount(m) != 1 || ailego_popcount(n) != 1) {
    return nullptr;
  }
  return distance_table[ailego_ctz(m)][ailego_ctz(n)];
}

//! Retrieve distance function for index features
inline IndexMetric::MatrixDistanceHandle CosineDistanceMatrixFp16(size_t m,
                                                                  size_t n) {
  static const IndexMetric::MatrixDistanceHandle distance_table[6][6] = {
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::CosineDistanceMatrix<ailego::Float16, 1, 1>::Compute),
       nullptr, nullptr, nullptr, nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::CosineDistanceMatrix<ailego::Float16, 2, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::CosineDistanceMatrix<ailego::Float16, 2, 2>::Compute),
       nullptr, nullptr, nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::CosineDistanceMatrix<ailego::Float16, 4, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::CosineDistanceMatrix<ailego::Float16, 4, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::CosineDistanceMatrix<ailego::Float16, 4, 4>::Compute),
       nullptr, nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::CosineDistanceMatrix<ailego::Float16, 8, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::CosineDistanceMatrix<ailego::Float16, 8, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::CosineDistanceMatrix<ailego::Float16, 8, 4>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::CosineDistanceMatrix<ailego::Float16, 8, 8>::Compute),
       nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::CosineDistanceMatrix<ailego::Float16, 16, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::CosineDistanceMatrix<ailego::Float16, 16, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::CosineDistanceMatrix<ailego::Float16, 16, 4>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::CosineDistanceMatrix<ailego::Float16, 16, 8>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::CosineDistanceMatrix<ailego::Float16, 16, 16>::Compute),
       nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::CosineDistanceMatrix<ailego::Float16, 32, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::CosineDistanceMatrix<ailego::Float16, 32, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::CosineDistanceMatrix<ailego::Float16, 32, 4>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::CosineDistanceMatrix<ailego::Float16, 32, 8>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::CosineDistanceMatrix<ailego::Float16, 32, 16>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::CosineDistanceMatrix<ailego::Float16, 32, 32>::Compute)},
  };
  if (m > 32 || n > 32 || ailego_popcount(m) != 1 || ailego_popcount(n) != 1) {
    return nullptr;
  }
  return distance_table[ailego_ctz(m)][ailego_ctz(n)];
}

/*! Cosine Metric
 */
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
  int cleanup(void) override {
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
  MatrixDistance distance(void) const override {
    switch (data_type_) {
      case IndexMeta::DataType::DT_FP16:
        return reinterpret_cast<MatrixDistanceHandle>(
            ailego::CosineDistanceMatrix<ailego::Float16, 1, 1>::Compute);
      case IndexMeta::DataType::DT_FP32:
        return reinterpret_cast<MatrixDistanceHandle>(
            ailego::CosineDistanceMatrix<float, 1, 1>::Compute);

      default:
        return nullptr;
    }
  }

  //! Retrieve distance function for index features
  MatrixDistance distance_matrix(size_t m, size_t n) const override {
    if (m != 1 || n != 1) {
      return nullptr;
    }
    return distance();
  }

  //! Retrieve distance function for query
  MatrixBatchDistance batch_distance(void) const override {
    switch (data_type_) {
      case IndexMeta::DataType::DT_FP32:
        return reinterpret_cast<IndexMetric::MatrixBatchDistanceHandle>(
            ailego::BaseDistance<ailego::CosineDistanceMatrix, float, 12,
                                 2>::ComputeBatch);
      case IndexMeta::DataType::DT_FP16:
        return reinterpret_cast<IndexMetric::MatrixBatchDistanceHandle>(
            ailego::BaseDistance<ailego::CosineDistanceMatrix, ailego::Float16,
                                 12, 2>::ComputeBatch);
      default:
        return nullptr;
    }
  }

  //! Retrieve params of Metric
  const ailego::Params &params(void) const override {
    return params_;
  }

  //! Retrieve query metric object of this index metric
  Pointer query_metric(void) const override {
    return nullptr;
  }

 private:
  IndexMeta::DataType data_type_{IndexMeta::DataType::DT_FP32};
  ailego::Params params_{};
};

INDEX_FACTORY_REGISTER_METRIC_ALIAS(Cosine, CosineMetric);

}  // namespace core
}  // namespace zvec
