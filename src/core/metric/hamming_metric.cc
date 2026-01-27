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

#include <ailego/math/hamming_distance_matrix.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include "ailego/math_batch/distance_batch.h"

namespace zvec {
namespace core {

//! Retrieve distance function for index features
static inline IndexMetric::MatrixDistanceHandle HammingDistanceMatrix32(
    size_t m, size_t n) {
  static const IndexMetric::MatrixDistanceHandle distance_table[6][6] = {
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint32_t, 1, 1>::Compute),
       nullptr, nullptr, nullptr, nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint32_t, 2, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint32_t, 2, 2>::Compute),
       nullptr, nullptr, nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint32_t, 4, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint32_t, 4, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint32_t, 4, 4>::Compute),
       nullptr, nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint32_t, 8, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint32_t, 8, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint32_t, 8, 4>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint32_t, 8, 8>::Compute),
       nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint32_t, 16, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint32_t, 16, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint32_t, 16, 4>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint32_t, 16, 8>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint32_t, 16, 16>::Compute),
       nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint32_t, 32, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint32_t, 32, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint32_t, 32, 4>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint32_t, 32, 8>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint32_t, 32, 16>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint32_t, 32, 32>::Compute)},
  };
  if (m > 32 || n > 32 || ailego_popcount(m) != 1 || ailego_popcount(n) != 1) {
    return nullptr;
  }
  return distance_table[ailego_ctz(m)][ailego_ctz(n)];
}

#if defined(AILEGO_M64)
static inline IndexMetric::MatrixDistanceHandle HammingDistanceMatrix64(
    size_t m, size_t n) {
  static const IndexMetric::MatrixDistanceHandle distance_table[6][6] = {
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint64_t, 1, 1>::Compute),
       nullptr, nullptr, nullptr, nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint64_t, 2, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint64_t, 2, 2>::Compute),
       nullptr, nullptr, nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint64_t, 4, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint64_t, 4, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint64_t, 4, 4>::Compute),
       nullptr, nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint64_t, 8, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint64_t, 8, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint64_t, 8, 4>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint64_t, 8, 8>::Compute),
       nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint64_t, 16, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint64_t, 16, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint64_t, 16, 4>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint64_t, 16, 8>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint64_t, 16, 16>::Compute),
       nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint64_t, 32, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint64_t, 32, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint64_t, 32, 4>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint64_t, 32, 8>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint64_t, 32, 16>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint64_t, 32, 32>::Compute)},
  };
  if (m > 32 || n > 32 || ailego_popcount(m) != 1 || ailego_popcount(n) != 1) {
    return nullptr;
  }
  return distance_table[ailego_ctz(m)][ailego_ctz(n)];
}
#endif  // AILEOG_M64

/*! Hamming Metric
 */
class HammingMetric : public IndexMetric {
 public:
  //! Initialize Metric
  int init(const IndexMeta &meta, const ailego::Params &index_params) override {
    if (meta.data_type() != IndexMeta::DataType::DT_BINARY32 &&
        meta.data_type() != IndexMeta::DataType::DT_BINARY64) {
      return IndexError_Unsupported;
    }
    if (IndexMeta::UnitSizeof(meta.data_type()) != meta.unit_size()) {
      return IndexError_Unsupported;
    }
    feature_type_ = meta.data_type();
    params_ = index_params;
    return 0;
  }

  //! Cleanup Metric
  int cleanup(void) override {
    return 0;
  }

  //! Retrieve if it matched
  bool is_matched(const IndexMeta &meta) const override {
    return (meta.data_type() == feature_type_ &&
            meta.unit_size() == IndexMeta::UnitSizeof(feature_type_));
  }

  //! Retrieve if it matched
  bool is_matched(const IndexMeta &meta,
                  const IndexQueryMeta &qmeta) const override {
    return (qmeta.data_type() == feature_type_ &&
            qmeta.unit_size() == IndexMeta::UnitSizeof(feature_type_) &&
            qmeta.dimension() == meta.dimension());
  }

  //! Retrieve distance function for query
  MatrixDistance distance(void) const override {
#if defined(AILEGO_M64)
    if (feature_type_ == IndexMeta::DataType::DT_BINARY64) {
      return reinterpret_cast<MatrixDistanceHandle>(
          ailego::HammingDistanceMatrix<uint64_t, 1, 1>::Compute);
    }
#endif
    if (feature_type_ == IndexMeta::DataType::DT_BINARY32) {
      return reinterpret_cast<MatrixDistanceHandle>(
          ailego::HammingDistanceMatrix<uint32_t, 1, 1>::Compute);
    }
    return nullptr;
  }

  MatrixBatchDistance batch_distance(void) const override {
#if defined(AILEGO_M64)
    if (feature_type_ == IndexMeta::DataType::DT_BINARY64) {
      return reinterpret_cast<IndexMetric::MatrixBatchDistanceHandle>(
          ailego::BaseDistance<ailego::HammingDistanceMatrix, uint64_t, 1,
                               1>::ComputeBatch);
    }
#endif
    if (feature_type_ == IndexMeta::DataType::DT_BINARY32) {
      return reinterpret_cast<IndexMetric::MatrixBatchDistanceHandle>(
          ailego::BaseDistance<ailego::HammingDistanceMatrix, uint32_t, 1,
                               1>::ComputeBatch);
    }
    return nullptr;
  }

  //! Retrieve distance function for index features
  MatrixDistance distance_matrix(size_t m, size_t n) const override {
#if defined(AILEGO_M64)
    if (feature_type_ == IndexMeta::DataType::DT_BINARY64) {
      return HammingDistanceMatrix64(m, n);
    }
#endif
    if (feature_type_ == IndexMeta::DataType::DT_BINARY32) {
      return HammingDistanceMatrix32(m, n);
    }
    return nullptr;
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
  IndexMeta::DataType feature_type_{IndexMeta::DataType::DT_BINARY32};
  ailego::Params params_{};
};

INDEX_FACTORY_REGISTER_METRIC_ALIAS(Hamming, HammingMetric);

}  // namespace core
}  // namespace zvec
