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
#include <ailego/math/euclidean_distance_matrix.h>
#include <ailego/math/inner_product_matrix.h>
#include <ailego/math/mips_euclidean_distance_matrix.h>
#include <ailego/math/norm2_matrix.h>
#include <ailego/math_batch/distance_batch.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include "metric_params.h"
#include "quantized_integer_metric_batch.h"
#include "quantized_integer_metric_matrix.h"

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
  int cleanup(void) override {
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
  MatrixDistance distance(void) const override {
    return distance_matrix(1, 1);
  }

  //! Retrieve matrix distance function for index features
  MatrixDistance distance_matrix(size_t m, size_t n) const override {
    switch (origin_metric_type_) {
      case MetricType::kSquaredEuclidean:
        if (meta_.data_type() == IndexMeta::DataType::DT_INT8) {
          return DistanceMatrixCompute<SquaredEuclidean, int8_t>(m, n);
        }
        if (meta_.data_type() == IndexMeta::DataType::DT_INT4) {
          return DistanceMatrixCompute<SquaredEuclidean, uint8_t>(m, n);
        }
        break;

      case MetricType::kInnerProduct:
        if (meta_.data_type() == IndexMeta::DataType::DT_INT8) {
          return DistanceMatrixCompute<MinusInnerProduct, int8_t>(m, n);
        }
        if (meta_.data_type() == IndexMeta::DataType::DT_INT4) {
          return DistanceMatrixCompute<MinusInnerProduct, uint8_t>(m, n);
        }
        break;

      case MetricType::kMipsSquaredEuclidean:
        if (meta_.data_type() == IndexMeta::DataType::DT_INT8) {
          return DistanceMatrixCompute<MipsSquaredEuclidean, int8_t>(m, n);
        }
        if (meta_.data_type() == IndexMeta::DataType::DT_INT4) {
          return DistanceMatrixCompute<MipsSquaredEuclidean, uint8_t>(m, n);
        }
        // TODO: support MipsSquaredEuclidean other injection type
        break;

      case MetricType::kNormalizedCosine:
        if (meta_.data_type() == IndexMeta::DataType::DT_INT8) {
          return DistanceMatrixCompute<MinusInnerProduct, int8_t>(m, n);
        }
        if (meta_.data_type() == IndexMeta::DataType::DT_INT4) {
          return DistanceMatrixCompute<MinusInnerProduct, uint8_t>(m, n);
        }
        break;
      case MetricType::kCosine:
        if (meta_.data_type() == IndexMeta::DataType::DT_INT8) {
          return DistanceMatrixCompute<CosineMinusInnerProduct, int8_t>(m, n);
        }
        if (meta_.data_type() == IndexMeta::DataType::DT_INT4) {
          return DistanceMatrixCompute<CosineMinusInnerProduct, uint8_t>(m, n);
        }
        break;
    }
    return nullptr;
  }

  //! Retrieve distance function for query
  MatrixBatchDistance batch_distance(void) const override {
    switch (origin_metric_type_) {
      case MetricType::kSquaredEuclidean:
        if (meta_.data_type() == IndexMeta::DataType::DT_INT8) {
          return reinterpret_cast<IndexMetric::MatrixBatchDistanceHandle>(
              BaseDistanceBatchWithScoreUnquantized<SquaredEuclidean, int8_t,
                                                    12, 2>::ComputeBatch);
        }
        if (meta_.data_type() == IndexMeta::DataType::DT_INT4) {
          return reinterpret_cast<IndexMetric::MatrixBatchDistanceHandle>(
              BaseDistanceBatchWithScoreUnquantized<SquaredEuclidean, uint8_t,
                                                    12, 2>::ComputeBatch);
        }
        break;

      case MetricType::kInnerProduct:
        if (meta_.data_type() == IndexMeta::DataType::DT_INT8) {
          return reinterpret_cast<IndexMetric::MatrixBatchDistanceHandle>(
              BaseDistanceBatchWithScoreUnquantized<MinusInnerProduct, int8_t,
                                                    12, 2>::ComputeBatch);
        }
        if (meta_.data_type() == IndexMeta::DataType::DT_INT4) {
          return reinterpret_cast<IndexMetric::MatrixBatchDistanceHandle>(
              BaseDistanceBatchWithScoreUnquantized<MinusInnerProduct, uint8_t,
                                                    12, 2>::ComputeBatch);
        }
        break;
      case MetricType::kMipsSquaredEuclidean:
        if (meta_.data_type() == IndexMeta::DataType::DT_INT8) {
          return reinterpret_cast<IndexMetric::MatrixBatchDistanceHandle>(
              BaseDistanceBatchWithScoreUnquantized<
                  MipsSquaredEuclidean, int8_t, 12, 2>::ComputeBatch);
        }
        if (meta_.data_type() == IndexMeta::DataType::DT_INT4) {
          return reinterpret_cast<IndexMetric::MatrixBatchDistanceHandle>(
              BaseDistanceBatchWithScoreUnquantized<
                  MipsSquaredEuclidean, uint8_t, 12, 2>::ComputeBatch);
        }
        // TODO: support MipsSquaredEuclidean other injection type
        break;
      case MetricType::kNormalizedCosine:
        if (meta_.data_type() == IndexMeta::DataType::DT_INT8) {
          return reinterpret_cast<IndexMetric::MatrixBatchDistanceHandle>(
              BaseDistanceBatchWithScoreUnquantized<MinusInnerProduct, int8_t,
                                                    12, 2>::ComputeBatch);
        }
        if (meta_.data_type() == IndexMeta::DataType::DT_INT4) {
          return reinterpret_cast<IndexMetric::MatrixBatchDistanceHandle>(
              BaseDistanceBatchWithScoreUnquantized<MinusInnerProduct, uint8_t,
                                                    12, 2>::ComputeBatch);
        }
        break;
      case MetricType::kCosine:
        if (meta_.data_type() == IndexMeta::DataType::DT_INT8) {
          return reinterpret_cast<IndexMetric::MatrixBatchDistanceHandle>(
              BaseDistanceBatchWithScoreUnquantized<
                  CosineMinusInnerProduct, int8_t, 12, 2>::ComputeBatch);
        }
        if (meta_.data_type() == IndexMeta::DataType::DT_INT4) {
          return reinterpret_cast<IndexMetric::MatrixBatchDistanceHandle>(
              BaseDistanceBatchWithScoreUnquantized<
                  CosineMinusInnerProduct, uint8_t, 12, 2>::ComputeBatch);
        }

        break;
    }
    return nullptr;
  }

  //! Retrieve params of Metric
  const ailego::Params &params(void) const override {
    return params_;
  }

  //! Train the metric
  int train(const void * /*vec*/, size_t /*dim*/) override {
    return 0;
  }

  //! Retrieve if it supports training
  bool support_train(void) const override {
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
  bool support_normalize(void) const override {
    return origin_metric_type_ == MetricType::kInnerProduct ||
           origin_metric_type_ == MetricType::kNormalizedCosine ||
           origin_metric_type_ == MetricType::kCosine;
  }

  //! Retrieve query metric object of this index metric
  Pointer query_metric(void) const override {
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

  virtual DistanceBatchQueryPreprocessFunc get_query_preprocess_func()
      const override {
    if (origin_metric_type_ == MetricType::kCosine &&
        meta_.data_type() == IndexMeta::DataType::DT_INT8) {
      return CosineMinusInnerProductDistanceBatchWithScoreUnquantized<
          int8_t, 1, 1>::GetQueryPreprocessFunc();
    }

    return nullptr;
  }


 private:
  //! Returns m x n distance matrix compute function.
  template <template <typename, size_t, size_t> class DistanceMatrix,
            typename T>
  static MatrixDistanceHandle DistanceMatrixCompute(size_t m, size_t n) {
    static void (*distance_table[6][6])(const T *, const T *, size_t,
                                        float *) = {
        {DistanceMatrix<T, 1, 1>::Compute, nullptr, nullptr, nullptr, nullptr,
         nullptr},
        {DistanceMatrix<T, 2, 1>::Compute, DistanceMatrix<T, 2, 2>::Compute,
         nullptr, nullptr, nullptr, nullptr},
        {DistanceMatrix<T, 4, 1>::Compute, DistanceMatrix<T, 4, 2>::Compute,
         DistanceMatrix<T, 4, 4>::Compute, nullptr, nullptr, nullptr},
        {DistanceMatrix<T, 8, 1>::Compute, DistanceMatrix<T, 8, 2>::Compute,
         DistanceMatrix<T, 8, 4>::Compute, DistanceMatrix<T, 8, 8>::Compute,
         nullptr, nullptr},
        {DistanceMatrix<T, 16, 1>::Compute, DistanceMatrix<T, 16, 2>::Compute,
         DistanceMatrix<T, 16, 4>::Compute, DistanceMatrix<T, 16, 8>::Compute,
         DistanceMatrix<T, 16, 16>::Compute, nullptr},
        {DistanceMatrix<T, 32, 1>::Compute, DistanceMatrix<T, 32, 2>::Compute,
         DistanceMatrix<T, 32, 4>::Compute, DistanceMatrix<T, 32, 8>::Compute,
         DistanceMatrix<T, 32, 16>::Compute,
         DistanceMatrix<T, 32, 32>::Compute}};
    if (m > 32 || n > 32 || ailego_popcount(m) != 1 ||
        ailego_popcount(n) != 1) {
      return nullptr;
    }
    return reinterpret_cast<MatrixDistanceHandle>(
        distance_table[ailego_ctz(m)][ailego_ctz(n)]);
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