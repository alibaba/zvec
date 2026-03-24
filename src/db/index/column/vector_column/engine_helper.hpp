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
#pragma once

#include <memory>
#include <zvec/ailego/pattern/expected.hpp>
#include <zvec/core/interface/index.h>
#include <zvec/core/interface/index_param_builders.h>
#include <zvec/db/doc.h>
#include <zvec/db/query_params.h>
#include <zvec/db/status.h>
#include "zvec/db/index_params.h"
#include "zvec/db/type.h"
#include "vector_column_params.h"


namespace zvec {
// TODO: rename file extension
class ProximaEngineHelper {
 public:
  static Result<vector_column_params::VectorDataBuffer>
  move_from_engine_vector_buffer(
      const core_interface::VectorDataBuffer &&vector_data_buffer,
      bool is_sparse) {
    if (is_sparse) {
      auto sparse_vector_buffer = std::get<core_interface::SparseVectorBuffer>(
          vector_data_buffer.vector_buffer);
      return vector_column_params::VectorDataBuffer{
          vector_column_params::SparseVectorBuffer{
              std::move(sparse_vector_buffer.indices),
              std::move(sparse_vector_buffer.values)}};
    }
    auto dense_vector_buffer = std::get<core_interface::DenseVectorBuffer>(
        vector_data_buffer.vector_buffer);
    return vector_column_params::VectorDataBuffer{
        vector_column_params::DenseVectorBuffer{
            std::move(dense_vector_buffer.data)}};
  }

  static Result<vector_column_params::VectorData> convert_from_engine_vector(
      const core_interface::VectorData &vector_data, bool is_sparse) {
    if (is_sparse) {
      auto engine_vector =
          std::get<core_interface::SparseVector>(vector_data.vector);
      return vector_column_params::VectorData{
          vector_column_params::SparseVector{engine_vector.count,
                                             engine_vector.indices,
                                             engine_vector.values}};
    }
    auto engine_vector =
        std::get<core_interface::DenseVector>(vector_data.vector);
    return vector_column_params::VectorData{
        vector_column_params::DenseVector{engine_vector.data}};
  }

  // convert to engine vector
  static Result<core_interface::VectorData> convert_to_engine_vector(
      const vector_column_params::VectorData &vector_data, bool is_sparse) {
    if (is_sparse) {
      auto db_vector =
          std::get<vector_column_params::SparseVector>(vector_data.vector);
      auto engine_vector = core_interface::SparseVector{
          db_vector.count, const_cast<void *>(db_vector.indices),
          const_cast<void *>(db_vector.values)};
      return core_interface::VectorData{engine_vector};
    }

    auto db_vector =
        std::get<vector_column_params::DenseVector>(vector_data.vector);
    auto engine_vector =
        core_interface::DenseVector{const_cast<void *>(db_vector.data)};
    return core_interface::VectorData{engine_vector};
  }

  // convert_filter
  static std::shared_ptr<core_interface::IndexFilter> convert_to_engine_filter(
      const IndexFilter *filter) {
    auto engine_filter = std::make_shared<core_interface::IndexFilter>();
    if (filter != nullptr) {
      engine_filter->set(
          [filter](uint64_t id) { return filter->is_filtered(id); });
    }
    return engine_filter;
  }

 private:
  template <typename EngineQueryParamType>
  static Result<std::unique_ptr<EngineQueryParamType>>
  _build_common_query_param(
      const vector_column_params::QueryParams &db_query_params) {
    auto engine_query_param = std::make_unique<EngineQueryParamType>();
    engine_query_param->topk = db_query_params.topk;
    engine_query_param->fetch_vector = db_query_params.fetch_vector;

    engine_query_param->filter =
        convert_to_engine_filter(db_query_params.filter);

    if (db_query_params.query_params) {
      engine_query_param->radius = db_query_params.query_params->radius();
      engine_query_param->is_linear = db_query_params.query_params->is_linear();
    }
    if (db_query_params.refiner_param) {
      engine_query_param->refiner_param =
          std::make_shared<core_interface::RefinerParam>(
              core_interface::RefinerParam{
                  .scale_factor_ = db_query_params.refiner_param->scale_factor_,
                  .reference_index =
                      db_query_params.refiner_param->reference_indexer->index});
    }

    return engine_query_param;
  }

 public:
  static Result<std::unique_ptr<core_interface::BaseIndexQueryParam>>
  convert_to_engine_query_param(
      const FieldSchema &field_schema,
      const vector_column_params::QueryParams &query_params) {
    if (!field_schema.index_params()) {
      return tl::make_unexpected(Status::InvalidArgument("nullptr"));
    }
    switch (field_schema.index_params()->type()) {
      case IndexType::FLAT: {
        // auto db_index_params =
        //     dynamic_cast<const FlatIndexParams
        //     *>(field_schema.index_params());
        auto flat_query_param_result =
            _build_common_query_param<core_interface::FlatQueryParam>(
                query_params);
        if (!flat_query_param_result.has_value()) {
          return tl::make_unexpected(Status::InvalidArgument(
              "failed to build query param: " +
              flat_query_param_result.error().message()));
        }
        return std::move(flat_query_param_result.value());
      }

      case IndexType::HNSW: {
        auto hnsw_query_param_result =
            _build_common_query_param<core_interface::HNSWQueryParam>(
                query_params);
        if (!hnsw_query_param_result.has_value()) {
          return tl::make_unexpected(Status::InvalidArgument(
              "failed to build query param: " +
              hnsw_query_param_result.error().message()));
        }
        auto &hnsw_query_param = hnsw_query_param_result.value();
        if (query_params.query_params) {
          auto db_hnsw_query_params = dynamic_cast<const HnswQueryParams *>(
              query_params.query_params.get());
          hnsw_query_param->ef_search = db_hnsw_query_params->ef();
        }
        return std::move(hnsw_query_param);
      }

      case IndexType::HNSW_RABITQ: {
        auto hnsw_query_param_result =
            _build_common_query_param<core_interface::HNSWRabitqQueryParam>(
                query_params);
        if (!hnsw_query_param_result.has_value()) {
          return tl::make_unexpected(Status::InvalidArgument(
              "failed to build query param: " +
              hnsw_query_param_result.error().message()));
        }
        auto &hnsw_query_param = hnsw_query_param_result.value();
        if (query_params.query_params) {
          auto db_hnsw_rabitq_query_params =
              dynamic_cast<const HnswRabitqQueryParams *>(
                  query_params.query_params.get());
          hnsw_query_param->ef_search = db_hnsw_rabitq_query_params->ef();
        }
        return std::move(hnsw_query_param);
      }

      case IndexType::IVF: {
        auto ivf_query_param_result =
            _build_common_query_param<core_interface::IVFQueryParam>(
                query_params);
        if (!ivf_query_param_result.has_value()) {
          return tl::make_unexpected(Status::InvalidArgument(
              "failed to build query param: " +
              ivf_query_param_result.error().message()));
        }
        auto &ivf_query_param = ivf_query_param_result.value();
        if (query_params.query_params) {
          auto db_ivf_query_params = dynamic_cast<const IVFQueryParams *>(
              query_params.query_params.get());
          ivf_query_param->nprobe = db_ivf_query_params->nprobe();
        }
        return std::move(ivf_query_param);
      }
      default:
        return tl::make_unexpected(Status::InvalidArgument("not supported"));
    }
  }

  static Result<core_interface::MetricType> convert_to_engine_metric_type(
      MetricType metric_type) {
    switch (metric_type) {
      case MetricType::MIPSL2:
        return core_interface::MetricType::kMIPSL2sq;
      case MetricType::IP:
        return core_interface::MetricType::kInnerProduct;
      case MetricType::L2:
        return core_interface::MetricType::kL2sq;
      case MetricType::COSINE:
        return core_interface::MetricType::kCosine;
      default:
        return tl::make_unexpected(
            Status::InvalidArgument("unsupported metric type"));
    }
  }

  static Result<core_interface::QuantizerType> convert_to_engine_quantize_type(
      QuantizeType quantize_type) {
    switch (quantize_type) {
      case QuantizeType::UNDEFINED:
        return core_interface::QuantizerType::kNone;
      case QuantizeType::FP16:
        return core_interface::QuantizerType::kFP16;
      case QuantizeType::INT8:
        return core_interface::QuantizerType::kInt8;
      case QuantizeType::INT4:
        return core_interface::QuantizerType::kInt4;
      case QuantizeType::RABITQ:
        return core_interface::QuantizerType::kRabitq;
      default:
        return tl::make_unexpected(
            Status::InvalidArgument("unsupported quantize type"));
    }
  }

  static Result<core_interface::DataType> convert_to_engine_data_type(
      DataType data_type) {
    switch (data_type) {
      case DataType::VECTOR_FP32:
      case DataType::SPARSE_VECTOR_FP32:
        return core_interface::DataType::DT_FP32;

      case DataType::VECTOR_FP16:
      case DataType::SPARSE_VECTOR_FP16:
        return core_interface::DataType::DT_FP16;

      case DataType::VECTOR_INT8:
        return core_interface::DataType::DT_INT8;

      default:
        return tl::make_unexpected(
            Status::InvalidArgument("unsupported data type"));
    }
  }

 private:
  template <typename DBIndexParamType, typename IndexParamBuilderType>
  static Result<std::shared_ptr<IndexParamBuilderType>>
  _build_common_index_param(const FieldSchema &field_schema) {
    auto db_index_params = dynamic_cast<const DBIndexParamType *>(
        field_schema.index_params().get());
    if (db_index_params == nullptr) {
      return tl::make_unexpected(Status::InvalidArgument("bad_cast"));
    }
    auto index_param_builder = std::make_shared<IndexParamBuilderType>();

    // db will ensure the id is consecutive
    index_param_builder->WithUseIDMap(false);

    index_param_builder->WithIsSparse(field_schema.is_sparse_vector())
        .WithDimension(field_schema.dimension());
    if (auto data_type_result =
            convert_to_engine_data_type(field_schema.data_type());
        data_type_result.has_value()) {
      index_param_builder->WithDataType(data_type_result.value());
    } else {
      return tl::make_unexpected(
          Status::InvalidArgument("unsupported data type"));
    }
    if (auto metric_type_result =
            convert_to_engine_metric_type(db_index_params->metric_type());
        metric_type_result.has_value()) {
      index_param_builder->WithMetricType(metric_type_result.value());
    } else {
      return tl::make_unexpected(
          Status::InvalidArgument("unsupported metric type"));
    }
    if (auto quantize_type =
            convert_to_engine_quantize_type(db_index_params->quantize_type());
        quantize_type.has_value()) {
      index_param_builder->WithQuantizerParam(
          core_interface::QuantizerParam(quantize_type.value()));
    } else {
      return tl::make_unexpected(
          Status::InvalidArgument("unsupported quantize type"));
    }
    return index_param_builder;
  }

 public:
  static Result<core_interface::BaseIndexParam::Pointer>
  convert_to_engine_index_param(const FieldSchema &field_schema) {
    if (!field_schema.index_params()) {
      return tl::make_unexpected(
          Status::InvalidArgument("field_schema.index_params nullptr"));
    }

    switch (field_schema.index_params()->type()) {
      case IndexType::FLAT: {
        auto index_param_builder =
            _build_common_index_param<FlatIndexParams,
                                      core_interface::FlatIndexParamBuilder>(
                field_schema);
        if (!index_param_builder.has_value()) {
          return tl::make_unexpected(
              Status::InvalidArgument("failed to build index param: " +
                                      index_param_builder.error().message()));
        }
        return index_param_builder.value()->Build();
      }

      case IndexType::HNSW: {
        auto index_param_builder_result =
            _build_common_index_param<HnswIndexParams,
                                      core_interface::HNSWIndexParamBuilder>(
                field_schema);
        if (!index_param_builder_result.has_value()) {
          return tl::make_unexpected(Status::InvalidArgument(
              "failed to build index param: " +
              index_param_builder_result.error().message()));
        }
        auto index_param_builder = index_param_builder_result.value();

        auto db_index_params = dynamic_cast<const HnswIndexParams *>(
            field_schema.index_params().get());
        index_param_builder->WithM(db_index_params->m());
        index_param_builder->WithEFConstruction(
            db_index_params->ef_construction());

        return index_param_builder->Build();
      }

      case IndexType::HNSW_RABITQ: {
        auto index_param_builder_result = _build_common_index_param<
            HnswRabitqIndexParams, core_interface::HNSWRabitqIndexParamBuilder>(
            field_schema);
        if (!index_param_builder_result.has_value()) {
          return tl::make_unexpected(Status::InvalidArgument(
              "failed to build index param: " +
              index_param_builder_result.error().message()));
        }
        auto index_param_builder = index_param_builder_result.value();

        auto db_index_params = dynamic_cast<const HnswRabitqIndexParams *>(
            field_schema.index_params().get());
        index_param_builder->WithM(db_index_params->m());
        index_param_builder->WithEFConstruction(
            db_index_params->ef_construction());
        index_param_builder->WithTotalBits(db_index_params->total_bits());
        index_param_builder->WithNumClusters(db_index_params->num_clusters());
        index_param_builder->WithSampleCount(db_index_params->sample_count());
        index_param_builder->WithProvider(
            db_index_params->raw_vector_provider());
        index_param_builder->WithReformer(db_index_params->rabitq_reformer());

        return index_param_builder->Build();
      }

      case IndexType::IVF: {
        auto index_param_builder_result = _build_common_index_param<
            IVFIndexParams, core_interface::IVFIndexParamBuilder>(field_schema);
        if (!index_param_builder_result.has_value()) {
          return tl::make_unexpected(Status::InvalidArgument(
              "failed to build index param: " +
              index_param_builder_result.error().message()));
        }
        auto index_param_builder = index_param_builder_result.value();

        auto db_index_params = dynamic_cast<const IVFIndexParams *>(
            field_schema.index_params().get());
        index_param_builder->WithNList(db_index_params->n_list());
        index_param_builder->WithNiters(db_index_params->n_iters());
        index_param_builder->WithUseSoar(db_index_params->use_soar());

        return index_param_builder->Build();
      }

      default:
        return tl::make_unexpected(Status::InvalidArgument("not supported"));
    }
  }
};
};  // namespace zvec