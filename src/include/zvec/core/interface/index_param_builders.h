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
#include <zvec/core/interface/index_param.h>

namespace zvec::core_interface {

// struct ConditionalIndexParam {
//     // predicate / rule / threshold
//     // candidate
// };


// chaining calls builder
template <typename ActualIndexParamBuilderType, typename ActualIndexParamType>
class BaseIndexParamBuilder {  //  : public
                               //  std::enable_shared_from_this<Resource>
 public:
  BaseIndexParamBuilder() : param(std::make_shared<ActualIndexParamType>()) {};
  virtual ~BaseIndexParamBuilder() = default;

  ActualIndexParamBuilderType &WithVersion(int version) {
    param.version = version;
    return static_cast<ActualIndexParamBuilderType &>(*this);
  }
  ActualIndexParamBuilderType &WithIndexType(IndexType index_type) {
    param->index_type = index_type;
    return static_cast<ActualIndexParamBuilderType &>(*this);
  }
  ActualIndexParamBuilderType &WithMetricType(MetricType metric_type) {
    param->metric_type = metric_type;
    return static_cast<ActualIndexParamBuilderType &>(*this);
  }
  ActualIndexParamBuilderType &WithDimension(int dimension) {
    param->dimension = dimension;
    return static_cast<ActualIndexParamBuilderType &>(*this);
  }
  ActualIndexParamBuilderType &WithPreprocessParam(
      const PreprocessorParam &preprocess_param) {
    param->preprocess_param =
        std::make_shared<PreprocessorParam>(preprocess_param);
    return static_cast<ActualIndexParamBuilderType &>(*this);
  }
  ActualIndexParamBuilderType &WithQuantizerParam(
      const QuantizerParam &quantizer_param) {
    param->quantizer_param = quantizer_param;
    return static_cast<ActualIndexParamBuilderType &>(*this);
  }
  // ActualIndexParamBuilderType &WithRefinerParam(
  //     const RefinerParam &refiner_param) {
  //   param->refiner_param = refiner_param;
  //   return static_cast<ActualIndexParamBuilderType &>(*this);
  // }
  // ActualIndexParamBuilderType &WithDefaultQueryParam(
  //     const BaseIndexQueryParam &default_query_param) {
  //   param->default_query_param = default_query_param;
  //   return static_cast<ActualIndexParamBuilderType &>(*this);
  // }

  ActualIndexParamBuilderType &WithIsSparse(bool is_sparse) {
    param->is_sparse = is_sparse;
    return static_cast<ActualIndexParamBuilderType &>(*this);
  }
  ActualIndexParamBuilderType &WithDataType(DataType data_type) {
    param->data_type = data_type;
    return static_cast<ActualIndexParamBuilderType &>(*this);
  }

  ActualIndexParamBuilderType &WithUseIDMap(bool use_id_map) {
    param->use_id_map = use_id_map;
    return static_cast<ActualIndexParamBuilderType &>(*this);
  }

  virtual std::shared_ptr<ActualIndexParamType> Build() = 0;

 protected:
  std::shared_ptr<ActualIndexParamType> param;
};

class FlatIndexParamBuilder
    : public BaseIndexParamBuilder<FlatIndexParamBuilder, FlatIndexParam> {
 public:
  FlatIndexParamBuilder() = default;
  std::shared_ptr<FlatIndexParam> Build() override {
    return param;
  }
};

class IVFIndexParamBuilder
    : public BaseIndexParamBuilder<IVFIndexParamBuilder, IVFIndexParam> {
 public:
  IVFIndexParamBuilder() = default;
  IVFIndexParamBuilder &WithNList(int nlist) {
    param->nlist = nlist;
    return *this;
  }
  IVFIndexParamBuilder &WithNiters(int niters) {
    param->niters = niters;
    return *this;
  }
  IVFIndexParamBuilder &WithL1Index(const BaseIndexParam &l1Index) {
    param->l1Index = std::make_shared<BaseIndexParam>(l1Index);
    return *this;
  }
  IVFIndexParamBuilder &WithL2Index(const BaseIndexParam &l2Index) {
    param->l2Index = std::make_shared<BaseIndexParam>(l2Index);
    return *this;
  }
  IVFIndexParamBuilder &WithUseSoar(bool use_soar) {
    param->use_soar = use_soar;
    return *this;
  }

  std::shared_ptr<IVFIndexParam> Build() override {
    return param;
  }
};

class HNSWIndexParamBuilder
    : public BaseIndexParamBuilder<HNSWIndexParamBuilder, HNSWIndexParam> {
 public:
  HNSWIndexParamBuilder() = default;
  HNSWIndexParamBuilder &WithM(int m) {
    param->m = m;
    return *this;
  }
  HNSWIndexParamBuilder &WithEFConstruction(int ef_construction) {
    param->ef_construction = ef_construction;
    return *this;
  }

  std::shared_ptr<HNSWIndexParam> Build() override {
    return param;
  }
};

//     class CompositeIndexParamBuilder : public
//     BaseIndexParamBuilder<CompositeIndexParamBuilder, CompositeIndexParam>
//     { public:
//         CompositeIndexParamBuilder() = default;
//         CompositeIndexParamBuilder &WithLayers(const
//         std::vector<std::shared_ptr<BaseIndexParam>> &layers) {
//             param.layers = layers;
//             return *this;
//         }
//         // with layer
//         CompositeIndexParamBuilder &WithLayer(const BaseIndexParam &layer)
//         {
//             param.layers.push_back(std::make_shared<BaseIndexParam>(layer));
//             return *this;
//         }

//         CompositeIndexParamBuilder &WithLayer(const BaseIndexParam &layer,
//                                               const BaseIndexQueryParam
//                                               &default_query_param) {
//             param.layers.push_back(std::make_shared<BaseIndexParam>(layer));
//             param.layers.back()->default_query_param =
//             std::make_shared<BaseIndexQueryParam>(default_query_param);
//             return *this;
//         }
//         std::shared_ptr<CompositeIndexParam> Build() { return
//         std::make_shared<CompositeIndexParam>(param); }

//     private:
//         CompositeIndexParam param;
//     };


#include <memory>
#include <vector>

template <typename T, typename Derived>
class BaseIndexQueryParamBuilder {
 public:
  // This allows derived builders to access the protected member
  T m_param;

  // Fluent setters for BaseIndexQueryParam fields
  Derived &with_topk(int topk) {
    m_param.topk = topk;
    return static_cast<Derived &>(*this);
  }

  Derived &with_fetch_vector(bool fetch_vector) {
    m_param.fetch_vector = fetch_vector;
    return static_cast<Derived &>(*this);
  }

  Derived &with_filter(std::shared_ptr<IndexFilter> filter) {
    m_param.filter = std::move(filter);
    return static_cast<Derived &>(*this);
  }

  // Using a vector of uint64_t for the next one
  Derived &with_bf_pks(std::shared_ptr<std::vector<uint64_t>> bf_pks) {
    m_param.bf_pks = std::move(bf_pks);
    return static_cast<Derived &>(*this);
  }

  Derived &with_radius(float radius) {
    m_param.radius = radius;
    return static_cast<Derived &>(*this);
  }

  Derived &with_is_linear(bool is_linear) {
    m_param.is_linear = is_linear;
    return static_cast<Derived &>(*this);
  }

  Derived &with_refiner_param(RefinerParam::Pointer refiner_param) {
    m_param.refiner_param = std::move(refiner_param);
    return static_cast<Derived &>(*this);
  }
};

// FLAT builder (no extra fields, just inherits base functionality)
class FlatQueryParamBuilder
    : public BaseIndexQueryParamBuilder<FlatQueryParam, FlatQueryParamBuilder> {
 public:
  FlatQueryParam::Pointer build() {
    return std::make_shared<FlatQueryParam>(std::move(m_param));
  }
};

// Example Usage:
// FlatQueryParam::Pointer flat_config = FlatQueryParamBuilder()
//     .with_topk(20)
//     .with_fetch_vector(true)
//     .build();

// HNSW builder (adds one specific field: ef_search)
class HNSWQueryParamBuilder
    : public BaseIndexQueryParamBuilder<HNSWQueryParam, HNSWQueryParamBuilder> {
 public:
  HNSWQueryParamBuilder &with_ef_search(int ef_search) {
    m_param.ef_search = ef_search;
    return *this;
  }

  HNSWQueryParam::Pointer build() {
    return std::make_shared<HNSWQueryParam>(std::move(m_param));
  }
};

// Example Usage:
// HNSWQueryParam::Pointer hnsw_config = HNSWQueryParamBuilder()
//     .with_topk(5)
//     .with_ef_search(128) // HNSW specific
//     .with_is_linear(false)
//     .build();

// IVF builder (adds specific fields: nprobe, l1QueryParam, l2QueryParam)
class IVFQueryParamBuilder
    : public BaseIndexQueryParamBuilder<IVFQueryParam, IVFQueryParamBuilder> {
 public:
  IVFQueryParamBuilder &with_nprobe(int nprobe) {
    m_param.nprobe = nprobe;
    return *this;
  }

  // Since l1QueryParam and l2QueryParam are shared_ptr to BaseIndexQueryParam,
  // they can accept ANY derived configuration object.
  IVFQueryParamBuilder &with_l1_query_param(
      BaseIndexQueryParam::Pointer l1QueryParam) {
    m_param.l1QueryParam = std::move(l1QueryParam);
    return *this;
  }

  IVFQueryParamBuilder &with_l2_query_param(
      BaseIndexQueryParam::Pointer l2QueryParam) {
    m_param.l2QueryParam = std::move(l2QueryParam);
    return *this;
  }

  IVFQueryParam::Pointer build() {
    return std::make_shared<IVFQueryParam>(std::move(m_param));
  }
};

// Example Usage:
// // First, build the required nested params
// auto nested_hnsw = HNSWQueryParamBuilder().with_ef_search(64).build();
//
// // Then, build the IVF param
// IVFQueryParam::Pointer ivf_config = IVFQueryParamBuilder()
//     .with_topk(10)
//     .with_nprobe(50) // IVF specific
//     .with_l1_query_param(nested_hnsw) // Set a nested config object
//     .build();


namespace predefined {
// some predefined index param builders, e.g., SCANN
class SCANNIndexParamBuilder {
 public:
  // alias SCANNIIndexParam = xxxxx
  std::shared_ptr<IVFIndexParam> Build() {
    // SCANN
    auto param_ptr =
        IVFIndexParamBuilder()
            .WithNList(40000)  //  10000000 -> 40000
            .WithUseSoar(
                true)  //  由于1个数据点可能对应2个partition，因此140个点中可能有重复，需要去重（保留一个取均值）
            .WithQuantizerParam(QuantizerParam(QuantizerType::kQuickADC))
            // .WithDefaultQueryParam(
            //     IVFQueryParamBuilder().with_topk(140).with_nprobe(68).build())
            // .WithRefinerParam(RefinerParam{
            //     10,  // 140 -> 10
            //     nullptr,
            //     std::make_shared<QuantizerParam>(
            //         QuantizerParam{QuantizerType::kFP16}),
            // })
            .WithL1Index(*(
                IVFIndexParamBuilder()
                    .WithMetricType(
                        MetricType::kInnerProduct)  // Layer2  flat index
                    .WithNList(700)                 //  40000 -> 700
                    .WithQuantizerParam(
                        QuantizerParam{QuantizerType::kQuickADC})
                    // .WithDefaultQueryParam(IVFQueryParamBuilder()
                    //                            .with_topk(68)
                    //                            .with_nprobe(20)
                    //                            .build())
                    .WithL1Index(*(
                        FlatIndexParamBuilder()
                            .WithMetricType(MetricType::kL2sq)
                            // implicit :
                            // .WithDefaultQueryParam(FlatQueryParamBuilder().with_topk(20).build())
                            .Build()))
                    .Build()))
            .Build();

    return param_ptr;
  }
};

}  // namespace predefined
}  // namespace zvec::core_interface