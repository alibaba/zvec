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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <zvec/ailego/encoding/json.h>
#include <zvec/ailego/parallel/thread_pool.h>
#include <zvec/core/framework/index_filter.h>
#include <zvec/core/framework/index_meta.h>
#include <zvec/core/interface/constants.h>
#include "zvec/core/framework/index_framework.h"

namespace zvec::core_interface {
#define MAX_DIMENSION 65536
// #define MAX_EF_CONSTRUCTION 65536
// #define MAX_EF_SEARCH 100

class IndexFactory;
class Index;
class BaseIndexParam;
class BaseIndexQueryParam;

struct StorageOptions {
  enum class StorageType { kNone, kMMAP, kMemory, kBufferPool };

  StorageType type = StorageType::kNone;
  bool create_new = false;
  bool read_only = false;
};

struct MergeOptions {
  uint32_t write_concurrency = 1;
  ailego::ThreadPool *pool = nullptr;
};

using IndexMeta = core::IndexMeta;
using IndexQueryMeta = core::IndexQueryMeta;
using DataType = core::IndexMeta::DataType;
using IndexFilter = core::IndexFilter;


// 定义支持的索引类型
enum class IndexType {
  // to do: support factory's register, may change to
  // `static constexpr std::string_view`, which may incur str comp overhead
  kNone,
  kFlat,
  kIVF,  // it's actual a two-layer index
  kHNSW,
  kHNSWRabitq,
};

enum class IVFSearchMethod { kBF, kHNSW };

enum class MetricType {
  kNone,
  kL2sq,  // Euclidean
  kInnerProduct,
  kCosine,
  kMIPSL2sq  // spherical?
};

enum class QuantizerType {
  kNone,
  kPQ,        // Product Quantization
  kQuickADC,  // TODO: +refiner ? // should be a type of index?
  kAQ,
  kFP16,
  kInt8,
  kInt4,
  kRabitq,
};

struct SerializableBase {
  std::string SerializeToJson(bool omit_empty_value = false) const {
    return zvec::ailego::JsonValue(SerializeToJsonObject(omit_empty_value))
        .as_json_string()
        .as_stl_string();
  }

  bool DeserializeFromJson(const std::string &json_str) {
    ailego::JsonValue json_value;
    if (!json_value.parse(json_str)) {
      return false;
    }
    return DeserializeFromJsonObject(json_value.as_object());
  }

 protected:
  virtual ailego::JsonObject SerializeToJsonObject(
      bool omit_empty_value = false) const = 0;
  virtual bool DeserializeFromJsonObject(
      const ailego::JsonObject &json_obj) = 0;
};

// TODO: maybe a base class for quantizer?
struct QuantizerParam : public SerializableBase {
  QuantizerType type = QuantizerType::kNone;
  int num_subquantizers = 8;  // M
  int num_bits = 8;           // bits per subquantizer

  // Constructors
  // QuantizerParam() = default;
  QuantizerParam(QuantizerType t = QuantizerType::kNone, int subquantizers = 8,
                 int bits = 8)
      : type(t), num_subquantizers(subquantizers), num_bits(bits) {}


 protected:
  friend class BaseIndexParam;
  virtual ailego::JsonObject SerializeToJsonObject(
      bool omit_empty_value = false) const override;

  virtual bool DeserializeFromJsonObject(
      const ailego::JsonObject &json_obj) override;
};

// preprocessor
enum class PreprocessorType {
  kNone,
  kPCA,
  kOPQ,
};

struct PreprocessorParam {
  PreprocessorType type = PreprocessorType::kNone;

  // Constructors
  // PreprocessorParam() = default;
  explicit PreprocessorParam(PreprocessorType t = PreprocessorType::kNone)
      : type(t) {}
};

struct RefinerParam {
  using Pointer = std::shared_ptr<RefinerParam>;

  float scale_factor_{0};
  std::shared_ptr<Index> reference_index = nullptr;
};

// --- Query Parameters (can be passed to search methods) ---
class BaseIndexQueryParam {
 public:
  using Pointer = std::shared_ptr<BaseIndexQueryParam>;

  virtual ~BaseIndexQueryParam() = default;

  uint32_t topk = 10;
  bool fetch_vector = false;
  std::shared_ptr<IndexFilter> filter = nullptr;
  std::shared_ptr<std::vector<uint64_t>> bf_pks = nullptr;
  float radius = 0.0f;
  bool is_linear = false;
  RefinerParam::Pointer refiner_param = nullptr;

  virtual Pointer Clone() const = 0;
};

struct FlatQueryParam : public BaseIndexQueryParam {
  using Pointer = std::shared_ptr<FlatQueryParam>;

  BaseIndexQueryParam::Pointer Clone() const override {
    return std::make_shared<FlatQueryParam>(*this);
  }
};

struct HNSWQueryParam : public BaseIndexQueryParam {
  using Pointer = std::shared_ptr<HNSWQueryParam>;

  uint32_t ef_search = kDefaultHnswEfSearch;

  BaseIndexQueryParam::Pointer Clone() const override {
    return std::make_shared<HNSWQueryParam>(*this);
  }
};

struct HNSWRabitqQueryParam : public BaseIndexQueryParam {
  using Pointer = std::shared_ptr<HNSWRabitqQueryParam>;

  uint32_t ef_search = kDefaultHnswEfSearch;

  BaseIndexQueryParam::Pointer Clone() const override {
    return std::make_shared<HNSWRabitqQueryParam>(*this);
  }
};

struct IVFQueryParam : public BaseIndexQueryParam {
  int nprobe = 10;
  std::shared_ptr<BaseIndexQueryParam> l1QueryParam = nullptr;
  std::shared_ptr<BaseIndexQueryParam> l2QueryParam = nullptr;

  using Pointer = std::shared_ptr<IVFQueryParam>;

  BaseIndexQueryParam::Pointer Clone() const override {
    auto cloned_this = std::make_shared<IVFQueryParam>(*this);
    cloned_this->l1QueryParam = l1QueryParam ? l1QueryParam->Clone() : nullptr;
    cloned_this->l2QueryParam = l2QueryParam ? l2QueryParam->Clone() : nullptr;
    return cloned_this;
  }
};

// --- Construction Parameters ---
// template<typename IndexQueryParamType>
class BaseIndexParam : public SerializableBase {
 public:
  using Pointer = std::shared_ptr<BaseIndexParam>;

  explicit BaseIndexParam(IndexType type = IndexType::kNone,
                          MetricType metric = MetricType::kL2sq, int dim = 0,
                          int ver = 0)
      : index_type(type), metric_type(metric), dimension(dim), version(ver) {}

  virtual ~BaseIndexParam() = default;

  IndexType index_type = IndexType::kNone;
  MetricType metric_type = MetricType::kL2sq;
  int dimension = 0;  // [1, MAX_DIMENSION]
  int version = 0;    // for compatibility
  bool is_sparse = false;
  bool is_huge_page = false;
  DataType data_type = DataType::DT_UNDEFINED;
  bool use_id_map = true;

  // IndexMeta meta;
  ailego::Params params;

  // pipeline
  PreprocessorParam preprocess_param;
  QuantizerParam quantizer_param;

  BaseIndexQueryParam::Pointer default_query_param = nullptr;
  // virtual std::shared_ptr<BaseIndexQueryParam> GetDefaultQueryParam() const
  // {
  //   return std::make_shared<BaseIndexQueryParam>();
  // }
  //

 protected:
  virtual bool DeserializeFromJsonObject(
      const ailego::JsonObject &json_obj) override;
  virtual ailego::JsonObject SerializeToJsonObject(
      bool omit_empty_value = false) const override;
};

struct FlatIndexParam : public BaseIndexParam {
  using Pointer = std::shared_ptr<FlatIndexParam>;
  FlatIndexParam() : BaseIndexParam(IndexType::kFlat) {}

  IndexMeta::MajorOrder major_order = IndexMeta::MajorOrder::MO_ROW;

 protected:
  bool DeserializeFromJsonObject(const ailego::JsonObject &json_obj) override;
  ailego::JsonObject SerializeToJsonObject(
      bool omit_empty_value = false) const override;
};

struct IVFIndexParam : public BaseIndexParam {
  using Pointer = std::shared_ptr<IVFIndexParam>;
  int nlist = 1024;
  int niters = 10;
  std::shared_ptr<BaseIndexParam> l1Index = nullptr;
  std::shared_ptr<BaseIndexParam> l2Index = nullptr;
  bool use_soar = false;

  // Constructors with delegation
  IVFIndexParam() : BaseIndexParam(IndexType::kIVF) {}

  IVFIndexParam(int nlist, int niters, std::shared_ptr<BaseIndexParam> l1Index,
                std::shared_ptr<BaseIndexParam> l2Index)
      : BaseIndexParam(IndexType::kIVF),
        nlist(nlist),
        niters(niters),
        l1Index(std::move(l1Index)),
        l2Index(std::move(l2Index)) {}

  IVFIndexParam(MetricType metric, int dim, int nlist, int niters,
                std::shared_ptr<BaseIndexParam> l1Index,
                std::shared_ptr<BaseIndexParam> l2Index)
      : BaseIndexParam(IndexType::kIVF, metric, dim),
        nlist(nlist),
        niters(niters),
        l1Index(std::move(l1Index)),
        l2Index(std::move(l2Index)) {}

  // query param:
  // topk of l1Index's param ==== IVFIndexQueryParam.nprobe
  // topk of l2Index's param ==== IVFIndexQueryParam.topK

  // IVFIndexParam.metric_type === l2Index's metric_type
  // IVFIndexParam.quantization === l2Index's quantization
};

struct HNSWIndexParam : public BaseIndexParam {
  using Pointer = std::shared_ptr<HNSWIndexParam>;
  int m = kDefaultHnswNeighborCnt;
  int ef_construction = kDefaultHnswEfConstruction;

  // Constructors with delegation
  HNSWIndexParam() : BaseIndexParam(IndexType::kHNSW) {}

  HNSWIndexParam(int m, int ef_construction)
      : BaseIndexParam(IndexType::kHNSW),
        m(m),
        ef_construction(ef_construction) {}

  HNSWIndexParam(MetricType metric, int dim, int m, int ef_construction)
      : BaseIndexParam(IndexType::kHNSW, metric, dim),
        m(m),
        ef_construction(ef_construction) {}

 protected:
  bool DeserializeFromJsonObject(const ailego::JsonObject &json_obj) override;
  ailego::JsonObject SerializeToJsonObject(
      bool omit_empty_value = false) const override;
};

struct HNSWRabitqIndexParam : public BaseIndexParam {
  using Pointer = std::shared_ptr<HNSWRabitqIndexParam>;

  // HNSW parameters
  int m = kDefaultHnswNeighborCnt;
  int ef_construction = kDefaultHnswEfConstruction;

  // Rabitq parameters
  int total_bits = kDefaultRabitqTotalBits;
  int num_clusters = kDefaultRabitqNumClusters;
  int sample_count = 0;
  core::IndexProvider::Pointer provider = nullptr;
  core::IndexReformer::Pointer reformer = nullptr;

  // Constructors with delegation
  HNSWRabitqIndexParam() : BaseIndexParam(IndexType::kHNSWRabitq) {}

  HNSWRabitqIndexParam(int m, int ef_construction)
      : BaseIndexParam(IndexType::kHNSWRabitq),
        m(m),
        ef_construction(ef_construction) {}

  HNSWRabitqIndexParam(MetricType metric, int dim, int m, int ef_construction)
      : BaseIndexParam(IndexType::kHNSWRabitq, metric, dim),
        m(m),
        ef_construction(ef_construction) {}

 protected:
  bool DeserializeFromJsonObject(const ailego::JsonObject &json_obj) override;
  ailego::JsonObject SerializeToJsonObject(
      bool omit_empty_value = false) const override;
};

}  // namespace zvec::core_interface