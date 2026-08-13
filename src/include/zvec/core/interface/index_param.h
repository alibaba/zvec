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
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <zvec/ailego/encoding/json.h>
#include <zvec/ailego/parallel/thread_pool.h>
#include <zvec/core/framework/index_filter.h>
#include <zvec/core/framework/index_meta.h>
#include <zvec/core/interface/constants.h>
#include <zvec/export.h>
#include "zvec/core/framework/index_framework.h"

namespace zvec::core_interface {
#define MAX_DIMENSION 65536
// #define MAX_EF_CONSTRUCTION 65536
// #define MAX_EF_SEARCH 100

class ZVEC_CORE_API IndexFactory;
class ZVEC_CORE_API Index;
class ZVEC_CORE_API BaseIndexParam;
class ZVEC_CORE_API BaseIndexQueryParam;

struct StorageOptions {
  enum class StorageType { kNone, kMMAP, kMemory, kBufferPool };

  StorageType type = StorageType::kNone;
  bool create_new = false;
  bool read_only = false;

  // Only meaningful when type == kMMAP.
  // false: MAP_SHARED. Writes through mmap auto-persist to the file.
  // true : MAP_PRIVATE on a writable file. Flush/close forces dirty pages
  //        back to disk via explicit pwrite.
  bool copy_on_write = false;
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
  kDiskAnn,
  kVamana,
  kIVFRabitq,
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
  // Global uniform quantization with a shared scale/bias and 128 code values
  // in [0, 127]. It can be slightly faster than kUniformUint8, but loses more
  // precision and may lower the recall ceiling. Prefer kUniformUint8 by
  // default; use this only after validating that recall is sufficient and
  // when pursuing maximum performance.
  kUniformUint7 = 8,
  // Global uniform quantization with the full uint8 code range [0, 255].
  kUniformUint8 = 9,
};

struct ZVEC_CORE_API SerializableBase {
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

//! Common quantizer params shared by all quantizer types
struct ZVEC_CORE_API QuantizerParam : public SerializableBase {
  using Pointer = std::shared_ptr<QuantizerParam>;

  QuantizerType type = QuantizerType::kNone;
  bool enable_rotate =
      false;  // rotate vectors before quantization to reduce error

  // Constructors
  QuantizerParam(QuantizerType t = QuantizerType::kNone, bool rotate = false)
      : type(t), enable_rotate(rotate) {}
  virtual ~QuantizerParam() = default;

  //! Duplicate the param object, keeping the concrete type
  virtual Pointer Clone() const {
    return std::make_shared<QuantizerParam>(*this);
  }

  //! Create the param object matching the quantizer type
  static Pointer Create(QuantizerType t);

 protected:
  friend class BaseIndexParam;
  ailego::JsonObject SerializeToJsonObject(
      bool omit_empty_value = false) const override;

  bool DeserializeFromJsonObject(const ailego::JsonObject &json_obj) override;
};

//! Product-Quantization specific params
struct PqQuantizerParam : public QuantizerParam {
  int num_chunk = 8;  // M: number of chunks
  int num_bits = 8;   // bits per chunk

  // Constructors
  PqQuantizerParam(int chunks = 8, int bits = 8, bool rotate = false)
      : QuantizerParam(QuantizerType::kPQ, rotate),
        num_chunk(chunks),
        num_bits(bits) {}

  QuantizerParam::Pointer Clone() const override {
    return std::make_shared<PqQuantizerParam>(*this);
  }

 protected:
  friend class BaseIndexParam;
  ailego::JsonObject SerializeToJsonObject(
      bool omit_empty_value = false) const override;

  bool DeserializeFromJsonObject(const ailego::JsonObject &json_obj) override;
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

// --- GroupBy Parameters ---
struct GroupByParam {
  uint32_t group_topk{0};
  uint32_t group_count{0};
  std::function<std::string(uint64_t key)> group_by{};
};

// --- Query Parameters (can be passed to search methods) ---
class ZVEC_CORE_API BaseIndexQueryParam {
 public:
  using Pointer = std::shared_ptr<BaseIndexQueryParam>;

  BaseIndexQueryParam();
  BaseIndexQueryParam(const BaseIndexQueryParam &);
  BaseIndexQueryParam(BaseIndexQueryParam &&) noexcept;
  BaseIndexQueryParam &operator=(const BaseIndexQueryParam &);
  BaseIndexQueryParam &operator=(BaseIndexQueryParam &&) noexcept;
  virtual ~BaseIndexQueryParam();

  uint32_t topk = 10;
  bool fetch_vector = false;
  std::shared_ptr<IndexFilter> filter = nullptr;
  std::shared_ptr<std::vector<uint64_t>> bf_pks = nullptr;
  float radius = 0.0f;
  bool is_linear = false;
  RefinerParam::Pointer refiner_param = nullptr;
  std::shared_ptr<GroupByParam> group_by_param = nullptr;

  virtual Pointer Clone() const = 0;
};

struct ZVEC_CORE_API FlatQueryParam : public BaseIndexQueryParam {
  using Pointer = std::shared_ptr<FlatQueryParam>;

  FlatQueryParam();
  FlatQueryParam(const FlatQueryParam &);
  FlatQueryParam(FlatQueryParam &&) noexcept;
  FlatQueryParam &operator=(const FlatQueryParam &);
  FlatQueryParam &operator=(FlatQueryParam &&) noexcept;
  ~FlatQueryParam() override;

  BaseIndexQueryParam::Pointer Clone() const override;
};

struct ZVEC_CORE_API HNSWQueryParam : public BaseIndexQueryParam {
  using Pointer = std::shared_ptr<HNSWQueryParam>;

  HNSWQueryParam();
  HNSWQueryParam(const HNSWQueryParam &);
  HNSWQueryParam(HNSWQueryParam &&) noexcept;
  HNSWQueryParam &operator=(const HNSWQueryParam &);
  HNSWQueryParam &operator=(HNSWQueryParam &&) noexcept;
  ~HNSWQueryParam() override;

  uint32_t ef_search = kDefaultHnswEfSearch;
  uint32_t prefetch_offset = kDefaultPrefetchOffset;
  uint32_t prefetch_lines = kDefaultPrefetchLines;

  BaseIndexQueryParam::Pointer Clone() const override;
};

struct ZVEC_CORE_API HNSWRabitqQueryParam : public BaseIndexQueryParam {
  using Pointer = std::shared_ptr<HNSWRabitqQueryParam>;

  HNSWRabitqQueryParam();
  HNSWRabitqQueryParam(const HNSWRabitqQueryParam &);
  HNSWRabitqQueryParam(HNSWRabitqQueryParam &&) noexcept;
  HNSWRabitqQueryParam &operator=(const HNSWRabitqQueryParam &);
  HNSWRabitqQueryParam &operator=(HNSWRabitqQueryParam &&) noexcept;
  ~HNSWRabitqQueryParam() override;

  uint32_t ef_search = kDefaultHnswEfSearch;

  BaseIndexQueryParam::Pointer Clone() const override;
};

struct ZVEC_CORE_API IVFQueryParam : public BaseIndexQueryParam {
  IVFQueryParam();
  IVFQueryParam(const IVFQueryParam &);
  IVFQueryParam(IVFQueryParam &&) noexcept;
  IVFQueryParam &operator=(const IVFQueryParam &);
  IVFQueryParam &operator=(IVFQueryParam &&) noexcept;
  ~IVFQueryParam() override;

  int nprobe = 10;
  std::shared_ptr<BaseIndexQueryParam> l1QueryParam = nullptr;
  std::shared_ptr<BaseIndexQueryParam> l2QueryParam = nullptr;

  using Pointer = std::shared_ptr<IVFQueryParam>;

  BaseIndexQueryParam::Pointer Clone() const override;
};

struct ZVEC_CORE_API DiskAnnQueryParam : public BaseIndexQueryParam {
  using Pointer = std::shared_ptr<DiskAnnQueryParam>;

  DiskAnnQueryParam();
  DiskAnnQueryParam(const DiskAnnQueryParam &);
  DiskAnnQueryParam(DiskAnnQueryParam &&) noexcept;
  DiskAnnQueryParam &operator=(const DiskAnnQueryParam &);
  DiskAnnQueryParam &operator=(DiskAnnQueryParam &&) noexcept;
  ~DiskAnnQueryParam() override;

  // Beam-search candidate list size used at query time. Larger values improve
  // recall at the cost of latency.
  uint32_t list_size = kDefaultDiskAnnListSize;

  BaseIndexQueryParam::Pointer Clone() const override;
};

// --- Construction Parameters ---
// template<typename IndexQueryParamType>
class ZVEC_CORE_API BaseIndexParam : public SerializableBase {
 public:
  using Pointer = std::shared_ptr<BaseIndexParam>;

  explicit BaseIndexParam(IndexType type = IndexType::kNone,
                          MetricType metric = MetricType::kL2sq, int dim = 0,
                          int ver = 0);
  BaseIndexParam(const BaseIndexParam &);
  BaseIndexParam &operator=(const BaseIndexParam &);
  virtual ~BaseIndexParam();

  IndexType index_type = IndexType::kNone;
  MetricType metric_type = MetricType::kL2sq;
  int dimension = 0;  // [1, MAX_DIMENSION]
  int version = 0;    // for compatibility
  bool is_sparse = false;
  bool is_huge_page = false;
  DataType data_type = DataType::DT_UNDEFINED;
  bool use_id_map = true;
  bool use_external_vector = false;

  // IndexMeta meta;
  ailego::Params params;

  // pipeline
  PreprocessorParam preprocess_param;
  //! nullptr means no quantizer is configured (equivalent to kNone)
  QuantizerParam::Pointer quantizer_param{nullptr};

  BaseIndexQueryParam::Pointer default_query_param = nullptr;
  // virtual std::shared_ptr<BaseIndexQueryParam> GetDefaultQueryParam() const
  // {
  //   return std::make_shared<BaseIndexQueryParam>();
  // }
  //

 protected:
  bool DeserializeFromJsonObject(const ailego::JsonObject &json_obj) override;
  ailego::JsonObject SerializeToJsonObject(
      bool omit_empty_value = false) const override;
};

struct ZVEC_CORE_API FlatIndexParam : public BaseIndexParam {
  using Pointer = std::shared_ptr<FlatIndexParam>;
  FlatIndexParam() : BaseIndexParam(IndexType::kFlat) {}

  IndexMeta::MajorOrder major_order = IndexMeta::MajorOrder::MO_ROW;

 protected:
  bool DeserializeFromJsonObject(const ailego::JsonObject &json_obj) override;
  ailego::JsonObject SerializeToJsonObject(
      bool omit_empty_value = false) const override;
};

struct ZVEC_CORE_API IVFIndexParam : public BaseIndexParam {
  using Pointer = std::shared_ptr<IVFIndexParam>;
  int nlist = 1024;
  int niters = 10;
  std::shared_ptr<BaseIndexParam> l1Index = nullptr;
  std::shared_ptr<BaseIndexParam> l2Index = nullptr;
  bool use_soar = false;

  // Constructors with delegation
  IVFIndexParam();
  IVFIndexParam(int nlist, int niters, std::shared_ptr<BaseIndexParam> l1Index,
                std::shared_ptr<BaseIndexParam> l2Index);
  IVFIndexParam(MetricType metric, int dim, int nlist, int niters,
                std::shared_ptr<BaseIndexParam> l1Index,
                std::shared_ptr<BaseIndexParam> l2Index);
  IVFIndexParam(const IVFIndexParam &);
  IVFIndexParam(IVFIndexParam &&);
  IVFIndexParam &operator=(const IVFIndexParam &);
  IVFIndexParam &operator=(IVFIndexParam &&);
  ~IVFIndexParam() override;

  // query param:
  // topk of l1Index's param ==== IVFIndexQueryParam.nprobe
  // topk of l2Index's param ==== IVFIndexQueryParam.topK

  // IVFIndexParam.metric_type === l2Index's metric_type
  // IVFIndexParam.quantization === l2Index's quantization
};

struct ZVEC_CORE_API HNSWIndexParam : public BaseIndexParam {
  using Pointer = std::shared_ptr<HNSWIndexParam>;
  int m = kDefaultHnswNeighborCnt;
  int ef_construction = kDefaultHnswEfConstruction;
  bool use_contiguous_memory = false;

  // Optional provider of the original vectors used to build the graph,
  // with their meta. Runtime only, not serialized.
  core::IndexProvider::Pointer provider = nullptr;
  core::IndexMeta provider_meta{};

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

struct ZVEC_CORE_API VamanaIndexParam : public BaseIndexParam {
  using Pointer = std::shared_ptr<VamanaIndexParam>;
  int max_degree = kDefaultVamanaMaxDegree;
  int search_list_size = kDefaultVamanaSearchListSize;
  float alpha = kDefaultVamanaAlpha;
  int max_occlusion_size = kDefaultVamanaMaxOcclusionSize;
  bool saturate_graph = kDefaultVamanaSaturateGraph;
  bool use_contiguous_memory = false;

  VamanaIndexParam() : BaseIndexParam(IndexType::kVamana) {}

  VamanaIndexParam(int max_degree, int search_list_size, float alpha)
      : BaseIndexParam(IndexType::kVamana),
        max_degree(max_degree),
        search_list_size(search_list_size),
        alpha(alpha) {}

  VamanaIndexParam(MetricType metric, int dim, int max_degree,
                   int search_list_size, float alpha)
      : BaseIndexParam(IndexType::kVamana, metric, dim),
        max_degree(max_degree),
        search_list_size(search_list_size),
        alpha(alpha) {}

 protected:
  bool DeserializeFromJsonObject(const ailego::JsonObject &json_obj) override;
  ailego::JsonObject SerializeToJsonObject(
      bool omit_empty_value = false) const override;
};

struct ZVEC_CORE_API VamanaQueryParam : public BaseIndexQueryParam {
  using Pointer = std::shared_ptr<VamanaQueryParam>;

  uint32_t ef_search = kDefaultVamanaEfSearch;
  uint32_t prefetch_offset = kDefaultPrefetchOffset;
  uint32_t prefetch_lines = kDefaultPrefetchLines;

  BaseIndexQueryParam::Pointer Clone() const override;
};

struct ZVEC_CORE_API HNSWRabitqIndexParam : public BaseIndexParam {
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
  HNSWRabitqIndexParam();
  HNSWRabitqIndexParam(int m, int ef_construction);
  HNSWRabitqIndexParam(MetricType metric, int dim, int m, int ef_construction);
  HNSWRabitqIndexParam(const HNSWRabitqIndexParam &);
  HNSWRabitqIndexParam(HNSWRabitqIndexParam &&);
  HNSWRabitqIndexParam &operator=(const HNSWRabitqIndexParam &);
  HNSWRabitqIndexParam &operator=(HNSWRabitqIndexParam &&);
  ~HNSWRabitqIndexParam() override;

 protected:
  bool DeserializeFromJsonObject(const ailego::JsonObject &json_obj) override;
  ailego::JsonObject SerializeToJsonObject(
      bool omit_empty_value = false) const override;
};

struct ZVEC_CORE_API IVFRabitqIndexParam : public BaseIndexParam {
  using Pointer = std::shared_ptr<IVFRabitqIndexParam>;

  // IVF parameters
  int nlist = kDefaultIvfRabitqNlist;

  // Rabitq parameters
  int total_bits = kDefaultRabitqTotalBits;
  int sample_count = 0;

  IVFRabitqIndexParam();
  explicit IVFRabitqIndexParam(int nlist);
  IVFRabitqIndexParam(MetricType metric, int dim, int nlist);
  IVFRabitqIndexParam(const IVFRabitqIndexParam &);
  IVFRabitqIndexParam(IVFRabitqIndexParam &&);
  IVFRabitqIndexParam &operator=(const IVFRabitqIndexParam &);
  IVFRabitqIndexParam &operator=(IVFRabitqIndexParam &&);
  ~IVFRabitqIndexParam() override;

 protected:
  bool DeserializeFromJsonObject(const ailego::JsonObject &json_obj) override;
  ailego::JsonObject SerializeToJsonObject(
      bool omit_empty_value = false) const override;
};

struct ZVEC_CORE_API IVFRabitqQueryParam : public BaseIndexQueryParam {
  using Pointer = std::shared_ptr<IVFRabitqQueryParam>;

  IVFRabitqQueryParam();
  IVFRabitqQueryParam(const IVFRabitqQueryParam &);
  IVFRabitqQueryParam(IVFRabitqQueryParam &&) noexcept;
  IVFRabitqQueryParam &operator=(const IVFRabitqQueryParam &);
  IVFRabitqQueryParam &operator=(IVFRabitqQueryParam &&) noexcept;
  ~IVFRabitqQueryParam() override;

  uint32_t nprobe = kDefaultIvfRabitqNprobe;

  BaseIndexQueryParam::Pointer Clone() const override;
};

struct ZVEC_CORE_API DiskAnnIndexParam : public BaseIndexParam {
  using Pointer = std::shared_ptr<DiskAnnIndexParam>;

  int max_degree = kDefaultDiskAnnMaxDegree;
  int list_size = kDefaultDiskAnnListSize;
  int pq_chunk_num = kDefaultDiskAnnPqChunkNum;

  // Constructors with delegation
  DiskAnnIndexParam() : BaseIndexParam(IndexType::kDiskAnn) {}

  DiskAnnIndexParam(MetricType metric, int dim, int max_degree, int list_size,
                    int pq_chunk_num)
      : BaseIndexParam(IndexType::kDiskAnn, metric, dim),
        max_degree(max_degree),
        list_size(list_size),
        pq_chunk_num(pq_chunk_num) {}

 protected:
  bool DeserializeFromJsonObject(const ailego::JsonObject &json_obj) override;
  ailego::JsonObject SerializeToJsonObject(
      bool omit_empty_value = false) const override;
};

}  // namespace zvec::core_interface
