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

#include <string.h>
#include <ailego/utility/memory_helper.h>
#include <zvec/ailego/container/heap.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/core/framework/index_dumper.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_storage.h>

namespace zvec {
namespace core {

using node_id_t = uint32_t;
using key_t = uint64_t;
using level_t = int32_t;
using dist_t = float;
using TopkHeap = ailego::KeyValueHeap<node_id_t, dist_t>;
using CandidateHeap =
    ailego::KeyValueHeap<node_id_t, dist_t, std::greater<dist_t>>;
constexpr node_id_t kInvalidNodeId = static_cast<node_id_t>(-1);
constexpr key_t kInvalidKey = static_cast<key_t>(-1);
class HnswSparseDistCalculator;

struct SparseGraphHeader {
  uint32_t size;
  uint32_t version;
  uint32_t graph_type;
  uint32_t doc_count;
  uint32_t vector_size;
  uint32_t node_size;
  uint32_t l0_neighbor_count;
  uint32_t prune_type;
  uint32_t prune_neighbor_count;
  uint32_t ef_construction;
  uint32_t options;
  uint32_t min_neighbor_count;
  uint32_t sparse_meta_size;
  uint32_t sparse_unit_size;
  uint32_t total_sparse_count;
  uint8_t reserved[868];
};

static_assert(sizeof(SparseGraphHeader) % 32 == 0,
              "SparseGraphHeader must be aligned with 32 bytes");

//! Hnsw upper neighbor header
struct HnswSparseHeader {
  uint32_t size;      // header size
  uint32_t revision;  // current total docs of the graph
  uint32_t upper_neighbor_count;
  uint32_t ef_construction;
  uint32_t scaling_factor;
  uint32_t max_level;
  uint32_t entry_point;
  uint32_t options;
  uint8_t reserved[30];
};

struct SparseData {
 public:
  SparseData() {};

  SparseData(uint32_t sparse_count, const uint32_t *sparse_indices,
             const void *sparse_vec)
      : count(sparse_count), indices(sparse_indices), vec(sparse_vec) {}

  uint32_t count{0};
  const uint32_t *indices{nullptr};
  const void *vec{nullptr};
};

static_assert(sizeof(HnswSparseHeader) % 32 == 0,
              "SparseGraphHeader must be aligned with 32 bytes");

//! Hnsw common header and upper neighbor header
struct HNSWSparseHeader {
  HNSWSparseHeader() {
    clear();
  }

  HNSWSparseHeader(const HNSWSparseHeader &header) {
    memcpy(this, &header, sizeof(header));
  }

  HNSWSparseHeader &operator=(const HNSWSparseHeader &header) {
    memcpy(this, &header, sizeof(header));
    return *this;
  }

  //! Reset state to zero, and the params is untouched
  void inline reset() {
    graph.doc_count = 0U;
    hnsw.entry_point = kInvalidNodeId;
    hnsw.max_level = 0;
    graph.total_sparse_count = 0U;
  }

  //! Clear all fields to init value
  void inline clear() {
    memset(this, 0, sizeof(HNSWSparseHeader));
    hnsw.entry_point = kInvalidNodeId;
    graph.size = sizeof(SparseGraphHeader);
    hnsw.size = sizeof(HnswSparseHeader);
    graph.total_sparse_count = 0U;
  }

  size_t neighbor_cnt() const {
    return graph.l0_neighbor_count;
  }

  size_t upper_neighbor_cnt() const {
    return hnsw.upper_neighbor_count;
  }

  size_t vector_size() const {
    return graph.vector_size;
  }

  size_t ef_construction() const {
    return graph.ef_construction;
  }

  size_t scaling_factor() const {
    return hnsw.scaling_factor;
  }

  size_t neighbor_prune_cnt() const {
    return graph.prune_neighbor_count;
  }

  node_id_t entry_point() const {
    return hnsw.entry_point;
  }

  node_id_t doc_cnt() const {
    return graph.doc_count;
  }

  uint32_t total_sparse_count() const {
    return graph.total_sparse_count;
  }

  SparseGraphHeader graph;
  HnswSparseHeader hnsw;
};

struct NeighborsHeader {
  uint32_t neighbor_cnt;
  node_id_t neighbors[0];
};

struct Neighbors {
  Neighbors() : cnt{0}, data{nullptr} {}

  Neighbors(uint32_t cnt_in, const node_id_t *data_in)
      : cnt{cnt_in}, data{data_in} {}

  Neighbors(IndexStorage::MemoryBlock &&mem_block)
      : neighbor_block{std::move(mem_block)} {
    auto hd = reinterpret_cast<const NeighborsHeader *>(neighbor_block.data());
    cnt = hd->neighbor_cnt;
    data = hd->neighbors;
  }

  size_t size(void) const {
    return cnt;
  }

  const node_id_t &operator[](size_t idx) const {
    return data[idx];
  }

  uint32_t cnt;
  const node_id_t *data;
  IndexStorage::MemoryBlock neighbor_block;
};

//! level 0 neighbors offset
struct SparseGraphNeighborMeta {
  SparseGraphNeighborMeta(size_t o, size_t cnt)
      : offset(o), neighbor_cnt(cnt) {}

  uint64_t offset : 48;
  uint64_t neighbor_cnt : 16;
};

//! hnsw upper neighbors meta
struct HnswSparseNeighborMeta {
  HnswSparseNeighborMeta(size_t o, size_t l) : offset(o), level(l) {}

  uint64_t offset : 48;  // offset = idx * upper neighors size
  uint64_t level : 16;
};

class HnswSparseEntity {
 public:
  //! Constructor
  HnswSparseEntity() {}

  //! Constructor
  HnswSparseEntity(const HNSWSparseHeader &hd) {
    header_ = hd;
  }

  //! Destructor
  virtual ~HnswSparseEntity() {}

  //! HnswSparseEntity Pointerd;
  typedef std::shared_ptr<HnswSparseEntity> Pointer;

  //! Get max neighbor size of graph level
  inline size_t neighbor_cnt(level_t level) const {
    return level == 0 ? header_.graph.l0_neighbor_count
                      : header_.hnsw.upper_neighbor_count;
  }

  //! get max neighbor size of graph level 0
  inline size_t l0_neighbor_cnt() const {
    return header_.graph.l0_neighbor_count;
  }

  //! get min neighbor size of graph
  inline size_t min_neighbor_cnt() const {
    return header_.graph.min_neighbor_count;
  }

  //! get upper neighbor size of graph level other than 0
  inline size_t upper_neighbor_cnt() const {
    return header_.hnsw.upper_neighbor_count;
  }

  //! Get current total doc of the hnsw graph
  inline node_id_t *mutable_doc_cnt() {
    return &header_.graph.doc_count;
  }

  inline node_id_t doc_cnt() const {
    return header_.graph.doc_count;
  }

  inline uint32_t *mutable_total_sparse_count() {
    return &header_.graph.total_sparse_count;
  }

  uint32_t total_sparse_count() const {
    return header_.graph.total_sparse_count;
  }

  //! Get hnsw graph scaling params
  inline size_t scaling_factor() const {
    return header_.hnsw.scaling_factor;
  }

  //! Get prune_size
  inline size_t prune_cnt() const {
    return header_.graph.prune_neighbor_count;
  }

  //! Current entity of top level graph
  inline node_id_t entry_point() const {
    return header_.hnsw.entry_point;
  }

  //! Current max graph level
  inline level_t cur_max_level() const {
    return header_.hnsw.max_level;
  }

  //! Retrieve index vector size
  size_t vector_size() const {
    return header_.graph.vector_size;
  }

  //! Retrieve node size
  size_t node_size() const {
    return header_.graph.node_size;
  }

  //! Retrieve ef constuction
  size_t ef_construction() const {
    return header_.graph.ef_construction;
  }

  //! Retrieve sparse meta size
  size_t sparse_meta_size() const {
    return header_.graph.sparse_meta_size;
  }

  //! Retrieve sparse unit size
  size_t sparse_unit_size() const {
    return header_.graph.sparse_unit_size;
  }

  void set_vector_size(size_t size) {
    header_.graph.vector_size = size;
  }

  void set_prune_cnt(size_t v) {
    header_.graph.prune_neighbor_count = v;
  }

  void set_scaling_factor(size_t val) {
    header_.hnsw.scaling_factor = val;
  }

  void set_l0_neighbor_cnt(size_t cnt) {
    header_.graph.l0_neighbor_count = cnt;
  }

  void set_min_neighbor_cnt(size_t cnt) {
    header_.graph.min_neighbor_count = cnt;
  }

  void set_upper_neighbor_cnt(size_t cnt) {
    header_.hnsw.upper_neighbor_count = cnt;
  }

  void set_ef_construction(size_t ef) {
    header_.graph.ef_construction = ef;
  }

  void set_sparse_meta_size(size_t size) {
    header_.graph.sparse_meta_size = size;
  }

  void set_sparse_unit_size(size_t size) {
    header_.graph.sparse_unit_size = size;
  }

 protected:
  inline const HNSWSparseHeader &header() const {
    return header_;
  }

  inline HNSWSparseHeader *mutable_header() {
    return &header_;
  }

  inline size_t header_size() const {
    return sizeof(header_);
  }

  void set_node_size(size_t size) {
    header_.graph.node_size = size;
  }

  //! Dump all segment by dumper
  //! Return dump size if success, errno(<0) in failure
  int64_t dump_segments(
      const IndexDumper::Pointer &dumper, key_t *keys,
      const std::function<level_t(node_id_t)> &get_level) const;

 private:
  //! dump mapping segment, for get_vector_by_key in provider
  int64_t dump_mapping_segment(const IndexDumper::Pointer &dumper,
                               const key_t *keys) const;

  //! dump hnsw head by dumper
  //! Return dump size if success, errno(<0) in failure
  int64_t dump_header(const IndexDumper::Pointer &dumper,
                      const HNSWSparseHeader &hd) const;

  //! dump vectors by dumper
  //! Return dump size if success, errno(<0) in failure
  int64_t dump_sparse_vector_meta(
      const IndexDumper::Pointer &dumper,
      const std::vector<node_id_t> &reorder_mapping) const;

  //! dump sparse vectors by dumper
  //! Return dump size if success, errno(<0) in failure
  int64_t dump_sparse_vector(
      const IndexDumper::Pointer &dumper,
      const std::vector<node_id_t> &reorder_mapping) const;

  //! dump hnsw neighbors by dumper
  //! Return dump size if success, errno(<0) in failure
  int64_t dump_neighbors(const IndexDumper::Pointer &dumper,
                         const std::function<level_t(node_id_t)> &get_level,
                         const std::vector<node_id_t> &reorder_mapping,
                         const std::vector<node_id_t> &neighbor_mapping) const {
    auto len1 = dump_graph_neighbors(dumper, reorder_mapping, neighbor_mapping);
    if (len1 < 0) {
      return len1;
    }
    auto len2 = dump_upper_neighbors(dumper, get_level, reorder_mapping,
                                     neighbor_mapping);
    if (len2 < 0) {
      return len2;
    }

    return len1 + len2;
  }

  //! dump segment by dumper
  //! Return dump size if success, errno(<0) in failure
  int64_t dump_segment(const IndexDumper::Pointer &dumper,
                       const std::string &segment_id, const void *data,
                       size_t size) const;

  //! Dump level 0 neighbors
  //! Return dump size if success, errno(<0) in failure
  int64_t dump_graph_neighbors(
      const IndexDumper::Pointer &dumper,
      const std::vector<node_id_t> &reorder_mapping,
      const std::vector<node_id_t> &neighbor_mapping) const;

  //! Dump upper level neighbors
  //! Return dump size if success, errno(<0) in failure
  int64_t dump_upper_neighbors(
      const IndexDumper::Pointer &dumper,
      const std::function<level_t(node_id_t)> &get_level,
      const std::vector<node_id_t> &reorder_mapping,
      const std::vector<node_id_t> &neighbor_mapping) const;

 public:
  //! Cleanup the entity
  virtual int cleanup(void) {
    header_.clear();
    return 0;
  }

  //! Make a copy of searcher entity, to support thread-safe operation.
  //! The segment in container cannot be read concurrenly
  virtual const HnswSparseEntity::Pointer clone() const {
    LOG_ERROR("Update neighbors not implemented");
    return HnswSparseEntity::Pointer();
  }

  //! Get primary key of the node id
  virtual key_t get_key(node_id_t id) const = 0;

  //! Get vector feature data by key
  virtual const void *get_vector_meta(node_id_t id) const = 0;

  virtual int get_vector_meta(const node_id_t id,
                              IndexStorage::MemoryBlock &block) const = 0;

  //! Get vectors feature data by keys
  virtual int get_vector_metas(const node_id_t *ids, uint32_t count,
                               const void **vecs) const = 0;
  virtual int get_vector_metas(
      const node_id_t *ids, uint32_t count,
      std::vector<IndexStorage::MemoryBlock> &block_vecs) const = 0;

  //! Retrieve a sparse vector using a primary key
  virtual int get_sparse_vector_by_key(
      uint64_t /*key*/, uint32_t * /*sparse_count*/,
      std::string * /*sparse_indices_buffer*/,
      std::string * /*sparse_values_buffer*/) const {
    LOG_ERROR("get sparse vector not implemented");
    return IndexError_NotImplemented;
  }

  //! Retrieve a sparse vector using a primary key
  virtual int get_sparse_vector_by_id(
      node_id_t /*id*/, uint32_t * /*sparse_count*/,
      std::string * /*sparse_indices_buffer*/,
      std::string * /*sparse_values_buffer*/) const {
    LOG_ERROR("get sparse vector not implemented");
    return IndexError_NotImplemented;
  }

  //! Get vector sparse feature data by chunk index and offset
  virtual const void *get_sparse_data(uint64_t offset, uint32_t len) const = 0;

  //! Get sparse data from id
  virtual const void *get_sparse_data(node_id_t id) const = 0;

  virtual int get_sparse_data(uint64_t offset, uint32_t len,
                              IndexStorage::MemoryBlock &block) const = 0;

  virtual int get_sparse_data(const node_id_t id,
                              IndexStorage::MemoryBlock &block) const = 0;

  //! Get sparse data from vector
  virtual std::pair<const void *, uint32_t> get_sparse_data_from_vector(
      const void *vec) const = 0;
  virtual int get_sparse_data_from_vector(const void *vec,
                                          IndexStorage::MemoryBlock &block,
                                          int &sparse_length) const = 0;

  //! Get the node id's neighbors on graph level
  //! Note: the neighbors cannot be modified, using the following
  //! method to get WritableNeighbors if want to
  virtual const Neighbors get_neighbors(level_t level, node_id_t id) const = 0;

  //! Add vector and key to hnsw entity, and local id will be saved in id
  virtual int add_vector(level_t /*level*/, key_t /*key*/,
                         const std::string & /*vec*/, uint32_t /*sparse_count*/,
                         node_id_t * /*id*/) {
    return IndexError_NotImplemented;
  }

  virtual int add_vector(level_t /*level*/, key_t /*key*/,
                         const uint32_t /*sparse_count*/,
                         const uint32_t * /*sparse_indices*/,
                         const void * /*sparse_vec*/, node_id_t * /*id*/) {
    return IndexError_NotImplemented;
  }

  //! Add vector and id
  virtual int add_vector_with_id(level_t /*level*/, node_id_t /*id*/,
                                 const std::string & /*vec*/,
                                 uint32_t /*sparse_count*/) {
    return IndexError_NotImplemented;
  }

  virtual int update_neighbors(
      level_t /*level*/, node_id_t /*id*/,
      const std::vector<std::pair<node_id_t, dist_t>> & /*neighbors*/) {
    LOG_ERROR("Update neighbors dense not implemented");

    return 0;
  }

  //! Append neighbor_id to node id neighbors on level, size is the current
  //! neighbors size. Notice: the caller must be ensure the neighbors not full
  virtual void add_neighbor(level_t /*level*/, node_id_t /*id*/,
                            uint32_t /*size*/, node_id_t /*neighbor_id*/) {
    LOG_ERROR("Add neighbor not implemented");
  }

  //! Update entry point and max level
  virtual void update_ep_and_level(node_id_t ep, level_t level) {
    header_.hnsw.entry_point = ep;
    header_.hnsw.max_level = level;
  }

  virtual int load(const IndexStorage::Pointer & /*container*/,
                   bool /*check_crc*/) {
    LOG_ERROR("Load not implemented");
    return IndexError_NotImplemented;
  }

  virtual int dump(const IndexDumper::Pointer & /*dumper*/) {
    LOG_ERROR("Dump not implemented");
    return IndexError_NotImplemented;
  }

  static int CalcAndAddPadding(const IndexDumper::Pointer &dumper,
                               size_t data_size, size_t *padding_size);

 protected:
  static inline size_t AlignSize(size_t size) {
    return (size + 0x1F) & (~0x1F);
  }

  static inline size_t AlignPageSize(size_t size) {
    size_t page_mask = ailego::MemoryHelper::PageSize() - 1;
    return (size + page_mask) & (~page_mask);
  }

  //! rearrange vectors to improve cache locality
  void reshuffle_vectors(const std::function<level_t(node_id_t)> &get_level,
                         std::vector<node_id_t> *n2o_mapping,
                         std::vector<node_id_t> *o2n_mapping,
                         key_t *keys) const;

 public:
  const static std::string kSparseGraphHeaderSegmentId;
  const static std::string kSparseGraphFeaturesSegmentId;
  const static std::string kSparseGraphKeysSegmentId;
  const static std::string kSparseGraphNeighborsSegmentId;
  const static std::string kSparseGraphOffsetsSegmentId;
  const static std::string kSparseGraphMappingSegmentId;
  const static std::string kSparseHnswHeaderSegmentId;
  const static std::string kSparseHnswNeighborsSegmentId;
  const static std::string kSparseHnswOffsetsSegmentId;
  const static std::string kSparseGraphVectorsSegmentId;
  const static std::string kSparseGraphVectorMetaSegmentId;

  constexpr static uint32_t kRevision = 0U;
  constexpr static size_t kMaxGraphLayers = 15;
  constexpr static uint32_t kDefaultEfConstruction = 500;
  constexpr static uint32_t kDefaultEf = 500;
  constexpr static uint32_t kDefaultUpperMaxNeighborCnt = 50;  // M of HNSW
  constexpr static uint32_t kDefaultL0MaxNeighborCnt = 100;
  constexpr static uint32_t kMaxNeighborCnt = 65535;
  constexpr static float kDefaultScanRatio = 0.1f;
  constexpr static uint32_t kDefaultMinScanLimit = 10000;
  constexpr static uint32_t kDefaultMaxScanLimit =
      std::numeric_limits<uint32_t>::max();
  constexpr static float kDefaultBFNegativeProbability = 0.001f;
  constexpr static uint32_t kDefaultScalingFactor = 50U;
  constexpr static uint32_t kDefaultBruteForceThreshold = 1000U;
  constexpr static uint32_t kDefaultDocsHardLimit = 1 << 30U;  // 1 billion
  constexpr static float kDefaultDocsSoftLimitRatio = 0.9f;
  constexpr static size_t kMaxChunkSize = 0xFFFFFFFF;
  constexpr static size_t kDefaultChunkSize = 2UL * 1024UL * 1024UL;
  constexpr static size_t kDefaultMaxChunkCnt = 50000UL;
  constexpr static float kDefaultNeighborPruneMultiplier =
      1.0f;  // prune_cnt = upper_max_neighbor_cnt * multiplier
  constexpr static float kDefaultL0MaxNeighborCntMultiplier =
      2.0f;  // l0_max_neighbor_cnt = upper_max_neighbor_cnt * multiplier

  constexpr static uint32_t kSparseMetaSize = 2u * sizeof(uint64_t);
  constexpr static float kDefaultSparseNeighborRatio = 0.5f;
  constexpr static uint32_t kSparseMaxDimSize = 4096;
  constexpr static float kDefaultQueryFilteringRatio = 0.0f;  // turn off

 protected:
  HNSWSparseHeader header_{};
};

}  // namespace core
}  // namespace zvec
