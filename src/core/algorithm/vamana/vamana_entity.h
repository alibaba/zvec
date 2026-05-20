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
#include <functional>
#include <ailego/utility/memory_helper.h>
#include <zvec/ailego/container/heap.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/core/framework/index_dumper.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_storage.h>

// Reuse typed MemoryBlock and NeighborsT from hnsw_entity.h
#include "algorithm/hnsw/hnsw_entity.h"

namespace zvec {
namespace core {

// Vamana graph header — single-layer graph (no hierarchical levels)
struct VamanaGraphHeader {
  uint32_t size;
  uint32_t version;
  uint32_t graph_type;
  uint32_t doc_count;
  uint32_t vector_size;
  uint32_t node_size;
  uint32_t max_degree;          // R: maximum out-degree
  uint32_t search_list_size;    // L: search list size for construction
  uint32_t max_occlusion_size;  // C: max candidate size for RobustPrune
  uint32_t entry_point;         // medoid node id
  uint32_t options;
  uint32_t reserved_pad;
  float alpha;  // alpha parameter for RobustPrune
  uint8_t reserved_[4076];
};

static_assert(sizeof(VamanaGraphHeader) % 32 == 0,
              "VamanaGraphHeader must be aligned with 32 bytes");

struct VamanaHeader {
  VamanaHeader() {
    clear();
  }

  VamanaHeader(const VamanaHeader &header) {
    memcpy(this, &header, sizeof(header));
  }

  VamanaHeader &operator=(const VamanaHeader &header) {
    memcpy(this, &header, sizeof(header));
    return *this;
  }

  void inline reset() {
    graph.doc_count = 0U;
    graph.entry_point = kInvalidNodeId;
  }

  void inline clear() {
    memset(this, 0, sizeof(VamanaHeader));
    graph.entry_point = kInvalidNodeId;
    graph.size = sizeof(VamanaGraphHeader);
    graph.alpha = 1.2f;
  }

  size_t max_degree() const {
    return graph.max_degree;
  }
  size_t vector_size() const {
    return graph.vector_size;
  }
  size_t search_list_size() const {
    return graph.search_list_size;
  }
  size_t max_occlusion_size() const {
    return graph.max_occlusion_size;
  }
  float alpha() const {
    return graph.alpha;
  }
  node_id_t entry_point() const {
    return graph.entry_point;
  }
  node_id_t doc_cnt() const {
    return graph.doc_count;
  }

  VamanaGraphHeader graph;
};

// VamanaEntity: base class for Vamana graph data management
class VamanaEntity {
 public:
  VamanaEntity() {}
  VamanaEntity(const VamanaHeader &hd) {
    header_ = hd;
  }
  virtual ~VamanaEntity() {}

  typedef std::shared_ptr<VamanaEntity> Pointer;

  // Options bit flags (stored in VamanaGraphHeader::options)
  static constexpr uint32_t kOptionSaturateGraph = 1U << 0;

  // Default constants
  static constexpr uint32_t kDefaultMaxDegree = 64;
  static constexpr uint32_t kDefaultSearchListSize = 100;
  static constexpr uint32_t kDefaultMaxOcclusionSize = 750;
  static constexpr float kDefaultAlpha = 1.2f;
  static constexpr bool kDefaultSaturateGraph = false;
  static constexpr uint32_t kDefaultEf = 200;
  static constexpr float kDefaultScanRatio = 0.1f;
  static constexpr uint32_t kDefaultBruteForceThreshold = 1000U;
  static constexpr uint32_t kDefaultDocsHardLimit = 1 << 30U;
  static constexpr float kDefaultDocsSoftLimitRatio = 0.9f;
  static constexpr size_t kMaxChunkSize = 0xFFFFFFFF;
  static constexpr size_t kDefaultChunkSize = 2UL * 1024UL * 1024UL;
  static constexpr size_t kDefaultMaxChunkCnt = 50000UL;
  static constexpr uint32_t kDefaultMinScanLimit = 10000;
  static constexpr uint32_t kDefaultMaxScanLimit =
      std::numeric_limits<uint32_t>::max();
  static constexpr float kDefaultBFNegativeProbability = 0.001f;

  inline size_t max_degree() const {
    return header_.graph.max_degree;
  }
  inline node_id_t *mutable_doc_cnt() {
    return &header_.graph.doc_count;
  }
  inline node_id_t doc_cnt() const {
    return header_.graph.doc_count;
  }
  inline float alpha() const {
    return header_.graph.alpha;
  }
  inline size_t search_list_size() const {
    return header_.graph.search_list_size;
  }
  inline size_t max_occlusion_size() const {
    return header_.graph.max_occlusion_size;
  }
  inline node_id_t entry_point() const {
    return header_.graph.entry_point;
  }
  inline size_t vector_size() const {
    return header_.graph.vector_size;
  }
  inline size_t node_size() const {
    return header_.graph.node_size;
  }

  void set_vector_size(size_t size) {
    header_.graph.vector_size = size;
  }
  void set_max_degree(uint32_t val) {
    header_.graph.max_degree = val;
  }
  void set_search_list_size(uint32_t val) {
    header_.graph.search_list_size = val;
  }
  void set_max_occlusion_size(uint32_t val) {
    header_.graph.max_occlusion_size = val;
  }
  void set_alpha(float val) {
    header_.graph.alpha = val;
  }

  inline bool saturate_graph() const {
    return (header_.graph.options & kOptionSaturateGraph) != 0;
  }
  void set_saturate_graph(bool val) {
    if (val) {
      header_.graph.options |= kOptionSaturateGraph;
    } else {
      header_.graph.options &= ~kOptionSaturateGraph;
    }
  }

  // Neighbor size: NeighborsHeader + max_degree * sizeof(node_id_t)
  inline size_t neighbors_size() const {
    return sizeof(NeighborsHeader) + max_degree() * sizeof(node_id_t);
  }

  virtual void update_entry_point(node_id_t ep) {
    header_.graph.entry_point = ep;
  }

  virtual int cleanup() {
    header_.clear();
    return 0;
  }

  virtual const VamanaEntity::Pointer clone() const {
    return VamanaEntity::Pointer();
  }

  // Pure virtual interface
  virtual key_t get_key(node_id_t id) const = 0;
  virtual const void *get_vector(node_id_t id) const = 0;
  virtual int get_vector(const node_id_t id,
                         IndexStorage::MemoryBlock &block) const = 0;
  virtual int get_vector(const node_id_t *ids, uint32_t count,
                         const void **vecs) const = 0;
  virtual int get_vector(
      const node_id_t *ids, uint32_t count,
      std::vector<IndexStorage::MemoryBlock> &vec_blocks) const = 0;
  virtual const Neighbors get_neighbors(node_id_t id) const = 0;

  virtual int add_vector(key_t key, const void *vec, node_id_t *id) {
    return IndexError_NotImplemented;
  }
  virtual int add_vector_with_id(node_id_t id, const void *vec) {
    return IndexError_NotImplemented;
  }
  virtual int update_neighbors(
      node_id_t id,
      const std::vector<std::pair<node_id_t, dist_t>> &neighbors) {
    return IndexError_NotImplemented;
  }
  virtual void add_neighbor(node_id_t id, uint32_t size,
                            node_id_t neighbor_id) {}

  // --- Neighbor distance storage (CSR-like, lazy-loaded) ---
  // Each node has max_degree dist_t slots, the i-th slot stores the distance
  // from this node to its i-th neighbor. Only allocated/loaded when needed
  // (first write operation). Search-only paths never touch this data.

  // Ensure distance storage is allocated/loaded. Must be called before
  // any get/set neighbor dist operations. Thread-safe (idempotent).
  virtual int ensure_dist_storage() {
    return 0;
  }

  // Whether distance storage is currently loaded
  virtual bool dist_storage_loaded() const {
    return false;
  }

  // Get pointer to the distance array for node `id`.
  // Returns nullptr if dist storage is not loaded.
  virtual const dist_t *get_neighbor_dists(node_id_t id) const {
    return nullptr;
  }

  // Update all neighbor distances for node `id` from a prune result.
  virtual void update_neighbor_dists(
      node_id_t id,
      const std::vector<std::pair<node_id_t, dist_t>> &neighbors) {}

  // Set the distance for the `idx`-th neighbor of node `id`.
  virtual void set_neighbor_dist(node_id_t id, uint32_t idx, dist_t dist) {}

  virtual int dump(const IndexDumper::Pointer &dumper) {
    return IndexError_NotImplemented;
  }

  virtual const void *get_vector_by_key(uint64_t /*key*/) const {
    return nullptr;
  }
  virtual int get_vector_by_key(const key_t /*key*/,
                                IndexStorage::MemoryBlock & /*block*/) const {
    return IndexError_NotImplemented;
  }

  static int CalcAndAddPadding(const IndexDumper::Pointer &dumper,
                               size_t data_size, size_t *padding_size);

 protected:
  inline const VamanaHeader &header() const {
    return header_;
  }
  inline VamanaHeader *mutable_header() {
    return &header_;
  }
  inline size_t header_size() const {
    return sizeof(header_);
  }

  void set_node_size(size_t size) {
    header_.graph.node_size = size;
  }

  int64_t dump_segments(const IndexDumper::Pointer &dumper, key_t *keys) const;

  int64_t dump_segment(const IndexDumper::Pointer &dumper,
                       const std::string &segment_id, const void *data,
                       size_t size) const;

  int64_t dump_header(const IndexDumper::Pointer &dumper,
                      const VamanaHeader &hd) const;

  int64_t dump_vectors(const IndexDumper::Pointer &dumper,
                       const std::vector<node_id_t> &reorder_mapping) const;

  int64_t dump_neighbors(const IndexDumper::Pointer &dumper,
                         const std::vector<node_id_t> &reorder_mapping,
                         const std::vector<node_id_t> &neighbor_mapping) const;

  int64_t dump_neighbor_dists(
      const IndexDumper::Pointer &dumper,
      const std::vector<node_id_t> &reorder_mapping) const;

  int64_t dump_mapping_segment(const IndexDumper::Pointer &dumper,
                               const key_t *keys) const;

  void reshuffle_vectors(std::vector<node_id_t> *n2o_mapping,
                         std::vector<node_id_t> *o2n_mapping,
                         key_t *keys) const;

  static inline size_t AlignSize(size_t size) {
    return (size + 0x1F) & (~0x1F);
  }

  static inline size_t AlignPageSize(size_t size) {
    size_t page_mask = ailego::MemoryHelper::PageSize() - 1;
    return (size + page_mask) & (~page_mask);
  }

  static inline size_t AlignHugePageSize(size_t size) {
    size_t page_mask = ailego::MemoryHelper::HugePageSize() - 1;
    return (size + page_mask) & (~page_mask);
  }

 public:
  const static std::string kGraphHeaderSegmentId;
  const static std::string kGraphFeaturesSegmentId;
  const static std::string kGraphKeysSegmentId;
  const static std::string kGraphNeighborsSegmentId;
  const static std::string kGraphOffsetsSegmentId;
  const static std::string kGraphMappingSegmentId;
  const static std::string kGraphNeighborDistsSegmentId;

  static constexpr uint32_t kRevision = 0U;

 protected:
  VamanaHeader header_{};
};

}  // namespace core
}  // namespace zvec
