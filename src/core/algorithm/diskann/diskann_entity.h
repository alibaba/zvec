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

#include <utility>
#include <zvec/core/framework/index_framework.h>

namespace zvec {
namespace core {

using dist_t = float;
using diskann_key_t = uint64_t;
using diskann_id_t = uint32_t;

constexpr diskann_id_t kInvalidId = static_cast<diskann_id_t>(-1);
constexpr diskann_key_t kInvalidKey = static_cast<key_t>(-1);

struct VectorInfo {
  float dist_;
  std::string vec_;

  VectorInfo() = default;
  VectorInfo(float dist, const std::string &vec) : dist_{dist}, vec_{vec} {}
  VectorInfo(float dist, std::string &&vec)
      : dist_{dist}, vec_{std::move(vec)} {}
};

/*! Key Value Vecotr Heap Comparer
 */
struct KeyValueVectorHeapComparer {
  //! Function call
  bool operator()(const std::pair<diskann_id_t, VectorInfo> &lhs,
                  const std::pair<diskann_id_t, VectorInfo> &rhs) const {
    return compare_(lhs.second.dist_, rhs.second.dist_);
  }

 private:
  std::less<float> compare_;
};

/*! Key Value Vector Heap
 */
using TopkHeap = ailego::Heap<std::pair<diskann_id_t, VectorInfo>,
                              KeyValueVectorHeapComparer>;

struct DiskAnnMetaHeader {
 public:
  uint64_t doc_cnt;
  uint64_t ndims;
  uint64_t medoid;
  uint64_t max_node_size;
  uint64_t max_degree;
  uint64_t node_per_sector;
  uint64_t vamana_frozen_num;
  uint64_t vamana_frozen_loc;
  uint64_t append_reorder_data;
  uint64_t index_size;
  uint8_t reserved[4016];  /// pad DiskAnnMetaHeader to 4096 bytes

  DiskAnnMetaHeader() {
    clear();
  }

  DiskAnnMetaHeader(const DiskAnnMetaHeader &header) {
    memcpy(this, &header, sizeof(header));
  }

  DiskAnnMetaHeader &operator=(const DiskAnnMetaHeader &header) {
    if (this != &header) {
      memcpy(this, &header, sizeof(header));
    }
    return *this;
  }

  void reset() {
    doc_cnt = 0U;
  }

  void clear() {
    memset(this, 0, sizeof(DiskAnnMetaHeader));
  }
};

struct DiskAnnPqMeta {
 public:
  uint64_t full_pivot_data_size{0};
  uint64_t centroid_data_size{0};
  uint64_t chunk_offsets_size{0};
  uint64_t chunk_num{0};
  uint8_t reserved[128];

  DiskAnnPqMeta() {
    clear();
  }

  DiskAnnPqMeta(const DiskAnnPqMeta &meta) {
    memcpy(this, &meta, sizeof(meta));
  }

  DiskAnnPqMeta &operator=(const DiskAnnPqMeta &meta) {
    memcpy(this, &meta, sizeof(meta));
    return *this;
  }

  void clear() {
    memset(this, 0, sizeof(DiskAnnPqMeta));
  }
};

static_assert(sizeof(DiskAnnMetaHeader) == 4096,
              "DiskAnnMetaHeader size must stay 4096 bytes (on-disk format)");

static_assert(sizeof(DiskAnnPqMeta) % 32 == 0,
              "DiskAnnPqMeta size must be a multiple of 32 bytes");

class DiskAnnEntity {
 public:
  DiskAnnEntity() = default;
  virtual ~DiskAnnEntity() = default;

  //! Constructor
  DiskAnnEntity(const DiskAnnMetaHeader &meta_header,
                const DiskAnnPqMeta &pq_meta) {
    meta_header_ = meta_header;
    pq_meta_ = pq_meta;
  }

  //! DiskAnnEntity Pointerd;
  typedef std::shared_ptr<DiskAnnEntity> Pointer;

 public:
  static size_t AlignSize(size_t size) {
    return (size + 0xFFF) & (~0xFFF);
  }

 public:
  virtual int add_vector(diskann_key_t /*key*/, const void * /*vec*/) {
    return IndexError_NotImplemented;
  }

  virtual const void *get_vector(diskann_id_t /*id*/) const {
    return nullptr;
  }

  virtual std::pair<uint32_t, const diskann_id_t *> get_neighbors(
      diskann_id_t /*id*/) const {
    return std::make_pair(0, nullptr);
  }

  virtual int set_neighbors(
      diskann_id_t /*id*/, const std::vector<diskann_id_t> & /*neighbor_ids*/) {
    return IndexError_NotImplemented;
  }

  virtual int add_neighbor(diskann_id_t /*id*/, diskann_id_t /*neighbor_id*/) {
    return IndexError_NotImplemented;
  }

  //! Get node id of primary key
  virtual diskann_id_t get_id(diskann_key_t key) const = 0;

  //! Get primary key of the node id
  virtual diskann_key_t get_key(diskann_id_t id) const = 0;

 public:
  uint64_t max_node_size() const {
    return meta_header_.max_node_size;
  }

  uint64_t medoid() const {
    return meta_header_.medoid;
  }

  uint64_t *mutable_medoid() {
    return &meta_header_.medoid;
  }

  uint64_t node_per_sector() const {
    return meta_header_.node_per_sector;
  }

  uint64_t pq_chunk_num() {
    return pq_meta_.chunk_num;
  }

  uint64_t doc_cnt() const {
    return meta_header_.doc_cnt;
  }

  uint64_t *mutable_doc_cnt() {
    return &meta_header_.doc_cnt;
  }

  uint64_t max_degree() {
    return meta_header_.max_degree;
  }

  DiskAnnPqMeta *mutable_pq_meta() {
    return &pq_meta_;
  }

 public:
  virtual const DiskAnnEntity::Pointer clone() const {
    LOG_ERROR("Update neighbors not implemented");
    return DiskAnnEntity::Pointer();
  }

 public:
  const static std::string kDiskAnnVectorSegmentId;
  const static std::string kDiskAnnMetaSegmentId;
  const static std::string kDiskAnnPqMetaSegmentId;
  const static std::string kDiskAnnPqDataSegmentId;
  const static std::string kDiskAnnDummpySegmentId;
  const static std::string kDiskAnnMappingSegmentId;
  const static std::string kDiskAnnKeyMappingSegmentId;
  const static std::string kDiskAnnEntryPointSegmentId;
  const static std::string kDiskAnnKeySegmentId;

  constexpr static float kDefaultBFNegativeProbility = 0.001f;
  constexpr static float kDefaultGraphSlackFactor = 1.3f;
  constexpr static float kDefaultAlpha = 1.2f;
  constexpr static uint32_t kDefaultMaxOcclusionSize = 750;
  constexpr static uint32_t kDefaultMaxDegree = 100;
  constexpr static uint32_t kDefaultCompressBatchSize = 5000000;
  constexpr static uint32_t kRevision = 0U;

 protected:
  DiskAnnMetaHeader meta_header_;
  DiskAnnPqMeta pq_meta_;
};

}  // namespace core
}  // namespace zvec
