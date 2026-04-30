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

#include <iostream>
#include <memory>
#include <mutex>
#if defined(__linux__) || defined(__APPLE__)
#include <sys/mman.h>
#endif
#include <ailego/parallel/lock.h>
#include <sparsehash/dense_hash_map>
#include <zvec/ailego/container/heap.h>
#include <zvec/core/framework/index_framework.h>
#include "algorithm/hnsw/hnsw_chunk.h"
#include "algorithm/hnsw/hnsw_entity.h"  // MmapMemoryBlock, BufferPoolMemoryBlock, NeighborsT
#include "vamana_entity.h"
#include "vamana_params.h"

namespace zvec {
namespace core {


// Storage mode for VamanaStreamerEntity
enum class VamanaStorageMode { kMmap = 0, kBufferPool = 1, kContiguous = 2 };

// VamanaStreamerEntity manages vector data, primary keys, and neighbors
// for a single-layer Vamana graph in streaming (incremental) mode.
// Unlike HNSW, Vamana has no upper-level neighbors — only a single
// neighbor list per node. Node layout in chunk:
//   [vector_data (vector_size) | key (sizeof(key_t)) | NeighborsHeader +
//   neighbors (neighbors_size)]
class VamanaStreamerEntity : public VamanaEntity {
 public:
  // Virtual interface implementation
  virtual int cleanup() override;
  virtual const VamanaEntity::Pointer clone() const override;
  virtual key_t get_key(node_id_t id) const override;
  virtual const void *get_vector(node_id_t id) const override;
  virtual int get_vector(const node_id_t id,
                         IndexStorage::MemoryBlock &block) const override;
  virtual int get_vector(const node_id_t *ids, uint32_t count,
                         const void **vecs) const override;
  virtual int get_vector(
      const node_id_t *ids, uint32_t count,
      std::vector<IndexStorage::MemoryBlock> &vec_blocks) const override;
  virtual const Neighbors get_neighbors(node_id_t id) const override;

  virtual int add_vector(key_t key, const void *vec, node_id_t *id) override;
  virtual int add_vector_with_id(node_id_t id, const void *vec) override;
  virtual int update_neighbors(
      node_id_t id,
      const std::vector<std::pair<node_id_t, dist_t>> &neighbors) override;
  virtual void add_neighbor(node_id_t id, uint32_t size,
                            node_id_t neighbor_id) override;
  virtual int dump(const IndexDumper::Pointer &dumper) override;
  virtual void update_entry_point(node_id_t ep) override;

  // --- Neighbor distance storage ---
  int ensure_dist_storage() override;
  bool dist_storage_loaded() const override {
    return dist_loaded_;
  }
  const dist_t *get_neighbor_dists(node_id_t id) const override;
  void update_neighbor_dists(
      node_id_t id,
      const std::vector<std::pair<node_id_t, dist_t>> &neighbors) override;
  void set_neighbor_dist(node_id_t id, uint32_t idx, dist_t dist) override;

  virtual VamanaStorageMode storage_mode() const {
    return VamanaStorageMode::kMmap;
  }

  void set_use_key_info_map(bool use_id_map) {
    use_key_info_map_ = use_id_map;
  }

 public:
  VamanaStreamerEntity(IndexStreamer::Stats &stats);
  ~VamanaStreamerEntity();

  virtual const void *get_vector_by_key(key_t key) const override {
    auto id = get_id(key);
    return id == kInvalidNodeId ? nullptr : get_vector(id);
  }

  virtual int get_vector_by_key(
      const key_t key, IndexStorage::MemoryBlock &block) const override {
    auto id = get_id(key);
    if (id != kInvalidNodeId) {
      return get_vector(id, block);
    }
    return IndexError_InvalidArgument;
  }

  int init(size_t max_doc_cnt);
  int flush(uint64_t checkpoint);
  int open(IndexStorage::Pointer stg, uint64_t max_index_size, bool check_crc);
  int close();

  int set_index_meta(const IndexMeta &meta) const {
    return IndexHelper::SerializeToStorage(meta, broker_->storage().get());
  }

  int get_index_meta(IndexMeta *meta) const {
    return IndexHelper::DeserializeFromStorage(broker_->storage().get(), meta);
  }

  inline void set_chunk_size(size_t val) {
    chunk_size_ = val;
  }
  inline void set_get_vector(bool val) {
    get_vector_enabled_ = val;
  }

  inline node_id_t get_id(key_t key) const {
    if (use_key_info_map_) {
      keys_map_lock_->lock_shared();
      auto it = keys_map_->find(key);
      keys_map_lock_->unlock_shared();
      return it == keys_map_->end() ? kInvalidNodeId : it->second;
    }
    return key;
  }

  // --- Typed access methods for hot-path optimization ---
  // These are templated on MemBlock type to avoid runtime branching.

  template <typename MemBlock>
  inline NeighborsT<MemBlock> get_neighbors_typed(node_id_t id) const;

  template <typename MemBlock>
  inline int get_vector_typed(const node_id_t *ids, uint32_t count,
                              std::vector<MemBlock> &vec_blocks) const;

  template <typename MemBlock>
  inline key_t get_key_typed(node_id_t id) const;

 protected:
  inline void sync_node_chunks(size_t idx) const {
    sync_chunks(ChunkBroker::CHUNK_TYPE_NODE, idx, &node_chunks_);
  }

 protected:
  template <class Key, class T>
  using HashMap = google::dense_hash_map<Key, T, std::hash<Key>>;
  template <class Key, class T>
  using HashMapPointer = std::shared_ptr<HashMap<Key, T>>;

  //! Clone constructor, used by clone method in subclasses
  VamanaStreamerEntity(IndexStreamer::Stats &stats, const VamanaHeader &hd,
                       size_t chunk_size, uint32_t node_index_mask_bits,
                       bool get_vector_enabled, bool use_key_info_map,
                       std::shared_ptr<ailego::SharedMutex> &keys_map_lock,
                       const HashMapPointer<key_t, node_id_t> &keys_map,
                       std::vector<Chunk::Pointer> &&node_chunks,
                       const ChunkBroker::Pointer &broker)
      : stats_(stats),
        chunk_size_(chunk_size),
        node_index_mask_bits_(node_index_mask_bits),
        node_cnt_per_chunk_(1UL << node_index_mask_bits_),
        node_index_mask_(node_cnt_per_chunk_ - 1),
        get_vector_enabled_(get_vector_enabled),
        use_key_info_map_(use_key_info_map),
        keys_map_lock_(keys_map_lock),
        keys_map_(keys_map),
        node_chunks_(std::move(node_chunks)),
        broker_(broker) {
    *mutable_header() = hd;
    neighbor_size_ = neighbors_size();
  }

  //! Lazy chunk synchronization: fetches chunks from broker when needed.
  //! Each clone entity has its own node_chunks_ vector, so concurrent
  //! search threads do not race with the writer's emplace_back.
  void sync_chunks(ChunkBroker::CHUNK_TYPE type, size_t idx,
                   std::vector<Chunk::Pointer> *chunks) const {
    if (ailego_likely(idx < chunks->size())) {
      return;
    }
    for (size_t i = chunks->size(); i <= idx; ++i) {
      auto chunk = broker_->get_chunk(type, i);
      ailego_assert_with(!!chunk, "get chunk failed");
      chunks->emplace_back(std::move(chunk));
    }
  }

  inline std::pair<uint32_t, uint32_t> get_vector_chunk_loc(
      node_id_t id) const {
    uint32_t chunk_idx = id >> node_index_mask_bits_;
    uint32_t offset = (id & node_index_mask_) * node_size();
    sync_chunks(ChunkBroker::CHUNK_TYPE_NODE, chunk_idx, &node_chunks_);
    return std::make_pair(chunk_idx, offset);
  }

  inline std::pair<uint32_t, uint32_t> get_key_chunk_loc(node_id_t id) const {
    uint32_t chunk_idx = id >> node_index_mask_bits_;
    uint32_t offset = (id & node_index_mask_) * node_size() + vector_size();
    sync_chunks(ChunkBroker::CHUNK_TYPE_NODE, chunk_idx, &node_chunks_);
    return std::make_pair(chunk_idx, offset);
  }

  inline std::pair<Chunk *, size_t> get_neighbor_chunk_loc(node_id_t id) const {
    uint32_t chunk_idx = id >> node_index_mask_bits_;
    uint32_t offset =
        (id & node_index_mask_) * node_size() + vector_size() + sizeof(key_t);
    sync_chunks(ChunkBroker::CHUNK_TYPE_NODE, chunk_idx, &node_chunks_);
    ailego_assert_abort(chunk_idx < node_chunks_.size(), "invalid chunk idx");
    return std::make_pair(node_chunks_[chunk_idx].get(), offset);
  }

  // Get chunk location for neighbor distance data.
  // Uses the same chunk indexing as node chunks but with dist_entry_size_.
  inline std::pair<uint32_t, uint32_t> get_dist_chunk_loc(node_id_t id) const {
    uint32_t chunk_idx = id >> node_index_mask_bits_;
    uint32_t offset = (id & node_index_mask_) * dist_entry_size_;
    return std::make_pair(chunk_idx, offset);
  }

  void sync_dist_chunks(size_t idx) const {
    sync_chunks(ChunkBroker::CHUNK_TYPE_NEIGHBOR_DIST, idx, &dist_chunks_);
  }

  int ensure_dist_chunk_for(uint32_t chunk_index);

  int alloc_dist_chunks_for_existing_nodes();

  size_t estimate_doc_capacity() const {
    return node_chunks_.capacity() * node_cnt_per_chunk_;
  }

  int init_chunk_params(size_t max_index_size, bool huge_page) {
    node_cnt_per_chunk_ = std::max<uint32_t>(1, chunk_size_ / node_size());
    node_index_mask_bits_ = std::ceil(std::log2(node_cnt_per_chunk_));
    node_cnt_per_chunk_ = 1UL << node_index_mask_bits_;
    if (huge_page) {
      chunk_size_ = AlignHugePageSize(node_cnt_per_chunk_ * node_size());
    } else {
      chunk_size_ = AlignPageSize(node_cnt_per_chunk_ * node_size());
    }
    node_index_mask_ = node_cnt_per_chunk_ - 1;

    if (max_index_size == 0UL) {
      max_index_size_ = chunk_size_ * kDefaultMaxChunkCnt;
    } else {
      max_index_size_ = max_index_size;
    }

    size_t max_node_chunk_cnt =
        std::ceil(static_cast<double>(max_index_size_) / chunk_size_);
    node_chunks_.reserve(max_node_chunk_cnt);

    LOG_DEBUG(
        "VamanaSettings: nodeSize=%zu chunkSize=%u nodeCntPerChunk=%u "
        "maxChunkCnt=%zu maxIndexSize=%zu",
        node_size(), chunk_size_, node_cnt_per_chunk_, max_node_chunk_cnt,
        max_index_size_);

    return 0;
  }

  int init_chunks(const Chunk::Pointer &header_chunk);

  int flush_header(void) {
    if (!broker_->dirty()) {
      return 0;
    }
    auto header_chunk = broker_->get_chunk(ChunkBroker::CHUNK_TYPE_HEADER,
                                           ChunkBroker::kDefaultChunkSeqId);
    if (ailego_unlikely(!header_chunk)) {
      LOG_ERROR("get header chunk failed");
      return IndexError_Runtime;
    }
    size_t size = header_chunk->write(0UL, &header(), header_size());
    if (ailego_unlikely(size != header_size())) {
      LOG_ERROR("Write header chunk failed");
      return IndexError_WriteData;
    }
    return 0;
  }

 protected:
  IndexStreamer::Stats &stats_;
  std::mutex mutex_{};
  size_t max_index_size_{0UL};
  uint32_t chunk_size_{kDefaultChunkSize};
  uint32_t node_index_mask_bits_{0U};
  uint32_t node_cnt_per_chunk_{0U};
  uint32_t node_index_mask_{0U};
  uint32_t neighbor_size_{0U};
  bool get_vector_enabled_{false};
  bool use_key_info_map_{true};

  mutable std::shared_ptr<ailego::SharedMutex> keys_map_lock_;
  HashMapPointer<key_t, node_id_t> keys_map_;

  ChunkBroker::Pointer broker_;

  mutable std::vector<Chunk::Pointer> node_chunks_{};

 private:
  mutable std::vector<Chunk::Pointer> dist_chunks_{};
  bool dist_loaded_{false};
  uint32_t dist_entry_size_{0};  // max_degree * sizeof(dist_t)
};

// --- Template specializations for typed MemoryBlock access ---

template <>
inline NeighborsT<MmapMemoryBlock>
VamanaStreamerEntity::get_neighbors_typed<MmapMemoryBlock>(node_id_t id) const {
  uint32_t chunk_idx = id >> node_index_mask_bits_;
  uint32_t offset =
      (id & node_index_mask_) * node_size() + vector_size() + sizeof(key_t);
  sync_chunks(ChunkBroker::CHUNK_TYPE_NODE, chunk_idx, &node_chunks_);
  ailego_assert_with(chunk_idx < node_chunks_.size(), "invalid chunk idx");
  ailego_assert_with(offset < node_chunks_[chunk_idx]->data_size(),
                     "invalid chunk offset");
  const void *ptr = nullptr;
  size_t ret = node_chunks_[chunk_idx]->read(offset, &ptr, neighbor_size_);
  if (ailego_unlikely(ret != neighbor_size_)) {
    LOG_ERROR("Read neighbor header failed, ret=%zu", ret);
    return NeighborsT<MmapMemoryBlock>();
  }
  MmapMemoryBlock block(const_cast<void *>(ptr));
  return NeighborsT<MmapMemoryBlock>(std::move(block));
}

template <>
inline NeighborsT<BufferPoolMemoryBlock>
VamanaStreamerEntity::get_neighbors_typed<BufferPoolMemoryBlock>(
    node_id_t id) const {
  uint32_t chunk_idx = id >> node_index_mask_bits_;
  uint32_t offset =
      (id & node_index_mask_) * node_size() + vector_size() + sizeof(key_t);
  sync_chunks(ChunkBroker::CHUNK_TYPE_NODE, chunk_idx, &node_chunks_);
  ailego_assert_with(chunk_idx < node_chunks_.size(), "invalid chunk idx");
  IndexStorage::MemoryBlock mem_block;
  size_t ret = node_chunks_[chunk_idx]->read(offset, mem_block, neighbor_size_);
  if (ailego_unlikely(ret != neighbor_size_)) {
    LOG_ERROR("Read neighbor header failed, ret=%zu", ret);
    return NeighborsT<BufferPoolMemoryBlock>();
  }
  BufferPoolMemoryBlock block(mem_block.buffer_pool_handle_,
                              mem_block.buffer_block_id_, mem_block.data_);
  mem_block.buffer_pool_handle_ = nullptr;
  return NeighborsT<BufferPoolMemoryBlock>(std::move(block));
}

template <>
inline int VamanaStreamerEntity::get_vector_typed<MmapMemoryBlock>(
    const node_id_t *ids, uint32_t count,
    std::vector<MmapMemoryBlock> &vec_blocks) const {
  vec_blocks.resize(count);
  for (auto i = 0U; i < count; ++i) {
    auto loc = get_vector_chunk_loc(ids[i]);
    ailego_assert_with(loc.first < node_chunks_.size(), "invalid chunk idx");
    const void *ptr = nullptr;
    size_t ret = node_chunks_[loc.first]->read(loc.second, &ptr, vector_size());
    if (ailego_unlikely(ret != vector_size())) {
      LOG_ERROR("Read vector failed, ret=%zu", ret);
      return IndexError_ReadData;
    }
    vec_blocks[i].reset(const_cast<void *>(ptr));
  }
  return 0;
}

template <>
inline int VamanaStreamerEntity::get_vector_typed<BufferPoolMemoryBlock>(
    const node_id_t *ids, uint32_t count,
    std::vector<BufferPoolMemoryBlock> &vec_blocks) const {
  vec_blocks.resize(count);
  for (auto i = 0U; i < count; ++i) {
    auto loc = get_vector_chunk_loc(ids[i]);
    ailego_assert_with(loc.first < node_chunks_.size(), "invalid chunk idx");
    IndexStorage::MemoryBlock mem_block;
    size_t ret =
        node_chunks_[loc.first]->read(loc.second, mem_block, vector_size());
    if (ailego_unlikely(ret != vector_size())) {
      LOG_ERROR("Read vector failed, ret=%zu", ret);
      return IndexError_ReadData;
    }
    vec_blocks[i] =
        BufferPoolMemoryBlock(mem_block.buffer_pool_handle_,
                              mem_block.buffer_block_id_, mem_block.data_);
    mem_block.buffer_pool_handle_ = nullptr;
  }
  return 0;
}

template <>
inline key_t VamanaStreamerEntity::get_key_typed<MmapMemoryBlock>(
    node_id_t id) const {
  if (!use_key_info_map_) return id;
  auto loc = get_key_chunk_loc(id);
  ailego_assert_with(loc.first < node_chunks_.size(), "invalid chunk idx");
  const void *ptr = nullptr;
  size_t ret = node_chunks_[loc.first]->read(loc.second, &ptr, sizeof(key_t));
  if (ailego_unlikely(ret != sizeof(key_t))) {
    LOG_ERROR("Read key failed, ret=%zu", ret);
    return kInvalidKey;
  }
  return *reinterpret_cast<const key_t *>(ptr);
}

template <>
inline key_t VamanaStreamerEntity::get_key_typed<BufferPoolMemoryBlock>(
    node_id_t id) const {
  if (!use_key_info_map_) return id;
  auto loc = get_key_chunk_loc(id);
  ailego_assert_with(loc.first < node_chunks_.size(), "invalid chunk idx");
  IndexStorage::MemoryBlock key_block;
  size_t ret =
      node_chunks_[loc.first]->read(loc.second, key_block, sizeof(key_t));
  if (ailego_unlikely(ret != sizeof(key_t))) {
    LOG_ERROR("Read key failed, ret=%zu", ret);
    return kInvalidKey;
  }
  return *reinterpret_cast<const key_t *>(key_block.data());
}

// --- Typed entity subclass for mmap mode ---
// Caches chunk base addresses to eliminate virtual function calls on the
// search hot path. For mmap mode, chunk data is memory-mapped at init time,
// so we can directly compute pointers via base_addr + offset.
class VamanaMmapStreamerEntity : public VamanaStreamerEntity {
 public:
  using MemoryBlock = MmapMemoryBlock;
  using TypedNeighbors = NeighborsT<MmapMemoryBlock>;

  using VamanaStreamerEntity::VamanaStreamerEntity;

  VamanaStorageMode storage_mode() const override {
    return VamanaStorageMode::kMmap;
  }

  //! Override clone to return correct subclass type, so that
  //! static_cast<const VamanaMmapStreamerEntity&> in the algorithm is safe.
  const VamanaEntity::Pointer clone() const override;

  inline TypedNeighbors get_neighbors_typed(node_id_t id) const {
    uint32_t chunk_idx = id >> node_index_mask_bits_;
    uint32_t offset =
        (id & node_index_mask_) * node_size() + vector_size() + sizeof(key_t);
    const char *base = get_node_chunk_base(chunk_idx);
    MmapMemoryBlock block(const_cast<char *>(base + offset));
    return TypedNeighbors(std::move(block));
  }

  inline int get_vector_typed(const node_id_t *ids, uint32_t count,
                              std::vector<MmapMemoryBlock> &vec_blocks) const {
    vec_blocks.resize(count);
    for (auto i = 0U; i < count; ++i) {
      uint32_t chunk_idx = ids[i] >> node_index_mask_bits_;
      uint32_t offset = (ids[i] & node_index_mask_) * node_size();
      const char *base = get_node_chunk_base(chunk_idx);
      vec_blocks[i].reset(const_cast<char *>(base + offset));
    }
    return 0;
  }

  inline key_t get_key_typed(node_id_t id) const {
    if (!use_key_info_map_) return id;
    uint32_t chunk_idx = id >> node_index_mask_bits_;
    uint32_t offset = (id & node_index_mask_) * node_size() + vector_size();
    const char *base = get_node_chunk_base(chunk_idx);
    return *reinterpret_cast<const key_t *>(base + offset);
  }

 private:
  inline const char *get_node_chunk_base(uint32_t chunk_idx) const {
    if (ailego_unlikely(chunk_idx >= node_chunk_bases_.size())) {
      sync_node_chunk_bases(chunk_idx);
    }
    return node_chunk_bases_[chunk_idx];
  }

  void sync_node_chunk_bases(uint32_t chunk_idx) const {
    sync_node_chunks(chunk_idx);
    const auto &chunks = node_chunks_;
    for (size_t i = node_chunk_bases_.size(); i <= chunk_idx; ++i) {
      const void *ptr = nullptr;
      chunks[i]->read(0, &ptr, 1);
      node_chunk_bases_.push_back(static_cast<const char *>(ptr));
    }
  }

  mutable std::vector<const char *> node_chunk_bases_{};
};

// --- Typed entity subclass for buffer pool mode ---
class VamanaBufferPoolStreamerEntity : public VamanaStreamerEntity {
 public:
  using MemoryBlock = BufferPoolMemoryBlock;
  using TypedNeighbors = NeighborsT<BufferPoolMemoryBlock>;

  using VamanaStreamerEntity::VamanaStreamerEntity;

  VamanaStorageMode storage_mode() const override {
    return VamanaStorageMode::kBufferPool;
  }

  inline TypedNeighbors get_neighbors_typed(node_id_t id) const {
    return VamanaStreamerEntity::get_neighbors_typed<BufferPoolMemoryBlock>(id);
  }

  inline int get_vector_typed(
      const node_id_t *ids, uint32_t count,
      std::vector<BufferPoolMemoryBlock> &vec_blocks) const {
    return VamanaStreamerEntity::get_vector_typed<BufferPoolMemoryBlock>(
        ids, count, vec_blocks);
  }

  inline key_t get_key_typed(node_id_t id) const {
    return VamanaStreamerEntity::get_key_typed<BufferPoolMemoryBlock>(id);
  }
};

// --- Typed entity subclass for contiguous memory mode ---
// Allocates contiguous memory and copies all chunk data into it.
// Access is via a single base pointer + offset, eliminating chunk-level
// indirection and maximizing memory locality.
class VamanaContiguousStreamerEntity : public VamanaMmapStreamerEntity {
 public:
  using VamanaMmapStreamerEntity::VamanaMmapStreamerEntity;

  VamanaStorageMode storage_mode() const override {
    return VamanaStorageMode::kContiguous;
  }

  //! Override clone to return correct subclass type.
  //! Cloned entity shares contiguous memory via shared_ptr.
  const VamanaEntity::Pointer clone() const override;

  ~VamanaContiguousStreamerEntity() = default;

  // Build contiguous memory from chunks after open.
  int build_contiguous_memory();

  //! Degrade to mmap mode by releasing contiguous memory and falling back
  //! to chunk-based access.
  void degrade_to_mmap() {
    node_memory_.reset();
    node_base_ = nullptr;
    LOG_INFO("Vamana contiguous entity degraded to mmap mode for insertion");
  }

  bool is_contiguous() const {
    return node_base_ != nullptr;
  }

  int add_vector(key_t key, const void *vec, node_id_t *id) override {
    if (ailego_unlikely(is_contiguous())) degrade_to_mmap();
    return VamanaMmapStreamerEntity::add_vector(key, vec, id);
  }

  int add_vector_with_id(node_id_t id, const void *vec) override {
    if (ailego_unlikely(is_contiguous())) degrade_to_mmap();
    return VamanaMmapStreamerEntity::add_vector_with_id(id, vec);
  }

  inline TypedNeighbors get_neighbors_typed(node_id_t id) const {
    if (ailego_likely(node_base_ != nullptr)) {
      const char *ptr = node_base_ + static_cast<size_t>(id) * node_size() +
                        vector_size() + sizeof(key_t);
      MmapMemoryBlock block(const_cast<char *>(ptr));
      return TypedNeighbors(std::move(block));
    }
    return VamanaMmapStreamerEntity::get_neighbors_typed(id);
  }

  inline int get_vector_typed(const node_id_t *ids, uint32_t count,
                              std::vector<MmapMemoryBlock> &vec_blocks) const {
    if (ailego_likely(node_base_ != nullptr)) {
      vec_blocks.resize(count);
      for (auto i = 0U; i < count; ++i) {
        const char *ptr =
            node_base_ + static_cast<size_t>(ids[i]) * node_size();
        vec_blocks[i].reset(const_cast<char *>(ptr));
      }
      return 0;
    }
    return VamanaMmapStreamerEntity::get_vector_typed(ids, count, vec_blocks);
  }

  inline key_t get_key_typed(node_id_t id) const {
    if (ailego_likely(node_base_ != nullptr)) {
      if (!use_key_info_map_) return id;
      const char *ptr =
          node_base_ + static_cast<size_t>(id) * node_size() + vector_size();
      return *reinterpret_cast<const key_t *>(ptr);
    }
    return VamanaMmapStreamerEntity::get_key_typed(id);
  }

 protected:
  //! Custom deleter for contiguous memory (munmap / _aligned_free / free)
  struct ContiguousDeleter {
    size_t size;
    void operator()(char *ptr) const {
      if (!ptr) return;
#if defined(__linux__) || defined(__APPLE__)
      ::munmap(ptr, size);
#elif defined(_WIN32)
      ::_aligned_free(ptr);
#else
      std::free(ptr);
#endif
    }
  };

  //! Shared ownership of contiguous memory (enables zero-copy clone)
  std::shared_ptr<char> node_memory_{};

  //! Raw pointer for hot-path access (derived from shared_ptr)
  char *node_base_{nullptr};

 private:
  static char *allocate_contiguous(size_t size);
};

}  // namespace core
}  // namespace zvec
