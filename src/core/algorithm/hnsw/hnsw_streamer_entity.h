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
#include <sparsehash/dense_hash_set>
#include <zvec/ailego/container/heap.h>
#include <zvec/core/framework/index_framework.h>
#include "hnsw_chunk.h"
#include "hnsw_entity.h"
#include "hnsw_index_hash.h"
#include "hnsw_params.h"

namespace zvec {
namespace core {


//! Storage mode for HnswStreamerEntity
enum class HnswStorageMode { kMmap = 0, kBufferPool = 1, kContiguous = 2 };

//! HnswStreamerEntity manage vector data, pkey, and node's neighbors
class HnswStreamerEntity : public HnswEntity {
 public:
  //! Cleanup
  //! return 0 on success, or errCode in failure
  virtual int cleanup() override;

  //! Make a copy of streamer entity, to support thread-safe operation.
  //! The segment in container cannot be read concurrenly
  virtual const HnswEntity::Pointer clone() const override;

  //! Get primary key of the node id
  virtual key_t get_key(node_id_t id) const override;

  //! Get vector feature data by key
  virtual const void *get_vector(node_id_t id) const override;

  //! Get vectors feature data by local ids
  virtual int get_vector(const node_id_t *ids, uint32_t count,
                         const void **vecs) const override;

  virtual int get_vector(const node_id_t id,
                         IndexStorage::MemoryBlock &block) const override;

  virtual int get_vector(
      const node_id_t *ids, uint32_t count,
      std::vector<IndexStorage::MemoryBlock> &vec_blocks) const override;

  //! Get the node id's neighbors on graph level
  //! Note: the neighbors cannot be modified, using the following
  //! method to get WritableNeighbors if want to
  virtual const Neighbors get_neighbors(level_t level,
                                        node_id_t id) const override;

  //! Add vector and key to hnsw entity, and local id will be saved in id
  virtual int add_vector(level_t level, key_t key, const void *vec,
                         node_id_t *id) override;

  //! Add vector and id to hnsw entity
  virtual int add_vector_with_id(level_t level, node_id_t id,
                                 const void *vec) override;

  virtual int update_neighbors(
      level_t level, node_id_t id,
      const std::vector<std::pair<node_id_t, dist_t>> &neighbors) override;

  //! Append neighbor_id to node id neighbors on level
  //! Notice: the caller must be ensure the neighbors not full
  virtual void add_neighbor(level_t level, node_id_t id, uint32_t size,
                            node_id_t neighbor_id) override;

  //! Dump index by dumper
  virtual int dump(const IndexDumper::Pointer &dumper) override;

  virtual void update_ep_and_level(node_id_t ep, level_t level) override;

  //! Get the storage mode of this entity
  virtual HnswStorageMode storage_mode() const {
    return HnswStorageMode::kMmap;
  }

  void set_use_key_info_map(bool use_id_map) {
    use_key_info_map_ = use_id_map;
    LOG_DEBUG("use_key_info_map_: %d", (int)use_key_info_map_);
  }

 public:
  //! Constructor
  HnswStreamerEntity(IndexStreamer::Stats &stats);

  //! Destructor
  ~HnswStreamerEntity();

  //! Get vector feature data by key
  virtual const void *get_vector_by_key(key_t key) const override {
    auto id = get_id(key);
    return id == kInvalidNodeId ? nullptr : get_vector(id);
  }

  virtual int get_vector_by_key(
      const key_t key, IndexStorage::MemoryBlock &block) const override {
    auto id = get_id(key);
    if (id != kInvalidNodeId) {
      return get_vector(id, block);
    } else {
      return IndexError_InvalidArgument;
    }
  }

  //! Init entity
  int init(size_t max_doc_cnt);

  //! Flush graph entity to disk
  //! return 0 on success, or errCode in failure
  int flush(uint64_t checkpoint);

  //! Open entity from storage
  //! return 0 on success, or errCode in failure
  int open(IndexStorage::Pointer stg, uint64_t max_index_size, bool check_crc);

  //! Close entity
  //! return 0 on success, or errCode in failure
  int close();

  //! Set meta information from entity
  int set_index_meta(const IndexMeta &meta) const {
    return IndexHelper::SerializeToStorage(meta, broker_->storage().get());
  }

  //! Get meta information from entity
  int get_index_meta(IndexMeta *meta) const {
    return IndexHelper::DeserializeFromStorage(broker_->storage().get(), meta);
  }

  //! Set params: chunk size
  inline void set_chunk_size(size_t val) {
    chunk_size_ = val;
  }

  //! Set params
  inline void set_filter_same_key(bool val) {
    filter_same_key_ = val;
  }

  //! Set params
  inline void set_get_vector(bool val) {
    get_vector_enabled_ = val;
  }

  //! Get vector local id by key
  inline node_id_t get_id(key_t key) const {
    if (use_key_info_map_) {
      keys_map_lock_->lock_shared();
      auto it = keys_map_->find(key);
      keys_map_lock_->unlock_shared();
      return it == keys_map_->end() ? kInvalidNodeId : it->second;
    } else {
      return key;
    }
  }

  void print_key_map() const {
    std::cout << "key map begins" << std::endl;

    auto iter = keys_map_->begin();
    while (iter != keys_map_->end()) {
      std::cout << "key: " << iter->first << ", id: " << iter->second
                << std::endl;
      ;
      iter++;
    }

    std::cout << "key map ends" << std::endl;
  }

  //! Typed get_neighbors: returns NeighborsT<MemBlock> without runtime
  //! branching on MemoryBlock type. MmapMemoryBlock specialization uses
  //! pointer-based read; BufferPoolMemoryBlock uses MemoryBlock-based read.
  template <typename MemBlock>
  inline NeighborsT<MemBlock> get_neighbors_typed(level_t level,
                                                  node_id_t id) const;

  //! Typed batch get_vector: fills vector<MemBlock> without runtime branching
  template <typename MemBlock>
  inline int get_vector_typed(const node_id_t *ids, uint32_t count,
                              std::vector<MemBlock> &vec_blocks) const;

  //! Typed get_key: reads key using typed MemBlock
  template <typename MemBlock>
  inline key_t get_key_typed(node_id_t id) const;

  //! Get l0 neighbors size
  inline size_t neighbors_size() const {
    return sizeof(NeighborsHeader) + l0_neighbor_cnt() * sizeof(node_id_t);
  }

  //! Get neighbors size for level > 0
  inline size_t upper_neighbors_size() const {
    return sizeof(NeighborsHeader) + upper_neighbor_cnt() * sizeof(node_id_t);
  }


 protected:
  union UpperNeighborIndexMeta {
    struct {
      uint32_t level : 4;
      uint32_t index : 28;  // index is composite type: chunk idx, and the
                            // N th neighbors in chunk, they two composite
                            // the 28 bits location
    };
    uint32_t data;
  };

 protected:
  template <class Key, class T>
  using HashMap = google::dense_hash_map<Key, T, std::hash<Key>>;
  template <class Key, class T>
  using HashMapPointer = std::shared_ptr<HashMap<Key, T>>;

  template <class Key>
  using HashSet = google::dense_hash_set<Key, std::hash<Key>>;
  template <class Key>
  using HashSetPointer = std::shared_ptr<HashSet<Key>>;

  //! upper neighbor index hashmap
  using NIHashMap = HnswIndexHashMap<node_id_t, uint32_t>;
  using NIHashMapPointer = std::shared_ptr<NIHashMap>;

  //! Clone construct, used by clone method in subclasses
  HnswStreamerEntity(IndexStreamer::Stats &stats, const HNSWHeader &hd,
                     size_t chunk_size, uint32_t node_index_mask_bits,
                     uint32_t upper_neighbor_mask_bits, bool filter_same_key,
                     bool get_vector_enabled,
                     const NIHashMapPointer &upper_neighbor_index,
                     std::shared_ptr<ailego::SharedMutex> &keys_map_lock,
                     const HashMapPointer<key_t, node_id_t> &keys_map,
                     bool use_key_info_map,
                     std::vector<Chunk::Pointer> &&node_chunks,
                     std::vector<Chunk::Pointer> &&upper_neighbor_chunks,
                     const ChunkBroker::Pointer &broker,
                     std::shared_ptr<std::vector<const uint8_t *>> node_bases,
                     std::shared_ptr<std::vector<const uint8_t *>> upper_bases)
      : stats_(stats),
        chunk_size_(chunk_size),
        node_index_mask_bits_(node_index_mask_bits),
        node_cnt_per_chunk_(1UL << node_index_mask_bits_),
        node_index_mask_(node_cnt_per_chunk_ - 1),
        upper_neighbor_mask_bits_(upper_neighbor_mask_bits),
        upper_neighbor_mask_((1U << upper_neighbor_mask_bits_) - 1),
        filter_same_key_(filter_same_key),
        get_vector_enabled_(get_vector_enabled),
        use_key_info_map_(use_key_info_map),
        upper_neighbor_index_(upper_neighbor_index),
        keys_map_lock_(keys_map_lock),
        keys_map_(keys_map),
        node_chunks_(std::move(node_chunks)),
        upper_neighbor_chunks_(std::move(upper_neighbor_chunks)),
        broker_(broker) {
    *mutable_header() = hd;

    neighbor_size_ = neighbors_size();
    upper_neighbor_size_ = upper_neighbors_size();

    // Reuse the shared base-pointer arrays created by init_chunks().
    // All clones share the same arrays so hot HNSW hub-node chunks are
    // collectively promoted to L1/L2 by every search thread instead of
    // each clone warming its own private copy in L3.
    node_chunk_bases_ = std::move(node_bases);
    upper_neighbor_chunk_bases_ = std::move(upper_bases);
  }

  //! Called only in searching procedure per context, so no need to lock
  void sync_chunks(ChunkBroker::CHUNK_TYPE type, size_t idx,
                   std::vector<Chunk::Pointer> *chunks) const {
    if (ailego_likely(idx < chunks->size())) {
      return;
    }
    for (size_t i = chunks->size(); i <= idx; ++i) {
      auto chunk = broker_->get_chunk(type, i);
      // the storage can ensure get chunk will success after the first get
      ailego_assert_with(!!chunk, "get chunk failed");
      chunks->emplace_back(std::move(chunk));
    }
  }

  //! return pair: chunk index + chunk offset
  inline std::pair<uint32_t, uint32_t> get_vector_chunk_loc(
      node_id_t id) const {
    uint32_t chunk_idx = id >> node_index_mask_bits_;
    uint32_t offset = (id & node_index_mask_) * node_size();

    sync_chunks(ChunkBroker::CHUNK_TYPE_NODE, chunk_idx, &node_chunks_);
    return std::make_pair(chunk_idx, offset);
  }

  //! return pair: chunk index + chunk offset
  inline std::pair<uint32_t, uint32_t> get_key_chunk_loc(node_id_t id) const {
    uint32_t chunk_idx = id >> node_index_mask_bits_;
    uint32_t offset = (id & node_index_mask_) * node_size() + vector_size();

    sync_chunks(ChunkBroker::CHUNK_TYPE_NODE, chunk_idx, &node_chunks_);
    return std::make_pair(chunk_idx, offset);
  }

  inline std::pair<uint32_t, uint32_t> get_upper_neighbor_chunk_loc(
      level_t level, node_id_t id) const {
    auto it = upper_neighbor_index_->find(id);
    ailego_assert_abort(it != upper_neighbor_index_->end(),
                        "Get upper neighbor header failed");
    auto meta = reinterpret_cast<const UpperNeighborIndexMeta *>(&it->second);
    uint32_t chunk_idx = (meta->index) >> upper_neighbor_mask_bits_;
    uint32_t offset = (((meta->index) & upper_neighbor_mask_) + level - 1) *
                      upper_neighbor_size_;
    sync_chunks(ChunkBroker::CHUNK_TYPE_UPPER_NEIGHBOR, chunk_idx,
                &upper_neighbor_chunks_);
    ailego_assert_abort(chunk_idx < upper_neighbor_chunks_.size(),
                        "invalid chunk idx");
    ailego_assert_abort(offset < upper_neighbor_chunks_[chunk_idx]->data_size(),
                        "invalid chunk offset");
    return std::make_pair(chunk_idx, offset);
  }

  //! return pair: chunk + chunk offset
  inline std::pair<Chunk *, size_t> get_neighbor_chunk_loc(level_t level,
                                                           node_id_t id) const {
    if (level == 0UL) {
      uint32_t chunk_idx = id >> node_index_mask_bits_;
      uint32_t offset =
          (id & node_index_mask_) * node_size() + vector_size() + sizeof(key_t);

      sync_chunks(ChunkBroker::CHUNK_TYPE_NODE, chunk_idx, &node_chunks_);
      ailego_assert_abort(chunk_idx < node_chunks_.size(), "invalid chunk idx");
      ailego_assert_abort(offset < node_chunks_[chunk_idx]->data_size(),
                          "invalid chunk offset");
      return std::make_pair(node_chunks_[chunk_idx].get(), offset);
    } else {
      auto p = get_upper_neighbor_chunk_loc(level, id);
      return std::make_pair(upper_neighbor_chunks_[p.first].get(), p.second);
    }
  }

  //! Chunk hnsw index valid
  int check_hnsw_index(const HNSWHeader *hd) const;

  size_t get_total_upper_neighbors_size(level_t level) const {
    return level * upper_neighbor_size_;
  }

  //! Add upper neighbor header and reserve space for upper neighbor
  int add_upper_neighbor(level_t level, node_id_t id) {
    if (level == 0) {
      return 0;
    }
    Chunk::Pointer chunk;
    uint64_t chunk_offset = UINT64_MAX;
    size_t neighbors_size = get_total_upper_neighbors_size(level);
    uint64_t chunk_index = upper_neighbor_chunks_.size() - 1ULL;
    if (chunk_index == UINT64_MAX ||
        (upper_neighbor_chunks_[chunk_index]->padding_size() <
         neighbors_size)) {  // no space left and need to alloc
      chunk_index++;
      if (ailego_unlikely(upper_neighbor_chunks_.capacity() ==
                          upper_neighbor_chunks_.size())) {
        LOG_ERROR("add upper neighbor failed for no memory quota");
        return IndexError_IndexFull;
      }
      auto p = broker_->alloc_chunk(ChunkBroker::CHUNK_TYPE_UPPER_NEIGHBOR,
                                    chunk_index, upper_neighbor_chunk_size_);
      if (ailego_unlikely(p.first != 0)) {
        LOG_ERROR("Alloc data chunk failed");
        return p.first;
      }
      chunk = p.second;
      chunk_offset = 0UL;
      upper_neighbor_chunks_.emplace_back(chunk);
    } else {
      chunk = upper_neighbor_chunks_[chunk_index];
      chunk_offset = chunk->data_size();
    }
    ailego_assert_with((size_t)level < kMaxGraphLayers, "invalid level");
    ailego_assert_with(chunk_offset % upper_neighbor_size_ == 0,
                       "invalid offset");
    ailego_assert_with((chunk_offset / upper_neighbor_size_) <
                           (1U << upper_neighbor_mask_bits_),
                       "invalid offset");
    ailego_assert_with(chunk_index < (1U << (28 - upper_neighbor_mask_bits_)),
                       "invalid chunk index");
    UpperNeighborIndexMeta meta;
    meta.level = level;
    meta.index = (chunk_index << upper_neighbor_mask_bits_) |
                 (chunk_offset / upper_neighbor_size_);
    chunk_offset += upper_neighbor_size_ * level;
    if (ailego_unlikely(!upper_neighbor_index_->insert(id, meta.data))) {
      LOG_ERROR("HashMap insert value failed");
      return IndexError_Runtime;
    }

    if (ailego_unlikely(chunk->resize(chunk_offset) != chunk_offset)) {
      LOG_ERROR("Chunk resize to %zu failed", (size_t)chunk_offset);
      return IndexError_Runtime;
    }

    return 0;
  }

  size_t estimate_doc_capacity() const {
    return node_chunks_.capacity() * node_cnt_per_chunk_;
  }

  int init_chunk_params(size_t max_index_size, bool huge_page) {
    node_cnt_per_chunk_ = std::max<uint32_t>(1, chunk_size_ / node_size());
    //! align node cnt per chunk to pow of 2
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

    //! To get a balanced upper neighbor chunk size.
    //! If the upper chunk size is equal to node chunk size, it may waste
    //! upper neighbor chunk space; if the upper neighbor chunk size is too
    //! small, the will need large upper neighbor chunks index space. So to
    //! get a balanced ratio be sqrt of the node/neighbor size ratio
    float ratio =
        std::sqrt(node_size() * scaling_factor() * 1.0f / upper_neighbor_size_);
    if (huge_page) {
      upper_neighbor_chunk_size_ = AlignHugePageSize(
          std::max(get_total_upper_neighbors_size(kMaxGraphLayers),
                   static_cast<size_t>(chunk_size_ / ratio)));
    } else {
      upper_neighbor_chunk_size_ = AlignPageSize(
          std::max(get_total_upper_neighbors_size(kMaxGraphLayers),
                   static_cast<size_t>(chunk_size_ / ratio)));
    }
    upper_neighbor_mask_bits_ =
        std::ceil(std::log2(upper_neighbor_chunk_size_ / upper_neighbor_size_));
    upper_neighbor_mask_ = (1 << upper_neighbor_mask_bits_) - 1;

    size_t max_node_chunk_cnt = std::ceil(max_index_size_ / chunk_size_);
    size_t max_upper_chunk_cnt = std::ceil(
        (max_node_chunk_cnt * node_cnt_per_chunk_ * 1.0f / scaling_factor()) /
        (upper_neighbor_chunk_size_ / upper_neighbor_size_));
    max_upper_chunk_cnt =
        max_upper_chunk_cnt + std::ceil(max_upper_chunk_cnt / scaling_factor());

    //! reserve space to avoid memmove in chunks vector emplace chunk, so
    //! as to lock-free in reading chunk
    node_chunks_.reserve(max_node_chunk_cnt);
    upper_neighbor_chunks_.reserve(max_upper_chunk_cnt);

    LOG_DEBUG(
        "Settings: nodeSize=%zu chunkSize=%u upperNeighborSize=%u "
        "upperNeighborChunkSize=%u "
        "nodeCntPerChunk=%u maxChunkCnt=%zu maxNeighborChunkCnt=%zu "
        "maxIndexSize=%zu ratio=%.3f",
        node_size(), chunk_size_, upper_neighbor_size_,
        upper_neighbor_chunk_size_, node_cnt_per_chunk_, max_node_chunk_cnt,
        max_upper_chunk_cnt, max_index_size_, ratio);

    return 0;
  }

  //! Init node chunk and neighbor chunks
  int init_chunks(const Chunk::Pointer &header_chunk);

  int flush_header(void) {
    if (!broker_->dirty()) {
      // do not need to flush
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
  //! Expose sync_chunks for subclass use
  inline void sync_node_chunks(size_t idx) const {
    sync_chunks(ChunkBroker::CHUNK_TYPE_NODE, idx, &node_chunks_);
  }
  inline void sync_upper_neighbor_chunks(size_t idx) const {
    sync_chunks(ChunkBroker::CHUNK_TYPE_UPPER_NEIGHBOR, idx,
                &upper_neighbor_chunks_);
  }

 private:
  HnswStreamerEntity(const HnswStreamerEntity &) = delete;
  HnswStreamerEntity &operator=(const HnswStreamerEntity &) = delete;
  static constexpr uint64_t kUpperHashMemoryInflateRatio = 2.0f;

 protected:
  IndexStreamer::Stats &stats_;
  std::mutex mutex_{};
  size_t max_index_size_{0UL};
  uint32_t chunk_size_{kDefaultChunkSize};
  uint32_t upper_neighbor_chunk_size_{kDefaultChunkSize};
  uint32_t node_index_mask_bits_{0U};
  uint32_t node_cnt_per_chunk_{0U};
  uint32_t node_index_mask_{0U};
  uint32_t neighbor_size_{0U};
  uint32_t upper_neighbor_size_{0U};
  //! UpperNeighborIndex.index composite chunkIdx and offset in chunk by the
  //! following mask
  uint32_t upper_neighbor_mask_bits_{0U};
  uint32_t upper_neighbor_mask_{0U};
  bool filter_same_key_{false};
  bool get_vector_enabled_{false};
  bool use_key_info_map_{true};

  NIHashMapPointer upper_neighbor_index_{};

  mutable std::shared_ptr<ailego::SharedMutex> keys_map_lock_{};
  HashMapPointer<key_t, node_id_t> keys_map_{};

  //! the chunks will be changed in searcher, so need mutable
  //! data chunk include: vector, key, level 0 neighbors
  mutable std::vector<Chunk::Pointer> node_chunks_{};

  //! Flat cache of base_data() pointers for node_chunks_ and
  //! upper_neighbor_chunks_. Non-empty only when the storage backend
  //! returns a stable mmap pointer (base_data() != nullptr). Avoids
  //! following the full shared_ptr -> Segment -> IndexMapping::Segment
  //! pointer chain on every get_vector() / get_neighbors() call, which
  //! is critical for small chunk sizes (e.g. 16 K) where node_chunks_
  //! can hold 100K+ entries and the metadata no longer fits in L2 cache.
  //!
  //! Shared across all clones (read-only after open) so that hot entries
  //! (hub-node chunks near the HNSW entry point) are promoted to L1/L2
  //! by all search threads collectively, instead of each clone warming
  //! its own private 250 KB copy in L3.
  mutable std::shared_ptr<std::vector<const uint8_t *>> node_chunk_bases_{};

  //! upper neighbor chunk inlude: UpperNeighborHeader + (1~level) neighbors
  mutable std::vector<Chunk::Pointer> upper_neighbor_chunks_{};
  mutable std::shared_ptr<std::vector<const uint8_t *>>
      upper_neighbor_chunk_bases_{};

  ChunkBroker::Pointer broker_{};  // chunk broker
};

// --- Template specializations for typed MemoryBlock access ---

//! MmapMemoryBlock specialization: uses pointer-based Chunk::read
template <>
inline NeighborsT<MmapMemoryBlock>
HnswStreamerEntity::get_neighbors_typed<MmapMemoryBlock>(level_t level,
                                                         node_id_t id) const {
  Chunk *chunk = nullptr;
  size_t offset = 0UL;
  size_t nbr_size = neighbor_size_;
  if (level == 0UL) {
    uint32_t chunk_idx = id >> node_index_mask_bits_;
    offset =
        (id & node_index_mask_) * node_size() + vector_size() + sizeof(key_t);
    sync_chunks(ChunkBroker::CHUNK_TYPE_NODE, chunk_idx, &node_chunks_);
    ailego_assert_with(chunk_idx < node_chunks_.size(), "invalid chunk idx");
    chunk = node_chunks_[chunk_idx].get();
  } else {
    auto p = get_upper_neighbor_chunk_loc(level, id);
    chunk = upper_neighbor_chunks_[p.first].get();
    offset = p.second;
    nbr_size = upper_neighbor_size_;
  }
  ailego_assert_with(offset < chunk->data_size(), "invalid chunk offset");
  const void *ptr = nullptr;
  size_t ret = chunk->read(offset, &ptr, nbr_size);
  if (ailego_unlikely(ret != nbr_size)) {
    LOG_ERROR("Read neighbor header failed, ret=%zu", ret);
    return NeighborsT<MmapMemoryBlock>();
  }
  MmapMemoryBlock block(const_cast<void *>(ptr));
  return NeighborsT<MmapMemoryBlock>(std::move(block));
}

//! BufferPoolMemoryBlock specialization: uses MemoryBlock-based Chunk::read
template <>
inline NeighborsT<BufferPoolMemoryBlock>
HnswStreamerEntity::get_neighbors_typed<BufferPoolMemoryBlock>(
    level_t level, node_id_t id) const {
  Chunk *chunk = nullptr;
  size_t offset = 0UL;
  size_t nbr_size = neighbor_size_;
  if (level == 0UL) {
    uint32_t chunk_idx = id >> node_index_mask_bits_;
    offset =
        (id & node_index_mask_) * node_size() + vector_size() + sizeof(key_t);
    sync_chunks(ChunkBroker::CHUNK_TYPE_NODE, chunk_idx, &node_chunks_);
    ailego_assert_with(chunk_idx < node_chunks_.size(), "invalid chunk idx");
    chunk = node_chunks_[chunk_idx].get();
  } else {
    auto p = get_upper_neighbor_chunk_loc(level, id);
    chunk = upper_neighbor_chunks_[p.first].get();
    offset = p.second;
    nbr_size = upper_neighbor_size_;
  }
  ailego_assert_with(offset < chunk->data_size(), "invalid chunk offset");
  IndexStorage::MemoryBlock mem_block;
  size_t ret = chunk->read(offset, mem_block, nbr_size);
  if (ailego_unlikely(ret != nbr_size)) {
    LOG_ERROR("Read neighbor header failed, ret=%zu", ret);
    return NeighborsT<BufferPoolMemoryBlock>();
  }
  BufferPoolMemoryBlock block(mem_block.buffer_pool_handle_,
                              mem_block.buffer_block_id_, mem_block.data_);
  mem_block.buffer_pool_handle_ = nullptr;
  return NeighborsT<BufferPoolMemoryBlock>(std::move(block));
}

//! MmapMemoryBlock specialization for batch get_vector
template <>
inline int HnswStreamerEntity::get_vector_typed<MmapMemoryBlock>(
    const node_id_t *ids, uint32_t count,
    std::vector<MmapMemoryBlock> &vec_blocks) const {
  vec_blocks.resize(count);
  for (auto i = 0U; i < count; ++i) {
    auto loc = get_vector_chunk_loc(ids[i]);
    ailego_assert_with(loc.first < node_chunks_.size(), "invalid chunk idx");
    ailego_assert_with(loc.second < node_chunks_[loc.first]->data_size(),
                       "invalid chunk offset");
    size_t read_size = vector_size();
    const void *ptr = nullptr;
    size_t ret = node_chunks_[loc.first]->read(loc.second, &ptr, read_size);
    if (ailego_unlikely(ret != read_size)) {
      LOG_ERROR("Read vector failed, offset=%u, read size=%zu, ret=%zu",
                loc.second, read_size, ret);
      return IndexError_ReadData;
    }
    vec_blocks[i].reset(const_cast<void *>(ptr));
  }
  return 0;
}

//! BufferPoolMemoryBlock specialization for batch get_vector
template <>
inline int HnswStreamerEntity::get_vector_typed<BufferPoolMemoryBlock>(
    const node_id_t *ids, uint32_t count,
    std::vector<BufferPoolMemoryBlock> &vec_blocks) const {
  vec_blocks.resize(count);
  for (auto i = 0U; i < count; ++i) {
    auto loc = get_vector_chunk_loc(ids[i]);
    ailego_assert_with(loc.first < node_chunks_.size(), "invalid chunk idx");
    ailego_assert_with(loc.second < node_chunks_[loc.first]->data_size(),
                       "invalid chunk offset");
    size_t read_size = vector_size();
    IndexStorage::MemoryBlock mem_block;
    size_t ret =
        node_chunks_[loc.first]->read(loc.second, mem_block, read_size);
    if (ailego_unlikely(ret != read_size)) {
      LOG_ERROR("Read vector failed, offset=%u, read size=%zu, ret=%zu",
                loc.second, read_size, ret);
      return IndexError_ReadData;
    }
    vec_blocks[i] =
        BufferPoolMemoryBlock(mem_block.buffer_pool_handle_,
                              mem_block.buffer_block_id_, mem_block.data_);
    mem_block.buffer_pool_handle_ = nullptr;
  }
  return 0;
}

//! MmapMemoryBlock specialization for get_key
template <>
inline key_t HnswStreamerEntity::get_key_typed<MmapMemoryBlock>(
    node_id_t id) const {
  if (!use_key_info_map_) {
    return id;
  }
  auto loc = get_key_chunk_loc(id);
  ailego_assert_with(loc.first < node_chunks_.size(), "invalid chunk idx");
  ailego_assert_with(loc.second < node_chunks_[loc.first]->data_size(),
                     "invalid chunk offset");
  const void *ptr = nullptr;
  size_t ret = node_chunks_[loc.first]->read(loc.second, &ptr, sizeof(key_t));
  if (ailego_unlikely(ret != sizeof(key_t))) {
    LOG_ERROR("Read key failed, ret=%zu", ret);
    return kInvalidKey;
  }
  return *reinterpret_cast<const key_t *>(ptr);
}

//! BufferPoolMemoryBlock specialization for get_key
template <>
inline key_t HnswStreamerEntity::get_key_typed<BufferPoolMemoryBlock>(
    node_id_t id) const {
  if (!use_key_info_map_) {
    return id;
  }
  auto loc = get_key_chunk_loc(id);
  ailego_assert_with(loc.first < node_chunks_.size(), "invalid chunk idx");
  ailego_assert_with(loc.second < node_chunks_[loc.first]->data_size(),
                     "invalid chunk offset");
  IndexStorage::MemoryBlock key_block;
  size_t ret =
      node_chunks_[loc.first]->read(loc.second, key_block, sizeof(key_t));
  if (ailego_unlikely(ret != sizeof(key_t))) {
    LOG_ERROR("Read key failed, ret=%zu", ret);
    return kInvalidKey;
  }
  return *reinterpret_cast<const key_t *>(key_block.data());
}

//! Typed entity subclass for mmap mode.
//! Caches chunk base addresses to eliminate virtual function calls on the
//! search hot path. For mmap mode, chunk data is memory-mapped at init time,
//! so we can directly compute pointers via base_addr + offset.
class HnswMmapStreamerEntity : public HnswStreamerEntity {
 public:
  using MemoryBlock = MmapMemoryBlock;
  using TypedNeighbors = NeighborsT<MmapMemoryBlock>;

  using HnswStreamerEntity::HnswStreamerEntity;

  HnswStorageMode storage_mode() const override {
    return HnswStorageMode::kMmap;
  }

  //! Override clone to return correct subclass type, so that
  //! static_cast<const HnswMmapStreamerEntity&> in the algorithm is safe.
  const HnswEntity::Pointer clone() const override;

  inline TypedNeighbors get_neighbors_typed(level_t level, node_id_t id) const {
    if (level == 0UL) {
      uint32_t chunk_idx = id >> node_index_mask_bits_;
      uint32_t offset =
          (id & node_index_mask_) * node_size() + vector_size() + sizeof(key_t);
      const char *base = get_node_chunk_base(chunk_idx);
      MmapMemoryBlock block(const_cast<char *>(base + offset));
      return TypedNeighbors(std::move(block));
    }
    // Upper level: use index to locate chunk and offset
    auto it = upper_neighbor_index_->find(id);
    ailego_assert_abort(it != upper_neighbor_index_->end(),
                        "Get upper neighbor header failed");
    auto meta = reinterpret_cast<const UpperNeighborIndexMeta *>(&it->second);
    uint32_t chunk_idx = (meta->index) >> upper_neighbor_mask_bits_;
    uint32_t offset = (((meta->index) & upper_neighbor_mask_) + level - 1) *
                      upper_neighbor_size_;
    const char *base = get_upper_neighbor_chunk_base(chunk_idx);
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
    if (!use_key_info_map_) {
      return id;
    }
    uint32_t chunk_idx = id >> node_index_mask_bits_;
    uint32_t offset = (id & node_index_mask_) * node_size() + vector_size();
    const char *base = get_node_chunk_base(chunk_idx);
    return *reinterpret_cast<const key_t *>(base + offset);
  }

 private:
  //! Get cached base address for a node chunk, syncing if needed
  inline const char *get_node_chunk_base(uint32_t chunk_idx) const {
    if (ailego_unlikely(chunk_idx >= node_chunk_bases_.size())) {
      sync_node_chunk_bases(chunk_idx);
    }
    return node_chunk_bases_[chunk_idx];
  }

  //! Get cached base address for an upper neighbor chunk, syncing if needed
  inline const char *get_upper_neighbor_chunk_base(uint32_t chunk_idx) const {
    if (ailego_unlikely(chunk_idx >= upper_neighbor_chunk_bases_.size())) {
      sync_upper_neighbor_chunk_bases(chunk_idx);
    }
    return upper_neighbor_chunk_bases_[chunk_idx];
  }

  //! Sync node chunk base addresses up to the given index
  void sync_node_chunk_bases(uint32_t chunk_idx) const {
    sync_node_chunks(chunk_idx);
    const auto &chunks = node_chunks_;
    for (size_t i = node_chunk_bases_.size(); i <= chunk_idx; ++i) {
      const void *ptr = nullptr;
      chunks[i]->read(0, &ptr, 1);
      node_chunk_bases_.push_back(static_cast<const char *>(ptr));
    }
  }

  //! Sync upper neighbor chunk base addresses up to the given index
  void sync_upper_neighbor_chunk_bases(uint32_t chunk_idx) const {
    sync_upper_neighbor_chunks(chunk_idx);
    const auto &chunks = upper_neighbor_chunks_;
    for (size_t i = upper_neighbor_chunk_bases_.size(); i <= chunk_idx; ++i) {
      const void *ptr = nullptr;
      chunks[i]->read(0, &ptr, 1);
      upper_neighbor_chunk_bases_.push_back(static_cast<const char *>(ptr));
    }
  }

  mutable std::vector<const char *> node_chunk_bases_{};
  mutable std::vector<const char *> upper_neighbor_chunk_bases_{};
};

//! Typed entity subclass for buffer pool mode.
class HnswBufferPoolStreamerEntity : public HnswStreamerEntity {
 public:
  using MemoryBlock = BufferPoolMemoryBlock;
  using TypedNeighbors = NeighborsT<BufferPoolMemoryBlock>;

  using HnswStreamerEntity::HnswStreamerEntity;

  HnswStorageMode storage_mode() const override {
    return HnswStorageMode::kBufferPool;
  }

  inline TypedNeighbors get_neighbors_typed(level_t level, node_id_t id) const {
    return HnswStreamerEntity::get_neighbors_typed<BufferPoolMemoryBlock>(level,
                                                                          id);
  }

  inline int get_vector_typed(
      const node_id_t *ids, uint32_t count,
      std::vector<BufferPoolMemoryBlock> &vec_blocks) const {
    return HnswStreamerEntity::get_vector_typed<BufferPoolMemoryBlock>(
        ids, count, vec_blocks);
  }

  inline key_t get_key_typed(node_id_t id) const {
    return HnswStreamerEntity::get_key_typed<BufferPoolMemoryBlock>(id);
  }
};

//! Typed entity subclass for contiguous memory mode.
//! Allocates contiguous memory (with hugepage/THP support) and copies all
//! chunk data into it. Access is via a single base pointer + offset,
//! eliminating chunk-level indirection and maximizing memory locality.
class HnswContiguousStreamerEntity : public HnswMmapStreamerEntity {
 public:
  using HnswMmapStreamerEntity::HnswMmapStreamerEntity;

  HnswStorageMode storage_mode() const override {
    return HnswStorageMode::kContiguous;
  }

  //! Override clone to return correct subclass type.
  //! Cloned entity shares contiguous memory via shared_ptr.
  const HnswEntity::Pointer clone() const override;

  ~HnswContiguousStreamerEntity() = default;

  //! Build contiguous memory from chunks after open.
  //! Must be called after the entity is fully opened and all chunks are loaded.
  int build_contiguous_memory();

  //! Degrade to mmap mode by releasing contiguous memory and falling back
  //! to chunk-based access.
  void degrade_to_mmap() {
    node_memory_.reset();
    node_base_ = nullptr;
    upper_neighbor_memory_.reset();
    upper_neighbor_base_ = nullptr;
    upper_chunk_offsets_.clear();
    LOG_INFO("HNSW contiguous entity degraded to mmap mode for insertion");
  }

  bool is_contiguous() const {
    return node_base_ != nullptr;
  }

  int add_vector(level_t level, key_t key, const void *vec,
                 node_id_t *id) override {
    if (ailego_unlikely(is_contiguous())) degrade_to_mmap();
    return HnswMmapStreamerEntity::add_vector(level, key, vec, id);
  }

  int add_vector_with_id(level_t level, node_id_t id,
                         const void *vec) override {
    if (ailego_unlikely(is_contiguous())) degrade_to_mmap();
    return HnswMmapStreamerEntity::add_vector_with_id(level, id, vec);
  }

  inline TypedNeighbors get_neighbors_typed(level_t level, node_id_t id) const {
    if (ailego_likely(node_base_ != nullptr)) {
      if (level == 0UL) {
        const char *ptr = node_base_ + static_cast<size_t>(id) * node_size() +
                          vector_size() + sizeof(key_t);
        MmapMemoryBlock block(const_cast<char *>(ptr));
        return TypedNeighbors(std::move(block));
      }
      // Upper level: use index to locate global offset
      auto it = upper_neighbor_index_->find(id);
      ailego_assert_abort(it != upper_neighbor_index_->end(),
                          "Get upper neighbor header failed");
      auto meta = reinterpret_cast<const UpperNeighborIndexMeta *>(&it->second);
      uint32_t chunk_idx = (meta->index) >> upper_neighbor_mask_bits_;
      uint32_t local_idx = (meta->index) & upper_neighbor_mask_;
      size_t global_offset =
          upper_chunk_offsets_[chunk_idx] +
          static_cast<size_t>(local_idx + level - 1) * upper_neighbor_size_;
      const char *ptr = upper_neighbor_base_ + global_offset;
      MmapMemoryBlock block(const_cast<char *>(ptr));
      return TypedNeighbors(std::move(block));
    }
    return HnswMmapStreamerEntity::get_neighbors_typed(level, id);
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
    return HnswMmapStreamerEntity::get_vector_typed(ids, count, vec_blocks);
  }

  inline key_t get_key_typed(node_id_t id) const {
    if (ailego_likely(node_base_ != nullptr)) {
      if (!use_key_info_map_) {
        return id;
      }
      const char *ptr =
          node_base_ + static_cast<size_t>(id) * node_size() + vector_size();
      return *reinterpret_cast<const key_t *>(ptr);
    }
    return HnswMmapStreamerEntity::get_key_typed(id);
  }

 protected:
  //! Custom deleter for contiguous memory (munmap / _aligned_free / free)
  //! Used by shared_ptr to properly release mmap'd memory.
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
  std::shared_ptr<char> upper_neighbor_memory_{};

  //! Raw pointers for hot-path access (derived from shared_ptr)
  char *node_base_{nullptr};
  char *upper_neighbor_base_{nullptr};

  //! Cumulative offsets for each upper neighbor chunk in contiguous memory
  std::vector<size_t> upper_chunk_offsets_{};

 private:
  //! Allocate contiguous memory with hugepage/THP support
  static char *allocate_contiguous(size_t size);
};

}  // namespace core
}  // namespace zvec