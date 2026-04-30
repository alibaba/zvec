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

#include "hnsw_streamer_entity.h"
#if defined(__linux__) || defined(__APPLE__)
#include <sys/mman.h>
#endif
#include <ailego/utility/memory_helper.h>

// #define DEBUG_PRINT

namespace zvec {
namespace core {

HnswStreamerEntity::HnswStreamerEntity(IndexStreamer::Stats &stats)
    : stats_(stats) {}

HnswStreamerEntity::~HnswStreamerEntity() {}

int HnswStreamerEntity::init(size_t max_doc_cnt) {
  if (std::pow(scaling_factor(), kMaxGraphLayers) < max_doc_cnt) {
    LOG_ERROR("scalingFactor=%zu is too small", scaling_factor());
    return IndexError_InvalidArgument;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  broker_ = std::make_shared<ChunkBroker>(stats_);
  upper_neighbor_index_ = std::make_shared<NIHashMap>();
  keys_map_lock_ = std::make_shared<ailego::SharedMutex>();
  keys_map_ = std::make_shared<HashMap<key_t, node_id_t>>();
  if (!keys_map_ || !upper_neighbor_index_ || !broker_ || !keys_map_lock_) {
    LOG_ERROR("HnswStreamerEntity new object failed");
    return IndexError_NoMemory;
  }
  keys_map_->set_empty_key(kInvalidKey);

  neighbor_size_ = neighbors_size();
  upper_neighbor_size_ = upper_neighbors_size();

  //! vector + key + level 0 neighbors
  size_t size = vector_size() + sizeof(key_t) + neighbor_size_;

  size = AlignSize(size);
  set_node_size(size);
  return 0;
}

int HnswStreamerEntity::cleanup() {
  std::lock_guard<std::mutex> lock(mutex_);
  mutable_header()->clear();
  chunk_size_ = kDefaultChunkSize;
  node_index_mask_bits_ = 0U;
  node_index_mask_ = 0U;
  node_cnt_per_chunk_ = 0U;
  neighbor_size_ = 0U;
  upper_neighbor_size_ = 0U;
  if (upper_neighbor_index_) {
    upper_neighbor_index_->cleanup();
  }
  if (keys_map_) {
    keys_map_->clear();
  }
  node_chunks_.clear();
  node_chunk_bases_.reset();
  upper_neighbor_chunks_.clear();
  upper_neighbor_chunk_bases_.reset();
  filter_same_key_ = false;
  get_vector_enabled_ = false;
  broker_.reset();

  return 0;
}

int HnswStreamerEntity::update_neighbors(
    level_t level, node_id_t id,
    const std::vector<std::pair<node_id_t, dist_t>> &neighbors) {
  std::vector<char> buffer(neighbor_size_);
  NeighborsHeader *hd = reinterpret_cast<NeighborsHeader *>(buffer.data());
  hd->neighbor_cnt = neighbors.size();
  size_t i = 0;
  for (; i < neighbors.size(); ++i) {
    hd->neighbors[i] = neighbors[i].first;
  }

  auto loc = get_neighbor_chunk_loc(level, id);
  size_t size = reinterpret_cast<char *>(&hd->neighbors[i]) - buffer.data();
  size_t ret = loc.first->write(loc.second, hd, size);
  if (ailego_unlikely(ret != size)) {
    LOG_ERROR("Write neighbor header failed, ret=%zu", ret);

    return IndexError_Runtime;
  }

  return 0;
}

const Neighbors HnswStreamerEntity::get_neighbors(level_t level,
                                                  node_id_t id) const {
  size_t offset = 0UL;
  size_t neighbor_size = neighbor_size_;
  IndexStorage::MemoryBlock neighbor_block;

  if (level == 0UL) {
    uint32_t chunk_idx = id >> node_index_mask_bits_;
    offset =
        (id & node_index_mask_) * node_size() + vector_size() + sizeof(key_t);

    // Fast path: use pre-cached stable base pointer (mmap backend).
    // Bounds-check guards against new chunks added after clone() was taken.
    if (node_chunk_bases_ && chunk_idx < node_chunk_bases_->size() &&
        (*node_chunk_bases_)[chunk_idx]) {
      neighbor_block.reset((void *)((*node_chunk_bases_)[chunk_idx] + offset));
    } else {
      sync_chunks(ChunkBroker::CHUNK_TYPE_NODE, chunk_idx, &node_chunks_);
      ailego_assert_with(chunk_idx < node_chunks_.size(), "invalid chunk idx");
      Chunk *chunk = node_chunks_[chunk_idx].get();
      ailego_assert_with(offset < chunk->data_size(), "invalid chunk offset");
      size_t size = chunk->read(offset, neighbor_block, neighbor_size);
      if (ailego_unlikely(size != neighbor_size)) {
        LOG_ERROR("Read neighbor header failed, ret=%zu", size);
        return Neighbors();
      }
      return Neighbors(neighbor_block);
    }
  } else {
    auto p = get_upper_neighbor_chunk_loc(level, id);
    offset = p.second;
    neighbor_size = upper_neighbor_size_;

    // Fast path: use pre-cached stable base pointer (mmap backend).
    // Bounds-check guards against new chunks added after clone() was taken.
    if (upper_neighbor_chunk_bases_ &&
        p.first < upper_neighbor_chunk_bases_->size() &&
        (*upper_neighbor_chunk_bases_)[p.first]) {
      neighbor_block.reset(
          (void *)((*upper_neighbor_chunk_bases_)[p.first] + offset));
    } else {
      Chunk *chunk = upper_neighbor_chunks_[p.first].get();
      ailego_assert_with(offset < chunk->data_size(), "invalid chunk offset");
      size_t size = chunk->read(offset, neighbor_block, neighbor_size);
      if (ailego_unlikely(size != neighbor_size)) {
        LOG_ERROR("Read neighbor header failed, ret=%zu", size);
        return Neighbors();
      }
      return Neighbors(neighbor_block);
    }
  }

  return Neighbors(neighbor_block);
}

//! Get vector data by key
const void *HnswStreamerEntity::get_vector(node_id_t id) const {
  auto loc = get_vector_chunk_loc(id);
  ailego_assert_with(loc.first < node_chunks_.size(), "invalid chunk idx");

  // Fast path: mmap backend — direct pointer arithmetic.
  // Bounds-check guards against new chunks added after clone() was taken.
  if (node_chunk_bases_ && loc.first < node_chunk_bases_->size() &&
      (*node_chunk_bases_)[loc.first]) {
    return (*node_chunk_bases_)[loc.first] + loc.second;
  }

  ailego_assert_with(loc.second < node_chunks_[loc.first]->data_size(),
                     "invalid chunk offset");
  const void *vec = nullptr;
  size_t read_size = vector_size();
  size_t ret = node_chunks_[loc.first]->read(loc.second, &vec, read_size);
  if (ailego_unlikely(ret != read_size)) {
    LOG_ERROR("Read vector failed, offset=%u, read size=%zu, ret=%zu",
              loc.second, read_size, ret);
  }
  return vec;
}

int HnswStreamerEntity::get_vector(const node_id_t *ids, uint32_t count,
                                   const void **vecs) const {
  for (auto i = 0U; i < count; ++i) {
    auto loc = get_vector_chunk_loc(ids[i]);
    ailego_assert_with(loc.first < node_chunks_.size(), "invalid chunk idx");

    // Fast path: mmap backend.
    // Bounds-check guards against new chunks added after clone() was taken.
    if (node_chunk_bases_ && loc.first < node_chunk_bases_->size() &&
        (*node_chunk_bases_)[loc.first]) {
      vecs[i] = (*node_chunk_bases_)[loc.first] + loc.second;
      continue;
    }

    ailego_assert_with(loc.second < node_chunks_[loc.first]->data_size(),
                       "invalid chunk offset");
    size_t read_size = vector_size();
    size_t ret = node_chunks_[loc.first]->read(loc.second, &vecs[i], read_size);
    if (ailego_unlikely(ret != read_size)) {
      LOG_ERROR("Read vector failed, offset=%u, read size=%zu, ret=%zu",
                loc.second, read_size, ret);
      return IndexError_ReadData;
    }
  }
  return 0;
}

int HnswStreamerEntity::get_vector(const node_id_t id,
                                   IndexStorage::MemoryBlock &block) const {
  auto loc = get_vector_chunk_loc(id);
  ailego_assert_with(loc.first < node_chunks_.size(), "invalid chunk idx");

  // Fast path: mmap backend.
  // Bounds-check guards against new chunks added after clone() was taken.
  if (node_chunk_bases_ && loc.first < node_chunk_bases_->size() &&
      (*node_chunk_bases_)[loc.first]) {
    block.reset((void *)((*node_chunk_bases_)[loc.first] + loc.second));
    return 0;
  }

  ailego_assert_with(loc.second < node_chunks_[loc.first]->data_size(),
                     "invalid chunk offset");
  size_t read_size = vector_size();
  size_t ret = node_chunks_[loc.first]->read(loc.second, block, read_size);
  if (ailego_unlikely(ret != read_size)) {
    LOG_ERROR("Read vector failed, offset=%u, read size=%zu, ret=%zu",
              loc.second, read_size, ret);
    return IndexError_ReadData;
  }
  return 0;
}

int HnswStreamerEntity::get_vector(
    const node_id_t *ids, uint32_t count,
    std::vector<IndexStorage::MemoryBlock> &vec_blocks) const {
  vec_blocks.resize(count);
  for (auto i = 0U; i < count; ++i) {
    auto loc = get_vector_chunk_loc(ids[i]);
    ailego_assert_with(loc.first < node_chunks_.size(), "invalid chunk idx");

    // Fast path: mmap backend.
    // Bounds-check guards against new chunks added after clone() was taken.
    if (node_chunk_bases_ && loc.first < node_chunk_bases_->size() &&
        (*node_chunk_bases_)[loc.first]) {
      vec_blocks[i].reset(
          (void *)((*node_chunk_bases_)[loc.first] + loc.second));
      continue;
    }

    ailego_assert_with(loc.second < node_chunks_[loc.first]->data_size(),
                       "invalid chunk offset");
    size_t read_size = vector_size();
    size_t ret =
        node_chunks_[loc.first]->read(loc.second, vec_blocks[i], read_size);
    if (ailego_unlikely(ret != read_size)) {
      LOG_ERROR("Read vector failed, offset=%u, read size=%zu, ret=%zu",
                loc.second, read_size, ret);
      return IndexError_ReadData;
    }
  }
  return 0;
}

key_t HnswStreamerEntity::get_key(node_id_t id) const {
  if (use_key_info_map_) {
    auto loc = get_key_chunk_loc(id);
    ailego_assert_with(loc.first < node_chunks_.size(), "invalid chunk idx");

    // Fast path: mmap backend.
    // Bounds-check guards against new chunks added after clone() was taken.
    if (node_chunk_bases_ && loc.first < node_chunk_bases_->size() &&
        (*node_chunk_bases_)[loc.first]) {
      return *reinterpret_cast<const key_t *>((*node_chunk_bases_)[loc.first] +
                                              loc.second);
    }

    ailego_assert_with(loc.second < node_chunks_[loc.first]->data_size(),
                       "invalid chunk offset");
    IndexStorage::MemoryBlock key_block;
    size_t ret =
        node_chunks_[loc.first]->read(loc.second, key_block, sizeof(key_t));
    if (ailego_unlikely(ret != sizeof(key_t))) {
      LOG_ERROR("Read vector failed, ret=%zu", ret);
      return kInvalidKey;
    }
    return *reinterpret_cast<const key_t *>(key_block.data());
  } else {
    return id;
  }
}

void HnswStreamerEntity::add_neighbor(level_t level, node_id_t id,
                                      uint32_t size, node_id_t neighbor_id) {
  auto loc = get_neighbor_chunk_loc(level, id);
  size_t offset =
      loc.second + sizeof(NeighborsHeader) + size * sizeof(node_id_t);
  ailego_assert_with(size < neighbor_cnt(level), "invalid neighbor size");
  ailego_assert_with(offset < loc.first->data_size(), "invalid chunk offset");
  size_t ret = loc.first->write(offset, &neighbor_id, sizeof(node_id_t));
  if (ailego_unlikely(ret != sizeof(node_id_t))) {
    LOG_ERROR("Write neighbor id failed, ret=%zu", ret);
    return;
  }

  uint32_t neighbors = size + 1;
  ret = loc.first->write(loc.second, &neighbors, sizeof(uint32_t));
  if (ailego_unlikely(ret != sizeof(uint32_t))) {
    LOG_ERROR("Write neighbor cnt failed, ret=%zu", ret);
  }

  return;
}

int HnswStreamerEntity::init_chunks(const Chunk::Pointer &header_chunk) {
  if (header_chunk->data_size() < header_size()) {
    LOG_ERROR("Invalid header chunk size");
    return IndexError_InvalidFormat;
  }
  IndexStorage::MemoryBlock header_block;
  size_t size = header_chunk->read(0UL, header_block, header_size());
  if (ailego_unlikely(size != header_size())) {
    LOG_ERROR("Read header chunk failed");
    return IndexError_ReadData;
  }
  *mutable_header() =
      *reinterpret_cast<const HNSWHeader *>(header_block.data());

  int ret = check_hnsw_index(&header());
  if (ret != 0) {
    broker_->close();
    return ret;
  }

  node_chunks_.resize(broker_->get_chunk_cnt(ChunkBroker::CHUNK_TYPE_NODE));
  node_chunk_bases_ = std::make_shared<std::vector<const uint8_t *>>(
      node_chunks_.size(), nullptr);
  for (auto seq = 0UL; seq < node_chunks_.size(); ++seq) {
    node_chunks_[seq] = broker_->get_chunk(ChunkBroker::CHUNK_TYPE_NODE, seq);
    if (!node_chunks_[seq]) {
      LOG_ERROR("Missing hnsw streamer data chunk %zu th of %zu", seq,
                node_chunks_.size());
      return IndexError_InvalidFormat;
    }
    (*node_chunk_bases_)[seq] = node_chunks_[seq]->base_data();
  }

  upper_neighbor_chunks_.resize(
      broker_->get_chunk_cnt(ChunkBroker::CHUNK_TYPE_UPPER_NEIGHBOR));
  upper_neighbor_chunk_bases_ = std::make_shared<std::vector<const uint8_t *>>(
      upper_neighbor_chunks_.size(), nullptr);
  for (auto seq = 0UL; seq < upper_neighbor_chunks_.size(); ++seq) {
    upper_neighbor_chunks_[seq] =
        broker_->get_chunk(ChunkBroker::CHUNK_TYPE_UPPER_NEIGHBOR, seq);
    if (!upper_neighbor_chunks_[seq]) {
      LOG_ERROR("Missing hnsw streamer index chunk %zu th of %zu", seq,
                upper_neighbor_chunks_.size());
      return IndexError_InvalidFormat;
    }
    (*upper_neighbor_chunk_bases_)[seq] =
        upper_neighbor_chunks_[seq]->base_data();
  }

  return 0;
}

int HnswStreamerEntity::open(IndexStorage::Pointer stg, uint64_t max_index_size,
                             bool check_crc) {
  std::lock_guard<std::mutex> lock(mutex_);
  bool huge_page = stg->isHugePage();
  LOG_DEBUG("huge_page: %d", (int)huge_page);
  int ret = broker_->open(std::move(stg), chunk_size_, check_crc);
  if (ailego_unlikely(ret != 0)) {
    LOG_ERROR("Open index failed for %s", IndexError::What(ret));
    return ret;
  }
  ret = init_chunk_params(max_index_size, huge_page);
  if (ailego_unlikely(ret != 0)) {
    LOG_ERROR("init_chunk_params failed for %s", IndexError::What(ret));
    return ret;
  }
  broker_->set_max_chunks_size(max_index_size_);

  ret = upper_neighbor_index_->init(broker_, upper_neighbor_chunk_size_,
                                    scaling_factor(), estimate_doc_capacity(),
                                    kUpperHashMemoryInflateRatio);
  if (ailego_unlikely(ret != 0)) {
    LOG_ERROR("Init neighbor hash map failed");
    return ret;
  }

  //! init header
  auto header_chunk = broker_->get_chunk(ChunkBroker::CHUNK_TYPE_HEADER,
                                         ChunkBroker::kDefaultChunkSeqId);
  if (!header_chunk) {  // open empty index, create one
    auto p =
        broker_->alloc_chunk(ChunkBroker::CHUNK_TYPE_HEADER,
                             ChunkBroker::kDefaultChunkSeqId, header_size());
    if (ailego_unlikely(p.first != 0)) {
      LOG_ERROR("Alloc header chunk failed");
      return p.first;
    }
    size_t size = p.second->write(0UL, &header(), header_size());
    if (ailego_unlikely(size != header_size())) {
      LOG_ERROR("Write header chunk failed");
      return IndexError_WriteData;
    }
    return 0;
  }

  //! Open an exist hnsw index
  ret = init_chunks(header_chunk);
  if (ailego_unlikely(ret != 0)) {
    return ret;
  }

  //! total docs including features wrote in index but neighbors may not ready
  node_id_t total_vecs = 0;
  if (node_chunks_.size() > 0) {
    size_t last_idx = node_chunks_.size() - 1;
    auto last_chunk = node_chunks_[last_idx];
    if (last_chunk->data_size() % node_size()) {
      LOG_WARN("The index may broken");
      return IndexError_InvalidFormat;
    }
    total_vecs = last_idx * node_cnt_per_chunk_ +
                 node_chunks_[last_idx]->data_size() / node_size();
  }

  LOG_INFO(
      "Open index, l0NeighborCnt=%zu upperNeighborCnt=%zu "
      "efConstruction=%zu curDocCnt=%u totalVecs=%u maxLevel=%u",
      l0_neighbor_cnt(), upper_neighbor_cnt(), ef_construction(), doc_cnt(),
      total_vecs, cur_max_level());
  //! try to correct the docCnt if index not fully flushed
  if (doc_cnt() != total_vecs) {
    LOG_WARN("Index closed abnormally, using totalVecs as curDocCnt");
    *mutable_doc_cnt() = total_vecs;
  }
  if (filter_same_key_ || get_vector_enabled_) {
    if (use_key_info_map_) {
      for (node_id_t id = 0U; id < doc_cnt(); ++id) {
        if (get_key(id) == kInvalidKey) {
          continue;
        }
        (*keys_map_)[get_key(id)] = id;
      }
    }
  }

  stats_.set_loaded_count(doc_cnt());

  return 0;
}

int HnswStreamerEntity::close() {
  LOG_DEBUG("close index");

  std::lock_guard<std::mutex> lock(mutex_);
  flush_header();
  mutable_header()->reset();
  upper_neighbor_index_->cleanup();
  keys_map_->clear();
  header_.clear();
  node_chunks_.clear();
  node_chunk_bases_.reset();
  upper_neighbor_chunks_.clear();
  upper_neighbor_chunk_bases_.reset();

  return broker_->close();
}

int HnswStreamerEntity::flush(uint64_t checkpoint) {
  LOG_INFO("Flush index, curDocs=%u", doc_cnt());

  std::lock_guard<std::mutex> lock(mutex_);
  flush_header();
  int ret = broker_->flush(checkpoint);
  if (ret != 0) {
    return ret;
  }

  return 0;
}

int HnswStreamerEntity::dump(const IndexDumper::Pointer &dumper) {
  LOG_INFO("Dump index, curDocs=%u", doc_cnt());

  //! sort by keys, to support get_vector by key in searcher
  std::vector<key_t> keys(doc_cnt());
  for (node_id_t i = 0; i < doc_cnt(); ++i) {
    keys[i] = get_key(i);
  }

  //! dump neighbors
  auto get_level = [&](node_id_t id) {
    auto it = upper_neighbor_index_->find(id);
    if (it == upper_neighbor_index_->end()) {
      return 0U;
    };
    auto meta = reinterpret_cast<const UpperNeighborIndexMeta *>(&it->second);
    return meta->level;
  };
  auto ret = dump_segments(dumper, keys.data(), get_level);
  if (ailego_unlikely(ret < 0)) {
    return ret;
  }
  *stats_.mutable_dumped_size() += ret;

  return 0;
}

int HnswStreamerEntity::check_hnsw_index(const HNSWHeader *hd) const {
  if (l0_neighbor_cnt() != hd->l0_neighbor_cnt() ||
      upper_neighbor_cnt() != hd->upper_neighbor_cnt()) {
    LOG_ERROR("Param neighbor cnt: %zu:%zu mismatch index previous %zu:%zu",
              l0_neighbor_cnt(), upper_neighbor_cnt(), hd->l0_neighbor_cnt(),
              hd->upper_neighbor_cnt());
    return IndexError_Mismatch;
  }
  if (vector_size() != hd->vector_size()) {
    LOG_ERROR("vector size %zu mismatch index previous %zu", vector_size(),
              hd->vector_size());
    return IndexError_Mismatch;
  }
  if (ef_construction() != hd->ef_construction()) {
    LOG_WARN("Param efConstruction %zu mismatch index previous %zu",
             ef_construction(), hd->ef_construction());
  }
  if (scaling_factor() != hd->scaling_factor()) {
    LOG_WARN("Param scalingFactor %zu mismatch index previous %zu",
             scaling_factor(), hd->scaling_factor());
    return IndexError_Mismatch;
  }
  if (prune_cnt() != hd->neighbor_prune_cnt()) {
    LOG_WARN("Param pruneCnt %zu mismatch index previous %zu", prune_cnt(),
             hd->neighbor_prune_cnt());
    return IndexError_Mismatch;
  }
  if ((hd->entry_point() != kInvalidNodeId &&
       hd->entry_point() >= hd->doc_cnt()) ||
      (hd->entry_point() == kInvalidNodeId && hd->doc_cnt() > 0U)) {
    LOG_WARN("Invalid entryPoint %u, docCnt %u", hd->entry_point(),
             hd->doc_cnt());
    return IndexError_InvalidFormat;
  }
  if (hd->entry_point() == kInvalidNodeId &&
      broker_->get_chunk_cnt(ChunkBroker::CHUNK_TYPE_NODE) > 0) {
    LOG_WARN("The index is broken, maybe it haven't flush");
    return IndexError_InvalidFormat;
  }

  return 0;
}

int HnswStreamerEntity::add_vector(level_t level, key_t key, const void *vec,
                                   node_id_t *id) {
  Chunk::Pointer node_chunk;
  // On MSVC, unsigned long is 32-bit, so -1UL is 0xFFFFFFFF not
  // 0xFFFFFFFFFFFFFFFF.
  size_t chunk_offset = static_cast<size_t>(-1);

  std::lock_guard<std::mutex> lock(mutex_);
  // duplicate check
  if (ailego_unlikely(filter_same_key_ && get_id(key) != kInvalidNodeId)) {
    LOG_WARN("Try to add duplicate key, ignore it");
    return IndexError_Duplicate;
  }

  node_id_t local_id = static_cast<node_id_t>(doc_cnt());
  uint32_t chunk_index = node_chunks_.size() - 1U;
  if (chunk_index == -1U ||
      (node_chunks_[chunk_index]->data_size() >=
       node_cnt_per_chunk_ * node_size())) {  // no space left and need to alloc
    if (ailego_unlikely(node_chunks_.capacity() == node_chunks_.size())) {
      LOG_ERROR("add vector failed for no memory quota");
      return IndexError_IndexFull;
    }
    chunk_index++;
    auto p = broker_->alloc_chunk(ChunkBroker::CHUNK_TYPE_NODE, chunk_index,
                                  chunk_size_);
    if (ailego_unlikely(p.first != 0)) {
      LOG_ERROR("Alloc data chunk failed");
      return p.first;
    }
    node_chunk = p.second;
    chunk_offset = 0UL;
    node_chunks_.emplace_back(node_chunk);
  } else {
    node_chunk = node_chunks_[chunk_index];
    chunk_offset = node_chunk->data_size();
  }

  size_t size = node_chunk->write(chunk_offset, vec, vector_size());
  if (ailego_unlikely(size != vector_size())) {
    LOG_ERROR("Chunk write vec failed, ret=%zu", size);
    return IndexError_WriteData;
  }
  size = node_chunk->write(chunk_offset + vector_size(), &key, sizeof(key_t));
  if (ailego_unlikely(size != sizeof(key_t))) {
    LOG_ERROR("Chunk write vec failed, ret=%zu", size);
    return IndexError_WriteData;
  }
  //! level 0 neighbors is inited to zero by default

  int ret = add_upper_neighbor(level, local_id);
  if (ret != 0) {
    return ret;
  }

  chunk_offset += node_size();
  if (ailego_unlikely(node_chunk->resize(chunk_offset) != chunk_offset)) {
    LOG_ERROR("Chunk resize to %zu failed", chunk_offset);
    return IndexError_Runtime;
  }
  if (filter_same_key_ || get_vector_enabled_) {
    if (use_key_info_map_) {
      keys_map_lock_->lock();
      (*keys_map_)[key] = local_id;
      keys_map_lock_->unlock();
    }
  }

  *mutable_doc_cnt() += 1;
  broker_->mark_dirty();
  *id = local_id;

  return 0;
}

int HnswStreamerEntity::add_vector_with_id(level_t level, node_id_t id,
                                           const void *vec) {
  Chunk::Pointer node_chunk;
  size_t chunk_offset = static_cast<size_t>(-1);
  key_t key = id;

  std::lock_guard<std::mutex> lock(mutex_);

  // duplicate check
  if (ailego_unlikely(filter_same_key_ && get_id(key) != kInvalidNodeId)) {
    LOG_WARN("Try to add duplicate key, ignore it");
    return IndexError_Duplicate;
  }

  // set node_chunk & chunk_offset if succeed
  auto func_get_node_chunk_and_offset = [&](node_id_t node_id) -> int {
    uint32_t chunk_index = node_id >> node_index_mask_bits_;
    ailego_assert_with(chunk_index <= node_chunks_.size(), "invalid chunk idx");
    // belongs to next chunk
    if (chunk_index == node_chunks_.size()) {
      if (ailego_unlikely(node_chunks_.capacity() == node_chunks_.size())) {
        LOG_ERROR("add vector failed for no memory quota");
        return IndexError_IndexFull;
      }
      auto p = broker_->alloc_chunk(ChunkBroker::CHUNK_TYPE_NODE, chunk_index,
                                    chunk_size_);
      if (ailego_unlikely(p.first != 0)) {
        LOG_ERROR("Alloc data chunk failed");
        return p.first;
      }
      node_chunk = p.second;
      node_chunks_.emplace_back(node_chunk);
    }

    node_chunk = node_chunks_[chunk_index];
    chunk_offset = (node_id & node_index_mask_) * node_size();
    return 0;
  };

  for (size_t start_id = doc_cnt(); start_id < id; ++start_id) {
    if (auto ret = func_get_node_chunk_and_offset(start_id); ret != 0) {
      LOG_ERROR("func_get_node_chunk_and_offset failed");
      return ret;
    }
    size_t size = node_chunk->write(chunk_offset + vector_size(), &kInvalidKey,
                                    sizeof(key_t));
    if (ailego_unlikely(size != sizeof(key_t))) {
      LOG_ERROR("Chunk write key failed, ret=%zu", size);
      return IndexError_WriteData;
    }

    chunk_offset += node_size();
    if (ailego_unlikely(node_chunk->resize(chunk_offset) != chunk_offset)) {
      LOG_ERROR("Chunk resize to %zu failed", chunk_offset);
      return IndexError_Runtime;
    }
  }

  if (auto ret = func_get_node_chunk_and_offset(id); ret != 0) {
    LOG_ERROR("func_get_node_chunk_and_offset failed");
    return ret;
  }

  size_t size = node_chunk->write(chunk_offset, vec, vector_size());
  if (ailego_unlikely(size != vector_size())) {
    LOG_ERROR("Chunk write vec failed, ret=%zu", size);
    return IndexError_WriteData;
  }

  size = node_chunk->write(chunk_offset + vector_size(), &key, sizeof(key_t));
  if (ailego_unlikely(size != sizeof(key_t))) {
    LOG_ERROR("Chunk write vec failed, ret=%zu", size);
    return IndexError_WriteData;
  }
  //! level 0 neighbors is inited to zero by default

  int ret = add_upper_neighbor(level, id);
  if (ret != 0) {
    return ret;
  }

  if (*mutable_doc_cnt() <= id) {
    *mutable_doc_cnt() = id + 1;
    chunk_offset += node_size();
    if (ailego_unlikely(node_chunk->resize(chunk_offset) != chunk_offset)) {
      LOG_ERROR("Chunk resize to %zu failed", chunk_offset);
      return IndexError_Runtime;
    }
  }

  if (filter_same_key_ || get_vector_enabled_) {
    if (use_key_info_map_) {
      keys_map_lock_->lock();
      (*keys_map_)[key] = id;
      keys_map_lock_->unlock();
    }
  }

  broker_->mark_dirty();

  return 0;
}

void HnswStreamerEntity::update_ep_and_level(node_id_t ep, level_t level) {
  HnswEntity::update_ep_and_level(ep, level);
  flush_header();

  return;
}

const HnswEntity::Pointer HnswStreamerEntity::clone() const {
  std::vector<Chunk::Pointer> node_chunks;
  node_chunks.reserve(node_chunks_.size());
  for (size_t i = 0UL; i < node_chunks_.size(); ++i) {
    node_chunks.emplace_back(node_chunks_[i]->clone());
    if (ailego_unlikely(!node_chunks[i])) {
      LOG_ERROR("HnswStreamerEntity get chunk failed in clone");
      return HnswEntity::Pointer();
    }
  }

  std::vector<Chunk::Pointer> upper_neighbor_chunks;
  upper_neighbor_chunks.reserve(upper_neighbor_chunks_.size());
  for (size_t i = 0UL; i < upper_neighbor_chunks_.size(); ++i) {
    upper_neighbor_chunks.emplace_back(upper_neighbor_chunks_[i]->clone());
    if (ailego_unlikely(!upper_neighbor_chunks[i])) {
      LOG_ERROR("HnswStreamerEntity get chunk failed in clone");
      return HnswEntity::Pointer();
    }
  }

  HnswStreamerEntity *entity = new (std::nothrow) HnswStreamerEntity(
      stats_, header(), chunk_size_, node_index_mask_bits_,
      upper_neighbor_mask_bits_, filter_same_key_, get_vector_enabled_,
      upper_neighbor_index_, keys_map_lock_, keys_map_, use_key_info_map_,
      std::move(node_chunks), std::move(upper_neighbor_chunks), broker_,
      node_chunk_bases_, upper_neighbor_chunk_bases_);
  if (ailego_unlikely(!entity)) {
    LOG_ERROR("HnswStreamerEntity new failed");
  }
  return HnswEntity::Pointer(entity);
}

const HnswEntity::Pointer HnswMmapStreamerEntity::clone() const {
  std::vector<Chunk::Pointer> node_chunks;
  node_chunks.reserve(node_chunks_.size());
  for (size_t i = 0UL; i < node_chunks_.size(); ++i) {
    node_chunks.emplace_back(node_chunks_[i]->clone());
    if (ailego_unlikely(!node_chunks[i])) {
      LOG_ERROR("HnswMmapStreamerEntity get chunk failed in clone");
      return HnswEntity::Pointer();
    }
  }

  std::vector<Chunk::Pointer> upper_neighbor_chunks;
  upper_neighbor_chunks.reserve(upper_neighbor_chunks_.size());
  for (size_t i = 0UL; i < upper_neighbor_chunks_.size(); ++i) {
    upper_neighbor_chunks.emplace_back(upper_neighbor_chunks_[i]->clone());
    if (ailego_unlikely(!upper_neighbor_chunks[i])) {
      LOG_ERROR("HnswMmapStreamerEntity get chunk failed in clone");
      return HnswEntity::Pointer();
    }
  }

  auto *entity = new (std::nothrow) HnswMmapStreamerEntity(
      stats_, header(), chunk_size_, node_index_mask_bits_,
      upper_neighbor_mask_bits_, filter_same_key_, get_vector_enabled_,
      upper_neighbor_index_, keys_map_lock_, keys_map_, use_key_info_map_,
      std::move(node_chunks), std::move(upper_neighbor_chunks), broker_,
      nullptr, nullptr);
  if (ailego_unlikely(!entity)) {
    LOG_ERROR("HnswMmapStreamerEntity new failed");
  }
  return HnswEntity::Pointer(entity);
}

const HnswEntity::Pointer HnswContiguousStreamerEntity::clone() const {
  std::vector<Chunk::Pointer> node_chunks;
  node_chunks.reserve(node_chunks_.size());
  for (size_t i = 0UL; i < node_chunks_.size(); ++i) {
    node_chunks.emplace_back(node_chunks_[i]->clone());
    if (ailego_unlikely(!node_chunks[i])) {
      LOG_ERROR("HnswContiguousStreamerEntity get chunk failed in clone");
      return HnswEntity::Pointer();
    }
  }

  std::vector<Chunk::Pointer> upper_neighbor_chunks;
  upper_neighbor_chunks.reserve(upper_neighbor_chunks_.size());
  for (size_t i = 0UL; i < upper_neighbor_chunks_.size(); ++i) {
    upper_neighbor_chunks.emplace_back(upper_neighbor_chunks_[i]->clone());
    if (ailego_unlikely(!upper_neighbor_chunks[i])) {
      LOG_ERROR("HnswContiguousStreamerEntity get chunk failed in clone");
      return HnswEntity::Pointer();
    }
  }

  auto *entity = new (std::nothrow) HnswContiguousStreamerEntity(
      stats_, header(), chunk_size_, node_index_mask_bits_,
      upper_neighbor_mask_bits_, filter_same_key_, get_vector_enabled_,
      upper_neighbor_index_, keys_map_lock_, keys_map_, use_key_info_map_,
      std::move(node_chunks), std::move(upper_neighbor_chunks), broker_,
      nullptr, nullptr);
  if (ailego_unlikely(!entity)) {
    LOG_ERROR("HnswContiguousStreamerEntity new failed");
    return HnswEntity::Pointer();
  }

  // Share contiguous memory with the clone (zero-copy)
  entity->node_memory_ = node_memory_;
  entity->node_base_ = node_base_;
  entity->upper_neighbor_memory_ = upper_neighbor_memory_;
  entity->upper_neighbor_base_ = upper_neighbor_base_;
  entity->upper_chunk_offsets_ = upper_chunk_offsets_;

  return HnswEntity::Pointer(entity);
}

// ============================================================================
// HnswContiguousStreamerEntity implementation
// ============================================================================

char *HnswContiguousStreamerEntity::allocate_contiguous(size_t size) {
  if (size == 0) {
    return nullptr;
  }
#if defined(__linux__)
  // Use mmap with MAP_ANONYMOUS for contiguous memory
  void *ptr = ::mmap(nullptr, size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (ptr == MAP_FAILED) {
    LOG_ERROR("mmap failed for contiguous memory, size=%zu", size);
    return nullptr;
  }
  // Request transparent huge pages
  ::madvise(ptr, size, MADV_HUGEPAGE);
  return static_cast<char *>(ptr);
#elif defined(__APPLE__)
  // macOS: use mmap with MAP_ANONYMOUS
  void *ptr = ::mmap(nullptr, size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANON, -1, 0);
  if (ptr == MAP_FAILED) {
    LOG_ERROR("mmap failed for contiguous memory, size=%zu", size);
    return nullptr;
  }
  return static_cast<char *>(ptr);
#elif defined(_WIN32)
  void *ptr = ::_aligned_malloc(size, ailego::MemoryHelper::PageSize());
  if (!ptr) {
    LOG_ERROR("_aligned_malloc failed for contiguous memory, size=%zu", size);
    return nullptr;
  }
  return static_cast<char *>(ptr);
#else
  // Fallback: aligned allocation
  void *ptr = std::aligned_alloc(ailego::MemoryHelper::PageSize(), size);
  if (!ptr) {
    LOG_ERROR("aligned_alloc failed for contiguous memory, size=%zu", size);
    return nullptr;
  }
  return static_cast<char *>(ptr);
#endif
}

int HnswContiguousStreamerEntity::build_contiguous_memory() {
  node_memory_.reset();
  node_base_ = nullptr;
  upper_neighbor_memory_.reset();
  upper_neighbor_base_ = nullptr;
  upper_chunk_offsets_.clear();

  const uint32_t total_docs = doc_cnt();
  if (total_docs == 0) {
    return 0;
  }

  // --- Build contiguous node memory ---
  const size_t per_node = node_size();
  const size_t total_node_data = static_cast<size_t>(total_docs) * per_node;
  size_t node_memory_size = AlignHugePageSize(total_node_data);
  char *raw_node = allocate_contiguous(node_memory_size);
  if (!raw_node) {
    return IndexError_Runtime;
  }
  node_memory_.reset(raw_node, ContiguousDeleter{node_memory_size});
  node_base_ = raw_node;

  // Copy node data from chunks into contiguous memory
  // Each chunk holds node_cnt_per_chunk nodes, laid out at offset
  // (id & mask) * node_size within the chunk.
  const auto &chunks = node_chunks_;
  const uint32_t nodes_per_chunk = 1U << node_index_mask_bits_;
  for (size_t chunk_idx = 0; chunk_idx < chunks.size(); ++chunk_idx) {
    const void *chunk_data = nullptr;
    size_t data_size = chunks[chunk_idx]->data_size();
    chunks[chunk_idx]->read(0, &chunk_data, data_size);

    // Number of nodes in this chunk
    uint32_t base_id = chunk_idx * nodes_per_chunk;
    uint32_t count_in_chunk = std::min(nodes_per_chunk, total_docs - base_id);

    // Copy each node's data
    const char *src = static_cast<const char *>(chunk_data);
    char *dst = node_base_ + static_cast<size_t>(base_id) * per_node;
    std::memcpy(dst, src, static_cast<size_t>(count_in_chunk) * per_node);
  }

  // --- Build contiguous upper neighbor memory ---
  const auto &upper_chunks = upper_neighbor_chunks_;
  if (upper_chunks.empty()) {
    return 0;
  }

  // Sync all upper neighbor chunks
  sync_upper_neighbor_chunks(upper_chunks.size() - 1);

  // Calculate cumulative offsets and total size
  upper_chunk_offsets_.resize(upper_chunks.size());
  size_t total_upper_size = 0;
  for (size_t i = 0; i < upper_chunks.size(); ++i) {
    upper_chunk_offsets_[i] = total_upper_size;
    total_upper_size += upper_chunks[i]->data_size();
  }

  size_t upper_memory_size = AlignHugePageSize(total_upper_size);
  char *raw_upper = allocate_contiguous(upper_memory_size);
  if (!raw_upper) {
    node_memory_.reset();
    node_base_ = nullptr;
    return IndexError_Runtime;
  }
  upper_neighbor_memory_.reset(raw_upper, ContiguousDeleter{upper_memory_size});
  upper_neighbor_base_ = raw_upper;

  // Copy upper neighbor data from chunks into contiguous memory
  for (size_t i = 0; i < upper_chunks.size(); ++i) {
    const void *chunk_data = nullptr;
    size_t data_size = upper_chunks[i]->data_size();
    upper_chunks[i]->read(0, &chunk_data, data_size);
    std::memcpy(upper_neighbor_base_ + upper_chunk_offsets_[i], chunk_data,
                data_size);
  }

  LOG_INFO(
      "Built contiguous memory: node_size=%zu upper_neighbor_size=%zu "
      "total_docs=%u node_chunks=%zu upper_chunks=%zu",
      node_memory_size, upper_memory_size, total_docs, chunks.size(),
      upper_chunks.size());

  return 0;
}

}  // namespace core
}  // namespace zvec
