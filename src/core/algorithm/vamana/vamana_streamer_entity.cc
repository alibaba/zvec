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
#include "vamana_streamer_entity.h"
#include <ailego/utility/memory_helper.h>
#include <zvec/ailego/hash/crc32c.h>
#include <zvec/core/framework/index_stats.h>

namespace zvec {
namespace core {

VamanaStreamerEntity::VamanaStreamerEntity(IndexStreamer::Stats &stats)
    : stats_(stats) {
  keys_map_lock_ = std::make_shared<ailego::SharedMutex>();
  keys_map_ = std::make_shared<HashMap<key_t, node_id_t>>();
  keys_map_->set_empty_key(kInvalidKey);
  broker_ = std::make_shared<ChunkBroker>(stats);
}

VamanaStreamerEntity::~VamanaStreamerEntity() {}

int VamanaStreamerEntity::cleanup() {
  node_chunks_.clear();
  if (keys_map_) {
    keys_map_->clear();
  }
  header_.clear();
  return 0;
}

int VamanaStreamerEntity::init(size_t /*max_doc_cnt*/) {
  // node_size = vector_size + key_size + neighbors_size
  set_node_size(vector_size() + sizeof(key_t) + neighbors_size());
  neighbor_size_ = neighbors_size();
  return 0;
}

key_t VamanaStreamerEntity::get_key(node_id_t id) const {
  if (!use_key_info_map_) return id;
  auto loc = get_key_chunk_loc(id);
  if (ailego_unlikely(loc.first >= node_chunks_.size())) return kInvalidKey;
  const void *ptr = nullptr;
  size_t ret = node_chunks_[loc.first]->read(loc.second, &ptr, sizeof(key_t));
  if (ailego_unlikely(ret != sizeof(key_t))) {
    LOG_ERROR("Read key failed, ret=%zu", ret);
    return kInvalidKey;
  }
  return *reinterpret_cast<const key_t *>(ptr);
}

const void *VamanaStreamerEntity::get_vector(node_id_t id) const {
  auto loc = get_vector_chunk_loc(id);
  if (ailego_unlikely(loc.first >= node_chunks_.size())) return nullptr;
  const void *ptr = nullptr;
  size_t ret = node_chunks_[loc.first]->read(loc.second, &ptr, vector_size());
  if (ailego_unlikely(ret != vector_size())) {
    LOG_ERROR("Read vector failed, ret=%zu", ret);
    return nullptr;
  }
  return ptr;
}

int VamanaStreamerEntity::get_vector(const node_id_t id,
                                     IndexStorage::MemoryBlock &block) const {
  auto loc = get_vector_chunk_loc(id);
  if (ailego_unlikely(loc.first >= node_chunks_.size()))
    return IndexError_NoExist;
  size_t ret = node_chunks_[loc.first]->read(loc.second, block, vector_size());
  if (ailego_unlikely(ret != vector_size())) {
    LOG_ERROR("Read vector failed, ret=%zu", ret);
    return IndexError_ReadData;
  }
  return 0;
}

int VamanaStreamerEntity::get_vector(const node_id_t *ids, uint32_t count,
                                     const void **vecs) const {
  for (uint32_t i = 0; i < count; ++i) {
    vecs[i] = get_vector(ids[i]);
    if (ailego_unlikely(vecs[i] == nullptr)) {
      return IndexError_NoExist;
    }
  }
  return 0;
}

int VamanaStreamerEntity::get_vector(
    const node_id_t *ids, uint32_t count,
    std::vector<IndexStorage::MemoryBlock> &vec_blocks) const {
  vec_blocks.resize(count);
  for (uint32_t i = 0; i < count; ++i) {
    int ret = get_vector(ids[i], vec_blocks[i]);
    if (ailego_unlikely(ret != 0)) return ret;
  }
  return 0;
}

const Neighbors VamanaStreamerEntity::get_neighbors(node_id_t id) const {
  auto loc = get_neighbor_chunk_loc(id);
  IndexStorage::MemoryBlock mem_block;
  size_t ret = loc.first->read(loc.second, mem_block, neighbor_size_);
  if (ailego_unlikely(ret != neighbor_size_)) {
    LOG_ERROR("Read neighbor header failed, ret=%zu", ret);
    return Neighbors();
  }
  return Neighbors(mem_block);
}

int VamanaStreamerEntity::add_vector(key_t key, const void *vec,
                                     node_id_t *id) {
  Chunk::Pointer node_chunk;
  size_t chunk_offset = static_cast<size_t>(-1);

  std::lock_guard<std::mutex> lock(mutex_);

  node_id_t local_id = static_cast<node_id_t>(doc_cnt());
  uint32_t chunk_index = node_chunks_.size() - 1U;
  if (chunk_index == static_cast<uint32_t>(-1) ||
      (node_chunks_[chunk_index]->data_size() >=
       node_cnt_per_chunk_ * node_size())) {
    if (ailego_unlikely(node_chunks_.capacity() == node_chunks_.size())) {
      LOG_ERROR("add vector failed for no memory quota");
      return IndexError_IndexFull;
    }
    chunk_index++;
    if (auto dret = ensure_dist_chunk_for(chunk_index); dret != 0) {
      return dret;
    }
    auto p = broker_->alloc_chunk(ChunkBroker::CHUNK_TYPE_NODE, chunk_index,
                                  chunk_size_);
    if (ailego_unlikely(p.first != 0)) {
      LOG_ERROR("Alloc data chunk failed");
      return p.first;
    }
    node_chunk = p.second;
    chunk_offset = 0UL;
    {
      std::lock_guard<std::mutex> chunks_lock(node_chunks_mutex_);
      node_chunks_.emplace_back(node_chunk);
    }
  } else {
    node_chunk = node_chunks_[chunk_index];
    chunk_offset = node_chunk->data_size();
  }

  // Write vector
  size_t size = node_chunk->write(chunk_offset, vec, vector_size());
  if (ailego_unlikely(size != vector_size())) {
    LOG_ERROR("Chunk write vec failed, ret=%zu", size);
    return IndexError_WriteData;
  }
  // Write key
  size = node_chunk->write(chunk_offset + vector_size(), &key, sizeof(key_t));
  if (ailego_unlikely(size != sizeof(key_t))) {
    LOG_ERROR("Chunk write key failed, ret=%zu", size);
    return IndexError_WriteData;
  }
  // Neighbors are initialized to zero by default (chunk is zero-filled)

  chunk_offset += node_size();
  if (ailego_unlikely(node_chunk->resize(chunk_offset) != chunk_offset)) {
    LOG_ERROR("Chunk resize to %zu failed", chunk_offset);
    return IndexError_Runtime;
  }

  if (use_key_info_map_) {
    keys_map_lock_->lock();
    (*keys_map_)[key] = local_id;
    keys_map_lock_->unlock();
  }

  *mutable_doc_cnt() += 1;
  broker_->mark_dirty();
  *id = local_id;

  return 0;
}

int VamanaStreamerEntity::add_vector_with_id(node_id_t id, const void *vec) {
  Chunk::Pointer node_chunk;
  size_t chunk_offset = static_cast<size_t>(-1);
  key_t key = id;

  std::lock_guard<std::mutex> lock(mutex_);

  auto func_get_node_chunk_and_offset = [&](node_id_t node_id) -> int {
    uint32_t chunk_idx = node_id >> node_index_mask_bits_;
    ailego_assert_with(chunk_idx <= node_chunks_.size(), "invalid chunk idx");
    if (chunk_idx == node_chunks_.size()) {
      if (ailego_unlikely(node_chunks_.capacity() == node_chunks_.size())) {
        LOG_ERROR("add vector failed for no memory quota");
        return IndexError_IndexFull;
      }
      if (auto dret = ensure_dist_chunk_for(chunk_idx); dret != 0) {
        return dret;
      }
      auto p = broker_->alloc_chunk(ChunkBroker::CHUNK_TYPE_NODE, chunk_idx,
                                    chunk_size_);
      if (ailego_unlikely(p.first != 0)) {
        LOG_ERROR("Alloc data chunk failed");
        return p.first;
      }
      node_chunk = p.second;
      {
        std::lock_guard<std::mutex> chunks_lock(node_chunks_mutex_);
        node_chunks_.emplace_back(node_chunk);
      }
    }
    node_chunk = node_chunks_[chunk_idx];
    chunk_offset = (node_id & node_index_mask_) * node_size();
    return 0;
  };

  // Fill gaps with invalid keys
  for (size_t start_id = doc_cnt(); start_id < id; ++start_id) {
    if (auto ret = func_get_node_chunk_and_offset(start_id); ret != 0) {
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
    return ret;
  }

  // Write vector
  size_t size = node_chunk->write(chunk_offset, vec, vector_size());
  if (ailego_unlikely(size != vector_size())) {
    LOG_ERROR("Chunk write vec failed, ret=%zu", size);
    return IndexError_WriteData;
  }
  // Write key
  size = node_chunk->write(chunk_offset + vector_size(), &key, sizeof(key_t));
  if (ailego_unlikely(size != sizeof(key_t))) {
    LOG_ERROR("Chunk write key failed, ret=%zu", size);
    return IndexError_WriteData;
  }

  if (*mutable_doc_cnt() <= id) {
    *mutable_doc_cnt() = id + 1;
    chunk_offset += node_size();
    if (ailego_unlikely(node_chunk->resize(chunk_offset) != chunk_offset)) {
      LOG_ERROR("Chunk resize to %zu failed", chunk_offset);
      return IndexError_Runtime;
    }
  }

  if (use_key_info_map_) {
    keys_map_lock_->lock();
    (*keys_map_)[key] = id;
    keys_map_lock_->unlock();
  }

  broker_->mark_dirty();
  return 0;
}

int VamanaStreamerEntity::update_neighbors(
    node_id_t id, const std::vector<std::pair<node_id_t, dist_t>> &neighbors) {
  auto loc = get_neighbor_chunk_loc(id);
  uint32_t count = std::min(static_cast<uint32_t>(neighbors.size()),
                            static_cast<uint32_t>(max_degree()));

  // Build neighbor data in a local buffer
  size_t nbr_size = neighbors_size();
  std::vector<uint8_t> buffer(nbr_size, 0);
  auto *hd = reinterpret_cast<NeighborsHeader *>(buffer.data());
  hd->neighbor_cnt = count;
  for (uint32_t i = 0; i < count; ++i) {
    hd->neighbors[i] = neighbors[i].first;
  }

  size_t ret = loc.first->write(loc.second, buffer.data(), nbr_size);
  if (ailego_unlikely(ret != nbr_size)) {
    LOG_ERROR("Write neighbors failed, ret=%zu", ret);
    return IndexError_WriteData;
  }
  return 0;
}

void VamanaStreamerEntity::add_neighbor(node_id_t id, uint32_t size,
                                        node_id_t neighbor_id) {
  auto loc = get_neighbor_chunk_loc(id);
  if (size >= max_degree()) return;

  // Read current neighbors
  IndexStorage::MemoryBlock mem_block;
  size_t ret = loc.first->read(loc.second, mem_block, neighbor_size_);
  if (ailego_unlikely(ret != neighbor_size_)) {
    LOG_ERROR("Read neighbor header failed, ret=%zu", ret);
    return;
  }

  // Copy to mutable buffer, update, and write back
  std::vector<uint8_t> buffer(neighbor_size_);
  memcpy(buffer.data(), mem_block.data(), neighbor_size_);
  auto *hd = reinterpret_cast<NeighborsHeader *>(buffer.data());
  hd->neighbors[size] = neighbor_id;
  hd->neighbor_cnt = size + 1;

  ret = loc.first->write(loc.second, buffer.data(), neighbor_size_);
  if (ailego_unlikely(ret != neighbor_size_)) {
    LOG_ERROR("Write neighbor failed, ret=%zu", ret);
  }
}

void VamanaStreamerEntity::update_entry_point(node_id_t ep) {
  VamanaEntity::update_entry_point(ep);
  flush_header();
}

int VamanaStreamerEntity::open(IndexStorage::Pointer stg,
                               uint64_t max_index_size, bool check_crc) {
  std::lock_guard<std::mutex> lock(mutex_);
  bool huge_page = stg->isHugePage();

  int ret = broker_->open(std::move(stg), chunk_size_, check_crc);
  if (ailego_unlikely(ret != 0)) {
    LOG_ERROR("Open index failed: %s", IndexError::What(ret));
    return ret;
  }

  ret = init_chunk_params(max_index_size, huge_page);
  if (ailego_unlikely(ret != 0)) {
    LOG_ERROR("init_chunk_params failed: %s", IndexError::What(ret));
    return ret;
  }
  broker_->set_max_chunks_size(max_index_size_);

  // Init header
  auto header_chunk = broker_->get_chunk(ChunkBroker::CHUNK_TYPE_HEADER,
                                         ChunkBroker::kDefaultChunkSeqId);
  if (!header_chunk) {
    // Open empty index, create header
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

  // Open existing index
  ret = init_chunks(header_chunk);
  if (ailego_unlikely(ret != 0)) return ret;

  // Verify total docs
  node_id_t total_vecs = 0;
  if (!node_chunks_.empty()) {
    size_t last_idx = node_chunks_.size() - 1;
    if (node_chunks_[last_idx]->data_size() % node_size()) {
      LOG_WARN("The index may be broken");
      return IndexError_InvalidFormat;
    }
    total_vecs = last_idx * node_cnt_per_chunk_ +
                 node_chunks_[last_idx]->data_size() / node_size();
  }

  LOG_INFO("Open Vamana index, maxDegree=%zu docCnt=%u totalVecs=%u",
           max_degree(), doc_cnt(), total_vecs);

  if (doc_cnt() != total_vecs) {
    LOG_WARN("Index closed abnormally, using totalVecs as curDocCnt");
    *mutable_doc_cnt() = total_vecs;
  }

  // Rebuild key map
  if (use_key_info_map_) {
    for (node_id_t i = 0; i < doc_cnt(); ++i) {
      key_t k = get_key(i);
      if (k != kInvalidKey) {
        (*keys_map_)[k] = i;
      }
    }
  }

  stats_.set_loaded_count(doc_cnt());
  return 0;
}

int VamanaStreamerEntity::init_chunks(const Chunk::Pointer &header_chunk) {
  // Read header from chunk
  const void *hd_ptr = nullptr;
  size_t ret = header_chunk->read(0UL, &hd_ptr, header_size());
  if (ailego_unlikely(ret != header_size())) {
    LOG_ERROR("Read header chunk failed");
    return IndexError_ReadData;
  }
  auto *hd = reinterpret_cast<const VamanaHeader *>(hd_ptr);

  // Validate
  if (vector_size() != hd->vector_size()) {
    LOG_ERROR("vector size %zu mismatch index previous %zu", vector_size(),
              hd->vector_size());
    return IndexError_Mismatch;
  }
  if (max_degree() != hd->max_degree()) {
    LOG_ERROR("max_degree %zu mismatch index previous %zu", max_degree(),
              hd->max_degree());
    return IndexError_Mismatch;
  }

  *mutable_header() = *hd;

  // Load node chunks
  size_t chunk_cnt = broker_->get_chunk_cnt(ChunkBroker::CHUNK_TYPE_NODE);
  for (size_t i = 0; i < chunk_cnt; ++i) {
    auto chunk = broker_->get_chunk(ChunkBroker::CHUNK_TYPE_NODE, i);
    if (ailego_unlikely(!chunk)) {
      LOG_ERROR("Get node chunk %zu failed", i);
      return IndexError_ReadData;
    }
    node_chunks_.emplace_back(std::move(chunk));
  }
  return 0;
}

int VamanaStreamerEntity::close() {
  LOG_DEBUG("close Vamana index");
  std::lock_guard<std::mutex> lock(mutex_);
  flush_header();
  mutable_header()->reset();
  keys_map_->clear();
  header_.clear();
  node_chunks_.clear();
  dist_chunks_.clear();
  dist_loaded_ = false;
  return broker_->close();
}

int VamanaStreamerEntity::flush(uint64_t checkpoint) {
  LOG_INFO("Flush Vamana index, curDocs=%u", doc_cnt());
  std::lock_guard<std::mutex> lock(mutex_);
  flush_header();
  return broker_->flush(checkpoint);
}

int VamanaStreamerEntity::dump(const IndexDumper::Pointer &dumper) {
  LOG_INFO("Dump Vamana index, curDocs=%u", doc_cnt());

  std::vector<key_t> keys(doc_cnt());
  auto ret = dump_segments(dumper, keys.data());
  if (ailego_unlikely(ret < 0)) {
    return static_cast<int>(ret);
  }
  *stats_.mutable_dumped_size() += ret;
  return 0;
}

const VamanaEntity::Pointer VamanaStreamerEntity::clone() const {
  std::vector<Chunk::Pointer> cloned_chunks;
  cloned_chunks.reserve(node_chunks_.size());
  for (size_t i = 0; i < node_chunks_.size(); ++i) {
    cloned_chunks.emplace_back(node_chunks_[i]->clone());
    if (ailego_unlikely(!cloned_chunks[i])) {
      LOG_ERROR("VamanaStreamerEntity get chunk failed in clone");
      return VamanaEntity::Pointer();
    }
  }

  auto *entity = new (std::nothrow) VamanaStreamerEntity(
      stats_, header(), chunk_size_, node_index_mask_bits_, get_vector_enabled_,
      use_key_info_map_, keys_map_lock_, keys_map_, std::move(cloned_chunks),
      broker_);
  if (ailego_unlikely(!entity)) {
    LOG_ERROR("VamanaStreamerEntity new failed");
  }
  return VamanaEntity::Pointer(entity);
}

const VamanaEntity::Pointer VamanaMmapStreamerEntity::clone() const {
  std::vector<Chunk::Pointer> cloned_chunks;
  cloned_chunks.reserve(node_chunks_.size());
  for (size_t i = 0; i < node_chunks_.size(); ++i) {
    cloned_chunks.emplace_back(node_chunks_[i]->clone());
    if (ailego_unlikely(!cloned_chunks[i])) {
      LOG_ERROR("VamanaMmapStreamerEntity get chunk failed in clone");
      return VamanaEntity::Pointer();
    }
  }

  auto *entity = new (std::nothrow) VamanaMmapStreamerEntity(
      stats_, header(), chunk_size_, node_index_mask_bits_, get_vector_enabled_,
      use_key_info_map_, keys_map_lock_, keys_map_, std::move(cloned_chunks),
      broker_);
  if (ailego_unlikely(!entity)) {
    LOG_ERROR("VamanaMmapStreamerEntity new failed");
  }
  return VamanaEntity::Pointer(entity);
}

const VamanaEntity::Pointer VamanaContiguousStreamerEntity::clone() const {
  std::vector<Chunk::Pointer> cloned_chunks;
  cloned_chunks.reserve(node_chunks_.size());
  for (size_t i = 0; i < node_chunks_.size(); ++i) {
    cloned_chunks.emplace_back(node_chunks_[i]->clone());
    if (ailego_unlikely(!cloned_chunks[i])) {
      LOG_ERROR("VamanaContiguousStreamerEntity get chunk failed in clone");
      return VamanaEntity::Pointer();
    }
  }

  auto *entity = new (std::nothrow) VamanaContiguousStreamerEntity(
      stats_, header(), chunk_size_, node_index_mask_bits_, get_vector_enabled_,
      use_key_info_map_, keys_map_lock_, keys_map_, std::move(cloned_chunks),
      broker_);
  if (ailego_unlikely(!entity)) {
    LOG_ERROR("VamanaContiguousStreamerEntity new failed");
    return VamanaEntity::Pointer();
  }

  // Share contiguous memory with the clone (zero-copy)
  entity->vector_memory_ = vector_memory_;
  entity->vector_base_ = vector_base_;
  entity->vector_stride_ = vector_stride_;
  entity->graph_memory_ = graph_memory_;
  entity->graph_base_ = graph_base_;
  entity->graph_stride_ = graph_stride_;

  return VamanaEntity::Pointer(entity);
}

// ============================================================================
// VamanaContiguousStreamerEntity implementation
// ============================================================================

char *VamanaContiguousStreamerEntity::allocate_contiguous(size_t size) {
  if (size == 0) return nullptr;
  void *ptr = ailego::MemoryHelper::AllocateHugePage(size);
  if (!ptr) {
    LOG_ERROR("AllocateHugePage failed for contiguous memory, size=%zu", size);
    return nullptr;
  }
  return static_cast<char *>(ptr);
}

int VamanaContiguousStreamerEntity::build_contiguous_memory() {
  vector_memory_.reset();
  vector_base_ = nullptr;
  vector_stride_ = 0;
  graph_memory_.reset();
  graph_base_ = nullptr;

  const uint32_t total_docs = doc_cnt();
  if (total_docs == 0) return 0;

  const size_t per_node = node_size();
  const size_t vec_size = vector_size();

  // Pad per-vector stride up to kVectorAlignment (64B) so every vector
  // starts on a cache-line boundary.
  vector_stride_ =
      (vec_size + (kVectorAlignment - 1)) & ~(kVectorAlignment - 1);
  // graph_stride = key + neighbors (everything except vector)
  graph_stride_ = sizeof(key_t) + neighbors_size();

  // Allocate flat vector array (stride = vector_stride_, padded for 64B)
  const size_t total_vec_data =
      static_cast<size_t>(total_docs) * vector_stride_;
  size_t vector_memory_size = AlignHugePageSize(total_vec_data);
  char *raw_vec = allocate_contiguous(vector_memory_size);
  if (!raw_vec) return IndexError_Runtime;
  vector_memory_.reset(raw_vec, ContiguousDeleter{vector_memory_size});
  vector_base_ = raw_vec;

  // Allocate graph array (stride = sizeof(key_t) + neighbors_size)
  const size_t total_graph_data =
      static_cast<size_t>(total_docs) * graph_stride_;
  size_t graph_memory_size = AlignHugePageSize(total_graph_data);
  char *raw_graph = allocate_contiguous(graph_memory_size);
  if (!raw_graph) {
    vector_memory_.reset();
    vector_base_ = nullptr;
    vector_stride_ = 0;
    return IndexError_Runtime;
  }
  graph_memory_.reset(raw_graph, ContiguousDeleter{graph_memory_size});
  graph_base_ = raw_graph;

  // Split node data from chunks into vector / graph arrays.
  // Original node layout: [vector (vec_size) | key (8B) | neighbors]
  // Padding bytes in vector_base_ are left zero (anon mmap is zero-filled).
  const auto &chunks = node_chunks_;
  const uint32_t nodes_per_chunk = 1U << node_index_mask_bits_;
  for (size_t chunk_idx = 0; chunk_idx < chunks.size(); ++chunk_idx) {
    const void *chunk_data = nullptr;
    size_t data_size = chunks[chunk_idx]->data_size();
    chunks[chunk_idx]->read(0, &chunk_data, data_size);

    uint32_t base_id = chunk_idx * nodes_per_chunk;
    uint32_t count_in_chunk = std::min(nodes_per_chunk, total_docs - base_id);

    const char *src = static_cast<const char *>(chunk_data);
    for (uint32_t i = 0; i < count_in_chunk; ++i) {
      const char *node_src = src + static_cast<size_t>(i) * per_node;
      size_t global_id = static_cast<size_t>(base_id + i);

      // Copy vector to flat vector array at padded stride
      std::memcpy(vector_base_ + global_id * vector_stride_, node_src,
                  vec_size);

      // Copy key + neighbors to graph array
      std::memcpy(graph_base_ + global_id * graph_stride_, node_src + vec_size,
                  graph_stride_);
    }
  }

  LOG_INFO(
      "Built Vamana contiguous memory: "
      "vector_mem=%zu graph_mem=%zu total_docs=%u "
      "node_chunks=%zu vector_size=%zu vector_stride=%zu "
      "(cache-line aligned to %zuB)",
      vector_memory_size, graph_memory_size, total_docs, chunks.size(),
      vec_size, vector_stride_, kVectorAlignment);

  return 0;
}

// ============================================================================
// Neighbor distance storage implementation (CSR-like, lazy-loaded)
// ============================================================================

int VamanaStreamerEntity::ensure_dist_storage() {
  if (dist_loaded_) return 0;

  std::lock_guard<std::mutex> lock(mutex_);
  if (dist_loaded_) return 0;  // double-check after lock

  dist_entry_size_ = static_cast<uint32_t>(max_degree() * sizeof(dist_t));

  // Pre-reserve dist_chunks_ to match node_chunks_ capacity so that
  // subsequent emplace_back in ensure_dist_chunk_for never triggers
  // reallocation while concurrent add_node threads read dist_chunks_.
  dist_chunks_.reserve(node_chunks_.capacity());

  // Calculate how many dist chunks we need for existing nodes
  uint32_t total_docs = doc_cnt();
  if (total_docs == 0) {
    dist_loaded_ = true;
    return 0;
  }

  // Check if dist chunks already exist in storage (reopened index)
  size_t existing_dist_chunks =
      broker_->get_chunk_cnt(ChunkBroker::CHUNK_TYPE_NEIGHBOR_DIST);
  if (existing_dist_chunks > 0) {
    // Load existing dist chunks
    for (size_t i = 0; i < existing_dist_chunks; ++i) {
      auto chunk = broker_->get_chunk(ChunkBroker::CHUNK_TYPE_NEIGHBOR_DIST, i);
      if (ailego_unlikely(!chunk)) {
        LOG_ERROR("Failed to load dist chunk %zu", i);
        return IndexError_ReadData;
      }
      dist_chunks_.emplace_back(std::move(chunk));
    }
    LOG_INFO("Loaded %zu existing dist chunks", existing_dist_chunks);
  } else {
    // Allocate new dist chunks for all existing nodes
    int ret = alloc_dist_chunks_for_existing_nodes();
    if (ret != 0) return ret;
  }

  dist_loaded_ = true;
  return 0;
}

int VamanaStreamerEntity::ensure_dist_chunk_for(uint32_t chunk_index) {
  // No-op when dist storage is not active.
  if (!dist_loaded_ || dist_entry_size_ == 0) return 0;

  // Idempotent: nothing to do if this dist chunk slot already exists and is
  // populated. (Slots created by the placeholder loop below will hold
  // nullptr and must still be (re-)allocated.)
  if (chunk_index < dist_chunks_.size() && dist_chunks_[chunk_index]) {
    return 0;
  }

  uint32_t dist_chunk_data_size = node_cnt_per_chunk_ * dist_entry_size_;
  uint32_t dist_chunk_size = AlignPageSize(dist_chunk_data_size);
  auto dp = broker_->alloc_chunk(ChunkBroker::CHUNK_TYPE_NEIGHBOR_DIST,
                                 chunk_index, dist_chunk_size);
  if (ailego_unlikely(dp.first != 0)) {
    LOG_ERROR("Alloc dist chunk %u failed", chunk_index);
    return dp.first;
  }
  dp.second->resize(dist_chunk_data_size);
  {
    // Protect dist_chunks_ modification against concurrent readers in
    // get_neighbor_dists/update_neighbor_dists (called from add_node without
    // mutex_). Uses node_chunks_mutex_ which is the same lock used by
    // sync_chunks for CHUNK_TYPE_NEIGHBOR_DIST.
    std::lock_guard<std::mutex> chunks_lock(node_chunks_mutex_);
    while (dist_chunks_.size() <= chunk_index) {
      dist_chunks_.emplace_back(nullptr);
    }
    dist_chunks_[chunk_index] = std::move(dp.second);
  }
  return 0;
}

int VamanaStreamerEntity::alloc_dist_chunks_for_existing_nodes() {
  uint32_t total_docs = doc_cnt();
  if (total_docs == 0) return 0;

  // Calculate dist chunk size: same number of nodes per chunk as node chunks
  uint32_t dist_chunk_data_size = node_cnt_per_chunk_ * dist_entry_size_;
  uint32_t dist_chunk_size = AlignPageSize(dist_chunk_data_size);

  uint32_t num_chunks_needed =
      (total_docs + node_cnt_per_chunk_ - 1) >> node_index_mask_bits_;

  for (uint32_t i = 0; i < num_chunks_needed; ++i) {
    auto p = broker_->alloc_chunk(ChunkBroker::CHUNK_TYPE_NEIGHBOR_DIST, i,
                                  dist_chunk_size);
    if (ailego_unlikely(p.first != 0)) {
      LOG_ERROR("Alloc dist chunk %u failed", i);
      return p.first;
    }
    // Resize to cover all nodes in this chunk
    uint32_t nodes_in_chunk =
        std::min(node_cnt_per_chunk_, total_docs - i * node_cnt_per_chunk_);
    size_t data_size = static_cast<size_t>(nodes_in_chunk) * dist_entry_size_;
    p.second->resize(data_size);
    dist_chunks_.emplace_back(std::move(p.second));
  }
  broker_->mark_dirty();
  LOG_INFO("Allocated %u dist chunks for %u existing nodes", num_chunks_needed,
           total_docs);
  return 0;
}

const dist_t *VamanaStreamerEntity::get_neighbor_dists(node_id_t id) const {
  if (!dist_loaded_) return nullptr;

  auto loc = get_dist_chunk_loc(id);
  if (ailego_unlikely(loc.first >= dist_chunks_.size())) {
    sync_dist_chunks(loc.first);
  }
  if (ailego_unlikely(loc.first >= dist_chunks_.size())) return nullptr;

  const void *ptr = nullptr;
  size_t ret =
      dist_chunks_[loc.first]->read(loc.second, &ptr, dist_entry_size_);
  if (ailego_unlikely(ret != dist_entry_size_)) return nullptr;
  return static_cast<const dist_t *>(ptr);
}

void VamanaStreamerEntity::update_neighbor_dists(
    node_id_t id, const std::vector<std::pair<node_id_t, dist_t>> &neighbors) {
  if (!dist_loaded_) return;

  auto loc = get_dist_chunk_loc(id);
  // Dist chunk must have been pre-allocated by add_vector or
  // ensure_dist_storage
  if (ailego_unlikely(loc.first >= dist_chunks_.size() ||
                      dist_chunks_[loc.first] == nullptr)) {
    LOG_ERROR("Dist chunk %u not allocated for node %u", loc.first, id);
    return;
  }

  // Write distances: fill max_degree slots, zero-pad unused slots
  uint32_t max_deg = static_cast<uint32_t>(max_degree());
  std::vector<dist_t> dists(max_deg, 0.0f);
  for (size_t i = 0; i < neighbors.size() && i < max_deg; ++i) {
    dists[i] = neighbors[i].second;
  }
  dist_chunks_[loc.first]->write(loc.second, dists.data(), dist_entry_size_);
}

void VamanaStreamerEntity::set_neighbor_dist(node_id_t id, uint32_t idx,
                                             dist_t dist) {
  if (!dist_loaded_) return;

  auto loc = get_dist_chunk_loc(id);
  if (ailego_unlikely(loc.first >= dist_chunks_.size())) return;

  uint32_t offset = loc.second + idx * sizeof(dist_t);
  dist_chunks_[loc.first]->write(offset, &dist, sizeof(dist_t));
}

// calculate_medoid: Compute the medoid (entry point) following DiskANN's
// standard approach:
//   1. Compute the centroid (component-wise mean) of all vectors in float
//   space.
//   2. Find the data point closest to this centroid using squared L2 distance
//      in float space.
//   3. Return that point's node ID as the medoid.
//
// Called at dump time to set the optimal entry point for the persisted index.
// data_type uses IndexMeta::DataType values: DT_FP16=1, DT_FP32=2, DT_INT8=4.
// ============================================================================
node_id_t VamanaStreamerEntity::calculate_medoid(uint32_t dimension,
                                                 uint32_t data_type) {
  uint32_t n = doc_cnt();
  if (n == 0) return kInvalidNodeId;
  if (dimension == 0) return kInvalidNodeId;

  // data_type constants matching IndexMeta::DataType
  constexpr uint32_t DT_FP16 = 1;
  constexpr uint32_t DT_FP32 = 2;
  constexpr uint32_t DT_INT8 = 4;

  if (data_type != DT_FP32 && data_type != DT_INT8 && data_type != DT_FP16) {
    LOG_WARN("calculate_medoid: unsupported data_type=%u, skip", data_type);
    return entry_point();
  }

  // Step 1: Compute centroid (mean) of all vectors in float space.
  std::vector<float> centroid(dimension, 0.0f);
  uint32_t valid_count = 0;

  for (node_id_t i = 0; i < n; ++i) {
    if (get_key(i) == kInvalidKey) continue;
    const void *vec = get_vector(i);
    if (vec == nullptr) continue;

    switch (data_type) {
      case DT_FP32: {
        const float *fv = static_cast<const float *>(vec);
        for (uint32_t d = 0; d < dimension; ++d) centroid[d] += fv[d];
        break;
      }
      case DT_INT8: {
        const int8_t *iv = static_cast<const int8_t *>(vec);
        for (uint32_t d = 0; d < dimension; ++d)
          centroid[d] += static_cast<float>(iv[d]);
        break;
      }
      case DT_FP16: {
        const uint16_t *hv = static_cast<const uint16_t *>(vec);
        for (uint32_t d = 0; d < dimension; ++d)
          centroid[d] += ailego::FloatHelper::ToFP32(hv[d]);
        break;
      }
    }
    valid_count++;
  }

  if (valid_count == 0) return kInvalidNodeId;
  if (valid_count == 1) {
    for (node_id_t i = 0; i < n; ++i) {
      if (get_key(i) != kInvalidKey && get_vector(i) != nullptr) return i;
    }
    return kInvalidNodeId;
  }

  float inv = 1.0f / static_cast<float>(valid_count);
  for (uint32_t d = 0; d < dimension; ++d) centroid[d] *= inv;

  // Step 2: Find the data point closest to the centroid (squared L2).
  node_id_t medoid = kInvalidNodeId;
  float min_dist = std::numeric_limits<float>::max();

  for (node_id_t i = 0; i < n; ++i) {
    if (get_key(i) == kInvalidKey) continue;
    const void *vec = get_vector(i);
    if (vec == nullptr) continue;

    float dist = 0.0f;
    switch (data_type) {
      case DT_FP32: {
        const float *fv = static_cast<const float *>(vec);
        for (uint32_t d = 0; d < dimension; ++d) {
          float diff = fv[d] - centroid[d];
          dist += diff * diff;
        }
        break;
      }
      case DT_INT8: {
        const int8_t *iv = static_cast<const int8_t *>(vec);
        for (uint32_t d = 0; d < dimension; ++d) {
          float diff = static_cast<float>(iv[d]) - centroid[d];
          dist += diff * diff;
        }
        break;
      }
      case DT_FP16: {
        const uint16_t *hv = static_cast<const uint16_t *>(vec);
        for (uint32_t d = 0; d < dimension; ++d) {
          float diff = ailego::FloatHelper::ToFP32(hv[d]) - centroid[d];
          dist += diff * diff;
        }
        break;
      }
    }

    if (dist < min_dist) {
      min_dist = dist;
      medoid = i;
    }
  }

  LOG_INFO("Calculated medoid: node_id=%u, min_sq_dist=%.4f, valid_docs=%u",
           medoid, min_dist, valid_count);
  return medoid;
}

}  // namespace core
}  // namespace zvec
