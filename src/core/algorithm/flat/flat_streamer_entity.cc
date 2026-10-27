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

#include "flat_streamer_entity.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <zvec/core/framework/index_error.h>
#include "utility/batch_topk.h"
#include "utility/legacy_read_buffer.h"
#include "flat_utility.h"

namespace zvec {
namespace core {

int FlatContiguousStreamerEntity::evaluate_distances(
    const ContiguousStorage &storage, const void *query,
    const std::vector<uint64_t> *p_keys, const IndexFilter &filter,
    size_t batch_size, FlatSearchScratch *scratch,
    IndexContext::Stats *context_stats, IndexDocumentHeap *heap) const {
  if (!storage.vector_memory || !query || !scratch || batch_size == 0 ||
      !heap) {
    return IndexError_InvalidArgument;
  }

  auto &vector_ptrs = scratch->vector_ptrs;
  auto &extra_values = scratch->extra_values;
  auto &vector_keys = scratch->vector_keys;
  auto &distances = scratch->distances;
  auto &query_buffer = scratch->query_buffer;
  const size_t extra_size = extra_values_size();
  const bool has_extra_values = extra_size != 0;
  const size_t vector_data_size = meta().element_size() - extra_size;
  const void *batch_query = query;
  if (const auto &preprocess = batch_query_preprocess();
      preprocess != nullptr) {
    const size_t query_size = meta().element_size();
    query_buffer.resize(query_size);
    std::memcpy(query_buffer.data(), query, query_size);
    preprocess(query_buffer.data(), meta().dimension());
    batch_query = query_buffer.data();
  }
  vector_ptrs.clear();
  extra_values.clear();
  vector_keys.clear();
  vector_ptrs.reserve(batch_size);
  if (has_extra_values) extra_values.reserve(batch_size);
  vector_keys.reserve(batch_size);

  auto evaluate_batch = [&]() {
    if (vector_ptrs.empty()) {
      return;
    }
    distances.resize(vector_ptrs.size());
    if (quantizer()) {
      quantizer()->calc_distance_dp_query_batch(
          vector_ptrs.data(), static_cast<int>(vector_ptrs.size()), query,
          distances.data());
    } else {
      const auto &batch_distance_func = batch_distance();
      if (batch_distance_func) {
        batch_distance_func(vector_ptrs.data(), batch_query, vector_ptrs.size(),
                            meta().dimension(), distances.data(),
                            has_extra_values ? extra_values.data() : nullptr);
      } else {
        const auto &distance_func = distance();
        for (size_t i = 0; i < vector_ptrs.size(); ++i) {
          distance_func(query, vector_ptrs[i], meta().dimension(),
                        distances.data() + i);
        }
      }
    }
    if (context_stats) {
      *context_stats->mutable_dist_calced_count() += vector_ptrs.size();
    }
    for (size_t i = 0; i < vector_ptrs.size(); ++i) {
      heap->emplace(vector_keys[i], distances[i]);
    }
    vector_ptrs.clear();
    extra_values.clear();
    vector_keys.clear();
  };

  auto append = [&](uint64_t key, const void *vector) {
    vector_ptrs.push_back(vector);
    if (has_extra_values) {
      extra_values.push_back(static_cast<const char *>(vector) +
                             vector_data_size);
    }
    vector_keys.push_back(key);
    if (vector_ptrs.size() == batch_size) {
      evaluate_batch();
    }
  };

  if (p_keys) {
    for (uint64_t key : *p_keys) {
      if (filter.is_valid() && filter(key)) {
        continue;
      }
      if (const void *vector = get_vector_ptr_by_key(storage, key)) {
        append(key, vector);
      }
    }
  } else {
    for (size_t position = 0; position < storage.vector_count; ++position) {
      const uint64_t vector_key = storage.position_to_key[position];
      if (vector_key == kInvalidKey ||
          (filter.is_valid() && filter(vector_key))) {
        if (context_stats) {
          ++(*context_stats->mutable_filtered_count());
        }
        continue;
      }
      const void *vector = get_vector_ptr(storage, position);
      if (!vector) {
        if (context_stats) {
          ++(*context_stats->mutable_filtered_count());
        }
        continue;
      }
      append(vector_key, vector);
    }
  }
  evaluate_batch();
  return 0;
}

FlatStreamerEntity::FlatStreamerEntity(IndexStreamer::Stats &stats)
    : stats_(stats) {}

int FlatStreamerEntity::open(IndexStorage::Pointer storage,
                             const IndexMeta & /*mt*/) {
  if (storage_) {
    LOG_ERROR("An storage instance is already opened");
    return IndexError_Duplicate;
  }
  // segments_[0] store the meta information of the linear list
  ailego_assert_with(segments_.size() == 0, "Invalid Size");

  all_keys_valid_ = std::make_shared<std::atomic<bool>>(true);
  key_info_map_lock_ = std::make_shared<ailego::SharedMutex>();
  key_info_map_.clear();
  id_key_vector_.clear();
  withid_key_info_map_.clear();
  withid_key_map_.clear();

  vec_unit_size_ = IndexMeta::AlignSizeof(index_meta_.data_type());
  vec_cols_ = index_meta_.element_size() / vec_unit_size_;
  meta_.header.block_size =
      ailego_align(sizeof(BlockHeader) + sizeof(DeletionMap) +
                       (index_meta_.element_size() + sizeof(uint64_t)) *
                           meta_.header.block_vector_count,
                   32);

  if (storage->get(FLAT_LINEAR_LIST_HEAD_SEG_ID) ||
      storage->get(FLAT_LINEAR_META_SEG_ID)) {
    int ret = this->load_storage(storage);
    if (ailego_unlikely(ret != 0)) {
      LOG_ERROR("Failed to load storage index");
      return ret;
    }
  } else {
    int ret = this->init_storage(storage);
    if (ailego_unlikely(ret != 0)) {
      LOG_ERROR("Failed to init storage");
      return ret;
    }
  }

  storage_ = storage;

  //! Create the distance calculator
  if (!quantizer_) {
    auto metric = IndexFactory::CreateMetric(index_meta_.metric_name());
    if (!metric) {
      LOG_ERROR("Failed to create metric %s",
                index_meta_.metric_name().c_str());
      return IndexError_NoExist;
    }
    int ret = metric->init(index_meta_, index_meta_.metric_params());
    if (ret != 0) {
      LOG_ERROR("Failed to initialize metric %s",
                index_meta_.metric_name().c_str());
      return ret;
    }
    row_distance_ = metric->distance();
    column_distance_ =
        metric->distance_matrix(meta_.header.block_vector_count, 1);
    batch_distance_ = metric->batch_distance();
    batch_query_preprocess_ = metric->get_query_preprocess_func();
    extra_values_size_ = metric->extra_values_size_per_vector();
    if (extra_values_size_ != 0 &&
        extra_values_size_ >= index_meta_.element_size()) {
      LOG_ERROR("Invalid Flat vector layout, vector_size=%u extra_size=%zu",
                index_meta_.element_size(), extra_values_size_);
      return IndexError_InvalidArgument;
    }
  }

  LOG_DEBUG("Open storage %s done, metric=%s", storage_->name().c_str(),
            index_meta_.metric_name().c_str());

  return 0;
}

int FlatStreamerEntity::close() {
  segments_.clear();
  storage_.reset();
  all_keys_valid_.reset();
  key_info_map_lock_.reset();
  key_info_map_.clear();
  withid_key_info_map_.clear();
  withid_key_map_.clear();
  id_key_vector_.clear();
  meta_.create_time = 0;
  meta_.update_time = 0;
  meta_.segment_count = 0;
  meta_.header.total_vector_count = 0;
  meta_.header.block_count = 0;
  meta_.header.block_size = 0;
  meta_.header.linear_body_size = 0;
  batch_query_preprocess_ = nullptr;
  extra_values_size_ = 0;
  meta_dirty_ = false;

  return 0;
}

int FlatStreamerEntity::flush_linear_meta() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!storage_ || !meta_dirty_) {
    return 0;
  }

  meta_.update_time = ailego::Realtime::Seconds();
  meta_.revision_id = stats_.revision_id();
  stats_.set_update_time(meta_.update_time);
  auto segment = storage_->get(FLAT_LINEAR_META_SEG_ID);
  if (ailego_unlikely(!segment)) {
    LOG_ERROR("Failed to get segment %s", FLAT_LINEAR_META_SEG_ID.c_str());
    return IndexError_Runtime;
  }
  if (segment->write(0, &meta_, sizeof(meta_)) != sizeof(meta_)) {
    LOG_ERROR("Failed to write segment %s", FLAT_LINEAR_META_SEG_ID.c_str());
    return IndexError_WriteData;
  }

  meta_dirty_ = false;

  return 0;
}

int FlatStreamerEntity::flush(uint64_t checkpoint) {
  int ret = this->flush_linear_meta();
  if (ailego_unlikely(ret != 0)) {
    return ret;
  }

  if (checkpoint != 0) {
    storage_->refresh(checkpoint);
  }
  ret = storage_->flush();
  if (ailego_unlikely(ret != 0)) {
    LOG_ERROR("Failed to refresh storage for %s", IndexError::What(ret));
    return ret;
  }
  if (checkpoint != 0) {
    stats_.set_check_point(checkpoint);
  }

  return 0;
}

int FlatStreamerEntity::add(uint64_t key, const void *vec, size_t size) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (filter_same_key_) {
    key_info_map_lock_->lock_shared();
    if (key_info_map_.find(key) != key_info_map_.end()) {
      key_info_map_lock_->unlock_shared();
      LOG_WARN("Try to add duplicate key, drop it");
      return IndexError_Duplicate;
    }
    key_info_map_lock_->unlock_shared();
  }
  if (size != static_cast<size_t>(index_meta_.element_size())) {
    LOG_ERROR("Failed to add, mismatch size %zu vs elemsize %u", size,
              index_meta_.element_size());
    return IndexError_Mismatch;
  }

  IndexStorage::MemoryBlock head_block;
  this->get_head_block(head_block);
  BlockLocation block;
  {
    const BlockLocation *bl =
        reinterpret_cast<const BlockLocation *>(head_block.data());
    if (ailego_unlikely(bl == nullptr)) {
      LOG_ERROR("Failed to get block loc");
      return IndexError_ReadData;
    }
    block = *bl;
  }
  // Release the head block reference early so that the buffer pool ref_count
  // and memory budget held by it do not block subsequent acquire/evict in this
  // function (alloc_block / add_to_block may compete for the same memory).
  head_block.reset(nullptr);

  meta_dirty_ = true;
  if (!this->is_valid_block(block)) {
    int ret = this->alloc_block(block, &block);
    if (ailego_unlikely(ret != 0)) {
      return ret;
    }
    ret = this->update_head_block(block);
    if (ailego_unlikely(ret != 0)) {
      return ret;
    }
  }

  int ret = this->add_to_block(block, key, vec, size);
  if (ret == IndexError_IndexFull) {
    ret = this->alloc_block(block, &block);
    if (ailego_unlikely(ret != 0)) {
      return ret;
    }
    ret = this->update_head_block(block);
    if (ailego_unlikely(ret != 0)) {
      return ret;
    }
    ret = this->add_to_block(block, key, vec, size);
    if (ailego_unlikely(ret != 0)) {
      return ret;
    }
  }
  if (ailego_unlikely(ret != 0)) {
    return ret;
  }

  (*stats_.mutable_added_count())++;
  stats_.set_revision_id(meta_.revision_id + 1);

  return 0;
}

int FlatStreamerEntity::search(const void *query, const IndexFilter &filter,
                               uint32_t *scan_count, IndexDocumentHeap *heap,
                               IndexContext::Stats *context_stats,
                               FlatSearchScratch * /*scratch*/,
                               size_t /*batch_size*/) const {
  const bool batch_rows =
      quantizer_ && index_meta_.major_order() == IndexMeta::MO_ROW &&
      block_vector_count() > 0 && block_vector_count() <= kDistanceBatchVectors;
  // Flat blocks have keys, a deletion map and a header after their rows.
  // Keep physical block boundaries when gathering the vector body.
  if (batch_rows && !filter.is_valid() && all_keys_valid_->load() &&
      heap->limit() > 0 &&
      heap->limit() != std::numeric_limits<size_t>::max()) {
    return search_rows_all_valid(query, scan_count, heap, context_stats);
  }
  // Filtered/deleted rows, unlimited results and other layouts retain the
  // original scan without an additional O(n) candidate buffer.
  return search_blocks(query, filter, scan_count, heap, context_stats);
}

int FlatStreamerEntity::search_rows_all_valid(
    const void *query, uint32_t *scan_count, IndexDocumentHeap *heap,
    IndexContext::Stats *context_stats) const {
  // Dispatch once within this sub-path. Quantizer owns distance-kernel
  // selection for the pointer batch interface.
  using Impl = BatchTopK::Implementation;
  switch (BatchTopK::implementation()) {
    case Impl::kAVX512:
      return search_rows_all_valid_impl<BatchTopKImpl<Impl::kAVX512>>(
          query, scan_count, heap, context_stats);
    case Impl::kAVX2:
      return search_rows_all_valid_impl<BatchTopKImpl<Impl::kAVX2>>(
          query, scan_count, heap, context_stats);
    case Impl::kSSE2:
      return search_rows_all_valid_impl<BatchTopKImpl<Impl::kSSE2>>(
          query, scan_count, heap, context_stats);
    case Impl::kNEON:
      return search_rows_all_valid_impl<BatchTopKImpl<Impl::kNEON>>(
          query, scan_count, heap, context_stats);
    default:
      return search_rows_all_valid_impl<BatchTopKImpl<Impl::kScalar>>(
          query, scan_count, heap, context_stats);
  }
}

int FlatStreamerEntity::search_blocks(
    const void *query, const IndexFilter &filter, uint32_t *scan_count,
    IndexDocumentHeap *heap, IndexContext::Stats *context_stats) const {
  IndexStorage::MemoryBlock head_block;
  this->get_head_block(head_block);
  const BlockLocation *bl =
      reinterpret_cast<const BlockLocation *>(head_block.data());
  if (ailego_unlikely(bl == nullptr)) {
    LOG_ERROR("Failed to get block loc");
    return IndexError_ReadData;
  }

  BlockLocation block = *bl;

  while (this->is_valid_block(block)) {
    IndexStorage::MemoryBlock block_header_block;
    this->get_block_header(block, block_header_block);
    const BlockHeader *hd =
        reinterpret_cast<const BlockHeader *>(block_header_block.data());
    if (ailego_unlikely(hd == nullptr)) {
      LOG_ERROR("Failed to get block header");
      return IndexError_ReadData;
    }

    if (hd->vector_count > 0) {
      *scan_count += hd->vector_count;
      IndexStorage::MemoryBlock deletion_map_block;
      this->get_block_deletion_map(block, deletion_map_block);
      const DeletionMap *deletion_map =
          reinterpret_cast<const DeletionMap *>(deletion_map_block.data());
      if (filter.is_valid() || deletion_map->is_dirty()) {
        this->search_block(query, block, hd, 1.0, filter, deletion_map, heap,
                           context_stats);
      } else {
        *(context_stats->mutable_dist_calced_count()) += hd->vector_count;
        this->search_block(query, block, hd, 1.0, heap);
      }
    }
    block = hd->next;
  }
  return 0;
}

template <typename TopK>
int FlatStreamerEntity::search_rows_all_valid_impl(
    const void *query, uint32_t *scan_count, IndexDocumentHeap *heap,
    IndexContext::Stats *context_stats) const {
  BlockLocation block;
  {
    IndexStorage::MemoryBlock head;
    if (get_head_block(head) != 0 || !head.data()) return IndexError_ReadData;
    block = *static_cast<const BlockLocation *>(head.data());
  }
  // A clone may need to populate its cache after the writer adds a segment.
  if (!get_segment(block.segment_id)) return IndexError_ReadData;
  std::vector<IndexStorage::Segment::Pointer> segments;
  {
    std::lock_guard<std::mutex> lock(segments_mutex_);
    segments = segments_;
  }
  const size_t block_size = linear_block_size();
  const size_t block_rows = block_vector_count();
  const size_t stride = index_meta_.element_size();
  // Physical slot IDs remain reversible even when segment sizes differ after
  // reopening. Only segment boundaries are stored, not per-vector keys/offsets.
  std::vector<uint64_t> offsets(segments.size() + 1, 0);
  for (size_t i = 1; i < segments.size(); ++i) {
    offsets[i + 1] =
        offsets[i] + segments[i]->data_size() / block_size * block_rows;
  }
  if (offsets.back() > uint64_t{1} + UINT32_MAX) {
    return search_blocks(query, IndexFilter{}, scan_count, heap, context_stats);
  }

  float threshold = heap->threshold();
  bool strict_threshold = false;
  if (heap->full() && heap->front().score() <= threshold) {
    threshold = heap->front().score();
    strict_threshold = true;
  }
  TopK topk;
  topk.reset(std::min<size_t>(heap->limit(), offsets.back()), threshold,
             strict_threshold);
  alignas(AILEGO_CACHE_LINE_SIZE) std::array<float, kDistanceBatchVectors>
      distances;
  std::array<const void *, kDistanceBatchVectors> candidates;
  uint32_t scanned = 0;
  while (is_valid_block(block)) {
    IndexStorage::MemoryBlock data;
    if (block.segment_id >= segments.size() ||
        segments[block.segment_id]->read(get_block_offset(block.block_index),
                                         data, block_size) != block_size) {
      return IndexError_ReadData;
    }
    const auto *vectors = static_cast<const char *>(data.data());
    const auto header = *reinterpret_cast<const BlockHeader *>(
        vectors + block_size - sizeof(BlockHeader));
    const size_t rows = header.vector_count;
    if (rows > block_rows) return IndexError_ReadData;
    if (rows > 0) {
      const uint32_t first_id =
          offsets[block.segment_id] + uint64_t{block.block_index} * block_rows;
      for (size_t i = 0; i < rows; ++i) {
        candidates[i] = vectors + i * stride;
      }
      quantizer_->calc_distance_dp_query_batch(
          candidates.data(), static_cast<int>(rows), query, distances.data());
      topk.push_block(distances.data(), rows, first_id);
      scanned += rows;
    }
    block = header.next;
  }
  // An append may have published invalid gap rows during the scan. No output
  // has been committed yet, so the original path can safely redo this search.
  if (!all_keys_valid_->load()) {
    return search_blocks(query, IndexFilter{}, scan_count, heap, context_stats);
  }
  topk.finish();
  for (size_t i = 0; i < topk.size(); ++i) {
    const auto &candidate = topk.data()[i];
    const size_t segment_id =
        std::upper_bound(offsets.begin() + 1, offsets.end(), candidate.id) -
        offsets.begin() - 1;
    const uint32_t slot = candidate.id - offsets[segment_id];
    IndexStorage::MemoryBlock key_block;
    uint64_t key;
    if (segments[segment_id]->read(
            get_block_key_offset(slot / block_rows, slot % block_rows),
            key_block, sizeof(key)) != sizeof(key)) {
      return IndexError_ReadData;
    }
    std::memcpy(&key, key_block.data(), sizeof(key));
    heap->emplace(key, candidate.score);
  }
  *scan_count += scanned;
  *context_stats->mutable_dist_calced_count() += scanned;
  return 0;
}

int FlatStreamerEntity::search_by_p_keys(const void *query,
                                         const std::vector<uint64_t> &p_keys,
                                         const IndexFilter &filter,
                                         IndexDocumentHeap *heap,
                                         FlatSearchScratch * /*scratch*/,
                                         size_t /*batch_size*/) const {
  for (uint64_t key : p_keys) {
    if (filter.is_valid() && filter(key)) {
      continue;
    }
    IndexStorage::MemoryBlock block;
    if (get_vector_by_key(key, block) != 0) {
      continue;
    }
    dist_t distance = 0;
    row_major_distance(query, block.data(), 1, &distance);
    heap->emplace(key, distance);
  }
  return 0;
}

//! Search in a block
void FlatStreamerEntity::search_block(const void *query,
                                      const BlockLocation &bl,
                                      const BlockHeader *hd, float norm_val,
                                      IndexDocumentHeap *heap) const {
  std::vector<float> distances(block_vector_count());
  IndexStorage::MemoryBlock vecs_block;
  this->get_block_vectors(bl, vecs_block);
  const char *vecs = reinterpret_cast<const char *>(vecs_block.data());
  IndexStorage::MemoryBlock keys_block;
  this->get_block_keys(bl, keys_block);
  const uint64_t *keys = reinterpret_cast<const uint64_t *>(keys_block.data());
  row_major_distance(query, vecs, hd->vector_count, distances.data());
  for (size_t k = 0; k < hd->vector_count; ++k) {
    if (keys[k] != kInvalidKey) {
      heap->emplace(keys[k], distances[k] * norm_val);
    }
  }
}

//! Search in a block with filter
void FlatStreamerEntity::search_block(
    const void *query, const BlockLocation &bl, const BlockHeader *hd,
    float norm_val, const IndexFilter &filter, const DeletionMap *deletion_map,
    IndexDocumentHeap *heap, IndexContext::Stats *context_stats) const {
  std::vector<float> distances(block_vector_count());

  IndexStorage::MemoryBlock vecs_block;
  this->get_block_vectors(bl, vecs_block);
  const char *vecs = reinterpret_cast<const char *>(vecs_block.data());
  IndexStorage::MemoryBlock keys_block;
  this->get_block_keys(bl, keys_block);
  const uint64_t *keys = reinterpret_cast<const uint64_t *>(keys_block.data());

  DeletionMap keeps;
  for (size_t k = 0; k < hd->vector_count; ++k) {
    const bool condition1 = !deletion_map->test(k);
    const bool condition2 = filter.is_valid() ? !filter(keys[k]) : true;
    const bool condition3 = keys[k] != kInvalidKey;
    if (condition1 && condition2 && condition3) {
      keeps.set(k);
    }
  }
  if (!keeps.is_dirty()) {
    (*context_stats->mutable_filtered_count()) += hd->vector_count;
    return;
  }
  for (size_t k = 0; k < hd->vector_count; ++k) {
    if (keeps.test(k)) {
      auto cur_vec = vecs + index_meta_.element_size() * k;
      row_major_distance(query, cur_vec, 1, distances.data() + k);
      ++(*context_stats->mutable_dist_calced_count());
    }
  }
  for (size_t k = 0; k < hd->vector_count; ++k) {
    if (keeps.test(k)) {
      heap->emplace(keys[k], distances[k] * norm_val);
    } else {
      ++(*context_stats->mutable_filtered_count());
    }
  }
}

int FlatStreamerEntity::search_bf(const void *query, const IndexFilter &filter,
                                  IndexDocumentHeap *heap,
                                  IndexContext::Stats *context_stats) const {
  uint32_t scan_count = 0;
  return this->search(query, filter, &scan_count, heap, context_stats);
}

FlatStreamerEntity::Pointer FlatStreamerEntity::clone() const {
  std::vector<IndexStorage::Segment::Pointer> segments;
  {
    std::lock_guard<std::mutex> lock(segments_mutex_);
    segments = segments_;
  }
  // Storage operations do not need to hold the cache lock.
  for (size_t i = 0; i < segments.size(); ++i) {
    segments[i] = segments[i]->clone();
    if (!segments[i]) {
      LOG_ERROR("Failed to clone segment, index=%zu", i);
      return nullptr;
    }
  }
  auto entity = new (std::nothrow) FlatStreamerEntity(stats_);
  if (!entity) {
    LOG_ERROR("Failed to New FlatStreamerEntity object");
    return nullptr;
  }
  entity->index_meta_ = this->index_meta_;
  entity->storage_ = this->storage_;
  // entity->reformer_ = this->reformer_;
  entity->segments_ = std::move(segments);
  entity->meta_ = this->meta_;
  entity->quantizer_ = this->quantizer_;
  entity->all_keys_valid_ = this->all_keys_valid_;
  entity->key_info_map_lock_ = this->key_info_map_lock_;
  entity->key_info_map_ = this->key_info_map_;
  entity->id_key_vector_ = this->id_key_vector_;
  entity->withid_key_info_map_ = this->withid_key_info_map_;
  entity->withid_key_map_ = this->withid_key_map_;
  entity->filter_same_key_ = this->filter_same_key_;
  entity->vec_unit_size_ = this->vec_unit_size_;
  entity->vec_cols_ = this->vec_cols_;
  return FlatStreamerEntity::Pointer(entity);
}

int FlatStreamerEntity::get_vector_by_position(
    uint32_t id, IndexStorage::MemoryBlock &block) const {
  if (id >= withid_key_info_map_.size()) {
    return -1;
  }
  const VectorLocation &loc = withid_key_info_map_[id];
  auto segment = this->get_segment(loc.segment_id);
  if (!segment ||
      segment->read(loc.offset, block, index_meta_.element_size()) !=
          index_meta_.element_size()) {
    LOG_ERROR("Failed to read vector by position, id=%u size=%u", id,
              index_meta_.element_size());
    return -1;
  }
  return 0;
}

int FlatStreamerEntity::get_key_by_position(uint32_t id, uint64_t *key) const {
  if (!key || id >= withid_key_info_map_.size() ||
      id >= withid_key_map_.size()) {
    return -1;
  }
  const auto &loc = withid_key_info_map_[id];
  auto segment = get_segment(loc.segment_id);
  IndexStorage::MemoryBlock key_block;
  if (!segment ||
      segment->read(withid_key_map_[id], key_block, sizeof(*key)) !=
          sizeof(*key) ||
      !key_block.data()) {
    LOG_ERROR("Failed to read Flat key by position, id=%u", id);
    return -1;
  }
  std::memcpy(key, key_block.data(), sizeof(*key));
  return 0;
}

int FlatContiguousStreamerEntity::search(
    const void *query, const IndexFilter &filter, uint32_t *scan_count,
    IndexDocumentHeap *heap, IndexContext::Stats *context_stats,
    FlatSearchScratch *scratch, size_t batch_size) const {
  auto storage = load_contiguous_storage();
  if (!storage) {
    return FlatStreamerEntity::search(query, filter, scan_count, heap,
                                      context_stats, scratch, batch_size);
  }

  FlatSearchScratch local_scratch;
  if (!scratch) {
    scratch = &local_scratch;
  }
  if (batch_size == 0) {
    batch_size = block_vector_count();
  }
  if (scan_count) {
    *scan_count += storage->vector_count;
  }
  return evaluate_distances(*storage, query, nullptr, filter, batch_size,
                            scratch, context_stats, heap);
}

int FlatContiguousStreamerEntity::search_by_p_keys(
    const void *query, const std::vector<uint64_t> &p_keys,
    const IndexFilter &filter, IndexDocumentHeap *heap,
    FlatSearchScratch *scratch, size_t batch_size) const {
  auto storage = load_contiguous_storage();
  if (!storage) {
    return FlatStreamerEntity::search_by_p_keys(query, p_keys, filter, heap,
                                                scratch, batch_size);
  }

  FlatSearchScratch local_scratch;
  if (!scratch) {
    scratch = &local_scratch;
  }
  if (batch_size == 0) {
    batch_size = block_vector_count();
  }
  return evaluate_distances(*storage, query, &p_keys, filter, batch_size,
                            scratch, nullptr, heap);
}

int FlatStreamerEntity::search_by_p_keys_fast(
    const void * /*query*/, const std::vector<uint64_t> & /*keys*/,
    int64_t * /*output_ids*/, float * /*output_scores*/, size_t /*topk*/,
    FlatSearchScratch * /*scratch*/) const {
  return IndexError_NotImplemented;
}

int FlatContiguousStreamerEntity::search_by_p_keys_fast(
    const void *query, const std::vector<uint64_t> &keys, int64_t *output_ids,
    float *output_scores, size_t topk, FlatSearchScratch *scratch) const {
  if (!query || !output_ids || topk == 0 || !scratch) {
    return IndexError_InvalidArgument;
  }
  // Hold the same immutable generation for pointer lookup and evaluation.
  // A concurrent degrade/add cannot free its rows while this call is active.
  auto storage = load_contiguous_storage();
  if (!storage || !storage->vector_memory || keys.size() < topk ||
      (quantizer() &&
       keys.size() > static_cast<size_t>(std::numeric_limits<int>::max()))) {
    return IndexError_NotImplemented;
  }
  auto &ptrs = scratch->vector_ptrs;
  auto &extras = scratch->extra_values;
  auto &distances = scratch->distances;
  const size_t count = keys.size();
  const size_t extra_size = extra_values_size();
  const bool has_extras = extra_size != 0;
  const size_t vector_data_size = meta().element_size() - extra_size;
  ptrs.resize(count);
  extras.resize(has_extras ? count : 0);
  distances.resize(count);
  for (size_t i = 0; i < count; ++i) {
    ptrs[i] = get_vector_ptr_by_key(*storage, keys[i]);
    if (!ptrs[i]) return IndexError_NotImplemented;
    if (has_extras) {
      extras[i] = static_cast<const char *>(ptrs[i]) + vector_data_size;
    }
  }
  // Use the same prepared-query dispatch as the regular Flat search.
  if (quantizer()) {
    quantizer()->calc_distance_dp_query_batch(
        ptrs.data(), static_cast<int>(count), query, distances.data());
  } else {
    const void *batch_query = query;
    if (const auto &preprocess = batch_query_preprocess();
        preprocess != nullptr) {
      auto &buffer = scratch->query_buffer;
      buffer.resize(meta().element_size());
      std::memcpy(buffer.data(), query, buffer.size());
      preprocess(buffer.data(), meta().dimension());
      batch_query = buffer.data();
    }
    if (const auto &batch = batch_distance(); batch) {
      batch(ptrs.data(), batch_query, count, meta().dimension(),
            distances.data(), has_extras ? extras.data() : nullptr);
    } else {
      for (size_t i = 0; i < count; ++i) {
        distance()(query, ptrs[i], meta().dimension(), distances.data() + i);
      }
    }
  }
  // Use the same heap operations and score-only ordering as IndexDocumentHeap,
  // with lightweight key/score entries rather than full result documents.
  ailego::KeyValueHeap<uint64_t, float> documents(topk);
  for (size_t i = 0; i < count; ++i) {
    // Match the regular heap: NaN and positive overflow are not candidates.
    // Let that path decide the result count without modifying caller output.
    if (!(distances[i] <= std::numeric_limits<float>::max())) {
      return IndexError_NotImplemented;
    }
    documents.emplace(keys[i], distances[i]);
  }
  documents.sort();
  for (size_t i = 0; i < topk; ++i) {
    output_ids[i] = static_cast<int64_t>(documents[i].first);
    if (output_scores) output_scores[i] = documents[i].second;
  }
  return 0;
}

int FlatContiguousStreamerEntity::build_contiguous_memory() {
  degrade_to_mmap();

  const size_t count = use_key_info_map() ? id_key_count() : vector_count();
  if (count == 0) {
    return 0;
  }

  const size_t vector_size = meta().element_size();
  const size_t stride =
      (vector_size + kVectorAlignment - 1) & ~(kVectorAlignment - 1);
  const size_t data_size = count * stride;
  const size_t allocation_size =
      ailego::MemoryHelper::AlignHugePageSize(data_size);
  char *raw = static_cast<char *>(
      ailego::MemoryHelper::AllocateHugePage(allocation_size));
  if (!raw) {
    LOG_ERROR("Failed to allocate Flat contiguous memory, size=%zu",
              allocation_size);
    return IndexError_Runtime;
  }

  auto storage = std::make_shared<ContiguousStorage>();
  storage->vector_memory =
      std::shared_ptr<char>(raw, ContiguousDeleter{allocation_size});
  storage->vector_stride = stride;
  storage->vector_count = count;
  if (use_key_info_map()) {
    storage->key_to_position.reserve(count);
  }
  storage->position_to_key.assign(count, static_cast<uint64_t>(kInvalidKey));

  for (uint32_t id = 0; id < count; ++id) {
    IndexStorage::MemoryBlock block;
    uint64_t vector_key = kInvalidKey;
    int ret = 0;
    if (use_key_info_map()) {
      vector_key = key(id);
      ret = FlatStreamerEntity::get_vector_by_key(vector_key, block);
      if (ret == 0) {
        storage->key_to_position[vector_key] = id;
      }
    } else {
      ret = get_vector_by_position(id, block);
      if (ret == 0) {
        ret = get_key_by_position(id, &vector_key);
      }
    }
    if (ret != 0 || block.data() == nullptr) {
      LOG_ERROR("Failed to load Flat vector into contiguous memory, id=%u", id);
      return IndexError_ReadData;
    }
    std::memcpy(raw + static_cast<size_t>(id) * stride, block.data(),
                vector_size);
    storage->position_to_key[id] = vector_key;
  }

  std::atomic_store_explicit(
      &contiguous_storage_,
      std::shared_ptr<const ContiguousStorage>(std::move(storage)),
      std::memory_order_release);
  LOG_INFO(
      "Built Flat contiguous memory: vectors=%zu vector_size=%zu "
      "vector_stride=%zu memory=%zu",
      count, vector_size, stride, allocation_size);
  return 0;
}

void FlatContiguousStreamerEntity::degrade_to_mmap() {
  std::shared_ptr<const ContiguousStorage> empty;
  auto storage = std::atomic_exchange_explicit(
      &contiguous_storage_, std::move(empty), std::memory_order_acq_rel);
  if (storage) {
    LOG_INFO("Flat contiguous entity degraded to mmap mode for insertion");
  }
}

int FlatContiguousStreamerEntity::close() {
  degrade_to_mmap();
  return FlatStreamerEntity::close();
}

int FlatContiguousStreamerEntity::add(uint64_t key, const void *vec,
                                      size_t size) {
  degrade_to_mmap();
  return FlatStreamerEntity::add(key, vec, size);
}

int FlatContiguousStreamerEntity::add_vector_with_id(uint32_t id,
                                                     const void *query,
                                                     uint32_t element_size) {
  degrade_to_mmap();
  return FlatStreamerEntity::add_vector_with_id(id, query, element_size);
}

const void *FlatContiguousStreamerEntity::get_vector_ptr_by_key(
    const ContiguousStorage &storage, uint64_t key) const {
  size_t position = key;
  if (use_key_info_map()) {
    auto it = storage.key_to_position.find(key);
    if (it == storage.key_to_position.end()) {
      return nullptr;
    }
    position = it->second;
  } else if (key >= storage.position_to_key.size() ||
             storage.position_to_key[key] != key) {
    return nullptr;
  }
  return get_vector_ptr(storage, position);
}

const void *FlatContiguousStreamerEntity::get_vector_ptr(
    const ContiguousStorage &storage, size_t position) const {
  if (!storage.vector_memory || position >= storage.vector_count) {
    return nullptr;
  }
  return storage.vector_memory.get() + position * storage.vector_stride;
}

const void *FlatStreamerEntity::get_vector_by_key(uint64_t key) const {
  // Keep the legacy pointer valid until the next call on this thread.
  static thread_local LegacyReadBuffer buffer;
  IndexStorage::MemoryBlock block;
  return get_vector_by_key(key, block) == 0
             ? buffer.assign(block, index_meta_.element_size())
             : nullptr;
}

int FlatStreamerEntity::get_vector_by_key(
    const uint64_t key, IndexStorage::MemoryBlock &block) const {
  VectorLocation loc{};
  key_info_map_lock_->lock_shared();
  if (use_key_info_map_) {
    auto iterator = key_info_map_.find(key);
    if (iterator == key_info_map_.end()) {
      key_info_map_lock_->unlock_shared();
      return -1;
    }
    loc = iterator->second;
  } else {
    if (key < withid_key_info_map_.size()) {
      loc = withid_key_info_map_[key];
    } else {
      key_info_map_lock_->unlock_shared();
      return -1;
    }
  }
  key_info_map_lock_->unlock_shared();

  auto segment = this->get_segment(loc.segment_id);
  if (segment->read(loc.offset, block, index_meta_.element_size()) !=
      index_meta_.element_size()) {
    LOG_ERROR("Failed to read segment, size=%u", index_meta_.element_size());
    return -1;
  }
  return 0;
}

int FlatStreamerEntity::get_vectors_by_key(
    const uint64_t *keys, uint32_t count,
    std::vector<IndexStorage::MemoryBlock> &blocks) const {
  std::vector<VectorLocation> locations(count);
  key_info_map_lock_->lock_shared();
  if (use_key_info_map_) {
    for (uint32_t i = 0; i < count; ++i) {
      auto iterator = key_info_map_.find(keys[i]);
      if (iterator == key_info_map_.end()) {
        key_info_map_lock_->unlock_shared();
        return -1;
      }
      locations[i] = iterator->second;
    }
  } else {
    for (uint32_t i = 0; i < count; ++i) {
      if (keys[i] >= withid_key_info_map_.size()) {
        key_info_map_lock_->unlock_shared();
        return -1;
      }
      locations[i] = withid_key_info_map_[keys[i]];
    }
  }
  key_info_map_lock_->unlock_shared();

  std::vector<IndexStorage::Segment::Pointer> segments(count);
  {
    std::lock_guard<std::mutex> lock(segments_mutex_);
    for (uint32_t i = 0; i < count; ++i) {
      const size_t segment_index = locations[i].segment_id;
      for (size_t j = segments_.size(); j <= segment_index; ++j) {
        auto segment_id =
            ailego::StringHelper::Concat(FLAT_SEGMENT_FEATURES_SEG_ID, j);
        auto segment = storage_->get(segment_id);
        if (!segment) {
          LOG_ERROR("Failed to get segment %s", segment_id.c_str());
          return -1;
        }
        segments_.emplace_back(std::move(segment));
      }
      segments[i] = segments_[segment_index];
    }
  }

  blocks.resize(count);
  for (uint32_t i = 0; i < count; ++i) {
    const VectorLocation &loc = locations[i];
    const auto &segment = segments[i];
    if (!segment ||
        segment->read(loc.offset, blocks[i], index_meta_.element_size()) !=
            index_meta_.element_size()) {
      LOG_ERROR("Failed to read segment, size=%u", index_meta_.element_size());
      return -1;
    }
  }
  return 0;
}

IndexProvider::Iterator::Pointer FlatStreamerEntity::creater_iterator() const {
  auto entity = this->clone();
  if (!entity) {
    LOG_ERROR("Failed to clone entity");
    return nullptr;
  }

  return Iterator::Pointer(new (std::nothrow)
                               FlatStreamerEntity::Iterator(std::move(entity)));
}

void FlatStreamerEntity::Iterator::read_next_block() {
  auto block_size = entity_->linear_block_size();
  while (segment_id_ < entity_->segments_.size()) {
    auto &segment = entity_->segments_[segment_id_];
    size_t off = block_index_ * block_size;
    if (off + block_size > segment->data_size()) {
      ++segment_id_;
      block_index_ = 0;
      continue;
    }
    if (segment->read(off, block_, block_size) != block_size) {
      LOG_ERROR("Failed to read block, off=%zu", off);
      break;
    }
    data_ = block_.data();
    auto hd = reinterpret_cast<const BlockHeader *>(
        static_cast<const char *>(data_) + block_size - sizeof(BlockHeader));
    if (hd->vector_count == 0) {
      ++block_index_;
      continue;
    }

    block_vector_count_ = hd->vector_count;
    block_vector_index_ = 0;
    size_t elemsize = entity_->index_meta_.element_size();
    keys_ = reinterpret_cast<const uint64_t *>(
        reinterpret_cast<const char *>(data_) +
        elemsize * entity_->block_vector_count());
    return;
  }

  is_valid_ = false;
}

int FlatStreamerEntity::init_storage(IndexStorage::Pointer storage) {
  meta_dirty_ = true;
  // Init Linear Meta Segment
  meta_.create_time = ailego::Realtime::Seconds();
  stats_.set_create_time(meta_.create_time);
  meta_.update_time = ailego::Realtime::Seconds();
  stats_.set_update_time(meta_.update_time);
  meta_.segment_count = 0;
  meta_.revision_id = 0;

  std::string str;
  index_meta_.serialize(&str);
  const size_t page = ailego::MemoryHelper::PageSize();

  meta_.header.header_size = sizeof(LinearIndexHeader) + str.size();
  meta_.header.total_vector_count = 0;
  meta_.header.linear_body_size = 0;
  meta_.header.block_count = 0;
  meta_.header.index_meta_size = str.size();
  meta_.header.linear_list_count = 1;

  adjust_segment_size(&meta_);

  LOG_DEBUG(
      "Create Streamer Index, VecSize=%u, BlockSize=%u SegmentSize=%u "
      "LinearListCount=%u",
      index_meta_.element_size(), meta_.header.block_size, meta_.segment_size,
      meta_.header.linear_list_count);

  size_t size = ailego_align(sizeof(meta_) + str.size(), page);
  int ret = storage->append(FLAT_LINEAR_META_SEG_ID, size);
  if (ailego_unlikely(ret != 0)) {
    LOG_ERROR("Failed to append segment %s", FLAT_LINEAR_META_SEG_ID.c_str());
    return ret;
  }
  auto segment = storage->get(FLAT_LINEAR_META_SEG_ID);
  if (ailego_unlikely(!segment)) {
    LOG_ERROR("Failed to get segment %s", FLAT_LINEAR_META_SEG_ID.c_str());
    return IndexError_Runtime;
  }
  if (segment->write(0, &meta_, sizeof(meta_)) != sizeof(meta_)) {
    LOG_ERROR("Failed to write segment data");
    return IndexError_WriteData;
  }
  if (segment->write(sizeof(meta_), str.data(), str.size()) != str.size()) {
    LOG_ERROR("Failed to write segment data, size=%zu", str.size());
    return IndexError_WriteData;
  }

  ret = storage->append("IndexMeta", str.size());
  if (ailego_unlikely(ret != 0)) {
    LOG_ERROR("Failed to append segment IndexMeta, code: %d", ret);
    return ret;
  }
  auto index_meta_segment = storage->get("IndexMeta");
  if (index_meta_segment->write(0, str.data(), str.size()) != str.size()) {
    LOG_ERROR("Failed to write segment data, size=%zu", str.size());
    return IndexError_WriteData;
  }
  *stats_.mutable_index_size() += size;

  // Init Linear List Head Segment
  size = ailego_align(sizeof(BlockLocation) * linear_list_count(), page);
  ret = storage->append(FLAT_LINEAR_LIST_HEAD_SEG_ID, size);
  if (ailego_unlikely(ret != 0)) {
    LOG_ERROR("Failed to append segment %s for %s, size=%zu",
              FLAT_LINEAR_LIST_HEAD_SEG_ID.c_str(), IndexError::What(ret),
              size);
    return ret;
  }
  segment = storage->get(FLAT_LINEAR_LIST_HEAD_SEG_ID);
  if (ailego_unlikely(!segment)) {
    LOG_ERROR("Failed to get segment %s", FLAT_LINEAR_LIST_HEAD_SEG_ID.c_str());
    return IndexError_Runtime;
  }
  if (segment->resize(size) != size) {
    LOG_ERROR("Failed to resize segment, size=%zu", size);
    return IndexError_WriteData;
  }
  segments_.emplace_back(std::move(segment));

  *stats_.mutable_index_size() += size;

  return 0;
}

int FlatStreamerEntity::load_linear_meta(IndexStorage::Pointer storage) {
  adjust_segment_size(&meta_);

  // Load Meta Segment
  auto segment = storage->get(FLAT_LINEAR_META_SEG_ID);
  if (!segment || segment->data_size() < sizeof(meta_)) {
    LOG_ERROR("Missing segment %s, or invalid segment size",
              FLAT_LINEAR_META_SEG_ID.c_str());
    return IndexError_InvalidFormat;
  }
  IndexStorage::MemoryBlock data_block;
  if (segment->read(0, data_block, segment->data_size()) !=
      segment->data_size()) {
    LOG_ERROR("Failed to read storage, size=%zu", segment->data_size());
    return IndexError_InvalidFormat;
  }
  auto *mt = reinterpret_cast<const decltype(meta_) *>(data_block.data());
  if (mt->header.block_vector_count != meta_.header.block_vector_count) {
    LOG_ERROR("Unmatched BlockVecCount Setting, Index %u vs Setting %u",
              mt->header.block_vector_count, meta_.header.block_vector_count);
    return IndexError_Mismatch;
  }
  if (mt->header.block_size != meta_.header.block_size) {
    LOG_ERROR("Unmatched BlockSize Setting, Index %u vs Setting %u",
              mt->header.block_size, meta_.header.block_size);
    return IndexError_Mismatch;
  }
  if (mt->header.index_meta_size + sizeof(meta_) > segment->data_size()) {
    LOG_ERROR("Invalid format, IndexMetaSize %u, SegmentSize %zu",
              mt->header.index_meta_size, segment->data_size());
    return IndexError_InvalidFormat;
  }
  if (mt->header.linear_list_count != meta_.header.linear_list_count) {
    LOG_ERROR("Unmatch LinearListCount, Index size %u vs Setting %u",
              mt->header.linear_list_count, meta_.header.linear_list_count);
    return IndexError_InvalidFormat;
  }
  IndexMeta index_meta;
  if (!index_meta.deserialize(mt->index_meta_data(),
                              mt->header.index_meta_size)) {
    LOG_ERROR("Failed to deserialize IndexMeta, size=%u",
              mt->header.index_meta_size);
    return IndexError_InvalidFormat;
  }
  if (!turbo::QuantizerStorageDataTypeMatches(index_meta, index_meta_)) {
    LOG_ERROR("Unmatched or invalid quantizer storage precision");
    return IndexError_Mismatch;
  }
  if (index_meta.data_type() != index_meta_.data_type() ||
      index_meta.dimension() != index_meta_.dimension() ||
      index_meta.element_size() != index_meta_.element_size() ||
      index_meta.metric_name() != index_meta_.metric_name() ||
      index_meta.quantizer_name() != index_meta_.quantizer_name()) {
    LOG_ERROR(
        "Unmatch IndexMeta, Index(type=%u dim=%u elemsize=%u metric=%s "
        "quantizer=%s) Setting(type=%u dim=%u elemsize=%u metric=%s "
        "quantizer=%s)",
        index_meta.data_type(), index_meta.dimension(),
        index_meta.element_size(), index_meta.metric_name().c_str(),
        index_meta.quantizer_name().c_str(), index_meta_.data_type(),
        index_meta_.dimension(), index_meta_.element_size(),
        index_meta_.metric_name().c_str(),
        index_meta_.quantizer_name().c_str());
    return IndexError_Mismatch;
  }
  // Segment Size can be reconfigurable
  auto segment_size = meta_.segment_size;
  std::memcpy(&meta_, mt, sizeof(meta_));
  meta_.segment_size = segment_size;
  return 0;
}

int FlatStreamerEntity::load_segment_keys(BlockLocation block) {
  while (this->is_valid_block(block)) {
    IndexStorage::MemoryBlock block_header_block;
    this->get_block_header(block, block_header_block);
    const BlockHeader *hd =
        reinterpret_cast<const BlockHeader *>(block_header_block.data());
    if (ailego_unlikely(hd == nullptr ||
                        hd->vector_count > block_vector_count())) {
      LOG_ERROR("Failed to get block header");
      return IndexError_ReadData;
    }
    IndexStorage::MemoryBlock keys_block;
    if (get_block_keys(block, keys_block) != 0) return IndexError_ReadData;
    const uint64_t *keys =
        reinterpret_cast<const uint64_t *>(keys_block.data());
    IndexStorage::MemoryBlock deletion_map_block;
    if (get_block_deletion_map(block, deletion_map_block) != 0) {
      return IndexError_ReadData;
    }
    const DeletionMap *deletion_map =
        reinterpret_cast<const DeletionMap *>(deletion_map_block.data());

    if (deletion_map->is_dirty()) all_keys_valid_->store(false);

    for (uint32_t vector_index = 0; vector_index < hd->vector_count;
         ++vector_index) {
      if (keys[vector_index] == kInvalidKey) all_keys_valid_->store(false);
      if (!use_key_info_map_ || deletion_map->test(vector_index)) {
        continue;
      }
      size_t vector_off =
          this->get_block_vector_offset(block.block_index, vector_index);
      key_info_map_[keys[vector_index]] =
          VectorLocation(block.segment_id, false, vector_off);
      id_key_vector_.push_back(keys[vector_index]);
    }
    block = hd->next;
  }
  return 0;
}

int FlatStreamerEntity::load_segment_keys_to_vector() {
  for (uint32_t i = 0; i < meta_.header.total_vector_count; i++) {
    size_t block_id = i / block_vector_count();
    uint32_t vector_index = i % block_vector_count();

    ailego_assert(segments_.size() > 1);
    size_t segment_block_count =
        segments_[1]->data_size() / linear_block_size();
    size_t segment_id = block_id / segment_block_count + 1;
    size_t real_block_id = block_id % segment_block_count;
    size_t vector_off =
        this->get_block_vector_offset(real_block_id, vector_index);

    withid_key_info_map_.push_back(
        VectorLocation(segment_id, false, vector_off));
    size_t key_off = get_block_key_offset(real_block_id, vector_index);
    withid_key_map_.push_back(key_off);
  }
  return 0;
}

int FlatStreamerEntity::load_storage(IndexStorage::Pointer storage) {
  meta_dirty_ = false;
  int ret = this->load_linear_meta(storage);
  if (ailego_unlikely(ret != 0)) {
    return ret;
  }

  // Load Linear List
  auto hd_segment = storage->get(FLAT_LINEAR_LIST_HEAD_SEG_ID);
  if (ailego_unlikely(!hd_segment)) {
    LOG_ERROR("Failed to get segment %s", FLAT_LINEAR_LIST_HEAD_SEG_ID.c_str());
    return IndexError_Runtime;
  }
  if (hd_segment->data_size() < linear_list_count() * sizeof(BlockLocation)) {
    LOG_ERROR("Invalid segment size, LinearListCount=%zu, size=%zu",
              linear_list_count(), hd_segment->data_size());
    return IndexError_InvalidFormat;
  }
  segments_.emplace_back(hd_segment);

  size_t index_size = hd_segment->capacity();
  for (size_t i = 1; i <= meta_.segment_count; ++i) {
    std::string segment_id =
        ailego::StringHelper::Concat(FLAT_SEGMENT_FEATURES_SEG_ID, i);
    auto seg = storage->get(segment_id);
    if (!seg || seg->data_size() < meta_.header.block_size) {
      LOG_ERROR("Failed to get segment %s, or invalid segment size",
                segment_id.c_str());
      return IndexError_InvalidFormat;
    }
    index_size += seg->capacity();
    segments_.emplace_back(std::move(seg));
  }

  for (size_t i = 0; i < linear_list_count(); i++) {
    IndexStorage::MemoryBlock head_block;
    this->get_head_block(head_block);
    const BlockLocation *bl =
        reinterpret_cast<const BlockLocation *>(head_block.data());
    if (ailego_unlikely(bl == nullptr)) {
      LOG_ERROR("Failed to get block loc");
      return IndexError_ReadData;
    }
    BlockLocation block = *bl;
    ret = this->load_segment_keys(block);
    if (ret == 0 && !use_key_info_map_) {
      ret = this->load_segment_keys_to_vector();
    }
    if (ailego_unlikely(ret != 0)) {
      return ret;
    }
  }

  char create_time[32];
  char update_time[32];
  ailego::Realtime::Gmtime(meta_.create_time, "%Y-%m-%d %H:%M:%S", create_time,
                           sizeof(create_time));
  ailego::Realtime::Gmtime(meta_.update_time, "%Y-%m-%d %H:%M:%S", update_time,
                           sizeof(update_time));
  LOG_DEBUG(
      "Load Index, IndexSize=%zu SegmentCount=%u SegmentSize=%u "
      "RevisionId=%zu BlockCount=%u BlockSize=%u "
      "BlockVectorCount=%u LinearListCount=%u TotalVecCount=%zu "
      "CreateTime=%s UpdateTime=%s",
      index_size, meta_.segment_count, meta_.segment_size,
      static_cast<size_t>(meta_.revision_id), meta_.header.block_count,
      meta_.header.block_size, meta_.header.block_vector_count,
      meta_.header.linear_list_count,
      static_cast<size_t>(meta_.header.total_vector_count), create_time,
      update_time);

  stats_.set_index_size(index_size);
  stats_.set_check_point(storage->check_point());
  stats_.set_create_time(meta_.create_time);
  stats_.set_revision_id(meta_.revision_id);
  stats_.set_update_time(meta_.update_time);
  stats_.set_loaded_count(meta_.header.total_vector_count);

  return 0;
}

int FlatStreamerEntity::alloc_segment() {
  // add()/add_vector_with_id() serialize allocation with mutex_. Keep the
  // cache lock out of storage allocation, which may wait for reader pins.
  size_t index;
  {
    std::lock_guard<std::mutex> lock(segments_mutex_);
    index = segments_.size();
  }
  if (index == kMaxSegmentId) {
    LOG_ERROR("Failed to alloc new segment, exceed max count %zu",
              kMaxSegmentId);
    return IndexError_IndexFull;
  }

  std::string segment_id =
      ailego::StringHelper::Concat(FLAT_SEGMENT_FEATURES_SEG_ID, index);
  size_t size =
      ailego_align(meta_.segment_size, ailego::MemoryHelper::PageSize());
  auto segment = storage_->get(segment_id);
  if (segment) {
    if (segment->padding_size() < linear_block_size()) {
      LOG_ERROR(
          "Unexpect segment, index=%zu, data_size=%zu "
          "padding_size=%zu block_size=%zu",
          index, segment->data_size(), segment->padding_size(),
          linear_block_size());
      return IndexError_Runtime;
    }
    LOG_WARN("Alloc an existing segment=%s capacity=%zu", segment_id.c_str(),
             segment->capacity());
  } else {
    int ret = storage_->append(segment_id, size);
    if (ailego_unlikely(ret != 0)) {
      LOG_ERROR("Failed to alloc segment from storage");
      return ret;
    }
    segment = storage_->get(segment_id);
    if (ailego_unlikely(!segment)) {
      LOG_ERROR("Failed to get segment %s", segment_id.c_str());
      return IndexError_Runtime;
    }
  }
  meta_.segment_count += 1;
  meta_.header.linear_body_size += size;
  {
    std::lock_guard<std::mutex> lock(segments_mutex_);
    segments_.emplace_back(std::move(segment));
  }
  *stats_.mutable_index_size() += size;

  // Update meta information
  auto meta_segment = storage_->get(FLAT_LINEAR_META_SEG_ID);
  if (ailego_unlikely(!meta_segment)) {
    LOG_ERROR("Failed to get segment %s", FLAT_LINEAR_META_SEG_ID.c_str());
    return IndexError_Runtime;
  }
  if (meta_segment->write(0, &meta_, sizeof(meta_)) != sizeof(meta_)) {
    LOG_ERROR("Failed to write meta segment");
    return IndexError_WriteData;
  }

  return 0;
}

int FlatStreamerEntity::alloc_block(const BlockLocation &next,
                                    BlockLocation *block) {
  IndexStorage::Segment::Pointer segment;
  size_t segment_id;
  {
    std::lock_guard<std::mutex> lock(segments_mutex_);
    segment_id = segments_.size() - 1;
    segment = segments_.back();
  }
  if (segment_id == 0 || segment->padding_size() < linear_block_size()) {
    int ret = this->alloc_segment();
    if (ailego_unlikely(ret != 0)) {
      return ret;
    }
    std::lock_guard<std::mutex> lock(segments_mutex_);
    segment_id = segments_.size() - 1;
    segment = segments_.back();
  }

  size_t block_index = segment->data_size() / linear_block_size();
  if (block_index == kMaxBlockId) {
    LOG_ERROR("Failed to alloc block, exceed max count %zu per segment",
              kMaxBlockId);
    return IndexError_IndexFull;
  }

  BlockHeader header;
  header.next = next;
  header.vector_count = 0;
  header.column_major = false;

  size_t hd_off = segment->data_size() + linear_block_size() - sizeof(header);
  if (segment->write(hd_off, &header, sizeof(header)) != sizeof(header)) {
    LOG_ERROR("Failed to write block header");
    return IndexError_WriteData;
  }

  size_t del_off = hd_off - sizeof(DeletionMap);
  DeletionMap reset_del_map{};
  if (segment->write(del_off, &reset_del_map, sizeof(reset_del_map)) !=
      sizeof(reset_del_map)) {
    LOG_ERROR("Failed to write block deletion map");
    return IndexError_WriteData;
  }

  ++meta_.header.block_count;
  block->segment_id = segment_id;
  block->block_index = (segment->data_size() / linear_block_size()) - 1;

  return 0;
}

int FlatStreamerEntity::add_to_block(const BlockLocation &block, uint64_t key,
                                     const void *data, size_t size) {
  IndexStorage::MemoryBlock block_header_block;
  this->get_block_header(block, block_header_block);
  const BlockHeader *header =
      reinterpret_cast<const BlockHeader *>(block_header_block.data());
  if (ailego_unlikely(header == nullptr)) {
    LOG_ERROR("Failed to get header");
    return IndexError_ReadData;
  }

  if (header->vector_count == block_vector_count()) {
    return IndexError_IndexFull;
  }

  auto segment = get_segment(block.segment_id);
  if (!segment) {
    return IndexError_WriteData;
  }

  size_t vector_off =
      get_block_vector_offset(block.block_index, header->vector_count);
  if (segment->write(vector_off, data, size) != size) {
    LOG_ERROR("Failed to write vector, off=%zu size=%zu", vector_off, size);
    return IndexError_WriteData;
  }

  size_t key_off =
      get_block_key_offset(block.block_index, header->vector_count);
  if (segment->write(key_off, &key, sizeof(key)) != sizeof(key)) {
    LOG_ERROR("Failed to write key, off=%zu", key_off);
    return IndexError_WriteData;
  }

  // Invalidate before publishing the new row count to concurrent readers.
  if (key == kInvalidKey) all_keys_valid_->store(false);

  BlockHeader hd = *header;
  hd.vector_count += 1;
  size_t hd_off = get_block_header_offset(block.block_index);
  if (segment->write(hd_off, &hd, sizeof(hd)) != sizeof(hd)) {
    LOG_ERROR("Failed to write block header, off=%zu", hd_off);
    return IndexError_WriteData;
  }

  VectorLocation loc(block.segment_id, false, vector_off);
  key_info_map_lock_->lock();
  key_info_map_[key] = loc;
  id_key_vector_.push_back(key);
  withid_key_info_map_.push_back(loc);
  withid_key_map_.push_back(key_off);
  key_info_map_lock_->unlock();

  ++meta_.header.total_vector_count;
  return 0;
}

int FlatStreamerEntity::add_vector_with_id(const uint32_t id, const void *query,
                                           const uint32_t size) {
  std::lock_guard<std::mutex> lock(mutex_);
  // if (filter_same_key_) {
  //   key_info_map_lock_->lock_shared();
  //   if (key_info_map_.find(id) != key_info_map_.end()) {
  //     key_info_map_lock_->unlock_shared();
  //     LOG_WARN("Try to add duplicate key, drop it");
  //     return IndexError_Duplicate;
  //   }
  //   key_info_map_lock_->unlock_shared();
  // }

  if (size != static_cast<size_t>(index_meta_.element_size())) {
    LOG_ERROR("Failed to add, mismatch size %u vs elemsize %u", size,
              index_meta_.element_size());
    return IndexError_Mismatch;
  }


  meta_dirty_ = true;
  if (id >= vector_count()) {
    IndexStorage::MemoryBlock head_block;
    this->get_head_block(head_block);
    BlockLocation block =
        *reinterpret_cast<const BlockLocation *>(head_block.data());
    // Release buffer-pool pin before any alloc_block() call that may trigger
    // append_segment() and rebuild the pool (same reason as in add()).
    head_block.reset(nullptr);
    if (!this->is_valid_block(block)) {
      int ret = this->alloc_block(block, &block);
      if (ailego_unlikely(ret != 0)) {
        return ret;
      }
      ret = this->update_head_block(block);
      if (ailego_unlikely(ret != 0)) {
        return ret;
      }
    }
    for (size_t start_id = vector_count(); start_id < id; ++start_id) {
      std::vector<char> vec(size);
      int ret = this->add_to_block(block, kInvalidKey, vec.data(), size);
      if (ret == IndexError_IndexFull) {
        ret = this->alloc_block(block, &block);
        if (ailego_unlikely(ret != 0)) {
          return ret;
        }
        ret = this->update_head_block(block);
        if (ailego_unlikely(ret != 0)) {
          return ret;
        }
        ret = this->add_to_block(block, kInvalidKey, vec.data(), size);
        if (ailego_unlikely(ret != 0)) {
          return ret;
        }
      }
    }

    int ret = this->add_to_block(block, id, query, size);
    if (ret == IndexError_IndexFull) {
      ret = this->alloc_block(block, &block);
      if (ailego_unlikely(ret != 0)) {
        return ret;
      }
      ret = this->update_head_block(block);
      if (ailego_unlikely(ret != 0)) {
        return ret;
      }
      ret = this->add_to_block(block, id, query, size);
      if (ailego_unlikely(ret != 0)) {
        return ret;
      }
    }
  } else {
    VectorLocation vector_loc = withid_key_info_map_[id];
    auto segment = this->get_segment(vector_loc.segment_id);
    size_t vector_off = vector_loc.offset;
    if (segment->write(vector_off, query, size) != size) {
      LOG_ERROR("Failed to write vector, off=%zu size=%u", vector_off, size);
      return IndexError_WriteData;
    }
    size_t key_off = withid_key_map_[id];
    uint64_t key = id;
    if (segment->write(key_off, &key, sizeof(key)) != sizeof(key)) {
      LOG_ERROR("Failed to write key, off=%zu", key_off);
      return IndexError_WriteData;
    }
    key_info_map_lock_->lock();
    key_info_map_[key] = vector_loc;
    key_info_map_lock_->unlock();
  }
  (*stats_.mutable_added_count())++;
  stats_.set_revision_id(meta_.revision_id + 1);

  return 0;
}

}  // namespace core
}  // namespace zvec
