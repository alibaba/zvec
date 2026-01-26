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
#include <cstdint>
#include "framework/index_error.h"
#include "flat_utility.h"

namespace zvec {
namespace core {

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
  auto metric = IndexFactory::CreateMetric(index_meta_.metric_name());
  if (!metric) {
    LOG_ERROR("Failed to create metric %s", index_meta_.metric_name().c_str());
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

  LOG_DEBUG("Open storage %s done, metric=%s", storage_->name().c_str(),
            index_meta_.metric_name().c_str());

  return 0;
}

int FlatStreamerEntity::close(void) {
  segments_.clear();
  storage_.reset();
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

  return 0;
}

int FlatStreamerEntity::flush_linear_meta(void) {
  if (!storage_) {
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
  const BlockLocation *bl =
      reinterpret_cast<const BlockLocation *>(head_block.data());
  if (ailego_unlikely(bl == nullptr)) {
    LOG_ERROR("Failed to get block loc");
    return IndexError_ReadData;
  }
  BlockLocation block = *bl;

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
                               IndexContext::Stats *context_stats) const {
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
  uint32_t scan_count;
  return this->search(query, filter, &scan_count, heap, context_stats);
}

FlatStreamerEntity::Pointer FlatStreamerEntity::clone(void) const {
  std::vector<IndexStorage::Segment::Pointer> segments;
  segments.reserve(segments_.size());
  for (size_t i = 0; i < segments_.size(); ++i) {
    segments.emplace_back(segments_[i]->clone());
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
  entity->segments_ = segments;
  entity->meta_ = this->meta_;
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

const void *FlatStreamerEntity::get_vector_by_key(uint64_t key) const {
  VectorLocation loc{};
  key_info_map_lock_->lock_shared();
  if (use_key_info_map_) {
    auto iterator = key_info_map_.find(key);
    if (iterator == key_info_map_.end()) {
      key_info_map_lock_->unlock_shared();
      return nullptr;
    }
    loc = iterator->second;
  } else {
    if (key < withid_key_info_map_.size()) {
      loc = withid_key_info_map_[key];
    } else {
      key_info_map_lock_->unlock_shared();
      return nullptr;
    }
  }
  key_info_map_lock_->unlock_shared();

  auto segment = this->get_segment(loc.segment_id);
  const void *data = nullptr;
  if (segment->read(loc.offset, &data, index_meta_.element_size()) !=
      index_meta_.element_size()) {
    LOG_ERROR("Failed to read segment, size=%u", index_meta_.element_size());
    return nullptr;
  }
  return data;
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

IndexProvider::Iterator::Pointer FlatStreamerEntity::creater_iterator(
    void) const {
  auto entity = this->clone();
  if (!entity) {
    LOG_ERROR("Failed to clone entity");
    return nullptr;
  }

  return Iterator::Pointer(new (std::nothrow)
                               FlatStreamerEntity::Iterator(std::move(entity)));
}

void FlatStreamerEntity::Iterator::read_next_block(void) {
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

  AdjustSegmentSize(&meta_);

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
  AdjustSegmentSize(&meta_);

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
  if (!index_meta.deserialize(mt->header.index_meta,
                              mt->header.index_meta_size)) {
    LOG_ERROR("Failed to deserialize IndexMeta, size=%u",
              mt->header.index_meta_size);
    return IndexError_InvalidFormat;
  }
  if (index_meta.data_type() != index_meta_.data_type() ||
      index_meta.dimension() != index_meta_.dimension() ||
      index_meta.element_size() != index_meta_.element_size() ||
      index_meta.metric_name() != index_meta_.metric_name()) {
    LOG_ERROR(
        "Unmatch IndexMeta, Index(type=%u dim=%u elemsize=%u "
        "metric=%s) Setting(type=%u dim=%u elemsize=%u metric=%s)",
        index_meta.data_type(), index_meta.dimension(),
        index_meta.element_size(), index_meta.metric_name().c_str(),
        index_meta_.data_type(), index_meta_.dimension(),
        index_meta_.element_size(), index_meta_.metric_name().c_str());
    return IndexError_Mismatch;
  }
  // Segment Size can be reconfigurable
  auto segment_size = meta_.segment_size;
  std::memcpy(&meta_, mt, sizeof(meta_));
  meta_.segment_size = segment_size;
  return 0;
}

int FlatStreamerEntity::load_segment_keys_to_map(BlockLocation block) {
  while (this->is_valid_block(block)) {
    auto segment = this->get_segment(block.segment_id);

    IndexStorage::MemoryBlock block_header_block;
    this->get_block_header(block, block_header_block);
    const BlockHeader *hd =
        reinterpret_cast<const BlockHeader *>(block_header_block.data());
    if (ailego_unlikely(hd == nullptr)) {
      LOG_ERROR("Failed to get block header");
      return IndexError_ReadData;
    }
    IndexStorage::MemoryBlock keys_block;
    this->get_block_keys(block, keys_block);
    const uint64_t *keys =
        reinterpret_cast<const uint64_t *>(keys_block.data());
    IndexStorage::MemoryBlock deletion_map_block;
    this->get_block_deletion_map(block, deletion_map_block);
    const DeletionMap *deletion_map =
        reinterpret_cast<const DeletionMap *>(deletion_map_block.data());

    for (uint32_t vector_index = 0; vector_index < hd->vector_count;
         ++vector_index) {
      if (deletion_map->test(vector_index)) {
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
    if (use_key_info_map_) {
      ret = this->load_segment_keys_to_map(block);
    } else {
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

int FlatStreamerEntity::alloc_segment(void) {
  size_t index = segments_.size();
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
  segments_.emplace_back(std::move(segment));
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
  if (segments_.size() <= 1 ||
      segments_.back()->padding_size() < linear_block_size()) {
    int ret = this->alloc_segment();
    if (ailego_unlikely(ret != 0)) {
      return ret;
    }
  }

  auto &segment = segments_.back();
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
  block->segment_id = segments_.size() - 1;
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

  auto &segment = segments_[block.segment_id];

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


  if (id >= vector_count()) {
    IndexStorage::MemoryBlock head_block;
    this->get_head_block(head_block);
    BlockLocation block =
        *reinterpret_cast<const BlockLocation *>(head_block.data());
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
