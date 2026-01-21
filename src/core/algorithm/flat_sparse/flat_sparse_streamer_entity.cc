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

#include "flat_sparse_streamer_entity.h"
#include <cstdint>
#include <memory>
#include <utility>
#include <ailego/utility/memory_helper.h>
#include <zvec/ailego/container/vector.h>
#include <zvec/ailego/utility/time_helper.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_logger.h>
#include "flat_sparse_index_format.h"
#include "flat_sparse_utility.h"

namespace zvec {
namespace core {

FlatSparseStreamerEntity::FlatSparseStreamerEntity(IndexStreamer::Stats &stats)
    : stats_(stats) {}

int FlatSparseStreamerEntity::open(IndexStorage::Pointer storage,
                                   const IndexMeta &meta) {
  if (storage_) {
    LOG_ERROR("An storage instance is already opened");
    return IndexError_Duplicate;
  }

  keys_map_lock_ = std::make_shared<ailego::SharedMutex>();
  if (!keys_map_lock_) {
    LOG_ERROR("FlatSparseStreamerEntity new object failed");
    return IndexError_NoMemory;
  }
  keys_map_ = std::make_shared<std::map<uint64_t, node_id_t>>();

  if (storage->get(PARAM_FLAT_SPARSE_META_SEG_ID) ||
      storage->get(PARAM_FLAT_SPARSE_STREAMER_META_SEG_ID)) {
    int ret = this->load_storage(storage, meta);
    if (ailego_unlikely(ret != 0)) {
      LOG_ERROR("Failed to load storage index");
      return ret;
    }
  } else {
    int ret = this->init_storage(storage, meta);
    if (ailego_unlikely(ret != 0)) {
      LOG_ERROR("Failed to load storage index");
      return ret;
    }
  }

  if (init_metric(meta) != 0) {
    LOG_ERROR("Failed to init metric");
    return IndexError_InvalidFormat;
  }

  // reserve data chunk
  meta.streamer_params().get(PARAM_FLAT_SPARSE_STREAMER_MAX_DATA_CHUNK_CNT,
                             &max_data_chunk_cnt_);
  sparse_data_chunks_.reserve(max_data_chunk_cnt_);

  // reserve offset chunk
  meta.streamer_params().get(PARAM_FLAT_SPARSE_STREAMER_MAX_DOC_CNT,
                             &max_doc_cnt_);
  sparse_offset_chunks_.reserve(max_doc_cnt_ / doc_cnt_per_offset_chunk() + 1);
  sparse_unit_size_ = meta.unit_size();

  LOG_DEBUG(
      "FlatSparseStreamerEntity open success, doc_count[%u], "
      "data_chunk_size[%u], offset_chunk_size[%u], data_chunk_count[%zu], "
      "offset_chunk_count[%zu]",
      meta_.doc_cnt, streamer_meta_.data_chunk_size,
      streamer_meta_.offset_chunk_size, sparse_data_chunks_.size(),
      sparse_offset_chunks_.size());

  storage_ = storage;
  return 0;
}

int FlatSparseStreamerEntity::init_metric(const IndexMeta &meta) {
  metric_ = IndexFactory::CreateMetric(meta.metric_name());
  if (!metric_) {
    LOG_ERROR("Failed to create metric %s", meta.metric_name().c_str());
    return IndexError_NoExist;
  }
  int ret = metric_->init(meta, meta.metric_params());
  if (ret != 0) {
    LOG_ERROR("Failled to init metric, ret=%d", ret);
    return ret;
  }

  if (!metric_->sparse_distance()) {
    LOG_ERROR("Invalid metric distance");
    return IndexError_InvalidArgument;
  }

  search_sparse_distance_ = metric_->sparse_distance();

  if (metric_->query_metric() && metric_->query_metric()->distance()) {
    search_sparse_distance_ = metric_->query_metric()->sparse_distance();
  }

  return 0;
}

int FlatSparseStreamerEntity::load_storage(IndexStorage::Pointer storage,
                                           const IndexMeta &meta) {
  size_t index_size{0};

  // load meta
  auto segment = storage->get(PARAM_FLAT_SPARSE_META_SEG_ID);

  if (!segment || segment->data_size() < sizeof(meta_)) {
    LOG_ERROR("Missing segment %s, or invalid segment size",
              PARAM_FLAT_SPARSE_META_SEG_ID.c_str());
    return IndexError_InvalidFormat;
  }
  IndexStorage::MemoryBlock data_block;
  if (ailego_unlikely(segment->read(0, data_block, sizeof(meta_)) !=
                      sizeof(meta_))) {
    LOG_ERROR("Failed to read meta segment %s",
              PARAM_FLAT_SPARSE_META_SEG_ID.c_str());
    return IndexError_ReadData;
  }
  meta_ = *(reinterpret_cast<const decltype(meta_) *>(data_block.data()));
  index_size += segment->capacity();

  // load streamer meta
  segment = storage->get(PARAM_FLAT_SPARSE_STREAMER_META_SEG_ID);
  if (!segment || segment->data_size() < sizeof(streamer_meta_)) {
    LOG_ERROR("Missing segment %s, or invalid segment size",
              PARAM_FLAT_SPARSE_STREAMER_META_SEG_ID.c_str());
    return IndexError_InvalidFormat;
  }
  if (ailego_unlikely(segment->read(0, data_block, sizeof(streamer_meta_)) !=
                      sizeof(streamer_meta_))) {
    LOG_ERROR("Failed to read streamer meta segment %s",
              PARAM_FLAT_SPARSE_STREAMER_META_SEG_ID.c_str());
    return IndexError_ReadData;
  }
  streamer_meta_ =
      *(reinterpret_cast<const decltype(streamer_meta_) *>(data_block.data()));
  index_size += segment->capacity();

  uint32_t meta_data_chunk_size{streamer_meta_.data_chunk_size};
  uint32_t meta_offset_chunk_size{streamer_meta_.offset_chunk_size};
  meta.streamer_params().get(PARAM_FLAT_SPARSE_STREAMER_DATA_CHUNK_SIZE,
                             &meta_data_chunk_size);
  meta.streamer_params().get(PARAM_FLAT_SPARSE_STREAMER_OFFSET_CHUNK_SIZE,
                             &meta_offset_chunk_size);
  if (streamer_meta_.data_chunk_size != meta_data_chunk_size ||
      streamer_meta_.offset_chunk_size != meta_offset_chunk_size) {
    LOG_ERROR(
        "Invalid streamer meta chunk size data[%u] offset[%u], expect data[%u] "
        "offset[%u]",
        streamer_meta_.data_chunk_size, streamer_meta_.offset_chunk_size,
        meta_data_chunk_size, meta_offset_chunk_size);
    return IndexError_InvalidFormat;
  }

  // check chunk cnt
  if (streamer_meta_.data_chunk_count > max_data_chunk_cnt_ ||
      meta_.doc_cnt > max_doc_cnt_) {
    LOG_ERROR(
        "Invalid data chunk count[%u] doc count[%u], expect less than "
        "chunk count[%u] doc count[%u]",
        streamer_meta_.data_chunk_count, meta_.doc_cnt, max_data_chunk_cnt_,
        max_doc_cnt_);
    return IndexError_InvalidFormat;
  }

  // load offset chunks
  for (size_t i = 0; i < streamer_meta_.offset_chunk_count; ++i) {
    std::string segment_id =
        ailego::StringHelper::Concat(PARAM_FLAT_SPARSE_OFFSET_SEG_ID_PREFIX, i);
    segment = storage->get(segment_id);
    if (!segment) {
      LOG_ERROR("Missing segment %s", segment_id.c_str());
      return IndexError_InvalidFormat;
    }
    sparse_offset_chunks_.emplace_back(segment);
    index_size += segment->capacity();
  }
  // load data chunks
  for (size_t i = 0; i < streamer_meta_.data_chunk_count; ++i) {
    std::string segment_id =
        ailego::StringHelper::Concat(PARAM_FLAT_SPARSE_DATA_SEG_ID_PREFIX, i);
    segment = storage->get(segment_id);
    if (!segment) {
      LOG_ERROR("Missing segment %s", segment_id.c_str());
    }
    sparse_data_chunks_.emplace_back(segment);
    index_size += segment->capacity();
  }

  // load keys
  for (node_id_t i = 0; i < meta_.doc_cnt; ++i) {
    (*keys_map_)[get_key(i)] = i;
  }

  stats_.set_index_size(index_size);
  stats_.set_check_point(storage->check_point());
  stats_.set_create_time(meta_.create_time);
  stats_.set_update_time(meta_.update_time);
  stats_.set_loaded_count(keys_map_->size());

  return 0;
}

int FlatSparseStreamerEntity::init_storage(IndexStorage::Pointer storage,
                                           const IndexMeta &meta) {
  meta_.create_time = ailego::Realtime::Seconds();
  stats_.set_create_time(meta_.create_time);
  meta_.update_time = ailego::Realtime::Seconds();
  stats_.set_update_time(meta_.update_time);
  meta_.doc_cnt = 0;

  meta.streamer_params().get(PARAM_FLAT_SPARSE_STREAMER_DATA_CHUNK_SIZE,
                             &streamer_meta_.data_chunk_size);
  meta.streamer_params().get(PARAM_FLAT_SPARSE_STREAMER_OFFSET_CHUNK_SIZE,
                             &streamer_meta_.offset_chunk_size);

  // append meta segment
  size_t size = ailego_align(sizeof(meta_), ailego::MemoryHelper::PageSize());
  int ret = storage->append(PARAM_FLAT_SPARSE_META_SEG_ID, size);
  if (ailego_unlikely(ret != 0)) {
    LOG_ERROR("Failed to append meta segment %s",
              PARAM_FLAT_SPARSE_META_SEG_ID.c_str());
    return ret;
  }
  auto segment = storage->get(PARAM_FLAT_SPARSE_META_SEG_ID);
  if (ailego_unlikely(!segment)) {
    LOG_ERROR("Failed to get meta segment %s",
              PARAM_FLAT_SPARSE_META_SEG_ID.c_str());
    return IndexError_Runtime;
  }
  if (segment->write(0, &meta_, sizeof(meta_)) != sizeof(meta_)) {
    LOG_ERROR("Failed to write meta segment %s",
              PARAM_FLAT_SPARSE_META_SEG_ID.c_str());
    return IndexError_WriteData;
  }

  *stats_.mutable_index_size() += size;

  // append streamer meta segment
  size = ailego_align(sizeof(streamer_meta_), ailego::MemoryHelper::PageSize());
  ret = storage->append(PARAM_FLAT_SPARSE_STREAMER_META_SEG_ID, size);
  if (ailego_unlikely(ret != 0)) {
    LOG_ERROR("Failed to append streamer meta segment %s",
              PARAM_FLAT_SPARSE_STREAMER_META_SEG_ID.c_str());
    return ret;
  }
  segment = storage->get(PARAM_FLAT_SPARSE_STREAMER_META_SEG_ID);
  if (ailego_unlikely(!segment)) {
    LOG_ERROR("Failed to get streamer meta segment %s",
              PARAM_FLAT_SPARSE_STREAMER_META_SEG_ID.c_str());
    return IndexError_Runtime;
  }
  if (segment->write(0, &streamer_meta_, sizeof(streamer_meta_)) !=
      sizeof(streamer_meta_)) {
    LOG_ERROR("Failed to write streamer meta segment %s",
              PARAM_FLAT_SPARSE_STREAMER_META_SEG_ID.c_str());
    return IndexError_WriteData;
  }

  *stats_.mutable_index_size() += size;

  return 0;
}

int FlatSparseStreamerEntity::close() {
  storage_.reset();
  sparse_data_chunks_.clear();
  sparse_offset_chunks_.clear();

  keys_map_lock_.reset();
  keys_map_.reset();

  return 0;
}

int FlatSparseStreamerEntity::flush(uint64_t checkpoint) {
  // flush meta
  meta_.update_time = ailego::Realtime::Seconds();
  stats_.set_update_time(meta_.update_time);
  auto segment = storage_->get(PARAM_FLAT_SPARSE_META_SEG_ID);
  if (ailego_unlikely(!segment)) {
    LOG_ERROR("Failed to get meta segment %s",
              PARAM_FLAT_SPARSE_META_SEG_ID.c_str());
    return IndexError_Runtime;
  }
  if (segment->write(0, &meta_, sizeof(meta_)) != sizeof(meta_)) {
    LOG_ERROR("Failed to write meta segment %s",
              PARAM_FLAT_SPARSE_META_SEG_ID.c_str());
    return IndexError_WriteData;
  }

  // flush streamer meta
  streamer_meta_.data_chunk_count = sparse_data_chunks_.size();
  streamer_meta_.offset_chunk_count = sparse_offset_chunks_.size();
  segment = storage_->get(PARAM_FLAT_SPARSE_STREAMER_META_SEG_ID);
  if (ailego_unlikely(!segment)) {
    LOG_ERROR("Failed to get streamer meta segment %s",
              PARAM_FLAT_SPARSE_STREAMER_META_SEG_ID.c_str());
    return IndexError_Runtime;
  }
  if (segment->write(0, &streamer_meta_, sizeof(streamer_meta_)) !=
      sizeof(streamer_meta_)) {
    LOG_ERROR("Failed to write streamer meta segment %s",
              PARAM_FLAT_SPARSE_STREAMER_META_SEG_ID.c_str());
    return IndexError_WriteData;
  }

  if (checkpoint != 0) {
    storage_->refresh(checkpoint);
  }
  int ret = storage_->flush();
  if (ailego_unlikely(ret != 0)) {
    LOG_ERROR("Failed to flush storage for %s", IndexError::What(ret));
    return ret;
  }
  if (checkpoint != 0) {
    stats_.set_check_point(checkpoint);
  }

  return 0;
}

int FlatSparseStreamerEntity::dump(const IndexDumper::Pointer &dumper) {
  ailego::ElapsedTime stamp;

  int ret;
  // meta
  ret = dump_meta(dumper.get());
  if (ret != 0) {
    return ret;
  }

  auto duration_dump_meta = stamp.milli_seconds();

  // offset & data
  ret = dump_offset_data(dumper.get());
  if (ret != 0) {
    return ret;
  }

  auto duration_dump_offset_data = stamp.milli_seconds() - duration_dump_meta;

  // keys
  std::vector<uint64_t> keys = get_keys();
  ret = dump_keys(keys, dumper.get());
  if (ret != 0) {
    return ret;
  }

  auto duration_dump_keys =
      stamp.milli_seconds() - duration_dump_offset_data - duration_dump_meta;

  // mapping
  ret = dump_mapping(keys, dumper.get());
  if (ret != 0) {
    return ret;
  }

  auto duration_dump_mapping = stamp.milli_seconds() -
                               duration_dump_offset_data - duration_dump_meta -
                               duration_dump_keys;

  LOG_INFO(
      "Dump index meta: %zu ms, offset & data: %zu ms, keys: %zu ms, "
      "mapping: %zu ms",
      (size_t)duration_dump_meta, (size_t)duration_dump_offset_data,
      (size_t)duration_dump_keys, (size_t)duration_dump_mapping);

  return 0;
}

int FlatSparseStreamerEntity::dump_offset_data(IndexDumper *dumper) {
  ailego::ElapsedTime stamp;

  uint64_t init_offset = dump_size_;
  std::vector<std::pair<uint64_t, uint32_t>> offset_length;

  // write data
  int ret;
  node_id_t total_doc_cnt = doc_cnt();
  for (node_id_t node_id = 0; node_id < total_doc_cnt; node_id++) {
    uint32_t target_vector_len;
    IndexStorage::MemoryBlock target_vector_block;
    ret = get_sparse_vector_ptr_by_id(node_id, target_vector_block,
                                      &target_vector_len);
    if (ret != 0) {
      LOG_ERROR("Failed to get vector, node_id=%u, error: %s", node_id,
                IndexError::What(ret));
      return ret;
    }
    const void *target_vector = target_vector_block.data();
    ret = dump_sparse_vector_data(target_vector, target_vector_len, dumper);
    if (ret != 0) {
      LOG_ERROR("Failed to dump sparse vector data, node_id=%u, error: %s",
                node_id, IndexError::What(ret));
      return ret;
    }

    offset_length.push_back({dump_size_ - init_offset, target_vector_len});
    dump_size_ += target_vector_len;
  }

  // append data segment
  if (dumper->append(PARAM_FLAT_SPARSE_DUMP_DATA_SEG_ID,
                     dump_size_ - init_offset, 0, 0) != 0) {
    LOG_ERROR("append data segment failed");
    return IndexError_WriteData;
  }

  auto duration_dump_data = stamp.milli_seconds();

  // write offset
  for (auto &offset_length_pair : offset_length) {
    if (dumper->write(&offset_length_pair.first,
                      sizeof(offset_length_pair.first)) !=
        sizeof(offset_length_pair.first)) {
      return IndexError_WriteData;
    }
    if (dumper->write(&offset_length_pair.second,
                      sizeof(offset_length_pair.second)) !=
        sizeof(offset_length_pair.second)) {
      return IndexError_WriteData;
    }
    dump_size_ +=
        sizeof(offset_length_pair.first) + sizeof(offset_length_pair.second);
  }

  // append offset segment
  if (dumper->append(
          PARAM_FLAT_SPARSE_DUMP_OFFSET_SEG_ID,
          offset_length.size() * (sizeof(uint64_t) + sizeof(uint32_t)), 0,
          0) != 0) {
    LOG_ERROR("append offset segment failed");
    return IndexError_WriteData;
  }

  auto duration_dump_offset = stamp.milli_seconds() - duration_dump_data;

  LOG_INFO("Dump offset: %zu ms, data: %zu ms", (size_t)duration_dump_offset,
           (size_t)duration_dump_data);

  return 0;
}

int FlatSparseStreamerEntity::dump_sparse_vector_data(const void *data,
                                                      uint32_t length,
                                                      IndexDumper *dumper) {
  if (dumper->write(data, length) != length) {
    return IndexError_WriteData;
  }
  return 0;
}

int FlatSparseStreamerEntity::dump_meta(IndexDumper *dumper) {
  if (dumper->write(&meta_, sizeof(meta_)) != sizeof(meta_)) {
    LOG_ERROR("write meta failed");
    return IndexError_WriteData;
  }

  size_t meta_padding_size = ailego_align(sizeof(meta_), 32) - sizeof(meta_);
  if (meta_padding_size) {
    std::string padding(meta_padding_size, '\0');
    if (dumper->write(padding.data(), meta_padding_size) != meta_padding_size) {
      LOG_ERROR("write meta padding failed");
      return IndexError_WriteData;
    }
  }
  return dumper->append(PARAM_FLAT_SPARSE_META_SEG_ID, sizeof(meta_),
                        meta_padding_size, 0);
}

int FlatSparseStreamerEntity::dump_keys(const std::vector<uint64_t> &keys,
                                        IndexDumper *dumper) {
  if (keys.size() == 1 && keys.back() == kInvalidKey) {
    return IndexError_Runtime;
  }

  size_t keys_size = keys.size() * sizeof(uint64_t);
  if (dumper->write(keys.data(), keys_size) != keys_size) {
    LOG_ERROR("Failed to write keys to dumper %s", dumper->name().c_str());
    return IndexError_WriteData;
  }
  size_t keys_padding_size = ailego_align(keys_size, 32) - keys_size;
  if (keys_padding_size) {
    std::string padding(keys_padding_size, '\0');
    if (dumper->write(padding.data(), padding.size()) != padding.size()) {
      LOG_ERROR("Failed to write padding to dumper %s", dumper->name().c_str());
      return IndexError_WriteData;
    }
  }
  return dumper->append(PARAM_FLAT_SPARSE_DUMP_KEYS_SEG_ID, keys_size,
                        keys_padding_size, 0);
}

int FlatSparseStreamerEntity::dump_mapping(const std::vector<uint64_t> &keys,
                                           IndexDumper *dumper) {
  std::vector<uint32_t> mapping(keys.size());
  std::iota(mapping.begin(), mapping.end(), 0);
  std::sort(
      mapping.begin(), mapping.end(),
      [&keys](uint32_t lhs, uint32_t rhs) { return (keys[lhs] < keys[rhs]); });

  size_t mapping_size = mapping.size() * sizeof(uint32_t);
  size_t mapping_padding_size = ailego_align(mapping_size, 32) - mapping_size;
  if (dumper->write(mapping.data(), mapping_size) != mapping_size) {
    LOG_ERROR("Failed to write data into dumper %s", dumper->name().c_str());
    return IndexError_WriteData;
  }

  // Write the padding if need
  if (mapping_padding_size) {
    std::string padding(mapping_padding_size, '\0');
    if (dumper->write(padding.data(), padding.size()) != padding.size()) {
      LOG_ERROR("Failed to write data into dumper %s", dumper->name().c_str());
      return IndexError_WriteData;
    }
  }
  return dumper->append(PARAM_FLAT_SPARSE_DUMP_MAPPING_SEG_ID, mapping_size,
                        mapping_padding_size, 0);
}

FlatSparseStreamerEntity::Pointer FlatSparseStreamerEntity::clone() const {
  auto entity = new (std::nothrow) FlatSparseStreamerEntity(
      stats_, meta_, streamer_meta_, keys_map_lock_, keys_map_,
      sparse_data_chunks_, sparse_offset_chunks_);
  return FlatSparseStreamerEntity::Pointer(entity);
}

int FlatSparseStreamerEntity::add(uint64_t key,
                                  const std::string &sparse_vector,
                                  const uint32_t sparse_count) {
  uint32_t sparse_vector_len = sparse_vector.size();

  sparse_vector_len = AlignSize(sparse_vector_len);

  if (sparse_vector_len > streamer_meta_.data_chunk_size) {
    LOG_ERROR(
        "Sparse Vector Length exceed the chunk size, sparse vec len: %u, chunk "
        "size: %u",
        sparse_vector_len, streamer_meta_.data_chunk_size);
    return IndexError_InvalidArgument;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  node_id_t local_id = doc_cnt();

  if (ailego_unlikely(local_id >= max_doc_cnt_)) {
    LOG_ERROR("Add vector failed for exceed max doc count: %u", max_doc_cnt_);
    return IndexError_IndexFull;
  }

  // duplicate check
  if (ailego_unlikely(get_id(key) != kInvalidNodeId)) {
    LOG_WARN("Try to add duplicate key, ignore it");
    return IndexError_Duplicate;
  }

  // get sparse data chunk and offset for write sparse vector
  Chunk::Pointer sparse_data_chunk;
  uint32_t sparse_data_chunk_offset = -1U;
  uint32_t sparse_data_chunk_index = sparse_data_chunks_.size() - 1U;
  if (sparse_data_chunk_index == -1U ||
      sparse_data_chunks_[sparse_data_chunk_index]->data_size() +
              sparse_vector_len >
          streamer_meta_.data_chunk_size) {
    if (ailego_unlikely(sparse_data_chunks_.capacity() ==
                        sparse_data_chunks_.size())) {
      LOG_ERROR("add vector failed for no memory quota");
      if (sparse_data_chunk_index != -1U) {
        LOG_ERROR(
            "capacity: %zu, chunk used size: %zu, chunk size: %u, "
            "sparse_vector_len: %u",
            sparse_data_chunks_.capacity(),
            sparse_data_chunks_[sparse_data_chunk_index]->data_size(),
            streamer_meta_.data_chunk_size, sparse_vector_len);
      }
      return IndexError_IndexFull;
    }

    sparse_data_chunk = alloc_new_data_chunk(sparse_data_chunks_.size());
    if (ailego_unlikely(!sparse_data_chunk)) {
      LOG_ERROR("allocate data chunk failed");
      return IndexError_NoMemory;
    }
    sparse_data_chunks_.emplace_back(sparse_data_chunk);
    sparse_data_chunk_index = sparse_data_chunks_.size() - 1U;
    sparse_data_chunk_offset = 0UL;
  } else {
    sparse_data_chunk = sparse_data_chunks_[sparse_data_chunk_index];
    sparse_data_chunk_offset = sparse_data_chunk->data_size();
  }

  // write sparse vector
  if (sparse_vector.size() > 0) {
    if (ailego_unlikely(write_sparse_vector_data(
                            sparse_data_chunk_index, sparse_data_chunk_offset,
                            sparse_vector.data(), sparse_vector.size()) != 0)) {
      LOG_ERROR("write sparse vector failed");
      return IndexError_NoMemory;
    }
  }

  uint64_t sparse_offset = sparse_data_chunk_index;
  sparse_offset = (sparse_offset << 32U) + sparse_data_chunk_offset;

  // get sparse offset chunk and offset for write new info
  Chunk::Pointer sparse_offset_chunk;
  uint32_t sparse_offset_chunk_offset = -1U;
  uint32_t sparse_offset_chunk_index = sparse_offset_chunks_.size() - 1U;
  if (sparse_offset_chunk_index == -1U ||
      sparse_offset_chunks_[sparse_offset_chunk_index]->data_size() +
              offset_size_per_node() >
          streamer_meta_.offset_chunk_size) {
    // no space left and need to allocate new offset chunk
    if (ailego_unlikely(sparse_offset_chunks_.capacity() ==
                        sparse_offset_chunks_.size())) {
      LOG_ERROR("add vector failed for no memory quota");
      return IndexError_IndexFull;
    }

    sparse_offset_chunk = alloc_new_offset_chunk(sparse_offset_chunks_.size());
    if (ailego_unlikely(!sparse_offset_chunk)) {
      LOG_ERROR("allocate offset chunk failed");
      return IndexError_NoMemory;
    }
    sparse_offset_chunks_.emplace_back(sparse_offset_chunk);
    sparse_offset_chunk_index = sparse_offset_chunks_.size() - 1U;
    sparse_offset_chunk_offset = 0UL;
  } else {
    sparse_offset_chunk = sparse_offset_chunks_[sparse_offset_chunk_index];
    sparse_offset_chunk_offset = sparse_offset_chunk->data_size();
  }

  // write offset
  size_t size = sparse_offset_chunk->write(sparse_offset_chunk_offset,
                                           &sparse_offset, sizeof(uint64_t));
  if (ailego_unlikely(size != sizeof(uint64_t))) {
    LOG_ERROR("Chunk write sparse vec offset failed, ret=%zu", size);
    return IndexError_WriteData;
  }

  // write length
  size =
      sparse_offset_chunk->write(sparse_offset_chunk_offset + sizeof(uint64_t),
                                 &sparse_vector_len, sizeof(uint32_t));
  if (ailego_unlikely(size != sizeof(uint32_t))) {
    LOG_ERROR("Chunk write sparse vec len failed, ret=%zu", size);
    return IndexError_WriteData;
  }

  // write key
  size = sparse_offset_chunk->write(
      sparse_offset_chunk_offset + 2 * sizeof(uint64_t), &key,
      sizeof(uint64_t));
  if (ailego_unlikely(size != sizeof(uint64_t))) {
    LOG_ERROR("Chunk write key failed, ret=%zu", size);
    return IndexError_WriteData;
  }

  // LOG_INFO("Write sparse vector, key=%lu, offset chunk=%u, offset=%u,
  // len=%u",
  //          key, sparse_offset_chunk_index, sparse_offset_chunk_offset,
  //          offset_size_per_node());

  // LOG_INFO("Write sparse vector, key=%lu, data chunk=%u, offset=%u, len=%u",
  //          key, sparse_data_chunk_index, sparse_data_chunk_offset,
  //          sparse_vector_len);

  // resize chunk
  if (sparse_vector_len > 0) {
    sparse_data_chunk_offset += sparse_vector_len;
    if (ailego_unlikely(sparse_data_chunk->resize(sparse_data_chunk_offset) !=
                        sparse_data_chunk_offset)) {
      LOG_ERROR("Sparse Chunk resize to %u failed", sparse_data_chunk_offset);
      return IndexError_Runtime;
    }
  }

  // persist in keys_map
  {
    keys_map_lock_->lock();
    (*keys_map_)[key] = local_id;
    keys_map_lock_->unlock();
  }

  inc_doc_count();
  inc_total_sparse_count(sparse_count);

  return 0;
}

int FlatSparseStreamerEntity::add_vector_with_id(
    uint32_t id, const std::string &sparse_vector,
    const uint32_t sparse_count) {
  uint32_t sparse_vector_len = sparse_vector.size();

  sparse_vector_len = AlignSize(sparse_vector_len);

  if (sparse_vector_len > streamer_meta_.data_chunk_size) {
    LOG_ERROR(
        "Sparse Vector Length exceed the chunk size, sparse vec len: %u, chunk "
        "size: %u",
        sparse_vector_len, streamer_meta_.data_chunk_size);
    return IndexError_InvalidArgument;
  }

  std::lock_guard<std::mutex> lock(mutex_);

  if (id >= doc_cnt()) {
    for (auto i = doc_cnt(); i <= id; i++) {
      node_id_t local_id = doc_cnt();
      if (ailego_unlikely(local_id >= max_doc_cnt_)) {
        LOG_ERROR("Add vector failed for exceed max doc count: %u",
                  max_doc_cnt_);
        return IndexError_IndexFull;
      }
      uint32_t sparse_data_chunk_index, sparse_data_chunk_offset,
          sparse_offset_chunk_index, sparse_offset_chunk_offset;
      if (i < id) {
        write_sparse_vector_to_chunk("", 0, sparse_data_chunk_index,
                                     sparse_data_chunk_offset);
      } else {
        write_sparse_vector_to_chunk(sparse_vector, sparse_vector_len,
                                     sparse_data_chunk_index,
                                     sparse_data_chunk_offset);
      }
      uint64_t sparse_offset =
          ((uint64_t)sparse_data_chunk_index << 32U) + sparse_data_chunk_offset;
      get_new_sparse_offset_chunk(sparse_offset_chunk_index,
                                  sparse_offset_chunk_offset);
      uint64_t written_key = kInvalidKey;
      if (i == id) {
        written_key = i;
      }
      write_sparse_offset_to_chunk(sparse_offset_chunk_index,
                                   sparse_offset_chunk_offset, sparse_offset,
                                   sparse_vector_len, written_key);
      {
        keys_map_lock_->lock();
        (*keys_map_)[i] = written_key;
        keys_map_lock_->unlock();
      }
      inc_doc_count();
    }
  } else {
    uint32_t sparse_data_chunk_index, sparse_data_chunk_offset;
    write_sparse_vector_to_chunk(sparse_vector, sparse_vector_len,
                                 sparse_data_chunk_index,
                                 sparse_data_chunk_offset);
    uint64_t sparse_offset =
        ((uint64_t)sparse_data_chunk_index << 32U) + sparse_data_chunk_offset;
    uint32_t sparse_offset_chunk_index =
        id / get_offset_info_number_per_chunk();
    uint32_t sparse_offset_chunk_offset =
        id % get_offset_info_number_per_chunk() * offset_size_per_node();
    write_sparse_offset_to_chunk(sparse_offset_chunk_index,
                                 sparse_offset_chunk_offset, sparse_offset,
                                 sparse_vector_len, id);
    {
      keys_map_lock_->lock();
      (*keys_map_)[id] = id;
      keys_map_lock_->unlock();
    }
  }
  inc_total_sparse_count(sparse_count);
  return 0;
}

int FlatSparseStreamerEntity::write_sparse_vector_to_chunk(
    const std::string &sparse_vector, const uint32_t sparse_vector_len,
    uint32_t &sparse_data_chunk_index, uint32_t &sparse_data_chunk_offset) {
  // get sparse data chunk and offset for write sparse vector
  Chunk::Pointer sparse_data_chunk;
  sparse_data_chunk_offset = -1U;
  sparse_data_chunk_index = sparse_data_chunks_.size() - 1U;
  if (sparse_data_chunk_index == -1U ||
      sparse_data_chunks_[sparse_data_chunk_index]->data_size() +
              sparse_vector_len >
          streamer_meta_.data_chunk_size) {
    if (ailego_unlikely(sparse_data_chunks_.capacity() ==
                        sparse_data_chunks_.size())) {
      LOG_ERROR("add vector failed for no memory quota");
      if (sparse_data_chunk_index != -1U) {
        LOG_ERROR(
            "capacity: %zu, chunk used size: %zu, chunk size: %u, "
            "sparse_vector_len: %u",
            sparse_data_chunks_.capacity(),
            sparse_data_chunks_[sparse_data_chunk_index]->data_size(),
            streamer_meta_.data_chunk_size, sparse_vector_len);
      }
      return IndexError_IndexFull;
    }

    sparse_data_chunk = alloc_new_data_chunk(sparse_data_chunks_.size());
    if (ailego_unlikely(!sparse_data_chunk)) {
      LOG_ERROR("allocate data chunk failed");
      return IndexError_NoMemory;
    }
    sparse_data_chunks_.emplace_back(sparse_data_chunk);
    sparse_data_chunk_index = sparse_data_chunks_.size() - 1U;
    sparse_data_chunk_offset = 0UL;
  } else {
    sparse_data_chunk = sparse_data_chunks_[sparse_data_chunk_index];
    sparse_data_chunk_offset = sparse_data_chunk->data_size();
  }

  // write sparse vector
  if (sparse_vector.size() > 0) {
    if (ailego_unlikely(write_sparse_vector_data(
                            sparse_data_chunk_index, sparse_data_chunk_offset,
                            sparse_vector.data(), sparse_vector.size()) != 0)) {
      LOG_ERROR("write sparse vector failed");
      return IndexError_NoMemory;
    }
  }

  // resize chunk
  if (sparse_vector_len > 0) {
    uint32_t sparse_data_chunk_size =
        sparse_data_chunk_offset + sparse_vector_len;
    if (ailego_unlikely(sparse_data_chunk->resize(sparse_data_chunk_size) !=
                        sparse_data_chunk_size)) {
      LOG_ERROR("Sparse Chunk resize to %u failed", sparse_data_chunk_size);
      return IndexError_Runtime;
    }
  }
  return 0;
}

int FlatSparseStreamerEntity::get_new_sparse_offset_chunk(
    uint32_t &sparse_offset_chunk_index, uint32_t &sparse_offset_chunk_offset) {
  // get sparse offset chunk and offset for write new info
  Chunk::Pointer sparse_offset_chunk;
  sparse_offset_chunk_offset = -1U;
  sparse_offset_chunk_index = sparse_offset_chunks_.size() - 1U;
  if (sparse_offset_chunk_index == -1U ||
      sparse_offset_chunks_[sparse_offset_chunk_index]->data_size() +
              offset_size_per_node() >
          streamer_meta_.offset_chunk_size) {
    // no space left and need to allocate new offset chunk
    if (ailego_unlikely(sparse_offset_chunks_.capacity() ==
                        sparse_offset_chunks_.size())) {
      LOG_ERROR("add vector failed for no memory quota");
      return IndexError_IndexFull;
    }

    sparse_offset_chunk = alloc_new_offset_chunk(sparse_offset_chunks_.size());
    if (ailego_unlikely(!sparse_offset_chunk)) {
      LOG_ERROR("allocate offset chunk failed");
      return IndexError_NoMemory;
    }
    sparse_offset_chunks_.emplace_back(sparse_offset_chunk);
    sparse_offset_chunk_index = sparse_offset_chunks_.size() - 1U;
    sparse_offset_chunk_offset = 0UL;
  } else {
    sparse_offset_chunk = sparse_offset_chunks_[sparse_offset_chunk_index];
    sparse_offset_chunk_offset = sparse_offset_chunk->data_size();
  }
  return 0;
}

int FlatSparseStreamerEntity::write_sparse_offset_to_chunk(
    const uint32_t sparse_offset_chunk_index,
    const uint32_t sparse_offset_chunk_offset, const uint64_t sparse_offset,
    const uint32_t sparse_vector_len, const uint64_t node_id) {
  // write offset
  Chunk::Pointer sparse_offset_chunk =
      sparse_offset_chunks_[sparse_offset_chunk_index];
  size_t size = sparse_offset_chunk->write(sparse_offset_chunk_offset,
                                           &sparse_offset, sizeof(uint64_t));
  if (ailego_unlikely(size != sizeof(uint64_t))) {
    LOG_ERROR("Chunk write sparse vec offset failed, ret=%zu", size);
    return IndexError_WriteData;
  }

  // write length
  size =
      sparse_offset_chunk->write(sparse_offset_chunk_offset + sizeof(uint64_t),
                                 &sparse_vector_len, sizeof(uint32_t));
  if (ailego_unlikely(size != sizeof(uint32_t))) {
    LOG_ERROR("Chunk write sparse vec len failed, ret=%zu", size);
    return IndexError_WriteData;
  }

  // write key
  size = sparse_offset_chunk->write(
      sparse_offset_chunk_offset + 2 * sizeof(uint64_t), &node_id,
      sizeof(uint64_t));
  if (ailego_unlikely(size != sizeof(uint64_t))) {
    LOG_ERROR("Chunk write key failed, ret=%zu", size);
    return IndexError_WriteData;
  }
  return 0;
}

uint64_t FlatSparseStreamerEntity::get_key(node_id_t node_id) const {
  uint32_t offset_chunk_index = node_id / get_offset_info_number_per_chunk();
  uint32_t offset_chunk_key_offset =
      node_id % get_offset_info_number_per_chunk() * offset_size_per_node() +
      2 * sizeof(uint64_t);

  IndexStorage::MemoryBlock block;
  if (ailego_unlikely(sparse_offset_chunks_[offset_chunk_index]->read(
                          offset_chunk_key_offset, block, sizeof(uint64_t)) !=
                      sizeof(uint64_t))) {
    LOG_ERROR("Read key failed, offset=%u, node_id=%u", offset_chunk_key_offset,
              node_id);
    return kInvalidKey;
  };

  return *reinterpret_cast<const uint64_t *>(block.data());
}

int FlatSparseStreamerEntity::get_sparse_vector_ptr_by_id(
    node_id_t node_id, const void **sparse_vector_ptr,
    uint32_t *sparse_vector_len_ptr) const {
  uint32_t offset_chunk_index = node_id / get_offset_info_number_per_chunk();
  uint32_t offset_chunk_offset =
      node_id % get_offset_info_number_per_chunk() * offset_size_per_node();

  // LOG_DEBUG("Read sparse vector, offset chunk=%u, offset=%u, len=%u",
  //           offset_chunk_index, offset_chunk_offset, offset_size_per_node());

  auto offset_chunk = sparse_offset_chunks_[offset_chunk_index];

  const void *offset_info = nullptr;
  size_t read_len = offset_chunk->read(offset_chunk_offset, &offset_info,
                                       offset_size_per_node());
  if (ailego_unlikely(read_len != offset_size_per_node())) {
    LOG_ERROR("Read offset info failed, offset=%u, read_len=%zu, expect=%u",
              offset_chunk_offset, read_len, offset_size_per_node());
    return IndexError_ReadData;
  };

  // sparse offset
  uint64_t sparse_offset = *(uint64_t *)offset_info;
  uint32_t sparse_vector_len =
      *(uint32_t *)((uint8_t *)offset_info + sizeof(uint64_t));

  uint32_t sparse_data_chunk_index =
      static_cast<uint32_t>((sparse_offset >> 32) & 0xFFFFFFFF);
  uint32_t sparse_data_chunk_offset =
      static_cast<uint32_t>(sparse_offset & 0xFFFFFFFF);

  if (sparse_vector_len > 0) {
    const void *sparse_data = get_sparse_vector_data(
        sparse_data_chunk_index, sparse_data_chunk_offset, sparse_vector_len);
    if (ailego_unlikely(sparse_data == nullptr)) {
      LOG_ERROR("Get nullptr sparse, offset=%zu, len=%u", (size_t)sparse_offset,
                sparse_vector_len);

      return IndexError_ReadData;
    }
    *sparse_vector_ptr = sparse_data;
    *sparse_vector_len_ptr = sparse_vector_len;
  }

  // LOG_DEBUG("Read sparse vector, data chunk=%u, offset=%u, len=%u",
  //           sparse_data_chunk_index, sparse_data_chunk_offset,
  //           sparse_vector_len);

  return 0;
}

int FlatSparseStreamerEntity::get_sparse_vector_ptr_by_id(
    node_id_t node_id, IndexStorage::MemoryBlock &sparse_vector_block,
    uint32_t *sparse_vector_len_ptr) const {
  uint32_t offset_chunk_index = node_id / get_offset_info_number_per_chunk();
  uint32_t offset_chunk_offset =
      node_id % get_offset_info_number_per_chunk() * offset_size_per_node();

  // LOG_DEBUG("Read sparse vector, offset chunk=%u, offset=%u, len=%u",
  //           offset_chunk_index, offset_chunk_offset, offset_size_per_node());

  auto offset_chunk = sparse_offset_chunks_[offset_chunk_index];

  const void *offset_info = nullptr;
  IndexStorage::MemoryBlock offset_info_block;
  size_t read_len = offset_chunk->read(offset_chunk_offset, offset_info_block,
                                       offset_size_per_node());
  if (ailego_unlikely(read_len != offset_size_per_node())) {
    LOG_ERROR("Read offset info failed, offset=%u, read_len=%zu, expect=%u",
              offset_chunk_offset, read_len, offset_size_per_node());
    return IndexError_ReadData;
  };
  offset_info = offset_info_block.data();

  // sparse offset
  uint64_t sparse_offset = *(uint64_t *)offset_info;
  uint32_t sparse_vector_len =
      *(uint32_t *)((uint8_t *)offset_info + sizeof(uint64_t));

  uint32_t sparse_data_chunk_index =
      static_cast<uint32_t>((sparse_offset >> 32) & 0xFFFFFFFF);
  uint32_t sparse_data_chunk_offset =
      static_cast<uint32_t>(sparse_offset & 0xFFFFFFFF);

  if (sparse_vector_len > 0) {
    get_sparse_vector_data(sparse_data_chunk_index, sparse_data_chunk_offset,
                           sparse_vector_len, sparse_vector_block);
    if (ailego_unlikely(sparse_vector_block.data() == nullptr)) {
      LOG_ERROR("Get nullptr sparse, offset=%zu, len=%u", (size_t)sparse_offset,
                sparse_vector_len);

      return IndexError_ReadData;
    }
    *sparse_vector_len_ptr = sparse_vector_len;
  }

  return 0;
}

int FlatSparseStreamerEntity::write_sparse_vector_data(uint32_t chunk_index,
                                                       uint64_t offset,
                                                       const void *data,
                                                       uint32_t length) {
  auto size = sparse_data_chunks_[chunk_index]->write(offset, data, length);
  if (size != length) {
    LOG_ERROR(
        "write sparse vector data failed: chunk_index=%u, offset=%zu, "
        "length=%u, size=%zu, chunk_data_size=%zu",
        chunk_index, (size_t)offset, length, size,
        sparse_data_chunks_[chunk_index]->data_size());
    return IndexError_WriteData;
  }
  // LOG_DEBUG(
  //     "write_sparse_vector_data: chunk_index=%u, offset=%lu, length=%u, "
  //     "data=%p",
  //     chunk_index, offset, length, data);
  return 0;
}

const void *FlatSparseStreamerEntity::get_sparse_vector_data(
    uint32_t chunk_index, uint64_t offset, uint32_t length) const {
  const void *data;
  auto size = sparse_data_chunks_[chunk_index]->read(offset, &data, length);
  if (size != length) {
    LOG_ERROR(
        "read sparse vector data failed: chunk_index=%u, offset=%zu, "
        "length=%u, size=%zu",
        chunk_index, (size_t)offset, length, size);
    return nullptr;
  }
  // LOG_DEBUG(
  //     "get_sparse_vector_data: chunk_index=%u, offset=%lu, length=%u, "
  //     "data=%p",
  //     chunk_index, offset, length, data);
  return data;
}

int FlatSparseStreamerEntity::get_sparse_vector_data(
    uint32_t chunk_index, uint64_t offset, uint32_t length,
    IndexStorage::MemoryBlock &block) const {
  auto size = sparse_data_chunks_[chunk_index]->read(offset, block, length);
  if (size != length) {
    LOG_ERROR(
        "read sparse vector data failed: chunk_index=%u, offset=%zu, "
        "length=%u, size=%zu",
        chunk_index, (size_t)offset, length, size);
    return IndexError_ReadData;
  }
  return 0;
}

}  // namespace core
}  // namespace zvec
