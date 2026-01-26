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

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <ailego/parallel/lock.h>
#include <framework/index_dumper.h>
#include <framework/index_meta.h>
#include <framework/index_storage.h>
#include <framework/index_streamer.h>
#include <zvec/ailego/utility/string_helper.h>
#include "flat_sparse_entity.h"
#include "flat_sparse_index_format.h"
#include "flat_sparse_utility.h"

namespace zvec {
namespace core {


/*! Flat Sparse Streamer Entity
 */
class FlatSparseStreamerEntity : public FlatSparseEntity {
 public:
  typedef std::shared_ptr<FlatSparseStreamerEntity> Pointer;

  using Chunk = IndexStorage::Segment;

  //! Constructor
  explicit FlatSparseStreamerEntity(IndexStreamer::Stats &stats);

  //! Destructor
  virtual ~FlatSparseStreamerEntity() = default;

  //! Disable them
  FlatSparseStreamerEntity(const FlatSparseStreamerEntity &) = delete;
  FlatSparseStreamerEntity &operator=(const FlatSparseStreamerEntity &) =
      delete;

  //! Open the entity with storage
  int open(IndexStorage::Pointer storage, const IndexMeta &meta);

  //! Close the entity
  int close();

  //! Flush linear index to storage
  int flush(uint64_t checkpoint);

  //! Dump index by dumper
  int dump(const IndexDumper::Pointer &dumper);

  //! Add sparse vector to linear index
  int add(uint64_t key, const std::string &sparse_vector,
          const uint32_t sparse_count);

  //! Add sparse vector to linear index with id
  int add_vector_with_id(uint32_t id, const std::string &sparse_vector,
                         uint32_t sparse_count);

  //! Clone entity
  FlatSparseStreamerEntity::Pointer clone() const;

  int get_index_sparse_meta(IndexMeta *meta) const {
    return IndexHelper::DeserializeFromStorage(storage_.get(), meta);
  }

  int set_index_sparse_meta(const IndexMeta &meta) const {
    return IndexHelper::SerializeToStorage(meta, storage_.get());
  }

 public:
  inline uint32_t doc_cnt() const override {
    return meta_.doc_cnt;
  }

  inline uint32_t total_sparse_count() const override {
    return meta_.total_sparse_count;
  }

  size_t sparse_unit_size() const override {
    return sparse_unit_size_;
  }

  inline node_id_t get_id(uint64_t key) const override {
    keys_map_lock_->lock_shared();
    auto it = keys_map_->find(key);
    keys_map_lock_->unlock_shared();
    return it == keys_map_->end() ? kInvalidNodeId : it->second;
  }

  uint64_t get_key(node_id_t node_id) const override;

  int get_sparse_vector_ptr_by_id(node_id_t id, const void **sparse_vector,
                                  uint32_t *sparse_vector_len) const override;
  int get_sparse_vector_ptr_by_id(
      const node_id_t id, IndexStorage::MemoryBlock &sparse_vector_block,
      uint32_t *sparse_vector_len) const override;

  float get_search_distance(const std::string &vector,
                            node_id_t target_node_id) const override {
    float dist;
    const void *target_vector;
    uint32_t target_vector_len;
    get_sparse_vector_ptr_by_id(target_node_id, &target_vector,
                                &target_vector_len);
    search_sparse_distance_(vector.c_str(), target_vector, &dist);
    return dist;
  }

 private:
  void inc_doc_count() {
    meta_.doc_cnt++;
  }
  void inc_total_sparse_count(uint32_t count) {
    meta_.total_sparse_count += count;
  }

  int init_metric(const IndexMeta &meta);

  int init_storage(IndexStorage::Pointer storage, const IndexMeta &meta);

  int load_storage(IndexStorage::Pointer storage, const IndexMeta &meta);

  static inline size_t AlignSize(size_t size) {
    return (size + 0x1F) & (~0x1F);
  }

  inline uint32_t offset_size_per_node() const {
    return 3 * sizeof(uint64_t);
  }

  inline uint32_t doc_cnt_per_offset_chunk() const {
    return streamer_meta_.offset_chunk_size / offset_size_per_node();
  }

  Chunk::Pointer alloc_new_offset_chunk(uint32_t chunk_id) {
    std::string segment_id = ailego::StringHelper::Concat(
        PARAM_FLAT_SPARSE_OFFSET_SEG_ID_PREFIX, chunk_id);
    // LOG_INFO("Alloc new offset chunk %s", segment_id.c_str());
    return alloc_new_chunk(segment_id, streamer_meta_.offset_chunk_size);
  }

  Chunk::Pointer alloc_new_data_chunk(uint32_t chunk_id) {
    std::string segment_id = ailego::StringHelper::Concat(
        PARAM_FLAT_SPARSE_DATA_SEG_ID_PREFIX, chunk_id);
    // LOG_INFO("Alloc new data chunk %s", segment_id.c_str());
    return alloc_new_chunk(segment_id, streamer_meta_.data_chunk_size);
  }

  Chunk::Pointer alloc_new_chunk(const std::string &segment_id, uint32_t size) {
    int ret = storage_->append(segment_id, size);
    if (ailego_unlikely(ret != 0)) {
      return nullptr;
    }
    *stats_.mutable_index_size() += size;
    return storage_->get(segment_id);
  }

  inline uint32_t get_offset_info_number_per_chunk() const {
    return streamer_meta_.offset_chunk_size / offset_size_per_node();
  }

  int write_sparse_vector_to_chunk(const std::string &sparse_vector,
                                   const uint32_t sparse_vector_len,
                                   uint32_t &sparse_data_chunk_index,
                                   uint32_t &sparse_data_chunk_offset);

  int get_new_sparse_offset_chunk(uint32_t &sparse_offset_chunk_index,
                                  uint32_t &sparse_offset_chunk_offset);

  int write_sparse_offset_to_chunk(const uint32_t sparse_offset_chunk_index,
                                   const uint32_t sparse_offset_chunk_offset,
                                   const uint64_t sparse_offset,
                                   const uint32_t sparse_vector_len,
                                   const uint64_t node_id);

  int write_sparse_vector_data(uint32_t chunk_index, uint64_t offset,
                               const void *data, uint32_t length);

  const void *get_sparse_vector_data(uint32_t chunk_index, uint64_t offset,
                                     uint32_t length) const;

  int get_sparse_vector_data(uint32_t chunk_index, uint64_t offset,
                             uint32_t length,
                             IndexStorage::MemoryBlock &block) const;

  int dump_sparse_vector_data(const void *data, uint32_t length,
                              IndexDumper *dumper);

  int dump_meta(IndexDumper *dumper);

  int dump_index_meta(IndexDumper *dumper);

  int dump_keys(const std::vector<uint64_t> &keys, IndexDumper *dumper);

  int dump_mapping(const std::vector<uint64_t> &keys, IndexDumper *dumper);

  int dump_offset_data(IndexDumper *dumper);


 private:
  FlatSparseStreamerEntity(
      IndexStreamer::Stats &stats, const FlatSparseMeta &meta,
      const FlatSparseStreamerMeta &streamer_meta,
      std::shared_ptr<ailego::SharedMutex> keys_map_lock,
      std::shared_ptr<std::map<uint64_t, node_id_t>> keys_map,
      std::vector<Chunk::Pointer> sparse_data_chunks,
      std::vector<Chunk::Pointer> sparse_offset_chunks)
      : stats_(stats),
        meta_(meta),
        streamer_meta_(streamer_meta),
        keys_map_lock_(keys_map_lock),
        keys_map_(keys_map),
        sparse_data_chunks_(std::move(sparse_data_chunks)),
        sparse_offset_chunks_(std::move(sparse_offset_chunks)) {}

 private:
  IndexStorage::Pointer storage_{};
  IndexStreamer::Stats &stats_;

  // meta
  FlatSparseMeta meta_;
  FlatSparseStreamerMeta streamer_meta_;

  // metric
  IndexMetric::Pointer metric_{};
  IndexMetric::MatrixSparseDistance search_sparse_distance_{};

  std::mutex mutex_{};

  // keys map
  mutable std::shared_ptr<ailego::SharedMutex> keys_map_lock_{};
  std::shared_ptr<std::map<uint64_t, node_id_t>> keys_map_{};

  // chunks
  mutable std::vector<Chunk::Pointer> sparse_data_chunks_{};
  mutable std::vector<Chunk::Pointer> sparse_offset_chunks_{};

  // config
  uint32_t max_doc_cnt_{1 << 24U};  // 16 million
  uint32_t max_data_chunk_cnt_{
      1 << 10U};  // 1024, default single_data_chunk_size = 8M,
                  // default_total_max = 1024 * 8M = 8G

  uint64_t dump_size_{0U};
  size_t sparse_unit_size_{0U};
};

}  // namespace core
}  // namespace zvec
