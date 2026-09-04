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

#include <atomic>
#include <memory>
#include <unordered_map>
#include <vector>
#include <ailego/parallel/lock.h>
#include <ailego/utility/memory_helper.h>
#include <zvec/ailego/utility/string_helper.h>
#include <zvec/core/framework/index_context.h>
#include <zvec/core/framework/index_framework.h>
#include <zvec/core/framework/index_streamer.h>
#include "flat_index_format.h"
#include "flat_utility.h"

namespace zvec {
namespace core {

//! Reusable request-local buffers for storage-specific Flat search paths.
struct FlatSearchScratch {
  std::vector<const void *> vector_ptrs{};
  std::vector<const void *> extra_values{};
  std::vector<uint64_t> vector_keys{};
  std::vector<float> distances{};
  std::vector<uint8_t> query_buffer{};
};

/*! Flat Streamer Entity
 */
class FlatStreamerEntity {
 public:
  typedef std::shared_ptr<FlatStreamerEntity> Pointer;

  //! Constructor
  explicit FlatStreamerEntity(IndexStreamer::Stats &stats);

  //! Destructor
  virtual ~FlatStreamerEntity(void) = default;

  //! Open the entity with storage
  int open(IndexStorage::Pointer storage, const IndexMeta &mt);

  //! Close the entity
  virtual int close(void);

  //! Flush Linear Meta information to storage
  int flush_linear_meta(void);

  //! Flush linear index to storage
  int flush(uint64_t checkpoint);

  //! Add vector to linear index
  virtual int add(uint64_t key, const void *vec, size_t size);

  //! Search in linear list with filter
  virtual int search(const void *query, const IndexFilter &filter,
                     uint32_t *scan_count, IndexDocumentHeap *heap,
                     IndexContext::Stats *context_stats,
                     FlatSearchScratch *scratch = nullptr,
                     size_t batch_size = 0) const;

  //! Search the requested primary keys.
  virtual int search_by_p_keys(const void *query,
                               const std::vector<uint64_t> &p_keys,
                               const IndexFilter &filter,
                               IndexDocumentHeap *heap,
                               FlatSearchScratch *scratch = nullptr,
                               size_t batch_size = 0) const;

  //! Search in a block
  void search_block(const void *query, const BlockLocation &bl,
                    const BlockHeader *hd, float norm_val,
                    IndexDocumentHeap *heap) const;

  //! Search in a block with filter
  void search_block(const void *query, const BlockLocation &bl,
                    const BlockHeader *hd, float norm_val,
                    const IndexFilter &filter, const DeletionMap *deletion_map,
                    IndexDocumentHeap *heap,
                    IndexContext::Stats *context_stats) const;

  //! Flat Search with filter
  int search_bf(const void *query, const IndexFilter &filter,
                IndexDocumentHeap *heap,
                IndexContext::Stats *context_stats) const;

  //! Clone the entity
  virtual FlatStreamerEntity::Pointer clone(void) const;

  //! Retrieve the total vectors in the index
  size_t vector_count(void) const {
    return meta_.header.total_vector_count;
  }

  //! Retrieve the linear list count
  size_t linear_list_count(void) const {
    return meta_.header.linear_list_count;
  }

  //! Retrieve block size of the linear vector
  size_t linear_block_size(void) const {
    return meta_.header.block_size;
  }

  //! Retrieve the vectors count in one block
  size_t block_vector_count(void) const {
    // assert(meta_.header.block_vector_count == 32);
    return meta_.header.block_vector_count;
  }

  //! Retrieve IndexMeta of the linear index
  const IndexMeta &meta(void) const {
    return index_meta_;
  }

  //! Retrieve mutable IndexMeta of the linear index
  IndexMeta *mutable_meta(void) {
    return &index_meta_;
  }

  //! Retrieve vector by local id
  virtual const void *get_vector_by_key(uint64_t key) const;

  virtual int get_vector_by_key(const uint64_t key,
                                IndexStorage::MemoryBlock &block) const;

  //! Create a new iterator
  IndexProvider::Iterator::Pointer creater_iterator(void) const;


  //! Set params
  void set_block_vector_count(uint32_t count) {
    meta_.header.block_vector_count = count;
  }

  void set_use_key_info_map(bool use_id_map) {
    use_key_info_map_ = use_id_map;
    LOG_DEBUG("use_key_info_map_: %d", (int)use_key_info_map_);
  }

  //! Set params
  void set_segment_size(uint32_t size) {
    meta_.segment_size = size;
  }

  //! Set params
  void set_linear_list_count(uint32_t count) {
    meta_.header.linear_list_count = count;
  }

  //! Set params
  void enable_filter_same_key(bool enabled) {
    filter_same_key_ = enabled;
  }

  inline uint64_t key(uint32_t id) const {
    if (id < id_key_vector_.size()) {
      return id_key_vector_[id];
    } else {
      return kInvalidKey;
    }
  }

  inline void row_major_distance(const void *query, const void *feature,
                                 size_t fnum, float *out) const {
    const uint8_t *cur_feature = reinterpret_cast<const uint8_t *>(feature);
    for (size_t f = 0; f < fnum; ++f) {
      row_distance_(query, cur_feature, index_meta_.dimension(), out + f);
      cur_feature += index_meta_.element_size();
    }
  }

  virtual int add_vector_with_id(const uint32_t id, const void *query,
                                 const uint32_t element_size);

 protected:
  const IndexMetric::MatrixDistance &distance(void) const {
    return row_distance_;
  }

  const IndexMetric::MatrixBatchDistance &batch_distance(void) const {
    return batch_distance_;
  }

  size_t extra_values_size(void) const {
    return extra_values_size_;
  }

  const IndexMetric::DistanceBatchQueryPreprocessFunc &batch_query_preprocess(
      void) const {
    return batch_query_preprocess_;
  }

  int get_vector_by_position(uint32_t id,
                             IndexStorage::MemoryBlock &block) const;

  int get_key_by_position(uint32_t id, uint64_t *key) const;

  size_t id_key_count(void) const {
    return id_key_vector_.size();
  }

  bool use_key_info_map(void) const {
    return use_key_info_map_;
  }

 private:
  //! Disable them
  FlatStreamerEntity(const FlatStreamerEntity &) = delete;
  FlatStreamerEntity &operator=(const FlatStreamerEntity &) = delete;

  /*! Iterator of all the linear list
   */
  class Iterator : public IndexProvider::Iterator {
   public:
    //! Constructor
    Iterator(const FlatStreamerEntity::Pointer &entity) : entity_(entity) {
      this->read_next_block();
    }
    //! Retrieve pointer of data
    const void *data(void) const override {
      return reinterpret_cast<const char *>(data_) +
             block_vector_index_ * entity_->index_meta_.element_size();
    }
    //! Test if the iterator is valid
    bool is_valid(void) const override {
      return is_valid_;
    }
    //! Retrieve primary key
    uint64_t key(void) const override {
      return keys_[block_vector_index_];
    }
    //! Next iterator
    void next(void) override {
      if (++block_vector_index_ == block_vector_count_) {
        ++block_index_;
        this->read_next_block();
      }
    }

   private:
    //! Read next non-empty block
    void read_next_block(void);

    //! Members
    std::string buffer_{};
    const FlatStreamerEntity::Pointer entity_;
    IndexStorage::MemoryBlock block_;
    const void *data_{nullptr};
    const uint64_t *keys_{nullptr};
    uint32_t segment_id_{1u};  // The first segment is header info
    uint32_t block_index_{0u};
    uint32_t block_vector_index_{0u};
    uint32_t block_vector_count_{0u};
    bool is_valid_{true};
  };

  //! Retrive storage segment by index
  const IndexStorage::Segment::Pointer get_segment(size_t index) const {
    for (size_t i = segments_.size(); i <= index; ++i) {
      auto segment_id =
          ailego::StringHelper::Concat(FLAT_SEGMENT_FEATURES_SEG_ID, i);
      auto segment = storage_->get(segment_id);
      if (!segment) {
        LOG_ERROR("Failed to get segment %s", segment_id.c_str());
        return IndexStorage::Segment::Pointer();
      }
      segments_.emplace_back(std::move(segment));
    }
    return segments_[index];
  }

  //! Rejust the segment size as to aligned by page size
  void AdjustSegmentSize(StreamerLinearMeta *mt) {
    if (mt->segment_size < mt->header.block_size) {
      mt->segment_size = mt->header.block_size;
    }
    mt->segment_size = ailego_align(
        mt->segment_size / mt->header.block_size * mt->header.block_size,
        ailego::MemoryHelper::PageSize());
  }

  //! Init with an empty storage
  int init_storage(IndexStorage::Pointer storage);

  //! Load linear meta information from storage
  int load_linear_meta(IndexStorage::Pointer storage);

  //! Load keys to keys map
  int load_segment_keys_to_map(BlockLocation block);

  //! Load keys to keys map
  int load_segment_keys_to_vector(void);

  //! Load index from storage
  int load_storage(IndexStorage::Pointer storage);

  //! Check whether the block is empty
  bool is_valid_block(const BlockLocation &block) const {
    return block.segment_id != 0;
  }

  //! Update header block of an linear list
  int update_head_block(const BlockLocation &block) {
    ailego_assert_with(segments_.size() != 0, "Invalid Segments");

    auto &hd_segment = segments_[0];
    if (hd_segment->write(0, &block, sizeof(block)) != sizeof(block)) {
      LOG_ERROR("Failed to write head block location");
      return IndexError_WriteData;
    }

    return 0;
  }

  //! Alloc a new segment
  int alloc_segment(void);

  //! Alloc a new block
  int alloc_block(const BlockLocation &next, BlockLocation *block);

  //! Add a record to a block
  int add_to_block(const BlockLocation &block, uint64_t key, const void *data,
                   size_t size);

 private:
  size_t get_block_offset(uint32_t block_index) const {
    return block_index * linear_block_size();
  }

  size_t get_block_header_offset(uint32_t block_index) const {
    return get_block_offset(block_index) + linear_block_size() -
           sizeof(BlockHeader);
  }

  size_t get_block_deletion_map_offset(uint32_t block_index) const {
    return get_block_header_offset(block_index) - sizeof(DeletionMap);
  }

  size_t get_block_key_offset(uint32_t block_index,
                              uint32_t vector_index) const {
    return get_block_offset(block_index) +
           block_vector_count() * index_meta_.element_size() +
           sizeof(uint64_t) * vector_index;
  }

  size_t get_block_vector_offset(uint32_t block_index,
                                 uint32_t vector_index) const {
    return this->get_block_offset(block_index) +
           vector_index * index_meta_.element_size();
  }

  //! Get header block of an linear list
  int get_head_block(IndexStorage::MemoryBlock &header_block) const {
    ailego_assert_with(segments_.size() != 0, "Invalid Segments");
    auto &hd_segment = segments_[0];
    if (hd_segment->read(0, header_block, sizeof(BlockLocation)) !=
        sizeof(BlockLocation)) {
      LOG_ERROR("Failed to read head block location");
      return -1;
    }
    return 0;
  }

  //! Get BlockHeader of the block
  int get_block_header(const BlockLocation &block,
                       IndexStorage::MemoryBlock &header_block) const {
    // The header is located in the end of a block to align features
    auto &segment = this->get_segment(block.segment_id);
    ailego_assert_with(segment != nullptr, "Index Overflow");
    size_t off = this->get_block_header_offset(block.block_index);
    if (segment->read(off, header_block, sizeof(BlockHeader)) !=
        sizeof(BlockHeader)) {
      LOG_ERROR("Failed to read block header, off=%zu", off);
      return -1;
    }
    return 0;
  }
  int get_block_deletion_map(
      const BlockLocation &block,
      IndexStorage::MemoryBlock &deletion_map_block) const {
    auto &segment = this->get_segment(block.segment_id);
    ailego_assert_with(segment != nullptr, "Index Overflow");
    size_t off = this->get_block_deletion_map_offset(block.block_index);
    if (segment->read(off, deletion_map_block, sizeof(DeletionMap)) !=
        sizeof(DeletionMap)) {
      LOG_ERROR("Failed to read deletion map, off=%zu", off);
      return -1;
    }
    return 0;
  }

  int get_block_keys(const BlockLocation &block,
                     IndexStorage::MemoryBlock &keys_block) const {
    auto &segment = this->get_segment(block.segment_id);
    ailego_assert_with(segment != nullptr, "Index Overflow");
    size_t off = this->get_block_key_offset(block.block_index, 0);
    if (segment->read(off, keys_block,
                      block_vector_count() * sizeof(uint64_t)) !=
        block_vector_count() * sizeof(uint64_t)) {
      LOG_ERROR("Failed to read block header, off=%zu", off);
      return -1;
    }
    return 0;
  }

  int get_block_vectors(const BlockLocation &block,
                        IndexStorage::MemoryBlock &vector_block) const {
    auto &segment = this->get_segment(block.segment_id);
    ailego_assert_with(segment != nullptr, "Index Overflow");
    size_t off = this->get_block_vector_offset(block.block_index, 0);
    if (segment->read(off, vector_block,
                      block_vector_count() * index_meta_.element_size()) !=
        block_vector_count() * index_meta_.element_size()) {
      LOG_ERROR("Failed to read block header, off=%zu", off);
      return -1;
    }
    return 0;
  }

 private:
  //! Constants
  static constexpr size_t kMaxSegmentId = std::numeric_limits<uint32_t>::max();
  static constexpr size_t kMaxBlockId = std::numeric_limits<uint32_t>::max();

  //! Members
  std::mutex mutex_{};
  IndexMeta index_meta_{};
  IndexStorage::Pointer storage_{};
  IndexMetric::MatrixDistance row_distance_{}, column_distance_{};
  IndexMetric::MatrixBatchDistance batch_distance_{};
  IndexMetric::DistanceBatchQueryPreprocessFunc batch_query_preprocess_{};
  size_t extra_values_size_{0};
  mutable std::vector<IndexStorage::Segment::Pointer> segments_{};
  IndexStreamer::Stats &stats_;
  mutable std::shared_ptr<ailego::SharedMutex> key_info_map_lock_{};
  std::unordered_map<uint64_t, VectorLocation> key_info_map_{};
  std::vector<VectorLocation> withid_key_info_map_{};
  std::vector<uint32_t> withid_key_map_{};
  std::vector<uint64_t> id_key_vector_{};
  bool filter_same_key_{false};
  bool use_key_info_map_{true};
  uint32_t vec_unit_size_{0};
  uint32_t vec_cols_{0};
  mutable std::string vec_buf_{};
  StreamerLinearMeta meta_{};
};

/*! A Flat entity backed by aligned row-major contiguous memory. */
class FlatContiguousStreamerEntity : public FlatStreamerEntity {
 public:
  explicit FlatContiguousStreamerEntity(IndexStreamer::Stats &stats)
      : FlatStreamerEntity(stats) {}
  ~FlatContiguousStreamerEntity(void) override = default;

  int build_contiguous_memory(void);
  int close(void) override;

  int add(uint64_t key, const void *vec, size_t size) override;
  int add_vector_with_id(uint32_t id, const void *query,
                         uint32_t element_size) override;

  int search(const void *query, const IndexFilter &filter, uint32_t *scan_count,
             IndexDocumentHeap *heap, IndexContext::Stats *context_stats,
             FlatSearchScratch *scratch, size_t batch_size) const override;
  int search_by_p_keys(const void *query, const std::vector<uint64_t> &p_keys,
                       const IndexFilter &filter, IndexDocumentHeap *heap,
                       FlatSearchScratch *scratch,
                       size_t batch_size) const override;

  bool is_contiguous(void) const {
    return !!load_contiguous_storage();
  }

 private:
  struct ContiguousDeleter {
    size_t size;
    void operator()(char *ptr) const {
      ailego::MemoryHelper::FreeHugePage(ptr, size);
    }
  };

  struct ContiguousStorage {
    std::shared_ptr<char> vector_memory{};
    size_t vector_stride{0};
    size_t vector_count{0};
    std::unordered_map<uint64_t, uint32_t> key_to_position{};
    std::vector<uint64_t> position_to_key{};
  };

  // Same lifetime model as HNSW/Vamana contiguous entity clones: a search
  // keeps shared ownership of one immutable generation while add() removes
  // the entity's owner reference and falls back to mmap storage.
  std::shared_ptr<const ContiguousStorage> load_contiguous_storage(void) const {
    return std::atomic_load_explicit(&contiguous_storage_,
                                     std::memory_order_acquire);
  }

  void degrade_to_mmap(void);
  int evaluate_distances(const ContiguousStorage &storage, const void *query,
                         const std::vector<uint64_t> *p_keys,
                         const IndexFilter &filter, size_t batch_size,
                         FlatSearchScratch *scratch,
                         IndexContext::Stats *context_stats,
                         IndexDocumentHeap *heap) const;
  const void *get_vector_ptr(const ContiguousStorage &storage,
                             size_t position) const;
  const void *get_vector_ptr_by_key(const ContiguousStorage &storage,
                                    uint64_t key) const;

  static constexpr size_t kVectorAlignment = 64;
  std::shared_ptr<const ContiguousStorage> contiguous_storage_{};
};

}  // namespace core
}  // namespace zvec
