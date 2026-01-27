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

#include <core/quantizer/quantizer_params.h>
#include <zvec/core/framework/index_framework.h>
#include "metric/metric_params.h"
#include "ivf_distance_calculator.h"
#include "ivf_index_format.h"
#include "ivf_params.h"

namespace zvec {
namespace core {

/*! IVF Entity
 */
class IVFEntity {
 public:
  typedef std::shared_ptr<IVFEntity> Pointer;

  class IVFReformerWrapper;

  //! Constructor
  IVFEntity() {}

  //! Destructor
  virtual ~IVFEntity() {}

  //! Disable them
  IVFEntity(const IVFEntity &) = delete;
  IVFEntity &operator=(const IVFEntity &) = delete;

  //! load the index from container
  virtual int load(const IndexStorage::Pointer &container);

  //! search in inverted list with filter
  int search(size_t inverted_list_id, const void *query,
             const IndexFilter &filter, uint32_t *scan_count,
             IndexDocumentHeap *heap, IndexContext::Stats *context_stats) const;

  //! search in inverted list without filter
  int search(size_t inverted_list_id, const void *query, uint32_t *scan_count,
             IndexDocumentHeap *heap, IndexContext::Stats *context_stats) const;

  //! search all inverted list with filter
  int search(const void *query, const IndexFilter &filter,
             IndexDocumentHeap *heap, IndexContext::Stats *context_stats) const;

  //! search all inverted list without filter
  int search(const void *query, IndexDocumentHeap *heap,
             IndexContext::Stats *context_stats) const;

  //! Clone the entity
  virtual IVFEntity::Pointer clone(void) const;

  //! Clone the entity
  IVFEntity::Pointer clone(const IVFEntity::Pointer &entity) const;

  //! Retrieve the primary keys by local id in heap
  int retrieve_keys(IndexDocumentHeap *heap) const {
    for (auto &it : (*heap)) {
      uint64_t key = this->get_key(it.index());
      if (key == kInvalidKey) {
        return IndexError_ReadData;
      }
      it.set_key(key);
    }

    return 0;
  }

  //! Retrieve the total vectors in the index
  size_t vector_count(void) const {
    return header_.total_vector_count;
  }

  //! Retrieve the inverted list count
  size_t inverted_list_count(void) const {
    return header_.inverted_list_count;
  }

  //! Retrieve block size of the inverted vector
  size_t inverted_block_size(void) const {
    return header_.block_size;
  }

  //! Retrieve the vectors count in one block
  size_t block_vector_count(void) const {
    return header_.block_vector_count;
  }

  //! Retrieve IndexMeta of the inverted index
  const IndexMeta &meta(void) const {
    return meta_;
  }

  //! Retrieve a block of vectors
  const void *read_block(size_t inverted_list_id, size_t local_block_id,
                         size_t *vecs_count) const {
    auto iv_meta = this->inverted_list_meta(inverted_list_id);
    if (!iv_meta || local_block_id >= iv_meta->block_count) {
      LOG_ERROR("Failed to read inverted list, listId=%zu blockIdx=%zu",
                inverted_list_id, local_block_id);
      return nullptr;
    }

    size_t block_vecs = header_.block_vector_count;
    *vecs_count = std::min(block_vecs,
                           iv_meta->vector_count - local_block_id * block_vecs);
    ailego_assert_with(*vecs_count <= header_.block_vector_count,
                       "invalid vecs");
    const size_t off = iv_meta->offset + local_block_id * header_.block_size;
    const size_t size = *vecs_count * meta_.element_size();
    const void *data = nullptr;
    if (inverted_->read(off, &data, size) != size) {
      LOG_ERROR("Failed to read block off=%zu size=%zu", off, size);
      return nullptr;
    }

    return data;
  }

  //! Retrieve the inverted list meta
  const InvertedListMeta *inverted_list_meta(size_t inverted_list_id) const {
    const void *data = nullptr;
    const size_t size = sizeof(InvertedListMeta);
    const size_t offset = inverted_list_id * size;
    if (inverted_meta_->read(offset, &data, size) != size) {
      LOG_ERROR("Failed to read inverted meta, id=%zu, size=%zu",
                inverted_list_id, size);
      return nullptr;
    }

    return static_cast<const InvertedListMeta *>(data);
  }

  //! Retrieve the keys by consecutive local ids
  const uint64_t *get_keys(size_t id, size_t count) const {
    const void *data = nullptr;
    const size_t offset = id * sizeof(uint64_t);
    const size_t size = count * sizeof(uint64_t);
    if (keys_->read(offset, &data, size) != size) {
      LOG_ERROR("Failed to read keys, id=%zu, size=%zu", id, size);
      return nullptr;
    }

    return static_cast<const uint64_t *>(data);
  }

  //! Retrieve the key by local id
  uint64_t get_key(size_t id) const {
    const void *data = nullptr;
    const size_t offset = id * sizeof(uint64_t);
    const size_t size = sizeof(uint64_t);
    if (keys_->read(offset, &data, size) != size) {
      LOG_ERROR("Failed to read key, id=%zu", id);
      return kInvalidKey;
    }

    return *static_cast<const uint64_t *>(data);
  }

  //! Retrieve vector by local id
  const void *get_vector(size_t id) const;

  //! Retrieve vector by local id
  const void *get_vector_by_key(uint64_t key) const;

  int get_vector(size_t id, IndexStorage::MemoryBlock &block) const;

  int get_vector_by_key(uint64_t key, IndexStorage::MemoryBlock &block) const;

  uint32_t key_to_id(uint64_t key) const;

  //! Transform a query
  int transform(const void *query, const IndexQueryMeta &qmeta,
                const void **out, IndexQueryMeta *ometa) const {
    return reformer_.transform(query, qmeta, out, ometa);
  }

  //! Transform queries
  int transform(const void *query, const IndexQueryMeta &qmeta, uint32_t count,
                const void **out, IndexQueryMeta *ometa) const {
    return reformer_.transform(query, qmeta, count, out, ometa);
  }

  //! Normalize the score in query part
  void normalize(size_t qidx, IndexDocumentHeap *heap) const {
    return reformer_.normalize(qidx, heap);
  }

  //! Retrieve the value for each inverted list to multiply for normalizing
  float inverted_list_normalize_value(size_t inverted_list_id) const {
    if (norm_value_ != 0.0f) {
      return norm_value_;
    }

    // ailego_assert_with(integer_quantizer_params_, "nullptr");
    if (integer_quantizer_params_ != nullptr) {
      const void *data = nullptr;
      size_t size = sizeof(InvertedIntegerQuantizerParams);
      size_t off = inverted_list_id * size;
      if (integer_quantizer_params_->read(off, &data, size) != size) {
        LOG_ERROR("Failed to read data from segment, off=%zu", off);
        return 1.0f;
      }
      auto scale =
          static_cast<const InvertedIntegerQuantizerParams *>(data)->scale;
      return this->convert_to_normalize_value(scale);
    }

    return norm_value_;
  }

  //! Check whether the feature segment exist
  bool has_orignal_feature() const {
    return !!features_;
  }

  //! Retrieve reformer
  const IVFReformerWrapper &reformer(void) const {
    return reformer_;
  }

  /*! Index Reformer Wrapper
   *  To transform query in inverted index searching, and normalize the score
   */
  class IVFReformerWrapper {
   public:
    //! Constructor
    IVFReformerWrapper() {}

    //! Assignment
    IVFReformerWrapper &operator=(const IVFReformerWrapper &wrapper) {
      reformer_ = wrapper.reformer_;
      type_ = wrapper.type_;
      buffer_.clear();
      buffer_.shrink_to_fit();
      reciprocal_ = wrapper.reciprocal_;
      return *this;
    }

    //! Initialize
    int init(const IndexMeta &imeta);

    //! Update
    int update(const IndexMeta &meta);

    //! Transform a query
    int transform(const void *query, const IndexQueryMeta &qmeta,
                  const void **out, IndexQueryMeta *ometa);

    //! Transform queries
    int transform(const void *query, const IndexQueryMeta &qmeta,
                  uint32_t count, const void **out, IndexQueryMeta *ometa);

    //! Convert a record
    virtual int convert(const void *record, const IndexQueryMeta &rmeta,
                        const void **out, IndexQueryMeta *ometa);

    //! Convert records
    virtual int convert(const void *records, const IndexQueryMeta &rmeta,
                        uint32_t count, const void **out,
                        IndexQueryMeta *ometa);

    //! Transform queries
    int transform_gpu(const void *query, const IndexQueryMeta &qmeta,
                      uint32_t count, const void **out, IndexQueryMeta *ometa);

    //! Normalize the score in query part
    void normalize(size_t qidx, IndexDocumentHeap *heap) const;

    //! Normalize the score in query part
    void normalize(size_t qidx, const void *query, const IndexQueryMeta &qmeta,
                   IndexDocumentHeap *heap) const;

   private:
    //! Transform query from fp32 to int8
    void transform(size_t qidx, const float *in, size_t dim, int8_t *out);

    //! Transform query from fp32 to int4
    void transform(size_t qidx, const float *in, size_t dim, uint8_t *out);

   private:
    //! Constants
    enum Type {
      kReformerTpNone = 0,
      kReformerTpInnerProductInt8 = 1,
      kReformerTpInnerProductInt4 = 2,
      kReformerTpInt8 = 3,
      kReformerTpInt4 = 4,
      kReformerTpDefault = 7,
    };

    //! Members
    Type type_{kReformerTpNone};
    IndexReformer::Pointer reformer_{};
    std::string buffer_{};
    float reciprocal_{0.0};        // for int8
    std::vector<float> scales_{};  // for int8 IP
  };

 private:
  //! Load the segment by seg_id in expect_size segment size
  IndexStorage::Segment::Pointer load_segment(const std::string &seg_id,
                                              size_t expect_size) const;

  //! Load the header segment
  int load_header(const IndexStorage::Pointer &container);

  //! Convert the int8 quantizer scale to normalize value
  float convert_to_normalize_value(float scale) const {
    auto v = scale == 0.0 ? 1.0 : (1.0 / scale);
    return !norm_value_sqrt_ ? v : std::sqrt(v);
  }

 protected:
  //! Constants
  static constexpr size_t kBatchBlocks = 10u;

  //! Members
  IndexMeta meta_{};
  mutable IVFReformerWrapper reformer_{};
  IVFDistanceCalculator::Pointer calculator_{};
  InvertedIndexHeader header_{};
  IndexStorage::Pointer container_{};
  IndexStorage::Segment::Pointer inverted_{};
  IndexStorage::Segment::Pointer inverted_meta_{};
  IndexStorage::Segment::Pointer keys_{};
  IndexStorage::Segment::Pointer offsets_{};
  IndexStorage::Segment::Pointer mapping_{};
  IndexStorage::Segment::Pointer features_{};
  IndexStorage::Segment::Pointer integer_quantizer_params_{};
  mutable std::string vector_{};  // temporary buffer for colomn major order
  float norm_value_{0.0f};  // normalize the inverted vector to orignal score
  bool norm_value_sqrt_{false};  // does the norm value need to sqrt
};

}  // namespace core
}  // namespace zvec
