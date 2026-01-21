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
#include "ivf_index_format.h"
#include "ivf_params.h"
#include "ivf_utility.h"

namespace zvec {
namespace core {

/*! Quantized Clustering Dumper
 */
class IVFDumper {
 public:
  typedef std::shared_ptr<IVFDumper> Pointer;

  //! Vectors block
  class Block {
   public:
    //! Initialize block
    void init(const IndexMeta &meta, uint32_t max_vec_count) {
      element_size_ = meta.element_size();
      auto bsize = IVFUtility::AlignedSize(max_vec_count, element_size_);
      data_.resize(bsize);
      count_ = 0u;
      major_order_ = meta.major_order();
      align_size_ = IndexMeta::AlignSizeof(meta.data_type());
      units_ = element_size_ / align_size_;
      max_vec_count_ = max_vec_count;
      keys_.reserve(max_vec_count_);
    }

    //! Add a vector to the block in row major order
    //! If the block is full and the block order is column, make a
    //! transpose
    void emplace(uint64_t key, const void *vec, IndexMeta::MajorOrder order) {
      switch (align_size_) {
        case 2:
          do_emplace<uint16_t>(vec, order);
          break;
        case 4:
          do_emplace<uint32_t>(vec, order);
          break;
        case 8:
          do_emplace<uint64_t>(vec, order);
          break;
        default:
          ailego_check_with(false, "Unsupport Aligned Size");
      }
      keys_.emplace_back(key);
    }

    bool full(void) const {
      return count_ == max_vec_count_;
    }

    const void *data(void) const {
      return data_.data();
    }

    void clear(void) {
      count_ = 0u;
      keys_.clear();
    }

    bool empty(void) const {
      return count_ == 0u;
    }

    size_t size(void) const {
      return count_;
    }

    size_t capacity(void) const {
      return max_vec_count_;
    }

    size_t align_size(void) const {
      return align_size_;
    }

    size_t element_size(void) const {
      return element_size_;
    }

    //! Retrieve block data size
    size_t bytes(void) const {
      return element_size_ * count_;
    }

    //! Retrieve max block size in bytes
    size_t block_size(void) const {
      return data_.size();
    }

    IndexMeta::MajorOrder major_order(void) const {
      return major_order_;
    }

    const std::vector<uint64_t> &keys(void) const {
      return keys_;
    }

    bool match_order(IndexMeta::MajorOrder column_major) const {
      return major_order_ == column_major;
    }

   private:
    //! Transpose the block vectors
    void transpose() {
      std::vector<uint8_t> buf(data_.size());
      IVFUtility::Transpose(align_size_, data_.data(), count_, units_,
                            buf.data());
      data_.swap(buf);
    }

    template <typename T>
    void do_emplace(const void *vec, IndexMeta::MajorOrder order) {
      ailego_assert_with(count_ < max_vec_count_, "emplace a full block");

      T *dst = reinterpret_cast<T *>(data_.data() + element_size_ * count_);
      const T *src = reinterpret_cast<const T *>(vec);
      size_t step = order == IndexMeta::MO_ROW ? 1 : max_vec_count_;
      for (auto i = 0u; i < units_; ++i) {
        *dst = *src;
        dst++;
        src += step;
      }

      count_++;
      if (full() && major_order_ == IndexMeta::MO_COLUMN) {
        transpose();
      }
    }

   private:
    //! Members
    std::vector<uint8_t> data_{};
    std::vector<uint64_t> keys_{};
    uint32_t count_{0u};
    uint32_t units_{0u};
    uint32_t align_size_{0u};
    uint32_t element_size_{0u};
    uint32_t max_vec_count_{0u};
    IndexMeta::MajorOrder major_order_{};
  };

  //! Constructor
  IVFDumper(const IndexMeta &meta, const IndexDumper::Pointer &dumper,
            size_t inverted_list_count, size_t block_vector_count)
      : meta_(meta),
        dumper_(dumper),
        block_vector_count_(block_vector_count),
        inverted_lists_meta_(inverted_list_count) {
    block_.init(meta, block_vector_count_);
    header_.block_vector_count = block_vector_count_;
  }

  //! Constructor
  IVFDumper(const IndexMeta &meta, const IndexDumper::Pointer &dumper,
            size_t inverted_list_count)
      : IVFDumper(meta, dumper, inverted_list_count, kDefaultBlockCount) {}

  //! Destructor
  ~IVFDumper() {
    // Check the dumper status
    if (dumped_feature_count_ > 0 &&
        dumped_feature_count_ != header_.total_vector_count) {
      LOG_ERROR("Dumped features=%u mismatch from invertedVecs=%u",
                dumped_feature_count_, header_.total_vector_count);
      ailego_assert_with(false, "invalid status");
    }
  }

  //! Dump a vector in row major order
  int dump_inverted_vector(uint32_t inverted_list_id, uint64_t key,
                           const void *vec);

  int dump_inverted_block(uint32_t inverted_list_id, const uint64_t *keys,
                          const void *vecs, uint32_t vector_count,
                          bool column_major);

  //! Finish dump the inverted vectors
  int dump_inverted_vector_finished(void);

  //! Dump the centroids index
  int dump_centroid_index(const void *data, size_t size);

  //! Dump params for each inverted list quantizer
  int dump_quantizer_params(
      const std::vector<IndexConverter::Pointer> &quantizers);

  //! Dump the original vector, which doesnot been quantized
  int dump_original_vector(const void *data, size_t size);

  //! Retrieve total dumped size
  size_t dumped_size(void) const {
    return dumped_size_;
  }

  //! Retrieve total dumped vector count
  size_t dumped_count(void) const {
    return header_.total_vector_count;
  }

  //! Dump the segment from container
  int dump_container_segment(const IndexStorage::Pointer &container,
                             const std::string &segmemt_id);

 private:
  int check_dump_inverted_list(uint32_t inverted_list_id);

  //! Dump offsets segment
  int dump_offsets_segment(void) const;

  //! Dump a segment
  int dump_segment(const std::string &segment_id, const void *data,
                   size_t size) const;

  //! Dump segment padding
  int dump_padding(size_t data_size, size_t *padding_size) const;

  //! Dump a vector block
  int dump_block(void);

 private:
  //! Constants
  static constexpr size_t kDefaultBlockCount = 32u;

  //! Members
  Block block_{};           // vectors grouped in block
  const IndexMeta meta_{};  // IndexMeta of the inverted index
  const IndexDumper::Pointer dumper_{};
  size_t block_vector_count_{kDefaultBlockCount};
  std::vector<InvertedListMeta> inverted_lists_meta_{};
  std::vector<uint64_t> keys_{};
  InvertedIndexHeader header_{};
  uint32_t cur_list_id_{0};
  uint32_t dumped_feature_count_{0};
  size_t dumped_features_size_{0};
  mutable size_t dumped_size_{0};
};

}  // namespace core
}  // namespace zvec
