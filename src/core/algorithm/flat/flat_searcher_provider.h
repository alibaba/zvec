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

#include <memory>
#include "flat_distance_matrix.h"
#include "flat_searcher.h"
// #include "flat_streamer.h"
#include "flat_utility.h"

namespace zvec {
namespace core {

/*! Brute Force Searcher Provider
 */
template <size_t BATCH_SIZE>
class FlatSearcherProvider : public IndexProvider {
 public:
  //! Constructor
  FlatSearcherProvider(const FlatSearcher<BATCH_SIZE> *owner) {
    feature_size_ = owner->meta().element_size();
    features_segment_ = owner->clone_features_segment();
    total_vector_count_ =
        features_segment_->data_size() / owner->meta().element_size();
    owner_ = owner;
    block_buffer_.resize(BATCH_SIZE * feature_size_);
  }

  //! Create a new iterator
  IndexProvider::Iterator::Pointer create_iterator() override {
    return IndexProvider::Iterator::Pointer(
        new (std::nothrow) FlatSearcherProvider::Iterator(owner_));
  }

  //! Retrieve count of vectors
  size_t count() const override {
    return total_vector_count_;
  }

  //! Retrieve dimension of vector
  size_t dimension() const override {
    return owner_->meta().dimension();
  }

  //! Retrieve type of vector
  IndexMeta::DataType data_type() const override {
    return owner_->meta().data_type();
  }

  //! Retrieve vector size in bytes
  size_t element_size() const override {
    return owner_->meta().element_size();
  }

  //! Retrieve a vector using a primary key
  const void *get_vector(uint64_t key) const override {
    return this->get_vector_by_index(owner_->get_id(key));
  }

  //! Retrieve the owner class
  const std::string &owner_class() const override {
    return owner_->name();
  }

 protected:
  /*! Brute Force Provider Iterator
   */
  class Iterator : public IndexProvider::Iterator {
   public:
    //! Constructor
    Iterator(const FlatSearcher<BATCH_SIZE> *owner) {
      block_buffer_.resize(BATCH_SIZE * owner->meta().element_size());
      feature_size_ = owner->meta().element_size();
      features_segment_ = owner->clone_features_segment();
      total_vector_count_ =
          features_segment_->data_size() / owner->meta().element_size();
      owner_ = owner;
      cursor_index_ = 0;
      offset_ = 0;
      this->next_block();
    }

    //! Retrieve pointer of data
    //! NOTICE: the vec feature will be changed after iterating to next, so
    //! the caller need to keep a copy of it before iterator to next vector
    const void *data() const override {
      return data_;
    }

    //! Test if the iterator is valid
    bool is_valid() const override {
      return (!invalid_ && cursor_index_ < total_vector_count_);
    }

    //! Retrieve primary key
    uint64_t key() const override {
      return owner_->key(cursor_index_);
    }

    //! Next iterator
    void next() override {
      ++cursor_index_;

      if ((cursor_index_ % BATCH_SIZE) != 0) {
        data_ += feature_size_;
      } else {
        this->next_block();
      }
    }

   protected:
    //! Read a block of data
    void next_block() {
      const void *read_data = nullptr;
      size_t read_size = 0;

      if (cursor_index_ >= total_vector_count_) {
        invalid_ = true;
        return;
      }

      if (cursor_index_ + BATCH_SIZE < total_vector_count_) {
        read_size = BATCH_SIZE * feature_size_;
      } else {
        read_size = (total_vector_count_ - cursor_index_) * feature_size_;
      }
      if (features_segment_->read(offset_, &read_data, read_size) !=
          read_size) {
        LOG_ERROR("Failed to read data (%zu bytes) from features segment",
                  read_size);
        invalid_ = true;
        return;
      }
      offset_ += read_size;

      // The order of data may be a column format, convert it to the row format.
      if (owner_->column_major_order() &&
          read_size == BATCH_SIZE * feature_size_) {
        uint32_t align_size =
            IndexMeta::AlignSizeof(owner_->meta().data_type());
        ReverseTranspose<BATCH_SIZE>(align_size, read_data,
                                     feature_size_ / align_size,
                                     &block_buffer_[0]);
        data_ = block_buffer_.data();
      } else {
        data_ = reinterpret_cast<const uint8_t *>(read_data);
      }
    }

   private:
    const FlatSearcher<BATCH_SIZE> *owner_{nullptr};
    IndexStorage::Segment::Pointer features_segment_{};
    uint32_t total_vector_count_{0};
    uint32_t feature_size_{0};
    std::vector<uint8_t> block_buffer_{};
    const uint8_t *data_{nullptr};
    uint64_t offset_{0};
    uint32_t cursor_index_{0};
    bool invalid_{false};
  };

  //! Retrieve a vector via local index
  const void *get_vector_by_index(uint32_t index) const {
    const void *read_data = nullptr;
    if (index == kInvalidNodeId) {
      LOG_ERROR("Failed to get vector by Invalid Id.");
      return nullptr;
    }

    if (owner_->column_major_order() &&
        index < (total_vector_count_ - (total_vector_count_ % BATCH_SIZE))) {
      uint32_t block_size = feature_size_ * BATCH_SIZE;
      uint64_t offset = (index - (index % BATCH_SIZE)) * feature_size_;

      if (features_segment_->read(offset, &read_data, block_size) !=
          block_size) {
        LOG_ERROR("Failed to read data (%u bytes) from features segment",
                  block_size);
        return nullptr;
      }

      uint32_t align_size = IndexMeta::AlignSizeof(owner_->meta().data_type());
      ReverseTranspose<BATCH_SIZE>(
          align_size, read_data, feature_size_ / align_size, &block_buffer_[0]);
      read_data = block_buffer_.data() + ((index % BATCH_SIZE) * feature_size_);

    } else {
      if (features_segment_->read(index * feature_size_, &read_data,
                                  feature_size_) != feature_size_) {
        LOG_ERROR("Failed to read data (%u bytes) from features segment",
                  feature_size_);
        return nullptr;
      }
    }
    return read_data;
  }

 private:
  //! Members
  const FlatSearcher<BATCH_SIZE> *owner_{nullptr};
  IndexStorage::Segment::Pointer features_segment_{};
  uint32_t feature_size_{0};
  uint32_t total_vector_count_{0};
  mutable std::vector<uint8_t> block_buffer_{};
};

}  // namespace core
}  // namespace zvec
