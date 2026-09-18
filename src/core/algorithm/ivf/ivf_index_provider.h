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

#include <algorithm>
#include <cstring>
#include <new>
#include <numeric>
#include <vector>
#include <zvec/core/framework/index_searcher.h>
#include "ivf_entity.h"

namespace zvec {
namespace core {

/*! IVF IndexProvider
 */
class IVFIndexProvider : public IndexProvider {
 public:
  IVFIndexProvider(const IndexMeta &meta, const IVFEntity::Pointer &entity,
                   const std::string &owner)
      : meta_(meta), entity_(entity), owner_class_(owner) {}

  IVFIndexProvider(const IVFIndexProvider &) = delete;
  IVFIndexProvider &operator=(const IVFIndexProvider &) = delete;

 public:
  //! Create a new iterator
  Iterator::Pointer create_iterator() override {
    return Iterator::Pointer(new (std::nothrow) SortedIterator(entity_));
  }

  //! Retrieve count of vectors
  size_t count() const override {
    return entity_->vector_count();
  }

  //! Retrieve dimension of vector
  size_t dimension() const override {
    return meta_.dimension();
  }

  //! Retrieve type of vector
  IndexMeta::DataType data_type() const override {
    return meta_.data_type();
  }

  //! Retrieve vector size in bytes
  size_t element_size() const override {
    return meta_.element_size();
  }

  //! Retrieve a vector using a primary key
  const void *get_vector(uint64_t key) const override {
    return DecodeVector(entity_, entity_->get_vector_by_key(key), &vector_);
  }

  //! Retrieve the owner class
  const std::string &owner_class() const override {
    return owner_class_;
  }

 private:
  // Providers feed clustering/reducers, which need vectors in the original
  // input space even when postings use a different encoded layout.
  static const void *DecodeVector(const IVFEntity::Pointer &entity,
                                  const void *data, std::string *buffer) {
    if (!data || !entity->quantizer() || entity->has_orignal_feature()) {
      return data;
    }
    const auto &meta = entity->meta();
    IndexQueryMeta qmeta(meta.data_type(), meta.dimension());
    qmeta.set_meta(meta.data_type(), meta.dimension(),
                   static_cast<uint32_t>(entity->quantizer()->type()),
                   meta.extra_meta_size());
    if (entity->quantizer()->dequantize(data, qmeta, buffer) != 0) {
      return nullptr;
    }
    const size_t dim = entity->quantizer()->dim();
    if (entity->quantizer()->input_data_type() == turbo::DataType::kFp16 &&
        buffer->size() == dim * sizeof(float)) {
      // PQ reconstruction is FP32 even when its training input was FP16.
      std::string half(dim * sizeof(ailego::Float16), '\0');
      for (size_t i = 0; i < dim; ++i) {
        float value;
        std::memcpy(&value, buffer->data() + i * sizeof(float), sizeof(value));
        ailego::Float16 encoded(value);
        std::memcpy(&half[i * sizeof(encoded)], &encoded, sizeof(encoded));
      }
      *buffer = std::move(half);
    }
    return buffer->data();
  }

  class SortedIterator : public IndexProvider::Iterator {
   public:
    SortedIterator(const IVFEntity::Pointer &entity) : entity_(entity) {
      count_ = entity_->vector_count();
      use_mapping_ = entity_->has_key_order_mapping();
      if (!use_mapping_) {
        // Fallback: compute sorting if mapping segment is unavailable
        fallback_.resize(count_);
        std::iota(fallback_.begin(), fallback_.end(), size_t(0));
        std::sort(fallback_.begin(), fallback_.end(), [&](size_t a, size_t b) {
          return entity_->get_key(a) < entity_->get_key(b);
        });
      }
    }

    //! Retrieve pointer of data
    //! NOTICE: the vec feature will be changed after iterating to next, so
    //! the caller need to keep a copy of it before iterator to next vector
    const void *data() const override {
      size_t local_id = current_local_id();
      if (local_id >= count_) {
        return nullptr;
      }
      const void *result =
          DecodeVector(entity_, entity_->get_vector(local_id), &vector_);
      if (!result) {
        status_ = IndexError_ReadData;
      }
      return result;
    }

    //! Test if the iterator is valid
    bool is_valid() const override {
      return status_ == 0 && pos_ < count_ &&
             (!use_mapping_ || ensure_mapping_chunk());
    }

    int status() const override {
      return status_;
    }

    //! Retrieve primary key
    uint64_t key() const override {
      size_t local_id = current_local_id();
      if (local_id >= count_) {
        return kInvalidKey;
      }
      const uint64_t result = entity_->get_key(local_id);
      if (result == kInvalidKey) {
        status_ = IndexError_ReadData;
      }
      return result;
    }

    //! Next iterator
    void next() override {
      if (status_ == 0 && pos_ < count_) {
        ++pos_;
      }
    }

   private:
    bool ensure_mapping_chunk() const {
      if (status_ != 0 || pos_ >= count_) {
        return false;
      }
      if (pos_ >= mapping_chunk_begin_ &&
          pos_ - mapping_chunk_begin_ < mapping_chunk_.size()) {
        return true;
      }
      mapping_chunk_begin_ = pos_;
      const size_t chunk_count =
          std::min(kMappingChunkEntries, count_ - mapping_chunk_begin_);
      try {
        mapping_chunk_.resize(chunk_count);
      } catch (const std::bad_alloc &) {
        status_ = IndexError_NoMemory;
        return false;
      }
      if (entity_->get_key_order_mapping(mapping_chunk_begin_,
                                         mapping_chunk_.data(),
                                         chunk_count) != chunk_count) {
        mapping_chunk_.clear();
        status_ = IndexError_ReadData;
        return false;
      }
      if (std::any_of(mapping_chunk_.begin(), mapping_chunk_.end(),
                      [this](uint32_t id) { return id >= count_; })) {
        mapping_chunk_.clear();
        status_ = IndexError_InvalidFormat;
        return false;
      }
      return true;
    }

    size_t current_local_id() const {
      if (status_ != 0 || pos_ >= count_) {
        return count_;
      }
      if (!use_mapping_) {
        return fallback_[pos_];
      }
      if (!ensure_mapping_chunk()) {
        return count_;
      }
      return static_cast<size_t>(mapping_chunk_[pos_ - mapping_chunk_begin_]);
    }

    //! Members
    static constexpr size_t kMappingChunkEntries = 4096;
    IVFEntity::Pointer entity_;
    mutable std::string vector_;
    bool use_mapping_{false};
    mutable int status_{0};
    mutable std::vector<uint32_t> mapping_chunk_;
    mutable size_t mapping_chunk_begin_{0};
    std::vector<size_t> fallback_;  // used only if mapping_ unavailable
    size_t count_{0};
    size_t pos_{0};
  };

  //! Original sequential iterator (kept for potential internal use)
  class Iterator : public IndexProvider::Iterator {
   public:
    Iterator(const IVFEntity::Pointer &entity) : entity_(entity) {}

    //! Retrieve pointer of data
    const void *data() const override {
      return DecodeVector(entity_, entity_->get_vector(index_), &vector_);
    }

    //! Test if the iterator is valid
    bool is_valid() const override {
      return index_ < entity_->vector_count();
    }

    //! Retrieve primary key
    uint64_t key() const override {
      return entity_->get_key(index_);
    }

    //! Next iterator
    void next() override {
      ++index_;
    }

   private:
    //! Members
    IVFEntity::Pointer entity_;
    mutable std::string vector_;
    size_t index_{0};
  };

 private:
  //! Members
  IndexMeta meta_;
  IVFEntity::Pointer entity_;
  std::string owner_class_;
  mutable std::string vector_;
};

}  // namespace core
}  // namespace zvec
