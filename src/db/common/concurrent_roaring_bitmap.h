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

#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <roaring.hh>
#include <roaring64map.hh>
#include <roaring/roaring.h>
#include <zvec/ailego/container/vector.h>
#include <zvec/ailego/internal/platform.h>
#include <zvec/ailego/io/file.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/db/status.h>


namespace zvec {


/*
 * A thread-safe 32-bit Roaring bitmap implementation.
 */
class ConcurrentRoaringBitmap32 {
 public:
  using Ptr = std::shared_ptr<ConcurrentRoaringBitmap32>;

  explicit ConcurrentRoaringBitmap32()
      : identifier_("Roaring bitmap[32-bit]"),
        bitmap_(roaring_bitmap_create()) {}

  ~ConcurrentRoaringBitmap32() {
    roaring_bitmap_free(bitmap_);
  }

  ConcurrentRoaringBitmap32(const ConcurrentRoaringBitmap32 &) = delete;
  ConcurrentRoaringBitmap32 &operator=(const ConcurrentRoaringBitmap32 &) =
      delete;
  ConcurrentRoaringBitmap32 &operator=(ConcurrentRoaringBitmap32 &&) = delete;


  /*****  Serialization and Deserialization - Start  *****/
 public:
  Status serialize(std::string *out);

  Status deserialize(const std::string &in);
  /*****  Serialization and Deserialization - End  *****/


 public:
  bool contains(uint32_t pos) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return roaring_bitmap_contains(bitmap_, pos);
  }


  size_t cardinality() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return roaring_bitmap_get_cardinality(bitmap_);
  }


  size_t range_cardinality(uint32_t min_doc_id, uint32_t max_doc_id) const {
    if (ailego_unlikely(min_doc_id > max_doc_id)) {
      LOG_WARN("%s: input range min_doc_id[%u] > max_doc_id[%u]",
               identifier_.c_str(), min_doc_id, max_doc_id);
      return 0;
    }
    std::shared_lock<std::shared_mutex> lock(mutex_);
    uint64_t max_rank{0}, min_rank{0};
    max_rank = roaring_bitmap_rank(bitmap_, max_doc_id);
    min_rank = roaring_bitmap_rank(bitmap_, min_doc_id - 1);
    return max_rank - min_rank;
  }


  void add(uint32_t pos) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    roaring_bitmap_add(bitmap_, pos);
  }


  void clear() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    roaring_bitmap_clear(bitmap_);
  }


  //! Remove all values in the closed interval [min, max]
  void remove_range_closed(uint32_t min, uint32_t max) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    roaring_bitmap_remove_range_closed(bitmap_, min, max);
  }


  size_t storage_size_in_bytes() const {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    return roaring_bitmap_portable_size_in_bytes(bitmap_);
  }


  roaring_bitmap_t *bitmap() const {
    return bitmap_;
  }


  roaring_bitmap_t *copy() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return roaring_bitmap_copy(bitmap_);
  }


 private:
  std::string identifier_;
  roaring_bitmap_t *bitmap_{nullptr};
  mutable std::shared_mutex mutex_;
};


/*
 * A thread-safe Roaring bitmap implementation supporting both 32-bit and 64-bit
 * bitmaps with transparent conversion between them.
 */
class ConcurrentRoaringBitmap64 {
 public:
  using Ptr = std::shared_ptr<ConcurrentRoaringBitmap64>;


  explicit ConcurrentRoaringBitmap64()
      : is_32bit_(true),
        identifier_("Roaring bitmap[32-bit]"),
        bitmap32_(std::make_unique<roaring::Roaring>()) {}

  explicit ConcurrentRoaringBitmap64(const std::string &name)
      : name_(name),
        is_32bit_(true),
        identifier_("Roaring bitmap[" + name_ + ", 32-bit]"),
        bitmap32_(std::make_unique<roaring::Roaring>()) {}

  ~ConcurrentRoaringBitmap64() = default;

  ConcurrentRoaringBitmap64 &operator=(const ConcurrentRoaringBitmap64 &other) {
    if (this != &other) {
      std::unique_lock<std::shared_mutex> lock(mutex_, std::defer_lock);
      std::shared_lock<std::shared_mutex> other_lock(other.mutex_,
                                                     std::defer_lock);
      std::lock(lock, other_lock);

      name_ = other.name_;
      is_32bit_ = other.is_32bit_;
      identifier_ = other.identifier_;

      if (other.is_32bit_) {
        bitmap32_ = std::make_unique<roaring::Roaring>(*other.bitmap32_);
        bitmap64_.reset();
      } else {
        bitmap64_ = std::make_unique<roaring::Roaring64Map>(*other.bitmap64_);
        bitmap32_.reset();
      }
    }
    return *this;
  }

  /*****  Serialization and Deserialization - Start  *****/
 public:
  Status serialize(const std::string &file_path, bool overwrite);

  Status deserialize(const std::string &file_path);

 private:
  static const uint64_t roaring_magic_number{0x362DDA444AC1B99A};

  struct BitmapMetaHeader {
    uint64_t magic;
    uint32_t is_32bit;
    uint32_t checksum;
    uint64_t timestamp;
    uint32_t reserved_[10];
  };
  /*****  Serialization and Deserialization - End  *****/


 public:
  bool contains(size_t pos) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    if (is_32bit_) {
      return bitmap32_->contains(static_cast<uint32_t>(pos));
    } else {
      return bitmap64_->contains(static_cast<uint64_t>(pos));
    }
  }


  size_t cardinality() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    if (is_32bit_) {
      return bitmap32_->cardinality();
    } else {
      return bitmap64_->cardinality();
    }
  }


  size_t range_cardinality(uint64_t min_doc_id, uint64_t max_doc_id) const {
    if (ailego_unlikely(min_doc_id > max_doc_id)) {
      LOG_WARN("%s: input range min_doc_id[%zu] > max_doc_id[%zu]",
               identifier_.c_str(), static_cast<size_t>(min_doc_id),
               static_cast<size_t>(max_doc_id));
      return 0;
    }
    std::shared_lock<std::shared_mutex> lock(mutex_);
    uint64_t max_rank{0}, min_rank{0};
    if (is_32bit_) {
      max_rank = bitmap32_->rank(max_doc_id);
      min_rank = min_doc_id <= 0 ? 0 : bitmap32_->rank(min_doc_id - 1);
    } else {
      max_rank = bitmap64_->rank(max_doc_id);
      min_rank = min_doc_id <= 0 ? 0 : bitmap64_->rank(min_doc_id - 1);
    }
    return max_rank - min_rank;
  }


  void add(size_t pos) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (ailego_unlikely(pos > std::numeric_limits<uint32_t>::max() &&
                        is_32bit_)) {
      upgrade_from_32_to_64();
    }
    if (is_32bit_) {
      return bitmap32_->add(static_cast<uint32_t>(pos));
    } else {
      return bitmap64_->add(static_cast<uint64_t>(pos));
    }
  }


  void clear() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    bitmap32_.reset();
    bitmap64_.reset();
    if (is_32bit_) {
      bitmap32_ = std::make_unique<roaring::Roaring>();
    } else {
      bitmap64_ = std::make_unique<roaring::Roaring64Map>();
    }
  }


  //! Remove all values in the closed interval [min, max]
  void remove_range_closed(uint64_t min, uint64_t max) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (!is_32bit_) {
      return bitmap64_->removeRangeClosed(min, max);
    }
    if (min > std::numeric_limits<uint32_t>::max()) {
      return;  // No valid values in the 32-bit range that can be removed
    }
    if (max > std::numeric_limits<uint32_t>::max()) {
      max = std::numeric_limits<uint32_t>::max();
    }
    bitmap32_->removeRangeClosed(min, max);
  }


  size_t storage_size_in_bytes() const {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (is_32bit_) {
      return bitmap32_->getSizeInBytes() + sizeof(BitmapMetaHeader);
    } else {
      return bitmap64_->getSizeInBytes() + sizeof(BitmapMetaHeader);
    }
  }


 private:
  using FILE = ailego::File;


  template <typename... Args>
  std::string debug_str(Args &&...args) {
    std::ostringstream oss;
    oss << identifier_ << ": ";
    (oss << ... << args);
    return oss.str();
  }


  void upgrade_from_32_to_64() {
    if (ailego_unlikely(!is_32bit_)) {
      LOG_WARN("%s: bitmap is already 64-bit", identifier_.c_str());
      return;
    }
    bitmap64_ = std::make_unique<roaring::Roaring64Map>(
        roaring::Roaring64Map{std::move(*bitmap32_)});
    is_32bit_ = false;
    bitmap32_.reset();
    identifier_ = "Roaring bitmap[" + name_ + ", 64-bit]";
    LOG_DEBUG("%s: upgraded to 64-bit", identifier_.c_str());
  }


  std::string name_;
  bool is_32bit_;
  std::string identifier_;
  std::unique_ptr<roaring::Roaring> bitmap32_{nullptr};
  std::unique_ptr<roaring::Roaring64Map> bitmap64_{nullptr};
  mutable std::shared_mutex mutex_;
};


}  // namespace zvec
