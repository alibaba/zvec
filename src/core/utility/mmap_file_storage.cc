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
#include <mutex>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_mapping.h>
#include <zvec/core/framework/index_version.h>
#include "utility_params.h"

namespace zvec {
namespace core {

/*! MMap File Storage
 */
class MMapFileStorage : public IndexStorage {
 public:
  /*! Index Storage Segment
   */
  class Segment : public IndexStorage::Segment,
                  public std::enable_shared_from_this<Segment> {
   public:
    //! Index Storage Pointer
    typedef std::shared_ptr<Segment> Pointer;

    //! Constructor
    Segment(MMapFileStorage *owner, IndexMapping::Segment *segment)
        : segment_(segment),
          owner_(owner),
          capacity_(static_cast<size_t>(segment->meta()->data_size +
                                        segment->meta()->padding_size)) {}

    //! Destructor
    virtual ~Segment(void) {}

    //! Retrieve size of data
    size_t data_size(void) const override {
      return static_cast<size_t>(segment_->meta()->data_size);
    }

    //! Retrieve crc of data
    uint32_t data_crc(void) const override {
      return segment_->meta()->data_crc;
    }

    //! Retrieve size of padding
    size_t padding_size(void) const override {
      return static_cast<size_t>(segment_->meta()->padding_size);
    }

    //! Retrieve capacity of segment
    size_t capacity(void) const override {
      return capacity_;
    }

    //! Fetch data from segment (with own buffer)
    size_t fetch(size_t offset, void *buf, size_t len) const override {
      if (ailego_unlikely(offset + len > segment_->meta()->data_size)) {
        auto meta = segment_->meta();
        if (offset > meta->data_size) {
          offset = meta->data_size;
        }
        len = meta->data_size - offset;
      }
      memmove(buf, (const uint8_t *)segment_->data() + offset, len);
      return len;
    }

    //! Read data from segment
    size_t read(size_t offset, const void **data, size_t len) override {
      if (ailego_unlikely(offset + len > segment_->meta()->data_size)) {
        auto meta = segment_->meta();
        if (offset > meta->data_size) {
          offset = meta->data_size;
        }
        len = meta->data_size - offset;
      }
      *data = (uint8_t *)segment_->data() + offset;
      return len;
    }

    size_t read(size_t offset, MemoryBlock &data, size_t len) override {
      if (ailego_unlikely(offset + len > segment_->meta()->data_size)) {
        auto meta = segment_->meta();
        if (offset > meta->data_size) {
          offset = meta->data_size;
        }
        len = meta->data_size - offset;
      }
      data.reset((uint8_t *)segment_->data() + offset);
      return len;
    }

    //! Write data into the storage with offset
    size_t write(size_t offset, const void *data, size_t len) override {
      size_t data_tail = offset + len;
      ailego_zero_if_false(data_tail <= capacity_);
      auto meta = segment_->meta();
      if (data_tail > meta->data_size) {
        meta->data_size = data_tail;
        meta->padding_size = capacity_ - data_tail;
        owner_->set_as_dirty();
      }
      memmove((uint8_t *)segment_->data() + offset, data, len);
      segment_->set_dirty();
      return len;
    }

    //! Resize size of data
    size_t resize(size_t size) override {
      auto meta = segment_->meta();
      if (meta->data_size != size) {
        if (size > capacity_) {
          size = capacity_;
        }
        meta->data_size = size;
        meta->padding_size = capacity_ - size;
        owner_->set_as_dirty();
      }
      return size;
    }

    //! Update crc of data
    void update_data_crc(uint32_t crc) override {
      segment_->meta()->data_crc = crc;
    }

    //! Clone the segment
    IndexStorage::Segment::Pointer clone(void) override {
      return shared_from_this();
    }

   private:
    IndexMapping::Segment *segment_{};
    MMapFileStorage *owner_{nullptr};
    size_t capacity_{};
  };

  //! Destructor
  virtual ~MMapFileStorage(void) {
    this->cleanup();
  }

  //! Initialize storage
  int init(const ailego::Params &params) override {
    uint32_t val = params.get_as_uint32(MMAPFILE_STORAGE_SEGMENT_META_CAPACITY);
    if (val != 0) {
      segment_meta_capacity_ = val;
    }
    params.get(MMAPFILE_STORAGE_COPY_ON_WRITE, &copy_on_write_);
    params.get(MMAPFILE_STORAGE_FORCE_FLUSH, &force_flush_);
    params.get(MMAPFILE_STORAGE_MEMORY_LOCKED, &memory_locked_);
    params.get(MMAPFILE_STORAGE_MEMORY_WARMUP, &memory_warmup_);
    return 0;
  }

  //! Cleanup storage
  int cleanup(void) override {
    this->close_index();
    return 0;
  }

  //! Open storage
  int open(const std::string &path, bool create) override {
    if (!ailego::File::IsExist(path) && create) {
      size_t last_slash = path.rfind('/');
      if (last_slash != std::string::npos) {
        ailego::File::MakePath(path.substr(0, last_slash));
      }

      int error_code = this->init_index(path);
      if (error_code != 0) {
        return error_code;
      }
    }
    return mapping_.open(path, copy_on_write_, force_flush_);
  }

  //! Flush storage
  int flush(void) override {
    return this->flush_index();
  }

  //! Close storage
  int close(void) override {
    this->close_index();
    return 0;
  }

  //! Append a segment into storage
  int append(const std::string &id, size_t size) override {
    return this->append_segment(id, size);
  }

  //! Refresh meta information (checksum, update time, etc.)
  void refresh(uint64_t chkp) override {
    this->refresh_index(chkp);
  }

  //! Retrieve check point of storage
  uint64_t check_point(void) const override {
    return mapping_.footer().check_point;
  }

  //! Retrieve a segment by id
  IndexStorage::Segment::Pointer get(const std::string &id, int) override {
    IndexMapping::Segment *segment = this->get_segment(id);
    if (!segment) {
      return MMapFileStorage::Segment::Pointer();
    }
    return std::make_shared<MMapFileStorage::Segment>(this, segment);
  }

  //! Test if it a segment exists
  bool has(const std::string &id) const override {
    return this->has_segment(id);
  }

  //! Retrieve magic number of index
  uint32_t magic(void) const override {
    return mapping_.magic();
  }

 protected:
  //! Initialize index version segment
  int init_version_segment(void) {
    size_t data_size = std::strlen(IndexVersion::Details());
    int error_code =
        this->append_segment(INDEX_VERSION_SEGMENT_NAME, data_size);
    if (error_code != 0) {
      return error_code;
    }

    IndexMapping::Segment *segment = get_segment(INDEX_VERSION_SEGMENT_NAME);
    if (!segment) {
      return IndexError_MMapFile;
    }
    auto meta = segment->meta();
    size_t capacity = static_cast<size_t>(meta->padding_size + meta->data_size);
    memcpy(segment->data(), IndexVersion::Details(), data_size);
    segment->set_dirty();
    meta->data_crc = ailego::Crc32c::Hash(segment->data(), data_size, 0);
    meta->data_size = data_size;
    meta->padding_size = capacity - data_size;
    return 0;
  }

  //! Initialize index file
  int init_index(const std::string &path) {
    int error_code = mapping_.create(path, segment_meta_capacity_);
    if (error_code != 0) {
      return error_code;
    }

    // Add index version
    error_code = this->init_version_segment();
    if (error_code != 0) {
      return error_code;
    }

    // Refresh mapping
    this->refresh_index(0);

    // Close mapping
    mapping_.close();
    return 0;
  }

  bool isHugePage(void) const override {
    return mapping_.huge_page();
  }

  //! Set the index file as dirty
  void set_as_dirty(void) {
    index_dirty_ = true;
  }

  //! Refresh meta information (checksum, update time, etc.)
  void refresh_index(uint64_t chkp) {
    mapping_.refresh(chkp);
    index_dirty_ = false;
  }

  //! Flush index storage
  int flush_index(void) {
    if (index_dirty_) {
      this->refresh_index(0);
    }
    std::lock_guard<std::mutex> latch(mapping_mutex_);
    return mapping_.flush();
  }

  //! Close index storage
  void close_index(void) {
    if (index_dirty_) {
      this->refresh_index(0);
    }
    std::lock_guard<std::mutex> latch(mapping_mutex_);
    mapping_.close();
  }

  //! Append a segment into storage
  int append_segment(const std::string &id, size_t size) {
    std::lock_guard<std::mutex> latch(mapping_mutex_);
    return mapping_.append(id, size);
  }

  //! Test if a segment exists
  bool has_segment(const std::string &id) const {
    std::lock_guard<std::mutex> latch(mapping_mutex_);
    return mapping_.has(id);
  }

  //! Get a segment from storage
  IndexMapping::Segment *get_segment(const std::string &id) {
    std::lock_guard<std::mutex> latch(mapping_mutex_);
    return mapping_.map(id, memory_warmup_, memory_locked_);
  }

 private:
  uint32_t segment_meta_capacity_{1024 * 1024};
  bool copy_on_write_{false};
  bool force_flush_{false};
  bool memory_locked_{false};
  bool memory_warmup_{false};
  bool index_dirty_{false};
  mutable IndexMapping mapping_{};
  mutable std::mutex mapping_mutex_{};
};

INDEX_FACTORY_REGISTER_STORAGE(MMapFileStorage);

}  // namespace core
}  // namespace zvec
