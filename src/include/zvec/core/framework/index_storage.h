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

#include <zvec/ailego/buffer/buffer_manager.h>
#include <zvec/ailego/container/params.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_module.h>

namespace zvec {
namespace core {

/*! Index Storage
 */
class IndexStorage : public IndexModule {
 public:
  //! Index Storage Pointer
  typedef std::shared_ptr<IndexStorage> Pointer;

  struct MemoryBlock {
    enum MemoryBlockType {
      MBT_UNKNOWN = 0,
      MBT_MMAP = 1,
      MBT_BUFFERPOOL = 2,
    };

    MemoryBlock() {}
    MemoryBlock(ailego::BufferHandle::Pointer &&buffer_handle)
        : type_(MemoryBlockType::MBT_BUFFERPOOL),
          buffer_handle_(std::move(buffer_handle)) {
      data_ = buffer_handle_->pin_vector_data();
    }
    MemoryBlock(void *data) : type_(MemoryBlockType::MBT_MMAP), data_(data) {}

    MemoryBlock(MemoryBlock &rhs) {
      switch (rhs.type_) {
        case MemoryBlockType::MBT_MMAP:
          this->reset(rhs.data_);
          break;
        case MemoryBlockType::MBT_BUFFERPOOL:
          this->reset(rhs.buffer_handle_);
          break;
        default:
          break;
      }
    }

    MemoryBlock(MemoryBlock &&rhs) {
      switch (rhs.type_) {
        case MemoryBlockType::MBT_MMAP:
          this->reset(std::move(rhs.data_));
          break;
        case MemoryBlockType::MBT_BUFFERPOOL:
          this->reset(std::move(rhs.buffer_handle_));
          break;
        default:
          break;
      }
    }

    MemoryBlock &operator=(MemoryBlock &rhs) {
      if (this != &rhs) {
        switch (rhs.type_) {
          case MemoryBlockType::MBT_MMAP:
            this->reset(rhs.data_);
            break;
          case MemoryBlockType::MBT_BUFFERPOOL:
            this->reset(rhs.buffer_handle_);
            break;
          default:
            break;
        }
      }
      return *this;
    }

    MemoryBlock &operator=(MemoryBlock &&rhs) {
      if (this != &rhs) {
        switch (rhs.type_) {
          case MemoryBlockType::MBT_MMAP:
            this->reset(std::move(rhs.data_));
            break;
          case MemoryBlockType::MBT_BUFFERPOOL:
            this->reset(std::move(rhs.buffer_handle_));
            break;
          default:
            break;
        }
      }
      return *this;
    }

    ~MemoryBlock() {
      switch (type_) {
        case MemoryBlockType::MBT_MMAP:
          break;
        case MemoryBlockType::MBT_BUFFERPOOL:
          if (buffer_handle_) {
            buffer_handle_->unpin_vector_data();
            // buffer_handle_.reset();
          }
          break;
        default:
          break;
      }
      data_ = nullptr;
    }

    const void *data() const {
      return data_;
    }

    void reset(ailego::BufferHandle::Pointer &buffer_handle) {
      if (type_ == MemoryBlockType::MBT_BUFFERPOOL) {
        buffer_handle_->unpin_vector_data();
        buffer_handle_.reset();
      }
      type_ = MemoryBlockType::MBT_BUFFERPOOL;
      if (buffer_handle) {
        buffer_handle_.reset(buffer_handle.release());
      }
      data_ = buffer_handle_->pin_vector_data();
    }

    void reset(ailego::BufferHandle::Pointer &&buffer_handle) {
      if (type_ == MemoryBlockType::MBT_BUFFERPOOL) {
        buffer_handle_->unpin_vector_data();
        buffer_handle_.reset();
      }
      type_ = MemoryBlockType::MBT_BUFFERPOOL;
      if (buffer_handle) {
        buffer_handle_ = std::move(buffer_handle);
      }
      data_ = buffer_handle_->pin_vector_data();
    }

    void reset(void *data) {
      if (type_ == MemoryBlockType::MBT_BUFFERPOOL) {
        buffer_handle_->unpin_vector_data();
        buffer_handle_.reset();
      }
      type_ = MemoryBlockType::MBT_MMAP;
      data_ = data;
    }

    MemoryBlockType type_{MBT_UNKNOWN};
    void *data_{nullptr};
    mutable ailego::BufferHandle::Pointer buffer_handle_{nullptr};
  };

  struct SegmentData {
    //! Constructor
    SegmentData(void) : offset(0u), length(0u), data(nullptr) {}

    //! Constructor
    SegmentData(size_t off, size_t len)
        : offset(off), length(len), data(nullptr) {}

    //! Members
    size_t offset;
    size_t length;
    const void *data;
  };

  /*! Index Storage Segment
   */
  struct Segment {
    //! Index Storage Pointer
    typedef std::shared_ptr<Segment> Pointer;

    //! Destructor
    virtual ~Segment(void) {}

    //! Retrieve size of data
    virtual size_t data_size(void) const = 0;

    //! Retrieve crc of data
    virtual uint32_t data_crc(void) const = 0;

    //! Retrieve size of padding
    virtual size_t padding_size(void) const = 0;

    //! Retrieve capacity of segment
    virtual size_t capacity(void) const = 0;

    //! Fetch data from segment (with own buffer)
    virtual size_t fetch(size_t offset, void *buf, size_t len) const = 0;

    //! Read data from segment
    virtual size_t read(size_t offset, const void **data, size_t len) = 0;

    virtual size_t read(size_t offset, MemoryBlock &data, size_t len) = 0;

    virtual bool read(SegmentData *, size_t) {
      return false;
    }

    //! Write data into the storage with offset
    virtual size_t write(size_t offset, const void *data, size_t len) = 0;

    //! Resize size of data
    virtual size_t resize(size_t size) = 0;

    //! Update crc of data
    virtual void update_data_crc(uint32_t crc) = 0;

    //! Clone the segment
    virtual Pointer clone(void) = 0;
  };

  //! Destructor
  virtual ~IndexStorage(void) {}

  //! Initialize storage
  virtual int init(const ailego::Params &params) = 0;

  //! Cleanup storage
  virtual int cleanup(void) = 0;

  //! Open storage
  virtual int open(const std::string &path, bool create) = 0;

  //! Flush storage
  virtual int flush(void) = 0;

  //! Close storage
  virtual int close(void) = 0;

  //! Append a segment into storage
  virtual int append(const std::string &id, size_t size) = 0;

  //! Refresh meta information (checksum, update time, etc.)
  virtual void refresh(uint64_t check_point) = 0;

  //! Retrieve check point of storage
  virtual uint64_t check_point(void) const = 0;

  //! Retrieve a segment by id
  virtual Segment::Pointer get(const std::string &id, int level = -1) = 0;

  virtual std::map<std::string, Segment::Pointer> get_all(void) const {
    // LOG_ERROR("get_all() Not Implemented");
    std::map<std::string, Segment::Pointer> result;
    return result;
  }

  //! Test if it a segment exists
  virtual bool has(const std::string &id) const = 0;

  //! Retrieve magic number of index
  virtual uint32_t magic(void) const = 0;

  //! huge page
  virtual bool isHugePage(void) const {
    return false;
  }
};

}  // namespace core
}  // namespace zvec
