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

#include <cstring>
#include <zvec/ailego/buffer/vector_page_table.h>
#include <zvec/ailego/container/params.h>
#include <zvec/ailego/io/file.h>
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
      MBT_HEAP_SCRATCH = 3,
    };

    MemoryBlock() {}
    MemoryBlock(ailego::VecBufferPoolHandle *buffer_pool_handle,
                size_t block_id, void *data)
        : type_(MemoryBlockType::MBT_BUFFERPOOL) {
      buffer_pool_handle_ = buffer_pool_handle;
      buffer_block_id_ = block_id;
      data_ = data;
    }
    MemoryBlock(void *data) : type_(MemoryBlockType::MBT_MMAP), data_(data) {}

    //! Build an HEAP_SCRATCH MemoryBlock that owns `owned` (allocated via
    //! ailego_malloc / ailego_aligned_malloc).  `size` is the byte length of
    //! the buffer and is required so that copy construction / copy
    //! assignment can deep-copy the buffer instead of aliasing it (a shallow
    //! copy would result in use-after-free once the original block is
    //! destructed and frees the buffer).
    static MemoryBlock MakeOwned(void *owned, size_t size) {
      MemoryBlock mb;
      mb.type_ = MemoryBlockType::MBT_HEAP_SCRATCH;
      mb.data_ = owned;
      mb.scratch_size_ = size;
      return mb;
    }

    MemoryBlock(const MemoryBlock &rhs) {
      switch (rhs.type_) {
        case MemoryBlockType::MBT_MMAP:
          this->reset(rhs.data_);
          break;
        case MemoryBlockType::MBT_BUFFERPOOL:
          this->reset(rhs.buffer_pool_handle_, rhs.buffer_block_id_, rhs.data_);
          buffer_pool_handle_->acquire_one(buffer_block_id_);
          break;
        case MemoryBlockType::MBT_HEAP_SCRATCH:
          // Deep copy: each owner must hold its own buffer, otherwise the
          // first destructor frees the buffer and leaves the surviving
          // copies dangling.
          deep_copy_from(rhs);
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
          this->reset(std::move(rhs.buffer_pool_handle_),
                      std::move(rhs.buffer_block_id_), std::move(rhs.data_));
          rhs.buffer_pool_handle_ = nullptr;
          rhs.type_ = MemoryBlockType::MBT_UNKNOWN;
          break;
        case MemoryBlockType::MBT_HEAP_SCRATCH:
          type_ = MemoryBlockType::MBT_HEAP_SCRATCH;
          data_ = rhs.data_;
          scratch_size_ = rhs.scratch_size_;
          rhs.data_ = nullptr;
          rhs.scratch_size_ = 0;
          rhs.type_ = MemoryBlockType::MBT_UNKNOWN;
          break;
        default:
          break;
      }
    }

    MemoryBlock &operator=(const MemoryBlock &rhs) {
      if (this != &rhs) {
        switch (rhs.type_) {
          case MemoryBlockType::MBT_MMAP:
            this->reset(rhs.data_);
            break;
          case MemoryBlockType::MBT_BUFFERPOOL:
            this->reset(rhs.buffer_pool_handle_, rhs.buffer_block_id_,
                        rhs.data_);
            buffer_pool_handle_->acquire_one(buffer_block_id_);
            break;
          case MemoryBlockType::MBT_HEAP_SCRATCH:
            release_current();
            deep_copy_from(rhs);
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
            this->reset(std::move(rhs.buffer_pool_handle_),
                        std::move(rhs.buffer_block_id_), std::move(rhs.data_));
            rhs.buffer_pool_handle_ = nullptr;
            rhs.type_ = MemoryBlockType::MBT_UNKNOWN;
            break;
          case MemoryBlockType::MBT_HEAP_SCRATCH:
            release_current();
            type_ = MemoryBlockType::MBT_HEAP_SCRATCH;
            data_ = rhs.data_;
            scratch_size_ = rhs.scratch_size_;
            rhs.data_ = nullptr;
            rhs.scratch_size_ = 0;
            rhs.type_ = MemoryBlockType::MBT_UNKNOWN;
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
          if (buffer_pool_handle_) {
            buffer_pool_handle_->release_one(buffer_block_id_);
          }
          break;
        case MemoryBlockType::MBT_HEAP_SCRATCH:
          release_owned();
          break;
        default:
          break;
      }
      data_ = nullptr;
      scratch_size_ = 0;
    }

    const void *data() const {
      return data_;
    }

    void reset(ailego::VecBufferPoolHandle *buffer_pool_handle, size_t block_id,
               void *data) {
      if (type_ == MemoryBlockType::MBT_BUFFERPOOL) {
        buffer_pool_handle_->release_one(buffer_block_id_);
      } else if (type_ == MemoryBlockType::MBT_HEAP_SCRATCH) {
        release_owned();
      }
      type_ = MemoryBlockType::MBT_BUFFERPOOL;
      buffer_pool_handle_ = buffer_pool_handle;
      buffer_block_id_ = block_id;
      data_ = data;
    }

    void reset(void *data) {
      if (type_ == MemoryBlockType::MBT_BUFFERPOOL) {
        buffer_pool_handle_->release_one(buffer_block_id_);
        buffer_pool_handle_ = nullptr;
      } else if (type_ == MemoryBlockType::MBT_HEAP_SCRATCH) {
        release_owned();
      }
      type_ = MemoryBlockType::MBT_MMAP;
      data_ = data;
    }

    MemoryBlockType type_{MBT_UNKNOWN};
    void *data_{nullptr};
    mutable ailego::VecBufferPoolHandle *buffer_pool_handle_{nullptr};
    size_t buffer_block_id_{0};
    //! Byte size of the heap-scratch buffer pointed to by `data_`; only used
    //! when type_ == MBT_HEAP_SCRATCH.  Required for safe deep-copy on
    //! copy-construction / copy-assignment of HEAP_SCRATCH blocks.
    size_t scratch_size_{0};

   private:
    void release_owned() {
      if (data_) {
        ailego_free(data_);
        data_ = nullptr;
      }
      scratch_size_ = 0;
    }

    //! Drop whatever the current MemoryBlock holds, regardless of type, so
    //! that the slot is ready to receive new ownership.  Mirrors what the
    //! destructor would do (minus zeroing data_) but leaves the type alone
    //! for the caller to overwrite immediately afterwards.
    void release_current() {
      switch (type_) {
        case MemoryBlockType::MBT_BUFFERPOOL:
          if (buffer_pool_handle_) {
            buffer_pool_handle_->release_one(buffer_block_id_);
            buffer_pool_handle_ = nullptr;
          }
          break;
        case MemoryBlockType::MBT_HEAP_SCRATCH:
          release_owned();
          break;
        default:
          break;
      }
      data_ = nullptr;
      type_ = MemoryBlockType::MBT_UNKNOWN;
    }

    //! Allocate a fresh buffer of the same size as `rhs.scratch_size_`,
    //! memcpy `rhs.data_` into it, and become the new owner.  Used by the
    //! HEAP_SCRATCH copy ctor / copy assignment so the original and the
    //! copy each free their own buffer independently.
    void deep_copy_from(const MemoryBlock &rhs) {
      type_ = MemoryBlockType::MBT_HEAP_SCRATCH;
      scratch_size_ = rhs.scratch_size_;
      if (scratch_size_ > 0 && rhs.data_) {
        data_ = ailego_malloc(scratch_size_);
        std::memcpy(data_, rhs.data_, scratch_size_);
      } else {
        data_ = nullptr;
      }
    }
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

    //! Retrieve offset of data
    virtual size_t data_offset(void) const {
      return 0;
    }

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

    //! Retrieve the stable base data pointer if the storage backend supports
    //! it (e.g. mmap-backed storage). Returns nullptr for backends with
    //! mutable/evictable buffers (e.g. BufferStorage). When non-null the
    //! caller may compute element addresses as base_data() + offset directly,
    //! avoiding the full pointer chain through chunk->read().
    virtual const uint8_t *base_data(void) const {
      return nullptr;
    }
  };

  //! Destructor
  ~IndexStorage(void) override {}

  //! Initialize storage
  virtual int init(const ailego::Params &params) = 0;

  //! Cleanup storage
  virtual int cleanup(void) = 0;

  //! Open storage
  virtual int open(const std::string &path, bool create_if_missing) = 0;

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

  //! Retrieve the memory block type of this storage
  virtual MemoryBlock::MemoryBlockType memory_block_type(void) const {
    return MemoryBlock::MBT_MMAP;
  }

  //! Test if the storage has unflushed data
  virtual bool is_dirty(void) const {
    return false;
  }

  //! Retrieve file ptr if has
  virtual std::shared_ptr<ailego::File> file(void) const {
    return nullptr;
  }

  virtual std::string file_path(void) const {
    return "";
  }
};

}  // namespace core
}  // namespace zvec
