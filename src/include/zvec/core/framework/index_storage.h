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
#include <memory>
#include <new>
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

    MemoryBlock() = default;
    MemoryBlock(ailego::VecBufferPoolHandle *buffer_pool_handle,
                size_t block_id, void *data)
        : type_(MemoryBlockType::MBT_BUFFERPOOL) {
      buffer_pool_handle_ = buffer_pool_handle;
      buffer_block_id_ = block_id;
      data_ = data;
    }
    MemoryBlock(
        const std::shared_ptr<ailego::VecBufferPoolHandle> &buffer_pool_handle,
        size_t block_id, void *data)
        : type_(MemoryBlockType::MBT_BUFFERPOOL),
          data_(data),
          buffer_pool_handle_owner_(buffer_pool_handle),
          buffer_pool_handle_(buffer_pool_handle.get()),
          buffer_block_id_(block_id) {}
    MemoryBlock(void *data) : type_(MemoryBlockType::MBT_MMAP), data_(data) {}

    //! Build an owned heap block; size enables safe deep copies.
    static MemoryBlock MakeOwned(void *owned, size_t size) {
      MemoryBlock mb;
      mb.type_ = MemoryBlockType::MBT_HEAP_SCRATCH;
      mb.data_ = owned;
      mb.scratch_size_ = size;
      return mb;
    }

    //! Build a non-owning view over caller-managed memory (e.g. a query-level
    //! scratch arena). The block frees and pins nothing on destruction, so the
    //! backing buffer must outlive every copy of this block. Uses the MMAP
    //! representation whose destructor is a no-op.
    static MemoryBlock MakeBorrowedView(void *data) {
      MemoryBlock mb;
      mb.type_ = MemoryBlockType::MBT_MMAP;
      mb.data_ = data;
      return mb;
    }

    MemoryBlock(const MemoryBlock &rhs) {
      switch (rhs.type_) {
        case MemoryBlockType::MBT_MMAP:
          this->reset(rhs.data_);
          break;
        case MemoryBlockType::MBT_BUFFERPOOL:
          if (rhs.buffer_pool_handle_owner_) {
            this->reset(rhs.buffer_pool_handle_owner_, rhs.buffer_block_id_,
                        rhs.data_);
          } else {
            this->reset(rhs.buffer_pool_handle_, rhs.buffer_block_id_,
                        rhs.data_);
          }
          buffer_pool_handle_->acquire_one(buffer_block_id_);
          break;
        case MemoryBlockType::MBT_HEAP_SCRATCH:
          // Heap blocks do not share ownership.
          deep_copy_from(rhs);
          break;
        default:
          break;
      }
    }

    MemoryBlock(MemoryBlock &&rhs) noexcept {
      switch (rhs.type_) {
        case MemoryBlockType::MBT_MMAP:
          this->reset(rhs.data_);
          break;
        case MemoryBlockType::MBT_BUFFERPOOL:
          type_ = MemoryBlockType::MBT_BUFFERPOOL;
          data_ = rhs.data_;
          buffer_pool_handle_owner_ = std::move(rhs.buffer_pool_handle_owner_);
          buffer_pool_handle_ = rhs.buffer_pool_handle_;
          buffer_block_id_ = rhs.buffer_block_id_;
          rhs.buffer_pool_handle_ = nullptr;
          rhs.data_ = nullptr;
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
            if (rhs.buffer_pool_handle_owner_) {
              this->reset(rhs.buffer_pool_handle_owner_, rhs.buffer_block_id_,
                          rhs.data_);
            } else {
              this->reset(rhs.buffer_pool_handle_, rhs.buffer_block_id_,
                          rhs.data_);
            }
            buffer_pool_handle_->acquire_one(buffer_block_id_);
            break;
          case MemoryBlockType::MBT_HEAP_SCRATCH:
            release_current();
            deep_copy_from(rhs);
            break;
          default:
            release_current();
            break;
        }
      }
      return *this;
    }

    MemoryBlock &operator=(MemoryBlock &&rhs) noexcept {
      if (this != &rhs) {
        switch (rhs.type_) {
          case MemoryBlockType::MBT_MMAP:
            this->reset(rhs.data_);
            break;
          case MemoryBlockType::MBT_BUFFERPOOL:
            release_current();
            type_ = MemoryBlockType::MBT_BUFFERPOOL;
            data_ = rhs.data_;
            buffer_pool_handle_owner_ =
                std::move(rhs.buffer_pool_handle_owner_);
            buffer_pool_handle_ = rhs.buffer_pool_handle_;
            buffer_block_id_ = rhs.buffer_block_id_;
            rhs.buffer_pool_handle_ = nullptr;
            rhs.data_ = nullptr;
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
            release_current();
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

    void reset() {
      release_current();
    }

    void reset(ailego::VecBufferPoolHandle *buffer_pool_handle, size_t block_id,
               void *data) {
      release_current();
      type_ = MemoryBlockType::MBT_BUFFERPOOL;
      buffer_pool_handle_ = buffer_pool_handle;
      buffer_block_id_ = block_id;
      data_ = data;
    }

    void reset(
        const std::shared_ptr<ailego::VecBufferPoolHandle> &buffer_pool_handle,
        size_t block_id, void *data) {
      release_current();
      type_ = MemoryBlockType::MBT_BUFFERPOOL;
      buffer_pool_handle_owner_ = buffer_pool_handle;
      buffer_pool_handle_ = buffer_pool_handle.get();
      buffer_block_id_ = block_id;
      data_ = data;
    }

    void reset(void *data) {
      if (type_ == MemoryBlockType::MBT_BUFFERPOOL) {
        if (buffer_pool_handle_) {
          buffer_pool_handle_->release_one(buffer_block_id_);
        }
        buffer_pool_handle_ = nullptr;
        buffer_pool_handle_owner_.reset();
      } else if (type_ == MemoryBlockType::MBT_HEAP_SCRATCH) {
        release_owned();
      }
      type_ = MemoryBlockType::MBT_MMAP;
      data_ = data;
    }

    MemoryBlockType type_{MBT_UNKNOWN};
    void *data_{nullptr};
    std::shared_ptr<ailego::VecBufferPoolHandle> buffer_pool_handle_owner_{};
    mutable ailego::VecBufferPoolHandle *buffer_pool_handle_{nullptr};
    size_t buffer_block_id_{0};
    //! Byte size used to copy heap scratch blocks.
    size_t scratch_size_{0};

   private:
    void release_owned() {
      if (data_) {
        ailego_free(data_);
        data_ = nullptr;
      }
      scratch_size_ = 0;
    }

    //! Release the current representation and reset the block.
    void release_current() {
      switch (type_) {
        case MemoryBlockType::MBT_BUFFERPOOL:
          if (buffer_pool_handle_) {
            buffer_pool_handle_->release_one(buffer_block_id_);
            buffer_pool_handle_ = nullptr;
          }
          buffer_pool_handle_owner_.reset();
          break;
        case MemoryBlockType::MBT_HEAP_SCRATCH:
          release_owned();
          break;
        default:
          break;
      }
      data_ = nullptr;
      scratch_size_ = 0;
      type_ = MemoryBlockType::MBT_UNKNOWN;
    }

    //! Deep-copy heap scratch ownership.
    void deep_copy_from(const MemoryBlock &rhs) {
      type_ = MemoryBlockType::MBT_HEAP_SCRATCH;
      scratch_size_ = rhs.scratch_size_;
      if (scratch_size_ > 0 && rhs.data_) {
        data_ = ailego_malloc(scratch_size_);
        if (data_ == nullptr) {
          scratch_size_ = 0;
          type_ = MemoryBlockType::MBT_UNKNOWN;
          throw std::bad_alloc();
        }
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

    //! Cache admission/eviction hint. Backends without an evictable cache
    //! ignore it; page-backed storage maps it to its eviction queues.
    enum class CachePriority : uint8_t {
      kLow = 0,
      kNormal = 1,
      kHigh = 2,
    };

    //! One bounded-lifetime read in a batch. Requests may reference different
    //! segments owned by the same storage so backends can merge their page
    //! misses into one I/O submission.
    struct BorrowedRead {
      BorrowedRead(Segment *segment_arg, size_t offset_arg, size_t length_arg,
                   MemoryBlock *block_arg)
          : segment(segment_arg),
            offset(offset_arg),
            length(length_arg),
            block(block_arg) {}

      Segment *segment;
      size_t offset;
      size_t length;
      MemoryBlock *block;
    };

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

    //! Read bytes that the caller guarantees will never be modified after
    //! publication. Page-backed writable storage may safely return a pinned
    //! direct view instead of copying the range for snapshot isolation.
    virtual size_t read_immutable(size_t offset, MemoryBlock &data,
                                  size_t len) {
      return read(offset, data, len);
    }

    //! Borrowed read; release the block before this Segment. The default keeps
    //! the owning read behavior.
    virtual size_t read_borrowed(size_t offset, MemoryBlock &data, size_t len) {
      return read(offset, data, len);
    }

    //! Borrowed-handle variant of read_immutable().
    virtual size_t read_borrowed_immutable(size_t offset, MemoryBlock &data,
                                           size_t len) {
      return read_immutable(offset, data, len);
    }

    //! Whether batching is currently preferable to scalar borrowed reads.
    virtual bool prefer_borrowed_batch() const {
      return false;
    }

    //! Batch borrowed reads. The default preserves compatibility by issuing
    //! scalar reads; page-backed implementations may override this to batch
    //! misses while keeping each returned MemoryBlock pinned independently.
    virtual bool read_borrowed_batch(BorrowedRead *reads, size_t count) {
      if (count == 0) {
        return true;
      }
      if (reads == nullptr) {
        return false;
      }
      for (size_t i = 0; i < count; ++i) {
        if (reads[i].segment == nullptr || reads[i].block == nullptr) {
          return false;
        }
      }
      for (size_t i = 0; i < count; ++i) {
        reads[i].block->reset();
      }
      for (size_t i = 0; i < count; ++i) {
        BorrowedRead &request = reads[i];
        if (request.segment->read_borrowed(request.offset, *request.block,
                                           request.length) != request.length) {
          for (size_t j = 0; j < count; ++j) {
            reads[j].block->reset();
          }
          return false;
        }
      }
      return true;
    }

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

    //! Return a stable base address, or nullptr for evictable storage.
    virtual const uint8_t *base_data(void) const {
      return nullptr;
    }

    virtual size_t abs_data_offset(void) const {
      return 0;
    }

    virtual void prefetch(size_t offset, size_t len,
                          CachePriority priority = CachePriority::kLow) {
      (void)offset;
      (void)len;
      (void)priority;
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

  //! Return the shared page cache when this storage is backed by VecBufferPool.
  virtual std::shared_ptr<ailego::VecBufferPool> vec_buffer_pool(void) const {
    return nullptr;
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
