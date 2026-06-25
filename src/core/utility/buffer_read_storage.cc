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
//
// BufferReadStorage is a read-only IndexStorage that mirrors the structure of
// MMapFileReadStorage (it parses the FileDumper container layout through
// IndexUnpacker and exposes segment-based access), but instead of mmap-ing the
// file it reads through a VecBufferPool.  This lets IVF / DiskANN(Vamana)
// indexes -- which are dumped via FileDumper -- benefit from the buffer-pool's
// paged cache + LRU eviction + memory-budget control, while keeping the same
// Segment interface that those indexes already consume.
#include <zvec/ailego/buffer/vector_page_table.h>
#include <zvec/ailego/internal/platform.h>
#include <zvec/ailego/utility/file_helper.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_format.h>
#include <zvec/core/framework/index_unpacker.h>
#include "utility_params.h"

namespace zvec {
namespace core {

/*! Buffer Read Storage (backed by VecBufferPool)
 */
class BufferReadStorage : public IndexStorage {
 public:
  /*! Buffer Read Storage Segment
   *
   * Each segment keeps the owning VecBufferPool / VecBufferPoolHandle alive
   * (shared_ptr) so that pages it reads remain valid for the segment's
   * lifetime.  Reads go through the pool's paged cache:
   *   - fetch()              -> read_range into the caller's buffer
   *   - read(const void**)   -> read_range into a per-segment buffer (stable
   *                             pointer, never pins a page)
   *   - read(MemoryBlock&)   -> single page: zero-copy pin tied to the
   *                             MemoryBlock lifecycle; cross page: owned copy
   *   - read(SegmentData*)   -> read_range into the per-segment buffer
   */
  class Segment : public IndexStorage::Segment,
                  public std::enable_shared_from_this<Segment> {
   public:
    //! Index Storage Pointer
    typedef std::shared_ptr<Segment> Pointer;

    //! Constructor
    Segment(const std::shared_ptr<ailego::VecBufferPool> &pool,
            const std::shared_ptr<ailego::VecBufferPoolHandle> &handle,
            size_t index_offset, const IndexUnpacker::SegmentMeta &segment)
        : data_offset_(index_offset + segment.data_offset()),
          data_size_(segment.data_size()),
          padding_size_(segment.padding_size()),
          region_size_(segment.data_size() + segment.padding_size()),
          data_crc_(segment.data_crc()),
          pool_(pool),
          handle_(handle) {}

    //! Constructor (clone)
    Segment(const Segment &rhs)
        : std::enable_shared_from_this<Segment>(),
          data_offset_(rhs.data_offset_),
          data_size_(rhs.data_size_),
          padding_size_(rhs.padding_size_),
          region_size_(rhs.region_size_),
          data_crc_(rhs.data_crc_),
          pool_(rhs.pool_),
          handle_(rhs.handle_) {}

    //! Destructor
    ~Segment(void) override {}

    //! Retrieve size of data
    size_t data_size(void) const override {
      return data_size_;
    }

    //! Retrieve crc of data
    uint32_t data_crc(void) const override {
      return data_crc_;
    }

    //! Retrieve size of padding
    size_t padding_size(void) const override {
      return padding_size_;
    }

    //! Retrieve capacity of segment
    size_t capacity(void) const override {
      return region_size_;
    }

    //! Fetch data from segment (copies into the caller-owned buffer)
    size_t fetch(size_t offset, void *buf, size_t len) const override {
      if (ailego_unlikely(offset + len > region_size_)) {
        if (offset > region_size_) {
          offset = region_size_;
        }
        len = region_size_ - offset;
      }
      if (len == 0) {
        return 0;
      }
      if (!handle_->read_range(data_offset_ + offset, len,
                               static_cast<char *>(buf))) {
        LOG_ERROR(
            "BufferReadStorage::Segment::fetch: read_range failed, "
            "abs_offset=%zu, len=%zu",
            data_offset_ + offset, len);
        return 0;
      }
      return len;
    }

    //! Read data from segment (stable pointer via per-segment buffer)
    size_t read(size_t offset, const void **data, size_t len) override {
      if (ailego_unlikely(offset + len > region_size_)) {
        if (offset > region_size_) {
          offset = region_size_;
        }
        len = region_size_ - offset;
      }
      if (len == 0) {
        *data = buffer_.data();
        return 0;
      }
      buffer_.reserve(len);
      if (!handle_->read_range(data_offset_ + offset, len,
                               reinterpret_cast<char *>(buffer_.data()))) {
        LOG_ERROR(
            "BufferReadStorage::Segment::read: read_range failed, "
            "abs_offset=%zu, len=%zu",
            data_offset_ + offset, len);
        *data = nullptr;
        return 0;
      }
      *data = buffer_.data();
      return len;
    }

    //! Read data from segment into a MemoryBlock
    size_t read(size_t offset, MemoryBlock &data, size_t len) override {
      if (ailego_unlikely(offset + len > region_size_)) {
        if (offset > region_size_) {
          offset = region_size_;
        }
        len = region_size_ - offset;
      }
      size_t abs_offset = data_offset_ + offset;
      size_t first_page = abs_offset / ailego::kVectorPageSize;
      size_t last_page = (len == 0)
                             ? first_page
                             : (abs_offset + len - 1) / ailego::kVectorPageSize;
      if (first_page == last_page) {
        // Single-page: zero-copy pin whose release is tied to the
        // MemoryBlock lifecycle (release_one on destruction).
        size_t page_id = 0;
        char *raw = handle_->get_single_page(abs_offset, len, page_id);
        if (!raw) {
          LOG_ERROR(
              "BufferReadStorage::Segment::read(MemoryBlock&): single-page "
              "acquire failed, abs_offset=%zu, len=%zu",
              abs_offset, len);
          return 0;
        }
        data.reset(handle_.get(), page_id, raw);
        return len;
      }
      // Cross-page: copy into a freshly-allocated 4K-aligned buffer that the
      // MemoryBlock owns (freed via ailego_free on destruction).
      static constexpr size_t kAlign = 4096UL;
      size_t alloc_size = (len + (kAlign - 1UL)) & ~(kAlign - 1UL);
      char *tmp =
          static_cast<char *>(ailego_aligned_malloc(alloc_size, kAlign));
      if (!tmp) {
        LOG_ERROR(
            "BufferReadStorage::Segment::read(MemoryBlock&): cross-page alloc "
            "failed, abs_offset=%zu, len=%zu",
            abs_offset, len);
        return 0;
      }
      if (!handle_->read_range(abs_offset, len, tmp)) {
        ailego_free(tmp);
        LOG_ERROR(
            "BufferReadStorage::Segment::read(MemoryBlock&): cross-page "
            "read_range failed, abs_offset=%zu, len=%zu",
            abs_offset, len);
        return 0;
      }
      data = MemoryBlock::MakeOwned(tmp, len);
      return len;
    }

    //! Read scattered data from segment (stable pointers via per-segment buf)
    bool read(SegmentData *iovec, size_t count) override {
      size_t total = 0u;
      for (auto *it = iovec, *end = iovec + count; it != end; ++it) {
        ailego_false_if_false(it->offset + it->length <= region_size_);
        total += it->length;
      }
      ailego_false_if_false(total != 0);

      buffer_.reserve(total);
      uint8_t *buf = buffer_.data();
      for (auto *it = iovec, *end = iovec + count; it != end; ++it) {
        ailego_false_if_false(
            handle_->read_range(data_offset_ + it->offset, it->length,
                                reinterpret_cast<char *>(buf)));
        it->data = buf;
        buf += it->length;
      }
      return true;
    }

    size_t write(size_t, const void *, size_t) override {
      return IndexError_NotImplemented;
    }

    size_t resize(size_t) override {
      return IndexError_NotImplemented;
    }

    void update_data_crc(uint32_t) override {
      return;
    }

    //! Clone the segment
    IndexStorage::Segment::Pointer clone(void) override {
      return std::make_shared<BufferReadStorage::Segment>(*this);
    }

    //! No stable base pointer: data lives in an evictable paged cache.
    const uint8_t *base_data(void) const override {
      return nullptr;
    }

   private:
    size_t data_offset_{0u};
    size_t data_size_{0u};
    size_t padding_size_{0u};
    size_t region_size_{0u};
    uint32_t data_crc_{0u};
    std::vector<uint8_t> buffer_{};
    std::shared_ptr<ailego::VecBufferPool> pool_{nullptr};
    std::shared_ptr<ailego::VecBufferPoolHandle> handle_{nullptr};
  };

  //! Destructor
  ~BufferReadStorage(void) override {}

  //! Initialize container
  int init(const ailego::Params &params) override {
    params.get(BUFFER_READ_STORAGE_CHECKSUM_VALIDATION, &checksum_validation_);
    params.get(BUFFER_READ_STORAGE_HEADER_OFFSET, &header_offset_);
    params.get(BUFFER_READ_STORAGE_FOOTER_OFFSET, &footer_offset_);
    params.get(BUFFER_READ_STORAGE_ENABLE_DIRECT_IO, &enable_direct_io_);
    enable_direct_io_ = true;
    return 0;
  }

  int flush(void) override {
    return 0;
  }

  int append(const std::string &, size_t) override {
    return IndexError_NotImplemented;
  }

  void refresh(uint64_t) override {
    return;
  }

  uint64_t check_point(void) const override {
    return 0;
  }

  //! Cleanup container
  int cleanup(void) override {
    return this->close();
  }

  //! Load an index file into the container
  int open(const std::string &path, bool) override {
    // Read-only buffer pool over the freshly-dumped FileDumper container.
    buffer_pool_ = std::make_shared<ailego::VecBufferPool>(
        path, /*writable=*/false, /*enable_direct_io=*/enable_direct_io_);
    if (!buffer_pool_) {
      LOG_ERROR("Failed to create VecBufferPool, path: %s", path.c_str());
      return IndexError_NoMemory;
    }
    handle_ = std::make_shared<ailego::VecBufferPoolHandle>(
        buffer_pool_->get_handle());

    size_t file_size = buffer_pool_->file_size();
    index_offset_ = (header_offset_ >= 0 ? 0 : file_size) + header_offset_;
    size_t end_offset = (footer_offset_ > 0 ? 0 : file_size) + footer_offset_;
    size_t size = end_offset > index_offset_ ? end_offset - index_offset_ : 0;

    // read_data for IndexUnpacker: provide a stable pointer by copying the
    // requested range into a reused scratch buffer via get_meta (direct
    // pread, valid before buffer_pool_->init()).
    auto read_data = [this, end_offset](size_t offset, const void **data,
                                        size_t len) -> size_t {
      size_t off = offset + index_offset_;
      if (off + len > end_offset) {
        if (off > end_offset) {
          off = end_offset;
        }
        len = end_offset - off;
      }
      scratch_.reserve(len);
      *data = scratch_.data();
      if (len == 0) {
        return 0;
      }
      if (handle_->get_meta(off, len,
                            reinterpret_cast<char *>(scratch_.data())) != 0) {
        return 0;
      }
      return len;
    };

    IndexUnpacker unpacker;
    if (!unpacker.unpack(read_data, size, checksum_validation_)) {
      LOG_ERROR("Failed to unpack file: %s", path.c_str());
      return IndexError_UnpackIndex;
    }
    segments_ = std::move(*unpacker.mutable_segments());
    magic_ = unpacker.magic();

    // Allocate the page table now that the layout is known.
    int ret = buffer_pool_->init();
    if (ret != 0) {
      LOG_ERROR("Failed to init VecBufferPool, path: %s", path.c_str());
      return IndexError_Runtime;
    }
    buffer_pool_->warmup();
    return 0;
  }

  int close(void) override {
    segments_.clear();
    handle_ = nullptr;
    buffer_pool_ = nullptr;
    return 0;
  }

  //! Retrieve a segment by id
  IndexStorage::Segment::Pointer get(const std::string &id, int) override {
    if (!buffer_pool_ || !handle_) {
      return IndexStorage::Segment::Pointer();
    }
    auto it = segments_.find(id);
    if (it == segments_.end()) {
      return IndexStorage::Segment::Pointer();
    }
    return std::make_shared<BufferReadStorage::Segment>(
        buffer_pool_, handle_, index_offset_, it->second);
  }

  std::map<std::string, IndexStorage::Segment::Pointer> get_all(
      void) const override {
    std::map<std::string, IndexStorage::Segment::Pointer> result;
    if (buffer_pool_ && handle_) {
      for (const auto &it : segments_) {
        result.emplace(it.first,
                       std::make_shared<BufferReadStorage::Segment>(
                           buffer_pool_, handle_, index_offset_, it.second));
      }
    }
    return result;
  }

  //! Test if a segment exists
  bool has(const std::string &id) const override {
    return (segments_.find(id) != segments_.end());
  }

  //! Retrieve magic number of index
  uint32_t magic(void) const override {
    return magic_;
  }

  //! Reads go through the VecBufferPool paged cache.
  MemoryBlock::MemoryBlockType memory_block_type(void) const override {
    return MemoryBlock::MBT_BUFFERPOOL;
  }

 private:
  bool checksum_validation_{false};
  bool enable_direct_io_{true};
  int64_t header_offset_{0};
  int64_t footer_offset_{0};
  size_t index_offset_{0};
  uint32_t magic_{0};
  std::vector<uint8_t> scratch_{};
  std::map<std::string, IndexUnpacker::SegmentMeta> segments_{};
  std::shared_ptr<ailego::VecBufferPool> buffer_pool_{nullptr};
  std::shared_ptr<ailego::VecBufferPoolHandle> handle_{nullptr};
};

INDEX_FACTORY_REGISTER_STORAGE(BufferReadStorage);

}  // namespace core
}  // namespace zvec
