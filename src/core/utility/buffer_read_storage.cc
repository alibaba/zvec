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
#include <algorithm>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
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

namespace {

bool ResolveContainerOffset(size_t file_size, int64_t configured_offset,
                            bool zero_from_end, size_t *resolved_offset) {
  const bool absolute =
      configured_offset > 0 || (configured_offset == 0 && !zero_from_end);
  if (absolute) {
    const uint64_t value = static_cast<uint64_t>(configured_offset);
    if (value > file_size) {
      return false;
    }
    *resolved_offset = static_cast<size_t>(value);
    return true;
  }

  const uint64_t distance =
      configured_offset == 0
          ? 0
          : static_cast<uint64_t>(-(configured_offset + 1)) + 1;
  if (distance > file_size) {
    return false;
  }
  *resolved_offset = file_size - static_cast<size_t>(distance);
  return true;
}

}  // namespace

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

    //! Retrieve absolute offset of data within the index file. DiskAnn relies
    //! on this to compute sector addresses; without the override the base
    //! class default (0) would make every sector address wrong.
    size_t data_offset(void) const override {
      return data_offset_;
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
      len = clamp_length(&offset, len);
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
      if (ailego_unlikely(data == nullptr)) {
        return 0;
      }
      len = clamp_length(&offset, len);
      if (len == 0) {
        *data = buffer_.data();
        return 0;
      }
      buffer_.resize(len);
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
      len = clamp_length(&offset, len);
      if (len == 0) {
        data.reset();
        return 0;
      }
      size_t abs_offset = data_offset_ + offset;
      const size_t offset_in_page = abs_offset % ailego::kVectorPageSize;
      if (len <= ailego::kVectorPageSize - offset_in_page) {
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
      if (ailego_unlikely(len >
                          std::numeric_limits<size_t>::max() - (kAlign - 1))) {
        LOG_ERROR(
            "BufferReadStorage::Segment::read(MemoryBlock&): cross-page "
            "length overflow, abs_offset=%zu, len=%zu",
            abs_offset, len);
        return 0;
      }
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
      ailego_false_if_false(iovec != nullptr && count != 0);
      size_t total = 0u;
      for (size_t i = 0; i < count; ++i) {
        const SegmentData &item = iovec[i];
        ailego_false_if_false(item.offset <= region_size_);
        ailego_false_if_false(item.length <= region_size_ - item.offset);
        ailego_false_if_false(item.length <=
                              std::numeric_limits<size_t>::max() - total);
        total += item.length;
      }
      ailego_false_if_false(total != 0);

      buffer_.resize(total);
      uint8_t *buf = buffer_.data();
      for (size_t i = 0; i < count; ++i) {
        SegmentData *it = &iovec[i];
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

    void prefetch(size_t offset, size_t len) override {
      len = clamp_length(&offset, len);
      if (len == 0) return;
      handle_->prefetch_range(data_offset_ + offset, len);
    }

    //! Free bytes in the shared buffer pool.  Used by the caller to decide
    //! whether a whole cluster fits before issuing prefetch.
    size_t prefetch_budget(void) const override {
      return ailego::MemoryLimitPool::get_instance().available();
    }

    //! No stable base pointer: data lives in an evictable paged cache.
    const uint8_t *base_data(void) const override {
      return nullptr;
    }

   private:
    size_t clamp_length(size_t *offset, size_t len) const {
      if (ailego_unlikely(*offset > region_size_)) {
        *offset = region_size_;
        return 0;
      }
      return std::min(len, region_size_ - *offset);
    }

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
    params.get(BUFFER_READ_STORAGE_ENABLE_IO_PROFILE, &enable_io_profile_);
    params.get(BUFFER_READ_STORAGE_WARMUP_MODE, &warmup_mode_);
    if (warmup_mode_ != BUFFER_READ_STORAGE_WARMUP_NONE &&
        warmup_mode_ != BUFFER_READ_STORAGE_WARMUP_SEQUENTIAL) {
      LOG_ERROR("Invalid BufferReadStorage warmup mode: %s",
                warmup_mode_.c_str());
      return IndexError_InvalidArgument;
    }
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
    const size_t shared_cache_capacity =
        ailego::MemoryLimitPool::get_instance().capacity();
    if (shared_cache_capacity < ailego::kVectorPageSize) {
      LOG_ERROR(
          "BufferReadStorage requires at least one cache page: "
          "capacity=%zu page_size=%zu path=%s",
          shared_cache_capacity, ailego::kVectorPageSize, path.c_str());
      return IndexError_InvalidArgument;
    }

    try {
      std::string candidate_file_path(path);
      // Keep all new state local until open is fully successful. A failed
      // reopen therefore cannot mix old segment metadata with a new file.
      auto candidate_pool = std::make_shared<ailego::VecBufferPool>(
          path, /*writable=*/false, /*enable_direct_io=*/enable_direct_io_,
          /*enable_io_profile=*/enable_io_profile_);
      auto candidate_handle = std::make_shared<ailego::VecBufferPoolHandle>(
          candidate_pool->get_handle());

      const size_t file_size = candidate_pool->file_size();
      size_t candidate_index_offset = 0;
      size_t end_offset = 0;
      if (!ResolveContainerOffset(file_size, header_offset_,
                                  /*zero_from_end=*/false,
                                  &candidate_index_offset) ||
          !ResolveContainerOffset(file_size, footer_offset_,
                                  /*zero_from_end=*/true, &end_offset) ||
          candidate_index_offset >= end_offset) {
        LOG_ERROR(
            "Invalid BufferReadStorage container offsets: path=%s "
            "file_size=%zu header_offset=%lld footer_offset=%lld",
            path.c_str(), file_size, static_cast<long long>(header_offset_),
            static_cast<long long>(footer_offset_));
        return IndexError_InvalidArgument;
      }
      const size_t container_size = end_offset - candidate_index_offset;

      // IndexUnpacker requires a stable pointer until its next callback.
      // The local scratch buffer also keeps a failed open from mutating the
      // currently-published storage state.
      std::vector<uint8_t> scratch;
      auto read_data = [&candidate_handle, &scratch, candidate_index_offset,
                        container_size](size_t offset, const void **data,
                                        size_t len) -> size_t {
        if (offset > container_size) {
          offset = container_size;
          len = 0;
        } else {
          len = std::min(len, container_size - offset);
        }
        scratch.resize(len);
        *data = scratch.data();
        if (len == 0) {
          return 0;
        }
        const size_t file_offset = candidate_index_offset + offset;
        if (candidate_handle->get_meta(
                file_offset, len, reinterpret_cast<char *>(scratch.data())) !=
            0) {
          return 0;
        }
        return len;
      };

      IndexUnpacker unpacker;
      if (!unpacker.unpack(read_data, container_size, checksum_validation_)) {
        LOG_ERROR("Failed to unpack file: %s", path.c_str());
        return IndexError_UnpackIndex;
      }
      auto candidate_segments = std::move(*unpacker.mutable_segments());
      for (const auto &item : candidate_segments) {
        const auto &segment = item.second;
        const size_t segment_offset = segment.data_offset();
        if (segment_offset > container_size ||
            segment.data_size() > container_size - segment_offset ||
            segment.padding_size() >
                container_size - segment_offset - segment.data_size()) {
          LOG_ERROR(
              "Invalid BufferReadStorage segment bounds: path=%s id=%s "
              "container_size=%zu offset=%zu data_size=%zu padding_size=%zu",
              path.c_str(), item.first.c_str(), container_size, segment_offset,
              segment.data_size(), segment.padding_size());
          return IndexError_InvalidLength;
        }
      }
      const uint32_t candidate_magic = unpacker.magic();

      // Allocate the page table now that the layout is known.
      if (candidate_pool->init() != 0) {
        LOG_ERROR("Failed to init VecBufferPool, path: %s", path.c_str());
        return IndexError_Runtime;
      }
      if (warmup_mode_ == BUFFER_READ_STORAGE_WARMUP_SEQUENTIAL) {
        candidate_pool->warmup();
      }

      file_path_ = std::move(candidate_file_path);
      index_offset_ = candidate_index_offset;
      magic_ = candidate_magic;
      segments_ = std::move(candidate_segments);
      handle_ = std::move(candidate_handle);
      buffer_pool_ = std::move(candidate_pool);
      return 0;
    } catch (const std::bad_alloc &) {
      LOG_ERROR("Out of memory opening BufferReadStorage: %s", path.c_str());
      return IndexError_NoMemory;
    } catch (const std::runtime_error &error) {
      LOG_ERROR("Failed to open BufferReadStorage file %s: %s", path.c_str(),
                error.what());
      return IndexError_OpenFile;
    } catch (const std::exception &error) {
      LOG_ERROR("Unexpected BufferReadStorage open failure for %s: %s",
                path.c_str(), error.what());
      return IndexError_Runtime;
    } catch (...) {
      LOG_ERROR("Unknown BufferReadStorage open failure for %s", path.c_str());
      return IndexError_Runtime;
    }
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

  //! Path of the opened index file (diagnostics / backend consistency).
  std::string file_path(void) const override {
    return file_path_;
  }

  //! Expose the backing VecBufferPool so callers (e.g. DiskAnn) can detect a
  //! pooled backend and route reads through the paged cache.
  ailego::VecBufferPool *vec_buffer_pool(void) const override {
    return buffer_pool_.get();
  }

 private:
  bool checksum_validation_{false};
  bool enable_direct_io_{true};
  bool enable_io_profile_{false};
  std::string warmup_mode_{BUFFER_READ_STORAGE_WARMUP_SEQUENTIAL};
  int64_t header_offset_{0};
  int64_t footer_offset_{0};
  size_t index_offset_{0};
  uint32_t magic_{0};
  std::string file_path_{};
  std::map<std::string, IndexUnpacker::SegmentMeta> segments_{};
  std::shared_ptr<ailego::VecBufferPool> buffer_pool_{nullptr};
  std::shared_ptr<ailego::VecBufferPoolHandle> handle_{nullptr};
};

INDEX_FACTORY_REGISTER_STORAGE(BufferReadStorage);

}  // namespace core
}  // namespace zvec
