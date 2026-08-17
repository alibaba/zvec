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
// Read-only FileDumper storage backed by VecBufferPool instead of mmap.
#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <zvec/ailego/buffer/vector_page_table.h>
#include <zvec/ailego/internal/platform.h>
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

class BufferPageLease {
 public:
  BufferPageLease(
      std::shared_ptr<ailego::VecBufferPoolHandle> handle,
      const std::vector<ailego::block_id_t> &page_ids)
      : handle_(std::move(handle)), page_ids_(page_ids) {}

  BufferPageLease(const BufferPageLease &) = delete;
  BufferPageLease &operator=(const BufferPageLease &) = delete;

  ~BufferPageLease() {
    if (handle_ && !page_ids_.empty()) {
      handle_->release_pages(page_ids_.data(), page_ids_.size());
    }
  }

 private:
  std::shared_ptr<ailego::VecBufferPoolHandle> handle_{};
  std::vector<ailego::block_id_t> page_ids_{};
};

}  // namespace

/*! Buffer Read Storage (backed by VecBufferPool)
 */
class BufferReadStorage : public IndexStorage {
 public:
  /*! Read-only segment. Resident single-page reads pin cache pages; other
   * reads use thread-local or owned scratch buffers.
   */
  class Segment : public IndexStorage::Segment {
   public:
    //! Constructor
    Segment(const std::shared_ptr<ailego::VecBufferPoolHandle> &handle,
            bool cache_enabled, size_t index_offset,
            const IndexUnpacker::SegmentMeta &segment)
        : data_offset_(index_offset + segment.data_offset()),
          data_size_(segment.data_size()),
          padding_size_(segment.padding_size()),
          region_size_(segment.data_size() + segment.padding_size()),
          data_crc_(segment.data_crc()),
          handle_(handle),
          cache_enabled_(cache_enabled) {}

    //! Constructor (clone)
    Segment(const Segment &rhs)
        : data_offset_(rhs.data_offset_),
          data_size_(rhs.data_size_),
          padding_size_(rhs.padding_size_),
          region_size_(rhs.region_size_),
          data_crc_(rhs.data_crc_),
          handle_(rhs.handle_),
          cache_enabled_(rhs.cache_enabled_) {}

    //! Destructor. Release scratch eagerly on long-lived worker threads, but
    //! do not touch the registry after it has entered TLS teardown (an Index
    //! context may outlive the registry because of TLS destruction order).
    ~Segment(void) override {
      if (thread_scratch_registry_alive()) {
        release_thread_scratch();
      }
    }

    //! Retrieve size of data
    size_t data_size(void) const override {
      return data_size_;
    }

    //! Retrieve the absolute data offset used by sector readers.
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
      if (!read_bytes(data_offset_ + offset, len, static_cast<char *>(buf))) {
        LOG_ERROR(
            "BufferReadStorage::Segment::fetch: read_range failed, "
            "abs_offset=%zu, len=%zu",
            data_offset_ + offset, len);
        return 0;
      }
      return len;
    }

    //! Read data from segment (stable until this thread's next pointer read)
    size_t read(size_t offset, const void **data, size_t len) override {
      if (ailego_unlikely(data == nullptr)) {
        return 0;
      }
      auto scratch = thread_scratch();
      scratch->release_pin();
      len = clamp_length(&offset, len);
      if (len == 0) {
        *data = scratch->buffer.data();
        return 0;
      }
      const size_t abs_offset = data_offset_ + offset;
      const size_t offset_in_page = abs_offset % ailego::kVectorPageSize;
      bool force_bypass = !cache_enabled_;
      if (cache_enabled_ && len <= ailego::kVectorPageSize - offset_in_page) {
        size_t page_id = 0;
        char *raw = scratch->handle->get_single_page(abs_offset, len, page_id);
        if (raw != nullptr) {
          scratch->pin(page_id);
          *data = raw;
          return len;
        }
        force_bypass = true;
      }
      scratch->buffer.resize(len);
      const bool read_ok =
          force_bypass
              ? scratch->handle->read_range_bypass(
                    abs_offset, len,
                    reinterpret_cast<char *>(scratch->buffer.data()))
              : read_bytes(abs_offset, len,
                           reinterpret_cast<char *>(scratch->buffer.data()));
      if (!read_ok) {
        LOG_ERROR(
            "BufferReadStorage::Segment::read: read_range failed, "
            "abs_offset=%zu, len=%zu",
            abs_offset, len);
        *data = nullptr;
        return 0;
      }
      *data = scratch->buffer.data();
      return len;
    }

    //! Read data from segment into a MemoryBlock
    size_t read(size_t offset, MemoryBlock &data, size_t len) override {
      len = clamp_length(&offset, len);
      if (len == 0) {
        data.reset();
        return 0;
      }
      return read_memory_block(data_offset_ + offset, data, len,
                               /*borrow_handle=*/false);
    }

    //! Borrowed read; the caller keeps this Segment alive until block release.
    size_t read_borrowed(size_t offset, MemoryBlock &data,
                         size_t len) override {
      len = clamp_length(&offset, len);
      if (len == 0) {
        data.reset();
        return 0;
      }
      return read_memory_block(data_offset_ + offset, data, len,
                               /*borrow_handle=*/true);
    }

    //! Return independently pinned resident pages without copying. Cold or
    //! partially resident ranges retain the existing contiguous fallback so
    //! cache admission and bypass behavior remain unchanged.
    size_t read_scatter(size_t offset, ScatterBlock &data,
                        size_t len) override {
      data.reset();
      len = clamp_length(&offset, len);
      if (len == 0) {
        return 0;
      }
      if (!cache_enabled_) {
        return IndexStorage::Segment::read_scatter(offset, data, len);
      }

      const size_t abs_offset = data_offset_ + offset;
      const size_t first_page = abs_offset / ailego::kVectorPageSize;
      const size_t last_page =
          (abs_offset + len - 1) / ailego::kVectorPageSize;
      const size_t page_count = last_page - first_page + 1;
      std::vector<ailego::block_id_t> page_ids(page_count);
      std::vector<char *> pages(page_count, nullptr);
      for (size_t i = 0; i < page_count; ++i) {
        page_ids[i] = static_cast<ailego::block_id_t>(first_page + i);
      }
      if (!handle_->try_acquire_resident_pages(page_ids.data(), page_count,
                                               pages.data())) {
        return IndexStorage::Segment::read_scatter(offset, data, len);
      }

      bool pins_owned_by_lease = false;
      try {
        std::vector<ReadSpan> spans;
        spans.reserve(page_count);
        const size_t range_end = abs_offset + len;
        for (size_t i = 0; i < page_count; ++i) {
          const size_t page_begin = (first_page + i) * ailego::kVectorPageSize;
          const size_t span_begin = std::max(abs_offset, page_begin);
          const size_t span_end =
              std::min(range_end, page_begin + ailego::kVectorPageSize);
          spans.push_back(
              {reinterpret_cast<const uint8_t *>(pages[i]) +
                   (span_begin - page_begin),
               span_end - span_begin});
        }
        auto lease =
            std::make_shared<BufferPageLease>(handle_, page_ids);
        pins_owned_by_lease = true;
        data.reset_scattered(std::move(spans), std::move(lease), len);
        return len;
      } catch (const std::bad_alloc &) {
        if (!pins_owned_by_lease) {
          handle_->release_pages(page_ids.data(), page_ids.size());
        }
        LOG_ERROR(
            "BufferReadStorage::Segment::read_scatter allocation failed, "
            "abs_offset=%zu, len=%zu, pages=%zu",
            abs_offset, len, page_count);
        return 0;
      }
    }

    //! Read scattered data (stable until this thread's next pointer read)
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

      auto scratch = thread_scratch();
      scratch->release_pin();
      scratch->buffer.resize(total);
      uint8_t *buf = scratch->buffer.data();
      for (size_t i = 0; i < count; ++i) {
        SegmentData *it = &iovec[i];
        ailego_false_if_false(read_bytes(data_offset_ + it->offset, it->length,
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

    void update_data_crc(uint32_t) override {}

    //! Clone the segment
    IndexStorage::Segment::Pointer clone(void) override {
      return std::make_shared<BufferReadStorage::Segment>(*this);
    }

    void prefetch(size_t offset, size_t len,
                  CachePriority priority = CachePriority::kLow) override {
      if (!cache_enabled_) return;
      len = clamp_length(&offset, len);
      if (len == 0) return;
      handle_->prefetch_range(data_offset_ + offset, len,
                              static_cast<uint8_t>(priority));
    }

    //! Cached pages do not expose a stable base address.
    const uint8_t *base_data(void) const override {
      return nullptr;
    }

   private:
    struct ThreadScratch {
      ThreadScratch(
          const std::shared_ptr<const uint8_t> &owner_arg,
          const std::shared_ptr<ailego::VecBufferPoolHandle> &handle_arg)
          : owner(owner_arg), handle(handle_arg) {}

      ThreadScratch(const ThreadScratch &) = delete;
      ThreadScratch &operator=(const ThreadScratch &) = delete;

      ~ThreadScratch() {
        release_pin();
      }

      void pin(size_t page_id) {
        pinned_page_id = page_id;
        page_pinned = true;
      }

      void release_pin() {
        if (page_pinned) {
          handle->release_one(pinned_page_id);
          page_pinned = false;
        }
      }

      std::weak_ptr<const uint8_t> owner;
      std::vector<uint8_t> buffer;
      std::shared_ptr<ailego::VecBufferPoolHandle> handle;
      size_t pinned_page_id{0};
      bool page_pinned{false};
    };

    static bool &thread_scratch_registry_alive() {
      // bool has a trivial destructor, so it remains safe to inspect while
      // non-trivial thread_local objects are being torn down.
      static thread_local bool alive = false;
      return alive;
    }

    struct ThreadScratchRegistry {
      ThreadScratchRegistry() {
        thread_scratch_registry_alive() = true;
      }

      ~ThreadScratchRegistry() {
        thread_scratch_registry_alive() = false;
      }

      std::unordered_map<const uint8_t *, ThreadScratch> scratches;
      const uint8_t *last_key{nullptr};
      ThreadScratch *last_scratch{nullptr};
    };

    static ThreadScratchRegistry &thread_scratch_registry() {
      static thread_local ThreadScratchRegistry registry;
      return registry;
    }

    void release_thread_scratch() const {
      ThreadScratchRegistry &registry = thread_scratch_registry();
      const uint8_t *key = scratch_token_.get();
      if (registry.last_key == key) {
        registry.last_key = nullptr;
        registry.last_scratch = nullptr;
      }
      registry.scratches.erase(key);
    }

    ThreadScratch *thread_scratch() const {
      // Keep one scratch/pin per (thread, Segment); weak tokens clean up dead
      // segments on long-lived worker threads.
      ThreadScratchRegistry &registry = thread_scratch_registry();
      const uint8_t *key = scratch_token_.get();
      if (registry.last_key == key && registry.last_scratch != nullptr &&
          !registry.last_scratch->owner.expired()) {
        return registry.last_scratch;
      }

      // Only Segment switches scan expired TLS entries.
      registry.last_key = nullptr;
      registry.last_scratch = nullptr;
      for (auto iter = registry.scratches.begin();
           iter != registry.scratches.end();) {
        if (iter->second.owner.expired()) {
          iter = registry.scratches.erase(iter);
        } else {
          ++iter;
        }
      }
      auto result =
          registry.scratches.try_emplace(key, scratch_token_, handle_);
      registry.last_key = key;
      registry.last_scratch = &result.first->second;
      return registry.last_scratch;
    }

    bool read_bytes(size_t abs_offset, size_t len, char *out) const {
      return cache_enabled_ ? handle_->read_range(abs_offset, len, out)
                            : handle_->read_range_bypass(abs_offset, len, out);
    }

    size_t read_memory_block(size_t abs_offset, MemoryBlock &data, size_t len,
                             bool borrow_handle) const {
      const size_t offset_in_page = abs_offset % ailego::kVectorPageSize;
      bool force_bypass = !cache_enabled_;
      if (cache_enabled_ && len <= ailego::kVectorPageSize - offset_in_page) {
        size_t page_id = 0;
        char *raw = handle_->get_single_page(abs_offset, len, page_id);
        if (raw != nullptr) {
          if (borrow_handle) {
            data.reset(handle_.get(), page_id, raw);
          } else {
            data.reset(handle_, page_id, raw);
          }
          return len;
        }
        force_bypass = true;
      }
      // Copy cross-page and bypass reads into owned memory.
      return read_owned(abs_offset, data, len, force_bypass);
    }

    size_t read_owned(size_t abs_offset, MemoryBlock &data, size_t len,
                      bool force_bypass) const {
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
      const bool read_ok =
          force_bypass ? handle_->read_range_bypass(abs_offset, len, tmp)
                       : read_bytes(abs_offset, len, tmp);
      if (!read_ok) {
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
    std::shared_ptr<const uint8_t> scratch_token_{
        std::make_shared<const uint8_t>(0)};
    std::shared_ptr<ailego::VecBufferPoolHandle> handle_{nullptr};
    bool cache_enabled_{false};
  };

  //! Destructor
  ~BufferReadStorage(void) override = default;

  //! Initialize container
  int init(const ailego::Params &params) override {
    params.get(BUFFER_READ_STORAGE_CHECKSUM_VALIDATION, &checksum_validation_);
    params.get(BUFFER_READ_STORAGE_HEADER_OFFSET, &header_offset_);
    params.get(BUFFER_READ_STORAGE_FOOTER_OFFSET, &footer_offset_);
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

  void refresh(uint64_t) override {}

  uint64_t check_point(void) const override {
    return 0;
  }

  //! Cleanup container
  int cleanup(void) override {
    return this->close();
  }

  //! Load an index file into the container
  int open(const std::string &path, bool) override {
    try {
      // Publish new state only after open succeeds.
      auto candidate_pool =
          std::make_shared<ailego::VecBufferPool>(path, /*writable=*/false);
      auto candidate_handle =
          std::make_shared<ailego::VecBufferPoolHandle>(candidate_pool);

      const size_t file_size = candidate_pool->file_size();
      const size_t page_count =
          file_size == 0 ? 0 : (file_size - 1) / ailego::kVectorPageSize + 1;
      const size_t metadata_bytes =
          ailego::VecBufferPool::metadata_bytes_for_page_count(page_count);
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

      // Fall back to bypass-only mode when metadata plus one page cannot fit.
      bool candidate_cache_enabled = false;
      const size_t available =
          ailego::MemoryLimitPool::get_instance().available();
      if (metadata_bytes != std::numeric_limits<size_t>::max() &&
          metadata_bytes <= available &&
          ailego::kVectorPageSize <= available - metadata_bytes) {
        candidate_cache_enabled = candidate_pool->init() == 0;
      }
      if (!candidate_cache_enabled) {
        LOG_INFO(
            "BufferReadStorage opened in bypass-only mode: path=%s "
            "available=%zu metadata=%zu page_size=%zu",
            path.c_str(), available, metadata_bytes, ailego::kVectorPageSize);
      }
      if (candidate_cache_enabled &&
          warmup_mode_ == BUFFER_READ_STORAGE_WARMUP_SEQUENTIAL) {
        candidate_pool->warmup();
      }

      file_path_ = path;
      index_offset_ = candidate_index_offset;
      magic_ = candidate_magic;
      segments_ = std::move(candidate_segments);
      handle_ = std::move(candidate_handle);
      buffer_pool_ = std::move(candidate_pool);
      cache_enabled_ = candidate_cache_enabled;
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
    cache_enabled_ = false;
    return 0;
  }

  //! Retrieve a segment by id
  IndexStorage::Segment::Pointer get(const std::string &id, int) override {
    if (!handle_) {
      return {};
    }
    auto it = segments_.find(id);
    if (it == segments_.end()) {
      return {};
    }
    return std::make_shared<BufferReadStorage::Segment>(
        handle_, cache_enabled_, index_offset_, it->second);
  }

  std::map<std::string, IndexStorage::Segment::Pointer> get_all(
      void) const override {
    std::map<std::string, IndexStorage::Segment::Pointer> result;
    if (handle_) {
      for (const auto &it : segments_) {
        result.emplace(it.first,
                       std::make_shared<BufferReadStorage::Segment>(
                           handle_, cache_enabled_, index_offset_, it.second));
      }
    }
    return result;
  }

  //! Test if a segment exists
  bool has(const std::string &id) const override {
    return segments_.find(id) != segments_.end();
  }

  //! Retrieve magic number of index
  uint32_t magic(void) const override {
    return magic_;
  }

  //! Reads go through the VecBufferPool paged cache.
  MemoryBlock::MemoryBlockType memory_block_type(void) const override {
    return MemoryBlock::MBT_BUFFERPOOL;
  }

  std::shared_ptr<ailego::VecBufferPool> vec_buffer_pool(void) const override {
    return cache_enabled_ ? buffer_pool_ : nullptr;
  }

  //! Path of the opened index file (diagnostics / backend consistency).
  std::string file_path(void) const override {
    return file_path_;
  }

 private:
  bool checksum_validation_{false};
  std::string warmup_mode_{BUFFER_READ_STORAGE_WARMUP_SEQUENTIAL};
  int64_t header_offset_{0};
  int64_t footer_offset_{0};
  size_t index_offset_{0};
  uint32_t magic_{0};
  std::string file_path_{};
  std::map<std::string, IndexUnpacker::SegmentMeta> segments_{};
  std::shared_ptr<ailego::VecBufferPool> buffer_pool_{nullptr};
  std::shared_ptr<ailego::VecBufferPoolHandle> handle_{nullptr};
  bool cache_enabled_{false};
};

INDEX_FACTORY_REGISTER_STORAGE(BufferReadStorage);

}  // namespace core
}  // namespace zvec
