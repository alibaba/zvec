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

#include <algorithm>
#include <atomic>
#include <functional>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <sys/stat.h>
#include <zvec/ailego/buffer/vector_page_table.h>
#include <zvec/ailego/io/file.h>
#include <zvec/ailego/utility/time_helper.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_mapping.h>
#include <zvec/core/framework/index_version.h>
#include "utility_params.h"

namespace zvec {
namespace core {

// Thread-local reusable scratch buffer for cross-page reads in the
// read(const void**) overload.  Avoids allocating a new buffer on
// every cross-page read by reusing the same allocation on each thread.  The
// returned pointer is valid only until the next cross-page read() on
// the same thread -- matching the single-page path's transient
// lifetime (ref released immediately, page may be evicted any time).
struct CrossPageScratch {
  char *buf = nullptr;
  size_t cap = 0;
  ~CrossPageScratch() {
    if (buf) ailego_free(buf);
  }
  char *ensure(size_t len) {
    if (cap < len) {
      if (buf) ailego_free(buf);
      // C11 aligned_alloc requires size to be a multiple of alignment.
      const size_t kAlign = 4096UL;
      size_t alloc_size = (len + (kAlign - 1UL)) & ~(kAlign - 1UL);
      buf = static_cast<char *>(ailego_aligned_malloc(alloc_size, kAlign));
      cap = buf ? alloc_size : 0;
    }
    return buf;
  }
};

/*! Buffer Storage
 */
class BufferStorage : public IndexStorage {
 public:
  /*! Index Storage Segment
   */
  class WrappedSegment : public IndexStorage::Segment,
                         public std::enable_shared_from_this<Segment> {
   public:
    //! Index Storage Pointer
    typedef std::shared_ptr<Segment> Pointer;

    //! Constructor
    //!
    //! `info` MUST be a pointer into BufferStorage::segments_ (an
    //! unordered_map mapped value).  C++ guarantees those pointers stay
    //! valid across insertions, so the WrappedSegment can safely fetch
    //! the LATEST segment_header / segment_header_start_offset / Segment
    //! after a re-parse caused by append_segment().  Storing the pointer
    //! (rather than copying header_/offset into local fields) is what
    //! prevents use-after-free when chain_headers_ is rebuilt.
    WrappedSegment(BufferStorage *owner, IndexMapping::SegmentInfo *info,
                   size_t segment_id)
        : segment_info_(info),
          owner_(owner),
          segment_id_(segment_id),
          capacity_(static_cast<size_t>(info->segment.meta()->data_size +
                                        info->segment.meta()->padding_size)) {}
    //! Destructor
    virtual ~WrappedSegment(void) {}

    //! Retrieve size of data
    size_t data_size(void) const override {
      return static_cast<size_t>(segment_info_->segment.meta()->data_size);
    }

    //! Retrieve crc of data
    uint32_t data_crc(void) const override {
      return segment_info_->segment.meta()->data_crc;
    }

    //! Retrieve size of padding
    size_t padding_size(void) const override {
      return static_cast<size_t>(segment_info_->segment.meta()->padding_size);
    }

    //! Retrieve capacity of segment
    size_t capacity(void) const override {
      return capacity_;
    }

    //! Fetch data from segment (with own buffer)
    //!
    //! C1: pool/handle are stable for the lifetime of the index
    //! (no retire/rebuild), so no lock is needed on the hot path.
    size_t fetch(size_t offset, void *buf, size_t len) const override {
      if (ailego_unlikely(!owner_->buffer_pool_handle_)) {
        LOG_ERROR("WrappedSegment::fetch: handle is null, file[%s], id[%zu]",
                  owner_->file_name_.c_str(), segment_id_);
        return 0;
      }
      const size_t data_size = segment_info_->segment.meta()->data_size;
      if (ailego_unlikely(offset > data_size || len > data_size - offset)) {
        if (offset > data_size) {
          offset = data_size;
        }
        len = data_size - offset;
      }
      size_t abs_offset = segment_info_->segment_header_start_offset +
                          segment_info_->segment_header->content_offset +
                          segment_info_->segment.meta()->data_index + offset;
      if (!owner_->buffer_pool_handle_->read_range(abs_offset, len,
                                                   static_cast<char *>(buf))) {
        return 0;
      }
      return len;
    }

    //! Read data from segment
    //! C1: lock-free hot path (pool/handle never change during operation).
    size_t read(size_t offset, const void **data, size_t len) override {
      if (ailego_unlikely(!owner_->buffer_pool_handle_)) {
        LOG_ERROR("WrappedSegment::read: handle is null, file[%s], id[%zu]",
                  owner_->file_name_.c_str(), segment_id_);
        *data = nullptr;
        return 0;
      }
      const size_t data_size = segment_info_->segment.meta()->data_size;
      if (ailego_unlikely(offset > data_size || len > data_size - offset)) {
        if (offset > data_size) {
          offset = data_size;
        }
        len = data_size - offset;
      }
      size_t abs_offset = segment_info_->segment_header_start_offset +
                          segment_info_->segment_header->content_offset +
                          segment_info_->segment.meta()->data_index + offset;
      size_t first_page = abs_offset / ailego::kVectorPageSize;
      size_t last_page = (len == 0)
                             ? first_page
                             : (abs_offset + len - 1) / ailego::kVectorPageSize;
      if (first_page == last_page) {
        size_t page_id = 0;
        char *raw = owner_->buffer_pool_handle_->get_single_page(abs_offset,
                                                                 len, page_id);
        if (!raw) {
          *data = nullptr;
          return 0;
        }
        *data = raw;
        // Release the buffer-pool ref count acquired by get_single_page().
        // The pointer remains valid as long as the page is not evicted; callers
        // needing a stable pin should use the read(MemoryBlock&) overload.
        owner_->buffer_pool_handle_->release_one(page_id);
        return len;
      }
      // Reuse a thread-local scratch buffer to avoid allocating on
      // every cross-page read.  The pointer is valid until the next
      // cross-page read(const void**) on the same thread.
      thread_local CrossPageScratch scratch;
      char *tmp = scratch.ensure(len);
      if (!tmp) {
        *data = nullptr;
        return 0;
      }
      if (!owner_->buffer_pool_handle_->read_range(abs_offset, len, tmp)) {
        *data = nullptr;
        return 0;
      }
      *data = tmp;
      return len;
    }

    //! C1: lock-free hot path (pool/handle never change during operation).
    size_t read(size_t offset, MemoryBlock &data, size_t len) override {
      if (ailego_unlikely(!owner_->buffer_pool_handle_)) {
        LOG_ERROR(
            "WrappedSegment::read(MemoryBlock&): handle is null, file[%s], "
            "id[%zu]",
            owner_->file_name_.c_str(), segment_id_);
        return 0;
      }
      const size_t data_size = segment_info_->segment.meta()->data_size;
      if (ailego_unlikely(offset > data_size || len > data_size - offset)) {
        if (offset > data_size) {
          offset = data_size;
        }
        len = data_size - offset;
      }
      size_t abs_offset = segment_info_->segment_header_start_offset +
                          segment_info_->segment_header->content_offset +
                          segment_info_->segment.meta()->data_index + offset;
      size_t first_page = abs_offset / ailego::kVectorPageSize;
      size_t last_page = (len == 0)
                             ? first_page
                             : (abs_offset + len - 1) / ailego::kVectorPageSize;
      if (first_page == last_page) {
        size_t page_id = 0;
        char *raw = owner_->buffer_pool_handle_->get_single_page(abs_offset,
                                                                 len, page_id);
        if (!raw) {
          LOG_ERROR("read error (single-page acquire failed).");
          return 0;
        }
        data.reset(owner_->buffer_pool_handle_.get(), page_id, raw);
        return len;
      }
      // C11 aligned_alloc requires the requested size to be a multiple of
      // the alignment; round len up to the next 4K boundary.  Without this
      // glibc treats the call as undefined behaviour and silently corrupts
      // heap metadata (manifesting later as `corrupted size vs. prev_size`).
      const size_t kAlign = 4096UL;
      size_t alloc_size = (len + (kAlign - 1UL)) & ~(kAlign - 1UL);
      char *tmp = static_cast<char *>(ailego_aligned_malloc(alloc_size, kAlign));
      if (!tmp) {
        LOG_ERROR("read error (alloc cross-page temp buffer failed).");
        return 0;
      }
      if (!owner_->buffer_pool_handle_->read_range(abs_offset, len, tmp)) {
        ailego_free(tmp);
        LOG_ERROR("read error (cross-page read_range failed).");
        return 0;
      }
      data = MemoryBlock::MakeOwned(tmp, len);
      return len;
    }

    //! Write data into the storage with offset
    //!
    //! Takes a SHARED latch on the owner's mapping shard.  This pairs with
    //! the EXCLUSIVE all-shards latch held by flush_index() / append_segment()
    //! around the meta_buf CRC + write_meta phase: writers parallelize
    //! across (and within) shards, but are fully excluded while CRC is
    //! computed over the meta_buf bytes that this method mutates
    //! (data_size / padding_size).  Without this latch the lock-free hot
    //! path raced with the CRC compute, producing footer.segments_meta_crc
    //! that did not match the bytes pwrite()'d to disk.
    size_t write(size_t offset, const void *data, size_t len) override {
      std::shared_lock<std::shared_mutex> latch(
          owner_->mapping_shards_[owner_->mapping_shard_id()].mtx);
      if (ailego_unlikely(!owner_->buffer_pool_handle_ ||
                          !owner_->buffer_pool_)) {
        LOG_ERROR("WrappedSegment::write: pool is null, file[%s], id[%zu]",
                  owner_->file_name_.c_str(), segment_id_);
        return 0;
      }
      // In read-only mode the write is a silent no-op so that callers that
      // unconditionally write (e.g. CRC updates) do not return an error.
      if (!owner_->buffer_pool_->writable()) {
        return len;
      }
      if (ailego_unlikely(offset > capacity_ || len > capacity_ - offset)) {
        LOG_ERROR("write() exceeds segment capacity: offset=%zu len=%zu cap=%zu",
                  offset, len, capacity_);
        return 0;
      }
      auto meta = segment_info_->segment.meta();
      if (offset + len > meta->data_size) {
        meta->data_size = offset + len;
        meta->padding_size = capacity_ - meta->data_size;
      }
      size_t abs_offset = segment_info_->segment_header_start_offset +
                          segment_info_->segment_header->content_offset +
                          segment_info_->segment.meta()->data_index + offset;
      if (owner_->buffer_pool_handle_->write_range(
              abs_offset, len, static_cast<const char *>(data)) != 0) {
        LOG_ERROR("write() page-cache write_range failed at abs_offset=%zu",
                  abs_offset);
        return 0;
      }
      // ALWAYS mark dirty after a successful page-cache write so that the
      // next flush_index() does NOT take the `if (!index_dirty_) return 0;`
      // short-circuit and skip flush_all().  Previously this was only set
      // when `data_size` grew, which meant fixed-size segments (e.g.
      // chunk_meta_segment writing HnswChunkMeta in place) never raised
      // the dirty flag -- their 4K page-cache pages were not flushed before
      // append_segment(), so the freshly-rebuilt page table
      // pread'd stale content from disk and chunk_cnts[NODE] lagged the
      // real segment count, eventually causing sync_chunks() to see a
      // mid-state segment and crash with a NULL Chunk::Pointer.
      owner_->set_as_dirty();
      return len;
    }

    //! Resize size of data
    //!
    //! Takes a SHARED latch for the same reason as write(): mutating
    //! meta->data_size / padding_size must be excluded from the CRC
    //! compute in flush_index() / append_segment().
    size_t resize(size_t size) override {
      std::shared_lock<std::shared_mutex> latch(
          owner_->mapping_shards_[owner_->mapping_shard_id()].mtx);
      auto meta = segment_info_->segment.meta();
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
    //!
    //! Takes a SHARED latch for the same reason as write(): mutating
    //! meta->data_crc must be excluded from the CRC compute in
    //! flush_index() / append_segment().
    void update_data_crc(uint32_t crc) override {
      std::shared_lock<std::shared_mutex> latch(
          owner_->mapping_shards_[owner_->mapping_shard_id()].mtx);
      segment_info_->segment.meta()->data_crc = crc;
      owner_->set_as_dirty();
    }

    //! Clone the segment
    IndexStorage::Segment::Pointer clone(void) override {
      return shared_from_this();
    }

   protected:
    friend BufferStorage;
    // Pointer into BufferStorage::segments_ (an unordered_map mapped value).
    // C++ guarantees the address stays valid across map insertions.  All
    // header / start-offset / segment-meta accesses go through this pointer
    // so that re-parses after append_segment() are observed without
    // needing to recreate WrappedSegment instances held by callers.
    IndexMapping::SegmentInfo *segment_info_{nullptr};

   private:
    BufferStorage *owner_{nullptr};
    size_t segment_id_{};
    size_t capacity_{};
  };

  //! Destructor
  virtual ~BufferStorage(void) {
    this->cleanup();
  }

  //! Retrieve the memory block type of this storage
  MemoryBlock::MemoryBlockType memory_block_type(void) const override {
    return MemoryBlock::MBT_BUFFERPOOL;
  }

  //! Initialize storage
  int init(const ailego::Params &params) override {
    uint32_t val = params.get_as_uint32(MMAPFILE_STORAGE_SEGMENT_META_CAPACITY);
    if (val != 0) {
      segment_meta_capacity_ = val;
    }
    return 0;
  }

  //! Cleanup storage
  int cleanup(void) override {
    this->close_index();
    return 0;
  }

  //! Open storage
  int open(const std::string &path, bool create_if_missing) override {
    file_name_ = path;
    if (!ailego::File::IsExist(path) && create_if_missing) {
      size_t last_slash = path.rfind('/');
      if (last_slash != std::string::npos) {
        ailego::File::MakePath(path.substr(0, last_slash));
      }
      int error_code = this->init_index(path);
      if (error_code != 0) {
        LOG_ERROR("init_index failed for %s, errno=%d", path.c_str(),
                  error_code);
        return error_code;
      }
    }

    // Open in writable mode when the caller expects to modify the index
    // (create_if_missing=true implies write intent, same as MMapFileStorage).
    buffer_pool_ = std::make_shared<ailego::VecBufferPool>(
        path, /*writable=*/create_if_missing, /*create=*/false);
    buffer_pool_handle_ = std::make_shared<ailego::VecBufferPoolHandle>(
        buffer_pool_->get_handle());
    int ret = ParseToMapping();
    if (ret != 0) {
      this->close_index();
      return ret;
    }
    ret = buffer_pool_->init();
    if (ret != 0) {
      this->close_index();
      return ret;
    }
    LOG_INFO(
        "BufferStorage opened: file=%s, writable=%d, max_segment_size=%lu, "
        "segment_count=%zu",
        file_name_.c_str(), static_cast<int>(create_if_missing),
        max_segment_size_, segments_.size());
    return 0;
  }

  int ParseHeader(size_t offset, IndexFormat::MetaHeader *out) {
    std::unique_ptr<char[]> buffer(new char[sizeof(*out)]);
    // ParseHeader is called from ParseToMapping which is itself called
    // from either open() (single-threaded) or append_segment() (under
    // AllShardsExclusiveLatch).  Do NOT add an internal lock here --
    // std::shared_mutex is not reentrant -> deadlock.
    if (buffer_pool_handle_->get_meta(offset, sizeof(*out), buffer.get()) !=
        0) {
      LOG_ERROR("Get segment header failed.");
      return IndexError_Runtime;
    }
    memcpy(out, buffer.get(), sizeof(*out));
    if (out->meta_header_size != sizeof(IndexFormat::MetaHeader)) {
      LOG_ERROR("Header meta size is invalid.");
      return IndexError_InvalidLength;
    }
    if (ailego::Crc32c::Hash(out, sizeof(*out), out->header_crc) !=
        out->header_crc) {
      LOG_ERROR("Header meta checksum is invalid.");
      return IndexError_InvalidChecksum;
    }
    return 0;
  }

  int ParseFooter(size_t offset) {
    std::unique_ptr<char[]> buffer(new char[sizeof(footer_)]);
    // Bypass wrapper -- see ParseHeader() comment for why.
    if (buffer_pool_handle_->get_meta(offset, sizeof(footer_), buffer.get()) !=
        0) {
      LOG_ERROR("Get segment footer failed.");
      return IndexError_Runtime;
    }
    uint8_t *footer_ptr = reinterpret_cast<uint8_t *>(buffer.get());
    memcpy(&footer_, footer_ptr, sizeof(footer_));
    if (offset < (size_t)footer_.segments_meta_size) {
      LOG_ERROR("Footer meta size is invalid.");
      return IndexError_InvalidLength;
    }
    if (ailego::Crc32c::Hash(&footer_, sizeof(footer_), footer_.footer_crc) !=
        footer_.footer_crc) {
      LOG_ERROR("Footer meta checksum is invalid.");
      return IndexError_InvalidChecksum;
    }
    return 0;
  }

  int ParseSegment(size_t offset, IndexFormat::MetaHeader *chain_header,
                   uint32_t *out_segment_ids_offset) {
    // NOTE: this function is only called from ParseToMapping(), which is
    // itself called from either open() (single-threaded construction) or
    // append_segment() (under AllShardsExclusiveLatch).  Do NOT add an
    // internal lock here -- doing so would deadlock the append path.
    std::unique_ptr<char[]> segment_buffer =
        std::make_unique<char[]>(footer_.segments_meta_size);
    // Bypass wrapper -- see ParseHeader() comment for why.
    if (buffer_pool_handle_->get_meta(offset, footer_.segments_meta_size,
                                      segment_buffer.get()) != 0) {
      LOG_ERROR("Get segment meta failed.");
      return IndexError_Runtime;
    }
    if (ailego::Crc32c::Hash(segment_buffer.get(), footer_.segments_meta_size,
                             0u) != footer_.segments_meta_crc) {
      LOG_ERROR("Index segments meta checksum is invalid.");
      return IndexError_InvalidChecksum;
    }
    IndexFormat::SegmentMeta *segment_start =
        reinterpret_cast<IndexFormat::SegmentMeta *>(segment_buffer.get());
    uint32_t segment_ids_offset = footer_.segments_meta_size;
    for (IndexFormat::SegmentMeta *iter = segment_start,
                                  *end = segment_start + footer_.segment_count;
         iter != end; ++iter) {
      if (iter->segment_id_offset >= footer_.segments_meta_size) {
        return IndexError_InvalidValue;
      }
      if (iter->data_index > footer_.content_size) {
        return IndexError_InvalidValue;
      }
      if (iter->data_index + iter->data_size > footer_.content_size) {
        return IndexError_InvalidLength;
      }

      if (iter->segment_id_offset < segment_ids_offset) {
        segment_ids_offset = iter->segment_id_offset;
      }
      // Assign a stable numeric ID (block_id in the page table) to this
      // segment.  We use id_hash_.size() rather than segments_.size() because
      // segments_ is intentionally NOT cleared between appends (to keep
      // existing WrappedSegment pointers valid), so segments_.size() would
      // reflect stale entries and produce wrong IDs on re-parse.
      const std::string seg_name(reinterpret_cast<const char *>(segment_start) +
                                 iter->segment_id_offset);
      const size_t seg_id = id_hash_.size();
      id_hash_[seg_name] = seg_id;
      // Update the segments_ entry in-place so that any WrappedSegment
      // instances that already hold a pointer to this entry (via
      // &segments_[name].segment) continue to use the refreshed meta_ptr_
      // after the re-parse.
      //
      // IMPORTANT: chain_header points into chain_headers_ which is a
      // std::vector<std::unique_ptr<MetaHeader>>; each chain owns its OWN
      // MetaHeader copy.  Do NOT use a shared &header_ here -- when there
      // are multiple meta-header chains in the file, the next ParseHeader()
      // would overwrite that single instance and break content_offset for
      // all earlier-chain segments.
      segments_[seg_name] = IndexMapping::SegmentInfo{
          IndexMapping::Segment{iter}, current_header_start_offset_,
          chain_header};
      max_segment_size_ =
          std::max(max_segment_size_, iter->data_size + iter->padding_size);
      if (sizeof(IndexFormat::SegmentMeta) * footer_.segment_count >
          footer_.segments_meta_size) {
        return IndexError_InvalidLength;
      }
    }
    buffer_pool_buffers_.push_back(std::move(segment_buffer));
    if (out_segment_ids_offset) {
      *out_segment_ids_offset = segment_ids_offset;
    }
    return 0;
  }

  int ParseToMapping() {
    while (true) {
      int ret;
      // Allocate an OWN MetaHeader for this chain so that subsequent chains
      // never overwrite earlier-chain headers (prior implementation used a
      // single header_ member, which corrupted content_offset for chain-0
      // segments once chain-1 was parsed).
      chain_headers_.emplace_back(
          std::make_unique<IndexFormat::MetaHeader>());
      IndexFormat::MetaHeader *chain_header = chain_headers_.back().get();
      ret = ParseHeader(current_header_start_offset_, chain_header);
      if (ret != 0) {
        LOG_ERROR("Failed to parse header, errno %d, %s", ret,
                  IndexError::What(ret));
        return ret;
      }

      switch (chain_header->version) {
        case IndexFormat::FORMAT_VERSION:
          break;
        default:
          LOG_ERROR("Unsupported index version: %u", chain_header->version);
          return IndexError_Unsupported;
      }

      // Unpack footer
      if (chain_header->meta_footer_size != sizeof(IndexFormat::MetaFooter)) {
        return IndexError_InvalidLength;
      }
      if ((int32_t)chain_header->meta_footer_offset < 0) {
        return IndexError_Unsupported;
      }
      uint64_t footer_offset =
          chain_header->meta_footer_offset + current_header_start_offset_;
      ret = ParseFooter(footer_offset);
      if (ret != 0) {
        LOG_ERROR("Failed to parse footer, errno %d, %s", ret,
                  IndexError::What(ret));
        return ret;
      }

      // Unpack segment table
      if (sizeof(IndexFormat::SegmentMeta) * footer_.segment_count >
          footer_.segments_meta_size) {
        return IndexError_InvalidLength;
      }
      const uint64_t segment_start_offset =
          footer_offset - footer_.segments_meta_size;
      uint32_t segment_ids_offset = footer_.segments_meta_size;
      ret = ParseSegment(segment_start_offset, chain_header,
                         &segment_ids_offset);
      if (ret != 0) {
        LOG_ERROR("Failed to parse segment, errno %d, %s", ret,
                  IndexError::What(ret));
        return ret;
      }

      // Record per-chain metadata offsets so flush_index() can write
      // updated segment metas and footers back to the backing file.
      meta_chains_.push_back({current_header_start_offset_, footer_offset,
                              segment_start_offset,
                              footer_.segments_meta_size,
                              segment_ids_offset, footer_});

      if (footer_.next_meta_header_offset == 0) {
        break;
      }
      current_header_start_offset_ = footer_.next_meta_header_offset;
    }
    return 0;
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
    return footer_.check_point;
  }

  //! Retrieve a segment by id
  IndexStorage::Segment::Pointer get(const std::string &id, int) override {
    std::shared_lock<std::shared_mutex> latch(
        mapping_shards_[mapping_shard_id()].mtx);
    auto seg_iter = segments_.find(id);
    if (seg_iter == segments_.end()) {
      return WrappedSegment::Pointer{};
    }
    auto id_iter = id_hash_.find(id);
    if (id_iter == id_hash_.end()) {
      return WrappedSegment::Pointer{};
    }
    return std::make_shared<WrappedSegment>(this, &seg_iter->second,
                                            id_iter->second);
  }

  //! Test if it a segment exists
  bool has(const std::string &id) const override {
    return this->has_segment(id);
  }

  //! Retrieve magic number of index
  uint32_t magic(void) const override {
    if (chain_headers_.empty()) {
      return 0u;
    }
    return chain_headers_.front()->magic;
  }

 protected:
  //! Initialize index version segment (writes content into an IndexMapping).
  //! Only intended to be called from init_index() while `mapping` is still
  //! open in create-mode.
  int init_version_segment(IndexMapping &mapping) {
    size_t data_size = std::strlen(IndexVersion::Details());
    int error_code = mapping.append(INDEX_VERSION_SEGMENT_NAME, data_size);
    if (error_code != 0) {
      return error_code;
    }
    IndexMapping::Segment *segment =
        mapping.map(INDEX_VERSION_SEGMENT_NAME, false, false);
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

  //! Create the initial on-disk index structure and write the mandatory
  //! version segment.  Uses IndexMapping (the same engine as MMapFileStorage)
  //! so the produced file is fully compatible with both storage backends.
  int init_index(const std::string &path) {
    IndexMapping mapping;
    int ret = mapping.create(path, segment_meta_capacity_);
    if (ret != 0) {
      LOG_ERROR(
          "BufferStorage failed to create index file: path[%s], errno[%d]",
          path.c_str(), ret);
      return ret;
    }
    ret = this->init_version_segment(mapping);
    if (ret != 0) {
      LOG_ERROR(
          "BufferStorage failed to append version segment: path[%s], errno[%d]",
          path.c_str(), ret);
      mapping.close();
      return ret;
    }
    mapping.refresh(0);
    ret = mapping.flush();
    mapping.close();
    if (ret != 0) {
      LOG_ERROR(
          "BufferStorage failed to flush new index file: path[%s], errno[%d]",
          path.c_str(), ret);
    }
    return ret;
  }

  //! Set the index file as dirty.
  //!
  //! HOT PATH: called once per WrappedSegment::write() / resize() /
  //! update_data_crc().  We MUST unconditionally store(true) here, not
  //! guard with a load-then-store: under relaxed semantics a writer can
  //! observe a stale dirty=true (its own core's cached value) AFTER
  //! flush_index() has CAS'd dirty to false on another core, then skip
  //! its own store and the writer's modification gets dropped (next
  //! flush_index() short-circuits at the top because dirty is false).
  //! The MESI ping-pong is the cost of correctness; it is bounded by the
  //! caller's write rate and amortized by the caller's actual I/O.
  void set_as_dirty(void) {
    index_dirty_.store(true, std::memory_order_relaxed);
  }

  //! Refresh meta information (checksum, update time, etc.)
  void refresh_index(uint64_t chkp) {
    // Monotonic merge: callers may invoke refresh() out of order under
    // concurrency (parallel writers, retries, batched commits delivered on
    // different threads).  An unconditional store would let a smaller chkp
    // arriving later overwrite a larger one, violating the upper-layer
    // invariant that the persisted check_point is non-decreasing.  CAS-loop
    // max guarantees the largest observed value wins regardless of arrival
    // order; relaxed ordering is sufficient because flush_index() takes the
    // all-shards exclusive latch which establishes the necessary
    // happens-before for the actual disk write.
    if (chkp != 0) {
      uint64_t cur = pending_check_point_.load(std::memory_order_relaxed);
      while (chkp > cur) {
        if (pending_check_point_.compare_exchange_weak(
                cur, chkp, std::memory_order_relaxed)) {
          break;
        }
        // compare_exchange_weak refreshed `cur`; loop checks chkp > cur
        // again and exits if some other thread already raised pending past
        // our value.
      }
    }
    // In BufferStorage the segment metadata lives in buffer_pool_buffers_.
    // CRC recomputation and disk write are deferred to flush_index().
    // Mark dirty unconditionally for the same reason as set_as_dirty():
    // a load-then-store guard would let a stale `true` observation skip
    // the store and lose this refresh.  Note: even when our chkp lost the
    // CAS race (was discarded as stale), we still set dirty -- the winning
    // larger chkp must be flushed, and flush_index()'s UpdateMetaFooter()
    // is a no-op for chkp==0 so a spurious extra flush is harmless.
    index_dirty_.store(true, std::memory_order_relaxed);
  }

  //! Flush index storage: persists any pending meta changes (segments_meta +
  //! footer) for each header chain, then asks the page cache to write back
  //! dirty data pages.
  int flush_index(void) {
    if (!index_dirty_.load(std::memory_order_relaxed)) {
      return 0;
    }
    // EXCLUSIVE all-shards latch: blocks the lock-free hot path
    // (WrappedSegment::write / resize / update_data_crc) which mutates
    // meta->data_size / padding_size / data_crc, the very bytes we hash
    // to recompute footer.segments_meta_crc and pwrite to disk.  Holding
    // a single shard's shared lock (the previous design) was insufficient
    // because writers on other shards could race with the CRC compute
    // and produce a checksum that mismatches the on-disk segment_meta
    // bytes, causing IndexError_InvalidChecksum on the next open().
    AllShardsExclusiveLatch latch(mapping_shards_);
    return flush_index_locked();
  }

  //! Internal flush implementation. PRECONDITION: caller MUST already hold
  //! AllShardsExclusiveLatch on mapping_shards_.  Used by flush_index()
  //! (which acquires the latch itself) and by close_index() (which must
  //! flush and tear down under a SINGLE continuous latch hold so that no
  //! writer can slip in between flush and pool reset and lose its dirty
  //! pages).
  int flush_index_locked(void) {
    // NULL GUARD: a previous append_segment() may have left the pool in a
    // torn-down state.
    if (!buffer_pool_ || !buffer_pool_handle_) {
      LOG_ERROR("BufferStorage::flush_index skipped: pool not ready, file[%s]",
                file_name_.c_str());
      return IndexError_Runtime;
    }
    if (!buffer_pool_->writable()) {
      // Read-only pool: nothing to flush.
      index_dirty_.store(false, std::memory_order_relaxed);
      return 0;
    }
    // Atomically claim the dirty flag at the START of the flush, not at the
    // end.  This prevents a TOCTOU race against the lock-free hot path:
    // any WrappedSegment::write() that happens between flush_all() and the
    // end of this function will simply re-set dirty=true (its set_as_dirty
    // observes our cleared flag), and the next flush_index() will pick up
    // those new dirty pages.  An unconditional store(false) at the end
    // would silently swallow that concurrent write.
    bool expected_dirty = true;
    if (!index_dirty_.compare_exchange_strong(expected_dirty, false,
                                              std::memory_order_relaxed)) {
      // Another thread already claimed and is performing the flush; treat
      // this call as a no-op.  The previous design (no CAS) allowed
      // duplicate concurrent flushers; bailing out here is strictly safer
      // because both flushers would otherwise race on per-chain footer
      // mutation in the loop below.
      return 0;
    }
    // Snapshot the pending checkpoint AFTER claiming dirty so that we
    // observe at least every refresh_index() that happened before we
    // claimed.  The CAS-reset at the end will preserve any newer chkp
    // stored by a concurrent refresh_index() during this flush.
    const uint64_t consumed_chkp =
        pending_check_point_.load(std::memory_order_relaxed);
    // Flush all dirty data blocks to the backing file first.
    if (buffer_pool_handle_->flush_all() != 0) {
      // Restore dirty so the next flush_index() retries.
      index_dirty_.store(true, std::memory_order_relaxed);
      LOG_ERROR("flush_all data blocks failed: file[%s]", file_name_.c_str());
      return IndexError_WriteData;
    }
    // For each metadata chain, recompute the segment-meta CRC, update the
    // in-memory footer (segments_meta_crc + footer_crc + update_time), and
    // write both the segment metadata and the footer back to the backing
    // file.  Uses the per-chain in-memory footer copy, avoiding a pread.
    for (size_t ci = 0;
         ci < meta_chains_.size() && ci < buffer_pool_buffers_.size(); ++ci) {
      MetaChain &mchain = meta_chains_[ci];
      const char *seg_buf = buffer_pool_buffers_[ci].get();
      // Recompute segment metadata CRC and refresh the per-chain footer.
      mchain.footer.segments_meta_crc =
          ailego::Crc32c::Hash(seg_buf, mchain.segment_meta_size, 0u);
      IndexFormat::UpdateMetaFooter(&mchain.footer, consumed_chkp);
      // Write segment metadata back to disk.
      if (buffer_pool_handle_->write_meta(mchain.segment_meta_file_offset,
                                          mchain.segment_meta_size,
                                          seg_buf) != 0) {
        LOG_ERROR("Failed to write segment meta: file[%s], chain[%zu]",
                  file_name_.c_str(), ci);
        index_dirty_.store(true, std::memory_order_relaxed);
        return IndexError_WriteData;
      }
      // Write the updated footer back to disk.
      if (buffer_pool_handle_->write_meta(
              mchain.footer_file_offset, sizeof(mchain.footer),
              reinterpret_cast<const char *>(&mchain.footer)) != 0) {
        LOG_ERROR("Failed to write footer: file[%s], chain[%zu]",
                  file_name_.c_str(), ci);
        index_dirty_.store(true, std::memory_order_relaxed);
        return IndexError_WriteData;
      }
    }
    // Keep the convenience alias in sync with the last chain.
    if (!meta_chains_.empty()) {
      footer_ = meta_chains_.back().footer;
    }
    // CAS-reset pending: only consume the checkpoint we observed at the
    // start.  If a concurrent refresh_index() stored a newer value during
    // the flush, CAS fails and the newer value remains in
    // pending_check_point_; refresh_index() also re-set dirty=true (since
    // we cleared it at the top), so the next flush_index() will persist
    // the newer chkp.
    uint64_t expected_chkp = consumed_chkp;
    pending_check_point_.compare_exchange_strong(expected_chkp, 0,
                                                 std::memory_order_relaxed);
    return 0;
  }

  //! Close index storage
  void close_index(void) {
    // Take the all-shards exclusive latch BEFORE flushing, and hold it for
    // the entire teardown sequence.  Earlier code released the latch
    // between flush and teardown, opening a window in which a writer could
    // grab a shared lock, mutate meta_buf via WrappedSegment::write() and
    // call set_as_dirty(true).  After this close_index() reacquired the
    // latch and reset buffer_pool_handle_, those dirty pages would be
    // dropped on the floor with no chance to flush.  Holding a SINGLE
    // latch instance across flush_index_locked() and the reset eliminates
    // that window: writers can only enter once we have fully torn down
    // (and at that point segments_/buffer_pool_handle_ are gone, so they
    // would fail the null/state guards in WrappedSegment).
    AllShardsExclusiveLatch latch(mapping_shards_);
    flush_index_locked();
    file_name_.clear();
    id_hash_.clear();
    segments_.clear();
    chain_headers_.clear();
    memset(&footer_, 0, sizeof(footer_));
    buffer_pool_handle_.reset();
    buffer_pool_.reset();
    max_segment_size_ = 0;
    buffer_pool_buffers_.clear();
    meta_chains_.clear();
    current_header_start_offset_ = 0;
    pending_check_point_.store(0, std::memory_order_relaxed);
    index_dirty_.store(false, std::memory_order_relaxed);
  }

  //! Append a segment into storage.
  //!
  //! C1: the page table extends in-place (no pool rotation).  The exclusive
  //! latch is held only briefly to protect segments_/id_hash_ insertion.
  int append_segment(const std::string &id, size_t size) {
    // Flush any in-memory metadata changes (data_size, padding_size, CRC)
    // accumulated by prior write()/resize() calls.
    this->flush_index();

    AllShardsExclusiveLatch latch(mapping_shards_);

    if (!buffer_pool_ || !buffer_pool_handle_) {
      LOG_ERROR("append_segment: pool not ready, file[%s]",
                file_name_.c_str());
      return IndexError_Runtime;
    }
    if (!buffer_pool_->writable()) {
      LOG_ERROR("append_segment: pool is read-only, file[%s]",
                file_name_.c_str());
      return IndexError_Runtime;
    }
    if (size == 0) {
      return IndexError_InvalidArgument;
    }
    if (segments_.find(id) != segments_.end()) {
      return IndexError_Duplicate;
    }
    if (meta_chains_.empty() || chain_headers_.empty() ||
        buffer_pool_buffers_.empty()) {
      LOG_ERROR("append_segment: invalid state, file[%s]",
                file_name_.c_str());
      return IndexError_Runtime;
    }

    // Page-aligned padded size for the new segment.  Matches IndexMapping's
    // CalcPageAlignedSize() so the on-disk layout stays identical.
    const size_t page_size = ailego::kVectorPageSize;
    const size_t padded_size = (size + page_size - 1) / page_size * page_size;

    // The "current last chain" is meta_chains_.back() / chain_headers_.back();
    // footer_ is always the last chain's footer (overwritten by ParseFooter
    // during ParseToMapping).
    size_t id_size = id.length() + 1;
    size_t need_size = sizeof(IndexFormat::SegmentMeta) + id_size;
    MetaChain *chain = &meta_chains_.back();
    IndexFormat::MetaHeader *header = chain_headers_.back().get();
    char *meta_buf = buffer_pool_buffers_.back().get();

    // Rollback handle for the (possibly committed) chain split below.
    // Default is a no-op; populated ONLY after Step 1's in-memory commit
    // succeeds so that a Step 2 disk-write failure can undo the split as
    // well, leaving meta_chains_ / chain_headers_ / buffer_pool_buffers_ /
    // footer_ / current_header_start_offset_ exactly as they were before
    // append_segment() ran.  Without this, a Step 2 failure would leave
    // an orphan empty chain permanently appended to the file (harmless
    // for correctness because it stays linked and gets reused on next
    // append, but disruptive for idempotent retries and unit tests).
    std::function<void()> rollback_step1 = []() {};

    // ---- Step 1: chain split if current chain has no meta capacity left.
    if (sizeof(IndexFormat::SegmentMeta) * footer_.segment_count + need_size >
        chain->segment_ids_offset) {
      size_t new_chain_start = buffer_pool_->file_size();
      new_chain_start =
          (new_chain_start + page_size - 1) / page_size * page_size;
      size_t new_meta_total =
          (segment_meta_capacity_ + sizeof(IndexFormat::MetaHeader) +
           sizeof(IndexFormat::MetaFooter) + page_size - 1) /
          page_size * page_size;
      uint32_t new_segments_meta_size = static_cast<uint32_t>(
          new_meta_total - sizeof(IndexFormat::MetaHeader) -
          sizeof(IndexFormat::MetaFooter));

      // Prepare the linked old footer WITHOUT mutating footer_ yet so
      // that a write failure leaves in-memory state untouched.
      const auto saved_footer_before_split = footer_;
      IndexFormat::MetaFooter linked_footer = footer_;
      linked_footer.next_meta_header_offset = new_chain_start;
      IndexFormat::UpdateMetaFooter(&linked_footer, 0);

      // Write old footer with forward link to disk.
      if (buffer_pool_handle_->write_meta(
              chain->footer_file_offset, sizeof(linked_footer),
              reinterpret_cast<const char *>(&linked_footer)) != 0) {
        LOG_ERROR("append_segment: write old footer failed, file[%s]",
                  file_name_.c_str());
        return IndexError_WriteData;
      }

      // Best-effort rollback: restore original old footer on disk if a
      // subsequent disk write in this split block fails.
      auto undo_old_footer = [&]() {
        buffer_pool_handle_->write_meta(
            chain->footer_file_offset, sizeof(saved_footer_before_split),
            reinterpret_cast<const char *>(&saved_footer_before_split));
      };

      // Extend the file and write the new chain's header + (zero) footer.
      // The segment_meta region is implicitly zero-filled by ftruncate,
      // matching the empty `new_meta_buf` we keep in memory.
      if (!buffer_pool_->extend_file(new_chain_start + new_meta_total)) {
        undo_old_footer();
        return IndexError_Runtime;
      }

      auto new_header = std::make_unique<IndexFormat::MetaHeader>();
      IndexFormat::SetupMetaHeader(
          new_header.get(),
          static_cast<uint32_t>(new_meta_total -
                                sizeof(IndexFormat::MetaFooter)),
          static_cast<uint32_t>(new_meta_total));

      auto new_meta_buf = std::make_unique<char[]>(new_segments_meta_size);
      std::memset(new_meta_buf.get(), 0, new_segments_meta_size);

      IndexFormat::MetaFooter new_footer;
      IndexFormat::SetupMetaFooter(&new_footer);
      new_footer.segments_meta_size = new_segments_meta_size;
      new_footer.total_size = new_meta_total;
      new_footer.segments_meta_crc = ailego::Crc32c::Hash(
          new_meta_buf.get(), new_segments_meta_size, 0u);
      IndexFormat::UpdateMetaFooter(&new_footer, 0);

      if (buffer_pool_handle_->write_meta(
              new_chain_start, sizeof(IndexFormat::MetaHeader),
              reinterpret_cast<const char *>(new_header.get())) != 0) {
        undo_old_footer();
        return IndexError_WriteData;
      }
      uint64_t new_segment_meta_file_offset =
          new_chain_start + sizeof(IndexFormat::MetaHeader);
      uint64_t new_footer_file_offset =
          new_chain_start + new_header->meta_footer_offset;
      if (buffer_pool_handle_->write_meta(
              new_footer_file_offset, sizeof(new_footer),
              reinterpret_cast<const char *>(&new_footer)) != 0) {
        undo_old_footer();
        return IndexError_WriteData;
      }

      // Snapshot the OLD chain's pre-commit state for rollback_step1.
      // Captured by value because `chain` will be reassigned below to point
      // at the new chain's slot in meta_chains_, and pop_back() during
      // rollback would invalidate any reference into the old slot.
      const auto saved_old_chain_footer = chain->footer;
      const uint64_t saved_old_footer_file_offset = chain->footer_file_offset;
      const uint64_t saved_current_header_start = current_header_start_offset_;

      // All split disk writes succeeded -- commit in-memory state.
      chain->footer = linked_footer;  // old chain keeps linked footer
      chain_headers_.push_back(std::move(new_header));
      buffer_pool_buffers_.push_back(std::move(new_meta_buf));
      meta_chains_.push_back(MetaChain{new_chain_start, new_footer_file_offset,
                                       new_segment_meta_file_offset,
                                       new_segments_meta_size,
                                       new_segments_meta_size, new_footer});
      footer_ = new_footer;
      current_header_start_offset_ = new_chain_start;

      chain = &meta_chains_.back();
      header = chain_headers_.back().get();
      meta_buf = buffer_pool_buffers_.back().get();

      // Install rollback for the committed split: pop the new chain and
      // restore the old chain on both disk and memory.  Captured fully by
      // value (except `this`-via-member-access) so a subsequent reassignment
      // of local pointers (chain/header/meta_buf) does not corrupt the
      // closure.
      rollback_step1 = [this, saved_footer_before_split,
                        saved_old_chain_footer, saved_old_footer_file_offset,
                        saved_current_header_start]() {
        // 1. Restore old chain's footer on disk (drop forward link).
        buffer_pool_handle_->write_meta(
            saved_old_footer_file_offset, sizeof(saved_footer_before_split),
            reinterpret_cast<const char *>(&saved_footer_before_split));
        // 2. Pop the freshly-pushed new chain from in-memory containers.
        //    The associated unique_ptr<MetaHeader> / unique_ptr<char[]>
        //    are released here.
        if (!meta_chains_.empty()) meta_chains_.pop_back();
        if (!chain_headers_.empty()) chain_headers_.pop_back();
        if (!buffer_pool_buffers_.empty()) buffer_pool_buffers_.pop_back();
        // 3. Restore old chain's in-memory footer (its forward link was
        //    set to the now-popped new chain).
        if (!meta_chains_.empty()) {
          meta_chains_.back().footer = saved_old_chain_footer;
        }
        // 4. Restore footer_ and current_header_start_offset_ to their
        //    pre-split values.  The on-disk file size is intentionally NOT
        //    shrunk: most buffer-pool backends offer no precise truncate,
        //    and the leftover bytes (the orphan new_header / new_footer
        //    region) are unreachable -- step 1 above has already removed
        //    the forward link from the old footer, so ParseToMapping()
        //    stops at the old chain and the leftover region is reusable
        //    by the next append_segment()'s split via file_size()
        //    realignment.
        footer_ = saved_footer_before_split;
        current_header_start_offset_ = saved_current_header_start;
      };
    }

    // ---- Step 2: append SegmentMeta + ID into the (possibly new) last
    //              chain, then persist meta_buf and footer.
    uint64_t new_data_index = footer_.content_size;
    uint64_t new_seg_abs_offset =
        chain->header_start_offset + header->content_offset + new_data_index;
    uint64_t new_file_size = new_seg_abs_offset + padded_size;
    if (new_file_size > buffer_pool_->file_size()) {
      if (!buffer_pool_->extend_file(new_file_size)) {
        return IndexError_Runtime;
      }
    }

    // Save mutable state for rollback if a disk write fails below.
    const auto saved_footer = footer_;
    const auto saved_chain_footer = chain->footer;
    const auto saved_segment_ids_offset = chain->segment_ids_offset;
    // Save the meta_buf regions that will be overwritten (SegmentMeta
    // entry and segment-ID string) so they can be restored exactly,
    // keeping the CRC consistent for a potential later flush_index().
    const size_t meta_entry_off =
        sizeof(IndexFormat::SegmentMeta) * footer_.segment_count;
    const uint32_t new_ids_off =
        chain->segment_ids_offset - static_cast<uint32_t>(id_size);
    char saved_meta_entry[sizeof(IndexFormat::SegmentMeta)];
    std::memcpy(saved_meta_entry, meta_buf + meta_entry_off,
                sizeof(IndexFormat::SegmentMeta));
    std::unique_ptr<char[]> saved_id_bytes(new char[id_size]);
    std::memcpy(saved_id_bytes.get(), meta_buf + new_ids_off, id_size);

    chain->segment_ids_offset -= static_cast<uint32_t>(id_size);
    IndexFormat::SegmentMeta *new_seg =
        reinterpret_cast<IndexFormat::SegmentMeta *>(meta_buf) +
        footer_.segment_count;
    new_seg->segment_id_offset = chain->segment_ids_offset;
    new_seg->data_index = new_data_index;
    new_seg->data_size = 0;
    new_seg->data_crc = 0;
    new_seg->padding_size = padded_size;
    std::memcpy(meta_buf + chain->segment_ids_offset, id.c_str(), id_size);

    footer_.segment_count += 1;
    footer_.content_size += padded_size;
    footer_.total_size += padded_size;
    footer_.segments_meta_crc =
        ailego::Crc32c::Hash(meta_buf, chain->segment_meta_size, 0u);
    IndexFormat::UpdateMetaFooter(&footer_, 0);
    chain->footer = footer_;  // sync in-memory copy for flush_index

    // Rollback helper: restore meta_buf, footer_, and chain fields to
    // their pre-Step-2 values so that flush_index() writes consistent
    // metadata and the next append_segment() can retry cleanly.
    auto rollback_step2 = [&]() {
      std::memcpy(meta_buf + meta_entry_off, saved_meta_entry,
                  sizeof(IndexFormat::SegmentMeta));
      std::memcpy(meta_buf + new_ids_off, saved_id_bytes.get(), id_size);
      footer_ = saved_footer;
      chain->footer = saved_chain_footer;
      chain->segment_ids_offset = saved_segment_ids_offset;
    };

    if (buffer_pool_handle_->write_meta(chain->segment_meta_file_offset,
                                        chain->segment_meta_size,
                                        meta_buf) != 0) {
      LOG_ERROR("append_segment: write segment_meta failed, file[%s]",
                file_name_.c_str());
      rollback_step2();
      rollback_step1();
      return IndexError_WriteData;
    }
    if (buffer_pool_handle_->write_meta(
            chain->footer_file_offset, sizeof(footer_),
            reinterpret_cast<const char *>(&footer_)) != 0) {
      LOG_ERROR("append_segment: write footer failed, file[%s]",
                file_name_.c_str());
      rollback_step2();
      rollback_step1();
      return IndexError_WriteData;
    }

    // All disk writes succeeded -- commit remaining in-memory state.
    // WrappedSegment instances already held by callers reference
    // &segments_[name], whose address is stable across unordered_map
    // insertions, so existing references stay valid.
    segments_[id] = IndexMapping::SegmentInfo{
        IndexMapping::Segment{new_seg}, chain->header_start_offset, header};
    const size_t new_id = id_hash_.size();
    id_hash_[id] = new_id;
    max_segment_size_ = std::max<uint64_t>(max_segment_size_, padded_size);

    // ---- Step 3: With the segmented page table (C1), extend_file()
    //              already extended the page table in-place.  No pool
    //              rotation or flush_all is needed — the same pool/handle
    //              continues to serve both old and new pages.
    return 0;
  }

  //! Test if a segment exists
  bool has_segment(const std::string &id) const {
    std::shared_lock<std::shared_mutex> latch(
        mapping_shards_[mapping_shard_id()].mtx);
    return (segments_.find(id) != segments_.end());
  }

 private:
  std::atomic<bool> index_dirty_{false};
  std::atomic<uint64_t> pending_check_point_{0};

  // Sharded reader-writer lock to eliminate cache-line ping-pong on the
  // reader counter.  Each concurrent reader hashes to its own shard,
  // avoiding cross-core contention.  Writers (append_segment/close_index)
  // lock ALL shards to achieve exclusive access.
  static constexpr size_t kMappingMutexShards = 32;
  struct alignas(64) MutexShard {
    std::shared_mutex mtx;
  };
  mutable MutexShard mapping_shards_[kMappingMutexShards]{};

  // Per-thread shard selection (stable hash, no syscall).
  size_t mapping_shard_id() const {
    thread_local const size_t id =
        std::hash<std::thread::id>()(std::this_thread::get_id()) %
        kMappingMutexShards;
    return id;
  }

  // RAII guard that locks ALL shards exclusively (for writers).
  struct AllShardsExclusiveLatch {
    MutexShard *shards_;
    AllShardsExclusiveLatch(MutexShard *shards) : shards_(shards) {
      for (size_t i = 0; i < kMappingMutexShards; ++i) shards_[i].mtx.lock();
    }
    ~AllShardsExclusiveLatch() {
      for (size_t i = 0; i < kMappingMutexShards; ++i) shards_[i].mtx.unlock();
    }
    AllShardsExclusiveLatch(const AllShardsExclusiveLatch &) = delete;
    AllShardsExclusiveLatch &operator=(const AllShardsExclusiveLatch &) = delete;
  };

  // buffer manager
  std::string file_name_;
  // Per-chain owning copies of MetaHeader.  segments_[name].segment_header
  // points into one of these, so each chain's content_offset stays stable
  // across re-parses (a single shared header_ would be overwritten by the
  // next chain's ParseHeader and corrupt earlier-chain segment reads).
  std::vector<std::unique_ptr<IndexFormat::MetaHeader>> chain_headers_{};
  IndexFormat::MetaFooter footer_{};
  std::unordered_map<std::string, IndexMapping::SegmentInfo> segments_{};
  std::unordered_map<std::string, size_t> id_hash_{};
  uint64_t max_segment_size_{0};
  std::vector<std::unique_ptr<char[]>> buffer_pool_buffers_{};

  ailego::VecBufferPool::Pointer buffer_pool_{nullptr};
  ailego::VecBufferPoolHandle::Pointer buffer_pool_handle_{nullptr};
  uint64_t current_header_start_offset_{0u};

  // Capacity (in bytes) of the segment metadata section written by
  // init_index().
  uint32_t segment_meta_capacity_{4096u};

  // Per-header-chain file offsets used by flush_index() to write updated
  // segment metadata and footer back to the backing file after writes.
  struct MetaChain {
    uint64_t header_start_offset;
    uint64_t footer_file_offset;
    uint64_t segment_meta_file_offset;
    uint32_t segment_meta_size;
    // Lowest offset of segment ID strings within the segment_meta region.
    // Equals segment_meta_size when no IDs have been written yet, and
    // decreases by `strlen(id)+1` for each appended segment.  Used by
    // append_segment() to detect when the chain runs out of meta capacity
    // and a new chain must be split off.
    uint32_t segment_ids_offset;
    // In-memory copy of this chain's MetaFooter.  Kept in sync with disk
    // by flush_index() and append_segment(), avoiding a pread per chain
    // on every flush.
    IndexFormat::MetaFooter footer;
  };
  std::vector<MetaChain> meta_chains_{};
};

INDEX_FACTORY_REGISTER_STORAGE(BufferStorage);

}  // namespace core
}  // namespace zvec
