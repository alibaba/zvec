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
#include <mutex>
#include <shared_mutex>
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

/*! MMap File Storage
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
    WrappedSegment(BufferStorage *owner, IndexMapping::Segment *segment,
                   uint64_t segment_header_start_offset,
                   IndexFormat::MetaHeader *segment_header, size_t segment_id)
        : segment_(segment),
          owner_(owner),
          segment_id_(segment_id),
          capacity_(static_cast<size_t>(segment->meta()->data_size +
                                        segment->meta()->padding_size)),
          segment_header_start_offset_(segment_header_start_offset),
          segment_header_(segment_header) {}
    //! Destructor
    virtual ~WrappedSegment(void) {}

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
    //!
    //! LOCKING: takes a shared_lock on owner_->mapping_mutex_ so that
    //! append_segment() / close_index() cannot tear down the pool mid-call.
    size_t fetch(size_t offset, void *buf, size_t len) const override {
      std::shared_lock<std::shared_mutex> latch(owner_->mapping_mutex_);
      if (ailego_unlikely(!owner_->buffer_pool_handle_)) {
        LOG_ERROR("WrappedSegment::fetch: handle is null, file[%s], id[%zu]",
                  owner_->file_name_.c_str(), segment_id_);
        return 0;
      }
      if (ailego_unlikely(offset + len > segment_->meta()->data_size)) {
        auto meta = segment_->meta();
        if (offset > meta->data_size) {
          offset = meta->data_size;
        }
        len = meta->data_size - offset;
      }
      size_t abs_offset = segment_header_start_offset_ +
                          segment_header_->content_offset +
                          segment_->meta()->data_index + offset;
      if (!owner_->buffer_pool_handle_->read_range(abs_offset, len,
                                                   static_cast<char *>(buf))) {
        return 0;
      }
      return len;
    }

    //! Read data from segment
    //! LOCKING: see fetch() above for rationale.
    size_t read(size_t offset, const void **data, size_t len) override {
      std::shared_lock<std::shared_mutex> latch(owner_->mapping_mutex_);
      if (ailego_unlikely(!owner_->buffer_pool_handle_)) {
        LOG_ERROR("WrappedSegment::read: handle is null, file[%s], id[%zu]",
                  owner_->file_name_.c_str(), segment_id_);
        *data = nullptr;
        return 0;
      }
      if (ailego_unlikely(offset + len > segment_->meta()->data_size)) {
        auto meta = segment_->meta();
        if (offset > meta->data_size) {
          offset = meta->data_size;
        }
        len = meta->data_size - offset;
      }
      size_t abs_offset = segment_header_start_offset_ +
                          segment_header_->content_offset +
                          segment_->meta()->data_index + offset;
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
      char *tmp = static_cast<char *>(ailego_aligned_malloc(len, 4096));
      if (!tmp) {
        *data = nullptr;
        return 0;
      }
      if (!owner_->buffer_pool_handle_->read_range(abs_offset, len, tmp)) {
        ailego_free(tmp);
        *data = nullptr;
        return 0;
      }
      owner_->register_tmp_buffer(tmp);
      *data = tmp;
      return len;
    }

    //! LOCKING: shared_lock held only while wiring the MemoryBlock.  The
    //! MemoryBlock carries its own ref_count (raised by get_single_page())
    //! and will release it via its destructor.
    size_t read(size_t offset, MemoryBlock &data, size_t len) override {
      std::shared_lock<std::shared_mutex> latch(owner_->mapping_mutex_);
      if (ailego_unlikely(!owner_->buffer_pool_handle_)) {
        LOG_ERROR(
            "WrappedSegment::read(MemoryBlock&): handle is null, file[%s], "
            "id[%zu]",
            owner_->file_name_.c_str(), segment_id_);
        return 0;
      }
      if (ailego_unlikely(offset + len > segment_->meta()->data_size)) {
        auto meta = segment_->meta();
        if (offset > meta->data_size) {
          offset = meta->data_size;
        }
        len = meta->data_size - offset;
      }
      size_t abs_offset = segment_header_start_offset_ +
                          segment_header_->content_offset +
                          segment_->meta()->data_index + offset;
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
          return -1;
        }
        data.reset(owner_->buffer_pool_handle_.get(), page_id, raw);
        return len;
      }
      char *tmp = static_cast<char *>(ailego_aligned_malloc(len, 4096));
      if (!tmp) {
        LOG_ERROR("read error (alloc cross-page temp buffer failed).");
        return -1;
      }
      if (!owner_->buffer_pool_handle_->read_range(abs_offset, len, tmp)) {
        ailego_free(tmp);
        LOG_ERROR("read error (cross-page read_range failed).");
        return -1;
      }
      data = MemoryBlock::MakeOwned(tmp);
      return len;
    }

    //! Write data into the storage with offset
    //! LOCKING: see fetch() above for rationale.
    size_t write(size_t offset, const void *data, size_t len) override {
      std::shared_lock<std::shared_mutex> latch(owner_->mapping_mutex_);
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
      if (ailego_unlikely(offset + len > capacity_)) {
        LOG_ERROR("write() exceeds segment capacity: offset=%zu len=%zu cap=%zu",
                  offset, len, capacity_);
        return 0;
      }
      auto meta = segment_->meta();
      if (offset + len > meta->data_size) {
        meta->data_size = offset + len;
        meta->padding_size = capacity_ - meta->data_size;
        owner_->set_as_dirty();
      }
      size_t abs_offset = segment_header_start_offset_ +
                          segment_header_->content_offset +
                          segment_->meta()->data_index + offset;
      if (owner_->buffer_pool_handle_->write_range(
              abs_offset, len, static_cast<const char *>(data)) != 0) {
        LOG_ERROR("write() page-cache write_range failed at abs_offset=%zu",
                  abs_offset);
        return 0;
      }
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
      owner_->set_as_dirty();
    }

    //! Clone the segment
    IndexStorage::Segment::Pointer clone(void) override {
      return shared_from_this();
    }

   protected:
    friend BufferStorage;
    IndexMapping::Segment *segment_{};

   private:
    BufferStorage *owner_{nullptr};
    size_t segment_id_{};
    size_t capacity_{};
    uint64_t segment_header_start_offset_;
    IndexFormat::MetaHeader *segment_header_;
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
      return ret;
    }
    ret = buffer_pool_->init();
    if (ret != 0) {
      return ret;
    }
    LOG_INFO(
        "BufferStorage opened: file=%s, writable=%d, max_segment_size=%lu, "
        "segment_count=%zu",
        file_name_.c_str(), static_cast<int>(create_if_missing),
        max_segment_size_, segments_.size());
    return 0;
  }

  void register_tmp_buffer(char *buf) {
    std::lock_guard<std::mutex> latch(tmp_buffers_mutex_);
    tmp_buffers_.push_back(buf);
  }

  //! Acquire a page-table block.
  //!
  //! LOCKING CONTRACT: caller MUST already hold a shared_lock (or
  //! unique_lock) on mapping_mutex_.
  char *get_buffer(size_t offset, size_t length, size_t /*block_id*/) {
    if (ailego_unlikely(!buffer_pool_handle_)) {
      LOG_ERROR(
          "BufferStorage::get_buffer: handle is null, file[%s], "
          "offset[%zu], length[%zu]",
          file_name_.c_str(), offset, length);
      return nullptr;
    }
    char *tmp = static_cast<char *>(ailego_aligned_malloc(length, 4096));
    if (!tmp) {
      return nullptr;
    }
    if (!buffer_pool_handle_->read_range(offset, length, tmp)) {
      ailego_free(tmp);
      return nullptr;
    }
    register_tmp_buffer(tmp);
    return tmp;
  }

  int ParseHeader(size_t offset) {
    std::unique_ptr<char[]> buffer(new char[sizeof(header_)]);
    // NOTE: bypass a wrapper get_meta() -- ParseHeader is called from
    // reopen_pool() which already holds a unique_lock on mapping_mutex_
    // (std::shared_mutex is not reentrant -> deadlock).
    if (buffer_pool_handle_->get_meta(offset, sizeof(header_), buffer.get()) !=
        0) {
      LOG_ERROR("Get segment header failed.");
      return IndexError_Runtime;
    }
    uint8_t *header_ptr = reinterpret_cast<uint8_t *>(buffer.get());
    memcpy(&header_, header_ptr, sizeof(header_));
    if (header_.meta_header_size != sizeof(IndexFormat::MetaHeader)) {
      LOG_ERROR("Header meta size is invalid.");
      return IndexError_InvalidLength;
    }
    if (ailego::Crc32c::Hash(&header_, sizeof(header_), header_.header_crc) !=
        header_.header_crc) {
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

  int ParseSegment(size_t offset) {
    // NOTE: this function is only called from ParseToMapping(), which is
    // itself called from either open() (single-threaded construction) or
    // reopen_pool() (always invoked under the unique_lock held by
    // append_segment()).  Do NOT add an internal lock here -- doing so would
    // deadlock the append_segment() path.
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
      if (iter->segment_id_offset > footer_.segments_meta_size) {
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
      id_hash_[seg_name] = id_hash_.size();
      // Update the segments_ entry in-place so that any WrappedSegment
      // instances that already hold a pointer to this entry (via
      // &segments_[name].segment) continue to use the refreshed meta_ptr_
      // after the re-parse.
      segments_[seg_name] = IndexMapping::SegmentInfo{
          IndexMapping::Segment{iter}, current_header_start_offset_, &header_};
      max_segment_size_ =
          std::max(max_segment_size_, iter->data_size + iter->padding_size);
      if (sizeof(IndexFormat::SegmentMeta) * footer_.segment_count >
          footer_.segments_meta_size) {
        return IndexError_InvalidLength;
      }
    }
    buffer_pool_buffers_.push_back(std::move(segment_buffer));
    return 0;
  }

  int ParseToMapping() {
    while (true) {
      int ret;
      ret = ParseHeader(current_header_start_offset_);
      if (ret != 0) {
        LOG_ERROR("Failed to parse header, errno %d, %s", ret,
                  IndexError::What(ret));
        return ret;
      }

      switch (header_.version) {
        case IndexFormat::FORMAT_VERSION:
          break;
        default:
          LOG_ERROR("Unsupported index version: %u", header_.version);
          return IndexError_Unsupported;
      }

      // Unpack footer
      if (header_.meta_footer_size != sizeof(IndexFormat::MetaFooter)) {
        return IndexError_InvalidLength;
      }
      if ((int32_t)header_.meta_footer_offset < 0) {
        return IndexError_Unsupported;
      }
      uint64_t footer_offset =
          header_.meta_footer_offset + current_header_start_offset_;
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
      ret = ParseSegment(segment_start_offset);
      if (ret != 0) {
        LOG_ERROR("Failed to parse segment, errno %d, %s", ret,
                  IndexError::What(ret));
        return ret;
      }

      // Record per-chain metadata offsets so flush_index() can write
      // updated segment metas and footers back to the backing file.
      meta_chains_.push_back({current_header_start_offset_, footer_offset,
                              segment_start_offset,
                              footer_.segments_meta_size});

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
    auto segment_info = this->get_segment_info(id);
    if (!segment_info) {
      return WrappedSegment::Pointer{};
    }
    return std::make_shared<WrappedSegment>(
        this, &segment_info->segment, segment_info->segment_header_start_offset,
        segment_info->segment_header, id_hash_[id]);
  }

  //! Test if it a segment exists
  bool has(const std::string &id) const override {
    return this->has_segment(id);
  }

  //! Retrieve magic number of index
  uint32_t magic(void) const override {
    return header_.magic;
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

  //! Set the index file as dirty
  void set_as_dirty(void) {
    index_dirty_ = true;
  }

  //! Refresh meta information (checksum, update time, etc.)
  void refresh_index(uint64_t /*chkp*/) {
    // In BufferStorage the segment metadata lives in buffer_pool_buffers_.
    // CRC recomputation and disk write are deferred to flush_index().
    // Just mark dirty so flush_index() will include the metadata write.
    index_dirty_ = true;
  }

  //! Flush index storage: persists any pending meta changes (segments_meta +
  //! footer) for each header chain, then asks the page cache to write back
  //! dirty data pages.
  int flush_index(void) {
    if (!index_dirty_) {
      return 0;
    }
    // SHARED LOCK: keep mapping_mutex_ held for the whole flush so that the
    // pool/handle cannot be torn down by append_segment()/close_index()
    // mid-flush.
    std::shared_lock<std::shared_mutex> latch(mapping_mutex_);
    // NULL GUARD: a previous append_segment() may have left the pool in a
    // torn-down state.
    if (!buffer_pool_ || !buffer_pool_handle_) {
      LOG_ERROR("BufferStorage::flush_index skipped: pool not ready, file[%s]",
                file_name_.c_str());
      return IndexError_Runtime;
    }
    if (!buffer_pool_->writable()) {
      // Read-only pool: nothing to flush.
      index_dirty_ = false;
      return 0;
    }
    // Flush all dirty data blocks to the backing file first.
    if (buffer_pool_handle_->flush_all() != 0) {
      LOG_ERROR("flush_all data blocks failed: file[%s]", file_name_.c_str());
      return IndexError_WriteData;
    }
    // For each metadata chain, recompute the segment-meta CRC, update the
    // footer (segments_meta_crc + footer_crc + update_time), and write both
    // the segment metadata and the footer back to the backing file.
    for (size_t ci = 0;
         ci < meta_chains_.size() && ci < buffer_pool_buffers_.size(); ++ci) {
      const MetaChain &chain = meta_chains_[ci];
      const char *seg_buf = buffer_pool_buffers_[ci].get();
      // Read the on-disk footer into a local copy so we can update it.
      IndexFormat::MetaFooter footer;
      if (buffer_pool_handle_->get_meta(
              chain.footer_file_offset, sizeof(footer),
              reinterpret_cast<char *>(&footer)) != 0) {
        LOG_ERROR("Failed to read footer for flush: file[%s], chain[%zu]",
                  file_name_.c_str(), ci);
        return IndexError_Runtime;
      }
      // Recompute segment metadata CRC and refresh the footer.
      footer.segments_meta_crc =
          ailego::Crc32c::Hash(seg_buf, chain.segment_meta_size, 0u);
      IndexFormat::UpdateMetaFooter(&footer, 0);
      // Write segment metadata back to disk.
      if (buffer_pool_handle_->write_meta(chain.segment_meta_file_offset,
                                          chain.segment_meta_size,
                                          seg_buf) != 0) {
        LOG_ERROR("Failed to write segment meta: file[%s], chain[%zu]",
                  file_name_.c_str(), ci);
        return IndexError_WriteData;
      }
      // Write the updated footer back to disk.
      if (buffer_pool_handle_->write_meta(
              chain.footer_file_offset, sizeof(footer),
              reinterpret_cast<const char *>(&footer)) != 0) {
        LOG_ERROR("Failed to write footer: file[%s], chain[%zu]",
                  file_name_.c_str(), ci);
        return IndexError_WriteData;
      }
    }
    index_dirty_ = false;
    return 0;
  }

  //! Close index storage
  void close_index(void) {
    // Flush any outstanding dirty metadata to disk before tearing down.
    // IMPORTANT: call flush_index() BEFORE taking the unique_lock below;
    // flush_index() internally takes a shared_lock on the same mutex and
    // std::shared_mutex is NOT reentrant.
    this->flush_index();
    std::unique_lock<std::shared_mutex> latch(mapping_mutex_);
    file_name_.clear();
    id_hash_.clear();
    segments_.clear();
    memset(&header_, 0, sizeof(header_));
    memset(&footer_, 0, sizeof(footer_));
    {
      std::lock_guard<std::mutex> tmp_latch(tmp_buffers_mutex_);
      for (char *p : tmp_buffers_) {
        if (p) {
          ailego_free(p);
        }
      }
      tmp_buffers_.clear();
    }
    buffer_pool_handle_.reset();
    buffer_pool_.reset();
    max_segment_size_ = 0;
    buffer_pool_buffers_.clear();
    meta_chains_.clear();
    // Drop retired pools last -- any stray MemoryBlock still holding a raw
    // handle pointer would hit use-after-free here, but by close_index()
    // time all build/search threads are expected to have joined.
    retired_handles_.clear();
    retired_pools_.clear();
    current_header_start_offset_ = 0;
  }

  //! Reopen the buffer pool and reload the mapping.  Used both as the final
  //! success step of append_segment() and as a rollback path when any
  //! IndexMapping operation fails mid-way through append_segment().
  //!
  //! VecBufferPool's constructor throws on open()/fstat() failure; we catch
  //! that here and translate it into an error code.
  int reopen_pool() {
    try {
      buffer_pool_ = std::make_shared<ailego::VecBufferPool>(
          file_name_, /*writable=*/true, /*create=*/false);
      buffer_pool_handle_ = std::make_shared<ailego::VecBufferPoolHandle>(
          buffer_pool_->get_handle());
    } catch (const std::exception &e) {
      LOG_ERROR(
          "BufferStorage::reopen_pool failed to create pool: file[%s], "
          "what[%s]",
          file_name_.c_str(), e.what());
      buffer_pool_.reset();
      buffer_pool_handle_.reset();
      return IndexError_Runtime;
    }
    int ret = ParseToMapping();
    if (ret != 0) {
      LOG_ERROR(
          "BufferStorage::reopen_pool failed to parse mapping: file[%s], "
          "errno[%d]",
          file_name_.c_str(), ret);
      return ret;
    }
    return buffer_pool_->init();
  }

  //! Append a segment into storage
  int append_segment(const std::string &id, size_t size) {
    // Flush any in-memory metadata changes (data_size, padding_size, CRC)
    // accumulated by prior write()/resize() calls BEFORE we reset the buffer
    // pool below.  Without this flush, those changes would be lost when
    // buffer_pool_buffers_ is cleared and re-populated from disk.
    // IMPORTANT: call flush_index() BEFORE taking the unique_lock below;
    // flush_index() internally takes a shared_lock on the same mutex and
    // std::shared_mutex is NOT reentrant.
    this->flush_index();

    // UNIQUE LOCK: hold the mutex for the entire structural modification
    // (reset -> IndexMapping.open/append/flush -> reopen_pool).  Concurrent
    // readers/writers taking shared_lock will block here.
    std::unique_lock<std::shared_mutex> latch(mapping_mutex_);

    // RETIRE the old pool instead of immediately destroying it.  MemoryBlock
    // objects held by other threads carry a ref_count on a block inside this
    // pool but store only a RAW VecBufferPoolHandle*; if we reset() the
    // shared_ptr here, the pool destructor fires while those ref_counts are
    // still > 0 and the is_released() assert trips.  By parking in
    // retired_pools_ the pool survives until all external refs are gone.
    auto prune_retired = [&]() {
      size_t w = 0;
      for (size_t r = 0; r < retired_pools_.size(); ++r) {
        bool any_held = false;
        auto &pt = retired_pools_[r]->page_table_;
        for (size_t i = 0; i < pt.entry_num(); ++i) {
          if (!pt.is_released(i)) {
            any_held = true;
            break;
          }
        }
        if (any_held) {
          if (w != r) {
            retired_pools_[w] = std::move(retired_pools_[r]);
            retired_handles_[w] = std::move(retired_handles_[r]);
          }
          ++w;
        }
      }
      retired_pools_.resize(w);
      retired_handles_.resize(w);
    };
    prune_retired();

    // Flush and release the buffer pool so IndexMapping can safely open
    // and structurally modify the same file.
    if (buffer_pool_handle_) {
      buffer_pool_handle_->flush_all();
    }
    // Park the old pool + handle.
    if (buffer_pool_) {
      retired_pools_.push_back(std::move(buffer_pool_));
      retired_handles_.push_back(std::move(buffer_pool_handle_));
    } else {
      buffer_pool_handle_.reset();
    }
    buffer_pool_.reset();
    // Reset parse-time state EXCEPT for segments_: WrappedSegment instances
    // held by callers store raw pointers into segments_' mapped values.
    // The C++ standard guarantees that unordered_map references/pointers to
    // mapped values are never invalidated by insertions, so we can safely
    // leave segments_ intact and update entries in-place during re-parse.
    id_hash_.clear();
    buffer_pool_buffers_.clear();
    meta_chains_.clear();
    current_header_start_offset_ = 0u;
    max_segment_size_ = 0u;
    memset(&header_, 0, sizeof(header_));
    memset(&footer_, 0, sizeof(footer_));

    // Delegate the structural append to IndexMapping (same engine used by
    // MMapFileStorage) so the on-disk format stays consistent.
    IndexMapping mapping;
    int ret = mapping.open(file_name_, /*cow=*/false, /*full_mode=*/false);
    if (ret != 0) {
      LOG_ERROR(
          "BufferStorage::append_segment failed to open IndexMapping: "
          "file[%s], id[%s], errno[%d]",
          file_name_.c_str(), id.c_str(), ret);
      reopen_pool();
      return ret;
    }
    ret = mapping.append(id, size);
    if (ret != 0) {
      LOG_ERROR(
          "BufferStorage::append_segment failed to append segment: "
          "file[%s], id[%s], errno[%d]",
          file_name_.c_str(), id.c_str(), ret);
      mapping.close();
      reopen_pool();
      return ret;
    }
    mapping.refresh(0);
    ret = mapping.flush();
    mapping.close();
    if (ret != 0) {
      LOG_ERROR(
          "BufferStorage::append_segment failed to flush: "
          "file[%s], id[%s], errno[%d]",
          file_name_.c_str(), id.c_str(), ret);
      reopen_pool();
      return ret;
    }

    // Reopen the buffer pool and reload the mapping so the new segment is
    // accessible via get_segment_info() / get().
    return reopen_pool();
  }

  //! Test if a segment exists
  bool has_segment(const std::string &id) const {
    std::shared_lock<std::shared_mutex> latch(mapping_mutex_);
    return (segments_.find(id) != segments_.end());
  }

  //! Get a segment from storage
  IndexMapping::SegmentInfo *get_segment_info(const std::string &id) {
    std::shared_lock<std::shared_mutex> latch(mapping_mutex_);
    auto iter = segments_.find(id);
    if (iter == segments_.end()) {
      return nullptr;
    }
    return &iter->second;
  }

 private:
  bool index_dirty_{false};
  mutable std::shared_mutex mapping_mutex_{};

  std::vector<char *> tmp_buffers_{};
  mutable std::mutex tmp_buffers_mutex_{};

  // buffer manager
  std::string file_name_;
  IndexFormat::MetaHeader header_{};
  IndexFormat::MetaFooter footer_{};
  std::unordered_map<std::string, IndexMapping::SegmentInfo> segments_{};
  std::unordered_map<std::string, size_t> id_hash_{};
  uint64_t max_segment_size_{0};
  std::vector<std::unique_ptr<char[]>> buffer_pool_buffers_{};

  // Retired pools: see prune_retired() in append_segment() for the
  // life-cycle contract.
  std::vector<ailego::VecBufferPool::Pointer> retired_pools_{};
  std::vector<ailego::VecBufferPoolHandle::Pointer> retired_handles_{};

  ailego::VecBufferPool::Pointer buffer_pool_{nullptr};
  ailego::VecBufferPoolHandle::Pointer buffer_pool_handle_{nullptr};
  uint64_t current_header_start_offset_{0u};
  uint64_t buffer_size_{2lu * 1024 * 1024 * 1024};  // 2G

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
  };
  std::vector<MetaChain> meta_chains_{};
};

INDEX_FACTORY_REGISTER_STORAGE(BufferStorage);

}  // namespace core
}  // namespace zvec
