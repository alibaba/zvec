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
#include <zvec/ailego/buffer/vector_page_table.h>
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
    //! LOCKING: takes a shared_lock on owner_->mapping_mutex_ for the WHOLE
    //! method.  This is required to keep the acquire (get_block, ref_count+1)
    //! and release (release_one, ref_count-1) pair inside the same critical
    //! section as seen by append_segment() / close_index() (which take a
    //! unique_lock).  Without this, append_segment() could reset
    //! buffer_pool_handle_ between acquire and release on this thread, so
    //! the ref_count on the OLD pool would never reach zero and the
    //! VecBufferPool destructor would fail `is_released(i)`.
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
      size_t buffer_offset = segment_header_start_offset_ +
                             segment_header_->content_offset +
                             segment_->meta()->data_index;
      auto *raw = owner_->get_buffer(buffer_offset, capacity_, segment_id_);
      if (!raw) {
        return 0;
      }
      auto *data = raw + offset;
      memmove(buf, data, len);
      // Release the buffer-pool ref count acquired by get_buffer(); the data
      // has already been copied into the caller's buffer so the block no
      // longer needs to be pinned in memory.
      owner_->buffer_pool_handle_->release_one(segment_id_);
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
      size_t buffer_offset = segment_header_start_offset_ +
                             segment_header_->content_offset +
                             segment_->meta()->data_index;
      auto *raw = owner_->get_buffer(buffer_offset, capacity_, segment_id_);
      if (!raw) {
        *data = nullptr;
        return 0;
      }
      *data = raw + offset;
      // Release the buffer-pool ref count acquired by get_buffer().
      // NOTE: this makes the returned raw pointer only safe to use before
      // any subsequent acquire_buffer() call that might trigger eviction
      // (i.e. single-threaded or with external locking). Callers that need
      // a stable pin across potential eviction points should use the
      // read(MemoryBlock&) overload instead.
      owner_->buffer_pool_handle_->release_one(segment_id_);
      return len;
    }

    //! LOCKING: shared_lock held only while wiring the MemoryBlock.  The
    //! MemoryBlock carries its own ref_count (raised by get_block()) and
    //! will release it via its destructor -- that release happens OUTSIDE
    //! this lock, so the caller must not keep a MemoryBlock alive across
    //! an append_segment() call on the same BufferStorage.  In practice
    //! MemoryBlock lifetimes are short (a single search / vector fetch),
    //! and append_segment() is only driven by index-build control flow.
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
      size_t buffer_offset = segment_header_start_offset_ +
                             segment_header_->content_offset +
                             segment_->meta()->data_index;
      auto *raw = owner_->get_buffer(buffer_offset, capacity_, segment_id_);
      if (!raw) {
        return 0;
      }

      data.reset(owner_->buffer_pool_handle_.get(), segment_id_, raw + offset);
      if (data.data()) {
        return len;
      } else {
        LOG_ERROR("read error.");
        return -1;
      }
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
      size_t data_tail = offset + len;
      ailego_zero_if_false(data_tail <= capacity_);
      // In read-only mode the write is a silent no-op so that callers that
      // unconditionally write (e.g. CRC updates) do not return an error.
      if (!owner_->buffer_pool_->is_writable()) {
        return len;
      }
      size_t file_offset = segment_header_start_offset_ +
                           segment_header_->content_offset +
                           segment_->meta()->data_index;
      char *raw = owner_->get_buffer(file_offset, capacity_, segment_id_);
      if (!raw) {
        return 0;
      }
      auto meta = segment_->meta();
      if (data_tail > meta->data_size) {
        meta->data_size = data_tail;
        meta->padding_size = capacity_ - data_tail;
        owner_->set_as_dirty();
      }
      memmove(raw + offset, data, len);
      // Mark the cached block dirty so flush_index() will persist it to disk.
      owner_->buffer_pool_->page_table_.mark_dirty(segment_id_);
      // Release the buffer-pool ref count acquired by get_buffer().
      // write() does not create a MemoryBlock so there is no automatic
      // release_one(); without this the block's ref count would remain > 0
      // until the pool is destroyed, triggering the is_released() assert in
      // VecBufferPool::~VecBufferPool().
      owner_->buffer_pool_handle_->release_one(segment_id_);
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
        return error_code;
      }
    }
    // Open in writable mode when the caller expects to modify the index
    // (create_if_missing mirrors the MMapFileStorage convention: true means
    // the index may need to be written, false means read-only).
    buffer_pool_ = std::make_shared<ailego::VecBufferPool>(
        path, /*writable=*/create_if_missing, /*create=*/false);
    buffer_pool_handle_ = std::make_shared<ailego::VecBufferPoolHandle>(
        buffer_pool_->get_handle());
    int ret = ParseToMapping();
    if (ret != 0) {
      return ret;
    }
    ret = buffer_pool_->init(segments_.size());
    if (ret != 0) {
      return ret;
    }
    LOG_INFO(
        "BufferStorage opened: file=%s, max_segment_size=%lu, "
        "segment_count=%zu, writable=%d",
        file_name_.c_str(), max_segment_size_, segments_.size(),
        static_cast<int>(create_if_missing));
    return 0;
  }

  //! Acquire a page-table block.
  //!
  //! LOCKING CONTRACT: caller MUST already hold a shared_lock (or
  //! unique_lock) on mapping_mutex_.  The shared_lock must stay held until
  //! AFTER the matching release_one() so that append_segment() cannot reset
  //! the handle between acquire and release on this thread (which would
  //! leak the ref_count on the old pool and trip is_released() in
  //! VecBufferPool::~VecBufferPool()).
  //!
  //! The null-pointer guard below is kept defensively for belt-and-braces:
  //! with the locking contract above it should never fire.
  char *get_buffer(size_t offset, size_t length, size_t block_id) {
    if (ailego_unlikely(!buffer_pool_handle_)) {
      LOG_ERROR(
          "BufferStorage::get_buffer: handle is null, file[%s], "
          "block_id[%zu], offset[%zu], length[%zu]",
          file_name_.c_str(), block_id, offset, length);
      return nullptr;
    }
    return buffer_pool_handle_->get_block(offset, length, block_id);
  }

  int ParseHeader(size_t offset) {
    std::unique_ptr<char[]> buffer(new char[sizeof(header_)]);
    // NOTE: bypass the public get_meta() here -- it takes a shared_lock on
    // mapping_mutex_, and ParseHeader is called from reopen_pool() which
    // already holds a unique_lock on the same mutex (std::shared_mutex is
    // not reentrant -> deadlock).  The handle was just reconstructed in
    // reopen_pool() so it is guaranteed non-null here; in open() the handle
    // is also populated before ParseToMapping() runs.
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
    // Bypass BufferStorage::get_meta() -- see ParseHeader() comment for why.
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
    // Bypass BufferStorage::get_meta() -- see ParseHeader() comment for why.
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
  //! Create the initial on-disk index structure and write the mandatory
  //! version segment.  Uses IndexMapping (the same engine as MMapFileStorage)
  //! so the produced file is fully compatible with both storage backends.
  int init_index(const std::string &path) {
    IndexMapping mapping;
    // Create file + write MetaHeader / padding / MetaFooter.
    int ret = mapping.create(path, segment_meta_capacity_);
    if (ret != 0) {
      LOG_ERROR(
          "BufferStorage failed to create index file: path[%s], errno[%d]",
          path.c_str(), ret);
      return ret;
    }

    // Append and populate the mandatory version segment.
    size_t data_size = std::strlen(IndexVersion::Details());
    ret = mapping.append(INDEX_VERSION_SEGMENT_NAME, data_size);
    if (ret != 0) {
      LOG_ERROR(
          "BufferStorage failed to append version segment: path[%s], errno[%d]",
          path.c_str(), ret);
      mapping.close();
      return ret;
    }
    IndexMapping::Segment *seg =
        mapping.map(INDEX_VERSION_SEGMENT_NAME, false, false);
    if (!seg) {
      LOG_ERROR("BufferStorage failed to map version segment: path[%s]",
                path.c_str());
      mapping.close();
      return IndexError_MMapFile;
    }
    auto meta = seg->meta();
    size_t capacity = meta->data_size + meta->padding_size;
    memcpy(seg->data(), IndexVersion::Details(), data_size);
    seg->set_dirty();
    meta->data_crc = ailego::Crc32c::Hash(seg->data(), data_size, 0);
    meta->data_size = data_size;
    meta->padding_size = capacity - data_size;

    // Refresh checksums, flush to disk, then release the mapping so the
    // VecBufferPool can open the completed file.
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

  //! Flush index storage
  int flush_index(void) {
    if (!index_dirty_) {
      return 0;
    }
    // SHARED LOCK: keep mapping_mutex_ held for the whole flush so that the
    // pool/handle cannot be torn down by append_segment()/close_index()
    // mid-flush.  flush_index() only performs read-only state inspection
    // plus pwrite()s via buffer_pool_handle_; it does not mutate any
    // mapping-state protected by the mutex, so shared ownership is enough.
    std::shared_lock<std::shared_mutex> latch(mapping_mutex_);
    // NULL GUARD: a previous append_segment() may have left the pool in a
    // torn-down state (reset but not yet reopened, or reopen failed).  Fail
    // fast instead of dereferencing a null buffer_pool_ / buffer_pool_handle_.
    if (!buffer_pool_ || !buffer_pool_handle_) {
      LOG_ERROR("BufferStorage::flush_index skipped: pool not ready, file[%s]",
                file_name_.c_str());
      return IndexError_Runtime;
    }
    if (!buffer_pool_->is_writable()) {
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
                                          seg_buf,
                                          chain.segment_meta_size) != 0) {
        LOG_ERROR("Failed to write segment meta: file[%s], chain[%zu]",
                  file_name_.c_str(), ci);
        return IndexError_WriteData;
      }
      // Write the updated footer back to disk.
      if (buffer_pool_handle_->write_meta(
              chain.footer_file_offset, reinterpret_cast<const char *>(&footer),
              sizeof(footer)) != 0) {
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
    // Flush any outstanding dirty metadata (segment data_size / CRC) to disk
    // before tearing down the buffer pool.  MMapFileStorage does the same via
    // mapping_.close() which triggers an munmap-flush; here we must be
    // explicit because the pool destructor only flushes dirty data blocks,
    // not the segment-meta area stored in buffer_pool_buffers_.
    //
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
  }

  //! Reopen the buffer pool and reload the mapping.  Used both as the final
  //! success step of append_segment() and as a rollback path when any
  //! IndexMapping operation fails mid-way through append_segment().
  //!
  //! On failure this method may leave buffer_pool_ / buffer_pool_handle_ as
  //! nullptr; in that case subsequent calls MUST go through the null-pointer
  //! guards in get_buffer() / get_meta() / flush_index() so that the storage
  //! fails fast with a log rather than crashing on a null shared_ptr deref.
  //!
  //! VecBufferPool's constructor throws on open()/fstat() failure; we catch
  //! that here and translate it into an error code so callers never have to
  //! handle exceptions from this path.
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
    return buffer_pool_->init(segments_.size());
  }

  //! Append a segment into storage
  int append_segment(const std::string &id, size_t size) {
    // Flush any in-memory metadata changes (data_size, padding_size, CRC)
    // accumulated by prior write()/resize() calls BEFORE we reset the buffer
    // pool below.  Without this flush, those changes would be lost when
    // buffer_pool_buffers_ is cleared and re-populated from disk.
    // Example: init_storage() writes to FLAT_LINEAR_META_SEG_ID (sets
    // data_size=4379), then calls append("IndexMeta").  Without this flush
    // the 4379 is never written to disk and the subsequent re-parse reads
    // the original on-disk value (0 / 128), causing a format error on reload.
    // IMPORTANT: call flush_index() BEFORE taking the unique_lock below;
    // flush_index() internally takes a shared_lock on the same mutex and
    // std::shared_mutex is NOT reentrant.
    this->flush_index();

    // UNIQUE LOCK: hold the mutex for the entire structural modification
    // (reset -> IndexMapping.open/append/flush -> reopen_pool).  Concurrent
    // readers/writers taking shared_lock in get_buffer()/get_meta()/
    // flush_index() will block here, which eliminates the "empty window"
    // race where the handle was temporarily null.  ParseSegment() is
    //! explicitly written as a no-lock inner helper to avoid reentrant-lock
    //! deadlock on this path.
    std::unique_lock<std::shared_mutex> latch(mapping_mutex_);

    // RETIRE the old pool instead of immediately destroying it.  The reason:
    // MemoryBlock / BufferPoolMemoryBlock / Neighbors objects held by other
    // threads carry a ref_count on a block inside this pool but store only a
    // RAW VecBufferPoolHandle*; they do not hold a shared_ptr keeping the
    // pool alive.  If we reset() the shared_ptr here, the pool destructor
    // fires while those ref_counts are still > 0 and the is_released()
    // assert in ~VecBufferPool() trips.  By parking the shared_ptr in
    // retired_pools_ / retired_handles_ the pool survives until (a) the
    // last external MemoryBlock is destroyed AND (b) BufferStorage itself
    // chooses to drop the shared_ptr, at which point ref_counts are back to
    // zero and ~VecBufferPool runs cleanly.  The prune below drops any
    // entries that are no longer referenced by anyone except us (use_count
    // == 1), so retired_pools_ does not grow without bound.
    auto prune_retired = [&]() {
      size_t w = 0;
      for (size_t r = 0; r < retired_pools_.size(); ++r) {
        // MemoryBlock objects held by other threads store a RAW handle
        // pointer; they do not bump shared_ptr::use_count.  So instead of
        // checking use_count(), walk the pool's page_table_ and keep the
        // entry as long as ANY block still has ref_count > 0.  Once every
        // block is released, no one can possibly call release_one() on us,
        // so it is safe to drop the shared_ptr and let ~VecBufferPool run.
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
    // and structurally modify (meta section + content area) the same file.
    if (buffer_pool_handle_) {
      buffer_pool_handle_->flush_all();
    }
    // Park the old pool + handle so that any MemoryBlock still holding a
    // raw handle pointer (on another thread) can complete its release_one()
    // without use-after-free on the pool.  We intentionally move the
    // shared_ptrs (not reset) here; prune_retired() above has already
    // removed every previously retired pool that no one else was using.
    if (buffer_pool_) {
      retired_pools_.push_back(std::move(buffer_pool_));
      retired_handles_.push_back(std::move(buffer_pool_handle_));
    } else {
      buffer_pool_handle_.reset();
    }
    buffer_pool_.reset();
    // Reset parse-time state EXCEPT for segments_: WrappedSegment instances
    // held by callers (e.g. FlatStreamerEntity::segments_[]) store raw
    // pointers into segments_' mapped values (IndexMapping::Segment).  The
    // C++ standard guarantees that unordered_map references/pointers to
    // mapped values are never invalidated by insertions, so we can safely
    // leave segments_ intact and update entries in-place during re-parse.
    // Clearing segments_ here would make those pointers dangle and cause a
    // use-after-free crash during the next read or search.
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
      // ROLLBACK: rebuild the pool/mapping so later calls don't crash on a
      // null buffer_pool_handle_.  The original errno is the first-class
      // failure signal, so we intentionally ignore the rollback return value.
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
    // accessible via get_segment_info() / get().  reopen_pool() encapsulates
    // make_shared<VecBufferPool>() (with try/catch around its throwing
    // constructor), the handle re-wrap, ParseToMapping() and the final
    // buffer_pool_->init(segments_.size()).
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

  // buffer manager
  std::string file_name_;
  IndexFormat::MetaHeader header_{};
  IndexFormat::MetaFooter footer_{};
  std::unordered_map<std::string, IndexMapping::SegmentInfo> segments_{};
  std::unordered_map<std::string, size_t> id_hash_{};
  uint64_t max_segment_size_{0};
  std::vector<std::unique_ptr<char[]>> buffer_pool_buffers_{};

  // Retired pools: see prune_retired() in append_segment() for the
  // life-cycle contract.  Each entry is a pool (and its matching handle)
  // that was replaced by a later append_segment() but still may be
  // referenced by in-flight MemoryBlock objects on other threads.
  // Cleared by close_index(); pruned on every append_segment() to avoid
  // unbounded growth.
  std::vector<ailego::VecBufferPool::Pointer> retired_pools_{};
  std::vector<ailego::VecBufferPoolHandle::Pointer> retired_handles_{};

  ailego::VecBufferPool::Pointer buffer_pool_{nullptr};
  ailego::VecBufferPoolHandle::Pointer buffer_pool_handle_{nullptr};
  uint64_t current_header_start_offset_{0u};
  uint64_t buffer_size_{2lu * 1024 * 1024 * 1024};  // 2G

  // Capacity (in bytes) of the segment metadata section written by
  // init_index(). Sized to hold ~128 SegmentMeta entries (128 × 32 B = 4096 B).
  // Mirrors the default used by MMapFileStorage; the section auto-expands via
  // header chaining when more segments are needed.
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
