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
#include <zvec/ailego/buffer/buffer_manager.h>
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
  class Segment : public IndexStorage::Segment,
                  public std::enable_shared_from_this<Segment> {
   public:
    //! Index Storage Pointer
    typedef std::shared_ptr<Segment> Pointer;

    //! Constructor
    Segment(BufferStorage *owner, IndexMapping::Segment *segment)
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
      ailego::BufferHandle buffer_handle =
          owner_->get_buffer_handle(offset, len);
      memmove(buf, (const uint8_t *)buffer_handle.pin_vector_data() + offset,
              len);
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
      size_t buffer_offset =
          segment_->meta()->data_index + owner_->get_context_offset() + offset;
      ailego::BufferHandle buffer_handle =
          owner_->get_buffer_handle(buffer_offset, len);
      *data = buffer_handle.pin_vector_data();
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
      size_t buffer_offset =
          segment_->meta()->data_index + owner_->get_context_offset() + offset;
      data.reset(owner_->get_buffer_handle_ptr(buffer_offset, len));
      if (data.data()) {
        return len;
      } else {
        LOG_ERROR(
            "Buffer handle is null, now used memory: %zu, new: %zu",
            (size_t)ailego::BufferManager::Instance().total_size_in_bytes(),
            len);
        return -1;
      }
    }

    //! Write data into the storage with offset
    size_t write(size_t /*offset*/, const void * /*data*/,
                 size_t len) override {
      return len;
    }

    //! Resize size of data
    size_t resize(size_t /*size*/) override {
      return 0;
    }

    //! Update crc of data
    void update_data_crc(uint32_t /*crc*/) override {}

    //! Clone the segment
    IndexStorage::Segment::Pointer clone(void) override {
      return shared_from_this();
    }

   private:
    IndexMapping::Segment *segment_{};
    BufferStorage *owner_{nullptr};
    size_t capacity_{};
  };

  //! Destructor
  virtual ~BufferStorage(void) {
    this->cleanup();
  }

  //! Initialize storage
  int init(const ailego::Params & /*params*/) override {
    return 0;
  }

  //! Cleanup storage
  int cleanup(void) override {
    this->close_index();
    return 0;
  }

  //! Open storage
  int open(const std::string &path, bool /*create*/) override {
    file_name_ = path;
    return ParseToMapping();
  }

  ailego::BufferHandle get_buffer_handle(int offset, int length) {
    ailego::BufferID buffer_id =
        ailego::BufferID::VectorID(file_name_, offset, length);
    return ailego::BufferManager::Instance().acquire(buffer_id);
  }

  ailego::BufferHandle::Pointer get_buffer_handle_ptr(int offset, int length) {
    ailego::BufferID buffer_id =
        ailego::BufferID::VectorID(file_name_, offset, length);
    return ailego::BufferManager::Instance().acquire_ptr(buffer_id);
  }

  int ParseHeader(int offset) {
    ailego::BufferHandle header_handle =
        get_buffer_handle(offset, sizeof(header_));
    void *buffer = header_handle.pin_vector_data();
    uint8_t *header_ptr = reinterpret_cast<uint8_t *>(buffer);
    memcpy(&header_, header_ptr, sizeof(header_));
    header_handle.unpin_vector_data();
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

  int ParseFooter(int offset) {
    ailego::BufferHandle footer_handle =
        get_buffer_handle(offset, sizeof(footer_));
    void *buffer = footer_handle.pin_vector_data();
    uint8_t *footer_ptr = reinterpret_cast<uint8_t *>(buffer);
    memcpy(&footer_, footer_ptr, sizeof(footer_));
    footer_handle.unpin_vector_data();
    if (offset < (int)footer_.segments_meta_size) {
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

  int ParseSegment(int offset) {
    ailego::BufferHandle segment_start_handle =
        get_buffer_handle(offset, footer_.segments_meta_size);
    void *segment_buffer = segment_start_handle.pin_vector_data();
    if (ailego::Crc32c::Hash(segment_buffer, footer_.segments_meta_size, 0u) !=
        footer_.segments_meta_crc) {
      LOG_ERROR("Index segments meta checksum is invalid.");
      return IndexError_InvalidChecksum;
    }
    IndexFormat::SegmentMeta *segment_start =
        reinterpret_cast<IndexFormat::SegmentMeta *>(segment_buffer);
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
      segments_.emplace(
          std::string(reinterpret_cast<const char *>(segment_start) +
                      iter->segment_id_offset),
          iter);
      if (sizeof(IndexFormat::SegmentMeta) * footer_.segment_count >
          footer_.segments_meta_size) {
        return IndexError_InvalidLength;
      }
    }
    return 0;
  }

  int ParseToMapping() {
    ParseHeader(0);

    // Unpack footer
    if (header_.meta_footer_size != sizeof(IndexFormat::MetaFooter)) {
      return IndexError_InvalidLength;
    }
    if ((int32_t)header_.meta_footer_offset < 0) {
      return IndexError_Unsupported;
    }
    size_t footer_offset = header_.meta_footer_offset;
    ParseFooter(footer_offset);

    // Unpack segment table
    if (sizeof(IndexFormat::SegmentMeta) * footer_.segment_count >
        footer_.segments_meta_size) {
      return IndexError_InvalidLength;
    }
    const int segment_start_offset = footer_offset - footer_.segments_meta_size;
    ParseSegment(segment_start_offset);
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
    IndexMapping::Segment *segment = this->get_segment(id);
    if (!segment) {
      return BufferStorage::Segment::Pointer();
    }
    return std::make_shared<BufferStorage::Segment>(this, segment);
  }

  //! Test if it a segment exists
  bool has(const std::string &id) const override {
    return this->has_segment(id);
  }

  //! Retrieve magic number of index
  uint32_t magic(void) const override {
    return header_.magic;
  }

  uint32_t get_context_offset() {
    return header_.content_offset;
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

  //! Set the index file as dirty
  void set_as_dirty(void) {
    index_dirty_ = true;
  }

  //! Refresh meta information (checksum, update time, etc.)
  void refresh_index(uint64_t /*chkp*/) {}

  //! Flush index storage
  int flush_index(void) {
    return 0;
  }

  //! Close index storage
  void close_index(void) {
    std::lock_guard<std::mutex> latch(mapping_mutex_);
    file_name_.clear();
    segments_.clear();
    memset(&header_, 0, sizeof(header_));
    memset(&footer_, 0, sizeof(footer_));
  }

  //! Append a segment into storage
  int append_segment(const std::string & /*id*/, size_t /*size*/) {
    return 0;
  }

  //! Test if a segment exists
  bool has_segment(const std::string &id) const {
    std::lock_guard<std::mutex> latch(mapping_mutex_);
    return (segments_.find(id) != segments_.end());
  }

  //! Get a segment from storage
  IndexMapping::Segment *get_segment(const std::string &id) {
    std::lock_guard<std::mutex> latch(mapping_mutex_);
    auto iter = segments_.find(id);
    if (iter == segments_.end()) {
      return nullptr;
    }
    IndexMapping::Segment *item = &iter->second;
    return item;
  }

 private:
  // mmap
  uint32_t segment_meta_capacity_{1024 * 1024};
  // bool copy_on_write_{false};
  // bool force_flush_{false};
  // bool memory_locked_{false};
  // bool memory_warmup_{false};
  bool index_dirty_{false};
  mutable IndexMapping mapping_{};
  mutable std::mutex mapping_mutex_{};

  // buffer manager
  std::string file_name_;
  IndexFormat::MetaHeader header_;
  IndexFormat::MetaFooter footer_;
  std::map<std::string, IndexMapping::Segment> segments_{};
};

INDEX_FACTORY_REGISTER_STORAGE(BufferStorage);

}  // namespace core
}  // namespace zvec
