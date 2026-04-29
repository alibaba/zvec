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
    size_t fetch(size_t offset, void *buf, size_t len) const override {
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
      size_t buffer_offset = segment_header_start_offset_ +
                             segment_header_->content_offset +
                             segment_->meta()->data_index;
      auto *raw = owner_->get_buffer(buffer_offset, capacity_, segment_id_);
      if (!raw) {
        return 0;
      }
      *data = raw + offset;
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
  int open(const std::string &path, bool /*create_if_missing*/) override {
    file_name_ = path;
    buffer_pool_ = std::make_shared<ailego::VecBufferPool>(path);
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
        "segment_count=%zu",
        file_name_.c_str(), max_segment_size_, segments_.size());
    return 0;
  }

  char *get_buffer(size_t offset, size_t length, size_t block_id) {
    return buffer_pool_handle_->get_block(offset, length, block_id);
  }

  int get_meta(size_t offset, size_t length, char *out) {
    return buffer_pool_handle_->get_meta(offset, length, out);
  }

  int ParseHeader(size_t offset) {
    std::unique_ptr<char[]> buffer(new char[sizeof(header_)]);
    if (get_meta(offset, sizeof(header_), buffer.get()) != 0) {
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
    if (get_meta(offset, sizeof(footer_), buffer.get()) != 0) {
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
    std::lock_guard<std::mutex> latch(mapping_mutex_);
    std::unique_ptr<char[]> segment_buffer =
        std::make_unique<char[]>(footer_.segments_meta_size);
    if (get_meta(offset, footer_.segments_meta_size, segment_buffer.get()) !=
        0) {
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
      id_hash_.emplace(
          std::string(reinterpret_cast<const char *>(segment_start) +
                      iter->segment_id_offset),
          segments_.size());
      segments_.emplace(
          std::string(reinterpret_cast<const char *>(segment_start) +
                      iter->segment_id_offset),
          IndexMapping::SegmentInfo{IndexMapping::Segment{iter},
                                    current_header_start_offset_, &header_});
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
  //! Initialize index version segment
  int init_version_segment(void) {
    size_t data_size = std::strlen(IndexVersion::Details());
    int error_code =
        this->append_segment(INDEX_VERSION_SEGMENT_NAME, data_size);
    if (error_code != 0) {
      return error_code;
    }

    auto segment = &get_segment_info(INDEX_VERSION_SEGMENT_NAME)->segment;
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
  int init_index(const std::string & /*path*/) {
    // Add index version
    int error_code = this->init_version_segment();
    if (error_code != 0) {
      return error_code;
    }

    // Refresh mapping
    this->refresh_index(0);
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
    id_hash_.clear();
    segments_.clear();
    memset(&header_, 0, sizeof(header_));
    memset(&footer_, 0, sizeof(footer_));
    buffer_pool_handle_.reset();
    buffer_pool_.reset();
    max_segment_size_ = 0;
    buffer_pool_buffers_.clear();
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
  IndexMapping::SegmentInfo *get_segment_info(const std::string &id) {
    std::lock_guard<std::mutex> latch(mapping_mutex_);
    auto iter = segments_.find(id);
    if (iter == segments_.end()) {
      return nullptr;
    }
    return &iter->second;
  }

 private:
  bool index_dirty_{false};
  mutable std::mutex mapping_mutex_{};

  // buffer manager
  std::string file_name_;
  IndexFormat::MetaHeader header_{};
  IndexFormat::MetaFooter footer_{};
  std::unordered_map<std::string, IndexMapping::SegmentInfo> segments_{};
  std::unordered_map<std::string, size_t> id_hash_{};
  uint64_t max_segment_size_{0};
  std::vector<std::unique_ptr<char[]>> buffer_pool_buffers_{};

  ailego::VecBufferPool::Pointer buffer_pool_{nullptr};
  ailego::VecBufferPoolHandle::Pointer buffer_pool_handle_{nullptr};
  uint64_t current_header_start_offset_{0u};
  uint64_t buffer_size_{2lu * 1024 * 1024 * 1024};  // 2G
};

INDEX_FACTORY_REGISTER_STORAGE(BufferStorage);

}  // namespace core
}  // namespace zvec
