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
#include <limits>
#include <mutex>
#include <ailego/pattern/defer.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_mapping.h>
#include <zvec/core/framework/index_version.h>
#include "utility_params.h"

namespace zvec {
namespace core {

/*! MMap File Storage
 */
class MMapFileStorage : public IndexStorage {
 public:
  /*! Index Storage Segment
   */
  class Segment : public IndexStorage::Segment,
                  public std::enable_shared_from_this<Segment> {
   public:
    //! Index Storage Pointer
    typedef std::shared_ptr<Segment> Pointer;

    //! Constructor
    Segment(MMapFileStorage *owner, IndexMapping::Segment *segment)
        : segment_(segment),
          owner_(owner),
          capacity_(static_cast<size_t>(segment->meta()->data_size +
                                        segment->meta()->padding_size)) {}

    //! Destructor
    ~Segment() override = default;

    //! Retrieve size of data
    size_t data_size() const override {
      return static_cast<size_t>(segment_->meta()->data_size);
    }

    //! Retrieve crc of data
    uint32_t data_crc() const override {
      return segment_->meta()->data_crc;
    }

    //! Retrieve size of padding
    size_t padding_size() const override {
      return static_cast<size_t>(segment_->meta()->padding_size);
    }

    //! Retrieve capacity of segment
    size_t capacity() const override {
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
      memmove(buf, (const uint8_t *)segment_->data() + offset, len);
      return len;
    }

    //! Read data from segment
    size_t read(size_t offset, MemoryBlock &data, size_t len) override {
      if (ailego_unlikely(offset + len > segment_->meta()->data_size)) {
        auto meta = segment_->meta();
        if (offset > meta->data_size) {
          offset = meta->data_size;
        }
        len = meta->data_size - offset;
      }
      data.reset((uint8_t *)segment_->data() + offset);
      return len;
    }

    //! Write data into the storage with offset
    size_t write(size_t offset, const void *data, size_t len) override {
      size_t data_tail = offset + len;
      ailego_zero_if_false(data_tail <= capacity_);
      auto meta = segment_->meta();
      if (data_tail > meta->data_size) {
        meta->data_size = data_tail;
        meta->padding_size = capacity_ - data_tail;
      }
      owner_->set_as_dirty();
      memmove((uint8_t *)segment_->data() + offset, data, len);
      segment_->set_dirty();
      return len;
    }

    int prepare_write_batch(const SegmentData *writes, size_t count,
                            std::unique_ptr<PreparedWrite> *out) override try {
      if (!out || *out || (!writes && count)) return IndexError_InvalidArgument;
      size_t bytes = 0;
      for (size_t i = 0; i < count; ++i) {
        const auto &write = writes[i];
        if (write.offset > capacity_ ||
            write.length > capacity_ - write.offset ||
            (write.length && !write.data) ||
            write.length > std::numeric_limits<size_t>::max() - bytes) {
          return IndexError_InvalidArgument;
        }
        bytes += write.length;
      }
      auto pending = std::make_unique<PreparedBatch>(shared_from_this());
      pending->payload = MemoryBlock::AllocateOwned(bytes);
      pending->writes.reserve(count);
      auto *cursor = static_cast<char *>(pending->payload.data_);
      for (size_t i = 0; i < count; ++i) {
        const auto &write = writes[i];
        if (!write.length) continue;
        std::memcpy(cursor, write.data, write.length);
        pending->writes.push_back({write.offset, write.length, cursor});
        cursor += write.length;
      }
      *out = std::move(pending);
      return 0;
    } catch (const std::bad_alloc &) {
      return IndexError_NoMemory;
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
    IndexStorage::Segment::Pointer clone() override {
      return shared_from_this();
    }

    //! Stable base data pointer — valid for the lifetime of the mmap.
    const uint8_t *base_data() const override {
      return (const uint8_t *)segment_->data();
    }

   private:
    class PreparedBatch final : public PreparedWrite {
     public:
      explicit PreparedBatch(std::shared_ptr<Segment> segment)
          : segment_(std::move(segment)) {}

      void commit() noexcept override {
        for (const auto &write : writes) {
          segment_->write(write.offset, write.data, write.length);
        }
        writes.clear();
        payload.reset();
      }

      MemoryBlock payload;
      ailego::BudgetedVector<SegmentData> writes;

     private:
      std::shared_ptr<Segment> segment_;
    };

    IndexMapping::Segment *segment_{};
    MMapFileStorage *owner_{nullptr};
    size_t capacity_{};
  };

  //! Destructor
  ~MMapFileStorage() override {
    this->cleanup();
  }

  //! Initialize storage
  int init(const ailego::Params &params) override {
    uint32_t val = params.get_as_uint32(MMAPFILE_STORAGE_SEGMENT_META_CAPACITY);
    if (val != 0) {
      segment_meta_capacity_ = val;
    }
    params.get(MMAPFILE_STORAGE_COPY_ON_WRITE, &copy_on_write_);
    params.get(MMAPFILE_STORAGE_FORCE_FLUSH, &force_flush_);
    params.get(MMAPFILE_STORAGE_MEMORY_LOCKED, &memory_locked_);
    params.get(MMAPFILE_STORAGE_MEMORY_WARMUP, &memory_warmup_);
    return 0;
  }

  //! Cleanup storage
  int cleanup() override {
    this->close_index();
    return 0;
  }

  //! Open storage
  int open(const std::string &path, bool create_if_missing) override {
    return open_index(path, OpenOptions{create_if_missing, false});
  }

  int open_with_options(const std::string &path,
                        const OpenOptions &options) override {
    if (options.read_only ||
        (!options.created_file && !options.defer_publication)) {
      return IndexError_Unsupported;
    }
    return open_index(path, options);
  }

  int open_index(const std::string &path, const OpenOptions &options) try {
    if (mapping_.index_size() != 0) return IndexError_OpenFile;
    std::unique_ptr<ailego::StagedPath> pending;
    ailego::File::Identity created_file;
    ailego::File::Identity private_identity;
    bool opened = false;
    AILEGO_DEFER([&] {
      if (!opened && (created_file.valid || private_identity.valid)) {
        close_index();
        if (created_file.valid &&
            !ailego::File::DeleteIfSame(path, created_file)) {
          LOG_ERROR("Failed to remove incomplete index file %s", path.c_str());
        }
      }
    });
    if (!ailego::File::IsExist(path) && options.create_if_missing) {
      if (ailego::StagedPath::supported()) {
        pending = std::make_unique<ailego::StagedPath>();
        if (!pending->prepare(path)) return IndexError_CreateFile;
      }
      if (!pending) {
        const size_t last_slash = path.rfind('/');
        if (last_slash != std::string::npos) {
          ailego::File::MakePath(path.substr(0, last_slash));
        }
      }
      int error_code =
          pending ? this->init_index(pending->path(), &private_identity)
                  : this->init_index(path, &created_file);
      if (error_code != 0) {
        return error_code;
      }
    }
    int ret = mapping_.open(pending ? pending->path() : path, copy_on_write_,
                            force_flush_);
    if (ret == 0 && created_file.valid &&
        !(mapping_.file_identity() == created_file)) {
      LOG_ERROR("Index file was replaced during creation %s", path.c_str());
      return IndexError_OpenFile;
    }
    if (ret != 0) return ret;
    staged_file_ = std::move(pending);
    if (!options.defer_publication) {
      const bool staged = static_cast<bool>(staged_file_);
      ret = publish_creation();
      if (ret != 0) {
        close_index();
        return ret;
      }
      if (staged) created_file = mapping_.file_identity();
    }
    opened = true;
    if (options.created_file) *options.created_file = created_file;
    return 0;
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  }

  int publish_creation() override {
    if (!staged_file_) return 0;
    const bool was_dirty = is_dirty();
    const int ret = flush_index();
    if (ret != 0) return ret;
    if (!staged_file_->publish()) return IndexError_CreateFile;
    staged_file_.reset();
    index_dirty_ = was_dirty;
    return 0;
  }

  //! Flush storage
  int flush() override {
    return this->flush_index();
  }

  //! Close storage
  int close() override {
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
  uint64_t check_point() const override {
    return mapping_.footer().check_point;
  }

  //! Retrieve a segment by id
  IndexStorage::Segment::Pointer get(const std::string &id, int) override {
    IndexMapping::Segment *segment = this->get_segment(id);
    if (!segment) {
      return MMapFileStorage::Segment::Pointer();
    }
    return std::make_shared<MMapFileStorage::Segment>(this, segment);
  }

  //! Test if it a segment exists
  bool has(const std::string &id) const override {
    return this->has_segment(id);
  }

  //! Retrieve magic number of index
  uint32_t magic() const override {
    return mapping_.magic();
  }

 protected:
  //! Initialize index version segment
  int init_version_segment() {
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
    set_as_dirty();
    meta->data_crc = ailego::Crc32c::Hash(segment->data(), data_size, 0);
    meta->data_size = data_size;
    meta->padding_size = capacity - data_size;
    return 0;
  }

  //! Initialize index file
  int init_index(const std::string &path,
                 ailego::File::Identity *created_file) {
    int error_code = mapping_.create_exclusive(path, segment_meta_capacity_);
    if (error_code != 0) {
      return error_code;
    }
    *created_file = mapping_.file_identity();

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

  bool is_huge_page() const override {
    return mapping_.huge_page();
  }

  bool is_dirty() const override {
    return index_dirty_ || mapping_.is_header_dirty();
  }

  //! Set the index file as dirty
  void set_as_dirty() {
    index_dirty_ = true;
  }

  //! Refresh meta information (checksum, update time, etc.)
  void refresh_index(uint64_t chkp) {
    mapping_.refresh(chkp);
    index_dirty_ = false;
  }

  //! Flush index storage
  int flush_index() {
    if (index_dirty_) {
      this->refresh_index(0);
    }
    std::lock_guard<std::mutex> latch(mapping_mutex_);
    return mapping_.flush();
  }

  //! Close index storage
  void close_index() {
    if (index_dirty_) {
      this->refresh_index(0);
    }
    std::lock_guard<std::mutex> latch(mapping_mutex_);
    mapping_.close();
    staged_file_.reset();
  }

  //! Append a segment into storage
  int append_segment(const std::string &id, size_t size) {
    std::lock_guard<std::mutex> latch(mapping_mutex_);
    return mapping_.append(id, size);
  }

  //! Test if a segment exists
  bool has_segment(const std::string &id) const {
    std::lock_guard<std::mutex> latch(mapping_mutex_);
    return mapping_.has(id);
  }

  //! Get a segment from storage
  IndexMapping::Segment *get_segment(const std::string &id) {
    std::lock_guard<std::mutex> latch(mapping_mutex_);
    return mapping_.map(id, memory_warmup_, memory_locked_);
  }

 private:
  uint32_t segment_meta_capacity_{1024 * 1024};
  bool copy_on_write_{false};
  bool force_flush_{false};
  bool memory_locked_{false};
  bool memory_warmup_{false};
  bool index_dirty_{false};
  std::unique_ptr<ailego::StagedPath> staged_file_;
  mutable IndexMapping mapping_{};
  mutable std::mutex mapping_mutex_{};
};

INDEX_FACTORY_REGISTER_STORAGE(MMapFileStorage);

}  // namespace core
}  // namespace zvec
