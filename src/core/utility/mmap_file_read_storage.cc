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
#include <cerrno>
#include <zvec/ailego/io/mmap_file.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_format.h>
#include <zvec/core/framework/index_unpacker.h>
#include "utility_params.h"

namespace zvec {
namespace core {

/*! MMap File Storage
 */
class MMapFileReadStorage : public IndexStorage {
 public:
  /*! MMap File Storage Segment
   */
  class Segment : public IndexStorage::Segment,
                  public std::enable_shared_from_this<Segment> {
   public:
    //! Index Storage Pointer
    typedef std::shared_ptr<Segment> Pointer;

    //! Constructor
    Segment(const std::shared_ptr<ailego::MMapFile> &file_ptr, size_t offset,
            const IndexUnpacker::SegmentMeta &segment)
        : data_ptr_(reinterpret_cast<uint8_t *>(file_ptr->region()) + offset +
                    segment.data_offset()),
          data_size_(segment.data_size()),
          padding_size_(segment.padding_size()),
          region_size_(segment.data_size() + segment.padding_size()),
          data_crc_(segment.data_crc()),
          file_ptr_(file_ptr) {}

    //! Destructor
    virtual ~Segment(void) {}

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

    size_t capacity(void) const override {
      return region_size_;
    }

    //! Fetch data from segment (with own buffer)
    size_t fetch(size_t offset, void *buf, size_t len) const override {
      if (ailego_unlikely(offset + len > region_size_)) {
        if (offset > region_size_) {
          offset = region_size_;
        }
        len = region_size_ - offset;
      }
      memcpy(buf, data_ptr_ + offset, len);
      return len;
    }

    //! Read data from segment
    size_t read(size_t offset, const void **data, size_t len) override {
      if (ailego_unlikely(offset + len > region_size_)) {
        if (offset > region_size_) {
          offset = region_size_;
        }
        len = region_size_ - offset;
      }
      *data = data_ptr_ + offset;
      return len;
    }

    size_t read(size_t offset, MemoryBlock &data, size_t len) override {
      if (ailego_unlikely(offset + len > region_size_)) {
        if (offset > region_size_) {
          offset = region_size_;
        }
        len = region_size_ - offset;
      }
      data.reset((void *)(data_ptr_ + offset));
      return len;
    }

    //! Read data from segment
    bool read(SegmentData *iovec, size_t count) override {
      for (auto *end = iovec + count; iovec != end; ++iovec) {
        ailego_false_if_false(iovec->offset + iovec->length <= region_size_);
        iovec->data = data_ptr_ + iovec->offset;
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
      return shared_from_this();
    }

   private:
    const uint8_t *data_ptr_{nullptr};
    size_t data_size_{0u};
    size_t padding_size_{0u};
    size_t region_size_{0u};
    uint32_t data_crc_{0u};
    std::shared_ptr<ailego::MMapFile> file_ptr_{nullptr};
  };

  //! Destructor
  virtual ~MMapFileReadStorage(void) {}

  //! Initialize container
  int init(const ailego::Params &params) override {
    params.get(MMAPFILE_READ_STORAGE_MEMORY_LOCKED, &memory_locked_);
    params.get(MMAPFILE_READ_STORAGE_MEMORY_WARMUP, &memory_warmup_);
    params.get(MMAPFILE_READ_STORAGE_MEMORY_SHARED, &memory_shared_);
    params.get(MMAPFILE_READ_STORAGE_CHECKSUM_VALIDATION,
               &checksum_validation_);
    params.get(MMAPFILE_READ_STORAGE_HEADER_OFFSET, &header_offset_);
    params.get(MMAPFILE_READ_STORAGE_FOOTER_OFFSET, &footer_offset_);
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

  //! Load a index file into container
  int open(const std::string &path, bool) override {
    file_ptr_ = std::make_shared<ailego::MMapFile>();
    if (!file_ptr_) {
      LOG_ERROR("Failed to create mmap file object, errno %d, %s", errno,
                std::strerror(errno));
      return IndexError_NoMemory;
    }

    if (!file_ptr_->open(path.c_str(), true, memory_shared_)) {
      LOG_ERROR("Failed to open file %s, errno %d, %s", path.c_str(), errno,
                std::strerror(errno));
      return IndexError_OpenFile;
    }

    index_offset_ =
        (header_offset_ >= 0 ? 0 : file_ptr_->size()) + header_offset_;
    size_t end_offset =
        (footer_offset_ > 0 ? 0 : file_ptr_->size()) + footer_offset_;
    size_t size = end_offset > index_offset_ ? end_offset - index_offset_ : 0;
    if (memory_locked_ && !file_ptr_->lock()) {
      LOG_WARN("Failed to lock pages with size %zu, errno %d, %s",
               file_ptr_->size(), errno, std::strerror(errno));
    }
    if (memory_warmup_ && !checksum_validation_) {
      ailego::File::MemoryWarmup(
          static_cast<char *>(file_ptr_->region()) + index_offset_, size);
    }

    auto read_data = [this, end_offset](size_t offset, const void **data,
                                        size_t len) {
      size_t off = offset + index_offset_;
      if (off + len > end_offset) {
        if (off > end_offset) {
          off = end_offset;
        }
        len = end_offset - off;
      }
      *data = (uint8_t *)file_ptr_->region() + off;
      return len;
    };

    IndexUnpacker unpacker;
    if (!unpacker.unpack(read_data, size, checksum_validation_)) {
      LOG_ERROR("Failed to unpack file: %s", path.c_str());
      return IndexError_UnpackIndex;
    }
    segments_ = std::move(*unpacker.mutable_segments());
    magic_ = unpacker.magic();
    return 0;
  }

  int close(void) override {
    file_ptr_ = nullptr;
    segments_.clear();
    return 0;
  }

  //! Retrieve a segment by id
  IndexStorage::Segment::Pointer get(const std::string &id, int) override {
    if (!file_ptr_) {
      return IndexStorage::Segment::Pointer();
    }
    auto it = segments_.find(id);
    if (it == segments_.end()) {
      return IndexStorage::Segment::Pointer();
    }
    return std::make_shared<MMapFileReadStorage::Segment>(
        file_ptr_, index_offset_, it->second);
  }

  std::map<std::string, IndexStorage::Segment::Pointer> get_all(
      void) const override {
    std::map<std::string, IndexStorage::Segment::Pointer> result;
    if (file_ptr_) {
      for (const auto &it : segments_) {
        result.emplace(it.first, std::make_shared<MMapFileReadStorage::Segment>(
                                     file_ptr_, index_offset_, it.second));
      }
    }
    return result;
  }

  //! Test if it a segment exists
  bool has(const std::string &id) const override {
    return (segments_.find(id) != segments_.end());
  }

  //! Retrieve magic number of index
  uint32_t magic(void) const override {
    return magic_;
  }

 private:
  bool memory_locked_{false};
  bool memory_warmup_{false};
  bool memory_shared_{false};
  bool checksum_validation_{false};
  int64_t header_offset_{0};
  int64_t footer_offset_{0};
  size_t index_offset_{0};
  uint32_t magic_{0};
  std::map<std::string, IndexUnpacker::SegmentMeta> segments_{};
  std::shared_ptr<ailego::MMapFile> file_ptr_{nullptr};
};

INDEX_FACTORY_REGISTER_STORAGE(MMapFileReadStorage);

}  // namespace core
}  // namespace zvec