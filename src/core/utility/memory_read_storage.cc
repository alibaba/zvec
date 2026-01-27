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
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_format.h>
#include <zvec/core/framework/index_memory.h>
#include <zvec/core/framework/index_unpacker.h>
#include "utility_params.h"

namespace zvec {
namespace core {

/*! Memory Storage
 */
class MemoryReadStorage : public IndexStorage {
 public:
  /*! Memory Storage Segment
   */
  class Segment : public IndexStorage::Segment,
                  public std::enable_shared_from_this<Segment> {
   public:
    //! Index Storage Pointer
    typedef std::shared_ptr<Segment> Pointer;

    //! Constructor
    Segment(const IndexMemory::Rope::Pointer &rope,
            const IndexUnpacker::SegmentMeta &segment)
        : data_offset_(segment.data_offset()),
          data_size_(segment.data_size()),
          padding_size_(segment.padding_size()),
          region_size_(segment.data_size() + segment.padding_size()),
          data_crc_(segment.data_crc()),
          block_(&(*rope)[0]),
          rope_(rope) {}

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
      return block_->fetch(data_offset_ + offset, buf, len);
    }

    //! Read data from segment
    size_t read(size_t offset, const void **data, size_t len) override {
      if (ailego_unlikely(offset + len > region_size_)) {
        if (offset > region_size_) {
          offset = region_size_;
        }
        len = region_size_ - offset;
      }
      return block_->read(data_offset_ + offset, data, len);
    }

    size_t read(size_t offset, MemoryBlock &data, size_t len) override {
      if (ailego_unlikely(offset + len > region_size_)) {
        if (offset > region_size_) {
          offset = region_size_;
        }
        len = region_size_ - offset;
      }
      const void *data_ptr = nullptr;
      size_t return_value = block_->read(data_offset_ + offset, &data_ptr, len);
      data.reset((void *)data_ptr);
      return return_value;
    }

    //! Read data from segment
    bool read(SegmentData *iovec, size_t count) override {
      for (auto *end = iovec + count; iovec != end; ++iovec) {
        ailego_false_if_false(iovec->offset + iovec->length <= region_size_);
        block_->read(data_offset_ + iovec->offset, &iovec->data, iovec->length);
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
    size_t data_offset_{0u};
    size_t data_size_{0u};
    size_t padding_size_{0u};
    size_t region_size_{0u};
    uint32_t data_crc_{0u};
    IndexMemory::Block *block_{nullptr};
    IndexMemory::Rope::Pointer rope_{};
  };

  //! Destructor
  virtual ~MemoryReadStorage(void) {}

  //! Initialize container
  int init(const ailego::Params &params) override {
    params.get(MEMORY_CONTAINER_CHECKSUM_VALIDATION, &checksum_validation_);
    return 0;
  }

  //! Cleanup container
  int flush(void) override {
    return IndexError_NotImplemented;
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
    rope_ = IndexMemory::Instance()->open(path);
    if (!rope_) {
      LOG_ERROR("Failed to open memory rope %s", path.c_str());
      return IndexError_NoExist;
    }
    if (rope_->empty()) {
      LOG_ERROR("The memory rope %s is empty.", path.c_str());
      return IndexError_NoExist;
    }

    auto read_data = [this](size_t offset, const void **data, size_t len) {
      return (*this->rope_)[0].read(offset, data, len);
    };

    IndexUnpacker unpacker;
    if (!unpacker.unpack(read_data, (*rope_)[0].size(), checksum_validation_)) {
      LOG_ERROR("Failed to unpack memory block: %s", path.c_str());
      return IndexError_UnpackIndex;
    }
    segments_ = std::move(*unpacker.mutable_segments());
    magic_ = unpacker.magic();
    return 0;
  }

  //! Unload all indexes
  int close(void) override {
    rope_ = nullptr;
    segments_.clear();
    return 0;
  }

  //! Retrieve a segment by id
  IndexStorage::Segment::Pointer get(const std::string &id, int) override {
    if (!rope_) {
      return IndexStorage::Segment::Pointer();
    }
    auto it = segments_.find(id);
    if (it == segments_.end()) {
      return IndexStorage::Segment::Pointer();
    }
    return std::make_shared<Segment>(rope_, it->second);
  }

  //! Retrieve all segments
  std::map<std::string, IndexStorage::Segment::Pointer> get_all(
      void) const override {
    std::map<std::string, IndexStorage::Segment::Pointer> result;
    if (rope_) {
      for (const auto &it : segments_) {
        result.emplace(it.first, std::make_shared<Segment>(rope_, it.second));
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
  bool checksum_validation_{false};
  uint32_t magic_{0};
  std::map<std::string, IndexUnpacker::SegmentMeta> segments_{};
  IndexMemory::Rope::Pointer rope_{};
};

INDEX_FACTORY_REGISTER_STORAGE(MemoryReadStorage);

}  // namespace core
}  // namespace zvec
