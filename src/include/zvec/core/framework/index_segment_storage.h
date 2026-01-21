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
#pragma once

#include <zvec/ailego/container/params.h>
#include <zvec/core/framework/index_module.h>
#include <zvec/core/framework/index_storage.h>
#include <zvec/core/framework/index_unpacker.h>

namespace zvec {
namespace core {

/*! Index Segment Container
 */
class IndexSegmentStorage : public IndexStorage {
 public:
  //! Index Segment Container Pointer
  typedef std::shared_ptr<IndexSegmentStorage> Pointer;

  /*! Index Container Segment
   */
  class Segment : public IndexStorage::Segment,
                  public std::enable_shared_from_this<Segment> {
   public:
    //! Index Container Pointer
    typedef std::shared_ptr<Segment> Pointer;

    //! Constructor
    Segment(const Segment &rhs) = delete;

    //! Constructor
    Segment(const IndexStorage::Segment::Pointer &parent,
            const IndexUnpacker::SegmentMeta &segment)
        : data_offset_(segment.data_offset()),
          data_size_(segment.data_size()),
          padding_size_(segment.padding_size()),
          region_size_(segment.data_size() + segment.padding_size()),
          data_crc_(segment.data_crc()),
          parent_(parent->clone()) {}

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
      return parent_->fetch(data_offset_ + offset, buf, len);
    }

    //! Read data from segment
    size_t read(size_t offset, const void **data, size_t len) override {
      return parent_->read(data_offset_ + offset, data, len);
    }

    size_t read(size_t offset, MemoryBlock &data, size_t len) override {
      const void **data_ptr = nullptr;
      size_t ret = parent_->read(data_offset_ + offset, data_ptr, len);
      data.reset((void *)*data_ptr);
      return ret;
    }

    //! Read data from segment
    bool read(SegmentData *iovec, size_t count) override {
      for (SegmentData *it = iovec, *end = iovec + count; it != end; ++it) {
        it->offset += data_offset_;
      }
      bool success = parent_->read(iovec, count);
      for (SegmentData *it = iovec, *end = iovec + count; it != end; ++it) {
        it->offset -= data_offset_;
      }
      return success;
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
    IndexStorage::Segment::Pointer parent_{nullptr};
  };

  //! Constructor
  IndexSegmentStorage(IndexStorage::Segment::Pointer &&seg)
      : parent_(std::move(seg)) {}

  //! Constructor
  IndexSegmentStorage(const IndexStorage::Segment::Pointer &seg)
      : parent_(seg) {}

  //! Destructor
  virtual ~IndexSegmentStorage(void) {}

  //! Initialize container
  int init(const ailego::Params &) override {
    return 0;
  }

  //! Cleanup container
  int cleanup(void) override {
    return 0;
  }

  //! Load the current segment, ignore path
  int open(const std::string &, bool) override {
    if (!parent_) {
      LOG_ERROR("Failed to load an empty segment");
      return IndexError_NoReady;
    }

    auto read_data = [this](size_t offset, const void **data, size_t len) {
      return this->parent_->read(offset, data, len);
    };

    IndexUnpacker unpacker;
    if (!unpacker.unpack(read_data, parent_->data_size(), false)) {
      LOG_ERROR("Failed to unpack segment data");
      return IndexError_UnpackIndex;
    }
    segments_ = std::move(*unpacker.mutable_segments());
    magic_ = unpacker.magic();
    return 0;
  }

  //! Retrieve a segment by id
  IndexStorage::Segment::Pointer get(const std::string &id, int) override {
    if (!parent_) {
      return IndexStorage::Segment::Pointer();
    }
    auto it = segments_.find(id);
    if (it == segments_.end()) {
      return IndexStorage::Segment::Pointer();
    }
    return std::make_shared<IndexSegmentStorage::Segment>(parent_, it->second);
  }

  //! Test if it a segment exists
  bool has(const std::string &id) const override {
    return (segments_.find(id) != segments_.end());
  }

  //! Retrieve all segments
  std::map<std::string, IndexStorage::Segment::Pointer> get_all(
      void) const override {
    std::map<std::string, IndexStorage::Segment::Pointer> result;
    if (parent_) {
      for (const auto &it : segments_) {
        result.emplace(it.first, std::make_shared<IndexSegmentStorage::Segment>(
                                     parent_, it.second));
      }
    }
    return result;
  }

  //! Unload all indexes
  int close(void) override {
    parent_ = nullptr;
    segments_.clear();
    return 0;
  }

  //! Retrieve magic number of index
  uint32_t magic(void) const override {
    return magic_;
  }

  int flush(void) override {
    return IndexError_NotImplemented;
  }

  int append(const std::string & /*id*/, size_t /*size*/) override {
    return IndexError_NotImplemented;
  }

  void refresh(uint64_t) override {
    return;
  }

  uint64_t check_point(void) const override {
    return 0;
  }

 private:
  uint32_t magic_{0};
  std::map<std::string, IndexUnpacker::SegmentMeta> segments_{};
  IndexStorage::Segment::Pointer parent_{};
};

}  // namespace core
}  // namespace zvec
