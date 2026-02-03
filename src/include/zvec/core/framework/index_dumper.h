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
#include <zvec/core/framework/index_packer.h>

namespace zvec {
namespace core {

/*! Index Dumper
 */
class IndexDumper : public IndexModule {
 public:
  //! Index Dumper Pointer
  typedef std::shared_ptr<IndexDumper> Pointer;

  //! Destructor
  virtual ~IndexDumper(void) {}

  //! Initialize dumper
  virtual int init(const ailego::Params &params) = 0;

  //! Cleanup dumper
  virtual int cleanup(void) = 0;

  //! Create a file for dumping
  virtual int create(const std::string &path) = 0;

  //! Close file
  virtual int close(void) = 0;

  //! Append a segment meta into table
  virtual int append(const std::string &id, size_t data_size,
                     size_t padding_size, uint32_t crc) = 0;

  //! Write data to the storage
  virtual size_t write(const void *data, size_t len) = 0;

  //! Retrieve magic number of index
  virtual uint32_t magic(void) const = 0;
};

/*! Index Segment Dumper
 */
class IndexSegmentDumper : public IndexDumper {
 public:
  //! Index Segment Dumper Pointer
  typedef std::shared_ptr<IndexSegmentDumper> Pointer;

  //! Constructor
  IndexSegmentDumper(IndexDumper::Pointer dumper, std::string segid)
      : segment_id_(std::move(segid)), dumper_(std::move(dumper)) {}

  //! Destructor
  virtual ~IndexSegmentDumper(void) {
    this->close_index();
  }

  //! Initialize dumper
  int init(const ailego::Params &) override {
    return 0;
  }

  //! Cleanup dumper
  int cleanup(void) override {
    return 0;
  }

  //! Create a file for dumping
  int create(const std::string &segid) override {
    if (dumped_size_ != 0) {
      return IndexError_NoReady;
    }

    auto write_data = [&](const void *buf, size_t size) {
      return this->write_to_dumper(buf, size);
    };
    if (!packer_.setup(write_data)) {
      return IndexError_WriteData;
    }
    segment_id_ = segid;
    return 0;
  }

  //! Close file
  int close(void) override {
    return this->close_index();
  }

  //! Append a segment meta into table
  int append(const std::string &id, size_t data_size, size_t padding_size,
             uint32_t crc) override {
    stab_.emplace_back(id, data_size, padding_size, crc);
    return 0;
  }

  //! Write data to the storage
  size_t write(const void *data, size_t len) override {
    auto write_data = [&](const void *buf, size_t size) {
      return this->write_to_dumper(buf, size);
    };

    if (dumped_size_ == 0 && !packer_.setup(write_data)) {
      return 0;
    }
    return packer_.pack(write_data, data, len);
  }

  //! Retrieve magic number of index
  uint32_t magic(void) const override {
    return packer_.magic();
  }

 protected:
  //! Write data to dumper
  size_t write_to_dumper(const void *data, size_t len) {
    size_t wrlen = dumper_->write(data, len);
    dumped_size_ += wrlen;
    return wrlen;
  }

  //! Close index file
  int close_index(void) {
    if (dumped_size_ == 0) {
      return 0;
    }

    auto write_data = [&](const void *buf, size_t size) {
      return this->write_to_dumper(buf, size);
    };
    if (!packer_.finish(write_data, stab_)) {
      return IndexError_WriteData;
    }
    stab_.clear();

    int ret = dumper_->append(segment_id_, dumped_size_, 0, 0);
    dumped_size_ = 0u;
    return ret;
  }

 private:
  size_t dumped_size_{0};
  std::string segment_id_{};
  IndexDumper::Pointer dumper_{};
  IndexPacker packer_{};
  std::vector<IndexPacker::SegmentMeta> stab_{};
};

}  // namespace core
}  // namespace zvec
