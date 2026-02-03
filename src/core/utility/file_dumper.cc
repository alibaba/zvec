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
#include <zvec/ailego/io/file.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_format.h>
#include <zvec/core/framework/index_packer.h>

namespace zvec {
namespace core {

/*! File Dumper
 */
struct FileDumper : public IndexDumper {
 public:
  //! Constructor
  FileDumper(void) {}

  //! Destructor
  virtual ~FileDumper(void) {
    this->cleanup();
  }

  //! Initialize dumper
  int init(const ailego::Params &) override {
    return 0;
  }

  //! Cleanup dumper
  int cleanup(void) override {
    if (!this->close_index()) {
      return IndexError_PackIndex;
    }
    return 0;
  }

  //! Create a file for dumping
  int create(const std::string &path) override {
    size_t last_slash = path.rfind('/');
    if (last_slash != std::string::npos) {
      ailego::File::MakePath(path.substr(0, last_slash));
    }

    if (!file_.create(path.c_str(), sizeof(IndexFormat::MetaHeader))) {
      LOG_ERROR("Failed to create file %s, errno %d, %s", path.c_str(), errno,
                std::strerror(errno));
      return IndexError_CreateFile;
    }

    auto write_data = [this](const void *buf, size_t size) {
      return this->file_.write(buf, size);
    };
    if (!packer_.setup(write_data)) {
      LOG_ERROR("Failed to setup index package, errno %d, %s", errno,
                std::strerror(errno));
      return IndexError_WriteData;
    }
    return 0;
  }

  //! Close file
  int close(void) override {
    if (!this->close_index()) {
      return IndexError_PackIndex;
    }
    return 0;
  }

  //! Append a segment meta into table
  int append(const std::string &id, size_t data_size, size_t padding_size,
             uint32_t crc) override {
    stab_.emplace_back(id, data_size, padding_size, crc);
    return 0;
  }

  //! Write data to the storage
  size_t write(const void *data, size_t len) override {
    return packer_.pack(
        [this](const void *buf, size_t size) {
          return this->file_.write(buf, size);
        },
        data, len);
  }

  //! Retrieve magic number of index
  uint32_t magic(void) const override {
    return packer_.magic();
  }

 protected:
  //! Close index file
  bool close_index(void) {
    if (file_.is_valid()) {
      auto write_data = [this](const void *buf, size_t size) {
        return this->file_.write(buf, size);
      };

      if (!packer_.finish(write_data, stab_)) {
        LOG_ERROR("Failed to finish packing index package");
        return false;
      }
      stab_.clear();
      file_.close();
      packer_.reset();
    }
    return true;
  }

 private:
  std::vector<IndexPacker::SegmentMeta> stab_{};
  ailego::File file_{};
  IndexPacker packer_{};
};

INDEX_FACTORY_REGISTER_DUMPER(FileDumper);

}  // namespace core
}  // namespace zvec
