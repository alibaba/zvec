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
#include <ailego/utility/memory_helper.h>
#include <zvec/ailego/io/file.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_unpacker.h>
#include "utility_params.h"

namespace zvec {
namespace core {

/*! File Storage
 */
class FileReadStorage : public IndexStorage {
 public:
  /*! File Storage Segment
   */
  class Segment : public IndexStorage::Segment {
   public:
    //! Index Storage Pointer
    typedef std::shared_ptr<Segment> Pointer;

    //! Constructor
    Segment(const FileReadStorage &container,
            const IndexUnpacker::SegmentMeta &segment,
            const std::shared_ptr<ailego::File> &file_ptr, size_t offset)
        : data_offset_(offset + segment.data_offset()),
          data_size_(segment.data_size()),
          padding_size_(segment.padding_size()),
          region_size_(segment.data_size() + segment.padding_size()),
          data_crc_(segment.data_crc()),
          enable_direct_io_(container.enable_direct_io_),
          alone_file_handle_(container.alone_file_handle_),
          file_ptr_(file_ptr) {
      if (alone_file_handle_) {
        file_path_ = container.file_path_;
      }
    }

    //! Constructor
    Segment(const Segment &rhs, const std::shared_ptr<ailego::File> &file_ptr)
        : data_offset_(rhs.data_offset_),
          data_size_(rhs.data_size_),
          padding_size_(rhs.padding_size_),
          region_size_(rhs.region_size_),
          data_crc_(rhs.data_crc_),
          file_ptr_(file_ptr),
          file_path_(rhs.file_path_) {}

    //! Destructor
    virtual ~Segment(void) {}

    //! Retrieve size of data
    size_t data_size(void) const override {
      return data_size_;
    }

    //! Retrieve size of padding
    size_t padding_size(void) const override {
      return padding_size_;
    }

    //! Retrieve crc of data
    uint32_t data_crc(void) const override {
      return data_crc_;
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
      return file_ptr_->read(data_offset_ + offset, buf, len);
    }

    //! Read data from segment
    size_t read(size_t offset, const void **data, size_t len) override {
      if (ailego_unlikely(offset + len > region_size_)) {
        if (offset > region_size_) {
          offset = region_size_;
        }
        len = region_size_ - offset;
      }
      buffer_.reserve(len);
      *data = buffer_.data();
      return file_ptr_->read(data_offset_ + offset, (void *)*data, len);
    }

    size_t read(size_t offset, MemoryBlock &data, size_t len) override {
      if (ailego_unlikely(offset + len > region_size_)) {
        if (offset > region_size_) {
          offset = region_size_;
        }
        len = region_size_ - offset;
      }
      buffer_.reserve(len);
      data.reset(buffer_.data());
      return file_ptr_->read(data_offset_ + offset, (void *)data.data(), len);
    }

    //! Read data from segment
    bool read(SegmentData *iovec, size_t count) override {
      size_t total = 0u;
      for (auto *it = iovec, *end = iovec + count; it != end; ++it) {
        ailego_false_if_false(it->offset + it->length <= region_size_);
        total += it->length;
      }
      ailego_false_if_false(total != 0);

      buffer_.reserve(total);
      uint8_t *buf = buffer_.data();
      for (auto *it = iovec, *end = iovec + count; it != end; ++it) {
        ailego_false_if_false(file_ptr_->read(data_offset_ + it->offset, buf,
                                              it->length) == it->length);
        it->data = buf;
        buf += it->length;
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
      return this->clone_segment<FileReadStorage>();
    }

   protected:
    //! Clone the segment
    template <typename T>
    inline IndexStorage::Segment::Pointer clone_segment(void) {
      auto file_ptr = alone_file_handle_ ? FileReadStorage::OpenFile(
                                               file_path_, enable_direct_io_)
                                         : file_ptr_;
      if (file_ptr) {
        return std::make_shared<typename T::Segment>(
            *(static_cast<typename T::Segment *>(this)), file_ptr);
      }
      return IndexStorage::Segment::Pointer();
    }

   protected:
    size_t data_offset_{0u};
    size_t data_size_{0u};
    size_t padding_size_{0u};
    size_t region_size_{0u};
    uint32_t data_crc_{0u};
    bool enable_direct_io_{false};
    bool alone_file_handle_{false};
    std::vector<uint8_t> buffer_{};
    std::shared_ptr<ailego::File> file_ptr_{};
    std::string file_path_{};
  };

  /*! MMapFile Storage Segment
   */
  class MMapSegment : public Segment,
                      public std::enable_shared_from_this<Segment> {
   public:
    //! Constructor
    MMapSegment(const FileReadStorage &container,
                const IndexUnpacker::SegmentMeta &segment,
                const std::shared_ptr<ailego::File> &file_ptr, size_t offset,
                const void *data, std::function<void()> &&cb)
        : Segment(container, segment, file_ptr, offset),
          data_(static_cast<const char *>(data)),
          cleanup_(std::move(cb)) {
      ailego_assert_with(data_, "Null Pointer");
    }

    virtual ~MMapSegment(void) {
      cleanup_();
    }

    //! Fetch data from segment (with own buffer)
    size_t fetch(size_t offset, void *buf, size_t len) const override {
      if (ailego_unlikely(offset + len > region_size_)) {
        if (offset > region_size_) {
          offset = region_size_;
        }
        len = region_size_ - offset;
      }
      memcpy(buf, data_ + offset, len);
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
      *data = data_ + offset;
      return len;
    }

    size_t read(size_t offset, MemoryBlock &data, size_t len) override {
      if (ailego_unlikely(offset + len > region_size_)) {
        if (offset > region_size_) {
          offset = region_size_;
        }
        len = region_size_ - offset;
      }
      data.reset((void *)(data_ + offset));
      return len;
    }

    //! Read data from segment
    bool read(SegmentData *iovec, size_t count) override {
      for (auto *it = iovec, *end = iovec + count; it != end; ++it) {
        ailego_false_if_false(it->offset + it->length <= region_size_);
        it->data = data_ + it->offset;
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
    const char *data_{nullptr};
    std::function<void()> cleanup_{nullptr};
  };

  //! Destructor
  virtual ~FileReadStorage(void) {}

  //! Initialize container
  int init(const ailego::Params &params) override {
    params.get(FILE_READ_STORAGE_CHECKSUM_VALIDATION, &checksum_validation_);
    params.get(FILE_READ_STORAGE_ENABLE_DIRECT_IO, &enable_direct_io_);
    params.get(FILE_READ_STORAGE_ALONE_FILE_HANDLE, &alone_file_handle_);
    params.get(FILE_READ_STORAGE_MEMORY_LOCKED, &memory_locked_);
    params.get(FILE_READ_STORAGE_MEMORY_WARMUP, &memory_warmup_);
    params.get(FILE_READ_STORAGE_MEMORY_SHARED, &memory_shared_);
    params.get(FILE_READ_STORAGE_HEADER_OFFSET, &header_offset_);
    params.get(FILE_READ_STORAGE_FOOTER_OFFSET, &footer_offset_);
    return 0;
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

  //! Cleanup container
  int cleanup(void) override {
    return this->close();
  }

  //! Load a index file into container
  int open(const std::string &path, bool) override {
    auto file_ptr = FileReadStorage::OpenFile(path, enable_direct_io_);
    if (!file_ptr) {
      return IndexError_OpenFile;
    }

    index_offset_ =
        (header_offset_ >= 0 ? 0 : file_ptr->size()) + header_offset_;
    size_t end_offset =
        (footer_offset_ > 0 ? 0 : file_ptr->size()) + footer_offset_;
    size_t size = end_offset > index_offset_ ? end_offset - index_offset_ : 0;
    auto read_data = [this, &file_ptr, end_offset](
                         size_t offset, const void **data, size_t len) {
      buffer_.reserve(len);
      *data = buffer_.data();
      size_t off = index_offset_ + offset;
      if (off + len > end_offset) {
        if (off > end_offset) {
          off = end_offset;
        }
        len = end_offset - off;
      }
      return file_ptr->read(off, (void *)*data, len);
    };

    IndexUnpacker unpacker;
    if (!unpacker.unpack(read_data, size, checksum_validation_)) {
      LOG_ERROR("Failed to unpack file: %s", path.c_str());
      return IndexError_UnpackIndex;
    }
    segments_ = std::move(*unpacker.mutable_segments());
    magic_ = unpacker.magic();
    file_path_ = path;
    file_ptr_ = alone_file_handle_ ? nullptr : file_ptr;
    return 0;
  }

  int close(void) override {
    file_ptr_ = nullptr;
    segments_.clear();
    return 0;
  }

  //! Retrieve a segment by id
  IndexStorage::Segment::Pointer get(const std::string &id,
                                     int level) override {
    return level == 0 ? this->get_mmap_segment<FileReadStorage>(id)
                      : this->get_segment<FileReadStorage>(id);
  }

  //! Retrieve all segments
  std::map<std::string, IndexStorage::Segment::Pointer> get_all(
      void) const override {
    std::map<std::string, IndexStorage::Segment::Pointer> result;
    auto file_ptr =
        alone_file_handle_ && !file_path_.empty()
            ? FileReadStorage::OpenFile(file_path_, enable_direct_io_)
            : file_ptr_;
    if (file_ptr) {
      for (const auto &it : segments_) {
        result.emplace(it.first,
                       std::make_shared<FileReadStorage::Segment>(
                           *(static_cast<const FileReadStorage *>(this)),
                           it.second, file_ptr, index_offset_));
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

 protected:
  //! Open a index file
  static inline std::shared_ptr<ailego::File> OpenFile(const std::string &path,
                                                       bool direct_io) {
    auto file_ptr = std::make_shared<ailego::File>();
    if (!file_ptr) {
      LOG_ERROR("Failed to create file object, errno %d, %s", errno,
                std::strerror(errno));
      return nullptr;
    }
    if (!file_ptr->open(path, true, direct_io)) {
      LOG_ERROR("Failed to open file %s, errno %d, %s", path.c_str(), errno,
                std::strerror(errno));
      return nullptr;
    }
    return file_ptr;
  }

  //! Retrieve a segment by id
  template <typename T>
  inline IndexStorage::Segment::Pointer get_segment(
      const std::string &id) const {
    auto it = segments_.find(id);
    if (it == segments_.end()) {
      return IndexStorage::Segment::Pointer();
    }
    auto file_ptr =
        alone_file_handle_ && !file_path_.empty()
            ? FileReadStorage::OpenFile(file_path_, enable_direct_io_)
            : file_ptr_;
    if (!file_ptr) {
      return IndexStorage::Segment::Pointer();
    }
    return std::make_shared<typename T::Segment>(
        *(static_cast<const T *>(this)), it->second, file_ptr, index_offset_);
  }

  //! Retrieve a mmap segment by id
  template <typename T>
  inline IndexStorage::Segment::Pointer get_mmap_segment(
      const std::string &id) const {
    auto it = segments_.find(id);
    if (it == segments_.end()) {
      return IndexStorage::Segment::Pointer();
    }
    const auto &segment = it->second;
    auto file_ptr =
        alone_file_handle_ && !file_path_.empty()
            ? FileReadStorage::OpenFile(file_path_, enable_direct_io_)
            : file_ptr_;
    if (!file_ptr) {
      return IndexStorage::Segment::Pointer();
    }

    int opt = memory_locked_ ? ailego::File::MMAP_LOCKED : 0;
    opt |= memory_warmup_ ? ailego::File::MMAP_WARMUP : 0;
    opt |= memory_shared_ ? ailego::File::MMAP_SHARED : 0;
    size_t size = segment.data_size() + segment.padding_size();
    size_t segment_offset = index_offset_ + segment.data_offset();
    size_t offset = segment_offset / ailego::MemoryHelper::PageSize() *
                    ailego::MemoryHelper::PageSize();
    size_t bias = segment_offset - offset;

    size += bias;
    void *data = file_ptr->map(offset, size, opt);
    if (data == nullptr) {
      LOG_ERROR("Failed to mmap file: %s, offset: %zu, size: %zu",
                file_path_.c_str(), offset, size);
      return IndexStorage::Segment::Pointer();
    }
    return std::make_shared<typename T::MMapSegment>(
        *(static_cast<const T *>(this)), segment, file_ptr, index_offset_,
        static_cast<char *>(data) + bias,
        [=]() { ailego::File::MemoryUnmap(data, size); });
  }

  //! Retrieve all segments
  template <typename T>
  inline std::map<std::string, IndexStorage::Segment::Pointer> get_all_segments(
      void) const {
    std::map<std::string, IndexStorage::Segment::Pointer> result;
    auto file_ptr =
        alone_file_handle_ && !file_path_.empty()
            ? FileReadStorage::OpenFile(file_path_, enable_direct_io_)
            : file_ptr_;
    if (file_ptr) {
      for (const auto &it : segments_) {
        result.emplace(it.first, std::make_shared<typename T::Segment>(
                                     *(static_cast<const T *>(this)), it.second,
                                     file_ptr, index_offset_));
      }
    }
    return result;
  }

 protected:
  bool checksum_validation_{false};
  bool enable_direct_io_{false};
  bool alone_file_handle_{false};
  bool memory_locked_{false};
  bool memory_warmup_{false};
  bool memory_shared_{false};
  uint32_t magic_{0};
  int64_t header_offset_{0};
  int64_t footer_offset_{0};
  size_t index_offset_{0};
  std::vector<uint8_t> buffer_{};
  std::map<std::string, IndexUnpacker::SegmentMeta> segments_{};
  std::shared_ptr<ailego::File> file_ptr_{nullptr};
  std::string file_path_{};
};

INDEX_FACTORY_REGISTER_STORAGE(FileReadStorage);

}  // namespace core
}  // namespace zvec
