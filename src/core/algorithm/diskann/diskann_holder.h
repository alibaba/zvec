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

#include <fstream>
#include <zvec/core/framework/index_holder.h>
#include "diskann_entity.h"

namespace zvec {
namespace core {

struct DiskAnnIndexHolderMeta {
  uint32_t element_size_;
  uint32_t key_size_;
  uint32_t sector_size_;
  uint32_t doc_cnt_;
  uint8_t reserve_[];
};

class DiskAnnIndexHolder : public IndexHolder {
 public:
  typedef std::shared_ptr<DiskAnnIndexHolder> Pointer;

 public:
  enum Status { STATUS_UNINITED = 0, STATUS_WRITE = 1, STATUS_READ = 2 };

 public:
  static constexpr uint32_t kDataSectorSize = 128 * 1024;
  static constexpr uint32_t kMetaSectorSize = 4096;

 public:
  inline static uint32_t get_sector_id(uint32_t id, uint32_t sector_vec_num) {
    return id / sector_vec_num;
  }

  inline static uint32_t get_sector_offset(uint32_t id, uint32_t sector_vec_num,
                                           uint32_t data_size) {
    return (id % sector_vec_num) * data_size;
  }

 public:
  /*! Random Access Index Holder Iterator
   */
  class Iterator : public IndexHolder::Iterator {
   public:
    //! Index Holder Iterator Pointer
    typedef std::unique_ptr<Iterator> Pointer;

    //! Constructor
    Iterator(DiskAnnIndexHolder *owner)
        : holder_(owner), sector_id_{0}, sector_offset_{0} {
      path_ = holder_->path();
      data_size_ = holder_->data_size();
      data_sector_size_ = holder_->data_sector_size();
      meta_sector_size_ = holder_->meta_sector_size();

      sector_buffer_.resize(data_sector_size_);

      sector_vec_num_ = data_sector_size_ / data_size_;
    }

    //! Destructor
    virtual ~Iterator(void) {
      if (file_.is_open()) {
        file_.close();
      }
    }

    int init() {
      file_.open(path_, std::ios::in);
      if (!file_.is_open()) {
        LOG_ERROR("file can not create, %s", path_.c_str());
        return IndexError_OpenFile;
      }

      file_.seekg(meta_sector_size_);

      read_sector();

      return 0;
    }

    //! Retrieve pointer of data
    const void *data(void) const override {
      const uint8_t *data_ptr =
          reinterpret_cast<const uint8_t *>(sector_buffer_.data());
      return data_ptr + sector_offset_ + sizeof(diskann_key_t);
    }

    //! Test if the iterator is valid
    bool is_valid(void) const override {
      return id_ < holder_->count();
    }

    //! Retrieve primary key
    uint64_t key(void) const override {
      const uint8_t *data_ptr =
          reinterpret_cast<const uint8_t *>(sector_buffer_.data());
      uint64_t key =
          *reinterpret_cast<const uint64_t *>(data_ptr + sector_offset_);

      return key;
    }

    //! Next iterator
    void next(void) override {
      ++id_;

      uint32_t sector_id = get_sector_id(id_, sector_vec_num_);
      if (sector_id > sector_id_) {
        file_.seekg(sector_id * data_sector_size_ + meta_sector_size_);
        read_sector();
        sector_id_ = sector_id;
      }

      sector_offset_ = get_sector_offset(id_, sector_vec_num_, data_size_);
    }

    int read_sector() {
      file_.read(&((sector_buffer_)[0]), data_sector_size_);
      if (!file_) {
        LOG_ERROR("Failed to read sector from file: %s", path_.c_str());
        return IndexError_ReadData;
      }

      return 0;
    }

   private:
    //! Members
    DiskAnnIndexHolder *holder_{nullptr};
    std::string path_;
    std::ifstream file_;
    uint32_t sector_id_{0};
    std::string sector_buffer_;
    uint32_t id_{0};
    uint32_t data_size_{0};
    uint32_t sector_offset_{0};
    uint32_t data_sector_size_{0};
    uint32_t meta_sector_size_{0};
    uint32_t sector_vec_num_{0};
  };

 public:
  DiskAnnIndexHolder(IndexMeta &meta, std::string &path) {
    path_ = path;

    data_size_ = meta.element_size() + sizeof(diskann_key_t);
    dimension_ = meta.dimension();
    type_ = meta.data_type();

    element_size_ = meta.element_size();
    sector_vec_num_ = data_sector_size_ / data_size_;
    padding_size_ = data_sector_size_ - sector_vec_num_ * data_size_;
    sector_buffer_.resize(data_sector_size_);
    sector_internal_id_ = 0;
  }

  ~DiskAnnIndexHolder() override {
    if (file_.is_open()) {
      file_.close();
    }
  }

  //! Init
  int init() {
    file_.open(path_, std::ios::out | std::ios::trunc);

    if (!file_.is_open()) {
      LOG_ERROR("file can not create, %s", path_.c_str());
      return IndexError_OpenFile;
    }

    DiskAnnIndexHolderMeta holder_meta;
    holder_meta.element_size_ = element_size_;
    holder_meta.key_size_ = sizeof(diskann_key_t);
    holder_meta.sector_size_ = data_sector_size_;

    std::vector<uint8_t> empty_sector;
    empty_sector.resize(meta_sector_size_);

    std::memset(&(empty_sector[0]), 0, meta_sector_size_);
    std::memcpy(&(empty_sector[0]), &holder_meta,
                sizeof(DiskAnnIndexHolderMeta));

    file_.write(reinterpret_cast<const char *>(&(empty_sector[0])),
                meta_sector_size_);
    if (!file_) {
      LOG_ERROR("Failed to write meta sector to file: %s", path_.c_str());
      return IndexError_WriteData;
    }

    status_ = STATUS_WRITE;

    return 0;
  }

  int close() {
    if (sector_internal_id_ != 0) {
      file_.write(reinterpret_cast<const char *>(&(sector_buffer_[0])),
                  data_sector_size_);
    }

    file_.close();

    return 0;
  }

  //! Retrieve count of elements in holder (-1 indicates unknown)
  size_t count(void) const override {
    return count_;
  }

  //! Retrieve dimension
  size_t dimension(void) const override {
    return dimension_;
  }

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return type_;
  }

  //! Retrieve element size in bytes
  size_t element_size(void) const override {
    return element_size_;
  }

  //! Retrieve if it can multi-pass
  bool multipass(void) const override {
    return true;
  }

  //! Create a new iterator
  IndexHolder::Iterator::Pointer create_iterator(void) override {
    auto pointer = std::make_unique<DiskAnnIndexHolder::Iterator>(this);

    if (pointer->init() != 0) {
      return nullptr;
    }

    return pointer;
  }

  int emplace(uint64_t pkey, const void *vec) {
    if (status_ != STATUS_WRITE) {
      return IndexError_NoReady;
    }

    uint8_t *data_ptr = reinterpret_cast<uint8_t *>(&(sector_buffer_[0])) +
                        sector_internal_id_ * data_size_;
    std::memcpy(data_ptr, &pkey, sizeof(diskann_key_t));
    std::memcpy(data_ptr + sizeof(diskann_key_t), vec, element_size_);

    sector_internal_id_++;
    if (sector_internal_id_ >= sector_vec_num_) {
      std::memset(data_ptr + data_size_, 0, padding_size_);

      file_.write(reinterpret_cast<const char *>(&(sector_buffer_[0])),
                  data_sector_size_);

      sector_internal_id_ = 0;
      sector_id_++;
    }

    count_++;

    return 0;
  }

  uint32_t data_sector_size() {
    return data_sector_size_;
  }

  uint32_t meta_sector_size() {
    return meta_sector_size_;
  }

  uint32_t data_size() {
    return data_size_;
  }

  uint32_t *mutable_sector_id() {
    return &sector_id_;
  }

  uint32_t sector_id() {
    return sector_id_;
  }

  std::string &path() {
    return path_;
  }

 private:
  std::string path_;
  std::ofstream file_;
  uint32_t element_size_{0};
  uint32_t dimension_{0};
  IndexMeta::DataType type_{IndexMeta::DataType::DT_UNDEFINED};
  uint32_t sector_vec_num_{0};
  uint32_t data_size_{0};
  uint32_t padding_size_{0};
  uint32_t meta_sector_size_{DiskAnnIndexHolder::kMetaSectorSize};
  uint32_t data_sector_size_{DiskAnnIndexHolder::kDataSectorSize};
  std::string sector_buffer_;
  uint32_t sector_internal_id_{0};
  uint32_t sector_id_{0};
  uint32_t count_{0};
  uint32_t status_{STATUS_UNINITED};
};

}  // namespace core
}  // namespace zvec