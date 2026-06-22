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

#include <iostream>
#include <ailego/io/mmap_file.h>
#include <framework/index_framework.h>
#include <zvec/core/framework/index_holder.h>

namespace zvec {
namespace core {

enum VecsBitMapIndex {
  BITMAP_INDEX_KEY = 0,
  BITMAP_INDEX_DENSE = 1,
  BITMAP_INDEX_SPARSE = 2,
  BITMAP_INDEX_PARTITION = 3,
  BITMAP_INDEX_TAGLIST = 4
};

#pragma pack(4)
struct VecsHeaderDetect {
  uint64_t num_vecs;
  uint16_t reserved;  // in v0, it's meta size
  uint16_t version;   // 0 for v0 format, 1 for new format
  uint8_t meta_buf[0];
};
#pragma pack()

#pragma pack(4)
struct VecsHeader {
  uint64_t num_vecs;
  uint32_t meta_size;
  uint8_t meta_buf[0];
};
#pragma pack()

#pragma pack(4)
struct VecsHeaderV1 {
  uint64_t num_vecs;
  uint16_t meta_size_v1;
  uint16_t version;
  uint32_t meta_size;
  uint64_t bitmap;            // set for data section
  uint64_t key_offset;        // offset for key
  uint64_t key_size;          // size for key
  uint64_t dense_offset;      // offset for dense
  uint64_t dense_size;        // size for dense
  uint64_t sparse_offset;     // offset for sparse
  uint64_t sparse_size;       // size for sparse
  uint64_t partition_offset;  // offset for partition
  uint64_t partition_size;    // size for partition
  uint64_t taglist_offset;    // offset for taglist
  uint64_t taglist_size;      // size for taglist
  uint8_t meta_buf[0];
};
#pragma pack()

class DiskAnnVecsReader {
 public:
  DiskAnnVecsReader()
      : mmap_file_(),
        index_meta_(),
        num_vecs_(0),
        vector_base_(nullptr),
        key_base_(nullptr),
        sparse_base_meta_{nullptr},
        sparse_base_data_{nullptr},
        partition_base_{nullptr},
        taglist_base_meta_{nullptr},
        taglist_base_data_{nullptr},
        taglist_size_{0} {}

  void set_measure(const std::string &name, const IndexParams &params) {
    index_meta_.set_measure(name, 0, params);
  }

  bool load(const std::string &fname) {
    return load(fname.c_str());
  }

  bool load(const char *fname) {
    if (!fname) {
      std::cerr << "Load fname is nullptr" << std::endl;
      return false;
    }
    if (!mmap_file_.open(fname, true)) {
      std::cerr << "Open file error: " << fname << std::endl;
      return false;
    }
    if (mmap_file_.size() < sizeof(VecsHeaderDetect)) {
      std::cerr << "File size is too small: " << mmap_file_.size() << std::endl;
      return false;
    }

    const VecsHeaderDetect *header =
        reinterpret_cast<const VecsHeaderDetect *>(mmap_file_.region());

    if (header->version == 0) {
      return load_v0();
    } else if (header->version == 1) {
      return load_v1();
    }

    std::cerr << "Can not recognize version: " << header->version << std::endl;

    return false;
  }

  bool load_v0() {
    const VecsHeader *header =
        reinterpret_cast<const VecsHeader *>(mmap_file_.region());
    // check
    num_vecs_ = header->num_vecs;

    // deserialize
    bool bret = index_meta_.deserialize(&header->meta_buf, header->meta_size);
    if (!bret) {
      std::cerr << "deserialize index meta error." << std::endl;
      return false;
    }

    if (!index_meta_.hybrid_vector()) {
      if ((mmap_file_.size() - sizeof(*header) - header->meta_size) %
              num_vecs_ !=
          0) {
        std::cerr << "input file foramt check error." << std::endl;
        return false;
      }
    }

    if (!index_meta_.hybrid_vector()) {
      vector_base_ =
          reinterpret_cast<const char *>(header + 1) + header->meta_size;
      key_base_ = reinterpret_cast<const uint64_t *>(
          vector_base_ + num_vecs_ * index_meta_.element_size());
    } else {
      vector_base_ =
          reinterpret_cast<const char *>(header + 1) + header->meta_size;
      key_base_ = reinterpret_cast<const uint64_t *>(
          vector_base_ + num_vecs_ * index_meta_.element_size());
      sparse_base_meta_ = reinterpret_cast<const char *>(key_base_ + num_vecs_);
      sparse_base_data_ = reinterpret_cast<const char *>(
          sparse_base_meta_ + num_vecs_ * sizeof(uint64_t));
    }

    return true;
  }

  bool load_v1() {
    const VecsHeaderV1 *header =
        reinterpret_cast<const VecsHeaderV1 *>(mmap_file_.region());
    // check
    num_vecs_ = header->num_vecs;

    // deserialize
    bool bret = index_meta_.deserialize(&header->meta_buf, header->meta_size);
    if (!bret) {
      std::cerr << "deserialize index meta error." << std::endl;
      return false;
    }

    const char *data_base_ptr =
        reinterpret_cast<const char *>(header + 1) + header->meta_size;

    vector_base_ = reinterpret_cast<const char *>(data_base_ptr);
    key_base_ = reinterpret_cast<const uint64_t *>(
        vector_base_ + num_vecs_ * index_meta_.element_size());

    if (header->sparse_offset != -1LLU) {
      sparse_base_meta_ = data_base_ptr + header->sparse_offset;
      sparse_base_data_ = sparse_base_meta_ + num_vecs_ * sizeof(uint64_t);
    }

    if (header->partition_offset != -1LLU) {
      partition_base_ = reinterpret_cast<const uint32_t *>(
          data_base_ptr + header->partition_offset);
    }

    if (header->taglist_offset != -1LLU) {
      taglist_base_meta_ = data_base_ptr + header->taglist_offset;
      taglist_base_data_ = taglist_base_meta_ + num_vecs_;
      taglist_size_ = header->taglist_size;
    }

    return true;
  }

  size_t num_vecs() const {
    return num_vecs_;
  }

  const void *vector_base() const {
    return vector_base_;
  }

  const uint64_t *key_base() const {
    return key_base_;
  }

  const IndexMeta &index_meta() const {
    return index_meta_;
  }

  uint64_t get_key(size_t index) const {
    return key_base_[index];
  }

  const void *get_vector(size_t index) const {
    return vector_base_ + index * index_meta_.element_size();
  }

  uint32_t get_sparse_count(size_t index) const {
    if (index_meta_.hybrid_vector()) {
      auto sparse_data_meta = sparse_base_meta_ + index * sizeof(uint64_t);
      uint64_t sparse_offset = *((uint64_t *)sparse_data_meta);
      uint32_t sparse_count =
          *((uint32_t *)(sparse_base_data_ + sparse_offset));

      return sparse_count;
    }

    return 0;
  }

  const uint32_t *get_sparse_indices(size_t index) const {
    if (index_meta_.hybrid_vector()) {
      auto sparse_data_meta = sparse_base_meta_ + index * sizeof(uint64_t);
      uint64_t sparse_offset = *((uint64_t *)sparse_data_meta);
      uint32_t *sparse_indices =
          (uint32_t *)(sparse_base_data_ + sparse_offset + sizeof(uint32_t));

      return sparse_indices;
    }

    return nullptr;
  }

  const void *get_sparse_data(size_t index) const {
    if (index_meta_.hybrid_vector()) {
      auto sparse_data_meta = sparse_base_meta_ + index * sizeof(uint64_t);
      uint64_t sparse_offset = *((uint64_t *)sparse_data_meta);
      uint32_t sparse_count =
          *((uint32_t *)(sparse_base_data_ + sparse_offset));
      void *sparse_data =
          (uint32_t *)(sparse_base_data_ + sparse_offset + sizeof(uint32_t) +
                       sparse_count * sizeof(uint32_t));

      return sparse_data;
    }

    return nullptr;
  }

  size_t get_total_sparse_count(void) const {
    size_t total_sparse_count = 0;
    for (size_t i = 0; i < num_vecs_; ++i) {
      total_sparse_count += get_sparse_count(i);
    }

    return total_sparse_count;
  }

  bool has_taglist(void) const {
    return taglist_base_meta_ != nullptr;
  }

  uint64_t get_taglist_count(size_t index) const {
    if (!taglist_base_data_ || !taglist_base_meta_) {
      return 0;
    }

    uint64_t taglist_count = *reinterpret_cast<const uint64_t *>(
        taglist_base_data_ + taglist_base_meta_[index]);
    return taglist_count;
  }

  const uint64_t *get_taglist(size_t index) const {
    if (!taglist_base_data_ || !taglist_base_meta_) {
      return nullptr;
    }

    return reinterpret_cast<const uint64_t *>(taglist_base_data_ +
                                              taglist_base_meta_[index]) +
           1;
  }

  const void *get_taglist_data(size_t &size) const {
    size = taglist_size_;

    return taglist_base_meta_;
  }

 private:
  ailego::MMapFile mmap_file_;
  IndexMeta index_meta_;
  size_t num_vecs_;
  const char *vector_base_;
  const uint64_t *key_base_;
  const char *sparse_base_meta_;
  const char *sparse_base_data_;
  const uint32_t *partition_base_;
  const char *taglist_base_meta_;
  const char *taglist_base_data_;
  uint64_t taglist_size_;
};

}  // namespace core
}  // namespace zvec
