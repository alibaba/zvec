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
#include <zvec/ailego/io/mmap_file.h>
#include "zvec/core/framework/index_meta.h"
#include "vecs_common.h"

namespace zvec {
namespace core {

class VecsReader {
 public:
  VecsReader()
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

  void set_metric(const std::string &name, const ailego::Params &params) {
    index_meta_.set_metric(name, 0, params);
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

    return load();
  }

  bool load() {
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
    auto sparse_data_meta = sparse_base_meta_ + index * sizeof(uint64_t);
    uint64_t sparse_offset = *((uint64_t *)sparse_data_meta);
    uint32_t sparse_count = *((uint32_t *)(sparse_base_data_ + sparse_offset));

    return sparse_count;

    return 0;
  }

  const uint32_t *get_sparse_indices(size_t index) const {
    auto sparse_data_meta = sparse_base_meta_ + index * sizeof(uint64_t);
    uint64_t sparse_offset = *((uint64_t *)sparse_data_meta);
    uint32_t *sparse_indices =
        (uint32_t *)(sparse_base_data_ + sparse_offset + sizeof(uint32_t));

    return sparse_indices;

    return nullptr;
  }

  const void *get_sparse_data(size_t index) const {
    auto sparse_data_meta = sparse_base_meta_ + index * sizeof(uint64_t);
    uint64_t sparse_offset = *((uint64_t *)sparse_data_meta);
    uint32_t sparse_count = *((uint32_t *)(sparse_base_data_ + sparse_offset));
    void *sparse_data =
        (uint32_t *)(sparse_base_data_ + sparse_offset + sizeof(uint32_t) +
                     sparse_count * sizeof(uint32_t));

    return sparse_data;
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

class SparseVecsReader {
 public:
  SparseVecsReader()
      : mmap_file_(),
        index_meta_(),
        num_vecs_(0),
        key_base_(nullptr),
        sparse_base_meta_(nullptr),
        sparse_base_data_{nullptr},
        partition_base_{nullptr},
        taglist_base_meta_{nullptr},
        taglist_base_data_{nullptr},
        taglist_size_{0} {}

  void set_metric(const std::string &name, const ailego::Params &params) {
    index_meta_.set_metric(name, 0, params);
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

    return load();
  }

  bool load() {
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

    const char *data_base_ptr =
        reinterpret_cast<const char *>(header + 1) + header->meta_size;

    key_base_ = reinterpret_cast<const uint64_t *>(
        reinterpret_cast<const char *>(header + 1) + header->meta_size);
    sparse_base_meta_ = reinterpret_cast<const char *>(key_base_ + num_vecs_);
    sparse_base_data_ = reinterpret_cast<const char *>(
        sparse_base_meta_ + num_vecs_ * sizeof(uint64_t));

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

  const void *sparse_meta_base() const {
    return sparse_base_meta_;
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

  uint32_t get_sparse_count(size_t index) const {
    auto sparse_data_meta = sparse_base_meta_ + index * sizeof(uint64_t);
    uint64_t sparse_offset = *((uint64_t *)sparse_data_meta);
    uint32_t sparse_count = *((uint32_t *)(sparse_base_data_ + sparse_offset));

    return sparse_count;

    return 0;
  }

  const uint32_t *get_sparse_indices(size_t index) const {
    auto sparse_data_meta = sparse_base_meta_ + index * sizeof(uint64_t);
    uint64_t sparse_offset = *((uint64_t *)sparse_data_meta);
    uint32_t *sparse_indices =
        (uint32_t *)(sparse_base_data_ + sparse_offset + sizeof(uint32_t));

    return sparse_indices;

    return nullptr;
  }

  const void *get_sparse_data(size_t index) const {
    auto sparse_data_meta = sparse_base_meta_ + index * sizeof(uint64_t);
    uint64_t sparse_offset = *((uint64_t *)sparse_data_meta);
    uint32_t sparse_count = *((uint32_t *)(sparse_base_data_ + sparse_offset));
    void *sparse_data =
        (uint32_t *)(sparse_base_data_ + sparse_offset + sizeof(uint32_t) +
                     sparse_count * sizeof(uint32_t));

    return sparse_data;
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
    uint64_t taglist_count = *reinterpret_cast<const uint64_t *>(
        taglist_base_data_ + taglist_base_meta_[index]);
    return taglist_count;
  }

  const uint64_t *get_taglist(size_t index) const {
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