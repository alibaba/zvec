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

#include <cstdint>
#include <string>
#include <zvec/ailego/container/heap.h>
#include <zvec/core/framework/index_storage.h>

namespace zvec {
namespace core {

/*! Index Sparse Document
 */
class IndexSparseDocument {
 public:
  //! Constructor
  IndexSparseDocument() = default;

  //! Retrieve mutable sparse count
  uint32_t *mutable_sparse_count(void) {
    return &sparse_count_;
  }

  //! Retrieve mutable sparse indices
  std::string *mutable_sparse_indices(void) {
    return &sparse_indices_;
  }

  //! Retrieve mutable sparse values
  std::string *mutable_sparse_values(void) {
    return &sparse_values_;
  }

  //! Retrieve sparse count
  uint32_t sparse_count(void) const {
    return sparse_count_;
  }

  //! Retrieve sparse indices
  const std::string &sparse_indices(void) const {
    return sparse_indices_;
  }

  //! Retrieve sparse values
  const std::string &sparse_values(void) const {
    return sparse_values_;
  }

 private:
  uint32_t sparse_count_{0};
  std::string sparse_indices_{};
  std::string sparse_values_{};
};

/*! Index Document
 */
class IndexDocument {
 public:
  //! Constructor
  IndexDocument() = default;

  //! Constructor
  IndexDocument(uint64_t k, float v) : key_(k), score_(v) {}

  //! Constructor
  IndexDocument(uint64_t k, float v, uint32_t i)
      : key_(k), score_(v), index_(i) {}

  //! Constructor
  IndexDocument(uint64_t k, float v, uint32_t i, const void *vector)
      : key_(k), score_(v), index_(i), vector_(vector) {}

  IndexDocument(uint64_t k, float v, uint32_t i,
                IndexStorage::MemoryBlock vec_block)
      : key_(k), score_(v), index_(i), vec_mem_block_(vec_block) {
    vector_ = vec_mem_block_.data();
    has_vec_mem_block_ = true;
  }

  //! Constructor
  IndexDocument(uint64_t k, float v, uint32_t i, const void *vector,
                IndexSparseDocument sparse_doc)
      : key_(k),
        score_(v),
        index_(i),
        vector_(vector),
        sparse_doc_(std::move(sparse_doc)) {}

  //! Constructor
  IndexDocument(uint64_t k, float v, uint32_t i, std::string vector_string)
      : key_(k),
        score_(v),
        index_(i),
        vector_string_(std::move(vector_string)) {}

  //! Constructor
  IndexDocument(uint64_t k, float v, uint32_t i, std::string vector_string,
                IndexSparseDocument sparse_doc)
      : key_(k),
        score_(v),
        index_(i),
        vector_{nullptr},
        vector_string_(std::move(vector_string)),
        sparse_doc_(std::move(sparse_doc)) {}

  IndexDocument(uint64_t k, float v, uint32_t i,
                IndexStorage::MemoryBlock vec_block,
                IndexSparseDocument sparse_doc)
      : key_(k),
        score_(v),
        index_(i),
        vec_mem_block_(vec_block),
        sparse_doc_(std::move(sparse_doc)) {
    has_vec_mem_block_ = true;
    vector_ = vec_mem_block_.data();
  }

  //! Constructor
  IndexDocument(const IndexDocument &rhs)
      : key_(rhs.key_),
        score_(rhs.score_),
        index_(rhs.index_),
        vector_(rhs.vector_),
        vector_string_(std::move(rhs.vector_string_)),
        sparse_doc_{rhs.sparse_doc_} {
    if (rhs.has_vec_mem_block_) {
      vec_mem_block_ = rhs.vec_mem_block_;
      has_vec_mem_block_ = true;
    }
  }

  //! Assignment
  IndexDocument &operator=(const IndexDocument &rhs) {
    if (this != &rhs) {
      key_ = rhs.key_;
      score_ = rhs.score_;
      index_ = rhs.index_;
      vector_ = rhs.vector_;
      vector_string_ = rhs.vector_string_;
      if (rhs.has_vec_mem_block_) {
        vec_mem_block_ = rhs.vec_mem_block_;
        has_vec_mem_block_ = true;
      }
      sparse_doc_ = rhs.sparse_doc_;
    }
    return *this;
  }

  //! Less than
  bool operator<(const IndexDocument &rhs) const {
    return (this->score_ < rhs.score_);
  }

  //! Greater than
  bool operator>(const IndexDocument &rhs) const {
    return (this->score_ > rhs.score_);
  }

  //! Retrieve primary key
  uint64_t key(void) const {
    return key_;
  }

  //! Retrieve score value
  float score(void) const {
    return score_;
  }

  //! Retrieve index id
  uint32_t index(void) const {
    return index_;
  }

  //! Retrieve vec
  const void *vector() const {
    return vector_;
  }

  //! Retrieve vec string
  const std::string &vector_string() const {
    return vector_string_;
  }

  //! Retrieve vec
  const IndexSparseDocument &sparse_doc() const {
    return sparse_doc_;
  }

  //! Retrieve mutable primary key
  uint64_t *mutable_key(void) {
    return &key_;
  }

  //! Retrieve mutable score value
  float *mutable_score(void) {
    return &score_;
  }

  //! Retrieve mutable index id
  uint32_t *mutable_index(void) {
    return &index_;
  }

  //! Retrieve primary key
  void set_key(uint64_t val) {
    key_ = val;
  }

  //! Retrieve score value
  void set_score(float val) {
    score_ = val;
  }

  //! Retrieve index id
  void set_index(uint32_t val) {
    index_ = val;
  }

 private:
  //! Data members
  uint64_t key_{0u};
  float score_{0.0f};
  uint32_t index_{0u};
  const void *vector_{nullptr};
  std::string vector_string_{};
  bool has_vec_mem_block_{false};
  mutable IndexStorage::MemoryBlock vec_mem_block_{};
  IndexSparseDocument sparse_doc_{};
};

/*! Index Document Heap
 */
class IndexDocumentHeap : public ailego::Heap<IndexDocument> {
 public:
  //! Constructor
  IndexDocumentHeap(void) : ailego::Heap<IndexDocument>() {}

  //! Constructor
  IndexDocumentHeap(size_t max) : ailego::Heap<IndexDocument>(max) {}

  //! Constructor
  IndexDocumentHeap(size_t max, float val)
      : ailego::Heap<IndexDocument>(max), threshold_(val) {}

  //! Constructor
  IndexDocumentHeap(const IndexDocumentHeap &rhs)
      : ailego::Heap<IndexDocument>(rhs), threshold_(rhs.threshold_) {}

  //! Constructor
  IndexDocumentHeap(IndexDocumentHeap &&rhs)
      : ailego::Heap<IndexDocument>(std::move(rhs)),
        threshold_(rhs.threshold_) {}

  //! Constructor
  IndexDocumentHeap(const std::vector<IndexDocument> &rhs)
      : ailego::Heap<IndexDocument>(rhs) {}

  //! Constructor
  IndexDocumentHeap(std::vector<IndexDocument> &&rhs)
      : ailego::Heap<IndexDocument>(std::move(rhs)) {}

  //! Insert a document into the heap
  void emplace(uint64_t key, float score) {
    if (score <= threshold_) {
      ailego::Heap<IndexDocument>::emplace(key, score);
    }
  }

  //! Insert a document into the heap
  void emplace(uint64_t key, float score, uint32_t index) {
    if (score <= threshold_) {
      ailego::Heap<IndexDocument>::emplace(key, score, index);
    }
  }

  //! Set threshold for RNN
  void set_threshold(float val) {
    threshold_ = val;
  }

  //! Retrieve value of threshold for RNN
  float threshold(void) const {
    return threshold_;
  }

 private:
  //! members
  float threshold_{std::numeric_limits<float>::max()};
};

class GroupIndexDocument {
 public:
  GroupIndexDocument() = default;

  GroupIndexDocument(const std::string &group_id,
                     const std::vector<IndexDocument> &docs)
      : group_id_(group_id), docs_(docs) {}

  const std::string &group_id() const {
    return group_id_;
  }

  const std::vector<IndexDocument> &docs() const {
    return docs_;
  }

  std::string *mutable_group_id() {
    return &group_id_;
  }

  std::vector<IndexDocument> *mutable_docs() {
    return &docs_;
  }

  void set_group_id(const std::string &group_id) {
    group_id_ = group_id;
  }

 private:
  std::string group_id_;
  std::vector<IndexDocument> docs_;
};

/*! Index Document List
 */
using IndexDocumentList = std::vector<IndexDocument>;
using IndexGroupDocumentList = std::vector<GroupIndexDocument>;

}  // namespace core
}  // namespace zvec
