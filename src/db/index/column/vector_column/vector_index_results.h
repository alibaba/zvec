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

#include <cstddef>
#include <memory>
#include <zvec/core/framework/index_document.h>
#include "db/common/typedef.h"
#include "db/index/column/common/index_results.h"

// TODO: eliminate aitheta2 dependency for decoupling

namespace zvec {

class VectorIndexResults : public IndexResults {
 public:
  class VectorIterator : public IndexResults::Iterator {
   public:
    VectorIterator(const VectorIndexResults *rs) : rs_(rs) {}

    VectorIterator(const VectorIndexResults *rs, uint32_t index)
        : rs_(rs), index_(index) {}

   public:
    idx_t doc_id() const override {
      return rs_->document(index_).key();
    }

    float score() const override {
      return rs_->document(index_).score();
    }

    void next() override {
      index_++;
    }

    bool valid() const override {
      return (index_ < rs_->count());
    }

    const vector_column_params::VectorData vector() const override {
      if (is_sparse()) {
        return vector_column_params::VectorData{
            vector_column_params::SparseVector{sparse_count(),
                                               sparse_indices().data(),
                                               sparse_values().data()}};
      }
      return vector_column_params::VectorData{
          vector_column_params::DenseVector{dense_vector()}};
    }

   private:
    const void *dense_vector() const {
      if (!rs_->reverted_vector_list_.empty()) {
        return rs_->reverted_vector_list_[index_].data();
      }
      return rs_->document(index_).vector();
    }
    uint32_t sparse_count() const {
      return rs_->document(index_).sparse_doc().sparse_count();
    }

    const std::string &sparse_indices() const {
      return rs_->document(index_).sparse_doc().sparse_indices();
    }

    const std::string &sparse_values() const {
      if (!rs_->reverted_sparse_values_list_.empty()) {
        return rs_->reverted_sparse_values_list_[index_];
      }
      return rs_->document(index_).sparse_doc().sparse_values();
    }

   private:
    const VectorIndexResults *rs_{nullptr};
    uint32_t index_{0U};
  };

  friend class VectorIterator;

 public:
  // VectorIndexResults(core::IndexDocumentList &&doc_list)
  //     : docs_(std::move(doc_list)) {}
  //
  // VectorIndexResults(core::IndexDocumentList &&doc_list,
  //                    std::vector<std::string> &&reverted_vector_list)
  //     : docs_(std::move(doc_list)),
  //       reverted_vector_list_(std::move(reverted_vector_list)) {}
  VectorIndexResults(bool is_sparse, core::IndexDocumentList &&doc_list,
                     std::vector<std::string> &&reverted_vector_list,
                     std::vector<std::string> &&reverted_sparse_values_list)
      : is_sparse_(is_sparse),
        docs_(std::move(doc_list)),
        reverted_vector_list_(std::move(reverted_vector_list)),
        reverted_sparse_values_list_(std::move(reverted_sparse_values_list)) {}

 public:
  IndexResults::IteratorUPtr create_iterator() override {
    auto ret = std::unique_ptr<VectorIterator>(new VectorIterator(this));
    ret->set_is_sparse(is_sparse_);
    return ret;
  }

  size_t count() const override {
    return docs_.size();
  }

 public:  // unique method
  core::IndexDocumentList &docs() {
    return docs_;
  }

  std::vector<std::string> &reverted_vector_list() {
    return reverted_vector_list_;
  }

  std::vector<std::string> &reverted_sparse_values_list() {
    return reverted_sparse_values_list_;
  }


 private:
  const core::IndexDocument &document(size_t index) const {
    return docs_[index];
  }

 private:
  bool is_sparse_;
  core::IndexDocumentList docs_{};
  std::vector<std::string> reverted_vector_list_{};
  std::vector<std::string> reverted_sparse_values_list_{};
};

class GroupVectorIndexResults : public IndexResults {
 public:
  class GroupVectorIterator : public IndexResults::Iterator {
   public:
    GroupVectorIterator(const GroupVectorIndexResults *rs) : rs_(rs) {}

   public:
    idx_t doc_id() const override {
      return rs_->document(group_index_, doc_index_).key();
    }

    float score() const override {
      return rs_->document(group_index_, doc_index_).score();
    }

    void next() override {
      doc_index_++;
      if (doc_index_ >= rs_->groups_[group_index_].docs().size()) {
        group_index_++;
        doc_index_ = 0;
      }
    }

    bool valid() const override {
      return group_index_ < rs_->groups_.size();
    }

    const vector_column_params::VectorData vector() const override {
      if (is_sparse()) {
        return vector_column_params::VectorData{
            vector_column_params::SparseVector{sparse_count(),
                                               sparse_indices().data(),
                                               sparse_values().data()}};
      }
      return vector_column_params::VectorData{
          vector_column_params::DenseVector{dense_vector()}};
    }

   private:
    const void *dense_vector() const {
      if (!rs_->reverted_vector_list_.empty()) {
        return rs_->reverted_vector_list_[group_index_][doc_index_].data();
      }
      return rs_->document(group_index_, doc_index_).vector();
    }

    uint32_t sparse_count() const {
      return rs_->document(group_index_, doc_index_)
          .sparse_doc()
          .sparse_count();
    }

    const std::string &sparse_indices() const {
      return rs_->document(group_index_, doc_index_)
          .sparse_doc()
          .sparse_indices();
    }

    const std::string &sparse_values() const {
      if (!rs_->reverted_sparse_values_list_.empty()) {
        return rs_->reverted_sparse_values_list_[group_index_][doc_index_];
      }
      return rs_->document(group_index_, doc_index_)
          .sparse_doc()
          .sparse_values();
    }

    const std::string &group_id() const override {
      return rs_->groups_[group_index_].group_id();
    }

   private:
    const GroupVectorIndexResults *rs_{nullptr};
    uint32_t group_index_{0U};
    uint32_t doc_index_{0U};
  };

  friend class GroupVectorIterator;

 public:
  GroupVectorIndexResults(core::IndexGroupDocumentList &&group_list)
      : groups_(std::move(group_list)) {
    init_count();
  }

  GroupVectorIndexResults(
      core::IndexGroupDocumentList &&group_list,
      std::vector<std::vector<std::string>> &&reverted_vector_list)
      : groups_(std::move(group_list)),
        reverted_vector_list_(std::move(reverted_vector_list)) {
    init_count();
  }

  GroupVectorIndexResults(
      core::IndexGroupDocumentList &&group_list,
      std::vector<std::vector<std::string>> &&reverted_vector_list,
      std::vector<std::vector<std::string>> &&reverted_sparse_values_list)
      : groups_(std::move(group_list)),
        reverted_vector_list_(std::move(reverted_vector_list)),
        reverted_sparse_values_list_(std::move(reverted_sparse_values_list)) {}

 public:
  IndexResults::IteratorUPtr create_iterator() override {
    return std::unique_ptr<GroupVectorIterator>(new GroupVectorIterator(this));
  }


  size_t count() const override {
    return count_;
  }

 public:  // unique method
  core::IndexGroupDocumentList &groups() {
    return groups_;
  }

 private:
  const core::IndexDocument &document(size_t group_index,
                                      size_t doc_index) const {
    return groups_[group_index].docs()[doc_index];
  }

  void init_count() {
    count_ = 0;
    for (const auto &group : groups_) {
      count_ += group.docs().size();
    }
  }

 private:
  core::IndexGroupDocumentList groups_{};
  std::vector<std::vector<std::string>> reverted_vector_list_{};
  std::vector<std::vector<std::string>> reverted_sparse_values_list_{};
  size_t count_{0};
};


}  // namespace zvec
