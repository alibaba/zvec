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

#include <string>
#include <zvec/ailego/container/params.h>
#include "zvec/core/framework/index_holder.h"
#include "vecs_reader.h"

namespace zvec {
namespace core {

/*!
 * Vecs Index Holder
 *  framwork will use IndexHolder in this way:
 *  for (iter = create_iterator(); iter->is_valid(); iter->next()) {
 *      key = iter->key();
 *      data = iter->data();
 *  }
 */
class VecsIndexHolder : public IndexHybridHolder {
 public:
  typedef std::shared_ptr<VecsIndexHolder> Pointer;

  bool load(const std::string &file_path) {
    return vecs_reader_.load(file_path);
  }

  const IndexMeta &index_meta(void) const {
    return vecs_reader_.index_meta();
  }

  void set_metric(const std::string &name, const ailego::Params &params) {
    vecs_reader_.set_metric(name, params);
  }

  /*!
   * Index Holder Iterator
   */
  class Iterator : public IndexHybridHolder::Iterator {
   public:
    //! Constructor
    Iterator(const VecsIndexHolder &holder, uint32_t cursor)
        : cursor_(cursor),
          vecs_reader_(holder.vecs_reader_),
          stop_(holder.stop_) {}

    //! Test if the iterator is valid
    virtual bool is_valid(void) const override {
      return !stop_ && cursor_ < vecs_reader_.num_vecs();
    }

    //! Retrieve primary key
    virtual uint64_t key(void) const override {
      return vecs_reader_.get_key(cursor_);
    }

    //! Retrieve pointer of data
    virtual const void *data() const override {
      return vecs_reader_.get_vector(cursor_);
    }

    //! Retrieve primary key
    virtual uint32_t sparse_count() const override {
      return vecs_reader_.get_sparse_count(cursor_);
    }

    //! Retrieve primary key
    virtual const uint32_t *sparse_indices() const override {
      return vecs_reader_.get_sparse_indices(cursor_);
    }

    //! Retrieve primary key
    virtual const void *sparse_data() const override {
      return vecs_reader_.get_sparse_data(cursor_);
    }

    //! Next iterator
    virtual void next(void) override {
      ++cursor_;
    }

    //! Reset the iterator
    virtual void reset(void) {
      cursor_ = 0;
    }

   private:
    size_t cursor_;
    const VecsReader &vecs_reader_;
    const bool &stop_;
  };

  virtual IndexHolder::Iterator::Pointer create_iterator(void) override {
    // make sure iter has value whenn create_iterator finished
    IndexHolder::Iterator::Pointer iter(
        new VecsIndexHolder::Iterator(*this, start_cursor_));
    return iter;
  }

  virtual IndexHybridHolder::Iterator::Pointer create_hybrid_iterator(
      void) override {
    // make sure iter has value whenn create_iterator finished
    IndexHybridHolder::Iterator::Pointer iter(
        new VecsIndexHolder::Iterator(*this, start_cursor_));
    return iter;
  }

  //! Retrieve count of elements in holder
  virtual size_t count(void) const override {
    return max_doc_count_ != 0
               ? std::min(max_doc_count_, vecs_reader_.num_vecs())
               : vecs_reader_.num_vecs();
  }

  //! Retrieve dimension
  virtual size_t dimension(void) const override {
    return vecs_reader_.index_meta().dimension();
  }

  //! Retrieve type information
  virtual IndexMeta::DataType data_type(void) const override {
    return vecs_reader_.index_meta().data_type();
  }

  //! Retrieve element size in bytes
  virtual size_t element_size(void) const override {
    return vecs_reader_.index_meta().element_size();
  }

  //! Retrieve if it can multi-pass
  virtual bool multipass(void) const override {
    return true;
  }

  void stop(void) {
    stop_ = true;
  }

  uint64_t get_num_vecs() const {
    return vecs_reader_.num_vecs();
  }

  uint64_t get_key(size_t idx) const {
    return vecs_reader_.get_key(idx);
  }

  const void *get_vector(size_t idx) const {
    return vecs_reader_.get_vector(idx);
  }

  uint32_t get_sparse_count(size_t idx) const {
    return vecs_reader_.get_sparse_count(idx);
  }

  const uint32_t *get_sparse_indices(size_t idx) const {
    return vecs_reader_.get_sparse_indices(idx);
  }

  const void *get_sparse_data(size_t idx) const {
    return vecs_reader_.get_sparse_data(idx);
  }

  void set_start_cursor(uint32_t index) {
    start_cursor_ = index;
  }

  void set_max_doc_count(size_t value) {
    max_doc_count_ = value;
  }

  uint32_t start_cursor() const {
    return start_cursor_;
  }

  size_t total_sparse_count(void) const override {
    return vecs_reader_.get_total_sparse_count();
  }

  bool has_taglist() const {
    return vecs_reader_.has_taglist();
  }

  uint64_t get_taglist_count(size_t index) const {
    return vecs_reader_.get_taglist_count(index);
  }

  const void *get_taglist(size_t index) const {
    return vecs_reader_.get_taglist(index);
  }

  const void *get_taglist_data(size_t &size) const {
    return vecs_reader_.get_taglist_data(size);
  }

  const void *get_key_base() const {
    return vecs_reader_.key_base();
  }

 private:
  bool stop_{false};
  uint32_t start_cursor_{0};
  VecsReader vecs_reader_;
  size_t max_doc_count_{0};
};


/*!
 * Vecs Index Sparse Holder
 *  framwork will use IndexHolder in this way:
 *  for (iter = create_iterator(); iter->is_valid(); iter->next()) {
 *      key = iter->key();
 *      data = iter->sparse_data();
 *  }
 */
class VecsIndexSparseHolder : public IndexSparseHolder {
 public:
  typedef std::shared_ptr<VecsIndexSparseHolder> Pointer;

  bool load(const std::string &file_path) {
    return vecs_reader_.load(file_path);
  }

  const IndexMeta &index_meta(void) const {
    return vecs_reader_.index_meta();
  }

  void set_metric(const std::string &name, const ailego::Params &params) {
    vecs_reader_.set_metric(name, params);
  }

  /*!
   * Index Holder Iterator
   */
  class Iterator : public IndexSparseHolder::Iterator {
   public:
    //! Constructor
    Iterator(const VecsIndexSparseHolder &holder, uint32_t cursor)
        : cursor_(cursor),
          vecs_reader_(holder.vecs_reader_),
          stop_(holder.stop_) {}

    //! Test if the iterator is valid
    virtual bool is_valid(void) const override {
      return !stop_ && cursor_ < vecs_reader_.num_vecs();
    }

    //! Retrieve primary key
    virtual uint64_t key(void) const override {
      return vecs_reader_.get_key(cursor_);
    }

    //! Retrieve primary key
    virtual uint32_t sparse_count() const override {
      return vecs_reader_.get_sparse_count(cursor_);
    }

    //! Retrieve primary key
    virtual const uint32_t *sparse_indices() const override {
      return vecs_reader_.get_sparse_indices(cursor_);
    }

    //! Retrieve primary key
    virtual const void *sparse_data() const override {
      return vecs_reader_.get_sparse_data(cursor_);
    }

    //! Next iterator
    virtual void next(void) override {
      ++cursor_;
    }

    //! Reset the iterator
    virtual void reset(void) {
      cursor_ = 0;
    }

   private:
    size_t cursor_;
    const SparseVecsReader &vecs_reader_;
    const bool &stop_;
  };

  virtual IndexSparseHolder::Iterator::Pointer create_iterator(void) override {
    // make sure iter has value whenn create_iterator finished
    IndexSparseHolder::Iterator::Pointer iter(
        new VecsIndexSparseHolder::Iterator(*this, start_cursor_));
    return iter;
  }

  //! Retrieve count of elements in holder
  virtual size_t count(void) const override {
    return max_doc_count_ != 0
               ? std::min(max_doc_count_, vecs_reader_.num_vecs())
               : vecs_reader_.num_vecs();
  }

  //! Retrieve type information
  virtual IndexMeta::DataType data_type(void) const override {
    return vecs_reader_.index_meta().data_type();
  }

  //! Retrieve if it can multi-pass
  virtual bool multipass(void) const override {
    return true;
  }

  void stop(void) {
    stop_ = true;
  }

  uint64_t get_key(size_t idx) const {
    return vecs_reader_.get_key(idx);
  }

  uint32_t get_sparse_count(size_t idx) const {
    return vecs_reader_.get_sparse_count(idx);
  }

  const uint32_t *get_sparse_indices(size_t idx) const {
    return vecs_reader_.get_sparse_indices(idx);
  }

  const void *get_sparse_data(size_t idx) const {
    return vecs_reader_.get_sparse_data(idx);
  }

  void set_start_cursor(uint32_t index) {
    start_cursor_ = index;
  }

  void set_max_doc_count(size_t value) {
    max_doc_count_ = value;
  }

  uint64_t get_num_vecs() const {
    return vecs_reader_.num_vecs();
  }

  uint32_t start_cursor() const {
    return start_cursor_;
  }

  size_t total_sparse_count(void) const override {
    return vecs_reader_.get_total_sparse_count();
  }

  bool has_taglist() const {
    return vecs_reader_.has_taglist();
  }

  uint64_t get_taglist_count(size_t index) const {
    return vecs_reader_.get_taglist_count(index);
  }

  const void *get_taglist(size_t index) const {
    return vecs_reader_.get_taglist(index);
  }

  const void *get_taglist_data(size_t &size) const {
    return vecs_reader_.get_taglist_data(size);
  }

  const void *get_key_base() const {
    return vecs_reader_.key_base();
  }

 private:
  bool stop_{false};
  uint32_t start_cursor_{0};
  SparseVecsReader vecs_reader_;
  size_t max_doc_count_{0};
};

}  // namespace core
}  // namespace zvec