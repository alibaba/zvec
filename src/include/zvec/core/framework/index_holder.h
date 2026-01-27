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

#include <cstring>
#include <list>
#include <memory>
#include <vector>
#include <zvec/ailego/container/vector.h>
#include <zvec/ailego/internal/platform.h>
#include <zvec/core/framework/index_features.h>
#include <zvec/core/framework/index_meta.h>

namespace zvec {
namespace core {

/*! Index Holder
 */
struct IndexHolder {
  //! Index Holder Pointer
  typedef std::shared_ptr<IndexHolder> Pointer;

  /*! Index Holder Iterator
   */
  struct Iterator {
    //! Index Holder Iterator Pointer
    typedef std::unique_ptr<Iterator> Pointer;

    //! Destructor
    virtual ~Iterator(void) {}

    //! Retrieve pointer of data
    virtual const void *data(void) const = 0;

    //! Test if the iterator is valid
    virtual bool is_valid(void) const = 0;

    //! Retrieve primary key
    virtual uint64_t key(void) const = 0;

    //! Next iterator
    virtual void next(void) = 0;
  };

  //! Destructor
  virtual ~IndexHolder(void) {}

  //! Retrieve count of elements in holder (-1 indicates unknown)
  virtual size_t count(void) const = 0;

  //! Retrieve dimension
  virtual size_t dimension(void) const = 0;

  //! Retrieve type information
  virtual IndexMeta::DataType data_type(void) const = 0;

  //! Retrieve element size in bytes
  virtual size_t element_size(void) const = 0;

  //! Retrieve if it can multi-pass
  virtual bool multipass(void) const = 0;

  //! Create a new iterator
  virtual Iterator::Pointer create_iterator(void) = 0;

  //! Test if matchs the meta
  bool is_matched(const IndexMeta &meta) const {
    return (this->data_type() == meta.data_type() &&
            this->dimension() == meta.dimension() &&
            this->element_size() == meta.element_size());
  }
};

/*! Index Hybrid Holder
 */
struct IndexHybridHolder : public IndexHolder {
  //! Index Holder Pointer
  typedef std::shared_ptr<IndexHybridHolder> Pointer;

  /*! Index Holder Iterator
   */
  struct Iterator : public IndexHolder::Iterator {
    //! Index Holder Pointer
    typedef std::unique_ptr<Iterator> Pointer;

    //! Destructor
    virtual ~Iterator(void) {}

    //! Retrieve pointer of data
    virtual const void *data(void) const = 0;

    //! Test if the iterator is valid
    virtual bool is_valid(void) const = 0;

    //! Retrieve primary key
    virtual uint64_t key(void) const = 0;

    //! Retrieve sparse count
    virtual uint32_t sparse_count() const = 0;

    //! Retrieve sparse indicies
    virtual const uint32_t *sparse_indices() const = 0;

    //! Retrieve sparse data
    virtual const void *sparse_data() const = 0;

    //! Next iterator
    virtual void next(void) = 0;
  };

  //! Destructor
  virtual ~IndexHybridHolder(void) {}

  //! Retrieve sparse count summing up over all the docs
  virtual size_t total_sparse_count(void) const = 0;

  //! Create a new hybrid iterator
  virtual Iterator::Pointer create_hybrid_iterator(void) = 0;
};

/*! Index Sparse Holder
 */
struct IndexSparseHolder {
  //! Index Sparse Holder Pointer
  typedef std::shared_ptr<IndexSparseHolder> Pointer;

  /*! Index Holder Iterator
   */
  struct Iterator {
    //! Index Holder Iterator Pointer
    typedef std::unique_ptr<Iterator> Pointer;

    //! Destructor
    virtual ~Iterator(void) {}

    //! Test if the iterator is valid
    virtual bool is_valid(void) const = 0;

    //! Retrieve primary key
    virtual uint64_t key(void) const = 0;

    //! Retrieve sparse count
    virtual uint32_t sparse_count() const = 0;

    //! Retrieve sparse indicies
    virtual const uint32_t *sparse_indices() const = 0;

    //! Retrieve sparse data
    virtual const void *sparse_data() const = 0;

    //! Next iterator
    virtual void next(void) = 0;
  };

  //! Destructor
  virtual ~IndexSparseHolder(void) {}

  //! Retrieve count of elements in holder (-1 indicates unknown)
  virtual size_t count(void) const = 0;

  //! Retrieve type information
  virtual IndexMeta::DataType data_type(void) const = 0;

  //! Retrieve if it can multi-pass
  virtual bool multipass(void) const = 0;

  //! Create a new iterator
  virtual Iterator::Pointer create_iterator(void) = 0;

  //! Test if matchs the meta
  bool is_matched(const IndexMeta &meta) const {
    return (this->data_type() == meta.data_type());
  }

  //! Retrieve sparse count summing up over all the docs for reserving space
  virtual size_t total_sparse_count(void) const = 0;
};

/*! One-Pass Numerical Index Holder
 */
template <typename T>
class OnePassNumericalIndexHolder : public IndexHolder {
 public:
  /*! One-Pass Index Holder Iterator
   */
  class Iterator : public IndexHolder::Iterator {
   public:
    //! Index Holder Iterator Pointer
    typedef std::unique_ptr<Iterator> Pointer;

    //! Constructor
    Iterator(OnePassNumericalIndexHolder *owner) : holder_(owner) {
      features_iter_ = holder_->features_.begin();
    }

    //! Destructor
    virtual ~Iterator(void) {}

    //! Retrieve pointer of data
    const void *data(void) const override {
      return features_iter_->second.data();
    }

    //! Test if the iterator is valid
    bool is_valid(void) const override {
      return (features_iter_ != holder_->features_.end());
    }

    //! Retrieve primary key
    uint64_t key(void) const override {
      return features_iter_->first;
    }

    //! Next iterator
    void next(void) override {
      holder_->features_.erase(features_iter_++);
    }

   private:
    OnePassNumericalIndexHolder *holder_{nullptr};
    typename std::list<std::pair<uint64_t, ailego::NumericalVector<T>>>::
        iterator features_iter_{};
  };

  //! Constructor
  OnePassNumericalIndexHolder(size_t dim) : dimension_(dim) {}

  //! Retrieve count of elements in holder (-1 indicates unknown)
  size_t count(void) const override {
    return features_.size();
  }

  //! Retrieve dimension
  size_t dimension(void) const override {
    return dimension_;
  }

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_UNDEFINED;
  }

  //! Retrieve element size in bytes
  size_t element_size(void) const override {
    return dimension_ * sizeof(T);
  }

  //! Retrieve if it can multi-pass
  bool multipass(void) const override {
    return false;
  }

  //! Create a new iterator
  IndexHolder::Iterator::Pointer create_iterator(void) override {
    return IndexHolder::Iterator::Pointer(
        new OnePassNumericalIndexHolder::Iterator(this));
  }

  //! Append an element into holder
  bool emplace(uint64_t key, const ailego::NumericalVector<T> &vec) {
    if (vec.size() != dimension_) {
      return false;
    }
    features_.emplace_back(key, vec);
    return true;
  }

  //! Append an element into holder
  bool emplace(uint64_t key, ailego::NumericalVector<T> &&vec) {
    if (vec.size() != dimension_) {
      return false;
    }
    features_.emplace_back(key, std::move(vec));
    return true;
  }

 private:
  //! Disable them
  OnePassNumericalIndexHolder(void) = delete;

  //! Members
  size_t dimension_{0};
  std::list<std::pair<uint64_t, ailego::NumericalVector<T>>> features_;
};

/*! Multi-Pass Numerical Index Holder
 */
template <typename T>
class MultiPassNumericalIndexHolder : public IndexHolder {
 public:
  /*! Multi-Pass Index Holder Iterator
   */
  class Iterator : public IndexHolder::Iterator {
   public:
    //! Index Holder Iterator Pointer
    typedef std::unique_ptr<Iterator> Pointer;

    //! Constructor
    Iterator(MultiPassNumericalIndexHolder *owner) : holder_(owner) {
      features_iter_ = holder_->features_.begin();
    }

    //! Destructor
    virtual ~Iterator(void) {}

    //! Retrieve pointer of data
    const void *data(void) const override {
      return features_iter_->second.data();
    }

    //! Test if the iterator is valid
    bool is_valid(void) const override {
      return (features_iter_ != holder_->features_.end());
    }

    //! Retrieve primary key
    uint64_t key(void) const override {
      return features_iter_->first;
    }

    //! Next iterator
    void next(void) override {
      ++features_iter_;
    }

   private:
    MultiPassNumericalIndexHolder *holder_{nullptr};
    typename std::vector<std::pair<uint64_t, ailego::NumericalVector<T>>>::
        iterator features_iter_{};
  };

  //! Constructor
  MultiPassNumericalIndexHolder(size_t dim) : dimension_(dim) {}

  //! Retrieve count of elements in holder (-1 indicates unknown)
  size_t count(void) const override {
    return features_.size();
  }

  //! Retrieve dimension
  size_t dimension(void) const override {
    return dimension_;
  }

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_UNDEFINED;
  }

  //! Retrieve element size in bytes
  size_t element_size(void) const override {
    return dimension_ * sizeof(T);
  }

  //! Retrieve if it can multi-pass
  bool multipass(void) const override {
    return true;
  }

  //! Create a new iterator
  IndexHolder::Iterator::Pointer create_iterator(void) override {
    return IndexHolder::Iterator::Pointer(
        new MultiPassNumericalIndexHolder::Iterator(this));
  }

  //! Append an element into holder
  bool emplace(uint64_t key, const ailego::NumericalVector<T> &vec) {
    if (vec.size() != dimension_) {
      return false;
    }
    features_.emplace_back(key, vec);
    return true;
  }

  //! Append an element into holder
  bool emplace(uint64_t key, ailego::NumericalVector<T> &&vec) {
    if (vec.size() != dimension_) {
      return false;
    }
    features_.emplace_back(key, std::move(vec));
    return true;
  }

  //! Request a change in capacity
  void reserve(size_t size) {
    features_.reserve(size);
  }

 private:
  //! Disable them
  MultiPassNumericalIndexHolder(void) = delete;

  //! Members
  size_t dimension_{0};
  std::vector<std::pair<uint64_t, ailego::NumericalVector<T>>> features_;
};

/*! One-Pass Binary Index Holder
 */
template <typename T>
class OnePassBinaryIndexHolder : public IndexHolder {
 public:
  /*! One-Pass Index Holder Iterator
   */
  class Iterator : public IndexHolder::Iterator {
   public:
    //! Index Holder Iterator Pointer
    typedef std::unique_ptr<Iterator> Pointer;

    //! Constructor
    Iterator(OnePassBinaryIndexHolder *owner) : holder_(owner) {
      features_iter_ = holder_->features_.begin();
    }

    //! Destructor
    virtual ~Iterator(void) {}

    //! Retrieve pointer of data
    const void *data(void) const override {
      return features_iter_->second.data();
    }

    //! Test if the iterator is valid
    bool is_valid(void) const override {
      return (features_iter_ != holder_->features_.end());
    }

    //! Retrieve primary key
    uint64_t key(void) const override {
      return features_iter_->first;
    }

    //! Next iterator
    void next(void) override {
      holder_->features_.erase(features_iter_++);
    }

   private:
    OnePassBinaryIndexHolder *holder_{nullptr};
    typename std::list<std::pair<uint64_t, ailego::BinaryVector<T>>>::iterator
        features_iter_{};
  };

  //! Constructor
  OnePassBinaryIndexHolder(size_t dim) : dimension_(dim) {}

  //! Retrieve count of elements in holder (-1 indicates unknown)
  size_t count(void) const override {
    return features_.size();
  }

  //! Retrieve dimension
  size_t dimension(void) const override {
    return dimension_;
  }

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_UNDEFINED;
  }

  //! Retrieve element size in bytes
  size_t element_size(void) const override {
    return (dimension_ + (sizeof(T) << 3) - 1) / (sizeof(T) << 3) * sizeof(T);
  }

  //! Retrieve if it can multi-pass
  bool multipass(void) const override {
    return false;
  }

  //! Create a new iterator
  IndexHolder::Iterator::Pointer create_iterator(void) override {
    return IndexHolder::Iterator::Pointer(
        new OnePassBinaryIndexHolder::Iterator(this));
  }

  //! Append an element into holder
  bool emplace(uint64_t key, const ailego::BinaryVector<T> &vec) {
    if (vec.size() != dimension_) {
      return false;
    }
    features_.emplace_back(key, vec);
    return true;
  }

  //! Append an element into holder
  bool emplace(uint64_t key, ailego::BinaryVector<T> &&vec) {
    if (vec.size() != dimension_) {
      return false;
    }
    features_.emplace_back(key, std::move(vec));
    return true;
  }

 private:
  //! Disable them
  OnePassBinaryIndexHolder(void) = delete;

  //! Members
  size_t dimension_{0};
  std::list<std::pair<uint64_t, ailego::BinaryVector<T>>> features_;
};

/*! Multi-Pass Binary Index Holder
 */
template <typename T>
class MultiPassBinaryIndexHolder : public IndexHolder {
 public:
  /*! Multi-Pass Index Holder Iterator
   */
  class Iterator : public IndexHolder::Iterator {
   public:
    //! Index Holder Iterator Pointer
    typedef std::unique_ptr<Iterator> Pointer;

    //! Constructor
    Iterator(MultiPassBinaryIndexHolder *owner) : holder_(owner) {
      features_iter_ = holder_->features_.begin();
    }

    //! Destructor
    virtual ~Iterator(void) {}

    //! Retrieve pointer of data
    const void *data(void) const override {
      return features_iter_->second.data();
    }

    //! Test if the iterator is valid
    bool is_valid(void) const override {
      return (features_iter_ != holder_->features_.end());
    }

    //! Retrieve primary key
    uint64_t key(void) const override {
      return features_iter_->first;
    }

    //! Next iterator
    void next(void) override {
      ++features_iter_;
    }

   private:
    MultiPassBinaryIndexHolder *holder_{nullptr};
    typename std::vector<std::pair<uint64_t, ailego::BinaryVector<T>>>::iterator
        features_iter_{};
  };

  //! Constructor
  MultiPassBinaryIndexHolder(size_t dim) : dimension_(dim) {}

  //! Retrieve count of elements in holder (-1 indicates unknown)
  size_t count(void) const override {
    return features_.size();
  }

  //! Retrieve dimension
  size_t dimension(void) const override {
    return dimension_;
  }

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_UNDEFINED;
  }

  //! Retrieve element size in bytes
  size_t element_size(void) const override {
    return (dimension_ + (sizeof(T) << 3) - 1) / (sizeof(T) << 3) * sizeof(T);
  }

  //! Retrieve if it can multi-pass
  bool multipass(void) const override {
    return true;
  }

  //! Create a new iterator
  IndexHolder::Iterator::Pointer create_iterator(void) override {
    return IndexHolder::Iterator::Pointer(
        new MultiPassBinaryIndexHolder::Iterator(this));
  }

  //! Append an element into holder
  bool emplace(uint64_t key, const ailego::BinaryVector<T> &vec) {
    if (vec.size() != dimension_) {
      return false;
    }
    features_.emplace_back(key, vec);
    return true;
  }

  //! Append an element into holder
  bool emplace(uint64_t key, ailego::BinaryVector<T> &&vec) {
    if (vec.size() != dimension_) {
      return false;
    }
    features_.emplace_back(key, std::move(vec));
    return true;
  }

  //! Request a change in capacity
  void reserve(size_t size) {
    features_.reserve(size);
  }

 private:
  //! Disable them
  MultiPassBinaryIndexHolder(void) = delete;

  //! Members
  size_t dimension_{0};
  std::vector<std::pair<uint64_t, ailego::BinaryVector<T>>> features_;
};

/*! One-Pass Index Hybrid Holder
 */
template <typename T>
class OnePassIndexHybridHolderBase : public IndexHybridHolder {
 public:
  /*! One-Pass Index Holder Iterator
   */
  class Iterator : public IndexHybridHolder::Iterator {
   public:
    //! Index Holder Iterator Pointer
    typedef std::unique_ptr<Iterator> Pointer;

    //! Constructor
    Iterator(OnePassIndexHybridHolderBase *owner) : holder_(owner) {
      features_iter_ = holder_->features_.begin();
    }

    //! Destructor
    virtual ~Iterator(void) {}

    //! Retrieve pointer of data
    const void *data(void) const override {
      return features_iter_->second.data();
    }

    //! Test if the iterator is valid
    bool is_valid(void) const override {
      return (features_iter_ != holder_->features_.end());
    }

    //! Retrieve primary key
    uint64_t key(void) const override {
      return features_iter_->first;
    }

    //! Next iterator
    void next(void) override {
      holder_->features_.erase(features_iter_++);
    }

    //! Retrieve primary key
    uint32_t sparse_count() const override {
      return features_iter_->second.sparse_count();
    }

    //! Retrieve primary key
    const uint32_t *sparse_indices() const override {
      return features_iter_->second.sparse_indices();
    }

    //! Retrieve primary key
    const void *sparse_data() const override {
      return features_iter_->second.sparse_data();
    }

   private:
    OnePassIndexHybridHolderBase *holder_{nullptr};
    typename std::list<std::pair<uint64_t, ailego::HybridVector<T>>>::iterator
        features_iter_{};
  };

  //! Constructor
  OnePassIndexHybridHolderBase(size_t dim) : dimension_(dim) {}

  //! Retrieve count of elements in holder (-1 indicates unknown)
  size_t count(void) const override {
    return features_.size();
  }

  //! Retrieve dimension
  size_t dimension(void) const override {
    return dimension_;
  }

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_UNDEFINED;
  }

  //! Retrieve element size in bytes
  size_t element_size(void) const override {
    return dimension_ * sizeof(T);
  }

  //! Retrieve if it can multi-pass
  bool multipass(void) const override {
    return false;
  }

  //! Create a new iterator
  IndexHolder::Iterator::Pointer create_iterator(void) override {
    return IndexHolder::Iterator::Pointer(
        new OnePassIndexHybridHolderBase::Iterator(this));
  }

  //! Create a new hybrid iterator
  IndexHybridHolder::Iterator::Pointer create_hybrid_iterator(void) override {
    return IndexHybridHolder::Iterator::Pointer(
        new OnePassIndexHybridHolderBase::Iterator(this));
  }

  //! Retrieve sparse count summing up over all the docs
  size_t total_sparse_count(void) const override {
    return total_sparse_count_;
    ;
  }

  //! Append an element into holder
  bool emplace(uint64_t key, const ailego::HybridVector<T> &vec) {
    if (vec.size() != dimension_) {
      return false;
    }
    features_.emplace_back(key, vec);

    total_sparse_count_ += vec.sparse_count();

    return true;
  }

  //! Append an element into holder
  bool emplace(uint64_t key, ailego::HybridVector<T> &&vec) {
    if (vec.size() != dimension_) {
      return false;
    }
    features_.emplace_back(key, std::move(vec));

    total_sparse_count_ += vec.sparse_count();

    return true;
  }

 private:
  //! Disable them
  OnePassIndexHybridHolderBase(void) = delete;

  //! Members
  size_t dimension_{0};
  std::list<std::pair<uint64_t, ailego::HybridVector<T>>> features_;
  size_t total_sparse_count_{0};
};

/*! Multi-Pass Index Hybrid Holder Base
 */
template <typename T>
class MultiPassIndexHybridHolderBase : public IndexHybridHolder {
 public:
  /*! Multi-Pass Index Holder Iterator
   */
  class Iterator : public IndexHybridHolder::Iterator {
   public:
    //! Index Holder Iterator Pointer
    typedef std::unique_ptr<Iterator> Pointer;

    //! Constructor
    Iterator(MultiPassIndexHybridHolderBase *owner) : holder_(owner) {
      features_iter_ = holder_->features_.begin();
    }

    //! Destructor
    virtual ~Iterator(void) {}

    //! Retrieve pointer of data
    const void *data(void) const override {
      return features_iter_->second.data();
    }

    //! Test if the iterator is valid
    bool is_valid(void) const override {
      return (features_iter_ != holder_->features_.end());
    }

    //! Retrieve primary key
    uint64_t key(void) const override {
      return features_iter_->first;
    }

    //! Next iterator
    void next(void) override {
      ++features_iter_;
    }

    //! Retrieve primary key
    uint32_t sparse_count() const override {
      return features_iter_->second.sparse_count();
    }

    //! Retrieve primary key
    const uint32_t *sparse_indices() const override {
      return features_iter_->second.sparse_indices();
    }

    //! Retrieve primary key
    const void *sparse_data() const override {
      return features_iter_->second.sparse_data();
    }

   private:
    MultiPassIndexHybridHolderBase *holder_{nullptr};
    typename std::vector<std::pair<uint64_t, ailego::HybridVector<T>>>::iterator
        features_iter_{};
  };

  //! Constructor
  MultiPassIndexHybridHolderBase(size_t dim) : dimension_(dim) {}

  //! Retrieve count of elements in holder (-1 indicates unknown)
  size_t count(void) const override {
    return features_.size();
  }

  //! Retrieve dimension
  size_t dimension(void) const override {
    return dimension_;
  }

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_UNDEFINED;
  }

  //! Retrieve element size in bytes
  size_t element_size(void) const override {
    return dimension_ * sizeof(T);
  }

  //! Retrieve if it can multi-pass
  bool multipass(void) const override {
    return true;
  }

  //! Create a new iterator
  IndexHolder::Iterator::Pointer create_iterator(void) override {
    return IndexHolder::Iterator::Pointer(
        new MultiPassIndexHybridHolderBase::Iterator(this));
  }

  //! Create a new hybrid iterator
  IndexHybridHolder::Iterator::Pointer create_hybrid_iterator(void) override {
    return IndexHybridHolder::Iterator::Pointer(
        new MultiPassIndexHybridHolderBase::Iterator(this));
  }

  //! Retrieve sparse count summing up over all the docs
  size_t total_sparse_count(void) const override {
    return 0;
  }

  //! Append an element into holder
  bool emplace(uint64_t key, const ailego::HybridVector<T> &vec) {
    if (vec.size() != dimension_) {
      return false;
    }

    features_.emplace_back(key, vec);

    total_sparse_count_ += vec.sparse_count();

    return true;
  }

  //! Append an element into holder
  bool emplace(uint64_t key, ailego::HybridVector<T> &&vec) {
    if (vec.size() != dimension_) {
      return false;
    }
    features_.emplace_back(key, std::move(vec));

    total_sparse_count_ += vec.sparse_count();

    return true;
  }

  //! Request a change in capacity
  void reserve(size_t size) {
    features_.reserve(size);
  }

 private:
  //! Disable them
  MultiPassIndexHybridHolderBase(void) = delete;

  //! Members
  size_t dimension_{0};
  std::vector<std::pair<uint64_t, ailego::HybridVector<T>>> features_;
  size_t total_sparse_count_{0};
};

/*! One-Pass Index Sparse Holder
 */
template <typename T>
class OnePassIndexSparseHolderBase : public IndexSparseHolder {
 public:
  /*! One-Pass Index Holder Iterator
   */
  class Iterator : public IndexSparseHolder::Iterator {
   public:
    //! Index Holder Iterator Pointer
    typedef std::unique_ptr<Iterator> Pointer;

    //! Constructor
    Iterator(OnePassIndexSparseHolderBase *owner) : holder_(owner) {
      features_iter_ = holder_->features_.begin();
    }

    //! Destructor
    virtual ~Iterator(void) {}

    //! Test if the iterator is valid
    bool is_valid(void) const override {
      return (features_iter_ != holder_->features_.end());
    }

    //! Retrieve primary key
    uint64_t key(void) const override {
      return features_iter_->first;
    }

    //! Next iterator
    void next(void) override {
      holder_->features_.erase(features_iter_++);
    }

    //! Retrieve primary key
    uint32_t sparse_count() const override {
      return features_iter_->second.sparse_count();
    }

    //! Retrieve primary key
    const uint32_t *sparse_indices() const override {
      return features_iter_->second.sparse_indices();
    }

    //! Retrieve primary key
    const void *sparse_data() const override {
      return features_iter_->second.sparse_data();
    }

   private:
    OnePassIndexSparseHolderBase *holder_{nullptr};
    typename std::list<std::pair<uint64_t, ailego::SparseVector<T>>>::iterator
        features_iter_{};
  };

  //! Constructor
  OnePassIndexSparseHolderBase() {}

  //! Retrieve count of elements in holder (-1 indicates unknown)
  size_t count(void) const override {
    return features_.size();
  }

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_UNDEFINED;
  }

  //! Retrieve if it can multi-pass
  bool multipass(void) const override {
    return false;
  }

  //! Create a new iterator
  IndexSparseHolder::Iterator::Pointer create_iterator(void) override {
    return IndexSparseHolder::Iterator::Pointer(
        new OnePassIndexSparseHolderBase::Iterator(this));
  }

  //! Retrieve sparse count summing up over all the docs
  size_t total_sparse_count(void) const override {
    return total_sparse_count_;
    ;
  }

  //! Append an element into holder
  bool emplace(uint64_t key, const ailego::SparseVector<T> &vec) {
    features_.emplace_back(key, vec);

    total_sparse_count_ += vec.sparse_count();

    return true;
  }

  //! Append an element into holder
  bool emplace(uint64_t key, ailego::SparseVector<T> &&vec) {
    features_.emplace_back(key, std::move(vec));

    total_sparse_count_ += vec.sparse_count();

    return true;
  }

 private:
  //! Members
  std::list<std::pair<uint64_t, ailego::SparseVector<T>>> features_;
  size_t total_sparse_count_{0};
};

/*! Multi-Pass Index Sparse Holder Base
 */
template <typename T>
class MultiPassIndexSparseHolderBase : public IndexSparseHolder {
 public:
  /*! Multi-Pass Index Holder Iterator
   */
  class Iterator : public IndexSparseHolder::Iterator {
   public:
    //! Index Holder Iterator Pointer
    typedef std::unique_ptr<Iterator> Pointer;

    //! Constructor
    Iterator(MultiPassIndexSparseHolderBase *owner) : holder_(owner) {
      features_iter_ = holder_->features_.begin();
    }

    //! Destructor
    virtual ~Iterator(void) {}

    //! Test if the iterator is valid
    bool is_valid(void) const override {
      return (features_iter_ != holder_->features_.end());
    }

    //! Retrieve primary key
    uint64_t key(void) const override {
      return features_iter_->first;
    }

    //! Next iterator
    void next(void) override {
      ++features_iter_;
    }

    //! Retrieve primary key
    uint32_t sparse_count() const override {
      return features_iter_->second.sparse_count();
    }

    //! Retrieve primary key
    const uint32_t *sparse_indices() const override {
      return features_iter_->second.sparse_indices();
    }

    //! Retrieve primary key
    const void *sparse_data() const override {
      return features_iter_->second.sparse_data();
    }

   private:
    MultiPassIndexSparseHolderBase *holder_{nullptr};
    typename std::vector<std::pair<uint64_t, ailego::SparseVector<T>>>::iterator
        features_iter_{};
  };

  //! Constructor
  MultiPassIndexSparseHolderBase() {}

  //! Retrieve count of elements in holder (-1 indicates unknown)
  size_t count(void) const override {
    return features_.size();
  }

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_UNDEFINED;
  }

  //! Retrieve if it can multi-pass
  bool multipass(void) const override {
    return true;
  }

  //! Create a new iterator
  IndexSparseHolder::Iterator::Pointer create_iterator(void) override {
    return IndexSparseHolder::Iterator::Pointer(
        new MultiPassIndexSparseHolderBase::Iterator(this));
  }

  //! Retrieve sparse count summing up over all the docs
  size_t total_sparse_count(void) const override {
    return 0;
  }

  //! Append an element into holder
  bool emplace(uint64_t key, const ailego::SparseVector<T> &vec) {
    features_.emplace_back(key, vec);

    total_sparse_count_ += vec.sparse_count();

    return true;
  }

  //! Append an element into holder
  bool emplace(uint64_t key, ailego::SparseVector<T> &&vec) {
    features_.emplace_back(key, std::move(vec));

    total_sparse_count_ += vec.sparse_count();

    return true;
  }

  //! Request a change in capacity
  void reserve(size_t size) {
    features_.reserve(size);
  }

 private:
  //! Members
  std::vector<std::pair<uint64_t, ailego::SparseVector<T>>> features_;
  size_t total_sparse_count_{0};
};

/*! One-Pass Index Holder
 */
template <IndexMeta::DataType FT>
struct OnePassIndexHolder;

/*! One-Pass Index Holder (BINARY32)
 */
template <>
struct OnePassIndexHolder<IndexMeta::DataType::DT_BINARY32>
    : public OnePassBinaryIndexHolder<uint32_t> {
  //! Constructor
  using OnePassBinaryIndexHolder::OnePassBinaryIndexHolder;

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_BINARY32;
  }
};

/*! One-Pass Index Holder (BINARY64)
 */
template <>
struct OnePassIndexHolder<IndexMeta::DataType::DT_BINARY64>
    : public OnePassBinaryIndexHolder<uint64_t> {
  //! Constructor
  using OnePassBinaryIndexHolder::OnePassBinaryIndexHolder;

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_BINARY64;
  }
};

/*! One-Pass Index Holder (FP16)
 */
template <>
struct OnePassIndexHolder<IndexMeta::DataType::DT_FP16>
    : public OnePassNumericalIndexHolder<ailego::Float16> {
  //! Constructor
  using OnePassNumericalIndexHolder::OnePassNumericalIndexHolder;

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_FP16;
  }
};

/*! One-Pass Index Holder (FP32)
 */
template <>
struct OnePassIndexHolder<IndexMeta::DataType::DT_FP32>
    : public OnePassNumericalIndexHolder<float> {
  //! Constructor
  using OnePassNumericalIndexHolder::OnePassNumericalIndexHolder;

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_FP32;
  }
};

/*! One-Pass Index Holder (FP64)
 */
template <>
struct OnePassIndexHolder<IndexMeta::DataType::DT_FP64>
    : public OnePassNumericalIndexHolder<double> {
  //! Constructor
  using OnePassNumericalIndexHolder::OnePassNumericalIndexHolder;

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_FP64;
  }
};

/*! One-Pass Index Holder (INT8)
 */
template <>
struct OnePassIndexHolder<IndexMeta::DataType::DT_INT8>
    : public OnePassNumericalIndexHolder<int8_t> {
  //! Constructor
  using OnePassNumericalIndexHolder::OnePassNumericalIndexHolder;

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_INT8;
  }
};

/*! One-Pass Index Holder (INT16)
 */
template <>
struct OnePassIndexHolder<IndexMeta::DataType::DT_INT16>
    : public OnePassNumericalIndexHolder<int16_t> {
  //! Constructor
  using OnePassNumericalIndexHolder::OnePassNumericalIndexHolder;

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_INT16;
  }
};

/*! Multi-Pass Index Holder
 */
template <IndexMeta::DataType FT>
struct MultiPassIndexHolder;

/*! Multi-Pass Index Holder (BINARY32)
 */
template <>
struct MultiPassIndexHolder<IndexMeta::DataType::DT_BINARY32>
    : public MultiPassBinaryIndexHolder<uint32_t> {
  //! Constructor
  using MultiPassBinaryIndexHolder::MultiPassBinaryIndexHolder;

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_BINARY32;
  }
};

/*! Multi-Pass Index Holder (BINARY64)
 */
template <>
struct MultiPassIndexHolder<IndexMeta::DataType::DT_BINARY64>
    : public MultiPassBinaryIndexHolder<uint64_t> {
  //! Constructor
  using MultiPassBinaryIndexHolder::MultiPassBinaryIndexHolder;

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_BINARY64;
  }
};

/*! Multi-Pass Index Holder (FP16)
 */
template <>
struct MultiPassIndexHolder<IndexMeta::DataType::DT_FP16>
    : public MultiPassNumericalIndexHolder<ailego::Float16> {
  //! Constructor
  using MultiPassNumericalIndexHolder::MultiPassNumericalIndexHolder;

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_FP16;
  }
};

/*! Multi-Pass Index Holder (FP32)
 */
template <>
struct MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>
    : public MultiPassNumericalIndexHolder<float> {
  //! Constructor
  using MultiPassNumericalIndexHolder::MultiPassNumericalIndexHolder;

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_FP32;
  }
};

/*! Multi-Pass Index Holder (FP64)
 */
template <>
struct MultiPassIndexHolder<IndexMeta::DataType::DT_FP64>
    : public MultiPassNumericalIndexHolder<double> {
  //! Constructor
  using MultiPassNumericalIndexHolder::MultiPassNumericalIndexHolder;

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_FP64;
  }
};

/*! Multi-Pass Index Holder (INT8)
 */
template <>
struct MultiPassIndexHolder<IndexMeta::DataType::DT_INT8>
    : public MultiPassNumericalIndexHolder<int8_t> {
  //! Constructor
  using MultiPassNumericalIndexHolder::MultiPassNumericalIndexHolder;

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_INT8;
  }
};

/*! Multi-Pass Index Holder (INT16)
 */
template <>
struct MultiPassIndexHolder<IndexMeta::DataType::DT_INT16>
    : public MultiPassNumericalIndexHolder<int16_t> {
  //! Constructor
  using MultiPassNumericalIndexHolder::MultiPassNumericalIndexHolder;

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_INT16;
  }
};

/*! One-Pass Index Hybrid Holder
 */
template <IndexMeta::DataType FT>
struct OnePassIndexHybridHolder;

/*! One-Pass Index Hybrid Holder (FP16)
 */
template <>
struct OnePassIndexHybridHolder<IndexMeta::DataType::DT_FP16>
    : public OnePassIndexHybridHolderBase<ailego::Float16> {
  //! Constructor
  using OnePassIndexHybridHolderBase::OnePassIndexHybridHolderBase;

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_FP16;
  }
};

/*! One-Pass Index Hybrid Holder (FP32)
 */
template <>
struct OnePassIndexHybridHolder<IndexMeta::DataType::DT_FP32>
    : public OnePassIndexHybridHolderBase<float> {
  //! Constructor
  using OnePassIndexHybridHolderBase::OnePassIndexHybridHolderBase;

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_FP32;
  }
};

/*! One-Pass Index Hybrid Holder (FP64)
 */
template <>
struct OnePassIndexHybridHolder<IndexMeta::DataType::DT_FP64>
    : public OnePassIndexHybridHolderBase<double> {
  //! Constructor
  using OnePassIndexHybridHolderBase::OnePassIndexHybridHolderBase;

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_FP64;
  }
};

/*! One-Pass Index Hybrid Holder (INT8)
 */
template <>
struct OnePassIndexHybridHolder<IndexMeta::DataType::DT_INT8>
    : public OnePassIndexHybridHolderBase<int8_t> {
  //! Constructor
  using OnePassIndexHybridHolderBase::OnePassIndexHybridHolderBase;

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_INT8;
  }
};

/*! One-Pass Index Hybrid Holder (INT16)
 */
template <>
struct OnePassIndexHybridHolder<IndexMeta::DataType::DT_INT16>
    : public OnePassIndexHybridHolderBase<int16_t> {
  //! Constructor
  using OnePassIndexHybridHolderBase::OnePassIndexHybridHolderBase;

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_INT16;
  }
};

/*! Multi-Pass Index Hybrid Holder
 */
template <IndexMeta::DataType FT>
struct MultiPassIndexHybridHolder;

/*! Multi-Pass Index Hybrid Holder (FP16)
 */
template <>
struct MultiPassIndexHybridHolder<IndexMeta::DataType::DT_FP16>
    : public MultiPassIndexHybridHolderBase<ailego::Float16> {
  //! Constructor
  using MultiPassIndexHybridHolderBase::MultiPassIndexHybridHolderBase;

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_FP16;
  }
};

/*! Multi-Pass Index Hybrid Holder (FP32)
 */
template <>
struct MultiPassIndexHybridHolder<IndexMeta::DataType::DT_FP32>
    : public MultiPassIndexHybridHolderBase<float> {
  //! Constructor
  using MultiPassIndexHybridHolderBase::MultiPassIndexHybridHolderBase;

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_FP32;
  }
};

/*! Multi-Pass Index Hybrid Holder (FP64)
 */
template <>
struct MultiPassIndexHybridHolder<IndexMeta::DataType::DT_FP64>
    : public MultiPassIndexHybridHolderBase<double> {
  //! Constructor
  using MultiPassIndexHybridHolderBase::MultiPassIndexHybridHolderBase;

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_FP64;
  }
};

/*! Multi-Pass Index Hybrid Holder (INT8)
 */
template <>
struct MultiPassIndexHybridHolder<IndexMeta::DataType::DT_INT8>
    : public MultiPassIndexHybridHolderBase<int8_t> {
  //! Constructor
  using MultiPassIndexHybridHolderBase::MultiPassIndexHybridHolderBase;

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_INT8;
  }
};

/*! Multi-Pass Index Hybrid Holder (INT16)
 */
template <>
struct MultiPassIndexHybridHolder<IndexMeta::DataType::DT_INT16>
    : public MultiPassIndexHybridHolderBase<int16_t> {
  //! Constructor
  using MultiPassIndexHybridHolderBase::MultiPassIndexHybridHolderBase;

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_INT16;
  }
};

/*! One-Pass Index Sparse Holder
 */
template <IndexMeta::DataType FT>
struct OnePassIndexSparseHolder;

/*! One-Pass Index Sparse Holder (FP16)
 */
template <>
struct OnePassIndexSparseHolder<IndexMeta::DataType::DT_FP16>
    : public OnePassIndexSparseHolderBase<ailego::Float16> {
  //! Constructor
  using OnePassIndexSparseHolderBase::OnePassIndexSparseHolderBase;

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_FP16;
  }
};

/*! One-Pass Index Sparse Holder (FP32)
 */
template <>
struct OnePassIndexSparseHolder<IndexMeta::DataType::DT_FP32>
    : public OnePassIndexSparseHolderBase<float> {
  //! Constructor
  using OnePassIndexSparseHolderBase::OnePassIndexSparseHolderBase;

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_FP32;
  }
};

/*! One-Pass Index Sparse Holder (FP64)
 */
template <>
struct OnePassIndexSparseHolder<IndexMeta::DataType::DT_FP64>
    : public OnePassIndexSparseHolderBase<double> {
  //! Constructor
  using OnePassIndexSparseHolderBase::OnePassIndexSparseHolderBase;

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_FP64;
  }
};

/*! One-Pass Index Sparse Holder (INT8)
 */
template <>
struct OnePassIndexSparseHolder<IndexMeta::DataType::DT_INT8>
    : public OnePassIndexSparseHolderBase<int8_t> {
  //! Constructor
  using OnePassIndexSparseHolderBase::OnePassIndexSparseHolderBase;

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_INT8;
  }
};

/*! One-Pass Index Sparse Holder (INT16)
 */
template <>
struct OnePassIndexSparseHolder<IndexMeta::DataType::DT_INT16>
    : public OnePassIndexSparseHolderBase<int16_t> {
  //! Constructor
  using OnePassIndexSparseHolderBase::OnePassIndexSparseHolderBase;

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_INT16;
  }
};

/*! Multi-Pass Index Sparse Holder
 */
template <IndexMeta::DataType FT>
struct MultiPassIndexSparseHolder;

/*! Multi-Pass Index Sparse Holder (FP16)
 */
template <>
struct MultiPassIndexSparseHolder<IndexMeta::DataType::DT_FP16>
    : public MultiPassIndexSparseHolderBase<ailego::Float16> {
  //! Constructor
  using MultiPassIndexSparseHolderBase::MultiPassIndexSparseHolderBase;

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_FP16;
  }
};

/*! Multi-Pass Index Sparse Holder (FP32)
 */
template <>
struct MultiPassIndexSparseHolder<IndexMeta::DataType::DT_FP32>
    : public MultiPassIndexSparseHolderBase<float> {
  //! Constructor
  using MultiPassIndexSparseHolderBase::MultiPassIndexSparseHolderBase;

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_FP32;
  }
};

/*! Multi-Pass Index Sparse Holder (FP64)
 */
template <>
struct MultiPassIndexSparseHolder<IndexMeta::DataType::DT_FP64>
    : public MultiPassIndexSparseHolderBase<double> {
  //! Constructor
  using MultiPassIndexSparseHolderBase::MultiPassIndexSparseHolderBase;

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_FP64;
  }
};

/*! Multi-Pass Index Sparse Holder (INT8)
 */
template <>
struct MultiPassIndexSparseHolder<IndexMeta::DataType::DT_INT8>
    : public MultiPassIndexSparseHolderBase<int8_t> {
  //! Constructor
  using MultiPassIndexSparseHolderBase::MultiPassIndexSparseHolderBase;

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_INT8;
  }
};

/*! Multi-Pass Index Sparse Holder (INT16)
 */
template <>
struct MultiPassIndexSparseHolder<IndexMeta::DataType::DT_INT16>
    : public MultiPassIndexSparseHolderBase<int16_t> {
  //! Constructor
  using MultiPassIndexSparseHolderBase::MultiPassIndexSparseHolderBase;

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_INT16;
  }
};

/*! Random Access Index Holder
 */
class RandomAccessIndexHolder : public IndexHolder {
 public:
  //! Index Holder Iterator Pointer
  typedef std::shared_ptr<RandomAccessIndexHolder> Pointer;

  /*! Random Access Index Holder Iterator
   */
  class Iterator : public IndexHolder::Iterator {
   public:
    //! Index Holder Iterator Pointer
    typedef std::unique_ptr<Iterator> Pointer;

    //! Constructor
    Iterator(RandomAccessIndexHolder *owner) : holder_(owner) {}

    //! Destructor
    virtual ~Iterator(void) {}

    //! Retrieve pointer of data
    virtual const void *data(void) const override {
      return holder_->element(id_);
    }

    //! Test if the iterator is valid
    virtual bool is_valid(void) const override {
      return id_ < holder_->count();
    }

    //! Retrieve primary key
    virtual uint64_t key(void) const override {
      return holder_->key(id_);
    }

    //! Next iterator
    virtual void next(void) override {
      ++id_;
    }

   private:
    //! Members
    RandomAccessIndexHolder *holder_{nullptr};
    uint32_t id_{0};
  };

  //! Constructor
  RandomAccessIndexHolder(const IndexMeta &meta)
      : features_(std::make_shared<CompactIndexFeatures>(meta)) {}

  //! Retrieve count of elements in holder (-1 indicates unknown)
  virtual size_t count(void) const override {
    return features_->count();
  }

  //! Retrieve dimension
  virtual size_t dimension(void) const override {
    return features_->dimension();
  }

  //! Retrieve type information
  virtual IndexMeta::DataType data_type(void) const override {
    return features_->data_type();
  }

  //! Retrieve element size in bytes
  virtual size_t element_size(void) const override {
    return features_->element_size();
  }

  //! Retrieve if it can multi-pass
  virtual bool multipass(void) const override {
    return true;
  }

  //! Create a new iterator
  virtual IndexHolder::Iterator::Pointer create_iterator(void) override {
    return IndexHolder::Iterator::Pointer(
        new RandomAccessIndexHolder::Iterator(this));
  }

  void reserve(size_t elems) {
    features_->reserve(elems);
    keys_.reserve(elems);
  }

  //! Append an element into holder
  void emplace(uint64_t pkey, const void *vec) {
    features_->emplace(vec);
    keys_.emplace_back(pkey);
  }

  //! Retrieve feature via local id
  const void *element(size_t id) const {
    return features_->element(id);
  }

  //! Retrieve key via local id
  uint64_t key(size_t id) const {
    ailego_assert_with(id < keys_.size(), "Index Overflow");
    return keys_[id];
  }

 private:
  //! Disable them
  RandomAccessIndexHolder(void) = delete;

  //! Members
  CompactIndexFeatures::Pointer features_{};
  std::vector<uint64_t> keys_{};
};

}  // namespace core
}  // namespace zvec
