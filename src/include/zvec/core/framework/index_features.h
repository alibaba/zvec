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
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <zvec/core/framework/index_meta.h>

namespace zvec {
namespace core {

/*! Index Features
 */
struct IndexFeatures {
  //! Index Features Pointer
  typedef std::shared_ptr<IndexFeatures> Pointer;

  //! Destructor
  virtual ~IndexFeatures(void) {}

  //! Retrieve feature via index
  virtual const void *element(size_t i) const = 0;

  //! Retrieve count of elements
  virtual size_t count(void) const = 0;

  //! Retrieve dimension
  virtual size_t dimension(void) const = 0;

  //! Retrieve type information
  virtual IndexMeta::DataType data_type(void) const = 0;

  //! Test if it is a compacted buffer
  virtual bool is_compacted(void) const {
    return false;
  }

  //! Retrieve pointer of compacted buffer
  virtual const void *data(void) const {
    return nullptr;
  }

  //! Retrieve size of feature
  virtual size_t element_size(void) const {
    return IndexMeta::ElementSizeof(this->data_type(), this->dimension());
  }

  //! Operator []
  const void *operator[](size_t i) const {
    return this->element(i);
  }

  //! Test if matchs the meta
  bool is_matched(const IndexMeta &meta) const {
    return (meta.data_type() == this->data_type() &&
            meta.dimension() == this->dimension() &&
            meta.element_size() == this->element_size());
  }
};

/*! Coherent Index Features
 */
class CoherentIndexFeatures : public IndexFeatures {
 public:
  //! Coherent Index Features Pointer
  typedef std::shared_ptr<CoherentIndexFeatures> Pointer;

  //! Constructor
  CoherentIndexFeatures(void)
      : features_buffer_(nullptr),
        features_count_(0),
        feature_size_(0),
        feature_dimension_(0),
        data_type_(IndexMeta::DataType::DT_UNDEFINED) {}

  //! Constructor
  CoherentIndexFeatures(const IndexMeta &meta)
      : features_buffer_(nullptr),
        features_count_(0),
        feature_size_(meta.element_size()),
        feature_dimension_(meta.dimension()),
        data_type_(meta.data_type()) {}

  //! Constructor
  CoherentIndexFeatures(const IndexMeta &meta, const void *buf, size_t len)
      : features_buffer_(buf),
        features_count_(len / meta.element_size()),
        feature_size_(meta.element_size()),
        feature_dimension_(meta.dimension()),
        data_type_(meta.data_type()) {}

  //! Mount features
  void mount(const IndexMeta &meta, const void *buf, size_t len) {
    features_buffer_ = buf;
    data_type_ = meta.data_type();
    feature_size_ = meta.element_size();
    feature_dimension_ = meta.dimension();
    features_count_ = len / feature_size_;
  }

  //! Mount features
  void mount(const void *buf, size_t len) {
    features_buffer_ = buf;
    features_count_ = len / feature_size_;
  }

  //! Retrieve count of elements
  virtual size_t count(void) const {
    return features_count_;
  }

  //! Retrieve dimension
  virtual size_t dimension(void) const {
    return feature_dimension_;
  }

  //! Retrieve feature via index
  virtual const void *element(size_t i) const {
    return (reinterpret_cast<const char *>(features_buffer_) +
            feature_size_ * i);
  }

  //! Retrieve type information
  virtual IndexMeta::DataType data_type(void) const {
    return data_type_;
  }

  //! Test if it is a compacted buffer
  virtual bool is_compacted(void) const {
    return true;
  }

  //! Retrieve pointer of compacted buffer
  virtual const void *data(void) const {
    return features_buffer_;
  }

  //! Retrieve size of feature
  virtual size_t element_size(void) const {
    return feature_size_;
  }

 private:
  const void *features_buffer_;
  size_t features_count_;
  size_t feature_size_;
  size_t feature_dimension_;
  IndexMeta::DataType data_type_;
};

/*! Flexible Index Features
 */
class FlexibleIndexFeatures : public IndexFeatures {
 public:
  //! Flexible Index Features Pointer
  typedef std::shared_ptr<FlexibleIndexFeatures> Pointer;

  //! Constructor
  FlexibleIndexFeatures(void)
      : features_(nullptr),
        features_count_(0),
        feature_size_(0),
        feature_dimension_(0),
        data_type_(IndexMeta::DataType::DT_UNDEFINED) {}

  //! Constructor
  FlexibleIndexFeatures(const IndexMeta &meta)
      : features_(nullptr),
        features_count_(0),
        feature_size_(meta.element_size()),
        feature_dimension_(meta.dimension()),
        data_type_(meta.data_type()) {}

  //! Constructor
  FlexibleIndexFeatures(const IndexMeta &meta, const void *const *feats,
                        size_t feats_count)
      : features_(feats),
        features_count_(feats_count),
        feature_size_(meta.element_size()),
        feature_dimension_(meta.dimension()),
        data_type_(meta.data_type()) {}

  //! Mount features
  void mount(const IndexMeta &meta, const void *const *feats,
             size_t feats_count) {
    features_ = feats;
    features_count_ = feats_count;
    data_type_ = meta.data_type();
    feature_size_ = meta.element_size();
    feature_dimension_ = meta.dimension();
  }

  //! Mount features
  void mount(const void *const *feats, size_t feats_count) {
    features_ = feats;
    features_count_ = feats_count;
  }

  //! Retrieve count of elements
  virtual size_t count(void) const {
    return features_count_;
  }

  //! Retrieve dimension
  virtual size_t dimension(void) const {
    return feature_dimension_;
  }

  //! Retrieve feature via index
  virtual const void *element(size_t i) const {
    return *(features_ + i);
  }

  //! Retrieve type information
  virtual IndexMeta::DataType data_type(void) const {
    return data_type_;
  }

  //! Retrieve size of feature
  virtual size_t element_size(void) const {
    return feature_size_;
  }

 private:
  const void *const *features_;
  size_t features_count_;
  size_t feature_size_;
  size_t feature_dimension_;
  IndexMeta::DataType data_type_;
};

/*! Gap Index Features
 */
class GapIndexFeatures : public IndexFeatures {
 public:
  //! Gap Index Features Pointer
  typedef std::shared_ptr<GapIndexFeatures> Pointer;

  //! Constructor
  GapIndexFeatures(const IndexMeta &meta)
      : features_(),
        bucket_limit_(0),
        features_count_(0),
        feature_size_(meta.element_size()),
        feature_dimension_(meta.dimension()),
        data_type_(meta.data_type()) {
    if (feature_size_ >= 1024 * 1024) {
      bucket_limit_ = 64u;
    } else {
      bucket_limit_ = (1024 * 1024 * 64) / feature_size_;
    }
  }

  //! Constructor
  GapIndexFeatures(const GapIndexFeatures &rhs)
      : features_(rhs.features_),
        bucket_limit_(rhs.bucket_limit_),
        features_count_(rhs.features_count_),
        feature_size_(rhs.feature_size_),
        feature_dimension_(rhs.feature_dimension_),
        data_type_(rhs.data_type_) {}

  //! Constructor
  GapIndexFeatures(GapIndexFeatures &&rhs)
      : features_(std::move(rhs.features_)),
        bucket_limit_(rhs.bucket_limit_),
        features_count_(rhs.features_count_),
        feature_size_(rhs.feature_size_),
        feature_dimension_(rhs.feature_dimension_),
        data_type_(rhs.data_type_) {}

  //! Assignment
  GapIndexFeatures &operator=(const GapIndexFeatures &rhs) {
    features_ = rhs.features_;
    bucket_limit_ = rhs.bucket_limit_;
    features_count_ = rhs.features_count_;
    feature_size_ = rhs.feature_size_;
    feature_dimension_ = rhs.feature_dimension_;
    data_type_ = rhs.data_type_;
    return *this;
  }

  //! Assignment
  GapIndexFeatures &operator=(GapIndexFeatures &&rhs) {
    features_ = std::move(rhs.features_);
    bucket_limit_ = rhs.bucket_limit_;
    features_count_ = rhs.features_count_;
    feature_size_ = rhs.feature_size_;
    feature_dimension_ = rhs.feature_dimension_;
    data_type_ = rhs.data_type_;
    return *this;
  }

  //! Append a feature
  void emplace(const void *feat) {
    if (features_count_ % bucket_limit_ == 0) {
      std::string bucket;
      bucket.reserve(bucket_limit_ * feature_size_);
      bucket.assign(reinterpret_cast<const char *>(feat), feature_size_);
      features_.push_back(std::move(bucket));
    } else {
      features_[features_count_ / bucket_limit_].append(
          reinterpret_cast<const char *>(feat), feature_size_);
    }
    ++features_count_;
  }

  //! Replace a feature
  void replace(size_t i, const void *feat) {
    std::memcpy(const_cast<char *>(features_[i / bucket_limit_].data()) +
                    feature_size_ * (i % bucket_limit_),
                feat, feature_size_);
  }

  //! Clear the features
  void clear(void) {
    features_.clear();
    features_count_ = 0;
  }

  //! Retrieve feature via index
  void *at(size_t i) {
    return (const_cast<char *>(features_[i / bucket_limit_].data()) +
            feature_size_ * (i % bucket_limit_));
  }

  //! Retrieve feature via index
  const void *at(size_t i) const {
    return (features_[i / bucket_limit_].data() +
            feature_size_ * (i % bucket_limit_));
  }

  //! Retrieve count of elements
  virtual size_t count(void) const {
    return features_count_;
  }

  //! Retrieve dimension
  virtual size_t dimension(void) const {
    return feature_dimension_;
  }

  //! Retrieve feature via index
  virtual const void *element(size_t i) const {
    return this->at(i);
  }

  //! Retrieve type information
  virtual IndexMeta::DataType data_type(void) const {
    return data_type_;
  }

  //! Test if it is a compacted buffer
  virtual bool is_compacted(void) const {
    return (features_.size() == 1u);
  }

  //! Retrieve pointer of compacted buffer
  virtual const void *data(void) const {
    return (features_.size() == 1u ? features_.front().data() : nullptr);
  }

  //! Retrieve size of feature
  virtual size_t element_size(void) const {
    return feature_size_;
  }

 private:
  //! Disable them
  GapIndexFeatures(void) = delete;

  //! Members
  std::vector<std::string> features_;
  size_t bucket_limit_;
  size_t features_count_;
  size_t feature_size_;
  size_t feature_dimension_;
  IndexMeta::DataType data_type_;
};

/*! Compact Index Features
 */
class CompactIndexFeatures : public IndexFeatures {
 public:
  //! Compact Index Features Pointer
  typedef std::shared_ptr<CompactIndexFeatures> Pointer;

  //! Constructor
  CompactIndexFeatures(const IndexMeta &meta)
      : features_(),
        feature_size_(meta.element_size()),
        feature_dimension_(meta.dimension()),
        data_type_(meta.data_type()) {}

  //! Constructor
  CompactIndexFeatures(const CompactIndexFeatures &rhs)
      : features_(rhs.features_),
        feature_size_(rhs.feature_size_),
        feature_dimension_(rhs.feature_dimension_),
        data_type_(rhs.data_type_) {}

  //! Constructor
  CompactIndexFeatures(CompactIndexFeatures &&rhs)
      : features_(std::move(rhs.features_)),
        feature_size_(rhs.feature_size_),
        feature_dimension_(rhs.feature_dimension_),
        data_type_(rhs.data_type_) {}

  //! Assignment
  CompactIndexFeatures &operator=(const CompactIndexFeatures &rhs) {
    features_ = rhs.features_;
    feature_size_ = rhs.feature_size_;
    feature_dimension_ = rhs.feature_dimension_;
    data_type_ = rhs.data_type_;
    return *this;
  }

  //! Assignment
  CompactIndexFeatures &operator=(CompactIndexFeatures &&rhs) {
    features_ = std::move(rhs.features_);
    feature_size_ = rhs.feature_size_;
    feature_dimension_ = rhs.feature_dimension_;
    data_type_ = rhs.data_type_;
    return *this;
  }

  //! Append a feature
  void emplace(const void *feat) {
    features_.append(reinterpret_cast<const char *>(feat), feature_size_);
  }

  //! Replace a feature
  void replace(size_t i, const void *feat) {
    std::memcpy(const_cast<char *>(features_.data()) + feature_size_ * i, feat,
                feature_size_);
  }

  //! Resize the container
  void resize(size_t n) {
    features_.resize(feature_size_ * n);
  }

  //! Reserve the container
  void reserve(size_t n) {
    features_.reserve(feature_size_ * n);
  }

  //! Clear the features
  void clear(void) {
    features_.clear();
  }

  //! Retrieve feature via index
  void *at(size_t i) {
    return (const_cast<char *>(features_.data()) + feature_size_ * i);
  }

  //! Retrieve feature via index
  const void *at(size_t i) const {
    return (features_.data() + feature_size_ * i);
  }

  //! Retrieve count of elements
  virtual size_t count(void) const {
    return (features_.size() / feature_size_);
  }

  //! Retrieve dimension
  virtual size_t dimension(void) const {
    return feature_dimension_;
  }

  //! Retrieve feature via index
  virtual const void *element(size_t i) const {
    return this->at(i);
  }

  //! Retrieve type information
  virtual IndexMeta::DataType data_type(void) const {
    return data_type_;
  }

  //! Test if it is a compacted buffer
  virtual bool is_compacted(void) const {
    return true;
  }

  //! Retrieve pointer of compacted buffer
  virtual const void *data(void) const {
    return features_.data();
  }

  //! Retrieve size of feature
  virtual size_t element_size(void) const {
    return feature_size_;
  }

 private:
  //! Disable them
  CompactIndexFeatures(void) = delete;

  //! Members
  std::string features_;
  size_t feature_size_;
  size_t feature_dimension_;
  IndexMeta::DataType data_type_;
};

/*! Sample Index Features
 */
template <typename TBase>
class SampleIndexFeatures : public TBase {
 public:
  //! Sample Index Features Pointer
  typedef std::shared_ptr<SampleIndexFeatures<TBase>> Pointer;

  //! Constructor
  SampleIndexFeatures(const IndexMeta &meta, size_t cnt)
      : TBase(meta), samples_(std::max<size_t>(cnt, 1u)), total_(0), mt_() {}

  //! Constructor
  SampleIndexFeatures(const SampleIndexFeatures &rhs)
      : TBase(rhs), samples_(rhs.samples_), total_(rhs.total_), mt_() {}

  //! Constructor
  SampleIndexFeatures(SampleIndexFeatures &&rhs)
      : TBase(std::move(rhs)),
        samples_(rhs.samples_),
        total_(rhs.total_),
        mt_() {}

  //! Assignment
  SampleIndexFeatures &operator=(const SampleIndexFeatures &rhs) {
    TBase::operator=(static_cast<const TBase &>(rhs));
    samples_ = rhs.samples_;
    total_ = rhs.total_;
    return *this;
  }

  //! Assignment
  SampleIndexFeatures &operator=(SampleIndexFeatures &&rhs) {
    TBase::operator=(std::move(static_cast<TBase &&>(rhs)));
    samples_ = rhs.samples_;
    total_ = rhs.total_;
    return *this;
  }

  //! Retrieve count of samples
  size_t samples(void) const {
    return samples_;
  }

  //! Retrieve count of total
  size_t total(void) const {
    return total_;
  }

  //! Append a feature
  void emplace(const void *feat) {
    if (TBase::count() >= samples_) {
      std::uniform_int_distribution<size_t> dt(0, total_);
      size_t i = dt(mt_);

      if (i < samples_) {
        TBase::replace(i, feat);
      }
    } else {
      TBase::emplace(feat);
    }
    ++total_;
  }

  //! Clear the features
  void clear(void) {
    TBase::clear();
    total_ = 0;
  }

 private:
  //! Disable them
  SampleIndexFeatures(void) = delete;

  //! Members
  size_t samples_;
  size_t total_;
  std::mt19937 mt_;
};

}  // namespace core
}  // namespace zvec
