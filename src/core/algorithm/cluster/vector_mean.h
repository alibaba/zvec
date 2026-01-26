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

#include <cmath>
#include <cstring>
#include <type_traits>
#include <vector>
#include <ailego/container/vector.h>
#include <ailego/utility/type_helper.h>
#include <zvec/ailego/utility/float_helper.h>

namespace zvec {
namespace core {

/*! Vector Mean
 */
struct VectorMean {
  //! Destructor
  virtual ~VectorMean(void) {}

  //! Reset accumulator
  virtual void reset(void) = 0;

  //! Plus a vector
  virtual bool plus(const void *vec, size_t len) = 0;

  //! Retrieve the mean of vectors
  virtual bool mean(void *out, size_t len) const = 0;

  //! Retrieve the mean of vectors
  virtual void mean(std::string *out) const = 0;

  //! Merge another vector mean
  virtual bool merge(const VectorMean &rhs) = 0;

  //! Retrieve the count of vectors
  virtual size_t count(void) const = 0;

  //! Retrieve the dimension of vectors
  virtual size_t dimension(void) const = 0;
};

/*! Vector Mean Array
 */
struct VectorMeanArray {
  //! Destructor
  virtual ~VectorMeanArray(void) {}

  //! Operator []
  VectorMean &operator[](size_t i) {
    return this->at(i);
  }

  //! Operator []
  const VectorMean &operator[](size_t i) const {
    return this->at(i);
  }

  //! Resize accumulators
  virtual void resize(size_t cnt) = 0;

  //! Clear accumulators
  virtual void clear(void) = 0;

  //! Retrieve an accumulator
  virtual VectorMean &at(size_t i) = 0;

  //! Retrieve an accumulator
  virtual const VectorMean &at(size_t i) const = 0;

  //! Retrieve the count of accumulators
  virtual size_t count(void) const = 0;

  //! Retrieve the dimension of accumulators
  virtual size_t dimension(void) const = 0;
};

/*! General Vector Mean Array
 */
template <typename T, typename = typename std::is_base_of<VectorMean, T>::type>
class GeneralVectorMeanArray : public VectorMeanArray {
 public:
  //! Constructor
  GeneralVectorMeanArray(size_t dim) : dimension_(dim), array_() {}

  //! Constructor
  GeneralVectorMeanArray(const GeneralVectorMeanArray &rhs)
      : dimension_(rhs.dimension_), array_(rhs.array_) {}

  //! Constructor
  GeneralVectorMeanArray(GeneralVectorMeanArray &&rhs)
      : dimension_(rhs.dimension_), array_(std::move(rhs.array_)) {}

  //! Emplace an accumulator
  template <typename... TArgs>
  bool emplace(TArgs &&...args) {
    T accum(std::forward<TArgs>(args)...);
    if (accum.dimension() != dimension_) {
      return false;
    }
    array_.push_back(std::move(accum));
    return true;
  }

  //! Resize accumulators
  virtual void resize(size_t cnt) {
    if (array_.size() < cnt) {
      for (size_t i = array_.size(); i < cnt; ++i) {
        array_.emplace_back(dimension_);
      }
    } else {
      array_.resize(cnt);
    }
  }

  //! Clear accumulators
  virtual void clear(void) {
    array_.clear();
  }

  //! Retrieve an accumulator
  virtual VectorMean &at(size_t i) {
    return array_[i];
  }

  //! Retrieve an accumulator
  virtual const VectorMean &at(size_t i) const {
    return array_[i];
  }

  //! Retrieve the count of accumulators
  virtual size_t count(void) const {
    return array_.size();
  }

  //! Retrieve the dimension of accumulators
  virtual size_t dimension(void) const {
    return dimension_;
  }

 private:
  //! Disable them
  GeneralVectorMeanArray(void) = delete;

  //! Members
  size_t dimension_;
  std::vector<T> array_;
};

/*! Numerical Vector Mean
 */
template <typename T,
          typename =
              typename std::enable_if<ailego::IsArithmetic<T>::value>::type>
class NumericalVectorMean : public VectorMean {
 public:
  //! Constructor
  NumericalVectorMean(void) : count_(0), accums_() {}

  //! Constructor
  NumericalVectorMean(const NumericalVectorMean &rhs)
      : count_(rhs.count_), accums_(rhs.accums_) {}

  //! Constructor
  NumericalVectorMean(NumericalVectorMean &&rhs)
      : count_(rhs.count_), accums_(std::move(rhs.accums_)) {}

  //! Constructor
  NumericalVectorMean(size_t dim) : count_(0), accums_(dim) {}

  //! Constructor
  NumericalVectorMean(const T *means, size_t dim, size_t cnt)
      : count_(cnt), accums_(dim) {
    for (size_t i = 0; i < dim; ++i) {
      accums_[i] = static_cast<double>(means[i]) * count_;
    }
  }

  //! Reset accumulator
  void reset(size_t dim) {
    count_ = 0u;
    accums_.clear();
    accums_.resize(dim, 0.0);
  }

  //! Reset accumulator
  virtual void reset(void) {
    this->reset(accums_.size());
  }

  //! Plus a vector
  virtual bool plus(const void *vec, size_t len) {
    size_t dim = accums_.size();
    if (dim * sizeof(T) != len) {
      return false;
    }
    for (size_t i = 0; i < dim; ++i) {
      accums_[i] += *(static_cast<const T *>(vec) + i);
    }
    ++count_;
    return true;
  }

  //! Retrieve the mean of vectors
  virtual bool mean(void *out, size_t len) const {
    size_t dim = accums_.size();
    if (dim * sizeof(T) != len) {
      return false;
    }
    for (size_t i = 0; i < dim; ++i) {
      *(static_cast<T *>(out) + i) = FloatCast<T>(accums_[i] / count_);
    }
    return true;
  }

  //! Retrieve the mean of vectors
  virtual void mean(std::string *out) const {
    ailego::NumericalVector<T> &vec =
        *static_cast<ailego::NumericalVector<T> *>(out);

    size_t dim = accums_.size();
    vec.resize(dim);
    for (size_t i = 0; i < dim; ++i) {
      vec[i] = FloatCast<T>(accums_[i] / count_);
    }
  }

  //! Merge another vector mean
  virtual bool merge(const VectorMean &rhs) {
    const NumericalVectorMean<T> &src =
        dynamic_cast<const NumericalVectorMean<T> &>(rhs);

    size_t dim = accums_.size();
    if (dim != src.accums_.size()) {
      return false;
    }
    count_ += src.count_;
    for (size_t i = 0; i < dim; ++i) {
      accums_[i] += src.accums_[i];
    }
    return true;
  }

  //! Retrieve the count of vectors
  virtual size_t count(void) const {
    return count_;
  }

  //! Retrieve dimension of accumulator
  virtual size_t dimension(void) const {
    return accums_.size();
  }

 protected:
  //! Convert float type to another type
  template <typename U>
  static auto FloatCast(const double &val) ->
      typename std::enable_if<!std::is_integral<U>::value, U>::type {
    return static_cast<U>(val);
  }

  //! Convert float type to another type
  template <typename U>
  static auto FloatCast(const double &val) ->
      typename std::enable_if<std::is_integral<U>::value, U>::type {
    return static_cast<U>(std::round(val));
  }

 private:
  //! Members
  size_t count_;
  std::vector<double> accums_;
};

/*! Numerical Vector Harmonic Mean
 */
template <typename T,
          typename =
              typename std::enable_if<ailego::IsArithmetic<T>::value>::type>
class NumericalVectorHarmonicMean : public VectorMean {
 public:
  //! Constructor
  NumericalVectorHarmonicMean(void) : count_(0), accums_() {}

  //! Constructor
  NumericalVectorHarmonicMean(const NumericalVectorHarmonicMean &rhs)
      : count_(rhs.count_), accums_(rhs.accums_) {}

  //! Constructor
  NumericalVectorHarmonicMean(NumericalVectorHarmonicMean &&rhs)
      : count_(rhs.count_), accums_(std::move(rhs.accums_)) {}

  //! Constructor
  NumericalVectorHarmonicMean(size_t dim) : count_(0), accums_(dim) {}

  //! Constructor
  NumericalVectorHarmonicMean(const T *means, size_t dim, size_t cnt)
      : count_(cnt), accums_(dim) {
    for (size_t i = 0; i < dim; ++i) {
      accums_[i] = static_cast<double>(count_) / static_cast<double>(means[i]);
    }
  }

  //! Reset accumulator
  void reset(size_t dim) {
    count_ = 0u;
    accums_.clear();
    accums_.resize(dim, 0.0);
  }

  //! Reset accumulator
  virtual void reset(void) {
    this->reset(accums_.size());
  }

  //! Plus a vector (harmonic)
  virtual bool plus(const void *vec, size_t len) {
    size_t dim = accums_.size();
    if (dim * sizeof(T) != len) {
      return false;
    }
    for (size_t i = 0; i < dim; ++i) {
      accums_[i] += 1.0 / *(static_cast<const T *>(vec) + i);
    }
    ++count_;
    return true;
  }

  //! Retrieve the mean of vectors (harmonic)
  virtual bool mean(void *out, size_t len) const {
    size_t dim = accums_.size();
    if (dim * sizeof(T) != len) {
      return false;
    }
    for (size_t i = 0; i < dim; ++i) {
      *(static_cast<T *>(out) + i) = FloatCast<T>(count_ / accums_[i]);
    }
    return true;
  }

  //! Retrieve the mean of vectors
  virtual void mean(std::string *out) const {
    ailego::NumericalVector<T> &vec =
        *static_cast<ailego::NumericalVector<T> *>(out);

    size_t dim = accums_.size();
    vec.resize(dim);
    for (size_t i = 0; i < dim; ++i) {
      vec[i] = FloatCast<T>(count_ / accums_[i]);
    }
  }

  //! Merge another vector mean
  virtual bool merge(const VectorMean &rhs) {
    const NumericalVectorHarmonicMean<T> &src =
        dynamic_cast<const NumericalVectorHarmonicMean<T> &>(rhs);

    size_t dim = accums_.size();
    if (dim != src.accums_.size()) {
      return false;
    }
    count_ += src.count_;
    for (size_t i = 0; i < dim; ++i) {
      accums_[i] += src.accums_[i];
    }
    return true;
  }

  //! Retrieve the count of vectors
  virtual size_t count(void) const {
    return count_;
  }

  //! Retrieve dimension of accumulator
  virtual size_t dimension(void) const {
    return accums_.size();
  }

 protected:
  //! Convert float type to another type
  template <typename U>
  static auto FloatCast(const double &val) ->
      typename std::enable_if<!std::is_integral<U>::value, U>::type {
    return static_cast<U>(val);
  }

  //! Convert float type to another type
  template <typename U>
  static auto FloatCast(const double &val) ->
      typename std::enable_if<std::is_integral<U>::value, U>::type {
    return static_cast<U>(std::round(val));
  }

 private:
  //! Members
  size_t count_;
  std::vector<double> accums_;
};

/*! Numerical Vector Geometric Mean
 */
template <typename T,
          typename =
              typename std::enable_if<ailego::IsArithmetic<T>::value>::type>
class NumericalVectorGeometricMean : public VectorMean {
 public:
  //! Constructor
  NumericalVectorGeometricMean(void) : count_(0), accums_() {}

  //! Constructor
  NumericalVectorGeometricMean(const NumericalVectorGeometricMean &rhs)
      : count_(rhs.count_), accums_(rhs.accums_) {}

  //! Constructor
  NumericalVectorGeometricMean(NumericalVectorGeometricMean &&rhs)
      : count_(rhs.count_), accums_(std::move(rhs.accums_)) {}

  //! Constructor
  NumericalVectorGeometricMean(size_t dim) : count_(0), accums_(dim, 1.0) {}

  //! Constructor
  NumericalVectorGeometricMean(const T *means, size_t dim, size_t cnt)
      : count_(cnt), accums_(dim, 1.0) {
    for (size_t i = 0; i < dim; ++i) {
      accums_[i] = std::pow(static_cast<double>(means[i]), count_);
    }
  }

  //! Reset accumulator
  void reset(size_t dim) {
    count_ = 0u;
    accums_.clear();
    accums_.resize(dim, 1.0);
  }

  //! Reset accumulator
  virtual void reset(void) {
    this->reset(accums_.size());
  }

  //! Plus a vector (geometric)
  virtual bool plus(const void *vec, size_t len) {
    size_t dim = accums_.size();
    if (dim * sizeof(T) != len) {
      return false;
    }
    for (size_t i = 0; i < dim; ++i) {
      accums_[i] *= *(static_cast<const T *>(vec) + i);
    }
    ++count_;
    return true;
  }

  //! Retrieve the mean of vectors (geometric)
  virtual bool mean(void *out, size_t len) const {
    size_t dim = accums_.size();
    if (dim * sizeof(T) != len) {
      return false;
    }
    for (size_t i = 0; i < dim; ++i) {
      *(static_cast<T *>(out) + i) =
          FloatCast<T>(std::pow(accums_[i], 1.0 / count_));
    }
    return true;
  }

  //! Retrieve the mean of vectors
  virtual void mean(std::string *out) const {
    ailego::NumericalVector<T> &vec =
        *static_cast<ailego::NumericalVector<T> *>(out);

    size_t dim = accums_.size();
    vec.resize(dim);
    for (size_t i = 0; i < dim; ++i) {
      vec[i] = FloatCast<T>(std::pow(accums_[i], 1.0 / count_));
    }
  }

  //! Merge another vector mean
  virtual bool merge(const VectorMean &rhs) {
    const NumericalVectorGeometricMean<T> &src =
        dynamic_cast<const NumericalVectorGeometricMean<T> &>(rhs);

    size_t dim = accums_.size();
    if (dim != src.accums_.size()) {
      return false;
    }
    count_ += src.count_;
    for (size_t i = 0; i < dim; ++i) {
      accums_[i] *= src.accums_[i];
    }
    return true;
  }

  //! Retrieve the count of vectors
  virtual size_t count(void) const {
    return count_;
  }

  //! Retrieve dimension of accumulator
  virtual size_t dimension(void) const {
    return accums_.size();
  }

 protected:
  //! Convert float type to another type
  template <typename U>
  static auto FloatCast(const double &val) ->
      typename std::enable_if<!std::is_integral<U>::value, U>::type {
    return static_cast<U>(val);
  }

  //! Convert float type to another type
  template <typename U>
  static auto FloatCast(const double &val) ->
      typename std::enable_if<std::is_integral<U>::value, U>::type {
    return static_cast<U>(std::round(val));
  }

 private:
  //! Members
  size_t count_;
  std::vector<double> accums_;
};

/*! Binary Vector Mean
 */
class BinaryVectorMean : public VectorMean {
 public:
  //! Constructor
  BinaryVectorMean(void) : count_(0), accums_() {}

  //! Constructor
  BinaryVectorMean(const BinaryVectorMean &rhs)
      : count_(rhs.count_), accums_(rhs.accums_) {}

  //! Constructor
  BinaryVectorMean(BinaryVectorMean &&rhs)
      : count_(rhs.count_), accums_(std::move(rhs.accums_)) {}

  //! Constructor
  BinaryVectorMean(size_t dim) : count_(0), accums_(((dim + 7) >> 3) << 3) {}

  //! Constructor
  BinaryVectorMean(const void *means, size_t dim, size_t cnt)
      : count_(cnt), accums_(((dim + 7) >> 3) << 3) {
    const uint8_t *bits = reinterpret_cast<const uint8_t *>(means);
    for (size_t i = 0; i < dim; ++i) {
      accums_[i] = (count_ >> 1);

      if (bits[i >> 3] & static_cast<uint8_t>(1 << (i & 0x7))) {
        accums_[i] += 1;
      }
    }
  }

  //! Reset accumulator
  void reset(size_t dim) {
    count_ = 0u;
    accums_.clear();
    accums_.resize(dim);
  }

  //! Reset accumulator
  virtual void reset(void) {
    this->reset(accums_.size());
  }

  //! Plus a vector
  virtual bool plus(const void *vec, size_t len) {
    size_t dim = accums_.size();
    if (dim != (len << 3)) {
      return false;
    }

    const uint8_t *bits = reinterpret_cast<const uint8_t *>(vec);
    for (size_t i = 0; i < dim; ++i) {
      if (bits[i >> 3] & static_cast<uint8_t>(1 << (i & 0x7))) {
        accums_[i] += 1;
      }
    }
    ++count_;
    return true;
  }

  //! Retrieve the mean of vectors
  virtual bool mean(void *out, size_t len) const {
    size_t dim = accums_.size();
    if (dim != (len << 3)) {
      return false;
    }
    memset(out, 0, len);

    uint8_t *bits = reinterpret_cast<uint8_t *>(out);
    size_t half_count = count_ >> 1;
    for (size_t i = 0; i < dim; ++i) {
      if (accums_[i] > half_count) {
        bits[i >> 3] |= static_cast<uint8_t>(1 << (i & 0x7));
      }
    }
    return true;
  }

  //! Retrieve the mean of vectors
  virtual void mean(std::string *out) const {
    size_t dim = accums_.size();
    out->clear();
    out->resize((dim + 7) / 8);

    uint8_t *bits =
        reinterpret_cast<uint8_t *>(const_cast<char *>(out->data()));
    size_t half_count = count_ >> 1;
    for (size_t i = 0; i < dim; ++i) {
      if (accums_[i] > half_count) {
        bits[i >> 3] |= static_cast<uint8_t>(1 << (i & 0x7));
      }
    }
  }

  //! Merge another vector mean
  virtual bool merge(const VectorMean &rhs) {
    const BinaryVectorMean &src = dynamic_cast<const BinaryVectorMean &>(rhs);

    size_t dim = accums_.size();
    if (dim != src.accums_.size()) {
      return false;
    }
    count_ += src.count_;
    for (size_t i = 0; i < dim; ++i) {
      accums_[i] += src.accums_[i];
    }
    return true;
  }

  //! Retrieve the count of vectors
  virtual size_t count(void) const {
    return count_;
  }

  //! Retrieve dimension of accumulator
  virtual size_t dimension(void) const {
    return accums_.size();
  }

 private:
  //! Members
  size_t count_;
  std::vector<size_t> accums_;
};

/*! Numerical Vector Mean
 */
template <typename T,
          typename = typename std::enable_if<std::is_integral<T>::value>::type>
class NibbleVectorMean : public VectorMean {
 public:
  //! Constructor
  NibbleVectorMean(void) : count_(0), accums_() {}

  //! Constructor
  NibbleVectorMean(const NibbleVectorMean &rhs)
      : count_(rhs.count_), accums_(rhs.accums_) {}

  //! Constructor
  NibbleVectorMean(NibbleVectorMean &&rhs)
      : count_(rhs.count_), accums_(std::move(rhs.accums_)) {}

  //! Constructor
  NibbleVectorMean(size_t dim) : count_(0), accums_(dim) {}

  //! Constructor
  NibbleVectorMean(const void *means, size_t dim, size_t cnt)
      : count_(cnt), accums_(dim) {
    const uint8_t *arr = reinterpret_cast<const uint8_t *>(means);
    for (size_t i = 0; i != dim; i += 2) {
      uint8_t val = arr[i >> 1];
      int lo = ((int8_t)(val << 4) >> 4);
      int hi = ((int8_t)(val) >> 4);
      accums_[i] = static_cast<double>(lo) * count_;
      accums_[i + 1] = static_cast<double>(hi) * count_;
    }
  }

  //! Reset accumulator
  void reset(size_t dim) {
    count_ = 0u;
    accums_.clear();
    accums_.resize(dim, 0.0);
  }

  //! Reset accumulator
  virtual void reset(void) {
    this->reset(accums_.size());
  }

  //! Plus a vector
  virtual bool plus(const void *vec, size_t len) {
    size_t dim = accums_.size();
    if (dim != (len << 1)) {
      return false;
    }

    const uint8_t *arr = reinterpret_cast<const uint8_t *>(vec);
    for (size_t i = 0; i != dim; i += 2) {
      uint8_t val = arr[i >> 1];
      accums_[i] += ((int8_t)(val << 4) >> 4);
      accums_[i + 1] += ((int8_t)(val) >> 4);
    }
    ++count_;
    return true;
  }

  //! Retrieve the mean of vectors
  virtual bool mean(void *out, size_t len) const {
    size_t dim = accums_.size();
    if (dim != (len << 1)) {
      return false;
    }
    memset(out, 0, len);

    uint8_t *arr = reinterpret_cast<uint8_t *>(out);

    for (size_t i = 0; i != dim; i += 2) {
      int lo = static_cast<int>(std::round(accums_[i] / count_));
      int hi = static_cast<int>(std::round(accums_[i + 1] / count_));
      arr[i >> 1] = (uint8_t)((hi << 4) & 0xf0) | (uint8_t)(lo & 0xf);
    }

    return true;
  }

  //! Retrieve the mean of vectors
  virtual void mean(std::string *out) const {
    size_t dim = accums_.size();
    out->clear();
    out->resize(dim >> 1);

    uint8_t *arr = reinterpret_cast<uint8_t *>(const_cast<char *>(out->data()));

    for (size_t i = 0; i != dim; i += 2) {
      int lo = static_cast<int>(std::round(accums_[i] / count_));
      int hi = static_cast<int>(std::round(accums_[i + 1] / count_));
      arr[i >> 1] = (uint8_t)((hi << 4) & 0xf0) | (uint8_t)(lo & 0xf);
    }
  }

  //! Merge another vector mean
  virtual bool merge(const VectorMean &rhs) {
    const NibbleVectorMean &src = dynamic_cast<const NibbleVectorMean &>(rhs);

    size_t dim = accums_.size();
    if (dim != src.accums_.size()) {
      return false;
    }
    count_ += src.count_;
    for (size_t i = 0; i < dim; ++i) {
      accums_[i] += src.accums_[i];
    }
    return true;
  }

  //! Retrieve the count of vectors
  virtual size_t count(void) const {
    return count_;
  }

  //! Retrieve dimension of accumulator
  virtual size_t dimension(void) const {
    return accums_.size();
  }

 private:
  //! Members
  size_t count_;
  std::vector<double> accums_;
};

}  // namespace core
}  // namespace zvec
