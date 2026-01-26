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

#include "ailego/internal/platform.h"
#include "vector.h"

namespace zvec {
namespace ailego {

/*! Numerical Vector Array
 */
template <typename T,
          typename =
              typename std::enable_if<IsTriviallyCopyable<T>::value>::type>
class NumericalVectorArray {
 public:
  //! Type of value
  using ValueType = typename NumericalVector<T>::ValueType;

  //! Constructor
  NumericalVectorArray(void) {}

  //! Constructor
  explicit NumericalVectorArray(size_t dim) : dimension_(dim) {}

  //! Constructor
  NumericalVectorArray(const NumericalVectorArray &rhs)
      : dimension_(rhs.dimension_), buffer_(rhs.buffer_) {}

  //! Constructor
  NumericalVectorArray(NumericalVectorArray &&rhs)
      : dimension_(rhs.dimension_), buffer_(std::move(rhs.buffer_)) {}

  //! Assignment
  NumericalVectorArray &operator=(const NumericalVectorArray &rhs) {
    dimension_ = rhs.dimension_;
    buffer_ = rhs.buffer_;
    return *this;
  }

  //! Assignment
  NumericalVectorArray &operator=(NumericalVectorArray &&rhs) {
    dimension_ = rhs.dimension_;
    buffer_ = std::move(rhs.buffer_);
    return *this;
  }

  //! Overloaded operator []
  ValueType *operator[](size_t i) {
    return (reinterpret_cast<ValueType *>(&buffer_[0]) + i * dimension_);
  }

  //! Overloaded operator []
  const ValueType *operator[](size_t i) const {
    return (reinterpret_cast<const ValueType *>(buffer_.data()) +
            i * dimension_);
  }

  //! Append a vector
  void append(const ValueType *vec, size_t dim) {
    if (ailego_unlikely(dim != dimension_)) {
      throw std::length_error("Unmatched dimension");
    }
    buffer_.append(reinterpret_cast<const char *>(vec),
                   dim * sizeof(ValueType));
  }

  //! Append vectors
  void append(const ValueType *vec, size_t dim, size_t cnt) {
    if (ailego_unlikely(dim != dimension_)) {
      throw std::length_error("Unmatched dimension");
    }
    buffer_.append(reinterpret_cast<const char *>(vec),
                   cnt * dim * sizeof(ValueType));
  }

  //! Append a vector
  void append(const NumericalVector<ValueType> &vec) {
    this->append(vec.data(), vec.dimension());
  }

  //! Replace a vector
  void replace(size_t index, const ValueType *vec, size_t dim) {
    if (ailego_unlikely(dim != dimension_)) {
      throw std::length_error("Unmatched dimension");
    }
    size_t element_size = dim * sizeof(ValueType);
    buffer_.replace(index * element_size, element_size,
                    reinterpret_cast<const char *>(vec), element_size);
  }

  //! Replace a vector
  void replace(size_t index, const ValueType *vec, size_t dim, size_t cnt) {
    if (ailego_unlikely(dim != dimension_)) {
      throw std::length_error("Unmatched dimension");
    }
    size_t element_size = dim * sizeof(ValueType);
    size_t total = element_size * cnt;
    buffer_.replace(index * element_size, total,
                    reinterpret_cast<const char *>(vec), total);
  }

  //! Replace a vector
  void replace(size_t index, const NumericalVector<ValueType> &vec) {
    this->replace(index, vec.data(), vec.dimension());
  }

  //! Request a change in capacity
  void reserve(size_t n) {
    buffer_.reserve(n * dimension_ * sizeof(ValueType));
  }

  //! Resize the array to a length of n elements
  void resize(size_t n) {
    buffer_.resize(n * dimension_ * sizeof(ValueType));
  }

  //! Clear the vector array
  void clear(void) {
    buffer_.clear();
  }

  //! Reset the vector array
  void reset(size_t dim) {
    dimension_ = dim;
    buffer_.clear();
  }

  //! Requests the removal of unused capacity.
  void shrink_to_fit(void) {
    buffer_.shrink_to_fit();
  }

  //! Retrieve pointer of data
  ValueType *data(void) {
    return reinterpret_cast<ValueType *>(&buffer_[0]);
  }

  //! Retrieve pointer of data
  const ValueType *data(void) const {
    return reinterpret_cast<const ValueType *>(buffer_.data());
  }

  //! Retrieve pointer of data
  ValueType *at(size_t i) {
    if (ailego_unlikely(i >= this->count())) {
      throw std::out_of_range("Index overflow");
    }
    return (reinterpret_cast<ValueType *>(&buffer_[0]) + i * dimension_);
  }

  //! Retrieve pointer of data
  const ValueType *at(size_t i) const {
    if (ailego_unlikely(i >= this->count())) {
      throw std::out_of_range("Index overflow");
    }
    return (reinterpret_cast<const ValueType *>(buffer_.data()) +
            i * dimension_);
  }

  //! Test if the array is empty
  bool empty(void) const {
    return buffer_.empty();
  }

  //! Retrieve count of vectors
  size_t count(void) const {
    return (dimension_ > 0 ? buffer_.size() / (dimension_ * sizeof(ValueType))
                           : 0u);
  }

  //! Retrieve dimension of vector
  size_t dimension(void) const {
    return dimension_;
  }

  //! Retrieve size of array in bytes
  size_t bytes(void) const {
    return buffer_.size();
  }

 private:
  size_t dimension_{0u};
  std::string buffer_{};
};

/*! Nibble Vector Array
 */
template <typename T,
          typename = typename std::enable_if<std::is_integral<T>::value>::type>
class NibbleVectorArray {
 public:
  //! Type of value
  using ValueType = typename NibbleVector<T>::ValueType;
  using StoreType = typename NibbleVector<T>::StoreType;

  //! Constructor
  NibbleVectorArray(void) {}

  //! Constructor
  explicit NibbleVectorArray(size_t dim)
      : dimension_((dim + (sizeof(ValueType) << 1) - 1) /
                       (sizeof(ValueType) << 1) * sizeof(ValueType)
                   << 1) {}

  //! Constructor
  NibbleVectorArray(const NibbleVectorArray &rhs)
      : dimension_(rhs.dimension_), buffer_(rhs.buffer_) {}

  //! Constructor
  NibbleVectorArray(NibbleVectorArray &&rhs)
      : dimension_(rhs.dimension_), buffer_(std::move(rhs.buffer_)) {}

  //! Assignment
  NibbleVectorArray &operator=(const NibbleVectorArray &rhs) {
    dimension_ = rhs.dimension_;
    buffer_ = rhs.buffer_;
    return *this;
  }

  //! Assignment
  NibbleVectorArray &operator=(NibbleVectorArray &&rhs) {
    dimension_ = rhs.dimension_;
    buffer_ = std::move(rhs.buffer_);
    return *this;
  }

  //! Overloaded operator []
  StoreType *operator[](size_t i) {
    return reinterpret_cast<StoreType *>(&buffer_[0] + i * (dimension_ >> 1));
  }

  //! Overloaded operator []
  const StoreType *operator[](size_t i) const {
    return reinterpret_cast<const StoreType *>(&buffer_[0] +
                                               i * (dimension_ >> 1));
  }

  //! Append a vector
  void append(const StoreType *vec, size_t dim) {
    if (ailego_unlikely(dim != dimension_)) {
      throw std::length_error("Unmatched dimension");
    }
    buffer_.append(reinterpret_cast<const char *>(vec), dim >> 1);
  }

  //! Append vectors
  void append(const StoreType *vec, size_t dim, size_t cnt) {
    if (ailego_unlikely(dim != dimension_)) {
      throw std::length_error("Unmatched dimension");
    }
    buffer_.append(reinterpret_cast<const char *>(vec), cnt * (dim >> 1));
  }

  //! Append a vector
  void append(const NibbleVector<ValueType> &vec) {
    this->append(vec.data(), vec.dimension());
  }

  //! Replace a vector
  void replace(size_t index, const StoreType *vec, size_t dim) {
    if (ailego_unlikely(dim != dimension_)) {
      throw std::length_error("Unmatched dimension");
    }
    size_t element_size = (dim >> 1);
    buffer_.replace(index * element_size, element_size,
                    reinterpret_cast<const char *>(vec), element_size);
  }

  //! Replace a vector
  void replace(size_t index, const StoreType *vec, size_t dim, size_t cnt) {
    if (ailego_unlikely(dim != dimension_)) {
      throw std::length_error("Unmatched dimension");
    }
    size_t element_size = (dim >> 1);
    size_t total = element_size * cnt;
    buffer_.replace(index * element_size, total,
                    reinterpret_cast<const char *>(vec), total);
  }

  //! Replace a vector
  void replace(size_t index, const NibbleVector<ValueType> &vec) {
    this->replace(index, vec.data(), vec.dimension());
  }

  //! Request a change in capacity
  void reserve(size_t n) {
    buffer_.reserve(n * (dimension_ >> 1));
  }

  //! Resize the array to a length of n elements
  void resize(size_t n) {
    buffer_.resize(n * (dimension_ >> 1));
  }

  //! Clear the vector array
  void clear(void) {
    buffer_.clear();
  }

  //! Reset the vector array
  void reset(size_t dim) {
    dimension_ = (dim + (sizeof(ValueType) << 1) - 1) /
                     (sizeof(ValueType) << 1) * sizeof(ValueType)
                 << 1;
    buffer_.clear();
  }

  //! Requests the removal of unused capacity.
  void shrink_to_fit(void) {
    buffer_.shrink_to_fit();
  }

  //! Retrieve pointer of data
  StoreType *data(void) {
    return reinterpret_cast<StoreType *>(&buffer_[0]);
  }

  //! Retrieve pointer of data
  const StoreType *data(void) const {
    return reinterpret_cast<const StoreType *>(buffer_.data());
  }

  //! Retrieve pointer of data
  StoreType *at(size_t i) {
    if (ailego_unlikely(i >= this->count())) {
      throw std::out_of_range("Index overflow");
    }
    return reinterpret_cast<StoreType *>(&buffer_[0] + i * (dimension_ >> 1));
  }

  //! Retrieve pointer of data
  const StoreType *at(size_t i) const {
    if (ailego_unlikely(i >= this->count())) {
      throw std::out_of_range("Index overflow");
    }
    return reinterpret_cast<const StoreType *>(buffer_.data() +
                                               i * (dimension_ >> 1));
  }

  //! Test if the array is empty
  bool empty(void) const {
    return buffer_.empty();
  }

  //! Retrieve count of vectors
  size_t count(void) const {
    return (dimension_ > 1 ? buffer_.size() / (dimension_ >> 1) : 0u);
  }

  //! Retrieve dimension of vector
  size_t dimension(void) const {
    return dimension_;
  }

  //! Retrieve size of array in bytes
  size_t bytes(void) const {
    return buffer_.size();
  }

 private:
  size_t dimension_{0u};
  std::string buffer_{};
};

/*! Binary Vector Array
 */
template <typename T,
          typename = typename std::enable_if<std::is_integral<T>::value>::type>
class BinaryVectorArray {
 public:
  //! Type of value
  using ValueType = typename BinaryVector<T>::ValueType;

  //! Constructor
  BinaryVectorArray(void) {}

  //! Constructor
  explicit BinaryVectorArray(size_t dim)
      : dimension_((dim + (sizeof(ValueType) << 3) - 1) /
                   (sizeof(ValueType) << 3) * (sizeof(ValueType) << 3)) {}

  //! Constructor
  BinaryVectorArray(const BinaryVectorArray &rhs)
      : dimension_(rhs.dimension_), buffer_(rhs.buffer_) {}

  //! Constructor
  BinaryVectorArray(BinaryVectorArray &&rhs)
      : dimension_(rhs.dimension_), buffer_(std::move(rhs.buffer_)) {}

  //! Assignment
  BinaryVectorArray &operator=(const BinaryVectorArray &rhs) {
    dimension_ = rhs.dimension_;
    buffer_ = rhs.buffer_;
    return *this;
  }

  //! Assignment
  BinaryVectorArray &operator=(BinaryVectorArray &&rhs) {
    dimension_ = rhs.dimension_;
    buffer_ = std::move(rhs.buffer_);
    return *this;
  }

  //! Overloaded operator []
  ValueType *operator[](size_t i) {
    return reinterpret_cast<ValueType *>(&buffer_[0] + i * (dimension_ >> 3));
  }

  //! Overloaded operator []
  const ValueType *operator[](size_t i) const {
    return reinterpret_cast<const ValueType *>(buffer_.data() +
                                               i * (dimension_ >> 3));
  }

  //! Append a vector
  void append(const ValueType *vec, size_t dim) {
    if (ailego_unlikely(dim != dimension_)) {
      throw std::length_error("Unmatched dimension");
    }
    buffer_.append(reinterpret_cast<const char *>(vec), (dim >> 3));
  }

  //! Append vectors
  void append(const ValueType *vec, size_t dim, size_t cnt) {
    if (ailego_unlikely(dim != dimension_)) {
      throw std::length_error("Unmatched dimension");
    }
    buffer_.append(reinterpret_cast<const char *>(vec), cnt * (dim >> 3));
  }

  //! Append a vector
  void append(const BinaryVector<ValueType> &vec) {
    this->append(vec.data(), vec.dimension());
  }

  //! Replace a vector
  void replace(size_t index, const ValueType *vec, size_t dim) {
    if (ailego_unlikely(dim != dimension_)) {
      throw std::length_error("Unmatched dimension");
    }
    size_t element_size = (dim >> 3);
    buffer_.replace(index * element_size, element_size,
                    reinterpret_cast<const char *>(vec), element_size);
  }

  //! Replace a vector
  void replace(size_t index, const ValueType *vec, size_t dim, size_t cnt) {
    if (ailego_unlikely(dim != dimension_)) {
      throw std::length_error("Unmatched dimension");
    }
    size_t element_size = (dim >> 3);
    size_t total = element_size * cnt;
    buffer_.replace(index * element_size, total,
                    reinterpret_cast<const char *>(vec), total);
  }

  //! Replace a vector
  void replace(size_t index, const BinaryVector<ValueType> &vec) {
    this->replace(index, vec.data(), vec.dimension());
  }

  //! Request a change in capacity
  void reserve(size_t n) {
    buffer_.reserve(n * (dimension_ >> 3));
  }

  //! Resize the array to a length of n elements
  void resize(size_t n) {
    buffer_.resize(n * (dimension_ >> 3));
  }

  //! Clear the vector array
  void clear(void) {
    buffer_.clear();
  }

  //! Reset the vector array
  void reset(size_t dim) {
    dimension_ = (dim + (sizeof(ValueType) << 3) - 1) /
                 (sizeof(ValueType) << 3) * (sizeof(ValueType) << 3);
    buffer_.clear();
  }

  //! Requests the removal of unused capacity.
  void shrink_to_fit(void) {
    buffer_.shrink_to_fit();
  }

  //! Retrieve pointer of data
  ValueType *data(void) {
    return reinterpret_cast<ValueType *>(&buffer_[0]);
  }

  //! Retrieve pointer of data
  const ValueType *data(void) const {
    return reinterpret_cast<const ValueType *>(buffer_.data());
  }

  //! Retrieve pointer of data
  ValueType *at(size_t i) {
    if (ailego_unlikely(i >= this->count())) {
      throw std::out_of_range("Index overflow");
    }
    return reinterpret_cast<ValueType *>(&buffer_[0] + i * (dimension_ >> 3));
  }

  //! Retrieve pointer of data
  const ValueType *at(size_t i) const {
    if (ailego_unlikely(i >= this->count())) {
      throw std::out_of_range("Index overflow");
    }
    return reinterpret_cast<const ValueType *>(buffer_.data() +
                                               i * (dimension_ >> 3));
  }

  //! Test if the array is empty
  bool empty(void) const {
    return buffer_.empty();
  }

  //! Retrieve count of vectors
  size_t count(void) const {
    return (dimension_ > 0 ? buffer_.size() / (dimension_ >> 3) : 0u);
  }

  //! Retrieve dimension of vector
  size_t dimension(void) const {
    return dimension_;
  }

  //! Retrieve size of array in bytes
  size_t bytes(void) const {
    return buffer_.size();
  }

 private:
  size_t dimension_{0u};
  std::string buffer_{};
};

}  // namespace ailego
}  // namespace zvec
