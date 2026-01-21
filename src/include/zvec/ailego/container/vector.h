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

#include <initializer_list>
#include <stdexcept>
#include <string>
#include <zvec/ailego/utility/type_helper.h>

namespace zvec {
namespace ailego {

/*! Fixed Vector
 */
template <typename T, size_t N>
class FixedVector {
 public:
  enum { MAX_SIZE = N };

  //! Constructor
  template <typename... U>
  FixedVector(U... vals) : data_{vals...} {}

  //! Overloaded operator []
  T &operator[](size_t i) {
    return data_[i];
  }

  //! Overloaded operator []
  constexpr const T &operator[](size_t i) const {
    return data_[i];
  }

  //! Retrieve data pointer
  T *data(void) {
    return data_;
  }

  //! Retrieve data pointer
  const T *data(void) const {
    return data_;
  }

  //! Retrieve count of elements in vector
  constexpr size_t size(void) const {
    return MAX_SIZE;
  }

  //! Convert a array pointer to vector pointer
  static FixedVector *Cast(T arr[N]) {
    return reinterpret_cast<FixedVector<T, N> *>(arr);
  }

  //! Convert a array pointer to vector pointer
  static const FixedVector *Cast(const T arr[N]) {
    return reinterpret_cast<const FixedVector<T, N> *>(arr);
  }

 private:
  //! Data member
  T data_[N];
};

/*! Numerical Vector Adapter
 */
template <typename T, typename TBase = std::string,
          typename =
              typename std::enable_if<IsTriviallyCopyable<T>::value>::type>
class NumericalVector : public TBase {
 public:
  typedef typename std::remove_cv<T>::type ValueType;
  typedef ValueType *iterator;
  typedef const ValueType *const_iterator;

  //! Constructor
  NumericalVector(void) : TBase() {}

  //! Constructor
  explicit NumericalVector(size_t dim) : TBase() {
    this->resize(dim);
  }

  //! Constructor
  NumericalVector(size_t dim, const ValueType &val) : TBase() {
    this->resize(dim, val);
  }

  //! Constructor
  NumericalVector(const NumericalVector &rhs) : TBase(rhs) {}

  //! Constructor
  NumericalVector(NumericalVector &&rhs) : TBase(std::forward<TBase>(rhs)) {}

  //! Constructor
  NumericalVector(const TBase &rhs) : TBase(rhs) {
    if (TBase::size() % sizeof(T) != 0) {
      throw std::length_error("Unmatched length");
    }
  }

  //! Constructor
  NumericalVector(TBase &&rhs) : TBase(std::move(rhs)) {
    if (TBase::size() % sizeof(T) != 0) {
      throw std::length_error("Unmatched length");
    }
  }

  //! Constructor
  NumericalVector(std::initializer_list<ValueType> il) : TBase() {
    for (const auto &it : il) {
      TBase::append(reinterpret_cast<const char *>(&it), sizeof(ValueType));
    }
  }

  //! Assignment
  NumericalVector &operator=(const NumericalVector &rhs) {
    TBase::operator=(static_cast<const TBase &>(rhs));
    return *this;
  }

  //! Assignment
  NumericalVector &operator=(NumericalVector &&rhs) {
    TBase::operator=(std::move(static_cast<TBase &&>(rhs)));
    return *this;
  }

  //! Assignment
  NumericalVector &operator=(const TBase &rhs) {
    TBase::operator=(rhs);
    return *this;
  }

  //! Assignment
  NumericalVector &operator=(TBase &&rhs) {
    TBase::operator=(std::move(rhs));
    return *this;
  }

  //! Overloaded operator []
  ValueType &operator[](size_t i) {
    return *(this->data() + i);
  }

  //! Overloaded operator []
  const ValueType &operator[](size_t i) const {
    return *(this->data() + i);
  }

  //! Appends a copy of value
  NumericalVector &append(const ValueType &val) {
    TBase::append(reinterpret_cast<const char *>(&val), sizeof(ValueType));
    return *this;
  }

  //! Append a copy of value
  void append(std::initializer_list<ValueType> il) {
    for (const auto &it : il) {
      TBase::append(reinterpret_cast<const char *>(&it), sizeof(ValueType));
    }
  }

  //! Assign content to vector
  void assign(const ValueType *vec, size_t len) {
    TBase::assign(reinterpret_cast<const char *>(vec), len * sizeof(ValueType));
  }

  //! Assign content to vector
  void assign(size_t n, const ValueType &val) {
    this->clear();
    this->resize(n, val);
  }

  //! Assign content to vector
  void assign(std::initializer_list<ValueType> il) {
    this->clear();
    for (const auto &it : il) {
      TBase::append(reinterpret_cast<const char *>(&it), sizeof(ValueType));
    }
  }

  //! Retrieve element
  ValueType &at(size_t i) {
    return *(this->data() + i);
  }

  //! Retrieve element
  const ValueType &at(size_t i) const {
    return *(this->data() + i);
  }

  //! Access last element
  ValueType &back(void) {
    return *(this->rbegin());
  }

  //! Access last element
  const ValueType &back(void) const {
    return *(this->rbegin());
  }

  //! Retrieve iterator to beginning
  iterator begin(void) {
    return this->data();
  }

  //! Retrieve iterator to beginning
  const_iterator begin(void) const {
    return this->data();
  }

  //! Retrieve size of allocated storage
  size_t capacity(void) const {
    return (TBase::capacity() / sizeof(ValueType));
  }

  //! Clear the vector
  void clear(void) {
    TBase::clear();
  }

  //! Retrieve pointer of data
  ValueType *data(void) {
    return reinterpret_cast<ValueType *>(&(TBase::operator[](0)));
  }

  //! Retrieve pointer of data
  const ValueType *data(void) const {
    return reinterpret_cast<const ValueType *>(TBase::data());
  }

  //! Test if vector is empty
  bool empty(void) const {
    return TBase::empty();
  }

  //! An iterator to the past-the-end
  iterator end(void) {
    return (this->data() + this->size());
  }

  //! An iterator to the past-the-end
  const_iterator end(void) const {
    return (this->data() + this->size());
  }

  //! Access first element
  ValueType &front(void) {
    return *(this->begin());
  }

  //! Access first element
  const ValueType &front(void) const {
    return *(this->begin());
  }

  //! Request a change in capacity
  void reserve(size_t n) {
    TBase::reserve(n * sizeof(ValueType));
  }

  //! Resize the vector to a length of n elements
  void resize(size_t n) {
    TBase::resize(n * sizeof(ValueType));
  }

  //! Resize the vector to a length of n elements
  void resize(size_t n, const ValueType &val) {
    size_t count = this->size();

    TBase::resize(n * sizeof(ValueType));
    for (size_t i = count; i < n; ++i) {
      *(this->data() + i) = val;
    }
  }

  //! Retrieve dimension of vector
  size_t size(void) const {
    return (TBase::size() / sizeof(ValueType));
  }

  //! Retrieve dimension of vector
  size_t dimension(void) const {
    return (TBase::size() / sizeof(ValueType));
  }

  //! Retrieve size of vector in bytes
  size_t bytes(void) const {
    return TBase::size();
  }

  //! Swap vector values
  void swap(NumericalVector &vec) {
    TBase::swap(static_cast<TBase &>(vec));
  }
};

/*! Nibble Vector Adapter
 */
template <typename T, typename TBase = std::string,
          typename = typename std::enable_if<std::is_integral<T>::value>::type>
class NibbleVector : public TBase {
 public:
  //! Type of value
  using ValueType = typename std::remove_cv<T>::type;
  using StoreType = typename std::make_unsigned<ValueType>::type;

  //! const_iterator of Nibble Vector
  class const_iterator {
   public:
    //! Constructor
    const_iterator(void) : i_(0), owner_(nullptr) {}

    //! Constructor
    const_iterator(const NibbleVector *owner, size_t i)
        : i_(i), owner_(owner) {}

    //! Equality
    bool operator==(const const_iterator &rhs) const {
      return (i_ == rhs.i_);
    }

    //! No equality
    bool operator!=(const const_iterator &rhs) const {
      return (i_ != rhs.i_);
    }

    //! Increment (Prefix)
    const_iterator &operator++() {
      ++i_;
      return *this;
    }

    //! Increment (Suffix)
    const_iterator operator++(int) {
      const_iterator tmp = *this;
      ++i_;
      return tmp;
    }

    //! Decrement (Prefix)
    const_iterator &operator--() {
      --i_;
      return *this;
    }

    //! Decrement (Suffix)
    const_iterator operator--(int) {
      const_iterator tmp = *this;
      --i_;
      return tmp;
    }

    //! operator "+="
    const_iterator &operator+=(size_t offset) {
      i_ += offset;
      return *this;
    }

    //! operator "-="
    const_iterator &operator-=(size_t offset) {
      i_ -= offset;
      return *this;
    }

    //! Indirection (Signed integral)
    ValueType operator*() const {
      return owner_->element<ValueType>(i_);
    }

   private:
    size_t i_;
    const NibbleVector *owner_;
  };

  //! Constructor
  NibbleVector(void) : TBase() {}

  //! Constructor
  explicit NibbleVector(size_t dim) : TBase() {
    this->resize(dim);
  }

  //! Constructor
  NibbleVector(size_t dim, ValueType val) : TBase() {
    this->resize(dim, val);
  }

  //! Constructor
  NibbleVector(const NibbleVector &rhs) : TBase(rhs) {}

  //! Constructor
  NibbleVector(NibbleVector &&rhs) : TBase(std::forward<TBase>(rhs)) {}

  //! Constructor
  NibbleVector(const TBase &rhs) : TBase(rhs) {}

  //! Constructor
  NibbleVector(TBase &&rhs) : TBase(std::move(rhs)) {}

  //! Constructor
  NibbleVector(std::initializer_list<ValueType> il) : TBase() {
    this->resize(il.size());

    size_t index = 0;
    uint8_t *arr = reinterpret_cast<uint8_t *>(&(TBase::operator[](0)));

    for (auto val : il) {
      arr[index >> 1] |= ((uint8_t)(val & 0xf) << ((index & 1) << 2));
      ++index;
    }
  }

  //! Assignment
  NibbleVector &operator=(const NibbleVector &rhs) {
    TBase::operator=(static_cast<const TBase &>(rhs));
    return *this;
  }

  //! Assignment
  NibbleVector &operator=(NibbleVector &&rhs) {
    TBase::operator=(std::move(static_cast<TBase &&>(rhs)));
    return *this;
  }

  //! Assignment
  NibbleVector &operator=(const TBase &rhs) {
    TBase::operator=(rhs);
    return *this;
  }

  //! Assignment
  NibbleVector &operator=(TBase &&rhs) {
    TBase::operator=(std::move(rhs));
    return *this;
  }

  //! Overloaded operator [] (Signed integral)
  ValueType operator[](size_t i) const {
    return this->at(i);
  }

  //! Appends a copy of value
  NibbleVector &append(ValueType lo, ValueType hi) {
    TBase::push_back(((uint8_t)(hi & 0xf) << 4) | (uint8_t)(lo & 0xf));
    return *this;
  }

  //! Append a copy of value
  void append(std::initializer_list<ValueType> il) {
    size_t index = this->size();
    this->resize(index + il.size());

    uint8_t *arr = reinterpret_cast<uint8_t *>(&(TBase::operator[](0)));
    for (auto val : il) {
      arr[index >> 1] |= ((uint8_t)(val & 0xf) << ((index & 1) << 2));
      ++index;
    }
  }

  //! Assign content to vector
  void assign(const ValueType *vec, size_t len) {
    this->clear();
    this->resize(len);

    uint8_t *arr = reinterpret_cast<uint8_t *>(&(TBase::operator[](0)));
    for (size_t i = 0; i != len; ++i) {
      arr[i >> 1] |= ((uint8_t)(vec[i] & 0xf) << ((i & 1) << 2));
    }
  }

  //! Assign content to vector
  void assign(size_t n, ValueType val) {
    this->clear();
    this->resize(n, val);
  }

  //! Assign content to vector
  void assign(std::initializer_list<ValueType> il) {
    this->clear();
    this->resize(il.size());

    uint8_t *arr = reinterpret_cast<uint8_t *>(&(TBase::operator[](0)));
    size_t index = 0;

    for (auto val : il) {
      arr[index >> 1] |= ((uint8_t)(val & 0xf) << ((index & 1) << 2));
      ++index;
    }
  }

  //! Set a element
  void set(size_t i, ValueType val) {
    uint8_t *it = reinterpret_cast<uint8_t *>(&(TBase::operator[](i >> 1)));
    if (i & 1) {
      *it = (*it & 0x0f) | ((uint8_t)(val & 0xf) << 4);
    } else {
      *it = (*it & 0xf0) | (uint8_t)(val & 0xf);
    }
  }

  //! Retrieve element
  ValueType at(size_t i) const {
    return this->element<ValueType>(i);
  }

  //! Access last element
  ValueType back(void) const {
    return this->at(this->size() - 1);
  }

  //! Retrieve iterator to beginning
  const_iterator begin(void) const {
    return const_iterator(this, 0);
  }

  //! Retrieve size of allocated storage
  size_t capacity(void) const {
    return (TBase::capacity() << 1);
  }

  //! Clear the vector
  void clear(void) {
    TBase::clear();
  }

  //! Retrieve pointer of data
  StoreType *data(void) {
    return reinterpret_cast<StoreType *>(&(TBase::operator[](0)));
  }

  //! Retrieve pointer of data
  const StoreType *data(void) const {
    return reinterpret_cast<const StoreType *>(TBase::data());
  }

  //! Test if vector is empty
  bool empty(void) const {
    return TBase::empty();
  }

  //! An iterator to the past-the-end
  const_iterator end(void) const {
    return const_iterator(this, this->size());
  }

  //! Access first element
  ValueType front(void) const {
    return this->at(0);
  }

  //! Request a change in capacity
  void reserve(size_t n) {
    TBase::reserve((n + (sizeof(ValueType) << 1) - 1) /
                   (sizeof(ValueType) << 1) * sizeof(ValueType));
  }

  //! Resize the vector to a length of n elements
  void resize(size_t n) {
    TBase::resize((n + (sizeof(ValueType) << 1) - 1) /
                  (sizeof(ValueType) << 1) * sizeof(ValueType));
  }

  //! Resize the vector to a length of n elements
  void resize(size_t n, ValueType val) {
    TBase::resize((n + (sizeof(ValueType) << 1) - 1) /
                      (sizeof(ValueType) << 1) * sizeof(ValueType),
                  ((uint8_t)(val & 0xf) << 4) | (uint8_t)(val & 0xf));
  }

  //! Retrieve dimension of vector
  size_t size(void) const {
    return (TBase::size() << 1);
  }

  //! Retrieve dimension of vector
  size_t dimension(void) const {
    return (TBase::size() << 1);
  }

  //! Retrieve size of vector in bytes
  size_t bytes(void) const {
    return TBase::size();
  }

  //! Swap vector values
  void swap(NibbleVector &vec) {
    TBase::swap(static_cast<TBase &>(vec));
  }

 protected:
  //! Retrieve element (Signed integral)
  template <typename U>
  auto element(size_t i) const ->
      typename std::enable_if<std::is_signed<U>::value, U>::type {
    const uint8_t *arr = reinterpret_cast<const uint8_t *>(TBase::data());
    return (static_cast<int8_t>(arr[i >> 1] << (~(i << 2) & 4)) >> 4);
  }

  //! Retrieve element (Unsigned integral)
  template <typename U>
  auto element(size_t i) const ->
      typename std::enable_if<std::is_unsigned<U>::value, U>::type {
    const uint8_t *arr = reinterpret_cast<const uint8_t *>(TBase::data());
    return ((arr[i >> 1] >> ((i & 1) << 2)) & 0xf);
  }
};

/*! Binary Vector Adapter
 */
template <typename T, typename TBase = std::string,
          typename = typename std::enable_if<std::is_integral<T>::value>::type>
class BinaryVector : public TBase {
 public:
  //! Type of value
  using ValueType = typename std::remove_cv<T>::type;

  //! const_iterator of Binary Vector
  class const_iterator {
   public:
    //! Constructor
    const_iterator(void) : i_(0), arr_(nullptr) {}

    //! Constructor
    const_iterator(const void *buf, size_t i)
        : i_(i), arr_(reinterpret_cast<const uint8_t *>(buf)) {}

    //! Equality
    bool operator==(const const_iterator &rhs) const {
      return (i_ == rhs.i_);
    }

    //! No equality
    bool operator!=(const const_iterator &rhs) const {
      return (i_ != rhs.i_);
    }

    //! Increment (Prefix)
    const_iterator &operator++() {
      ++i_;
      return *this;
    }

    //! Increment (Suffix)
    const_iterator operator++(int) {
      const_iterator tmp = *this;
      ++i_;
      return tmp;
    }

    //! Decrement (Prefix)
    const_iterator &operator--() {
      --i_;
      return *this;
    }

    //! Decrement (Suffix)
    const_iterator operator--(int) {
      const_iterator tmp = *this;
      --i_;
      return tmp;
    }

    //! operator "+="
    const_iterator &operator+=(size_t offset) {
      i_ += offset;
      return *this;
    }

    //! operator "-="
    const_iterator &operator-=(size_t offset) {
      i_ -= offset;
      return *this;
    }

    //! Indirection (eg. *iter)
    bool operator*() const {
      return ((arr_[i_ >> 3] & (1u << (i_ & 7))) != 0);
    }

   private:
    size_t i_;
    const uint8_t *arr_;
  };

  //! Constructor
  BinaryVector(void) : TBase() {}

  //! Constructor
  explicit BinaryVector(size_t dim) : TBase() {
    this->resize(dim);
  }

  //! Constructor
  BinaryVector(size_t dim, bool val) : TBase() {
    this->resize(dim, val);
  }

  //! Constructor
  BinaryVector(const BinaryVector &rhs) : TBase(rhs) {}

  //! Constructor
  BinaryVector(BinaryVector &&rhs) : TBase(std::move(rhs)) {}

  //! Constructor
  BinaryVector(const TBase &rhs) : TBase(rhs) {
    if (TBase::size() % sizeof(T) != 0) {
      throw std::length_error("Unmatched length");
    }
  }

  //! Constructor
  BinaryVector(TBase &&rhs) : TBase(std::move(rhs)) {
    if (TBase::size() % sizeof(T) != 0) {
      throw std::length_error("Unmatched length");
    }
  }

  //! Constructor
  BinaryVector(std::initializer_list<bool> il) : TBase() {
    this->resize(il.size());

    size_t index = 0;
    uint8_t *arr = reinterpret_cast<uint8_t *>(&(TBase::operator[](0)));

    for (auto val : il) {
      if (val) {
        arr[index >> 3] |= (uint8_t)(1u << (index & 7));
      }
      ++index;
    }
  }

  //! Assignment
  BinaryVector &operator=(const BinaryVector &rhs) {
    TBase::operator=(static_cast<const TBase &>(rhs));
    return *this;
  }

  //! Assignment
  BinaryVector &operator=(BinaryVector &&rhs) {
    TBase::operator=(std::move(static_cast<TBase &&>(rhs)));
    return *this;
  }

  //! Assignment
  BinaryVector &operator=(const TBase &rhs) {
    TBase::operator=(rhs);
    return *this;
  }

  //! Assignment
  BinaryVector &operator=(TBase &&rhs) {
    TBase::operator=(std::move(rhs));
    return *this;
  }

  //! Overloaded operator []
  bool operator[](size_t i) const {
    const uint8_t *arr = reinterpret_cast<const uint8_t *>(TBase::data());
    return ((arr[i >> 3] & (1u << (i & 7))) != 0);
  }

  //! Assign content to vector
  void assign(const bool *vec, size_t len) {
    this->clear();
    this->resize(len);

    uint8_t *arr = reinterpret_cast<uint8_t *>(&(TBase::operator[](0)));
    for (size_t i = 0; i < len; ++i) {
      bool val = vec[i];
      if (val) {
        arr[i >> 3] |= (1u << (i & 7));
      }
    }
  }

  //! Assign content to vector
  void assign(size_t n, bool val) {
    this->clear();
    this->resize(n, val);
  }

  //! Assign content to vector
  void assign(std::initializer_list<bool> il) {
    this->clear();
    this->resize(il.size());

    size_t index = 0;
    uint8_t *arr = reinterpret_cast<uint8_t *>(&(TBase::operator[](0)));
    for (auto val : il) {
      if (val) {
        arr[index >> 3] |= (uint8_t)(1u << (index & 7));
      }
      ++index;
    }
  }

  //! Retrieve element
  bool at(size_t i) const {
    const uint8_t *arr = reinterpret_cast<const uint8_t *>(TBase::data());
    return ((arr[i >> 3] & (1u << (i & 7))) != 0);
  }

  //! Set a bit
  void set(size_t i) {
    uint8_t *arr = reinterpret_cast<uint8_t *>(&(TBase::operator[](0)));
    arr[i >> 3] |= (uint8_t)(1u << (i & 7));
  }

  //! Reset a bit
  void reset(size_t i) {
    uint8_t *arr = reinterpret_cast<uint8_t *>(&(TBase::operator[](0)));
    arr[i >> 3] &= (uint8_t)(~(1u << (i & 7)));
  }

  //! Toggle a bit
  void flip(size_t i) {
    uint8_t *arr = reinterpret_cast<uint8_t *>(&(TBase::operator[](0)));
    arr[i >> 3] ^= (uint8_t)(1u << (i & 7));
  }

  //! Access last element
  bool back(void) const {
    return this->at(this->size() - 1);
  }

  //! Retrieve const_iterator to beginning
  const_iterator begin(void) const {
    return const_iterator(this->data(), 0);
  }

  //! Retrieve size of allocated storage
  size_t capacity(void) const {
    return (TBase::capacity() << 3);
  }

  //! Clear the vector
  void clear(void) {
    TBase::clear();
  }

  //! Retrieve pointer of data
  ValueType *data(void) {
    return reinterpret_cast<ValueType *>(&(TBase::operator[](0)));
  }

  //! Retrieve pointer of data
  const ValueType *data(void) const {
    return reinterpret_cast<const ValueType *>(TBase::data());
  }

  //! Test if vector is empty
  bool empty(void) const {
    return TBase::empty();
  }

  //! An const_iterator to the past-the-end
  const_iterator end(void) const {
    return const_iterator(this->data(), this->size());
  }

  //! Access first element
  bool front(void) const {
    return this->at(0);
  }

  //! Request a change in capacity
  void reserve(size_t n) {
    TBase::reserve((n + (sizeof(ValueType) << 3) - 1) /
                   (sizeof(ValueType) << 3) * sizeof(ValueType));
  }

  //! Resize the vector to a length of n elements
  void resize(size_t n) {
    TBase::resize((n + (sizeof(ValueType) << 3) - 1) /
                  (sizeof(ValueType) << 3) * sizeof(ValueType));
  }

  //! Resize the vector to a length of n elements
  void resize(size_t n, bool val) {
    TBase::resize((n + (sizeof(ValueType) << 3) - 1) /
                      (sizeof(ValueType) << 3) * sizeof(ValueType),
                  val ? 0xffu : 0u);
  }

  //! Retrieve dimension of vector
  size_t size(void) const {
    return (TBase::size() << 3);
  }

  //! Retrieve dimension of vector
  size_t dimension(void) const {
    return (TBase::size() << 3);
  }

  //! Retrieve size of vector in bytes
  size_t bytes(void) const {
    return TBase::size();
  }

  //! Swap vector values
  void swap(BinaryVector &vec) {
    TBase::swap(static_cast<TBase &>(vec));
  }
};

/*! Hybrid Vector Adapter
 */
template <typename T, typename TBase = std::string,
          typename =
              typename std::enable_if<IsTriviallyCopyable<T>::value>::type>
class HybridVector : public NumericalVector<T, TBase> {
 public:
  typedef typename std::remove_cv<T>::type ValueType;
  typedef ValueType *iterator;
  typedef const ValueType *const_iterator;

  //! Constructor
  HybridVector(void) = default;

  //! Constructor
  explicit HybridVector(size_t dim) : NumericalVector<T, TBase>(dim) {}

  //! Constructor
  HybridVector(const HybridVector &rhs)
      : NumericalVector<T, TBase>(rhs),
        sparse_count_(rhs.sparse_count_),
        sparse_indices_(rhs.sparse_indices_),
        sparse_data_(rhs.sparse_data_) {}

  //! Constructor
  HybridVector(HybridVector &&rhs)
      : NumericalVector<T, TBase>(std::forward<NumericalVector<T, TBase>>(rhs)),
        sparse_count_(rhs.sparse_count_),
        sparse_indices_(std::move(rhs.sparse_indices_)),
        sparse_data_(std::move(rhs.sparse_data_)) {}

  //! Assignment
  HybridVector &operator=(const HybridVector &rhs) {
    NumericalVector<T, TBase>::operator=(
        static_cast<const NumericalVector<T, TBase> &>(rhs));
    sparse_count_ = rhs.sparse_count_;
    sparse_indices_ = rhs.sparse_indices_;
    sparse_data_ = rhs.sparse_data_;

    return *this;
  }

  //! Assignment
  HybridVector &operator=(HybridVector &&rhs) {
    NumericalVector<T, TBase>::operator=(
        std::move(static_cast<NumericalVector<T, TBase> &&>(rhs)));
    sparse_count_ = rhs.sparse_count_;
    sparse_indices_ = std::move(rhs.sparse_indices_);
    sparse_data_ = std::move(rhs.sparse_data_);

    return *this;
  }

  size_t sparse_count() const {
    return sparse_count_;
  }

  const uint32_t *sparse_indices() const {
    return reinterpret_cast<const uint32_t *>(sparse_indices_.data());
  }

  const ValueType *sparse_data(void) const {
    return reinterpret_cast<const ValueType *>(sparse_data_.data());
  }

  //! Request a change in capacity
  void resize_for_sparse(size_t n) {
    sparse_indices_.resize(n * sizeof(uint32_t));
    sparse_data_.resize(n * sizeof(ValueType));
  }

  int add_sparses(const NumericalVector<uint32_t> &indexes,
                  const NumericalVector<ValueType> &values) {
    sparse_count_ = indexes.size();

    sparse_indices_ = (const std::string &)indexes;
    sparse_data_ = (const std::string &)values;

    return 0;
  }

 private:
  //! Data Member
  size_t sparse_count_;
  std::string sparse_indices_;
  std::string sparse_data_;
};

/*! Sparse Vector Adapter
 */
template <typename T>
class SparseVector {
 public:
  typedef typename std::remove_cv<T>::type ValueType;
  typedef ValueType *iterator;
  typedef const ValueType *const_iterator;

  //! Constructor
  SparseVector(void) = default;

  size_t sparse_count() const {
    return sparse_count_;
  }

  const uint32_t *sparse_indices() const {
    return reinterpret_cast<const uint32_t *>(sparse_indices_.data());
  }

  const ValueType *sparse_data(void) const {
    return reinterpret_cast<const ValueType *>(sparse_data_.data());
  }

  //! Request a change in capacity
  void resize_for_sparse(size_t n) {
    sparse_indices_.resize(n * sizeof(uint32_t));
    sparse_data_.resize(n * sizeof(ValueType));
  }

  int add_sparses(const NumericalVector<uint32_t> &indexes,
                  const NumericalVector<ValueType> &values) {
    sparse_count_ = indexes.size();

    sparse_indices_ = (const std::string &)indexes;
    sparse_data_ = (const std::string &)values;

    return 0;
  }

 private:
  //! Data Member
  size_t sparse_count_;
  std::string sparse_indices_;
  std::string sparse_data_;
};

}  // namespace ailego
}  // namespace zvec
