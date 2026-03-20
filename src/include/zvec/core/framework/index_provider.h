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

#include <unordered_map>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_holder.h>
#include <zvec/core/framework/index_storage.h>

namespace zvec {
namespace core {

/*! Index Provider
 */
struct IndexProvider : public IndexHolder {
  //! Index Provider Pointer
  typedef std::shared_ptr<IndexProvider> Pointer;

  //! Destructor
  virtual ~IndexProvider(void) {}

  bool multipass() const override {
    return true;
  }

 public:  // Provider's unique method
  //! Retrieve a vector using a primary key
  virtual const void *get_vector(const uint64_t key) const = 0;

  //! Retrieve a vector using a primary key
  virtual int get_vector(const uint64_t /*key*/,
                         IndexStorage::MemoryBlock & /*block*/) const {
    return IndexError_NotImplemented;
  }

  //! Retrieve the owner class
  virtual const std::string &owner_class(void) const = 0;
};

/*! Index SparseProvider
 */
struct IndexSparseProvider : IndexSparseHolder {
  //! Index Provider Pointer
  typedef std::shared_ptr<IndexSparseProvider> Pointer;

  //! Destructor
  virtual ~IndexSparseProvider(void) {}

  bool multipass() const override {
    return true;
  }

 public:  // Provider's unique method
  //! Retrieve a vector using a primary key
  virtual int get_sparse_vector(uint64_t key, uint32_t *sparse_count,
                                std::string *sparse_indices_buffer,
                                std::string *sparse_values_buffer) const = 0;

  //! Retrieve the owner class
  virtual const std::string &owner_class(void) const = 0;
};

/*! Multi-Pass Numerical Index Provider
 */
template <typename T>
class MultiPassNumericalIndexProvider : public IndexProvider {
 public:
  //! Constructor
  explicit MultiPassNumericalIndexProvider(size_t dim)
      : holder_(dim), owner_class_("MultiPassNumericalIndexProvider") {}

  //! Destructor
  virtual ~MultiPassNumericalIndexProvider(void) {}

  //! Retrieve count of elements in holder
  size_t count(void) const override {
    return holder_.count();
  }

  //! Retrieve dimension
  size_t dimension(void) const override {
    return holder_.dimension();
  }

  //! Retrieve element size in bytes
  size_t element_size(void) const override {
    return holder_.element_size();
  }

  //! Create a new iterator
  IndexHolder::Iterator::Pointer create_iterator(void) override {
    return holder_.create_iterator();
  }

  //! Retrieve a vector using a primary key
  const void *get_vector(const uint64_t key) const override {
    auto it = indice_map_.find(key);
    if (it == indice_map_.end()) {
      return nullptr;
    }
    return holder_.get_vector_by_index(it->second);
  }

  //! Retrieve a vector using a primary key
  int get_vector(const uint64_t key,
                 IndexStorage::MemoryBlock &block) const override {
    const void *data = get_vector(key);
    if (data == nullptr) {
      return IndexError_NoExist;
    }
    block.reset(const_cast<void *>(data));
    return 0;
  }

  //! Retrieve the owner class
  const std::string &owner_class(void) const override {
    return owner_class_;
  }

  //! Append an element into holder
  bool emplace(uint64_t key, const ailego::NumericalVector<T> &vec) {
    if (!holder_.emplace(key, vec)) {
      return false;
    }
    indice_map_[key] = static_cast<int>(holder_.count() - 1);
    return true;
  }

  //! Append an element into holder
  bool emplace(uint64_t key, ailego::NumericalVector<T> &&vec) {
    if (!holder_.emplace(key, std::move(vec))) {
      return false;
    }
    indice_map_[key] = static_cast<int>(holder_.count() - 1);
    return true;
  }

 private:
  //! Members
  MultiPassNumericalIndexHolder<T> holder_;
  std::unordered_map<uint64_t, int> indice_map_;
  std::string owner_class_;
};

/*! Multi-Pass Binary Index Provider
 */
template <typename T>
class MultiPassBinaryIndexProvider : public IndexProvider {
 public:
  //! Constructor
  explicit MultiPassBinaryIndexProvider(size_t dim)
      : holder_(dim), owner_class_("MultiPassBinaryIndexProvider") {}

  //! Destructor
  virtual ~MultiPassBinaryIndexProvider(void) {}

  //! Retrieve count of elements in holder
  size_t count(void) const override {
    return holder_.count();
  }

  //! Retrieve dimension
  size_t dimension(void) const override {
    return holder_.dimension();
  }

  //! Retrieve element size in bytes
  size_t element_size(void) const override {
    return holder_.element_size();
  }

  //! Create a new iterator
  IndexHolder::Iterator::Pointer create_iterator(void) override {
    return holder_.create_iterator();
  }

  //! Retrieve a vector using a primary key
  const void *get_vector(const uint64_t key) const override {
    auto it = indice_map_.find(key);
    if (it == indice_map_.end()) {
      return nullptr;
    }
    return holder_.get_vector_by_index(it->second);
  }

  //! Retrieve a vector using a primary key
  int get_vector(const uint64_t key,
                 IndexStorage::MemoryBlock &block) const override {
    const void *data = get_vector(key);
    if (data == nullptr) {
      return IndexError_NoExist;
    }
    block.reset(const_cast<void *>(data));
    return 0;
  }

  //! Retrieve the owner class
  const std::string &owner_class(void) const override {
    return owner_class_;
  }

  //! Append an element into holder
  bool emplace(uint64_t key, const ailego::BinaryVector<T> &vec) {
    if (!holder_.emplace(key, vec)) {
      return false;
    }
    indice_map_[key] = static_cast<int>(holder_.count() - 1);
    return true;
  }

  //! Append an element into holder
  bool emplace(uint64_t key, ailego::BinaryVector<T> &&vec) {
    if (!holder_.emplace(key, std::move(vec))) {
      return false;
    }
    indice_map_[key] = static_cast<int>(holder_.count() - 1);
    return true;
  }

 private:
  //! Members
  MultiPassBinaryIndexHolder<T> holder_;
  std::unordered_map<uint64_t, int> indice_map_;
  std::string owner_class_;
};

/*! Multi-Pass Index Provider
 */
template <IndexMeta::DataType FT>
struct MultiPassIndexProvider;

/*! Multi-Pass Index Provider (BINARY32)
 */
template <>
struct MultiPassIndexProvider<IndexMeta::DataType::DT_BINARY32>
    : public MultiPassBinaryIndexProvider<uint32_t> {
  //! Constructor
  using MultiPassBinaryIndexProvider::MultiPassBinaryIndexProvider;

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_BINARY32;
  }
};

/*! Multi-Pass Index Provider (BINARY64)
 */
template <>
struct MultiPassIndexProvider<IndexMeta::DataType::DT_BINARY64>
    : public MultiPassBinaryIndexProvider<uint64_t> {
  //! Constructor
  using MultiPassBinaryIndexProvider::MultiPassBinaryIndexProvider;

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_BINARY64;
  }
};

/*! Multi-Pass Index Provider (FP16)
 */
template <>
struct MultiPassIndexProvider<IndexMeta::DataType::DT_FP16>
    : public MultiPassNumericalIndexProvider<ailego::Float16> {
  //! Constructor
  using MultiPassNumericalIndexProvider::MultiPassNumericalIndexProvider;

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_FP16;
  }
};

/*! Multi-Pass Index Provider (FP32)
 */
template <>
struct MultiPassIndexProvider<IndexMeta::DataType::DT_FP32>
    : public MultiPassNumericalIndexProvider<float> {
  //! Constructor
  using MultiPassNumericalIndexProvider::MultiPassNumericalIndexProvider;

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_FP32;
  }
};

/*! Multi-Pass Index Provider (FP64)
 */
template <>
struct MultiPassIndexProvider<IndexMeta::DataType::DT_FP64>
    : public MultiPassNumericalIndexProvider<double> {
  //! Constructor
  using MultiPassNumericalIndexProvider::MultiPassNumericalIndexProvider;

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_FP64;
  }
};

/*! Multi-Pass Index Provider (INT8)
 */
template <>
struct MultiPassIndexProvider<IndexMeta::DataType::DT_INT8>
    : public MultiPassNumericalIndexProvider<int8_t> {
  //! Constructor
  using MultiPassNumericalIndexProvider::MultiPassNumericalIndexProvider;

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_INT8;
  }
};

/*! Multi-Pass Index Provider (INT16)
 */
template <>
struct MultiPassIndexProvider<IndexMeta::DataType::DT_INT16>
    : public MultiPassNumericalIndexProvider<int16_t> {
  //! Constructor
  using MultiPassNumericalIndexProvider::MultiPassNumericalIndexProvider;

  //! Retrieve type information
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_INT16;
  }
};

/*! Convert IndexHolder to IndexProvider
 *  @param holder The IndexHolder to convert
 *  @return IndexProvider::Pointer
 */
inline IndexProvider::Pointer convert_holder_to_provider(
    const IndexHolder::Pointer &holder) {
  if (!holder) {
    return nullptr;
  }

  IndexMeta::DataType data_type = holder->data_type();
  size_t dimension = holder->dimension();

  switch (data_type) {
    case IndexMeta::DataType::DT_FP16: {
      auto provider = std::make_shared<
          MultiPassIndexProvider<IndexMeta::DataType::DT_FP16>>(dimension);
      auto iter = holder->create_iterator();
      while (iter->is_valid()) {
        uint64_t key = iter->key();
        const ailego::Float16 *data =
            static_cast<const ailego::Float16 *>(iter->data());
        ailego::NumericalVector<ailego::Float16> vec(dimension);
        std::memcpy(vec.data(), data, dimension * sizeof(ailego::Float16));
        provider->emplace(key, std::move(vec));
        iter->next();
      }
      return provider;
    }

    case IndexMeta::DataType::DT_FP32: {
      auto provider = std::make_shared<
          MultiPassIndexProvider<IndexMeta::DataType::DT_FP32>>(dimension);
      auto iter = holder->create_iterator();
      while (iter->is_valid()) {
        uint64_t key = iter->key();
        const float *data = static_cast<const float *>(iter->data());
        ailego::NumericalVector<float> vec(dimension);
        std::memcpy(vec.data(), data, dimension * sizeof(float));
        provider->emplace(key, std::move(vec));
        iter->next();
      }
      return provider;
    }

    case IndexMeta::DataType::DT_FP64: {
      auto provider = std::make_shared<
          MultiPassIndexProvider<IndexMeta::DataType::DT_FP64>>(dimension);
      auto iter = holder->create_iterator();
      while (iter->is_valid()) {
        uint64_t key = iter->key();
        const double *data = static_cast<const double *>(iter->data());
        ailego::NumericalVector<double> vec(dimension);
        std::memcpy(vec.data(), data, dimension * sizeof(double));
        provider->emplace(key, std::move(vec));
        iter->next();
      }
      return provider;
    }

    case IndexMeta::DataType::DT_INT8: {
      auto provider = std::make_shared<
          MultiPassIndexProvider<IndexMeta::DataType::DT_INT8>>(dimension);
      auto iter = holder->create_iterator();
      while (iter->is_valid()) {
        uint64_t key = iter->key();
        const int8_t *data = static_cast<const int8_t *>(iter->data());
        ailego::NumericalVector<int8_t> vec(dimension);
        std::memcpy(vec.data(), data, dimension * sizeof(int8_t));
        provider->emplace(key, std::move(vec));
        iter->next();
      }
      return provider;
    }

    case IndexMeta::DataType::DT_INT16: {
      auto provider = std::make_shared<
          MultiPassIndexProvider<IndexMeta::DataType::DT_INT16>>(dimension);
      auto iter = holder->create_iterator();
      while (iter->is_valid()) {
        uint64_t key = iter->key();
        const int16_t *data = static_cast<const int16_t *>(iter->data());
        ailego::NumericalVector<int16_t> vec(dimension);
        std::memcpy(vec.data(), data, dimension * sizeof(int16_t));
        provider->emplace(key, std::move(vec));
        iter->next();
      }
      return provider;
    }

    case IndexMeta::DataType::DT_BINARY32: {
      auto provider = std::make_shared<
          MultiPassIndexProvider<IndexMeta::DataType::DT_BINARY32>>(dimension);
      auto iter = holder->create_iterator();
      while (iter->is_valid()) {
        uint64_t key = iter->key();
        const uint32_t *data = static_cast<const uint32_t *>(iter->data());
        size_t binary_size = (dimension + 31) / 32;
        ailego::BinaryVector<uint32_t> vec(dimension);
        std::memcpy(vec.data(), data, binary_size * sizeof(uint32_t));
        provider->emplace(key, std::move(vec));
        iter->next();
      }
      return provider;
    }

    case IndexMeta::DataType::DT_BINARY64: {
      auto provider = std::make_shared<
          MultiPassIndexProvider<IndexMeta::DataType::DT_BINARY64>>(dimension);
      auto iter = holder->create_iterator();
      while (iter->is_valid()) {
        uint64_t key = iter->key();
        const uint64_t *data = static_cast<const uint64_t *>(iter->data());
        size_t binary_size = (dimension + 63) / 64;
        ailego::BinaryVector<uint64_t> vec(dimension);
        std::memcpy(vec.data(), data, binary_size * sizeof(uint64_t));
        provider->emplace(key, std::move(vec));
        iter->next();
      }
      return provider;
    }

    default:
      return nullptr;
  }
}

}  // namespace core
}  // namespace zvec
