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

}  // namespace core
}  // namespace zvec
