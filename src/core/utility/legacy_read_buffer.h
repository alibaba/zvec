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
#include <zvec/core/framework/index_storage.h>

namespace zvec {
namespace core {

// Compatibility storage for pointer-returning reads. Retain copied bytes, never
// a page pin or a storage handle. Mmap views keep their original storage
// lifetime.
class LegacyReadBuffer {
 public:
  const void *assign(const IndexStorage::MemoryBlock &block, size_t size) {
    if (!block.data() || block.type_ == IndexStorage::MemoryBlock::MBT_MMAP) {
      return block.data();
    }
    data_.assign(static_cast<const char *>(block.data()), size);
    return data_.data();
  }

 private:
  std::string data_;
};

}  // namespace core
}  // namespace zvec
