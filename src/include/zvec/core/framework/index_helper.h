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

#include <zvec/core/framework/index_dumper.h>
#include <zvec/core/framework/index_holder.h>
#include <zvec/core/framework/index_meta.h>
#include <zvec/core/framework/index_storage.h>

namespace zvec {
namespace core {

/*! Index Helper
 */
struct IndexHelper {
  //! Serialize meta information to dumper
  static int SerializeToDumper(const IndexMeta &mt, IndexDumper *dumper,
                               const std::string &key);

  //! Serialize meta information to storage
  static int SerializeToStorage(const IndexMeta &mt, IndexStorage *storage,
                                const std::string &key);

  //! Derialize meta information from storage
  static int DeserializeFromStorage(IndexStorage *storage,
                                    const std::string &key, IndexMeta *out);

  //! Serialize meta information to dumper
  static int SerializeToDumper(const IndexMeta &mt, IndexDumper *dumper) {
    return SerializeToDumper(mt, dumper, "IndexMeta");
  }

  //! Serialize meta information to storage
  static int SerializeToStorage(const IndexMeta &mt, IndexStorage *storage) {
    return SerializeToStorage(mt, storage, "IndexMeta");
  }

  //! Derialize meta information from storage
  static int DeserializeFromStorage(IndexStorage *storage, IndexMeta *out) {
    return DeserializeFromStorage(storage, "IndexMeta", out);
  }

  //! Create a proxy holder that can be traversed twice.
  static IndexHolder::Pointer MakeTwoPassHolder(IndexHolder::Pointer holder);
};

}  // namespace core
}  // namespace zvec
