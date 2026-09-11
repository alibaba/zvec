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

#include <cstddef>
#include <cstdint>
#include <memory>

namespace zvec {
namespace core {

// Internal, optional capability; does not change the public IndexHolder ABI.
// The source must remain immutable and outlive its readers. Ordinals refer to
// the holder's iteration order, not its keys. Each reader is single-threaded.
class OrdinalAccessHolder {
 public:
  class Reader {
   public:
    using Pointer = std::unique_ptr<Reader>;
    virtual ~Reader() = default;

    // On success, data is valid until the next read/reset or reader
    // destruction. Consumers must copy it before then; it may be a provider's
    // scratch buffer.
    virtual int read(size_t ordinal, uint64_t *key, const void **data) = 0;

    // Release the current provider/buffers, retaining the ordinal mapping so
    // that a later dump can read the source again.
    virtual void reset() = 0;
  };

  virtual ~OrdinalAccessHolder() = default;

  // All errors must leave reader unchanged. NotImplemented must consume no
  // source data; other errors are not a request to fall back to
  // materialization.
  virtual int create_ordinal_reader(Reader::Pointer *reader) = 0;
};

}  // namespace core
}  // namespace zvec
