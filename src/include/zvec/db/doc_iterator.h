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

#include <memory>
#include <zvec/db/doc.h>
#include <zvec/db/status.h>

namespace zvec {

class DocIterator {
 public:
  // Pimpl: the implementation holds Arrow types (RecordBatchReader,
  // RecordBatch). Only a forward declaration is exposed here; the definition
  // lives in an internal header (src/db/doc_iterator_internal.h), so external
  // code cannot construct or inspect it.
  struct Impl;

  using Ptr = std::shared_ptr<DocIterator>;

  // Constructed only by the collection implementation (which can build an
  // Impl via the internal header). Public so std::make_shared can be used;
  // external code cannot call it because Impl is incomplete here.
  explicit DocIterator(std::unique_ptr<Impl> impl);

  // !has_value() → error
  // has_value() && value() == nullptr → EOF
  // has_value() && value() != nullptr → success
  Result<Doc::Ptr> Next();

  void Close();

  ~DocIterator();

 private:
  std::unique_ptr<Impl> impl_;
};

}  // namespace zvec
