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
#include <zvec/export.h>

namespace zvec {

class ZVEC_API DocIterator {
 public:
  // Pimpl: Impl holds Arrow types and is defined only in an internal
  // header (src/db/doc_iterator_internal.h).
  struct Impl;

  using Ptr = std::shared_ptr<DocIterator>;

  // Public only so the collection implementation can use std::make_shared;
  // Impl is incomplete here, so external code cannot call this.
  explicit DocIterator(std::unique_ptr<Impl> impl);

  // !has_value() → error
  // has_value() && value() == nullptr → EOF
  // has_value() && value() != nullptr → success
  Result<Doc::Ptr> next();

  void close();

  ~DocIterator();

 private:
  std::unique_ptr<Impl> impl_;
};

}  // namespace zvec
