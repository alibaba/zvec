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

// Internal header — NOT in public includes (src/include/).
// Gives the write path access to Doc::validate_and_sanitize(), which is private
// because it validates against an engine-resolved schema and may rewrite the
// document in place.
#pragma once

#include <memory>
#include <zvec/db/doc.h>
#include <zvec/db/schema.h>
#include <zvec/db/status.h>

namespace zvec {
namespace internal {

struct DocAccess {
  static Status validate_and_sanitize(Doc &doc,
                                      const CollectionSchema::Ptr &schema,
                                      bool is_update = false) {
    return doc.validate_and_sanitize(schema, is_update);
  }
};

}  // namespace internal
}  // namespace zvec
