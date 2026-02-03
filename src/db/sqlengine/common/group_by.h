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
#include <string>
#include <zvec/ailego/utility/string_helper.h>
#include <zvec/db/doc.h>

namespace zvec::sqlengine {

struct GroupBy {
  using Ptr = std::shared_ptr<GroupBy>;

  GroupBy() = default;

  GroupBy(std::string group_by_field, uint32_t group_topk, uint32_t group_count)
      : group_by_field(std::move(group_by_field)),
        group_topk(group_topk),
        group_count(group_count) {}

  std::string to_string() {
    return ailego::StringHelper::Concat("field[", group_by_field, "] topk[",
                                        group_topk, "] count[", group_count,
                                        "]");
  }

  std::string group_by_field;
  uint32_t group_topk{0};
  uint32_t group_count{0};
};


}  // namespace zvec::sqlengine
