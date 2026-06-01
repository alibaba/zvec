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
#include "db/index/column/fts_column/fts_query_ast.h"

namespace zvec::sqlengine {

struct FtsCondInfo {
  using Ptr = std::shared_ptr<FtsCondInfo>;

  FtsCondInfo() = default;

  FtsCondInfo(std::string field_name, fts::FtsAstNodePtr ast)
      : field_name(std::move(field_name)), fts_ast(std::move(ast)) {}

  std::string to_string() const {
    std::string str = field_name + " MATCH ";
    if (fts_ast) {
      str += fts_ast->text();
    }
    return str;
  }

  std::string field_name;
  fts::FtsAstNodePtr fts_ast;
};

}  // namespace zvec::sqlengine
