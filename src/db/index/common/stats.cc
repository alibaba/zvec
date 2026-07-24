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

#include <sstream>
#include <zvec/db/stats.h>
#include "db/common/utils.h"

namespace zvec {
std::string CollectionStats::to_string() const {
  std::ostringstream oss;
  oss << "CollectionStats{"
      << "doc_count:" << doc_count << ",index_completeness:{";

  size_t i = 0;
  for (const auto &pair : index_completeness) {
    if (i > 0) oss << ",";
    oss << pair.first << ":" << pair.second;
    ++i;
  }

  oss << "}}";
  return oss.str();
}

std::string CollectionStats::to_string_formatted(int indent_level) const {
  std::ostringstream oss;
  oss << indent(indent_level) << "CollectionStats{\n"
      << indent(indent_level + 1) << "doc_count: " << doc_count << ",\n"
      << indent(indent_level + 1) << "index_completeness: {\n";

  size_t i = 0;
  for (const auto &pair : index_completeness) {
    if (i > 0) oss << ",\n";
    oss << indent(indent_level + 2) << pair.first << ": " << pair.second;
    ++i;
  }

  if (!index_completeness.empty()) {
    oss << "\n";
  }
  oss << indent(indent_level + 1) << "}\n" << indent(indent_level) << "}";

  return oss.str();
}

}  // namespace zvec