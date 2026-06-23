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

#include "node.h"
#include "sql_info.h"

namespace zvec::sqlengine {

using VoidPtr = void *;

class ZVecParser {
 public:
  using Ptr = std::shared_ptr<ZVecParser>;

  ZVecParser() = default;
  virtual ~ZVecParser() {};

  virtual SQLInfo::Ptr parse(const std::string &query,
                             bool formatted_tree = false) = 0;

  virtual Node::Ptr parse_filter(const std::string &filter,
                                 bool need_formatted_tree = false) = 0;


 protected:
  std::string trim(std::string &value);
  virtual std::string to_formatted_string_tree(void *tree, void *parser);
  virtual void save_to_file(const std::string &file_name,
                            const std::string &formatted);

 public:
  virtual const std::string &err_msg();
  virtual const std::string &formatted_tree();

  Node::Ptr parse_vector_text(std::string *vector_text);

 public:
  static ZVecParser::Ptr create();
  static ZVecParser::Ptr create(int cache_count);
  const static int32_t DEFAULT_CACHE_COUNT{100};

 protected:
  std::string err_msg_{""};
  std::string formatted_tree_{""};
};

}  // namespace zvec::sqlengine
