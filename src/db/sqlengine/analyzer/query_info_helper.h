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
#include "query_info.h"

namespace zvec::sqlengine {

struct SubRootResult {
  QueryNode *subroot;
  int32_t num_of_child;

  SubRootResult() : subroot(nullptr), num_of_child(0) {}

  SubRootResult(QueryNode *node, int32_t num)
      : subroot(node), num_of_child(num) {}

  void set_result(QueryNode *node, int32_t num) {
    if (subroot == nullptr || num_of_child < num) {
      subroot = node;
      num_of_child = num;
    }
  }
};

class QueryInfoHelper {
 public:
  static bool is_numeric_type(zvec::DataType data_type) {
    // include INT32, INT64, UINT32, UINT64, FLOAT, DOUBLE
    // use following code to reduce the runtime comparison cost
    return (data_type >= zvec::DataType::INT32 &&
            data_type <= zvec::DataType::DOUBLE);
  }

  static bool text_2_data_buf(const std::string &text, zvec::DataType data_type,
                              std::string *data_buf);
  static bool data_buf_2_text(const std::string &data_buf,
                              zvec::DataType data_type, std::string *text);
  static void constant_node_data_buf_2_text(DataType data_type,
                                            bool is_array_type,
                                            QueryNode *node);

  static void find_subroot_by_rule(
      QueryNode *root, const std::function<bool(QueryRelNode *node)> &rule,
      SubRootResult *subroot_result);

  static bool traverse_node_by_rule(
      QueryNode *node, const std::function<bool(QueryRelNode *node)> &rule,
      SubRootResult *subroot_result, int32_t *num_of_child);
};

}  // namespace zvec::sqlengine
