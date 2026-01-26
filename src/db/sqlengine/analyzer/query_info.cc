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

#include "query_info.h"
#include <zvec/ailego/utility/float_helper.h>
#include <zvec/ailego/utility/string_helper.h>
#include "db/common/constants.h"

namespace zvec::sqlengine {

std::string QueryInfo::to_string() const {
  std::string str = "Query Info: {\n";

  if (!query_fields_.empty()) {
    str += "query_fields: ";
    for (auto iter = query_fields_.begin(); iter != query_fields_.end();
         iter++) {
      if (iter != query_fields_.begin()) {
        str += ", ";
      }
      QueryFieldInfo::Ptr query_field_info_ptr = *iter;
      str += query_field_info_ptr->to_string();
    }
    str += "\n";
  }

  if (!query_orderbys_.empty()) {
    str += "query_orderbys: ";
    for (auto iter = query_orderbys_.begin(); iter != query_orderbys_.end();
         iter++) {
      if (iter != query_orderbys_.begin()) {
        str += ", ";
      }
      QueryOrderbyInfo::Ptr query_orderby_info_ptr = *iter;
      str += query_orderby_info_ptr->to_string();
    }
    str += "\n";
  }

  if (!all_fetched_schema_schemas_.empty()) {
    str += "all_fetched_field_schemas: ";
    for (auto iter = all_fetched_schema_schemas_.begin();
         iter != all_fetched_schema_schemas_.end(); iter++) {
      if (iter != all_fetched_schema_schemas_.begin()) {
        str += ", ";
      }
      str += iter->first;
    }
    str += "\n";
  }

  if (group_by_ != nullptr) {
    str += "group_by: " + group_by_->to_string() + "\n";
  }

  str += "query_topn: " + std::to_string(query_topn_) + " ";
  str += "\n";

  str += "search_cond:\n";
  if (search_cond_ != nullptr) {
    str += search_cond_->text();
    str += "\n";
  }

  str += "vector_cond:\n";
  if (vector_cond_info_ != nullptr) {
    ailego::StringHelper::Append(
        &str, vector_cond_info_->vector_field_name(), "=", "feature(",
        vector_cond_info_->batch() > 1 ? "matrix[[...],...]" : "vector[...]",
        ", ", vector_cond_info_->data_type(), ",", vector_cond_info_->batch(),
        ")(FEATURE",
        vector_cond_info_->vector_sparse_indices().empty() ? ""
                                                           : "_WITH_SPARSE",
        ")\n");
  }

  str += "filter_cond:\n";
  if (filter_cond_ != nullptr) {
    str += filter_cond_->text();
    str += "\n";
  }

  str += "invert_cond:\n";
  if (invert_cond_ != nullptr) {
    str += invert_cond_->text();
    str += "\n";
  }

  str += "}";
  return str;
}

bool QueryInfo::is_filter_unsatisfiable() const {
  if (invert_cond_ && invert_cond_->predictate_result().has_value() &&
      !invert_cond_->predictate_result().value()) {
    return true;
  }
  if (filter_cond_ && filter_cond_->predictate_result().has_value() &&
      !filter_cond_->predictate_result().value()) {
    return true;
  }
  return false;
}

}  // namespace zvec::sqlengine
