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

#include <shared_mutex>
#include <unordered_map>
#include "select_info.h"
#include "zvec_sql_parser.h"

namespace zvec::sqlengine {

class ZVecCachedSQLParser : public ZVecParser {
 public:
  ZVecCachedSQLParser(uint32_t cache_count);
  ~ZVecCachedSQLParser() override;

  SQLInfo::Ptr parse(const std::string &query,
                     bool need_formatted_tree = false) override;

  Node::Ptr parse_filter(const std::string &filter,
                         bool need_formatted_tree = false) override;

 private:
  void put_into_cache(const std::string &query_cache_key,
                      const SQLInfo::Ptr &sql_info);
  SQLInfo::Ptr get_from_cache(const std::string &query,
                              std::string *query_cache_key);

  int replace_vector_node(SQLInfo::Ptr cached_sql_info, Node::Ptr vector_node);
  int traverse_to_replace(Node::Ptr ptr, Node::Ptr vector_node);

 private:
  static std::unordered_map<std::string, SQLInfo::Ptr> sql_info_map_;
  static std::unordered_map<std::string, Node::Ptr> filter_map_;
  static uint32_t hit;
  static uint32_t miss;
  inline static std::shared_mutex shared_mutex_;

 private:
  bool replace_flag_{false};
  ZVecSQLParser real_parser_;
  uint32_t cache_count_{0};
};

}  // namespace zvec::sqlengine
