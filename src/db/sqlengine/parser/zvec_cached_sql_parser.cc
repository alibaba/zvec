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

#include "zvec_cached_sql_parser.h"
#include <exception>
#include <typeinfo>
#include <ailego/logger/logger.h>
#include <zvec/ailego/utility/string_helper.h>
#include "atn/ParserATNSimulator.h"
#include "db/sqlengine/antlr/gen/SQLLexer.h"
#include "db/sqlengine/antlr/gen/SQLParser.h"
#include "db/sqlengine/common/util.h"
#include "case_changing_charstream.h"
#include "error_verbose_listener.h"
#include "node.h"
#include "select_info.h"
#include "selected_elem_info.h"

using namespace antlr4;
using namespace tree;
using namespace atn;

namespace zvec::sqlengine {

std::unordered_map<std::string, SQLInfo::Ptr>
    ZVecCachedSQLParser::sql_info_map_{};
std::unordered_map<std::string, Node::Ptr> ZVecCachedSQLParser::filter_map_;
uint32_t ZVecCachedSQLParser::Hit{0};
uint32_t ZVecCachedSQLParser::Miss{0};

ZVecCachedSQLParser::ZVecCachedSQLParser(uint32_t cache_count)
    : cache_count_(cache_count) {}

ZVecCachedSQLParser::~ZVecCachedSQLParser() {}

SQLInfo::Ptr ZVecCachedSQLParser::parse(const std::string &query,
                                        bool need_formatted_tree) {
  std::string query_cache_key{""};
  SQLInfo::Ptr cached_sql_info = get_from_cache(query, &query_cache_key);
  if (cached_sql_info != nullptr) {
    return cached_sql_info;
  }

  SQLInfo::Ptr new_sql_info = real_parser_.parse(query, need_formatted_tree);
  if (new_sql_info == nullptr) {
    // no need to cache parse failed sql. just return.
    err_msg_ = real_parser_.err_msg();
    return nullptr;
  }

  put_into_cache(query_cache_key, new_sql_info);

  return new_sql_info;
}

void ZVecCachedSQLParser::put_into_cache(const std::string &query_cache_key,
                                         const SQLInfo::Ptr &new_sql_info) {
  {
    std::unique_lock guard(shared_mutex_);
    if (sql_info_map_.size() >= cache_count_) {
      // if full, clear to refresh new sql
      sql_info_map_.clear();
      Hit = Miss = 0;
    }
    sql_info_map_.emplace(query_cache_key, new_sql_info);
  }

  LOG_DEBUG("cache emplaced. [%s] [%s] ", query_cache_key.c_str(),
            new_sql_info->to_string().c_str());

  return;
}

SQLInfo::Ptr ZVecCachedSQLParser::get_from_cache(const std::string &query,
                                                 std::string *query_cache_key) {
  // find [ and ], must only one occurrence.
  std::string::size_type left_pos, right_pos;
  left_pos = query.find("[");
  if (left_pos == query.npos) {
    return nullptr;
  }
  // find from left_pos+1
  right_pos = query.rfind("]");
  if (right_pos == query.npos) {
    return nullptr;
  }

  // ok, let's find it.
  *query_cache_key = query.substr(0, left_pos);
  query_cache_key->append(query.begin() + right_pos + 1, query.end());
  std::string vector_text = query.substr(left_pos, right_pos - left_pos + 1);

  SQLInfo::Ptr cached_sql_info = nullptr;
  SQLInfo::Ptr copied_sql_info = nullptr;
  {  // lock only in this block. after sql_info is copied, just unlock.
    std::shared_lock guard(shared_mutex_);
    auto iter = sql_info_map_.find(*query_cache_key);
    if (iter == sql_info_map_.end()) {
      ++Miss;
      LOG_DEBUG("cache miss. key: [%s]", query_cache_key->c_str());
      return nullptr;
    }

    cached_sql_info = iter->second;
    // copy cached_sql_info
    copied_sql_info = std::make_shared<SQLInfo>(*cached_sql_info);
  }

  // parse vector part
  Node::Ptr vector_node = parse_vector_text(&vector_text);
  if (vector_node == nullptr) {
    LOG_DEBUG("wrong vector format: [%s]", vector_text.c_str());
    return nullptr;
  }
  // replace vector in copied_sql_info
  if (replace_vector_node(copied_sql_info, vector_node) != 0) {
    LOG_WARN("replace_vector_node failed. [%s][%s]", query.c_str(),
             vector_text.c_str());
    return nullptr;
  }

  ++Hit;
  LOG_DEBUG("cache hit. key: [%s] sql_info: [%s]", query_cache_key->c_str(),
            copied_sql_info->to_string().c_str());
  return copied_sql_info;
}

int ZVecCachedSQLParser::replace_vector_node(SQLInfo::Ptr cached_sql_info,
                                             Node::Ptr vector_node) {
  SelectInfo::Ptr cached_select_info =
      std::dynamic_pointer_cast<SelectInfo>(cached_sql_info->base_info());
  if (cached_select_info == nullptr) {
    LOG_WARN("wrong select_info in cache. [%s]",
             cached_sql_info->to_string().c_str());
    return -1;
  }

  Node::Ptr search_cond = cached_select_info->mutable_search_cond();
  if (search_cond == nullptr) {
    LOG_WARN("wrong search_cond in cache. [%s]",
             cached_sql_info->to_string().c_str());
    return -1;
  }

  replace_flag_ = false;
  if (traverse_to_replace(search_cond, vector_node) != 0 ||
      replace_flag_ == false) {
    LOG_WARN("replace search_cond in cache failed. [%s]",
             cached_sql_info->to_string().c_str());
    return -1;
  }

  return 0;
}

int ZVecCachedSQLParser::traverse_to_replace(Node::Ptr ptr,
                                             Node::Ptr vector_node) {
  if (ptr->op() == NodeOp::T_VECTOR_MATRIX_VALUE) {
    Node *parent = dynamic_cast<Node *>(ptr->parent());
    if (parent == nullptr) {
      LOG_WARN("wrong parent node in cache. [%s]", ptr->to_string().c_str());
      return -1;
    }
    if (parent->left() == ptr) {
      parent->set_left(vector_node);
      replace_flag_ = true;
    } else if (parent->right() == ptr) {
      parent->set_right(vector_node);
      replace_flag_ = true;
    } else {
      LOG_WARN("wrong node in cache. [%s]", ptr->to_string().c_str());
      return -1;
    }
    return 0;
  }

  if (ptr->left() != nullptr) {
    if (traverse_to_replace(ptr->left(), vector_node) < 0) {
      return -1;
    }
    if (replace_flag_) {
      return 0;
    }
  }
  if (ptr->right() != nullptr) {
    if (traverse_to_replace(ptr->right(), vector_node) != 0) {
      return -1;
    }
    if (replace_flag_) {
      return 0;
    }
  }

  return 0;
}

Node::Ptr ZVecCachedSQLParser::parse_filter(const std::string &filter,
                                            bool need_formatted_tree) {
  {
    std::shared_lock guard(shared_mutex_);
    auto iter = filter_map_.find(filter);
    if (iter != filter_map_.end()) {
      ++Hit;
      return iter->second;
    }
    ++Miss;
  }
  auto node = real_parser_.parse_filter(filter, need_formatted_tree);
  err_msg_ = real_parser_.err_msg();
  formatted_tree_ = real_parser_.formatted_tree();
  if (node != nullptr) {
    std::unique_lock guard(shared_mutex_);
    if (filter_map_.size() >= cache_count_) {
      // clear cache if full
      filter_map_.clear();
      Hit = Miss = 0;
    }
    filter_map_.emplace(filter, node);
  }
  return node;
}

}  // namespace zvec::sqlengine
