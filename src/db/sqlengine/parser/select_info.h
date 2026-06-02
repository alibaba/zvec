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

#include <map>
#include <memory>
#include <vector>
#include "db/sqlengine/common/fts_cond_info.h"
#include "db/sqlengine/common/group_by.h"
#include "base_info.h"
#include "node.h"
#include "orderby_elem_info.h"
#include "selected_elem_info.h"

namespace zvec::sqlengine {

class SelectInfo : public BaseInfo {
 public:
  using Ptr = std::shared_ptr<SelectInfo>;

  SelectInfo(const std::string &m_table_name);
  ~SelectInfo();

  SelectInfo(const SelectInfo &info);
  SelectInfo &operator=(const SelectInfo &info);

  const std::vector<SelectedElemInfo::Ptr> &selected_elems();
  const std::vector<OrderByElemInfo::Ptr> &orderby_elems();
  int limit();
  const Node::Ptr &search_cond() const;
  Node::Ptr &mutable_search_cond();

  void add_selected_elem(SelectedElemInfo::Ptr selected_elem_info);
  void add_order_by_elem(OrderByElemInfo::Ptr orderby_elem_info);
  void set_limit(int value);
  void set_search_cond(Node::Ptr cond);

  void set_include_vector(bool value) {
    include_vector_ = value;
  }

  bool include_vector() {
    return include_vector_;
  }

  void set_include_doc_id(bool value) {
    include_doc_id_ = value;
  }

  bool is_include_doc_id() {
    return include_doc_id_;
  }

  void set_group_by(GroupBy::Ptr group_by) {
    group_by_ = std::move(group_by);
  }
  const GroupBy::Ptr &group_by() const {
    return group_by_;
  }

  void set_fts_cond_info(FtsCondInfo::Ptr value) {
    fts_cond_info_ = std::move(value);
  }

  const FtsCondInfo::Ptr &fts_cond_info() const {
    return fts_cond_info_;
  }

  bool has_fts_query() const {
    return fts_cond_info_ != nullptr;
  }

  std::string to_string();

 private:
  Node::Ptr copy_node(const Node::Ptr &node);

 private:
  std::vector<SelectedElemInfo::Ptr> selected_elems_{};
  std::vector<OrderByElemInfo::Ptr> orderby_elems_{};
  Node::Ptr search_cond_{nullptr};
  GroupBy::Ptr group_by_{};
  int limit_{-1};
  bool include_vector_{false};
  bool include_doc_id_{false};
  FtsCondInfo::Ptr fts_cond_info_{nullptr};
};

}  // namespace zvec::sqlengine
