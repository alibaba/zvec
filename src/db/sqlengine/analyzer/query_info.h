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

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <zvec/ailego/logger/logger.h>
#include <zvec/core/framework/index_meta.h>
#include <zvec/db/schema.h>
#include "db/common/constants.h"
#include "db/sqlengine/common/group_by.h"
#include "query_field_info.h"
#include "query_node.h"
#include "query_orderby_info.h"

namespace zvec::sqlengine {

struct FieldAndSchema {
  FieldAndSchema(std::string field, const FieldSchema *schema)
      : field_name(std::move(field)), field_schema_ptr(schema) {}

  std::string field_name;
  const FieldSchema *field_schema_ptr;
};

class QueryInfo {
 public:
  using Ptr = std::shared_ptr<QueryInfo>;

  class QueryVectorCondInfo {
   public:
    using Ptr = std::shared_ptr<QueryVectorCondInfo>;

    QueryVectorCondInfo(const FieldSchema *vector_schema,
                        const std::string &vector_term,
                        core::IndexMeta::DataType core_data_type, int dimension,
                        std::string vector_sparse_indices,
                        std::string vector_sparse_values,
                        QueryParams::Ptr query_params)
        : vector_schema_(vector_schema),
          vector_term_(vector_term),
          data_type_(core_data_type),
          dimension_(dimension),
          vector_sparse_indices_(std::move(vector_sparse_indices)),
          vector_sparse_values_(std::move(vector_sparse_values)),
          query_params_(std::move(query_params)) {
      auto *vector_params = dynamic_cast<VectorIndexParams *>(
          vector_schema_->index_params().get());
      if (vector_params && vector_params->metric_type() == MetricType::IP) {
        reverse_sort_ = true;
      }
    }

   public:
    std::string vector_field_name() const {
      return vector_schema_->name();
    }

    const FieldSchema *vector_schema() const {
      return vector_schema_;
    }

    const std::string &vector_term() const {
      return vector_term_;
    }

    core::IndexMeta::DataType data_type() const {
      return data_type_;
    }

    uint32_t dimension() const {
      return dimension_;
    }

    uint32_t post_filter_topk() const {
      return 0;
    }

    int batch() const {
      return 1;
    }

    uint32_t sparse_count() const {
      return vector_sparse_indices_.size() / sizeof(uint32_t);
    }

    const std::string &vector_sparse_indices() const {
      return vector_sparse_indices_;
    }

    const std::string &vector_sparse_values() const {
      return vector_sparse_values_;
    }

    bool is_reverse_sort() const {
      return reverse_sort_;
    }

    const QueryParams::Ptr &query_params() const {
      return query_params_;
    }

   private:
    const FieldSchema *vector_schema_{nullptr};
    std::string vector_term_{""};
    core::IndexMeta::DataType data_type_;
    uint32_t dimension_{0};
    std::string vector_sparse_indices_{""};
    std::string vector_sparse_values_{""};
    QueryParams::Ptr query_params_;
    bool reverse_sort_{false};
  };

 public:
  QueryInfo() = default;
  ~QueryInfo() = default;

  void set_search_cond(QueryNode::Ptr value) {
    search_cond_ = std::move(value);
  }

  QueryNode::Ptr search_cond() const {
    return search_cond_;
  }

  void set_invert_cond(QueryNode::Ptr value) {
    invert_cond_ = std::move(value);
  }

  QueryNode::Ptr invert_cond() const {
    return invert_cond_;
  }

  void set_filter_cond(QueryNode::Ptr value) {
    filter_cond_ = std::move(value);
  }

  QueryNode::Ptr filter_cond() const {
    return filter_cond_;
  }

  void set_vector_cond_info(QueryVectorCondInfo::Ptr value) {
    vector_cond_info_ = std::move(value);
  }

  const QueryVectorCondInfo::Ptr &vector_cond_info() const {
    return vector_cond_info_;
  }

  void set_query_topn(uint32_t value) {
    query_topn_ = value;
  }

  uint32_t query_topn() const {
    return query_topn_;
  }

  const std::vector<QueryFieldInfo::Ptr> &query_fields() const {
    return query_fields_;
  }

  void add_query_field(QueryFieldInfo::Ptr &&query_field_info) {
    query_fields_.emplace_back(query_field_info);
  }

  const std::vector<QueryOrderbyInfo::Ptr> &query_orderbys() const {
    return query_orderbys_;
  }

  void add_query_orderby(QueryOrderbyInfo::Ptr &&query_orderby_info) {
    query_orderbys_.emplace_back(query_orderby_info);
  }

  void add_select_item_schema_ptr(
      std::string field, const zvec::FieldSchema *select_item_schema_ptr) {
    bool is_vector_field = false;
    if (select_item_schema_ptr != nullptr &&
        FieldSchema::is_vector_field(select_item_schema_ptr->data_type())) {
      is_vector_field = true;
    }
    add_fetched_schema(field, select_item_schema_ptr);
    if (is_vector_field) {
      selected_vector_fields_.emplace_back(field, select_item_schema_ptr);
    } else {
      selectd_scalar_field_names_.emplace_back(field);
    }
    select_item_schema_ptrs_.emplace_back(std::move(field),
                                          std::move(select_item_schema_ptr));
  }

  const std::vector<FieldAndSchema> &select_item_schema_ptrs() const {
    return select_item_schema_ptrs_;
  }

  void add_forward_filter_schema_ptr(
      std::string field, const zvec::FieldSchema *forward_filter_schema_ptr) {
    add_fetched_schema(field, forward_filter_schema_ptr);
    if (forward_filter_field_names_set_.emplace(field).second) {
      forward_filter_field_names_.emplace_back(std::move(field));
    }
  }

  void add_orderby_item_schema_ptr(
      std::string field, const zvec::FieldSchema *orderby_item_schema_ptr) {
    add_fetched_schema(field, orderby_item_schema_ptr);
    orderby_item_schema_ptrs_.emplace_back(std::move(field),
                                           orderby_item_schema_ptr);
  }

  const std::vector<FieldAndSchema> &orderby_item_schema_ptrs() const {
    return orderby_item_schema_ptrs_;
  }

  void add_fetched_schema(std::string field,
                          const zvec::FieldSchema *other_item_schema_ptr) {
    auto res = all_fetched_schema_schemas_.emplace(std::move(field),
                                                   other_item_schema_ptr);
    if (res.second &&
        !FieldSchema::is_vector_field(other_item_schema_ptr->data_type())) {
      all_fetched_scalar_field_names_.emplace_back(
          other_item_schema_ptr->name());
    }
  }

  const std::unordered_map<std::string, const FieldSchema *> &
  all_fetched_schemas() const {
    return all_fetched_schema_schemas_;
  }

  bool is_field_fetched(const std::string &field) const {
    return all_fetched_schema_schemas_.count(field) > 0;
  }

  const std::vector<std::string> &get_selected_scalar_field_names() {
    return selectd_scalar_field_names_;
  }

  const std::vector<std::string> &get_all_fetched_scalar_field_names() {
    return all_fetched_scalar_field_names_;
  }

  const std::vector<std::string> &get_forward_filter_field_names() {
    return forward_filter_field_names_;
  };


  bool exists_in_query_fields(const std::string &field_name) {
    for (auto query_field_info : query_fields_) {
      if (field_name == query_field_info->field_name()) {
        return true;
      }
    }
    return false;
  }

  void set_post_invert_cond(const QueryNode::Ptr &value) {
    post_invert_cond_ = value;
  }

  const QueryNode::Ptr &post_invert_cond() const {
    return post_invert_cond_;
  }

  void set_post_filter_cond(const QueryNode::Ptr &value) {
    post_filter_cond_ = value;
  }

  const QueryNode::Ptr &post_filter_cond() const {
    return post_filter_cond_;
  }

  void set_asterisk(bool value) {
    asterisk_ = value;
  }

  bool is_asterisk() const {
    return asterisk_;
  }

  void set_include_vector(bool value) {
    include_vector_ = value;
  }

  bool is_include_vector() const {
    return include_vector_;
  }

  void set_include_doc_id(bool value) {
    include_doc_id_ = value;
    if (include_doc_id_) {
      selectd_scalar_field_names_.emplace_back(GLOBAL_DOC_ID);
      all_fetched_scalar_field_names_.emplace_back(GLOBAL_DOC_ID);
    }
  }

  bool is_include_doc_id() const {
    return include_doc_id_;
  }

  const std::vector<FieldAndSchema> &selected_vector_fields() const {
    return selected_vector_fields_;
  }

  void set_group_by(GroupBy::Ptr group_by) {
    group_by_ = std::move(group_by);
  }
  const GroupBy::Ptr &group_by() const {
    return group_by_;
  }

  void set_group_by_schema_ptr(const FieldSchema *group_by_schema_ptr) {
    group_by_schema_ptr_ = group_by_schema_ptr;
  }
  const FieldSchema *group_by_schema_ptr() const {
    return group_by_schema_ptr_;
  }

  std::string to_string() const;

  bool is_filter_unsatisfiable() const;

 private:
  QueryNode::Ptr search_cond_{nullptr};

  QueryNode::Ptr invert_cond_{nullptr};
  QueryNode::Ptr filter_cond_{nullptr};

  QueryVectorCondInfo::Ptr vector_cond_info_{nullptr};

  // these two are for post filtering only
  QueryNode::Ptr post_invert_cond_{nullptr};
  QueryNode::Ptr post_filter_cond_{nullptr};

  uint32_t query_topn_{0};
  std::vector<QueryFieldInfo::Ptr> query_fields_{};
  std::vector<QueryOrderbyInfo::Ptr> query_orderbys_{};

  GroupBy::Ptr group_by_{};

  // from analyzing
  std::unordered_set<std::string> forward_filter_field_names_set_{};
  std::vector<std::string> forward_filter_field_names_{};
  // USER_ID are system needed fields
  std::vector<std::string> selectd_scalar_field_names_{USER_ID};
  std::vector<FieldAndSchema> select_item_schema_ptrs_{};
  std::vector<FieldAndSchema> orderby_item_schema_ptrs_{};
  // all fetched field schemas from forward, including user select fields and
  // system needed fields
  std::unordered_map<std::string, const FieldSchema *>
      all_fetched_schema_schemas_{};
  std::vector<std::string> all_fetched_scalar_field_names_{USER_ID,
                                                           LOCAL_ROW_ID};

  bool asterisk_{false};
  bool include_vector_{false};
  bool include_doc_id_{false};
  std::vector<FieldAndSchema> selected_vector_fields_{};
  const FieldSchema *group_by_schema_ptr_{};
};

}  // namespace zvec::sqlengine
