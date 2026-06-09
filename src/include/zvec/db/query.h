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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>
#include <zvec/db/doc.h>
#include <zvec/db/query_params.h>
#include <zvec/db/reranker.h>

namespace zvec {

struct VectorClause {
  std::string query_vector_;
  std::string sparse_indices_;
  std::string sparse_values_;
};

struct FtsClause {
  std::string query_string_;
  std::string match_string_;
};

struct QueryTarget {
  std::string field_name_;
  std::variant<VectorClause, FtsClause> clause_;
  QueryParams::Ptr query_params_;

  // Mutators ensure clause_ holds a VectorClause.
  void set_vector(std::string vector);
  void set_sparse_vector(std::string indices, std::string values);

  // nullptr when clause_ holds a non-VectorClause alternative.
  VectorClause *get_vector_clause() {
    return std::get_if<VectorClause>(&clause_);
  }
  const VectorClause *get_vector_clause() const {
    return std::get_if<VectorClause>(&clause_);
  }

  // nullptr when clause_ holds a non-FtsClause alternative.
  FtsClause *get_fts_clause() {
    return std::get_if<FtsClause>(&clause_);
  }
  const FtsClause *get_fts_clause() const {
    return std::get_if<FtsClause>(&clause_);
  }

 private:
  // Resets clause_ to an empty VectorClause unless it already holds one.
  VectorClause &ensure_vector_clause() {
    if (!std::holds_alternative<VectorClause>(clause_)) {
      clause_ = VectorClause{};
    }
    return std::get<VectorClause>(clause_);
  }
};

inline void QueryTarget::set_vector(std::string vector) {
  ensure_vector_clause().query_vector_ = std::move(vector);
}

inline void QueryTarget::set_sparse_vector(std::string indices,
                                           std::string values) {
  auto &vc = ensure_vector_clause();
  vc.sparse_indices_ = std::move(indices);
  vc.sparse_values_ = std::move(values);
}

struct SearchQuery {
  QueryTarget target_;
  int topk_{0};
  std::string filter_;
  bool include_vector_{false};
  bool include_doc_id_{false};
  // Field selection:
  //   nullopt   -> select all fields (select *)
  //   empty     -> select no field
  //   non-empty -> select only the listed fields
  std::optional<std::vector<std::string>> output_fields_;

  // FtsClause currently bypasses validation (FTS not yet implemented).
  Status validate_and_sanitize(const FieldSchema *schema);
};

struct GroupByVectorQuery {
  QueryTarget target_;
  std::string filter_;
  bool include_vector_{false};
  // Field selection:
  //   nullopt   -> select all fields (select *)
  //   empty     -> select no field
  //   non-empty -> select only the listed fields
  std::optional<std::vector<std::string>> output_fields_;
  std::string group_by_field_name_;
  uint32_t group_count_{2};
  uint32_t group_topk_{3};
};

struct GroupResult {
  std::string group_by_value_;
  std::vector<Doc> docs_;
};

using GroupResults = std::vector<GroupResult>;

//! Multi query structure for combining multiple sub-queries
//! (vector, full-text, etc.) with optional re-ranking of results.
struct SubQuery {
  QueryTarget target_;
  int num_candidates_{10};
};

struct MultiQuery {
  std::vector<SubQuery> queries;
  int topk{10};
  std::string filter;
  bool include_vector{false};
  bool include_doc_id_{false};
  // Field selection:
  //   nullopt   -> select all fields (select *)
  //   empty     -> select no field
  //   non-empty -> select only the listed fields
  std::optional<std::vector<std::string>> output_fields;
  reranker::RerankParams rerank;  // Value semantics, defaults to RRF k=60
};


}  // namespace zvec
