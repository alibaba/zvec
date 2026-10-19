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

#include "db/sqlengine/planner/doc_filter.h"
#include <optional>
#include <arrow/acero/exec_plan.h>
#include <arrow/table.h>
#include <zvec/ailego/logger/logger.h>
#include "db/sqlengine/planner/invert_search.h"

namespace zvec::sqlengine {

arrow::Future<> DocFilter::compute_filter_async() {
  if (invert_cond_) {
    InvertSearch search(segment_.get());
    auto invert_res = search.exec_invert_cond_tree(invert_cond_.get());
    if (!invert_res) {
      return arrow::Status::ExecutionError("Execute invert search failed: ",
                                           invert_res.error().message());
    }
    invert_result_ = invert_res.value();
    invert_filter_ = invert_result_->make_filter();
  }

  if (forward_filter_expr_) {
    // get schema to bind to Expression
    auto table = segment_->fetch(query_info_->get_forward_filter_field_names(),
                                 std::vector<int>{});
    if (!table) {
      return arrow::Status::ExecutionError("Fetch forward failed");
    }
    auto bind_res = forward_filter_expr_->Bind(*table->schema());
    if (!bind_res.ok()) {
      return arrow::Status::ExecutionError(
          "Bind forward filter expression failed",
          bind_res.status().ToString());
    }
    *forward_filter_expr_ = bind_res.MoveValueUnsafe();
  }
  if (forward_plan_) {
    // This subplan can run inside an Arrow source callback. Return its future
    // so the parent releases its CPU worker while the bitmap is computed.
    return arrow::acero::DeclarationToTableAsync(*forward_plan_)
        .Then([self = shared_from_this()](
                  const std::shared_ptr<arrow::Table> &table) {
          // The bitmap has only one column with boolean type.
          if (table->num_columns() != 1 ||
              table->column(0)->type() != arrow::boolean()) {
            return arrow::Status::ExecutionError(
                "Filter bitmap is not boolean type");
          }
          self->forward_bitmap_ = table->column(0);
          return arrow::Status::OK();
        });
  }
  return arrow::Status::OK();
}

bool DocFilter::empty() const {
  return !(delete_filter_ || invert_filter_ || forward_plan_ ||
           forward_filter_expr_);
}

bool DocFilter::is_filtered(uint64_t id) const {
  if (delete_filter_ && delete_filter_->is_filtered(id)) {
    return true;
  }
  if (invert_filter_ && invert_filter_->is_filtered(id)) {
    return true;
  }
  auto forward_bit = get_forward_bit(id);
  if (!forward_bit) {
    return false;
  }
  // revert to return false if forward filter is matched
  return !forward_bit.value();
}

std::optional<bool> DocFilter::get_forward_bit(uint64_t id) const {
  if (forward_filter_expr_) {
    return is_matched_by_forward_filter(id);
  }
  if (!forward_bitmap_) {
    return std::nullopt;
  }
  uint64_t rows_seen = 0;
  for (int c = 0; c < forward_bitmap_->num_chunks(); c++) {
    const auto &arr = forward_bitmap_->chunk(c);
    if (id < rows_seen + arr->length()) {
      auto *bool_array = static_cast<arrow::BooleanArray *>(arr.get());
      return (*bool_array)[id - rows_seen].value_or(false);
    }
    rows_seen += arr->length();
  }
  LOG_ERROR("ID is out or range: id[%zu] count[%zu]", (size_t)id,
            (size_t)rows_seen);
  return std::nullopt;
}

std::optional<std::vector<uint64_t>> DocFilter::get_bf_by_keys_and_update(
    float ratio) {
  // TODO: support forward
  if (!invert_result_) {
    return std::nullopt;
  }
  // Insert/flush() rewrite the block behind doc_count(), so read it locked.
  size_t doc_count = segment_->doc_count_snapshot();
  uint64_t bf_by_keys_threshold = static_cast<uint64_t>(doc_count * ratio);

  // decide to use brute force by keys or not
  if (size_t match_count = invert_result_->count();
      match_count <= bf_by_keys_threshold) {
    std::vector<uint32_t> ids;
    invert_result_->extract_ids(&ids);
    invert_filter_.reset();
    invert_result_.reset();
    LOG_INFO(
        "Use brute force by keys, doc_count[%zu] invert_result_count[%zu] "
        "ratio[%.4f]",
        doc_count, match_count, ratio);
    return std::vector<uint64_t>(ids.begin(), ids.end());
  } else {
    LOG_DEBUG(
        "Not use brute force by keys, doc_count[%zu] invert_result_count[%zu] "
        "ratio[%.4f]",
        doc_count, match_count, ratio);
  }
  return std::nullopt;
}

std::optional<bool> DocFilter::is_matched_by_forward_filter(uint64_t id) const {
  auto exec_batch =
      segment_->fetch(query_info_->get_forward_filter_field_names(), id);
  if (!exec_batch) {
    LOG_ERROR("Fetch forward failed, id[%zu]", (size_t)id);
    return std::nullopt;
  }
  auto maybe_result = arrow::compute::ExecuteScalarExpression(
      *forward_filter_expr_, *exec_batch);
  if (!maybe_result.ok()) {
    LOG_ERROR("Execute scalar expression failed, id[%zu] err[%s]", (size_t)id,
              maybe_result.status().ToString().c_str());
    return std::nullopt;
  }
  arrow::Datum datum = maybe_result.MoveValueUnsafe();
  if (datum.is_scalar()) {
    const auto &scalar = datum.scalar_as<arrow::BooleanScalar>();
    return scalar.is_valid && scalar.value;
  }
  LOG_ERROR("Datum is not scalar, id[%zu] type[%s]", (size_t)id,
            datum.type()->ToString().c_str());
  return std::nullopt;
}


}  // namespace zvec::sqlengine
