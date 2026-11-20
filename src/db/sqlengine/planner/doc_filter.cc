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
#include <arrow/array.h>
#include <zvec/ailego/logger/logger.h>
#include "db/sqlengine/planner/invert_search.h"

namespace zvec::sqlengine {

Status DocFilter::compute_filter() {
  forward_prepared_ = false;
  forward_view_.reset();
  forward_rows_ = 0;
  sealed_deletion_.reset();
  if (invert_cond_) {
    InvertSearch search(segment_.get());
    auto invert_res = search.exec_invert_cond_tree(invert_cond_.get());
    if (!invert_res) {
      return Status::InternalError("Execute invert search failed: ",
                                   invert_res.error().message());
    }
    invert_result_ = invert_res.value();
    invert_filter_ = invert_result_->make_filter();
  }

  if (bitmap_filter_) {
    forward_bitmap_.reset();
    auto bitmap = build_forward_bitmap();
    if (!bitmap.ok()) {
      return Status::InternalError("Execute filter bitmap failed: ",
                                   bitmap.status().ToString());
    }
    forward_bitmap_ = std::move(bitmap).ValueUnsafe();
  }

  // A sealed IPC segment also supplies a lock-free deletion check. Only a
  // filter that exists now is replaced, so the delete bitmap observed at
  // construction still decides whether deletion is checked at all.
  const bool with_deletion = delete_filter_ != nullptr;
  if (forward_filter_expr_) {
    // get schema to bind to Expression
    const auto columns = query_info_->get_forward_filter_field_names();
    auto prepared = segment_->prepare_forward_read(columns, with_deletion);
    if (!prepared.ok()) {
      return Status::InternalError("Prepare forward read failed: ",
                                   prepared.status().ToString());
    }
    auto state = prepared.MoveValueUnsafe();
    forward_rows_ = state.num_rows;
    forward_view_ = std::move(state.view);
    sealed_deletion_ = std::move(state.deletion_filter);
    const auto &schema = state.schema;
    // Log outside segment synchronization.
    if (forward_view_) {
      LOG_DEBUG("Forward filter view: segment[%d] rows[%zu] columns[%zu]",
                segment_->id(), static_cast<size_t>(forward_rows_),
                columns.size());
    } else {
      LOG_DEBUG("Forward filter locked fetch: segment[%d] rows[%zu]",
                segment_->id(), static_cast<size_t>(forward_rows_));
    }
    auto bind_res = forward_filter_expr_->Bind(*schema);
    if (!bind_res.ok()) {
      return Status::InternalError("Bind forward filter expression failed",
                                   bind_res.status().ToString());
    }
    *forward_filter_expr_ = bind_res.MoveValueUnsafe();
    forward_prepared_ = true;
  } else if (with_deletion) {
    // Without a forward predicate the capture is only an optimization: on
    // any failure keep the segment's filter, which gives the same results.
    auto prepared = segment_->prepare_forward_read({}, true);
    if (prepared.ok()) {
      sealed_deletion_ = std::move(prepared->deletion_filter);
    }
  }
  return Status::OK();
}

bool DocFilter::empty() const {
  return !(delete_filter_ || invert_filter_ || bitmap_filter_ ||
           forward_filter_expr_);
}

bool DocFilter::is_filtered(uint64_t id) const {
  // An unprepared (or failed) forward filter never admits candidates.
  if (forward_filter_expr_ && !forward_prepared_) return true;
  // An owning view holds exactly the rows captured at preparation; nothing
  // beyond them can be read. Locked fetch keeps reading the live segment.
  if (forward_view_ && id >= forward_rows_) return true;
  if (delete_filter_ &&
      (sealed_deletion_ ? sealed_deletion_ : delete_filter_)->is_filtered(id)) {
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
  ExecBatchPtr exec_batch;
  if (forward_view_) {
    exec_batch = forward_view_->read_row(id);
  } else {
    const auto &columns = query_info_->get_forward_filter_field_names();
    // Literal predicates need a one-row batch but no scalar storage access.
    exec_batch = columns.empty() ? std::make_shared<arrow::compute::ExecBatch>(
                                       std::vector<arrow::Datum>{}, 1)
                                 : segment_->fetch(columns, id);
  }
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
