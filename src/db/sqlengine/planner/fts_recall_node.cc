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

#include "db/sqlengine/planner/fts_recall_node.h"
#include <arrow/api.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/db/config.h>
#include "db/sqlengine/common/util.h"

namespace cp = arrow::compute;

namespace zvec::sqlengine {

FtsRecallNode::FtsRecallNode(Segment::Ptr segment, QueryInfo::Ptr query_info,
                             DocFilter::Ptr doc_filter, int batch_size)
    : segment_(std::move(segment)),
      query_info_(std::move(query_info)),
      doc_filter_(std::move(doc_filter)),
      fetched_columns_(query_info_->get_all_fetched_scalar_field_names()),
      batch_size_(batch_size) {
  auto table = segment_->fetch(fetched_columns_, std::vector<int>{});
  // Append BM25 score column so downstream fill_doc_score() surfaces it to
  // the Python Doc.score, matching the vector-recall path.
  schema_ = Util::append_field(*table->schema(), kFieldScore, arrow::float32());
}

arrow::AsyncGenerator<std::optional<cp::ExecBatch>> FtsRecallNode::gen() {
  auto state_ptr = std::make_shared<State>();
  return [self = shared_from_this(), state_ptr = std::move(state_ptr)]()
             -> arrow::Future<std::optional<cp::ExecBatch>> {
    auto &state = *state_ptr;

    if (!state.iter_) {
      auto fts_ret = self->prepare();
      if (!fts_ret) {
        return arrow::Future<std::optional<cp::ExecBatch>>::MakeFinished(
            arrow::Status::ExecutionError("prepare fts failed:",
                                          fts_ret.error().c_str()));
      }
      state.fts_result_ = fts_ret.value();
      state.iter_ = state.fts_result_->create_iterator();
    }

    if (!state.iter_->valid()) {
      return arrow::Future<std::optional<cp::ExecBatch>>::MakeFinished(
          std::nullopt);
    }

    std::vector<int> indices;
    indices.reserve(self->batch_size_);
    arrow::FloatBuilder score_builder;
    for (int i = 0; state.iter_->valid() && i < self->batch_size_;
         i++, state.iter_->next()) {
      indices.push_back(state.iter_->doc_id());
      auto s = score_builder.Append(state.iter_->score());
      if (!s.ok()) {
        return arrow::Future<std::optional<cp::ExecBatch>>::MakeFinished(
            arrow::Status::ExecutionError("score builder append failed:",
                                          s.ToString()));
      }
    }
    if (indices.empty()) {
      return arrow::Future<std::optional<cp::ExecBatch>>::MakeFinished(
          std::nullopt);
    }

    auto table = self->segment_->fetch(self->fetched_columns_, indices);
    if (!table) {
      return arrow::Future<std::optional<cp::ExecBatch>>::MakeFinished(
          arrow::Status::UnknownError("fetch table failed"));
    }
    auto batch = table->CombineChunksToBatch();
    if (!batch.ok()) {
      return arrow::Future<std::optional<cp::ExecBatch>>::MakeFinished(
          arrow::Status::ExecutionError("combine chunks to batch failed:",
                                        batch.status().ToString()));
    }
    auto score_array = score_builder.Finish();
    if (!score_array.ok()) {
      return arrow::Future<std::optional<cp::ExecBatch>>::MakeFinished(
          arrow::Status::ExecutionError("finish score builder failed:",
                                        score_array.status().ToString()));
    }
    auto record_batch = std::move(batch.ValueUnsafe());
    auto with_score =
        record_batch->AddColumn(record_batch->num_columns(), kFieldScore,
                                score_array.MoveValueUnsafe());
    if (!with_score.ok()) {
      return arrow::Future<std::optional<cp::ExecBatch>>::MakeFinished(
          arrow::Status::ExecutionError("add score column failed:",
                                        with_score.status().ToString()));
    }
    cp::ExecBatch exec_batch(*with_score.ValueUnsafe());
    return arrow::Future<std::optional<cp::ExecBatch>>::MakeFinished(
        std::move(exec_batch));
  };
}

Result<FtsIndexResults::Ptr> FtsRecallNode::prepare() {
  auto filter_status = doc_filter_->compute_filter();
  if (!filter_status.ok()) {
    return tl::make_unexpected(filter_status);
  }

  const auto &fts_cond = query_info_->fts_cond_info();
  if (!fts_cond) {
    return tl::make_unexpected(
        Status::InvalidArgument("FtsRecallNode: no fts_cond_info in query"));
  }

  fts::FtsQueryParams params;
  params.topk = query_info_->query_topn();
  // Brute-force path: get_bf_by_keys_and_update also clears invert_filter_
  // when it returns ids, so the filter set below won't double-check them.
  if (auto bf_keys = doc_filter_->get_bf_by_keys_and_update(
          GlobalConfig::Instance().fts_brute_force_by_keys_ratio())) {
    params.candidate_ids = std::move(bf_keys.value());
  }
  // Push down remaining filters (delete / forward) so filtered docs are
  // skipped during scoring and we still return up to topk results.
  params.filter = doc_filter_->empty() ? nullptr : doc_filter_;

  auto results =
      segment_->fts_search(fts_cond->field_name, *fts_cond->fts_ast, params);
  if (!results) {
    return tl::make_unexpected(results.error());
  }

  return std::make_shared<FtsIndexResults>(std::move(results.value()));
}

}  // namespace zvec::sqlengine
