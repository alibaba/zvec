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

#include "db/sqlengine/planner/invert_recall_node.h"
#include <arrow/api.h>
#include <zvec/ailego/logger/logger.h>
#include "db/sqlengine/planner/invert_search.h"

namespace cp = arrow::compute;

namespace zvec::sqlengine {

arrow::AsyncGenerator<std::optional<cp::ExecBatch>> InvertRecallNode::gen() {
  auto state_ptr = std::make_shared<State>();
  return [self = shared_from_this(), state_ptr = std::move(state_ptr)]()
             -> arrow::Future<std::optional<cp::ExecBatch>> {
    auto &state = *state_ptr;

    if (!state.iter_) {
      auto invert_ret = self->prepare();
      if (!invert_ret) {
        return arrow::Future<std::optional<cp::ExecBatch>>::MakeFinished(
            arrow::Status::ExecutionError("prepare invert failed:",
                                          invert_ret.error().c_str()));
      }
      state.invert_result_ = invert_ret.value();
      state.iter_ = state.invert_result_->create_iterator();
    }

    if (!state.iter_->valid()) {
      // return nullopt to indicate end
      return arrow::Future<std::optional<cp::ExecBatch>>::MakeFinished(
          std::nullopt);
    }

    // collect a batch
    std::vector<int> indices;
    indices.reserve(self->batch_size_);
    for (int i = 0; state.iter_->valid() && i < self->batch_size_;
         state.iter_->next()) {
      if (self->seg_filter_ &&
          self->seg_filter_->is_filtered(state.iter_->doc_id())) {
        continue;
      }
      i++;
      indices.push_back(state.iter_->doc_id());
    }
    if (indices.empty()) {
      // return nullopt to indicate end
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
    cp::ExecBatch exec_batch(*batch.ValueUnsafe());
    return arrow::Future<std::optional<cp::ExecBatch>>::MakeFinished(
        std::move(exec_batch));
  };
}

Result<InvertedSearchResult::Ptr> InvertRecallNode::prepare() {
  InvertSearch search(segment_.get());
  return search.exec_invert_cond_tree(query_info_->invert_cond().get());
}

}  // namespace zvec::sqlengine