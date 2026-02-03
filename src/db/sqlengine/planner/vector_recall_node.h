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

#include <memory>
#include <arrow/acero/api.h>
#include <arrow/api.h>
#include <zvec/db/status.h>
#include "db/index/column/common/index_results.h"
#include "db/index/segment/segment.h"
#include "db/sqlengine/analyzer/query_info.h"
#include "db/sqlengine/planner/doc_filter.h"

namespace zvec::sqlengine {

class VectorRecallNode : public std::enable_shared_from_this<VectorRecallNode> {
 public:
  using Ptr = std::shared_ptr<VectorRecallNode>;
  VectorRecallNode(Segment::Ptr segment, QueryInfo::Ptr query_info,
                   DocFilter::Ptr doc_filter, int batch_size,
                   bool single_stage_search);

  //! get schema
  std::shared_ptr<arrow::Schema> schema() const {
    return schema_;
  }

  arrow::AsyncGenerator<std::optional<cp::ExecBatch>> gen();

  const QueryInfo::Ptr &query_info() const {
    return query_info_;
  }

 private:
  Result<IndexResults::Ptr> prepare();

 private:
  struct State {
    State(VectorRecallNode::Ptr self) : self_(std::move(self)) {}

    arrow::Result<std::shared_ptr<arrow::RecordBatch>> collect_batch();

    VectorRecallNode::Ptr self_;
    IndexResults::Ptr vector_result_;
    IndexResults::IteratorUPtr iter_;
  };

  Segment::Ptr segment_;
  QueryInfo::Ptr query_info_;
  DocFilter::Ptr doc_filter_;
  int batch_size_;
  const std::vector<std::string> &fetched_columns_;
  std::shared_ptr<arrow::Schema> schema_;
};

}  // namespace zvec::sqlengine