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
#include "db/index/column/common/index_results.h"
#include "db/index/column/fts_column/fts_index_results.h"
#include "db/index/segment/segment.h"
#include "db/sqlengine/analyzer/query_info.h"
#include "db/sqlengine/planner/doc_filter.h"

namespace cp = arrow::compute;

namespace zvec::sqlengine {

class FtsRecallNode : public std::enable_shared_from_this<FtsRecallNode> {
 public:
  FtsRecallNode(Segment::Ptr segment, QueryInfo::Ptr query_info,
                DocFilter::Ptr doc_filter, int batch_size);

  //! get schema
  std::shared_ptr<arrow::Schema> schema() const {
    return schema_;
  }

  arrow::AsyncGenerator<std::optional<cp::ExecBatch>> gen();

 private:
  Result<FtsIndexResults::Ptr> prepare();

 private:
  struct State {
    FtsIndexResults::Ptr fts_result_;
    IndexResults::IteratorUPtr iter_;
  };

  Segment::Ptr segment_;
  QueryInfo::Ptr query_info_;
  DocFilter::Ptr doc_filter_;
  const std::vector<std::string> &fetched_columns_;
  int batch_size_;
  std::shared_ptr<arrow::Schema> schema_;
};

}  // namespace zvec::sqlengine
