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
#include <string>
#include <arrow/acero/exec_plan.h>
#include <arrow/compute/expression.h>
#include <zvec/ailego/pattern/expected.hpp>
#include <zvec/db/status.h>
#include "db/index/segment/segment.h"
#include "db/sqlengine/analyzer/query_info.h"
#include "plan_info.h"

namespace zvec::sqlengine {

class QueryPlanner {
 public:
  QueryPlanner(CollectionSchema *schema);

  Result<PlanInfo::Ptr> make_plan(
      const std::vector<Segment::Ptr> &segments, const std::string &trace_id,
      std::vector<sqlengine::QueryInfo::Ptr> *query_infos);


 private:
  Result<PlanInfo::Ptr> make_physical_plan(
      const std::vector<Segment::Ptr> &segments, const std::string &trace_id,
      std::vector<sqlengine::QueryInfo::Ptr> *query_infos);

  Result<PlanInfo::Ptr> make_group_by_physical_plan(
      const std::vector<Segment::Ptr> &segments, const std::string &trace_id,
      std::vector<sqlengine::QueryInfo::Ptr> *query_infos);

 private:
  Result<cp::Expression> parse_filter(const QueryNode *node);

  Result<cp::Expression> create_filter_node(const QueryNode *node);

  Result<PlanInfo::Ptr> vector_scan(
      Segment::Ptr seg, QueryInfo::Ptr query_info,
      std::unique_ptr<arrow::compute::Expression> forward_filter,
      bool single_stage_search);
  Result<PlanInfo::Ptr> invert_scan(
      Segment::Ptr seg, QueryInfo::Ptr query_info,
      std::unique_ptr<arrow::compute::Expression> forward_filter);
  Result<PlanInfo::Ptr> forward_scan(
      Segment::Ptr seg, QueryInfo::Ptr query_info,
      std::unique_ptr<arrow::compute::Expression> forward_filter);

  static int get_batch_size(const QueryInfo &info, bool has_later_filter);

 private:
  CollectionSchema *schema_{nullptr};
};

}  // namespace zvec::sqlengine
