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
// limitations under the License

#pragma once

#include <memory.h>
#include <memory>
#include <vector>
#include <arrow/api.h>
#include <zvec/db/doc.h>
#include <zvec/db/schema.h>
#include "analyzer/query_info.h"
#include "common/group_by.h"
#include "db/sqlengine/common/util.h"
#include "db/sqlengine/parser/sql_info.h"
#include "db/sqlengine/sqlengine.h"

namespace zvec::sqlengine {

class SQLEngineImpl : public SQLEngine {
 public:
  SQLEngineImpl(zvec::Profiler::Ptr profiler);

  //! Parse pb request
  Result<QueryInfo::Ptr> parse_request(CollectionSchema::Ptr collection,
                                       const VectorQuery &request,
                                       std::shared_ptr<GroupBy> group_by);

  //! Perform search with given query_info, segments and index filter
  Result<std::unique_ptr<arrow::RecordBatchReader>> search_by_query_info(
      CollectionSchema::Ptr collection,
      const std::vector<Segment::Ptr> &segments,
      std::vector<sqlengine::QueryInfo::Ptr> *query_infos);

  Result<DocPtrList> execute(
      CollectionSchema::Ptr collection, const VectorQuery &query,
      const std::vector<Segment::Ptr> &segments) override;

  Result<GroupResults> execute_group_by(
      CollectionSchema::Ptr collection,
      const GroupByVectorQuery &group_by_query,
      const std::vector<Segment::Ptr> &segments) override;

  const std::string &execution_time_info() {
    return execution_time_info_;
  }

 private:
  Result<DocPtrList> fill_result(
      const std::vector<FieldAndSchema> &output_fields,
      arrow::RecordBatchReader *reader);

  Result<QueryInfo::Ptr> parse_sql_info(const CollectionSchema &schema,
                                        const SQLInfo::Ptr &sql_info);

  Result<GroupResults> fill_group_by_result(const QueryInfo &query_info,
                                            arrow::RecordBatchReader *reader);

 private:
  zvec::Profiler::Ptr profiler_;
  std::string execution_time_info_{};
};

}  // namespace zvec::sqlengine