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

#include <zvec/db/doc.h>
#include "db/sqlengine/common/group_by.h"
#include "db/sqlengine/parser/node.h"
#include "db/sqlengine/parser/sql_info.h"

namespace zvec::sqlengine {

class SQLInfoHelper {
 public:
  //! Perform QueryRequest to sql info conversion:
  static bool MessageToSQLInfo(const VectorQuery *query, Node::Ptr filter_node,
                               std::shared_ptr<GroupBy> group_by,
                               sqlengine::SQLInfo::Ptr *sql_info,
                               std::string *err_msg);
};

}  // namespace zvec::sqlengine
