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

#include "query_parser.h"

namespace zvec::sqlengine {

SQLInfo::Ptr QueryParser::parse(const std::string &query) {
  ZVecSQLParser se_sql_parser_;

  SQLInfo::Ptr sql_info = se_sql_parser_.parse(query);
  if (sql_info == nullptr) {
    err_msg_ = se_sql_parser_.err_msg();
    return nullptr;
  }

  return sql_info;
}

}  // namespace zvec::sqlengine
