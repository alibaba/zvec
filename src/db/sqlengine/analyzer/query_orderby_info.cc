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

#include "query_orderby_info.h"

namespace zvec::sqlengine {

QueryOrderbyInfo::QueryOrderbyInfo() = default;

QueryOrderbyInfo::QueryOrderbyInfo(const std::string &m_field_name, bool m_desc)
    : field_name_(m_field_name), desc_(m_desc) {}


std::string QueryOrderbyInfo::to_string() const {
  std::string str = field_name_;
  str = str + " " + (desc_ ? "DESC" : "ASC");
  return str;
}

}  // namespace zvec::sqlengine
