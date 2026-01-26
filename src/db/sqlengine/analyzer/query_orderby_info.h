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
#include <zvec/db/schema.h>

namespace zvec::sqlengine {

class QueryOrderbyInfo {
 public:
  using Ptr = std::shared_ptr<QueryOrderbyInfo>;

  QueryOrderbyInfo();
  QueryOrderbyInfo(const std::string &m_field_name, bool m_desc);
  ~QueryOrderbyInfo() = default;

  void set_field_name(const std::string &value) {
    field_name_ = value;
  }

  const std::string &field_name() const {
    return field_name_;
  }

  void set_desc() {
    desc_ = true;
  }
  bool is_desc() const {
    return desc_;
  }

  void set_field_schema_ptr(const zvec::FieldSchema *field_schema_ptr) {
    field_schema_ptr_ = field_schema_ptr;
  }
  const zvec::FieldSchema *field_schema_ptr() {
    return field_schema_ptr_;
  }

  std::string to_string() const;

 private:
  std::string field_name_{""};
  bool desc_{false};

  const zvec::FieldSchema *field_schema_ptr_;
};

}  // namespace zvec::sqlengine
