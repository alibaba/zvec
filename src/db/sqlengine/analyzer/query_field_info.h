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

class QueryFieldInfo {
 public:
  using Ptr = std::shared_ptr<QueryFieldInfo>;

  QueryFieldInfo() {}

  QueryFieldInfo(const std::string &m_field_name, const std::string &m_alias,
                 const std::string &m_func_name,
                 const std::string &m_func_param, bool m_func_param_asterisk)
      : field_name_(m_field_name),
        alias_(m_alias),
        func_name_(m_func_name),
        func_param_(m_func_param),
        func_param_asterisk_(m_func_param_asterisk) {}

  ~QueryFieldInfo() {}

  void set_field_name(const std::string &value) {
    field_name_ = value;
  }

  const std::string &field_name() const {
    return field_name_;
  }

  void set_alias(const std::string &value) {
    alias_ = value;
  }
  const std::string &alias() const {
    return alias_;
  }

  const std::string &func_name() const {
    return func_name_;
  }

  void set_func_name(const std::string &value) {
    func_name_ = value;
  }

  const std::string &func_param() const {
    return func_param_;
  }

  void set_func_param(const std::string &value) {
    func_param_ = value;
  }

  bool is_func_call() const {
    return (!func_name_.empty());
  }

  void set_func_param_asterisk(bool value) {
    func_param_asterisk_ = value;
  }
  bool is_func_param_asterisk() const {
    return func_param_asterisk_;
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
  std::string alias_{""};

  std::string func_name_{""};
  std::string func_param_{""};
  bool func_param_asterisk_{false};

  const zvec::FieldSchema *field_schema_ptr_;
};

}  // namespace zvec::sqlengine
