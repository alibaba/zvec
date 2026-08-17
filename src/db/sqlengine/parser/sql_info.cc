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

#include "sql_info.h"
#include <memory>
#include "db/sqlengine/common/util.h"
#include "select_info.h"


namespace zvec::sqlengine {

SQLInfo::SQLInfo(SQLType m_type, BaseInfo::Ptr m_base_info) {
  type_ = m_type;
  base_info_ = m_base_info;
}

SQLInfo::~SQLInfo() = default;

SQLInfo::SQLInfo(const SQLInfo &info) {
  type_ = info.type_;
  if (type_ == SQLType::SELECT) {
    SelectInfo::Ptr select_info = std::make_shared<SelectInfo>(
        *(std::dynamic_pointer_cast<SelectInfo>(info.base_info_)));
    base_info_ = select_info;
  } else {
    base_info_ = nullptr;
  }
}

SQLInfo &SQLInfo::operator=(const SQLInfo &info) {
  type_ = info.type_;
  if (type_ == SQLType::SELECT) {
    SelectInfo::Ptr select_info = std::make_shared<SelectInfo>(
        *(std::dynamic_pointer_cast<SelectInfo>(info.base_info_)));
    base_info_ = select_info;
  } else {
    base_info_ = nullptr;
  }

  return *this;
}

void SQLInfo::set_base_info(BaseInfo::Ptr value) {
  base_info_ = std::move(value);
}

const BaseInfo::Ptr &SQLInfo::base_info() const {
  return base_info_;
}

void SQLInfo::set_type(SQLType value) {
  type_ = value;
}

SQLInfo::SQLType SQLInfo::type() const {
  return type_;
}

std::string SQLInfo::type_name() const {
  return type_to_str(type_);
}

std::string SQLInfo::to_string() {
  std::string str = "SQL Info: {\n";
  str += "Type: " + type_name();
  str += "\n";
  str += "Info:";
  str += "\n";
  str += base_info_->to_string();
  str += "}";
  return str;
}

}  // namespace zvec::sqlengine
