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

#include <iostream>
#include <memory>

namespace zvec::sqlengine {

class BaseInfo {
 public:
  using Ptr = std::shared_ptr<BaseInfo>;

  BaseInfo(const std::string &value) {
    table_name_ = value;
  }

  virtual ~BaseInfo() = default;

  BaseInfo(const BaseInfo &info) {
    table_name_ = info.table_name_;
  }

  BaseInfo &operator=(const BaseInfo &info) {
    table_name_ = info.table_name_;
    return *this;
  }

  std::string table_name() {
    return table_name_;
  }

  bool validate() {
    return true;
  }

  const std::string &err_msg() {
    return err_msg_;
  }

  void set_err_msg(const std::string &value) {
    err_msg_ = value;
  }

  virtual std::string to_string() = 0;

 private:
  std::string table_name_{""};
  std::string err_msg_{""};
};

}  // namespace zvec::sqlengine
