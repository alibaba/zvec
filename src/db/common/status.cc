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

#include <unordered_map>
#include <zvec/ailego/utility/string_helper.h>
#include <zvec/db/status.h>

namespace zvec {

const char *GetDefaultMessage(StatusCode code) {
  static const std::unordered_map<StatusCode, const char *> kMessages = {
      {StatusCode::OK, "OK"},
      {StatusCode::NOT_FOUND, "Not found"},
      {StatusCode::ALREADY_EXISTS, "Already exists"},
      {StatusCode::INVALID_ARGUMENT, "Invalid argument"},
      {StatusCode::PERMISSION_DENIED, "Permission denied"},
      {StatusCode::FAILED_PRECONDITION, "Failed precondition"},
      {StatusCode::RESOURCE_EXHAUSTED, "Resource exhausted"},
      {StatusCode::UNAVAILABLE, "Unavailable"},
      {StatusCode::INTERNAL_ERROR, "Internal error"},
      {StatusCode::NOT_SUPPORTED, "Not supported"},
      {StatusCode::UNKNOWN, "Unknown error"}};
  auto it = kMessages.find(code);
  return it != kMessages.end() ? it->second : "Unknown status code";
}

// Implementation of operator<<
std::ostream &operator<<(std::ostream &os, const Status &s) {
  if (s.ok()) {
    os << "OK";
  } else {
    os << "Status(" << GetDefaultMessage(s.code()) << ", " << s.message()
       << ")";
  }
  return os;
}

// Implementation of comparison
bool Status::operator==(const Status &other) const noexcept {
  if (code_ != other.code_) return false;
  if (code_ == StatusCode::OK) return true;
  return msg_ == other.msg_;
}

}  // namespace zvec