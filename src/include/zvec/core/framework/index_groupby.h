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

#include <functional>
#include <string>

namespace zvec {
namespace core {

/*! Index GroupBy
 */
class IndexGroupBy {
 public:
  //! Function call
  std::string operator()(uint64_t key) const {
    return (group_by_ ? group_by_(key) : "");
  }

  //! Set the group by function
  template <typename T>
  void set(T &&func) {
    group_by_ = std::forward<T>(func);
  }

  //! Reset the group by function
  void reset(void) {
    group_by_ = nullptr;
  }

  //! Test if the function is valid
  bool is_valid(void) const {
    return (!!group_by_);
  }

 private:
  //! Members
  std::function<std::string(uint64_t key)> group_by_{};
};

}  // namespace core
}  // namespace zvec