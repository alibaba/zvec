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

namespace zvec {
namespace core {

/*! Index Filter
 */
class IndexFilter {
 public:
  //! Constructor
  IndexFilter(void) {}

  //! Constructor
  IndexFilter(const IndexFilter &rhs) : filter_(rhs.filter_) {}

  //! Constructor
  IndexFilter(IndexFilter &&rhs)
      : filter_(std::forward<decltype(filter_)>(rhs.filter_)) {}

  //! Copy assignment operator
  IndexFilter &operator=(const IndexFilter &rhs) {
    filter_ = rhs.filter_;
    return *this;
  }

  //! Copy assignment operator
  IndexFilter &operator=(IndexFilter &&rhs) {
    filter_ = std::forward<decltype(filter_)>(rhs.filter_);
    return *this;
  }

  //! Function call
  bool operator()(uint64_t key) const {
    return (filter_ ? filter_(key) : false);
  }

  //! Set the filter function
  template <typename T>
  void set(T &&func) {
    filter_ = std::forward<T>(func);
  }

  //! Reset the filter function
  void reset(void) {
    filter_ = nullptr;
  }

  //! Test if the function is valid
  bool is_valid(void) const {
    return (!!filter_);
  }

 private:
  //! Members
  std::function<bool(uint64_t key)> filter_{};
};

}  // namespace core
}  // namespace zvec
