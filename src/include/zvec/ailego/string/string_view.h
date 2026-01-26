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

#include <cstring>
#include <string>

namespace zvec {
namespace ailego {

//! StringView provides a lightweight view into the string data provided by
//! a `std::string`, double-quoted string literal, character array, or even
//! another `StringView`.
//!
//! A `StringView` does *not* own the string to which it
//! points, and that data cannot be modified through the view.
class StringView {
 public:
  //! Default constructor
  StringView() = default;

  //! Construct from c-string
  StringView(const char *str)
      : data_(str), size_(str != nullptr ? strlen(str) : 0) {}

  //! Construct from [str, str+s)
  StringView(const char *str, size_t len) : data_(str), size_(len) {}

  //! Construct from std::string
  StringView(const std::string &str) : data_(str.data()), size_(str.size()) {}

  //! Retrieve data of string
  const char *data() const {
    return data_;
  }

  //! Retrieve size of string
  size_t size() const {
    return size_;
  }

  //! Retrieve non-zero if it is empty
  bool empty() const {
    return size_ == 0;
  }

 private:
  const char *data_{nullptr};
  size_t size_{0};
};

}  // namespace ailego
}  // namespace zvec
