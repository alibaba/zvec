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

#include <cstdlib>
#include <vector>
#include <zvec/ailego/string/string_view.h>

namespace zvec {
namespace ailego {
namespace internal {

//! Helper class to convert integer and float types to string, facilitating
//! string concatenation memory allocation.
class Alphameric {
 public:
  //! Deals with int, int8_t, int16_t, int32_t, bool, short, signed char, non
  //! class enum
  Alphameric(int n)
      : view_(buffer_, snprintf(buffer_, sizeof(buffer_), "%d", n)) {}

  //! Deals with unsigned int, uint8_t, uint16_t, uint32_t, unsigned short,
  //! unsigned char
  Alphameric(unsigned int n)
      : view_(buffer_, snprintf(buffer_, sizeof(buffer_), "%u", n)) {}

  //! Deals with long, int32_t, int64_t
  Alphameric(long n)
      : view_(buffer_, snprintf(buffer_, sizeof(buffer_), "%ld", n)) {}

  //! Deals with unsigned long, uint32_t, uint64_t
  Alphameric(unsigned long n)
      : view_(buffer_, snprintf(buffer_, sizeof(buffer_), "%lu", n)) {}

  //! Deals with long long, int64_t
  Alphameric(long long n)
      : view_(buffer_, snprintf(buffer_, sizeof(buffer_), "%lld", n)) {}

  //! Deals with unsigned long long, uint64_t
  Alphameric(unsigned long long n)
      : view_(buffer_, snprintf(buffer_, sizeof(buffer_), "%llu", n)) {}

  //! Deals with float, with 6 precision digit the same as std::to_string
  Alphameric(float f)
      : view_(buffer_, snprintf(buffer_, sizeof(buffer_), "%g", f)) {}

  //! Deals with double, with 6 precision digit the same as std::to_string
  Alphameric(double f)
      : view_(buffer_, snprintf(buffer_, sizeof(buffer_), "%g", f)) {}

  //! Deals with long double, with 6 precision digit the same as std::to_string
  Alphameric(long double f)
      : view_(buffer_, snprintf(buffer_, sizeof(buffer_), "%Lg", f)) {}

  //! Deals with const char*
  Alphameric(const char *s) : view_(s) {}

  //! Deals with std::string
  Alphameric(const std::string &s) : view_(s) {}

  //! Deals with StringView
  Alphameric(StringView s) : view_(s) {}

  // Use string literals ":" instead of character literals ':'.
  Alphameric(char c) = delete;
  Alphameric(const Alphameric &) = delete;
  Alphameric &operator=(const Alphameric &) = delete;

  //! Deals with enum class with non int underlying type
  template <typename T,
            typename = typename std::enable_if<
                std::is_enum<T>{} && !std::is_convertible<T, int>{}>::type>
  Alphameric(T e)
      : Alphameric(static_cast<typename std::underlying_type<T>::type>(e)) {}

  //! Deals with std::vector<bool> subscript reference
  template <typename T,
            typename std::enable_if<
                std::is_class<T>::value &&
                (std::is_same<T, std::vector<bool>::reference>::value ||
                 std::is_same<T, std::vector<bool>::const_reference>::value)>::
                type * = nullptr>
  Alphameric(T e) : Alphameric(static_cast<bool>(e)) {}

  //! string size
  size_t size() const {
    return view_.size();
  }

  //! string data
  const char *data() const {
    return view_.data();
  }

  //! string view
  StringView view() const {
    return view_;
  }

 private:
  static constexpr int kBufferSize = 32;
  char buffer_[kBufferSize];
  StringView view_;
};

}  // namespace internal
}  // namespace ailego
}  // namespace zvec
