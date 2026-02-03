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
#include <vector>

namespace zvec {
namespace ailego {
namespace details {

//! Convert string to integers or floating point numbers
template <typename T>
static T CStringToType(const char *begin, char **endptr) {
  static_assert(
      std::is_same<T, int32_t>::value || std::is_same<T, int16_t>::value ||
          std::is_same<T, int8_t>::value || std::is_same<T, int64_t>::value ||
          std::is_same<T, uint64_t>::value ||
          std::is_same<T, uint32_t>::value ||
          std::is_same<T, uint16_t>::value || std::is_same<T, uint8_t>::value ||
          std::is_same<T, float>::value || std::is_same<T, double>::value,
      "type not supported");
  if (std::is_same<T, int32_t>::value || std::is_same<T, int16_t>::value ||
      std::is_same<T, int8_t>::value) {
    return static_cast<T>(strtol(begin, endptr, 0));
  } else if (std::is_same<T, int64_t>::value) {
    return static_cast<T>(strtoll(begin, endptr, 0));
  } else if (std::is_same<T, uint32_t>::value ||
             std::is_same<T, uint16_t>::value ||
             std::is_same<T, uint8_t>::value) {
    return static_cast<T>(strtoul(begin, endptr, 0));
  } else if (std::is_same<T, uint64_t>::value) {
    return static_cast<T>(strtoull(begin, endptr, 0));
  } else if (std::is_same<T, float>::value) {
    return static_cast<T>(strtof(begin, endptr));
  } else {
    return static_cast<T>(strtod(begin, endptr));
  }
}

//! Convert [begin, end) to T
//! If [end, ) contains valid T symbol, extra overhead will be incurred by
//! constructing std::string
template <typename T>
struct StringToType {
  T operator()(const char *begin, const char *end) {
    char *eptr = nullptr;
    auto v = CStringToType<T>(begin, &eptr);
    if (eptr > end) {
      // NOTE: [begin, end) is not 0 terminated
      // If delimiter contains valid T symbol, eptr might point to location
      // after end.
      // We create string here, which is guaranteed to be 0 terminated.
      std::string s{begin, end};
      return CStringToType<T>(s.c_str(), &eptr);
    }
    return v;
  }
};

//! Specialization for std::string
template <>
struct StringToType<std::string> {
  std::string operator()(const char *begin, const char *end) {
    return {begin, end};
  }
};

//! Return delimiter length.
template <typename T>
struct DelimiterLen {
  size_t operator()(T delimiter);
};

//! Return delimiter length for char.
template <>
struct DelimiterLen<char> {
  size_t operator()(char) {
    return 1;
  }
};

//! Return delimiter length for const char*.
template <>
struct DelimiterLen<const char *> {
  size_t operator()(const char *delimiter) {
    return delimiter == nullptr ? 0 : std::strlen(delimiter);
  }
};

//! Return delimiter length for std::string.
template <>
struct DelimiterLen<const std::string &> {
  size_t operator()(const std::string &delimiter) {
    return delimiter.size();
  }
};

//! Split implementation.
template <typename D, typename T,
          typename = typename std::enable_if<
              std::is_same<char, D>::value ||
                  std::is_same<const std::string &, D>::value ||
                  std::is_same<const char *, D>::value,
              D>::type>
static void SplitImpl(const std::string &str, D delim, std::vector<T> *out) {
  StringToType<T> func;
  out->clear();

  auto s = str.data();
  size_t delimiter_len = DelimiterLen<D>()(delim);
  if (delimiter_len != 0) {
    size_t a = 0, b = str.find(delim);
    while (b != std::string::npos) {
      out->push_back(func(s + a, s + b));
      a = b + delimiter_len;
      b = str.find(delim, a);
    }
    out->push_back(func(s + a, s + str.length()));
  } else {
    out->push_back(func(s + 0, s + str.length()));
  }
}

//! Split implementation.
template <typename D, typename T,
          typename = typename std::enable_if<
              std::is_same<char, D>::value ||
                  std::is_same<const std::string &, D>::value ||
                  std::is_same<const char *, D>::value,
              D>::type>
static void SplitImpl(const std::string &str, D delim, std::vector<T> *out,
                      bool skip_empty) {
  StringToType<T> func;
  out->clear();

  auto s = str.data();
  size_t delimiter_len = DelimiterLen<D>()(delim);
  if (delimiter_len != 0) {
    size_t a = 0, b = str.find(delim);
    while (b != std::string::npos) {
      if (!skip_empty || b - a > 0) {
        out->push_back(func(s + a, s + b));
      }

      a = b + delimiter_len;
      b = str.find(delim, a);
    }
    if (!skip_empty || str.length() - a > 0) {
      out->push_back(func(s + a, s + str.length()));
    }
  } else {
    if (!skip_empty || str.length() > 0) {
      out->push_back(func(s + 0, s + str.length()));
    }
  }
}
}  // namespace details
}  // namespace ailego
}  // namespace zvec