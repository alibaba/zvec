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

#include <cstdint>
#include <string>
#include <vector>
#include <zvec/ailego/string/string_concat_helper.h>
#include <zvec/ailego/utility/string_helper_impl.h>

namespace zvec {
namespace ailego {

/*! String Helper
 */
struct StringHelper {
  //! Return true if the `ref` starts with the given prefix
  static bool StartsWith(const std::string &ref, const std::string &prefix);

  //! Return true if the `ref` ends with the given suffix
  static bool EndsWith(const std::string &ref, const std::string &suffix);

  //! Split a string into a vector of T
  //! NOTE: delim better NOT contain valid symbol for T,
  //!       i.e. digits + - for integers,
  //!            digits + - E e . for floating numbers
  //!       otherwise there will be performance overhead.
  template <typename T>
  static void Split(const std::string &str, char delim, std::vector<T> *out) {
    return details::SplitImpl<char, T>(str, delim, out);
  }
  template <typename T>
  static void Split(const std::string &str, const char *delim,
                    std::vector<T> *out) {
    return details::SplitImpl<const char *, T>(str, delim, out);
  }
  template <typename T>
  static void Split(const std::string &str, const std::string &delim,
                    std::vector<T> *out) {
    return details::SplitImpl<const std::string &, T>(str, delim, out);
  }

  template <typename T>
  static void Split(const std::string &str, char delim, std::vector<T> *out,
                    bool skip_empty) {
    return details::SplitImpl<char, T>(str, delim, out, skip_empty);
  }
  template <typename T>
  static void Split(const std::string &str, const char *delim,
                    std::vector<T> *out, bool skip_empty) {
    return details::SplitImpl<const char *, T>(str, delim, out, skip_empty);
  }
  template <typename T>
  static void Split(const std::string &str, const std::string &delim,
                    std::vector<T> *out, bool skip_empty) {
    return details::SplitImpl<const std::string &, T>(str, delim, out,
                                                      skip_empty);
  }

  // Trim from start (in place)
  static void LeftTrim(std::string &str);

  // Trim from end (in place)
  static void RightTrim(std::string &str);

  // Trim from both ends (in place)
  static void Trim(std::string &str);

  // Trim from start (copying)
  static std::string CopyLeftTrim(std::string str);

  // Trim from end (copying)
  static std::string CopyRightTrim(std::string str);

  // Trim from both ends (copying)
  static std::string CopyTrim(std::string str);

  //! Compare ignore case
  static bool CompareIgnoreCase(const std::string &a, const std::string &b);

  //! Convert string to floating-point number (double)
  static bool ToDouble(const std::string &str, double *val) {
    char *endptr = nullptr;
    *val = std::strtod(str.c_str(), &endptr);
    return (endptr && *endptr == '\0');
  }

  //! Convert string to floating-point number (float)
  static bool ToFloat(const std::string &str, float *val) {
    char *endptr = nullptr;
    *val = std::strtof(str.c_str(), &endptr);
    return (endptr && *endptr == '\0');
  }

  //! Convert string to integer number (int8_t)
  static bool ToInt8(const std::string &str, int8_t *val) {
    char *endptr = nullptr;
    *val = static_cast<int8_t>(std::strtol(str.c_str(), &endptr, 0));
    return (endptr && *endptr == '\0');
  }

  //! Convert string to integer number (int16_t)
  static bool ToInt16(const std::string &str, int16_t *val) {
    char *endptr = nullptr;
    *val = static_cast<int16_t>(std::strtol(str.c_str(), &endptr, 0));
    return (endptr && *endptr == '\0');
  }

  //! Convert string to integer number (int32_t)
  static bool ToInt32(const std::string &str, int32_t *val) {
    char *endptr = nullptr;
    *val = static_cast<int32_t>(std::strtol(str.c_str(), &endptr, 0));
    return (endptr && *endptr == '\0');
  }

  //! Convert string to integer number (int64_t)
  static bool ToInt64(const std::string &str, int64_t *val) {
    char *endptr = nullptr;
    *val = static_cast<int64_t>(std::strtoll(str.c_str(), &endptr, 0));
    return (endptr && *endptr == '\0');
  }

  //! Convert string to unsigned integer number (uint8_t)
  static bool ToUint8(const std::string &str, uint8_t *val) {
    char *endptr = nullptr;
    *val = static_cast<uint8_t>(std::strtoul(str.c_str(), &endptr, 0));
    return (endptr && *endptr == '\0');
  }

  //! Convert string to unsigned integer number (uint16_t)
  static bool ToUint16(const std::string &str, uint16_t *val) {
    char *endptr = nullptr;
    *val = static_cast<uint16_t>(std::strtoul(str.c_str(), &endptr, 0));
    return (endptr && *endptr == '\0');
  }

  //! Convert string to unsigned integer number (uint32_t)
  static bool ToUint32(const std::string &str, uint32_t *val) {
    char *endptr = nullptr;
    *val = static_cast<uint32_t>(std::strtoul(str.c_str(), &endptr, 0));
    return (endptr && *endptr == '\0');
  }

  //! Convert string to unsigned integer number (uint64_t)
  static bool ToUint64(const std::string &str, uint64_t *val) {
    char *endptr = nullptr;
    *val = static_cast<uint64_t>(std::strtoull(str.c_str(), &endptr, 0));
    return (endptr && *endptr == '\0');
  }

  //! Convert floating-point number string (double)
  static std::string ToString(double val) {
    return std::to_string(val);
  }

  //! Convert floating-point number string (float)
  static std::string ToString(float val) {
    return std::to_string(val);
  }

  //! Convert integer number to string (int8_t)
  static std::string ToString(int8_t val) {
    return std::to_string(val);
  }

  //! Convert integer number to string (int16_t)
  static std::string ToString(int16_t val) {
    return std::to_string(val);
  }

  //! Convert integer number to string (int32_t)
  static std::string ToString(int32_t val) {
    return std::to_string(val);
  }

  //! Convert integer number to string (int64_t)
  static std::string ToString(int64_t val) {
    return std::to_string(val);
  }

  //! Convert unsigned integer number to string (uint8_t)
  static std::string ToString(uint8_t val) {
    return std::to_string(val);
  }

  //! Convert unsigned integer number to string (uint16_t)
  static std::string ToString(uint16_t val) {
    return std::to_string(val);
  }

  //! Convert unsigned integer number to string (uint32_t)
  static std::string ToString(uint32_t val) {
    return std::to_string(val);
  }

  //! Convert unsigned integer number to string (uint64_t)
  static std::string ToString(uint64_t val) {
    return std::to_string(val);
  }

  //! Concatenation of arbitrary number of std::string, c-string, integers,
  //! floating point numbers with one memory allocation.
  //! E.g. auto s = Concat("foo", 123, std::string("bar"), 3.14159);
  //!
  //! Do not do the following, use Append instead
  //! str = Concat(str, ...);
  //! str.append(Concat(str, ...));
  //! str += Concat(str, ...);
  //!
  //! NOTE: char literal(e.g. ':') is not allowed,
  //! use string literal(e.g. ":") instead.
  static std::string Concat() {
    return {};
  }
  static std::string Concat(const internal::Alphameric &a);
  static std::string Concat(const internal::Alphameric &a,
                            const internal::Alphameric &b);
  static std::string Concat(const internal::Alphameric &a,
                            const internal::Alphameric &b,
                            const internal::Alphameric &c);
  static std::string Concat(const internal::Alphameric &a,
                            const internal::Alphameric &b,
                            const internal::Alphameric &c,
                            const internal::Alphameric &d);
  // Support 5 or more arguments
  template <typename... T>
  static std::string Concat(const internal::Alphameric &a,
                            const internal::Alphameric &b,
                            const internal::Alphameric &c,
                            const internal::Alphameric &d,
                            const internal::Alphameric &e, const T &...args) {
    std::string result;
    Append(&result, a, b, c, d, e, args...);
    return result;
  }

  //! Append arbitrary number of std::string, c-string, integers,
  //! floating point numbers to existing string with one memory allocation.
  //! E.g. Append(&str, "foo", 123, std::string("bar"), 3.14159);
  //!
  //! WARNING: Append requires that none of the arguments be a reference to
  //! destination str.
  //!
  //! Do not do the following
  //! std::string s = "foo";
  //! Append(&s, s);
  //!
  //! NOTE: char literal(e.g. ':') is not allowed,
  //! use string literal(e.g. ":") instead.
  static void Append(std::string *) {}
  static void Append(std::string *str, const internal::Alphameric &a);
  static void Append(std::string *str, const internal::Alphameric &a,
                     const internal::Alphameric &b);
  static void Append(std::string *str, const internal::Alphameric &a,
                     const internal::Alphameric &b,
                     const internal::Alphameric &c);
  static void Append(std::string *str, const internal::Alphameric &a,
                     const internal::Alphameric &b,
                     const internal::Alphameric &c,
                     const internal::Alphameric &d);
  // Support 5 or more arguments
  template <typename... T>
  static void Append(std::string *str, const internal::Alphameric &a,
                     const internal::Alphameric &b,
                     const internal::Alphameric &c,
                     const internal::Alphameric &d,
                     const internal::Alphameric &e, const T &...args) {
    AppendViews(str,
                {a.view(), b.view(), c.view(), d.view(), e.view(),
                 static_cast<const internal::Alphameric &>(args).view()...});
  }

  //! Append list of StringView to str.
  static void AppendViews(std::string *str,
                          std::initializer_list<StringView> views);
};

inline std::string StringHelper::Concat(const internal::Alphameric &a) {
  std::string result;
  Append(&result, a);
  return result;
}

inline std::string StringHelper::Concat(const internal::Alphameric &a,
                                        const internal::Alphameric &b) {
  std::string result;
  Append(&result, a, b);
  return result;
}

inline std::string StringHelper::Concat(const internal::Alphameric &a,
                                        const internal::Alphameric &b,
                                        const internal::Alphameric &c) {
  std::string result;
  Append(&result, a, b, c);
  return result;
}

inline std::string StringHelper::Concat(const internal::Alphameric &a,
                                        const internal::Alphameric &b,
                                        const internal::Alphameric &c,
                                        const internal::Alphameric &d) {
  std::string result;
  Append(&result, a, b, c, d);
  return result;
}

}  // namespace ailego
}  // namespace zvec
