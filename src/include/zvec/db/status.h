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
#include <cstdio>
#include <iosfwd>
#include <string>
#include <type_traits>
#include <zvec/ailego/pattern/expected.hpp>
#include <zvec/export.h>

namespace zvec {

namespace detail {

/// @name Status message formatting
/// @{
/// Appends one argument to a Status error message.  These overloads replace a
/// dependency on the internal `ailego::StringHelper::Concat`, keeping engine
/// headers out of the published API.  Formatting is deliberately identical to
/// what it replaced -- `std::to_string` for integers (which the standard
/// specifies as `%d`/`%u`/`%ld`/`%lu`/`%lld`/`%llu`) and `%g` for floating
/// point -- so existing message text, and the tests that assert on it, are
/// unaffected.

/// Buffer for one numeric conversion; matches the width the previous
/// implementation used.
constexpr int kStatusFormatBufferSize = 32;

inline void append_status(std::string &out, const char *text) {
  out.append(text);
}

inline void append_status(std::string &out, const std::string &text) {
  out.append(text);
}

template <typename T>
inline typename std::enable_if<std::is_integral<T>::value, void>::type
append_status(std::string &out, T value) {
  out.append(std::to_string(value));
}

template <typename... Args>
inline void append_status_formatted(std::string &out, const char *format,
                                    Args... args) {
  char buffer[kStatusFormatBufferSize];
  const int written = std::snprintf(buffer, sizeof(buffer), format, args...);
  if (written <= 0) {
    return;
  }
  const size_t length = written < kStatusFormatBufferSize
                            ? static_cast<size_t>(written)
                            : sizeof(buffer) - 1;
  out.append(buffer, length);
}

inline void append_status(std::string &out, float value) {
  append_status_formatted(out, "%g", static_cast<double>(value));
}

inline void append_status(std::string &out, double value) {
  append_status_formatted(out, "%g", value);
}

inline void append_status(std::string &out, long double value) {
  append_status_formatted(out, "%Lg", value);
}

/// Scoped enums, and unscoped enums that do not implicitly convert to int.
template <typename T>
inline typename std::enable_if<
    std::is_enum<T>::value && !std::is_convertible<T, int>::value, void>::type
append_status(std::string &out, T value) {
  append_status(out,
                static_cast<typename std::underlying_type<T>::type>(value));
}

/// Unscoped enums convert like the integral type they decay to.
template <typename T>
inline typename std::enable_if<
    std::is_enum<T>::value && std::is_convertible<T, int>::value, void>::type
append_status(std::string &out, T value) {
  append_status(out, static_cast<int>(value));
}

/// Character literals are ambiguous next to string literals; require `"x"`.
void append_status(std::string &out, char value) = delete;

/// Concatenates any number of arguments into a single string.
template <typename... Args>
inline std::string concat_status(Args &&...args) {
  std::string out;
  if constexpr (sizeof...(Args) > 0) {
    (append_status(out, std::forward<Args>(args)), ...);
  }
  return out;
}

/// @}

}  // namespace detail

class ZVEC_API Status;
template <typename T>
using Result = tl::expected<T, Status>;

ZVEC_API std::ostream &operator<<(std::ostream &os, const Status &s);

/**
 * @brief Enumeration of common error codes.
 */
enum class StatusCode {
  OK = 0,
  NOT_FOUND,
  ALREADY_EXISTS,
  INVALID_ARGUMENT,
  PERMISSION_DENIED,
  FAILED_PRECONDITION,
  RESOURCE_EXHAUSTED,
  UNAVAILABLE,
  INTERNAL_ERROR,
  NOT_SUPPORTED,
  UNKNOWN
};

// Helper: get default message for code
ZVEC_API const char *GetDefaultMessage(StatusCode code);

/**
 * @class Status
 * @brief Represents the result of an operation: success or failure with
 * message.
 *
 * This class is used to return error information from functions without using
 * exceptions. It stores a status code and an optional error message.
 *
 * @note This class is thread-compatible: const methods can be called from
 * multiple threads.
 */
class ZVEC_API Status {
 public:
  /// @brief Default constructor: OK status
  Status() noexcept : code_(StatusCode::OK) {}

  /// @brief Construct a failed status with code and message
  Status(StatusCode code, const std::string &msg) : code_(code), msg_(msg) {
    ensure_not_ok(code);
  }

  /// @brief Construct a failed status with code and rvalue message
  Status(StatusCode code, std::string &&msg)
      : code_(code), msg_(std::move(msg)) {
    ensure_not_ok(code);
  }

  /// @brief Copy constructor
  Status(const Status &) = default;

  /// @brief Copy assignment
  Status &operator=(const Status &) = default;

  /// @brief Move constructor
  Status(Status &&) = default;

  /// @brief Move assignment
  Status &operator=(Status &&) = default;

  /// @brief Destructor
  ~Status() = default;

  /// @brief Check if the status is OK (no error)
  bool ok() const noexcept {
    return code_ == StatusCode::OK;
  }

  /// @brief Get the status code
  StatusCode code() const noexcept {
    return code_;
  }

  /// @brief Get the error message (empty if OK)
  const std::string &message() const noexcept {
    return msg_;
  }

  /// @brief Get C-style string (safe because msg_ owns the string)
  const char *c_str() const noexcept {
    return msg_.c_str();
  }

  /// @brief Comparison operators
  bool operator==(const Status &other) const noexcept;
  bool operator!=(const Status &other) const noexcept {
    return !(*this == other);
  }

  /// @brief Factory: Success
  static Status OK() noexcept {
    return Status();
  }

  /// @brief Factory: Invalid argument
  template <typename... Args>
  static Status InvalidArgument(Args &&...args) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  concat(std::forward<Args>(args)...));
  }

  /// @brief Factory: Not found
  template <typename... Args>
  static Status NotFound(Args &&...args) {
    return Status(StatusCode::NOT_FOUND, concat(std::forward<Args>(args)...));
  }

  /// @brief Factory: Already exists
  template <typename... Args>
  static Status AlreadyExists(Args &&...args) {
    return Status(StatusCode::ALREADY_EXISTS,
                  concat(std::forward<Args>(args)...));
  }

  /// @brief Factory: Internal error
  template <typename... Args>
  static Status InternalError(Args &&...args) {
    return Status(StatusCode::INTERNAL_ERROR,
                  concat(std::forward<Args>(args)...));
  }

  /// @brief Factory: Permission denied
  template <typename... Args>
  static Status PermissionDenied(Args &&...args) {
    return Status(StatusCode::PERMISSION_DENIED,
                  concat(std::forward<Args>(args)...));
  }

  /// @brief Factory: Failed precondition
  template <typename... Args>
  static Status FailedPrecondition(Args &&...args) {
    return Status(StatusCode::FAILED_PRECONDITION,
                  concat(std::forward<Args>(args)...));
  }

  /// @brief Factory: Not supported
  template <typename... Args>
  static Status NotSupported(Args &&...args) {
    return Status(StatusCode::NOT_SUPPORTED,
                  concat(std::forward<Args>(args)...));
  }

  // Add more factories as needed...

 private:
  /// @brief Ensure non-OK status has non-empty message (optional)
  static void ensure_not_ok(StatusCode /*code*/) noexcept {
    // Optional: assert(code == StatusCode::OK || "non-OK status should have
    // message")
  }

  /// @brief Helper: concatenate any number of arguments into a string
  template <typename... Args>
  static std::string concat(Args &&...args) {
    return detail::concat_status(std::forward<Args>(args)...);
  }

  StatusCode code_;
  std::string msg_;
};

}  // namespace zvec
