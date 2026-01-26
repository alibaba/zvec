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
#include <string>
#include <zvec/ailego/pattern/expected.hpp>
#include <zvec/ailego/utility/string_helper.h>

namespace zvec {

class Status;
template <typename T>
using Result = tl::expected<T, Status>;

std::ostream &operator<<(std::ostream &os, const Status &s);

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
const char *GetDefaultMessage(StatusCode code);

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
class Status {
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
    return ailego::StringHelper::Concat(std::forward<Args>(args)...);
  }

  StatusCode code_;
  std::string msg_;
};

}  // namespace zvec