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

#include <cstdlib>
#include <limits>
#include <zvec/ailego/utility/env_helper.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace zvec {
namespace ailego {

#ifdef _WIN32
namespace {

bool utf8_to_wide(const std::string &name, std::wstring &result) {
  if (name.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return false;
  }
  const int size = static_cast<int>(name.size());
  const int length = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                         name.data(), size, nullptr, 0);
  if (length == 0) {
    return false;
  }
  result.resize(length);
  return MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, name.data(), size,
                             result.data(), length) == length;
}

bool wide_to_utf8(const std::wstring &value, std::string &result) {
  const int size = static_cast<int>(value.size());
  const int length =
      WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value.data(), size,
                          nullptr, 0, nullptr, nullptr);
  if (length == 0) {
    return false;
  }
  result.resize(length);
  return WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value.data(), size,
                             result.data(), length, nullptr, nullptr) == length;
}

}  // namespace
#endif

EnvHelper::Status EnvHelper::Get(const std::string &name, std::string *env) {
  if (env == nullptr) {
    return Status::kError;
  }
  if (name.empty() || name.find('=') != std::string::npos ||
      name.find('\0') != std::string::npos) {
    env->clear();
    return Status::kError;
  }
#ifdef _WIN32
  std::wstring wide_name;
  if (!utf8_to_wide(name, wide_name)) {
    env->clear();
    return Status::kError;
  }
  // Read the process environment directly: SDK hosts can update it without
  // updating this DLL's CRT environment, and narrow CRT values are not UTF-8.
  std::wstring value;
  for (;;) {
    SetLastError(ERROR_SUCCESS);
    const DWORD length = GetEnvironmentVariableW(
        wide_name.c_str(), value.data(), static_cast<DWORD>(value.size()));
    if (length == 0) {
      const DWORD error = GetLastError();
      env->clear();
      if (error == ERROR_ENVVAR_NOT_FOUND) {
        return Status::kNotExist;
      }
      return error == ERROR_SUCCESS ? Status::kOk : Status::kError;
    }
    if (length < value.size()) {
      value.resize(length);
      if (!wide_to_utf8(value, *env)) {
        env->clear();
        return Status::kError;
      }
      return Status::kOk;
    }
    // Retry if the value grew between the size query and the read.
    value.resize(length);
  }
#else
  const char *value = std::getenv(name.c_str());
  if (value == nullptr) {
    env->clear();
    return Status::kNotExist;
  }
  *env = value;
  return Status::kOk;
#endif
}

}  // namespace ailego
}  // namespace zvec
