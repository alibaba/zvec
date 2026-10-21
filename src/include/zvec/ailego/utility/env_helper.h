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

#include <string>
#include <zvec/export.h>

namespace zvec {
namespace ailego {

/*! Process Environment Helper
 */
struct ZVEC_AILEGO_API EnvHelper {
  enum class Status { kOk, kNotExist, kError };

  //! Retrieve a process environment variable by its UTF-8 name.
  //! Returns kOk and writes the value to env, including an empty value.
  //! Returns kNotExist if unset, or kError for invalid arguments, lookup
  //! failures, or encoding failures. Clears env on kNotExist or kError.
  //! env must not be null; name must not be empty or contain '=' or NUL.
  //! Windows reads the Unicode process environment; POSIX returns native
  //! bytes unchanged, so names and values are expected to be UTF-8.
  //! Callers must synchronize environment mutations with reads on POSIX.
  static Status Get(const std::string &name, std::string *env);
};

}  // namespace ailego
}  // namespace zvec
