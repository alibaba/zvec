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

#include <zvec/ailego/container/vector.h>

namespace zvec {
namespace ailego {

/*! Process Helper
 */
struct ProcessHelper {
  //! Retrieve current process id
  static uint32_t SelfPid(void);

  //! Retrieve current thread id
  static uint32_t SelfTid(void);

  //! Retrieve current parent process id
  static uint32_t ParentPid(void);

  //! Retrieve a backtrace for calling program
  static size_t BackTrace(void **buf, size_t size);

  //! Retrieve non-zero if the process id exists
  static bool IsExist(uint32_t pid);

  //! Run the process in the background
  static void Daemon(const char *out, const char *err);

  //! Ignore signal
  static void IgnoreSignal(int sig);

  //! Register signal
  static void RegisterSignal(int sig, void f(int));

  //! Retrieve the name of a signal id
  static const char *SignalName(int sig);
};

}  // namespace ailego
}  // namespace zvec
