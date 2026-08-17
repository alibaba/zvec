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

#include "version.h"
#include "version.i"

#ifdef ailego_VERSION
#define AILEGO_VERSION_STRING ailego_VERSION
#else
#define AILEGO_VERSION_STRING "unknown"
#endif

namespace zvec {

namespace ailego {

static const char AILEGO_VERSION_DETAILS[] = AILEGO_VERSION_COMPILE_DETAILS(
    "AiLego Library Version " AILEGO_VERSION_STRING
    ".\nCopyright (C) The Software Authors. All rights reserved.\n");

const char *Version::String() {
  return AILEGO_VERSION_STRING;
}

const char *Version::Details() {
  return AILEGO_VERSION_DETAILS;
}

}  // namespace ailego
}  // namespace zvec

// extern "C" int __wrap_main(int, char *[]) {
//   fwrite(ailego::AILEGO_VERSION_DETAILS, 1,
//          strlen(ailego::AILEGO_VERSION_DETAILS), stdout);
//   fflush(stdout);
//   _Exit(0);
// }
