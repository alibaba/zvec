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

#include <ailego/version.i>
#include <zvec/core/framework/index_version.h>


namespace zvec {
namespace core {

static const char AITHETA_VERSION_DETAILS[] =
    AILEGO_VERSION_COMPILE_DETAILS("All rights reserved.\n");

const char *IndexVersion::String(void) {
  return AITHETA_VERSION_DETAILS;
}

const char *IndexVersion::Details(void) {
  return AITHETA_VERSION_DETAILS;
}

}  // namespace core
}  // namespace zvec
