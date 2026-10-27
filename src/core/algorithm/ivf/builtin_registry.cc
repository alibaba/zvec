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

#include <zvec/core/framework/index_factory_builtins.h>

namespace zvec {
namespace core {

// Keep this list explicit: add new built-ins here beside their module.
void RegisterBuiltinBUILDER_IVFBuilder();
void RegisterBuiltinSEARCHER_IVFSearcher();
void RegisterBuiltinSTREAMER_IVFStreamer();

void RegisterBuiltinIVF() {
  RegisterBuiltinBUILDER_IVFBuilder();
  RegisterBuiltinSEARCHER_IVFSearcher();
  RegisterBuiltinSTREAMER_IVFStreamer();
}

}  // namespace core
}  // namespace zvec
