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

void RegisterBuiltinIndexModules() {
  static const bool initialized = [] {
    RegisterBuiltinMetrics();
    RegisterBuiltinConvertersAndReformers();
    RegisterBuiltinUtilities();
    RegisterBuiltinMixedReducers();
    RegisterBuiltinClusters();
    RegisterBuiltinFlat();
    RegisterBuiltinFlatSparse();
    RegisterBuiltinHnsw();
    RegisterBuiltinHnswSparse();
    RegisterBuiltinIVF();
    RegisterBuiltinVamana();
#if DISKANN_SUPPORTED
    RegisterBuiltinDiskAnn();
#endif
#if RABITQ_SUPPORTED
    RegisterBuiltinHnswRabitq();
    RegisterBuiltinIvfRabitq();
#endif
    return true;
  }();
  (void)initialized;
}

namespace {
// Preserve startup registration for clients compiled against the old headers.
const bool registered = [] {
  RegisterBuiltinIndexModules();
  return true;
}();
}  // namespace

}  // namespace core
}  // namespace zvec
