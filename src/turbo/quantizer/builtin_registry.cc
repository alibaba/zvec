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
void RegisterBuiltinQUANTIZER_Fp16Quantizer();
void RegisterBuiltinQUANTIZER_Fp32Quantizer();
void RegisterBuiltinQUANTIZER_Int4Quantizer();
void RegisterBuiltinQUANTIZER_Int8Quantizer();
void RegisterBuiltinQUANTIZER_PqFastQuantizer();
void RegisterBuiltinQUANTIZER_PqInt4Quantizer();
void RegisterBuiltinQUANTIZER_PqInt8Quantizer();

void RegisterBuiltinQuantizers() {
  static const bool initialized = [] {
    RegisterBuiltinQUANTIZER_Fp16Quantizer();
    RegisterBuiltinQUANTIZER_Fp32Quantizer();
    RegisterBuiltinQUANTIZER_Int4Quantizer();
    RegisterBuiltinQUANTIZER_Int8Quantizer();
    RegisterBuiltinQUANTIZER_PqFastQuantizer();
    RegisterBuiltinQUANTIZER_PqInt4Quantizer();
    RegisterBuiltinQUANTIZER_PqInt8Quantizer();
    return true;
  }();
  (void)initialized;
}

namespace {
// Older clients do not call FactoryBuiltinLoader.
const bool registered = [] {
  RegisterBuiltinQuantizers();
  return true;
}();
}  // namespace

}  // namespace core
}  // namespace zvec
