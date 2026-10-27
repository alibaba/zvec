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
void RegisterBuiltinMETRIC_Cosine();
void RegisterBuiltinMETRIC_Euclidean();
void RegisterBuiltinMETRIC_InnerProduct();
void RegisterBuiltinMETRIC_InnerProductSparse();
void RegisterBuiltinMETRIC_MipsSquaredEuclidean();
void RegisterBuiltinMETRIC_MipsSquaredEuclideanSparse();
void RegisterBuiltinMETRIC_NormalizedCosine();
void RegisterBuiltinMETRIC_QuantizedInteger();
void RegisterBuiltinMETRIC_SquaredEuclidean();
void RegisterBuiltinMETRIC_SquaredEuclideanSparse();
void RegisterBuiltinMETRIC_UniformUint4();
void RegisterBuiltinMETRIC_UniformUint7();
void RegisterBuiltinMETRIC_UniformUint8();

void RegisterBuiltinMetrics() {
  RegisterBuiltinMETRIC_Cosine();
  RegisterBuiltinMETRIC_Euclidean();
  RegisterBuiltinMETRIC_InnerProduct();
  RegisterBuiltinMETRIC_InnerProductSparse();
  RegisterBuiltinMETRIC_MipsSquaredEuclidean();
  RegisterBuiltinMETRIC_MipsSquaredEuclideanSparse();
  RegisterBuiltinMETRIC_NormalizedCosine();
  RegisterBuiltinMETRIC_QuantizedInteger();
  RegisterBuiltinMETRIC_SquaredEuclidean();
  RegisterBuiltinMETRIC_SquaredEuclideanSparse();
  RegisterBuiltinMETRIC_UniformUint4();
  RegisterBuiltinMETRIC_UniformUint7();
  RegisterBuiltinMETRIC_UniformUint8();
}

}  // namespace core
}  // namespace zvec
