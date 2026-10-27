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
void RegisterBuiltinCONVERTER_BinaryConverter();
void RegisterBuiltinCONVERTER_CosineFp16Converter();
void RegisterBuiltinCONVERTER_CosineFp32Converter();
void RegisterBuiltinCONVERTER_CosineHalfFloatConverter();
void RegisterBuiltinCONVERTER_CosineInt4Converter();
void RegisterBuiltinCONVERTER_CosineInt8Converter();
void RegisterBuiltinCONVERTER_CosineNormalizeConverter();
void RegisterBuiltinCONVERTER_CosineRawFp16Converter();
void RegisterBuiltinCONVERTER_HalfFloatConverter();
void RegisterBuiltinCONVERTER_HalfFloatSparseConverter();
void RegisterBuiltinCONVERTER_Int4QuantizerConverter();
void RegisterBuiltinCONVERTER_Int4StreamingConverter();
void RegisterBuiltinCONVERTER_Int8QuantizerConverter();
void RegisterBuiltinCONVERTER_Int8StreamingConverter();
void RegisterBuiltinCONVERTER_MipsConverter();
void RegisterBuiltinCONVERTER_RawUint8Converter();
void RegisterBuiltinCONVERTER_UniformUint4Converter();
void RegisterBuiltinCONVERTER_UniformUint7Converter();
void RegisterBuiltinCONVERTER_UniformUint8Converter();
void RegisterBuiltinREFORMER_BinaryReformer();
void RegisterBuiltinREFORMER_CosineFp16Reformer();
void RegisterBuiltinREFORMER_CosineFp32Reformer();
void RegisterBuiltinREFORMER_CosineHalfFloatReformer();
void RegisterBuiltinREFORMER_CosineInt4Reformer();
void RegisterBuiltinREFORMER_CosineInt8Reformer();
void RegisterBuiltinREFORMER_CosineNormalizeReformer();
void RegisterBuiltinREFORMER_CosineRawFp16Reformer();
void RegisterBuiltinREFORMER_HalfFloatReformer();
void RegisterBuiltinREFORMER_HalfFloatSparseReformer();
void RegisterBuiltinREFORMER_Int4QuantizerReformer();
void RegisterBuiltinREFORMER_Int4StreamingReformer();
void RegisterBuiltinREFORMER_Int8QuantizerReformer();
void RegisterBuiltinREFORMER_Int8StreamingReformer();
void RegisterBuiltinREFORMER_MipsReformer();
void RegisterBuiltinREFORMER_RawUint8Reformer();
void RegisterBuiltinREFORMER_UniformUint4Reformer();
void RegisterBuiltinREFORMER_UniformUint7Reformer();
void RegisterBuiltinREFORMER_UniformUint8Reformer();

void RegisterBuiltinConvertersAndReformers() {
  RegisterBuiltinCONVERTER_BinaryConverter();
  RegisterBuiltinCONVERTER_CosineFp16Converter();
  RegisterBuiltinCONVERTER_CosineFp32Converter();
  RegisterBuiltinCONVERTER_CosineHalfFloatConverter();
  RegisterBuiltinCONVERTER_CosineInt4Converter();
  RegisterBuiltinCONVERTER_CosineInt8Converter();
  RegisterBuiltinCONVERTER_CosineNormalizeConverter();
  RegisterBuiltinCONVERTER_CosineRawFp16Converter();
  RegisterBuiltinCONVERTER_HalfFloatConverter();
  RegisterBuiltinCONVERTER_HalfFloatSparseConverter();
  RegisterBuiltinCONVERTER_Int4QuantizerConverter();
  RegisterBuiltinCONVERTER_Int4StreamingConverter();
  RegisterBuiltinCONVERTER_Int8QuantizerConverter();
  RegisterBuiltinCONVERTER_Int8StreamingConverter();
  RegisterBuiltinCONVERTER_MipsConverter();
  RegisterBuiltinCONVERTER_RawUint8Converter();
  RegisterBuiltinCONVERTER_UniformUint4Converter();
  RegisterBuiltinCONVERTER_UniformUint7Converter();
  RegisterBuiltinCONVERTER_UniformUint8Converter();
  RegisterBuiltinREFORMER_BinaryReformer();
  RegisterBuiltinREFORMER_CosineFp16Reformer();
  RegisterBuiltinREFORMER_CosineFp32Reformer();
  RegisterBuiltinREFORMER_CosineHalfFloatReformer();
  RegisterBuiltinREFORMER_CosineInt4Reformer();
  RegisterBuiltinREFORMER_CosineInt8Reformer();
  RegisterBuiltinREFORMER_CosineNormalizeReformer();
  RegisterBuiltinREFORMER_CosineRawFp16Reformer();
  RegisterBuiltinREFORMER_HalfFloatReformer();
  RegisterBuiltinREFORMER_HalfFloatSparseReformer();
  RegisterBuiltinREFORMER_Int4QuantizerReformer();
  RegisterBuiltinREFORMER_Int4StreamingReformer();
  RegisterBuiltinREFORMER_Int8QuantizerReformer();
  RegisterBuiltinREFORMER_Int8StreamingReformer();
  RegisterBuiltinREFORMER_MipsReformer();
  RegisterBuiltinREFORMER_RawUint8Reformer();
  RegisterBuiltinREFORMER_UniformUint4Reformer();
  RegisterBuiltinREFORMER_UniformUint7Reformer();
  RegisterBuiltinREFORMER_UniformUint8Reformer();
}

}  // namespace core
}  // namespace zvec
