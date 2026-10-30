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
void RegisterBuiltinConverterBinaryConverter();
void RegisterBuiltinConverterCosineFp16Converter();
void RegisterBuiltinConverterCosineFp32Converter();
void RegisterBuiltinConverterCosineHalfFloatConverter();
void RegisterBuiltinConverterCosineInt4Converter();
void RegisterBuiltinConverterCosineInt8Converter();
void RegisterBuiltinConverterCosineNormalizeConverter();
void RegisterBuiltinConverterCosineRawFp16Converter();
void RegisterBuiltinConverterHalfFloatConverter();
void RegisterBuiltinConverterHalfFloatSparseConverter();
void RegisterBuiltinConverterInt4QuantizerConverter();
void RegisterBuiltinConverterInt4StreamingConverter();
void RegisterBuiltinConverterInt8QuantizerConverter();
void RegisterBuiltinConverterInt8StreamingConverter();
void RegisterBuiltinConverterMipsConverter();
void RegisterBuiltinConverterRawUint8Converter();
void RegisterBuiltinConverterUniformUint4Converter();
void RegisterBuiltinConverterUniformUint7Converter();
void RegisterBuiltinConverterUniformUint8Converter();
void RegisterBuiltinReformerBinaryReformer();
void RegisterBuiltinReformerCosineFp16Reformer();
void RegisterBuiltinReformerCosineFp32Reformer();
void RegisterBuiltinReformerCosineHalfFloatReformer();
void RegisterBuiltinReformerCosineInt4Reformer();
void RegisterBuiltinReformerCosineInt8Reformer();
void RegisterBuiltinReformerCosineNormalizeReformer();
void RegisterBuiltinReformerCosineRawFp16Reformer();
void RegisterBuiltinReformerHalfFloatReformer();
void RegisterBuiltinReformerHalfFloatSparseReformer();
void RegisterBuiltinReformerInt4QuantizerReformer();
void RegisterBuiltinReformerInt4StreamingReformer();
void RegisterBuiltinReformerInt8QuantizerReformer();
void RegisterBuiltinReformerInt8StreamingReformer();
void RegisterBuiltinReformerMipsReformer();
void RegisterBuiltinReformerRawUint8Reformer();
void RegisterBuiltinReformerUniformUint4Reformer();
void RegisterBuiltinReformerUniformUint7Reformer();
void RegisterBuiltinReformerUniformUint8Reformer();

void RegisterBuiltinConvertersAndReformers() {
  RegisterBuiltinConverterBinaryConverter();
  RegisterBuiltinConverterCosineFp16Converter();
  RegisterBuiltinConverterCosineFp32Converter();
  RegisterBuiltinConverterCosineHalfFloatConverter();
  RegisterBuiltinConverterCosineInt4Converter();
  RegisterBuiltinConverterCosineInt8Converter();
  RegisterBuiltinConverterCosineNormalizeConverter();
  RegisterBuiltinConverterCosineRawFp16Converter();
  RegisterBuiltinConverterHalfFloatConverter();
  RegisterBuiltinConverterHalfFloatSparseConverter();
  RegisterBuiltinConverterInt4QuantizerConverter();
  RegisterBuiltinConverterInt4StreamingConverter();
  RegisterBuiltinConverterInt8QuantizerConverter();
  RegisterBuiltinConverterInt8StreamingConverter();
  RegisterBuiltinConverterMipsConverter();
  RegisterBuiltinConverterRawUint8Converter();
  RegisterBuiltinConverterUniformUint4Converter();
  RegisterBuiltinConverterUniformUint7Converter();
  RegisterBuiltinConverterUniformUint8Converter();
  RegisterBuiltinReformerBinaryReformer();
  RegisterBuiltinReformerCosineFp16Reformer();
  RegisterBuiltinReformerCosineFp32Reformer();
  RegisterBuiltinReformerCosineHalfFloatReformer();
  RegisterBuiltinReformerCosineInt4Reformer();
  RegisterBuiltinReformerCosineInt8Reformer();
  RegisterBuiltinReformerCosineNormalizeReformer();
  RegisterBuiltinReformerCosineRawFp16Reformer();
  RegisterBuiltinReformerHalfFloatReformer();
  RegisterBuiltinReformerHalfFloatSparseReformer();
  RegisterBuiltinReformerInt4QuantizerReformer();
  RegisterBuiltinReformerInt4StreamingReformer();
  RegisterBuiltinReformerInt8QuantizerReformer();
  RegisterBuiltinReformerInt8StreamingReformer();
  RegisterBuiltinReformerMipsReformer();
  RegisterBuiltinReformerRawUint8Reformer();
  RegisterBuiltinReformerUniformUint4Reformer();
  RegisterBuiltinReformerUniformUint7Reformer();
  RegisterBuiltinReformerUniformUint8Reformer();
}

}  // namespace core
}  // namespace zvec
