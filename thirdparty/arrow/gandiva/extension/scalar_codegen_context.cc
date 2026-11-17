// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.
//
// Modifications Copyright 2025-present the zvec project.
// Modified for zvec's scalar JIT integration.

#include "gandiva/scalar_codegen_context.h"

#include <string>

namespace gandiva {

ScalarCodeGenContext::ScalarCodeGenContext(
    llvm::IRBuilder<>* builder, LLVMTypes* types,
    const std::unordered_set<int>& parameter_buffers, int local_bitmap_count)
    : builder_(builder), types_(types), parameter_buffers_(parameter_buffers) {
  validities_.reserve(local_bitmap_count);
  for (int index = 0; index < local_bitmap_count; ++index) {
    auto* slot = builder_->CreateAlloca(types_->i1_type(), nullptr,
                                        "scalar_validity_" + std::to_string(index));
    builder_->CreateStore(types_->true_constant(), slot);
    validities_.push_back(slot);
  }
}

llvm::Value* ScalarCodeGenContext::InputIndex(int buffer_idx, llvm::Value* row,
                                              llvm::Value* slice_offset) const {
  if (parameter_buffers_.count(buffer_idx) != 0) {
    return types_->i64_constant(0);
  }
  return builder_->CreateAdd(row, slice_offset);
}

llvm::Value* ScalarCodeGenContext::ReadValidity(int bitmap_idx) const {
  return builder_->CreateLoad(types_->i1_type(), validities_.at(bitmap_idx));
}

void ScalarCodeGenContext::ClearValidityIfNotValid(int bitmap_idx,
                                                   llvm::Value* is_valid) const {
  auto* slot = validities_.at(bitmap_idx);
  auto* previous = builder_->CreateLoad(types_->i1_type(), slot);
  builder_->CreateStore(builder_->CreateAnd(previous, is_valid), slot);
}

}  // namespace gandiva
