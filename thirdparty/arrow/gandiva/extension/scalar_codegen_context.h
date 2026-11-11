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

#pragma once

#include <unordered_set>
#include <vector>

#include "gandiva/llvm_types.h"

namespace gandiva {

// Compile-time policy for reusing the expression visitor in a single-row entry.
// A null policy leaves the visitor's batch code generation unchanged.
class ScalarCodeGenContext {
 public:
  // The builder must point to the function's entry block. These temporary i1
  // slots never escape and can be promoted to SSA by LLVM.
  ScalarCodeGenContext(llvm::IRBuilder<>* builder, LLVMTypes* types,
                       const std::unordered_set<int>& parameter_buffers,
                       int local_bitmap_count);

  llvm::Value* InputIndex(int buffer_idx, llvm::Value* row,
                          llvm::Value* slice_offset) const;
  llvm::Value* ReadValidity(int bitmap_idx) const;
  void ClearValidityIfNotValid(int bitmap_idx, llvm::Value* is_valid) const;

 private:
  llvm::IRBuilder<>* builder_;
  LLVMTypes* types_;
  const std::unordered_set<int>& parameter_buffers_;
  std::vector<llvm::AllocaInst*> validities_;
};

}  // namespace gandiva
