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

#include "gandiva/scalar_generator.h"

#include <utility>

#include "arrow/array/data.h"
#include "arrow/record_batch.h"
#include "gandiva/configuration.h"
#include "gandiva/expr_decomposer.h"
#include "gandiva/expression.h"
#include "gandiva/field_descriptor.h"
#include "gandiva/llvm_generator.h"
#include "gandiva/scalar_codegen_context.h"

namespace gandiva {

ScalarGenerator::ScalarGenerator(std::unique_ptr<LLVMGenerator> generator)
    : generator_(std::move(generator)) {}

ScalarGenerator::~ScalarGenerator() = default;

arrow::Result<std::unique_ptr<ScalarGenerator>> ScalarGenerator::Make(
    const std::shared_ptr<Configuration>& configuration) {
  ARROW_ASSIGN_OR_RAISE(auto generator, LLVMGenerator::Make(configuration, false));
  return std::unique_ptr<ScalarGenerator>(new ScalarGenerator(std::move(generator)));
}

LLVMTypes* ScalarGenerator::types() const { return generator_->types(); }

const std::string& ScalarGenerator::ir() const { return generator_->ir(); }

const void* const* ScalarGenerator::holders() const {
  return generator_->annotator_.GetHolderPointersArray();
}

const std::unordered_map<std::string, FieldDescriptorPtr>&
ScalarGenerator::input_descriptors() const {
  return generator_->annotator_.input_descriptors();
}

arrow::Result<ScalarEvalFunction> ScalarGenerator::Build(
    const ExpressionPtr& expression,
    const std::vector<std::string>& parameter_field_names) {
  ARROW_RETURN_IF(expression_, arrow::Status::Invalid("Scalar generator must be fresh"));
  ARROW_RETURN_IF(!expression, arrow::Status::Invalid("Scalar expression is null"));
  auto& annotator = generator_->annotator_;
  ExprDecomposer decomposer(*generator_->function_registry_, annotator);
  ARROW_RETURN_NOT_OK(decomposer.Decompose(*expression->root(), &expression_));
  std::unordered_set<int> parameter_buffers;
  for (size_t index = 0; index < parameter_field_names.size(); ++index) {
    const auto& name = parameter_field_names[index];
    ARROW_RETURN_IF(!parameter_indices_.emplace(name, index).second,
                    arrow::Status::Invalid("Duplicate parameter field: ", name));
    auto found = annotator.input_descriptors().find(name);
    if (found != annotator.input_descriptors().end()) {
      parameter_buffers.insert(found->second->data_idx());
      parameter_buffers.insert(found->second->validity_idx());
      if (found->second->HasOffsetsIdx()) {
        parameter_buffers.insert(found->second->offsets_idx());
      }
    }
  }
  auto& engine = *generator_->engine_;
  ARROW_RETURN_NOT_OK(engine.LoadFunctionIRs());
  const std::string function_name = "gandiva_scalar_predicate_v1";
  ARROW_RETURN_NOT_OK(CodeGen(*expression_, parameter_buffers, function_name));
  ARROW_RETURN_NOT_OK(engine.FinalizeModule());
  ARROW_ASSIGN_OR_RAISE(auto address, engine.CompiledFunction(function_name));
  return reinterpret_cast<ScalarEvalFunction>(address);
}

arrow::Status ScalarGenerator::BindBuffers(
    const arrow::RecordBatch& batch, const std::vector<ScalarParameterSlot>& parameters,
    std::vector<const uint8_t*>* buffers, std::vector<int64_t>* offsets) const {
  ARROW_RETURN_IF(
      parameters.size() != parameter_indices_.size(),
      arrow::Status::Invalid("Parameter slot count differs from compiled predicate"));
  const auto& annotator = generator_->annotator_;
  buffers->assign(annotator.buffer_count(), nullptr);
  offsets->assign(annotator.buffer_count(), 0);
  for (const auto& entry : annotator.input_descriptors()) {
    const auto& descriptor = *entry.second;
    auto parameter = parameter_indices_.find(entry.first);
    if (parameter != parameter_indices_.end()) {
      const auto& slot = parameters[parameter->second];
      (*buffers)[descriptor.data_idx()] =
          descriptor.HasOffsetsIdx() ? slot.variable_data : slot.data;
      (*buffers)[descriptor.validity_idx()] = &slot.validity;
      if (descriptor.HasOffsetsIdx()) {
        ARROW_RETURN_IF(!slot.variable_data,
                        arrow::Status::Invalid("Missing variable-width parameter data"));
        (*buffers)[descriptor.offsets_idx()] =
            reinterpret_cast<const uint8_t*>(slot.variable_offsets);
      }
      continue;
    }
    const int column_index = batch.schema()->GetFieldIndex(entry.first);
    ARROW_RETURN_IF(column_index < 0,
                    arrow::Status::Invalid("Missing or ambiguous field: ", entry.first));
    const auto& array = *batch.column_data(column_index);
    // The owning runtime validates Arrow metadata before binding. Null byte
    // storage is valid for an empty string payload, but not for nonempty data.
    const bool variable_width = descriptor.HasOffsetsIdx();
    const size_t value_index = variable_width ? 2 : 1;
    ARROW_RETURN_IF(
        array.buffers.size() <= value_index,
        arrow::Status::Invalid("Missing Arrow buffers for ", entry.first));
    for (size_t index = 0; index <= value_index; ++index) {
      const auto& buffer = array.buffers[index];
      ARROW_RETURN_IF(buffer && !buffer->is_cpu(),
                      arrow::Status::NotImplemented("Scalar inputs require CPU buffers"));
      ARROW_RETURN_IF(buffer && buffer->size() > 0 && !buffer->data(),
                      arrow::Status::Invalid("Arrow buffer has no data for ", entry.first));
    }
    const auto& values = array.buffers[value_index];
    ARROW_RETURN_IF(batch.num_rows() > 0 &&
                        (!values || (!values->data() && values->size() != 0) ||
                         (!variable_width && !values->data())),
                    arrow::Status::Invalid("Missing value buffer for ", entry.first));
    ARROW_RETURN_IF(array.null_count > 0 && !array.buffers[0],
                    arrow::Status::Invalid("Missing validity buffer for ", entry.first));
    static const uint8_t empty_value = 0;
    (*buffers)[descriptor.validity_idx()] =
        array.buffers[0] ? array.buffers[0]->data() : nullptr;
    (*buffers)[descriptor.data_idx()] =
        values && values->data() ? values->data() : &empty_value;
    (*offsets)[descriptor.validity_idx()] = array.offset;
    (*offsets)[descriptor.data_idx()] = array.offset;
    if (variable_width) {
      ARROW_RETURN_IF(batch.num_rows() > 0 &&
                          (!array.buffers[1] || !array.buffers[1]->data()),
                      arrow::Status::Invalid("Missing offsets buffer for ", entry.first));
      (*buffers)[descriptor.offsets_idx()] =
          array.buffers[1] ? array.buffers[1]->data() : nullptr;
      (*offsets)[descriptor.offsets_idx()] = array.offset;
    }
  }
  return arrow::Status::OK();
}

arrow::Status ScalarGenerator::CodeGen(const ValueValidityPair& expression,
                                       const std::unordered_set<int>& parameter_buffers,
                                       const std::string& function_name) {
  auto& generator = *generator_;
  auto* builder = generator.ir_builder();
  auto* llvm_types = generator.types();
  auto* prototype =
      llvm::FunctionType::get(llvm_types->i1_type(),
                              {llvm_types->i64_ptr_type(), llvm_types->i64_ptr_type(),
                               llvm_types->i64_ptr_type(), llvm_types->i64_type()},
                              false);
  generator.engine_->AddFunctionToCompile(function_name);
  auto* function = llvm::Function::Create(prototype, llvm::GlobalValue::ExternalLinkage,
                                          function_name, generator.module());
  function->addRetAttr(llvm::Attribute::ZExt);
  auto arguments = function->arg_begin();
  llvm::Value* buffers = &*arguments++;
  llvm::Value* offsets = &*arguments++;
  llvm::Value* holders = &*arguments++;
  llvm::Value* row = &*arguments;
  buffers->setName("buffers");
  offsets->setName("offsets");
  holders->setName("holders");
  row->setName("row");

  auto* entry = llvm::BasicBlock::Create(*generator.context(), "entry", function);
  auto* body = llvm::BasicBlock::Create(*generator.context(), "predicate", function);
  builder->SetInsertPoint(entry);
  std::vector<llvm::Value*> slice_offsets;
  for (int index = 0; index < generator.annotator_.buffer_count(); ++index) {
    auto* address = builder->CreateGEP(llvm_types->i64_type(), offsets,
                                       llvm_types->i32_constant(index));
    slice_offsets.push_back(builder->CreateLoad(llvm_types->i64_type(), address));
  }
  ScalarCodeGenContext scalar_context(builder, llvm_types, parameter_buffers,
                                      generator.annotator_.local_bitmap_count());
  builder->SetInsertPoint(body);
  LLVMGenerator::Visitor visitor(&generator, function, entry, buffers, nullptr, holders,
                                 std::move(slice_offsets), llvm_types->i64_constant(0),
                                 row, &scalar_context);
  auto result = visitor.BuildValueAndValidity(expression);
  ARROW_RETURN_IF(
      visitor.has_arena_allocs(),
      arrow::Status::NotImplemented("Scalar execution rejects arena allocations"));
  builder->CreateRet(builder->CreateAnd(result->data(), result->validity(), "passed"));
  builder->SetInsertPoint(entry);
  builder->CreateBr(body);
  return arrow::Status::OK();
}

}  // namespace gandiva
