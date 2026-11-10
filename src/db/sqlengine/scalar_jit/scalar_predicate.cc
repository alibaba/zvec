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

#include "db/sqlengine/scalar_jit/scalar_predicate.h"
#include <algorithm>
#include <cstring>
#include <unordered_set>
#include <utility>
#include "arrow/record_batch.h"
#include "arrow/scalar.h"
#include "arrow/type.h"
#include "gandiva/condition.h"
#include "gandiva/configuration.h"
#include "gandiva/expr_validator.h"
#include "gandiva/field_descriptor.h"
#include "gandiva/function_registry.h"
#include "gandiva/function_signature.h"
#include "gandiva/native_function.h"
#include "gandiva/node.h"
#include "gandiva/node_visitor.h"
#include "gandiva/scalar_generator.h"

namespace zvec::sqlengine::scalar_jit {
namespace {

arrow::Status ValidateScalarType(const std::shared_ptr<arrow::DataType> &type) {
  if (!type) return arrow::Status::Invalid("Scalar expression has a null type");
  switch (type->id()) {
    case arrow::Type::BOOL:
    case arrow::Type::INT8:
    case arrow::Type::INT16:
    case arrow::Type::INT32:
    case arrow::Type::INT64:
    case arrow::Type::UINT8:
    case arrow::Type::UINT16:
    case arrow::Type::UINT32:
    case arrow::Type::UINT64:
    case arrow::Type::FLOAT:
    case arrow::Type::DOUBLE:
      return arrow::Status::OK();
    default:
      return arrow::Status::NotImplemented("Scalar POC does not support type ",
                                           type->ToString());
  }
}

template <typename ScalarType>
arrow::Status CopyParameterValue(const arrow::Scalar &parameter,
                                 gandiva::ScalarParameterSlot *slot) {
  const auto *typed = dynamic_cast<const ScalarType *>(&parameter);
  ARROW_RETURN_IF(!typed, arrow::Status::Invalid(
                              "Scalar class does not match its Arrow type"));
  static_assert(sizeof(typed->value) <= sizeof(slot->data),
                "Parameter slot too small");
  // memcpy preserves the exact native integer/float representation, including
  // negative zero and NaN payloads, and does not alias the caller's Scalar.
  std::memcpy(slot->data, &typed->value, sizeof(typed->value));
  return arrow::Status::OK();
}

arrow::Status SnapshotParameter(const arrow::Scalar &parameter,
                                gandiva::ScalarParameterSlot *slot) {
  slot->validity = parameter.is_valid ? 1 : 0;
  if (!parameter.is_valid) return arrow::Status::OK();
  switch (parameter.type->id()) {
    case arrow::Type::BOOL: {
      const auto *typed =
          dynamic_cast<const arrow::BooleanScalar *>(&parameter);
      ARROW_RETURN_IF(
          !typed, arrow::Status::Invalid("Expected BooleanScalar parameter"));
      slot->data[0] = typed->value ? 1 : 0;
      return arrow::Status::OK();
    }
    case arrow::Type::INT8:
      return CopyParameterValue<arrow::Int8Scalar>(parameter, slot);
    case arrow::Type::INT16:
      return CopyParameterValue<arrow::Int16Scalar>(parameter, slot);
    case arrow::Type::INT32:
      return CopyParameterValue<arrow::Int32Scalar>(parameter, slot);
    case arrow::Type::INT64:
      return CopyParameterValue<arrow::Int64Scalar>(parameter, slot);
    case arrow::Type::UINT8:
      return CopyParameterValue<arrow::UInt8Scalar>(parameter, slot);
    case arrow::Type::UINT16:
      return CopyParameterValue<arrow::UInt16Scalar>(parameter, slot);
    case arrow::Type::UINT32:
      return CopyParameterValue<arrow::UInt32Scalar>(parameter, slot);
    case arrow::Type::UINT64:
      return CopyParameterValue<arrow::UInt64Scalar>(parameter, slot);
    case arrow::Type::FLOAT:
      return CopyParameterValue<arrow::FloatScalar>(parameter, slot);
    case arrow::Type::DOUBLE:
      return CopyParameterValue<arrow::DoubleScalar>(parameter, slot);
    default:
      return arrow::Status::NotImplemented("Unsupported parameter type: ",
                                           parameter.type->ToString());
  }
}

// An explicit restricted subset, not a claim that every Gandiva function can
// run without context, allocation, mutable global state, or error reporting.
class ScalarSubsetValidator : public gandiva::NodeVisitor {
 public:
  explicit ScalarSubsetValidator(
      std::shared_ptr<gandiva::FunctionRegistry> registry)
      : registry_(std::move(registry)) {}

  arrow::Status Visit(const gandiva::FieldNode &node) override {
    return ValidateScalarType(node.return_type());
  }
  arrow::Status Visit(const gandiva::LiteralNode &node) override {
    return ValidateScalarType(node.return_type());
  }
  arrow::Status Visit(const gandiva::FunctionNode &node) override {
    ARROW_RETURN_NOT_OK(ValidateScalarType(node.return_type()));
    const auto &descriptor = *node.descriptor();
    gandiva::FunctionSignature signature(descriptor.name(), descriptor.params(),
                                         descriptor.return_type());
    const auto *native = registry_->LookupSignature(signature);
    ARROW_RETURN_IF(!native, arrow::Status::Invalid("Unknown function: ",
                                                    descriptor.name()));
    ARROW_RETURN_IF(native->NeedsContext() || native->CanReturnErrors() ||
                        native->NeedsFunctionHolder(),
                    arrow::Status::NotImplemented(
                        "Scalar POC rejects context/error/holder function: ",
                        descriptor.name()));
    static const std::unordered_set<std::string> allowed = {
        "equal",
        "not_equal",
        "less_than",
        "less_than_or_equal_to",
        "greater_than",
        "greater_than_or_equal_to",
        "add",
        "subtract",
        "multiply",
        "not",
        "isnull",
        "isnotnull",
        "isnumeric",
        "is_distinct_from",
        "is_not_distinct_from",
        "istrue",
        "isfalse",
        "isnottrue",
        "isnotfalse",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "xor",
        "bitwise_not"};
    ARROW_RETURN_IF(
        allowed.find(descriptor.name()) == allowed.end(),
        arrow::Status::NotImplemented("Function outside scalar POC subset: ",
                                      descriptor.name()));
    for (const auto &child : node.children()) {
      ARROW_RETURN_NOT_OK(child->Accept(*this));
    }
    return arrow::Status::OK();
  }
  arrow::Status Visit(const gandiva::IfNode &node) override {
    ARROW_RETURN_NOT_OK(ValidateScalarType(node.return_type()));
    ARROW_RETURN_NOT_OK(node.condition()->Accept(*this));
    ARROW_RETURN_NOT_OK(node.then_node()->Accept(*this));
    return node.else_node()->Accept(*this);
  }
  arrow::Status Visit(const gandiva::BooleanNode &node) override {
    for (const auto &child : node.children()) {
      ARROW_RETURN_NOT_OK(child->Accept(*this));
    }
    return arrow::Status::OK();
  }
  arrow::Status Visit(const gandiva::InExpressionNode<int32_t> &node) override {
    return node.eval_expr()->Accept(*this);
  }
  arrow::Status Visit(const gandiva::InExpressionNode<int64_t> &node) override {
    return node.eval_expr()->Accept(*this);
  }
  arrow::Status Visit(const gandiva::InExpressionNode<float> &node) override {
    return node.eval_expr()->Accept(*this);
  }
  arrow::Status Visit(const gandiva::InExpressionNode<double> &node) override {
    return node.eval_expr()->Accept(*this);
  }
  arrow::Status Visit(
      const gandiva::InExpressionNode<gandiva::DecimalScalar128> &) override {
    return arrow::Status::NotImplemented("Scalar POC rejects decimal IN");
  }
  arrow::Status Visit(const gandiva::InExpressionNode<std::string> &) override {
    return arrow::Status::NotImplemented("Scalar POC rejects string IN");
  }

 private:
  std::shared_ptr<gandiva::FunctionRegistry> registry_;
};

}  // namespace

BoundScalarPredicate::BoundScalarPredicate(
    std::shared_ptr<const ScalarPredicate> compiled,
    std::shared_ptr<arrow::RecordBatch> batch,
    gandiva::ScalarEvalFunction function,
    std::vector<gandiva::ScalarParameterSlot> parameter_slots,
    std::vector<const uint8_t *> buffers, std::vector<int64_t> offsets,
    const void *const *holders, int64_t num_rows)
    : compiled_(std::move(compiled)),
      batch_(std::move(batch)),
      function_(function),
      parameter_slots_(std::move(parameter_slots)),
      buffers_(std::move(buffers)),
      offsets_(std::move(offsets)),
      holders_(holders),
      num_rows_(num_rows) {}

arrow::Result<bool> BoundScalarPredicate::evaluate_checked(int64_t row) const {
  if (row < 0 || row >= num_rows_) {
    return arrow::Status::Invalid("Scalar row ", row, " is outside [0, ",
                                  num_rows_, ")");
  }
  return evaluate(row);
}

ScalarPredicate::ScalarPredicate(
    std::shared_ptr<arrow::Schema> schema,
    std::shared_ptr<arrow::Schema> parameter_schema,
    std::unique_ptr<gandiva::ScalarGenerator> generator,
    gandiva::ScalarEvalFunction function,
    std::vector<ScalarDataBufferBinding> data_buffer_bindings)
    : schema_(std::move(schema)),
      parameter_schema_(std::move(parameter_schema)),
      generator_(std::move(generator)),
      function_(function),
      data_buffer_bindings_(std::move(data_buffer_bindings)) {}

ScalarPredicate::~ScalarPredicate() = default;

arrow::Result<std::shared_ptr<ScalarPredicate>> ScalarPredicate::Make(
    std::shared_ptr<arrow::Schema> schema,
    std::shared_ptr<gandiva::Condition> condition) {
  return MakeParameterized(std::move(schema), std::move(condition),
                           arrow::schema({}));
}

arrow::Result<std::shared_ptr<ScalarPredicate>>
ScalarPredicate::MakeParameterized(
    std::shared_ptr<arrow::Schema> schema,
    std::shared_ptr<gandiva::Condition> condition,
    std::shared_ptr<arrow::Schema> parameter_schema) {
  ARROW_RETURN_IF(!schema, arrow::Status::Invalid("Schema cannot be null"));
  ARROW_RETURN_IF(!parameter_schema,
                  arrow::Status::Invalid("Parameter schema cannot be null"));
  ARROW_RETURN_IF(!condition || !condition->root(),
                  arrow::Status::Invalid("Condition/root cannot be null"));
  ARROW_RETURN_IF(
      !condition->root()->return_type() ||
          condition->root()->return_type()->id() != arrow::Type::BOOL,
      arrow::Status::Invalid("Scalar condition root must return bool"));
  std::unordered_set<std::string> field_names;
  for (const auto &field : schema->fields()) {
    ARROW_RETURN_IF(
        !field_names.insert(field->name()).second,
        arrow::Status::Invalid("Duplicate schema field: ", field->name()));
  }
  auto augmented_fields = schema->fields();
  std::vector<std::string> parameter_names;
  for (const auto &field : parameter_schema->fields()) {
    ARROW_RETURN_IF(
        !field_names.insert(field->name()).second,
        arrow::Status::Invalid("Duplicate or colliding parameter field: ",
                               field->name()));
    ARROW_RETURN_NOT_OK(ValidateScalarType(field->type()));
    parameter_names.push_back(field->name());
    augmented_fields.push_back(field);
  }
  auto augmented_schema = arrow::schema(augmented_fields, schema->metadata());

  // A private configuration avoids modifying Gandiva's process-wide default.
  // Capture optimized IR for verification; only compilation is affected.
  auto configuration = std::make_shared<gandiva::Configuration>();
  configuration->set_dump_ir(true);
  ARROW_ASSIGN_OR_RAISE(auto generator,
                        gandiva::ScalarGenerator::Make(configuration));
  gandiva::ExprValidator validator(generator->types(), augmented_schema,
                                   configuration->function_registry());
  ARROW_RETURN_NOT_OK(validator.Validate(condition));
  ScalarSubsetValidator scalar_validator(configuration->function_registry());
  ARROW_RETURN_NOT_OK(condition->root()->Accept(scalar_validator));
  ARROW_ASSIGN_OR_RAISE(auto function,
                        generator->Build(condition, parameter_names));
  std::vector<ScalarDataBufferBinding> data_bindings;
  data_bindings.reserve(generator->input_descriptors().size());
  for (const auto &entry : generator->input_descriptors()) {
    const int column_index = schema->GetFieldIndex(entry.first);
    if (column_index < 0) {
      ARROW_RETURN_IF(
          parameter_schema->GetFieldIndex(entry.first) < 0,
          arrow::Status::Invalid("Unbound scalar input: ", entry.first));
      continue;
    }
    const auto &descriptor = *entry.second;
    data_bindings.push_back(
        {column_index, descriptor.data_idx(), descriptor.validity_idx()});
  }
  std::sort(data_bindings.begin(), data_bindings.end(),
            [](const ScalarDataBufferBinding &left,
               const ScalarDataBufferBinding &right) {
              return left.column_index < right.column_index;
            });
  return std::shared_ptr<ScalarPredicate>(new ScalarPredicate(
      std::move(schema), std::move(parameter_schema), std::move(generator),
      function, std::move(data_bindings)));
}

arrow::Result<std::shared_ptr<BoundScalarPredicate>> ScalarPredicate::bind(
    std::shared_ptr<arrow::RecordBatch> batch) const {
  return bind(std::move(batch), {});
}

arrow::Result<std::shared_ptr<BoundScalarPredicate>> ScalarPredicate::bind(
    std::shared_ptr<arrow::RecordBatch> batch,
    const std::vector<std::shared_ptr<arrow::Scalar>> &params) const {
  ARROW_RETURN_IF(
      params.size() != static_cast<size_t>(num_parameters()),
      arrow::Status::Invalid("Expected ", num_parameters(),
                             " parameters, received ", params.size()));
  ARROW_RETURN_IF(!batch, arrow::Status::Invalid("RecordBatch cannot be null"));
  ARROW_RETURN_IF(
      !schema_->Equals(*batch->schema()),
      arrow::Status::Invalid("RecordBatch schema differs from predicate"));
  // Validate() checks metadata and buffer sizes. Unlike ValidateFull(), it does
  // not inspect all array values or scan the row validity bitmap.
  ARROW_RETURN_NOT_OK(batch->Validate());
  std::vector<gandiva::ScalarParameterSlot> parameter_slots(params.size());
  for (size_t index = 0; index < params.size(); ++index) {
    const auto &parameter = params[index];
    const auto &field = parameter_schema_->field(static_cast<int>(index));
    ARROW_RETURN_IF(
        !parameter,
        arrow::Status::Invalid("Parameter ", field->name(), " is nullptr"));
    ARROW_RETURN_IF(
        !parameter->type || !parameter->type->Equals(field->type()),
        arrow::Status::Invalid("Parameter ", field->name(), " must have type ",
                               field->type()->ToString()));
    ARROW_RETURN_IF(!field->nullable() && !parameter->is_valid,
                    arrow::Status::Invalid("Non-nullable parameter ",
                                           field->name(), " cannot be NULL"));
    ARROW_RETURN_NOT_OK(SnapshotParameter(*parameter, &parameter_slots[index]));
  }
  std::vector<const uint8_t *> buffers;
  std::vector<int64_t> offsets;
  ARROW_RETURN_NOT_OK(
      generator_->BindBuffers(*batch, parameter_slots, &buffers, &offsets));
  const int64_t num_rows = batch->num_rows();
  return std::shared_ptr<BoundScalarPredicate>(new BoundScalarPredicate(
      shared_from_this(), std::move(batch), function_,
      std::move(parameter_slots), std::move(buffers), std::move(offsets),
      generator_->holders(), num_rows));
}

int ScalarPredicate::num_parameters() const {
  return parameter_schema_->num_fields();
}

const std::string &ScalarPredicate::dump_ir() const {
  return generator_->ir();
}

}  // namespace zvec::sqlengine::scalar_jit
