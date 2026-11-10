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

#include "scalar_predicate_expr.h"

#ifdef ZVEC_ENABLE_SCALAR_JIT

#include <array>
#include <string>
#include <unordered_set>
#include <utility>
#include <arrow/compute/api_scalar.h>
#include <arrow/scalar.h>
#include <gandiva/tree_expr_builder.h>

namespace zvec::sqlengine {
namespace {
namespace cp = arrow::compute;
using TB = gandiva::TreeExprBuilder;

bool IsNumeric(const arrow::DataType *type) {
  if (!type) return false;
  switch (type->id()) {
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
      return true;
    default:
      return false;
  }
}

bool IsSupportedType(const arrow::DataType *type) {
  return type && (type->id() == arrow::Type::BOOL || IsNumeric(type));
}

arrow::Result<gandiva::NodePtr> MakeLiteral(const arrow::Datum &literal) {
  if (!literal.is_scalar() || !literal.scalar() ||
      !IsSupportedType(literal.type().get())) {
    return arrow::Status::NotImplemented(
        "Scalar JIT requires a fixed-width numeric/bool scalar literal");
  }
  const auto &scalar = *literal.scalar();
  if (!scalar.is_valid) return TB::MakeNull(scalar.type);
  switch (scalar.type->id()) {
    case arrow::Type::BOOL:
      return TB::MakeLiteral(
          static_cast<const arrow::BooleanScalar &>(scalar).value);
    case arrow::Type::INT8:
      return TB::MakeLiteral(
          static_cast<const arrow::Int8Scalar &>(scalar).value);
    case arrow::Type::INT16:
      return TB::MakeLiteral(
          static_cast<const arrow::Int16Scalar &>(scalar).value);
    case arrow::Type::INT32:
      return TB::MakeLiteral(
          static_cast<const arrow::Int32Scalar &>(scalar).value);
    case arrow::Type::INT64:
      return TB::MakeLiteral(
          static_cast<const arrow::Int64Scalar &>(scalar).value);
    case arrow::Type::UINT8:
      return TB::MakeLiteral(
          static_cast<const arrow::UInt8Scalar &>(scalar).value);
    case arrow::Type::UINT16:
      return TB::MakeLiteral(
          static_cast<const arrow::UInt16Scalar &>(scalar).value);
    case arrow::Type::UINT32:
      return TB::MakeLiteral(
          static_cast<const arrow::UInt32Scalar &>(scalar).value);
    case arrow::Type::UINT64:
      return TB::MakeLiteral(
          static_cast<const arrow::UInt64Scalar &>(scalar).value);
    case arrow::Type::FLOAT:
      return TB::MakeLiteral(
          static_cast<const arrow::FloatScalar &>(scalar).value);
    case arrow::Type::DOUBLE:
      return TB::MakeLiteral(
          static_cast<const arrow::DoubleScalar &>(scalar).value);
    default:
      return arrow::Status::NotImplemented("Unsupported scalar literal");
  }
}

class Converter {
 public:
  explicit Converter(const arrow::Schema &schema) : schema_(schema) {}

  arrow::Result<gandiva::NodePtr> convert(
      const cp::Expression &expression) const {
    if (!IsSupportedType(expression.type())) {
      return arrow::Status::NotImplemented(
          "Unsupported scalar expression type");
    }
    if (const auto *literal = expression.literal())
      return MakeLiteral(*literal);
    if (const auto *parameter = expression.parameter()) {
      // Bind records the resolved path, including when the original reference
      // was an index or a field name containing punctuation. Nested fields are
      // intentionally outside the initial zero-copy binding contract.
      if (parameter->indices.size() != 1) {
        return arrow::Status::NotImplemented(
            "Scalar JIT requires a top-level field");
      }
      int index = parameter->indices[0];
      if (index < 0 || index >= schema_.num_fields()) {
        return arrow::Status::Invalid(
            "Bound field index does not match schema");
      }
      const auto &field = schema_.field(index);
      if (!parameter->type.type ||
          !field->type()->Equals(*parameter->type.type)) {
        return arrow::Status::Invalid("Bound field type does not match schema");
      }
      ARROW_ASSIGN_OR_RAISE(auto path, parameter->ref.FindOne(schema_));
      if (path.indices().size() != 1 || path.indices()[0] != index) {
        return arrow::Status::Invalid(
            "Bound field reference does not match schema");
      }
      return TB::MakeField(field);
    }

    const auto *call = expression.call();
    if (!call)
      return arrow::Status::NotImplemented("Unsupported expression node");
    const auto &name = call->function_name;
    if (call->options) {
      // QueryPlanner's cp::is_null always supplies NullOptions(false).
      // Accept that exact behavior; treating NaN as NULL is a different
      // operation and must continue through Arrow.
      const auto *null_options =
          dynamic_cast<const cp::NullOptions *>(call->options.get());
      if (name != "is_null" || !null_options || null_options->nan_is_null) {
        return arrow::Status::NotImplemented(
            "Scalar JIT rejects function options: ", name);
      }
    }

    static constexpr std::array<std::pair<const char *, const char *>, 6>
        comparisons = {{{"equal", "equal"},
                        {"not_equal", "not_equal"},
                        {"less", "less_than"},
                        {"less_equal", "less_than_or_equal_to"},
                        {"greater", "greater_than"},
                        {"greater_equal", "greater_than_or_equal_to"}}};
    for (const auto &comparison : comparisons) {
      if (name != comparison.first) continue;
      if (call->arguments.size() != 2 ||
          !IsNumeric(call->arguments[0].type()) ||
          !IsNumeric(call->arguments[1].type()) ||
          !call->arguments[0].type()->Equals(*call->arguments[1].type())) {
        return arrow::Status::NotImplemented(
            "Scalar JIT requires same-type numeric comparison");
      }
      ARROW_ASSIGN_OR_RAISE(auto left, convert(call->arguments[0]));
      ARROW_ASSIGN_OR_RAISE(auto right, convert(call->arguments[1]));
      return TB::MakeFunction(comparison.second, {left, right},
                              arrow::boolean());
    }

    if (name == "and_kleene" || name == "or_kleene") {
      if (call->arguments.size() != 2 ||
          call->arguments[0].type()->id() != arrow::Type::BOOL ||
          call->arguments[1].type()->id() != arrow::Type::BOOL) {
        return arrow::Status::NotImplemented(
            "Scalar JIT requires two boolean operands");
      }
      ARROW_ASSIGN_OR_RAISE(auto left, convert(call->arguments[0]));
      ARROW_ASSIGN_OR_RAISE(auto right, convert(call->arguments[1]));
      return name == "and_kleene" ? TB::MakeAnd({left, right})
                                  : TB::MakeOr({left, right});
    }
    if (name == "invert" || name == "is_null" || name == "is_valid") {
      if (call->arguments.size() != 1 ||
          (name == "invert" &&
           call->arguments[0].type()->id() != arrow::Type::BOOL)) {
        return arrow::Status::NotImplemented(
            "Unsupported unary scalar predicate");
      }
      ARROW_ASSIGN_OR_RAISE(auto argument, convert(call->arguments[0]));
      const char *target = name == "invert"
                               ? "not"
                               : (name == "is_null" ? "isnull" : "isnotnull");
      return TB::MakeFunction(target, {argument}, arrow::boolean());
    }
    // No implicit broadening to arithmetic, casts, IN, strings or functions
    // with side effects/errors. In particular Gandiva IN has different NULL
    // behavior from QueryPlanner's Arrow INCONCLUSIVE set lookup.
    return arrow::Status::NotImplemented(
        "Scalar JIT does not support Arrow function: ", name);
  }

 private:
  const arrow::Schema &schema_;
};
}  // namespace

arrow::Result<gandiva::ConditionPtr> MakeScalarPredicateCondition(
    const arrow::compute::Expression &expression,
    const std::shared_ptr<arrow::Schema> &schema) {
  if (!schema || !expression.is_valid() || !expression.IsBound()) {
    return arrow::Status::NotImplemented(
        "Scalar JIT requires a schema and bound expression");
  }
  if (!expression.type() || expression.type()->id() != arrow::Type::BOOL) {
    return arrow::Status::NotImplemented("Scalar predicate must return bool");
  }
  std::unordered_set<std::string> names;
  for (const auto &field : schema->fields()) {
    if (!names.insert(field->name()).second) {
      return arrow::Status::NotImplemented(
          "Scalar JIT rejects duplicate field names");
    }
  }
  ARROW_ASSIGN_OR_RAISE(auto root, Converter(*schema).convert(expression));
  return TB::MakeCondition(std::move(root));
}

}  // namespace zvec::sqlengine

#endif  // ZVEC_ENABLE_SCALAR_JIT
