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
#include "db/sqlengine/scalar_jit/predicate_helpers.h"

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
  return type && (type->id() == arrow::Type::BOOL ||
                  type->id() == arrow::Type::STRING ||
                  type->id() == arrow::Type::BINARY || IsNumeric(type));
}

arrow::Result<gandiva::NodePtr> MakeLiteral(const arrow::Datum &literal) {
  if (!literal.is_scalar() || !literal.scalar() ||
      !IsSupportedType(literal.type().get())) {
    return arrow::Status::NotImplemented(
        "Scalar JIT requires a supported scalar literal");
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
    case arrow::Type::STRING:
      return TB::MakeStringLiteral(
          std::string(static_cast<const arrow::StringScalar &>(scalar).view()));
    case arrow::Type::BINARY:
      return TB::MakeBinaryLiteral(
          std::string(static_cast<const arrow::BinaryScalar &>(scalar).view()));
    default:
      return arrow::Status::NotImplemented("Unsupported scalar literal");
  }
}

class Converter {
 public:
  Converter(const arrow::Schema &schema,
            std::vector<std::shared_ptr<const void>> *query_owners)
      : schema_(schema), query_owners_(query_owners) {}

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
    if (name == "match_like" || name == "is_in") {
      if (!query_owners_ || call->arguments.size() != 1) {
        return arrow::Status::NotImplemented(
            "Scalar LIKE/IN requires query state ownership and one argument");
      }
      std::shared_ptr<const void> context;
      bool set_has_null = false;
      if (name == "match_like") {
        const auto *options = dynamic_cast<const cp::MatchSubstringOptions *>(
            call->options.get());
        if (!options ||
            call->arguments[0].type()->id() != arrow::Type::STRING) {
          return arrow::Status::NotImplemented("Unsupported LIKE options/type");
        }
        ARROW_ASSIGN_OR_RAISE(context, scalar_jit::MakeLikeContext(*options));
      } else {
        const auto *options =
            dynamic_cast<const cp::SetLookupOptions *>(call->options.get());
        const auto *type = call->arguments[0].type();
        if (!options ||
            options->GetNullMatchingBehavior() !=
                cp::SetLookupOptions::INCONCLUSIVE ||
            (!IsNumeric(type) && type->id() != arrow::Type::STRING) ||
            !options->value_set.type() ||
            !type->Equals(*options->value_set.type())) {
          return arrow::Status::NotImplemented("Unsupported IN options/types");
        }
        ARROW_ASSIGN_OR_RAISE(
            context, scalar_jit::MakeInContext(*options, &set_has_null));
      }
      ARROW_ASSIGN_OR_RAISE(auto value, convert(call->arguments[0]));
      auto pointer = TB::MakeLiteral(
          static_cast<int64_t>(reinterpret_cast<intptr_t>(context.get())));
      query_owners_->push_back(std::move(context));
      auto match = TB::MakeFunction(
          name == "match_like" ? "zvec_scalar_like" : "zvec_scalar_in",
          {value, pointer}, arrow::boolean());
      if (name == "match_like") return match;
      // Arrow INCONCLUSIVE: a miss becomes NULL when the set contains NULL.
      // OR with a nullable false preserves true matches and NULL inputs with
      // only one lookup. Cache parameterization also keeps set nullness out of
      // the compiled structure. Gandiva's external C ABI needs no extension.
      return TB::MakeOr({match, set_has_null ? TB::MakeNull(arrow::boolean())
                                             : TB::MakeLiteral(false)});
    }
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
          !IsSupportedType(call->arguments[0].type()) ||
          !IsSupportedType(call->arguments[1].type()) ||
          !call->arguments[0].type()->Equals(*call->arguments[1].type())) {
        return arrow::Status::NotImplemented(
            "Scalar JIT requires same-type comparison");
      }
      if (call->arguments[0].type()->id() == arrow::Type::BOOL &&
          name != "equal" && name != "not_equal") {
        return arrow::Status::NotImplemented("Unsupported boolean comparison");
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
    // Lists (CONTAIN and array_length), casts and functions with side effects
    // or errors keep the existing Arrow path.
    return arrow::Status::NotImplemented(
        "Scalar JIT does not support Arrow function: ", name);
  }

 private:
  const arrow::Schema &schema_;
  std::vector<std::shared_ptr<const void>> *query_owners_;
};
}  // namespace

arrow::Result<gandiva::ConditionPtr> MakeScalarPredicateCondition(
    const arrow::compute::Expression &expression,
    const std::shared_ptr<arrow::Schema> &schema,
    std::vector<std::shared_ptr<const void>> *query_owners) {
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
  std::vector<std::shared_ptr<const void>> owners;
  ARROW_ASSIGN_OR_RAISE(
      auto root,
      Converter(*schema, query_owners ? &owners : nullptr).convert(expression));
  if (query_owners) *query_owners = std::move(owners);
  return TB::MakeCondition(std::move(root));
}

}  // namespace zvec::sqlengine

#endif  // ZVEC_ENABLE_SCALAR_JIT
