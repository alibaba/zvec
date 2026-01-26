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
// limitations under the License

#include "db/sqlengine/planner/ops/contain_op.h"
#include <memory>
#include <arrow/api.h>
#include <zvec/db/type.h>
#include "db/sqlengine/common/util.h"

namespace zvec::sqlengine {

enum class ContainType { kContainAll, kContainAny };
template <typename ArrowArrayType, ContainType contain_type>
bool match_value(const arrow::Array *value_array, int64_t offset,
                 int64_t length, const arrow::Array *value_set_array) {
  auto *value_typed_arr = static_cast<const ArrowArrayType *>(value_array);
  auto *value_set_typed_arr =
      static_cast<const ArrowArrayType *>(value_set_array);
  if (contain_type == ContainType::kContainAll) {
    for (int j = 0; j < value_set_typed_arr->length(); ++j) {
      bool contain = false;
      for (int i = 0; i < length; ++i) {
        if constexpr (std::is_same_v<ArrowArrayType, arrow::StringArray> ||
                      std::is_same_v<ArrowArrayType, arrow::LargeStringArray> ||
                      std::is_same_v<ArrowArrayType, arrow::BinaryArray> ||
                      std::is_same_v<ArrowArrayType, arrow::LargeBinaryArray>) {
          if (value_typed_arr->GetView(offset + i) ==
              value_set_typed_arr->GetView(j)) {
            contain = true;
            break;
          }
        } else {
          if (value_typed_arr->Value(offset + i) ==
              value_set_typed_arr->Value(j)) {
            contain = true;
            break;
          }
        }
      }
      if (!contain) {
        return false;
      }
    }
    return true;
  } else {  // contain_type == kContainAny
    for (int j = 0; j < value_set_typed_arr->length(); ++j) {
      for (int i = 0; i < length; ++i) {
        if constexpr (std::is_same_v<ArrowArrayType, arrow::StringArray> ||
                      std::is_same_v<ArrowArrayType, arrow::LargeStringArray> ||
                      std::is_same_v<ArrowArrayType, arrow::BinaryArray> ||
                      std::is_same_v<ArrowArrayType, arrow::LargeBinaryArray>) {
          if (value_typed_arr->GetView(offset + i) ==
              value_set_typed_arr->GetView(j)) {
            return true;
          }
        } else {
          if (value_typed_arr->Value(offset + i) ==
              value_set_typed_arr->Value(j)) {
            return true;
          }
        }
      }
    }
    return false;
  }
}

template <ContainType contain_type>
arrow::Status ContainFunction(cp::KernelContext *ctx, const cp::ExecSpan &batch,
                              cp::ExecResult *out) {
  auto *state = static_cast<ContainOp::ContainState *>(ctx->state());
  const auto &value_set = state->args.value_set;
  if (value_set == nullptr) {
    return arrow::Status::ExecutionError("value_set is null");
  }

  const auto &input_array = batch[0].array;
  if (batch[0].type()->id() != arrow::Type::LIST) {
    return arrow::Status::ExecutionError("batch type is not list");
  }
  if (!input_array.type->field(0)->type()->Equals(value_set->type())) {
    return arrow::Status::ExecutionError(
        "value_set type is not equal to batch type");
  }
  auto list_array =
      std::dynamic_pointer_cast<arrow::ListArray>(input_array.ToArray());

  std::shared_ptr<arrow::BooleanBuilder> builder =
      std::make_shared<arrow::BooleanBuilder>(ctx->memory_pool());
  ARROW_RETURN_NOT_OK(builder->Reserve(batch.length));
  const auto &list_value_array = list_array->values();
  for (int i = 0; i < batch.length; i++) {
    // a whole list may be null for a doc
    if (list_array->IsNull(i)) {
      ARROW_RETURN_NOT_OK(builder->AppendNull());
      continue;
    }
    auto length = list_array->value_length(i);
    auto offset = list_array->value_offset(i);
    bool match = false;
    switch (state->args.data_type) {
      case DataType::ARRAY_INT32:
        match = match_value<arrow::Int32Array, contain_type>(
            list_value_array.get(), offset, length, value_set.get());
        break;

      case DataType::ARRAY_UINT32:
        match = match_value<arrow::UInt32Array, contain_type>(
            list_value_array.get(), offset, length, value_set.get());
        break;

      case DataType::ARRAY_INT64:
        match = match_value<arrow::Int64Array, contain_type>(
            list_value_array.get(), offset, length, value_set.get());
        break;

      case DataType::ARRAY_UINT64:
        match = match_value<arrow::UInt64Array, contain_type>(
            list_value_array.get(), offset, length, value_set.get());
        break;

      case DataType::ARRAY_FLOAT:
        match = match_value<arrow::FloatArray, contain_type>(
            list_value_array.get(), offset, length, value_set.get());
        break;

      case DataType::ARRAY_DOUBLE:
        match = match_value<arrow::DoubleArray, contain_type>(
            list_value_array.get(), offset, length, value_set.get());
        break;

      case DataType::ARRAY_STRING:
        match = match_value<arrow::StringArray, contain_type>(
            list_value_array.get(), offset, length, value_set.get());
        break;

      case DataType::ARRAY_BOOL:
        match = match_value<arrow::BooleanArray, contain_type>(
            list_value_array.get(), offset, length, value_set.get());
        break;

      default:
        return arrow::Status::ExecutionError("unsupported data type");
    }
    ARROW_RETURN_NOT_OK(builder->Append(match));
  }

  std::shared_ptr<arrow::Array> result_array;
  ARROW_RETURN_NOT_OK(builder->Finish(&result_array));

  out->value = std::move(result_array->data());
  //   out->array_data()->type = batch[0].type()->GetShared::Ptr();
  return arrow::Status::OK();
}

arrow::Result<std::unique_ptr<arrow::compute::KernelState>>
ContainOp::InitExprValue(arrow::compute::KernelContext *,
                         const arrow::compute::KernelInitArgs &args) {
  auto func_options = static_cast<const ContainOp::Options *>(args.options);
  return std::make_unique<ContainOp::ContainState>(func_options ? func_options
                                                                : nullptr);
}


arrow::Status ContainOp::register_op() {
  static Options options = Options::Defaults();

  {
    auto func = std::make_shared<cp::ScalarFunction>(
        kContainAll, cp::Arity::Unary(), func_doc, &options, false);
    for (const auto &type :
         {arrow::int32(), arrow::uint32(), arrow::int64(), arrow::uint64(),
          arrow::float32(), arrow::float64(), arrow::utf8(),
          arrow::boolean()}) {
      cp::ScalarKernel kernel({arrow::list(type)}, arrow::boolean(),
                              ContainFunction<ContainType::kContainAll>,
                              InitExprValue);
      kernel.mem_allocation = cp::MemAllocation::NO_PREALLOCATE;
      kernel.null_handling = cp::NullHandling::INTERSECTION;
      ARROW_RETURN_NOT_OK(func->AddKernel(std::move(kernel)));
    }

    auto registry = cp::GetFunctionRegistry();
    ARROW_RETURN_NOT_OK(registry->AddFunction(std::move(func)));
  }

  {
    auto func = std::make_shared<cp::ScalarFunction>(
        kContainAny, cp::Arity::Unary(), func_doc, &options, false);
    for (const auto &type :
         {arrow::int32(), arrow::uint32(), arrow::int64(), arrow::uint64(),
          arrow::float32(), arrow::float64(), arrow::utf8(),
          arrow::boolean()}) {
      cp::ScalarKernel kernel({arrow::list(type)}, arrow::boolean(),
                              ContainFunction<ContainType::kContainAny>,
                              InitExprValue);
      kernel.mem_allocation = cp::MemAllocation::NO_PREALLOCATE;
      kernel.null_handling = cp::NullHandling::INTERSECTION;
      ARROW_RETURN_NOT_OK(func->AddKernel(std::move(kernel)));
    }

    auto registry = cp::GetFunctionRegistry();
    ARROW_RETURN_NOT_OK(registry->AddFunction(std::move(func)));
  }

  return arrow::Status::OK();
}

}  // namespace zvec::sqlengine