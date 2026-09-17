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

#include "db/sqlengine/planner/ops/fetch_vector_op.h"
#include <arrow/status.h>
#include "db/index/column/vector_column/combined_vector_column_indexer.h"
#include "db/sqlengine/common/util.h"

namespace zvec::sqlengine {

#define CHECK_ARROW_STATUS(msg, status)                                        \
  if (!status.ok()) {                                                          \
    return tl::make_unexpected(Status::InternalError(msg, status.ToString())); \
  }

template <typename Iter>
Result<std::shared_ptr<arrow::Array>> fetch_dense_vector_helper(
    const CombinedVectorColumnIndexer *indexer, Iter begin, Iter end) {
  size_t count = std::distance(begin, end);
  arrow::BinaryBuilder vector_builder;
  CHECK_ARROW_STATUS("Reserve vector builder failed:",
                     vector_builder.Reserve(count));
  for (Iter i = begin; i != end; ++i) {
    auto vector_res = indexer->fetch(*i);
    if (!vector_res) {
      return tl::make_unexpected(vector_res.error());
    }
    const auto &data = std::get<vector_column_params::DenseVectorBuffer>(
                           vector_res.value().vector_buffer)
                           .data;
    if (data.empty()) {
      CHECK_ARROW_STATUS("Append null failed:", vector_builder.AppendNull());
    } else {
      CHECK_ARROW_STATUS("Append vector failed:", vector_builder.Append(data));
    }
  }
  auto vector_array_ret = vector_builder.Finish();
  if (!vector_array_ret.ok()) {
    return tl::make_unexpected(Status::InternalError(
        "finish vector builder failed:", vector_array_ret.status().ToString()));
  }
  return vector_array_ret.MoveValueUnsafe();
}

Result<std::shared_ptr<arrow::Array>> FetchVectorOp::fetch_dense_vector(
    const Segment &segment, const std::string &vector_name,
    const std::vector<int> &rows) {
  auto indexer = segment.get_combined_vector_indexer(vector_name);
  if (!indexer) {
    return tl::make_unexpected(
        Status::InvalidArgument("vector index not found:", vector_name));
  }
  return fetch_dense_vector_helper(indexer.get(), rows.begin(), rows.end());
}


template <typename Iter>
Result<std::shared_ptr<arrow::Array>> fetch_sparse_vector_helper(
    const CombinedVectorColumnIndexer *indexer, Iter begin, Iter end) {
  size_t count = std::distance(begin, end);
  std::unique_ptr<arrow::StructBuilder> sparse_builder;
  arrow::BinaryBuilder *sparse_index_builder = nullptr;
  arrow::BinaryBuilder *sparse_value_builder = nullptr;
  auto array_builder = arrow::MakeBuilder(Util::sparse_type());
  if (!array_builder.ok()) {
    return tl::make_unexpected(Status::InternalError(
        "make builder failed:", array_builder.status().ToString()));
  }
  sparse_builder.reset(dynamic_cast<arrow::StructBuilder *>(
      array_builder.ValueUnsafe().release()));
  sparse_index_builder =
      dynamic_cast<arrow::BinaryBuilder *>(sparse_builder->field_builder(0));
  sparse_value_builder =
      dynamic_cast<arrow::BinaryBuilder *>(sparse_builder->field_builder(1));

  CHECK_ARROW_STATUS("Reserve failed:", sparse_builder->Reserve(count));
  CHECK_ARROW_STATUS("Reserve failed:", sparse_index_builder->Reserve(count));
  CHECK_ARROW_STATUS("Reserve failed:", sparse_value_builder->Reserve(count));
  for (auto i = begin; i != end; i++) {
    auto vector_res = indexer->fetch(*i);
    if (!vector_res) {
      return tl::make_unexpected(vector_res.error());
    }
    const auto &data = std::get<vector_column_params::SparseVectorBuffer>(
        vector_res.value().vector_buffer);
    if (data.indices.empty()) {
      // will auto append to sub builder
      CHECK_ARROW_STATUS("Append failed", sparse_builder->AppendNull());
    } else {
      CHECK_ARROW_STATUS("Append failed", sparse_builder->Append(true));
      CHECK_ARROW_STATUS("Append failed",
                         sparse_index_builder->Append(data.indices));
      CHECK_ARROW_STATUS("Append failed",
                         sparse_value_builder->Append(data.values));
    }
  }

  auto vector_array_ret = sparse_builder->Finish();
  if (!vector_array_ret.ok()) {
    return tl::make_unexpected(Status::InternalError(
        "finish vector builder failed:", vector_array_ret.status().ToString()));
  }
  return vector_array_ret.MoveValueUnsafe();
}

Result<std::shared_ptr<arrow::Array>> FetchVectorOp::fetch_sparse_vector(
    const Segment &segment, const std::string &vector_name,
    const std::vector<int> &rows) {
  auto indexer = segment.get_combined_vector_indexer(vector_name);
  if (!indexer) {
    return tl::make_unexpected(
        Status::InvalidArgument("vector index not found:", vector_name));
  }
  return fetch_sparse_vector_helper(indexer.get(), rows.begin(), rows.end());
}

std::unique_ptr<cp::FunctionOptions> FetchVectorOp::FunctionOptionsType::Copy(
    const cp::FunctionOptions &) const {
  return std::make_unique<FetchVectorFunctionOptions>();
}

arrow::Status FetchVectorOp::FetchVectorFunction(cp::KernelContext *ctx,
                                                 const cp::ExecSpan &batch,
                                                 cp::ExecResult *out) {
  FetchVectorState *state = static_cast<FetchVectorState *>(ctx->state());
  if (state->args.indexer == nullptr) {
    return arrow::Status::ExecutionError("indexer is null");
  }

  auto row_span = batch[0].array.GetSpan<uint64_t>(1, batch.length);
  Result<std::shared_ptr<arrow::Array>> res;
  if (state->args.is_dense) {
    res = fetch_dense_vector_helper(state->args.indexer.get(), row_span.begin(),
                                    row_span.end());
  } else {
    res = fetch_sparse_vector_helper(state->args.indexer.get(),
                                     row_span.begin(), row_span.end());
  }
  if (!res) {
    return arrow::Status::ExecutionError("fetch vector failed:",
                                         res.error().c_str());
  }

  out->value = std::move(res.value()->data());
  return arrow::Status::OK();
}

arrow::Result<std::unique_ptr<arrow::compute::KernelState>>
FetchVectorOp::InitExprValue(arrow::compute::KernelContext *,
                             const arrow::compute::KernelInitArgs &args) {
  auto func_options = static_cast<const FetchVectorOp::Options *>(args.options);
  return std::make_unique<FetchVectorOp::FetchVectorState>(
      func_options ? func_options : nullptr);
}


arrow::Status FetchVectorOp::register_op() {
  static Options options = Options::Defaults();
  {
    const std::string name = "fetch_vector";
    auto func = std::make_shared<cp::ScalarFunction>(name, cp::Arity::Unary(),
                                                     func_doc, &options, false);
    cp::ScalarKernel kernel({arrow::uint64()}, arrow::binary(),
                            FetchVectorFunction, InitExprValue);

    kernel.mem_allocation = cp::MemAllocation::NO_PREALLOCATE;
    kernel.null_handling = cp::NullHandling::COMPUTED_NO_PREALLOCATE;

    ARROW_RETURN_NOT_OK(func->AddKernel(std::move(kernel)));

    auto registry = cp::GetFunctionRegistry();
    ARROW_RETURN_NOT_OK(registry->AddFunction(std::move(func)));
  }

  {
    const std::string name = "fetch_sparse_vector";
    auto func = std::make_shared<cp::ScalarFunction>(name, cp::Arity::Unary(),
                                                     func_doc, &options, false);
    cp::ScalarKernel kernel({arrow::uint64()}, Util::sparse_type(),
                            FetchVectorFunction, InitExprValue);

    kernel.mem_allocation = cp::MemAllocation::NO_PREALLOCATE;
    kernel.null_handling = cp::NullHandling::COMPUTED_NO_PREALLOCATE;

    ARROW_RETURN_NOT_OK(func->AddKernel(std::move(kernel)));

    auto registry = cp::GetFunctionRegistry();
    ARROW_RETURN_NOT_OK(registry->AddFunction(std::move(func)));
  }

  return arrow::Status::OK();
}


}  // namespace zvec::sqlengine