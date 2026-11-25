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

#include <algorithm>
#include <arrow/array/util.h>
#include <arrow/compute/exec.h>
#include <arrow/record_batch.h>
#include "db/sqlengine/planner/doc_filter.h"

namespace zvec::sqlengine {
namespace cp = arrow::compute;

namespace {

arrow::Status AppendPredicateResult(const cp::Expression &predicate,
                                    const cp::ExecBatch &input,
                                    cp::ExecContext *context,
                                    arrow::ArrayVector *parts) {
  ARROW_ASSIGN_OR_RAISE(auto value,
                        cp::ExecuteScalarExpression(predicate, input, context));
  if (value.is_scalar()) {
    ARROW_ASSIGN_OR_RAISE(
        auto array, arrow::MakeArrayFromScalar(*value.scalar(), input.length,
                                               context->memory_pool()));
    parts->push_back(std::move(array));
  } else if (!value.is_arraylike() || value.length() != input.length) {
    return arrow::Status::Invalid("Unexpected forward predicate output shape");
  } else if (value.is_array()) {
    parts->push_back(value.make_array());
  } else {
    const auto &chunks = value.chunked_array()->chunks();
    parts->insert(parts->end(), chunks.begin(), chunks.end());
  }
  return arrow::Status::OK();
}

}  // namespace

arrow::Result<std::shared_ptr<arrow::ChunkedArray>>
DocFilter::build_forward_bitmap() const {
  const auto error = [&](const char *stage, const arrow::Status &status) {
    return status.WithMessage("Forward bitmap ", stage, " failed: segment[",
                              segment_->id(), "] ", status.message());
  };
  const auto &fields = query_info_->get_forward_filter_field_names();
  auto prepared = segment_->prepare_forward_read(fields, false);
  if (!prepared.ok()) return error("prepare", prepared.status());
  auto state = std::move(prepared).ValueUnsafe();
  std::vector<arrow::Datum> columns;
  if (state.view) {
    columns.reserve(fields.size());
    for (int i = 0; i < state.view->num_columns(); ++i) {
      const auto &data = state.view->column(i)->data();
      if (!data) break;  // Virtual row ids are supplied by the scanner.
      columns.emplace_back(data);
    }
  }

  auto schema = state.schema;
  RecordBatchReaderPtr reader;
  if (columns.size() != fields.size()) {
    reader = segment_->scan(fields);
    if (!reader) {
      return error("scan",
                   arrow::Status::Invalid("Missing forward filter scan"));
    }
    // An empty scanner has no schema. Bind against the captured projection.
    if (auto scan_schema = reader->schema()) {
      schema = std::move(scan_schema);
    } else if (state.num_rows != 0) {
      return error(
          "scan", arrow::Status::Invalid("Missing forward filter scan schema"));
    }
  }
  // 默认 executor 为空，标量表达式在当前查询线程同步执行。
  // Arrow 21 尚未使用 use_threads 开关；不能仅依赖该开关保证不调度线程池。
  cp::ExecContext context;
  context.set_use_threads(false);
  auto binding = bitmap_filter_->Bind(*schema, &context);
  if (!binding.ok()) return error("bind", binding.status());
  // The scanner and the view both supply unconditional batches.
  auto simplified = cp::SimplifyWithGuarantee(std::move(binding).ValueUnsafe(),
                                              cp::literal(true));
  if (!simplified.ok()) return error("simplify", simplified.status());
  auto bound = std::move(simplified).ValueUnsafe();
  if (bound.type()->id() != arrow::Type::BOOL) {
    return error("bind", arrow::Status::TypeError(
                             "Expected a boolean forward predicate"));
  }

  arrow::ArrayVector parts;
  auto evaluate = [&](const cp::ExecBatch &input) -> arrow::Status {
    // Cap intermediate arrays for compound predicates. Slices share the input
    // buffers even when their physical chunk boundaries differ by column.
    constexpr int64_t kBatchRows = 256 * 1024;
    if (input.length == 0) return arrow::Status::OK();
    if (input.length <= kBatchRows) {
      return AppendPredicateResult(bound, input, &context, &parts);
    }
    for (int64_t offset = 0; offset < input.length;) {
      const int64_t length = std::min(kBatchRows, input.length - offset);
      ARROW_RETURN_NOT_OK(AppendPredicateResult(
          bound, input.Slice(offset, length), &context, &parts));
      offset += length;
    }
    return arrow::Status::OK();
  };
  if (reader) {
    for (;;) {
      auto next = reader->Next();
      if (!next.ok()) return error("scan", next.status());
      auto batch = std::move(next).ValueUnsafe();
      if (!batch) break;
      auto status = evaluate(cp::ExecBatch(*batch));
      if (!status.ok()) return error("evaluate", status);
    }
  } else {
    auto status = evaluate(cp::ExecBatch(std::move(columns),
                                         static_cast<int64_t>(state.num_rows)));
    if (!status.ok()) return error("evaluate", status);
  }

  return std::make_shared<arrow::ChunkedArray>(std::move(parts),
                                               arrow::boolean());
}

}  // namespace zvec::sqlengine
