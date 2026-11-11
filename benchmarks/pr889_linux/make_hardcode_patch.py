#!/usr/bin/env python3
"""Generate an opt-in, benchmark-only patch; never changes the working sources."""

from __future__ import annotations

import difflib
from pathlib import Path

root = Path(__file__).resolve().parents[2]
changes = {}
name = "src/db/sqlengine/planner/scalar_doc_filter.h"
a = (root / name).read_text()
needle = """    return prototype_->function()(
        buffer_count_ ? buffers_.data() + base : nullptr,
        buffer_count_ ? offsets_.data() + base : nullptr, prototype_->holders(),
        static_cast<int64_t>(id - begin));"""
replacement = (
    """#ifdef PR889_BENCH_HARDCODE
    // Benchmark-only INT32 comparison; identical chunk routing and snapshot
    // lifetime, no code generation, per-row Arrow access or shared-state lock.
    const auto *values = reinterpret_cast<const int32_t *>(buffers_[base]);
    return values[offsets_[base] + id - begin] >
           static_cast<int32_t>(offsets_[base + 1]);
#else
"""
    + needle
    + "\n#endif"
)
assert a.count(needle) == 1
changes[name] = (a, a.replace(needle, replacement))
name = "src/db/sqlengine/planner/scalar_doc_filter.cc"
a = (root / name).read_text()
needle = """  std::vector<std::shared_ptr<const void>> query_owners;"""
hardcode = """#ifdef PR889_BENCH_HARDCODE
  // This binary is a benchmark comparator, not a supported execution backend.
  // Reject every shape except a bound, non-null INT32 dataTime > INT32 literal.
  const auto *call = expression.call();
  if (!call || call->function_name != "greater" || call->arguments.size() != 2 ||
      call->options || snapshot->num_columns() != 1) {
    return arrow::Status::NotImplemented("Benchmark hardcode predicate shape");
  }
  const auto *parameter = call->arguments[0].parameter();
  const auto *literal = call->arguments[1].literal();
  if (!parameter || parameter->indices.size() != 1 ||
      parameter->indices[0] != 0 ||
      snapshot->schema()->field(0)->name() != "dataTime" ||
      snapshot->schema()->field(0)->type()->id() != arrow::Type::INT32 ||
      !literal || !literal->is_scalar() || !literal->scalar() ||
      !literal->scalar()->is_valid || literal->type()->id() != arrow::Type::INT32) {
    return arrow::Status::NotImplemented("Benchmark hardcode INT32 predicate");
  }
  const int32_t threshold = literal->scalar_as<arrow::Int32Scalar>().value;
  auto result = std::unique_ptr<ScalarDocFilter>(new ScalarDocFilter);
  result->snapshot_ = snapshot;
  result->buffer_count_ = 2;
  for (const auto &chunk : snapshot->column(0)->chunks()) {
    if (chunk->length() == 0) continue;
    if (chunk->null_count() != 0) {
      return arrow::Status::NotImplemented("Benchmark hardcode null input");
    }
    const auto &array = *chunk->data();
    if (array.buffers.size() < 2 || !array.buffers[1] ||
        !array.buffers[1]->is_cpu() || !array.buffers[1]->data() ||
        array.buffers[1]->size() < (array.offset + array.length) * 4) {
      return arrow::Status::Invalid("Benchmark hardcode invalid INT32 buffer");
    }
    result->buffers_.push_back(array.buffers[1]->data());
    result->buffers_.push_back(nullptr);
    result->offsets_.push_back(array.offset);
    result->offsets_.push_back(threshold);
    result->num_rows_ += array.length;
    result->ends_.push_back(result->num_rows_);
  }
  if (result->ends_.empty() ||
      result->num_rows_ != static_cast<uint64_t>(snapshot->num_rows())) {
    return arrow::Status::Invalid("Benchmark hardcode incomplete snapshot");
  }
  result->stride_ = result->ends_.front();
  for (size_t i = 1; i < result->ends_.size(); ++i) {
    const uint64_t rows = result->ends_[i] - result->ends_[i - 1];
    if (rows > result->stride_ ||
        (i + 1 < result->ends_.size() && rows != result->stride_)) {
      result->stride_ = 0;
      break;
    }
  }
  return result;
#else
"""
assert a.count(needle) == 1
b = a.replace(needle, hardcode + needle)
needle2 = """  return result;
}

}  // namespace zvec::sqlengine"""
assert b.count(needle2) == 1
b = b.replace(
    needle2,
    """  return result;
#endif  // PR889_BENCH_HARDCODE
}

}  // namespace zvec::sqlengine""",
)
# Int32Scalar definition is required for the independent direct predicate.
b = b.replace(
    "#include <arrow/record_batch.h>",
    "#include <arrow/record_batch.h>\n#include <arrow/scalar.h>",
)
changes[name] = a, b
name = "src/db/sqlengine/planner/doc_filter.cc"
a = (root / name).read_text()
needle = '        LOG_DEBUG("Scalar JIT forward filter ready");'
if needle in a:
    b = a.replace(
        needle,
        '#ifdef PR889_BENCH_HARDCODE\n        LOG_DEBUG("Benchmark hardcoded INT32 forward filter ready");\n#else\n'
        + needle
        + "\n#endif",
    )
    changes[name] = a, b
out = "".join(
    "".join(
        difflib.unified_diff(
            a.splitlines(True),
            b.splitlines(True),
            fromfile="a/" + name,
            tofile="b/" + name,
        )
    )
    for name, (a, b) in changes.items()
)
(root / "benchmarks/pr889_linux/hardcode.patch").write_text(out)
print("Wrote benchmarks/pr889_linux/hardcode.patch")  # noqa: T201 - CLI output
