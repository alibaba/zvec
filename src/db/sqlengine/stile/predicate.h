// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(__FAST_MATH__) ||                                  \
    (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__) || \
    defined(__NO_SIGNED_ZEROS__) || defined(_M_FP_FAST)
#error \
    "Stile requires IEEE floating-point semantics. Disable fast-math and finite-only modes."
#endif

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <arrow/compute/expression.h>
#include <arrow/result.h>
#include "db/sqlengine/stile/functions.h"
#include "db/sqlengine/stile/table_index.h"

namespace zvec::stile {

// SQL three-valued truth.
enum class Truth : uint8_t { kFalse = 0, kTrue = 1, kNull = 2 };

// Register Arrow compute and prepare its built-ins. Call before binding raw
// Arrow expressions, replacing global built-ins or starting independent Arrow
// LIKE workers. Compile also calls this automatically.
arrow::Status Initialize();

struct CompileOptions {
  // Application functions; null compiles Arrow's functions only.
  const FunctionRegistry *functions = nullptr;
  // Deeper boolean trees return NotImplemented.
  int max_depth = 64;
  // Hard, per-compilation budgets. Exceeding one returns CapacityError.
  size_t max_nodes = 16384;
  size_t max_operands = 4096;     // per flattened AND/OR
  size_t max_set_values = 65536;  // cumulative, before deduplication
  size_t max_set_chunks = 65536;  // includes empty chunks
  size_t max_literal_bytes = 64 * 1024 * 1024;
  // A soft optimization budget. Remaining comparisons compile separately.
  size_t max_fusion_pairs = 4096;
  bool validate_full = false;  // also validate query literals and sets
};

namespace detail {

// A row, located once when every referenced column shares one chunk layout.
struct Row {
  static constexpr size_t kUnresolved = ~size_t{0};
  int64_t id;
  size_t chunk;
  int64_t local;  // row within `chunk` when resolved
};

inline Row LocateRow(int64_t id, const ColumnIndex *layout) {
  Row row{id, Row::kUnresolved, 0};
  if (layout) row.chunk = layout->locate(id, &row.local);
  return row;
}

// Evaluation results encode truth in two bits: bit 0 "definitely true",
// bit 1 "definitely false", neither for NULL. Kleene logic is then plain
// bitwise arithmetic without branches.
using Tf = uint8_t;
inline constexpr Tf kTfNull = 0, kTfTrue = 1, kTfFalse = 2;

class Node {
 public:
  // Rows per EvalBatch / Prefetch call.
  static constexpr size_t kMaxBatch = 64;

  virtual ~Node() = default;
  virtual Tf eval(const Row &row) const = 0;
  // out[i] = eval(rows[i]) for count <= kMaxBatch rows.
  virtual void eval_batch(const Row *rows, size_t count, Tf *out) const {
    for (size_t i = 0; i < count; ++i) out[i] = eval(rows[i]);
  }
  // Starts loading the memory eval(rows[i]) reads.
  virtual void prefetch(const Row *, size_t) const {}
  // Relative cost of one Eval, for ordering and short-circuiting.
  virtual int cost() const {
    return 1;
  }
};

}  // namespace detail

// A boolean Arrow compute expression compiled for one table, evaluated one
// row at a time without materializing scalars or batches.
//
// Results use the same three-valued semantics as Arrow's
// ExecuteScalarExpression. Compile() returns NotImplemented for expressions,
// types or layouts outside its supported subset, including inputs on which
// Arrow can fail. A Predicate is immutable and safe to evaluate concurrently.
class Predicate final {
 public:
  // `bound` must be bound to `table->schema()`. Hard limits return
  // CapacityError. Allocation failure returns OutOfMemory when Arrow can
  // allocate the status. Sustained heap exhaustion can throw std::bad_alloc.
  // The expression and its options must stay immutable during compilation.
  static arrow::Result<std::unique_ptr<Predicate>> Compile(
      const arrow::compute::Expression &bound, TableIndex::Ptr table,
      const CompileOptions &options = {});

  ~Predicate();

  // Three-valued result; `row` must be in [0, num_rows()).
  Truth evaluate(int64_t row) const {
    assert(row >= 0 && row < num_rows_);
    constexpr Truth kTruth[3] = {Truth::kNull, Truth::kTrue, Truth::kFalse};
    return kTruth[root_->eval(locate(row))];
  }

  // evaluate(row) == kTrue; false for rows outside the table.
  bool matches(int64_t row) const {
    return static_cast<uint64_t>(row) < static_cast<uint64_t>(num_rows_) &&
           (root_->eval(locate(row)) & detail::kTfTrue) != 0;
  }

  // matches() of `count` rows at once: matches[i] = matches(rows[i]).
  // Both arrays must have count entries; either may be null for count == 0.
  // Returns the number of matches. Faster than one row at a time when the
  // rows are scattered, since it overlaps their memory accesses.
  size_t match_many(const int64_t *rows, size_t count, bool *matches) const;

  // evaluate() of `count` rows at once; every row must be in range.
  // Both arrays must have count entries; either may be null for count == 0.
  void evaluate_many(const int64_t *rows, size_t count, Truth *out) const;

  int64_t num_rows() const {
    return num_rows_;
  }
  const TableIndex::Ptr &table() const {
    return table_;
  }
  // Whether rows are located once for all columns (for diagnostics).
  bool shares_chunk_layout() const {
    return layout_ != nullptr;
  }

 private:
  Predicate(TableIndex::Ptr table, std::unique_ptr<detail::Node> root,
            const ColumnIndex *layout);

  detail::Row locate(int64_t row) const {
    return detail::LocateRow(row, layout_);
  }
  // Evaluates rows[0, count) for count <= kMaxBatch; out-of-range rows
  // (when `checked`) give NULL.
  void eval_block(const int64_t *rows, size_t count, bool checked,
                  detail::Tf *out) const;

  TableIndex::Ptr table_;  // owns every buffer the nodes read
  std::unique_ptr<detail::Node> root_;
  const ColumnIndex *layout_;
  int64_t num_rows_;
};

}  // namespace zvec::stile
