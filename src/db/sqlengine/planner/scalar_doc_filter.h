// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#ifdef ZVEC_ENABLE_SCALAR_JIT

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>
#include <arrow/compute/expression.h>
#include <arrow/table.h>
#include "db/sqlengine/scalar_jit/scalar_predicate_cache.h"

namespace zvec::sqlengine {

// One query's immutable executable, parameters and owning storage bindings.
// A candidate is routed to a batch and evaluated at its batch-local offset.
class ScalarDocFilter {
 public:
  static bool Enabled();
  static arrow::Result<std::unique_ptr<ScalarDocFilter>> Make(
      const arrow::compute::Expression &expression,
      const std::shared_ptr<arrow::Table> &snapshot);

  static scalar_jit::ScalarPredicateCacheStats CacheStats();

  std::optional<bool> evaluate(uint64_t id) const {
    if (id >= num_rows_) {
      return std::nullopt;
    }
    const size_t index =
        stride_
            ? id / stride_
            : std::upper_bound(ends_.begin(), ends_.end(), id) - ends_.begin();
    const uint64_t begin = index ? ends_[index - 1] : 0;
    const size_t base = index * buffer_count_;
    // The query's parameter pointers are already in each descriptor. Neither
    // Arrow objects nor mutable shared state are touched on this path.
    return prototype_->function()(
        buffer_count_ ? buffers_.data() + base : nullptr,
        buffer_count_ ? offsets_.data() + base : nullptr, prototype_->holders(),
        static_cast<int64_t>(id - begin));
  }

  size_t num_batches() const {
    return ends_.size();
  }

 private:
  friend class DocFilter;
  // Only DocFilter may supply Segment's metadata-validated snapshot. Public
  // Make still checks arbitrary caller-owned tables before raw buffer access.
  static arrow::Result<std::unique_ptr<ScalarDocFilter>> MakeValidated(
      const arrow::compute::Expression &expression,
      const std::shared_ptr<arrow::Table> &snapshot);
  // prototype_ owns executable code, holders and this query's scalar slots.
  // snapshot_ owns every Arrow buffer referenced by the flat descriptors.
  scalar_jit::MemoryTicket memory_ticket_;
  std::shared_ptr<scalar_jit::BoundScalarPredicate> prototype_;
  std::shared_ptr<arrow::Table> snapshot_;
  std::vector<const uint8_t *> buffers_;
  std::vector<int64_t> offsets_;
  size_t buffer_count_ = 0;
  std::vector<uint64_t> ends_;
  uint64_t num_rows_ = 0;
  uint64_t stride_ = 0;
};

}  // namespace zvec::sqlengine
#endif
