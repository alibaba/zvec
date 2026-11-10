// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include "db/sqlengine/scalar_jit/scalar_predicate.h"

namespace zvec::sqlengine::scalar_jit {

/// One query's immutable executable and extracted, typed literal arguments.
/// bind snapshots the argument values; changing these Scalars afterwards does
/// not affect an existing binding. Different queries never mutate compiled.
struct PreparedScalarPredicate {
  std::shared_ptr<ScalarPredicate> compiled;
  std::vector<std::shared_ptr<arrow::Scalar>> parameters;
  bool cache_hit = false;
  std::vector<std::shared_ptr<const void>> query_owners;

  arrow::Result<std::shared_ptr<BoundScalarPredicate>> bind(
      std::shared_ptr<arrow::RecordBatch> batch) const;
};

struct ScalarPredicateCacheStats {
  uint64_t lookups = 0;  // Requests passing structural analysis.
  uint64_t hits = 0;     // Includes requests joining an in-flight compile.
  uint64_t misses = 0;
  uint64_t compilations = 0;  // Compile attempts, including failed attempts.
  uint64_t failed_compilations = 0;
  uint64_t evictions = 0;
  uint64_t waits = 0;
  size_t entries = 0;  // Completed resident entries, bounded by capacity.
};

/// Process-local, thread-safe LRU of parameterized executable predicates.
///
/// prepare analyzes typed Gandiva nodes, extracts scalar literals in
/// traversal order, and looks up the structural expression and exact schema.
/// Literal values/nullness and RecordBatch addresses are not part of the key.
/// Operators, operand order, field identities/types and tree structure are.
/// Adapter-generated LIKE/IN use owned query state passed as parameters;
/// patterns and sets do not enter the key. Direct Gandiva IN nodes still carry
/// compile-time constants, which ARE included in the key.
/// There is no algebraic/commutative canonicalization or SQL text parser.
///
/// Capacity bounds completed cached programs; in-flight compilations are
/// coalesced per key but may temporarily exceed capacity across distinct keys.
/// Compilation runs outside the cache mutex. Bound handles keep evicted code
/// alive. The scalar function registry is immutable after initialization.
class ScalarPredicateCache {
 public:
  explicit ScalarPredicateCache(size_t capacity = 128);
  ~ScalarPredicateCache();
  ScalarPredicateCache(const ScalarPredicateCache &) = delete;
  ScalarPredicateCache &operator=(const ScalarPredicateCache &) = delete;

  arrow::Result<PreparedScalarPredicate> prepare(
      std::shared_ptr<arrow::Schema> schema,
      std::shared_ptr<gandiva::Condition> condition,
      std::vector<std::shared_ptr<const void>> query_owners = {});

  ScalarPredicateCacheStats stats() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace zvec::sqlengine::scalar_jit
