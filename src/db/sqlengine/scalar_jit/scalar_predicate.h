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

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "arrow/result.h"
#include "arrow/type_fwd.h"
#include "gandiva/scalar_predicate_abi.h"

namespace gandiva {
class Condition;
class ScalarGenerator;
}  // namespace gandiva

namespace zvec::sqlengine::scalar_jit {

class ScalarPredicate;

// Compile-time mapping for a referenced data column. Parameter fields are
// excluded. Indices address the same flat buffer/offset arrays exposed by a
// BoundScalarPredicate; column_index addresses the original data schema.
struct ScalarDataBufferBinding {
  int column_index;
  int data_buffer_index;
  int validity_buffer_index;
  // -1 for fixed-width values; otherwise indexes the int32 Arrow offsets.
  int offsets_buffer_index = -1;
};

/// An immutable binding of one compiled predicate to an Arrow RecordBatch.
/// Keeps both the input buffers and executable code alive. Concurrent evaluate
/// calls are supported, provided callers do not mutate the input Arrow buffers.
class BoundScalarPredicate {
 public:
  // Flattened buffer pointers may point into parameter_slots_. Prevent copies
  // or moves that could detach those pointers from their owning allocation.
  BoundScalarPredicate(const BoundScalarPredicate &) = delete;
  BoundScalarPredicate &operator=(const BoundScalarPredicate &) = delete;
  BoundScalarPredicate(BoundScalarPredicate &&) = delete;
  BoundScalarPredicate &operator=(BoundScalarPredicate &&) = delete;

  /// Hot path: row MUST satisfy 0 <= row < num_rows(). This wrapper performs
  /// no Arrow object lookup, bounds checking, or batch scan. Direct numeric
  /// predicates allocate no memory and take no locks; helper implementations
  /// such as regular-expression matching may have additional runtime costs.
  bool evaluate(int64_t row) const {
    return function_(buffers_.data(), offsets_.data(), holders_, row);
  }

  arrow::Result<bool> evaluate_checked(int64_t row) const;

  int64_t num_rows() const {
    return num_rows_;
  }
  gandiva::ScalarEvalFunction function() const {
    return function_;
  }
  const uint8_t *const *buffers() const {
    return buffers_.data();
  }
  const int64_t *offsets() const {
    return offsets_.data();
  }
  const void *const *holders() const {
    return holders_;
  }
  size_t num_buffers() const {
    return buffers_.size();
  }

 private:
  friend class ScalarPredicate;
  BoundScalarPredicate(
      std::shared_ptr<const ScalarPredicate> compiled,
      std::shared_ptr<arrow::RecordBatch> batch,
      gandiva::ScalarEvalFunction function,
      std::vector<gandiva::ScalarParameterSlot> parameter_slots,
      std::vector<std::shared_ptr<arrow::Buffer>> parameter_values,
      std::vector<std::shared_ptr<const void>> owners,
      std::vector<const uint8_t *> buffers, std::vector<int64_t> offsets,
      const void *const *holders, int64_t num_rows);

  std::shared_ptr<const ScalarPredicate> compiled_;
  std::shared_ptr<arrow::RecordBatch> batch_;
  gandiva::ScalarEvalFunction function_;
  std::vector<gandiva::ScalarParameterSlot> parameter_slots_;
  std::vector<std::shared_ptr<arrow::Buffer>> parameter_values_;
  std::vector<std::shared_ptr<const void>> owners_;
  std::vector<const uint8_t *> buffers_;
  std::vector<int64_t> offsets_;
  const void *const *holders_;
  int64_t num_rows_;
};

/// Experimental scalar backend for Gandiva expressions.
/// Make/MakeParameterized compile on every call; callers reuse the returned
/// object, or use ScalarPredicateCache for automatic cross-query reuse. bind
/// validates array metadata and snapshots parameters without scanning input
/// rows. Its storage is proportional to columns and parameter payloads.
///
/// Supported values are bool, fixed-width integer/float, utf8 and binary types.
/// Variable-width parameters borrow no caller-owned scalar memory. The explicit
/// function subset includes comparisons, add/subtract/multiply, bitwise ops,
/// NOT and null/boolean tests, with Gandiva's arithmetic semantics. Function
/// nodes needing context, function holders, errors, or arena allocations
/// are rejected. AND/OR/NOT and IF preserve intermediate validity; the final
/// predicate maps an invalid result to false. Numeric IN follows Gandiva 21:
/// NULL IN (...) is valid false, so NOT(IN(NULL)) is true (not SQL IN
/// semantics).
class ScalarPredicate : public std::enable_shared_from_this<ScalarPredicate> {
 public:
  static arrow::Result<std::shared_ptr<ScalarPredicate>> Make(
      std::shared_ptr<arrow::Schema> schema,
      std::shared_ptr<gandiva::Condition> condition);

  /// Parameter fields are referenced with ordinary Gandiva FieldNodes. Their
  /// names must be unique and disjoint from data fields. The expression is
  /// validated against data_schema + parameter_schema, while bind accepts only
  /// data_schema in its RecordBatch. Parameters are broadcast scalars, never
  /// accessed at the candidate row. Make() is the zero-parameter shorthand.
  static arrow::Result<std::shared_ptr<ScalarPredicate>> MakeParameterized(
      std::shared_ptr<arrow::Schema> data_schema,
      std::shared_ptr<gandiva::Condition> condition,
      std::shared_ptr<arrow::Schema> parameter_schema);

  ~ScalarPredicate();

  arrow::Result<std::shared_ptr<BoundScalarPredicate>> bind(
      std::shared_ptr<arrow::RecordBatch> batch) const;

  /// Params are ordered by parameter_schema, require exact Arrow types, and
  /// are copied into bound-owned slots. Later caller mutations do not change
  /// the binding. Typed NULL is allowed only for nullable parameter fields.
  /// Owners keep immutable query helper state referenced by parameters alive.
  arrow::Result<std::shared_ptr<BoundScalarPredicate>> bind(
      std::shared_ptr<arrow::RecordBatch> batch,
      const std::vector<std::shared_ptr<arrow::Scalar>> &params,
      std::vector<std::shared_ptr<const void>> owners = {}) const;

  const std::shared_ptr<arrow::Schema> &parameter_schema() const {
    return parameter_schema_;
  }
  int num_parameters() const;

  /// Immutable metadata for building additional bindings from validated Arrow
  /// arrays without redoing schema lookup. Callers supplying raw buffers must
  /// retain the owning data and parameter binding, preserve field types, and
  /// populate each data/validity/variable-offset offset with its ArrayData
  /// slice offset. Variable-width data buffers contain bytes; offsets buffers
  /// contain the int32 indices into those bytes.
  const std::vector<ScalarDataBufferBinding> &data_buffer_bindings() const {
    return data_buffer_bindings_;
  }

  gandiva::ScalarEvalFunction function() const {
    return function_;
  }
  const std::string &dump_ir() const;

 private:
  ScalarPredicate(std::shared_ptr<arrow::Schema> schema,
                  std::shared_ptr<arrow::Schema> parameter_schema,
                  std::unique_ptr<gandiva::ScalarGenerator> generator,
                  gandiva::ScalarEvalFunction function,
                  std::vector<ScalarDataBufferBinding> data_buffer_bindings);

  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<arrow::Schema> parameter_schema_;
  std::unique_ptr<gandiva::ScalarGenerator> generator_;
  gandiva::ScalarEvalFunction function_;
  std::vector<ScalarDataBufferBinding> data_buffer_bindings_;
};

}  // namespace zvec::sqlengine::scalar_jit
