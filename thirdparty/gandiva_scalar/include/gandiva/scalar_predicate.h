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

namespace gandiva {

class Condition;
class LLVMGenerator;
class ScalarPredicate;

// The generated function evaluates exactly one row. It has no output vector,
// selection vector, execution context, or batch-sized temporary storage.
using ScalarEvalFunction = bool (*)(const uint8_t* const* buffers,
                                   const int64_t* offsets,
                                   const void* const* holders, int64_t row);

// A snapshot of one fixed-width runtime parameter. Value bytes use native
// Arrow primitive representation; booleans and validity use bit zero. Slots
// are per binding and never replicated to match the RecordBatch row count.
struct alignas(8) ScalarParameterSlot {
  uint8_t data[8] = {};
  uint8_t validity = 1;
};

// Compile-time mapping for a referenced data column. Parameter fields are
// excluded. Indices address the same flat buffer/offset arrays exposed by a
// BoundScalarPredicate; column_index addresses the original data schema.
struct ScalarDataBufferBinding {
  int column_index;
  int data_buffer_index;
  int validity_buffer_index;
};

/// An immutable binding of one compiled predicate to an Arrow RecordBatch.
/// Keeps both the input buffers and executable code alive. Concurrent Evaluate
/// calls are supported, provided callers do not mutate the input Arrow buffers.
class BoundScalarPredicate {
 public:
  // Flattened buffer pointers may point into parameter_slots_. Prevent copies
  // or moves that could detach those pointers from their owning allocation.
  BoundScalarPredicate(const BoundScalarPredicate&) = delete;
  BoundScalarPredicate& operator=(const BoundScalarPredicate&) = delete;
  BoundScalarPredicate(BoundScalarPredicate&&) = delete;
  BoundScalarPredicate& operator=(BoundScalarPredicate&&) = delete;

  /// Hot path: row MUST satisfy 0 <= row < num_rows(). No bounds checking,
  /// allocation, Arrow object lookup, locking, or batch scan is performed here.
  bool Evaluate(int64_t row) const {
    return function_(buffers_.data(), offsets_.data(), holders_, row);
  }

  arrow::Result<bool> EvaluateChecked(int64_t row) const;

  int64_t num_rows() const { return num_rows_; }
  ScalarEvalFunction function() const { return function_; }
  const uint8_t* const* buffers() const { return buffers_.data(); }
  const int64_t* offsets() const { return offsets_.data(); }
  const void* const* holders() const { return holders_; }
  size_t num_buffers() const { return buffers_.size(); }

 private:
  friend class ScalarPredicate;
  BoundScalarPredicate(std::shared_ptr<const ScalarPredicate> compiled,
                       std::shared_ptr<arrow::RecordBatch> batch,
                       ScalarEvalFunction function,
                       std::vector<ScalarParameterSlot> parameter_slots,
                       std::vector<const uint8_t*> buffers,
                       std::vector<int64_t> offsets,
                       const void* const* holders, int64_t num_rows);

  std::shared_ptr<const ScalarPredicate> compiled_;
  std::shared_ptr<arrow::RecordBatch> batch_;
  ScalarEvalFunction function_;
  std::vector<ScalarParameterSlot> parameter_slots_;
  std::vector<const uint8_t*> buffers_;
  std::vector<int64_t> offsets_;
  const void* const* holders_;
  int64_t num_rows_;
};

/// Experimental scalar backend for Gandiva expressions.
/// Make/MakeParameterized compile on every call; callers reuse the returned
/// object, or use ScalarPredicateCache for automatic cross-query reuse. Bind
/// performs metadata work only, with storage proportional to columns/parameters.
///
/// Supported values are bool and fixed-width integer/float types. The explicit
/// function subset includes comparisons, add/subtract/multiply, bitwise ops,
/// NOT and null/boolean tests, with Gandiva's arithmetic semantics. Function
/// nodes needing context, function holders, errors, or variable-width values
/// are rejected. AND/OR/NOT and IF preserve intermediate validity; the final
/// predicate maps an invalid result to false. Numeric IN follows Gandiva 21:
/// NULL IN (...) is valid false, so NOT(IN(NULL)) is true (not SQL IN semantics).
class ScalarPredicate : public std::enable_shared_from_this<ScalarPredicate> {
 public:
  static arrow::Result<std::shared_ptr<ScalarPredicate>> Make(
      std::shared_ptr<arrow::Schema> schema,
      std::shared_ptr<Condition> condition);

  /// Parameter fields are referenced with ordinary Gandiva FieldNodes. Their
  /// names must be unique and disjoint from data fields. The expression is
  /// validated against data_schema + parameter_schema, while Bind accepts only
  /// data_schema in its RecordBatch. Parameters are broadcast scalars, never
  /// accessed at the candidate row. Make() is the zero-parameter shorthand.
  static arrow::Result<std::shared_ptr<ScalarPredicate>> MakeParameterized(
      std::shared_ptr<arrow::Schema> data_schema,
      std::shared_ptr<Condition> condition,
      std::shared_ptr<arrow::Schema> parameter_schema);

  ~ScalarPredicate();

  arrow::Result<std::shared_ptr<BoundScalarPredicate>> Bind(
      std::shared_ptr<arrow::RecordBatch> batch) const;

  /// Params are ordered by parameter_schema, require exact Arrow types, and
  /// are copied into bound-owned slots. Later caller mutations do not change
  /// the binding. Typed NULL is allowed only for nullable parameter fields.
  arrow::Result<std::shared_ptr<BoundScalarPredicate>> Bind(
      std::shared_ptr<arrow::RecordBatch> batch,
      const std::vector<std::shared_ptr<arrow::Scalar>>& params) const;

  const std::shared_ptr<arrow::Schema>& parameter_schema() const {
    return parameter_schema_;
  }
  int num_parameters() const;

  /// Immutable metadata for building additional bindings from validated Arrow
  /// arrays without redoing schema lookup. Callers supplying raw buffers must
  /// retain the owning data and parameter binding, preserve field types, and
  /// populate each data/validity offset with its ArrayData slice offset.
  const std::vector<ScalarDataBufferBinding>& data_buffer_bindings() const {
    return data_buffer_bindings_;
  }

  ScalarEvalFunction function() const { return function_; }
  const std::string& DumpIR() const;

 private:
  ScalarPredicate(std::shared_ptr<arrow::Schema> schema,
                  std::shared_ptr<arrow::Schema> parameter_schema,
                  std::unique_ptr<LLVMGenerator> generator,
                  ScalarEvalFunction function,
                  std::vector<ScalarDataBufferBinding> data_buffer_bindings);

  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<arrow::Schema> parameter_schema_;
  std::unique_ptr<LLVMGenerator> generator_;
  ScalarEvalFunction function_;
  std::vector<ScalarDataBufferBinding> data_buffer_bindings_;
};

}  // namespace gandiva
