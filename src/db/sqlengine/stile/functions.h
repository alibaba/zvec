// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <arrow/array.h>
#include <arrow/compute/expression.h>
#include <arrow/result.h>
#include <arrow/status.h>

namespace zvec::stile {

// Does a list contain any (or all) of a set of values?
//
// A NULL list gives NULL. Otherwise elements and set values compare with
// operator== (NaN equals nothing, -0 equals +0); binary values compare
// bytewise. `kAll` with an empty set is true, `kAny` with one is false.
struct ListMembership {
  enum class Mode { kAny, kAll };
  // How NULL elements and NULL set values take part.
  enum class Nulls {
    kSkip,         // ignored
    kStoredValue,  // compared by the value stored in their slot
  };
  Mode mode = Mode::kAny;
  Nulls nulls = Nulls::kSkip;
  // The set; its type must equal the list's element type.
  std::shared_ptr<arrow::Array> values;
};

// Maps application-defined Arrow functions onto built-in evaluators.
//
// A Predicate only compiles functions it knows. Applications that register
// their own Arrow compute functions describe them here so the same calls
// evaluate per row. Not thread-safe while being modified; read-only use
// from many threads is safe.
class FunctionRegistry {
 public:
  // Describes one call (with a single list argument) as list membership.
  // Return NotImplemented to keep the whole expression on Arrow. Compile()
  // propagates other status codes to the caller.
  using DescribeListMembership = std::function<arrow::Result<ListMembership>(
      const arrow::compute::Expression::Call &)>;

  arrow::Status add_list_membership(const std::string &function_name,
                                    DescribeListMembership describe);
  const DescribeListMembership *find_list_membership(
      const std::string &function_name) const;

 private:
  std::unordered_map<std::string, DescribeListMembership> list_membership_;
};

}  // namespace zvec::stile
