// Copyright 2026 The Stile Authors
// SPDX-License-Identifier: Apache-2.0
// Adapted from Stile v0.1.0 for zvec.

#pragma once

#include <unordered_map>
#include "db/sqlengine/stile/predicate.h"

namespace zvec::stile::internal {

class CompileChecks {
 public:
  explicit CompileChecks(const CompileOptions &options)
      : options_(options),
        values_(options.max_set_values),
        chunks_(options.max_set_chunks),
        bytes_(options.max_literal_bytes) {}
  arrow::Status check_tree(const arrow::compute::Expression &root);
  arrow::Status check_set_chunk(const arrow::ArrayData &chunk) const;
  const ListMembership *membership_for(
      const arrow::compute::Expression::Call &call) const;

 private:
  arrow::Status check_node(const arrow::compute::Expression &expr);
  arrow::Status check_membership(
      const arrow::compute::Expression::Call &call,
      const FunctionRegistry::DescribeListMembership &describe);
  arrow::Status charge_set(const arrow::Datum &set);
  const CompileOptions &options_;
  size_t values_, chunks_, bytes_;
  std::unordered_map<const arrow::compute::Expression::Call *, ListMembership>
      memberships_;
};
}  // namespace zvec::stile::internal
