// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#include "db/sqlengine/stile/functions.h"
#include <utility>

namespace zvec::stile {

arrow::Status FunctionRegistry::add_list_membership(
    const std::string &function_name, DescribeListMembership describe) try {
  if (!describe) return arrow::Status::Invalid("Missing description");
  if (!list_membership_.emplace(function_name, std::move(describe)).second) {
    return arrow::Status::KeyError("Function already registered: ",
                                   function_name);
  }
  return arrow::Status::OK();
} catch (const std::bad_alloc &) {
  return arrow::Status::OutOfMemory("stile: registering a function");
} catch (const std::length_error &e) {
  return arrow::Status::CapacityError("stile: ", e.what());
}

const FunctionRegistry::DescribeListMembership *
FunctionRegistry::find_list_membership(const std::string &function_name) const {
  const auto found = list_membership_.find(function_name);
  return found == list_membership_.end() ? nullptr : &found->second;
}

}  // namespace zvec::stile
