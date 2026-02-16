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

#pragma once

#include <zvec/db/doc.h>
#include <zvec/db/schema.h>
#include <zvec/db/status.h>
#include <zvec/db/type.h>
#include "duckdb/common/types.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb.hpp"

namespace duckdb {

LogicalType ZvecTypeToDuckDB(zvec::DataType type);

void DocFieldToDuckDB(const zvec::Doc &doc, const std::string &field_name,
                      zvec::DataType type, Vector &result, idx_t row_idx);

std::string FloatArrayToQueryVector(const vector<float> &floats);

template <typename T>
T UnwrapOrThrow(zvec::Result<T> result, const std::string &context = "") {
  if (!result.has_value()) {
    auto &err = result.error();
    throw IOException("zvec: %s%s", context.empty() ? "" : context + ": ",
                      err.message());
  }
  return std::move(result).value();
}

inline void ThrowIfError(const zvec::Status &status,
                         const std::string &context = "") {
  if (!status.ok()) {
    throw IOException("zvec: %s%s", context.empty() ? "" : context + ": ",
                      status.message());
  }
}

}  // namespace duckdb
