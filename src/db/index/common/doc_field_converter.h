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

// Shared row-level converters that fill Doc fields, used by SegmentImpl,
// DocIterator and the SQL engine so that field-type coverage and null/list
// semantics stay in one place.
#pragma once

#include <memory>
#include <type_traits>
#include <vector>
#include <arrow/api.h>
#include <zvec/db/doc.h>
#include <zvec/db/schema.h>
#include <zvec/db/status.h>
#include "db/index/column/vector_column/vector_column_params.h"

namespace zvec {

//! Extract the values of a typed Arrow array into a std::vector. Shared by
//! the row-level list converter (iterator/SQL-engine path, which starts from
//! a ListArray) and SegmentImpl::Fetch (which holds the inner value array of
//! a ListScalar directly). Inner nulls are skipped; write paths never produce
//! them, so this is only defensive.
template <typename ArrowArrayT, typename T>
std::vector<T> ExtractTypedArrayValues(const ArrowArrayT *values) {
  std::vector<T> vec;
  vec.reserve(values->length());
  // null_count() is O(1); skip per-element validity checks when the array
  // has no nulls (the common case).
  if (values->null_count() == 0) {
    for (int64_t i = 0; i < values->length(); ++i) {
      vec.emplace_back(values->Value(i));
    }
  } else {
    for (int64_t i = 0; i < values->length(); ++i) {
      if (values->IsNull(i)) {
        continue;
      }
      vec.emplace_back(values->Value(i));
    }
  }
  return vec;
}

//! Convert a VectorDataBuffer (dense or sparse) fetched from a vector indexer
//! into the corresponding Doc vector field.
Status ConvertVectorDataBufferToDocField(
    const FieldSchema::Ptr &field,
    const vector_column_params::VectorDataBuffer &buf, Doc *doc);

//! Convert row `row` of an Arrow array into the corresponding Doc scalar or
//! array field. Null rows are skipped (the field is left unset). Covers every
//! scalar DataType (BINARY/STRING/BOOL/INT32/INT64/UINT32/UINT64/FLOAT/DOUBLE)
//! and every ARRAY_* DataType.
Status ConvertArrowRowToDocField(const arrow::Array *array, int64_t row,
                                 const FieldSchema &field, Doc *doc);

}  // namespace zvec
