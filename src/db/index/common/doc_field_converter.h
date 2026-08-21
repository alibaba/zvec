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

// Shared converters that fill Doc fields, used by SegmentImpl, DocIterator
// and the SQL engine so that field-type coverage and null semantics stay in
// one place.
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
//! the list converters and SegmentImpl::Fetch. zvec array fields never store
//! null elements, so no per-element null check is needed.
template <typename ArrowArrayT, typename T>
std::vector<T> ExtractTypedArrayValues(const ArrowArrayT *values) {
  std::vector<T> vec;
  vec.reserve(values->length());
  for (int64_t i = 0; i < values->length(); ++i) {
    vec.emplace_back(values->Value(i));
  }
  return vec;
}

//! Convert a VectorDataBuffer (dense or sparse) fetched from a vector indexer
//! into the corresponding Doc vector field.
Status ConvertVectorDataBufferToDocField(
    const FieldSchema::Ptr &field,
    const vector_column_params::VectorDataBuffer &buf, Doc *doc);

//! Convert every row of an Arrow array into the docs starting at `doc_it`
//! (one doc per row). Type dispatch and null checks happen once per column
//! (null rows leave the field unset). Covers every scalar and ARRAY_*
//! DataType.
Status ConvertArrowColumnToDocFields(const arrow::Array *array,
                                     const FieldSchema &field,
                                     DocPtrList::iterator doc_it);

}  // namespace zvec
