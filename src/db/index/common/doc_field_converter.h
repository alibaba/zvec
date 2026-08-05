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
#include <arrow/api.h>
#include <zvec/db/doc.h>
#include <zvec/db/schema.h>
#include <zvec/db/status.h>
#include "db/index/column/vector_column/vector_column_params.h"

namespace zvec {

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
