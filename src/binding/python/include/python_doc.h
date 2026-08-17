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
// limitations under the License.#pragma once

#include <pybind11/pybind11.h>
#include <zvec/db/doc.h>
#include <zvec/db/schema.h>

namespace py = pybind11;

namespace zvec {

class ZVecPyDoc {
 public:
  ZVecPyDoc() = delete;

 public:
  static void Initialize(py::module_ &m);

  // Materialize a single doc into (id, score, fields, vectors) following the
  // collection schema. Shared by the per-doc `get_all` binding and the batch
  // materialization path in the collection DQL bindings. Requires the GIL.
  static py::tuple doc_to_tuple(Doc &self, const CollectionSchema &schema);

  // Same as doc_to_tuple but takes the pre-resolved forward/vector field lists
  // directly, so batch materialization can resolve them once per batch instead
  // of once per doc. Requires the GIL.
  static py::tuple doc_to_tuple_with_fields(
      Doc &self, const FieldSchemaPtrList &forward_fields,
      const FieldSchemaPtrList &vector_fields);

  // Convert a single Doc field value into a Python object according to its
  // DataType. Shared by the per-field `get_any` binding and `doc_to_tuple`.
  // Requires the GIL.
  static py::object doc_value_to_py(Doc &self, const std::string &field,
                                    DataType type);

 private:
  static void bind_doc_operator(py::module_ &m);
  static void bind_doc(py::module_ &m);
};

}  // namespace zvec
