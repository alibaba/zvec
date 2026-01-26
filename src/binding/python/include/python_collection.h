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
#include <zvec/db/collection.h>

namespace py = pybind11;

namespace zvec {

class ZVecPyCollection {
 public:
  ZVecPyCollection() = delete;

 public:
  static void Initialize(py::module_ &m);

 private:
  static void bind_db_methods(py::class_<Collection, Collection::Ptr> &col);
  static void bind_ddl_methods(py::class_<Collection, Collection::Ptr> &col);
  static void bind_dml_methods(py::class_<Collection, Collection::Ptr> &col);
  static void bind_dql_methods(py::class_<Collection, Collection::Ptr> &col);
};

}  // namespace zvec
