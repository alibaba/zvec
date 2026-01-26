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

#include "python_schema.h"
#include <pybind11/stl.h>
#include <zvec/db/schema.h>
#include <zvec/db/stats.h>

namespace zvec {

void ZVecPySchemas::Initialize(pybind11::module_ &parent) {
  auto m =
      parent.def_submodule("schema", "This module contains the schema of Zvec");

  bind_field_schema(m);
  bind_collection_schema(m);
  bind_collection_stats(m);
}

void ZVecPySchemas::bind_field_schema(py::module_ &m) {
  py::class_<FieldSchema, FieldSchema::Ptr>(m, "_FieldSchema")
      .def(py::init<const std::string &, DataType, uint32_t, bool,
                    const IndexParams::Ptr &>(),
           py::arg("name"), py::arg("data_type"), py::arg("dimension") = 0,
           py::arg("nullable") = false, py::arg("index_param") = nullptr)
      .def_property_readonly("name", &FieldSchema::name)
      .def_property_readonly("data_type", &FieldSchema::data_type)
      .def_property_readonly("nullable", &FieldSchema::nullable)
      .def_property_readonly("dimension", &FieldSchema::dimension)
      .def_property_readonly("is_dense_vector", &FieldSchema::is_dense_vector)
      .def_property_readonly("is_sparse_vector", &FieldSchema::is_sparse_vector)
      .def_property_readonly("index_type",
                             [](const FieldSchema &self) {
                               return self.index_params()
                                          ? self.index_type()
                                          : IndexType::UNDEFINED;
                             })
      .def_property_readonly("index_param",
                             [](const FieldSchema &self) -> py::object {
                               if (self.index_params()) {
                                 return py::cast(self.index_params());
                               }
                               return py::none();
                             })
      .def("__eq__", &FieldSchema::operator==)
      .def("__ne__", &FieldSchema::operator!=)
      .def(py::pickle(
          [](const FieldSchema &self) {
            return py::make_tuple(self.name(), self.data_type(),
                                  self.dimension(), self.nullable(),
                                  self.index_params()
                                      ? py::cast(self.index_params())
                                      : py::none());
          },
          [](py::tuple t) {
            if (t.size() != 5) {
              throw std::runtime_error(
                  "Invalid tuple size for FieldSchema pickle");
            }
            std::string name = t[0].cast<std::string>();
            DataType dtype = t[1].cast<DataType>();
            uint32_t dim = t[2].cast<uint32_t>();
            bool nullable = t[3].cast<bool>();

            IndexParams::Ptr idx_params = nullptr;
            if (!t[4].is_none()) {
              idx_params = t[4].cast<IndexParams::Ptr>();
            }

            return std::make_shared<FieldSchema>(name, dtype, dim, nullable,
                                                 idx_params);
          }));
}

void ZVecPySchemas::bind_collection_schema(py::module_ &m) {
  py::class_<CollectionSchema, CollectionSchema::Ptr>(m, "_CollectionSchema")
      .def(py::init<const std::string &, const FieldSchemaPtrList &>(),
           py::arg("name"), py::arg("fields"),
           "Construct with name and list of fields")
      .def_property_readonly("name", &CollectionSchema::name)
      .def("has_field", &CollectionSchema::has_field, py::arg("field_name"),
           "Check if a field exists.")
      .def(
          "get_field",
          [](const CollectionSchema &self, const std::string &name)
              -> const FieldSchema * { return self.get_field(name); },
          py::arg("field_name"), py::return_value_policy::reference_internal,
          "Get field by name (const pointer), returns None if not found.")
      .def(
          "get_forward_field",
          [](const CollectionSchema &self, const std::string &name)
              -> const FieldSchema * { return self.get_forward_field(name); },
          py::arg("field_name"), py::return_value_policy::reference_internal,
          "Get forward field (used for filtering).")
      .def(
          "get_vector_field",
          [](const CollectionSchema &self, const std::string &name)
              -> const FieldSchema * { return self.get_vector_field(name); },
          py::arg("field_name"), py::return_value_policy::reference_internal,
          "Get vector field by name.")
      .def("fields", &CollectionSchema::fields,
           "Return list of all field schemas.", py::return_value_policy::copy)
      .def("forward_fields", &CollectionSchema::forward_fields,
           "Return list of forward-indexed fields.",
           py::return_value_policy::copy)
      .def("vector_fields", &CollectionSchema::vector_fields,
           "Return list of vector fields.", py::return_value_policy::copy)
      .def("__eq__", &CollectionSchema::operator==)
      .def("__ne__", &CollectionSchema::operator!=)
      .def(py::pickle(
          [](const CollectionSchema &cs) {
            return py::make_tuple(cs.name(), cs.fields(),
                                  cs.max_doc_count_per_segment());
          },
          [](py::tuple t) {
            if (t.size() != 3)
              throw std::runtime_error("Invalid state for CollectionSchema!");

            auto name = t[0].cast<std::string>();
            auto fields = t[1].cast<FieldSchemaPtrList>();
            auto max_docs = t[2].cast<uint64_t>();

            auto cs = std::make_shared<CollectionSchema>(name, fields);
            cs->set_max_doc_count_per_segment(max_docs);
            return cs;
          }));
}

void ZVecPySchemas::bind_collection_stats(py::module_ &m) {
  pybind11::class_<CollectionStats>(m, "CollectionStats")
      .def(pybind11::init<>())
      .def_property_readonly(
          "doc_count", [](const CollectionStats &c) { return c.doc_count; })
      .def_property_readonly(
          "index_completeness",
          [](const CollectionStats &c) { return c.index_completeness; })
      .def("__repr__", [](const CollectionStats &c) {
        std::string map_str = "{";
        bool first = true;
        for (const auto &[k, v] : c.index_completeness) {
          if (!first) map_str += ", ";
          map_str += "\"" + k + "\":" + std::to_string(v);
          first = false;
        }
        map_str += "}";
        return "{\"doc_count\":" + std::to_string(c.doc_count) +
               ", \"index_completeness\":" + map_str + "}";
      });
}

}  // namespace zvec