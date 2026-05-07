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

#include "python_doc.h"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace zvec {

template <typename T>
T checked_cast(const py::object &obj, const std::string &field,
               const std::string &expected_type) {
  try {
    return obj.cast<T>();
  } catch (const py::cast_error &e) {
    std::string actual_type = std::string(py::str(py::type::of(obj)));
    std::string msg = "Field '" + field + "': expected " + expected_type +
                      ", got " + actual_type;
    throw py::type_error(msg);
  }
}

void ZVecPyDoc::Initialize(pybind11::module_ &m) {
  bind_doc_operator(m);
  bind_doc(m);
}


void ZVecPyDoc::bind_doc_operator(py::module_ &m) {
  py::enum_<Operator>(m, "_DocOp")
      .value("INSERT", Operator::INSERT)
      .value("UPDATE", Operator::UPDATE)
      .value("DELETE", Operator::DELETE)
      .value("UPSERT", Operator::UPSERT);
}


void ZVecPyDoc::bind_doc(py::module_ &m) {
  // binding doc
  py::class_<Doc, Doc::Ptr> doc(m, "_Doc");

  doc.def(py::init([]() { return std::make_shared<Doc>(); }))
      .def("set_pk", &Doc::set_pk)
      .def("pk", &Doc::pk)
      .def("set_score", &Doc::set_score)
      .def("score", &Doc::score)
      .def("has_field", &Doc::has)
      .def("field_names", &Doc::field_names)
      .def(py::pickle(
          [](const Doc &d) {
            std::vector<uint8_t> data = d.serialize();
            return py::bytes(reinterpret_cast<const char *>(data.data()),
                             data.size());
          },
          [](py::bytes b) {
            py::buffer_info info(py::buffer(b).request());
            const uint8_t *buf = reinterpret_cast<const uint8_t *>(info.ptr);
            size_t size = static_cast<size_t>(info.size);
            Doc::Ptr d = Doc::deserialize(buf, size);
            if (!d) throw std::runtime_error("Failed to unpickle Doc");
            return d;
          }));


  // binding doc set field
  doc.def(
      "set_any",
      [](Doc &self, const std::string &field, const FieldSchema &field_schema,
         const py::object &obj) -> bool {
        if (obj.is_none()) {
          if (field_schema.nullable()) {
            self.set_null(field);
            return true;
          }
          throw py::value_error("Field '" + field +
                                "': expected non-nullable type");
        }
        switch (field_schema.data_type()) {
          // base datatypes
          case DataType::STRING:
            return self.set(field,
                            checked_cast<std::string>(obj, field, "STRING"));
          case DataType::BOOL:
            return self.set(field, checked_cast<bool>(obj, field, "BOOL"));
          case DataType::INT32:
            return self.set(field, checked_cast<int32_t>(obj, field, "INT32"));
          case DataType::INT64:
            return self.set(field, checked_cast<int64_t>(obj, field, "INT64"));
          case DataType::UINT32:
            return self.set(field,
                            checked_cast<uint32_t>(obj, field, "UINT32"));
          case DataType::UINT64:
            return self.set(field,
                            checked_cast<uint64_t>(obj, field, "UINT64"));
          case DataType::FLOAT:
            return self.set(field, checked_cast<float>(obj, field, "FLOAT"));
          case DataType::DOUBLE:
            return self.set(field, checked_cast<double>(obj, field, "DOUBLE"));

          // array datatypes
          case DataType::ARRAY_STRING:
            return self.set(field, checked_cast<std::vector<std::string>>(
                                       obj, field, "ARRAY_STRING"));
          case DataType::ARRAY_BOOL:
            return self.set(field, checked_cast<std::vector<bool>>(
                                       obj, field, "ARRAY_BOOL"));
          case DataType::ARRAY_INT32:
            return self.set(field, checked_cast<std::vector<int32_t>>(
                                       obj, field, "ARRAY_INT32"));
          case DataType::ARRAY_UINT32:
            return self.set(field, checked_cast<std::vector<uint32_t>>(
                                       obj, field, "ARRAY_UINT32"));
          case DataType::ARRAY_INT64:
            return self.set(field, checked_cast<std::vector<int64_t>>(
                                       obj, field, "ARRAY_INT64"));
          case DataType::ARRAY_UINT64:
            return self.set(field, checked_cast<std::vector<uint64_t>>(
                                       obj, field, "ARRAY_UINT64"));
          case DataType::ARRAY_FLOAT:
            return self.set(field, checked_cast<std::vector<float>>(
                                       obj, field, "ARRAY_FLOAT"));
          case DataType::ARRAY_DOUBLE:
            return self.set(field, checked_cast<std::vector<double>>(
                                       obj, field, "ARRAY_DOUBLE"));

          // dense vector datatypes
          case DataType::VECTOR_FP16: {
            const auto value = checked_cast<py::list>(
                obj, field, "VECTOR_FP16 (list of numbers)");
            std::vector<ailego::Float16> new_value;
            new_value.reserve(value.size());
            for (const auto &item : value) {
              try {
                new_value.emplace_back(item.cast<float>());
              } catch (const py::cast_error &e) {
                throw py::type_error("Vector '" + field +
                                     "': expected VECTOR_FP16, got " +
                                     std::string(py::str(py::type::of(obj))));
              }
            }
            return self.set(field, new_value);
          }
          case DataType::VECTOR_FP32:
            return self.set(field, checked_cast<std::vector<float>>(
                                       obj, field, "VECTOR_FP32"));
          case DataType::VECTOR_FP64:
            return self.set(field, checked_cast<std::vector<double>>(
                                       obj, field, "VECTOR_FP64"));
          case DataType::VECTOR_INT8:
            return self.set(field, checked_cast<std::vector<int8_t>>(
                                       obj, field, "VECTOR_INT8"));

          // sparse vector datatypes
          case DataType::SPARSE_VECTOR_FP32: {
            const auto sparse_dict =
                checked_cast<py::dict>(obj, field, "SPARSE_VECTOR_FP32 (dict)");
            std::vector<uint32_t> indices;
            std::vector<float> values;
            for (const auto &item : sparse_dict) {
              try {
                indices.push_back(item.first.cast<uint32_t>());
                values.push_back(item.second.cast<float>());
              } catch (const py::cast_error &e) {
                throw py::type_error(
                    "Vector '" + field +
                    "': sparse vector key/value must be (uint32, float), "
                    "got key=" +
                    std::string(py::str(py::type::of(item.first))) +
                    ", value=" +
                    std::string(py::str(py::type::of(item.second))));
              }
            }
            const std::pair<std::vector<uint32_t>, std::vector<float>>
                sparse_vector{std::move(indices), std::move(values)};
            return self.set(field, sparse_vector);
          }
          case DataType::SPARSE_VECTOR_FP16: {
            const auto sparse_dict =
                checked_cast<py::dict>(obj, field, "SPARSE_VECTOR_FP16 (dict)");
            std::vector<uint32_t> indices;
            std::vector<ailego::Float16> values;
            for (const auto &item : sparse_dict) {
              try {
                indices.push_back(item.first.cast<uint32_t>());
                values.push_back(ailego::Float16(item.second.cast<float>()));
              } catch (const py::cast_error &e) {
                throw py::type_error(
                    "Field '" + field +
                    "': sparse vector key/value must be (uint32, float), "
                    "got key=" +
                    std::string(py::str(py::type::of(item.first))) +
                    ", value=" +
                    std::string(py::str(py::type::of(item.second))));
              }
            }
            const std::pair<std::vector<uint32_t>, std::vector<ailego::Float16>>
                sparse_vector{std::move(indices), std::move(values)};
            return self.set(field, sparse_vector);
          }
          default:
            throw py::type_error("Unsupported type for field: " + field);
        }
      });

  // binding doc get field
  doc.def(
      "get_any",
      [](Doc &self, const std::string &field,
         const DataType &type) -> py::object {
        switch (type) {
          // base datatypes
          case DataType::STRING:
            return py::cast(self.get<std::string>(field));
          case DataType::BOOL:
            return py::cast(self.get<bool>(field));
          case DataType::INT32:
            return py::cast(self.get<int32_t>(field));
          case DataType::UINT32:
            return py::cast(self.get<uint32_t>(field));
          case DataType::INT64:
            return py::cast(self.get<int64_t>(field));
          case DataType::UINT64:
            return py::cast(self.get<uint64_t>(field));
          case DataType::FLOAT:
            return py::cast(self.get<float>(field));
          case DataType::DOUBLE:
            return py::cast(self.get<double>(field));

          // array datatypes
          case DataType::ARRAY_STRING:
            return py::cast(self.get<std::vector<std::string>>(field));
          case DataType::ARRAY_INT32:
            return py::cast(self.get<std::vector<int32_t>>(field));
          case DataType::ARRAY_INT64:
            return py::cast(self.get<std::vector<int64_t>>(field));
          case DataType::ARRAY_UINT32:
            return py::cast(self.get<std::vector<uint32_t>>(field));
          case DataType::ARRAY_UINT64:
            return py::cast(self.get<std::vector<uint64_t>>(field));
          case DataType::ARRAY_FLOAT:
            return py::cast(self.get<std::vector<float>>(field));
          case DataType::ARRAY_DOUBLE:
            return py::cast(self.get<std::vector<double>>(field));
          case DataType::ARRAY_BOOL:
            return py::cast(self.get<std::vector<bool>>(field));

          // vector datatypes
          case DataType::VECTOR_INT8:
            return py::cast(self.get<std::vector<int8_t>>(field));
          case DataType::VECTOR_FP16: {
            auto value = self.get<std::vector<ailego::Float16>>(field);
            if (value.has_value()) {
              std::vector<float> new_value;
              new_value.reserve(value.value().size());
              for (auto &item : value.value()) {
                new_value.push_back(static_cast<float>(item));
              }
              return py::cast(new_value);
            }
            return py::none();
          }
          case DataType::VECTOR_FP32:
            return py::cast(self.get<std::vector<float>>(field));
          case DataType::VECTOR_FP64:
            return py::cast(self.get<std::vector<double>>(field));
          case DataType::SPARSE_VECTOR_FP16: {
            auto vector = self.get<
                std::pair<std::vector<uint32_t>, std::vector<ailego::Float16>>>(
                field);
            const auto &indices = vector->first;
            const auto &values = vector->second;
            py::dict d;
            for (size_t i = 0; i < indices.size(); ++i) {
              d[py::int_(indices[i])] =
                  py::float_(static_cast<float>(values[i]));
            }
            return std::move(d);
          }
          case DataType::SPARSE_VECTOR_FP32: {
            auto vector =
                self.get<std::pair<std::vector<uint32_t>, std::vector<float>>>(
                    field);
            const auto &indices = vector->first;
            const auto &values = vector->second;
            py::dict d;
            for (size_t i = 0; i < indices.size(); ++i) {
              d[py::int_(indices[i])] = py::float_(values[i]);
            }
            return std::move(d);
          }
          default:
            throw py::type_error("Unsupported type for field: " + field);
        }
      });
  doc.def(
      "get_all",
      [](Doc &self, const CollectionSchema &schema) -> py::tuple {
        py::tuple result(4);
        // 1. set doc id and score
        result[0] = py::str(self.pk());
        result[1] = py::float_(self.score());

        if (self.is_empty()) {
          result[2] = py::none();
          result[3] = py::none();
          return result;
        }
        // 2. set scalar fields
        py::dict fields;
        for (const auto &field_meta : schema.forward_fields()) {
          const std::string &field = field_meta->name();
          if (!self.has_value(field)) {
            continue;
          }

          try {
            auto val = [&]() -> py::object {
              switch (field_meta->data_type()) {
                // base datatypes
                case DataType::STRING:
                  return py::str(self.get<std::string>(field).value());
                case DataType::BOOL:
                  return py::cast(self.get<bool>(field));
                case DataType::INT32:
                  return py::cast(self.get<int32_t>(field));
                case DataType::UINT32:
                  return py::cast(self.get<uint32_t>(field));
                case DataType::INT64:
                  return py::cast(self.get<int64_t>(field));
                case DataType::UINT64:
                  return py::cast(self.get<uint64_t>(field));
                case DataType::FLOAT:
                  return py::cast(self.get<float>(field));
                case DataType::DOUBLE:
                  return py::cast(self.get<double>(field));

                // array datatypes
                case DataType::ARRAY_STRING:
                  return py::cast(self.get<std::vector<std::string>>(field));
                case DataType::ARRAY_INT32:
                  return py::cast(self.get<std::vector<int32_t>>(field));
                case DataType::ARRAY_INT64:
                  return py::cast(self.get<std::vector<int64_t>>(field));
                case DataType::ARRAY_UINT32:
                  return py::cast(self.get<std::vector<uint32_t>>(field));
                case DataType::ARRAY_UINT64:
                  return py::cast(self.get<std::vector<uint64_t>>(field));
                case DataType::ARRAY_FLOAT:
                  return py::cast(self.get<std::vector<float>>(field));
                case DataType::ARRAY_DOUBLE:
                  return py::cast(self.get<std::vector<double>>(field));
                case DataType::ARRAY_BOOL:
                  return py::cast(self.get<std::vector<bool>>(field));
                default:
                  throw py::type_error("Unsupported type for field: " + field);
              }
            }();
            fields[py::str(field)] = val;
          } catch (const std::exception &e) {
            fields[py::str(field)] = py::none();
          }
        }
        if (!fields.empty()) {
          result[2] = fields;
        } else {
          result[2] = py::none();
        }
        // 3. set vector fields
        py::dict vectors;
        for (const auto &vec_meta : schema.vector_fields()) {
          const std::string &vec = vec_meta->name();
          if (!self.has_value(vec)) continue;

          try {
            auto array = [&]() -> py::object {
              switch (vec_meta->data_type()) {
                case DataType::VECTOR_INT8:
                  return py::cast(self.get<std::vector<int8_t>>(vec));
                case DataType::VECTOR_FP16: {
                  auto value = self.get<std::vector<ailego::Float16>>(vec);
                  if (value.has_value()) {
                    std::vector<float> new_value;
                    new_value.reserve(value.value().size());
                    for (auto &item : value.value()) {
                      new_value.push_back(static_cast<float>(item));
                    }
                    return py::cast(new_value);
                  }
                  return py::none();
                }
                case DataType::VECTOR_FP32:
                  return py::cast(self.get<std::vector<float>>(vec));
                case DataType::VECTOR_FP64:
                  return py::cast(self.get<std::vector<double>>(vec));
                case DataType::SPARSE_VECTOR_FP16: {
                  auto vector =
                      self.get<std::pair<std::vector<uint32_t>,
                                         std::vector<ailego::Float16>>>(vec);
                  const auto &indices = vector->first;
                  const auto &values = vector->second;
                  py::dict d;
                  for (size_t i = 0; i < indices.size(); ++i) {
                    d[py::int_(indices[i])] =
                        py::float_(static_cast<float>(values[i]));
                  }
                  return std::move(d);
                }
                case DataType::SPARSE_VECTOR_FP32: {
                  auto vector = self.get<
                      std::pair<std::vector<uint32_t>, std::vector<float>>>(
                      vec);
                  const auto &indices = vector->first;
                  const auto &values = vector->second;
                  py::dict d;
                  for (size_t i = 0; i < indices.size(); ++i) {
                    d[py::int_(indices[i])] = py::float_(values[i]);
                  }
                  return std::move(d);
                }
                default:
                  throw py::type_error("Unsupported type for field: " + vec);
              }
            }();
            vectors[py::str(vec)] = array;
          } catch (const std::exception &e) {
            vectors[py::str(vec)] = py::none();
          }
        }
        if (!vectors.empty()) {
          result[3] = vectors;
        } else {
          result[3] = py::none();
        }
        return result;
      },
      py::arg("schema"),
      "Get all fields and vectors as a tuple: (id, score, fields, vectors). "
      "Vectors are zero-copy numpy arrays (dense: ndarray, sparse: (indices, "
      "values) tuple).");
}
}  // namespace zvec