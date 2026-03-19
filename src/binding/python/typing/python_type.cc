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

#include "python_type.h"

namespace zvec {

void ZVecPyTyping::Initialize(pybind11::module_ &parent) {
  auto m = parent.def_submodule(
      "typing", "This module contains the basic data types of Zvec");
  // binding base types
  bind_datatypes(m);
  bind_index_types(m);
  bind_metric_types(m);
  bind_quantize_types(m);
  bind_status(m);
}

void ZVecPyTyping::bind_datatypes(pybind11::module_ &m) {
  py::enum_<DataType>(m, "DataType", R"pbdoc(
Enumeration of supported data types in Zvec.

Includes scalar types, dense/sparse vector types, and array types.

Examples:
    >>> from zvec.typing import DataType
    >>> print(DataType.FLOAT)
    DataType.FLOAT
    >>> print(DataType.VECTOR_FP32)
    DataType.VECTOR_FP32
)pbdoc")
      // field type
      .value("STRING", DataType::STRING)
      .value("BOOL", DataType::BOOL)
      .value("INT32", DataType::INT32)
      .value("INT64", DataType::INT64)
      .value("FLOAT", DataType::FLOAT)
      .value("DOUBLE", DataType::DOUBLE)
      .value("UINT32", DataType::UINT32)
      .value("UINT64", DataType::UINT64)


      // dense vector type
      .value("VECTOR_FP16", DataType::VECTOR_FP16)
      .value("VECTOR_FP32", DataType::VECTOR_FP32)
      .value("VECTOR_FP64", DataType::VECTOR_FP64)
      .value("VECTOR_INT8", DataType::VECTOR_INT8)


      // sparse vector type
      .value("SPARSE_VECTOR_FP32", DataType::SPARSE_VECTOR_FP32)
      .value("SPARSE_VECTOR_FP16", DataType::SPARSE_VECTOR_FP16)


      // array type [not support bool/bytes]
      .value("ARRAY_STRING", DataType::ARRAY_STRING)
      .value("ARRAY_INT32", DataType::ARRAY_INT32)
      .value("ARRAY_INT64", DataType::ARRAY_INT64)
      .value("ARRAY_FLOAT", DataType::ARRAY_FLOAT)
      .value("ARRAY_DOUBLE", DataType::ARRAY_DOUBLE)
      .value("ARRAY_BOOL", DataType::ARRAY_BOOL)
      .value("ARRAY_UINT32", DataType::ARRAY_UINT32)
      .value("ARRAY_UINT64", DataType::ARRAY_UINT64)


      // non support
      // .value("BINARY",    DataType::BINARY)
      // .value("ARRAY_BINARY", DataType::ARRAY_BINARY)
      // .value("VECTOR_INT4",    DataType::VECTOR_INT4)
      // .value("VECTOR_INT16",   DataType::VECTOR_INT16)
      // .value("VECTOR_BINARY32", DataType::VECTOR_BINARY32)
      // .value("VECTOR_BINARY64", DataType::VECTOR_BINARY64)
      // .value("UNDEFINED", DataType::UNDEFINED)
      ;
}

void ZVecPyTyping::bind_index_types(pybind11::module_ &m) {
  py::enum_<IndexType>(m, "IndexType", R"pbdoc(
Enumeration of supported index types in Zvec.

Examples:
    >>> from zvec.typing import IndexType
    >>> print(IndexType.HNSW)
    IndexType.HNSW
)pbdoc")
      .value("UNDEFINED", IndexType::UNDEFINED)
      .value("HNSW", IndexType::HNSW)
      .value("HNSW_RABITQ", IndexType::HNSW_RABITQ)
      .value("IVF", IndexType::IVF)
      .value("FLAT", IndexType::FLAT)
      .value("INVERT", IndexType::INVERT);
}

void ZVecPyTyping::bind_metric_types(pybind11::module_ &m) {
  py::enum_<MetricType>(m, "MetricType", R"pbdoc(
Enumeration of supported distance/similarity metrics.

- COSINE: Cosine similarity.
- IP: Inner product (dot product).
- L2: Euclidean distance (L2 norm).

Examples:
    >>> from zvec.typing import MetricType
    >>> print(MetricType.COSINE)
    MetricType.COSINE
)pbdoc")
      .value("COSINE", MetricType::COSINE)
      .value("IP", MetricType::IP)
      .value("L2", MetricType::L2);
}

void ZVecPyTyping::bind_quantize_types(py::module_ &m) {
  py::enum_<QuantizeType>(m, "QuantizeType", R"pbdoc(
Enumeration of supported quantization types for vector compression.

Examples:
    >>> from zvec.typing import QuantizeType
    >>> print(QuantizeType.INT8)
    QuantizeType.INT8
)pbdoc")
      .value("UNDEFINED", QuantizeType::UNDEFINED)
      .value("FP16", QuantizeType::FP16)
      .value("INT8", QuantizeType::INT8)
      .value("INT4", QuantizeType::INT4)
      .value("RABITQ", QuantizeType::RABITQ);
}

void ZVecPyTyping::bind_status(py::module_ &m) {
  // bind status code
  py::enum_<StatusCode>(m, "StatusCode", R"pbdoc(
Enumeration of possible status codes for Zvec operations.

Used by the `Status` class to indicate success or failure reason.
)pbdoc")
      .value("OK", StatusCode::OK)
      .value("NOT_FOUND", StatusCode::NOT_FOUND)
      .value("ALREADY_EXISTS", StatusCode::ALREADY_EXISTS)
      .value("INVALID_ARGUMENT", StatusCode::INVALID_ARGUMENT)
      .value("PERMISSION_DENIED", StatusCode::PERMISSION_DENIED)
      .value("FAILED_PRECONDITION", StatusCode::FAILED_PRECONDITION)
      .value("RESOURCE_EXHAUSTED", StatusCode::RESOURCE_EXHAUSTED)
      .value("UNAVAILABLE", StatusCode::UNAVAILABLE)
      .value("INTERNAL_ERROR", StatusCode::INTERNAL_ERROR)
      .value("NOT_SUPPORTED", StatusCode::NOT_SUPPORTED)
      .value("UNKNOWN", StatusCode::UNKNOWN);

  // bind status
  py::class_<Status>(m, "Status", R"pbdoc(
Represents the outcome of a Zvec operation.

A `Status` object is either OK (success) or carries an error code and message.

Examples:
    >>> from zvec.typing import Status, StatusCode
    >>> s = Status()
    >>> print(s.ok())
    True
    >>> s = Status(StatusCode.INVALID_ARGUMENT, "Field not found")
    >>> print(s.code() == StatusCode.INVALID_ARGUMENT)
    True
    >>> print(s.message())
    Field not found
)pbdoc")
      .def(py::init<>())
      .def(py::init<StatusCode, const std::string &>(), py::arg("code"),
           py::arg("message") = "", R"pbdoc(
Construct a status with the given code and optional message.

Args:
    code (StatusCode): The status code.
    message (str, optional): Error message. Defaults to empty string.
)pbdoc")
      .def("ok", &Status::ok, "bool: Returns True if the status is OK.")
      .def("code", &Status::code, "StatusCode: Returns the status code.")
      .def("message", &Status::message,
           "str: Returns the error message (may be empty).")
      .def_static("OK", &Status::OK, "Create an OK status.")
      .def_static(
          "InvalidArgument",
          [](const std::string &msg) { return Status::InvalidArgument(msg); },
          py::arg("message"))
      .def_static(
          "NotFound",
          [](const std::string &msg) { return Status::NotFound(msg); },
          py::arg("message"))
      .def_static(
          "AlreadyExists",
          [](const std::string &msg) { return Status::AlreadyExists(msg); },
          py::arg("message"))
      .def_static(
          "InternalError",
          [](const std::string &msg) { return Status::InternalError(msg); },
          py::arg("message"))
      .def_static(
          "PermissionDenied",
          [](const std::string &msg) { return Status::PermissionDenied(msg); },
          py::arg("message"))
      .def("__eq__", [](const Status &self,
                        const Status &other) { return self == other; })
      .def("__ne__", [](const Status &self,
                        const Status &other) { return self != other; })
      .def("__repr__", [](const Status &self) {
        std::string result =
            "{"
            "\"code\":" +
            std::to_string(static_cast<int>(self.code()));

        if (!self.message().empty()) {
          result += ", \"message\":\"" + self.message() + "\"";
        }

        result += "}";
        return result;
      });
}

}  // namespace zvec