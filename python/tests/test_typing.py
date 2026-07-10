# Copyright 2025-present the zvec project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import pytest
from zvec import (
    DataType,
    IndexType,
    IOBackendType,
    MetricType,
    QuantizeType,
    Status,
    StatusCode,
)


# ----------------------------
# Enum Test Case
# ----------------------------
@pytest.mark.parametrize(
    "member, name",
    [
        (DataType.FLOAT, "FLOAT"),
        (IndexType.HNSW, "HNSW"),
        (IOBackendType.PREAD, "PREAD"),
        (MetricType.COSINE, "COSINE"),
        (QuantizeType.INT8, "INT8"),
        (StatusCode.OK, "OK"),
    ],
)
def test_enum_names(member, name):
    assert member.name == name


@pytest.mark.parametrize(
    "member, value",
    [
        (DataType.FLOAT, 8),
        (IndexType.HNSW, 1),
        (IOBackendType.PREAD, 0),
        (MetricType.COSINE, 3),
        (QuantizeType.INT8, 2),
        (StatusCode.OK, 0),
    ],
)
def test_enum_values(member, value):
    assert member.value == value


@pytest.mark.parametrize("member", ["L2", "IP", "COSINE"])
def test_metric_type_has_member(member):
    assert member in MetricType.__members__


@pytest.mark.parametrize(
    "member",
    [
        "STRING",
        "BOOL",
        "INT32",
        "INT64",
        "FLOAT",
        "DOUBLE",
        "UINT32",
        "UINT64",
        "VECTOR_FP16",
        "VECTOR_FP32",
        "VECTOR_FP64",
        "VECTOR_INT8",
        "SPARSE_VECTOR_FP32",
        "SPARSE_VECTOR_FP16",
        "ARRAY_STRING",
        "ARRAY_INT32",
        "ARRAY_INT64",
        "ARRAY_FLOAT",
        "ARRAY_DOUBLE",
        "ARRAY_BOOL",
        "ARRAY_UINT32",
        "ARRAY_UINT64",
    ],
)
def test_data_type_has_member(member):
    assert member in DataType.__members__


@pytest.mark.parametrize(
    "member",
    [
        "UNDEFINED",
        "HNSW",
        "IVF",
        "FLAT",
        "HNSW_RABITQ",
        "DISKANN",
        "VAMANA",
        "INVERT",
        "FTS",
    ],
)
def test_index_type_has_member(member):
    assert member in IndexType.__members__


@pytest.mark.parametrize("member", ["PREAD", "LIBAIO"])
def test_io_backend_type_has_member(member):
    assert member in IOBackendType.__members__


@pytest.mark.parametrize("member", ["FP16", "INT8", "INT4", "UNDEFINED"])
def test_quantize_type_has_member(member):
    assert member in QuantizeType.__members__


@pytest.mark.parametrize(
    "member",
    [
        "OK",
        "UNKNOWN",
        "NOT_FOUND",
        "ALREADY_EXISTS",
        "INVALID_ARGUMENT",
        "PERMISSION_DENIED",
        "FAILED_PRECONDITION",
        "RESOURCE_EXHAUSTED",
        "UNAVAILABLE",
        "INTERNAL_ERROR",
        "NOT_SUPPORTED",
    ],
)
def test_status_code_has_member(member):
    assert member in StatusCode.__members__


# ----------------------------
# Status Test Case
# ----------------------------
class TestStatus:
    def test_status_code(self):
        status = Status(StatusCode.OK)
        assert status.code() == StatusCode.OK

    def test_status_message(self):
        status = Status(StatusCode.OK, "OK")
        assert status.message() == "OK"

        status = Status(StatusCode.NOT_FOUND, "Not Found")
        assert status.message() == "Not Found"

    def test_status_ok(self):
        status = Status(StatusCode.OK)
        assert status.ok()
