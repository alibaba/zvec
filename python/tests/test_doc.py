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

import math
import pytest


from zvec._zvec import _Doc
from zvec import FieldSchema, VectorSchema, Doc, DataType


# ----------------------------
# PyDoc Test Case
# ----------------------------
class TestPyDoc:
    def test_default(self):
        Doc(id="1")

    def test_with_single_vector(self):
        doc = Doc(id="1", vectors={"dense": [1, 2, 3]})
        assert doc is not None
        assert doc.id == "1"
        assert doc.vector("dense") == [1, 2, 3]

    def test_with_hybrid_vectors(self):
        doc = Doc(
            id="1", vectors={"dense": [1, 2, 3], "sparse": {1: 1.0, 2: 2.0, 3: 3.0}}
        )
        assert doc is not None
        assert doc.id == "1"
        assert doc.vector("dense") == [1, 2, 3]
        assert doc.vector("sparse") == {1: 1.0, 2: 2.0, 3: 3.0}

    def test_with_multi_vectors(self):
        doc = Doc(
            id="1",
            vectors={
                "image": [1, 2, 3],
                "description": [4, 5, 6],
                "keys": {1: 1.0, 2: 2.0, 3: 3.0},
            },
            fields={"author": "Tom", "age": 19, "is_male": True, "weight": 60.5},
        )
        assert doc is not None
        assert doc.id == "1"
        assert doc.vector("image") == [1, 2, 3]
        assert doc.vector("description") == [4, 5, 6]
        assert doc.vector("keys") == {1: 1.0, 2: 2.0, 3: 3.0}
        assert doc.field("author") == "Tom"
        assert doc.field("age") == 19
        assert doc.field("is_male") == True
        assert doc.field("weight") == 60.5

    def test_with_numpy_array(self):
        import numpy as np

        doc = Doc._from_tuple(
            (
                "1",
                0.0,
                None,
                {
                    "image": np.array([1, 2, 3]),
                    "description": np.random.random(512),
                    "keys": {1: 1.0, 2: 2.0, 3: 3.0},
                },
            )
        )
        assert doc is not None
        assert doc.id == "1"
        assert doc.vector("image") == [1, 2, 3]
        assert doc.vector("keys") == {1: 1.0, 2: 2.0, 3: 3.0}


# ----------------------------
# CppDoc Test Case
# ----------------------------
class TestCppDoc:
    def test_default(self):
        doc = _Doc()
        assert doc is not None

    def test_doc_set_pk(self):
        doc = _Doc()
        doc.set_pk("1")
        assert doc.pk() == "1"

    def test_doc_set_score(self):
        doc = _Doc()
        doc.set_score(0.9)
        assert math.isclose(doc.score(), 0.9, rel_tol=1e-6)

    def test_doc_get_null_field(self):
        doc = _Doc()
        schema = FieldSchema("author", DataType.STRING, nullable=True)
        doc.set_any("author", schema._get_object(), None)
        assert doc.has_field("author")
        assert doc.get_any("author", schema.data_type) is None

    def test_doc_get_set_has_null_field(self):
        doc = _Doc()
        schema = FieldSchema("author", DataType.STRING, nullable=False)
        with pytest.raises(ValueError):
            doc.set_any("author", schema._get_object(), None)

    def test_doc_get_set_has_string_field(self):
        doc = _Doc()
        schema = FieldSchema("author", DataType.STRING)
        doc.set_any("author", schema._get_object(), "Tom")
        assert doc.has_field("author")
        assert doc.get_any("author", DataType.STRING) == "Tom"

    def test_doc_get_set_has_bool_field(self):
        doc = _Doc()
        schema = FieldSchema("is_male", DataType.BOOL)
        doc.set_any("is_male", schema._get_object(), True)
        assert doc.has_field("is_male")
        assert doc.get_any("is_male", DataType.BOOL) == True

    def test_doc_get_set_has_int32_field(self):
        doc = _Doc()
        schema = FieldSchema("age", DataType.INT32)
        doc.set_any("age", schema._get_object(), 19)
        assert doc.has_field("age")
        assert doc.get_any("age", DataType.INT32) == 19

    def test_doc_get_set_has_int64_field(self):
        doc = _Doc()
        schema = FieldSchema("id", DataType.INT64)
        doc.set_any("id", schema._get_object(), 1111111111111111111)
        assert doc.has_field("id")
        assert doc.get_any("id", DataType.INT64) == 1111111111111111111

    def test_doc_get_set_has_float_field(self):
        doc = _Doc()
        schema = FieldSchema("weight", DataType.FLOAT)
        doc.set_any("weight", schema._get_object(), 60.5)
        assert doc.has_field("weight")
        assert math.isclose(doc.get_any("weight", DataType.FLOAT), 60.5, rel_tol=1e-6)

    def test_doc_get_set_has_double_field(self):
        doc = _Doc()
        schema = FieldSchema("height", DataType.DOUBLE)
        doc.set_any("height", schema._get_object(), 1.77777777777)
        assert doc.has_field("height")
        assert math.isclose(
            doc.get_any("height", DataType.DOUBLE), 1.7777777777, rel_tol=1e-9
        )

    def test_doc_get_set_has_uint32_field(self):
        doc = _Doc()
        schema = FieldSchema("id", DataType.UINT32)
        doc.set_any("id", schema._get_object(), 4294967295)
        assert doc.has_field("id")
        assert doc.get_any("id", DataType.UINT32) == 4294967295

    def test_doc_get_set_has_uint64_field(self):
        doc = _Doc()
        schema = FieldSchema("id", DataType.UINT64)
        doc.set_any("id", schema._get_object(), 18446744073709551615)
        assert doc.has_field("id")
        assert doc.get_any("id", DataType.UINT64) == 18446744073709551615

    def test_doc_get_set_has_array_string_field(self):
        doc = _Doc()
        schema = FieldSchema("tags", DataType.ARRAY_STRING)
        doc.set_any("tags", schema._get_object(), ["tag1", "tag2", "tag3"])
        assert doc.has_field("tags")
        assert doc.get_any("tags", DataType.ARRAY_STRING) == ["tag1", "tag2", "tag3"]

    def test_doc_get_set_has_array_int32_field(self):
        doc = _Doc()
        schema = FieldSchema("ids", DataType.ARRAY_INT32)
        doc.set_any("ids", schema._get_object(), [1, 2, 3])
        assert doc.has_field("ids")
        assert doc.get_any("ids", DataType.ARRAY_INT32) == [1, 2, 3]

    def test_doc_get_set_has_array_int64_field(self):
        doc = _Doc()
        schema = FieldSchema("ids", DataType.ARRAY_INT64)
        doc.set_any("ids", schema._get_object(), [1, 2, 3])
        assert doc.has_field("ids")
        assert doc.get_any("ids", DataType.ARRAY_INT64) == [1, 2, 3]

    def test_doc_get_set_has_array_float_field(self):
        doc = _Doc()
        schema = FieldSchema("weights", DataType.ARRAY_FLOAT)
        doc.set_any("weights", schema._get_object(), [1.0, 2.0, 3.0])
        assert doc.has_field("weights")
        assert doc.get_any("weights", DataType.ARRAY_FLOAT) == [1.0, 2.0, 3.0]

    def test_doc_get_set_has_array_double_field(self):
        doc = _Doc()
        schema = FieldSchema("heights", DataType.ARRAY_DOUBLE)
        doc.set_any("heights", schema._get_object(), [1.0, 2.0, 3.0])
        assert doc.has_field("heights")
        assert doc.get_any("heights", DataType.ARRAY_DOUBLE) == [1.0, 2.0, 3.0]

    def test_doc_get_set_has_array_bool_field(self):
        doc = _Doc()
        schema = FieldSchema("bools", DataType.ARRAY_BOOL)
        doc.set_any("bools", schema._get_object(), [True, False, True])
        assert doc.has_field("bools")
        assert doc.get_any("bools", DataType.ARRAY_BOOL) == [True, False, True]

    def test_doc_get_set_has_vector_fp16(self):
        doc = _Doc()
        schema = VectorSchema("image", DataType.VECTOR_FP16)
        doc.set_any("image", schema._get_object(), [1.0, 2.0, 3.0])
        assert doc.has_field("image")
        image_vector = doc.get_any("image", DataType.VECTOR_FP16)
        assert image_vector is not None
        for i in range(len(image_vector)):
            assert math.isclose(image_vector[i], [1.0, 2.0, 3.0][i], rel_tol=1e-6)

    def test_doc_get_set_has_vector_fp32(self):
        doc = _Doc()
        schema = VectorSchema("image", DataType.VECTOR_FP32)
        doc.set_any("image", schema._get_object(), [1.111111, 2.222222, 3.333333])
        assert doc.has_field("image")
        vector = doc.get_any("image", DataType.VECTOR_FP32)
        assert vector is not None
        for i in range(len(vector)):
            assert math.isclose(
                vector[i], [1.111111, 2.222222, 3.333333][i], rel_tol=1e-6
            )

    def test_doc_get_set_has_vector_int8(self):
        doc = _Doc()
        schema = VectorSchema("image", DataType.VECTOR_INT8)
        doc.set_any("image", schema._get_object(), [1, 2, 3])
        assert doc.has_field("image")
        assert doc.get_any("image", DataType.VECTOR_INT8) == [1, 2, 3]

    def test_doc_get_set_has_sparse_vector_fp32(self):
        doc = _Doc()
        sparse = {1: 1.111111, 2: 2.222222, 3: 3.333333}
        schema = VectorSchema("key", DataType.SPARSE_VECTOR_FP32)
        doc.set_any("key", schema._get_object(), sparse)
        assert doc.has_field("key")
        vector = doc.get_any("key", DataType.SPARSE_VECTOR_FP32)
        assert vector is not None
        assert isinstance(vector, dict)
        for key, value in sparse.items():
            assert math.isclose(vector[key], value, rel_tol=1e-6)

    def test_doc_get_set_has_sparse_vector_fp16(self):
        doc = _Doc()
        sparse = {1: 1.1, 2: 2.2, 3: 3.3}
        schema = VectorSchema("key", DataType.SPARSE_VECTOR_FP16)
        doc.set_any("key", schema._get_object(), sparse)
        assert doc.has_field("key")
        vector = doc.get_any("key", DataType.SPARSE_VECTOR_FP16)
        assert vector is not None
        assert isinstance(vector, dict)
        for key, value in sparse.items():
            assert math.isclose(vector[key], value, rel_tol=1e-1)
