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
    CollectionSchema,
    CollectionStats,
    FieldSchema,
    VectorSchema,
    HnswIndexParam,
    InvertIndexParam,
    DataType,
    IndexType,
    MetricType,
)

# ----------------------------
# FieldSchema Test Case
# ----------------------------


class TestFieldSchema:
    def test_default(self):
        field = FieldSchema("field", data_type=DataType.FLOAT)
        assert field.name == "field"
        assert field.data_type == DataType.FLOAT
        assert field.nullable is False
        assert field.index_param is None

    def test_custom(self):
        field_1 = FieldSchema(
            name="float",
            data_type=DataType.FLOAT,
            nullable=True,
            index_param=InvertIndexParam(),
        )
        assert field_1.name == "float"
        assert field_1.data_type == DataType.FLOAT
        assert field_1.nullable is True
        assert field_1.index_param.enable_range_optimization is False

        field_2 = FieldSchema(
            name="str",
            data_type=DataType.STRING,
            nullable=True,
            index_param=InvertIndexParam(enable_range_optimization=True),
        )
        assert field_2.name == "str"
        assert field_2.data_type == DataType.STRING
        assert field_2.nullable is True
        assert field_2.index_param.enable_range_optimization is True

    def test_readonly(self):
        field = FieldSchema(
            name="float",
            data_type=DataType.FLOAT,
            nullable=True,
            index_param=InvertIndexParam(),
        )

        import sys

        if sys.version_info >= (3, 11):
            match_pattern = r"(can't set attribute|has no setter|readonly attribute)"
        else:
            match_pattern = r"can't set attribute"
        with pytest.raises(AttributeError, match=match_pattern):
            field.index_param = InvertIndexParam(enable_range_optimization=True)


# ----------------------------
# VectorSchema Test Case
# ----------------------------
class TestVectorSchema:
    def test_default(self):
        field = VectorSchema("vector", data_type=DataType.VECTOR_FP32, dimension=128)
        assert field.name == "vector"
        assert field.data_type == DataType.VECTOR_FP32
        assert field.dimension == 128
        assert field.index_param is not None
        assert field.index_param.type == IndexType.FLAT
        assert field.index_param.metric_type == MetricType.IP

    def test_custom(self):
        field = VectorSchema(
            name="vector",
            data_type=DataType.VECTOR_INT8,
            dimension=512,
            index_param=HnswIndexParam(
                metric_type=MetricType.COSINE, m=15, ef_construction=300
            ),
        )
        assert field.name == "vector"
        assert field.data_type == DataType.VECTOR_INT8
        assert field.index_param.metric_type == MetricType.COSINE
        assert field.index_param.m == 15
        assert field.index_param.ef_construction == 300

    def test_readonly(self):
        field = VectorSchema(
            name="vector",
            dimension=128,
            data_type=DataType.VECTOR_INT8,
        )

        import sys

        if sys.version_info >= (3, 11):
            match_pattern = r"(can't set attribute|has no setter|readonly attribute)"
        else:
            match_pattern = r"can't set attribute"
        with pytest.raises(AttributeError, match=match_pattern):
            field.dimension = 4


# ----------------------------
# CollectionSchema Test Case
# ----------------------------
class TestCollectionSchema:
    def test_collection_schema_with_single_field(self):
        collection_schema = CollectionSchema(
            name="test_collection",
            fields=FieldSchema(
                name="id",
                data_type=DataType.INT64,
                index_param=InvertIndexParam(),
                nullable=False,
            ),
            vectors=VectorSchema(
                name="vector",
                data_type=DataType.VECTOR_INT8,
                dimension=128,
                index_param=HnswIndexParam(),
            ),
        )

        assert collection_schema is not None
        assert collection_schema.name == "test_collection"
        assert len(collection_schema.fields) == 1
        assert len(collection_schema.vectors) == 1

        field = collection_schema.field("id")
        assert field is not None
        assert field.name == "id"
        assert field.data_type == DataType.INT64
        assert not field.nullable
        assert field.index_param.type == IndexType.INVERT
        assert not field.index_param.enable_range_optimization

        vector = collection_schema.vector("vector")
        assert vector is not None
        assert vector.name == "vector"
        assert vector.data_type == DataType.VECTOR_INT8
        assert vector.dimension == 128
        assert vector.index_param.type == IndexType.HNSW
        assert vector.index_param.m == 50
        assert vector.index_param.ef_construction == 500
        assert vector.index_param.metric_type == MetricType.IP

    def test_collection_schema_with_multi_fields(self):
        collection_schema = CollectionSchema(
            name="test_collection",
            fields=[
                FieldSchema(
                    "id",
                    DataType.INT64,
                    nullable=False,
                    index_param=InvertIndexParam(enable_range_optimization=True),
                ),
                FieldSchema(
                    "name",
                    DataType.STRING,
                    nullable=False,
                    index_param=InvertIndexParam(),
                ),
                FieldSchema(
                    "weight",
                    DataType.INT32,
                    nullable=True,
                ),
            ],
            vectors=[
                VectorSchema(
                    "dense",
                    DataType.VECTOR_FP32,
                    dimension=128,
                    index_param=HnswIndexParam(),
                ),
                VectorSchema(
                    "sparse", DataType.SPARSE_VECTOR_FP32, index_param=HnswIndexParam()
                ),
            ],
        )
        assert collection_schema is not None
        assert collection_schema.name == "test_collection"
        assert len(collection_schema.fields) == 3
        assert len(collection_schema.vectors) == 2

        field_id = collection_schema.field("id")
        assert field_id is not None
        assert field_id.name == "id"
        assert field_id.data_type == DataType.INT64
        assert not field_id.nullable
        assert field_id.index_param.type == IndexType.INVERT

        dense = collection_schema.vector("dense")
        assert dense is not None
        assert dense.name == "dense"
        assert dense.data_type == DataType.VECTOR_FP32
        assert dense.dimension == 128
        assert dense.index_param.type == IndexType.HNSW

        sparse = collection_schema.vector("sparse")
        assert sparse is not None
        assert sparse.name == "sparse"
        assert sparse.data_type == DataType.SPARSE_VECTOR_FP32
        assert sparse.dimension == 0
        assert sparse.index_param.type == IndexType.HNSW

        assert str(collection_schema) is not None


# ----------------------------
# CollectionStats Test Case
# ----------------------------
class TestCollectionStats:
    """
    The constructor of CollectionStats is not provided.
    It can only be obtained through collection.stats()
    """

    def test_collection_stats(self):
        stats = CollectionStats()
        assert stats is not None
