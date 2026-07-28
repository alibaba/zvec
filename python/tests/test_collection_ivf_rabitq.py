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

import platform
import sys

import pytest
import zvec
from zvec import (
    Collection,
    CollectionOption,
    CollectionSchema,
    DataType,
    Doc,
    FieldSchema,
    IndexType,
    InvertIndexParam,
    IvfRabitqIndexParam,
    IvfRabitqQueryParam,
    MetricType,
    OptimizeOption,
    Query,
    QuantizeType,
    VectorSchema,
)

pytestmark = pytest.mark.skipif(
    not (sys.platform == "linux" and platform.machine() in ("x86_64", "AMD64")),
    reason="IVF RaBitQ only supported on Linux x86_64",
)

DIMENSION = 128
NLIST = 16
NPROBE = 16
TOPK = 10
SMALL_DOC_COUNT = 64
INDEXED_DOC_COUNT = 1200


def _vector_for(doc_id: int) -> list[float]:
    return [float(doc_id)] * DIMENSION


def _make_docs(count: int) -> list[Doc]:
    return [
        Doc(
            id=str(doc_id),
            fields={
                "id": doc_id,
                "name": f"test_doc_{doc_id}",
                "group_id": doc_id % 4,
            },
            vectors={"embedding": _vector_for(doc_id)},
        )
        for doc_id in range(count)
    ]


def _build_schema(name: str) -> CollectionSchema:
    return CollectionSchema(
        name=name,
        fields=[
            FieldSchema(
                "id",
                DataType.INT64,
                nullable=False,
                index_param=InvertIndexParam(enable_range_optimization=True),
            ),
            FieldSchema("name", DataType.STRING, nullable=False),
            FieldSchema("group_id", DataType.INT64, nullable=False),
        ],
        vectors=[
            VectorSchema(
                "embedding",
                DataType.VECTOR_FP32,
                dimension=DIMENSION,
                index_param=IvfRabitqIndexParam(
                    metric_type=MetricType.L2,
                    nlist=NLIST,
                    total_bits=7,
                    sample_count=0,
                ),
            ),
        ],
    )


def _create_collection(tmp_path_factory, schema_name: str) -> Collection:
    path = tmp_path_factory.mktemp("zvec_ivf_rabitq") / schema_name
    return zvec.create_and_open(
        path=str(path),
        schema=_build_schema(schema_name),
        option=CollectionOption(read_only=False, enable_mmap=True),
    )


def _insert_docs(coll: Collection, docs: list[Doc]) -> None:
    batch_size = 512
    for offset in range(0, len(docs), batch_size):
        batch = docs[offset : offset + batch_size]
        result = coll.insert(batch)
        assert len(result) == len(batch)
        for item in result:
            assert item.ok()
    assert coll.stats.doc_count == len(docs)


def _query(
    coll: Collection,
    query_vector: list[float],
    *,
    filter: str | None = None,
    include_vector: bool = False,
) -> list[Doc]:
    query = Query(
        field_name="embedding",
        vector=query_vector,
        param=IvfRabitqQueryParam(nprobe=NPROBE),
    )
    result = coll.query(
        queries=query,
        topk=TOPK,
        filter=filter,
        include_vector=include_vector,
    )
    assert result is not None
    assert 0 < len(result) <= TOPK
    return result


class TestIvfRabitqCollection:
    def test_schema_round_trip(self, tmp_path_factory):
        coll = _create_collection(tmp_path_factory, "ivf_rabitq_schema_rt")
        try:
            vector_schema = coll.schema.vector("embedding")
            assert vector_schema is not None
            assert vector_schema.data_type == DataType.VECTOR_FP32
            assert vector_schema.dimension == DIMENSION

            index_param = vector_schema.index_param
            assert index_param.type == IndexType.IVF_RABITQ
            assert index_param.metric_type == MetricType.L2
            assert index_param.quantize_type == QuantizeType.RABITQ
            assert index_param.nlist == NLIST
            assert index_param.total_bits == 7
            assert index_param.sample_count == 0
            assert coll.stats.index_completeness["embedding"] == 1
        finally:
            coll.destroy()

    def test_insert_fetch_and_query_writer_segment(self, tmp_path_factory):
        coll = _create_collection(tmp_path_factory, "ivf_rabitq_writer_query")
        try:
            docs = _make_docs(SMALL_DOC_COUNT)
            _insert_docs(coll, docs)

            fetched = coll.fetch(ids=["7", "42"])
            assert set(fetched.keys()) == {"7", "42"}
            assert fetched["42"].field("name") == "test_doc_42"
            assert fetched["42"].vector("embedding") == _vector_for(42)

            result = _query(coll, docs[42].vector("embedding"))
            assert result[0].id == "42"
        finally:
            coll.destroy()

    def test_optimize_reopen_and_query(self, tmp_path_factory):
        schema_name = "ivf_rabitq_optimize_reopen"
        collection_path = tmp_path_factory.mktemp("zvec_ivf_rabitq") / schema_name
        option = CollectionOption(read_only=False, enable_mmap=True)
        coll = zvec.create_and_open(
            path=str(collection_path),
            schema=_build_schema(schema_name),
            option=option,
        )
        reopened = None
        try:
            docs = _make_docs(INDEXED_DOC_COUNT)
            _insert_docs(coll, docs)

            coll.optimize(OptimizeOption())
            assert coll.stats.doc_count == INDEXED_DOC_COUNT
            assert coll.stats.index_completeness["embedding"] == 1

            query_vector = docs[42].vector("embedding")
            result = _query(coll, query_vector)
            assert result[0].id == "42"

            filtered = _query(coll, query_vector, filter="id < 50", include_vector=True)
            assert filtered[0].id == "42"
            for doc in filtered:
                assert doc.field("id") < 50
            assert len(filtered[0].vector("embedding")) == DIMENSION

            path = str(collection_path)
            del coll
            coll = None

            reopened = zvec.open(path=path, option=option)
            assert reopened.stats.doc_count == INDEXED_DOC_COUNT
            reopened_result = _query(reopened, query_vector)
            assert reopened_result[0].id == "42"
        finally:
            if reopened is not None:
                reopened.destroy()
            elif coll is not None:
                coll.destroy()

    def test_group_by_after_optimize(self, tmp_path_factory):
        coll = _create_collection(tmp_path_factory, "ivf_rabitq_group_by")
        try:
            docs = _make_docs(INDEXED_DOC_COUNT)
            _insert_docs(coll, docs)
            coll.optimize(OptimizeOption())

            query = Query(
                field_name="embedding",
                vector=docs[42].vector("embedding"),
                param=IvfRabitqQueryParam(nprobe=NPROBE),
            )
            results = coll.group_by_query(
                query,
                group_by_field_name="group_id",
                group_count=4,
                topk_per_group=2,
            )
            assert {int(group.group_by_value) for group in results} == set(range(4))
            for group in results:
                assert 1 <= len(group.docs) <= 2
                assert all(
                    doc.field("group_id") == int(group.group_by_value)
                    for doc in group.docs
                )

            filtered = coll.group_by_query(
                query,
                group_by_field_name="group_id",
                group_count=4,
                topk_per_group=2,
                filter="id < 8",
                include_vector=True,
            )
            assert {int(group.group_by_value) for group in filtered} == set(range(4))
            for group in filtered:
                for doc in group.docs:
                    assert doc.field("id") < 8
                    assert len(doc.vector("embedding")) == DIMENSION
        finally:
            coll.destroy()
