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

import gc

import numpy as np
import pytest

import zvec
from zvec import (
    CollectionOption,
    CollectionSchema,
    Doc,
    FieldSchema,
    HnswIndexParam,
    HnswQueryParam,
    InvertIndexParam,
    Query,
    VamanaIndexParam,
    VamanaQueryParam,
    VectorSchema,
)
from zvec.typing import DataType, MetricType, QuantizeType


@pytest.mark.parametrize(
    ("configured_flat_data_type", "effective_flat_data_type"),
    [
        (None, DataType.VECTOR_FP32),
        (DataType.VECTOR_FP16, DataType.VECTOR_FP16),
        (DataType.VECTOR_UINT8, DataType.VECTOR_UINT8),
    ],
    ids=["default_fp32", "fp16", "uint8"],
)
@pytest.mark.parametrize("index_kind", ["hnsw", "vamana"])
@pytest.mark.parametrize("use_flat_contiguous_memory", [False, True])
def test_refine_flat_native_storage_roundtrip(
    tmp_path,
    configured_flat_data_type,
    effective_flat_data_type,
    index_kind,
    use_flat_contiguous_memory,
):
    dimension = 17
    initial_doc_count = 96
    doc_count = 112
    collection_path = tmp_path / (
        f"refine_{index_kind}_{effective_flat_data_type.name.lower()}_"
        f"contiguous_{use_flat_contiguous_memory}"
    )
    flat_data_type_option = (
        {}
        if configured_flat_data_type is None
        else {"flat_data_type": configured_flat_data_type}
    )
    if index_kind == "hnsw":
        index_param = HnswIndexParam(
            metric_type=MetricType.L2,
            m=16,
            ef_construction=100,
            quantize_type=QuantizeType.INT8,
            use_flat_contiguous_memory=use_flat_contiguous_memory,
            **flat_data_type_option,
        )
    else:
        index_param = VamanaIndexParam(
            metric_type=MetricType.L2,
            max_degree=16,
            search_list_size=64,
            quantize_type=QuantizeType.INT8,
            use_flat_contiguous_memory=use_flat_contiguous_memory,
            **flat_data_type_option,
        )
    schema = CollectionSchema(
        name="refine_flat_native_storage",
        fields=[
            FieldSchema(
                "ordinal",
                DataType.INT64,
                nullable=False,
                index_param=InvertIndexParam(),
            )
        ],
        vectors=[
            VectorSchema(
                "dense",
                DataType.VECTOR_FP32,
                dimension=dimension,
                index_param=index_param,
            )
        ],
    )
    option = CollectionOption(read_only=False, enable_mmap=True)
    docs = []
    for i in range(doc_count):
        vector = np.asarray(
            [
                (i * 7 + d * 3) % 239 + ((i * 11 + d * 5) % 17) / 23.0
                for d in range(dimension)
            ],
            dtype=np.float32,
        )
        docs.append(
            Doc(
                id=str(i),
                fields={"ordinal": i},
                vectors={"dense": vector.tolist()},
            )
        )

    def refined_query(collection, doc_index, *, include_vector=False):
        if index_kind == "hnsw":
            query_param = HnswQueryParam(ef=128, is_using_refiner=True)
        else:
            query_param = VamanaQueryParam(ef_search=128, is_using_refiner=True)
        query = Query(
            field_name="dense",
            vector=docs[doc_index].vector("dense"),
            param=query_param,
        )
        hits = collection.query(query, topk=5, include_vector=include_vector)
        assert hits
        return hits

    def native_vector(values):
        vector = np.asarray(values, dtype=np.float32)
        if effective_flat_data_type == DataType.VECTOR_FP16:
            return vector.astype(np.float16).astype(np.float32)
        if effective_flat_data_type == DataType.VECTOR_UINT8:
            return vector.astype(np.uint8).astype(np.float32)
        return vector

    def assert_native_fetch(collection, doc_index):
        fetched = collection.fetch(ids=[str(doc_index)])[str(doc_index)]
        np.testing.assert_array_equal(
            np.asarray(fetched.vector("dense"), dtype=np.float32),
            native_vector(docs[doc_index].vector("dense")),
        )

    def assert_native_refine_scores(collection, doc_index):
        query = np.asarray(docs[doc_index].vector("dense"), dtype=np.float32)
        hits = refined_query(collection, doc_index, include_vector=True)
        observed_native_difference = False
        for hit in hits:
            stored = np.asarray(docs[int(hit.id)].vector("dense"), dtype=np.float32)
            if effective_flat_data_type == DataType.VECTOR_FP16:
                query_native = query.astype(np.float16).astype(np.float32)
                stored_native = stored.astype(np.float16).astype(np.float32)
            elif effective_flat_data_type == DataType.VECTOR_UINT8:
                query_native = query.astype(np.uint8).astype(np.int32)
                stored_native = stored.astype(np.uint8).astype(np.int32)
            else:
                query_native = query
                stored_native = stored
            expected = float(np.sum((stored_native - query_native) ** 2))
            fp32_expected = float(np.sum((stored - query) ** 2))
            observed_native_difference |= not np.isclose(
                expected, fp32_expected, rtol=1e-7, atol=1e-5
            )
            assert hit.score == pytest.approx(expected, rel=1e-5, abs=1e-5)
            np.testing.assert_array_equal(
                np.asarray(hit.vector("dense"), dtype=np.float32),
                native_vector(stored),
            )
        assert observed_native_difference is (
            effective_flat_data_type != DataType.VECTOR_FP32
        )
        assert_native_fetch(collection, doc_index)
        return hits

    collection = zvec.create_and_open(
        path=str(collection_path), schema=schema, option=option
    )
    reopened = None
    try:
        for status in collection.insert(docs[:initial_doc_count]):
            assert status.ok()
        assert refined_query(collection, 23)[0].id == "23"
        assert_native_fetch(collection, 23)

        collection.optimize()
        assert_native_refine_scores(collection, 23)
        restored_param = collection.schema.vectors[0].index_param
        assert restored_param.use_flat_contiguous_memory is use_flat_contiguous_memory
        assert restored_param.flat_data_type == effective_flat_data_type

        for status in collection.insert(docs[initial_doc_count:]):
            assert status.ok()
        assert refined_query(collection, 103)[0].id == "103"
        collection.optimize()
        assert_native_refine_scores(collection, 103)

        collection = None
        gc.collect()
        reopened = zvec.open(path=str(collection_path), option=option)
        restored_param = reopened.schema.vectors[0].index_param
        assert restored_param.use_flat_contiguous_memory is use_flat_contiguous_memory
        assert restored_param.flat_data_type == effective_flat_data_type
        assert_native_refine_scores(reopened, 23)
        assert_native_refine_scores(reopened, 103)
    finally:
        if reopened is not None:
            reopened.destroy()
        elif collection is not None:
            collection.destroy()


@pytest.mark.parametrize("index_kind", ["hnsw", "vamana"])
def test_pooled_graph_context_refreshes_metric(tmp_path, index_kind):
    """A pooled graph context must not retain a prior index's query metric."""
    dimension = 16
    doc_count = 48
    query_id = 11
    docs = [
        Doc(
            id=str(i),
            vectors={
                "dense": np.asarray(
                    [(i * 17 + d * 7) % 101 for d in range(dimension)],
                    dtype=np.float32,
                ).tolist()
            },
        )
        for i in range(doc_count)
    ]
    option = CollectionOption(read_only=False, enable_mmap=True)

    def create_collection(name, quantize_type):
        if index_kind == "hnsw":
            index_param = HnswIndexParam(
                metric_type=MetricType.L2,
                m=12,
                ef_construction=64,
                quantize_type=quantize_type,
            )
        else:
            index_param = VamanaIndexParam(
                metric_type=MetricType.L2,
                max_degree=12,
                search_list_size=64,
                quantize_type=quantize_type,
            )
        schema = CollectionSchema(
            name=name,
            vectors=[
                VectorSchema(
                    "dense",
                    DataType.VECTOR_FP32,
                    dimension=dimension,
                    index_param=index_param,
                )
            ],
        )
        collection = zvec.create_and_open(
            path=str(tmp_path / name), schema=schema, option=option
        )
        for status in collection.insert(docs):
            assert status.ok()
        collection.optimize()
        return collection

    def linear_query(collection):
        if index_kind == "hnsw":
            query_param = HnswQueryParam(ef=128, is_linear=True)
        else:
            query_param = VamanaQueryParam(ef_search=128, is_linear=True)
        hits = collection.query(
            Query(
                field_name="dense",
                vector=docs[query_id].vector("dense"),
                param=query_param,
            ),
            topk=5,
        )
        assert hits
        return [hit.id for hit in hits]

    quantized = create_collection(f"{index_kind}_pooled_quantized", QuantizeType.INT8)
    raw = None
    try:
        assert linear_query(quantized)[0] == str(query_id)

        # HNSW/Vamana contexts are reused per thread. Searching a raw FP32
        # index after an INT8 index must replace the old query preprocess.
        raw = create_collection(f"{index_kind}_pooled_raw", QuantizeType.UNDEFINED)
        assert linear_query(raw)[0] == str(query_id)
    finally:
        if raw is not None:
            raw.destroy()
        quantized.destroy()


@pytest.mark.parametrize("index_kind", ["hnsw", "vamana"])
def test_fp16_cosine_refine_uses_native_flat_pipeline(tmp_path, index_kind):
    dimension = 17
    doc_count = 80
    collection_path = tmp_path / f"fp16_cosine_refine_{index_kind}"
    if index_kind == "hnsw":
        index_param = HnswIndexParam(
            metric_type=MetricType.COSINE,
            m=16,
            ef_construction=100,
            quantize_type=QuantizeType.INT8,
            flat_data_type=DataType.VECTOR_FP16,
            use_flat_contiguous_memory=True,
        )
        query_param = HnswQueryParam(ef=128, is_using_refiner=True)
    else:
        index_param = VamanaIndexParam(
            metric_type=MetricType.COSINE,
            max_degree=16,
            search_list_size=64,
            quantize_type=QuantizeType.INT8,
            flat_data_type=DataType.VECTOR_FP16,
            use_flat_contiguous_memory=True,
        )
        query_param = VamanaQueryParam(ef_search=128, is_using_refiner=True)

    schema = CollectionSchema(
        name="fp16_cosine_refine",
        vectors=[
            VectorSchema(
                "dense",
                DataType.VECTOR_FP32,
                dimension=dimension,
                index_param=index_param,
            )
        ],
    )
    docs = []
    for i in range(doc_count):
        vector = np.asarray(
            [
                0.25 + d * 0.017 + (((i * 37 + d * 19) % 97) - 48) * 0.00011
                for d in range(dimension)
            ],
            dtype=np.float32,
        )
        docs.append(Doc(id=str(i), vectors={"dense": vector.tolist()}))

    collection = zvec.create_and_open(
        path=str(collection_path),
        schema=schema,
        option=CollectionOption(read_only=False, enable_mmap=True),
    )
    try:
        for status in collection.insert(docs):
            assert status.ok()
        collection.optimize()

        query_id = 37
        query = Query(
            field_name="dense",
            vector=docs[query_id].vector("dense"),
            param=query_param,
        )
        hits = collection.query(query, topk=10, include_vector=True)
        assert hits
        assert hits[0].id == str(query_id)

        query_native = np.asarray(docs[query_id].vector("dense"), dtype=np.float16)
        query_norm = np.linalg.norm(query_native.astype(np.float32))
        query_normalized = (query_native.astype(np.float32) / query_norm).astype(
            np.float16
        )
        for hit in hits:
            stored_native = np.asarray(
                docs[int(hit.id)].vector("dense"), dtype=np.float16
            )
            stored_norm = np.linalg.norm(stored_native.astype(np.float32))
            stored_normalized = (stored_native.astype(np.float32) / stored_norm).astype(
                np.float16
            )
            expected = 1.0 - float(
                np.dot(
                    query_normalized.astype(np.float32),
                    stored_normalized.astype(np.float32),
                )
            )
            assert hit.score == pytest.approx(expected, abs=3e-4)
            np.testing.assert_allclose(
                np.asarray(hit.vector("dense"), dtype=np.float32),
                stored_native.astype(np.float32),
                rtol=0,
                atol=3e-4,
            )

        fetched = collection.fetch(ids=[str(query_id)])[str(query_id)]
        np.testing.assert_allclose(
            np.asarray(fetched.vector("dense"), dtype=np.float32),
            query_native.astype(np.float32),
            rtol=0,
            atol=3e-4,
        )
    finally:
        collection.destroy()
