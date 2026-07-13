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
import zvec
from zvec import (
    Collection,
    CollectionOption,
    DataType,
    Doc,
    FieldSchema,
    FlatIndexParam,
    Fts,
    GroupResult,
    HnswIndexParam,
    HnswQueryParam,
    InvertIndexParam,
    Query,
    VectorSchema,
)

# ==================== Constants ====================

GB_DIMENSION = 4
GB_NUM_DOCS = 12
GB_NUM_GROUPS = 3
GB_TOPK_PER_GROUP = 2


# ==================== Fixtures ====================


@pytest.fixture(scope="session")
def group_by_collection_schema():
    """Collection schema for group-by end-to-end tests.

    Mirrors the data layout in ``vector_column_indexer_test.cc``:
    a 4-dimensional dense vector and a scalar ``group_id`` field used
    for grouping.
    """
    return zvec.CollectionSchema(
        name="test_group_by_collection",
        fields=[
            FieldSchema(
                "id",
                DataType.INT64,
                nullable=False,
                index_param=InvertIndexParam(enable_range_optimization=True),
            ),
            FieldSchema(
                "group_id",
                DataType.INT64,
                nullable=False,
                index_param=InvertIndexParam(),
            ),
        ],
        vectors=[
            VectorSchema(
                "dense",
                DataType.VECTOR_FP32,
                dimension=GB_DIMENSION,
                index_param=HnswIndexParam(),
            ),
            VectorSchema(
                "dense_flat",
                DataType.VECTOR_FP32,
                dimension=GB_DIMENSION,
                index_param=FlatIndexParam(),
            ),
        ],
    )


@pytest.fixture(scope="session")
def collection_option():
    return CollectionOption(read_only=False, enable_mmap=True)


@pytest.fixture
def group_by_docs():
    """Generate docs matching the C++ GroupByIndexerTest fixture.

    Doc ``i`` has vector ``[i, i, i, i]`` and ``group_id = i % 3``.
    """
    return [
        Doc(
            id=f"{i}",
            fields={"id": i, "group_id": i % GB_NUM_GROUPS},
            vectors={
                "dense": [float(i)] * GB_DIMENSION,
                "dense_flat": [float(i)] * GB_DIMENSION,
            },
        )
        for i in range(GB_NUM_DOCS)
    ]


@pytest.fixture(scope="function")
def group_by_collection(
    tmp_path_factory, group_by_collection_schema, collection_option
) -> Collection:
    """Function-scoped fixture: creates and opens a collection for group-by tests."""
    temp_dir = tmp_path_factory.mktemp("zvec_group_by")
    collection_path = temp_dir / "test_group_by_collection"

    coll = zvec.create_and_open(
        path=str(collection_path),
        schema=group_by_collection_schema,
        option=collection_option,
    )

    assert coll is not None, "Failed to create and open group-by collection"
    assert coll.path == str(collection_path)
    assert coll.schema.name == group_by_collection_schema.name

    try:
        yield coll
    finally:
        if hasattr(coll, "destroy") and coll is not None:
            try:
                coll.destroy()
            except Exception as e:
                print(f"Warning: failed to destroy collection: {e}")


@pytest.fixture
def group_by_collection_with_docs(
    group_by_collection: Collection, group_by_docs
) -> Collection:
    """Setup: insert group-by fixture docs."""
    assert group_by_collection.stats.doc_count == 0
    result = group_by_collection.insert(group_by_docs)
    assert len(result) == len(group_by_docs)
    for item in result:
        assert item.ok()
    assert group_by_collection.stats.doc_count == len(group_by_docs)

    yield group_by_collection

    # Teardown
    group_by_collection.delete([doc.id for doc in group_by_docs])


# ==================== Helpers ====================


def _assert_grouped_results(results, num_groups, topk_per_group, query_value):
    """Validate group-by result structure and ordering.

    Each returned group must:
      - contain only docs whose ``group_id`` matches ``group_by_value``
      - have at most ``topk_per_group`` docs
      - have docs sorted by descending score
    """
    assert len(results) == num_groups, (
        f"Expected {num_groups} groups, got {len(results)}"
    )

    group_values = set()
    for group in results:
        assert isinstance(group, GroupResult)
        group_value = int(group.group_by_value)
        group_values.add(group_value)
        docs = group.docs
        assert 1 <= len(docs) <= topk_per_group

        for doc in docs:
            assert int(doc.field("group_id")) == group_value

        scores = [doc.score for doc in docs]
        assert scores == sorted(scores, reverse=True), (
            "Docs must be sorted by score desc"
        )

        # Score sanity: for query [1,1,1,1] and vector [i,i,i,i],
        # IP score is 4 * i.
        for doc in docs:
            doc_id = int(doc.field("id"))
            expected_score = float(doc_id * sum(query_value))
            assert abs(doc.score - expected_score) < 0.1

    assert group_values == set(range(num_groups))


# ==================== Tests ====================


@pytest.mark.usefixtures("group_by_collection_with_docs")
class TestGroupBySearch:
    def test_group_by_defaults(self, group_by_collection: Collection):
        results = group_by_collection.group_by_query(
            Query(field_name="dense_flat", vector=[1.0] * GB_DIMENSION),
            group_by_field_name="group_id",
        )
        assert len(results) == 2
        assert all(1 <= len(group.docs) <= 3 for group in results)

    def test_group_by_hnsw(self, group_by_collection: Collection):
        """Group-by search over an HNSW index."""
        query_vector = [1.0] * GB_DIMENSION
        results = group_by_collection.group_by_query(
            Query(
                field_name="dense", vector=query_vector, param=HnswQueryParam(ef=300)
            ),
            group_by_field_name="group_id",
            group_count=GB_NUM_GROUPS,
            topk_per_group=GB_TOPK_PER_GROUP,
        )
        _assert_grouped_results(results, GB_NUM_GROUPS, GB_TOPK_PER_GROUP, query_vector)

    def test_group_by_flat(self, group_by_collection: Collection):
        """Group-by search over a FLAT index."""
        query_vector = [1.0] * GB_DIMENSION
        results = group_by_collection.group_by_query(
            Query(field_name="dense_flat", vector=query_vector),
            group_by_field_name="group_id",
            group_count=GB_NUM_GROUPS,
            topk_per_group=GB_TOPK_PER_GROUP,
        )
        _assert_grouped_results(results, GB_NUM_GROUPS, GB_TOPK_PER_GROUP, query_vector)

    def test_group_by_with_filter(self, group_by_collection: Collection):
        """Group-by search with a scalar filter."""
        query_vector = [1.0] * GB_DIMENSION
        results = group_by_collection.group_by_query(
            Query(field_name="dense_flat", vector=query_vector),
            group_by_field_name="group_id",
            group_count=GB_NUM_GROUPS,
            topk_per_group=GB_TOPK_PER_GROUP,
            filter="id < 6",
        )
        # Only docs 0..5 are visible; every group still has at least one doc.
        assert len(results) == GB_NUM_GROUPS
        for group in results:
            for doc in group.docs:
                assert int(doc.field("id")) < 6

    def test_group_by_include_vector(self, group_by_collection: Collection):
        """Group-by search returns original vectors when requested."""
        query_vector = [1.0] * GB_DIMENSION
        results = group_by_collection.group_by_query(
            Query(field_name="dense_flat", vector=query_vector),
            group_by_field_name="group_id",
            group_count=GB_NUM_GROUPS,
            topk_per_group=GB_TOPK_PER_GROUP,
            include_vector=True,
        )
        assert len(results) == GB_NUM_GROUPS
        for group in results:
            for doc in group.docs:
                vec = doc.vector("dense_flat")
                doc_id = int(doc.field("id"))
                assert vec == pytest.approx([float(doc_id)] * GB_DIMENSION, abs=1e-5)

    def test_group_by_output_fields(self, group_by_collection: Collection):
        """Group-by search honors scalar output field selection."""
        query_vector = [1.0] * GB_DIMENSION
        results = group_by_collection.group_by_query(
            Query(field_name="dense_flat", vector=query_vector),
            group_by_field_name="group_id",
            group_count=GB_NUM_GROUPS,
            topk_per_group=GB_TOPK_PER_GROUP,
            output_fields=["group_id"],
        )
        assert len(results) == GB_NUM_GROUPS
        for group in results:
            for doc in group.docs:
                assert doc.has_field("group_id")

    def test_group_by_invalid_field(self, group_by_collection: Collection):
        """Group-by with a non-existent vector field raises an error."""
        with pytest.raises(ValueError):
            group_by_collection.group_by_query(
                Query(field_name="nonexistent", vector=[1.0] * GB_DIMENSION),
                group_by_field_name="group_id",
            )

    def test_group_by_query_by_id(self, group_by_collection: Collection):
        results = group_by_collection.group_by_query(
            Query(field_name="dense_flat", id="11"),
            group_by_field_name="group_id",
            group_count=GB_NUM_GROUPS,
            topk_per_group=GB_TOPK_PER_GROUP,
        )
        assert len(results) == GB_NUM_GROUPS

    @pytest.mark.parametrize(
        ("query", "error"),
        [
            (Query(field_name="content", fts=Fts(match_string="text")), "FTS"),
            (Query(field_name="dense_flat"), "vector or document id"),
        ],
    )
    def test_group_by_rejects_unsupported_query(
        self, group_by_collection: Collection, query: Query, error: str
    ):
        with pytest.raises(ValueError, match=error):
            group_by_collection.group_by_query(query, "group_id")

    @pytest.mark.parametrize(
        ("kwargs", "error"),
        [
            ({"group_by_field_name": ""}, "group_by_field_name"),
            ({"group_by_field_name": "group_id", "group_count": 0}, "group_count"),
            (
                {"group_by_field_name": "group_id", "topk_per_group": 0},
                "topk_per_group",
            ),
        ],
    )
    def test_group_by_rejects_invalid_group_params(
        self, group_by_collection: Collection, kwargs: dict, error: str
    ):
        with pytest.raises(ValueError, match=error):
            group_by_collection.group_by_query(
                Query(field_name="dense_flat", vector=[1.0] * GB_DIMENSION),
                **kwargs,
            )


class TestGroupByEmptyCollection:
    def test_group_by_empty_collection(self, group_by_collection: Collection):
        """Group-by on an empty collection returns an empty list."""
        results = group_by_collection.group_by_query(
            Query(field_name="dense_flat", vector=[1.0] * GB_DIMENSION),
            group_by_field_name="group_id",
            group_count=GB_NUM_GROUPS,
            topk_per_group=GB_TOPK_PER_GROUP,
        )
        assert results == []


def test_group_by_public_api_exports():
    assert zvec.GroupResult is GroupResult
    assert not hasattr(zvec, "GroupByQuery")
    assert not hasattr(Collection, "groupby_query")
