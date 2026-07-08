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
    GroupByQuery,
    HnswIndexParam,
    HnswQueryParam,
    InvertIndexParam,
    VectorSchema,
)

# ==================== Constants ====================

GB_DIMENSION = 4
GB_NUM_DOCS = 12
GB_NUM_GROUPS = 3
GB_GROUP_TOPK = 2


# ==================== Fixtures ====================


@pytest.fixture(scope="session")
def groupby_collection_schema():
    """Collection schema for group-by end-to-end tests.

    Mirrors the data layout in ``vector_column_indexer_test.cc``:
    a 4-dimensional dense vector and a scalar ``group_id`` field used
    for grouping.
    """
    return zvec.CollectionSchema(
        name="test_groupby_collection",
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
def groupby_docs():
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
def groupby_collection(
    tmp_path_factory, groupby_collection_schema, collection_option
) -> Collection:
    """Function-scoped fixture: creates and opens a collection for group-by tests."""
    temp_dir = tmp_path_factory.mktemp("zvec_groupby")
    collection_path = temp_dir / "test_groupby_collection"

    coll = zvec.create_and_open(
        path=str(collection_path),
        schema=groupby_collection_schema,
        option=collection_option,
    )

    assert coll is not None, "Failed to create and open group-by collection"
    assert coll.path == str(collection_path)
    assert coll.schema.name == groupby_collection_schema.name

    try:
        yield coll
    finally:
        if hasattr(coll, "destroy") and coll is not None:
            try:
                coll.destroy()
            except Exception as e:
                print(f"Warning: failed to destroy collection: {e}")


@pytest.fixture
def groupby_collection_with_docs(
    groupby_collection: Collection, groupby_docs
) -> Collection:
    """Setup: insert group-by fixture docs."""
    assert groupby_collection.stats.doc_count == 0
    result = groupby_collection.insert(groupby_docs)
    assert len(result) == len(groupby_docs)
    for item in result:
        assert item.ok()
    assert groupby_collection.stats.doc_count == len(groupby_docs)

    yield groupby_collection

    # Teardown
    groupby_collection.delete([doc.id for doc in groupby_docs])


# ==================== Helpers ====================


def _assert_grouped_results(results, num_groups, group_topk, query_value):
    """Validate group-by result structure and ordering.

    Each returned group must:
      - contain only docs whose ``group_id`` matches ``group_by_value``
      - have at most ``group_topk`` docs
      - have docs sorted by descending score
    """
    assert len(results) == num_groups, (
        f"Expected {num_groups} groups, got {len(results)}"
    )

    group_values = set()
    for group in results:
        group_value = int(group["group_by_value"])
        group_values.add(group_value)
        docs = group["docs"]
        assert 1 <= len(docs) <= group_topk

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


@pytest.mark.usefixtures("groupby_collection_with_docs")
class TestGroupBySearch:
    def test_groupby_hnsw(self, groupby_collection: Collection):
        """Group-by search over an HNSW index."""
        query_vector = [1.0] * GB_DIMENSION
        results = groupby_collection.groupby_query(
            GroupByQuery(
                field_name="dense",
                group_by_field_name="group_id",
                vector=query_vector,
                param=HnswQueryParam(ef=300),
                group_count=GB_NUM_GROUPS,
                group_topk=GB_GROUP_TOPK,
            )
        )
        _assert_grouped_results(results, GB_NUM_GROUPS, GB_GROUP_TOPK, query_vector)

    def test_groupby_flat(self, groupby_collection: Collection):
        """Group-by search over a FLAT index."""
        query_vector = [1.0] * GB_DIMENSION
        results = groupby_collection.groupby_query(
            GroupByQuery(
                field_name="dense_flat",
                group_by_field_name="group_id",
                vector=query_vector,
                group_count=GB_NUM_GROUPS,
                group_topk=GB_GROUP_TOPK,
            )
        )
        _assert_grouped_results(results, GB_NUM_GROUPS, GB_GROUP_TOPK, query_vector)

    def test_groupby_with_filter(self, groupby_collection: Collection):
        """Group-by search with a scalar filter."""
        query_vector = [1.0] * GB_DIMENSION
        results = groupby_collection.groupby_query(
            GroupByQuery(
                field_name="dense_flat",
                group_by_field_name="group_id",
                vector=query_vector,
                filter="id < 6",
                group_count=GB_NUM_GROUPS,
                group_topk=GB_GROUP_TOPK,
            )
        )
        # Only docs 0..5 are visible; every group still has at least one doc.
        assert len(results) == GB_NUM_GROUPS
        for group in results:
            for doc in group["docs"]:
                assert int(doc.field("id")) < 6

    def test_groupby_include_vector(self, groupby_collection: Collection):
        """Group-by search returns original vectors when requested."""
        query_vector = [1.0] * GB_DIMENSION
        results = groupby_collection.groupby_query(
            GroupByQuery(
                field_name="dense_flat",
                group_by_field_name="group_id",
                vector=query_vector,
                include_vector=True,
                group_count=GB_NUM_GROUPS,
                group_topk=GB_GROUP_TOPK,
            )
        )
        assert len(results) == GB_NUM_GROUPS
        for group in results:
            for doc in group["docs"]:
                vec = doc.vector("dense_flat")
                doc_id = int(doc.field("id"))
                assert vec == pytest.approx([float(doc_id)] * GB_DIMENSION, abs=1e-5)

    def test_groupby_output_fields(self, groupby_collection: Collection):
        """Group-by search honors scalar output field selection."""
        query_vector = [1.0] * GB_DIMENSION
        results = groupby_collection.groupby_query(
            GroupByQuery(
                field_name="dense_flat",
                group_by_field_name="group_id",
                vector=query_vector,
                output_fields=["group_id"],
                group_count=GB_NUM_GROUPS,
                group_topk=GB_GROUP_TOPK,
            )
        )
        assert len(results) == GB_NUM_GROUPS
        for group in results:
            for doc in group["docs"]:
                assert doc.has_field("group_id")

    def test_groupby_invalid_field(self, groupby_collection: Collection):
        """Group-by with a non-existent vector field raises an error."""
        with pytest.raises(ValueError):
            groupby_collection.groupby_query(
                GroupByQuery(
                    field_name="nonexistent",
                    group_by_field_name="group_id",
                    vector=[1.0] * GB_DIMENSION,
                )
            )


class TestGroupByEmptyCollection:
    def test_groupby_empty_collection(self, groupby_collection: Collection):
        """Group-by on an empty collection returns an empty list."""
        results = groupby_collection.groupby_query(
            GroupByQuery(
                field_name="dense_flat",
                group_by_field_name="group_id",
                vector=[1.0] * GB_DIMENSION,
                group_count=GB_NUM_GROUPS,
                group_topk=GB_GROUP_TOPK,
            )
        )
        assert results == []
