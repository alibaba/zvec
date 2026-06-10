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
"""End-to-end collection tests for the DiskAnn index.

Mirrors ``test_collection_hnsw_rabitq.py`` but targets the DiskAnn plugin.

Two platform-level prerequisites are enforced at module import time:

1. DiskAnn is currently built only for Linux x86_64 — other platforms are
   skipped wholesale.
2. The DiskAnn backend lives in a *runtime-loaded* plugin
   (``libzvec_diskann_plugin.so``). It must be loaded with ``RTLD_GLOBAL |
   RTLD_NOW`` BEFORE ``import zvec`` so that the plugin's ``IndexFactory``
   singleton is unified with the one inside ``_zvec.so``. After ``import
   zvec`` we must also call ``zvec.load_diskann_plugin()`` exactly once.

If either prerequisite fails the whole module is skipped so the rest of the
test-suite is not affected.
"""

from __future__ import annotations

import math
import os
import platform
import sys

import pytest

# --------------------------------------------------------------------------- #
# Platform gating (must happen BEFORE we touch zvec).
# --------------------------------------------------------------------------- #
pytestmark = pytest.mark.skipif(
    not (sys.platform == "linux" and platform.machine() in ("x86_64", "AMD64")),
    reason="DiskAnn plugin is only supported on Linux x86_64",
)

# Promote all symbols in subsequently-loaded DSOs to the global namespace and
# resolve relocations eagerly. This is REQUIRED so the DiskAnn plugin can see
# the ``IndexFactory`` singleton that lives in ``_zvec.so`` and vice versa.
# See: DiskAnn RTLD_GLOBAL + RTLD_NOW Requirement.
if sys.platform == "linux":
    sys.setdlopenflags(sys.getdlopenflags() | os.RTLD_GLOBAL | os.RTLD_NOW)

import zvec  # noqa: E402

from zvec import (  # noqa: E402
    Collection,
    CollectionOption,
    DataType,
    DiskAnnIndexParam,
    DiskAnnQueryParam,
    Doc,
    FieldSchema,
    MetricType,
    Query,
    VectorSchema,
)
from zvec.typing import QuantizeType  # noqa: E402


@pytest.fixture(scope="session")
def diskann_collection_schema():
    """Create a collection schema with a DiskAnn index."""
    return zvec.CollectionSchema(
        name="test_diskann_collection",
        fields=[
            FieldSchema("id", DataType.INT64, nullable=False),
            FieldSchema("name", DataType.STRING, nullable=False),
        ],
        vectors=[
            VectorSchema(
                "embedding",
                DataType.VECTOR_FP32,
                dimension=128,
                index_param=DiskAnnIndexParam(
                    metric_type=MetricType.L2,
                    max_degree=64,
                    list_size=100,
                    pq_chunk_num=0,
                    quantize_type=QuantizeType.UNDEFINED,
                ),
            ),
        ],
    )


@pytest.fixture(scope="session")
def collection_option():
    """Create collection options."""
    return CollectionOption(read_only=False, enable_mmap=True)


@pytest.fixture
def single_doc():
    """Create a single document for testing."""
    return Doc(
        id="0",
        fields={"id": 0, "name": "test_doc_0"},
        vectors={"embedding": [0.1 + i * 0.01 for i in range(128)]},
    )


@pytest.fixture
def multiple_docs():
    """Create multiple documents for testing."""
    return [
        Doc(
            id=f"{i}",
            fields={"id": i, "name": f"test_doc_{i}"},
            vectors={"embedding": [i * 0.1 + j * 0.01 for j in range(128)]},
        )
        for i in range(1, 101)
    ]


@pytest.fixture(scope="function")
def diskann_collection(
    tmp_path_factory, diskann_collection_schema, collection_option
) -> Collection:
    """
    Function-scoped fixture: creates and opens a collection with DiskAnn index.
    """
    temp_dir = tmp_path_factory.mktemp("zvec_diskann")
    collection_path = temp_dir / "test_diskann_collection"

    coll = zvec.create_and_open(
        path=str(collection_path),
        schema=diskann_collection_schema,
        option=collection_option,
    )

    assert coll is not None, "Failed to create and open DiskAnn collection"
    assert coll.path == str(collection_path)
    assert coll.schema.name == diskann_collection_schema.name

    try:
        yield coll
    finally:
        if hasattr(coll, "destroy") and coll is not None:
            try:
                coll.destroy()
            except Exception as e:
                print(f"Warning: failed to destroy collection: {e}")


@pytest.fixture
def collection_with_single_doc(
    diskann_collection: Collection, single_doc: Doc
) -> Collection:
    """Setup: insert single doc into collection."""
    assert diskann_collection.stats.doc_count == 0
    result = diskann_collection.insert(single_doc)
    assert bool(result)
    assert result.ok()
    assert diskann_collection.stats.doc_count == 1

    yield diskann_collection

    # Teardown: delete single doc
    diskann_collection.delete(single_doc.id)
    assert diskann_collection.stats.doc_count == 0


@pytest.fixture
def collection_with_multiple_docs(
    diskann_collection: Collection, multiple_docs: list[Doc]
) -> Collection:
    """Setup: insert multiple docs into collection."""
    assert diskann_collection.stats.doc_count == 0
    result = diskann_collection.insert(multiple_docs)
    assert len(result) == len(multiple_docs)
    for item in result:
        assert item.ok()
    assert diskann_collection.stats.doc_count == len(multiple_docs)

    yield diskann_collection

    # Teardown: delete multiple docs
    diskann_collection.delete([doc.id for doc in multiple_docs])


# ==================== Tests ====================


@pytest.mark.usefixtures("diskann_collection")
class TestDiskAnnCollectionCreation:
    """Test DiskAnn collection creation and schema validation."""

    def test_collection_creation(
        self, diskann_collection: Collection, diskann_collection_schema
    ):
        """Test that collection is created with correct schema."""
        assert diskann_collection is not None
        assert diskann_collection.schema.name == diskann_collection_schema.name
        assert len(diskann_collection.schema.fields) == len(
            diskann_collection_schema.fields
        )
        assert len(diskann_collection.schema.vectors) == len(
            diskann_collection_schema.vectors
        )

    def test_vector_schema_validation(self, diskann_collection: Collection):
        """Test that vector schema has correct DiskAnn configuration."""
        vector_schema = diskann_collection.schema.vector("embedding")
        assert vector_schema is not None
        assert vector_schema.name == "embedding"
        assert vector_schema.data_type == DataType.VECTOR_FP32
        assert vector_schema.dimension == 128

        index_param = vector_schema.index_param
        assert index_param is not None
        assert index_param.metric_type == MetricType.L2
        assert index_param.max_degree == 64
        assert index_param.list_size == 100
        assert index_param.pq_chunk_num == 0

    def test_collection_stats(self, diskann_collection: Collection):
        """Test initial collection statistics."""
        stats = diskann_collection.stats
        assert stats is not None
        assert stats.doc_count == 0
        assert len(stats.index_completeness) == 1
        assert stats.index_completeness["embedding"] == 1


@pytest.mark.usefixtures("diskann_collection")
class TestDiskAnnCollectionInsert:
    """Test document insertion into DiskAnn collection."""

    def test_insert_single_doc(self, diskann_collection: Collection, single_doc: Doc):
        """Test inserting a single document."""
        result = diskann_collection.insert(single_doc)
        assert bool(result)
        assert result.ok()

        stats = diskann_collection.stats
        assert stats is not None
        assert stats.doc_count == 1

    def test_insert_multiple_docs(
        self, diskann_collection: Collection, multiple_docs: list[Doc]
    ):
        """Test inserting multiple documents."""
        result = diskann_collection.insert(multiple_docs)
        assert len(result) == len(multiple_docs)
        for item in result:
            assert item.ok()

        stats = diskann_collection.stats
        assert stats is not None
        assert stats.doc_count == len(multiple_docs)


@pytest.mark.usefixtures("diskann_collection")
class TestDiskAnnCollectionFetch:
    """Test document fetching from DiskAnn collection."""

    def test_fetch_single_doc(
        self, collection_with_single_doc: Collection, single_doc: Doc
    ):
        """Test fetching a single document by ID."""
        result = collection_with_single_doc.fetch(ids=[single_doc.id])
        assert bool(result)
        assert single_doc.id in result.keys()

        doc = result[single_doc.id]
        assert doc is not None
        assert doc.id == single_doc.id
        assert doc.field("id") == single_doc.field("id")
        assert doc.field("name") == single_doc.field("name")

    def test_fetch_multiple_docs(
        self, collection_with_multiple_docs: Collection, multiple_docs: list[Doc]
    ):
        """Test fetching multiple documents by IDs."""
        ids = [doc.id for doc in multiple_docs[:10]]
        result = collection_with_multiple_docs.fetch(ids=ids)
        assert bool(result)
        assert len(result) == len(ids)

        for doc_id in ids:
            assert doc_id in result
            doc = result[doc_id]
            assert doc is not None
            assert doc.id == doc_id

    def test_fetch_nonexistent_doc(self, collection_with_single_doc: Collection):
        """Test fetching a non-existent document."""
        result = collection_with_single_doc.fetch(ids=["nonexistent_id"])
        assert len(result) == 0


@pytest.mark.usefixtures("diskann_collection")
class TestDiskAnnCollectionQuery:
    """Test vector search queries on DiskAnn collection."""

    def test_query_by_vector(
        self, collection_with_multiple_docs: Collection, multiple_docs: list[Doc]
    ):
        """Test querying by vector with DiskAnn index."""
        query_vector = multiple_docs[0].vector("embedding")
        query = Query(
            field_name="embedding",
            vector=query_vector,
            param=DiskAnnQueryParam(list_size=100),
        )

        result = collection_with_multiple_docs.query(queries=query, topk=10)
        assert len(result) > 0
        assert len(result) <= 10

        # First result should be the query document itself (or very close)
        first_doc = result[0]
        assert first_doc is not None
        assert first_doc.id is not None

    def test_query_by_id(
        self, collection_with_multiple_docs: Collection, multiple_docs: list[Doc]
    ):
        """Test querying by document ID with DiskAnn index."""
        query = Query(
            field_name="embedding",
            id=multiple_docs[0].id,
            param=DiskAnnQueryParam(list_size=100),
        )

        result = collection_with_multiple_docs.query(queries=query, topk=10)
        assert len(result) > 0
        assert len(result) <= 10

    def test_query_with_different_list_size(
        self, collection_with_multiple_docs: Collection, multiple_docs: list[Doc]
    ):
        """Test querying with different list_size parameter values."""
        query_vector = multiple_docs[0].vector("embedding")

        # Test with list_size=50
        query_small = Query(
            field_name="embedding",
            vector=query_vector,
            param=DiskAnnQueryParam(list_size=50),
        )
        result_small = collection_with_multiple_docs.query(queries=query_small, topk=10)
        assert len(result_small) > 0

        # Test with list_size=200
        query_large = Query(
            field_name="embedding",
            vector=query_vector,
            param=DiskAnnQueryParam(list_size=200),
        )
        result_large = collection_with_multiple_docs.query(queries=query_large, topk=10)
        assert len(result_large) > 0

    def test_query_with_topk(
        self, collection_with_multiple_docs: Collection, multiple_docs: list[Doc]
    ):
        """Test querying with different topk values."""
        query_vector = multiple_docs[0].vector("embedding")
        query = Query(
            field_name="embedding",
            vector=query_vector,
            param=DiskAnnQueryParam(list_size=100),
        )

        # Test topk=5
        result_5 = collection_with_multiple_docs.query(queries=query, topk=5)
        assert len(result_5) <= 5

        # Test topk=20
        result_20 = collection_with_multiple_docs.query(queries=query, topk=20)
        assert len(result_20) <= 20

    def test_query_with_filter(
        self, collection_with_multiple_docs: Collection, multiple_docs: list[Doc]
    ):
        """Test querying with filter conditions."""
        query_vector = multiple_docs[0].vector("embedding")
        query = Query(
            field_name="embedding",
            vector=query_vector,
            param=DiskAnnQueryParam(list_size=100),
        )

        # Query with id filter
        result = collection_with_multiple_docs.query(
            queries=query, topk=10, filter="id < 50"
        )
        assert len(result) > 0
        for doc in result:
            assert doc.field("id") < 50

    def test_query_with_output_fields(
        self, collection_with_multiple_docs: Collection, multiple_docs: list[Doc]
    ):
        """Test querying with specific output fields."""
        query_vector = multiple_docs[0].vector("embedding")
        query = Query(
            field_name="embedding",
            vector=query_vector,
            param=DiskAnnQueryParam(list_size=100),
        )

        result = collection_with_multiple_docs.query(
            queries=query, topk=10, output_fields=["id", "name"]
        )
        assert len(result) > 0

        first_doc = result[0]
        assert "id" in first_doc.field_names()
        assert "name" in first_doc.field_names()

    def test_query_with_include_vector(
        self, collection_with_multiple_docs: Collection, multiple_docs: list[Doc]
    ):
        """Test querying with vector data included in results."""
        query_vector = multiple_docs[0].vector("embedding")
        query = Query(
            field_name="embedding",
            vector=query_vector,
            param=DiskAnnQueryParam(list_size=100),
        )

        result = collection_with_multiple_docs.query(
            queries=query, topk=10, include_vector=True
        )
        assert len(result) > 0

        first_doc = result[0]
        assert first_doc.vector("embedding") is not None
        assert len(first_doc.vector("embedding")) == 128


@pytest.mark.usefixtures("diskann_collection")
class TestDiskAnnCollectionUpdate:
    """Test document update in DiskAnn collection."""

    def test_update_doc_fields(
        self, collection_with_single_doc: Collection, single_doc: Doc
    ):
        """Test updating document fields."""
        updated_doc = Doc(
            id=single_doc.id,
            fields={"id": single_doc.field("id"), "name": "updated_name"},
        )

        result = collection_with_single_doc.update(updated_doc)
        assert bool(result)
        assert result.ok()

        # Verify update
        fetched = collection_with_single_doc.fetch(ids=[single_doc.id])
        assert single_doc.id in fetched
        doc = fetched[single_doc.id]
        assert doc.field("name") == "updated_name"

    def test_update_doc_vector(
        self, collection_with_single_doc: Collection, single_doc: Doc
    ):
        """Test updating document vector."""
        new_vector = [0.5 + i * 0.01 for i in range(128)]
        updated_doc = Doc(
            id=single_doc.id,
            vectors={"embedding": new_vector},
        )

        result = collection_with_single_doc.update(updated_doc)
        assert bool(result)
        assert result.ok()

        # Verify update
        fetched = collection_with_single_doc.fetch(
            ids=[single_doc.id],
        )
        assert single_doc.id in fetched
        doc = fetched[single_doc.id]
        assert doc.vector("embedding") is not None
        embedding = doc.vector("embedding")
        assert len(embedding) == 128
        # Verify vector values are approximately equal (float comparison)
        for i in range(128):
            assert math.isclose(embedding[i], new_vector[i], rel_tol=1e-5)


@pytest.mark.usefixtures("diskann_collection")
class TestDiskAnnCollectionDelete:
    """Test document deletion from DiskAnn collection."""

    def test_delete_single_doc(
        self, collection_with_single_doc: Collection, single_doc: Doc
    ):
        """Test deleting a single document."""
        result = collection_with_single_doc.delete(single_doc.id)
        assert bool(result)
        assert result.ok()

        stats = collection_with_single_doc.stats
        assert stats.doc_count == 0

    def test_delete_multiple_docs(
        self, collection_with_multiple_docs: Collection, multiple_docs: list[Doc]
    ):
        """Test deleting multiple documents."""
        ids_to_delete = [doc.id for doc in multiple_docs[:10]]
        result = collection_with_multiple_docs.delete(ids_to_delete)
        assert len(result) == len(ids_to_delete)
        for item in result:
            assert item.ok()

        stats = collection_with_multiple_docs.stats
        assert stats.doc_count == len(multiple_docs) - len(ids_to_delete)


@pytest.mark.usefixtures("diskann_collection")
class TestDiskAnnCollectionOptimizeAndReopen:
    """Test collection optimize and reopen functionality."""

    def test_optimize_close_reopen_and_query(
        self,
        tmp_path_factory,
        diskann_collection_schema,
        collection_option,
        multiple_docs: list[Doc],
    ):
        """Test inserting 100 docs, optimize, close, reopen and query."""
        # Create collection and insert 100 documents
        temp_dir = tmp_path_factory.mktemp("zvec_diskann_optimize")
        collection_path = temp_dir / "test_optimize_collection"

        coll = zvec.create_and_open(
            path=str(collection_path),
            schema=diskann_collection_schema,
            option=collection_option,
        )

        assert coll is not None
        assert coll.stats.doc_count == 0

        # Insert 100 documents
        result = coll.insert(multiple_docs)
        assert len(result) == len(multiple_docs)
        for item in result:
            assert item.ok()
        assert coll.stats.doc_count == len(multiple_docs)

        # Call optimize
        from zvec import OptimizeOption

        coll.optimize(option=OptimizeOption())

        # Verify data is still accessible after optimize
        query_vector = multiple_docs[0].vector("embedding")
        query = Query(
            field_name="embedding",
            vector=query_vector,
            param=DiskAnnQueryParam(list_size=100),
        )
        result_before_close = coll.query(queries=query, topk=10)
        assert len(result_before_close) > 0

        # Close collection (destroy will close it)
        collection_path_str = str(collection_path)
        del coll

        # Reopen collection
        reopened_coll = zvec.open(path=collection_path_str, option=collection_option)
        assert reopened_coll is not None
        assert reopened_coll.stats.doc_count == len(multiple_docs)

        # Execute query on reopened collection
        query_after_reopen = Query(
            field_name="embedding",
            vector=query_vector,
            param=DiskAnnQueryParam(list_size=100),
        )
        result_after_reopen = reopened_coll.query(queries=query_after_reopen, topk=10)
        assert len(result_after_reopen) > 0
        assert len(result_after_reopen) <= 10

        # Verify query results are valid
        first_doc = result_after_reopen[0]
        assert first_doc is not None
        assert first_doc.id is not None
        assert first_doc.field("id") is not None
        assert first_doc.field("name") is not None

        # Cleanup
        reopened_coll.destroy()
