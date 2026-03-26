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


import threading
import numpy as np

from fixture_helper import *

COLLECTION_OPTION_TEST_CASES_VALID = [
    # (read_only, enable_mmap, description)
    (False, True, "Read-write with mmap enabled"),
    (False, False, "Read-write with mmap disabled"),
    (True, True, "Read-only with mmap enabled"),
    (True, False, "Read-only with mmap disabled"),
]

# Test data for invalid paths
INVALID_PATH_LIST = [
    "/nonexistent/directory/test_collection",
    "invalid:path",
    "",  # Empty path
]


@pytest.fixture(scope="session")
def collection_schema():
    return zvec.CollectionSchema(
        name="test_collection",
        fields=[
            FieldSchema(
                "id",
                DataType.INT64,
                nullable=False,
                index_param=InvertIndexParam(enable_range_optimization=True),
            ),
            FieldSchema(
                "name", DataType.STRING, nullable=False, index_param=InvertIndexParam()
            ),
            FieldSchema(
                "weight", DataType.FLOAT, nullable=False, index_param=InvertIndexParam()
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


@pytest.fixture
def single_doc():
    id = 0
    return Doc(
        id=f"{id}",
        fields={"id": id, "name": "test"},
        vectors={
            "dense": [id + 0.1] * 128,
        },
    )


@pytest.fixture(scope="function")
def test_collection(
    tmp_path_factory, collection_schema, collection_option
) -> Generator[Any, Any, Collection]:
    temp_dir = tmp_path_factory.mktemp("zvec")
    collection_path = temp_dir / "test_collection"

    coll = zvec.create_and_open(
        path=str(collection_path), schema=collection_schema, option=collection_option
    )

    assert coll is not None, "Failed to create and open collection"
    assert coll.path == str(collection_path)
    assert coll.schema.name == collection_schema.name
    assert list(coll.schema.fields) == list(collection_schema.fields)
    assert list(coll.schema.vectors) == list(collection_schema.vectors)
    assert coll.option.read_only == collection_option.read_only
    assert coll.option.enable_mmap == collection_option.enable_mmap

    try:
        yield coll
    finally:
        if hasattr(coll, "destroy") and coll is not None:
            try:
                coll.destroy()
            except Exception as e:
                print(f"Warning: failed to destroy collection: {e}")


class TestCollectionOpen:
    def test_open_basic_functionality(
        self, tmp_path_factory, collection_schema, collection_option
    ):
        import sys
        import time
        import os

        # Create unique temp directory
        temp_dir = tmp_path_factory.mktemp("zvec")
        collection_path = temp_dir / "test_collection"

        # Ensure the path exists
        collection_path_str = str(collection_path)
        print(f"DEBUG: Collection path: {collection_path_str}")
        print(f"DEBUG: Temp directory exists: {temp_dir.exists()}")

        # Create and open collection first
        created_coll = zvec.create_and_open(
            path=collection_path_str, schema=collection_schema, option=collection_option
        )

        assert created_coll is not None, (
            f"Failed to create collection, returned None instead of valid Collection object. Path: {collection_path_str}"
        )
        assert created_coll.path == collection_path_str, (
            f"Collection path mismatch. Expected: {collection_path_str}, Actual: {created_coll.path}"
        )
        assert created_coll.schema.name == "test_collection", (
            f"Collection schema name mismatch. Expected: test_collection, Actual: {created_coll.schema.name}"
        )

        # Insert multiple documents to verify persistence
        docs = []
        for i in range(3):
            doc = Doc(
                id=f"{i}",
                fields={"id": i, "name": f"test_{i}", "weight": float(i * 10)},
                vectors={
                    "dense": [float(j + i) for j in range(128)],
                    "sparse": {j: float(j + i) for j in range(5)},
                },
            )
            docs.append(doc)

        result = created_coll.insert(docs)
        assert len(result) == 3, f"Expected 3 insertion results, but got {len(result)}"
        for i, res in enumerate(result):
            assert res.ok(), (
                f"Insertion result {i} is not OK. Status code: {res.code()}, Message: {res.message()}"
            )

        # Verify documents were inserted using fetch interface
        fetched_docs_after_insert = created_coll.fetch(["0", "1", "2"])
        assert len(fetched_docs_after_insert) == 3, (
            f"Expected 3 fetched documents after insertion, but got {len(fetched_docs_after_insert)}"
        )
        assert "0" in fetched_docs_after_insert, (
            "Document with ID '0' not found in fetched results after insertion"
        )
        assert "1" in fetched_docs_after_insert, (
            "Document with ID '1' not found in fetched results after insertion"
        )
        assert "2" in fetched_docs_after_insert, (
            "Document with ID '2' not found in fetched results after insertion"
        )

        # Verify fetched document content after insertion
        for i in range(3):
            doc = fetched_docs_after_insert[f"{i}"]
            assert doc is not None, (
                f"Fetched document with ID '{i}' is None after insertion"
            )
            assert doc.id == f"{i}", (
                f"Document ID mismatch for document '{i}' after insertion. Expected: {i}, Actual: {doc.id}"
            )
            assert doc.field("id") == i, (
                f"Document id field mismatch for document '{i}' after insertion. Expected: {i}, Actual: {doc.field('id')}"
            )
            assert doc.field("name") == f"test_{i}", (
                f"Document name field mismatch for document '{i}' after insertion. Expected: test_{i}, Actual: {doc.field('name')}"
            )
            assert doc.field("weight") == float(i * 10), (
                f"Document weight field mismatch for document '{i}' after insertion. Expected: {float(i * 10)}, Actual: {doc.field('weight')}"
            )

            # Verify vector access after insertion
            assert doc.vector("dense") is not None, (
                f"Document {i} should have dense vector after insertion"
            )
            assert doc.vector("sparse") is not None, (
                f"Document {i} should have sparse vector after insertion"
            )

            # Verify vector types after insertion
            assert isinstance(doc.vector("dense"), list), (
                f"Document {i} dense vector should be dict after insertion, got {type(doc.vector('dense'))}"
            )
            assert isinstance(doc.vector("sparse"), dict), (
                f"Document {i} sparse vector should be dict after insertion, got {type(doc.vector('sparse'))}"
            )

        # Verify documents were inserted using stats
        stats = created_coll.stats
        assert stats is not None, "Collection stats should not be None"
        assert stats.doc_count == 3, (
            f"Document count mismatch after insertion. Expected: 3, Actual: {stats.doc_count}"
        )

        # Store the collection path before cleanup
        collection_path = created_coll.path

        # Clean up the created collection reference
        del created_coll

        # Wait and verify the path still exists
        print(f"DEBUG: Collection path after destroy: {collection_path}")
        print(f"DEBUG: Path exists after destroy: {os.path.exists(collection_path)}")

        # Now open the existing collection
        try:
            print(f"DEBUG: Path exists before open: {os.path.exists(collection_path)}")

            # List contents of parent directory for debugging
            parent_dir = os.path.dirname(collection_path)
            if os.path.exists(parent_dir):
                print(f"DEBUG: Parent directory contents: {os.listdir(parent_dir)}")

            opened_coll = zvec.open(path=collection_path, option=collection_option)

            assert opened_coll is not None, (
                f"Failed to open existing collection at path: {collection_path}. Returned None instead of valid Collection object"
            )
            assert opened_coll.path == collection_path, (
                f"Opened collection path mismatch. Expected: {collection_path}, Actual: {opened_coll.path}"
            )
            assert opened_coll.schema.name == "test_collection", (
                f"Opened collection schema name mismatch. Expected: test_collection, Actual: {opened_coll.schema.name}"
            )

            # Check reference count of opened collection
            opened_ref_count = sys.getrefcount(opened_coll)
            print(f"DEBUG: Reference count of opened collection: {opened_ref_count}")

            # Verify data persistence
            # Verify data persistence using fetch interface
            fetched_docs = opened_coll.fetch(["0", "1", "2"])
            assert len(fetched_docs) == 3, (
                f"Expected 3 fetched documents after reopening, but got {len(fetched_docs)}"
            )
            assert "0" in fetched_docs, (
                "Document with ID '0' not found in fetched results after reopening"
            )
            assert "1" in fetched_docs, (
                "Document with ID '1' not found in fetched results after reopening"
            )
            assert "2" in fetched_docs, (
                "Document with ID '2' not found in fetched results after reopening"
            )

            # Verify fetched document content after reopening collection
            for i in range(3):
                doc = fetched_docs[f"{i}"]
                assert doc is not None, (
                    f"Fetched document with ID '{i}' is None after reopening collection"
                )
                assert doc.id == f"{i}", (
                    f"Document ID mismatch for document '{i}' after reopening. Expected: {i}, Actual: {doc.id}"
                )
                assert doc.field("id") == i, (
                    f"Document id field mismatch for document '{i}' after reopening. Expected: {i}, Actual: {doc.field('id')}"
                )
                assert doc.field("name") == f"test_{i}", (
                    f"Document name field mismatch for document '{i}' after reopening. Expected: test_{i}, Actual: {doc.field('name')}"
                )
                assert doc.field("weight") == float(i * 10), (
                    f"Document weight field mismatch for document '{i}' after reopening. Expected: {float(i * 10)}, Actual: {doc.field('weight')}"
                )

                # Verify vector access after reopening
                assert doc.vector("dense") is not None, (
                    f"Document {i} should have dense vector after reopening"
                )
                assert doc.vector("sparse") is not None, (
                    f"Document {i} should have sparse vector after reopening"
                )

                # Verify vector types after reopening
                assert isinstance(doc.vector("dense"), list), (
                    f"Document {i} dense vector should be dict after reopening, got {type(doc.vector('dense'))}"
                )
                assert isinstance(doc.vector("sparse"), dict), (
                    f"Document {i} sparse vector should be dict after reopening, got {type(doc.vector('sparse'))}"
                )

                # Verify score attribute exists
                assert hasattr(doc, "score"), (
                    f"Document {i} should have a score attribute after reopening"
                )
                assert isinstance(doc.score, (int, float)), (
                    f"Document {i} score should be numeric after reopening, got {type(doc.score)}"
                )
                # For fetch operations, score is typically 0.0
                assert doc.score == 0.0, (
                    f"Document {i} score should be 0.0 for fetch operation after reopening, but got {doc.score}"
                )

            # Test query functionality
            query_result = opened_coll.query(include_vector=True)
            assert len(query_result) == 3, (
                f"Expected 3 query results, but got {len(query_result)}"
            )

            # Verify query results have proper structure and content with detailed validation
            returned_doc_ids = set()
            for doc in query_result:
                # Verify basic document structure
                assert doc.id is not None, f"Query result document should have an ID"
                assert doc.id in ["0", "1", "2"], (
                    f"Query result document ID should be one of ['0', '1', '2'], but got {doc.id}"
                )
                returned_doc_ids.add(doc.id)

                # Verify field access
                assert doc.field("id") is not None, (
                    f"Document {doc.id} should have id field"
                )
                assert doc.field("name") is not None, (
                    f"Document {doc.id} should have name field"
                )
                assert doc.field("weight") is not None, (
                    f"Document {doc.id} should have weight field"
                )

                # Verify field values
                expected_id = int(doc.id)
                assert doc.field("id") == expected_id, (
                    f"Document {doc.id} id field mismatch. Expected: {expected_id}, Actual: {doc.field('id')}"
                )
                assert doc.field("name") == f"test_{expected_id}", (
                    f"Document {doc.id} name field mismatch. Expected: test_{expected_id}, Actual: {doc.field('name')}"
                )
                assert doc.field("weight") == float(expected_id * 10), (
                    f"Document {doc.id} weight field mismatch. Expected: {float(expected_id * 10)}, Actual: {doc.field('weight')}"
                )

                # Verify vector access
                assert doc.vector("dense") is not None, (
                    f"Document {doc.id} should have dense vector"
                )
                assert doc.vector("sparse") is not None, (
                    f"Document {doc.id} should have sparse vector"
                )

                # Verify vector types
                assert isinstance(doc.vector("dense"), list), (
                    f"Document {doc.id} dense vector should be list, got {type(doc.vector('dense'))}"
                )
                assert isinstance(doc.vector("sparse"), dict), (
                    f"Document {doc.id} sparse vector should be dict, got {type(doc.vector('sparse'))}"
                )

                # Verify score attribute exists
                assert hasattr(doc, "score"), (
                    f"Document {doc.id} should have a score attribute"
                )
                assert isinstance(doc.score, (int, float)), (
                    f"Document {doc.id} score should be numeric, got {type(doc.score)}"
                )

            # Verify all expected documents are returned
            expected_doc_ids = {"0", "1", "2"}
            assert returned_doc_ids == expected_doc_ids, (
                f"Query should return all expected documents. Expected: {expected_doc_ids}, Actual: {returned_doc_ids}"
            )

            # === Enhanced validation based on test_collection_dql_operations.py ===

            # Verify vector field names accessibility for all documents
            for doc in query_result:
                vector_names = doc.vector_names()
                expected_vector_names = {"dense", "sparse"}
                assert set(vector_names) == expected_vector_names, (
                    f"Document {doc.id} vector names mismatch. Expected: {expected_vector_names}, Actual: {set(vector_names)}"
                )

                # Verify all vector fields can be accessed
                for vector_name in expected_vector_names:
                    vector_data = doc.vector(vector_name)
                    assert vector_data is not None, (
                        f"Document {doc.id} should have accessible vector '{vector_name}'"
                    )
                    if vector_name == "dense":
                        assert isinstance(vector_data, list), (
                            f"Document {doc.id} vector '{vector_name}' should be list, got {type(vector_data)}"
                        )
                    else:
                        assert isinstance(vector_data, dict), (
                            f"Document {doc.id} vector '{vector_name}' should be dict, got {type(vector_data)}"
                        )

            # Test query with filter
            filtered_result = opened_coll.query(filter="id >= 1", include_vector=True)
            assert len(filtered_result) == 2, (
                f"Expected 2 filtered query results (id >= 1), but got {len(filtered_result)}"
            )

            # Verify filtered query results
            filtered_doc_ids = set()
            for doc in filtered_result:
                assert doc.id is not None, (
                    f"Filtered query result document should have an ID"
                )
                assert doc.id in ["1", "2"], (
                    f"Filtered query result document ID should be one of ['1', '2'], but got {doc.id}"
                )
                filtered_doc_ids.add(doc.id)

                # Verify filter condition is satisfied
                doc_id = int(doc.id)
                assert doc_id >= 1, (
                    f"Document {doc.id} should satisfy filter condition id >= 1"
                )

                # Verify document structure
                assert doc.field("id") is not None, (
                    f"Document {doc.id} should have id field"
                )
                assert doc.field("name") is not None, (
                    f"Document {doc.id} should have name field"
                )
                assert doc.field("weight") is not None, (
                    f"Document {doc.id} should have weight field"
                )

                # Verify field values
                assert doc.field("id") == doc_id, (
                    f"Document {doc.id} id field mismatch. Expected: {doc_id}, Actual: {doc.field('id')}"
                )
                assert doc.field("name") == f"test_{doc_id}", (
                    f"Document {doc.id} name field mismatch. Expected: test_{doc_id}, Actual: {doc.field('name')}"
                )
                assert doc.field("weight") == float(doc_id * 10), (
                    f"Document {doc.id} weight field mismatch. Expected: {float(doc_id * 10)}, Actual: {doc.field('weight')}"
                )

                # Verify vector access
                assert doc.vector("dense") is not None, (
                    f"Document {doc.id} should have dense vector"
                )
                assert doc.vector("sparse") is not None, (
                    f"Document {doc.id} should have sparse vector"
                )

                # Verify score attribute exists
                assert hasattr(doc, "score"), (
                    f"Document {doc.id} should have a score attribute"
                )
                assert isinstance(doc.score, (int, float)), (
                    f"Document {doc.id} score should be numeric, got {type(doc.score)}"
                )

            # Verify filtered documents
            expected_filtered_ids = {"1", "2"}
            assert filtered_doc_ids == expected_filtered_ids, (
                f"Filtered query should return expected documents. Expected: {expected_filtered_ids}, Actual: {filtered_doc_ids}"
            )

            # Test vector query functionality for dense vectors
            query_vector_dense = [0.1] * 128
            vector_query_result = opened_coll.query(
                VectorQuery(field_name="dense", vector=query_vector_dense)
            )
            assert len(vector_query_result) > 0, (
                f"Expected at least 1 vector query result, but got {len(vector_query_result)}"
            )

            # Verify vector query results structure
            for doc in vector_query_result[:3]:  # Check first 3 results
                assert doc.id is not None, (
                    f"Vector query result document should have an ID"
                )
                assert doc.id in ["0", "1", "2"], (
                    f"Vector query result document ID should be one of ['0', '1', '2'], but got {doc.id}"
                )

                # Verify document structure
                assert doc.field("id") is not None, (
                    f"Document {doc.id} should have id field"
                )
                assert doc.field("name") is not None, (
                    f"Document {doc.id} should have name field"
                )
                assert doc.field("weight") is not None, (
                    f"Document {doc.id} should have weight field"
                )

                # Verify vector access
                assert doc.vector("dense") is not None, (
                    f"Document {doc.id} should have dense vector"
                )
                assert doc.vector("sparse") is not None, (
                    f"Document {doc.id} should have sparse vector"
                )

                # Verify score attribute exists and is numeric
                assert hasattr(doc, "score"), (
                    f"Document {doc.id} should have a score attribute"
                )
                assert isinstance(doc.score, (int, float)), (
                    f"Document {doc.id} score should be numeric, got {type(doc.score)}"
                )

                # For dense vector queries, score should typically be non-negative (depending on metric)
                # Note: This may vary based on the metric type used
                assert doc.score >= 0 or doc.score < 0, (
                    f"Document {doc.id} score should be a valid number"
                )

            # Test vector query functionality for sparse vectors
            query_vector_sparse = {1: 1.0, 2: 2.0, 3: 3.0}
            sparse_vector_query_result = opened_coll.query(
                VectorQuery(field_name="sparse", vector=query_vector_sparse)
            )
            assert len(sparse_vector_query_result) > 0, (
                f"Expected at least 1 sparse vector query result, but got {len(sparse_vector_query_result)}"
            )

            # Verify sparse vector query results structure
            for doc in sparse_vector_query_result[:3]:  # Check first 3 results
                assert doc.id is not None, (
                    f"Sparse vector query result document should have an ID"
                )
                assert doc.id in ["0", "1", "2"], (
                    f"Sparse vector query result document ID should be one of ['0', '1', '2'], but got {doc.id}"
                )

                # Verify document structure
                assert doc.field("id") is not None, (
                    f"Document {doc.id} should have id field"
                )
                assert doc.field("name") is not None, (
                    f"Document {doc.id} should have name field"
                )
                assert doc.field("weight") is not None, (
                    f"Document {doc.id} should have weight field"
                )

                # Verify vector access
                assert doc.vector("dense") is not None, (
                    f"Document {doc.id} should have dense vector"
                )
                assert doc.vector("sparse") is not None, (
                    f"Document {doc.id} should have sparse vector"
                )

                # Verify score attribute exists and is numeric
                assert hasattr(doc, "score"), (
                    f"Document {doc.id} should have a score attribute"
                )
                assert isinstance(doc.score, (int, float)), (
                    f"Document {doc.id} score should be numeric, got {type(doc.score)}"
                )

            # Clean up
            if hasattr(opened_coll, "destroy") and opened_coll is not None:
                opened_coll.destroy()
                print("DEBUG: Opened collection destroyed successfully")

        except Exception as e:
            logging.error("Exception occurred: [{}]".format(e))
            raise e

    @pytest.mark.parametrize(
        "read_only,enable_mmap,description", COLLECTION_OPTION_TEST_CASES_VALID
    )
    @pytest.mark.parametrize("createAndopen_enable_mmap", [True, False])
    def test_open_with_different_collection_options_valid(
        self,
        tmp_path_factory,
        createAndopen_enable_mmap,
        read_only,
        enable_mmap,
        description,
        collection_schema,
    ):
        # Create collection with initial option
        temp_dir = tmp_path_factory.mktemp("zvec")
        collection_path = temp_dir / "test_collection"

        initial_option = CollectionOption(
            read_only=False, enable_mmap=createAndopen_enable_mmap
        )

        # Create and open collection first
        created_coll = zvec.create_and_open(
            path=str(collection_path), schema=collection_schema, option=initial_option
        )

        assert created_coll is not None, "Failed to create collection"

        # Clean up the created collection reference
        del created_coll

        # Now open with different options
        collection_option = CollectionOption(
            read_only=read_only, enable_mmap=enable_mmap
        )

        try:
            opened_coll = zvec.open(path=str(collection_path), option=collection_option)

            assert opened_coll is not None, (
                f"Failed to open collection with option: {description}. Returned None instead of valid Collection object. Path: {collection_path}"
            )
            assert opened_coll.path == str(collection_path), (
                f"Opened collection path mismatch. Expected: {collection_path}, Actual: {opened_coll.path}"
            )
            assert opened_coll.schema.name == collection_schema.name, (
                f"Opened collection schema name mismatch. Expected: {collection_schema.name}, Actual: {opened_coll.schema.name}"
            )
            assert opened_coll.option.read_only == read_only, (
                f"Opened collection read_only option mismatch. Expected: {read_only}, Actual: {opened_coll.option.read_only}"
            )
            assert opened_coll.option.enable_mmap == createAndopen_enable_mmap, (
                f"Opened collection mmap option mismatch. Expected: {createAndopen_enable_mmap}, Actual: {opened_coll.option.enable_mmap}"
            )

            # Clean up
            if (
                hasattr(opened_coll, "destroy")
                and opened_coll is not None
                and read_only == False
            ):
                opened_coll.destroy()

        except Exception as e:
            logging.error("Exception occurred: [{}]".format(e))
            pytest.fail(f"Failed to open collection with different options: {e}")

    def test_open_with_none_option(self, tmp_path_factory, collection_schema):
        # Create collection
        temp_dir = tmp_path_factory.mktemp("zvec")
        collection_path = temp_dir / "test_collection"

        initial_option = CollectionOption(read_only=False, enable_mmap=True)

        # Create and open collection first
        created_coll = zvec.create_and_open(
            path=str(collection_path), schema=collection_schema, option=initial_option
        )

        assert created_coll is not None, (
            f"Failed to create collection. Returned None instead of valid Collection object. Path: {collection_path}"
        )

        # Clean up the created collection reference
        del created_coll

        # Now open with None option
        with pytest.raises(Exception) as exc_info:
            zvec.open(path=str(collection_path), option=None)

        assert "incompatible function arguments" in str(exc_info.value), (
            f"Expected 'incompatible function arguments' error, but got: {exc_info.value}"
        )

    def test_reopen_collection(self, tmp_path_factory):
        # Prepare schema
        collection_schema = zvec.CollectionSchema(
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
                    "description",
                    DataType.STRING,
                    nullable=True,
                    index_param=InvertIndexParam(),
                ),
            ],
            vectors=[
                VectorSchema(
                    "dense",
                    DataType.VECTOR_FP32,
                    dimension=128,
                    index_param=HnswIndexParam(),
                )
            ],
        )

        collection_option = CollectionOption(read_only=False, enable_mmap=True)

        # Create collection
        temp_dir = tmp_path_factory.mktemp("zvec")
        collection_path = temp_dir / "test_collection"

        # Create and open collection
        coll1 = zvec.create_and_open(
            path=str(collection_path),
            schema=collection_schema,
            option=collection_option,
        )

        assert coll1 is not None, "Failed to create and open collection"

        # Insert some data
        doc = Doc(
            id="1",
            fields={"id": 1, "name": "test", "description": "这是一个中文描述。"},
            vectors={"dense": np.random.random(128).tolist()},
        )

        result = coll1.insert(doc)
        assert result.ok()

        # Close the first collection (delete reference)
        del coll1

        # Reopen the collection
        coll2 = zvec.open(path=str(collection_path), option=collection_option)

        assert coll2 is not None, "Failed to reopen collection"
        assert coll2.path == str(collection_path)
        assert coll2.schema.name == collection_schema.name

        # Verify data is still there
        fetched_docs = coll2.fetch(["1"])
        assert "1" in fetched_docs
        fetched_doc = fetched_docs["1"]
        assert fetched_doc.id == "1"
        assert fetched_doc.field("name") == "test"
        assert fetched_doc.field("description") == "这是一个中文描述。"

        # Clean up
        if hasattr(coll2, "destroy") and coll2 is not None:
            try:
                coll2.destroy()
            except Exception as e:
                print(f"Warning: failed to destroy collection: {e}")

    def test_open_concurrent_same_path(self, tmp_path_factory):
        # First create a collection
        collection_schema = zvec.CollectionSchema(
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
            ],
            vectors=[
                VectorSchema(
                    "dense",
                    DataType.VECTOR_FP32,
                    dimension=128,
                    index_param=HnswIndexParam(),
                )
            ],
        )

        collection_option = CollectionOption(read_only=False, enable_mmap=True)

        # Create collection path
        temp_dir = tmp_path_factory.mktemp("zvec")
        collection_path = temp_dir / "test_collection"

        # First create the collection
        created_coll = zvec.create_and_open(
            path=str(collection_path),
            schema=collection_schema,
            option=collection_option,
        )

        assert created_coll is not None, "Failed to create collection"

        # Close the collection so we can test concurrent opening
        if hasattr(created_coll, "close") and created_coll is not None:
            created_coll.close()

        # Shared variables to collect results from threads
        results = []
        errors = []

        # Lock for thread-safe operations
        lock = threading.Lock()
        # Clean up the created collection reference
        del created_coll

        # Function to be executed by each thread
        def open_collection_thread(thread_id):
            try:
                coll = zvec.open(path=str(collection_path), option=collection_option)
                with lock:
                    results.append((thread_id, coll))
                # Close the collection if opened successfully
                if hasattr(coll, "close") and coll is not None:
                    coll.close()
            except Exception as e:
                with lock:
                    errors.append((thread_id, str(e)))

        # Create 5 threads to call open concurrently
        threads = []
        for i in range(5):
            thread = threading.Thread(target=open_collection_thread, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify concurrency safety: only one should succeed, others should fail
        assert len(results) == 1, (
            f"Expected exactly one successful open, but got {len(results)}"
        )
        assert len(errors) == 4, (
            f"Expected exactly four failures, but got {len(errors)}"
        )

        # Additional verification: check that the successful open has a valid collection
        successful_thread_id, successful_collection = results[0]
        assert successful_collection is not None, (
            "Successful open should return a valid collection"
        )
        assert successful_collection.path == str(collection_path), (
            "Collection path mismatch"
        )

        # Clean up the successfully opened collection
        if (
            hasattr(successful_collection, "destroy")
            and successful_collection is not None
        ):
            try:
                successful_collection.destroy()
            except Exception as e:
                print(f"Warning: failed to destroy collection: {e}")

    def test_open_with_corrupted_files(self, tmp_path_factory):
        # First create a collection
        collection_schema = zvec.CollectionSchema(
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
            ],
            vectors=[
                VectorSchema(
                    "dense",
                    DataType.VECTOR_FP32,
                    dimension=128,
                    index_param=HnswIndexParam(),
                )
            ],
        )

        collection_option = CollectionOption(read_only=False, enable_mmap=True)

        # Create collection path
        temp_dir = tmp_path_factory.mktemp("zvec")
        collection_path = temp_dir / "test_collection"

        # First create the collection
        created_coll = zvec.create_and_open(
            path=str(collection_path),
            schema=collection_schema,
            option=collection_option,
        )

        assert created_coll is not None, "Failed to create collection"

        # Close the collection so we can manipulate its files
        if hasattr(created_coll, "close") and created_coll is not None:
            created_coll.close()

        # Test case 1: Delete some files in the collection directory (simulate partial corruption)
        import os
        import shutil
        import random

        # Get the collection directory path
        collection_dir = str(collection_path)

        # List all files in the collection directory
        files_in_dir = []
        for root, dirs, files in os.walk(collection_dir):
            for file in files:
                files_in_dir.append(os.path.join(root, file))

        # Randomly delete approximately half of the files to simulate partial corruption
        if files_in_dir:
            # Shuffle the list to randomly select files
            random.shuffle(files_in_dir)
            files_to_delete = files_in_dir[: len(files_in_dir) // 2]
            for file_path in files_to_delete:
                try:
                    os.remove(file_path)
                except Exception as e:
                    pass  # Ignore errors during deletion

        # Try to open the collection with missing files - should raise an exception
        with pytest.raises(Exception):
            zvec.open(path=str(collection_path), option=collection_option)

        # Test case 2: Delete all files in the collection directory (simulate complete corruption)
        # Recreate the collection
        recreated_coll = zvec.create_and_open(
            path=str(collection_path) + "_all",
            schema=collection_schema,
            option=collection_option,
        )

        assert recreated_coll is not None, "Failed to recreate collection"

        # Close the collection so we can manipulate its files
        if hasattr(recreated_coll, "close") and recreated_coll is not None:
            recreated_coll.close()

        # Delete all files in the collection directory
        try:
            shutil.rmtree(collection_dir)
            os.makedirs(collection_dir)  # Recreate empty directory
        except Exception as e:
            pass  # Ignore errors during deletion

        # Try to open the collection with missing files - should raise an exception
        with pytest.raises(Exception):
            zvec.open(path=str(collection_path), option=collection_option)
