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
    def test_close_releases_collection_for_reopen(
        self, tmp_path_factory, collection_schema, collection_option
    ):
        temp_dir = tmp_path_factory.mktemp("zvec")
        collection_path = temp_dir / "test_collection"
        collection_path_str = str(collection_path)

        created_coll = zvec.create_and_open(
            path=collection_path_str, schema=collection_schema, option=collection_option
        )

        created_coll.close()

        opened_coll = zvec.open(path=collection_path_str, option=collection_option)
        assert opened_coll.path == collection_path_str
        opened_coll.destroy()

    def test_close_releases_collection_with_outstanding_reference(
        self, tmp_path_factory, collection_schema, collection_option
    ):
        temp_dir = tmp_path_factory.mktemp("zvec")
        collection_path = temp_dir / "test_collection"
        collection_path_str = str(collection_path)

        created_coll = zvec.create_and_open(
            path=collection_path_str, schema=collection_schema, option=collection_option
        )

        # Keep an extra reference to the native handle, as a shallow copy or a
        # stale reference would. close() must release the file lock anyway.
        native_ref = created_coll._obj

        created_coll.close()

        # The path can be reopened even though native_ref is still alive.
        opened_coll = zvec.open(path=collection_path_str, option=collection_option)
        assert opened_coll.path == collection_path_str

        # Operations through the stale handle fail cleanly instead of touching
        # released native state.
        with pytest.raises(ValueError, match="already closed"):
            native_ref.Stats()

        opened_coll.destroy()

    def test_close_is_idempotent(
        self, tmp_path_factory, collection_schema, collection_option
    ):
        temp_dir = tmp_path_factory.mktemp("zvec")
        collection_path = temp_dir / "test_collection"
        collection_path_str = str(collection_path)

        created_coll = zvec.create_and_open(
            path=collection_path_str, schema=collection_schema, option=collection_option
        )

        native_ref = created_coll._obj
        created_coll.close()
        created_coll.close()
        # Closing the native handle again is also a no-op.
        native_ref.Close()

        opened_coll = zvec.open(path=collection_path_str, option=collection_option)
        assert opened_coll.path == collection_path_str
        opened_coll.destroy()

    def test_open_basic_functionality(
        self, tmp_path_factory, collection_schema, collection_option
    ):
        collection_path = str(tmp_path_factory.mktemp("zvec") / "test_collection")
        docs = {
            str(i): Doc(
                id=str(i),
                fields={"id": i, "name": f"test_{i}", "weight": float(i * 10)},
                vectors={
                    "dense": [float(j + i) for j in range(128)],
                    "sparse": {j: float(j + i) for j in range(5)},
                },
            )
            for i in range(3)
        }

        def check_docs(results, expected_ids):
            results = list(results)
            assert len(results) == len(expected_ids)
            assert {doc.id for doc in results} == expected_ids
            for doc in results:
                expected = docs[doc.id]
                assert doc.fields == expected.fields
                assert set(doc.vector_names()) == {"dense", "sparse"}
                assert isinstance(doc.vector("dense"), list)
                assert isinstance(doc.vector("sparse"), dict)
                assert doc.vectors == expected.vectors
                assert isinstance(doc.score, (int, float))
                assert np.isfinite(doc.score)

        created_coll = zvec.create_and_open(
            path=collection_path, schema=collection_schema, option=collection_option
        )
        assert created_coll.path == collection_path
        assert created_coll.schema.name == collection_schema.name
        results = created_coll.insert(list(docs.values()))
        assert len(results) == len(docs)
        assert all(result.ok() for result in results)
        assert created_coll.stats.doc_count == len(docs)
        check_docs(created_coll.fetch(list(docs)).values(), set(docs))

        # Releasing the last reference must persist data and release the file lock.
        del created_coll
        opened_coll = zvec.open(path=collection_path, option=collection_option)
        try:
            assert opened_coll.path == collection_path
            assert opened_coll.schema.name == collection_schema.name
            assert opened_coll.stats.doc_count == len(docs)
            fetched = opened_coll.fetch(list(docs))
            check_docs(fetched.values(), set(docs))
            assert all(doc.score == 0.0 for doc in fetched.values())
            check_docs(opened_coll.query(include_vector=True), set(docs))
            check_docs(
                opened_coll.query(filter="id >= 1", include_vector=True), {"1", "2"}
            )
            for field_name, vector in (
                ("dense", [0.1] * 128),
                ("sparse", {1: 1.0, 2: 2.0, 3: 3.0}),
            ):
                check_docs(
                    opened_coll.query(
                        Query(field_name=field_name, vector=vector),
                        topk=len(docs),
                        include_vector=True,
                    ),
                    set(docs),
                )
        finally:
            opened_coll.destroy()

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
        # Successful opens must stay alive until every thread has tried:
        # closing inside the thread would release the LOCK file early and let
        # a later-starting thread legitimately acquire it.
        opened = []

        # Lock for thread-safe operations
        lock = threading.Lock()
        # Clean up the created collection reference
        del created_coll

        # Function to be executed by each thread
        def open_collection_thread(thread_id):
            try:
                coll = zvec.open(path=str(collection_path), option=collection_option)
                collection_path_result = coll.path
                with lock:
                    results.append((thread_id, collection_path_result))
                    opened.append(coll)
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

        # Release the lock held by the successful open before further checks.
        for coll in opened:
            coll.close()

        # Additional verification: check that the successful open has a valid collection
        _, successful_collection_path = results[0]
        assert successful_collection_path is not None, (
            "Successful open should return a valid collection"
        )
        assert successful_collection_path == str(collection_path), (
            "Collection path mismatch"
        )

        # Clean up after the successful worker released its collection handle.
        cleanup_coll = zvec.open(path=str(collection_path), option=collection_option)
        cleanup_coll.destroy()

    @pytest.mark.parametrize("read_only", [False, True])
    def test_open_reports_idmap_recovery_error(
        self, tmp_path, collection_schema, read_only
    ):
        zvec.init()
        path = tmp_path / "collection"
        created = zvec.create_and_open(
            path=str(path),
            schema=collection_schema,
            option=CollectionOption(enable_mmap=False),
        )
        created.close()
        # Keep collection metadata valid and fail specifically in IDMap's
        # RocksDB manifest recovery, after the collection lock is acquired.
        current = path / "idmap.0" / "CURRENT"
        # RocksDB requires an LF terminator, including on Windows.
        original = current.read_bytes()
        current.write_bytes(b"MANIFEST-999999\n")

        with pytest.raises(RuntimeError, match="recovery idmap failed") as error:
            zvec.open(
                path=str(path),
                option=CollectionOption(read_only=read_only, enable_mmap=False),
            )
        assert "Failed to list RocksDB column families" in str(error.value)
        assert "MANIFEST-999999" in str(error.value)

        current.write_bytes(original)
        reopened = zvec.open(
            path=str(path),
            option=CollectionOption(read_only=read_only, enable_mmap=False),
        )
        reopened.close()

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

        # Test case 1: Remove required metadata while retaining the other files.
        import shutil

        # A random subset may contain only recoverable RocksDB auxiliary files.
        # Removing every collection manifest makes partial corruption definite.
        manifest_files = list(collection_path.glob("manifest.*"))
        assert manifest_files, "Collection manifest files were not created"
        for manifest in manifest_files:
            manifest.unlink()

        # Try to open the collection with missing files - should raise an exception
        with pytest.raises(KeyError, match="Failed to find the version file"):
            zvec.open(path=str(collection_path), option=collection_option)

        # Test case 2: Delete all files in the collection directory (simulate complete corruption)
        # Recreate the collection
        all_collection_path = temp_dir / "test_collection_all"
        recreated_coll = zvec.create_and_open(
            path=str(all_collection_path),
            schema=collection_schema,
            option=collection_option,
        )

        assert recreated_coll is not None, "Failed to recreate collection"

        # Close the collection so we can manipulate its files
        if hasattr(recreated_coll, "close") and recreated_coll is not None:
            recreated_coll.close()

        # Delete all files in the collection directory
        shutil.rmtree(all_collection_path)
        all_collection_path.mkdir()

        # Try to open the collection with missing files - should raise an exception
        with pytest.raises(RuntimeError):
            zvec.open(path=str(all_collection_path), option=collection_option)
