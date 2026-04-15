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
import os

from distance_helper import *
from fixture_helper import *
from doc_helper import *
from params_helper import *


def check_collection_info(
    coll: Collection, schema: CollectionSchema, option: CollectionOption, path: str
):
    assert coll is not None, "Failed to create and open collection"
    assert coll.path == path
    assert coll.schema.name == schema.name
    assert list(coll.schema.fields) == list(schema.fields)
    assert list(coll.schema.vectors) == list(schema.vectors)
    assert coll.option.read_only == option.read_only
    assert coll.option.enable_mmap == option.enable_mmap


def check_collection_basic(coll: Collection, optimize: bool = False):
    schema = coll.schema

    docs = [generate_doc(i, schema) for i in range(10)]

    results = coll.insert(docs=docs)
    assert len(results) == len(docs)
    for result in results:
        assert result.ok()

    assert coll.stats.doc_count == len(docs)

    def check_fetch_query():
        results = coll.fetch([str(i) for i in range(len(docs))])
        assert len(results) == len(docs)
        for i in range(len(docs)):
            assert str(i) in results

        results = coll.query()
        assert len(results) == len(docs)

    check_fetch_query()

    if optimize:
        coll.optimize()
        check_fetch_query()


def check_collection_full(coll: Collection):
    test_doc = generate_doc(1, coll.schema)

    insert_result = coll.insert(test_doc)
    assert insert_result.ok()

    stats = coll.stats
    assert stats.doc_count == 1

    fetched_docs = coll.fetch(ids=["1"])
    assert len(fetched_docs) == 1
    assert "1" in fetched_docs
    assert fetched_docs["1"] is not None
    assert is_doc_equal(fetched_docs["1"], test_doc, coll.schema)

    query_result = coll.query()
    assert len(query_result) == 1

    updated_doc = Doc(
        id="1",
        fields={"int32_field": 1},
        vectors={"vector_fp32_field": [0.2] * 128},
    )
    update_result = coll.update(updated_doc)
    assert update_result.ok()

    upserted_doc = generate_doc(1, coll.schema)
    upsert_result = coll.upsert(upserted_doc)
    assert upsert_result.ok()

    # 8. Delete document
    delete_result = coll.delete("1")
    assert delete_result.ok()

    # Verify document was deleted
    stats = coll.stats
    assert stats.doc_count == 0


valid_collection_options = [
    # (read_only, enable_mmap)
    (False, True),
    (False, False),
]
invalid_collection_options = [
    # (read_only, enable_mmap)
    (True, True),
    (True, False),
]
duplicate_names_test = [
    ("field1", "field1", "vector1", "vector2"),
    ("field1", "field2", "vector1", "vector1"),
    (
        "shared_name1",
        "shared_name2",
        "shared_name1",
        "shared_name2",
    ),
]
long_names = [
    "a" * 100,  # 100 characters
    "b" * 200,  # 200 characters
]

valid_path_list = [
    "/tmp/nonexistent/directory/test_collection",
    "test/collection/with/slashes",
    "test/collection/with/slashes/哈哈",
]
invalid_path_list = [
    "invalid\0path",
    "",
]


class TestCreateAndOpen:
    @pytest.mark.parametrize("collection_name", COLLECTION_NAME_VALID_LIST)
    def test_valid_collection_name(
        self,
        collection_temp_dir,
        collection_name,
        collection_option,
        sample_field_list,
        sample_vector_list,
    ):
        collection_schema = zvec.CollectionSchema(
            name=collection_name,
            fields=sample_field_list,
            vectors=sample_vector_list,
        )

        coll = zvec.create_and_open(
            path=collection_temp_dir,
            schema=collection_schema,
            option=collection_option,
        )

        check_collection_info(
            coll, collection_schema, collection_option, collection_temp_dir
        )
        check_collection_basic(coll)

        coll.destroy()

    @pytest.mark.parametrize("collection_name", COLLECTION_NAME_INVALID_LIST)
    def test_invalid_collection_name(
        self,
        collection_temp_dir,
        collection_name,
        collection_option,
        sample_field_list,
        sample_vector_list,
    ):
        with pytest.raises(Exception) as exc_info:
            collection_schema = zvec.CollectionSchema(
                name=collection_name,
                fields=sample_field_list,
                vectors=sample_vector_list,
            )

            coll = zvec.create_and_open(
                path=collection_temp_dir,
                schema=collection_schema,
                option=collection_option,
            )

        assert SCHEMA_VALIDATE_ERROR_MSG in str(exc_info.value), str(exc_info.value)

    @pytest.mark.parametrize("name_prefix", FIELD_NAME_VALID_LIST)
    def test_valid_field_vector_name(
        self,
        collection_temp_dir,
        collection_option,
        name_prefix,
        sample_field_list,
        sample_vector_list,
    ):
        collection_schema = zvec.CollectionSchema(
            name="test_collection",
            fields=sample_field_list,
            vectors=sample_vector_list,
        )

        coll = zvec.create_and_open(
            path=collection_temp_dir,
            schema=collection_schema,
            option=collection_option,
        )

        check_collection_info(
            coll, collection_schema, collection_option, collection_temp_dir
        )
        check_collection_basic(coll)

        coll.destroy()

    @pytest.mark.parametrize("field_name", FIELD_NAME_INVALID_LIST)
    def test_invalid_field_name(
        self, collection_temp_dir, collection_option, field_name
    ):
        with pytest.raises(Exception) as exc_info:
            field_list = [FieldSchema(field_name, DataType.STRING)]
            vector_list = [
                VectorSchema(
                    "dense",
                    DataType.VECTOR_FP32,
                    dimension=128,
                    index_param=HnswIndexParam(),
                )
            ]

            collection_schema = zvec.CollectionSchema(
                name="collection_name", fields=field_list, vectors=vector_list
            )

            coll = zvec.create_and_open(
                path=collection_temp_dir,
                schema=collection_schema,
                option=collection_option,
            )

        assert SCHEMA_VALIDATE_ERROR_MSG in str(exc_info.value), str(exc_info.value)

    @pytest.mark.parametrize("vector_name", FIELD_NAME_INVALID_LIST)
    def test_invalid_vector_name(
        self, collection_temp_dir, collection_option, vector_name
    ):
        with pytest.raises(Exception) as exc_info:
            field_list = [
                FieldSchema(
                    "id",
                    DataType.INT64,
                    nullable=False,
                    index_param=InvertIndexParam(enable_range_optimization=True),
                )
            ]
            vector_list = [
                VectorSchema(vector_name, DataType.VECTOR_FP32, dimension=128)
            ]

            collection_schema = zvec.CollectionSchema(
                name="collection_name", fields=field_list, vectors=vector_list
            )

            coll = zvec.create_and_open(
                path=collection_temp_dir,
                schema=collection_schema,
                option=collection_option,
            )

        assert SCHEMA_VALIDATE_ERROR_MSG in str(exc_info.value), str(exc_info.value)

    @pytest.mark.parametrize(
        "field_list_len,vector_list_len,dimension",
        FIELD_VECTOR_LIST_DIMENSION_VALID_LIST,
    )
    def test_valid_field_vector_size_dimension(
        self,
        collection_temp_dir,
        collection_option,
        field_list_len,
        vector_list_len,
        dimension,
    ):
        field_list = []
        vector_list = []
        for i in range(0, field_list_len):
            field_list.append(
                FieldSchema("id_" + str(i), DataType.INT64, nullable=True)
            )

        for i in range(0, vector_list_len):
            vector_list.append(
                VectorSchema(
                    "dense_vector_" + str(i),
                    DataType.VECTOR_FP32,
                    dimension=dimension,
                    index_param=HnswIndexParam(),
                )
            )

        collection_schema = zvec.CollectionSchema(
            name="test_dense_vector_list", fields=field_list, vectors=vector_list
        )

        coll = zvec.create_and_open(
            path=collection_temp_dir,
            schema=collection_schema,
            option=collection_option,
        )

        check_collection_info(
            coll, collection_schema, collection_option, collection_temp_dir
        )
        check_collection_basic(coll)

        coll.destroy()

    @pytest.mark.parametrize(
        "field_list_len,vector_list_len,dimension",
        FIELD_VECTOR_LIST_DIMENSION_INVALID_LIST,
    )
    def test_invalid_field_vector_size_dimension(
        self,
        collection_temp_dir,
        collection_option,
        vector_list_len,
        field_list_len,
        dimension,
    ):
        with pytest.raises(Exception) as exc_info:
            field_list = []
            vector_list = []
            for i in range(0, field_list_len):
                field_list.append(
                    FieldSchema(
                        "id_" + str(i),
                        DataType.INT64,
                        nullable=False,
                    )
                )

            for i in range(0, vector_list_len):
                vector_list.append(
                    VectorSchema(
                        "dense_vector_" + str(i),
                        DataType.VECTOR_FP32,
                        dimension=dimension,
                        index_param=HnswIndexParam(),
                    )
                )

            collection_schema = zvec.CollectionSchema(
                name="test_dense_vector_list", fields=field_list, vectors=vector_list
            )

            coll = zvec.create_and_open(
                path=collection_temp_dir,
                schema=collection_schema,
                option=collection_option,
            )

        assert SCHEMA_VALIDATE_ERROR_MSG in str(exc_info.value), str(exc_info.value)

    def test_valid_single_vector_field_construction(
        self, collection_temp_dir, collection_option
    ):
        field = FieldSchema(
            "id",
            DataType.INT64,
            nullable=True,
            index_param=InvertIndexParam(enable_range_optimization=True),
        )

        vector = VectorSchema(
            "dense_vector",
            DataType.VECTOR_FP32,
            dimension=128,
            index_param=HnswIndexParam(),
        )

        collection_schema = zvec.CollectionSchema(
            name="test_single_dense_vector_non_list",
            fields=field,
            vectors=vector,  # Non-list form
        )

        coll = zvec.create_and_open(
            path=collection_temp_dir,
            schema=collection_schema,
            option=collection_option,
        )

        check_collection_info(
            coll, collection_schema, collection_option, collection_temp_dir
        )
        check_collection_basic(coll)
        coll.destroy()

    def test_collection_concurrent_create(
        self, collection_temp_dir, basic_schema, collection_option
    ):
        results = []
        errors = []
        lock = threading.Lock()

        # Function to be executed by each thread
        def create_collection_thread(thread_id):
            try:
                coll = zvec.create_and_open(
                    path=collection_temp_dir,
                    schema=basic_schema,
                    option=collection_option,
                )
                with lock:
                    results.append((thread_id, coll))
            except Exception as e:
                with lock:
                    errors.append((thread_id, str(e)))

        threads = []
        for i in range(5):
            thread = threading.Thread(target=create_collection_thread, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()
        assert len(results) == 1, (
            f"Expected exactly one successful creation, but got {len(results)}"
        )
        assert len(errors) == 4, (
            f"Expected exactly four failures, but got {len(errors)}"
        )

        successful_thread_id, successful_collection = results[0]
        assert successful_collection is not None, (
            "Successful creation should return a valid collection"
        )
        assert successful_collection.path == collection_temp_dir, (
            "Collection path mismatch"
        )

    def test_create_open_loop(
        self, collection_temp_dir, collection_option, full_schema
    ):
        for cycle in range(10):
            coll = zvec.create_and_open(
                path=collection_temp_dir,
                schema=full_schema,
                option=collection_option,
            )
            assert coll is not None, (
                f"Failed to create and open collection in cycle {cycle}"
            )
            assert coll.path == collection_temp_dir, (
                f"Collection path mismatch in cycle {cycle}"
            )

            del coll

            reopened_coll = zvec.open(
                path=collection_temp_dir, option=collection_option
            )
            assert reopened_coll is not None, (
                f"Failed to reopen collection in cycle {cycle}"
            )
            assert reopened_coll.path == collection_temp_dir, (
                f"Reopened collection path mismatch in cycle {cycle}"
            )

            check_collection_full(reopened_coll)

            reopened_coll.destroy()

    @pytest.mark.parametrize(
        "data_type, index_param", VALID_VECTOR_DATA_TYPE_INDEX_PARAM_MAP_PARAMS
    )
    def test_valid_vector_index_params(
        self,
        data_type,
        index_param,
        single_vector_schema_with_index_param,
        collection_temp_dir,
        collection_option,
    ):
        coll = zvec.create_and_open(
            path=collection_temp_dir,
            schema=single_vector_schema_with_index_param,
            option=collection_option,
        )

        check_collection_info(
            coll,
            single_vector_schema_with_index_param,
            collection_option,
            collection_temp_dir,
        )

        check_collection_basic(coll, True)

    @pytest.mark.parametrize(
        "data_type, index_param", INVALID_VECTOR_DATA_TYPE_INDEX_PARAM_MAP_PARAMS
    )
    def test_invalid_vector_index_params(
        self,
        data_type,
        index_param,
        single_vector_schema_with_index_param,
        collection_temp_dir,
        collection_option,
    ):
        with pytest.raises(Exception) as exc_info:
            coll = zvec.create_and_open(
                path=collection_temp_dir,
                schema=single_vector_schema_with_index_param,
                option=collection_option,
            )

        assert SCHEMA_VALIDATE_ERROR_MSG in str(exc_info.value), str(exc_info.value)

    def test_open_concurrent_same_path(self, tmp_path_factory, collection_option):
        """Test concurrent opening of the same collection path.

        - Multi-threading concurrency: 5 threads simultaneously open the same collection
        - Result verification: Verify that only one can open successfully, others must fail
        """
        # Create a temporary directory and path for the collection
        temp_dir = tmp_path_factory.mktemp("zvec")
        collection_path = temp_dir / "concurrent_open_test_collection"

        # First, create a collection that we'll try to open concurrently
        field_list = [
            FieldSchema(
                "id",
                DataType.INT64,
                nullable=False,
                index_param=InvertIndexParam(enable_range_optimization=True),
            ),
            FieldSchema(
                "name", DataType.STRING, nullable=False, index_param=InvertIndexParam()
            ),
        ]

        vector_list = [
            VectorSchema(
                "dense_vector",
                DataType.VECTOR_FP32,
                dimension=128,
                index_param=HnswIndexParam(),
            )
        ]

        collection_schema = zvec.CollectionSchema(
            name="concurrent_open_test_collection",
            fields=field_list,
            vectors=vector_list,
        )

        # Create the collection first
        coll = zvec.create_and_open(
            path=str(collection_path),
            schema=collection_schema,
            option=collection_option,
        )

        # Close the collection so we can test opening it
        if hasattr(coll, "close") and coll is not None:
            coll.close()

        # Shared variables to collect results from threads
        results = []
        errors = []

        # Lock for thread-safe operations
        lock = threading.Lock()
        # Clean up the created collection reference
        del coll

        # Function to be executed by each thread
        def open_collection_thread(thread_id):
            try:
                reopened_coll = zvec.open(
                    path=str(collection_path), option=collection_option
                )
                with lock:
                    results.append((thread_id, reopened_coll))
                # Clean up the collection if opened successfully
                if hasattr(reopened_coll, "close") and reopened_coll is not None:
                    reopened_coll.close()
            except Exception as e:
                with lock:
                    errors.append((thread_id, str(e)))

        # Create and start 5 threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=open_collection_thread, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify results:
        # 1. Only one open should succeed (exactly one collection in results)
        # 2. Others should fail (4 errors in errors)
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

    @pytest.mark.parametrize("read_only,enable_mmap", valid_collection_options)
    def test_valid_option(
        self, collection_temp_dir, basic_schema, read_only, enable_mmap
    ):
        option = CollectionOption(read_only=read_only, enable_mmap=enable_mmap)

        coll = zvec.create_and_open(
            path=collection_temp_dir,
            schema=basic_schema,
            option=option,
        )

        check_collection_info(coll, basic_schema, option, collection_temp_dir)
        check_collection_basic(coll)

        coll.destroy()

    def test_valid_none_option(self, collection_temp_dir, basic_schema):
        zvec.create_and_open(
            path=collection_temp_dir,
            schema=basic_schema,
            option=None,
        )

    @pytest.mark.parametrize("read_only,enable_mmap", invalid_collection_options)
    def test_invalid_option(
        self, collection_temp_dir, basic_schema, read_only, enable_mmap
    ):
        with pytest.raises(Exception) as exc_info:
            coll = zvec.create_and_open(
                path=collection_temp_dir,
                schema=basic_schema,
                option=CollectionOption(read_only=read_only, enable_mmap=enable_mmap),
            )

        assert CREATE_READ_ONLY_ERROR_MSG in str(exc_info.value), str(exc_info.value)

    @pytest.mark.parametrize(
        "field_name1,field_name2,vector_name1,vector_name2",
        duplicate_names_test,
    )
    def test_duplicate_field_names(
        self,
        collection_temp_dir,
        collection_option,
        field_name1,
        field_name2,
        vector_name1,
        vector_name2,
    ):
        with pytest.raises(Exception) as exc_info:
            collection_schema = zvec.CollectionSchema(
                name="test_collection",
                fields=[
                    FieldSchema(
                        field_name1,
                        DataType.INT64,
                        nullable=False,
                        index_param=InvertIndexParam(enable_range_optimization=True),
                    ),
                    FieldSchema(
                        field_name2,
                        DataType.INT64,
                        nullable=False,
                        index_param=InvertIndexParam(enable_range_optimization=True),
                    ),
                ],
                vectors=[
                    VectorSchema(
                        vector_name1,
                        DataType.VECTOR_FP32,
                        dimension=128,
                        index_param=HnswIndexParam(),
                    ),
                    VectorSchema(
                        vector_name2,
                        DataType.VECTOR_FP32,
                        dimension=128,
                        index_param=HnswIndexParam(),
                    ),
                ],
            )

            coll = zvec.create_and_open(
                path=collection_temp_dir,
                schema=collection_schema,
                option=collection_option,
            )

        assert SCHEMA_VALIDATE_ERROR_MSG in str(exc_info.value), str(exc_info.value)

    @pytest.mark.parametrize("long_name", long_names)
    def test_invalid_long_field_names(
        self, collection_option, collection_temp_dir, long_name
    ):
        collection_schema = zvec.CollectionSchema(
            name=long_name,
            fields=[
                FieldSchema(
                    long_name + "_field",
                    DataType.INT64,
                    nullable=False,
                    index_param=InvertIndexParam(enable_range_optimization=True),
                ),
            ],
            vectors=[
                VectorSchema(
                    long_name + "_vector",
                    DataType.VECTOR_FP32,
                    dimension=128,
                    index_param=HnswIndexParam(),
                )
            ],
        )

        with pytest.raises(Exception) as exc_info:
            coll = zvec.create_and_open(
                path=collection_temp_dir,
                schema=collection_schema,
                option=collection_option,
            )

        assert SCHEMA_VALIDATE_ERROR_MSG in str(exc_info.value), str(exc_info.value)

    def test_invalid_empty_fields_and_vectors(
        self, collection_temp_dir, collection_option
    ):
        collection_schema = zvec.CollectionSchema(
            name="test_collection",
            fields=[],  # Empty fields
            vectors=[],  # Empty vectors
        )

        with pytest.raises(Exception) as exc_info:
            coll = zvec.create_and_open(
                path=collection_temp_dir,
                schema=collection_schema,
                option=collection_option,
            )

        assert SCHEMA_VALIDATE_ERROR_MSG in str(exc_info.value), str(exc_info.value)

    @pytest.mark.parametrize("valid_path", valid_path_list)
    def test_valid_path(self, basic_schema, collection_option, valid_path):
        if os.path.exists(valid_path):
            import shutil

            shutil.rmtree(valid_path)

        coll = zvec.create_and_open(
            path=valid_path, schema=basic_schema, option=collection_option
        )

        check_collection_info(coll, basic_schema, collection_option, valid_path)

        coll.destroy()

    @pytest.mark.parametrize("invalid_path", invalid_path_list)
    def test_invalid_path(self, basic_schema, collection_option, invalid_path):
        with pytest.raises(Exception) as exc_info:
            coll = zvec.create_and_open(
                path=invalid_path, schema=basic_schema, option=collection_option
            )

        assert INVALID_PATH_ERROR_MSG in str(exc_info.value), str(exc_info.value)
