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

from distance_helper import *
from fixture_helper import *
from doc_helper import *
from params_helper import *


class TestDDL:
    def test_collection_stats(self, basic_collection: Collection):
        assert basic_collection.stats is not None
        stats = basic_collection.stats
        assert stats.doc_count == 0
        assert len(stats.index_completeness) == 2
        assert stats.index_completeness["dense"] == 1
        assert stats.index_completeness["sparse"] == 1

    def test_collection_destroy(
        self, basic_collection: Collection, collection_temp_dir, collection_option
    ):
        doc = generate_doc(1, basic_collection.schema)

        result = basic_collection.insert(doc)
        assert bool(result)
        assert result.ok()

        stats = basic_collection.stats
        assert stats is not None
        assert stats.doc_count == 1

        basic_collection.destroy()

        with pytest.raises(Exception) as exc_info:
            stats = basic_collection.stats
        assert ACCESS_DESTROYED_COLLECTION_ERROR_MSG in str(exc_info.value)

        with pytest.raises(Exception) as exc_info:
            zvec.open(path=collection_temp_dir, option=collection_option)
        assert COLLECTION_PATH_NOT_EXIST_ERROR_MSG in str(exc_info.value)

    def test_collection_flush(self, basic_collection: Collection):
        doc = generate_doc(1, basic_collection.schema)

        result = basic_collection.insert(doc)
        assert bool(result)
        assert result.ok()

        basic_collection.flush()

        fetched_docs = basic_collection.fetch(["1"])
        assert "1" in fetched_docs
        assert fetched_docs["1"].id == "1"


class TestIndexDDL:
    @pytest.mark.parametrize("field_name", DEFAULT_SCALAR_FIELD_NAME.values())
    @pytest.mark.parametrize("index_type", SUPPORT_SCALAR_INDEX_TYPES)
    def test_scalar_index_operation(
        self,
        full_collection: Collection,
        field_name: str,
        index_type: IndexType,
    ):
        # INSERT 0~5 Doc
        docs = [generate_doc(i, full_collection.schema) for i in range(5)]

        result = full_collection.insert(docs)
        assert len(result) == 5
        for item in result:
            assert item.ok()

        stats = full_collection.stats
        assert stats is not None
        assert stats.doc_count == 5

        if field_name in ["bool_field"]:
            query_filter = f"{field_name} = true"
        elif field_name in ["double_field", "float_field"]:
            query_filter = f"{field_name} >= 3.0"
        elif field_name in [
            "int32_field",
            "int64_field",
            "uint32_field",
            "uint64_field",
        ]:
            query_filter = f"{field_name} >= 30"
        elif field_name in ["string_field"]:
            query_filter = f"{field_name} >= 'test_3'"
        elif field_name in ["array_bool_field"]:
            query_filter = f"{field_name} contain_any (false)"
        elif field_name in ["array_double_field", "array_float_field"]:
            query_filter = f"{field_name} contain_any (3.0, 4.0)"
        elif field_name in [
            "array_int64_field",
            "array_int32_field",
            "array_uint64_field",
            "array_uint32_field",
        ]:
            query_filter = f"{field_name} contain_any (3, 4)"
        elif field_name == "array_string_field":
            query_filter = f"{field_name} contain_any ('test_3', 'test_4')"
        else:
            assert False, f"Unsupported field type for index creation: {field_name}"

        query_result_before = full_collection.query(filter=query_filter, topk=10)

        if index_type not in DEFAULT_INDEX_PARAMS:
            pytest.fail(f"Unsupported index type for index creation: {index_type}")
        index_param = DEFAULT_INDEX_PARAMS[index_type]

        full_collection.create_index(
            field_name=field_name, index_param=index_param, option=IndexOption()
        )
        stats_after_create = full_collection.stats
        assert stats_after_create is not None
        assert stats_after_create.doc_count == 5

        query_result_after = full_collection.query(filter=query_filter, topk=10)

        assert len(query_result_before) == len(query_result_after), (
            f"Query result count mismatch for {field_name} with index type {index_type}: before={len(query_result_before)}, after={len(query_result_after)}"
        )

        before_ids = set(doc.id for doc in query_result_before)
        after_ids = set(doc.id for doc in query_result_after)
        assert before_ids == after_ids, (
            f"Query result IDs mismatch for {field_name} with index type {index_type}: before={before_ids}, after={after_ids}"
        )

        # INSERT 5~8 Doc
        new_docs = [generate_doc(i, full_collection.schema) for i in range(5, 8)]

        result = full_collection.insert(new_docs)
        assert len(result) == 3
        for item in result:
            assert item.ok()

        stats_after_insert1 = full_collection.stats
        assert stats_after_insert1 is not None
        assert stats_after_insert1.doc_count == 8

        fetched_docs = full_collection.fetch([f"{i}" for i in range(5, 8)])
        assert len(fetched_docs) == 3

        for i in range(5, 8):
            doc_id = f"{i}"
            assert doc_id in fetched_docs

        query_result = full_collection.query(filter=query_filter, topk=20)
        assert len(query_result) >= len(query_result_before)

        full_collection.drop_index(field_name=field_name)

        # Insert 8~10 Doc
        more_docs = [generate_doc(i, full_collection.schema) for i in range(8, 10)]

        result = full_collection.insert(more_docs)
        assert len(result) == 2
        for item in result:
            assert item.ok()

        stats_after_insert2 = full_collection.stats
        assert stats_after_insert2 is not None
        assert stats_after_insert2.doc_count == 10

        fetched_docs = full_collection.fetch([f"{i}" for i in range(8, 10)])
        assert len(fetched_docs) == 2

        for i in range(8, 10):
            doc_id = f"{i}"
            assert doc_id in fetched_docs

        query_result = full_collection.query(filter=query_filter, topk=20)
        assert len(query_result) >= len(query_result_before)

        final_stats = full_collection.stats
        assert final_stats is not None
        assert final_stats.doc_count == 10
        full_collection.destroy()

    @pytest.mark.parametrize("field_name", DEFAULT_SCALAR_FIELD_NAME.values())
    @pytest.mark.parametrize("index_type", SUPPORT_SCALAR_INDEX_TYPES)
    def test_duplicate_create_index(
        self, full_collection: Collection, field_name: str, index_type: IndexType
    ):
        docs = [generate_doc(i, full_collection.schema) for i in range(10)]

        result = full_collection.insert(docs)
        assert bool(result)
        for item in result:
            assert item.ok()

        stats = full_collection.stats
        assert stats is not None
        assert stats.doc_count == 10

        if field_name in ["bool_field"]:
            query_filter = f"{field_name} = true"
        elif field_name in ["double_field", "float_field"]:
            query_filter = f"{field_name} >= 3.0"
        elif field_name in [
            "int32_field",
            "int64_field",
            "uint32_field",
            "uint64_field",
        ]:
            query_filter = f"{field_name} >= 30"
        elif field_name in ["string_field"]:
            query_filter = f"{field_name} >= 'test_3'"
        elif field_name in ["array_bool_field"]:
            query_filter = f"{field_name} contain_any (false)"
        elif field_name in ["array_double_field", "array_float_field"]:
            query_filter = f"{field_name} contain_any (3.0, 4.0)"
        elif field_name in [
            "array_int64_field",
            "array_int32_field",
            "array_uint64_field",
            "array_uint32_field",
        ]:
            query_filter = f"{field_name} contain_any (3, 4)"
        elif field_name == "array_string_field":
            query_filter = f"{field_name} contain_any ('test_3', 'test_4')"
        else:
            assert False, f"Unsupported field type for index creation: {field_name}"

        query_result_before = full_collection.query(filter=query_filter, topk=5)

        if index_type not in DEFAULT_INDEX_PARAMS:
            pytest.fail(f"Unsupported index type for index creation: {index_type}")
        index_param = DEFAULT_INDEX_PARAMS[index_type]

        full_collection.create_index(
            field_name=field_name, index_param=index_param, option=IndexOption()
        )

        query_result_after = full_collection.query(filter=query_filter, topk=5)

        assert len(query_result_before) == len(query_result_after), (
            f"Query result count mismatch: before={len(query_result_before)}, after={len(query_result_after)}"
        )

        before_ids = set(doc.id for doc in query_result_before)
        after_ids = set(doc.id for doc in query_result_after)
        assert before_ids == after_ids, (
            f"Query result IDs mismatch: before={before_ids}, after={after_ids}"
        )

        full_collection.create_index(
            field_name=field_name, index_param=index_param, option=IndexOption()
        )

    def test_optimize(self, full_collection: Collection):
        docs = [generate_doc(i, full_collection.schema) for i in range(10)]

        result = full_collection.insert(docs)
        assert bool(result)
        for item in result:
            assert item.ok()

        stats = full_collection.stats
        assert stats is not None
        assert stats.doc_count == 10

        full_collection.optimize(option=OptimizeOption())

        fetched_docs = full_collection.fetch(["1"])
        assert "1" in fetched_docs
        assert fetched_docs["1"].id == "1"

    @pytest.mark.parametrize(
        "vector_type, index_type", SUPPORT_VECTOR_DATA_TYPE_INDEX_MAP_PARAMS
    )
    def test_vector_index_operation(
        self,
        full_collection: Collection,
        vector_type: DataType,
        index_type: IndexType,
    ):
        vector_field_name = DEFAULT_VECTOR_FIELD_NAME[vector_type]

        docs = [generate_doc(i, full_collection.schema) for i in range(5)]

        result = full_collection.insert(docs)
        assert len(result) == 5, (
            f"Expected 5 insertion results, got {len(result)} for vector type {vector_type} and index type {index_type}"
        )
        for i, item in enumerate(result):
            assert item.ok(), (
                f"Before create_index,result={result},Insertion result {i} is not OK for vector type {vector_type} and index type {index_type} and result={result}"
            )

        stats = full_collection.stats
        assert stats is not None, (
            f"stats is None for vector type {vector_type} and index type {index_type}"
        )
        assert stats.doc_count == 5, (
            f"doc_count!=5 for vector type {vector_type} and index type {index_type}"
        )

        if index_type not in DEFAULT_INDEX_PARAMS:
            pytest.fail(
                f"Unsupported index type {index_type} for vector type {vector_type} in test_vector_all_data_types_index_create_drop_validation"
            )
        index_param = DEFAULT_INDEX_PARAMS[index_type]

        full_collection.create_index(
            field_name=vector_field_name,
            index_param=index_param,
            option=IndexOption(),
        )

        stats_after_create = full_collection.stats
        assert stats_after_create is not None, (
            f"stats_after_create_index is None for vector type {vector_type} and index type {index_type}"
        )

        new_docs = [generate_doc(i, full_collection.schema) for i in range(5, 8)]

        result = full_collection.insert(new_docs)
        assert len(result) == 3, (
            f"Expected 3 insertion results, got {len(result)} for vector type {vector_type} and index type {index_type}"
        )
        for i, item in enumerate(result):
            assert item.ok(), (
                f"Before drop_index,result={result},BInsertion result {i} is not OK for vector type {vector_type} and index type {index_type} and "
            )

        stats_after_insert1 = full_collection.stats
        assert stats_after_insert1 is not None, (
            f"stats_after_insert1 is None for vector type {vector_type} and index type {index_type}"
        )
        assert stats_after_insert1.doc_count == 8, (
            f"Expected 8 documents, got {stats_after_insert1.doc_count} for vector type {vector_type} and index type {index_type}"
        )

        fetched_docs = full_collection.fetch([f"{i}" for i in range(5, 8)])
        assert len(fetched_docs) == 3, (
            f"Expected 3 fetched documents, got {len(fetched_docs)} for vector type {vector_type} and index type {index_type}"
        )

        for i in range(5, 8):
            doc_id = f"{i}"
            assert doc_id in fetched_docs, (
                f"Document ID {doc_id} not found in fetched results for vector type {vector_type} and index type {index_type}"
            )
            assert fetched_docs[doc_id].id == doc_id, (
                f"Document {doc_id} has incorrect ID field value for vector type {vector_type} and index type {index_type}"
            )

        full_collection.drop_index(field_name=vector_field_name)

        more_docs = [generate_doc(i, full_collection.schema) for i in range(8, 10)]
        result = full_collection.insert(more_docs)
        assert len(result) == 2, (
            f"Expected 2 insertion results, got {len(result)} for vector type {vector_type} and index type {index_type}"
        )
        for i, item in enumerate(result):
            assert item.ok(), (
                f"After drop_index,Insertion result {i} is not OK for vector type {vector_type} and index type {index_type} and result={result}"
            )

        # Verify document count after second insertion
        stats_after_insert2 = full_collection.stats
        assert stats_after_insert2 is not None, (
            f"stats_after_insert2 is None for vector type {vector_type} and index type {index_type}"
        )
        assert stats_after_insert2.doc_count == 10, (
            f"Expected 10 documents, got {stats_after_insert2.doc_count} for vector type {vector_type} and index type {index_type}"
        )

        # Fetch data
        fetched_docs = full_collection.fetch([f"{i}" for i in range(8, 10)])
        assert len(fetched_docs) == 2, (
            f"Expected 2 fetched documents, got {len(fetched_docs)} for vector type {vector_type} and index type {index_type}"
        )

        # Verify fetched documents have correct data
        for i in range(8, 10):
            doc_id = f"{i}"
            assert doc_id in fetched_docs, (
                f"Document ID {doc_id} not found in fetched results for vector type {vector_type} and index type {index_type}"
            )
            assert fetched_docs[doc_id].id == doc_id, (
                f"Document {doc_id} has incorrect ID field value for vector type {vector_type} and index type {index_type}"
            )

        # Final verification
        final_stats = full_collection.stats
        assert final_stats is not None, (
            f"final_stats is None for vector type {vector_type} and index type {index_type}"
        )
        assert final_stats.doc_count == 10, (
            f"Expected 10 documents, got {final_stats.doc_count} for vector type {vector_type} and index type {index_type}"
        )
        full_collection.destroy()

    @staticmethod
    def create_collection(
        collection_path, collection_option: CollectionOption
    ) -> Collection:
        schema = CollectionSchema(
            name="test_collection_invalid_vector_index",
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
                ),
            ],
        )
        coll = zvec.create_and_open(
            path=collection_path, schema=schema, option=collection_option
        )
        assert coll is not None, "Failed to create and open collection"
        return coll

    @staticmethod
    def check_error_message(exc_info, invalid_name):
        if type(invalid_name) is str:
            assert INDEX_NON_EXISTENT_COLUMN_ERROR_MSG in str(exc_info.value), (
                "Error message is unreasonable: e=" + str(exc_info.value)
            )
        else:
            assert INCOMPATIBLE_FUNCTION_ERROR_MSG in str(exc_info.value), (
                "Error message is unreasonable: e=" + str(exc_info.value)
            )

    @pytest.mark.parametrize(
        "invalid_field_name,invalid_vector_name",
        [
            ("", ""),  # Empty string
            (" ", " "),  # Space only
            ("v" * 33, "v" * 33),  # Too long (33 characters, exceeds 32)
            ("vector name", "vector_name"),  # Contains space
            ("vector@name", "vector@name"),  # Contains special character
            ("vector/name", "vector/name"),  # Contains slash
            ("vector\\name", "vector\\name"),  # Contains backslash
            ("vector.name", "vector.name"),  # Contains dot
            ("vector$data", "vector$data"),  # Contains dollar sign
            ("vector+name", "vector+name"),  # Contains plus sign
            ("vector=name", "vector=name"),  # Contains equals sign
            (None, None),  # None value,
            (1, 1),
            (1.1, 1.1),
        ],
    )
    def test_invalid_field_and_vector_name(
        self,
        collection_temp_dir,
        collection_option: CollectionOption,
        invalid_field_name: Any,
        invalid_vector_name: Any,
    ):
        coll = self.create_collection(collection_temp_dir, collection_option)
        with pytest.raises(Exception) as exc_info:
            coll.create_index(
                field_name=invalid_vector_name,
                index_param=HnswIndexParam(),
                option=IndexOption(),
            )
        self.check_error_message(exc_info, invalid_vector_name)
        with pytest.raises(Exception) as exc_info:
            coll.create_index(
                field_name=invalid_field_name,
                index_param=InvertIndexParam(),
                option=IndexOption(),
            )
        self.check_error_message(exc_info, invalid_field_name)
        coll.destroy()
        coll = self.create_collection(collection_temp_dir, collection_option)
        with pytest.raises(Exception) as exc_info:
            coll.drop_index(field_name=invalid_vector_name)
        self.check_error_message(exc_info, invalid_vector_name)
        with pytest.raises(Exception) as exc_info:
            coll.drop_index(field_name=invalid_field_name)
        self.check_error_message(exc_info, invalid_field_name)
        coll.destroy()

    @pytest.mark.parametrize(
        "field_name,vector_name",
        [
            ("2", "3"),
            ("col", "co1"),
            ("ID", "IM"),
            ("name-1", "name2"),
            ("Weigt_12", "Weigt_13"),
            ("123age", "123agl"),
        ],
    )
    def test_valid_field_and_vector_name(
        self,
        collection_temp_dir,
        collection_option: CollectionOption,
        field_name: str,
        vector_name: str,
    ):
        schema = zvec.CollectionSchema(
            name="test_index_names",
            fields=[
                FieldSchema(
                    "id",
                    DataType.INT64,
                    nullable=False,
                    index_param=InvertIndexParam(enable_range_optimization=True),
                ),
                FieldSchema(field_name, DataType.STRING, nullable=True),
            ],
            vectors=[
                VectorSchema(
                    vector_name,
                    DataType.VECTOR_FP32,
                    dimension=128,
                    index_param=HnswIndexParam(),
                )
            ],
        )

        coll = zvec.create_and_open(
            path=collection_temp_dir, schema=schema, option=collection_option
        )

        assert coll is not None, (
            f"Failed to create and open collection with field_name={field_name}, vector_name={vector_name}"
        )

        # Insert some data
        docs = [
            Doc(
                id=f"{i}",
                fields={"id": i, field_name: f"value_{i}"},
                vectors={vector_name: [float(j % 10) for j in range(128)]},
            )
            for i in range(5)
        ]

        result = coll.insert(docs)
        assert len(result) == 5, (
            f"Expected 5 insertion results, got {len(result)} for field_name={field_name}, vector_name={vector_name}"
        )
        for item in result:
            assert item.ok(), (
                f"Insertion failed for field_name={field_name}, vector_name={vector_name}: {item}"
            )

        # Create index on field
        coll.create_index(
            field_name=field_name,
            index_param=InvertIndexParam(),
            option=IndexOption(),
        )

        # Create index on vector
        coll.create_index(
            field_name=vector_name,
            index_param=HnswIndexParam(),
            option=IndexOption(),
        )

        # Verify indexes were created successfully
        stats = coll.stats
        assert stats is not None, (
            f"Stats is None for field_name={field_name}, vector_name={vector_name}"
        )

        coll.destroy()

    def test_compicated_workflow(
        self,
        collection_temp_dir,
        basic_schema: CollectionSchema,
        collection_option: CollectionOption,
    ):
        """
        Test the complete workflow:
        1. Create collection
        2. Create index
        3. Insert doc
        4. Upsert
        5. Update doc
        6. Fetch doc
        7. Query doc
        8. Drop index
        9. Insert doc
        10. Update doc
        11. Upsert doc
        12. Fetch doc
        13. Query doc
        14. Flush
        15. Destroy
        """
        # Step 1: Create collection
        coll = zvec.create_and_open(
            path=collection_temp_dir,
            schema=basic_schema,
            option=collection_option,
        )

        assert coll is not None, "Failed to create and open collection"
        assert coll.path == collection_temp_dir
        assert coll.schema.name == basic_schema.name
        assert coll.stats.doc_count == 0

        # Step 2: Create index
        coll.create_index(
            field_name="name", index_param=InvertIndexParam(), option=IndexOption()
        )
        # Verify index was created
        stats = coll.stats
        assert stats is not None, "coll.stats is None!"

        # Step 3: Insert doc
        doc1 = Doc(
            id="1",
            fields={"id": 1, "name": "test1", "weight": 80.5},
            vectors={
                "dense": np.random.random(128).tolist(),
                "sparse": {1: 1.0, 2: 2.0},
            },
        )

        result = coll.insert(doc1)
        assert bool(result)
        assert result.ok()
        assert coll.stats.doc_count == 1

        # Step 4: Upsert (existing doc)
        doc1_updated = Doc(
            id="1",
            fields={"id": 1, "name": "test1_updated", "weight": 85.0},
            vectors={
                "dense": np.random.random(128).tolist(),
                "sparse": {1: 1.5, 2: 2.5},
            },
        )

        result = coll.upsert(doc1_updated)
        assert bool(result)
        assert result.ok()
        assert coll.stats.doc_count == 1

        # Step 5: Update doc
        doc2 = Doc(
            id="2",
            fields={"id": 2, "name": "test2", "weight": 90.0},
            vectors={
                "dense": np.random.random(128).tolist(),
                "sparse": {1: 3.0, 2: 4.0},
            },
        )

        # First insert doc2
        result = coll.insert(doc2)
        assert bool(result)
        assert result.ok()
        assert coll.stats.doc_count == 2

        # Then update it
        doc2_updated = Doc(
            id="2",
            fields={"id": 2, "name": "test2_updated", "weight": 95.0},
            vectors={
                "dense": np.random.random(128).tolist(),
                "sparse": {1: 3.5, 2: 4.5},
            },
        )

        result = coll.update(doc2_updated)
        assert bool(result)
        assert result.ok()
        assert coll.stats.doc_count == 2

        # Step 6: Fetch doc
        fetched_docs = coll.fetch(["1", "2"])
        assert len(fetched_docs) == 2
        assert "1" in fetched_docs
        assert "2" in fetched_docs
        assert fetched_docs["1"].field("name") == "test1_updated"
        assert fetched_docs["2"].field("name") == "test2_updated"

        # Step 7: Query doc
        query_result = coll.query(filter="id >= 1", topk=10)
        assert len(query_result) == 2

        # Step 8: Drop index
        coll.drop_index(field_name="name")

        # Step 9: Insert doc
        doc3 = Doc(
            id="3",
            fields={"id": 3, "name": "test3", "weight": 100.0},
            vectors={
                "dense": np.random.random(128).tolist(),
                "sparse": {1: 5.0, 2: 6.0},
            },
        )

        result = coll.insert(doc3)
        assert bool(result)
        assert result.ok()
        assert coll.stats.doc_count == 3

        # Step 10: Update doc
        doc3_updated = Doc(
            id="3",
            fields={"id": 3, "name": "test3_updated", "weight": 105.0},
            vectors={
                "dense": np.random.random(128).tolist(),
                "sparse": {1: 5.5, 2: 6.5},
            },
        )

        result = coll.update(doc3_updated)
        assert bool(result)
        assert result.ok()
        assert coll.stats.doc_count == 3

        # Step 11: Upsert doc
        doc4 = Doc(
            id="4",
            fields={"id": 4, "name": "test4", "weight": 110.0},
            vectors={
                "dense": np.random.random(128).tolist(),
                "sparse": {1: 7.0, 2: 8.0},
            },
        )

        result = coll.upsert(doc4)
        assert bool(result)
        assert result.ok()
        assert coll.stats.doc_count == 4

        # Step 12: Fetch doc
        fetched_docs = coll.fetch(["3", "4"])
        assert len(fetched_docs) == 2
        assert "3" in fetched_docs
        assert "4" in fetched_docs
        assert fetched_docs["3"].field("name") == "test3_updated"
        assert fetched_docs["4"].field("name") == "test4"

        # Step 13: Query doc
        query_result = coll.query(filter="id >= 3", topk=10)
        assert len(query_result) == 2

        # Step 14: Flush
        coll.flush()

        # Verify data is still accessible after flush
        fetched_docs = coll.fetch(["1", "2", "3", "4"])
        assert len(fetched_docs) == 4

        # Step 15: Destroy
        coll.destroy()

    @pytest.mark.parametrize(
        "data_type, index_param", VALID_VECTOR_DATA_TYPE_INDEX_PARAM_MAP_PARAMS
    )
    def test_vector_index_params(
        self,
        collection_temp_dir,
        collection_option: CollectionOption,
        data_type: DataType,
        index_param,
        single_vector_schema,
    ):
        vector_name = DEFAULT_VECTOR_FIELD_NAME[data_type]
        dimension = DEFAULT_VECTOR_DIMENSION

        coll = zvec.create_and_open(
            path=collection_temp_dir,
            schema=single_vector_schema,
            option=collection_option,
        )

        assert coll is not None, (
            f"Failed to create and open collection, {data_type}, {index_param}"
        )

        docs = {str(i): generate_doc(i, single_vector_schema) for i in range(5)}
        result = coll.insert(docs.values())
        assert len(result) == len(docs), (
            f"Expected 5 results, got {len(result)}, {data_type}, {index_param}"
        )
        for item in result:
            assert item.ok(), f"Insertion failed for, {data_type}, {index_param}"

        def check_result(
            label: str, metric_type: MetricType, quantize_type: QuantizeType
        ):
            query_vector = [1] * dimension
            if data_type in [DataType.SPARSE_VECTOR_FP16, DataType.SPARSE_VECTOR_FP32]:
                query_vector = {1: 1}

            fetch_result = coll.fetch([str(i) for i in range(len(docs))])
            assert len(fetch_result) == len(docs), (
                f"{label}, Expected 5 fetched docs, got {len(fetch_result)}, {data_type}, {index_param}"
            )
            for i in range(len(docs)):
                doc_id = str(i)
                assert doc_id in fetch_result, (
                    f"{label}, Document ID '{doc_id}' not found, {data_type}, {index_param}"
                )
                fetched_doc = fetch_result[doc_id]
                # Verify doc equal
                assert is_doc_equal(fetched_doc, docs[doc_id], single_vector_schema), (
                    f"{label}, doc not equal, insert: {docs[doc_id]}, fetched: {fetched_doc}, {data_type}, {index_param}"
                )

            query_result: list[Doc] = coll.query(
                Query(field_name=vector_name, vector=query_vector),
                include_vector=False,
                topk=len(docs),
            )
            assert len(query_result) == len(docs), (
                f"{label}, Expected {len(docs)} result, got {len(query_result)}, {data_type}, {index_param}"
            )
            inserted_ids = [str(i) for i in range(len(docs))]
            queried_ids = [doc.id for doc in query_result]
            assert set(inserted_ids) == set(queried_ids), (
                f"{label}, inserted_ids != queried_ids, insert: {inserted_ids}, query: {queried_ids}, {data_type}, {index_param}"
            )

            last_score = None
            for i, doc in enumerate(query_result):
                # Get the document's vector for comparison
                expect_doc = generate_doc(int(doc.id), single_vector_schema)
                doc_vector = expect_doc.vector(vector_name)
                expected_score = distance(
                    doc_vector,
                    query_vector,
                    metric_type,
                    data_type,
                    quantize_type,
                )
                print(f"query: {doc}, expect_core: {expected_score}")
                if quantize_type is QuantizeType.UNDEFINED:
                    assert is_float_equal(doc.score, expected_score), (
                        f"{label} top{i} pk{doc.id} score {doc.score:6f} expected:{expected_score:6f}, {data_type}, {index_param}"
                    )
                if last_score is not None:
                    if metric_type == MetricType.IP:
                        assert last_score >= doc.score, (
                            f"{label}, score not sorted, last_score: {last_score}, current_score: {doc.score}, {data_type}, {index_param}"
                        )
                    else:
                        assert last_score <= doc.score, (
                            f"{label}, score not sorted, last_score: {last_score}, current_score: {doc.score}, {data_type}, {index_param}"
                        )
                last_score = doc.score

        # default metric_type=IP, quantize_type=None
        check_result("pre_create_index", MetricType.IP, QuantizeType.UNDEFINED)

        # create index
        coll.create_index(
            field_name=vector_name,
            index_param=index_param,
            option=IndexOption(),
        )
        check_result(
            "post_create_index", index_param.metric_type, index_param.quantize_type
        )

        coll.drop_index(field_name=vector_name)
        check_result("post_drop_index", MetricType.IP, QuantizeType.UNDEFINED)

        new_docs = {str(i): generate_doc(i, single_vector_schema) for i in range(5, 8)}
        new_result = coll.insert(new_docs.values())
        assert len(new_result) == len(new_docs), (
            f"Expected {len(new_docs)} insertion results for new docs, got {len(new_result)} for vector {vector_name}"
        )
        for item in new_result:
            assert item.ok(), (
                f"New document insertion failed for vector {vector_name}: {item}"
            )
        docs |= new_docs
        coll.create_index(
            field_name=vector_name,
            index_param=index_param,
            option=IndexOption(),
        )

        check_result(
            "post_create_index2", index_param.metric_type, index_param.quantize_type
        )
        coll.destroy()


class TestColumnDDL:
    def test_add_column(self, basic_collection: Collection):
        basic_collection.add_column(
            field_schema=FieldSchema("income", DataType.INT32),
            expression="'weight' * 2",  # Simple expression
        )
        doc = Doc(
            id="1",
            fields={"id": 1, "name": "test", "weight": 80.5, "income": 1},
            vectors={
                "dense": np.random.random(128).tolist(),
                "sparse": {1: 1.0, 2: 2.0},
            },
        )

        result = basic_collection.insert(doc)
        assert bool(result), f"Expected 1 result, but got {len(result)}"
        assert result.ok(), (
            f"result={result},Insert operation failed with code = {result.code()}"
        )

        stats = basic_collection.stats
        assert stats is not None
        assert stats.doc_count == 1

    def test_add_column_with_default_option(self, basic_collection: Collection):
        # Add a new column with default option
        basic_collection.add_column(
            field_schema=FieldSchema("test_column_default", DataType.INT32),
            expression="100",
            option=AddColumnOption(),  # Default option
        )
        # Verify column was added by inserting data
        doc = Doc(
            id="1",
            fields={"id": 1, "name": "test", "weight": 80.5, "test_column_default": 1},
            vectors={
                "dense": np.random.random(128).tolist(),
                "sparse": {1: 1.0, 2: 2.0},
            },
        )

        result = basic_collection.insert(doc)
        assert bool(result), f"Expected 1 result, but got {len(result)}"
        assert result.ok(), (
            f"result={result},Insert operation failed with code = {result.code()}"
        )
        # Verify document was inserted
        stats = basic_collection.stats
        assert stats is not None
        assert stats.doc_count == 1

    @pytest.mark.parametrize("concurrency", [0, 1, 4, 8])
    def test_add_column_with_various_concurrency_options(
        self, basic_collection: Collection, concurrency
    ):
        field_name = f"test_column_concurrent_{concurrency}"
        basic_collection.add_column(
            field_schema=FieldSchema(field_name, DataType.INT32),
            expression="100",
            option=AddColumnOption(concurrency=concurrency),
        )

        doc = Doc(
            id="1",
            fields={"id": 1, "name": "test", "weight": 80.5, field_name: 200},
            vectors={
                "dense": np.random.random(128).tolist(),
                "sparse": {1: 1.0, 2: 2.0},
            },
        )
        result = basic_collection.insert(doc)
        assert bool(result), f"Expected 1 result, but got {len(result)}"
        assert result.ok(), (
            f"result={result},Insert operation failed with code = {result.code()}"
        )

        stats = basic_collection.stats
        assert stats is not None
        assert stats.doc_count == 1

    @pytest.mark.parametrize("data_type", SUPPORT_ADD_COLUMN_DATA_TYPE)
    def test_add_column_valid_data_types(self, basic_collection: Collection, data_type):
        field_name = f"test_field_{data_type.name.lower()}"

        # Add a new column with specific data type
        basic_collection.add_column(
            field_schema=FieldSchema(field_name, data_type),
            expression="1" if data_type != DataType.STRING else "'test'",
        )

        # Verify column was added by inserting data
        if data_type == DataType.STRING:
            field_value = "test_value"
        elif data_type in [DataType.ARRAY_STRING]:
            field_value = ["test_value"]
        elif data_type in [DataType.ARRAY_INT32, DataType.ARRAY_INT64]:
            field_value = [1, 2, 3]
        elif data_type in [DataType.ARRAY_FLOAT, DataType.ARRAY_DOUBLE]:
            field_value = [1.1, 2.2, 3.3]
        elif data_type == DataType.ARRAY_BOOL:
            field_value = [True, False]
        elif data_type in [DataType.FLOAT, DataType.DOUBLE]:
            field_value = 1.5
        elif data_type in [DataType.INT32, DataType.INT64]:
            field_value = 100
        elif data_type == DataType.BOOL:
            field_value = True
        else:
            field_value = 1

        doc = Doc(
            id="1",
            fields={
                "id": 1,
                "name": "test",
                "weight": 80.5,
                field_name: field_value,
            },
            vectors={
                "dense": np.random.random(128).tolist(),
                "sparse": {1: 1.0, 2: 2.0},
            },
        )
        result = basic_collection.insert(doc)
        assert bool(result), f"Expected 1 result, but got {len(result)}"
        assert result.ok(), (
            f"result={result},Insert operation failed with code = {result.code()}"
        )

        # Verify document was inserted
        stats = basic_collection.stats
        assert stats is not None
        assert stats.doc_count == 1

    @pytest.mark.parametrize("data_type", NOT_SUPPORT_ADD_COLUMN_DATA_TYPE)
    def test_add_column_invalid_data_types(
        self, basic_collection: Collection, data_type
    ):
        with pytest.raises(Exception) as exc_info:
            field_name = f"test_field_{data_type.name.lower()}"

            # Add a new column with specific data type
            basic_collection.add_column(
                field_schema=FieldSchema(field_name, data_type),
                expression="1" if data_type != DataType.STRING else "'test'",
            )

        assert NOT_SUPPORT_ADD_COLUMN_ERROR_MSG in str(exc_info.value)

    @pytest.mark.parametrize("nullable", [True, False])
    def test_add_column_with_nullable_options(
        self, basic_collection: Collection, nullable
    ):
        field_name = f"test_field_nullable_{str(nullable).lower()}"

        # Add a new column with specific nullable option
        basic_collection.add_column(
            field_schema=FieldSchema(field_name, DataType.INT32, nullable=nullable),
            expression="100",
        )

        # Verify column was added by inserting data
        doc = Doc(
            id="1",
            fields={"id": 1, "name": "test", "weight": 80.5, field_name: 200},
            vectors={
                "dense": np.random.random(128).tolist(),
                "sparse": {1: 1.0, 2: 2.0},
            },
        )
        result = basic_collection.insert(doc)
        assert bool(result), f"Expected 1 result, but got {len(result)}"
        assert result.ok(), (
            f"result={result},Insert operation failed with code = {result.code()}"
        )

        # Verify document was inserted
        stats = basic_collection.stats
        assert stats is not None
        assert stats.doc_count == 1

        # Verify column was added by inserting data
        doc = Doc(
            id="2",
            fields={"id": 2, "name": "test", "weight": 80.5, field_name: None},
            vectors={
                "dense": np.random.random(128).tolist(),
                "sparse": {1: 1.0, 2: 2.0},
            },
        )
        if nullable:
            result = basic_collection.insert(doc)
            assert bool(result), f"Expected 1 result, but got {len(result)}"
            assert result.ok(), (
                f"result={result},Insert operation failed with code = {result.code()}"
            )
        else:
            with pytest.raises(ValueError) as e:
                basic_collection.insert(doc)
            assert (
                "Field 'test_field_nullable_false': expected non-nullable type"
                in str(e.value)
            )

        # Verify document was inserted
        stats = basic_collection.stats
        assert stats is not None
        if nullable:
            assert stats.doc_count == 2
        else:
            assert stats.doc_count == 1

    @pytest.mark.parametrize(
        "expression",
        [
            "1",  # Constant integer
            "1.5",  # Constant float
            "'test'",  # Constant string
            "id",  # Reference to existing field
            "weight * 2",  # Simple arithmetic
            "weight + id",  # Complex arithmetic
            "CASE WHEN weight > 50 THEN 1 ELSE 0 END",  # Conditional expression
        ],
    )
    def test_add_column_with_different_expressions(
        self, basic_collection: Collection, expression
    ):
        field_name = f"test_field_expr_{abs(hash(expression)) % 1000}"

        # Add a new column with specific expression
        basic_collection.add_column(
            field_schema=FieldSchema(field_name, DataType.INT32),
            expression=expression,
        )

        # Verify column was added by inserting data
        doc = Doc(
            id="1",
            fields={"id": 1, "name": "test", "weight": 80.5, field_name: 200},
            vectors={
                "dense": np.random.random(128).tolist(),
                "sparse": {1: 1.0, 2: 2.0},
            },
        )
        result = basic_collection.insert(doc)
        assert bool(result), f"Expected 1 result, but got {len(result)}"
        assert result.ok(), (
            f"result={result},Insert operation failed with code = {result.code()}"
        )

        # Verify document was inserted
        stats = basic_collection.stats
        assert stats is not None
        assert stats.doc_count == 1

    def test_add_column_with_index_param(self, basic_collection: Collection):
        basic_collection.add_column(
            field_schema=FieldSchema(
                "indexed_field",
                DataType.INT32,
                index_param=InvertIndexParam(enable_range_optimization=True),
            ),
            expression="id * 2",
        )

        # Verify column was added by inserting data
        doc = Doc(
            id="1",
            fields={"id": 1, "name": "test", "weight": 80.5, "indexed_field": 200},
            vectors={
                "dense": np.random.random(128).tolist(),
                "sparse": {1: 1.0, 2: 2.0},
            },
        )
        result = basic_collection.insert(doc)
        assert bool(result), f"Expected 1 result, but got {len(result)}"
        assert result.ok(), (
            f"result={result},Insert operation failed with code = {result.code()}"
        )

        # Verify document was inserted
        stats = basic_collection.stats
        assert stats is not None
        assert stats.doc_count == 1

    @pytest.mark.parametrize(
        "field_name",
        [
            "a",  # Minimum length
            "a" * 32,  # Maximum length (32 characters)
            "valid_field_name_123",  # Alphanumeric with underscore
            "Valid-Field-Name",  # With hyphens
            "_underscore_start",  # Starting with underscore
            "field_name_with_123_numbers",  # Numbers in middle
            "FIELD_NAME_UPPERCASE",  # Uppercase
            # "field_with_nums_123_and_hyphens-456",  # Complex valid name within limit
        ],
    )
    def test_add_column_with_valid_field_names(
        self, basic_collection: Collection, field_name
    ):
        basic_collection.add_column(
            field_schema=FieldSchema(field_name, DataType.INT32), expression="200"
        )

        doc = Doc(
            id="1",
            fields={"id": 1, "name": "test", "weight": 80.5, field_name: 300},
            vectors={
                "dense": np.random.random(128).tolist(),
                "sparse": {1: 1.0, 2: 2.0},
            },
        )
        result = basic_collection.insert(doc)
        assert bool(result), f"Expected 1 result, but got {len(result)}"
        assert result.ok(), (
            f"result={result},Insert operation failed with code = {result.code()}"
        )

        stats = basic_collection.stats
        assert stats is not None
        assert stats.doc_count == 1

    @pytest.mark.parametrize(
        "invalid_field_name",
        [
            "",  # Empty string
            " ",  # Space only
            "a" * 33,  # Too long (33 characters, exceeds 32)
            "field name",  # Contains space
            "field.name",  # Contains dot
            "field@name",  # Contains special character
            "field/name",  # Contains slash
            "field\\name",  # Contains backslash
            "field$name",  # Contains dollar sign
            "field+name",  # Contains plus sign
            "field=name",  # Contains equals sign
            None,  # None value
        ],
    )
    def test_add_column_with_invalid_field_names(
        self, basic_collection: Collection, invalid_field_name
    ):
        with pytest.raises(Exception) as exc_info:
            basic_collection.add_column(
                field_schema=FieldSchema(invalid_field_name, DataType.INT32),
                expression="100",
            )

        if invalid_field_name is None:
            assert "validate failed" in str(exc_info.value), (
                "Error message is unreasonable: e=" + str(exc_info.value)
            )
        else:
            assert (
                "invalid" in str(exc_info.value).lower()
                or "name" in str(exc_info.value).lower()
            )

    def test_alter_column_rename(self, basic_collection: Collection):
        basic_collection.alter_column(
            old_name="weight",
            new_name="mass",
            option=AlterColumnOption(),
        )
        doc = Doc(
            id="1",
            fields={"id": 1, "name": "test", "mass": 80.5},  # Use new name
            vectors={
                "dense": np.random.random(128).tolist(),
                "sparse": {1: 1.0, 2: 2.0},
            },
        )
        result = basic_collection.insert(doc)
        assert bool(result), f"Expected 1 result, but got {len(result)}"
        assert result.ok(), (
            f"result={result},Insert operation failed with code = {result.code()}"
        )

        stats = basic_collection.stats
        assert stats is not None
        assert stats.doc_count == 1

    def test_alter_column_non_exist(self, basic_collection: Collection):
        with pytest.raises(Exception) as exc_info:
            basic_collection.alter_column(
                old_name="non_existing",
                new_name="new_name",
                field_schema=FieldSchema("new_name", DataType.STRING),
            )
        assert "column non_existing not found" in str(exc_info.value), (
            "Error message is unreasonable: e=" + str(exc_info.value)
        )

    def test_alter_column_with_default_option(self, basic_collection: Collection):
        basic_collection.add_column(
            field_schema=FieldSchema("original_field", DataType.INT32), expression="100"
        )

        basic_collection.alter_column(
            old_name="original_field",
            new_name="renamed_field",
            option=AlterColumnOption(),
        )

        doc = Doc(
            id="1",
            fields={"id": 1, "name": "test", "weight": 80.5, "renamed_field": 200},
            vectors={
                "dense": np.random.random(128).tolist(),
                "sparse": {1: 1.0, 2: 2.0},
            },
        )
        result = basic_collection.insert(doc)
        assert bool(result), f"Expected 1 result, but got {len(result)}"
        assert result.ok(), (
            f"result={result},Insert operation failed with code = {result.code()}"
        )

        stats = basic_collection.stats
        assert stats is not None
        assert stats.doc_count == 1

    @pytest.mark.parametrize("concurrency", [0, 1, 4, 8])
    def test_alter_column_with_various_concurrency_options(
        self, basic_collection: Collection, concurrency
    ):
        old_field_name = f"orig_field_{concurrency}"
        new_field_name = f"modified_field_{concurrency}"

        basic_collection.add_column(
            field_schema=FieldSchema(old_field_name, DataType.INT32),
            expression="100",
        )

        basic_collection.alter_column(
            old_name=old_field_name,
            new_name=new_field_name,
            option=AlterColumnOption(concurrency=concurrency),
        )

        doc = Doc(
            id="1",
            fields={"id": 1, "name": "test", "weight": 80.5, new_field_name: 200},
            vectors={
                "dense": np.random.random(128).tolist(),
                "sparse": {1: 1.0, 2: 2.0},
            },
        )

        result = basic_collection.insert(doc)
        assert bool(result), f"Expected 1 result, but got {len(result)}"
        assert result.ok(), (
            f"result={result},Insert operation failed with code = {result.code()}"
        )

        stats = basic_collection.stats
        assert stats is not None
        assert stats.doc_count == 1

    @pytest.mark.parametrize(
        "old_field_name,new_field_name",
        [
            ("a", "new_a"),  # Minimum length
            (
                "abcdefghijklmnopqrstuvwxyz123456",
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            ),  # Maximum length (32 characters)
            ("valid_field_name_123", "new_valid_field"),  # Alphanumeric with underscore
            ("Valid-Field-Name", "New-Field-Name"),  # With hyphens
            ("_underscore_start", "new_underscore"),  # Starting with underscore
            ("field_name_with_123_numbers", "new_with_nums"),  # Numbers in middle
            ("FIELD_NAME_UPPERCASE", "new_uppercase"),  # Uppercase
            (
                "field_with_nums_3_and_hyphens-6",
                "new_field_hyphens",
            ),  # Complex valid name
        ],
    )
    def test_alter_column_field_name_valid(
        self, basic_collection: Collection, old_field_name, new_field_name
    ):
        basic_collection.add_column(
            field_schema=FieldSchema(old_field_name, DataType.INT32),
            expression="100",
        )
        basic_collection.alter_column(
            old_name=old_field_name,
            new_name=new_field_name,
            option=AlterColumnOption(),
        )
        doc = Doc(
            id="1",
            fields={"id": 1, "name": "test", "weight": 80.5, new_field_name: 200},
            vectors={
                "dense": np.random.random(128).tolist(),
                "sparse": {1: 1.0, 2: 2.0},
            },
        )

        result = basic_collection.insert(doc)
        assert bool(result), f"Expected 1 result, but got {len(result)}"
        assert result.ok(), (
            f"result={result},Insert operation failed with code = {result.code()}"
        )

        stats = basic_collection.stats
        assert stats is not None
        assert stats.doc_count == 1

    @pytest.mark.parametrize(
        "valid_old_name,invalid_new_name",
        [
            ("temp_field", ""),  # Empty new name
            ("temp_field", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),  # Too long new name
            ("temp_field", "field name"),  # New name with space
            ("temp_field", "field.name"),  # New name with dot
            ("temp_field", "field@name"),  # New name with special character
            ("temp_field", "field/name"),  # New name with slash
            ("temp_field", "field\\name"),  # New name with backslash
            ("temp_field", "field$name"),  # New name with dollar sign
            ("temp_field", "field+name"),  # New name with plus sign
            ("temp_field", "field=name"),  # New name with equals sign
            ("temp_field", None),  # None new name
        ],
    )
    def test_alter_column_with_invalid_field_names(
        self, basic_collection: Collection, valid_old_name, invalid_new_name
    ):
        basic_collection.add_column(
            field_schema=FieldSchema("temp_field", DataType.INT32), expression="100"
        )
        with pytest.raises(Exception) as exc_info:
            basic_collection.alter_column(
                old_name=valid_old_name,
                new_name=invalid_new_name if invalid_new_name is not None else "",
                field_schema=FieldSchema(
                    invalid_new_name if invalid_new_name is not None else "",
                    DataType.INT32,
                ),
            )

        assert (
            "invalid" in str(exc_info.value).lower()
            or "name" in str(exc_info.value).lower()
            or "incompatible" in str(exc_info.value).lower()
        )

    def test_drop_column_exist(self, basic_collection: Collection):
        basic_collection.add_column(
            field_schema=FieldSchema("temp_field", DataType.INT32), expression="100"
        )
        doc = Doc(
            id="1",
            fields={"id": 1, "name": "test", "weight": 80.5, "temp_field": 1},
            vectors={
                "dense": np.random.random(128).tolist(),
                "sparse": {1: 1.0, 2: 2.0},
            },
        )

        result = basic_collection.insert(doc)
        assert bool(result), f"Expected 1 result, but got {len(result)}"
        assert result.ok(), (
            f"result={result},Insert operation failed with code = {result.code()}"
        )

        stats = basic_collection.stats
        assert stats is not None
        assert stats.doc_count == 1

        basic_collection.drop_column("temp_field")
        doc = Doc(
            id="2",
            fields={"id": 2, "name": "test", "weight": 80.5, "temp_field": 1},
            vectors={
                "dense": np.random.random(128).tolist(),
                "sparse": {1: 1.0, 2: 2.0},
            },
        )
        with pytest.raises(Exception) as exc_info:
            result = basic_collection.insert(doc)

        assert SCHEMA_VALIDATE_ERROR_MSG in str(exc_info.value)

    def test_drop_column_non_exist(self, basic_collection: Collection):
        with pytest.raises(Exception) as exc_info:
            basic_collection.drop_column("non_existing_column")
        assert NOT_EXIST_COLUMN_TO_DROP_ERROR_MSG in str(exc_info.value)

    @pytest.mark.parametrize(
        "field_name",
        [
            "a",  # Minimum length
            "a" * 32,  # Maximum length (32 characters)
            "valid_field_name_123",  # Alphanumeric with underscore
            "Valid-Field-Name",  # With hyphens
            "_underscore_start",  # Starting with underscore
            "field_name_with_123_numbers",  # Numbers in middle
            "FIELD_NAME_UPPERCASE",  # Uppercase
            "field_with_nums_3_and_hyphens-6",  # Complex valid name within limit
        ],
    )
    def test_drop_column_field_name_valid(
        self, basic_collection: Collection, field_name
    ):
        basic_collection.add_column(
            field_schema=FieldSchema(field_name, DataType.INT32), expression="100"
        )
        doc = Doc(
            id="1",
            fields={"id": 1, "name": "test", "weight": 80.5, field_name: 200},
            vectors={
                "dense": np.random.random(128).tolist(),
                "sparse": {1: 1.0, 2: 2.0},
            },
        )

        result = basic_collection.insert(doc)
        assert bool(result), f"Expected 1 result, but got {len(result)}"
        assert result.ok(), (
            f"result={result},Insert operation failed with code = {result.code()}"
        )

        stats = basic_collection.stats
        assert stats is not None
        assert stats.doc_count == 1

        basic_collection.drop_column(field_name)

        doc = Doc(
            id="2",
            fields={"id": 2, "name": "test", "weight": 80.5, field_name: 200},
            vectors={
                "dense": np.random.random(128).tolist(),
                "sparse": {1: 1.0, 2: 2.0},
            },
        )
        with pytest.raises(Exception) as exc_info:
            result = basic_collection.insert(doc)

        assert SCHEMA_VALIDATE_ERROR_MSG in str(exc_info.value)
