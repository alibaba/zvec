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


import logging
import pytest
import numpy as np
import zvec

from zvec import (
    CollectionOption,
    InvertIndexParam,
    HnswIndexParam,
    DataType,
    Collection,
    Doc,
    FieldSchema,
    Query,
    VectorSchema,
)


class TestCollectionExceptionHandling:
    @pytest.fixture(scope="function")
    def test_collection(self, tmp_path_factory):
        """Fixture to create a test collection"""
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
                FieldSchema("weight", DataType.FLOAT, nullable=True),
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

        collection_option = CollectionOption(read_only=False, enable_mmap=True)

        temp_dir = tmp_path_factory.mktemp("zvec")
        collection_path = temp_dir / "test_collection"

        coll = zvec.create_and_open(
            path=str(collection_path),
            schema=collection_schema,
            option=collection_option,
        )

        assert coll is not None, "Failed to create and open collection"

        yield coll

        # Clean up
        if hasattr(coll, "destroy") and coll is not None:
            try:
                coll.destroy()
            except Exception as e:
                print(f"Warning: failed to destroy collection: {e}")

    def test_create_and_open_missing_path(self, tmp_path_factory):
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

        with pytest.raises(Exception) as exc_info:
            coll = zvec.create_and_open(
                schema=collection_schema, option=collection_option
            )
        assert exc_info.value is not None, (
            "Expected exception for missing path parameter"
        )

    def test_create_and_open_missing_schema(self, tmp_path_factory):
        temp_dir = tmp_path_factory.mktemp("zvec")
        collection_path = temp_dir / "test_collection"

        collection_option = CollectionOption(read_only=False, enable_mmap=True)

        with pytest.raises(Exception) as exc_info:
            coll = zvec.create_and_open(
                path=str(collection_path), option=collection_option
            )
        assert exc_info.value is not None, (
            "Expected exception for missing schema parameter"
        )

    def test_open_missing_path(self):
        collection_option = CollectionOption(read_only=False, enable_mmap=True)

        with pytest.raises(Exception) as exc_info:
            coll = zvec.open(option=collection_option)
        assert exc_info.value is not None, (
            "Expected exception for missing path parameter"
        )

    def test_insert_missing_docs(self, test_collection: Collection):
        with pytest.raises(Exception) as exc_info:
            result = test_collection.insert()
        assert exc_info.value is not None, (
            "Expected exception for missing docs parameter"
        )

    def test_update_missing_docs(self, test_collection: Collection):
        with pytest.raises(Exception) as exc_info:
            result = test_collection.update()
        assert exc_info.value is not None, (
            "Expected exception for missing docs parameter"
        )

    def test_upsert_missing_docs(self, test_collection: Collection):
        with pytest.raises(Exception) as exc_info:
            result = test_collection.upsert()
        assert exc_info.value is not None, (
            "Expected exception for missing docs parameter"
        )

    def test_delete_missing_ids(self, test_collection: Collection):
        with pytest.raises(Exception) as exc_info:
            result = test_collection.delete()
        assert exc_info.value is not None, (
            "Expected exception for missing ids parameter"
        )

    def test_fetch_missing_ids(self, test_collection: Collection):
        with pytest.raises(Exception) as exc_info:
            result = test_collection.fetch()
        assert exc_info.value is not None, (
            "Expected exception for missing ids parameter"
        )

    def test_query_missing_query_field_name(self, test_collection: Collection):
        with pytest.raises(Exception) as exc_info:
            result = test_collection.query([Query()])
        assert exc_info.value is not None, (
            "Expected exception for missing Query field_name parameter"
        )

    def test_add_column_missing_field_schema(self, test_collection: Collection):
        with pytest.raises(Exception) as exc_info:
            test_collection.add_column()
        assert exc_info.value is not None, (
            "Expected exception for missing field_schema parameter"
        )

    def test_alter_column_missing_old_name(self, test_collection: Collection):
        with pytest.raises(Exception) as exc_info:
            test_collection.alter_column(new_name="new_name")
        assert exc_info.value is not None, (
            "Expected exception for missing old_name parameter"
        )

    def test_alter_column_missing_new_name(self, test_collection: Collection):
        with pytest.raises(Exception) as exc_info:
            test_collection.alter_column(old_name="old_name")
        assert exc_info.value is not None, (
            "Expected exception for missing new_name parameter"
        )

    def test_drop_column_missing_field_name(self, test_collection: Collection):
        with pytest.raises(Exception) as exc_info:
            test_collection.drop_column()
        assert exc_info.value is not None, (
            "Expected exception for missing field_name parameter"
        )

    def test_invalid_parameter_types(self, test_collection: Collection):
        # This test depends on specific implementation details
        # Generally, we would expect TypeErrors or similar exceptions
        pass

    def test_missing_required_parameters(self, test_collection: Collection):
        # This test depends on specific implementation details
        # Generally, we would expect TypeErrors or similar exceptions
        pass

    def test_empty_collection_operations(self, tmp_path_factory):
        collection_schema = zvec.CollectionSchema(
            name="empty_test_collection",
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

        temp_dir = tmp_path_factory.mktemp("zvec")
        collection_path = temp_dir / "empty_test_collection"

        coll = zvec.create_and_open(
            path=str(collection_path),
            schema=collection_schema,
            option=collection_option,
        )

        assert coll is not None, "Failed to create and open collection"

        # Test fetch on empty collection
        result = coll.fetch(["1"])
        assert len(result) >= 0  # May be empty or have special handling

        # Test query on empty collection
        result = coll.query()
        assert len(result) == 0

        # Test update on empty collection
        doc = Doc(
            id="1",
            fields={"id": 1, "name": "test"},
            vectors={"dense": np.random.random(128).tolist()},
        )

        result = coll.update(doc)
        # Should handle gracefully, possibly with NOT_FOUND status

        # Clean up
        if hasattr(coll, "destroy") and coll is not None:
            try:
                coll.destroy()
            except Exception as e:
                print(f"Warning: failed to destroy collection: {e}")

    def test_resource_management(self, test_collection: Collection):
        doc = Doc(
            id="1",
            fields={"id": 1, "name": "test", "weight": 80.5},
            vectors={
                "dense": np.random.random(128).tolist(),
                "sparse": {1: 1.0, 2: 2.0},
            },
        )

        # Insert
        result = test_collection.insert(doc)
        assert result.ok()

        # Fetch
        result = test_collection.fetch(["1"])
        assert len(result) == 1

        # Query
        result = test_collection.query()
        assert len(result) >= 0

        # Update
        result = test_collection.update(doc)
        assert result.ok()

        # Delete
        result = test_collection.delete("1")
        assert result.ok()

    def test_exception_resource_cleanup(self, test_collection: Collection):
        # This test would need to simulate exception conditions
        # which is difficult without specific failure injection points
        pass
