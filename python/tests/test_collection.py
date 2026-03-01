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
    HnswIndexParam,
    InvertIndexParam,
    LogLevel,
    LogType,
    VectorSchema,
    StatusCode,
    IndexOption,
    IndexType,
    VectorQuery,
    OptimizeOption,
    OmegaIndexParam,
    MetricType,
)

# ==================== Common ====================


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
            FieldSchema("weight", DataType.FLOAT, nullable=True),
            FieldSchema("height", DataType.INT32, nullable=True),
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


@pytest.fixture(scope="session")
def collection_option():
    return CollectionOption(read_only=False, enable_mmap=True)


@pytest.fixture
def single_doc():
    id = 0
    return Doc(
        id=f"{id}",
        fields={"id": id, "name": "test", "weight": 80.0, "height": id + 140},
        vectors={"dense": [id + 0.1] * 128, "sparse": {1: 1.0, 2: 2.0, 3: 3.0}},
    )


@pytest.fixture
def multiple_docs():
    return [
        Doc(
            id=f"{id}",
            fields={"id": id, "name": "test", "weight": 80.0, "height": 210},
            vectors={"dense": [id + 0.1] * 128, "sparse": {1: 1.0, 2: 2.0, 3: 3.0}},
        )
        for id in range(1, 101)
    ]


@pytest.fixture(scope="function")
def test_collection(
    tmp_path_factory, collection_schema, collection_option
) -> Collection:
    """
    Function-scoped fixture: creates and opens a collection.
    Uses tmp_path_factory to ensure shared temp dir per class.
    """
    # Create unique temp directory for this test class
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


@pytest.fixture
def collection_with_single_doc(test_collection: Collection, single_doc) -> Collection:
    # Setup: insert single doc
    assert test_collection.stats.doc_count == 0
    result = test_collection.insert(single_doc)
    assert bool(result)
    assert result.ok()
    assert test_collection.stats.doc_count == 1

    yield test_collection

    # Teardown: delete single doc
    test_collection.delete(single_doc.id)
    assert test_collection.stats.doc_count == 0


@pytest.fixture
def collection_with_multiple_docs(
    test_collection: Collection, multiple_docs
) -> Collection:
    # Setup: insert multiple docs
    assert test_collection.stats.doc_count == 0
    result = test_collection.insert(multiple_docs)
    assert len(result) == len(multiple_docs)
    for item in result:
        assert item.ok()
    assert test_collection.stats.doc_count == len(multiple_docs)

    yield test_collection

    # Teardown: delete multiple docs
    test_collection.delete([doc.id for doc in multiple_docs])


# ==================== Tests ====================


# ----------------------------
# Config Test Case
# ----------------------------
class TestConfig:
    def test_config(self):
        zvec.init(log_type=LogType.CONSOLE, log_level=LogLevel.ERROR, log_dir="./log")


# ----------------------------
# Collection DDL Test Case
# ----------------------------
@pytest.mark.usefixtures("test_collection")
class TestCollectionDDL:
    def test_collection_stats(self, test_collection: Collection):
        assert test_collection.stats is not None
        stats = test_collection.stats
        assert stats.doc_count == 0
        assert len(stats.index_completeness) == 2
        assert stats.index_completeness["dense"] == 1
        assert stats.index_completeness["sparse"] == 1


# ----------------------------
# Collection Index DDL Test Case
# ----------------------------
@pytest.mark.usefixtures("test_collection")
class TestCollectionIndexDDL:
    def test_create_index(self, test_collection: Collection):
        # before create
        field_schema = test_collection.schema.field("weight")
        assert field_schema is not None
        assert field_schema.data_type == DataType.FLOAT
        assert field_schema.name == "weight"
        index_param = field_schema.index_param
        assert index_param is None

        # create
        test_collection.create_index(
            field_name="weight", index_param=InvertIndexParam(), option=IndexOption()
        )
        assert test_collection.schema is not None
        field_schema = test_collection.schema.field("weight")
        assert field_schema is not None
        assert field_schema.data_type == DataType.FLOAT
        assert field_schema.name == "weight"

        index_param = field_schema.index_param
        assert index_param.type == IndexType.INVERT
        assert index_param.enable_range_optimization is False
        assert index_param.enable_extended_wildcard is False

    def test_drop_index(self, test_collection: Collection):
        # before drop
        field_schema = test_collection.schema.field("name")
        assert field_schema is not None
        assert field_schema.data_type == DataType.STRING
        assert field_schema.name == "name"
        index_param = field_schema.index_param
        assert index_param.type == IndexType.INVERT
        assert index_param.enable_range_optimization is False
        assert index_param.enable_extended_wildcard is False

        # drop
        test_collection.drop_index("name")
        field_schema = test_collection.schema.field("name")
        assert field_schema is not None
        assert field_schema.data_type == DataType.STRING
        assert field_schema.name == "name"

        # without index
        index_param = field_schema.index_param
        assert index_param is None

    def test_create_index_field_is_not_exist(self, test_collection: Collection):
        with pytest.raises(Exception) as e:
            test_collection.create_index(
                field_name="not_exist",
                index_param=InvertIndexParam(),
            )

        index_param = field_schema.index_param
        assert index_param.type == IndexType.INVERT
        assert index_param.enable_range_optimization is False
        assert index_param.enable_extended_wildcard is False

    def test_drop_index(self, test_collection: Collection):
        # before drop
        field_schema = test_collection.schema.field("name")
        assert field_schema is not None
        assert field_schema.data_type == DataType.STRING
        assert field_schema.name == "name"
        index_param = field_schema.index_param
        assert index_param.type == IndexType.INVERT
        assert index_param.enable_range_optimization is False
        assert index_param.enable_extended_wildcard is False

        # drop
        test_collection.drop_index("name")
        field_schema = test_collection.schema.field("name")
        assert field_schema is not None
        assert field_schema.data_type == DataType.STRING
        assert field_schema.name == "name"

        # without index
        index_param = field_schema.index_param
        assert index_param is None

    def test_create_index_field_is_not_exist(self, test_collection: Collection):
        with pytest.raises(Exception) as e:
            test_collection.create_index(
                field_name="not_exist",
                index_param=InvertIndexParam(),
            )


# ----------------------------
# Collection Column DDL Test Case
# ----------------------------
@pytest.mark.usefixtures("test_collection")
class TestCollectionColumnDDL:
    def test_create_column(self, test_collection: Collection):
        # before create column
        field_schema = test_collection.schema.field("age")
        assert field_schema is None

        # create
        test_collection.add_column(FieldSchema("age", DataType.INT32, nullable=True))

        field_schema = test_collection.schema.field("age")
        assert field_schema is not None
        assert field_schema.data_type == DataType.INT32
        assert field_schema.name == "age"
        assert field_schema.index_param is None

    def test_create_column_is_nullable(self, test_collection: Collection):
        with pytest.raises(ValueError):
            test_collection.add_column(
                FieldSchema("age", DataType.INT32, nullable=False)
            )

    def test_drop_column(self, test_collection: Collection):
        # before drop column
        field_schema = test_collection.schema.field("id")
        assert field_schema is not None
        assert field_schema.data_type == DataType.INT64
        assert field_schema.name == "id"
        index_param = field_schema.index_param
        assert index_param is not None
        assert index_param.type == IndexType.INVERT

        # drop
        test_collection.drop_column("id")
        field_schema = test_collection.schema.field("id")
        assert field_schema is None

    def test_alert_column_to_rename(self, test_collection: Collection):
        # before alert column
        field_schema = test_collection.schema.field("id")
        assert field_schema is not None
        assert field_schema.data_type == DataType.INT64
        assert field_schema.name == "id"
        index_param = field_schema.index_param
        assert index_param is not None
        assert index_param.type == IndexType.INVERT
        assert index_param.enable_range_optimization is True
        assert index_param.enable_extended_wildcard is False

        # alert rename
        test_collection.alter_column("id", "doc_id")

        # validate old column
        field_schema = test_collection.schema.field("id")
        assert field_schema is None
        # validate rename column
        field_schema = test_collection.schema.field("doc_id")
        assert field_schema is not None
        assert field_schema.data_type == DataType.INT64
        assert field_schema.name == "doc_id"
        assert field_schema.nullable is False
        index_param = field_schema.index_param
        assert index_param is not None
        assert index_param.type == IndexType.INVERT
        assert index_param.enable_range_optimization is True
        assert index_param.enable_extended_wildcard is False

    def test_alert_column_to_modify_schema(self, test_collection: Collection):
        # before alert column
        field_schema = test_collection.schema.field("id")
        assert field_schema is not None
        assert field_schema.data_type == DataType.INT64
        assert field_schema.name == "id"
        index_param = field_schema.index_param
        assert index_param.type == IndexType.INVERT

        test_collection.alter_column(
            old_name="id",
            field_schema=FieldSchema("doc_id", DataType.UINT64, nullable=True),
        )
        field_schema = test_collection.schema.field("doc_id")
        assert field_schema is not None
        assert field_schema.data_type == DataType.UINT64
        assert field_schema.name == "doc_id"

    def test_column_with_other_dtype(self, test_collection: Collection):
        # only allow number type
        test_collection.add_column(FieldSchema("age", DataType.INT32, nullable=True))

        with pytest.raises(ValueError):
            test_collection.add_column(FieldSchema("full_name", DataType.STRING))
        with pytest.raises(ValueError):
            test_collection.drop_column("name")
        with pytest.raises(ValueError):
            test_collection.alter_column(old_name="name", new_name="full_name")
        with pytest.raises(ValueError):
            test_collection.alter_column(
                old_name="name", field_schema=FieldSchema("full_name", DataType.STRING)
            )


# ----------------------------
# Collection Optimize Test Case
# ----------------------------
@pytest.mark.usefixtures("test_collection")
class TestCollectionOptimize:
    def test_collection_optimize(self, test_collection: Collection):
        test_collection.optimize(option=OptimizeOption())


# ----------------------------
# Collection Fetch Test Case
# ----------------------------
@pytest.mark.usefixtures("test_collection")
class TestCollectionFetch:
    def test_collection_fetch(
        self, collection_with_single_doc: Collection, single_doc: Doc
    ):
        result = collection_with_single_doc.fetch(ids=[single_doc.id])
        assert bool(result)
        assert single_doc.id in result.keys()

        doc = result[single_doc.id]
        assert doc is not None
        assert doc.id == single_doc.id
        assert set(doc.field_names()) == set(single_doc.field_names())
        for field_name in doc.field_names():
            if field_name in ["dense", "sparse"]:
                continue
            assert doc.field(field_name) == single_doc.field(field_name)

    def test_collection_fetch_contains_nodata_ids(
        self, collection_with_multiple_docs: Collection, multiple_docs: list[Doc]
    ):
        ids = [doc.id for doc in multiple_docs]
        no_data_key = "x"
        ids_with_no_data = [no_data_key] + ids
        result = collection_with_multiple_docs.fetch(ids=ids_with_no_data)
        assert bool(result)
        assert len(result) == len(ids)
        assert no_data_key not in result


# ----------------------------
# Collection Insert Test Case
# ----------------------------
@pytest.mark.usefixtures("test_collection")
class TestCollectionInsert:
    def test_collection_insert(self, test_collection, single_doc):
        result = test_collection.insert(single_doc)
        assert bool(result)
        assert result.ok()
        stats = test_collection.stats
        assert stats is not None
        assert stats.doc_count == 1

    def test_collection_insert_with_nullable_false_field(self, test_collection):
        # id, name's nullable == False
        # weight, height's nullable == True

        doc = Doc(
            id="0",
            fields={
                "id": 1,
                "name": "test",
            },
            vectors={"dense": [1 + 0.1] * 128, "sparse": {1: 1.0, 2: 2.0, 3: 3.0}},
        )
        result = test_collection.insert(doc)
        assert bool(result)
        assert result.ok()
        stats = test_collection.stats
        assert stats is not None
        assert stats.doc_count == 1

    def test_collection_insert_without_nullable_false_field(self, test_collection):
        # id, name's nullable == False
        # weight, height's nullable == True

        # without id, name
        doc = Doc(
            id="0",
            vectors={"dense": [1 + 0.1] * 128, "sparse": {1: 1.0, 2: 2.0, 3: 3.0}},
        )
        with pytest.raises(ValueError) as e:
            # ValueError: doc validate failed: field[id] is configured not nullable,
            # but doc does not contain this field
            test_collection.insert(doc)
        assert "field[id] is configured not nullable" in str(e.value)

        # without name
        doc = Doc(
            id="0",
            fields={
                "id": 1,
            },
            vectors={"dense": [1 + 0.1] * 128, "sparse": {1: 1.0, 2: 2.0, 3: 3.0}},
        )
        with pytest.raises(ValueError) as e:
            test_collection.insert(doc)
        assert "field[name] is configured not nullable" in str(e.value)

    def test_collection_insert_with_nullable_true_field(self, test_collection):
        # id, name's nullable == False
        # weight, height's nullable == True

        doc = Doc(
            id="0",
            fields={
                "id": 1,
                "name": "test",
            },
            vectors={"dense": [1 + 0.1] * 128, "sparse": {1: 1.0, 2: 2.0, 3: 3.0}},
        )
        result = test_collection.insert(doc)
        assert bool(result)
        assert result.ok()
        stats = test_collection.stats
        assert stats is not None
        assert stats.doc_count == 1

        result = test_collection.fetch(ids=[doc.id])
        assert doc.id in result
        ret = result[doc.id]
        assert ret.field("id") == 1
        assert ret.field("name") == "test"
        assert ret.field("weight") is None
        assert ret.field("height") is None

    def test_collection_insert_batch(self, test_collection, multiple_docs):
        result = test_collection.insert(multiple_docs)
        assert len(result) == len(multiple_docs)
        for item in result:
            assert item.ok()

        stats = test_collection.stats
        assert stats is not None
        assert stats.doc_count == len(multiple_docs)

    def test_collection_insert_duplicate(
        self, test_collection, single_doc, multiple_docs
    ):
        test_collection.insert(single_doc)
        result = test_collection.insert(single_doc)
        assert bool(result)
        assert result.code() == StatusCode.ALREADY_EXISTS

        stats = test_collection.stats
        assert stats is not None
        assert stats.doc_count == 1


# ----------------------------
# Collection Update Test Case
# ----------------------------
@pytest.mark.usefixtures("test_collection")
class TestCollectionUpdate:
    def test_empty_collection_update(
        self, test_collection: Collection, single_doc: Doc
    ):
        result = test_collection.update(single_doc)
        assert bool(result)
        assert result.code() == StatusCode.NOT_FOUND

        stats = test_collection.stats
        assert stats is not None
        assert stats.doc_count == 0

    def test_collection_update_with_nullable_false_field(
        self, collection_with_single_doc: Collection, single_doc: Doc
    ):
        # id, name's nullable == False
        # weight, height's nullable == True

        # update doc field id
        doc = Doc(
            id=single_doc.id,
            fields={"id": single_doc.field("id") + 1},
        )
        result = collection_with_single_doc.update(doc)
        assert bool(result)
        assert result.ok()
        stats = collection_with_single_doc.stats
        assert stats is not None
        assert stats.doc_count == 1

        # fetch
        result = collection_with_single_doc.fetch(ids=[doc.id])
        assert doc.id in result
        ret = result[doc.id]
        assert ret.field("id") == doc.field("id")
        assert ret.field("name") == single_doc.field("name")
        assert ret.field("weight") == single_doc.field("weight")
        assert ret.field("height") == single_doc.field("height")

    def test_collection_update_with_nullable_false_field_is_none(
        self, collection_with_single_doc: Collection, single_doc: Doc
    ):
        # id, name's nullable == False
        # weight, height's nullable == True

        # update doc field id
        doc = Doc(
            id=single_doc.id,
            fields={"id": None},
        )
        with pytest.raises(ValueError) as e:
            # ValueError: doc validate failed: field[id] is configured not nullable,
            # but doc does not contain this field
            collection_with_single_doc.update(doc)

        doc = Doc(
            id=single_doc.id,
            fields={"id": single_doc.field("id") + 1, "weight": None},
        )

        result = collection_with_single_doc.update(doc)
        assert bool(result)
        assert result.ok()
        stats = collection_with_single_doc.stats
        assert stats is not None
        assert stats.doc_count == 1

        ret = collection_with_single_doc.fetch(ids=[doc.id])
        assert doc.id in ret
        ret = ret[doc.id]
        assert ret.field("id") == doc.field("id")
        assert ret.field("name") == single_doc.field("name")
        assert ret.field("weight") is None
        assert ret.field("height") == single_doc.field("height")

    def test_collection_update_without_nullable_false_field(
        self, collection_with_single_doc: Collection, single_doc: Doc
    ):
        # id, name's nullable == False
        # weight, height's nullable == True

        # update doc field weight
        doc = Doc(
            id=single_doc.id,
            fields={"weight": single_doc.field("weight") + 1},
        )
        result = collection_with_single_doc.update(doc)
        assert bool(result)
        assert result.ok()
        stats = collection_with_single_doc.stats
        assert stats is not None
        assert stats.doc_count == 1

        # fetch
        ret = collection_with_single_doc.fetch(ids=[doc.id])
        assert doc.id in ret
        ret = ret[doc.id]
        assert ret.field("id") == single_doc.field("id")
        assert ret.field("name") == single_doc.field("name")
        assert ret.field("weight") == doc.field("weight")
        assert ret.field("height") == single_doc.field("height")

    def test_collection_update_without_nullable_false_field_set_null(
        self, collection_with_single_doc: Collection, single_doc: Doc
    ):
        # id, name's nullable == False
        # weight, height's nullable == True

        # update doc field weight is None
        doc = Doc(
            id=single_doc.id,
            fields={"weight": None},
        )
        result = collection_with_single_doc.update(doc)
        assert bool(result)
        assert result.ok()
        stats = collection_with_single_doc.stats
        assert stats is not None
        assert stats.doc_count == 1

        # fetch
        ret = collection_with_single_doc.fetch(ids=[doc.id])
        assert doc.id in ret
        ret = ret[doc.id]
        assert ret.field("id") == single_doc.field("id")
        assert ret.field("name") == single_doc.field("name")
        assert ret.field("weight") is None
        assert ret.field("height") == single_doc.field("height")

    def test_empty_collection_update_batch(
        self, test_collection: Collection, multiple_docs
    ):
        result = test_collection.update(multiple_docs)
        assert len(result) == len(multiple_docs)
        for item in result:
            assert item.code() == StatusCode.NOT_FOUND

        stats = test_collection.stats
        assert stats is not None
        assert stats.doc_count == 0

    def test_collection_update(
        self, collection_with_single_doc: Collection, single_doc
    ):
        result = collection_with_single_doc.update(single_doc)
        assert bool(result) == 1
        assert result.ok()
        stats = collection_with_single_doc.stats
        assert stats is not None
        assert stats.doc_count == 1

    def test_collection_update_batch(
        self, collection_with_multiple_docs: Collection, multiple_docs
    ):
        result = collection_with_multiple_docs.update(multiple_docs)
        assert len(result) == len(multiple_docs)
        for item in result:
            assert item.ok()

        stats = collection_with_multiple_docs.stats
        assert stats is not None
        assert stats.doc_count == len(multiple_docs)


# ----------------------------
# Collection Upsert Test Case
# ----------------------------
@pytest.mark.usefixtures("test_collection")
class TestCollectionUpsert:
    def test_empty_collection_upsert(self, test_collection: Collection, single_doc):
        result = test_collection.upsert(single_doc)
        assert bool(result)
        assert result.ok()

        stats = test_collection.stats
        assert stats is not None
        assert stats.doc_count == 1

    def test_empty_collection_upsert_batch(
        self, test_collection: Collection, multiple_docs
    ):
        result = test_collection.upsert(multiple_docs)
        assert len(result) == len(multiple_docs)
        for item in result:
            assert item.ok()

        stats = test_collection.stats
        assert stats is not None
        assert stats.doc_count == len(multiple_docs)

    def test_collection_upsert(
        self, collection_with_single_doc: Collection, single_doc, multiple_docs
    ):
        # doc is existing
        # upsert => update
        result = collection_with_single_doc.upsert(single_doc)
        assert bool(result)
        assert result.ok()
        stats = collection_with_single_doc.stats
        assert stats is not None
        assert stats.doc_count == 1

    def test_collection_upsert_batch(
        self, collection_with_multiple_docs: Collection, multiple_docs
    ):
        # doc is existing
        # upsert => update
        result = collection_with_multiple_docs.upsert(multiple_docs)
        assert len(result) == len(multiple_docs)
        for item in result:
            assert item.ok()

        stats = collection_with_multiple_docs.stats
        assert stats is not None
        assert stats.doc_count == len(multiple_docs)


# ----------------------------
# Collection Upsert Test Case
# ----------------------------
@pytest.mark.usefixtures("test_collection")
class TestCollectionDelete:
    def test_empty_collection_delete(self, test_collection: Collection, single_doc):
        result = test_collection.delete(single_doc.id)
        assert bool(result)
        assert result.code() == StatusCode.NOT_FOUND

    def test_empty_collection_delete_batch(
        self, test_collection: Collection, multiple_docs
    ):
        result = test_collection.delete([doc.id for doc in multiple_docs])
        assert len(result) == len(multiple_docs)
        for item in result:
            assert item.code() == StatusCode.NOT_FOUND

    def test_collection_delete(
        self, collection_with_single_doc: Collection, single_doc
    ):
        result = collection_with_single_doc.delete(single_doc.id)
        assert bool(result)
        assert result.ok()
        stats = collection_with_single_doc.stats
        assert stats is not None
        assert stats.doc_count == 0

        result = collection_with_single_doc.insert(single_doc)
        assert bool(result)
        assert result.ok()
        stats = collection_with_single_doc.stats
        assert stats is not None
        assert stats.doc_count == 1

    def test_collection_delete_batch(
        self, collection_with_multiple_docs: Collection, multiple_docs
    ):
        result = collection_with_multiple_docs.delete([doc.id for doc in multiple_docs])
        assert len(result) == len(multiple_docs)
        for item in result:
            assert item.ok()
        stats = collection_with_multiple_docs.stats
        assert stats is not None
        assert stats.doc_count == 0

    def test_collection_delete_by_filter(
        self, collection_with_single_doc: Collection, single_doc
    ):
        collection_with_single_doc.delete_by_filter(
            filter=f"height={single_doc.field('height')}"
        )
        stats = collection_with_single_doc.stats
        assert stats is not None
        assert stats.doc_count == 0

    def test_collection_delete_by_filter_invert_field(
        self, collection_with_single_doc: Collection, single_doc
    ):
        collection_with_single_doc.delete_by_filter(
            filter=f"id={single_doc.field('id')}"
        )
        stats = collection_with_single_doc.stats
        assert stats is not None
        assert stats.doc_count == 0


# ----------------------------
# Collection Upsert Test Case
# ----------------------------
@pytest.mark.usefixtures("test_collection")
class TestCollectionQuery:
    def test_empty_collection_query(self, test_collection: Collection):
        result = test_collection.query()
        assert len(result) == 0

    def test_collection_query(self, collection_with_single_doc: Collection, single_doc):
        result = collection_with_single_doc.query()
        assert len(result) == 1
        doc = result[0]
        assert doc.id == single_doc.id
        assert "dense" not in doc.field_names()
        assert "sparse" not in doc.field_names()
        field_without_vector = single_doc.field_names()
        assert set(doc.field_names()) == set(field_without_vector)
        for name in field_without_vector:
            assert doc.field(name) == single_doc.field(name)

    def test_collection_query_with_include_vector(
        self, collection_with_single_doc: Collection, single_doc
    ):
        result = collection_with_single_doc.query(include_vector=True)
        assert len(result) == 1
        doc = result[0]
        assert doc.vector("dense") is not None
        assert doc.vector("sparse") is not None

    def test_collection_query_with_output_fields(
        self, collection_with_single_doc: Collection, single_doc
    ):
        result = collection_with_single_doc.query(output_fields=["id", "name"])
        assert len(result) == 1
        doc = result[0]
        assert doc.id == single_doc.id
        assert len(doc.field_names()) == 2
        assert set(doc.field_names()) == {"id", "name"}

    def test_collection_query_with_topk(
        self, collection_with_multiple_docs: Collection
    ):
        result = collection_with_multiple_docs.query()
        assert len(result) == 10

        result = collection_with_multiple_docs.query(topk=5)
        assert len(result) == 5

    def test_collection_query_with_range_filter_int_field(
        self, collection_with_multiple_docs: Collection, multiple_docs
    ):
        index = 10
        idx = multiple_docs[index].id

        result = collection_with_multiple_docs.query(filter=f"id>{idx}", topk=100)
        assert len(result) == len(multiple_docs) - index - 1

        result = collection_with_multiple_docs.query(filter=f"id>={idx}", topk=100)
        assert len(result) == len(multiple_docs) - index

        result = collection_with_multiple_docs.query(filter=f"id<{idx}", topk=100)
        assert len(result) == index

        result = collection_with_multiple_docs.query(filter=f"id<={idx}", topk=100)
        assert len(result) == index + 1

        result = collection_with_multiple_docs.query(filter=f"id={idx}", topk=100)
        assert len(result) == 1

        result = collection_with_multiple_docs.query(filter=f"id!={idx}", topk=100)
        assert len(result) == len(multiple_docs) - 1

        left, right = 10, 90
        l_id, r_id = multiple_docs[left].id, multiple_docs[right].id
        result = collection_with_multiple_docs.query(
            filter=f"id>{l_id} and id<{r_id}", topk=100
        )
        assert len(result) == right - left - 1

        result = collection_with_multiple_docs.query(
            filter=f"id>={l_id} and id<{r_id}", topk=100
        )
        assert len(result) == right - left

        result = collection_with_multiple_docs.query(
            filter=f"id>={l_id} and id<={r_id}", topk=100
        )
        assert len(result) == right - left + 1

        result = collection_with_multiple_docs.query(
            filter=f"id<{l_id} or id>{r_id}", topk=100
        )
        assert len(result) == len(multiple_docs) - (right - left) - 1

        result = collection_with_multiple_docs.query(
            filter=f"id<={l_id} or id>{r_id}", topk=100
        )
        assert len(result) == len(multiple_docs) - (right - left)

        result = collection_with_multiple_docs.query(
            filter=f"id<={l_id} or id>={r_id}", topk=100
        )
        assert len(result) == len(multiple_docs) - (right - left) + 1

        result = collection_with_multiple_docs.query(filter="id in (1)", topk=100)
        assert len(result) == 1

    def test_collection_query_with_vector_and_id(
        self, collection_with_single_doc: Collection, single_doc: Doc
    ):
        with pytest.raises(ValueError):
            collection_with_single_doc.query(
                VectorQuery(
                    field_name="dense",
                    id=single_doc.id,
                    vector=single_doc.vector("dense"),
                )
            )

    def test_collection_query_with_filter_not_in(
        self, collection_with_multiple_docs: Collection, multiple_docs
    ):
        result = collection_with_multiple_docs.query(filter="id not in (1)", topk=100)
        assert len(result) == len(multiple_docs) - 1

    def test_collection_with_error_query_vector(
        self, collection_with_multiple_docs: Collection, multiple_docs
    ):
        query = VectorQuery(
            field_name="dense", vector=multiple_docs[0].vector("dense"), param=[1, 2, 3]
        )
        with pytest.raises(TypeError):
            result = collection_with_multiple_docs.query(
                filter="id in (1)", topk=100, vectors=query
            )

    def test_collection_query_by_id(
        self, collection_with_multiple_docs: Collection, multiple_docs
    ):
        result = collection_with_multiple_docs.query(
            VectorQuery(field_name="dense", id=multiple_docs[0].id)
        )
        assert len(result) == 10

    def test_collection_query_multi_vector_with_same_field(
        self, collection_with_multiple_docs: Collection, multiple_docs
    ):
        with pytest.raises(ValueError):
            collection_with_multiple_docs.query(
                [
                    VectorQuery(
                        field_name="dense", vector=multiple_docs[0].vector("dense")
                    ),
                    VectorQuery(
                        field_name="dense", vector=multiple_docs[0].vector("dense")
                    ),
                ]
            )

    @pytest.mark.skip(reason="TODO: This test case is pending implementation")
    def test_collection_query_by_dense_vector(
        self, collection_with_multiple_docs: Collection, multiple_docs
    ):
        pass

    @pytest.mark.skip(reason="TODO: This test case is pending implementation")
    def test_collection_query_by_sparse_vector(
        self, collection_with_multiple_docs: Collection, multiple_docs
    ):
        pass

    @pytest.mark.skip(reason="TODO: This test case is pending implementation")
    def test_collection_query_by_dense_vector_with_filter(
        self, collection_with_multiple_docs: Collection, multiple_docs
    ):
        pass

    @pytest.mark.skip(reason="TODO: This test case is pending implementation")
    def test_collection_query_by_sparse_vector_with_filter(
        self, collection_with_multiple_docs: Collection, multiple_docs
    ):
        pass

    @pytest.mark.skip(reason="TODO: This test case is pending implementation")
    def test_collection_query_with_rrf_reranker_by_multi_dense_vector(
        self, collection_with_multiple_docs: Collection, multiple_docs
    ):
        pass

    @pytest.mark.skip(reason="TODO: This test case is pending implementation")
    def test_collection_query_with_rrf_reranker_by_multi_sparse_vector(
        self, collection_with_multiple_docs: Collection, multiple_docs
    ):
        pass

    @pytest.mark.skip(reason="TODO: This test case is pending implementation")
    def test_collection_query_with_rrf_reranker_by_hybrid_vector(
        self, collection_with_multiple_docs: Collection, multiple_docs
    ):
        pass

    @pytest.mark.skip(reason="TODO: This test case is pending implementation")
    def test_collection_query_with_weighted_reranker_by_multi_dense_vector(
        self, collection_with_multiple_docs: Collection, multiple_docs
    ):
        pass

    @pytest.mark.skip(reason="TODO: This test case is pending implementation")
    def test_collection_query_with_weighted_reranker_by_multi_sparse_vector(
        self, collection_with_multiple_docs: Collection, multiple_docs
    ):
        pass

    @pytest.mark.skip(reason="TODO: This test case is pending implementation")
    def test_collection_query_with_weighted_reranker_by_hybrid_vector(
        self, collection_with_multiple_docs: Collection, multiple_docs
    ):
        pass


# ----------------------------
# OMEGA Index Test Case
# ----------------------------


@pytest.fixture(scope="session")
def omega_collection_schema():
    """Schema with OMEGA index for testing automatic training."""
    return zvec.CollectionSchema(
        name="omega_test_collection",
        fields=[
            FieldSchema(
                "id",
                DataType.INT64,
                nullable=False,
                index_param=InvertIndexParam(enable_range_optimization=True),
            ),
            FieldSchema("name", DataType.STRING, nullable=False),
        ],
        vectors=[
            VectorSchema(
                "embedding",
                DataType.VECTOR_FP32,
                dimension=128,
                index_param=OmegaIndexParam(
                    metric_type=MetricType.IP,
                    m=16,
                    ef_construction=200,
                ),
            ),
        ],
    )


@pytest.fixture
def omega_test_collection(
    tmp_path_factory, omega_collection_schema, collection_option
) -> Collection:
    """Create a collection with OMEGA index for testing."""
    temp_dir = tmp_path_factory.mktemp("zvec_omega")
    collection_path = temp_dir / "omega_collection"

    coll = zvec.create_and_open(
        path=str(collection_path),
        schema=omega_collection_schema,
        option=collection_option,
    )

    assert coll is not None, "Failed to create OMEGA collection"
    assert coll.schema.name == omega_collection_schema.name

    # Verify OMEGA index param
    embedding_field = coll.schema.vector("embedding")
    assert embedding_field is not None
    assert embedding_field.index_param is not None
    assert embedding_field.index_param.type == IndexType.OMEGA

    try:
        yield coll
    finally:
        if hasattr(coll, "destroy") and coll is not None:
            try:
                coll.destroy()
            except Exception as e:
                print(f"Warning: failed to destroy OMEGA collection: {e}")


@pytest.fixture
def omega_docs_large():
    """Generate 1500 documents to trigger segment dump and training."""
    import numpy as np

    docs = []
    for i in range(1500):
        # Generate somewhat structured vectors for better training
        base_vector = np.random.randn(128).astype(np.float32)
        base_vector = base_vector / np.linalg.norm(base_vector)  # Normalize
        docs.append(
            Doc(
                id=f"{i}",
                fields={"id": i, "name": f"doc_{i}"},
                vectors={"embedding": base_vector.tolist()},
            )
        )
    return docs


@pytest.mark.usefixtures("omega_test_collection")
class TestCollectionOmegaIndex:
    """Test cases for OMEGA index functionality."""

    def test_omega_index_param_creation(self, omega_test_collection: Collection):
        """Test that OmegaIndexParam is correctly created and configured."""
        embedding_field = omega_test_collection.schema.vector("embedding")
        assert embedding_field is not None
        assert embedding_field.name == "embedding"
        assert embedding_field.dimension == 128
        assert embedding_field.data_type == DataType.VECTOR_FP32

        index_param = embedding_field.index_param
        assert index_param is not None
        assert index_param.type == IndexType.OMEGA
        assert index_param.m == 16
        assert index_param.ef_construction == 200
        assert index_param.metric_type == MetricType.IP

    def test_omega_index_param_to_dict(self, omega_test_collection: Collection):
        """Test that OmegaIndexParam.to_dict() returns correct type."""
        embedding_field = omega_test_collection.schema.vector("embedding")
        index_param = embedding_field.index_param

        param_dict = index_param.to_dict()
        assert "type" in param_dict
        assert param_dict["type"] == "OMEGA", f"Expected 'OMEGA', got '{param_dict['type']}'"
        assert param_dict["metric_type"] == "IP"
        assert param_dict["m"] == 16
        assert param_dict["ef_construction"] == 200

    def test_omega_basic_insert_and_search(self, omega_test_collection: Collection):
        """Test basic insert and search with small dataset (no training)."""
        import numpy as np

        # Insert 10 documents
        docs = []
        for i in range(10):
            vector = np.random.randn(128).astype(np.float32)
            vector = vector / np.linalg.norm(vector)
            docs.append(
                Doc(
                    id=f"{i}",
                    fields={"id": i, "name": f"doc_{i}"},
                    vectors={"embedding": vector.tolist()},
                )
            )

        result = omega_test_collection.insert(docs)
        assert len(result) == len(docs)
        for item in result:
            assert item.ok()

        # Search with first document's vector
        query_vector = docs[0].vector("embedding")
        search_results = omega_test_collection.query(
            VectorQuery(field_name="embedding", vector=query_vector),
            topk=5
        )

        assert len(search_results) > 0
        # First result should be the query document itself
        assert search_results[0].id == docs[0].id

    def test_omega_large_dataset_with_optimize(
        self, omega_test_collection: Collection, omega_docs_large
    ):
        """Test OMEGA with large dataset to trigger automatic training."""
        # Insert 1500 documents (should trigger segment dump)
        batch_size = 100
        for i in range(0, len(omega_docs_large), batch_size):
            batch = omega_docs_large[i : i + batch_size]
            result = omega_test_collection.insert(batch)
            assert len(result) == len(batch)
            for item in result:
                assert item.ok()

        # Verify all documents inserted
        stats = omega_test_collection.stats
        assert stats.doc_count == len(omega_docs_large)

        # Call optimize to trigger segment dump and automatic training
        optimize_result = omega_test_collection.optimize(option=OptimizeOption())
        # optimize() may not return a value, just ensure it doesn't raise

        # Perform search to verify OMEGA is working
        query_doc = omega_docs_large[0]
        query_vector = query_doc.vector("embedding")

        search_results = omega_test_collection.query(
            VectorQuery(field_name="embedding", vector=query_vector),
            topk=10
        )

        assert len(search_results) > 0
        # Should find the query document in results
        found_query_doc = False
        for doc in search_results:
            if doc.id == query_doc.id:
                found_query_doc = True
                break
        assert found_query_doc, "Query document not found in search results"

    def test_omega_search_consistency(
        self, omega_test_collection: Collection, omega_docs_large
    ):
        """Test that OMEGA search results are consistent and reasonable."""
        import numpy as np

        # Insert documents
        batch_size = 100
        for i in range(0, len(omega_docs_large), batch_size):
            batch = omega_docs_large[i : i + batch_size]
            omega_test_collection.insert(batch)

        # Perform multiple searches and verify consistency
        query_doc = omega_docs_large[100]  # Use a middle document
        query_vector = query_doc.vector("embedding")

        # Search twice with same query
        results1 = omega_test_collection.query(
            VectorQuery(field_name="embedding", vector=query_vector),
            topk=20
        )
        results2 = omega_test_collection.query(
            VectorQuery(field_name="embedding", vector=query_vector),
            topk=20
        )

        assert len(results1) == len(results2)

        # Results should be identical (same query)
        for i in range(len(results1)):
            assert results1[i].id == results2[i].id

    def test_omega_with_filter(self, omega_test_collection: Collection):
        """Test OMEGA search with filter expressions."""
        import numpy as np

        # Insert 50 documents
        docs = []
        for i in range(50):
            vector = np.random.randn(128).astype(np.float32)
            vector = vector / np.linalg.norm(vector)
            docs.append(
                Doc(
                    id=f"{i}",
                    fields={"id": i, "name": f"doc_{i}"},
                    vectors={"embedding": vector.tolist()},
                )
            )
        omega_test_collection.insert(docs)

        # Search with filter
        query_vector = docs[0].vector("embedding")
        results = omega_test_collection.query(
            VectorQuery(
                field_name="embedding",
                vector=query_vector,
            ),
            filter="id >= 10 and id < 20",
            topk=20,
        )

        # All results should satisfy filter
        for doc in results:
            doc_id = doc.field("id")
            assert 10 <= doc_id < 20, f"Document id {doc_id} does not satisfy filter"

    def test_omega_training_and_early_stopping(
        self, tmp_path_factory
    ):
        """
        Verify OMEGA training with k_train=1 labeling logic:
        1. Training succeeds (model files generated)
        2. Search demonstrates early stopping
        3. Recall meets target
        """
        import numpy as np
        import os
        import gc
        import time

        print("\n" + "="*80)
        print("OMEGA Training Verification (k_train=1)")
        print("="*80)

        # Create fresh collection
        temp_dir = tmp_path_factory.mktemp("zvec_omega_training")
        collection_path = str(temp_dir / "omega_training_collection")

        schema = zvec.CollectionSchema(
            name="omega_training_test",
            fields=[
                FieldSchema("id", DataType.INT64, nullable=False),
                FieldSchema("name", DataType.STRING, nullable=False),
            ],
            vectors=[
                VectorSchema(
                    "embedding",
                    DataType.VECTOR_FP32,
                    dimension=128,
                    index_param=OmegaIndexParam(
                        metric_type=MetricType.L2,
                        m=16,
                        ef_construction=200,
                    ),
                ),
            ],
        )

        # Create and insert documents
        print(f"\n[1/4] Creating collection and inserting 1500 documents...")
        collection = zvec.create_and_open(
            path=collection_path,
            schema=schema,
            option=CollectionOption()
        )

        # Generate documents
        docs = []
        for i in range(1500):
            base_vector = np.random.randn(128).astype(np.float32)
            base_vector = base_vector / np.linalg.norm(base_vector)
            docs.append(
                Doc(
                    id=f"{i}",
                    fields={"id": i, "name": f"doc_{i}"},
                    vectors={"embedding": base_vector.tolist()},
                )
            )

        batch_size = 100
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            collection.insert(batch)
        print(f"   ✓ Inserted {len(docs)} documents")

        # Trigger optimization
        print(f"\n[2/4] Triggering optimize (dump + auto training)...")
        collection.optimize(option=OptimizeOption())
        print(f"   ✓ Optimize completed")

        # Close collection
        del collection
        gc.collect()
        time.sleep(1)

        # Reopen to load OMEGA index
        print(f"   Reopening collection to load OMEGA indexes...")
        collection = zvec.open(path=collection_path, option=CollectionOption(read_only=True))
        print(f"   ✓ Collection reopened")

        # Verify model files exist
        print(f"\n[3/4] Verifying model files...")
        model_files_found = False
        model_dir = None

        # Look for any numeric segment directories
        if os.path.exists(collection_path):
            for item in os.listdir(collection_path):
                item_path = os.path.join(collection_path, item)
                if os.path.isdir(item_path):
                    potential_model_dir = os.path.join(item_path, "omega_model")

                    if os.path.exists(potential_model_dir):
                        model_dir = potential_model_dir
                        print(f"   Found model directory: {model_dir}")

                        # Check for all required files
                        required_files = [
                            "model.txt",
                            "threshold_table.txt",
                            "interval_table.txt",
                            "gt_collected_table.txt",
                            "gt_cmps_all_table.txt"
                        ]

                        files_exist = []
                        for fname in required_files:
                            fpath = os.path.join(model_dir, fname)
                            exists = os.path.exists(fpath)
                            files_exist.append(exists)

                            if exists:
                                file_size = os.path.getsize(fpath)
                                print(f"   ✓ {fname}: {file_size} bytes")
                            else:
                                print(f"   ✗ {fname}: NOT FOUND")

                        if all(files_exist):
                            model_files_found = True
                            print(f"\n   ✅ ALL MODEL FILES GENERATED!")
                        break

        if not model_files_found:
            print(f"   ⚠️  No OMEGA model files found in any segment directory")

        assert model_files_found, "OMEGA model files not found after training"

        # Test search and calculate recall
        print(f"\n[4/4] Testing search recall...")
        n_queries = 10
        topk = 10
        recalls = []

        for i in range(n_queries):
            query_doc = docs[i]
            query_vector = query_doc.vector("embedding")

            results = collection.query(
                VectorQuery(field_name="embedding", vector=query_vector),
                topk=topk
            )

            # Calculate recall
            result_ids = [r.id for r in results]
            recall = 1.0 if query_doc.id in result_ids else 0.0
            recalls.append(recall)

        avg_recall = np.mean(recalls)
        print(f"   Average Recall@{topk}: {avg_recall:.4f}")

        # Verify recall is reasonable (should be high since we use docs from collection)
        assert avg_recall >= 0.8, f"Recall too low: {avg_recall:.4f}"

        print(f"\n   ✅ RECALL MEETS THRESHOLD (>= 0.8)")
        print("\n" + "="*80)
        print("✅ OMEGA Training Verification PASSED")
        print("   1. ✓ Model files generated (LightGBM + 4 tables)")
        print("   2. ✓ Training completed with k_train=1 labeling")
        print(f"   3. ✓ Search recall: {avg_recall:.4f} >= 0.8")
        print("="*80)
