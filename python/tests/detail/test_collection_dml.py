import logging
import pytest


from zvec import (
    CollectionOption,
    InvertIndexParam,
    HnswIndexParam,
    FieldSchema,
    VectorSchema,
    CollectionSchema,
    Collection,
    Doc,
    VectorQuery,
    StatusCode,
)
from distance_helper import *
from fixture_helper import *
from doc_helper import *

Maximum = 1024

DOCID_VALID_LIST = [
    "1valid_Id",
    "123.45",
    "123abc",
    "-!@#$%+=.123abc_+",
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ123456789012",
]
DOCID_INVALID_LIST = [
    None,
    "",
    "()qsd123",
    " ",
    "/&AS12",
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890121",
]

FIELD_VALUE_VALID_LIST = [
    (
        "bool_field",
        [
            None,
            True,
            False,
        ],
    ),
    (
        "float_field",
        [
            None,
            0.0,
            -1.0,
            1.0,
            3.4028235e38,
            -3.4028235e38,
            1.17549435e-38,
            -1.17549435e-38,
            float("inf"),
            float("-inf"),
        ],
    ),
    (
        "double_field",
        [
            None,
            0.0,
            -1.0,
            1.0,
            1.7976931348623157e308,
            -1.7976931348623157e308,
            2.2250738585072014e-308,
            -2.2250738585072014e-308,
            float("inf"),
            float("-inf"),
        ],
    ),
    (
        "int32_field",
        [
            None,
            0,
            1,
            -1,
            2147483647,
            -2147483648,
        ],
    ),
    (
        "int64_field",
        [
            None,
            0,
            1,
            -1,
            9223372036854775807,
            -9223372036854775808,
        ],
    ),
    (
        "uint32_field",
        [
            None,
            0,
            1,
            4294967295,
        ],
    ),
    (
        "uint64_field",
        [
            None,
            0,
            1,
            18446744073709551615,
        ],
    ),
    (
        "string_field",
        [
            None,
            "",
            "a",
            "test_name",
            "这是一个中文名称测试",
            "a" * 1000,
        ],
    ),
    (
        "array_bool_field",
        [
            None,
            [],
            [True],
            [False, True],
            [True, False, True, False] * 10,
        ],
    ),
    (
        "array_float_field",
        [
            None,
            [],
            [0.0],
            [1.0, 2.0, 3.0],
            [3.4028235e38, -3.4028235e38],
        ],
    ),
    (
        "array_double_field",
        [
            None,
            [],
            [0.0],
            [1.0, 2.0, 3.0],
            [1.7976931348623157e308, -1.7976931348623157e308],
        ],
    ),
    (
        "array_int32_field",
        [
            None,
            [],
            [0],
            [1, 2, 3],
            [2147483647, -2147483648],
        ],
    ),
    (
        "array_int64_field",
        [
            None,
            [],
            [0],
            [1, 2, 3],
            [9223372036854775807, -9223372036854775808],
        ],
    ),
    (
        "array_uint32_field",
        [
            None,
            [],
            [0],
            [1, 2, 3],
            [4294967295],
        ],
    ),
    (
        "array_uint64_field",
        [
            None,
            [],
            [0],
            [1, 2, 3],
            [18446744073709551615],
        ],
    ),
    (
        "array_string_field",
        [
            None,
            [],
            [""],
            ["a", "b", "c"],
            ["test_string", "测试字符串"],
            ["a" * 100] * 5,
        ],
    ),
]
FIELD_VALUE_INVALID_LIST = [
    (
        "bool_field",
        [
            "True",
            "False",
            "",
        ],
    ),
    ("float_field", ["invalid", [1.0], {"value": 1.0}]),
    ("double_field", ["invalid", [1.0], {"value": 1.0}]),
    (
        "int32_field",
        [
            "invalid",
            [1],
            {"value": 1},
            2147483648,
            -2147483649,
        ],
    ),
    (
        "int64_field",
        [
            "invalid",
            [1],
            {"value": 1},
            9223372036854775808,
            -9223372036854775809,
        ],
    ),
    (
        "uint32_field",
        [
            "invalid",
            [1],
            {"value": 1},
            4294967296,
            -1,
        ],
    ),
    (
        "uint64_field",
        [
            "invalid",
            [1],
            {"value": 1},
            18446744073709551616,
            -1,
        ],
    ),
    (
        "string_field",
        [
            123,
            12.34,
            True,
            ["array"],
            {"key": "value"},
        ],
    ),
    (
        "array_bool_field",
        [
            True,
            False,
            [True, "invalid"],
            {"key": True},
        ],
    ),
    (
        "array_float_field",
        [
            [1.0, "invalid"],
            [1.0, None],
            "invalid",
            [1.0, [2.0]],
            1.0,
        ],
    ),
    (
        "array_double_field",
        [
            [1.0, "invalid"],
            [1.0, None],
            "invalid",
            [1.0, [2.0]],
            1.0,
        ],
    ),
    (
        "array_int32_field",
        [
            [1, "invalid"],
            [1, None],
            "invalid",
            [1, [2]],
            1,
        ],
    ),
    (
        "array_int64_field",
        [
            [1, "invalid"],
            [1, None],
            "invalid",
            [1, [2]],
            1,
        ],
    ),
    (
        "array_uint32_field",
        [
            [1, "invalid"],
            [1, None],
            [1, -1],
            "invalid",
            [1, [2]],
            1,
        ],
    ),
    (
        "array_uint64_field",
        [
            [1, "invalid"],
            [1, None],
            [1, -1],
            "invalid",
            [1, [2]],
            1,
        ],
    ),
    (
        "array_string_field",
        [
            ["valid", 123],
            ["valid", None],
            "invalid",
            [["nested"]],
            123,
        ],
    ),
]

VECTOR_VALUE_VALID_LIST = [
    (
        "vector_fp32_field",
        [
            [0.0] * 128,
            [1.0] * 128,
            [-1.0] * 128,
            [float("inf")] * 128,
            [float("-inf")] * 128,
            [i / 128.0 for i in range(128)],
            [-i / 128.0 for i in range(128)],
        ],
    ),
    (
        "vector_fp16_field",
        [
            [0.0] * 128,
            [1.0] * 128,
            [-1.0] * 128,
            [float("inf")] * 128,
            [float("-inf")] * 128,
            [i / 128.0 for i in range(128)],
            [-i / 128.0 for i in range(128)],
        ],
    ),
    ("vector_int8_field", [[100] * 128, [0] * 128, [-100] * 128]),
    (
        "sparse_vector_fp32_field",
        [
            {0: 1.0},
            {0: 0.0, 1: 1.0, 2: -1.0},
            {0: float("inf"), 1: float("-inf")},
            {i: float(i) for i in range(10)},
            {128: 1.0, 256: -1.0, 512: 0.5},
        ],
    ),
    (
        "sparse_vector_fp16_field",
        [
            {0: 1.0},
            {0: 0.0, 1: 1.0, 2: -1.0},
            {0: float("inf"), 1: float("-inf")},
            {i: float(i) for i in range(10)},
            {128: 1.0, 256: -1.0, 512: 0.5},
        ],
    ),
]
VECTOR_VALUE_INVALID_LIST = [
    (
        "vector_fp32_field",
        [
            None,
            [],
            [0.0] * 127,
            [0.0] * 129,
            [0.0] * 1000,
            ["invalid"],
            [0, 1, 2],
            [None] * 128,
        ],
    ),
    (
        "vector_fp16_field",
        [
            None,
            [],
            [0.0] * 127,
            [0.0] * 129,
            [0.0] * 1000,
            ["invalid"],
            [0, 1, 2],
            [None] * 128,
        ],
    ),
    (
        "vector_int8_field",
        [
            None,
            [],
            [1] * 127,
            [10] * 129,
            [0] * 1000,
            ["invalid"],
            [0, 1, 2],
            [None] * 128,
        ],
    ),
    (
        "sparse_vector_fp32_field",
        [
            None,
            "invalid",
            {None: 1.0},
            {"0": 1.0},
            {0: "invalid"},
            {0: None},
            {-1: 1.0},
        ],
    ),
    (
        "sparse_vector_fp16_field",
        [
            None,
            "invalid",
            {None: 1.0},
            {"0": 1.0},
            {0: "invalid"},
            {0: None},
            {-1: 1.0},
        ],
    ),
]

UPDATE_PARTIAL_VALUE = [
    (
        "partial_fields",
        {"string_field": "partially_updated_test", "float_field": 95.5},
        {},
    ),
    ("dense_vector_only", {}, {"vector_fp32_field": [0.3] * 128}),
    ("dense_vector_only", {}, {"vector_fp16_field": [0.6] * 128}),
    ("dense_vector_only", {}, {"vector_int8_field": [3] * 128}),
    ("sparse_vector_only", {}, {"sparse_vector_fp32_field": {1: 2.0, 2: 3.0, 4: 4.0}}),
    (
        "sparse_vector_only",
        {},
        {"sparse_vector_fp16_field": {10: 2.1, 20: 3.1, 40: 4.1}},
    ),
    (
        "fields_and_vectors",
        {"string_field": "fully_updated_test", "bool_field": False},
        {
            "vector_fp32_field": [0.4] * 128,
            "sparse_vector_fp32_field": {1: 3.0, 3: 5.0},
        },
    ),
]


# ==================== helper ====================
def singledoc_and_check(
    collection: Collection, insert_doc, operator="insert", is_delete=1
):
    if operator == "insert":
        result = collection.insert(insert_doc)
    elif operator == "upsert":
        result = collection.upsert(insert_doc)
    elif operator == "update":
        result = collection.update(insert_doc)
    else:
        logging.error("operator value is error!")

    assert bool(result)
    assert result.ok()

    stats = collection.stats
    assert stats is not None
    assert stats.doc_count == 1

    fetched_docs = collection.fetch([insert_doc.id])
    assert len(fetched_docs) == 1
    assert insert_doc.id in fetched_docs

    fetched_doc = fetched_docs[insert_doc.id]

    assert is_doc_equal(fetched_doc, insert_doc, collection.schema)
    assert hasattr(fetched_doc, "score"), "Document should have a score attribute"
    assert fetched_doc.score == 0.0, (
        "Fetch operation should return default score of 0.0"
    )

    for k, v in DEFAULT_VECTOR_FIELD_NAME.items():
        if v != {}:
            query_result = collection.query(
                VectorQuery(field_name=v, vector=insert_doc.vectors[v]),
                topk=10,
            )
            assert len(query_result) > 0, (
                f"Expected at least 1 query result, but got {len(query_result)}"
            )

            found_doc = None
            for doc in query_result:
                if doc.id == insert_doc.id:
                    found_doc = doc
                    break
            assert found_doc is not None, (
                f"Inserted document {insert_doc.id} not found in query results"
            )
            assert is_doc_equal(found_doc, insert_doc, collection.schema, True, False)
    if is_delete == 1:
        collection.delete(insert_doc.id)
        assert collection.stats.doc_count == 0, "Document should be deleted"


def updatedoc_partial_check(
    collection, update_doc_partial, update_doc_full, operator="update", is_delete=1
):
    if operator == "upsert":
        result = collection.upsert(update_doc_partial)
    elif operator == "update":
        result = collection.update(update_doc_partial)
    else:
        logging.error("operator value is error!")

    assert bool(result)
    assert result.ok()

    stats = collection.stats
    assert stats is not None
    assert stats.doc_count == 1

    fetched_docs = collection.fetch([update_doc_partial.id])
    assert len(fetched_docs) == 1, (
        f"fetched_docs={fetched_docs},Expected 1 fetched document, but got {len(fetched_docs)}"
    )
    assert update_doc_partial.id in fetched_docs, (
        f"Expected document ID {update_doc_partial.id} in fetched documents"
    )

    fetched_doc = fetched_docs[update_doc_partial.id]
    assert is_doc_equal(fetched_doc, update_doc_full, collection.schema)
    assert hasattr(fetched_doc, "score"), "Document should have a score attribute"
    assert fetched_doc.score == 0.0, (
        "Fetch operation should return default score of 0.0"
    )

    for k, v in DEFAULT_VECTOR_FIELD_NAME.items():
        if v != {}:
            query_result = collection.query(
                VectorQuery(field_name=v, vector=update_doc_full.vectors[v]),
                topk=10,
            )
            assert len(query_result) > 0, (
                f"Expected at least 1 query result, but got {len(query_result)}"
            )

            found_doc = None
            for doc in query_result:
                if doc.id == update_doc_partial.id:
                    found_doc = doc
                    break
            assert found_doc is not None, (
                f"Inserted document {update_doc_partial.id} not found in query results"
            )
            assert is_doc_equal(
                found_doc, update_doc_full, collection.schema, True, False
            )
    if is_delete == 1:
        collection.delete(update_doc_partial.id)
        assert collection.stats.doc_count == 0, "Document should be deleted"


def batchdoc_and_check(collection, multiple_docs, doc_num, operator="insert"):
    if operator == "insert":
        result = collection.insert(multiple_docs)
    elif operator == "upsert":
        result = collection.upsert(multiple_docs)

    elif operator == "update":
        result = collection.update(multiple_docs)
    else:
        logging.error("operator value is error!")

    assert len(result) == len(multiple_docs)
    for item in result:
        assert item.ok(), (
            f"result={result},Insert operation failed with code {item.code()}"
        )

    stats = collection.stats
    assert stats is not None, "Collection stats should not be None"
    assert stats.doc_count == len(multiple_docs), (
        f"Document count should be {len(multiple_docs)} after insert, but got {stats.doc_count}"
    )

    doc_ids = [doc.id for doc in multiple_docs]
    fetched_docs = collection.fetch(doc_ids)
    assert len(fetched_docs) == len(multiple_docs), (
        f"fetched_docs={fetched_docs},Expected {len(multiple_docs)} fetched documents, but got {len(fetched_docs)}"
    )

    for original_doc in multiple_docs:
        assert original_doc.id in fetched_docs, (
            f"Expected document ID {original_doc.id} in fetched documents"
        )
        fetched_doc = fetched_docs[original_doc.id]

        assert is_doc_equal(fetched_doc, original_doc, collection.schema)

        assert hasattr(fetched_doc, "score"), "Document should have a score attribute"
        assert fetched_doc.score == 0.0, (
            "Fetch operation should return default score of 0.0"
        )

    first_doc = multiple_docs[doc_num - 1]
    for k, v in DEFAULT_VECTOR_FIELD_NAME.items():
        query_result = collection.query(
            VectorQuery(field_name=v, vector=first_doc.vectors[v]),
            topk=1024,
        )
        assert len(query_result) > 0, (
            f"Expected at least 1 query result, but got {len(query_result)}"
        )

        found_doc = None

        for doc in query_result:
            if doc.id == first_doc.id:
                found_doc = doc
                break
        assert found_doc is not None, (
            f"Inserted document {first_doc.id} not found in query results"
        )

        assert is_doc_equal(found_doc, first_doc, collection.schema, True, False)


# ==================== Tests ====================
# ----------------------------
# Collection Insert Test Case
# ----------------------------


class TestCollectionInsert:
    def test_insert(self, full_collection: Collection):
        single_doc = generate_doc(1, full_collection.schema)
        singledoc_and_check(full_collection, single_doc)

    @pytest.mark.parametrize("doc_num", [1, 5, Maximum])
    def test_insert_batch(self, full_collection: Collection, doc_num):
        multiple_docs = [
            generate_doc(i, full_collection.schema) for i in range(doc_num)
        ]
        batchdoc_and_check(full_collection, multiple_docs, doc_num)

    def test_insert_duplicate(self, full_collection: Collection):
        insert_doc = generate_doc(1, full_collection.schema)

        result = full_collection.insert(insert_doc)
        assert result.code().value == 0
        assert result.ok()

        # Verify documents were inserted
        stats = full_collection.stats
        assert stats is not None
        assert stats.doc_count == 1

        insert_doc_duplicate = full_collection.insert(insert_doc)
        assert bool(insert_doc_duplicate)
        assert insert_doc_duplicate.code() == StatusCode.ALREADY_EXISTS, (
            f"Second insert operation should fail with ALREADY_EXISTS, but got code {insert_doc_duplicate.code()}"
        )

        stats = full_collection.stats
        assert stats is not None, "Collection stats should not be None"
        assert stats.doc_count == 1, (
            f"Document count should still be 1 after failed insert, but got {stats.doc_count}"
        )

    @pytest.mark.parametrize("doc_id", DOCID_VALID_LIST)
    def test_insert_docid_valid(self, full_collection: Collection, doc_id):
        insert_doc = generate_doc_random(doc_id, full_collection.schema)
        singledoc_and_check(full_collection, insert_doc)

    @pytest.mark.parametrize("doc_id", DOCID_INVALID_LIST)
    def test_insert_docid_invalid(self, full_collection: Collection, doc_id):
        insert_doc = generate_doc_random(doc_id, full_collection.schema)

        with pytest.raises(Exception) as exc_info:
            full_collection.insert(insert_doc)

        assert exc_info.value is not None
        stats = full_collection.stats
        assert stats is not None
        assert stats.doc_count == 0

    @pytest.mark.parametrize("field_name, field_values", FIELD_VALUE_VALID_LIST)
    @pytest.mark.parametrize(
        "full_schema_new",
        [(True, True, HnswIndexParam()), (False, True, HnswIndexParam())],
        indirect=True,
    )
    def test_insert_fields_valid(
        self, full_collection_new: Collection, field_name: str, field_values, request
    ):
        for i, field_value in enumerate(field_values):
            doc_id = str(field_value) if field_name == "id" else str(i)
            doc_fields, doc_vectors = generate_vectordict_random(
                full_collection_new.schema
            )
            full_schema_params = request.getfixturevalue("full_schema_new")
            target_field = None
            for field in full_schema_params.fields:
                if field.name == field_name:
                    target_field = field
                    break
            doc_fields[field_name] = field_value
            insert_doc = Doc(id=doc_id, fields=doc_fields, vectors=doc_vectors)
            if target_field and not target_field.nullable and field_value is None:
                with pytest.raises(Exception) as exc_info:
                    full_collection_new.insert(insert_doc)
                assert exc_info.value is not None
            else:
                singledoc_and_check(full_collection_new, insert_doc)

    @pytest.mark.parametrize("field_name, field_values", FIELD_VALUE_INVALID_LIST)
    def test_insert_fields_invalid(
        self, full_collection: Collection, field_name: str, field_values
    ):
        for i, field_value in enumerate(field_values):
            doc_id = str(field_value) if field_name == "id" else str(i)
            doc_fields, doc_vectors = generate_vectordict_random(full_collection.schema)
            doc_fields[field_name] = field_value
            insert_doc = Doc(id=doc_id, fields=doc_fields, vectors=doc_vectors)

            with pytest.raises(Exception) as exc_info:
                full_collection.insert(insert_doc)
            assert exc_info.value is not None
            stats = full_collection.stats
            assert stats is not None
            assert stats.doc_count == 0

    @pytest.mark.parametrize("vector_field, vector_values", VECTOR_VALUE_VALID_LIST)
    def test_insert_vector_valid(
        self, full_collection: Collection, vector_field: str, vector_values
    ):
        for i, vector_value in enumerate(vector_values):
            doc_fields, doc_vectors = generate_vectordict_random(full_collection.schema)

            doc_vectors[vector_field] = vector_value

            insert_doc = Doc(id=str(i), fields=doc_fields, vectors=doc_vectors)

            singledoc_and_check(full_collection, insert_doc)

    @pytest.mark.parametrize("vector_field, vector_values", VECTOR_VALUE_INVALID_LIST)
    def test_insert_vector_invalid(
        self, full_collection: Collection, vector_field: str, vector_values
    ):
        for i, vector_value in enumerate(vector_values):
            doc_fields, doc_vectors = generate_vectordict_random(full_collection.schema)
            doc_vectors[vector_field] = vector_value
            insert_doc = Doc(id=str(i), fields=doc_fields, vectors=doc_vectors)
            with pytest.raises(Exception) as exc_info:
                full_collection.insert(insert_doc)

            assert exc_info.value is not None
            stats = full_collection.stats
            assert stats is not None
            assert stats.doc_count == 0


class TestCollectionUpdate:
    def test_update(self, full_collection: Collection):
        insert_doc = generate_doc(1, full_collection.schema)
        singledoc_and_check(full_collection, insert_doc, is_delete=0)
        updated_doc = generate_update_doc(1, full_collection.schema)
        singledoc_and_check(full_collection, updated_doc, operator="update")

    @pytest.mark.parametrize("doc_num", [1, 5, Maximum])
    def test_update_batch(self, full_collection: Collection, doc_num):
        multiple_docs = [
            generate_doc(i, full_collection.schema) for i in range(doc_num)
        ]
        batchdoc_and_check(full_collection, multiple_docs, doc_num)
        multiple_update_docs = [
            generate_update_doc(i, full_collection.schema) for i in range(doc_num)
        ]
        batchdoc_and_check(
            full_collection, multiple_update_docs, doc_num, operator="update"
        )

    def test_empty_collection_update(self, full_collection: Collection):
        updated_doc = generate_update_doc(1, full_collection.schema)
        result = full_collection.update(updated_doc)
        assert bool(result), f"Expected 1 result, but got {len(result)}"
        assert result.code() == StatusCode.NOT_FOUND, (
            f"Update operation should fail with NOT_FOUND, but got code {result.code()}"
        )
        fetched_docs = full_collection.fetch([updated_doc.id])
        assert len(fetched_docs) == 0

        stats = full_collection.stats
        assert stats is not None, "Collection stats should not be None"
        assert stats.doc_count == 0, (
            f"Document count should be 0, but got {stats.doc_count}"
        )

    @pytest.mark.parametrize("doc_num", [1, 5, Maximum])
    def test_empty_collection_update_batch(self, full_collection: Collection, doc_num):
        multiple_update_docs = [
            generate_update_doc(i, full_collection.schema) for i in range(doc_num)
        ]
        result = full_collection.update(multiple_update_docs)
        assert len(result) == len(multiple_update_docs), (
            f"Expected {len(multiple_update_docs)} results, but got {len(result)}"
        )
        for item in result:
            assert item.code() == StatusCode.NOT_FOUND, (
                f"Update operation should fail with NOT_FOUND, but got code {item.code()}"
            )

        stats = full_collection.stats
        assert stats is not None, "Collection stats should not be None"
        assert stats.doc_count == 0, (
            f"Document count should be 0, but got {stats.doc_count}"
        )

        doc_ids = [doc.id for doc in multiple_update_docs]
        fetched_docs = full_collection.fetch(doc_ids)
        assert len(fetched_docs) == 0

    @pytest.mark.parametrize("field_name, field_values", FIELD_VALUE_VALID_LIST)
    @pytest.mark.parametrize(
        "full_schema_new",
        [(True, True, HnswIndexParam()), (False, True, HnswIndexParam())],
        indirect=True,
    )
    def test_update_fields_valid(
        self, full_collection_new: Collection, field_name: str, field_values, request
    ):
        for i, field_value in enumerate(field_values):
            insert_doc = generate_doc(i, full_collection_new.schema)
            singledoc_and_check(full_collection_new, insert_doc, is_delete=0)
            update_doc_fields, update_doc_vectors = generate_vectordict_random(
                full_collection_new.schema
            )
            full_schema_params = request.getfixturevalue("full_schema_new")
            target_field = None
            for field in full_schema_params.fields:
                if field.name == field_name:
                    target_field = field
                    break
            update_doc_fields[field_name] = field_value
            update_doc = Doc(
                id=str(i), fields=update_doc_fields, vectors=update_doc_vectors
            )
            if target_field and not target_field.nullable and field_value is None:
                with pytest.raises(Exception) as exc_info:
                    update_doc_fields[field_name] = field_value
                    full_collection_new.update(update_doc)
                assert exc_info.value is not None
                full_collection_new.delete(insert_doc.id)
            else:
                singledoc_and_check(
                    full_collection_new, update_doc, operator="update", is_delete=1
                )

    @pytest.mark.parametrize("field_name, field_values", FIELD_VALUE_INVALID_LIST)
    def test_update_fields_invalid(
        self, full_collection: Collection, field_name: str, field_values
    ):
        for i, field_value in enumerate(field_values):
            insert_doc = generate_doc(i, full_collection.schema)
            singledoc_and_check(full_collection, insert_doc, is_delete=0)
            update_doc_fields, update_doc_vectors = generate_vectordict_random(
                full_collection.schema
            )
            update_doc_fields[field_name] = field_value
            update_doc = Doc(
                id=str(i), fields=update_doc_fields, vectors=update_doc_vectors
            )

            with pytest.raises(Exception) as exc_info:
                full_collection.update(update_doc)

            assert exc_info.value is not None
            full_collection.delete(insert_doc.id)
            stats = full_collection.stats
            assert stats is not None
            assert stats.doc_count == 0

    @pytest.mark.parametrize("vector_field, vector_values", VECTOR_VALUE_VALID_LIST)
    def test_update_doc_vector_valid(
        self,
        full_collection: Collection,
        collection_temp_dir,
        collection_option,
        vector_field: str,
        vector_values,
    ):
        for i, vector_value in enumerate(vector_values):
            insert_doc = generate_doc(i, full_collection.schema)
            singledoc_and_check(full_collection, insert_doc, is_delete=0)
            update_doc_fields, update_doc_vectors = generate_vectordict_random(
                full_collection.schema
            )
            update_doc_vectors[vector_field] = vector_value
            update_doc = Doc(
                id=str(i), fields=update_doc_fields, vectors=update_doc_vectors
            )
            singledoc_and_check(full_collection, update_doc, operator="update")

    @pytest.mark.parametrize("vector_field, vector_values", VECTOR_VALUE_INVALID_LIST)
    def test_update_doc_vector_invalid(
        self,
        full_collection: Collection,
        collection_temp_dir,
        collection_option,
        vector_field: str,
        vector_values,
    ):
        for i, vector_value in enumerate(vector_values):
            insert_doc = generate_doc(i, full_collection.schema)
            singledoc_and_check(full_collection, insert_doc, is_delete=0)
            update_doc_fields, update_doc_vectors = generate_vectordict_random(
                full_collection.schema
            )
            update_doc_vectors[vector_field] = vector_value
            update_doc = Doc(
                id=str(i), fields=update_doc_fields, vectors=update_doc_vectors
            )
            with pytest.raises(Exception) as exc_info:
                full_collection.update(update_doc)
            assert exc_info.value is not None
            full_collection.delete(insert_doc.id)
            stats = full_collection.stats
            assert stats is not None
            assert stats.doc_count == 0

    @pytest.mark.parametrize(
        "update_type, fields_to_update, vectors_to_update", UPDATE_PARTIAL_VALUE
    )
    def test_update_partial_fields(
        self,
        full_collection: Collection,
        collection_temp_dir,
        collection_option,
        update_type: str,
        fields_to_update: dict,
        vectors_to_update: dict,
        doc_id=1,
    ):
        insert_doc = generate_doc(doc_id, full_collection.schema)
        singledoc_and_check(full_collection, insert_doc, is_delete=0)

        update_doc_fields, update_doc_vectors = insert_doc.fields, insert_doc.vectors
        for k, v in fields_to_update.items():
            update_doc_fields[k] = v
        for k, v in vectors_to_update.items():
            update_doc_vectors[k] = v

        update_doc_full = Doc(
            id=str(doc_id), fields=update_doc_fields, vectors=update_doc_vectors
        )

        update_doc_partial = Doc(
            id=str(doc_id), fields=fields_to_update, vectors=vectors_to_update
        )

        updatedoc_partial_check(
            full_collection,
            update_doc_partial,
            update_doc_full,
            operator="update",
            is_delete=1,
        )


class TestCollectionUpsert:
    def test_new_doc_upsert(self, full_collection: Collection):
        single_doc = generate_doc(1, full_collection.schema)
        singledoc_and_check(full_collection, single_doc, operator="upsert", is_delete=1)

    @pytest.mark.parametrize("doc_num", [1, 5, Maximum])
    def test_new_doc_upsert_batch(self, full_collection: Collection, doc_num):
        multiple_docs = [
            generate_doc(i, full_collection.schema) for i in range(doc_num)
        ]
        batchdoc_and_check(full_collection, multiple_docs, doc_num, operator="upsert")

    def test_existing_doc_upsert(self, full_collection: Collection):
        insert_doc = generate_doc(1, full_collection.schema)
        singledoc_and_check(full_collection, insert_doc, is_delete=0)
        updated_doc = generate_update_doc(1, full_collection.schema)
        singledoc_and_check(full_collection, updated_doc, operator="upsert")

    @pytest.mark.parametrize("doc_id", DOCID_VALID_LIST)
    def test_upsert_docid_valid(self, full_collection: Collection, doc_id):
        upsert_doc = generate_doc_random(doc_id, full_collection.schema)
        singledoc_and_check(full_collection, upsert_doc, operator="upsert", is_delete=1)

    @pytest.mark.parametrize("doc_id", DOCID_INVALID_LIST)
    def test_upsert_docid_invalid(self, full_collection: Collection, doc_id):
        upsert_doc = generate_doc_random(doc_id, full_collection.schema)

        with pytest.raises(Exception) as exc_info:
            full_collection.upsert(upsert_doc)

        assert exc_info.value is not None

        stats = full_collection.stats
        assert stats is not None
        assert stats.doc_count == 0

    @pytest.mark.parametrize("field_name, field_values", FIELD_VALUE_VALID_LIST)
    @pytest.mark.parametrize(
        "full_schema_new",
        [(True, True, HnswIndexParam()), (False, True, HnswIndexParam())],
        indirect=True,
    )
    def test_upsert_fields_valid(
        self, full_collection_new: Collection, field_name: str, field_values, request
    ):
        for i, field_value in enumerate(field_values):
            doc_id = str(field_value) if field_name == "id" else str(i)
            doc_fields, doc_vectors = generate_vectordict_random(
                full_collection_new.schema
            )

            full_schema_params = request.getfixturevalue("full_schema_new")
            target_field = None
            for field in full_schema_params.fields:
                if field.name == field_name:
                    target_field = field
                    break
            doc_fields[field_name] = field_value
            upsert_doc = Doc(id=doc_id, fields=doc_fields, vectors=doc_vectors)
            if target_field and not target_field.nullable and field_value is None:
                with pytest.raises(Exception) as exc_info:
                    full_collection_new.upsert(upsert_doc)
                assert exc_info.value is not None
            else:
                singledoc_and_check(
                    full_collection_new, upsert_doc, operator="upsert", is_delete=1
                )

    @pytest.mark.parametrize("field_name, field_values", FIELD_VALUE_INVALID_LIST)
    def test_upsert_fields_invalid(
        self, full_collection: Collection, field_name: str, field_values
    ):
        for i, field_value in enumerate(field_values):
            doc_id = str(field_value) if field_name == "id" else str(i)
            doc_fields, doc_vectors = generate_vectordict_random(full_collection.schema)
            doc_fields[field_name] = field_value
            upsert_doc = Doc(id=doc_id, fields=doc_fields, vectors=doc_vectors)

            with pytest.raises(Exception) as exc_info:
                full_collection.upsert(upsert_doc)
            assert exc_info.value is not None
            stats = full_collection.stats
            assert stats is not None
            assert stats.doc_count == 0

    @pytest.mark.parametrize("vector_field, vector_values", VECTOR_VALUE_VALID_LIST)
    def test_upsert_vector_valid(
        self, full_collection: Collection, vector_field: str, vector_values
    ):
        for i, vector_value in enumerate(vector_values):
            doc_fields, doc_vectors = generate_vectordict_random(full_collection.schema)

            doc_vectors[vector_field] = vector_value

            upsert_doc = Doc(id=str(i), fields=doc_fields, vectors=doc_vectors)

            singledoc_and_check(
                full_collection, upsert_doc, operator="upsert", is_delete=1
            )

    @pytest.mark.parametrize("vector_field, vector_values", VECTOR_VALUE_INVALID_LIST)
    def test_upsert_vector_invalid(
        self, full_collection: Collection, vector_field: str, vector_values
    ):
        for i, vector_value in enumerate(vector_values):
            doc_fields, doc_vectors = generate_vectordict_random(full_collection.schema)
            doc_vectors[vector_field] = vector_value
            upsert_doc = Doc(id=str(i), fields=doc_fields, vectors=doc_vectors)
            with pytest.raises(Exception) as exc_info:
                full_collection.upsert(upsert_doc)

            assert exc_info.value is not None
            stats = full_collection.stats
            assert stats is not None
            assert stats.doc_count == 0


class TestCollectionDelete:
    @pytest.mark.parametrize("doc_num", [1, 5, Maximum])
    def test_delete_batch(self, full_collection: Collection, doc_num):
        multiple_docs = [
            generate_doc(i, full_collection.schema) for i in range(doc_num)
        ]
        batchdoc_and_check(full_collection, multiple_docs, doc_num, operator="insert")

        doc_ids = [doc.id for doc in multiple_docs]
        result = full_collection.delete(doc_ids)
        assert len(result) == len(doc_ids)
        for item in result:
            assert item.ok()

    def test_delete_non_exist(self, full_collection: Collection):
        result = full_collection.delete("non_existing_id")
        assert result.code().value == 1
        assert result.code() == StatusCode.NOT_FOUND

    @pytest.mark.parametrize("doc_num", [5])
    def test_delete_batch_part_non_exist(self, full_collection: Collection, doc_num):
        multiple_docs = [
            generate_doc(i, full_collection.schema) for i in range(doc_num)
        ]
        batchdoc_and_check(full_collection, multiple_docs, doc_num, operator="insert")
        doc_ids = [doc.id for doc in multiple_docs]
        doc_ids.extend([str(doc_num), str(doc_num + 1)])
        result = full_collection.delete(doc_ids)

        assert len(result) == len(doc_ids)
        for i in range(len(result)):
            if i < doc_num:
                assert result[i].ok()
            else:
                assert result[i].code().value == 1
                assert result[i].code() == StatusCode.NOT_FOUND

    @pytest.mark.parametrize("doc_num", [5])
    def test_delete_by_filter(self, full_collection: Collection, doc_num):
        multiple_docs = [
            generate_doc(i, full_collection.schema) for i in range(doc_num)
        ]
        batchdoc_and_check(full_collection, multiple_docs, doc_num, operator="insert")

        result = full_collection.delete_by_filter("int32_field > 0")
        assert result is None

    def test_delete_empty_ids(self, full_collection: Collection):
        result = full_collection.delete([])
        assert len(result) == 0
