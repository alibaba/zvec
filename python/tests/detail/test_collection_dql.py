# Copyright 2025-present the zvec project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from zvec.typing import DataType, StatusCode, MetricType, QuantizeType
from zvec.model import Collection, Doc, VectorQuery
from zvec.model.param import (
    CollectionOption,
    InvertIndexParam,
    HnswIndexParam,
    FlatIndexParam,
    IVFIndexParam,
    HnswQueryParam,
    IVFQueryParam,
)
from zvec.model.schema import FieldSchema, VectorSchema
from zvec.extension import RrfReRanker, WeightedReRanker, QwenReRanker
from distance_helper import *
from fixture_helper import *
from doc_helper import *
from params_helper import *


# ==================== helper ====================
def batchdoc_and_check(
    collection: Collection, multiple_docs, doc_num, operator="insert"
):
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
            include_vector=True,
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

        assert is_doc_equal(found_doc, first_doc, collection.schema)
        assert hasattr(found_doc, "score")
        assert isinstance(found_doc.score, (int, float))


def batchdoc_and_check_ivf(
    collection: Collection, multiple_docs, doc_num, operator="insert"
):
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
        if v in ["vector_fp16_field", "vector_fp32_field"]:
            query_result = collection.query(
                VectorQuery(field_name=v, vector=first_doc.vectors[v]),
                topk=1024,
                include_vector=True,
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

            assert is_doc_equal(found_doc, first_doc, collection.schema)
            assert hasattr(found_doc, "score")
            assert isinstance(found_doc.score, (int, float))


def single_querydoc_check(
    multiple_docs,
    query_result,
    full_collection: Collection,
    is_by_vector=0,
    query_vector=None,
    data_type=None,
    vector_name=None,
    metric_type=MetricType.IP,
    id_include_vector: bool = False,
    is_output_fields=0,
):
    for original_doc in multiple_docs:
        for doc in query_result:
            if doc.id == original_doc.id:
                found_doc = doc
                if is_output_fields == 0:
                    assert is_doc_equal(
                        found_doc,
                        original_doc,
                        full_collection.schema,
                        True,
                        id_include_vector,
                    )
                assert hasattr(found_doc, "score")
                # assert found_doc.score >= 0.0
                if not id_include_vector:
                    for k, v in DEFAULT_VECTOR_FIELD_NAME.items():
                        assert found_doc.vector(v) == {}
                else:
                    for k, v in DEFAULT_VECTOR_FIELD_NAME.items():
                        assert found_doc.vector(v) != {}
                if is_by_vector:
                    prev_score = float("inf")
                    for i, doc in enumerate(query_result):
                        doc_vector = full_collection.fetch(doc.id)[doc.id].vector(
                            vector_name
                        )
                        expected_score = distance(
                            query_vector, doc_vector, metric_type, data_type, k
                        )
                        if (
                            full_collection.schema.vector(vector_name).data_type
                            != DataType.VECTOR_FP16
                        ):
                            assert abs(doc.score - expected_score) < 0.001, (
                                f"{data_type} {vector_name} :Expected score {expected_score:.6f}, but got {doc.score:.6f} for document {doc.id}"
                            )
                        assert doc.score <= prev_score, (
                            f"{data_type} {vector_name} :Scores should be in descending order. Current: {doc.score}, Previous: {prev_score}"
                        )
                        prev_score = doc.score


def multi_querydoc_check(multiple_docs, query_result, full_collection):
    for original_doc in multiple_docs:
        for doc in query_result:
            if doc.id == original_doc.id:
                found_doc = doc
                assert is_doc_equal(
                    found_doc, original_doc, full_collection.schema, False, False
                )
                assert hasattr(found_doc, "score"), (
                    "Document should have a score attribute"
                )
                assert found_doc.score >= 0.0, (
                    "Fetch operation should return default score of 0.0"
                )
                for k, v in DEFAULT_VECTOR_FIELD_NAME.items():
                    assert found_doc.vector(v) == {}


# ==================== Tests ====================
class TestCollectionFetch:
    def test_fetch_non_existing(self, full_collection: Collection):
        result = full_collection.fetch(ids=["non_existing_id1", "non_existing_id2"])
        assert len(result) == 0

    @pytest.mark.parametrize("doc_num", [3])
    def test_fetch_partial_non_existing(self, full_collection: Collection, doc_num):
        multiple_docs = [
            generate_doc(i, full_collection.schema) for i in range(doc_num)
        ]
        batchdoc_and_check(full_collection, multiple_docs, doc_num, operator="insert")

        fetch_id_list = [doc.id for doc in multiple_docs]
        fetch_id_list.append("non_existing_id")
        result = full_collection.fetch(ids=fetch_id_list)

        assert len(result) == doc_num
        assert "non_existing_id" not in result.keys()

    def test_fetch_empty_ids(self, full_collection: Collection):
        result = full_collection.fetch(ids=[])
        assert len(result) == 0, (
            f"Expected 0 results for empty ID list, but got {len(result)}"
        )


class TestCollectionQuery:
    @pytest.mark.parametrize("doc_num", [5])
    def test_query_with_no_condition(self, full_collection: Collection, doc_num):
        multiple_docs = [
            generate_doc(i, full_collection.schema) for i in range(doc_num)
        ]
        batchdoc_and_check(full_collection, multiple_docs, doc_num, operator="insert")
        query_result = full_collection.query()
        assert len(query_result) == doc_num
        single_querydoc_check(multiple_docs, query_result, full_collection)

    @pytest.mark.parametrize("doc_num", [10])
    def test_query_with_filter_empty(self, full_collection: Collection, doc_num):
        multiple_docs = [
            generate_doc(i, full_collection.schema) for i in range(doc_num)
        ]
        batchdoc_and_check(full_collection, multiple_docs, doc_num, operator="insert")
        result1 = full_collection.query(filter="")
        assert len(result1) == doc_num
        single_querydoc_check(multiple_docs, result1, full_collection)
        result2 = full_collection.query(filter=None)
        assert len(result2) == doc_num
        single_querydoc_check(multiple_docs, result2, full_collection)
        ids1 = set(doc.id for doc in result1)
        ids2 = set(doc.id for doc in result2)
        assert ids1 == ids2

    @pytest.mark.parametrize("field_name", DEFAULT_SCALAR_FIELD_NAME.values())
    @pytest.mark.parametrize("doc_num", [10])
    def test_query_with_filter_single_condition(
        self, full_collection: Collection, doc_num, field_name
    ):
        multiple_docs = [
            generate_doc(i, full_collection.schema) for i in range(doc_num)
        ]
        batchdoc_and_check(full_collection, multiple_docs, doc_num, operator="insert")

        # Construct different filter conditions based on field type
        if field_name == "bool_field":
            filter = field_name + " = true"
            expected_doc_indices = [
                i
                for i in range(doc_num)
                if generate_doc(i, full_collection.schema).field(field_name) == True
            ]
        elif field_name in ["float_field", "double_field"]:
            filter = field_name + " > 5.0"
            expected_doc_indices = [
                i
                for i in range(doc_num)
                if generate_doc(i, full_collection.schema).field(field_name) > 5.0
            ]
        elif field_name.startswith("array_"):
            # For array types, we check that the array length is greater than 0 (indicating the array is not empty)
            filter = (
                "array_length(" + field_name + ") > 0"
            )  # Use array_length function to check array length
            expected_doc_indices = [
                i
                for i in range(doc_num)
                if len(generate_doc(i, full_collection.schema).field(field_name)) > 0
            ]
        elif field_name in ["string_field"]:
            filter = field_name + " != 'lcy'"
            expected_doc_indices = [
                i
                for i in range(doc_num)
                if generate_doc(i, full_collection.schema).field(field_name) != "lcy"
            ]
        elif field_name in [
            "int32_field",
            "int64_field",
            "uint32_field",
            "uint64_field",
        ]:  # Integer type
            filter = field_name + " > 5"
            expected_doc_indices = [i for i in range(6, doc_num)]
        else:
            raise ValueError(f"Unsupported field type for filtering: {field_name}")

        query_result = full_collection.query(filter=filter)

        assert len(query_result) == len(expected_doc_indices)

        returned_doc_ids = set()
        for doc in query_result:
            returned_doc_ids.add(doc.id)

        expected_doc_ids = set(str(i) for i in expected_doc_indices)

        for doc in query_result:
            assert doc.id in expected_doc_ids
            if field_name == "bool_field":
                assert doc.field(field_name) == True
            elif field_name in ["float_field", "double_field"]:
                assert doc.field(field_name) > 5.0
            elif field_name.startswith("array_"):
                # For array types, validate that the array length is greater than 0
                field_val = doc.field(field_name)
                assert len(field_val) > 0
            elif field_name in [
                "int32_field",
                "int64_field",
                "uint32_field",
                "uint64_field",
            ]:  # Integer type
                assert int(doc.field(field_name)) > 5
            elif field_name in ["string_field"]:
                assert doc.field(field_name) != "lcy"
            else:
                raise ValueError(f"Unsupported field type for validation: {field_name}")

        single_querydoc_check(multiple_docs, query_result, full_collection)

    @pytest.mark.parametrize("field_name", ["int32_field"])
    @pytest.mark.parametrize(
        "filter",
        [
            "int32_field > 3 and int32_field < 9",
            "int32_field >= 5 and int32_field <= 7",
        ],
    )
    @pytest.mark.parametrize("doc_num", [10])
    def test_query_with_filter_and(
        self, full_collection: Collection, doc_num, field_name, filter
    ):
        multiple_docs = [
            generate_doc(i, full_collection.schema) for i in range(doc_num)
        ]
        batchdoc_and_check(full_collection, multiple_docs, doc_num, operator="insert")
        filter = field_name + " > 3 and " + field_name + " < 9"
        query_result = full_collection.query(filter=filter)
        if filter == "int32_field > 3 and int32_field < 9":
            assert len(query_result) == doc_num - 4 - 1
            expected_doc_ids = set(str(i) for i in range(4, 9))

            for doc in query_result:
                assert doc.id in expected_doc_ids
                field_value = int(doc.field(field_name))
                assert field_value > 3 and field_value < 9
        else:
            assert len(query_result) == 3
            expected_doc_ids = set(str(i) for i in range(5, 8))

            for doc in query_result:
                assert doc.id in expected_doc_ids
                field_value = int(doc.field(field_name))
                assert field_value >= 5 and field_value <= 7

        single_querydoc_check(multiple_docs, query_result, full_collection)

    @pytest.mark.parametrize("field_name", ["int32_field"])
    @pytest.mark.parametrize(
        "filter",
        [
            "int32_field < 3 or int32_field > 8",
            "int32_field = 3 or int32_field = 7",
            "int32_field <= 3 or int32_field >= 8",
        ],
    )
    @pytest.mark.parametrize("doc_num", [10])
    def test_query_with_filter_or(
        self, full_collection: Collection, doc_num, field_name, filter
    ):
        multiple_docs = [
            generate_doc(i, full_collection.schema) for i in range(doc_num)
        ]
        batchdoc_and_check(full_collection, multiple_docs, doc_num, operator="insert")
        query_result = full_collection.query(filter=filter)
        if filter == "int32_field < 3 or int32_field > 8":
            assert len(query_result) == 4
            expected_doc_ids = set([str(0), str(1), str(2), str(9)])
            for doc in query_result:
                assert doc.id in expected_doc_ids
                field_value = int(doc.field(field_name))
                assert field_value < 3 or field_value > 8
        elif filter == "int32_field = 3 or int32_field = 7":
            assert len(query_result) == 2
            expected_doc_ids = set([str(3), str(7)])
            for doc in query_result:
                assert doc.id in expected_doc_ids
                field_value = int(doc.field(field_name))
                assert field_value == 3 or field_value == 7
        else:
            assert len(query_result) == 6
            expected_doc_ids = set([str(0), str(1), str(2), str(3), str(8), str(9)])
            for doc in query_result:
                assert doc.id in expected_doc_ids
                field_value = int(doc.field(field_name))
                assert field_value <= 3 or field_value >= 8

        single_querydoc_check(multiple_docs, query_result, full_collection)

    @pytest.mark.parametrize("field_names", [("int32_field", "bool_field")])
    @pytest.mark.parametrize(
        "filter",
        [
            "(int32_field < 3 or int32_field > 8) and bool_field = false",
            "(int32_field > 2 and int32_field < 5) or (int32_field > 7 and bool_field = true)",
        ],
    )
    @pytest.mark.parametrize("doc_num", [10])
    def test_query_with_filter_parentheses(
        self, full_collection: Collection, doc_num, field_names, filter
    ):
        multiple_docs = [
            generate_doc(i, full_collection.schema) for i in range(doc_num)
        ]
        batchdoc_and_check(full_collection, multiple_docs, doc_num, operator="insert")
        query_result = full_collection.query(filter=filter)
        if filter == "(int32_field < 3 or int32_field > 8) and bool_field = false":
            assert len(query_result) == 2
            expected_doc_ids = set([str(1), str(9)])
            for doc in query_result:
                assert doc.id in expected_doc_ids
                assert (
                    int(doc.field(field_names[0])) < 3
                    or int(doc.field(field_names[0])) > 8
                ) and doc.field(field_names[1]) == False
        else:
            assert len(query_result) == 3
            expected_doc_ids = set([str(3), str(4), str(8)])
            for doc in query_result:
                assert doc.id in expected_doc_ids
                assert (
                    (
                        int(doc.field(field_names[0])) > 2
                        and int(doc.field(field_names[0])) < 5
                    )
                    or (doc.field(field_names[0])) > 7
                    and doc.field(field_names[1]) == True
                )
        single_querydoc_check(multiple_docs, query_result, full_collection)

    @pytest.mark.parametrize(
        "filter",
        [
            # Test combinations with different scalar field types using AND, OR, parentheses
            "(int32_field > 2 AND int32_field < 8) OR bool_field = true",
            "(float_field > 3.0 AND float_field < 7.0) AND string_field != 'exclude'",
            "(double_field >= 1.5 OR double_field <= 8.5) AND uint32_field > 2",
            "bool_field = false OR (int64_field > 3 AND int64_field < 9)",
            "(string_field = 'special' OR string_field = 'test') AND int32_field > 1",
            "(array_length(array_int32_field) > 0 AND int32_field > 5) OR bool_field = true",
            "uint64_field > 1 AND uint64_field < 9 AND (float_field > 2.0 OR double_field < 9.0)",
            "(bool_field = true OR string_field = 'special') AND (int32_field > 4 OR int64_field < 6)",
            # More complex combinations covering more field types
            "((int32_field > 1 AND int32_field < 5) OR (int64_field > 6 AND int64_field < 9)) AND bool_field = true",
            "(uint32_field > 2 OR uint64_field < 8) AND (float_field > 1.0 AND double_field < 10.0)",
            "(string_field != 'skip' AND int32_field >= 2) OR (bool_field = false AND double_field <= 7.5)",
            # Additional combinations with array_length and other supported operations on array fields
            "(array_length(array_string_field) > 0 OR int32_field > 5) AND bool_field = true",
            "(array_length(array_int32_field) >= 1 AND float_field > 2.0) OR (array_length(array_float_field) > 0)",
            "(array_length(array_bool_field) > 0 OR array_length(array_double_field) > 0) AND string_field != ''",
            # Additional combinations with other supported scalar operations using range comparisons
            "(int32_field > 1 AND int32_field < 10 OR string_field != 'exclude1') AND bool_field = true",
            "(float_field > 1.0 AND float_field < 10.0 AND double_field > 0.5) OR (uint32_field > 5 AND uint32_field < 50)",
            "(int64_field > 50 OR uint64_field < 1000) AND string_field != ''",
        ],
    )
    @pytest.mark.parametrize(
        "doc_num", [20]
    )  # Increased doc number for better coverage
    def test_query_with_filter_complex_combinations(
        self, full_collection: Collection, doc_num, filter
    ):
        multiple_docs = [
            generate_doc(i, full_collection.schema) for i in range(doc_num)
        ]
        batchdoc_and_check(full_collection, multiple_docs, doc_num, operator="insert")
        query_result = full_collection.query(filter=filter)
        assert query_result is not None

        for doc in query_result:
            assert hasattr(doc, "id")
            assert doc.id in [d.id for d in multiple_docs]
        single_querydoc_check(multiple_docs, query_result, full_collection)

    @pytest.mark.parametrize(
        "filter",
        [
            "int32_field >",
            "int32_field = 'string'",
            "nonexistent_field = 5",
            "int32_field > 5 and",
            "int32_field > > 5",
        ],
    )
    @pytest.mark.parametrize("doc_num", [10])
    def test_query_filter_invalid(self, full_collection: Collection, doc_num, filter):
        multiple_docs = [
            generate_doc(i, full_collection.schema) for i in range(doc_num)
        ]
        batchdoc_and_check(full_collection, multiple_docs, doc_num, operator="insert")
        with pytest.raises(Exception) as exc_info:
            full_collection.query(filter=filter)
        if filter in ["int32_field = 'string'", "nonexistent_field = 5"]:
            assert "Analyze sql info failed" in str(exc_info.value)
        else:
            assert "Invalid filter" in str(exc_info.value)

    @pytest.mark.parametrize("field_name", ["int32_field"])
    @pytest.mark.parametrize("topk_value", [1, 5, 10, 50, 100, 500, 1000, 1024])
    def test_query_with_filter_topk_valid(
        self, full_collection: Collection, topk_value: int, field_name
    ):
        multiple_docs = [
            generate_doc(i, full_collection.schema) for i in range(topk_value)
        ]
        batchdoc_and_check(
            full_collection, multiple_docs, topk_value, operator="insert"
        )
        filter = (
            field_name + f" >={topk_value - 1} and " + field_name + f" <={topk_value}"
        )
        print("filter:\n")
        print(filter)
        query_result = full_collection.query(filter=filter, topk=topk_value)
        assert len(query_result) == 1
        expected_doc_ids = [str(topk_value - 1)]

        for doc in query_result:
            assert doc.id in expected_doc_ids
            field_value = int(doc.field(field_name))
            assert field_value >= topk_value - 1 and field_value <= topk_value
        single_querydoc_check(multiple_docs, query_result, full_collection)

    @pytest.mark.parametrize("field_name", ["int32_field"])
    @pytest.mark.parametrize("topk_value", [1, 5, 10, 50, 100, 500, 1000, 1024])
    def test_query_without_filter_topk_valid(
        self, full_collection: Collection, topk_value: int, field_name
    ):
        multiple_docs = [
            generate_doc(i, full_collection.schema) for i in range(topk_value)
        ]
        batchdoc_and_check(
            full_collection, multiple_docs, topk_value, operator="insert"
        )

        query_result = full_collection.query(topk=topk_value)
        assert len(query_result) == topk_value
        single_querydoc_check(multiple_docs, query_result, full_collection)

    @pytest.mark.parametrize("doc_num", [10])
    def test_query_with_include_vector(self, full_collection: Collection, doc_num):
        multiple_docs = [
            generate_doc(i, full_collection.schema) for i in range(doc_num)
        ]
        batchdoc_and_check(full_collection, multiple_docs, doc_num, operator="insert")

        query_result = full_collection.query(include_vector=True)
        assert len(query_result) > 0
        single_querydoc_check(
            multiple_docs, query_result, full_collection, id_include_vector=1
        )

    @pytest.mark.parametrize("output_fields", [["int32_field", "int64_field"]])
    @pytest.mark.parametrize("doc_num", [10])
    def test_query_with_output_fields(
        self, full_collection: Collection, doc_num, output_fields
    ):
        multiple_docs = [
            generate_doc(i, full_collection.schema) for i in range(doc_num)
        ]
        batchdoc_and_check(full_collection, multiple_docs, doc_num, operator="insert")
        query_result = full_collection.query(output_fields=output_fields)
        assert len(query_result) > 0
        for doc in query_result:
            field_names = doc.field_names()
            assert field_names == output_fields

    @pytest.mark.parametrize(
        "filter",
        [
            "int32_field >= 10 and int32_field <= 20",
            "int32_field = 3 and int32_field = 8",
        ],
    )
    @pytest.mark.parametrize("doc_num", [10])
    def test_query_empty_result(self, full_collection: Collection, doc_num, filter):
        multiple_docs = [
            generate_doc(i, full_collection.schema) for i in range(doc_num)
        ]
        batchdoc_and_check(full_collection, multiple_docs, doc_num, operator="insert")
        result = full_collection.query(filter=filter)
        assert len(result) == 0

    @pytest.mark.parametrize(
        "full_schema_new",
        [(True, True, HnswIndexParam()), (False, True, FlatIndexParam())],
        indirect=True,
    )
    @pytest.mark.parametrize("doc_num", [10])
    def test_query_by_id(
        self, full_collection_new: Collection, doc_num, full_schema_new
    ):
        multiple_docs = [
            generate_doc(i, full_collection_new.schema) for i in range(doc_num)
        ]
        batchdoc_and_check(
            full_collection_new, multiple_docs, doc_num, operator="insert"
        )
        for k, v in DEFAULT_VECTOR_FIELD_NAME.items():
            query_result = full_collection_new.query(VectorQuery(field_name=v, id="1"))
            assert len(query_result) > 0
            query_doc = full_collection_new.fetch(ids=["1"])
            query_vector = query_doc["1"].vector(v)
            single_querydoc_check(
                multiple_docs,
                query_result,
                full_collection_new,
                is_by_vector=1,
                query_vector=query_vector,
                data_type=k,
                vector_name=v,
            )

    @pytest.mark.parametrize("doc_num", [10])
    def test_query_by_id_ivf(self, full_collection_ivf: Collection, doc_num):
        multiple_docs = [
            generate_doc(i, full_collection_ivf.schema) for i in range(doc_num)
        ]
        batchdoc_and_check_ivf(
            full_collection_ivf, multiple_docs, doc_num, operator="insert"
        )
        for k, v in DEFAULT_VECTOR_FIELD_NAME.items():
            if v in ["vector_fp16_field", "vector_fp32_field"]:
                query_result = full_collection_ivf.query(
                    VectorQuery(field_name=v, id="1")
                )
                assert len(query_result) > 0
                query_doc = full_collection_ivf.fetch(ids=["1"])
                query_vector = query_doc["1"].vector(v)
                single_querydoc_check(
                    multiple_docs,
                    query_result,
                    full_collection_ivf,
                    is_by_vector=1,
                    query_vector=query_vector,
                    data_type=k,
                    vector_name=v,
                )

    @pytest.mark.parametrize(
        "full_schema_new",
        [(True, True, HnswIndexParam()), (False, True, FlatIndexParam())],
        indirect=True,
    )
    @pytest.mark.parametrize("doc_num", [10])
    @pytest.mark.parametrize("topk", [None, 1024])
    @pytest.mark.parametrize("filter", [None, "int32_field >= 3 and int32_field <= 7"])
    def test_query_by_vector(
        self, full_collection_new: Collection, doc_num, full_schema_new, topk, filter
    ):
        multiple_docs = [
            generate_doc(i, full_collection_new.schema) for i in range(doc_num)
        ]
        batchdoc_and_check(
            full_collection_new, multiple_docs, doc_num, operator="insert"
        )
        for k, v in DEFAULT_VECTOR_FIELD_NAME.items():
            doc_fields, doc_vectors = generate_vectordict_random(
                full_collection_new.schema
            )
            query_vector = doc_vectors[v]
            if topk and filter:
                query_result = full_collection_new.query(
                    filter=filter,
                    vectors=VectorQuery(field_name=v, vector=query_vector),
                    topk=topk,
                )
            elif topk and not filter:
                query_result = full_collection_new.query(
                    VectorQuery(field_name=v, vector=query_vector), topk=topk
                )
            elif not topk and filter:
                query_result = full_collection_new.query(
                    filter=filter,
                    vectors=VectorQuery(field_name=v, vector=query_vector),
                )
            else:
                query_result = full_collection_new.query(
                    VectorQuery(field_name=v, vector=query_vector)
                )
            assert len(query_result) > 0, (
                f"Expected at least 1 query result, but got {len(query_result)}"
            )
            single_querydoc_check(
                multiple_docs,
                query_result,
                full_collection_new,
                is_by_vector=1,
                query_vector=query_vector,
                data_type=k,
                vector_name=v,
            )

    @pytest.mark.parametrize("doc_num", [10])
    def test_query_by_vector_ivf(self, full_collection_ivf: Collection, doc_num):
        multiple_docs = [
            generate_doc(i, full_collection_ivf.schema) for i in range(doc_num)
        ]
        batchdoc_and_check_ivf(
            full_collection_ivf, multiple_docs, doc_num, operator="insert"
        )
        for k, v in DEFAULT_VECTOR_FIELD_NAME.items():
            if v in ["vector_fp16_field", "vector_fp32_field"]:
                doc_fields, doc_vectors = generate_vectordict_random(
                    full_collection_ivf.schema
                )
                query_vector = doc_vectors[v]
                query_result = full_collection_ivf.query(
                    VectorQuery(field_name=v, vector=query_vector),
                    topk=1024,
                )
                assert len(query_result) > 0, (
                    f"Expected at least 1 query result, but got {len(query_result)}"
                )
                single_querydoc_check(
                    multiple_docs,
                    query_result,
                    full_collection_ivf,
                    is_by_vector=1,
                    query_vector=query_vector,
                    data_type=k,
                    vector_name=v,
                )

    @pytest.mark.parametrize("doc_num", [10])
    def test_query_multivector_rrf(self, full_collection: Collection, doc_num):
        multiple_docs = [
            generate_doc(i, full_collection.schema) for i in range(doc_num)
        ]
        batchdoc_and_check(full_collection, multiple_docs, doc_num, operator="insert")

        doc_fields, doc_vectors = generate_vectordict_random(full_collection.schema)
        single_query_results = {}
        for k, v in DEFAULT_VECTOR_FIELD_NAME.items():
            single_query_results[v] = full_collection.query(
                VectorQuery(field_name=v, vector=doc_vectors[v])
            )
        expected_rrf_scores = calculate_multi_vector_rrf_scores(single_query_results)
        multi_query_vectors = []
        for k, v in DEFAULT_VECTOR_FIELD_NAME.items():
            multi_query_vectors.append(VectorQuery(field_name=v, vector=doc_vectors[v]))

        rrf_reranker = RrfReRanker(topn=3)
        multi_query_result = full_collection.query(
            vectors=multi_query_vectors,
            reranker=rrf_reranker,
        )
        assert len(multi_query_result) > 0, (
            f"Expected at least 1 result, but got {len(multi_query_result)}"
        )

        multi_querydoc_check(multiple_docs, multi_query_result, full_collection)

        prev_score = float("inf")
        for i, doc in enumerate(multi_query_result):
            doc_id = doc.id
            assert doc_id in expected_rrf_scores, (
                f"Document {doc_id} should be in expected RRF scores"
            )
            expected_score = expected_rrf_scores[doc_id]
            actual_score = doc.score
            assert abs(actual_score - expected_score) < 1e-10, (
                f"RRF score mismatch for document {doc_id}: expected {expected_score}, got {actual_score}"
            )
            assert doc.score <= prev_score, (
                f"Scores should be in descending order. Current: {doc.score}, Previous: {prev_score}"
            )
            prev_score = doc.score

    @pytest.mark.parametrize(
        "weights",
        [
            {
                "vector_fp32_field": 0.3,
                "vector_fp16_field": 0.2,
                "vector_int8_field": 0.3,
                "sparse_vector_fp32_field": 0.1,
                "sparse_vector_fp16_field": 0.1,
            }
        ],
    )
    @pytest.mark.parametrize(
        "metric_type", [MetricType.L2, MetricType.IP, MetricType.COSINE]
    )
    @pytest.mark.parametrize("doc_num", [10])
    def test_query_multivector_weighted(
        self, full_collection: Collection, doc_num, weights, metric_type
    ):
        multiple_docs = [
            generate_doc(i, full_collection.schema) for i in range(doc_num)
        ]
        batchdoc_and_check(full_collection, multiple_docs, doc_num, operator="insert")
        doc_fields, doc_vectors = generate_vectordict_random(full_collection.schema)

        weighted_reranker = WeightedReRanker(
            topn=3, weights=weights, metric=MetricType.IP
        )

        single_query_results = {}
        for k, v in DEFAULT_VECTOR_FIELD_NAME.items():
            single_query_results[v] = full_collection.query(
                VectorQuery(field_name=v, vector=doc_vectors[v])
            )
        expected_weighted_scores = calculate_multi_vector_weighted_scores(
            single_query_results, weights, MetricType.IP
        )

        multi_query_vectors = []
        for k, v in DEFAULT_VECTOR_FIELD_NAME.items():
            multi_query_vectors.append(VectorQuery(field_name=v, vector=doc_vectors[v]))

        multi_query_result = full_collection.query(
            vectors=multi_query_vectors,
            reranker=weighted_reranker,
        )
        assert len(multi_query_result) > 0, (
            f"Expected at least 1 result, but got {len(multi_query_result)}"
        )

        multi_querydoc_check(multiple_docs, multi_query_result, full_collection)

        prev_score = float("inf")
        for i, doc in enumerate(multi_query_result):
            doc_id = doc.id
            assert doc_id in expected_weighted_scores, (
                f"Document {doc_id} should be in expected  scores"
            )
            expected_score = expected_weighted_scores[doc_id]
            actual_score = doc.score
            assert abs(actual_score - expected_score) < 1e-10, (
                f"score mismatch for document {doc_id}: expected {expected_score}, got {actual_score}"
            )
            assert doc.score <= prev_score, (
                f"Scores should be in descending order. Current: {doc.score}, Previous: {prev_score}"
            )
            prev_score = doc.score

    @pytest.mark.parametrize("topk", [5])
    @pytest.mark.parametrize("doc_num", [10])
    @pytest.mark.parametrize("filter", ["int32_field >= 3 and int32_field <= 7"])
    def test_query_consistency(
        self, full_collection: Collection, filter, doc_num, topk
    ):
        multiple_docs = [
            generate_doc(i, full_collection.schema) for i in range(doc_num)
        ]
        batchdoc_and_check(full_collection, multiple_docs, doc_num, operator="insert")
        results = []
        for i in range(5):
            query_result = full_collection.query(filter=filter, topk=topk)
            single_querydoc_check(multiple_docs, query_result, full_collection)

            results.append(query_result)
        assert len(results) == 5
        expected_count = len(results[0])
        for i, result in enumerate(results):
            assert len(result) == expected_count

        expected_ids = set(doc.id for doc in results[0])
        for i, result in enumerate(results):
            result_ids = set(doc.id for doc in result)
            assert result_ids == expected_ids

        for i, result in enumerate(results):
            result_ids = [doc.id for doc in result]
            expected_sorted_ids = sorted(result_ids, key=lambda x: int(x))
            assert result_ids == expected_sorted_ids

    @pytest.mark.parametrize("ef", [0, 100, 1024, 2048])
    @pytest.mark.parametrize("doc_num", [10])
    @pytest.mark.parametrize("topk", [1024])
    @pytest.mark.parametrize("filter", ["int32_field >= 3 and int32_field <= 7"])
    def test_query_vector_with_HnswQueryParam_valid(
        self,
        full_collection_new: Collection,
        doc_num,
        full_schema_new,
        topk,
        filter,
        ef,
    ):
        multiple_docs = [
            generate_doc(i, full_collection_new.schema) for i in range(doc_num)
        ]
        batchdoc_and_check(
            full_collection_new, multiple_docs, doc_num, operator="insert"
        )
        for k, v in DEFAULT_VECTOR_FIELD_NAME.items():
            doc_fields, doc_vectors = generate_vectordict_random(
                full_collection_new.schema
            )
            query_vector = doc_vectors[v]
            query_result = full_collection_new.query(
                filter=filter,
                vectors=VectorQuery(
                    field_name=v, vector=query_vector, param=HnswQueryParam(ef=ef)
                ),
                topk=topk,
            )
            assert len(query_result) > 0, (
                f"Expected at least 1 query result, but got {len(query_result)}"
            )
            single_querydoc_check(
                multiple_docs,
                query_result,
                full_collection_new,
                is_by_vector=1,
                query_vector=query_vector,
                data_type=k,
                vector_name=v,
            )

    @pytest.mark.parametrize("ef", [None, "invalid", 10.5])
    @pytest.mark.parametrize("doc_num", [10])
    @pytest.mark.parametrize("topk", [10])
    @pytest.mark.parametrize("filter", ["int32_field >= 3 and int32_field <= 7"])
    def test_query_vector_with_HnswQueryParam_invalid(
        self, full_collection: Collection, doc_num, topk, ef, filter
    ):
        multiple_docs = [
            generate_doc(i, full_collection.schema) for i in range(doc_num)
        ]
        batchdoc_and_check(full_collection, multiple_docs, doc_num, operator="insert")
        for k, v in DEFAULT_VECTOR_FIELD_NAME.items():
            doc_fields, doc_vectors = generate_vectordict_random(full_collection.schema)
            query_vector = doc_vectors[v]
            with pytest.raises(Exception) as exc_info:
                full_collection.query(
                    filter=filter,
                    vectors=VectorQuery(
                        field_name=v, vector=query_vector, param=HnswQueryParam(ef=ef)
                    ),
                    topk=topk,
                )
            assert INCOMPATIBLE_CONSTRUCTOR_ERROR_MSG in str(exc_info.value)

    @pytest.mark.parametrize("nprobe", [1, 10, 100, 2048])
    @pytest.mark.parametrize("doc_num", [10])
    @pytest.mark.parametrize("topk", [10])
    @pytest.mark.parametrize("filter", ["int32_field >= 3 and int32_field <= 7"])
    def test_query_vector_with_IVFQueryParam_valid(
        self, full_collection_ivf: Collection, nprobe, doc_num, topk, filter
    ):
        multiple_docs = [
            generate_doc(i, full_collection_ivf.schema) for i in range(doc_num)
        ]
        batchdoc_and_check_ivf(
            full_collection_ivf, multiple_docs, doc_num, operator="insert"
        )
        for k, v in DEFAULT_VECTOR_FIELD_NAME.items():
            doc_fields, doc_vectors = generate_vectordict_random(
                full_collection_ivf.schema
            )
            if v in ["vector_fp32_field"]:
                query_vector = doc_vectors[v]

                query_result = full_collection_ivf.query(
                    filter=filter,
                    vectors=VectorQuery(
                        field_name=v,
                        vector=query_vector,
                        param=IVFQueryParam(nprobe=nprobe),
                    ),
                    topk=topk,
                )
                assert len(query_result) > 0
                single_querydoc_check(
                    multiple_docs,
                    query_result,
                    full_collection_ivf,
                    is_by_vector=1,
                    query_vector=query_vector,
                    data_type=k,
                    vector_name=v,
                )

    @pytest.mark.parametrize("nprobe", [None, 10.5])
    @pytest.mark.parametrize("doc_num", [10])
    @pytest.mark.parametrize("topk", [10])
    @pytest.mark.parametrize("filter", ["int32_field >= 3 and int32_field <= 7"])
    def test_query_vector_with_IVFQueryParam_invalid(
        self, full_collection_ivf: Collection, nprobe, doc_num, topk, filter
    ):
        multiple_docs = [
            generate_doc(i, full_collection_ivf.schema) for i in range(doc_num)
        ]
        batchdoc_and_check_ivf(
            full_collection_ivf, multiple_docs, doc_num, operator="insert"
        )
        for k, v in DEFAULT_VECTOR_FIELD_NAME.items():
            doc_fields, doc_vectors = generate_vectordict_random(
                full_collection_ivf.schema
            )
            if v in ["vector_fp32_field"]:
                print("v:\n")
                print(v)
                query_vector = doc_vectors[v]
                with pytest.raises(Exception) as exc_info:
                    full_collection_ivf.query(
                        # filter=filter,
                        vectors=VectorQuery(
                            field_name=v,
                            vector=query_vector,
                            param=IVFQueryParam(nprobe=nprobe),
                        ),
                        topk=topk,
                    )
                assert INCOMPATIBLE_CONSTRUCTOR_ERROR_MSG in str(exc_info.value)

    @pytest.mark.parametrize("filter", ["int32_field >= 3 and int32_field <= 7"])
    @pytest.mark.parametrize("doc_num", [10])
    def test_query_vector_with_param_invalid(
        self, full_collection: Collection, doc_num, filter
    ):
        multiple_docs = [
            generate_doc(i, full_collection.schema) for i in range(doc_num)
        ]
        batchdoc_and_check(full_collection, multiple_docs, doc_num, operator="insert")

        with pytest.raises(Exception) as exc_info:
            for k, v in DEFAULT_VECTOR_FIELD_NAME.items():
                doc_fields, doc_vectors = generate_vectordict_random(
                    full_collection.schema
                )
                query_vector = doc_vectors[v]
                if v in ["vector_fp16_field", "vector_fp32_field"]:
                    full_collection.query(
                        filter=filter,
                        vectors=VectorQuery(
                            field_name=v, vector=query_vector, param=HnswIndexParam()
                        ),
                    )
        assert INCOMPATIBLE_FUNCTION_ERROR_MSG in str(exc_info.value)

    @pytest.mark.parametrize("doc_num", [10])
    @pytest.mark.parametrize(
        "test_case_name,vector_query,expected_error_msg",
        [
            (
                "Non-existent vector field name",
                lambda ref_dense_vector: VectorQuery(
                    field_name="nonexistent_vector", vector=ref_dense_vector
                ),
                "Expected exception for non-existent vector field name",
            ),
            (
                "Invalid vector data type for dense vector (string instead of list)",
                lambda ref_dense_vector: VectorQuery(
                    field_name="vector_fp32_field", vector="invalid_vector_data"
                ),
                "Expected exception for invalid dense vector data type",
            ),
            (
                "Invalid vector data type for sparse vector (list instead of dict)",
                lambda ref_dense_vector: VectorQuery(
                    field_name="sparse_fp32", vector=[1.0, 2.0, 3.0]
                ),
                "Expected exception for invalid sparse vector data type",
            ),
            (
                "Empty vector data for dense vector",
                lambda ref_dense_vector: VectorQuery(
                    field_name="vector_fp32_field", vector=[]
                ),
                "Expected exception for empty dense vector data",
            ),
            (
                "Invalid dimension for dense vector",
                lambda ref_dense_vector: VectorQuery(
                    field_name="vector_fp32_field", vector=[1.0, 2.0]
                ),  # Only 2 dimensions instead of 128
                "Expected exception for invalid dense vector dimension",
            ),
            (
                "Non-existent document ID for by_id query",
                lambda ref_dense_vector: VectorQuery(
                    field_name="vector_fp32_field", id="999"
                ),  # Non-existent ID
                "Expected exception for non-existent document ID",
            ),
            (
                "Both vector and id specified (invalid combination)",
                lambda ref_dense_vector: VectorQuery(
                    field_name="vector_fp32_field", vector=ref_dense_vector, id="5"
                ),
                "Expected exception for specifying both vector and id",
            ),
            (
                "Neither vector nor id specified",
                lambda ref_dense_vector: VectorQuery(
                    field_name="vector_fp32_field"
                ),  # Neither vector nor id
                "Expected exception for specifying neither vector nor id",
            ),
        ],
    )
    def test_query_vector_with_vectors_invalid(
        self,
        full_collection: Collection,
        doc_num,
        test_case_name,
        vector_query,
        expected_error_msg,
    ):
        multiple_docs = [
            generate_doc(i, full_collection.schema) for i in range(doc_num)
        ]
        batchdoc_and_check(full_collection, multiple_docs, doc_num, operator="insert")
        ref_doc_result = full_collection.fetch(ids=["5"])
        assert "5" in ref_doc_result
        ref_doc = ref_doc_result["5"]
        ref_dense_vector = ref_doc.vector("vector_fp32_field")

        with pytest.raises(Exception) as exc_info:
            full_collection.query(vectors=[vector_query(ref_dense_vector)])
        assert exc_info.value is not None, expected_error_msg

    @pytest.mark.parametrize("filter", ["int32_field >= 3 and int32_field <= 7"])
    @pytest.mark.parametrize("doc_num", [10])
    def test_query_invalid_param_incompatible_type(
        self, full_collection: Collection, doc_num, filter
    ):
        multiple_docs = [
            generate_doc(i, full_collection.schema) for i in range(doc_num)
        ]
        batchdoc_and_check(full_collection, multiple_docs, doc_num, operator="insert")

        with pytest.raises(Exception) as exc_info:
            for k, v in DEFAULT_VECTOR_FIELD_NAME.items():
                doc_fields, doc_vectors = generate_vectordict_random(
                    full_collection.schema
                )
                query_vector = doc_vectors[v]
                full_collection.query(
                    filter=filter,
                    vectors=VectorQuery(field_name=v, vector=query_vector),
                    param=HnswIndexParam(),
                    topk=3,
                )

        assert "query() got an unexpected keyword argument 'param'" in str(
            exc_info.value
        )


class TestRRFScoreCalculation:
    class MockDoc:
        def __init__(self, id, score=0.0):
            self._id = id
            self._score = score

        @property
        def id(self):
            return self._id

        @property
        def score(self):
            return self._score

        @score.setter
        def score(self, score):
            self._score = score

    def test_rrf_score_calculation_formula(self):
        k = 60

        assert abs(calculate_rrf_score(0, k) - 1.0 / 61) < 1e-10, (
            "RRF score for rank 0 should be 1/61"
        )
        assert abs(calculate_rrf_score(1, k) - 1.0 / 62) < 1e-10, (
            "RRF score for rank 1 should be 1/62"
        )
        assert abs(calculate_rrf_score(2, k) - 1.0 / 63) < 1e-10, (
            "RRF score for rank 2 should be 1/63"
        )
        assert abs(calculate_rrf_score(10, k) - 1.0 / 71) < 1e-10, (
            "RRF score for rank 10 should be 1/71"
        )

        k = 10
        assert abs(calculate_rrf_score(0, k) - 1.0 / 11) < 1e-10, (
            "RRF score for rank 0 with k=10 should be 1/11"
        )
        assert abs(calculate_rrf_score(1, k) - 1.0 / 12) < 1e-10, (
            "RRF score for rank 1 with k=10 should be 1/12"
        )

    def test_multi_vector_rrf_scores(self):
        query1_results = [self.MockDoc("1"), self.MockDoc("2"), self.MockDoc("3")]
        query2_results = [self.MockDoc("3"), self.MockDoc("1"), self.MockDoc("4")]
        query3_results = [self.MockDoc("2"), self.MockDoc("4"), self.MockDoc("5")]
        query_results = {
            "vector1": query1_results,
            "vector2": query2_results,
            "vector3": query3_results,
        }
        rrf_scores = calculate_multi_vector_rrf_scores(query_results, k=60)

        expected_doc1_score = 1.0 / 61 + 1.0 / 62
        assert abs(rrf_scores["1"] - expected_doc1_score) < 1e-10, (
            f"RRF score for doc1 mismatch: expected {expected_doc1_score}, got {rrf_scores['1']}"
        )
        expected_doc2_score = 1.0 / 62 + 1.0 / 61
        assert abs(rrf_scores["2"] - expected_doc2_score) < 1e-10, (
            f"RRF score for doc2 mismatch: expected {expected_doc2_score}, got {rrf_scores['2']}"
        )
        expected_doc3_score = 1.0 / 63 + 1.0 / 61
        assert abs(rrf_scores["3"] - expected_doc3_score) < 1e-10, (
            f"RRF score for doc3 mismatch: expected {expected_doc3_score}, got {rrf_scores['3']}"
        )
        expected_doc4_score = 1.0 / 63 + 1.0 / 62
        assert abs(rrf_scores["4"] - expected_doc4_score) < 1e-10, (
            f"RRF score for doc4 mismatch: expected {expected_doc4_score}, got {rrf_scores['4']}"
        )

        expected_doc5_score = 1.0 / 63
        assert abs(rrf_scores["5"] - expected_doc5_score) < 1e-10, (
            f"RRF score for doc5 mismatch: expected {expected_doc5_score}, got {rrf_scores['5']}"
        )
        sorted_scores = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        expected_order = ["1", "2", "3", "4", "5"]
        actual_order = [item[0] for item in sorted_scores]
        assert actual_order == expected_order, (
            f"RRF score ranking mismatch: expected {expected_order}, got {actual_order}"
        )


class TestCollectionConcurrencyOperations:
    @pytest.mark.parametrize("doc_num", [10])
    def test_concurrent_insert_update_upsert_query(
        self, full_collection: Collection, doc_num
    ):
        import threading

        results = []
        errors = []
        multiple_docs = [
            generate_doc(i, full_collection.schema) for i in range(1000, 1010)
        ]

        batchdoc_and_check(full_collection, multiple_docs, doc_num, operator="insert")

        def insert_operation(thread_id):
            try:
                multiple_docs = [
                    generate_doc(i, full_collection.schema)
                    for i in range(thread_id, thread_id + 5)
                ]
                result = full_collection.insert(multiple_docs)
                results.append(("insert", thread_id, len(result)))
            except Exception as e:
                errors.append(("insert", thread_id, str(e)))

        def update_operation(thread_id):
            try:
                multiple_docs = [
                    generate_doc_random(i, full_collection.schema)
                    for i in range(1000, 1001)
                ]
                result = full_collection.update(multiple_docs)
                results.append(("update", thread_id, len(result)))
            except Exception as e:
                errors.append(("update", thread_id, str(e)))

        def upsert_operation(thread_id):
            try:
                multiple_docs = [
                    generate_doc(i, full_collection.schema)
                    for i in range(thread_id, thread_id + 5)
                ]
                result = full_collection.upsert(multiple_docs)
                results.append(("upsert", thread_id, len(result)))
            except Exception as e:
                errors.append(("upsert", thread_id, str(e)))

        def query_operation(thread_id):
            try:
                if thread_id % 3 == 0:
                    result = full_collection.query(filter="int32_field > 1", topk=5)
                elif thread_id % 3 == 1:
                    result = full_collection.query(filter="bool_field = true", topk=3)
                else:
                    query_vector = [0.1] * 128
                    result = full_collection.query(
                        VectorQuery(
                            field_name="vector_fp32_field", vector=query_vector
                        ),
                        topk=3,
                    )

                results.append(("query", thread_id, len(result)))
            except Exception as e:
                errors.append(("query", thread_id, str(e)))

        def delete_operation(thread_id):
            try:
                # Delete some existing documents
                delete_ids = (
                    [f"{thread_id + 1}", f"{thread_id + 2}"]
                    if thread_id < 5
                    else [f"{thread_id % 5 + 1}"]
                )
                result = full_collection.delete(delete_ids)
                results.append(("delete", thread_id, len(result)))
            except Exception as e:
                errors.append(("delete", thread_id, str(e)))

        threads = []
        for i in range(1):
            thread = threading.Thread(target=insert_operation, args=(i,))
            threads.append(thread)
            thread.start()
        for i in range(1):
            thread = threading.Thread(target=update_operation, args=(i,))
            threads.append(thread)
            thread.start()
        for i in range(1):
            thread = threading.Thread(target=upsert_operation, args=(i,))
            threads.append(thread)
            thread.start()
        for i in range(1):
            thread = threading.Thread(target=query_operation, args=(i,))
            threads.append(thread)
            thread.start()
        for i in range(1):
            thread = threading.Thread(target=delete_operation, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        insert_results = [r for r in results if r[0] == "insert"]
        update_results = [r for r in results if r[0] == "update"]
        upsert_results = [r for r in results if r[0] == "upsert"]
        query_results = [r for r in results if r[0] == "query"]
        delete_results = [r for r in results if r[0] == "delete"]

        assert (
            len(insert_results)
            + len(update_results)
            + len(upsert_results)
            + len(query_results)
            + len(delete_results)
            > 0
        ), f"No operations succeeded. Errors: {errors}"

        critical_errors = [
            e for e in errors if "critical" in e[2].lower() or "fatal" in e[2].lower()
        ]
        assert len(critical_errors) == 0, f"Critical errors occurred: {critical_errors}"
