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

import pytest

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

from zvec import StatusCode
from distance_helper import *
from fixture_helper import *
from doc_helper import *
from params_helper import *

import time


# ==================== helper ====================
def batchdoc_and_check(collection: Collection, multiple_docs, operator="insert"):
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


def compute_exact_similarity_scores(
    vectors_a,
    vectors_b,
    metric_type=MetricType.IP,
    DataType=DataType.VECTOR_FP32,
    QuantizeType=QuantizeType.UNDEFINED,
):
    similarities = []
    for i, vec_a in enumerate(vectors_a):
        for j, vec_b in enumerate(vectors_b):
            similarity = distance_recall(vec_a, vec_b, metric_type, DataType)
            similarities.append((j, similarity))

    # For L2,COSINE metric, smaller distances mean higher similarity, so sort in ascending order
    if (
        metric_type in [MetricType.L2]
        and DataType
        in [DataType.VECTOR_FP32, DataType.VECTOR_FP16, DataType.VECTOR_INT8]
    ) or (
        metric_type in [MetricType.COSINE]
        and DataType in [DataType.VECTOR_FP32, DataType.VECTOR_FP16]
    ):
        similarities.sort(key=lambda x: x[1], reverse=False)  # Ascending order for L2

    else:
        similarities.sort(
            key=lambda x: x[1], reverse=True
        )  # Descending order for others

    # Special handling for COSINE in FP16 to address precision issues
    if metric_type == MetricType.COSINE and DataType == DataType.VECTOR_FP16:
        # Clamp values to valid cosine distance range [0, 2] and handle floating point errors
        similarities = [(idx, max(0.0, min(2.0, score))) for idx, score in similarities]

    return similarities


def get_ground_truth_for_vector_query(
    collection,
    query_vector,
    field_name,
    all_docs,
    query_idx,
    metric_type,
    k,
    use_exact_computation=False,
):
    if use_exact_computation:
        all_vectors = [doc.vectors[field_name] for doc in all_docs]

        for d, f in DEFAULT_VECTOR_FIELD_NAME.items():
            if field_name == f:
                DataType = d
                break
        similarities = compute_exact_similarity_scores(
            [query_vector],
            all_vectors,
            metric_type,
            DataType=DataType,
            QuantizeType=QuantizeType,
        )

        if metric_type == MetricType.COSINE and DataType == DataType.VECTOR_FP16:
            # Filter out tiny non-zero values that may be caused by precision errors
            similarities = [
                (idx, max(0.0, min(2.0, score))) for idx, score in similarities
            ]

        ground_truth_ids_scores = similarities[:k]

        return ground_truth_ids_scores

    else:
        full_result = collection.query(
            VectorQuery(field_name=field_name, vector=query_vector),
            topk=min(len(all_docs), 1024),
            include_vector=True,
        )

        ground_truth_ids_scores = [
            (result.id, result.score) for result in full_result[:k]
        ]

        if not ground_truth_ids_scores:
            ground_truth_ids_scores = [(all_docs[query_idx].id, 0)]

        return ground_truth_ids_scores


def get_ground_truth_map(collection, test_docs, query_vectors_map, metric_type, k):
    ground_truth_map = {}

    for field_name, query_vectors in query_vectors_map.items():
        ground_truth_map[field_name] = {}

        for i, query_vector in enumerate(query_vectors):
            # Get the ground truth for this query
            relevant_doc_ids_scores = get_ground_truth_for_vector_query(
                collection, query_vector, field_name, test_docs, i, metric_type, k, True
            )
            ground_truth_map[field_name][i] = relevant_doc_ids_scores

    return ground_truth_map


def calculate_recall_at_k(
    collection: Collection,
    test_docs,
    query_vectors_map,
    schema,
    k=1,
    expected_doc_ids_scores_map=None,
    tolerance=0.01,
):
    recall_stats = {}

    for field_name, query_vectors in query_vectors_map.items():
        recall_stats[field_name] = {
            "relevant_retrieved_count": 0,
            "total_relevant_count": 0,
            "retrieved_count": 0,
            "recall_at_k": 0.0,
        }

        for i, query_vector in enumerate(query_vectors):
            print("Starting %dth query" % i)

            query_result_list = collection.query(
                VectorQuery(field_name=field_name, vector=query_vector),
                topk=1024,
                include_vector=True,
            )
            retrieved_count = len(query_result_list)

            query_result_ids_scores = []
            for word in query_result_list:
                query_result_ids_scores.append((word.id, word.score))

            recall_stats[field_name]["retrieved_count"] += retrieved_count

            if i in (expected_doc_ids_scores_map[field_name]):
                expected_relevant_ids_scores = expected_doc_ids_scores_map[field_name][
                    i
                ]

            recall_stats[field_name]["total_relevant_count"] += len(
                expected_relevant_ids_scores
            )

            relevant_found_count = 0
            for ids_scores_except in expected_relevant_ids_scores:
                for ids_scores_result in query_result_ids_scores[:k]:
                    if int(ids_scores_result[0]) == int(ids_scores_except[0]):
                        relevant_found_count += 1
                        break
                    elif (
                        int(ids_scores_result[0]) != int(ids_scores_except[0])
                        and abs(ids_scores_result[1] - ids_scores_except[1])
                        <= tolerance
                    ):
                        print("IDs are not equal, but the error is small, tolerance")
                        print(
                            ids_scores_result[0],
                            ids_scores_except[0],
                            ids_scores_result[1],
                            ids_scores_except[1],
                            tolerance,
                        )
                        relevant_found_count += 1
                        break
                    else:
                        continue

            recall_stats[field_name]["relevant_retrieved_count"] += relevant_found_count

        # Calculate Recall@K
        if recall_stats[field_name]["total_relevant_count"] > 0:
            recall_stats[field_name]["recall_at_k"] = (
                recall_stats[field_name]["relevant_retrieved_count"]
                / recall_stats[field_name]["total_relevant_count"]
            )

    return recall_stats


def calculate_recall_at_k_multi_rrf(
    collection: Collection,
    test_docs,
    query_vectors_list,
    schema,
    top_k=1,
    expected_doc_ids_scores_map=None,
    tolerance=0.01,
):
    result_doc_ids_scores_map = []

    for doc_vectors in query_vectors_list:
        multi_query_vectors = []
        for k, v in DEFAULT_VECTOR_FIELD_NAME.items():
            multi_query_vectors.append(VectorQuery(field_name=v, vector=doc_vectors[v]))

        rrf_reranker = RrfReRanker(topn=10)
        multi_query_result = collection.query(
            vectors=multi_query_vectors,
            reranker=rrf_reranker,
        )
        result_dict = {}

        for doc in multi_query_result[:top_k]:
            result_dict[doc.id] = doc.score
        result_doc_ids_scores_map.append(result_dict)

    recall_stats = {
        "relevant_retrieved_count": 0,
        "total_relevant_count": 0,
        "retrieved_count": 0,
        "recall_at_k": 0.0,
    }

    for result_dict in result_doc_ids_scores_map:
        recall_stats["retrieved_count"] = recall_stats["retrieved_count"] + len(
            result_dict
        )

    for expected_dict in result_doc_ids_scores_map:
        recall_stats["total_relevant_count"] = recall_stats[
            "total_relevant_count"
        ] + len(expected_dict)

    for i in range(0, len(result_doc_ids_scores_map)):
        relevant_found_count = 0
        for k, v in result_doc_ids_scores_map[i].items():
            for k1, v1 in expected_doc_ids_scores_map[i].items():
                if k == k1:
                    relevant_found_count += 1
                    break
                elif k != k1 and abs(v - v1) <= tolerance:
                    print("IDs are not equal, but the error is small, tolerance")
                    print(k, k1, v, v1, tolerance)
                    relevant_found_count += 1
                    break
                else:
                    continue

        recall_stats["relevant_retrieved_count"] += relevant_found_count

        if recall_stats["total_relevant_count"] > 0:
            recall_stats["recall_at_k"] = (
                recall_stats["relevant_retrieved_count"]
                / recall_stats["total_relevant_count"]
            )

    return recall_stats


def calculate_recall_at_k_multi_weight(
    collection: Collection,
    test_docs,
    query_vectors_list,
    schema,
    weights,
    metric_type,
    top_k=1,
    expected_doc_ids_scores_map=None,
    tolerance=0.01,
):
    result_doc_ids_scores_map = []

    for doc_vectors in query_vectors_list:
        weighted_reranker = WeightedReRanker(
            topn=10, weights=weights, metric=metric_type
        )

        multi_query_vectors = []
        for k, v in DEFAULT_VECTOR_FIELD_NAME.items():
            multi_query_vectors.append(VectorQuery(field_name=v, vector=doc_vectors[v]))

        multi_query_result = collection.query(
            vectors=multi_query_vectors,
            reranker=weighted_reranker,
        )

        result_dict = {}

        for doc in multi_query_result[:top_k]:
            result_dict[doc.id] = doc.score
        result_doc_ids_scores_map.append(result_dict)

    recall_stats = {
        "relevant_retrieved_count": 0,
        "total_relevant_count": 0,
        "retrieved_count": 0,
        "recall_at_k": 0.0,
    }

    for result_dict in result_doc_ids_scores_map:
        recall_stats["retrieved_count"] = recall_stats["retrieved_count"] + len(
            result_dict
        )

    for expected_dict in result_doc_ids_scores_map:
        recall_stats["total_relevant_count"] = recall_stats[
            "total_relevant_count"
        ] + len(expected_dict)

    for i in range(0, len(result_doc_ids_scores_map)):
        relevant_found_count = 0
        for k, v in result_doc_ids_scores_map[i].items():
            for k1, v1 in expected_doc_ids_scores_map[i].items():
                if k == k1:
                    relevant_found_count += 1
                    break
                elif k != k1 and abs(v - v1) <= tolerance:
                    print("IDs are not equal, but the error is small, tolerance")
                    print(k, k1, v, v1, tolerance)
                    relevant_found_count += 1
                    break
                else:
                    continue

        recall_stats["relevant_retrieved_count"] += relevant_found_count

        if recall_stats["total_relevant_count"] > 0:
            recall_stats["recall_at_k"] = (
                recall_stats["relevant_retrieved_count"]
                / recall_stats["total_relevant_count"]
            )

    return recall_stats


class TestRecall:
    @pytest.mark.parametrize(
        "full_schema_new",
        [
            (True, True, HnswIndexParam()),
            (False, True, IVFIndexParam()),
            (False, True, FlatIndexParam()),
            (
                True,
                True,
                HnswIndexParam(
                    metric_type=MetricType.IP,
                    m=16,
                    ef_construction=100,
                ),
            ),
            (
                True,
                True,
                HnswIndexParam(
                    metric_type=MetricType.COSINE,
                    m=24,
                    ef_construction=150,
                ),
            ),
            (
                True,
                True,
                HnswIndexParam(
                    metric_type=MetricType.L2,
                    m=32,
                    ef_construction=200,
                ),
            ),
            (
                False,
                True,
                FlatIndexParam(
                    metric_type=MetricType.IP,
                ),
            ),
            (
                True,
                True,
                FlatIndexParam(
                    metric_type=MetricType.COSINE,
                ),
            ),
            (
                True,
                True,
                FlatIndexParam(
                    metric_type=MetricType.L2,
                ),
            ),
            (
                True,
                True,
                IVFIndexParam(
                    metric_type=MetricType.IP,
                    n_list=100,
                    n_iters=10,
                    use_soar=False,
                ),
            ),
            (
                True,
                True,
                IVFIndexParam(
                    metric_type=MetricType.L2,
                    n_list=200,
                    n_iters=20,
                    use_soar=True,
                ),
            ),
            (
                True,
                True,
                IVFIndexParam(
                    metric_type=MetricType.COSINE,
                    n_list=150,
                    n_iters=15,
                    use_soar=False,
                ),
            ),
        ],
        indirect=True,
    )
    @pytest.mark.parametrize("doc_num", [500])
    @pytest.mark.parametrize("query_num", [10])
    @pytest.mark.parametrize("top_k", [1])
    def test_recall_with_single_vector_valid_500(
        self,
        full_collection_new: Collection,
        doc_num,
        query_num,
        top_k,
        full_schema_new,
        request,
    ):
        full_schema_params = request.getfixturevalue("full_schema_new")

        for vector_para in full_schema_params.vectors:
            if vector_para.name == "vector_fp32_field":
                metric_type = vector_para.index_param.metric_type
                break

        multiple_docs = [
            generate_doc_recall(i, full_collection_new.schema) for i in range(doc_num)
        ]

        for i in range(10):
            batchdoc_and_check(
                full_collection_new,
                multiple_docs[i * 1000 : 1000 * (i + 1)],
                operator="insert",
            )

        stats = full_collection_new.stats
        assert stats.doc_count == len(multiple_docs)

        full_collection_new.optimize(option=OptimizeOption())

        time.sleep(2)

        query_vectors_map = {}
        for field_name in DEFAULT_VECTOR_FIELD_NAME.values():
            query_vectors_map[field_name] = [
                multiple_docs[i].vectors[field_name] for i in range(query_num)
            ]

        ground_truth_map = get_ground_truth_map(
            full_collection_new, multiple_docs, query_vectors_map, metric_type, top_k
        )

        for field_name in DEFAULT_VECTOR_FIELD_NAME.values():
            assert field_name in ground_truth_map
            field_gt = ground_truth_map[field_name]
            assert len(field_gt) == query_num

            for query_idx in range(query_num):
                assert query_idx in field_gt
                relevant_ids = field_gt[query_idx]
                assert isinstance(relevant_ids, list)
                assert len(relevant_ids) <= top_k

        recall_at_k_stats = calculate_recall_at_k(
            full_collection_new,
            multiple_docs,
            query_vectors_map,
            full_schema_new,
            k=top_k,
            expected_doc_ids_scores_map=ground_truth_map,
            tolerance=0.01,
        )
        print(f"Recall@{top_k} using Ground Truth:")
        for field_name, stats in recall_at_k_stats.items():
            print(f"  {field_name}:")
            print(
                f"    Relevant Retrieved: {stats['relevant_retrieved_count']}/{stats['total_relevant_count']}"
            )
            print(f"    Recall@{top_k}: {stats['recall_at_k']:.4f}")
        for k, v in recall_at_k_stats.items():
            assert v["recall_at_k"] == 1.0

    @pytest.mark.parametrize(
        "full_schema_new",
        [
            (True, True, HnswIndexParam()),
            (False, True, IVFIndexParam()),
            (False, True, FlatIndexParam()),  # ——ok
            (
                True,
                True,
                HnswIndexParam(
                    metric_type=MetricType.IP,
                    m=16,
                    ef_construction=100,
                ),
            ),
            (
                True,
                True,
                HnswIndexParam(
                    metric_type=MetricType.COSINE,
                    m=24,
                    ef_construction=150,
                ),
            ),
            # (True, True, HnswIndexParam(metric_type=MetricType.L2, m=32, ef_construction=200, )),
            (
                False,
                True,
                FlatIndexParam(
                    metric_type=MetricType.IP,
                ),
            ),
            (
                True,
                True,
                FlatIndexParam(
                    metric_type=MetricType.COSINE,
                ),
            ),
            # (True, True, FlatIndexParam(metric_type=MetricType.L2, )),
            (
                True,
                True,
                IVFIndexParam(
                    metric_type=MetricType.IP,
                    n_list=100,
                    n_iters=10,
                    use_soar=False,
                ),
            ),
            (
                True,
                True,
                IVFIndexParam(
                    metric_type=MetricType.L2,
                    n_list=200,
                    n_iters=20,
                    use_soar=True,
                ),
            ),
            # (True, True, IVFIndexParam(metric_type=MetricType.COSINE, n_list=150, n_iters=15, use_soar=False, )),
        ],
        indirect=True,
    )
    @pytest.mark.parametrize("doc_num", [2000])
    @pytest.mark.parametrize("query_num", [2])
    @pytest.mark.parametrize("top_k", [1])
    @pytest.mark.skip(reason="known bug")
    def test_recall_with_single_vector_valid_2000(
        self,
        full_collection_new: Collection,
        doc_num,
        query_num,
        top_k,
        full_schema_new,
        request,
    ):
        full_schema_params = request.getfixturevalue("full_schema_new")

        for vector_para in full_schema_params.vectors:
            if vector_para.name == "vector_fp32_field":
                metric_type = vector_para.index_param.metric_type
                break

        multiple_docs = [
            generate_doc_recall(i, full_collection_new.schema) for i in range(doc_num)
        ]

        for i in range(10):
            batchdoc_and_check(
                full_collection_new,
                multiple_docs[i * 1000 : 1000 * (i + 1)],
                operator="insert",
            )
        stats = full_collection_new.stats
        assert stats.doc_count == len(multiple_docs)

        full_collection_new.optimize(option=OptimizeOption())

        time.sleep(2)

        query_vectors_map = {}
        for field_name in DEFAULT_VECTOR_FIELD_NAME.values():
            query_vectors_map[field_name] = [
                multiple_docs[i].vectors[field_name] for i in range(query_num)
            ]

        ground_truth_map = get_ground_truth_map(
            full_collection_new, multiple_docs, query_vectors_map, metric_type, top_k
        )

        for field_name in DEFAULT_VECTOR_FIELD_NAME.values():
            assert field_name in ground_truth_map
            field_gt = ground_truth_map[field_name]
            assert len(field_gt) == query_num

            for query_idx in range(query_num):
                assert query_idx in field_gt
                relevant_ids = field_gt[query_idx]
                assert isinstance(relevant_ids, list)
                assert len(relevant_ids) <= top_k

        print(f"Ground Truth for Top-{top_k} Retrieval:")
        for field_name, field_gt in ground_truth_map.items():
            print(f"  {field_name}:")
            for query_idx, relevant_ids in field_gt.items():
                print(
                    f" Query {query_idx}: {len(relevant_ids)} relevant docs - {relevant_ids[:5]}{'...' if len(relevant_ids) > 5 else ''}"
                )

        # Calculate Recall@K using ground truth
        recall_at_k_stats = calculate_recall_at_k(
            full_collection_new,
            multiple_docs,
            query_vectors_map,
            full_schema_new,
            k=top_k,
            expected_doc_ids_scores_map=ground_truth_map,
            tolerance=0.01,
        )

        print(f"Recall@{top_k} using Ground Truth:")
        for field_name, stats in recall_at_k_stats.items():
            print(f"  {field_name}:")
            print(
                f"    Relevant Retrieved: {stats['relevant_retrieved_count']}/{stats['total_relevant_count']}"
            )
            print(f"    Recall@{top_k}: {stats['recall_at_k']:.4f}")
        for k, v in recall_at_k_stats.items():
            assert v["recall_at_k"] == 1.0

    @pytest.mark.parametrize(
        "full_schema_new",
        [
            (True, True, HnswIndexParam()),
            (False, True, IVFIndexParam()),
            (False, True, FlatIndexParam()),
            (
                True,
                True,
                HnswIndexParam(
                    metric_type=MetricType.IP,
                    m=16,
                    ef_construction=100,
                ),
            ),
            (
                True,
                True,
                HnswIndexParam(
                    metric_type=MetricType.COSINE,
                    m=24,
                    ef_construction=150,
                ),
            ),
            (
                True,
                True,
                HnswIndexParam(
                    metric_type=MetricType.L2,
                    m=32,
                    ef_construction=200,
                ),
            ),
            (
                False,
                True,
                FlatIndexParam(
                    metric_type=MetricType.IP,
                ),
            ),
            (
                True,
                True,
                FlatIndexParam(
                    metric_type=MetricType.COSINE,
                ),
            ),
            (
                True,
                True,
                FlatIndexParam(
                    metric_type=MetricType.L2,
                ),
            ),
            (
                True,
                True,
                IVFIndexParam(
                    metric_type=MetricType.IP,
                    n_list=100,
                    n_iters=10,
                    use_soar=False,
                ),
            ),
            (
                True,
                True,
                IVFIndexParam(
                    metric_type=MetricType.L2,
                    n_list=200,
                    n_iters=20,
                    use_soar=True,
                ),
            ),
            (
                True,
                True,
                IVFIndexParam(
                    metric_type=MetricType.COSINE,
                    n_list=150,
                    n_iters=15,
                    use_soar=False,
                ),
            ),
        ],
        indirect=True,
    )
    @pytest.mark.parametrize("doc_num", [500])
    @pytest.mark.parametrize("query_num", [10])
    @pytest.mark.parametrize("top_k", [1])
    def test_recall_with_multi_vector_rrf(
        self,
        full_collection_new: Collection,
        doc_num,
        query_num,
        top_k,
        full_schema_new,
        request,
    ):
        full_schema_params = request.getfixturevalue("full_schema_new")

        for vector_para in full_schema_params.vectors:
            if vector_para.name == "vector_fp32_field":
                metric_type = vector_para.index_param.metric_type
                break

        multiple_docs = [
            generate_doc_recall(i, full_collection_new.schema) for i in range(doc_num)
        ]

        for i in range(10):
            batchdoc_and_check(
                full_collection_new,
                multiple_docs[i * 1000 : 1000 * (i + 1)],
                operator="insert",
            )

        stats = full_collection_new.stats
        assert stats.doc_count == len(multiple_docs)

        full_collection_new.optimize(option=OptimizeOption())

        time.sleep(2)

        query_vectors_list = [multiple_docs[i].vectors for i in range(query_num)]

        expected_result_map = []

        for doc_vectors in query_vectors_list:
            single_query_results = {}
            for k, v in DEFAULT_VECTOR_FIELD_NAME.items():
                single_query_results[v] = full_collection_new.query(
                    VectorQuery(field_name=v, vector=doc_vectors[v])
                )
            expected_rrf_scores_dict = calculate_multi_vector_rrf_scores(
                single_query_results
            )

            sorted_dict_desc = dict(
                sorted(
                    expected_rrf_scores_dict.items(), key=lambda x: x[1], reverse=True
                )[:top_k]
            )

            expected_result_map.append(sorted_dict_desc)

        recall_at_k_stats = calculate_recall_at_k_multi_rrf(
            full_collection_new,
            multiple_docs,
            query_vectors_list,
            full_schema_new,
            top_k=top_k,
            expected_doc_ids_scores_map=expected_result_map,
            tolerance=0.01,
        )

        # Print Recall@K statistics
        print(f"Recall@{top_k} using Ground Truth:")

        print(
            f"Relevant Retrieved: {recall_at_k_stats['relevant_retrieved_count']}/{recall_at_k_stats['total_relevant_count']}"
        )
        print(f" Recall@{top_k}: {recall_at_k_stats['recall_at_k']:.4f}")
        assert recall_at_k_stats["recall_at_k"] == 1.0

    @pytest.mark.parametrize(
        "weights",
        [
            {
                "vector_fp32_field": 0.49,
                "vector_fp16_field": 0.01,
                "vector_int8_field": 0.3,
                "sparse_vector_fp32_field": 0.1,
                "sparse_vector_fp16_field": 0.1,
            }
        ],
    )
    @pytest.mark.parametrize(
        "metrictype",
        [MetricType.COSINE, MetricType.IP, MetricType.L2],
    )
    @pytest.mark.parametrize(
        "full_schema_new",
        [
            (True, True, HnswIndexParam()),
            (False, True, IVFIndexParam()),
            (False, True, FlatIndexParam()),
            (
                True,
                True,
                HnswIndexParam(
                    metric_type=MetricType.IP,
                    m=16,
                    ef_construction=100,
                ),
            ),
            (
                True,
                True,
                HnswIndexParam(
                    metric_type=MetricType.COSINE,
                    m=24,
                    ef_construction=150,
                ),
            ),
            (
                True,
                True,
                HnswIndexParam(
                    metric_type=MetricType.L2,
                    m=32,
                    ef_construction=200,
                ),
            ),
            (
                False,
                True,
                FlatIndexParam(
                    metric_type=MetricType.IP,
                ),
            ),
            (
                True,
                True,
                FlatIndexParam(
                    metric_type=MetricType.COSINE,
                ),
            ),
            (
                True,
                True,
                FlatIndexParam(
                    metric_type=MetricType.L2,
                ),
            ),
            (
                True,
                True,
                IVFIndexParam(
                    metric_type=MetricType.IP,
                    n_list=100,
                    n_iters=10,
                    use_soar=False,
                ),
            ),
            (
                True,
                True,
                IVFIndexParam(
                    metric_type=MetricType.L2,
                    n_list=200,
                    n_iters=20,
                    use_soar=True,
                ),
            ),
            (
                True,
                True,
                IVFIndexParam(
                    metric_type=MetricType.COSINE,
                    n_list=150,
                    n_iters=15,
                    use_soar=False,
                ),
            ),
        ],
        indirect=True,
    )
    @pytest.mark.parametrize("doc_num", [500])
    @pytest.mark.parametrize("query_num", [10])
    @pytest.mark.parametrize("top_k", [1])
    def test_recall_with_multi_vector_weight(
        self,
        full_collection_new: Collection,
        doc_num,
        query_num,
        top_k,
        full_schema_new,
        request,
        weights,
        metrictype,
    ):
        multiple_docs = [
            generate_doc_recall(i, full_collection_new.schema) for i in range(doc_num)
        ]

        for i in range(10):
            batchdoc_and_check(
                full_collection_new,
                multiple_docs[i * 1000 : 1000 * (i + 1)],
                operator="insert",
            )

        stats = full_collection_new.stats
        assert stats.doc_count == len(multiple_docs)

        full_collection_new.optimize(option=OptimizeOption())

        time.sleep(2)

        query_vectors_list = [multiple_docs[i].vectors for i in range(query_num)]

        print("query_vectors_list:\n")
        print(query_vectors_list)

        expected_result_map = []

        for doc_vectors in query_vectors_list:
            single_query_results = {}
            for k, v in DEFAULT_VECTOR_FIELD_NAME.items():
                single_query_results[v] = full_collection_new.query(
                    VectorQuery(field_name=v, vector=doc_vectors[v])
                )

            expected_weighted_scores = calculate_multi_vector_weighted_scores(
                single_query_results, weights, metrictype
            )

            sorted_dict_desc = dict(
                sorted(
                    expected_weighted_scores.items(), key=lambda x: x[1], reverse=True
                )[:top_k]
            )

            expected_result_map.append(sorted_dict_desc)

        recall_at_k_stats = calculate_recall_at_k_multi_weight(
            full_collection_new,
            multiple_docs,
            query_vectors_list,
            full_schema_new,
            weights,
            metrictype,
            top_k=top_k,
            expected_doc_ids_scores_map=expected_result_map,
            tolerance=0.01,
        )
        print(f"Recall@{top_k} using Ground Truth:")

        print(
            f"Relevant Retrieved: {recall_at_k_stats['relevant_retrieved_count']}/{recall_at_k_stats['total_relevant_count']}"
        )
        print(f" Recall@{top_k}: {recall_at_k_stats['recall_at_k']:.4f}")
        assert recall_at_k_stats["recall_at_k"] == 1.0
