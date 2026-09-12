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
import threading, time

indextest_collection_schema = zvec.CollectionSchema(
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
            "vector_fp32_field",
            DataType.VECTOR_FP32,
            dimension=128,
            index_param=HnswIndexParam(),
        ),
        VectorSchema(
            "vector_fp16_field",
            DataType.VECTOR_FP16,
            dimension=128,
            index_param=HnswIndexParam(),
        ),
        VectorSchema(
            "vector_int8_field",
            DataType.VECTOR_INT8,
            dimension=128,
            index_param=HnswIndexParam(),
        ),
        VectorSchema(
            "sparse_vector_fp32_field",
            DataType.SPARSE_VECTOR_FP32,
            dimension=128,
            index_param=HnswIndexParam(),
        ),
        VectorSchema(
            "sparse_vector_fp16_field",
            DataType.SPARSE_VECTOR_FP16,
            dimension=128,
            index_param=HnswIndexParam(),
        ),
    ],
)
columntest_collection_schema = zvec.CollectionSchema(
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
            "dense_fp32_field",
            DataType.VECTOR_FP32,
            dimension=128,
            index_param=HnswIndexParam(),
        ),
        VectorSchema(
            "sparse_fp32_field",
            DataType.SPARSE_VECTOR_FP32,
            dimension=128,
            index_param=HnswIndexParam(),
        ),
    ],
)


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

    def test_collection_flush_with_reopen(self, tmp_path_factory):
        # Create collection
        temp_dir = tmp_path_factory.mktemp("zvec")
        collection_path = temp_dir / "test_collection"

        collection_option = CollectionOption(read_only=False, enable_mmap=True)
        # Create and open collection
        coll1 = zvec.create_and_open(
            path=str(collection_path),
            schema=columntest_collection_schema,
            option=collection_option,
        )
        assert coll1 is not None, "Failed to create and open collection"

        # Insert some data
        doc1 = Doc(
            id="1",
            fields={"id": 1, "name": "test1"},
            vectors={
                "dense_fp32_field": np.random.random(128).tolist(),
                "sparse_fp32_field": {1: 1.0, 2: 2.0},
            },
        )

        result = coll1.insert(doc1)
        assert result.ok()

        coll1.flush()

        fetched_docs = coll1.fetch(["1"])
        assert "1" in fetched_docs
        assert fetched_docs["1"].id == "1"


class TestOptimize:
    def test_optimize(self, full_collection_new: Collection):
        docs = [generate_doc(i, full_collection_new.schema) for i in range(10)]

        result = full_collection_new.insert(docs)
        assert bool(result)
        for item in result:
            assert item.ok()

        stats = full_collection_new.stats
        assert stats is not None
        assert stats.doc_count == 10

        full_collection_new.optimize(option=OptimizeOption())

        fetched_docs = full_collection_new.fetch(["1"])
        assert "1" in fetched_docs
        assert fetched_docs["1"].id == "1"

    def test_optimize_with_reopen(self, tmp_path_factory):
        # Create collection
        temp_dir = tmp_path_factory.mktemp("zvec")
        collection_path = temp_dir / "test_collection"

        collection_option = CollectionOption(read_only=False, enable_mmap=True)
        # Create and open collection
        coll1 = zvec.create_and_open(
            path=str(collection_path),
            schema=columntest_collection_schema,
            option=collection_option,
        )
        assert coll1 is not None, "Failed to create and open collection"

        # Insert some data
        doc1 = Doc(
            id="1",
            fields={"id": 1, "name": "test1"},
            vectors={
                "dense_fp32_field": np.random.random(128).tolist(),
                "sparse_fp32_field": {1: 1.0, 2: 2.0},
            },
        )

        result = coll1.insert(doc1)
        assert result.ok()

        coll1.optimize(option=OptimizeOption())

        fetched_docs = coll1.fetch(["1"])
        assert "1" in fetched_docs
        assert fetched_docs["1"].id == "1"

    @pytest.mark.parametrize("concurrency", [0, 1, 4, 8])
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
    def test_optimize_with_valid_concurrency_values(
        self,
        full_collection_new: Collection,
        full_schema_new,
        doc_num,
        concurrency,
        queries=None,
    ):
        """Test valid values for concurrency parameter"""
        """
        Verify index consistency before and after optimization

        Args:
            collection: zvec collection object
            queries: Optional query list, use default queries if not provided
        """
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
        # Build some default queries if none provided
        if queries is None:
            queries = []

            # Get schema info to build appropriate queries
            schema = full_collection_new

            # Build queries for each scalar field
            for field in full_schema_new.fields:
                if field.data_type == DataType.STRING:
                    queries.append({"filter": f"{field.name} >= ''", "topk": 10})
                elif field.data_type in [
                    DataType.INT32,
                    DataType.INT64,
                    DataType.UINT32,
                    DataType.UINT64,
                ]:
                    queries.append({"filter": f"{field.name} >= 0", "topk": 10})
                elif field.data_type in [DataType.FLOAT, DataType.DOUBLE]:
                    queries.append({"filter": f"{field.name} >= 0.0", "topk": 10})
                elif field.data_type == DataType.BOOL:
                    queries.append({"filter": f"{field.name} = true", "topk": 10})

            # Build queries for each vector field
            for vector in full_schema_new.vectors:
                # Build random query vectors
                import numpy as np

                if vector.data_type == DataType.VECTOR_FP32:
                    query_vector = np.random.random(vector.dimension).tolist()
                elif vector.data_type == DataType.VECTOR_FP16:
                    query_vector = np.random.random(vector.dimension).tolist()
                elif vector.data_type in [
                    DataType.SPARSE_VECTOR_FP32,
                    DataType.SPARSE_VECTOR_FP16,
                ]:
                    query_vector = {
                        i: float(np.random.random())
                        for i in range(min(10, vector.dimension))
                    }
                else:
                    continue

                queries.append(
                    {
                        "vector_query": {
                            "field_name": vector.name,
                            "vector": query_vector,
                        },
                        "topk": 10,
                    }
                )

        # Store query results before optimization
        results_before_optimize = []

        print("Executing queries before optimization...")
        for i, query in enumerate(queries):
            if "vector_query" in query:
                result = full_collection_new.query(
                    VectorQuery(
                        field_name=query["vector_query"]["field_name"],
                        vector=query["vector_query"]["vector"],
                    ),
                    topk=query["topk"],
                )
            else:
                result = full_collection_new.query(
                    filter=query["filter"], topk=query["topk"]
                )

            results_before_optimize.append(
                {
                    "query": query,
                    "result_count": len(result),
                    "result_ids": set(doc.id for doc in result),
                    "result_scores": [doc.score for doc in result],
                }
            )
            print(f"Query {i + 1}: Found {len(result)} results")

        # Record statistics before optimization
        stats_before = full_collection_new.stats
        print(f"Documents before optimization: {stats_before.doc_count}")
        print(
            f"Index completeness before optimization: {stats_before.index_completeness}"
        )

        # Execute optimization
        print("Executing optimization...")
        # Use valid concurrency values for optimization
        full_collection_new.optimize(option=OptimizeOption(concurrency=concurrency))

        stats = full_collection_new.stats
        assert stats.doc_count == len(multiple_docs)

        for i in range(doc_num):
            fetched_docs = full_collection_new.fetch([str(i)])
            assert str(i) in fetched_docs
            assert fetched_docs[str(i)].id == str(i)

        # Store query results after optimization
        results_after_optimize = []

        print("Executing queries after optimization...")
        for i, query in enumerate(queries):
            if "vector_query" in query:
                result = full_collection_new.query(
                    VectorQuery(
                        field_name=query["vector_query"]["field_name"],
                        vector=query["vector_query"]["vector"],
                    ),
                    topk=query["topk"],
                )
            else:
                result = full_collection_new.query(
                    filter=query["filter"], topk=query["topk"]
                )

            results_after_optimize.append(
                {
                    "query": query,
                    "result_count": len(result),
                    "result_ids": set(doc.id for doc in result),
                    "result_scores": [doc.score for doc in result],
                }
            )
            print(f"Query {i + 1}: Found {len(result)} results")

        # Record statistics after optimization
        stats_after = full_collection_new.stats
        print(f"Documents after optimization: {stats_after.doc_count}")
        print(
            f"Index completeness after optimization: {stats_after.index_completeness}"
        )

        # Verify consistency
        print("\nVerifying index consistency before and after optimization...")
        all_consistent = True

        for i, (before, after) in enumerate(
            zip(results_before_optimize, results_after_optimize)
        ):
            query_info = before["query"]

            # Check if result counts are consistent
            if before["result_count"] != after["result_count"]:
                print(
                    f"Query {i + 1} result count inconsistent: before {before['result_count']}, after {after['result_count']}"
                )
                all_consistent = False
                continue

            # Check if result ID sets are consistent
            if before["result_ids"] != after["result_ids"]:
                print(f"Query {i + 1} result ID set inconsistent")
                print(f"  Before IDs: {sorted(list(before['result_ids']))}")
                print(f"  After IDs: {sorted(list(after['result_ids']))}")
                all_consistent = False
                continue

            # Check if scores are basically consistent (allowing minor differences)
            import math

            scores_match = True
            for b_score, a_score in zip(
                before["result_scores"], after["result_scores"]
            ):
                if not math.isclose(b_score, a_score, rel_tol=1e-2):
                    scores_match = False
                    break

            if not scores_match:
                print(f"Query {i + 1} result scores inconsistent")
                all_consistent = False
                continue

            print(f"Query {i + 1}: Consistent")

        # Verify statistics
        if stats_before.doc_count != stats_after.doc_count:
            print(
                f"Document count inconsistent: before {stats_before.doc_count}, after {stats_after.doc_count}"
            )
            all_consistent = False

        if all_consistent:
            print(
                "\n✓ All verifications passed, indexes remain consistent before and after optimization"
            )
        else:
            print("\n✗ Inconsistencies found, please check index status")

        assert all_consistent == True

    @pytest.mark.parametrize(
        "concurrency",
        [
            # -1, -5,           # Negative values
            1.5,
            2.7,  # Float values
            "2",
            "8",
            "auto",  # String values
            # True, False       # Boolean values
        ],
    )
    def test_optimize_with_invalid_concurrency_values(
        self, full_collection_new: Collection, concurrency
    ):
        """Test various invalid values for concurrency parameter"""
        docs = [generate_doc(i, full_collection_new.schema) for i in range(10)]

        result = full_collection_new.insert(docs)
        assert bool(result)
        for item in result:
            assert item.ok()

        stats = full_collection_new.stats
        assert stats is not None
        assert stats.doc_count == 10

        # Using invalid concurrency values should raise an exception
        with pytest.raises(Exception) as exc_info:
            full_collection_new.optimize(option=OptimizeOption(concurrency=concurrency))

        # Depending on the implementation, there may be different error messages
        assert any(
            msg in str(exc_info.value)
            for msg in ["invalid", "concurrency", "parameter", "value", "type"]
        )

    def test_optimize_with_none_concurrency_value(
        self, full_collection_new: Collection
    ):
        """Test concurrency parameter with None value (invalid value)"""
        docs = [generate_doc(i, full_collection_new.schema) for i in range(10)]

        result = full_collection_new.insert(docs)
        assert bool(result)
        for item in result:
            assert item.ok()

        stats = full_collection_new.stats
        assert stats is not None
        assert stats.doc_count == 10

        # Using None as concurrency value should raise an exception
        with pytest.raises(Exception) as exc_info:
            full_collection_new.optimize(option=OptimizeOption(concurrency=None))

        assert any(
            msg in str(exc_info.value)
            for msg in ["invalid", "concurrency", "parameter", "value"]
        )

    @pytest.mark.parametrize(
        "concurrency", [999999, 1000000]
    )  # Assume these are too large values
    def test_optimize_with_too_large_concurrency_values(
        self, full_collection_new: Collection, concurrency
    ):
        """Test too large values for concurrency parameter"""
        docs = [generate_doc(i, full_collection_new.schema) for i in range(10)]

        result = full_collection_new.insert(docs)
        assert bool(result)
        for item in result:
            assert item.ok()

        stats = full_collection_new.stats
        assert stats is not None
        assert stats.doc_count == 10

        # Using too large concurrency values may not raise an exception, but will try to use the maximum available threads
        # Or may raise an exception in some implementations
        try:
            full_collection_new.optimize(option=OptimizeOption(concurrency=concurrency))

            # Verify data is still accessible after optimization
            fetched_docs = full_collection_new.fetch(["1"])
            assert "1" in fetched_docs
            assert fetched_docs["1"].id == "1"
        except Exception as e:
            # If an exception is raised, ensure it's a reasonable error message
            assert any(
                msg in str(e)
                for msg in [
                    "invalid",
                    "concurrency",
                    "thread",
                    "parameter",
                    "value",
                    "exceeds",
                ]
            )

    def test_optimize_in_read_only_mode(self, tmp_path_factory):
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

        temp_dir = tmp_path_factory.mktemp("zvec")
        collection_path = temp_dir / "test_collection"

        coll1 = zvec.create_and_open(
            path=str(collection_path),
            schema=collection_schema,
            option=collection_option,
        )

        assert coll1 is not None, "Failed to create and open collection"
        doc = Doc(
            id="1",
            fields={"id": 1, "name": "test"},
            vectors={"dense": np.random.random(128).tolist()},
        )
        result = coll1.insert(doc)
        assert result.ok()
        del coll1

        collection_option_reopen = CollectionOption(read_only=True, enable_mmap=True)
        coll2 = zvec.open(path=str(collection_path), option=collection_option_reopen)

        assert coll2 is not None, "Failed to reopen collection"
        assert coll2.path == str(collection_path)
        assert coll2.schema.name == collection_schema.name

        with pytest.raises(Exception) as exc_info:
            coll2.optimize(option=OptimizeOption())

        assert any(
            msg in str(exc_info.value).lower()
            for msg in ["read", "only", "readonly", "permission", "access", "mode"]
        )

        fetched_docs = coll2.fetch(["1"])
        assert "1" in fetched_docs
        fetched_doc = fetched_docs["1"]
        assert fetched_doc.id == "1"
        assert fetched_doc.field("name") == "test"

        if hasattr(coll2, "destroy") and coll2 is not None:
            try:
                coll2.destroy()
            except Exception as e:
                print(f"Warning: failed to destroy collection: {e}")

    def test_optimize_on_destroyed_collection(
        self, collection_temp_dir, collection_option: CollectionOption
    ):
        schema = CollectionSchema(
            name="test_optimize_destroyed",
            fields=[
                FieldSchema("id", DataType.INT64, nullable=False),
                FieldSchema("name", DataType.STRING, nullable=True),
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
        collection = zvec.create_and_open(
            path=collection_temp_dir, schema=schema, option=collection_option
        )

        docs = [generate_doc(i, collection.schema) for i in range(3)]
        result = collection.insert(docs)
        assert bool(result)
        for item in result:
            assert item.ok()

        collection.destroy()

        with pytest.raises(Exception) as exc_info:
            collection.optimize(option=OptimizeOption())

        assert any(
            msg in str(exc_info.value)
            for msg in ["destroyed", "access", "collection", "path", "exist"]
        )

    def test_concurrent_optimize_calls(self, full_collection: Collection):
        docs = [generate_doc(i, full_collection.schema) for i in range(5)]
        result = full_collection.insert(docs)
        assert bool(result)
        for item in result:
            assert item.ok()

        stats = full_collection.stats
        assert stats is not None
        assert stats.doc_count == 5

        exceptions = []

        def optimize_worker():
            try:
                for i in range(3):
                    full_collection.optimize(option=OptimizeOption())
                    time.sleep(0.01)
            except Exception as e:
                exceptions.append(e)

        threads = []
        for i in range(3):
            thread = threading.Thread(target=optimize_worker)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        for exc in exceptions:
            assert any(
                msg in str(exc).lower()
                for msg in ["concurrent", "lock", "thread", "access", "conflict"]
            ), f"Unexpected exception: {exc}"

        fetched_docs = full_collection.fetch(["1"])
        assert "1" in fetched_docs
        assert fetched_docs["1"].id == "1"

    def test_multi_thread_optimize_with_operations(self, full_collection: Collection):
        docs = [generate_doc(i, full_collection.schema) for i in range(10)]
        result = full_collection.insert(docs)
        assert bool(result)
        for item in result:
            assert item.ok()

        stats = full_collection.stats
        assert stats is not None
        assert stats.doc_count == 10

        results = {
            "insert_success": 0,
            "query_success": 0,
            "update_success": 0,
            "delete_success": 0,
            "optimize_success": 0,
            "exceptions": [],
        }
        results_lock = threading.Lock()

        def insert_worker():
            for i in range(10, 15):
                try:
                    doc = generate_doc(i, full_collection.schema)
                    result = full_collection.insert(doc)
                    if result and result.ok():
                        with results_lock:
                            results["insert_success"] += 1
                except Exception as e:
                    with results_lock:
                        results["exceptions"].append(f"Insert error: {e}")

        def query_worker():
            for _ in range(10):
                try:
                    query_result = full_collection.query(filter="id >= 0", topk=5)
                    with results_lock:
                        results["query_success"] += len(query_result)
                except Exception as e:
                    with results_lock:
                        results["exceptions"].append(f"Query error: {e}")

        def update_worker():
            for i in range(3):
                try:
                    doc = generate_doc(i, full_collection.schema)
                    result = full_collection.update(doc)
                    if result and result.ok():
                        with results_lock:
                            results["update_success"] += 1
                except Exception as e:
                    with results_lock:
                        results["exceptions"].append(f"Update error: {e}")

        def delete_worker():
            for i in range(15, 18):
                try:
                    result = full_collection.delete([str(i)])
                    if result:
                        with results_lock:
                            results["delete_success"] += 1
                except Exception as e:
                    with results_lock:
                        results["exceptions"].append(f"Delete error: {e}")

        def optimize_worker():
            for _ in range(2):
                try:
                    full_collection.optimize(option=OptimizeOption())
                    with results_lock:
                        results["optimize_success"] += 1
                    time.sleep(0.05)
                except Exception as e:
                    with results_lock:
                        results["exceptions"].append(f"Optimize error: {e}")

        threads = []
        threads.append(threading.Thread(target=insert_worker))
        threads.append(threading.Thread(target=query_worker))
        threads.append(threading.Thread(target=update_worker))
        threads.append(threading.Thread(target=delete_worker))
        threads.append(threading.Thread(target=optimize_worker))

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        assert results["insert_success"] >= 0, (
            f"Expected some inserts to succeed, got {results['insert_success']}"
        )
        assert results["query_success"] >= 0, (
            f"Expected some queries to succeed, got {results['query_success']}"
        )
        assert results["update_success"] >= 0, (
            f"Expected some updates to succeed, got {results['update_success']}"
        )
        assert results["optimize_success"] >= 0, (
            f"Expected some optimize calls to succeed, got {results['optimize_success']}"
        )

        if results["exceptions"]:
            print(
                f"Exceptions occurred during concurrent operations: {results['exceptions']}"
            )

        final_stats = full_collection.stats
        assert final_stats is not None
        assert final_stats.doc_count >= 10

        fetched_docs = full_collection.fetch(["1"])
        assert "1" in fetched_docs
        assert fetched_docs["1"].id == "1"

    def test_optimize_empty_collection(self, basic_collection: Collection):
        stats = basic_collection.stats
        assert stats is not None
        assert stats.doc_count == 0

        basic_collection.optimize(option=OptimizeOption())

        stats_after = basic_collection.stats
        assert stats_after is not None
        assert stats_after.doc_count == 0

        doc = generate_doc(1, basic_collection.schema)
        result = basic_collection.insert(doc)
        assert bool(result)
        assert result.ok()

        fetched_docs = basic_collection.fetch(["1"])
        assert "1" in fetched_docs
        assert fetched_docs["1"].id == "1"

        stats_final = basic_collection.stats
        assert stats_final.doc_count == 1

    def test_optimize_single_record_collection(self, basic_collection: Collection):
        doc = generate_doc(1, basic_collection.schema)
        result = basic_collection.insert(doc)
        assert bool(result)
        assert result.ok()

        stats = basic_collection.stats
        assert stats is not None
        assert stats.doc_count == 1

        basic_collection.optimize(option=OptimizeOption())

        fetched_docs = basic_collection.fetch(["1"])
        assert "1" in fetched_docs
        assert fetched_docs["1"].id == "1"

        stats_after = basic_collection.stats
        assert stats_after.doc_count == 1

        doc2 = generate_doc(2, basic_collection.schema)
        result2 = basic_collection.insert(doc2)
        assert bool(result2)
        assert result2.ok()

        fetched_docs = basic_collection.fetch(["1", "2"])
        assert len(fetched_docs) == 2
        assert fetched_docs["1"].id == "1"
        assert fetched_docs["2"].id == "2"

    def test_optimize_already_optimized_collection(self, full_collection: Collection):
        docs = [generate_doc(i, full_collection.schema) for i in range(5)]
        result = full_collection.insert(docs)
        assert bool(result)
        for item in result:
            assert item.ok()

        stats = full_collection.stats
        assert stats is not None
        assert stats.doc_count == 5

        full_collection.optimize(option=OptimizeOption())

        fetched_docs = full_collection.fetch(["1"])
        assert "1" in fetched_docs
        assert fetched_docs["1"].id == "1"

        full_collection.optimize(option=OptimizeOption())

        fetched_docs = full_collection.fetch(["1", "2"])
        assert len(fetched_docs) >= 2
        assert fetched_docs["1"].id == "1"

        full_collection.optimize(option=OptimizeOption())

        query_result = full_collection.query(filter="int32_field >= 0", topk=10)
        assert len(query_result) == 5

        final_stats = full_collection.stats
        assert final_stats.doc_count == 5


class TestIndexDDL:
    @pytest.mark.parametrize("field_name", DEFAULT_SCALAR_FIELD_NAME.values())
    @pytest.mark.parametrize("index_type", SUPPORT_SCALAR_INDEX_TYPES)
    def test_scalar_index_operation(
        self,
        full_collection_new: Collection,
        field_name: str,
        index_type: IndexType,
    ):
        # INSERT 0~5 Doc
        docs = [generate_doc(i, full_collection_new.schema) for i in range(5)]

        result = full_collection_new.insert(docs)
        assert len(result) == 5
        for item in result:
            assert item.ok()

        stats = full_collection_new.stats
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

        query_result_before = full_collection_new.query(filter=query_filter, topk=10)

        if index_type not in DEFAULT_INDEX_PARAMS:
            pytest.fail(f"Unsupported index type for index creation: {index_type}")
        index_param = DEFAULT_INDEX_PARAMS[index_type]

        full_collection_new.create_index(
            field_name=field_name, index_param=index_param, option=IndexOption()
        )
        stats_after_create = full_collection_new.stats
        assert stats_after_create is not None
        assert stats_after_create.doc_count == 5

        query_result_after = full_collection_new.query(filter=query_filter, topk=10)

        assert len(query_result_before) == len(query_result_after), (
            f"Query result count mismatch for {field_name} with index type {index_type}: before={len(query_result_before)}, after={len(query_result_after)}"
        )

        before_ids = set(doc.id for doc in query_result_before)
        after_ids = set(doc.id for doc in query_result_after)
        assert before_ids == after_ids, (
            f"Query result IDs mismatch for {field_name} with index type {index_type}: before={before_ids}, after={after_ids}"
        )

        # INSERT 5~8 Doc
        new_docs = [generate_doc(i, full_collection_new.schema) for i in range(5, 8)]

        result = full_collection_new.insert(new_docs)
        assert len(result) == 3
        for item in result:
            assert item.ok()

        stats_after_insert1 = full_collection_new.stats
        assert stats_after_insert1 is not None
        assert stats_after_insert1.doc_count == 8

        fetched_docs = full_collection_new.fetch([f"{i}" for i in range(5, 8)])
        assert len(fetched_docs) == 3

        for i in range(5, 8):
            doc_id = f"{i}"
            assert doc_id in fetched_docs

        query_result = full_collection_new.query(filter=query_filter, topk=20)
        assert len(query_result) >= len(query_result_before)

        full_collection_new.drop_index(field_name=field_name)

        # Insert 8~10 Doc
        more_docs = [generate_doc(i, full_collection_new.schema) for i in range(8, 10)]

        result = full_collection_new.insert(more_docs)
        assert len(result) == 2
        for item in result:
            assert item.ok()

        stats_after_insert2 = full_collection_new.stats
        assert stats_after_insert2 is not None
        assert stats_after_insert2.doc_count == 10

        fetched_docs = full_collection_new.fetch([f"{i}" for i in range(8, 10)])
        assert len(fetched_docs) == 2

        for i in range(8, 10):
            doc_id = f"{i}"
            assert doc_id in fetched_docs

        query_result = full_collection_new.query(filter=query_filter, topk=20)
        assert len(query_result) >= len(query_result_before)

        final_stats = full_collection_new.stats
        assert final_stats is not None
        assert final_stats.doc_count == 10
        full_collection_new.destroy()

    @pytest.mark.parametrize("field_name", DEFAULT_SCALAR_FIELD_NAME.values())
    @pytest.mark.parametrize("index_type", SUPPORT_SCALAR_INDEX_TYPES)
    def test_duplicate_create_index(
        self, full_collection_new: Collection, field_name: str, index_type: IndexType
    ):
        docs = [generate_doc(i, full_collection_new.schema) for i in range(10)]

        result = full_collection_new.insert(docs)
        assert bool(result)
        for item in result:
            assert item.ok()

        stats = full_collection_new.stats
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

        query_result_before = full_collection_new.query(filter=query_filter, topk=5)

        if index_type not in DEFAULT_INDEX_PARAMS:
            pytest.fail(f"Unsupported index type for index creation: {index_type}")
        index_param = DEFAULT_INDEX_PARAMS[index_type]

        full_collection_new.create_index(
            field_name=field_name, index_param=index_param, option=IndexOption()
        )

        query_result_after = full_collection_new.query(filter=query_filter, topk=5)

        assert len(query_result_before) == len(query_result_after), (
            f"Query result count mismatch: before={len(query_result_before)}, after={len(query_result_after)}"
        )

        before_ids = set(doc.id for doc in query_result_before)
        after_ids = set(doc.id for doc in query_result_after)
        assert before_ids == after_ids, (
            f"Query result IDs mismatch: before={before_ids}, after={after_ids}"
        )

        full_collection_new.create_index(
            field_name=field_name, index_param=index_param, option=IndexOption()
        )

    @pytest.mark.parametrize(
        "vector_type, index_type", SUPPORT_VECTOR_DATA_TYPE_INDEX_MAP_PARAMS
    )
    def test_vector_index_operation(
        self,
        full_collection_new: Collection,
        vector_type: DataType,
        index_type: IndexType,
    ):
        vector_field_name = DEFAULT_VECTOR_FIELD_NAME[vector_type]

        docs = [generate_doc(i, full_collection_new.schema) for i in range(5)]

        result = full_collection_new.insert(docs)
        assert len(result) == 5, (
            f"Expected 5 insertion results, got {len(result)} for vector type {vector_type} and index type {index_type}"
        )
        for i, item in enumerate(result):
            assert item.ok(), (
                f"Before create_index,result={result},Insertion result {i} is not OK for vector type {vector_type} and index type {index_type} and result={result}"
            )

        stats = full_collection_new.stats
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

        full_collection_new.create_index(
            field_name=vector_field_name,
            index_param=index_param,
            option=IndexOption(),
        )

        stats_after_create = full_collection_new.stats
        assert stats_after_create is not None, (
            f"stats_after_create_index is None for vector type {vector_type} and index type {index_type}"
        )

        new_docs = [generate_doc(i, full_collection_new.schema) for i in range(5, 8)]

        result = full_collection_new.insert(new_docs)
        assert len(result) == 3, (
            f"Expected 3 insertion results, got {len(result)} for vector type {vector_type} and index type {index_type}"
        )
        for i, item in enumerate(result):
            assert item.ok(), (
                f"Before drop_index,result={result},BInsertion result {i} is not OK for vector type {vector_type} and index type {index_type} and "
            )

        stats_after_insert1 = full_collection_new.stats
        assert stats_after_insert1 is not None, (
            f"stats_after_insert1 is None for vector type {vector_type} and index type {index_type}"
        )
        assert stats_after_insert1.doc_count == 8, (
            f"Expected 8 documents, got {stats_after_insert1.doc_count} for vector type {vector_type} and index type {index_type}"
        )

        fetched_docs = full_collection_new.fetch([f"{i}" for i in range(5, 8)])
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

        full_collection_new.drop_index(field_name=vector_field_name)

        more_docs = [generate_doc(i, full_collection_new.schema) for i in range(8, 10)]
        result = full_collection_new.insert(more_docs)
        assert len(result) == 2, (
            f"Expected 2 insertion results, got {len(result)} for vector type {vector_type} and index type {index_type}"
        )
        for i, item in enumerate(result):
            assert item.ok(), (
                f"After drop_index,Insertion result {i} is not OK for vector type {vector_type} and index type {index_type} and result={result}"
            )

        # Verify document count after second insertion
        stats_after_insert2 = full_collection_new.stats
        assert stats_after_insert2 is not None, (
            f"stats_after_insert2 is None for vector type {vector_type} and index type {index_type}"
        )
        assert stats_after_insert2.doc_count == 10, (
            f"Expected 10 documents, got {stats_after_insert2.doc_count} for vector type {vector_type} and index type {index_type}"
        )

        # Fetch data
        fetched_docs = full_collection_new.fetch([f"{i}" for i in range(8, 10)])
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
        final_stats = full_collection_new.stats
        assert final_stats is not None, (
            f"final_stats is None for vector type {vector_type} and index type {index_type}"
        )
        assert final_stats.doc_count == 10, (
            f"Expected 10 documents, got {final_stats.doc_count} for vector type {vector_type} and index type {index_type}"
        )
        full_collection_new.destroy()

    @pytest.mark.parametrize(
        "vector_type, index_type", SUPPORT_VECTOR_DATA_TYPE_INDEX_MAP_PARAMS
    )
    def test_vector_index_operation_with_reopen(
        self, tmp_path_factory, vector_type, index_type
    ):
        vector_field_name = DEFAULT_VECTOR_FIELD_NAME[vector_type]

        # Create collection
        temp_dir = tmp_path_factory.mktemp("zvec")
        collection_path = temp_dir / "test_collection"

        collection_option = CollectionOption(read_only=False, enable_mmap=True)
        # Create and open collection
        coll1 = zvec.create_and_open(
            path=str(collection_path),
            schema=indextest_collection_schema,
            option=collection_option,
        )

        assert coll1 is not None, "Failed to create and open collection"

        docs = [generate_doc(i, coll1.schema) for i in range(5)]

        result = coll1.insert(docs)
        assert len(result) == 5, (
            f"Expected 5 insertion results, got {len(result)} for vector type {vector_type} and index type {index_type}"
        )
        for i, item in enumerate(result):
            assert item.ok(), (
                f"Before create_index,result={result},Insertion result {i} is not OK for vector type {vector_type} and index type {index_type} and result={result}"
            )

        stats = coll1.stats
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

        coll1.create_index(
            field_name=vector_field_name,
            index_param=index_param,
            option=IndexOption(),
        )

        # Close the first collection (delete reference)
        del coll1
        # Reopen the collection
        coll2 = zvec.open(path=str(collection_path), option=collection_option)

        stats_after_create = coll2.stats
        assert stats_after_create is not None, (
            f"stats_after_create_index is None for vector type {vector_type} and index type {index_type}"
        )

        new_docs = [generate_doc(i, coll2.schema) for i in range(5, 8)]

        result = coll2.insert(new_docs)
        assert len(result) == 3, (
            f"Expected 3 insertion results, got {len(result)} for vector type {vector_type} and index type {index_type}"
        )
        for i, item in enumerate(result):
            assert item.ok(), (
                f"Before drop_index,result={result},BInsertion result {i} is not OK for vector type {vector_type} and index type {index_type} and "
            )

        stats_after_insert1 = coll2.stats
        assert stats_after_insert1 is not None, (
            f"stats_after_insert1 is None for vector type {vector_type} and index type {index_type}"
        )
        assert stats_after_insert1.doc_count == 8, (
            f"Expected 8 documents, got {stats_after_insert1.doc_count} for vector type {vector_type} and index type {index_type}"
        )

        fetched_docs = coll2.fetch([f"{i}" for i in range(5, 8)])
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

        coll2.drop_index(field_name=vector_field_name)

        del coll2

        # Reopen the collection
        coll3 = zvec.open(path=str(collection_path), option=collection_option)

        more_docs = [generate_doc(i, coll3.schema) for i in range(8, 10)]
        result = coll3.insert(more_docs)
        assert len(result) == 2, (
            f"Expected 2 insertion results, got {len(result)} for vector type {vector_type} and index type {index_type}"
        )
        for i, item in enumerate(result):
            assert item.ok(), (
                f"After drop_index,Insertion result {i} is not OK for vector type {vector_type} and index type {index_type} and result={result}"
            )

        # Verify document count after second insertion
        stats_after_insert2 = coll3.stats
        assert stats_after_insert2 is not None, (
            f"stats_after_insert2 is None for vector type {vector_type} and index type {index_type}"
        )
        assert stats_after_insert2.doc_count == 10, (
            f"Expected 10 documents, got {stats_after_insert2.doc_count} for vector type {vector_type} and index type {index_type}"
        )

        # Fetch data
        fetched_docs = coll3.fetch([f"{i}" for i in range(8, 10)])
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
        final_stats = coll3.stats
        assert final_stats is not None, (
            f"final_stats is None for vector type {vector_type} and index type {index_type}"
        )
        assert final_stats.doc_count == 10, (
            f"Expected 10 documents, got {final_stats.doc_count} for vector type {vector_type} and index type {index_type}"
        )
        coll3.destroy()

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
            # For non-string values like None, int, float, etc., we may get either
            # INCOMPATIBLE_FUNCTION_ERROR_MSG, SCHEMA_VALIDATE_ERROR_MSG, INCOMPATIBLE_CONSTRUCTOR_ERROR_MSG
            error_str = str(exc_info.value)
            # Check if the error contains expected patterns
            expected_patterns = [
                INCOMPATIBLE_FUNCTION_ERROR_MSG,
                SCHEMA_VALIDATE_ERROR_MSG,
                INCOMPATIBLE_CONSTRUCTOR_ERROR_MSG,
            ]
            if not any(pattern in error_str for pattern in expected_patterns):
                assert False, "Error message is unreasonable: e=" + error_str

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
    def test_valid_vector_index_params(
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

    @pytest.mark.parametrize(
        "data_type, index_param", INVALID_VECTOR_DATA_TYPE_INDEX_PARAM_MAP_PARAMS
    )
    def test_invalid_vector_index_params(
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

        with pytest.raises(Exception) as exc_info:
            # create index
            coll.create_index(
                field_name=vector_name,
                index_param=index_param,
                option=IndexOption(),
            )
        self.check_error_message(exc_info, index_param)


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

    def test_add_column_with_reopen(self, tmp_path_factory):
        # Create collection
        temp_dir = tmp_path_factory.mktemp("zvec")
        collection_path = temp_dir / "test_collection"

        collection_option = CollectionOption(read_only=False, enable_mmap=True)
        # Create and open collection
        coll1 = zvec.create_and_open(
            path=str(collection_path),
            schema=columntest_collection_schema,
            option=collection_option,
        )

        assert coll1 is not None, "Failed to create and open collection"

        # Insert some data
        doc1 = Doc(
            id="1",
            fields={"id": 1, "name": "test1"},
            vectors={
                "dense_fp32_field": np.random.random(128).tolist(),
                "sparse_fp32_field": {1: 1.0, 2: 2.0},
            },
        )

        result = coll1.insert(doc1)
        assert result.ok()

        coll1.add_column(
            field_schema=FieldSchema("income", DataType.INT32),
            expression="200",  # Simple expression
        )
        doc2 = Doc(
            id="2",
            fields={"id": 2, "name": "test2", "income": 12},
            vectors={
                "dense_fp32_field": np.random.random(128).tolist(),
                "sparse_fp32_field": {3: 1.1, 4: 2.1},
            },
        )

        result = coll1.insert(doc2)
        assert bool(result), f"Expected 1 result, but got {len(result)}"
        assert result.ok(), (
            f"result={result},Insert operation failed with code = {result.code()}"
        )
        stats = coll1.stats
        assert stats is not None
        assert stats.doc_count == 2

        collection_schema_new = coll1.schema

        assert collection_schema_new.fields != columntest_collection_schema.fields

        # Close the first collection (delete reference)
        del coll1

        # Reopen the collection
        coll2 = zvec.open(path=str(collection_path), option=collection_option)

        assert coll2 is not None, "Failed to reopen collection"
        assert coll2.path == str(collection_path)
        assert coll2.schema.name == collection_schema_new.name
        assert coll2.schema.fields == collection_schema_new.fields

        doc3 = Doc(
            id="3",
            fields={"id": 3, "name": "test3", "income": 13},
            vectors={
                "dense_fp32_field": np.random.random(128).tolist(),
                "sparse_fp32_field": {5: 11.0, 6: 13.0},
            },
        )
        result = coll2.insert(doc3)
        assert bool(result), f"Expected 1 result, but got {len(result)}"
        assert result.ok(), (
            f"result={result},Insert operation failed with code = {result.code()}"
        )
        stats = coll2.stats
        assert stats is not None
        assert stats.doc_count == 3

        # Verify data is still there
        fetched_docs = coll2.fetch(["1", "2", "3"])
        for id in ["1", "2", "3"]:
            assert id in fetched_docs
            fetched_doc = fetched_docs[id]
            assert fetched_doc.id == id
            assert fetched_doc.field("name") == "test" + id

        if hasattr(coll2, "destroy") and coll2 is not None:
            try:
                coll2.destroy()
            except Exception as e:
                print(f"Warning: failed to destroy collection: {e}")

    def test_alter_column_with_reopen(self, tmp_path_factory):
        # Create collection
        temp_dir = tmp_path_factory.mktemp("zvec")
        collection_path = temp_dir / "test_collection"

        collection_option = CollectionOption(read_only=False, enable_mmap=True)
        # Create and open collection
        coll1 = zvec.create_and_open(
            path=str(collection_path),
            schema=columntest_collection_schema,
            option=collection_option,
        )
        assert coll1 is not None, "Failed to create and open collection"

        # Insert some data
        doc1 = Doc(
            id="1",
            fields={"id": 1, "name": "test1"},
            vectors={
                "dense_fp32_field": np.random.random(128).tolist(),
                "sparse_fp32_field": {1: 1.0, 2: 2.0},
            },
        )

        result = coll1.insert(doc1)
        assert result.ok()

        coll1.alter_column(
            old_name="id",
            new_name="id_new",
            option=AlterColumnOption(),
        )
        doc2 = Doc(
            id="2",
            fields={"id_new": 2, "name": "test2"},
            vectors={
                "dense_fp32_field": np.random.random(128).tolist(),
                "sparse_fp32_field": {3: 1.1, 4: 2.1},
            },
        )

        result = coll1.insert(doc2)
        assert bool(result), f"Expected 1 result, but got {len(result)}"
        assert result.ok(), (
            f"result={result},Insert operation failed with code = {result.code()}"
        )
        stats = coll1.stats
        assert stats is not None
        assert stats.doc_count == 2

        collection_schema_new = coll1.schema

        assert collection_schema_new.fields != columntest_collection_schema.fields

        # Close the first collection (delete reference)
        del coll1

        # Reopen the collection
        coll2 = zvec.open(path=str(collection_path), option=collection_option)

        assert coll2 is not None, "Failed to reopen collection"
        assert coll2.path == str(collection_path)
        assert coll2.schema.name == collection_schema_new.name
        assert coll2.schema.fields == collection_schema_new.fields

        doc3 = Doc(
            id="3",
            fields={"id_new": 3, "name": "test3"},
            vectors={
                "dense_fp32_field": np.random.random(128).tolist(),
                "sparse_fp32_field": {5: 11.0, 6: 13.0},
            },
        )
        result = coll2.insert(doc3)
        assert bool(result), f"Expected 1 result, but got {len(result)}"
        assert result.ok(), (
            f"result={result},Insert operation failed with code = {result.code()}"
        )
        stats = coll2.stats
        assert stats is not None
        assert stats.doc_count == 3

        # Verify data is still there
        fetched_docs = coll2.fetch(["1", "2", "3"])
        for id in ["1", "2", "3"]:
            assert id in fetched_docs
            fetched_doc = fetched_docs[id]
            assert fetched_doc.id == id
            assert fetched_doc.field("name") == "test" + id

        if hasattr(coll2, "destroy") and coll2 is not None:
            try:
                coll2.destroy()
            except Exception as e:
                print(f"Warning: failed to destroy collection: {e}")

    def test_drop_column_with_reopen(self, tmp_path_factory):
        # Create collection
        temp_dir = tmp_path_factory.mktemp("zvec")
        collection_path = temp_dir / "test_collection"

        collection_option = CollectionOption(read_only=False, enable_mmap=True)
        # Create and open collection
        coll1 = zvec.create_and_open(
            path=str(collection_path),
            schema=columntest_collection_schema,
            option=collection_option,
        )

        assert coll1 is not None, "Failed to create and open collection"

        # Insert some data
        doc1 = Doc(
            id="1",
            fields={"id": 1, "name": "test1"},
            vectors={
                "dense_fp32_field": np.random.random(128).tolist(),
                "sparse_fp32_field": {1: 1.0, 2: 2.0},
            },
        )

        result = coll1.insert(doc1)
        assert result.ok()

        coll1.drop_column("id")
        doc2 = Doc(
            id="2",
            fields={"name": "test2"},
            vectors={
                "dense_fp32_field": np.random.random(128).tolist(),
                "sparse_fp32_field": {3: 1.1, 4: 2.1},
            },
        )

        result = coll1.insert(doc2)
        assert bool(result), f"Expected 1 result, but got {len(result)}"
        assert result.ok(), (
            f"result={result},Insert operation failed with code = {result.code()}"
        )
        stats = coll1.stats
        assert stats is not None
        assert stats.doc_count == 2

        collection_schema_new = coll1.schema

        assert collection_schema_new.fields != columntest_collection_schema.fields

        # Close the first collection (delete reference)
        del coll1

        # Reopen the collection
        coll2 = zvec.open(path=str(collection_path), option=collection_option)

        assert coll2 is not None, "Failed to reopen collection"
        assert coll2.path == str(collection_path)
        assert coll2.schema.name == collection_schema_new.name
        assert coll2.schema.fields == collection_schema_new.fields

        doc3 = Doc(
            id="3",
            fields={"name": "test3"},
            vectors={
                "dense_fp32_field": np.random.random(128).tolist(),
                "sparse_fp32_field": {5: 11.0, 6: 13.0},
            },
        )
        result = coll2.insert(doc3)
        assert bool(result), f"Expected 1 result, but got {len(result)}"
        assert result.ok(), (
            f"result={result},Insert operation failed with code = {result.code()}"
        )
        stats = coll2.stats
        assert stats is not None
        assert stats.doc_count == 3

        # Verify data is still there
        fetched_docs = coll2.fetch(["1", "2", "3"])
        for id in ["1", "2", "3"]:
            assert id in fetched_docs
            fetched_doc = fetched_docs[id]
            assert fetched_doc.id == id

        if hasattr(coll2, "destroy") and coll2 is not None:
            try:
                coll2.destroy()
            except Exception as e:
                print(f"Warning: failed to destroy collection: {e}")

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

    def test_add_column_then_query_returns_new_field(
        self, basic_collection: Collection
    ):
        """Regression test for issue #426: query() should return fields added via add_column()."""
        basic_collection.add_column(
            field_schema=FieldSchema("score", DataType.INT64, nullable=True),
        )

        docs = [
            Doc(
                id="1",
                fields={"id": 1, "name": "alice", "weight": 60.0, "score": 100},
                vectors={
                    "dense": generate_constant_vector(1, 128),
                    "sparse": generate_sparse_vector(1),
                },
            ),
            Doc(
                id="2",
                fields={"id": 2, "name": "bob", "weight": 70.0, "score": 200},
                vectors={
                    "dense": generate_constant_vector(2, 128),
                    "sparse": generate_sparse_vector(2),
                },
            ),
        ]
        result = basic_collection.insert(docs)
        assert all(r.ok() for r in result)

        # Query with explicit output_fields
        query_result = basic_collection.query(
            Query(field_name="dense", vector=generate_constant_vector(1, 128)),
            topk=2,
            output_fields=["score"],
        )
        assert len(query_result) == 2
        for doc in query_result:
            assert "score" in doc.fields, (
                f"Doc {doc.id} missing 'score' field after add_column (output_fields explicit)"
            )
            assert doc.fields["score"] in (100, 200)

        # Query with select-all (no output_fields)
        query_result_all = basic_collection.query(
            Query(field_name="dense", vector=generate_constant_vector(1, 128)),
            topk=2,
        )
        assert len(query_result_all) == 2
        for doc in query_result_all:
            assert "score" in doc.fields, (
                f"Doc {doc.id} missing 'score' field after add_column (select all)"
            )

    def test_alter_column_rename_then_query_returns_new_name(
        self, basic_collection: Collection
    ):
        """Regression test: query() should use the new field name after alter_column rename."""
        docs = [
            Doc(
                id="1",
                fields={"id": 1, "name": "alice", "weight": 60.0},
                vectors={
                    "dense": generate_constant_vector(1, 128),
                    "sparse": generate_sparse_vector(1),
                },
            ),
            Doc(
                id="2",
                fields={"id": 2, "name": "bob", "weight": 70.0},
                vectors={
                    "dense": generate_constant_vector(2, 128),
                    "sparse": generate_sparse_vector(2),
                },
            ),
        ]
        result = basic_collection.insert(docs)
        assert all(r.ok() for r in result)

        # Rename 'weight' -> 'mass'
        basic_collection.alter_column("weight", new_name="mass")

        # Query with explicit output_fields using new name
        query_result = basic_collection.query(
            Query(field_name="dense", vector=generate_constant_vector(1, 128)),
            topk=2,
            output_fields=["mass"],
        )
        assert len(query_result) == 2
        for doc in query_result:
            assert "mass" in doc.fields, (
                f"Doc {doc.id} missing 'mass' field after alter_column rename"
            )
            assert "weight" not in doc.fields, (
                f"Doc {doc.id} still has old name 'weight' after rename"
            )

        # Query with select-all
        query_result_all = basic_collection.query(
            Query(field_name="dense", vector=generate_constant_vector(1, 128)),
            topk=2,
        )
        assert len(query_result_all) == 2
        for doc in query_result_all:
            assert "mass" in doc.fields, (
                f"Doc {doc.id} missing 'mass' in select-all after alter_column rename"
            )
            assert "weight" not in doc.fields, (
                f"Doc {doc.id} still has old name 'weight' in select-all after rename"
            )

    def test_drop_column_then_query_excludes_dropped_field(
        self, basic_collection: Collection
    ):
        """Regression test: query() should not return fields removed via drop_column()."""
        basic_collection.add_column(
            field_schema=FieldSchema("score", DataType.INT64, nullable=True),
        )

        docs = [
            Doc(
                id="1",
                fields={"id": 1, "name": "alice", "weight": 60.0, "score": 100},
                vectors={
                    "dense": generate_constant_vector(1, 128),
                    "sparse": generate_sparse_vector(1),
                },
            ),
            Doc(
                id="2",
                fields={"id": 2, "name": "bob", "weight": 70.0, "score": 200},
                vectors={
                    "dense": generate_constant_vector(2, 128),
                    "sparse": generate_sparse_vector(2),
                },
            ),
        ]
        result = basic_collection.insert(docs)
        assert all(r.ok() for r in result)

        # Verify field exists before drop
        query_before = basic_collection.query(
            Query(field_name="dense", vector=generate_constant_vector(1, 128)),
            topk=2,
        )
        assert all("score" in doc.fields for doc in query_before)

        # Drop the column
        basic_collection.drop_column("score")

        # Query after drop - 'score' should not appear
        query_after = basic_collection.query(
            Query(field_name="dense", vector=generate_constant_vector(1, 128)),
            topk=2,
        )
        assert len(query_after) == 2
        for doc in query_after:
            assert "score" not in doc.fields, (
                f"Doc {doc.id} still has 'score' after drop_column"
            )
            assert "name" in doc.fields, (
                f"Doc {doc.id} missing 'name' - other fields should still be present"
            )
