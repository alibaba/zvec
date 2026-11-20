# Copyright 2025-present the zvec project
# Licensed under the Apache License, Version 2.0.
"""RaBitQ through the ordinary IVF public API, including collection rebuilds."""

import platform
import sys

import pytest
import zvec

pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or platform.machine() not in ("x86_64", "AMD64"),
    reason="RaBitQ requires Linux x86_64",
)


@pytest.mark.parametrize(
    "metric", [zvec.MetricType.L2, zvec.MetricType.IP, zvec.MetricType.COSINE]
)
@pytest.mark.parametrize("integrated", [True, False], ids=["ivf", "legacy"])
def test_ivf_rabitq_optimize_reopen_and_filter(tmp_path, metric, integrated):
    import random
    import math

    rng = random.Random(8121)
    vectors = [[rng.gauss(0, 1) for _ in range(128)] for _ in range(1200)]
    for v in vectors:
        norm = math.sqrt(sum(x * x for x in v))
        v[:] = [x / norm for x in v]
    path = str(tmp_path / "ivf_rabitq")
    index_param = (
        zvec.IVFIndexParam(
            metric_type=metric,
            n_list=16,
            n_iters=20,
            quantize_type=zvec.QuantizeType.RABITQ,
            total_bits=7,
            sample_count=512,
        )
        if integrated
        else zvec.IvfRabitqIndexParam(
            metric_type=metric,
            nlist=16,
            total_bits=7,
            sample_count=512,
        )
    )
    query_cls = zvec.IVFQueryParam if integrated else zvec.IvfRabitqQueryParam
    schema = zvec.CollectionSchema(
        name="ivf_quantized",
        fields=[
            zvec.FieldSchema("ordinal", zvec.DataType.INT64),
            zvec.FieldSchema("group_id", zvec.DataType.INT64),
        ],
        vectors=[
            zvec.VectorSchema(
                "embedding",
                zvec.DataType.VECTOR_FP32,
                dimension=128,
                index_param=index_param,
            )
        ],
    )
    coll = zvec.create_and_open(path=path, schema=schema)
    try:
        for start in range(0, len(vectors), 200):
            statuses = coll.insert(
                [
                    zvec.Doc(
                        id=str(i),
                        fields={"ordinal": i, "group_id": i % 4},
                        vectors={"embedding": vectors[i]},
                    )
                    for i in range(start, min(start + 200, len(vectors)))
                ]
            )
            assert all(status.ok() for status in statuses)
        # Validate even while queries still use the raw Flat fallback.
        for nprobe in (-1, 0):
            invalid = zvec.Query(
                field_name="embedding",
                vector=vectors[17],
                param=query_cls(nprobe=nprobe),
            )
            with pytest.raises(Exception, match="nprobe"):
                coll.query(queries=invalid, topk=5)
        coll.optimize()
        assert coll.stats.index_completeness["embedding"] == 1
        query = zvec.Query(
            field_name="embedding", vector=vectors[17], param=query_cls(nprobe=16)
        )
        before = coll.query(queries=query, topk=5)
        assert before[0].id == "17"
        filtered = coll.query(queries=query, topk=5, filter="ordinal >= 600")
        assert filtered and all(int(doc.id) >= 600 for doc in filtered)

        def check_group_and_refiner():
            groups = coll.group_by_query(
                query,
                group_by_field_name="group_id",
                group_count=4,
                topk_per_group=2,
                filter="ordinal >= 600",
                include_vector=True,
            )
            assert {int(g.group_by_value) for g in groups} == set(range(4))
            for group in groups:
                assert len(group.docs) == 2
                for doc in group.docs:
                    assert int(doc.id) >= 600
                    assert doc.field("group_id") == int(group.group_by_value)
                    assert doc.vector("embedding") == pytest.approx(
                        vectors[int(doc.id)]
                    )
            refined = coll.query(
                queries=zvec.Query(
                    field_name="embedding",
                    vector=vectors[17],
                    param=query_cls(nprobe=16, is_using_refiner=True, scale_factor=4),
                ),
                topk=5,
                include_vector=True,
            )
            assert refined[0].id == "17"
            assert refined[0].vector("embedding") == pytest.approx(vectors[17])
            # Group state must not leak into the following ordinary query.
            actual_ids = [d.id for d in coll.query(queries=query, topk=5)]
            assert actual_ids == [d.id for d in before]

        check_group_and_refiner()
        coll.close()
        coll = zvec.open(path=path)
        params = coll.schema.vector("embedding").index_param
        assert params.type == (
            zvec.IndexType.IVF if integrated else zvec.IndexType.IVF_RABITQ
        )
        assert params.quantize_type == zvec.QuantizeType.RABITQ
        assert params.total_bits == 7
        assert params.sample_count == 512
        after = coll.query(queries=query, topk=5)
        assert [doc.id for doc in after] == [doc.id for doc in before]
        check_group_and_refiner()
    finally:
        coll.close()
