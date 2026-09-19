"""Fast queries must include Uniform segments that have not been trained yet."""

import numpy as np
import pytest

import zvec


@pytest.mark.parametrize("index_kind", ["hnsw", "vamana"])
@pytest.mark.parametrize(
    "quantizer",
    [
        zvec.QuantizeType.UNIFORM_UINT7,
        zvec.QuantizeType.UNIFORM_UINT8,
        zvec.QuantizeType.UNIFORM_UINT4,
    ],
)
@pytest.mark.parametrize(
    "flat_type",
    [zvec.DataType.VECTOR_FP32, zvec.DataType.VECTOR_FP16, zvec.DataType.VECTOR_UINT8],
)
@pytest.mark.parametrize("enable_mmap", [False, True])
def test_untrained_and_mixed_uniform_segments(
    tmp_path, index_kind, quantizer, flat_type, enable_mmap
):
    zvec.init()
    vectors = np.random.default_rng(754).uniform(1, 240, (65, 17)).astype(np.float32)
    index_type, query_type = (
        (zvec.HnswIndexParam, zvec.HnswQueryParam)
        if index_kind == "hnsw"
        else (zvec.VamanaIndexParam, zvec.VamanaQueryParam)
    )
    schema = zvec.CollectionSchema(
        name="fast_uniform_lifecycle",
        vectors=[
            zvec.VectorSchema(
                "vector",
                zvec.DataType.VECTOR_FP32,
                17,
                index_param=index_type(
                    metric_type=zvec.MetricType.L2,
                    quantize_type=quantizer,
                    flat_data_type=flat_type,
                    use_flat_contiguous_memory=True,
                ),
            )
        ],
    )
    path = str(tmp_path / "collection")
    writer_option = zvec.CollectionOption(enable_mmap=enable_mmap)
    writer = zvec.create_and_open(path, schema, writer_option)

    def doc(i):
        return zvec.Doc(id=f"row-{i}", vectors={"vector": vectors[i].tolist()})

    def check_reader(count):
        reader = zvec.open(
            path, zvec.CollectionOption(read_only=True, enable_mmap=enable_mmap)
        )
        try:
            for param in (
                None,
                query_type(),
                query_type(is_using_refiner=True),
                query_type(is_using_refiner=True, scale_factor=2.0),
                query_type(is_linear=True),
                query_type(is_linear=True, is_using_refiner=True),
            ):
                for row in (12, count - 1):
                    query = vectors[row]
                    for topk in (10, count + 3):
                        docs = reader.query(
                            zvec.Query("vector", vector=query, param=param), topk=topk
                        )
                        expected = np.array([int(d.id[4:]) for d in docs])
                        ids, scores = reader.fast_query(
                            "vector", query, param, topk=topk, return_scores=True
                        )
                        assert docs[0].id == f"row-{row}"
                        actual = ids[: len(docs)]
                        assert set(actual) == set(expected)
                        # SQL and fast_query merge segments differently. Both
                        # order by score, without a shared tie-break rule. Only
                        # exactly equal scores may exchange positions.
                        expected_scores = {int(d.id[4:]): d.score for d in docs}
                        np.testing.assert_array_equal(
                            [expected_scores[i] for i in actual],
                            [d.score for d in docs],
                        )
                        np.testing.assert_allclose(
                            scores[: len(docs)],
                            [expected_scores[i] for i in actual],
                            rtol=2e-5,
                            atol=2e-5,
                        )
                        np.testing.assert_array_equal(
                            ids, reader.fast_query("vector", query, param, topk=topk)
                        )
                        assert np.all(ids[len(docs) :] == -1)
                        assert np.all(np.isnan(scores[len(docs) :]))
                        if topk > count:
                            assert set(expected) == set(range(count))
        finally:
            reader.close()

    try:
        assert all(s.ok() for s in writer.insert([doc(i) for i in range(64)]))
        writer.close()
        check_reader(64)
        writer = zvec.open(path, writer_option)
        writer.optimize()
        writer.close()
        check_reader(64)
        writer = zvec.open(path, writer_option)
        assert writer.insert(doc(64)).ok()
        writer.flush()
        writer.close()
        check_reader(65)
        writer = zvec.open(path, writer_option)
        writer.optimize()
        writer.close()
        check_reader(65)
    finally:
        writer.close()
