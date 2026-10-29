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
import zvec


@pytest.fixture(scope="module", autouse=True)
def initialize():
    zvec.init(memory_limit_mb=512, query_threads=2, optimize_threads=2)


def _doc(key):
    return zvec.Doc(
        id=f"doc_{key}",
        fields={"value": key, "group_id": key % 3},
        vectors={"v": [float(key)] * 4},
    )


def _check_groups(collection, keys, kind):
    query = zvec.Query(
        "v",
        vector=[1.0] * 4,
        param=zvec.HnswQueryParam(is_linear=True) if kind == "hnsw" else None,
    )
    for filter_expr in [None, "value >= 3"]:
        groups = collection.group_by_query(
            query,
            group_by_field_name="group_id",
            group_count=3,
            topk_per_group=2,
            filter=filter_expr,
            include_vector=True,
        )
        assert len(groups) == 3
        for group in groups:
            expected = sorted(
                (
                    key
                    for key in keys
                    if key % 3 == int(group.group_by_value)
                    and (filter_expr is None or key >= 3)
                ),
                reverse=True,
            )[:2]
            assert [doc.id for doc in group.docs] == [f"doc_{key}" for key in expected]
            for doc, key in zip(group.docs, expected):
                assert doc.fields["value"] == key
                assert doc.vectors["v"] == pytest.approx([float(key)] * 4)
                assert doc.score == pytest.approx(float(key * 4))


@pytest.mark.parametrize("enable_mmap", [True, False])
@pytest.mark.parametrize("read_only", [True, False])
@pytest.mark.parametrize("kind", ["flat", "hnsw"])
@pytest.mark.parametrize("shape", ["unoptimized", "mixed", "optimized"])
def test_group_by_after_reopen(tmp_path, enable_mmap, read_only, kind, shape):
    path = tmp_path / "collection"
    param = zvec.FlatIndexParam if kind == "flat" else zvec.HnswIndexParam
    schema = zvec.CollectionSchema(
        name="group_by_reopen",
        fields=[
            zvec.FieldSchema("group_id", zvec.DataType.INT64),
            zvec.FieldSchema(
                "value", zvec.DataType.INT64, index_param=zvec.InvertIndexParam()
            ),
        ],
        vectors=zvec.VectorSchema(
            "v",
            zvec.DataType.VECTOR_FP32,
            4,
            index_param=param(metric_type=zvec.MetricType.IP),
        ),
    )
    writer = zvec.create_and_open(
        str(path), schema, zvec.CollectionOption(enable_mmap=enable_mmap)
    )
    assert all(status.ok() for status in writer.insert([_doc(i) for i in range(9)]))
    if shape != "unoptimized":
        writer.optimize()
    assert all(status.ok() for status in writer.insert([_doc(i) for i in range(9, 12)]))
    if shape == "optimized":
        writer.optimize()
    _check_groups(writer, range(12), kind)
    writer.close()

    collection = zvec.open(
        str(path), zvec.CollectionOption(read_only=read_only, enable_mmap=enable_mmap)
    )
    try:
        _check_groups(collection, range(12), kind)
        if not read_only:
            assert collection.insert(_doc(12)).ok()
            assert collection.delete("doc_11").ok()
            _check_groups(collection, [*range(11), 12], kind)
    finally:
        collection.close()
