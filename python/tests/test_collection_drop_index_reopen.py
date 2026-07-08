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

import gc
from pathlib import Path

import zvec
from zvec import (
    CollectionOption,
    DataType,
    Doc,
    FieldSchema,
    FtsIndexParam,
    HnswIndexParam,
    InvertIndexParam,
    OptimizeOption,
    Query,
    VectorSchema,
)


def _vector(i: int) -> list[float]:
    return [float(i % 3), 0.1, 0.2, 0.3]


def _assert_vector_query_works(collection) -> None:
    result = collection.query(Query(field_name="embedding", vector=_vector(0)))
    assert len(result) > 0


def test_drop_one_fts_index_after_optimize_can_reopen(tmp_path: Path):
    collection_path = tmp_path / "drop_fts_reopen"
    schema = zvec.CollectionSchema(
        name="drop_fts_reopen",
        fields=[
            FieldSchema(
                "text",
                DataType.STRING,
                nullable=False,
                index_param=FtsIndexParam(),
            ),
            FieldSchema(
                "other_text",
                DataType.STRING,
                nullable=False,
                index_param=FtsIndexParam(),
            ),
        ],
        vectors=[
            VectorSchema(
                "embedding",
                DataType.VECTOR_FP32,
                dimension=4,
                index_param=HnswIndexParam(),
            )
        ],
    )
    docs = [
        Doc(
            id=f"pk_{i}",
            fields={
                "text": f"needle text {i}",
                "other_text": f"other needle {i}",
            },
            vectors={"embedding": _vector(i)},
        )
        for i in range(20)
    ]

    coll = zvec.create_and_open(
        path=str(collection_path),
        schema=schema,
        option=CollectionOption(read_only=False, enable_mmap=True),
    )
    assert all(r.ok() for r in coll.insert(docs))
    coll.optimize(OptimizeOption())
    del coll
    gc.collect()

    coll = zvec.open(
        path=str(collection_path),
        option=CollectionOption(read_only=False, enable_mmap=True),
    )
    coll.drop_index("text")
    del coll
    gc.collect()

    reopened = zvec.open(
        path=str(collection_path),
        option=CollectionOption(read_only=False, enable_mmap=True),
    )
    assert reopened.stats.doc_count == 20
    assert reopened.schema.field("text").index_param is None
    assert reopened.schema.field("other_text").index_param is not None
    _assert_vector_query_works(reopened)


def test_drop_one_inverted_index_after_optimize_can_reopen(tmp_path: Path):
    collection_path = tmp_path / "drop_invert_reopen"
    schema = zvec.CollectionSchema(
        name="drop_invert_reopen",
        fields=[
            FieldSchema(
                "tag",
                DataType.STRING,
                nullable=False,
                index_param=InvertIndexParam(),
            ),
            FieldSchema(
                "category",
                DataType.STRING,
                nullable=False,
                index_param=InvertIndexParam(),
            ),
        ],
        vectors=[
            VectorSchema(
                "embedding",
                DataType.VECTOR_FP32,
                dimension=4,
                index_param=HnswIndexParam(),
            )
        ],
    )
    docs = [
        Doc(
            id=f"pk_{i}",
            fields={"tag": f"tag_{i % 5}", "category": f"cat_{i % 3}"},
            vectors={"embedding": _vector(i)},
        )
        for i in range(20)
    ]

    coll = zvec.create_and_open(
        path=str(collection_path),
        schema=schema,
        option=CollectionOption(read_only=False, enable_mmap=True),
    )
    assert all(r.ok() for r in coll.insert(docs))
    coll.optimize(OptimizeOption())
    del coll
    gc.collect()

    coll = zvec.open(
        path=str(collection_path),
        option=CollectionOption(read_only=False, enable_mmap=True),
    )
    coll.drop_index("tag")
    del coll
    gc.collect()

    reopened = zvec.open(
        path=str(collection_path),
        option=CollectionOption(read_only=False, enable_mmap=True),
    )
    assert reopened.stats.doc_count == 20
    assert reopened.schema.field("tag").index_param is None
    assert reopened.schema.field("category").index_param is not None
    _assert_vector_query_works(reopened)
