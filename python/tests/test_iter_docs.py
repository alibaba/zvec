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
"""Tests for Collection.iter_docs (document iterator)."""

from __future__ import annotations

import pytest
import zvec
from zvec import (
    CollectionOption,
    DataType,
    Doc,
    FieldSchema,
    HnswIndexParam,
    VectorSchema,
)


@pytest.fixture(scope="session")
def iter_schema():
    return zvec.CollectionSchema(
        name="iter_test_collection",
        fields=[
            FieldSchema("id", DataType.INT64, nullable=False),
            FieldSchema("name", DataType.STRING, nullable=False),
            FieldSchema("weight", DataType.FLOAT, nullable=True),
        ],
        vectors=[
            VectorSchema(
                "dense",
                DataType.VECTOR_FP32,
                dimension=8,
                index_param=HnswIndexParam(),
            ),
        ],
    )


@pytest.fixture(scope="function")
def iter_collection(tmp_path_factory, iter_schema):
    temp_dir = tmp_path_factory.mktemp("zvec_iter")
    path = temp_dir / "iter_collection"
    coll = zvec.create_and_open(
        path=str(path),
        schema=iter_schema,
        option=CollectionOption(read_only=False, enable_mmap=True),
    )
    assert coll is not None
    try:
        yield coll
    finally:
        try:
            coll.destroy()
        except Exception as e:
            print(f"Warning: failed to destroy collection: {e}")


def _make_docs(n: int) -> list[Doc]:
    return [
        Doc(
            id=f"{i}",
            fields={"id": i, "name": f"name_{i}", "weight": float(i)},
            vectors={"dense": [float(i)] * 8},
        )
        for i in range(n)
    ]


def test_iter_docs_basic(iter_collection):
    """Insert N docs, iterate, verify count + PK + scalar fields."""
    n = 50
    result = iter_collection.insert(_make_docs(n))
    assert bool(result)
    iter_collection.flush()

    seen_ids = set()
    count = 0
    for doc in iter_collection.iter_docs():
        assert isinstance(doc, Doc)
        assert doc.id != ""
        # scalar fields present
        assert doc.field("id") is not None
        assert doc.field("name") is not None
        seen_ids.add(doc.id)
        count += 1

    assert count == n
    assert len(seen_ids) == n
    assert seen_ids == {f"{i}" for i in range(n)}


def test_iter_docs_include_vector(iter_collection):
    """include_vector=True (default) returns vectors of correct dimension."""
    iter_collection.insert(_make_docs(10))
    iter_collection.flush()

    count = 0
    for doc in iter_collection.iter_docs(include_vector=True):
        vec = doc.vector("dense")
        assert vec is not None, f"dense vector missing for {doc.id}"
        assert len(vec) == 8
        count += 1
    assert count == 10


def test_iter_docs_exclude_vector(iter_collection):
    """include_vector=False omits vector fields."""
    iter_collection.insert(_make_docs(10))
    iter_collection.flush()

    count = 0
    for doc in iter_collection.iter_docs(include_vector=False):
        # scalar present, vector absent
        assert doc.field("id") is not None
        # Doc.vector() returns {} (falsy) when no vectors are present.
        assert not doc.vector("dense")
        assert "dense" not in doc.vector_names()
        count += 1
    assert count == 10


def test_iter_docs_output_fields(iter_collection):
    """output_fields limits returned scalar fields."""
    iter_collection.insert(_make_docs(10))
    iter_collection.flush()

    for doc in iter_collection.iter_docs(output_fields=["id"], include_vector=False):
        assert doc.field("id") is not None
        assert not doc.has_field("name")
        assert not doc.has_field("weight")


def test_iter_docs_empty(iter_collection):
    """Empty collection yields nothing."""
    docs = list(iter_collection.iter_docs())
    assert docs == []


def test_iter_docs_after_delete(iter_collection):
    """Deleted docs must not appear in iteration."""
    iter_collection.insert(_make_docs(20))
    # delete even ids
    to_delete = [f"{i}" for i in range(0, 20, 2)]
    iter_collection.delete(to_delete)
    iter_collection.flush()

    deleted = set(to_delete)
    ids = []
    for doc in iter_collection.iter_docs():
        assert doc.id not in deleted
        ids.append(doc.id)
    assert len(ids) == 10


def test_iter_docs_isolation(iter_collection):
    """Docs written after the iterator is created are not visible."""
    iter_collection.insert(_make_docs(10))
    iter_collection.flush()

    it = iter_collection.iter_docs()
    # Consume the first doc so the snapshot is established.
    first = next(it)
    assert first is not None

    # Insert more docs after the iterator started.
    iter_collection.insert(
        [
            Doc(
                id=f"new_{i}",
                fields={"id": 1000 + i, "name": "new", "weight": 1.0},
                vectors={"dense": [1.0] * 8},
            )
            for i in range(5)
        ]
    )
    iter_collection.flush()

    # Count remaining from the original snapshot (should be 9, total 10).
    remaining = sum(1 for _ in it)
    assert remaining == 9

    # A fresh iterator sees all 15.
    assert sum(1 for _ in iter_collection.iter_docs()) == 15


def test_iter_docs_is_generator(iter_collection):
    """iter_docs returns a lazy generator (constant memory)."""
    iter_collection.insert(_make_docs(5))
    iter_collection.flush()

    gen = iter_collection.iter_docs()
    # It is an iterator: next() works and StopIteration terminates it.
    got = [next(gen) for _ in range(5)]
    assert len(got) == 5
    with pytest.raises(StopIteration):
        next(gen)
