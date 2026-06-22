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
"""End-to-end tests for FTS-only collections (no vector field).

The schema validation rule "must have at least one vector field" has been
lifted; these tests pin the new behavior so insert / query / delete /
optimize all work on a vector-less collection.
"""

from __future__ import annotations

import pytest
import zvec
from zvec import (
    Collection,
    CollectionOption,
    DataType,
    Doc,
    FieldSchema,
    FtsIndexParam,
    OptimizeOption,
)
from zvec.model.param.query import Fts, Query


# ==================== Fixtures ====================


@pytest.fixture(scope="function")
def fts_collection(tmp_path_factory) -> Collection:
    """FTS-only collection: a STRING field for forward + an FTS-indexed STRING."""
    temp_dir = tmp_path_factory.mktemp("zvec_fts_only")
    collection_path = temp_dir / "fts_collection"

    schema = zvec.CollectionSchema(
        name="fts_only",
        fields=[
            FieldSchema("title", DataType.STRING, nullable=False),
            FieldSchema(
                "content",
                DataType.STRING,
                nullable=False,
                index_param=FtsIndexParam(
                    tokenizer_name="standard",
                    filters=["lowercase"],
                ),
            ),
        ],
        # vectors omitted on purpose — schema validation must accept this.
    )

    coll = zvec.create_and_open(
        path=str(collection_path),
        schema=schema,
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


def _make_docs() -> list[Doc]:
    """5-doc corpus where 4 contain 'hello' and doc 4 is the only outlier."""
    return [
        Doc(id="pk_0", fields={"title": "intro", "content": "hello world"}),
        Doc(id="pk_1", fields={"title": "guide", "content": "hello foo bar"}),
        Doc(id="pk_2", fields={"title": "tips", "content": "hello baz"}),
        Doc(id="pk_3", fields={"title": "more", "content": "hello hello"}),
        Doc(id="pk_4", fields={"title": "other", "content": "nothing relevant"}),
    ]


def _fts_query(coll: Collection, term: str) -> list[Doc]:
    """Run a single-term FTS match query against the `content` field."""
    return coll.query(
        queries=Query(field_name="content", fts=Fts(match_string=term)),
        topk=10,
    )


# ==================== Tests ====================


class TestFtsOnlyCollectionSchema:
    def test_create_and_open_without_vectors(self, fts_collection: Collection):
        """Schema with zero vector fields must be accepted by validate()."""
        assert fts_collection.schema.name == "fts_only"
        assert {f.name for f in fts_collection.schema.fields} == {"title", "content"}
        # Empty vectors is the whole point of the test.
        assert list(fts_collection.schema.vectors) == []
        assert fts_collection.stats.doc_count == 0

    def test_create_schema_omitting_vectors_kwarg(self):
        """Constructing CollectionSchema without `vectors=` argument is valid."""
        schema = zvec.CollectionSchema(
            name="bare_fts",
            fields=[
                FieldSchema(
                    "content",
                    DataType.STRING,
                    nullable=False,
                    index_param=FtsIndexParam(),
                ),
            ],
        )
        assert list(schema.vectors) == []
        assert {f.name for f in schema.fields} == {"content"}


class TestFtsOnlyCollectionLifecycle:
    def test_insert_and_fts_query(self, fts_collection: Collection):
        """FTS-only collection supports insert + FTS query end-to-end."""
        results = fts_collection.insert(_make_docs())
        assert all(r.ok() for r in results)
        assert fts_collection.stats.doc_count == 5

        hits = _fts_query(fts_collection, "hello")
        assert len(hits) == 4
        assert {doc.id for doc in hits} == {"pk_0", "pk_1", "pk_2", "pk_3"}

        # Term that nothing in the surviving corpus contains.
        assert _fts_query(fts_collection, "missing_term_xyz") == []

    def test_delete_then_query(self, fts_collection: Collection):
        """Tombstone filter must drop deleted docs from FTS results."""
        fts_collection.insert(_make_docs())
        statuses = fts_collection.delete(["pk_0", "pk_4"])
        assert all(s.ok() for s in statuses)
        assert fts_collection.stats.doc_count == 3

        hits = _fts_query(fts_collection, "hello")
        assert len(hits) == 3
        assert {doc.id for doc in hits} == {"pk_1", "pk_2", "pk_3"}
        # pk_4's unique term is filtered out post-delete.
        assert _fts_query(fts_collection, "nothing") == []

    def test_optimize_rebuilds_fts(self, fts_collection: Collection):
        """Optimize with >30% deletes triggers ReduceFts; recall unchanged."""
        fts_collection.insert(_make_docs())
        # 40% delete ratio — above COMPACT_DELETE_RATIO_THRESHOLD=0.3, so
        # build_compact_task picks the rebuild path and ReduceFts runs.
        fts_collection.delete(["pk_0", "pk_4"])

        before = {doc.id for doc in _fts_query(fts_collection, "hello")}
        assert before == {"pk_1", "pk_2", "pk_3"}

        fts_collection.optimize(option=OptimizeOption())
        assert fts_collection.stats.doc_count == 3

        after = {doc.id for doc in _fts_query(fts_collection, "hello")}
        assert after == before
        assert _fts_query(fts_collection, "nothing") == []


class TestFtsOnlyCollectionQueryValidation:
    def test_vector_query_rejected(self, fts_collection: Collection):
        """Vector query on a no-vector collection must raise."""
        with pytest.raises(ValueError, match="No vector field found"):
            fts_collection.query(
                queries=Query(field_name="content", vector=[0.1, 0.2, 0.3]),
                topk=5,
            )

    def test_id_query_rejected(self, fts_collection: Collection):
        """ID-based query on a no-vector collection must raise."""
        fts_collection.insert(_make_docs()[:1])
        with pytest.raises(ValueError, match="No vector field found"):
            fts_collection.query(
                queries=Query(field_name="content", id="pk_0"),
                topk=5,
            )
