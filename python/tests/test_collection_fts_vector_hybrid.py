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
"""Tests for FTS + vector hybrid retrieval via multi-query with reranker."""

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
    HnswIndexParam,
    VectorSchema,
)
from zvec.extension.multi_vector_reranker import RrfReRanker, WeightedReRanker
from zvec.model.param.query import Fts, Query


DIM = 16


# ==================== Fixtures ====================


@pytest.fixture(scope="function")
def hybrid_collection(tmp_path_factory) -> Collection:
    """Collection with one vector field + one FTS field."""
    temp_dir = tmp_path_factory.mktemp("zvec_hybrid")
    collection_path = temp_dir / "hybrid_collection"

    schema = zvec.CollectionSchema(
        name="hybrid_test",
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
        vectors=[
            VectorSchema(
                "embedding",
                DataType.VECTOR_FP32,
                dimension=DIM,
                index_param=HnswIndexParam(),
            ),
        ],
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
    """Corpus with both text content and vectors.

    Docs 0-2: AI/ML topic, vectors clustered in one region.
    Docs 3-4: retrieval topic, vectors clustered in another region.
    Doc 5: unrelated topic.
    """
    # AI cluster vectors
    ai_vec = [1.0] * 8 + [0.0] * 8
    # Retrieval cluster vectors
    ret_vec = [0.0] * 8 + [1.0] * 8
    # Unrelated vector
    other_vec = [0.5] * 16

    return [
        Doc(
            id="pk_0",
            fields={
                "title": "ML Intro",
                "content": "machine learning is a branch of artificial intelligence",
            },
            vectors={"embedding": ai_vec},
        ),
        Doc(
            id="pk_1",
            fields={
                "title": "Deep Learning",
                "content": "deep learning uses neural networks for pattern recognition",
            },
            vectors={"embedding": [0.9] * 8 + [0.1] * 8},
        ),
        Doc(
            id="pk_2",
            fields={
                "title": "NLP",
                "content": "natural language processing handles text with artificial intelligence",
            },
            vectors={"embedding": [0.8] * 8 + [0.2] * 8},
        ),
        Doc(
            id="pk_3",
            fields={
                "title": "Search Engine",
                "content": "search engine uses inverted index for text retrieval",
            },
            vectors={"embedding": ret_vec},
        ),
        Doc(
            id="pk_4",
            fields={
                "title": "Vector DB",
                "content": "vector database enables similarity retrieval and search",
            },
            vectors={"embedding": [0.1] * 8 + [0.9] * 8},
        ),
        Doc(
            id="pk_5",
            fields={
                "title": "Cooking",
                "content": "baking bread requires flour water yeast and salt",
            },
            vectors={"embedding": other_vec},
        ),
    ]


@pytest.fixture(scope="function")
def hybrid_collection_with_docs(hybrid_collection: Collection) -> Collection:
    """Hybrid collection pre-populated with test documents."""
    results = hybrid_collection.insert(_make_docs())
    assert all(r.ok() for r in results)
    return hybrid_collection


# ==================== Tests ====================


class TestFtsVectorHybridQuery:
    """Test FTS + vector hybrid retrieval using multi-query with RRF reranker."""

    def test_hybrid_fts_and_vector_basic(self, hybrid_collection_with_docs: Collection):
        """FTS + vector multi-query with RRF reranker returns results."""
        reranker = RrfReRanker(rank_constant=60)
        result = hybrid_collection_with_docs.query(
            queries=[
                Query(field_name="content", fts=Fts(match_string="retrieval")),
                Query(field_name="embedding", vector=[0.0] * 8 + [1.0] * 8),
            ],
            topk=5,
            reranker=reranker,
        )
        assert len(result) > 0
        assert len(result) <= 5
        # Results should have scores
        for doc in result:
            assert doc.score > 0

    def test_hybrid_fts_and_vector_ranking(
        self, hybrid_collection_with_docs: Collection
    ):
        """Docs relevant in both FTS and vector should rank higher."""
        reranker = RrfReRanker(rank_constant=60)
        # FTS: "retrieval search" matches pk_3, pk_4
        # Vector: ret_vec cluster matches pk_3, pk_4
        # Both signals agree: pk_3 and pk_4 should rank top
        result = hybrid_collection_with_docs.query(
            queries=[
                Query(field_name="content", fts=Fts(match_string="retrieval search")),
                Query(field_name="embedding", vector=[0.0] * 8 + [1.0] * 8),
            ],
            topk=5,
            reranker=reranker,
        )
        top_ids = {doc.id for doc in result[:3]}
        assert "pk_3" in top_ids or "pk_4" in top_ids

    def test_hybrid_scores_descending(self, hybrid_collection_with_docs: Collection):
        """Hybrid query results must be sorted by score descending."""
        reranker = RrfReRanker(rank_constant=60)
        result = hybrid_collection_with_docs.query(
            queries=[
                Query(field_name="content", fts=Fts(match_string="intelligence")),
                Query(field_name="embedding", vector=[1.0] * 8 + [0.0] * 8),
            ],
            topk=6,
            reranker=reranker,
        )
        assert len(result) >= 2
        scores = [doc.score for doc in result]
        assert scores == sorted(scores, reverse=True)

    def test_hybrid_with_filter(self, hybrid_collection_with_docs: Collection):
        """Hybrid query respects SQL filter."""
        reranker = RrfReRanker(rank_constant=60)
        result = hybrid_collection_with_docs.query(
            queries=[
                Query(field_name="content", fts=Fts(match_string="learning")),
                Query(field_name="embedding", vector=[1.0] * 8 + [0.0] * 8),
            ],
            topk=10,
            reranker=reranker,
            filter="title like '%Learning%'",
        )
        for doc in result:
            assert "Learning" in doc.fields["title"]

    def test_hybrid_fts_no_match_still_returns_vector_results(
        self, hybrid_collection_with_docs: Collection
    ):
        """When FTS matches nothing, vector results still appear."""
        reranker = RrfReRanker(rank_constant=60)
        result = hybrid_collection_with_docs.query(
            queries=[
                Query(
                    field_name="content",
                    fts=Fts(match_string="nonexistent_term_xyz"),
                ),
                Query(field_name="embedding", vector=[1.0] * 8 + [0.0] * 8),
            ],
            topk=5,
            reranker=reranker,
        )
        # Vector query alone should still produce results
        assert len(result) > 0

    def test_hybrid_query_string_syntax(self, hybrid_collection_with_docs: Collection):
        """Hybrid query works with FTS query_string (advanced syntax)."""
        reranker = RrfReRanker(rank_constant=60)
        result = hybrid_collection_with_docs.query(
            queries=[
                Query(
                    field_name="content",
                    fts=Fts(query_string="artificial AND intelligence"),
                ),
                Query(field_name="embedding", vector=[1.0] * 8 + [0.0] * 8),
            ],
            topk=5,
            reranker=reranker,
        )
        assert len(result) > 0
        # pk_0 and pk_2 contain "artificial intelligence"
        hit_ids = {doc.id for doc in result}
        assert "pk_0" in hit_ids or "pk_2" in hit_ids


class TestFtsVectorHybridValidation:
    """Test validation rules for FTS + vector hybrid queries."""

    def test_hybrid_requires_reranker(self, hybrid_collection_with_docs: Collection):
        """Multi-query with FTS + vector without reranker should raise."""
        with pytest.raises(ValueError, match="[Rr]eranker"):
            hybrid_collection_with_docs.query(
                queries=[
                    Query(field_name="content", fts=Fts(match_string="learning")),
                    Query(field_name="embedding", vector=[1.0] * DIM),
                ],
                topk=5,
            )

    def test_duplicate_field_name_allowed(
        self, hybrid_collection_with_docs: Collection
    ):
        """Multi-query with duplicate field names is allowed and returns results."""
        reranker = RrfReRanker(rank_constant=60)
        result = hybrid_collection_with_docs.query(
            queries=[
                Query(field_name="content", fts=Fts(match_string="learning")),
                Query(field_name="content", fts=Fts(match_string="intelligence")),
            ],
            topk=5,
            reranker=reranker,
        )
        assert len(result) > 0
        assert len(result) <= 5

    def test_multiple_vectors_allowed(self, hybrid_collection_with_docs: Collection):
        """Two vector queries on the same field are allowed with a reranker."""
        reranker = RrfReRanker(rank_constant=60)
        result = hybrid_collection_with_docs.query(
            queries=[
                Query(field_name="embedding", vector=[1.0] * DIM),
                Query(field_name="embedding", vector=[0.5] * DIM),
            ],
            topk=5,
            reranker=reranker,
        )
        assert len(result) > 0
        assert len(result) <= 5


class TestFtsVectorHybridWeightedReranker:
    """Test FTS + vector hybrid retrieval using WeightedReranker."""

    def test_weighted_reranker_fts_and_vector(
        self, hybrid_collection_with_docs: Collection
    ):
        """WeightedReranker correctly normalizes FTS scores alongside vector scores."""
        weights = [0.5, 0.5]
        reranker = WeightedReRanker(weights=weights)
        result = hybrid_collection_with_docs.query(
            queries=[
                Query(field_name="content", fts=Fts(match_string="retrieval search")),
                Query(field_name="embedding", vector=[0.0] * 8 + [1.0] * 8),
            ],
            topk=5,
            reranker=reranker,
        )
        assert len(result) > 0
        assert len(result) <= 5
        for doc in result:
            assert doc.score > 0

    def test_weighted_reranker_scores_descending(
        self, hybrid_collection_with_docs: Collection
    ):
        """WeightedReranker hybrid results are sorted by score descending."""
        weights = [0.4, 0.6]
        reranker = WeightedReRanker(weights=weights)
        result = hybrid_collection_with_docs.query(
            queries=[
                Query(field_name="content", fts=Fts(match_string="intelligence")),
                Query(field_name="embedding", vector=[1.0] * 8 + [0.0] * 8),
            ],
            topk=6,
            reranker=reranker,
        )
        assert len(result) >= 2
        scores = [doc.score for doc in result]
        assert scores == sorted(scores, reverse=True)

    def test_weighted_reranker_fts_weight_influence(
        self, hybrid_collection_with_docs: Collection
    ):
        """Higher FTS weight should boost FTS-relevant docs in ranking."""
        # High FTS weight: FTS signal dominates
        weights_fts_heavy = [0.9, 0.1]
        reranker_fts = WeightedReRanker(weights=weights_fts_heavy)
        result_fts = hybrid_collection_with_docs.query(
            queries=[
                Query(field_name="content", fts=Fts(match_string="retrieval")),
                Query(field_name="embedding", vector=[1.0] * 8 + [0.0] * 8),
            ],
            topk=5,
            reranker=reranker_fts,
        )

        # High vector weight: vector signal dominates
        weights_vec_heavy = [0.1, 0.9]
        reranker_vec = WeightedReRanker(weights=weights_vec_heavy)
        result_vec = hybrid_collection_with_docs.query(
            queries=[
                Query(field_name="content", fts=Fts(match_string="retrieval")),
                Query(field_name="embedding", vector=[1.0] * 8 + [0.0] * 8),
            ],
            topk=5,
            reranker=reranker_vec,
        )

        # Both should return results
        assert len(result_fts) > 0
        assert len(result_vec) > 0
        # With FTS-heavy weight, FTS-relevant docs (pk_3, pk_4) should rank higher
        fts_top = [doc.id for doc in result_fts[:2]]
        vec_top = [doc.id for doc in result_vec[:2]]
        # The rankings should differ due to weight difference
        assert fts_top != vec_top or len(result_fts) == len(result_vec) == 1
