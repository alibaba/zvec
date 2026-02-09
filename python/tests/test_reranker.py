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

from unittest.mock import patch, MagicMock
import pytest
import math
import os

from zvec import Doc, MetricType
from zvec.extension import RrfReRanker, WeightedReRanker, QwenReRanker

# Set ZVEC_RUN_INTEGRATION_TESTS=1 to run real API tests
RUN_INTEGRATION_TESTS = os.environ.get("ZVEC_RUN_INTEGRATION_TESTS", "0") == "1"


# ----------------------------
# RrfRanker Test Case
# ----------------------------
class TestRrfReRanker:
    def test_init(self):
        reranker = RrfReRanker(topn=5, rerank_field="content", rank_constant=100)
        assert reranker.topn == 5
        assert reranker.rerank_field == "content"
        assert reranker.rank_constant == 100

    def test_rrf_score(self):
        reranker = RrfReRanker(rank_constant=60)
        # 根据公式 1.0 / (k + rank + 1)，其中k=60
        assert reranker._rrf_score(0) == 1.0 / (60 + 0 + 1)
        assert reranker._rrf_score(1) == 1.0 / (60 + 1 + 1)
        assert reranker._rrf_score(10) == 1.0 / (60 + 10 + 1)

    def test_rerank(self):
        reranker = RrfReRanker(topn=3)

        doc1 = Doc(id="1", score=0.8)
        doc2 = Doc(id="2", score=0.7)
        doc3 = Doc(id="3", score=0.9)
        doc4 = Doc(id="4", score=0.6)

        query_results = {"vector1": [doc1, doc2, doc3], "vector2": [doc3, doc1, doc4]}

        results = reranker.rerank(query_results)

        assert len(results) <= reranker.topn

        for doc in results:
            assert hasattr(doc, "score")

        scores = [doc.score for doc in results]
        assert scores == sorted(scores, reverse=True)


# ----------------------------
# WeightedRanker Test Case
# ----------------------------
class TestWeightedReRanker:
    def test_init(self):
        weights = {"vector1": 0.7, "vector2": 0.3}
        reranker = WeightedReRanker(
            topn=5,
            rerank_field="content",
            metric=MetricType.L2,
            weights=weights,
        )
        assert reranker.topn == 5
        assert reranker.rerank_field == "content"
        assert reranker.metric == MetricType.L2
        assert reranker.weights == weights

    def test_normalize_score(self):
        reranker = WeightedReRanker()

        score = reranker._normalize_score(1.0, MetricType.L2)
        expected = 1.0 - 2 * math.atan(1.0) / math.pi
        assert score == expected

        score = reranker._normalize_score(1.0, MetricType.IP)
        expected = 0.5 + math.atan(1.0) / math.pi
        assert score == expected

        score = reranker._normalize_score(1.0, MetricType.COSINE)
        expected = 1.0 - 1.0 / 2.0
        assert score == expected

        with pytest.raises(ValueError, match="Unsupported metric type"):
            reranker._normalize_score(1.0, "unsupported_metric")

    def test_rerank(self):
        weights = {"vector1": 0.7, "vector2": 0.3}
        reranker = WeightedReRanker(topn=3, weights=weights, metric=MetricType.L2)

        doc1 = Doc(id="1", score=0.8)
        doc2 = Doc(id="2", score=0.7)
        doc3 = Doc(id="3", score=0.9)

        query_results = {"vector1": [doc1, doc2], "vector2": [doc2, doc3]}

        results = reranker.rerank(query_results)

        assert len(results) <= reranker.topn

        for doc in results:
            assert hasattr(doc, "score")

        scores = [doc.score for doc in results]
        assert scores == sorted(scores, reverse=True)


# ----------------------------
# QwenReRanker Test Case
# ----------------------------
class TestQwenReRanker:
    def test_init_without_query(self):
        with pytest.raises(ValueError, match="Query is required for QwenReRanker"):
            QwenReRanker(api_key="test_key")

    def test_init_without_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="DashScope API key is required"):
                QwenReRanker(query="test")

    @patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test_key"})
    def test_init_with_env_api_key(self):
        reranker = QwenReRanker(query="test", rerank_field="content")
        assert reranker.query == "test"
        assert reranker._api_key == "test_key"
        assert reranker.rerank_field == "content"

    def test_init_with_explicit_api_key(self):
        reranker = QwenReRanker(
            query="test", api_key="explicit_key", rerank_field="content"
        )
        assert reranker.query == "test"
        assert reranker._api_key == "explicit_key"

    def test_model_property(self):
        reranker = QwenReRanker(
            query="test", api_key="test_key", rerank_field="content"
        )
        assert reranker.model == "gte-rerank-v2"

        reranker = QwenReRanker(
            query="test",
            model="custom-model",
            api_key="test_key",
            rerank_field="content",
        )
        assert reranker.model == "custom-model"

    def test_query_property(self):
        reranker = QwenReRanker(
            query="test query", api_key="test_key", rerank_field="content"
        )
        assert reranker.query == "test query"

    def test_topn_property(self):
        reranker = QwenReRanker(
            query="test", topn=5, api_key="test_key", rerank_field="content"
        )
        assert reranker.topn == 5

    def test_rerank_field_property(self):
        reranker = QwenReRanker(query="test", api_key="test_key", rerank_field="title")
        assert reranker.rerank_field == "title"

    def test_rerank_empty_results(self):
        reranker = QwenReRanker(
            query="test", api_key="test_key", rerank_field="content"
        )
        results = reranker.rerank({})
        assert results == []

    def test_rerank_no_valid_documents(self):
        reranker = QwenReRanker(
            query="test", api_key="test_key", rerank_field="content"
        )
        # Document without the rerank_field
        query_results = {"vector1": [Doc(id="1")]}
        with pytest.raises(ValueError, match="No documents to rerank"):
            reranker.rerank(query_results)

    def test_rerank_skip_empty_content(self):
        reranker = QwenReRanker(
            query="test", api_key="test_key", rerank_field="content"
        )
        query_results = {
            "vector1": [
                Doc(id="1", fields={"content": ""}),
                Doc(id="2", fields={"content": "   "}),
            ]
        }
        with pytest.raises(ValueError, match="No documents to rerank"):
            reranker.rerank(query_results)

    @patch("zvec.extension.qwen_function.require_module")
    def test_rerank_success(self, mock_require_module):
        # Mock dashscope module
        mock_dashscope = MagicMock()
        mock_require_module.return_value = mock_dashscope

        # Mock API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.output = {
            "results": [
                {"index": 0, "relevance_score": 0.95},
                {"index": 1, "relevance_score": 0.85},
            ]
        }
        mock_dashscope.TextReRank.call.return_value = mock_response

        reranker = QwenReRanker(
            query="test query", topn=2, api_key="test_key", rerank_field="content"
        )

        query_results = {
            "vector1": [
                Doc(id="1", fields={"content": "Document 1"}),
                Doc(id="2", fields={"content": "Document 2"}),
            ]
        }

        results = reranker.rerank(query_results)

        assert len(results) == 2
        assert results[0].id == "1"
        assert results[0].score == 0.95
        assert results[1].id == "2"
        assert results[1].score == 0.85

        # Verify API call
        mock_dashscope.TextReRank.call.assert_called_once_with(
            model="gte-rerank-v2",
            query="test query",
            documents=["Document 1", "Document 2"],
            top_n=2,
            return_documents=False,
        )

    @patch("zvec.extension.qwen_function.require_module")
    def test_rerank_deduplicate_documents(self, mock_require_module):
        # Mock dashscope module
        mock_dashscope = MagicMock()
        mock_require_module.return_value = mock_dashscope

        # Mock API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.output = {
            "results": [
                {"index": 0, "relevance_score": 0.9},
            ]
        }
        mock_dashscope.TextReRank.call.return_value = mock_response

        reranker = QwenReRanker(
            query="test", topn=5, api_key="test_key", rerank_field="content"
        )

        # Same document in multiple vector results
        doc1 = Doc(id="1", fields={"content": "Document 1"})
        query_results = {"vector1": [doc1], "vector2": [doc1]}

        results = reranker.rerank(query_results)

        # Should only call API with document once
        call_args = mock_dashscope.TextReRank.call.call_args
        assert len(call_args[1]["documents"]) == 1

    @patch("zvec.extension.qwen_function.require_module")
    def test_rerank_api_error(self, mock_require_module):
        # Mock dashscope module
        mock_dashscope = MagicMock()
        mock_require_module.return_value = mock_dashscope

        # Mock API error response
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.message = "Invalid request"
        mock_response.code = "InvalidParameter"
        mock_dashscope.TextReRank.call.return_value = mock_response

        reranker = QwenReRanker(
            query="test", api_key="test_key", rerank_field="content"
        )

        query_results = {"vector1": [Doc(id="1", fields={"content": "Document 1"})]}

        with pytest.raises(ValueError, match="DashScope API error"):
            reranker.rerank(query_results)

    @patch("zvec.extension.qwen_function.require_module")
    def test_rerank_runtime_error(self, mock_require_module):
        # Mock dashscope module that raises exception
        mock_dashscope = MagicMock()
        mock_require_module.return_value = mock_dashscope
        mock_dashscope.TextReRank.call.side_effect = Exception("Network error")

        reranker = QwenReRanker(
            query="test", api_key="test_key", rerank_field="content"
        )

        query_results = {"vector1": [Doc(id="1", fields={"content": "Document 1"})]}

        with pytest.raises(RuntimeError, match="Failed to call DashScope API"):
            reranker.rerank(query_results)

    @pytest.mark.skipif(
        not RUN_INTEGRATION_TESTS,
        reason="Integration test skipped. Set ZVEC_RUN_INTEGRATION_TESTS=1 to run.",
    )
    def test_real_qwen_rerank(self):
        """Integration test with real DashScope TextReRank API.

        To run this test, set environment variables:
            export ZVEC_RUN_INTEGRATION_TESTS=1
            export DASHSCOPE_API_KEY=your-api-key
        """
        # Create reranker with real API
        reranker = QwenReRanker(
            query="What is machine learning?",
            topn=3,
            rerank_field="content",
            model="gte-rerank-v2",
        )

        # Prepare test documents
        query_results = {
            "vector1": [
                Doc(
                    id="1",
                    score=0.8,
                    fields={
                        "content": "Machine learning is a subset of artificial intelligence that focuses on building systems that can learn from data."
                    },
                ),
                Doc(
                    id="2",
                    score=0.7,
                    fields={
                        "content": "The weather is nice today with clear skies and sunshine."
                    },
                ),
                Doc(
                    id="3",
                    score=0.75,
                    fields={
                        "content": "Deep learning is a specialized branch of machine learning using neural networks with multiple layers."
                    },
                ),
            ],
            "vector2": [
                Doc(
                    id="4",
                    score=0.6,
                    fields={
                        "content": "Python is a popular programming language for data science and machine learning applications."
                    },
                ),
                Doc(
                    id="5",
                    score=0.65,
                    fields={
                        "content": "A recipe for chocolate cake includes flour, sugar, eggs, and cocoa powder."
                    },
                ),
            ],
        }

        # Call real API
        results = reranker.rerank(query_results)

        # Verify results
        assert len(results) <= 3, "Should return at most topn documents"
        assert len(results) > 0, "Should return at least one document"

        # All results should have valid scores
        for doc in results:
            assert hasattr(doc, "score"), "Each document should have a score"
            assert isinstance(doc.score, (int, float)), "Score should be numeric"
            assert doc.score > 0, "Score should be positive"

        # Verify scores are in descending order
        scores = [doc.score for doc in results]
        assert scores == sorted(scores, reverse=True), (
            "Results should be sorted by score in descending order"
        )

        # Verify relevant documents are ranked higher
        # Document 1 and 3 are about machine learning, should rank higher than weather/recipe docs
        result_ids = [doc.id for doc in results]

        # At least one of the ML-related documents should be in top results
        ml_related_docs = {"1", "3", "4"}
        assert any(doc_id in ml_related_docs for doc_id in result_ids[:2]), (
            "ML-related documents should rank higher"
        )

        # Print results for manual verification (useful during development)
        print("\nReranking results:")
        for i, doc in enumerate(results, 1):
            print(f"{i}. ID={doc.id}, Score={doc.score:.4f}")
            if doc.fields:
                content = doc.field("content")
                if content:
                    print(f"   Content: {content[:80]}...")
