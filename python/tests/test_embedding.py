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

import os
from http import HTTPStatus
from unittest.mock import MagicMock, patch, Mock

import numpy as np
import pytest
from zvec.extension import (
    BM25EmbeddingFunction,
    DefaultLocalDenseEmbedding,
    DefaultLocalSparseEmbedding,
    OpenAIDenseEmbedding,
    QwenDenseEmbedding,
    QwenSparseEmbedding,
)

# Environment variable to control integration tests
# Set ZVEC_RUN_INTEGRATION_TESTS=1 to run real API/model tests
RUN_INTEGRATION_TESTS = os.environ.get("ZVEC_RUN_INTEGRATION_TESTS", "0") == "1"


# ----------------------------
# QwenDenseEmbedding Test Case
# ----------------------------
class TestQwenDenseEmbedding:
    def test_init_with_api_key(self):
        # Test initialization with explicit API key
        embedding_func = QwenDenseEmbedding(dimension=128, api_key="test_key")
        assert embedding_func.dimension == 128
        assert embedding_func.model == "text-embedding-v4"
        assert embedding_func._api_key == "test_key"

    @patch.dict(os.environ, {"DASHSCOPE_API_KEY": "env_key"})
    def test_init_with_env_api_key(self):
        # Test initialization with API key from environment
        embedding_func = QwenDenseEmbedding(dimension=128)
        assert embedding_func._api_key == "env_key"

    @patch.dict(os.environ, {"DASHSCOPE_API_KEY": ""})
    def test_init_with_empty_env_api_key(self):
        # Test initialization with empty API key from environment
        with pytest.raises(ValueError, match="DashScope API key is required"):
            QwenDenseEmbedding(dimension=128)

    def test_model_property(self):
        embedding_func = QwenDenseEmbedding(dimension=128, api_key="test_key")
        assert embedding_func.model == "text-embedding-v4"

        embedding_func = QwenDenseEmbedding(
            dimension=128, model="custom-model", api_key="test_key"
        )
        assert embedding_func.model == "custom-model"

    @patch("zvec.extension.qwen_function.require_module")
    def test_embed_with_empty_text(self, mock_require_module):
        # Test embed method with empty text raises ValueError
        embedding_func = QwenDenseEmbedding(dimension=128, api_key="test_key")

        with pytest.raises(
            ValueError, match="Input text cannot be empty or whitespace only"
        ):
            embedding_func.embed("")

        with pytest.raises(TypeError):
            embedding_func.embed(None)

    @patch("zvec.extension.qwen_function.require_module")
    def test_embed_success(self, mock_require_module):
        # Test successful embedding
        mock_dashscope = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = HTTPStatus.OK
        mock_response.output = {"embeddings": [{"embedding": [0.1, 0.2, 0.3]}]}
        mock_dashscope.TextEmbedding.call.return_value = mock_response
        mock_require_module.return_value = mock_dashscope

        embedding_func = QwenDenseEmbedding(dimension=3, api_key="test_key")
        # Clear cache to avoid interference
        embedding_func.embed.cache_clear()
        result = embedding_func.embed("test text")

        assert result == [0.1, 0.2, 0.3]
        mock_dashscope.TextEmbedding.call.assert_called_once_with(
            model="text-embedding-v4",
            input="test text",
            dimension=3,
            output_type="dense",
        )

    @patch("zvec.extension.qwen_function.require_module")
    def test_embed_http_error(self, mock_require_module):
        # Test embedding with HTTP error
        mock_dashscope = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = HTTPStatus.BAD_REQUEST
        mock_response.message = "Bad Request"
        mock_dashscope.TextEmbedding.call.return_value = mock_response
        mock_require_module.return_value = mock_dashscope

        embedding_func = QwenDenseEmbedding(dimension=128, api_key="test_key")
        embedding_func.embed.cache_clear()

        with pytest.raises(ValueError):
            embedding_func.embed("test text")

    @patch("zvec.extension.qwen_function.require_module")
    def test_embed_invalid_response(self, mock_require_module):
        # Test embedding with invalid response (wrong number of embeddings)
        mock_dashscope = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = HTTPStatus.OK
        mock_response.output = {"embeddings": []}
        mock_dashscope.TextEmbedding.call.return_value = mock_response
        mock_require_module.return_value = mock_dashscope

        embedding_func = QwenDenseEmbedding(dimension=128, api_key="test_key")
        embedding_func.embed.cache_clear()

        with pytest.raises(ValueError):
            embedding_func.embed("test text")

    @pytest.mark.skipif(
        not RUN_INTEGRATION_TESTS,
        reason="Integration test skipped. Set ZVEC_RUN_INTEGRATION_TESTS=1 to run.",
    )
    def test_real_embed_success(self):
        """Integration test with real DashScope API.

        To run this test, set environment variable:
            export ZVEC_RUN_INTEGRATION_TESTS=1
            export DASHSCOPE_API_KEY=your-api-key
        """
        embedding_func = QwenDenseEmbedding(dimension=128)
        dense = embedding_func("test text")
        assert len(dense) == 128


# ----------------------------
# QwenSparseEmbedding Test Case
# ----------------------------
class TestQwenSparseEmbedding:
    """Test suite for QwenSparseEmbedding (Qwen sparse embedding via DashScope API)."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        embedding_func = QwenSparseEmbedding(dimension=1024, api_key="test_key")
        assert embedding_func._dimension == 1024
        assert embedding_func.model == "text-embedding-v4"
        assert embedding_func._api_key == "test_key"
        # encoding_type defaults to "query" via extra_params
        assert embedding_func.extra_params.get("encoding_type", "query") == "query"

    def test_init_with_custom_encoding_type(self):
        """Test initialization with custom encoding type."""
        embedding_func = QwenSparseEmbedding(
            dimension=1024, encoding_type="document", api_key="test_key"
        )
        assert embedding_func.extra_params.get("encoding_type") == "document"

    @patch.dict(os.environ, {"DASHSCOPE_API_KEY": "env_key"})
    def test_init_with_env_api_key(self):
        """Test initialization with API key from environment."""
        embedding_func = QwenSparseEmbedding(dimension=1024)
        assert embedding_func._api_key == "env_key"

    @patch.dict(os.environ, {"DASHSCOPE_API_KEY": ""})
    def test_init_without_api_key(self):
        """Test initialization fails without API key."""
        with pytest.raises(ValueError, match="DashScope API key is required"):
            QwenSparseEmbedding(dimension=1024)

    def test_model_property(self):
        """Test model property."""
        embedding_func = QwenSparseEmbedding(dimension=1024, api_key="test_key")
        assert embedding_func.model == "text-embedding-v4"

        embedding_func = QwenSparseEmbedding(
            dimension=1024, model="text-embedding-v3", api_key="test_key"
        )
        assert embedding_func.model == "text-embedding-v3"

    def test_encoding_type_property(self):
        """Test encoding_type via extra_params."""
        query_emb = QwenSparseEmbedding(
            dimension=1024, encoding_type="query", api_key="test_key"
        )
        assert query_emb.extra_params.get("encoding_type") == "query"

        doc_emb = QwenSparseEmbedding(
            dimension=1024, encoding_type="document", api_key="test_key"
        )
        assert doc_emb.extra_params.get("encoding_type") == "document"

    @patch("zvec.extension.qwen_function.require_module")
    def test_embed_with_empty_text(self, mock_require_module):
        """Test embed method with empty text raises ValueError."""
        embedding_func = QwenSparseEmbedding(dimension=1024, api_key="test_key")

        with pytest.raises(
            ValueError, match="Input text cannot be empty or whitespace only"
        ):
            embedding_func.embed("")

        with pytest.raises(
            ValueError, match="Input text cannot be empty or whitespace only"
        ):
            embedding_func.embed("   ")

    @patch("zvec.extension.qwen_function.require_module")
    def test_embed_with_non_string_input(self, mock_require_module):
        """Test embed method with non-string input raises TypeError."""
        embedding_func = QwenSparseEmbedding(dimension=1024, api_key="test_key")

        with pytest.raises(TypeError, match="Expected 'input' to be str"):
            embedding_func.embed(123)

        with pytest.raises(TypeError, match="Expected 'input' to be str"):
            embedding_func.embed(None)

    @patch("zvec.extension.qwen_function.require_module")
    def test_embed_success(self, mock_require_module):
        """Test successful sparse embedding generation."""
        mock_dashscope = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = HTTPStatus.OK
        # Sparse embedding returns array of {index, value, token} objects
        mock_response.output = {
            "embeddings": [
                {
                    "sparse_embedding": [
                        {"index": 10, "value": 0.5, "token": "机器"},
                        {"index": 245, "value": 0.8, "token": "学习"},
                        {"index": 1023, "value": 1.2, "token": "算法"},
                    ]
                }
            ]
        }
        mock_dashscope.TextEmbedding.call.return_value = mock_response
        mock_require_module.return_value = mock_dashscope

        embedding_func = QwenSparseEmbedding(dimension=1024, api_key="test_key")
        # Clear cache to avoid interference
        embedding_func.embed.cache_clear()
        result = embedding_func.embed("test text")

        # Verify result is a dict
        assert isinstance(result, dict)
        # Verify keys are integers
        assert all(isinstance(k, int) for k in result.keys())
        # Verify values are floats
        assert all(isinstance(v, float) for v in result.values())
        # Verify all values are positive
        assert all(v > 0 for v in result.values())
        # Verify sorted by indices
        keys = list(result.keys())
        assert keys == sorted(keys)
        # Verify specific keys
        assert keys == [10, 245, 1023]

        mock_dashscope.TextEmbedding.call.assert_called_once_with(
            model="text-embedding-v4",
            input="test text",
            dimension=1024,
            output_type="sparse",
            text_type="query",
        )

    @patch("zvec.extension.qwen_function.require_module")
    def test_embed_with_document_encoding_type(self, mock_require_module):
        """Test embedding with document encoding type."""
        mock_dashscope = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = HTTPStatus.OK
        mock_response.output = {
            "embeddings": [
                {
                    "sparse_embedding": [
                        {"index": 5, "value": 0.3, "token": "文档"},
                        {"index": 100, "value": 0.7, "token": "内容"},
                        {"index": 500, "value": 0.9, "token": "检索"},
                    ]
                }
            ]
        }
        mock_dashscope.TextEmbedding.call.return_value = mock_response
        mock_require_module.return_value = mock_dashscope

        embedding_func = QwenSparseEmbedding(
            dimension=1024, encoding_type="document", api_key="test_key"
        )
        embedding_func.embed.cache_clear()
        result = embedding_func.embed("test document")

        assert isinstance(result, dict)
        assert list(result.keys()) == [5, 100, 500]

        # Verify text_type parameter is "document"
        call_args = mock_dashscope.TextEmbedding.call.call_args
        assert call_args[1]["text_type"] == "document"
        assert call_args[1]["output_type"] == "sparse"

    @patch("zvec.extension.qwen_function.require_module")
    def test_embed_output_sorted_by_indices(self, mock_require_module):
        """Test that output is always sorted by indices in ascending order."""
        mock_dashscope = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = HTTPStatus.OK
        # Return unsorted indices
        mock_response.output = {
            "embeddings": [
                {
                    "sparse_embedding": [
                        {"index": 9999, "value": 1.5, "token": "A"},
                        {"index": 5, "value": 2.0, "token": "B"},
                        {"index": 1234, "value": 0.8, "token": "C"},
                        {"index": 77, "value": 3.2, "token": "D"},
                        {"index": 500, "value": 1.1, "token": "E"},
                    ]
                }
            ]
        }
        mock_dashscope.TextEmbedding.call.return_value = mock_response
        mock_require_module.return_value = mock_dashscope

        embedding_func = QwenSparseEmbedding(dimension=1024, api_key="test_key")
        embedding_func.embed.cache_clear()
        result = embedding_func.embed("test sorting")

        # Verify keys are sorted
        result_keys = list(result.keys())
        assert result_keys == sorted(result_keys)
        # Verify expected sorted order
        assert result_keys == [5, 77, 500, 1234, 9999]

    @patch("zvec.extension.qwen_function.require_module")
    def test_embed_filters_zero_values(self, mock_require_module):
        """Test that zero and negative values are filtered out."""
        mock_dashscope = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = HTTPStatus.OK
        # Include zero and negative values
        mock_response.output = {
            "embeddings": [
                {
                    "sparse_embedding": [
                        {"index": 10, "value": 0.5, "token": "正"},
                        {
                            "index": 20,
                            "value": 0.0,
                            "token": "零",
                        },  # Should be filtered
                        {
                            "index": 30,
                            "value": -0.3,
                            "token": "负",
                        },  # Should be filtered
                        {"index": 40, "value": 0.8, "token": "正"},
                        {
                            "index": 50,
                            "value": 0.0,
                            "token": "零",
                        },  # Should be filtered
                    ]
                }
            ]
        }
        mock_dashscope.TextEmbedding.call.return_value = mock_response
        mock_require_module.return_value = mock_dashscope

        embedding_func = QwenSparseEmbedding(dimension=1024, api_key="test_key")
        embedding_func.embed.cache_clear()
        result = embedding_func.embed("test filtering")

        # Only positive values should remain
        assert list(result.keys()) == [10, 40]
        assert all(v > 0 for v in result.values())

    @patch("zvec.extension.qwen_function.require_module")
    def test_embed_http_error(self, mock_require_module):
        """Test embedding with HTTP error."""
        mock_dashscope = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = HTTPStatus.BAD_REQUEST
        mock_response.message = "Bad Request"
        mock_dashscope.TextEmbedding.call.return_value = mock_response
        mock_require_module.return_value = mock_dashscope

        embedding_func = QwenSparseEmbedding(dimension=1024, api_key="test_key")
        embedding_func.embed.cache_clear()

        with pytest.raises(ValueError, match="DashScope API error"):
            embedding_func.embed("test text")

    @patch("zvec.extension.qwen_function.require_module")
    def test_embed_invalid_response_no_embeddings(self, mock_require_module):
        """Test embedding with invalid response (no embeddings)."""
        mock_dashscope = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = HTTPStatus.OK
        mock_response.output = {"embeddings": []}
        mock_dashscope.TextEmbedding.call.return_value = mock_response
        mock_require_module.return_value = mock_dashscope

        embedding_func = QwenSparseEmbedding(dimension=1024, api_key="test_key")
        embedding_func.embed.cache_clear()

        with pytest.raises(ValueError, match="Expected exactly 1 embedding"):
            embedding_func.embed("test text")

    @patch("zvec.extension.qwen_function.require_module")
    def test_embed_invalid_response_not_dict(self, mock_require_module):
        """Test embedding with invalid response (sparse_embedding not list)."""
        mock_dashscope = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = HTTPStatus.OK
        # sparse_embedding should be list, not dict
        mock_response.output = {
            "embeddings": [{"sparse_embedding": {"index": 10, "value": 0.5}}]
        }
        mock_dashscope.TextEmbedding.call.return_value = mock_response
        mock_require_module.return_value = mock_dashscope

        embedding_func = QwenSparseEmbedding(dimension=1024, api_key="test_key")
        embedding_func.embed.cache_clear()

        with pytest.raises(
            ValueError, match="'sparse_embedding' field is missing or not a list"
        ):
            embedding_func.embed("test text")

    @patch("zvec.extension.qwen_function.require_module")
    def test_embed_callable_interface(self, mock_require_module):
        """Test that embedding function is callable."""
        mock_dashscope = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = HTTPStatus.OK
        mock_response.output = {
            "embeddings": [
                {
                    "sparse_embedding": [
                        {"index": 100, "value": 1.0, "token": "测试"},
                        {"index": 200, "value": 0.5, "token": "调用"},
                    ]
                }
            ]
        }
        mock_dashscope.TextEmbedding.call.return_value = mock_response
        mock_require_module.return_value = mock_dashscope

        embedding_func = QwenSparseEmbedding(dimension=1024, api_key="test_key")
        embedding_func.embed.cache_clear()

        # Test calling the function directly
        result = embedding_func("test text")
        assert isinstance(result, dict)
        assert list(result.keys()) == [100, 200]

    @patch("zvec.extension.qwen_function.require_module")
    def test_embed_api_connection_error(self, mock_require_module):
        """Test handling of API connection errors."""
        mock_dashscope = MagicMock()
        mock_dashscope.TextEmbedding.call.side_effect = Exception("Connection timeout")
        mock_require_module.return_value = mock_dashscope

        embedding_func = QwenSparseEmbedding(dimension=1024, api_key="test_key")
        embedding_func.embed.cache_clear()

        with pytest.raises(RuntimeError, match="Failed to call DashScope API"):
            embedding_func.embed("test text")

    @pytest.mark.skipif(
        not RUN_INTEGRATION_TESTS,
        reason="Integration test skipped. Set ZVEC_RUN_INTEGRATION_TESTS=1 to run.",
    )
    def test_real_embed_success(self):
        """Integration test with real DashScope API.

        To run this test, set environment variable:
            export ZVEC_RUN_INTEGRATION_TESTS=1
            export DASHSCOPE_API_KEY=your-api-key
        """
        # Test query embedding
        query_emb = QwenSparseEmbedding(dimension=1024, encoding_type="query")
        query_vec = query_emb.embed("machine learning")

        assert isinstance(query_vec, dict)
        assert len(query_vec) > 0
        assert all(isinstance(k, int) for k in query_vec.keys())
        assert all(isinstance(v, float) and v > 0 for v in query_vec.values())

        # Verify sorted output
        keys = list(query_vec.keys())
        assert keys == sorted(keys)

        # Test document embedding
        doc_emb = QwenSparseEmbedding(dimension=1024, encoding_type="document")
        doc_vec = doc_emb.embed("Machine learning is a subset of AI")

        assert isinstance(doc_vec, dict)
        assert len(doc_vec) > 0

        # Verify sorted output
        doc_keys = list(doc_vec.keys())
        assert doc_keys == sorted(doc_keys)


# ----------------------------
# OpenAIDenseEmbedding Test Case
# ----------------------------
class TestOpenAIDenseEmbedding:
    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        embedding_func = OpenAIDenseEmbedding(api_key="sk-test-key")
        assert embedding_func.dimension == 1536  # Default for text-embedding-3-small
        assert embedding_func.model == "text-embedding-3-small"
        assert embedding_func._api_key == "sk-test-key"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-env-key"})
    def test_init_with_env_api_key(self):
        """Test initialization with API key from environment."""
        embedding_func = OpenAIDenseEmbedding()
        assert embedding_func._api_key == "sk-env-key"

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""})
    def test_init_without_api_key(self):
        """Test initialization fails without API key."""
        with pytest.raises(ValueError, match="OpenAI API key is required"):
            OpenAIDenseEmbedding()

    def test_init_with_custom_dimension(self):
        """Test initialization with custom dimension."""
        embedding_func = OpenAIDenseEmbedding(
            model="text-embedding-3-large", dimension=1024, api_key="sk-test"
        )
        assert embedding_func.dimension == 1024
        assert embedding_func.model == "text-embedding-3-large"

    def test_init_with_base_url(self):
        """Test initialization with custom base URL."""
        embedding_func = OpenAIDenseEmbedding(
            api_key="sk-test", base_url="https://custom.openai.com/"
        )
        assert embedding_func._base_url == "https://custom.openai.com/"

    def test_model_property(self):
        """Test model property."""
        embedding_func = OpenAIDenseEmbedding(api_key="sk-test")
        assert embedding_func.model == "text-embedding-3-small"

        embedding_func = OpenAIDenseEmbedding(
            model="text-embedding-ada-002", api_key="sk-test"
        )
        assert embedding_func.model == "text-embedding-ada-002"

    def test_extra_params(self):
        """Test extra_params property."""
        # Test without extra params
        embedding_func = OpenAIDenseEmbedding(api_key="sk-test")
        assert embedding_func.extra_params == {}

        # Test with extra params
        embedding_func = OpenAIDenseEmbedding(
            api_key="sk-test",
            encoding_format="float",
            user="test-user",
        )
        assert embedding_func.extra_params == {
            "encoding_format": "float",
            "user": "test-user",
        }

    @patch("zvec.extension.openai_function.require_module")
    def test_embed_with_empty_text(self, mock_require_module):
        """Test embed method with empty text raises ValueError."""
        embedding_func = OpenAIDenseEmbedding(api_key="sk-test")

        with pytest.raises(
            ValueError, match="Input text cannot be empty or whitespace only"
        ):
            embedding_func.embed("")

        with pytest.raises(
            ValueError, match="Input text cannot be empty or whitespace only"
        ):
            embedding_func.embed("   ")

    @patch("zvec.extension.openai_function.require_module")
    def test_embed_with_non_string_input(self, mock_require_module):
        """Test embed method with non-string input raises TypeError."""
        embedding_func = OpenAIDenseEmbedding(api_key="sk-test")

        with pytest.raises(TypeError, match="Expected 'input' to be str"):
            embedding_func.embed(123)

        with pytest.raises(TypeError, match="Expected 'input' to be str"):
            embedding_func.embed(None)

    @patch("zvec.extension.openai_function.require_module")
    def test_embed_success(self, mock_require_module):
        """Test successful embedding generation."""
        # Mock OpenAI client
        mock_openai = Mock()
        mock_client = Mock()
        mock_response = Mock()

        # Create mock embedding data
        fake_embedding = [0.1, 0.2, 0.3]
        mock_embedding_obj = Mock()
        mock_embedding_obj.embedding = fake_embedding
        mock_response.data = [mock_embedding_obj]

        mock_client.embeddings.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client
        mock_require_module.return_value = mock_openai

        embedding_func = OpenAIDenseEmbedding(dimension=3, api_key="sk-test")
        embedding_func.embed.cache_clear()
        result = embedding_func.embed("test text")

        assert result == [0.1, 0.2, 0.3]
        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small", input="test text", dimensions=3
        )

    @patch("zvec.extension.openai_function.require_module")
    def test_embed_with_custom_model(self, mock_require_module):
        """Test embedding with custom model."""
        mock_openai = Mock()
        mock_client = Mock()
        mock_response = Mock()

        fake_embedding = [0.1] * 1536
        mock_embedding_obj = Mock()
        mock_embedding_obj.embedding = fake_embedding
        mock_response.data = [mock_embedding_obj]

        mock_client.embeddings.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client
        mock_require_module.return_value = mock_openai

        embedding_func = OpenAIDenseEmbedding(
            model="text-embedding-ada-002", api_key="sk-test"
        )
        embedding_func.embed.cache_clear()
        result = embedding_func.embed("test text")

        assert len(result) == 1536
        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-ada-002", input="test text"
        )

    @patch("zvec.extension.openai_function.require_module")
    def test_embed_api_error(self, mock_require_module):
        """Test handling of API errors."""
        mock_openai = Mock()
        mock_client = Mock()

        # Simulate API error
        api_error = Mock()
        api_error.__class__.__name__ = "APIError"
        mock_openai.APIError = type("APIError", (Exception,), {})
        mock_openai.APIConnectionError = type("APIConnectionError", (Exception,), {})

        mock_client.embeddings.create.side_effect = mock_openai.APIError(
            "Rate limit exceeded"
        )
        mock_openai.OpenAI.return_value = mock_client
        mock_require_module.return_value = mock_openai

        embedding_func = OpenAIDenseEmbedding(api_key="sk-test")
        embedding_func.embed.cache_clear()

        with pytest.raises(RuntimeError, match="Failed to call OpenAI API"):
            embedding_func.embed("test text")

    @patch("zvec.extension.openai_function.require_module")
    def test_embed_invalid_response(self, mock_require_module):
        """Test handling of invalid API response."""
        mock_openai = Mock()
        mock_client = Mock()
        mock_response = Mock()

        # Empty response data
        mock_response.data = []

        mock_client.embeddings.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client
        mock_openai.APIError = type("APIError", (Exception,), {})
        mock_openai.APIConnectionError = type("APIConnectionError", (Exception,), {})
        mock_require_module.return_value = mock_openai

        embedding_func = OpenAIDenseEmbedding(api_key="sk-test")
        embedding_func.embed.cache_clear()

        with pytest.raises(ValueError, match="no embedding data returned"):
            embedding_func.embed("test text")

    @patch("zvec.extension.openai_function.require_module")
    def test_embed_dimension_mismatch(self, mock_require_module):
        """Test handling of dimension mismatch."""
        mock_openai = Mock()
        mock_client = Mock()
        mock_response = Mock()

        # Return embedding with wrong dimension
        fake_embedding = [0.1] * 512
        mock_embedding_obj = Mock()
        mock_embedding_obj.embedding = fake_embedding
        mock_response.data = [mock_embedding_obj]

        mock_client.embeddings.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client
        mock_openai.APIError = type("APIError", (Exception,), {})
        mock_openai.APIConnectionError = type("APIConnectionError", (Exception,), {})
        mock_require_module.return_value = mock_openai

        embedding_func = OpenAIDenseEmbedding(dimension=1536, api_key="sk-test")
        embedding_func.embed.cache_clear()

        with pytest.raises(ValueError, match="Dimension mismatch"):
            embedding_func.embed("test text")

    @patch("zvec.extension.openai_function.require_module")
    def test_embed_callable(self, mock_require_module):
        """Test that embedding function is callable."""
        mock_openai = Mock()
        mock_client = Mock()
        mock_response = Mock()

        fake_embedding = [0.1] * 1536
        mock_embedding_obj = Mock()
        mock_embedding_obj.embedding = fake_embedding
        mock_response.data = [mock_embedding_obj]

        mock_client.embeddings.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client
        mock_openai.APIError = type("APIError", (Exception,), {})
        mock_openai.APIConnectionError = type("APIConnectionError", (Exception,), {})
        mock_require_module.return_value = mock_openai

        embedding_func = OpenAIDenseEmbedding(api_key="sk-test")
        embedding_func.embed.cache_clear()

        # Test calling the function directly
        result = embedding_func("test text")
        assert isinstance(result, list)
        assert len(result) == 1536

    @patch("zvec.extension.openai_function.require_module")
    def test_embed_with_base_url(self, mock_require_module):
        """Test embedding with custom base URL."""
        mock_openai = Mock()
        mock_client = Mock()
        mock_response = Mock()

        fake_embedding = [0.1] * 1536
        mock_embedding_obj = Mock()
        mock_embedding_obj.embedding = fake_embedding
        mock_response.data = [mock_embedding_obj]

        mock_client.embeddings.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client
        mock_openai.APIError = type("APIError", (Exception,), {})
        mock_openai.APIConnectionError = type("APIConnectionError", (Exception,), {})
        mock_require_module.return_value = mock_openai

        embedding_func = OpenAIDenseEmbedding(
            api_key="sk-test", base_url="https://custom.openai.com/"
        )
        embedding_func.embed.cache_clear()
        result = embedding_func.embed("test text")

        # Verify client was created with custom base URL
        mock_openai.OpenAI.assert_called_once_with(
            api_key="sk-test", base_url="https://custom.openai.com/"
        )
        assert len(result) == 1536

    @pytest.mark.skipif(
        not RUN_INTEGRATION_TESTS,
        reason="Integration test skipped. Set ZVEC_RUN_INTEGRATION_TESTS=1 to run.",
    )
    def test_real_embed_success(self):
        """Integration test with real OpenAI API.

        To run this test, set environment variable:
            export ZVEC_RUN_INTEGRATION_TESTS=1
            export OPENAI_API_KEY=sk-...
        """
        embedding_func = OpenAIDenseEmbedding(
            model="text-embedding-v4",
            dimension=256,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        vector = embedding_func.embed("Hello, world!")
        assert len(vector) == 256
        assert isinstance(vector, list)
        assert all(isinstance(x, float) for x in vector)


# ----------------------------
# DefaultLocalDenseEmbedding Test Case
# ----------------------------
class TestDefaultLocalDenseEmbedding:
    """Test cases for DefaultLocalDenseEmbedding."""

    @patch("zvec.extension.sentence_transformer_function.require_module")
    def test_init_success(self, mock_require_module):
        """Test successful initialization with mocked model."""
        # Mock sentence_transformers module
        mock_st = Mock()
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.device = "cpu"
        mock_st.SentenceTransformer.return_value = mock_model
        mock_require_module.return_value = mock_st

        # Initialize embedding function
        emb_func = DefaultLocalDenseEmbedding()

        # Assertions
        assert emb_func.dimension == 384
        assert emb_func.model_name == "all-MiniLM-L6-v2"
        assert emb_func.model_source == "huggingface"
        assert emb_func.device == "cpu"
        mock_st.SentenceTransformer.assert_called_once_with(
            "all-MiniLM-L6-v2", device=None, trust_remote_code=True
        )

    @patch("zvec.extension.sentence_transformer_function.require_module")
    def test_init_with_custom_device(self, mock_require_module):
        """Test initialization with custom device."""
        mock_st = Mock()
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.device = "cuda"
        mock_st.SentenceTransformer.return_value = mock_model
        mock_require_module.return_value = mock_st

        emb_func = DefaultLocalDenseEmbedding(device="cuda")

        assert emb_func.device == "cuda"
        mock_st.SentenceTransformer.assert_called_once_with(
            "all-MiniLM-L6-v2", device="cuda", trust_remote_code=True
        )

    @pytest.mark.skipif(
        not RUN_INTEGRATION_TESTS,
        reason="Integration test skipped. Set ZVEC_RUN_INTEGRATION_TESTS=1 to run.",
    )
    @patch("zvec.extension.sentence_transformer_function.require_module")
    def test_init_with_modelscope(self, mock_require_module):
        """Test initialization with ModelScope as model source."""
        mock_st = Mock()
        mock_ms = Mock()
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.device = "cpu"
        mock_st.SentenceTransformer.return_value = mock_model

        def require_module_side_effect(module_name):
            if module_name == "sentence_transformers":
                return mock_st
            elif module_name == "modelscope":
                return mock_ms
            raise ImportError(f"No module named '{module_name}'")

        mock_require_module.side_effect = require_module_side_effect

        # Mock snapshot_download at the correct import location
        with patch(
            "modelscope.hub.snapshot_download.snapshot_download",
            return_value="/path/to/cached/model",
        ):
            emb_func = DefaultLocalDenseEmbedding(model_source="modelscope")

        # Assertions
        assert emb_func.dimension == 384
        assert emb_func.model_name == "iic/nlp_gte_sentence-embedding_chinese-small"
        assert emb_func.model_source == "modelscope"

    @patch("zvec.extension.sentence_transformer_function.require_module")
    def test_init_with_invalid_model_source(self, mock_require_module):
        """Test initialization with invalid model_source raises ValueError."""
        mock_st = Mock()
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.SentenceTransformer.return_value = mock_model
        mock_require_module.return_value = mock_st

        with pytest.raises(ValueError, match="Invalid model_source"):
            DefaultLocalDenseEmbedding(model_source="invalid_source")

    @patch("zvec.extension.sentence_transformer_function.require_module")
    def test_embed_success(self, mock_require_module):
        """Test successful embedding generation."""
        # Mock embedding output
        fake_embedding = np.random.rand(384).astype(np.float32)

        mock_st = Mock()
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384

        # Configure encode method
        mock_model.encode = Mock(return_value=fake_embedding)

        mock_st.SentenceTransformer.return_value = mock_model
        mock_require_module.return_value = mock_st

        emb_func = DefaultLocalDenseEmbedding()
        result = emb_func.embed("Hello, world!")

        # Assertions
        assert isinstance(result, list)
        assert len(result) == 384
        assert all(isinstance(x, float) for x in result)
        mock_model.encode.assert_called_once_with(
            "Hello, world!",
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=32,
        )

    @patch("zvec.extension.sentence_transformer_function.require_module")
    def test_embed_with_normalization(self, mock_require_module):
        """Test embedding with L2 normalization."""
        # Create a normalized vector
        fake_embedding = np.random.rand(384).astype(np.float32)
        fake_embedding = fake_embedding / np.linalg.norm(fake_embedding)

        mock_st = Mock()
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384

        # Configure encode method
        mock_model.encode = Mock(return_value=fake_embedding)

        mock_st.SentenceTransformer.return_value = mock_model
        mock_require_module.return_value = mock_st

        emb_func = DefaultLocalDenseEmbedding(normalize_embeddings=True)
        result = emb_func.embed("Test sentence")

        # Check if vector is normalized (L2 norm should be close to 1.0)
        result_array = np.array(result)
        norm = np.linalg.norm(result_array)
        assert abs(norm - 1.0) < 1e-5

    @patch("zvec.extension.sentence_transformer_function.require_module")
    def test_embed_empty_string(self, mock_require_module):
        """Test embedding with empty string raises ValueError."""
        mock_st = Mock()
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.SentenceTransformer.return_value = mock_model
        mock_require_module.return_value = mock_st

        emb_func = DefaultLocalDenseEmbedding()

        with pytest.raises(ValueError, match="Input text cannot be empty"):
            emb_func.embed("")

        with pytest.raises(ValueError, match="Input text cannot be empty"):
            emb_func.embed("   ")

    @patch("zvec.extension.sentence_transformer_function.require_module")
    def test_embed_non_string_input(self, mock_require_module):
        """Test embedding with non-string input raises TypeError."""
        mock_st = Mock()
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.SentenceTransformer.return_value = mock_model
        mock_require_module.return_value = mock_st

        emb_func = DefaultLocalDenseEmbedding()

        with pytest.raises(TypeError, match="Expected 'input' to be str"):
            emb_func.embed(123)

        with pytest.raises(TypeError, match="Expected 'input' to be str"):
            emb_func.embed(None)

    @patch("zvec.extension.sentence_transformer_function.require_module")
    def test_embed_callable(self, mock_require_module):
        """Test that embedding function is callable."""
        fake_embedding = np.random.rand(384).astype(np.float32)

        mock_st = Mock()
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384

        # Configure encode method
        mock_model.encode = Mock(return_value=fake_embedding)

        mock_st.SentenceTransformer.return_value = mock_model
        mock_require_module.return_value = mock_st

        emb_func = DefaultLocalDenseEmbedding()

        # Test calling the function directly
        result = emb_func("Test text")
        assert isinstance(result, list)
        assert len(result) == 384

    @patch("zvec.extension.sentence_transformer_function.require_module")
    def test_semantic_similarity(self, mock_require_module):
        """Test semantic similarity between similar and different texts."""
        # Create mock embeddings for similar and different texts
        similar_emb_1 = np.array([1.0, 0.0, 0.0] + [0.0] * 381, dtype=np.float32)
        similar_emb_2 = np.array([0.9, 0.1, 0.0] + [0.0] * 381, dtype=np.float32)
        different_emb = np.array([0.0, 0.0, 1.0] + [0.0] * 381, dtype=np.float32)

        # Normalize
        similar_emb_1 = similar_emb_1 / np.linalg.norm(similar_emb_1)
        similar_emb_2 = similar_emb_2 / np.linalg.norm(similar_emb_2)
        different_emb = different_emb / np.linalg.norm(different_emb)

        mock_st = Mock()
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384

        # Configure encode method with side_effect for multiple calls
        mock_model.encode = Mock(
            side_effect=[similar_emb_1, similar_emb_2, different_emb]
        )

        mock_st.SentenceTransformer.return_value = mock_model
        mock_require_module.return_value = mock_st

        emb_func = DefaultLocalDenseEmbedding()

        v1 = emb_func.embed("The cat sits on the mat")
        v2 = emb_func.embed("A feline rests on a rug")
        v3 = emb_func.embed("Python programming")

        # Calculate similarities
        similarity_high = np.dot(v1, v2)
        similarity_low = np.dot(v1, v3)

        assert similarity_high > similarity_low

    @patch("zvec.extension.sentence_transformer_function.require_module")
    def test_model_loading_error(self, mock_require_module):
        """Test handling of model loading failure."""
        # Clear model cache
        from zvec.extension.sentence_transformer_embedding_function import (
            DefaultLocalSparseEmbedding,
        )

        DefaultLocalSparseEmbedding.clear_cache()
        mock_st = Mock()
        mock_st.SentenceTransformer.side_effect = Exception("Model not found")
        mock_require_module.return_value = mock_st

        with pytest.raises(
            ValueError, match="Failed to load Sentence Transformer model"
        ):
            DefaultLocalDenseEmbedding()

    @patch("zvec.extension.sentence_transformer_function.require_module")
    def test_modelscope_import_error(self, mock_require_module):
        """Test handling of ModelScope import error."""
        mock_st = Mock()

        def require_module_side_effect(module_name):
            if module_name == "sentence_transformers":
                return mock_st
            elif module_name == "modelscope":
                raise ImportError("No module named 'modelscope'")

        mock_require_module.side_effect = require_module_side_effect

        with pytest.raises(
            ImportError, match="ModelScope support requires the 'modelscope' package"
        ):
            DefaultLocalDenseEmbedding(model_source="modelscope")

    @patch("zvec.extension.sentence_transformer_function.require_module")
    def test_embed_dimension_mismatch(self, mock_require_module):
        """Test handling of dimension mismatch in embedding output."""
        # Return embedding with wrong dimension
        fake_embedding = np.random.rand(256).astype(np.float32)

        mock_st = Mock()
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384

        # Configure encode method
        mock_model.encode = Mock(return_value=fake_embedding)

        mock_st.SentenceTransformer.return_value = mock_model
        mock_require_module.return_value = mock_st

        emb_func = DefaultLocalDenseEmbedding()

        with pytest.raises(ValueError, match="Dimension mismatch"):
            emb_func.embed("Test text")

    @pytest.mark.skipif(
        not RUN_INTEGRATION_TESTS,
        reason="Integration test skipped. Set ZVEC_RUN_INTEGRATION_TESTS=1 to run.",
    )
    def test_real_embedding_generation(self):
        """Integration test with real model (requires sentence-transformers).

        To run this test, set environment variable:
            export ZVEC_RUN_INTEGRATION_TESTS=1

        Note: First run will download the model (~80MB).
        """
        emb_func = DefaultLocalDenseEmbedding()

        # Test basic embedding
        vector = emb_func.embed("Hello, world!")
        assert len(vector) == 384
        assert isinstance(vector, list)
        assert all(isinstance(x, float) for x in vector)

        # Test normalization
        norm = np.linalg.norm(vector)
        assert abs(norm - 1.0) < 1e-5

        # Test semantic similarity
        v1 = emb_func.embed("The cat sits on the mat")
        v2 = emb_func.embed("A feline rests on a rug")
        v3 = emb_func.embed("Python programming language")

        similarity_high = np.dot(v1, v2)
        similarity_low = np.dot(v1, v3)
        assert similarity_high > similarity_low

    @pytest.mark.skipif(
        not RUN_INTEGRATION_TESTS,
        reason="Integration test skipped. Set ZVEC_RUN_INTEGRATION_TESTS=1 to run.",
    )
    @patch("zvec.extension.sentence_transformer_function.require_module")
    def test_model_properties(self, mock_require_module):
        """Test model_name and model_source properties."""
        mock_st = Mock()
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.device = "cpu"
        mock_st.SentenceTransformer.return_value = mock_model
        mock_require_module.return_value = mock_st

        # Test Hugging Face
        emb_func_hf = DefaultLocalDenseEmbedding(model_source="huggingface")
        assert emb_func_hf.model_name == "all-MiniLM-L6-v2"
        assert emb_func_hf.model_source == "huggingface"

        # Test ModelScope
        with patch(
            "modelscope.hub.snapshot_download.snapshot_download",
            return_value="/path/to/model",
        ):
            mock_ms = Mock()
            mock_require_module.side_effect = (
                lambda m: mock_st if m == "sentence_transformers" else mock_ms
            )
            emb_func_ms = DefaultLocalDenseEmbedding(model_source="modelscope")
            assert (
                emb_func_ms.model_name == "iic/nlp_gte_sentence-embedding_chinese-small"
            )
            assert emb_func_ms.model_source == "modelscope"


# -----------------------------------
# DefaultLocalSparseEmbedding Test Case
# -----------------------------------
class TestDefaultLocalSparseEmbedding:
    """Test suite for DefaultLocalSparseEmbedding (SPLADE sparse embedding).

    Note:
        DefaultLocalSparseEmbedding uses naver/splade-cocondenser-ensembledistil
        instead of naver/splade-v3 because:

        - splade-v3 is a gated model requiring Hugging Face authentication
        - cocondenser-ensembledistil is publicly accessible
        - Performance difference is minimal (~2%)
        - Avoids "Access to model is restricted" errors

        This allows all users to run tests without authentication setup.
    """

    @patch("zvec.extension.sentence_transformer_function.require_module")
    def test_init_success(self, mock_require_module):
        """Test successful initialization.

        Verifies that DefaultLocalSparseEmbedding initializes with the publicly
        accessible naver/splade-cocondenser-ensembledistil model instead of
        the gated naver/splade-v3 model.
        """
        mock_st = Mock()
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_st.SentenceTransformer.return_value = mock_model
        mock_require_module.return_value = mock_st

        sparse_emb = DefaultLocalSparseEmbedding()

        assert sparse_emb.model_name == "naver/splade-cocondenser-ensembledistil"
        assert sparse_emb.model_source == "huggingface"
        assert sparse_emb.device == "cpu"
        mock_st.SentenceTransformer.assert_called_once_with(
            "naver/splade-cocondenser-ensembledistil",
            device=None,
            trust_remote_code=True,
        )

    @patch("zvec.extension.sentence_transformer_function.require_module")
    def test_init_with_custom_device(self, mock_require_module):
        """Test initialization with custom device."""
        mock_st = Mock()
        mock_model = Mock()
        mock_model.device = "cuda"
        mock_st.SentenceTransformer.return_value = mock_model
        mock_require_module.return_value = mock_st

        sparse_emb = DefaultLocalSparseEmbedding(device="cuda")

        assert sparse_emb.device == "cuda"
        mock_st.SentenceTransformer.assert_called_once_with(
            "naver/splade-cocondenser-ensembledistil",
            device="cuda",
            trust_remote_code=True,
        )

    @pytest.mark.skipif(
        not RUN_INTEGRATION_TESTS,
        reason="Integration test skipped. Set ZVEC_RUN_INTEGRATION_TESTS=1 to run.",
    )
    @patch("zvec.extension.sentence_transformer_function.require_module")
    def test_embed_success(self, mock_require_module):
        """Test successful sparse embedding generation with official API."""
        import numpy as np

        # Clear model cache to ensure fresh mock
        from zvec.extension.sentence_transformer_embedding_function import (
            DefaultLocalSparseEmbedding,
        )

        DefaultLocalSparseEmbedding.clear_cache()

        # Create a mock sparse matrix that simulates scipy.sparse behavior
        # The code will call: sparse_matrix[0].toarray().flatten()
        mock_sparse_matrix = Mock()

        # Create a dense array representation with vocab_size=30522
        vocab_size = 30522
        dense_array = np.zeros(vocab_size)
        # Set specific non-zero values at indices [10, 245, 1023, 5678]
        dense_array[10] = 0.5
        dense_array[245] = 0.8
        dense_array[1023] = 1.2
        dense_array[5678] = 0.3

        # Mock the method chain: sparse_matrix[0].toarray().flatten()
        mock_row = Mock()
        mock_dense = Mock()
        mock_row.toarray.return_value = mock_dense
        mock_dense.flatten.return_value = dense_array
        mock_sparse_matrix.__getitem__ = Mock(return_value=mock_row)

        # Also mock hasattr check for 'toarray'
        mock_sparse_matrix.toarray = Mock()

        mock_st = Mock()
        mock_model = Mock()
        mock_model.device = "cpu"

        # Configure mock methods to return sparse matrix
        # Must set return_value BEFORE hasattr() check in the code
        mock_model.encode_query = Mock(return_value=mock_sparse_matrix)
        mock_model.encode_document = Mock(return_value=mock_sparse_matrix)

        mock_st.SentenceTransformer.return_value = mock_model
        mock_require_module.return_value = mock_st

        sparse_emb = DefaultLocalSparseEmbedding()
        result = sparse_emb.embed("machine learning")

        # Verify result is a dictionary
        assert isinstance(result, dict)
        # Verify keys are integers and values are floats
        assert all(isinstance(k, int) for k in result.keys())
        assert all(isinstance(v, float) for v in result.values())
        # Verify all values are positive
        assert all(v > 0 for v in result.values())
        # Sparse vectors should have specific dimensions
        assert len(result) == 4

        # Verify output is sorted by indices (keys)
        keys = list(result.keys())
        assert keys == sorted(keys), (
            "Sparse vector keys must be sorted in ascending order"
        )

        # Verify expected keys
        assert keys == [10, 245, 1023, 5678]

        # Verify encode_query was called with a list
        mock_model.encode_query.assert_called_once()
        call_args = mock_model.encode_query.call_args[0][0]
        assert isinstance(call_args, list)
        assert call_args == ["machine learning"]

    @patch("zvec.extension.sentence_transformer_function.require_module")
    def test_embed_empty_input(self, mock_require_module):
        """Test embedding with empty input."""
        mock_st = Mock()
        mock_model = Mock()
        mock_st.SentenceTransformer.return_value = mock_model
        mock_require_module.return_value = mock_st

        sparse_emb = DefaultLocalSparseEmbedding()

        with pytest.raises(ValueError, match="Input text cannot be empty"):
            sparse_emb.embed("")

        with pytest.raises(ValueError, match="Input text cannot be empty"):
            sparse_emb.embed("   ")

    @patch("zvec.extension.sentence_transformer_function.require_module")
    def test_embed_non_string_input(self, mock_require_module):
        """Test embedding with non-string input."""
        mock_st = Mock()
        mock_model = Mock()
        mock_st.SentenceTransformer.return_value = mock_model
        mock_require_module.return_value = mock_st

        sparse_emb = DefaultLocalSparseEmbedding()

        with pytest.raises(TypeError, match="Expected 'input' to be str"):
            sparse_emb.embed(123)

        with pytest.raises(TypeError, match="Expected 'input' to be str"):
            sparse_emb.embed(["text"])

    @pytest.mark.skipif(
        not RUN_INTEGRATION_TESTS,
        reason="Integration test skipped. Set ZVEC_RUN_INTEGRATION_TESTS=1 to run.",
    )
    @patch("zvec.extension.sentence_transformer_function.require_module")
    def test_callable_interface(self, mock_require_module):
        """Test that DefaultSparseEmbedding is callable."""
        import numpy as np

        # Clear model cache
        from zvec.extension.sentence_transformer_embedding_function import (
            DefaultLocalSparseEmbedding,
        )

        DefaultLocalSparseEmbedding.clear_cache()

        # Create a mock sparse matrix
        mock_sparse_matrix = Mock()

        # Create a dense array representation with vocab_size=30522
        vocab_size = 30522
        dense_array = np.zeros(vocab_size)
        # Set specific non-zero values at indices [100, 200, 300]
        dense_array[100] = 1.0
        dense_array[200] = 0.5
        dense_array[300] = 0.8

        # Mock the method chain: sparse_matrix[0].toarray().flatten()
        mock_row = Mock()
        mock_dense = Mock()
        mock_row.toarray.return_value = mock_dense
        mock_dense.flatten.return_value = dense_array
        mock_sparse_matrix.__getitem__ = Mock(return_value=mock_row)

        # Also mock hasattr check for 'toarray'
        mock_sparse_matrix.toarray = Mock()

        mock_st = Mock()
        mock_model = Mock()
        mock_model.device = "cpu"

        # Configure mock methods
        mock_model.encode_query = Mock(return_value=mock_sparse_matrix)
        mock_model.encode_document = Mock(return_value=mock_sparse_matrix)

        mock_st.SentenceTransformer.return_value = mock_model
        mock_require_module.return_value = mock_st

        sparse_emb = DefaultLocalSparseEmbedding()

        # Test callable interface
        result = sparse_emb("test input")
        assert isinstance(result, dict)
        assert all(isinstance(k, int) for k in result.keys())

        # Verify sorted output
        keys = list(result.keys())
        assert keys == sorted(keys), "Callable interface must also return sorted keys"
        assert keys == [100, 200, 300]

    @patch("zvec.extension.sentence_transformer_function.require_module")
    def test_model_loading_failure(self, mock_require_module):
        """Test handling of model loading failure."""
        # Clear model cache to ensure the test actually tries to load the model
        from zvec.extension.sentence_transformer_embedding_function import (
            DefaultLocalSparseEmbedding,
        )

        DefaultLocalSparseEmbedding.clear_cache()

        mock_st = Mock()
        mock_st.SentenceTransformer.side_effect = Exception("Model not found")
        mock_require_module.return_value = mock_st

        with pytest.raises(
            ValueError, match="Failed to load Sentence Transformer model"
        ):
            DefaultLocalSparseEmbedding()

    @patch("zvec.extension.sentence_transformer_function.require_module")
    def test_inference_failure(self, mock_require_module):
        """Test handling of inference failure."""
        # Clear model cache
        from zvec.extension.sentence_transformer_embedding_function import (
            DefaultLocalSparseEmbedding,
        )

        DefaultLocalSparseEmbedding.clear_cache()

        mock_st = Mock()
        mock_model = Mock()
        mock_model.device = "cpu"

        # Configure mock methods to raise RuntimeError
        mock_model.encode_query = Mock(side_effect=RuntimeError("CUDA out of memory"))
        mock_model.encode_document = Mock(
            side_effect=RuntimeError("CUDA out of memory")
        )

        mock_st.SentenceTransformer.return_value = mock_model
        mock_require_module.return_value = mock_st

        sparse_emb = DefaultLocalSparseEmbedding()

        with pytest.raises(RuntimeError, match="Failed to generate sparse embedding"):
            sparse_emb.embed("test input")

    @patch("zvec.extension.sentence_transformer_function.require_module")
    def test_sparse_vector_properties(self, mock_require_module):
        """Test properties of sparse vectors (sparsity, non-zero values, sorted order)."""
        import numpy as np

        # Clear model cache
        from zvec.extension.sentence_transformer_embedding_function import (
            DefaultLocalSparseEmbedding,
        )

        DefaultLocalSparseEmbedding.clear_cache()

        # Create a mock sparse matrix that simulates scipy.sparse behavior
        # The code will call: sparse_matrix[0].toarray().flatten()
        mock_sparse_matrix = Mock()

        # Create a dense array representation with vocab_size=30522
        vocab_size = 30522
        dense_array = np.zeros(vocab_size)
        # Set specific non-zero values at indices [50, 100, 200, 400, 500]
        dense_array[50] = 3.0
        dense_array[100] = 2.0
        dense_array[200] = 1.5
        dense_array[400] = 2.5
        dense_array[500] = 1.8

        # Mock the method chain: sparse_matrix[0].toarray().flatten()
        mock_row = Mock()
        mock_dense = Mock()
        mock_row.toarray.return_value = mock_dense
        mock_dense.flatten.return_value = dense_array
        mock_sparse_matrix.__getitem__ = Mock(return_value=mock_row)

        # Also mock hasattr check for 'toarray'
        mock_sparse_matrix.toarray = Mock()

        mock_st = Mock()
        mock_model = Mock()
        mock_model.device = "cpu"

        # Configure mock methods
        mock_model.encode_query = Mock(return_value=mock_sparse_matrix)
        mock_model.encode_document = Mock(return_value=mock_sparse_matrix)

        mock_st.SentenceTransformer.return_value = mock_model
        mock_require_module.return_value = mock_st

        sparse_emb = DefaultLocalSparseEmbedding()
        result = sparse_emb.embed("test")

        # Verify sparsity: result should have much fewer dimensions than vocab_size
        assert len(result) < vocab_size
        # All values should be positive
        assert all(v > 0 for v in result.values())

        # Verify keys are sorted in ascending order
        keys = list(result.keys())
        assert keys == sorted(keys), "Sparse vector keys must be sorted"

        # Verify the specific non-zero indices are present and sorted
        # Expected order: [50, 100, 200, 400, 500] (sorted)
        expected_keys = [50, 100, 200, 400, 500]
        assert keys == expected_keys, f"Expected {expected_keys}, got {keys}"

        # First key should be smallest
        if len(result) > 0:
            first_key = next(iter(result.keys()))
            assert first_key == min(result.keys()), "First key must be the smallest"

    @patch("zvec.extension.sentence_transformer_function.require_module")
    def test_output_sorted_by_indices(self, mock_require_module):
        """Test that output dictionary is always sorted by indices (keys) in ascending order."""
        import numpy as np

        # Clear model cache
        from zvec.extension.sentence_transformer_embedding_function import (
            DefaultLocalSparseEmbedding,
        )

        DefaultLocalSparseEmbedding.clear_cache()

        # Create sparse output with deliberately out-of-order indices
        # Non-sequential indices: 9999, 5, 1234, 77, 500
        mock_sparse_matrix = Mock()

        # Create a dense array representation with vocab_size=30522
        vocab_size = 30522
        dense_array = np.zeros(vocab_size)
        # Set specific non-zero values at out-of-order indices
        dense_array[9999] = 1.5
        dense_array[5] = 2.0
        dense_array[1234] = 0.8
        dense_array[77] = 3.2
        dense_array[500] = 1.1

        # Mock the method chain: sparse_matrix[0].toarray().flatten()
        mock_row = Mock()
        mock_dense = Mock()
        mock_row.toarray.return_value = mock_dense
        mock_dense.flatten.return_value = dense_array
        mock_sparse_matrix.__getitem__ = Mock(return_value=mock_row)

        # Also mock hasattr check for 'toarray'
        mock_sparse_matrix.toarray = Mock()

        mock_st = Mock()
        mock_model = Mock()
        mock_model.device = "cpu"

        # Configure mock methods
        mock_model.encode_query = Mock(return_value=mock_sparse_matrix)
        mock_model.encode_document = Mock(return_value=mock_sparse_matrix)

        mock_st.SentenceTransformer.return_value = mock_model
        mock_require_module.return_value = mock_st

        sparse_emb = DefaultLocalSparseEmbedding()
        result = sparse_emb.embed("test sorting")

        # Extract keys from result
        result_keys = list(result.keys())

        # Verify keys are sorted
        assert result_keys == sorted(result_keys), (
            f"Keys must be sorted in ascending order. "
            f"Got: {result_keys}, Expected: {sorted(result_keys)}"
        )

        # Verify expected keys are present and in correct order
        # Expected sorted order: [5, 77, 500, 1234, 9999]
        expected_sorted_keys = [5, 77, 500, 1234, 9999]
        assert result_keys == expected_sorted_keys, (
            f"All expected keys should be present in sorted order. "
            f"Expected: {expected_sorted_keys}, Got: {result_keys}"
        )

        # Verify first and last keys
        assert result_keys[0] == 5, "First key must be minimum"
        assert result_keys[-1] == 9999, "Last key must be maximum"

        # Verify iteration order matches sorted order
        for i, (key, value) in enumerate(result.items()):
            if i > 0:
                prev_key = list(result.keys())[i - 1]
                assert key > prev_key, (
                    f"Key at position {i} must be greater than previous key"
                )

    @patch("zvec.extension.sentence_transformer_function.require_module")
    def test_device_property(self, mock_require_module):
        """Test device property returns correct device."""
        mock_st = Mock()
        mock_model = Mock()
        mock_model.device = "cuda"
        mock_st.SentenceTransformer.return_value = mock_model
        mock_require_module.return_value = mock_st

        sparse_emb = DefaultLocalSparseEmbedding(device="cuda")
        assert sparse_emb.device == "cuda"

    @pytest.mark.skipif(
        not RUN_INTEGRATION_TESTS,
        reason="Integration test: requires ZVEC_RUN_INTEGRATION_TESTS=1 and model download",
    )
    @patch("zvec.extension.sentence_transformer_function.require_module")
    def test_modelscope_source(self, mock_require_module):
        """Test initialization with ModelScope source."""
        mock_st = Mock()
        mock_ms = Mock()
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_st.SentenceTransformer.return_value = mock_model

        # Mock ModelScope snapshot_download
        with patch(
            "modelscope.hub.snapshot_download.snapshot_download",
            return_value="/cache/splade-cocondenser",
        ):
            mock_require_module.side_effect = (
                lambda m: mock_st if m == "sentence_transformers" else mock_ms
            )

            sparse_emb = DefaultLocalSparseEmbedding(model_source="modelscope")

            assert sparse_emb.model_name == "naver/splade-cocondenser-ensembledistil"
            assert sparse_emb.model_source == "modelscope"

    @pytest.mark.skipif(
        not RUN_INTEGRATION_TESTS,
        reason="Integration test: requires ZVEC_RUN_INTEGRATION_TESTS=1 and model download",
    )
    def test_integration_real_model(self):
        """Integration test with real SPLADE model (requires model download).

        This test uses naver/splade-cocondenser-ensembledistil instead of
        naver/splade-v3 because splade-v3 requires Hugging Face authentication.
        The cocondenser-ensembledistil model is publicly accessible and provides
        comparable performance.

        To run this test:
            export ZVEC_RUN_INTEGRATION_TESTS=1
            pytest tests/test_embedding.py::TestDefaultSparseEmbedding::test_integration_real_model -v

        Note: First run will download ~100MB model from Hugging Face.

        Alternative models:
            If you have access to splade-v3, you can create a custom embedding
            class following the example in DefaultSparseEmbedding docstring.
        """
        # Clear model cache to ensure fresh load
        from zvec.extension.sentence_transformer_embedding_function import (
            DefaultLocalSparseEmbedding,
        )

        DefaultLocalSparseEmbedding.clear_cache()

        sparse_emb = DefaultLocalSparseEmbedding()

        # Test with real input
        text = "machine learning and artificial intelligence"
        result = sparse_emb.embed(text)

        # Verify result structure
        assert isinstance(result, dict)
        assert len(result) > 0
        assert all(isinstance(k, int) and k >= 0 for k in result.keys())
        assert all(isinstance(v, float) and v > 0 for v in result.values())

        # SPLADE typically produces 100-300 non-zero dimensions
        assert 50 < len(result) < 500

        # Verify keys are sorted in ascending order
        keys = list(result.keys())
        assert keys == sorted(keys), "Real model output must be sorted by indices"

        # Test callable interface
        result2 = sparse_emb(text)
        assert result == result2

    @pytest.mark.skipif(
        not RUN_INTEGRATION_TESTS,
        reason="Integration test: requires ZVEC_RUN_INTEGRATION_TESTS=1",
    )
    def test_integration_multiple_inputs(self):
        """Integration test with multiple different inputs."""
        # Clear model cache
        from zvec.extension.sentence_transformer_embedding_function import (
            DefaultLocalSparseEmbedding,
        )

        DefaultLocalSparseEmbedding.clear_cache()

        sparse_emb = DefaultLocalSparseEmbedding()

        texts = [
            "Hello, world!",
            "Machine learning is fascinating",
            "Python programming language",
        ]

        results = [sparse_emb.embed(text) for text in texts]

        # All results should be different
        assert len(results) == 3
        assert all(isinstance(r, dict) for r in results)

        # Different inputs should produce different sparse vectors
        assert results[0] != results[1]
        assert results[1] != results[2]

        # All results must be sorted by indices
        for i, result in enumerate(results):
            keys = list(result.keys())
            assert keys == sorted(keys), f"Result {i} must have sorted keys"


# ----------------------------
# BM25EmbeddingFunction Test Case
# ----------------------------
class TestBM25EmbeddingFunction:
    """Test suite for BM25EmbeddingFunction (BM25-based sparse embedding using DashText SDK)."""

    def test_init_with_built_in_encoder(self):
        """Test successful initialization with built-in encoder (no corpus)."""
        with patch(
            "zvec.extension.bm25_embedding_function.require_module"
        ) as mock_require:
            mock_dashtext = Mock()
            mock_encoder = Mock()
            mock_dashtext.SparseVectorEncoder.default.return_value = mock_encoder
            mock_require.return_value = mock_dashtext

            # Test with default language (Chinese)
            bm25 = BM25EmbeddingFunction()

            assert bm25.corpus_size == 0
            assert bm25.encoding_type == "query"
            assert bm25.language == "zh"
            mock_dashtext.SparseVectorEncoder.default.assert_called_once_with(name="zh")

    def test_init_with_custom_encoder(self):
        """Test successful initialization with custom encoder (with corpus)."""
        corpus = [
            "a cat is a feline and likes to purr",
            "a dog is the human's best friend",
            "a bird is a beautiful animal that can fly",
        ]

        with patch(
            "zvec.extension.bm25_embedding_function.require_module"
        ) as mock_require:
            mock_dashtext = Mock()
            mock_encoder = Mock()
            mock_dashtext.SparseVectorEncoder.return_value = mock_encoder
            mock_require.return_value = mock_dashtext

            bm25 = BM25EmbeddingFunction(corpus=corpus, b=0.75, k1=1.2)

            assert bm25.corpus_size == 3
            assert bm25.encoding_type == "query"
            mock_dashtext.SparseVectorEncoder.assert_called_once_with(b=0.75, k1=1.2)
            mock_encoder.train.assert_called_once_with(corpus)

    def test_init_with_empty_corpus(self):
        """Test initialization with empty corpus raises ValueError."""
        with pytest.raises(ValueError, match="Corpus must be a non-empty list"):
            BM25EmbeddingFunction(corpus=[])

    def test_init_with_invalid_corpus(self):
        """Test initialization with invalid corpus elements."""
        with pytest.raises(ValueError, match="All corpus documents must be strings"):
            BM25EmbeddingFunction(corpus=["text", 123, "another"])

        with pytest.raises(ValueError, match="All corpus documents must be strings"):
            BM25EmbeddingFunction(corpus=[None, "text"])

    def test_init_with_language_parameter(self):
        """Test initialization with different language settings."""
        with patch(
            "zvec.extension.bm25_embedding_function.require_module"
        ) as mock_require:
            mock_dashtext = Mock()
            mock_encoder = Mock()
            mock_dashtext.SparseVectorEncoder.default.return_value = mock_encoder
            mock_require.return_value = mock_dashtext

            # Test English language
            bm25_en = BM25EmbeddingFunction(language="en")
            assert bm25_en.language == "en"
            mock_dashtext.SparseVectorEncoder.default.assert_called_with(name="en")

    def test_init_with_encoding_type(self):
        """Test initialization with different encoding types."""
        with patch(
            "zvec.extension.bm25_embedding_function.require_module"
        ) as mock_require:
            mock_dashtext = Mock()
            mock_encoder = Mock()
            mock_dashtext.SparseVectorEncoder.default.return_value = mock_encoder
            mock_require.return_value = mock_dashtext

            # Test document encoding type
            bm25_doc = BM25EmbeddingFunction(encoding_type="document")
            assert bm25_doc.encoding_type == "document"

    def test_init_with_missing_dashtext_library(self):
        """Test initialization fails when dashtext library is not installed."""
        with patch(
            "zvec.extension.bm25_embedding_function.require_module"
        ) as mock_require:
            mock_require.side_effect = ImportError("dashtext package is required")

            with pytest.raises(ImportError, match="dashtext package is required"):
                BM25EmbeddingFunction()

    def test_embed_with_query_encoding(self):
        """Test successful sparse embedding generation with query encoding."""
        with patch(
            "zvec.extension.bm25_embedding_function.require_module"
        ) as mock_require:
            mock_dashtext = Mock()
            mock_encoder = Mock()

            # Mock encode_queries to return sparse vector
            mock_encoder.encode_queries.return_value = {
                5: 0.89,
                12: 1.45,
                23: 0.67,
                45: 1.12,
            }

            mock_dashtext.SparseVectorEncoder.default.return_value = mock_encoder
            mock_require.return_value = mock_dashtext

            bm25 = BM25EmbeddingFunction(encoding_type="query")
            # Clear LRU cache to ensure fresh call
            bm25.embed.cache_clear()
            result = bm25.embed("cat purr loud")

            # Verify result structure
            assert isinstance(result, dict)
            assert all(isinstance(k, int) for k in result.keys())
            assert all(isinstance(v, float) for v in result.values())

            # Verify all values are positive
            assert all(v > 0 for v in result.values())

            # Verify output is sorted by indices
            keys = list(result.keys())
            assert keys == sorted(keys), "Output must be sorted by indices"

            # Verify expected keys from mock response
            assert result == {5: 0.89, 12: 1.45, 23: 0.67, 45: 1.12}

            # Verify encode_queries was called
            mock_encoder.encode_queries.assert_called_once_with("cat purr loud")

    def test_embed_with_document_encoding(self):
        """Test successful sparse embedding generation with document encoding."""
        with patch(
            "zvec.extension.bm25_embedding_function.require_module"
        ) as mock_require:
            mock_dashtext = Mock()
            mock_encoder = Mock()

            # Mock encode_documents to return sparse vector
            mock_encoder.encode_documents.return_value = {10: 1.5, 20: 2.3}

            mock_dashtext.SparseVectorEncoder.default.return_value = mock_encoder
            mock_require.return_value = mock_dashtext

            bm25 = BM25EmbeddingFunction(encoding_type="document")
            bm25.embed.cache_clear()
            result = bm25.embed("document text")

            assert result == {10: 1.5, 20: 2.3}
            mock_encoder.encode_documents.assert_called_once_with("document text")

    def test_embed_with_empty_input(self):
        """Test embedding with empty input raises ValueError."""
        with patch(
            "zvec.extension.bm25_embedding_function.require_module"
        ) as mock_require:
            mock_dashtext = Mock()
            mock_encoder = Mock()
            mock_dashtext.SparseVectorEncoder.default.return_value = mock_encoder
            mock_require.return_value = mock_dashtext

            bm25 = BM25EmbeddingFunction()

            with pytest.raises(ValueError, match="Input text cannot be empty"):
                bm25.embed("")

            with pytest.raises(ValueError, match="Input text cannot be empty"):
                bm25.embed("   ")

    def test_embed_with_non_string_input(self):
        """Test embedding with non-string input raises TypeError."""
        with patch(
            "zvec.extension.bm25_embedding_function.require_module"
        ) as mock_require:
            mock_dashtext = Mock()
            mock_encoder = Mock()
            mock_dashtext.SparseVectorEncoder.default.return_value = mock_encoder
            mock_require.return_value = mock_dashtext

            bm25 = BM25EmbeddingFunction()

            # Test with hashable non-string types - should get our custom error message
            with pytest.raises(TypeError, match="Expected 'input' to be str"):
                bm25.embed(123)

            with pytest.raises(TypeError, match="Expected 'input' to be str"):
                bm25.embed(None)

            # Test with unhashable type (list)
            # Note: lru_cache raises TypeError("unhashable type: 'list'") before our type check
            # This is still a valid type error, just caught at a different layer
            with pytest.raises(TypeError, match="unhashable type"):
                bm25.embed(["text"])

    def test_embed_callable_interface(self):
        """Test that BM25EmbeddingFunction is callable."""
        with patch(
            "zvec.extension.bm25_embedding_function.require_module"
        ) as mock_require:
            mock_dashtext = Mock()
            mock_encoder = Mock()
            mock_encoder.encode_queries.return_value = {10: 1.5}
            mock_dashtext.SparseVectorEncoder.default.return_value = mock_encoder
            mock_require.return_value = mock_dashtext

            bm25 = BM25EmbeddingFunction()
            bm25.embed.cache_clear()

            # Test callable interface
            result = bm25("test query")
            assert isinstance(result, dict)
            assert 10 in result

    def test_embed_output_sorted_by_indices(self):
        """Test that output is always sorted by indices in ascending order."""
        with patch(
            "zvec.extension.bm25_embedding_function.require_module"
        ) as mock_require:
            mock_dashtext = Mock()
            mock_encoder = Mock()

            # Mock encode_queries with unsorted indices
            mock_encoder.encode_queries.return_value = {
                9999: 1.5,
                5: 2.0,
                1234: 0.8,
                77: 3.2,
                500: 1.1,
            }

            mock_dashtext.SparseVectorEncoder.default.return_value = mock_encoder
            mock_require.return_value = mock_dashtext

            bm25 = BM25EmbeddingFunction()
            bm25.embed.cache_clear()
            result = bm25.embed("test query")

            # Verify keys are sorted
            result_keys = list(result.keys())
            assert result_keys == sorted(result_keys), (
                f"Keys must be sorted. Got: {result_keys}, Expected: {sorted(result_keys)}"
            )

            # Verify expected sorted order: [5, 77, 500, 1234, 9999]
            expected_keys = [5, 77, 500, 1234, 9999]
            assert result_keys == expected_keys

    def test_embed_filters_zero_values(self):
        """Test that zero and negative values are filtered out."""
        with patch(
            "zvec.extension.bm25_embedding_function.require_module"
        ) as mock_require:
            mock_dashtext = Mock()
            mock_encoder = Mock()

            # Mock encode_queries with zero and negative values
            mock_encoder.encode_queries.return_value = {
                0: 1.5,  # Positive - should be included
                1: 0.0,  # Zero - should be filtered
                2: -0.5,  # Negative - should be filtered
            }

            mock_dashtext.SparseVectorEncoder.default.return_value = mock_encoder
            mock_require.return_value = mock_dashtext

            bm25 = BM25EmbeddingFunction()
            bm25.embed.cache_clear()
            result = bm25.embed("test")

            # Only positive token should be in result
            assert 0 in result
            assert 1 not in result  # Zero value filtered
            assert 2 not in result  # Negative value filtered
            assert all(v > 0 for v in result.values())

    def test_properties(self):
        """Test property accessors."""
        corpus = ["doc1", "doc2", "doc3"]

        with patch(
            "zvec.extension.bm25_embedding_function.require_module"
        ) as mock_require:
            mock_dashtext = Mock()
            mock_encoder = Mock()
            mock_dashtext.SparseVectorEncoder.return_value = mock_encoder
            mock_require.return_value = mock_dashtext

            bm25 = BM25EmbeddingFunction(
                corpus=corpus,
                encoding_type="document",
                language="en",
                b=0.8,
                k1=1.5,
                custom_param="test",
            )

            assert bm25.corpus_size == 3
            assert bm25.encoding_type == "document"
            assert bm25.language == "en"
            assert bm25.extra_params == {"custom_param": "test"}

    @pytest.mark.skipif(
        not RUN_INTEGRATION_TESTS,
        reason="Integration test skipped. Set ZVEC_RUN_INTEGRATION_TESTS=1 to run.",
    )
    def test_real_dashtext_bm25_embedding(self):
        """Integration test with real DashText library.

        To run this test:
            export ZVEC_RUN_INTEGRATION_TESTS=1
            pip install dashtext

        Note: This test requires the dashtext package to be installed.
        """
        # Test built-in encoder (Chinese)
        bm25_zh = BM25EmbeddingFunction(language="zh", encoding_type="query")

        query_zh = "什么是向量检索服务"
        result_zh = bm25_zh.embed(query_zh)

        assert isinstance(result_zh, dict)
        assert len(result_zh) > 0
        assert all(isinstance(k, int) for k in result_zh.keys())
        assert all(isinstance(v, float) and v > 0 for v in result_zh.values())

        # Verify sorted output
        keys = list(result_zh.keys())
        assert keys == sorted(keys), "Real DashText BM25 output must be sorted"

        # Test custom corpus
        corpus = [
            "The cat sits on the mat",
            "The dog plays in the garden",
            "Birds fly in the sky",
            "Fish swim in the water",
        ]

        bm25_custom = BM25EmbeddingFunction(corpus=corpus, encoding_type="query")

        query_en = "cat on mat"
        result_en = bm25_custom.embed(query_en)

        assert isinstance(result_en, dict)
        assert len(result_en) > 0
        assert all(isinstance(k, int) for k in result_en.keys())
        assert all(isinstance(v, float) and v > 0 for v in result_en.values())

        # Test callable interface
        result2 = bm25_custom(query_en)
        assert result_en == result2

        # Verify properties
        assert bm25_custom.corpus_size == 4
