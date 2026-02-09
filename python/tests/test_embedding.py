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
from zvec.extension.openai_embedding_function import OpenAIDenseEmbedding
from zvec.extension.qwen_embedding_function import QwenDenseEmbedding
from zvec.extension.sentence_transformer_embedding_function import (
    DefaultDenseEmbedding,
    SentenceTransformerEmbeddingFunction,
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

    @patch("zvec.extension.qwen_embedding_function.require_module")
    def test_embed_with_empty_text(self, mock_require_module):
        # Test embed method with empty text raises ValueError
        embedding_func = QwenDenseEmbedding(dimension=128, api_key="test_key")

        with pytest.raises(
            ValueError, match="Input text cannot be empty or whitespace only"
        ):
            embedding_func.embed("")

        with pytest.raises(TypeError):
            embedding_func.embed(None)

    @patch("zvec.extension.qwen_embedding_function.require_module")
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

    @patch("zvec.extension.qwen_embedding_function.require_module")
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

    @patch("zvec.extension.qwen_embedding_function.require_module")
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

    @patch("zvec.extension.openai_embedding_function.require_module")
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

    @patch("zvec.extension.openai_embedding_function.require_module")
    def test_embed_with_non_string_input(self, mock_require_module):
        """Test embed method with non-string input raises TypeError."""
        embedding_func = OpenAIDenseEmbedding(api_key="sk-test")

        with pytest.raises(TypeError, match="Expected 'input' to be str"):
            embedding_func.embed(123)

        with pytest.raises(TypeError, match="Expected 'input' to be str"):
            embedding_func.embed(None)

    @patch("zvec.extension.openai_embedding_function.require_module")
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

    @patch("zvec.extension.openai_embedding_function.require_module")
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

    @patch("zvec.extension.openai_embedding_function.require_module")
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

    @patch("zvec.extension.openai_embedding_function.require_module")
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

    @patch("zvec.extension.openai_embedding_function.require_module")
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

    @patch("zvec.extension.openai_embedding_function.require_module")
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

    @patch("zvec.extension.openai_embedding_function.require_module")
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
# DefaultDenseEmbedding Test Case
# ----------------------------
class TestDefaultDenseEmbedding:
    """Test cases for DefaultDenseEmbedding."""

    @patch("zvec.extension.sentence_transformer_embedding_function.require_module")
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
        emb_func = DefaultDenseEmbedding()

        # Assertions
        assert emb_func.dimension == 384
        assert emb_func.model_name == "all-MiniLM-L6-v2"
        assert emb_func.model_source == "huggingface"
        assert emb_func.device == "cpu"
        mock_st.SentenceTransformer.assert_called_once_with(
            "all-MiniLM-L6-v2", device=None, trust_remote_code=True
        )

    @patch("zvec.extension.sentence_transformer_embedding_function.require_module")
    def test_init_with_custom_device(self, mock_require_module):
        """Test initialization with custom device."""
        mock_st = Mock()
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.device = "cuda"
        mock_st.SentenceTransformer.return_value = mock_model
        mock_require_module.return_value = mock_st

        emb_func = DefaultDenseEmbedding(device="cuda")

        assert emb_func.device == "cuda"
        mock_st.SentenceTransformer.assert_called_once_with(
            "all-MiniLM-L6-v2", device="cuda", trust_remote_code=True
        )

    @patch("zvec.extension.sentence_transformer_embedding_function.require_module")
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
            emb_func = DefaultDenseEmbedding(model_source="modelscope")

        # Assertions
        assert emb_func.dimension == 384
        assert emb_func.model_name == "iic/nlp_gte_sentence-embedding_chinese-small"
        assert emb_func.model_source == "modelscope"

    @patch("zvec.extension.sentence_transformer_embedding_function.require_module")
    def test_init_with_invalid_model_source(self, mock_require_module):
        """Test initialization with invalid model_source raises ValueError."""
        mock_st = Mock()
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.SentenceTransformer.return_value = mock_model
        mock_require_module.return_value = mock_st

        with pytest.raises(ValueError, match="Invalid model_source"):
            DefaultDenseEmbedding(model_source="invalid_source")

    @patch("zvec.extension.sentence_transformer_embedding_function.require_module")
    def test_embed_success(self, mock_require_module):
        """Test successful embedding generation."""
        mock_st = Mock()
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384

        # Mock embedding output
        fake_embedding = np.random.rand(384).astype(np.float32)
        mock_model.encode.return_value = fake_embedding
        mock_st.SentenceTransformer.return_value = mock_model
        mock_require_module.return_value = mock_st

        emb_func = DefaultDenseEmbedding()
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

    @patch("zvec.extension.sentence_transformer_embedding_function.require_module")
    def test_embed_with_normalization(self, mock_require_module):
        """Test embedding with L2 normalization."""
        mock_st = Mock()
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384

        # Create a normalized vector
        fake_embedding = np.random.rand(384).astype(np.float32)
        fake_embedding = fake_embedding / np.linalg.norm(fake_embedding)
        mock_model.encode.return_value = fake_embedding
        mock_st.SentenceTransformer.return_value = mock_model
        mock_require_module.return_value = mock_st

        emb_func = DefaultDenseEmbedding(normalize_embeddings=True)
        result = emb_func.embed("Test sentence")

        # Check if vector is normalized (L2 norm should be close to 1.0)
        result_array = np.array(result)
        norm = np.linalg.norm(result_array)
        assert abs(norm - 1.0) < 1e-5

    @patch("zvec.extension.sentence_transformer_embedding_function.require_module")
    def test_embed_empty_string(self, mock_require_module):
        """Test embedding with empty string raises ValueError."""
        mock_st = Mock()
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.SentenceTransformer.return_value = mock_model
        mock_require_module.return_value = mock_st

        emb_func = DefaultDenseEmbedding()

        with pytest.raises(ValueError, match="Input text cannot be empty"):
            emb_func.embed("")

        with pytest.raises(ValueError, match="Input text cannot be empty"):
            emb_func.embed("   ")

    @patch("zvec.extension.sentence_transformer_embedding_function.require_module")
    def test_embed_non_string_input(self, mock_require_module):
        """Test embedding with non-string input raises TypeError."""
        mock_st = Mock()
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.SentenceTransformer.return_value = mock_model
        mock_require_module.return_value = mock_st

        emb_func = DefaultDenseEmbedding()

        with pytest.raises(TypeError, match="Expected 'input' to be str"):
            emb_func.embed(123)

        with pytest.raises(TypeError, match="Expected 'input' to be str"):
            emb_func.embed(None)

    @patch("zvec.extension.sentence_transformer_embedding_function.require_module")
    def test_embed_callable(self, mock_require_module):
        """Test that embedding function is callable."""
        mock_st = Mock()
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        fake_embedding = np.random.rand(384).astype(np.float32)
        mock_model.encode.return_value = fake_embedding
        mock_st.SentenceTransformer.return_value = mock_model
        mock_require_module.return_value = mock_st

        emb_func = DefaultDenseEmbedding()

        # Test calling the function directly
        result = emb_func("Test text")
        assert isinstance(result, list)
        assert len(result) == 384

    @patch("zvec.extension.sentence_transformer_embedding_function.require_module")
    def test_semantic_similarity(self, mock_require_module):
        """Test semantic similarity between similar and different texts."""
        mock_st = Mock()
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384

        # Create mock embeddings for similar and different texts
        similar_emb_1 = np.array([1.0, 0.0, 0.0] + [0.0] * 381, dtype=np.float32)
        similar_emb_2 = np.array([0.9, 0.1, 0.0] + [0.0] * 381, dtype=np.float32)
        different_emb = np.array([0.0, 0.0, 1.0] + [0.0] * 381, dtype=np.float32)

        # Normalize
        similar_emb_1 = similar_emb_1 / np.linalg.norm(similar_emb_1)
        similar_emb_2 = similar_emb_2 / np.linalg.norm(similar_emb_2)
        different_emb = different_emb / np.linalg.norm(different_emb)

        mock_model.encode.side_effect = [similar_emb_1, similar_emb_2, different_emb]
        mock_st.SentenceTransformer.return_value = mock_model
        mock_require_module.return_value = mock_st

        emb_func = DefaultDenseEmbedding()

        v1 = emb_func.embed("The cat sits on the mat")
        v2 = emb_func.embed("A feline rests on a rug")
        v3 = emb_func.embed("Python programming")

        # Calculate similarities
        similarity_high = np.dot(v1, v2)
        similarity_low = np.dot(v1, v3)

        assert similarity_high > similarity_low

    @patch("zvec.extension.sentence_transformer_embedding_function.require_module")
    def test_model_loading_error(self, mock_require_module):
        """Test handling of model loading failure."""
        mock_st = Mock()
        mock_st.SentenceTransformer.side_effect = Exception("Model not found")
        mock_require_module.return_value = mock_st

        with pytest.raises(
            ValueError, match="Failed to load Sentence Transformer model"
        ):
            DefaultDenseEmbedding()

    @patch("zvec.extension.sentence_transformer_embedding_function.require_module")
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
            DefaultDenseEmbedding(model_source="modelscope")

    @patch("zvec.extension.sentence_transformer_embedding_function.require_module")
    def test_embed_dimension_mismatch(self, mock_require_module):
        """Test handling of dimension mismatch in embedding output."""
        mock_st = Mock()
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384

        # Return embedding with wrong dimension
        fake_embedding = np.random.rand(256).astype(np.float32)
        mock_model.encode.return_value = fake_embedding
        mock_st.SentenceTransformer.return_value = mock_model
        mock_require_module.return_value = mock_st

        emb_func = DefaultDenseEmbedding()

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
        emb_func = DefaultDenseEmbedding()

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


# ----------------------------
# Custom SentenceTransformerEmbeddingFunction Test
# ----------------------------
class TestCustomSentenceTransformerEmbedding:
    """Test cases for custom subclasses of SentenceTransformerEmbeddingFunction."""

    @patch("zvec.extension.sentence_transformer_embedding_function.require_module")
    def test_custom_model_inheritance(self, mock_require_module):
        """Test creating a custom embedding function with different model."""

        # Create a custom embedding class
        class CustomEmbedding(SentenceTransformerEmbeddingFunction):
            def __init__(self, device=None):
                super().__init__(
                    model_name="all-mpnet-base-v2",
                    device=device,
                    normalize_embeddings=True,
                    batch_size=64,
                )

        mock_st = Mock()
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.device = "cpu"
        mock_st.SentenceTransformer.return_value = mock_model
        mock_require_module.return_value = mock_st

        custom_emb = CustomEmbedding()

        assert custom_emb.dimension == 768
        assert custom_emb.model_name == "all-mpnet-base-v2"
        mock_st.SentenceTransformer.assert_called_once_with(
            "all-mpnet-base-v2", device=None, trust_remote_code=True
        )

    @patch("zvec.extension.sentence_transformer_embedding_function.require_module")
    def test_custom_chinese_model_with_modelscope(self, mock_require_module):
        """Test creating a custom Chinese embedding function with ModelScope."""

        # Create a custom Chinese embedding class
        class ChineseEmbedding(SentenceTransformerEmbeddingFunction):
            def __init__(self, device=None):
                super().__init__(
                    model_name="iic/nlp_gte_sentence-embedding_chinese-small",
                    model_source="modelscope",
                    device=device,
                    normalize_embeddings=True,
                )

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

        with patch(
            "modelscope.hub.snapshot_download.snapshot_download",
            return_value="/path/to/cached/chinese/model",
        ):
            chinese_emb = ChineseEmbedding()

        assert chinese_emb.dimension == 384
        assert chinese_emb.model_name == "iic/nlp_gte_sentence-embedding_chinese-small"
        assert chinese_emb.model_source == "modelscope"

    @patch("zvec.extension.sentence_transformer_embedding_function.require_module")
    def test_model_properties(self, mock_require_module):
        """Test model_name and model_source properties."""
        mock_st = Mock()
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.device = "cpu"
        mock_st.SentenceTransformer.return_value = mock_model
        mock_require_module.return_value = mock_st

        # Test Hugging Face
        emb_func_hf = DefaultDenseEmbedding(model_source="huggingface")
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
            emb_func_ms = DefaultDenseEmbedding(model_source="modelscope")
            assert (
                emb_func_ms.model_name == "iic/nlp_gte_sentence-embedding_chinese-small"
            )
            assert emb_func_ms.model_source == "modelscope"
