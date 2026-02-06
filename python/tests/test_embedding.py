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
from zvec.extension import QwenEmbeddingFunction
from zvec.extension.sentence_transformer_embedding_function import (
    DefaultSentenceTransformerEmbedding,
    SentenceTransformerEmbeddingFunction,
)


# ----------------------------
# QwenEmbeddingFunction Test Case
# ----------------------------
class TestQwenEmbeddingFunction:
    def test_init_with_api_key(self):
        # Test initialization with explicit API key
        embedding_func = QwenEmbeddingFunction(dimension=128, api_key="test_key")
        assert embedding_func.dimension == 128
        assert embedding_func.model == "text-embedding-v4"
        assert embedding_func._api_key == "test_key"

    @patch.dict(os.environ, {"DASHSCOPE_API_KEY": "env_key"})
    def test_init_with_env_api_key(self):
        # Test initialization with API key from environment
        embedding_func = QwenEmbeddingFunction(dimension=128)
        assert embedding_func._api_key == "env_key"

    def test_init_without_api_key(self):
        # Test initialization without API key raises ValueError
        with pytest.raises(ValueError, match="DashScope API key is required"):
            QwenEmbeddingFunction(dimension=128)

    @patch.dict(os.environ, {"DASHSCOPE_API_KEY": ""})
    def test_init_with_empty_env_api_key(self):
        # Test initialization with empty API key from environment
        with pytest.raises(ValueError, match="DashScope API key is required"):
            QwenEmbeddingFunction(dimension=128)

    def test_model_property(self):
        embedding_func = QwenEmbeddingFunction(dimension=128, api_key="test_key")
        assert embedding_func.model == "text-embedding-v4"

        embedding_func = QwenEmbeddingFunction(
            dimension=128, model="custom-model", api_key="test_key"
        )
        assert embedding_func.model == "custom-model"

    @patch("zvec.extension.embedding.require_module")
    def test_embed_with_empty_text(self, mock_require_module):
        # Test embed method with empty text raises ValueError
        embedding_func = QwenEmbeddingFunction(dimension=128, api_key="test_key")

        with pytest.raises(
            ValueError, match="Input text cannot be empty or whitespace only"
        ):
            embedding_func.embed("")

        with pytest.raises(TypeError):
            embedding_func.embed(None)

    @patch("zvec.extension.embedding.require_module")
    def test_embed_success(self, mock_require_module):
        # Test successful embedding
        mock_dashscope = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = HTTPStatus.OK
        mock_response.output = {"embeddings": [{"embedding": [0.1, 0.2, 0.3]}]}
        mock_dashscope.TextEmbedding.call.return_value = mock_response
        mock_require_module.return_value = mock_dashscope

        embedding_func = QwenEmbeddingFunction(dimension=128, api_key="test_key")
        result = embedding_func.embed("test text")

        assert result == [0.1, 0.2, 0.3]
        mock_dashscope.TextEmbedding.call.assert_called_once_with(
            model="text-embedding-v4",
            input="test text",
            dimension=128,
            output_type="dense",
        )

    @patch("zvec.extension.embedding.require_module")
    def test_embed_http_error(self, mock_require_module):
        # Test embedding with HTTP error
        mock_dashscope = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = HTTPStatus.BAD_REQUEST
        mock_response.message = "Bad Request"
        mock_dashscope.TextEmbedding.call.return_value = mock_response
        mock_require_module.return_value = mock_dashscope

        embedding_func = QwenEmbeddingFunction(dimension=128, api_key="test_key")

        with pytest.raises(ValueError):
            embedding_func.embed("test text")

    @patch("zvec.extension.embedding.require_module")
    def test_embed_invalid_response(self, mock_require_module):
        # Test embedding with invalid response (wrong number of embeddings)
        mock_dashscope = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = HTTPStatus.OK
        mock_response.output.embeddings = []
        mock_dashscope.TextEmbedding.call.return_value = mock_response
        mock_require_module.return_value = mock_dashscope

        embedding_func = QwenEmbeddingFunction(dimension=128, api_key="test_key")

        with pytest.raises(ValueError):
            embedding_func.embed("test text")

    @pytest.mark.skip(reason="Qwen Embedding is not available in CI")
    def test_embed(self):
        # Test embedding with invalid dimension
        embedding_func = QwenEmbeddingFunction(dimension=128, api_key="xxx")
        dense = embedding_func("test text")
        assert len(dense) == 128


# ----------------------------
# DefaultSentenceTransformerEmbedding Test Case
# ----------------------------
class TestDefaultSentenceTransformerEmbedding:
    """Test cases for DefaultSentenceTransformerEmbedding."""

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
        emb_func = DefaultSentenceTransformerEmbedding()

        # Assertions
        assert emb_func.dimension == 384
        assert emb_func.model_name == "all-MiniLM-L6-v2"
        assert emb_func.device == "cpu"
        mock_st.SentenceTransformer.assert_called_once_with(
            "all-MiniLM-L6-v2", device=None
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

        emb_func = DefaultSentenceTransformerEmbedding(device="cuda")

        assert emb_func.device == "cuda"
        mock_st.SentenceTransformer.assert_called_once_with(
            "all-MiniLM-L6-v2", device="cuda"
        )

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

        emb_func = DefaultSentenceTransformerEmbedding()
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

        emb_func = DefaultSentenceTransformerEmbedding(normalize_embeddings=True)
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

        emb_func = DefaultSentenceTransformerEmbedding()

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

        emb_func = DefaultSentenceTransformerEmbedding()

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

        emb_func = DefaultSentenceTransformerEmbedding()

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

        emb_func = DefaultSentenceTransformerEmbedding()

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
            DefaultSentenceTransformerEmbedding()

    @pytest.mark.skip(
        reason="Requires sentence-transformers installed and model download"
    )
    def test_real_embedding_generation(self):
        """Integration test with real model (skipped in CI)."""
        emb_func = DefaultSentenceTransformerEmbedding()

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
            "all-mpnet-base-v2", device=None
        )
