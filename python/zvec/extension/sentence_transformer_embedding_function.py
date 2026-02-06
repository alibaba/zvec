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

from abc import ABC
from typing import Optional

import numpy as np

from ..common.constants import TEXT, DenseVectorType
from ..tool import require_module
from ..typing import DataType
from .embedding_function import DenseEmbeddingFunction


class SentenceTransformerEmbeddingFunction(DenseEmbeddingFunction[TEXT], ABC):
    """Abstract base class for Sentence Transformer-based embedding functions.

    This abstract class provides a foundation for text-to-vector embedding capabilities
    using the sentence-transformers library. It inherits from ``DenseEmbeddingFunction``
    and adds Sentence Transformer-specific features. You can inherit this class to
    create custom embedding functions with specific models or configurations.

    The implementation supports all models from Hugging Face Hub compatible with
    sentence-transformers, runs locally without API calls, and supports both
    CPU and GPU acceleration.

    Args:
        model_name (str): Hugging Face model identifier or local path.
            Common options:
            - ``"all-MiniLM-L6-v2"``: Fast, 384 dims, good for general use
            - ``"all-mpnet-base-v2"``: High quality, 768 dims
            - ``"paraphrase-multilingual-MiniLM-L12-v2"``: Multilingual, 384 dims
            - ``"multi-qa-MiniLM-L6-cos-v1"``: Optimized for Q&A, 384 dims
            Browse more: https://huggingface.co/models?library=sentence-transformers
        device (Optional[str], optional): Device to run the model on.
            Options: ``"cpu"``, ``"cuda"``, ``"mps"`` (for Apple Silicon), or ``None``
            for automatic detection. Defaults to ``None``.
        normalize_embeddings (bool, optional): Whether to normalize embeddings to
            unit length (L2 normalization). Useful for cosine similarity.
            Defaults to ``True``.
        batch_size (int, optional): Batch size for encoding when processing
            multiple texts. Only affects performance. Defaults to ``32``.

    Attributes:
        dimension (int): The embedding vector dimension (auto-detected from model).
        data_type (DataType): Always ``DataType.VECTOR_FP32`` for this implementation.
        model_name (str): The model identifier being used.
        device (str): The device the model is running on.

    Raises:
        ValueError: If the model cannot be loaded or input is invalid.
        TypeError: If input to ``embed()`` is not a string.
        RuntimeError: If model inference fails.

    Note:
        - Requires Python 3.10, 3.11, or 3.12
        - Requires the ``sentence-transformers`` package:
          ``pip install sentence-transformers``
        - First run downloads the model from Hugging Face Hub (~80MB-500MB)
        - Models are cached locally in ``~/.cache/torch/sentence_transformers/``
        - GPU acceleration available with CUDA or Apple Silicon (MPS)
        - No API keys or network required after initial model download
        - Subclasses should typically only need to set the default ``model_name``

    Examples:
        >>> # Using built-in concrete implementation
        >>> from zvec.extension import DefaultSentenceTransformerEmbedding
        >>>
        >>> emb_func = DefaultSentenceTransformerEmbedding()
        >>> vector = emb_func.embed("Hello, world!")
        >>> len(vector)
        384

        >>> # Creating custom embedding function with specific model
        >>> class MyCustomEmbedding(SentenceTransformerEmbeddingFunction):
        ...     def __init__(self, device: Optional[str] = None):
        ...         super().__init__(
        ...             model_name="all-mpnet-base-v2",
        ...             device=device,
        ...             normalize_embeddings=True,
        ...             batch_size=64
        ...         )
        >>>
        >>> custom_emb = MyCustomEmbedding(device="cuda")
        >>> vector = custom_emb.embed("Machine learning")
        >>> len(vector)
        768

        >>> # Creating multilingual embedding function
        >>> class MultilingualEmbedding(SentenceTransformerEmbeddingFunction):
        ...     def __init__(self):
        ...         super().__init__(
        ...             model_name="paraphrase-multilingual-MiniLM-L12-v2",
        ...             normalize_embeddings=True
        ...         )
        >>>
        >>> multilingual = MultilingualEmbedding()
        >>> vector_en = multilingual.embed("Hello")
        >>> vector_zh = multilingual.embed("你好")
        >>> # Both produce embeddings in same 384-dim space

    See Also:
        - ``DenseEmbeddingFunction``: Base class for dense embeddings
        - ``DefaultSentenceTransformerEmbedding``: Concrete implementation with all-MiniLM-L6-v2
        - ``QwenEmbeddingFunction``: Alternative using Qwen API
        - ``SparseEmbeddingFunction``: Base class for sparse embeddings
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
    ):
        """Initialize the Sentence Transformer embedding function.

        Args:
            model_name (str): Hugging Face model name or local path. Required.
            device (Optional[str]): Target device ("cpu", "cuda", "mps", or None).
                None means automatic detection.
            normalize_embeddings (bool): Whether to L2-normalize output vectors.
                Defaults to True.
            batch_size (int): Batch size for encoding. Defaults to 32.

        Raises:
            ImportError: If sentence-transformers is not installed.
            ValueError: If model cannot be loaded.
        """
        self._model_name = model_name
        self._device = device
        self._normalize_embeddings = normalize_embeddings
        self._batch_size = batch_size
        self._model = None

        # Load model and get dimension
        model = self._get_model()
        dimension = model.get_sentence_embedding_dimension()

        super().__init__(dimension, DataType.VECTOR_FP32)

    @property
    def model_name(self) -> str:
        """str: The Sentence Transformer model name currently in use.

        Returns:
            str: Model identifier (e.g., "all-MiniLM-L6-v2").
        """
        return self._model_name

    @property
    def device(self) -> str:
        """str: The device the model is running on.

        Returns:
            str: Device name (e.g., "cpu", "cuda", "mps").
        """
        if self._model is not None:
            return str(self._model.device)
        return self._device or "cpu"

    def _get_model(self):
        """Load or retrieve the Sentence Transformer model.

        Returns:
            SentenceTransformer: The loaded model instance.

        Raises:
            ImportError: If sentence-transformers package is not installed.
            ValueError: If model cannot be loaded.
        """
        if self._model is None:
            try:
                sentence_transformers = require_module("sentence_transformers")
                self._model = sentence_transformers.SentenceTransformer(
                    self._model_name, device=self._device
                )
            except Exception as e:
                raise ValueError(
                    f"Failed to load Sentence Transformer model '{self._model_name}': {e!s}"
                ) from e
        return self._model

    def embed(self, input: str) -> DenseVectorType:
        """Generate dense embedding vector for the input text.

        This method uses the Sentence Transformer model to convert input text
        into a dense vector representation. The model runs locally without
        requiring API calls.

        Args:
            input (str): Input text string to embed. Must be non-empty after
                stripping whitespace. Maximum length depends on the model used
                (typically 128-512 tokens for most models).

        Returns:
            DenseVectorType: A list of floats representing the embedding vector.
                Length equals ``self.dimension``. If ``normalize_embeddings=True``,
                the vector has unit length. Example:
                ``[0.123, -0.456, 0.789, ...]``

        Raises:
            TypeError: If ``input`` is not a string.
            ValueError: If input is empty or whitespace-only.
            RuntimeError: If model inference fails.

        Examples:
            >>> emb = SentenceTransformerEmbeddingFunction()
            >>> vector = emb.embed("Natural language processing")
            >>> len(vector)
            384
            >>> isinstance(vector[0], float)
            True

            >>> # Normalized vectors have unit length
            >>> import numpy as np
            >>> emb = SentenceTransformerEmbeddingFunction(normalize_embeddings=True)
            >>> vector = emb.embed("Test sentence")
            >>> np.linalg.norm(vector)
            1.0

            >>> # Error: empty input
            >>> emb.embed("   ")
            ValueError: Input text cannot be empty or whitespace only

            >>> # Error: non-string input
            >>> emb.embed(123)
            TypeError: Expected 'input' to be str, got int

            >>> # Semantic similarity example
            >>> v1 = emb.embed("The cat sits on the mat")
            >>> v2 = emb.embed("A feline rests on a rug")
            >>> similarity = np.dot(v1, v2)  # High similarity due to semantic meaning
            >>> similarity > 0.7
            True

        Note:
            - First call may be slower due to model loading
            - Subsequent calls are much faster as the model stays in memory
            - For batch processing, consider encoding multiple texts together
              (though this method handles single texts only)
            - GPU acceleration provides 5-10x speedup over CPU
        """
        if not isinstance(input, str):
            raise TypeError(f"Expected 'input' to be str, got {type(input).__name__}")

        input = input.strip()
        if not input:
            raise ValueError("Input text cannot be empty or whitespace only")

        try:
            model = self._get_model()
            embedding = model.encode(
                input,
                convert_to_numpy=True,
                normalize_embeddings=self._normalize_embeddings,
                batch_size=self._batch_size,
            )

            # Convert numpy array to list
            if isinstance(embedding, np.ndarray):
                embedding_list = embedding.tolist()
            else:
                embedding_list = list(embedding)

            # Validate dimension
            if len(embedding_list) != self.dimension:
                raise ValueError(
                    f"Dimension mismatch: expected {self.dimension}, "
                    f"got {len(embedding_list)}"
                )

            return embedding_list

        except Exception as e:
            if isinstance(e, (TypeError, ValueError)):
                raise
            raise RuntimeError(f"Failed to generate embedding: {e!s}") from e


class DefaultSentenceTransformerEmbedding(SentenceTransformerEmbeddingFunction):
    """Default Sentence Transformer embedding using all-MiniLM-L6-v2 model.

    This is a concrete implementation of ``SentenceTransformerEmbeddingFunction``
    that uses the ``all-MiniLM-L6-v2`` model by default. This model provides a
    good balance between speed and quality for general-purpose text embedding.

    The model produces 384-dimensional embeddings and is optimized for semantic
    similarity tasks. It runs locally without requiring API keys.

    Args:
        device (Optional[str], optional): Device to run the model on.
            Options: ``"cpu"``, ``"cuda"``, ``"mps"`` (for Apple Silicon), or ``None``
            for automatic detection. Defaults to ``None``.
        normalize_embeddings (bool, optional): Whether to normalize embeddings to
            unit length (L2 normalization). Defaults to ``True``.
        batch_size (int, optional): Batch size for encoding. Defaults to ``32``.

    Attributes:
        dimension (int): Always 384 for all-MiniLM-L6-v2.
        data_type (DataType): Always ``DataType.VECTOR_FP32``.
        model_name (str): Always "all-MiniLM-L6-v2".

    Raises:
        ValueError: If the model cannot be loaded or input is invalid.
        TypeError: If input to ``embed()`` is not a string.
        RuntimeError: If model inference fails.

    Note:
        - Requires Python 3.10, 3.11, or 3.12
        - Requires the ``sentence-transformers`` package:
          ``pip install sentence-transformers``
        - First run downloads the model (~80MB) from Hugging Face Hub
        - Model is cached locally in ``~/.cache/torch/sentence_transformers/``
        - No API keys or network required after initial download
        - Inference speed: ~1000 sentences/sec on CPU, ~10000 on GPU

    Examples:
        >>> # Basic usage with CPU
        >>> from zvec.extension import DefaultSentenceTransformerEmbedding
        >>>
        >>> emb_func = DefaultSentenceTransformerEmbedding()
        >>> vector = emb_func.embed("Hello, world!")
        >>> len(vector)
        384
        >>> isinstance(vector, list)
        True

        >>> # Using GPU for faster inference
        >>> emb_func = DefaultSentenceTransformerEmbedding(device="cuda")
        >>> vector = emb_func("Machine learning is fascinating")
        >>> # Normalized vector has unit length
        >>> import numpy as np
        >>> np.linalg.norm(vector)
        1.0

        >>> # Batch processing
        >>> texts = ["First text", "Second text", "Third text"]
        >>> vectors = [emb_func.embed(text) for text in texts]
        >>> len(vectors)
        3
        >>> all(len(v) == 384 for v in vectors)
        True

        >>> # Semantic similarity
        >>> v1 = emb_func.embed("The cat sits on the mat")
        >>> v2 = emb_func.embed("A feline rests on a rug")
        >>> v3 = emb_func.embed("Python programming")
        >>> similarity_high = np.dot(v1, v2)  # Similar sentences
        >>> similarity_low = np.dot(v1, v3)   # Different topics
        >>> similarity_high > similarity_low
        True

        >>> # Error handling
        >>> try:
        ...     emb_func.embed("")  # Empty string
        ... except ValueError as e:
        ...     print(f"Error: {e}")
        Error: Input text cannot be empty or whitespace only

    See Also:
        - ``SentenceTransformerEmbeddingFunction``: Base class for custom models
        - ``DenseEmbeddingFunction``: Abstract base for all dense embeddings
        - ``QwenEmbeddingFunction``: Alternative using Qwen API
    """

    def __init__(
        self,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
    ):
        """Initialize with all-MiniLM-L6-v2 model.

        Args:
            device (Optional[str]): Target device ("cpu", "cuda", "mps", or None).
                Defaults to None (automatic detection).
            normalize_embeddings (bool): Whether to L2-normalize output vectors.
                Defaults to True.
            batch_size (int): Batch size for encoding. Defaults to 32.

        Raises:
            ImportError: If sentence-transformers is not installed.
            ValueError: If model cannot be loaded.
        """
        super().__init__(
            model_name="all-MiniLM-L6-v2",
            device=device,
            normalize_embeddings=normalize_embeddings,
            batch_size=batch_size,
        )
