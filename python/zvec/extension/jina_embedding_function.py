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

from functools import lru_cache
from typing import Optional

from ..common.constants import TEXT, DenseVectorType
from .embedding_function import DenseEmbeddingFunction
from .jina_function import JinaFunctionBase


class JinaDenseEmbedding(JinaFunctionBase, DenseEmbeddingFunction[TEXT]):
    """Dense text embedding function using Jina AI API.

    This class provides text-to-vector embedding capabilities using Jina AI's
    embedding models. It inherits from ``DenseEmbeddingFunction`` and implements
    dense text embedding via the Jina Embeddings API (OpenAI-compatible).

    Jina Embeddings v5 models support task-specific embedding through the
    ``task`` parameter, which optimizes the embedding for different use cases
    such as retrieval, text matching, or classification. They also support
    Matryoshka Representation Learning, allowing flexible output dimensions.

    Args:
        model (str, optional): Jina embedding model identifier.
            Defaults to ``"jina-embeddings-v5-text-nano"``. Available models:
            - ``"jina-embeddings-v5-text-nano"``: 768 dims, 239M params, 8K context
            - ``"jina-embeddings-v5-text-small"``: 1024 dims, 677M params, 32K context
        dimension (Optional[int], optional): Desired output embedding dimension.
            If ``None``, uses model's default dimension. Supports Matryoshka
            dimensions: 32, 64, 128, 256, 512, 768 (nano) / 1024 (small).
            Defaults to ``None``.
        api_key (Optional[str], optional): Jina API authentication key.
            If ``None``, reads from ``JINA_API_KEY`` environment variable.
            Obtain your key from: https://jina.ai/api-dashboard
        task (Optional[str], optional): Task type to optimize embeddings for.
            Defaults to ``None``. Valid values:
            - ``"retrieval.query"``: For search queries
            - ``"retrieval.passage"``: For documents/passages to be searched
            - ``"text-matching"``: For symmetric text similarity
            - ``"classification"``: For text classification
            - ``"separation"``: For clustering/separation tasks

    Attributes:
        dimension (int): The embedding vector dimension.
        data_type (DataType): Always ``DataType.VECTOR_FP32`` for this implementation.
        model (str): The Jina model name being used.
        task (Optional[str]): The task type for embedding optimization.

    Raises:
        ValueError: If API key is not provided and not found in environment,
            if task is not a valid task type, or if API returns an error response.
        TypeError: If input to ``embed()`` is not a string.
        RuntimeError: If network error or Jina service error occurs.

    Note:
        - Requires Python 3.10, 3.11, or 3.12
        - Requires the ``openai`` package: ``pip install openai``
        - Jina API is OpenAI-compatible, so it uses the ``openai`` Python client
        - Embedding results are cached (LRU cache, maxsize=10) to reduce API calls
        - For retrieval tasks, use ``"retrieval.query"`` for queries and
          ``"retrieval.passage"`` for documents
        - API usage requires a Jina API key from https://jina.ai/api-dashboard

    Examples:
        >>> # Basic usage with default model
        >>> from zvec.extension import JinaDenseEmbedding
        >>> import os
        >>> os.environ["JINA_API_KEY"] = "jina_..."
        >>>
        >>> emb_func = JinaDenseEmbedding()
        >>> vector = emb_func.embed("Hello, world!")
        >>> len(vector)
        768

        >>> # Retrieval use case: embed queries and documents differently
        >>> query_emb = JinaDenseEmbedding(task="retrieval.query")
        >>> doc_emb = JinaDenseEmbedding(task="retrieval.passage")
        >>>
        >>> query_vector = query_emb.embed("What is machine learning?")
        >>> doc_vector = doc_emb.embed("Machine learning is a subset of AI...")

        >>> # Using larger model with custom dimension (Matryoshka)
        >>> emb_func = JinaDenseEmbedding(
        ...     model="jina-embeddings-v5-text-small",
        ...     dimension=256,
        ...     api_key="jina_...",
        ...     task="text-matching",
        ... )
        >>> vector = emb_func.embed("Semantic similarity comparison")
        >>> len(vector)
        256

        >>> # Using with zvec collection
        >>> import zvec
        >>> emb_func = JinaDenseEmbedding(task="retrieval.passage")
        >>> schema = zvec.CollectionSchema(
        ...     name="docs",
        ...     vectors=zvec.VectorSchema(
        ...         "embedding", zvec.DataType.VECTOR_FP32, emb_func.dimension
        ...     ),
        ... )
        >>> collection = zvec.create_and_open(path="./my_docs", schema=schema)

    See Also:
        - ``DenseEmbeddingFunction``: Base class for dense embeddings
        - ``OpenAIDenseEmbedding``: Alternative using OpenAI API
        - ``QwenDenseEmbedding``: Alternative using Qwen/DashScope API
        - ``DefaultLocalDenseEmbedding``: Local model without API calls
    """

    def __init__(
        self,
        model: str = "jina-embeddings-v5-text-nano",
        dimension: Optional[int] = None,
        api_key: Optional[str] = None,
        task: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the Jina dense embedding function.

        Args:
            model (str): Jina model name. Defaults to "jina-embeddings-v5-text-nano".
            dimension (Optional[int]): Target embedding dimension or None for default.
            api_key (Optional[str]): API key or None to use environment variable.
            task (Optional[str]): Task type for embedding optimization or None.
            **kwargs: Additional parameters for API calls.

        Raises:
            ValueError: If API key is not provided and not in environment,
                or if task is not a valid task type.
        """
        # Initialize base class for API connection
        JinaFunctionBase.__init__(self, model=model, api_key=api_key, task=task)

        # Store dimension configuration
        self._custom_dimension = dimension

        # Determine actual dimension
        if dimension is None:
            self._dimension = self._MODEL_DIMENSIONS.get(model, 768)
        else:
            self._dimension = dimension

        # Store extra attributes
        self._extra_params = kwargs

    @property
    def dimension(self) -> int:
        """int: The expected dimensionality of the embedding vector."""
        return self._dimension

    @property
    def extra_params(self) -> dict:
        """dict: Extra parameters for model-specific customization."""
        return self._extra_params

    def __call__(self, input: TEXT) -> DenseVectorType:
        """Make the embedding function callable."""
        return self.embed(input)

    @lru_cache(maxsize=10)
    def embed(self, input: TEXT) -> DenseVectorType:
        """Generate dense embedding vector for the input text.

        This method calls the Jina Embeddings API to convert input text
        into a dense vector representation. Results are cached to improve
        performance for repeated inputs.

        Args:
            input (TEXT): Input text string to embed. Must be non-empty after
                stripping whitespace. Maximum length depends on model:
                8192 tokens for v5-nano, 32768 tokens for v5-small.

        Returns:
            DenseVectorType: A list of floats representing the embedding vector.
                Length equals ``self.dimension``. Example:
                ``[0.123, -0.456, 0.789, ...]``

        Raises:
            TypeError: If ``input`` is not a string.
            ValueError: If input is empty/whitespace-only, or if the API returns
                an error or malformed response.
            RuntimeError: If network connectivity issues or Jina service
                errors occur.

        Examples:
            >>> emb = JinaDenseEmbedding(task="retrieval.query")
            >>> vector = emb.embed("What is deep learning?")
            >>> len(vector)
            768
            >>> isinstance(vector[0], float)
            True

            >>> # Error: empty input
            >>> emb.embed("   ")
            ValueError: Input text cannot be empty or whitespace only

            >>> # Error: non-string input
            >>> emb.embed(123)
            TypeError: Expected 'input' to be str, got int

        Note:
            - This method is cached (maxsize=10). Identical inputs return cached results.
            - The cache is based on exact string match (case-sensitive).
            - Task type affects embedding optimization but not caching behavior.
        """
        if not isinstance(input, TEXT):
            raise TypeError(f"Expected 'input' to be str, got {type(input).__name__}")

        input = input.strip()
        if not input:
            raise ValueError("Input text cannot be empty or whitespace only")

        # Call API
        embedding_vector = self._call_text_embedding_api(
            input=input,
            dimension=self._custom_dimension,
        )

        # Verify dimension
        if len(embedding_vector) != self.dimension:
            raise ValueError(
                f"Dimension mismatch: expected {self.dimension}, "
                f"got {len(embedding_vector)}"
            )

        return embedding_vector
