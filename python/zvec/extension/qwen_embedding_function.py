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
from functools import lru_cache
from http import HTTPStatus
from typing import Optional

from ..common.constants import TEXT, DenseVectorType
from ..tool import require_module
from ..typing import DataType
from .embedding_function import DenseEmbeddingFunction


class QwenDenseEmbedding(DenseEmbeddingFunction[TEXT]):
    """Dense text embedding function using Qwen (DashScope) API.

    This class provides text-to-vector embedding capabilities using Alibaba Cloud's
    DashScope service and Qwen embedding models. It inherits from
    ``DenseEmbeddingFunction`` and implements dense text embedding.

    The implementation supports various Qwen embedding models with configurable
    dimensions and includes automatic result caching for improved performance.

    Args:
        dimension (int): Desired output embedding dimension. Common values:
            - 512: Balanced performance and accuracy
            - 1024: Higher accuracy, larger storage
            - 1536: Maximum accuracy for supported models
        model (str, optional): DashScope embedding model identifier.
            Defaults to ``"text-embedding-v4"``. Other options include:
            - ``"text-embedding-v3"``
            - ``"text-embedding-v2"``
            - ``"text-embedding-v1"``
        api_key (Optional[str], optional): DashScope API authentication key.
            If ``None``, reads from ``DASHSCOPE_API_KEY`` environment variable.
            Obtain your key from: https://dashscope.console.aliyun.com/
        **kwargs: Additional DashScope API parameters. Supported options:
            - ``output_type`` (str): Embedding output format. Options:
              ``"dense"`` (default) or ``"sparse"``. Dense embeddings are
              continuous vectors suitable for semantic similarity; sparse
              embeddings are keyword-weighted vectors for lexical matching.
            - ``text_type`` (str): Specifies the text role in retrieval tasks.
              Options: ``"query"`` (search query) or ``"document"`` (indexed content).
              This parameter optimizes embeddings for asymmetric search scenarios.

            Reference: https://help.aliyun.com/zh/model-studio/text-embedding-synchronous-api

    Attributes:
        dimension (int): The embedding vector dimension.
        data_type (DataType): Always ``DataType.VECTOR_FP32`` for this implementation.
        model (str): The DashScope model name being used.

    Raises:
        ValueError: If API key is not provided and not found in environment,
            or if API returns an error response.
        TypeError: If input to ``embed()`` is not a string.
        RuntimeError: If network error or DashScope service error occurs.

    Note:
        - Requires Python 3.10, 3.11, or 3.12
        - Requires the ``dashscope`` package: ``pip install dashscope``
        - Embedding results are cached (LRU cache, maxsize=10) to reduce API calls
        - Network connectivity to DashScope API endpoints is required
        - API usage may incur costs based on your DashScope subscription plan

        **Parameter Guidelines:**

        - Use ``text_type="query"`` for search queries and ``text_type="document"``
          for indexed content to optimize asymmetric retrieval tasks.
        - ``output_type="dense"`` is recommended for semantic similarity and neural
          search; ``output_type="sparse"`` for keyword-based and BM25-style retrieval.
        - For detailed API specifications and parameter usage, refer to:
          https://help.aliyun.com/zh/model-studio/text-embedding-synchronous-api

    Examples:
        >>> # Basic usage with default model
        >>> from zvec.extension import QwenDenseEmbedding
        >>> import os
        >>> os.environ["DASHSCOPE_API_KEY"] = "your-api-key"
        >>>
        >>> emb_func = QwenDenseEmbedding(dimension=1024)
        >>> vector = emb_func.embed("Hello, world!")
        >>> len(vector)
        1024

        >>> # Using specific model with explicit API key
        >>> emb_func = QwenDenseEmbedding(
        ...     dimension=512,
        ...     model="text-embedding-v3",
        ...     api_key="sk-xxxxx"
        ... )
        >>> vector = emb_func("Machine learning is fascinating")
        >>> isinstance(vector, list)
        True

        >>> # Using with custom parameters (output_type, text_type)
        >>> # For search queries - optimize for query-document matching
        >>> emb_func = QwenDenseEmbedding(
        ...     dimension=1024,
        ...     output_type="dense",
        ...     text_type="query"
        ... )
        >>> query_vector = emb_func.embed("What is machine learning?")
        >>>
        >>> # For document embeddings - optimize for being matched by queries
        >>> doc_emb_func = QwenDenseEmbedding(
        ...     dimension=1024,
        ...     output_type="dense",
        ...     text_type="document"
        ... )
        >>> doc_vector = doc_emb_func.embed(
        ...     "Machine learning is a subset of artificial intelligence..."
        ... )
        >>>
        >>> # Using sparse embeddings for lexical matching (BM25-style)
        >>> sparse_emb = QwenDenseEmbedding(
        ...     dimension=1024,
        ...     output_type="sparse"
        ... )
        >>> sparse_vector = sparse_emb.embed("keyword-based search query")

        >>> # Batch processing with caching benefit
        >>> texts = ["First text", "Second text", "First text"]
        >>> vectors = [emb_func.embed(text) for text in texts]
        >>> # Third call uses cached result for "First text"

        >>> # Error handling
        >>> try:
        ...     emb_func.embed("")  # Empty string
        ... except ValueError as e:
        ...     print(f"Error: {e}")
        Error: Input text cannot be empty or whitespace only

    See Also:
        - ``DenseEmbeddingFunction``: Base class for dense embeddings
        - ``SparseEmbeddingFunction``: Base class for sparse embeddings
    """

    def __init__(
        self,
        dimension: int,
        model: str = "text-embedding-v4",
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the Qwen embedding function.

        Args:
            dimension (int): Target embedding dimension.
            model (str): DashScope model name. Defaults to "text-embedding-v4".
            api_key (Optional[str]): API key or None to use environment variable.
            **kwargs: Additional DashScope API parameters. Supported options:
                - ``output_type`` (str): Embedding output format.
                  * ``"dense"`` (default): Continuous vector representation for
                    semantic similarity and neural retrieval. Returns float array.
                  * ``"sparse"``: Sparse keyword-weighted vector for lexical
                    matching and BM25-style retrieval. Returns sparse dict.
                - ``text_type`` (str): Text role in asymmetric retrieval.
                  * ``"query"``: Optimize for search queries (short, question-like).
                  * ``"document"``: Optimize for indexed documents (longer content).
                  Using appropriate text_type improves retrieval accuracy by
                  optimizing the embedding space for query-document matching.

                For detailed API documentation, see:
                https://help.aliyun.com/zh/model-studio/text-embedding-synchronous-api

        Raises:
            ValueError: If API key is not provided and not in environment.
        """
        super().__init__(dimension, DataType.VECTOR_FP32, **kwargs)
        self._model = model
        self._api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self._api_key:
            raise ValueError(
                "DashScope API key is required. Please provide 'api_key' parameter "
                "or set the 'DASHSCOPE_API_KEY' environment variable."
            )

    @property
    def model(self) -> str:
        """str: The DashScope embedding model name currently in use.

        Returns:
            str: Model identifier (e.g., "text-embedding-v4").
        """
        return self._model

    def _get_connection(self):
        """Establish connection to DashScope API.

        Returns:
            module: The dashscope module with API key configured.

        Raises:
            ImportError: If dashscope package is not installed.
        """
        dashscope = require_module("dashscope")
        dashscope.api_key = self._api_key
        return dashscope

    @lru_cache(maxsize=10)
    def embed(self, input: str) -> DenseVectorType:
        """Generate dense embedding vector for the input text.

        This method calls the DashScope TextEmbedding API to convert input text
        into a dense vector representation. Results are cached to improve
        performance for repeated inputs.

        Args:
            input (str): Input text string to embed. Must be non-empty after
                stripping whitespace. Maximum length depends on the model used
                (typically 2048-8192 tokens).

        Returns:
            DenseVectorType: A list of floats representing the embedding vector.
                Length equals ``self.dimension``. Example:
                ``[0.123, -0.456, 0.789, ...]``

        Raises:
            TypeError: If ``input`` is not a string.
            ValueError: If input is empty/whitespace-only, or if the API returns
                an error or malformed response.
            RuntimeError: If network connectivity issues or DashScope service
                errors occur.

        Examples:
            >>> emb = QwenDenseEmbedding(dimension=1024)
            >>> vector = emb.embed("Natural language processing")
            >>> len(vector)
            1024
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
            - Consider pre-processing text (lowercasing, normalization) for better caching.
        """
        if not isinstance(input, str):
            raise TypeError(f"Expected 'input' to be str, got {type(input).__name__}")

        input = input.strip()
        if not input:
            raise ValueError("Input text cannot be empty or whitespace only")

        try:
            # Prepare API call parameters
            call_params = {
                "model": self.model,
                "input": input,
                "dimension": self.dimension,
                "output_type": self.extra_params.get("output_type", "dense"),
            }

            # Add optional text_type parameter if provided
            if "text_type" in self.extra_params:
                call_params["text_type"] = self.extra_params["text_type"]

            resp = self._get_connection().TextEmbedding.call(**call_params)
        except Exception as e:
            raise RuntimeError(f"Failed to call DashScope API: {e!s}") from e

        if resp.status_code != HTTPStatus.OK:
            error_msg = getattr(resp, "message", "Unknown error")
            error_code = getattr(resp, "code", "N/A")
            raise ValueError(
                f"DashScope API error: [Code={error_code}, "
                f"Status={resp.status_code}] {error_msg}"
            )

        output = getattr(resp, "output", None)
        if not isinstance(output, dict):
            raise ValueError(
                "Invalid API response: missing or malformed 'output' field"
            )

        embeddings = output.get("embeddings")
        if not isinstance(embeddings, list):
            raise ValueError(
                "Invalid API response: 'embeddings' field is missing or not a list"
            )

        if len(embeddings) != 1:
            raise ValueError(
                f"Expected exactly 1 embedding in response, got {len(embeddings)}"
            )

        first_emb = embeddings[0]
        if not isinstance(first_emb, dict):
            raise ValueError("Invalid API response: embedding item is not a dictionary")

        embedding_vector = first_emb.get("embedding")
        if not isinstance(embedding_vector, list):
            raise ValueError(
                "Invalid API response: 'embedding' field is missing or not a list"
            )

        if len(embedding_vector) != self.dimension:
            raise ValueError(
                f"Dimension mismatch: expected {self.dimension}, "
                f"got {len(embedding_vector)}"
            )

        return list(embedding_vector)
