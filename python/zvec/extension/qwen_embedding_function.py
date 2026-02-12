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

from ..common.constants import TEXT, DenseVectorType, SparseVectorType
from .embedding_function import DenseEmbeddingFunction, SparseEmbeddingFunction
from .qwen_function import QwenFunctionBase


class QwenDenseEmbedding(QwenFunctionBase, DenseEmbeddingFunction[TEXT]):
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

        >>> # Using with custom parameters (text_type)
        >>> # For search queries - optimize for query-document matching
        >>> emb_func = QwenDenseEmbedding(
        ...     dimension=1024,
        ...     text_type="query"
        ... )
        >>> query_vector = emb_func.embed("What is machine learning?")
        >>>
        >>> # For document embeddings - optimize for being matched by queries
        >>> doc_emb_func = QwenDenseEmbedding(
        ...     dimension=1024,
        ...     text_type="document"
        ... )
        >>> doc_vector = doc_emb_func.embed(
        ...     "Machine learning is a subset of artificial intelligence..."
        ... )

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
        """Initialize the Qwen dense embedding function.

        Args:
            dimension (int): Target embedding dimension.
            model (str): DashScope model name. Defaults to "text-embedding-v4".
            api_key (Optional[str]): API key or None to use environment variable.
            **kwargs: Additional DashScope API parameters. Supported options:
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
        # Initialize base class for API connection
        QwenFunctionBase.__init__(self, model=model, api_key=api_key)

        # Store dense-specific attributes
        self._dimension = dimension
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

        This method calls the DashScope TextEmbedding API to convert input text
        into a dense vector representation. Results are cached to improve
        performance for repeated inputs.

        Args:
            input (TEXT): Input text string to embed. Must be non-empty after
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
        if not isinstance(input, TEXT):
            raise TypeError(f"Expected 'input' to be str, got {type(input).__name__}")

        input = input.strip()
        if not input:
            raise ValueError("Input text cannot be empty or whitespace only")

        # Call API with dense output type
        output = self._call_text_embedding_api(
            input=input,
            dimension=self.dimension,
            output_type="dense",
            text_type=self.extra_params.get("text_type"),
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


class QwenSparseEmbedding(QwenFunctionBase, SparseEmbeddingFunction[TEXT]):
    """Sparse text embedding function using Qwen (DashScope) API.

    This class provides text-to-sparse-vector embedding capabilities using
    Alibaba Cloud's DashScope service and Qwen embedding models. It generates
    sparse keyword-weighted vectors suitable for lexical matching and BM25-style
    retrieval scenarios.

    Sparse embeddings are particularly useful for:
    - Keyword-based search and exact matching
    - Hybrid retrieval (combining with dense embeddings)
    - Interpretable search results (weights show term importance)

    Args:
        dimension (int): Desired output embedding dimension. Common values:
            - 512: Balanced performance and accuracy
            - 1024: Higher accuracy, larger storage
            - 1536: Maximum accuracy for supported models
        model (str, optional): DashScope embedding model identifier.
            Defaults to ``"text-embedding-v4"``. Other options include:
            - ``"text-embedding-v3"``
            - ``"text-embedding-v2"``
        api_key (Optional[str], optional): DashScope API authentication key.
            If ``None``, reads from ``DASHSCOPE_API_KEY`` environment variable.
            Obtain your key from: https://dashscope.console.aliyun.com/
        **kwargs: Additional DashScope API parameters. Supported options:
            - ``encoding_type`` (Literal["query", "document"]): Encoding type.
              * ``"query"``: Optimize for search queries (default).
              * ``"document"``: Optimize for indexed documents.
              This distinction is important for asymmetric retrieval tasks.

    Attributes:
        model (str): The DashScope model name being used.
        encoding_type (str): The encoding type ("query" or "document").

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
        - Sparse vectors have only non-zero dimensions stored as dict
        - Output is sorted by indices (keys) in ascending order

        **Parameter Guidelines:**

        - Use ``encoding_type="query"`` for search queries and
          ``encoding_type="document"`` for indexed content to optimize
          asymmetric retrieval tasks.
        - For detailed API specifications, refer to:
          https://help.aliyun.com/zh/model-studio/text-embedding-synchronous-api

    Examples:
        >>> # Basic usage for query embedding
        >>> from zvec.extension import QwenSparseEmbedding
        >>> import os
        >>> os.environ["DASHSCOPE_API_KEY"] = "your-api-key"
        >>>
        >>> query_emb = QwenSparseEmbedding(dimension=1024, encoding_type="query")
        >>> query_vec = query_emb.embed("machine learning")
        >>> type(query_vec)
        <class 'dict'>
        >>> len(query_vec)  # Only non-zero dimensions
        156

        >>> # Document embedding
        >>> doc_emb = QwenSparseEmbedding(dimension=1024, encoding_type="document")
        >>> doc_vec = doc_emb.embed("Machine learning is a subset of AI")
        >>> isinstance(doc_vec, dict)
        True

        >>> # Asymmetric retrieval example
        >>> query_vec = query_emb.embed("what causes aging fast")
        >>> doc_vec = doc_emb.embed(
        ...     "UV-A light causes tanning, skin aging, and cataracts..."
        ... )
        >>>
        >>> # Calculate similarity (dot product for sparse vectors)
        >>> similarity = sum(
        ...     query_vec.get(k, 0) * doc_vec.get(k, 0)
        ...     for k in set(query_vec) | set(doc_vec)
        ... )

        >>> # Output is sorted by indices
        >>> list(query_vec.items())[:5]  # First 5 dimensions (by index)
        [(10, 0.45), (23, 0.87), (56, 0.32), (89, 1.12), (120, 0.65)]

        >>> # Hybrid retrieval (combining dense + sparse)
        >>> from zvec.extension import QwenDenseEmbedding
        >>> dense_emb = QwenDenseEmbedding(dimension=1024)
        >>> sparse_emb = QwenSparseEmbedding(dimension=1024)
        >>>
        >>> query = "deep learning neural networks"
        >>> dense_vec = dense_emb.embed(query)   # [0.1, -0.3, 0.5, ...]
        >>> sparse_vec = sparse_emb.embed(query)  # {12: 0.8, 45: 1.2, ...}

        >>> # Error handling
        >>> try:
        ...     sparse_emb.embed("")  # Empty string
        ... except ValueError as e:
        ...     print(f"Error: {e}")
        Error: Input text cannot be empty or whitespace only

    See Also:
        - ``SparseEmbeddingFunction``: Base class for sparse embeddings
        - ``QwenDenseEmbedding``: Dense embedding using Qwen API
        - ``DefaultSparseEmbedding``: Sparse embedding with SPLADE model
    """

    def __init__(
        self,
        dimension: int,
        model: str = "text-embedding-v4",
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the Qwen sparse embedding function.

        Args:
            dimension (int): Target embedding dimension.
            model (str): DashScope model name. Defaults to "text-embedding-v4".
            api_key (Optional[str]): API key or None to use environment variable.
            **kwargs: Additional DashScope API parameters. Supported options:
                - ``encoding_type`` (Literal["query", "document"]): Encoding type.
                  * ``"query"``: Optimize for search queries (default).
                  * ``"document"``: Optimize for indexed documents.
                  This distinction is important for asymmetric retrieval tasks.

        Raises:
            ValueError: If API key is not provided and not in environment.
        """
        # Initialize base class for API connection
        QwenFunctionBase.__init__(self, model=model, api_key=api_key)

        self._dimension = dimension
        self._extra_params = kwargs

    @property
    def extra_params(self) -> dict:
        """dict: Extra parameters for model-specific customization."""
        return self._extra_params

    def __call__(self, input: TEXT) -> SparseVectorType:
        """Make the embedding function callable."""
        return self.embed(input)

    @lru_cache(maxsize=10)
    def embed(self, input: TEXT) -> SparseVectorType:
        """Generate sparse embedding vector for the input text.

        This method calls the DashScope TextEmbedding API with sparse output type
        to convert input text into a sparse vector representation. The result is
        a dictionary where keys are dimension indices and values are importance
        weights (only non-zero values included).

        The embedding is optimized based on the ``encoding_type`` specified during
        initialization: "query" for search queries or "document" for indexed content.

        Args:
            input (TEXT): Input text string to embed. Must be non-empty after
                stripping whitespace. Maximum length depends on the model used
                (typically 2048-8192 tokens).

        Returns:
            SparseVectorType: A dictionary mapping dimension index to weight.
                Only non-zero dimensions are included. The dictionary is sorted
                by indices (keys) in ascending order for consistent output.
                Example: ``{10: 0.5, 245: 0.8, 1023: 1.2, 5678: 0.5}``

        Raises:
            TypeError: If ``input`` is not a string.
            ValueError: If input is empty/whitespace-only, or if the API returns
                an error or malformed response.
            RuntimeError: If network connectivity issues or DashScope service
                errors occur.

        Examples:
            >>> emb = QwenSparseEmbedding(dimension=1024, encoding_type="query")
            >>> sparse_vec = emb.embed("machine learning")
            >>> isinstance(sparse_vec, dict)
            True
            >>>
            >>> # Verify sorted output
            >>> keys = list(sparse_vec.keys())
            >>> keys == sorted(keys)
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
            - Output dictionary is always sorted by indices for consistency.
        """
        if not isinstance(input, TEXT):
            raise TypeError(f"Expected 'input' to be str, got {type(input).__name__}")

        input = input.strip()
        if not input:
            raise ValueError("Input text cannot be empty or whitespace only")

        # Call API with sparse output type
        output = self._call_text_embedding_api(
            input=input,
            dimension=self._dimension,
            output_type="sparse",
            text_type=self.extra_params.get("encoding_type", "query"),
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

        sparse_embedding = first_emb.get("sparse_embedding")
        if not isinstance(sparse_embedding, list):
            raise ValueError(
                "Invalid API response: 'sparse_embedding' field is missing or not a list"
            )

        # Parse sparse embedding: convert array of {index, value, token} to dict
        sparse_dict = {}
        for item in sparse_embedding:
            if not isinstance(item, dict):
                raise ValueError(
                    "Invalid API response: sparse_embedding item is not a dictionary"
                )

            index = item.get("index")
            value = item.get("value")

            if index is None or value is None:
                raise ValueError(
                    "Invalid API response: sparse_embedding item missing 'index' or 'value'"
                )

            # Convert to int and float, filter positive values
            idx = int(index)
            val = float(value)
            if val > 0:
                sparse_dict[idx] = val

        # Sort by indices (keys) to ensure consistent ordering
        return dict(sorted(sparse_dict.items()))
