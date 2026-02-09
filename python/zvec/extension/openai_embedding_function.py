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
from .openai_function import OpenAIFunctionBase


class OpenAIDenseEmbedding(OpenAIFunctionBase, DenseEmbeddingFunction[TEXT]):
    """Dense text embedding function using OpenAI API.

    This class provides text-to-vector embedding capabilities using OpenAI's
    embedding models. It inherits from ``DenseEmbeddingFunction`` and implements
    dense text embedding via the OpenAI API.

    The implementation supports various OpenAI embedding models with different
    dimensions and includes automatic result caching for improved performance.

    Args:
        model (str, optional): OpenAI embedding model identifier.
            Defaults to ``"text-embedding-3-small"``. Common options:
            - ``"text-embedding-3-small"``: 1536 dims, cost-efficient, good performance
            - ``"text-embedding-3-large"``: 3072 dims, highest quality
            - ``"text-embedding-ada-002"``: 1536 dims, legacy model
        dimension (Optional[int], optional): Desired output embedding dimension.
            If ``None``, uses model's default dimension. For text-embedding-3 models,
            you can specify custom dimensions (e.g., 256, 512, 1024, 1536).
            Defaults to ``None``.
        api_key (Optional[str], optional): OpenAI API authentication key.
            If ``None``, reads from ``OPENAI_API_KEY`` environment variable.
            Obtain your key from: https://platform.openai.com/api-keys
        base_url (Optional[str], optional): Custom API base URL for OpenAI-compatible
            services. Defaults to ``None`` (uses official OpenAI endpoint).

    Attributes:
        dimension (int): The embedding vector dimension.
        data_type (DataType): Always ``DataType.VECTOR_FP32`` for this implementation.
        model (str): The OpenAI model name being used.

    Raises:
        ValueError: If API key is not provided and not found in environment,
            or if API returns an error response.
        TypeError: If input to ``embed()`` is not a string.
        RuntimeError: If network error or OpenAI service error occurs.

    Note:
        - Requires Python 3.10, 3.11, or 3.12
        - Requires the ``openai`` package: ``pip install openai``
        - Embedding results are cached (LRU cache, maxsize=10) to reduce API calls
        - Network connectivity to OpenAI API endpoints is required
        - API usage incurs costs based on your OpenAI subscription plan
        - Rate limits apply based on your OpenAI account tier

    Examples:
        >>> # Basic usage with default model
        >>> from zvec.extension import OpenAIDenseEmbedding
        >>> import os
        >>> os.environ["OPENAI_API_KEY"] = "sk-..."
        >>>
        >>> emb_func = OpenAIDenseEmbedding()
        >>> vector = emb_func.embed("Hello, world!")
        >>> len(vector)
        1536

        >>> # Using specific model with custom dimension
        >>> emb_func = OpenAIDenseEmbedding(
        ...     model="text-embedding-3-large",
        ...     dimension=1024,
        ...     api_key="sk-..."
        ... )
        >>> vector = emb_func.embed("Machine learning is fascinating")
        >>> len(vector)
        1024

        >>> # Using with custom base URL (e.g., Azure OpenAI)
        >>> emb_func = OpenAIDenseEmbedding(
        ...     model="text-embedding-ada-002",
        ...     api_key="your-azure-key",
        ...     base_url="https://your-resource.openai.azure.com/"
        ... )
        >>> vector = emb_func("Natural language processing")
        >>> isinstance(vector, list)
        True

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
        - ``QwenDenseEmbedding``: Alternative using Qwen/DashScope API
        - ``DefaultDenseEmbedding``: Local model without API calls
        - ``SparseEmbeddingFunction``: Base class for sparse embeddings
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dimension: Optional[int] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the OpenAI dense embedding function.

        Args:
            model (str): OpenAI model name. Defaults to "text-embedding-3-small".
            dimension (Optional[int]): Target embedding dimension or None for default.
            api_key (Optional[str]): API key or None to use environment variable.
            base_url (Optional[str]): Custom API base URL or None for default.
            **kwargs: Additional parameters for API calls. Examples:
                - ``encoding_format`` (str): Format of embeddings, "float" or "base64".
                - ``user`` (str): User identifier for tracking.

        Raises:
            ValueError: If API key is not provided and not in environment.
        """
        # Initialize base class for API connection
        OpenAIFunctionBase.__init__(
            self, model=model, api_key=api_key, base_url=base_url
        )

        # Store dimension configuration
        self._custom_dimension = dimension

        # Determine actual dimension
        if dimension is None:
            # Use model default dimension
            self._dimension = self._MODEL_DIMENSIONS.get(model, 1536)
        else:
            self._dimension = dimension

        # Store dense-specific attributes
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

        This method calls the OpenAI Embeddings API to convert input text
        into a dense vector representation. Results are cached to improve
        performance for repeated inputs.

        Args:
            input (TEXT): Input text string to embed. Must be non-empty after
                stripping whitespace. Maximum length is 8191 tokens for most models.

        Returns:
            DenseVectorType: A list of floats representing the embedding vector.
                Length equals ``self.dimension``. Example:
                ``[0.123, -0.456, 0.789, ...]``

        Raises:
            TypeError: If ``input`` is not a string.
            ValueError: If input is empty/whitespace-only, or if the API returns
                an error or malformed response.
            RuntimeError: If network connectivity issues or OpenAI service
                errors occur.

        Examples:
            >>> emb = OpenAIDenseEmbedding()
            >>> vector = emb.embed("Natural language processing")
            >>> len(vector)
            1536
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
