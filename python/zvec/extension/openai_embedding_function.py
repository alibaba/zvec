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
from typing import ClassVar, Optional

from ..common.constants import TEXT, DenseVectorType
from ..tool import require_module
from ..typing import DataType
from .embedding_function import DenseEmbeddingFunction


class OpenAIDenseEmbedding(DenseEmbeddingFunction[TEXT]):
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

    # Model default dimensions
    _MODEL_DIMENSIONS: ClassVar[dict[str, int]] = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dimension: Optional[int] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the OpenAI embedding function.

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
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._base_url = base_url
        self._custom_dimension = dimension

        if not self._api_key:
            raise ValueError(
                "OpenAI API key is required. Please provide 'api_key' parameter "
                "or set the 'OPENAI_API_KEY' environment variable."
            )

        # Determine dimension
        if dimension is None:
            # Use model default dimension
            dimension = self._MODEL_DIMENSIONS.get(model, 1536)

        super().__init__(dimension, DataType.VECTOR_FP32, **kwargs)

    @property
    def model(self) -> str:
        """str: The OpenAI embedding model name currently in use.

        Returns:
            str: Model identifier (e.g., "text-embedding-3-small").
        """
        return self._model

    def _get_client(self):
        """Get OpenAI client instance.

        Returns:
            OpenAI: Configured OpenAI client.

        Raises:
            ImportError: If openai package is not installed.
        """
        openai = require_module("openai")

        if self._base_url:
            return openai.OpenAI(api_key=self._api_key, base_url=self._base_url)
        return openai.OpenAI(api_key=self._api_key)

    @lru_cache(maxsize=10)
    def embed(self, input: str) -> DenseVectorType:
        """Generate dense embedding vector for the input text.

        This method calls the OpenAI Embeddings API to convert input text
        into a dense vector representation. Results are cached to improve
        performance for repeated inputs.

        Args:
            input (str): Input text string to embed. Must be non-empty after
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
        if not isinstance(input, str):
            raise TypeError(f"Expected 'input' to be str, got {type(input).__name__}")

        input = input.strip()
        if not input:
            raise ValueError("Input text cannot be empty or whitespace only")

        try:
            client = self._get_client()

            # Prepare embedding parameters
            params = {"model": self.model, "input": input}

            # Add dimension parameter for models that support it
            if self._custom_dimension is not None:
                params["dimensions"] = self._custom_dimension

            # Call OpenAI API
            response = client.embeddings.create(**params)

        except Exception as e:
            # Check if it's an OpenAI API error
            openai = require_module("openai")
            if isinstance(e, (openai.APIError, openai.APIConnectionError)):
                raise RuntimeError(f"Failed to call OpenAI API: {e!s}") from e
            raise RuntimeError(f"Unexpected error during API call: {e!s}") from e

        # Extract embedding from response
        try:
            if not response.data:
                raise ValueError("Invalid API response: no embedding data returned")

            embedding_vector = response.data[0].embedding

            if not isinstance(embedding_vector, list):
                raise ValueError(
                    "Invalid API response: embedding is not a list of numbers"
                )

            if len(embedding_vector) != self.dimension:
                raise ValueError(
                    f"Dimension mismatch: expected {self.dimension}, "
                    f"got {len(embedding_vector)}"
                )

            return embedding_vector

        except (AttributeError, IndexError, TypeError) as e:
            raise ValueError(f"Failed to parse API response: {e!s}") from e
