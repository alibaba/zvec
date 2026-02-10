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

from typing import ClassVar, Literal, Optional

import numpy as np

from ..common.constants import TEXT, DenseVectorType, SparseVectorType
from .embedding_function import DenseEmbeddingFunction, SparseEmbeddingFunction
from .sentence_transformer_function import SentenceTransformerFunctionBase


class DefaultLocalDenseEmbedding(
    SentenceTransformerFunctionBase, DenseEmbeddingFunction[TEXT]
):
    """Default local dense embedding using all-MiniLM-L6-v2 model.

    This is the default implementation for dense text embedding that uses the
    ``all-MiniLM-L6-v2`` model from Hugging Face by default. This model provides
    a good balance between speed and quality for general-purpose text embedding.

    The class provides text-to-vector dense embedding capabilities using the
    sentence-transformers library. It supports models from Hugging Face Hub and
    ModelScope, runs locally without API calls, and supports CPU/GPU acceleration.

    The model produces 384-dimensional embeddings and is optimized for semantic
    similarity tasks. It runs locally without requiring API keys.

    Args:
        model_source (Literal["huggingface", "modelscope"], optional): Model source.
            - ``"huggingface"``: Use Hugging Face Hub (default, for international users)
            - ``"modelscope"``: Use ModelScope (recommended for users in China)
            Defaults to ``"huggingface"``.
        device (Optional[str], optional): Device to run the model on.
            Options: ``"cpu"``, ``"cuda"``, ``"mps"`` (for Apple Silicon), or ``None``
            for automatic detection. Defaults to ``None``.
        normalize_embeddings (bool, optional): Whether to normalize embeddings to
            unit length (L2 normalization). Useful for cosine similarity.
            Defaults to ``True``.
        batch_size (int, optional): Batch size for encoding. Defaults to ``32``.
        **kwargs: Additional parameters for future extension.

    Attributes:
        dimension (int): Always 384 for both models.
        model_name (str): "all-MiniLM-L6-v2" (HF) or "iic/nlp_gte_sentence-embedding_chinese-small" (MS).
        model_source (str): The model source being used.
        device (str): The device the model is running on.

    Raises:
        ValueError: If the model cannot be loaded or input is invalid.
        TypeError: If input to ``embed()`` is not a string.
        RuntimeError: If model inference fails.

    Note:
        - Requires Python 3.10, 3.11, or 3.12
        - Requires the ``sentence-transformers`` package:
          ``pip install sentence-transformers``
        - For ModelScope, also requires: ``pip install modelscope``
        - First run downloads the model (~50-80MB) from chosen source
        - Hugging Face cache: ``~/.cache/torch/sentence_transformers/``
        - ModelScope cache: ``~/.cache/modelscope/hub/``
        - No API keys or network required after initial download
        - Inference speed: ~1000 sentences/sec on CPU, ~10000 on GPU

        **For users in China:**

        If you encounter Hugging Face access issues, use ModelScope instead:

        .. code-block:: python

            # Recommended for users in China
            emb = DefaultLocalDenseEmbedding(model_source="modelscope")

        Alternatively, use Hugging Face mirror:

        .. code-block:: bash

            export HF_ENDPOINT=https://hf-mirror.com
            # Then use default Hugging Face mode

    Examples:
        >>> # Basic usage with Hugging Face (default)
        >>> from zvec.extension import DefaultLocalDenseEmbedding
        >>>
        >>> emb_func = DefaultLocalDenseEmbedding()
        >>> vector = emb_func.embed("Hello, world!")
        >>> len(vector)
        384
        >>> isinstance(vector, list)
        True

        >>> # Recommended for users in China (uses ModelScope)
        >>> emb_func = DefaultLocalDenseEmbedding(model_source="modelscope")
        >>> vector = emb_func.embed("你好，世界！")  # Works well with Chinese text
        >>> len(vector)
        384

        >>> # Alternative for China users: Use Hugging Face mirror
        >>> import os
        >>> os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        >>> emb_func = DefaultLocalDenseEmbedding()  # Uses HF mirror
        >>> vector = emb_func.embed("Hello, world!")

        >>> # Using GPU for faster inference
        >>> emb_func = DefaultLocalDenseEmbedding(device="cuda")
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
        - ``DenseEmbeddingFunction``: Base class for dense embeddings
        - ``DefaultLocalSparseEmbedding``: Sparse embedding with SPLADE
        - ``QwenDenseEmbedding``: Alternative using Qwen API
    """

    def __init__(
        self,
        model_source: Literal["huggingface", "modelscope"] = "huggingface",
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        **kwargs,
    ):
        """Initialize with all-MiniLM-L6-v2 model.

        Args:
            model_source (Literal["huggingface", "modelscope"]): Model source.
                Defaults to "huggingface".
            device (Optional[str]): Target device ("cpu", "cuda", "mps", or None).
                Defaults to None (automatic detection).
            normalize_embeddings (bool): Whether to L2-normalize output vectors.
                Defaults to True.
            batch_size (int): Batch size for encoding. Defaults to 32.
            **kwargs: Additional parameters for future extension.

        Raises:
            ImportError: If sentence-transformers or modelscope is not installed.
            ValueError: If model cannot be loaded.
        """
        # Use different models based on source
        if model_source == "modelscope":
            # Use Chinese-optimized model for ModelScope (better for Chinese text)
            model_name = "iic/nlp_gte_sentence-embedding_chinese-small"
        else:
            model_name = "all-MiniLM-L6-v2"

        # Initialize base class for model loading
        SentenceTransformerFunctionBase.__init__(
            self, model_name=model_name, model_source=model_source, device=device
        )

        self._normalize_embeddings = normalize_embeddings
        self._batch_size = batch_size

        # Load model and get dimension
        model = self._get_model()
        self._dimension = model.get_sentence_embedding_dimension()

        # Store extra parameters
        self._extra_params = kwargs

    @property
    def dimension(self) -> int:
        """int: The expected dimensionality of the embedding vector."""
        return self._dimension

    @property
    def extra_params(self) -> dict:
        """dict: Extra parameters for model-specific customization."""
        return self._extra_params

    def __call__(self, input: str) -> DenseVectorType:
        """Make the embedding function callable."""
        return self.embed(input)

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
            >>> emb = DefaultLocalDenseEmbedding()
            >>> vector = emb.embed("Natural language processing")
            >>> len(vector)
            384
            >>> isinstance(vector[0], float)
            True

            >>> # Normalized vectors have unit length
            >>> import numpy as np
            >>> emb = DefaultLocalDenseEmbedding(normalize_embeddings=True)
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


class DefaultLocalSparseEmbedding(
    SentenceTransformerFunctionBase, SparseEmbeddingFunction[TEXT]
):
    """Default local sparse embedding using SPLADE model.

    This class provides sparse vector embedding using the SPLADE (SParse Lexical
    AnD Expansion) model. SPLADE generates sparse, interpretable representations
    where each dimension corresponds to a vocabulary term with learned importance
    weights. It's ideal for lexical matching, BM25-style retrieval, and hybrid
    search scenarios.

    The default model is ``naver/splade-cocondenser-ensembledistil``, which is
    publicly available without authentication. It produces sparse vectors with
    thousands of dimensions but only hundreds of non-zero values, making them
    efficient for storage and retrieval while maintaining strong lexical matching.

    **Model Caching:**

    This class uses class-level caching to share the SPLADE model across all instances
    with the same configuration (model_source, device). This significantly reduces
    memory usage when creating multiple instances for different encoding types
    (query vs document).

    **Cache Management:**

    The class provides methods to manage the model cache:

    - ``clear_cache()``: Clear all cached models to free memory
    - ``get_cache_info()``: Get information about cached models
    - ``remove_from_cache(model_source, device)``: Remove a specific model from cache

    .. note::
        **Why not use splade-v3?**

        The newer ``naver/splade-v3`` model is gated (requires access approval).
        We use ``naver/splade-cocondenser-ensembledistil`` instead.

        **To use splade-v3 (if you have access):**

        1. Request access at https://huggingface.co/naver/splade-v3
        2. Get your Hugging Face token from https://huggingface.co/settings/tokens
        3. Set environment variable:

           .. code-block:: bash

               export HF_TOKEN="your_huggingface_token"

        4. Or login programmatically:

           .. code-block:: python

               from huggingface_hub import login
               login(token="your_huggingface_token")

        5. To use a custom SPLADE model, you can subclass this class and override
           the model_name in ``__init__``, or create your own implementation
           inheriting from ``SentenceTransformerFunctionBase`` and
           ``SparseEmbeddingFunction``.

    Args:
        model_source (Literal["huggingface", "modelscope"], optional): Model source.
            Defaults to ``"huggingface"``. ModelScope support may vary for SPLADE models.
        device (Optional[str], optional): Device to run the model on.
            Options: ``"cpu"``, ``"cuda"``, ``"mps"`` (for Apple Silicon), or ``None``
            for automatic detection. Defaults to ``None``.
        encoding_type (Literal["query", "document"], optional): Encoding type.
            - ``"query"``: Optimize for search queries (default)
            - ``"document"``: Optimize for indexed documents
        **kwargs: Additional parameters (currently unused, for future extension).

    Attributes:
        model_name (str): Model identifier.
        model_source (str): The model source being used.
        device (str): The device the model is running on.

    Raises:
        ValueError: If the model cannot be loaded or input is invalid.
        TypeError: If input to ``embed()`` is not a string.
        RuntimeError: If model inference fails.

    Note:
        - Requires Python 3.10, 3.11, or 3.12
        - Requires the ``sentence-transformers`` package:
          ``pip install sentence-transformers``
        - First run downloads the model (~100MB) from Hugging Face
        - Cache location: ``~/.cache/torch/sentence_transformers/``
        - No API keys or authentication required
        - Sparse vectors have ~30k dimensions but only ~100-200 non-zero values
        - Best combined with dense embeddings for hybrid retrieval

        **SPLADE vs Dense Embeddings:**

        - **Dense**: Continuous semantic vectors, good for semantic similarity
        - **Sparse**: Lexical keyword-based, interpretable, good for exact matching
        - **Hybrid**: Combine both for best retrieval performance

    Examples:
        >>> # Memory-efficient: both instances share the same model (~200MB)
        >>> from zvec.extension import DefaultLocalSparseEmbedding
        >>>
        >>> # Query embedding
        >>> query_emb = DefaultLocalSparseEmbedding(encoding_type="query")
        >>> query_vec = query_emb.embed("machine learning algorithms")
        >>> type(query_vec)
        <class 'dict'>
        >>> len(query_vec)  # Only non-zero dimensions
        156

        >>> # Document embedding (shares model with query_emb)
        >>> doc_emb = DefaultLocalSparseEmbedding(encoding_type="document")
        >>> doc_vec = doc_emb.embed("Machine learning is a subset of AI")
        >>> # Total memory: ~200MB (not 400MB) thanks to model caching

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

        >>> # Batch processing
        >>> queries = ["query 1", "query 2", "query 3"]
        >>> query_vecs = [query_emb.embed(q) for q in queries]
        >>>
        >>> documents = ["doc 1", "doc 2", "doc 3"]
        >>> doc_vecs = [doc_emb.embed(d) for d in documents]

        >>> # Inspecting sparse dimensions (output is sorted by indices)
        >>> query_vec = query_emb.embed("machine learning")
        >>> list(query_vec.items())[:5]  # First 5 dimensions (by index)
        [(10, 0.45), (23, 0.87), (56, 0.32), (89, 1.12), (120, 0.65)]
        >>>
        >>> # Sort by weight to find most important terms
        >>> sorted_by_weight = sorted(query_vec.items(), key=lambda x: x[1], reverse=True)
        >>> top_5 = sorted_by_weight[:5]  # Top 5 most important terms
        >>> top_5
        [(1023, 1.45), (245, 1.23), (8901, 0.98), (5678, 0.87), (12034, 0.76)]

        >>> # Using GPU for faster inference
        >>> sparse_emb = DefaultLocalSparseEmbedding(device="cuda")
        >>> vector = sparse_emb.embed("natural language processing")

        >>> # Hybrid retrieval example (combining dense + sparse)
        >>> from zvec.extension import DefaultDenseEmbedding
        >>> dense_emb = DefaultDenseEmbedding()
        >>> sparse_emb = DefaultLocalSparseEmbedding()
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

        >>> # Cache management
        >>> # Check cache status
        >>> info = DefaultLocalSparseEmbedding.get_cache_info()
        >>> print(f"Cached models: {info['cached_models']}")
        Cached models: 1
        >>>
        >>> # Clear cache to free memory
        >>> DefaultLocalSparseEmbedding.clear_cache()
        >>> info = DefaultLocalSparseEmbedding.get_cache_info()
        >>> print(f"Cached models: {info['cached_models']}")
        Cached models: 0
        >>>
        >>> # Remove specific model from cache
        >>> query_emb = DefaultLocalSparseEmbedding()  # Creates CPU model
        >>> cuda_emb = DefaultLocalSparseEmbedding(device="cuda")  # Creates CUDA model
        >>> info = DefaultLocalSparseEmbedding.get_cache_info()
        >>> print(f"Cached models: {info['cached_models']}")
        Cached models: 2
        >>>
        >>> # Remove only CPU model
        >>> removed = DefaultLocalSparseEmbedding.remove_from_cache(device=None)
        >>> print(f"Removed: {removed}")
        True
        >>> info = DefaultLocalSparseEmbedding.get_cache_info()
        >>> print(f"Cached models: {info['cached_models']}")
        Cached models: 1

    See Also:
        - ``SparseEmbeddingFunction``: Base class for sparse embeddings
        - ``DefaultDenseEmbedding``: Dense embedding with all-MiniLM-L6-v2
        - ``QwenDenseEmbedding``: Alternative using Qwen API

    References:
        - SPLADE Paper: https://arxiv.org/abs/2109.10086
        - Model: https://huggingface.co/naver/splade-cocondenser-ensembledistil
    """

    # Class-level model cache: {(model_name, model_source, device): model}
    # Shared across all DefaultLocalSparseEmbedding instances to save memory
    _model_cache: ClassVar[dict] = {}

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached SPLADE models from memory.

        This is useful for:
        - Freeing memory when models are no longer needed
        - Forcing a fresh model reload
        - Testing and debugging
                Examples:
            >>> # Clear cache to free memory
            >>> DefaultLocalSparseEmbedding.clear_cache()

            >>> # Or in tests to ensure fresh model loading
            >>> def test_something():
            ...     DefaultLocalSparseEmbedding.clear_cache()
            ...     emb = DefaultLocalSparseEmbedding()
            ...     # Test with fresh model
        """
        cls._model_cache.clear()

    @classmethod
    def get_cache_info(cls) -> dict:
        """Get information about currently cached models.

        Returns:
            dict: Dictionary with cache statistics:
                - cached_models (int): Number of cached model instances
                - cache_keys (list): List of cache keys (model_name, model_source, device)

        Examples:
            >>> info = DefaultLocalSparseEmbedding.get_cache_info()
            >>> print(f"Cached models: {info['cached_models']}")
            Cached models: 2
            >>> print(f"Cache keys: {info['cache_keys']}")
            Cache keys: [('naver/splade-cocondenser-ensembledistil', 'huggingface', None),
                        ('naver/splade-cocondenser-ensembledistil', 'huggingface', 'cuda')]
        """
        return {
            "cached_models": len(cls._model_cache),
            "cache_keys": list(cls._model_cache.keys()),
        }

    @classmethod
    def remove_from_cache(
        cls, model_source: str = "huggingface", device: Optional[str] = None
    ) -> bool:
        """Remove a specific model from cache.

        Args:
            model_source (str): Model source ("huggingface" or "modelscope").
                Defaults to "huggingface".
            device (Optional[str]): Device identifier. Defaults to None.

        Returns:
            bool: True if model was found and removed, False otherwise.

        Examples:
            >>> # Remove CPU model from cache
            >>> removed = DefaultLocalSparseEmbedding.remove_from_cache()
            >>> print(f"Removed: {removed}")
            True

            >>> # Remove CUDA model from cache
            >>> removed = DefaultLocalSparseEmbedding.remove_from_cache(device="cuda")
            >>> print(f"Removed: {removed}")
            True
        """
        model_name = "naver/splade-cocondenser-ensembledistil"
        cache_key = (model_name, model_source, device)

        if cache_key in cls._model_cache:
            del cls._model_cache[cache_key]
            return True
        return False

    def __init__(
        self,
        model_source: Literal["huggingface", "modelscope"] = "huggingface",
        device: Optional[str] = None,
        encoding_type: Literal["query", "document"] = "query",
        **kwargs,
    ):
        """Initialize with SPLADE model.

        Args:
            model_source (Literal["huggingface", "modelscope"]): Model source.
                Defaults to "huggingface".
            device (Optional[str]): Target device ("cpu", "cuda", "mps", or None).
                Defaults to None (automatic detection).
            encoding_type (Literal["query", "document"]): Encoding type for embeddings.
                - "query": Optimize for search queries (default)
                - "document": Optimize for indexed documents
                This distinction is important for asymmetric retrieval tasks.
            **kwargs: Additional parameters (reserved for future use).

        Raises:
            ImportError: If sentence-transformers is not installed.
            ValueError: If model cannot be loaded.

        Note:
            Multiple instances with the same (model_source, device) configuration
            will share the same underlying model to save memory. Different
            instances can use different encoding_type settings while sharing
            the model.

            **Model Selection:**

            Uses ``naver/splade-cocondenser-ensembledistil`` instead of the newer
            ``naver/splade-v3`` because splade-v3 is a gated model requiring
            Hugging Face authentication. The cocondenser-ensembledistil variant:

            - Does not require authentication or API tokens
            - Is immediately available for all users
            - Provides comparable retrieval performance (~2% difference)
            - Avoids "Access to model is restricted" errors

            If you need splade-v3 and have obtained access, you can subclass
            this class and override the model_name parameter.

        Examples:
            >>> # Both instances share the same model (saves memory)
            >>> query_emb = DefaultLocalSparseEmbedding(encoding_type="query")
            >>> doc_emb = DefaultLocalSparseEmbedding(encoding_type="document")
            >>> # Only one model is loaded in memory
        """
        # Use publicly available SPLADE model (no gated access required)
        # Note: naver/splade-v3 requires authentication, so we use the
        # cocondenser-ensembledistil variant which is publicly accessible
        model_name = "naver/splade-cocondenser-ensembledistil"

        # Initialize base class for model loading
        SentenceTransformerFunctionBase.__init__(
            self, model_name=model_name, model_source=model_source, device=device
        )

        self._encoding_type = encoding_type
        self._extra_params = kwargs

        # Create cache key for this model configuration
        self._cache_key = (model_name, model_source, device)

        # Load model to ensure it's available (will use cache if exists)
        self._get_model()

    @property
    def extra_params(self) -> dict:
        """dict: Extra parameters for model-specific customization."""
        return self._extra_params

    def __call__(self, input: str) -> SparseVectorType:
        """Make the embedding function callable."""
        return self.embed(input)

    def embed(self, input: str) -> SparseVectorType:
        """Generate sparse embedding vector for the input text.

        This method uses the SPLADE model to convert input text into a sparse
        vector representation. The result is a dictionary where keys are dimension
        indices and values are importance weights (only non-zero values included).

        The embedding is optimized based on the ``encoding_type`` specified during
        initialization: "query" for search queries or "document" for indexed content.

        Args:
            input (str): Input text string to embed. Must be non-empty after
                stripping whitespace.

        Returns:
            SparseVectorType: A dictionary mapping dimension index to weight.
                Only non-zero dimensions are included. The dictionary is sorted
                by indices (keys) in ascending order for consistent output.
                Example: ``{10: 0.5, 245: 0.8, 1023: 1.2, 5678: 0.5}``

        Raises:
            TypeError: If ``input`` is not a string.
            ValueError: If input is empty or whitespace-only.
            RuntimeError: If model inference fails.

        Examples:
            >>> # Query embedding
            >>> query_emb = DefaultLocalSparseEmbedding(encoding_type="query")
            >>> query_vec = query_emb.embed("machine learning")
            >>> isinstance(query_vec, dict)
            True

        Note:
            - First call may be slower due to model loading
            - Subsequent calls are much faster as the model stays in memory
            - GPU acceleration provides significant speedup
            - Sparse vectors are memory-efficient (only store non-zero values)
        """
        if not isinstance(input, str):
            raise TypeError(f"Expected 'input' to be str, got {type(input).__name__}")

        input = input.strip()
        if not input:
            raise ValueError("Input text cannot be empty or whitespace only")

        try:
            model = self._get_model()

            # Use appropriate encoding method based on type
            if self._encoding_type == "document" and hasattr(model, "encode_document"):
                # Use document encoding
                sparse_matrix = model.encode_document([input])
            elif hasattr(model, "encode_query"):
                # Use query encoding (default)
                sparse_matrix = model.encode_query([input])
            else:
                # Fallback: manual implementation for older sentence-transformers
                return self._manual_sparse_encode(input)

            # Convert sparse matrix to dictionary
            # SPLADE returns shape [1, vocab_size] for single input

            # Check if it's a sparse matrix (duck typing - has toarray method)
            if hasattr(sparse_matrix, "toarray"):
                # Sparse matrix (CSR/CSC/etc.) - convert to dense array
                sparse_array = sparse_matrix[0].toarray().flatten()
                sparse_dict = {
                    int(idx): float(val)
                    for idx, val in enumerate(sparse_array)
                    if val > 0
                }
            else:
                # Dense array format (numpy array or similar)
                if isinstance(sparse_matrix, np.ndarray):
                    sparse_array = sparse_matrix[0]
                else:
                    sparse_array = sparse_matrix

                sparse_dict = {
                    int(idx): float(val)
                    for idx, val in enumerate(sparse_array)
                    if val > 0
                }

            # Sort by indices (keys) to ensure consistent ordering
            return dict(sorted(sparse_dict.items()))

        except Exception as e:
            if isinstance(e, (TypeError, ValueError)):
                raise
            raise RuntimeError(f"Failed to generate sparse embedding: {e!s}") from e

    def _manual_sparse_encode(self, input: str) -> SparseVectorType:
        """Fallback manual SPLADE encoding for older sentence-transformers.

        Args:
            input (str): Input text to encode.

        Returns:
            SparseVectorType: Sparse vector as dictionary.
        """
        import torch

        model = self._get_model()

        # Tokenize input
        features = model.tokenize([input])

        # Move to correct device
        features = {k: v.to(model.device) for k, v in features.items()}

        # Forward pass with no gradient
        with torch.no_grad():
            embeddings = model.forward(features)

            # Get logits from model output
            # SPLADE models typically output 'token_embeddings'
            if isinstance(embeddings, dict) and "token_embeddings" in embeddings:
                logits = embeddings["token_embeddings"][0]  # First batch item
            elif hasattr(embeddings, "token_embeddings"):
                logits = embeddings.token_embeddings[0]
            # Fallback: try to get first value
            elif isinstance(embeddings, dict):
                logits = next(iter(embeddings.values()))[0]
            else:
                logits = embeddings[0]

            # Apply SPLADE activation: log(1 + relu(x))
            relu_log = torch.log(1 + torch.relu(logits))

            # Max pooling over token dimension (reduce to vocab size)
            if relu_log.dim() > 1:
                sparse_vec, _ = torch.max(relu_log, dim=0)
            else:
                sparse_vec = relu_log

            # Convert to sparse dictionary (only non-zero values)
            sparse_vec_np = sparse_vec.cpu().numpy()
            sparse_dict = {
                int(idx): float(val) for idx, val in enumerate(sparse_vec_np) if val > 0
            }

            # Sort by indices (keys) to ensure consistent ordering
            return dict(sorted(sparse_dict.items()))

    def _get_model(self):
        """Load or retrieve the SPLADE model from class-level cache.

        Returns:
            SentenceTransformer: The loaded SPLADE model instance.

        Raises:
            ImportError: If required packages are not installed.
            ValueError: If model cannot be loaded.

        Note:
            Models are cached at class level and shared across all instances
            with the same (model_name, model_source, device) configuration.
            This allows memory-efficient usage when creating multiple instances
            with different encoding_type settings.
        """
        # Check class-level cache first
        if self._cache_key in self._model_cache:
            return self._model_cache[self._cache_key]

        # Use parent class method to load model
        model = super()._get_model()

        # Cache the model at class level
        self._model_cache[self._cache_key] = model

        return model
