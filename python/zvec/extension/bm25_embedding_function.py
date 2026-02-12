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
from typing import Literal, Optional

from ..common.constants import TEXT, SparseVectorType
from ..tool import require_module
from .embedding_function import SparseEmbeddingFunction


class BM25EmbeddingFunction(SparseEmbeddingFunction[TEXT]):
    """BM25-based sparse embedding function using DashText SDK.

    This class provides text-to-sparse-vector embedding capabilities using
    the DashText library with BM25 algorithm. BM25 (Best Matching 25) is a
    probabilistic retrieval function used for lexical search and document
    ranking based on term frequency and inverse document frequency.

    BM25 generates sparse vectors where each dimension corresponds to a term in
    the vocabulary, and the value represents the BM25 score for that term. It's
    particularly effective for:

    - Lexical search and keyword matching
    - Document ranking and information retrieval
    - Combining with dense embeddings for hybrid search
    - Traditional IR tasks where exact term matching is important

    This implementation uses DashText's SparseVectorEncoder, which provides
    efficient BM25 computation for Chinese and English text using either a
    built-in encoder or custom corpus training.

    Args:
        corpus (Optional[list[str]], optional): List of documents to train the
            BM25 encoder. If provided, creates a custom encoder trained on this
            corpus for better domain-specific accuracy. If ``None``, uses the
            built-in encoder. Defaults to ``None``.
        encoding_type (Literal["query", "document"], optional): Encoding mode
            for text processing. Use ``"query"`` for search queries (default) and
            ``"document"`` for document indexing. This distinction optimizes the
            BM25 scoring for asymmetric retrieval tasks. Defaults to ``"query"``.
        language (Literal["zh", "en"], optional): Language for built-in encoder.
            Only used when corpus is None. ``"zh"`` for Chinese (trained on Chinese
            Wikipedia), ``"en"`` for English. Defaults to ``"zh"``.
        b (float, optional): Document length normalization parameter for BM25.
            Range [0, 1]. 0 means no normalization, 1 means full normalization.
            Only used with custom corpus. Defaults to ``0.75``.
        k1 (float, optional): Term frequency saturation parameter for BM25.
            Higher values give more weight to term frequency. Only used with
            custom corpus. Defaults to ``1.2``.
        **kwargs: Additional parameters for DashText encoder customization.

    Attributes:
        corpus_size (int): Number of documents in the training corpus (0 if using built-in encoder).
        encoding_type (str): The encoding type being used ("query" or "document").
        language (str): The language of the built-in encoder ("zh" or "en").

    Raises:
        ValueError: If corpus is provided but empty or contains non-string elements.
        TypeError: If input to ``embed()`` is not a string.
        RuntimeError: If DashText encoder initialization or training fails.

    Note:
        - Requires Python 3.10, 3.11, or 3.12
        - Requires the ``dashtext`` package: ``pip install dashtext``
        - Two encoder options available:

          1. **Built-in encoder** (no corpus needed): Pre-trained models for
             Chinese (zh) and English (en), good generalization, works out-of-the-box
          2. **Custom encoder** (corpus required): Better accuracy for domain-specific
             terminology, requires training on your full corpus with BM25 parameters

        - Encoding types:

          * ``encoding_type="query"``: Optimized for search queries (shorter text)
          * ``encoding_type="document"``: Optimized for document indexing (longer text)

        - BM25 parameters (b, k1) only apply to custom encoder training
        - Output is sorted by indices (vocabulary term IDs) for consistency
        - Results are cached (LRU cache, maxsize=10) to reduce computation
        - No API key or network connectivity required (local computation)

    Examples:
        >>> # Option 1: Using built-in encoder for Chinese (no corpus needed)
        >>> from zvec.extension import BM25EmbeddingFunction
        >>>
        >>> # For query encoding (Chinese)
        >>> bm25_query_zh = BM25EmbeddingFunction(language="zh", encoding_type="query")
        >>> query_vec = bm25_query_zh.embed("什么是机器学习")
        >>> isinstance(query_vec, dict)
        True
        >>> # query_vec: {1169440797: 0.29, 2045788977: 0.70, ...}

        >>> # For document encoding (Chinese)
        >>> bm25_doc_zh = BM25EmbeddingFunction(language="zh", encoding_type="document")
        >>> doc_vec = bm25_doc_zh.embed("机器学习是人工智能的一个重要分支...")
        >>> isinstance(doc_vec, dict)
        True

        >>> # Using built-in encoder for English
        >>> bm25_query_en = BM25EmbeddingFunction(language="en", encoding_type="query")
        >>> query_vec_en = bm25_query_en.embed("what is vector search service")
        >>> isinstance(query_vec_en, dict)
        True

        >>> # Option 2: Using custom corpus for domain-specific accuracy
        >>> corpus = [
        ...     "机器学习是人工智能的一个重要分支",
        ...     "深度学习使用多层神经网络进行特征提取",
        ...     "自然语言处理技术用于理解和生成人类语言"
        ... ]
        >>> bm25_custom = BM25EmbeddingFunction(
        ...     corpus=corpus,
        ...     encoding_type="query",
        ...     b=0.75,
        ...     k1=1.2
        ... )
        >>> custom_vec = bm25_custom.embed("机器学习算法")
        >>> isinstance(custom_vec, dict)
        True

        >>> # Hybrid search: combining with dense embeddings
        >>> from zvec.extension import DefaultLocalDenseEmbedding
        >>> dense_emb = DefaultLocalDenseEmbedding()
        >>> bm25_emb = BM25EmbeddingFunction(language="zh", encoding_type="query")
        >>>
        >>> query = "machine learning algorithms"
        >>> dense_vec = dense_emb.embed(query)  # Semantic similarity
        >>> sparse_vec = bm25_emb.embed(query)  # Lexical matching
        >>> # Combine scores for hybrid retrieval

        >>> # Callable interface
        >>> sparse_vec = bm25_query_zh("information retrieval")
        >>> isinstance(sparse_vec, dict)
        True

        >>> # Error handling
        >>> try:
        ...     bm25_query_zh.embed("")  # Empty query
        ... except ValueError as e:
        ...     print(f"Error: {e}")
        Error: Input text cannot be empty or whitespace only

    See Also:
        - ``SparseEmbeddingFunction``: Base class for sparse embeddings
        - ``DefaultLocalSparseEmbedding``: SPLADE-based sparse embedding
        - ``QwenSparseEmbedding``: API-based sparse embedding using Qwen
        - ``DefaultLocalDenseEmbedding``: Dense embedding for semantic search

    References:
        - DashText Documentation: https://help.aliyun.com/zh/document_detail/2546039.html
        - DashText PyPI: https://pypi.org/project/dashtext/
        - BM25 Algorithm: Robertson & Zaragoza (2009)
    """

    def __init__(
        self,
        corpus: Optional[list[str]] = None,
        encoding_type: Literal["query", "document"] = "query",
        language: Literal["zh", "en"] = "zh",
        b: float = 0.75,
        k1: float = 1.2,
        **kwargs,
    ):
        """Initialize the BM25 embedding function.

        Args:
            corpus (Optional[list[str]]): Optional corpus for training custom encoder.
                If None, uses built-in encoder. Defaults to None.
            encoding_type (Literal["query", "document"]): Text encoding mode.
                Use "query" for search queries, "document" for indexing.
                Defaults to "query".
            language (Literal["zh", "en"]): Language for built-in encoder.
                "zh" for Chinese, "en" for English. Defaults to "zh".
            b (float): Document length normalization for BM25 [0, 1].
                Only used with custom corpus. Defaults to 0.75.
            k1 (float): Term frequency saturation for BM25.
                Only used with custom corpus. Defaults to 1.2.
            **kwargs: Additional DashText encoder parameters.

        Raises:
            ValueError: If corpus is provided but empty or invalid.
            ImportError: If dashtext package is not installed.
            RuntimeError: If encoder initialization or training fails.
        """
        # Validate corpus if provided
        if corpus is not None:
            if not corpus or not isinstance(corpus, list):
                raise ValueError("Corpus must be a non-empty list of strings")

            if not all(isinstance(doc, str) for doc in corpus):
                raise ValueError("All corpus documents must be strings")

        # Import dashtext
        self._dashtext = require_module("dashtext")

        self._corpus = corpus
        self._encoding_type = encoding_type
        self._language = language
        self._b = b
        self._k1 = k1
        self._extra_params = kwargs

        # Initialize the BM25 encoder
        self._build_encoder()

    def _build_encoder(self):
        """Build the BM25 sparse vector encoder.

        Creates either a built-in encoder (pre-trained) or a custom encoder
        trained on the provided corpus.

        Raises:
            RuntimeError: If encoder initialization or training fails.
            ImportError: If dashtext package is not installed.
        """
        try:
            if self._corpus is None:
                # Use built-in encoder (pre-trained on Wikipedia)
                # language: 'zh' for Chinese, 'en' for English
                self._encoder = self._dashtext.SparseVectorEncoder.default(
                    name=self._language
                )
            else:
                # Create custom encoder with BM25 parameters
                self._encoder = self._dashtext.SparseVectorEncoder(
                    b=self._b, k1=self._k1, **self._extra_params
                )

                # Train encoder with the corpus
                self._encoder.train(self._corpus)

        except ImportError as e:
            raise ImportError(
                "dashtext package is required for BM25EmbeddingFunction. "
                "Install it with: pip install dashtext"
            ) from e
        except Exception as e:
            if isinstance(e, (ValueError, RuntimeError)):
                raise
            raise RuntimeError(f"Failed to build BM25 encoder: {e!s}") from e

    @property
    def corpus_size(self) -> int:
        """int: Number of documents in the training corpus (0 if using built-in encoder)."""
        return len(self._corpus) if self._corpus is not None else 0

    @property
    def encoding_type(self) -> str:
        """str: The encoding type being used ("query" or "document")."""
        return self._encoding_type

    @property
    def language(self) -> str:
        """str: The language of the built-in encoder ("zh" or "en")."""
        return self._language

    @property
    def extra_params(self) -> dict:
        """dict: Extra parameters for DashText encoder customization."""
        return self._extra_params

    def __call__(self, input: TEXT) -> SparseVectorType:
        """Make the embedding function callable.

        Args:
            input (TEXT): Input text to embed.

        Returns:
            SparseVectorType: Sparse vector as dictionary.
        """
        return self.embed(input)

    @lru_cache(maxsize=10)
    def embed(self, input: TEXT) -> SparseVectorType:
        """Generate BM25 sparse embedding for the input text.

        This method computes BM25 scores for the input text using DashText's
        SparseVectorEncoder. The encoding behavior depends on the encoding_type:

        - ``encoding_type="query"``: Uses ``encode_queries()`` for search queries
        - ``encoding_type="document"``: Uses ``encode_documents()`` for documents

        The result is a sparse vector where keys are term indices in the
        vocabulary and values are BM25 scores.

        Args:
            input (TEXT): Input text string to embed. Must be non-empty after
                stripping whitespace.

        Returns:
            SparseVectorType: A dictionary mapping vocabulary term index to BM25 score.
                Only non-zero scores are included. The dictionary is sorted by indices
                (keys) in ascending order for consistent output.
                Example: ``{1169440797: 0.29, 2045788977: 0.70, ...}``

        Raises:
            TypeError: If ``input`` is not a string.
            ValueError: If input is empty or whitespace-only.
            RuntimeError: If BM25 encoding fails.

        Examples:
            >>> bm25 = BM25EmbeddingFunction(language="zh", encoding_type="query")
            >>> sparse_vec = bm25.embed("query text")
            >>> isinstance(sparse_vec, dict)
            True
            >>> all(isinstance(k, int) and isinstance(v, float) for k, v in sparse_vec.items())
            True

            >>> # Verify sorted output
            >>> keys = list(sparse_vec.keys())
            >>> keys == sorted(keys)
            True

            >>> # Error: empty input
            >>> bm25.embed("   ")
            ValueError: Input text cannot be empty or whitespace only

            >>> # Error: non-string input
            >>> bm25.embed(123)
            TypeError: Expected 'input' to be str, got int

        Note:
            - BM25 scores are relative to the vocabulary statistics
            - Output dictionary is always sorted by indices for consistency
            - Terms not in the vocabulary will have zero scores (not included)
            - This method is cached (maxsize=10) for performance
            - DashText automatically handles Chinese/English text segmentation
        """
        if not isinstance(input, str):
            raise TypeError(f"Expected 'input' to be str, got {type(input).__name__}")

        input = input.strip()
        if not input:
            raise ValueError("Input text cannot be empty or whitespace only")

        try:
            # Encode based on encoding_type
            if self._encoding_type == "query":
                sparse_vector = self._encoder.encode_queries(input)
            else:  # encoding_type == "document"
                sparse_vector = self._encoder.encode_documents(input)

            # DashText returns dict with int/long keys and float values
            # Convert to standard format: {int: float}
            sparse_dict: dict[int, float] = {}
            for key, value in sparse_vector.items():
                try:
                    idx = int(key)
                    val = float(value)
                    if val > 0:
                        sparse_dict[idx] = val
                except (ValueError, TypeError):
                    # Skip invalid entries
                    continue

            # Sort by indices (keys) to ensure consistent ordering
            return dict(sorted(sparse_dict.items()))

        except Exception as e:
            if isinstance(e, (TypeError, ValueError)):
                raise
            raise RuntimeError(f"Failed to generate BM25 embedding: {e!s}") from e
