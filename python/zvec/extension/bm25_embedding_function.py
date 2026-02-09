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

from typing import Literal, Optional

from ..common.constants import TEXT, SparseVectorType
from ..tool import require_module
from .embedding_function import SparseEmbeddingFunction


class BM25EmbeddingFunction(SparseEmbeddingFunction[TEXT]):
    """BM25-based sparse embedding function using bm25s library.

    This class provides text-to-sparse-vector embedding capabilities using the
    BM25 (Best Matching 25) algorithm, implemented via the bm25s library. BM25
    is a probabilistic retrieval function used for lexical search and document
    ranking based on term frequency and inverse document frequency.

    BM25 generates sparse vectors where each dimension corresponds to a term in
    the vocabulary, and the value represents the BM25 score for that term. It's
    particularly effective for:

    - Lexical search and keyword matching
    - Document ranking and information retrieval
    - Combining with dense embeddings for hybrid search
    - Traditional IR tasks where exact term matching is important

    This implementation uses the bm25s library (https://github.com/xhluca/bm25s),
    which provides fast and memory-efficient BM25 computation using Scipy sparse
    matrices.

    Args:
        corpus (list[str]): List of documents to build the BM25 index from.
            This corpus is used to calculate IDF statistics and build the vocabulary.
        method (Literal["robertson", "atire", "bm25l", "bm25+", "lucene"], optional):
            BM25 variant to use. Defaults to ``"lucene"``.
            - ``"robertson"``: Original BM25 formulation
            - ``"atire"``: ATIRE variant
            - ``"bm25l"``: BM25L variant (handles document length normalization)
            - ``"bm25+"``: BM25+ variant (adds term independence)
            - ``"lucene"``: Lucene implementation (default, widely used)
        k1 (float, optional): BM25 parameter controlling term frequency saturation.
            Higher values give more weight to term frequency. Defaults to ``1.5``.
        b (float, optional): BM25 parameter controlling length normalization.
            Range [0, 1]. 0 means no normalization, 1 means full normalization.
            Defaults to ``0.75``.
        stemmer (Optional[str], optional): Stemmer to use for text preprocessing.
            Options: ``"english"``, ``"porter"``, or ``None`` for no stemming.
            Defaults to ``None``.
        stopwords (Optional[str], optional): Language for stopword removal.
            Options: ``"english"``, ``"en"``, or ``None`` for no stopword removal.
            Defaults to ``None``.
        **kwargs: Additional parameters passed to bm25s tokenizer.

    Attributes:
        method (str): The BM25 variant being used.
        corpus_size (int): Number of documents in the corpus.
        vocab_size (int): Size of the vocabulary.

    Raises:
        ValueError: If corpus is empty or invalid.
        ImportError: If bm25s package is not installed.
        TypeError: If input to ``embed()`` is not a string.

    Note:
        - Requires Python 3.10, 3.11, or 3.12
        - Requires the ``bm25s`` package: ``pip install bm25s[full]``
        - The corpus must be provided at initialization and cannot be changed
        - BM25 scores are relative to the corpus statistics
        - Output is sorted by indices (vocabulary term IDs) for consistency
        - For best results, use the same preprocessing for corpus and queries

    Examples:
        >>> # Basic usage with a small corpus
        >>> from zvec.extension import BM25EmbeddingFunction
        >>>
        >>> corpus = [
        ...     "a cat is a feline and likes to purr",
        ...     "a dog is the human's best friend and loves to play",
        ...     "a bird is a beautiful animal that can fly",
        ... ]
        >>> bm25_emb = BM25EmbeddingFunction(corpus=corpus)
        >>>
        >>> # Generate sparse embedding for a query
        >>> query = "does the cat purr?"
        >>> sparse_vec = bm25_emb.embed(query)
        >>> isinstance(sparse_vec, dict)
        True
        >>> # sparse_vec: {12: 0.89, 45: 1.23, 67: 0.56, ...}

        >>> # Using different BM25 variants
        >>> bm25_robertson = BM25EmbeddingFunction(corpus=corpus, method="robertson")
        >>> bm25_plus = BM25EmbeddingFunction(corpus=corpus, method="bm25+")

        >>> # With stemming and stopword removal
        >>> bm25_advanced = BM25EmbeddingFunction(
        ...     corpus=corpus,
        ...     method="lucene",
        ...     stemmer="english",
        ...     stopwords="english",
        ...     k1=1.2,
        ...     b=0.75
        ... )
        >>> sparse_vec = bm25_advanced.embed("cat playing with a ball")

        >>> # Hybrid search: combining with dense embeddings
        >>> from zvec.extension import DefaultDenseEmbedding
        >>> dense_emb = DefaultDenseEmbedding()
        >>> bm25_emb = BM25EmbeddingFunction(corpus=large_corpus)
        >>>
        >>> query = "machine learning algorithms"
        >>> dense_vec = dense_emb.embed(query)  # Semantic similarity
        >>> sparse_vec = bm25_emb.embed(query)  # Lexical matching
        >>> # Combine scores for hybrid retrieval

        >>> # Document ranking
        >>> documents = [
        ...     "Machine learning is a subset of AI",
        ...     "Deep learning uses neural networks",
        ...     "Natural language processing handles text",
        ... ]
        >>> bm25 = BM25EmbeddingFunction(corpus=documents)
        >>>
        >>> # Rank documents by BM25 scores
        >>> query = "machine learning neural networks"
        >>> query_vec = bm25.embed(query)
        >>> # Calculate similarity with each document

        >>> # Callable interface
        >>> sparse_vec = bm25_emb("information retrieval")
        >>> isinstance(sparse_vec, dict)
        True

        >>> # Error handling
        >>> try:
        ...     bm25_emb.embed("")  # Empty query
        ... except ValueError as e:
        ...     print(f"Error: {e}")
        Error: Input text cannot be empty or whitespace only

    See Also:
        - ``SparseEmbeddingFunction``: Base class for sparse embeddings
        - ``DefaultSparseEmbedding``: SPLADE-based sparse embedding
        - ``QwenSparseEmbedding``: API-based sparse embedding using Qwen
        - ``DefaultDenseEmbedding``: Dense embedding for semantic search

    References:
        - BM25S Library: https://github.com/xhluca/bm25s
        - BM25 Algorithm: Robertson & Zaragoza (2009)
        - Technical Report: https://arxiv.org/abs/2407.03618
    """

    def __init__(
        self,
        corpus: list[str],
        method: Literal["robertson", "atire", "bm25l", "bm25+", "lucene"] = "lucene",
        k1: float = 1.5,
        b: float = 0.75,
        stemmer: Optional[str] = None,
        stopwords: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the BM25 embedding function with a corpus.

        Args:
            corpus (list[str]): List of documents for building the BM25 index.
            method (Literal["robertson", "atire", "bm25l", "bm25+", "lucene"]):
                BM25 variant. Defaults to "lucene".
            k1 (float): Term frequency saturation parameter. Defaults to 1.5.
            b (float): Length normalization parameter [0, 1]. Defaults to 0.75.
            stemmer (Optional[str]): Stemmer language. Defaults to None.
            stopwords (Optional[str]): Stopwords language. Defaults to None.
            **kwargs: Additional tokenizer parameters.

        Raises:
            ValueError: If corpus is empty or invalid.
            ImportError: If bm25s package is not installed.
        """
        if not corpus or not isinstance(corpus, list):
            raise ValueError("Corpus must be a non-empty list of strings")

        if not all(isinstance(doc, str) for doc in corpus):
            raise ValueError("All corpus documents must be strings")

        # Import bm25s
        self._bm25s = require_module("bm25s")

        self._corpus = corpus
        self._method = method
        self._k1 = k1
        self._b = b
        self._stemmer = stemmer
        self._stopwords = stopwords
        self._extra_params = kwargs

        # Build the BM25 index
        self._build_index()

    def _build_index(self):
        """Build the BM25 index from the corpus."""
        # Tokenize the corpus
        tokenizer_kwargs = {}
        if self._stemmer:
            tokenizer_kwargs["stemmer"] = self._stemmer
        if self._stopwords:
            tokenizer_kwargs["stopwords"] = self._stopwords
        tokenizer_kwargs.update(self._extra_params)

        # Tokenize corpus
        corpus_tokens = self._bm25s.tokenize(self._corpus, **tokenizer_kwargs)

        # Create BM25 retriever
        self._retriever = self._bm25s.BM25(
            corpus=self._corpus,
            method=self._method,
            k1=self._k1,
            b=self._b,
        )

        # Index the corpus
        self._retriever.index(corpus_tokens)

        # Get vocabulary size
        if hasattr(self._retriever, "vocab"):
            # Some bm25s versions may expose vocab on the retriever
            self._vocab_size = len(self._retriever.vocab)
        elif hasattr(corpus_tokens, "vocab"):
            # Fallback: use vocab from the tokenized corpus object
            self._vocab_size = len(corpus_tokens.vocab)
        else:
            # As a last resort, derive size from max token id
            ids = getattr(corpus_tokens, "ids", corpus_tokens)
            max_id = max((max(seq) for seq in ids if seq), default=-1)
            self._vocab_size = max_id + 1 if max_id >= 0 else 0

    @property
    def method(self) -> str:
        """str: The BM25 variant being used."""
        return self._method

    @property
    def corpus_size(self) -> int:
        """int: Number of documents in the corpus."""
        return len(self._corpus)

    @property
    def vocab_size(self) -> int:
        """int: Size of the vocabulary."""
        return self._vocab_size

    @property
    def extra_params(self) -> dict:
        """dict: Extra parameters for tokenizer customization."""
        return self._extra_params

    def __call__(self, input: TEXT) -> SparseVectorType:
        """Make the embedding function callable.

        Args:
            input (TEXT): Input text to embed.

        Returns:
            SparseVectorType: Sparse vector as dictionary.
        """
        return self.embed(input)

    def embed(self, input: TEXT) -> SparseVectorType:
        """Generate BM25 sparse embedding for the input text.

        This method computes BM25 scores for the input query against the corpus
        vocabulary. The result is a sparse vector where keys are term indices in
        the vocabulary and values are BM25 scores.

        Args:
            input (TEXT): Input text string to embed. Must be non-empty after
                stripping whitespace.

        Returns:
            SparseVectorType: A dictionary mapping vocabulary term index to BM25 score.
                Only non-zero scores are included. The dictionary is sorted by indices
                (keys) in ascending order for consistent output.
                Example: ``{5: 0.89, 12: 1.45, 23: 0.67, 45: 1.12}``

        Raises:
            TypeError: If ``input`` is not a string.
            ValueError: If input is empty or whitespace-only.
            RuntimeError: If BM25 computation fails.

        Examples:
            >>> bm25 = BM25EmbeddingFunction(corpus=["doc1", "doc2"])
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
            - BM25 scores are relative to the corpus statistics
            - Longer documents tend to have lower scores due to length normalization
            - Output dictionary is always sorted by indices for consistency
            - Terms not in the vocabulary will have zero scores (not included)
        """
        if not isinstance(input, str):
            raise TypeError(f"Expected 'input' to be str, got {type(input).__name__}")

        input = input.strip()
        if not input:
            raise ValueError("Input text cannot be empty or whitespace only")

        try:
            # Tokenize the query
            tokenizer_kwargs = {}
            if self._stemmer:
                tokenizer_kwargs["stemmer"] = self._stemmer
            if self._stopwords:
                tokenizer_kwargs["stopwords"] = self._stopwords
            tokenizer_kwargs.update(self._extra_params)

            query_tokens = self._bm25s.tokenize(
                [input],  # bm25s expects a list
                **tokenizer_kwargs,
            )

            # For BM25, we compute term-level scores for each query token ID
            # bm25s may return either a raw list-of-lists or a Tokenized object
            ids_obj = getattr(query_tokens, "ids", query_tokens)
            if isinstance(ids_obj, list) and ids_obj and isinstance(ids_obj[0], list):
                # Shape: [[id0, id1, ...]] -> take first query
                query_token_ids = ids_obj[0]
            else:
                # Shape: [id0, id1, ...] or empty
                query_token_ids = ids_obj or []

            sparse_dict: dict[int, float] = {}

            for token_id in query_token_ids:
                # token_id is already the vocabulary index in bm25s
                if token_id < 0:
                    continue

                # Compute term frequency in query
                term_freq = query_token_ids.count(token_id)

                # Get IDF value from retriever.idf
                idf_source = getattr(self._retriever, "idf", None)
                if idf_source is not None:
                    if hasattr(idf_source, "get"):
                        # dict or dict-like (including Mock with get)
                        idf = float(idf_source.get(token_id, 0.0))
                    # sequence/array-like
                    elif 0 <= token_id < len(idf_source):
                        idf = float(idf_source[token_id])
                    else:
                        idf = 0.0
                else:
                    # Fallback IDF when not available
                    idf = 1.0

                # BM25 score calculation (simplified)
                score = 0.0
                if term_freq > 0:
                    score = idf * (term_freq * (self._k1 + 1)) / (term_freq + self._k1)

                if score > 0:
                    # If the same token appears multiple times we accumulate
                    if token_id in sparse_dict:
                        sparse_dict[int(token_id)] += float(score)
                    else:
                        sparse_dict[int(token_id)] = float(score)

            # Sort by indices (keys) to ensure consistent ordering
            return dict(sorted(sparse_dict.items()))

        except Exception as e:
            if isinstance(e, (TypeError, ValueError)):
                raise
            raise RuntimeError(f"Failed to generate BM25 embedding: {e!s}") from e
