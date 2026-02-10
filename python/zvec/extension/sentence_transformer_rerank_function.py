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

from ..model.doc import Doc
from ..tool import require_module
from .rerank_function import RerankFunction
from .sentence_transformer_function import SentenceTransformerFunctionBase


class DefaultLocalReRanker(SentenceTransformerFunctionBase, RerankFunction):
    """Re-ranker using Sentence Transformer cross-encoder models for semantic re-ranking.

    This re-ranker leverages pre-trained cross-encoder models to perform deep semantic
    re-ranking of search results. It runs locally without API calls, supports GPU
    acceleration, and works with models from Hugging Face or ModelScope.

    Cross-encoder models evaluate query-document pairs jointly, providing more
    accurate relevance scores than bi-encoder (embedding-based) similarity.

    Args:
        query (str): Query text for semantic re-ranking. **Required**.
        topn (int, optional): Maximum number of documents to return after re-ranking.
            Defaults to 10.
        rerank_field (Optional[str], optional): Document field name to use as
            re-ranking input text. **Required** (e.g., "content", "title", "body").
        model_name (str, optional): Cross-encoder model identifier or local path.
            Defaults to ``"cross-encoder/ms-marco-MiniLM-L6-v2"`` (MS MARCO MiniLM).
            Common options:
            - ``"cross-encoder/ms-marco-MiniLM-L6-v2"``: Lightweight, fast (~80MB, recommended)
            - ``"cross-encoder/ms-marco-MiniLM-L12-v2"``: Better accuracy (~120MB)
            - ``"BAAI/bge-reranker-base"``: BGE Reranker Base (~280MB)
            - ``"BAAI/bge-reranker-large"``: BGE Reranker Large (highest quality, ~560MB)
        model_source (Literal["huggingface", "modelscope"], optional): Model source.
            Defaults to ``"huggingface"``.
            - ``"huggingface"``: Load from Hugging Face Hub
            - ``"modelscope"``: Load from ModelScope (recommended for users in China)
        device (Optional[str], optional): Device to run the model on.
            Options: ``"cpu"``, ``"cuda"``, ``"mps"`` (for Apple Silicon), or ``None``
            for automatic detection. Defaults to ``None``.
        batch_size (int, optional): Batch size for processing query-document pairs.
            Larger values speed up processing but use more memory. Defaults to ``32``.

    Attributes:
        query (str): The query text used for re-ranking.
        topn (int): Maximum number of documents to return.
        rerank_field (Optional[str]): Field name used for re-ranking input.
        model_name (str): The cross-encoder model being used.
        model_source (str): The model source ("huggingface" or "modelscope").
        device (str): The device the model is running on.

    Raises:
        ValueError: If ``query`` is empty/None, ``rerank_field`` is None,
            or model cannot be loaded.
        TypeError: If input types are invalid.
        RuntimeError: If model inference fails.

    Note:
        - Requires Python 3.10, 3.11, or 3.12
        - Requires ``sentence-transformers`` package: ``pip install sentence-transformers``
        - For ModelScope support, also requires: ``pip install modelscope``
        - First run downloads the model (~80-560MB depending on model) from chosen source
        - No API keys or network required after initial download
        - Cross-encoders are slower than bi-encoders but more accurate
        - GPU acceleration provides significant speedup (5-10x)

        **MS MARCO MiniLM-L6-v2 Model (Default):**

        The default model ``cross-encoder/ms-marco-MiniLM-L6-v2`` is a lightweight and
        efficient cross-encoder trained on MS MARCO dataset. It provides:

        - Fast inference speed (suitable for real-time applications)
        - Small model size (~80MB, quick to download)
        - Good balance between speed and accuracy
        - Trained on 500K+ query-document pairs
        - Public availability without authentication

        **For users in China:**

        If you encounter Hugging Face access issues, use ModelScope instead:

        .. code-block:: python

            # Recommended for users in China
            reranker = SentenceTransformerReRanker(
                query="机器学习算法",
                rerank_field="content",
                model_source="modelscope"
            )

        Alternatively, use Hugging Face mirror:

        .. code-block:: bash

            export HF_ENDPOINT=https://hf-mirror.com

    Examples:
        >>> # Basic usage with default MS MARCO MiniLM model
        >>> from zvec.extension import SentenceTransformerReRanker
        >>>
        >>> reranker = SentenceTransformerReRanker(
        ...     query="machine learning algorithms",
        ...     topn=5,
        ...     rerank_field="content"
        ... )
        >>>
        >>> # Use in collection.query()
        >>> results = collection.query(
        ...     data={"vector_field": query_vector},
        ...     reranker=reranker,
        ...     topk=20
        ... )

        >>> # Using ModelScope for users in China
        >>> reranker = SentenceTransformerReRanker(
        ...     query="深度学习",
        ...     topn=10,
        ...     rerank_field="content",
        ...     model_source="modelscope"
        ... )

        >>> # Using larger model for better quality
        >>> reranker = SentenceTransformerReRanker(
        ...     query="neural networks",
        ...     topn=5,
        ...     rerank_field="content",
        ...     model_name="BAAI/bge-reranker-large",
        ...     device="cuda",
        ...     batch_size=64
        ... )

        >>> # Direct rerank call (for testing)
        >>> query_results = {
        ...     "vector1": [
        ...         Doc(id="1", score=0.9, fields={"content": "Machine learning is..."}),
        ...         Doc(id="2", score=0.8, fields={"content": "Deep learning is..."}),
        ...     ]
        ... }
        >>> reranked = reranker.rerank(query_results)
        >>> for doc in reranked:
        ...     print(f"ID: {doc.id}, Score: {doc.score:.4f}")
        ID: 2, Score: 0.9234
        ID: 1, Score: 0.8567

    See Also:
        - ``RerankFunction``: Abstract base class for re-rankers
        - ``QwenReRanker``: Re-ranker using Qwen API
        - ``RrfReRanker``: Multi-vector re-ranker using RRF
        - ``WeightedReRanker``: Multi-vector re-ranker using weighted scores

    References:
        - MS MARCO Cross-Encoder: https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2
        - BGE Reranker: https://huggingface.co/BAAI/bge-reranker-base
        - Cross-Encoder vs Bi-Encoder: https://www.sbert.net/examples/applications/cross-encoder/README.html
    """

    def __init__(
        self,
        query: Optional[str] = None,
        topn: int = 10,
        rerank_field: Optional[str] = None,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        model_source: Literal["huggingface", "modelscope"] = "huggingface",
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        """Initialize SentenceTransformerReRanker with query and configuration.

        Args:
            query (Optional[str]): Query text for semantic matching. Required.
            topn (int): Number of top results to return.
            rerank_field (Optional[str]): Document field for re-ranking input.
            model_name (str): Cross-encoder model identifier.
            model_source (Literal["huggingface", "modelscope"]): Model source.
            device (Optional[str]): Target device ("cpu", "cuda", "mps", or None).
            batch_size (int): Batch size for processing query-document pairs.

        Raises:
            ValueError: If query is empty or model cannot be loaded.
        """
        # Initialize base class for model loading
        SentenceTransformerFunctionBase.__init__(
            self, model_name=model_name, model_source=model_source, device=device
        )

        # Initialize rerank function
        RerankFunction.__init__(self, topn=topn, rerank_field=rerank_field)

        # Validate query
        if not query:
            raise ValueError("Query is required for DefaultLocalReRanker")
        self._query = query
        self._batch_size = batch_size

        # Load and validate cross-encoder model
        model = self._get_model()
        if not hasattr(model, "predict"):
            raise ValueError(
                f"Model '{model_name}' does not appear to be a cross-encoder model. "
                "Cross-encoder models should have a 'predict' method."
            )
        self._model = model

    def _get_model(self):
        """Load or retrieve the CrossEncoder model.

        This overrides the base class method to load CrossEncoder instead of
        SentenceTransformer, as reranking requires cross-encoder models.

        Returns:
            CrossEncoder: The loaded cross-encoder model instance.

        Raises:
            ImportError: If required packages are not installed.
            ValueError: If model cannot be loaded.
        """
        # Return cached model if exists
        if self._model is not None:
            return self._model

        # Load cross-encoder model
        try:
            sentence_transformers = require_module("sentence_transformers")

            if self._model_source == "modelscope":
                # Load from ModelScope
                require_module("modelscope")
                from modelscope.hub.snapshot_download import snapshot_download

                # Download model to cache
                model_dir = snapshot_download(self._model_name)

                # Load CrossEncoder from local path
                model = sentence_transformers.CrossEncoder(
                    model_dir, device=self._device
                )
            else:
                # Load CrossEncoder from Hugging Face (default)
                model = sentence_transformers.CrossEncoder(
                    self._model_name, device=self._device
                )

            return model

        except ImportError as e:
            if "modelscope" in str(e) and self._model_source == "modelscope":
                raise ImportError(
                    "ModelScope support requires the 'modelscope' package. "
                    "Please install it with: pip install modelscope"
                ) from e
            raise
        except Exception as e:
            raise ValueError(
                f"Failed to load CrossEncoder model '{self._model_name}' "
                f"from {self._model_source}: {e!s}"
            ) from e

    @property
    def query(self) -> str:
        """str: Query text used for semantic re-ranking."""
        return self._query

    @property
    def batch_size(self) -> int:
        """int: Batch size for processing query-document pairs."""
        return self._batch_size

    def rerank(self, query_results: dict[str, list[Doc]]) -> list[Doc]:
        """Re-rank documents using Sentence Transformer cross-encoder model.

        Evaluates each query-document pair using the cross-encoder model to compute
        relevance scores. Documents are then sorted by these scores and the top-k
        results are returned.

        Args:
            query_results (dict[str, list[Doc]]): Mapping from vector field names
                to lists of retrieved documents. Documents from all fields are
                deduplicated and re-ranked together.

        Returns:
            list[Doc]: Re-ranked documents (up to ``topn``) with updated ``score``
                fields containing relevance scores from the cross-encoder model.

        Raises:
            ValueError: If no valid documents are found or model inference fails.

        Note:
            - Duplicate documents (same ID) across fields are processed once
            - Documents with empty/missing ``rerank_field`` content are skipped
            - Returned scores are logits from the cross-encoder model
            - Higher scores indicate higher relevance
            - Processing time is O(n) where n is the number of documents

        Examples:
            >>> reranker = SentenceTransformerReRanker(
            ...     query="machine learning",
            ...     topn=3,
            ...     rerank_field="content"
            ... )
            >>> query_results = {
            ...     "vector1": [
            ...         Doc(id="1", score=0.9, fields={"content": "ML basics"}),
            ...         Doc(id="2", score=0.8, fields={"content": "DL tutorial"}),
            ...     ]
            ... }
            >>> reranked = reranker.rerank(query_results)
            >>> len(reranked) <= 3
            True
        """
        if not query_results:
            return []

        # Collect and deduplicate documents
        id_to_doc: dict[str, Doc] = {}
        doc_ids: list[str] = []
        contents: list[str] = []

        for _, query_result in query_results.items():
            for doc in query_result:
                doc_id = doc.id
                if doc_id in id_to_doc:
                    continue

                # Extract text content from specified field
                field_value = doc.field(self.rerank_field)
                rank_content = str(field_value).strip() if field_value else ""
                if not rank_content:
                    continue

                id_to_doc[doc_id] = doc
                doc_ids.append(doc_id)
                contents.append(rank_content)

        if not contents:
            raise ValueError("No documents to rerank")

        try:
            # Use standard cross-encoder predict method
            pairs = [[self.query, content] for content in contents]
            scores = self._model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            # Convert to float list if needed
            if hasattr(scores, "tolist"):
                scores = scores.tolist()
            else:
                scores = [float(s) for s in scores]

        except Exception as e:
            raise RuntimeError(f"Failed to compute rerank scores: {e!s}") from e

        # Create scored documents
        scored_docs = [
            (doc_ids[i], id_to_doc[doc_ids[i]], scores[i]) for i in range(len(doc_ids))
        ]

        # Sort by score (descending) and take top-k
        scored_docs.sort(key=lambda x: x[2], reverse=True)
        top_scored_docs = scored_docs[: self.topn]

        # Build result list with updated scores
        results: list[Doc] = []
        for _, doc, score in top_scored_docs:
            new_doc = doc._replace(score=score)
            results.append(new_doc)

        return results
