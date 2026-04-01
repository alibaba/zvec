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

from abc import abstractmethod
from typing import Protocol, runtime_checkable

from ..common.constants import MD, DenseVectorType, SparseVectorType


@runtime_checkable
class DenseEmbeddingFunction(Protocol[MD]):
    """Protocol for dense vector embedding functions.

    Dense embedding functions map multimodal input (text, image, or audio) to
    fixed-length real-valued vectors. This is a Protocol class that defines
    the interface - implementations should provide their own initialization
    and properties.

    Type Parameters:
        MD: The type of input data (bound to Embeddable: TEXT, IMAGE, or AUDIO).

    Note:
        - This is a Protocol class - it only defines the ``embed()`` interface.
        - Implementations are free to define their own ``__init__``, properties,
          and additional methods as needed.
        - The ``embed()`` method is the only required interface.

    Examples:
        >>> # Custom text embedding implementation
        >>> class MyTextEmbedding:
        ...     def __init__(self, dimension: int, model_name: str):
        ...         self.dimension = dimension
        ...         self.model = load_model(model_name)
        ...
        ...     def embed(self, input: str) -> list[float]:
        ...         return self.model.encode(input).tolist()

        >>> # Custom image embedding implementation
        >>> class MyImageEmbedding:
        ...     def __init__(self, dimension: int = 512):
        ...         self.dimension = dimension
        ...         self.model = load_image_model()
        ...
        ...     def embed(self, input: Union[str, bytes, np.ndarray]) -> list[float]:
        ...         if isinstance(input, str):
        ...             image = load_image_from_path(input)
        ...         else:
        ...             image = input
        ...         return self.model.extract_features(image).tolist()

        >>> # Using built-in implementations
        >>> from zvec.extension import QwenDenseEmbedding
        >>> text_emb = QwenDenseEmbedding(dimension=768, api_key="sk-xxx")
        >>> vector = text_emb.embed("Hello world")
    """

    @abstractmethod
    def embed(self, input: MD) -> DenseVectorType:
        """Generate a dense embedding vector for the input data.

        Args:
            input (MD): Multimodal input data to embed. Can be:
                - TEXT (str): Text string
                - IMAGE (str | bytes | np.ndarray): Image file path, raw bytes, or array
                - AUDIO (str | bytes | np.ndarray): Audio file path, raw bytes, or array

        Returns:
            DenseVectorType: A dense vector representing the embedding.
                Can be list[float], list[int], or np.ndarray.
                Length should match the implementation's dimension.
        """
        ...


@runtime_checkable
class SparseEmbeddingFunction(Protocol[MD]):
    """Abstract base class for sparse vector embedding functions.

    Sparse embedding functions map multimodal input (text, image, or audio) to
    a dictionary of {index: weight}, where only non-zero dimensions are stored.
    You can inherit this class to create custom sparse embedding functions.

    Type Parameters:
        MD: The type of input data (bound to Embeddable: TEXT, IMAGE, or AUDIO).

    Note:
        Subclasses must implement the ``embed()`` method.

    Examples:
        >>> # Using built-in text sparse embedding (e.g., BM25, TF-IDF)
        >>> sparse_emb = SomeSparseEmbedding()
        >>> vector = sparse_emb.embed("Hello world")
        >>> # Returns: {0: 0.5, 42: 1.2, 100: 0.8}

        >>> # Custom BM25 sparse embedding function
        >>> class MyBM25Embedding(SparseEmbeddingFunction):
        ...     def __init__(self, vocab_size: int = 10000):
        ...         self.vocab_size = vocab_size
        ...         self.tokenizer = MyTokenizer()
        ...
        ...     def embed(self, input: str) -> dict[int, float]:
        ...         tokens = self.tokenizer.tokenize(input)
        ...         sparse_vector = {}
        ...         for token_id, weight in self._calculate_bm25(tokens):
        ...             if weight > 0:
        ...                 sparse_vector[token_id] = weight
        ...         return sparse_vector
        ...
        ...     def _calculate_bm25(self, tokens):
        ...         # BM25 calculation logic
        ...         pass

        >>> # Custom sparse image feature extractor
        >>> class MySparseImageEmbedding(SparseEmbeddingFunction):
        ...     def embed(self, input: Union[str, bytes, np.ndarray]) -> dict[int, float]:
        ...         image = self._load_image(input)
        ...         features = self._extract_sparse_features(image)
        ...         return {idx: val for idx, val in enumerate(features) if val != 0}
    """

    @abstractmethod
    def embed(self, input: MD) -> SparseVectorType:
        """Generate a sparse embedding for the input data.

        Args:
            input (MD): Multimodal input data to embed. Can be:
                - TEXT (str): Text string
                - IMAGE (str | bytes | np.ndarray): Image file path, raw bytes, or array
                - AUDIO (str | bytes | np.ndarray): Audio file path, raw bytes, or array

        Returns:
            SparseVectorType: Mapping from dimension index to non-zero weight.
                Only dimensions with non-zero values are included.
        """
        ...
