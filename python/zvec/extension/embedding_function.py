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

from abc import ABC, abstractmethod

from typing_extensions import Protocol, runtime_checkable

from ..common.constants import MD, DenseVectorType, SparseVectorType
from ..typing import DataType


@runtime_checkable
class DenseEmbeddingFunction(Protocol[MD], ABC):
    """Abstract base class for dense vector embedding functions.

    Dense embedding functions map multimodal input (text, image, or audio) to
    fixed-length real-valued vectors. You can inherit this class to create
    custom embedding functions for different modalities.

    Type Parameters:
        MD: The type of input data (bound to Embeddable: TEXT, IMAGE, or AUDIO).

    Args:
        dimension (int): Dimensionality of the output embedding vector.
        data_type (DataType, optional): Numeric type of the embedding.
            Defaults to ``DataType.VECTOR_FP32``.

    Note:
        - Subclasses must implement the ``embed()`` method.
        - This class is callable: ``embedding_func(input)`` is equivalent to
          ``embedding_func.embed(input)``.

    Examples:
        >>> # Using built-in text embedding
        >>> text_emb = SomeTextEmbedding(dimension=768)
        >>> vector = text_emb.embed("Hello world")

        >>> # Using built-in image embedding
        >>> img_emb = SomeImageEmbedding(dimension=512)
        >>> vector = img_emb.embed("/path/to/image.jpg")

        >>> # Custom text embedding function
        >>> class MyTextEmbedding(DenseEmbeddingFunction):
        ...     def __init__(self):
        ...         super().__init__(dimension=384)
        ...         self.model = load_my_model()
        ...
        ...     def embed(self, input: str) -> list[float]:
        ...         return self.model.encode(input).tolist()

        >>> # Custom image embedding function
        >>> class MyImageEmbedding(DenseEmbeddingFunction):
        ...     def __init__(self):
        ...         super().__init__(dimension=2048, data_type=DataType.VECTOR_FP32)
        ...         self.model = load_image_model()
        ...
        ...     def embed(self, input: Union[str, bytes, np.ndarray]) -> np.ndarray:
        ...         if isinstance(input, str):
        ...             image = load_image_from_path(input)
        ...         elif isinstance(input, bytes):
        ...             image = decode_image_bytes(input)
        ...         else:
        ...             image = input
        ...         return self.model.extract_features(image)
    """

    def __init__(self, dimension: int, data_type: DataType = DataType.VECTOR_FP32):
        self._dimension = dimension
        self._data_type = data_type

    @property
    def dimension(self) -> int:
        """int: The expected dimensionality of the embedding vector."""
        return self._dimension

    @property
    def data_type(self) -> DataType:
        """DataType: The numeric data type of the embedding (e.g., VECTOR_FP32)."""
        return self._data_type

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
                Length must equal ``self.dimension``.
        """
        raise NotImplementedError

    def __call__(self, input: MD) -> DenseVectorType:
        return self.embed(input)


class SparseEmbeddingFunction(Protocol[MD], ABC):
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
        raise NotImplementedError
