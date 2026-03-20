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

from ..tool import require_module


class SentenceTransformerFunctionBase:
    """Base class for Sentence Transformer functions (both dense and sparse).

    This base class provides common functionality for loading and managing
    sentence-transformers models from Hugging Face or ModelScope. It supports
    both dense models (e.g., all-MiniLM-L6-v2) and sparse models (e.g., SPLADE).

    This class is not meant to be used directly. Use concrete implementations:
    - ``SentenceTransformerEmbeddingFunction`` for dense embeddings
    - ``SentenceTransformerSparseEmbeddingFunction`` for sparse embeddings
    - ``DefaultDenseEmbedding`` for default dense embeddings
    - ``DefaultSparseEmbedding`` for default sparse embeddings

    Args:
        model_name (str): Model identifier or local path.
        model_source (Literal["huggingface", "modelscope"]): Model source.
        device (Optional[str]): Device to run the model on.

    Note:
        - This is an internal base class for code reuse
        - Subclasses should inherit from appropriate Protocol (Dense/Sparse)
        - Provides model loading and management functionality
    """

    def __init__(
        self,
        model_name: str,
        model_source: Literal["huggingface", "modelscope"] = "huggingface",
        device: Optional[str] = None,
    ):
        """Initialize the base Sentence Transformer functionality.

        Args:
            model_name (str): Model identifier or local path.
            model_source (Literal["huggingface", "modelscope"]): Model source.
            device (Optional[str]): Device to run the model on.

        Raises:
            ValueError: If model_source is invalid.
        """
        # Validate model_source
        if model_source not in ("huggingface", "modelscope"):
            raise ValueError(
                f"Invalid model_source: '{model_source}'. "
                "Must be 'huggingface' or 'modelscope'."
            )

        self._model_name = model_name
        self._model_source = model_source
        self._device = device
        self._model = None

    @property
    def model_name(self) -> str:
        """str: The Sentence Transformer model name currently in use."""
        return self._model_name

    @property
    def model_source(self) -> str:
        """str: The model source being used ("huggingface" or "modelscope")."""
        return self._model_source

    @property
    def device(self) -> str:
        """str: The device the model is running on."""
        model = self._get_model()
        if model is not None:
            return str(model.device)
        return self._device or "cpu"

    def _get_model(self):
        """Load or retrieve the Sentence Transformer model.

        Returns:
            SentenceTransformer or SparseEncoder: The loaded model instance.

        Raises:
            ImportError: If required packages are not installed.
            ValueError: If model cannot be loaded.
        """
        # Return cached model if exists
        if self._model is not None:
            return self._model

        # Load model
        try:
            sentence_transformers = require_module("sentence_transformers")

            if self._model_source == "modelscope":
                # Load from ModelScope
                require_module("modelscope")
                from modelscope.hub.snapshot_download import snapshot_download

                # Download model to cache
                model_dir = snapshot_download(self._model_name)

                # Load from local path
                self._model = sentence_transformers.SentenceTransformer(
                    model_dir, device=self._device, trust_remote_code=True
                )
            else:
                # Load from Hugging Face (default)
                self._model = sentence_transformers.SentenceTransformer(
                    self._model_name, device=self._device, trust_remote_code=True
                )

            return self._model

        except ImportError as e:
            if "modelscope" in str(e) and self._model_source == "modelscope":
                raise ImportError(
                    "ModelScope support requires the 'modelscope' package. "
                    "Please install it with: pip install modelscope"
                ) from e
            raise
        except Exception as e:
            raise ValueError(
                f"Failed to load Sentence Transformer model '{self._model_name}' "
                f"from {self._model_source}: {e!s}"
            ) from e

    def _is_sparse_model(self) -> bool:
        """Check if the loaded model is a sparse encoder (e.g., SPLADE).

        Returns:
            bool: True if model supports sparse encoding.
        """
        model = self._get_model()
        # Check if model has sparse encoding methods
        return hasattr(model, "encode_query") or hasattr(model, "encode_document")
