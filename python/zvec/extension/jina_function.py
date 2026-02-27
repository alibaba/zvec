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
from typing import ClassVar, Optional

from ..common.constants import TEXT
from ..tool import require_module


class JinaFunctionBase:
    """Base class for Jina AI functions.

    This base class provides common functionality for calling Jina AI APIs
    and handling responses. It supports embeddings (dense) operations via
    the OpenAI-compatible Jina Embeddings API.

    This class is not meant to be used directly. Use concrete implementations:
    - ``JinaDenseEmbedding`` for dense embeddings

    Args:
        model (str): Jina embedding model identifier.
        api_key (Optional[str]): Jina API authentication key.
        task (Optional[str]): Task type for the embedding model.

    Note:
        - This is an internal base class for code reuse across Jina features
        - Subclasses should inherit from appropriate Protocol
        - Provides unified API connection and response handling
        - Jina API is OpenAI-compatible, using the ``openai`` Python client
    """

    _BASE_URL: ClassVar[str] = "https://api.jina.ai/v1"

    # Model default dimensions
    _MODEL_DIMENSIONS: ClassVar[dict[str, int]] = {
        "jina-embeddings-v5-text-nano": 768,
        "jina-embeddings-v5-text-small": 1024,
    }

    # Model max tokens
    _MODEL_MAX_TOKENS: ClassVar[dict[str, int]] = {
        "jina-embeddings-v5-text-nano": 8192,
        "jina-embeddings-v5-text-small": 32768,
    }

    # Valid task types
    _VALID_TASKS: ClassVar[tuple[str, ...]] = (
        "retrieval.query",
        "retrieval.passage",
        "text-matching",
        "classification",
        "separation",
    )

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        task: Optional[str] = None,
    ):
        """Initialize the base Jina functionality.

        Args:
            model (str): Jina model name.
            api_key (Optional[str]): API key or None to use environment variable.
            task (Optional[str]): Task type for the embedding model.
                Valid values: "retrieval.query", "retrieval.passage",
                "text-matching", "classification", "separation".

        Raises:
            ValueError: If API key is not provided and not in environment,
                or if task is not a valid task type.
        """
        self._model = model
        self._api_key = api_key or os.environ.get("JINA_API_KEY")
        self._task = task

        if not self._api_key:
            raise ValueError(
                "Jina API key is required. Please provide 'api_key' parameter "
                "or set the 'JINA_API_KEY' environment variable. "
                "Get your key from: https://jina.ai/api-dashboard"
            )

        if task is not None and task not in self._VALID_TASKS:
            raise ValueError(
                f"Invalid task '{task}'. Valid tasks: {', '.join(self._VALID_TASKS)}"
            )

    @property
    def model(self) -> str:
        """str: The Jina model name currently in use."""
        return self._model

    @property
    def task(self) -> Optional[str]:
        """Optional[str]: The task type for the embedding model."""
        return self._task

    def _get_client(self):
        """Get OpenAI-compatible client instance configured for Jina API.

        Returns:
            OpenAI: Configured OpenAI client pointing to Jina API.

        Raises:
            ImportError: If openai package is not installed.
        """
        openai = require_module("openai")
        return openai.OpenAI(api_key=self._api_key, base_url=self._BASE_URL)

    def _call_text_embedding_api(
        self,
        input: TEXT,
        dimension: Optional[int] = None,
    ) -> list:
        """Call Jina Embeddings API.

        Args:
            input (TEXT): Input text to embed.
            dimension (Optional[int]): Target dimension for Matryoshka embeddings.

        Returns:
            list: Embedding vector as list of floats.

        Raises:
            RuntimeError: If API call fails.
            ValueError: If API returns error response.
        """
        try:
            client = self._get_client()

            # Prepare embedding parameters
            params = {"model": self.model, "input": input}

            # Add dimension parameter for Matryoshka support
            if dimension is not None:
                params["dimensions"] = dimension

            # Add task parameter via extra_body
            if self._task is not None:
                params["extra_body"] = {"task": self._task}

            # Call Jina API (OpenAI-compatible)
            response = client.embeddings.create(**params)

        except Exception as e:
            # Check if it's an OpenAI API error
            openai = require_module("openai")
            if isinstance(e, (openai.APIError, openai.APIConnectionError)):
                raise RuntimeError(f"Failed to call Jina API: {e!s}") from e
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

            return embedding_vector

        except (AttributeError, IndexError, TypeError) as e:
            raise ValueError(f"Failed to parse API response: {e!s}") from e
