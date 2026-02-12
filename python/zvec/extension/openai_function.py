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


class OpenAIFunctionBase:
    """Base class for OpenAI functions.

    This base class provides common functionality for calling OpenAI APIs
    and handling responses. It supports embeddings (dense) operations.

    This class is not meant to be used directly. Use concrete implementations:
    - ``OpenAIDenseEmbedding`` for dense embeddings

    Args:
        model (str): OpenAI model identifier.
        api_key (Optional[str]): OpenAI API authentication key.
        base_url (Optional[str]): Custom API base URL.

    Note:
        - This is an internal base class for code reuse across OpenAI features
        - Subclasses should inherit from appropriate Protocol
        - Provides unified API connection and response handling
    """

    # Model default dimensions
    _MODEL_DIMENSIONS: ClassVar[dict[str, int]] = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """Initialize the base OpenAI functionality.

        Args:
            model (str): OpenAI model name.
            api_key (Optional[str]): API key or None to use environment variable.
            base_url (Optional[str]): Custom API base URL or None for default.

        Raises:
            ValueError: If API key is not provided and not in environment.
        """
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._base_url = base_url

        if not self._api_key:
            raise ValueError(
                "OpenAI API key is required. Please provide 'api_key' parameter "
                "or set the 'OPENAI_API_KEY' environment variable."
            )

    @property
    def model(self) -> str:
        """str: The OpenAI model name currently in use."""
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

    def _call_text_embedding_api(
        self,
        input: TEXT,
        dimension: Optional[int] = None,
    ) -> list:
        """Call OpenAI Embeddings API.

        Args:
            input (TEXT): Input text to embed.
            dimension (Optional[int]): Target dimension (for models that support it).

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

            # Add dimension parameter for models that support it
            if dimension is not None:
                params["dimensions"] = dimension

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

            return embedding_vector

        except (AttributeError, IndexError, TypeError) as e:
            raise ValueError(f"Failed to parse API response: {e!s}") from e
