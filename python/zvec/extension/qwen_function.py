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
from http import HTTPStatus
from typing import Optional

from ..common.constants import TEXT
from ..tool import require_module


class QwenFunctionBase:
    """Base class for Qwen (DashScope) embedding functions.

    This base class provides common functionality for calling DashScope API
    and handling responses. It supports both dense and sparse embeddings.

    This class is not meant to be used directly. Use ``QwenDenseEmbedding``
    for dense embeddings or ``QwenSparseEmbedding`` for sparse embeddings.

    Args:
        model (str): DashScope embedding model identifier.
        api_key (Optional[str]): DashScope API authentication key.

    Note:
        - This is an internal base class for code reuse
        - Subclasses should inherit from appropriate Protocol (Dense/Sparse)
        - Provides API connection and response handling functionality
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
    ):
        """Initialize the base Qwen embedding functionality.

        Args:
            model (str): DashScope model name.
            api_key (Optional[str]): API key or None to use environment variable.

        Raises:
            ValueError: If API key is not provided and not in environment.
        """
        self._model = model
        self._api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self._api_key:
            raise ValueError(
                "DashScope API key is required. Please provide 'api_key' parameter "
                "or set the 'DASHSCOPE_API_KEY' environment variable."
            )

    @property
    def model(self) -> str:
        """str: The DashScope embedding model name currently in use."""
        return self._model

    def _get_connection(self):
        """Establish connection to DashScope API.

        Returns:
            module: The dashscope module with API key configured.

        Raises:
            ImportError: If dashscope package is not installed.
        """
        dashscope = require_module("dashscope")
        dashscope.api_key = self._api_key
        return dashscope

    def _call_text_embedding_api(
        self,
        input: TEXT,
        dimension: int,
        output_type: str,
        text_type: Optional[str] = None,
    ) -> dict:
        """Call DashScope TextEmbedding API.

        Args:
            input (TEXT): Input text to embed.
            dimension (int): Target embedding dimension.
            output_type (str): Output type ("dense" or "sparse").
            text_type (Optional[str]): Text type ("query" or "document").

        Returns:
            dict: API response output field.

        Raises:
            RuntimeError: If API call fails.
            ValueError: If API returns error response.
        """
        try:
            # Prepare API call parameters
            call_params = {
                "model": self.model,
                "input": input,
                "dimension": dimension,
                "output_type": output_type,
            }

            # Add optional text_type parameter if provided
            if text_type is not None:
                call_params["text_type"] = text_type

            resp = self._get_connection().TextEmbedding.call(**call_params)
        except Exception as e:
            raise RuntimeError(f"Failed to call DashScope API: {e!s}") from e

        if resp.status_code != HTTPStatus.OK:
            error_msg = getattr(resp, "message", "Unknown error")
            error_code = getattr(resp, "code", "N/A")
            raise ValueError(
                f"DashScope API error: [Code={error_code}, "
                f"Status={resp.status_code}] {error_msg}"
            )

        output = getattr(resp, "output", None)
        if not isinstance(output, dict):
            raise ValueError(
                "Invalid API response: missing or malformed 'output' field"
            )

        return output
