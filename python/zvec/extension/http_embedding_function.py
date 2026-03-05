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

import json
import os
import urllib.request
from functools import lru_cache
from typing import Optional

from ..common.constants import TEXT, DenseVectorType
from .embedding_function import DenseEmbeddingFunction


class HTTPDenseEmbedding(DenseEmbeddingFunction[TEXT]):
    """Dense text embedding function using any OpenAI-compatible HTTP endpoint.

    This class calls any server that implements the ``/v1/embeddings`` API
    (LM Studio, Ollama, vLLM, LocalAI, etc.) using only the Python standard
    library — no extra dependencies are required.

    The embedding dimension is detected automatically from the first server
    response.

    Args:
        base_url (str, optional): Base URL of the embedding server.
            Defaults to ``"http://localhost:1234"`` (LM Studio).
            Common values:

            - ``"http://localhost:1234"``  — LM Studio
            - ``"http://localhost:11434"`` — Ollama
        model (str, optional): Model identifier as expected by the server.
            Defaults to ``"text-embedding-nomic-embed-text-v1.5@f16"``.
        api_key (Optional[str], optional): Bearer token for authenticated
            endpoints.  Falls back to the ``OPENAI_API_KEY`` environment
            variable.  Leave as ``None`` for local servers that do not
            require authentication.
        timeout (int, optional): HTTP request timeout in seconds.
            Defaults to 30.

    Attributes:
        dimension (int): Embedding vector dimensionality (auto-detected).

    Raises:
        TypeError: If ``embed()`` receives a non-string input.
        ValueError: If input is empty/whitespace-only or the server returns
            an unexpected response format.
        RuntimeError: If the HTTP request fails or the server is unreachable.

    Examples:
        >>> from zvec.extension import HTTPDenseEmbedding
        >>>
        >>> # LM Studio (default)
        >>> emb = HTTPDenseEmbedding()
        >>> vector = emb.embed("Hello, world!")
        >>> len(vector)
        768
        >>>
        >>> # Ollama
        >>> emb = HTTPDenseEmbedding(
        ...     base_url="http://localhost:11434",
        ...     model="nomic-embed-text",
        ... )
        >>> vector = emb.embed("Semantic search with local models")

    See Also:
        - ``DenseEmbeddingFunction``: Protocol for dense embeddings.
        - ``OpenAIDenseEmbedding``: Cloud embedding via the OpenAI API.
    """

    ENDPOINT = "/v1/embeddings"

    def __init__(
        self,
        base_url: str = "http://localhost:1234",
        model: str = "text-embedding-nomic-embed-text-v1.5@f16",
        api_key: Optional[str] = None,
        timeout: int = 30,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._timeout = timeout
        self._dimension: Optional[int] = None

    @property
    def dimension(self) -> int:
        """int: Embedding vector dimensionality (auto-detected on first call)."""
        if self._dimension is None:
            self._dimension = len(self.embed("dimension probe"))
        return self._dimension

    def __call__(self, input: TEXT) -> DenseVectorType:
        """Make the embedding function callable."""
        return self.embed(input)

    @lru_cache(maxsize=256)
    def embed(self, input: TEXT) -> DenseVectorType:
        """Generate a dense embedding vector for the input text.

        Results are cached (LRU, up to 256 entries) so repeated strings
        do not trigger extra HTTP requests.

        Args:
            input (TEXT): Input text string to embed.  Must be non-empty
                after stripping whitespace.

        Returns:
            DenseVectorType: A list of floats representing the embedding.

        Raises:
            TypeError: If *input* is not a string.
            ValueError: If *input* is empty/whitespace-only or the server
                returns an unexpected response format.
            RuntimeError: If the HTTP request fails.
        """
        if not isinstance(input, TEXT):
            raise TypeError(f"Expected 'input' to be str, got {type(input).__name__}")

        input = input.strip()
        if not input:
            raise ValueError("Input text cannot be empty or whitespace only")

        url = self._base_url + self.ENDPOINT
        payload = json.dumps({"model": self._model, "input": input}).encode()

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"Embedding server returned HTTP {exc.code}: {exc.read().decode()}"
            ) from exc
        except OSError as exc:
            raise RuntimeError(
                f"Could not reach embedding server at {url}: {exc}"
            ) from exc

        try:
            vector: list[float] = body["data"][0]["embedding"]
        except (KeyError, IndexError) as exc:
            raise ValueError(
                f"Unexpected response format from embedding server: {body}"
            ) from exc

        return vector
