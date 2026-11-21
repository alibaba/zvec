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
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from zvec.extension import HTTPDenseEmbedding


@pytest.fixture
def embedding_server(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    requests = []
    responses = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.path == "/v1/embeddings"
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload["input"])
            body = json.dumps(
                responses.get(
                    payload["input"], {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
                )
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests, responses
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_dimension_is_learned_from_first_embedding(embedding_server):
    url, requests, _ = embedding_server
    embedding = HTTPDenseEmbedding(base_url=url)

    assert embedding.embed("actual text") == [0.1, 0.2, 0.3]
    assert embedding.dimension == 3
    assert requests == ["actual text"]


def test_dimension_probe_is_used_when_no_embedding_exists(embedding_server):
    url, requests, _ = embedding_server
    embedding = HTTPDenseEmbedding(base_url=url)

    assert embedding.dimension == 3
    assert embedding.dimension == 3
    assert requests == ["dimension probe"]


@pytest.mark.parametrize("probe_first", [False, True])
def test_dimension_mismatch_is_reported(embedding_server, probe_first):
    url, _, responses = embedding_server
    responses["different"] = {"data": [{"embedding": [0.4, 0.5]}]}
    embedding = HTTPDenseEmbedding(base_url=url)
    if probe_first:
        assert embedding.dimension == 3
    else:
        embedding.embed("actual text")

    with pytest.raises(ValueError, match="[Dd]imension mismatch"):
        embedding.embed("different")
    assert embedding.dimension == 3
    responses["different"] = {"data": [{"embedding": [0.4, 0.5, 0.6]}]}
    assert embedding.embed("different") == [0.4, 0.5, 0.6]


@pytest.mark.parametrize(
    "body",
    [{"data": None}, {"data": [{"embedding": []}]}, {"data": [{"embedding": "bad"}]}],
)
def test_invalid_response_does_not_set_dimension(embedding_server, body):
    url, requests, responses = embedding_server
    responses["invalid"] = body
    embedding = HTTPDenseEmbedding(base_url=url)

    with pytest.raises(ValueError, match="Unexpected response format"):
        embedding.embed("invalid")
    assert embedding.embed("actual text") == [0.1, 0.2, 0.3]
    assert embedding.dimension == 3
    assert requests == ["invalid", "actual text"]
