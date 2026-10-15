"""Regression checks for local formats, subset GT and actual bounded HTTP ranges."""

from __future__ import annotations

import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import download_text2image as dl
import numpy as np
import prepare_local_datasets as local
import pytest


def test_vecs_preserves_rows_and_rejects_bad_headers(tmp_path):
    path = tmp_path / "test.fvecs"
    values = np.arange(12, dtype="<f4").reshape(4, 3)
    path.write_bytes(b"".join(struct.pack("<I", 3) + row.tobytes() for row in values))
    np.testing.assert_array_equal(local.read_vecs(path, "<f4", 3), values)
    with path.open("r+b") as f:
        f.seek(16)
        f.write(struct.pack("<I", 2))
    with pytest.raises(ValueError, match="dimension"):
        local.read_vecs(path, "<f4", 3)


def test_ip_gt_cache_tracks_source_and_matches_brute_force(tmp_path):
    local.faiss.omp_set_num_threads(1)
    rng = np.random.default_rng(2)
    base = rng.normal(size=(30, 5)).astype("float32")
    queries = rng.normal(size=(7, 5)).astype("float32")
    path = tmp_path / "gt"
    for factor in (1, -1):
        current = base * factor
        result = local.exact_ip_gt(path, current, queries, k=4, batch_size=3)
        np.testing.assert_array_equal(
            result, np.argsort(-(queries @ current.T), axis=1)[:, :4]
        )
        ids, scores = local.read_gt(path, 7, 4, 30)
        np.testing.assert_allclose(
            scores, np.take_along_axis(queries @ current.T, ids, axis=1), atol=1e-5
        )


@pytest.mark.parametrize("expected_header", [(40, 3), None])
def test_curl_download_bounds_and_repairs_corruption(
    tmp_path, monkeypatch, expected_header
):
    monkeypatch.setattr(dl, "CHUNK", 32)
    payload = struct.pack("<II", 40, 3) + np.arange(120, dtype="<f4").tobytes()
    requested = []
    truncated = False

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_HEAD(self):
            self.send_response(200)
            self.send_header("Content-Length", len(payload))
            self.send_header("ETag", '"fixture"')
            self.end_headers()

        def do_GET(self):
            nonlocal truncated
            start, end = map(
                int, self.headers["Range"].removeprefix("bytes=").split("-")
            )
            requested.append((start, end))
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
            self.send_header("Content-Length", end - start + 1)
            self.end_headers()
            if start == 0 and not truncated:
                truncated = True
                self.wfile.write(payload[:16])
                return
            self.wfile.write(payload[start : end + 1])

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        # A prefix of 8 rows, even though the source header announces 40 rows.
        size = 8 + 8 * 3 * 4
        entry = (
            "query.fbin",
            f"http://127.0.0.1:{server.server_port}/source",
            size,
            expected_header,
        )
        dl.download(tmp_path, entry, 2)
        assert (tmp_path / "query.fbin").read_bytes() == payload[:size]
        assert max(end for _, end in requested) == size - 1
        assert (16, 31) in requested  # Resume within an interrupted range.
        requests = len(requested)
        dl.download(tmp_path, entry, 2)
        assert len(requested) == requests
        (tmp_path / "query.fbin").write_bytes(b"broken")
        dl.download(tmp_path, entry, 2)
        assert (tmp_path / "query.fbin").read_bytes() == payload[:size]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
