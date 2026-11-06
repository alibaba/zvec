"""Offline regression tests for the benchmark data and command pipeline."""

from __future__ import annotations

import io
import json
import shlex
import struct
import subprocess
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import prepare_datasets as prep
import pyarrow.parquet as pq
import pytest


@pytest.fixture
def vectors():
    rng = np.random.default_rng(42)
    # Unequal norms make accidental cosine normalization visible.
    base = rng.normal(size=(17, 5)).astype("float32")
    base *= np.arange(1, 18, dtype="float32")[:, None]
    queries = rng.normal(size=(7, 5)).astype("float32")
    return base, queries


@pytest.mark.parametrize("metric", ["L2", "IP"])
def test_parquet_and_exact_ground_truth(tmp_path, vectors, metric):
    base, queries = vectors
    prep.prepare_arrays(
        tmp_path, base, queries, metric=metric, k=4, batch_size=6, query_batch_size=3
    )
    train = pq.read_table(tmp_path / "train.parquet").to_pydict()
    test = pq.read_table(tmp_path / "test.parquet").to_pydict()
    gt = pq.read_table(tmp_path / "neighbors.parquet").to_pydict()
    assert train["id"] == list(range(len(base)))
    assert test["id"] == gt["id"] == list(range(len(queries)))
    np.testing.assert_array_equal(train["emb"], base)
    np.testing.assert_array_equal(test["emb"], queries)
    if metric == "L2":
        distances = ((queries[:, None, :] - base[None, :, :]) ** 2).sum(axis=2)
    else:
        distances = -(queries @ base.T)
    np.testing.assert_array_equal(
        gt["neighbors_id"], np.argsort(distances, axis=1)[:, :4]
    )


def test_supplied_gt_validation(tmp_path, vectors):
    base, queries = vectors
    bad_gt = np.full((len(queries), 4), len(base), dtype=np.int64)
    with pytest.raises(ValueError, match="outside"):
        prep.prepare_arrays(
            tmp_path, base, queries, metric="L2", k=4, supplied_gt=bad_gt
        )
    bad_gt[:] = 0
    with pytest.raises(ValueError, match="duplicate"):
        prep.prepare_arrays(
            tmp_path, base, queries, metric="L2", k=4, supplied_gt=bad_gt
        )


def test_nonfinite_input_rejected(tmp_path, vectors):
    base, queries = vectors
    base[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        prep.prepare_arrays(tmp_path, base, queries, metric="IP", k=4)


def test_fbin_prefix_shape_and_header(tmp_path):
    path = tmp_path / "base.fbin"
    values = np.arange(12, dtype="<f4").reshape(4, 3)
    # Prefix deliberately retains the original, larger row count in its header.
    path.write_bytes(struct.pack("<II", 1_000_000_000, 3) + values.tobytes())
    np.testing.assert_array_equal(prep.read_fbin_prefix(path, 4, 3), values)
    with pytest.raises(ValueError, match="shape"):
        prep.read_fbin_prefix(path, 4, 5)
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="size"):
        prep.read_fbin_prefix(path, 4, 3)


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {"Content-Length": str(len(data))}
        self.bytes_read = 0

    def read(self, size=-1):
        value = super().read(size)
        self.bytes_read += len(value)
        return value


@pytest.mark.parametrize("status", [200, 206])
def test_download_is_bounded_and_cache_verified(tmp_path, status):
    response = Response(
        b"0123456789" * 100, status, {"Content-Range": "bytes 0-9/1000"}
    )
    path = tmp_path / "prefix.fbin"
    with patch.object(prep.urllib.request, "urlopen", return_value=response) as fetch:
        prep.download("https://example.test/base", path, prefix_bytes=10)
        assert fetch.call_args.args[0].get_header("Range") == "bytes=0-9"
        assert (
            fetch.call_args.args[0].get_header("User-agent")
            == "zvec-continuous-bench/1.0"
        )
    assert response.bytes_read == 10
    assert path.read_bytes() == b"0123456789"
    with patch.object(
        prep.urllib.request, "urlopen", side_effect=AssertionError("cache miss")
    ):
        prep.download("https://example.test/base", path, prefix_bytes=10)
    path.write_bytes(b"corruption")
    with patch.object(
        prep.urllib.request, "urlopen", return_value=Response(b"0123456789")
    ) as fetch:
        prep.download("https://example.test/base", path, prefix_bytes=10)
        assert fetch.call_count == 1


@pytest.mark.parametrize("bad_range", [None, "bytes 1-10/100", "bytes 0-8/100"])
def test_invalid_partial_response_not_published(tmp_path, bad_range):
    def response(*args, **kwargs):
        del args, kwargs
        return Response(b"0123456789", 206, {"Content-Range": bad_range or ""})

    path = tmp_path / "base.fbin"
    with (
        patch.object(prep.urllib.request, "urlopen", side_effect=response),
        patch.object(prep.time, "sleep"),
        pytest.raises(ValueError, match="Content-Range"),
    ):
        prep.download("https://example.test/base", path, prefix_bytes=10)
    assert not path.exists()
    assert not path.with_suffix(".fbin.part").exists()


def test_truncated_response_not_published(tmp_path):
    path = tmp_path / "data.hdf5"
    with (
        patch.object(
            prep.urllib.request,
            "urlopen",
            side_effect=lambda *_a, **_k: Response(
                b"short", headers={"Content-Length": "100"}
            ),
        ),
        patch.object(prep.time, "sleep"),
        pytest.raises(ValueError, match="Truncated"),
    ):
        prep.download("https://example.test/data", path)
    assert not path.exists()


def test_gist_full_pipeline_cache_and_repair(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, "BASE_COUNT", 8)
    monkeypatch.setattr(prep, "K", 3)
    raw = tmp_path / "_sources"
    raw.mkdir()
    with h5py.File(raw / "gist-960-euclidean.hdf5", "w") as data:
        data.attrs["distance"] = "euclidean"
        data["train"] = np.zeros((8, 960), dtype="float32")
        data["test"] = np.zeros((1000, 960), dtype="float32")
        data["neighbors"] = np.tile(np.arange(3), (1000, 1))
    with patch.object(prep, "download", return_value={"fixture": True}) as fetch:
        prep.prepare_one("gist", tmp_path, raw, recompute_gt=False, force=False)
        assert fetch.call_count == 1
        destination = tmp_path / "gist/gist_1m_v1"
        manifest = json.loads((destination / "manifest.json").read_text())
        assert prep.cache_valid(destination, manifest["config"])
        assert pq.read_table(destination / "neighbors.parquet").num_rows == 1000
        prep.prepare_one("gist", tmp_path, raw, recompute_gt=False, force=False)
        assert fetch.call_count == 1
        (destination / "train.parquet").write_bytes(b"broken")
        assert not prep.cache_valid(destination, manifest["config"])
        prep.prepare_one("gist", tmp_path, raw, recompute_gt=False, force=False)
        assert fetch.call_count == 2
        assert prep.cache_valid(destination, manifest["config"])
        with (
            patch.object(prep, "prepare_arrays", side_effect=ValueError("GT failed")),
            pytest.raises(ValueError, match="GT failed"),
        ):
            prep.prepare_one("gist", tmp_path, raw, recompute_gt=False, force=True)
        assert prep.cache_valid(destination, manifest["config"])


def test_text2image_full_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, "BASE_COUNT", 8)
    monkeypatch.setattr(prep, "K", 3)
    prep.faiss.omp_set_num_threads(1)
    raw = tmp_path / "_sources"
    raw.mkdir()
    rng = np.random.default_rng(19)
    base = rng.normal(size=(8, 200)).astype("<f4")
    queries = rng.normal(size=(10_000, 200)).astype("<f4")
    (raw / "text2image-base-1m.fbin").write_bytes(
        struct.pack("<II", 1_000_000_000, 200) + base.tobytes()
    )
    (raw / "text2image-query-10k.fbin").write_bytes(
        struct.pack("<II", 100_000, 200) + queries.tobytes()
    )
    with patch.object(prep, "download", return_value={"fixture": True}) as fetch:
        prep.prepare_one("text2image", tmp_path, raw, recompute_gt=False, force=False)
        assert [call.kwargs["prefix_bytes"] for call in fetch.call_args_list] == [
            8 + 8 * 200 * 4,
            8 + 10_000 * 200 * 4,
        ]
        destination = tmp_path / "text2image/text2image_1m_v1"
        manifest = json.loads((destination / "manifest.json").read_text())
        assert manifest["config"]["metric"] == "IP"
        assert manifest["config"]["gt"] == "faiss-flat-float32"
        assert prep.cache_valid(destination, manifest["config"])
        gt = pq.read_table(destination / "neighbors.parquet").to_pydict()
        np.testing.assert_array_equal(
            gt["neighbors_id"], np.argsort(-(queries @ base.T), axis=1)[:, :3]
        )


def test_dry_run_preserves_existing_matrix_and_adds_two_cases(tmp_path):
    script = Path(__file__).with_name("run_vdb.sh")
    result = subprocess.run(
        ["bash", str(script), "--dry-run"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    commands = [shlex.split(line) for line in result.stdout.splitlines()]
    assert len(commands) == 20
    originals = {
        "Performance768D1M": ("15", "180", False),
        "Performance768D10M": ("50", "118", True),
        "Performance1536D500K": ("50", "100", True),
    }
    for index, command in enumerate(commands):

        def arg(flag, command=command):
            return command[command.index(flag) + 1]

        assert command[:2] == ["vectordbbench", "zvec"]
        quantization = ("int8", "int4", "fp16", None)[index % 4]
        if quantization is None:
            assert "--quantize-type" not in command
        else:
            assert arg("--quantize-type") == quantization
        if index < 12:
            m, ef, refiner = originals[arg("--case-type")]
            assert (arg("--m"), arg("--ef-search")) == (m, ef)
            assert ("--is-using-refiner" in command) == refiner
        else:
            assert arg("--case-type") == "PerformanceCustomDataset"
            assert arg("--custom-dataset-metric-type") == ("L2" if index < 16 else "IP")
            assert arg("--custom-dataset-dim") == ("960" if index < 16 else "200")
    assert len({command[command.index("--path") + 1] for command in commands}) == 20
    assert not list(tmp_path.iterdir())
    workflow = script.parent.parent / "continuous_bench.yml"
    assert "    if: false" in workflow.read_text()
