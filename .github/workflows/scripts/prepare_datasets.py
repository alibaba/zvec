"""Prepare versioned GIST/L2 and Text2Image/IP Parquet datasets for VectorDBBench."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import logging
import os
import re
import shutil
import struct
import tempfile
import time
import urllib.request
from pathlib import Path

import faiss
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

LOG = logging.getLogger(__name__)
VERSION = 1
BASE_COUNT = 1_000_000
K = 100
GIST_URL = "https://ann-benchmarks.com/gist-960-euclidean.hdf5"
T2I_ROOT = "https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/"
FILES = ("train.parquet", "test.parquet", "neighbors.parquet")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def download(url: str, target: Path, *, prefix_bytes: int | None = None) -> dict:
    """Cache complete downloads, or bounded prefixes; never consume a full T2I base."""
    metadata_path = target.with_suffix(target.suffix + ".json")
    identity = {"url": url, "prefix_bytes": prefix_bytes}
    if target.exists() and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, ValueError):
            metadata = {}
        if all(
            metadata.get(key) == value for key, value in identity.items()
        ) and metadata.get("sha256") == sha256(target):
            return metadata
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    # ANN-Benchmarks rejects urllib's default User-Agent; identify this client.
    headers = {
        "Accept-Encoding": "identity",
        "User-Agent": "zvec-continuous-bench/1.0",
    }
    if prefix_bytes is not None:
        headers["Range"] = f"bytes=0-{prefix_bytes - 1}"
    for attempt in range(3):
        try:
            LOG.info("Downloading %s -> %s", url, target)
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status == 206:
                    match = re.fullmatch(
                        r"bytes 0-(\d+)/(\d+)",
                        response.headers.get("Content-Range", ""),
                    )
                    if (
                        prefix_bytes is None
                        or not match
                        or int(match[1]) + 1 != prefix_bytes
                        or int(match[2]) < prefix_bytes
                    ):
                        raise ValueError(
                            "Unexpected Content-Range; refusing incorrect prefix"
                        )
                elif response.status != 200:
                    raise ValueError(f"Unexpected HTTP status: {response.status}")
                length = response.headers.get("Content-Length")
                expected = (
                    prefix_bytes if prefix_bytes is not None else int(length or 0)
                )
                if not expected:
                    raise ValueError("A complete download requires Content-Length")
                # Some servers ignore Range. Read only the prefix and close the connection.
                remaining = expected
                with partial.open("wb") as output:
                    while remaining:
                        block = response.read(min(8 * 1024 * 1024, remaining))
                        if not block:
                            raise ValueError(
                                f"Truncated download: missing {remaining} bytes"
                            )
                        output.write(block)
                        remaining -= len(block)
                metadata = {
                    **identity,
                    "bytes": expected,
                    "etag": response.headers.get("ETag"),
                    "last_modified": response.headers.get("Last-Modified"),
                    "sha256": sha256(partial),
                }
            partial.replace(target)
            write_json(metadata_path, metadata)
            return metadata
        except (OSError, ValueError):
            partial.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def read_fbin_prefix(path: Path, rows: int, dimension: int) -> np.memmap:
    with path.open("rb") as stream:
        header = stream.read(8)
    if len(header) != 8:
        raise ValueError(f"Missing fbin header: {path}")
    source_rows, source_dimension = struct.unpack("<II", header)
    if source_rows < rows or source_dimension != dimension:
        raise ValueError(f"Unexpected fbin shape: {(source_rows, source_dimension)}")
    if path.stat().st_size != 8 + rows * dimension * 4:
        raise ValueError(f"Incorrect fbin prefix size: {path}")
    # Preserve the original header as provenance; explicitly map only the selected prefix.
    return np.memmap(path, dtype="<f4", mode="r", offset=8, shape=(rows, dimension))


def float_batch(values, start: int, end: int) -> np.ndarray:
    batch = np.ascontiguousarray(values[start:end], dtype=np.float32)
    if not np.isfinite(batch).all():
        raise ValueError("Dataset contains non-finite vectors")
    return batch


def vector_table(ids: np.ndarray, vectors: np.ndarray) -> pa.Table:
    embeddings = pa.FixedSizeListArray.from_arrays(
        pa.array(vectors.reshape(-1), type=pa.float32()), vectors.shape[1]
    )
    return pa.table({"id": pa.array(ids, type=pa.int64()), "emb": embeddings})


def prepare_arrays(
    destination: Path,
    base,
    queries,
    *,
    metric: str,
    k: int,
    supplied_gt=None,
    batch_size: int = 8192,
    query_batch_size: int = 32,
) -> None:
    """Stream Parquet rows and generate exact GT, keeping one flat index in memory."""
    if metric not in {"L2", "IP"}:
        raise ValueError(f"Unsupported metric: {metric}")
    if (
        len(base.shape) != 2
        or len(queries.shape) != 2
        or base.shape[1] != queries.shape[1]
    ):
        raise ValueError("Base/query dimensions do not match")
    if (
        not 0 < k <= len(base)
        or not len(queries)
        or min(batch_size, query_batch_size) <= 0
    ):
        raise ValueError("Invalid row count, K or batch size")
    if supplied_gt is not None and supplied_gt.shape != (len(queries), k):
        raise ValueError("Ground truth shape does not match queries and K")
    dimension = base.shape[1]
    index = None
    if supplied_gt is None:
        index = (
            faiss.IndexFlatL2(dimension)
            if metric == "L2"
            else faiss.IndexFlatIP(dimension)
        )
    destination.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([("id", pa.int64()), ("emb", pa.list_(pa.float32(), dimension))])
    with pq.ParquetWriter(
        destination / "train.parquet", schema, compression="zstd"
    ) as writer:
        for start in range(0, len(base), batch_size):
            end = min(start + batch_size, len(base))
            vectors = float_batch(base, start, end)
            writer.write_table(vector_table(np.arange(start, end), vectors))
            if index is not None:
                index.add(vectors)
    gt_schema = pa.schema([("id", pa.int64()), ("neighbors_id", pa.list_(pa.int64()))])
    with (
        pq.ParquetWriter(
            destination / "test.parquet", schema, compression="zstd"
        ) as test_writer,
        pq.ParquetWriter(
            destination / "neighbors.parquet", gt_schema, compression="zstd"
        ) as gt_writer,
    ):
        for start in range(0, len(queries), query_batch_size):
            end = min(start + query_batch_size, len(queries))
            vectors = float_batch(queries, start, end)
            ids = np.arange(start, end, dtype=np.int64)
            test_writer.write_table(vector_table(ids, vectors))
            if index is not None:
                _, neighbors = index.search(vectors, k)
            else:
                neighbors = np.asarray(supplied_gt[start:end])
            if not np.issubdtype(neighbors.dtype, np.integer):
                raise ValueError("Ground truth IDs must be integers")
            if (neighbors < 0).any() or (neighbors >= len(base)).any():
                raise ValueError("Ground truth IDs are outside the selected base")
            if any(len(set(row)) != k for row in neighbors.tolist()):
                raise ValueError("Ground truth contains duplicate neighbor IDs")
            gt_writer.write_table(
                pa.table(
                    {"id": ids, "neighbors_id": neighbors.tolist()}, schema=gt_schema
                )
            )
            LOG.info("%s GT: %d/%d queries", metric, end, len(queries))


def cache_valid(destination: Path, config: dict) -> bool:
    try:
        manifest = json.loads((destination / "manifest.json").read_text())
        return manifest["config"] == config and all(
            sha256(destination / name) == manifest["outputs"][name] for name in FILES
        )
    except (OSError, ValueError, KeyError):
        return False


def prepare_one(
    name: str, root: Path, raw: Path, *, recompute_gt: bool, force: bool
) -> None:
    gist = name == "gist"
    dimension, query_count, metric = (960, 1000, "L2") if gist else (200, 10_000, "IP")
    sources = (
        [GIST_URL]
        if gist
        else [T2I_ROOT + "base.1B.fbin", T2I_ROOT + "query.public.100K.fbin"]
    )
    config = {
        "version": VERSION,
        "name": name,
        "base_count": BASE_COUNT,
        "query_count": query_count,
        "dimension": dimension,
        "metric": metric,
        "k": K,
        "selection": "first rows in published file order; no normalization",
        "gt": "published" if gist and not recompute_gt else "faiss-flat-float32",
        "sources": sources,
    }
    destination = root / name / f"{name}_1m_v1"
    if not force and cache_valid(destination, config):
        LOG.info("Verified cached dataset: %s", destination)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.ExitStack() as stack:
        if gist:
            path = raw / "gist-960-euclidean.hdf5"
            source_metadata = [download(sources[0], path)]
            data = stack.enter_context(h5py.File(path, "r"))
            distance = data.attrs.get("distance")
            if distance not in ("euclidean", b"euclidean"):
                raise ValueError("GIST source is not an euclidean dataset")
            base, queries = data["train"], data["test"]
            gt = None if recompute_gt else data["neighbors"][:, :K]
        else:
            base_path, query_path = (
                raw / "text2image-base-1m.fbin",
                raw / "text2image-query-10k.fbin",
            )
            source_metadata = [
                download(
                    sources[0], base_path, prefix_bytes=8 + BASE_COUNT * dimension * 4
                ),
                download(
                    sources[1], query_path, prefix_bytes=8 + query_count * dimension * 4
                ),
            ]
            base = read_fbin_prefix(base_path, BASE_COUNT, dimension)
            queries = read_fbin_prefix(query_path, query_count, dimension)
            gt = None
        if base.shape != (BASE_COUNT, dimension) or queries.shape != (
            query_count,
            dimension,
        ):
            raise ValueError(
                f"Unexpected dataset shapes: {base.shape}, {queries.shape}"
            )
        staging = Path(
            stack.enter_context(tempfile.TemporaryDirectory(dir=destination.parent))
        )
        prepare_arrays(staging, base, queries, metric=metric, k=K, supplied_gt=gt)
        write_json(
            staging / "manifest.json",
            {
                "config": config,
                "sources": source_metadata,
                "faiss_version": faiss.__version__,
                "outputs": {filename: sha256(staging / filename) for filename in FILES},
            },
        )
        # Publish the completion marker last. A failed rebuild never validates as complete.
        destination.mkdir(exist_ok=True)
        (destination / "manifest.json").unlink(missing_ok=True)
        for filename in (*FILES, "manifest.json"):
            shutil.move(staging / filename, destination / filename)
    LOG.info("Prepared %s", destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=["all", "gist", "text2image"], default="all"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            os.environ.get("DATASET_LOCAL_DIR", "/tmp/vectordb_bench/dataset")
        ),
    )
    parser.add_argument("--raw-dir", type=Path, help="Default: ROOT/_sources")
    parser.add_argument("--threads", type=int, default=min(os.cpu_count() or 1, 16))
    parser.add_argument(
        "--recompute-gt",
        action="store_true",
        help="Also regenerate GIST GT with exact L2 search",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild Parquet/GT; keep verified raw downloads",
    )
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("--threads must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    faiss.omp_set_num_threads(args.threads)
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / ".prepare-datasets.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        names = ("gist", "text2image") if args.dataset == "all" else (args.dataset,)
        for name in names:
            prepare_one(
                name,
                args.root,
                args.raw_dir or args.root / "_sources",
                recompute_gt=args.recompute_gt,
                force=args.force,
            )


if __name__ == "__main__":
    main()
