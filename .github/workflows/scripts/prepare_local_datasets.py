"""Convert local TexMex GIST and Big ANN Text2Image files to VectorDBBench Parquet."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import logging
import shutil
import struct
import tempfile
from pathlib import Path

import faiss
import numpy as np
import prepare_datasets as prep
from download_text2image import save

LOG = logging.getLogger(__name__)


def read_vecs(path, dtype, dimension):
    size = path.stat().st_size
    if size == 0 or size % (4 * (dimension + 1)):
        raise ValueError(f"Invalid vecs size: {path}")
    raw = np.memmap(path, dtype="<i4", mode="r").reshape(-1, dimension + 1)
    if not np.all(raw[:, 0] == dimension):
        raise ValueError(f"Inconsistent vecs dimension: {path}")
    return raw[:, 1:].view(dtype)


def read_gt(path, rows, k, base_count):
    with path.open("rb") as f:
        header = struct.unpack("<II", f.read(8))
    if header != (rows, k) or path.stat().st_size != 8 + rows * k * 8:
        raise ValueError(f"Invalid Big ANN GT shape: {path}")
    ids = np.memmap(path, dtype="<i4", mode="r", offset=8, shape=(rows, k))
    scores = np.memmap(
        path, dtype="<f4", mode="r", offset=8 + rows * k * 4, shape=(rows, k)
    )
    if (ids < 0).any() or (ids >= base_count).any() or not np.isfinite(scores).all():
        raise ValueError("Invalid GT IDs/scores")
    return ids, scores


def exact_ip_gt(path, base, queries, k=100, batch_size=512):
    """Resumable exact search; official 10M GT cannot be filtered into 1M GT."""
    config = {
        "base_count": len(base),
        "queries": len(queries),
        "dimension": base.shape[1],
        "k": k,
        "base_sha256": hashlib.sha256(np.ascontiguousarray(base)).hexdigest(),
        "queries_sha256": hashlib.sha256(np.ascontiguousarray(queries)).hexdigest(),
    }
    metadata_path = path.with_name(path.name + ".json")
    if path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("config") == config and metadata.get("sha256") == prep.sha256(
            path
        ):
            return read_gt(path, len(queries), k, len(base))[0]
    partial = path.with_name(path.name + ".part")
    checkpoint = path.with_name(path.name + ".progress.json")
    done = 0
    if partial.exists() and checkpoint.exists():
        state = json.loads(checkpoint.read_text())
        if state.get("config") == config:
            done = state["completed"]
    size = 8 + len(queries) * k * 8
    if not done:
        with partial.open("wb") as f:
            f.write(struct.pack("<II", len(queries), k))
            f.truncate(size)
    ids = np.memmap(partial, dtype="<i4", mode="r+", offset=8, shape=(len(queries), k))
    scores = np.memmap(
        partial,
        dtype="<f4",
        mode="r+",
        offset=8 + len(queries) * k * 4,
        shape=(len(queries), k),
    )
    index = faiss.IndexFlatIP(base.shape[1])
    for start in range(0, len(base), 8192):
        index.add(prep.float_batch(base, start, min(start + 8192, len(base))))
    for start in range(done, len(queries), batch_size):
        end = min(start + batch_size, len(queries))
        distances, neighbors = index.search(prep.float_batch(queries, start, end), k)
        ids[start:end], scores[start:end] = neighbors, distances
        ids.flush()
        scores.flush()
        save(checkpoint, {"config": config, "completed": end})
        LOG.info("Exact 1M IP GT: %d/%d queries", end, len(queries))
    del ids, scores, index
    partial.replace(path)
    prep.write_json(
        metadata_path,
        {"config": config, "method": "faiss.IndexFlatIP", "sha256": prep.sha256(path)},
    )
    checkpoint.unlink(missing_ok=True)
    return read_gt(path, len(queries), k, len(base))[0]


def convert(root, name, directory, base, queries, gt, metric, sources, method):
    config = {
        "version": 1,
        "name": name,
        "base_count": len(base),
        "query_count": len(queries),
        "dimension": base.shape[1],
        "metric": metric,
        "k": 100,
        "selection": "published row order; zero-based IDs; no normalization",
        "gt": method,
    }
    destination = root / name / directory
    source_metadata = [
        {"path": str(p.resolve()), "bytes": p.stat().st_size, "sha256": prep.sha256(p)}
        for p in sources
    ]
    config["source_sha256"] = [s["sha256"] for s in source_metadata]
    if prep.cache_valid(destination, config):
        LOG.info("Verified prepared cache: %s", destination)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        staging = Path(temporary)
        prep.prepare_arrays(
            staging,
            base,
            queries,
            metric=metric,
            k=100,
            supplied_gt=gt,
            query_batch_size=1024,
        )
        prep.write_json(
            staging / "manifest.json",
            {
                "config": config,
                "sources": source_metadata,
                "faiss_version": faiss.__version__,
                "outputs": {n: prep.sha256(staging / n) for n in prep.FILES},
            },
        )
        destination.mkdir(exist_ok=True)
        (destination / "manifest.json").unlink(missing_ok=True)
        for filename in (*prep.FILES, "manifest.json"):
            shutil.move(staging / filename, destination / filename)
    LOG.info("PREPARED %s", destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, help="Default ROOT/_sources")
    parser.add_argument(
        "--dataset",
        choices=["all", "gist", "text2image", "text2image-1m", "text2image-10m"],
        default="all",
    )
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.threads < 1:
        parser.error("--threads must be positive")
    faiss.omp_set_num_threads(args.threads)
    raw = args.raw_dir or args.root / "_sources"
    args.root.mkdir(parents=True, exist_ok=True)
    with contextlib.ExitStack() as stack:
        lock = stack.enter_context((args.root / ".prepare-local.lock").open("w"))
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.dataset in ("all", "gist"):
            paths = [
                raw / "gist" / f"gist_{n}.{ext}"
                for n, ext in [
                    ("base", "fvecs"),
                    ("query", "fvecs"),
                    ("groundtruth", "ivecs"),
                ]
            ]
            base, queries, gt = [
                read_vecs(p, dtype, d)
                for p, dtype, d in zip(
                    paths, ["<f4", "<f4", "<i4"], [960, 960, 100], strict=True
                )
            ]
            if (
                base.shape != (1_000_000, 960)
                or queries.shape != (1000, 960)
                or gt.shape != (1000, 100)
            ):
                raise ValueError("Unexpected GIST shapes")
            convert(
                args.root,
                "gist",
                "gist_texmex_1m_v1",
                base,
                queries,
                gt,
                "L2",
                paths,
                "TexMex published top100",
            )
        if args.dataset in ("all", "text2image", "text2image-1m", "text2image-10m"):
            t2i = raw / "text2image1B"
            query_path = t2i / "query.public.100K.fbin"
            queries = prep.read_fbin_prefix(query_path, 100_000, 200)
            for count in (1_000_000, 10_000_000):
                if args.dataset == "text2image-1m" and count != 1_000_000:
                    continue
                if args.dataset == "text2image-10m" and count != 10_000_000:
                    continue
                base_path = t2i / f"base.1B.fbin.crop_nb_{count}"
                base = prep.read_fbin_prefix(base_path, count, 200)
                gt_path = t2i / (
                    "text2image-1M" if count == 1_000_000 else "text2image-10M"
                )
                gt = (
                    exact_ip_gt(gt_path, base, queries)
                    if count == 1_000_000
                    else read_gt(gt_path, 100_000, 100, count)[0]
                )
                convert(
                    args.root,
                    "text2image",
                    f"text2image_{count // 1_000_000}m_public100k_v1",
                    base,
                    queries,
                    gt,
                    "IP",
                    [base_path, query_path, gt_path],
                    "faiss.IndexFlatIP exact"
                    if count == 1_000_000
                    else "Big ANN published top100",
                )


if __name__ == "__main__":
    main()
