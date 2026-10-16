"""Download VectorDBBench's Cohere 1M/10M evaluation files without conversion.

Uses the checked, resumable range downloader shared with Text2Image. Original
Parquet names, vector values, IDs and shuffled insertion order are preserved.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
import logging
import re
import subprocess
from pathlib import Path

import pyarrow.parquet as pq
from download_text2image import download, save

LOG = logging.getLogger(__name__)
SOURCES = {
    "AliyunOSS": "https://assets.zilliz.com.cn/benchmark",
    "S3": "https://assets.zilliz.com/benchmark",
}
DATASETS = {
    "1m": ("cohere_medium_1m", 1_000_000, ["shuffle_train.parquet"]),
    "10m": (
        "cohere_large_10m",
        10_000_000,
        [f"shuffle_train-{i:02d}-of-10.parquet" for i in range(10)],
    ),
}


def prepare(root: Path, source: str, selection: str, workers: int, parallel_files: int):
    for key, (directory, rows, train_files) in DATASETS.items():
        if selection not in ("all", key):
            continue
        target = root / "cohere" / directory
        target.mkdir(parents=True, exist_ok=True)
        files = [
            "test.parquet",
            "neighbors.parquet",
            "scalar_labels.parquet",
            *train_files,
        ]

        def fetch_file(name, directory=directory, target=target):
            url = f"{SOURCES[source]}/{directory}/{name}"
            headers = subprocess.run(
                ["curl", "-fsSI", "--retry", "3", "--max-time", "30", url],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            size = int(re.findall(r"(?im)^content-length:\s*(\d+)", headers)[-1])
            # Multiple files share the link; do not restart healthy slower streams.
            download(target, (name, url, size, None), workers, speed_limit=65536)
            parquet = pq.ParquetFile(target / name)
            metadata = json.loads((target / (name + ".json")).read_text())
            metadata.update(
                name=name,
                rows=parquet.metadata.num_rows,
                columns=parquet.schema_arrow.names,
            )
            LOG.info("PARQUET %s rows=%d", name, parquet.metadata.num_rows)
            return metadata

        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_files) as pool:
            artifacts = list(pool.map(fetch_file, files))
        actual_rows = sum(
            item["rows"] for item in artifacts if item["name"] in train_files
        )
        if actual_rows != rows:
            raise ValueError(f"Unexpected base size: {actual_rows}, expected {rows}")
        queries, neighbors = artifacts[:2]
        if queries["rows"] != neighbors["rows"]:
            raise ValueError("Query/GT row count mismatch")
        save(
            target / "manifest.json",
            {
                "dataset": "Cohere",
                "size": rows,
                "dimension": 768,
                "metric": "COSINE",
                "source": source,
                "directory": directory,
                "train_files": train_files,
                "query_count": queries["rows"],
                "use_shuffled": True,
                "conversion": "none; original VectorDBBench Parquet files",
                "files": artifacts,
            },
        )
        LOG.info("READY %s", directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", choices=["all", *DATASETS], default="all")
    parser.add_argument("--source", choices=SOURCES, default="AliyunOSS")
    parser.add_argument("--workers", type=int, default=8, help="Range workers per file")
    parser.add_argument("--parallel-files", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1 or args.parallel_files < 1:
        parser.error("--workers and --parallel-files must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".download-cohere.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        prepare(root, args.source, args.dataset, args.workers, args.parallel_files)
    LOG.info("ALL REQUESTED COHERE DATASETS DOWNLOADED")


if __name__ == "__main__":
    main()
