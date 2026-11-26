"""Download bounded Big ANN Text2Image prefixes with resumable, checked ranges."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import logging
import os
import re
import struct
import subprocess
import tempfile
import time
from pathlib import Path

REMOTE = "https://yandex-research.storage.yandexcloud.net/ann-datasets/T2I/"
LOG = logging.getLogger(__name__)
CHUNK = 8 * 1024 * 1024
FILES = [
    (
        "base.1B.fbin.crop_nb_10000000",
        REMOTE + "base.1B.fbin",
        8_000_000_008,
        (1_000_000_000, 200),
    ),
    (
        "query.public.100K.fbin",
        REMOTE + "query.public.100K.fbin",
        80_000_008,
        (100_000, 200),
    ),
    (
        "text2image-10M",
        "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_10M/text2image-10M",
        80_000_008,
        (100_000, 100),
    ),
]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def save(path, obj):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2) + "\n")
    temporary.replace(path)


def download(root, entry, workers, *, speed_limit=262144):  # noqa: PLR0915
    name, url, size, expected_header = entry
    final = root / name
    metadata_path = root / (name + ".json")
    if final.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if (
            metadata.get("url") == url
            and final.stat().st_size == size
            and sha256(final) == metadata.get("sha256")
        ):
            LOG.info("CACHED %s", name)
            return
    head = subprocess.run(
        ["curl", "-fsSI", "--retry", "3", "--max-time", "30", url],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    etag = re.findall(r"(?im)^etag:\s*(.*?)\s*$", head)[-1]
    total = int(re.findall(r"(?im)^content-length:\s*(\d+)", head)[-1])
    if total < size:
        raise ValueError("Source is too short")
    state_path = root / (name + ".progress.json")
    partial = root / (name + ".part")
    state = {"url": url, "etag": etag, "size": size, "chunk_size": CHUNK, "chunks": {}}
    if partial.exists() and state_path.exists():
        old = json.loads(state_path.read_text())
        if all(old.get(k) == state[k] for k in ("url", "etag", "size", "chunk_size")):
            state = old
    fd = os.open(partial, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        os.ftruncate(fd, size)
        for offset, checksum in list(state["chunks"].items()):
            start = int(offset)
            if (
                hashlib.sha256(
                    os.pread(fd, min(CHUNK, size - start), start)
                ).hexdigest()
                != checksum
            ):
                del state["chunks"][offset]
        pending = [
            start
            for start in range(0, size, CHUNK)
            if str(start) not in state["chunks"]
        ]
        complete = size - sum(min(CHUNK, size - start) for start in pending)
        initial, started = complete, time.monotonic()
        LOG.info("START %s %d/%d", name, complete, size)

        def fetch(start):
            end = min(start + CHUNK, size) - 1
            cursor = start
            with tempfile.TemporaryDirectory(dir=root) as tmp:
                data_path, headers_path = Path(tmp) / "data", Path(tmp) / "headers"
                for attempt in range(64):
                    data_path.unlink(missing_ok=True)
                    result = subprocess.run(
                        [
                            "curl",
                            "-fsS",
                            "--connect-timeout",
                            "20",
                            "--max-time",
                            "120",
                            "--speed-limit",
                            str(speed_limit),
                            "--speed-time",
                            "10",
                            "--max-filesize",
                            str(end - cursor + 1),
                            "--range",
                            f"{cursor}-{end}",
                            "--header",
                            f"If-Match: {etag}",
                            "--dump-header",
                            str(headers_path),
                            "--output",
                            str(data_path),
                            url + f"?bench-range={cursor}",
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    # A timed-out response can still contain a valid range prefix.
                    # Keep those bytes and request only the remainder on retry.
                    if result.returncode in (0, 18, 28) and data_path.exists():
                        headers = headers_path.read_text().lower()
                        if (
                            f"content-range: bytes {cursor}-{end}/{total}"
                            not in headers
                        ):
                            raise ValueError("Incorrect Content-Range")
                        data = data_path.read_bytes()
                        if len(data) > end - cursor + 1 or (
                            result.returncode == 0 and len(data) != end - cursor + 1
                        ):
                            raise ValueError("Truncated range")
                        view, offset = memoryview(data), cursor
                        while view:
                            n = os.pwrite(fd, view, offset)
                            if n <= 0:
                                raise OSError("Short write")
                            view, offset = view[n:], offset + n
                        cursor += len(data)
                        if cursor == end + 1:
                            data = os.pread(fd, end - start + 1, start)
                            return start, len(data), hashlib.sha256(data).hexdigest()
                    if attempt % 8 == 0:
                        LOG.warning(
                            "RETRY %s %d %s", name, cursor, result.stderr.strip()
                        )
                    time.sleep(1)
                raise RuntimeError(f"Download failed at {name}:{start}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for future in concurrent.futures.as_completed(
                [pool.submit(fetch, start) for start in pending]
            ):
                start, length, checksum = future.result()
                state["chunks"][str(start)] = checksum
                complete += length
                save(state_path, state)
                if len(state["chunks"]) % 8 == 0 or complete == size:
                    speed = (
                        (complete - initial)
                        / max(time.monotonic() - started, 0.001)
                        / 1e6
                    )
                    LOG.info(
                        "PROGRESS %s: %.1f%% %s/%s %.2f MB/s",
                        name,
                        100 * complete / size,
                        complete,
                        size,
                        speed,
                    )
        if (
            expected_header is not None
            and struct.unpack("<II", os.pread(fd, 8, 0)) != expected_header
        ):
            raise ValueError("Unexpected source header")
        if expected_header is not None and name.startswith("base."):
            os.pwrite(fd, struct.pack("<I", 10_000_000), 0)
        os.fsync(fd)
    finally:
        os.close(fd)
    if (
        name == "text2image-10M"
        and hashlib.md5(partial.read_bytes()).hexdigest()
        != "424b34cfde4fd418d466a7ebceb7df1c"
    ):
        raise ValueError("Official GT MD5 mismatch")
    partial.replace(final)
    save(
        metadata_path,
        {"url": url, "etag": etag, "bytes": size, "sha256": sha256(final)},
    )
    state_path.unlink(missing_ok=True)
    LOG.info("DONE %s", name)


def prepare(root, workers):
    # Complete the small files first; no training/private data is downloaded.
    for entry in [FILES[1], FILES[2], FILES[0]]:
        download(root, entry, workers)
    source = root / FILES[0][0]
    target = root / "base.1B.fbin.crop_nb_1000000"
    partial = target.with_name(target.name + ".part")
    with source.open("rb") as src, partial.open("wb") as dst:
        src.seek(8)
        dst.write(struct.pack("<II", 1_000_000, 200))
        remaining = 800_000_000
        while remaining:
            data = src.read(min(CHUNK, remaining))
            if not data:
                raise ValueError("Truncated 10M source")
            dst.write(data)
            remaining -= len(data)
    partial.replace(target)
    save(
        target.with_name(target.name + ".json"),
        {
            "derived_from": source.name,
            "selection": "first 1M rows",
            "sha256": sha256(target),
        },
    )
    LOG.info("ALL DOWNLOADS COMPLETE")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.workers < 1:
        parser.error("--workers must be positive")
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / ".download.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare(args.root, args.workers)


if __name__ == "__main__":
    main()
