#!/usr/bin/env python3
r"""Run the DiskANN benchmark matrix on Windows.

The defaults intentionally match the benchmark matrix documented in macos.md:

* FP32 and FP16 indexes
* list_size: 100, 300, 500
* query threads: 1, 2, 4
* 30 seconds per throughput run
* max degree 32, builder list size 50, PQ chunks 384
* 10,000 cached nodes

The script drives the repository's native C++ benchmark tools so the results
are comparable with macos.md. It creates all YAML files and writes raw logs,
CSV/JSON data, environment metadata, and a Markdown summary under
diskann_bench_windows/<timestamp>_<git-sha>.

Run from an "x64 Native Tools Command Prompt for VS 2022", with the project's
virtual environment activated:

    python scripts\benchmark_diskann_windows.py ^
      --train-file C:\data\cohere_train_vector_1m.new.centaur.vecs ^
      --query-file C:\data\cohere_test_vector_1000.new.txt

If --ground-truth-file is omitted, the script generates an exact external
ground-truth file with blocked NumPy matrix multiplication and reuses it for
all recall runs. Pass --ground-truth-mode internal only to use
recall_original's slower built-in linear scan.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import ctypes.wintypes
import datetime as dt
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


MIB = 1024 * 1024
GIB = 1024 * 1024 * 1024
IS_WINDOWS = sys.platform == "win32"
UINT64_MAX = (1 << 64) - 1
VECS_HEADER = struct.Struct("<QHHI11Q")


@dataclass
class ProcessMetrics:
    wall_seconds: float = 0.0
    peak_working_set_mb: float | None = None
    read_operations: int | None = None
    read_megabytes: float | None = None
    read_iops: float | None = None
    read_mb_per_second: float | None = None


@dataclass
class BuildResult:
    precision: str
    index_path: str
    index_size_gib: float | None
    train_seconds: float | None
    build_seconds: float | None
    dump_seconds: float | None
    wall_seconds: float | None
    log_path: str


@dataclass
class SearchResult:
    precision: str
    list_size: int
    threads: int
    recall_at_1: float | None
    recall_at_10: float | None
    recall_at_50: float | None
    qps: float | None
    avg_latency_ms: float | None
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    p99_latency_ms: float | None
    min_latency_ms: float | None
    max_latency_ms: float | None
    query_count: int | None
    peak_working_set_mb: float | None
    process_read_iops: float | None
    process_read_mb_per_second: float | None
    reads_per_query: float | None
    recall_log_path: str
    bench_log_path: str


@dataclass(frozen=True)
class VecsLayout:
    num_vecs: int
    dimension: int
    data_offset: int
    dense_offset: int
    key_offset: int


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.wintypes.DWORD),
        ("PageFaultCount", ctypes.wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def discover_repo_root() -> Path:
    """Find the repository even if this script was copied to another level."""

    script_path = Path(__file__).resolve()
    candidates = [Path.cwd().resolve(), script_path.parent, *script_path.parents]
    visited: set[Path] = set()
    for candidate in candidates:
        if candidate in visited:
            continue
        visited.add(candidate)
        if (candidate / "CMakeLists.txt").is_file() and (
            candidate / "pyproject.toml"
        ).is_file():
            return candidate
    return Path.cwd().resolve()


def parse_args() -> argparse.Namespace:
    repo_default = discover_repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "Run the Windows DiskANN build/recall/QPS matrix using the native "
            "zvec benchmark tools."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--ground-truth-file", type=Path)
    parser.add_argument(
        "--ground-truth-mode",
        choices=("auto", "internal"),
        default="auto",
        help=(
            "When --ground-truth-file is omitted, 'auto' generates an "
            "external neighbors file with blocked NumPy matrix multiplication; "
            "'internal' uses recall_original's much slower per-query scan."
        ),
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=768,
        help="Dense vector dimension used by the train/query dataset.",
    )
    parser.add_argument(
        "--ground-truth-k",
        type=int,
        default=100,
        help="Exact neighbors generated for each query.",
    )
    parser.add_argument(
        "--ground-truth-block-size",
        type=int,
        default=8192,
        help="Database vectors processed per exact-search matrix block.",
    )
    parser.add_argument("--repo-root", type=Path, default=repo_default)
    parser.add_argument(
        "--build-dir",
        type=Path,
        help="CMake build directory; defaults to <repo-root>/build.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Result directory; defaults to "
            "<repo-root>/diskann_bench_windows/<timestamp>_<git-sha>."
        ),
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        help="Index directory; defaults to <output-dir>/indexes.",
    )
    parser.add_argument(
        "--precision",
        nargs="+",
        choices=("fp32", "fp16"),
        default=("fp32", "fp16"),
    )
    parser.add_argument(
        "--list-sizes", nargs="+", type=int, default=(100, 300, 500)
    )
    parser.add_argument(
        "--thread-counts", nargs="+", type=int, default=(1, 2, 4)
    )
    parser.add_argument("--build-threads", type=int, default=8)
    parser.add_argument("--recall-threads", type=int, default=16)
    parser.add_argument("--bench-seconds", type=int, default=30)
    parser.add_argument("--bench-iterations", type=int, default=10_000_000)
    parser.add_argument("--cache-nodes", type=int, default=10_000)
    parser.add_argument(
        "--beam-size",
        type=int,
        default=2,
        help=(
            "DiskANN search beam size. Keep 2 for baseline comparison; try "
            "4, 8, and 16 to increase overlapped I/O depth on Windows."
        ),
    )
    parser.add_argument("--max-degree", type=int, default=32)
    parser.add_argument("--builder-list-size", type=int, default=50)
    parser.add_argument("--pq-chunks", type=int, default=384)
    parser.add_argument("--memory-limit", type=float, default=100.0)
    parser.add_argument("--top-k", default="1,10,50")
    parser.add_argument(
        "--parallel-builds",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Parallel jobs used when building benchmark tools.",
    )
    parser.add_argument(
        "--skip-tool-build",
        action="store_true",
        help="Require existing benchmark executables instead of building them.",
    )
    parser.add_argument("--skip-index-build", action="store_true")
    parser.add_argument("--skip-recall", action="store_true")
    parser.add_argument("--skip-bench", action="store_true")
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Overwrite indexes that already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create configs and print commands without executing them.",
    )
    return parser.parse_args()


def require_positive(values: Iterable[int], label: str) -> None:
    invalid = [value for value in values if value <= 0]
    if invalid:
        raise ValueError(f"{label} must contain positive integers: {invalid}")


def resolved(path: Path, base: Path | None = None) -> Path:
    if not path.is_absolute() and base is not None:
        path = base / path
    return path.expanduser().resolve()


def git_sha(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=8", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def append_msys2_to_path(env: dict[str, str]) -> None:
    """Append MSYS2 paths so Snowball can find make, Perl, and GCC.

    They are appended, not prepended, to avoid shadowing MSVC's link.exe.
    """

    candidates = (
        Path(r"C:\msys64\ucrt64\bin"),
        Path(r"C:\msys64\usr\bin"),
    )
    current_parts = env.get("PATH", "").split(os.pathsep)
    normalized = {os.path.normcase(part) for part in current_parts if part}
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate.is_dir() and os.path.normcase(candidate_str) not in normalized:
            current_parts.append(candidate_str)
    env["PATH"] = os.pathsep.join(current_parts)


def executable(name: str, repo_root: Path, env: dict[str, str]) -> str:
    found = shutil.which(name, path=env.get("PATH"))
    if found:
        return found
    suffix = ".exe" if IS_WINDOWS else ""
    candidate = repo_root / ".venv" / "Scripts" / f"{name}{suffix}"
    if candidate.is_file():
        return str(candidate)
    raise FileNotFoundError(
        f"Cannot find {name}. Activate .venv and install the build dependencies."
    )


def command_text(command: Sequence[str | Path]) -> str:
    return subprocess.list2cmdline([str(part) for part in command])


def run_simple(
    command: Sequence[str | Path],
    *,
    cwd: Path,
    env: dict[str, str],
    dry_run: bool,
) -> None:
    print(f"\n> {command_text(command)}")
    if dry_run:
        return
    subprocess.run([str(part) for part in command], cwd=cwd, env=env, check=True)


def check_x64_msvc(env: dict[str, str]) -> None:
    try:
        result = subprocess.run(
            ["cl"],
            env=env,
            capture_output=True,
            text=True,
            errors="replace",
        )
    except OSError as exc:
        raise RuntimeError(
            "MSVC cl.exe was not found. Run this script from "
            "'x64 Native Tools Command Prompt for VS 2022'."
        ) from exc
    output = f"{result.stdout}\n{result.stderr}"
    if not re.search(r"\bfor x64\b", output, re.IGNORECASE):
        raise RuntimeError(
            "The active MSVC compiler is not x64. Close this terminal and use "
            "'x64 Native Tools Command Prompt for VS 2022'."
        )


def find_tool(build_dir: Path, name: str) -> Path | None:
    filename = f"{name}.exe" if IS_WINDOWS else name
    preferred = (
        build_dir / "bin" / filename,
        build_dir / "bin" / "Release" / filename,
        build_dir / "Release" / filename,
    )
    for candidate in preferred:
        if candidate.is_file():
            return candidate.resolve()
    if build_dir.is_dir():
        for candidate in build_dir.rglob(filename):
            if "CMakeFiles" not in candidate.parts and candidate.is_file():
                return candidate.resolve()
    return None


def ensure_tools(
    repo_root: Path,
    build_dir: Path,
    env: dict[str, str],
    *,
    skip_build: bool,
    parallel_builds: int,
    dry_run: bool,
) -> dict[str, Path]:
    names = ("local_builder_original", "recall_original", "bench_original")
    tools = {name: find_tool(build_dir, name) for name in names}
    if all(tools.values()):
        return {name: path for name, path in tools.items() if path is not None}
    if skip_build:
        missing = [name for name, path in tools.items() if path is None]
        raise FileNotFoundError(
            f"Missing benchmark tools in {build_dir}: {', '.join(missing)}"
        )

    if IS_WINDOWS and not dry_run:
        check_x64_msvc(env)
    cmake = executable("cmake", repo_root, env)
    configure_command: list[str | Path] = [
        cmake,
        "-S",
        repo_root,
        "-B",
        build_dir,
    ]
    if not (build_dir / "CMakeCache.txt").exists():
        configure_command.extend(["-G", "Ninja"])
    configure_command.extend(
        [
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_TOOLS=ON",
            "-DBUILD_PYTHON_BINDINGS=OFF",
            "-DBUILD_C_BINDINGS=OFF",
            "-DBUILD_ZVEC_SHARED=OFF",
            "-DBUILD_ZVEC_CORE_SHARED=OFF",
            "-DBUILD_ZVEC_AILEGO_SHARED=OFF",
        ]
    )
    run_simple(
        configure_command, cwd=repo_root, env=env, dry_run=dry_run
    )
    run_simple(
        [
            cmake,
            "--build",
            build_dir,
            "--target",
            *names,
            "--config",
            "Release",
            "--parallel",
            str(parallel_builds),
        ],
        cwd=repo_root,
        env=env,
        dry_run=dry_run,
    )
    if dry_run:
        return {
            name: (build_dir / "bin" / f"{name}.exe").resolve() for name in names
        }
    tools = {name: find_tool(build_dir, name) for name in names}
    missing = [name for name, path in tools.items() if path is None]
    if missing:
        raise FileNotFoundError(
            f"Build completed but tools were not found: {', '.join(missing)}"
        )
    return {name: path for name, path in tools.items() if path is not None}


def yaml_path(path: Path) -> str:
    value = str(path.resolve()).replace("\\", "/").replace("'", "''")
    return f"'{value}'"


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def read_vecs_layout(train_file: Path, dimension: int) -> VecsLayout:
    """Read the offsets needed for dense FP32 vectors and uint64 keys."""

    with train_file.open("rb") as stream:
        raw_header = stream.read(VECS_HEADER.size)
    if len(raw_header) != VECS_HEADER.size:
        raise ValueError(
            f"Training file is too small for a VecsHeader: {train_file}"
        )

    (
        num_vecs,
        _meta_size_v1,
        version,
        meta_size,
        bitmap,
        key_offset,
        key_size,
        dense_offset,
        dense_size,
        _sparse_offset,
        _sparse_size,
        _partition_offset,
        _partition_size,
        _taglist_offset,
        _taglist_size,
    ) = VECS_HEADER.unpack(raw_header)

    if version != 1:
        raise ValueError(
            "Automatic ground-truth generation currently requires a version "
            f"1 .vecs file; found version {version}. Supply "
            "--ground-truth-file or use --ground-truth-mode internal."
        )
    if num_vecs == 0:
        raise ValueError(f"Training file contains no vectors: {train_file}")
    if not bitmap & (1 << 1) or dense_offset == UINT64_MAX:
        raise ValueError("Training file does not contain dense vectors.")
    if not bitmap & (1 << 0) or key_offset == UINT64_MAX:
        raise ValueError("Training file does not contain uint64 vector keys.")

    dense_row_bytes, remainder = divmod(dense_size, num_vecs)
    if remainder or dense_row_bytes % 4:
        raise ValueError(
            "Dense vector section is not a contiguous FP32 matrix."
        )
    stored_dimension = dense_row_bytes // 4
    if stored_dimension != dimension:
        raise ValueError(
            f"--dimension is {dimension}, but the training file stores "
            f"{stored_dimension} FP32 values per vector."
        )
    expected_key_size = num_vecs * 8
    if key_size < expected_key_size:
        raise ValueError(
            f"Key section is too small: expected at least "
            f"{expected_key_size} bytes, found {key_size}."
        )

    data_offset = VECS_HEADER.size + meta_size
    file_size = train_file.stat().st_size
    dense_end = data_offset + dense_offset + dense_size
    key_end = data_offset + key_offset + expected_key_size
    if max(dense_end, key_end) > file_size:
        raise ValueError(
            "Training file header points beyond the end of the file."
        )

    return VecsLayout(
        num_vecs=num_vecs,
        dimension=stored_dimension,
        data_offset=data_offset,
        dense_offset=dense_offset,
        key_offset=key_offset,
    )


def load_query_matrix(
    query_file: Path, dimension: int, np: Any
) -> tuple[list[int], Any]:
    """Load the dense query field using recall_original's text convention."""

    query_ids: list[int] = []
    query_vectors: list[Any] = []
    with query_file.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split(";")
            if len(fields) < 2:
                raise ValueError(
                    f"{query_file}:{line_number}: expected "
                    "'query_id;dense vector'."
                )
            try:
                query_id = int(fields[0])
            except ValueError as exc:
                raise ValueError(
                    f"{query_file}:{line_number}: invalid query id "
                    f"{fields[0]!r}."
                ) from exc
            vector = np.fromstring(fields[1], dtype=np.float32, sep=" ")
            if vector.size != dimension:
                raise ValueError(
                    f"{query_file}:{line_number}: expected {dimension} "
                    f"dense values, found {vector.size}."
                )
            query_ids.append(query_id)
            query_vectors.append(vector)

    if not query_vectors:
        raise ValueError(f"Query file contains no vectors: {query_file}")
    return query_ids, np.stack(query_vectors)


def display_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, remaining = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{remaining:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def generate_external_ground_truth(
    *,
    train_file: Path,
    query_file: Path,
    output_file: Path,
    dimension: int,
    neighbor_count: int,
    block_size: int,
    dry_run: bool,
) -> None:
    """Generate exact cosine neighbors with bounded-memory matrix blocks."""

    if output_file.is_file():
        print(f"\nReusing ground truth: {output_file}")
        return
    if dry_run:
        print(
            "\nWould generate exact ground truth with NumPy: "
            f"{output_file} (dimension={dimension}, k={neighbor_count}, "
            f"block_size={block_size})"
        )
        return

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "NumPy is required for automatic ground-truth generation. "
            "Activate the project .venv and run 'python -m pip install "
            "numpy', or supply --ground-truth-file."
        ) from exc

    layout = read_vecs_layout(train_file, dimension)
    if neighbor_count > layout.num_vecs:
        raise ValueError(
            f"--ground-truth-k ({neighbor_count}) exceeds the number of "
            f"training vectors ({layout.num_vecs})."
        )

    query_ids, queries = load_query_matrix(query_file, dimension, np)
    query_norms = np.sqrt(
        np.einsum("ij,ij->i", queries, queries, optimize=True)
    )
    if np.any(~np.isfinite(query_norms)) or np.any(query_norms == 0):
        raise ValueError("Query file contains a zero or non-finite vector.")
    queries /= query_norms[:, np.newaxis]
    query_transpose = np.ascontiguousarray(queries.T)
    query_count = len(query_ids)
    query_columns = np.arange(query_count)[np.newaxis, :]

    vectors = np.memmap(
        train_file,
        mode="r",
        dtype="<f4",
        offset=layout.data_offset + layout.dense_offset,
        shape=(layout.num_vecs, layout.dimension),
    )
    keys = np.memmap(
        train_file,
        mode="r",
        dtype="<u8",
        offset=layout.data_offset + layout.key_offset,
        shape=(layout.num_vecs,),
    )

    best_scores = np.full(
        (neighbor_count, query_count), -np.inf, dtype=np.float32
    )
    best_rows = np.full(
        (neighbor_count, query_count), -1, dtype=np.int64
    )
    started = time.perf_counter()
    print(
        f"\nGenerating exact cosine ground truth: "
        f"{layout.num_vecs:,} vectors x {query_count:,} queries, "
        f"k={neighbor_count}"
    )

    for start in range(0, layout.num_vecs, block_size):
        stop = min(start + block_size, layout.num_vecs)
        block = np.array(
            vectors[start:stop], dtype=np.float32, order="C", copy=True
        )
        block_norms = np.sqrt(
            np.einsum("ij,ij->i", block, block, optimize=True)
        )
        invalid = ~np.isfinite(block_norms)
        if np.any(invalid):
            first_bad = start + int(np.flatnonzero(invalid)[0])
            raise ValueError(
                f"Training vector row {first_bad} contains non-finite values."
            )
        block_norms[block_norms == 0] = 1.0
        block /= block_norms[:, np.newaxis]

        scores = block @ query_transpose
        block_k = min(neighbor_count, stop - start)
        local_rows = np.argpartition(
            scores, scores.shape[0] - block_k, axis=0
        )[-block_k:, :]
        block_scores = scores[local_rows, query_columns]
        block_rows = local_rows.astype(np.int64, copy=False) + start

        candidate_scores = np.concatenate(
            (best_scores, block_scores), axis=0
        )
        candidate_rows = np.concatenate((best_rows, block_rows), axis=0)
        keep = np.argpartition(
            candidate_scores,
            candidate_scores.shape[0] - neighbor_count,
            axis=0,
        )[-neighbor_count:, :]
        best_scores = candidate_scores[keep, query_columns]
        best_rows = candidate_rows[keep, query_columns]

        elapsed = time.perf_counter() - started
        completed = stop / layout.num_vecs
        eta = elapsed * (1.0 - completed) / completed
        print(
            f"\rGround truth: {completed:6.2%} "
            f"({stop:,}/{layout.num_vecs:,})  "
            f"elapsed {display_duration(elapsed)}, "
            f"ETA {display_duration(eta)}",
            end="",
            flush=True,
        )

    order = np.argsort(best_scores, axis=0)[::-1, :]
    best_rows = best_rows[order, query_columns]
    neighbor_keys = keys[np.ascontiguousarray(best_rows.T)]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = output_file.with_suffix(output_file.suffix + ".tmp")
    with temporary_file.open("w", encoding="utf-8", newline="\n") as stream:
        for query_id, neighbor_row in zip(query_ids, neighbor_keys):
            neighbors = " ".join(str(int(key)) for key in neighbor_row)
            stream.write(f"{query_id};{neighbors}\n")
    os.replace(temporary_file, output_file)

    elapsed = time.perf_counter() - started
    print(
        f"\nGround truth complete in {display_duration(elapsed)}: "
        f"{output_file}"
    )


def build_yaml(
    *,
    train_file: Path,
    index_path: Path,
    converter: str,
    build_threads: int,
    max_degree: int,
    builder_list_size: int,
    memory_limit: float,
    pq_chunks: int,
) -> str:
    return f"""BuilderCommon:
    BuilderClass: DiskAnnBuilder
    BuildFile: {yaml_path(train_file)}
    NeedTrain: true
    TrainFile: {yaml_path(train_file)}
    DumpPath: {yaml_path(index_path)}
    IndexPath: {yaml_path(index_path)}
    MetricName: Cosine
    ConverterName: {converter}
    ThreadCount: {build_threads}
    LogLevel: Info
BuilderParams:
    zvec.general.builder.thread_count: !!int {build_threads}
    zvec.diskann.builder.thread_count: !!int {build_threads}
    zvec.diskann.builder.max_degree: !!int {max_degree}
    zvec.diskann.builder.list_size: !!int {builder_list_size}
    zvec.diskann.builder.memory_limit: !!float {memory_limit}
    zvec.diskann.builder.max_pq_chunk_num: !!int {pq_chunks}
"""


def search_yaml(
    *,
    index_path: Path,
    query_file: Path,
    ground_truth_file: Path | None,
    recall_log_dir: Path,
    top_k: str,
    recall_threads: int,
    bench_threads: int,
    bench_seconds: int,
    bench_iterations: int,
    cache_nodes: int,
    list_size: int,
    beam_size: int,
) -> str:
    ground_truth = ""
    if ground_truth_file is not None:
        ground_truth = f"    GroundTruthFile: {yaml_path(ground_truth_file)}\n"
    return f"""SearcherCommon:
    SearcherClass: DiskAnnSearcher
    IndexPath: {yaml_path(index_path)}
    TopK: {top_k}
    QueryFile: {yaml_path(query_file)}
    QueryType: float
    QueryFirstSep: ";"
    QuerySecondSep: " "
{ground_truth}    RecallLogDir: {yaml_path(recall_log_dir)}
    RecallThreadCount: {recall_threads}
    RecallScorePrecision: 1e-4
    BenchThreadCount: {bench_threads}
    BenchSecs: {bench_seconds}
    BenchIterCount: {bench_iterations}
    CompareById: true
    ContainerType: FileReadStorage
    LogLevel: Info
SearcherParams:
    zvec.diskann.searcher.cache_node_num: !!int {cache_nodes}
    zvec.diskann.searcher.list_size: !!int {list_size}
    zvec.diskann.searcher.beam_size: !!int {beam_size}
ContainerParams: {{}}
"""


def process_sample(process: subprocess.Popen[str]) -> tuple[int, int, int] | None:
    if not IS_WINDOWS:
        return None
    try:
        handle = ctypes.wintypes.HANDLE(int(process._handle))  # type: ignore[attr-defined]
        memory = PROCESS_MEMORY_COUNTERS_EX()
        memory.cb = ctypes.sizeof(memory)
        io = IO_COUNTERS()
        psapi = ctypes.windll.psapi
        kernel32 = ctypes.windll.kernel32
        memory_ok = psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(memory), ctypes.sizeof(memory)
        )
        io_ok = kernel32.GetProcessIoCounters(handle, ctypes.byref(io))
        if not memory_ok or not io_ok:
            return None
        return (
            int(memory.WorkingSetSize),
            int(io.ReadOperationCount),
            int(io.ReadTransferCount),
        )
    except (AttributeError, OSError, ValueError):
        return None


def run_logged(
    command: Sequence[str | Path],
    *,
    log_path: Path,
    cwd: Path,
    env: dict[str, str],
    dry_run: bool,
    monitor_query: bool = False,
) -> ProcessMetrics:
    print(f"\n> {command_text(command)}")
    print(f"  log: {log_path}")
    if dry_run:
        return ProcessMetrics()

    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    query_started = threading.Event()
    with log_path.open("w", encoding="utf-8", newline="\n") as log_file:
        process = subprocess.Popen(
            [str(part) for part in command],
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        def copy_output() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log_file.write(line)
                log_file.flush()
                if "Load index done!" in line:
                    query_started.set()

        output_thread = threading.Thread(target=copy_output, daemon=True)
        output_thread.start()
        peak_working_set = 0
        query_baseline: tuple[int, int, int] | None = None
        query_baseline_time: float | None = None
        final_sample: tuple[int, int, int] | None = None
        try:
            while process.poll() is None:
                sample = process_sample(process)
                if sample is not None:
                    final_sample = sample
                    peak_working_set = max(peak_working_set, sample[0])
                    if (
                        monitor_query
                        and query_started.is_set()
                        and query_baseline is None
                    ):
                        query_baseline = sample
                        query_baseline_time = time.perf_counter()
                time.sleep(0.2)
        except KeyboardInterrupt:
            process.terminate()
            raise
        finally:
            output_thread.join(timeout=10)

        return_code = process.wait()
        wall_seconds = time.perf_counter() - started
        if return_code != 0:
            raise subprocess.CalledProcessError(
                return_code, [str(part) for part in command]
            )

    metrics = ProcessMetrics(wall_seconds=wall_seconds)
    if peak_working_set:
        metrics.peak_working_set_mb = peak_working_set / MIB
    if (
        monitor_query
        and query_baseline is not None
        and query_baseline_time is not None
        and final_sample is not None
    ):
        duration = max(time.perf_counter() - query_baseline_time, 0.001)
        read_operations = max(0, final_sample[1] - query_baseline[1])
        read_bytes = max(0, final_sample[2] - query_baseline[2])
        metrics.read_operations = read_operations
        metrics.read_megabytes = read_bytes / MIB
        metrics.read_iops = read_operations / duration
        metrics.read_mb_per_second = (read_bytes / MIB) / duration
    return metrics


def read_log(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def match_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
    return float(match.group(1)) if match else None


def match_int(pattern: str, text: str) -> int | None:
    match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
    return int(match.group(1)) if match else None


def parse_build_result(
    precision: str,
    index_path: Path,
    log_path: Path,
    wall_seconds: float | None,
) -> BuildResult:
    text = read_log(log_path)
    size = index_path.stat().st_size / GIB if index_path.is_file() else None
    train_ms = match_float(r"Train finished,\s*consume\s+([\d.]+)ms", text)
    build_ms = match_float(r"Build finished,\s*consume\s+([\d.]+)ms", text)
    dump_ms = match_float(r"Dump .*?consume\s+([\d.]+)ms", text)
    return BuildResult(
        precision=precision,
        index_path=str(index_path),
        index_size_gib=size,
        train_seconds=train_ms / 1000 if train_ms is not None else None,
        build_seconds=build_ms / 1000 if build_ms is not None else None,
        dump_seconds=dump_ms / 1000 if dump_ms is not None else None,
        wall_seconds=wall_seconds,
        log_path=str(log_path),
    )


def parse_recall(log_path: Path) -> dict[int, float]:
    text = read_log(log_path)
    return {
        int(k): float(value)
        for k, value in re.findall(
            r"Recall@(\d+):\s*([\d.]+)", text, re.IGNORECASE
        )
    }


def parse_search_result(
    *,
    precision: str,
    list_size: int,
    threads: int,
    recall: dict[int, float],
    recall_log_path: Path,
    bench_log_path: Path,
    metrics: ProcessMetrics,
) -> SearchResult:
    text = read_log(bench_log_path)
    query_count = match_int(r"Process query:\s*(\d+)", text)
    read_operations = metrics.read_operations
    reads_per_query = None
    if read_operations is not None and query_count:
        reads_per_query = read_operations / query_count
    return SearchResult(
        precision=precision,
        list_size=list_size,
        threads=threads,
        recall_at_1=recall.get(1),
        recall_at_10=recall.get(10),
        recall_at_50=recall.get(50),
        qps=match_float(r"\bqps:\s*([\d.]+)", text),
        avg_latency_ms=match_float(r"Avg latency:\s*([\d.]+)ms", text),
        p50_latency_ms=match_float(r"50 Percentile:\s*([\d.]+)\s*ms", text),
        p95_latency_ms=match_float(r"95 Percentile:\s*([\d.]+)\s*ms", text),
        p99_latency_ms=match_float(r"99 Percentile:\s*([\d.]+)\s*ms", text),
        min_latency_ms=match_float(r"\bmin:\s*([\d.]+)ms", text),
        max_latency_ms=match_float(r"\bmax:\s*([\d.]+)ms", text),
        query_count=query_count,
        peak_working_set_mb=metrics.peak_working_set_mb,
        process_read_iops=metrics.read_iops,
        process_read_mb_per_second=metrics.read_mb_per_second,
        reads_per_query=reads_per_query,
        recall_log_path=str(recall_log_path),
        bench_log_path=str(bench_log_path),
    )


def fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    output_dir: Path,
    metadata: dict[str, Any],
    builds: list[BuildResult],
    searches: list[SearchResult],
) -> None:
    write_text(
        output_dir / "metadata.json",
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
    )
    build_rows = [asdict(row) for row in builds]
    search_rows = [asdict(row) for row in searches]
    write_csv(output_dir / "build_results.csv", build_rows)
    write_csv(output_dir / "results.csv", search_rows)
    write_text(
        output_dir / "results.json",
        json.dumps(
            {"builds": build_rows, "searches": search_rows},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
    )

    lines = [
        "# Zvec DiskANN Windows benchmark",
        "",
        "测试参数与仓库 `macos.md` 的 FP32/FP16 DiskANN 基准口径对齐。",
        "",
        "## Environment",
        "",
        f"- Time: {metadata.get('timestamp', '')}",
        f"- Git: `{metadata.get('git_sha', '')}`",
        f"- OS: {metadata.get('platform', '')}",
        f"- CPU: {metadata.get('processor', '')}",
        f"- Logical CPUs: {metadata.get('logical_cpu_count', '')}",
        f"- Python: {metadata.get('python', '')}",
        f"- I/O backend: `{metadata.get('io_backend', '')}`",
        f"- I/O description: {metadata.get('io_backend_description', '')}",
        f"- Beam size: {metadata.get('parameters', {}).get('beam_size', '')}",
        "",
        "## Build",
        "",
        "| Precision | Index GiB | Train s | Build s | Dump s | Wall s |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in builds:
        lines.append(
            "| "
            + " | ".join(
                (
                    row.precision.upper(),
                    fmt(row.index_size_gib, 2),
                    fmt(row.train_seconds, 3),
                    fmt(row.build_seconds, 3),
                    fmt(row.dump_seconds, 3),
                    fmt(row.wall_seconds, 3),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Search",
            "",
            "| Precision | List | Threads | R@1 % | R@10 % | R@50 % | "
            "QPS | Avg ms | P50 ms | P95 ms | P99 ms | Peak MiB | "
            "Read IOPS | Read MiB/s | Reads/query |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in searches:
        lines.append(
            "| "
            + " | ".join(
                (
                    row.precision.upper(),
                    str(row.list_size),
                    str(row.threads),
                    fmt(row.recall_at_1, 3),
                    fmt(row.recall_at_10, 3),
                    fmt(row.recall_at_50, 3),
                    fmt(row.qps, 1),
                    fmt(row.avg_latency_ms, 1),
                    fmt(row.p50_latency_ms, 1),
                    fmt(row.p95_latency_ms, 1),
                    fmt(row.p99_latency_ms, 1),
                    fmt(row.peak_working_set_mb, 1),
                    fmt(row.process_read_iops, 1),
                    fmt(row.process_read_mb_per_second, 1),
                    fmt(row.reads_per_query, 1),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "> Read IOPS / MiB/s come from the benchmark process's Windows "
            "I/O counters after `Load index done!`. With the DiskANN reader's "
            "`FILE_FLAG_NO_BUFFERING`, they describe the query read workload, "
            "but they are not device-wide hardware counters.",
            "",
        ]
    )
    write_text(output_dir / "summary.md", "\n".join(lines))


def get_zvec_backend() -> tuple[str, str]:
    try:
        import zvec  # type: ignore[import-not-found]

        return str(zvec.io_backend_type()), str(zvec.io_backend_description())
    except Exception as exc:  # pragma: no cover - environment dependent
        return "unavailable", f"{type(exc).__name__}: {exc}"


def get_disk_metadata() -> Any:
    if not IS_WINDOWS:
        return []
    powershell = shutil.which("powershell.exe")
    if not powershell:
        return []
    command = (
        "Get-CimInstance Win32_DiskDrive | "
        "Select-Object Model,InterfaceType,MediaType,Size | "
        "ConvertTo-Json -Compress"
    )
    try:
        result = subprocess.run(
            [powershell, "-NoProfile", "-Command", command],
            check=True,
            capture_output=True,
            text=True,
            errors="replace",
        )
        output = result.stdout.strip()
        return json.loads(output) if output else []
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
        return []


def main() -> int:
    args = parse_args()
    repo_root = resolved(args.repo_root)
    train_file = resolved(args.train_file)
    query_file = resolved(args.query_file)
    ground_truth_file = (
        resolved(args.ground_truth_file) if args.ground_truth_file else None
    )
    build_dir = (
        resolved(args.build_dir, repo_root)
        if args.build_dir
        else (repo_root / "build").resolve()
    )

    require_positive(args.list_sizes, "list sizes")
    require_positive(args.thread_counts, "thread counts")
    require_positive(
        (
            args.build_threads,
            args.recall_threads,
            args.bench_seconds,
            args.bench_iterations,
            args.parallel_builds,
            args.dimension,
            args.ground_truth_k,
            args.ground_truth_block_size,
            args.beam_size,
        ),
        "benchmark parameters",
    )
    try:
        requested_top_k = [int(value) for value in args.top_k.split(",")]
    except ValueError as exc:
        raise ValueError(
            f"--top-k must be a comma-separated integer list: {args.top_k!r}"
        ) from exc
    require_positive(requested_top_k, "top-k values")
    if (
        not args.skip_recall
        and ground_truth_file is None
        and args.ground_truth_mode == "auto"
        and args.ground_truth_k < max(requested_top_k)
    ):
        raise ValueError(
            f"--ground-truth-k ({args.ground_truth_k}) must be at least the "
            f"largest --top-k value ({max(requested_top_k)})."
        )
    if not args.dry_run and not IS_WINDOWS:
        raise RuntimeError("This benchmark runner must be executed on Windows.")
    for label, path in (
        ("repository", repo_root),
        ("train file", train_file),
        ("query file", query_file),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if ground_truth_file is not None and not ground_truth_file.is_file():
        raise FileNotFoundError(
            f"ground-truth file does not exist: {ground_truth_file}"
        )

    sha = git_sha(repo_root)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        resolved(args.output_dir, repo_root)
        if args.output_dir
        else (repo_root / "diskann_bench_windows" / f"{stamp}_{sha}").resolve()
    )
    index_dir = (
        resolved(args.index_dir, repo_root)
        if args.index_dir
        else output_dir / "indexes"
    )
    config_dir = output_dir / "configs"
    log_dir = output_dir / "logs"
    recall_detail_dir = output_dir / "recall"
    for directory in (
        output_dir,
        index_dir,
        config_dir,
        log_dir,
        recall_detail_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    if (
        ground_truth_file is None
        and not args.skip_recall
        and args.ground_truth_mode == "auto"
    ):
        ground_truth_file = (
            output_dir
            / f"ground_truth_d{args.dimension}_k{args.ground_truth_k}.txt"
        ).resolve()
        generate_external_ground_truth(
            train_file=train_file,
            query_file=query_file,
            output_file=ground_truth_file,
            dimension=args.dimension,
            neighbor_count=args.ground_truth_k,
            block_size=args.ground_truth_block_size,
            dry_run=args.dry_run,
        )

    env = os.environ.copy()
    append_msys2_to_path(env)
    io_backend, io_description = get_zvec_backend()
    metadata: dict[str, Any] = {
        "timestamp": dt.datetime.now().astimezone().isoformat(),
        "git_sha": sha,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "io_backend": io_backend,
        "io_backend_description": io_description,
        "disks": get_disk_metadata(),
        "train_file": str(train_file),
        "train_file_gib": (
            train_file.stat().st_size / GIB if train_file.is_file() else None
        ),
        "query_file": str(query_file),
        "ground_truth_file": (
            str(ground_truth_file) if ground_truth_file else None
        ),
        "parameters": {
            "precision": list(args.precision),
            "list_sizes": list(args.list_sizes),
            "thread_counts": list(args.thread_counts),
            "build_threads": args.build_threads,
            "recall_threads": args.recall_threads,
            "bench_seconds": args.bench_seconds,
            "bench_iterations": args.bench_iterations,
            "cache_nodes": args.cache_nodes,
            "beam_size": args.beam_size,
            "max_degree": args.max_degree,
            "builder_list_size": args.builder_list_size,
            "pq_chunks": args.pq_chunks,
            "memory_limit": args.memory_limit,
            "top_k": args.top_k,
            "ground_truth_mode": args.ground_truth_mode,
            "dimension": args.dimension,
            "ground_truth_k": args.ground_truth_k,
            "ground_truth_block_size": args.ground_truth_block_size,
        },
    }
    write_outputs(output_dir, metadata, [], [])

    tools = ensure_tools(
        repo_root,
        build_dir,
        env,
        skip_build=args.skip_tool_build,
        parallel_builds=args.parallel_builds,
        dry_run=args.dry_run,
    )
    print(f"\nResults: {output_dir}")
    print(f"I/O backend: {io_backend}")

    converters = {
        "fp32": "CosineFp32Converter",
        "fp16": "CosineFp16Converter",
    }
    indexes: dict[str, Path] = {}
    build_results: list[BuildResult] = []
    for precision in args.precision:
        index_path = (index_dir / f"diskann_{precision}.index").resolve()
        indexes[precision] = index_path
        config_path = config_dir / f"build_{precision}.yaml"
        write_text(
            config_path,
            build_yaml(
                train_file=train_file,
                index_path=index_path,
                converter=converters[precision],
                build_threads=args.build_threads,
                max_degree=args.max_degree,
                builder_list_size=args.builder_list_size,
                memory_limit=args.memory_limit,
                pq_chunks=args.pq_chunks,
            ),
        )
        build_log = log_dir / f"build_{precision}.log"
        wall_seconds: float | None = None
        should_build = not args.skip_index_build and (
            args.rebuild_index or not index_path.exists()
        )
        if should_build:
            if args.rebuild_index and index_path.exists() and not args.dry_run:
                if not index_path.is_file():
                    raise RuntimeError(
                        f"Refusing to replace a non-file index path: {index_path}"
                    )
                index_path.unlink()
            metrics = run_logged(
                [tools["local_builder_original"], config_path],
                log_path=build_log,
                cwd=output_dir,
                env=env,
                dry_run=args.dry_run,
            )
            wall_seconds = metrics.wall_seconds
        elif index_path.exists():
            print(f"\nReusing existing {precision.upper()} index: {index_path}")
        elif not args.dry_run:
            raise FileNotFoundError(
                f"{precision.upper()} index is missing: {index_path}"
            )
        build_results.append(
            parse_build_result(precision, index_path, build_log, wall_seconds)
        )
        write_outputs(output_dir, metadata, build_results, [])

    recall_results: dict[tuple[str, int], dict[int, float]] = {}
    recall_logs: dict[tuple[str, int], Path] = {}
    if not args.skip_recall:
        for precision in args.precision:
            for list_size in args.list_sizes:
                recall_log = log_dir / f"recall_{precision}_l{list_size}.log"
                recall_logs[(precision, list_size)] = recall_log
                config_path = (
                    config_dir / f"recall_{precision}_l{list_size}.yaml"
                )
                write_text(
                    config_path,
                    search_yaml(
                        index_path=indexes[precision],
                        query_file=query_file,
                        ground_truth_file=ground_truth_file,
                        recall_log_dir=(
                            recall_detail_dir
                            / f"{precision}_l{list_size}"
                        ),
                        top_k=args.top_k,
                        recall_threads=args.recall_threads,
                        bench_threads=max(args.thread_counts),
                        bench_seconds=args.bench_seconds,
                        bench_iterations=args.bench_iterations,
                        cache_nodes=args.cache_nodes,
                        list_size=list_size,
                        beam_size=args.beam_size,
                    ),
                )
                run_logged(
                    [tools["recall_original"], config_path],
                    log_path=recall_log,
                    cwd=output_dir,
                    env=env,
                    dry_run=args.dry_run,
                )
                recall_results[(precision, list_size)] = parse_recall(
                    recall_log
                )
    else:
        for precision in args.precision:
            for list_size in args.list_sizes:
                recall_log = log_dir / f"recall_{precision}_l{list_size}.log"
                recall_logs[(precision, list_size)] = recall_log
                recall_results[(precision, list_size)] = parse_recall(
                    recall_log
                )

    search_results: list[SearchResult] = []
    if not args.skip_bench:
        for precision in args.precision:
            for list_size in args.list_sizes:
                for threads in args.thread_counts:
                    bench_log = (
                        log_dir
                        / f"bench_{precision}_l{list_size}_t{threads}.log"
                    )
                    config_path = (
                        config_dir
                        / f"bench_{precision}_l{list_size}_t{threads}.yaml"
                    )
                    write_text(
                        config_path,
                        search_yaml(
                            index_path=indexes[precision],
                            query_file=query_file,
                            ground_truth_file=ground_truth_file,
                            recall_log_dir=(
                                recall_detail_dir
                                / f"{precision}_l{list_size}"
                            ),
                            top_k=args.top_k,
                            recall_threads=args.recall_threads,
                            bench_threads=threads,
                            bench_seconds=args.bench_seconds,
                            bench_iterations=args.bench_iterations,
                            cache_nodes=args.cache_nodes,
                            list_size=list_size,
                            beam_size=args.beam_size,
                        ),
                    )
                    metrics = run_logged(
                        [tools["bench_original"], config_path],
                        log_path=bench_log,
                        cwd=output_dir,
                        env=env,
                        dry_run=args.dry_run,
                        monitor_query=True,
                    )
                    search_results.append(
                        parse_search_result(
                            precision=precision,
                            list_size=list_size,
                            threads=threads,
                            recall=recall_results.get(
                                (precision, list_size), {}
                            ),
                            recall_log_path=recall_logs[
                                (precision, list_size)
                            ],
                            bench_log_path=bench_log,
                            metrics=metrics,
                        )
                    )
                    write_outputs(
                        output_dir, metadata, build_results, search_results
                    )

    write_outputs(output_dir, metadata, build_results, search_results)
    print("\nBenchmark complete.")
    print(f"Markdown summary: {output_dir / 'summary.md'}")
    print(f"CSV results:      {output_dir / 'results.csv'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
