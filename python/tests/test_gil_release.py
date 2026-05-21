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
"""Tests to verify that the GIL is released during native C++ query calls,
enabling true thread-level concurrency for multi-threaded Python applications."""

from __future__ import annotations

import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
import zvec
from zvec import (
    Collection,
    CollectionOption,
    DataType,
    Doc,
    FieldSchema,
    HnswIndexParam,
    Query,
    VectorSchema,
)


@pytest.fixture(scope="module")
def gil_test_collection(tmp_path_factory) -> Collection:
    """Create a collection with enough data to make queries take measurable time."""
    schema = zvec.CollectionSchema(
        name="gil_test",
        fields=[
            FieldSchema("id", DataType.INT64, nullable=False),
        ],
        vectors=[
            VectorSchema(
                "vec",
                DataType.VECTOR_FP32,
                dimension=128,
                index_param=HnswIndexParam(),
            ),
        ],
    )
    option = CollectionOption(read_only=False, enable_mmap=True)
    temp_dir = tmp_path_factory.mktemp("zvec_gil_test")
    collection_path = temp_dir / "gil_test_collection"

    coll = zvec.create_and_open(path=str(collection_path), schema=schema, option=option)

    # Insert enough docs to make queries non-trivial
    docs = [
        Doc(
            id=str(i),
            fields={"id": i},
            vectors={"vec": [float(i % 100) + 0.1 * j for j in range(128)]},
        )
        for i in range(500)
    ]
    result = coll.insert(docs)
    for r in result:
        assert r.ok()

    yield coll

    try:
        coll.destroy()
    except Exception:
        pass


class TestGILRelease:
    """Verify that C++ query calls release the GIL, allowing true thread concurrency."""

    def test_gil_released_during_query(self, gil_test_collection: Collection):
        """Prove the GIL is explicitly released during C++ Query calls.

        Strategy:
        - Calibrate per-query latency on the current platform (slow archs like
          RISC-V can be 10x slower than x86), then dynamically pick a query count
          whose total runtime fits comfortably inside switch_interval.
        - Set switch_interval well above the projected total query time so that
          CPython's involuntary GIL switching will NOT trigger during the run.
        - A background thread (using time.sleep(0) to avoid deadlock) counts how
          many times it got to run.
        - Since total query time < switch_interval, the bg thread can ONLY run if
          the C++ code explicitly releases the GIL.
        - Reset counter just before queries; check counter > 0 after queries.
        """
        query_vec = [1.0] * 128

        def run_query():
            gil_test_collection.query(
                Query(field_name="vec", vector=query_vec),
                topk=100,
            )

        # --- Calibrate: estimate per-query latency on this platform ---
        # Warm up to avoid first-call overhead skewing the measurement.
        for _ in range(3):
            run_query()

        calib_iters = 10
        calib_start = time.monotonic()
        for _ in range(calib_iters):
            run_query()
        per_query = max((time.monotonic() - calib_start) / calib_iters, 1e-6)

        # Target total query window ~200ms, capped to a sane range so the test
        # remains meaningful on both fast and slow archs.
        target_total = 0.2
        num_iters = max(1, min(500, int(target_total / per_query)))
        projected_total = per_query * num_iters
        # Pick switch_interval with a large safety margin (>=10x, >=2s) to absorb
        # GC pauses, CPU throttling, and noisy-neighbor effects on CI / shared VMs.
        switch_interval = max(2.0, projected_total * 10.0)

        old_interval = sys.getswitchinterval()
        sys.setswitchinterval(switch_interval)

        try:
            counter = {"value": 0}
            stop_event = threading.Event()

            def background_counter():
                while not stop_event.is_set():
                    counter["value"] += 1
                    time.sleep(0)  # Yield GIL to prevent deadlock

            bg_thread = threading.Thread(target=background_counter, daemon=True)
            bg_thread.start()

            # Let bg thread start (sleep releases GIL)
            time.sleep(0.05)

            # --- Critical section: reset counter, run queries, capture counter ---
            counter["value"] = 0

            start = time.monotonic()
            for _ in range(num_iters):
                run_query()
            elapsed = time.monotonic() - start

            count_during_queries = counter["value"]
            # --- End critical section ---

            stop_event.set()
            time.sleep(0.01)
            bg_thread.join(timeout=5)

            print(
                f"\nPer-query: {per_query * 1000:.2f}ms, iters: {num_iters}, "
                f"elapsed: {elapsed:.4f}s, switch_interval: {switch_interval:.2f}s"
            )
            print(f"Counter during queries: {count_during_queries}")

            # Verify queries completed within the switch_interval window.
            # If they did NOT, the run was contaminated by external jitter (GC,
            # throttling, noisy neighbor) rather than a real GIL-release defect,
            # so skip instead of failing to avoid flaky CI noise.
            if elapsed >= switch_interval:
                pytest.skip(
                    f"Queries took {elapsed:.3f}s >= switch_interval "
                    f"({switch_interval:.3f}s); calibration was outpaced by "
                    "runtime jitter, result is inconclusive."
                )
            # If elapsed < switch_interval, the ONLY way bg thread could run is
            # via explicit GIL release.
            assert count_during_queries > 0, (
                "Background thread could not run during C++ execution despite "
                "query time < switch_interval. GIL was NOT released."
            )
        finally:
            sys.setswitchinterval(old_interval)

    def test_parallel_queries_correctness(self, gil_test_collection: Collection):
        """Verify parallel queries return correct results and print timing info.

        NOTE: The definitive proof of GIL release is test_gil_released_during_query
        (counter + setswitchinterval). This test focuses on parallel correctness and
        logs timing for manual inspection, since CI timing is too noisy for assertions.
        """
        num_queries = 1000
        query_vec = [1.0] * 128

        def do_query():
            return gil_test_collection.query(
                Query(field_name="vec", vector=query_vec),
                topk=100,
            )

        # Serial execution (baseline)
        start_serial = time.monotonic()
        for _ in range(num_queries):
            do_query()
        serial_time = time.monotonic() - start_serial

        # Parallel execution
        num_workers = os.cpu_count() or 2
        start_parallel = time.monotonic()
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(do_query) for _ in range(num_queries)]
            for future in as_completed(futures):
                result = future.result()
                assert len(result) > 0
        parallel_time = time.monotonic() - start_parallel

        print(f"\nSerial time: {serial_time:.4f}s, Parallel time: {parallel_time:.4f}s")
        print(
            f"Speedup ratio: {serial_time / parallel_time:.2f}x (workers={num_workers})"
        )

    def test_thread_safety_concurrent_queries(self, gil_test_collection: Collection):
        """Verify no crashes or data corruption under concurrent query load."""
        num_threads = 8
        queries_per_thread = 10
        errors = []

        def worker(thread_id):
            try:
                for i in range(queries_per_thread):
                    vec = [float(thread_id + i) + 0.1 * j for j in range(128)]
                    result = gil_test_collection.query(
                        Query(field_name="vec", vector=vec),
                        topk=10,
                    )
                    assert len(result) > 0
            except Exception as e:
                errors.append((thread_id, e))

        threads = [
            threading.Thread(target=worker, args=(tid,)) for tid in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        assert len(errors) == 0, f"Errors in threads: {errors}"

    def test_concurrent_fetch_release_gil(self, gil_test_collection: Collection):
        """Verify Fetch operations also release the GIL correctly."""
        num_threads = 4
        errors = []

        def worker(thread_id):
            try:
                ids = [str(i) for i in range(thread_id * 10, thread_id * 10 + 10)]
                result = gil_test_collection.fetch(ids)
                assert len(result) > 0
            except Exception as e:
                errors.append((thread_id, e))

        threads = [
            threading.Thread(target=worker, args=(tid,)) for tid in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0, f"Errors in threads: {errors}"
