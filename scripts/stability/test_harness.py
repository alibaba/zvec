# Standard-library unittest keeps the storage harness independent of pytest.
# ruff: noqa: PT009, PT027
from __future__ import annotations

import contextlib
import io
import os
import signal
import struct
import sys
import tempfile
import types
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from run_storage_faults import Suite, choose_cuts, paused_worker, read_log, resume


class LogReplayTests(unittest.TestCase):
    def test_record_offsets_account_for_payload_and_discard(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "writes.img"
            sector = 512
            records = [
                (0, 8, b"operation-begin\0"),
                (2, 0, b""),
                (100, 4, b""),
                (0, 1, b""),
                (0, 8, b"operation-durable\0"),
            ]
            with path.open("wb") as stream:
                stream.write(
                    struct.pack("<QQQI", 0x6A736677736872, 1, 5, sector).ljust(
                        sector, b"\0"
                    )
                )
                for sectors, flags, mark in records:
                    stream.write(
                        (
                            struct.pack("<QQQQ", 0, sectors, flags, len(mark)) + mark
                        ).ljust(sector, b"\0")
                    )
                    if not flags & 4:
                        stream.write(bytes(sectors * sector))
            self.assertEqual(
                read_log(path),
                [
                    (0, 8, "operation-begin"),
                    (1, 0, None),
                    (2, 4, None),
                    (3, 1, None),
                    (4, 8, "operation-durable"),
                ],
            )
            cuts, durable = choose_cuts(read_log(path), 8, 759)
            self.assertEqual(cuts, [1, 2, 3, 4, 5])
            self.assertEqual(durable, 5)

    def test_missing_durability_mark_is_an_error(self):
        with self.assertRaises(KeyError):
            choose_cuts([(0, 8, "operation-begin"), (1, 0, None)], 8, 759)

    def test_cut_sampling_is_repeatable_and_excludes_unmount(self):
        entries = [(0, 8, "operation-begin")]
        entries += [(index, 1 if index % 3 == 0 else 0, None) for index in range(1, 40)]
        entries += [(40, 8, "operation-durable"), (41, 1, None)]
        cuts, _ = choose_cuts(entries, 8, 1234)
        self.assertEqual(cuts, choose_cuts(entries, 8, 1234)[0])
        self.assertEqual(len(cuts), 8)
        self.assertEqual(cuts[0], 1)
        self.assertEqual(cuts[-1], 41)

    def test_truncated_log_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "writes.img"
            path.write_bytes(
                struct.pack("<QQQI", 0x6A736677736872, 1, 1, 512).ljust(512, b"\0")
            )
            with self.assertRaisesRegex(ValueError, "Truncated"):
                read_log(path)


class ProcessControlTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.worker = self.folder / "worker"
        self.worker.write_text(
            f"#!{sys.executable}\n"
            + """import os, signal, sys
if sys.argv[2] == "early-error":
    print("STORAGE_ERROR: deliberate error", flush=True)
    sys.exit(2)
if sys.argv[2] != "missing-ready":
    print("READY", flush=True)
os.kill(os.getpid(), signal.SIGSTOP)
sys.exit(2)
"""
        )
        self.worker.chmod(0o755)
        self.args = types.SimpleNamespace(worker=self.worker)

    def test_resumes_exact_checkpoint_and_keeps_failure_exit_code(self):
        with paused_worker(
            self.args, self.folder / "db", "flush", False, self.folder
        ) as process:
            self.assertIsNone(process.poll())
            self.assertEqual(resume(process), 2)

    def test_early_failure_is_not_mistaken_for_checkpoint(self):
        with (
            self.assertRaisesRegex(RuntimeError, "deliberate error"),
            paused_worker(
                self.args, self.folder / "db", "early-error", False, self.folder
            ),
        ):
            self.fail("A failed worker must not reach the test body")

    def test_stopped_worker_without_acknowledgement_fails(self):
        with (
            self.assertRaisesRegex(RuntimeError, "without READY"),
            paused_worker(
                self.args, self.folder / "db", "missing-ready", False, self.folder
            ),
        ):
            self.fail("A missing checkpoint acknowledgement must fail")

    def test_exception_kills_and_reaps_stopped_worker(self):
        with (
            self.assertRaisesRegex(RuntimeError, "parent failure"),
            paused_worker(
                self.args, self.folder / "db", "flush", False, self.folder
            ) as process,
        ):
            raise RuntimeError("parent failure")
        self.assertEqual(process.returncode, -signal.SIGKILL)
        with self.assertRaises(ChildProcessError):
            os.waitpid(process.pid, os.WNOHANG)

    def test_report_records_failures_and_continues_other_cases(self):
        suite = Suite(self.folder)

        def fail(_):
            raise RuntimeError("fault was not hit")

        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            suite.case("failure", fail)
            suite.case("success", lambda _: None)
        report = ET.parse(self.folder / "junit.xml").getroot()
        self.assertEqual(report.get("tests"), "2")
        self.assertEqual(report.get("failures"), "1")
        self.assertIn("fault was not hit", report.find("testcase/failure").text)


if __name__ == "__main__":
    unittest.main()
