# Copyright 2025-present the zvec project
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0

"""Contracts for the IO and reporting helpers shared by suite drivers."""

# Keep these tests runnable with Python's standard library alone.
# ruff: noqa: PT009, PT027
from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "inverted_benchmark_helpers", Path(__file__).with_name("benchmark_helpers.py")
)
helpers = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helpers)


class WorkerTest(unittest.TestCase):
    def test_worker_forwards_only_the_options_its_suite_supplies(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            report = directory / "build.json"
            report.write_text(json.dumps({"schema_version": 2, "phase": "build"}))
            for options in (
                {"run_dir": directory},
                {"index": directory / "index", "output": report},
            ):
                with (
                    self.subTest(options=options),
                    patch.object(helpers.subprocess, "run") as run,
                ):
                    run.return_value.returncode = 0
                    log = directory / f"{next(iter(options))}.log"
                    result = helpers.run_worker(
                        Path("benchmark"), "build", report, log, **options
                    )
                self.assertEqual(
                    run.call_args.args[0],
                    ["benchmark", "--mode=build"]
                    + [f"--{key}={value}" for key, value in options.items()],
                )
                self.assertTrue(log.exists())
                self.assertEqual(result["phase"], "build")

    def test_failed_process_is_rejected_even_if_a_report_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            report = directory / "build.json"
            report.write_text('{"schema_version": 2, "phase": "build"}')
            log = directory / "build.log"
            with patch.object(helpers.subprocess, "run") as run:
                run.return_value.returncode = 7
                with self.assertRaisesRegex(RuntimeError, "exited with 7; see"):
                    helpers.run_worker(Path("benchmark"), "build", report, log)

    def test_successful_process_must_produce_a_matching_report(self):
        for content in (
            None,
            {"schema_version": 1, "phase": "build"},
            {"schema_version": 2, "phase": "query"},
        ):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                report = directory / "build.json"
                if content is not None:
                    report.write_text(json.dumps(content))
                with patch.object(helpers.subprocess, "run") as run:
                    run.return_value.returncode = 0
                    with self.assertRaises(RuntimeError):
                        helpers.run_worker(
                            Path("benchmark"), "build", report, directory / "build.log"
                        )


class ReportingTest(unittest.TestCase):
    def test_custom_zero_and_signed_metrics_are_aggregated(self):
        samples = [
            {
                "metric_names": ["custom.delta", "optional"],
                "custom": {"delta": value, "method": "direct"},
                "optional": None,
            }
            for value in (0, -4)
        ]
        self.assertEqual(
            helpers.summarize_metrics(samples),
            {
                "custom.delta": {"median": -2.0, "min": -4, "max": 0},
                "optional": None,
            },
        )
        before = copy.deepcopy(samples[0])
        self.assertEqual(
            helpers.report_details(samples[0]), {"custom": {"method": "direct"}}
        )
        self.assertEqual(samples[0], before)

    def test_invalid_or_inconsistent_metric_declarations_fail(self):
        for report in (
            {"metric_names": ["missing"]},
            {"metric_names": ["x"], "x": float("nan")},
            {"metric_names": ["x"], "x": True},
            {"metric_names": ["x", "x"], "x": 1},
        ):
            with self.subTest(report=report), self.assertRaises(RuntimeError):
                helpers.validate_metrics(report)
        with self.assertRaisesRegex(RuntimeError, "changed across repetitions"):
            helpers.summarize_metrics(
                [{"metric_names": ["x"], "x": 1}, {"metric_names": ["y"], "y": 1}]
            )

    def test_json_requires_an_object_and_finite_numbers(self):
        for text in ("[]", '{"value": NaN}', '{"value": Infinity}'):
            with self.subTest(text=text), self.assertRaises(RuntimeError):
                helpers.load_json(text)
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.json"
            helpers.write_json(report, {"status": "running"})
            helpers.write_json(report, {"status": "passed"})
            self.assertEqual(
                helpers.load_json(report.read_text()), {"status": "passed"}
            )
            self.assertFalse(report.with_suffix(".json.tmp").exists())

    def test_report_names_cannot_escape_result_directory(self):
        for name in ("../escape", "/absolute", "", "a/b", None):
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                helpers.safe_name(name)
        self.assertEqual(helpers.safe_name("eq_hit"), "eq_hit")


if __name__ == "__main__":
    unittest.main()
