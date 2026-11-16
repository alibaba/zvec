# Copyright 2025-present the zvec project
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0

"""Numeric driver contracts, independent of the C++ index implementation."""

# Keep these tests runnable with Python's standard library alone.
# ruff: noqa: PT009, PT027
from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "numeric_benchmark_driver", Path(__file__).with_name("run_benchmarks.py")
)
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


def profile(data_type="int64", name="uniform"):
    return {"type": data_type, "name": name}


def arguments(*extra):
    return driver.parse_args(["--type", "int64", "--num-docs", "100000", *extra])


def reports(selected=None, cases=("eq_hit", "eq_miss"), **fields):
    selected = selected or profile()
    build = {
        "dataset": {
            **selected,
            "distribution": selected["name"],
            "num_docs": 100000,
            "seed": driver.NUMERIC_SEED,
            "data_fingerprint": "data",
            **fields,
        },
        "cases": [
            {"name": name, "inputs": [index], "query_fingerprint": name}
            for index, name in enumerate(cases)
        ],
        "metric_names": ["build_seconds"],
        "build_seconds": 1,
    }
    verification = {
        "passed": True,
        "data_fingerprint": "data",
        "cases": [
            {
                "name": name,
                "query_fingerprint": name,
                "counts": [index],
                "id_fingerprints": [name],
            }
            for index, name in enumerate(cases)
        ],
    }
    query = {
        "data_fingerprint": "data",
        "cases": [
            {
                "case": name,
                "data_fingerprint": "data",
                "query_fingerprint": name,
                "match_counts": [index],
                "metric_names": ["latency"],
                "latency": index + 1,
            }
            for index, name in enumerate(cases)
        ],
    }
    return build, verification, query


class SelectionTest(unittest.TestCase):
    def test_local_matrix_has_two_types_and_three_distributions(self):
        selected = driver.select_profiles(driver.parse_args(["--list"]))
        self.assertEqual(
            selected,
            [
                profile(data_type, distribution)
                for data_type in ("int32", "int64")
                for distribution in ("uniform", "hotspot_moderate", "hotspot_extreme")
            ],
        )
        self.assertTrue(all(set(item) == {"type", "name"} for item in selected))

    def test_type_and_profile_filters_select_only_requested_combinations(self):
        selected = driver.select_profiles(
            arguments("--type", "int32,int64", "--profile", "uniform,hotspot_extreme")
        )
        self.assertEqual(
            selected,
            [
                profile(t, d)
                for t in ("int32", "int64")
                for d in ("uniform", "hotspot_extreme")
            ],
        )
        self.assertEqual(len(driver.select_profiles(arguments("--type", "all"))), 6)
        self.assertEqual(
            driver.select_profiles(arguments("--profile", "hotspot_moderate")),
            [profile(name="hotspot_moderate")],
        )

    def test_unknown_profile_and_malformed_filters_are_rejected(self):
        for value in ("missing", "uniform,", "../escape"):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                driver.select_profiles(arguments("--profile", value))

    def test_configuration_forwards_explicit_zero_without_copying_defaults(self):
        self.assertEqual(driver.configuration(arguments()), {})
        args = arguments(
            "--value-range",
            "0_to_type_max",
            "--value-order",
            "shuffled",
            "--null-ratio",
            "0",
        )
        overrides = driver.configuration(args)
        self.assertEqual(
            overrides,
            {
                "value_range": "0_to_type_max",
                "value_order": "shuffled",
                "null_ratio": 0.0,
            },
        )
        self.assertEqual(
            driver.workload_flags(profile(), overrides),
            {"data_type": "int64", "distribution": "uniform", **overrides},
        )


class CliTest(unittest.TestCase):
    def test_default_binary_path_uses_repository_build(self):
        args = driver.parse_args(["--list"])
        self.assertEqual(
            args.bin_dir, Path(__file__).resolve().parents[3] / "build/bin"
        )
        self.assertEqual(args.repetitions, 3)

    def test_numeric_requires_types_and_document_count_only_for_runs(self):
        self.assertIsNone(driver.parse_args(["--list"]).type)
        for options in ([], ["--type", "int64"], ["--num-docs", "100000"]):
            with (
                self.subTest(options=options),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                driver.parse_args(options)
        for count in (100000, 10000000):
            self.assertEqual(arguments("--num-docs", str(count)).num_docs, count)

    def test_unsupported_options_and_invalid_numeric_values_are_rejected(self):
        for options in (
            ("--suite", "numeric"),
            ("--type", "array_int32"),
            ("--type", "numeric"),
            ("--type", ""),
            ("--type", "int32,"),
            ("--case", "eq_hit"),
            ("--cardinality", "200"),
            ("--array-length", "8"),
            ("--empty-ratio", "0.1"),
            ("--duplicate-ratio", "0.1"),
            ("--seed", "42"),
            ("--warmup-seconds", "0"),
            ("--measure-seconds", "0.01"),
            ("--num-docs", "300"),
            ("--num-docs", "0"),
            ("--null-ratio", "0.2"),
            ("--value-range", "custom"),
            ("--value-order", "descending"),
            ("--hot-value-count", "300"),
            ("--hot-doc-ratio", "0.9"),
            ("--repetitions", "0"),
            ("--bin", "/unused"),
        ):
            with (
                self.subTest(options=options),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                arguments(*options)

    def test_list_previews_workloads_without_accessing_binary_or_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "results"
            with (
                patch.object(driver, "select_binary") as binary,
                patch.object(driver, "run_profile") as run,
                contextlib.redirect_stdout(io.StringIO()) as printed,
            ):
                result = driver.main(
                    [
                        "--list",
                        "--bin-dir",
                        str(Path(tmp) / "missing"),
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(result, 0)
            self.assertEqual(len(printed.getvalue().splitlines()), 6)
            self.assertIn("numeric/int32/uniform", printed.getvalue())
            self.assertIn("numeric/int64/hotspot_extreme", printed.getvalue())
            self.assertNotIn("eq_hit", printed.getvalue())
            self.assertFalse(output.exists())
            binary.assert_not_called()
            run.assert_not_called()

    def test_list_includes_explicit_document_count_and_zero_null_ratio(self):
        with (
            patch.object(driver, "select_binary") as binary,
            contextlib.redirect_stdout(io.StringIO()) as printed,
        ):
            result = driver.main(
                [
                    "--list",
                    "--type",
                    "int32",
                    "--profile",
                    "uniform",
                    "--num-docs",
                    "10000000",
                    "--null-ratio",
                    "0",
                ]
            )
        self.assertEqual(result, 0)
        self.assertIn("Document count: 10000000", printed.getvalue())
        self.assertIn('"null_ratio": 0.0', printed.getvalue())
        binary.assert_not_called()

    def test_main_requires_only_numeric_binary_and_records_actual_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / driver.BENCHMARK
            binary.write_bytes(b"numeric benchmark placeholder")
            output = Path(tmp) / "results"
            with (
                patch.object(
                    driver, "run_worker", side_effect=reports(cases=("new_cpp_case",))
                ) as worker,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = driver.main(
                    [
                        "--type",
                        "int64",
                        "--profile",
                        "uniform",
                        "--num-docs",
                        "100000",
                        "--repetitions",
                        "1",
                        "--bin-dir",
                        tmp,
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(result, 0)
            self.assertEqual(
                [call.args[1] for call in worker.call_args_list],
                ["build", "verify", "query"],
            )
            self.assertTrue(
                all(call.args[0] == binary.resolve() for call in worker.call_args_list)
            )
            metadata = json.loads((output / "run.json").read_text())
            self.assertEqual(
                metadata["binaries"],
                {"numeric": driver.binary_record(binary.resolve())},
            )
            self.assertEqual(metadata["status"], "passed")
            self.assertEqual(
                metadata["selected"],
                [
                    {
                        "suite": "numeric",
                        "type": "int64",
                        "profile": "uniform",
                        "cases": ["new_cpp_case"],
                    }
                ],
            )
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["profiles"][0]["cases"][0]["name"], "new_cpp_case")

    def test_existing_results_are_not_modified(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / driver.BENCHMARK).touch()
            output = Path(tmp) / "results"
            output.mkdir()
            marker = output / "run.json"
            marker.write_text("keep me")
            with (
                patch.object(driver, "run_profile") as run,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                result = driver.main(
                    [
                        "--type",
                        "int64",
                        "--num-docs",
                        "100000",
                        "--bin-dir",
                        tmp,
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(result, 1)
            self.assertEqual(marker.read_text(), "keep me")
            run.assert_not_called()

    def test_failure_and_interrupt_are_recorded(self):
        for error, code in (
            (RuntimeError("worker failed"), 1),
            (KeyboardInterrupt(), 130),
        ):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as tmp:
                (Path(tmp) / driver.BENCHMARK).touch()
                output = Path(tmp) / "results"
                with (
                    patch.object(driver, "run_profile", side_effect=error),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    result = driver.main(
                        [
                            "--type",
                            "int64",
                            "--num-docs",
                            "100000",
                            "--bin-dir",
                            tmp,
                            "--output",
                            str(output),
                        ]
                    )
                self.assertEqual(result, code)
                metadata = json.loads((output / "run.json").read_text())
                self.assertEqual(metadata["status"], "failed")
                self.assertEqual(metadata["error"], str(error) or "Interrupted")
                self.assertFalse((output / "summary.json").exists())


class ExecutionTest(unittest.TestCase):
    def test_build_defines_all_queries_and_metrics_are_aggregated(self):
        selected = profile()
        overrides = {
            "null_ratio": 0.0,
            "value_range": "0_to_type_max",
            "value_order": "shuffled",
        }
        phases = reports(selected, cases=("new_cpp_case", "another_case"), **overrides)
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(driver, "run_worker", side_effect=list(phases) * 2) as worker,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = driver.run_profile(
                Path(driver.BENCHMARK),
                Path(tmp),
                selected,
                arguments("--repetitions", "2"),
                overrides,
            )
        self.assertEqual(
            [call.args[1] for call in worker.call_args_list],
            ["build", "verify", "query"] * 2,
        )
        for call in worker.call_args_list:
            self.assertEqual(call.args[0], Path(driver.BENCHMARK))
            self.assertEqual(call.kwargs["run_dir"], call.args[2].parent)
            if call.args[1] == "build":
                self.assertEqual(
                    call.kwargs,
                    {
                        "run_dir": call.args[2].parent,
                        "num_docs": 100000,
                        "data_type": "int64",
                        "distribution": "uniform",
                        **overrides,
                    },
                )
            else:
                self.assertEqual(call.kwargs, {"run_dir": call.args[2].parent})
        self.assertEqual(
            [case["name"] for case in result["cases"]], ["new_cpp_case", "another_case"]
        )
        self.assertEqual(
            [case["metrics"]["latency"]["median"] for case in result["cases"]], [1, 2]
        )
        self.assertTrue(all(len(case["verification"]) == 2 for case in result["cases"]))

    def test_queries_do_not_run_after_failed_verification(self):
        build, verification, _ = reports()
        verification["passed"] = False
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(
                driver, "run_worker", side_effect=[build, verification]
            ) as worker,
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(RuntimeError, "did not pass"),
        ):
            driver.run_profile(
                Path(driver.BENCHMARK),
                Path(tmp),
                profile(),
                arguments("--repetitions", "1"),
                {},
            )
        self.assertEqual(
            [call.args[1] for call in worker.call_args_list], ["build", "verify"]
        )

    def test_changed_data_queries_or_metadata_stop_repetitions(self):
        for change in ("data", "query", "metadata"):
            first = reports()
            next_build = copy.deepcopy(first[0])
            if change == "data":
                next_build["dataset"]["data_fingerprint"] = "different"
            elif change == "query":
                next_build["cases"][0]["query_fingerprint"] = "different"
            else:
                next_build["dataset"]["unrequested_setting"] = "different"
            with (
                self.subTest(change=change),
                tempfile.TemporaryDirectory() as tmp,
                patch.object(
                    driver, "run_worker", side_effect=[*first, next_build]
                ) as worker,
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaisesRegex(RuntimeError, "changed across repetitions"),
            ):
                driver.run_profile(
                    Path(driver.BENCHMARK),
                    Path(tmp),
                    profile(),
                    arguments("--repetitions", "2"),
                    {},
                )
            self.assertEqual(worker.call_count, 4)


class ReportingTest(unittest.TestCase):
    def test_build_case_set_must_be_nonempty_unique_and_valid(self):
        build, _, _ = reports()
        cases = driver.validate_build(build, profile(), arguments(), {})
        self.assertEqual(list(cases), ["eq_hit", "eq_miss"])
        for entries in (
            [],
            None,
            [None],
            [*build["cases"], build["cases"][0]],
            [{"name": "../escape", "query_fingerprint": "x"}],
            [{"name": "query"}],
        ):
            with self.subTest(entries=entries), self.assertRaises(RuntimeError):
                driver.validate_build(
                    {**build, "cases": entries}, profile(), arguments(), {}
                )

    def test_query_report_requires_every_build_case_exactly_once(self):
        build, verification, report = reports()
        cases = {case["name"]: case for case in build["cases"]}
        verified = {case["name"]: case for case in verification["cases"]}
        self.assertEqual(
            set(driver.validate_numeric_queries(report, build, cases, verified)),
            set(cases),
        )
        entries = report["cases"]
        for malformed in (
            entries[:1],
            [*entries, entries[0]],
            [*entries, {**entries[0], "case": "unexpected"}],
            [],
            None,
        ):
            with self.subTest(entries=malformed), self.assertRaises(RuntimeError):
                driver.validate_numeric_queries(
                    {**report, "cases": malformed}, build, cases, verified
                )
        with self.assertRaisesRegex(RuntimeError, "data fingerprint mismatch"):
            driver.validate_numeric_queries(
                {**report, "data_fingerprint": "wrong"}, build, cases, verified
            )

    def test_build_must_match_requested_profile_and_fixed_seed(self):
        overrides = {"value_order": "ascending", "null_ratio": 0.5}
        build, _, _ = reports(**overrides)
        driver.validate_build(build, profile(), arguments(), overrides)
        for key, value in (
            ("value_order", "shuffled"),
            ("seed", 42),
            ("num_docs", 10),
            ("distribution", "hotspot_extreme"),
            ("type", "int32"),
        ):
            invalid = copy.deepcopy(build)
            invalid["dataset"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, key):
                driver.validate_build(invalid, profile(), arguments(), overrides)

    def test_verification_must_match_inputs_counts_and_all_cases(self):
        build, verification, _ = reports()
        cases = {case["name"]: case for case in build["cases"]}
        for mutate in (
            lambda record: record.update(data_fingerprint="wrong"),
            lambda record: record.update(cases=record["cases"][:1]),
            lambda record: record["cases"][0].update(query_fingerprint="wrong"),
            lambda record: record["cases"][0].update(counts=[-1]),
            lambda record: record["cases"][0].update(id_fingerprints=[]),
        ):
            invalid = copy.deepcopy(verification)
            mutate(invalid)
            with self.subTest(verification=invalid), self.assertRaises(RuntimeError):
                driver.validate_verification(invalid, build, cases)

    def test_query_hit_counts_must_match_verification(self):
        build, verification, report = reports()
        case = report["cases"][0]
        case["match_counts"] = [100]
        with self.assertRaisesRegex(RuntimeError, "hit counts differ"):
            driver.validate_query(case, case["case"], build, verification["cases"][0])

    def test_case_specific_metrics_and_verification_details_are_preserved(self):
        build = {"dataset": {"num_docs": 10, "data_fingerprint": "data"}}
        case = {"name": "ratio", "query_fingerprint": "input", "inputs": [5]}
        verified = {"name": "ratio", "query_fingerprint": "input", "ratio": 0.5}
        verification = {"passed": True, "data_fingerprint": "data", "cases": [verified]}
        self.assertEqual(
            driver.validate_verification(verification, build, {"ratio": case}),
            {"ratio": verified},
        )
        query = {
            "case": "ratio",
            "data_fingerprint": "data",
            "query_fingerprint": "input",
            "metric_names": ["error"],
            "error": 0,
            "estimate": 0.5,
        }
        driver.validate_query(query, "ratio", build, verified)
        self.assertEqual(driver.report_details(query), {"estimate": 0.5})


if __name__ == "__main__":
    unittest.main()
