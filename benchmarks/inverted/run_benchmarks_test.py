# Copyright 2025-present the zvec project
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0

"""Driver contracts independent of the index and C++ benchmark implementation."""

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
    "inverted_benchmark_driver", Path(__file__).with_name("run_benchmarks.py")
)
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


def profile(data_type, name=None, **fields):
    numeric = data_type in ("int32", "int64")
    name = name or ("uniform" if numeric else "random_l8")
    workload = (
        {
            "num_docs": 100000,
            "distribution": name,
            "value_range": "zero_to_30000",
            "value_order": "shuffled",
            "value_min": 0,
            "value_max": 30000,
            "null_ratio": 0.1,
            "hot_value_count": 100,
            "hot_doc_ratio": 0.8,
        }
        if numeric
        else {"cardinality": 1000, "array_length": 8}
    )
    return {
        "name": name,
        "type": data_type,
        "supplemental": False,
        "cases": ["eq_hit"] if numeric else ["any_1_hit"],
        **workload,
        **fields,
    }


def catalog():
    return {
        "schema_version": driver.SCHEMA_VERSION,
        "suites": [
            {
                "name": "numeric",
                "profiles": [profile("int32"), profile("int64")],
            },
            {
                "name": "array",
                "profiles": [
                    profile(
                        "array_int32",
                        "random_l8",
                        array_length=8,
                        cases=["any_1_hit", "all_m_witness"],
                    ),
                    profile(
                        "array_int32",
                        "random_l1",
                        array_length=1,
                        supplemental=True,
                        cases=["any_1_hit"],
                    ),
                ],
            },
        ],
    }


def arguments(*extra):
    return driver.parse_args(["--num-docs", "100000", *extra])


class SelectionTest(unittest.TestCase):
    def test_same_profile_name_can_belong_to_multiple_data_types(self):
        args = arguments(
            "--suite", "numeric", "--type", "int32,int64", "--profile", "uniform"
        )
        selected = driver.select_profiles(catalog(), args)
        self.assertEqual([p["type"] for p in selected], ["int32", "int64"])
        self.assertTrue(all(p["suite"] == "numeric" for p in selected))

    def test_types_are_not_suite_aliases_and_must_match_selection(self):
        for options in (("--type", "numeric"), ("--suite", "array", "--type", "int64")):
            with self.subTest(options=options), self.assertRaises(RuntimeError):
                driver.select_profiles(catalog(), arguments(*options))

    def test_default_and_listing_handle_supplemental_profiles(self):
        self.assertEqual(len(driver.select_profiles(catalog(), arguments())), 3)
        self.assertEqual(len(driver.select_profiles(catalog(), arguments("--list"))), 4)

    def test_configuration_is_resolved_before_filtering_cases(self):
        args = arguments(
            "--type",
            "array_int32",
            "--profile",
            "random_l1",
            "--array-length",
            "8",
            "--case",
            "all_m_witness",
        )
        profiles = driver.select_profiles(catalog(), args)
        overrides = driver.configuration(args, profiles)
        resolved = profile(
            "array_int32",
            "random_l1",
            array_length=8,
            supplemental=True,
            cases=["any_1_hit", "all_m_witness"],
        )
        resolved_catalog = {
            "schema_version": driver.SCHEMA_VERSION,
            "suites": [{"name": "array", "profiles": [resolved]}],
        }
        with patch.object(
            driver, "read_catalog", return_value=resolved_catalog
        ) as read:
            profiles = driver.resolve_profiles(
                {"array": Path("inverted_array_bench")}, profiles, overrides
            )
        self.assertEqual(read.call_args.args[0], Path("inverted_array_bench"))
        self.assertNotIn("type", read.call_args.kwargs)
        self.assertNotIn("data_type", read.call_args.kwargs)
        self.assertEqual(profiles[0]["array_length"], 8)
        self.assertEqual(
            driver.select_cases(profiles, args.case)[0][1], ["all_m_witness"]
        )

    def test_type_specific_overrides_cannot_be_silently_ignored(self):
        args = arguments("--array-length", "16")
        with self.assertRaisesRegex(RuntimeError, "only the array suite"):
            driver.configuration(args, driver.select_profiles(catalog(), args))
        args = arguments("--suite", "array", "--value-range", "zero_to_type_max")
        with self.assertRaisesRegex(RuntimeError, "only the numeric suite"):
            driver.configuration(args, driver.select_profiles(catalog(), args))

    def test_numeric_profile_selects_one_distribution(self):
        available = catalog()
        available["suites"][0]["profiles"].append(profile("int64", "hotspot"))
        args = arguments("--type", "int64", "--profile", "hotspot")
        selected = driver.select_profiles(available, args)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["distribution"], "hotspot")
        self.assertEqual(driver.configuration(args, selected), {})

    def test_numeric_workload_controls_are_forwarded_without_affecting_array(self):
        args = arguments(
            "--type",
            "int64",
            "--profile",
            "uniform",
            "--value-range",
            "zero_to_type_max",
            "--value-order",
            "ascending",
            "--null-ratio",
            "0.5",
            "--hot-value-count",
            "32",
            "--hot-doc-ratio",
            "2",
        )
        self.assertEqual(
            driver.configuration(args, driver.select_profiles(catalog(), args)),
            {
                "value_range": "zero_to_type_max",
                "value_order": "ascending",
                "null_ratio": 0.5,
                "hot_value_count": 32,
                "hot_doc_ratio": 2,
            },
        )
        # Active hotspot constraints are checked by the selected numeric catalog.
        for value in ("nan", "inf", "-inf"):
            with (
                self.subTest(value=value),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                arguments(f"--hot-doc-ratio={value}")
        array_args = arguments(
            "--suite", "array", "--cardinality", "200", "--null-ratio", "0.25"
        )
        self.assertEqual(
            driver.configuration(
                array_args, driver.select_profiles(catalog(), array_args)
            ),
            {"cardinality": 200, "null_ratio": 0.25},
        )
        numeric_args = arguments("--type", "int32", "--cardinality", "200")
        with self.assertRaisesRegex(RuntimeError, "only the array suite"):
            driver.configuration(
                numeric_args, driver.select_profiles(catalog(), numeric_args)
            )

    def test_removed_numeric_controls_and_unknown_enum_values_are_rejected(self):
        for options in (
            ("--value-min", "0"),
            ("--zipf-exponent", "1"),
            ("--hot-key-ratio", "0.1"),
            ("--distribution", "hotspot"),
            ("--value-range", "custom"),
            ("--value-order", "descending"),
            ("--hot-value-count", "0"),
            ("--hot-value-count", str(2**32)),
        ):
            with (
                self.subTest(options=options),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                arguments(*options)

    def test_array_document_count_remains_a_positive_uint32(self):
        self.assertEqual(
            arguments("--suite", "array", "--num-docs", "1234").num_docs, 1234
        )
        for value in ("0", "-1", str(2**32)):
            with (
                self.subTest(value=value),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                driver.parse_args(["--num-docs", value])


class RoutingTest(unittest.TestCase):
    def test_numeric_document_counts_are_checked_before_listing_or_running(self):
        for data_type in ("int64", "all"):
            for listing in (False, True):
                with (
                    self.subTest(data_type=data_type, listing=listing),
                    tempfile.TemporaryDirectory() as tmp,
                ):
                    output = Path(tmp) / "results"
                    errors = io.StringIO()
                    with (
                        patch.object(driver, "select_binaries", return_value={}),
                        patch.object(driver, "read_catalogs", return_value=catalog()),
                        patch.object(driver, "resolve_profiles") as resolve,
                        patch.object(driver, "run_profile") as run,
                        contextlib.redirect_stderr(errors),
                    ):
                        result = driver.main(
                            [
                                "--type",
                                data_type,
                                "--num-docs",
                                "1234",
                                "--output",
                                str(output),
                                *(["--list"] if listing else []),
                            ]
                        )
                    self.assertEqual(result, 1)
                    self.assertIn("100000 or 10000000", errors.getvalue())
                    self.assertFalse(output.exists())
                    resolve.assert_not_called()
                    run.assert_not_called()

    def test_numeric_presets_and_arbitrary_array_document_counts_are_accepted(self):
        for data_type, count in (
            ("int32", "100000"),
            ("int64", "10000000"),
            ("all", "10000000"),
            ("array_int32", "1234"),
        ):
            with (
                self.subTest(data_type=data_type, count=count),
                patch.object(driver, "select_binaries", return_value={}),
                patch.object(driver, "read_catalogs", return_value=catalog()),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(
                    driver.main(["--type", data_type, "--num-docs", count, "--list"]),
                    0,
                )

    def test_selected_suite_runs_without_the_other_executable(self):
        for suite, selector in (
            ("numeric", ["--suite", "numeric", "--type", "all"]),
            ("numeric", ["--type", "int64"]),
            ("array", ["--suite", "array"]),
            ("array", ["--type", "array_int32"]),
        ):
            with self.subTest(selector=selector), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                binary = directory / driver.BENCHMARKS[suite]
                binary.write_bytes(b"unit-test executable placeholder")
                one_catalog = {
                    "schema_version": driver.SCHEMA_VERSION,
                    "suites": [s for s in catalog()["suites"] if s["name"] == suite],
                }
                output = directory / "results"
                with (
                    patch.object(
                        driver, "read_catalog", return_value=one_catalog
                    ) as read,
                    patch.object(driver, "run_profile", return_value={}) as run,
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    result = driver.main(
                        [
                            "--bin-dir",
                            str(directory),
                            "--num-docs",
                            "100000" if suite == "numeric" else "1234",
                            "--output",
                            str(output),
                            *selector,
                        ]
                    )
                self.assertEqual(result, 0)
                read.assert_called_once_with(binary.resolve())
                self.assertTrue(run.called)
                for call in run.call_args_list:
                    self.assertEqual(call.args[0], binary.resolve())
                    self.assertEqual(call.args[2]["suite"], suite)
                metadata = json.loads((output / "run.json").read_text())
                self.assertEqual(
                    metadata["binaries"],
                    {suite: driver.binary_record(binary.resolve())},
                )
                self.assertTrue(
                    all(item["suite"] == suite for item in metadata["selected"])
                )

    def test_numeric_resolution_uses_data_type_flag(self):
        args = arguments("--suite", "numeric", "--value-range", "zero_to_type_max")
        profiles = driver.select_profiles(catalog(), args)

        def resolved_catalog(binary, **flags):
            self.assertEqual(binary, Path("inverted_numeric_bench"))
            self.assertNotIn("type", flags)
            return {
                "schema_version": driver.SCHEMA_VERSION,
                "suites": [
                    {
                        "name": "numeric",
                        "profiles": [
                            profile(
                                flags["data_type"],
                                value_range="zero_to_type_max",
                                value_max=2
                                ** (31 if flags["data_type"] == "int32" else 63)
                                - 1,
                            )
                        ],
                    }
                ],
            }

        with patch.object(driver, "read_catalog", side_effect=resolved_catalog):
            resolved = driver.resolve_profiles(
                {"numeric": Path("inverted_numeric_bench")},
                profiles,
                {"value_range": "zero_to_type_max"},
            )
        self.assertEqual([p["type"] for p in resolved], ["int32", "int64"])

    def test_numeric_workers_derive_paths_and_array_workers_keep_explicit_options(self):
        for suite, data_type, case in (
            ("numeric", "int64", "eq_hit"),
            ("array", "array_int32", "any_1_hit"),
        ):
            with self.subTest(suite=suite), tempfile.TemporaryDirectory() as tmp:
                selected = profile(data_type, suite=suite, cases=[case])
                build = {
                    "dataset": {"data_fingerprint": "data"},
                    "metric_names": ["build_seconds"],
                    "build_seconds": 1,
                }
                cases = {case: {"query_fingerprint": "query"}}
                verified = {case: {"counts": [1], "query_fingerprint": "query"}}
                query = {"metric_names": ["latency"], "latency": 1}
                options = ["--repetitions", "1"]
                if suite == "array":
                    options += [
                        "--seed",
                        "42",
                        "--warmup-seconds",
                        "0",
                        "--measure-seconds",
                        "0.001",
                    ]
                with (
                    patch.object(
                        driver, "run_worker", side_effect=[build, {}, query]
                    ) as worker,
                    patch.object(driver, "validate_build", return_value=cases),
                    patch.object(
                        driver, "validate_verification", return_value=verified
                    ),
                    patch.object(driver, "validate_query"),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    driver.run_profile(
                        Path(driver.BENCHMARKS[suite]),
                        Path(tmp),
                        selected,
                        [case],
                        arguments(*options),
                        {},
                    )
                self.assertEqual(worker.call_count, 3)
                for call in worker.call_args_list:
                    self.assertEqual(call.args[0], Path(driver.BENCHMARKS[suite]))
                    self.assertNotIn("type", call.kwargs)
                    if suite == "numeric":
                        self.assertFalse(call.kwargs["explicit_output"])
                        self.assertEqual(call.kwargs["run_dir"], call.args[2].parent)
                        for key in (
                            "index",
                            "manifest",
                            "verification",
                            "seed",
                            "warmup_seconds",
                            "measure_seconds",
                        ):
                            self.assertNotIn(key, call.kwargs)
                        if call.args[1] == "build":
                            self.assertEqual(call.kwargs["data_type"], data_type)
                        else:
                            self.assertNotIn("data_type", call.kwargs)
                    else:
                        self.assertTrue(call.kwargs["explicit_output"])
                        self.assertIn("index", call.kwargs)
                        self.assertNotIn("run_dir", call.kwargs)
                        self.assertNotIn("data_type", call.kwargs)
                        if call.args[1] == "build":
                            self.assertEqual(call.kwargs["seed"], 42)
                        if call.args[1] == "query":
                            self.assertIn("manifest", call.kwargs)
                            self.assertIn("verification", call.kwargs)
                            self.assertEqual(call.kwargs["warmup_seconds"], 0)
                            self.assertEqual(call.kwargs["measure_seconds"], 0.001)

    def test_worker_uses_only_the_selected_output_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            report = directory / "build.json"
            report.write_text(json.dumps({"schema_version": 2, "phase": "build"}))
            for explicit_output in (True, False):
                flags = (
                    {"index": directory / "index"}
                    if explicit_output
                    else {"run_dir": directory}
                )
                with patch.object(driver.subprocess, "run") as run:
                    run.return_value.returncode = 0
                    driver.run_worker(
                        Path("benchmark"),
                        "build",
                        report,
                        directory / f"{explicit_output}.log",
                        explicit_output=explicit_output,
                        **flags,
                    )
                command = run.call_args.args[0]
                self.assertEqual(f"--output={report}" in command, explicit_output)
                self.assertFalse(any("explicit_output" in flag for flag in command))
                if not explicit_output:
                    self.assertIn(f"--run_dir={directory}", command)

    def test_numeric_runs_require_explicit_types_and_fixed_execution_parameters(self):
        numeric_catalog = {
            "schema_version": driver.SCHEMA_VERSION,
            "suites": [catalog()["suites"][0]],
        }
        for options, message in (
            ([], "explicit --type"),
            (["--type", "int64", "--seed", "42"], "fixed seed"),
            (["--type", "int64", "--warmup-seconds", "0"], "fixed seed"),
            (["--type", "int64", "--measure-seconds", "0.001"], "fixed seed"),
        ):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "results"
                errors = io.StringIO()
                with (
                    patch.object(driver, "select_binaries", return_value={}),
                    patch.object(driver, "read_catalogs", return_value=numeric_catalog),
                    patch.object(driver, "run_profile") as run,
                    contextlib.redirect_stderr(errors),
                ):
                    result = driver.main(
                        [
                            "--suite",
                            "numeric",
                            "--num-docs",
                            "100000",
                            "--output",
                            str(output),
                            *options,
                        ]
                    )
                self.assertEqual(result, 1)
                self.assertIn(message, errors.getvalue())
                self.assertFalse(output.exists())
                run.assert_not_called()
        with (
            patch.object(driver, "select_binaries", return_value={}),
            patch.object(driver, "read_catalogs", return_value=numeric_catalog),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(driver.main(["--suite", "numeric", "--list"]), 0)

    def test_mismatched_suite_or_catalog_is_rejected(self):
        for selector in (
            ("--suite", "numeric", "--type", "array_int32"),
            ("--suite", "array", "--type", "int32,int64"),
            ("--type", "numeric"),
        ):
            with self.subTest(selector=selector), self.assertRaises(RuntimeError):
                driver.select_binaries(arguments(*selector))
        wrong_catalog = {
            "schema_version": driver.SCHEMA_VERSION,
            "suites": [catalog()["suites"][0]],
        }
        with (
            patch.object(driver, "read_catalog", return_value=wrong_catalog),
            self.assertRaisesRegex(RuntimeError, "Unexpected catalog"),
        ):
            driver.read_catalogs({"array": Path("inverted_array_bench")})

    def test_removed_bin_option_is_not_an_abbreviation_for_bin_dir(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            arguments("--bin", "/unused/inverted_bench")


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
            driver.summarize_metrics(samples),
            {
                "custom.delta": {"median": -2.0, "min": -4, "max": 0},
                "optional": None,
            },
        )
        before = copy.deepcopy(samples[0])
        self.assertEqual(
            driver.report_details(samples[0]), {"custom": {"method": "direct"}}
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
                driver.validate_metrics(report)
        with self.assertRaisesRegex(RuntimeError, "changed across repetitions"):
            driver.summarize_metrics(
                [
                    {"metric_names": ["x"], "x": 1},
                    {"metric_names": ["y"], "y": 1},
                ]
            )

    def test_custom_case_does_not_require_bitmap_counts_or_qps(self):
        build = {"dataset": {"num_docs": 10, "data_fingerprint": "data"}}
        case = {"name": "ratio", "query_fingerprint": "input", "inputs": [5]}
        verified = {"name": "ratio", "query_fingerprint": "input", "ratio": 0.5}
        verification = {"passed": True, "data_fingerprint": "data", "cases": [verified]}
        self.assertEqual(
            driver.validate_verification(
                verification, build, {"ratio": case}, ["ratio"]
            ),
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

    def test_existing_count_validation_and_extra_verification_fields_remain(self):
        build = {"dataset": {"num_docs": 10, "data_fingerprint": "data"}}
        case = {"name": "lt", "query_fingerprint": "input", "inputs": [5]}
        verified = {
            "name": "lt",
            "query_fingerprint": "input",
            "counts": [4],
            "id_fingerprints": ["ids"],
            "nonnull_match_ratios": [0.5],
        }
        verification = {"passed": True, "data_fingerprint": "data", "cases": [verified]}
        driver.validate_verification(verification, build, {"lt": case}, ["lt"])
        self.assertEqual(driver.report_details(verified)["nonnull_match_ratios"], [0.5])
        verified["counts"] = [-1]
        with self.assertRaisesRegex(RuntimeError, "hit count"):
            driver.validate_verification(verification, build, {"lt": case}, ["lt"])

    def test_build_must_match_resolved_configuration(self):
        selected = profile(
            "int32", value_order="ascending", null_ratio=0.5, suite="numeric"
        )
        dataset = {
            key: value
            for key, value in selected.items()
            if key not in ("cases", "suite")
        }
        dataset.update(num_docs=100000, seed=20261115, data_fingerprint="data")
        build = {
            "dataset": dataset,
            "metric_names": ["build_seconds"],
            "build_seconds": 1,
            "cases": [{"name": "eq_hit", "inputs": [1], "query_fingerprint": "input"}],
        }
        driver.validate_build(build, selected, arguments("--seed", "42"))
        build["dataset"]["value_order"] = "shuffled"
        with self.assertRaisesRegex(RuntimeError, "value_order"):
            driver.validate_build(build, selected, arguments())


if __name__ == "__main__":
    unittest.main()
