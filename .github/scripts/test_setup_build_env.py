# Keep these checks usable with the standard library in the lint job.
# ruff: noqa: PT009, PT027
from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import setup_build_env


class BuildEnvironmentTest(unittest.TestCase):
    def test_affinity_limits_the_default_budget(self):
        with (
            patch.object(os, "sched_getaffinity", return_value={2, 3}, create=True),
            patch.object(os, "cpu_count", return_value=64),
        ):
            self.assertEqual(setup_build_env.available_cpus(), 2)

    def test_cpu_count_fallback(self):
        with (
            patch.object(os, "sched_getaffinity", side_effect=OSError, create=True),
            patch.object(os, "cpu_count", return_value=None),
        ):
            self.assertEqual(setup_build_env.available_cpus(), 1)

    def test_explicit_budget_and_empty_default(self):
        for value, expected in [("2", 2), ("", 8), ("0002", 2)]:
            with (
                self.subTest(value=value),
                patch.dict(os.environ, {"CMAKE_BUILD_PARALLEL_LEVEL": value}),
            ):
                self.assertEqual(setup_build_env.build_jobs(8), expected)

    def test_invalid_budgets_are_rejected(self):
        for value in ["0", "-1", "auto", "2\nOTHER=value"]:
            with (
                self.subTest(value=value),
                patch.dict(os.environ, {"CMAKE_BUILD_PARALLEL_LEVEL": value}),
                self.assertRaises(ValueError),
            ):
                setup_build_env.build_jobs(8)

    def test_exports_budget_without_changing_test_cpu_count(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / "github-env"
            with (
                patch.dict(
                    os.environ,
                    {"CMAKE_BUILD_PARALLEL_LEVEL": "2", "GITHUB_ENV": str(env_file)},
                ),
                patch.object(setup_build_env, "available_cpus", return_value=8),
                patch("sys.argv", ["setup_build_env.py"]),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                setup_build_env.main()
            self.assertEqual(
                env_file.read_text(encoding="utf-8"),
                "CMAKE_BUILD_PARALLEL_LEVEL=2\nNPROC=8\n",
            )

    def test_container_probe_does_not_write_github_environment(self):
        output = io.StringIO()
        with (
            patch.dict(os.environ, {"CMAKE_BUILD_PARALLEL_LEVEL": "3"}),
            patch("sys.argv", ["setup_build_env.py", "--print"]),
            contextlib.redirect_stdout(output),
        ):
            setup_build_env.main()
        self.assertEqual(output.getvalue(), "3\n")


if __name__ == "__main__":
    unittest.main()
