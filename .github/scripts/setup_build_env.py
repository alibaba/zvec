"""Set the CI build budget before configuring or building the project."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def available_cpus() -> int:
    if hasattr(os, "sched_getaffinity"):
        try:
            return len(os.sched_getaffinity(0)) or 1
        except OSError:
            pass
    return os.cpu_count() or 1


def build_jobs(cpus: int) -> int:
    value = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "")
    if not value:
        return cpus
    if not value.isascii() or not value.isdecimal() or int(value) < 1:
        msg = "CMAKE_BUILD_PARALLEL_LEVEL must be a positive integer"
        raise ValueError(msg)
    return int(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--print", action="store_true", help="Print only the budget, for container jobs"
    )
    args = parser.parse_args()
    cpus = available_cpus()
    jobs = build_jobs(cpus)
    if args.print:
        print(jobs)  # noqa: T201
        return

    with Path(os.environ["GITHUB_ENV"]).open("a", encoding="utf-8") as output:
        output.write(f"CMAKE_BUILD_PARALLEL_LEVEL={jobs}\n")
        # Existing CTest steps use NPROC independently of the compile budget.
        output.write(f"NPROC={cpus}\n")
    print(f"Using {jobs} build jobs ({cpus} CPUs available)")  # noqa: T201


if __name__ == "__main__":
    main()
