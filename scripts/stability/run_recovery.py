#!/usr/bin/env python3
"""Repeat crash recovery suites with recorded seeds and per-run JUnit files."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def discover_suites(build):
    names = (build / "recovery-suites.txt").read_text().splitlines()
    if not names or len(names) != len(set(names)):
        raise ValueError("Empty or duplicate recovery suite list")
    for name in names:
        if Path(name).name != name or not name.endswith("_test"):
            raise ValueError(f"Invalid recovery suite: {name}")
        if not (build / "bin" / name).is_file():
            raise FileNotFoundError(f"Missing recovery executable: {name}")
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=759)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"Interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    build = args.build.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "configuration.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "rounds": args.rounds,
                "build": str(build),
                "commit": os.environ.get("GITHUB_SHA"),
                "run_id": os.environ.get("GITHUB_RUN_ID"),
            },
            indent=2,
        )
    )
    suites = discover_suites(build)
    failed = False
    for round_id in range(args.rounds):
        seed = (args.seed + round_id) % 99999 + 1
        for name in suites:
            folder = output / f"{name}-{round_id}"
            folder.mkdir()
            report = folder / "junit.xml"
            command = [
                str(build / "bin" / name),
                "--gtest_shuffle",
                f"--gtest_random_seed={seed}",
                f"--gtest_output=xml:{report}",
            ]
            with (folder / "output.log").open("w") as log:
                log.write(f"command={command!r}\n")
                log.flush()
                try:
                    process = subprocess.Popen(
                        command,
                        cwd=folder,
                        env={**os.environ, "TEST_BINARY_DIR": str(build)},
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    try:
                        code = process.wait(timeout=300)
                    finally:
                        if process.poll() is None:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait(timeout=30)
                    count = int(ET.parse(report).getroot().get("tests", "0"))
                    passed = code == 0 and count > 0
                except (
                    OSError,
                    subprocess.TimeoutExpired,
                    ET.ParseError,
                    ValueError,
                ) as error:
                    log.write(f"Runner failure: {error}\n")
                    root = ET.Element("testsuite", name=name, tests="1", failures="1")
                    case = ET.SubElement(root, "testcase", name="runner")
                    ET.SubElement(case, "failure", message=str(error)).text = str(error)
                    ET.ElementTree(root).write(
                        folder / "runner-junit.xml",
                        encoding="utf-8",
                        xml_declaration=True,
                    )
                    passed = False
            if not passed:
                failed = True
                sys.stderr.write((folder / "output.log").read_text() + "\n")
            sys.stdout.write(f"{'PASS' if passed else 'FAIL'} {name} seed={seed}\n")
            sys.stdout.flush()
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
