# Copyright 2025-present the zvec project
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Colored menus and phase progress for the numeric benchmark CLI."""

# Terminal output is this module's purpose.
# ruff: noqa: T201
from __future__ import annotations

import os
import sys
from contextlib import contextmanager, suppress
from functools import cache
from pathlib import Path

if sys.platform == "win32":
    import ctypes
    import msvcrt
    from ctypes import wintypes

CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"


@cache
def _enable_windows_color(stream):
    """Enable ANSI sequences once per Windows console; fall back to plain text."""
    kernel32 = ctypes.windll.kernel32
    mode = wintypes.DWORD()
    try:
        handle = wintypes.HANDLE(msvcrt.get_osfhandle(stream.fileno()))
        return bool(
            kernel32.GetConsoleMode(handle, ctypes.byref(mode))
            # ENABLE_PROCESSED_OUTPUT | ENABLE_VIRTUAL_TERMINAL_PROCESSING.
            and kernel32.SetConsoleMode(handle, mode.value | 0x0001 | 0x0004)
        )
    except OSError:
        return False


def _use_color(stream):
    return (
        stream.isatty()
        and os.environ.get("TERM") != "dumb"
        and "NO_COLOR" not in os.environ
        and (sys.platform != "win32" or _enable_windows_color(stream))
    )


def _colored(text, color, stream=None):
    stream = sys.stdout if stream is None else stream
    return f"{color}{text}{RESET}" if _use_color(stream) else text


def print_error(message):
    print(_colored(f"error: {message}", RED, sys.stderr), file=sys.stderr)


def print_report_path(path):
    encoding = sys.stdout.encoding or "utf-8"
    display_path = str(path).encode(encoding, "backslashreplace").decode(encoding)
    try:
        print(_colored(f"Report: {display_path}", GREEN), flush=True)
    except OSError:
        # The report is on disk. Avoid retrying buffered output at process exit.
        with suppress(OSError):
            sys.stdout.close()


def print_workload(index, total, profile):
    print(_colored(f"\nWorkload {index}/{total}", CYAN))
    print(
        f"  {profile['data_type']} / {profile['distribution']} / "
        f"{profile['value_range']} / {profile['value_order']} / "
        f"NULL {profile['null_ratio'] * 100:.0f}%",
        flush=True,
    )


@contextmanager
def phase_progress(label):
    """Update a terminal line at phase boundaries without a background spinner."""
    live = _use_color(sys.stdout)
    print(_colored(f"[RUN] {label}", CYAN), end="" if live else "\n", flush=True)
    try:
        yield
    except BaseException:
        print(
            ("\r\033[2K" if live else "") + _colored(f"[FAILED] {label}", RED),
            flush=True,
        )
        raise
    else:
        if live:
            print("\r\033[2K" + _colored(f"[OK] {label}", GREEN), flush=True)


def _choose(title, options, default=None):
    print("\n" + _colored(title, CYAN))
    default_index = None
    for index, (label, value) in enumerate(options, 1):
        suffix = ""
        if value == default:
            default_index = index
            suffix = _colored(" (default)", YELLOW)
        print(f"  {index}. {label}{suffix}")
    prompt = f"Choice [{default_index}]: " if default_index else "Choice: "
    while True:
        answer = input(_colored(prompt, CYAN)).strip()
        if not answer and default_index is not None:
            return default
        if answer == "all" and any(value == "all" for _, value in options):
            return "all"
        if answer.isdecimal() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1][1]
        print_error(f"Choose a number from 1 to {len(options)}.")


def _read_repetitions(default):
    print("\n" + _colored("repetitions", CYAN))
    while True:
        answer = input(_colored(f"Positive integer [{default}]: ", CYAN)).strip()
        if not answer:
            return default
        try:
            value = int(answer)
        except ValueError:
            value = 0
        if value > 0:
            return value
        print_error("Enter a positive integer.")


def _read_directory(name, current, default):
    print("\n" + _colored(name, CYAN))
    description = (
        "Output directory for data, indexes, and reports; must not already exist."
        if name == "run_dir"
        else "Directory containing inverted_numeric_bench."
    )
    print(description)
    print("Press Enter to keep the current value; enter '-' to reset to the default.")
    answer = input(_colored(f"Directory [{current}]: ", CYAN)).strip()
    if not answer:
        return current
    return default if answer == "-" else Path(answer).expanduser().resolve()


def interactive_args(profile_choices, document_counts, defaults):
    """Collect workload choices and return the normal CLI arguments, or None to exit."""
    print(_colored("Numeric inverted-index benchmarks", CYAN))
    print("Choose options by number; press Enter to accept a default.")
    options = {
        name: [*((str(value), str(value)) for value in choices), ("all", "all")]
        for name, choices in profile_choices.items()
    }
    options["num_docs"] = [(str(value), value) for value in document_counts]
    values = {
        "data_type": None,
        "num_docs": document_counts[0],
        **defaults,
    }

    def choose_value(name):
        if name == "repetitions":
            return _read_repetitions(values[name])
        if name in ("run_dir", "bin_dir"):
            return _read_directory(name, values[name], defaults[name])
        return _choose(name, options[name], default=values[name])

    # Ask for every workload dimension; directories remain optional settings.
    for name in values:
        if name not in ("run_dir", "bin_dir"):
            values[name] = choose_value(name)
    while True:
        print("\n" + _colored("Run configuration", CYAN))
        for name, value in values.items():
            print(f"  --{name:<13} {_colored(str(value), GREEN)}")
        workloads = 1
        for name, choices in profile_choices.items():
            if values[name] == "all":
                workloads *= len(choices)
        print(
            _colored(
                f"\n{workloads} workloads x {values['repetitions']} repetitions", YELLOW
            )
        )
        action = _choose(
            "Action",
            [
                ("Run benchmark", "run"),
                ("Edit configuration", "edit"),
                ("Exit", "exit"),
            ],
            default="run",
        )
        if action == "exit":
            return None
        if action == "run":
            return [f"--{name}={value}" for name, value in values.items()]
        fields = [(name, name) for name in values]
        name = _choose("Edit setting", [*fields, ("Back", None)])
        if name is not None:
            values[name] = choose_value(name)
