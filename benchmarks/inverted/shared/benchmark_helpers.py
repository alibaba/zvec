# Copyright 2025-present the zvec project
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""JSON, subprocess, and metric helpers for suite-owned benchmark drivers."""

from __future__ import annotations

import copy
import datetime
import hashlib
import json
import math
import re
import statistics
import subprocess


def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def fail(message):
    raise RuntimeError(message)


def load_json(text):
    def invalid_constant(value):
        fail(f"Invalid JSON numeric constant: {value}")

    result = json.loads(text, parse_constant=invalid_constant)
    if not isinstance(result, dict):
        fail("Expected a JSON object")
    return result


def format_json(value, indent=0):
    """Format report floats to two decimal places, retaining JSON number types."""
    if isinstance(value, float):
        if not math.isfinite(value):
            fail("Cannot write a non-finite JSON number")
        return f"{value:.2f}"
    if isinstance(value, dict):
        opening, closing = "{", "}"
        items = [
            f"{json.dumps(key)}: {format_json(item, indent + 2)}"
            for key, item in value.items()
        ]
    elif isinstance(value, list):
        opening, closing = "[", "]"
        items = [format_json(item, indent + 2) for item in value]
    else:
        return json.dumps(value, allow_nan=False)
    if not items:
        return opening + closing
    body = ",\n".join(" " * (indent + 2) + item for item in items)
    return f"{opening}\n{body}\n{' ' * indent}{closing}"


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(format_json(value) + "\n")
    temporary.replace(path)


def names(value):
    if not value:
        return set()
    result = {part.strip() for part in value.split(",")}
    if "" in result:
        fail("Filters must contain nonempty comma-separated names")
    return result


def safe_name(value):
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", value
    ):
        fail(f"Invalid benchmark name: {value!r}")
    return value


def binary_record(binary):
    digest = hashlib.sha256()
    with binary.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(binary), "sha256": digest.hexdigest()}


def run_worker(binary, phase, log, *, report=None, **flags):
    command = [str(binary), f"--mode={phase}"]
    command.extend(f"--{key}={value}" for key, value in flags.items())
    with log.open("x") as stream:
        completed = subprocess.run(
            command, stdout=stream, stderr=subprocess.STDOUT, check=False
        )
    if completed.returncode:
        fail(f"{phase} exited with {completed.returncode}; see {log}")
    if report is None:
        return None
    if not report.is_file():
        fail(f"{phase} produced no report; see {log}")
    return load_json(report.read_text())


def number(value, label, positive=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        fail(f"Invalid metric {label}: {value!r}")
    if value < 0 or (positive and value == 0):
        fail(f"Invalid metric {label}: {value!r}")
    return value


def metric_value(report, key):
    value = report
    for component in key.split("."):
        if not isinstance(value, dict) or component not in value:
            fail(f"Missing metric: {key}")
        value = value[component]
    if value is not None and (
        type(value) not in (int, float) or not math.isfinite(value)
    ):
        fail(f"Invalid metric {key}: {value!r}")
    return value


def validate_metrics(report, keys):
    for key in keys:
        metric_value(report, key)


def summarize_metrics(samples, keys):
    result = {}
    for key in keys:
        values = [metric_value(sample, key) for sample in samples]
        result[key] = (
            statistics.mean(values)
            if all(value is not None for value in values)
            else None
        )
    return result


def report_details(report, metrics=()):
    details = copy.deepcopy(report)
    for key in (
        "name",
        "index_path",
        "environment",
    ):
        details.pop(key, None)
    for key in metrics:
        components = key.split(".")
        parent = details
        for component in components[:-1]:
            parent = parent.get(component, {})
        parent.pop(components[-1], None)
    return {key: value for key, value in details.items() if value != {}}
