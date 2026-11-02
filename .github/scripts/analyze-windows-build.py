"""Summarize actual Ninja command durations, separate from CTest execution."""

from __future__ import annotations

import csv
import json
import re
import statistics
import sys
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def ninja_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        fields = line.split("\t")
        if len(fields) != 5 or line.startswith("#"):
            continue
        start, end = map(int, fields[:2])
        records.append(
            {
                "start_ms": start,
                "end_ms": end,
                "duration_s": (end - start) / 1000,
                "output": fields[3],
                "hash": fields[4],
                "identity": tuple(fields),
            }
        )
    return records


def normalize(path: str) -> str:
    return path.replace("\\", "/").lower()


def compilation_commands(path: Path, source_root: str) -> tuple[dict, dict]:
    sources, flag_counts = {}, {}
    if not path.exists():
        return sources, flag_counts
    for entry in load_json(path):
        command = entry.get("command", " ".join(entry.get("arguments", [])))
        output = entry.get("output")
        if not output:
            match = re.search(r'(?:/Fo|-o\s+)(?:"([^"]+)"|(\S+))', command)
            output = (match.group(1) or match.group(2)) if match else None
        source = entry["file"].replace("\\", "/")
        prefix = source_root.replace("\\", "/").rstrip("/") + "/"
        if source.lower().startswith(prefix.lower()):
            source = source[len(prefix) :]
        if output:
            sources[normalize(output)] = source
            directory = normalize(entry.get("directory", "")).rstrip("/") + "/"
            if normalize(output).startswith(directory):
                sources[normalize(output)[len(directory) :]] = source
        for flag in ("/MT", "/MD", "/MP", "/O2", "/Ob2", "/showIncludes"):
            if re.search(
                r"(?:^|\s)" + re.escape(flag) + r"(?:d|\d+)?(?:\s|$)", command
            ):
                flag_counts[flag] = flag_counts.get(flag, 0) + 1
    return sources, flag_counts


def summarize(records: list[dict], sources: dict) -> dict:
    # A Ninja edge can have multiple outputs (relative and absolute stamps).
    # Count it once, rather than manufacturing extra concurrent commands.
    unique = {}
    for record in records:
        key = record["start_ms"], record["end_ms"], record["hash"]
        unique.setdefault(key, record)
    commands = list(unique.values())
    events = []
    for record in commands:
        if record["end_ms"] > record["start_ms"]:
            events.extend([(record["start_ms"], 1), (record["end_ms"], -1)])
    active = peak = 0
    for _, change in sorted(events):
        active += change
        peak = max(peak, active)
    compiles, links = [], []
    for record in commands:
        output = record["output"]
        item = {"output": output, "seconds": record["duration_s"]}
        if output.endswith((".obj", ".o")):
            item["source"] = sources.get(normalize(output), output)
            compiles.append(item)
        elif output.endswith((".exe", ".dll", ".pyd", ".lib", ".a")):
            links.append(item)
    return {
        "commands": len(commands),
        "compile_commands": len(compiles),
        "peak_ninja_edges": peak,
        "wall_s": max((r["end_ms"] for r in commands), default=0) / 1000,
        "compile_work_s": sum(item["seconds"] for item in compiles),
        "slowest_compiles": sorted(
            compiles, key=lambda item: item["seconds"], reverse=True
        )[:25],
        "slowest_links": sorted(links, key=lambda item: item["seconds"], reverse=True)[
            :15
        ],
    }


def resource_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    summary = {}
    for phase in ("pip-wheel", "test-compile"):
        subset = [row for row in rows if row["phase"] == phase]
        result = {"samples": len(subset)}
        for key in (
            "cpu_percent",
            "available_mb",
            "page_reads_sec",
            "disk_bytes_sec",
            "cl",
            "link",
            "ninja",
        ):
            values = [float(row[key]) for row in subset if row.get(key)]
            if values:
                result[key] = {
                    "mean": statistics.mean(values),
                    "min": min(values),
                    "max": max(values),
                }
        summary[phase] = result
    return summary


def analyze(case_dir: Path) -> dict:
    measurement = load_json(case_dir / "measurement.json")
    summary = {
        "measurement": measurement,
        "resources": resource_summary(case_dir / "resources.csv"),
    }
    for name in ("root", "arrow", "lz4"):
        before = ninja_records(case_dir / "main" / f"{name}-.ninja_log")
        after = ninja_records(case_dir / "tests" / f"{name}-.ninja_log")
        existing = {record["identity"] for record in before}
        test_records = [
            record for record in after if record["identity"] not in existing
        ]
        sources, flags = compilation_commands(
            case_dir / "tests" / f"{name}-compile_commands.json", measurement["source"]
        )
        summary[name] = {
            "main": summarize(before, sources),
            "test_compile": summarize(test_records, sources),
            "flags": flags,
        }
    output = case_dir / "summary.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(  # noqa: T201 - publish the benchmark summary in the Actions log
        json.dumps(
            {
                "case": measurement["case"],
                "status": measurement["status"],
                "pip_seconds": measurement.get("pip_seconds"),
                "test_compile_seconds": measurement.get("test_compile_seconds"),
                "slowest_tests": summary["root"]["test_compile"]["slowest_compiles"][
                    :8
                ],
            },
            indent=2,
        )
    )
    return summary


if __name__ == "__main__":
    analyze(Path(sys.argv[1]))
