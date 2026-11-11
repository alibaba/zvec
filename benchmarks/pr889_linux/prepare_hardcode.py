#!/usr/bin/env python3
"""Build the restricted hardcoded comparator without modifying production files.

Recompiles exactly the two changed implementation objects, replaces them in a
private copy of the packed DB archive, then replays the normal benchmark link.
All original artifacts remain unchanged. Requires a finished JIT ON build.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--build", required=True, type=Path)
p.add_argument("--output", required=True, type=Path)
p.add_argument("--ninja", default="ninja")
a = p.parse_args()
root = Path(__file__).resolve().parents[2]
build, output = a.build.resolve(), a.output.resolve()
if output.exists():
    raise RuntimeError(f"Refusing to overwrite {output}")
output.mkdir(parents=True)
overlay = output / "source"
files = [
    "src/db/sqlengine/planner/scalar_doc_filter.h",
    "src/db/sqlengine/planner/scalar_doc_filter.cc",
    "src/db/sqlengine/planner/doc_filter.cc",
]
for name in files:
    destination = overlay / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / name, destination)
patch = root / "benchmarks/pr889_linux/hardcode.patch"
subprocess.run(["git", "apply", "--check", str(patch)], cwd=overlay, check=True)
subprocess.run(["git", "apply", str(patch)], cwd=overlay, check=True)
commands = []
entries = json.loads((build / "compile_commands.json").read_text())
objects = []
for name in files[1:]:
    candidates = [
        c
        for c in entries
        if c["file"] == str(root / name) and "/CMakeFiles/zvec.dir/" in c["output"]
    ]
    assert len(candidates) == 1, candidates
    original = candidates[0]
    cmd = shlex.split(original["command"])
    assert "-DZVEC_ENABLE_SCALAR_JIT=1" in cmd, "JIT ON build required"
    destination = output / Path(original["output"]).name
    cmd[cmd.index("-o") + 1] = str(destination)
    cmd[cmd.index("-c") + 1] = str(overlay / name)
    cmd[1:1] = ["-DPR889_BENCH_HARDCODE=1", "-I" + str(overlay / "src")]
    commands.append(
        {
            "operation": "compile",
            "original": original["command"],
            "argv": cmd,
            "cwd": str(build),
        }
    )
    with (output / (destination.name + ".log")).open("w") as log:
        subprocess.run(cmd, cwd=build, stdout=log, stderr=subprocess.STDOUT, check=True)
    objects.append(destination)
archive = output / "libzvec_hardcode.a"
original_archive = build / "lib/libzvec.a"
shutil.copy2(original_archive, archive)
members = subprocess.check_output(["ar", "t", str(archive)], text=True).splitlines()
for obj in objects:
    assert members.count(obj.name) == 1, (obj.name, members.count(obj.name))
cmd = ["ar", "r", str(archive)] + [str(obj) for obj in objects]
commands.append({"operation": "archive_replace", "argv": cmd})
subprocess.run(cmd, check=True)
link_lines = subprocess.check_output(
    [a.ninja, "-C", str(build), "-t", "commands", "pr889_filter_bench"], text=True
).splitlines()
line = next(
    line for line in reversed(link_lines) if " -o bin/pr889_filter_bench " in line
)
cmd = shlex.split(line)
if cmd[:2] == [":", "&&"]:
    cmd = cmd[2:]
if cmd[-2:] == ["&&", ":"]:
    cmd = cmd[:-2]
assert "&&" not in cmd, line
binary = output / "pr889_filter_bench_hardcode"
cmd[cmd.index("-o") + 1] = str(binary)
cmd = [
    x.replace(str(original_archive), str(archive)).replace(
        "lib/libzvec.a", str(archive)
    )
    for x in cmd
]
assert any(str(archive) in x for x in cmd), line
commands.append({"operation": "link", "original": line, "argv": cmd, "cwd": str(build)})
with (output / "link.log").open("w") as log:
    subprocess.run(cmd, cwd=build, stdout=log, stderr=subprocess.STDOUT, check=True)
hash_file = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest = {
    "commands": commands,
    "patch_sha256": hash_file(patch),
    "original_archive_sha256": hash_file(original_archive),
    "archive_sha256": hash_file(archive),
    "binary_sha256": hash_file(binary),
    "source_sha256": {name: hash_file(overlay / name) for name in files},
    "note": "Two-object restricted hardcode comparator. Same ScalarDocFilter layout, snapshot lifetime and row routing; original production archives unchanged.",
}
(output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(binary)  # noqa: T201 - CLI output
