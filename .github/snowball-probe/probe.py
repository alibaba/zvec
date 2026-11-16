import hashlib
import json
import pathlib
import subprocess


root = pathlib.Path.cwd()
results = root / "snowball-probe-results"
results.mkdir(exist_ok=True)
summary = {}
reference = None
for mode in ["Od", "O1", "O2", "analyser-Od", "tokeniser-Od", "space-Od"]:
    print(f"::group::Snowball {mode}", flush=True)
    build = results / mode
    configure = subprocess.run(
        [
            "cmake", "-S", str(root / ".github/snowball-probe"),
            "-B", str(build), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
            f"-DPROBE_MODE={mode}",
        ],
        capture_output=True, text=True,
    )
    print(configure.stdout, configure.stderr, flush=True)
    if configure.returncode:
        raise RuntimeError(f"Configuration failed: {mode}")
    compiled = subprocess.run(
        ["cmake", "--build", str(build), "--target", "snowball_codegen", "--parallel", "4"],
        capture_output=True, text=True,
    )
    (results / f"{mode}.log").write_text(compiled.stdout + compiled.stderr)
    print(compiled.stdout, compiled.stderr, flush=True)
    files = {}
    for path in sorted((build / "snowball/snowball-codegen/src_c").glob("*")):
        content = path.read_bytes().replace(b"\r\n", b"\n")
        files[path.name] = hashlib.sha256(content).hexdigest()
    if mode == "Od" and compiled.returncode == 0:
        reference = files
    summary[mode] = {
        "exit_code": compiled.returncode,
        "generated_files": len(files),
        "matches_Od": files == reference,
        "hashes": files,
    }
    print(f"RESULT {mode}: exit={compiled.returncode}, files={len(files)}, matches_Od={files == reference}")
    print("::endgroup::", flush=True)
(results / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps({k: {n: v for n, v in value.items() if n != "hashes"} for k, value in summary.items()}, indent=2))
if reference is None:
    raise RuntimeError("Even the unoptimized reference failed")
