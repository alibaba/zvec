"""Check portable ELF artifacts and the baseline of Gandiva's embedded IR."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def output(*command: str) -> str:
    return subprocess.check_output(command, text=True)


def check_elf(path: Path) -> dict:
    dynamic = output("readelf", "-d", str(path))
    needed = re.findall(r"\(NEEDED\).*\[(.*?)\]", dynamic)
    forbidden = re.compile(r"lib(?:LLVM|clang|ssl|crypto|zstd|ffi|xml2|tinfo|edit)")
    assert not any(forbidden.match(name) for name in needed), (path, needed)
    versions = output("readelf", "--version-info", str(path))
    # Match auditwheel's manylinux_2_28 x86_64 numeric ABI ceilings. SDKs do
    # not pass through wheel repair, so check C++ ABI versions here as well.
    maxima = {}
    for namespace, ceiling in (
        ("GLIBC", (2, 28)),
        ("GLIBCXX", (3, 4, 24)),
        ("CXXABI", (1, 3, 11)),
    ):
        found = [
            tuple(map(int, version.split(".")))
            for version in re.findall(rf"\b{namespace}_([0-9]+(?:\.[0-9]+)+)", versions)
        ]
        maximum = max(found) if found else None
        assert maximum is None or maximum <= ceiling, (path, namespace, maximum)
        maxima[f"max_{namespace.lower()}"] = (
            ".".join(map(str, maximum)) if maximum else None
        )
    exports = output("nm", "-D", "--defined-only", str(path))
    assert not re.search(
        r"\b(?:LLVM[A-Za-z_]|llvm_[A-Za-z_]|_ZNK?4llvm|_ZT[VIS]N4llvm)", exports
    ), path
    return {"path": path.name, "sha256": sha256(path), "needed": needed, **maxima}


def check_directory(path: Path) -> list[dict]:
    files = [
        p
        for p in path.rglob("*")
        if p.is_file() and not p.is_symlink() and ".so" in p.name
    ]
    assert files, f"No ELF shared libraries in {path}"
    return [check_elf(p) for p in files]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_licenses(path: Path) -> list[str]:
    required = (
        "gandiva/LICENSE",
        "gandiva/NOTICE",
        "gandiva/PROVENANCE.json",
        "jit-toolchain/LLVM-LICENSE.txt",
        "jit-toolchain/OpenSSL-LICENSE.txt",
    )
    files = [p for p in path.rglob("*") if p.is_file()]
    for suffix in required:
        assert any(p.as_posix().endswith(suffix) and p.stat().st_size for p in files), (
            suffix
        )
    return list(required)


# Only known runtime-dispatched translation units are exempt from the baseline.
# Never exempt a dispatcher or an entire third-party library.
DISPATCHED = re.compile(
    r"(?:/src/turbo/(?:distance/(?:sse2|avx2|avx512|avx512_vnni|avx512_fp16)/[^ ]+|"
    r"conversion/avx512/[^ ]+|distance/sparse/sparse_inner_product_fp(?:32_sse|16_avx|16_avx512fp16)\.cc|"
    r"topk/batch_topk_(?:sse2|avx2|avx512)\.cc)|"
    r"/src/db/index/column/fts_column/posting/bitpacked_simd_(?:sse41|avx2)\.cc|"
    r"/cpp/src/arrow/(?:acero/(?:bloom_filter|swiss_join)_avx2\.cc|"
    r"util/bpacking_avx(?:2|512)\.cc|compute/(?:key_(?:hash|map)_internal_avx2|"
    r"row/(?:compare|encode)_internal_avx2|util_avx2)\.cc))$"
)


def check_flags(tokens: list[str], source: str, helper: bool = False) -> dict:
    assert not any(t.startswith("@") for t in tokens), "Unexpanded response file"
    dispatched = bool(DISPATCHED.search(source)) and not helper
    flags = [
        t
        for t in tokens
        if t.startswith(("-m", "/arch:", "-Xclang", "-target-cpu", "-target-feature"))
    ]
    baseline = {"-m64", "-march=x86-64", "-mtune=generic", "-msse", "-msse2"}
    if not helper:
        baseline |= {"-msse3", "-mssse3", "-msse4.1", "-msse4.2", "-mpopcnt"}
    for flag in flags:
        assert "native" not in flag, (source, flag)
        assert dispatched or flag in baseline or flag.startswith("-mno-"), (
            source,
            flag,
        )
    return {"source": source, "flags": flags, "dispatched": dispatched}


def check_bitcode(bitcode: Path, llvm_dis: str) -> dict:
    ir = output(llvm_dis, str(bitcode), "-o", "-")
    cpus = set(re.findall(r'"target-cpu"="([^"]+)"', ir))
    features = set(re.findall(r'"target-features"="([^"]+)"', ir))
    assert cpus == {"x86-64"}, cpus
    baseline = {"+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"}
    assert features, "Missing helper target features"
    assert all(
        feature in baseline or feature.startswith("-")
        for attributes in features
        for feature in attributes.split(",")
    ), features
    return {
        "bitcode": str(bitcode),
        "sha256": sha256(bitcode),
        "target_cpus": sorted(cpus),
        "target_features": sorted(features),
    }


def check_build(build: Path, llvm_dis: str) -> dict:
    arrow = build / "thirdparty/arrow/arrow/src/ARROW.BUILD-build"
    evidence = []
    for database in (build / "compile_commands.json", arrow / "compile_commands.json"):
        records = json.loads(database.read_text())
        assert records, database
        for record in records:
            source = Path(record["file"]).as_posix()
            # Test/probe binaries are not shipped runtime code.
            if "/tests/" in source or "/test/" in source or "/CMakeFiles/" in source:
                continue
            tokens = record.get("arguments") or shlex.split(record["command"])
            evidence.append(check_flags(tokens, source))
    assert any(e["source"].endswith("/scalar_jit/runtime.cc") for e in evidence)
    assert any(e["source"].endswith("/gandiva/engine.cc") for e in evidence)
    commands = output(
        "ninja", "-C", str(arrow), "-t", "commands", "src/gandiva/irhelpers.bc"
    )
    helpers = [
        check_flags(shlex.split(line), "gandiva-helper", helper=True)
        for line in commands.splitlines()
        if "-emit-llvm" in line
    ]
    assert helpers
    assert all("-march=x86-64" in h["flags"] for h in helpers)
    return {
        "build": str(build),
        "commands": evidence,
        "helpers": helpers,
        "bitcode": check_bitcode(arrow / "src/gandiva/irhelpers.bc", llvm_dis),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", nargs="*", type=Path)
    parser.add_argument("--bitcode", type=Path)
    parser.add_argument("--llvm-dis", default="llvm-dis")
    parser.add_argument("--build-directory", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    assert args.artifacts or args.bitcode or args.build_directory, (
        "Supply an artifact or build"
    )
    report = {"passed": False, "checks": []}
    results = report["checks"]
    try:
        for path in args.artifacts:
            if path.suffix == ".whl":
                with tempfile.TemporaryDirectory() as directory:
                    with zipfile.ZipFile(path) as wheel:
                        wheel.extractall(directory)
                    results.append(
                        {
                            "artifact": path.name,
                            "sha256": sha256(path),
                            "licenses": check_licenses(Path(directory)),
                            "libraries": check_directory(Path(directory)),
                        }
                    )
            elif path.is_dir():
                results.append(
                    {
                        "sdk": str(path),
                        "licenses": check_licenses(path),
                        "libraries": check_directory(path),
                    }
                )
            else:
                results.append(check_elf(path))
        if args.bitcode:
            results.append(check_bitcode(args.bitcode, args.llvm_dis))
        if args.build_directory:
            results.append(check_build(args.build_directory, args.llvm_dis))
        report["passed"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n")
    sys.stdout.write(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
