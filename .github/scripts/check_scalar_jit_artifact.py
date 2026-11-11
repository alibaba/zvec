"""Check portable ELF artifacts and the baseline of Gandiva's embedded IR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile
import zipfile


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
    for namespace, ceiling in (("GLIBC", (2, 28)), ("GLIBCXX", (3, 4, 24)),
                               ("CXXABI", (1, 3, 11))):
        found = [tuple(map(int, version.split("."))) for version in
                 re.findall(rf"\b{namespace}_([0-9]+(?:\.[0-9]+)+)", versions)]
        maximum = max(found) if found else None
        assert maximum is None or maximum <= ceiling, (path, namespace, maximum)
        maxima[f"max_{namespace.lower()}"] = (
            ".".join(map(str, maximum)) if maximum else None
        )
    exports = output("nm", "-D", "--defined-only", str(path))
    assert not re.search(r"\b(?:LLVM[A-Za-z_]|llvm_[A-Za-z_]|_ZNK?4llvm|_ZT[VIS]N4llvm)", exports), path
    return {"path": path.name, "needed": needed, **maxima}


def check_directory(path: Path) -> list[dict]:
    files = [p for p in path.rglob("*") if p.is_file() and not p.is_symlink()
             and ".so" in p.name]
    assert files, f"No ELF shared libraries in {path}"
    return [check_elf(p) for p in files]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", nargs="*", type=Path)
    parser.add_argument("--bitcode", type=Path)
    parser.add_argument("--llvm-dis", default="llvm-dis")
    args = parser.parse_args()
    assert args.artifacts or args.bitcode, "Supply an artifact or --bitcode"
    results = []
    for path in args.artifacts:
        if path.suffix == ".whl":
            with tempfile.TemporaryDirectory() as directory:
                with zipfile.ZipFile(path) as wheel:
                    wheel.extractall(directory)
                results.extend(check_directory(Path(directory)))
        elif path.is_dir():
            results.extend(check_directory(path))
        else:
            results.append(check_elf(path))
    if args.bitcode:
        ir = output(args.llvm_dis, str(args.bitcode), "-o", "-")
        cpus = set(re.findall(r'"target-cpu"="([^"]+)"', ir))
        features = set(re.findall(r'"target-features"="([^"]+)"', ir))
        assert cpus == {"x86-64"}, cpus
        assert not any(re.search(r"\+(?:avx|fma|f16c|bmi)", f) for f in features), features
        results.append({"bitcode": str(args.bitcode), "target_cpus": sorted(cpus),
                        "target_features": sorted(features)})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
