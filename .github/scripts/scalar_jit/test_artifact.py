# Copyright 2025-present the zvec project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Portable artifact audit fixtures; no Linux tools or LLVM required."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "audit", Path(__file__).with_name("check_artifact.py")
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


@pytest.fixture
def elf(tmp_path, monkeypatch):
    path = tmp_path / "libzvec.so"
    path.write_bytes(b"ELF fixture")
    texts = {
        "readelf-d": "(NEEDED) Shared library: [libc.so.6]",
        "readelf--version-info": "GLIBC_2.28 GLIBCXX_3.4.24 CXXABI_1.3.11",
        "nm-D": "0000000 T zvec_query",
    }
    monkeypatch.setattr(audit, "output", lambda *cmd: texts[cmd[0] + cmd[1]])
    return path, texts


def test_good_elf(elf):
    path, _ = elf
    assert audit.check_elf(path)["sha256"] == audit.sha256(path)


@pytest.mark.parametrize(
    ("key", "text"),
    [
        ("readelf-d", "(NEEDED) Shared library: [libLLVM.so.20]"),
        ("readelf-d", "(NEEDED) Shared library: [libcrypto.so.3]"),
        ("readelf--version-info", "GLIBC_2.29"),
        ("readelf--version-info", "GLIBCXX_3.4.25"),
        ("readelf--version-info", "CXXABI_1.3.12"),
        ("nm-D", "000 T LLVMInitializeX86Target"),
        ("nm-D", "000 T _ZN4llvm11LLVMContextC1Ev"),
        ("nm-D", "000 T _ZTIN4llvm3FooE"),
    ],
)
def test_bad_elf(elf, key, text):
    path, texts = elf
    texts[key] = text
    with pytest.raises(AssertionError):
        audit.check_elf(path)


@pytest.mark.parametrize(
    "flag",
    [
        "-march=native",
        "-mtune=native",
        "-march=haswell",
        "-mavx2",
        "-mavx512f",
        "-mfma",
        "-Xclang",
        "@flags.rsp",
    ],
)
def test_host_flags_rejected(flag):
    with pytest.raises(AssertionError):
        audit.check_flags(["c++", flag], "/work/src/db/sqlengine/scalar_jit/runtime.cc")


def test_baseline_and_explicit_dispatch():
    assert not audit.check_flags(["-march=x86-64", "-msse4.2"], "/src/db/runtime.cc")[
        "dispatched"
    ]
    assert audit.check_flags(
        ["-march=haswell"], "/work/src/turbo/distance/avx2/distance.cc"
    )["dispatched"]
    assert audit.check_flags(
        ["-mavx512f"], "/work/cpp/src/arrow/util/bpacking_avx512.cc"
    )["dispatched"]
    with pytest.raises(AssertionError):
        audit.check_flags(["-mavx2"], "/work/src/turbo/distance/dispatch.cc")
    with pytest.raises(AssertionError):
        audit.check_flags(
            ["-march=native"], "/work/src/turbo/distance/avx2/distance.cc"
        )
    with pytest.raises(AssertionError):
        audit.check_flags(["-msse4.2"], "helper", helper=True)


@pytest.mark.parametrize(
    ("cpu", "features", "valid"),
    [
        ("x86-64", "+sse2,-avx", True),
        ("haswell", "+sse2", False),
        ("x86-64", "+avx2", False),
        ("x86-64", "+sse4.2", False),
        ("x86-64", "+aes", False),
        ("x86-64", "+pclmul", False),
        ("x86-64", "+lzcnt", False),
    ],
)
def test_helper_ir(tmp_path, monkeypatch, cpu, features, valid):
    bitcode = tmp_path / "helper.bc"
    bitcode.write_bytes(b"bitcode fixture")
    monkeypatch.setattr(
        audit,
        "output",
        lambda *_args: f'"target-cpu"="{cpu}" "target-features"="{features}"',
    )
    if valid:
        assert audit.check_bitcode(bitcode, "llvm-dis")["sha256"] == audit.sha256(
            bitcode
        )
    else:
        with pytest.raises(AssertionError):
            audit.check_bitcode(bitcode, "llvm-dis")


@pytest.fixture
def licensed_artifact(tmp_path):
    root = Path(__file__).resolve().parents[3] / "thirdparty/arrow/gandiva"
    gandiva = tmp_path / "gandiva"
    gandiva.mkdir()
    provenance = (root / "PROVENANCE.json").read_bytes()
    (gandiva / "PROVENANCE.json").write_bytes(provenance)
    for name, metadata in audit.json.loads(provenance)["licenses"].items():
        (gandiva / name).write_bytes((root / metadata["source"]).read_bytes())
    toolchain = tmp_path / "jit-toolchain"
    toolchain.mkdir()
    for name in ("LLVM-LICENSE.txt", "OpenSSL-LICENSE.txt"):
        (toolchain / name).write_text("license fixture")
    return tmp_path


def test_licenses_required(tmp_path):
    with pytest.raises(AssertionError):
        audit.check_licenses(tmp_path)


def test_complete_licenses(licensed_artifact):
    assert len(audit.check_licenses(licensed_artifact)) == 5


@pytest.mark.parametrize("name", ["LICENSE", "NOTICE"])
def test_truncated_notices_rejected(licensed_artifact, name):
    notice = licensed_artifact / "gandiva" / name
    notice.write_bytes(notice.read_bytes()[:128])
    with pytest.raises(AssertionError, match="Changed or truncated notice"):
        audit.check_licenses(licensed_artifact)


def test_report_written_on_failure(tmp_path, monkeypatch):
    report = tmp_path / "audit.json"
    monkeypatch.setattr(
        audit.sys,
        "argv",
        ["audit", str(tmp_path / "missing.whl"), "--report", str(report)],
    )
    with pytest.raises(FileNotFoundError):
        audit.main()
    assert audit.json.loads(report.read_text())["passed"] is False


def test_maintained_extension_provenance():
    root = Path(__file__).resolve().parents[3] / "thirdparty/arrow/gandiva"
    provenance = audit.json.loads((root / "PROVENANCE.json").read_text())
    for name, expected in provenance["sha256"].items():
        assert audit.sha256(root / name) == expected, name
    for name, metadata in provenance["licenses"].items():
        assert audit.sha256(root / metadata["source"]) == metadata["sha256"], name


@pytest.mark.parametrize("host_flags", [False, True])
def test_effective_build_commands(tmp_path, monkeypatch, host_flags):
    arrow = tmp_path / "thirdparty/arrow/arrow/src/ARROW.BUILD-build"
    arrow.mkdir(parents=True)
    for directory, source in (
        (tmp_path, "/work/src/db/sqlengine/scalar_jit/runtime.cc"),
        (arrow, "/work/cpp/src/gandiva/engine.cc"),
    ):
        flag = "-march=native" if host_flags else "-march=x86-64"
        (directory / "compile_commands.json").write_text(
            audit.json.dumps([{"file": source, "arguments": ["c++", flag]}])
        )
    bitcode = arrow / "src/gandiva/irhelpers.bc"
    bitcode.parent.mkdir(parents=True)
    bitcode.write_bytes(b"bitcode fixture")
    monkeypatch.setattr(
        audit,
        "output",
        lambda *cmd: (
            "clang -march=x86-64 -emit-llvm helper.cc"
            if cmd[0] == "ninja"
            else '"target-cpu"="x86-64" "target-features"="+sse2,-avx"'
        ),
    )
    if host_flags:
        with pytest.raises(AssertionError):
            audit.check_build(tmp_path, "llvm-dis")
    else:
        evidence = audit.check_build(tmp_path, "llvm-dis")
        assert len(evidence["commands"]) == 2
        assert len(evidence["helpers"]) == 1
