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

"""Read-only opens preserve files, including after an unclean shutdown."""

import contextlib
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import zvec


@pytest.fixture(scope="module", autouse=True)
def initialize():
    zvec.init(memory_limit_mb=512, query_threads=2, optimize_threads=2)


def _schema(kind="flat"):
    fields = [
        zvec.FieldSchema(
            "value", zvec.DataType.INT64, index_param=zvec.InvertIndexParam()
        ),
        zvec.FieldSchema(
            "text", zvec.DataType.STRING, index_param=zvec.FtsIndexParam()
        ),
    ]
    vectors = []
    if kind != "scalar":
        sparse = kind == "sparse"
        param_type = zvec.FlatIndexParam if kind == "flat" else zvec.HnswIndexParam
        param = param_type(
            metric_type=zvec.MetricType.IP if sparse else zvec.MetricType.L2
        )
        vectors = [
            zvec.VectorSchema(
                "v",
                zvec.DataType.SPARSE_VECTOR_FP32
                if sparse
                else zvec.DataType.VECTOR_FP32,
                index_param=param,
                **({} if sparse else {"dimension": 4}),
            )
        ]
    return zvec.CollectionSchema(
        name="readonly_storage", fields=fields, vectors=vectors
    )


def _doc(key, value, kind):
    vectors = (
        {}
        if kind == "scalar"
        else {"v": {value: 1.0} if kind == "sparse" else [float(value), 1.0, 0.0, 0.0]}
    )
    return zvec.Doc(
        id=key, fields={"value": value, "text": "hello readonly"}, vectors=vectors
    )


def _snapshot(path):
    return {
        str(p.relative_to(path)): (
            p.read_bytes(),
            p.stat().st_ino,
            p.stat().st_mtime_ns,
        )
        for p in path.rglob("*")
        if p.is_file()
    }


def _crash_writer(path, enable_mmap, kind, operation):
    # A fresh process using os._exit leaves a WAL without running destructors.
    subprocess.run(
        [sys.executable, __file__, str(path), str(int(enable_mmap)), kind, operation],
        check=True,
        timeout=60,
    )


def _assert_docs(collection, expected):
    fetched = collection.fetch(["a", "b", "c"])
    assert {key: doc.fields["value"] for key, doc in fetched.items()} == expected
    assert collection.stats.doc_count == len(expected)
    assert {doc.id for doc in collection.iter_docs()} == set(expected)


@pytest.mark.parametrize("enable_mmap", [True, False])
@pytest.mark.parametrize("kind", ["flat", "sparse", "scalar"])
@pytest.mark.parametrize("operation", ["insert", "update", "upsert", "delete"])
def test_pending_wal_requires_writable_recovery(tmp_path, enable_mmap, kind, operation):
    path = tmp_path / "collection"
    _crash_writer(path, enable_mmap, kind, operation)
    before = _snapshot(path)
    assert list(path.rglob("*.wal"))
    for _ in range(2):
        with pytest.raises(
            (ValueError, RuntimeError), match="requires WAL recovery.*read_only=false"
        ):
            zvec.open(
                str(path),
                zvec.CollectionOption(read_only=True, enable_mmap=enable_mmap),
            )
        assert _snapshot(path) == before

    expected = {"a": 1, "b": 2}
    if operation == "insert":
        expected["c"] = 3
    elif operation in ("update", "upsert"):
        expected["a"] = 4
    else:
        del expected["a"]
    writer = zvec.open(str(path), zvec.CollectionOption(enable_mmap=enable_mmap))
    _assert_docs(writer, expected)
    writer.close()
    recovered = _snapshot(path)
    reader = zvec.open(
        str(path), zvec.CollectionOption(read_only=True, enable_mmap=enable_mmap)
    )
    _assert_docs(reader, expected)
    reader.close()
    assert _snapshot(path) == recovered


@pytest.mark.parametrize("enable_mmap", [True, False])
def test_header_only_wal_can_be_opened_readonly(tmp_path, enable_mmap):
    path = tmp_path / "collection"
    _crash_writer(path, enable_mmap, "flat", "empty")
    wal = next(path.rglob("*.wal"))
    assert wal.stat().st_size == 64
    before = _snapshot(path)
    reader = zvec.open(
        str(path), zvec.CollectionOption(read_only=True, enable_mmap=enable_mmap)
    )
    _assert_docs(reader, {"a": 1, "b": 2})
    reader.close()
    assert _snapshot(path) == before


@contextlib.contextmanager
def _read_only_tree(path):
    paths = [path, *path.rglob("*")]
    original = {p: stat.S_IMODE(p.stat().st_mode) for p in paths}
    try:
        for p in paths:
            p.chmod(0o555 if p.is_dir() else 0o444)
        yield
    finally:
        for p, mode in original.items():
            p.chmod(mode)


permissions_not_enforced = os.name == "nt" or (
    hasattr(os, "geteuid") and os.geteuid() == 0
)


@pytest.mark.skipif(
    permissions_not_enforced, reason="requires POSIX permissions and a non-root user"
)
@pytest.mark.parametrize("enable_mmap", [True, False])
@pytest.mark.parametrize("kind", ["flat", "hnsw", "sparse", "scalar"])
def test_readonly_files_support_queries(tmp_path, enable_mmap, kind):
    path = tmp_path / "collection"
    writer = zvec.create_and_open(
        str(path), _schema(kind), zvec.CollectionOption(enable_mmap=enable_mmap)
    )
    assert all(s.ok() for s in writer.insert([_doc("a", 1, kind), _doc("b", 2, kind)]))
    writer.optimize()
    writer.close()
    before = _snapshot(path)
    with _read_only_tree(path):
        reader = zvec.open(
            str(path), zvec.CollectionOption(read_only=True, enable_mmap=enable_mmap)
        )
        _assert_docs(reader, {"a": 1, "b": 2})
        assert reader.query(filter="value = 1", topk=1)[0].id == "a"
        assert (
            len(
                reader.query(
                    zvec.Query("text", fts=zvec.Fts(match_string="hello")), topk=2
                )
            )
            == 2
        )
        if kind != "scalar":
            vector = {1: 1.0} if kind == "sparse" else [1.0, 1.0, 0.0, 0.0]
            assert reader.query(zvec.Query("v", vector=vector), topk=1)[0].id == "a"
        reader.close()
    assert _snapshot(path) == before


@pytest.mark.skipif(
    permissions_not_enforced, reason="requires POSIX permissions and a non-root user"
)
@pytest.mark.parametrize("operation", ["insert", "empty"])
def test_wal_inspection_does_not_require_write_permission(tmp_path, operation):
    path = tmp_path / "collection"
    _crash_writer(path, True, "flat", operation)
    before = _snapshot(path)
    with _read_only_tree(path):
        if operation == "insert":
            with pytest.raises(
                (ValueError, RuntimeError), match="requires WAL recovery"
            ):
                zvec.open(str(path), zvec.CollectionOption(read_only=True))
        else:
            reader = zvec.open(str(path), zvec.CollectionOption(read_only=True))
            _assert_docs(reader, {"a": 1, "b": 2})
            reader.close()
    assert _snapshot(path) == before


if __name__ == "__main__":
    path, mmap, kind, operation = sys.argv[1:]
    zvec.init(memory_limit_mb=512, query_threads=2, optimize_threads=2)
    collection = zvec.create_and_open(
        path, _schema(kind), zvec.CollectionOption(enable_mmap=bool(int(mmap)))
    )
    assert all(
        s.ok() for s in collection.insert([_doc("a", 1, kind), _doc("b", 2, kind)])
    )
    collection.flush()
    if operation == "insert":
        assert collection.insert(_doc("c", 3, kind)).ok()
    elif operation in ("update", "upsert"):
        assert getattr(collection, operation)(_doc("a", 4, kind)).ok()
    elif operation == "delete":
        assert collection.delete("a").ok()
    elif operation == "empty":
        # Simulate an interrupted first append while preserving the WAL header.
        assert collection.insert(_doc("c", 3, kind)).ok()
        wal = next(Path(path).rglob("*.wal"))
        wal.write_bytes(wal.read_bytes()[:64])
    os._exit(0)
