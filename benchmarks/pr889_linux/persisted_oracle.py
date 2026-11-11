#!/usr/bin/env python3
"""57 independent persisted mmap/HNSW correctness queries (19 filters x 3 vectors).

This recreates coverage, not the unavailable macOS oracle's exact dataset.
The Python oracle evaluates predicates independently and computes exhaustive
FP32 squared-L2 ranks. ef=1024 exceeds all 512 rows. Run in separate fresh
processes against each installed wheel with JIT enabled/disabled.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
from pathlib import Path

import zvec

p = argparse.ArgumentParser()
p.add_argument("--path", required=True, type=Path)
p.add_argument("--output", required=True, type=Path)
p.add_argument("--debug", action="store_true")
a = p.parse_args()
zvec.init(
    query_threads=8,
    optimize_threads=8,
    invert_to_forward_scan_ratio=0.5,
    log_level=zvec.LogLevel.DEBUG if a.debug else zvec.LogLevel.WARN,
)

f32 = lambda x: struct.unpack("f", struct.pack("f", x))[0]
rows = []
for i in range(512):
    rows.append(
        {
            "id": str(i),
            "vector": [i / 512, (i % 17) / 32, (i % 5) / 8, 1.0],
            "fields": {
                "pid": i,
                "x": None if i % 17 == 0 else i % 512,
                "flag": None if i % 11 == 0 else bool(i % 2),
                "tag": None if i % 13 == 0 else f"group{i % 8}",
                "arr": None if i % 19 == 0 else list(range(i % 5)),
            },
        }
    )

if not a.path.exists():
    schema = zvec.CollectionSchema(
        name="pr889_oracle",
        fields=[
            zvec.FieldSchema("pid", zvec.DataType.INT64),
            zvec.FieldSchema(
                "x",
                zvec.DataType.INT32,
                nullable=True,
                index_param=zvec.InvertIndexParam(enable_range_optimization=True),
            ),
            zvec.FieldSchema("flag", zvec.DataType.BOOL, nullable=True),
            zvec.FieldSchema(
                "tag",
                zvec.DataType.STRING,
                nullable=True,
                index_param=zvec.InvertIndexParam(enable_range_optimization=True),
            ),
            zvec.FieldSchema("arr", zvec.DataType.ARRAY_INT32, nullable=True),
        ],
        vectors=[
            zvec.VectorSchema(
                "vector",
                zvec.DataType.VECTOR_FP32,
                dimension=4,
                index_param=zvec.HnswIndexParam(
                    metric_type=zvec.MetricType.L2, m=16, ef_construction=200
                ),
            )
        ],
    )
    coll = zvec.create_and_open(str(a.path), schema)
    for start in range(0, len(rows), 512):
        statuses = coll.insert(
            [
                zvec.Doc(
                    id=r["id"], fields=r["fields"], vectors={"vector": r["vector"]}
                )
                for r in rows[start : start + 512]
            ]
        )
        if statuses is not None:
            for status in statuses:
                if hasattr(status, "ok"):
                    ok = status.ok() if callable(status.ok) else status.ok
                    assert ok, status
    coll.optimize()
    coll.close()
coll = zvec.open(str(a.path), zvec.CollectionOption(read_only=True, enable_mmap=True))

nonnull = lambda v, fn: v is not None and fn(v)
filters = [
    ("x > 64", lambda r: nonnull(r["x"], lambda v: v > 64)),
    ("x <= 400", lambda r: nonnull(r["x"], lambda v: v <= 400)),
    ("x >= 128 AND x < 384", lambda r: nonnull(r["x"], lambda v: 128 <= v < 384)),
    ("x = 127", lambda r: r["x"] == 127),
    ("x != 127", lambda r: nonnull(r["x"], lambda v: v != 127)),
    ("x IS NULL", lambda r: r["x"] is None),
    ("x IS NOT NULL", lambda r: r["x"] is not None),
    ("x > 400 OR x IS NULL", lambda r: r["x"] is None or r["x"] > 400),
    ("flag = true", lambda r: r["flag"] is True),
    ("flag = false", lambda r: r["flag"] is False),
    ("flag IS NULL", lambda r: r["flag"] is None),
    ("tag = 'group1'", lambda r: r["tag"] == "group1"),
    ("tag != 'group1'", lambda r: nonnull(r["tag"], lambda v: v != "group1")),
    ("tag > 'group3'", lambda r: nonnull(r["tag"], lambda v: v > "group3")),
    ("tag LIKE 'group%'", lambda r: nonnull(r["tag"], lambda v: v.startswith("group"))),
    ("x IN (31, 63, 127, 255, 511)", lambda r: r["x"] in (31, 63, 127, 255, 511)),
    (
        "x NOT IN (31, 63, 127, 255, 511)",
        lambda r: nonnull(r["x"], lambda v: v not in (31, 63, 127, 255, 511)),
    ),
    (
        "tag IN ('group1', 'group3', 'group5')",
        lambda r: r["tag"] in ("group1", "group3", "group5"),
    ),
    ("array_length(arr) >= 2", lambda r: nonnull(r["arr"], lambda v: len(v) >= 2)),
]
query_vectors = [
    [f32(x) for x in q]
    for q in [
        [0.12345, 0.25, 0.5, 1.0],
        [0.66789, 0.125, 0.25, 1.0],
        [0.912345, 0.375, 0.0, 1.0],
    ]
]
results = []
for text, predicate in filters:
    for qi, query in enumerate(query_vectors):
        expected = sorted(
            (
                sum((x - y) ** 2 for x, y in zip(r["vector"], query, strict=True)),
                int(r["id"]),
            )
            for r in rows
            if predicate(r["fields"])
        )[:10]
        assert all(
            expected[i][0] != expected[i - 1][0] for i in range(1, len(expected))
        ), "Oracle has tied top-10 scores"
        docs = coll.query(
            zvec.Query(
                field_name="vector", vector=query, param=zvec.HnswQueryParam(ef=1024)
            ),
            topk=10,
            filter=text,
        )
        assert [int(d.id) for d in docs] == [i for _, i in expected], (
            text,
            qi,
            [d.id for d in docs],
            expected,
        )
        for doc, (score, rid) in zip(docs, expected, strict=True):
            assert math.isclose(doc.score, score, rel_tol=2e-5, abs_tol=2e-6), (
                text,
                doc.score,
                score,
            )
            expected_fields = rows[rid]["fields"]
            assert set(doc.fields).issubset(expected_fields), (text, doc.fields)
            normalized_fields = {key: doc.fields.get(key) for key in expected_fields}
            assert normalized_fields == expected_fields, (
                text,
                normalized_fields,
                expected_fields,
            )
        results.append(
            {
                "filter": text,
                "query": qi,
                "matches": sum(predicate(r["fields"]) for r in rows),
                "ids": [d.id for d in docs],
                "scores": [d.score for d in docs],
            }
        )
coll.close()
assert len(results) == 57
obj = {
    "passed": True,
    "queries": 57,
    "filters": 19,
    "rows": 512,
    "vector_dimension": 4,
    "metric": "L2",
    "quantization": "UNDEFINED",
    "ef": 1024,
    "topk": 10,
    "read_only": True,
    "enable_mmap": True,
    "jit_env": os.environ.get("ZVEC_SCALAR_JIT", "unset"),
    "zvec_module": zvec.__file__,
    "results": results,
}
a.output.write_text(json.dumps(obj, indent=2) + "\n")
print(json.dumps({k: v for k, v in obj.items() if k != "results"}))  # noqa: T201 - CLI output
