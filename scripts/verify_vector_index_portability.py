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

"""Validate migrated indexes against independently built destination indexes.

Run with the destination's zvec extension on PYTHONPATH. Both platforms use the
same .npy inputs in ROOT/datasets. Each invocation owns one case and saves all
findings, including failures. See --help for the three phases. Linux source and
return operations use its native temporary filesystem, not the shared mount.
Exit status reflects all checks; execution_success in JSON separately records
whether data operations completed, even when a local-reference comparison fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import tempfile
import traceback
from pathlib import Path

import numpy as np

import zvec

CASES = [
    {
        "name": f"{alg}_{quant}",
        "algorithm": alg,
        "quant": quant,
        "count": 20000 if alg == "hnsw" and quant == "fp32" else 4096,
        "dimension": 1024 if alg == "hnsw" and quant == "fp32" else 128,
    }
    for alg in ("flat", "hnsw", "ivf", "vamana")
    for quant in ("fp32", "fp16", "int8", "int4")
] + [
    {
        "name": "diskann_fp32",
        "algorithm": "diskann",
        "quant": "fp32",
        "count": 4096,
        "dimension": 128,
    },
    {
        "name": "flat_sparse",
        "algorithm": "flat_sparse",
        "quant": "fp32",
        "count": 4096,
        "dimension": 768,
    },
    {
        "name": "hnsw_sparse",
        "algorithm": "hnsw_sparse",
        "quant": "fp32",
        "count": 20000,
        "dimension": 768,
    },
]
SCORE_ATOL = 2e-5
# A declared smoke-test quality budget, not an assertion of identical ANN graphs.
# Full-scan queries and same-index reopens still require matching IDs/scores.
ANN_RECALL_DROP = 0.02


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def manifest(path):
    result = {}
    for f in sorted(path.rglob("*")):
        if f.is_file():
            digest = hashlib.sha256()
            with f.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            result[str(f.relative_to(path))] = digest.hexdigest()
    return result


def query_comparison(expected, actual, exact):
    findings, max_gap, overlaps, exact_orders = [], 0.0, [], 0
    assert len(expected) == len(actual)
    assert expected
    for i, (e, a) in enumerate(zip(expected, actual, strict=True)):
        es, ac = (
            dict(zip(e["ids"], e["scores"], strict=True)),
            dict(zip(a["ids"], a["scores"], strict=True)),
        )
        common = es.keys() & ac.keys()
        overlaps.append(len(common) / len(es))
        gap = max((abs(es[k] - ac[k]) for k in common), default=0.0)
        max_gap = max(max_gap, gap)
        if gap > SCORE_ATOL:
            findings.append(
                {
                    "query": i,
                    "reason": "destination_score_difference",
                    "difference": gap,
                }
            )
        same_order = e["ids"] == a["ids"]
        exact_orders += same_order
        if exact and es.keys() != ac.keys():
            findings.append(
                {"query": i, "reason": "id_set", "expected": e, "actual": a}
            )
        elif exact and not same_order:
            positions = {k: j for j, k in enumerate(a["ids"])}
            changed = [
                (x, y)
                for j, x in enumerate(e["ids"])
                for y in e["ids"][j + 1 :]
                if positions[x] > positions[y]
            ]
            if not all(es[x] == es[y] and ac[x] == ac[y] for x, y in changed):
                findings.append(
                    {"query": i, "reason": "non_tie_order", "expected": e, "actual": a}
                )
    return {
        "success": not findings,
        "findings": findings,
        "max_score_difference": max_gap,
        "mean_top10_overlap": float(np.mean(overlaps)),
        "identical_order_queries": exact_orders,
        "query_count": len(expected),
        "exact_required": exact,
    }


def compare_local(expected, actual, exact=False, flat=False):
    assert expected["doc_count"] == actual["doc_count"]
    ordinary = query_comparison(expected["queries"], actual["queries"], exact or flat)
    if expected["exhaustive"] and actual["exhaustive"]:
        exhaustive = query_comparison(
            expected["exhaustive"], actual["exhaustive"], True
        )
    else:
        assert not expected["exhaustive"]
        assert not actual["exhaustive"]
        exhaustive = {
            "success": True,
            "skipped": True,
            "reason": "DiskANN has no exposed exhaustive mode; FP64 per-hit scoring checked separately",
        }
    recall_drop = expected["recall_at_10"] - actual["recall_at_10"]
    return {
        "success": ordinary["success"]
        and exhaustive["success"]
        and (exact or recall_drop <= ANN_RECALL_DROP + 1e-12),
        "ordinary": ordinary,
        "exhaustive": exhaustive,
        "reference_recall": expected["recall_at_10"],
        "actual_recall": actual["recall_at_10"],
        "recall_drop": recall_drop,
        "allowed_recall_drop": 0.0 if exact else ANN_RECALL_DROP,
    }


class Validation:
    def __init__(self, args):
        self.a = args
        self.root = args.root
        self.case = next(c for c in CASES if c["name"] == args.case)
        self.alg = self.case["algorithm"]
        self.sparse = self.alg.endswith("_sparse")
        self.flat = self.alg.startswith("flat")
        self.count, self.dim = self.case["count"], self.case["dimension"]
        self.extra = 1500 if self.alg in ("hnsw", "hnsw_sparse") else 64
        self.vectors = np.load(
            self.root / "datasets" / f"{self.count + self.extra}_{self.dim}.npy"
        )
        assert self.vectors.dtype == np.float32
        self.key = args.case + "_" + args.backend
        self.platform = "linux" if platform.system() == "Linux" else "macos"
        self.query_ids = [
            *np.linspace(0, self.count - 1, 20, dtype=int).tolist(),
            self.count,
            self.count + self.extra - 1,
            2,
        ]
        self.checks = []
        self.truth_cache = {}
        self.index_param, self.query_param, self.exhaustive_param = self.params()

    def value(self, i):
        vector = self.vectors[i]
        if self.sparse:
            selected = np.argsort(-np.abs(vector), kind="stable")[:16]
            return {int(k): float(vector[k]) for k in sorted(selected)}
        return vector.tolist()

    def params(self):
        opts = {
            "metric_type": zvec.MetricType.IP if self.sparse else zvec.MetricType.COSINE
        }
        if self.case["quant"] != "fp32":
            opts["quantize_type"] = getattr(
                zvec.QuantizeType, self.case["quant"].upper()
            )
        if self.flat:
            return zvec.FlatIndexParam(**opts), None, None
        if self.alg.startswith("hnsw"):
            return (
                zvec.HnswIndexParam(
                    m=36 if self.sparse else 16, ef_construction=100, **opts
                ),
                zvec.HnswQueryParam(ef=200),
                zvec.HnswQueryParam(ef=200, is_linear=True),
            )
        if self.alg == "ivf":
            return (
                zvec.IVFIndexParam(n_list=32, n_iters=5, **opts),
                zvec.IVFQueryParam(nprobe=16),
                zvec.IVFQueryParam(nprobe=32),
            )
        if self.alg == "vamana":
            return (
                zvec.VamanaIndexParam(max_degree=32, search_list_size=80, **opts),
                zvec.VamanaQueryParam(ef_search=200),
                zvec.VamanaQueryParam(ef_search=200, is_linear=True),
            )
        return (
            zvec.DiskAnnIndexParam(max_degree=24, list_size=64, **opts),
            zvec.DiskAnnQueryParam(list_size=150),
            None,
        )

    def option(self, readonly=False):
        return zvec.CollectionOption(
            read_only=readonly, enable_mmap=self.a.backend == "mmap"
        )

    def truth(self, mutated):
        if mutated in self.truth_cache:
            return self.truth_cache[mutated]
        n = self.count + self.extra if mutated else self.count
        data = self.vectors[:n].astype(np.float64)
        if self.sparse:
            data[:] = 0
            for i in range(n):
                for k, v in self.value(i).items():
                    data[i, k] = v
        if mutated:
            data[0] = data[-1]
        norms = np.linalg.norm(data, axis=1) if not self.sparse else None
        values = []
        for i in self.query_ids:
            q = self.vectors[i].astype(np.float64)
            if self.sparse:
                q[:] = 0
                for k, v in self.value(i).items():
                    q[k] = v
            scores = data @ q
            if not self.sparse:
                scores = 1.0 - scores / (norms * np.linalg.norm(q))
            if mutated:
                scores[2] = -np.inf if self.sparse else np.inf
            order = np.argsort(-scores if self.sparse else scores, kind="stable")[:10]
            values.append({"ids": [str(j) for j in order], "scores": scores})
        self.truth_cache[mutated] = values
        return values

    def snapshot(self, collection, mutated):
        n = self.count + self.extra if mutated else self.count
        assert collection.stats.doc_count == n - int(mutated)
        ids = sorted(
            set(
                np.linspace(0, self.count - 1, 32, dtype=int).tolist()
                + [0, 2]
                + (list(range(self.count, n)) if mutated else [])
            )
        )
        if mutated:
            ids.remove(2)
            assert not collection.fetch(["2"])
        checked = 0
        for start in range(0, len(ids), 128):
            batch = ids[start : start + 128]
            fetched = collection.fetch([str(i) for i in batch])
            assert len(fetched) == len(batch)
            for i in batch:
                doc = fetched[str(i)]
                assert doc.fields["row"] == (999999 if mutated and i == 0 else i)
                expected = self.value(n - 1 if mutated and i == 0 else i)
                actual = doc.vectors["embedding"]
                if self.sparse:
                    assert expected.keys() == actual.keys()
                    actual, expected = (
                        [actual[k] for k in expected],
                        list(expected.values()),
                    )
                assert np.allclose(expected, actual, rtol=2e-6, atol=2e-6), (
                    "raw_vector",
                    i,
                )
                checked += 1
        truth = self.truth(mutated)
        observations, exhaustive, recall = [], [], []
        score_error = 0.0
        for qi, i in enumerate(self.query_ids):

            def search(param, i=i):
                hits = collection.query(
                    zvec.Query("embedding", vector=self.value(i), param=param),
                    topk=10,
                    include_vector=False,
                )
                assert len(hits) == 10
                assert len({h.id for h in hits}) == 10
                assert all(np.isfinite(h.score) for h in hits)
                assert not mutated or "2" not in [h.id for h in hits]
                return {"ids": [h.id for h in hits], "scores": [h.score for h in hits]}

            ordinary = search(self.query_param)
            observations.append(ordinary)
            # DiskANN has no exposed exhaustive mode; its score is additionally
            # checked against FP64 raw-vector distance below.
            if self.flat:
                exhaustive.append(ordinary)
            elif self.exhaustive_param is not None:
                exhaustive.append(search(self.exhaustive_param))
            recall.append(len(set(ordinary["ids"]) & set(truth[qi]["ids"])) / 10)
            for k, score in zip(ordinary["ids"], ordinary["scores"], strict=True):
                score_error = max(score_error, abs(score - truth[qi]["scores"][int(k)]))
                if self.alg == "diskann":
                    assert np.isclose(
                        score, truth[qi]["scores"][int(k)], rtol=1e-4, atol=2e-5
                    )
        return {
            "doc_count": collection.stats.doc_count,
            "fetched_documents_checked": checked,
            "queries": observations,
            "exhaustive": exhaustive,
            "recall_at_10": float(np.mean(recall)),
            "max_score_error_vs_fp64": float(score_error),
            "exhaustive_mode": "unavailable; per-hit FP64 score check"
            if self.alg == "diskann"
            else "all_ivf_lists"
            if self.alg == "ivf"
            else "linear",
        }

    def compare(self, label, expected, actual, exact=False):
        comparison = compare_local(expected, actual, exact=exact, flat=self.flat)
        comparison["label"] = label
        self.checks.append(comparison)
        return comparison

    def observe_path(self, path, mutated, readonly=False):
        c = zvec.open(str(path), option=self.option(readonly))
        try:
            return self.snapshot(c, mutated)
        finally:
            c.close()

    def mutate(self, path):
        c = zvec.open(str(path), option=self.option())
        try:
            for start in range(self.count, self.count + self.extra, 500):
                docs = [
                    zvec.Doc(
                        id=str(i),
                        fields={"row": i},
                        vectors={"embedding": self.value(i)},
                    )
                    for i in range(start, min(start + 500, self.count + self.extra))
                ]
                assert all(s.ok() for s in c.insert(docs))
            assert c.delete("2").ok()
            assert c.update(
                zvec.Doc(
                    id="0",
                    fields={"row": 999999},
                    vectors={"embedding": self.value(self.count + self.extra - 1)},
                )
            ).ok()
            c.flush()
        finally:
            c.close()
        first = self.observe_path(path, True)
        self.compare(
            "mutation_readonly_reopen",
            first,
            self.observe_path(path, True, True),
            exact=True,
        )
        if self.alg == "ivf":
            for i, ro in enumerate([False, True, False, True, False, True]):
                self.compare(
                    f"ivf_reopen_{i}",
                    first,
                    self.observe_path(path, True, ro),
                    exact=True,
                )
        return first

    def local(self):
        # Independent local construction. Never reconstruct a migrated index.
        with tempfile.TemporaryDirectory(prefix="zvec-portability-local-") as work:
            path = Path(work) / "index"
            schema = zvec.CollectionSchema(
                name="platform_local_reference",
                fields=[zvec.FieldSchema("row", zvec.DataType.INT64)],
                vectors=[
                    zvec.VectorSchema(
                        "embedding",
                        zvec.DataType.SPARSE_VECTOR_FP32
                        if self.sparse
                        else zvec.DataType.VECTOR_FP32,
                        0 if self.sparse else self.dim,
                        index_param=self.index_param,
                    )
                ],
            )
            c = zvec.create_and_open(str(path), schema, option=self.option())
            try:
                for start in range(0, self.count, 512):
                    docs = [
                        zvec.Doc(
                            id=str(i),
                            fields={"row": i},
                            vectors={"embedding": self.value(i)},
                        )
                        for i in range(start, min(start + 512, self.count))
                    ]
                    assert all(s.ok() for s in c.insert(docs))
                c.flush()
                c.optimize()
            finally:
                c.close()
            baseline = self.observe_path(path, False)
            self.compare(
                "local_readonly_reopen",
                baseline,
                self.observe_path(path, False, True),
                exact=True,
            )
            export = self.root / "indexes" / self.platform / self.key
            assert not export.exists()
            export.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(path, export)
            write_json(
                self.root / "results" / f"manifest_{self.platform}_{self.key}.json",
                manifest(export),
            )
            mutated = self.mutate(path)
            return {"baseline": baseline, "mutated": mutated}

    def cross(self):
        native = json.loads(
            (
                self.root / "results" / f"{self.platform}_local_{self.key}.json"
            ).read_text()
        )
        source = self.root / "indexes" / "linux" / self.key
        records = {}
        for readonly in (True, False):
            with tempfile.TemporaryDirectory(prefix="zvec-portability-cross-") as work:
                path = Path(work) / "index"
                shutil.copytree(source, path)
                observed = self.observe_path(path, False, readonly)
                mode = "readonly" if readonly else "default"
                self.compare(f"destination_local_{mode}", native["baseline"], observed)
                records[mode] = observed
        self.compare(
            "migrated_readonly_vs_default",
            records["readonly"],
            records["default"],
            exact=True,
        )
        path = self.root / "roundtrip" / self.key
        assert not path.exists()
        shutil.copytree(source, path)
        mutated = self.mutate(path)
        self.compare("destination_local_mutated", native["mutated"], mutated)
        write_json(
            self.root / "results" / f"manifest_mutated_{self.key}.json", manifest(path)
        )
        return dict(**records, mutated=mutated)

    def returned(self):
        native = json.loads(
            (self.root / "results" / f"linux_local_{self.key}.json").read_text()
        )
        with tempfile.TemporaryDirectory(prefix="zvec-portability-return-") as work:
            path = Path(work) / "index"
            shutil.copytree(self.root / "roundtrip" / self.key, path)
            observed = self.observe_path(path, True, True)
            self.compare("returned_vs_linux_local_mutated", native["mutated"], observed)
            self.compare(
                "returned_readonly_repeat",
                observed,
                self.observe_path(path, True, True),
                exact=True,
            )
            return {"mutated": observed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["local", "cross", "return"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case", choices=[c["name"] for c in CASES], required=True)
    parser.add_argument("--backend", choices=["mmap", "buffer"], required=True)
    args = parser.parse_args()
    args.root = args.root.resolve()
    platform_name = "linux" if platform.system() == "Linux" else "macos"
    result_path = (
        args.root
        / "results"
        / f"{platform_name}_{args.action}_{args.case}_{args.backend}.json"
    )
    assert not result_path.exists(), result_path
    zvec.init(log_level=zvec.LogLevel.WARN, query_threads=2, optimize_threads=2)
    validator = None
    try:
        validator = Validation(args)
        result = {
            "local": validator.local,
            "cross": validator.cross,
            "return": validator.returned,
        }[args.action]()
        result["execution_success"] = True
    except Exception as error:
        result = {
            "execution_success": False,
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }
    result["checks"] = validator.checks if validator else []
    result["success"] = result["execution_success"] and all(
        c["success"] for c in result["checks"]
    )
    result["configuration"] = {
        "case": args.case,
        "backend": args.backend,
        "action": args.action,
    }
    result["runtime"] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "page_size": os.sysconf("SC_PAGESIZE"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "zvec_module": zvec.__file__,
    }
    write_json(result_path, result)
    print(  # noqa: T201 - CLI progress output
        json.dumps(
            {k: result[k] for k in ("configuration", "execution_success", "success")}
        ),
        flush=True,
    )
    raise SystemExit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
