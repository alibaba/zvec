#!/usr/bin/env python3
"""v11 P1 variant generator: qid-shuffle and row-shuffle query files + yamls.

Usage: python3 make_v11_variants.py [OUTPUT_DIR]
Defaults to this script's directory (repo copy: pass the run dir).
"""
import random
import sys
from pathlib import Path

QUERIES = Path("/home/xingrui/bench/dataset/cohere/1m/cohere_test_vector_1000.new.txt")
V8 = Path("/home/xingrui/bench/rabitq_refactor/v8")
RUN = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
RUN.mkdir(parents=True, exist_ok=True)

lines = QUERIES.read_text().splitlines()
assert len(lines) == 1000, f"expected 1000 query lines, got {len(lines)}"
seps = [ln.split(";", 1) for ln in lines]
assert all(len(s) == 2 for s in seps)
qids = [s[0] for s in seps]
vecs = [s[1] for s in seps]
assert [int(q) for q in qids] == list(range(1000)), "source qids are not 0..999"

# (a) qid shuffle: rows stay in file order, qids permuted -> qid != row index.
qid_perm = qids[:]
random.Random(1234).shuffle(qid_perm)
(RUN / "queries_qid_shuffle.txt").write_text(
    "\n".join(f"{qid_perm[i]};{vecs[i]}" for i in range(1000)) + "\n")

# (b) row shuffle: whole lines permuted, qids travel with their vectors ->
# file qids non-contiguous and out of order in file order.
row_perm = lines[:]
random.Random(4321).shuffle(row_perm)
(RUN / "queries_row_shuffle.txt").write_text("\n".join(row_perm) + "\n")

# Variant diagnostic configs: only QueryFile (and RecallLogDir) change.
variants = [
    ("int_qid", "v8_base_np293.yaml", "queries_qid_shuffle.txt"),
    ("int_row", "v8_base_np293.yaml", "queries_row_shuffle.txt"),
    ("ded_qid", "v8_ded_diag_np249.yaml", "queries_qid_shuffle.txt"),
    ("ded_row", "v8_ded_diag_np249.yaml", "queries_row_shuffle.txt"),
]
for name, base_cfg, qfile in variants:
    src = (V8 / base_cfg).read_text()
    assert src.count("QueryFile: ") == 1
    src = src.replace(
        f"QueryFile: {QUERIES}", f"QueryFile: {RUN / qfile}")
    src = src.replace("RecallLogDir: ./", "RecallLogDir: ./v11_")
    (RUN / f"{name}.yaml").write_text(src)

print(f"ok: variants written to {RUN} "
      "(queries_qid_shuffle, queries_row_shuffle, 4 yamls)")
print("qid_shuffle first 10 qids:", qid_perm[:10])
print("row_shuffle first 10 qids:", [ln.split(";", 1)[0] for ln in row_perm[:10]])
