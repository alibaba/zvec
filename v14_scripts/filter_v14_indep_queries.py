#!/usr/bin/env python3
"""Filter the v14 independent query set to vectors disjoint from the 1M base.

The cohere 10m test split shares vectors with the 1m train base: an exact
md5 scan (3072B raw fp32 rows, correct .centaur.vecs layout -- see
parse_v14_gt_knn.py) finds 100/1000 of the 10m test queries present in the
base and 1/1000 of the tuning queries. A query that is itself indexed is
trivially its own top-1 neighbor, so those rows are removed to keep the
independent set strictly disjoint from the indexed data. Kept queries are
renumbered 0..N-1 (query and GT files stay aligned); the original qids of
the removed rows go into the overlap report.

Writes:
  indep/queries_indep_q10m_900.txt   filtered, renumbered queries
  indep/overlap_report.txt           scan record (kept/removed counts, ids)
"""
import hashlib
import struct
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = '/home/xingrui/bench/dataset/cohere/1m/cohere_train_vector_1m.new.centaur.vecs'
SRC = HERE / 'indep' / 'queries_indep_q10m_1000.txt'
OUT = HERE / 'indep' / 'queries_indep_q10m_900.txt'
REPORT = HERE / 'indep' / 'overlap_report.txt'


def load_base_raw(path):
    with open(path, 'rb') as f:
        hdr = f.read(104)
    num_vecs, _v1, _ver, meta_size = struct.unpack_from('<QHHI', hdr, 0)
    (_bm, _ko, _ks, dense_offset, dense_size, _so, _ss, _po, _ps, _to,
     _ts) = struct.unpack_from('<11Q', hdr, 16)
    data_base = 104 + meta_size
    if dense_size != num_vecs * 3072:
        raise SystemExit('unexpected dense size')
    return np.memmap(path, dtype=np.uint8, mode='r',
                     offset=data_base + dense_offset,
                     shape=(num_vecs, 3072))


def main():
    lines = [l for l in SRC.read_text().splitlines() if l.strip()]
    raw = load_base_raw(BASE)
    base_md5 = set()
    for i in range(raw.shape[0]):
        base_md5.add(hashlib.md5(raw[i].tobytes()).digest())

    kept, removed = [], []
    for qid, line in enumerate(lines):
        vec_txt = line.rstrip(';').split(';', 1)[1]
        q = np.array([float(x) for x in vec_txt.split()], dtype=np.float32)
        if hashlib.md5(q.tobytes()).digest() in base_md5:
            removed.append(qid)
        else:
            kept.append((qid, line))

    with open(OUT, 'w', encoding='utf-8') as f:
        for new_qid, (_old_qid, line) in enumerate(kept):
            qid_txt, rest = line.split(';', 1)
            f.write('%d;%s' % (new_qid, rest))
            if not rest.endswith(';'):
                f.write(';')
            f.write('\n')

    report = [
        '=== v14 independent query overlap scan (exact md5, corrected reader) ===',
        'base: %s (%d rows)' % (BASE, raw.shape[0]),
        'source queries: %s (%d)' % (SRC, len(lines)),
        'kept (disjoint): %d -> %s (renumbered 0..%d)' %
        (len(kept), OUT, len(kept) - 1),
        'removed (exact members of base): %d' % len(removed),
        'removed original qids: %s' % ' '.join(str(q) for q in removed),
        'note: the pre-filter membership scan recorded in v14_baseline.txt used a',
        'misaligned reader (phantom 20B per-record header) and is superseded by',
        'this scan; tuning queries re-checked with the same reader: 1/1000 in base.',
    ]
    REPORT.write_text('\n'.join(report) + '\n', encoding='utf-8')
    print('\n'.join(report))


if __name__ == '__main__':
    main()
