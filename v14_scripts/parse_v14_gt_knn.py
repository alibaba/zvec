#!/usr/bin/env python3
"""Parse the kFlat GT-generation .knn logs into a neighbors-style GT file
and cross-validate a query sample by numpy brute force over the 1M base.

.knn line format (one per result, in ranked order):
    query[<qid>]\tkey[<id>], dist[<score>]
Output GT line format (matches /dataset/cohere/1m/neighbors.txt):
    <qid>;<id1> <id2> ... <id100>

.centaur.vecs container layout (tools/core/vecs_common.h VecsHeader):
    104B packed header (num_vecs u64; meta_size_v1 u16; version u16;
    meta_size u32; bitmap/key/dense/sparse/partition/taglist offsets+sizes),
    then meta (meta_size B). data_base = 104 + meta_size. Dense vectors are
    CONTIGUOUS fp32 rows (no per-record header) at data_base+dense_offset;
    the u64 key section follows the dense block -- keys are a permutation,
    so search-result ids are keys[row], never raw row positions.
Brute-force check: cosine top-100 over the dense rows, mapped through the
key section, compared against the parsed GT top-100 / top-50 key sets.
"""
import argparse
import re
import struct
from pathlib import Path

import numpy as np

KNN = re.compile(r'query\[(\d+)\]\tkey\[(\d+)\], dist\[([0-9.eE+-]+)\]')
BASE = '/home/xingrui/bench/dataset/cohere/1m/cohere_train_vector_1m.new.centaur.vecs'


def load_base(path, dim=768):
    with open(path, 'rb') as f:
        hdr = f.read(104)
    num_vecs, _v1, _ver, meta_size = struct.unpack_from('<QHHI', hdr, 0)
    (_bitmap, _key_off, _key_size, dense_offset, dense_size, _sparse_off,
     _sparse_size, _part_off, _part_size, _tag_off, _tag_size) = \
        struct.unpack_from('<11Q', hdr, 16)
    data_base = 104 + meta_size
    if dense_size != num_vecs * dim * 4:
        raise SystemExit('unexpected dense size %d for %d vecs' %
                         (dense_size, num_vecs))
    base = np.memmap(path, dtype='<f4', mode='r',
                     offset=data_base + dense_offset,
                     shape=(num_vecs, dim))
    keys = np.memmap(path, dtype='<u8', mode='r',
                     offset=data_base + dense_offset + dense_size,
                     shape=(num_vecs,))
    return base, keys, num_vecs


def parse_knn_files(logdir):
    rows = {}
    for f in sorted(Path(logdir).glob('t*.knn')):
        for line in f.read_text().splitlines():
            m = KNN.match(line)
            if not m:
                continue
            qid, key = int(m.group(1)), int(m.group(2))
            rows.setdefault(qid, []).append(key)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logdir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--queries', required=True)
    ap.add_argument('--keys', type=int, default=100)
    ap.add_argument('--check-sample', type=int, default=10)
    args = ap.parse_args()

    rows = parse_knn_files(args.logdir)
    qids = sorted(rows)
    if not qids:
        raise SystemExit('no knn rows parsed')
    if qids != list(range(len(qids))):
        raise SystemExit('qids are not 0..N-1 in order: %s' % qids[:5])
    bad = [(q, len(rows[q])) for q in qids if len(rows[q]) != args.keys]
    if bad:
        raise SystemExit('key-count anomalies (expected %d): %s'
                         % (args.keys, bad[:5]))
    with open(args.out, 'w', encoding='utf-8') as f:
        for q in qids:
            f.write('%d;%s\n' % (q, ' '.join(str(k) for k in rows[q][:args.keys])))
    print('wrote GT: %d queries x %d keys -> %s' % (len(qids), args.keys, args.out))

    n = args.check_sample
    if n > 0:
        base, keys, num_vecs = load_base(BASE)
        bn = np.linalg.norm(base, axis=1)
        base_n = base / np.where(bn == 0, 1.0, bn)[:, None]
        gt_file = Path(args.out).read_text().splitlines()
        qlines = Path(args.queries).read_text().splitlines()
        if len(qlines) != len(gt_file):
            raise SystemExit('query/GT line count mismatch: %d vs %d'
                             % (len(qlines), len(gt_file)))
        rng = np.random.default_rng(20261016)
        sample = sorted(rng.choice(len(gt_file), size=min(n, len(gt_file)),
                                   replace=False).tolist())
        ok100 = ok50 = 0
        for qi in sample:
            qid, rest = gt_file[qi].split(';', 1)
            gt = [int(x) for x in rest.split()]
            if int(qid) != qi:
                raise SystemExit('GT row %d has qid %s' % (qi, qid))
            v = np.array([float(x) for x in
                          qlines[qi].rstrip(';').split(';', 1)[1].split()],
                         dtype=np.float32)
            v = v / np.linalg.norm(v)
            sims = base_n @ v
            top = np.argpartition(-sims, 100)[:100]
            top = top[np.argsort(-sims[top])]
            top_keys = keys[top].tolist()
            ok100 += int(set(top_keys) == set(gt))
            ok50 += int(set(top_keys[:50]) == set(gt[:50]))
        print('brute-force check on %d sampled queries (keys mapped): '
              'top100 set match %d/%d, top50 set match %d/%d'
              % (len(sample), ok100, len(sample), ok50, len(sample)))
        if ok50 != len(sample):
            raise SystemExit('GT spot check FAILED')


if __name__ == '__main__':
    main()
