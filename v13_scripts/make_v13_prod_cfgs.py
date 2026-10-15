#!/usr/bin/env python3
"""Generate the v13 4-way production configs:
  integrated        prod50@np293              (v12 yaml, unchanged)
  integrated+refine rrfN@np                   (N-keyed coarse heap via SF)
  native            np249                     (v13_native yamls exist)
  native+refine     natref@np                 (v13_natref yamls exist)
ScaleFactor is a C++ float: coarse_topk = floor(50 * (float)sf). For N=52 the
naive sf=1.04 gives 50*1.04f=51.999998 -> floor 51, so each sf is searched
and verified in float32 before use. Also emits holdout (qid 500-999) variants
for the recall confirmation and ded diagnostic yamls for scan-volume stats.
"""
import argparse
import math
import struct
from pathlib import Path


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def coarse_topk(sf):
    return int(math.floor(f32(f32(50) * f32(sf))))

V12 = Path('/home/xingrui/bench/integrate_rabitq_in_hnsw/v12_runs/prod')
RUN = Path(__file__).parent
QUERY_FULL = '/home/xingrui/bench/dataset/cohere/1m/cohere_test_vector_1000.new.txt'
GT_FULL = '/home/xingrui/bench/dataset/cohere/1m/neighbors.txt'
QUERY_HOLD = str(RUN / 'holdout/queries_holdout_q500_999.txt')
GT_HOLD = str(RUN / 'holdout/neighbors_holdout_q500_999.txt')


def find_scale_factor(n):
    sf = n / 50.0
    for _ in range(200):
        if coarse_topk(sf) == n:
            return round(sf, 6)
        sf += 0.001
    raise SystemExit('no scale factor found for N=%d' % n)


def holdout_swap(text):
    text = text.replace('QueryFile: ' + QUERY_FULL, 'QueryFile: ' + QUERY_HOLD)
    text = text.replace('GroundTruthFile: ' + GT_FULL, 'GroundTruthFile: ' + GT_HOLD)
    assert QUERY_HOLD in text and GT_HOLD in text
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--np', type=int, required=True, help='integrated+refine nprobe')
    ap.add_argument('--topn', type=int, required=True, help='integrated+refine N')
    ap.add_argument('--natref-np', type=int, required=True, help='native+refine nprobe')
    args = ap.parse_args()
    sf = find_scale_factor(args.topn)
    assert coarse_topk(sf) == args.topn
    print('scale factor for N=%d: %r (floor(50*sf_f32)=%d)' % (args.topn, sf, coarse_topk(sf)))

    rrf_t1 = (V12 / 'v12_rrf60_np200_t1.yaml').read_text()
    assert rrf_t1.count('\\"nprobe\\":200') == 1
    assert rrf_t1.count('ScaleFactor: 1.2') == 1
    assert rrf_t1.count('RecallLogDir: ./rrf60_np200_t1.recall') == 1

    written = []
    for threads in (1, 4):
        name = 'v13_rrf%d_np%d_t%d' % (args.topn, args.np, threads)
        text = rrf_t1.replace('\\"nprobe\\":200', '\\"nprobe\\":%d' % args.np)
        text = text.replace('ScaleFactor: 1.2', 'ScaleFactor: %s' % sf)
        text = text.replace('RecallLogDir: ./rrf60_np200_t1.recall',
                            'RecallLogDir: ./%s.recall' % name)
        if threads == 4:
            assert text.count('BenchThreadCount: 1') == 1
            text = text.replace('BenchThreadCount: 1', 'BenchThreadCount: 4')
        out = RUN / (name + '.yaml')
        out.write_text(text)
        written.append(out.name)
        hold = holdout_swap(text).replace(
            'RecallLogDir: ./%s.recall' % name, 'RecallLogDir: ./%s_holdout.recall' % name)
        outh = RUN / (name + '_holdout.yaml')
        outh.write_text(hold)
        written.append(outh.name)

    # integrated baseline holdout (prod50@np293)
    prod_t1 = (V12 / 'v12_prod50_np293_t1.yaml').read_text()
    outh = RUN / 'v13_int50_np293_t1_holdout.yaml'
    outh.write_text(holdout_swap(prod_t1).replace(
        'RecallLogDir: ./prod50_np293_t1.recall',
        'RecallLogDir: ./v13_int50_np293_t1_holdout.recall'))
    written.append(outh.name)

    # int diagnostic yaml at (np, N) for scan-volume stats
    int_diag_src = RUN / ('int_np%d.yaml' % args.np)
    if not int_diag_src.exists():
        int_diag_src = Path('/home/xingrui/bench/integrate_rabitq_in_hnsw/v11_runs/int_np%d.yaml' % args.np)
    text = int_diag_src.read_text()
    assert text.count('TopK: 50') == 1 and text.count('"topk":50') == 1
    text = text.replace('TopK: 50', 'TopK: %d' % args.topn)
    text = text.replace('"topk":50', '"topk":%d' % args.topn)
    text = text.replace('RecallLogDir: ./int_np%d.recall' % args.np,
                        'RecallLogDir: ./v13_intdiag_np%d_topk%d.recall' % (args.np, args.topn))
    out = RUN / ('v13_intdiag_np%d_topk%d.yaml' % (args.np, args.topn))
    out.write_text(text)
    written.append(out.name)

    # ded diagnostic yamls for scan-volume stats
    nat_t1 = (RUN / 'v13_native_np249_t1.yaml').read_text()
    for np_, topk, name in ((249, 50, 'v13_deddiag_np249_topk50'),
                            (args.natref_np, 60, 'v13_deddiag_np%d_topk60' % args.natref_np)):
        text = nat_t1.replace('RecallLogDir: ./v13_native_np249_t1.recall',
                              'RecallLogDir: ./%s.recall' % name)
        if np_ != 249:
            assert text.count('\\"nprobe\\":249') == 1
            text = text.replace('\\"nprobe\\":249', '\\"nprobe\\":%d' % np_)
        if topk != 50:
            assert text.count('TopK: 50') == 1 and text.count('\\"topk\\":50') == 1
            text = text.replace('TopK: 50', 'TopK: %d' % topk)
            text = text.replace('\\"topk\\":50', '\\"topk\\":%d' % topk)
        out = RUN / (name + '.yaml')
        out.write_text(text)
        written.append(out.name)

    print('wrote %d yamls to %s:' % (len(written), RUN))
    for w in written:
        print('  ' + w)


if __name__ == '__main__':
    main()
