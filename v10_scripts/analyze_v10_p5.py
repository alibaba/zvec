#!/usr/bin/env python3
"""v10 P5 analysis: bucket quantization-ordering loss samples.

Integrated unpruned lost 327 GT hits vs covered at TopK=50 (v9); dedicated
control = 249. The TopK=100 exports make lost GT hits at quantized ranks
51-100 visible, so each loss can be paired with the top-50 intruders that
displaced it.

For every lost GT hit G (covered, not in top-50):
  - quantized rank of G (51-100 if visible, else '>100')
  - boundary intruder I = key at rank 49 (and all non-GT top-50 intruders)
  - exact distance ex_d = 1 - cos(q, x) from raw .vecs vectors vs the
    quantized score s from results.csv; err = s - ex_d
  - identity: flip requires err(G) - err(I) > exact margin
    (margin = ex_d(I) - ex_d(G) > 0 by GT definition)
  - list attribution via key->list parsed from the index containers, list
    centroid cosine to the query, residual ||x - centroid(list)||
  - same-list (within-bucket ordering) vs cross-list displacement

Aggregates: bias (mean err), variance (err std), boundary-depth histogram,
residual-norm comparison (lost GT hits vs intruders), err-vs-residual
binned means, integrated vs dedicated side by side.

Also verifies the v9 covered/baseline/unpruned totals from the TopK=50 P0
reruns and from ranks 1-50 of the TopK=100 exports.

Usage: python3 analyze_v10_p5.py [P5_RUN_DIR]
"""
import csv
import mmap
import math
import struct
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

HOME = Path.home()
REPO = HOME / 'workspace' / 'zvec' / 'rabitq_refactor' / 'zvec'
LV = REPO.parent / 'local_validation' / 'v10'
P0 = LV / 'p0_20261013-192634'
P5 = Path(sys.argv[1]) if len(sys.argv) > 1 else None
V9 = REPO / 'v9_out' / 'export_20261013-174804'
INT_INDEX = (HOME / 'bench' / 'rabitq_refactor' / 'ivf' /
             'cohere_1m_ivf_rabitq_norm1.index')
DED_INDEX = (HOME / 'bench' / 'rabitq_refactor' / 'ivf_dedicated' /
             'cohere_1m_ivfrabitq_dedicated.index')
VECS = (HOME / 'bench' / 'dataset' / 'cohere' / '1m' /
        'cohere_train_vector_1m.new.centaur.vecs')
QFILE = HOME / 'bench' / 'dataset' / 'cohere' / '1m' / 'cohere_test_vector_1000.new.txt'
GT_FILE = HOME / 'bench' / 'dataset' / 'cohere' / '1m' / 'neighbors.txt'

HDR_FMT = '<IHHIIHHIQQ3Q'          # 64B
FOOT_FMT = '<6I13Q'                 # 128B
SEGM_FMT = '<IIQQQ'                 # 32B
IVF_HDR_FMT = '<IIQIIIII'
INVERTED_LIST_FMT = '<QIII'         # 40B stride
IVF_RABITQ_HDR_FMT = '<IIIIIIBBBBQQ4I'
DED_CLUSTER_FMT = '<QQIIII'
CONV_HDR_FMT = '<IIIIIBBB2x3I'      # 32B


class Container:
    def __init__(self, path, base=0, size=None):
        self.f = open(path, 'rb')
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        self.base = base
        self.total = len(self.mm) - base if size is None else size
        self.segments = {}
        hdr_start = 0
        total = self.total
        while True:
            (_hc, _r1, _ver, _rev, magic, mhs, mfs, _mfo, content_off,
             _st, _a, _b, _c) = struct.unpack_from(
                 HDR_FMT, self.mm, base + hdr_start)
            mfo_s = struct.unpack_from('<i', self.mm, base + hdr_start + 20)[0]
            foot_off = (total + mfo_s) if mfo_s < 0 else mfo_s
            (_fc, _smc, _cc, seg_cnt, seg_meta_sz, _f1, _cs, _cp, _cpd, _ut,
             *_rs, next_hdr, _ts) = struct.unpack_from(
                 FOOT_FMT, self.mm, base + hdr_start + foot_off)
            meta_base = base + hdr_start + foot_off - seg_meta_sz
            for i in range(seg_cnt):
                (id_off, _dc, data_idx, data_sz, _ps) = struct.unpack_from(
                    SEGM_FMT, self.mm, meta_base + 32 * i)
                end = self.mm.find(b'\x00', meta_base + id_off)
                name = self.mm[meta_base + id_off:end].decode()
                if name not in self.segments:
                    self.segments[name] = (base + hdr_start + content_off +
                                           data_idx, data_sz)
            if next_hdr == 0:
                break
            hdr_start = next_hdr
        self.magic = magic

    def seg(self, name):
        return self.segments[name]

    def close(self):
        self.mm.close()
        self.f.close()


def load_int_side():
    """key -> (list_id, rank), list centroids (original space)."""
    c = Container(str(INT_INDEX))
    iho, _ihs = c.seg('ivf.inverted_header')
    (_hs, _tot, _body, list_cnt, _bv, _bs, _bc, _ms) = struct.unpack_from(
        IVF_HDR_FMT, c.mm, iho)
    imo, ims = c.seg('ivf.inverted_meta')
    assert ims == 40 * list_cnt
    lists = [struct.unpack_from(INVERTED_LIST_FMT, c.mm, imo + 40 * l)
             for l in range(list_cnt)]
    vec_sum = sum(v for _, _, v, _ in lists)
    ko, ks = c.seg('hc.keys')
    assert ks == 8 * vec_sum, f'hc.keys {ks} != 8*{vec_sum}'
    keys = struct.unpack_from(f'<{vec_sum}Q', c.mm, ko)
    key2list = {}
    id_base = 0
    for l, (_, _, vc, _) in enumerate(lists):
        for r in range(vc):
            key2list[keys[id_base + r]] = l
        id_base += vc
    # centroids: nested ivf.centroid container -> flat.features,
    # 3076B stride = [768 f32][1 f32 augmented]
    coff, csz = c.seg('ivf.centroid')
    cc = Container(str(INT_INDEX), base=coff, size=csz)
    fo, fs = cc.seg('flat.features')
    assert fs == 3076 * list_cnt, f'flat.features {fs} != 3076*{list_cnt}'
    cents = [struct.unpack_from('<768f', cc.mm, fo + l * 3076)
             for l in range(list_cnt)]
    cc.close()
    c.close()
    return key2list, cents


def load_ded_side():
    """key -> (cluster_id, rank), cluster centroids (original space)."""
    c = Container(str(DED_INDEX))
    dho, _dhs = c.seg('ivf_rabitq.header')
    (_mg, _vr, _tot, cluster_cnt, _dim, _padded, _ex, _rt, _mt, _pd,
     _bd, _ed, *_rr) = struct.unpack_from(IVF_RABITQ_HDR_FMT, c.mm, dho)
    cmo, cms = c.seg('ivf_rabitq.cluster_meta')
    assert cms == 32 * cluster_cnt
    clusters = [struct.unpack_from(DED_CLUSTER_FMT, c.mm, cmo + 32 * cl)
                for cl in range(cluster_cnt)]
    vec_sum = sum(v for _, _, v, _, _, _ in clusters)
    ko, ks = c.seg('ivf_rabitq.keys')
    assert ks == 8 * vec_sum
    keys = struct.unpack_from(f'<{vec_sum}Q', c.mm, ko)
    key2cluster = {}
    for cl, (_, _, vc, _, k_off, _r) in enumerate(clusters):
        for r in range(vc):
            key2cluster[keys[k_off + r]] = cl
    # converter: [ConverterHeader 32B][rotated cents n*padded f32]
    # [original cents n*dim f32][rotator]
    rvo, rvs = c.seg('rabitq.converter')
    (n_cl, dim, padded, rot_sz, ex_bits, rot_type, *_r) = struct.unpack_from(
        CONV_HDR_FMT, c.mm, rvo)
    assert n_cl == cluster_cnt and dim == 768
    orig_off = rvo + 32 + n_cl * padded * 4
    cents = [struct.unpack_from('<768f', c.mm, orig_off + cl * dim * 4)
             for cl in range(cluster_cnt)]
    c.close()
    return key2cluster, cents


class RawVecs:
    def __init__(self):
        self.f = open(VECS, 'rb')
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        n = 1000000
        self.vkeys = struct.unpack_from(f'<{n}Q', self.mm, 4295 + n * 3072)
        self.vrow = {k: i for i, k in enumerate(self.vkeys)}

    def vec(self, key):
        row = self.vrow[key]
        return struct.unpack_from('<768f', self.mm, 4295 + row * 3072)

    def close(self):
        self.mm.close()
        self.f.close()


def load_queries():
    qs = {}
    for line in open(QFILE):
        line = line.strip()
        if not line:
            continue
        qid_s, _, rest = line.partition(';')
        rest = rest.rstrip(';')  # lines carry a trailing separator
        qs[int(qid_s)] = [float(x) for x in rest.split()]
    return qs


def load_gt():
    gt = {}
    for line in open(GT_FILE):
        qid, _, ids = line.strip().partition(';')
        gt[int(qid)] = [int(x) for x in ids.split()[:50]]
    return gt


def load_results(path):
    res = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for row in csv.DictReader(f):
            res[row['mode']][int(row['query'])].append(
                (int(row['rank']), int(row['key']), float(row['score'])))
    for m in res:
        for q in res[m]:
            res[m][q].sort()
    return dict(res)


def load_lists(path):
    lists = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for row in csv.DictReader(f):
            lists[row['mode']][int(row['query'])].append(int(row['list_id']))
    return {m: {q: set(v) for q, v in qs.items()} for m, qs in lists.items()}


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(a):
    return math.sqrt(sum(x * x for x in a))


def exact_dist(q, qn, vec):
    d = dot(q, vec) / (qn * norm(vec))
    return 1.0 - d


def gt_in_top(res_rows, gt_keys, top_n):
    keys = {k for _, k, _ in res_rows[:top_n]}
    return len(keys & set(gt_keys))


def covered_gt(gt_keys, key2bucket, probed):
    return [g for g in gt_keys if key2bucket[g] in probed]


def side_analysis(side, res, lists, gt, key2bucket, cents, raw, queries,
                  qnorms, out_csv):
    """res/lists: single-mode dicts for the unpruned TopK=100 run."""
    print(f'\n=== {side}: ordering-loss sample analysis (TopK=100) ===')
    samples = []
    rank_hist = defaultdict(int)
    beyond = 0
    sel_loss = 0
    err_gt = []
    err_int = []
    same_list = 0
    cross_list = 0
    topk = 100
    for qid in sorted(gt):
        q = queries[qid]
        qn = qnorms[qid]
        rows = res.get(qid, [])
        if not rows:
            continue
        keys = [k for _, k, _ in rows]
        scores = {k: s for _, k, s in rows}
        rank_of = {k: r for r, k, _ in rows}
        gt_keys = gt[qid]
        gt_set = set(gt_keys)
        probed = lists.get(qid, set())
        cov = [g for g in gt_keys if key2bucket[g] in probed]
        top_keys = keys[:50]
        lost = [g for g in cov if g not in set(top_keys)]
        sel_loss += len(gt_keys) - len(cov)
        # exact distances for every visible row
        exd = {k: exact_dist(q, qn, raw.vec(k)) for k in set(keys)}
        for k in keys[:topk]:
            e = scores[k] - exd[k]
            if k in gt_set:
                err_gt.append(e)
            else:
                err_int.append(e)
        for g in lost:
            if g in rank_of and rank_of[g] < topk:
                rank_hist[rank_of[g] // 10 * 10 + 1] += 1
            else:
                beyond += 1
                continue
            b_key = keys[49]
            if b_key in gt_set:
                # boundary slot held by a GT hit: find last non-GT intruder
                nb = [k for k in top_keys if k not in gt_set]
                if not nb:
                    continue
                b_key = nb[-1]
            ex_g, ex_b = exd[g], exd[b_key]
            margin = ex_b - ex_g  # >0 by GT definition
            flip = scores[g] - scores[b_key]
            err_g = scores[g] - ex_g
            err_b = scores[b_key] - ex_b
            if key2bucket[g] == key2bucket[b_key]:
                same_list += 1
            else:
                cross_list += 1
            cg = cents[key2bucket[g]]
            cb = cents[key2bucket[b_key]]
            vg, vb = raw.vec(g), raw.vec(b_key)
            res_g = math.sqrt(max(0.0, sum(
                (a - b) ** 2 for a, b in zip(vg, cg))))
            res_b = math.sqrt(max(0.0, sum(
                (a - b) ** 2 for a, b in zip(vb, cb))))
            samples.append({
                'qid': qid, 'gt_key': g, 'qrank': rank_of[g],
                'boundary_key': b_key,
                'list_g': key2bucket[g], 'list_b': key2bucket[b_key],
                'same_list': int(key2bucket[g] == key2bucket[b_key]),
                'score_g': round(scores[g], 6),
                'score_b': round(scores[b_key], 6),
                'exact_g': round(ex_g, 6), 'exact_b': round(ex_b, 6),
                'err_g': round(err_g, 6), 'err_b': round(err_b, 6),
                'margin_exact': round(margin, 6),
                'flip_margin_quant': round(flip, 6),
                'residual_g': round(res_g, 4), 'residual_b': round(res_b, 4),
                'centroid_cos_g': round(dot(q, cg) / qn, 4),
                'centroid_cos_b': round(dot(q, cb) / qn, 4),
            })
    if out_csv and samples:
        with open(out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(samples[0].keys()))
            w.writeheader()
            w.writerows(samples)
        print(f'  wrote {len(samples)} samples to {out_csv}')
    print(f'  selection loss (GT in unprobed lists): {sel_loss}')
    print(f'  covered-but-lost GT hits: visible(51-100)={len(samples)} '
          f'beyond-100={beyond}')
    print(f'  visible lost-hit quantized rank histogram '
          f'(bucketed by 10): {dict(sorted(rank_hist.items()))}')
    print(f'  displacing intruder same-list={same_list} '
          f'cross-list={cross_list}')
    if err_gt:
        print(f'  err=score-exact over top-{topk}: GT hits: n={len(err_gt)} '
              f'mean={mean(err_gt):+.5f} std={pstdev(err_gt):.5f}')
        print(f'                                 non-GT:    n={len(err_int)} '
              f'mean={mean(err_int):+.5f} std={pstdev(err_int):.5f}')
    if samples:
        mg = [s['margin_exact'] for s in samples]
        fg = [s['flip_margin_quant'] for s in samples]
        eg = [s['err_g'] for s in samples]
        eb = [s['err_b'] for s in samples]
        rg = [s['residual_g'] for s in samples]
        rb = [s['residual_b'] for s in samples]
        print(f'  boundary pairs: exact margin mean={mean(mg):.5f} '
              f'p50={sorted(mg)[len(mg)//2]:.5f} max={max(mg):.5f}')
        print(f'    err(lost GT hit) mean={mean(eg):+.5f} |err| '
              f'mean={mean(abs(x) for x in eg):.5f}')
        print(f'    err(boundary intruder) mean={mean(eb):+.5f} |err| '
              f'mean={mean(abs(x) for x in eb):.5f}')
        print(f'    residual: lost GT hits mean={mean(rg):.4f} '
              f'vs boundary intruders mean={mean(rb):.4f}')
        print(f'    flip needs err_g - err_b > margin; observed: '
              f'mean(err_g-err_b)={mean(eg[i]-eb[i] for i in range(len(eg))):+.5f} '
              f'vs mean margin={mean(mg):.5f}')
        close_calls = sum(1 for m in mg if m < 1e-4)
        print(f'    close calls (exact margin < 1e-4): {close_calls}/{len(mg)}')
        print('  detailed boundary samples (smallest 8 exact margins):')
        for s in sorted(samples, key=lambda x: x['margin_exact'])[:8]:
            print(f'    q={s["qid"]} gt={s["gt_key"]} qrank={s["qrank"]} '
                  f'b={s["boundary_key"]} same_list={s["same_list"]} '
                  f'sc={s["score_g"]:.4f}/{s["score_b"]:.4f} '
                  f'ex={s["exact_g"]:.4f}/{s["exact_b"]:.4f} '
                  f'err={s["err_g"]:+.4f}/{s["err_b"]:+.4f} '
                  f'margin={s["margin_exact"]:.2e} '
                  f'res={s["residual_g"]:.3f}/{s["residual_b"]:.3f} '
                  f'ccos={s["centroid_cos_g"]:.3f}/{s["centroid_cos_b"]:.3f}')
    return samples


def main():
    print('loading indexes...')
    key2list, int_cents = load_int_side()
    key2cluster, ded_cents = load_ded_side()
    raw = RawVecs()
    queries = load_queries()
    gt = load_gt()
    qnorms = {qid: norm(v) for qid, v in queries.items()}
    print(f'key2list={len(key2list)} key2cluster={len(key2cluster)} '
          f'queries={len(queries)}')

    # centroid sanity: argmax over original centroids == assigned bucket
    import random
    random.seed(11)
    for side, k2b, cents in [('int', key2list, int_cents),
                             ('ded', key2cluster, ded_cents)]:
        ok = 0
        for k in random.sample(list(k2b), 24):
            v = raw.vec(k)
            best = max(range(len(cents)), key=lambda c: dot(v, cents[c]))
            ok += best == k2b[k]
        print(f'centroid argmax==assigned ({side}): {ok}/24')

    print('\n=== verification vs v9 (TopK=50 P0 reruns) ===')
    res = load_results(P0 / 'p0_int_np293_export.results.csv')
    lst = load_lists(P0 / 'p0_int_np293_export.lists.csv')
    modes = sorted(res)
    # list selection identical across the 4 int modes?
    same = all(lst[m] == lst['baseline'] for m in modes)
    print(f'  int: 4 modes share list selection: {same}')
    for m in modes:
        cov = sum(len(covered_gt(gt[q], key2list, lst[m][q])) for q in gt)
        top = sum(gt_in_top(res[m][q], gt[q], 50) for q in gt)
        print(f'  int [{m}]: covered={cov} top50_gt={top}')
    res_d = load_results(P0 / 'p0_ded_np249_unpruned.results.csv')
    lst_d = load_lists(P0 / 'p0_ded_np249_unpruned.lists.csv')
    cov = sum(len(covered_gt(gt[q], key2cluster, lst_d['dedicated'][q]))
              for q in gt)
    top = sum(gt_in_top(res_d['dedicated'][q], gt[q], 50) for q in gt)
    print(f'  ded [unpruned]: covered={cov} top50_gt={top}')
    res_dp = load_results(P0 / 'p0_ded_np249_export.results.csv')
    top = sum(gt_in_top(res_dp['dedicated'][q], gt[q], 50) for q in gt)
    print(f'  ded [pruned]:   top50_gt={top}')
    print('  v9 reference: int covered=49854 baseline=49501 unpruned=49527 '
          'reuse=49503 reuse_unpruned=49529')
    print('  v9 reference: ded covered=49772 baseline=49501 unpruned=49523')

    if P5 is None or not P5.exists():
        print('\nno P5 run dir given; TopK=100 analysis skipped '
              '(pass the p5_* dir as argv[1])')
        raw.close()
        return 0

    print(f'\n=== TopK=100 exports in {P5} ===')
    res5 = load_results(P5 / 'v10s5_int.results.csv')
    lst5 = load_lists(P5 / 'v10s5_int.lists.csv')
    for m in sorted(res5):
        cov = sum(len(covered_gt(gt[q], key2list, lst5[m][q])) for q in gt)
        t50 = sum(gt_in_top(res5[m][q], gt[q], 50) for q in gt)
        t100 = sum(gt_in_top(res5[m][q], gt[q], 100) for q in gt)
        print(f'  int [{m}] TopK=100: covered={cov} top50_gt={t50} '
              f'top100_gt={t100} '
              f'(top50 must equal the TopK=50 unpruned number)')
    res5d = load_results(P5 / 'v10s5_ded_unpruned.results.csv')
    lst5d = load_lists(P5 / 'v10s5_ded_unpruned.lists.csv')
    cov = sum(len(covered_gt(gt[q], key2cluster, lst5d['dedicated'][q]))
              for q in gt)
    t50 = sum(gt_in_top(res5d['dedicated'][q], gt[q], 50) for q in gt)
    t100 = sum(gt_in_top(res5d['dedicated'][q], gt[q], 100) for q in gt)
    print(f'  ded [unpruned] TopK=100: covered={cov} top50_gt={t50} '
          f'top100_gt={t100}')

    side_analysis('integrated (np293 unpruned)', res5['unpruned'],
                  lst5['unpruned'], gt, key2list, int_cents, raw, queries,
                  qnorms, 'v10_p5_samples_int.csv')
    side_analysis('dedicated (np249 unpruned)', res5d['dedicated'],
                  lst5d['dedicated'], gt, key2cluster, ded_cents, raw,
                  queries, qnorms, 'v10_p5_samples_ded.csv')
    raw.close()
    print('\nDONE')
    return 0


if __name__ == '__main__':
    sys.exit(main())
