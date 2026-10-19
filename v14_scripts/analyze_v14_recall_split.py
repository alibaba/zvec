#!/usr/bin/env python3
"""v14 review task 1 (revised): decompose the independent-set recall loss
with REAL probed-bucket coverage and per-query stage evidence.

Stages per frozen scheme (all GT@50 hit percentages over 900 queries):
  1. real bucket coverage [measured]: GT@50 members living in the lists the
     production-shape search actually probed. From the diag --export-lists
     bucket sets (v14_lists_*.lists.csv, baseline/dedicated mode) joined
     with the offline key->list mapping (v14_gtcov_*.gtlists.csv, built
     from raw index segments by ivf_rabitq_gt_coverage). The tool's
     recomputed selection (v14_gtcov_*.selected.csv) is cross-checked
     against the diag-exported sets per query BEFORE any coverage number is
     trusted; mismatches abort. This replaces the previous Top100-unpruned
     proxy: a GT vector can rank below 100 under approximate quantized
     scoring, so that proxy was an approximation, not an exact measure.
  2. unpruned candidate coverage [measured]: GT@50 inside the TopN
     candidate list at the same np with rabitq pruning disabled
     (v14_prof_*_indep.results.csv 'unpruned' mode; dedicated schemes use
     the v14_split_*_noprune runs).
  3. pruned candidate coverage [measured]: GT@50 inside the
     production-shape TopN candidates (pruning on, refiner-free diag twin
     -- exactly the set that enters the kFlat refiner for TopN=55).
  4. final Recall@50 [measured]: recomputed from the production per-query
     .knn logs and cross-checked against v14_indep_summary.txt at the
     tool's printed precision.

Per-query signed deltas between consecutive stages (new/lost hits, may be
negative; no recall-monotonicity presupposition). For TopN=55 refiner
schemes the final top-50 must be a subset of the pruned TopN candidates
(checked per query); when containment holds, all per-query deltas are
<= 0 and the aggregate delta is 0, the FP32 rerank is proven per-query
lossless w.r.t. GT hits, not merely on average. For TopN=50 no-refiner
schemes the final should equal the pruned TopN exactly; any drift between
the production run and the diag run is reported, not hidden.

The Top100 / np1024 grid is kept as a clearly separated extension
section: it is approximate and is NOT used in the single-factor chain.
Writes v14_recall_split_analysis.txt. Usage: python3
analyze_v14_recall_split.py [RUN_DIR]
"""
import re
import sys
from pathlib import Path

RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
GT_FILE = RUN / 'indep' / 'neighbors_indep_q10m_900.txt'
INDEP_SUM = RUN / 'v14_indep_summary.txt'
NQ = 900
GT_TOP = 50
TOTAL = GT_TOP * NQ

SCHEMES = {
    'int': dict(np=293, topn=50, refiner=False, integrated=True,
                lists='v14_lists_int_np293', gtcov='v14_gtcov_int_np293',
                prod='v14_prof_int_np293_indep.results.csv',
                prod_recall_cfg='v12_prod50_np293_t1_indep',
                knn='v12_prod50_np293_t1_indep.recall',
                c100='v14_split_int_np293_top100.results.csv',
                c1024='v14_split_int_np1024_top100.results.csv'),
    'rrf': dict(np=180, topn=55, refiner=True, integrated=True,
                lists='v14_lists_rrf_np180', gtcov='v14_gtcov_rrf_np180',
                prod='v14_prof_int_np180_top55_indep.results.csv',
                prod_recall_cfg='v13_rrf55_np180_t1_indep',
                knn='v13_rrf55_np180_t1_indep.recall',
                c100='v14_split_rrf_np180_top100.results.csv',
                c1024='v14_split_rrf_np1024_top100.results.csv'),
    'native': dict(np=249, topn=50, refiner=False, integrated=False,
                   lists='v14_lists_native_np249',
                   gtcov='v14_gtcov_native_np249',
                   prod='v14_prof_ded_np249_indep.results.csv',
                   prod_noprune='v14_split_native_np249_top50_noprune.'
                                'results.csv',
                   prod_recall_cfg='v13_native_np249_t1_indep',
                   knn='v13_native_np249_t1_indep.recall',
                   c100='v14_split_native_np249_top100.results.csv',
                   c100_noprune='v14_split_native_np249_top100_noprune.'
                                'results.csv',
                   c1024='v14_split_native_np1024_top100.results.csv',
                   c1024_noprune='v14_split_native_np1024_top100_noprune.'
                                 'results.csv'),
    'natref': dict(np=170, topn=55, refiner=True, integrated=False,
                   lists='v14_lists_natref_np170',
                   gtcov='v14_gtcov_natref_np170',
                   prod='v14_prof_ded_np170_top55_indep.results.csv',
                   prod_noprune='v14_split_natref_np170_top55_noprune.'
                                'results.csv',
                   prod_recall_cfg='v14_natref_np170_top55_t1_indep',
                   knn='v14_natref_np170_top55_t1_indep.recall',
                   c100='v14_split_natref_np170_top100.results.csv',
                   c100_noprune='v14_split_natref_np170_top100_noprune.'
                                'results.csv',
                   c1024='v14_split_natref_np1024_top100.results.csv',
                   c1024_noprune='v14_split_natref_np1024_top100_noprune.'
                                 'results.csv'),
}

KNN_RE = re.compile(r'^query\[(\d+)\]\s+key\[(\d+)\]')


def load_gt(errors):
    gt = {}
    with GT_FILE.open(encoding='utf-8') as f:
        for line in f:
            idx, _, keys = line.partition(';')
            gt[int(idx)] = set(int(k) for k in keys.split()[:GT_TOP])
    if len(gt) != NQ:
        errors.append(f'GT has {len(gt)} queries, expected {NQ}')
    if any(len(v) != GT_TOP for v in gt.values()):
        errors.append('GT lines with fewer than 50 keys')
    return gt


def load_prod_recalls():
    out = {}
    for line in INDEP_SUM.read_text(encoding='utf-8').splitlines():
        m = re.match(r'indep recall (\S+) rc=0 Recall@50: ([0-9.]+)', line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def load_results(path, errors):
    """{(mode): {query: [key, ...] ordered by rank}}."""
    by_mode = {}
    with Path(path).open(encoding='utf-8') as f:
        header = f.readline().strip()
        if header != 'query,mode,rank,key,score':
            errors.append(f'{path.name}: unexpected header {header!r}')
            return by_mode
        for line in f:
            q, mode, rank, key, _ = line.rstrip('\n').split(',')
            by_mode.setdefault(mode, {}).setdefault(int(q), []).append(
                (int(rank), int(key)))
    for mode, qs in by_mode.items():
        if len(qs) != NQ:
            errors.append(f'{path.name}: mode {mode} has {len(qs)} queries')
        for q, rows in qs.items():
            rows.sort()
            qs[q] = [k for _, k in rows]
    return by_mode


def load_lists(path, np_, errors):
    """{mode: {query: set(list_id)}}; also counts limited rows."""
    modes = {}
    limited = 0
    with Path(path).open(encoding='utf-8') as f:
        header = f.readline().strip()
        if header != 'query,mode,list_id,scan_count,limited':
            errors.append(f'{path.name}: unexpected header {header!r}')
            return modes, limited
        for line in f:
            q, mode, lid, _scan, lim = line.rstrip('\n').split(',')
            modes.setdefault(mode, {}).setdefault(int(q), set()).add(int(lid))
            if int(lim):
                limited += 1
    for mode, qs in modes.items():
        if len(qs) != NQ:
            errors.append(f'{path.name}: mode {mode} has {len(qs)} queries')
        bad = [q for q, s in qs.items() if len(s) != np_]
        if bad:
            errors.append(f'{path.name}: mode {mode} list count != {np_} '
                          f'for {len(bad)} queries (first: {bad[:5]})')
    return modes, limited


def load_gtcov(prefix, np_, errors):
    """selected {query: set(list)}, covered {query: int},
    gtlists {query: {gt_id: list}}."""
    selected, covered, gtlists = {}, {}, {}
    with (RUN / f'{prefix}.selected.csv').open(encoding='utf-8') as f:
        for line in f:
            q, _, rest = line.partition(';')
            selected[int(q)] = set(int(x) for x in rest.split())
    with (RUN / f'{prefix}.coverage.csv').open(encoding='utf-8') as f:
        header = f.readline().strip()
        if header != 'query,covered,gt_total,selected_vectors':
            errors.append(f'{prefix}.coverage.csv: unexpected header')
        for line in f:
            q, cov, gt_total, _selvec = line.rstrip('\n').split(',')
            covered[int(q)] = int(cov)
            if int(gt_total) != GT_TOP:
                errors.append(f'{prefix}.coverage.csv: query {q} gt_total='
                              f'{gt_total}')
    with (RUN / f'{prefix}.gtlists.csv').open(encoding='utf-8') as f:
        header = f.readline().strip()
        if header != 'query,gt_id,list':
            errors.append(f'{prefix}.gtlists.csv: unexpected header')
        for line in f:
            q, gt_id, lst = line.rstrip('\n').split(',')
            gtlists.setdefault(int(q), {})[int(gt_id)] = int(lst)
    if len(selected) != NQ:
        errors.append(f'{prefix}.selected.csv: {len(selected)} queries')
    bad = [q for q, s in selected.items() if len(s) != np_]
    if bad:
        errors.append(f'{prefix}.selected.csv: list count != {np_} for '
                      f'{len(bad)} queries (first: {bad[:5]})')
    if len(covered) != NQ:
        errors.append(f'{prefix}.coverage.csv: {len(covered)} queries')
    if len(gtlists) != NQ:
        errors.append(f'{prefix}.gtlists.csv: {len(gtlists)} queries')
    return selected, covered, gtlists


def load_knn(name, gt, errors):
    """{query: [key, ...] in rank order}; strict per-query validation."""
    d = RUN / name
    if not d.is_dir():
        errors.append(f'{name}: missing RecallLogDir {d.name}')
        return None
    shards = sorted(d.glob('t*.knn'))
    if not shards:
        errors.append(f'{name}: no t*.knn shards')
        return None
    qs, qid_shard, bad = {}, {}, 0
    for p in shards:
        shard = p.name
        for lineno, line in enumerate(
                p.read_text(encoding='utf-8').splitlines(), 1):
            s = line.strip()
            if not s:
                continue
            m = KNN_RE.match(s)
            if not m:
                bad += 1
                if bad <= 5:
                    errors.append(f'{name}/{shard}:{lineno}: unparseable '
                                  f'line: {s[:60]}')
                continue
            q, k = int(m.group(1)), int(m.group(2))
            if q in qid_shard and qid_shard[q] != shard:
                bad += 1
                if bad <= 5:
                    errors.append(f'{name}/{shard}:{lineno}: query {q} '
                                  f'already logged by {qid_shard[q]}')
                continue
            qid_shard[q] = shard
            qs.setdefault(q, []).append(k)
    if bad > 5:
        errors.append(f'{name}: {bad} per-query log problems in total')
    missing = sorted(set(gt) - set(qs))
    extra = sorted(set(qs) - set(gt))
    if missing:
        errors.append(f'{name}: {len(missing)} query id(s) missing from .knn '
                      f'(first: {missing[:5]})')
    if extra:
        errors.append(f'{name}: {len(extra)} unexpected query id(s) '
                      f'(first: {extra[:5]})')
    for q in sorted(set(qs) & set(gt)):
        ks = qs[q]
        if len(ks) != GT_TOP:
            errors.append(f'{name}: query {q} returned {len(ks)} results, '
                          f'expected {GT_TOP}')
        elif len(set(ks)) != GT_TOP:
            errors.append(f'{name}: query {q} returned '
                          f'{GT_TOP - len(set(ks))} duplicate id(s)')
    if missing or extra or bad:
        return None
    return qs


def hits_of(qs, gt):
    """{query: GT@50 hit count} for an ordered candidate list."""
    return {q: len(gt[q] & set(ks)) for q, ks in qs.items()}


def stage_delta(a, b):
    """Signed per-query deltas b-a: (new, lost, unchanged, max_abs)."""
    new = sum(1 for q in a if b[q] > a[q])
    lost = sum(1 for q in a if b[q] < a[q])
    mx = max((abs(b[q] - a[q]) for q in a), default=0)
    return new, lost, NQ - new - lost, mx


def pct(x):
    return 100.0 * x / TOTAL


def cov(by_mode, gt, mode):
    return sum(len(gt[q] & set(by_mode[mode][q])) for q in gt)


def main():
    errors = []
    gt = load_gt(errors)
    prod_recalls = load_prod_recalls()
    lines = ['=== v14/v15 independent-set recall decomposition (revised): '
             'real bucket coverage + per-query stage evidence ===']

    for scheme, cfg in SCHEMES.items():
        lines.append(f'--- {scheme} (np={cfg["np"]}, TopN={cfg["topn"]}, '
                     f'{"kFlat refiner" if cfg["refiner"] else "no refiner"}) '
                     f'---')
        for key in ('prod', 'prod_noprune', 'knn'):
            if cfg.get(key) and not (RUN / cfg[key]).exists():
                errors.append(f'missing {cfg[key]}')
        if not (RUN / f'{cfg["lists"]}.lists.csv').exists():
            errors.append(f'missing {cfg["lists"]}.lists.csv')
        for suffix in ('.selected.csv', '.coverage.csv', '.gtlists.csv'):
            if not (RUN / (cfg['gtcov'] + suffix)).exists():
                errors.append(f'missing {cfg["gtcov"]}{suffix}')
        for key in ('c100', 'c100_noprune', 'c1024', 'c1024_noprune'):
            if cfg.get(key) and not (RUN / cfg[key]).exists():
                errors.append(f'missing {cfg[key]} (extension grid)')
        if errors:
            continue

        # --- stage 1: real bucket coverage from measured data ---
        lists, limited = load_lists(RUN / f'{cfg["lists"]}.lists.csv',
                                    cfg['np'], errors)
        selected, covered, gtlists = load_gtcov(cfg['gtcov'], cfg['np'],
                                                errors)
        probed_mode = 'baseline' if cfg['integrated'] else 'dedicated'
        if probed_mode not in lists:
            errors.append(f'{cfg["lists"]}.lists.csv: no {probed_mode} mode')
            continue
        probed = lists[probed_mode]
        # gate: tool-recomputed selection == diag-exported probed lists
        mism = [q for q in sorted(gt)
                if selected.get(q) != probed.get(q)]
        if mism:
            errors.append(f'{scheme}: recomputed selection != diag-exported '
                          f'probed lists for {len(mism)} queries '
                          f'(first: {mism[:5]}) -- tool replication broken')
            continue
        lines.append(f'  selection cross-check: recomputed selection == '
                     f'diag-exported probed lists for {NQ}/{NQ} queries '
                     f'({cfg["np"]} lists each) [gate OK]')
        if limited:
            lines.append(f'  NOTE: {limited} list rows report limited=1 '
                         f'(max_scan truncation present)')
        # gtlists must cover exactly GT@50
        bad_gt = [q for q in gt if set(gtlists.get(q, {})) != gt[q]]
        if bad_gt:
            errors.append(f'{scheme}: gtlists key set != GT@50 for '
                          f'{len(bad_gt)} queries')
            continue
        hits_bucket = {q: sum(1 for g in gt[q]
                              if gtlists[q][g] in probed[q]) for q in gt}
        mism = [q for q in gt if covered.get(q) != hits_bucket[q]]
        if mism:
            errors.append(f'{scheme}: recomputed bucket coverage != tool '
                          f'coverage.csv for {len(mism)} queries '
                          f'(first: {mism[:5]})')
            continue

        # --- stages 2-3: candidate coverage from diag results ---
        prod = load_results(RUN / cfg['prod'], errors)
        if cfg['integrated']:
            unpruned = prod['unpruned']
            pruned = prod['baseline']
        else:
            noprune = load_results(RUN / cfg['prod_noprune'], errors)
            unpruned = noprune['dedicated']
            pruned = prod['dedicated']
        topn = cfg['topn']
        for label, stage in (('unpruned', unpruned), ('pruned', pruned)):
            if len(stage) != NQ:
                errors.append(f'{scheme}: {label} stage has {len(stage)} '
                              f'queries')
        if errors:
            continue
        hits_unpruned = hits_of({q: ks[:topn] for q, ks in
                                 unpruned.items()}, gt)
        hits_pruned = hits_of({q: ks[:topn] for q, ks in pruned.items()}, gt)

        # --- stage 4: final recall from production .knn ---
        r_tool = prod_recalls.get(cfg['prod_recall_cfg'])
        if r_tool is None:
            errors.append(f'no production recall for '
                          f'{cfg["prod_recall_cfg"]} in {INDEP_SUM.name}')
            continue
        knn = load_knn(cfg['knn'], gt, errors)
        if knn is None or errors:
            continue
        hits_final = hits_of(knn, gt)
        t_final = sum(hits_final.values())
        r_knn = pct(t_final)
        if abs(r_knn - r_tool) > 5e-5 + 1e-9:
            errors.append(f'{scheme}: final recall mismatch: '
                          f'.knn={r_knn:.6f} ({t_final}/{TOTAL}) vs '
                          f'tool={r_tool}')

        # --- report the chain ---
        t_bucket = sum(hits_bucket.values())
        t_unpruned = sum(hits_unpruned.values())
        t_pruned = sum(hits_pruned.values())
        lines.append(f'  stage 1 real bucket coverage   : {pct(t_bucket):.4f} '
                     f'({t_bucket}/{TOTAL}) [measured]')
        lines.append(f'  stage 2 unpruned Top{topn} coverage: '
                     f'{pct(t_unpruned):.4f} ({t_unpruned}/{TOTAL}) '
                     f'[measured]')
        n, l, u, mx = stage_delta(hits_bucket, hits_unpruned)
        lines.append(f'     bucket -> unpruned  : '
                     f'{pct(t_unpruned) - pct(t_bucket):+.4f}pp; per-query '
                     f'new {n}, lost {l}, unchanged {u}, max |d| {mx}')
        lines.append(f'  stage 3 pruned Top{topn} coverage  : '
                     f'{pct(t_pruned):.4f} ({t_pruned}/{TOTAL}) [measured]')
        n, l, u, mx = stage_delta(hits_unpruned, hits_pruned)
        lines.append(f'     unpruned -> pruned  : '
                     f'{pct(t_pruned) - pct(t_unpruned):+.4f}pp; per-query '
                     f'new {n}, lost {l}, unchanged {u}, max |d| {mx}')
        lines.append(f'  stage 4 final Recall@50 (.knn) : {r_knn:.4f} '
                     f'({t_final}/{TOTAL}) [measured; tool {r_tool} OK]')
        n, l, u, mx = stage_delta(hits_pruned, hits_final)
        agg = t_final - t_pruned
        lines.append(f'     pruned -> final     : '
                     f'{pct(t_final) - pct(t_pruned):+.4f}pp (aggregate '
                     f'hits {agg:+d}); per-query new {n}, lost {l}, '
                     f'unchanged {u}, max |d| {mx}')
        if cfg['refiner']:
            viol = [q for q in gt if not set(knn[q]) <= set(pruned[q][:topn])]
            if viol:
                errors.append(f'{scheme}: final top-50 NOT a subset of '
                              f'pruned Top{topn} candidates for '
                              f'{len(viol)} queries (first: {viol[:5]})')
            elif n == 0 and l == 0:
                lines.append(f'     final == pruned candidate hits for every '
                             f'query (0/0 new/lost, subset check '
                             f'{NQ}/{NQ}): FP32 rerank per-query lossless '
                             f'w.r.t. GT hits [verified]')
            else:
                lines.append(f'     subset check {NQ}/{NQ} OK; rerank '
                             f'redistributed {n + l} queries (see deltas '
                             f'above)')
        else:
            same = sum(1 for q in gt if knn[q] == pruned[q][:topn])
            if same == NQ:
                lines.append(f'     identical ordered ids pruned-vs-final: '
                             f'{NQ}/{NQ}: diag run reproduces the production '
                             f'result exactly [verified]')
            else:
                lines.append(f'     NOTE: identical ordered ids '
                             f'pruned-vs-final only {same}/{NQ} (diag vs '
                             f'production run drift); hit deltas above are '
                             f'the authoritative comparison')
        gaps = [100.0 - pct(t_bucket), pct(t_bucket) - pct(t_unpruned),
                pct(t_unpruned) - pct(t_pruned),
                pct(t_pruned) - pct(t_final)]
        chain_ok = abs(sum(gaps) - (100.0 - r_knn)) < 1e-9
        lines.append(f'  chain: bucket miss {gaps[0]:.4f} + ranking-at-TopN '
                     f'{gaps[1]:.4f} + pruning {gaps[2]:.4f} + final '
                     f'{gaps[3]:.4f} = {sum(gaps):.4f} == '
                     f'100 - {r_knn:.4f} '
                     f'{"[OK]" if chain_ok else "[MISMATCH]"}')
        int_chain = ((TOTAL - t_bucket) + (t_bucket - t_unpruned) +
                     (t_unpruned - t_pruned) + (t_pruned - t_final))
        if int_chain != TOTAL - t_final:
            errors.append(f'{scheme}: integer chain sum {int_chain} != '
                          f'{TOTAL - t_final}')

        # --- extension grid (approximate; NOT part of the chain) ---
        c100 = load_results(RUN / cfg['c100'], errors)
        if cfg['integrated']:
            g = (cov(c100, gt, 'baseline'), cov(c100, gt, 'unpruned'))
            c1024 = load_results(RUN / cfg['c1024'], errors)
            g2 = (cov(c1024, gt, 'baseline'), cov(c1024, gt, 'unpruned'))
        else:
            c100n = load_results(RUN / cfg['c100_noprune'], errors)
            c1024 = load_results(RUN / cfg['c1024'], errors)
            c1024n = load_results(RUN / cfg['c1024_noprune'], errors)
            g = (cov(c100, gt, 'dedicated'), cov(c100n, gt, 'dedicated'))
            g2 = (cov(c1024, gt, 'dedicated'), cov(c1024n, gt, 'dedicated'))
        lines.append(f'  extension (TopK=100 grid, approximate, not used in '
                     f'the chain): pruned np_prod={pct(g[0]):.4f} '
                     f'unpruned np_prod={pct(g[1]):.4f} | pruned '
                     f'np1024={pct(g2[0]):.4f} unpruned '
                     f'np1024={pct(g2[1]):.4f}')

    if errors:
        print(f'INTEGRITY FAILURE: {len(errors)} problem(s) -- '
              f'v14_recall_split_analysis.txt NOT written')
        for e in errors:
            print(f'  - {e}', file=sys.stderr)
        return 1

    out = '\n'.join(lines)
    print(out)
    (RUN / 'v14_recall_split_analysis.txt').write_text(out + '\n',
                                                       encoding='utf-8')
    print(f"\nwrote {RUN / 'v14_recall_split_analysis.txt'}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
