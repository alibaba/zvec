#!/usr/bin/env python3
"""v14 review task 4: decompose the independent-set recall loss.

Ground truth: indep/neighbors_indep_q10m_900.txt (900 queries x 100 keys,
exact kFlat scan); GT@50 = first 50 keys per query. For every diagnostic
results.csv (query,mode,rank,key,score) it computes the coverage of GT@50
inside the returned top-N (all rows) and inside the first 50 ranks.

Decomposition per scheme (all numbers are GT@50 percentage points):
  entered-TopN   coverage of the production-shape base search (pruning on,
                 np_prod, TopK 50/55) -- for rrf/natref this is the set that
                 enters the FP32 refiner, and because GT@50 members are the
                 globally nearest vectors, the FP32 rerank keeps every GT
                 hit present in the candidate set, so final Recall@50 should
                 equal this coverage (checked, delta reported).
  bucket miss    coverage(unpruned, np=1024, TopK=100) - coverage(unpruned,
                 np_prod, TopK=100). np=1024 probes every list; TopK=100 is
                 wide enough that quantized ranking essentially never pushes
                 a true top-50 member below rank 100, so the unpruned top-100
                 coverage is the probed-bucket coverage.
  pruning loss   coverage(unpruned, np_prod) - coverage(pruned, np_prod),
                 measured at TopK=100 (no truncation confound) and at the
                 production TopK.
  ranking loss   100 - coverage(unpruned, np=1024, TopK=100): the pure
                 quantized-ranking residue over a full unpruned scan.

Consumes run_v14_recall_split.sh outputs plus the frozen v14_prof_*_indep
twins; production Recall@50 comes from v14_indep_summary.txt. Also verifies
that reuse ON/OFF return identical per-query key sequences. Writes
v14_recall_split_analysis.txt. Usage: python3 analyze_v14_recall_split.py
"""
import re
import sys
from pathlib import Path

RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
GT_FILE = RUN / 'indep' / 'neighbors_indep_q10m_900.txt'
INDEP_SUM = RUN / 'v14_indep_summary.txt'
NQ = 900

SCHEMES = {
    'int': dict(np=293, topn=50, integrated=True,
                prod_recall_cfg='v12_prod50_np293_t1_indep',
                prod='v14_prof_int_np293_indep.results.csv',
                c100='v14_split_int_np293_top100.results.csv',
                c1024='v14_split_int_np1024_top100.results.csv'),
    'rrf': dict(np=180, topn=55, integrated=True,
                prod_recall_cfg='v13_rrf55_np180_t1_indep',
                prod='v14_prof_int_np180_top55_indep.results.csv',
                c100='v14_split_rrf_np180_top100.results.csv',
                c1024='v14_split_rrf_np1024_top100.results.csv'),
    'native': dict(np=249, topn=50, integrated=False,
                   prod_recall_cfg='v13_native_np249_t1_indep',
                   prod='v14_prof_ded_np249_indep.results.csv',
                   prod_noprune='v14_split_native_np249_top50_noprune.results.csv',
                   c100='v14_split_native_np249_top100.results.csv',
                   c100_noprune='v14_split_native_np249_top100_noprune.results.csv',
                   c1024='v14_split_native_np1024_top100.results.csv',
                   c1024_noprune='v14_split_native_np1024_top100_noprune.results.csv'),
    'natref': dict(np=170, topn=55, integrated=False,
                   prod_recall_cfg='v14_natref_np170_top55_t1_indep',
                   prod='v14_prof_ded_np170_top55_indep.results.csv',
                   prod_noprune='v14_split_natref_np170_top55_noprune.results.csv',
                   c100='v14_split_natref_np170_top100.results.csv',
                   c100_noprune='v14_split_natref_np170_top100_noprune.results.csv',
                   c1024='v14_split_natref_np1024_top100.results.csv',
                   c1024_noprune='v14_split_natref_np1024_top100_noprune.results.csv'),
}

INTEGRATED_MODES = ('baseline', 'unpruned', 'reuse', 'reuse_unpruned')
DEDICATED_MODE = 'dedicated'


def load_gt():
    gt = {}
    with GT_FILE.open(encoding='utf-8') as f:
        for line in f:
            idx, _, keys = line.partition(';')
            gt[int(idx)] = [int(k) for k in keys.split()]
    if len(gt) != NQ:
        sys.exit(f'GT has {len(gt)} queries, expected {NQ}')
    if any(len(v) < 50 for v in gt.values()):
        sys.exit('GT lines with fewer than 50 keys')
    return {q: set(v[:50]) for q, v in gt.items()}


def load_prod_recalls():
    out = {}
    for line in INDEP_SUM.read_text(encoding='utf-8').splitlines():
        m = re.match(r'indep recall (\S+) rc=0 Recall@50: ([0-9.]+)', line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def load_results(path, expected_nq=NQ):
    """Returns {(mode): {query: [key, ...] ordered by rank}}."""
    by_mode = {}
    with Path(path).open(encoding='utf-8') as f:
        header = f.readline().strip()
        if header != 'query,mode,rank,key,score':
            sys.exit(f'{path}: unexpected header {header!r}')
        for line in f:
            q, mode, rank, key, _ = line.rstrip('\n').split(',')
            by_mode.setdefault(mode, {}).setdefault(int(q), []).append(
                (int(rank), int(key)))
    for mode, qs in by_mode.items():
        if len(qs) != expected_nq:
            sys.exit(f'{path}: mode {mode} has {len(qs)} queries')
        for q, rows in qs.items():
            rows.sort()
            qs[q] = [k for _, k in rows]
    return by_mode


def coverage(by_mode, gt, mode, topn=None):
    hits = 0
    for q, keys in by_mode[mode].items():
        ks = keys[:topn] if topn else keys
        hits += len(gt[q] & set(ks))
    return 100.0 * hits / (50.0 * len(gt))


def identical(a, b):
    """Count queries whose full ordered key sequences match across modes."""
    if set(a) != set(b):
        return None
    return sum(1 for q in a if a[q] == b[q])


def main():
    gt = load_gt()
    prod_recalls = load_prod_recalls()
    lines = ['=== v14 independent-set recall-loss decomposition ===']

    for scheme, cfg in SCHEMES.items():
        files = {}
        for role, key in (('prod', 'prod'), ('prod_noprune', 'prod_noprune'),
                          ('c100', 'c100'),
                          ('c100_noprune', 'c100_noprune'),
                          ('c1024', 'c1024'), ('c1024_noprune', 'c1024_noprune')):
            if cfg.get(key):
                p = RUN / cfg[key]
                if not p.exists():
                    sys.exit(f'missing {p} (run run_v14_recall_split.sh)')
                files[role] = load_results(p)

        r_prod = prod_recalls.get(cfg['prod_recall_cfg'])
        if r_prod is None:
            sys.exit(f'no production recall for {cfg["prod_recall_cfg"]} in '
                     f'{INDEP_SUM}')

        lines.append(f'--- {scheme} (np={cfg["np"]}, TopN={cfg["topn"]}, '
                     f'final Recall@50={r_prod:.4f}) ---')

        # production-shape base search (pruning on)
        prod = files['prod']
        pruned_mode = 'baseline' if cfg['integrated'] else DEDICATED_MODE
        entered = coverage(prod, gt, pruned_mode)
        lines.append(f'entered-Top{cfg["topn"]} (pruned, np_prod, prod shape): '
                     f'{entered:.4f}  (delta vs final recall '
                     f'{entered - r_prod:+.4f})')

        # production-shape unpruned
        if cfg['integrated']:
            unpruned_prod = files['prod']
            unpruned_mode = 'unpruned'
        else:
            unpruned_prod = files['prod_noprune']
            unpruned_mode = DEDICATED_MODE
        entered_unpruned = coverage(unpruned_prod, gt, unpruned_mode)
        lines.append(f'entered-Top{cfg["topn"]} (unpruned, np_prod): '
                     f'{entered_unpruned:.4f}  -> pruning loss at prod TopK: '
                     f'{entered_unpruned - entered:+.4f}')

        # TopK=100 grid
        if cfg['integrated']:
            c100_base = coverage(files['c100'], gt, 'baseline')
            c100_unpruned = coverage(files['c100'], gt, 'unpruned')
            c1024_base = coverage(files['c1024'], gt, 'baseline')
            c1024_unpruned = coverage(files['c1024'], gt, 'unpruned')
        else:
            c100_base = coverage(files['c100'], gt, DEDICATED_MODE)
            c100_unpruned = coverage(files['c100_noprune'], gt, DEDICATED_MODE)
            c1024_base = coverage(files['c1024'], gt, DEDICATED_MODE)
            c1024_unpruned = coverage(files['c1024_noprune'], gt, DEDICATED_MODE)
        lines.append(f'TopK=100 grid: pruned np_prod={c100_base:.4f} '
                     f'unpruned np_prod={c100_unpruned:.4f} | '
                     f'pruned np1024={c1024_base:.4f} '
                     f'unpruned np1024={c1024_unpruned:.4f}')
        lines.append(f'  probed-bucket coverage (unpruned, np_prod, top100): '
                     f'{c100_unpruned:.4f}')
        lines.append(f'  bucket miss    = unpruned np1024 - np_prod: '
                     f'{c1024_unpruned - c100_unpruned:+.4f}')
        lines.append(f'  pruning loss   = unpruned - pruned (np_prod, top100): '
                     f'{c100_unpruned - c100_base:+.4f}')
        lines.append(f'  ranking loss   = 100 - unpruned np1024 top100: '
                     f'{100.0 - c1024_unpruned:+.4f}')
        lines.append(f'  topN rank/truncation = pruned np_prod top100 - '
                     f'entered-Top{cfg["topn"]}: '
                     f'{c100_base - entered:+.4f}')
        if c100_unpruned > c1024_unpruned + 1e-9:
            lines.append('  WARN: unpruned np_prod top100 coverage exceeds '
                         'np1024 (impossible)')
        if c100_base > c100_unpruned + 1e-9:
            lines.append('  WARN: pruned coverage exceeds unpruned (impossible)')

        # reuse identity check (integrated files carry reuse modes)
        if cfg['integrated']:
            for f_role in ('prod', 'c100', 'c1024'):
                fm = files[f_role]
                for a, b in (('baseline', 'reuse'),
                             ('unpruned', 'reuse_unpruned')):
                    same = identical(fm[a], fm[b])
                    cov_b = coverage(fm, gt, b)
                    cov_a = coverage(fm, gt, a)
                    same_str = (f'{same}/{NQ}' if same is not None
                                else 'query sets differ')
                    lines.append(f'  reuse {f_role} {b}: coverage {cov_b:.4f} '
                                 f'(vs {a} {cov_a:.4f}, delta '
                                 f'{cov_b - cov_a:+.4f}); identical ordered '
                                 f'ids {same_str}'
                                 + ('' if same == NQ else
                                    '  (reuse perturbs pruning decisions)'))

    out = '\n'.join(lines)
    print(out)
    (RUN / 'v14_recall_split_analysis.txt').write_text(out + '\n',
                                                       encoding='utf-8')
    print(f"\nwrote {RUN / 'v14_recall_split_analysis.txt'}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
