#!/usr/bin/env python3
"""Split the 1000-query test set into disjoint halves by qid:
- tune   qid 0-499   (parameter selection)
- holdout qid 500-999 (independent confirmation)
Both the query file (qid;f1 f2 ...) and the GT file (qid;id1 id2 ...)
are split consistently. Holdout answers whether the (nprobe, N) choices
tuned on the full 1000-query set generalize beyond it.
"""
import os

DS = "/home/xingrui/bench/dataset/cohere/1m"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "holdout")


def split(path, out_tune, out_hold, sep):
    qids = []
    with open(path) as f:
        tune = open(out_tune, "w")
        hold = open(out_hold, "w")
        for line in f:
            if not line.strip():
                continue
            qid = int(line.split(sep, 1)[0])
            qids.append(qid)
            (tune if qid < 500 else hold).write(line)
        tune.close()
        hold.close()
    assert len(qids) == len(set(qids)), "duplicate qids in %s" % path
    return qids


def main():
    os.makedirs(OUT, exist_ok=True)
    qq = split(os.path.join(DS, "cohere_test_vector_1000.new.txt"),
               os.path.join(OUT, "queries_tune_q0_499.txt"),
               os.path.join(OUT, "queries_holdout_q500_999.txt"), ";")
    gq = split(os.path.join(DS, "neighbors.txt"),
               os.path.join(OUT, "neighbors_tune_q0_499.txt"),
               os.path.join(OUT, "neighbors_holdout_q500_999.txt"), ";")
    assert set(qq) == set(gq), "query/GT qid sets differ"
    tune_n = sum(1 for q in qq if q < 500)
    hold_n = sum(1 for q in qq if q >= 500)
    print("queries: %d total -> tune %d, holdout %d" % (len(qq), tune_n, hold_n))
    print("qid range: %d..%d" % (min(qq), max(qq)))
    print("wrote 4 files to %s" % OUT)


if __name__ == "__main__":
    main()
