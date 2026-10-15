#!/usr/bin/env python3
"""v12 patch: refine-keyed rerank scheme + per-query production verification
+ NaN/Inf verifier hardening + row-order rotation.

Applies count-asserted replacements to tools/core/ivf_rabitq_scan_replay.cc.
Run twice is refused (first hunk would no longer match).
"""
import sys
from pathlib import Path

REPO = Path.home() / 'workspace' / 'zvec' / 'rabitq_refactor' / 'zvec'
TARGET = REPO / 'tools' / 'core' / 'ivf_rabitq_scan_replay.cc'

R = []  # (old, new, note)

# ---------------------------------------------------------------- H1: header
R.append((
"""// Candidate rerank experiment (--rerank-topn=N[,N...] with --vecs and --gt):
// the int side additionally runs "prod50" (production-path replica: coarse
// estimate + interleaved lower-bound pruning + refine + top-50 heap, minus
// probe selection and finalize) and "rerankN" (top-N candidates collected
// by the same lower-bound pruning with heap size N, then exact rerank with
// original vectors from the .vecs file). Recall uses the first
// min(50, ids) ground-truth ids per line, like tools/core/recall. The
// dedicated side is skipped in rerank mode.
""",
"""// Candidate rerank experiment (--rerank-topn=N[,N...] with --vecs and --gt):
// the int side additionally runs "prod50" (production-path replica: coarse
// estimate + interleaved lower-bound pruning + refine + top-50 heap, minus
// probe selection and finalize), "rrfN" (the same production walk with the
// heap widened to N keyed by refine scores, then exact rerank of those N
// candidates with original vectors from the .vecs file), and "rlbN"
// (control: candidates collected by a lower_bound-keyed heap of size N --
// the v11-4b first-round scheme; it changes the pruning-threshold
// semantics). Recall uses the first min(50, ids) ground-truth ids per
// line, like tools/core/recall. With --results-int, rrfN (and prod50 when
// the file holds topk=50) verify their candidate sets per query against
// the production baseline results: ids must match as sets, common-key
// scores within --tolerance, boundary ties at the cutoff score allowed.
// The dedicated side is skipped in rerank mode. Extra rows rotate
// (reversed every second round); timings pair within one invocation but
// are not immune to intra-invocation drift.
""", 'header comment'))

# --------------------------------------------------------------- H2: flags
R.append((
"""DEFINE_string(rerank_topn, "",
              "comma-separated top-N cutoffs for the rerank experiment "
              "(e.g. 60,80,100); skips the dedicated side");
""",
"""DEFINE_string(rerank_topn, "",
              "comma-separated top-N cutoffs for the rerank experiment "
              "(e.g. 60,80,100); skips the dedicated side");
DEFINE_string(results_int, "",
              "diagnostic results.csv (baseline rows) of a topk=N run: "
              "rrfN/prod50 candidates are verified against it per query");
""", 'results_int flag'))

# ------------------------------------------------------- H3: ReplayStats
R.append((
"""  size_t gt_ids = 0;
  size_t recall_hits = 0;
  size_t coverage_hits = 0;
};
""",
"""  size_t gt_ids = 0;
  size_t recall_hits = 0;
  size_t coverage_hits = 0;
  size_t refines = 0;
  size_t cmp_queries = 0;
  size_t cmp_exact = 0;
  size_t cmp_tie = 0;
  size_t cmp_id_mismatch = 0;
  size_t cmp_score_mismatch = 0;
  double cmp_max_delta = 0;
};
""", 'ReplayStats fields'))

# ------------------------------------------- H4: samples loader isfinite
R.append((
"""    s.distance = std::stof(f[6]);
    s.lower_bound = std::stof(f[7]);
    s.coarse_dot = std::stof(f[8]);
    (*out)[s.query].push_back(s);
""",
"""    s.distance = std::stof(f[6]);
    s.lower_bound = std::stof(f[7]);
    s.coarse_dot = std::stof(f[8]);
    if (!std::isfinite(s.distance) || !std::isfinite(s.lower_bound) ||
        !std::isfinite(s.coarse_dot)) {
      std::cerr << "samples.csv " << path << ":" << line_no
                << " has a non-finite sample value" << std::endl;
      return false;
    }
    (*out)[s.query].push_back(s);
""", 'samples loader isfinite'))

# ------------------------------------------- H5: sample comparison guards
R.append((
"""          v.max_distance_err = std::max(v.max_distance_err, d_err);
          v.max_lower_err = std::max(v.max_lower_err, l_err);
          v.max_dot_err = std::max(v.max_dot_err, c_err);
          if (d_err > tol || l_err > tol || c_err > tol) {
""",
"""          if (std::isfinite(d_err))
            v.max_distance_err = std::max(v.max_distance_err, d_err);
          if (std::isfinite(l_err))
            v.max_lower_err = std::max(v.max_lower_err, l_err);
          if (std::isfinite(c_err))
            v.max_dot_err = std::max(v.max_dot_err, c_err);
          if (!std::isfinite(d_err) || !std::isfinite(l_err) ||
              !std::isfinite(c_err) || d_err > tol || l_err > tol ||
              c_err > tol) {
""", 'comparison isfinite'))

# ------------------- H6: HeapPairs + ProdTopN + CompareProdTopN + loader
R.append((
"""std::vector<uint64_t> HeapKeys(const IndexDocumentHeap &heap) {
  std::vector<uint64_t> keys;
  keys.reserve(heap.size());
  for (const auto &doc : heap.container()) keys.push_back(doc.key());
  return keys;
}
""",
"""std::vector<uint64_t> HeapKeys(const IndexDocumentHeap &heap) {
  std::vector<uint64_t> keys;
  keys.reserve(heap.size());
  for (const auto &doc : heap.container()) keys.push_back(doc.key());
  return keys;
}

std::vector<std::pair<uint64_t, float>> HeapPairs(
    const IndexDocumentHeap &heap) {
  std::vector<std::pair<uint64_t, float>> pairs;
  pairs.reserve(heap.size());
  for (const auto &doc : heap.container())
    pairs.emplace_back(doc.key(), doc.score());
  return pairs;
}

// Production topk=N baseline results (rows of results.csv), keyed by the
// load-order query row, for per-query candidate verification.
struct ProdTopN {
  size_t n = 0;
  std::vector<std::vector<std::pair<uint64_t, float>>> per_query;
};

// Candidate-set verification: the collected (key, refine score) pairs must
// match the production top-N ids as a set; common-key scores must agree
// within tol. When the id sets differ, the comparison still passes as a
// boundary tie only if every differing key sits at its own side's cutoff
// score (the largest kept score, the N-th best) within tol: heap
// tie-breaking at the cutoff is not an ordering guarantee of production.
void CompareProdTopN(const std::vector<std::pair<uint64_t, float>> &cand,
                     const std::vector<std::pair<uint64_t, float>> &prod,
                     double tol, ReplayStats *stats) {
  ++stats->cmp_queries;
  std::unordered_map<uint64_t, float> pm;
  for (const auto &p : prod) pm.emplace(p.first, p.second);
  std::unordered_map<uint64_t, float> cm;
  for (const auto &c : cand) cm.emplace(c.first, c.second);
  float p_cut = -std::numeric_limits<float>::infinity();
  float c_cut = -std::numeric_limits<float>::infinity();
  for (const auto &p : pm) p_cut = std::max(p_cut, p.second);
  for (const auto &c : cm) c_cut = std::max(c_cut, c.second);
  bool exact = pm.size() == cm.size();
  if (exact) {
    for (const auto &c : cm) {
      if (!pm.count(c.first)) {
        exact = false;
        break;
      }
    }
  }
  bool tie_ok = true;
  for (const auto &c : cm) {
    const auto it = pm.find(c.first);
    if (it != pm.end()) {
      const double delta = std::abs(double(it->second) - double(c.second));
      if (std::isfinite(delta)) {
        stats->cmp_max_delta = std::max(stats->cmp_max_delta, delta);
        if (!(delta <= tol)) ++stats->cmp_score_mismatch;
      } else {
        ++stats->cmp_score_mismatch;
      }
    } else if (!(std::abs(double(c.second) - double(c_cut)) <= tol)) {
      tie_ok = false;
    }
  }
  for (const auto &p : pm) {
    if (!cm.count(p.first) &&
        !(std::abs(double(p.second) - double(p_cut)) <= tol)) {
      tie_ok = false;
    }
  }
  if (exact) {
    ++stats->cmp_exact;
  } else if (tie_ok) {
    ++stats->cmp_tie;
  } else {
    ++stats->cmp_id_mismatch;
  }
}

// Loads baseline rows of the diagnostic results.csv
// ("query,mode,rank,key,score"). The query column is the load-order row
// index, like lists.csv; non-finite scores and inconsistent per-query row
// counts are hard errors.
bool LoadResultsCsv(const std::string &path, size_t qcount,
                    ProdTopN *out) {
  std::ifstream in(path);
  if (!in) return false;
  std::string line;
  if (!std::getline(in, line) || line != "query,mode,rank,key,score") {
    std::cerr << "results.csv " << path << " has an unexpected header"
              << std::endl;
    return false;
  }
  out->per_query.assign(qcount, {});
  out->n = 0;
  size_t line_no = 1;
  while (std::getline(in, line)) {
    ++line_no;
    if (line.empty()) continue;
    std::array<std::string, 5> f;
    size_t pos = 0;
    bool ok = true;
    for (size_t i = 0; i < 5; ++i) {
      const size_t next = line.find(',', pos);
      if ((i < 4) != (next != std::string::npos)) {
        ok = false;
        break;
      }
      f[i] = line.substr(pos, next - pos);
      pos = next + 1;
    }
    if (!ok) {
      std::cerr << "results.csv " << path << ":" << line_no
                << " malformed line" << std::endl;
      return false;
    }
    if (f[1] != "baseline") continue;
    const long long q = std::stoll(f[0]);
    if (q < 0) {
      std::cerr << "results.csv " << path << ":" << line_no
                << " negative query row " << q << std::endl;
      return false;
    }
    if (static_cast<size_t>(q) >= qcount) continue;
    const float score = std::stof(f[4]);
    if (!std::isfinite(score)) {
      std::cerr << "results.csv " << path << ":" << line_no
                << " has a non-finite score" << std::endl;
      return false;
    }
    out->per_query[static_cast<size_t>(q)].emplace_back(std::stoull(f[3]),
                                                        score);
  }
  for (size_t q = 0; q < qcount; ++q) {
    const size_t got = out->per_query[q].size();
    if (got == 0) {
      std::cerr << "results.csv " << path << " has no baseline rows for "
                << "query " << q << std::endl;
      return false;
    }
    if (out->n == 0) {
      out->n = got;
    } else if (got != out->n) {
      std::cerr << "results.csv " << path << " query " << q << " has "
                << got << " rows, expected " << out->n << std::endl;
      return false;
    }
  }
  return true;
}
""", 'HeapPairs/ProdTopN/Compare/LoadResultsCsv'))

# ------------------------------------- H7: prod50 signature + refines
R.append((
"""ReplayStats ReplayIntegratedProd50(
    const IntegratedIndex &idx,
    const std::vector<std::vector<float>> &queries,
    const std::vector<long long> &qids, const RecordedLists &rec,
    const GroundTruth &gt, bool timed) {
""",
"""ReplayStats ReplayIntegratedProd50(
    const IntegratedIndex &idx,
    const std::vector<std::vector<float>> &queries,
    const std::vector<long long> &qids, const RecordedLists &rec,
    const GroundTruth &gt, const ProdTopN *prod, bool timed) {
""", 'prod50 signature'))

R.append((
"""          const float score = scanner->refine(row + i * idx.stride, est[i]);
          heap.emplace(keys[i], score);
          if (heap.full()) worst = heap.begin()->score();
          stats.checksum += score;
""",
"""          const float score = scanner->refine(row + i * idx.stride, est[i]);
          heap.emplace(keys[i], score);
          if (heap.full()) worst = heap.begin()->score();
          ++stats.refines;
          stats.checksum += score;
""", 'prod50 refines counter'))

R.append((
"""    const std::vector<uint64_t> top = HeapKeys(heap);
    CountRecall(it->second, &stats, top, top);
""",
"""    const std::vector<uint64_t> top = HeapKeys(heap);
    CountRecall(it->second, &stats, top, top);
    if (prod && prod->n == 50) {
      CompareProdTopN(HeapPairs(heap), prod->per_query[q], FLAGS_tolerance,
                      &stats);
    }
""", 'prod50 verification hook'))

# --------------------------------------------- H8: rerank scheme rewrite
R.append((
"""// Rerank experiment (v11-4b): collect the top-N candidates of the
// production pruning path (same rejected() test with a lower-bound heap of
// size N), then rerank them with original vectors from the .vecs file and
// keep the exact top-50. The exact score is -cosine (smaller = better),
// matching the quantizer's distance convention for kCosine/kInnerProduct.
ReplayStats ReplayIntegratedRerank(
    const IntegratedIndex &idx,
    const std::vector<std::vector<float>> &queries,
    const std::vector<long long> &qids, const RecordedLists &rec,
    const GroundTruth &gt, uint32_t topn, const VecsStore &vecs,
    bool timed) {
""",
"""// Rerank experiment: collect the top-N candidates of the production
// pruning walk, then rerank them with original vectors from the .vecs file
// and keep the exact top-50. The exact score is -cosine (smaller = better),
// matching the quantizer's distance convention for kCosine/kInnerProduct.
// kRefine keys the collection heap by the refined RaBitQ score; the heap
// threshold feeds the same rejected() pruning as production, with the heap
// widened to N. kLowerBound keys it by the coarse lower_bound (v11-4b
// first-round control) which also changes the pruning threshold.
enum class RerankScheme { kRefine, kLowerBound };

ReplayStats ReplayIntegratedRerank(
    const IntegratedIndex &idx,
    const std::vector<std::vector<float>> &queries,
    const std::vector<long long> &qids, const RecordedLists &rec,
    const GroundTruth &gt, uint32_t topn, RerankScheme scheme,
    const VecsStore &vecs, const ProdTopN *prod, bool timed) {
""", 'rerank signature + scheme'))

R.append((
"""        for (uint32_t i = 0; i < count; ++i) {
          if (!(keeps & (uint32_t(1) << i))) continue;
          if (collect.full() && est[i].lower_bound > worst) continue;
          collect.emplace(keys[i], est[i].lower_bound);
          if (collect.full()) worst = collect.begin()->score();
        }
""",
"""        const char *row = reinterpret_cast<const char *>(
            idx.row_base + list.offset +
            static_cast<size_t>(b) * idx.block_size);
        for (uint32_t i = 0; i < count; ++i) {
          if (!(keeps & (uint32_t(1) << i))) continue;
          if (collect.full() && est[i].lower_bound > worst) continue;
          if (scheme == RerankScheme::kRefine) {
            const float score = scanner->refine(row + i * idx.stride,
                                                est[i]);
            collect.emplace(keys[i], score);
            ++stats.refines;
          } else {
            collect.emplace(keys[i], est[i].lower_bound);
          }
          if (collect.full()) worst = collect.begin()->score();
        }
""", 'rerank collection walk'))

R.append((
"""    // Rerank the collected candidates with original vectors.
    cand_keys = HeapKeys(collect);
""",
"""    // Rerank the collected candidates with original vectors.
    cand_keys = HeapKeys(collect);
    if (prod && prod->n == topn && scheme == RerankScheme::kRefine) {
      CompareProdTopN(HeapPairs(collect), prod->per_query[q], FLAGS_tolerance,
                      &stats);
    }
""", 'rerank verification hook'))

# ------------------------------------------------------- H10: usage text
R.append((
"""             "[--tolerance=1e-4] [--int-variant=prod] "
             "[--rerank-topn=60,80,100 --vecs=FILE --gt=FILE]"
          << std::endl;
""",
"""             "[--tolerance=1e-4] [--int-variant=prod] "
             "[--rerank-topn=60,80,100 --vecs=FILE --gt=FILE] "
             "[--results-int=results.csv]"
          << std::endl;
""", 'usage text'))

# ------------------------------------------------- H11: tolerance check
R.append((
"""    if (((FLAGS_samples_int.empty() != FLAGS_samples_ded.empty()) &&
         !rerank_mode) ||
        FLAGS_tolerance < 0) {
      std::cerr << "--samples-int and --samples-ded must be used together; "
                   "--tolerance must be >= 0"
                << std::endl;
      return 1;
    }
""",
"""    if (((FLAGS_samples_int.empty() != FLAGS_samples_ded.empty()) &&
         !rerank_mode) ||
        !(FLAGS_tolerance >= 0)) {
      std::cerr << "--samples-int and --samples-ded must be used together; "
                   "--tolerance must be finite and >= 0"
                << std::endl;
      return 1;
    }
""", 'tolerance finite check'))

# ------------------------------------------------ H12: load ProdTopN
R.append((
"""    VecsStore vecs;
    GroundTruth gt;
    if (rerank_mode) {
      if (!LoadVecs(FLAGS_vecs, &vecs) || !LoadGroundTruth(FLAGS_gt, &gt))
        return 1;
    }
""",
"""    VecsStore vecs;
    GroundTruth gt;
    ProdTopN prod_topn;
    if (rerank_mode) {
      if (!LoadVecs(FLAGS_vecs, &vecs) || !LoadGroundTruth(FLAGS_gt, &gt))
        return 1;
      if (!FLAGS_results_int.empty()) {
        if (!LoadResultsCsv(FLAGS_results_int, qcount, &prod_topn))
          return 1;
        bool n_known = prod_topn.n == 50;
        for (uint32_t n : rerank_topn_values)
          if (static_cast<size_t>(n) == prod_topn.n) n_known = true;
        if (!n_known) {
          std::cerr << "--results-int topk=" << prod_topn.n
                    << " matches no prod50/rrfN row; nothing will be "
                       "verified"
                    << std::endl;
        }
        std::cout << "results-int: " << FLAGS_results_int
                  << " topk=" << prod_topn.n << " queries=" << qcount
                  << std::endl;
      }
    }
""", 'load results-int'))

# ------------------------------------------------------- H13: warmup
R.append((
"""      if (rerank_mode) {
        ReplayIntegratedProd50(int_idx, queries, qs.qids, rec_int, gt,
                               false);
        for (uint32_t n : rerank_topn_values) {
          ReplayIntegratedRerank(int_idx, queries, qs.qids, rec_int, gt, n,
                                 vecs, false);
        }
      }
""",
"""      if (rerank_mode) {
        ReplayIntegratedProd50(int_idx, queries, qs.qids, rec_int, gt,
                               nullptr, false);
        for (uint32_t n : rerank_topn_values) {
          ReplayIntegratedRerank(int_idx, queries, qs.qids, rec_int, gt, n,
                                 RerankScheme::kRefine, vecs, nullptr,
                                 false);
          ReplayIntegratedRerank(int_idx, queries, qs.qids, rec_int, gt, n,
                                 RerankScheme::kLowerBound, vecs, nullptr,
                                 false);
        }
      }
""", 'warmup both schemes'))

# -------------------------------------------- H14: row names + rotation
R.append((
"""    std::vector<std::string> rerank_row_names;
    for (uint32_t n : rerank_topn_values)
      rerank_row_names.push_back("rerank" + std::to_string(n));
""",
"""    std::vector<std::string> rerank_row_names;
    std::vector<std::string> rerank_ctl_names;
    for (uint32_t n : rerank_topn_values) {
      rerank_row_names.push_back("rrf" + std::to_string(n));
      rerank_ctl_names.push_back("rlb" + std::to_string(n));
    }
""", 'row names rrf/rlb'))

R.append((
"""            if (!seq_order) {
              if (iv_cluster0) {
                ReplayStats st = ReplayIntegratedDedShape(
                    int_idx, queries, rec_int, true, true);
                rows.push_back({r, "int", "orig", "cluster0", std::move(st)});
              }
              if (iv_dedshape) {
                ReplayStats st = ReplayIntegratedDedShape(
                    int_idx, queries, rec_int, false, true);
                rows.push_back({r, "int", "orig", "dedshape", std::move(st)});
              }
              if (flat_base) {
                ReplayStats st = ReplayIntegrated(
                    int_idx, queries, rec_int, false, true, flat_base);
                rows.push_back(
                    {r, "int", "orig", iv_flat16 ? "flat16" : "flat",
                     std::move(st)});
              }
              if (rerank_mode) {
                {
                  ReplayStats st = ReplayIntegratedProd50(
                      int_idx, queries, qs.qids, rec_int, gt, true);
                  rows.push_back(
                      {r, "int", "orig", "prod50", std::move(st)});
                }
                for (size_t ni = 0; ni < rerank_topn_values.size(); ++ni) {
                  ReplayStats st = ReplayIntegratedRerank(
                      int_idx, queries, qs.qids, rec_int, gt,
                      rerank_topn_values[ni], vecs, true);
                  rows.push_back({r, "int", "orig",
                                  rerank_row_names[ni].c_str(),
                                  std::move(st)});
                }
              }
            }
""",
"""            if (!seq_order) {
              // Extra rows are generated in a fixed order but appended
              // rotated (reversed every second round), so no variant keeps
              // a permanent warm/cold position inside the invocation.
              std::vector<std::pair<const char *, ReplayStats>> extra;
              if (iv_cluster0) {
                extra.emplace_back(
                    "cluster0",
                    ReplayIntegratedDedShape(int_idx, queries, rec_int, true,
                                             true));
              }
              if (iv_dedshape) {
                extra.emplace_back(
                    "dedshape",
                    ReplayIntegratedDedShape(int_idx, queries, rec_int,
                                             false, true));
              }
              if (flat_base) {
                extra.emplace_back(
                    iv_flat16 ? "flat16" : "flat",
                    ReplayIntegrated(int_idx, queries, rec_int, false, true,
                                     flat_base));
              }
              if (rerank_mode) {
                extra.emplace_back(
                    "prod50",
                    ReplayIntegratedProd50(
                        int_idx, queries, qs.qids, rec_int, gt,
                        prod_topn.n == 50 ? &prod_topn : nullptr, true));
                for (size_t ni = 0; ni < rerank_topn_values.size(); ++ni) {
                  const bool verify =
                      prod_topn.n != 0 &&
                      prod_topn.n == rerank_topn_values[ni];
                  extra.emplace_back(
                      rerank_row_names[ni].c_str(),
                      ReplayIntegratedRerank(
                          int_idx, queries, qs.qids, rec_int, gt,
                          rerank_topn_values[ni], RerankScheme::kRefine,
                          vecs, verify ? &prod_topn : nullptr, true));
                }
                for (size_t ni = 0; ni < rerank_topn_values.size(); ++ni) {
                  extra.emplace_back(
                      rerank_ctl_names[ni].c_str(),
                      ReplayIntegratedRerank(
                          int_idx, queries, qs.qids, rec_int, gt,
                          rerank_topn_values[ni], RerankScheme::kLowerBound,
                          vecs, nullptr, true));
                }
              }
              if (r % 2 == 0) std::reverse(extra.begin(), extra.end());
              for (auto &e : extra) {
                rows.push_back(
                    {r, "int", "orig", e.first, std::move(e.second)});
              }
            }
""", 'round loop rotation + schemes'))

# ------------------------------------------------ H15: CSV columns
R.append((
"""    out << "round,side,order,variant,queries,lists,vectors,blocks,"
           "prep_us_per_query,scan_us_per_query,ns_per_vector,ns_per_block,"
           "checksum,rerank_us_per_query,recall,coverage\\n";
""",
"""    out << "round,side,order,variant,queries,lists,vectors,blocks,"
           "prep_us_per_query,scan_us_per_query,ns_per_vector,ns_per_block,"
           "checksum,rerank_us_per_query,recall,coverage,refines\\n";
""", 'csv header'))

R.append((
"""          << (row.stats.gt_ids
                  ? double(row.stats.coverage_hits) / row.stats.gt_ids
                  : std::numeric_limits<double>::quiet_NaN())
          << '\\n';
""",
"""          << (row.stats.gt_ids
                  ? double(row.stats.coverage_hits) / row.stats.gt_ids
                  : std::numeric_limits<double>::quiet_NaN())
          << ',' << row.stats.refines << '\\n';
""", 'csv row refines'))

# ------------------------------------------------ H16: summary block
R.append((
"""      const double p_scan = med_field("prod50", &ReplayStats::scan_ns);
      const double p_prep = med_field("prod50", &ReplayStats::prep_ns);
      const double p_recall = med_ratio("prod50", &ReplayStats::recall_hits,
                                        &ReplayStats::gt_ids);
      std::cout << "  rerank prod50: scan_us/q=" << p_scan
                << " prep_us/q=" << p_prep
                << " recall=" << p_recall * 100.0 << "%" << std::endl;
      for (const auto &name : rerank_row_names) {
        const double c_scan = med_field(name, &ReplayStats::scan_ns);
        const double c_prep = med_field(name, &ReplayStats::prep_ns);
        const double c_rr = med_field(name, &ReplayStats::rerank_ns);
        const double c_cov = med_ratio(name, &ReplayStats::coverage_hits,
                                       &ReplayStats::gt_ids);
        const double c_recall = med_ratio(name, &ReplayStats::recall_hits,
                                          &ReplayStats::gt_ids);
        const double total = c_scan + c_prep + c_rr;
        const double ptotal = p_scan + p_prep;
        std::cout << "  rerank " << name << ": scan_us/q=" << c_scan
                  << " prep_us/q=" << c_prep << " rerank_us/q=" << c_rr
                  << " total_us/q=" << total << " recall="
                  << c_recall * 100.0 << "% coverage=" << c_cov * 100.0
                  << "% | vs prod50 total x"
                  << (ptotal ? total / ptotal : 0.0) << " recall delta "
                  << (c_recall - p_recall) * 100.0 << "pt" << std::endl;
      }
      std::cout << "  rerank scope: prod50 and rerank rows both exclude "
                   "probe selection and finalize; totals pair within one "
                   "invocation"
                << std::endl;
""",
"""      auto med_count = [&](const std::string &variant,
                           size_t ReplayStats::*field) {
        std::vector<double> vals;
        for (const auto &row : rows) {
          if (std::string(row.side) == "int" &&
              std::string(row.order) == "orig" &&
              std::string(row.variant) == variant)
            vals.push_back(double(row.stats.*field) / qcount);
        }
        if (vals.empty()) return 0.0;
        std::sort(vals.begin(), vals.end());
        return vals[vals.size() / 2];
      };
      auto paired_ratios = [&](const std::string &variant) {
        std::map<int, double> prod;
        std::map<int, double> var;
        for (const auto &row : rows) {
          if (std::string(row.side) != "int" ||
              std::string(row.order) != "orig")
            continue;
          const double total = (row.stats.prep_ns + row.stats.scan_ns +
                                row.stats.rerank_ns) /
                               1000.0;
          if (std::string(row.variant) == "prod50") prod[row.round] = total;
          if (std::string(row.variant) == variant) var[row.round] = total;
        }
        std::vector<double> out;
        for (const auto &vr : var) {
          const auto it = prod.find(vr.first);
          if (it != prod.end() && it->second > 0)
            out.push_back(vr.second / it->second);
        }
        return out;
      };
      auto last_stats =
          [&](const std::string &variant) -> const ReplayStats * {
        for (auto it = rows.rbegin(); it != rows.rend(); ++it) {
          if (std::string(it->side) == "int" &&
              std::string(it->order) == "orig" &&
              std::string(it->variant) == variant)
            return &it->stats;
        }
        return nullptr;
      };
      const double p_scan = med_field("prod50", &ReplayStats::scan_ns);
      const double p_prep = med_field("prod50", &ReplayStats::prep_ns);
      const double p_recall = med_ratio("prod50", &ReplayStats::recall_hits,
                                        &ReplayStats::gt_ids);
      const double p_ref = med_count("prod50", &ReplayStats::refines);
      std::cout << "  rerank prod50: scan_us/q=" << p_scan
                << " prep_us/q=" << p_prep << " refine/q=" << p_ref
                << " recall=" << p_recall * 100.0 << "%" << std::endl;
      std::vector<std::string> rerank_names = rerank_row_names;
      rerank_names.insert(rerank_names.end(), rerank_ctl_names.begin(),
                          rerank_ctl_names.end());
      for (const auto &name : rerank_names) {
        const double c_scan = med_field(name, &ReplayStats::scan_ns);
        const double c_prep = med_field(name, &ReplayStats::prep_ns);
        const double c_rr = med_field(name, &ReplayStats::rerank_ns);
        const double c_ref = med_count(name, &ReplayStats::refines);
        const double c_cov = med_ratio(name, &ReplayStats::coverage_hits,
                                       &ReplayStats::gt_ids);
        const double c_recall = med_ratio(name, &ReplayStats::recall_hits,
                                          &ReplayStats::gt_ids);
        const double total = c_scan + c_prep + c_rr;
        const double ptotal = p_scan + p_prep;
        const std::vector<double> ratios = paired_ratios(name);
        std::ostringstream rr;
        rr << std::fixed << std::setprecision(3);
        for (double x : ratios) rr << ' ' << x;
        std::cout << "  rerank " << name << ": scan_us/q=" << c_scan
                  << " prep_us/q=" << c_prep << " rerank_us/q=" << c_rr
                  << " refine/q=" << c_ref << " total_us/q=" << total
                  << " recall=" << c_recall * 100.0 << "% coverage="
                  << c_cov * 100.0 << "% | vs prod50 total med x"
                  << (ptotal ? total / ptotal : 0.0) << " paired" << rr.str()
                  << " recall delta " << (c_recall - p_recall) * 100.0
                  << "pt" << std::endl;
      }
      std::vector<std::string> verify_names = {"prod50"};
      verify_names.insert(verify_names.end(), rerank_row_names.begin(),
                          rerank_row_names.end());
      for (const auto &name : verify_names) {
        const ReplayStats *st = last_stats(name);
        if (!st || st->cmp_queries == 0) continue;
        std::cout << "  rerank verify " << name << ": queries="
                  << st->cmp_queries << " exact=" << st->cmp_exact
                  << " tie=" << st->cmp_tie
                  << " id_mismatch=" << st->cmp_id_mismatch
                  << " score_mismatch=" << st->cmp_score_mismatch
                  << " max_score_delta=" << st->cmp_max_delta << std::endl;
        if (st->cmp_id_mismatch || st->cmp_score_mismatch) ok = false;
      }
      std::cout << "  rerank scope: prod50 and rerank rows both exclude "
                   "probe selection and finalize; totals pair within one "
                   "invocation but are not immune to intra-invocation drift "
                   "(row order rotates per round)"
                << std::endl;
""", 'summary block: refines, paired ratios, verify report'))


def main():
    text = TARGET.read_text()
    for i, (old, new, note) in enumerate(R, 1):
        cnt = text.count(old)
        if cnt != 1:
            print(f'HUNK {i} ({note}): matched {cnt} times (expect 1) -- '
                  f'ABORT, no changes written')
            return 1
    for (old, new, note) in R:
        text = text.replace(old, new)
    TARGET.write_text(text)
    print(f'applied {len(R)}/{len(R)} hunks to {TARGET}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
