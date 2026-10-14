#!/usr/bin/env python3
"""v11-4b: candidate rerank experiment in ivf_rabitq_scan_replay.cc.

Adds --rerank-topn=N[,N...]: pairs, inside the same invocation, a production-
path replica row ("prod50": coarse estimate + interleaved lower-bound
pruning + BlockScanner::refine + top-50 heap, exactly like
IVFEntity::search_packed_impl with pruning on) with rerank rows ("rerankN":
coarse estimate + top-N collection by lower bound + original-vector exact
rerank from the .vecs file that built the index). Recall is measured against
the ground truth (first min(50, ids) per line, matching tools/core/recall).
The dedicated side is skipped in rerank mode. Count-asserted hunks.
"""
from pathlib import Path

TARGET = (Path.home() / 'workspace' / 'zvec' / 'rabitq_refactor' / 'zvec' /
          'tools' / 'core' / 'ivf_rabitq_scan_replay.cc')

HUNKS = [
    # 1. header docs: rerank experiment description before the scope block
    ('''//   flat16    flat copy shifted +16B (alignment sensitivity)
//
// Scope: the replay executes the coarse-estimate scan only -- query''',
     '''//   flat16    flat copy shifted +16B (alignment sensitivity)
//
// Candidate rerank experiment (--rerank-topn=N[,N...] with --vecs and --gt):
// the int side additionally runs "prod50" (production-path replica: coarse
// estimate + interleaved lower-bound pruning + refine + top-50 heap, minus
// probe selection and finalize) and "rerankN" (top-N candidates collected
// by the same lower-bound pruning with heap size N, then exact rerank with
// original vectors from the .vecs file). Recall uses the first
// min(50, ids) ground-truth ids per line, like tools/core/recall. The
// dedicated side is skipped in rerank mode.
//
// Scope: the replay executes the coarse-estimate scan only -- query'''),
    # 2. includes
    ('''#include <turbo/quantizer/quantizer.h>
#include <zvec/core/framework/index_metric.h>''',
     '''#include <turbo/quantizer/quantizer.h>
#include <zvec/core/framework/index_document.h>
#include <zvec/core/framework/index_metric.h>'''),
    ('''#include "algorithm/ivf/ivf_index_format.h"
#include "algorithm/ivf_rabitq/ivf_rabitq_index_format.h"
#include "algorithm/ivf_rabitq/ivf_rabitq_params.h"''',
     '''#include "algorithm/ivf/ivf_index_format.h"
#include "algorithm/ivf_rabitq/ivf_rabitq_index_format.h"
#include "algorithm/ivf_rabitq/ivf_rabitq_params.h"
#include "vecs_reader.h"'''),
    # 3. flags
    ('''DEFINE_string(int_variant, "prod",
              "integrated single-factor variant to pair with prod rows: "
              "prod|cluster0|dedshape|flat|flat16");
''',
     '''DEFINE_string(int_variant, "prod",
              "integrated single-factor variant to pair with prod rows: "
              "prod|cluster0|dedshape|flat|flat16");
DEFINE_string(vecs, "",
              "original .vecs file that built the index (--rerank-topn)");
DEFINE_string(gt, "",
              "ground truth file 'qid;id id ...' (--rerank-topn)");
DEFINE_string(rerank_topn, "",
              "comma-separated top-N cutoffs for the rerank experiment "
              "(e.g. 60,80,100); skips the dedicated side");
'''),
    # 4. usage
    ('''             "[--tolerance=1e-4] [--int-variant=prod]"
          << std::endl;''',
     '''             "[--tolerance=1e-4] [--int-variant=prod] "
             "[--rerank-topn=60,80,100 --vecs=FILE --gt=FILE]"
          << std::endl;'''),
    # 5. IntegratedIndex: row segment fields
    ('''  const uint8_t *coarse_base = nullptr;
  const uint8_t *key_base = nullptr;
  zvec::turbo::Quantizer::Pointer quantizer;
  size_t packed_size = 0;
  size_t block_vecs = 0;
  size_t scan_bytes = 0;''',
     '''  const uint8_t *coarse_base = nullptr;
  const uint8_t *key_base = nullptr;
  const uint8_t *row_base = nullptr;
  zvec::turbo::Quantizer::Pointer quantizer;
  size_t packed_size = 0;
  size_t block_vecs = 0;
  size_t block_size = 0;
  size_t stride = 0;
  size_t scan_bytes = 0;'''),
    # 6. LoadIntegrated: map the inverted body (row) segment
    ('''  auto key_seg = storage->get(IVF_KEYS_SEG_ID);
  if (!key_seg) {
    std::cerr << "Missing hc.keys" << std::endl;
    return false;
  }
  out->key_base = key_seg->base_data();
  if (!out->key_base) {
    std::cerr << "hc.keys is not mmap-backed" << std::endl;
    return false;
  }
''',
     '''  auto key_seg = storage->get(IVF_KEYS_SEG_ID);
  if (!key_seg) {
    std::cerr << "Missing hc.keys" << std::endl;
    return false;
  }
  out->key_base = key_seg->base_data();
  if (!out->key_base) {
    std::cerr << "hc.keys is not mmap-backed" << std::endl;
    return false;
  }
  auto row_seg = storage->get(IVF_INVERTED_BODY_SEG_ID);
  if (!row_seg) {
    std::cerr << "Missing ivf.inverted_body" << std::endl;
    return false;
  }
  out->row_base = row_seg->base_data();
  if (!out->row_base) {
    std::cerr << "ivf.inverted_body is not mmap-backed" << std::endl;
    return false;
  }
  out->stride = out->meta.element_size();
  if (out->stride != out->quantizer->quantized_datapoint_vector_length()) {
    std::cerr << "Row stride " << out->stride
              << " != quantized datapoint length "
              << out->quantizer->quantized_datapoint_vector_length()
              << std::endl;
    return false;
  }
  out->block_size = out->header.block_size;
  if (out->block_size != out->block_vecs * out->stride) {
    std::cerr << "block_size " << out->block_size
              << " != block_vector_count * stride "
              << out->block_vecs * out->stride << std::endl;
    return false;
  }
'''),
    # 7. ReplayStats: rerank fields
    ('''struct ReplayStats {
  double prep_ns = 0;
  double scan_ns = 0;
  size_t lists = 0;
  size_t vectors = 0;
  size_t blocks = 0;
  double checksum = 0;
};''',
     '''struct ReplayStats {
  double prep_ns = 0;
  double scan_ns = 0;
  double rerank_ns = 0;
  size_t lists = 0;
  size_t vectors = 0;
  size_t blocks = 0;
  double checksum = 0;
  size_t gt_ids = 0;
  size_t recall_hits = 0;
  size_t coverage_hits = 0;
};'''),
    # 8. rerank machinery + prod50/rerank replays, before the dedicated replay
    ('''    const auto t3 = Clock::now();
    if (timed) stats.scan_ns += ElapsedNs(t2, t3);
  }
  return stats;
}

// Dedicated replay. The timed scan loop matches IvfRabitqEntity's per-cluster''',
     '''    const auto t3 = Clock::now();
    if (timed) stats.scan_ns += ElapsedNs(t2, t3);
  }
  return stats;
}

// Original-vector store for the rerank experiment: the .vecs file that
// built the index; keys are not row-sequential, so a key->row map is built.
struct VecsStore {
  zvec::core::VecsReader reader;
  std::unordered_map<uint64_t, size_t> key_to_row;
};

bool LoadVecs(const std::string &path, VecsStore *out) {
  if (!out->reader.load(path)) {
    std::cerr << "Failed to load vecs file " << path << std::endl;
    return false;
  }
  out->key_to_row.reserve(out->reader.num_vecs());
  for (size_t i = 0; i < out->reader.num_vecs(); ++i)
    out->key_to_row.emplace(out->reader.get_key(i), i);
  std::cout << "vecs: vectors=" << out->reader.num_vecs()
            << " dim=" << out->reader.index_meta().dimension() << std::endl;
  return true;
}

// Ground truth: "qid;id id id ...", ranked; recall uses the first
// min(50, ids) ids per line, matching tools/core/recall.
using GroundTruth = std::unordered_map<long long, std::vector<uint64_t>>;

bool LoadGroundTruth(const std::string &path, GroundTruth *out) {
  std::ifstream in(path);
  if (!in) {
    std::cerr << "Failed to open ground truth " << path << std::endl;
    return false;
  }
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    const auto sep = line.find(';');
    if (sep == std::string::npos) {
      std::cerr << "Ground truth line without ';': " << line << std::endl;
      return false;
    }
    long long qid = 0;
    try {
      qid = std::stoll(line.substr(0, sep));
    } catch (...) {
      return false;
    }
    std::vector<uint64_t> ids;
    std::istringstream iss(line.substr(sep + 1));
    uint64_t id = 0;
    while (iss >> id) ids.push_back(id);
    if (ids.empty() || !out->emplace(qid, std::move(ids)).second) {
      std::cerr << "Bad or duplicate ground truth qid " << qid << std::endl;
      return false;
    }
  }
  return !out->empty();
}

void CountRecall(const std::vector<uint64_t> &gt, ReplayStats *stats,
                 const std::vector<uint64_t> &cands,
                 const std::vector<uint64_t> &topk) {
  const size_t n = std::min<size_t>(50, gt.size());
  stats->gt_ids += n;
  std::unordered_set<uint64_t> cand_set(cands.begin(), cands.end());
  for (size_t i = 0; i < n; ++i)
    if (cand_set.count(gt[i])) ++stats->coverage_hits;
  std::unordered_set<uint64_t> top_set(topk.begin(), topk.end());
  for (size_t i = 0; i < n; ++i)
    if (top_set.count(gt[i])) ++stats->recall_hits;
}

std::vector<uint64_t> HeapKeys(const IndexDocumentHeap &heap) {
  std::vector<uint64_t> keys;
  keys.reserve(heap.size());
  for (const auto &doc : heap.container()) keys.push_back(doc.key());
  return keys;
}

// Production-path replica (v11-4b): the per-list loop of
// IVFEntity::search_packed_impl with pruning on and topk=50 -- key-keep
// mask, block estimate, then interleaved rejected()/refine()/heap --
// minus probe selection and finalize. Its recall cross-checks the
// diagnostic tool's baseline results for the same recorded lists.
ReplayStats ReplayIntegratedProd50(
    const IntegratedIndex &idx,
    const std::vector<std::vector<float>> &queries,
    const std::vector<long long> &qids, const RecordedLists &rec,
    const GroundTruth &gt, bool timed) {
  ReplayStats stats;
  const IndexQueryMeta qmeta(idx.dtype, idx.dim);
  const auto &seq = rec.per_query;
  std::string encoded;
  IndexQueryMeta emeta;
  std::array<zvec::turbo::DistanceEstimate, 32> est;
  for (size_t q = 0; q < queries.size(); ++q) {
    const auto t0 = Clock::now();
    if (idx.quantizer->quantize(queries[q].data(), qmeta, &encoded, &emeta) !=
        0) {
      throw Fail{"integrated quantize failed at query " + std::to_string(q)};
    }
    auto distance = idx.quantizer->distance(encoded.data(), emeta);
    auto scanner = idx.quantizer->block_scanner(distance);
    if (!scanner) {
      throw Fail{"integrated block_scanner unavailable at query " +
                 std::to_string(q)};
    }
    const auto t1 = Clock::now();
    if (timed) stats.prep_ns += ElapsedNs(t0, t1);

    IndexDocumentHeap heap(50);
    float worst = std::numeric_limits<float>::infinity();
    const auto t2 = Clock::now();
    for (const auto &entry : seq[q]) {
      const uint32_t lid = entry.first;
      if (lid >= idx.lists.size()) {
        throw Fail{"list id " + std::to_string(lid) + " out of range"};
      }
      const auto &list = idx.lists[lid];
      if (list.vector_count != entry.second) {
        throw Fail{"query " + std::to_string(q) + " list " +
                   std::to_string(lid) + " replayed " +
                   std::to_string(list.vector_count) + " != recorded scan " +
                   std::to_string(entry.second)};
      }
      ++stats.lists;
      if (list.vector_count == 0) continue;
      const size_t coarse_offset = idx.scan_list_offsets[lid] * idx.packed_size;
      for (uint32_t b = 0; b < list.block_count; ++b) {
        const uint32_t count = static_cast<uint32_t>(
            std::min(idx.block_vecs,
                     static_cast<size_t>(list.vector_count) -
                         static_cast<size_t>(b) * idx.block_vecs));
        const uint64_t *keys = reinterpret_cast<const uint64_t *>(
            idx.key_base +
            (static_cast<size_t>(list.id_offset) +
             static_cast<size_t>(b) * idx.block_vecs) *
                sizeof(uint64_t));
        uint32_t keeps = 0;
        for (uint32_t i = 0; i < count; ++i) {
          if (keys[i] != kInvalidKey) keeps |= uint32_t(1) << i;
        }
        if (!keeps) continue;
        const void *packed =
            idx.coarse_base + coarse_offset +
            static_cast<size_t>(b) * idx.packed_size;
        if (scanner->estimate(packed, count, est.data()) != 0) {
          throw Fail{"estimate failed at query " + std::to_string(q) +
                     " list " + std::to_string(lid) + " block " +
                     std::to_string(b)};
        }
        stats.vectors += count;
        ++stats.blocks;
        const char *row = reinterpret_cast<const char *>(
            idx.row_base + list.offset +
            static_cast<size_t>(b) * idx.block_size);
        for (uint32_t i = 0; i < count; ++i) {
          if (!(keeps & (uint32_t(1) << i))) continue;
          if (heap.full() && est[i].lower_bound > worst) continue;
          const float score = scanner->refine(row + i * idx.stride, est[i]);
          heap.emplace(keys[i], score);
          if (heap.full()) worst = heap.begin()->score();
          stats.checksum += score;
        }
      }
    }
    const auto t3 = Clock::now();
    if (timed) stats.scan_ns += ElapsedNs(t2, t3);

    const auto it = gt.find(qids[q]);
    if (it == gt.end()) {
      throw Fail{"query " + std::to_string(q) + " qid " +
                 std::to_string(qids[q]) + " missing from ground truth"};
    }
    const std::vector<uint64_t> top = HeapKeys(heap);
    CountRecall(it->second, &stats, top, top);
  }
  return stats;
}

// Rerank experiment (v11-4b): collect the top-N candidates of the
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
  ReplayStats stats;
  const IndexQueryMeta qmeta(idx.dtype, idx.dim);
  const auto &seq = rec.per_query;
  std::string encoded;
  IndexQueryMeta emeta;
  std::array<zvec::turbo::DistanceEstimate, 32> est;
  std::vector<float> norm_query(idx.dim);
  std::vector<uint64_t> cand_keys;
  cand_keys.reserve(topn);
  for (size_t q = 0; q < queries.size(); ++q) {
    const auto t0 = Clock::now();
    if (idx.quantizer->quantize(queries[q].data(), qmeta, &encoded, &emeta) !=
        0) {
      throw Fail{"integrated quantize failed at query " + std::to_string(q)};
    }
    auto distance = idx.quantizer->distance(encoded.data(), emeta);
    auto scanner = idx.quantizer->block_scanner(distance);
    if (!scanner) {
      throw Fail{"integrated block_scanner unavailable at query " +
                 std::to_string(q)};
    }
    // Normalize like RabitqQuantizer::rotate does for kCosine.
    double norm2 = 0;
    for (size_t i = 0; i < idx.dim; ++i) {
      norm2 += static_cast<double>(queries[q][i]) * queries[q][i];
    }
    const float norm = static_cast<float>(std::sqrt(norm2));
    for (size_t i = 0; i < idx.dim; ++i) {
      norm_query[i] = norm > 0 ? queries[q][i] / norm : queries[q][i];
    }
    const auto t1 = Clock::now();
    if (timed) stats.prep_ns += ElapsedNs(t0, t1);

    IndexDocumentHeap collect(topn);
    float worst = std::numeric_limits<float>::infinity();
    const auto t2 = Clock::now();
    for (const auto &entry : seq[q]) {
      const uint32_t lid = entry.first;
      if (lid >= idx.lists.size()) {
        throw Fail{"list id " + std::to_string(lid) + " out of range"};
      }
      const auto &list = idx.lists[lid];
      if (list.vector_count != entry.second) {
        throw Fail{"query " + std::to_string(q) + " list " +
                   std::to_string(lid) + " replayed " +
                   std::to_string(list.vector_count) + " != recorded scan " +
                   std::to_string(entry.second)};
      }
      ++stats.lists;
      if (list.vector_count == 0) continue;
      const size_t coarse_offset = idx.scan_list_offsets[lid] * idx.packed_size;
      for (uint32_t b = 0; b < list.block_count; ++b) {
        const uint32_t count = static_cast<uint32_t>(
            std::min(idx.block_vecs,
                     static_cast<size_t>(list.vector_count) -
                         static_cast<size_t>(b) * idx.block_vecs));
        const uint64_t *keys = reinterpret_cast<const uint64_t *>(
            idx.key_base +
            (static_cast<size_t>(list.id_offset) +
             static_cast<size_t>(b) * idx.block_vecs) *
                sizeof(uint64_t));
        uint32_t keeps = 0;
        for (uint32_t i = 0; i < count; ++i) {
          if (keys[i] != kInvalidKey) keeps |= uint32_t(1) << i;
        }
        if (!keeps) continue;
        const void *packed =
            idx.coarse_base + coarse_offset +
            static_cast<size_t>(b) * idx.packed_size;
        if (scanner_or_fail(scanner_storage, packed, count, est.data()) != 0) {
          throw Fail{"estimate failed at query " + std::to_string(q) +
                     " list " + std::to_string(lid) + " block " +
                     std::to_string(b)};
        }
        stats.vectors += count;
        ++stats.blocks;
        for (uint32_t i = 0; i < count; ++i) {
          if (!(keeps & (uint32_t(1) << i))) continue;
          if (collect.full() && est[i].lower_bound > worst) continue;
          collect.emplace(keys[i], est[i].lower_bound);
          if (collect.full()) worst = collect.begin()->score();
        }
      }
    }
    const auto t3 = Clock::now();
    if (timed) stats.scan_ns += ElapsedNs(t2, t3);

    // Rerank the collected candidates with original vectors.
    cand_keys = HeapKeys(collect);
    IndexDocumentHeap result(50);
    for (uint64_t key : cand_keys) {
      const auto it = vecs.key_to_row.find(key);
      if (it == vecs.key_to_row.end()) {
        throw Fail{"candidate key " + std::to_string(key) + " not in vecs"};
      }
      const float *v = static_cast<const float *>(
          vecs.reader.get_vector(it->second));
      double normv2 = 0, dot = 0;
      for (size_t i = 0; i < idx.dim; ++i) {
        normv2 += static_cast<double>(v[i]) * v[i];
        dot += static_cast<double>(norm_query[i]) * v[i];
      }
      const float score =
          static_cast<float>(-dot / std::sqrt(normv2 > 0 ? normv2 : 1.0));
      result.emplace(key, score);
      stats.checksum += score;
    }
    const auto t4 = Clock::now();
    if (timed) stats.rerank_ns += ElapsedNs(t3, t4);

    const auto gt_it = gt.find(qids[q]);
    if (gt_it == gt.end()) {
      throw Fail{"query " + std::to_string(q) + " qid " +
                 std::to_string(qids[q]) + " missing from ground truth"};
    }
    CountRecall(gt_it->second, &stats, cand_keys, HeapKeys(result));
  }
  return stats;
}

// Dedicated replay. The timed scan loop matches IvfRabitqEntity's per-cluster'''),
    # 9. main: validation with rerank mode
    ('''    if (FLAGS_index_int.empty() || FLAGS_index_ded.empty() ||
        FLAGS_queries.empty() || FLAGS_lists_int.empty() ||
        FLAGS_lists_ded.empty() || FLAGS_csv.empty()) {''',
     '''    std::vector<uint32_t> rerank_topn_values;
    const bool rerank_mode = !FLAGS_rerank_topn.empty();
    if (rerank_mode) {
      std::istringstream iss(FLAGS_rerank_topn);
      std::string token;
      while (std::getline(iss, token, ',')) {
        if (token.empty()) continue;
        size_t parsed = 0;
        const unsigned long n = std::stoul(token, &parsed);
        if (parsed != token.size() || n == 0 || n > 65536) {
          std::cerr << "Invalid --rerank-topn entry " << token << std::endl;
          return 1;
        }
        rerank_topn_values.push_back(static_cast<uint32_t>(n));
      }
      if (rerank_topn_values.empty() || FLAGS_vecs.empty() ||
          FLAGS_gt.empty()) {
        std::cerr << "--rerank-topn requires --vecs and --gt" << std::endl;
        return 1;
      }
    }
    if (FLAGS_index_int.empty() || FLAGS_queries.empty() ||
        FLAGS_lists_int.empty() || FLAGS_csv.empty() ||
        (!rerank_mode &&
         (FLAGS_index_ded.empty() || FLAGS_lists_ded.empty()))) {'''),
    ('''    if ((FLAGS_samples_int.empty() != FLAGS_samples_ded.empty()) ||
        FLAGS_tolerance < 0) {''',
     '''    if (((FLAGS_samples_int.empty() != FLAGS_samples_ded.empty()) &&
         !rerank_mode) ||
        FLAGS_tolerance < 0) {'''),
    # 10. main: ded loading conditional
    ('''    IntegratedIndex int_idx;
    if (!LoadIntegrated(FLAGS_index_int, &int_idx)) return 1;
    DedicatedIndex ded_idx;
    if (!LoadDedicated(FLAGS_index_ded, &ded_idx)) return 1;
''',
     '''    IntegratedIndex int_idx;
    if (!LoadIntegrated(FLAGS_index_int, &int_idx)) return 1;
    DedicatedIndex ded_idx;
    bool ded_loaded = false;
    if (!rerank_mode) {
      if (!LoadDedicated(FLAGS_index_ded, &ded_idx)) return 1;
      ded_loaded = true;
    }
    VecsStore vecs;
    GroundTruth gt;
    if (rerank_mode) {
      if (!LoadVecs(FLAGS_vecs, &vecs) || !LoadGroundTruth(FLAGS_gt, &gt))
        return 1;
    }
'''),
    ('''    if (qs.dim != int_idx.dim || qs.dim != ded_idx.dim) {
      std::cerr << "Query dim " << qs.dim
                << " disagrees with index dims (int="
                << int_idx.dim << " ded=" << ded_idx.dim << ")"
                << std::endl;
      return 1;
    }''',
     '''    if (qs.dim != int_idx.dim || (ded_loaded && qs.dim != ded_idx.dim)) {
      std::cerr << "Query dim " << qs.dim
                << " disagrees with index dims (int="
                << int_idx.dim << " ded=" << ded_idx.dim << ")"
                << std::endl;
      return 1;
    }'''),
    # 11. main: rec_ded conditional
    ('''    if (!LoadListsCsv(FLAGS_lists_ded, "dedicated", qcount, qs.rows.size(),
                      &rec_ded)) {
      return 1;
    }
    if (rec_int.queries() != rec_ded.queries() ||
        rec_int.queries() != qcount) {
      std::cerr << "Query set mismatch between lists.csv files" << std::endl;
      return 1;
    }
    for (size_t q = 0; q < qcount; ++q) {
      if (rec_int.per_query[q].empty() != rec_ded.per_query[q].empty()) {
        std::cerr << "Query " << q << " present in one lists.csv only"
                  << std::endl;
        return 1;
      }
    }''',
     '''    if (ded_loaded) {
      if (!LoadListsCsv(FLAGS_lists_ded, "dedicated", qcount, qs.rows.size(),
                        &rec_ded)) {
        return 1;
      }
      if (rec_int.queries() != rec_ded.queries() ||
          rec_int.queries() != qcount) {
        std::cerr << "Query set mismatch between lists.csv files" << std::endl;
        return 1;
      }
      for (size_t q = 0; q < qcount; ++q) {
        if (rec_int.per_query[q].empty() != rec_ded.per_query[q].empty()) {
          std::cerr << "Query " << q << " present in one lists.csv only"
                    << std::endl;
          return 1;
        }
      }
    } else if (rec_int.queries() != qcount) {
      std::cerr << "lists.csv query set mismatch" << std::endl;
      return 1;
    }'''),
    # 12. main: ded stdout line conditional
    ('''    std::cout << "ded: lists_total=" << rec_ded.total_lists
              << " lists/query=" << double(rec_ded.total_lists) / qcount
              << " vectors_total=" << rec_ded.total_scan
              << " vectors/query=" << double(rec_ded.total_scan) / qcount
              << " batch_bytes=" << ded_idx.batch_bytes << std::endl;''',
     '''    if (ded_loaded) {
      std::cout << "ded: lists_total=" << rec_ded.total_lists
                << " lists/query=" << double(rec_ded.total_lists) / qcount
                << " vectors_total=" << rec_ded.total_scan
                << " vectors/query=" << double(rec_ded.total_scan) / qcount
                << " batch_bytes=" << ded_idx.batch_bytes << std::endl;
    }'''),
    # 13. main: samples block -- ded part conditional
    ('''    if (!FLAGS_samples_int.empty()) {
      std::vector<std::vector<SampleRow>> samples_int, samples_ded;
      size_t skipped_int = 0, skipped_ded = 0;
      if (!LoadSamplesCsv(FLAGS_samples_int, FLAGS_mode_int, qcount,
                          qs.rows.size(), &samples_int, &skipped_int) ||
          !LoadSamplesCsv(FLAGS_samples_ded, "dedicated", qcount,
                          qs.rows.size(), &samples_ded, &skipped_ded)) {
        std::cerr << "Failed to load samples.csv files" << std::endl;
        return 1;
      }''',
     '''    if (!FLAGS_samples_int.empty()) {
      std::vector<std::vector<SampleRow>> samples_int, samples_ded;
      size_t skipped_int = 0, skipped_ded = 0;
      if (!LoadSamplesCsv(FLAGS_samples_int, FLAGS_mode_int, qcount,
                          qs.rows.size(), &samples_int, &skipped_int) ||
          (!FLAGS_samples_ded.empty() &&
           !LoadSamplesCsv(FLAGS_samples_ded, "dedicated", qcount,
                           qs.rows.size(), &samples_ded, &skipped_ded))) {
        std::cerr << "Failed to load samples.csv files" << std::endl;
        return 1;
      }'''),
    ('''      const SampleVerify v_ded = VerifyDedicatedSamples(
          ded_idx, queries, rec_ded, samples_ded, FLAGS_tolerance);
      PrintSampleVerify("ded", "list_ids", v_ded, FLAGS_tolerance);
      if (v_ded.checked == 0) {
        std::cerr << "no dedicated samples verified (mode/column mismatch?)"
                  << std::endl;
        return 1;
      }
      if (v_int.key_mismatches || v_int.value_mismatches ||
          v_ded.key_mismatches || v_ded.value_mismatches) {
        std::cerr << "sample verification FAILED" << std::endl;
        return 1;
      }''',
     '''      if (!FLAGS_samples_ded.empty()) {
        const SampleVerify v_ded = VerifyDedicatedSamples(
            ded_idx, queries, rec_ded, samples_ded, FLAGS_tolerance);
        PrintSampleVerify("ded", "list_ids", v_ded, FLAGS_tolerance);
        if (v_ded.checked == 0) {
          std::cerr
              << "no dedicated samples verified (mode/column mismatch?)"
              << std::endl;
          return 1;
        }
        if (v_ded.key_mismatches || v_ded.value_mismatches) {
          std::cerr << "sample verification FAILED" << std::endl;
          return 1;
        }
      }
      if (v_int.key_mismatches || v_int.value_mismatches) {
        std::cerr << "sample verification FAILED" << std::endl;
        return 1;
      }'''),
    # 14. main: warmup -- ded conditional + rerank warmups
    ('''      if (flat_base) {
        ReplayIntegrated(int_idx, queries, rec_int, false, false, flat_base);
      }
      ReplayDedicated(ded_idx, queries, rec_ded, false, false, false);
      ReplayDedicated(ded_idx, queries, rec_ded, true, false, false);
      ReplayDedicated(ded_idx, queries, rec_ded, false, true, false);
      ReplayDedicated(ded_idx, queries, rec_ded, true, true, false);
    }''',
     '''      if (flat_base) {
        ReplayIntegrated(int_idx, queries, rec_int, false, false, flat_base);
      }
      if (rerank_mode) {
        ReplayIntegratedProd50(int_idx, queries, qs.qids, rec_int, gt,
                               false);
        for (uint32_t n : rerank_topn_values) {
          ReplayIntegratedRerank(int_idx, queries, qs.qids, rec_int, gt, n,
                                 vecs, false);
        }
      }
      if (ded_loaded) {
        ReplayDedicated(ded_idx, queries, rec_ded, false, false, false);
        ReplayDedicated(ded_idx, queries, rec_ded, true, false, false);
        ReplayDedicated(ded_idx, queries, rec_ded, false, true, false);
        ReplayDedicated(ded_idx, queries, rec_ded, true, true, false);
      }
    }'''),
    # 15. main: round loop -- rerank rows + ded conditional
    ('''              if (flat_base) {
                ReplayStats st = ReplayIntegrated(
                    int_idx, queries, rec_int, false, true, flat_base);
                rows.push_back(
                    {r, "int", "orig", iv_flat16 ? "flat16" : "flat",
                     std::move(st)});
              }
            }
          } else {''',
     '''              if (flat_base) {
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
          } else if (ded_loaded) {'''),
    # 16. main: stable storage for rerank row names, before the round loop
    ('''    std::vector<Row> rows;
    std::unordered_map<std::string, double> first_checksum;
    bool ok = true;
    for (int r = 1; r <= FLAGS_rounds; ++r) {''',
     '''    std::vector<std::string> rerank_row_names;
    for (uint32_t n : rerank_topn_values)
      rerank_row_names.push_back("rerank" + std::to_string(n));
    std::vector<Row> rows;
    std::unordered_map<std::string, double> first_checksum;
    bool ok = true;
    for (int r = 1; r <= FLAGS_rounds; ++r) {'''),
    # 17. CSV: extra columns
    ('''    out << "round,side,order,variant,queries,lists,vectors,blocks,"
           "prep_us_per_query,scan_us_per_query,ns_per_vector,ns_per_block,"
           "checksum\\n";''',
     '''    out << "round,side,order,variant,queries,lists,vectors,blocks,"
           "prep_us_per_query,scan_us_per_query,ns_per_vector,ns_per_block,"
           "checksum,rerank_us_per_query,recall,coverage\\n";'''),
    ('''          << (row.stats.blocks
                  ? row.stats.scan_ns / row.stats.blocks
                  : std::numeric_limits<double>::quiet_NaN())
          << ',' << row.stats.checksum << '\\n';
    }''',
     '''          << (row.stats.blocks
                  ? row.stats.scan_ns / row.stats.blocks
                  : std::numeric_limits<double>::quiet_NaN())
          << ',' << row.stats.checksum << ','
          << row.stats.rerank_ns / nq / 1000.0 << ','
          << (row.stats.gt_ids
                  ? double(row.stats.recall_hits) / row.stats.gt_ids
                  : std::numeric_limits<double>::quiet_NaN())
          << ','
          << (row.stats.gt_ids
                  ? double(row.stats.coverage_hits) / row.stats.gt_ids
                  : std::numeric_limits<double>::quiet_NaN())
          << '\\n';
    }'''),
    # 18. summary: rerank block after the factor loop
    ('''    for (const auto &name : extra_variants) {
      const double v_scan = median_scan("int", "orig", name);
      const double p_scan = median_scan("int", "orig", "full");
      const double d_scan = median_scan("ded", "orig", "full");
      std::cout << "  factor " << name << ": int scan " << v_scan
                << " us/q vs int prod " << p_scan << " (x"
                << (p_scan ? v_scan / p_scan : 0.0) << "), ded " << d_scan
                << " (x" << (d_scan ? v_scan / d_scan : 0.0) << ")"
                << std::endl;
    }''',
     '''    for (const auto &name : extra_variants) {
      const double v_scan = median_scan("int", "orig", name);
      const double p_scan = median_scan("int", "orig", "full");
      const double d_scan = median_scan("ded", "orig", "full");
      std::cout << "  factor " << name << ": int scan " << v_scan
                << " us/q vs int prod " << p_scan << " (x"
                << (p_scan ? v_scan / p_scan : 0.0) << "), ded " << d_scan
                << " (x" << (d_scan ? v_scan / d_scan : 0.0) << ")"
                << std::endl;
    }
    if (rerank_mode) {
      auto med_field = [&](const std::string &variant,
                           double ReplayStats::*field) {
        std::vector<double> vals;
        for (const auto &row : rows) {
          if (std::string(row.side) == "int" &&
              std::string(row.order) == "orig" &&
              std::string(row.variant) == variant)
            vals.push_back(row.stats.*field / qcount / 1000.0);
        }
        if (vals.empty()) return 0.0;
        std::sort(vals.begin(), vals.end());
        return vals[vals.size() / 2];
      };
      auto med_ratio = [&](const std::string &variant,
                            size_t ReplayStats::*hits,
                            size_t ReplayStats::*total) {
        std::vector<double> vals;
        for (const auto &row : rows) {
          if (std::string(row.side) == "int" &&
              std::string(row.order) == "orig" &&
              std::string(row.variant) == variant &&
              (row.stats.*total ? true : false))
            vals.push_back(double(row.stats.*hits) / (row.stats.*total));
        }
        if (vals.empty()) return 0.0;
        std::sort(vals.begin(), vals.end());
        return vals[vals.size() / 2];
      };
      const double p_scan = med_field("prod50", &ReplayStats::scan_ns);
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
    }'''),
]


def main():
    text = TARGET.read_text()
    n = 0
    for old, new in HUNKS:
        cnt = text.count(old)
        assert cnt == 1, f'hunk {n + 1} matched {cnt} times: {old[:70]!r}'
        text = text.replace(old, new)
        n += 1
    TARGET.write_text(text)
    print(f'applied {n}/{len(HUNKS)} hunks to {TARGET}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
