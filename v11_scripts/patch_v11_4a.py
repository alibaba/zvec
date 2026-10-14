#!/usr/bin/env python3
"""v11-4a: single-factor coarse-scan variants in ivf_rabitq_scan_replay.cc.

Factors isolated on the integrated side, paired with prod rows inside the
same invocation (immune to cross-invocation drift):
  cluster0  split_batch_estdist on the block's BatchDataMap with cluster-0
            coefficients -> removes per-row model-ID prefix load, g_add/
            g_error gather, scalar epilogue and AoS struct stores.
  dedshape  cluster0 without the key-keep mask walk.
  flat      production scanner over a page-aligned contiguous copy of the
            scan segment -> isolates mapped-segment locality.
  flat16    flat copy shifted +16B -> alignment sensitivity.
Count-asserted exact-match hunks.
"""
from pathlib import Path

TARGET = (Path.home() / 'workspace' / 'zvec' / 'rabitq_refactor' / 'zvec' /
          'tools' / 'core' / 'ivf_rabitq_scan_replay.cc')

HUNKS = [
    # 1. header docs
    ('''// list ids, order, block counts and tail valid-row counts are verified too.
//
// Scope: the replay executes the coarse-estimate scan only -- query''',
     '''// list ids, order, block counts and tail valid-row counts are verified too.
//
// Single-factor experiments (--int-variant): both coarse kernels share
// rabitqlib accumulate_hacc over an identical 3456B BatchDataMap; the
// integrated block adds a 128B model-ID prefix and a scalar per-row
// epilogue (prefix load + g_add/g_error gather + AoS struct stores) plus
// the key-keep mask. Variants isolate one factor at a time on the same
// bytes, paired with the production rows inside the same invocation:
//   cluster0  split_batch_estdist on the block's BatchDataMap with
//             cluster-0 coefficients (timing-valid, values approximate)
//   dedshape  cluster0 without the key-keep mask walk
//   flat      production scanner over a page-aligned contiguous copy of
//             the scan segment (mapped-segment locality)
//   flat16    flat copy shifted +16B (alignment sensitivity)
//
// Scope: the replay executes the coarse-estimate scan only -- query'''),
    # 2. flag
    ('''DEFINE_string(csv, "", "output csv prefix (required)");
''',
     '''DEFINE_string(csv, "", "output csv prefix (required)");
DEFINE_string(int_variant, "prod",
              "integrated single-factor variant to pair with prod rows: "
              "prod|cluster0|dedshape|flat|flat16");
'''),
    # 3. usage
    ('''             "[--warmup=1] [--samples-int=CSV --samples-ded=CSV] "
             "[--tolerance=1e-4]"
          << std::endl;''',
     '''             "[--warmup=1] [--samples-int=CSV --samples-ded=CSV] "
             "[--tolerance=1e-4] [--int-variant=prod]"
          << std::endl;'''),
    # 4. ReplayIntegrated signature + coarse base override
    ('''ReplayStats ReplayIntegrated(const IntegratedIndex &idx,
                             const std::vector<std::vector<float>> &queries,
                             const RecordedLists &rec, bool order_seq,
                             bool timed) {
  ReplayStats stats;
  const IndexQueryMeta qmeta(idx.dtype, idx.dim);
  const auto &seq = order_seq ? rec.sorted : rec.per_query;
  std::string encoded;
  IndexQueryMeta emeta;
  std::array<zvec::turbo::DistanceEstimate, 32> est;''',
     '''ReplayStats ReplayIntegrated(const IntegratedIndex &idx,
                             const std::vector<std::vector<float>> &queries,
                             const RecordedLists &rec, bool order_seq,
                             bool timed, const uint8_t *coarse_override) {
  ReplayStats stats;
  const IndexQueryMeta qmeta(idx.dtype, idx.dim);
  const auto &seq = order_seq ? rec.sorted : rec.per_query;
  const uint8_t *const coarse_base =
      coarse_override ? coarse_override : idx.coarse_base;
  std::string encoded;
  IndexQueryMeta emeta;
  std::array<zvec::turbo::DistanceEstimate, 32> est;'''),
    # ReplayIntegrated only; VerifyIntegratedSamples keeps idx.coarse_base
    # (verification must read the real mapped segment, not the flat copy).
    ('''        const void *packed =
            idx.coarse_base + coarse_offset +
            static_cast<size_t>(b) * idx.packed_size;
        if (scanner->estimate(packed, count, est.data()) != 0) {
          throw Fail{"estimate failed at query "''',
     '''        const void *packed =
            coarse_base + coarse_offset +
            static_cast<size_t>(b) * idx.packed_size;
        if (scanner->estimate(packed, count, est.data()) != 0) {
          throw Fail{"estimate failed at query "'''),
    # 5. variant replay function, inserted before the dedicated replay
    ('''    if (timed) stats.scan_ns += ElapsedNs(t2, t3);
  }
  return stats;
}

// Dedicated replay. The timed scan loop matches IvfRabitqEntity's per-cluster''',
     '''    if (timed) stats.scan_ns += ElapsedNs(t2, t3);
  }
  return stats;
}

// Single-factor variant (v11-4a): ded-shaped kernel work on the integrated
// block bytes. The encoded query blob carries the rotated query and the
// per-cluster scalars, so split_batch_estdist runs on the same data with the
// cluster-0 coefficients the production epilogue would gather. Values are
// approximate (rows may belong to other clusters); the byte traffic, kernel
// and store pattern are the dedicated ones. with_mask keeps the production
// key-keep walk so cluster0-vs-prod isolates only the epilogue and
// dedshape-vs-cluster0 isolates only the mask.
ReplayStats ReplayIntegratedDedShape(const IntegratedIndex &idx,
                                     const std::vector<std::vector<float>> &queries,
                                     const RecordedLists &rec, bool with_mask,
                                     bool timed) {
  ReplayStats stats;
  const IndexQueryMeta qmeta(idx.dtype, idx.dim);
  const auto &seq = rec.per_query;
  const std::string mname = idx.meta.metric_name();
  const bool l2_metric = mname == "kSquaredEuclidean" ||
                         mname == "SquaredEuclidean";
  std::string encoded;
  IndexQueryMeta emeta;
  std::array<float, 32> est, low, dot;
  std::vector<float> rotated_q;
  std::unique_ptr<rabitqlib::SplitBatchQuery<float>> bq;
  size_t padded = 0;
  uint32_t nclusters = 0;
  for (size_t q = 0; q < queries.size(); ++q) {
    const auto t0 = Clock::now();
    if (idx.quantizer->quantize(queries[q].data(), qmeta, &encoded, &emeta) !=
        0) {
      throw Fail{"integrated quantize failed at query " + std::to_string(q)};
    }
    if (padded == 0) {
      // Blob layout (RabitqQuantizer::quantize_query):
      //   [padded floats rotated][padded/2 B LUT][(4+2N) floats]
      // scalars: delta, vl, k1xsumq, kbxsumq, g_add[N], g_error[N].
      // Solve (padded, N) from the encoded length; require a unique answer.
      const size_t len = emeta.element_size();
      const size_t p_lo =
          (static_cast<size_t>(idx.dim) + 63) / 64 * 64;
      for (size_t p = p_lo; p < p_lo + 1024; p += 64) {
        const size_t base = p * 4 + p / 2 + 16;
        if (base > len) break;
        if ((len - base) % 8 != 0) continue;
        const size_t n = (len - base) / 8;
        if (n == 0 || n > 1024) continue;
        if (padded != 0) throw Fail{"ambiguous encoded query layout"};
        padded = p;
        nclusters = static_cast<uint32_t>(n);
      }
      if (padded == 0) throw Fail{"cannot decode encoded query layout"};
      rotated_q.resize(padded);
      bq.reset(new rabitqlib::SplitBatchQuery<float>(
          rotated_q.data(), padded, 6,
          l2_metric ? rabitqlib::METRIC_L2 : rabitqlib::METRIC_IP, true));
    }
    std::memcpy(rotated_q.data(), encoded.data(), padded * sizeof(float));
    bq->reset(rotated_q.data(), padded, 6,
              l2_metric ? rabitqlib::METRIC_L2 : rabitqlib::METRIC_IP, true);
    const float *scalars = reinterpret_cast<const float *>(
        encoded.data() + padded * sizeof(float) + padded * 4 / 8);
    const float g_add0 = scalars[4];
    const float g_error0 = scalars[4 + nclusters];
    if (l2_metric) {
      bq->set_g_add(g_error0);
    } else {
      bq->set_g_add(g_error0, -g_add0);
    }
    const auto t1 = Clock::now();
    if (timed) stats.prep_ns += ElapsedNs(t0, t1);

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
        if (with_mask) {
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
        }
        const char *batch = reinterpret_cast<const char *>(
            idx.coarse_base + coarse_offset +
            static_cast<size_t>(b) * idx.packed_size +
            32 * sizeof(uint32_t));
        rabitqlib::split_batch_estdist(batch, *bq, padded, est.data(),
                                       low.data(), dot.data(), true);
        stats.vectors += count;
        ++stats.blocks;
        stats.checksum += est[0] + est[count - 1];
      }
    }
    const auto t3 = Clock::now();
    if (timed) stats.scan_ns += ElapsedNs(t2, t3);
  }
  return stats;
}

// Dedicated replay. The timed scan loop matches IvfRabitqEntity's per-cluster'''),
    # 6. main: variant parsing + flat copy
    ('''    IntegratedIndex int_idx;
    if (!LoadIntegrated(FLAGS_index_int, &int_idx)) return 1;
    DedicatedIndex ded_idx;
    if (!LoadDedicated(FLAGS_index_ded, &ded_idx)) return 1;''',
     '''    IntegratedIndex int_idx;
    if (!LoadIntegrated(FLAGS_index_int, &int_idx)) return 1;
    DedicatedIndex ded_idx;
    if (!LoadDedicated(FLAGS_index_ded, &ded_idx)) return 1;

    // Single-factor variant selection (v11-4a).
    const std::string &int_variant = FLAGS_int_variant;
    bool iv_cluster0 = false, iv_dedshape = false, iv_flat = false,
         iv_flat16 = false;
    if (int_variant == "prod") {
    } else if (int_variant == "cluster0") {
      iv_cluster0 = true;
    } else if (int_variant == "dedshape") {
      iv_dedshape = true;
    } else if (int_variant == "flat") {
      iv_flat = true;
    } else if (int_variant == "flat16") {
      iv_flat16 = true;
    } else {
      std::cerr << "Unknown --int-variant " << int_variant << std::endl;
      return 1;
    }
    std::vector<uint8_t> flat_copy;
    const uint8_t *flat_base = nullptr;
    if (iv_flat || iv_flat16) {
      flat_copy.resize(int_idx.scan_bytes + 4096 + 64, 0);
      const uintptr_t raw = reinterpret_cast<uintptr_t>(flat_copy.data());
      const uintptr_t aligned = (raw + 4095) & ~uintptr_t(4095);
      flat_base = reinterpret_cast<const uint8_t *>(aligned) +
                  (iv_flat16 ? 16 : 0);
      std::memcpy(const_cast<uint8_t *>(flat_base), int_idx.coarse_base,
                  int_idx.scan_bytes);
      std::cout << "int-variant: flat copy of " << int_idx.scan_bytes
                << " bytes at " << (iv_flat16 ? "+16B" : "page-aligned")
                << std::endl;
    }
    std::vector<std::string> extra_variants;
    if (iv_cluster0) extra_variants.push_back("cluster0");
    if (iv_dedshape) extra_variants.push_back("dedshape");
    if (iv_flat) extra_variants.push_back("flat");
    if (iv_flat16) extra_variants.push_back("flat16");'''),
    # 7. warmup
    ('''    // Warmup both sides, both orders, both variants.
    for (int w = 0; w < FLAGS_warmup; ++w) {
      ReplayIntegrated(int_idx, queries, rec_int, false, false);
      ReplayIntegrated(int_idx, queries, rec_int, true, false);''',
     '''    // Warmup both sides, both orders, both variants, and the factor runs.
    for (int w = 0; w < FLAGS_warmup; ++w) {
      ReplayIntegrated(int_idx, queries, rec_int, false, false, nullptr);
      ReplayIntegrated(int_idx, queries, rec_int, true, false, nullptr);
      if (iv_cluster0) {
        ReplayIntegratedDedShape(int_idx, queries, rec_int, true, false);
      }
      if (iv_dedshape) {
        ReplayIntegratedDedShape(int_idx, queries, rec_int, false, false);
      }
      if (flat_base) {
        ReplayIntegrated(int_idx, queries, rec_int, false, false, flat_base);
      }'''),
    # 8. round loop: variant rows for the int side, orig order only
    ('''          if (is_int) {
            // Integrated has no per-list vector prepare, so kernel == full;
            // run both anyway so the CSV shows the comparison directly.
            for (int v = 0; v < 2; ++v) {
              ReplayStats st = ReplayIntegrated(
                  int_idx, queries, rec_int, seq_order, true);
              rows.push_back({r, "int", seq_order ? "seq" : "orig",
                              v == 0 ? "kernel" : "full", std::move(st)});
            }
          } else {''',
     '''          if (is_int) {
            // Integrated has no per-list vector prepare, so kernel == full;
            // run both anyway so the CSV shows the comparison directly.
            for (int v = 0; v < 2; ++v) {
              ReplayStats st = ReplayIntegrated(
                  int_idx, queries, rec_int, seq_order, true, nullptr);
              rows.push_back({r, "int", seq_order ? "seq" : "orig",
                              v == 0 ? "kernel" : "full", std::move(st)});
            }
            if (!seq_order) {
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
            }
          } else {'''),
    # 9. summary: extra variant labels + in-invocation factor deltas
    ('''    std::cout << "summary (median over " << FLAGS_rounds << " rounds):"
              << std::endl;
    for (const char *side : {"int", "ded"}) {
      for (const char *order : {"orig", "seq"}) {
        for (const char *variant : {"kernel", "full"}) {''',
     '''    auto median_scan = [&](const char *side, const char *order,
                           const std::string &variant) -> double {
      std::vector<double> vals;
      for (const auto &row : rows) {
        if (std::string(row.side) == side &&
            std::string(row.order) == order &&
            std::string(row.variant) == variant) {
          vals.push_back(row.stats.scan_ns / qcount / 1000.0);
        }
      }
      if (vals.empty()) return 0.0;
      std::sort(vals.begin(), vals.end());
      return vals[vals.size() / 2];
    };
    std::cout << "summary (median over " << FLAGS_rounds << " rounds):"
              << std::endl;
    std::vector<std::string> variant_names = {"kernel", "full"};
    for (const auto &name : extra_variants) variant_names.push_back(name);
    for (const char *side : {"int", "ded"}) {
      for (const char *order : {"orig", "seq"}) {
        for (const auto &variant : variant_names) {'''),
    ('''          if (scan_us.empty()) continue;
          std::sort(scan_us.begin(), scan_us.end());
          std::sort(prep_us.begin(), prep_us.end());
          std::sort(ns_vec.begin(), ns_vec.end());
          std::cout << "  " << side << '/' << order << '/' << variant
                    << ": scan_us/query=" << scan_us[scan_us.size() / 2]
                    << " prep_us/query=" << prep_us[prep_us.size() / 2]
                    << " ns/vector=" << ns_vec[ns_vec.size() / 2]
                    << std::endl;
        }
      }
    }''',
     '''          if (scan_us.empty()) continue;
          std::sort(scan_us.begin(), scan_us.end());
          std::sort(prep_us.begin(), prep_us.end());
          std::sort(ns_vec.begin(), ns_vec.end());
          std::cout << "  " << side << '/' << order << '/' << variant
                    << ": scan_us/query=" << scan_us[scan_us.size() / 2]
                    << " prep_us/query=" << prep_us[prep_us.size() / 2]
                    << " ns/vector=" << ns_vec[ns_vec.size() / 2]
                    << std::endl;
        }
      }
    }
    for (const auto &name : extra_variants) {
      const double v_scan = median_scan("int", "orig", name);
      const double p_scan = median_scan("int", "orig", "full");
      const double d_scan = median_scan("ded", "orig", "full");
      std::cout << "  factor " << name << ": int scan " << v_scan
                << " us/q vs int prod " << p_scan << " (x"
                << (p_scan ? v_scan / p_scan : 0.0) << "), ded " << d_scan
                << " (x" << (d_scan ? v_scan / d_scan : 0.0) << ")"
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
