// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Replays REAL recorded per-query list/block access sequences against the
// coarse-estimate kernels of both implementations, straight from the index
// segments. Unlike the synthetic kernel bench, every block, cluster id,
// coefficient set, byte layout and working-set page is the production one.
//
//   integrated side: RabitqQuantizer::quantize/distance/block_scanner +
//                    BlockScanner::estimate over ivf.turbo_scan.v1 blocks,
//                    including the production per-block key-keep logic.
//   dedicated  side: rabitqlib rotator + SplitBatchQuery +
//                    split_batch_estdist over ivf_rabitq.batch_data batches,
//                    with the production per-cluster set_g_add prepare.
//
// Both original (recorded) and sequential (sorted) list orders are timed,
// with the per-list prepare inside ("full") and outside ("kernel") the timer.
// Sides are interleaved across rounds; warmup rounds fault the mmap working
// set in, so timed rounds are cache-warm by construction.
//
// Work verification: per list, the replayed vector count must equal the
// scan_count recorded in lists.csv (checked hard); the tool also requires
// both lists.csv files to cover the identical query set, and rejects
// negative, out-of-range or duplicate query/list entries.
//
// With --samples-int/--samples-ded, every estimate row the production scan
// exported (key, distance, lower_bound, coarse_dot) is recomputed from the
// same segments and compared within --tolerance; the recorded sample rows
// must be consumed exactly by walking the recorded lists and blocks, so
// list ids, order, block counts and tail valid-row counts are verified too.
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
// Scope: the replay executes the coarse-estimate scan only -- query
// quantization/rotation, per-list/cluster prepare, the key-keep mask and
// the estimate kernels. It does NOT run heap maintenance, refinement
// (split_distance_boosting / BlockScanner::refine) or result finalization,
// so "kernel"/"full"/"unpruned" labels measure the coarse scan, not the
// complete pruning pipeline.
//
// Usage:
//   ivf_rabitq_scan_replay --index-int=PATH --index-ded=PATH \
//       --queries=FILE --lists-int=CSV --lists-ded=CSV \
//       [--mode-int=baseline] [--queries-count=N] [--rounds=5] \
//       [--warmup=1] [--samples-int=CSV --samples-ded=CSV] \
//       [--tolerance=1e-4] --csv=PREFIX
//
// Outputs PREFIX.replay.csv (round,side,order,variant,...) and a stdout
// summary. Exits nonzero on any verification failure.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <ailego/math/normalizer.h>
#include <gflags/gflags.h>
#include <turbo/quantizer/quantizer.h>
#include <zvec/core/framework/index_metric.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_helper.h>
#include <zvec/core/framework/index_storage.h>
#if RABITQ_SUPPORTED
#include <rabitqlib/index/estimator.hpp>
#include <rabitqlib/index/query.hpp>
#include <rabitqlib/quantization/data_layout.hpp>
#include <rabitqlib/utils/cpu_features.hpp>
#include <rabitqlib/utils/rotator.hpp>
#include <rabitqlib/utils/space.hpp>
#endif
#include "algorithm/ivf/ivf_index_format.h"
#include "algorithm/ivf_rabitq/ivf_rabitq_index_format.h"
#include "algorithm/ivf_rabitq/ivf_rabitq_params.h"

using namespace zvec::core;

DEFINE_string(index_int, "", "integrated IVF RaBitQ index file");
DEFINE_string(index_ded, "", "dedicated IVF RaBitQ index file");
DEFINE_string(queries, "", "query text file (qid;f1 f2 ...)");
DEFINE_string(lists_int, "", "integrated lists.csv from the diagnostic tool");
DEFINE_string(lists_ded, "", "dedicated lists.csv from the diagnostic tool");
DEFINE_string(mode_int, "baseline",
              "integrated mode column to replay (baseline|unpruned|reuse|"
              "reuse_unpruned)");
DEFINE_int32(queries_count, 0, "replay only the first N queries (0 = all)");
DEFINE_int32(rounds, 5, "timed rounds; sides are interleaved");
DEFINE_int32(warmup, 1, "untimed warmup rounds before timing");
DEFINE_string(samples_int, "",
              "integrated samples.csv from the diagnostic tool (verification)");
DEFINE_string(samples_ded, "",
              "dedicated samples.csv from the diagnostic tool (verification)");
DEFINE_double(tolerance, 1e-4,
              "max abs diff for sample value verification (values are expected "
              "to be bit-identical; the tolerance only guards drift)");
DEFINE_string(csv, "", "output csv prefix (required)");
DEFINE_string(int_variant, "prod",
              "integrated single-factor variant to pair with prod rows: "
              "prod|cluster0|dedshape|flat|flat16");


#if RABITQ_SUPPORTED
namespace {

using Clock = std::chrono::steady_clock;

double ElapsedNs(const Clock::time_point &a, const Clock::time_point &b) {
  return std::chrono::duration<double, std::nano>(b - a).count();
}

inline std::string LoadAvg() {
  std::ifstream in("/proc/loadavg");
  std::string one, two, three;
  if (in >> one >> two >> three) return one + ' ' + two + ' ' + three;
  return "n/a";
}

// Mirrors RabitqConverterHeader without pulling in reformer internals.
struct ConverterHeader {
  uint32_t num_clusters;
  uint32_t dim;
  uint32_t padded_dim;
  uint32_t rotator_size;
  uint8_t ex_bits;
  uint8_t rotator_type;
  uint8_t padding[2];
  uint32_t reserve[3];
};
static_assert(sizeof(ConverterHeader) == 32, "converter header must be 32B");

struct QuerySet {
  std::vector<std::vector<float>> rows;
  // File qid per row; informational only. The diagnostic CSVs record the
  // load-order row index of each query, not the file qid.
  std::vector<long long> qids;
  size_t dim = 0;
};

bool LoadQueries(const std::string &path, QuerySet *out) {
  std::ifstream in(path);
  if (!in) return false;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    const size_t sep = line.find(';');
    if (sep == std::string::npos) continue;
    const long long qid = std::stoll(line.substr(0, sep));
    std::istringstream iss(line.substr(sep + 1));
    std::vector<float> vals;
    float v;
    while (iss >> v) vals.push_back(v);
    if (vals.empty()) continue;
    if (out->dim == 0) out->dim = vals.size();
    if (vals.size() != out->dim) return false;
    out->qids.push_back(qid);
    out->rows.push_back(std::move(vals));
  }
  return !out->rows.empty() && out->qids.size() == out->rows.size();
}

struct RecordedLists {
  // query row -> recorded (list_id, scan_count) in real access order.
  std::vector<std::vector<std::pair<uint32_t, uint32_t>>> per_query;
  std::vector<std::vector<std::pair<uint32_t, uint32_t>>> sorted;
  size_t total_lists = 0;
  size_t total_scan = 0;
  size_t queries() const { return per_query.size(); }
};

bool LoadListsCsv(const std::string &path, const std::string &mode,
                  size_t qcount, size_t total_rows, RecordedLists *out) {
  std::ifstream in(path);
  if (!in) return false;
  std::string line;
  if (!std::getline(in, line)) return false;
  if (line != "query,mode,list_id,scan_count,limited") return false;
  // The diagnostic tool writes the load-order row index of the query into
  // the "query" column (not the file qid), so rows map 1:1 onto the loaded
  // query order and no qid translation may be applied on top.
  out->per_query.assign(qcount, {});
  size_t line_no = 1;
  size_t skipped = 0;
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
      std::cerr << "lists.csv " << path << ":" << line_no
                << " malformed line" << std::endl;
      return false;
    }
    if (f[1] != mode) continue;
    const long long q = std::stoll(f[0]);
    const unsigned long long lid = std::stoull(f[2]);
    const unsigned long long scan = std::stoull(f[3]);
    const int limited = std::stoi(f[4]);
    if (q < 0 || static_cast<unsigned long long>(q) >= total_rows) {
      std::cerr << "lists.csv " << path << ":" << line_no
                << " query row " << q << " out of range (" << total_rows
                << " queries loaded)" << std::endl;
      return false;
    }
    if (static_cast<size_t>(q) >= qcount) {
      // Belongs to a trimmed tail (--queries-count); not replayed.
      ++skipped;
      continue;
    }
    const size_t row = static_cast<size_t>(q);
    if (limited != 0) {
      std::cerr << "lists.csv " << path << ":" << line_no
                << " records a limited scan; replay requires max_scan_count "
                   "large enough to disable truncation"
                << std::endl;
      return false;
    }
    out->per_query[row].emplace_back(static_cast<uint32_t>(lid),
                                     static_cast<uint32_t>(scan));
    ++out->total_lists;
    out->total_scan += scan;
  }
  for (size_t r = 0; r < out->per_query.size(); ++r) {
    if (out->per_query[r].empty()) {
      std::cerr << "lists.csv " << path << " has no rows for query row " << r
                << " (mode=" << mode << ")" << std::endl;
      return false;
    }
    // Duplicate list entries within one query mean a broken recording.
    std::vector<uint32_t> ids;
    ids.reserve(out->per_query[r].size());
    for (const auto &e : out->per_query[r]) ids.push_back(e.first);
    std::sort(ids.begin(), ids.end());
    for (size_t i = 1; i < ids.size(); ++i) {
      if (ids[i] == ids[i - 1]) {
        std::cerr << "lists.csv " << path << " lists list " << ids[i]
                  << " twice for query row " << r << std::endl;
        return false;
      }
    }
  }
  if (skipped) {
    std::cout << "lists.csv " << path << ": skipped " << skipped
              << " rows beyond --queries-count=" << qcount << std::endl;
  }
  out->sorted = out->per_query;
  for (auto &lists : out->sorted) {
    std::sort(lists.begin(), lists.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });
  }
  return true;
}

const void *ReadWhole(const IndexStorage::Segment::Pointer &seg, size_t bytes,
                      const char *what) {
  const void *data = nullptr;
  if (!seg || seg->read(0, &data, bytes) != bytes) {
    std::cerr << "Failed to read " << what << std::endl;
    return nullptr;
  }
  return data;
}

struct IntegratedIndex {
  // Owns the mmaps backing coarse_base/key_base and any zero-copy quantizer
  // state; must outlive every replay call.
  IndexStorage::Pointer storage;
  InvertedIndexHeader header{};
  IndexMeta meta;
  std::vector<InvertedListMeta> lists;
  std::vector<size_t> scan_list_offsets;
  const uint8_t *coarse_base = nullptr;
  const uint8_t *key_base = nullptr;
  zvec::turbo::Quantizer::Pointer quantizer;
  size_t packed_size = 0;
  size_t block_vecs = 0;
  size_t scan_bytes = 0;
  uint32_t dim = 0;
  IndexMeta::DataType dtype = IndexMeta::DT_FP32;
};

bool LoadIntegrated(const std::string &path, IntegratedIndex *out) {
  out->storage = IndexFactory::CreateStorage("MMapFileReadStorage");
  auto &storage = out->storage;
  if (!storage || storage->open(path, false) != 0) {
    std::cerr << "Failed to open integrated index " << path << std::endl;
    return false;
  }
  auto header_seg = storage->get(IVF_INVERTED_HEADER_SEG_ID);
  if (!header_seg) {
    std::cerr << "Missing ivf.inverted_header (not an integrated index?)"
              << std::endl;
    return false;
  }
  const void *data = ReadWhole(header_seg, header_seg->data_size(),
                               "ivf.inverted_header");
  if (!data) return false;
  std::memcpy(&out->header, data, sizeof(out->header));
  if (out->header.header_size < sizeof(out->header) + out->header.index_meta_size ||
      out->header.header_size > header_seg->data_size()) {
    std::cerr << "Invalid ivf.inverted_header size" << std::endl;
    return false;
  }
  if (!out->meta.deserialize(
          reinterpret_cast<const uint8_t *>(data) + sizeof(out->header),
          out->header.index_meta_size)) {
    std::cerr << "Failed to deserialize IVF meta" << std::endl;
    return false;
  }
  if (out->meta.quantizer_name().empty()) {
    std::cerr << "IVF meta has no turbo quantizer" << std::endl;
    return false;
  }

  // Quantizer: same restore path as IVFEntity::load_quantizer.
  IndexMeta raw_meta;
  if (IndexHelper::DeserializeFromStorage(storage.get(), &raw_meta) != 0) {
    std::cerr << "Failed to read top-level meta" << std::endl;
    return false;
  }
  raw_meta.set_quantizer(out->meta.quantizer_name(),
                         out->meta.quantizer_revision(),
                         out->meta.quantizer_params());
  out->quantizer = IndexFactory::CreateQuantizer(out->meta.quantizer_name());
  if (!out->quantizer) {
    std::cerr << "Failed to create quantizer " << out->meta.quantizer_name()
              << std::endl;
    return false;
  }
  if (out->quantizer->init(raw_meta, out->meta.quantizer_params()) != 0) {
    std::cerr << "Failed to init quantizer" << std::endl;
    return false;
  }
  auto qseg = storage->get(IVF_TURBO_QUANTIZER_SEG_ID);
  if (!qseg || qseg->data_size() == 0) {
    std::cerr << "Missing ivf.turbo_quantizer state" << std::endl;
    return false;
  }
  const void *qdata = ReadWhole(qseg, qseg->data_size(), "ivf.turbo_quantizer");
  if (!qdata) return false;
  if (out->quantizer->deserialize(qdata, qseg->data_size()) != 0) {
    std::cerr << "Failed to restore quantizer" << std::endl;
    return false;
  }
  out->dim = raw_meta.dimension();
  out->dtype = raw_meta.data_type();
  out->packed_size = out->quantizer->scan_block_size();
  if (out->packed_size == 0) {
    std::cerr << "Quantizer has no packed scan layout" << std::endl;
    return false;
  }
  out->block_vecs = out->header.block_vector_count;
  if (out->block_vecs == 0 || out->block_vecs > 32) {
    std::cerr << "Unsupported block_vector_count="
              << out->header.block_vector_count << std::endl;
    return false;
  }

  const size_t list_count = out->header.inverted_list_count;
  const size_t meta_bytes = list_count * sizeof(InvertedListMeta);
  const void *mdata =
      ReadWhole(storage->get(IVF_INVERTED_META_SEG_ID), meta_bytes,
                "ivf.inverted_meta");
  if (!mdata) return false;
  out->lists.resize(list_count);
  std::memcpy(out->lists.data(), mdata, meta_bytes);

  auto scan_seg = storage->get(IVF_TURBO_SCAN_SEG_ID);
  if (!scan_seg) {
    std::cerr << "Missing ivf.turbo_scan.v1" << std::endl;
    return false;
  }
  out->coarse_base = scan_seg->base_data();
  out->scan_bytes = scan_seg->data_size();
  if (!out->coarse_base) {
    std::cerr << "ivf.turbo_scan.v1 is not mmap-backed" << std::endl;
    return false;
  }
  auto key_seg = storage->get(IVF_KEYS_SEG_ID);
  if (!key_seg) {
    std::cerr << "Missing hc.keys" << std::endl;
    return false;
  }
  out->key_base = key_seg->base_data();
  if (!out->key_base) {
    std::cerr << "hc.keys is not mmap-backed" << std::endl;
    return false;
  }

  out->scan_list_offsets.assign(list_count, 0);
  size_t total_blocks = 0;
  for (size_t l = 0; l < list_count; ++l) {
    const auto &list = out->lists[l];
    const size_t expect_blocks =
        (static_cast<size_t>(list.vector_count) + out->block_vecs - 1) /
        out->block_vecs;
    if (list.block_count != expect_blocks) {
      std::cerr << "List " << l << " block_count " << list.block_count
                << " != ceil(vector_count/" << out->block_vecs << ")="
                << expect_blocks << std::endl;
      return false;
    }
    out->scan_list_offsets[l] = total_blocks;
    total_blocks += list.block_count;
  }
  if (total_blocks != out->header.block_count ||
      scan_seg->data_size() != total_blocks * out->packed_size) {
    std::cerr << "Scan block segment size mismatch" << std::endl;
    return false;
  }
  return true;
}

struct DedicatedIndex {
  // Owns the mmap backing batch_base.
  IndexStorage::Pointer storage;
  IvfRabitqHeader header{};
  std::vector<IvfRabitqClusterMeta> clusters;
  const uint8_t *batch_base = nullptr;
  const uint64_t *key_base = nullptr;
  size_t batch_stride = 0;
  size_t batch_bytes = 0;
  uint32_t dim = 0;
  uint32_t padded = 0;
  uint8_t ex_bits = 0;
  rabitqlib::MetricType mtype = rabitqlib::METRIC_L2;
  std::vector<float> rotated_centroids;
  std::vector<float> centroids;  // unrotated, num_clusters * dim
  std::vector<float> centroid_norm_sqs;
  IndexMetric::Pointer centroid_metric;
  IndexMetric::MatrixDistance centroid_distance_func;
  std::unique_ptr<rabitqlib::Rotator<float>> rotator;
};

bool LoadDedicated(const std::string &path, DedicatedIndex *out) {
  out->storage = IndexFactory::CreateStorage("MMapFileReadStorage");
  auto &storage = out->storage;
  if (!storage || storage->open(path, false) != 0) {
    std::cerr << "Failed to open dedicated index " << path << std::endl;
    return false;
  }
  const void *data = ReadWhole(storage->get(IVF_RABITQ_HEADER_SEG_ID),
                                sizeof(out->header), "ivf_rabitq.header");
  if (!data) return false;
  std::memcpy(&out->header, data, sizeof(out->header));
  if (out->header.magic != kIvfRabitqIndexMagic) {
    std::cerr << "Bad ivf_rabitq header magic" << std::endl;
    return false;
  }
  out->dim = out->header.dimension;
  out->padded = out->header.padded_dim;
  out->ex_bits = out->header.ex_bits;
  out->mtype = out->header.metric_type == 1 ? rabitqlib::METRIC_IP
                                            : rabitqlib::METRIC_L2;

  const size_t cluster_count = out->header.cluster_count;
  const size_t meta_bytes = cluster_count * sizeof(IvfRabitqClusterMeta);
  const void *mdata = ReadWhole(storage->get(IVF_RABITQ_CLUSTER_META_SEG_ID),
                                meta_bytes, "ivf_rabitq.cluster_meta");
  if (!mdata) return false;
  out->clusters.resize(cluster_count);
  std::memcpy(out->clusters.data(), mdata, meta_bytes);

  auto batch_seg = storage->get(IVF_RABITQ_BATCH_DATA_SEG_ID);
  if (!batch_seg) {
    std::cerr << "Missing ivf_rabitq.batch_data" << std::endl;
    return false;
  }
  out->batch_base = batch_seg->base_data();
  out->batch_bytes = batch_seg->data_size();
  if (!out->batch_base) {
    std::cerr << "ivf_rabitq.batch_data is not mmap-backed" << std::endl;
    return false;
  }
  out->batch_stride = rabitqlib::BatchDataMap<float>::data_bytes(out->padded);
  size_t expect_batches = 0;
  for (size_t c = 0; c < cluster_count; ++c) {
    const auto &cm = out->clusters[c];
    const size_t expect =
        (static_cast<size_t>(cm.vector_count) +
         rabitqlib::fastscan::kBatchSize - 1) /
        rabitqlib::fastscan::kBatchSize;
    if (cm.batch_count != expect) {
      std::cerr << "Cluster " << c << " batch_count " << cm.batch_count
                << " != ceil(vector_count/32)=" << expect << std::endl;
      return false;
    }
    expect_batches += cm.batch_count;
  }
  if (out->batch_bytes != expect_batches * out->batch_stride) {
    std::cerr << "batch_data size mismatch" << std::endl;
    return false;
  }

  // Keys segment (same layout IvfRabitqEntity loads: one u64 per vector,
  // addressed as keys[key_offset + within-cluster index]).
  auto key_seg = storage->get(IVF_RABITQ_KEYS_SEG_ID);
  const size_t keys_size =
      static_cast<size_t>(out->header.total_vector_count) * sizeof(uint64_t);
  if (!key_seg || key_seg->data_size() != keys_size) {
    std::cerr << "Bad ivf_rabitq.keys segment size" << std::endl;
    return false;
  }
  out->key_base = reinterpret_cast<const uint64_t *>(key_seg->base_data());
  if (!out->key_base) {
    std::cerr << "ivf_rabitq.keys is not mmap-backed" << std::endl;
    return false;
  }

  // Converter segment: [header][rotated centroids][centroids][rotator blob].
  auto conv_seg = storage->get("rabitq.converter");
  if (!conv_seg) {
    std::cerr << "Missing rabitq.converter" << std::endl;
    return false;
  }
  ConverterHeader conv;
  const void *cdata = ReadWhole(conv_seg, sizeof(conv), "rabitq.converter");
  if (!cdata) return false;
  std::memcpy(&conv, cdata, sizeof(conv));
  if (conv.num_clusters != out->header.cluster_count ||
      conv.dim != out->dim || conv.padded_dim != out->padded) {
    std::cerr << "Converter header disagrees with index header" << std::endl;
    return false;
  }
  const size_t rotated_bytes =
      static_cast<size_t>(conv.num_clusters) * conv.padded_dim * sizeof(float);
  const size_t centroid_bytes =
      static_cast<size_t>(conv.num_clusters) * conv.dim * sizeof(float);
  if (conv_seg->data_size() <
      sizeof(conv) + rotated_bytes + centroid_bytes + conv.rotator_size) {
    std::cerr << "rabitq.converter too small" << std::endl;
    return false;
  }
  out->rotated_centroids.resize(static_cast<size_t>(conv.num_clusters) *
                                conv.padded_dim);
  {
    const void *rc = nullptr;
    if (conv_seg->read(sizeof(conv), &rc, rotated_bytes) != rotated_bytes) {
      std::cerr << "Failed to read rotated centroids" << std::endl;
      return false;
    }
    std::memcpy(out->rotated_centroids.data(), rc, rotated_bytes);
  }
  out->centroids.resize(static_cast<size_t>(conv.num_clusters) * out->dim);
  {
    const void *cc = nullptr;
    if (conv_seg->read(sizeof(conv) + rotated_bytes, &cc, centroid_bytes) !=
        centroid_bytes) {
      std::cerr << "Failed to read centroids" << std::endl;
      return false;
    }
    std::memcpy(out->centroids.data(), cc, centroid_bytes);
  }
  // Same accumulation as IvfRabitqReformer::load (std::inner_product over the
  // unrotated centroids), so the g_add inputs are bit-identical.
  out->centroid_norm_sqs.resize(static_cast<size_t>(conv.num_clusters));
  for (size_t i = 0; i < out->centroid_norm_sqs.size(); ++i) {
    const float *centroid = out->centroids.data() + i * out->dim;
    out->centroid_norm_sqs[i] =
        std::inner_product(centroid, centroid + out->dim, centroid, 0.0f);
  }
  {
    // The exact centroid distance function IvfRabitqReformer uses for probe
    // selection and per-cluster g_add preparation.
    IndexMeta centroid_meta;
    centroid_meta.set_data_type(IndexMeta::DataType::DT_FP32);
    centroid_meta.set_dimension(out->dim);
    centroid_meta.set_metric(
        out->mtype == rabitqlib::METRIC_L2 ? "SquaredEuclidean"
                                           : "InnerProduct",
        0, zvec::ailego::Params());
    out->centroid_metric =
        IndexFactory::CreateMetric(centroid_meta.metric_name());
    if (!out->centroid_metric ||
        out->centroid_metric->init(centroid_meta,
                                   centroid_meta.metric_params()) != 0) {
      std::cerr << "Failed to create centroid metric" << std::endl;
      return false;
    }
    out->centroid_distance_func = out->centroid_metric->distance_matrix(1, 1);
    if (!out->centroid_distance_func) {
      std::cerr << "Failed to get centroid distance function" << std::endl;
      return false;
    }
  }
  {
    IndexStorage::MemoryBlock block;
    if (conv_seg->read(sizeof(conv) + rotated_bytes + centroid_bytes, block,
                       conv.rotator_size) != conv.rotator_size) {
      std::cerr << "Failed to read rotator blob" << std::endl;
      return false;
    }
    out->rotator.reset(rabitqlib::choose_rotator<float>(
        out->dim, static_cast<rabitqlib::RotatorType>(conv.rotator_type),
        out->padded));
    if (!out->rotator) {
      std::cerr << "Failed to create rotator" << std::endl;
      return false;
    }
    out->rotator->load(reinterpret_cast<const char *>(block.data()));
  }
  return true;
}

// Mirrors CosineNormalizeReformer::transform's FP32 fast path: the kCosine
// query pipeline L2-normalizes a copy of the raw query (ailego::Normalizer)
// before IvfRabitqReformer::create_query_state rotates it.
void NormalizeQueryFp32(const float *query, size_t dim,
                        std::vector<float> *normalized) {
  normalized->assign(query, query + dim);
  float norm = 0.0f;
  zvec::ailego::Normalizer<float>::L2(normalized->data(), dim, &norm);
  (void)norm;
}

// Mirrors IvfRabitqReformer::select_probe_centroids +
// prepare_for_cluster: production derives the per-cluster g_add inputs from
// the unrotated centroid distance of the normalized query, not from
// rotated-space residuals.
struct ClusterPrepare {
  float residual_norm = 0;
  float inner_product = 0;
};

ClusterPrepare PrepareCluster(const DedicatedIndex &idx,
                              const std::vector<float> &nquery,
                              float query_norm_sq, uint32_t cid) {
  ClusterPrepare p;
  float dist = 0.0f;
  idx.centroid_distance_func(
      idx.centroids.data() + static_cast<size_t>(cid) * idx.dim, nquery.data(),
      idx.dim, &dist);
  if (idx.mtype == rabitqlib::METRIC_L2) {
    p.residual_norm = std::sqrt(std::max(dist, 0.0f));
  } else {
    p.inner_product = -dist;
    const float residual_norm_sq =
        query_norm_sq + idx.centroid_norm_sqs[cid] - 2.0f * p.inner_product;
    p.residual_norm = std::sqrt(std::max(residual_norm_sq, 0.0f));
  }
  return p;
}

struct ReplayStats {
  double prep_ns = 0;
  double scan_ns = 0;
  size_t lists = 0;
  size_t vectors = 0;
  size_t blocks = 0;
  double checksum = 0;
};

struct Fail {
  std::string msg;
};

struct SampleRow {
  size_t query;
  uint32_t list_id;
  uint32_t block;
  uint32_t row;
  uint64_t key;
  float distance;
  float lower_bound;
  float coarse_dot;
};

// Loads the diagnostic tool's samples.csv for one side. The "query" column
// is the load-order row index. Rows beyond --queries-count (trimmed tail)
// are skipped and counted; malformed or duplicate rows are hard errors.
bool LoadSamplesCsv(const std::string &path, const std::string &mode,
                    size_t qcount, size_t total_rows,
                    std::vector<std::vector<SampleRow>> *out,
                    size_t *skipped) {
  *skipped = 0;
  std::ifstream in(path);
  if (!in) return false;
  std::string line;
  if (!std::getline(in, line)) return false;
  if (line !=
      "query,mode,list_id,block,row,key,distance,lower_bound,coarse_dot") {
    std::cerr << "samples.csv " << path << " has an unexpected header"
              << std::endl;
    return false;
  }
  out->assign(qcount, {});
  size_t line_no = 1;
  while (std::getline(in, line)) {
    ++line_no;
    if (line.empty()) continue;
    std::array<std::string, 9> f;
    size_t pos = 0;
    bool ok = true;
    for (size_t i = 0; i < 9; ++i) {
      const size_t next = line.find(',', pos);
      if ((i < 8) != (next != std::string::npos)) {
        ok = false;
        break;
      }
      f[i] = line.substr(pos, next - pos);
      pos = next + 1;
    }
    if (!ok) {
      std::cerr << "samples.csv " << path << ":" << line_no
                << " malformed line" << std::endl;
      return false;
    }
    if (f[1] != mode) continue;
    const long long q = std::stoll(f[0]);
    if (q < 0 || static_cast<unsigned long long>(q) >= total_rows) {
      std::cerr << "samples.csv " << path << ":" << line_no
                << " query row " << q << " out of range (" << total_rows
                << " queries loaded)" << std::endl;
      return false;
    }
    if (static_cast<size_t>(q) >= qcount) {
      ++*skipped;
      continue;
    }
    SampleRow s;
    s.query = static_cast<size_t>(q);
    s.list_id = static_cast<uint32_t>(std::stoul(f[2]));
    s.block = static_cast<uint32_t>(std::stoul(f[3]));
    s.row = static_cast<uint32_t>(std::stoul(f[4]));
    s.key = std::stoull(f[5]);
    s.distance = std::stof(f[6]);
    s.lower_bound = std::stof(f[7]);
    s.coarse_dot = std::stof(f[8]);
    (*out)[s.query].push_back(s);
  }
  for (size_t r = 0; r < out->size(); ++r) {
    const auto &rows = (*out)[r];
    if (rows.size() < 2) continue;
    std::vector<const SampleRow *> ptrs;
    ptrs.reserve(rows.size());
    for (const auto &s : rows) ptrs.push_back(&s);
    std::sort(ptrs.begin(), ptrs.end(),
              [](const SampleRow *a, const SampleRow *b) {
                if (a->list_id != b->list_id) return a->list_id < b->list_id;
                if (a->block != b->block) return a->block < b->block;
                return a->row < b->row;
              });
    for (size_t i = 1; i < ptrs.size(); ++i) {
      if (ptrs[i]->list_id == ptrs[i - 1]->list_id &&
          ptrs[i]->block == ptrs[i - 1]->block &&
          ptrs[i]->row == ptrs[i - 1]->row) {
        std::cerr << "samples.csv " << path
                  << ": duplicate sample row for query " << r << " list "
                  << ptrs[i]->list_id << " block " << ptrs[i]->block
                  << " row " << ptrs[i]->row << std::endl;
        return false;
      }
    }
  }
  return true;
}

struct SampleVerify {
  size_t checked = 0;
  size_t queries = 0;
  size_t key_mismatches = 0;
  size_t value_mismatches = 0;
  size_t tail_blocks = 0;
  double max_distance_err = 0;
  double max_lower_err = 0;
  double max_dot_err = 0;
  // Integrated: per-record model ids (u32 block prefix); dedicated: list ids.
  std::map<uint32_t, size_t> ids;
};

void PrintSampleVerify(const char *side, const char *id_kind,
                       const SampleVerify &v, double tol) {
  std::cout << "samples " << side << ": checked=" << v.checked
            << " queries_with_samples=" << v.queries
            << " key_mismatches=" << v.key_mismatches
            << " value_mismatches=" << v.value_mismatches
            << " tail_blocks=" << v.tail_blocks
            << " distinct_" << id_kind << "=" << v.ids.size()
            << " max_err distance=" << v.max_distance_err
            << " lower_bound=" << v.max_lower_err
            << " coarse_dot=" << v.max_dot_err << " tolerance=" << tol
            << std::endl;
}

// Recomputes every recorded integrated sample row by walking the recorded
// lists in order and the blocks of each list exactly like
// IVFEntity::search_packed_impl (key-keep mask included). The recorded rows
// must be consumed in order, which verifies list ids, order, scan counts,
// block counts and tail valid rows in addition to the estimate values.
SampleVerify VerifyIntegratedSamples(
    const IntegratedIndex &idx, const std::vector<std::vector<float>> &queries,
    const RecordedLists &rec, const std::vector<std::vector<SampleRow>> &samples,
    double tol) {
  SampleVerify v;
  const IndexQueryMeta qmeta(idx.dtype, idx.dim);
  std::string encoded;
  IndexQueryMeta emeta;
  std::array<zvec::turbo::DistanceEstimate, 32> est;
  for (size_t q = 0; q < queries.size(); ++q) {
    if (samples[q].empty()) continue;
    ++v.queries;
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
    const auto &rows = samples[q];
    size_t ri = 0;
    for (const auto &entry : rec.per_query[q]) {
      const uint32_t lid = entry.first;
      if (lid >= idx.lists.size()) {
        throw Fail{"sampled list id " + std::to_string(lid) + " out of range"};
      }
      const auto &list = idx.lists[lid];
      if (list.vector_count != entry.second) {
        throw Fail{"query " + std::to_string(q) + " list " +
                   std::to_string(lid) + " replayed " +
                   std::to_string(list.vector_count) + " != recorded scan " +
                   std::to_string(entry.second)};
      }
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
        if (ri >= rows.size() || rows[ri].list_id != lid ||
            rows[ri].block != b) {
          continue;
        }
        const void *packed =
            idx.coarse_base + coarse_offset +
            static_cast<size_t>(b) * idx.packed_size;
        if (scanner->estimate(packed, count, est.data()) != 0) {
          throw Fail{"sample estimate failed at query " + std::to_string(q) +
                     " list " + std::to_string(lid) + " block " +
                     std::to_string(b)};
        }
        if (count < idx.block_vecs) ++v.tail_blocks;
        while (ri < rows.size() && rows[ri].list_id == lid &&
               rows[ri].block == b) {
          const SampleRow &s = rows[ri];
          ++ri;
          if (s.row >= count) {
            throw Fail{"query " + std::to_string(q) + " list " +
                       std::to_string(lid) + " block " + std::to_string(b) +
                       " sampled row " + std::to_string(s.row) +
                       " beyond tail count " + std::to_string(count)};
          }
          uint32_t model = 0;
          std::memcpy(&model,
                      static_cast<const uint8_t *>(packed) +
                          static_cast<size_t>(s.row) * sizeof(uint32_t),
                      sizeof(model));
          ++v.ids[model];
          if (s.key != keys[s.row]) {
            ++v.key_mismatches;
            std::cerr << "sample key mismatch: query " << q << " list " << lid
                      << " block " << b << " row " << s.row << std::endl;
            continue;
          }
          const double d_err =
              std::abs(double(est[s.row].distance) - double(s.distance));
          const double l_err =
              std::abs(double(est[s.row].lower_bound) - double(s.lower_bound));
          const double c_err =
              std::abs(double(est[s.row].coarse_dot) - double(s.coarse_dot));
          v.max_distance_err = std::max(v.max_distance_err, d_err);
          v.max_lower_err = std::max(v.max_lower_err, l_err);
          v.max_dot_err = std::max(v.max_dot_err, c_err);
          if (d_err > tol || l_err > tol || c_err > tol) {
            ++v.value_mismatches;
            std::cerr << "sample value mismatch: query " << q << " list "
                      << lid << " block " << b << " row " << s.row
                      << " distance " << s.distance << " vs "
                      << est[s.row].distance << " lower_bound " << s.lower_bound
                      << " vs " << est[s.row].lower_bound << " coarse_dot "
                      << s.coarse_dot << " vs " << est[s.row].coarse_dot
                      << std::endl;
          }
          ++v.checked;
        }
      }
    }
    if (ri != rows.size()) {
      throw Fail{"query " + std::to_string(q) + ": " +
                 std::to_string(rows.size() - ri) +
                 " recorded sample rows were not matched by the replay walk "
                 "(missing or out-of-order lists/blocks)"};
    }
  }
  return v;
}

// Recomputes every recorded dedicated sample row by walking the recorded
// clusters in order and their batches exactly like
// IvfRabitqEntity::search_cluster_impl (per-cluster set_g_add included).
SampleVerify VerifyDedicatedSamples(
    const DedicatedIndex &idx, const std::vector<std::vector<float>> &queries,
    const RecordedLists &rec, const std::vector<std::vector<SampleRow>> &samples,
    double tol) {
  SampleVerify v;
  std::vector<float> rotated(idx.padded);
  rabitqlib::SplitBatchQuery<float> bq(rotated.data(), idx.padded, idx.ex_bits,
                                       idx.mtype, true);
  std::array<float, 32> est, low, dot;
  std::vector<float> normalized;
  for (size_t q = 0; q < queries.size(); ++q) {
    if (samples[q].empty()) continue;
    ++v.queries;
    // Production pipeline: normalize, then rotate.
    NormalizeQueryFp32(queries[q].data(), idx.dim, &normalized);
    idx.rotator->rotate(normalized.data(), rotated.data());
    bq.reset(rotated.data(), idx.padded, idx.ex_bits, idx.mtype, true);
    const float query_norm_sq =
        std::inner_product(normalized.data(), normalized.data() + idx.dim,
                           normalized.data(), 0.0f);
    const auto &rows = samples[q];
    size_t ri = 0;
    for (const auto &entry : rec.per_query[q]) {
      const uint32_t cid = entry.first;
      if (cid >= idx.clusters.size()) {
        throw Fail{"sampled cluster id " + std::to_string(cid) +
                   " out of range"};
      }
      const auto &cm = idx.clusters[cid];
      if (cm.vector_count != entry.second) {
        throw Fail{"query " + std::to_string(q) + " cluster " +
                   std::to_string(cid) + " replayed " +
                   std::to_string(cm.vector_count) + " != recorded scan " +
                   std::to_string(entry.second)};
      }
      if (cm.vector_count == 0) continue;
      ++v.ids[cid];
      // Same per-cluster prepare as the production streamer loop
      // (select_probe_centroids -> prepare_for_cluster).
      const ClusterPrepare prep =
          PrepareCluster(idx, normalized, query_norm_sq, cid);
      if (idx.mtype == rabitqlib::METRIC_IP) {
        bq.set_g_add(prep.residual_norm, prep.inner_product);
      } else {
        bq.set_g_add(prep.residual_norm);
      }
      const uint32_t total_batches =
          (cm.vector_count + rabitqlib::fastscan::kBatchSize - 1) /
          rabitqlib::fastscan::kBatchSize;
      const char *batch =
          reinterpret_cast<const char *>(idx.batch_base) + cm.batch_data_offset;
      uint32_t remaining = cm.vector_count;
      uint32_t offset = 0;
      while (remaining > 0) {
        const uint32_t bs = static_cast<uint32_t>(
            std::min<size_t>(remaining, rabitqlib::fastscan::kBatchSize));
        const uint32_t batch_id =
            offset / rabitqlib::fastscan::kBatchSize;
        const bool sampled_batch =
            batch_id == 0 || batch_id == total_batches / 2 ||
            batch_id == total_batches - 1;
        if (sampled_batch && ri < rows.size() && rows[ri].list_id == cid &&
            rows[ri].block == batch_id) {
          rabitqlib::split_batch_estdist(batch, bq, idx.padded, est.data(),
                                         low.data(), dot.data(), true);
          if (bs < rabitqlib::fastscan::kBatchSize) ++v.tail_blocks;
          const uint64_t *keys = idx.key_base + cm.key_offset + offset;
          while (ri < rows.size() && rows[ri].list_id == cid &&
                 rows[ri].block == batch_id) {
            const SampleRow &s = rows[ri];
            ++ri;
            if (s.row >= bs) {
              throw Fail{"query " + std::to_string(q) + " cluster " +
                         std::to_string(cid) + " batch " +
                         std::to_string(batch_id) + " sampled row " +
                         std::to_string(s.row) + " beyond tail size " +
                         std::to_string(bs)};
            }
            if (s.key != keys[s.row]) {
              ++v.key_mismatches;
              std::cerr << "sample key mismatch: query " << q << " cluster "
                        << cid << " batch " << batch_id << " row " << s.row
                        << std::endl;
              continue;
            }
            const double d_err =
                std::abs(double(est[s.row]) - double(s.distance));
            const double l_err =
                std::abs(double(low[s.row]) - double(s.lower_bound));
            const double c_err =
                std::abs(double(dot[s.row]) - double(s.coarse_dot));
            v.max_distance_err = std::max(v.max_distance_err, d_err);
            v.max_lower_err = std::max(v.max_lower_err, l_err);
            v.max_dot_err = std::max(v.max_dot_err, c_err);
            if (d_err > tol || l_err > tol || c_err > tol) {
              ++v.value_mismatches;
              std::cerr << "sample value mismatch: query " << q << " cluster "
                        << cid << " batch " << batch_id << " row " << s.row
                        << " distance " << s.distance << " vs " << est[s.row]
                        << " lower_bound " << s.lower_bound << " vs "
                        << low[s.row] << " coarse_dot " << s.coarse_dot
                        << " vs " << dot[s.row] << std::endl;
            }
            ++v.checked;
          }
        }
        batch += idx.batch_stride;
        offset += bs;
        remaining -= bs;
      }
    }
    if (ri != rows.size()) {
      throw Fail{"query " + std::to_string(q) + ": " +
                 std::to_string(rows.size() - ri) +
                 " recorded sample rows were not matched by the replay walk "
                 "(missing or out-of-order lists/batches)"};
    }
  }
  return v;
}

// Integrated replay. The timed scan loop matches IVFEntity::search_packed_impl
// for the mapped path: per-block key-keep mask, then estimate() on the exact
// packed block bytes. scan_count from lists.csv must equal vector_count.
ReplayStats ReplayIntegrated(const IntegratedIndex &idx,
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
  std::array<zvec::turbo::DistanceEstimate, 32> est;
  for (size_t q = 0; q < queries.size(); ++q) {
    const auto t0 = Clock::now();
    if (idx.quantizer->quantize(queries[q].data(), qmeta, &encoded, &emeta) !=
        0) {
      throw Fail{"integrated quantize failed at query " + std::to_string(q)};
    }
    auto distance = idx.quantizer->distance(encoded.data(), emeta);
    auto scanner = idx.quantizer->block_scanner(distance);
    const auto t1 = Clock::now();
    if (!scanner) {
      throw Fail{"integrated block_scanner unavailable at query " +
                 std::to_string(q)};
    }
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
        // Production key-keep mask over this block's keys.
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
            coarse_base + coarse_offset +
            static_cast<size_t>(b) * idx.packed_size;
        if (scanner->estimate(packed, count, est.data()) != 0) {
          throw Fail{"estimate failed at query " + std::to_string(q) +
                     " list " + std::to_string(lid) + " block " +
                     std::to_string(b)};
        }
        stats.vectors += count;
        ++stats.blocks;
        stats.checksum += est[0].distance + est[count - 1].distance;
      }
    }
    const auto t3 = Clock::now();
    if (timed) stats.scan_ns += ElapsedNs(t2, t3);
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

// Dedicated replay. The timed scan loop matches IvfRabitqEntity's per-cluster
// batch loop; "kernel" hoists the per-cluster residual/IP prepare out of the
// timer, "full" keeps it inside, exactly as the production search pays it.
ReplayStats ReplayDedicated(const DedicatedIndex &idx,
                            const std::vector<std::vector<float>> &queries,
                            const RecordedLists &rec, bool order_seq,
                            bool prep_inside, bool timed) {
  ReplayStats stats;
  const auto &seq = order_seq ? rec.sorted : rec.per_query;
  std::vector<float> rotated(idx.padded);
  // Reuse one query object across queries, like IvfRabitqQueryState does.
  rabitqlib::SplitBatchQuery<float> bq(rotated.data(), idx.padded, idx.ex_bits,
                                       idx.mtype, true);
  std::array<float, 32> est, low, dot;
  std::vector<ClusterPrepare> prepared;
  std::vector<float> normalized;
  for (size_t q = 0; q < queries.size(); ++q) {
    const auto t0 = Clock::now();
    // Production pipeline: normalize (CosineNormalizeReformer::transform),
    // then rotate (IvfRabitqReformer::create_query_state).
    NormalizeQueryFp32(queries[q].data(), idx.dim, &normalized);
    idx.rotator->rotate(normalized.data(), rotated.data());
    bq.reset(rotated.data(), idx.padded, idx.ex_bits, idx.mtype, true);
    const float query_norm_sq =
        std::inner_product(normalized.data(), normalized.data() + idx.dim,
                           normalized.data(), 0.0f);
    const auto t1 = Clock::now();
    if (timed) stats.prep_ns += ElapsedNs(t0, t1);

    if (!prep_inside) {
      prepared.clear();
      for (const auto &entry : seq[q]) {
        prepared.push_back(
            PrepareCluster(idx, normalized, query_norm_sq, entry.first));
      }
    }

    const auto t2 = Clock::now();
    size_t li = 0;
    for (const auto &entry : seq[q]) {
      const uint32_t cid = entry.first;
      if (cid >= idx.clusters.size()) {
        throw Fail{"cluster id " + std::to_string(cid) + " out of range"};
      }
      const auto &cm = idx.clusters[cid];
      if (cm.vector_count != entry.second) {
        throw Fail{"query " + std::to_string(q) + " cluster " +
                   std::to_string(cid) + " replayed " +
                   std::to_string(cm.vector_count) + " != recorded scan " +
                   std::to_string(entry.second)};
      }
      ++stats.lists;
      if (cm.vector_count == 0) {
        ++li;
        continue;
      }
      if (prep_inside) {
        const ClusterPrepare prep =
            PrepareCluster(idx, normalized, query_norm_sq, cid);
        if (idx.mtype == rabitqlib::METRIC_IP) {
          bq.set_g_add(prep.residual_norm, prep.inner_product);
        } else {
          bq.set_g_add(prep.residual_norm);
        }
      } else if (idx.mtype == rabitqlib::METRIC_IP) {
        bq.set_g_add(prepared[li].residual_norm, prepared[li].inner_product);
      } else {
        bq.set_g_add(prepared[li].residual_norm);
      }
      const char *batch = reinterpret_cast<const char *>(idx.batch_base) +
                          cm.batch_data_offset;
      uint32_t remaining = cm.vector_count;
      while (remaining > 0) {
        const uint32_t bs = static_cast<uint32_t>(
            std::min<size_t>(remaining, rabitqlib::fastscan::kBatchSize));
        rabitqlib::split_batch_estdist(batch, bq, idx.padded, est.data(),
                                       low.data(), dot.data(), true);
        stats.vectors += bs;
        ++stats.blocks;
        stats.checksum += est[0] + est[bs - 1];
        batch += idx.batch_stride;
        remaining -= bs;
      }
      ++li;
    }
    const auto t3 = Clock::now();
    if (timed) stats.scan_ns += ElapsedNs(t2, t3);
  }
  return stats;
}

struct Row {
  int round;
  const char *side;
  const char *order;
  const char *variant;
  ReplayStats stats;
};

}  // namespace
#endif

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
#if RABITQ_SUPPORTED
  try {
    if (FLAGS_index_int.empty() || FLAGS_index_ded.empty() ||
        FLAGS_queries.empty() || FLAGS_lists_int.empty() ||
        FLAGS_lists_ded.empty() || FLAGS_csv.empty()) {
      std::cerr
          << "Usage: ivf_rabitq_scan_replay --index-int=PATH --index-ded=PATH "
             "--queries=FILE --lists-int=CSV --lists-ded=CSV --csv=PREFIX "
             "[--mode-int=baseline] [--queries-count=N] [--rounds=5] "
             "[--warmup=1] [--samples-int=CSV --samples-ded=CSV] "
             "[--tolerance=1e-4] [--int-variant=prod]"
          << std::endl;
      return 1;
    }
    if (FLAGS_rounds <= 0 || FLAGS_warmup < 0 || FLAGS_queries_count < 0) {
      std::cerr << "Invalid rounds/warmup/queries_count" << std::endl;
      return 1;
    }
    if ((FLAGS_samples_int.empty() != FLAGS_samples_ded.empty()) ||
        FLAGS_tolerance < 0) {
      std::cerr << "--samples-int and --samples-ded must be used together; "
                   "--tolerance must be >= 0"
                << std::endl;
      return 1;
    }

    QuerySet qs;
    if (!LoadQueries(FLAGS_queries, &qs)) {
      std::cerr << "Failed to load queries " << FLAGS_queries << std::endl;
      return 1;
    }
    size_t qcount = qs.rows.size();
    if (FLAGS_queries_count > 0) {
      qcount = std::min<size_t>(qcount, FLAGS_queries_count);
    }

    IntegratedIndex int_idx;
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
    if (iv_flat16) extra_variants.push_back("flat16");
    if (qs.dim != int_idx.dim || qs.dim != ded_idx.dim) {
      std::cerr << "Query dim " << qs.dim
                << " disagrees with index dims (int="
                << int_idx.dim << " ded=" << ded_idx.dim << ")"
                << std::endl;
      return 1;
    }

    // Only the first qcount rows are replayed. The diagnostic CSVs record
    // load-order row indices, so no qid translation is applied; qids only
    // need to be unique so a stray qid-based join cannot silently pass.
    {
      std::unordered_set<long long> seen;
      for (size_t r = 0; r < qcount; ++r) {
        if (!seen.insert(qs.qids[r]).second) {
          std::cerr << "Duplicate qids in the query file" << std::endl;
          return 1;
        }
      }
    }
    RecordedLists rec_int, rec_ded;
    if (!LoadListsCsv(FLAGS_lists_int, FLAGS_mode_int, qcount, qs.rows.size(),
                      &rec_int)) {
      return 1;
    }
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

    const rabitqlib::cpu::Features &feat = rabitqlib::cpu::features();
    std::cout << "queries=" << qcount << " dim=" << qs.dim
              << " mode_int=" << FLAGS_mode_int << std::endl;
    std::cout << "cpu: avx2=" << feat.avx2 << " fma=" << feat.fma
              << " avx512f=" << feat.avx512f
              << " avx512bw=" << feat.avx512bw
              << " avx512dq=" << feat.avx512dq
              << " avx512vpopcntdq=" << feat.avx512vpopcntdq
              << " has_avx512_core="
              << rabitqlib::cpu::has_avx512_core()
              << " has_avx512_popcnt="
              << rabitqlib::cpu::has_avx512_popcnt() << std::endl;
    std::cout << "loadavg: " << LoadAvg() << std::endl;
    std::cout << "int: lists_total=" << rec_int.total_lists
              << " lists/query=" << double(rec_int.total_lists) / qcount
              << " vectors_total=" << rec_int.total_scan
              << " vectors/query=" << double(rec_int.total_scan) / qcount
              << " scan_bytes=" << int_idx.scan_bytes << std::endl;
    std::cout << "ded: lists_total=" << rec_ded.total_lists
              << " lists/query=" << double(rec_ded.total_lists) / qcount
              << " vectors_total=" << rec_ded.total_scan
              << " vectors/query=" << double(rec_ded.total_scan) / qcount
              << " batch_bytes=" << ded_idx.batch_bytes << std::endl;
    std::cout << "scope: coarse scan kernels only (query prepare + estimate + "
                 "key-keep walk); no heap/refine/finalize"
              << std::endl;
    std::cout << "warmup=" << FLAGS_warmup
              << " rounds=" << FLAGS_rounds
              << " (mmap pages are faulted in by warmup; timed rounds are "
                 "cache-warm)"
              << std::endl;

    std::vector<std::vector<float>> queries(qcount);
    for (size_t q = 0; q < qcount; ++q) queries[q] = qs.rows[q];

    // Untimed equivalence pass: recompute every exported production sample
    // (key + distance + lower_bound + coarse_dot) from the same segments.
    if (!FLAGS_samples_int.empty()) {
      std::vector<std::vector<SampleRow>> samples_int, samples_ded;
      size_t skipped_int = 0, skipped_ded = 0;
      if (!LoadSamplesCsv(FLAGS_samples_int, FLAGS_mode_int, qcount,
                          qs.rows.size(), &samples_int, &skipped_int) ||
          !LoadSamplesCsv(FLAGS_samples_ded, "dedicated", qcount,
                          qs.rows.size(), &samples_ded, &skipped_ded)) {
        std::cerr << "Failed to load samples.csv files" << std::endl;
        return 1;
      }
      if (skipped_int || skipped_ded) {
        std::cout << "samples: skipped trimmed-tail rows int=" << skipped_int
                  << " ded=" << skipped_ded << std::endl;
      }
      std::cout << "sample verification: recomputing recorded estimate rows "
                   "from the index segments (tolerance="
                << FLAGS_tolerance << ")" << std::endl;
      const SampleVerify v_int = VerifyIntegratedSamples(
          int_idx, queries, rec_int, samples_int, FLAGS_tolerance);
      PrintSampleVerify("int", "model_ids", v_int, FLAGS_tolerance);
      if (v_int.checked == 0) {
        std::cerr << "no integrated samples verified for mode "
                  << FLAGS_mode_int << " (mode/column mismatch?)"
                  << std::endl;
        return 1;
      }
      const SampleVerify v_ded = VerifyDedicatedSamples(
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
      }
      std::cout << "sample verification: PASSED (keys and estimate values "
                   "recomputed from the segments match the recorded "
                   "production scan)"
                << std::endl;
    }

    // Warmup both sides, both orders, both variants, and the factor runs.
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
      }
      ReplayDedicated(ded_idx, queries, rec_ded, false, false, false);
      ReplayDedicated(ded_idx, queries, rec_ded, true, false, false);
      ReplayDedicated(ded_idx, queries, rec_ded, false, true, false);
      ReplayDedicated(ded_idx, queries, rec_ded, true, true, false);
    }

    std::vector<Row> rows;
    std::unordered_map<std::string, double> first_checksum;
    bool ok = true;
    for (int r = 1; r <= FLAGS_rounds; ++r) {
      // Alternate side order per round to interleave the implementations.
      const bool int_first = (r % 2 == 1);
      for (int s = 0; s < 2; ++s) {
        const bool is_int = (s == 0) == int_first;
        for (int o = 0; o < 2; ++o) {
          const bool seq_order = (o == 1);
          if (is_int) {
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
          } else {
            for (int v = 0; v < 2; ++v) {
              ReplayStats st = ReplayDedicated(
                  ded_idx, queries, rec_ded, seq_order, v == 1, true);
              rows.push_back({r, "ded", seq_order ? "seq" : "orig",
                              v == 0 ? "kernel" : "full", std::move(st)});
            }
          }
        }
      }
    }

    // Determinism: identical checksums across rounds per (side, order,
    // variant); the replay must do identical work every round.
    std::unordered_map<std::string, size_t> vectors_seen;
    for (const auto &row : rows) {
      const std::string key = std::string(row.side) + '/' + row.order + '/' +
                              row.variant;
      auto it = first_checksum.find(key);
      if (it == first_checksum.end()) {
        first_checksum.emplace(key, row.stats.checksum);
        vectors_seen.emplace(key, row.stats.vectors);
      } else if (it->second != row.stats.checksum ||
                 vectors_seen[key] != row.stats.vectors) {
        std::cerr << "nondeterministic replay for " << key << std::endl;
        ok = false;
      }
    }

    std::ofstream out(FLAGS_csv + ".replay.csv");
    if (!out) {
      std::cerr << "Failed to open " << FLAGS_csv << ".replay.csv"
                << std::endl;
      return 1;
    }
    out << "round,side,order,variant,queries,lists,vectors,blocks,"
           "prep_us_per_query,scan_us_per_query,ns_per_vector,ns_per_block,"
           "checksum\n";
    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (const auto &row : rows) {
      const double nq = static_cast<double>(qcount);
      out << row.round << ',' << row.side << ',' << row.order << ','
          << row.variant << ',' << qcount << ',' << row.stats.lists << ','
          << row.stats.vectors << ',' << row.stats.blocks << ','
          << row.stats.prep_ns / nq / 1000.0 << ','
          << row.stats.scan_ns / nq / 1000.0 << ','
          << (row.stats.vectors
                  ? row.stats.scan_ns / row.stats.vectors
                  : std::numeric_limits<double>::quiet_NaN())
          << ','
          << (row.stats.blocks
                  ? row.stats.scan_ns / row.stats.blocks
                  : std::numeric_limits<double>::quiet_NaN())
          << ',' << row.stats.checksum << '\n';
    }
    if (!out) {
      std::cerr << "Writing output failed" << std::endl;
      return 1;
    }

    // Per (side, order, variant) medians over rounds.
    auto median_scan = [&](const char *side, const char *order,
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
        for (const auto &variant : variant_names) {
          std::vector<double> scan_us, prep_us, ns_vec;
          for (const auto &row : rows) {
            if (std::string(row.side) != side ||
                std::string(row.order) != order ||
                std::string(row.variant) != variant) {
              continue;
            }
            scan_us.push_back(row.stats.scan_ns / qcount / 1000.0);
            prep_us.push_back(row.stats.prep_ns / qcount / 1000.0);
            ns_vec.push_back(row.stats.vectors
                                 ? row.stats.scan_ns / row.stats.vectors
                                 : 0.0);
          }
          if (scan_us.empty()) continue;
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
    }
    std::cout << (ok ? "verify: all replayed work matched lists.csv and "
                       "checksums were deterministic"
                     : "verify: FAILED")
              << std::endl;
    return ok ? 0 : 1;
  } catch (const Fail &e) {
    std::cerr << "replay failed: " << e.msg << std::endl;
    return 1;
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
#else
  std::cerr << "RABITQ_SUPPORTED is off; rebuild with RaBitQ enabled"
            << std::endl;
  return 1;
#endif
}
