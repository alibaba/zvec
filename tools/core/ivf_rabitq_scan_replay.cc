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
// both lists.csv files to cover the identical query set.
//
// Usage:
//   ivf_rabitq_scan_replay --index-int=PATH --index-ded=PATH \
//       --queries=FILE --lists-int=CSV --lists-ded=CSV \
//       [--mode-int=baseline] [--queries-count=N] [--rounds=5] \
//       [--warmup=1] --csv=PREFIX
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
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <gflags/gflags.h>
#include <turbo/quantizer/quantizer.h>
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
DEFINE_string(csv, "", "output csv prefix (required)");

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
  // File qid per row; lists.csv records these, not row indices.
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
                  const std::unordered_map<long long, size_t> &qid_to_row,
                  RecordedLists *out) {
  std::ifstream in(path);
  if (!in) return false;
  std::string line;
  if (!std::getline(in, line)) return false;
  if (line != "query,mode,list_id,scan_count,limited") return false;
  out->per_query.assign(qid_to_row.size(), {});
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
      std::cerr << "lists.csv " << path << ":" << line_no
                << " malformed line" << std::endl;
      return false;
    }
    if (f[1] != mode) continue;
    const long long q = std::stoll(f[0]);
    const unsigned long long lid = std::stoull(f[2]);
    const unsigned long long scan = std::stoull(f[3]);
    const int limited = std::stoi(f[4]);
    const auto row_it = qid_to_row.find(q);
    if (row_it == qid_to_row.end()) {
      // Belongs to a trimmed tail (--queries-count); not replayed.
      continue;
    }
    const size_t row = row_it->second;
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
  size_t batch_stride = 0;
  size_t batch_bytes = 0;
  uint32_t dim = 0;
  uint32_t padded = 0;
  uint8_t ex_bits = 0;
  rabitqlib::MetricType mtype = rabitqlib::METRIC_L2;
  std::vector<float> rotated_centroids;
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

// Integrated replay. The timed scan loop matches IVFEntity::search_packed_impl
// for the mapped path: per-block key-keep mask, then estimate() on the exact
// packed block bytes. scan_count from lists.csv must equal vector_count.
ReplayStats ReplayIntegrated(const IntegratedIndex &idx,
                             const std::vector<std::vector<float>> &queries,
                             const RecordedLists &rec, bool order_seq,
                             bool timed) {
  ReplayStats stats;
  const IndexQueryMeta qmeta(idx.dtype, idx.dim);
  const auto &seq = order_seq ? rec.sorted : rec.per_query;
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
            idx.coarse_base + coarse_offset +
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
  std::vector<std::pair<float, float>> prepared;
  for (size_t q = 0; q < queries.size(); ++q) {
    const auto t0 = Clock::now();
    idx.rotator->rotate(queries[q].data(), rotated.data());
    bq.reset(rotated.data(), idx.padded, idx.ex_bits, idx.mtype, true);
    const auto t1 = Clock::now();
    if (timed) stats.prep_ns += ElapsedNs(t0, t1);

    if (!prep_inside) {
      prepared.clear();
      for (const auto &entry : seq[q]) {
        const uint32_t cid = entry.first;
        const float *centroid =
            idx.rotated_centroids.data() +
            static_cast<size_t>(cid) * idx.padded;
        const float residual_norm = std::sqrt(std::max(
            rabitqlib::euclidean_sqr(rotated.data(), centroid, idx.padded),
            0.0f));
        const float ip =
            idx.mtype == rabitqlib::METRIC_IP
                ? rabitqlib::dot_product(rotated.data(), centroid, idx.padded)
                : 0.0f;
        prepared.emplace_back(residual_norm, ip);
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
        const float *centroid =
            idx.rotated_centroids.data() +
            static_cast<size_t>(cid) * idx.padded;
        const float residual_norm = std::sqrt(std::max(
            rabitqlib::euclidean_sqr(rotated.data(), centroid, idx.padded),
            0.0f));
        if (idx.mtype == rabitqlib::METRIC_IP) {
          bq.set_g_add(residual_norm,
                       rabitqlib::dot_product(rotated.data(), centroid,
                                              idx.padded));
        } else {
          bq.set_g_add(residual_norm);
        }
      } else {
        bq.set_g_add(prepared[li].first, prepared[li].second);
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
             "[--warmup=1]"
          << std::endl;
      return 1;
    }
    if (FLAGS_rounds <= 0 || FLAGS_warmup < 0 || FLAGS_queries_count < 0) {
      std::cerr << "Invalid rounds/warmup/queries_count" << std::endl;
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
    if (qs.dim != int_idx.dim || qs.dim != ded_idx.dim) {
      std::cerr << "Query dim " << qs.dim
                << " disagrees with index dims (int="
                << int_idx.dim << " ded=" << ded_idx.dim << ")"
                << std::endl;
      return 1;
    }

    // Only the first qcount rows are replayed; map their file qids to rows.
    std::unordered_map<long long, size_t> qid_to_row;
    for (size_t r = 0; r < qcount; ++r) qid_to_row.emplace(qs.qids[r], r);
    if (qid_to_row.size() != qcount) {
      std::cerr << "Duplicate qids in the query file" << std::endl;
      return 1;
    }
    RecordedLists rec_int, rec_ded;
    if (!LoadListsCsv(FLAGS_lists_int, FLAGS_mode_int, qid_to_row, &rec_int)) {
      return 1;
    }
    if (!LoadListsCsv(FLAGS_lists_ded, "dedicated", qid_to_row, &rec_ded)) {
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
    std::cout << "int: lists/query=" << double(rec_int.total_lists) / qcount
              << " vectors/query=" << double(rec_int.total_scan) / qcount
              << " scan_bytes=" << int_idx.scan_bytes << std::endl;
    std::cout << "ded: lists/query=" << double(rec_ded.total_lists) / qcount
              << " vectors/query=" << double(rec_ded.total_scan) / qcount
              << " batch_bytes=" << ded_idx.batch_bytes << std::endl;
    std::cout << "warmup=" << FLAGS_warmup
              << " rounds=" << FLAGS_rounds
              << " (mmap pages are faulted in by warmup; timed rounds are "
                 "cache-warm)"
              << std::endl;

    std::vector<std::vector<float>> queries(qcount);
    for (size_t q = 0; q < qcount; ++q) queries[q] = qs.rows[q];

    // Warmup both sides, both orders, both variants.
    for (int w = 0; w < FLAGS_warmup; ++w) {
      ReplayIntegrated(int_idx, queries, rec_int, false, false);
      ReplayIntegrated(int_idx, queries, rec_int, true, false);
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
                  int_idx, queries, rec_int, seq_order, true);
              rows.push_back({r, "int", seq_order ? "seq" : "orig",
                              v == 0 ? "kernel" : "full", std::move(st)});
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
    std::cout << "summary (median over " << FLAGS_rounds << " rounds):"
              << std::endl;
    for (const char *side : {"int", "ded"}) {
      for (const char *order : {"orig", "seq"}) {
        for (const char *variant : {"kernel", "full"}) {
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
