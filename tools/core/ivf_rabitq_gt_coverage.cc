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

// Offline GT candidate-coverage measurement for IVF(+RaBitQ) indexes.
// Replicates the exact FP32 centroid selection used at query time and
// computes, per query, how many of the GT@50 vectors live in the selected
// lists. Reads raw index segments only; never runs the search stack.
//
// Usage:
//   ivf_rabitq_gt_coverage --index=PATH --queries=FILE --gt=FILE \
//       --nprobe=N [--gt_top=50] [--csv=PREFIX]
//
// Outputs: PREFIX.coverage.csv (query,covered,gt_total,selected_vectors),
// PREFIX.selected.csv (query;list ids), PREFIX.gtlists.csv (query,gt_id,list).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <gflags/gflags.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_helper.h>
#include <zvec/core/framework/index_segment_storage.h>
#include "algorithm/ivf/ivf_index_format.h"
#include "algorithm/ivf_rabitq/ivf_rabitq_index_format.h"
#include "algorithm/ivf_rabitq/ivf_rabitq_params.h"

using namespace zvec::core;

DEFINE_string(index, "", "input index file path");
DEFINE_string(queries, "", "query text file (qid;f1 f2 ...)");
DEFINE_string(gt, "", "ground truth file (qid;id1 id2 ...)");
DEFINE_int32(nprobe, 0, "number of lists to select");
DEFINE_int32(gt_top, 50, "use the first N gt ids per query");
DEFINE_string(csv, "", "output csv prefix");

namespace {

// Mirrors RabitqConverterHeader (hnsw_rabitq/rabitq_utils.h) without pulling
// in rabitqlib headers.
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
const std::string kRabitqConverterSegId{"rabitq.converter"};

struct TextTable {
  // query id -> values
  std::unordered_map<uint64_t, std::vector<float>> queries;
  std::unordered_map<uint64_t, std::vector<uint64_t>> gt;
  size_t dimension = 0;
};

bool LoadTextTable(const std::string &path, bool as_float, TextTable *out) {
  std::ifstream in(path);
  if (!in) return false;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    const size_t sep = line.find(';');
    if (sep == std::string::npos) continue;
    const uint64_t qid = std::stoull(line.substr(0, sep));
    std::istringstream iss(line.substr(sep + 1));
    if (as_float) {
      std::vector<float> vals;
      float v;
      while (iss >> v) vals.push_back(v);
      if (out->dimension == 0) out->dimension = vals.size();
      if (!vals.empty()) out->queries.emplace(qid, std::move(vals));
    } else {
      std::vector<uint64_t> ids;
      uint64_t id;
      while (iss >> id) ids.push_back(id);
      if (!ids.empty()) out->gt.emplace(qid, std::move(ids));
    }
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

struct Centroids {
  size_t count = 0;
  size_t dim = 0;       // floats used by the coarse dot
  std::vector<float> rows;  // count * dim floats
};

}  // namespace

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_index.empty() || FLAGS_queries.empty() || FLAGS_gt.empty() ||
      FLAGS_nprobe <= 0) {
    std::cerr << "--index/--queries/--gt/--nprobe are required" << std::endl;
    return 1;
  }

  auto storage = IndexFactory::CreateStorage("MMapFileReadStorage");
  if (!storage || storage->open(FLAGS_index, false) != 0) {
    std::cerr << "Failed to open index " << FLAGS_index << std::endl;
    return 1;
  }

  // key -> list/cluster membership
  std::unordered_map<uint64_t, uint32_t> key_to_list;
  std::vector<uint32_t> list_sizes;
  Centroids cents;
  const bool integrated = static_cast<bool>(storage->get(IVF_INVERTED_HEADER_SEG_ID));

  if (integrated) {
    InvertedIndexHeader header;
    {
      const void *data = ReadWhole(storage->get(IVF_INVERTED_HEADER_SEG_ID),
                                   sizeof(header), "ivf.inverted_header");
      if (!data) return 1;
      std::memcpy(&header, data, sizeof(header));
    }
    const size_t list_count = header.inverted_list_count;
    std::vector<InvertedListMeta> lists(list_count);
    {
      const size_t bytes = list_count * sizeof(InvertedListMeta);
      const void *data =
          ReadWhole(storage->get(IVF_INVERTED_META_SEG_ID), bytes,
                    "ivf.inverted_meta");
      if (!data) return 1;
      std::memcpy(lists.data(), data, bytes);
    }
    // global key array, ordered list by list
    std::vector<uint64_t> keys(header.total_vector_count);
    {
      const size_t bytes = keys.size() * sizeof(uint64_t);
      const void *data =
          ReadWhole(storage->get(IVF_KEYS_SEG_ID), bytes, "hc.keys");
      if (!data) return 1;
      std::memcpy(keys.data(), data, bytes);
    }
    list_sizes.resize(list_count);
    for (size_t l = 0; l < list_count; ++l) {
      list_sizes[l] = lists[l].vector_count;
      for (size_t i = 0; i < lists[l].vector_count; ++i) {
        key_to_list[keys[lists[l].id_offset + i]] = static_cast<uint32_t>(l);
      }
    }
    // centroids: nested flat index; Cosine rows are [normalized | norm]
    auto centroid_seg = storage->get(IVF_CENTROID_SEG_ID, 0);
    if (!centroid_seg) {
      std::cerr << "Missing ivf.centroid" << std::endl;
      return 1;
    }
    IndexStorage::Pointer container =
        std::make_shared<IndexSegmentStorage>(centroid_seg);
    if (container->open(std::string(), false) != 0) {
      std::cerr << "Failed to open centroid container" << std::endl;
      return 1;
    }
    IndexMeta meta;
    if (IndexHelper::DeserializeFromStorage(container.get(), &meta) != 0) {
      std::cerr << "Failed to read centroid meta" << std::endl;
      return 1;
    }
    auto keys_seg = container->get("flat.keys");
    auto features_seg = container->get("flat.features");
    if (!keys_seg || !features_seg) {
      std::cerr << "Missing centroid flat.keys/flat.features" << std::endl;
      return 1;
    }
    cents.count = keys_seg->data_size() / sizeof(uint64_t);
    const size_t element_size = meta.element_size();
    const size_t dimension = meta.dimension();
    if (meta.data_type() != IndexMeta::DataType::DT_FP32 ||
        element_size != dimension * sizeof(float)) {
      std::cerr << "Unsupported centroid layout" << std::endl;
      return 1;
    }
    // Cosine converter appends the original norm as the last float; the
    // coarse dot runs over the effective (dimension - 1) components.
    cents.dim = dimension - 1;
    cents.rows.resize(cents.count * cents.dim);
    const void *features = nullptr;
    if (features_seg->read(0, &features, cents.count * element_size) !=
        cents.count * element_size) {
      std::cerr << "Failed to read centroid features" << std::endl;
      return 1;
    }
    for (size_t i = 0; i < cents.count; ++i) {
      std::memcpy(cents.rows.data() + i * cents.dim,
                  static_cast<const char *>(features) + i * element_size,
                  cents.dim * sizeof(float));
    }
  } else {
    IvfRabitqHeader header;
    {
      const void *data =
          ReadWhole(storage->get(IVF_RABITQ_HEADER_SEG_ID), sizeof(header),
                    "ivf_rabitq.header");
      if (!data) return 1;
      std::memcpy(&header, data, sizeof(header));
      if (header.magic != kIvfRabitqIndexMagic) {
        std::cerr << "Bad ivf_rabitq header magic" << std::endl;
        return 1;
      }
    }
    const size_t cluster_count = header.cluster_count;
    std::vector<IvfRabitqClusterMeta> clusters(cluster_count);
    {
      const size_t bytes = cluster_count * sizeof(IvfRabitqClusterMeta);
      const void *data =
          ReadWhole(storage->get(IVF_RABITQ_CLUSTER_META_SEG_ID), bytes,
                    "ivf_rabitq.cluster_meta");
      if (!data) return 1;
      std::memcpy(clusters.data(), data, bytes);
    }
    std::vector<uint64_t> keys(header.total_vector_count);
    {
      const size_t bytes = keys.size() * sizeof(uint64_t);
      const void *data =
          ReadWhole(storage->get(IVF_RABITQ_KEYS_SEG_ID), bytes,
                    "ivf_rabitq.keys");
      if (!data) return 1;
      std::memcpy(keys.data(), data, bytes);
    }
    list_sizes.resize(cluster_count);
    for (size_t c = 0; c < cluster_count; ++c) {
      list_sizes[c] = clusters[c].vector_count;
      for (size_t i = 0; i < clusters[c].vector_count; ++i) {
        key_to_list[keys[clusters[c].key_offset + i]] = static_cast<uint32_t>(c);
      }
    }
    // original centroids: second block of rabitq.converter, after the
    // rotated centroids (num_clusters * padded_dim floats)
    auto conv_seg = storage->get(kRabitqConverterSegId);
    if (!conv_seg) {
      std::cerr << "Missing rabitq.converter" << std::endl;
      return 1;
    }
    ConverterHeader conv;
    {
      const void *data = ReadWhole(conv_seg, sizeof(conv), "rabitq.converter");
      if (!data) return 1;
      std::memcpy(&conv, data, sizeof(conv));
    }
    cents.count = conv.num_clusters;
    cents.dim = conv.dim;
    const size_t rotated_bytes =
        static_cast<size_t>(conv.num_clusters) * conv.padded_dim * sizeof(float);
    const size_t orig_bytes = cents.count * cents.dim * sizeof(float);
    if (conv_seg->data_size() < sizeof(conv) + rotated_bytes + orig_bytes) {
      std::cerr << "rabitq.converter too small" << std::endl;
      return 1;
    }
    cents.rows.resize(cents.count * cents.dim);
    {
      const void *data = nullptr;
      if (conv_seg->read(sizeof(conv) + rotated_bytes, &data, orig_bytes) !=
          orig_bytes) {
        std::cerr << "Failed to read original centroids" << std::endl;
        return 1;
      }
      std::memcpy(cents.rows.data(), data, orig_bytes);
    }
  }

  TextTable table;
  if (!LoadTextTable(FLAGS_queries, true, &table) ||
      !LoadTextTable(FLAGS_gt, false, &table) || table.queries.empty()) {
    std::cerr << "Failed to load queries/gt" << std::endl;
    return 1;
  }
  if (table.dimension != cents.dim) {
    std::cerr << "Query dim " << table.dimension << " != centroid dim "
              << cents.dim << std::endl;
    return 1;
  }

  const size_t nprobe = static_cast<size_t>(FLAGS_nprobe);
  const size_t gt_top = static_cast<size_t>(FLAGS_gt_top);
  const std::string prefix =
      FLAGS_csv.empty() ? FLAGS_index + ".coverage" : FLAGS_csv;
  std::ofstream cov(prefix + ".coverage.csv");
  std::ofstream sel(prefix + ".selected.csv");
  std::ofstream gtl(prefix + ".gtlists.csv");
  if (!cov || !sel || !gtl) {
    std::cerr << "Failed to open output files" << std::endl;
    return 1;
  }
  cov << "query,covered,gt_total,selected_vectors\n";
  gtl << "query,gt_id,list\n";

  std::vector<uint64_t> qids;
  qids.reserve(table.queries.size());
  for (const auto &it : table.queries) qids.push_back(it.first);
  std::sort(qids.begin(), qids.end());

  std::vector<std::pair<double, uint32_t>> ranked(cents.count);
  std::vector<size_t> covered_hist(gt_top + 1, 0);
  double covered_sum = 0.0;
  double selected_sum = 0.0;
  size_t gt_missing = 0;

  for (const uint64_t qid : qids) {
    const std::vector<float> &q = table.queries.at(qid);
    // Cosine query transform: normalize components
    double norm = 0.0;
    for (float v : q) norm += static_cast<double>(v) * v;
    norm = std::sqrt(norm);
    // Integrated selection scores Cosine distance (1 - dot); dedicated
    // scores the negated IP. Both rank by ascending score with id tie-break,
    // so a single ascending sort suffices for either convention.
    for (size_t c = 0; c < cents.count; ++c) {
      double dot = 0.0;
      const float *row = cents.rows.data() + c * cents.dim;
      for (size_t j = 0; j < cents.dim; ++j) {
        dot += static_cast<double>(q[j]) * row[j];
      }
      const double score = integrated ? (1.0 - dot / norm) : (-dot / norm);
      ranked[c] = {score, static_cast<uint32_t>(c)};
    }
    std::stable_sort(ranked.begin(), ranked.end(),
                     [](const auto &a, const auto &b) {
                       if (a.first != b.first) return a.first < b.first;
                       return a.second < b.second;
                     });
    std::unordered_set<uint32_t> selected;
    size_t selected_vectors = 0;
    sel << qid << ';';
    for (size_t i = 0; i < nprobe && i < ranked.size(); ++i) {
      const uint32_t c = ranked[i].second;
      selected.insert(c);
      selected_vectors += list_sizes[c];
      sel << c << (i + 1 < nprobe && i + 1 < ranked.size() ? ' ' : '\n');
    }
    selected_sum += static_cast<double>(selected_vectors);

    const auto git = table.gt.find(qid);
    size_t covered = 0;
    if (git != table.gt.end()) {
      const size_t n = std::min(gt_top, git->second.size());
      for (size_t i = 0; i < n; ++i) {
        const uint64_t gt_id = git->second[i];
        const auto lit = key_to_list.find(gt_id);
        if (lit == key_to_list.end()) {
          ++gt_missing;
          continue;
        }
        gtl << qid << ',' << gt_id << ',' << lit->second << '\n';
        if (selected.count(lit->second)) ++covered;
      }
      covered_hist[covered]++;
      covered_sum += static_cast<double>(covered);
      cov << qid << ',' << covered << ',' << n << ',' << selected_vectors
          << '\n';
    }
  }

  const size_t nq = qids.size();
  std::cout << "index=" << FLAGS_index << " nprobe=" << nprobe
            << " format=" << (integrated ? "integrated" : "dedicated")
            << " queries=" << nq << std::endl;
  std::cout << "gt_coverage_mean=" << covered_sum / (nq * gt_top)
            << " selected_vectors_mean=" << selected_sum / nq
            << " gt_ids_missing=" << gt_missing << std::endl;
  std::cout << "covered_histogram:";
  for (size_t i = 0; i <= gt_top; ++i) {
    if (covered_hist[i]) std::cout << ' ' << i << ':' << covered_hist[i];
  }
  std::cout << std::endl;
  return 0;
}
