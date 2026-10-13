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

// Dump IVF partition statistics straight from the index segments: per-list
// vector counts (ivf.inverted_meta) and per-centroid feature rows (the flat
// index nested in ivf.centroid). Compares old/new Cosine centroid
// normalization without loading the full search stack.
//
// Usage:
//   ivf_rabitq_partition_stat --index=PATH [--csv=PREFIX]

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <gflags/gflags.h>
#include <zvec/ailego/utility/float_helper.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_helper.h>
#include <zvec/core/framework/index_segment_storage.h>
#include "algorithm/ivf/ivf_index_format.h"

using namespace zvec::core;

DEFINE_string(index, "", "input index file path");
DEFINE_string(csv, "", "output csv prefix (default: index path)");

namespace {

void PrintDistribution(const char *name, const std::vector<double> &values) {
  // Non-finite entries (NaN centroids from empty lists) must not poison the
  // aggregate stats; they are counted and reported separately.
  std::vector<double> sorted;
  size_t nonfinite = 0;
  for (double value : values) {
    if (std::isfinite(value)) {
      sorted.push_back(value);
    } else {
      ++nonfinite;
    }
  }
  if (nonfinite > 0) {
    std::cout << ' ' << name << "_nonfinite=" << nonfinite;
  }
  if (sorted.empty()) {
    std::cout << ' ' << name << "_count=0";
    return;
  }
  std::sort(sorted.begin(), sorted.end());
  double sum = 0.0;
  for (double value : sorted) sum += value;
  const double mean = sum / sorted.size();
  double variance = 0.0;
  for (double value : sorted) variance += (value - mean) * (value - mean);
  const auto percentile = [&](double fraction) {
    return sorted[static_cast<size_t>(std::ceil(fraction * sorted.size())) - 1];
  };
  std::cout << std::fixed << std::setprecision(6) << ' ' << name
            << "_mean=" << mean << ' ' << name
            << "_std=" << std::sqrt(variance / sorted.size()) << ' ' << name
            << "_min=" << sorted.front() << ' ' << name
            << "_p50=" << percentile(0.50) << ' ' << name
            << "_p95=" << percentile(0.95) << ' ' << name
            << "_max=" << sorted.back();
}

}  // namespace

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_index.empty()) {
    std::cerr << "--index is required" << std::endl;
    return 1;
  }

  auto storage = IndexFactory::CreateStorage("MMapFileReadStorage");
  if (!storage) {
    std::cerr << "Failed to create MMapFileReadStorage" << std::endl;
    return 1;
  }
  int ret = storage->open(FLAGS_index, false);
  if (ret != 0) {
    std::cerr << "Failed to open index " << FLAGS_index << ": " << ret
              << std::endl;
    return 1;
  }

  // Inverted header: list count and total vectors
  auto header_seg = storage->get(IVF_INVERTED_HEADER_SEG_ID);
  if (!header_seg) {
    std::cerr << "Missing segment " << IVF_INVERTED_HEADER_SEG_ID
              << std::endl;
    return 1;
  }
  InvertedIndexHeader header;
  {
    const void *data = nullptr;
    if (header_seg->read(0, &data, sizeof(header)) != sizeof(header)) {
      std::cerr << "Failed to read inverted header" << std::endl;
      return 1;
    }
    std::memcpy(&header, data, sizeof(header));
  }

  // Inverted meta: one InvertedListMeta per list
  const size_t list_count = header.inverted_list_count;
  std::vector<InvertedListMeta> lists(list_count);
  if (list_count != 0u) {
    auto meta_seg = storage->get(IVF_INVERTED_META_SEG_ID);
    if (!meta_seg) {
      std::cerr << "Missing segment " << IVF_INVERTED_META_SEG_ID
                << std::endl;
      return 1;
    }
    const size_t list_bytes = list_count * sizeof(InvertedListMeta);
    if (meta_seg->data_size() < list_bytes) {
      std::cerr << "Inverted meta too small: " << meta_seg->data_size()
                << " < " << list_bytes << std::endl;
      return 1;
    }
    const void *data = nullptr;
    if (meta_seg->read(0, &data, list_bytes) != list_bytes) {
      std::cerr << "Failed to read inverted meta" << std::endl;
      return 1;
    }
    std::memcpy(lists.data(), data, list_bytes);
  }

  // Centroid features: flat index nested in ivf.centroid
  auto centroid_seg = storage->get(IVF_CENTROID_SEG_ID, 0);
  if (!centroid_seg) {
    std::cerr << "Missing segment " << IVF_CENTROID_SEG_ID << std::endl;
    return 1;
  }
  IndexStorage::Pointer centroid_container =
      std::make_shared<IndexSegmentStorage>(centroid_seg);
  ret = centroid_container->open(std::string(), false);
  if (ret != 0) {
    std::cerr << "Failed to open centroid container: " << ret << std::endl;
    return 1;
  }
  IndexMeta centroid_meta;
  ret = IndexHelper::DeserializeFromStorage(centroid_container.get(),
                                            &centroid_meta);
  if (ret != 0) {
    std::cerr << "Failed to read centroid meta: " << ret << std::endl;
    return 1;
  }

  auto keys_seg = centroid_container->get("flat.keys");
  auto features_seg = centroid_container->get("flat.features");
  if (!keys_seg || !features_seg) {
    std::cerr << "Missing flat.keys/flat.features in centroid index"
              << std::endl;
    return 1;
  }
  const size_t centroid_count = keys_seg->data_size() / sizeof(uint64_t);
  const size_t element_size = centroid_meta.element_size();
  if (centroid_count * element_size > features_seg->data_size()) {
    std::cerr << "Features segment too small: " << features_seg->data_size()
              << " < " << centroid_count * element_size << std::endl;
    return 1;
  }
  const void *features_data = nullptr;
  if (features_seg->read(0, &features_data,
                         centroid_count * element_size) !=
      centroid_count * element_size) {
    std::cerr << "Failed to read centroid features" << std::endl;
    return 1;
  }

  // Cosine converter layout: [normalized components | original norm]
  const size_t dimension = centroid_meta.dimension();
  const IndexMeta::DataType data_type = centroid_meta.data_type();
  size_t effective_dim = dimension;
  size_t unit_size = 0u;
  switch (data_type) {
    case IndexMeta::DataType::DT_FP32:
      unit_size = sizeof(float);
      effective_dim = dimension - sizeof(float) / sizeof(float);
      break;
    case IndexMeta::DataType::DT_FP16:
      unit_size = sizeof(zvec::ailego::Float16);
      effective_dim = dimension - sizeof(float) / sizeof(zvec::ailego::Float16);
      break;
    default:
      std::cerr << "Unsupported centroid data type "
                << static_cast<int>(data_type) << std::endl;
      return 1;
  }
  if (element_size != dimension * unit_size || effective_dim == 0u) {
    std::cerr << "Unexpected element layout: dimension=" << dimension
              << " unit=" << unit_size << " element_size=" << element_size
              << std::endl;
    return 1;
  }

  std::vector<double> norms;
  std::vector<double> trailing;
  std::vector<double> nonfinite;
  norms.reserve(centroid_count);
  trailing.reserve(centroid_count);
  nonfinite.reserve(centroid_count);
  for (size_t i = 0; i < centroid_count; ++i) {
    const char *row =
        static_cast<const char *>(features_data) + i * element_size;
    double accum = 0.0;
    size_t bad = 0u;
    if (data_type == IndexMeta::DataType::DT_FP32) {
      const auto *vec = reinterpret_cast<const float *>(row);
      for (size_t j = 0; j < effective_dim; ++j) {
        if (!std::isfinite(vec[j])) ++bad;
        accum += static_cast<double>(vec[j]) * vec[j];
      }
      float tail = 0.0f;
      std::memcpy(&tail, row + effective_dim * sizeof(float), sizeof(tail));
      trailing.push_back(tail);
    } else {
      const auto *vec =
          reinterpret_cast<const zvec::ailego::Float16 *>(row);
      for (size_t j = 0; j < effective_dim; ++j) {
        const float value = vec[j];
        if (!std::isfinite(value)) ++bad;
        accum += static_cast<double>(value) * value;
      }
      float tail = 0.0f;
      std::memcpy(&tail, row + effective_dim * sizeof(zvec::ailego::Float16),
                  sizeof(tail));
      trailing.push_back(tail);
    }
    norms.push_back(std::sqrt(accum));
    nonfinite.push_back(static_cast<double>(bad));
  }

  std::vector<double> sizes;
  sizes.reserve(list_count);
  size_t total_vectors = 0u;
  size_t empty_lists = 0u;
  for (const auto &it : lists) {
    sizes.push_back(static_cast<double>(it.vector_count));
    total_vectors += it.vector_count;
    if (it.vector_count == 0u) ++empty_lists;
  }

  std::cout << "index=" << FLAGS_index << std::endl;
  std::cout << "centroid_meta: dimension=" << dimension
            << " data_type=" << static_cast<int>(data_type)
            << " metric=" << centroid_meta.metric_name()
            << " element_size=" << element_size << std::endl;
  std::cout << "lists:" << std::endl;
  std::cout << " inverted_list_count=" << list_count
            << " centroid_count=" << centroid_count
            << " total_vectors=" << total_vectors
            << " header_total_vectors=" << header.total_vector_count
            << " empty_lists=" << empty_lists << std::endl;
  std::cout << "list_sizes:";
  PrintDistribution("size", sizes);
  std::cout << std::endl;
  std::cout << "centroid_norms:";
  PrintDistribution("norm", norms);
  std::cout << std::endl;
  std::cout << "centroid_trailing:";
  PrintDistribution("trailing", trailing);
  std::cout << std::endl;
  std::cout << "centroid_nonfinite_dims:";
  PrintDistribution("nonfinite", nonfinite);
  std::cout << std::endl;

  const std::string prefix =
      FLAGS_csv.empty() ? FLAGS_index : FLAGS_csv;
  {
    std::ofstream lists_csv(prefix + ".lists.csv");
    if (lists_csv) {
      lists_csv << "list_id,vector_count\n";
      for (size_t i = 0; i < lists.size(); ++i) {
        lists_csv << i << ',' << lists[i].vector_count << '\n';
      }
    }
    std::ofstream centroids_csv(prefix + ".centroids.csv");
    if (centroids_csv) {
      centroids_csv << "centroid_id,effective_norm,trailing_value,"
                       "nonfinite_dims\n";
      for (size_t i = 0; i < norms.size(); ++i) {
        centroids_csv << i << ',' << norms[i] << ',' << trailing[i] << ','
                      << nonfinite[i] << '\n';
      }
    }
  }
  std::cout << "csv_prefix=" << prefix << std::endl;
  return 0;
}
