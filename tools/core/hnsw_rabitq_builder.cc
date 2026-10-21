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

// Build a standard HNSW index with an optional turbo quantizer (e.g.
// RaBitQ) through the core_interface API: the original FP32 vectors are pushed
// into a MultiPassIndexProvider before the index is created; the graph
// is built from the originals while stored codes come from the
// quantizer.
//
// Usage:
//   hnsw_rabitq_builder --vecs=train.vecs --index=out.index
//     [--quantizer=rabitq|none] [--bits=7] [--num_clusters=16]
//     [--metric=cosine|l2sq|ip] [--threads=8] [--m=50]
//     [--ef_construction=500]

#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <gflags/gflags.h>
#include <zvec/ailego/parallel/thread_pool.h>
#include <zvec/ailego/utility/time_helper.h>
#include <zvec/core/framework/index_framework.h>
#include <zvec/core/interface/index.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param_builders.h>
#include "vecs_reader.h"

using namespace std;
using namespace zvec::core_interface;

DEFINE_string(vecs, "", "input .vecs file");
DEFINE_string(index, "", "output index file path");
DEFINE_string(quantizer, "rabitq", "quantizer: rabitq or none");
DEFINE_int32(bits, 7, "rabitq total bits (1-9)");
DEFINE_int32(num_clusters, 16, "rabitq centroid count");
DEFINE_int32(sample_count, 0, "rabitq kmeans sample count (0 = all)");
DEFINE_string(metric, "cosine", "metric: cosine, l2sq or ip");
DEFINE_int32(threads, 8, "build thread count");
DEFINE_int32(m, 50, "HNSW max neighbor count");
DEFINE_int32(ef_construction, 500, "HNSW ef construction");

static MetricType ParseMetric(const string &name) {
  if (name == "cosine") return MetricType::kCosine;
  if (name == "l2sq") return MetricType::kL2sq;
  if (name == "ip") return MetricType::kInnerProduct;
  return MetricType::kL2sq;
}

static const char *MetricName(MetricType metric) {
  switch (metric) {
    case MetricType::kL2sq:
      return "SquaredEuclidean";
    case MetricType::kCosine:
      return "Cosine";
    case MetricType::kInnerProduct:
      return "InnerProduct";
    default:
      return "";
  }
}

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_vecs.empty() || FLAGS_index.empty()) {
    cerr << "--vecs and --index are required" << endl;
    return -1;
  }

  zvec::core::VecsReader reader;
  if (!reader.load(FLAGS_vecs)) {
    cerr << "failed to load " << FLAGS_vecs << endl;
    return -1;
  }
  const size_t num_vecs = reader.num_vecs();
  const uint32_t dimension =
      static_cast<uint32_t>(reader.index_meta().dimension());
  const size_t element_size = reader.index_meta().element_size();
  cout << "vecs loaded: count[" << num_vecs << "] dimension[" << dimension
       << "] element_size[" << element_size << "]" << endl;

  const MetricType metric = ParseMetric(FLAGS_metric);

  // Populate the original-vector provider before creating the index: the
  // RaBitQ centroids are trained from it on the create-open.
  auto provider = make_shared<zvec::core::MultiPassIndexProvider<
      zvec::core::IndexMeta::DataType::DT_FP32>>(dimension);
  zvec::ailego::NumericalVector<float> vector(dimension);
  for (size_t i = 0; i < num_vecs; ++i) {
    // The interface add() keys documents by uint32 id, so every key must
    // fit; ground-truth ids and vecs keys share this space.
    if (reader.get_key(i) >= (1ULL << 32)) {
      cerr << "key " << reader.get_key(i) << " at row " << i
           << " does not fit uint32" << endl;
      return -1;
    }
    memcpy(vector.data(), reader.get_vector(i), element_size);
    if (!provider->emplace(reader.get_key(i), vector)) {
      cerr << "provider emplace failed at " << i << endl;
      return -1;
    }
  }
  cout << "provider populated: " << provider->count() << " vectors" << endl;

  zvec::core::IndexMeta provider_meta(zvec::core::IndexMeta::DT_FP32,
                                      dimension);
  provider_meta.set_metric(MetricName(metric), 0, zvec::ailego::Params{});

  auto param = HNSWIndexParamBuilder()
                   .with_metric_type(metric)
                   .with_data_type(DataType::DT_FP32)
                   .with_dimension(dimension)
                   .with_is_sparse(false)
                   .with_m(FLAGS_m)
                   .with_ef_construction(FLAGS_ef_construction)
                   .with_use_id_map(true)
                   .with_provider(provider, provider_meta);
  if (FLAGS_quantizer == "rabitq") {
    RabitqQuantizerParam quantizer(FLAGS_bits);
    quantizer.num_clusters = FLAGS_num_clusters;
    quantizer.sample_count = FLAGS_sample_count;
    param.with_quantizer_param(quantizer);
  } else if (FLAGS_quantizer != "none") {
    cerr << "unknown quantizer " << FLAGS_quantizer << endl;
    return -1;
  }
  auto index_param = param.build();

  zvec::ailego::ElapsedTime timer;
  auto index = IndexFactory::CreateAndInitIndex(*index_param);
  if (!index) {
    cerr << "CreateAndInitIndex failed" << endl;
    return -1;
  }
  if (index->open(FLAGS_index, {StorageOptions::StorageType::kMMAP, true}) !=
      0) {
    cerr << "index open failed" << endl;
    return -1;
  }
  size_t open_ms = timer.milli_seconds();
  cout << "index created and opened in " << open_ms << "ms" << endl;

  // Parallel adds; every thread owns its thread-local streamer context.
  atomic<size_t> failed{0};
  atomic<size_t> finished{0};
  zvec::ailego::ThreadPool pool(static_cast<uint32_t>(FLAGS_threads), false);
  const uint32_t thread_count = static_cast<uint32_t>(pool.count());
  auto do_add = [&](size_t idx) {
    for (size_t id = idx; id < num_vecs && failed.load() == 0;
         id += thread_count) {
      const uint32_t key = static_cast<uint32_t>(reader.get_key(id));
      VectorData vector_data{
          DenseVector{reinterpret_cast<const float *>(reader.get_vector(id))}};
      if (index->add(vector_data, key) != 0) {
        ++failed;
        cerr << "add failed for key " << key << endl;
        return;
      }
      const size_t done = ++finished;
      if (done % 100000 == 0) {
        cout << "added " << done << " / " << num_vecs << endl;
      }
    }
  };
  timer.reset();
  for (size_t i = 0; i < thread_count; ++i) {
    pool.execute(do_add, i);
  }
  pool.wait_finish();
  size_t build_ms = timer.milli_seconds();
  if (failed.load() != 0) {
    cerr << "build failed" << endl;
    return -1;
  }
  cout << "added " << num_vecs << " vectors in " << build_ms << "ms" << endl;

  timer.reset();
  if (index->close() != 0) {
    cerr << "index close failed" << endl;
    return -1;
  }
  cout << "index closed in " << timer.milli_seconds() << "ms" << endl;
  cout << "TOTAL open[" << open_ms << "ms] build[" << build_ms << "ms]" << endl;
  return 0;
}
