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

// Build a standard IVF index with a turbo RaBitQ quantizer through the
// core_interface API, following docs/ivf-rabitq.md: vectors are added and
// cached, then train() clusters the lists, trains the quantizer model and
// writes the postings.
//
// Usage:
//   ivf_rabitq_builder --vecs=train.vecs --index=out.index
//     [--bits=7] [--num_clusters=16] [--sample_count=0]
//     [--metric=cosine|l2sq|ip] [--n_list=1024] [--n_iters=20]

#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <gflags/gflags.h>
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
DEFINE_int32(bits, 7, "rabitq total bits (1-9)");
DEFINE_int32(num_clusters, 16, "rabitq centroid count");
DEFINE_int32(sample_count, 0, "rabitq kmeans sample count (0 = all)");
DEFINE_string(metric, "cosine", "metric: cosine, l2sq or ip");
DEFINE_int32(n_list, 1024, "IVF list count");
DEFINE_int32(n_iters, 20, "IVF kmeans iterations");

static MetricType ParseMetric(const string &name) {
  if (name == "cosine") return MetricType::kCosine;
  if (name == "l2sq") return MetricType::kL2sq;
  if (name == "ip") return MetricType::kInnerProduct;
  return MetricType::kL2sq;
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

  RabitqQuantizerParam quantizer(FLAGS_bits);
  quantizer.num_clusters = FLAGS_num_clusters;
  quantizer.sample_count = FLAGS_sample_count;
  quantizer.niters = FLAGS_n_iters;

  auto param = IVFIndexParamBuilder()
                   .with_metric_type(metric)
                   .with_data_type(DataType::DT_FP32)
                   .with_dimension(dimension)
                   .with_n_list(FLAGS_n_list)
                   .with_n_iters(FLAGS_n_iters)
                   .with_quantizer_param(quantizer)
                   .build();

  zvec::ailego::ElapsedTime timer;
  auto index = IndexFactory::CreateAndInitIndex(*param);
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

  timer.reset();
  for (size_t i = 0; i < num_vecs; ++i) {
    // The interface add() keys documents by uint32 id, so every key must
    // fit; ground-truth ids and vecs keys share this space.
    if (reader.get_key(i) >= (1ULL << 32)) {
      cerr << "key " << reader.get_key(i) << " at row " << i
           << " does not fit uint32" << endl;
      return -1;
    }
    const uint32_t key = static_cast<uint32_t>(reader.get_key(i));
    VectorData vector_data{
        DenseVector{reinterpret_cast<const float *>(reader.get_vector(i))}};
    if (index->add(vector_data, key) != 0) {
      cerr << "add failed for key " << key << endl;
      return -1;
    }
    const size_t done = i + 1;
    if (done % 100000 == 0) {
      cout << "added " << done << " / " << num_vecs << endl;
    }
  }
  size_t add_ms = timer.milli_seconds();
  cout << "added " << num_vecs << " vectors in " << add_ms << "ms" << endl;

  timer.reset();
  if (index->train() != 0) {
    cerr << "index train failed" << endl;
    return -1;
  }
  size_t train_ms = timer.milli_seconds();
  cout << "index trained in " << train_ms << "ms" << endl;
  cout << "doc count: " << index->get_doc_count() << endl;

  timer.reset();
  if (index->close() != 0) {
    cerr << "index close failed" << endl;
    return -1;
  }
  cout << "index closed in " << timer.milli_seconds() << "ms" << endl;
  cout << "TOTAL open[" << open_ms << "ms] add[" << add_ms << "ms] train["
       << train_ms << "ms]" << endl;
  return 0;
}
