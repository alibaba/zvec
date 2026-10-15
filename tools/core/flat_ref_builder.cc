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

// Build an exact kFlat reference index through the core_interface API.
// Unlike local_builder (FlatBuilder + FileDumper, whose package layout is
// only readable through MMapFileReadStorage/BufferReadStorage consumers such
// as IVFIndex::open), this tool creates the file with Index::open(kMMAP,
// create_new=true) so the on-disk layout is the IndexMapping container that
// the read-only MMapFileStorage path used by FlatIndex::open expects. The
// id map is enabled so the refiner's search_by_p_keys lookups resolve keys.
//
// Usage:
//   flat_ref_builder --vecs=train.vecs --index=out.index [--metric=cosine]

#include <iostream>
#include <memory>
#include <string>
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
DEFINE_string(metric, "cosine", "metric: cosine, l2sq or ip");

static bool ParseMetric(const string &name, MetricType *metric) {
  if (name == "cosine") {
    *metric = MetricType::kCosine;
  } else if (name == "l2sq") {
    *metric = MetricType::kL2sq;
  } else if (name == "ip") {
    *metric = MetricType::kInnerProduct;
  } else {
    return false;
  }
  return true;
}

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_vecs.empty() || FLAGS_index.empty()) {
    cerr << "--vecs and --index are required" << endl;
    return -1;
  }

  MetricType metric;
  if (!ParseMetric(FLAGS_metric, &metric)) {
    cerr << "invalid --metric='" << FLAGS_metric
         << "'; expected cosine, l2sq or ip (case-sensitive)" << endl;
    return -1;
  }

  zvec::core::VecsReader reader;
  if (!reader.load(FLAGS_vecs)) {
    cerr << "failed to load " << FLAGS_vecs << endl;
    return -1;
  }
  const auto &meta = reader.index_meta();
  if (meta.data_type() != zvec::core::IndexMeta::DT_FP32 ||
      meta.dimension() == 0 ||
      meta.element_size() != size_t(meta.dimension()) * sizeof(float)) {
    cerr << "input must contain dense FP32 vectors with "
            "element_size == dimension * sizeof(float); got data_type["
         << static_cast<int>(meta.data_type()) << "] dimension["
         << meta.dimension() << "] element_size[" << meta.element_size() << "]"
         << endl;
    return -1;
  }
  const size_t num_vecs = reader.num_vecs();
  const uint32_t dimension =
      static_cast<uint32_t>(reader.index_meta().dimension());
  cout << "vecs loaded: count[" << num_vecs << "] dimension[" << dimension
       << "] element_size[" << reader.index_meta().element_size() << "]" << endl;

  auto param = FlatIndexParamBuilder()
                   .with_metric_type(metric)
                   .with_data_type(DataType::DT_FP32)
                   .with_dimension(dimension)
                   .with_is_sparse(false)
                   .with_use_id_map(true)
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
  cout << "index trained in " << timer.milli_seconds() << "ms" << endl;

  timer.reset();
  if (index->close() != 0) {
    cerr << "index close failed" << endl;
    return -1;
  }
  cout << "index closed in " << timer.milli_seconds() << "ms" << endl;
  cout << "TOTAL open[" << open_ms << "ms] add[" << add_ms << "ms]" << endl;
  return 0;
}
