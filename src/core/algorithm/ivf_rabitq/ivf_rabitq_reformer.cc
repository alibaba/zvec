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

#include "ivf_rabitq_reformer.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <vector>
#include <rabitqlib/defines.hpp>
#include <rabitqlib/index/estimator.hpp>
#include <rabitqlib/index/query.hpp>
#include <rabitqlib/quantization/data_layout.hpp>
#include <rabitqlib/quantization/rabitq.hpp>
#include <rabitqlib/utils/rotator.hpp>
#include <rabitqlib/utils/space.hpp>
#include <zvec/ailego/logger/logger.h>
#include "algorithm/hnsw_rabitq/rabitq_utils.h"
#include "core/algorithm/cluster/linear_seeker.h"
#include "zvec/core/framework/index_error.h"
#include "zvec/core/framework/index_factory.h"
#include "zvec/core/framework/index_features.h"
#include "zvec/core/framework/index_metric.h"
#include "zvec/core/framework/index_storage.h"

namespace zvec {
namespace core {

namespace {

struct ReformerLayout {
  size_t rotated_centroids_size{0};
  size_t centroids_size{0};
  size_t rotator_size{0};
};

int ValidateReformerHeader(const RabitqConverterHeader &header,
                           size_t segment_size, ReformerLayout *layout) {
  if (!layout) {
    return IndexError_InvalidArgument;
  }
  if (header.dim < kMinRabitqDimSize || header.dim > kMaxRabitqDimSize) {
    LOG_ERROR("Invalid IVF RaBitQ reformer dimension=%zu", (size_t)header.dim);
    return IndexError_InvalidFormat;
  }
  uint32_t expected_padded_dim = ((header.dim + 63U) / 64U) * 64U;
  if (header.padded_dim != expected_padded_dim || header.num_clusters == 0 ||
      header.ex_bits > 8 ||
      header.rotator_type >
          static_cast<uint8_t>(rabitqlib::RotatorType::FhtKacRotator)) {
    LOG_ERROR(
        "Invalid IVF RaBitQ reformer header: padded_dim=%zu, clusters=%zu, "
        "ex_bits=%zu, rotator_type=%zu",
        (size_t)header.padded_dim, (size_t)header.num_clusters,
        (size_t)header.ex_bits, (size_t)header.rotator_type);
    return IndexError_InvalidFormat;
  }

  layout->rotated_centroids_size = static_cast<size_t>(header.num_clusters) *
                                   header.padded_dim * sizeof(float);
  layout->centroids_size =
      static_cast<size_t>(header.num_clusters) * header.dim * sizeof(float);
  if (header.rotator_type ==
      static_cast<uint8_t>(rabitqlib::RotatorType::MatrixRotator)) {
    layout->rotator_size =
        static_cast<size_t>(header.dim) * header.padded_dim * sizeof(float);
  } else {
    layout->rotator_size = header.padded_dim / 2U;
  }
  if (header.rotator_size != layout->rotator_size) {
    LOG_ERROR("Invalid IVF RaBitQ rotator size=%zu, expected=%zu",
              (size_t)header.rotator_size, layout->rotator_size);
    return IndexError_InvalidFormat;
  }

  size_t expected_size = sizeof(RabitqConverterHeader) +
                         layout->rotated_centroids_size +
                         layout->centroids_size + layout->rotator_size;
  if (expected_size != segment_size) {
    LOG_ERROR("Invalid IVF RaBitQ reformer segment size=%zu, expected=%zu",
              segment_size, expected_size);
    return IndexError_InvalidFormat;
  }
  return 0;
}

}  // namespace

// All rabitqlib types are confined to this translation unit via pimpl.
struct IvfRabitqReformer::Impl {
  // RaBitQ parameters
  size_t num_clusters{0};
  size_t ex_bits{0};
  size_t dimension{0};
  size_t padded_dim{0};
  bool is_loaded{false};

  // Original centroids: num_clusters * dimension (for nearest centroid search)
  std::vector<float> centroids;
  // Rotated centroids: num_clusters * padded_dim (for quantization)
  std::vector<float> rotated_centroids;
  // Squared norms of original centroids, used to derive residual norms from
  // coarse InnerProduct scores.
  std::vector<float> centroid_norm_sqs;

  rabitqlib::RotatorType rotator_type{rabitqlib::RotatorType::FhtKacRotator};
  std::unique_ptr<rabitqlib::Rotator<float>> rotator;

  // Precomputed configs for quantization
  rabitqlib::quant::RabitqConfig query_config;
  rabitqlib::quant::RabitqConfig build_config;
  rabitqlib::MetricType metric_type{rabitqlib::METRIC_L2};

  // LinearSeeker for SIMD-optimized nearest centroid search
  LinearSeeker::Pointer centroid_seeker;
  CoherentIndexFeatures::Pointer centroid_features;
  IndexMetric::Pointer centroid_metric;
  IndexMetric::MatrixDistance centroid_distance_func;

  // Translate string metric to rabitqlib enum
  static rabitqlib::MetricType metric_from_string(const std::string &name) {
    if (name == "InnerProduct" || name == "Cosine") {
      return rabitqlib::METRIC_IP;
    }
    return rabitqlib::METRIC_L2;
  }

  // Translate rabitqlib enum to local enum
  static RabitqMetricType to_local(rabitqlib::MetricType m) {
    return m == rabitqlib::METRIC_IP ? RabitqMetricType::kIP
                                     : RabitqMetricType::kL2;
  }
};

IvfRabitqReformer::IvfRabitqReformer() : impl_(std::make_unique<Impl>()) {}

IvfRabitqReformer::~IvfRabitqReformer() = default;

size_t IvfRabitqReformer::num_clusters() const {
  return impl_->num_clusters;
}

size_t IvfRabitqReformer::dimension() const {
  return impl_->dimension;
}

size_t IvfRabitqReformer::padded_dim() const {
  return impl_->padded_dim;
}

size_t IvfRabitqReformer::ex_bits() const {
  return impl_->ex_bits;
}

RabitqMetricType IvfRabitqReformer::rabitq_metric_type() const {
  return Impl::to_local(impl_->metric_type);
}

bool IvfRabitqReformer::loaded() const {
  return impl_->is_loaded;
}

int IvfRabitqReformer::init(const std::string &metric_name) {
  impl_->metric_type = Impl::metric_from_string(metric_name);
  LOG_DEBUG("IvfRabitqReformer init done. metric_name=%s metric_type=%d",
            metric_name.c_str(), static_cast<int>(impl_->metric_type));
  return 0;
}

int IvfRabitqReformer::load(IndexStorage::Pointer storage) {
  if (!storage) {
    LOG_ERROR("Invalid storage for load");
    return IndexError_InvalidArgument;
  }

  auto segment = storage->get(RABITQ_CONVERTER_SEG_ID);
  if (!segment) {
    LOG_ERROR("Failed to get segment %s", RABITQ_CONVERTER_SEG_ID.c_str());
    return IndexError_InvalidFormat;
  }

  size_t offset = 0;
  RabitqConverterHeader header;
  IndexStorage::MemoryBlock block;

  size_t size = segment->read(offset, block, sizeof(header));
  if (size != sizeof(header)) {
    LOG_ERROR("Failed to read header");
    return IndexError_InvalidFormat;
  }
  memcpy(static_cast<void *>(&header), block.data(), sizeof(header));
  ReformerLayout layout;
  int ret = ValidateReformerHeader(header, segment->data_size(), &layout);
  if (ret != 0) {
    return ret;
  }
  impl_->dimension = header.dim;
  impl_->padded_dim = header.padded_dim;
  impl_->ex_bits = header.ex_bits;
  impl_->num_clusters = header.num_clusters;
  impl_->rotator_type =
      static_cast<rabitqlib::RotatorType>(header.rotator_type);
  offset += sizeof(header);

  // Read rotated centroids
  size = segment->read(offset, block, layout.rotated_centroids_size);
  if (size != layout.rotated_centroids_size) {
    LOG_ERROR("Failed to read rotated centroids");
    return IndexError_InvalidFormat;
  }
  impl_->rotated_centroids.resize(header.num_clusters * header.padded_dim);
  memcpy(impl_->rotated_centroids.data(), block.data(),
         layout.rotated_centroids_size);
  offset += size;

  // Read original centroids
  size = segment->read(offset, block, layout.centroids_size);
  if (size != layout.centroids_size) {
    LOG_ERROR("Failed to read centroids");
    return IndexError_InvalidFormat;
  }
  impl_->centroids.resize(header.num_clusters * header.dim);
  memcpy(impl_->centroids.data(), block.data(), layout.centroids_size);
  offset += size;
  impl_->centroid_norm_sqs.resize(header.num_clusters);
  for (size_t i = 0; i < header.num_clusters; ++i) {
    const float *centroid = impl_->centroids.data() + i * header.dim;
    impl_->centroid_norm_sqs[i] =
        std::inner_product(centroid, centroid + header.dim, centroid, 0.0f);
  }

  // Read rotator
  size = segment->read(offset, block, layout.rotator_size);
  if (size != layout.rotator_size) {
    LOG_ERROR("Failed to read rotator");
    return IndexError_InvalidFormat;
  }
  impl_->rotator.reset(rabitqlib::choose_rotator<float>(
      impl_->dimension, impl_->rotator_type, impl_->padded_dim));
  if (!impl_->rotator) {
    LOG_ERROR("Failed to create IVF RaBitQ rotator");
    return IndexError_InvalidFormat;
  }
  impl_->rotator->load(reinterpret_cast<const char *>(block.data()));
  offset += size;

  // Precompute configs
  impl_->query_config = rabitqlib::quant::faster_config(
      impl_->padded_dim, rabitqlib::SplitSingleQuery<float>::kNumBits);
  impl_->build_config =
      rabitqlib::quant::faster_config(impl_->padded_dim, impl_->ex_bits + 1);

  // Initialize LinearSeeker for SIMD-optimized centroid search
  IndexMeta centroid_meta;
  centroid_meta.set_data_type(IndexMeta::DataType::DT_FP32);
  centroid_meta.set_dimension(static_cast<uint32_t>(impl_->dimension));
  // Cosine maps to IP internally; use InnerProduct for centroid search
  // since input vectors are already normalized by CosineNormalizeConverter
  centroid_meta.set_metric(impl_->metric_type == rabitqlib::METRIC_L2
                               ? "SquaredEuclidean"
                               : "InnerProduct",
                           0, ailego::Params());

  impl_->centroid_features = std::make_shared<CoherentIndexFeatures>();
  impl_->centroid_features->mount(centroid_meta, impl_->centroids.data(),
                                  impl_->centroids.size() * sizeof(float));

  impl_->centroid_seeker = std::make_shared<LinearSeeker>();
  ret = impl_->centroid_seeker->init(centroid_meta);
  if (ret != 0) {
    LOG_ERROR("Failed to init centroid seeker, ret=%d", ret);
    return ret;
  }
  ret = impl_->centroid_seeker->mount(impl_->centroid_features);
  if (ret != 0) {
    LOG_ERROR("Failed to mount centroid features, ret=%d", ret);
    return ret;
  }
  impl_->centroid_metric =
      IndexFactory::CreateMetric(centroid_meta.metric_name());
  if (!impl_->centroid_metric) {
    LOG_ERROR("Failed to create centroid metric %s",
              centroid_meta.metric_name().c_str());
    return IndexError_Unsupported;
  }
  ret = impl_->centroid_metric->init(centroid_meta,
                                     centroid_meta.metric_params());
  if (ret != 0) {
    LOG_ERROR("Failed to init centroid metric, ret=%d", ret);
    return ret;
  }
  impl_->centroid_distance_func = impl_->centroid_metric->distance_matrix(1, 1);
  if (!impl_->centroid_distance_func) {
    LOG_ERROR("Failed to get centroid distance function");
    return IndexError_Unsupported;
  }

  impl_->is_loaded = true;

  LOG_INFO(
      "IvfRabitqReformer loaded: dimension=%zu, padded_dim=%zu, "
      "ex_bits=%zu, num_clusters=%zu, rotator_type=%d",
      impl_->dimension, impl_->padded_dim, impl_->ex_bits, impl_->num_clusters,
      static_cast<int>(impl_->rotator_type));
  return 0;
}

int IvfRabitqReformer::dump(const IndexStorage::Pointer &storage) {
  if (!storage) {
    LOG_ERROR("Null storage");
    return IndexError_InvalidArgument;
  }
  if (!impl_->is_loaded || impl_->rotated_centroids.empty() ||
      impl_->centroids.empty()) {
    LOG_ERROR("No centroids to dump");
    return IndexError_NoReady;
  }

  auto align_size = [](size_t s) -> size_t { return (s + 0x1F) & (~0x1F); };

  size_t header_size = sizeof(RabitqConverterHeader);
  size_t rotated_centroids_size =
      impl_->rotated_centroids.size() * sizeof(float);
  size_t centroids_size = impl_->centroids.size() * sizeof(float);
  size_t rotator_size = impl_->rotator->dump_bytes();
  size_t data_size =
      header_size + rotated_centroids_size + centroids_size + rotator_size;
  size_t total_size = align_size(data_size);

  int ret = storage->append(RABITQ_CONVERTER_SEG_ID, total_size);
  if (ret != 0) {
    LOG_ERROR("Failed to append segment %s, ret=%d",
              RABITQ_CONVERTER_SEG_ID.c_str(), ret);
    return ret;
  }

  auto segment = storage->get(RABITQ_CONVERTER_SEG_ID);
  if (!segment) {
    LOG_ERROR("Failed to get segment %s", RABITQ_CONVERTER_SEG_ID.c_str());
    return IndexError_ReadData;
  }

  size_t offset = 0;

  RabitqConverterHeader header;
  header.dim = static_cast<uint32_t>(impl_->dimension);
  header.padded_dim = static_cast<uint32_t>(impl_->padded_dim);
  header.num_clusters = static_cast<uint32_t>(impl_->num_clusters);
  header.ex_bits = static_cast<uint8_t>(impl_->ex_bits);
  header.rotator_type = static_cast<uint8_t>(impl_->rotator_type);
  header.rotator_size = static_cast<uint32_t>(rotator_size);

  size_t written = segment->write(offset, &header, header_size);
  if (written != header_size) {
    LOG_ERROR("Failed to write header");
    return IndexError_WriteData;
  }
  offset += header_size;

  written = segment->write(offset, impl_->rotated_centroids.data(),
                           rotated_centroids_size);
  if (written != rotated_centroids_size) {
    LOG_ERROR("Failed to write rotated centroids");
    return IndexError_WriteData;
  }
  offset += rotated_centroids_size;

  written = segment->write(offset, impl_->centroids.data(), centroids_size);
  if (written != centroids_size) {
    LOG_ERROR("Failed to write centroids");
    return IndexError_WriteData;
  }
  offset += centroids_size;

  std::vector<char> buffer(rotator_size);
  impl_->rotator->save(buffer.data());
  written = segment->write(offset, buffer.data(), rotator_size);
  if (written != rotator_size) {
    LOG_ERROR("Failed to write rotator data");
    return IndexError_WriteData;
  }

  LOG_INFO("IvfRabitqReformer dumped to storage: %zu bytes", data_size);
  return 0;
}

int IvfRabitqReformer::dump(const IndexDumper::Pointer &dumper) {
  if (!dumper) {
    LOG_ERROR("Null dumper");
    return IndexError_InvalidArgument;
  }
  if (!impl_->is_loaded || impl_->rotated_centroids.empty() ||
      impl_->centroids.empty()) {
    LOG_ERROR("No centroids to dump");
    return IndexError_NoReady;
  }

  size_t dumped_size = 0;
  int ret = dump_rabitq_centroids(
      dumper, impl_->dimension, impl_->padded_dim, impl_->ex_bits,
      impl_->num_clusters, impl_->rotator_type, impl_->rotated_centroids,
      impl_->centroids, impl_->rotator, &dumped_size);
  if (ret != 0) {
    return ret;
  }

  LOG_INFO("IvfRabitqReformer dump completed: %zu bytes", dumped_size);
  return 0;
}

int IvfRabitqReformer::create_query_state(const float *query,
                                          IvfRabitqQueryState *state) const {
  if (!impl_->is_loaded) {
    LOG_ERROR("Reformer not loaded yet");
    return IndexError_NoReady;
  }
  if (!query || !state) {
    LOG_ERROR("Invalid arguments for create_query_state");
    return IndexError_InvalidArgument;
  }

  // Rotate query
  state->rotated_query.resize(impl_->padded_dim);
  impl_->rotator->rotate(query, state->rotated_query.data());
  state->query_norm_sq =
      std::inner_product(query, query + impl_->dimension, query, 0.0f);

  // Create SplitBatchQuery on first use, then reuse its LUT buffers.
  auto batch_query =
      std::static_pointer_cast<rabitqlib::SplitBatchQuery<float>>(
          state->batch_query);
  if (!batch_query) {
    batch_query = std::make_shared<rabitqlib::SplitBatchQuery<float>>(
        state->rotated_query.data(), impl_->padded_dim, impl_->ex_bits,
        impl_->metric_type, true /* use_hacc */);
    state->batch_query = batch_query;
  } else {
    batch_query->reset(state->rotated_query.data(), impl_->padded_dim,
                       impl_->ex_bits, impl_->metric_type, true /* use_hacc */);
  }

  return 0;
}

int IvfRabitqReformer::prepare_for_cluster(
    const IvfRabitqProbeCentroid &centroid, IvfRabitqQueryState *state) const {
  if (!impl_->is_loaded || !state || !state->batch_query) {
    return IndexError_NoReady;
  }
  if (centroid.id >= impl_->num_clusters) {
    LOG_ERROR("Invalid centroid_id=%zu, num_clusters=%zu", (size_t)centroid.id,
              impl_->num_clusters);
    return IndexError_OutOfRange;
  }

  auto *bq = static_cast<rabitqlib::SplitBatchQuery<float> *>(
      state->batch_query.get());

  if (impl_->metric_type == rabitqlib::METRIC_L2) {
    bq->set_g_add(centroid.residual_norm);
  } else {
    bq->set_g_add(centroid.residual_norm, centroid.inner_product);
  }

  return 0;
}

int IvfRabitqReformer::prepare_for_cluster(uint32_t centroid_id,
                                           IvfRabitqQueryState *state) const {
  if (!impl_->is_loaded || !state || !state->batch_query) {
    return IndexError_NoReady;
  }
  if (centroid_id >= impl_->num_clusters) {
    LOG_ERROR("Invalid centroid_id=%zu, num_clusters=%zu", (size_t)centroid_id,
              impl_->num_clusters);
    return IndexError_OutOfRange;
  }

  const float *centroid =
      impl_->rotated_centroids.data() + centroid_id * impl_->padded_dim;
  IvfRabitqProbeCentroid probe;
  probe.id = centroid_id;
  probe.residual_norm =
      std::sqrt(std::max(rabitqlib::euclidean_sqr(state->rotated_query.data(),
                                                  centroid, impl_->padded_dim),
                         0.0f));
  if (impl_->metric_type == rabitqlib::METRIC_IP) {
    probe.inner_product = rabitqlib::dot_product(state->rotated_query.data(),
                                                 centroid, impl_->padded_dim);
  }
  return prepare_for_cluster(probe, state);
}

int IvfRabitqReformer::rotate_vector(const float *input, float *output) const {
  if (!impl_->is_loaded || !impl_->rotator) {
    return IndexError_NoReady;
  }
  impl_->rotator->rotate(input, output);
  return 0;
}

int IvfRabitqReformer::quantize_batch(const float *rotated_data,
                                      uint32_t centroid_id, size_t num_points,
                                      char *batch_data, char *ex_data) const {
  if (!impl_->is_loaded) {
    return IndexError_NoReady;
  }
  if (centroid_id >= impl_->num_clusters) {
    LOG_ERROR("Invalid centroid_id=%zu, num_clusters=%zu", (size_t)centroid_id,
              impl_->num_clusters);
    return IndexError_OutOfRange;
  }

  const float *rotated_centroid =
      impl_->rotated_centroids.data() + (centroid_id * impl_->padded_dim);

  rabitqlib::quant::quantize_split_batch(
      rotated_data, rotated_centroid, num_points, impl_->padded_dim,
      impl_->ex_bits, batch_data, ex_data, impl_->metric_type,
      impl_->build_config);

  return 0;
}

uint32_t IvfRabitqReformer::find_nearest_centroid(const float *vector) const {
  if (!impl_->is_loaded || impl_->num_clusters == 0) {
    return 0;
  }

  Seeker::Document doc;
  int ret = impl_->centroid_seeker->seek(
      vector, impl_->dimension * sizeof(float), &doc);
  if (ret != 0) {
    LOG_ERROR("Failed to seek nearest centroid, ret=%d", ret);
    return 0;
  }
  return doc.index;
}

int IvfRabitqReformer::select_probe_centroids(
    const float *query, size_t nprobe, std::vector<uint32_t> *centroids) const {
  if (!centroids) {
    return IndexError_InvalidArgument;
  }
  IvfRabitqQueryState state;
  if (query) {
    state.query_norm_sq =
        std::inner_product(query, query + impl_->dimension, query, 0.0f);
  }
  std::vector<IvfRabitqProbeCentroid> probes;
  int ret = select_probe_centroids(query, nprobe, &state, &probes);
  if (ret != 0) {
    return ret;
  }
  centroids->clear();
  centroids->reserve(probes.size());
  for (const auto &probe : probes) {
    centroids->push_back(probe.id);
  }
  return 0;
}

int IvfRabitqReformer::select_probe_centroids(
    const float *query, size_t nprobe, IvfRabitqQueryState *state,
    std::vector<IvfRabitqProbeCentroid> *centroids) const {
  if (!impl_->is_loaded || impl_->num_clusters == 0) {
    return IndexError_NoReady;
  }
  if (!query || !centroids) {
    return IndexError_InvalidArgument;
  }
  if (!state) {
    return IndexError_InvalidArgument;
  }
  if (!impl_->centroid_distance_func) {
    LOG_ERROR("Centroid distance function is not initialized");
    return IndexError_NoReady;
  }

  size_t probe_count = std::min(nprobe, impl_->num_clusters);
  centroids->clear();
  if (probe_count == 0) {
    return 0;
  }
  if (probe_count == 1) {
    Seeker::Document doc;
    int ret = impl_->centroid_seeker->seek(
        query, impl_->dimension * sizeof(float), &doc);
    if (ret != 0) {
      LOG_ERROR("Failed to seek probe centroid, ret=%d", ret);
      return ret;
    }
    IvfRabitqProbeCentroid probe;
    probe.id = doc.index;
    if (impl_->metric_type == rabitqlib::METRIC_L2) {
      probe.residual_norm = std::sqrt(std::max(doc.score, 0.0f));
    } else {
      probe.inner_product = -doc.score;
      const float residual_norm_sq = state->query_norm_sq +
                                     impl_->centroid_norm_sqs[doc.index] -
                                     2.0f * probe.inner_product;
      probe.residual_norm = std::sqrt(std::max(residual_norm_sq, 0.0f));
    }
    centroids->push_back(probe);
    return 0;
  }

  struct CentroidDist {
    uint32_t id;
    float dist;
  };

  std::vector<CentroidDist> centroid_dists(impl_->num_clusters);
  for (size_t i = 0; i < impl_->num_clusters; ++i) {
    float score = 0.0f;
    impl_->centroid_distance_func(impl_->centroid_features->element(i), query,
                                  impl_->dimension, &score);
    centroid_dists[i].id = static_cast<uint32_t>(i);
    centroid_dists[i].dist = score;
  }

  auto compare_dist = [](const CentroidDist &a, const CentroidDist &b) {
    if (a.dist == b.dist) {
      return a.id < b.id;
    }
    return a.dist < b.dist;
  };
  auto top_end = centroid_dists.begin() + probe_count;
  if (top_end != centroid_dists.end()) {
    std::nth_element(centroid_dists.begin(), top_end, centroid_dists.end(),
                     compare_dist);
  }
  std::sort(centroid_dists.begin(), top_end, compare_dist);

  centroids->reserve(probe_count);
  for (size_t i = 0; i < probe_count; ++i) {
    uint32_t centroid_id = centroid_dists[i].id;
    IvfRabitqProbeCentroid probe;
    probe.id = centroid_id;
    if (impl_->metric_type == rabitqlib::METRIC_L2) {
      probe.residual_norm = std::sqrt(std::max(centroid_dists[i].dist, 0.0f));
    } else {
      probe.inner_product = -centroid_dists[i].dist;
      const float residual_norm_sq = state->query_norm_sq +
                                     impl_->centroid_norm_sqs[centroid_id] -
                                     2.0f * probe.inner_product;
      probe.residual_norm = std::sqrt(std::max(residual_norm_sq, 0.0f));
    }
    centroids->push_back(probe);
  }
  return 0;
}

}  // namespace core
}  // namespace zvec
