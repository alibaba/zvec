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

#include "rabitq_reformer.h"
#include <string>
#include <vector>
#include <rabitqlib/defines.hpp>
#include <rabitqlib/index/query.hpp>
#include <rabitqlib/quantization/rabitq.hpp>
#include <rabitqlib/utils/rotator.hpp>
#include <zvec/ailego/logger/logger.h>
#include <zvec/ailego/utility/string_helper.h>
#include "core/algorithm/cluster/linear_seeker.h"
#include "zvec/core/framework/index_error.h"
#include "zvec/core/framework/index_factory.h"
#include "zvec/core/framework/index_features.h"
#include "zvec/core/framework/index_meta.h"
#include "zvec/core/framework/index_storage.h"
#include "hnsw_rabitq_query_entity.h"
#include "rabitq_converter.h"
#include "rabitq_utils.h"

namespace zvec {
namespace core {

// All rabitqlib types are confined to this translation unit via pimpl.
struct RabitqReformer::Impl {
  // RaBitQ parameters
  size_t num_clusters{0};
  size_t ex_bits{0};
  size_t dimension{0};
  size_t padded_dim{0};
  size_t size_bin_data{0};
  size_t size_ex_data{0};
  bool loaded{false};

  // Original centroids: num_clusters * dimension (for LinearSeeker query)
  std::vector<float> centroids;
  // Rotated centroids: num_clusters * padded_dim (for quantization)
  std::vector<float> rotated_centroids;

  rabitqlib::RotatorType rotator_type{rabitqlib::RotatorType::FhtKacRotator};
  std::unique_ptr<rabitqlib::Rotator<float>> rotator;
  rabitqlib::quant::RabitqConfig query_config;
  rabitqlib::quant::RabitqConfig config;
  rabitqlib::MetricType metric_type{rabitqlib::METRIC_L2};

  LinearSeeker::Pointer centroid_seeker;
  CoherentIndexFeatures::Pointer centroid_features;

  // Translate local enum to rabitqlib enum (used only inside this .cc).
  static rabitqlib::MetricType to_rabitq(RabitqMetricType m) {
    return m == RabitqMetricType::kIP ? rabitqlib::METRIC_IP
                                      : rabitqlib::METRIC_L2;
  }

  // Translate rabitqlib enum to local enum.
  static RabitqMetricType from_rabitq(rabitqlib::MetricType m) {
    return m == rabitqlib::METRIC_IP ? RabitqMetricType::kIP
                                     : RabitqMetricType::kL2;
  }

  int quantize_vector(const float *raw_vector, uint32_t cluster_id,
                      std::string *quantized_data) const;
};

RabitqReformer::RabitqReformer() : impl_(std::make_unique<Impl>()) {}

RabitqReformer::~RabitqReformer() {
  this->cleanup();
}

size_t RabitqReformer::num_clusters() const {
  return impl_->num_clusters;
}

RabitqMetricType RabitqReformer::rabitq_metric_type() const {
  return Impl::from_rabitq(impl_->metric_type);
}

int RabitqReformer::init(const ailego::Params &params) {
  std::string metric_name = params.get_as_string(PARAM_RABITQ_METRIC_NAME);
  if (metric_name == "SquaredEuclidean") {
    impl_->metric_type = rabitqlib::METRIC_L2;
  } else if (metric_name == "InnerProduct") {
    impl_->metric_type = rabitqlib::METRIC_IP;
  } else if (metric_name == "Cosine") {
    impl_->metric_type = rabitqlib::METRIC_IP;
  } else {
    LOG_ERROR("Unsupported metric name: %s", metric_name.c_str());
    return IndexError_InvalidArgument;
  }
  LOG_DEBUG("Rabitq reformer init done. metric_name=%s metric_type=%d",
            metric_name.c_str(), static_cast<int>(impl_->metric_type));
  return 0;
}

int RabitqReformer::cleanup() {
  impl_->centroids.clear();
  impl_->rotated_centroids.clear();
  impl_->centroid_seeker.reset();
  impl_->centroid_features.reset();
  impl_->loaded = false;
  impl_->rotator.reset();
  return 0;
}

int RabitqReformer::unload() {
  return this->cleanup();
}

int RabitqReformer::load(IndexStorage::Pointer storage) {
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
  memcpy(&header, block.data(), sizeof(header));
  impl_->dimension = header.dim;
  impl_->padded_dim = header.padded_dim;
  impl_->ex_bits = header.ex_bits;
  impl_->num_clusters = header.num_clusters;
  impl_->rotator_type =
      static_cast<rabitqlib::RotatorType>(header.rotator_type);
  offset += sizeof(header);

  // Read rotated centroids
  size_t rotated_centroids_size =
      sizeof(float) * header.num_clusters * header.padded_dim;
  size = segment->read(offset, block, rotated_centroids_size);
  if (size != rotated_centroids_size) {
    LOG_ERROR("Failed to read rotated centroids");
    return IndexError_InvalidFormat;
  }
  impl_->rotated_centroids.resize(header.num_clusters * header.padded_dim);
  memcpy(impl_->rotated_centroids.data(), block.data(), rotated_centroids_size);
  offset += size;

  // Read original centroids (for LinearSeeker query)
  size_t centroids_size = sizeof(float) * header.num_clusters * header.dim;
  size = segment->read(offset, block, centroids_size);
  if (size != centroids_size) {
    LOG_ERROR("Failed to read centroids");
    return IndexError_InvalidFormat;
  }
  impl_->centroids.resize(header.num_clusters * header.dim);
  memcpy(impl_->centroids.data(), block.data(), centroids_size);
  offset += size;

  // Read rotator
  size_t rotator_size = header.rotator_size;
  size = segment->read(offset, block, rotator_size);
  if (size != rotator_size) {
    LOG_ERROR("Failed to read rotator");
    return IndexError_InvalidFormat;
  }
  impl_->rotator.reset(rabitqlib::choose_rotator<float>(
      impl_->dimension, impl_->rotator_type, impl_->padded_dim));
  impl_->rotator->load(reinterpret_cast<const char *>(block.data()));
  offset += size;

  impl_->query_config = rabitqlib::quant::faster_config(
      impl_->padded_dim, rabitqlib::SplitSingleQuery<float>::kNumBits);
  impl_->config =
      rabitqlib::quant::faster_config(impl_->padded_dim, impl_->ex_bits + 1);

  impl_->size_bin_data =
      rabitqlib::BinDataMap<float>::data_bytes(impl_->padded_dim);
  impl_->size_ex_data = rabitqlib::ExDataMap<float>::data_bytes(
      impl_->padded_dim, impl_->ex_bits);

  // Initialize LinearSeeker for centroid search
  IndexMeta centroid_meta;
  centroid_meta.set_data_type(IndexMeta::DataType::DT_FP32);
  centroid_meta.set_dimension(static_cast<uint32_t>(impl_->dimension));
  // Note:
  // 1. spherical kmeans is used for InnerProduct and Cosine, so centroids are
  // normalized.
  // 2. for Cosine metric, `transform_to_entity` input is normalized, need to
  // use InnerProduct metric as Cosine metric requires extra dimension which is
  // unsuitable for centroids.
  centroid_meta.set_metric(impl_->metric_type == rabitqlib::METRIC_L2
                               ? "SquaredEuclidean"
                               : "InnerProduct",
                           0, ailego::Params());

  impl_->centroid_features = std::make_shared<CoherentIndexFeatures>();
  impl_->centroid_features->mount(centroid_meta, impl_->centroids.data(),
                                  impl_->centroids.size() * sizeof(float));

  impl_->centroid_seeker = std::make_shared<LinearSeeker>();
  int ret = impl_->centroid_seeker->init(centroid_meta);
  if (ret != 0) {
    LOG_ERROR("Failed to init centroid seeker. ret[%d]", ret);
    return ret;
  }
  ret = impl_->centroid_seeker->mount(impl_->centroid_features);
  if (ret != 0) {
    LOG_ERROR("Failed to mount centroid features. ret[%d]", ret);
    return ret;
  }

  LOG_INFO(
      "Rabitq reformer load done. dimension=%zu, padded_dim=%zu, "
      "ex_bits=%zu, num_clusters=%zu, size_bin_data=%zu, size_ex_data=%zu "
      "rotator_type=%d",
      impl_->dimension, impl_->padded_dim, impl_->ex_bits, impl_->num_clusters,
      impl_->size_bin_data, impl_->size_ex_data, (int)impl_->rotator_type);
  impl_->loaded = true;
  return 0;
}

int RabitqReformer::convert(const void *record, const IndexQueryMeta &rmeta,
                            std::string *out, IndexQueryMeta *ometa) const {
  if (!impl_->loaded) {
    LOG_ERROR("Centroids not loaded yet");
    return IndexError_NoReady;
  }

  if (!record || !out) {
    LOG_ERROR("Invalid arguments for convert");
    return IndexError_InvalidArgument;
  }

  // input may be transformed, require rmeta.dimension >= dimension
  if (rmeta.dimension() < impl_->dimension ||
      rmeta.data_type() != IndexMeta::DataType::DT_FP32) {
    LOG_ERROR("Invalid record meta: dimension=%zu, data_type=%d",
              static_cast<size_t>(rmeta.dimension()), (int)rmeta.data_type());
    return IndexError_InvalidArgument;
  }

  // Find nearest centroid using LinearSeeker
  Seeker::Document doc;
  int ret = impl_->centroid_seeker->seek(
      record, impl_->dimension * sizeof(float), &doc);
  if (ret != 0) {
    LOG_ERROR("Failed to seek centroid. ret[%d]", ret);
    return ret;
  }
  uint32_t cluster_id = doc.index;

  const float *vector = static_cast<const float *>(record);
  ret = impl_->quantize_vector(vector, cluster_id, out);
  if (ret != 0) {
    LOG_ERROR("Failed to quantize vector");
    return ret;
  }

  ometa->set_meta(IndexMeta::DataType::DT_INT8, (uint32_t)out->size());
  return 0;
}

int RabitqReformer::transform(const void *, const IndexQueryMeta &,
                              std::string *, IndexQueryMeta *) const {
  return IndexError_NotImplemented;
}

int RabitqReformer::transform_to_entity(const void *query,
                                        HnswRabitqQueryEntity *entity) const {
  if (!impl_->loaded) {
    LOG_ERROR("Centroids not loaded yet");
    return IndexError_NoReady;
  }

  if (!query) {
    LOG_ERROR("Invalid arguments for transform");
    return IndexError_InvalidArgument;
  }

  const float *query_vector = static_cast<const float *>(query);

  // Apply rotator
  entity->rotated_query.resize(impl_->padded_dim);
  impl_->rotator->rotate(query_vector, entity->rotated_query.data());

  // Quantize query to 4-bit representation
  entity->query_wrapper = std::make_unique<rabitqlib::SplitSingleQuery<float>>(
      entity->rotated_query.data(), impl_->padded_dim, impl_->ex_bits,
      impl_->query_config, impl_->metric_type);

  // Preprocess - get the distance from query to all centroids
  entity->q_to_centroids.resize(impl_->num_clusters);

  if (impl_->metric_type == rabitqlib::METRIC_L2) {
    for (size_t i = 0; i < impl_->num_clusters; i++) {
      entity->q_to_centroids[i] = std::sqrt(rabitqlib::euclidean_sqr(
          entity->rotated_query.data(),
          impl_->rotated_centroids.data() + (i * impl_->padded_dim),
          impl_->padded_dim));
    }
  } else if (impl_->metric_type == rabitqlib::METRIC_IP) {
    entity->q_to_centroids.resize(impl_->num_clusters * 2);
    // first half as g_add, second half as g_error
    for (size_t i = 0; i < impl_->num_clusters; i++) {
      entity->q_to_centroids[i] = rabitqlib::dot_product(
          entity->rotated_query.data(),
          impl_->rotated_centroids.data() + (i * impl_->padded_dim),
          impl_->padded_dim);
      entity->q_to_centroids[i + impl_->num_clusters] =
          std::sqrt(rabitqlib::euclidean_sqr(
              entity->rotated_query.data(),
              impl_->rotated_centroids.data() + (i * impl_->padded_dim),
              impl_->padded_dim));
    }
  }

  return 0;
}

int RabitqReformer::Impl::quantize_vector(const float *raw_vector,
                                          uint32_t cluster_id,
                                          std::string *quantized_data) const {
  std::vector<float> rotated_data(padded_dim);
  rotator->rotate(raw_vector, rotated_data.data());

  // quantized format: cluster_id + bin_data + ex_data
  quantized_data->resize(sizeof(cluster_id) + size_bin_data + size_ex_data);
  memcpy(&(*quantized_data)[0], &cluster_id, sizeof(cluster_id));
  int bin_data_offset = sizeof(cluster_id);
  int ex_data_offset = bin_data_offset + size_bin_data;
  rabitqlib::quant::quantize_split_single(
      rotated_data.data(), rotated_centroids.data() + (cluster_id * padded_dim),
      padded_dim, ex_bits, &(*quantized_data)[bin_data_offset],
      &(*quantized_data)[ex_data_offset], metric_type, config);

  return 0;
}

int RabitqReformer::dump(const IndexDumper::Pointer &dumper) {
  if (!dumper) {
    LOG_ERROR("Null dumper");
    return IndexError_InvalidArgument;
  }

  if (!impl_->loaded || impl_->rotated_centroids.empty() ||
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

  LOG_INFO("RabitqReformer dump completed: %zu bytes", dumped_size);
  return 0;
}

int RabitqReformer::dump(const IndexStorage::Pointer &storage) {
  if (!storage) {
    LOG_ERROR("Null storage");
    return IndexError_InvalidArgument;
  }

  if (!impl_->loaded || impl_->rotated_centroids.empty() ||
      impl_->centroids.empty()) {
    LOG_ERROR("No centroids to dump");
    return IndexError_NoReady;
  }

  auto align_size = [](size_t size) -> size_t {
    return (size + 0x1F) & (~0x1F);
  };

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
    LOG_ERROR("Failed to write header: written=%zu, expected=%zu", written,
              header_size);
    return IndexError_WriteData;
  }
  offset += header_size;

  written = segment->write(offset, impl_->rotated_centroids.data(),
                           rotated_centroids_size);
  if (written != rotated_centroids_size) {
    LOG_ERROR("Failed to write rotated centroids: written=%zu, expected=%zu",
              written, rotated_centroids_size);
    return IndexError_WriteData;
  }
  offset += rotated_centroids_size;

  written = segment->write(offset, impl_->centroids.data(), centroids_size);
  if (written != centroids_size) {
    LOG_ERROR("Failed to write centroids: written=%zu, expected=%zu", written,
              centroids_size);
    return IndexError_WriteData;
  }
  offset += centroids_size;

  std::vector<char> buffer(rotator_size);
  impl_->rotator->save(buffer.data());
  written = segment->write(offset, buffer.data(), rotator_size);
  if (written != rotator_size) {
    LOG_ERROR("Failed to write rotator data: written=%zu, expected=%zu",
              written, rotator_size);
    return IndexError_WriteData;
  }

  LOG_INFO("RabitqReformer dump to storage completed: %zu bytes", data_size);
  return 0;
}


}  // namespace core
}  // namespace zvec
