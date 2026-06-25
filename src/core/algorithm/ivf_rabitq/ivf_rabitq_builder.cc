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
#include "ivf_rabitq_builder.h"
#include <algorithm>
#include <cstring>
#include <numeric>
#include <rabitqlib/fastscan/fastscan.hpp>
#include <rabitqlib/quantization/data_layout.hpp>
#include <rabitqlib/quantization/rabitq.hpp>
#include <zvec/ailego/hash/crc32c.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/ailego/utility/string_helper.h>
#include <zvec/ailego/utility/time_helper.h>
#include "algorithm/hnsw_rabitq/rabitq_converter.h"
#include "algorithm/hnsw_rabitq/rabitq_params.h"
#include "zvec/core/framework/index_error.h"
#include "zvec/core/framework/index_factory.h"
#include "zvec/core/framework/index_memory.h"
#include "ivf_rabitq_params.h"
#include "ivf_rabitq_reformer.h"
#include "ivf_rabitq_util.h"

namespace zvec {
namespace core {

namespace {

struct IvfRabitqVectorEntry {
  uint64_t key{0};
  std::vector<float> owned_data;
};

struct IvfRabitqBuildData {
  IvfRabitqHeader header;
  std::vector<char> batch_data_buf;
  std::vector<char> ex_data_buf;
  std::vector<IvfRabitqClusterMeta> cluster_metas;
  std::vector<uint64_t> keys_buf;
  std::vector<uint32_t> mapping_buf;
};

int BuildIvfRabitqData(const std::shared_ptr<IvfRabitqReformer> &reformer,
                       const IndexHolder::Pointer &holder,
                       std::vector<IvfRabitqVectorEntry> extra_vectors,
                       IvfRabitqBuildData *build_data) {
  if (!reformer || !reformer->loaded()) {
    LOG_ERROR("Reformer not loaded");
    return IndexError_NoReady;
  }
  if (!build_data) {
    LOG_ERROR("Null IVF RaBitQ build data");
    return IndexError_InvalidArgument;
  }

  *build_data = IvfRabitqBuildData();

  size_t dimension = reformer->dimension();
  size_t padded_dim = reformer->padded_dim();
  size_t ex_bits = reformer->ex_bits();
  size_t cluster_count = reformer->num_clusters();

  std::vector<IvfRabitqVectorEntry> all_vectors;
  if (holder) {
    if (holder->dimension() < dimension) {
      LOG_ERROR("Holder dimension=%zu smaller than RaBitQ dimension=%zu",
                holder->dimension(), dimension);
      return IndexError_Mismatch;
    }
    auto iter = holder->create_iterator();
    while (iter && iter->is_valid()) {
      IvfRabitqVectorEntry entry;
      entry.key = iter->key();
      entry.owned_data.assign(
          static_cast<const float *>(iter->data()),
          static_cast<const float *>(iter->data()) + dimension);
      all_vectors.push_back(std::move(entry));
      iter->next();
    }
  }

  all_vectors.reserve(all_vectors.size() + extra_vectors.size());
  for (auto &entry : extra_vectors) {
    all_vectors.push_back(std::move(entry));
  }

  size_t total_count = all_vectors.size();
  if (total_count == 0) {
    LOG_WARN("No vectors to build");
    return 0;
  }

  LOG_INFO("Building IVF RaBitQ index: %zu vectors, %zu clusters", total_count,
           cluster_count);

  std::vector<std::vector<uint32_t>> cluster_assignments(cluster_count);
  for (size_t i = 0; i < total_count; ++i) {
    uint32_t cid =
        reformer->find_nearest_centroid(all_vectors[i].owned_data.data());
    cluster_assignments[cid].push_back(static_cast<uint32_t>(i));
  }

  size_t batch_data_per_batch =
      rabitqlib::BatchDataMap<float>::data_bytes(padded_dim);
  size_t ex_data_per_vector =
      rabitqlib::ExDataMap<float>::data_bytes(padded_dim, ex_bits);

  size_t total_batch_data_size = 0;
  size_t total_ex_data_size = 0;
  std::vector<uint32_t> cluster_batch_counts(cluster_count);

  for (size_t cid = 0; cid < cluster_count; ++cid) {
    uint32_t vec_count = static_cast<uint32_t>(cluster_assignments[cid].size());
    uint32_t num_batches = (vec_count + rabitqlib::fastscan::kBatchSize - 1) /
                           rabitqlib::fastscan::kBatchSize;
    cluster_batch_counts[cid] = num_batches;
    total_batch_data_size += num_batches * batch_data_per_batch;
    total_ex_data_size += vec_count * ex_data_per_vector;
  }

  build_data->batch_data_buf.assign(total_batch_data_size, 0);
  build_data->ex_data_buf.assign(total_ex_data_size, 0);
  build_data->keys_buf.resize(total_count);
  build_data->mapping_buf.resize(total_count);
  build_data->cluster_metas.resize(cluster_count);

  size_t batch_data_offset = 0;
  size_t ex_data_offset = 0;
  uint32_t key_offset = 0;
  std::vector<float> rotated_batch(rabitqlib::fastscan::kBatchSize *
                                   padded_dim);

  for (size_t cid = 0; cid < cluster_count; ++cid) {
    auto &assignments = cluster_assignments[cid];
    uint32_t vec_count = static_cast<uint32_t>(assignments.size());

    build_data->cluster_metas[cid].batch_data_offset = batch_data_offset;
    build_data->cluster_metas[cid].ex_data_offset = ex_data_offset;
    build_data->cluster_metas[cid].vector_count = vec_count;
    build_data->cluster_metas[cid].batch_count = cluster_batch_counts[cid];
    build_data->cluster_metas[cid].key_offset = key_offset;

    for (uint32_t i = 0; i < vec_count; ++i) {
      auto &entry = all_vectors[assignments[i]];
      build_data->keys_buf[key_offset + i] = entry.key;
    }

    uint32_t processed = 0;
    while (processed < vec_count) {
      uint32_t batch_size =
          std::min(static_cast<uint32_t>(rabitqlib::fastscan::kBatchSize),
                   vec_count - processed);

      std::memset(rotated_batch.data(), 0,
                  rabitqlib::fastscan::kBatchSize * padded_dim * sizeof(float));
      for (uint32_t i = 0; i < batch_size; ++i) {
        const float *src =
            all_vectors[assignments[processed + i]].owned_data.data();
        float *dst = rotated_batch.data() + (i * padded_dim);
        reformer->rotate_vector(src, dst);
      }

      char *bd_ptr = build_data->batch_data_buf.data() + batch_data_offset;
      char *ex_ptr = nullptr;
      if (ex_data_per_vector > 0) {
        ex_ptr = build_data->ex_data_buf.data() + ex_data_offset;
      }

      int ret = reformer->quantize_batch(rotated_batch.data(),
                                         static_cast<uint32_t>(cid), batch_size,
                                         bd_ptr, ex_ptr);
      if (ret != 0) {
        LOG_ERROR("Failed to quantize batch for cluster %zu, ret=%d", cid, ret);
        return ret;
      }

      batch_data_offset += batch_data_per_batch;
      ex_data_offset += batch_size * ex_data_per_vector;
      processed += batch_size;
    }

    key_offset += vec_count;
  }

  std::iota(build_data->mapping_buf.begin(), build_data->mapping_buf.end(), 0U);
  std::sort(build_data->mapping_buf.begin(), build_data->mapping_buf.end(),
            [&keys_buf = build_data->keys_buf](uint32_t lhs, uint32_t rhs) {
              if (keys_buf[lhs] == keys_buf[rhs]) {
                return lhs < rhs;
              }
              return keys_buf[lhs] < keys_buf[rhs];
            });

  build_data->header.total_vector_count = static_cast<uint32_t>(total_count);
  build_data->header.cluster_count = static_cast<uint32_t>(cluster_count);
  build_data->header.dimension = static_cast<uint32_t>(dimension);
  build_data->header.padded_dim = static_cast<uint32_t>(padded_dim);
  build_data->header.ex_bits = static_cast<uint8_t>(ex_bits);
  build_data->header.rotator_type = 0;
  build_data->header.metric_type =
      (reformer->rabitq_metric_type() == RabitqMetricType::kIP) ? 1 : 0;
  build_data->header.batch_data_size = total_batch_data_size;
  build_data->header.ex_data_size = total_ex_data_size;

  return 0;
}

}  // namespace

IvfRabitqBuilder::IvfRabitqBuilder() = default;

IvfRabitqBuilder::~IvfRabitqBuilder() {
  this->cleanup();
}

// --------------------------------------------------------------------------
// init
// --------------------------------------------------------------------------
int IvfRabitqBuilder::init(const IndexMeta &meta,
                           const ailego::Params &params) {
  if (state_ != INIT) {
    LOG_ERROR("IvfRabitqBuilder state wrong. state=%d", state_);
    return IndexError_Logic;
  }

  meta_ = meta;
  params_ = params;

  params.get(PARAM_IVF_RABITQ_NLIST, &nlist_);
  if (nlist_ == 0) {
    nlist_ = kDefaultIvfRabitqNlist;
  }

  params.get(PARAM_RABITQ_TOTAL_BITS, &total_bits_);
  if (total_bits_ == 0) {
    total_bits_ = kDefaultRabitqTotalBits;
  }
  if (total_bits_ < 1 || total_bits_ > 9) {
    LOG_ERROR("Invalid total_bits=%u, must be in [1, 9]", total_bits_);
    return IndexError_InvalidArgument;
  }

  params.get(PARAM_RABITQ_SAMPLE_COUNT, &sample_count_);

  int ret = PrepareAndCheckIvfRabitqInternalMeta(meta_, params, &rabitq_meta_,
                                                 &metric_name_);
  if (ret != 0) {
    return ret;
  }

  uint32_t dim = rabitq_meta_.dimension();
  LOG_INFO(
      "IvfRabitqBuilder initialized: dim=%u, nlist=%u, total_bits=%u, "
      "metric=%s, sample_count=%u",
      dim, nlist_, total_bits_, metric_name_.c_str(), sample_count_);

  state_ = INITED;
  return 0;
}

// --------------------------------------------------------------------------
// cleanup
// --------------------------------------------------------------------------
int IvfRabitqBuilder::cleanup() {
  state_ = INIT;
  stats_.clear();
  converter_.reset();
  reformer_.reset();
  header_ = IvfRabitqHeader();
  batch_data_buf_.clear();
  ex_data_buf_.clear();
  cluster_metas_.clear();
  keys_buf_.clear();
  mapping_buf_.clear();
  return 0;
}

// --------------------------------------------------------------------------
// train — KMeans clustering + rotator training
// --------------------------------------------------------------------------
int IvfRabitqBuilder::train(IndexThreads::Pointer /*threads*/,
                            IndexHolder::Pointer holder) {
  if (state_ != INITED) {
    LOG_ERROR("IvfRabitqBuilder train failed, wrong state=%d", state_);
    return IndexError_Runtime;
  }

  ailego::ElapsedTime timer;

  // Create and configure RabitqConverter
  converter_ = std::make_shared<RabitqConverter>();
  ailego::Params conv_params;
  conv_params.set(PARAM_RABITQ_NUM_CLUSTERS, nlist_);
  conv_params.set(PARAM_RABITQ_TOTAL_BITS, total_bits_);
  if (sample_count_ > 0) {
    conv_params.set(PARAM_RABITQ_SAMPLE_COUNT, sample_count_);
  }

  int ret = converter_->init(rabitq_meta_, conv_params);
  if (ret != 0) {
    LOG_ERROR("Failed to init RabitqConverter, ret=%d", ret);
    return ret;
  }

  // Train KMeans centroids
  ret = converter_->train(holder);
  if (ret != 0) {
    LOG_ERROR("Failed to train RabitqConverter, ret=%d", ret);
    return ret;
  }

  // Create IvfRabitqReformer via dump+load cycle
  auto memory_dumper = IndexFactory::CreateDumper("MemoryDumper");
  memory_dumper->init(ailego::Params());
  std::string file_id = ailego::StringHelper::Concat(
      "ivf_rabitq_builder_train_", ailego::Monotime::MilliSeconds(), rand());
  ret = memory_dumper->create(file_id);
  if (ret != 0) {
    LOG_ERROR("Failed to create memory dumper, ret=%d", ret);
    return ret;
  }

  ret = converter_->dump(memory_dumper);
  if (ret != 0) {
    LOG_ERROR("Failed to dump converter, ret=%d", ret);
    IndexMemory::Instance()->remove(file_id);
    return ret;
  }
  ret = memory_dumper->close();
  if (ret != 0) {
    LOG_ERROR("Failed to close memory dumper, ret=%d", ret);
    IndexMemory::Instance()->remove(file_id);
    return ret;
  }

  reformer_ = std::make_shared<IvfRabitqReformer>();
  ret = reformer_->init(metric_name_);
  if (ret != 0) {
    LOG_ERROR("Failed to init IvfRabitqReformer, ret=%d", ret);
    IndexMemory::Instance()->remove(file_id);
    return ret;
  }

  auto memory_storage = IndexFactory::CreateStorage("MemoryReadStorage");
  ret = memory_storage->open(file_id, false);
  if (ret != 0) {
    LOG_ERROR("Failed to open memory storage, ret=%d", ret);
    IndexMemory::Instance()->remove(file_id);
    return ret;
  }

  ret = reformer_->load(memory_storage);
  if (ret != 0) {
    LOG_ERROR("Failed to load IvfRabitqReformer, ret=%d", ret);
    IndexMemory::Instance()->remove(file_id);
    return ret;
  }

  IndexMemory::Instance()->remove(file_id);

  stats_.set_trained_count(converter_->stats().trained_count());
  stats_.set_trained_costtime(timer.milli_seconds());

  LOG_INFO("IvfRabitqBuilder training completed: %u clusters, cost %zu ms",
           static_cast<uint32_t>(reformer_->num_clusters()),
           static_cast<size_t>(timer.milli_seconds()));

  state_ = TRAINED;
  return 0;
}

// --------------------------------------------------------------------------
// build — assign vectors to centroids, quantize, store results in memory
// --------------------------------------------------------------------------
int IvfRabitqBuilder::build(IndexThreads::Pointer /*threads*/,
                            IndexHolder::Pointer holder) {
  if (state_ != TRAINED) {
    LOG_ERROR("Train the index first before build");
    return IndexError_Runtime;
  }

  if (!reformer_ || !reformer_->loaded()) {
    LOG_ERROR("Reformer not loaded");
    return IndexError_NoReady;
  }

  ailego::ElapsedTime timer;

  IvfRabitqBuildData build_data;
  int ret = BuildIvfRabitqData(reformer_, holder, {}, &build_data);
  if (ret != 0) {
    return ret;
  }

  header_ = build_data.header;
  batch_data_buf_ = std::move(build_data.batch_data_buf);
  ex_data_buf_ = std::move(build_data.ex_data_buf);
  cluster_metas_ = std::move(build_data.cluster_metas);
  keys_buf_ = std::move(build_data.keys_buf);
  mapping_buf_ = std::move(build_data.mapping_buf);

  stats_.set_built_count(header_.total_vector_count);
  stats_.set_built_costtime(timer.milli_seconds());

  LOG_INFO(
      "IvfRabitqBuilder build completed: %zu vectors, %zu clusters, "
      "batch_data=%zu bytes, ex_data=%zu bytes, cost %zu ms",
      static_cast<size_t>(header_.total_vector_count),
      static_cast<size_t>(header_.cluster_count),
      static_cast<size_t>(header_.batch_data_size),
      static_cast<size_t>(header_.ex_data_size),
      static_cast<size_t>(timer.milli_seconds()));

  state_ = BUILT;
  return 0;
}

// --------------------------------------------------------------------------
// dump — write meta + reformer + segments to dumper
// --------------------------------------------------------------------------
int IvfRabitqBuilder::dump(const IndexDumper::Pointer &dumper) {
  if (state_ != BUILT) {
    LOG_ERROR("Build the index before dump");
    return IndexError_Runtime;
  }

  if (!dumper) {
    LOG_ERROR("Null dumper");
    return IndexError_InvalidArgument;
  }

  ailego::ElapsedTime timer;

  // Write meta
  int ret = IndexHelper::SerializeToDumper(meta_, dumper.get());
  if (ret != 0) {
    LOG_ERROR("Failed to serialize meta into dumper");
    return ret;
  }

  // Write reformer (centroids, rotator)
  if (reformer_ && reformer_->loaded()) {
    ret = reformer_->dump(dumper);
    if (ret != 0) {
      LOG_ERROR("Failed to dump reformer");
      return ret;
    }
  }

  // Write segments
  auto dump_segment = [&](const std::string &seg_id, const void *data,
                          size_t size) -> int {
    if (size == 0) {
      return 0;
    }
    uint32_t crc = ailego::Crc32c::Hash(data, size, 0);
    size_t align_size = (size + 0x1F) & (~0x1F);
    size_t padding_size = align_size - size;

    size_t written = dumper->write(data, size);
    if (written != size) {
      LOG_ERROR("Failed to write segment %s to dumper", seg_id.c_str());
      return IndexError_WriteData;
    }
    if (padding_size > 0) {
      std::string padding(padding_size, '\0');
      dumper->write(padding.data(), padding_size);
    }
    return dumper->append(seg_id, size, padding_size, crc);
  };

  ret = dump_segment(IVF_RABITQ_HEADER_SEG_ID, &header_, sizeof(header_));
  if (ret != 0) {
    return ret;
  }

  if (!batch_data_buf_.empty()) {
    ret = dump_segment(IVF_RABITQ_BATCH_DATA_SEG_ID, batch_data_buf_.data(),
                       batch_data_buf_.size());
    if (ret != 0) {
      return ret;
    }
  }

  if (!ex_data_buf_.empty()) {
    ret = dump_segment(IVF_RABITQ_EX_DATA_SEG_ID, ex_data_buf_.data(),
                       ex_data_buf_.size());
    if (ret != 0) {
      return ret;
    }
  }

  if (!cluster_metas_.empty()) {
    ret = dump_segment(IVF_RABITQ_CLUSTER_META_SEG_ID, cluster_metas_.data(),
                       cluster_metas_.size() * sizeof(IvfRabitqClusterMeta));
    if (ret != 0) {
      return ret;
    }
  }

  if (!keys_buf_.empty()) {
    ret = dump_segment(IVF_RABITQ_KEYS_SEG_ID, keys_buf_.data(),
                       keys_buf_.size() * sizeof(uint64_t));
    if (ret != 0) {
      return ret;
    }
  }

  if (!mapping_buf_.empty()) {
    ret = dump_segment(IVF_RABITQ_MAPPING_SEG_ID, mapping_buf_.data(),
                       mapping_buf_.size() * sizeof(uint32_t));
    if (ret != 0) {
      return ret;
    }
  }

  stats_.set_dumped_count(header_.total_vector_count);
  stats_.set_dumped_costtime(timer.milli_seconds());

  LOG_INFO("IvfRabitqBuilder dump completed, cost %zu ms",
           static_cast<size_t>(timer.milli_seconds()));

  return 0;
}

}  // namespace core
}  // namespace zvec
