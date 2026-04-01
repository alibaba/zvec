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
#include "hnsw_rabitq_builder.h"
#include <cstdlib>
#include <iostream>
#include <thread>
#include <ailego/pattern/defer.h>
#include <zvec/ailego/container/params.h>
#include <zvec/ailego/utility/string_helper.h>
#include <zvec/ailego/utility/time_helper.h>
#include "zvec/core/framework/index_error.h"
#include "zvec/core/framework/index_factory.h"
#include "zvec/core/framework/index_logger.h"
#include "zvec/core/framework/index_memory.h"
#include "zvec/core/framework/index_meta.h"
#include "zvec/core/framework/index_provider.h"
#include "hnsw_rabitq_algorithm.h"
#include "hnsw_rabitq_entity.h"
#include "hnsw_rabitq_params.h"
#include "rabitq_converter.h"
#include "rabitq_params.h"
#include "rabitq_reformer.h"

namespace zvec {
namespace core {

HnswRabitqBuilder::HnswRabitqBuilder() {}

int HnswRabitqBuilder::init(const IndexMeta &meta,
                            const ailego::Params &params) {
  LOG_INFO("Begin HnswRabitqBuilder::init");

  meta_ = meta;
  auto params_copy = params;
  meta_.set_builder("HnswRabitqBuilder", HnswRabitqEntity::kRevision,
                    std::move(params_copy));

  size_t memory_quota = 0UL;
  params.get(PARAM_HNSW_RABITQ_BUILDER_MEMORY_QUOTA, &memory_quota);
  params.get(PARAM_HNSW_RABITQ_BUILDER_THREAD_COUNT, &thread_cnt_);
  params.get(PARAM_HNSW_RABITQ_BUILDER_MIN_NEIGHBOR_COUNT, &min_neighbor_cnt_);
  params.get(PARAM_HNSW_RABITQ_BUILDER_EFCONSTRUCTION, &ef_construction_);
  params.get(PARAM_HNSW_RABITQ_BUILDER_CHECK_INTERVAL_SECS,
             &check_interval_secs_);

  params.get(PARAM_HNSW_RABITQ_BUILDER_MAX_NEIGHBOR_COUNT,
             &upper_max_neighbor_cnt_);
  float multiplier = HnswRabitqEntity::kDefaultL0MaxNeighborCntMultiplier;
  params.get(PARAM_HNSW_RABITQ_BUILDER_L0_MAX_NEIGHBOR_COUNT_MULTIPLIER,
             &multiplier);
  l0_max_neighbor_cnt_ = multiplier * upper_max_neighbor_cnt_;
  scaling_factor_ = upper_max_neighbor_cnt_;
  params.get(PARAM_HNSW_RABITQ_BUILDER_SCALING_FACTOR, &scaling_factor_);

  multiplier = HnswRabitqEntity::kDefaultNeighborPruneMultiplier;
  params.get(PARAM_HNSW_RABITQ_BUILDER_NEIGHBOR_PRUNE_MULTIPLIER, &multiplier);
  size_t prune_cnt = multiplier * upper_max_neighbor_cnt_;

  if (ef_construction_ == 0) {
    ef_construction_ = HnswRabitqEntity::kDefaultEfConstruction;
  }
  if (upper_max_neighbor_cnt_ == 0) {
    upper_max_neighbor_cnt_ = HnswRabitqEntity::kDefaultUpperMaxNeighborCnt;
  }
  if (upper_max_neighbor_cnt_ > kMaxNeighborCnt) {
    LOG_ERROR("[%s] must be in range (0,%d]",
              PARAM_HNSW_RABITQ_BUILDER_MAX_NEIGHBOR_COUNT.c_str(),
              kMaxNeighborCnt);
    return IndexError_InvalidArgument;
  }
  if (min_neighbor_cnt_ > upper_max_neighbor_cnt_) {
    LOG_ERROR("[%s]-[%d] must be <= [%s]-[%d]",
              PARAM_HNSW_RABITQ_BUILDER_MIN_NEIGHBOR_COUNT.c_str(),
              min_neighbor_cnt_,
              PARAM_HNSW_RABITQ_BUILDER_MAX_NEIGHBOR_COUNT.c_str(),
              upper_max_neighbor_cnt_);
    return IndexError_InvalidArgument;
  }
  if (l0_max_neighbor_cnt_ == 0) {
    l0_max_neighbor_cnt_ = HnswRabitqEntity::kDefaultUpperMaxNeighborCnt;
  }
  if (l0_max_neighbor_cnt_ > HnswRabitqEntity::kMaxNeighborCnt) {
    LOG_ERROR("L0MaxNeighborCnt must be in range (0,%d)",
              HnswRabitqEntity::kMaxNeighborCnt);
    return IndexError_InvalidArgument;
  }
  if (scaling_factor_ == 0U) {
    scaling_factor_ = HnswRabitqEntity::kDefaultScalingFactor;
  }
  if (scaling_factor_ < 5 || scaling_factor_ > 1000) {
    LOG_ERROR("[%s] must be in range [5,1000]",
              PARAM_HNSW_RABITQ_BUILDER_SCALING_FACTOR.c_str());
    return IndexError_InvalidArgument;
  }
  if (thread_cnt_ == 0) {
    thread_cnt_ = std::thread::hardware_concurrency();
  }
  if (thread_cnt_ > std::thread::hardware_concurrency()) {
    LOG_WARN("[%s] greater than cpu cores %zu",
             PARAM_HNSW_RABITQ_BUILDER_THREAD_COUNT.c_str(),
             static_cast<size_t>(std::thread::hardware_concurrency()));
  }
  if (prune_cnt == 0UL) {
    prune_cnt = upper_max_neighbor_cnt_;
  }

  metric_ = IndexFactory::CreateMetric(meta_.metric_name());
  if (!metric_) {
    LOG_ERROR("CreateMetric failed, name: %s", meta_.metric_name().c_str());
    return IndexError_NoExist;
  }
  int ret = metric_->init(meta_, meta_.metric_params());
  if (ret != 0) {
    LOG_ERROR("IndexMetric init failed, ret=%d", ret);
    return ret;
  }

  uint32_t total_bits = 0;
  params.get(PARAM_RABITQ_TOTAL_BITS, &total_bits);
  if (total_bits == 0) {
    total_bits = kDefaultRabitqTotalBits;
  }
  if (total_bits < 1 || total_bits > 9) {
    LOG_ERROR("Invalid total_bits: %zu, must be in [1, 9]", (size_t)total_bits);
    return IndexError_InvalidArgument;
  }
  uint8_t ex_bits = total_bits - 1;
  entity_.set_ex_bits(ex_bits);

  uint32_t dimension = 0;
  params.get(PARAM_HNSW_RABITQ_GENERAL_DIMENSION, &dimension);
  if (dimension == 0) {
    LOG_ERROR("%s not set", PARAM_HNSW_RABITQ_GENERAL_DIMENSION.c_str());
    return IndexError_InvalidArgument;
  }
  if (dimension < kMinRabitqDimSize || dimension > kMaxRabitqDimSize) {
    LOG_ERROR("Invalid dimension: %u, must be in [%d, %d]", dimension,
              kMinRabitqDimSize, kMaxRabitqDimSize);
    return IndexError_InvalidArgument;
  }
  entity_.update_rabitq_params_and_vector_size(dimension);

  entity_.set_ef_construction(ef_construction_);
  entity_.set_l0_neighbor_cnt(l0_max_neighbor_cnt_);
  entity_.set_min_neighbor_cnt(min_neighbor_cnt_);
  entity_.set_upper_neighbor_cnt(upper_max_neighbor_cnt_);
  entity_.set_scaling_factor(scaling_factor_);
  entity_.set_memory_quota(memory_quota);
  entity_.set_prune_cnt(prune_cnt);

  ret = entity_.init();
  if (ret != 0) {
    return ret;
  }

  alg_ = HnswRabitqAlgorithm::UPointer(new HnswRabitqAlgorithm(entity_));

  ret = alg_->init();
  if (ret != 0) {
    return ret;
  }

  // Create and initialize RaBitQ converter
  converter_ = std::make_shared<RabitqConverter>();

  IndexMeta converter_meta = meta_;
  converter_meta.set_dimension(dimension);
  ret = converter_->init(converter_meta, params);
  if (ret != 0) {
    LOG_ERROR("Failed to initialize RabitqConverter: %d", ret);
    return ret;
  }

  state_ = BUILD_STATE_INITED;
  LOG_INFO(
      "End HnswRabitqBuilder::init, params: rawVectorSize=%u vectorSize=%zu "
      "efConstruction=%u "
      "l0NeighborCnt=%u upperNeighborCnt=%u scalingFactor=%u "
      "memoryQuota=%zu neighborPruneCnt=%zu metricName=%s ",
      meta_.element_size(), entity_.vector_size(), ef_construction_,
      l0_max_neighbor_cnt_, upper_max_neighbor_cnt_, scaling_factor_,
      memory_quota, prune_cnt, meta_.metric_name().c_str());

  return 0;
}

int HnswRabitqBuilder::cleanup(void) {
  LOG_INFO("Begin HnswRabitqBuilder::cleanup");

  l0_max_neighbor_cnt_ = HnswRabitqEntity::kDefaultL0MaxNeighborCnt;
  min_neighbor_cnt_ = 0;
  upper_max_neighbor_cnt_ = HnswRabitqEntity::kDefaultUpperMaxNeighborCnt;
  ef_construction_ = HnswRabitqEntity::kDefaultEfConstruction;
  scaling_factor_ = HnswRabitqEntity::kDefaultScalingFactor;
  check_interval_secs_ = kDefaultLogIntervalSecs;
  errcode_ = 0;
  error_ = false;
  entity_.cleanup();
  if (alg_) {
    alg_->cleanup();
  }
  meta_.clear();
  metric_.reset();
  stats_.clear_attributes();
  stats_.set_trained_count(0UL);
  stats_.set_built_count(0UL);
  stats_.set_dumped_count(0UL);
  stats_.set_discarded_count(0UL);
  stats_.set_trained_costtime(0UL);
  stats_.set_built_costtime(0UL);
  stats_.set_dumped_costtime(0UL);
  state_ = BUILD_STATE_INIT;

  LOG_INFO("End HnswRabitqBuilder::cleanup");

  return 0;
}

int HnswRabitqBuilder::train(IndexThreads::Pointer,
                             IndexHolder::Pointer holder) {
  if (state_ != BUILD_STATE_INITED) {
    LOG_ERROR("Init the builder before HnswRabitqBuilder::train");
    return IndexError_NoReady;
  }

  if (!holder) {
    LOG_ERROR("Input holder is nullptr while training index");
    return IndexError_InvalidArgument;
  }
  if (!holder->is_matched(meta_)) {
    LOG_ERROR("Input holder doesn't match index meta while training index");
    return IndexError_Mismatch;
  }
  LOG_INFO("Begin HnswRabitqBuilder::train");
  size_t trained_cost_time = 0;
  size_t trained_count = 0;

  int ret = train_converter_and_load_reformer(holder);
  if (ret != 0) {
    return ret;
  }

  if (metric_->support_train()) {
    auto start_time = ailego::Monotime::MilliSeconds();
    auto iter = holder->create_iterator();
    if (!iter) {
      LOG_ERROR("Create iterator for holder failed");
      return IndexError_Runtime;
    }
    while (iter->is_valid()) {
      ret = metric_->train(iter->data(), meta_.dimension());
      if (ailego_unlikely(ret != 0)) {
        LOG_ERROR("Hnsw build measure train failed, ret=%d", ret);
        return ret;
      }
      iter->next();
      ++trained_count;
    }
    trained_cost_time = ailego::Monotime::MilliSeconds() - start_time;
  }
  stats_.set_trained_count(trained_count);
  stats_.set_trained_costtime(trained_cost_time);
  state_ = BUILD_STATE_TRAINED;

  LOG_INFO("End HnswRabitqBuilder::train");

  return 0;
}

int HnswRabitqBuilder::train_converter_and_load_reformer(
    IndexHolder::Pointer holder) {
  // Train converter (KMeans clustering)
  int ret = converter_->train(holder);
  if (ret != 0) {
    LOG_ERROR("Failed to train RabitqConverter: %d", ret);
    return ret;
  }
  auto memory_dumper = IndexFactory::CreateDumper("MemoryDumper");
  memory_dumper->init(ailego::Params());
  std::string file_id = ailego::StringHelper::Concat(
      "rabitq_converter_", ailego::Monotime::MilliSeconds(), rand());
  ret = memory_dumper->create(file_id);
  if (ret != 0) {
    LOG_ERROR("Failed to create memory dumper: %d", ret);
    return ret;
  }
  // Release memory
  AILEGO_DEFER([&file_id]() { IndexMemory::Instance()->remove(file_id); });
  ret = converter_->dump(memory_dumper);
  if (ret != 0) {
    LOG_ERROR("Failed to dump RabitqConverter: %d", ret);
    return ret;
  }
  ret = memory_dumper->close();
  if (ret != 0) {
    LOG_ERROR("Failed to close memory dumper: %d", ret);
    return ret;
  }

  reformer_ = std::make_shared<RabitqReformer>();
  ailego::Params reformer_params;
  reformer_params.set(PARAM_RABITQ_METRIC_NAME, meta_.metric_name());
  ret = reformer_->init(reformer_params);
  if (ret != 0) {
    LOG_ERROR("Failed to initialize RabitqReformer: %d", ret);
    return ret;
  }
  auto memory_storage = IndexFactory::CreateStorage("MemoryReadStorage");
  ret = memory_storage->open(file_id, false);
  if (ret != 0) {
    LOG_ERROR("Failed to open memory storage: %d", ret);
    return ret;
  }
  ret = reformer_->load(memory_storage);
  if (ret != 0) {
    LOG_ERROR("Failed to load RabitqReformer: %d", ret);
    return ret;
  }
  return 0;
}

int HnswRabitqBuilder::train(const IndexTrainer::Pointer & /*trainer*/) {
  if (state_ != BUILD_STATE_INITED) {
    LOG_ERROR("Init the builder before HnswRabitqBuilder::train");
    return IndexError_NoReady;
  }

  LOG_INFO("Begin HnswRabitqBuilder::train by trainer");

  stats_.set_trained_count(0UL);
  stats_.set_trained_costtime(0UL);
  state_ = BUILD_STATE_TRAINED;

  LOG_INFO("End HnswRabitqBuilder::train by trainer");

  return 0;
}

int HnswRabitqBuilder::build(IndexThreads::Pointer threads,
                             IndexHolder::Pointer holder) {
  if (state_ != BUILD_STATE_TRAINED) {
    LOG_ERROR("Train the index before HnswRabitqBuilder::build");
    return IndexError_NoReady;
  }

  if (!holder) {
    LOG_ERROR("Input holder is nullptr while building index");
    return IndexError_InvalidArgument;
  }
  if (!holder->is_matched(meta_)) {
    LOG_ERROR("Input holder doesn't match index meta while building index");
    return IndexError_Mismatch;
  }
  IndexProvider::Pointer provider =
      std::dynamic_pointer_cast<IndexProvider>(holder);
  if (!provider) {
    LOG_ERROR("Rabitq builder expect IndexProvider");
    return IndexError_InvalidArgument;
  }

  if (!threads) {
    threads = std::make_shared<SingleQueueIndexThreads>(thread_cnt_, false);
  }

  auto start_time = ailego::Monotime::MilliSeconds();
  LOG_INFO("Begin HnswRabitqBuilder::build");

  if (holder->count() != static_cast<size_t>(-1)) {
    LOG_DEBUG("HnswRabitqBuilder holder documents count %lu", holder->count());
    int ret = entity_.reserve_space(holder->count());
    if (ret != 0) {
      LOG_ERROR("HnswBuilde reserver space failed");
      return ret;
    }
  }
  auto iter = holder->create_iterator();
  if (!iter) {
    LOG_ERROR("Create iterator for holder failed");
    return IndexError_Runtime;
  }
  int ret;
  error_ = false;
  IndexQueryMeta ometa;
  ometa.set_meta(holder->data_type(), holder->dimension());
  while (iter->is_valid()) {
    const void *vec = iter->data();
    // quantize vector
    std::string converted_vector;
    IndexQueryMeta converted_meta;
    ret = reformer_->convert(vec, ometa, &converted_vector, &converted_meta);
    if (ret != 0) {
      LOG_ERROR("Rabitq hnsw convert failed, ret=%d", ret);
      return ret;
    }


    level_t level = alg_->get_random_level();
    node_id_t id;

    if (converted_vector.size() != entity_.vector_size()) {
      LOG_ERROR(
          "Converted vector size %zu is not equal to entity vector size %zu",
          converted_vector.size(), entity_.vector_size());
      return IndexError_InvalidArgument;
    }
    ret = entity_.add_vector(level, iter->key(), converted_vector.data(), &id);
    if (ailego_unlikely(ret != 0)) {
      return ret;
    }
    iter->next();
  }

  LOG_INFO("Finished save vector, start build graph...");

  auto task_group = threads->make_group();
  if (!task_group) {
    LOG_ERROR("Failed to create task group");
    return IndexError_Runtime;
  }

  std::atomic<node_id_t> finished{0};
  for (size_t i = 0; i < threads->count(); ++i) {
    task_group->submit(ailego::Closure ::New(this, &HnswRabitqBuilder::do_build,
                                             i, threads->count(), provider,
                                             &finished));
  }

  while (!task_group->is_finished()) {
    std::unique_lock<std::mutex> lk(mutex_);
    cond_.wait_until(lk, std::chrono::system_clock::now() +
                             std::chrono::seconds(check_interval_secs_));
    if (error_.load(std::memory_order_acquire)) {
      LOG_ERROR("Failed to build index while waiting finish");
      return errcode_;
    }
    LOG_INFO("Built cnt %zu, finished percent %.3f%%",
             static_cast<size_t>(finished.load()),
             finished.load() * 100.0f / entity_.doc_cnt());
  }
  if (error_.load(std::memory_order_acquire)) {
    LOG_ERROR("Failed to build index while waiting finish");
    return errcode_;
  }
  task_group->wait_finish();

  stats_.set_built_count(finished.load());
  stats_.set_built_costtime(ailego::Monotime::MilliSeconds() - start_time);

  state_ = BUILD_STATE_BUILT;
  LOG_INFO("End HnswRabitqBuilder::build with RaBitQ quantization");
  return 0;
}

void HnswRabitqBuilder::do_build(node_id_t idx, size_t step_size,
                                 IndexProvider::Pointer provider,
                                 std::atomic<node_id_t> *finished) {
  AILEGO_DEFER([&]() {
    std::lock_guard<std::mutex> latch(mutex_);
    cond_.notify_one();
  });
  HnswRabitqContext *ctx = new (std::nothrow) HnswRabitqContext(
      meta_.dimension(), metric_,
      std::shared_ptr<HnswRabitqEntity>(&entity_, [](HnswRabitqEntity *) {}));
  if (ailego_unlikely(ctx == nullptr)) {
    if (!error_.exchange(true)) {
      LOG_ERROR("Failed to create context");
      errcode_ = IndexError_NoMemory;
    }
    return;
  }
  HnswRabitqContext::Pointer auto_ptr(ctx);
  ctx->set_provider(std::move(provider));
  ctx->set_max_scan_num(entity_.doc_cnt());
  int ret = ctx->init(HnswRabitqContext::kBuilderContext);
  if (ret != 0) {
    if (!error_.exchange(true)) {
      LOG_ERROR("Failed to init context");
      errcode_ = IndexError_Runtime;
    }
    return;
  }

  for (node_id_t id = idx; id < entity_.doc_cnt(); id += step_size) {
    ctx->reset_query(ctx->dist_calculator().get_vector(id));
    ret = alg_->add_node(id, entity_.get_level(id), ctx);
    if (ailego_unlikely(ret != 0)) {
      if (!error_.exchange(true)) {
        LOG_ERROR("Hnsw graph add node failed");
        errcode_ = ret;
      }
      return;
    }
    ctx->clear();
    (*finished)++;
  }
}

int HnswRabitqBuilder::dump(const IndexDumper::Pointer &dumper) {
  if (state_ != BUILD_STATE_BUILT) {
    LOG_INFO("Build the index before HnswRabitqBuilder::dump");
    return IndexError_NoReady;
  }

  LOG_INFO("Begin HnswRabitqBuilder::dump");

  meta_.set_searcher("HnswRabitqSearcher", HnswRabitqEntity::kRevision,
                     ailego::Params());
  auto start_time = ailego::Monotime::MilliSeconds();

  int ret = IndexHelper::SerializeToDumper(meta_, dumper.get());
  if (ret != 0) {
    LOG_ERROR("Failed to serialize meta into dumper.");
    return ret;
  }

  // Dump RaBitQ centroids first
  if (converter_) {
    ret = converter_->dump(dumper);
    if (ret != 0) {
      LOG_ERROR("Failed to dump RabitqConverter: %d", ret);
      return ret;
    }
    LOG_INFO("RaBitQ centroids dumped: %zu bytes, cost %zu ms",
             converter_->stats().dumped_size(),
             static_cast<size_t>(converter_->stats().dumped_costtime()));
  }

  ret = entity_.dump(dumper);
  if (ret != 0) {
    LOG_ERROR("HnswRabitqBuilder dump index failed");
    return ret;
  }

  stats_.set_dumped_count(entity_.doc_cnt());
  stats_.set_dumped_costtime(ailego::Monotime::MilliSeconds() - start_time);

  LOG_INFO("End HnswRabitqBuilder::dump");
  return 0;
}


}  // namespace core
}  // namespace zvec
