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
#include "hnsw_builder.h"
#include <iostream>
#include <thread>
#include <ailego/pattern/defer.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_logger.h>
#include "hnsw_algorithm.h"
#include "hnsw_params.h"

namespace zvec {
namespace core {

HnswBuilder::HnswBuilder() = default;

int HnswBuilder::init(const IndexMeta &meta, const ailego::Params &params) {
  LOG_INFO("Begin HnswBuilder::init");

  meta_ = meta;
  auto params_copy = params;
  meta_.set_builder("HnswBuilder", HnswEntity::kRevision,
                    std::move(params_copy));

  size_t memory_quota = 0UL;
  params.get(PARAM_HNSW_BUILDER_MEMORY_QUOTA, &memory_quota);
  params.get(PARAM_HNSW_BUILDER_THREAD_COUNT, &thread_cnt_);
  params.get(PARAM_HNSW_BUILDER_MIN_NEIGHBOR_COUNT, &min_neighbor_cnt_);
  params.get(PARAM_HNSW_BUILDER_EFCONSTRUCTION, &ef_construction_);
  params.get(PARAM_HNSW_BUILDER_CHECK_INTERVAL_SECS, &check_interval_secs_);

  params.get(PARAM_HNSW_BUILDER_MAX_NEIGHBOR_COUNT, &upper_max_neighbor_cnt_);
  float multiplier = HnswEntity::kDefaultL0MaxNeighborCntMultiplier;
  params.get(PARAM_HNSW_BUILDER_L0_MAX_NEIGHBOR_COUNT_MULTIPLIER, &multiplier);
  l0_max_neighbor_cnt_ = multiplier * upper_max_neighbor_cnt_;
  scaling_factor_ = upper_max_neighbor_cnt_;
  params.get(PARAM_HNSW_BUILDER_SCALING_FACTOR, &scaling_factor_);

  multiplier = HnswEntity::kDefaultNeighborPruneMultiplier;
  params.get(PARAM_HNSW_BUILDER_NEIGHBOR_PRUNE_MULTIPLIER, &multiplier);
  size_t prune_cnt = multiplier * upper_max_neighbor_cnt_;

  if (ef_construction_ == 0) {
    ef_construction_ = HnswEntity::kDefaultEfConstruction;
  }
  if (upper_max_neighbor_cnt_ == 0) {
    upper_max_neighbor_cnt_ = HnswEntity::kDefaultUpperMaxNeighborCnt;
  }
  if (upper_max_neighbor_cnt_ > kMaxNeighborCnt) {
    LOG_ERROR("[%s] must be in range (0,%d]",
              PARAM_HNSW_BUILDER_MAX_NEIGHBOR_COUNT.c_str(), kMaxNeighborCnt);
    return IndexError_InvalidArgument;
  }
  if (min_neighbor_cnt_ > upper_max_neighbor_cnt_) {
    LOG_ERROR("[%s]-[%d] must be <= [%s]-[%d]",
              PARAM_HNSW_BUILDER_MIN_NEIGHBOR_COUNT.c_str(), min_neighbor_cnt_,
              PARAM_HNSW_BUILDER_MAX_NEIGHBOR_COUNT.c_str(),
              upper_max_neighbor_cnt_);
    return IndexError_InvalidArgument;
  }
  if (l0_max_neighbor_cnt_ == 0) {
    l0_max_neighbor_cnt_ = HnswEntity::kDefaultUpperMaxNeighborCnt;
  }
  if (l0_max_neighbor_cnt_ > HnswEntity::kMaxNeighborCnt) {
    LOG_ERROR("L0MaxNeighborCnt must be in range (0,%d)",
              HnswEntity::kMaxNeighborCnt);
    return IndexError_InvalidArgument;
  }
  if (scaling_factor_ == 0U) {
    scaling_factor_ = HnswEntity::kDefaultScalingFactor;
  }
  if (scaling_factor_ < 5 || scaling_factor_ > 1000) {
    LOG_ERROR("[%s] must be in range [5,1000]",
              PARAM_HNSW_BUILDER_SCALING_FACTOR.c_str());
    return IndexError_InvalidArgument;
  }
  if (thread_cnt_ == 0) {
    thread_cnt_ = std::thread::hardware_concurrency();
  }
  if (thread_cnt_ > std::thread::hardware_concurrency()) {
    LOG_WARN("[%s] greater than cpu cores %u",
             PARAM_HNSW_BUILDER_THREAD_COUNT.c_str(),
             std::thread::hardware_concurrency());
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

  entity_.set_vector_size(meta_.element_size());

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

  alg_ = HnswAlgorithm::UPointer(new HnswAlgorithm(entity_));

  ret = alg_->init();
  if (ret != 0) {
    return ret;
  }

  state_ = BUILD_STATE_INITED;
  LOG_INFO(
      "End HnswBuilder::init, params: vectorSize=%u efConstruction=%u "
      "l0NeighborCnt=%u upperNeighborCnt=%u scalingFactor=%u "
      "memoryQuota=%zu neighborPruneCnt=%zu metricName=%s ",
      meta_.element_size(), ef_construction_, l0_max_neighbor_cnt_,
      upper_max_neighbor_cnt_, scaling_factor_, memory_quota, prune_cnt,
      meta_.metric_name().c_str());

  return 0;
}

int HnswBuilder::cleanup(void) {
  LOG_INFO("Begin HnswBuilder::cleanup");

  l0_max_neighbor_cnt_ = HnswEntity::kDefaultL0MaxNeighborCnt;
  min_neighbor_cnt_ = 0;
  upper_max_neighbor_cnt_ = HnswEntity::kDefaultUpperMaxNeighborCnt;
  ef_construction_ = HnswEntity::kDefaultEfConstruction;
  scaling_factor_ = HnswEntity::kDefaultScalingFactor;
  check_interval_secs_ = kDefaultLogIntervalSecs;
  errcode_ = 0;
  error_ = false;
  entity_.cleanup();
  alg_->cleanup();
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

  LOG_INFO("End HnswBuilder::cleanup");

  return 0;
}

int HnswBuilder::train(IndexThreads::Pointer, IndexHolder::Pointer holder) {
  if (state_ != BUILD_STATE_INITED) {
    LOG_ERROR("Init the builder before HnswBuilder::train");
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
  LOG_INFO("Begin HnswBuilder::train");
  size_t trained_cost_time = 0;
  size_t trained_count = 0;

  if (metric_->support_train()) {
    auto start_time = ailego::Monotime::MilliSeconds();
    auto iter = holder->create_iterator();
    if (!iter) {
      LOG_ERROR("Create iterator for holder failed");
      return IndexError_Runtime;
    }
    while (iter->is_valid()) {
      int ret = metric_->train(iter->data(), meta_.dimension());
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

  LOG_INFO("End HnswBuilder::train");

  return 0;
}

int HnswBuilder::train(const IndexTrainer::Pointer & /*trainer*/) {
  if (state_ != BUILD_STATE_INITED) {
    LOG_ERROR("Init the builder before HnswBuilder::train");
    return IndexError_NoReady;
  }

  LOG_INFO("Begin HnswBuilder::train by trainer");

  stats_.set_trained_count(0UL);
  stats_.set_trained_costtime(0UL);
  state_ = BUILD_STATE_TRAINED;

  LOG_INFO("End HnswBuilder::train by trainer");

  return 0;
}

int HnswBuilder::build(IndexThreads::Pointer threads,
                       IndexHolder::Pointer holder) {
  if (state_ != BUILD_STATE_TRAINED) {
    LOG_ERROR("Train the index before HnswBuilder::build");
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
  if (!threads) {
    threads = std::make_shared<SingleQueueIndexThreads>(thread_cnt_, false);
    if (!threads) {
      return IndexError_NoMemory;
    }
  }

  auto start_time = ailego::Monotime::MilliSeconds();
  LOG_INFO("Begin HnswBuilder::build");

  if (holder->count() != static_cast<size_t>(-1)) {
    LOG_DEBUG("HnswBuilder holder documents count %lu", holder->count());
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
  while (iter->is_valid()) {
    level_t level = alg_->get_random_level();
    node_id_t id;

    const void *vec = iter->data();
    ret = entity_.add_vector(level, iter->key(), vec, &id);
    if (ailego_unlikely(ret != 0)) {
      return ret;
    }
    iter->next();
  }
  // Holder is not needed, cleanup it.
  holder.reset();

  LOG_INFO("Finished save vector, start build graph...");

  auto task_group = threads->make_group();
  if (!task_group) {
    LOG_ERROR("Failed to create task group");
    return IndexError_Runtime;
  }

  std::atomic<node_id_t> finished{0};
  for (size_t i = 0; i < threads->count(); ++i) {
    task_group->submit(ailego::Closure ::New(this, &HnswBuilder::do_build, i,
                                             threads->count(), &finished));
  }

  while (!task_group->is_finished()) {
    std::unique_lock<std::mutex> lk(mutex_);
    cond_.wait_until(lk, std::chrono::system_clock::now() +
                             std::chrono::seconds(check_interval_secs_));
    if (error_.load(std::memory_order_acquire)) {
      LOG_ERROR("Failed to build index while waiting finish");
      return errcode_;
    }
    LOG_INFO("Built cnt %u, finished percent %.3f%%", finished.load(),
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

  LOG_INFO("End HnswBuilder::build");
  return 0;
}

void HnswBuilder::do_build(node_id_t idx, size_t step_size,
                           std::atomic<node_id_t> *finished) {
  AILEGO_DEFER([&]() {
    std::lock_guard<std::mutex> latch(mutex_);
    cond_.notify_one();
  });
  HnswContext *ctx = new (std::nothrow)
      HnswContext(meta_.dimension(), metric_,
                  std::shared_ptr<HnswEntity>(&entity_, [](HnswEntity *) {}));
  if (ailego_unlikely(ctx == nullptr)) {
    if (!error_.exchange(true)) {
      LOG_ERROR("Failed to create context");
      errcode_ = IndexError_NoMemory;
    }
    return;
  }
  HnswContext::Pointer auto_ptr(ctx);
  ctx->set_max_scan_num(entity_.doc_cnt());
  int ret = ctx->init(HnswContext::kBuilderContext);
  if (ret != 0) {
    if (!error_.exchange(true)) {
      LOG_ERROR("Failed to init context");
      errcode_ = IndexError_Runtime;
    }
    return;
  }

  IndexQueryMeta qmeta(meta_.data_type(), meta_.dimension());
  for (node_id_t id = idx; id < entity_.doc_cnt(); id += step_size) {
    ctx->reset_query(entity_.get_vector(id));
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

int HnswBuilder::dump(const IndexDumper::Pointer &dumper) {
  if (state_ != BUILD_STATE_BUILT) {
    LOG_INFO("Build the index before HnswBuilder::dump");
    return IndexError_NoReady;
  }

  LOG_INFO("Begin HnswBuilder::dump");

  meta_.set_searcher("HnswSearcher", HnswEntity::kRevision, ailego::Params());
  auto start_time = ailego::Monotime::MilliSeconds();

  int ret = IndexHelper::SerializeToDumper(meta_, dumper.get());
  if (ret != 0) {
    LOG_ERROR("Failed to serialize meta into dumper.");
    return ret;
  }

  ret = entity_.dump(dumper);
  if (ret != 0) {
    LOG_ERROR("HnswBuilder dump index failed");
    return ret;
  }

  stats_.set_dumped_count(entity_.doc_cnt());
  stats_.set_dumped_costtime(ailego::Monotime::MilliSeconds() - start_time);

  LOG_INFO("EndHnswBuilder::dump");
  return 0;
}

INDEX_FACTORY_REGISTER_BUILDER(HnswBuilder);

}  // namespace core
}  // namespace zvec
