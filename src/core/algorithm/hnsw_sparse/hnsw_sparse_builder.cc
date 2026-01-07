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
#include "hnsw_sparse_builder.h"
#include <iostream>
#include <thread>
#include <ailego/pattern/defer.h>
#include "framework/index_error.h"
#include "framework/index_factory.h"
#include "framework/index_logger.h"
#include "hnsw_sparse_algorithm.h"
#include "hnsw_sparse_params.h"

namespace zvec {
namespace core {

HnswSparseBuilder::HnswSparseBuilder() {}

int HnswSparseBuilder::init(const IndexMeta &meta,
                            const ailego::Params &params) {
  LOG_INFO("Begin HnswSparseBuilder::init");

  meta_ = meta;
  auto params_copy = params;
  meta_.set_builder("HnswSparseBuilder", HnswSparseEntity::kRevision,
                    std::move(params_copy));

  size_t memory_quota = 0UL;
  params.get(PARAM_HNSW_SPARSE_BUILDER_MEMORY_QUOTA, &memory_quota);
  params.get(PARAM_HNSW_SPARSE_BUILDER_THREAD_COUNT, &thread_cnt_);
  params.get(PARAM_HNSW_SPARSE_BUILDER_EFCONSTRUCTION, &ef_construction_);
  params.get(PARAM_HNSW_SPARSE_BUILDER_SCALING_FACTOR, &scaling_factor_);
  params.get(PARAM_HNSW_SPARSE_BUILDER_CHECK_INTERVAL_SECS,
             &check_interval_secs_);

  params.get(PARAM_HNSW_SPARSE_BUILDER_MAX_NEIGHBOR_COUNT,
             &upper_max_neighbor_cnt_);
  float multiplier = HnswSparseEntity::kDefaultL0MaxNeighborCntMultiplier;
  params.get(PARAM_HNSW_SPARSE_BUILDER_L0_MAX_NEIGHBOR_COUNT_MULTIPLIER,
             &multiplier);
  l0_max_neighbor_cnt_ = multiplier * upper_max_neighbor_cnt_;

  multiplier = HnswSparseEntity::kDefaultNeighborPruneMultiplier;
  params.get(PARAM_HNSW_SPARSE_BUILDER_NEIGHBOR_PRUNE_MULTIPLIER, &multiplier);
  size_t prune_cnt = multiplier * upper_max_neighbor_cnt_;

  if (ef_construction_ == 0) {
    ef_construction_ = HnswSparseEntity::kDefaultEfConstruction;
  }
  if (upper_max_neighbor_cnt_ == 0) {
    upper_max_neighbor_cnt_ = HnswSparseEntity::kDefaultUpperMaxNeighborCnt;
  }
  if (upper_max_neighbor_cnt_ > kMaxNeighborCnt) {
    LOG_ERROR("[%s] must be in range (0,%d]",
              PARAM_HNSW_SPARSE_BUILDER_MAX_NEIGHBOR_COUNT.c_str(),
              kMaxNeighborCnt);
    return IndexError_InvalidArgument;
  }
  if (min_neighbor_cnt_ > upper_max_neighbor_cnt_) {
    LOG_ERROR("[%s]-[%d] must be <= [%s]-[%d]",
              PARAM_HNSW_SPARSE_BUILDER_MIN_NEIGHBOR_COUNT.c_str(),
              min_neighbor_cnt_,
              PARAM_HNSW_SPARSE_BUILDER_MAX_NEIGHBOR_COUNT.c_str(),
              upper_max_neighbor_cnt_);
    return IndexError_InvalidArgument;
  }
  if (l0_max_neighbor_cnt_ == 0) {
    l0_max_neighbor_cnt_ = HnswSparseEntity::kDefaultUpperMaxNeighborCnt;
  }
  if (l0_max_neighbor_cnt_ > HnswSparseEntity::kMaxNeighborCnt) {
    LOG_ERROR("L0MaxNeighborCnt must be in range (0,%d)",
              HnswSparseEntity::kMaxNeighborCnt);
    return IndexError_InvalidArgument;
  }
  if (scaling_factor_ == 0U) {
    scaling_factor_ = HnswSparseEntity::kDefaultScalingFactor;
  }
  if (scaling_factor_ < 5 || scaling_factor_ > 1000) {
    LOG_ERROR("[%s] must be in range [5,1000]",
              PARAM_HNSW_SPARSE_BUILDER_SCALING_FACTOR.c_str());
    return IndexError_InvalidArgument;
  }
  if (thread_cnt_ == 0) {
    thread_cnt_ = std::thread::hardware_concurrency();
  }
  if (thread_cnt_ > std::thread::hardware_concurrency()) {
    LOG_WARN("[%s] greater than cpu cores %u",
             PARAM_HNSW_SPARSE_BUILDER_THREAD_COUNT.c_str(),
             std::thread::hardware_concurrency());
  }
  if (prune_cnt == 0UL) {
    prune_cnt = upper_max_neighbor_cnt_;
  }

  metric_ = IndexFactory::CreateMetric(meta_.metric_name());
  if (!metric_) {
    LOG_ERROR("CreateMeasure failed, name: %s", meta_.metric_name().c_str());
    return IndexError_NoExist;
  }
  int ret = metric_->init(meta_, meta_.metric_params());
  if (ret != 0) {
    LOG_ERROR("IndexMeasure init failed, ret=%d", ret);
    return ret;
  }

  entity_.set_ef_construction(ef_construction_);
  entity_.set_l0_neighbor_cnt(l0_max_neighbor_cnt_);
  entity_.set_min_neighbor_cnt(min_neighbor_cnt_);
  entity_.set_upper_neighbor_cnt(upper_max_neighbor_cnt_);
  entity_.set_scaling_factor(scaling_factor_);
  entity_.set_memory_quota(memory_quota);
  entity_.set_prune_cnt(prune_cnt);

  entity_.set_sparse_meta_size(HnswSparseEntity::kSparseMetaSize);
  entity_.set_sparse_unit_size(meta.unit_size());

  ret = entity_.init();
  if (ret != 0) {
    return ret;
  }

  alg_ = HnswSparseAlgorithm::UPointer(new HnswSparseAlgorithm(entity_));

  ret = alg_->init();
  if (ret != 0) {
    return ret;
  }

  state_ = BUILD_STATE_INITED;
  LOG_INFO(
      "End HnswSparseBuilder::init, params: efConstruction=%u "
      "l0NeighborCnt=%u upperNeighborCnt=%u scalingFactor=%u "
      "memoryQuota=%zu neighborPruneCnt=%zu measureName=%s ",
      ef_construction_, l0_max_neighbor_cnt_, upper_max_neighbor_cnt_,
      scaling_factor_, memory_quota, prune_cnt, meta_.metric_name().c_str());

  return 0;
}

int HnswSparseBuilder::cleanup(void) {
  LOG_INFO("Begin HnswSparseBuilder::cleanup");

  l0_max_neighbor_cnt_ = HnswSparseEntity::kDefaultL0MaxNeighborCnt;
  min_neighbor_cnt_ = 0;
  upper_max_neighbor_cnt_ = HnswSparseEntity::kDefaultUpperMaxNeighborCnt;
  ef_construction_ = HnswSparseEntity::kDefaultEfConstruction;
  scaling_factor_ = HnswSparseEntity::kDefaultScalingFactor;
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

  LOG_INFO("End HnswSparseBuilder::cleanup");

  return 0;
}

int HnswSparseBuilder::train(IndexThreads::Pointer,
                             IndexSparseHolder::Pointer /*holder*/) {
  if (state_ != BUILD_STATE_INITED) {
    LOG_ERROR("Init the builder before HnswSparseBuilder::train");
    return IndexError_NoReady;
  }

  stats_.set_trained_count(0UL);
  stats_.set_trained_costtime(0UL);
  state_ = BUILD_STATE_TRAINED;

  LOG_INFO("End HnswSparseBuilder::train");

  return 0;
}

int HnswSparseBuilder::train(const IndexTrainer::Pointer & /*trainer*/) {
  if (state_ != BUILD_STATE_INITED) {
    LOG_ERROR("Init the builder before HnswSparseBuilder::train");
    return IndexError_NoReady;
  }

  LOG_INFO("Begin HnswSparseBuilder::train by trainer");

  stats_.set_trained_count(0UL);
  stats_.set_trained_costtime(0UL);
  state_ = BUILD_STATE_TRAINED;

  LOG_INFO("End HnswSparseBuilder::train by trainer");

  return 0;
}

int HnswSparseBuilder::build(IndexThreads::Pointer threads,
                             IndexSparseHolder::Pointer holder) {
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

  LOG_INFO("Begin HnswSparseBuilder::build sparse");

  // holder should be hybrid holder
  auto sparse_holder = std::dynamic_pointer_cast<IndexSparseHolder>(holder);

  if (sparse_holder == nullptr) {
    LOG_ERROR("HnswSparseBuilder failed to cast holder");
    return IndexError_Runtime;
  }

  if (sparse_holder->count() != static_cast<size_t>(-1)) {
    LOG_DEBUG("HnswSparseBuilder holder documents count %lu",
              sparse_holder->count());

    int ret = entity_.reserve_space(sparse_holder->count(),
                                    sparse_holder->total_sparse_count());
    if (ret != 0) {
      LOG_ERROR("HnswBuilde reserver space failed");
      return ret;
    }
  }
  auto iter = sparse_holder->create_iterator();
  if (!iter) {
    LOG_ERROR("Create iterator for holder failed");
    return IndexError_Runtime;
  }

  int ret;
  error_ = false;
  while (iter->is_valid()) {
    level_t level = alg_->get_random_level();
    node_id_t id;

    ret = entity_.add_vector(level, iter->key(), iter->sparse_count(),
                             iter->sparse_indices(), iter->sparse_data(), &id);

    if (ailego_unlikely(ret != 0) && ret != IndexError_InvalidValue) {
      return ret;
    }

    iter->next();
  }
  // Holder is not needed, cleanup it.
  sparse_holder.reset();

  LOG_INFO("Finished save vector, start build graph...");

  std::atomic<node_id_t> finished{0};

  ret = build_graph(threads, finished);
  if (ret != 0) {
    LOG_ERROR("Failed to build graph");
    return ret;
  }

  stats_.set_built_count(finished.load());
  stats_.set_built_costtime(ailego::Monotime::MilliSeconds() - start_time);
  state_ = BUILD_STATE_BUILT;

  LOG_INFO("End HnswSparseBuilder::build");
  return 0;
}

int HnswSparseBuilder::build_graph(IndexThreads::Pointer threads,
                                   std::atomic<node_id_t> &finished) {
  auto task_group = threads->make_group();
  if (!task_group) {
    LOG_ERROR("Failed to create task group");
    return IndexError_Runtime;
  }

  for (size_t i = 0; i < threads->count(); ++i) {
    task_group->submit(ailego::Closure ::New(this, &HnswSparseBuilder::do_build,
                                             i, threads->count(), &finished));
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

  return 0;
}

void HnswSparseBuilder::do_build(node_id_t idx, size_t step_size,
                                 std::atomic<node_id_t> *finished) {
  AILEGO_DEFER([&]() {
    std::lock_guard<std::mutex> latch(mutex_);
    cond_.notify_one();
  });

  HnswSparseContext *ctx = new (std::nothrow) HnswSparseContext(
      metric_,
      std::shared_ptr<HnswSparseEntity>(&entity_, [](HnswSparseEntity *) {}));
  if (ailego_unlikely(ctx == nullptr)) {
    if (!error_.exchange(true)) {
      LOG_ERROR("Failed to create context");
      errcode_ = IndexError_NoMemory;
    }
    return;
  }
  HnswSparseContext::Pointer auto_ptr(ctx);
  ctx->set_max_scan_num(entity_.doc_cnt());
  int ret = ctx->init(HnswSparseContext::kSparseBuilderContext);
  if (ret != 0) {
    if (!error_.exchange(true)) {
      LOG_ERROR("Failed to init context");
      errcode_ = IndexError_Runtime;
    }
    return;
  }

  IndexQueryMeta qmeta(meta_.data_type());
  for (node_id_t id = idx; id < entity_.doc_cnt(); id += step_size) {
    const void *vec = entity_.get_vector_meta(id);

    auto sparse_data = entity_.get_sparse_data_from_vector(vec);

    ctx->reset_query(sparse_data.first);

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

int HnswSparseBuilder::dump(const IndexDumper::Pointer &dumper) {
  if (state_ != BUILD_STATE_BUILT) {
    LOG_INFO("Build the index before HnswSparseBuilder::dump");
    return IndexError_NoReady;
  }

  LOG_INFO("Begin HnswSparseBuilder::dump");

  meta_.set_searcher("HnswSparseSearcher", HnswSparseEntity::kRevision,
                     ailego::Params());
  auto start_time = ailego::Monotime::MilliSeconds();

  int ret = IndexHelper::SerializeToDumper(meta_, dumper.get());
  if (ret != 0) {
    LOG_ERROR("Failed to serialize meta into dumper.");
    return ret;
  }

  ret = entity_.dump(dumper);
  if (ret != 0) {
    LOG_ERROR("HnswSparseBuilder dump index failed");
    return ret;
  }

  stats_.set_dumped_count(entity_.doc_cnt());
  stats_.set_dumped_costtime(ailego::Monotime::MilliSeconds() - start_time);

  LOG_INFO("EndHnswSparseBuilder::dump");
  return 0;
}

int HnswSparseBuilder::build(IndexThreads::Pointer threads, size_t count,
                             const uint64_t *keys,
                             const uint64_t *sparse_indptr,
                             const uint32_t *sparse_indices,
                             const void *sparse_data) {
  IndexQueryMeta qmeta(meta_.data_type());

  return build(threads, qmeta, count, keys, sparse_indptr, sparse_indices,
               sparse_data);
}

int HnswSparseBuilder::build(IndexThreads::Pointer threads,
                             const IndexQueryMeta &qmeta, size_t count,
                             const uint64_t *keys,
                             const uint64_t *sparse_indptr,
                             const uint32_t *sparse_indices,
                             const void *sparse_data) {
  if (!threads) {
    threads = std::make_shared<SingleQueueIndexThreads>(thread_cnt_, false);
    if (!threads) {
      return IndexError_NoMemory;
    }
  }

  auto start_time = ailego::Monotime::MilliSeconds();

  LOG_INFO("Begin HnswSparseBuilder::build sparse, documents count %lu", count);

  size_t total_sparse_count = sparse_indptr[count];

  int ret = entity_.reserve_space(count, total_sparse_count);
  if (ret != 0) {
    LOG_ERROR("HnswBuilde reserver space failed");
    return ret;
  }

  if (qmeta.data_type() == meta_.data_type()) {
    for (size_t i = 0; i < count; i++) {
      level_t level = alg_->get_random_level();
      node_id_t id;

      uint32_t sparse_count = sparse_indptr[i + 1] - sparse_indptr[i];
      const uint32_t *sparse_indices_temp = sparse_indices + sparse_indptr[i];

      const void *sparse_data_temp = static_cast<const char *>(sparse_data) +
                                     sparse_indptr[i] * qmeta.unit_size();

      ret = entity_.add_vector(level, keys[i], sparse_count,
                               sparse_indices_temp, sparse_data_temp, &id);
      if (ailego_unlikely(ret != 0) && ret != IndexError_InvalidValue) {
        return ret;
      }
    }
  } else if (meta_.data_type() == IndexMeta::DataType::DT_FP16 &&
             qmeta.data_type() == IndexMeta::DataType::DT_FP32) {
    // transform from float 32 to float 16
    auto reformer = IndexFactory::CreateReformer("HalfFloatSparseReformer");
    if (!reformer) {
      LOG_ERROR("Sparse reformer not existed.");

      return IndexError_NoExist;
    }

    meta_.set_converter("HalfFloatSparseConverter", 0, ailego::Params());
    meta_.set_reformer("HalfFloatSparseReformer", 0, ailego::Params());

    for (size_t i = 0; i < count; i++) {
      level_t level = alg_->get_random_level();
      node_id_t id;

      uint32_t sparse_count = sparse_indptr[i + 1] - sparse_indptr[i];
      const uint32_t *sparse_indices_temp = sparse_indices + sparse_indptr[i];

      const void *sparse_data_temp = static_cast<const char *>(sparse_data) +
                                     sparse_indptr[i] * qmeta.unit_size();

      std::string query_fp16;
      IndexQueryMeta ometa;

      reformer->transform(sparse_count, sparse_indices_temp, sparse_data_temp,
                          qmeta, &query_fp16, &ometa);

      ret = entity_.add_vector(level, keys[i], sparse_count,
                               sparse_indices_temp, query_fp16.data(), &id);
      if (ailego_unlikely(ret != 0) && ret != IndexError_InvalidValue) {
        return ret;
      }
    }
  } else {
    LOG_ERROR("Format not supported.");

    return IndexError_Unsupported;
  }

  LOG_INFO("Finished save vector, start build graph...");

  std::atomic<node_id_t> finished{0};

  ret = build_graph(threads, finished);
  if (ret != 0) {
    LOG_ERROR("Failed to build graph");
    return ret;
  }

  stats_.set_built_count(finished.load());
  stats_.set_built_costtime(ailego::Monotime::MilliSeconds() - start_time);
  state_ = BUILD_STATE_BUILT;

  LOG_INFO("End HnswSparseBuilder::build");
  return 0;
}

INDEX_FACTORY_REGISTER_BUILDER(HnswSparseBuilder);

}  // namespace core
}  // namespace zvec