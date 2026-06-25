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

#include "ivf_rabitq_streamer.h"
#include <new>
#include <utility>
#include <vector>
#include <zvec/ailego/logger/logger.h>
#include <zvec/ailego/utility/time_helper.h>
#include "zvec/core/framework/index_error.h"
#include "zvec/core/framework/index_helper.h"
#include "zvec/core/framework/index_meta.h"
#include "ivf_rabitq_context.h"
#include "ivf_rabitq_entity.h"
#include "ivf_rabitq_index_provider.h"
#include "ivf_rabitq_params.h"
#include "ivf_rabitq_reformer.h"
#include "ivf_rabitq_util.h"

namespace zvec {
namespace core {

// --------------------------------------------------------------------------
// init
// --------------------------------------------------------------------------
int IvfRabitqStreamer::init(const IndexMeta &meta,
                            const ailego::Params &params) {
  meta_ = meta;
  params_ = params;

  // Parse IVF RaBitQ specific params
  params.get(PARAM_IVF_RABITQ_NPROBE, &nprobe_);

  params.get(PARAM_IVF_RABITQ_SCAN_RATIO, &scan_ratio_);
  if (scan_ratio_ <= 0.0f || scan_ratio_ > 1.0f) {
    LOG_ERROR("Invalid params %s=%f", PARAM_IVF_RABITQ_SCAN_RATIO.c_str(),
              scan_ratio_);
    return IndexError_InvalidArgument;
  }

  params.get(PARAM_IVF_RABITQ_BRUTE_FORCE_THRESHOLD, &brute_force_threshold_);

  int ret = PrepareAndCheckIvfRabitqInternalMeta(meta_, params, &rabitq_meta_,
                                                 &metric_name_);
  if (ret != 0) {
    return ret;
  }

  uint32_t dim = rabitq_meta_.dimension();
  state_ = STATE_INITED;

  LOG_INFO(
      "IvfRabitqStreamer initialized: dim=%u, nprobe=%u, metric=%s, "
      "scan_ratio=%.3f, bf_threshold=%u",
      dim, nprobe_, metric_name_.c_str(), scan_ratio_, brute_force_threshold_);

  return 0;
}

// --------------------------------------------------------------------------
// open
// --------------------------------------------------------------------------
int IvfRabitqStreamer::open(IndexStorage::Pointer storage) {
  if (!storage) {
    LOG_ERROR("Invalid storage");
    return IndexError_InvalidArgument;
  }
  if (state_ != STATE_INITED) {
    LOG_ERROR("Streamer not initialized");
    return IndexError_NoReady;
  }

  int ret = IndexHelper::DeserializeFromStorage(storage.get(), &meta_);
  if (ret != 0) {
    LOG_ERROR("Failed to deserialize meta from storage");
    return ret;
  }
  ret = PrepareAndCheckIvfRabitqInternalMeta(meta_, params_, &rabitq_meta_,
                                             &metric_name_);
  if (ret != 0) {
    return ret;
  }

  storage_ = std::move(storage);
  ret = load_index(storage_);
  if (ret != 0) {
    LOG_ERROR("Failed to load index, ret=%d", ret);
    return ret;
  }

  state_ = STATE_LOADED;
  return 0;
}

// --------------------------------------------------------------------------
// load_index
// --------------------------------------------------------------------------
int IvfRabitqStreamer::load_index(IndexStorage::Pointer storage) {
  ailego::ElapsedTime timer;

  // Load reformer
  reformer_ = std::make_shared<IvfRabitqReformer>();
  int ret = reformer_->init(metric_name_);
  if (ret != 0) {
    LOG_ERROR("Failed to init IvfRabitqReformer, ret=%d", ret);
    return ret;
  }
  ret = reformer_->load(storage);
  if (ret != 0) {
    LOG_ERROR("Failed to load IvfRabitqReformer, ret=%d", ret);
    return ret;
  }
  if (reformer_->dimension() != rabitq_meta_.dimension()) {
    LOG_ERROR("RaBitQ dimension mismatch: reformer=%zu, meta=%zu",
              reformer_->dimension(),
              static_cast<size_t>(rabitq_meta_.dimension()));
    return IndexError_Mismatch;
  }

  // Load entity data
  entity_ = std::make_shared<IvfRabitqEntity>();
  ret = entity_->load(storage);
  if (ret != 0) {
    LOG_ERROR("Failed to load IvfRabitqEntity, ret=%d", ret);
    return ret;
  }

  stats_.set_loaded_count(entity_->total_vector_count());
  stats_.set_loaded_costtime(timer.milli_seconds());

  LOG_INFO("IvfRabitqStreamer loaded: %u vectors, %u clusters, cost %zu ms",
           entity_->total_vector_count(), entity_->cluster_count(),
           static_cast<size_t>(timer.milli_seconds()));

  return 0;
}

// --------------------------------------------------------------------------
// create_context
// --------------------------------------------------------------------------
IndexStreamer::Context::Pointer IvfRabitqStreamer::create_context() const {
  if (state_ != STATE_LOADED) {
    LOG_ERROR("Load the index first before create context");
    return Context::Pointer();
  }

  auto *ctx = new (std::nothrow) IvfRabitqContext();
  if (!ctx) {
    LOG_ERROR("Failed to allocate IvfRabitqContext");
    return Context::Pointer();
  }
  ailego::Params defaults;
  defaults.set(PARAM_IVF_RABITQ_NPROBE, nprobe_);
  defaults.set(PARAM_IVF_RABITQ_SCAN_RATIO, scan_ratio_);
  defaults.set(PARAM_IVF_RABITQ_BRUTE_FORCE_THRESHOLD, brute_force_threshold_);
  ctx->update(defaults);
  return Context::Pointer(ctx);
}

IndexProvider::Pointer IvfRabitqStreamer::create_provider(void) const {
  if (state_ != STATE_LOADED) {
    LOG_ERROR("Load the index first before create provider");
    return Provider::Pointer();
  }
  if (!entity_ || entity_->quantized_vector_element_size() == 0) {
    LOG_ERROR("Quantized vectors are not available for IVF RaBitQ provider");
    return Provider::Pointer();
  }

  auto *provider = new (std::nothrow)
      IvfRabitqIndexProvider(meta_, entity_, "IvfRabitqStreamer");
  if (!provider) {
    LOG_ERROR("Failed to alloc IvfRabitqIndexProvider");
    return Provider::Pointer();
  }
  return Provider::Pointer(provider);
}

const void *IvfRabitqStreamer::get_vector(uint64_t key) const {
  if (!entity_) {
    return nullptr;
  }
  return entity_->get_vector_by_key(key);
}

int IvfRabitqStreamer::get_vector(const uint64_t key,
                                  IndexStorage::MemoryBlock &block) const {
  if (!entity_) {
    return IndexError_NoReady;
  }
  return entity_->get_vector_by_key(key, block);
}

// --------------------------------------------------------------------------
// search_bf_impl (brute force - search all clusters, single query)
// --------------------------------------------------------------------------
int IvfRabitqStreamer::search_bf_impl(const void *query,
                                      const IndexQueryMeta &qmeta,
                                      Context::Pointer &context) const {
  return search_impl_internal(query, qmeta, 1, context, true);
}

// --------------------------------------------------------------------------
// search_bf_impl (brute force - search all clusters, multi query)
// --------------------------------------------------------------------------
int IvfRabitqStreamer::search_bf_impl(const void *query,
                                      const IndexQueryMeta &qmeta,
                                      uint32_t count,
                                      Context::Pointer &context) const {
  return search_impl_internal(query, qmeta, count, context, true);
}

// --------------------------------------------------------------------------
// search_impl (single query)
// --------------------------------------------------------------------------
int IvfRabitqStreamer::search_impl(const void *query,
                                   const IndexQueryMeta &qmeta,
                                   Context::Pointer &context) const {
  return search_impl(query, qmeta, 1, context);
}

// --------------------------------------------------------------------------
// search_impl
// --------------------------------------------------------------------------
int IvfRabitqStreamer::search_impl(const void *query,
                                   const IndexQueryMeta &qmeta, uint32_t count,
                                   Context::Pointer &context) const {
  return search_impl_internal(query, qmeta, count, context, false);
}

int IvfRabitqStreamer::search_impl_internal(const void *query,
                                            const IndexQueryMeta &qmeta,
                                            uint32_t count,
                                            Context::Pointer &context,
                                            bool force_brute_force) const {
  if (!query) {
    LOG_ERROR("Null query");
    return IndexError_InvalidArgument;
  }
  if (!reformer_ || !reformer_->loaded()) {
    LOG_ERROR("Reformer not loaded for search");
    return IndexError_NoReady;
  }
  if (!entity_) {
    LOG_ERROR("Entity not loaded for search");
    return IndexError_NoReady;
  }

  IvfRabitqContext *ctx = dynamic_cast<IvfRabitqContext *>(context.get());
  if (!ctx) {
    LOG_ERROR("Invalid context type");
    return IndexError_Cast;
  }

  bool brute_force = force_brute_force || entity_->total_vector_count() <=
                                              ctx->bruteforce_threshold();

  uint32_t topk = ctx->topk();
  if (topk == 0) {
    topk = 10;
  }
  uint32_t nprobe = 0;
  uint32_t max_scan = 0;
  if (brute_force) {
    nprobe = entity_->cluster_count();
    max_scan = entity_->total_vector_count();
    ctx->set_search_limits(max_scan);
  } else {
    int ret = ctx->update_search_limits(entity_->total_vector_count(),
                                        entity_->cluster_count(), &nprobe);
    if (ret != 0) {
      return ret;
    }
    max_scan = ctx->max_scan_count();
  }

  size_t padded_dim = reformer_->padded_dim();
  size_t ex_bits = reformer_->ex_bits();
  size_t dimension = reformer_->dimension();
  if (qmeta.dimension() != meta_.dimension() ||
      qmeta.data_type() != meta_.data_type() ||
      qmeta.element_size() != meta_.element_size()) {
    LOG_ERROR("Unsupported query meta");
    return IndexError_Mismatch;
  }
  if (qmeta.dimension() < dimension) {
    LOG_ERROR("Query dimension=%zu smaller than RaBitQ dimension=%zu",
              static_cast<size_t>(qmeta.dimension()), dimension);
    return IndexError_Mismatch;
  }

  // Reset results and heap for all queries
  ctx->reset_results(count);

  for (uint32_t q = 0; q < count; ++q) {
    const float *q_vec = reinterpret_cast<const float *>(
        static_cast<const char *>(query) +
        (static_cast<size_t>(q) * qmeta.element_size()));

    // Create query state (rotate query and prepare per-query scan state)
    IvfRabitqQueryState query_state;
    int ret = reformer_->create_query_state(q_vec, &query_state);
    if (ret != 0) {
      LOG_ERROR("Failed to create query state, ret=%d", ret);
      return ret;
    }

    // Select probe centroids with the same metric used for build assignment.
    std::vector<uint32_t> probe_centroids;
    ret = reformer_->select_probe_centroids(q_vec, nprobe, &query_state,
                                            &probe_centroids);
    if (ret != 0) {
      LOG_ERROR("Failed to select probe centroids, ret=%d", ret);
      return ret;
    }

    // Use context heap for online distk-gated pruning
    IndexDocumentHeap &heap = ctx->mutable_result_heap();
    heap.clear();
    heap.limit(topk);
    const auto &filter = ctx->filter();

    uint32_t scanned = 0;

    for (uint32_t p = 0; p < probe_centroids.size() && scanned < max_scan;
         ++p) {
      uint32_t cid = probe_centroids[p];

      ret = reformer_->prepare_for_cluster(cid, &query_state);
      if (ret != 0) {
        LOG_ERROR("Failed to prepare for cluster %u, ret=%d", cid, ret);
        continue;
      }

      // Scan cluster with 1-bit lower-bound pruning before extra-bit boosting
      if (!filter.is_valid()) {
        ret = entity_->search_cluster(cid, query_state, padded_dim, ex_bits,
                                      &heap);
      } else {
        ret = entity_->search_cluster(cid, query_state, padded_dim, ex_bits,
                                      filter, &heap);
      }
      if (ret != 0) {
        LOG_ERROR("Failed to search cluster %u, ret=%d", cid, ret);
        continue;
      }

      scanned += entity_->cluster_meta(cid).vector_count;
    }

    // Drain heap into sorted result list for query q
    ctx->topk_to_result(q);
  }

  return 0;
}

int IvfRabitqStreamer::unload() {
  reformer_.reset();
  entity_.reset();
  storage_.reset();
  stats_.set_loaded_count(0UL);
  stats_.set_loaded_costtime(0UL);
  stats_.clear_attributes();
  state_ = STATE_INITED;

  return 0;
}

// --------------------------------------------------------------------------
// cleanup
// --------------------------------------------------------------------------
int IvfRabitqStreamer::cleanup() {
  LOG_INFO("IvfRabitqStreamer cleanup");

  this->unload();
  params_.clear();
  nprobe_ = kDefaultIvfRabitqNprobe;
  scan_ratio_ = kDefaultIvfRabitqScanRatio;
  brute_force_threshold_ = kDefaultIvfRabitqBruteForceThreshold;
  state_ = STATE_INIT;
  return 0;
}

}  // namespace core
}  // namespace zvec
