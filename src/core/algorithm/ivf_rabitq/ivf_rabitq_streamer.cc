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
#include <limits>
#include <map>
#include <new>
#include <utility>
#include <vector>
#include <zvec/ailego/logger/logger.h>
#include <zvec/ailego/utility/time_helper.h>
#include "zvec/core/framework/index_error.h"
#include "zvec/core/framework/index_factory.h"
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
  int64_t configured_nprobe = static_cast<int64_t>(kDefaultIvfRabitqNprobe);
  if (params.get(PARAM_IVF_RABITQ_NPROBE, &configured_nprobe) &&
      configured_nprobe < 0) {
    LOG_ERROR("Invalid nprobe, must be greater than or equal to 0");
    return IndexError_InvalidArgument;
  }
  nprobe_ = static_cast<uint32_t>(configured_nprobe);

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
      "IvfRabitqStreamer initialized: dim=%zu, nprobe=%zu, metric=%s, "
      "scan_ratio=%.3f, bf_threshold=%zu",
      (size_t)dim, (size_t)nprobe_, metric_name_.c_str(), scan_ratio_,
      (size_t)brute_force_threshold_);

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

  // Validate the main header and all entity segment ranges before loading the
  // reformer, whose persisted dimensions control allocations and rotator setup.
  entity_ = std::make_shared<IvfRabitqEntity>();
  int ret = entity_->load(storage);
  if (ret != 0) {
    LOG_ERROR("Failed to load IvfRabitqEntity, ret=%d", ret);
    return ret;
  }
  if (entity_->dimension() != rabitq_meta_.dimension()) {
    LOG_ERROR("RaBitQ dimension mismatch: entity=%zu, meta=%zu",
              (size_t)entity_->dimension(),
              static_cast<size_t>(rabitq_meta_.dimension()));
    return IndexError_InvalidFormat;
  }

  // Load reformer
  reformer_ = std::make_shared<IvfRabitqReformer>();
  ret = reformer_->init(metric_name_);
  if (ret != 0) {
    LOG_ERROR("Failed to init IvfRabitqReformer, ret=%d", ret);
    return ret;
  }
  ret = reformer_->load(storage);
  if (ret != 0) {
    LOG_ERROR("Failed to load IvfRabitqReformer, ret=%d", ret);
    return ret;
  }
  if (reformer_->dimension() != entity_->dimension() ||
      reformer_->padded_dim() != entity_->padded_dim() ||
      reformer_->ex_bits() != entity_->ex_bits() ||
      reformer_->num_clusters() != entity_->cluster_count()) {
    LOG_ERROR(
        "IVF RaBitQ entity and reformer metadata mismatch: "
        "dimension=%zu/%zu, padded_dim=%zu/%zu, ex_bits=%zu/%zu, "
        "clusters=%zu/%zu",
        (size_t)entity_->dimension(), reformer_->dimension(),
        (size_t)entity_->padded_dim(), reformer_->padded_dim(),
        (size_t)entity_->ex_bits(), reformer_->ex_bits(),
        (size_t)entity_->cluster_count(), reformer_->num_clusters());
    return IndexError_InvalidFormat;
  }

  stats_.set_loaded_count(entity_->total_vector_count());
  stats_.set_loaded_costtime(timer.milli_seconds());

  LOG_INFO("IvfRabitqStreamer loaded: %zu vectors, %zu clusters, cost %zu ms",
           (size_t)entity_->total_vector_count(),
           (size_t)entity_->cluster_count(),
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
  if (!entity_) {
    LOG_ERROR("IVF RaBitQ entity is not available");
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

const void *IvfRabitqStreamer::get_vector(uint64_t /*key*/) const {
  return nullptr;
}

int IvfRabitqStreamer::get_vector(const uint64_t /*key*/,
                                  IndexStorage::MemoryBlock & /*block*/) const {
  return IndexError_Unsupported;
}

// --------------------------------------------------------------------------
// search_bf_impl (brute force - search all clusters, single query)
// --------------------------------------------------------------------------
int IvfRabitqStreamer::search_bf_impl(const void *query,
                                      const IndexQueryMeta &qmeta,
                                      Context::Pointer &context) const {
  if (context && context->group_by().is_valid()) {
    return search_group_by_impl_internal(query, qmeta, 1, context, true);
  }
  return search_impl_internal(query, qmeta, 1, context, true);
}

// --------------------------------------------------------------------------
// search_bf_impl (brute force - search all clusters, multi query)
// --------------------------------------------------------------------------
int IvfRabitqStreamer::search_bf_impl(const void *query,
                                      const IndexQueryMeta &qmeta,
                                      uint32_t count,
                                      Context::Pointer &context) const {
  if (context && context->group_by().is_valid()) {
    return search_group_by_impl_internal(query, qmeta, count, context, true);
  }
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
  if (context && context->group_by().is_valid()) {
    return search_group_by_impl_internal(query, qmeta, count, context, false);
  }
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
    auto &query_state = ctx->query_state;
    int ret = reformer_->create_query_state(q_vec, &query_state);
    if (ret != 0) {
      LOG_ERROR("Failed to create query state, ret=%d", ret);
      return ret;
    }

    // Select probe centroids with the same metric used for build assignment.
    auto &probe_centroids = ctx->probe_centroids;
    ret = reformer_->select_probe_centroids(q_vec, nprobe, &query_state,
                                            &probe_centroids);
    if (ret != 0) {
      LOG_ERROR("Failed to select probe centroids, ret=%d", ret);
      return ret;
    }

    // Use context heap for online dist-k gated pruning
    IndexDocumentHeap &heap = ctx->mutable_result_heap();
    heap.clear();
    heap.limit(topk);
    const auto &filter = ctx->filter();

    uint32_t scanned = 0;

    for (uint32_t p = 0; p < probe_centroids.size() && scanned < max_scan;
         ++p) {
      const auto &probe = probe_centroids[p];
      uint32_t cid = probe.id;

      ret = reformer_->prepare_for_cluster(probe, &query_state);
      if (ret != 0) {
        LOG_ERROR("Failed to prepare for cluster %zu, ret=%d", (size_t)cid,
                  ret);
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
        LOG_ERROR("Failed to search cluster %zu, ret=%d", (size_t)cid, ret);
        continue;
      }

      scanned += entity_->cluster_meta(cid).vector_count;
    }

    // Drain heap into sorted result list for query q
    ctx->topk_to_result(q);
  }

  return 0;
}

int IvfRabitqStreamer::search_group_by_impl_internal(
    const void *query, const IndexQueryMeta &qmeta, uint32_t count,
    Context::Pointer &context, bool force_brute_force) const {
  if (!query) {
    LOG_ERROR("Null query");
    return IndexError_InvalidArgument;
  }
  if (!reformer_ || !reformer_->loaded() || !entity_) {
    return IndexError_NoReady;
  }

  auto *ctx = dynamic_cast<IvfRabitqContext *>(context.get());
  if (!ctx) {
    return IndexError_Cast;
  }
  if (!ctx->group_by_search() || !ctx->group_by().is_valid()) {
    LOG_ERROR("Invalid group-by state");
    return IndexError_InvalidArgument;
  }

  const bool brute_force = force_brute_force || entity_->total_vector_count() <=
                                                    ctx->bruteforce_threshold();
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

  const size_t padded_dim = reformer_->padded_dim();
  const size_t ex_bits = reformer_->ex_bits();
  const size_t dimension = reformer_->dimension();
  if (qmeta.dimension() != meta_.dimension() ||
      qmeta.data_type() != meta_.data_type() ||
      qmeta.element_size() != meta_.element_size() ||
      qmeta.dimension() < dimension) {
    return IndexError_Mismatch;
  }

  ctx->reset_group_results(count);
  for (uint32_t q = 0; q < count; ++q) {
    const float *query_vector = reinterpret_cast<const float *>(
        static_cast<const char *>(query) +
        (static_cast<size_t>(q) * qmeta.element_size()));
    auto &query_state = ctx->query_state;
    int ret = reformer_->create_query_state(query_vector, &query_state);
    if (ret != 0) {
      return ret;
    }

    auto &probe_centroids = ctx->probe_centroids;
    ret = reformer_->select_probe_centroids(query_vector, nprobe, &query_state,
                                            &probe_centroids);
    if (ret != 0) {
      return ret;
    }

    auto &heaps = ctx->group_topk_heaps();
    heaps.clear();
    const auto &filter = ctx->filter();
    uint32_t scanned = 0;
    for (const auto &probe : probe_centroids) {
      const uint32_t cluster_id = probe.id;
      if (scanned >= max_scan) {
        break;
      }
      ret = reformer_->prepare_for_cluster(probe, &query_state);
      if (ret != 0) {
        LOG_ERROR("Failed to prepare for cluster %zu, ret=%d",
                  (size_t)cluster_id, ret);
        continue;
      }

      if (filter.is_valid()) {
        ret = entity_->search_cluster_group_by(
            cluster_id, query_state, padded_dim, ex_bits, filter,
            ctx->group_by(), ctx->group_topk(), ctx->threshold(), &heaps);
      } else {
        ret = entity_->search_cluster_group_by(
            cluster_id, query_state, padded_dim, ex_bits, ctx->group_by(),
            ctx->group_topk(), ctx->threshold(), &heaps);
      }
      if (ret != 0) {
        LOG_ERROR("Failed to search cluster %zu, ret=%d", (size_t)cluster_id,
                  ret);
        continue;
      }
      scanned += entity_->cluster_meta(cluster_id).vector_count;
    }
    ctx->topk_to_group_result(q);
  }
  return 0;
}

int IvfRabitqStreamer::search_bf_by_p_keys_impl(
    const void *query, const std::vector<std::vector<uint64_t>> &p_keys,
    const IndexQueryMeta &qmeta, uint32_t count,
    Context::Pointer &context) const {
  if (!query || p_keys.size() != count) {
    return IndexError_InvalidArgument;
  }
  if (!reformer_ || !reformer_->loaded() || !entity_) {
    return IndexError_NoReady;
  }
  if (qmeta.dimension() != meta_.dimension() ||
      qmeta.data_type() != meta_.data_type() ||
      qmeta.element_size() != meta_.element_size()) {
    return IndexError_Mismatch;
  }

  auto *ctx = dynamic_cast<IvfRabitqContext *>(context.get());
  if (!ctx) {
    return IndexError_Cast;
  }
  const bool has_group_by = ctx->group_by_search();
  if (has_group_by && !ctx->group_by().is_valid()) {
    return IndexError_InvalidArgument;
  }
  if (has_group_by) {
    ctx->reset_group_results(count);
  } else {
    ctx->reset_results(count);
  }

  const size_t padded_dim = reformer_->padded_dim();
  const size_t ex_bits = reformer_->ex_bits();
  const uint32_t topk = ctx->topk() == 0 ? 10 : ctx->topk();
  const auto &filter = ctx->filter();
  for (uint32_t q = 0; q < count; ++q) {
    const float *query_vector = reinterpret_cast<const float *>(
        static_cast<const char *>(query) +
        (static_cast<size_t>(q) * qmeta.element_size()));
    auto &query_state = ctx->query_state;
    int ret = reformer_->create_query_state(query_vector, &query_state);
    if (ret != 0) {
      return ret;
    }

    std::map<uint32_t, std::vector<uint32_t>> cluster_ids;
    for (uint64_t key : p_keys[q]) {
      if (filter.is_valid() && filter(key)) {
        continue;
      }
      const uint32_t id = entity_->key_to_id(key);
      if (id == std::numeric_limits<uint32_t>::max()) {
        continue;
      }
      const uint32_t cluster_id = entity_->get_cluster_id(id);
      if (cluster_id == std::numeric_limits<uint32_t>::max()) {
        continue;
      }
      cluster_ids[cluster_id].push_back(id);
    }

    auto &result_heap = ctx->mutable_result_heap();
    if (!has_group_by) {
      result_heap.clear();
      result_heap.limit(topk);
      result_heap.set_threshold(ctx->threshold());
    } else {
      ctx->group_topk_heaps().clear();
    }

    for (const auto &entry : cluster_ids) {
      const uint32_t cluster_id = entry.first;
      ret = reformer_->prepare_for_cluster(cluster_id, &query_state);
      if (ret != 0) {
        return ret;
      }
      IndexDocumentList documents;
      ret = entity_->compute_distances(cluster_id, entry.second, query_state,
                                       padded_dim, ex_bits, &documents);
      if (ret != 0) {
        return ret;
      }
      for (const auto &document : documents) {
        if (!has_group_by) {
          result_heap.emplace(document.key(), document.score());
          continue;
        }
        const std::string group_id = ctx->group_by()(document.key());
        auto &heap = ctx->group_topk_heaps()[group_id];
        if (heap.empty()) {
          heap.limit(ctx->group_topk());
          heap.set_threshold(ctx->threshold());
        }
        heap.emplace(document.key(), document.score());
      }
    }

    if (has_group_by) {
      ctx->topk_to_group_result(q);
    } else {
      ctx->topk_to_result(q);
    }
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

INDEX_FACTORY_REGISTER_STREAMER(IvfRabitqStreamer);

}  // namespace core
}  // namespace zvec
