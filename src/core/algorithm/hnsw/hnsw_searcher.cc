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
#include "hnsw_searcher.h"
#include "hnsw_algorithm.h"
#include "hnsw_index_provider.h"
#include "hnsw_params.h"

namespace zvec {
namespace core {

HnswSearcher::HnswSearcher() {}

HnswSearcher::~HnswSearcher() {}

int HnswSearcher::init(const ailego::Params &search_params) {
  params_ = search_params;
  params_.get(PARAM_HNSW_SEARCHER_EF, &ef_);
  params_.get(PARAM_HNSW_SEARCHER_MAX_SCAN_RATIO, &max_scan_ratio_);
  params_.get(PARAM_HNSW_SEARCHER_VISIT_BLOOMFILTER_ENABLE, &bf_enabled_);
  params_.get(PARAM_HNSW_SEARCHER_CHECK_CRC_ENABLE, &check_crc_enabled_);
  params_.get(PARAM_HNSW_SEARCHER_NEIGHBORS_IN_MEMORY_ENABLE,
              &neighbors_in_memory_enabled_);
  params_.get(PARAM_HNSW_SEARCHER_VISIT_BLOOMFILTER_NEGATIVE_PROB,
              &bf_negative_probility_);
  params_.get(PARAM_HNSW_SEARCHER_BRUTE_FORCE_THRESHOLD,
              &bruteforce_threshold_);
  params_.get(PARAM_HNSW_SEARCHER_FORCE_PADDING_RESULT_ENABLE,
              &force_padding_topk_enabled_);

  if (ef_ == 0) {
    ef_ = HnswEntity::kDefaultEf;
  }
  if (bf_negative_probility_ <= 0.0f || bf_negative_probility_ >= 1.0f) {
    LOG_ERROR("[%s] must be in range (0,1)",
              PARAM_HNSW_SEARCHER_VISIT_BLOOMFILTER_NEGATIVE_PROB.c_str());
    return IndexError_InvalidArgument;
  }

  entity_.set_neighbors_in_memory(neighbors_in_memory_enabled_);

  state_ = STATE_INITED;

  LOG_DEBUG(
      "Init params: ef=%u maxScanRatio=%f bfEnabled=%u checkCrcEnabled=%u "
      "neighborsInMemoryEnabled=%u bfNagtiveProb=%f bruteForceThreshold=%u "
      "forcePadding=%u",
      ef_, max_scan_ratio_, bf_enabled_, check_crc_enabled_,
      neighbors_in_memory_enabled_, bf_negative_probility_,
      bruteforce_threshold_, force_padding_topk_enabled_);

  return 0;
}

void HnswSearcher::print_debug_info() {
  for (node_id_t id = 0; id < entity_.doc_cnt(); ++id) {
    Neighbors neighbours = entity_.get_neighbors(0, id);
    std::cout << "node: " << id << "; ";
    for (uint32_t i = 0; i < neighbours.size(); ++i) {
      std::cout << neighbours[i];

      if (i == neighbours.size() - 1) {
        std::cout << std::endl;
      } else {
        std::cout << ", ";
      }
    }
  }
}

int HnswSearcher::cleanup() {
  LOG_INFO("Begin HnswSearcher:cleanup");

  metric_.reset();
  meta_.clear();
  stats_.clear_attributes();
  stats_.set_loaded_count(0UL);
  stats_.set_loaded_costtime(0UL);
  max_scan_ratio_ = HnswEntity::kDefaultScanRatio;
  max_scan_num_ = 0U;
  ef_ = HnswEntity::kDefaultEf;
  bf_enabled_ = false;
  bf_negative_probility_ = HnswEntity::kDefaultBFNegativeProbility;
  bruteforce_threshold_ = HnswEntity::kDefaultBruteForceThreshold;
  check_crc_enabled_ = false;
  neighbors_in_memory_enabled_ = false;
  entity_.cleanup();
  state_ = STATE_INIT;

  LOG_INFO("End HnswSearcher:cleanup");

  return 0;
}

int HnswSearcher::load(IndexStorage::Pointer container,
                       IndexMetric::Pointer metric) {
  if (state_ != STATE_INITED) {
    LOG_ERROR("Init the searcher first before load index");
    return IndexError_Runtime;
  }

  LOG_INFO("Begin HnswSearcher:load");

  auto start_time = ailego::Monotime::MilliSeconds();

  int ret = IndexHelper::DeserializeFromStorage(container.get(), &meta_);
  if (ret != 0) {
    LOG_ERROR("Failed to deserialize meta from container");
    return ret;
  }

  ret = entity_.load(container, check_crc_enabled_);
  if (ret != 0) {
    LOG_ERROR("HnswSearcher load index failed");
    return ret;
  }

  alg_ = HnswAlgorithm::UPointer(new HnswAlgorithm(entity_));

  if (metric) {
    metric_ = metric;
  } else {
    metric_ = IndexFactory::CreateMetric(meta_.metric_name());
    if (!metric_) {
      LOG_ERROR("CreateMetric failed, name: %s", meta_.metric_name().c_str());
      return IndexError_NoExist;
    }
    ret = metric_->init(meta_, meta_.metric_params());
    if (ret != 0) {
      LOG_ERROR("IndexMetric init failed, ret=%d", ret);
      return ret;
    }
    if (metric_->query_metric()) {
      metric_ = metric_->query_metric();
    }
  }

  if (!metric_->is_matched(meta_)) {
    LOG_ERROR("IndexMetric not match index meta");
    return IndexError_Mismatch;
  }

  max_scan_num_ = static_cast<uint32_t>(max_scan_ratio_ * entity_.doc_cnt());
  max_scan_num_ = std::max(4096U, max_scan_num_);

  stats_.set_loaded_count(entity_.doc_cnt());
  stats_.set_loaded_costtime(ailego::Monotime::MilliSeconds() - start_time);
  state_ = STATE_LOADED;
  magic_ = IndexContext::GenerateMagic();

  LOG_INFO("End HnswSearcher::load");

  return 0;
}

int HnswSearcher::unload() {
  LOG_INFO("HnswSearcher unload index");

  meta_.clear();
  entity_.cleanup();
  metric_.reset();
  max_scan_num_ = 0;
  stats_.set_loaded_count(0UL);
  stats_.set_loaded_costtime(0UL);
  state_ = STATE_INITED;

  return 0;
}

int HnswSearcher::update_context(HnswContext *ctx) const {
  const HnswEntity::Pointer entity = entity_.clone();
  if (!entity) {
    LOG_ERROR("Failed to clone search context entity");
    return IndexError_Runtime;
  }
  ctx->set_max_scan_num(max_scan_num_);
  ctx->set_bruteforce_threshold(bruteforce_threshold_);

  return ctx->update_context(HnswContext::kSearcherContext, meta_, metric_,
                             entity, magic_);
}

int HnswSearcher::search_impl(const void *query, const IndexQueryMeta &qmeta,
                              uint32_t count, Context::Pointer &context) const {
  if (ailego_unlikely(!query || !context)) {
    LOG_ERROR("The context is not created by this searcher");
    return IndexError_Mismatch;
  }
  HnswContext *ctx = dynamic_cast<HnswContext *>(context.get());
  ailego_do_if_false(ctx) {
    LOG_ERROR("Cast context to HnswContext failed");
    return IndexError_Cast;
  }

  if (entity_.doc_cnt() <= ctx->get_bruteforce_threshold()) {
    return search_bf_impl(query, qmeta, count, context);
  }

  if (ctx->magic() != magic_) {
    //! context is created by another searcher or streamer
    int ret = update_context(ctx);
    if (ret != 0) {
      return ret;
    }
  }

  ctx->clear();
  ctx->resize_results(count);
  for (size_t q = 0; q < count; ++q) {
    ctx->reset_query(query);
    int ret = alg_->fast_search(ctx);
    if (ailego_unlikely(ret != 0)) {
      LOG_ERROR("Hnsw searcher fast search failed");
      return ret;
    }
    ctx->topk_to_result(q);
    query = static_cast<const char *>(query) + qmeta.element_size();
  }

  if (ailego_unlikely(ctx->error())) {
    return IndexError_Runtime;
  }

  return 0;
}

int HnswSearcher::search_bf_impl(const void *query, const IndexQueryMeta &qmeta,
                                 uint32_t count,
                                 Context::Pointer &context) const {
  if (ailego_unlikely(!query || !context)) {
    LOG_ERROR("The context is not created by this searcher");
    return IndexError_Mismatch;
  }
  HnswContext *ctx = dynamic_cast<HnswContext *>(context.get());
  ailego_do_if_false(ctx) {
    LOG_ERROR("Cast context to HnswContext failed");
    return IndexError_Cast;
  }
  if (ctx->magic() != magic_) {
    //! context is created by another searcher or streamer
    int ret = update_context(ctx);
    if (ret != 0) {
      return ret;
    }
  }

  ctx->clear();
  ctx->resize_results(count);

  if (ctx->group_by_search()) {
    if (!ctx->group_by().is_valid()) {
      LOG_ERROR("Invalid group-by function");
      return IndexError_InvalidArgument;
    }

    std::function<std::string(node_id_t)> group_by = [&](node_id_t id) {
      return ctx->group_by()(entity_.get_key(id));
    };

    for (size_t q = 0; q < count; ++q) {
      ctx->reset_query(query);
      ctx->group_topk_heaps().clear();

      for (node_id_t id = 0; id < entity_.doc_cnt(); ++id) {
        if (entity_.get_key(id) == kInvalidKey) {
          continue;
        }
        if (!ctx->filter().is_valid() || !ctx->filter()(entity_.get_key(id))) {
          dist_t dist = ctx->dist_calculator().batch_dist(id);

          std::string group_id = group_by(id);

          auto &topk_heap = ctx->group_topk_heaps()[group_id];
          if (topk_heap.empty()) {
            topk_heap.limit(ctx->group_topk());
          }
          topk_heap.emplace_back(id, dist);
        }
      }
      ctx->topk_to_result(q);
      query = static_cast<const char *>(query) + qmeta.element_size();
    }
  } else {
    for (size_t q = 0; q < count; ++q) {
      ctx->reset_query(query);
      ctx->topk_heap().clear();
      for (node_id_t id = 0; id < entity_.doc_cnt(); ++id) {
        if (entity_.get_key(id) == kInvalidKey) {
          continue;
        }
        if (!ctx->filter().is_valid() || !ctx->filter()(entity_.get_key(id))) {
          dist_t dist = ctx->dist_calculator().batch_dist(id);
          ctx->topk_heap().emplace(id, dist);
        }
      }
      ctx->topk_to_result(q);
      query = static_cast<const char *>(query) + qmeta.element_size();
    }
  }

  if (ailego_unlikely(ctx->error())) {
    return IndexError_Runtime;
  }

  return 0;
}

int HnswSearcher::search_bf_by_p_keys_impl(
    const void *query, const std::vector<std::vector<uint64_t>> &p_keys,
    const IndexQueryMeta &qmeta, uint32_t count,
    Context::Pointer &context) const {
  if (ailego_unlikely(!query || !context)) {
    LOG_ERROR("The context is not created by this searcher");
    return IndexError_Mismatch;
  }

  if (ailego_unlikely(p_keys.size() != count)) {
    LOG_ERROR("The size of p_keys is not equal to count");
    return IndexError_InvalidArgument;
  }

  HnswContext *ctx = dynamic_cast<HnswContext *>(context.get());
  ailego_do_if_false(ctx) {
    LOG_ERROR("Cast context to HnswContext failed");
    return IndexError_Cast;
  }
  if (ctx->magic() != magic_) {
    //! context is created by another searcher or streamer
    int ret = update_context(ctx);
    if (ret != 0) {
      return ret;
    }
  }

  ctx->clear();
  ctx->resize_results(count);

  if (ctx->group_by_search()) {
    if (!ctx->group_by().is_valid()) {
      LOG_ERROR("Invalid group-by function");
      return IndexError_InvalidArgument;
    }

    std::function<std::string(node_id_t)> group_by = [&](node_id_t id) {
      return ctx->group_by()(entity_.get_key(id));
    };

    for (size_t q = 0; q < count; ++q) {
      ctx->reset_query(query);
      ctx->group_topk_heaps().clear();

      for (size_t idx = 0; idx < p_keys[q].size(); ++idx) {
        uint64_t pk = p_keys[q][idx];
        if (!ctx->filter().is_valid() || !ctx->filter()(pk)) {
          node_id_t id = entity_.get_id(pk);
          if (id != kInvalidNodeId) {
            dist_t dist = ctx->dist_calculator().batch_dist(id);
            std::string group_id = group_by(id);

            auto &topk_heap = ctx->group_topk_heaps()[group_id];
            if (topk_heap.empty()) {
              topk_heap.limit(ctx->group_topk());
            }
            topk_heap.emplace_back(id, dist);
          }
        }
      }
      ctx->topk_to_result(q);
      query = static_cast<const char *>(query) + qmeta.element_size();
    }
  } else {
    for (size_t q = 0; q < count; ++q) {
      ctx->reset_query(query);
      ctx->topk_heap().clear();
      for (size_t idx = 0; idx < p_keys[q].size(); ++idx) {
        uint64_t pk = p_keys[q][idx];
        if (!ctx->filter().is_valid() || !ctx->filter()(pk)) {
          node_id_t id = entity_.get_id(pk);
          if (id != kInvalidNodeId) {
            dist_t dist = ctx->dist_calculator().batch_dist(id);
            ctx->topk_heap().emplace(id, dist);
          }
        }
      }
      ctx->topk_to_result(q);
      query = static_cast<const char *>(query) + qmeta.element_size();
    }
  }

  if (ailego_unlikely(ctx->error())) {
    return IndexError_Runtime;
  }

  return 0;
}

IndexSearcher::Context::Pointer HnswSearcher::create_context() const {
  if (ailego_unlikely(state_ != STATE_LOADED)) {
    LOG_ERROR("Load the index first before create context");
    return Context::Pointer();
  }
  const HnswEntity::Pointer search_ctx_entity = entity_.clone();
  if (!search_ctx_entity) {
    LOG_ERROR("Failed to create search context entity");
    return Context::Pointer();
  }
  HnswContext *ctx = new (std::nothrow)
      HnswContext(meta_.dimension(), metric_, search_ctx_entity);
  if (ailego_unlikely(ctx == nullptr)) {
    LOG_ERROR("Failed to new HnswContext");
    return Context::Pointer();
  }
  ctx->set_ef(ef_);
  ctx->set_max_scan_num(max_scan_num_);
  uint32_t filter_mode =
      bf_enabled_ ? VisitFilter::BloomFilter : VisitFilter::ByteMap;
  ctx->set_filter_mode(filter_mode);
  ctx->set_filter_negative_probility(bf_negative_probility_);
  ctx->set_magic(magic_);
  ctx->set_force_padding_topk(force_padding_topk_enabled_);
  ctx->set_bruteforce_threshold(bruteforce_threshold_);
  if (ailego_unlikely(ctx->init(HnswContext::kSearcherContext)) != 0) {
    LOG_ERROR("Init HnswContext failed");
    delete ctx;
    return Context::Pointer();
  }

  return Context::Pointer(ctx);
}

IndexProvider::Pointer HnswSearcher::create_provider(void) const {
  LOG_DEBUG("HnswSearcher create provider");

  auto entity = entity_.clone();
  if (ailego_unlikely(!entity)) {
    LOG_ERROR("Clone HnswEntity failed");
    return Provider::Pointer();
  }
  return Provider::Pointer(
      new (std::nothrow) HnswIndexProvider(meta_, entity, "HnswSearcher"));
}

const void *HnswSearcher::get_vector(uint64_t key) const {
  return entity_.get_vector_by_key(key);
}

INDEX_FACTORY_REGISTER_SEARCHER(HnswSearcher);

}  // namespace core
}  // namespace zvec
