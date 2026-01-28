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
#include "hnsw_streamer.h"
#include <iostream>
#include <ailego/internal/cpu_features.h>
#include <ailego/pattern/defer.h>
#include <ailego/utility/memory_helper.h>
#include "utility/sparse_utility.h"
#include "hnsw_algorithm.h"
#include "hnsw_context.h"
#include "hnsw_dist_calculator.h"
#include "hnsw_index_provider.h"

namespace zvec {
namespace core {

HnswStreamer::HnswStreamer() : entity_(stats_) {}

HnswStreamer::~HnswStreamer() {
  if (state_ == STATE_INITED) {
    this->cleanup();
  }
}

int HnswStreamer::init(const IndexMeta &imeta, const ailego::Params &params) {
  meta_ = imeta;
  meta_.set_streamer("HnswStreamer", HnswEntity::kRevision, params);

  params.get(PARAM_HNSW_STREAMER_MAX_INDEX_SIZE, &max_index_size_);

  params.get(PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT, &upper_max_neighbor_cnt_);
  float multiplier = HnswEntity::kDefaultL0MaxNeighborCntMultiplier;
  params.get(PARAM_HNSW_STREAMER_L0_MAX_NEIGHBOR_COUNT_MULTIPLIER, &multiplier);
  l0_max_neighbor_cnt_ = multiplier * upper_max_neighbor_cnt_;

  multiplier = HnswEntity::kDefaultNeighborPruneMultiplier;
  params.get(PARAM_HNSW_STREAMER_NEIGHBOR_PRUNE_MULTIPLIER, &multiplier);
  size_t prune_cnt = multiplier * upper_max_neighbor_cnt_;
  scaling_factor_ = upper_max_neighbor_cnt_;
  params.get(PARAM_HNSW_STREAMER_SCALING_FACTOR, &scaling_factor_);

  params.get(PARAM_HNSW_STREAMER_DOCS_HARD_LIMIT, &docs_hard_limit_);
  params.get(PARAM_HNSW_STREAMER_EF, &ef_);
  params.get(PARAM_HNSW_STREAMER_EFCONSTRUCTION, &ef_construction_);
  params.get(PARAM_HNSW_STREAMER_VISIT_BLOOMFILTER_ENABLE, &bf_enabled_);
  params.get(PARAM_HNSW_STREAMER_VISIT_BLOOMFILTER_NEGATIVE_PROB,
             &bf_negative_prob_);
  params.get(PARAM_HNSW_STREAMER_BRUTE_FORCE_THRESHOLD, &bruteforce_threshold_);
  params.get(PARAM_HNSW_STREAMER_MAX_SCAN_RATIO, &max_scan_ratio_);
  params.get(PARAM_HNSW_STREAMER_MAX_SCAN_LIMIT, &max_scan_limit_);
  params.get(PARAM_HNSW_STREAMER_MIN_SCAN_LIMIT, &min_scan_limit_);
  params.get(PARAM_HNSW_STREAMER_CHECK_CRC_ENABLE, &check_crc_enabled_);
  params.get(PARAM_HNSW_STREAMER_CHUNK_SIZE, &chunk_size_);
  params.get(PARAM_HNSW_STREAMER_FILTER_SAME_KEY, &filter_same_key_);
  params.get(PARAM_HNSW_STREAMER_GET_VECTOR_ENABLE, &get_vector_enabled_);
  params.get(PARAM_HNSW_STREAMER_MIN_NEIGHBOR_COUNT, &min_neighbor_cnt_);
  params.get(PARAM_HNSW_STREAMER_FORCE_PADDING_RESULT_ENABLE,
             &force_padding_topk_enabled_);
  params.get(PARAM_HNSW_STREAMER_USE_ID_MAP, &use_id_map_);
  entity_.set_use_key_info_map(use_id_map_);

  params.get(PARAM_HNSW_STREAMER_DOCS_SOFT_LIMIT, &docs_soft_limit_);
  if (docs_soft_limit_ > 0 && docs_soft_limit_ > docs_hard_limit_) {
    LOG_ERROR("[%s] must be >= [%s]",
              PARAM_HNSW_STREAMER_DOCS_HARD_LIMIT.c_str(),
              PARAM_HNSW_STREAMER_DOCS_SOFT_LIMIT.c_str());
    return IndexError_InvalidArgument;
  } else if (docs_soft_limit_ == 0UL) {
    docs_soft_limit_ =
        docs_hard_limit_ * HnswEntity::kDefaultDocsSoftLimitRatio;
  }

  if (ef_ == 0U) {
    ef_ = HnswEntity::kDefaultEf;
  }
  if (ef_construction_ == 0U) {
    ef_construction_ = HnswEntity::kDefaultEfConstruction;
  }
  if (upper_max_neighbor_cnt_ == 0U) {
    upper_max_neighbor_cnt_ = HnswEntity::kDefaultUpperMaxNeighborCnt;
  }
  if (upper_max_neighbor_cnt_ > HnswEntity::kMaxNeighborCnt) {
    LOG_ERROR("[%s] must be in range (0,%d)",
              PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT.c_str(),
              HnswEntity::kMaxNeighborCnt);
    return IndexError_InvalidArgument;
  }
  if (l0_max_neighbor_cnt_ == 0U) {
    l0_max_neighbor_cnt_ = HnswEntity::kDefaultL0MaxNeighborCnt;
  }
  if (l0_max_neighbor_cnt_ > HnswEntity::kMaxNeighborCnt) {
    LOG_ERROR("MaxL0NeighborCnt must be in range (0,%d)",
              HnswEntity::kMaxNeighborCnt);
    return IndexError_InvalidArgument;
  }
  if (min_neighbor_cnt_ > upper_max_neighbor_cnt_) {
    LOG_ERROR("[%s]-[%u] must be <= [%s]-[%u]",
              PARAM_HNSW_STREAMER_MIN_NEIGHBOR_COUNT.c_str(), min_neighbor_cnt_,
              PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT.c_str(),
              upper_max_neighbor_cnt_);
    return IndexError_InvalidArgument;
  }

  if (bf_negative_prob_ <= 0.0f || bf_negative_prob_ >= 1.0f) {
    LOG_ERROR("[%s] must be in range (0,1)",
              PARAM_HNSW_STREAMER_VISIT_BLOOMFILTER_NEGATIVE_PROB.c_str());
    return IndexError_InvalidArgument;
  }

  if (scaling_factor_ == 0U) {
    scaling_factor_ = HnswEntity::kDefaultScalingFactor;
  }
  if (scaling_factor_ < 5 || scaling_factor_ > 1000) {
    LOG_ERROR("[%s] must be in range [5,1000]",
              PARAM_HNSW_STREAMER_SCALING_FACTOR.c_str());
    return IndexError_InvalidArgument;
  }

  if (max_scan_ratio_ <= 0.0f || max_scan_ratio_ > 1.0f) {
    LOG_ERROR("[%s] must be in range (0.0f,1.0f]",
              PARAM_HNSW_STREAMER_MAX_SCAN_RATIO.c_str());
    return IndexError_InvalidArgument;
  }

  if (max_scan_limit_ < min_scan_limit_) {
    LOG_ERROR("[%s] must be >= [%s]",
              PARAM_HNSW_STREAMER_MAX_SCAN_LIMIT.c_str(),
              PARAM_HNSW_STREAMER_MIN_SCAN_LIMIT.c_str());
    return IndexError_InvalidArgument;
  }

  if (prune_cnt == 0UL) {
    prune_cnt = upper_max_neighbor_cnt_;
  }
  if (chunk_size_ == 0UL) {
    chunk_size_ = HnswEntity::kDefaultChunkSize;
  }
  if (chunk_size_ > HnswEntity::kMaxChunkSize) {
    LOG_ERROR("[%s] must be < %zu", PARAM_HNSW_STREAMER_CHUNK_SIZE.c_str(),
              HnswEntity::kMaxChunkSize);
    return IndexError_InvalidArgument;
  }

  entity_.set_ef_construction(ef_construction_);
  entity_.set_upper_neighbor_cnt(upper_max_neighbor_cnt_);
  entity_.set_l0_neighbor_cnt(l0_max_neighbor_cnt_);
  entity_.set_scaling_factor(scaling_factor_);
  entity_.set_prune_cnt(prune_cnt);

  entity_.set_vector_size(meta_.element_size());

  entity_.set_chunk_size(chunk_size_);
  entity_.set_filter_same_key(filter_same_key_);
  entity_.set_get_vector(get_vector_enabled_);
  entity_.set_min_neighbor_cnt(min_neighbor_cnt_);

  int ret = entity_.init(docs_hard_limit_);
  if (ret != 0) {
    LOG_ERROR("Hnsw entity init failed for %s", IndexError::What(ret));
    return ret;
  }

  LOG_DEBUG(
      "Init params: maxIndexSize=%zu docsHardLimit=%zu docsSoftLimit=%zu "
      "efConstruction=%u ef=%u upperMaxNeighborCnt=%u l0MaxNeighborCnt=%u "
      "scalingFactor=%u maxScanRatio=%.3f minScanLimit=%zu maxScanLimit=%zu "
      "bfEnabled=%d bruteFoceThreshold=%zu bfNegativeProbility=%.5f "
      "checkCrcEnabled=%d pruneSize=%zu vectorSize=%u chunkSize=%zu "
      "filterSameKey=%u getVectorEnabled=%u minNeighborCount=%u "
      "forcePadding=%u ",
      max_index_size_, docs_hard_limit_, docs_soft_limit_, ef_construction_,
      ef_, upper_max_neighbor_cnt_, l0_max_neighbor_cnt_, scaling_factor_,
      max_scan_ratio_, min_scan_limit_, max_scan_limit_, bf_enabled_,
      bruteforce_threshold_, bf_negative_prob_, check_crc_enabled_, prune_cnt,
      meta_.element_size(), chunk_size_, filter_same_key_, get_vector_enabled_,
      min_neighbor_cnt_, force_padding_topk_enabled_);

  alg_ = HnswAlgorithm::UPointer(new HnswAlgorithm(entity_));

  ret = alg_->init();
  if (ret != 0) {
    return ret;
  }

  state_ = STATE_INITED;

  return 0;
}

int HnswStreamer::cleanup(void) {
  if (state_ == STATE_OPENED) {
    this->close();
  }

  LOG_INFO("HnswStreamer cleanup");

  meta_.clear();
  metric_.reset();
  stats_.clear();
  entity_.cleanup();

  if (alg_) {
    alg_->cleanup();
  }

  max_index_size_ = 0UL;
  docs_hard_limit_ = HnswEntity::kDefaultDocsHardLimit;
  docs_soft_limit_ = 0UL;
  upper_max_neighbor_cnt_ = HnswEntity::kDefaultUpperMaxNeighborCnt;
  l0_max_neighbor_cnt_ = HnswEntity::kDefaultL0MaxNeighborCnt;
  ef_ = HnswEntity::kDefaultEf;
  ef_construction_ = HnswEntity::kDefaultEfConstruction;
  bf_enabled_ = false;
  scaling_factor_ = HnswEntity::kDefaultScalingFactor;
  bruteforce_threshold_ = HnswEntity::kDefaultBruteForceThreshold;
  max_scan_limit_ = HnswEntity::kDefaultMaxScanLimit;
  min_scan_limit_ = HnswEntity::kDefaultMinScanLimit;
  chunk_size_ = HnswEntity::kDefaultChunkSize;
  bf_negative_prob_ = HnswEntity::kDefaultBFNegativeProbility;
  max_scan_ratio_ = HnswEntity::kDefaultScanRatio;
  state_ = STATE_INIT;
  check_crc_enabled_ = false;
  filter_same_key_ = false;
  get_vector_enabled_ = false;

  return 0;
}

int HnswStreamer::open(IndexStorage::Pointer stg) {
  LOG_INFO("HnswStreamer open");

  if (ailego_unlikely(state_ != STATE_INITED)) {
    LOG_ERROR("Open storage failed, init streamer first!");
    return IndexError_NoReady;
  }
  int ret = entity_.open(std::move(stg), max_index_size_, check_crc_enabled_);
  if (ret != 0) {
    return ret;
  }
  IndexMeta index_meta;
  ret = entity_.get_index_meta(&index_meta);
  if (ret == IndexError_NoExist) {
    // Set IndexMeta for the new index
    ret = entity_.set_index_meta(meta_);
    if (ret != 0) {
      LOG_ERROR("Failed to set index meta for %s", IndexError::What(ret));
      return ret;
    }
  } else if (ret != 0) {
    LOG_ERROR("Failed to get index meta for %s", IndexError::What(ret));
    return ret;
  } else {
    if (index_meta.dimension() != meta_.dimension() ||
        index_meta.element_size() != meta_.element_size() ||
        index_meta.metric_name() != meta_.metric_name() ||
        index_meta.data_type() != meta_.data_type()) {
      LOG_ERROR("IndexMeta mismatch from the previous in index");
      return IndexError_Mismatch;
    }
    // The IndexMetric Params may be updated like MipsSquaredEuclidean
    auto metric_params = index_meta.metric_params();
    metric_params.merge(meta_.metric_params());
    meta_.set_metric(index_meta.metric_name(), 0, metric_params);
  }

  metric_ = IndexFactory::CreateMetric(meta_.metric_name());
  if (!metric_) {
    LOG_ERROR("Failed to create metric %s", meta_.metric_name().c_str());
    return IndexError_NoExist;
  }
  ret = metric_->init(meta_, meta_.metric_params());
  if (ret != 0) {
    LOG_ERROR("Failled to init metric, ret=%d", ret);
    return ret;
  }

  if (!metric_->distance()) {
    LOG_ERROR("Invalid metric distance");
    return IndexError_InvalidArgument;
  }

  if (!metric_->batch_distance()) {
    LOG_ERROR("Invalid metric batch distance");
    return IndexError_InvalidArgument;
  }

  add_distance_ = metric_->distance();
  add_batch_distance_ = metric_->batch_distance();

  search_distance_ = add_distance_;
  search_batch_distance_ = add_batch_distance_;

  if (metric_->query_metric() && metric_->query_metric()->distance() &&
      metric_->query_metric()->batch_distance()) {
    search_distance_ = metric_->query_metric()->distance();
    search_batch_distance_ = metric_->query_metric()->batch_distance();
  }

  state_ = STATE_OPENED;
  magic_ = IndexContext::GenerateMagic();

  return 0;
}

int HnswStreamer::close(void) {
  LOG_INFO("HnswStreamer close");

  stats_.clear();
  meta_.set_metric(metric_->name(), 0, metric_->params());
  entity_.set_index_meta(meta_);
  int ret = entity_.close();
  if (ret != 0) {
    return ret;
  }
  state_ = STATE_INITED;

  return 0;
}

int HnswStreamer::flush(uint64_t checkpoint) {
  LOG_INFO("HnswStreamer flush checkpoint=%zu", (size_t)checkpoint);

  meta_.set_metric(metric_->name(), 0, metric_->params());
  entity_.set_index_meta(meta_);
  return entity_.flush(checkpoint);
}

int HnswStreamer::dump(const IndexDumper::Pointer &dumper) {
  LOG_INFO("HnswStreamer dump");

  shared_mutex_.lock();
  AILEGO_DEFER([&]() { shared_mutex_.unlock(); });

  meta_.set_searcher("HnswSearcher", HnswEntity::kRevision, ailego::Params());

  int ret = IndexHelper::SerializeToDumper(meta_, dumper.get());
  if (ret != 0) {
    LOG_ERROR("Failed to serialize meta into dumper.");
    return ret;
  }
  return entity_.dump(dumper);
}

IndexStreamer::Context::Pointer HnswStreamer::create_context(void) const {
  if (ailego_unlikely(state_ != STATE_OPENED)) {
    LOG_ERROR("Create context failed, open storage first!");
    return Context::Pointer();
  }

  HnswEntity::Pointer entity = entity_.clone();
  if (ailego_unlikely(!entity)) {
    LOG_ERROR("CreateContext clone init failed");
    return Context::Pointer();
  }
  HnswContext *ctx =
      new (std::nothrow) HnswContext(meta_.dimension(), metric_, entity);
  if (ailego_unlikely(ctx == nullptr)) {
    LOG_ERROR("Failed to new HnswContext");
    return Context::Pointer();
  }
  ctx->set_ef(ef_);
  ctx->set_max_scan_limit(max_scan_limit_);
  ctx->set_min_scan_limit(min_scan_limit_);
  ctx->set_max_scan_ratio(max_scan_ratio_);
  ctx->set_filter_mode(bf_enabled_ ? VisitFilter::BloomFilter
                                   : VisitFilter::ByteMap);
  ctx->set_filter_negative_probility(bf_negative_prob_);
  ctx->set_magic(magic_);
  ctx->set_force_padding_topk(force_padding_topk_enabled_);
  ctx->set_bruteforce_threshold(bruteforce_threshold_);

  if (ailego_unlikely(ctx->init(HnswContext::kStreamerContext)) != 0) {
    LOG_ERROR("Init HnswContext failed");
    delete ctx;
    return Context::Pointer();
  }
  uint32_t estimate_doc_count = 0;
  if (meta_.streamer_params().get(PARAM_HNSW_STREAMER_ESTIMATE_DOC_COUNT,
                                  &estimate_doc_count)) {
    LOG_DEBUG("HnswStreamer doc_count[%zu] estimate[%zu]",
              (size_t)entity_.doc_cnt(), (size_t)estimate_doc_count);
  }
  ctx->check_need_adjuct_ctx(std::max(entity_.doc_cnt(), estimate_doc_count));

  return Context::Pointer(ctx);
}

IndexProvider::Pointer HnswStreamer::create_provider(void) const {
  LOG_DEBUG("HnswStreamer create provider");

  auto entity = entity_.clone();
  if (ailego_unlikely(!entity)) {
    LOG_ERROR("Clone HnswEntity failed");
    return nullptr;
  }
  return Provider::Pointer(
      new HnswIndexProvider(meta_, entity, "HnswStreamer"));
}

int HnswStreamer::update_context(HnswContext *ctx) const {
  const HnswEntity::Pointer entity = entity_.clone();
  if (!entity) {
    LOG_ERROR("Failed to clone search context entity");
    return IndexError_Runtime;
  }
  ctx->set_max_scan_limit(max_scan_limit_);
  ctx->set_min_scan_limit(min_scan_limit_);
  ctx->set_max_scan_ratio(max_scan_ratio_);
  ctx->set_bruteforce_threshold(bruteforce_threshold_);
  return ctx->update_context(HnswContext::kStreamerContext, meta_, metric_,
                             entity, magic_);
}

//! Add a vector with id into index
int HnswStreamer::add_with_id_impl(uint32_t id, const void *query,
                                   const IndexQueryMeta &qmeta,
                                   IndexStreamer::Context::Pointer &context) {
  int ret = check_params(query, qmeta);
  if (ailego_unlikely(ret != 0)) {
    return ret;
  }

  HnswContext *ctx = dynamic_cast<HnswContext *>(context.get());
  ailego_do_if_false(ctx) {
    LOG_ERROR("Cast context to HnswContext failed");
    return IndexError_Cast;
  }
  if (ctx->magic() != magic_) {
    //! context is created by another searcher or streamer
    ret = update_context(ctx);
    if (ret != 0) {
      return ret;
    }
  }

  if (ailego_unlikely(entity_.doc_cnt() >= docs_soft_limit_)) {
    if (entity_.doc_cnt() >= docs_hard_limit_) {
      LOG_ERROR("Current docs %u exceed [%s]", entity_.doc_cnt(),
                PARAM_HNSW_STREAMER_DOCS_HARD_LIMIT.c_str());
      const std::lock_guard<std::mutex> lk(mutex_);
      (*stats_.mutable_discarded_count())++;
      return IndexError_IndexFull;
    } else {
      LOG_WARN("Current docs %u exceed [%s]", entity_.doc_cnt(),
               PARAM_HNSW_STREAMER_DOCS_SOFT_LIMIT.c_str());
    }
  }
  if (ailego_unlikely(!shared_mutex_.try_lock_shared())) {
    LOG_ERROR("Cannot add vector while dumping index");
    (*stats_.mutable_discarded_count())++;
    return IndexError_Unsupported;
  }
  AILEGO_DEFER([&]() { shared_mutex_.unlock_shared(); });

  ctx->clear();
  ctx->update_dist_caculator_distance(add_distance_, add_batch_distance_);
  ctx->reset_query(query);
  ctx->check_need_adjuct_ctx(entity_.doc_cnt());

  if (metric_->support_train()) {
    const std::lock_guard<std::mutex> lk(mutex_);
    ret = metric_->train(query, meta_.dimension());
    if (ailego_unlikely(ret != 0)) {
      LOG_ERROR("Hnsw streamer metric train failed");
      (*stats_.mutable_discarded_count())++;
      return ret;
    }
  }

  level_t level = alg_->get_random_level();
  ret = entity_.add_vector_with_id(level, id, query);
  if (ailego_unlikely(ret != 0)) {
    LOG_ERROR("Hnsw streamer add vector failed");
    (*stats_.mutable_discarded_count())++;
    return ret;
  }

  ret = alg_->add_node(id, level, ctx);
  if (ailego_unlikely(ret != 0)) {
    LOG_ERROR("Hnsw stramer add node failed");
    (*stats_.mutable_discarded_count())++;
    return ret;
  }

  if (ailego_unlikely(ctx->error())) {
    (*stats_.mutable_discarded_count())++;
    return IndexError_Runtime;
  }
  (*stats_.mutable_added_count())++;

  return 0;
}

//! Add a vector into index
int HnswStreamer::add_impl(uint64_t pkey, const void *query,
                           const IndexQueryMeta &qmeta,
                           IndexStreamer::Context::Pointer &context) {
  int ret = check_params(query, qmeta);
  if (ailego_unlikely(ret != 0)) {
    return ret;
  }

  HnswContext *ctx = dynamic_cast<HnswContext *>(context.get());
  ailego_do_if_false(ctx) {
    LOG_ERROR("Cast context to HnswContext failed");
    return IndexError_Cast;
  }
  if (ctx->magic() != magic_) {
    //! context is created by another searcher or streamer
    ret = update_context(ctx);
    if (ret != 0) {
      return ret;
    }
  }

  if (ailego_unlikely(entity_.doc_cnt() >= docs_soft_limit_)) {
    if (entity_.doc_cnt() >= docs_hard_limit_) {
      LOG_ERROR("Current docs %u exceed [%s]", entity_.doc_cnt(),
                PARAM_HNSW_STREAMER_DOCS_HARD_LIMIT.c_str());
      const std::lock_guard<std::mutex> lk(mutex_);
      (*stats_.mutable_discarded_count())++;
      return IndexError_IndexFull;
    } else {
      LOG_WARN("Current docs %u exceed [%s]", entity_.doc_cnt(),
               PARAM_HNSW_STREAMER_DOCS_SOFT_LIMIT.c_str());
    }
  }
  if (ailego_unlikely(!shared_mutex_.try_lock_shared())) {
    LOG_ERROR("Cannot add vector while dumping index");
    (*stats_.mutable_discarded_count())++;
    return IndexError_Unsupported;
  }
  AILEGO_DEFER([&]() { shared_mutex_.unlock_shared(); });

  ctx->clear();
  ctx->update_dist_caculator_distance(add_distance_, add_batch_distance_);
  ctx->reset_query(query);
  ctx->check_need_adjuct_ctx(entity_.doc_cnt());

  if (metric_->support_train()) {
    const std::lock_guard<std::mutex> lk(mutex_);
    ret = metric_->train(query, meta_.dimension());
    if (ailego_unlikely(ret != 0)) {
      LOG_ERROR("Hnsw streamer metric train failed");
      (*stats_.mutable_discarded_count())++;
      return ret;
    }
  }

  level_t level = alg_->get_random_level();
  node_id_t id;
  ret = entity_.add_vector(level, pkey, query, &id);
  if (ailego_unlikely(ret != 0)) {
    LOG_ERROR("Hnsw streamer add vector failed");
    (*stats_.mutable_discarded_count())++;
    return ret;
  }

  ret = alg_->add_node(id, level, ctx);
  if (ailego_unlikely(ret != 0)) {
    LOG_ERROR("Hnsw stramer add node failed");
    (*stats_.mutable_discarded_count())++;
    return ret;
  }

  if (ailego_unlikely(ctx->error())) {
    (*stats_.mutable_discarded_count())++;
    return IndexError_Runtime;
  }
  (*stats_.mutable_added_count())++;

  return 0;
}


int HnswStreamer::search_impl(const void *query, const IndexQueryMeta &qmeta,
                              IndexStreamer::Context::Pointer &context) const {
  return search_impl(query, qmeta, 1, context);
}

//! Similarity search
int HnswStreamer::search_impl(const void *query, const IndexQueryMeta &qmeta,
                              uint32_t count,
                              IndexStreamer::Context::Pointer &context) const {
  int ret = check_params(query, qmeta);
  if (ailego_unlikely(ret != 0)) {
    return ret;
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
    ret = update_context(ctx);
    if (ret != 0) {
      return ret;
    }
  }

  ctx->clear();
  ctx->update_dist_caculator_distance(search_distance_, search_batch_distance_);
  ctx->resize_results(count);
  ctx->check_need_adjuct_ctx(entity_.doc_cnt());
  for (size_t q = 0; q < count; ++q) {
    ctx->reset_query(query);
    ret = alg_->search(ctx);
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

void HnswStreamer::print_debug_info() {
  for (node_id_t id = 0; id < entity_.doc_cnt(); ++id) {
    if (entity_.get_key(id) == kInvalidKey) {
      continue;
    }
    Neighbors neighbours = entity_.get_neighbors(0, id);
    std::cout << "node: " << id << "; ";
    if (neighbours.size() == 0) std::cout << std::endl;
    for (uint32_t i = 0; i < neighbours.size(); ++i) {
      std::cout << neighbours[i];

      if (i == neighbours.size() - 1) {
        std::cout << std::endl;
      } else {
        std::cout << ", ";
      }
    }
  }

  // entity_.print_key_map();
}

int HnswStreamer::search_bf_impl(
    const void *query, const IndexQueryMeta &qmeta,
    IndexStreamer::Context::Pointer &context) const {
  return search_bf_impl(query, qmeta, 1, context);
}

int HnswStreamer::search_bf_impl(
    const void *query, const IndexQueryMeta &qmeta, uint32_t count,
    IndexStreamer::Context::Pointer &context) const {
  int ret = check_params(query, qmeta);
  if (ailego_unlikely(ret != 0)) {
    return ret;
  }
  HnswContext *ctx = dynamic_cast<HnswContext *>(context.get());
  ailego_do_if_false(ctx) {
    LOG_ERROR("Cast context to HnswContext failed");
    return IndexError_Cast;
  }
  if (ctx->magic() != magic_) {
    //! context is created by another searcher or streamer
    ret = update_context(ctx);
    if (ret != 0) {
      return ret;
    }
  }

  ctx->clear();
  ctx->update_dist_caculator_distance(search_distance_, search_batch_distance_);
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
    auto &filter = ctx->filter();
    auto &topk = ctx->topk_heap();

    for (size_t q = 0; q < count; ++q) {
      ctx->reset_query(query);
      topk.clear();
      for (node_id_t id = 0; id < entity_.doc_cnt(); ++id) {
        if (entity_.get_key(id) == kInvalidKey) {
          continue;
        }

        if (!filter.is_valid() || !filter(entity_.get_key(id))) {
          dist_t dist = ctx->dist_calculator().batch_dist(id);
          topk.emplace(id, dist);
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

int HnswStreamer::search_bf_by_p_keys_impl(
    const void *query, const std::vector<std::vector<uint64_t>> &p_keys,
    const IndexQueryMeta &qmeta, uint32_t count,
    Context::Pointer &context) const {
  int ret = check_params(query, qmeta);
  if (ailego_unlikely(ret != 0)) {
    return ret;
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
    ret = update_context(ctx);
    if (ret != 0) {
      return ret;
    }
  }

  ctx->clear();
  ctx->update_dist_caculator_distance(search_distance_, search_batch_distance_);
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
    auto &filter = ctx->filter();
    auto &topk = ctx->topk_heap();

    for (size_t q = 0; q < count; ++q) {
      ctx->reset_query(query);
      topk.clear();
      for (size_t idx = 0; idx < p_keys[q].size(); ++idx) {
        key_t pk = p_keys[q][idx];
        if (!filter.is_valid() || !filter(pk)) {
          node_id_t id = entity_.get_id(pk);
          if (id != kInvalidNodeId) {
            dist_t dist = ctx->dist_calculator().batch_dist(id);
            topk.emplace(id, dist);
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


INDEX_FACTORY_REGISTER_STREAMER(HnswStreamer);

}  // namespace core
}  // namespace zvec