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
#include "hnsw_sparse_streamer.h"
#include <iostream>
#include <ailego/internal/cpu_features.h>
#include <ailego/pattern/defer.h>
#include <ailego/utility/memory_helper.h>
#include "hnsw_sparse_algorithm.h"
#include "hnsw_sparse_context.h"
#include "hnsw_sparse_dist_calculator.h"
#include "hnsw_sparse_index_provider.h"

namespace zvec {
namespace core {

HnswSparseStreamer::HnswSparseStreamer() : entity_(stats_) {}

HnswSparseStreamer::~HnswSparseStreamer() {
  if (state_ == STATE_INITED) {
    this->cleanup();
  }
}

int HnswSparseStreamer::init(const IndexMeta &imeta,
                             const ailego::Params &params) {
  meta_ = imeta;
  meta_.set_streamer("HnswSparseStreamer", HnswSparseEntity::kRevision, params);

  params.get(PARAM_HNSW_SPARSE_STREAMER_MAX_INDEX_SIZE, &max_index_size_);
  params.get(PARAM_HNSW_SPARSE_STREAMER_MAX_NEIGHBOR_COUNT,
             &upper_max_neighbor_cnt_);
  float multiplier = HnswSparseEntity::kDefaultL0MaxNeighborCntMultiplier;
  params.get(PARAM_HNSW_SPARSE_STREAMER_L0_MAX_NEIGHBOR_COUNT_MULTIPLIER,
             &multiplier);
  l0_max_neighbor_cnt_ = multiplier * upper_max_neighbor_cnt_;

  multiplier = HnswSparseEntity::kDefaultNeighborPruneMultiplier;
  params.get(PARAM_HNSW_SPARSE_STREAMER_NEIGHBOR_PRUNE_MULTIPLIER, &multiplier);
  size_t prune_cnt = multiplier * upper_max_neighbor_cnt_;
  scaling_factor_ = upper_max_neighbor_cnt_;
  params.get(PARAM_HNSW_SPARSE_STREAMER_SCALING_FACTOR, &scaling_factor_);

  params.get(PARAM_HNSW_SPARSE_STREAMER_DOCS_HARD_LIMIT, &docs_hard_limit_);
  params.get(PARAM_HNSW_SPARSE_STREAMER_EF, &ef_);
  params.get(PARAM_HNSW_SPARSE_STREAMER_EFCONSTRUCTION, &ef_construction_);
  params.get(PARAM_HNSW_SPARSE_STREAMER_VISIT_BLOOMFILTER_ENABLE, &bf_enabled_);
  params.get(PARAM_HNSW_SPARSE_STREAMER_VISIT_BLOOMFILTER_NEGATIVE_PROB,
             &bf_negative_prob_);
  params.get(PARAM_HNSW_SPARSE_STREAMER_BRUTE_FORCE_THRESHOLD,
             &bruteforce_threshold_);
  params.get(PARAM_HNSW_SPARSE_STREAMER_MAX_SCAN_RATIO, &max_scan_ratio_);
  params.get(PARAM_HNSW_SPARSE_STREAMER_MAX_SCAN_LIMIT, &max_scan_limit_);
  params.get(PARAM_HNSW_SPARSE_STREAMER_MIN_SCAN_LIMIT, &min_scan_limit_);
  params.get(PARAM_HNSW_SPARSE_STREAMER_CHECK_CRC_ENABLE, &check_crc_enabled_);

  params.get(PARAM_HNSW_SPARSE_STREAMER_CHUNK_SIZE, &chunk_size_);
  params.get(PARAM_HNSW_SPARSE_STREAMER_FILTER_SAME_KEY, &filter_same_key_);
  params.get(PARAM_HNSW_SPARSE_STREAMER_GET_VECTOR_ENABLE,
             &get_vector_enabled_);
  params.get(PARAM_HNSW_SPARSE_STREAMER_MIN_NEIGHBOR_COUNT, &min_neighbor_cnt_);
  params.get(PARAM_HNSW_SPARSE_STREAMER_FORCE_PADDING_RESULT_ENABLE,
             &force_padding_topk_enabled_);

  query_filtering_enabled_ =
      params.get(PARAM_HNSW_SPARSE_STREAMER_QUERY_FILTERING_RATIO,
                 &query_filtering_ratio_);

  params.get(PARAM_HNSW_SPARSE_STREAMER_DOCS_SOFT_LIMIT, &docs_soft_limit_);
  if (docs_soft_limit_ > 0 && docs_soft_limit_ > docs_hard_limit_) {
    LOG_ERROR("[%s] must be >= [%s]",
              PARAM_HNSW_SPARSE_STREAMER_DOCS_HARD_LIMIT.c_str(),
              PARAM_HNSW_SPARSE_STREAMER_DOCS_SOFT_LIMIT.c_str());
    return IndexError_InvalidArgument;
  } else if (docs_soft_limit_ == 0UL) {
    docs_soft_limit_ =
        docs_hard_limit_ * HnswSparseEntity::kDefaultDocsSoftLimitRatio;
  }

  if (ef_ == 0U) {
    ef_ = HnswSparseEntity::kDefaultEf;
  }
  if (ef_construction_ == 0U) {
    ef_construction_ = HnswSparseEntity::kDefaultEfConstruction;
  }
  if (upper_max_neighbor_cnt_ == 0U) {
    upper_max_neighbor_cnt_ = HnswSparseEntity::kDefaultUpperMaxNeighborCnt;
  }
  if (upper_max_neighbor_cnt_ > HnswSparseEntity::kMaxNeighborCnt) {
    LOG_ERROR("[%s] must be in range (0,%d)",
              PARAM_HNSW_SPARSE_STREAMER_MAX_NEIGHBOR_COUNT.c_str(),
              HnswSparseEntity::kMaxNeighborCnt);
    return IndexError_InvalidArgument;
  }
  if (l0_max_neighbor_cnt_ == 0U) {
    l0_max_neighbor_cnt_ = HnswSparseEntity::kDefaultL0MaxNeighborCnt;
  }
  if (l0_max_neighbor_cnt_ > HnswSparseEntity::kMaxNeighborCnt) {
    LOG_ERROR("UpperNeighborCnt must be in range (0,%d)",
              HnswSparseEntity::kMaxNeighborCnt);
    return IndexError_InvalidArgument;
  }
  if (min_neighbor_cnt_ > upper_max_neighbor_cnt_) {
    LOG_ERROR("[%s]-[%u] must be <= [%s]-[%u]",
              PARAM_HNSW_SPARSE_STREAMER_MIN_NEIGHBOR_COUNT.c_str(),
              min_neighbor_cnt_,
              PARAM_HNSW_SPARSE_STREAMER_MAX_NEIGHBOR_COUNT.c_str(),
              upper_max_neighbor_cnt_);
    return IndexError_InvalidArgument;
  }

  if (bf_negative_prob_ <= 0.0f || bf_negative_prob_ >= 1.0f) {
    LOG_ERROR(
        "[%s] must be in range (0,1)",
        PARAM_HNSW_SPARSE_STREAMER_VISIT_BLOOMFILTER_NEGATIVE_PROB.c_str());
    return IndexError_InvalidArgument;
  }

  if (scaling_factor_ == 0U) {
    scaling_factor_ = HnswSparseEntity::kDefaultScalingFactor;
  }
  if (scaling_factor_ < 5 || scaling_factor_ > 1000) {
    LOG_ERROR("[%s] must be in range [5,1000]",
              PARAM_HNSW_SPARSE_STREAMER_SCALING_FACTOR.c_str());
    return IndexError_InvalidArgument;
  }

  if (max_scan_ratio_ <= 0.0f || max_scan_ratio_ > 1.0f) {
    LOG_ERROR("[%s] must be in range (0.0f,1.0f]",
              PARAM_HNSW_SPARSE_STREAMER_MAX_SCAN_RATIO.c_str());
    return IndexError_InvalidArgument;
  }

  if (max_scan_limit_ < min_scan_limit_) {
    LOG_ERROR("[%s] must be >= [%s]",
              PARAM_HNSW_SPARSE_STREAMER_MAX_SCAN_LIMIT.c_str(),
              PARAM_HNSW_SPARSE_STREAMER_MIN_SCAN_LIMIT.c_str());
    return IndexError_InvalidArgument;
  }

  if (prune_cnt == 0UL) {
    prune_cnt = upper_max_neighbor_cnt_;
  }
  if (chunk_size_ == 0UL) {
    chunk_size_ = HnswSparseEntity::kDefaultChunkSize;
  }
  if (chunk_size_ > HnswSparseEntity::kMaxChunkSize) {
    LOG_ERROR("[%s] must be < %zu",
              PARAM_HNSW_SPARSE_STREAMER_CHUNK_SIZE.c_str(),
              HnswSparseEntity::kMaxChunkSize);
    return IndexError_InvalidArgument;
  }

  if (query_filtering_enabled_ &&
      (query_filtering_ratio_ <= 0.0f || query_filtering_ratio_ >= 1.0f)) {
    LOG_ERROR("[%s] must be in range (0, 1)",
              PARAM_HNSW_SPARSE_SEARCHER_QUERY_FILTERING_RATIO.c_str());
    return IndexError_InvalidArgument;
  }

  entity_.set_ef_construction(ef_construction_);
  entity_.set_l0_neighbor_cnt(l0_max_neighbor_cnt_);
  entity_.set_upper_neighbor_cnt(upper_max_neighbor_cnt_);
  entity_.set_scaling_factor(scaling_factor_);
  entity_.set_prune_cnt(prune_cnt);

  entity_.set_chunk_size(chunk_size_);
  entity_.set_filter_same_key(filter_same_key_);
  entity_.set_get_vector(get_vector_enabled_);
  entity_.set_min_neighbor_cnt(min_neighbor_cnt_);

  entity_.set_sparse_meta_size(HnswSparseEntity::kSparseMetaSize);
  entity_.set_sparse_unit_size(meta_.unit_size());

  int ret = entity_.init(max_index_size_, docs_hard_limit_);
  if (ret != 0) {
    LOG_ERROR("Hnsw entity init failed for %s", IndexError::What(ret));
    return ret;
  }
  LOG_DEBUG(
      "Init params: maxIndexSize=%zu docsHardLimit=%zu docsSoftLimit=%zu "
      "efConstruction=%u ef=%u l0NeighborCnt=%u upperNeighborCnt=%u "
      "scalingFactor=%u maxScanRatio=%.3f minScanLimit=%zu maxScanLimit=%zu "
      "bfEnabled=%d bruteFoceThreshold=%zu bfNegativeProbability=%.5f "
      "checkCrcEnabled=%d pruneSize=%zu chunkSize=%zu "
      "filterSameKey=%u getVectorEnabled=%u "
      "minNeighborCount=%u forcePadding=%u filteringRatio=%f",
      max_index_size_, docs_hard_limit_, docs_soft_limit_, ef_construction_,
      ef_, l0_max_neighbor_cnt_, upper_max_neighbor_cnt_, scaling_factor_,
      max_scan_ratio_, min_scan_limit_, max_scan_limit_, bf_enabled_,
      bruteforce_threshold_, bf_negative_prob_, check_crc_enabled_, prune_cnt,
      chunk_size_, filter_same_key_, get_vector_enabled_, min_neighbor_cnt_,
      force_padding_topk_enabled_, query_filtering_ratio_);

  alg_ = HnswSparseAlgorithm::UPointer(new HnswSparseAlgorithm(entity_));

  ret = alg_->init();
  if (ret != 0) {
    return ret;
  }

  state_ = STATE_INITED;

  return 0;
}

int HnswSparseStreamer::cleanup(void) {
  if (state_ == STATE_OPENED) {
    this->close();
  }

  LOG_INFO("HnswSparseStreamer cleanup");

  meta_.clear();
  metric_.reset();
  stats_.clear();
  entity_.cleanup();

  if (alg_) {
    alg_->cleanup();
  }

  max_index_size_ = 0UL;
  docs_hard_limit_ = HnswSparseEntity::kDefaultDocsHardLimit;
  docs_soft_limit_ = 0UL;
  upper_max_neighbor_cnt_ = HnswSparseEntity::kDefaultUpperMaxNeighborCnt;
  ef_ = HnswSparseEntity::kDefaultEf;
  ef_construction_ = HnswSparseEntity::kDefaultEfConstruction;
  bf_enabled_ = false;
  scaling_factor_ = HnswSparseEntity::kDefaultScalingFactor;
  bruteforce_threshold_ = HnswSparseEntity::kDefaultBruteForceThreshold;
  max_scan_limit_ = HnswSparseEntity::kDefaultMaxScanLimit;
  min_scan_limit_ = HnswSparseEntity::kDefaultMinScanLimit;
  chunk_size_ = HnswSparseEntity::kDefaultChunkSize;
  bf_negative_prob_ = HnswSparseEntity::kDefaultBFNegativeProbability;
  max_scan_ratio_ = HnswSparseEntity::kDefaultScanRatio;
  state_ = STATE_INIT;
  check_crc_enabled_ = false;
  filter_same_key_ = false;
  get_vector_enabled_ = false;

  sparse_neighbor_ratio_ = HnswSparseEntity::kDefaultSparseNeighborRatio;
  sparse_neighbor_cnt_ = 0UL;
  sparse_min_neighbor_cnt_ = 0UL;
  upper_sparse_neighbor_cnt_ = 0UL;

  return 0;
}

int HnswSparseStreamer::open(IndexStorage::Pointer stg) {
  LOG_INFO("HnswSparseStreamer open");

  if (ailego_unlikely(state_ != STATE_INITED)) {
    LOG_ERROR("Open storage failed, init streamer first!");
    return IndexError_NoReady;
  }
  int ret = entity_.open(std::move(stg), check_crc_enabled_);
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
    if (index_meta.metric_name() != meta_.metric_name() ||
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

  if (!metric_->sparse_distance()) {
    LOG_ERROR("Invalid metric distance");
    return IndexError_InvalidArgument;
  }

  add_distance_ = metric_->sparse_distance();
  search_distance_ = add_distance_;

  if (metric_->query_metric() && metric_->query_metric()->distance()) {
    search_distance_ = metric_->query_metric()->sparse_distance();
  }

  state_ = STATE_OPENED;
  magic_ = IndexContext::GenerateMagic();

  return 0;
}

int HnswSparseStreamer::close(void) {
  LOG_INFO("HnswSparseStreamer close");

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

int HnswSparseStreamer::flush(uint64_t checkpoint) {
  LOG_INFO("HnswSparseStreamer flush checkpoint=%zu", (size_t)checkpoint);

  meta_.set_metric(metric_->name(), 0, metric_->params());
  entity_.set_index_meta(meta_);
  return entity_.flush(checkpoint);
}

int HnswSparseStreamer::dump(const IndexDumper::Pointer &dumper) {
  LOG_INFO("HnswSparseStreamer dump");

  shared_mutex_.lock();
  AILEGO_DEFER([&]() { shared_mutex_.unlock(); });

  meta_.set_searcher("HnswSparseSearcher", HnswSparseEntity::kRevision,
                     ailego::Params());

  int ret = IndexHelper::SerializeToDumper(meta_, dumper.get());
  if (ret != 0) {
    LOG_ERROR("Failed to serialize meta into dumper.");
    return ret;
  }
  return entity_.dump(dumper);
}

IndexStreamer::Context::Pointer HnswSparseStreamer::create_context(void) const {
  if (ailego_unlikely(state_ != STATE_OPENED)) {
    LOG_ERROR("Create context failed, open storage first!");
    return Context::Pointer();
  }

  HnswSparseEntity::Pointer entity = entity_.clone();
  if (ailego_unlikely(!entity)) {
    LOG_ERROR("CreateContext clone init failed");
    return Context::Pointer();
  }
  HnswSparseContext *ctx =
      new (std::nothrow) HnswSparseContext(metric_, entity);
  if (ailego_unlikely(ctx == nullptr)) {
    LOG_ERROR("Failed to new HnswSparseContext");
    return Context::Pointer();
  }
  ctx->set_ef(ef_);
  ctx->set_max_scan_limit(max_scan_limit_);
  ctx->set_min_scan_limit(min_scan_limit_);
  ctx->set_max_scan_ratio(max_scan_ratio_);
  ctx->set_filter_mode(bf_enabled_ ? VisitFilter::BloomFilter
                                   : VisitFilter::ByteMap);
  ctx->set_filter_negative_probability(bf_negative_prob_);
  ctx->set_magic(magic_);
  ctx->set_force_padding_topk(force_padding_topk_enabled_);
  ctx->set_bruteforce_threshold(bruteforce_threshold_);

  if (ailego_unlikely(ctx->init(HnswSparseContext::kSparseStreamerContext)) !=
      0) {
    LOG_ERROR("Init HnswSparseContext failed");
    delete ctx;
    return Context::Pointer();
  }

  return Context::Pointer(ctx);
}

IndexStreamer::SparseProvider::Pointer
HnswSparseStreamer::create_sparse_provider(void) const {
  LOG_DEBUG("HnswSparseStreamer create sparse provider");

  auto entity = entity_.clone();
  if (ailego_unlikely(!entity)) {
    LOG_ERROR("Clone HnswSparseEntity failed");
    return SparseProvider::Pointer();
  }
  return SparseProvider::Pointer(
      new HnswSparseIndexProvider(meta_, entity, "HnswSparseStreamer"));
}

int HnswSparseStreamer::update_context(HnswSparseContext *ctx) const {
  const HnswSparseEntity::Pointer entity = entity_.clone();
  if (!entity) {
    LOG_ERROR("Failed to clone search context entity");
    return IndexError_Runtime;
  }
  ctx->set_max_scan_limit(max_scan_limit_);
  ctx->set_min_scan_limit(min_scan_limit_);
  ctx->set_max_scan_ratio(max_scan_ratio_);
  ctx->set_bruteforce_threshold(bruteforce_threshold_);
  return ctx->update_context(HnswSparseContext::kSparseStreamerContext, meta_,
                             metric_, entity, magic_);
}

//! Add a vector with id  into index with sparse inputs
int HnswSparseStreamer::add_with_id_impl(uint32_t id,
                                         const uint32_t sparse_count,
                                         const uint32_t *sparse_indices,
                                         const void *sparse_query,
                                         const IndexQueryMeta &qmeta,
                                         Context::Pointer &context) {
  int ret = check_params(qmeta);
  if (ailego_unlikely(ret != 0)) {
    return ret;
  }

  if (ailego_unlikely(sparse_count >= HnswSparseEntity::kSparseMaxDimSize)) {
    LOG_WARN("Add vector failed, dim size too larg, dim_size=%u, id=%u",
             sparse_count, id);
    return IndexError_InvalidValue;
  }

  HnswSparseContext *ctx = dynamic_cast<HnswSparseContext *>(context.get());
  ailego_do_if_false(ctx) {
    LOG_ERROR("Cast context to HnswSparseContext failed");
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
                PARAM_HNSW_SPARSE_STREAMER_DOCS_HARD_LIMIT.c_str());
      const std::lock_guard<std::mutex> lk(mutex_);
      (*stats_.mutable_discarded_count())++;
      return IndexError_IndexFull;
    } else {
      LOG_WARN("Current docs %u exceed [%s]", entity_.doc_cnt(),
               PARAM_HNSW_SPARSE_STREAMER_DOCS_SOFT_LIMIT.c_str());
    }
  }
  if (ailego_unlikely(!shared_mutex_.try_lock_shared())) {
    LOG_ERROR("Cannot add vector while dumping index");
    (*stats_.mutable_discarded_count())++;
    return IndexError_Unsupported;
  }
  AILEGO_DEFER([&]() { shared_mutex_.unlock_shared(); });

  ctx->clear();
  ctx->update_dist_caculator_distance(add_distance_);

  std::string sparse_query_buffer;
  SparseUtility::TransSparseFormat(sparse_count, sparse_indices, sparse_query,
                                   entity_.sparse_unit_size(),
                                   sparse_query_buffer);

  ctx->reset_query(sparse_query_buffer.data());
  ctx->check_need_adjuct_ctx(entity_.doc_cnt());

  level_t level = alg_->get_random_level();
  ret =
      entity_.add_vector_with_id(level, id, sparse_query_buffer, sparse_count);
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

//! Add a vector into index with sparse inputs
int HnswSparseStreamer::add_impl(uint64_t pkey, const uint32_t sparse_count,
                                 const uint32_t *sparse_indices,
                                 const void *sparse_query,
                                 const IndexQueryMeta &qmeta,
                                 Context::Pointer &context) {
  int ret = check_params(qmeta);
  if (ailego_unlikely(ret != 0)) {
    return ret;
  }

  if (ailego_unlikely(sparse_count >= HnswSparseEntity::kSparseMaxDimSize)) {
    LOG_WARN("Add vector failed, dim size too larg, dim_size=%u, key=%zu",
             sparse_count, (size_t)pkey);
    return IndexError_InvalidValue;
  }

  HnswSparseContext *ctx = dynamic_cast<HnswSparseContext *>(context.get());
  ailego_do_if_false(ctx) {
    LOG_ERROR("Cast context to HnswSparseContext failed");
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
                PARAM_HNSW_SPARSE_STREAMER_DOCS_HARD_LIMIT.c_str());
      const std::lock_guard<std::mutex> lk(mutex_);
      (*stats_.mutable_discarded_count())++;
      return IndexError_IndexFull;
    } else {
      LOG_WARN("Current docs %u exceed [%s]", entity_.doc_cnt(),
               PARAM_HNSW_SPARSE_STREAMER_DOCS_SOFT_LIMIT.c_str());
    }
  }
  if (ailego_unlikely(!shared_mutex_.try_lock_shared())) {
    LOG_ERROR("Cannot add vector while dumping index");
    (*stats_.mutable_discarded_count())++;
    return IndexError_Unsupported;
  }
  AILEGO_DEFER([&]() { shared_mutex_.unlock_shared(); });

  ctx->clear();
  ctx->update_dist_caculator_distance(add_distance_);

  std::string sparse_query_buffer;
  SparseUtility::TransSparseFormat(sparse_count, sparse_indices, sparse_query,
                                   entity_.sparse_unit_size(),
                                   sparse_query_buffer);

  ctx->reset_query(sparse_query_buffer.data());
  ctx->check_need_adjuct_ctx(entity_.doc_cnt());

  level_t level = alg_->get_random_level();
  node_id_t id;
  ret = entity_.add_vector(level, pkey, sparse_query_buffer, sparse_count, &id);
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

//! Similarity search with sparse inputs
int HnswSparseStreamer::search_impl(
    const uint32_t sparse_count, const uint32_t *sparse_indices,
    const void *sparse_query, const IndexQueryMeta &qmeta,
    IndexStreamer::Context::Pointer &context) const {
  return search_impl(&sparse_count, sparse_indices, sparse_query, qmeta, 1,
                     context);
}

//! Similarity search with sparse inputs
int HnswSparseStreamer::search_impl(
    const uint32_t *sparse_count, const uint32_t *sparse_indices,
    const void *sparse_query, const IndexQueryMeta &qmeta, uint32_t count,
    IndexStreamer::Context::Pointer &context) const {
  int ret = check_params(qmeta);
  if (ailego_unlikely(ret != 0)) {
    return ret;
  }
  HnswSparseContext *ctx = dynamic_cast<HnswSparseContext *>(context.get());
  ailego_do_if_false(ctx) {
    LOG_ERROR("Cast context to HnswSparseContext failed");
    return IndexError_Cast;
  }

  if (entity_.doc_cnt() <= ctx->get_bruteforce_threshold()) {
    return search_bf_impl(sparse_count, sparse_indices, sparse_query, qmeta,
                          count, context);
  }

  if (ctx->magic() != magic_) {
    //! context is created by another searcher or streamer
    ret = update_context(ctx);
    if (ret != 0) {
      return ret;
    }
  }

  ctx->clear();
  ctx->update_dist_caculator_distance(search_distance_);
  ctx->resize_results(count);
  ctx->check_need_adjuct_ctx(entity_.doc_cnt());

  const uint32_t *sparse_indices_tmp = sparse_indices;
  const void *sparse_query_tmp = sparse_query;

  for (size_t q = 0; q < count; ++q) {
    std::string sparse_query_buffer;
    std::string sparse_query_filtered_buffer;

    SparseUtility::TransSparseFormat(
        sparse_count[q], sparse_indices_tmp, sparse_query_tmp,
        entity_.sparse_unit_size(), sparse_query_buffer);

    if (query_filtering_enabled_) {
      if (!SparseUtility::FilterSparseQuery(
              sparse_count[q], sparse_indices_tmp, sparse_query_tmp,
              qmeta.data_type(), entity_.sparse_unit_size(),
              query_filtering_ratio_, &sparse_query_filtered_buffer)) {
        LOG_ERROR("Hnsw filtering failed");

        return IndexError_Runtime;
      }

      ctx->reset_query(sparse_query_filtered_buffer.data());
    } else {
      ctx->reset_query(sparse_query_buffer.data());
    }

    ret = alg_->search(ctx);
    if (ailego_unlikely(ret != 0)) {
      LOG_ERROR("Hnsw searcher fast search failed");
      return ret;
    }

    if (query_filtering_enabled_) {
      ctx->reset_query(sparse_query_buffer.data());
      ctx->recal_topk_dist();
    }

    ctx->topk_to_result(q);

    sparse_indices_tmp += sparse_count[q];
    sparse_query_tmp = reinterpret_cast<const char *>(sparse_query_tmp) +
                       sparse_count[q] * qmeta.unit_size();
  }

  if (ailego_unlikely(ctx->error())) {
    return IndexError_Runtime;
  }

  return 0;
}

//! Similarity search with sparse inputs
int HnswSparseStreamer::search_bf_impl(
    const uint32_t sparse_count, const uint32_t *sparse_indices,
    const void *sparse_query, const IndexQueryMeta &qmeta,
    IndexStreamer::Context::Pointer &context) const {
  return search_bf_impl(&sparse_count, sparse_indices, sparse_query, qmeta, 1,
                        context);
}

//! Similarity search with sparse inputs
int HnswSparseStreamer::search_bf_impl(
    const uint32_t *sparse_count, const uint32_t *sparse_indices,
    const void *sparse_query, const IndexQueryMeta &qmeta, uint32_t count,
    IndexStreamer::Context::Pointer &context) const {
  int ret = check_params(qmeta);
  if (ailego_unlikely(ret != 0)) {
    return ret;
  }

  HnswSparseContext *ctx = dynamic_cast<HnswSparseContext *>(context.get());
  ailego_do_if_false(ctx) {
    LOG_ERROR("Cast context to HnswSparseContext failed");
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
  ctx->update_dist_caculator_distance(search_distance_);
  ctx->resize_results(count);

  const uint32_t *sparse_indices_tmp = sparse_indices;
  const void *sparse_query_tmp = sparse_query;

  if (ctx->group_by_search()) {
    if (!ctx->group_by().is_valid()) {
      LOG_ERROR("Invalid group-by function");
      return IndexError_Runtime;
    }

    std::function<std::string(node_id_t)> group_by = [&](node_id_t id) {
      return ctx->group_by()(entity_.get_key(id));
    };

    for (size_t q = 0; q < count; ++q) {
      std::string sparse_query_buffer;
      SparseUtility::TransSparseFormat(
          sparse_count[q], sparse_indices_tmp, sparse_query_tmp,
          entity_.sparse_unit_size(), sparse_query_buffer);

      ctx->reset_query(sparse_query_buffer.data());
      ctx->group_topk_heaps().clear();

      for (node_id_t id = 0; id < entity_.doc_cnt(); ++id) {
        if (entity_.get_key(id) == kInvalidKey) {
          continue;
        }

        if (!ctx->filter().is_valid() || !ctx->filter()(entity_.get_key(id))) {
          dist_t dist = ctx->dist_calculator().dist(id);

          std::string group_id = group_by(id);

          auto &topk_heap = ctx->group_topk_heaps()[group_id];
          if (topk_heap.empty()) {
            topk_heap.limit(ctx->group_topk());
          }
          topk_heap.emplace_back(id, dist);
        }
      }
      ctx->topk_to_result(q);

      sparse_indices_tmp += sparse_count[q];
      sparse_query_tmp = reinterpret_cast<const char *>(sparse_query_tmp) +
                         sparse_count[q] * qmeta.unit_size();
    }
  } else {
    auto &filter = ctx->filter();
    auto &topk = ctx->topk_heap();

    for (size_t q = 0; q < count; ++q) {
      std::string sparse_query_buffer;
      SparseUtility::TransSparseFormat(
          sparse_count[q], sparse_indices_tmp, sparse_query_tmp,
          entity_.sparse_unit_size(), sparse_query_buffer);

      ctx->reset_query(sparse_query_buffer.data());
      topk.clear();
      for (node_id_t id = 0; id < entity_.doc_cnt(); ++id) {
        if (entity_.get_key(id) == kInvalidKey) {
          continue;
        }

        if (!filter.is_valid() || !filter(entity_.get_key(id))) {
          dist_t dist = ctx->dist_calculator().dist(id);
          topk.emplace(id, dist);
        }
      }
      ctx->topk_to_result(q);

      sparse_indices_tmp += sparse_count[q];
      sparse_query_tmp = reinterpret_cast<const char *>(sparse_query_tmp) +
                         sparse_count[q] * qmeta.unit_size();
    }

    if (ailego_unlikely(ctx->error())) {
      return IndexError_Runtime;
    }
  }

  return 0;
}

//! Linear search by primary keys
int HnswSparseStreamer::search_bf_by_p_keys_impl(
    const uint32_t sparse_count, const uint32_t *sparse_indices,
    const void *sparse_query, const std::vector<std::vector<uint64_t>> &p_keys,
    const IndexQueryMeta &qmeta, ContextPointer &context) const {
  return search_bf_by_p_keys_impl(&sparse_count, sparse_indices, sparse_query,
                                  p_keys, qmeta, 1, context);
}

//! Linear search by primary keys with sparse inputs
int HnswSparseStreamer::search_bf_by_p_keys_impl(
    const uint32_t *sparse_count, const uint32_t *sparse_indices,
    const void *sparse_query, const std::vector<std::vector<uint64_t>> &p_keys,
    const IndexQueryMeta &qmeta, uint32_t count,
    Context::Pointer &context) const {
  int ret = check_params(qmeta);
  if (ailego_unlikely(ret != 0)) {
    return ret;
  }

  if (ailego_unlikely(p_keys.size() != count)) {
    LOG_ERROR("The size of p_keys is not equal to count");
    return IndexError_InvalidArgument;
  }

  HnswSparseContext *ctx = dynamic_cast<HnswSparseContext *>(context.get());
  ailego_do_if_false(ctx) {
    LOG_ERROR("Cast context to HnswSparseContext failed");
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
  ctx->update_dist_caculator_distance(search_distance_);
  ctx->resize_results(count);

  const uint32_t *sparse_indices_tmp = sparse_indices;
  const void *sparse_query_tmp = sparse_query;

  if (ctx->group_by_search()) {
    if (!ctx->group_by().is_valid()) {
      LOG_ERROR("Invalid group-by function");
      return IndexError_Runtime;
    }

    std::function<std::string(node_id_t)> group_by = [&](node_id_t id) {
      return ctx->group_by()(entity_.get_key(id));
    };

    for (size_t q = 0; q < count; ++q) {
      std::string sparse_query_buffer;
      SparseUtility::TransSparseFormat(
          sparse_count[q], sparse_indices_tmp, sparse_query_tmp,
          entity_.sparse_unit_size(), sparse_query_buffer);

      ctx->reset_query(sparse_query_buffer.data());
      ctx->group_topk_heaps().clear();

      for (size_t idx = 0; idx < p_keys[q].size(); ++idx) {
        uint64_t pk = p_keys[q][idx];
        if (!ctx->filter().is_valid() || !ctx->filter()(pk)) {
          node_id_t id = entity_.get_id(pk);
          if (id != kInvalidNodeId) {
            dist_t dist = ctx->dist_calculator().dist(id);

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

      sparse_indices_tmp += sparse_count[q];
      sparse_query_tmp = reinterpret_cast<const char *>(sparse_query_tmp) +
                         sparse_count[q] * qmeta.unit_size();
    }
  } else {
    auto &filter = ctx->filter();
    auto &topk = ctx->topk_heap();

    for (size_t q = 0; q < count; ++q) {
      std::string sparse_query_buffer;
      SparseUtility::TransSparseFormat(
          sparse_count[q], sparse_indices_tmp, sparse_query_tmp,
          entity_.sparse_unit_size(), sparse_query_buffer);

      ctx->reset_query(sparse_query_buffer.data());
      topk.clear();
      for (size_t idx = 0; idx < p_keys[q].size(); ++idx) {
        key_t pk = p_keys[q][idx];
        if (!filter.is_valid() || !filter(pk)) {
          node_id_t id = entity_.get_id(pk);
          if (id != kInvalidNodeId) {
            dist_t dist = ctx->dist_calculator().dist(id);
            topk.emplace(id, dist);
          }
        }
      }
      ctx->topk_to_result(q);

      sparse_indices_tmp += sparse_count[q];
      sparse_query_tmp = reinterpret_cast<const char *>(sparse_query_tmp) +
                         sparse_count[q] * qmeta.unit_size();
    }
  }

  if (ailego_unlikely(ctx->error())) {
    return IndexError_Runtime;
  }

  return 0;
}

void HnswSparseStreamer::print_debug_info() {
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

  // entity_.print_key_map();
}

INDEX_FACTORY_REGISTER_STREAMER(HnswSparseStreamer);

}  // namespace core
}  // namespace zvec
