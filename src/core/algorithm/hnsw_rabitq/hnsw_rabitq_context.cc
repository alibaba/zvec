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
#include "hnsw_rabitq_context.h"
#include <chrono>
#include "hnsw_rabitq_params.h"

namespace zvec {
namespace core {

HnswRabitqContext::HnswRabitqContext(size_t dimension,
                                     const IndexMetric::Pointer &metric,
                                     const HnswRabitqEntity::Pointer &entity)
    : IndexContext(metric),
      entity_(entity),
      add_dc_(entity_.get(), metric, dimension) {}

HnswRabitqContext::HnswRabitqContext(const IndexMetric::Pointer &metric,
                                     const HnswRabitqEntity::Pointer &entity)
    : IndexContext(metric), entity_(entity), add_dc_(entity_.get(), metric) {}

HnswRabitqContext::~HnswRabitqContext() {
  visit_filter_.destroy();
}

int HnswRabitqContext::init(ContextType type) {
  int ret;
  uint32_t doc_cnt;

  type_ = type;

  switch (type) {
    case kBuilderContext:
      ret = visit_filter_.init(VisitFilter::ByteMap, entity_->doc_cnt(),
                               max_scan_num_, negative_probability_);
      if (ret != 0) {
        LOG_ERROR("Create filter failed,  mode %d", filter_mode_);
        return ret;
      }
      candidates_.limit(max_scan_num_);
      update_heap_.limit(entity_->l0_neighbor_cnt() + 1);
      break;

    case kSearcherContext:
      ret = visit_filter_.init(filter_mode_, entity_->doc_cnt(), max_scan_num_,
                               negative_probability_);
      if (ret != 0) {
        LOG_ERROR("Create filter failed,  mode %d", filter_mode_);
        return ret;
      }
      candidates_.limit(max_scan_num_);
      break;

    case kStreamerContext:
      // maxScanNum is unknown if inited from streamer, so the docCnt may
      // change. we need to compute maxScanNum by scan ratio, and preserve
      // max_doc_cnt space from visit filter
      doc_cnt = entity_->doc_cnt();
      max_scan_num_ = compute_max_scan_num(doc_cnt);
      reserve_max_doc_cnt_ = doc_cnt + compute_reserve_cnt(doc_cnt);
      ret = visit_filter_.init(filter_mode_, reserve_max_doc_cnt_,
                               max_scan_num_, negative_probability_);
      if (ret != 0) {
        LOG_ERROR("Create filter failed,  mode %d", filter_mode_);
        return ret;
      }

      update_heap_.limit(entity_->l0_neighbor_cnt() + 1);
      candidates_.limit(max_scan_num_);

      check_need_adjuct_ctx();
      break;

    default:
      LOG_ERROR("Init context failed");
      return IndexError_Runtime;
  }

  return 0;
}

int HnswRabitqContext::update(const ailego::Params &params) {
  auto update_visit_filter_param = [&]() {
    bool need_update = false;
    std::string p;
    switch (type_) {
      case kSearcherContext:
        p = PARAM_HNSW_RABITQ_SEARCHER_VISIT_BLOOMFILTER_ENABLE;
        break;
      case kStreamerContext:
        p = PARAM_HNSW_RABITQ_STREAMER_VISIT_BLOOMFILTER_ENABLE;
        break;
    }

    if (params.has(p)) {
      bool bf_enabled;
      params.get(p, &bf_enabled);
      if (bf_enabled ^ (filter_mode_ == VisitFilter::BloomFilter)) {
        need_update = true;
        filter_mode_ =
            bf_enabled ? VisitFilter::BloomFilter : VisitFilter::ByteMap;
      }
    }

    float prob = negative_probability_;
    p.clear();
    switch (type_) {
      case kSearcherContext:
        p = PARAM_HNSW_RABITQ_SEARCHER_VISIT_BLOOMFILTER_NEGATIVE_PROB;
        break;
      case kStreamerContext:
        p = PARAM_HNSW_RABITQ_STREAMER_VISIT_BLOOMFILTER_NEGATIVE_PROB;
        break;
    }
    params.get(p, &prob);
    if (filter_mode_ == VisitFilter::BloomFilter &&
        std::abs(prob - negative_probability_) > 1e-6) {
      need_update = true;
    }
    if (need_update) {
      visit_filter_.destroy();
      int max_doc_cnt = 0;
      if (type_ == kSearcherContext) {
        max_doc_cnt = entity_->doc_cnt();
      } else {
        max_doc_cnt = reserve_max_doc_cnt_;
      }
      int ret = visit_filter_.init(filter_mode_, max_doc_cnt, max_scan_num_,
                                   negative_probability_);
      if (ret != 0) {
        LOG_ERROR("Create filter failed,  mode %d", filter_mode_);
        return ret;
      }
    }
    return 0;
  };

  switch (type_) {
    case kSearcherContext:
      if (params.has(PARAM_HNSW_RABITQ_SEARCHER_EF)) {
        params.get(PARAM_HNSW_RABITQ_SEARCHER_EF, &ef_);
        topk_heap_.limit(std::max(topk_, ef_));
      }

      if (params.has(PARAM_HNSW_RABITQ_SEARCHER_MAX_SCAN_RATIO)) {
        params.get(PARAM_HNSW_RABITQ_SEARCHER_MAX_SCAN_RATIO, &max_scan_ratio_);
        max_scan_num_ =
            static_cast<uint32_t>(max_scan_ratio_ * entity_->doc_cnt());
        max_scan_num_ = std::max(10000U, max_scan_num_);
      }

      if (params.has(PARAM_HNSW_RABITQ_SEARCHER_BRUTE_FORCE_THRESHOLD)) {
        params.get(PARAM_HNSW_RABITQ_SEARCHER_BRUTE_FORCE_THRESHOLD,
                   &bruteforce_threshold_);
      }

      return update_visit_filter_param();

    case kStreamerContext:
      if (params.has(PARAM_HNSW_RABITQ_STREAMER_EF)) {
        params.get(PARAM_HNSW_RABITQ_STREAMER_EF, &ef_);
        topk_heap_.limit(std::max(topk_, ef_));
      }
      params.get(PARAM_HNSW_RABITQ_STREAMER_EF, &ef_);
      params.get(PARAM_HNSW_RABITQ_STREAMER_MAX_SCAN_RATIO, &max_scan_ratio_);
      params.get(PARAM_HNSW_RABITQ_STREAMER_MAX_SCAN_LIMIT, &max_scan_limit_);
      params.get(PARAM_HNSW_RABITQ_STREAMER_MIN_SCAN_LIMIT, &min_scan_limit_);
      if (max_scan_ratio_ <= 0.0f || max_scan_ratio_ > 1.0f) {
        LOG_ERROR("[%s] must be in range (0.0f,1.0f]",
                  PARAM_HNSW_RABITQ_STREAMER_MAX_SCAN_RATIO.c_str());
        return IndexError_InvalidArgument;
      }
      if (max_scan_limit_ < min_scan_limit_) {
        LOG_ERROR("[%s] must be >= [%s]",
                  PARAM_HNSW_RABITQ_STREAMER_MAX_SCAN_LIMIT.c_str(),
                  PARAM_HNSW_RABITQ_STREAMER_MIN_SCAN_LIMIT.c_str());
        return IndexError_InvalidArgument;
      }

      if (params.has(PARAM_HNSW_RABITQ_STREAMER_BRUTE_FORCE_THRESHOLD)) {
        params.get(PARAM_HNSW_RABITQ_STREAMER_BRUTE_FORCE_THRESHOLD,
                   &bruteforce_threshold_);
      }

      return update_visit_filter_param();

    default:
      LOG_ERROR("update context failed, type=%zu", static_cast<size_t>(type_));
      return IndexError_Runtime;
  }
}

int HnswRabitqContext::update_context(ContextType type, const IndexMeta &meta,
                                      const IndexMetric::Pointer &metric,
                                      const HnswRabitqEntity::Pointer &entity,
                                      uint32_t magic_num) {
  uint32_t doc_cnt;

  if (ailego_unlikely(type != type_)) {
    LOG_ERROR(
        "HnswRabitqContext doesn't support shared by different type, "
        "src=%u dst=%u",
        type_, type);
    return IndexError_Unsupported;
  }

  magic_ = kInvalidMgic;

  // TODO: support change filter mode?
  switch (type) {
    case kBuilderContext:
      LOG_ERROR("BuildContext doesn't support update");
      return IndexError_NotImplemented;

    case kSearcherContext:
      if (!visit_filter_.reset(entity->doc_cnt(), max_scan_num_)) {
        LOG_ERROR("Reset filter failed, mode %d", visit_filter_.get_mode());
        return IndexError_Runtime;
      }

      candidates_.limit(max_scan_num_);
      topk_heap_.limit(std::max(topk_, ef_));
      break;

    case kStreamerContext:
      doc_cnt = entity->doc_cnt();
      max_scan_num_ = compute_max_scan_num(doc_cnt);
      reserve_max_doc_cnt_ = doc_cnt + compute_reserve_cnt(doc_cnt);
      if (!visit_filter_.reset(reserve_max_doc_cnt_, max_scan_num_)) {
        LOG_ERROR("Reset filter failed, mode %d", visit_filter_.get_mode());
        return IndexError_Runtime;
      }

      update_heap_.limit(entity->l0_neighbor_cnt() + 1);
      candidates_.limit(max_scan_num_);
      topk_heap_.limit(std::max(topk_, ef_));
      break;

    default:
      LOG_ERROR("update context failed");
      return IndexError_Runtime;
  }

  entity_ = entity;
  dc().update(entity_.get(), metric, meta.dimension());
  magic_ = magic_num;
  level_topks_.clear();

  return 0;
}

void HnswRabitqContext::fill_random_to_topk_full(void) {
  static std::mt19937 mt(
      std::chrono::system_clock::now().time_since_epoch().count());
  std::uniform_int_distribution<node_id_t> dt(0, entity_->doc_cnt() - 1);
  std::function<node_id_t()> gen;
  node_id_t seqid;
  std::function<bool(node_id_t)> myfilter = [](node_id_t) { return false; };
  if (this->filter().is_valid()) {
    myfilter = [&](node_id_t id) {
      return this->filter()(entity_->get_key(id));
    };
  }

  if (topk_heap_.limit() < entity_->doc_cnt() / 2) {
    gen = [&](void) { return dt(mt); };
  } else {
    // If topk limit is big value, gen sequential id from an random initial
    seqid = dt(mt);
    gen = [&](void) {
      seqid = seqid == (entity_->doc_cnt() - 1) ? 0 : (seqid + 1);
      return seqid;
    };
  }

  for (size_t i = 0; !topk_heap_.full() && i < entity_->doc_cnt(); ++i) {
    const auto id = gen();
    if (!visit_filter_.visited(id) && !myfilter(id)) {
      visit_filter_.set_visited(id);
      topk_heap_.emplace(id, dc().dist(id));
    }
  }
  return;
}

}  // namespace core
}  // namespace zvec