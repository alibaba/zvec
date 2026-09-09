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
#include "vamana_context.h"
#include <algorithm>
#include <random>
#include "vamana_params.h"

namespace zvec {
namespace core {

namespace {

// Vamana resolves the shared query defaults against the loaded graph layout.
// Keep this policy local to Vamana rather than exposing algorithm-tuning
// constants through the public interface.
constexpr uint32_t kPrefetchCacheLineBytes = 64;
constexpr uint32_t kPrefetchBudgetBytes = 6 * 1024;
constexpr uint32_t kPrefetchTargetLines = 2;
constexpr uint32_t kPrefetchMaxValue = 256;

}  // namespace

VamanaContext::VamanaContext(size_t dimension,
                             const IndexMetric::Pointer &metric,
                             const VamanaEntity::Pointer &entity)
    : IndexContext(metric),
      entity_(entity),
      dc_(entity.get(), metric, dimension),
      metric_(metric) {
  if (metric) {
    build_distance_offset_ = metric->build_distance_offset();
  }
}

VamanaContext::VamanaContext(const IndexMetric::Pointer &metric,
                             const VamanaEntity::Pointer &entity)
    : IndexContext(metric),
      entity_(entity),
      dc_(entity.get(), metric),
      metric_(metric) {
  if (metric) {
    build_distance_offset_ = metric->build_distance_offset();
  }
}

VamanaContext::~VamanaContext() {
  visit_filter_.destroy();
}

int VamanaContext::init(ContextType type) {
  int ret;
  uint32_t doc_cnt;

  type_ = type;
  results_.resize(1);
  topk_heap_.limit(std::max(topk_, ef_));
  update_heap_.limit(entity_->max_degree());

  switch (type) {
    case kBuilderContext:
      ret = visit_filter_.init(filter_mode_, entity_->doc_cnt(), max_scan_num_,
                               filter_negative_prob_);
      if (ret != 0) {
        LOG_ERROR("Create visit filter failed, mode %d", filter_mode_);
        return ret;
      }
      candidates_.limit(max_scan_num_);
      break;

    case kSearcherContext:
      ret = visit_filter_.init(filter_mode_, entity_->doc_cnt(), max_scan_num_,
                               filter_negative_prob_);
      if (ret != 0) {
        LOG_ERROR("Create visit filter failed, mode %d", filter_mode_);
        return ret;
      }
      candidates_.limit(max_scan_num_);
      prepare_query_prefetch();
      break;

    case kStreamerContext:
      doc_cnt = entity_->doc_cnt();
      max_scan_num_ = compute_max_scan_num(doc_cnt);
      reserve_max_doc_cnt_ = doc_cnt + compute_reserve_cnt(doc_cnt);
      ret = visit_filter_.init(filter_mode_, reserve_max_doc_cnt_,
                               max_scan_num_, filter_negative_prob_);
      if (ret != 0) {
        LOG_ERROR("Create visit filter failed, mode %d", filter_mode_);
        return ret;
      }
      candidates_.limit(max_scan_num_);
      check_need_adjuct_ctx();
      break;

    default:
      break;
  }

  return 0;
}

int VamanaContext::update_context(ContextType type, const IndexMeta &meta,
                                  const IndexMetric::Pointer &metric,
                                  const VamanaEntity::Pointer &entity,
                                  uint32_t magic_num) {
  if (magic_ == magic_num) {
    return 0;
  }
  type_ = type;
  entity_ = entity;
  metric_ = metric;
  update_index_metric(metric);
  magic_ = magic_num;
  if (metric) {
    build_distance_offset_ = metric->build_distance_offset();
  }
  dc_.update(entity.get(), metric, meta.dimension());
  if (query_prefetch_ready_) {
    update_query_prefetch();
  }
  return 0;
}

int VamanaContext::update(const ailego::Params &params) {
  uint32_t ef = ef_;
  params.get(PARAM_VAMANA_STREAMER_EF, &ef);
  ef_ = ef;
  topk_heap_.limit(std::max(topk_, ef_));
  uint32_t requested_po = requested_po_;
  uint32_t requested_pl = requested_pl_;
  params.get(PARAM_VAMANA_STREAMER_PO, &requested_po);
  params.get(PARAM_VAMANA_STREAMER_PL, &requested_pl);
  // Compare requests, not effective values: an automatic PO may have resolved
  // to the same number as a new manual PO but must retain different semantics.
  if (!query_prefetch_ready_ || requested_po != requested_po_ ||
      requested_pl != requested_pl_) {
    requested_po_ = requested_po;
    requested_pl_ = requested_pl;
    update_query_prefetch();
  }
  return 0;
}

void VamanaContext::update_query_prefetch() {
  const auto resolved = resolve_query_prefetch(
      entity_->vector_data_size(), static_cast<uint32_t>(entity_->max_degree()),
      requested_po_, requested_pl_);
  po_ = resolved.first;
  pl_ = resolved.second;
  query_prefetch_ready_ = true;
}

std::pair<uint32_t, uint32_t> VamanaContext::resolve_query_prefetch(
    size_t vector_data_size, uint32_t max_degree, uint32_t requested_offset,
    uint32_t requested_lines) {
  using namespace core_interface;

  if (vector_data_size == 0 || max_degree == 0) {
    return {0U, 0U};
  }

  const uint32_t body_lines =
      static_cast<uint32_t>((vector_data_size + kPrefetchCacheLineBytes - 1) /
                            kPrefetchCacheLineBytes);

  uint32_t resolved_lines;
  if (requested_lines == kDefaultPrefetchLines) {
    resolved_lines = std::min(kPrefetchTargetLines, body_lines);
  } else {
    resolved_lines = std::min({requested_lines, body_lines, kPrefetchMaxValue});
  }

  uint32_t resolved_offset;
  if (requested_offset == kDefaultPrefetchOffset) {
    const uint64_t bytes_per_neighbor =
        static_cast<uint64_t>(kPrefetchCacheLineBytes) * resolved_lines;
    const uint32_t budget_offset = static_cast<uint32_t>(
        std::max<uint64_t>(1, kPrefetchBudgetBytes / bytes_per_neighbor));
    resolved_offset = std::min({budget_offset, max_degree, kPrefetchMaxValue});
  } else {
    resolved_offset =
        std::min({requested_offset, max_degree, kPrefetchMaxValue});
  }

  return {resolved_offset, resolved_lines};
}

void VamanaContext::topk_to_result(uint32_t idx) {
  if (force_padding_topk_ && !topk_heap_.full() &&
      topk_heap_.size() < entity_->doc_cnt()) {
    this->fill_random_to_topk_full();
  }
  if (ailego_unlikely(topk_heap_.size() == 0)) {
    return;
  }

  ailego_assert_with(idx < results_.size(), "invalid idx");
  int size = std::min(topk_, static_cast<uint32_t>(topk_heap_.size()));
  topk_heap_.sort();
  results_[idx].clear();

  for (int i = 0; i < size; ++i) {
    auto score = topk_heap_[i].second;
    if (score > this->threshold()) {
      break;
    }
    node_id_t id = topk_heap_[i].first;
    if (fetch_vector_) {
      results_[idx].emplace_back(entity_->get_key(id), score, id,
                                 entity_->get_vector(id));
    } else {
      results_[idx].emplace_back(entity_->get_key(id), score, id);
    }
  }
}

void VamanaContext::fill_random_to_topk_full() {
  std::mt19937 rng(42);
  uint32_t doc_cnt = entity_->doc_cnt();
  uint32_t max_attempts = doc_cnt * 2;
  uint32_t attempts = 0;
  while (!topk_heap_.full() && doc_cnt > 0 && attempts < max_attempts) {
    node_id_t random_id = rng() % doc_cnt;
    if (entity_->get_key(random_id) != kInvalidKey) {
      dist_t random_dist = dc_.dist(random_id);
      topk_heap_.emplace(random_id, random_dist);
    }
    ++attempts;
  }
}

}  // namespace core
}  // namespace zvec
