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

#include "omega_streamer.h"
#include <omega/omega_api.h>
#include <omega/search_context.h>
#include <zvec/ailego/io/file.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_helper.h>
#include <zvec/core/framework/index_logger.h>
#include "omega_context.h"
#include "omega_hook_utils.h"
#include "omega_params.h"
#include "../hnsw/hnsw_context.h"
#include "../hnsw/hnsw_entity.h"

namespace zvec {
namespace core {

namespace {

struct OmegaHookSetup {
  OmegaHookState state;
  SearchHooks hooks;
};

OmegaHookSetup CreateOmegaHookSetup(omega::SearchContext *omega_search_ctx,
                                    bool enable_early_stopping,
                                    bool per_cmp_reporting) {
  OmegaHookSetup setup;
  setup.state.search_ctx = omega_search_ctx;
  setup.state.enable_early_stopping = enable_early_stopping;
  setup.state.per_cmp_reporting = per_cmp_reporting;
  ResetOmegaHookState(&setup.state);

  setup.hooks.user_data = &setup.state;
  setup.hooks.on_level0_entry = OnOmegaLevel0Entry;
  setup.hooks.on_hop = OnOmegaHop;
  setup.hooks.on_visit_candidate = OnOmegaVisitCandidate;
  return setup;
}

void EnableOmegaTrainingIfNeeded(
    OmegaSearchHandle omega_search, int query_id, bool training_mode_enabled,
    const std::vector<std::vector<uint64_t>> &training_ground_truth,
    int training_k_train) {
  if (!training_mode_enabled) {
    return;
  }

  std::vector<int> gt_for_query;
  if (query_id >= 0 &&
      static_cast<size_t>(query_id) < training_ground_truth.size()) {
    const auto &gt = training_ground_truth[query_id];
    gt_for_query.reserve(gt.size());
    for (uint64_t node_id : gt) {
      gt_for_query.push_back(static_cast<int>(node_id));
    }
  }

  omega_search_enable_training(omega_search, query_id, gt_for_query.data(),
                               gt_for_query.size(), training_k_train);
  LOG_DEBUG("Training mode enabled for query_id=%d with %zu GT nodes", query_id,
            gt_for_query.size());
}

void CollectOmegaTrainingOutputs(OmegaSearchHandle omega_search,
                                 OmegaContext *omega_ctx, int query_id) {
  if (omega_ctx == nullptr) {
    return;
  }

  size_t record_count = omega_search_get_training_records_count(omega_search);
  if (record_count > 0) {
    const void *records_ptr = omega_search_get_training_records(omega_search);
    const auto *records_vec =
        static_cast<const std::vector<omega::TrainingRecord> *>(records_ptr);

    for (size_t i = 0; i < record_count; ++i) {
      const auto &omega_record = (*records_vec)[i];
      core_interface::TrainingRecord record;
      record.query_id = omega_record.query_id;
      record.hops_visited = omega_record.hops_visited;
      record.cmps_visited = omega_record.cmps_visited;
      record.dist_1st = omega_record.dist_1st;
      record.dist_start = omega_record.dist_start;

      if (omega_record.traversal_window_stats.size() == 7) {
        std::copy(omega_record.traversal_window_stats.begin(),
                  omega_record.traversal_window_stats.end(),
                  record.traversal_window_stats.begin());
      }

      record.label = omega_record.label;
      omega_ctx->add_training_record(std::move(record));
    }

    LOG_DEBUG("Collected %zu training records for query_id=%d", record_count,
              query_id);
  }

  size_t gt_cmps_count = omega_search_get_gt_cmps_count(omega_search);
  if (gt_cmps_count == 0) {
    return;
  }

  const int *gt_cmps_ptr = omega_search_get_gt_cmps(omega_search);
  int total_cmps = omega_search_get_total_cmps(omega_search);
  if (gt_cmps_ptr == nullptr) {
    return;
  }

  std::vector<int> gt_cmps_vec(gt_cmps_ptr, gt_cmps_ptr + gt_cmps_count);
  for (auto &v : gt_cmps_vec) {
    if (v < 0) {
      v = total_cmps;
    }
  }
  omega_ctx->set_gt_cmps(gt_cmps_vec, total_cmps);
}

}  // namespace

bool OmegaStreamer::LoadModel(const std::string &model_dir) {
  std::lock_guard<std::mutex> lock(model_mutex_);

  if (omega_model_ != nullptr) {
    omega_model_destroy(omega_model_);
    omega_model_ = nullptr;
  }

  omega_model_ = omega_model_create();
  if (omega_model_ == nullptr) {
    LOG_ERROR("Failed to create OMEGA model manager");
    return false;
  }

  if (omega_model_load(omega_model_, model_dir.c_str()) != 0) {
    LOG_ERROR("Failed to load OMEGA model from %s", model_dir.c_str());
    omega_model_destroy(omega_model_);
    omega_model_ = nullptr;
    return false;
  }

  LOG_INFO("OMEGA model loaded successfully from %s", model_dir.c_str());
  return true;
}

bool OmegaStreamer::IsModelLoaded() const {
  std::lock_guard<std::mutex> lock(model_mutex_);
  return omega_model_ != nullptr && omega_model_is_loaded(omega_model_);
}

int OmegaStreamer::open(IndexStorage::Pointer stg) {
  std::string index_path = stg ? stg->file_path() : "";

  int ret = HnswStreamer::open(std::move(stg));
  if (ret != 0) {
    return ret;
  }

  {
    std::lock_guard<std::mutex> lock(model_mutex_);
    if (omega_model_ != nullptr) {
      omega_model_destroy(omega_model_);
      omega_model_ = nullptr;
    }
  }

  if (index_path.empty()) {
    LOG_WARN(
        "OmegaStreamer open: storage file path is empty, using HNSW fallback");
    return 0;
  }

  size_t last_slash = index_path.rfind('/');
  if (last_slash == std::string::npos) {
    LOG_WARN(
        "OmegaStreamer open: cannot derive omega_model path from index path %s",
        index_path.c_str());
    return 0;
  }

  std::string model_dir = index_path.substr(0, last_slash) + "/omega_model";
  std::string model_path = model_dir + "/model.txt";
  if (!ailego::File::IsExist(model_path)) {
    LOG_INFO(
        "OmegaStreamer open: no OMEGA model found at %s, using HNSW fallback",
        model_dir.c_str());
    return 0;
  }

  if (!LoadModel(model_dir)) {
    LOG_WARN(
        "OmegaStreamer open: failed to load OMEGA model from %s, using HNSW "
        "fallback",
        model_dir.c_str());
  }

  return 0;
}

int OmegaStreamer::search_impl(const void *query, const IndexQueryMeta &qmeta,
                               Context::Pointer &context) const {
  return search_impl(query, qmeta, 1, context);
}

int OmegaStreamer::search_impl(const void *query, const IndexQueryMeta &qmeta,
                               uint32_t count,
                               Context::Pointer &context) const {
  // Determine mode: training (no early stopping) vs inference (with early
  // stopping)
  bool enable_early_stopping = !training_mode_enabled_ && IsModelLoaded() &&
                               !DisableOmegaModelPrediction();

  if (training_mode_enabled_) {
    LOG_DEBUG("OmegaStreamer: training mode, early stopping DISABLED");
  } else if (enable_early_stopping) {
    LOG_DEBUG("OmegaStreamer: inference mode with OMEGA model");
  } else {
    LOG_DEBUG("OmegaStreamer: OMEGA hooks mode without model prediction");
  }

  return omega_search_impl(query, qmeta, count, context, enable_early_stopping);
}

int OmegaStreamer::omega_search_impl(const void *query,
                                     const IndexQueryMeta &qmeta,
                                     uint32_t count, Context::Pointer &context,
                                     bool enable_early_stopping) const {
  (void)qmeta;

  // Cast context to OmegaContext to access training_query_id
  auto *omega_ctx = dynamic_cast<OmegaContext *>(context.get());
  int query_id = current_query_id_;  // Default to member variable
  if (omega_ctx != nullptr && omega_ctx->training_query_id() >= 0) {
    query_id = omega_ctx->training_query_id();
  }

  // Get target recall from context if available
  float target_recall = target_recall_;
  if (omega_ctx != nullptr) {
    target_recall = omega_ctx->target_recall();
  }

  // Cast context to HnswContext to access HNSW-specific features
  auto *hnsw_ctx = dynamic_cast<HnswContext *>(context.get());
  if (hnsw_ctx == nullptr) {
    LOG_ERROR("Context is not HnswContext");
    return IndexError_InvalidArgument;
  }

  // Create OMEGA search context.
  // OMEGA's k is the search top-k, not the batch/query count.
  int omega_topk = static_cast<int>(hnsw_ctx->topk());
  if (omega_topk <= 0) {
    omega_topk = static_cast<int>(count);
  }

  // In training mode: model=nullptr (collect features only)
  // In inference mode: model=omega_model_ (use for early stopping)
  OmegaModelHandle model_to_use =
      enable_early_stopping ? omega_model_ : nullptr;

  OmegaSearchHandle omega_search = omega_search_create_with_params(
      model_to_use, target_recall, omega_topk, window_size_);

  if (omega_search == nullptr) {
    LOG_ERROR("Failed to create OMEGA search context");
    return IndexError_Runtime;
  }
  omega::SearchContext *omega_search_ctx =
      omega_search_get_cpp_context(omega_search);
  if (omega_search_ctx == nullptr) {
    omega_search_destroy(omega_search);
    LOG_ERROR("Failed to get OMEGA search context");
    return IndexError_Runtime;
  }

  // Training state is attached before the shared HNSW loop starts so label
  // collection sees the full query trajectory.
  EnableOmegaTrainingIfNeeded(omega_search, query_id, training_mode_enabled_,
                              training_ground_truth_, training_k_train_);

  // Rebind the context if it originated from a different searcher/streamer
  // instance so the HNSW state matches this streamer before search begins.
  if (hnsw_ctx->magic() != magic_) {
    int ret = update_context(hnsw_ctx);
    if (ret != 0) {
      omega_search_destroy(omega_search);
      return ret;
    }
  }

  // Initialize context for search
  hnsw_ctx->clear();
  hnsw_ctx->update_dist_caculator_distance(search_distance_,
                                           search_batch_distance_);
  hnsw_ctx->resize_results(count);
  hnsw_ctx->check_need_adjuct_ctx(entity_->doc_cnt());
  hnsw_ctx->reset_query(query);
  OmegaHookSetup hook_setup = CreateOmegaHookSetup(
      omega_search_ctx, enable_early_stopping, training_mode_enabled_);
  bool early_stop_hit = false;
  int ret =
      alg_->search_with_hooks(hnsw_ctx, &hook_setup.hooks, &early_stop_hit);
  if (ret != 0) {
    omega_search_destroy(omega_search);
    LOG_ERROR("OMEGA search failed");
    return ret;
  }
  MaybeFlushOmegaPendingCandidates(&hook_setup.state);
  int hops = 0;
  int cmps = 0;
  omega_search_ctx->GetStats(&hops, &cmps, nullptr);
  LOG_DEBUG(
      "OMEGA search completed: cmps=%d, hops=%d, results=%zu, early_stop=%d",
      cmps, hops, hnsw_ctx->topk_heap().size(),
      (early_stop_hit || omega_search_ctx->EarlyStopHit()) ? 1 : 0);

  // Match HNSW timing semantics: result materialization is outside the
  // search-core timer and happens after logging.
  hnsw_ctx->topk_to_result();

  if (training_mode_enabled_) {
    CollectOmegaTrainingOutputs(omega_search, omega_ctx, query_id);
  }

  omega_search_destroy(omega_search);
  return 0;
}

IndexStreamer::Context::Pointer OmegaStreamer::create_context(void) const {
  if (ailego_unlikely(state_ != STATE_OPENED)) {
    LOG_ERROR("Create OmegaContext failed, open storage first!");
    return Context::Pointer();
  }

  HnswEntity::Pointer entity = entity_->clone();
  if (ailego_unlikely(!entity)) {
    LOG_ERROR("OmegaContext clone entity failed");
    return Context::Pointer();
  }

  OmegaContext *ctx =
      new (std::nothrow) OmegaContext(meta_.dimension(), metric_, entity);
  if (ailego_unlikely(ctx == nullptr)) {
    LOG_ERROR("Failed to new OmegaContext");
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

  if (ailego_unlikely(ctx->init(HnswContext::kStreamerContext)) != 0) {
    LOG_ERROR("Init OmegaContext failed");
    delete ctx;
    return Context::Pointer();
  }
  return Context::Pointer(ctx);
}

int OmegaStreamer::dump(const IndexDumper::Pointer &dumper) {
  LOG_INFO("OmegaStreamer dump");

  shared_mutex_.lock();
  AILEGO_DEFER([&]() { shared_mutex_.unlock(); });

  // Persist the OMEGA searcher params alongside the dumped index metadata so a
  // reopened index reconstructs the same searcher-side behavior.
  ailego::Params searcher_params;
  const auto &streamer_params = meta_.streamer_params();

  // Copy the omega.* params into the searcher metadata for index persistence.
  if (streamer_params.has("omega.enabled")) {
    searcher_params.insert("omega.enabled",
                           streamer_params.get_as_bool("omega.enabled"));
  }
  if (streamer_params.has("omega.min_vector_threshold")) {
    searcher_params.insert(
        "omega.min_vector_threshold",
        streamer_params.get_as_uint32("omega.min_vector_threshold"));
  }
  if (streamer_params.has("omega.window_size")) {
    searcher_params.insert("omega.window_size",
                           streamer_params.get_as_int32("omega.window_size"));
  }

  LOG_INFO(
      "OmegaStreamer::dump: passing omega params to searcher "
      "(enabled=%d, min_threshold=%u, window_size=%d)",
      searcher_params.has("omega.enabled")
          ? searcher_params.get_as_bool("omega.enabled")
          : false,
      searcher_params.has("omega.min_vector_threshold")
          ? searcher_params.get_as_uint32("omega.min_vector_threshold")
          : 0,
      searcher_params.has("omega.window_size")
          ? searcher_params.get_as_int32("omega.window_size")
          : 0);

  meta_.set_searcher("OmegaSearcher", HnswEntity::kRevision, searcher_params);

  int ret = IndexHelper::SerializeToDumper(meta_, dumper.get());
  if (ret != 0) {
    LOG_ERROR("Failed to serialize meta into dumper.");
    return ret;
  }

  return entity_->dump(dumper);
}

INDEX_FACTORY_REGISTER_STREAMER(OmegaStreamer);

}  // namespace core
}  // namespace zvec
