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

#include "diskann_searcher.h"
#include "diskann_context.h"
#include "diskann_indexer.h"
#include "diskann_params.h"

namespace zvec {
namespace core {

namespace {

bool query_meta_matches(const IndexMeta &meta, const IndexQueryMeta &qmeta) {
  return qmeta.data_type() == meta.data_type() &&
         qmeta.dimension() == meta.dimension() &&
         qmeta.element_size() == meta.element_size();
}

bool group_options_valid(const DiskAnnContext *ctx) {
  return !ctx->group_by_search() ||
         (ctx->group_topk() > 0 && ctx->group_by().is_valid());
}

}  // namespace

DiskAnnSearcher::DiskAnnSearcher() {}

DiskAnnSearcher::~DiskAnnSearcher() {}

int DiskAnnSearcher::init(const ailego::Params &search_params) {
  if (state_ == STATE_LOADED) {
    LOG_ERROR("Unload DiskAnnSearcher before reinitializing it");
    return IndexError_NoReady;
  }

  params_ = search_params;
  list_size_ = 200;
  cache_nodes_num_ = 0;
  log_diskann_io_backend();

  params_.get(PARAM_DISKANN_SEARCHER_LIST_SIZE, &list_size_);
  params_.get(PARAM_DISKANN_SEARCHER_CACHE_NODE_NUM, &cache_nodes_num_);
  state_ = STATE_INITED;
  return 0;
}

void DiskAnnSearcher::print_debug_info() {}

int DiskAnnSearcher::cleanup() {
  LOG_INFO("Begin DiskAnnSearcher:cleanup");

  unload();
  params_.clear();
  list_size_ = 200;
  cache_nodes_num_ = 0;
  state_ = STATE_INIT;

  LOG_INFO("End DiskAnnSearcher:cleanup");

  return 0;
}

int DiskAnnSearcher::load(IndexStorage::Pointer storage,
                          IndexMetric::Pointer measure) {
  LOG_INFO("DiskAnnSearcher::load Begin");

  if (!storage) {
    LOG_ERROR("Invalid storage");
    return IndexError_InvalidArgument;
  }
  if (state_ != STATE_INITED) {
    LOG_ERROR("Initialize and unload DiskAnnSearcher before loading an index");
    return IndexError_NoReady;
  }

  diskann_indexer_.reset();
  entity_.clear();
  measure_.reset();
  meta_.clear();
  stats_.clear();

  auto start_time = ailego::Monotime::MilliSeconds();

  int ret = IndexHelper::DeserializeFromStorage(storage.get(), &meta_);
  if (ret != 0) {
    LOG_ERROR("Failed to deserialize meta from storage");
    return ret;
  }

  ret = entity_.load(meta_, storage);
  if (ret != 0) {
    LOG_INFO("Searcher Entity Load Failed");
    return ret;
  }

  diskann_indexer_ = std::make_shared<DiskAnnIndexer>(meta_);

  int res = diskann_indexer_->init(entity_);
  if (res != 0) {
    return res;
  }

  if (cache_nodes_num_ != 0) {
    std::vector<diskann_id_t> node_list;
    LOG_INFO("Caching %u nodes around medoid(s)", cache_nodes_num_);

    diskann_indexer_->cache_bfs_levels(cache_nodes_num_, node_list);

    ret = diskann_indexer_->load_cache_list(node_list);
    if (ret != 0) {
      return ret;
    }

    node_list.clear();
    node_list.shrink_to_fit();
  }

  if (measure) {
    measure_ = measure;
  } else {
    measure_ = IndexFactory::CreateMetric(meta_.metric_name());
    if (!measure_) {
      LOG_ERROR("CreateMetric failed, name: %s", meta_.metric_name().c_str());
      return IndexError_NoExist;
    }
    ret = measure_->init(meta_, meta_.metric_params());
    if (ret != 0) {
      LOG_ERROR("IndexMetric init failed, ret=%d", ret);
      return ret;
    }
    if (measure_->query_metric()) {
      measure_ = measure_->query_metric();
    }
  }

  stats_.set_loaded_costtime(ailego::Monotime::MilliSeconds() - start_time);
  stats_.set_loaded_count(entity_.doc_cnt());
  state_ = STATE_LOADED;

  magic_ = IndexContext::GenerateMagic();

  LOG_INFO("DiskAnnSearcher::load Done");

  return 0;
}

int DiskAnnSearcher::unload() {
  LOG_INFO("DiskAnnSearcher unload index");

  const State next_state = state_ == STATE_INIT ? STATE_INIT : STATE_INITED;
  diskann_indexer_.reset();
  entity_.clear();
  measure_.reset();
  meta_.clear();
  stats_.clear();
  state_ = next_state;

  return 0;
}

int DiskAnnSearcher::update_context(DiskAnnContext *ctx) const {
  const DiskAnnEntity::Pointer entity = entity_.clone();
  if (!entity) {
    LOG_ERROR("Failed to clone search context entity");
    return IndexError_Runtime;
  }

  return ctx->update_context(DiskAnnContext::kSearcherContext, meta_, measure_,
                             entity, magic_);
}

int DiskAnnSearcher::ensure_compatible_context(ContextPointer &context,
                                               DiskAnnContext *&ctx) const {
  if (ctx->magic() == magic_) {
    return 0;
  }

  auto replacement = create_context();
  if (!replacement) {
    LOG_ERROR("Failed to recreate context for current searcher");
    return IndexError_Runtime;
  }
  auto *replacement_ctx = dynamic_cast<DiskAnnContext *>(replacement.get());
  if (!replacement_ctx) {
    LOG_ERROR("Failed to cast recreated DiskAnn context");
    return IndexError_Cast;
  }
  replacement_ctx->copy_query_options_from(*ctx);
  context = std::move(replacement);
  ctx = replacement_ctx;
  return 0;
}

int DiskAnnSearcher::search_impl(const void *query, const IndexQueryMeta &qmeta,
                                 uint32_t count,
                                 Context::Pointer &context) const {
  if (ailego_unlikely(state_ != STATE_LOADED)) {
    LOG_ERROR("Load DiskAnnSearcher before searching");
    return IndexError_NoReady;
  }
  if (ailego_unlikely(!query || !context)) {
    LOG_ERROR("The context is not created by this searcher");
    return IndexError_Mismatch;
  }
  if (ailego_unlikely(!query_meta_matches(meta_, qmeta))) {
    LOG_ERROR("Query meta does not match DiskAnn index meta");
    return IndexError_Mismatch;
  }

  DiskAnnContext *ctx = dynamic_cast<DiskAnnContext *>(context.get());
  ailego_do_if_false(ctx) {
    LOG_ERROR("Cast context to DiskAnnContext failed");
    return IndexError_Cast;
  }

  int ret = ensure_compatible_context(context, ctx);
  if (ret != 0) {
    return ret;
  }
  if (ailego_unlikely(!group_options_valid(ctx))) {
    LOG_ERROR("Group search requires a callback and a positive group topk");
    return IndexError_InvalidArgument;
  }

  ctx->clear();
  ctx->resize_results(count);

  for (uint32_t i = 0; i < count; i++) {
    ctx->reset_query(query);

    ret = diskann_indexer_->knn_search(ctx);
    if (ailego_unlikely(ret != 0)) {
      return ret;
    }

    if (ailego_unlikely(ctx->error())) {
      return IndexError_Runtime;
    }

    ctx->topk_to_result(i);

    query = static_cast<const char *>(query) + qmeta.element_size();
  }

  return 0;
}

int DiskAnnSearcher::search_bf_impl(const void *query,
                                    const IndexQueryMeta &qmeta, uint32_t count,
                                    Context::Pointer &context) const {
  if (ailego_unlikely(state_ != STATE_LOADED)) {
    LOG_ERROR("Load DiskAnnSearcher before searching");
    return IndexError_NoReady;
  }
  if (ailego_unlikely(!query || !context)) {
    LOG_ERROR("The context is not created by this searcher");
    return IndexError_Mismatch;
  }
  if (ailego_unlikely(!query_meta_matches(meta_, qmeta))) {
    LOG_ERROR("Query meta does not match DiskAnn index meta");
    return IndexError_Mismatch;
  }

  DiskAnnContext *ctx = dynamic_cast<DiskAnnContext *>(context.get());
  ailego_do_if_false(ctx) {
    LOG_ERROR("Cast context to DiskAnnContext failed");
    return IndexError_Cast;
  }

  int ret = ensure_compatible_context(context, ctx);
  if (ret != 0) {
    return ret;
  }
  if (ailego_unlikely(!group_options_valid(ctx))) {
    LOG_ERROR("Group search requires a callback and a positive group topk");
    return IndexError_InvalidArgument;
  }

  ctx->clear();
  ctx->resize_results(count);

  for (size_t i = 0; i < count; ++i) {
    ctx->reset_query(query);

    ret = diskann_indexer_->linear_search(ctx);
    if (ailego_unlikely(ret != 0)) {
      return ret;
    }

    ctx->topk_to_result(i);

    query = static_cast<const char *>(query) + qmeta.element_size();
  }

  if (ailego_unlikely(ctx->error())) {
    return IndexError_Runtime;
  }

  return 0;
}

int DiskAnnSearcher::search_bf_by_p_keys_impl(
    const void *query, const std::vector<std::vector<uint64_t>> &p_keys,
    const IndexQueryMeta &qmeta, uint32_t count,
    Context::Pointer &context) const {
  if (ailego_unlikely(state_ != STATE_LOADED)) {
    LOG_ERROR("Load DiskAnnSearcher before searching");
    return IndexError_NoReady;
  }
  if (ailego_unlikely(!query || !context)) {
    LOG_ERROR("The context is not created by this searcher");
    return IndexError_Mismatch;
  }
  if (ailego_unlikely(!query_meta_matches(meta_, qmeta))) {
    LOG_ERROR("Query meta does not match DiskAnn index meta");
    return IndexError_Mismatch;
  }

  DiskAnnContext *ctx = dynamic_cast<DiskAnnContext *>(context.get());
  ailego_do_if_false(ctx) {
    LOG_ERROR("Cast context to DiskAnnContext failed");
    return IndexError_Cast;
  }

  if (ailego_unlikely(p_keys.size() != count)) {
    LOG_ERROR("The size of p_keys is not equal to count");
    return IndexError_InvalidArgument;
  }

  int ret = ensure_compatible_context(context, ctx);
  if (ret != 0) {
    return ret;
  }
  if (ailego_unlikely(!group_options_valid(ctx))) {
    LOG_ERROR("Group search requires a callback and a positive group topk");
    return IndexError_InvalidArgument;
  }

  ctx->clear();
  ctx->resize_results(count);

  for (size_t i = 0; i < count; ++i) {
    ctx->reset_query(query);

    ret = diskann_indexer_->keys_search(p_keys[i], ctx);
    if (ailego_unlikely(ret != 0)) {
      return ret;
    }

    ctx->topk_to_result(i);

    query = static_cast<const char *>(query) + qmeta.element_size();
  }

  if (ailego_unlikely(ctx->error())) {
    return IndexError_Runtime;
  }

  return 0;
}

int DiskAnnSearcher::get_vector(uint64_t key, Context::Pointer &context,
                                std::string &vector) const {
  vector.clear();
  if (state_ != STATE_LOADED) {
    LOG_ERROR("Load DiskAnnSearcher before fetching vectors");
    return IndexError_NoReady;
  }
  if (!context) {
    LOG_ERROR("Invalid context for get_vector");
    return IndexError_Mismatch;
  }
  auto *ctx = dynamic_cast<DiskAnnContext *>(context.get());
  if (!ctx) {
    LOG_ERROR("Cast context to DiskAnnContext failed");
    return IndexError_Cast;
  }
  int ret = ensure_compatible_context(context, ctx);
  if (ret != 0) {
    return ret;
  }

  diskann_id_t id = diskann_indexer_->get_id(key);
  if (id == kInvalidId) {
    LOG_ERROR("Vector key does not exist: %lu", (unsigned long)key);
    return IndexError_NoExist;
  }
  return diskann_indexer_->get_vector(id, context, vector);
}

IndexSearcher::Context::Pointer DiskAnnSearcher::create_context() const {
  if (state_ != STATE_LOADED) {
    LOG_ERROR("Load DiskAnnSearcher before creating a context");
    return Context::Pointer();
  }
  const DiskAnnEntity::Pointer search_ctx_entity = entity_.clone();
  if (!search_ctx_entity) {
    LOG_ERROR("Failed to create search context entity");
    return Context::Pointer();
  }

  DiskAnnContext *ctx =
      new (std::nothrow) DiskAnnContext(meta_, measure_, search_ctx_entity);
  if (ctx == nullptr) {
    LOG_ERROR("Failed to allocate DiskAnn Context");
    return Context::Pointer();
  }
  if (ailego_unlikely(ctx->init(
          DiskAnnContext::kSearcherContext, search_ctx_entity->max_degree(),
          search_ctx_entity->pq_chunk_num(), meta_.element_size())) != 0) {
    LOG_ERROR("Init DiskAnn Context failed");
    delete ctx;

    return Context::Pointer();
  }

  ctx->set_list_size(list_size_);
  ctx->set_magic(magic_);

  return Context::Pointer(ctx);
}

INDEX_FACTORY_REGISTER_SEARCHER(DiskAnnSearcher);

}  // namespace core
}  // namespace zvec
