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
#include <limits>
#include <ailego/pattern/defer.h>
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

DiskAnnSearcher::DiskAnnSearcher() = default;

DiskAnnSearcher::~DiskAnnSearcher() = default;

int DiskAnnSearcher::init(const ailego::Params &search_params) {
  if (state_ == STATE_LOADED) {
    LOG_ERROR("Unload DiskAnnSearcher before reinitializing it");
    return IndexError_NoReady;
  }

  log_diskann_io_backend();

  uint32_t list_size = 200;
  uint32_t cache_nodes_num = 0;
  search_params.get(PARAM_DISKANN_SEARCHER_LIST_SIZE, &list_size);
  long long configured_cache_nodes = 0;
  if (search_params.get(PARAM_DISKANN_SEARCHER_CACHE_NODE_NUM,
                        &configured_cache_nodes)) {
    if (configured_cache_nodes < 0 ||
        static_cast<unsigned long long>(configured_cache_nodes) >
            std::numeric_limits<uint32_t>::max()) {
      LOG_ERROR("cache_node_num must be in [0, UINT32_MAX]");
      return IndexError_InvalidArgument;
    }
    cache_nodes_num = static_cast<uint32_t>(configured_cache_nodes);
  }

  // Commit only after every value has been validated. A failed re-init must
  // leave either the previous valid configuration or STATE_INIT untouched.
  params_ = search_params;
  list_size_ = list_size;
  cache_nodes_num_ = cache_nodes_num;
  data_quantizer_.reset();
  state_ = STATE_INITED;
  return 0;
}

int DiskAnnSearcher::init(const ailego::Params &search_params,
                          const turbo::Quantizer::Pointer &quantizer) {
  int ret = init(search_params);
  if (ret != 0) {
    return ret;
  }
  data_quantizer_ = quantizer;
  return 0;
}

void DiskAnnSearcher::print_debug_info() {}

int DiskAnnSearcher::cleanup() {
  LOG_INFO("Begin DiskAnnSearcher:cleanup");

  unload();
  params_.clear();
  data_quantizer_.reset();
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

  // Construct the quantizer from the persisted meta buffer here; the entity
  // only owns the bytes and the indexer receives the ready-to-use quantizer.
  // The implementation is resolved from the meta buffer header, so any
  // supported quantize type (PQ today, others later) plugs in transparently.
  DiskAnnSearcherEntity::QuantizerMetaBuffer quantizer_meta_buffer;
  ret = entity_.read_pq_quantizer_meta_buffer(&quantizer_meta_buffer);
  if (ret != 0) {
    LOG_ERROR("Read quantizer meta buffer failed, ret=%d", ret);
    return ret;
  }
  auto quantizer = DiskAnnUtil::create_quantizer_from_meta_buffer(
      quantizer_meta_buffer.bytes, meta_,
      static_cast<uint32_t>(entity_.pq_chunk_num()),
      entity_.legacy_pq_layout());
  if (!quantizer) {
    return IndexError_NoExist;
  }

  diskann_indexer_ = std::make_shared<DiskAnnIndexer>(meta_);

  int res = diskann_indexer_->init(entity_, std::move(quantizer));
  if (res != 0) {
    return res;
  }

  ret = diskann_indexer_->configure_cache(cache_nodes_num_);
  if (ret != 0) {
    return ret;
  }

  if (measure) {
    measure_ = std::move(measure);
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

  entity_.prepare_pq();
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
                             entity, magic_, data_quantizer_);
}

int DiskAnnSearcher::ensure_compatible_context(ContextPointer &context,
                                               DiskAnnContext *&ctx) const {
  if (ctx->magic() == magic_) {
    return 0;
  }

  Context::Pointer replacement;
  const int ret = create_search_context(replacement);
  if (ret != 0) return ret;
  auto *replacement_ctx = dynamic_cast<DiskAnnContext *>(replacement.get());
  if (!replacement_ctx) {
    LOG_ERROR("Failed to cast recreated DiskAnn context");
    return IndexError_Cast;
  }
  replacement_ctx->copy_query_options_from(*ctx);
  replacement_ctx->swap_results(*ctx);
  context = std::move(replacement);
  ctx = replacement_ctx;
  return 0;
}

int DiskAnnSearcher::search_impl(const void *query, const IndexQueryMeta &qmeta,
                                 uint32_t count,
                                 Context::Pointer &context) const try {
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
  AILEGO_DEFER(diskann_indexer_.get(), &DiskAnnIndexer::release_io_ctx, ctx);
  if (ailego_unlikely(!group_options_valid(ctx))) {
    LOG_ERROR("Group search requires a callback and a positive group topk");
    return IndexError_InvalidArgument;
  }

  DiskAnnContext::SearchResults results(*ctx);
  ctx->resize_results(count);

  for (uint32_t i = 0; i < count; i++) {
    ctx->reset_query(query);

    ret = diskann_indexer_->knn_search(ctx);
    if (ailego_unlikely(ret != 0)) {
      return ret;
    }

    if (ailego_unlikely(ctx->error())) {
      return ctx->error_code();
    }

    ctx->topk_to_result(i);
    if (ctx->error()) return ctx->error_code();

    query = static_cast<const char *>(query) + qmeta.element_size();
  }

  results.commit();
  return 0;
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
}

int DiskAnnSearcher::search_bf_impl(const void *query,
                                    const IndexQueryMeta &qmeta, uint32_t count,
                                    Context::Pointer &context) const try {
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
  AILEGO_DEFER(diskann_indexer_.get(), &DiskAnnIndexer::release_io_ctx, ctx);
  if (ailego_unlikely(!group_options_valid(ctx))) {
    LOG_ERROR("Group search requires a callback and a positive group topk");
    return IndexError_InvalidArgument;
  }

  DiskAnnContext::SearchResults results(*ctx);
  ctx->resize_results(count);

  for (size_t i = 0; i < count; ++i) {
    ctx->reset_query(query);

    ret = diskann_indexer_->linear_search(ctx);
    if (ailego_unlikely(ret != 0)) {
      return ret;
    }

    ctx->topk_to_result(i);
    if (ctx->error()) return ctx->error_code();

    query = static_cast<const char *>(query) + qmeta.element_size();
  }

  if (ailego_unlikely(ctx->error())) {
    return ctx->error_code();
  }

  results.commit();
  return 0;
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
}

int DiskAnnSearcher::search_bf_by_p_keys_impl(
    const void *query, const std::vector<std::vector<uint64_t>> &p_keys,
    const IndexQueryMeta &qmeta, uint32_t count,
    Context::Pointer &context) const try {
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
  AILEGO_DEFER(diskann_indexer_.get(), &DiskAnnIndexer::release_io_ctx, ctx);
  if (ailego_unlikely(!group_options_valid(ctx))) {
    LOG_ERROR("Group search requires a callback and a positive group topk");
    return IndexError_InvalidArgument;
  }

  DiskAnnContext::SearchResults results(*ctx);
  ctx->resize_results(count);

  for (size_t i = 0; i < count; ++i) {
    ctx->reset_query(query);

    ret = diskann_indexer_->keys_search(p_keys[i], ctx);
    if (ailego_unlikely(ret != 0)) {
      return ret;
    }

    ctx->topk_to_result(i);
    if (ctx->error()) return ctx->error_code();

    query = static_cast<const char *>(query) + qmeta.element_size();
  }

  if (ailego_unlikely(ctx->error())) {
    return ctx->error_code();
  }

  results.commit();
  return 0;
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
}

int DiskAnnSearcher::get_vector(uint64_t key, Context::Pointer &context,
                                std::string &vector) const try {
  const auto fail = [&](int status) {
    if (status != IndexError_NoMemory) vector.clear();
    return status;
  };
  if (state_ != STATE_LOADED) {
    LOG_ERROR("Load DiskAnnSearcher before fetching vectors");
    return fail(IndexError_NoReady);
  }
  if (!context) {
    LOG_ERROR("Invalid context for get_vector");
    return fail(IndexError_Mismatch);
  }
  auto *ctx = dynamic_cast<DiskAnnContext *>(context.get());
  if (!ctx) {
    LOG_ERROR("Cast context to DiskAnnContext failed");
    return fail(IndexError_Cast);
  }
  int ret = ensure_compatible_context(context, ctx);
  if (ret != 0) {
    return fail(ret);
  }

  diskann_id_t id;
  ret = diskann_indexer_->find_id(key, &id);
  if (ret != 0) return fail(ret);
  if (id == kInvalidId) {
    LOG_ERROR("Vector key does not exist: %lu", (unsigned long)key);
    return fail(IndexError_NoExist);
  }
  ret = diskann_indexer_->get_vector(id, context, vector);
  return ret == 0 ? 0 : fail(ret);
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
}

IndexSearcher::Context::Pointer DiskAnnSearcher::create_context() const {
  Context::Pointer context;
  create_search_context(context);
  return context;
}

int DiskAnnSearcher::create_search_context(Context::Pointer &context) const
    try {
  if (state_ != STATE_LOADED) {
    LOG_ERROR("Load DiskAnnSearcher before creating a context");
    return IndexError_NoReady;
  }
  const DiskAnnEntity::Pointer search_ctx_entity = entity_.clone();
  if (!search_ctx_entity) {
    LOG_ERROR("Failed to create search context entity");
    return IndexError_Runtime;
  }

  auto ctx = std::make_unique<DiskAnnContext>(
      meta_, measure_, search_ctx_entity, data_quantizer_);
  const int status = ctx->init(
      DiskAnnContext::kSearcherContext, search_ctx_entity->max_degree(),
      search_ctx_entity->pq_chunk_num(), meta_.element_size(),
      diskann_indexer_->requires_io_context());
  if (status != 0) return status;

  ctx->set_list_size(list_size_);
  ctx->set_magic(magic_);

  context = std::move(ctx);
  return 0;
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
}

INDEX_FACTORY_REGISTER_SEARCHER(DiskAnnSearcher);

}  // namespace core
}  // namespace zvec
