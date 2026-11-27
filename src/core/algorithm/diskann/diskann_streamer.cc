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

#include "diskann_streamer.h"
#include <limits>
#include <ailego/pattern/defer.h>
#include "diskann_context.h"
#include "diskann_index_provider.h"
#include "diskann_indexer.h"
#include "diskann_params.h"
#include "diskann_quantizer.h"

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

DiskAnnStreamer::DiskAnnStreamer() = default;

DiskAnnStreamer::~DiskAnnStreamer() = default;

int DiskAnnStreamer::init(const IndexMeta &meta,
                          const ailego::Params &search_params) {
  if (state_ == STATE_LOADED) {
    LOG_ERROR("Close DiskAnnStreamer before reinitializing it");
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
  configured_meta_ = meta;
  params_ = search_params;
  list_size_ = list_size;
  cache_nodes_num_ = cache_nodes_num;
  data_quantizer_.reset();
  state_ = STATE_INITED;
  return 0;
}

int DiskAnnStreamer::init(const IndexMeta &meta,
                          const ailego::Params &search_params,
                          const turbo::Quantizer::Pointer &quantizer) {
  int ret = init(meta, search_params);
  if (ret != 0) {
    return ret;
  }
  data_quantizer_ = quantizer;
  return 0;
}

void DiskAnnStreamer::print_debug_info() {}

int DiskAnnStreamer::cleanup() {
  LOG_INFO("Begin DiskAnnStreamer:cleanup");

  unload();
  configured_meta_.clear();
  params_.clear();
  data_quantizer_.reset();
  list_size_ = 200;
  cache_nodes_num_ = 0;
  state_ = STATE_INIT;

  LOG_INFO("End DiskAnnStreamer:cleanup");

  return 0;
}

int DiskAnnStreamer::open(IndexStorage::Pointer storage) {
  LOG_INFO("DiskAnnStreamer::load Begin");

  if (!storage) {
    LOG_ERROR("Invalid storage");
    return IndexError_InvalidArgument;
  }
  if (state_ != STATE_INITED) {
    LOG_ERROR("Initialize and close DiskAnnStreamer before opening an index");
    return IndexError_NoReady;
  }

  {
    std::lock_guard<std::mutex> lock(fetch_mutex_);
    fetch_ctx_.reset();
    fetch_vector_buffer_.clear();
    diskann_indexer_.reset();
    entity_.clear();
    measure_.reset();
    stats_.clear();
  }

  bool loaded = false;
  AILEGO_DEFER([&] {
    if (!loaded) reset_loaded_index();
  });

  auto start_time = ailego::Monotime::MilliSeconds();

  int ret = IndexHelper::DeserializeFromStorage(storage.get(), &meta_);
  if (ret != 0) {
    LOG_ERROR("Failed to deserialize meta from storage");
    return ret;
  }

#if !RABITQ_SUPPORTED
  if (meta_.quantizer_name() == "RabitqQuantizer")
    return IndexError_Unsupported;
#endif
  // Candidate codes are selected from disk. Full vectors must match the
  // configured query and fetch layout, including any converter's reformer.
  if (meta_.dimension() != configured_meta_.dimension() ||
      meta_.data_type() != configured_meta_.data_type() ||
      meta_.element_size() != configured_meta_.element_size() ||
      meta_.metric_name() != configured_meta_.metric_name() ||
      meta_.reformer_name() != configured_meta_.reformer_name()) {
    return IndexError_Mismatch;
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
  auto quantizer = DiskAnnQuantizer::create_quantizer_from_meta_buffer(
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

  entity_.prepare_pq();
  stats_.set_loaded_costtime(ailego::Monotime::MilliSeconds() - start_time);
  stats_.set_loaded_count(entity_.doc_cnt());
  state_ = STATE_LOADED;

  magic_ = IndexContext::GenerateMagic();

  LOG_INFO("DiskAnnStreamer::load Done");
  loaded = true;
  return 0;
}

int DiskAnnStreamer::unload() {
  LOG_INFO("DiskAnnStreamer unload index");
  reset_loaded_index();
  return 0;
}

void DiskAnnStreamer::reset_loaded_index() {
  const State next_state = state_ == STATE_INIT ? STATE_INIT : STATE_INITED;
  {
    std::lock_guard<std::mutex> lock(fetch_mutex_);
    fetch_ctx_.reset();
    fetch_vector_buffer_.clear();
    diskann_indexer_.reset();
    entity_.clear();
    measure_.reset();
    meta_.clear();
    stats_.clear();
  }

  state_ = next_state;
}

int DiskAnnStreamer::update_context(DiskAnnContext *ctx) const {
  const DiskAnnEntity::Pointer entity = entity_.clone();
  if (!entity) {
    LOG_ERROR("Failed to clone search context entity");
    return IndexError_NoMemory;
  }

  return ctx->update_context(DiskAnnContext::kSearcherContext, meta_, measure_,
                             entity, magic_, data_quantizer_);
}

int DiskAnnStreamer::ensure_compatible_context(ContextPointer &context,
                                               DiskAnnContext *&ctx) const {
  if (ctx->magic() == magic_) {
    return 0;
  }

  Context::Pointer replacement;
  const int ret = create_context_with_status(replacement);
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

int DiskAnnStreamer::search_impl(const void *query, const IndexQueryMeta &qmeta,
                                 uint32_t count,
                                 Context::Pointer &context) const try {
  if (ailego_unlikely(state_ != STATE_LOADED)) {
    LOG_ERROR("Open DiskAnnStreamer before searching");
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

int DiskAnnStreamer::search_bf_impl(const void *query,
                                    const IndexQueryMeta &qmeta, uint32_t count,
                                    Context::Pointer &context) const try {
  if (ailego_unlikely(state_ != STATE_LOADED)) {
    LOG_ERROR("Open DiskAnnStreamer before searching");
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

int DiskAnnStreamer::search_bf_by_p_keys_impl(
    const void *query, const std::vector<std::vector<uint64_t>> &p_keys,
    const IndexQueryMeta &qmeta, uint32_t count,
    Context::Pointer &context) const try {
  if (ailego_unlikely(state_ != STATE_LOADED)) {
    LOG_ERROR("Open DiskAnnStreamer before searching");
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

int DiskAnnStreamer::get_vector(uint64_t key, Context::Pointer &context,
                                std::string &vector) const try {
  const auto fail = [&](int status) {
    if (status != IndexError_NoMemory) vector.clear();
    return status;
  };
  if (state_ != STATE_LOADED) {
    LOG_ERROR("Open DiskAnnStreamer before fetching vectors");
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

const void *DiskAnnStreamer::get_vector_by_id(uint32_t /*id*/) const {
  // DiskAnn vectors are stored on disk in sector format;
  // a const void* access requires sector I/O via create_context
  // Return nullptr to indicate this path is not supported.
  return nullptr;
}

int DiskAnnStreamer::get_vector_by_id(const uint32_t id,
                                      IndexStorage::MemoryBlock &block) const
    try {
  if (state_ != STATE_LOADED) {
    LOG_ERROR("Open DiskAnnStreamer before fetching vectors");
    return IndexError_NoReady;
  }
  if (id >= entity_.doc_cnt()) {
    LOG_ERROR("Vector id out of range: %u", id);
    return IndexError_OutOfRange;
  }

  std::lock_guard<std::mutex> lock(fetch_mutex_);
  if (!fetch_ctx_) {
    const DiskAnnEntity::Pointer fetch_entity = entity_.clone();
    if (!fetch_entity) return IndexError_NoMemory;
    int status = 0;
    fetch_ctx_ = DiskAnnContext::create_fetch_context(meta_, measure_,
                                                      fetch_entity, &status);
    if (!fetch_ctx_) return status;
  }

  int ret = diskann_indexer_->get_vector(id, fetch_ctx_, fetch_vector_buffer_);
  if (ret != 0) {
    LOG_ERROR("Failed to get vector by id: %u", id);
    return ret;
  }

  auto owned = std::make_shared<ailego::BudgetedAlignedBuffer>();
  owned->resize_process_rss(fetch_vector_buffer_.size(), alignof(void *));
  std::memcpy(owned->data(), fetch_vector_buffer_.data(),
              fetch_vector_buffer_.size());
  block = IndexStorage::MemoryBlock::MakeSharedView(owned->data(), owned);
  return 0;
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
}

IndexSearcher::Provider::Pointer DiskAnnStreamer::create_provider() const {
  if (state_ != STATE_LOADED) {
    LOG_ERROR("Load the index first before creating a provider");
    return nullptr;
  }
  const DiskAnnEntity::Pointer entity = entity_.clone();
  if (!entity) {
    LOG_ERROR("Failed to clone DiskAnn entity for provider");
    return nullptr;
  }
  std::unique_ptr<DiskAnnIndexProvider> provider(
      new (std::nothrow) DiskAnnIndexProvider(
          meta_, measure_, entity, diskann_indexer_, "DiskAnnStreamer"));
  if (!provider || !provider->ready()) {
    LOG_ERROR("Failed to initialize DiskAnn provider dependencies");
    return nullptr;
  }
  return IndexProvider::Pointer(provider.release());
}

IndexSearcher::Context::Pointer DiskAnnStreamer::create_context() const {
  Context::Pointer context;
  create_context_with_status(context);
  return context;
}

int DiskAnnStreamer::create_context_with_status(Context::Pointer &context) const
    try {
  if (state_ != STATE_LOADED) {
    LOG_ERROR("Open DiskAnnStreamer before creating a context");
    return IndexError_NoReady;
  }
  const DiskAnnEntity::Pointer search_ctx_entity = entity_.clone();
  if (!search_ctx_entity) {
    LOG_ERROR("Failed to create search context entity");
    return IndexError_NoMemory;
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

INDEX_FACTORY_REGISTER_STREAMER(DiskAnnStreamer);

}  // namespace core
}  // namespace zvec
