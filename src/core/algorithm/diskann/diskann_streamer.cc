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
#include "diskann_context.h"
#include "diskann_index_provider.h"
#include "diskann_indexer.h"
#include "diskann_params.h"

namespace zvec {
namespace core {

DiskAnnStreamer::DiskAnnStreamer() {}

DiskAnnStreamer::~DiskAnnStreamer() {}

int DiskAnnStreamer::init(const IndexMeta &meta,
                          const ailego::Params &search_params) {
  meta_ = meta;
  search_params.get(PARAM_DISKANN_SEARCHER_LIST_SIZE, &list_size_);
  search_params.get(PARAM_DISKANN_SEARCHER_CACHE_NODE_NUM, &cache_nodes_num_);
  return 0;
}

void DiskAnnStreamer::print_debug_info() {}

int DiskAnnStreamer::cleanup() {
  LOG_INFO("Begin DiskAnnStreamer:cleanup");

  LOG_INFO("End DiskAnnStreamer:cleanup");

  return 0;
}

int DiskAnnStreamer::open(IndexStorage::Pointer storage) {
  LOG_INFO("DiskAnnStreamer::load Begin");

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

    diskann_indexer_->load_cache_list(node_list);

    node_list.clear();
    node_list.shrink_to_fit();
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

  stats_.set_loaded_costtime(ailego::Monotime::MilliSeconds() - start_time);
  state_ = STATE_LOADED;

  magic_ = IndexContext::GenerateMagic();

  LOG_INFO("DiskAnnStreamer::load Done");

  return 0;
}

int DiskAnnStreamer::unload() {
  LOG_INFO("DiskAnnStreamer unload index");

  state_ = STATE_INITED;

  return 0;
}

int DiskAnnStreamer::update_context(DiskAnnContext *ctx) const {
  const DiskAnnEntity::Pointer entity = entity_.clone();
  if (!entity) {
    LOG_ERROR("Failed to clone search context entity");
    return IndexError_Runtime;
  }

  return ctx->update_context(DiskAnnContext::kSearcherContext, meta_, measure_,
                             entity, magic_);
}

int DiskAnnStreamer::search_impl(const void *query, const IndexQueryMeta &qmeta,
                                 uint32_t count,
                                 Context::Pointer &context) const {
  // do search
  if (ailego_unlikely(!query || !context)) {
    LOG_ERROR("The context is not created by this searcher");
    return IndexError_Mismatch;
  }

  DiskAnnContext *ctx = dynamic_cast<DiskAnnContext *>(context.get());
  ailego_do_if_false(ctx) {
    LOG_ERROR("Cast context to DiskAnnContext failed");
    return IndexError_Cast;
  }

  // Context is pooled per index type. When switching between DiskAnn indexes
  // with different element sizes (e.g., fp16 vs fp32), the cached context has
  // undersized buffers. Recreate it to ensure correct buffer allocations.
  if (ctx->magic() != magic_) {
    uint32_t saved_topk = ctx->topk();
    context = create_context();
    if (!context) {
      LOG_ERROR("Failed to recreate context for current streamer");
      return IndexError_Runtime;
    }
    ctx = dynamic_cast<DiskAnnContext *>(context.get());
    ctx->set_topk(saved_topk);
  }

  ctx->clear();
  ctx->resize_results(count);

  for (uint32_t i = 0; i < count; i++) {
    ctx->reset_query(query);

    diskann_indexer_->knn_search(ctx);

    ctx->topk_to_result(i);

    query = static_cast<const char *>(query) + qmeta.element_size();
  }

  return 0;
}

int DiskAnnStreamer::search_bf_impl(const void *query,
                                    const IndexQueryMeta &qmeta, uint32_t count,
                                    Context::Pointer &context) const {
  if (ailego_unlikely(!query || !context)) {
    LOG_ERROR("The context is not created by this searcher");
    return IndexError_Mismatch;
  }

  DiskAnnContext *ctx = dynamic_cast<DiskAnnContext *>(context.get());
  ailego_do_if_false(ctx) {
    LOG_ERROR("Cast context to DiskAnnContext failed");
    return IndexError_Cast;
  }

  if (ctx->magic() != magic_) {
    //! context is created by another searcher or streamer, recreate it
    //! to ensure buffers are correctly sized for this index's parameters.
    uint32_t saved_topk = ctx->topk();
    context = create_context();
    if (!context) {
      LOG_ERROR("Failed to recreate context for current streamer");
      return IndexError_Runtime;
    }
    ctx = dynamic_cast<DiskAnnContext *>(context.get());
    ctx->set_topk(saved_topk);
  }

  ctx->clear();
  ctx->resize_results(count);

  for (size_t i = 0; i < count; ++i) {
    ctx->reset_query(query);

    diskann_indexer_->linear_search(ctx);

    ctx->topk_to_result(i);

    query = static_cast<const char *>(query) + qmeta.element_size();
  }

  if (ailego_unlikely(ctx->error())) {
    return IndexError_Runtime;
  }

  return 0;
}

int DiskAnnStreamer::search_bf_by_p_keys_impl(
    const void *query, const std::vector<std::vector<uint64_t>> &p_keys,
    const IndexQueryMeta &qmeta, uint32_t count,
    Context::Pointer &context) const {
  if (ailego_unlikely(!query || !context)) {
    LOG_ERROR("The context is not created by this searcher");
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

  if (ctx->magic() != magic_) {
    //! context is created by another searcher or streamer, recreate it
    //! to ensure buffers are correctly sized for this index's parameters.
    uint32_t saved_topk = ctx->topk();
    context = create_context();
    if (!context) {
      LOG_ERROR("Failed to recreate context for current streamer");
      return IndexError_Runtime;
    }
    ctx = dynamic_cast<DiskAnnContext *>(context.get());
    ctx->set_topk(saved_topk);
  }

  ctx->clear();
  ctx->resize_results(count);

  for (size_t i = 0; i < count; ++i) {
    ctx->reset_query(query);

    diskann_indexer_->keys_search(p_keys[i], ctx);

    ctx->topk_to_result(i);

    query = static_cast<const char *>(query) + qmeta.element_size();
  }

  if (ailego_unlikely(ctx->error())) {
    return IndexError_Runtime;
  }

  return 0;
}

int DiskAnnStreamer::get_vector(uint64_t key, Context::Pointer &context,
                                std::string &vector) const {
  return diskann_indexer_->get_vector(key, context, vector);
}

const void *DiskAnnStreamer::get_vector_by_id(uint32_t /*id*/) const {
  // DiskAnn vectors are stored on disk in sector format;
  // a const void* access requires sector I/O via create_context
  // Return nullptr to indicate this path is not supported.
  return nullptr;
}

int DiskAnnStreamer::get_vector_by_id(const uint32_t id,
                                      IndexStorage::MemoryBlock &block) const {
  // Lazily create a reusable context for fetch operations
  if (!fetch_ctx_) {
    fetch_ctx_ = create_context();
    if (!fetch_ctx_) {
      LOG_ERROR("Failed to create context for get_vector_by_id");
      return IndexError_Runtime;
    }
  }
  int ret = diskann_indexer_->get_vector(id, fetch_ctx_, fetch_vector_buffer_);
  if (ret != 0) {
    LOG_ERROR("Failed to get vector by id: %u", id);
    return IndexError_Runtime;
  }
  block.reset((void *)fetch_vector_buffer_.data());
  return 0;
}

IndexSearcher::Provider::Pointer DiskAnnStreamer::create_provider(void) const {
  if (state_ != STATE_LOADED) {
    LOG_ERROR("Load the index first before creating a provider");
    return nullptr;
  }
  const DiskAnnEntity::Pointer entity = entity_.clone();
  if (!entity) {
    LOG_ERROR("Failed to clone DiskAnn entity for provider");
    return nullptr;
  }
  return IndexProvider::Pointer(new (std::nothrow) DiskAnnIndexProvider(
      meta_, entity, "DiskAnnStreamer"));
}

IndexSearcher::Context::Pointer DiskAnnStreamer::create_context() const {
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

  return Context::Pointer(ctx);
}

INDEX_FACTORY_REGISTER_STREAMER(DiskAnnStreamer);

}  // namespace core
}  // namespace zvec
