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
#include "ivf_streamer.h"
#include <zvec/ailego/utility/time_helper.h>
#include <zvec/core/framework/index_segment_storage.h>
#include "ivf_centroid_index.h"
#include "ivf_index_provider.h"
#include "ivf_params.h"

namespace zvec {
namespace core {

int IVFStreamer::init(const IndexMeta &meta, const ailego::Params &parameters) {
  meta_ = meta;
  params_ = parameters;

  params_.get(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, &bruteforce_threshold_);

  searcher_state_ = STATE_INITED;

  return 0;
}

int IVFStreamer::cleanup(void) {
  this->unload();

  params_.clear();
  bruteforce_threshold_ = kDefaultBfThreshold;

  searcher_state_ = STATE_INIT;
  return 0;
}

int IVFStreamer::open(IndexStorage::Pointer storage) {
  if (!storage) {
    LOG_ERROR("Invalid storage");
    return IndexError_InvalidArgument;
  }
  if (searcher_state_ != STATE_INITED) {
    LOG_ERROR("Initalize the searcher first before load index");
    return IndexError_Runtime;
  }

  ailego::ElapsedTime timer;
  int ret = IndexHelper::DeserializeFromStorage(storage.get(), &meta_);
  if (ret != 0) {
    LOG_ERROR("Failed to deserialize meta from storage");
    return ret;
  }

  //! Load centroid index
  centroid_index_ = std::make_shared<IVFCentroidIndex>();
  if (!centroid_index_) {
    return IndexError_NoMemory;
  }
  auto seg = storage->get(IVF_CENTROID_SEG_ID, 0);
  if (!seg) {
    LOG_ERROR("Failed to get segment %s", IVF_CENTROID_SEG_ID.c_str());
    return IndexError_InvalidFormat;
  }
  IndexStorage::Pointer seg_container =
      std::make_shared<IndexSegmentStorage>(seg);
  if (!seg_container) {
    return IndexError_NoMemory;
  }
  ret = seg_container->open(std::string(), false);
  if (ret != 0) {
    LOG_ERROR("IndexSegmentStorage load failed for %s", IndexError::What(ret));
    return ret;
  }
  ret = centroid_index_->load(seg_container, params_);
  if (ret != 0) {
    LOG_ERROR("Failed to load index for %s", IndexError::What(ret));
    return ret;
  }

  auto reformer = centroid_index_->reformer();
  params_.set(PARAM_IVF_SEARCHER_CONVERTER_REFORMER, reformer);

  //! load iverted index
  entity_ = std::make_shared<IVFEntity>();
  if (!entity_) {
    return IndexError_NoMemory;
  }
  ret = entity_->load(storage);
  ivf_check_error_code(ret);

  magic_ = IndexContext::GenerateMagic();

  stats_.set_loaded_count(entity_->vector_count());
  stats_.set_loaded_costtime(timer.milli_seconds());

  searcher_state_ = STATE_LOADED;
  return 0;
}

int IVFStreamer::unload(void) {
  magic_ = 0;
  centroid_index_.reset();
  entity_.reset();
  stats_.set_loaded_count(0UL);
  stats_.set_loaded_costtime(0UL);
  stats_.clear_attributes();
  searcher_state_ = STATE_INITED;

  return 0;
}

int IVFStreamer::search_bf_impl(const void *query, const IndexQueryMeta &qmeta,
                                Context::Pointer &context) const {
  return search_bf_impl(query, qmeta, 1, context);
}

int IVFStreamer::search_bf_impl(const void *query, const IndexQueryMeta &qmeta,
                                uint32_t count,
                                Context::Pointer &context) const {
  if (!query || qmeta.element_size() != meta_.element_size()) {
    LOG_ERROR("Null query or invalid qmeta");
    return IndexError_InvalidArgument;
  }
  IVFSearcherContext *ctx = dynamic_cast<IVFSearcherContext *>(context.get());
  if (!ctx || ctx->topk() == 0) {
    LOG_ERROR("Invalid context or topk not set yet");
    return IndexError_InvalidArgument;
  }
  if (ctx->magic() != magic_) {
    //! context is created by another searcher
    int ret = this->update_context(ctx);
    ivf_check_error_code(ret);
  }

  ctx->reset_results(count);
  auto &entity = ctx->entity();
  auto &filter = ctx->filter();

  //! Transform the querys for querying in inverted vector index later
  IndexQueryMeta iv_qmeta;
  int ret = entity->transform(query, qmeta, count, &query, &iv_qmeta);
  ivf_check_with_msg(ret, "Failed to transform querys");

  // TODO: do batch search in matrix
  for (size_t q = 0; q < count; ++q) {
    auto &context_stats = ctx->mutable_stats(q);
    auto &heap = ctx->mutable_result_heap();
    heap.clear();
    if (!filter.is_valid()) {
      ret = entity->search(query, &heap, &context_stats);
    } else {
      ret = entity->search(query, filter, &heap, &context_stats);
    }
    ivf_check_with_msg(ret, "Failed to search in entity for %s",
                       IndexError::What(ret));
    heap.sort();  // sort the results
    if (!filter.is_valid()) {
      // mapping the local id to key if query without filter
      ret = entity->retrieve_keys(&heap);
      ivf_check_error_code(ret);
    }
    entity->normalize(q, &heap);
    ctx->topk_to_result(q);
    query = static_cast<const char *>(query) + iv_qmeta.element_size();
  }

  return 0;
}

int IVFStreamer::search_impl(const void *query, const IndexQueryMeta &qmeta,
                             Context::Pointer &context) const {
  return this->search_impl(query, qmeta, 1, context);
}

int IVFStreamer::search_impl(const void *query, const IndexQueryMeta &qmeta,
                             uint32_t count, Context::Pointer &context) const {
  if (entity_->vector_count() <= bruteforce_threshold_) {
    return this->search_bf_impl(query, qmeta, count, context);
  }
  if (!query || qmeta.element_size() != meta_.element_size()) {
    LOG_ERROR("Null query or invalid qmeta");
    return IndexError_InvalidArgument;
  }

  IVFSearcherContext *ctx = dynamic_cast<IVFSearcherContext *>(context.get());
  if (!ctx || ctx->topk() == 0) {
    LOG_ERROR("Invalid context or topk not set yet");
    return IndexError_InvalidArgument;
  }
  if (ctx->magic() != magic_) {
    //! context is created by another searcher
    int ret = update_context(ctx);
    ivf_check_error_code(ret);
  }

  ctx->reset_results(count);
  auto &entity = ctx->entity();
  auto &filter = ctx->filter();

  auto &centroid_index_ctx = ctx->centroid_searcher_ctx();
  int ret = centroid_index_->search(query, qmeta, count, centroid_index_ctx);
  ivf_check_error_code(ret);

  //! Transform the querys for querying in inverted vector index later
  IndexQueryMeta iv_qmeta;
  ret = entity->transform(query, qmeta, count, &query, &iv_qmeta);
  ivf_check_with_msg(ret, "Failed to transform querys");

  for (size_t q = 0; q < count; ++q) {
    auto &centroids = centroid_index_ctx->result(q);
    auto &context_stats = ctx->mutable_stats(q);
    auto &heap = ctx->mutable_result_heap();
    heap.clear();
    uint32_t total_scan_count = 0;
    for (size_t i = 0;
         i < centroids.size() && total_scan_count < ctx->max_scan_count();
         ++i) {
      auto cid = centroids[i].key();
      uint32_t scan_count = 0;
      if (!filter.is_valid()) {
        ret = entity->search(cid, query, &scan_count, &heap, &context_stats);
      } else {
        ret = entity->search(cid, query, filter, &scan_count, &heap,
                             &context_stats);
      }
      ivf_check_with_msg(ret, "Failed to search in entity for %s",
                         IndexError::What(ret));
      total_scan_count += scan_count;
    }
    heap.sort();  // sort the results
    if (!filter.is_valid()) {
      // mapping the local id to key if query without filter
      ret = entity->retrieve_keys(&heap);
      ivf_check_error_code(ret);
    }
    entity->normalize(q, &heap);
    ctx->topk_to_result(q);
    query = static_cast<const char *>(query) + iv_qmeta.element_size();
  }

  return 0;
}

const IndexSearcher::Stats &IVFStreamer::stats(void) const {
  return stats_;
}

IndexSearcher::Context::Pointer IVFStreamer::create_context() const {
  if (searcher_state_ != STATE_LOADED) {
    LOG_ERROR("Load the index first before create context");
    return nullptr;
  }

  auto entity = entity_->clone();
  if (!entity) {
    LOG_ERROR("Failed to clone IVFEntity");
    return nullptr;
  }

  auto centroid_index_ctx = centroid_index_->create_context();
  if (!centroid_index_ctx) {
    LOG_ERROR("Failed to create centroid index context");
    return nullptr;
  }

  auto context =
      new (std::nothrow) IVFSearcherContext(entity, centroid_index_ctx);
  if (!context) {
    LOG_ERROR("Failed to alloc IVFSearcherContext");
    return nullptr;
  }
  int ret = context->init(params_);
  if (ret != 0) {
    delete context;
    return nullptr;
  }

  context->set_magic(magic_);

  return Context::Pointer(context);
}

IndexProvider::Pointer IVFStreamer::create_provider(void) const {
  if (searcher_state_ != STATE_LOADED) {
    LOG_ERROR("Load the index first before create provider");
    return nullptr;
  }

  auto entity = entity_->clone();
  if (!entity) {
    LOG_ERROR("Failed to clone IVFEntity");
    return Provider::Pointer();
  }

  auto *provider = new (std::nothrow)
      IVFIndexProvider(entity->has_orignal_feature() ? meta_ : entity->meta(),
                       entity, "IVFStreamer");
  if (!provider) {
    LOG_ERROR("Failed to alloc IVFIndexProvider");
    return Provider::Pointer();
  }

  return Provider::Pointer(provider);
}

INDEX_FACTORY_REGISTER_STREAMER(IVFStreamer);

}  // namespace core
}  // namespace zvec
