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

#include "flat_sparse_streamer.h"
#include <cstdint>
#include <utility/sparse_utility.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_meta.h>
#include "flat_sparse_context.h"
#include "flat_sparse_provider.h"
#include "flat_sparse_search.h"

namespace zvec {
namespace core {

const uint32_t FlatSparseStreamer::VERSION = 0U;

FlatSparseStreamer::FlatSparseStreamer() : entity_(stats_) {}

FlatSparseStreamer::~FlatSparseStreamer() {
  this->close();
}

int FlatSparseStreamer::init(const IndexMeta &imeta,
                             const ailego::Params &params) {
  LOG_DEBUG("FlatSparseStreamer init");

  meta_ = imeta;
  meta_.set_streamer("FlatSparseStreamer", VERSION, params);

  state_ = STATE_INITED;

  return 0;
}

int FlatSparseStreamer::cleanup() {
  LOG_DEBUG("FlatSparseStreamer cleanup");

  this->close();

  meta_.clear();

  return 0;
}

int FlatSparseStreamer::open(IndexStorage::Pointer stg) {
  LOG_DEBUG("FlatSparseStreamer open");

  if (ailego_unlikely(state_ != STATE_INITED)) {
    LOG_ERROR("Open storage failed, init streamer first!");
    return IndexError_NoReady;
  }

  int ret = entity_.open(std::move(stg), meta_);
  if (ret != 0) {
    LOG_ERROR("FlatSparseStreamer entity failed to open storage");
    return ret;
  }

  IndexMeta index_meta;
  ret = entity_.get_index_sparse_meta(&index_meta);
  if (ret == IndexError_NoExist) {
    // Set IndexMeta for the new index
    ret = entity_.set_index_sparse_meta(meta_);
    if (ret != 0) {
      LOG_ERROR("Failed to set index meta for %s", IndexError::What(ret));
      return ret;
    }
  } else {
    if (index_meta.streamer_revision() != meta_.streamer_revision()) {
      LOG_ERROR("Streamer revision mismatch, expect=%u, actual=%u",
                meta_.streamer_revision(), index_meta.streamer_revision());
      return IndexError_Mismatch;
    }
    if (index_meta.metric_name() != meta_.metric_name() ||
        index_meta.data_type() != meta_.data_type()) {
      LOG_ERROR("IndexMeta mismatch from the previous in index");
      return IndexError_Mismatch;
    }
    // The IndexMeasure Params may be updated like MipsSquaredEuclidean
    auto metric_params = index_meta.metric_params();
    metric_params.merge(meta_.metric_params());
    meta_.set_metric(index_meta.metric_name(), 0, metric_params);
  }

  state_ = STATE_OPENED;
  magic_ = IndexContext::GenerateMagic();

  return 0;
}

int FlatSparseStreamer::close() {
  if (state_ != STATE_OPENED) {
    return 0;
  }

  LOG_DEBUG("FlatSparseStreamer close");

  stats_.clear();
  int ret = entity_.close();
  if (ret != 0) {
    LOG_ERROR("Failed to close entity %s", IndexError::What(ret));
    return ret;
  }
  state_ = STATE_INITED;
  return 0;
}

int FlatSparseStreamer::flush(uint64_t checkpoint) {
  if (state_ != STATE_OPENED) {
    LOG_ERROR("Failed to flush, open streamer first!");
    return IndexError_NoReady;
  }

  LOG_INFO("FlatSparseStreamer flush, checkpoint=%zu", (size_t)checkpoint);

  return entity_.flush(checkpoint);
}

int FlatSparseStreamer::dump(const IndexDumper::Pointer &dumper) {
  if (state_ != STATE_OPENED) {
    LOG_ERROR("Failed to dump, open streamer first!");
    return IndexError_NoReady;
  }

  LOG_INFO("FlatSparseStreamer dump");

  shared_mutex_.lock();
  AILEGO_DEFER([&]() { shared_mutex_.unlock(); });

  meta_.set_searcher("FlatSparseSearcher", VERSION, ailego::Params());

  int ret = IndexHelper::SerializeToDumper(meta_, dumper.get());
  if (ret != 0) {
    LOG_ERROR("Failed to serialize meta into dumper.");
    return ret;
  }

  return entity_.dump(dumper);
}

FlatSparseStreamer::ContextPointer FlatSparseStreamer::create_context() const {
  if (state_ != STATE_OPENED) {
    LOG_ERROR("Failed to create Context, open streamer first!");
    return Context::UPointer();
  }
  FlatSparseStreamerEntity::Pointer entity = entity_.clone();
  return FlatSparseStreamer::ContextPointer(new FlatSparseContext(this));
}

IndexStreamer::SparseProvider::Pointer
FlatSparseStreamer::create_sparse_provider(void) const {
  if (state_ != STATE_OPENED) {
    LOG_ERROR("Failed to create provider, open streamer first!");
    return SparseProvider::Pointer();
  }

  auto entity = entity_.clone();
  if (ailego_unlikely(!entity)) {
    LOG_ERROR("Clone entity failed");
    return SparseProvider::Pointer();
  }
  return SparseProvider::Pointer(
      new FlatSparseIndexProvider<FlatSparseStreamerEntity>(
          entity, meta_, "FlatSparseStreamerProvider"));
}

int FlatSparseStreamer::add_impl(uint64_t pkey, const uint32_t sparse_count,
                                 const uint32_t *sparse_indices,
                                 const void *sparse_query,
                                 const IndexQueryMeta &qmeta,
                                 Context::Pointer &context) {
  if (state_ != STATE_OPENED) {
    LOG_ERROR("Failed to add_impl, open streamer first!");
    (*stats_.mutable_discarded_count())++;
    return IndexError_NoReady;
  }

  int ret = check_params(qmeta);
  if (ailego_unlikely(ret != 0)) {
    (*stats_.mutable_discarded_count())++;
    return ret;
  }

  if (ailego_unlikely(sparse_count > PARAM_FLAT_SPARSE_MAX_DIM_SIZE)) {
    LOG_ERROR("Add vector failed, dim size too larg, dim_size=%u, key=%zu",
              sparse_count, (size_t)pkey);
    (*stats_.mutable_discarded_count())++;
    return IndexError_InvalidValue;
  }

  // context is trivial here
  FlatSparseContext *ctx = dynamic_cast<FlatSparseContext *>(context.get());
  ailego_do_if_false(ctx) {
    LOG_ERROR("Cast context to FlatSparseContext failed");
    (*stats_.mutable_discarded_count())++;
    return IndexError_Cast;
  }

  if (ailego_unlikely(!shared_mutex_.try_lock_shared())) {
    LOG_ERROR("Cannot add vector while dumping index");
    (*stats_.mutable_discarded_count())++;
    return IndexError_Unsupported;
  }
  AILEGO_DEFER([&]() { shared_mutex_.unlock_shared(); });

  // convert to sparse format and add to entity
  std::string sparse_query_buffer;
  SparseUtility::TransSparseFormat(sparse_count, sparse_indices, sparse_query,
                                   meta_.unit_size(), sparse_query_buffer);

  ret = entity_.add(pkey, sparse_query_buffer, sparse_count);
  if (ret != 0) {
    LOG_ERROR("Failed to add sparse vector, key=%zu, ret=%s", (size_t)pkey,
              IndexError::What(ret));
    (*stats_.mutable_discarded_count())++;
    return ret;
  }

  (*stats_.mutable_added_count())++;
  return 0;
}

int FlatSparseStreamer::add_with_id_impl(uint32_t pkey,
                                         const uint32_t sparse_count,
                                         const uint32_t *sparse_indices,
                                         const void *sparse_query,
                                         const IndexQueryMeta &qmeta,
                                         Context::Pointer &context) {
  if (state_ != STATE_OPENED) {
    LOG_ERROR("Failed to add_with_id_impl, open streamer first!");
    (*stats_.mutable_discarded_count())++;
    return IndexError_NoReady;
  }

  int ret = check_params(qmeta);
  if (ailego_unlikely(ret != 0)) {
    (*stats_.mutable_discarded_count())++;
    return ret;
  }

  if (ailego_unlikely(sparse_count > PARAM_FLAT_SPARSE_MAX_DIM_SIZE)) {
    LOG_ERROR("Add vector failed, dim size too larg, dim_size=%u, key=%zu",
              sparse_count, (size_t)pkey);
    (*stats_.mutable_discarded_count())++;
    return IndexError_InvalidValue;
  }

  // context is trivial here
  FlatSparseContext *ctx = dynamic_cast<FlatSparseContext *>(context.get());
  ailego_do_if_false(ctx) {
    LOG_ERROR("Cast context to FlatSparseContext failed");
    (*stats_.mutable_discarded_count())++;
    return IndexError_Cast;
  }

  if (ailego_unlikely(!shared_mutex_.try_lock_shared())) {
    LOG_ERROR("Cannot add vector while dumping index");
    (*stats_.mutable_discarded_count())++;
    return IndexError_Unsupported;
  }
  AILEGO_DEFER([&]() { shared_mutex_.unlock_shared(); });

  // convert to sparse format and add to entity
  std::string sparse_query_buffer;
  SparseUtility::TransSparseFormat(sparse_count, sparse_indices, sparse_query,
                                   meta_.unit_size(), sparse_query_buffer);

  ret = entity_.add_vector_with_id(pkey, sparse_query_buffer, sparse_count);
  if (ret != 0) {
    LOG_ERROR("Failed to add sparse vector, key=%zu, ret=%s", (size_t)pkey,
              IndexError::What(ret));
    (*stats_.mutable_discarded_count())++;
    return ret;
  }

  (*stats_.mutable_added_count())++;
  return 0;
}

//! Similarity search with sparse inputs
int FlatSparseStreamer::search_impl(const uint32_t sparse_count,
                                    const uint32_t *sparse_indices,
                                    const void *sparse_query,
                                    const IndexQueryMeta &qmeta,
                                    Context::Pointer &context) const {
  return search_impl(&sparse_count, sparse_indices, sparse_query, qmeta, 1,
                     context);
}

//! Similarity search with sparse inputs
int FlatSparseStreamer::search_impl(const uint32_t *sparse_count,
                                    const uint32_t *sparse_indices,
                                    const void *sparse_query,
                                    const IndexQueryMeta &qmeta, uint32_t count,
                                    Context::Pointer &context) const {
  return search_bf_impl(sparse_count, sparse_indices, sparse_query, qmeta,
                        count, context);
}

//! Similarity brute force search with sparse inputs
int FlatSparseStreamer::search_bf_impl(const uint32_t sparse_count,
                                       const uint32_t *sparse_indices,
                                       const void *sparse_query,
                                       const IndexQueryMeta &qmeta,
                                       Context::Pointer &context) const {
  return search_bf_impl(&sparse_count, sparse_indices, sparse_query, qmeta, 1,
                        context);
}

//! Linear search by primary keys
int FlatSparseStreamer::search_bf_by_p_keys_impl(
    const uint32_t sparse_count, const uint32_t *sparse_indices,
    const void *sparse_query, const std::vector<std::vector<uint64_t>> &p_keys,
    const IndexQueryMeta &qmeta, ContextPointer &context) const {
  return search_bf_by_p_keys_impl(&sparse_count, sparse_indices, sparse_query,
                                  p_keys, qmeta, 1, context);
}

//! Similarity brute force search with sparse inputs
int FlatSparseStreamer::search_bf_impl(const uint32_t *sparse_count,
                                       const uint32_t *sparse_indices,
                                       const void *sparse_query,
                                       const IndexQueryMeta &qmeta,
                                       uint32_t count,
                                       Context::Pointer &context) const {
  return do_search(sparse_count, sparse_indices, sparse_query, false, {}, qmeta,
                   count, context);
}

//! Linear search by primary keys with sparse inputs
int FlatSparseStreamer::search_bf_by_p_keys_impl(
    const uint32_t *sparse_count, const uint32_t *sparse_indices,
    const void *sparse_query, const std::vector<std::vector<uint64_t>> &p_keys,
    const IndexQueryMeta &qmeta, uint32_t count,
    ContextPointer &context) const {
  return do_search(sparse_count, sparse_indices, sparse_query, true, p_keys,
                   qmeta, count, context);
}

//! Fetch sparse vector by key
int FlatSparseStreamer::get_sparse_vector(
    uint64_t key, uint32_t *sparse_count, std::string *sparse_indices_buffer,
    std::string *sparse_values_buffer) const {
  if (state_ != STATE_OPENED) {
    LOG_ERROR("Failed to get_sparse_vector, open streamer first!");
    return IndexError_NoReady;
  }

  std::string sparse_data;

  int ret = entity_.get_sparse_vector_by_key(key, &sparse_data);
  if (ailego_unlikely(ret != 0)) {
    LOG_ERROR("Failed to get sparse vector, key=%zu, ret=%s", (size_t)key,
              IndexError::What(ret));
    return ret;
  }

  SparseUtility::ReverseSparseFormat(sparse_data, sparse_count,
                                     sparse_indices_buffer,
                                     sparse_values_buffer, meta_.unit_size());

  return 0;
}

int FlatSparseStreamer::do_search(
    const uint32_t *sparse_count, const uint32_t *sparse_indices,
    const void *sparse_query, bool with_p_keys,
    const std::vector<std::vector<uint64_t>> &p_keys,
    const IndexQueryMeta &qmeta, uint32_t count,
    ContextPointer &context) const {
  if (state_ != STATE_OPENED) {
    LOG_ERROR("Failed to do_search, open streamer first!");
    return IndexError_NoReady;
  }

  int ret = check_params(qmeta);
  if (ailego_unlikely(ret != 0)) {
    return ret;
  }

  FlatSparseContext *ctx = dynamic_cast<FlatSparseContext *>(context.get());
  if (ctx->magic() != magic_) {
    ctx->reset(this);
  }

  return FlatSearch(sparse_count, sparse_indices, sparse_query, with_p_keys,
                    p_keys, qmeta, count, meta_, context,
                    (FlatSparseEntity *)&entity_);
}

INDEX_FACTORY_REGISTER_STREAMER(FlatSparseStreamer);

}  // namespace core
}  // namespace zvec