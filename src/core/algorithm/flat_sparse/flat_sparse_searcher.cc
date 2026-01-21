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

#include "flat_sparse_searcher.h"
#include <utility/sparse_utility.h>
#include <zvec/core/framework/index_error.h>
#include "flat_sparse_context.h"
#include "flat_sparse_provider.h"
#include "flat_sparse_search.h"

namespace zvec {
namespace core {

const uint32_t FlatSparseSearcher::VERSION = 0U;

FlatSparseSearcher::FlatSparseSearcher(void) {}

FlatSparseSearcher::~FlatSparseSearcher(void) {}

int FlatSparseSearcher::init(const ailego::Params & /*params*/) {
  state_ = STATE_INITED;

  return 0;
}

int FlatSparseSearcher::cleanup(void) {
  this->unload();
  return 0;
}

int FlatSparseSearcher::load(IndexStorage::Pointer container,
                             IndexMetric::Pointer /*measure*/) {
  if (state_ != STATE_INITED) {
    LOG_ERROR("Init the searcher first before load index");
    return IndexError_Runtime;
  }

  LOG_INFO("Begin FlatSparseSearcher::load");

  int ret = IndexHelper::DeserializeFromStorage(container.get(), &meta_);
  if (ret != 0) {
    LOG_ERROR("Failed to deserialize meta from container");
    return ret;
  }

  if (meta_.searcher_revision() != VERSION) {
    LOG_ERROR("Unsupported searcher revision %u", meta_.searcher_revision());
    return IndexError_Unsupported;
  }

  ret = entity_.load(container, meta_);
  if (ret != 0) {
    LOG_ERROR("FlatSparseSearcher load index failed");
    return ret;
  }

  state_ = STATE_LOADED;
  magic_ = IndexContext::GenerateMagic();

  LOG_INFO("End FlatSparseSearcher::load");

  return 0;
}

int FlatSparseSearcher::unload(void) {
  LOG_INFO("Begin FlatSparseSearcher::unload");

  meta_.clear();
  entity_.unload();
  state_ = STATE_INITED;

  LOG_INFO("End FlatSparseSearcher::unload");

  return 0;
}

int FlatSparseSearcher::search_bf_impl(const uint32_t *sparse_count,
                                       const uint32_t *sparse_indices,
                                       const void *sparse_query,
                                       const IndexQueryMeta &qmeta,
                                       uint32_t count,
                                       Context::Pointer &context) const {
  return do_search(sparse_count, sparse_indices, sparse_query, false, {}, qmeta,
                   count, context);
}

int FlatSparseSearcher::search_bf_by_p_keys_impl(
    const uint32_t *sparse_count, const uint32_t *sparse_indices,
    const void *sparse_query, const std::vector<std::vector<uint64_t>> &p_keys,
    const IndexQueryMeta &qmeta, uint32_t count,
    ContextPointer &context) const {
  return do_search(sparse_count, sparse_indices, sparse_query, true, p_keys,
                   qmeta, count, context);
}

int FlatSparseSearcher::get_sparse_vector(
    uint64_t key, uint32_t *sparse_count, std::string *sparse_indices_buffer,
    std::string *sparse_values_buffer) const {
  if (state_ != STATE_LOADED) {
    LOG_ERROR("Failed to get sparse vector, load container first!");
    return IndexError_NoIndexLoaded;
  }

  std::string sparse_data;

  int ret = entity_.get_sparse_vector(key, &sparse_data);
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

FlatSparseSearcher::ContextPointer FlatSparseSearcher::create_context() const {
  if (state_ != STATE_LOADED) {
    LOG_ERROR("Failed to create Context, load container first!");
    return Context::UPointer();
  }
  FlatSparseSearcherEntity::Pointer entity = entity_.clone();
  return FlatSparseSearcher::ContextPointer(new FlatSparseContext(this));
}

//! Create a new iterator
IndexSearcher::SparseProvider::Pointer
FlatSparseSearcher::create_sparse_provider(void) const {
  if (state_ != STATE_LOADED) {
    LOG_ERROR("Failed to create provider, load container first!");
    return SparseProvider::Pointer();
  }

  auto entity = entity_.clone();
  if (ailego_unlikely(!entity)) {
    LOG_ERROR("Clone entity failed");
    return SparseProvider::Pointer();
  }
  return SparseProvider::Pointer(
      new FlatSparseIndexProvider<FlatSparseSearcherEntity>(
          entity, meta_, "FlatSparseSearcher"));
}

int FlatSparseSearcher::do_search(
    const uint32_t *sparse_count, const uint32_t *sparse_indices,
    const void *sparse_query, bool with_p_keys,
    const std::vector<std::vector<uint64_t>> &p_keys,
    const IndexQueryMeta &qmeta, uint32_t count,
    ContextPointer &context) const {
  if (state_ != STATE_LOADED) {
    LOG_ERROR("Failed to do search, load container first!");
    return IndexError_NoIndexLoaded;
  }

  int ret = check_params(qmeta);
  if (ailego_unlikely(ret != 0)) {
    return ret;
  }

  return FlatSearch(sparse_count, sparse_indices, sparse_query, with_p_keys,
                    p_keys, qmeta, count, meta_, context,
                    (FlatSparseEntity *)&entity_);
}

INDEX_FACTORY_REGISTER_SEARCHER(FlatSparseSearcher);

}  // namespace core
}  // namespace zvec