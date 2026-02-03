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

#include <ailego/pattern/defer.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_flow.h>
#include <zvec/core/framework/index_helper.h>

//! Default storage
#define INDEX_FLOW_STORAGE_DEFAULT "MMapFileReadStorage"

namespace zvec {
namespace core {

// Index Flow
int IndexFlow::set_storage(const std::string &name,
                           const ailego::Params &params) {
  storage_ = IndexFactory::CreateStorage(name);
  if (!storage_) {
    LOG_ERROR("Failed to create a index storage with name: %s", name.c_str());
    return IndexError_NoExist;
  }
  int ret = storage_->init(params);
  if (ret < 0) {
    storage_ = nullptr;
    LOG_ERROR("Failed to initialize index storage %s", name.c_str());
    return ret;
  }
  return 0;
}

int IndexFlow::set_searcher(IndexSearcher::Pointer searcher) {
  user_searcher_ = searcher;

  return 0;
}

int IndexFlow::set_searcher(const std::string &name,
                            const ailego::Params &params) {
  user_searcher_ = IndexFactory::CreateSearcher(name);
  if (!user_searcher_) {
    LOG_ERROR("Failed to create a index searcher with name: %s", name.c_str());
    return IndexError_NoExist;
  }
  int ret = user_searcher_->init(params);
  if (ret < 0) {
    user_searcher_ = nullptr;
    LOG_ERROR("Failed to initialize index searcher %s", name.c_str());
    return ret;
  }
  return 0;
}

int IndexFlow::set_reformer(const std::string &name,
                            const ailego::Params &params) {
  user_reformer_ = IndexFactory::CreateReformer(name);
  if (!user_reformer_) {
    LOG_ERROR("Failed to create a index reformer with name: %s", name.c_str());
    return IndexError_NoExist;
  }
  int ret = user_reformer_->init(params);
  if (ret < 0) {
    user_reformer_ = nullptr;
    LOG_ERROR("Failed to initialize index reformer %s", name.c_str());
    return ret;
  }
  return 0;
}

int IndexFlow::set_metric(const std::string &name,
                          const ailego::Params &params) {
  if (!IndexFactory::HasMetric(name)) {
    LOG_ERROR("The index metric with name %s does not exist.", name.c_str());
    return IndexError_NoExist;
  }
  user_metric_name_ = name;
  user_metric_params_ = params;
  return 0;
}

int IndexFlow::load(const std::string &path) {
  // Prepare storage
  if (!storage_) {
    this->set_storage(INDEX_FLOW_STORAGE_DEFAULT, ailego::Params());
  }

  if (!storage_) {
    LOG_ERROR("The index storage is uninitialized.");
    return IndexError_Uninitialized;
  }

  int ret = storage_->open(path, false);
  if (ret != 0) {
    LOG_ERROR("Failed to load index with storage %s", storage_->name().c_str());
    return ret;
  }

  ret = IndexHelper::DeserializeFromStorage(storage_.get(), &meta_);
  if (ret != 0) {
    LOG_ERROR("Failed to deserialize index meta with storage %s",
              storage_->name().c_str());
    return ret;
  }

  ret = load_internal();
  if (ret != 0) {
    LOG_ERROR("Failed to load index with storage %s", storage_->name().c_str());
    return ret;
  }

  return 0;
}

int IndexFlow::load_internal() {
  // Prepare metric
  const std::string &metric_name =
      user_metric_name_.empty() ? meta_.metric_name() : user_metric_name_;
  const ailego::Params &metric_params =
      user_metric_name_.empty() ? meta_.metric_params() : user_metric_params_;
  if (metric_name.empty()) {
    LOG_ERROR("The metric name from index file is empty.");
    return IndexError_NoExist;
  }
  metric_ = IndexFactory::CreateMetric(metric_name);
  if (!metric_) {
    LOG_ERROR("Failed to create a index metric with name: %s",
              metric_name.c_str());
    return IndexError_NoExist;
  }
  int ret = metric_->init(meta_, metric_params);
  if (ret < 0) {
    LOG_ERROR("Failed to initialize index metric %s", metric_name.c_str());
    metric_ = nullptr;
    return ret;
  }
  if (!metric_->is_matched(meta_)) {
    LOG_ERROR("The index meta is unmatched for index metric %s",
              metric_->name().c_str());
    return IndexError_Mismatch;
  }
  auto query_metric = metric_->query_metric();
  if (query_metric) {
    metric_ = query_metric;
  }

  // Prepare reformer
  if (!user_reformer_) {
    const std::string &reformer_name = meta_.reformer_name();
    if (!reformer_name.empty()) {
      reformer_ = IndexFactory::CreateReformer(reformer_name);
      if (!reformer_) {
        LOG_ERROR("Failed to create a index reformer with name: %s",
                  reformer_name.c_str());
        return IndexError_NoExist;
      }
      ret = reformer_->init(meta_.reformer_params());
      if (ret < 0) {
        LOG_ERROR("Failed to initialize index reformer %s",
                  reformer_name.c_str());
        reformer_ = nullptr;
        return ret;
      }
    }
  } else {
    // Using user reformer
    reformer_ = user_reformer_;
  }

  if (reformer_) {
    ret = reformer_->load(storage_);
    if (ret < 0) {
      LOG_ERROR("Failed to load index with reformer %s, storage %s",
                reformer_->name().c_str(), storage_->name().c_str());
      return ret;
    }
  }

  // Prepare searcher
  if (!user_searcher_) {
    const std::string &name = meta_.searcher_name();
    if (name.empty()) {
      LOG_ERROR("The searcher name from index file is empty.");
      return IndexError_NoExist;
    }
    searcher_ = IndexFactory::CreateSearcher(name);
    if (!searcher_) {
      LOG_ERROR("Failed to create a index searcher with name: %s",
                name.c_str());
      return IndexError_NoExist;
    }
    ret = searcher_->init(meta_.searcher_params());
    if (ret < 0) {
      LOG_ERROR("Failed to initialize index searcher %s", name.c_str());
      searcher_ = nullptr;
      return ret;
    }
  } else {
    // Using user searcher
    searcher_ = user_searcher_;
  }

  ret = searcher_->load(storage_, metric_);
  if (ret < 0) {
    LOG_ERROR("Failed to load index with searcher %s, storage %s, metric %s",
              searcher_->name().c_str(), storage_->name().c_str(),
              metric_->name().c_str());
    return ret;
  }

  // searcher_->print_all_neighbour();

  return 0;
}

int IndexFlow::unload(void) {
  if (searcher_) {
    int ret = searcher_->unload();
    if (ret < 0) {
      LOG_WARN("Unload index searcher %s error, %d", searcher_->name().c_str(),
               ret);
    }
    searcher_ = nullptr;
  }
  if (reformer_) {
    int ret = reformer_->unload();
    if (ret < 0) {
      LOG_WARN("Unload index reformer %s error, %d", reformer_->name().c_str(),
               ret);
    }
    reformer_ = nullptr;
  }
  if (metric_) {
    int ret = metric_->cleanup();
    if (ret < 0) {
      LOG_WARN("Cleanup index metric %s error, %d", metric_->name().c_str(),
               ret);
    }
    metric_ = nullptr;
  }
  if (storage_) {
    int ret = storage_->cleanup();
    if (ret < 0) {
      LOG_WARN("Unload index searcher %s error, %d", storage_->name().c_str(),
               ret);
    }
    storage_ = nullptr;
  }
  return 0;
}

int IndexFlow::search_bf_impl(const void *query, const IndexQueryMeta &qmeta,
                              Context::Pointer &context) const {
  if (ailego_unlikely(!query || !context)) {
    return IndexError_InvalidArgument;
  }

  int error_code = 0;
  if (reformer_) {
    IndexQueryMeta new_qmeta;
    error_code = reformer_->transform(query, qmeta, context->mutable_features(),
                                      &new_qmeta);
    if (error_code == 0) {
      if (ailego_unlikely(!metric_->is_matched(meta_, new_qmeta))) {
        return IndexError_Mismatch;
      }
      error_code = searcher_->search_bf_impl(
          reinterpret_cast<const void *>(context->features().data()), new_qmeta,
          context->searcher_context());
    }
  } else {
    if (ailego_unlikely(!metric_->is_matched(meta_, qmeta))) {
      return IndexError_Mismatch;
    }
    error_code =
        searcher_->search_bf_impl(query, qmeta, context->searcher_context());
  }

  if (error_code == 0) {
    if (metric_->support_normalize()) {
      for (auto &it : const_cast<IndexDocumentList &>(
               context->searcher_context()->result())) {
        metric_->normalize(it.mutable_score());
      }
    }
    if (reformer_) {
      error_code =
          reformer_->normalize(query, qmeta,
                               const_cast<IndexDocumentList &>(
                                   context->searcher_context()->result()));
    }
  }
  return error_code;
}

int IndexFlow::search_impl(const void *query, const IndexQueryMeta &qmeta,
                           Context::Pointer &context) const {
  if (ailego_unlikely(!query || !context)) {
    return IndexError_InvalidArgument;
  }

  int error_code = 0;
  if (reformer_) {
    IndexQueryMeta new_qmeta;
    error_code = reformer_->transform(query, qmeta, context->mutable_features(),
                                      &new_qmeta);
    if (error_code == 0) {
      if (ailego_unlikely(!metric_->is_matched(meta_, new_qmeta))) {
        return IndexError_Mismatch;
      }
      error_code = searcher_->search_impl(
          reinterpret_cast<const void *>(context->features().data()), new_qmeta,
          context->searcher_context());
    }
  } else {
    if (ailego_unlikely(!metric_->is_matched(meta_, qmeta))) {
      return IndexError_Mismatch;
    }
    error_code =
        searcher_->search_impl(query, qmeta, context->searcher_context());
  }

  if (error_code == 0) {
    if (metric_->support_normalize()) {
      for (auto &it : const_cast<IndexDocumentList &>(
               context->searcher_context()->result())) {
        metric_->normalize(it.mutable_score());
      }
    }
    if (reformer_) {
      error_code =
          reformer_->normalize(query, qmeta,
                               const_cast<IndexDocumentList &>(
                                   context->searcher_context()->result()));
    }
  }
  return error_code;
}

int IndexFlow::search_bf_impl(const void *query, const IndexQueryMeta &qmeta,
                              uint32_t count, Context::Pointer &context) const {
  if (ailego_unlikely(!query || !count || !context)) {
    return IndexError_InvalidArgument;
  }

  int error_code = 0;
  if (reformer_) {
    IndexQueryMeta new_qmeta;
    error_code = reformer_->transform(query, qmeta, count,
                                      context->mutable_features(), &new_qmeta);
    if (error_code == 0) {
      if (ailego_unlikely(!metric_->is_matched(meta_, new_qmeta))) {
        return IndexError_Mismatch;
      }
      error_code = searcher_->search_bf_impl(
          reinterpret_cast<const void *>(context->features().data()), new_qmeta,
          count, context->searcher_context());
    }
  } else {
    if (ailego_unlikely(!metric_->is_matched(meta_, qmeta))) {
      return IndexError_Mismatch;
    }
    error_code = searcher_->search_bf_impl(query, qmeta, count,
                                           context->searcher_context());
  }

  if (error_code == 0) {
    if (metric_->support_normalize()) {
      for (uint32_t i = 0; i < count; ++i) {
        IndexDocumentList &result = const_cast<IndexDocumentList &>(
            context->searcher_context()->result(i));

        for (auto &it : result) {
          metric_->normalize(it.mutable_score());
        }
      }
    }
    if (reformer_) {
      size_t offset = 0;
      for (uint32_t i = 0; i < count; ++i) {
        error_code = reformer_->normalize(
            reinterpret_cast<const uint8_t *>(query) + offset, qmeta,
            const_cast<IndexDocumentList &>(
                context->searcher_context()->result(i)));
        if (error_code != 0) {
          break;
        }
        offset += qmeta.element_size();
      }
    }
  }
  return error_code;
}

int IndexFlow::search_impl(const void *query, const IndexQueryMeta &qmeta,
                           uint32_t count, Context::Pointer &context) const {
  if (ailego_unlikely(!query || !count || !context)) {
    return IndexError_InvalidArgument;
  }

  int error_code = 0;
  if (reformer_) {
    IndexQueryMeta new_qmeta;
    error_code = reformer_->transform(query, qmeta, count,
                                      context->mutable_features(), &new_qmeta);
    if (error_code == 0) {
      if (ailego_unlikely(!metric_->is_matched(meta_, new_qmeta))) {
        return IndexError_Mismatch;
      }
      error_code = searcher_->search_impl(
          reinterpret_cast<const void *>(context->features().data()), new_qmeta,
          count, context->searcher_context());
    }
  } else {
    if (ailego_unlikely(!metric_->is_matched(meta_, qmeta))) {
      return IndexError_Mismatch;
    }
    error_code = searcher_->search_impl(query, qmeta, count,
                                        context->searcher_context());
  }

  if (error_code == 0) {
    if (metric_->support_normalize()) {
      for (uint32_t i = 0; i < count; ++i) {
        IndexDocumentList &result = const_cast<IndexDocumentList &>(
            context->searcher_context()->result(i));

        for (auto &it : result) {
          metric_->normalize(it.mutable_score());
        }
      }
    }
    if (reformer_) {
      size_t offset = 0;
      for (uint32_t i = 0; i < count; ++i) {
        error_code = reformer_->normalize(
            reinterpret_cast<const uint8_t *>(query) + offset, qmeta,
            const_cast<IndexDocumentList &>(
                context->searcher_context()->result(i)));
        if (error_code != 0) {
          break;
        }
        offset += qmeta.element_size();
      }
    }
  }
  return error_code;
}

// Index Sparse Flow
int IndexSparseFlow::set_storage(const std::string &name,
                                 const ailego::Params &params) {
  storage_ = IndexFactory::CreateStorage(name);
  if (!storage_) {
    LOG_ERROR("Failed to create a index storage with name: %s", name.c_str());
    return IndexError_NoExist;
  }
  int ret = storage_->init(params);
  if (ret < 0) {
    storage_ = nullptr;
    LOG_ERROR("Failed to initialize index storage %s", name.c_str());
    return ret;
  }
  return 0;
}

int IndexSparseFlow::set_searcher(IndexSearcher::Pointer searcher) {
  user_searcher_ = searcher;

  return 0;
}

int IndexSparseFlow::set_searcher(const std::string &name,
                                  const ailego::Params &params) {
  user_searcher_ = IndexFactory::CreateSearcher(name);
  if (!user_searcher_) {
    LOG_ERROR("Failed to create a index sparse searcher with name: %s",
              name.c_str());
    return IndexError_NoExist;
  }
  int ret = user_searcher_->init(params);
  if (ret < 0) {
    user_searcher_ = nullptr;
    LOG_ERROR("Failed to initialize index sparse searcher %s", name.c_str());
    return ret;
  }
  return 0;
}

int IndexSparseFlow::set_reformer(const std::string &name,
                                  const ailego::Params &params) {
  user_reformer_ = IndexFactory::CreateReformer(name);
  if (!user_reformer_) {
    LOG_ERROR("Failed to create a index sparse reformer with name: %s",
              name.c_str());
    return IndexError_NoExist;
  }
  int ret = user_reformer_->init(params);
  if (ret < 0) {
    user_reformer_ = nullptr;
    LOG_ERROR("Failed to initialize index sparse reformer %s", name.c_str());
    return ret;
  }
  return 0;
}

int IndexSparseFlow::set_metric(const std::string &name,
                                const ailego::Params &params) {
  if (!IndexFactory::HasMetric(name)) {
    LOG_ERROR("The index metric with name %s does not exist.", name.c_str());
    return IndexError_NoExist;
  }
  user_metric_name_ = name;
  user_metric_params_ = params;
  return 0;
}

int IndexSparseFlow::load(const std::string &path) {
  // Prepare storage
  if (!storage_) {
    this->set_storage(INDEX_FLOW_STORAGE_DEFAULT, ailego::Params());
  }

  if (!storage_) {
    LOG_ERROR("The index storage is uninitialized.");
    return IndexError_Uninitialized;
  }

  int ret = storage_->open(path, false);
  if (ret != 0) {
    LOG_ERROR("Failed to load index with storage %s", storage_->name().c_str());
    return ret;
  }

  ret = IndexHelper::DeserializeFromStorage(storage_.get(), &meta_);
  if (ret != 0) {
    LOG_ERROR("Failed to deserialize index meta with storage %s",
              storage_->name().c_str());
    return ret;
  }

  ret = load_internal();
  if (ret != 0) {
    LOG_ERROR("Failed to load index with storage %s", storage_->name().c_str());
    return ret;
  }

  return 0;
}

int IndexSparseFlow::load_internal() {
  // Prepare metric
  const std::string &metric_name =
      user_metric_name_.empty() ? meta_.metric_name() : user_metric_name_;
  const ailego::Params &metric_params =
      user_metric_name_.empty() ? meta_.metric_params() : user_metric_params_;
  if (metric_name.empty()) {
    LOG_ERROR("The metric name from index file is empty.");
    return IndexError_NoExist;
  }
  metric_ = IndexFactory::CreateMetric(metric_name);
  if (!metric_) {
    LOG_ERROR("Failed to create a index metric with name: %s",
              metric_name.c_str());
    return IndexError_NoExist;
  }
  int ret = metric_->init(meta_, metric_params);
  if (ret < 0) {
    LOG_ERROR("Failed to initialize index metric %s", metric_name.c_str());
    metric_ = nullptr;
    return ret;
  }

  auto query_metric = metric_->query_metric();
  if (query_metric) {
    metric_ = query_metric;
  }

  // Prepare reformer
  if (!user_reformer_) {
    const std::string &reformer_name = meta_.reformer_name();
    if (!reformer_name.empty()) {
      reformer_ = IndexFactory::CreateReformer(reformer_name);
      if (!reformer_) {
        LOG_ERROR("Failed to create a index sparse reformer with name: %s",
                  reformer_name.c_str());
        return IndexError_NoExist;
      }
      ret = reformer_->init(meta_.reformer_params());
      if (ret < 0) {
        LOG_ERROR("Failed to initialize index reformer %s",
                  reformer_name.c_str());
        reformer_ = nullptr;
        return ret;
      }
    }
  } else {
    // Using user reformer
    reformer_ = user_reformer_;
  }

  if (reformer_) {
    ret = reformer_->load(storage_);
    if (ret < 0) {
      LOG_ERROR("Failed to load index with reformer %s, storage %s",
                reformer_->name().c_str(), storage_->name().c_str());
      return ret;
    }
  }

  // Prepare searcher
  if (!user_searcher_) {
    const std::string &name = meta_.searcher_name();
    if (name.empty()) {
      LOG_ERROR("The searcher name from index file is empty.");
      return IndexError_NoExist;
    }
    searcher_ = IndexFactory::CreateSearcher(name);
    if (!searcher_) {
      LOG_ERROR("Failed to create a index searcher with name: %s",
                name.c_str());
      return IndexError_NoExist;
    }
    ret = searcher_->init(meta_.searcher_params());
    if (ret < 0) {
      LOG_ERROR("Failed to initialize index searcher %s", name.c_str());
      searcher_ = nullptr;
      return ret;
    }
  } else {
    // Using user searcher
    searcher_ = user_searcher_;
  }

  ret = searcher_->load(storage_, metric_);
  if (ret < 0) {
    LOG_ERROR("Failed to load index with searcher %s, storage %s, metric %s",
              searcher_->name().c_str(), storage_->name().c_str(),
              metric_->name().c_str());
    return ret;
  }

  // searcher_->print_all_neighbour();

  return 0;
}

int IndexSparseFlow::unload(void) {
  if (searcher_) {
    int ret = searcher_->unload();
    if (ret < 0) {
      LOG_WARN("Unload index searcher %s error, %d", searcher_->name().c_str(),
               ret);
    }
    searcher_ = nullptr;
  }
  if (reformer_) {
    int ret = reformer_->unload();
    if (ret < 0) {
      LOG_WARN("Unload index reformer %s error, %d", reformer_->name().c_str(),
               ret);
    }
    reformer_ = nullptr;
  }
  if (metric_) {
    int ret = metric_->cleanup();
    if (ret < 0) {
      LOG_WARN("Cleanup index metric %s error, %d", metric_->name().c_str(),
               ret);
    }
    metric_ = nullptr;
  }
  if (storage_) {
    int ret = storage_->cleanup();
    if (ret < 0) {
      LOG_WARN("Unload index searcher %s error, %d", storage_->name().c_str(),
               ret);
    }
    storage_ = nullptr;
  }
  return 0;
}

int IndexSparseFlow::search_bf_impl(const uint32_t sparse_count,
                                    const uint32_t *sparse_indices,
                                    const void *sparse_query,
                                    const IndexQueryMeta &qmeta,
                                    Context::Pointer &context) const {
  if (ailego_unlikely(!context)) {
    return IndexError_InvalidArgument;
  }

  int error_code = 0;
  if (reformer_) {
    std::string ovec;
    IndexQueryMeta new_qmeta;
    error_code = reformer_->transform(sparse_count, sparse_indices,
                                      sparse_query, qmeta, &ovec, &new_qmeta);
    if (ailego_unlikely(!metric_->is_matched(meta_, new_qmeta))) {
      return IndexError_Mismatch;
    }
    if (error_code == 0) {
      error_code =
          searcher_->search_bf_impl(sparse_count, sparse_indices, ovec.data(),
                                    new_qmeta, context->searcher_context());
    }
  } else {
    if (ailego_unlikely(!metric_->is_matched(meta_, qmeta))) {
      return IndexError_Mismatch;
    }
    error_code =
        searcher_->search_bf_impl(sparse_count, sparse_indices, sparse_query,
                                  qmeta, context->searcher_context());
  }

  if (error_code == 0) {
    if (metric_->support_normalize()) {
      for (auto &it : const_cast<IndexDocumentList &>(
               context->searcher_context()->result())) {
        metric_->normalize(it.mutable_score());
      }
    }
  }
  return error_code;
}

int IndexSparseFlow::search_impl(const uint32_t sparse_count,
                                 const uint32_t *sparse_indices,
                                 const void *sparse_query,
                                 const IndexQueryMeta &qmeta,
                                 Context::Pointer &context) const {
  if (ailego_unlikely(!context)) {
    return IndexError_InvalidArgument;
  }

  int error_code = 0;
  if (reformer_) {
    std::string ovec;
    IndexQueryMeta new_qmeta;
    error_code = reformer_->transform(sparse_count, sparse_indices,
                                      sparse_query, qmeta, &ovec, &new_qmeta);
    if (ailego_unlikely(!metric_->is_matched(meta_, new_qmeta))) {
      return IndexError_Mismatch;
    }
    if (error_code == 0) {
      error_code =
          searcher_->search_impl(sparse_count, sparse_indices, ovec.data(),
                                 new_qmeta, context->searcher_context());
    }
  } else {
    if (ailego_unlikely(!metric_->is_matched(meta_, qmeta))) {
      return IndexError_Mismatch;
    }
    error_code =
        searcher_->search_impl(sparse_count, sparse_indices, sparse_query,
                               qmeta, context->searcher_context());
  }

  if (error_code == 0) {
    if (metric_->support_normalize()) {
      for (auto &it : const_cast<IndexDocumentList &>(
               context->searcher_context()->result())) {
        metric_->normalize(it.mutable_score());
      }
    }
  }
  return error_code;
}

int IndexSparseFlow::search_bf_impl(const uint32_t *sparse_count,
                                    const uint32_t *sparse_indices,
                                    const void *sparse_query,
                                    const IndexQueryMeta &qmeta, uint32_t count,
                                    Context::Pointer &context) const {
  if (ailego_unlikely(!count || !context)) {
    return IndexError_InvalidArgument;
  }

  int error_code = 0;
  if (reformer_) {
    std::string ovec;
    IndexQueryMeta new_qmeta;
    error_code =
        reformer_->transform(sparse_count, sparse_indices, sparse_query, qmeta,
                             count, &ovec, &new_qmeta);

    if (ailego_unlikely(!metric_->is_matched(meta_, new_qmeta))) {
      return IndexError_Mismatch;
    }

    if (error_code == 0) {
      error_code = searcher_->search_bf_impl(sparse_count, sparse_indices,
                                             ovec.data(), new_qmeta, count,
                                             context->searcher_context());
    }
  } else {
    if (ailego_unlikely(!metric_->is_matched(meta_, qmeta))) {
      return IndexError_Mismatch;
    }
    error_code =
        searcher_->search_bf_impl(sparse_count, sparse_indices, sparse_query,
                                  qmeta, count, context->searcher_context());
  }

  if (error_code == 0) {
    if (metric_->support_normalize()) {
      for (uint32_t i = 0; i < count; ++i) {
        IndexDocumentList &result = const_cast<IndexDocumentList &>(
            context->searcher_context()->result(i));

        for (auto &it : result) {
          metric_->normalize(it.mutable_score());
        }
      }
    }
  }
  return error_code;
}

int IndexSparseFlow::search_impl(const uint32_t *sparse_count,
                                 const uint32_t *sparse_indices,
                                 const void *sparse_query,
                                 const IndexQueryMeta &qmeta, uint32_t count,
                                 Context::Pointer &context) const {
  if (ailego_unlikely(!count || !context)) {
    return IndexError_InvalidArgument;
  }

  int error_code = 0;
  if (reformer_) {
    std::string ovec;
    IndexQueryMeta new_qmeta;
    error_code =
        reformer_->transform(sparse_count, sparse_indices, sparse_query, qmeta,
                             count, &ovec, &new_qmeta);

    if (ailego_unlikely(!metric_->is_matched(meta_, new_qmeta))) {
      return IndexError_Mismatch;
    }

    if (error_code == 0) {
      error_code =
          searcher_->search_impl(sparse_count, sparse_indices, ovec.data(),
                                 new_qmeta, count, context->searcher_context());
    }
  } else {
    if (ailego_unlikely(!metric_->is_matched(meta_, qmeta))) {
      return IndexError_Mismatch;
    }
    error_code =
        searcher_->search_impl(sparse_count, sparse_indices, sparse_query,
                               qmeta, count, context->searcher_context());
  }

  if (error_code == 0) {
    if (metric_->support_normalize()) {
      for (uint32_t i = 0; i < count; ++i) {
        IndexDocumentList &result = const_cast<IndexDocumentList &>(
            context->searcher_context()->result(i));

        for (auto &it : result) {
          metric_->normalize(it.mutable_score());
        }
      }
    }
  }
  return error_code;
}

}  // namespace core
}  // namespace zvec
