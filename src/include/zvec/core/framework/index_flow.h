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
#pragma once

#include <memory>
#include <zvec/core/framework/index_reformer.h>
#include <zvec/core/framework/index_searcher.h>
namespace zvec {
namespace core {

/*! Index Flow
 */
class IndexFlow {
 public:
  /*! Index Flow Context
   */
  class Context {
   public:
    //! Index Flow Pointer
    typedef std::unique_ptr<Context> Pointer;

    //! Index Flow UPointer
    typedef std::unique_ptr<Context> UPointer;

    //! Retrieve searcher context
    IndexSearcher::Context::Pointer &searcher_context(void) {
      return searcher_context_;
    }

    //! Set topk of search result
    void set_topk(uint32_t topk) {
      return searcher_context_->set_topk(topk);
    }

    //! Retrieve search results
    const IndexDocumentList &result(void) const {
      return searcher_context_->result();
    }

    //! Retrieve search result with index
    const IndexDocumentList &result(size_t index) const {
      return searcher_context_->result(index);
    }

    //! Set the filter of context
    template <typename T>
    void set_filter(T &&func) {
      searcher_context_->set_filter(std::forward<T>(func));
    }

    //! Reset the filter of context
    void reset_filter(void) {
      searcher_context_->reset_filter();
    }

    //! Set mode of debug
    void set_debug_mode(bool enable) {
      searcher_context_->set_debug_mode(enable);
    }

    //! Update the parameters of context
    int update(const ailego::Params &params) {
      return searcher_context_->update(params);
    }

    //! Retrieve debug information
    std::string debug_string(void) const {
      return searcher_context_->debug_string();
    }

    //! Retrieve magic number
    uint32_t magic(void) const {
      return searcher_context_->magic();
    }

    //! Retrieve mode of debug
    bool debug_mode(void) const {
      return searcher_context_->debug_mode();
    }

    //! Retrieve mutable features buffer
    std::string *mutable_features(void) {
      return &features_;
    }

    //! Retrieve features buffer
    const std::string &features(void) const {
      return features_;
    }

   protected:
    friend class IndexFlow;

    //! Constructor
    Context(IndexSearcher::Context::Pointer &&ctx)
        : searcher_context_(std::move(ctx)) {}

   private:
    IndexSearcher::Context::Pointer searcher_context_{};
    std::string features_{};
  };

  //! Constructor
  IndexFlow(void) {}

  //! Constructor
  IndexFlow(IndexFlow &&rhs)
      : storage_(std::move(rhs.storage_)),
        reformer_(std::move(rhs.reformer_)),
        searcher_(std::move(rhs.searcher_)),
        metric_(std::move(rhs.metric_)),
        user_reformer_(std::move(rhs.user_reformer_)),
        user_searcher_(std::move(rhs.user_searcher_)),
        user_metric_name_(std::move(rhs.user_metric_name_)),
        user_metric_params_(std::move(rhs.user_metric_params_)) {}

  //! Assignment
  IndexFlow &operator=(IndexFlow &&rhs) {
    storage_ = std::move(rhs.storage_);
    reformer_ = std::move(rhs.reformer_);
    searcher_ = std::move(rhs.searcher_);
    metric_ = std::move(rhs.metric_);
    user_reformer_ = std::move(rhs.user_reformer_);
    user_searcher_ = std::move(rhs.user_searcher_);
    user_metric_name_ = std::move(rhs.user_metric_name_);
    user_metric_params_ = std::move(rhs.user_metric_params_);
    return *this;
  }

  //! Retrieve index meta
  const IndexMeta &meta(void) const {
    return meta_;
  }

  //! Retrieve index reformer
  const IndexReformer::Pointer &reformer(void) const {
    return reformer_;
  }

  //! Retrieve index searcher
  const IndexSearcher::Pointer &searcher(void) const {
    return searcher_;
  }

  //! Retrieve index metric
  const IndexMetric::Pointer &metric(void) const {
    return metric_;
  }

  //! Set the index storage (user)
  int set_storage(const std::string &name, const ailego::Params &params);

  //! Set the index reformer (user)
  int set_reformer(const std::string &name, const ailego::Params &params);

  //! Set the index searcher (user)
  int set_searcher(const std::string &name, const ailego::Params &params);

  //! Set the index searcher (user)
  int set_searcher(IndexSearcher::Pointer searcher);

  //! Set the index metric (user)
  int set_metric(const std::string &name, const ailego::Params &params);

  //! Load index
  int load(const std::string &path);

  //! Unload index
  int unload(void);

  //! Similarity brute force search
  int search_bf_impl(const void *query, const IndexQueryMeta &qmeta,
                     Context::Pointer &context) const;

  //! Similarity search
  int search_impl(const void *query, const IndexQueryMeta &qmeta,
                  Context::Pointer &context) const;

  //! Similarity brute force search
  int search_bf_impl(const void *query, const IndexQueryMeta &qmeta,
                     uint32_t count, Context::Pointer &context) const;

  //! Similarity search
  int search_impl(const void *query, const IndexQueryMeta &qmeta,
                  uint32_t count, Context::Pointer &context) const;

  //! Similarity search (FP16)
  template <IndexMeta::DataType DT,
            typename = typename std::enable_if<
                DT == IndexMeta::DataType::DT_FP16>::type>
  int search_bf(const ailego::Float16 *vec, size_t dim,
                Context::Pointer &context) const {
    return this->search_bf_impl(vec, IndexQueryMeta(DT, dim), context);
  }

  //! Similarity search (FP32)
  template <IndexMeta::DataType DT,
            typename = typename std::enable_if<
                DT == IndexMeta::DataType::DT_FP32>::type>
  int search_bf(const float *vec, size_t dim, Context::Pointer &context) const {
    return this->search_bf_impl(vec, IndexQueryMeta(DT, dim), context);
  }

  //! Similarity search (INT8)
  template <IndexMeta::DataType DT,
            typename = typename std::enable_if<
                DT == IndexMeta::DataType::DT_INT8>::type>
  int search_bf(const int8_t *vec, size_t dim,
                Context::Pointer &context) const {
    return this->search_bf_impl(vec, IndexQueryMeta(DT, dim), context);
  }

  //! Similarity search (INT4)
  template <IndexMeta::DataType DT,
            typename = typename std::enable_if<
                DT == IndexMeta::DataType::DT_INT4>::type>
  int search_bf(const uint8_t *vec, size_t dim,
                Context::Pointer &context) const {
    return this->search_bf_impl(vec, IndexQueryMeta(DT, dim), context);
  }

  //! Similarity search (BINARY)
  template <IndexMeta::DataType DT,
            typename = typename std::enable_if<
                DT == IndexMeta::DataType::DT_BINARY32>::type>
  int search_bf(const uint32_t *vec, size_t dim,
                Context::Pointer &context) const {
    return this->search_bf_impl(vec, IndexQueryMeta(DT, dim), context);
  }

  //! Similarity search in batch (FP16)
  template <IndexMeta::DataType DT,
            typename = typename std::enable_if<
                DT == IndexMeta::DataType::DT_FP16>::type>
  int search_bf(const ailego::Float16 *vec, size_t dim, size_t rows,
                Context::Pointer &context) const {
    return this->search_bf_impl(vec, IndexQueryMeta(DT, dim), rows, context);
  }

  //! Similarity search in batch (FP32)
  template <IndexMeta::DataType DT,
            typename = typename std::enable_if<
                DT == IndexMeta::DataType::DT_FP32>::type>
  int search_bf(const float *vec, size_t dim, size_t rows,
                Context::Pointer &context) const {
    return this->search_bf_impl(vec, IndexQueryMeta(DT, dim), rows, context);
  }

  //! Similarity search in batch (INT8)
  template <IndexMeta::DataType DT,
            typename = typename std::enable_if<
                DT == IndexMeta::DataType::DT_INT8>::type>
  int search_bf(const int8_t *vec, size_t dim, size_t rows,
                Context::Pointer &context) const {
    return this->search_bf_impl(vec, IndexQueryMeta(DT, dim), rows, context);
  }

  //! Similarity search in batch (INT4)
  template <IndexMeta::DataType DT,
            typename = typename std::enable_if<
                DT == IndexMeta::DataType::DT_INT4>::type>
  int search_bf(const uint8_t *vec, size_t dim, size_t rows,
                Context::Pointer &context) const {
    return this->search_bf_impl(vec, IndexQueryMeta(DT, dim), rows, context);
  }

  //! Similarity Search in batch (BINARY)
  template <IndexMeta::DataType DT,
            typename = typename std::enable_if<
                DT == IndexMeta::DataType::DT_BINARY32>::type>
  int search_bf(const uint32_t *vec, size_t dim, size_t rows,
                Context::Pointer &context) const {
    return this->search_bf_impl(vec, IndexQueryMeta(DT, dim), rows, context);
  }

  //! Similarity search (FP16)
  template <IndexMeta::DataType DT,
            typename = typename std::enable_if<
                DT == IndexMeta::DataType::DT_FP16>::type>
  int search(const ailego::Float16 *vec, size_t dim,
             Context::Pointer &context) const {
    return this->search_impl(vec, IndexQueryMeta(DT, dim), context);
  }

  //! Similarity search (FP32)
  template <IndexMeta::DataType DT,
            typename = typename std::enable_if<
                DT == IndexMeta::DataType::DT_FP32>::type>
  int search(const float *vec, size_t dim, Context::Pointer &context) const {
    return this->search_impl(vec, IndexQueryMeta(DT, dim), context);
  }

  //! Similarity search (INT8)
  template <IndexMeta::DataType DT,
            typename = typename std::enable_if<
                DT == IndexMeta::DataType::DT_INT8>::type>
  int search(const int8_t *vec, size_t dim, Context::Pointer &context) const {
    return this->search_impl(vec, IndexQueryMeta(DT, dim), context);
  }

  //! Similarity search (INT4)
  template <IndexMeta::DataType DT,
            typename = typename std::enable_if<
                DT == IndexMeta::DataType::DT_INT4>::type>
  int search(const uint8_t *vec, size_t dim, Context::Pointer &context) const {
    return this->search_impl(vec, IndexQueryMeta(DT, dim), context);
  }

  //! Similarity search (BINARY32)
  template <IndexMeta::DataType DT,
            typename = typename std::enable_if<
                DT == IndexMeta::DataType::DT_BINARY32>::type>
  int search(const uint32_t *vec, size_t dim, Context::Pointer &context) const {
    return this->search_impl(vec, IndexQueryMeta(DT, dim), context);
  }

  //! Similarity search in batch (FP16)
  template <IndexMeta::DataType DT,
            typename = typename std::enable_if<
                DT == IndexMeta::DataType::DT_FP16>::type>
  int search(const ailego::Float16 *vec, size_t dim, size_t rows,
             Context::Pointer &context) const {
    return this->search_impl(vec, IndexQueryMeta(DT, dim), rows, context);
  }

  //! Similarity search in batch (FP32)
  template <IndexMeta::DataType DT,
            typename = typename std::enable_if<
                DT == IndexMeta::DataType::DT_FP32>::type>
  int search(const float *vec, size_t dim, size_t rows,
             Context::Pointer &context) const {
    return this->search_impl(vec, IndexQueryMeta(DT, dim), rows, context);
  }

  //! Similarity search in batch (INT8)
  template <IndexMeta::DataType DT,
            typename = typename std::enable_if<
                DT == IndexMeta::DataType::DT_INT8>::type>
  int search(const int8_t *vec, size_t dim, size_t rows,
             Context::Pointer &context) const {
    return this->search_impl(vec, IndexQueryMeta(DT, dim), rows, context);
  }

  //! Similarity search in batch (INT4)
  template <IndexMeta::DataType DT,
            typename = typename std::enable_if<
                DT == IndexMeta::DataType::DT_INT4>::type>
  int search(const uint8_t *vec, size_t dim, size_t rows,
             Context::Pointer &context) const {
    return this->search_impl(vec, IndexQueryMeta(DT, dim), rows, context);
  }

  //! Similarity Search in batch (BINARY)
  template <IndexMeta::DataType DT,
            typename = typename std::enable_if<
                DT == IndexMeta::DataType::DT_BINARY32>::type>
  int search(const uint32_t *vec, size_t dim, size_t rows,
             Context::Pointer &context) const {
    return this->search_impl(vec, IndexQueryMeta(DT, dim), rows, context);
  }

  //! Create a flow context
  Context::Pointer create_context(void) const {
    return Context::Pointer(new Context(searcher_->create_context()));
  }

 private:
  //! Disable them
  IndexFlow(const IndexFlow &) = delete;
  IndexFlow &operator=(const IndexFlow &) = delete;

  int load_internal();

  //! Members
  IndexMeta meta_{};
  IndexStorage::Pointer storage_{};
  IndexReformer::Pointer reformer_{};
  IndexSearcher::Pointer searcher_{};
  IndexMetric::Pointer metric_{};
  IndexReformer::Pointer user_reformer_{};
  IndexSearcher::Pointer user_searcher_{};
  std::string user_metric_name_{};
  ailego::Params user_metric_params_{};
};


/*! Index Sparse Flow
 */
class IndexSparseFlow {
 public:
  /*! Index Sparse Flow Context
   */
  class Context {
   public:
    //! Index Flow Pointer
    typedef std::unique_ptr<Context> Pointer;

    //! Index Flow UPointer
    typedef std::unique_ptr<Context> UPointer;

    //! Retrieve searcher context
    IndexSearcher::Context::Pointer &searcher_context(void) {
      return searcher_context_;
    }

    //! Set topk of search result
    void set_topk(uint32_t topk) {
      return searcher_context_->set_topk(topk);
    }

    //! Retrieve search results
    const IndexDocumentList &result(void) const {
      return searcher_context_->result();
    }

    //! Retrieve search result with index
    const IndexDocumentList &result(size_t index) const {
      return searcher_context_->result(index);
    }

    //! Set the filter of context
    template <typename T>
    void set_filter(T &&func) {
      searcher_context_->set_filter(std::forward<T>(func));
    }

    //! Reset the filter of context
    void reset_filter(void) {
      searcher_context_->reset_filter();
    }

    //! Set mode of debug
    void set_debug_mode(bool enable) {
      searcher_context_->set_debug_mode(enable);
    }

    //! Update the parameters of context
    int update(const ailego::Params &params) {
      return searcher_context_->update(params);
    }

    //! Retrieve debug information
    std::string debug_string(void) const {
      return searcher_context_->debug_string();
    }

    //! Retrieve magic number
    uint32_t magic(void) const {
      return searcher_context_->magic();
    }

    //! Retrieve mode of debug
    bool debug_mode(void) const {
      return searcher_context_->debug_mode();
    }

    //! Retrieve mutable features buffer
    std::string *mutable_features(void) {
      return &features_;
    }

    //! Retrieve features buffer
    const std::string &features(void) const {
      return features_;
    }

   protected:
    friend class IndexSparseFlow;

    //! Constructor
    Context(IndexSearcher::Context::Pointer &&ctx)
        : searcher_context_(std::move(ctx)) {}

   private:
    IndexSearcher::Context::Pointer searcher_context_{};
    std::string features_{};
  };

  //! Constructor
  IndexSparseFlow(void) {}

  //! Constructor
  IndexSparseFlow(IndexSparseFlow &&rhs)
      : storage_(std::move(rhs.storage_)),
        reformer_(std::move(rhs.reformer_)),
        searcher_(std::move(rhs.searcher_)),
        metric_(std::move(rhs.metric_)),
        user_reformer_(std::move(rhs.user_reformer_)),
        user_searcher_(std::move(rhs.user_searcher_)),
        user_metric_name_(std::move(rhs.user_metric_name_)),
        user_metric_params_(std::move(rhs.user_metric_params_)) {}

  //! Assignment
  IndexSparseFlow &operator=(IndexSparseFlow &&rhs) {
    storage_ = std::move(rhs.storage_);
    reformer_ = std::move(rhs.reformer_);
    searcher_ = std::move(rhs.searcher_);
    metric_ = std::move(rhs.metric_);
    user_reformer_ = std::move(rhs.user_reformer_);
    user_searcher_ = std::move(rhs.user_searcher_);
    user_metric_name_ = std::move(rhs.user_metric_name_);
    user_metric_params_ = std::move(rhs.user_metric_params_);
    return *this;
  }

  //! Retrieve index sparse meta
  const IndexMeta &meta(void) const {
    return meta_;
  }

  //! Retrieve index reformer
  const IndexReformer::Pointer &reformer(void) const {
    return reformer_;
  }

  //! Retrieve index searcher
  const IndexSearcher::Pointer &searcher(void) const {
    return searcher_;
  }

  //! Retrieve index metric
  const IndexMetric::Pointer &metric(void) const {
    return metric_;
  }

  //! Set the index storage (user)
  int set_storage(const std::string &name, const ailego::Params &params);

  //! Set the index reformer (user)
  int set_reformer(const std::string &name, const ailego::Params &params);

  //! Set the index searcher (user)
  int set_searcher(const std::string &name, const ailego::Params &params);

  //! Set the index searcher (user)
  int set_searcher(IndexSearcher::Pointer searcher);

  //! Set the index metric (user)
  int set_metric(const std::string &name, const ailego::Params &params);

  //! Load index
  int load(const std::string &path);

  //! Unload index
  int unload(void);

  //! Similarity search with sparse inputs
  int search_impl(const uint32_t sparse_count, const uint32_t *sparse_indices,
                  const void *sparse_query, const IndexQueryMeta &qmeta,
                  Context::Pointer &context) const;

  //! Similarity search with sparse inputs
  int search_impl(const uint32_t *sparse_count, const uint32_t *sparse_indices,
                  const void *sparse_query, const IndexQueryMeta &qmeta,
                  uint32_t count, Context::Pointer &context) const;

  //! Similarity brute force search and sparse inputs
  int search_bf_impl(const uint32_t sparse_count,
                     const uint32_t *sparse_indices, const void *sparse_query,
                     const IndexQueryMeta &qmeta,
                     Context::Pointer &context) const;

  //! Similarity brute force search with sparse inputs
  int search_bf_impl(const uint32_t *sparse_count,
                     const uint32_t *sparse_indices, const void *sparse_query,
                     const IndexQueryMeta &qmeta, uint32_t count,
                     Context::Pointer &context) const;

  //! Similarity search (FP16)
  template <IndexMeta::DataType DT,
            typename = typename std::enable_if<
                DT == IndexMeta::DataType::DT_FP16>::type>
  int search_bf(const uint32_t sparse_count, const uint32_t *sparse_indices,
                const ailego::Float16 *sparse_query,
                Context::Pointer &context) const {
    return this->search_bf_impl(sparse_count, sparse_indices, sparse_query,
                                IndexQueryMeta(DT), context);
  }

  //! Similarity search (FP32)
  template <IndexMeta::DataType DT,
            typename = typename std::enable_if<
                DT == IndexMeta::DataType::DT_FP32>::type>
  int search_bf(const uint32_t sparse_count, const uint32_t *sparse_indices,
                const float *sparse_query, Context::Pointer &context) const {
    return this->search_bf_impl(sparse_count, sparse_indices, sparse_query,
                                IndexQueryMeta(DT), context);
  }

  //! Similarity search (FP16)
  template <IndexMeta::DataType DT,
            typename = typename std::enable_if<
                DT == IndexMeta::DataType::DT_FP16>::type>
  int search(const uint32_t sparse_count, const uint32_t *sparse_indices,
             const ailego::Float16 *sparse_query,
             Context::Pointer &context) const {
    return this->search_impl(sparse_count, sparse_indices, sparse_query,
                             IndexQueryMeta(DT), context);
  }

  //! Similarity search (FP32)
  template <IndexMeta::DataType DT,
            typename = typename std::enable_if<
                DT == IndexMeta::DataType::DT_FP32>::type>
  int search(const uint32_t sparse_count, const uint32_t *sparse_indices,
             const float *sparse_query, Context::Pointer &context) const {
    return this->search_impl(sparse_count, sparse_indices, sparse_query,
                             IndexQueryMeta(DT), context);
  }

  //! Create a flow context
  Context::Pointer create_context(void) const {
    return Context::Pointer(new Context(searcher_->create_context()));
  }

 private:
  //! Disable them
  IndexSparseFlow(const IndexSparseFlow &) = delete;
  IndexSparseFlow &operator=(const IndexSparseFlow &) = delete;

  int load_internal();

  //! Members
  IndexMeta meta_{};
  IndexStorage::Pointer storage_{};
  IndexReformer::Pointer reformer_{};
  IndexSearcher::Pointer searcher_{};
  IndexMetric::Pointer metric_{};
  IndexReformer::Pointer user_reformer_{};
  IndexSearcher::Pointer user_searcher_{};
  std::string user_metric_name_{};
  ailego::Params user_metric_params_{};
};

}  // namespace core
}  // namespace zvec
