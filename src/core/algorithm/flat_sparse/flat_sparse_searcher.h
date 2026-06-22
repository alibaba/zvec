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

#include "flat_sparse_searcher_entity.h"

namespace zvec {
namespace core {

class FlatSparseSearcher : public IndexSearcher {
 public:
  static const uint32_t VERSION;

 public:
  using ContextPointer = IndexSearcher::Context::Pointer;

 public:
  FlatSparseSearcher(void);
  ~FlatSparseSearcher(void) override;

  FlatSparseSearcher(const FlatSparseSearcher &) = delete;
  FlatSparseSearcher &operator=(const FlatSparseSearcher &) = delete;

 public:
  //! Initialize Searcher
  int init(const ailego::Params &params) override;

  //! Cleanup Searcher
  int cleanup(void) override;

  //! Load Index from storage
  int load(IndexStorage::Pointer container,
           IndexMetric::Pointer /*measure*/) override;

  //! Unload index from storage
  int unload(void) override;

  int search_impl(const void * /*query*/, const IndexQueryMeta & /*qmeta*/,
                  Context::Pointer & /*context*/) const override {
    return IndexError_NotImplemented;
  }

  int search_impl(const void * /*query*/, const IndexQueryMeta & /*qmeta*/,
                  uint32_t /*count*/,
                  Context::Pointer & /*context*/) const override {
    return IndexError_NotImplemented;
  }

  int search_bf_impl(const void * /*query*/, const IndexQueryMeta & /*qmeta*/,
                     Context::Pointer & /*context*/) const override {
    return IndexError_NotImplemented;
  }

  int search_bf_impl(const void * /*query*/, const IndexQueryMeta & /*qmeta*/,
                     uint32_t /*count*/,
                     Context::Pointer & /*context*/) const override {
    return IndexError_NotImplemented;
  }

  //! Similarity search with sparse inputs
  int search_impl(const uint32_t sparse_count, const uint32_t *sparse_indices,
                  const void *sparse_query, const IndexQueryMeta &qmeta,
                  Context::Pointer &context) const override {
    return search_impl(&sparse_count, sparse_indices, sparse_query, qmeta, 1,
                       context);
  }

  //! Similarity search with sparse inputs
  int search_impl(const uint32_t *sparse_count, const uint32_t *sparse_indices,
                  const void *sparse_query, const IndexQueryMeta &qmeta,
                  uint32_t count, Context::Pointer &context) const override {
    return search_bf_impl(sparse_count, sparse_indices, sparse_query, qmeta,
                          count, context);
  }

  //! Similarity brute force search with sparse inputs
  int search_bf_impl(const uint32_t sparse_count,
                     const uint32_t *sparse_indices, const void *sparse_query,
                     const IndexQueryMeta &qmeta,
                     Context::Pointer &context) const override {
    return search_bf_impl(&sparse_count, sparse_indices, sparse_query, qmeta, 1,
                          context);
  }

  //! Similarity brute force search with sparse inputs
  int search_bf_impl(const uint32_t *sparse_count,
                     const uint32_t *sparse_indices, const void *sparse_query,
                     const IndexQueryMeta &qmeta, uint32_t count,
                     Context::Pointer &context) const override;

  //! Linear search by primary keys
  int search_bf_by_p_keys_impl(const uint32_t sparse_count,
                               const uint32_t *sparse_indices,
                               const void *sparse_query,
                               const std::vector<std::vector<uint64_t>> &p_keys,
                               const IndexQueryMeta &qmeta,
                               ContextPointer &context) const override {
    return search_bf_by_p_keys_impl(&sparse_count, sparse_indices, sparse_query,
                                    p_keys, qmeta, 1, context);
  }

  //! Linear search by primary keys
  int search_bf_by_p_keys_impl(const uint32_t *sparse_count,
                               const uint32_t *sparse_indices,
                               const void *sparse_query,
                               const std::vector<std::vector<uint64_t>> &p_keys,
                               const IndexQueryMeta &qmeta, uint32_t count,
                               ContextPointer &context) const override;

  //! Fetch sparser vector by key
  int get_sparse_vector(uint64_t key, uint32_t *sparse_count,
                        std::string *sparse_indices_buffer,
                        std::string *sparse_values_buffer) const override;

  //! Create a searcher context
  ContextPointer create_context() const override;

  //! Create a new iterator
  IndexSearcher::SparseProvider::Pointer create_sparse_provider(
      void) const override;

  //! Retrieve statistics
  const Stats &stats(void) const override {
    return stats_;
  }

  //! Retrieve meta of index
  const IndexMeta &meta(void) const override {
    return meta_;
  }

  //! Retrieve params of index
  const ailego::Params &params(void) const override {
    return params_;
  }

  const FlatSparseSearcherEntity &entity(void) const {
    return entity_;
  }

  uint32_t magic(void) const {
    return magic_;
  }

 private:
  inline int check_params(const IndexQueryMeta &qmeta) const {
    if (ailego_unlikely(qmeta.data_type() != meta_.data_type())) {
      LOG_ERROR("Unsupported query meta");
      return IndexError_Mismatch;
    }
    return 0;
  }

  int do_search(const uint32_t *sparse_count, const uint32_t *sparse_indices,
                const void *sparse_query, bool with_p_keys,
                const std::vector<std::vector<uint64_t>> &p_keys,
                const IndexQueryMeta &qmeta, uint32_t count,
                ContextPointer &context) const;

 private:
  enum State { STATE_INIT = 0, STATE_INITED = 1, STATE_LOADED = 2 };

  FlatSparseSearcherEntity entity_{};
  IndexMeta meta_{};
  ailego::Params params_{};
  uint32_t magic_{0U};

  Stats stats_;
  State state_{STATE_INIT};
};

}  // namespace core
}  // namespace zvec
