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

#include <zvec/core/framework/index_framework.h>
#include "../hnsw/hnsw_searcher.h"
#include <omega/omega_api.h>

namespace zvec {
namespace core {

// OmegaSearcher owns the loaded OMEGA runtime on the searcher side:
// model loading, mode selection, HNSW-hook wiring, and per-search context
// creation. It reuses the HNSW search loop instead of defining an independent
// graph-search implementation.
class OmegaSearcher : public HnswSearcher {
 public:
  using ContextPointer = IndexSearcher::Context::Pointer;

 public:
  OmegaSearcher(void);
  ~OmegaSearcher(void);

 OmegaSearcher(const OmegaSearcher &) = delete;
  OmegaSearcher &operator=(const OmegaSearcher &) = delete;

 protected:
  //! Initialize Searcher
  virtual int init(const ailego::Params &params) override;

  //! Cleanup Searcher
  virtual int cleanup(void) override;

  //! Load Index from storage
  virtual int load(IndexStorage::Pointer container,
                   IndexMetric::Pointer metric) override;

  //! Unload index from storage
  virtual int unload(void) override;

  //! KNN Search
  virtual int search_impl(const void *query, const IndexQueryMeta &qmeta,
                          ContextPointer &context) const override {
    return search_impl(query, qmeta, 1, context);
  }

  //! KNN Search with OMEGA adaptive search
  virtual int search_impl(const void *query, const IndexQueryMeta &qmeta,
                          uint32_t count,
                          ContextPointer &context) const override;

  //! Create a searcher context (creates OmegaContext instead of HnswContext)
  virtual ContextPointer create_context() const override;

 private:
  //! Check if OMEGA mode should be used
  bool should_use_omega() const;

  //! Adaptive search with OMEGA predictions
  int adaptive_search(const void *query, const IndexQueryMeta &qmeta,
                      uint32_t count, ContextPointer &context) const;

 private:
  // OMEGA components
  OmegaModelHandle omega_model_;
  bool omega_enabled_;
  bool use_omega_mode_;
  uint32_t min_vector_threshold_;
  size_t current_vector_count_;
  int window_size_;
};

}  // namespace core
}  // namespace zvec
