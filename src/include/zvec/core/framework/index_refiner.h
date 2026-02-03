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
#include <zvec/ailego/container/heap.h>
#include <zvec/core/framework/index_context.h>
#include <zvec/core/framework/index_helper.h>
#include <zvec/core/framework/index_searcher.h>
#include <zvec/core/framework/index_streamer.h>

namespace zvec {
namespace core {

/*! Index Refiner
 */
class IndexRefiner : public IndexModule {
 public:
  //! Index Refiner Pointer
  typedef std::shared_ptr<IndexRefiner> Pointer;

  /*! Index Searcher Context
   */
  struct Context : public IndexContext {
   public:
    Context() = default;
    ~Context() = default;

    virtual int set_contexts(IndexRunner::Context::Pointer base_ctx,
                             IndexRunner::Context::Pointer refine_ctx) = 0;
  };

  //! Initialize refiner with streamer
  virtual int init(IndexRunner::Pointer base_runner,
                   IndexRunner::Pointer refine_runner,
                   const ailego::Params &params) = 0;

  //! Cleanup
  virtual int cleanup() = 0;

  //! Create a context
  virtual Context::Pointer create_context(void) const = 0;

  //! Add a vector into index
  virtual int add_impl(uint64_t key, const void *base_query,
                       const IndexQueryMeta &base_qmeta,
                       const void *refine_query,
                       const IndexQueryMeta &refine_qmeta,
                       Context::Pointer &context) = 0;

  //! Similarity search
  virtual int search_impl(const void *base_query,
                          const IndexQueryMeta &base_qmeta,
                          const void *refine_query,
                          const IndexQueryMeta &refine_qmeta,
                          Context::Pointer &context) const = 0;
  //! Similarity search
  virtual int search_impl(const void *base_query,
                          const IndexQueryMeta &base_qmeta,
                          const void *refine_query,
                          const IndexQueryMeta &refine_qmeta, uint32_t count,
                          Context::Pointer &context) const = 0;

  //! Similarity brute force search
  virtual int search_bf_impl(const void *base_query,
                             const IndexQueryMeta &base_qmeta,
                             const void *refine_query,
                             const IndexQueryMeta &refine_qmeta,
                             Context::Pointer &context) const = 0;

  //! Similarity brute force search
  virtual int search_bf_impl(const void *base_query,
                             const IndexQueryMeta &base_qmeta,
                             const void *refine_query,
                             const IndexQueryMeta &refine_qmeta, uint32_t count,
                             Context::Pointer &context) const = 0;
};

}  // namespace core
}  // namespace zvec
