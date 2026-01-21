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

#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_refiner.h>

namespace zvec {
namespace core {

/*! Basic Refiner
 */
class BasicRefiner : public IndexRefiner {
 public:
  const uint32_t kScaleFactor = 10;

 public:
  class BasicRefinerContext : public Context {
   public:
    //! Construct
    BasicRefinerContext() = default;
    ~BasicRefinerContext() = default;

    int set_contexts(IndexRunner::Context::Pointer base_ctx,
                     IndexRunner::Context::Pointer refine_ctx) override {
      base_ctx_ = std::move(base_ctx);
      refine_ctx_ = std::move(refine_ctx);

      return 0;
    }

    //! Set topk of search result
    void set_topk(uint32_t topk) override {
      topk_ = topk;
    }

    uint32_t topk() const override {
      return topk_;
    }

    //! Retrieve search result with index
    const IndexDocumentList &result(void) const override {
      return results_[0];
    }

    //! Retrieve search result with index
    const IndexDocumentList &result(size_t idx) const override {
      return results_[idx];
    }

    //! Retrieve mutable result with index
    IndexDocumentList *mutable_result(size_t idx) override {
      ailego_assert_with(idx < results_.size(), "invalid idx");
      return &results_[idx];
    }

    void resize_results(size_t size) {
      results_.resize(size);
    }

    IndexRunner::Context::Pointer &base_context() {
      return base_ctx_;
    }

    IndexRunner::Context::Pointer &refine_context() {
      return refine_ctx_;
    }

   private:
    uint32_t topk_{0};
    std::vector<IndexDocumentList> results_{};
    std::vector<IndexGroupDocumentList> group_results_{};

    IndexRunner::Context::Pointer base_ctx_{nullptr};
    IndexRunner::Context::Pointer refine_ctx_{nullptr};
  };

 public:
  //! Create a context
  Context::Pointer create_context(void) const override {
    auto base_ctx = base_runner_->create_context();
    auto refine_ctx = refine_runner_->create_context();

    BasicRefinerContext *ctx = new (std::nothrow) BasicRefinerContext();

    ctx->set_contexts(std::move(base_ctx), std::move(refine_ctx));

    return Context::Pointer(ctx);
  }

  //! Initialize refiner with streamer
  int init(IndexRunner::Pointer base_runner, IndexRunner::Pointer refine_runner,
           const ailego::Params &params) override {
    base_runner_ = base_runner;
    refine_runner_ = refine_runner;

    params_ = params;

    return 0;
  }

  //! Cleanup
  int cleanup() override {
    return 0;
  }

  //! Add a vector into index
  virtual int add_impl(uint64_t key, const void *base_query,
                       const IndexQueryMeta &base_qmeta,
                       const void *refine_query,
                       const IndexQueryMeta &refine_qmeta,
                       Context::Pointer &context) override {
    BasicRefinerContext *ctx =
        dynamic_cast<BasicRefinerContext *>(context.get());

    int ret = base_runner_->add_impl(key, base_query, base_qmeta,
                                     ctx->base_context());
    if (ret != 0) {
      LOG_ERROR("Error in adding vector to base index");

      return ret;
    }

    ret = refine_runner_->add_impl(key, refine_query, refine_qmeta,
                                   ctx->refine_context());
    if (ret != 0) {
      LOG_ERROR("Error in adding vector to refine index");

      return ret;
    }

    return 0;
  }

  //! Similarity search
  virtual int search_impl(const void *base_query,
                          const IndexQueryMeta &base_qmeta,
                          const void *refine_query,
                          const IndexQueryMeta &refine_qmeta, uint32_t count,
                          Context::Pointer &context) const override {
    BasicRefinerContext *ctx =
        dynamic_cast<BasicRefinerContext *>(context.get());

    uint32_t topk = ctx->topk();

    ctx->resize_results(count);

    int ret;
    for (size_t q = 0; q < count; ++q) {
      auto &base_ctx = ctx->base_context();
      auto &refine_ctx = ctx->refine_context();

      base_ctx->set_topk(topk * scale_factor_);
      ret = base_runner_->search_impl(base_query, base_qmeta, base_ctx);
      if (ret != 0) {
        LOG_ERROR("Error in searching vector from base index");

        return ret;
      }

      auto base_result = base_ctx->result();

      std::vector<uint64_t> keys;
      for (size_t i = 0; i < base_result.size(); ++i) {
        keys.push_back(base_result[i].key());
      }

      std::vector<std::vector<uint64_t>> keys_array;
      keys_array.push_back(std::move(keys));

      refine_ctx->set_topk(topk);
      ret = refine_runner_->search_bf_by_p_keys_impl(refine_query, keys_array,
                                                     refine_qmeta, refine_ctx);
      if (ret != 0) {
        LOG_ERROR("Error in searching vector from refine index");

        return ret;
      }

      auto refine_result = refine_ctx->result();
      *ctx->mutable_result(q) = refine_result;

      base_query =
          static_cast<const char *>(base_query) + base_qmeta.element_size();
      refine_query =
          static_cast<const char *>(refine_query) + refine_qmeta.element_size();
    }

    return 0;
  }

  //! Similarity search
  virtual int search_impl(const void *base_query,
                          const IndexQueryMeta &base_qmeta,
                          const void *refine_query,
                          const IndexQueryMeta &refine_qmeta,
                          Context::Pointer &context) const override {
    return search_impl(base_query, base_qmeta, refine_query, refine_qmeta, 1,
                       context);
  }

  //! Similarity brute force search
  virtual int search_bf_impl(const void *base_query,
                             const IndexQueryMeta &base_qmeta,
                             const void *refine_query,
                             const IndexQueryMeta &refine_qmeta, uint32_t count,
                             Context::Pointer &context) const override {
    BasicRefinerContext *ctx =
        dynamic_cast<BasicRefinerContext *>(context.get());

    for (size_t q = 0; q < count; ++q) {
      int ret;

      auto &base_ctx = ctx->base_context();
      auto &refine_ctx = ctx->refine_context();

      ret = base_runner_->search_impl(base_query, base_qmeta, base_ctx);
      if (ret != 0) {
        LOG_ERROR("Error in searching vector from base index");

        return ret;
      }

      auto results = base_ctx->result();
      std::vector<std::vector<uint64_t>> keys;

      ret = refine_runner_->search_bf_by_p_keys_impl(refine_query, keys,
                                                     refine_qmeta, refine_ctx);
      if (ret != 0) {
        LOG_ERROR("Error in searching vector from refine index");

        return ret;
      }
      auto refine_result = refine_ctx->result();
      *ctx->mutable_result(q) = refine_result;

      base_query =
          static_cast<const char *>(base_query) + base_qmeta.element_size();
      refine_query =
          static_cast<const char *>(refine_query) + refine_qmeta.element_size();
    }

    return 0;
  }

  //! Similarity brute force search
  virtual int search_bf_impl(const void *base_query,
                             const IndexQueryMeta &base_qmeta,
                             const void *refine_query,
                             const IndexQueryMeta &refine_qmeta,
                             Context::Pointer &context) const override {
    return search_bf_impl(base_query, base_qmeta, refine_query, refine_qmeta, 1,
                          context);
  }

 private:
  uint32_t scale_factor_{kScaleFactor};
  ailego::Params params_;

  IndexRunner::Pointer base_runner_{nullptr};
  IndexRunner::Pointer refine_runner_{nullptr};
};

INDEX_FACTORY_REGISTER_REFINER(BasicRefiner);

}  // namespace core
}  // namespace zvec
