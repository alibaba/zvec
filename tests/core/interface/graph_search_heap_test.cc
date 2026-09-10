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
#include <tuple>
#include <vector>
#include <gtest/gtest.h>
#include "algorithm/hnsw/hnsw_context.h"
#include "algorithm/hnsw/hnsw_streamer_entity.h"
#include "algorithm/vamana/vamana_streamer.h"
#include "tests/test_util.h"
#include "zvec/core/interface/index_factory.h"
#include "zvec/core/interface/index_param_builders.h"

namespace zvec::core_interface {
namespace {

BaseIndexParam::Pointer GraphIndexParam(bool vamana, bool contiguous) {
  if (vamana) {
    return VamanaIndexParamBuilder()
        .with_metric_type(MetricType::kL2sq)
        .with_data_type(DataType::DT_FP32)
        .with_dimension(16)
        .with_max_degree(16)
        .with_search_list_size(64)
        .with_max_occlusion_size(64)
        .with_two_pass_build(true)
        .with_use_contiguous_memory(contiguous)
        .build();
  }
  return HNSWIndexParamBuilder()
      .with_metric_type(MetricType::kL2sq)
      .with_data_type(DataType::DT_FP32)
      .with_dimension(16)
      .with_m(16)
      .with_ef_construction(64)
      .with_use_contiguous_memory(contiguous)
      .build();
}

using GraphSearchHeapTest = testing::TestWithParam<
    std::tuple<bool, bool, bool, core::VisitFilter::Mode>>;

TEST_P(GraphSearchHeapTest, ReuseAcrossGraphFilteredAndBruteForceSearch) {
  const bool vamana = std::get<0>(GetParam());
  const bool contiguous = std::get<1>(GetParam());
  const bool ties = std::get<2>(GetParam());
  const auto visit_mode = std::get<3>(GetParam());
  constexpr uint32_t kCount = 64;
  const std::string path = "graph_search_heap.index";
  test_util::RemoveTestFiles(path);
  auto index =
      IndexFactory::CreateAndInitIndex(*GraphIndexParam(vamana, contiguous));
  ASSERT_TRUE(index);
  ASSERT_EQ(0, index->open(path, {StorageOptions::StorageType::kMMAP, true}));
  std::vector<float> vector(16, 0.0f);
  for (uint32_t id = 0; id < kCount; ++id) {
    vector[0] = float(ties ? id % 8 : id);
    ASSERT_EQ(0, index->add(VectorData{DenseVector{vector.data()}}, id));
  }
  if (vamana) {
    auto *streamer =
        dynamic_cast<core::VamanaStreamer *>(index->index_searcher().get());
    ASSERT_NE(nullptr, streamer);
    ASSERT_EQ(0, streamer->finalize_build());
  }
  ASSERT_EQ(0, index->flush());
  ASSERT_EQ(0, index->close());
  index =
      IndexFactory::CreateAndInitIndex(*GraphIndexParam(vamana, contiguous));
  ASSERT_TRUE(index);
  ASSERT_EQ(0, index->open(path, {StorageOptions::StorageType::kMMAP, false}));
  auto streamer = index->index_searcher();
  auto context = streamer->create_context();
  ASSERT_TRUE(context);
  auto *vctx = dynamic_cast<core::VamanaContext *>(context.get());
  auto *hctx = dynamic_cast<core::HnswContext *>(context.get());
  ASSERT_TRUE(vamana ? vctx != nullptr : hctx != nullptr);
  if (contiguous) {
    if (vctx) {
      const auto *entity =
          dynamic_cast<const core::VamanaContiguousStreamerEntity *>(
              &vctx->get_entity());
      ASSERT_NE(nullptr, entity);
      ASSERT_TRUE(entity->is_contiguous());
    } else {
      const auto *entity =
          dynamic_cast<const core::HnswContiguousStreamerEntity *>(
              &hctx->get_entity());
      ASSERT_NE(nullptr, entity);
      ASSERT_TRUE(entity->is_contiguous());
    }
  }
  vector[0] = 0.25f;
  const core::IndexQueryMeta meta(core::IndexMeta::DT_FP32, 16);
  const std::vector<std::vector<uint64_t>> p_keys{{0, 7, 17, 31, 63}};
  auto configure = [&](core::IndexContext::Pointer &ctx, uint32_t topk,
                       int mode, bool fetch_vector) {
    core::VisitFilter *visit = nullptr;
    if (auto *v = dynamic_cast<core::VamanaContext *>(ctx.get())) {
      v->set_ef(kCount);
      v->set_force_padding_topk(mode == 3);
      v->set_filter_mode(visit_mode);
      visit = &v->visit_filter();
    } else {
      auto *h = dynamic_cast<core::HnswContext *>(ctx.get());
      ASSERT_NE(nullptr, h);
      h->set_ef(kCount);
      h->set_force_padding_topk(mode == 3);
      h->set_filter_mode(visit_mode);
      visit = &h->visit_filter();
    }
    if (visit->get_mode() != visit_mode) {
      visit->destroy();
      ASSERT_EQ(0, visit->init(visit_mode, kCount, kCount, 0.001f));
    }
    ctx->set_topk(topk);
    ctx->set_fetch_vector(fetch_vector);
    ctx->set_bruteforce_threshold(mode == 4 ? kCount : 0);
    ctx->reset_filter();
    ctx->reset_threshold();
    if (mode == 1 || mode == 3) {
      ctx->set_filter([](uint64_t key) { return key >= 32; });
    }
    if (mode == 5) ctx->set_filter([](uint64_t) { return true; });
    if (mode == 2) ctx->set_threshold(16.0f);
  };
  auto compare = [](const core::IndexDocumentList &expected,
                    const core::IndexDocumentList &actual) {
    ASSERT_EQ(expected.size(), actual.size());
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_EQ(expected[i].key(), actual[i].key());
      EXPECT_FLOAT_EQ(expected[i].score(), actual[i].score());
    }
  };

  for (uint32_t topk : {0U, 1U, 12U, 32U, 80U}) {
    // End by returning to the pool after empty results and BF-by-keys.
    for (int mode : {0, 1, 2, 3, 4, 5, 6, 0}) {
      for (bool fetch_vector : {false, true}) {
        SCOPED_TRACE(topk);
        SCOPED_TRACE(mode);
        SCOPED_TRACE(fetch_vector);
        auto fresh = streamer->create_context();
        ASSERT_TRUE(fresh);
        configure(fresh, topk, mode, fetch_vector);
        configure(context, topk, mode, fetch_vector);
        auto search = [&](core::IndexContext::Pointer &ctx) {
          if (mode == 6) {
            return streamer->search_bf_by_p_keys_impl(vector.data(), p_keys,
                                                      meta, 1, ctx);
          }
          return streamer->search_impl(vector.data(), meta, 1, ctx);
        };
        // Bloom hashes are randomized per context. Compare repeated searches
        // on the same storage so false positives do not make the test flaky.
        auto &reference =
            visit_mode == core::VisitFilter::BloomFilter ? context : fresh;
        ASSERT_EQ(0, search(reference));
        const auto expected = reference->result();
        const auto expected_scans =
            vamana ? static_cast<core::VamanaContext *>(reference.get())
                         ->get_scan_num()
                   : static_cast<core::HnswContext *>(reference.get())
                         ->get_scan_num();
        ASSERT_EQ(0, search(context));
        compare(expected, context->result());
        EXPECT_EQ(expected_scans,
                  vctx ? vctx->get_scan_num() : hctx->get_scan_num());
        auto &heap = vctx ? vctx->search_heap() : hctx->search_heap();
        heap.dispatch([&](const auto &buffer) {
          const bool uses_topk =
              std::is_same_v<std::decay_t<decltype(buffer)>, core::TopkHeap>;
          EXPECT_EQ(mode != 0 && mode != 2, uses_topk);
        });
      }
    }
  }

  // A later query must not change an earlier query's materialized documents.
  std::vector<float> batch(32, 0.0f);
  batch[0] = 0.25f;
  batch[16] = 63.25f;
  for (int mode : {0, 4, 0}) {
    configure(context, 12, mode, false);
    std::vector<core::IndexDocumentList> expected(2);
    for (size_t q = 0; q < 2; ++q) {
      ASSERT_EQ(0,
                streamer->search_impl(batch.data() + q * 16, meta, 1, context));
      expected[q] = context->result();
    }
    ASSERT_EQ(0, streamer->search_impl(batch.data(), meta, 2, context));
    for (size_t q = 0; q < 2; ++q) compare(expected[q], context->result(q));
  }

  // Invalid dispatch must fail both pool and filtered search, without leaving
  // results from the previous query. Restore valid storage before teardown.
  configure(context, 12, 0, false);
  auto &visit = vctx ? vctx->visit_filter() : hctx->visit_filter();
  visit.destroy();
  ASSERT_EQ(0, visit.init(core::VisitFilter::Default, kCount, kCount, 0.001f));
  for (bool filtered : {false, true}) {
    if (filtered) context->set_filter([](uint64_t) { return false; });
    EXPECT_EQ(core::IndexError_Runtime,
              streamer->search_impl(vector.data(), meta, 1, context));
    EXPECT_TRUE(context->result().empty());
    auto &heap = vctx ? vctx->search_heap() : hctx->search_heap();
    heap.dispatch([](const auto &buffer) { EXPECT_EQ(0U, buffer.size()); });
  }
  ASSERT_EQ(0, visit.init(visit_mode, kCount, kCount, 0.001f));
  context.reset();
  streamer.reset();
  ASSERT_EQ(0, index->close());
  test_util::RemoveTestFiles(path);
}

INSTANTIATE_TEST_SUITE_P(
    Backends, GraphSearchHeapTest,
    testing::Combine(testing::Bool(), testing::Bool(), testing::Bool(),
                     testing::Values(core::VisitFilter::BitMap,
                                     core::VisitFilter::ByteMap,
                                     core::VisitFilter::BloomFilter)));

}  // namespace
}  // namespace zvec::core_interface
