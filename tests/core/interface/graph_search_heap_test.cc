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
#include <cmath>
#include <tuple>
#include <vector>
#include <gtest/gtest.h>
#include "algorithm/hnsw/hnsw_algorithm.h"
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

TEST(GraphSearchHeap, HnswConstructionClearsStateAcrossLevels) {
  const std::string path = "hnsw_search_dispatch.index";
  const core::IndexMeta meta(core::IndexMeta::DT_FP32, 16);
  for (auto mode : {core::VisitFilter::BitMap, core::VisitFilter::ByteMap}) {
    for (bool filtered : {false, true}) {
      SCOPED_TRACE(mode);
      SCOPED_TRACE(filtered);
      test_util::RemoveTestFiles(path);
      {
        core::IndexStreamer::Stats stats;
        auto storage = core::IndexFactory::CreateStorage("MMapFileStorage");
        ASSERT_TRUE(storage);
        ASSERT_EQ(0, storage->init(ailego::Params()));
        ASSERT_EQ(0, storage->open(path, true));
        auto entity = std::make_shared<core::HnswMmapStreamerEntity>(stats);
        entity->set_vector_size(16 * sizeof(float));
        entity->set_l0_neighbor_cnt(8);
        entity->set_upper_neighbor_cnt(8);
        entity->set_prune_cnt(8);
        entity->set_scaling_factor(2);
        entity->set_ef_construction(8);
        ASSERT_EQ(0, entity->init(16));
        ASSERT_EQ(0, entity->open(storage, 0, false));

        // Seed two connected nodes at every level, then insert a closer node
        // with a new maximum level. No random level generation is involved.
        std::vector<float> vector(16, 0.0f);
        for (core::node_id_t id = 0; id < 3; ++id) {
          vector[0] = id == 2 ? 9.0f : 10.0f * id;
          core::node_id_t actual = core::kInvalidNodeId;
          ASSERT_EQ(0, entity->add_vector(id == 2 ? 3 : 2, id, vector.data(),
                                          &actual));
          ASSERT_EQ(id, actual);
        }
        for (core::level_t level = 0; level <= 2; ++level) {
          ASSERT_EQ(0, entity->update_neighbors(level, 0, {{1, 100.0f}}));
          ASSERT_EQ(0, entity->update_neighbors(level, 1, {{0, 100.0f}}));
        }
        entity->update_ep_and_level(0, 2);

        auto metric = core::IndexFactory::CreateMetric("SquaredEuclidean");
        ASSERT_TRUE(metric);
        ASSERT_EQ(0, metric->init(meta, ailego::Params()));
        core::HnswContext context(16, metric, entity);
        context.set_filter_mode(mode);
        ASSERT_EQ(0, context.init(core::HnswContext::kStreamerContext));
        context.set_max_scan_num(10000);
        core::HnswAlgorithm<core::HnswMmapStreamerEntity> algorithm(*entity);
        std::vector<uint64_t> filtered_keys;
        if (filtered) {
          context.set_filter([&](uint64_t key) {
            filtered_keys.push_back(key);
            return key == 0;
          });
        }

        // Invalid dispatch must release the new-max-level lock without
        // publishing graph changes; the same insertion can then be retried.
        auto &visit = context.visit_filter();
        visit.destroy();
        ASSERT_EQ(0, visit.init(core::VisitFilter::Default, 16, 16, 0.001f));
        context.reset_query(vector.data(), meta);
        EXPECT_EQ(core::IndexError_Runtime, algorithm.add_node(2, 3, &context));
        EXPECT_EQ(0U, entity->entry_point());
        EXPECT_EQ(2, entity->cur_max_level());
        for (core::level_t level = 0; level <= 3; ++level) {
          EXPECT_EQ(0U, entity->get_neighbors(level, 2).size());
        }

        ASSERT_EQ(0, visit.init(mode, 16, 16, 0.001f));
        context.reset_query(vector.data(), meta);
        ASSERT_EQ(0, algorithm.add_node(2, 3, &context));
        EXPECT_EQ(2U, entity->entry_point());
        EXPECT_EQ(3, entity->cur_max_level());
        for (core::level_t level = 0; level <= 2; ++level) {
          const auto neighbors = entity->get_neighbors(level, 2);
          ASSERT_EQ(filtered ? 1U : 2U, neighbors.size());
          EXPECT_EQ(1U, neighbors[0]);
          if (!filtered) EXPECT_EQ(0U, neighbors[1]);
          EXPECT_TRUE(context.level_topk(level).empty());
        }
        EXPECT_EQ(0U, entity->get_neighbors(3, 2).size());
        if (filtered) {
          // Entry point 0 is filtered but must still lead to node 1. Each
          // lower level restarts at 1 and revisits 0 with cleared visit state.
          EXPECT_EQ((std::vector<uint64_t>{0, 1, 1, 0, 1, 0}), filtered_keys);
        }
        ASSERT_EQ(0, entity->close());
        ASSERT_EQ(0, storage->close());
      }
      test_util::RemoveTestFiles(path);
    }
  }
}

using GraphSearchHeapTest = testing::TestWithParam<
    std::tuple<bool, bool, bool, core::VisitFilter::Mode>>;

TEST(GraphSearchHeap, RefineHonorsScaleFactorAndSearchModes) {
  constexpr uint32_t kCount = 64;
  constexpr uint32_t kTopk = 5;
  const std::string coarse_path = "graph_refine_coarse.index";
  const std::string fine_path = "graph_refine_fine.index";
  for (bool vamana : {false, true}) {
    for (bool contiguous : {false, true}) {
      SCOPED_TRACE(vamana);
      SCOPED_TRACE(contiguous);
      test_util::RemoveTestFiles(coarse_path);
      auto coarse = IndexFactory::CreateAndInitIndex(
          *GraphIndexParam(vamana, contiguous));
      ASSERT_TRUE(coarse);
      ASSERT_EQ(0, coarse->open(coarse_path,
                                {StorageOptions::StorageType::kMMAP, true}));
      std::vector<float> vector(16, 0.0f);
      for (uint32_t id = 0; id < kCount; ++id) {
        vector[0] = float(id);
        ASSERT_EQ(
            0, coarse->add(VectorData{DenseVector{vector.data()}}, 1000 + id));
      }
      if (vamana) {
        auto *streamer = dynamic_cast<core::VamanaStreamer *>(
            coarse->index_searcher().get());
        ASSERT_NE(nullptr, streamer);
        ASSERT_EQ(0, streamer->finalize_build());
      }
      ASSERT_EQ(0, coarse->flush());
      ASSERT_EQ(0, coarse->close());
      coarse = IndexFactory::CreateAndInitIndex(
          *GraphIndexParam(vamana, contiguous));
      ASSERT_EQ(0, coarse->open(coarse_path,
                                {StorageOptions::StorageType::kMMAP, false}));

      for (auto type : {DataType::DT_FP16, DataType::DT_UINT8}) {
        SCOPED_TRACE(static_cast<int>(type));
        test_util::RemoveTestFiles(fine_path);
        auto fine_param = FlatIndexParamBuilder()
                              .with_metric_type(MetricType::kL2sq)
                              .with_data_type(DataType::DT_FP32)
                              .with_storage_data_type(type)
                              .with_dimension(16)
                              .with_use_contiguous_memory(contiguous)
                              .build();
        auto fine = IndexFactory::CreateAndInitIndex(*fine_param);
        ASSERT_TRUE(fine);
        ASSERT_EQ(0, fine->open(fine_path,
                                {StorageOptions::StorageType::kMMAP, true}));
        for (uint32_t id = 0; id < kCount; ++id) {
          // Reverse the ranking in the fine index. Exporting the whole ef pool
          // instead of coarse topk now changes the final nearest neighbors.
          vector[0] = float(kCount - id);
          ASSERT_EQ(
              0, fine->add(VectorData{DenseVector{vector.data()}}, 1000 + id));
        }
        vector[0] = 0.0f;
        const VectorData query{DenseVector{vector.data()}};
        SearchResult actual;
        for (float scale : {0.0f, 0.6f, 1.5f, 4.0f, 20.0f, 1.0f}) {
          const uint32_t budget = static_cast<uint32_t>(
              std::floor(kTopk * (scale == 0.0f ? 1.0f : scale)));
          for (int mode : {0, 1, 2, 3, 4, 5, 0}) {
            SCOPED_TRACE(scale);
            SCOPED_TRACE(mode);
            auto make_param = [&](uint32_t topk) {
              BaseIndexQueryParam::Pointer param;
              if (vamana) {
                param = VamanaQueryParamBuilder()
                            .with_topk(topk)
                            .with_ef_search(32)
                            .build();
              } else {
                param = HNSWQueryParamBuilder()
                            .with_topk(topk)
                            .with_ef_search(32)
                            .build();
              }
              if (mode == 1) param->is_linear = true;
              if (mode == 2) {
                param->bf_pks = std::make_shared<std::vector<uint64_t>>(
                    std::initializer_list<uint64_t>{1047, 1017, 1007, 1001,
                                                    1000, 99999});
              }
              if (mode == 3)
                param->bf_pks = std::make_shared<std::vector<uint64_t>>();
              if (mode == 4) {
                param->filter = std::make_shared<IndexFilter>();
                param->filter->set([](uint64_t key) { return key >= 1016; });
              }
              if (mode == 5) param->radius = 9.0f;
              return param;
            };
            SearchResult candidates;
            ASSERT_EQ(0,
                      coarse->search(query, make_param(budget), &candidates));
            EXPECT_LE(candidates.doc_list_.size(), budget);
            if (mode == 0 || mode == 1) {
              ASSERT_EQ(std::min(budget, kCount), candidates.doc_list_.size());
            }
            auto explicit_param = FlatQueryParamBuilder()
                                      .with_topk(kTopk)
                                      .with_fetch_vector(true)
                                      .build();
            explicit_param->bf_pks = std::make_shared<std::vector<uint64_t>>();
            for (const auto &doc : candidates.doc_list_) {
              explicit_param->bf_pks->push_back(doc.key());
            }
            SearchResult expected;
            ASSERT_EQ(0, fine->search(query, explicit_param, &expected));
            auto refiner = std::make_shared<RefinerParam>();
            refiner->scale_factor_ = scale;
            refiner->reference_index = fine;
            auto param = make_param(kTopk);
            param->fetch_vector = true;
            param->refiner_param = refiner;
            ASSERT_EQ(0, coarse->search(query, param, &actual));
            ASSERT_EQ(expected.doc_list_.size(), actual.doc_list_.size());
            for (size_t i = 0; i < actual.doc_list_.size(); ++i) {
              EXPECT_EQ(expected.doc_list_[i].key(), actual.doc_list_[i].key());
              EXPECT_FLOAT_EQ(expected.doc_list_[i].score(),
                              actual.doc_list_[i].score());
            }
            EXPECT_EQ(expected.reverted_vector_list_,
                      actual.reverted_vector_list_);
          }
        }
        ASSERT_EQ(0, fine->close());
        test_util::RemoveTestFiles(fine_path);
      }
      ASSERT_EQ(0, coarse->close());
      test_util::RemoveTestFiles(coarse_path);
    }
  }
}

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
    for (int mode : {0, 1, 2, 3, 4, 5, 6, 7, 0}) {
      for (bool fetch_vector : {false, true}) {
        SCOPED_TRACE(topk);
        SCOPED_TRACE(mode);
        SCOPED_TRACE(fetch_vector);
        auto fresh = streamer->create_context();
        ASSERT_TRUE(fresh);
        configure(fresh, topk, mode, fetch_vector);
        configure(context, topk, mode, fetch_vector);
        auto search = [&](core::IndexContext::Pointer &ctx) {
          if (mode == 7) {
            return streamer->search_bf_impl(vector.data(), meta, 1, ctx);
          }
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

        std::vector<uint64_t> keys{99999};
        int ret;
        if (mode == 6) {
          ret = streamer->search_candidates_by_p_keys_impl(
              vector.data(), p_keys, meta, keys, context);
        } else if (mode == 7) {
          ret = streamer->search_bf_candidates_impl(vector.data(), meta, keys,
                                                    context);
        } else {
          ret = streamer->search_candidates_impl(vector.data(), meta, keys,
                                                 context);
        }
        ASSERT_EQ(0, ret);
        ASSERT_EQ(expected.size(), keys.size());
        for (size_t i = 0; i < keys.size(); ++i) {
          EXPECT_EQ(expected[i].key(), keys[i]);
        }
        // No document or vector result is materialized, even when requested
        // on the same reused context immediately before candidate-only search.
        EXPECT_TRUE(context->result().empty());
        EXPECT_EQ(expected_scans,
                  vctx ? vctx->get_scan_num() : hctx->get_scan_num());
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
  for (int mode : {0, 4, 7, 0}) {
    configure(context, 12, mode, false);
    auto search_batch = [&](const float *queries, uint32_t count) {
      if (mode == 7)
        return streamer->search_bf_impl(queries, meta, count, context);
      return streamer->search_impl(queries, meta, count, context);
    };
    std::vector<core::IndexDocumentList> expected(2);
    for (size_t q = 0; q < 2; ++q) {
      ASSERT_EQ(0, search_batch(batch.data() + q * 16, 1));
      expected[q] = context->result();
    }
    ASSERT_EQ(0, search_batch(batch.data(), 2));
    for (size_t q = 0; q < 2; ++q) compare(expected[q], context->result(q));
  }

  if (hctx && !ties && visit_mode != core::VisitFilter::BloomFilter) {
    // Exercise real padding without depending on graph connectivity: seed one
    // candidate and let the context add every other unvisited node in place.
    for (bool dual_heap : {false, true}) {
      auto reset_for_padding = [&]() {
        configure(context, kCount, 0, false);
        hctx->clear();
        hctx->set_force_padding_topk(true);
        hctx->reset_query(vector.data(), streamer->meta());
        hctx->visit_filter().clear();
        hctx->visit_filter().set_visited(0);
        auto &heap = hctx->search_heap();
        if (dual_heap)
          heap.reset<core::TopkHeap>(kCount);
        else
          heap.reset_pool(kCount, 16);
        heap.dispatch([&](auto &buffer) {
          core::SearchHeap::emplace(buffer, 0, 0.0625f);
        });
      };
      reset_for_padding();
      hctx->topk_to_result();
      const auto padded = context->result();
      ASSERT_EQ(kCount, padded.size());
      reset_for_padding();
      std::vector<uint64_t> keys;
      hctx->topk_to_keys(keys);
      ASSERT_EQ(padded.size(), keys.size());
      for (size_t i = 0; i < keys.size(); ++i) {
        EXPECT_EQ(padded[i].key(), keys[i]);
      }
      hctx->search_heap().dispatch([&](const auto &buffer) {
        EXPECT_EQ(
            dual_heap,
            (std::is_same_v<std::decay_t<decltype(buffer)>, core::TopkHeap>));
      });
    }
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
    std::vector<uint64_t> keys{99999};
    EXPECT_EQ(
        core::IndexError_Runtime,
        streamer->search_candidates_impl(vector.data(), meta, keys, context));
    EXPECT_TRUE(keys.empty());
  }
  ASSERT_EQ(0, visit.init(visit_mode, kCount, kCount, 0.001f));
  if (hctx) {
    hctx->set_group_params(2, 2);
    hctx->set_group_by([](uint64_t key) { return std::to_string(key % 2); });
    std::vector<uint64_t> keys{99999};
    EXPECT_EQ(
        core::IndexError_InvalidArgument,
        streamer->search_candidates_impl(vector.data(), meta, keys, context));
    EXPECT_TRUE(keys.empty());
    keys.push_back(99999);
    EXPECT_EQ(core::IndexError_InvalidArgument,
              streamer->search_bf_candidates_impl(vector.data(), meta, keys,
                                                  context));
    EXPECT_TRUE(keys.empty());
    keys.push_back(99999);
    EXPECT_EQ(core::IndexError_InvalidArgument,
              streamer->search_candidates_by_p_keys_impl(vector.data(), p_keys,
                                                         meta, keys, context));
    EXPECT_TRUE(keys.empty());

    // Both explicit and automatic BF must export each query's grouped heaps
    // before the next query clears them.
    for (int mode : {4, 7}) {
      configure(context, 4, mode, false);
      auto search_batch = [&](const float *queries, uint32_t count) {
        if (mode == 7)
          return streamer->search_bf_impl(queries, meta, count, context);
        return streamer->search_impl(queries, meta, count, context);
      };
      std::vector<core::IndexGroupDocumentList> expected(2);
      for (size_t q = 0; q < 2; ++q) {
        ASSERT_EQ(0, search_batch(batch.data() + q * 16, 1));
        expected[q] = context->group_result();
        ASSERT_EQ(2U, expected[q].size());
      }
      ASSERT_EQ(0, search_batch(batch.data(), 2));
      for (size_t q = 0; q < 2; ++q) {
        const auto &actual = context->group_result(q);
        ASSERT_EQ(expected[q].size(), actual.size());
        for (size_t group = 0; group < actual.size(); ++group) {
          EXPECT_EQ(expected[q][group].group_id(), actual[group].group_id());
          compare(expected[q][group].docs(), actual[group].docs());
        }
      }
    }
  }
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
