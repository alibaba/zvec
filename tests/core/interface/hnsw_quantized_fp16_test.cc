// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <turbo/quantizer/quantizer.h>
#include <zvec/ailego/utility/float_helper.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/interface/index.h>
#include <zvec/core/interface/index_param.h>
#include "algorithm/hnsw/hnsw_context.h"
#include "algorithm/hnsw/hnsw_quantized_graph.h"
#include "algorithm/hnsw/hnsw_streamer.h"
#include "tests/test_util.h"

using namespace zvec::core_interface;

class BudgetHnswIndex : public HNSWIndex {
 public:
  using HNSWIndex::init;
  uint32_t expanded = 0;
  size_t comparisons = 0;
  uint32_t exact_scan_budget = 0;
  bool force_graph_search = true;
  zvec::core::IndexStreamer *native_streamer() {
    return streamer_.get();
  }
  int train_router(size_t anchors, size_t projected) {
    auto *streamer = dynamic_cast<zvec::core::HnswStreamer *>(streamer_.get());
    return streamer ? streamer->train_quantized_graph_router(anchors, projected,
                                                             202610261259ULL)
                    : zvec::core::IndexError_Unsupported;
  }

 protected:
  int _dense_search(const VectorData &query,
                    const BaseIndexQueryParam::Pointer &param,
                    SearchResult *result,
                    zvec::core::IndexContext::Pointer &context) override {
    context->set_debug_mode(true);
    // Small fixtures otherwise take the streamer's brute-force shortcut.
    context->set_bruteforce_threshold(force_graph_search ? 0 : 1000);
    auto *ctx = dynamic_cast<zvec::core::HnswContext *>(context.get());
    if (ctx && exact_scan_budget) ctx->set_max_scan_num(exact_scan_budget);
    const int ret = HNSWIndex::_dense_search(query, param, result, context);
    if (ctx) {
      expanded = *ctx->mutable_stats_get_neighbors();
      comparisons = ctx->get_scan_num();
    }
    context->set_debug_mode(false);
    return ret;
  }
};

class QuantizedGraphBudgetTest : public testing::TestWithParam<MetricType> {
 protected:
  virtual QuantizerParam::Pointer quantizer() const {
    return nullptr;
  }
  void SetUp() override {
    if (!zvec::core::HnswQuantizedGraph::supported())
      GTEST_SKIP() << "Quantized graph requires AVX512/VNNI";
    zvec::test_util::RemoveTestFiles(path_ + "*");
    HNSWIndexParam param(GetParam(), kDimension, 16, 100);
    param.data_type = DataType::DT_FP32;
    param.use_id_map = true;
    param.quantizer_param = quantizer();
    BudgetHnswIndex builder;
    ASSERT_EQ(0, builder.init(param));
    ASSERT_EQ(0, builder.open(path_, {StorageOptions::StorageType::kMMAP,
                                      /*create_new=*/true}));
    std::mt19937 rng(17);
    std::uniform_real_distribution<float> random(0, 1);
    std::array<float, kDimension> vector;
    for (uint32_t id = 0; id < 128; ++id) {
      for (float &v : vector) v = random(rng);
      if (id == 0) query_ = vector;
      vectors_.push_back(vector);
      ASSERT_EQ(0, builder.add({DenseVector{vector.data()}}, id));
    }
    ASSERT_EQ(0, builder.flush());
    ASSERT_EQ(0, builder.close());
    param.use_quantized_graph = true;
    param.quantizer_param = quantizer();
    ASSERT_EQ(0, index_.init(param));
    StorageOptions storage;
    storage.type = StorageOptions::StorageType::kMMAP;
    storage.read_only = true;
    ASSERT_EQ(0, index_.open(path_, storage));
  }
  void TearDown() override {
    index_.close();
    zvec::test_util::RemoveTestFiles(path_ + "*");
  }
  void search(uint32_t cap, uint32_t topk,
              std::function<bool(uint64_t)> excluded) {
    auto param = std::make_shared<HNSWQueryParam>();
    param->topk = topk;
    param->ef_search = 800;
    param->quantized_graph_max_expansions = cap;
    // Disable distance-based early stopping to exercise the expansion cap.
    param->quantized_graph_margin = 1e6f;
    param->filter = std::make_shared<IndexFilter>();
    param->filter->set(std::move(excluded));
    result_.doc_list_.clear();
    ASSERT_EQ(0, index_.search({DenseVector{query_.data()}}, param, &result_));
    if (index_.exact_scan_budget) {
      // L2 refines during traversal, so the distance budget can stop it
      // before the expansion cap. Cosine refines after traversal.
      EXPECT_LE(index_.expanded, cap);
    } else {
      EXPECT_EQ(cap, index_.expanded);
    }
  }
  static constexpr size_t kDimension = 64;
  const std::string path_ = "test_quantized_graph_budget.index";
  std::array<float, kDimension> query_{};
  std::vector<std::array<float, kDimension>> vectors_;
  BudgetHnswIndex index_;
  SearchResult result_;
};

class QuantizedGraphFp16Test : public QuantizedGraphBudgetTest {
 protected:
  QuantizerParam::Pointer quantizer() const override {
    return std::make_shared<QuantizerParam>(QuantizerType::kFP16);
  }
};

TEST_P(QuantizedGraphFp16Test, StoredValuesRefinedWithOriginalFp32Query) {
  // Even below the usual small-index threshold, FP16 SymQG must preserve
  // the original FP32 query instead of taking the rounded-query BF path.
  index_.force_graph_search = false;
  auto codec = zvec::core::IndexFactory::CreateQuantizer("Fp16Quantizer");
  zvec::core::IndexMeta meta;
  meta.set_meta(zvec::core::IndexMeta::DT_FP32, kDimension);
  meta.set_metric(
      GetParam() == MetricType::kCosine ? "Cosine" : "SquaredEuclidean", 0,
      zvec::ailego::Params{});
  ASSERT_EQ(0, codec->init(meta, {}));
  for (size_t repeat = 0; repeat < 3; ++repeat) {
    // Retain non-FP16-representable query values. A rounded query would fail
    // the score comparison against this independent double-precision oracle.
    for (size_t d = 0; d < kDimension; ++d)
      query_[d] = .137123f + float((d + repeat * 7) % 19) * .0312345f;
    search(128, 10, [](uint64_t key) { return key % 7 == 0; });
    std::vector<std::pair<double, uint64_t>> expected;
    double norm = 0;
    for (float x : query_) norm += double(x) * x;
    norm = std::sqrt(norm);
    std::string encoded(codec->quantized_datapoint_vector_length(), '\0');
    std::array<float, kDimension> stored;
    for (size_t id = 0; id < vectors_.size(); ++id) {
      if (id % 7 == 0) continue;
      codec->quantize_data(vectors_[id].data(), encoded.data());
      zvec::ailego::FloatHelper::ToFP32(
          reinterpret_cast<const uint16_t *>(encoded.data()), kDimension,
          stored.data());
      double distance = GetParam() == MetricType::kCosine ? 1 : 0;
      for (size_t d = 0; d < kDimension; ++d) {
        if (GetParam() == MetricType::kCosine)
          distance -= double(query_[d]) / norm * stored[d];
        else {
          const double diff = double(query_[d]) - stored[d];
          distance += diff * diff;
        }
      }
      expected.emplace_back(distance, id);
    }
    std::sort(expected.begin(), expected.end());
    ASSERT_EQ(10u, result_.doc_list_.size());
    for (size_t i = 0; i < 10; ++i) {
      EXPECT_EQ(expected[i].second, result_.doc_list_[i].key());
      EXPECT_NEAR(expected[i].first, result_.doc_list_[i].score(), 2e-6);
    }
  }
}

TEST_P(QuantizedGraphFp16Test, BudgetsAndStorageReopens) {
  for (auto type : {StorageOptions::StorageType::kBufferPool,
                    StorageOptions::StorageType::kMMAP}) {
    ASSERT_EQ(0, index_.close());
    HNSWIndexParam param(GetParam(), kDimension, 16, 100);
    param.data_type = DataType::DT_FP32;
    param.use_id_map = true;
    param.quantizer_param = quantizer();
    param.use_quantized_graph = true;
    ASSERT_EQ(0, index_.init(param));
    StorageOptions storage;
    storage.type = type;
    storage.read_only = true;
    ASSERT_EQ(0, index_.open(path_, storage));
    search(3, 10, [](uint64_t) { return true; });
    EXPECT_LE(index_.expanded, 3u);
    EXPECT_EQ(0u, index_.comparisons);
    EXPECT_TRUE(result_.doc_list_.empty());
    for (uint32_t cap = 1; cap <= 5; ++cap) {
      index_.exact_scan_budget = cap;
      search(128, 10, [](uint64_t) { return false; });
      EXPECT_LE(index_.comparisons, cap);
      EXPECT_GT(index_.comparisons, 0u);
    }
    index_.exact_scan_budget = 0;
  }
}

TEST_P(QuantizedGraphFp16Test, NativeFp32BatchRestoresStoredQueryContext) {
  auto *streamer = index_.native_streamer();
  ASSERT_NE(nullptr, streamer);
  auto context = streamer->create_context();
  auto *ctx = dynamic_cast<zvec::core::HnswContext *>(context.get());
  ASSERT_NE(nullptr, ctx);
  ctx->set_topk(10);
  ctx->set_ef(800);
  ctx->set_quantized_graph_options({1e6f, 0, 128, 0});
  zvec::core::IndexQueryMeta fp32_meta(zvec::core::IndexMeta::DT_FP32,
                                       kDimension);
  std::vector<float> queries(2 * kDimension);
  std::copy(query_.begin(), query_.end(), queries.begin());
  for (size_t d = 0; d < kDimension; ++d)
    queries[kDimension + d] = .137123f + float(d % 19) * .0312345f;
  const auto original_queries = queries;

  auto codec = zvec::core::IndexFactory::CreateQuantizer("Fp16Quantizer");
  zvec::core::IndexMeta meta(zvec::core::IndexMeta::DT_FP32, kDimension);
  meta.set_metric(
      GetParam() == MetricType::kCosine ? "Cosine" : "SquaredEuclidean", 0,
      zvec::ailego::Params{});
  ASSERT_EQ(0, codec->init(meta, {}));
  std::string encoded;
  zvec::core::IndexQueryMeta stored_meta;
  ASSERT_EQ(0,
            codec->quantize(queries.data(), fp32_meta, &encoded, &stored_meta));
  ASSERT_EQ(0, streamer->search_impl(encoded.data(), stored_meta, 1, context));
  std::vector<std::pair<uint64_t, float>> stored_results;
  for (const auto &doc : context->result())
    stored_results.emplace_back(doc.key(), doc.score());
  ASSERT_EQ(10u, stored_results.size());

  auto param = std::make_shared<HNSWQueryParam>();
  param->ef_search = 800;
  param->quantized_graph_margin = 1e6f;
  param->quantized_graph_max_expansions = 128;
  param->quantized_graph_route_beam = 0;
  std::vector<std::vector<std::pair<uint64_t, float>>> expected(2);
  for (size_t q = 0; q < 2; ++q) {
    SearchResult result;
    ASSERT_EQ(0, index_.search({DenseVector{queries.data() + q * kDimension}},
                               param, &result));
    for (const auto &doc : result.doc_list_)
      expected[q].emplace_back(doc.key(), doc.score());
  }
  ASSERT_EQ(0, streamer->search_impl(queries.data(), fp32_meta, 2, context));
  EXPECT_EQ(nullptr, ctx->quantized_graph_fp32_query());
  EXPECT_EQ(original_queries, queries);
  for (size_t q = 0; q < 2; ++q) {
    const auto &actual = context->result(q);
    ASSERT_EQ(expected[q].size(), actual.size());
    for (size_t i = 0; i < actual.size(); ++i) {
      EXPECT_EQ(expected[q][i].first, actual[i].key());
      EXPECT_EQ(expected[q][i].second, actual[i].score());
    }
  }
  ASSERT_EQ(0, streamer->search_impl(encoded.data(), stored_meta, 1, context));
  ASSERT_EQ(stored_results.size(), context->result().size());
  for (size_t i = 0; i < stored_results.size(); ++i) {
    EXPECT_EQ(stored_results[i].first, context->result()[i].key());
    EXPECT_EQ(stored_results[i].second, context->result()[i].score());
  }
  EXPECT_EQ(
      zvec::core::IndexError_Mismatch,
      streamer->search_impl(queries.data(),
                            zvec::core::IndexQueryMeta(
                                zvec::core::IndexMeta::DT_FP32, kDimension + 1),
                            1, context));
  EXPECT_EQ(zvec::core::IndexError_InvalidArgument,
            streamer->search_impl(nullptr, fp32_meta, 1, context));
}

TEST_P(QuantizedGraphFp16Test, FallbackSearchesPreserveStoredQueryScores) {
  HNSWIndexParam config(GetParam(), kDimension, 16, 100);
  config.data_type = DataType::DT_FP32;
  config.use_id_map = true;
  config.quantizer_param = quantizer();
  BudgetHnswIndex reference;
  ASSERT_EQ(0, reference.init(config));
  StorageOptions storage;
  storage.type = StorageOptions::StorageType::kMMAP;
  storage.read_only = true;
  ASSERT_EQ(0, reference.open(path_, storage));
  for (int mode = 0; mode < 3; ++mode) {
    auto param = std::make_shared<HNSWQueryParam>();
    param->ef_search = 800;
    param->fetch_vector = true;
    if (mode == 1) {
      param->bf_pks = std::make_shared<std::vector<uint64_t>>();
      for (uint64_t id = 0; id < vectors_.size(); id += 3)
        param->bf_pks->push_back(id);
    } else if (mode == 0) {
      param->is_linear = true;
    } else {
      // Small grouped queries retain the ordinary brute-force shortcut.
      index_.force_graph_search = false;
      reference.force_graph_search = false;
      param->ef_search = 16;
      param->group_by_param = std::make_shared<GroupByParam>();
      param->group_by_param->group_topk = 2;
      param->group_by_param->group_count = 3;
      param->group_by_param->group_by = [](uint64_t key) {
        return std::to_string(key % 3);
      };
    }
    SearchResult expected, actual;
    ASSERT_EQ(0,
              reference.search({DenseVector{query_.data()}}, param, &expected));
    ASSERT_EQ(0, index_.search({DenseVector{query_.data()}}, param, &actual));
    if (mode == 2) {
      EXPECT_EQ(0u, index_.expanded);
      ASSERT_EQ(3u, actual.group_doc_list_.size());
      ASSERT_EQ(expected.group_doc_list_.size(), actual.group_doc_list_.size());
      EXPECT_EQ(expected.group_reverted_vector_list_,
                actual.group_reverted_vector_list_);
      for (size_t g = 0; g < actual.group_doc_list_.size(); ++g) {
        const auto &a = actual.group_doc_list_[g];
        const auto &e = expected.group_doc_list_[g];
        EXPECT_EQ(e.group_id(), a.group_id());
        ASSERT_EQ(e.docs().size(), a.docs().size());
        for (size_t i = 0; i < a.docs().size(); ++i) {
          EXPECT_EQ(e.docs()[i].key(), a.docs()[i].key());
          EXPECT_EQ(e.docs()[i].score(), a.docs()[i].score());
        }
      }
      continue;
    }
    ASSERT_EQ(10u, actual.doc_list_.size());
    ASSERT_EQ(expected.doc_list_.size(), actual.doc_list_.size());
    EXPECT_EQ(expected.reverted_vector_list_, actual.reverted_vector_list_);
    for (size_t i = 0; i < actual.doc_list_.size(); ++i) {
      EXPECT_EQ(expected.doc_list_[i].key(), actual.doc_list_[i].key());
      EXPECT_EQ(expected.doc_list_[i].score(), actual.doc_list_[i].score());
    }
  }
  ASSERT_EQ(0, reference.close());
}

TEST(QuantizedGraphFp16, RouterTrainingNormalizesRoundedDirections) {
  if (!zvec::core::HnswQuantizedGraph::supported()) GTEST_SKIP();
  const std::string path = "qg_fp16_router_rounding.index";
  zvec::test_util::RemoveTestFiles(path + "*");
  HNSWIndexParam param(MetricType::kCosine, 3, 16, 100);
  param.data_type = DataType::DT_FP32;
  param.use_id_map = true;
  param.quantizer_param =
      std::make_shared<QuantizerParam>(QuantizerType::kFP16);
  BudgetHnswIndex builder;
  ASSERT_EQ(0, builder.init(param));
  ASSERT_EQ(0, builder.open(path, {StorageOptions::StorageType::kMMAP, true}));
  std::array<float, 3> v{1.f, 1.f, 1.f};
  zvec::ailego::Float16 rounded;
  rounded = 1.f / std::sqrt(3.f);
  ASSERT_GT(std::abs(std::sqrt(3.) * float(rounded) - 1.), 1e-4);
  for (uint64_t id = 0; id < 32; ++id)
    ASSERT_EQ(0, builder.add({DenseVector{v.data()}}, id));
  ASSERT_EQ(0, builder.train_router(16, 3));
  ASSERT_EQ(0, builder.close());
  zvec::test_util::RemoveTestFiles(path + "*");
}


INSTANTIATE_TEST_SUITE_P(CosineAndL2, QuantizedGraphFp16Test,
                         testing::Values(MetricType::kCosine,
                                         MetricType::kL2sq));
