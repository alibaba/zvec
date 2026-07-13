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

#include "flat/flat_searcher.h"
#include <future>
#include <iostream>
#include <vector>
#include <ailego/utility/math_helper.h>
#include <gtest/gtest.h>
#include <zvec/ailego/container/vector.h>
#include "flat/flat_builder.h"

using namespace zvec::core;
using namespace zvec::ailego;
using namespace std;

static const std::string INDEX_PATH = "brute_force_searcher_test/out.indexes";

static void BuildIndex(const IndexMeta &meta, IndexHolder::Pointer holder,
                       const std::string &path) {
  auto builder = IndexFactory::CreateBuilder("FlatBuilder");
  auto dumper = IndexFactory::CreateDumper("FileDumper");

  ASSERT_NE(nullptr, builder);
  ASSERT_NE(nullptr, dumper);

  Params params;
  ASSERT_EQ(0, builder->init(meta, params));
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, IndexBuilder::TrainBuildAndDump(builder, holder, dumper));
  ASSERT_EQ(0, dumper->close());

  auto stats = builder->stats();
  ASSERT_EQ(0UL, stats.trained_count());
  ASSERT_EQ(0UL, stats.discarded_count());
}

static void BuildIndex(const IndexMeta &meta, const Params &params,
                       IndexHolder::Pointer holder, const std::string &path) {
  auto builder = IndexFactory::CreateBuilder("FlatBuilder");
  auto dumper = IndexFactory::CreateDumper("FileDumper");

  ASSERT_NE(nullptr, builder);
  ASSERT_NE(nullptr, dumper);

  ASSERT_EQ(0, builder->init(meta, params));
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, IndexBuilder::TrainBuildAndDump(builder, holder, dumper));
  ASSERT_EQ(0, dumper->close());

  auto stats = builder->stats();
  ASSERT_EQ(0UL, stats.trained_count());
  ASSERT_EQ(0UL, stats.discarded_count());
}

static void LoadIndex(const std::string &path,
                      IndexSearcher::Pointer &searcher) {
  searcher = IndexFactory::CreateSearcher("FlatSearcher");
  auto storage = IndexFactory::CreateStorage("MMapFileReadStorage");

  ASSERT_NE(nullptr, searcher);
  ASSERT_NE(nullptr, storage);

  Params params;
  ASSERT_EQ(0, searcher->init(params));
  ASSERT_EQ(0, storage->open(path, false));
  ASSERT_EQ(0, searcher->load(storage, IndexMetric::Pointer()));
}

static void Shuffle(std::vector<uint32_t> &keys) {
  if (keys.size() <= 1) {
    return;
  }
  for (size_t i = keys.size() - 1; i > 0; i--) {
    std::mt19937 gen((std::random_device())());
    std::uniform_int_distribution<size_t> dist(0, i);
    size_t pos = dist(gen);
    std::swap(keys[i], keys[pos]);
  }
}

TEST(FlatSearcher, NoBatch_FP32) {
  std::mt19937 gen((std::random_device())());
  auto dist = std::uniform_real_distribution<float>(0.0f, 1.0f);
  size_t dim = (std::uniform_int_distribution<size_t>(1, 512))(gen);

  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("FlatBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder = std::make_shared<MultiPassIndexHolder<IndexMeta::DT_FP32>>(dim);
  uint32_t document_count =
      (std::uniform_int_distribution<size_t>(1, 10000))(gen);
  for (uint32_t i = 0; i < document_count; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < vec.size(); ++j) {
      vec[j] = dist(gen) + static_cast<float>(i * 5);
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  IndexMeta meta1;
  meta1.set_meta(IndexMeta::DataType::DT_FP32, dim);
  meta1.set_metric("SquaredEuclidean", 0, Params());
  meta1.set_major_order(IndexMeta::MO_ROW);
  BuildIndex(meta1, holder, INDEX_PATH + ".1");

  IndexMeta meta2;
  meta2.set_meta(IndexMeta::DataType::DT_FP32, dim);
  meta2.set_metric("SquaredEuclidean", 0, Params());
  meta2.set_major_order(IndexMeta::MO_COLUMN);
  BuildIndex(meta2, holder, INDEX_PATH + ".2");

  IndexSearcher::Pointer searcher1, searcher2;
  LoadIndex(INDEX_PATH + ".1", searcher1);
  LoadIndex(INDEX_PATH + ".2", searcher2);

  auto context1 = searcher1->create_context();
  auto context2 = searcher2->create_context();
  auto context3 = searcher1->create_context();
  auto context4 = searcher2->create_context();
  auto context5 = searcher1->create_context();
  auto context6 = searcher2->create_context();
  uint32_t topk = std::min(10u, document_count);
  context1->set_topk(topk);
  context2->set_topk(topk);
  context3->set_topk(topk);
  context4->set_topk(topk);
  context3->set_filter([](uint64_t) { return false; });
  context4->set_filter([](uint64_t) { return false; });
  context5->set_filter([](uint64_t) { return true; });
  context6->set_filter([](uint64_t) { return true; });

  uint32_t query_count = (std::uniform_int_distribution<size_t>(1, 100))(gen);
  for (uint32_t i = 0; i < query_count; i++) {
    NumericalVector<float> vec(dim);
    for (uint32_t j = 0; j < vec.size(); ++j) {
      vec[j] = dist(gen);
    }
    ASSERT_EQ(
        0, searcher1->search_impl(
               vec.data(), IndexQueryMeta(IndexMeta::DT_FP32, vec.dimension()),
               context1));
    ASSERT_EQ(
        0, searcher2->search_impl(
               vec.data(), IndexQueryMeta(IndexMeta::DT_FP32, vec.dimension()),
               context2));
    ASSERT_EQ(topk, context1->result().size());
    ASSERT_EQ(topk, context2->result().size());

    // Test shared context
    ASSERT_EQ(
        0, searcher1->search_impl(
               vec.data(), IndexQueryMeta(IndexMeta::DT_FP32, vec.dimension()),
               context4));
    ASSERT_EQ(
        0, searcher2->search_impl(
               vec.data(), IndexQueryMeta(IndexMeta::DT_FP32, vec.dimension()),
               context3));
    ASSERT_EQ(topk, context3->result().size());
    ASSERT_EQ(topk, context4->result().size());

    ASSERT_EQ(
        0, searcher1->search_impl(
               vec.data(), IndexQueryMeta(IndexMeta::DT_FP32, vec.dimension()),
               context5));
    ASSERT_EQ(
        0, searcher2->search_impl(
               vec.data(), IndexQueryMeta(IndexMeta::DT_FP32, vec.dimension()),
               context6));
    ASSERT_EQ(0u, context5->result().size());
    ASSERT_EQ(0u, context6->result().size());

    for (uint32_t j = 0; j < topk; ++j) {
      auto &result1 = context1->result();
      auto &result2 = context2->result();
      auto &result3 = context3->result();
      auto &result4 = context4->result();
      ASSERT_EQ(result1[j].index(), result2[j].index());
      ASSERT_EQ(result1[j].key(), result2[j].key());
      MathHelper math_help = MathHelper();
      bool score_cmp_result =
          math_help.IsAlmostEqual(result1[j].score(), result2[j].score(), 10);
      ASSERT_FLOAT_EQ(true, score_cmp_result);

      ASSERT_EQ(result1[j].index(), result3[j].index());
      ASSERT_EQ(result1[j].key(), result3[j].key());
      score_cmp_result =
          math_help.IsAlmostEqual(result1[j].score(), result3[j].score(), 10);
      ASSERT_FLOAT_EQ(true, score_cmp_result);

      ASSERT_EQ(result2[j].index(), result4[j].index());
      ASSERT_EQ(result2[j].key(), result4[j].key());
      score_cmp_result =
          math_help.IsAlmostEqual(result2[j].score(), result4[j].score(), 10);
      ASSERT_FLOAT_EQ(true, score_cmp_result);
    }
  }
}

TEST(FlatSearcher, NoBatch_FP16) {
  std::mt19937 gen((std::random_device())());
  auto dist = std::uniform_real_distribution<float>(-1.0f, 1.0f);
  size_t dim = (std::uniform_int_distribution<size_t>(1, 64))(gen);

  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("FlatBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder = std::make_shared<MultiPassIndexHolder<IndexMeta::DT_FP16>>(dim);
  uint32_t document_count =
      (std::uniform_int_distribution<size_t>(1, 10000))(gen);
  for (uint32_t i = 0; i < document_count; i++) {
    NumericalVector<Float16> vec(dim);
    for (size_t j = 0; j < vec.size(); ++j) {
      vec[j] = dist(gen) + static_cast<float>(i * 5);
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  IndexMeta meta1;
  meta1.set_meta(IndexMeta::DataType::DT_FP16, dim);
  meta1.set_metric("SquaredEuclidean", 0, Params());
  meta1.set_major_order(IndexMeta::MO_ROW);
  BuildIndex(meta1, holder, INDEX_PATH + ".1");

  IndexMeta meta2;
  meta2.set_meta(IndexMeta::DataType::DT_FP16, dim);
  meta2.set_metric("SquaredEuclidean", 0, Params());
  meta2.set_major_order(IndexMeta::MO_COLUMN);
  BuildIndex(meta2, holder, INDEX_PATH + ".2");

  IndexSearcher::Pointer searcher1, searcher2;
  LoadIndex(INDEX_PATH + ".1", searcher1);
  LoadIndex(INDEX_PATH + ".2", searcher2);

  auto context1 = searcher1->create_context();
  auto context2 = searcher2->create_context();
  auto context3 = searcher1->create_context();
  auto context4 = searcher2->create_context();
  auto context5 = searcher1->create_context();
  auto context6 = searcher2->create_context();
  uint32_t topk = std::min(10u, document_count);
  context1->set_topk(topk);
  context2->set_topk(topk);
  context3->set_topk(topk);
  context4->set_topk(topk);
  context3->set_filter([](uint64_t) { return false; });
  context4->set_filter([](uint64_t) { return false; });
  context5->set_filter([](uint64_t) { return true; });
  context6->set_filter([](uint64_t) { return true; });

  uint32_t query_count = (std::uniform_int_distribution<size_t>(1, 100))(gen);
  for (uint32_t i = 0; i < query_count; i++) {
    NumericalVector<Float16> vec(dim);
    for (uint32_t j = 0; j < vec.size(); ++j) {
      vec[j] = dist(gen);
    }
    ASSERT_EQ(
        0, searcher1->search_impl(
               vec.data(), IndexQueryMeta(IndexMeta::DT_FP16, vec.dimension()),
               context1));
    ASSERT_EQ(
        0, searcher2->search_impl(
               vec.data(), IndexQueryMeta(IndexMeta::DT_FP16, vec.dimension()),
               context2));
    ASSERT_EQ(topk, context1->result().size());
    ASSERT_EQ(topk, context2->result().size());

    // Test shared context
    ASSERT_EQ(
        0, searcher1->search_impl(
               vec.data(), IndexQueryMeta(IndexMeta::DT_FP16, vec.dimension()),
               context4));
    ASSERT_EQ(
        0, searcher2->search_impl(
               vec.data(), IndexQueryMeta(IndexMeta::DT_FP16, vec.dimension()),
               context3));
    ASSERT_EQ(topk, context3->result().size());
    ASSERT_EQ(topk, context4->result().size());

    ASSERT_EQ(
        0, searcher1->search_impl(
               vec.data(), IndexQueryMeta(IndexMeta::DT_FP16, vec.dimension()),
               context5));
    ASSERT_EQ(
        0, searcher2->search_impl(
               vec.data(), IndexQueryMeta(IndexMeta::DT_FP16, vec.dimension()),
               context6));
    ASSERT_EQ(0u, context5->result().size());
    ASSERT_EQ(0u, context6->result().size());

    for (uint32_t j = 0; j < topk; ++j) {
      auto &result1 = context1->result();
      auto &result2 = context2->result();
      auto &result3 = context3->result();
      auto &result4 = context4->result();
      ASSERT_EQ(result1[j].index(), result2[j].index());
      ASSERT_EQ(result1[j].key(), result2[j].key());
      MathHelper math_help = MathHelper();
      bool score_cmp_result = math_help.IsAlmostEqual(
          result1[j].score(), result2[j].score(), 10000);
      ASSERT_FLOAT_EQ(true, score_cmp_result);

      ASSERT_EQ(result1[j].index(), result3[j].index());
      ASSERT_EQ(result1[j].key(), result3[j].key());
      score_cmp_result = math_help.IsAlmostEqual(result1[j].score(),
                                                 result3[j].score(), 10000);
      ASSERT_FLOAT_EQ(true, score_cmp_result);

      ASSERT_EQ(result2[j].index(), result4[j].index());
      ASSERT_EQ(result2[j].key(), result4[j].key());
      score_cmp_result = math_help.IsAlmostEqual(result2[j].score(),
                                                 result4[j].score(), 10000);
      ASSERT_FLOAT_EQ(true, score_cmp_result);
    }
  }
}

TEST(FlatSearcher, NoBatch_INT8) {
  std::mt19937 gen((std::random_device())());
  auto dist = std::uniform_int_distribution<int>(-127, 127);
  size_t dim =
      ((std::uniform_int_distribution<size_t>(1, 512))(gen) + 3) / 4 * 4;

  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("FlatBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder = std::make_shared<MultiPassIndexHolder<IndexMeta::DT_INT8>>(dim);
  uint32_t document_count =
      (std::uniform_int_distribution<uint32_t>(1, 10000))(gen);
  for (uint32_t i = 0; i < document_count; i++) {
    NumericalVector<int8_t> vec(dim);
    for (size_t j = 0; j < vec.size(); ++j) {
      vec[j] = (int8_t)dist(gen);
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  IndexMeta meta1;
  meta1.set_meta(IndexMeta::DataType::DT_INT8, dim);
  meta1.set_metric("SquaredEuclidean", 0, Params());
  meta1.set_major_order(IndexMeta::MO_ROW);
  BuildIndex(meta1, holder, INDEX_PATH + ".1");

  IndexMeta meta2;
  meta2.set_meta(IndexMeta::DataType::DT_INT8, dim);
  meta2.set_metric("SquaredEuclidean", 0, Params());
  meta2.set_major_order(IndexMeta::MO_COLUMN);
  BuildIndex(meta2, holder, INDEX_PATH + ".2");

  IndexSearcher::Pointer searcher1, searcher2;
  LoadIndex(INDEX_PATH + ".1", searcher1);
  LoadIndex(INDEX_PATH + ".2", searcher2);

  auto context1 = searcher1->create_context();
  auto context2 = searcher2->create_context();
  auto context3 = searcher1->create_context();
  auto context4 = searcher2->create_context();
  auto context5 = searcher1->create_context();
  auto context6 = searcher2->create_context();
  uint32_t topk = std::min(10u, document_count);
  context1->set_topk(topk);
  context2->set_topk(topk);
  context3->set_topk(topk);
  context4->set_topk(topk);
  context3->set_filter([](uint64_t) { return false; });
  context4->set_filter([](uint64_t) { return false; });
  context5->set_filter([](uint64_t) { return true; });
  context6->set_filter([](uint64_t) { return true; });

  uint32_t query_count = (std::uniform_int_distribution<size_t>(1, 100))(gen);
  for (uint32_t i = 0; i < query_count; i++) {
    NumericalVector<int8_t> vec(dim);
    for (uint32_t j = 0; j < vec.size(); ++j) {
      vec[j] = (int8_t)dist(gen);
    }
    ASSERT_EQ(
        0, searcher1->search_impl(
               vec.data(), IndexQueryMeta(IndexMeta::DT_INT8, vec.dimension()),
               context1));
    ASSERT_EQ(
        0, searcher2->search_impl(
               vec.data(), IndexQueryMeta(IndexMeta::DT_INT8, vec.dimension()),
               context2));
    ASSERT_EQ(topk, context1->result().size());
    ASSERT_EQ(topk, context2->result().size());

    // Test shared context
    ASSERT_EQ(
        0, searcher1->search_impl(
               vec.data(), IndexQueryMeta(IndexMeta::DT_INT8, vec.dimension()),
               context4));
    ASSERT_EQ(
        0, searcher2->search_impl(
               vec.data(), IndexQueryMeta(IndexMeta::DT_INT8, vec.dimension()),
               context3));
    ASSERT_EQ(topk, context3->result().size());
    ASSERT_EQ(topk, context4->result().size());

    ASSERT_EQ(
        0, searcher1->search_impl(
               vec.data(), IndexQueryMeta(IndexMeta::DT_INT8, vec.dimension()),
               context5));
    ASSERT_EQ(
        0, searcher2->search_impl(
               vec.data(), IndexQueryMeta(IndexMeta::DT_INT8, vec.dimension()),
               context6));
    ASSERT_EQ(0u, context5->result().size());
    ASSERT_EQ(0u, context6->result().size());

    for (uint32_t j = 0; j < topk; ++j) {
      auto &result1 = context1->result();
      auto &result2 = context2->result();
      auto &result3 = context3->result();
      auto &result4 = context4->result();
      ASSERT_EQ(result1[j].index(), result2[j].index());
      ASSERT_EQ(result1[j].key(), result2[j].key());
      ASSERT_FLOAT_EQ(result1[j].score(), result2[j].score());

      ASSERT_EQ(result1[j].index(), result3[j].index());
      ASSERT_EQ(result1[j].key(), result3[j].key());
      ASSERT_FLOAT_EQ(result1[j].score(), result3[j].score());

      ASSERT_EQ(result2[j].index(), result4[j].index());
      ASSERT_EQ(result2[j].key(), result4[j].key());
      ASSERT_FLOAT_EQ(result2[j].score(), result4[j].score());
    }
  }
}

TEST(FlatSearcher, NoBatch_Binary32) {
  std::mt19937 gen((std::random_device())());
  auto dist = std::uniform_int_distribution<uint32_t>(1, 512);
  size_t dim = (dist(gen) + 31) / 32 * 32;

  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("FlatBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder =
      std::make_shared<MultiPassIndexHolder<IndexMeta::DT_BINARY32>>(dim);
  uint32_t document_count =
      (std::uniform_int_distribution<uint32_t>(1, 10000))(gen);
  for (uint32_t i = 0; i < document_count; i++) {
    BinaryVector<uint32_t> vec(dim);
    for (size_t j = 0; j < vec.dimension(); ++j) {
      if (dist(gen) % 3 == 0) {
        vec.set(j);
      }
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  IndexMeta meta1;
  meta1.set_meta(IndexMeta::DataType::DT_BINARY32, dim);
  meta1.set_metric("SquaredEuclidean", 0, Params());
  meta1.set_major_order(IndexMeta::MO_ROW);
  BuildIndex(meta1, holder, INDEX_PATH + ".1");

  IndexMeta meta2;
  meta2.set_meta(IndexMeta::DataType::DT_BINARY32, dim);
  meta2.set_metric("SquaredEuclidean", 0, Params());
  meta2.set_major_order(IndexMeta::MO_COLUMN);
  BuildIndex(meta2, holder, INDEX_PATH + ".2");

  IndexSearcher::Pointer searcher1, searcher2;
  LoadIndex(INDEX_PATH + ".1", searcher1);
  LoadIndex(INDEX_PATH + ".2", searcher2);

  auto context1 = searcher1->create_context();
  auto context2 = searcher2->create_context();
  auto context3 = searcher1->create_context();
  auto context4 = searcher2->create_context();
  auto context5 = searcher1->create_context();
  auto context6 = searcher2->create_context();
  uint32_t topk = std::min(10u, document_count);
  context1->set_topk(topk);
  context2->set_topk(topk);
  context3->set_topk(topk);
  context4->set_topk(topk);
  context3->set_filter([](uint64_t) { return false; });
  context4->set_filter([](uint64_t) { return false; });
  context5->set_filter([](uint64_t) { return true; });
  context6->set_filter([](uint64_t) { return true; });

  uint32_t query_count = (std::uniform_int_distribution<size_t>(1, 100))(gen);
  for (uint32_t i = 0; i < query_count; i++) {
    BinaryVector<uint32_t> vec(dim);
    for (size_t j = 0; j < vec.dimension(); ++j) {
      if (dist(gen) % 7 == 0) {
        vec.set(j);
      }
    }
    ASSERT_EQ(0, searcher1->search_impl(
                     vec.data(),
                     IndexQueryMeta(IndexMeta::DT_BINARY32, vec.dimension()),
                     context1));
    ASSERT_EQ(0, searcher2->search_impl(
                     vec.data(),
                     IndexQueryMeta(IndexMeta::DT_BINARY32, vec.dimension()),
                     context2));
    ASSERT_EQ(topk, context1->result().size());
    ASSERT_EQ(topk, context2->result().size());

    // Test shared context
    ASSERT_EQ(0, searcher1->search_impl(
                     vec.data(),
                     IndexQueryMeta(IndexMeta::DT_BINARY32, vec.dimension()),
                     context4));
    ASSERT_EQ(0, searcher2->search_impl(
                     vec.data(),
                     IndexQueryMeta(IndexMeta::DT_BINARY32, vec.dimension()),
                     context3));
    ASSERT_EQ(topk, context3->result().size());
    ASSERT_EQ(topk, context4->result().size());

    ASSERT_EQ(0, searcher1->search_impl(
                     vec.data(),
                     IndexQueryMeta(IndexMeta::DT_BINARY32, vec.dimension()),
                     context5));
    ASSERT_EQ(0, searcher2->search_impl(
                     vec.data(),
                     IndexQueryMeta(IndexMeta::DT_BINARY32, vec.dimension()),
                     context6));
    ASSERT_EQ(0u, context5->result().size());
    ASSERT_EQ(0u, context6->result().size());

    for (uint32_t j = 0; j < topk; ++j) {
      auto &result1 = context1->result();
      auto &result2 = context2->result();
      auto &result3 = context3->result();
      auto &result4 = context4->result();
      ASSERT_EQ(result1[j].index(), result2[j].index());
      ASSERT_EQ(result1[j].key(), result2[j].key());
      ASSERT_FLOAT_EQ(result1[j].score(), result2[j].score());

      ASSERT_EQ(result1[j].index(), result3[j].index());
      ASSERT_EQ(result1[j].key(), result3[j].key());
      ASSERT_FLOAT_EQ(result1[j].score(), result3[j].score());

      ASSERT_EQ(result2[j].index(), result4[j].index());
      ASSERT_EQ(result2[j].key(), result4[j].key());
      ASSERT_FLOAT_EQ(result2[j].score(), result4[j].score());
    }
  }
}

TEST(FlatSearcher, RowBatch_FP32) {
  std::mt19937 gen((std::random_device())());
  auto dist = std::uniform_real_distribution<float>(0.0f, 1.0f);
  size_t dim = (std::uniform_int_distribution<size_t>(1, 512))(gen);

  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("FlatBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder = std::make_shared<MultiPassIndexHolder<IndexMeta::DT_FP32>>(dim);
  uint32_t document_count =
      (std::uniform_int_distribution<size_t>(1, 10000))(gen);
  for (uint32_t i = 0; i < document_count; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < vec.size(); ++j) {
      vec[j] = dist(gen) + static_cast<float>(i * 5);
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  IndexMeta meta1;
  meta1.set_meta(IndexMeta::DataType::DT_FP32, dim);
  meta1.set_metric("SquaredEuclidean", 0, Params());
  meta1.set_major_order(IndexMeta::MO_ROW);
  BuildIndex(meta1, holder, INDEX_PATH + ".1");

  IndexMeta meta2;
  meta2.set_meta(IndexMeta::DataType::DT_FP32, dim);
  meta2.set_metric("SquaredEuclidean", 0, Params());
  meta2.set_major_order(IndexMeta::MO_COLUMN);
  BuildIndex(meta2, holder, INDEX_PATH + ".2");

  IndexSearcher::Pointer searcher1, searcher2;
  LoadIndex(INDEX_PATH + ".1", searcher1);
  LoadIndex(INDEX_PATH + ".2", searcher2);

  auto context1 = searcher1->create_context();
  auto context2 = searcher2->create_context();
  uint32_t topk = std::min(10u, document_count);
  context1->set_topk(topk);
  context1->set_filter([](uint64_t) { return false; });
  context2->set_topk(topk);

  std::string query_buffer;
  uint32_t query_count = (std::uniform_int_distribution<size_t>(1, 100))(gen);

  for (uint32_t i = 0; i < query_count; i++) {
    NumericalVector<float> vec(dim);
    for (uint32_t j = 0; j < vec.size(); ++j) {
      vec[j] = dist(gen);
    }
    query_buffer.append((const char *)vec.data(), vec.bytes());
  }
  ASSERT_EQ(0, searcher1->search_impl(query_buffer.data(),
                                      IndexQueryMeta(IndexMeta::DT_FP32, dim),
                                      query_count, context1));

  NumericalVector<float> vec(dim);
  for (uint32_t i = 0; i < query_count; i++) {
    ASSERT_EQ(0, searcher2->search_impl(
                     (const float *)(&query_buffer[i * vec.bytes()]),
                     IndexQueryMeta(IndexMeta::DT_FP32, dim), context2));

    for (uint32_t j = 0; j < topk; ++j) {
      auto &result1 = context1->result(i);
      auto &result2 = context2->result();
      ASSERT_EQ(result1[j].index(), result2[j].index());
      ASSERT_EQ(result1[j].key(), result2[j].key());
      MathHelper math_help = MathHelper();
      bool score_cmp_result =
          math_help.IsAlmostEqual(result1[j].score(), result2[j].score(), 10);
      ASSERT_FLOAT_EQ(true, score_cmp_result);
    }
  }
}

TEST(FlatSearcher, RowBatch_FP16) {
  std::mt19937 gen((std::random_device())());
  auto dist = std::uniform_real_distribution<float>(-1.0f, 1.0f);
  size_t dim = (std::uniform_int_distribution<size_t>(1, 256))(gen);

  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("FlatBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder = std::make_shared<MultiPassIndexHolder<IndexMeta::DT_FP16>>(dim);
  uint32_t document_count =
      (std::uniform_int_distribution<size_t>(1, 10000))(gen);
  for (uint32_t i = 0; i < document_count; i++) {
    NumericalVector<Float16> vec(dim);
    for (size_t j = 0; j < vec.size(); ++j) {
      vec[j] = dist(gen) + static_cast<float>(i * 5);
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  IndexMeta meta1;
  meta1.set_meta(IndexMeta::DataType::DT_FP16, dim);
  meta1.set_metric("SquaredEuclidean", 0, Params());
  meta1.set_major_order(IndexMeta::MO_ROW);
  BuildIndex(meta1, holder, INDEX_PATH + ".1");

  IndexMeta meta2;
  meta2.set_meta(IndexMeta::DataType::DT_FP16, dim);
  meta2.set_metric("SquaredEuclidean", 0, Params());
  meta2.set_major_order(IndexMeta::MO_COLUMN);
  BuildIndex(meta2, holder, INDEX_PATH + ".2");

  IndexSearcher::Pointer searcher1, searcher2;
  LoadIndex(INDEX_PATH + ".1", searcher1);
  LoadIndex(INDEX_PATH + ".2", searcher2);

  auto context1 = searcher1->create_context();
  auto context2 = searcher2->create_context();
  uint32_t topk = std::min(10u, document_count);
  context1->set_topk(topk);
  context2->set_topk(topk);
  context2->set_filter([](uint64_t) { return false; });

  std::string query_buffer;
  uint32_t query_count = (std::uniform_int_distribution<size_t>(1, 100))(gen);

  for (uint32_t i = 0; i < query_count; i++) {
    NumericalVector<Float16> vec(dim);
    for (uint32_t j = 0; j < vec.size(); ++j) {
      vec[j] = dist(gen);
    }
    query_buffer.append((const char *)vec.data(), vec.bytes());
  }
  ASSERT_EQ(0, searcher1->search_impl(query_buffer.data(),
                                      IndexQueryMeta(IndexMeta::DT_FP16, dim),
                                      query_count, context1));

  NumericalVector<Float16> vec(dim);
  for (uint32_t i = 0; i < query_count; i++) {
    ASSERT_EQ(0, searcher2->search_impl((&query_buffer[i * vec.bytes()]),
                                        IndexQueryMeta(IndexMeta::DT_FP16, dim),
                                        context2));

    for (uint32_t j = 0; j < topk; ++j) {
      auto &result1 = context1->result(i);
      auto &result2 = context2->result();
      ASSERT_EQ(result1[j].index(), result2[j].index());
      ASSERT_EQ(result1[j].key(), result2[j].key());
      MathHelper math_help = MathHelper();
      bool score_cmp_result = math_help.IsAlmostEqual(
          result1[j].score(), result2[j].score(), 10000);
      ASSERT_FLOAT_EQ(true, score_cmp_result);
    }
  }
}

TEST(FlatSearcher, RowBatch_INT8) {
  std::mt19937 gen((std::random_device())());
  auto dist = std::uniform_int_distribution<int>(-127, 127);
  size_t dim =
      ((std::uniform_int_distribution<size_t>(1, 512))(gen) + 3) / 4 * 4;

  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("FlatBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder = std::make_shared<MultiPassIndexHolder<IndexMeta::DT_INT8>>(dim);
  uint32_t document_count =
      (std::uniform_int_distribution<uint32_t>(1, 10000))(gen);
  for (uint32_t i = 0; i < document_count; i++) {
    NumericalVector<int8_t> vec(dim);
    for (size_t j = 0; j < vec.size(); ++j) {
      vec[j] = (int8_t)dist(gen);
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  IndexMeta meta1;
  meta1.set_meta(IndexMeta::DataType::DT_INT8, dim);
  meta1.set_metric("SquaredEuclidean", 0, Params());
  meta1.set_major_order(IndexMeta::MO_ROW);
  BuildIndex(meta1, holder, INDEX_PATH + ".1");

  IndexMeta meta2;
  meta2.set_meta(IndexMeta::DataType::DT_INT8, dim);
  meta2.set_metric("SquaredEuclidean", 0, Params());
  meta2.set_major_order(IndexMeta::MO_COLUMN);
  BuildIndex(meta2, holder, INDEX_PATH + ".2");

  IndexSearcher::Pointer searcher1, searcher2;
  LoadIndex(INDEX_PATH + ".1", searcher1);
  LoadIndex(INDEX_PATH + ".2", searcher2);

  auto context1 = searcher1->create_context();
  auto context2 = searcher2->create_context();
  uint32_t topk = std::min(10u, document_count);
  context1->set_topk(topk);
  context1->set_filter([](uint64_t) { return false; });
  context2->set_topk(topk);
  context2->set_filter([](uint64_t) { return false; });

  std::string query_buffer;
  uint32_t query_count = (std::uniform_int_distribution<size_t>(1, 100))(gen);

  for (uint32_t i = 0; i < query_count; i++) {
    NumericalVector<int8_t> vec(dim);
    for (uint32_t j = 0; j < vec.size(); ++j) {
      vec[j] = (int8_t)dist(gen);
    }
    query_buffer.append((const char *)vec.data(), vec.bytes());
  }
  ASSERT_EQ(0, searcher1->search_impl(query_buffer.data(),
                                      IndexQueryMeta(IndexMeta::DT_INT8, dim),
                                      query_count, context1));

  NumericalVector<int8_t> vec(dim);
  for (uint32_t i = 0; i < query_count; i++) {
    ASSERT_EQ(0, searcher2->search_impl((&query_buffer[i * vec.bytes()]),
                                        IndexQueryMeta(IndexMeta::DT_INT8, dim),
                                        context2));

    for (uint32_t j = 0; j < topk; ++j) {
      auto &result1 = context1->result(i);
      auto &result2 = context2->result();
      ASSERT_EQ(result1[j].index(), result2[j].index());
      ASSERT_EQ(result1[j].key(), result2[j].key());
      ASSERT_FLOAT_EQ(result1[j].score(), result2[j].score());
    }
  }
}

TEST(FlatSearcher, RowBatch_Binary32) {
  std::mt19937 gen((std::random_device())());
  auto dist = std::uniform_int_distribution<uint32_t>(1, 512);
  size_t dim = (dist(gen) + 31) / 32 * 32;

  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("FlatBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder =
      std::make_shared<MultiPassIndexHolder<IndexMeta::DT_BINARY32>>(dim);
  uint32_t document_count =
      (std::uniform_int_distribution<uint32_t>(1, 10000))(gen);
  for (uint32_t i = 0; i < document_count; i++) {
    BinaryVector<uint32_t> vec(dim);
    for (size_t j = 0; j < vec.dimension(); ++j) {
      if (dist(gen) % 3 == 0) {
        vec.set(j);
      }
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  IndexMeta meta1;
  meta1.set_meta(IndexMeta::DataType::DT_BINARY32, dim);
  meta1.set_metric("SquaredEuclidean", 0, Params());
  meta1.set_major_order(IndexMeta::MO_ROW);
  BuildIndex(meta1, holder, INDEX_PATH + ".1");

  IndexMeta meta2;
  meta2.set_meta(IndexMeta::DataType::DT_BINARY32, dim);
  meta2.set_metric("SquaredEuclidean", 0, Params());
  meta2.set_major_order(IndexMeta::MO_COLUMN);
  BuildIndex(meta2, holder, INDEX_PATH + ".2");

  IndexSearcher::Pointer searcher1, searcher2;
  LoadIndex(INDEX_PATH + ".1", searcher1);
  LoadIndex(INDEX_PATH + ".2", searcher2);

  auto context1 = searcher1->create_context();
  auto context2 = searcher2->create_context();
  uint32_t topk = std::min(10u, document_count);
  context1->set_topk(topk);
  context2->set_topk(topk);

  std::string query_buffer;
  uint32_t query_count = (std::uniform_int_distribution<size_t>(1, 100))(gen);

  for (uint32_t i = 0; i < query_count; i++) {
    BinaryVector<uint32_t> vec(dim);
    for (uint32_t j = 0; j < vec.dimension(); ++j) {
      if (dist(gen) % 7 == 0) {
        vec.set(j);
      }
    }
    query_buffer.append((const char *)vec.data(), vec.bytes());
  }
  ASSERT_EQ(0,
            searcher1->search_impl(query_buffer.data(),
                                   IndexQueryMeta(IndexMeta::DT_BINARY32, dim),
                                   query_count, context1));

  BinaryVector<uint32_t> vec(dim);
  for (uint32_t i = 0; i < query_count; i++) {
    ASSERT_EQ(0, searcher2->search_impl(
                     (&query_buffer[i * vec.bytes()]),
                     IndexQueryMeta(IndexMeta::DT_BINARY32, dim), context2));

    for (uint32_t j = 0; j < topk; ++j) {
      auto &result1 = context1->result(i);
      auto &result2 = context2->result();
      ASSERT_EQ(result1[j].index(), result2[j].index());
      ASSERT_EQ(result1[j].key(), result2[j].key());
      ASSERT_FLOAT_EQ(result1[j].score(), result2[j].score());
    }
  }
}

TEST(FlatSearcher, ColumnBatch_Binary32) {
  std::mt19937 gen((std::random_device())());
  auto dist = std::uniform_int_distribution<uint32_t>(1, 512);
  size_t dim = (dist(gen) + 31) / 32 * 32;

  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("FlatBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder =
      std::make_shared<MultiPassIndexHolder<IndexMeta::DT_BINARY32>>(dim);
  uint32_t document_count =
      (std::uniform_int_distribution<uint32_t>(1, 10000))(gen);
  for (uint32_t i = 0; i < document_count; i++) {
    BinaryVector<uint32_t> vec(dim);
    for (size_t j = 0; j < vec.dimension(); ++j) {
      if (dist(gen) % 3 == 0) {
        vec.set(j);
      }
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  IndexMeta meta1;
  meta1.set_meta(IndexMeta::DataType::DT_BINARY32, dim);
  meta1.set_metric("SquaredEuclidean", 0, Params());
  meta1.set_major_order(IndexMeta::MO_ROW);

  Params params1;
  params1.set(PARAM_FLAT_COLUMN_MAJOR_ORDER, true);
  BuildIndex(meta1, params1, holder, INDEX_PATH + ".1");

  IndexMeta meta2;
  meta2.set_meta(IndexMeta::DataType::DT_BINARY32, dim);
  meta2.set_metric("SquaredEuclidean", 0, Params());
  meta2.set_major_order(IndexMeta::MO_ROW);
  BuildIndex(meta2, holder, INDEX_PATH + ".2");

  IndexSearcher::Pointer searcher1, searcher2;
  LoadIndex(INDEX_PATH + ".1", searcher1);
  LoadIndex(INDEX_PATH + ".2", searcher2);

  auto context1 = searcher1->create_context();
  auto context2 = searcher2->create_context();
  auto context3 = searcher1->create_context();
  auto context4 = searcher1->create_context();
  uint32_t topk = std::min(10u, document_count);
  context1->set_topk(topk);
  context2->set_topk(topk);
  context3->set_topk(topk);
  context3->set_filter([](uint64_t) { return false; });
  context4->set_topk(topk);
  context4->set_filter([](uint64_t) { return true; });

  std::string query_buffer;
  uint32_t query_count = (std::uniform_int_distribution<size_t>(1, 100))(gen);

  for (uint32_t i = 0; i < query_count; i++) {
    BinaryVector<uint32_t> vec(dim);
    for (uint32_t j = 0; j < vec.dimension(); ++j) {
      if (dist(gen) % 7 == 0) {
        vec.set(j);
      }
    }
    query_buffer.append((const char *)vec.data(), vec.bytes());
  }
  ASSERT_EQ(0,
            searcher1->search_impl(query_buffer.data(),
                                   IndexQueryMeta(IndexMeta::DT_BINARY32, dim),
                                   query_count, context1));
  ASSERT_EQ(0,
            searcher1->search_impl(query_buffer.data(),
                                   IndexQueryMeta(IndexMeta::DT_BINARY32, dim),
                                   query_count, context3));
  ASSERT_EQ(0,
            searcher1->search_impl(query_buffer.data(),
                                   IndexQueryMeta(IndexMeta::DT_BINARY32, dim),
                                   query_count, context4));

  BinaryVector<uint32_t> vec(dim);
  for (uint32_t i = 0; i < query_count; i++) {
    ASSERT_EQ(0, searcher1->search_impl(
                     (&query_buffer[i * vec.bytes()]),
                     IndexQueryMeta(IndexMeta::DT_BINARY32, dim), context2));

    for (uint32_t j = 0; j < topk; ++j) {
      auto &result1 = context1->result(i);
      auto &result2 = context2->result();
      auto &result3 = context3->result(i);
      auto &result4 = context4->result(i);
      EXPECT_TRUE(result4.empty());

      EXPECT_EQ(result1[j].index(), result2[j].index());
      EXPECT_EQ(result1[j].key(), result2[j].key());
      ASSERT_FLOAT_EQ(result1[j].score(), result2[j].score());

      EXPECT_EQ(result1[j].index(), result3[j].index());
      EXPECT_EQ(result1[j].key(), result3[j].key());
      ASSERT_FLOAT_EQ(result1[j].score(), result3[j].score());
    }
  }
}

TEST(FlatSearcher, ColumnBatch_FP32) {
  std::mt19937 gen((std::random_device())());
  auto dist = std::uniform_real_distribution<float>(0.0f, 1.0f);
  size_t dim = (std::uniform_int_distribution<size_t>(1, 512))(gen);

  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("FlatBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder = std::make_shared<MultiPassIndexHolder<IndexMeta::DT_FP32>>(dim);
  uint32_t document_count =
      (std::uniform_int_distribution<size_t>(1, 10000))(gen);
  for (uint32_t i = 0; i < document_count; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < vec.size(); ++j) {
      vec[j] = dist(gen) + static_cast<float>(i * 5);
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }


  IndexMeta meta1;
  meta1.set_meta(IndexMeta::DataType::DT_FP32, dim);
  meta1.set_metric("SquaredEuclidean", 0, Params());
  meta1.set_major_order(IndexMeta::MO_ROW);  // will invalide when set
                                             // column_major_order in params

  Params params1;
  params1.set(PARAM_FLAT_COLUMN_MAJOR_ORDER,
              true);  // make it MO_COLUMN
  BuildIndex(meta1, params1, holder, INDEX_PATH + ".1");

  IndexMeta meta2;
  meta2.set_meta(IndexMeta::DataType::DT_FP32, dim);
  meta2.set_metric("SquaredEuclidean", 0, Params());
  meta2.set_major_order(IndexMeta::MO_ROW);
  BuildIndex(meta2, holder, INDEX_PATH + ".2");

  IndexSearcher::Pointer searcher1, searcher2;
  LoadIndex(INDEX_PATH + ".1", searcher1);
  LoadIndex(INDEX_PATH + ".2", searcher2);

  auto context1 = searcher1->create_context();
  auto context2 = searcher2->create_context();
  auto context3 = searcher1->create_context();
  auto context4 = searcher1->create_context();
  uint32_t topk = std::min(10u, document_count);
  context1->set_topk(topk);
  context2->set_topk(topk);
  context3->set_topk(topk);
  context3->set_filter([](uint64_t) { return false; });  // same as no filter
  context4->set_topk(topk);
  context4->set_filter([](uint64_t) { return true; });  // filter all result

  std::string query_buffer;
  uint32_t query_count = (std::uniform_int_distribution<size_t>(1, 100))(gen);

  for (uint32_t i = 0; i < query_count; i++) {
    NumericalVector<float> vec(dim);
    for (uint32_t j = 0; j < vec.size(); ++j) {
      vec[j] = dist(gen);
    }
    query_buffer.append((const char *)vec.data(), vec.bytes());
  }

  ASSERT_EQ(0, searcher1->search_impl(query_buffer.data(),
                                      IndexQueryMeta(IndexMeta::DT_FP32, dim),
                                      query_count, context1));
  ASSERT_EQ(0, searcher1->search_impl(query_buffer.data(),
                                      IndexQueryMeta(IndexMeta::DT_FP32, dim),
                                      query_count, context3));
  ASSERT_EQ(0, searcher1->search_impl(query_buffer.data(),
                                      IndexQueryMeta(IndexMeta::DT_FP32, dim),
                                      query_count, context4));

  NumericalVector<float> vec(dim);
  for (uint32_t i = 0; i < query_count; i++) {
    // not batch
    ASSERT_EQ(0, searcher2->search_impl((&query_buffer[i * vec.bytes()]),
                                        IndexQueryMeta(IndexMeta::DT_FP32, dim),
                                        context2));

    for (uint32_t j = 0; j < topk; ++j) {
      auto &result1 = context1->result(i);
      auto &result2 = context2->result();
      auto &result3 = context3->result(i);
      auto &result4 = context4->result(i);
      EXPECT_TRUE(result4.empty());

      // batch result is equal to not batch result
      EXPECT_EQ(result1[j].index(), result2[j].index());
      EXPECT_EQ(result1[j].key(), result2[j].key());
      MathHelper math_help = MathHelper();
      bool score_cmp_result =
          math_help.IsAlmostEqual(result1[j].score(), result2[j].score(), 10);
      ASSERT_FLOAT_EQ(true, score_cmp_result);

      // test filter
      EXPECT_EQ(result1[j].index(), result3[j].index());
      EXPECT_EQ(result1[j].key(), result3[j].key());
      score_cmp_result =
          math_help.IsAlmostEqual(result1[j].score(), result3[j].score(), 10);
      ASSERT_FLOAT_EQ(true, score_cmp_result);
    }
  }
}

TEST(FlatSearcher, ColumnBatch_FP16) {
  std::mt19937 gen((std::random_device())());
  auto dist = std::uniform_real_distribution<float>(-1.0f, 1.0f);
  size_t dim = (std::uniform_int_distribution<size_t>(1, 256))(gen);

  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("FlatBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder = std::make_shared<MultiPassIndexHolder<IndexMeta::DT_FP16>>(dim);
  uint32_t document_count =
      (std::uniform_int_distribution<size_t>(1, 10000))(gen);
  for (uint32_t i = 0; i < document_count; i++) {
    NumericalVector<Float16> vec(dim);
    for (size_t j = 0; j < vec.size(); ++j) {
      vec[j] = dist(gen) + static_cast<float>(i * 5);
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }


  IndexMeta meta1;
  meta1.set_meta(IndexMeta::DataType::DT_FP16, dim);
  meta1.set_metric("SquaredEuclidean", 0, Params());
  meta1.set_major_order(IndexMeta::MO_ROW);  // will invalide when set
                                             // column_major_order in params

  Params params1;
  params1.set(PARAM_FLAT_COLUMN_MAJOR_ORDER,
              true);  // make it MO_COLUMN
  BuildIndex(meta1, params1, holder, INDEX_PATH + ".1");

  IndexMeta meta2;
  meta2.set_meta(IndexMeta::DataType::DT_FP16, dim);
  meta2.set_metric("SquaredEuclidean", 0, Params());
  meta2.set_major_order(IndexMeta::MO_ROW);
  BuildIndex(meta2, holder, INDEX_PATH + ".2");

  IndexSearcher::Pointer searcher1, searcher2;
  LoadIndex(INDEX_PATH + ".1", searcher1);
  LoadIndex(INDEX_PATH + ".2", searcher2);

  auto context1 = searcher1->create_context();
  auto context2 = searcher2->create_context();
  auto context3 = searcher1->create_context();
  auto context4 = searcher1->create_context();
  uint32_t topk = std::min(10u, document_count);
  context1->set_topk(topk);
  context2->set_topk(topk);
  context3->set_topk(topk);
  context3->set_filter([](uint64_t) { return false; });  // same as no filter
  context4->set_topk(topk);
  context4->set_filter([](uint64_t) { return true; });  // filter all result

  std::string query_buffer;
  uint32_t query_count = (std::uniform_int_distribution<size_t>(1, 100))(gen);

  for (uint32_t i = 0; i < query_count; i++) {
    NumericalVector<Float16> vec(dim);
    for (uint32_t j = 0; j < vec.size(); ++j) {
      vec[j] = dist(gen);
    }
    query_buffer.append((const char *)vec.data(), vec.bytes());
  }

  ASSERT_EQ(0, searcher1->search_impl(query_buffer.data(),
                                      IndexQueryMeta(IndexMeta::DT_FP16, dim),
                                      query_count, context1));
  ASSERT_EQ(0, searcher1->search_impl(query_buffer.data(),
                                      IndexQueryMeta(IndexMeta::DT_FP16, dim),
                                      query_count, context3));
  ASSERT_EQ(0, searcher1->search_impl(query_buffer.data(),
                                      IndexQueryMeta(IndexMeta::DT_FP16, dim),
                                      query_count, context4));

  NumericalVector<Float16> vec(dim);
  for (uint32_t i = 0; i < query_count; i++) {
    // not batch
    ASSERT_EQ(0, searcher2->search_impl((&query_buffer[i * vec.bytes()]),
                                        IndexQueryMeta(IndexMeta::DT_FP16, dim),
                                        query_count, context2));

    for (uint32_t j = 0; j < topk; ++j) {
      auto &result1 = context1->result(i);
      auto &result2 = context2->result();
      auto &result3 = context3->result(i);
      auto &result4 = context4->result(i);
      EXPECT_TRUE(result4.empty());

      // batch result is equal to not batch result
      EXPECT_EQ(result1[j].index(), result2[j].index());
      EXPECT_EQ(result1[j].key(), result2[j].key());
      MathHelper math_help = MathHelper();
      bool score_cmp_result = math_help.IsAlmostEqual(
          result1[j].score(), result2[j].score(), 10000);
      ASSERT_FLOAT_EQ(true, score_cmp_result);

      // test filter
      EXPECT_EQ(result1[j].index(), result3[j].index());
      EXPECT_EQ(result1[j].key(), result3[j].key());
      score_cmp_result = math_help.IsAlmostEqual(result1[j].score(),
                                                 result3[j].score(), 10000);
      ASSERT_FLOAT_EQ(true, score_cmp_result);
    }
  }
}

TEST(FlatSearcher, ColumnBatch_INT8) {
  std::mt19937 gen((std::random_device())());
  auto dist = std::uniform_int_distribution<int>(-127, 127);
  size_t dim =
      ((std::uniform_int_distribution<size_t>(1, 512))(gen) + 3) / 4 * 4;

  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("FlatBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder = std::make_shared<MultiPassIndexHolder<IndexMeta::DT_INT8>>(dim);
  uint32_t document_count =
      (std::uniform_int_distribution<uint32_t>(1, 10000))(gen);
  for (uint32_t i = 0; i < document_count; i++) {
    NumericalVector<int8_t> vec(dim);
    for (size_t j = 0; j < vec.size(); ++j) {
      vec[j] = (int8_t)dist(gen);
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  IndexMeta meta1;
  meta1.set_meta(IndexMeta::DataType::DT_INT8, dim);
  meta1.set_metric("SquaredEuclidean", 0, Params());
  meta1.set_major_order(IndexMeta::MO_ROW);  // will invalide when set
                                             // column_major_order in params

  Params params1;
  params1.set(PARAM_FLAT_COLUMN_MAJOR_ORDER,
              true);  // make it MO_COLUMN
  BuildIndex(meta1, params1, holder, INDEX_PATH + ".1");

  IndexMeta meta2;
  meta2.set_meta(IndexMeta::DataType::DT_INT8, dim);
  meta2.set_metric("SquaredEuclidean", 0, Params());
  meta2.set_major_order(IndexMeta::MO_ROW);
  BuildIndex(meta2, holder, INDEX_PATH + ".2");

  IndexSearcher::Pointer searcher1, searcher2;
  LoadIndex(INDEX_PATH + ".1", searcher1);
  LoadIndex(INDEX_PATH + ".2", searcher2);

  auto context1 = searcher1->create_context();
  auto context2 = searcher2->create_context();
  auto context3 = searcher1->create_context();
  auto context4 = searcher1->create_context();
  uint32_t topk = std::min(10u, document_count);
  context1->set_topk(topk);
  context2->set_topk(topk);
  context3->set_topk(topk);
  context3->set_filter([](uint64_t) { return false; });  // same as no filter
  context4->set_topk(topk);
  context4->set_filter([](uint64_t) { return true; });  // filter all result

  std::string query_buffer;
  uint32_t query_count = (std::uniform_int_distribution<size_t>(1, 100))(gen);

  for (uint32_t i = 0; i < query_count; i++) {
    NumericalVector<int8_t> vec(dim);
    for (uint32_t j = 0; j < vec.size(); ++j) {
      vec[j] = (int8_t)dist(gen);
    }
    query_buffer.append((const char *)vec.data(), vec.bytes());
  }

  ASSERT_EQ(0, searcher1->search_impl(query_buffer.data(),
                                      IndexQueryMeta(IndexMeta::DT_INT8, dim),
                                      query_count, context1));
  ASSERT_EQ(0, searcher1->search_impl(query_buffer.data(),
                                      IndexQueryMeta(IndexMeta::DT_INT8, dim),
                                      query_count, context3));
  ASSERT_EQ(0, searcher1->search_impl(query_buffer.data(),
                                      IndexQueryMeta(IndexMeta::DT_INT8, dim),
                                      query_count, context4));

  NumericalVector<int8_t> vec(dim);
  for (uint32_t i = 0; i < query_count; i++) {
    // not batch
    ASSERT_EQ(0, searcher2->search_impl((&query_buffer[i * vec.bytes()]),
                                        IndexQueryMeta(IndexMeta::DT_INT8, dim),
                                        context2));

    for (uint32_t j = 0; j < topk; ++j) {
      auto &result1 = context1->result(i);
      auto &result2 = context2->result();
      auto &result3 = context3->result(i);
      auto &result4 = context4->result(i);
      EXPECT_TRUE(result4.empty());

      // batch result is equal to not batch result
      EXPECT_EQ(result1[j].index(), result2[j].index());
      EXPECT_EQ(result1[j].key(), result2[j].key());
      ASSERT_FLOAT_EQ(result1[j].score(), result2[j].score());

      // test filter
      EXPECT_EQ(result1[j].index(), result3[j].index());
      EXPECT_EQ(result1[j].key(), result3[j].key());
      ASSERT_FLOAT_EQ(result1[j].score(), result3[j].score());
    }
  }
}

TEST(FlatProvider, Provider_FP32) {
  std::mt19937 gen((std::random_device())());
  auto dist = std::uniform_real_distribution<float>(0.0f, 1.0f);
  size_t dim = (std::uniform_int_distribution<size_t>(1, 512))(gen);

  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("FlatBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder = std::make_shared<MultiPassIndexHolder<IndexMeta::DT_FP32>>(dim);
  uint32_t document_count =
      (std::uniform_int_distribution<size_t>(1, 10000))(gen);

  std::vector<uint32_t> keys(document_count);
  for (uint32_t i = 0; i < document_count; i++) {
    keys[i] = i;
  }
  Shuffle(keys);

  for (uint32_t i = 0; i < document_count; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < vec.size(); ++j) {
      vec[j] = dist(gen) + static_cast<float>(i * 5);
    }
    ASSERT_TRUE(holder->emplace(keys[i], vec));
  }

  IndexMeta meta1;
  meta1.set_meta(IndexMeta::DataType::DT_FP32, dim);
  meta1.set_metric("SquaredEuclidean", 0, Params());
  meta1.set_major_order(IndexMeta::MO_ROW);
  BuildIndex(meta1, holder, INDEX_PATH + ".1");

  IndexMeta meta2;
  meta2.set_meta(IndexMeta::DataType::DT_FP32, dim);
  meta2.set_metric("SquaredEuclidean", 0, Params());
  meta2.set_major_order(IndexMeta::MO_COLUMN);
  BuildIndex(meta2, holder, INDEX_PATH + ".2");

  IndexSearcher::Pointer searcher1, searcher2;
  LoadIndex(INDEX_PATH + ".1", searcher1);
  LoadIndex(INDEX_PATH + ".2", searcher2);

  auto provider1 = searcher1->create_provider();
  auto provider2 = searcher2->create_provider();

  ASSERT_TRUE(!!provider1);
  ASSERT_TRUE(!!provider2);

  ASSERT_EQ(document_count, provider1->count());
  ASSERT_EQ(document_count, provider2->count());

  ASSERT_EQ("FlatSearcher", provider1->owner_class());
  ASSERT_EQ("FlatSearcher", provider2->owner_class());

  auto it1 = provider1->create_iterator();
  auto it2 = provider2->create_iterator();
  auto holder_it = holder->create_iterator();

  uint32_t readed_count = 0;
  while (it1->is_valid() && it2->is_valid()) {
    ASSERT_EQ(it1->key(), it2->key());
    const float *data1 = (const float *)it1->data();
    const float *data2 = (const float *)it2->data();
    const float *holder_data = (const float *)holder_it->data();
    for (size_t idx = 0; idx < dim; idx++) {
      ASSERT_EQ(*data1, *data2) << "Fail when dim is: " << dim
                                << " document_count is: " << document_count;
      ASSERT_EQ(*data1, *holder_data);
      data1++;
      data2++;
      holder_data++;
    }
    readed_count++;
    const float *features1 = (const float *)provider1->get_vector(it1->key());
    const float *features2 = (const float *)provider2->get_vector(it2->key());
    for (size_t idx = 0; idx < dim; idx++) {
      ASSERT_FLOAT_EQ(*features1, *features2);
      features1++;
      features2++;
    }
    it1->next();
    it2->next();
    holder_it->next();
  }

  ASSERT_FALSE(holder_it->is_valid());

  ASSERT_EQ(readed_count, provider1->count());
  ASSERT_EQ(readed_count, provider2->count());
  ASSERT_EQ(readed_count, holder->count());
}

TEST(FlatSearcher, TestGroup) {
  const int dim = 32;
  static std::shared_ptr<IndexMeta> index_meta_ptr_;
  index_meta_ptr_.reset(new (std::nothrow)
                            IndexMeta(IndexMeta::DataType::DT_FP32, dim));
  index_meta_ptr_->set_metric("SquaredEuclidean", 0, Params());

  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("FlatBuilder");
  ASSERT_NE(builder, nullptr);
  auto holder = std::make_shared<MultiPassIndexHolder<IndexMeta::DT_FP32>>(dim);
  size_t doc_cnt = 5000UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i / 10.0;
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  Params params;

  ASSERT_EQ(0, builder->init(*index_meta_ptr_, params));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  std::string path = INDEX_PATH + "/TestGroup";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  // test searcher
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("FlatSearcher");
  ASSERT_NE(searcher, nullptr);
  Params searcherParams;
  ASSERT_EQ(0, searcher->init(searcherParams));

  auto storage = IndexFactory::CreateStorage("MMapFileReadStorage");
  ASSERT_EQ(0, storage->open(path, false));
  ASSERT_EQ(0, searcher->load(storage, IndexMetric::Pointer()));

  auto ctx = searcher->create_context();
  ASSERT_TRUE(!!ctx);

  NumericalVector<float> vec(dim);
  IndexQueryMeta qmeta(IndexMeta::DT_FP32, dim);
  size_t group_topk = 20;
  uint64_t total_time = 0;

  auto groupbyFunc = [](uint64_t key) {
    uint32_t group_id = key / 10 % 10;

    // std::cout << "key: " << key << ", group id: " << group_id << std::endl;

    return std::string("g_") + std::to_string(group_id);
  };

  size_t group_num = 5;

  ctx->set_group_params(group_num, group_topk);
  ctx->set_group_by(groupbyFunc);

  size_t query_value = doc_cnt / 2;
  for (size_t j = 0; j < dim; ++j) {
    vec[j] = float(query_value) / 10 + 0.1f;
  }

  auto t1 = Realtime::MicroSeconds();
  ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, 1, ctx));
  auto t2 = Realtime::MicroSeconds();

  total_time += t2 - t1;
  std::cout << "Total time: " << total_time << std::endl;

  auto &group_result = ctx->group_result();

  for (uint32_t i = 0; i < group_result.size(); ++i) {
    const std::string &group_id = group_result[i].group_id();
    auto &result = group_result[i].docs();

    ASSERT_GT(result.size(), 0);
    std::cout << "Group ID: " << group_id << std::endl;

    for (uint32_t j = 0; j < result.size(); ++j) {
      std::cout << "\tKey: " << result[j].key() << std::fixed
                << std::setprecision(3) << ", Score: " << result[j].score()
                << std::endl;
    }
  }

  // do linear search by p_keys test
  auto groupbyFuncLinear = [](uint64_t key) {
    uint32_t group_id = key % 10;

    return std::string("g_") + std::to_string(group_id);
  };

  auto linear_pk_ctx = searcher->create_context();

  linear_pk_ctx->set_group_params(group_num, group_topk);
  linear_pk_ctx->set_group_by(groupbyFuncLinear);

  std::vector<std::vector<uint64_t>> p_keys;
  p_keys.resize(1);
  p_keys[0] = {4, 3, 2, 1, 5, 6, 7, 8, 9, 10};

  ASSERT_EQ(0, searcher->search_bf_by_p_keys_impl(vec.data(), p_keys, qmeta,
                                                  linear_pk_ctx));
  auto &linear_by_pkeys_group_result = linear_pk_ctx->group_result();
  ASSERT_EQ(linear_by_pkeys_group_result.size(), group_num);

  for (uint32_t i = 0; i < linear_by_pkeys_group_result.size(); ++i) {
    const std::string &group_id = linear_by_pkeys_group_result[i].group_id();
    auto &result = linear_by_pkeys_group_result[i].docs();

    ASSERT_GT(result.size(), 0);
    std::cout << "Group ID: " << group_id << std::endl;

    for (uint32_t j = 0; j < result.size(); ++j) {
      std::cout << "\tKey: " << result[j].key() << std::fixed
                << std::setprecision(3) << ", Score: " << result[j].score()
                << std::endl;
    }

    ASSERT_EQ(10 - i, result[0].key());
  }
}
