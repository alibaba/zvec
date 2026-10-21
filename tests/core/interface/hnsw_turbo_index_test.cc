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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <turbo/quantizer/quantizer.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/core/framework/index_framework.h>
#include <zvec/core/interface/index.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param_builders.h>
#include "algorithm/hnsw/hnsw_context.h"
#include "algorithm/hnsw/hnsw_params.h"
#include "algorithm/hnsw/hnsw_streamer.h"
#include "tests/test_util.h"

using namespace zvec::core_interface;

namespace {

constexpr uint32_t kDimension = 36;
constexpr size_t kVectorCount = 200;
constexpr uint32_t kTopK = 10;
constexpr size_t kGraphVectorCount =
    zvec::core::HnswEntity::kDefaultBruteForceThreshold + 1;
constexpr uint32_t kGraphQueryIds[] = {37, 101, kGraphVectorCount - 1};

using SearchRowList = std::vector<std::pair<uint32_t, float>>;

struct TurboQuantizerCase {
  QuantizerType type;
  const char *name;
  float fetch_tolerance;
  float score_tolerance;
  const char *test_name;
};

struct TurboMetricCase {
  MetricType type;
  const char *test_name;
};

constexpr TurboQuantizerCase kTurboQuantizers[] = {
    {QuantizerType::kNone, "Fp32Quantizer", 1e-6f, 1e-6f, "Fp32"},
    {QuantizerType::kFP16, "Fp16Quantizer", 1e-3f, 1e-3f, "Fp16"},
    {QuantizerType::kInt8, "Int8Quantizer", 1e-2f, 1e-2f, "Int8"},
    {QuantizerType::kInt4, "Int4Quantizer", 2e-1f, 5e-2f, "Int4"},
};
constexpr TurboMetricCase kTurboMetrics[] = {
    {MetricType::kL2sq, "L2"},
    {MetricType::kCosine, "Cosine"},
    {MetricType::kInnerProduct, "InnerProduct"},
};

class HnswTurboIndexTest
    : public testing::TestWithParam<
          std::tuple<TurboQuantizerCase, TurboMetricCase>> {
 protected:
  std::string index_path(const char *suffix) const {
    const auto &[quantizer, metric] = GetParam();
    return std::string("hnsw_turbo_") + quantizer.test_name + "_" +
           metric.test_name + "_" + suffix + ".index";
  }
};

class TestExternalVectorSource final : public zvec::core::VectorSource {
 public:
  explicit TestExternalVectorSource(
      const std::vector<std::vector<float>> *vectors)
      : vectors_(vectors) {}

  const void *get_vector(uint32_t node_id) const override {
    return (*vectors_)[node_id].data();
  }

 private:
  const std::vector<std::vector<float>> *vectors_;
};

std::vector<std::vector<float>> RandomVectors(size_t count = kVectorCount) {
  std::mt19937 gen(2026);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<std::vector<float>> vectors(count,
                                          std::vector<float>(kDimension));
  for (auto &vector : vectors) {
    float norm = 0.0f;
    for (float &value : vector) {
      value = dist(gen);
      norm += value * value;
    }
    norm = std::sqrt(norm);
    for (float &value : vector) {
      value /= norm;
    }
  }
  return vectors;
}

HNSWIndexParam::Pointer MakeParam(MetricType metric, QuantizerType quantizer,
                                  bool enable_rotate = false) {
  return HNSWIndexParamBuilder()
      .with_metric_type(metric)
      .with_data_type(DataType::DT_FP32)
      .with_dimension(kDimension)
      .with_is_sparse(false)
      .with_m(16)
      .with_ef_construction(100)
      .with_quantizer_param(QuantizerParam(quantizer, enable_rotate))
      .build();
}

HNSWIndexParam::Pointer MakeDefaultParam(MetricType metric) {
  return HNSWIndexParamBuilder()
      .with_metric_type(metric)
      .with_data_type(DataType::DT_FP32)
      .with_dimension(kDimension)
      .with_is_sparse(false)
      .with_m(16)
      .with_ef_construction(100)
      .build();
}

const char *MetricName(MetricType metric) {
  switch (metric) {
    case MetricType::kL2sq:
      return "SquaredEuclidean";
    case MetricType::kCosine:
      return "Cosine";
    case MetricType::kInnerProduct:
      return "InnerProduct";
    default:
      return "";
  }
}

SearchRowList SearchRows(Index *index, const std::vector<float> &query,
                         bool linear, bool fetch_vector = false,
                         const zvec::core::VectorSource *source = nullptr,
                         uint32_t ef_search = 100) {
  auto query_param = HNSWQueryParamBuilder()
                         .with_topk(kTopK)
                         .with_ef_search(ef_search)
                         .with_is_linear(linear)
                         .with_fetch_vector(fetch_vector)
                         .build();
  VectorData query_data{DenseVector{query.data()}};
  SearchResult result;
  EXPECT_EQ(0, source == nullptr
                   ? index->search(query_data, query_param, &result)
                   : index->search_with_source(query_data, query_param, *source,
                                               &result));
  SearchRowList rows;
  for (const auto &doc : result.doc_list_) {
    rows.emplace_back(doc.key(), doc.score());
  }
  if (fetch_vector) {
    if (source == nullptr) {
      EXPECT_EQ(rows.size(), result.reverted_vector_list_.size());
    } else {
      EXPECT_TRUE(result.reverted_vector_list_.empty());
      for (const auto &doc : result.doc_list_) {
        const auto *fetched = static_cast<const float *>(doc.vector());
        EXPECT_NE(nullptr, fetched);
        if (fetched == nullptr) {
          continue;
        }
        const auto *original =
            static_cast<const float *>(source->get_vector(doc.key()));
        for (uint32_t d = 0; d < kDimension; ++d) {
          EXPECT_FLOAT_EQ(original[d], fetched[d]);
        }
      }
    }
  }
  return rows;
}

void CheckGraphSearchEnabled(Index *index) {
  auto context = index->index_searcher()->create_context();
  auto *hnsw_context = dynamic_cast<zvec::core::HnswContext *>(context.get());
  ASSERT_NE(nullptr, hnsw_context);
  // The public interface inherits this threshold. A small data set would
  // silently run brute force even with is_linear=false.
  ASSERT_GT(index->get_doc_count(), hnsw_context->get_bruteforce_threshold());
}

SearchRowList CanonicalizeScoreTies(SearchRowList rows) {
  // Result heaps compare scores only. Canonicalize IDs within each exact tie
  // without changing the ranking of distinct scores or accepting different IDs.
  for (auto begin = rows.begin(); begin != rows.end();) {
    auto end = std::find_if(std::next(begin), rows.end(),
                            [score = begin->second](const auto &row) {
                              return row.second != score;
                            });
    std::sort(begin, end);
    begin = end;
  }
  return rows;
}

TEST(HnswTurboResultComparison, AllowsOnlyExactScoreTiePermutations) {
  const SearchRowList linear{
      {1000, 0.0f}, {729, 1.19504f}, {581, 1.19504f}, {716, 1.20319f}};
  const SearchRowList graph{
      {1000, 0.0f}, {581, 1.19504f}, {729, 1.19504f}, {716, 1.20319f}};
  EXPECT_EQ(CanonicalizeScoreTies(linear), CanonicalizeScoreTies(graph));

  auto wrong_id = graph;
  wrong_id[1].first = 582;
  EXPECT_NE(CanonicalizeScoreTies(linear), CanonicalizeScoreTies(wrong_id));
  auto wrong_score = graph;
  wrong_score[1].second = std::nextafter(wrong_score[1].second, 2.0f);
  EXPECT_NE(CanonicalizeScoreTies(linear), CanonicalizeScoreTies(wrong_score));
  auto wrong_rank = graph;
  std::swap(wrong_rank.front(), wrong_rank.back());
  EXPECT_NE(CanonicalizeScoreTies(linear), CanonicalizeScoreTies(wrong_rank));
  auto missing = graph;
  missing.pop_back();
  EXPECT_NE(CanonicalizeScoreTies(linear), CanonicalizeScoreTies(missing));
}

void CheckGraphRecall(const SearchRowList &linear_rows,
                      const SearchRowList &graph_rows) {
  ASSERT_EQ(kTopK, linear_rows.size());
  ASSERT_EQ(kTopK, graph_rows.size());
  size_t matches = 0;
  for (const auto &graph_row : graph_rows) {
    const auto match = std::find_if(
        linear_rows.begin(), linear_rows.end(),
        [key = graph_row.first](const auto &row) { return row.first == key; });
    if (match != linear_rows.end()) {
      ++matches;
      // Single-vector and batch kernels can round differently near zero,
      // where EXPECT_FLOAT_EQ's ULP comparison is too strict (e.g. cosine).
      EXPECT_NEAR(match->second, graph_row.second, 1e-6f);
    }
  }
  EXPECT_GE(matches, 9U) << "Graph search must recover at least 90% of the "
                            "linear top-10 using the same Turbo distances";
}

void AddVectors(Index *index, const std::vector<std::vector<float>> &vectors) {
  for (size_t i = 0; i < vectors.size(); ++i) {
    VectorData vector_data{DenseVector{vectors[i].data()}};
    ASSERT_EQ(0, index->add(vector_data, static_cast<uint32_t>(i)));
  }
}

void CheckTurboAddSearchReopen(MetricType metric, QuantizerType quantizer,
                               const char *quantizer_name,
                               float fetch_tolerance, const std::string &path) {
  zvec::test_util::RemoveTestFiles(path);
  auto vectors = RandomVectors(kGraphVectorCount);
  auto param = MakeParam(metric, quantizer);

  auto index = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, index);
  ASSERT_EQ(quantizer_name, index->index_searcher()->meta().quantizer_name());
  ASSERT_EQ(0, index->open(path, {StorageOptions::StorageType::kMMAP, true}));
  auto streamer = std::dynamic_pointer_cast<zvec::core::HnswStreamer>(
      index->index_searcher());
  ASSERT_NE(nullptr, streamer);
  EXPECT_TRUE(streamer->uses_turbo_distance());
  EXPECT_TRUE(streamer->uses_turbo_build_distance());
  AddVectors(index.get(), vectors);
  ASSERT_EQ(0, index->train());

  CheckGraphSearchEnabled(index.get());
  std::vector<SearchRowList> linear_results;
  std::vector<SearchRowList> graph_results;
  for (uint32_t query_id : kGraphQueryIds) {
    SCOPED_TRACE(query_id);
    auto linear_rows = SearchRows(index.get(), vectors[query_id], true, true);
    auto graph_rows = SearchRows(index.get(), vectors[query_id], false, true);
    CheckGraphRecall(linear_rows, graph_rows);
    ASSERT_FALSE(graph_rows.empty());
    EXPECT_EQ(query_id, graph_rows.front().first);
    linear_results.push_back(std::move(linear_rows));
    graph_results.push_back(std::move(graph_rows));
  }

  VectorDataBuffer fetched;
  ASSERT_EQ(0, index->fetch(37, &fetched));
  const auto *fetched_vector = reinterpret_cast<const float *>(
      std::get<DenseVectorBuffer>(fetched.vector_buffer).data.data());
  for (uint32_t i = 0; i < kDimension; ++i) {
    EXPECT_NEAR(vectors[37][i], fetched_vector[i], fetch_tolerance);
  }

  ASSERT_EQ(0, index->close());

  auto reopened = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, reopened);
  ASSERT_EQ(0,
            reopened->open(path, {StorageOptions::StorageType::kMMAP, false}));
  EXPECT_EQ(quantizer_name,
            reopened->index_searcher()->meta().quantizer_name());
  CheckGraphSearchEnabled(reopened.get());
  for (size_t i = 0; i < std::size(kGraphQueryIds); ++i) {
    SCOPED_TRACE(kGraphQueryIds[i]);
    auto linear_rows =
        SearchRows(reopened.get(), vectors[kGraphQueryIds[i]], true, true);
    auto graph_rows =
        SearchRows(reopened.get(), vectors[kGraphQueryIds[i]], false, true);
    EXPECT_EQ(linear_results[i], linear_rows);
    EXPECT_EQ(graph_results[i], graph_rows);
    CheckGraphRecall(linear_rows, graph_rows);
  }
  ASSERT_EQ(0, reopened->close());
  zvec::test_util::RemoveTestFiles(path);
}

void CheckExternalTurboAddSearchReopen(MetricType metric,
                                       QuantizerType quantizer,
                                       const char *quantizer_name,
                                       const std::string &path) {
  zvec::test_util::RemoveTestFiles(path);
  auto vectors = RandomVectors(kGraphVectorCount);
  if (metric == MetricType::kCosine) {
    for (size_t i = 0; i < vectors.size(); ++i) {
      const float scale = static_cast<float>(i % 7 + 1);
      for (float &value : vectors[i]) {
        value *= scale;
      }
    }
  }
  TestExternalVectorSource source(&vectors);
  auto param = MakeParam(metric, quantizer);
  param->use_external_vector = true;

  auto index = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, index);
  ASSERT_EQ(quantizer_name, index->index_searcher()->meta().quantizer_name());
  ASSERT_EQ(0, index->open(path, {StorageOptions::StorageType::kMMAP, true}));
  auto streamer = std::dynamic_pointer_cast<zvec::core::HnswStreamer>(
      index->index_searcher());
  ASSERT_NE(nullptr, streamer);
  EXPECT_TRUE(streamer->uses_turbo_distance());
  EXPECT_TRUE(streamer->uses_turbo_build_distance());

  for (size_t i = 0; i < vectors.size(); ++i) {
    VectorData vector_data{DenseVector{vectors[i].data()}};
    ASSERT_EQ(0, index->add_with_source(vector_data, static_cast<uint32_t>(i),
                                        source));
  }

  CheckGraphSearchEnabled(index.get());
  std::vector<SearchRowList> linear_results;
  std::vector<SearchRowList> graph_results;
  for (uint32_t query_id : kGraphQueryIds) {
    SCOPED_TRACE(query_id);
    auto linear_rows =
        SearchRows(index.get(), vectors[query_id], true, true, &source);
    auto graph_rows =
        SearchRows(index.get(), vectors[query_id], false, true, &source);
    CheckGraphRecall(linear_rows, graph_rows);
    ASSERT_FALSE(graph_rows.empty());
    EXPECT_EQ(query_id, graph_rows.front().first);
    linear_results.push_back(std::move(linear_rows));
    graph_results.push_back(std::move(graph_rows));
  }

  ASSERT_EQ(0, index->close());

  auto reopened = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, reopened);
  ASSERT_EQ(0,
            reopened->open(path, {StorageOptions::StorageType::kMMAP, false}));
  CheckGraphSearchEnabled(reopened.get());
  for (size_t i = 0; i < std::size(kGraphQueryIds); ++i) {
    SCOPED_TRACE(kGraphQueryIds[i]);
    auto linear_rows = SearchRows(reopened.get(), vectors[kGraphQueryIds[i]],
                                  true, true, &source);
    auto graph_rows = SearchRows(reopened.get(), vectors[kGraphQueryIds[i]],
                                 false, true, &source);
    EXPECT_EQ(linear_results[i], linear_rows);
    EXPECT_EQ(graph_results[i], graph_rows);
    CheckGraphRecall(linear_rows, graph_rows);
  }
  ASSERT_EQ(0, reopened->close());
  zvec::test_util::RemoveTestFiles(path);
}

void CheckOriginalProviderUsesTurbo(MetricType metric, QuantizerType quantizer,
                                    const char *quantizer_name,
                                    const std::string &path,
                                    int rabitq_bits = 7,
                                    StorageOptions::StorageType storage_type =
                                        StorageOptions::StorageType::kMMAP,
                                    bool contiguous = false) {
  zvec::test_util::RemoveTestFiles(path);
  auto vectors = RandomVectors(kGraphVectorCount);
  if (metric == MetricType::kCosine) {
    for (size_t i = 0; i < vectors.size(); ++i) {
      const float scale = static_cast<float>(i % 7 + 1);
      for (float &value : vectors[i]) {
        value *= scale;
      }
    }
  }

  auto provider = std::make_shared<zvec::core::MultiPassIndexProvider<
      zvec::core::IndexMeta::DataType::DT_FP32>>(kDimension);
  for (size_t i = 0; i < vectors.size(); ++i) {
    zvec::ailego::NumericalVector<float> vector(kDimension);
    for (uint32_t d = 0; d < kDimension; ++d) {
      vector[d] = vectors[i][d];
    }
    ASSERT_TRUE(provider->emplace(i, vector));
  }
  zvec::core::IndexMeta provider_meta(zvec::core::IndexMeta::DT_FP32,
                                      kDimension);
  provider_meta.set_metric(MetricName(metric), 0, zvec::ailego::Params{});

  auto param = MakeParam(metric, quantizer);
  if (quantizer == QuantizerType::kRabitq)
    param->quantizer_param =
        std::make_shared<RabitqQuantizerParam>(rabitq_bits);
  param->provider = provider;
  param->provider_meta = provider_meta;
  param->use_contiguous_memory = contiguous;
  auto index = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, index);
  ASSERT_EQ(quantizer_name, index->index_searcher()->meta().quantizer_name());
  ASSERT_EQ(0, index->open(path, {storage_type, true}));
  auto streamer = std::dynamic_pointer_cast<zvec::core::HnswStreamer>(
      index->index_searcher());
  ASSERT_NE(nullptr, streamer);
  EXPECT_TRUE(streamer->uses_turbo_distance());
  EXPECT_TRUE(streamer->uses_turbo_build_distance());

  AddVectors(index.get(), vectors);
  CheckGraphSearchEnabled(index.get());
  // This test checks encoding and persistence, not approximate-search recall.
  // A raw-vector graph with a random 1-bit RaBitQ model can miss quantized
  // top-10 neighbors at ef=100. Give RaBitQ enough candidates to traverse the
  // entire graph and require exact agreement except for score ties. Keep the
  // graph path active; bounded traversal and pruning have a separate test.
  const bool exhaustive_graph = quantizer == QuantizerType::kRabitq;
  const uint32_t graph_ef =
      exhaustive_graph ? static_cast<uint32_t>(vectors.size()) : 100;
  std::vector<SearchRowList> linear_results;
  std::vector<SearchRowList> graph_results;
  for (uint32_t query_id : kGraphQueryIds) {
    SCOPED_TRACE(query_id);
    auto linear_rows = SearchRows(index.get(), vectors[query_id], true);
    auto graph_rows = SearchRows(index.get(), vectors[query_id], false, false,
                                 nullptr, graph_ef);
    if (exhaustive_graph) {
      EXPECT_EQ(CanonicalizeScoreTies(linear_rows),
                CanonicalizeScoreTies(graph_rows));
    } else {
      CheckGraphRecall(linear_rows, graph_rows);
    }
    ASSERT_FALSE(graph_rows.empty());
    EXPECT_EQ(query_id, graph_rows.front().first);
    linear_results.push_back(std::move(linear_rows));
    graph_results.push_back(std::move(graph_rows));
  }

  ASSERT_EQ(0, index->close());

  if (quantizer == QuantizerType::kRabitq) param->provider.reset();
  auto reopened = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, reopened);
  ASSERT_EQ(0, reopened->open(path, {storage_type, false}));
  EXPECT_EQ(quantizer_name,
            reopened->index_searcher()->meta().quantizer_name());
  CheckGraphSearchEnabled(reopened.get());
  for (size_t i = 0; i < std::size(kGraphQueryIds); ++i) {
    SCOPED_TRACE(kGraphQueryIds[i]);
    auto linear_rows =
        SearchRows(reopened.get(), vectors[kGraphQueryIds[i]], true);
    auto graph_rows = SearchRows(reopened.get(), vectors[kGraphQueryIds[i]],
                                 false, false, nullptr, graph_ef);
    EXPECT_EQ(linear_results[i], linear_rows);
    EXPECT_EQ(graph_results[i], graph_rows);
    if (exhaustive_graph) {
      EXPECT_EQ(CanonicalizeScoreTies(linear_rows),
                CanonicalizeScoreTies(graph_rows));
    } else {
      CheckGraphRecall(linear_rows, graph_rows);
    }
  }
  ASSERT_EQ(0, reopened->close());
  zvec::test_util::RemoveTestFiles(path);
}

void BuildLegacyFp32Hnsw(const std::string &path,
                         const std::vector<std::vector<float>> &vectors) {
  namespace core = zvec::core;
  core::IndexMeta legacy_meta(core::IndexMeta::DT_FP32, kDimension);
  legacy_meta.set_meta_type(core::IndexMeta::MetaType::MT_DENSE);
  legacy_meta.set_metric("SquaredEuclidean", 0, zvec::ailego::Params());
  ASSERT_TRUE(legacy_meta.quantizer_name().empty());

  zvec::ailego::Params params;
  params.set(core::PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT, 16U);
  params.set(core::PARAM_HNSW_STREAMER_SCALING_FACTOR, 16U);
  params.set(core::PARAM_HNSW_STREAMER_EFCONSTRUCTION, 100U);
  params.set(core::PARAM_HNSW_STREAMER_EF, 100U);
  params.set(core::PARAM_HNSW_STREAMER_GET_VECTOR_ENABLE, true);
  auto streamer = core::IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_NE(nullptr, streamer);
  ASSERT_EQ(0, streamer->init(legacy_meta, params));
  auto storage = core::IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ASSERT_EQ(0, storage->init(zvec::ailego::Params()));
  ASSERT_EQ(0, storage->open(path, true));
  ASSERT_EQ(0, streamer->open(storage));

  auto context = streamer->create_context();
  core::IndexQueryMeta qmeta(core::IndexMeta::DT_FP32, kDimension);
  for (size_t i = 0; i < vectors.size(); ++i) {
    ASSERT_EQ(0, streamer->add_with_id_impl(static_cast<uint32_t>(i),
                                            vectors[i].data(), qmeta, context));
  }
  ASSERT_EQ(0, streamer->flush(0));
  ASSERT_EQ(0, streamer->close());
  ASSERT_EQ(0, storage->close());
}

void BuildLegacyInt8Hnsw(const std::string &path, MetricType metric,
                         const std::vector<std::vector<float>> &vectors) {
  namespace core = zvec::core;
  core::IndexMeta raw_meta(core::IndexMeta::DT_FP32, kDimension);
  raw_meta.set_meta_type(core::IndexMeta::MetaType::MT_DENSE);
  raw_meta.set_metric(
      metric == MetricType::kCosine ? "Cosine" : "SquaredEuclidean", 0,
      zvec::ailego::Params());
  const char *converter_name = metric == MetricType::kCosine
                                   ? "CosineInt8Converter"
                                   : "Int8StreamingConverter";
  raw_meta.set_converter(converter_name, 0, zvec::ailego::Params());
  auto converter = core::IndexFactory::CreateConverter(converter_name);
  ASSERT_NE(nullptr, converter);
  ASSERT_EQ(0, converter->init(raw_meta, zvec::ailego::Params()));

  core::IndexMeta legacy_meta = converter->meta();
  ASSERT_TRUE(legacy_meta.quantizer_name().empty());
  auto reformer =
      core::IndexFactory::CreateReformer(legacy_meta.reformer_name());
  ASSERT_NE(nullptr, reformer);
  ASSERT_EQ(0, reformer->init(legacy_meta.reformer_params()));

  zvec::ailego::Params params;
  params.set(core::PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT, 16U);
  params.set(core::PARAM_HNSW_STREAMER_SCALING_FACTOR, 16U);
  params.set(core::PARAM_HNSW_STREAMER_EFCONSTRUCTION, 100U);
  params.set(core::PARAM_HNSW_STREAMER_EF, 100U);
  params.set(core::PARAM_HNSW_STREAMER_GET_VECTOR_ENABLE, true);
  auto streamer = core::IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_NE(nullptr, streamer);
  ASSERT_EQ(0, streamer->init(legacy_meta, params));
  auto storage = core::IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ASSERT_EQ(0, storage->init(zvec::ailego::Params()));
  ASSERT_EQ(0, storage->open(path, true));
  ASSERT_EQ(0, streamer->open(storage));

  auto context = streamer->create_context();
  core::IndexQueryMeta raw_qmeta(core::IndexMeta::DT_FP32, kDimension);
  for (size_t i = 0; i < vectors.size(); ++i) {
    std::string converted;
    core::IndexQueryMeta converted_meta;
    ASSERT_EQ(0, reformer->convert(vectors[i].data(), raw_qmeta, &converted,
                                   &converted_meta));
    ASSERT_EQ(0, streamer->add_with_id_impl(static_cast<uint32_t>(i),
                                            converted.data(), converted_meta,
                                            context));
  }
  ASSERT_EQ(0, streamer->flush(0));
  ASSERT_EQ(0, streamer->close());
  ASSERT_EQ(0, storage->close());
}

}  // namespace

TEST(HnswTurboQuantizerIndex, DefaultUsesFp32TurboQuantizer) {
  for (MetricType metric :
       {MetricType::kL2sq, MetricType::kCosine, MetricType::kInnerProduct}) {
    auto index = IndexFactory::CreateAndInitIndex(*MakeDefaultParam(metric));
    ASSERT_NE(nullptr, index);
    EXPECT_EQ("Fp32Quantizer",
              index->index_searcher()->meta().quantizer_name());
  }
}

TEST(HnswTurboQuantizerIndex, InnerProductScoreAndRadiusUseCallerSpace) {
  const std::string path{"hnsw_turbo_fp32_ip_radius.index"};
  zvec::test_util::RemoveTestFiles(path);
  auto index = IndexFactory::CreateAndInitIndex(
      *MakeDefaultParam(MetricType::kInnerProduct));
  ASSERT_NE(nullptr, index);
  ASSERT_EQ("Fp32Quantizer", index->index_searcher()->meta().quantizer_name());
  ASSERT_EQ(0, index->open(path, {StorageOptions::StorageType::kMMAP, true}));

  std::vector<std::vector<float>> vectors(3, std::vector<float>(kDimension));
  vectors[0][0] = 1.0f;
  vectors[1][0] = 0.75f;
  vectors[2][0] = 0.25f;
  AddVectors(index.get(), vectors);

  auto query_param = HNSWQueryParamBuilder()
                         .with_topk(3)
                         .with_ef_search(100)
                         .with_is_linear(true)
                         .with_radius(0.5f)
                         .build();
  std::vector<float> query(kDimension);
  query[0] = 1.0f;
  SearchResult result;
  ASSERT_EQ(0, index->search(VectorData{DenseVector{query.data()}}, query_param,
                             &result));
  ASSERT_EQ(2U, result.doc_list_.size());
  EXPECT_EQ(0U, result.doc_list_[0].key());
  EXPECT_FLOAT_EQ(1.0f, result.doc_list_[0].score());
  EXPECT_EQ(1U, result.doc_list_[1].key());
  EXPECT_FLOAT_EQ(0.75f, result.doc_list_[1].score());

  ASSERT_EQ(0, index->close());
  zvec::test_util::RemoveTestFiles(path);
}

TEST_P(HnswTurboIndexTest, SelectsTurboQuantizer) {
  const auto &[quantizer, metric] = GetParam();
  auto index =
      IndexFactory::CreateAndInitIndex(*MakeParam(metric.type, quantizer.type));
  ASSERT_NE(nullptr, index);
  EXPECT_EQ(quantizer.name, index->index_searcher()->meta().quantizer_name());
}

TEST_P(HnswTurboIndexTest, KnownScoresAndRadiusUseCallerSpace) {
  const auto &[quantizer, metric] = GetParam();
  const std::string path = index_path("scores");
  zvec::test_util::RemoveTestFiles(path);
  auto index =
      IndexFactory::CreateAndInitIndex(*MakeParam(metric.type, quantizer.type));
  ASSERT_NE(nullptr, index);
  ASSERT_EQ(0, index->open(path, {StorageOptions::StorageType::kMMAP, true}));

  // Non-unit vectors distinguish cosine normalization from inner product.
  std::vector<std::vector<float>> vectors(3, std::vector<float>(kDimension));
  vectors[0][0] = 1.0f;
  vectors[1][0] = 0.75f;
  vectors[1][1] = 0.25f;
  vectors[2][0] = 0.25f;
  vectors[2][1] = 0.75f;
  AddVectors(index.get(), vectors);

  auto query_param = HNSWQueryParamBuilder()
                         .with_topk(3)
                         .with_is_linear(true)
                         .with_radius(0.5f)
                         .build();
  SearchResult result;
  ASSERT_EQ(0, index->search(VectorData{DenseVector{vectors[0].data()}},
                             query_param, &result));
  ASSERT_EQ(2U, result.doc_list_.size());
  EXPECT_EQ(0U, result.doc_list_[0].key());
  EXPECT_EQ(1U, result.doc_list_[1].key());
  float expected_first = 0.0f;
  float expected_second = 0.125f;
  if (metric.type == MetricType::kCosine) {
    expected_second = 1.0f - 0.75f / std::sqrt(0.625f);
  } else if (metric.type == MetricType::kInnerProduct) {
    expected_first = 1.0f;
    expected_second = 0.75f;
  }
  EXPECT_NEAR(expected_first, result.doc_list_[0].score(),
              quantizer.score_tolerance);
  EXPECT_NEAR(expected_second, result.doc_list_[1].score(),
              quantizer.score_tolerance);

  ASSERT_EQ(0, index->close());
  zvec::test_util::RemoveTestFiles(path);
}

TEST_P(HnswTurboIndexTest, AddSearchReopenFetch) {
  const auto &[quantizer, metric] = GetParam();
  CheckTurboAddSearchReopen(metric.type, quantizer.type, quantizer.name,
                            quantizer.fetch_tolerance, index_path("stored"));
}

TEST_P(HnswTurboIndexTest, ExternalVectorsUseTurbo) {
  const auto &[quantizer, metric] = GetParam();
  CheckExternalTurboAddSearchReopen(metric.type, quantizer.type, quantizer.name,
                                    index_path("external"));
}

TEST_P(HnswTurboIndexTest, OriginalProviderBuildUsesTurbo) {
  const auto &[quantizer, metric] = GetParam();
  CheckOriginalProviderUsesTurbo(metric.type, quantizer.type, quantizer.name,
                                 index_path("provider"));
}

TEST(HnswTurboQuantizerIndex, UnsupportedCombinationsUseLegacyPipeline) {
  auto rotated = IndexFactory::CreateAndInitIndex(
      *MakeParam(MetricType::kCosine, QuantizerType::kInt8, true));
  ASSERT_NE(nullptr, rotated);
  EXPECT_TRUE(rotated->index_searcher()->meta().quantizer_name().empty());

  auto mips = IndexFactory::CreateAndInitIndex(
      *MakeParam(MetricType::kMIPSL2sq, QuantizerType::kNone));
  ASSERT_NE(nullptr, mips);
  EXPECT_TRUE(mips->index_searcher()->meta().quantizer_name().empty());
}

TEST_P(HnswTurboIndexTest, MergePreservesTurboLayout) {
  const auto &[quantizer, metric] = GetParam();
  const std::string source_path = index_path("merge_source");
  const std::string target_path = index_path("merge_target");
  zvec::test_util::RemoveTestFiles(source_path);
  zvec::test_util::RemoveTestFiles(target_path);
  auto vectors = RandomVectors();
  auto param = MakeParam(metric.type, quantizer.type);

  auto source = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, source);
  ASSERT_EQ(
      0, source->open(source_path, {StorageOptions::StorageType::kMMAP, true}));
  AddVectors(source.get(), vectors);

  auto target = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, target);
  ASSERT_EQ(
      0, target->open(target_path, {StorageOptions::StorageType::kMMAP, true}));
  ASSERT_EQ(0, target->merge({source}, IndexFilter()));
  EXPECT_EQ(kVectorCount, target->get_doc_count());
  EXPECT_EQ(quantizer.name, target->index_searcher()->meta().quantizer_name());

  auto rows = SearchRows(target.get(), vectors[73], true);
  ASSERT_EQ(kTopK, rows.size());
  EXPECT_EQ(73U, rows[0].first);

  ASSERT_EQ(0, target->close());
  ASSERT_EQ(0, source->close());
  zvec::test_util::RemoveTestFiles(source_path);
  zvec::test_util::RemoveTestFiles(target_path);
}

TEST_P(HnswTurboIndexTest, MergeAcrossMetricsReencodesTurboLayout) {
  const auto &[quantizer, source_metric] = GetParam();
  auto vectors = RandomVectors(kTopK);
  // Exactly representable in INT4: copying the IP tail into an L2 index
  // used to produce a negative self-distance for this vector.
  vectors[0].assign(kDimension, 0.0f);
  vectors[0].front() = vectors[0].back() = 1.0f;
  for (size_t i = 1; i < vectors.size(); ++i) {
    for (float &value : vectors[i]) {
      value *= static_cast<float>(i + 1);
    }
  }

  auto make_param = [quantizer_type = quantizer.type](
                        bool flat,
                        MetricType metric) -> BaseIndexParam::Pointer {
    if (!flat) {
      return MakeParam(metric, quantizer_type);
    }
    return FlatIndexParamBuilder()
        .with_metric_type(metric)
        .with_data_type(DataType::DT_FP32)
        .with_dimension(kDimension)
        .with_is_sparse(false)
        .with_quantizer_param(QuantizerParam(quantizer_type))
        .build();
  };

  for (bool source_flat : {false, true}) {
    const std::string source_path = index_path("cross_metric_source");
    zvec::test_util::RemoveTestFiles(source_path);
    auto source = IndexFactory::CreateAndInitIndex(
        *make_param(source_flat, source_metric.type));
    ASSERT_NE(nullptr, source);
    ASSERT_EQ(0, source->open(source_path,
                              {StorageOptions::StorageType::kMMAP, true}));
    AddVectors(source.get(), vectors);

    // A lossy source cannot recover the initial FP32 values exactly. Build
    // the reference from fetched (decoded) values, then encode for the target.
    auto decoded = vectors;
    for (uint32_t i = 0; i < vectors.size(); ++i) {
      VectorDataBuffer fetched;
      ASSERT_EQ(0, source->fetch(i, &fetched));
      const auto &bytes =
          std::get<DenseVectorBuffer>(fetched.vector_buffer).data;
      ASSERT_EQ(kDimension * sizeof(float), bytes.size());
      std::memcpy(decoded[i].data(), bytes.data(), bytes.size());
    }

    for (const auto &target_metric : kTurboMetrics) {
      if (target_metric.type == source_metric.type) {
        continue;
      }
      for (bool target_flat : {false, true}) {
        SCOPED_TRACE(testing::Message()
                     << "source_flat=" << source_flat
                     << " target_flat=" << target_flat
                     << " target_metric=" << target_metric.test_name);
        const std::string target_path = index_path("cross_metric_target");
        const std::string reference_path = index_path("cross_metric_reference");
        zvec::test_util::RemoveTestFiles(target_path);
        zvec::test_util::RemoveTestFiles(reference_path);
        auto param = make_param(target_flat, target_metric.type);
        auto target = IndexFactory::CreateAndInitIndex(*param);
        auto reference = IndexFactory::CreateAndInitIndex(*param);
        ASSERT_NE(nullptr, target);
        ASSERT_NE(nullptr, reference);
        ASSERT_EQ(0, target->open(target_path,
                                  {StorageOptions::StorageType::kMMAP, true}));
        ASSERT_EQ(0,
                  reference->open(reference_path,
                                  {StorageOptions::StorageType::kMMAP, true}));
        AddVectors(reference.get(), decoded);
        ASSERT_EQ(0, target->merge({source}, IndexFilter()));
        EXPECT_EQ(vectors.size(), target->get_doc_count());

        auto search = [&](Index *index, const std::vector<float> &query) {
          BaseIndexQueryParam::Pointer query_param;
          if (target_flat) {
            query_param = FlatQueryParamBuilder().with_topk(kTopK).build();
          } else {
            query_param = HNSWQueryParamBuilder()
                              .with_topk(kTopK)
                              .with_is_linear(true)
                              .build();
          }
          SearchResult result;
          EXPECT_EQ(0, index->search(VectorData{DenseVector{query.data()}},
                                     query_param, &result));
          SearchRowList rows;
          for (const auto &doc : result.doc_list_) {
            rows.emplace_back(doc.key(), doc.score());
          }
          std::sort(rows.begin(), rows.end());
          return rows;
        };
        for (const auto &query : vectors) {
          const auto expected = search(reference.get(), query);
          const auto actual = search(target.get(), query);
          ASSERT_EQ(kTopK, expected.size());
          ASSERT_EQ(expected.size(), actual.size());
          for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(expected[i].first, actual[i].first);
            EXPECT_FLOAT_EQ(expected[i].second, actual[i].second);
          }
        }
        ASSERT_EQ(0, target->close());
        ASSERT_EQ(0, reference->close());
        zvec::test_util::RemoveTestFiles(target_path);
        zvec::test_util::RemoveTestFiles(reference_path);
      }
    }
    ASSERT_EQ(0, source->close());
    zvec::test_util::RemoveTestFiles(source_path);
  }
}

INSTANTIATE_TEST_SUITE_P(
    QuantizersAndMetrics, HnswTurboIndexTest,
    testing::Combine(testing::ValuesIn(kTurboQuantizers),
                     testing::ValuesIn(kTurboMetrics)),
    [](const testing::TestParamInfo<HnswTurboIndexTest::ParamType> &info) {
      return std::string(std::get<0>(info.param).test_name) + "_" +
             std::get<1>(info.param).test_name;
    });

class HnswLegacyReopenTest : public testing::TestWithParam<StorageOptions> {
 protected:
  static void SetUpTestSuite() {
    ASSERT_EQ(0, zvec::ailego::MemoryLimitPool::get_instance().init(100 * 1024 *
                                                                    1024));
  }
};

INSTANTIATE_TEST_SUITE_P(
    StorageModes, HnswLegacyReopenTest,
    testing::Values(
        StorageOptions{StorageOptions::StorageType::kMMAP, false},
        StorageOptions{StorageOptions::StorageType::kMMAP, false, false, true},
        StorageOptions{StorageOptions::StorageType::kBufferPool, false}));

TEST_P(HnswLegacyReopenTest, LegacyLayoutReopenFallsBack) {
  const std::string path{"hnsw_int8_legacy_layout.index"};
  zvec::test_util::RemoveTestFiles(path);
  auto vectors = RandomVectors();
  BuildLegacyInt8Hnsw(path, MetricType::kL2sq, vectors);
  if (::testing::Test::HasFatalFailure()) {
    return;
  }

  auto index = IndexFactory::CreateAndInitIndex(
      *MakeParam(MetricType::kL2sq, QuantizerType::kInt8));
  ASSERT_NE(nullptr, index);
  ASSERT_EQ(0, index->open(path, GetParam()));
  EXPECT_TRUE(index->index_searcher()->meta().quantizer_name().empty());
  auto rows = SearchRows(index.get(), vectors[37], true);
  ASSERT_EQ(kTopK, rows.size());
  EXPECT_EQ(37U, rows[0].first);

  VectorDataBuffer fetched;
  ASSERT_EQ(0, index->fetch(37, &fetched));
  const auto *fetched_vector = reinterpret_cast<const float *>(
      std::get<DenseVectorBuffer>(fetched.vector_buffer).data.data());
  for (uint32_t i = 0; i < kDimension; ++i) {
    EXPECT_NEAR(vectors[37][i], fetched_vector[i], 5e-2f);
  }
  ASSERT_EQ(0, index->close());
  zvec::test_util::RemoveTestFiles(path);
}

TEST_P(HnswLegacyReopenTest, LegacyFp32LayoutReopenFallsBack) {
  const std::string path{"hnsw_fp32_legacy_layout.index"};
  zvec::test_util::RemoveTestFiles(path);
  auto vectors = RandomVectors();
  BuildLegacyFp32Hnsw(path, vectors);
  if (::testing::Test::HasFatalFailure()) {
    return;
  }

  auto index = IndexFactory::CreateAndInitIndex(
      *MakeParam(MetricType::kL2sq, QuantizerType::kNone));
  ASSERT_NE(nullptr, index);
  ASSERT_EQ(0, index->open(path, GetParam()));
  EXPECT_TRUE(index->index_searcher()->meta().quantizer_name().empty());
  auto rows = SearchRows(index.get(), vectors[37], true);
  ASSERT_EQ(kTopK, rows.size());
  EXPECT_EQ(37U, rows[0].first);

  VectorDataBuffer fetched;
  ASSERT_EQ(0, index->fetch(37, &fetched));
  const auto *fetched_vector = reinterpret_cast<const float *>(
      std::get<DenseVectorBuffer>(fetched.vector_buffer).data.data());
  for (uint32_t i = 0; i < kDimension; ++i) {
    EXPECT_FLOAT_EQ(vectors[37][i], fetched_vector[i]);
  }
  ASSERT_EQ(0, index->close());
  zvec::test_util::RemoveTestFiles(path);
}

namespace zvec {
namespace core {
namespace {

constexpr size_t kDimension = 8;
constexpr size_t kCount = 16;

class ExternalVectorSource final : public VectorSource {
 public:
  ExternalVectorSource() : vectors(kCount, std::vector<float>(kDimension)) {
    for (size_t i = 0; i < kCount; ++i) {
      for (size_t j = 0; j < kDimension; ++j) {
        vectors[i][j] = i * 0.125f + j * 0.03125f;
      }
    }
  }

  const void *get_vector(uint32_t node_id) const override {
    return vectors[node_id].data();
  }

  std::vector<std::vector<float>> vectors;
};

class HnswExternalCoreCompatibilityTest : public testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    zvec::test_util::RemoveTestPath(directory_);
  }

  void TearDown() override {
    if (streamer_) {
      streamer_->close();
    }
    if (storage_) {
      storage_->close();
    }
    zvec::test_util::RemoveTestPath(directory_);
  }

  void open(bool turbo) {
    IndexMeta meta(IndexMeta::DataType::DT_FP32, kDimension);
    meta.set_metric("SquaredEuclidean", 0, ailego::Params());
    if (turbo) {
      quantizer_ = IndexFactory::CreateQuantizer("Int8Quantizer");
      ASSERT_NE(nullptr, quantizer_);
      ASSERT_EQ(0, quantizer_->init(meta, ailego::Params()));
      meta = quantizer_->meta();
    }
    ailego::Params params;
    params.set(PARAM_HNSW_STREAMER_USE_EXTERNAL_VECTOR, true);
    params.set(PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT, 16U);
    params.set(PARAM_HNSW_STREAMER_EFCONSTRUCTION, 100U);
    params.set(PARAM_HNSW_STREAMER_EF, 100U);
    params.set(PARAM_HNSW_STREAMER_BRUTE_FORCE_THRESHOLD, 0U);
    streamer_ = IndexFactory::CreateStreamer("HnswStreamer");
    ASSERT_NE(nullptr, streamer_);
    ASSERT_EQ(0, turbo ? streamer_->init(meta, params, quantizer_)
                       : streamer_->init(meta, params));
    storage_ = IndexFactory::CreateStorage("MMapFileStorage");
    ASSERT_NE(nullptr, storage_);
    ASSERT_EQ(0, storage_->init(ailego::Params()));
    ASSERT_EQ(0, storage_->open(directory_ + "external.index", true));
    ASSERT_EQ(0, streamer_->open(storage_));
    context_ = streamer_->create_context();
    ASSERT_NE(nullptr, context_);
    ctx_ = dynamic_cast<HnswContext *>(context_.get());
    ASSERT_NE(nullptr, ctx_);
    ctx_->set_vector_source(&source_);
  }

  int add(uint32_t id, const void *query, const IndexQueryMeta &meta) {
    return GetParam() ? streamer_->add_with_id_impl(id, query, meta, context_)
                      : streamer_->add_impl(id, query, meta, context_);
  }

  void check_node_count(size_t expected) const {
    // Context entities retain their creation-time header. A fresh provider
    // snapshots the streamer's current entity, including any orphan nodes.
    auto provider = streamer_->create_provider();
    ASSERT_NE(nullptr, provider);
    ASSERT_EQ(expected, provider->count());
  }

  const std::string directory_ = "hnsw_streamer_turbo_compat_test_dir/";
  IndexStreamer::Pointer streamer_;
  IndexStorage::Pointer storage_;
  IndexStreamer::Context::Pointer context_;
  HnswContext *ctx_ = nullptr;
  turbo::Quantizer::Pointer quantizer_;
  ExternalVectorSource source_;
};

TEST_P(HnswExternalCoreCompatibilityTest,
       LegacyBuildAcceptsQueryWithoutExternalBuildField) {
  ASSERT_NO_FATAL_FAILURE(open(false));
  IndexQueryMeta meta(IndexMeta::DataType::DT_FP32, kDimension);
  for (uint32_t i = 0; i < kCount; ++i) {
    ASSERT_EQ(nullptr, ctx_->external_build_query());
    ASSERT_EQ(0, add(i, source_.get_vector(i), meta));
  }
  ASSERT_NO_FATAL_FAILURE(check_node_count(kCount));
  EXPECT_EQ(kCount, streamer_->stats().added_count());

  context_->set_topk(1);
  for (uint32_t probe : {0U, 7U, 15U}) {
    ASSERT_EQ(
        0, streamer_->search_impl(source_.get_vector(probe), meta, context_));
    ASSERT_EQ(1U, context_->result().size());
    EXPECT_EQ(probe, context_->result()[0].key());
    EXPECT_FLOAT_EQ(0.0f, context_->result()[0].score());
  }
}

TEST_P(HnswExternalCoreCompatibilityTest,
       TurboRejectsMissingRawQueryBeforeMutatingNodes) {
  ASSERT_NO_FATAL_FAILURE(open(true));
  IndexQueryMeta raw_meta(IndexMeta::DataType::DT_FP32, kDimension);
  IndexQueryMeta encoded_meta;
  std::vector<std::string> codes(kCount);
  for (uint32_t i = 0; i < kCount; ++i) {
    ASSERT_EQ(0, quantizer_->quantize(source_.get_vector(i), raw_meta,
                                      &codes[i], &encoded_meta));
    ctx_->set_external_build_query(nullptr);
    ASSERT_EQ(IndexError_InvalidArgument,
              add(i, codes[i].data(), encoded_meta));
    ASSERT_NO_FATAL_FAILURE(check_node_count(i));
    EXPECT_EQ(i, streamer_->stats().added_count());
    ctx_->set_external_build_query(source_.get_vector(i));
    ASSERT_EQ(0, add(i, codes[i].data(), encoded_meta));
    ASSERT_NO_FATAL_FAILURE(check_node_count(i + 1));
  }
  EXPECT_EQ(kCount, streamer_->stats().added_count());
  EXPECT_EQ(kCount, streamer_->stats().discarded_count());

  context_->set_topk(1);
  for (uint32_t probe : {0U, 7U, 15U}) {
    ASSERT_EQ(
        0, streamer_->search_impl(codes[probe].data(), encoded_meta, context_));
    ASSERT_EQ(1U, context_->result().size());
    EXPECT_EQ(probe, context_->result()[0].key());
  }
}

INSTANTIATE_TEST_SUITE_P(AddApis, HnswExternalCoreCompatibilityTest,
                         testing::Bool());

}  // namespace
}  // namespace core
}  // namespace zvec

#if RABITQ_SUPPORTED
TEST(HnswTurboRabitq, BuildsFromOriginalAndReopensWithoutProvider) {
  for (auto metric :
       {MetricType::kL2sq, MetricType::kCosine, MetricType::kInnerProduct}) {
    for (int bits : {1, 4, 7, 8, 9}) {
      SCOPED_TRACE(static_cast<int>(metric));
      SCOPED_TRACE(bits);
      CheckOriginalProviderUsesTurbo(metric, QuantizerType::kRabitq,
                                     "RabitqQuantizer",
                                     "hnsw_turbo_rabitq.index", bits);
    }
  }
}

TEST(HnswTurboRabitq,
     RequiresOriginalVectorsForInsertAndRejectsUnsupportedLayouts) {
  const std::string path = "hnsw_turbo_rabitq_missing_provider.index";
  zvec::test_util::RemoveTestFiles(path);
  auto param = MakeParam(MetricType::kL2sq, QuantizerType::kRabitq);
  auto index = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, index);
  ASSERT_EQ(0, index->open(path, {StorageOptions::StorageType::kMMAP, true}));
  auto vectors = RandomVectors(1);
  VectorData vector;
  vector.vector = DenseVector{vectors[0].data()};
  EXPECT_NE(0, index->add(vector, 0));
  ASSERT_EQ(0, index->close());
  zvec::test_util::RemoveTestFiles(path);
  param->use_external_vector = true;
  EXPECT_EQ(nullptr, IndexFactory::CreateAndInitIndex(*param));
  param->use_external_vector = false;
  param->quantizer_param = std::make_shared<RabitqQuantizerParam>(10);
  EXPECT_EQ(nullptr, IndexFactory::CreateAndInitIndex(*param));
}

TEST(HnswTurboRabitq, BufferPoolAndContiguousStorage) {
  ASSERT_EQ(
      0, zvec::ailego::MemoryLimitPool::get_instance().init(100 * 1024 * 1024));
  CheckOriginalProviderUsesTurbo(MetricType::kL2sq, QuantizerType::kRabitq,
                                 "RabitqQuantizer",
                                 "hnsw_turbo_rabitq_buffer.index", 7,
                                 StorageOptions::StorageType::kBufferPool);
  CheckOriginalProviderUsesTurbo(MetricType::kL2sq, QuantizerType::kRabitq,
                                 "RabitqQuantizer",
                                 "hnsw_turbo_rabitq_contiguous.index", 7,
                                 StorageOptions::StorageType::kMMAP, true);
}

TEST(HnswTurboRabitq, RejectsCorruptPersistedRotation) {
  const std::string path = "hnsw_turbo_rabitq_corrupt.index";
  zvec::test_util::RemoveTestFiles(path);
  auto param = MakeParam(MetricType::kL2sq, QuantizerType::kRabitq);
  auto index = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, index);
  ASSERT_EQ(0, index->open(path, {StorageOptions::StorageType::kMMAP, true}));
  ASSERT_EQ(0, index->close());
  auto storage = zvec::core::IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init(zvec::ailego::Params{}));
  ASSERT_EQ(0, storage->open(path, false));
  auto segment = storage->get("hnsw.quantizer");
  ASSERT_NE(nullptr, segment);
  const char bad_magic = 0;
  ASSERT_EQ(1u, segment->write(0, &bad_magic, 1));
  segment.reset();
  ASSERT_EQ(0, storage->close());
  auto reopened = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, reopened);
  EXPECT_NE(0,
            reopened->open(path, {StorageOptions::StorageType::kMMAP, false}));
  reopened.reset();
  zvec::test_util::RemoveTestFiles(path);
}

namespace {
class InspectableRabitqHnsw : public HNSWIndex {
 public:
  using Index::init;
  // Keep the context statistics available after search; the public wrapper
  // normally resets the thread-local context after collecting its results.
  int search(const VectorData &query, const BaseIndexQueryParam::Pointer &param,
             SearchResult *result) override {
    auto &ctx = acquire_context();
    ctx->reset();
    int ret = _prepare_for_search(query, param, ctx);
    if (ret != 0) return ret;
    return _dense_search(query, param, result, ctx);
  }
  zvec::core::HnswContext *context() {
    return dynamic_cast<zvec::core::HnswContext *>(acquire_context().get());
  }
  std::string quantizer_state() {
    std::string state;
    EXPECT_EQ(0, turbo_quantizer_->serialize(&state));
    return state;
  }
};
}  // namespace

TEST(HnswTurboRabitq, ScreensDuringTraversalAndClearsRefinementForBuild) {
  const std::string path = "hnsw_turbo_rabitq_stages.index";
  zvec::test_util::RemoveTestFiles(path);
  auto vectors = RandomVectors(kGraphVectorCount);
  auto provider = std::make_shared<
      zvec::core::MultiPassIndexProvider<zvec::core::IndexMeta::DT_FP32>>(
      kDimension);
  for (size_t i = 0; i < vectors.size(); ++i) {
    zvec::ailego::NumericalVector<float> value(kDimension);
    std::copy(vectors[i].begin(), vectors[i].end(), value.begin());
    ASSERT_TRUE(provider->emplace(i, value));
  }
  auto param = MakeParam(MetricType::kL2sq, QuantizerType::kRabitq);
  param->provider = provider;
  param->provider_meta =
      zvec::core::IndexMeta(zvec::core::IndexMeta::DT_FP32, kDimension);
  param->provider_meta.set_metric("SquaredEuclidean", 0,
                                  zvec::ailego::Params{});
  InspectableRabitqHnsw index;
  ASSERT_EQ(0, index.init(*param));
  ASSERT_EQ(0, index.open(path, {StorageOptions::StorageType::kMMAP, true}));
  const auto trained_state = index.quantizer_state();
  AddVectors(&index, vectors);
  auto linear = SearchRows(&index, vectors[37], true);
  auto graph = SearchRows(&index, vectors[37], false);
  CheckGraphRecall(linear, graph);
  auto *ctx = index.context();
  ASSERT_NE(nullptr, ctx);
  EXPECT_TRUE(ctx->dist_calculator().has_refinement());
  EXPECT_GT(ctx->dist_calculator().refinement_count(), 0u);
  EXPECT_GT(ctx->dist_calculator().bound_pruned_count(), 0u);
  EXPECT_GT(ctx->dist_calculator().estimate_count(),
            ctx->dist_calculator().refinement_count());

  // Alternating search and construction must never evaluate raw FP32 vectors
  // through a stale RaBitQ coarse-distance callback.
  const uint32_t id = vectors.size();
  zvec::ailego::NumericalVector<float> extra(kDimension);
  std::copy(vectors[37].begin(), vectors[37].end(), extra.begin());
  ASSERT_TRUE(provider->emplace(id, extra));
  ASSERT_EQ(0, index.add(VectorData{DenseVector{extra.data()}}, id));
  EXPECT_FALSE(index.context()->dist_calculator().has_refinement());
  EXPECT_EQ(trained_state, index.quantizer_state());

  auto query_param =
      HNSWQueryParamBuilder().with_topk(kTopK).with_ef_search(100).build();
  query_param->filter = std::make_shared<IndexFilter>();
  query_param->filter->set([](uint64_t key) { return key % 2 == 0; });
  SearchResult filtered;
  ASSERT_EQ(0, index.search(VectorData{DenseVector{vectors[37].data()}},
                            query_param, &filtered));
  EXPECT_EQ(kTopK, filtered.doc_list_.size());
  for (const auto &doc : filtered.doc_list_) EXPECT_EQ(1u, doc.key() % 2);
  EXPECT_TRUE(index.context()->dist_calculator().has_refinement());
  EXPECT_GT(index.context()->dist_calculator().refinement_count(), 0u);
  query_param->filter.reset();
  query_param->group_by_param = std::make_shared<GroupByParam>();
  query_param->group_by_param->group_count = 3;
  query_param->group_by_param->group_topk = 2;
  query_param->group_by_param->group_by = [](uint64_t key) {
    return std::to_string(key % 3);
  };
  SearchResult grouped;
  ASSERT_EQ(0, index.search(VectorData{DenseVector{vectors[37].data()}},
                            query_param, &grouped));
  EXPECT_EQ(3u, grouped.group_doc_list_.size());
  for (const auto &group : grouped.group_doc_list_) {
    EXPECT_EQ(2u, group.docs().size());
    for (const auto &doc : group.docs())
      EXPECT_EQ(group.group_id(), std::to_string(doc.key() % 3));
  }
  ASSERT_EQ(0, index.close());

  // A provider on reopen must not silently retrain and reinterpret old codes.
  InspectableRabitqHnsw reopened;
  ASSERT_EQ(0, reopened.init(*param));
  ASSERT_EQ(0,
            reopened.open(path, {StorageOptions::StorageType::kMMAP, false}));
  EXPECT_EQ(trained_state, reopened.quantizer_state());
  auto rows = SearchRows(&reopened, vectors[37], false);
  ASSERT_FALSE(rows.empty());
  ASSERT_EQ(0, reopened.close());
  zvec::test_util::RemoveTestFiles(path);
}

#else
TEST(HnswTurboRabitq, RejectsUnsupportedPlatform) {
  class TestHnswIndex : public HNSWIndex {
   public:
    using HNSWIndex::create_and_init_converter_reformer;
    using Index::init;
  };
  for (auto metric :
       {MetricType::kL2sq, MetricType::kCosine, MetricType::kInnerProduct}) {
    for (int bits : {1, 7, 9}) {
      auto param = MakeParam(metric, QuantizerType::kRabitq);
      param->quantizer_param = std::make_shared<RabitqQuantizerParam>(bits);
      TestHnswIndex index;
      EXPECT_EQ(zvec::core::IndexError_Unsupported,
                index.create_and_init_converter_reformer(
                    *param->quantizer_param, *param));
      EXPECT_NE(0, index.init(*param));
      EXPECT_EQ(nullptr, IndexFactory::CreateAndInitIndex(*param));
    }
  }
  // Unsupported builds must not register a usable RaBitQ backend.
  EXPECT_EQ(nullptr,
            zvec::core::IndexFactory::CreateQuantizer("RabitqQuantizer"));
}
#endif  // RABITQ_SUPPORTED
