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
#include <iterator>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <turbo/quantizer/quantizer.h>
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
  std::string IndexPath(const char *suffix) const {
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
                         const zvec::core::VectorSource *source = nullptr) {
  auto query_param = HNSWQueryParamBuilder()
                         .with_topk(kTopK)
                         .with_ef_search(100)
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
      EXPECT_FLOAT_EQ(match->second, graph_row.second);
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
  param->provider = provider;
  param->provider_meta = provider_meta;
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
  CheckGraphSearchEnabled(index.get());
  std::vector<SearchRowList> linear_results;
  std::vector<SearchRowList> graph_results;
  for (uint32_t query_id : kGraphQueryIds) {
    SCOPED_TRACE(query_id);
    auto linear_rows = SearchRows(index.get(), vectors[query_id], true);
    auto graph_rows = SearchRows(index.get(), vectors[query_id], false);
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
  EXPECT_EQ(quantizer_name,
            reopened->index_searcher()->meta().quantizer_name());
  CheckGraphSearchEnabled(reopened.get());
  for (size_t i = 0; i < std::size(kGraphQueryIds); ++i) {
    SCOPED_TRACE(kGraphQueryIds[i]);
    auto linear_rows =
        SearchRows(reopened.get(), vectors[kGraphQueryIds[i]], true);
    auto graph_rows =
        SearchRows(reopened.get(), vectors[kGraphQueryIds[i]], false);
    EXPECT_EQ(linear_results[i], linear_rows);
    EXPECT_EQ(graph_results[i], graph_rows);
    CheckGraphRecall(linear_rows, graph_rows);
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
  const std::string path = IndexPath("scores");
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
                            quantizer.fetch_tolerance, IndexPath("stored"));
}

TEST_P(HnswTurboIndexTest, ExternalVectorsUseTurbo) {
  const auto &[quantizer, metric] = GetParam();
  CheckExternalTurboAddSearchReopen(metric.type, quantizer.type, quantizer.name,
                                    IndexPath("external"));
}

TEST_P(HnswTurboIndexTest, OriginalProviderBuildUsesTurbo) {
  const auto &[quantizer, metric] = GetParam();
  CheckOriginalProviderUsesTurbo(metric.type, quantizer.type, quantizer.name,
                                 IndexPath("provider"));
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
  const std::string source_path = IndexPath("merge_source");
  const std::string target_path = IndexPath("merge_target");
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

INSTANTIATE_TEST_SUITE_P(
    QuantizersAndMetrics, HnswTurboIndexTest,
    testing::Combine(testing::ValuesIn(kTurboQuantizers),
                     testing::ValuesIn(kTurboMetrics)),
    [](const testing::TestParamInfo<HnswTurboIndexTest::ParamType> &info) {
      return std::string(std::get<0>(info.param).test_name) + "_" +
             std::get<1>(info.param).test_name;
    });

TEST(HnswTurboQuantizerIndex, LegacyLayoutReopenFallsBack) {
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
  ASSERT_EQ(0, index->open(path, {StorageOptions::StorageType::kMMAP, false}));
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

TEST(HnswTurboQuantizerIndex, LegacyFp32LayoutReopenFallsBack) {
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
  ASSERT_EQ(0, index->open(path, {StorageOptions::StorageType::kMMAP, false}));
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

  void Open(bool turbo) {
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

  int Add(uint32_t id, const void *query, const IndexQueryMeta &meta) {
    return GetParam() ? streamer_->add_with_id_impl(id, query, meta, context_)
                      : streamer_->add_impl(id, query, meta, context_);
  }

  void CheckNodeCount(size_t expected) const {
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
  ASSERT_NO_FATAL_FAILURE(Open(false));
  IndexQueryMeta meta(IndexMeta::DataType::DT_FP32, kDimension);
  for (uint32_t i = 0; i < kCount; ++i) {
    ASSERT_EQ(nullptr, ctx_->external_build_query());
    ASSERT_EQ(0, Add(i, source_.get_vector(i), meta));
  }
  ASSERT_NO_FATAL_FAILURE(CheckNodeCount(kCount));
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
  ASSERT_NO_FATAL_FAILURE(Open(true));
  IndexQueryMeta raw_meta(IndexMeta::DataType::DT_FP32, kDimension);
  IndexQueryMeta encoded_meta;
  std::vector<std::string> codes(kCount);
  for (uint32_t i = 0; i < kCount; ++i) {
    ASSERT_EQ(0, quantizer_->quantize(source_.get_vector(i), raw_meta,
                                      &codes[i], &encoded_meta));
    ctx_->set_external_build_query(nullptr);
    ASSERT_EQ(IndexError_InvalidArgument,
              Add(i, codes[i].data(), encoded_meta));
    ASSERT_NO_FATAL_FAILURE(CheckNodeCount(i));
    EXPECT_EQ(i, streamer_->stats().added_count());
    ctx_->set_external_build_query(source_.get_vector(i));
    ASSERT_EQ(0, Add(i, codes[i].data(), encoded_meta));
    ASSERT_NO_FATAL_FAILURE(CheckNodeCount(i + 1));
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
