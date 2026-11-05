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
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <new>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <turbo/quantizer/fp32_quantizer/fp32_quantizer.h>
#include <turbo/quantizer/quantizer.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/parallel/thread_pool.h>
#include <zvec/core/framework/index_framework.h>
#include <zvec/core/interface/index.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param_builders.h>
#include "algorithm/hnsw/hnsw_build_dumper.h"
#include "algorithm/hnsw/hnsw_context.h"
#include "algorithm/hnsw/hnsw_params.h"
#include "algorithm/hnsw/hnsw_streamer.h"
#include "tests/test_util.h"
#include "utility/stable_vector_holder.h"

// Fault injection is confined to allocations on the test's calling thread.
// Other tests and worker allocations use the normal malloc/free path.
namespace bulk_alloc_test {
thread_local size_t countdown = 0;
thread_local void (*before_failure)() = nullptr;
thread_local void (*on_delete)(void *) = nullptr;
thread_local void *state = nullptr;

void Reset() {
  countdown = 0;
  before_failure = nullptr;
  on_delete = nullptr;
  state = nullptr;
}
}  // namespace bulk_alloc_test

void *operator new(std::size_t size) {
  if (bulk_alloc_test::countdown && --bulk_alloc_test::countdown == 0) {
    if (bulk_alloc_test::before_failure) bulk_alloc_test::before_failure();
    throw std::bad_alloc();
  }
  if (void *ptr = std::malloc(size ? size : 1)) return ptr;
  throw std::bad_alloc();
}

void operator delete(void *ptr) noexcept {
  if (ptr && bulk_alloc_test::on_delete) bulk_alloc_test::on_delete(ptr);
  std::free(ptr);
}

void operator delete(void *ptr, std::size_t) noexcept {
  ::operator delete(ptr);
}

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

class TestContiguousVectorSource final
    : public zvec::core::ContiguousVectorSource {
 public:
  TestContiguousVectorSource(const std::vector<std::vector<float>> &vectors,
                             size_t padding)
      : stride_elements_(kDimension + padding),
        data_(vectors.size() * stride_elements_, -999.0f) {
    for (size_t i = 0; i < vectors.size(); ++i) {
      std::copy(vectors[i].begin(), vectors[i].end(),
                data_.begin() + i * stride_elements_);
    }
  }

  zvec::core::ContiguousBatch get_contiguous_view() const override {
    return {data_.data(),
            0,
            static_cast<uint32_t>(data_.size() / stride_elements_),
            stride_elements_ * sizeof(float),
            {}};
  }

 private:
  size_t stride_elements_;
  std::vector<float> data_;
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

std::vector<std::vector<float>> ExternalVectors(MetricType metric) {
  auto vectors = RandomVectors(kGraphVectorCount);
  if (metric == MetricType::kCosine) {
    for (size_t i = 0; i < vectors.size(); ++i) {
      const float scale = static_cast<float>(i % 7 + 1);
      for (float &value : vectors[i]) {
        value *= scale;
      }
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

void CheckExternalTurboAddSearchReopen(
    MetricType metric, QuantizerType quantizer, const char *quantizer_name,
    const std::string &path, const std::vector<std::vector<float>> &vectors,
    const zvec::core::VectorSource &source) {
  zvec::test_util::RemoveTestFiles(path);
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
  auto vectors = ExternalVectors(metric.type);
  TestExternalVectorSource source(&vectors);
  CheckExternalTurboAddSearchReopen(metric.type, quantizer.type, quantizer.name,
                                    index_path("external"), vectors, source);
}

TEST_P(HnswTurboIndexTest, ContiguousExternalVectorsUseTurbo) {
  const auto &[quantizer, metric] = GetParam();
  auto vectors = ExternalVectors(metric.type);
  for (size_t padding : {0, 3}) {
    SCOPED_TRACE(padding);
    TestContiguousVectorSource source(vectors, padding);
    ASSERT_NO_FATAL_FAILURE(CheckExternalTurboAddSearchReopen(
        metric.type, quantizer.type, quantizer.name,
        index_path("contiguous_external"), vectors, source));
  }
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

TEST(HnswBulkMergeTest, DumperPreservesPayloadsBeyondFormerMetadataLimit) {
  namespace ailego = zvec::ailego;
  const std::string path = "hnsw_bulk_many_segments.index";
  AILEGO_DEFER([&]() { zvec::test_util::RemoveTestFiles(path); });
  constexpr uint32_t kSegments = 30000;
  // Include the version segment. The former fixed 1 MiB table overwrote
  // HnswT3S23907 when it grew; every record must survive reopening.
  zvec::core::HnswBuildDumper dumper(kSegments + 1);
  ASSERT_EQ(0, dumper.create(path));
  for (uint32_t i = 0; i < kSegments; ++i) {
    uint32_t payload[16];
    std::fill(std::begin(payload), std::end(payload), i + 1);
    ASSERT_EQ(sizeof(payload), dumper.write(payload, sizeof(payload)));
    ASSERT_EQ(
        0, dumper.append("HnswT3S" + std::to_string(i), sizeof(payload), 0, 0));
  }
  ASSERT_EQ(0, dumper.close());
  auto storage = zvec::core::IndexFactory::CreateStorage("MMapFileReadStorage");
  ASSERT_NE(nullptr, storage);
  ASSERT_EQ(0, storage->init({}));
  ASSERT_EQ(0, storage->open(path, false));
  EXPECT_EQ(kSegments + 1, storage->get_all().size());
  for (uint32_t i = 0; i < kSegments; ++i) {
    auto segment = storage->get("HnswT3S" + std::to_string(i));
    ASSERT_NE(nullptr, segment);
    zvec::core::IndexStorage::MemoryBlock block;
    ASSERT_EQ(64U, segment->read(0, block, 64));
    const auto *values = static_cast<const uint32_t *>(block.data());
    for (size_t j = 0; j < 16; ++j) ASSERT_EQ(i + 1, values[j]) << i;
  }
  ASSERT_EQ(0, storage->close());
}

TEST(HnswBulkMergeTest, DumperRejectsUnderestimatedMetadataBudget) {
  namespace ailego = zvec::ailego;
  const std::string path = "hnsw_bulk_metadata_budget.index";
  AILEGO_DEFER([&]() { zvec::test_util::RemoveTestFiles(path); });
  zvec::core::HnswBuildDumper dumper(1);
  ASSERT_EQ(0, dumper.create(path));
  const uint32_t value = 42;
  ASSERT_EQ(sizeof(value), dumper.write(&value, sizeof(value)));
  EXPECT_EQ(zvec::core::IndexError_IndexFull,
            dumper.append("HnswT3S0", sizeof(value), 0, 0));
}

struct BulkContextState {
  size_t workers;
  bool fail_first;
  mutable std::atomic<size_t> context_calls{0};
  mutable std::atomic<size_t> contexts_ready{0};
  std::atomic<size_t> adds{0};
  std::atomic<size_t> premature_adds{0};
  bool fail_worker_allocation{false};
  bool fail_task_allocation{false};
  std::array<const void *, 4> context_ptrs{};
  std::atomic<size_t> active_adds{0};
  std::atomic<size_t> context_deletes{0};
  std::atomic<size_t> early_context_deletes{0};
  std::atomic<bool> failure_triggered{false};
  std::atomic<bool> failed_with_worker_active{false};
  std::atomic<bool> release_worker{false};
};

bool WaitForFlag(const std::atomic<bool> &flag) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!flag.load() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  return flag.load();
}

void BeforeBulkAllocationFailure() {
  auto *state = static_cast<BulkContextState *>(bulk_alloc_test::state);
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (state->active_adds.load() == 0 &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  state->failed_with_worker_active.store(state->active_adds.load() != 0);
  state->failure_triggered.store(true);
}

void CheckBulkContextDeletion(void *ptr) {
  auto *state = static_cast<BulkContextState *>(bulk_alloc_test::state);
  for (const void *context : state->context_ptrs) {
    if (context == ptr) {
      state->context_deletes.fetch_add(1);
      if (state->active_adds.load() != 0) {
        state->early_context_deletes.fetch_add(1);
      }
    }
  }
}

class CheckedBulkStreamer : public zvec::core::HnswStreamer {
 public:
  explicit CheckedBulkStreamer(BulkContextState *state) : state_(state) {}

  int open_with_borrowed_vectors(
      zvec::core::IndexStorage::Pointer storage,
      const zvec::core::VectorSource *source) override {
    const int ret = HnswStreamer::open_with_borrowed_vectors(storage, source);
    borrowed_build_ = ret == 0;
    return ret;
  }

  zvec::core::IndexStreamer::Context::Pointer create_context() const override {
    if (!borrowed_build_) return HnswStreamer::create_context();
    const auto call = state_->context_calls.fetch_add(1);
    if (state_->fail_first && call == 0) return nullptr;
    auto context = HnswStreamer::create_context();
    if (context) {
      state_->context_ptrs[call] = context.get();
      const auto ready = state_->contexts_ready.fetch_add(1) + 1;
      if (state_->fail_task_allocation && ready == state_->workers) {
        bulk_alloc_test::state = state_;
        bulk_alloc_test::before_failure = BeforeBulkAllocationFailure;
        bulk_alloc_test::on_delete = CheckBulkContextDeletion;
        // One group allocation, the first closure, then the second closure.
        // The external pool's deque is primed before this fault is armed.
        bulk_alloc_test::countdown = 3;
      }
    }
    return context;
  }

  int add_with_id_impl(
      uint32_t id, const void *data, const zvec::core::IndexQueryMeta &meta,
      zvec::core::IndexStreamer::Context::Pointer &context) override {
    if (state_->contexts_ready.load() != state_->workers) {
      state_->premature_adds.fetch_add(1);
    }
    state_->adds.fetch_add(1);
    if (state_->fail_worker_allocation) throw std::bad_alloc();
    if (state_->fail_task_allocation) {
      state_->active_adds.fetch_add(1);
      while (!state_->release_worker.load()) std::this_thread::yield();
      state_->active_adds.fetch_sub(1);
      return zvec::core::IndexError_Runtime;
    }
    return HnswStreamer::add_with_id_impl(id, data, meta, context);
  }

 private:
  BulkContextState *state_;
  bool borrowed_build_{false};
};

class ScopedBulkStreamerFactory {
 public:
  explicit ScopedBulkStreamerFactory(BulkContextState *state) {
    zvec::ailego::Factory<zvec::core::IndexStreamer>::Register<
        CheckedBulkStreamer>
        replace("HnswStreamer", static_cast<BulkContextState *>(state));
  }
  ~ScopedBulkStreamerFactory() {
    zvec::ailego::Factory<zvec::core::IndexStreamer>::Register<
        zvec::core::HnswStreamer>
        restore("HnswStreamer");
  }
};

TEST(HnswBulkMergeTest, PreparesAllContextsBeforeWritesAndPropagatesFailure) {
  for (int failure : {0, 1, 2}) {
    namespace ailego = zvec::ailego;
    const std::string source_path = "hnsw_bulk_context_source.index";
    const std::string target_path = "hnsw_bulk_context_target.index";
    AILEGO_DEFER([&]() {
      zvec::test_util::RemoveTestFiles(source_path);
      zvec::test_util::RemoveTestFiles(target_path);
    });
    zvec::ailego::ThreadPool pool(4, false);
    const bool fail_first = failure == 1;
    BulkContextState state{std::min<size_t>(4, pool.count()), fail_first};
    state.fail_worker_allocation = failure == 2;
    auto flat_param = FlatIndexParamBuilder()
                          .with_data_type(DataType::DT_FP32)
                          .with_dimension(kDimension)
                          .with_metric_type(MetricType::kL2sq)
                          .build();
    auto source = IndexFactory::CreateAndInitIndex(*flat_param);
    auto target = IndexFactory::CreateAndInitIndex(
        *MakeParam(MetricType::kL2sq, QuantizerType::kNone));
    ASSERT_NE(nullptr, source);
    ASSERT_NE(nullptr, target);
    ASSERT_EQ(0, source->open(source_path,
                              {StorageOptions::StorageType::kMMAP, true}));
    ASSERT_EQ(0, target->open(target_path,
                              {StorageOptions::StorageType::kMMAP, true}));
    auto vectors = RandomVectors();
    AddVectors(source.get(), vectors);
    {
      ScopedBulkStreamerFactory factory(&state);
      EXPECT_EQ(failure ? zvec::core::IndexError_NoMemory : 0,
                target->merge({source}, IndexFilter(), {4, &pool}));
    }
    EXPECT_EQ(fail_first ? 1U : state.workers, state.context_calls.load());
    EXPECT_EQ(0U, state.premature_adds.load());
    if (state.fail_worker_allocation) {
      EXPECT_GT(state.adds.load(), 0U);
    } else {
      EXPECT_EQ(fail_first ? 0U : kVectorCount, state.adds.load());
    }
    EXPECT_EQ(failure ? 0U : kVectorCount, target->get_doc_count());
    if (failure) {
      // The original empty target remains open and can accept a later write.
      EXPECT_EQ(0, target->add({DenseVector{vectors[0].data()}}, 0));
    }
    EXPECT_EQ(0, target->close());
    EXPECT_EQ(0, source->close());
  }
}

TEST(HnswBulkMergeTest, TaskAllocationFailureWaitsBeforeDestroyingContexts) {
  namespace ailego = zvec::ailego;
  BulkContextState state{0, false};
  AILEGO_DEFER([]() { bulk_alloc_test::Reset(); });
  ailego::ThreadPool pool(2, false);
  if (pool.count() < 2) GTEST_SKIP() << "Requires two workers";
  state.workers = pool.count();
  state.fail_task_allocation = true;
  pool.execute_and_wait([]() {});
  const std::string source_path = "hnsw_bulk_task_source.index";
  const std::string target_path = "hnsw_bulk_task_target.index";
  AILEGO_DEFER([&]() {
    zvec::test_util::RemoveTestFiles(source_path);
    zvec::test_util::RemoveTestFiles(target_path);
  });
  auto flat_param = FlatIndexParamBuilder()
                        .with_data_type(DataType::DT_FP32)
                        .with_dimension(kDimension)
                        .with_metric_type(MetricType::kL2sq)
                        .build();
  auto source = IndexFactory::CreateAndInitIndex(*flat_param);
  auto target = IndexFactory::CreateAndInitIndex(
      *MakeParam(MetricType::kL2sq, QuantizerType::kNone));
  ASSERT_NE(nullptr, source);
  ASSERT_NE(nullptr, target);
  ASSERT_EQ(
      0, source->open(source_path, {StorageOptions::StorageType::kMMAP, true}));
  ASSERT_EQ(
      0, target->open(target_path, {StorageOptions::StorageType::kMMAP, true}));
  auto vectors = RandomVectors();
  AddVectors(source.get(), vectors);
  std::thread controller([&]() {
    WaitForFlag(state.failure_triggered);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    state.release_worker.store(true);
  });
  AILEGO_DEFER([&]() {
    state.release_worker.store(true);
    if (controller.joinable()) controller.join();
  });
  int ret;
  {
    ScopedBulkStreamerFactory factory(&state);
    ret = target->merge({source}, IndexFilter(), {2, &pool});
    bulk_alloc_test::Reset();
  }
  controller.join();
  EXPECT_EQ(zvec::core::IndexError_NoMemory, ret);
  EXPECT_TRUE(state.failure_triggered.load());
  EXPECT_TRUE(state.failed_with_worker_active.load());
  EXPECT_EQ(state.workers, state.context_deletes.load());
  EXPECT_EQ(0U, state.early_context_deletes.load());
  EXPECT_EQ(0U, state.active_adds.load());
  EXPECT_EQ(0U, target->get_doc_count());
  bool reused = false;
  pool.execute_and_wait([&]() { reused = true; });
  EXPECT_TRUE(reused);
  EXPECT_EQ(0, target->add({DenseVector{vectors[0].data()}}, 0));
  EXPECT_EQ(0, target->close());
  EXPECT_EQ(0, source->close());
}

TEST(HnswBulkMergeTest, FailedQueueAllocationLeavesPoolReusable) {
  namespace ailego = zvec::ailego;
  std::atomic<bool> release{false};
  std::atomic<bool> started{false};
  std::atomic<size_t> completed{0};
  ailego::ThreadPool pool(1, false);
  AILEGO_DEFER([&]() {
    bulk_alloc_test::Reset();
    pool.stop();
    release.store(true);
  });
  auto blockers = pool.make_group();
  blockers->submit(ailego::Closure::New([&]() {
    started.store(true);
    while (!release.load()) std::this_thread::yield();
  }));
  ASSERT_TRUE(WaitForFlag(started));
  auto group = pool.make_group();
  std::vector<ailego::ClosureHandler> tasks;
  tasks.reserve(1024);
  for (size_t i = 0; i < 1024; ++i) {
    tasks.push_back(ailego::Closure::New([&]() { completed.fetch_add(1); }));
  }
  size_t accepted = 0;
  bool failed = false;
  bulk_alloc_test::countdown = 1;
  try {
    for (auto &task : tasks) {
      group->submit(std::move(task));
      ++accepted;
    }
  } catch (const std::bad_alloc &) {
    failed = true;
  }
  bulk_alloc_test::Reset();
  EXPECT_TRUE(failed);
  EXPECT_EQ(accepted, group->pending_count());
  EXPECT_EQ(accepted, pool.pending_count());
  // Avoid hanging the test if counter consistency regresses.
  if (group->pending_count() != accepted || pool.pending_count() != accepted) {
    return;
  }
  release.store(true);
  blockers->wait_finish();
  group->wait_finish();
  EXPECT_EQ(accepted, completed.load());
  bool reused = false;
  pool.execute_and_wait([&]() { reused = true; });
  EXPECT_TRUE(reused);
}

class HnswBulkMergeTest
    : public testing::TestWithParam<std::tuple<MetricType, bool, bool>> {};

TEST_P(HnswBulkMergeTest, FilteredMixedSourcesReopenAndAppend) {
  const auto &[metric, graph_source, read_only_source] = GetParam();
  const std::string prefix =
      "hnsw_bulk_" + std::to_string(static_cast<int>(metric)) + "_" +
      std::to_string(graph_source) + "_" + std::to_string(read_only_source);
  const std::string source_path = prefix + "_source.index";
  const std::string tail_path = prefix + "_tail.index";
  const std::string target_path = prefix + "_target.index";
  for (const auto &path : {source_path, tail_path, target_path}) {
    zvec::test_util::RemoveTestFiles(path + "*");
  }
  // The retained output spans multiple ordinary node chunks even though the
  // graph-only build fits in one. Scale cosine inputs to exercise norm tails.
  auto vectors = RandomVectors(14000);
  for (size_t i = 0; i < vectors.size(); ++i) {
    for (float &value : vectors[i]) value *= static_cast<float>(i % 7 + 1);
  }
  auto flat_param = FlatIndexParamBuilder()
                        .with_metric_type(metric)
                        .with_data_type(DataType::DT_FP32)
                        .with_dimension(kDimension)
                        .build();
  auto graph_param = MakeParam(metric, QuantizerType::kNone);
  BaseIndexParam::Pointer source_param =
      graph_source ? std::static_pointer_cast<BaseIndexParam>(graph_param)
                   : std::static_pointer_cast<BaseIndexParam>(flat_param);
  auto source = IndexFactory::CreateAndInitIndex(*source_param);
  auto tail = IndexFactory::CreateAndInitIndex(*flat_param);
  ASSERT_NE(nullptr, source);
  ASSERT_NE(nullptr, tail);
  ASSERT_EQ(
      0, source->open(source_path, {StorageOptions::StorageType::kMMAP, true}));
  ASSERT_EQ(0,
            tail->open(tail_path, {StorageOptions::StorageType::kMMAP, true}));
  constexpr size_t kSplit = 9000;
  for (size_t i = 0; i < vectors.size(); ++i) {
    auto &index = i < kSplit ? source : tail;
    ASSERT_EQ(0, index->add({DenseVector{vectors[i].data()}},
                            i < kSplit ? i : i - kSplit));
  }
  if (read_only_source) {
    ASSERT_EQ(0, source->close());
    ASSERT_EQ(0, tail->close());
    source = IndexFactory::CreateAndInitIndex(*source_param);
    tail = IndexFactory::CreateAndInitIndex(*flat_param);
    ASSERT_EQ(0, source->open(source_path, {StorageOptions::StorageType::kMMAP,
                                            false, true}));
    ASSERT_EQ(0, tail->open(tail_path,
                            {StorageOptions::StorageType::kMMAP, false, true}));
  }
  auto provider = source->create_index_provider();
  auto *stable = dynamic_cast<zvec::core::StableVectorHolder *>(provider.get());
  ASSERT_NE(nullptr, stable);
  std::shared_ptr<const void> lifetime;
  ASSERT_EQ(0, stable->retain_stable_data(&lifetime));
  auto iterator = provider->create_iterator();
  ASSERT_TRUE(iterator->is_valid());
  const auto *first = static_cast<const float *>(iterator->data());
  std::vector<float> saved(first, first + kDimension);
  for (size_t i = 0; i < kSplit; ++i) {
    ASSERT_TRUE(iterator->is_valid());
    ASSERT_NE(nullptr, iterator->data());
    iterator->next();
  }
  iterator.reset();
  provider.reset();
  EXPECT_TRUE(std::equal(saved.begin(), saved.end(), first));

  auto target = IndexFactory::CreateAndInitIndex(*graph_param);
  ASSERT_EQ(
      0, target->open(target_path, {StorageOptions::StorageType::kMMAP, true}));
  std::vector<uint32_t> calls(vectors.size(), 0);
  IndexFilter filter;
  filter.set([&](uint64_t key) {
    EXPECT_LT(key, calls.size());
    if (key >= calls.size()) return true;
    ++calls[key];
    return key % 4 == 1;
  });
  zvec::ailego::ThreadPool pool(4, false);
  ASSERT_EQ(0, target->merge({source, tail}, filter, {4, &pool}));
  EXPECT_TRUE(std::all_of(calls.begin(), calls.end(),
                          [](uint32_t count) { return count == 1; }));
  ASSERT_EQ(10500U, target->get_doc_count());
  std::vector<size_t> kept;
  for (size_t i = 0; i < vectors.size(); ++i) {
    if (i % 4 != 1) kept.push_back(i);
  }
  std::vector<SearchRowList> rows_before_reopen;
  auto check = [&](bool reopened) {
    size_t query_index = 0;
    for (size_t id : {size_t{0}, size_t{4095}, kept.size() - 1}) {
      VectorDataBuffer fetched;
      ASSERT_EQ(0, target->fetch(id, &fetched));
      const auto &bytes =
          std::get<DenseVectorBuffer>(fetched.vector_buffer).data;
      ASSERT_EQ(kDimension * sizeof(float), bytes.size());
      for (size_t dim = 0; dim < kDimension; ++dim) {
        float value;
        std::memcpy(&value, bytes.data() + dim * sizeof(float), sizeof(float));
        EXPECT_NEAR(vectors[kept[id]][dim], value, 2e-6f);
      }
      auto graph = SearchRows(target.get(), vectors[kept[id]], false);
      ASSERT_EQ(kTopK, graph.size());
      if (reopened) {
        EXPECT_EQ(rows_before_reopen[query_index], graph);
      } else {
        rows_before_reopen.push_back(graph);
      }
      ++query_index;
    }
  };
  ASSERT_NO_FATAL_FAILURE(check(false));
  ASSERT_EQ(0, target->close());
  target = IndexFactory::CreateAndInitIndex(*graph_param);
  ASSERT_EQ(0, target->open(target_path,
                            {StorageOptions::StorageType::kMMAP, false}));
  ASSERT_NO_FATAL_FAILURE(check(true));
  const uint32_t appended_id = kept.size();
  ASSERT_EQ(0, target->add({DenseVector{vectors[1].data()}}, appended_id));
  // A subsequent merge must append using the ordinary mutable format.
  ASSERT_EQ(0, target->merge({tail}, IndexFilter(), {2, nullptr}));
  EXPECT_EQ(15501U, target->get_doc_count());
  ASSERT_EQ(0, target->close());
  auto storage = zvec::core::IndexFactory::CreateStorage("MMapFileReadStorage");
  ASSERT_EQ(0, storage->init(zvec::ailego::Params{}));
  ASSERT_EQ(0, storage->open(target_path, false));
  for (const auto &[name, segment] : storage->get_all()) {
    if (name.rfind("HnswT", 0) == 0) {
      EXPECT_EQ(0U, segment->data_crc());
    }
    if (segment->data_crc() != 0) {
      zvec::core::IndexStorage::MemoryBlock block;
      ASSERT_EQ(segment->data_size(),
                segment->read(0, block, segment->data_size()));
      EXPECT_EQ(
          segment->data_crc(),
          zvec::ailego::Crc32c::Hash(block.data(), segment->data_size(), 0));
    }
  }
  ASSERT_EQ(0, storage->close());
  ASSERT_EQ(0, source->close());
  ASSERT_EQ(0, tail->close());
  lifetime.reset();
  for (const auto &path : {source_path, tail_path, target_path}) {
    zvec::test_util::RemoveTestFiles(path);
  }
}

INSTANTIATE_TEST_SUITE_P(
    StableSources, HnswBulkMergeTest,
    testing::Combine(testing::Values(MetricType::kL2sq, MetricType::kCosine,
                                     MetricType::kInnerProduct),
                     testing::Bool(), testing::Bool()));

TEST(HnswBulkMergeTest, EmptyFilterAndCrashResidueKeepTargetUsable) {
  const std::string source_path = "hnsw_bulk_empty_source.index";
  const std::string target_path = "hnsw_bulk_empty_target.index";
  zvec::test_util::RemoveTestFiles(source_path + "*");
  zvec::test_util::RemoveTestFiles(target_path + "*");
  auto param = MakeParam(MetricType::kL2sq, QuantizerType::kNone);
  auto source = IndexFactory::CreateAndInitIndex(*param);
  auto target = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_EQ(
      0, source->open(source_path, {StorageOptions::StorageType::kMMAP, true}));
  ASSERT_EQ(
      0, target->open(target_path, {StorageOptions::StorageType::kMMAP, true}));
  auto vectors = RandomVectors();
  AddVectors(source.get(), vectors);
  size_t calls = 0;
  IndexFilter all;
  all.set([&](uint64_t) {
    ++calls;
    return true;
  });
  ASSERT_EQ(0, target->merge({source}, all));
  EXPECT_EQ(kVectorCount, calls);
  EXPECT_EQ(0U, target->get_doc_count());
  EXPECT_TRUE(target->is_trained());
  // Scratch from an interrupted attempt must neither block the next build
  // nor be mistaken for files owned by the current attempt.
  const std::string residue = target_path + ".hnsw-pack.stale";
  ASSERT_TRUE(zvec::ailego::File::MakePath(residue));
  ASSERT_EQ(0, target->merge({source}, IndexFilter()));
  EXPECT_EQ(kVectorCount, target->get_doc_count());
  EXPECT_TRUE(zvec::ailego::File::IsExist(residue));
  zvec::test_util::RemoveTestPath(residue);
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

TEST(HnswPqIndexTest, ReadOnlyOpenSkipsBuildPreparation) {
  using namespace zvec;
  class CountingQuantizer : public turbo::Fp32Quantizer {
   public:
    int prepare_build() override {
      ++prepare_count;
      return 0;
    }
    int prepare_count{0};
  };

  const std::string path = "hnsw_read_only_prepare.index";
  test_util::RemoveTestFiles(path);
  core::IndexMeta meta(core::IndexMeta::DT_FP32, kDimension);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  auto quantizer = std::make_shared<CountingQuantizer>();
  ASSERT_EQ(0, quantizer->init(meta, ailego::Params()));
  auto param = MakeParam(MetricType::kL2sq, QuantizerType::kNone);
  auto seed = IndexFactory::CreateAndInitIndex(*param);
  core::IndexStreamer::Pointer streamer =
      std::make_shared<core::HnswStreamer>();
  ASSERT_EQ(
      0, streamer->init(meta, seed->index_searcher()->meta().streamer_params(),
                        quantizer));
  auto storage = core::IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init(ailego::Params()));
  ASSERT_EQ(0, storage->open(path, true));
  ASSERT_EQ(0, streamer->open(storage));
  EXPECT_EQ(1, quantizer->prepare_count);
  ASSERT_EQ(0, streamer->close());

  quantizer->prepare_count = 0;
  ASSERT_EQ(0, streamer->open(storage, true));
  EXPECT_EQ(0, quantizer->prepare_count);
  auto context = streamer->create_context();
  std::vector<float> vector(kDimension, 1.0f);
  core::IndexQueryMeta qmeta;
  qmeta.set_meta(meta.data_type(), meta.dimension());
  EXPECT_NE(0, streamer->add_impl(0, vector.data(), qmeta, context));
  EXPECT_NE(0, streamer->add_with_id_impl(0, vector.data(), qmeta, context));
  ASSERT_EQ(0, streamer->close());

  // The legacy open entry point still means writable, including after a
  // read-only open on the same streamer.
  ASSERT_EQ(0, streamer->open(storage));
  EXPECT_EQ(1, quantizer->prepare_count);
  context = streamer->create_context();
  ASSERT_EQ(0, streamer->add_with_id_impl(0, vector.data(), qmeta, context));
  ASSERT_EQ(0, streamer->close());
  ASSERT_EQ(0, storage->close());
  test_util::RemoveTestFiles(path);
}

TEST(HnswPqIndexTest, ReopenAndAddUsesDataCodes) {
  using namespace zvec;
  const std::string path = "hnsw_pq_reopen_test.index";
  test_util::RemoveTestFiles(path);
  const auto vectors = RandomVectors(256);
  auto holder =
      std::make_shared<core::MultiPassIndexHolder<core::IndexMeta::DT_FP32>>(
          kDimension);
  for (size_t i = 0; i < vectors.size(); ++i) {
    ailego::NumericalVector<float> v(kDimension);
    std::copy(vectors[i].begin(), vectors[i].end(), v.data());
    holder->emplace(i, v);
  }
  core::IndexMeta raw(core::IndexMeta::DT_FP32, kDimension);
  raw.set_metric("SquaredEuclidean", 0, ailego::Params());
  ailego::Params params;
  params.set("num_chunk", 4U);
  auto quantizer = core::IndexFactory::CreateQuantizer("PqInt8Quantizer");
  ASSERT_EQ(0, quantizer->init(raw, params));
  ASSERT_EQ(0, quantizer->train(holder));
  auto meta = quantizer->meta();
  meta.set_quantizer("PqInt8Quantizer", 0, params);
  auto param = MakeParam(MetricType::kL2sq, QuantizerType::kNone);
  auto seed = IndexFactory::CreateAndInitIndex(*param);
  core::IndexStreamer::Pointer streamer =
      std::make_shared<core::HnswStreamer>();
  ASSERT_EQ(
      0, streamer->init(meta, seed->index_searcher()->meta().streamer_params(),
                        quantizer));
  auto storage = core::IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init(ailego::Params()));
  ASSERT_EQ(0, storage->open(path, true));
  ASSERT_EQ(0, streamer->open(storage));
  auto ctx = streamer->create_context();
  core::IndexQueryMeta codes;
  codes.set_meta(meta.data_type(), meta.dimension(),
                 static_cast<uint32_t>(quantizer->type()),
                 meta.extra_meta_size());
  std::string code(quantizer->quantized_datapoint_vector_length(), '\0');
  for (size_t i = 0; i < 32; ++i) {
    quantizer->quantize_data(vectors[i].data(), code.data());
    ASSERT_EQ(0, streamer->add_with_id_impl(i, code.data(), codes, ctx));
  }
  ASSERT_EQ(0, streamer->close());
  ASSERT_EQ(0, storage->close());
  auto index = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_EQ(0, index->open(path, {StorageOptions::StorageType::kMMAP, false}));
  ASSERT_EQ(0, index->add(VectorData{DenseVector{vectors[32].data()}}, 32));
  quantizer->quantize_data(vectors[32].data(), code.data());
  std::string expected;
  ASSERT_EQ(0, quantizer->dequantize(code.data(), codes, &expected));
  VectorDataBuffer fetched;
  ASSERT_EQ(0, index->fetch(32, &fetched));
  EXPECT_EQ(expected, std::get<DenseVectorBuffer>(fetched.vector_buffer).data);
  auto rows = SearchRows(index.get(), vectors[32], true);
  ASSERT_FALSE(rows.empty());
  EXPECT_EQ(32U, rows.front().first);
  ASSERT_EQ(0, index->close());
  test_util::RemoveTestFiles(path);
}

class HnswPqLifecycleTest
    : public testing::TestWithParam<
          std::tuple<int, bool, MetricType, StorageOptions::StorageType>> {};

TEST_P(HnswPqLifecycleTest, TrainMergeReopenAndAppend) {
  using namespace zvec;
  const auto [bits, rotate, metric, storage_type] = GetParam();
  const std::string raw_path = "hnsw_pq_raw.index";
  const std::string pq_path = "hnsw_pq_target.index";
  const std::string merge_path = "hnsw_pq_merge.index";
  for (auto &path : {raw_path, pq_path, merge_path})
    test_util::RemoveTestFiles(path);
  ailego::MemoryLimitPool::get_instance().init(256 * 1024 * 1024);
  auto vectors = RandomVectors(300);
  auto raw_param = FlatIndexParamBuilder()
                       .with_dimension(kDimension)
                       .with_data_type(DataType::DT_FP32)
                       .with_metric_type(metric)
                       .build();
  auto raw = IndexFactory::CreateAndInitIndex(*raw_param);
  ASSERT_NE(nullptr, raw);
  ASSERT_EQ(0, raw->open(raw_path, {storage_type, true}));
  AddVectors(raw.get(), vectors);
  PqQuantizerParam pq(bits == 8 ? 5 : 4, bits, rotate);
  pq.opq_iter = 2;
  pq.opq_pq_iter = 2;
  auto param = MakeDefaultParam(metric);
  param->quantizer_param = pq.clone();
  auto index = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, index);
  ASSERT_EQ(0, index->open(pq_path, {storage_type, true}));
  EXPECT_EQ(0U, index->get_doc_count());
  EXPECT_NE(0, index->add(VectorData{DenseVector{vectors[0].data()}}, 0));
  EXPECT_NE(0, index->train());
  ASSERT_EQ(0, index->merge({raw}, IndexFilter()));
  EXPECT_EQ(vectors.size(), index->get_doc_count());
  auto before = SearchRows(index.get(), vectors[11], true);
  auto graph_before = SearchRows(index.get(), vectors[11], false);
  ASSERT_FALSE(before.empty());
  ASSERT_EQ(0, index->close());
  index = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_EQ(0, index->open(pq_path, {storage_type, false, true}));
  EXPECT_EQ(before, SearchRows(index.get(), vectors[11], true));
  EXPECT_EQ(graph_before, SearchRows(index.get(), vectors[11], false));
  EXPECT_NE(0, index->add(VectorData{DenseVector{vectors[0].data()}}, 300));
  ASSERT_EQ(0, index->close());
  // A writable reopen must restore the SDC table before adding vectors.
  index = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_EQ(0, index->open(pq_path, {storage_type, false}));
  EXPECT_EQ(before, SearchRows(index.get(), vectors[11], true));
  ASSERT_EQ(0, index->add(VectorData{DenseVector{vectors[0].data()}}, 300));
  EXPECT_EQ(301U, index->get_doc_count());

  // Candidate extraction accepts the PQ query layout and preserves scores.
  auto query_param =
      HNSWQueryParamBuilder().with_topk(kTopK).with_ef_search(100).build();
  const VectorData query{DenseVector{vectors[11].data()}};
  SearchResult expected_rows;
  ASSERT_EQ(0, index->search(query, query_param, &expected_rows));
  std::vector<int64_t> ids(kTopK);
  std::vector<float> scores(kTopK);
  ASSERT_EQ(0, index->search_internal_ids(query, query_param, ids.data(),
                                          scores.data()));
  ASSERT_EQ(kTopK, expected_rows.doc_list_.size());
  for (size_t i = 0; i < kTopK; ++i) {
    EXPECT_EQ(expected_rows.doc_list_[i].key(), static_cast<uint64_t>(ids[i]));
    EXPECT_FLOAT_EQ(expected_rows.doc_list_[i].score(), scores[i]);
  }

  // Train a separate codebook from reconstructed source vectors. Codes from
  // the first index must be decoded and encoded again, even with equal layout.
  auto merged = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, merged);
  ASSERT_EQ(0, merged->open(merge_path, {storage_type, true}));
  ASSERT_EQ(0, merged->merge({index}, IndexFilter()));
  ASSERT_EQ(0, merged->flush());
  auto storage = core::IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init(ailego::Params()));
  ASSERT_EQ(0, storage->open(merge_path, false));
  std::shared_ptr<turbo::Quantizer> quantizer;
  ASSERT_EQ(0, core::HnswStreamer::RestoreQuantizer(
                   storage, merged->index_searcher()->meta(), &quantizer));
  std::string code(quantizer->quantized_datapoint_vector_length(), '\0');
  for (uint32_t id : {0U, 11U, 299U, 300U}) {
    VectorDataBuffer source, actual;
    ASSERT_EQ(0, index->fetch(id, &source));
    const auto &decoded =
        std::get<DenseVectorBuffer>(source.vector_buffer).data;
    quantizer->quantize_data(decoded.data(), code.data());
    std::string expected;
    ASSERT_EQ(0, quantizer->dequantize(code.data(), core::IndexQueryMeta(),
                                       &expected));
    ASSERT_EQ(0, merged->fetch(id, &actual));
    EXPECT_EQ(expected, std::get<DenseVectorBuffer>(actual.vector_buffer).data);
  }
  ASSERT_EQ(0, storage->close());
  ASSERT_EQ(0, merged->close());
  ASSERT_EQ(0, index->close());
  ASSERT_EQ(0, raw->close());
  for (auto &path : {raw_path, pq_path, merge_path})
    test_util::RemoveTestFiles(path);
}

INSTANTIATE_TEST_SUITE_P(
    Pq, HnswPqLifecycleTest,
    testing::Combine(
        testing::Values(4, 8), testing::Bool(),
        testing::Values(MetricType::kL2sq, MetricType::kInnerProduct,
                        MetricType::kCosine),
        testing::Values(StorageOptions::StorageType::kMMAP,
                        StorageOptions::StorageType::kBufferPool)));
