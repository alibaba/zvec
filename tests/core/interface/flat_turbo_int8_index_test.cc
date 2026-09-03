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
#include <random>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_framework.h>
#include <zvec/core/framework/index_streamer.h>
#include <zvec/core/interface/index.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param.h>
#include <zvec/core/interface/index_param_builders.h>
#include "tests/test_util.h"

using namespace zvec::core_interface;

namespace {

constexpr uint32_t kDimension = 32;
constexpr size_t kVectorCount = 200;
constexpr uint32_t kTopK = 10;

std::vector<std::vector<float>> RandomVectors(size_t count, uint32_t dim) {
  std::mt19937 gen(2026);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<std::vector<float>> vectors(count, std::vector<float>(dim, 0.0f));
  for (auto &vec : vectors) {
    float norm = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) {
      vec[i] = dist(gen);
      norm += vec[i] * vec[i];
    }
    norm = std::sqrt(norm);
    for (uint32_t i = 0; i < dim; ++i) {
      vec[i] /= norm;
    }
  }
  return vectors;
}

float CosineDistance(const std::vector<float> &a, const std::vector<float> &b) {
  float dot = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) {
    dot += a[i] * b[i];
  }
  return 1.0f - dot;
}

float L2SquaredDistance(const std::vector<float> &a,
                        const std::vector<float> &b) {
  float sum = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) {
    float diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

// Brute-force top-k over the corpus; returns (key, score) sorted by score.
std::vector<std::pair<uint32_t, float>> BruteForceTopK(
    const std::vector<std::vector<float>> &vectors,
    const std::vector<float> &query, uint32_t topk,
    float (*distance)(const std::vector<float> &, const std::vector<float> &)) {
  std::vector<std::pair<uint32_t, float>> scored;
  scored.reserve(vectors.size());
  for (size_t i = 0; i < vectors.size(); ++i) {
    scored.emplace_back(static_cast<uint32_t>(i), distance(vectors[i], query));
  }
  std::partial_sort(scored.begin(), scored.begin() + topk, scored.end(),
                    [](const auto &lhs, const auto &rhs) {
                      if (lhs.second != rhs.second) {
                        return lhs.second < rhs.second;
                      }
                      return lhs.first < rhs.first;
                    });
  scored.resize(topk);
  return scored;
}

FlatIndexParam::Pointer MakeParam(MetricType metric, QuantizerType quantizer,
                                  bool enable_rotate = false) {
  return FlatIndexParamBuilder()
      .with_metric_type(metric)
      .with_data_type(DataType::DT_FP32)
      .with_dimension(kDimension)
      .with_is_sparse(false)
      .with_quantizer_param(QuantizerParam(quantizer, enable_rotate))
      .build();
}

struct SearchOutcome {
  // (key, score) pairs returned by the index.
  std::vector<std::pair<uint32_t, float>> rows;
};

SearchOutcome RunSearch(Index *index, const std::vector<float> &query) {
  SearchOutcome outcome;
  auto query_param = FlatQueryParamBuilder().with_topk(kTopK).build();
  SearchResult result;
  VectorData vector_data;
  vector_data.vector = DenseVector{query.data()};
  EXPECT_EQ(0, index->search(vector_data, query_param, &result));
  for (const auto &doc : result.doc_list_) {
    outcome.rows.emplace_back(doc.key(), doc.score());
  }
  return outcome;
}

// Adds the corpus, searches a query, compares against brute-force truth.
// Returns nullptr-checked index; the caller closes it.
void CheckTopKAgainstBruteForce(Index *index,
                                const std::vector<std::vector<float>> &vectors,
                                const std::vector<float> &query,
                                float (*distance)(const std::vector<float> &,
                                                  const std::vector<float> &)) {
  auto got = RunSearch(index, query);
  ASSERT_EQ(kTopK, got.rows.size());
  auto want = BruteForceTopK(vectors, query, kTopK, distance);
  // INT8 quantization may reorder near neighbours; compare score sets with
  // tolerance instead of exact key order.
  for (uint32_t i = 0; i < kTopK; ++i) {
    EXPECT_NEAR(want[i].second, got.rows[i].second, 0.15f)
        << "rank " << i << " truth score " << want[i].second << " vs "
        << got.rows[i].second;
  }
  // Exact top-1 must match (random unit vectors are well separated).
  EXPECT_EQ(want[0].first, got.rows[0].first);
}

std::vector<std::pair<uint32_t, float>> ScoredRows(SearchResult *result) {
  std::vector<std::pair<uint32_t, float>> rows;
  for (const auto &doc : result->doc_list_) {
    rows.emplace_back(doc.key(), doc.score());
  }
  return rows;
}

}  // namespace

TEST(FlatTurboInt8Index, CosineAddSearchReopenFetch) {
  const std::string index_name{"flat_turbo_int8_cosine.index"};
  zvec::test_util::RemoveTestFiles(index_name);
  auto vectors = RandomVectors(kVectorCount, kDimension);

  auto param = MakeParam(MetricType::kCosine, QuantizerType::kInt8);
  auto index = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, index);
  ASSERT_EQ(
      0, index->open(index_name, {StorageOptions::StorageType::kMMAP, true}));

  for (size_t i = 0; i < vectors.size(); ++i) {
    VectorData vector_data;
    vector_data.vector = DenseVector{vectors[i].data()};
    ASSERT_EQ(0, index->add(vector_data, static_cast<uint32_t>(i)));
  }
  ASSERT_EQ(0, index->train());

  CheckTopKAgainstBruteForce(index.get(), vectors, vectors[7], CosineDistance);
  CheckTopKAgainstBruteForce(index.get(), vectors, vectors[101],
                             CosineDistance);

  // fetch must dequantize back to the original (unnormalized-by-int8) vector.
  VectorDataBuffer fetched;
  ASSERT_EQ(0, index->fetch(7, &fetched));
  const auto *fetched_vector = reinterpret_cast<const float *>(
      std::get<DenseVectorBuffer>(fetched.vector_buffer).data.data());
  for (uint32_t i = 0; i < kDimension; ++i) {
    EXPECT_NEAR(vectors[7][i], fetched_vector[i], 1e-2f);
  }

  // results before close, for comparison after reopen
  auto query_param = FlatQueryParamBuilder().with_topk(kTopK).build();
  SearchResult before;
  VectorData query_data;
  query_data.vector = DenseVector{vectors[7].data()};
  ASSERT_EQ(0, index->search(query_data, query_param, &before));
  auto before_rows = ScoredRows(&before);
  ASSERT_EQ(0, index->close());

  // reopen with identical params: turbo path must be rebuilt from the param
  auto reopened_index = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, reopened_index);
  ASSERT_EQ(0, reopened_index->open(
                   index_name, {StorageOptions::StorageType::kMMAP, false}));
  SearchResult after;
  ASSERT_EQ(0, reopened_index->search(query_data, query_param, &after));
  auto after_rows = ScoredRows(&after);
  ASSERT_EQ(before_rows.size(), after_rows.size());
  for (size_t i = 0; i < before_rows.size(); ++i) {
    EXPECT_EQ(before_rows[i].first, after_rows[i].first);
    EXPECT_FLOAT_EQ(before_rows[i].second, after_rows[i].second);
  }
  VectorDataBuffer fetched_after;
  ASSERT_EQ(0, reopened_index->fetch(7, &fetched_after));
  const auto *fetched_after_vector = reinterpret_cast<const float *>(
      std::get<DenseVectorBuffer>(fetched_after.vector_buffer).data.data());
  for (uint32_t i = 0; i < kDimension; ++i) {
    EXPECT_NEAR(vectors[7][i], fetched_after_vector[i], 1e-2f);
  }
  ASSERT_EQ(0, reopened_index->close());

  zvec::test_util::RemoveTestFiles(index_name);
}

TEST(FlatTurboInt8Index, L2AddSearchReopen) {
  const std::string index_name{"flat_turbo_int8_l2.index"};
  zvec::test_util::RemoveTestFiles(index_name);
  auto vectors = RandomVectors(kVectorCount, kDimension);

  auto param = MakeParam(MetricType::kL2sq, QuantizerType::kInt8);
  auto index = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, index);
  ASSERT_EQ(
      0, index->open(index_name, {StorageOptions::StorageType::kMMAP, true}));
  for (size_t i = 0; i < vectors.size(); ++i) {
    VectorData vector_data;
    vector_data.vector = DenseVector{vectors[i].data()};
    ASSERT_EQ(0, index->add(vector_data, static_cast<uint32_t>(i)));
  }
  ASSERT_EQ(0, index->train());

  CheckTopKAgainstBruteForce(index.get(), vectors, vectors[42],
                             L2SquaredDistance);

  SearchResult before;
  auto query_param = FlatQueryParamBuilder().with_topk(kTopK).build();
  VectorData query_data;
  query_data.vector = DenseVector{vectors[42].data()};
  ASSERT_EQ(0, index->search(query_data, query_param, &before));
  auto before_rows = ScoredRows(&before);
  ASSERT_EQ(0, index->close());

  auto reopened_index = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, reopened_index);
  ASSERT_EQ(0, reopened_index->open(
                   index_name, {StorageOptions::StorageType::kMMAP, false}));
  SearchResult after;
  ASSERT_EQ(0, reopened_index->search(query_data, query_param, &after));
  auto after_rows = ScoredRows(&after);
  ASSERT_EQ(before_rows.size(), after_rows.size());
  for (size_t i = 0; i < before_rows.size(); ++i) {
    EXPECT_EQ(before_rows[i].first, after_rows[i].first);
    EXPECT_FLOAT_EQ(before_rows[i].second, after_rows[i].second);
  }
  ASSERT_EQ(0, reopened_index->close());

  zvec::test_util::RemoveTestFiles(index_name);
}

// enable_rotate and inner product cannot be expressed by the turbo
// quantizer; INT8 must fall back to the legacy converter path and still work
// end to end.
void CheckLegacyFallback(MetricType metric, bool enable_rotate) {
  const std::string index_name{"flat_int8_legacy_fallback.index"};
  zvec::test_util::RemoveTestFiles(index_name);
  auto vectors = RandomVectors(kVectorCount, kDimension);

  auto param = MakeParam(metric, QuantizerType::kInt8, enable_rotate);
  auto index = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, index);
  ASSERT_EQ(
      0, index->open(index_name, {StorageOptions::StorageType::kMMAP, true}));
  for (size_t i = 0; i < vectors.size(); ++i) {
    VectorData vector_data;
    vector_data.vector = DenseVector{vectors[i].data()};
    ASSERT_EQ(0, index->add(vector_data, static_cast<uint32_t>(i)));
  }
  ASSERT_EQ(0, index->train());

  auto got = RunSearch(index.get(), vectors[7]);
  ASSERT_EQ(kTopK, got.rows.size());
  EXPECT_EQ(7u, got.rows[0].first);
  ASSERT_EQ(0, index->close());

  zvec::test_util::RemoveTestFiles(index_name);
}

TEST(FlatTurboInt8Index, RotateFallsBackToLegacyConverter) {
  CheckLegacyFallback(MetricType::kCosine, /*enable_rotate=*/true);
}

TEST(FlatTurboInt8Index, InnerProductFallsBackToLegacyConverter) {
  CheckLegacyFallback(MetricType::kInnerProduct, /*enable_rotate=*/false);
}

TEST(FlatTurboInt8Index, RotateReopenOfTurboLayoutRejected) {
  const std::string index_name{"flat_turbo_int8_cross.index"};
  zvec::test_util::RemoveTestFiles(index_name);
  auto vectors = RandomVectors(kVectorCount, kDimension);

  // INT8 (no rotate) writes the turbo quantizer layout
  auto turbo_param = MakeParam(MetricType::kCosine, QuantizerType::kInt8);
  auto index = IndexFactory::CreateAndInitIndex(*turbo_param);
  ASSERT_NE(nullptr, index);
  ASSERT_EQ(
      0, index->open(index_name, {StorageOptions::StorageType::kMMAP, true}));
  for (size_t i = 0; i < vectors.size(); ++i) {
    VectorData vector_data;
    vector_data.vector = DenseVector{vectors[i].data()};
    ASSERT_EQ(0, index->add(vector_data, static_cast<uint32_t>(i)));
  }
  ASSERT_EQ(0, index->train());
  ASSERT_EQ(0, index->close());

  // INT8 + enable_rotate builds the legacy converter meta instead; the
  // stored turbo layout must be rejected by the open-time meta guard rather
  // than silently miscomputed. The rejection also proves the plain INT8
  // create above routed to the turbo quantizer (a legacy layout would match
  // these reopen params and open successfully).
  auto legacy_param = MakeParam(MetricType::kCosine, QuantizerType::kInt8,
                                /*enable_rotate=*/true);
  auto legacy_index = IndexFactory::CreateAndInitIndex(*legacy_param);
  ASSERT_NE(nullptr, legacy_index);
  EXPECT_NE(0, legacy_index->open(index_name,
                                  {StorageOptions::StorageType::kMMAP, false}));

  zvec::test_util::RemoveTestFiles(index_name);
}

namespace {

// Builds an INT8 flat index file the way pre-turbo versions persisted it:
// through the converter/reformer pipeline, whose stored meta carries no
// quantizer attachment.
void BuildLegacyInt8IndexFile(const std::string &path, MetricType metric,
                              const std::vector<std::vector<float>> &vectors) {
  namespace core = zvec::core;
  core::IndexMeta meta(core::IndexMeta::DataType::DT_FP32, kDimension);
  meta.set_meta_type(core::IndexMeta::MetaType::MT_DENSE);
  meta.set_metric(metric == MetricType::kCosine ? "Cosine" : "SquaredEuclidean",
                  0, zvec::ailego::Params());
  const std::string converter_name = metric == MetricType::kCosine
                                         ? "CosineInt8Converter"
                                         : "Int8StreamingConverter";
  meta.set_converter(converter_name, 0, zvec::ailego::Params());
  auto converter = core::IndexFactory::CreateConverter(converter_name);
  ASSERT_NE(nullptr, converter);
  ASSERT_EQ(0, converter->init(meta, zvec::ailego::Params()));
  core::IndexMeta quantized_meta = converter->meta();
  ASSERT_FALSE(quantized_meta.reformer_name().empty());
  auto reformer =
      core::IndexFactory::CreateReformer(quantized_meta.reformer_name());
  ASSERT_NE(nullptr, reformer);
  ASSERT_EQ(0, reformer->init(quantized_meta.reformer_params()));

  auto streamer = core::IndexFactory::CreateStreamer("FlatStreamer");
  ASSERT_NE(nullptr, streamer);
  ASSERT_EQ(0, streamer->init(quantized_meta, zvec::ailego::Params()));
  auto storage = core::IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ASSERT_EQ(0, storage->init(zvec::ailego::Params()));
  ASSERT_EQ(0, storage->open(path, true));
  ASSERT_EQ(0, streamer->open(storage));

  auto context = streamer->create_context();
  ASSERT_NE(nullptr, context);
  core::IndexQueryMeta fp32_meta(core::IndexMeta::DT_FP32, kDimension);
  for (size_t i = 0; i < vectors.size(); ++i) {
    std::string converted;
    core::IndexQueryMeta new_meta;
    ASSERT_EQ(0, reformer->convert(vectors[i].data(), fp32_meta, &converted,
                                   &new_meta));
    ASSERT_EQ(0,
              streamer->add_with_id_impl(static_cast<uint32_t>(i),
                                         converted.data(), new_meta, context));
  }
  ASSERT_EQ(0, streamer->flush(0));
  ASSERT_EQ(0, streamer->close());
  ASSERT_EQ(0, storage->close());
}

// A legacy INT8 index must reopen through the interface with the same
// params: FlatIndex::open detects the missing quantizer attachment in the
// persisted meta and falls back to the converter pipeline instead of
// failing the streamer's open-time meta guard.
void CheckLegacyLayoutReopen(MetricType metric) {
  const std::string index_name{"flat_int8_legacy_layout.index"};
  zvec::test_util::RemoveTestFiles(index_name);
  auto vectors = RandomVectors(kVectorCount, kDimension);
  BuildLegacyInt8IndexFile(index_name, metric, vectors);
  if (::testing::Test::HasFatalFailure()) {
    return;
  }

  auto param = MakeParam(metric, QuantizerType::kInt8);
  auto index = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, index);
  ASSERT_EQ(
      0, index->open(index_name, {StorageOptions::StorageType::kMMAP, false}));

  auto got = RunSearch(index.get(), vectors[7]);
  ASSERT_EQ(kTopK, got.rows.size());
  EXPECT_EQ(7u, got.rows[0].first);

  // fetch must revert through the legacy reformer
  VectorDataBuffer fetched;
  ASSERT_EQ(0, index->fetch(7, &fetched));
  const auto *fetched_vector = reinterpret_cast<const float *>(
      std::get<DenseVectorBuffer>(fetched.vector_buffer).data.data());
  for (uint32_t i = 0; i < kDimension; ++i) {
    EXPECT_NEAR(vectors[7][i], fetched_vector[i], 5e-2f);
  }

  // legacy indexes stay writable after the fallback
  VectorData vector_data;
  vector_data.vector = DenseVector{vectors[3].data()};
  ASSERT_EQ(0, index->add(vector_data, static_cast<uint32_t>(kVectorCount)));
  ASSERT_EQ(0, index->close());

  zvec::test_util::RemoveTestFiles(index_name);
}

}  // namespace

TEST(FlatTurboInt8Index, CosineLegacyLayoutReopenFallsBack) {
  CheckLegacyLayoutReopen(MetricType::kCosine);
}

TEST(FlatTurboInt8Index, L2LegacyLayoutReopenFallsBack) {
  CheckLegacyLayoutReopen(MetricType::kL2sq);
}
