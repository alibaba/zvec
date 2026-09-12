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
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <turbo/quantizer/quantizer.h>
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

FlatIndexParam::Pointer MakeParam(MetricType metric, QuantizerType quantizer,
                                  bool enable_rotate = false,
                                  bool use_contiguous_memory = false) {
  return FlatIndexParamBuilder()
      .with_metric_type(metric)
      .with_data_type(DataType::DT_FP32)
      .with_dimension(kDimension)
      .with_is_sparse(false)
      .with_quantizer_param(QuantizerParam(quantizer, enable_rotate))
      .with_use_contiguous_memory(use_contiguous_memory)
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

// Compare the interface's top-k and scores with a direct per-record scan.
// In particular, INT4's record-tail scoring is not simply the FP32 distance
// between fetched vectors, so use the selected quantizer as the reference.
void CheckTopKAgainstBruteForce(Index *index,
                                const std::vector<std::vector<float>> &vectors,
                                const std::vector<float> &query,
                                const zvec::turbo::Quantizer &quantizer) {
  auto got = RunSearch(index, query);
  ASSERT_EQ(kTopK, got.rows.size());

  std::string query_code(quantizer.quantized_query_vector_length(), '\0');
  std::string code(quantizer.quantized_datapoint_vector_length(), '\0');
  quantizer.quantize_query(query.data(), query_code.data());
  std::vector<float> scores(vectors.size());
  std::vector<std::pair<uint32_t, float>> want;
  for (size_t i = 0; i < vectors.size(); ++i) {
    quantizer.quantize_data(vectors[i].data(), code.data());
    float distance =
        quantizer.calc_distance_dp_query(code.data(), query_code.data());
    want.emplace_back(static_cast<uint32_t>(i), distance);
    if (quantizer.support_score_normalization()) {
      quantizer.normalize_score(&distance);
    }
    scores[i] = distance;
  }
  // Sort by internal distance before converting to caller-facing IP scores.
  std::partial_sort(want.begin(), want.begin() + kTopK, want.end(),
                    [](const auto &lhs, const auto &rhs) {
                      return lhs.second != rhs.second ? lhs.second < rhs.second
                                                      : lhs.first < rhs.first;
                    });
  std::vector<bool> seen(vectors.size(), false);
  for (uint32_t i = 0; i < kTopK; ++i) {
    SCOPED_TRACE(testing::Message() << "rank " << i);
    const auto key = got.rows[i].first;
    ASSERT_LT(key, vectors.size());
    EXPECT_FALSE(seen[key]);
    seen[key] = true;
    const float expected = scores[want[i].first];
    // SIMD single/batch accumulation may reorder nearly tied neighbours.
    const float tolerance = 1e-4f * std::max(1.0f, std::abs(expected));
    EXPECT_NEAR(expected, scores[key], tolerance);
    EXPECT_NEAR(scores[key], got.rows[i].second, tolerance);
  }
}

void CheckFetch(Index *index, uint32_t key, const std::vector<float> &expected,
                QuantizerType quantizer) {
  VectorDataBuffer fetched;
  ASSERT_EQ(0, index->fetch(key, &fetched));
  ASSERT_TRUE(std::holds_alternative<DenseVectorBuffer>(fetched.vector_buffer));
  const auto &data = std::get<DenseVectorBuffer>(fetched.vector_buffer).data;
  ASSERT_EQ(expected.size() * sizeof(float), data.size());
  const auto *actual = reinterpret_cast<const float *>(data.data());
  const auto range = std::minmax_element(expected.begin(), expected.end());
  const float max_abs =
      std::max(std::abs(*range.first), std::abs(*range.second));
  float tolerance = 1e-5f * std::max(1.0f, max_abs);
  // Affine integer rounding is bounded by half a quantization step. Cosine
  // normalization and reconstruction cancel the norm in this bound.
  if (quantizer == QuantizerType::kInt8) {
    tolerance += (*range.second - *range.first) / 508.0f;
  } else if (quantizer == QuantizerType::kInt4) {
    tolerance += (*range.second - *range.first) / 30.0f;
  } else if (quantizer == QuantizerType::kFP16) {
    // Allow one FP16 ULP, including conversions that truncate.
    tolerance += max_abs / 1024.0f;
  }
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(expected[i], actual[i], tolerance) << "component " << i;
  }
}

struct TurboFormat {
  const char *name;
  const char *quantizer_name;
  QuantizerType quantizer;
  zvec::core::IndexMeta::DataType data_type;
};

struct FlatTurboTestCase {
  TurboFormat format;
  MetricType metric;
  bool contiguous;

  std::string name() const {
    return std::string(format.name) + "_" +
           Index::get_metric_name(metric, false) +
           (contiguous ? "_Contiguous" : "_Generic");
  }
};

std::vector<FlatTurboTestCase> FlatTurboTestCases() {
  const TurboFormat formats[] = {
      {"FP32", "Fp32Quantizer", QuantizerType::kNone, DataType::DT_FP32},
      {"FP16", "Fp16Quantizer", QuantizerType::kFP16, DataType::DT_FP16},
      {"INT8", "Int8Quantizer", QuantizerType::kInt8, DataType::DT_INT8},
      {"INT4", "Int4Quantizer", QuantizerType::kInt4, DataType::DT_INT4},
  };
  std::vector<FlatTurboTestCase> cases;
  for (const auto &format : formats) {
    for (const auto metric :
         {MetricType::kL2sq, MetricType::kCosine, MetricType::kInnerProduct}) {
      // Affine integer quantizers use the turbo path for L2/Cosine only.
      if (metric == MetricType::kInnerProduct &&
          (format.quantizer == QuantizerType::kInt8 ||
           format.quantizer == QuantizerType::kInt4)) {
        continue;
      }
      for (bool contiguous : {false, true}) {
        cases.push_back({format, metric, contiguous});
      }
    }
  }
  return cases;
}

class FlatTurboIndex : public testing::TestWithParam<FlatTurboTestCase> {};

}  // namespace

TEST_P(FlatTurboIndex, AddSearchFetchReopen) {
  const auto &test_case = GetParam();
  const auto &format = test_case.format;
  const std::string index_name = "flat_turbo_" + test_case.name() + ".index";
  zvec::test_util::RemoveTestFiles(index_name);
  auto vectors = RandomVectors(kVectorCount, kDimension);
  // Vary the norms to exercise cosine normalization and fetch reconstruction.
  for (size_t i = 0; i < vectors.size(); ++i) {
    for (float &value : vectors[i]) {
      value *= 0.5f + 0.5f * static_cast<float>(i % 4);
    }
  }

  auto param = MakeParam(test_case.metric, format.quantizer, false,
                         test_case.contiguous);
  auto index = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, index);
  ASSERT_EQ(
      0, index->open(index_name, {StorageOptions::StorageType::kMMAP, true}));
  ASSERT_NE(nullptr, index->index_searcher());
  EXPECT_EQ(format.quantizer_name,
            index->index_searcher()->meta().quantizer_name());
  EXPECT_EQ(format.data_type, index->index_searcher()->meta().data_type());

  // Initialize the reference independently from the requested input config,
  // rather than inheriting a possibly incorrect encoding from the index.
  auto reference =
      zvec::core::IndexFactory::CreateQuantizer(format.quantizer_name);
  ASSERT_NE(nullptr, reference);
  zvec::core::IndexMeta input_meta(DataType::DT_FP32, kDimension);
  input_meta.set_metric(Index::get_metric_name(test_case.metric, false), 0,
                        zvec::ailego::Params{});
  ASSERT_EQ(0, reference->init(input_meta, {}));

  for (size_t i = 0; i < vectors.size(); ++i) {
    VectorData vector_data;
    vector_data.vector = DenseVector{vectors[i].data()};
    ASSERT_EQ(0, index->add(vector_data, static_cast<uint32_t>(i)));
  }
  ASSERT_EQ(0, index->train());

  for (uint32_t key : {7u, 101u}) {
    CheckTopKAgainstBruteForce(index.get(), vectors, vectors[key], *reference);
    CheckFetch(index.get(), key, vectors[key], format.quantizer);
  }

  auto before_rows = RunSearch(index.get(), vectors[7]).rows;
  ASSERT_EQ(kTopK, before_rows.size());
  ASSERT_EQ(0, index->close());

  // reopen with identical params: turbo path must be rebuilt from the param
  auto reopened_index = IndexFactory::CreateAndInitIndex(*param);
  ASSERT_NE(nullptr, reopened_index);
  ASSERT_EQ(0, reopened_index->open(
                   index_name, {StorageOptions::StorageType::kMMAP, false}));
  ASSERT_NE(nullptr, reopened_index->index_searcher());
  EXPECT_EQ(format.quantizer_name,
            reopened_index->index_searcher()->meta().quantizer_name());
  EXPECT_EQ(format.data_type,
            reopened_index->index_searcher()->meta().data_type());
  auto after_rows = RunSearch(reopened_index.get(), vectors[7]).rows;
  ASSERT_EQ(before_rows.size(), after_rows.size());
  for (size_t i = 0; i < before_rows.size(); ++i) {
    EXPECT_EQ(before_rows[i].first, after_rows[i].first);
    EXPECT_FLOAT_EQ(before_rows[i].second, after_rows[i].second);
  }
  CheckFetch(reopened_index.get(), 7, vectors[7], format.quantizer);

  // Reopened indexes must also encode new records with the same format.
  auto appended = vectors[42];
  for (float &value : appended) {
    value *= -1.5f;
  }
  const auto appended_key = static_cast<uint32_t>(vectors.size());
  VectorData added;
  added.vector = DenseVector{appended.data()};
  ASSERT_EQ(0, reopened_index->add(added, appended_key));
  vectors.push_back(appended);
  CheckFetch(reopened_index.get(), appended_key, appended, format.quantizer);
  CheckTopKAgainstBruteForce(reopened_index.get(), vectors, appended,
                             *reference);
  ASSERT_EQ(0, reopened_index->close());

  zvec::test_util::RemoveTestFiles(index_name);
}

INSTANTIATE_TEST_SUITE_P(
    Formats, FlatTurboIndex, testing::ValuesIn(FlatTurboTestCases()),
    [](const testing::TestParamInfo<FlatTurboTestCase> &info) {
      return info.param.name();
    });

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
