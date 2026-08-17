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

#include "flat/flat_builder.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <future>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <turbo/quantizer/quantizer.h>
#include <zvec/ailego/container/vector.h>
#include <zvec/core/framework/index_streamer.h>
#include "tests/test_util.h"

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

using namespace zvec::core;
using namespace zvec::ailego;
using namespace std;

static inline size_t RandomDimension() {
  std::mt19937 gen((std::random_device())());
  return (std::uniform_int_distribution<size_t>(1, 129))(gen);
}

static size_t DIMENSION = RandomDimension();
class FlatBuilderTest : public testing::Test {
 protected:
  void SetUp() override;
  void TearDown() override;

 public:
  static std::string dir_;
  static IndexMeta meta_;
};

std::string FlatBuilderTest ::dir_("flat_builder_test/");
IndexMeta FlatBuilderTest::meta_;

void FlatBuilderTest::SetUp() {
  meta_.set_meta(IndexMeta::DataType::DT_FP32, DIMENSION);
  meta_.set_metric("SquaredEuclidean", 0, Params());
  meta_.set_major_order(IndexMeta::MO_COLUMN);
}

//! self-check column-major and row-major search.
void FlatBuilderTest::TearDown() {
  zvec::test_util::RemoveTestPath(dir_);
}

void build_process(IndexBuilder::Pointer &builder,
                   IndexHolder::Pointer holder) {
  Params params;
  ASSERT_EQ(0, builder->init(FlatBuilderTest::meta_, params));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);

  std::string path = FlatBuilderTest::dir_ + "TestGeneral";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  auto &stats = builder->stats();
  ASSERT_EQ(0UL, stats.trained_count());
  ASSERT_EQ(0UL, stats.discarded_count());
}

TEST_F(FlatBuilderTest, TestInitSuccess) {
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("FlatBuilder");
  ASSERT_NE(builder, nullptr);
  Params params;
  ASSERT_EQ(0, builder->init(meta_, params));
}

TEST_F(FlatBuilderTest, TestInitFailedWithInvalidMeasure) {
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("FlatBuilder");
  meta_.set_meta(IndexMeta::DataType::DT_FP32, DIMENSION);
  meta_.set_metric("invalid", 0, Params());
  Params params;
  int ret = builder->init(meta_, params);
  EXPECT_EQ(IndexError_InvalidArgument, ret);
}

TEST_F(FlatBuilderTest, TestInt8InvalidColumnMajor) {
  size_t dim = (DIMENSION + 3) / 4 * 4;
  meta_.set_meta(IndexMeta::DataType::DT_INT8, dim + 2);
  meta_.set_metric("SquaredEuclidean", 0, Params());
  meta_.set_major_order(IndexMeta::MO_COLUMN);
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("FlatBuilder");
  ASSERT_NE(builder, nullptr);

  ASSERT_EQ(IndexMeta::MO_COLUMN, meta_.major_order());
  Params params;
  ASSERT_NE(0, builder->init(meta_, params));
}

TEST_F(FlatBuilderTest, TestInt8WithRandomDimension) {
  size_t dim = DIMENSION;
  meta_.set_meta(IndexMeta::DataType::DT_INT8, dim);
  meta_.set_metric("SquaredEuclidean", 0, Params());
  meta_.set_major_order(IndexMeta::MO_UNDEFINED);
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("FlatBuilder");
  ASSERT_NE(builder, nullptr);

  Params params;
  ASSERT_EQ(0, builder->init(meta_, params));
}

TEST_F(FlatBuilderTest, TestBuildWithRowMajor) {
  meta_.set_metric("SquaredEuclidean", 0, Params());
  meta_.set_major_order(IndexMeta::MO_ROW);
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("FlatBuilder");
  ASSERT_NE(builder, nullptr);
  Params params;
  ASSERT_EQ(0, builder->init(meta_, params));
  std::string path = dir_ + "TestGeneral";

  auto holder =
      std::make_shared<OnePassIndexHolder<IndexMeta::DT_FP32>>(DIMENSION);
  size_t doc_cnt = 2000UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(DIMENSION);
    for (size_t j = 0; j < DIMENSION; ++j) {
      vec[j] = i;
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  int ret = builder->train(holder);
  EXPECT_EQ(0, ret);

  ret = builder->build(holder);
  EXPECT_EQ(0, ret);
}

TEST_F(FlatBuilderTest, TestInt8BuildWithRowMajor) {
  meta_.set_metric("SquaredEuclidean", 0, Params());
  meta_.set_meta(IndexMeta::DT_INT8, DIMENSION);
  meta_.set_major_order(IndexMeta::MO_ROW);
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("FlatBuilder");
  ASSERT_NE(builder, nullptr);
  Params params;
  ASSERT_EQ(0, builder->init(meta_, params));
  std::string path = dir_ + "TestGeneral";

  auto holder =
      std::make_shared<OnePassIndexHolder<IndexMeta::DT_INT8>>(DIMENSION);
  size_t doc_cnt = 128UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<int8_t> vec(DIMENSION);
    for (size_t j = 0; j < DIMENSION; ++j) {
      vec[j] = (int8_t)(i % 128);
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  int ret = builder->train(holder);
  EXPECT_EQ(0, ret);

  ret = builder->build(holder);
  EXPECT_EQ(0, ret);
}

TEST_F(FlatBuilderTest, TestBuildWithColumnMajor) {
  meta_.set_meta(IndexMeta::DataType::DT_FP32, DIMENSION);
  meta_.set_metric("SquaredEuclidean", 0, Params());
  meta_.set_major_order(IndexMeta::MO_COLUMN);
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("FlatBuilder");
  ASSERT_NE(builder, nullptr);
  Params params;
  ASSERT_EQ(0, builder->init(meta_, params));
  std::string path = dir_ + "TestGeneral";

  auto holder =
      std::make_shared<OnePassIndexHolder<IndexMeta::DT_FP32>>(DIMENSION);
  size_t doc_cnt = 2000UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(DIMENSION);
    for (size_t j = 0; j < DIMENSION; ++j) {
      vec[j] = i;
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  int ret = builder->train(holder);
  EXPECT_EQ(0, ret);

  ret = builder->build(holder);
  EXPECT_EQ(0, ret);
}

TEST_F(FlatBuilderTest, TestInt8BuildWithColumnMajor) {
  size_t dim = (DIMENSION + 3) / 4 * 4;
  meta_.set_meta(IndexMeta::DataType::DT_INT8, dim);
  meta_.set_metric("SquaredEuclidean", 0, Params());
  meta_.set_major_order(IndexMeta::MO_COLUMN);
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("FlatBuilder");
  ASSERT_NE(builder, nullptr);
  Params params;
  ASSERT_EQ(0, builder->init(meta_, params));
  std::string path = dir_ + "TestGeneral";

  auto holder = std::make_shared<OnePassIndexHolder<IndexMeta::DT_INT8>>(dim);
  size_t doc_cnt = 128UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<int8_t> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = (int8_t)(i % 128);
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  int ret = builder->train(holder);
  EXPECT_EQ(0, ret);

  ret = builder->build(holder);
  EXPECT_EQ(0, ret);
}

TEST_F(FlatBuilderTest, TestWithRowMajor) {
  meta_.set_meta(IndexMeta::DataType::DT_FP32, DIMENSION);
  meta_.set_metric("SquaredEuclidean", 0, Params());
  meta_.set_major_order(IndexMeta::MO_ROW);
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("FlatBuilder");
  ASSERT_NE(builder, nullptr);
  Params params;
  std::string path = dir_ + "TestGeneral";

  auto holder =
      std::make_shared<OnePassIndexHolder<IndexMeta::DT_FP32>>(DIMENSION);
  size_t doc_cnt = 2000UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(DIMENSION);
    for (size_t j = 0; j < DIMENSION; ++j) {
      vec[j] = i;
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }
  build_process(builder, holder);

  // cleanup and rebuild
  ASSERT_EQ(0, builder->cleanup());
}

TEST_F(FlatBuilderTest, TestInt8WithRowMajor) {
  meta_.set_meta(IndexMeta::DataType::DT_INT8, DIMENSION);
  meta_.set_metric("SquaredEuclidean", 0, Params());
  meta_.set_major_order(IndexMeta::MO_ROW);
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("FlatBuilder");
  ASSERT_NE(builder, nullptr);
  Params params;
  std::string path = dir_ + "TestGeneral";

  auto holder =
      std::make_shared<OnePassIndexHolder<IndexMeta::DT_INT8>>(DIMENSION);
  size_t doc_cnt = 128UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<int8_t> vec(DIMENSION);
    for (size_t j = 0; j < DIMENSION; ++j) {
      vec[j] = (int8_t)(i % 128);
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }
  build_process(builder, holder);

  // cleanup and rebuild
  ASSERT_EQ(0, builder->cleanup());
}

TEST_F(FlatBuilderTest, TestWithColumnMajor) {
  meta_.set_meta(IndexMeta::DataType::DT_FP32, DIMENSION);
  meta_.set_metric("SquaredEuclidean", 0, Params());
  meta_.set_major_order(IndexMeta::MO_COLUMN);
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("FlatBuilder");
  ASSERT_NE(builder, nullptr);
  Params params;
  std::string path = dir_ + "TestGeneral";

  auto holder =
      std::make_shared<OnePassIndexHolder<IndexMeta::DT_FP32>>(DIMENSION);
  size_t doc_cnt = 2000UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(DIMENSION);
    for (size_t j = 0; j < DIMENSION; ++j) {
      vec[j] = i;
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }
  build_process(builder, holder);

  // cleanup and rebuild
  ASSERT_EQ(0, builder->cleanup());
}

TEST_F(FlatBuilderTest, TestInt8WithColumnMajor) {
  size_t dim = (DIMENSION + 3) / 4 * 4;
  meta_.set_meta(IndexMeta::DataType::DT_INT8, dim);
  meta_.set_metric("SquaredEuclidean", 0, Params());
  meta_.set_major_order(IndexMeta::MO_COLUMN);
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("FlatBuilder");
  ASSERT_NE(builder, nullptr);
  Params params;
  std::string path = dir_ + "TestGeneral";

  auto holder = std::make_shared<OnePassIndexHolder<IndexMeta::DT_INT8>>(dim);
  size_t doc_cnt = 128UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<int8_t> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = (int8_t)(i % 128);
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }
  build_process(builder, holder);

  // cleanup and rebuild
  ASSERT_EQ(0, builder->cleanup());
}

// Store complete encoded records, including packed INT4 codes and tails,
// while reporting the encoding's datatype and logical dimension.
struct QuantizedHolder : public MultiPassNumericalIndexHolder<uint8_t> {
  explicit QuantizedHolder(const IndexMeta &meta)
      : MultiPassNumericalIndexHolder<uint8_t>(meta.element_size()),
        meta_(meta) {}

  size_t dimension() const override {
    return meta_.dimension();
  }

  IndexMeta::DataType data_type() const override {
    return meta_.data_type();
  }

 private:
  IndexMeta meta_;
};

static const char *const kTurboQuantizers[] = {
    "Fp32Quantizer", "Fp16Quantizer", "Int8Quantizer", "Int4Quantizer"};

static std::vector<std::vector<float>> RandomData(size_t count, size_t dim) {
  std::mt19937 gen(2026);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<std::vector<float>> data(count);
  for (auto &vec : data) {
    vec.resize(dim);
    for (auto &v : vec) {
      v = dist(gen);
    }
  }
  return data;
}

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
}

//! Create a quantizer and build a flat index with quantized datapoints
static void BuildQuantizedIndex(const std::string &metric_name,
                                const std::vector<std::vector<float>> &data,
                                size_t dim, const std::string &path,
                                std::shared_ptr<zvec::turbo::Quantizer> *out,
                                const char *quantizer_name = "Fp32Quantizer") {
  IndexMeta raw_meta;
  raw_meta.set_meta(IndexMeta::DataType::DT_FP32, dim);
  raw_meta.set_metric(metric_name, 0, Params());

  auto quantizer = IndexFactory::CreateQuantizer(quantizer_name);
  ASSERT_NE(nullptr, quantizer);
  ASSERT_EQ(0, quantizer->init(raw_meta, Params()));

  IndexMeta meta = quantizer->meta();
  size_t code_bytes = quantizer->quantized_datapoint_vector_length();
  ASSERT_EQ(code_bytes, meta.element_size());
  meta.set_quantizer(quantizer_name, 0, Params());
  meta.set_major_order(IndexMeta::MO_ROW);

  auto holder = std::make_shared<QuantizedHolder>(meta);
  NumericalVector<uint8_t> vec(code_bytes);
  for (size_t i = 0; i < data.size(); ++i) {
    quantizer->quantize_data(data[i].data(), &vec[0]);
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  auto builder = IndexFactory::CreateBuilder("FlatBuilder");
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(nullptr, builder);
  ASSERT_NE(nullptr, dumper);

  Params params;
  ASSERT_EQ(0, builder->init(meta, params, quantizer));
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, IndexBuilder::TrainBuildAndDump(builder, holder, dumper));
  ASSERT_EQ(0, dumper->close());
  *out = quantizer;
}

static void LoadQuantizedIndex(
    const std::string &path,
    const std::shared_ptr<zvec::turbo::Quantizer> &quantizer,
    IndexSearcher::Pointer &searcher) {
  searcher = IndexFactory::CreateSearcher("FlatSearcher");
  auto storage = IndexFactory::CreateStorage("MMapFileReadStorage");
  ASSERT_NE(nullptr, searcher);
  ASSERT_NE(nullptr, storage);

  Params params;
  ASSERT_EQ(0, searcher->init(params, quantizer));
  ASSERT_EQ(0, storage->open(path, false));
  ASSERT_EQ(0, searcher->load(storage, IndexMetric::Pointer()));
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

namespace {
class ThresholdContext : public IndexContext {
 public:
  using IndexContext::copy_common_query_options_from;
  using IndexContext::update_index_metric;
  using IndexContext::update_index_quantizer;
  void set_topk(uint32_t) override {}
  const IndexDocumentList &result() const override {
    return docs_;
  }
  IndexDocumentList *mutable_result(size_t) override {
    return &docs_;
  }

 private:
  IndexDocumentList docs_;
};
}  // namespace

TEST_F(FlatBuilderTest, ThresholdBackendRebinding) {
  IndexMeta meta(IndexMeta::DT_FP32, 32);
  meta.set_metric("InnerProduct", 0, Params());
  auto ip_metric = IndexFactory::CreateMetric("InnerProduct");
  ASSERT_NE(nullptr, ip_metric);
  ASSERT_EQ(0, ip_metric->init(meta, Params()));
  auto ip_quantizer = IndexFactory::CreateQuantizer("Fp16Quantizer");
  ASSERT_NE(nullptr, ip_quantizer);
  ASSERT_EQ(0, ip_quantizer->init(meta, Params()));

  ThresholdContext context;
  context.update_index_quantizer(ip_quantizer);
  EXPECT_FALSE(context.threshold_set());
  EXPECT_FLOAT_EQ(std::numeric_limits<float>::max(), context.threshold());
  for (int repeat = 0; repeat < 2; ++repeat) {
    context.set_threshold(0.5f);
    context.update_index_quantizer(ip_quantizer);
    EXPECT_FLOAT_EQ(0.5f, context.raw_threshold());
    EXPECT_FLOAT_EQ(-0.5f, context.threshold());
  }

  for (const char *metric_name : {"SquaredEuclidean", "Cosine"}) {
    meta.set_metric(metric_name, 0, Params());
    auto metric = IndexFactory::CreateMetric(metric_name);
    ASSERT_NE(nullptr, metric);
    ASSERT_EQ(0, metric->init(meta, Params()));
    context.update_index_metric(metric);
    EXPECT_FLOAT_EQ(0.5f, context.threshold());
    context.update_index_metric(ip_metric);
    EXPECT_FLOAT_EQ(-0.5f, context.threshold());
    for (const char *name : kTurboQuantizers) {
      auto quantizer = IndexFactory::CreateQuantizer(name);
      ASSERT_NE(nullptr, quantizer);
      ASSERT_EQ(0, quantizer->init(meta, Params()));
      context.update_index_quantizer(quantizer);
      EXPECT_FLOAT_EQ(0.5f, context.threshold());
      // Copy from an IP context into another score space using the raw radius.
      ThresholdContext source;
      source.update_index_metric(ip_metric);
      source.set_threshold(0.75f);
      context.copy_common_query_options_from(source);
      EXPECT_FLOAT_EQ(0.75f, context.threshold());
      EXPECT_FLOAT_EQ(0.75f, context.raw_threshold());
      source.reset_threshold();
      context.copy_common_query_options_from(source);
      EXPECT_FALSE(context.threshold_set());
      EXPECT_FLOAT_EQ(std::numeric_limits<float>::max(), context.threshold());
      context.set_threshold(0.5f);
      context.update_index_metric(ip_metric);
      EXPECT_FLOAT_EQ(-0.5f, context.threshold());
    }
  }

  context.update_index_quantizer(ip_quantizer);
  std::weak_ptr<zvec::turbo::Quantizer> weak = ip_quantizer;
  ip_quantizer.reset();
  EXPECT_TRUE(weak.expired());
  EXPECT_FALSE(context.threshold_is_valid());
  context.set_threshold(0.25f);
  EXPECT_FLOAT_EQ(0.25f, context.raw_threshold());
  EXPECT_EQ(-std::numeric_limits<float>::infinity(), context.threshold());
  // Rebinding recovers even after the previous converter has expired.
  context.update_index_metric(ip_metric);
  EXPECT_TRUE(context.threshold_is_valid());
  EXPECT_FLOAT_EQ(-0.25f, context.threshold());
  context.reset_threshold();
  context.update_index_metric(ip_metric);
  EXPECT_FLOAT_EQ(std::numeric_limits<float>::max(), context.threshold());
}

TEST_F(FlatBuilderTest, RadiusAcrossFlatBackendsAndQueryModes) {
  constexpr size_t dimension = 32;
  const std::vector<float> values{1.0f, 0.5f, 0.25f, -0.25f, -0.5f, -1.0f};
  std::vector<std::vector<float>> vectors(values.size(),
                                          std::vector<float>(dimension, 0));
  for (size_t i = 0; i < values.size(); ++i) vectors[i][0] = values[i];
  // Reuse the same context across different owners, encodings and metrics.
  IndexContext::Pointer contexts[2];
  auto reusable_streamer = IndexFactory::CreateStreamer("FlatStreamer");
  ASSERT_NE(nullptr, reusable_streamer);
  const std::pair<const char *, const char *> cases[] = {
      {"Fp32Quantizer", "InnerProduct"},     {nullptr, "SquaredEuclidean"},
      {"Fp16Quantizer", "InnerProduct"},     {nullptr, "InnerProduct"},
      {"Fp32Quantizer", "SquaredEuclidean"}, {"Fp32Quantizer", "Cosine"},
      {"Fp16Quantizer", "SquaredEuclidean"}, {"Fp16Quantizer", "Cosine"},
      {"Int8Quantizer", "SquaredEuclidean"}, {"Int8Quantizer", "Cosine"},
      {"Int4Quantizer", "SquaredEuclidean"}, {"Int4Quantizer", "Cosine"},
      {"Fp32Quantizer", "InnerProduct"}};
  for (const auto &test_case : cases) {
    const char *name = test_case.first;
    const std::string metric = test_case.second;
    const bool ip = metric == "InnerProduct", cosine = metric == "Cosine";
    IndexMeta meta(IndexMeta::DT_FP32, dimension);
    meta.set_metric(metric, 0, Params());
    meta.set_major_order(IndexMeta::MO_ROW);
    const std::string suffix = std::string(name ? name : "legacy") + metric;
    const std::string path = dir_ + "radius_" + suffix;
    std::shared_ptr<zvec::turbo::Quantizer> quantizer;
    if (name) {
      BuildQuantizedIndex(metric, vectors, dimension, path, &quantizer, name);
      ASSERT_NE(nullptr, quantizer);
      meta = quantizer->meta();
      meta.set_quantizer(name, 0, Params());
    } else {
      auto holder =
          std::make_shared<MultiPassIndexHolder<IndexMeta::DT_FP32>>(dimension);
      for (size_t i = 0; i < values.size(); ++i) {
        NumericalVector<float> vector(dimension);
        memcpy(&vector[0], vectors[i].data(), dimension * sizeof(float));
        ASSERT_TRUE(holder->emplace(i, vector));
      }
      BuildIndex(meta, holder, path);
    }
    const auto encode = [&quantizer](const std::vector<float> &raw) {
      if (!quantizer)
        return std::string(reinterpret_cast<const char *>(raw.data()),
                           raw.size() * sizeof(float));
      std::string code(quantizer->quantized_query_vector_length(), '\0');
      quantizer->quantize_query(raw.data(), code.data());
      return code;
    };
    IndexQueryMeta qmeta(IndexMeta::DT_FP32, dimension);
    if (quantizer) {
      std::string code;
      const IndexQueryMeta raw_meta(IndexMeta::DT_FP32, dimension);
      ASSERT_EQ(
          0, quantizer->quantize(vectors[0].data(), raw_meta, &code, &qmeta));
    }
    auto query = vectors[0];
    const auto positive = encode(query);
    query[0] = -1.0f;
    const auto batch = positive + encode(query);
    const std::vector<std::vector<uint64_t>> candidates{{0, 1, 3, 4, 5},
                                                        {0, 1, 3, 4, 5}};

    for (int kind = 0; kind < 2; ++kind) {
      SCOPED_TRACE(testing::Message() << suffix << " streamer " << kind);
      IndexSearcher::Pointer searcher;
      IndexStreamer::Pointer streamer;
      IndexRunner *runner = nullptr;
      if (kind == 0) {
        if (quantizer) {
          LoadQuantizedIndex(path, quantizer, searcher);
        } else {
          LoadIndex(path, searcher);
        }
        runner = searcher.get();
      } else {
        // Reinitialize the owner too: a stale legacy metric must not take
        // precedence over its newly selected turbo quantizer.
        streamer = reusable_streamer;
        if (quantizer) {
          ASSERT_EQ(0, streamer->init(meta, Params(), quantizer));
        } else {
          ASSERT_EQ(0, streamer->init(meta, Params()));
        }
        auto storage = IndexFactory::CreateStorage("MMapFileStorage");
        ASSERT_NE(nullptr, storage);
        ASSERT_EQ(0, storage->init(Params()));
        const auto stream_path = path + "_stream";
        zvec::test_util::RemoveTestFiles(stream_path);
        ASSERT_EQ(0, storage->open(stream_path, true));
        ASSERT_EQ(0, streamer->open(storage));
        auto add_context = streamer->create_context();
        for (size_t key = 0; key < values.size(); ++key) {
          std::string code;
          if (quantizer) {
            code.resize(quantizer->quantized_datapoint_vector_length());
            quantizer->quantize_data(vectors[key].data(), code.data());
          } else {
            code = encode(vectors[key]);
          }
          ASSERT_EQ(0,
                    streamer->add_impl(key, code.data(), qmeta, add_context));
        }
        runner = streamer.get();
      }
      ASSERT_NE(nullptr, runner);
      auto &context = contexts[kind];
      if (!context) context = runner->create_context();
      ASSERT_NE(nullptr, context);
      context->set_topk(values.size());
      const std::vector<float> radii =
          ip ? std::vector<float>{0.5f, 0, 2, 0, 1, 0.25f}
             : std::vector<float>{0.3f, 0, 0.8f, 0, 1.8f, 5};
      for (float radius : radii) {
        SCOPED_TRACE(testing::Message() << "radius " << radius);
        if (radius > 0) {
          context->set_threshold(radius);
        } else {
          context->reset_threshold();
        }
        for (bool grouped : {false, true}) {
          context->set_group_params(grouped ? 2 : 0, values.size());
          context->set_group_by(
              [](uint64_t key) { return std::to_string(key % 2); });
          for (bool filtered : {false, true}) {
            if (filtered) {
              context->set_filter([](uint64_t key) { return key == 0; });
            } else {
              context->reset_filter();
            }
            for (int mode = 0; mode < 3; ++mode) {
              SCOPED_TRACE(testing::Message()
                           << "group " << grouped << " filter " << filtered
                           << " mode " << mode);
              if (mode == 0) {
                ASSERT_EQ(0, runner->search_impl(batch.data(), qmeta, context));
              } else if (mode == 1) {
                ASSERT_EQ(0,
                          runner->search_impl(batch.data(), qmeta, 2, context));
              } else {
                ASSERT_EQ(0, runner->search_bf_by_p_keys_impl(
                                 batch.data(), candidates, qmeta, 2, context));
              }
              EXPECT_TRUE(context->threshold_is_valid());
              EXPECT_FLOAT_EQ(
                  radius > 0 ? radius : std::numeric_limits<float>::max(),
                  context->raw_threshold());
              EXPECT_FLOAT_EQ(radius > 0 ? (ip ? -radius : radius)
                                         : std::numeric_limits<float>::max(),
                              context->threshold());
              for (int q = 0; q < (mode == 0 ? 1 : 2); ++q) {
                std::vector<std::pair<uint64_t, float>> actual, expected;
                const auto collect = [&actual](const auto &docs) {
                  for (const auto &doc : docs)
                    actual.emplace_back(doc.key(), doc.score());
                };
                if (grouped) {
                  for (const auto &group : context->group_result(q))
                    collect(group.docs());
                } else {
                  collect(context->result(q));
                }
                const float direction = q == 0 ? 1.0f : -1.0f;
                for (size_t key = 0; key < values.size(); ++key) {
                  if ((filtered && key == 0) || (mode == 2 && key == 2))
                    continue;
                  const float dot = direction * values[key];
                  const float score = ip       ? -dot
                                      : cosine ? (dot > 0 ? 0.0f : 2.0f)
                                               : (direction - values[key]) *
                                                     (direction - values[key]);
                  if (radius == 0 || score <= (ip ? -radius : radius))
                    expected.emplace_back(key, score);
                }
                std::sort(actual.begin(), actual.end());
                ASSERT_EQ(expected.size(), actual.size());
                for (size_t i = 0; i < expected.size(); ++i) {
                  EXPECT_EQ(expected[i].first, actual[i].first);
                  EXPECT_NEAR(expected[i].second, actual[i].second, 1e-5f);
                }
              }
            }
          }
        }
      }
      if (searcher) {
        ASSERT_EQ(0, searcher->unload());
      } else {
        ASSERT_EQ(0, streamer->cleanup());
      }
      EXPECT_EQ(nullptr, runner->create_context());
      EXPECT_EQ(IndexError_NoReady,
                runner->search_impl(batch.data(), qmeta, context));
      EXPECT_EQ(IndexError_NoReady,
                runner->search_impl(batch.data(), qmeta, 2, context));
      EXPECT_EQ(IndexError_NoReady,
                runner->search_bf_by_p_keys_impl(batch.data(), candidates,
                                                 qmeta, 2, context));
    }
  }
}

TEST_F(FlatBuilderTest, TestInitWithTurboQuantizer) {
  constexpr size_t dim = 34;  // Even for packed INT4, with a SIMD tail.
  for (const char *name : kTurboQuantizers) {
    for (const char *metric : {"SquaredEuclidean", "Cosine"}) {
      SCOPED_TRACE(testing::Message() << name << "/" << metric);
      IndexMeta raw_meta(IndexMeta::DT_FP32, dim);
      raw_meta.set_metric(metric, 0, Params());
      auto quantizer = IndexFactory::CreateQuantizer(name);
      ASSERT_NE(nullptr, quantizer);
      ASSERT_EQ(0, quantizer->init(raw_meta, Params()));
      IndexMeta meta = quantizer->meta();
      meta.set_quantizer(name, 0, Params());
      auto builder = IndexFactory::CreateBuilder("FlatBuilder");
      ASSERT_NE(nullptr, builder);

      meta.set_major_order(IndexMeta::MO_COLUMN);
      EXPECT_EQ(IndexError_Unsupported,
                builder->init(meta, Params(), quantizer));
      meta.set_major_order(IndexMeta::MO_ROW);
      Params column_params;
      column_params.set(PARAM_FLAT_COLUMN_MAJOR_ORDER, true);
      EXPECT_EQ(IndexError_Unsupported,
                builder->init(meta, column_params, quantizer));
      EXPECT_EQ(0, builder->init(meta, Params(), quantizer));
      meta.set_major_order(IndexMeta::MO_UNDEFINED);
      EXPECT_EQ(0, builder->init(meta, Params(), quantizer));

      // A null quantizer must still follow legacy metric validation.
      EXPECT_EQ(0, builder->init(raw_meta, Params(), nullptr));
      if (std::string(metric) == "Cosine" &&
          (meta.data_type() == IndexMeta::DT_INT8 ||
           meta.data_type() == IndexMeta::DT_INT4)) {
        EXPECT_EQ(IndexError_InvalidArgument, builder->init(meta, Params()));
        EXPECT_EQ(IndexError_InvalidArgument,
                  builder->init(meta, Params(), nullptr));
      }
      raw_meta.set_metric("UnknownFlatMetric", 0, Params());
      EXPECT_EQ(IndexError_InvalidArgument, builder->init(raw_meta, Params()));
      EXPECT_EQ(IndexError_InvalidArgument,
                builder->init(raw_meta, Params(), nullptr));
    }
  }
}

// Under SquaredEuclidean the FP32 quantizer is an identity transform, so
// the quantizer path must match the plain metric-based searcher.
TEST_F(FlatBuilderTest, TestTurboQuantizerMatchPlainSearcher) {
  // Odd dimension and document count to exercise the tail paths
  const size_t dim = 35;
  const size_t doc_count = 1003;
  const uint32_t topk = 10;
  auto data = RandomData(doc_count, dim);

  // Plain index
  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, dim);
  meta.set_metric("SquaredEuclidean", 0, Params());
  meta.set_major_order(IndexMeta::MO_ROW);
  auto holder = std::make_shared<MultiPassIndexHolder<IndexMeta::DT_FP32>>(dim);
  for (size_t i = 0; i < doc_count; ++i) {
    NumericalVector<float> vec(dim);
    memcpy(&vec[0], data[i].data(), dim * sizeof(float));
    ASSERT_TRUE(holder->emplace(i, vec));
  }
  BuildIndex(meta, holder, dir_ + "plain.index");

  // Quantized index
  std::shared_ptr<zvec::turbo::Quantizer> quantizer;
  BuildQuantizedIndex("SquaredEuclidean", data, dim, dir_ + "quantized.index",
                      &quantizer);

  IndexSearcher::Pointer plain_searcher, turbo_searcher;
  LoadIndex(dir_ + "plain.index", plain_searcher);
  LoadQuantizedIndex(dir_ + "quantized.index", quantizer, turbo_searcher);

  auto plain_ctx = plain_searcher->create_context();
  auto turbo_ctx = turbo_searcher->create_context();
  plain_ctx->set_topk(topk);
  turbo_ctx->set_topk(topk);

  auto queries = RandomData(5, dim);
  for (const auto &query : queries) {
    IndexQueryMeta raw_qmeta(IndexMeta::DT_FP32, dim);
    ASSERT_EQ(0,
              plain_searcher->search_impl(query.data(), raw_qmeta, plain_ctx));

    std::string quantized;
    IndexQueryMeta turbo_qmeta;
    ASSERT_EQ(0, quantizer->quantize(query.data(), raw_qmeta, &quantized,
                                     &turbo_qmeta));
    ASSERT_EQ(0, turbo_searcher->search_impl(quantized.data(), turbo_qmeta,
                                             turbo_ctx));

    auto &expected = plain_ctx->result();
    auto &actual = turbo_ctx->result();
    ASSERT_EQ(expected.size(), actual.size());
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_EQ(expected[i].key(), actual[i].key());
      EXPECT_NEAR(expected[i].score(), actual[i].score(),
                  1e-4f * std::fabs(expected[i].score()) + 1e-5f);
    }
  }

  // Filtered search must agree as well
  plain_ctx->set_filter([](uint64_t key) { return (key % 2 == 0); });
  turbo_ctx->set_filter([](uint64_t key) { return (key % 2 == 0); });
  {
    IndexQueryMeta raw_qmeta(IndexMeta::DT_FP32, dim);
    const auto &query = queries[0];
    ASSERT_EQ(0,
              plain_searcher->search_impl(query.data(), raw_qmeta, plain_ctx));

    std::string quantized;
    IndexQueryMeta turbo_qmeta;
    ASSERT_EQ(0, quantizer->quantize(query.data(), raw_qmeta, &quantized,
                                     &turbo_qmeta));
    ASSERT_EQ(0, turbo_searcher->search_impl(quantized.data(), turbo_qmeta,
                                             turbo_ctx));

    auto &expected = plain_ctx->result();
    auto &actual = turbo_ctx->result();
    ASSERT_EQ(expected.size(), actual.size());
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_EQ(expected[i].key(), actual[i].key());
      EXPECT_EQ(1UL, actual[i].key() % 2);
      EXPECT_NEAR(expected[i].score(), actual[i].score(),
                  1e-4f * std::fabs(expected[i].score()) + 1e-5f);
    }
  }
}

// Both record tails and packed codes must survive build/dump/load, and every
// search entry point must use the supplied quantizer rather than a metric.
TEST_F(FlatBuilderTest, TestTurboQuantizerDistance) {
  constexpr size_t dim = 34;
  constexpr size_t doc_count = 67;  // Two full row batches plus a tail.
  constexpr uint32_t topk = 7;
  constexpr uint32_t query_count = 3;
  auto data = RandomData(doc_count, dim);
  auto queries = RandomData(query_count, dim);
  std::vector<uint64_t> all_keys(doc_count);
  std::vector<std::vector<uint64_t>> restricted_keys(query_count);
  for (size_t i = 0; i < doc_count; ++i) {
    all_keys[i] = i;
    restricted_keys[i % query_count].push_back(i);
  }

  for (const char *name : kTurboQuantizers) {
    for (const char *metric : {"SquaredEuclidean", "Cosine"}) {
      SCOPED_TRACE(testing::Message() << name << "/" << metric);
      const std::string path = dir_ + name + "_" + metric + ".index";
      std::shared_ptr<zvec::turbo::Quantizer> quantizer;
      ASSERT_NO_FATAL_FAILURE(
          BuildQuantizedIndex(metric, data, dim, path, &quantizer, name));
      ASSERT_NE(nullptr, quantizer);
      IndexSearcher::Pointer searcher;
      ASSERT_NO_FATAL_FAILURE(LoadQuantizedIndex(path, quantizer, searcher));
      EXPECT_EQ(name, searcher->meta().quantizer_name());
      EXPECT_EQ(quantizer->meta().data_type(), searcher->meta().data_type());
      EXPECT_EQ(quantizer->meta().element_size(),
                searcher->meta().element_size());
      EXPECT_EQ(IndexMeta::MO_ROW, searcher->meta().major_order());
      for (int mismatch : {0, 1, 2}) {
        SCOPED_TRACE(mismatch);
        const bool wrong_dimension = mismatch == 1;
        const bool asymmetric = mismatch == 2;
        const char *wrong_name =
            asymmetric ? "PqInt8Quantizer"
            : wrong_dimension
                ? name
                : (std::string(name) == "Fp32Quantizer" ? "Fp16Quantizer"
                                                        : "Fp32Quantizer");
        auto incompatible = IndexFactory::CreateQuantizer(wrong_name);
        ASSERT_NE(nullptr, incompatible);
        IndexMeta wrong_meta(IndexMeta::DT_FP32,
                             dim + (wrong_dimension ? 2 : 0));
        wrong_meta.set_metric(metric, 0, Params());
        Params wrong_params;
        if (asymmetric) wrong_params.set("num_chunk", 2);
        ASSERT_EQ(0, incompatible->init(wrong_meta, wrong_params));
        if (asymmetric) {
          ASSERT_NE(incompatible->quantized_query_vector_length(),
                    incompatible->quantized_datapoint_vector_length());
        }
        auto wrong_searcher = IndexFactory::CreateSearcher("FlatSearcher");
        auto storage = IndexFactory::CreateStorage("MMapFileReadStorage");
        ASSERT_NE(nullptr, wrong_searcher);
        ASSERT_NE(nullptr, storage);
        ASSERT_EQ(0, wrong_searcher->init(Params(), incompatible));
        ASSERT_EQ(0, storage->open(path, false));
        EXPECT_EQ(asymmetric ? IndexError_Unsupported : IndexError_Mismatch,
                  wrong_searcher->load(storage, IndexMetric::Pointer()));
      }
      auto context = searcher->create_context();
      ASSERT_NE(nullptr, context);
      context->set_topk(topk);

      std::vector<std::string> codes(doc_count);
      for (size_t i = 0; i < doc_count; ++i) {
        codes[i].resize(quantizer->quantized_datapoint_vector_length());
        quantizer->quantize_data(data[i].data(), codes[i].data());
      }
      std::vector<std::string> query_codes(query_count);
      std::string batch_queries;
      IndexQueryMeta query_meta;
      std::vector<std::vector<float>> scores(query_count,
                                             std::vector<float>(doc_count));
      for (size_t q = 0; q < query_count; ++q) {
        ASSERT_EQ(0,
                  quantizer->quantize(queries[q].data(),
                                      IndexQueryMeta(IndexMeta::DT_FP32, dim),
                                      &query_codes[q], &query_meta));
        ASSERT_EQ(quantizer->quantized_query_vector_length(),
                  query_meta.element_size());
        ASSERT_EQ(query_meta.element_size(), query_codes[q].size());
        batch_queries.append(query_codes[q]);
        for (size_t i = 0; i < doc_count; ++i) {
          scores[q][i] = quantizer->calc_distance_dp_query(
              codes[i].data(), query_codes[q].data());
        }
      }

      // Reject malformed metadata before interpreting query bytes. Keep the
      // full allocation and never enlarge the stride to avoid short buffers.
      auto check_mismatch = [&](const IndexQueryMeta &bad) {
        ASSERT_LE(bad.element_size(), query_meta.element_size());
        EXPECT_EQ(IndexError_Mismatch,
                  searcher->search_impl(batch_queries.data(), bad, context));
        EXPECT_EQ(IndexError_Mismatch,
                  searcher->search_impl(batch_queries.data(), bad, query_count,
                                        context));
        EXPECT_EQ(IndexError_Mismatch,
                  searcher->search_bf_by_p_keys_impl(batch_queries.data(),
                                                     restricted_keys, bad,
                                                     query_count, context));
      };
      auto bad = query_meta;
      bad.set_meta_type(IndexMeta::MT_SPARSE);
      check_mismatch(bad);
      bad = query_meta;
      bad.set_data_type(query_meta.data_type() == IndexMeta::DT_FP32
                            ? IndexMeta::DT_FP16
                            : IndexMeta::DT_FP32);
      check_mismatch(bad);  // set_data_type preserves the original stride.
      bad = query_meta;
      bad.set_dimension(query_meta.dimension() - 2);
      check_mismatch(bad);
      bad.set_extra_meta_size(query_meta.extra_meta_size() +
                              query_meta.element_size() - bad.element_size());
      check_mismatch(bad);  // Same byte length must not hide a wrong dimension.
      // INT4 cannot use a zero unit; dim/2 keeps the packed byte length intact.
      const uint32_t wrong_unit = query_meta.data_type() == IndexMeta::DT_INT4
                                      ? query_meta.dimension() / 2
                                      : query_meta.unit_size() - 1;
      check_mismatch(IndexQueryMeta(
          query_meta.meta_type(), query_meta.data_type(), wrong_unit,
          query_meta.dimension(), query_meta.quantize_type(),
          query_meta.extra_meta_size()));
      if (query_meta.extra_meta_size() != 0) {
        bad = query_meta;
        bad.set_extra_meta_size(query_meta.extra_meta_size() - 1);
        check_mismatch(bad);
      }
      check_mismatch(IndexQueryMeta(
          query_meta.meta_type(), query_meta.data_type(),
          query_meta.unit_size(), query_meta.dimension(),
          query_meta.quantize_type() + 1, query_meta.extra_meta_size()));

      // Filtered scans exercise scalar distances; unfiltered scans use SIMD
      // row batches. Allow equivalent neighbours at a numerical tie boundary.
      for (bool filtered : {false, true}) {
        SCOPED_TRACE(filtered);
        if (filtered) {
          context->set_filter([](uint64_t key) { return key % 2 == 0; });
        } else {
          context->reset_filter();
        }
        auto check_result = [&](const IndexDocumentList &actual, size_t q,
                                const std::vector<uint64_t> &keys) {
          std::vector<std::pair<float, uint64_t>> expected;
          std::vector<bool> allowed(doc_count, false), seen(doc_count, false);
          for (uint64_t key : keys) {
            if (!filtered || key % 2 != 0) {
              expected.emplace_back(scores[q][key], key);
              allowed[key] = true;
            }
          }
          ASSERT_GE(expected.size(), topk);
          std::partial_sort(expected.begin(), expected.begin() + topk,
                            expected.end());
          ASSERT_EQ(topk, actual.size());
          for (size_t i = 0; i < topk; ++i) {
            const uint64_t key = actual[i].key();
            ASSERT_LT(key, doc_count);
            EXPECT_TRUE(allowed[key]);
            EXPECT_FALSE(seen[key]);
            seen[key] = true;
            // FP16 SIMD kernels may accumulate at half precision.
            const float relative =
                std::string(name) == "Fp16Quantizer" ? 5e-3f : 1e-4f;
            const float tolerance =
                relative * std::max(1.0f, std::fabs(expected[i].first));
            EXPECT_NEAR(expected[i].first, scores[q][key], tolerance);
            EXPECT_NEAR(scores[q][key], actual[i].score(), tolerance);
          }
        };

        for (size_t q = 0; q < query_count; ++q) {
          ASSERT_EQ(0, searcher->search_impl(query_codes[q].data(), query_meta,
                                             context));
          check_result(context->result(), q, all_keys);
        }
        ASSERT_EQ(0, searcher->search_impl(batch_queries.data(), query_meta,
                                           query_count, context));
        for (size_t q = 0; q < query_count; ++q) {
          check_result(context->result(q), q, all_keys);
        }
        ASSERT_EQ(0, searcher->search_bf_by_p_keys_impl(
                         batch_queries.data(), restricted_keys, query_meta,
                         query_count, context));
        for (size_t q = 0; q < query_count; ++q) {
          check_result(context->result(q), q, restricted_keys[q]);
        }
      }

      // Drop caller/context references so unload must release the last owner.
      context.reset();
      std::weak_ptr<zvec::turbo::Quantizer> weak_quantizer = quantizer;
      quantizer.reset();
      EXPECT_FALSE(weak_quantizer.expired());
      ASSERT_EQ(0, searcher->unload());
      EXPECT_TRUE(weak_quantizer.expired());

      // Reuse the searcher with a newly initialized quantizer after unload.
      quantizer = IndexFactory::CreateQuantizer(name);
      ASSERT_NE(nullptr, quantizer);
      IndexMeta raw_meta(IndexMeta::DT_FP32, dim);
      raw_meta.set_metric(metric, 0, Params());
      ASSERT_EQ(0, quantizer->init(raw_meta, Params()));
      ASSERT_EQ(0, searcher->init(Params(), quantizer));
      auto storage = IndexFactory::CreateStorage("MMapFileReadStorage");
      ASSERT_NE(nullptr, storage);
      ASSERT_EQ(0, storage->open(path, false));
      ASSERT_EQ(0, searcher->load(storage, IndexMetric::Pointer()));
      context = searcher->create_context();
      ASSERT_NE(nullptr, context);
      context->set_topk(topk);
      ASSERT_EQ(
          0, searcher->search_impl(query_codes[0].data(), query_meta, context));
      EXPECT_EQ(topk, context->result().size());

      context.reset();
      weak_quantizer = quantizer;
      quantizer.reset();
      EXPECT_FALSE(weak_quantizer.expired());
      ASSERT_EQ(0, searcher->cleanup());
      EXPECT_TRUE(weak_quantizer.expired());
      ASSERT_EQ(0, searcher->cleanup());
    }
  }
}

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif
