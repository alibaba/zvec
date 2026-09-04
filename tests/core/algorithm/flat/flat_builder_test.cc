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
#include "tests/test_util.h"

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

using namespace zvec::core;
using namespace zvec::ailego;
using namespace std;

static inline size_t RandomDimension(void) {
  std::mt19937 gen((std::random_device())());
  return (std::uniform_int_distribution<size_t>(1, 129))(gen);
}

static size_t DIMENSION = RandomDimension();
class FlatBuilderTest : public testing::Test {
 protected:
  void SetUp(void) override;
  void TearDown(void) override;

 public:
  static std::string dir_;
  static IndexMeta meta_;
};

std::string FlatBuilderTest ::dir_("flat_builder_test/");
IndexMeta FlatBuilderTest::meta_;

void FlatBuilderTest::SetUp(void) {
  meta_.set_meta(IndexMeta::DataType::DT_FP32, DIMENSION);
  meta_.set_metric("SquaredEuclidean", 0, Params());
  meta_.set_major_order(IndexMeta::MO_COLUMN);
}

//! self-check column-major and row-major search.
void FlatBuilderTest::TearDown(void) {
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

// Holder for turbo-quantized datapoints: rows hold the full encoded size
// while the reported dimension stays the raw dimension.
struct QuantizedHolder : public MultiPassIndexHolder<IndexMeta::DT_FP32> {
  QuantizedHolder(size_t alloc_dim, size_t raw_dim)
      : MultiPassIndexHolder<IndexMeta::DT_FP32>(alloc_dim),
        raw_dim_(raw_dim) {}

  //! Retrieve dimension
  size_t dimension(void) const override {
    return raw_dim_;
  }

 private:
  size_t raw_dim_{0};
};

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
                                std::shared_ptr<zvec::turbo::Quantizer> *out) {
  IndexMeta raw_meta;
  raw_meta.set_meta(IndexMeta::DataType::DT_FP32, dim);
  raw_meta.set_metric(metric_name, 0, Params());

  auto quantizer = IndexFactory::CreateQuantizer("Fp32Quantizer");
  ASSERT_NE(nullptr, quantizer);
  ASSERT_EQ(0, quantizer->init(raw_meta, Params()));

  IndexMeta meta = quantizer->meta();
  size_t code_bytes = quantizer->quantized_datapoint_vector_length();
  ASSERT_EQ(code_bytes, meta.element_size());
  uint32_t alloc_dim = static_cast<uint32_t>(code_bytes / meta.unit_size());
  meta.set_quantizer("Fp32Quantizer", 0, Params());
  meta.set_major_order(IndexMeta::MO_ROW);

  auto holder = std::make_shared<QuantizedHolder>(alloc_dim, dim);
  NumericalVector<float> vec(alloc_dim);
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

TEST_F(FlatBuilderTest, TestInitWithTurboQuantizer) {
  IndexMeta raw_meta;
  raw_meta.set_meta(IndexMeta::DataType::DT_FP32, DIMENSION);
  raw_meta.set_metric("SquaredEuclidean", 0, Params());

  auto quantizer = IndexFactory::CreateQuantizer("Fp32Quantizer");
  ASSERT_NE(nullptr, quantizer);
  ASSERT_EQ(0, quantizer->init(raw_meta, Params()));

  IndexMeta meta = quantizer->meta();
  meta.set_quantizer("Fp32Quantizer", 0, Params());

  // Quantizer distance requires the row major layout
  meta.set_major_order(IndexMeta::MO_COLUMN);
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("FlatBuilder");
  ASSERT_NE(builder, nullptr);
  Params params;
  ASSERT_EQ(IndexError_Unsupported, builder->init(meta, params, quantizer));

  meta.set_major_order(IndexMeta::MO_ROW);
  builder = IndexFactory::CreateBuilder("FlatBuilder");
  ASSERT_NE(builder, nullptr);
  ASSERT_EQ(0, builder->init(meta, params, quantizer));
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

// Under Cosine the FP32 quantizer normalizes the datapoints and appends
// the norm as a record tail; the searcher scores must match the quantizer.
TEST_F(FlatBuilderTest, TestTurboQuantizerCosineDistance) {
  const size_t dim = 24;
  const size_t doc_count = 500;
  const uint32_t topk = 10;
  auto data = RandomData(doc_count, dim);

  std::shared_ptr<zvec::turbo::Quantizer> quantizer;
  BuildQuantizedIndex("Cosine", data, dim, dir_ + "cosine.index", &quantizer);

  IndexSearcher::Pointer searcher;
  LoadQuantizedIndex(dir_ + "cosine.index", quantizer, searcher);

  auto context = searcher->create_context();
  context->set_topk(topk);

  // Quantize the datapoints once for the reference distances
  size_t code_bytes = quantizer->quantized_datapoint_vector_length();
  std::vector<std::string> codes(doc_count);
  for (size_t i = 0; i < doc_count; ++i) {
    codes[i].resize(code_bytes);
    quantizer->quantize_data(data[i].data(), &codes[i][0]);
  }

  auto queries = RandomData(5, dim);
  for (const auto &query : queries) {
    IndexQueryMeta raw_qmeta(IndexMeta::DT_FP32, dim);
    std::string quantized;
    IndexQueryMeta turbo_qmeta;
    ASSERT_EQ(0, quantizer->quantize(query.data(), raw_qmeta, &quantized,
                                     &turbo_qmeta));
    ASSERT_EQ(0, searcher->search_impl(quantized.data(), turbo_qmeta, context));

    // Reference topk with the quantizer scalar distance
    std::vector<std::pair<float, uint64_t>> ref(doc_count);
    for (size_t i = 0; i < doc_count; ++i) {
      ref[i] = {
          quantizer->calc_distance_dp_query(codes[i].data(), quantized.data()),
          i};
    }
    std::partial_sort(ref.begin(), ref.begin() + topk, ref.end());

    auto &actual = context->result();
    ASSERT_EQ(topk, actual.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ(ref[i].second, actual[i].key());
      EXPECT_NEAR(ref[i].first, actual[i].score(),
                  1e-4f * std::fabs(ref[i].first) + 1e-5f);
    }
  }
}

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif