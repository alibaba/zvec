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
#include <future>
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
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
  void SetUp(void);
  void TearDown(void);

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

TEST_F(FlatBuilderTest, TestBinaryInvalidColumnMajor) {
  size_t dim = (DIMENSION + 31) / 32 * 32;
  meta_.set_metric("Hamming", 0, Params());
  meta_.set_meta(IndexMeta::DT_BINARY32, dim + 2);
  meta_.set_major_order(IndexMeta::MO_COLUMN);
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("FlatBuilder");
  ASSERT_NE(builder, nullptr);
  Params params;
  ASSERT_EQ(0, builder->init(meta_, params));
  std::string path = dir_ + "TestGeneral";
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

TEST_F(FlatBuilderTest, TestBinaryBuildWithRowMajor) {
  size_t dim = (DIMENSION + 31) / 32 * 32;
  meta_.set_metric("Hamming", 0, Params());
  meta_.set_meta(IndexMeta::DT_BINARY32, dim);
  meta_.set_major_order(IndexMeta::MO_ROW);
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("FlatBuilder");
  ASSERT_NE(builder, nullptr);
  Params params;
  ASSERT_EQ(0, builder->init(meta_, params));
  std::string path = dir_ + "TestGeneral";

  auto holder =
      std::make_shared<OnePassIndexHolder<IndexMeta::DT_BINARY32>>(dim);
  size_t doc_cnt = 128UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    BinaryVector<uint32_t> vec(dim);
    for (size_t j = 0; j < dim && j < i; ++j) {
      vec.set(j);
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

TEST_F(FlatBuilderTest, TestBinaryBuildWithColumnMajor) {
  size_t dim = (DIMENSION + 31) / 32 * 32;
  meta_.set_metric("Hamming", 0, Params());
  meta_.set_meta(IndexMeta::DT_BINARY32, dim);
  meta_.set_major_order(IndexMeta::MO_COLUMN);
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("FlatBuilder");
  ASSERT_NE(builder, nullptr);
  Params params;
  ASSERT_EQ(0, builder->init(meta_, params));
  std::string path = dir_ + "TestGeneral";

  auto holder =
      std::make_shared<OnePassIndexHolder<IndexMeta::DT_BINARY32>>(dim);
  size_t doc_cnt = 128UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    BinaryVector<uint32_t> vec(dim);
    for (size_t j = 0; j < dim && j < i; ++j) {
      vec.set(j);
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

TEST_F(FlatBuilderTest, TestBinaryWithRowMajor) {
  size_t dim = (DIMENSION + 31) / 32 * 32;
  meta_.set_metric("Hamming", 0, Params());
  meta_.set_meta(IndexMeta::DT_BINARY32, dim);
  meta_.set_major_order(IndexMeta::MO_ROW);
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("FlatBuilder");
  ASSERT_NE(builder, nullptr);
  Params params;
  std::string path = dir_ + "TestGeneral";

  auto holder =
      std::make_shared<OnePassIndexHolder<IndexMeta::DT_BINARY32>>(dim);
  size_t doc_cnt = 128UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    BinaryVector<uint32_t> vec(dim);
    for (size_t j = 0; j < dim && j < i; ++j) {
      vec.set(j);
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

TEST_F(FlatBuilderTest, TestBinaryWithColumnMajor) {
  size_t dim = (DIMENSION + 31) / 32 * 32;
  meta_.set_metric("Hamming", 0, Params());
  meta_.set_meta(IndexMeta::DT_BINARY32, dim);
  meta_.set_major_order(IndexMeta::MO_COLUMN);
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("FlatBuilder");
  ASSERT_NE(builder, nullptr);
  Params params;
  std::string path = dir_ + "TestGeneral";

  auto holder =
      std::make_shared<OnePassIndexHolder<IndexMeta::DT_BINARY32>>(dim);
  size_t doc_cnt = 128UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    BinaryVector<uint32_t> vec(dim);
    for (size_t j = 0; j < dim && j < i; ++j) {
      vec.set(j);
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }
  build_process(builder, holder);

  // cleanup and rebuild
  ASSERT_EQ(0, builder->cleanup());
}

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif