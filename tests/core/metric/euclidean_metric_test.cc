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
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "zvec/core/framework/index_factory.h"

using namespace zvec;
using namespace zvec::core;

template <typename T>
void CheckBatchDistanceMatchesSingle(const IndexMetric::Pointer &metric,
                                     const std::vector<std::vector<T>> &vecs,
                                     const std::vector<T> &query, size_t dim,
                                     float tolerance) {
  auto single_distance = metric->distance();
  auto batch_distance = metric->batch_distance();
  ASSERT_TRUE(single_distance);
  ASSERT_TRUE(batch_distance);

  std::vector<const void *> ptrs;
  ptrs.reserve(vecs.size());
  for (const auto &vec : vecs) {
    ptrs.push_back(vec.data());
  }

  std::vector<float> batch_result(vecs.size(), 0.0f);
  batch_distance(ptrs.data(), query.data(), vecs.size(), dim,
                 batch_result.data());

  for (size_t i = 0; i < vecs.size(); ++i) {
    float single_result = 0.0f;
    single_distance(vecs[i].data(), query.data(), dim, &single_result);
    EXPECT_NEAR(single_result, batch_result[i], tolerance);
  }
}

TEST(SquaredEuclideanMetric, General) {
  auto metric = IndexFactory::CreateMetric("SquaredEuclidean");
  EXPECT_TRUE(metric);

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_INT16, 64);
  ASSERT_NE(0, metric->init(meta, ailego::Params()));
  meta.set_meta(IndexMeta::DataType::DT_BINARY32, 64);
  ASSERT_EQ(0, metric->init(meta, ailego::Params()));
  meta.set_meta(IndexMeta::DataType::DT_BINARY64, 64);
  ASSERT_EQ(0, metric->init(meta, ailego::Params()));
  meta.set_meta(IndexMeta::DataType::DT_FP16, 64);
  ASSERT_EQ(0, metric->init(meta, ailego::Params()));
  meta.set_meta(IndexMeta::DataType::DT_FP32, 64);
  ASSERT_EQ(0, metric->init(meta, ailego::Params()));
  meta.set_meta(IndexMeta::DataType::DT_INT4, 64);
  ASSERT_EQ(0, metric->init(meta, ailego::Params()));
  meta.set_meta(IndexMeta::DataType::DT_INT8, 64);
  ASSERT_EQ(0, metric->init(meta, ailego::Params()));

  IndexMeta meta2;
  meta2.set_meta(IndexMeta::DataType::DT_BINARY32, 64);
  EXPECT_TRUE(metric->is_matched(meta));
  EXPECT_FALSE(metric->is_matched(meta2));
  EXPECT_TRUE(metric->is_matched(
      meta, IndexQueryMeta(IndexMeta::DataType::DT_INT8, 64)));
  EXPECT_FALSE(metric->is_matched(
      meta, IndexQueryMeta(IndexMeta::DataType::DT_INT8, 63)));

  EXPECT_FALSE(metric->distance_matrix(0, 0));
  EXPECT_FALSE(metric->distance_matrix(3, 5));
  EXPECT_FALSE(metric->distance_matrix(31, 65));
  EXPECT_TRUE(metric->distance_matrix(1, 1));
  EXPECT_TRUE(metric->distance_matrix(2, 1));
  EXPECT_TRUE(metric->distance_matrix(2, 2));
  EXPECT_TRUE(metric->distance_matrix(4, 1));
  EXPECT_TRUE(metric->distance_matrix(4, 2));
  EXPECT_TRUE(metric->distance_matrix(4, 4));
  EXPECT_TRUE(metric->distance_matrix(8, 1));
  EXPECT_TRUE(metric->distance_matrix(8, 2));
  EXPECT_TRUE(metric->distance_matrix(8, 4));
  EXPECT_TRUE(metric->distance_matrix(8, 8));
  EXPECT_FALSE(metric->distance_matrix(8, 32));
  EXPECT_FALSE(metric->distance_matrix(8, 9));
  EXPECT_TRUE(metric->distance_matrix(16, 1));
  EXPECT_TRUE(metric->distance_matrix(16, 2));
  EXPECT_TRUE(metric->distance_matrix(16, 4));
  EXPECT_TRUE(metric->distance_matrix(16, 8));
  EXPECT_TRUE(metric->distance_matrix(16, 16));
  EXPECT_FALSE(metric->distance_matrix(16, 17));
  EXPECT_TRUE(metric->distance_matrix(32, 1));
  EXPECT_TRUE(metric->distance_matrix(32, 2));
  EXPECT_TRUE(metric->distance_matrix(32, 4));
  EXPECT_TRUE(metric->distance_matrix(32, 8));
  EXPECT_TRUE(metric->distance_matrix(32, 16));
  EXPECT_TRUE(metric->distance_matrix(32, 32));

  EXPECT_FALSE(metric->support_normalize());
  float result = 1.0f;
  metric->normalize(&result);
  EXPECT_FLOAT_EQ(1.0f, result);
}

TEST(SquaredEuclideanMetric, BatchDistanceMatchesSingleFp32) {
  auto metric = IndexFactory::CreateMetric("SquaredEuclidean");
  ASSERT_TRUE(metric);

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP32, 8);
  ASSERT_EQ(0, metric->init(meta, ailego::Params()));

  std::vector<std::vector<float>> vecs{
      {1.0f, 2.0f, 3.0f, 4.0f, 1.5f, -2.0f, 0.5f, -1.0f},
      {0.0f, -1.0f, 1.0f, 2.0f, -0.5f, 4.0f, -2.5f, 3.0f},
      {-3.0f, 2.5f, 1.5f, 0.0f, 3.0f, -1.0f, 2.0f, 1.0f},
  };
  std::vector<float> query{2.0f, -1.0f, 0.5f, 3.0f, 1.0f, -2.0f, 4.0f, -1.5f};

  CheckBatchDistanceMatchesSingle(metric, vecs, query, 8, 1e-5f);
}

TEST(SquaredEuclideanMetric, BatchDistanceMatchesSingleFp16) {
  auto metric = IndexFactory::CreateMetric("SquaredEuclidean");
  ASSERT_TRUE(metric);

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_FP16, 8);
  ASSERT_EQ(0, metric->init(meta, ailego::Params()));

  std::vector<std::vector<ailego::Float16>> vecs{
      {0.25f, -0.5f, 1.0f, 2.0f, -1.5f, 0.75f, 1.25f, -0.25f},
      {1.5f, 2.5f, -1.0f, 0.5f, 0.25f, -0.75f, 2.0f, -2.0f},
      {-0.5f, 1.0f, 0.75f, -1.25f, 1.5f, 2.0f, -0.25f, 0.5f},
  };
  std::vector<ailego::Float16> query{
      0.5f, -1.0f, 1.5f, -0.25f, 0.75f, -1.5f, 2.0f, 0.25f};

  CheckBatchDistanceMatchesSingle(metric, vecs, query, 8, 5e-2f);
}

TEST(SquaredEuclideanMetric, BatchDistanceMatchesSingleInt8) {
  auto metric = IndexFactory::CreateMetric("SquaredEuclidean");
  ASSERT_TRUE(metric);

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_INT8, 8);
  ASSERT_EQ(0, metric->init(meta, ailego::Params()));

  std::vector<std::vector<int8_t>> vecs{
      {1, 2, 3, 4, 5, 6, 7, 8},
      {-1, 0, 2, -3, 4, -5, 6, -7},
      {8, 7, 6, 5, 4, 3, 2, 1},
  };
  std::vector<int8_t> query{2, -1, 3, 1, -2, 4, -3, 5};

  CheckBatchDistanceMatchesSingle(metric, vecs, query, 8, 1e-5f);
}

TEST(SquaredEuclideanMetric, BatchDistanceMatchesSingleInt4) {
  auto metric = IndexFactory::CreateMetric("SquaredEuclidean");
  ASSERT_TRUE(metric);

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_INT4, 8);
  ASSERT_EQ(0, metric->init(meta, ailego::Params()));

  std::vector<std::vector<uint8_t>> vecs{
      {0x12, 0x34, 0x56, 0x78, 0x21, 0x43, 0x65, 0x87},
      {0x10, 0x32, 0x54, 0x76, 0x11, 0x22, 0x33, 0x44},
      {0x89, 0x67, 0x45, 0x23, 0x98, 0x76, 0x54, 0x32},
  };
  std::vector<uint8_t> query{0x01, 0x23, 0x45, 0x67, 0x10, 0x32, 0x54, 0x76};

  CheckBatchDistanceMatchesSingle(metric, vecs, query, 8, 1e-5f);
}

TEST(EuclideanMetric, General) {
  auto metric = IndexFactory::CreateMetric("Euclidean");
  EXPECT_TRUE(metric);

  IndexMeta meta;
  meta.set_meta(IndexMeta::DataType::DT_INT16, 64);
  ASSERT_NE(0, metric->init(meta, ailego::Params()));
  meta.set_meta(IndexMeta::DataType::DT_BINARY32, 64);
  ASSERT_EQ(0, metric->init(meta, ailego::Params()));
  meta.set_meta(IndexMeta::DataType::DT_BINARY64, 64);
  ASSERT_EQ(0, metric->init(meta, ailego::Params()));
  meta.set_meta(IndexMeta::DataType::DT_FP16, 64);
  ASSERT_EQ(0, metric->init(meta, ailego::Params()));
  meta.set_meta(IndexMeta::DataType::DT_FP32, 64);
  ASSERT_EQ(0, metric->init(meta, ailego::Params()));
  meta.set_meta(IndexMeta::DataType::DT_INT4, 64);
  ASSERT_EQ(0, metric->init(meta, ailego::Params()));
  meta.set_meta(IndexMeta::DataType::DT_INT8, 64);
  ASSERT_EQ(0, metric->init(meta, ailego::Params()));

  IndexMeta meta2;
  meta2.set_meta(IndexMeta::DataType::DT_BINARY32, 64);
  EXPECT_TRUE(metric->is_matched(meta));
  EXPECT_FALSE(metric->is_matched(meta2));
  EXPECT_TRUE(metric->is_matched(
      meta, IndexQueryMeta(IndexMeta::DataType::DT_INT8, 64)));
  EXPECT_FALSE(metric->is_matched(
      meta, IndexQueryMeta(IndexMeta::DataType::DT_INT8, 63)));

  EXPECT_FALSE(metric->distance_matrix(0, 0));
  EXPECT_FALSE(metric->distance_matrix(3, 5));
  EXPECT_FALSE(metric->distance_matrix(31, 65));
  EXPECT_TRUE(metric->distance_matrix(1, 1));
  EXPECT_TRUE(metric->distance_matrix(2, 1));
  EXPECT_TRUE(metric->distance_matrix(2, 2));
  EXPECT_TRUE(metric->distance_matrix(4, 1));
  EXPECT_TRUE(metric->distance_matrix(4, 2));
  EXPECT_TRUE(metric->distance_matrix(4, 4));
  EXPECT_TRUE(metric->distance_matrix(8, 1));
  EXPECT_TRUE(metric->distance_matrix(8, 2));
  EXPECT_TRUE(metric->distance_matrix(8, 4));
  EXPECT_TRUE(metric->distance_matrix(8, 8));
  EXPECT_TRUE(metric->distance_matrix(16, 1));
  EXPECT_TRUE(metric->distance_matrix(16, 2));
  EXPECT_TRUE(metric->distance_matrix(16, 4));
  EXPECT_TRUE(metric->distance_matrix(16, 8));
  EXPECT_TRUE(metric->distance_matrix(16, 16));
  EXPECT_TRUE(metric->distance_matrix(32, 1));
  EXPECT_TRUE(metric->distance_matrix(32, 2));
  EXPECT_TRUE(metric->distance_matrix(32, 4));
  EXPECT_TRUE(metric->distance_matrix(32, 8));
  EXPECT_TRUE(metric->distance_matrix(32, 16));
  EXPECT_TRUE(metric->distance_matrix(32, 32));

  EXPECT_FALSE(metric->support_normalize());
  float result = 1.0f;
  metric->normalize(&result);
  EXPECT_FLOAT_EQ(1.0f, result);
}
