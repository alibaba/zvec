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

#include "zvec/db/index_params.h"
#include <gtest/gtest.h>

using namespace zvec;

TEST(IndexParamsTest, IndexParamsBaseClass) {
  // Test that IndexParams is abstract and can't be instantiated directly
  // This is more of a compile-time check - we can't directly instantiate an
  // abstract class

  // Test is_vector_index_type method
  HnswIndexParams hnsw_params(MetricType::L2, 16, 100);
  EXPECT_TRUE(hnsw_params.is_vector_index_type());

  FlatIndexParams flat_params(MetricType::IP);
  EXPECT_TRUE(flat_params.is_vector_index_type());

  IVFIndexParams ivf_params(MetricType::COSINE, 100);
  EXPECT_TRUE(ivf_params.is_vector_index_type());

  InvertIndexParams invert_params(true);
  EXPECT_FALSE(invert_params.is_vector_index_type());
}

TEST(IndexParamsTest, InvertIndexParams) {
  // Test constructor
  InvertIndexParams params(true);
  EXPECT_EQ(params.type(), IndexType::INVERT);
  EXPECT_TRUE(params.enable_range_optimization());

  InvertIndexParams params2(false);
  EXPECT_FALSE(params2.enable_range_optimization());

  // Test clone method
  auto cloned = params.clone();
  EXPECT_NE(cloned.get(), &params);  // Should be different objects
  EXPECT_EQ(cloned->type(), IndexType::INVERT);

  // Test comparison operators
  InvertIndexParams params3(true);
  InvertIndexParams params4(false);

  EXPECT_TRUE(params == params3);
  EXPECT_FALSE(params == params4);
  EXPECT_TRUE(params != params4);

  // Test setter
  params2.set_enable_range_optimization(true);
  EXPECT_TRUE(params2.enable_range_optimization());
  EXPECT_TRUE(params2 == params);
}

TEST(IndexParamsTest, VectorIndexParamsBase) {
  // Test constructor and basic methods
  FlatIndexParams flat_params(MetricType::L2, QuantizeType::FP16);
  EXPECT_EQ(flat_params.type(), IndexType::FLAT);
  EXPECT_EQ(flat_params.metric_type(), MetricType::L2);
  EXPECT_EQ(flat_params.quantize_type(), QuantizeType::FP16);

  // Test setters
  flat_params.set_metric_type(MetricType::IP);
  EXPECT_EQ(flat_params.metric_type(), MetricType::IP);

  flat_params.set_quantize_type(QuantizeType::INT8);
  EXPECT_EQ(flat_params.quantize_type(), QuantizeType::INT8);
}

TEST(IndexParamsTest, HnswIndexParams) {
  // Test constructor
  HnswIndexParams params(MetricType::COSINE, 20, 150, QuantizeType::INT4);
  EXPECT_EQ(params.type(), IndexType::HNSW);
  EXPECT_EQ(params.metric_type(), MetricType::COSINE);
  EXPECT_EQ(params.m(), 20);
  EXPECT_EQ(params.ef_construction(), 150);
  EXPECT_EQ(params.quantize_type(), QuantizeType::INT4);

  // Test clone
  auto cloned = params.clone();
  EXPECT_NE(cloned.get(), &params);
  EXPECT_EQ(cloned->type(), IndexType::HNSW);

  // Test comparison
  HnswIndexParams params2(MetricType::COSINE, 20, 150, QuantizeType::INT4);
  HnswIndexParams params3(MetricType::L2, 20, 150, QuantizeType::INT4);
  HnswIndexParams params4(MetricType::COSINE, 16, 150, QuantizeType::INT4);
  HnswIndexParams params5(MetricType::COSINE, 20, 200, QuantizeType::INT4);

  EXPECT_TRUE(params == params2);
  EXPECT_FALSE(params == params3);
  EXPECT_FALSE(params == params4);
  EXPECT_FALSE(params == params5);

  // Test setters
  params.set_m(10);
  EXPECT_EQ(params.m(), 10);

  params.set_ef_construction(75);
  EXPECT_EQ(params.ef_construction(), 75);
}

TEST(IndexParamsTest, FlatIndexParams) {
  // Test constructor
  FlatIndexParams params(MetricType::IP, QuantizeType::FP16);
  EXPECT_EQ(params.type(), IndexType::FLAT);
  EXPECT_EQ(params.metric_type(), MetricType::IP);
  EXPECT_EQ(params.quantize_type(), QuantizeType::FP16);

  // Test clone
  auto cloned = params.clone();
  EXPECT_NE(cloned.get(), &params);
  EXPECT_EQ(cloned->type(), IndexType::FLAT);

  // Test comparison
  FlatIndexParams params2(MetricType::IP, QuantizeType::FP16);
  FlatIndexParams params3(MetricType::L2, QuantizeType::FP16);
  FlatIndexParams params4(MetricType::IP, QuantizeType::INT8);

  EXPECT_TRUE(params == params2);
  EXPECT_FALSE(params == params3);
  EXPECT_FALSE(params == params4);
}

TEST(IndexParamsTest, IVFIndexParams) {
  // Test constructor
  IVFIndexParams params(MetricType::L2, 128, 10, false, QuantizeType::INT8);
  EXPECT_EQ(params.type(), IndexType::IVF);
  EXPECT_EQ(params.metric_type(), MetricType::L2);
  EXPECT_EQ(params.n_list(), 128);
  EXPECT_EQ(params.quantize_type(), QuantizeType::INT8);

  // Test clone
  auto cloned = params.clone();
  EXPECT_NE(cloned.get(), &params);
  EXPECT_EQ(cloned->type(), IndexType::IVF);

  // Test comparison
  IVFIndexParams params2(MetricType::L2, 128, 10, false, QuantizeType::INT8);
  IVFIndexParams params3(MetricType::IP, 128, 10, false, QuantizeType::INT8);
  IVFIndexParams params4(MetricType::L2, 256, 10, false, QuantizeType::INT8);
  IVFIndexParams params5(MetricType::L2, 128, 10, false, QuantizeType::FP16);

  EXPECT_TRUE(params == params2);
  EXPECT_FALSE(params == params3);
  EXPECT_FALSE(params == params4);
  EXPECT_FALSE(params == params5);

  // Test setter
  params.set_n_list(64);
  EXPECT_EQ(params.n_list(), 64);
}

TEST(IndexParamsTest, DefaultVectorIndexParams) {
  // Test default vector index params
  EXPECT_EQ(DefaultVectorIndexParams.type(), IndexType::FLAT);
  EXPECT_EQ(DefaultVectorIndexParams.metric_type(), MetricType::IP);
  EXPECT_EQ(DefaultVectorIndexParams.quantize_type(), QuantizeType::UNDEFINED);
}

TEST(IndexParamsTest, DynamicPointerCast) {
  // Test dynamic_pointer_cast functionality with IndexParams
  IndexParams::Ptr base_ptr =
      std::make_shared<HnswIndexParams>(MetricType::L2, 16, 100);
  auto hnsw_ptr = std::dynamic_pointer_cast<HnswIndexParams>(base_ptr);
  EXPECT_NE(hnsw_ptr, nullptr);
  EXPECT_EQ(hnsw_ptr->type(), IndexType::HNSW);

  // Test casting to wrong type
  auto flat_ptr = std::dynamic_pointer_cast<FlatIndexParams>(base_ptr);
  EXPECT_EQ(flat_ptr, nullptr);

  // Test casting from base class reference
  IndexParams &base_ref = *base_ptr;
  auto &hnsw_ref = dynamic_cast<HnswIndexParams &>(base_ref);
  EXPECT_EQ(hnsw_ref.type(), IndexType::HNSW);
}