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
  HnswIndexParams params(MetricType::COSINE, 20, 150, QuantizeType::INT4, true,
                         QuantizerParam(true), true, DataType::VECTOR_FP16);
  EXPECT_EQ(params.type(), IndexType::HNSW);
  EXPECT_EQ(params.metric_type(), MetricType::COSINE);
  EXPECT_EQ(params.m(), 20);
  EXPECT_EQ(params.ef_construction(), 150);
  EXPECT_EQ(params.quantize_type(), QuantizeType::INT4);
  EXPECT_TRUE(params.use_contiguous_memory());
  EXPECT_TRUE(params.use_flat_contiguous_memory());
  EXPECT_EQ(params.flat_data_type(), DataType::VECTOR_FP16);

  // Test clone
  auto cloned = params.clone();
  EXPECT_NE(cloned.get(), &params);
  EXPECT_EQ(cloned->type(), IndexType::HNSW);
  auto cloned_hnsw = std::dynamic_pointer_cast<HnswIndexParams>(cloned);
  ASSERT_NE(cloned_hnsw, nullptr);
  EXPECT_TRUE(cloned_hnsw->use_flat_contiguous_memory());
  EXPECT_EQ(cloned_hnsw->flat_data_type(), DataType::VECTOR_FP16);
  EXPECT_TRUE(*cloned_hnsw == params);

  // Test comparison
  HnswIndexParams params2(MetricType::COSINE, 20, 150, QuantizeType::INT4, true,
                          QuantizerParam(true), true, DataType::VECTOR_FP16);
  HnswIndexParams params3(MetricType::L2, 20, 150, QuantizeType::INT4);
  HnswIndexParams params4(MetricType::COSINE, 16, 150, QuantizeType::INT4);
  HnswIndexParams params5(MetricType::COSINE, 20, 200, QuantizeType::INT4);
  HnswIndexParams params6(MetricType::COSINE, 20, 150, QuantizeType::INT4, true,
                          QuantizerParam(true), true, DataType::VECTOR_FP32);

  EXPECT_TRUE(params == params2);
  EXPECT_FALSE(params == params3);
  EXPECT_FALSE(params == params4);
  EXPECT_FALSE(params == params5);
  EXPECT_FALSE(params == params6);

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

TEST(IndexParamsTest, DefaultFlatReferenceParams) {
  HnswIndexParams hnsw(MetricType::L2);
  VamanaIndexParams vamana(MetricType::L2);
  EXPECT_EQ(hnsw.flat_data_type(), DataType::VECTOR_FP32);
  EXPECT_EQ(vamana.flat_data_type(), DataType::VECTOR_FP32);

  hnsw.set_flat_data_type(DataType::UNDEFINED);
  vamana.set_flat_data_type(DataType::UNDEFINED);
  EXPECT_EQ(hnsw.flat_data_type(), DataType::VECTOR_FP32);
  EXPECT_EQ(vamana.flat_data_type(), DataType::VECTOR_FP32);

  auto flat =
      MakeDefaultVectorIndexParams(MetricType::L2, true, DataType::VECTOR_FP16);
  EXPECT_EQ(flat.quantize_type(), QuantizeType::UNDEFINED);
  EXPECT_TRUE(flat.use_contiguous_memory());
  EXPECT_EQ(flat.storage_data_type(), DataType::VECTOR_FP16);
  auto cloned = std::dynamic_pointer_cast<FlatIndexParams>(flat.clone());
  ASSERT_NE(cloned, nullptr);
  EXPECT_TRUE(cloned->use_contiguous_memory());
  EXPECT_EQ(cloned->storage_data_type(), DataType::VECTOR_FP16);
  EXPECT_TRUE(*cloned == flat);
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

TEST(IndexParamsTest, DiskAnnIndexParams) {
  DiskAnnIndexParams params(MetricType::L2, 48, 80, 16, QuantizeType::FP16,
                            QuantizerParam(true));

  EXPECT_EQ(IndexType::DISKANN, params.type());
  EXPECT_EQ(MetricType::L2, params.metric_type());
  EXPECT_EQ(48, params.max_degree());
  EXPECT_EQ(80, params.list_size());
  EXPECT_EQ(16, params.pq_chunk_num());
  EXPECT_EQ(QuantizeType::FP16, params.quantize_type());
  EXPECT_TRUE(params.quantizer_param().enable_rotate());

  auto cloned = params.clone();
  auto *cloned_diskann = dynamic_cast<DiskAnnIndexParams *>(cloned.get());
  ASSERT_NE(nullptr, cloned_diskann);
  EXPECT_EQ(params, *cloned_diskann);

  cloned_diskann->set_list_size(40);
  EXPECT_NE(params, *cloned_diskann);
}

#if RABITQ_SUPPORTED
TEST(IndexParamsTest, IvfRabitqIndexParams) {
  IvfRabitqIndexParams params(MetricType::COSINE, 32, 1, 5);
  EXPECT_EQ(params.type(), IndexType::IVF_RABITQ);
  EXPECT_EQ(params.metric_type(), MetricType::COSINE);
  EXPECT_EQ(params.quantize_type(), QuantizeType::RABITQ);
  EXPECT_EQ(params.nlist(), 32);
  EXPECT_EQ(params.total_bits(), 1);
  EXPECT_EQ(params.sample_count(), 5);

  auto cloned = params.clone();
  EXPECT_NE(cloned.get(), &params);
  EXPECT_EQ(cloned->type(), IndexType::IVF_RABITQ);
  EXPECT_TRUE(*cloned == params);

  IvfRabitqIndexParams same_params(MetricType::COSINE, 32, 1, 5);
  IvfRabitqIndexParams diff_metric(MetricType::IP, 32, 1, 5);
  IvfRabitqIndexParams diff_nlist(MetricType::COSINE, 64, 1, 5);
  IvfRabitqIndexParams diff_total_bits(MetricType::COSINE, 32, 7, 5);
  IvfRabitqIndexParams diff_sample_count(MetricType::COSINE, 32, 1, 0);

  EXPECT_TRUE(params == same_params);
  EXPECT_FALSE(params == diff_metric);
  EXPECT_FALSE(params == diff_nlist);
  EXPECT_FALSE(params == diff_total_bits);
  EXPECT_FALSE(params == diff_sample_count);

  params.set_nlist(48);
  params.set_total_bits(7);
  params.set_sample_count(0);
  EXPECT_EQ(params.nlist(), 48);
  EXPECT_EQ(params.total_bits(), 7);
  EXPECT_EQ(params.sample_count(), 0);
}
#endif

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

// ==================== QuantizerParam tests ====================

TEST(IndexParamsTest, QuantizerParamBasic) {
  // Default constructor: enable_rotate should be false
  QuantizerParam qp_default;
  EXPECT_FALSE(qp_default.enable_rotate());

  // Constructor with true
  QuantizerParam qp_true(true);
  EXPECT_TRUE(qp_true.enable_rotate());

  // Constructor with false
  QuantizerParam qp_false(false);
  EXPECT_FALSE(qp_false.enable_rotate());

  // Setter
  qp_default.set_enable_rotate(true);
  EXPECT_TRUE(qp_default.enable_rotate());
  qp_default.set_enable_rotate(false);
  EXPECT_FALSE(qp_default.enable_rotate());

  // Equality
  EXPECT_TRUE(qp_true == QuantizerParam(true));
  EXPECT_TRUE(qp_false == QuantizerParam(false));
  EXPECT_FALSE(qp_true == qp_false);

  // Inequality
  EXPECT_TRUE(qp_true != qp_false);
  EXPECT_FALSE(qp_true != QuantizerParam(true));
}

TEST(IndexParamsTest, QuantizerParamWithVectorIndex) {
  // HnswIndexParams
  {
    HnswIndexParams params(MetricType::COSINE, 16, 100, QuantizeType::INT8);
    EXPECT_FALSE(params.quantizer_param().enable_rotate());
    EXPECT_FALSE(params.enable_rotate());  // convenience getter

    params.set_quantizer_param(QuantizerParam(true));
    EXPECT_TRUE(params.quantizer_param().enable_rotate());
    EXPECT_TRUE(params.enable_rotate());

    // Clone preserves quantizer_param
    auto cloned = params.clone();
    auto *cloned_hnsw = dynamic_cast<HnswIndexParams *>(cloned.get());
    ASSERT_NE(cloned_hnsw, nullptr);
    EXPECT_TRUE(cloned_hnsw->quantizer_param().enable_rotate());
    EXPECT_TRUE(*cloned == params);

    // Equality: different enable_rotate -> not equal
    HnswIndexParams params2(MetricType::COSINE, 16, 100, QuantizeType::INT8);
    params2.set_quantizer_param(QuantizerParam(false));
    EXPECT_FALSE(params == params2);
  }

  // FlatIndexParams
  {
    FlatIndexParams params(MetricType::L2, QuantizeType::INT8);
    EXPECT_FALSE(params.quantizer_param().enable_rotate());

    params.set_quantizer_param(QuantizerParam(true));
    EXPECT_TRUE(params.quantizer_param().enable_rotate());
    EXPECT_TRUE(params.enable_rotate());

    auto cloned = params.clone();
    auto *cloned_flat = dynamic_cast<FlatIndexParams *>(cloned.get());
    ASSERT_NE(cloned_flat, nullptr);
    EXPECT_TRUE(cloned_flat->quantizer_param().enable_rotate());

    FlatIndexParams params2(MetricType::L2, QuantizeType::INT8);
    EXPECT_FALSE(params == params2);
  }

  // IVFIndexParams
  {
    IVFIndexParams params(MetricType::IP, 128, 10, false, QuantizeType::INT8);
    EXPECT_FALSE(params.quantizer_param().enable_rotate());

    params.set_quantizer_param(QuantizerParam(true));
    EXPECT_TRUE(params.quantizer_param().enable_rotate());
    EXPECT_TRUE(params.enable_rotate());

    auto cloned = params.clone();
    auto *cloned_ivf = dynamic_cast<IVFIndexParams *>(cloned.get());
    ASSERT_NE(cloned_ivf, nullptr);
    EXPECT_TRUE(cloned_ivf->quantizer_param().enable_rotate());

    IVFIndexParams params2(MetricType::IP, 128, 10, false, QuantizeType::INT8);
    EXPECT_FALSE(params == params2);
  }
}
