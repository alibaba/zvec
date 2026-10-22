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

#include "db/index/column/vector_column/engine_helper.hpp"
#include <gtest/gtest.h>

using namespace zvec;

TEST(EngineHelperTest, IVFRabitqQuantizerParameters) {
  for (int bits : {3, 7, 9}) {
    for (int samples : {0, 24}) {
      for (bool rotate : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "bits=" << bits << " samples="
                                          << samples << " rotate=" << rotate);
        FieldSchema field("vector", DataType::VECTOR_FP32, 128, false,
                          std::make_shared<IVFIndexParams>(
                              MetricType::L2, 4, 3, false, QuantizeType::RABITQ,
                              QuantizerParam(rotate), bits, samples));
        auto result = ProximaEngineHelper::convert_to_engine_index_param(field);
        ASSERT_TRUE(result) << result.error().message();
        auto ivf = std::dynamic_pointer_cast<core_interface::IVFIndexParam>(
            result.value());
        ASSERT_NE(ivf, nullptr);
        auto quantizer =
            std::dynamic_pointer_cast<core_interface::RabitqQuantizerParam>(
                ivf->quantizer_param);
        ASSERT_NE(quantizer, nullptr);
        EXPECT_EQ(quantizer->total_bits, bits);
        EXPECT_EQ(quantizer->sample_count, samples);
        EXPECT_EQ(quantizer->enable_rotate, rotate);
        // Quantizer training stays independent of IVF partition training.
        EXPECT_EQ(quantizer->num_clusters, 16);
        EXPECT_EQ(quantizer->niters, 20);
        EXPECT_EQ(ivf->nlist, 4);
        EXPECT_EQ(ivf->niters, 3);
      }
    }
  }
}

TEST(EngineHelperTest, HnswRabitqQuantizerParameters) {
  auto params = std::make_shared<HnswIndexParams>(
      MetricType::L2, 16, 100, QuantizeType::RABITQ, false,
      QuantizerParam(false, 3, 4, 24, 6));
  FieldSchema field("vector", DataType::VECTOR_FP32, 128, false, params);
  auto result = ProximaEngineHelper::convert_to_engine_index_param(field);
  ASSERT_TRUE(result);
  auto hnsw =
      std::dynamic_pointer_cast<core_interface::HNSWIndexParam>(*result);
  ASSERT_NE(hnsw, nullptr);
  auto q = std::dynamic_pointer_cast<core_interface::RabitqQuantizerParam>(
      hnsw->quantizer_param);
  ASSERT_NE(q, nullptr);
  EXPECT_EQ(q->total_bits, 3);
  EXPECT_EQ(q->num_clusters, 4);
  EXPECT_EQ(q->sample_count, 24);
  EXPECT_EQ(q->niters, 6);
  EXPECT_EQ(hnsw->provider, nullptr);  // Reopen does not require training data.
  auto provider =
      std::make_shared<core::MultiPassIndexProvider<core::IndexMeta::DT_FP32>>(
          128);
  params->set_raw_vector_provider(provider);
  field.set_index_params(params->clone());
  result = ProximaEngineHelper::convert_to_engine_index_param(field);
  ASSERT_TRUE(result);
  hnsw = std::dynamic_pointer_cast<core_interface::HNSWIndexParam>(*result);
  ASSERT_NE(hnsw, nullptr);
  EXPECT_EQ(hnsw->provider, provider);
  EXPECT_EQ(hnsw->provider_meta.data_type(), core::IndexMeta::DT_FP32);
  EXPECT_EQ(hnsw->provider_meta.dimension(), 128U);
  EXPECT_EQ(hnsw->provider_meta.metric_name(), "SquaredEuclidean");
}

TEST(EngineHelperTest, DiskAnnRabitqQuantizerParameters) {
  auto params = std::make_shared<DiskAnnIndexParams>(
      MetricType::COSINE, 16, 100, 0, QuantizeType::RABITQ,
      QuantizerParam(false, 9, 4, 24, 6));
  FieldSchema field("vector", DataType::VECTOR_FP32, 128, false, params);
  auto result = ProximaEngineHelper::convert_to_engine_index_param(field);
  ASSERT_TRUE(result);
  auto diskann =
      std::dynamic_pointer_cast<core_interface::DiskAnnIndexParam>(*result);
  ASSERT_NE(diskann, nullptr);
  auto q = std::dynamic_pointer_cast<core_interface::RabitqQuantizerParam>(
      diskann->quantizer_param);
  ASSERT_NE(q, nullptr);
  EXPECT_EQ(q->total_bits, 9);
  EXPECT_EQ(q->num_clusters, 4);
  EXPECT_EQ(q->sample_count, 24);
  EXPECT_EQ(q->niters, 6);
  EXPECT_EQ(diskann->pq_chunk_num, 0);
  EXPECT_EQ(diskann->metric_type, core_interface::MetricType::kCosine);
}
