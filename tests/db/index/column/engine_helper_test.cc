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
#include <random>
#include <gtest/gtest.h>
#include <zvec/core/interface/index_factory.h>
#include "tests/test_util.h"

using namespace zvec;

#if RABITQ_SUPPORTED
TEST(EngineHelperTest, HnswRabitqBuildsFromReopenedCosineFlat) {
  namespace ci = core_interface;
  const std::string flat_path = "engine_helper_cosine_flat.index";
  const std::string graph_path = "engine_helper_cosine_rabitq.index";
  test_util::RemoveTestFiles(flat_path);
  constexpr uint32_t dim = 128;
  constexpr size_t count = 1001;  // Above HNSW's default brute-force threshold.
  std::mt19937 rng(2026);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<std::vector<float>> vectors(count, std::vector<float>(dim));
  auto flat_param = ci::FlatIndexParamBuilder()
                        .with_metric_type(ci::MetricType::kCosine)
                        .with_data_type(ci::DataType::DT_FP32)
                        .with_dimension(dim)
                        .build();
  auto flat = ci::IndexFactory::CreateAndInitIndex(*flat_param);
  ASSERT_NE(flat, nullptr);
  const ci::StorageOptions create{ci::StorageOptions::StorageType::kMMAP, true};
  const ci::StorageOptions reopen{ci::StorageOptions::StorageType::kMMAP,
                                  false};
  ASSERT_EQ(flat->open(flat_path, create), 0);
  for (size_t i = 0; i < count; ++i) {
    for (float &v : vectors[i]) v = dist(rng) * (i % 7 + 1);
    // Include a zero vector; the cosine provider view must preserve it too.
    if (i == 0) std::fill(vectors[i].begin(), vectors[i].end(), 0.0f);
    ASSERT_EQ(flat->add(ci::VectorData{ci::DenseVector{vectors[i].data()}}, i),
              0);
  }
  ASSERT_EQ(flat->flush(), 0);
  ASSERT_EQ(flat->close(), 0);
  flat = ci::IndexFactory::CreateAndInitIndex(*flat_param);
  ASSERT_NE(flat, nullptr);
  ASSERT_EQ(flat->open(flat_path, reopen), 0);
  auto stored = flat->create_index_provider();
  ASSERT_EQ(stored->element_size(), (dim + 1) * sizeof(float));

  // Only the RaBitQ cosine build path should strip the norm from its view.
  for (auto metric : {MetricType::L2, MetricType::IP, MetricType::COSINE}) {
    for (auto quantizer :
         {QuantizeType::UNDEFINED, QuantizeType::FP16, QuantizeType::RABITQ}) {
      if (metric == MetricType::COSINE && quantizer == QuantizeType::RABITQ)
        continue;
      auto params =
          std::make_shared<HnswIndexParams>(metric, 16, 100, quantizer);
      params->set_raw_vector_provider(stored);
      FieldSchema field("vector", DataType::VECTOR_FP32, dim, false, params);
      auto converted =
          ProximaEngineHelper::convert_to_engine_index_param(field);
      ASSERT_TRUE(converted);
      auto hnsw = std::dynamic_pointer_cast<ci::HNSWIndexParam>(*converted);
      ASSERT_NE(hnsw, nullptr);
      EXPECT_EQ(hnsw->provider, stored);
    }
  }

  for (int bits : {3, 9}) {
    SCOPED_TRACE(bits);
    test_util::RemoveTestFiles(graph_path);
    auto params = std::make_shared<HnswIndexParams>(
        MetricType::COSINE, 16, 100, QuantizeType::RABITQ, false,
        QuantizerParam(false, bits, 4, 128, 6));
    params->set_raw_vector_provider(stored);
    FieldSchema field("vector", DataType::VECTOR_FP32, dim, false, params);
    auto converted = ProximaEngineHelper::convert_to_engine_index_param(field);
    ASSERT_TRUE(converted);
    auto hnsw = std::dynamic_pointer_cast<ci::HNSWIndexParam>(*converted);
    ASSERT_NE(hnsw, nullptr);
    ASSERT_EQ(hnsw->provider->element_size(), dim * sizeof(float));
    EXPECT_TRUE(hnsw->provider->is_matched(hnsw->provider_meta));
    EXPECT_EQ(hnsw->provider->count(), count);
    EXPECT_EQ(hnsw->provider->get_vector(17), stored->get_vector(17));
    core::IndexStorage::MemoryBlock missing;
    EXPECT_EQ(hnsw->provider->get_vector(count + 1, missing),
              stored->get_vector(count + 1, missing));
    const uint64_t keys[] = {0, 17, 101};
    std::vector<core::IndexStorage::MemoryBlock> blocks;
    ASSERT_EQ(hnsw->provider->get_vectors(keys, 3, blocks), 0);
    ASSERT_EQ(blocks.size(), 3U);
    for (size_t i = 0; i < 3; ++i) {
      ASSERT_NE(blocks[i].data(), nullptr);
      EXPECT_EQ(std::memcmp(blocks[i].data(), stored->get_vector(keys[i]),
                            dim * sizeof(float)),
                0);
    }
    auto graph = ci::IndexFactory::CreateAndInitIndex(*hnsw);
    ASSERT_NE(graph, nullptr);
    ASSERT_EQ(graph->open(graph_path, create), 0);
    ASSERT_EQ(graph->merge({flat}, ci::IndexFilter()), 0);
    EXPECT_EQ(graph->get_doc_count(), count);
    auto query_param =
        ci::HNSWQueryParamBuilder().with_topk(5).with_ef_search(count).build();
    const ci::VectorData query{ci::DenseVector{vectors[17].data()}};
    ci::SearchResult before;
    ASSERT_EQ(graph->search(query, query_param, &before), 0);
    ASSERT_EQ(before.doc_list_.size(), 5U);
    EXPECT_EQ(before.doc_list_.front().key(), 17U);
    auto scaled = vectors[17];
    for (float &v : scaled) v *= 4.0f;
    ci::SearchResult scaled_result;
    ASSERT_EQ(graph->search(ci::VectorData{ci::DenseVector{scaled.data()}},
                            query_param, &scaled_result),
              0);
    ASSERT_EQ(scaled_result.doc_list_.size(), before.doc_list_.size());
    for (size_t i = 0; i < before.doc_list_.size(); ++i) {
      EXPECT_EQ(scaled_result.doc_list_[i].key(), before.doc_list_[i].key());
      EXPECT_FLOAT_EQ(scaled_result.doc_list_[i].score(),
                      before.doc_list_[i].score());
    }
    ASSERT_EQ(graph->close(), 0);
    hnsw->provider.reset();
    graph = ci::IndexFactory::CreateAndInitIndex(*hnsw);
    ASSERT_NE(graph, nullptr);
    ASSERT_EQ(graph->open(graph_path, reopen), 0);
    ci::SearchResult after;
    ASSERT_EQ(graph->search(query, query_param, &after), 0);
    ASSERT_EQ(after.doc_list_.size(), before.doc_list_.size());
    for (size_t i = 0; i < before.doc_list_.size(); ++i) {
      EXPECT_EQ(after.doc_list_[i].key(), before.doc_list_[i].key());
      EXPECT_FLOAT_EQ(after.doc_list_[i].score(), before.doc_list_[i].score());
    }
    ASSERT_EQ(graph->close(), 0);
    test_util::RemoveTestFiles(graph_path);
  }
  ASSERT_EQ(flat->close(), 0);
  test_util::RemoveTestFiles(flat_path);
}
#endif

TEST(EngineHelperTest, RabitqTemporaryFlatUsesRawVectors) {
  for (auto metric : {MetricType::L2, MetricType::IP, MetricType::COSINE}) {
    auto params = MakeDefaultQuantVectorIndexParams(
        metric, QuantizeType::RABITQ, QuantizerParam(false, 7, 4, 128, 6));
    EXPECT_EQ(params.quantize_type(), QuantizeType::UNDEFINED);
    EXPECT_EQ(params.metric_type(), metric);
    FieldSchema field("vector", DataType::VECTOR_FP32, 128, false,
                      params.clone());
    ASSERT_TRUE(field.validate().ok());
    auto result = ProximaEngineHelper::convert_to_engine_index_param(field);
    ASSERT_TRUE(result) << result.error().message();
    ASSERT_NE((*result)->quantizer_param, nullptr);
    EXPECT_EQ((*result)->index_type, core_interface::IndexType::kFlat);
    EXPECT_EQ((*result)->quantizer_param->type,
              core_interface::QuantizerType::kNone);
  }
  // Other quantizers must keep the same temporary-index configuration.
  for (auto type : {QuantizeType::UNDEFINED, QuantizeType::FP16,
                    QuantizeType::INT8, QuantizeType::INT4}) {
    auto params = MakeDefaultQuantVectorIndexParams(MetricType::L2, type,
                                                    QuantizerParam(true));
    EXPECT_EQ(params.quantize_type(), type);
    EXPECT_TRUE(params.quantizer_param().enable_rotate());
  }
}

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
