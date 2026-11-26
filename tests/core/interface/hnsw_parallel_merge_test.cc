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

#include <cmath>
#include <random>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/core/interface/index.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param_builders.h>
#include "tests/test_util.h"

namespace {
using namespace zvec::core_interface;

class HnswParallelMergeTest : public testing::TestWithParam<MetricType> {};

TEST_P(HnswParallelMergeTest, Fp16GraphSurvivesParallelMergeAndReopen) {
  constexpr uint32_t kDimension = 128;
  constexpr uint32_t kCount = 2048;
  const auto metric = GetParam();
  const std::string prefix =
      "hnsw_parallel_merge_" + std::to_string(static_cast<int>(metric));
  const auto source_path = prefix + "_source.index";
  const auto target_path = prefix + "_target.index";
  zvec::test_util::RemoveTestFiles(source_path);
  zvec::test_util::RemoveTestFiles(target_path);

  auto source_param = FlatIndexParamBuilder()
                          .with_data_type(DataType::DT_FP32)
                          .with_dimension(kDimension)
                          .with_metric_type(metric)
                          .build();
  auto source = IndexFactory::CreateAndInitIndex(*source_param);
  ASSERT_NE(nullptr, source);
  ASSERT_EQ(
      0, source->open(source_path, {StorageOptions::StorageType::kMMAP, true}));
  std::mt19937 random(827);
  std::uniform_real_distribution<float> value(-1.0f, 1.0f);
  std::vector<std::vector<float>> vectors(kCount,
                                          std::vector<float>(kDimension));
  for (uint32_t id = 0; id < kCount; ++id) {
    float norm = 0.0f;
    for (auto &element : vectors[id]) {
      element = value(random);
      norm += element * element;
    }
    for (auto &element : vectors[id]) element /= std::sqrt(norm);
    ASSERT_EQ(0, source->add(VectorData{DenseVector{vectors[id].data()}}, id));
  }
  ASSERT_EQ(0, source->flush());

  auto target_param =
      HNSWIndexParamBuilder()
          .with_data_type(DataType::DT_FP32)
          .with_dimension(kDimension)
          .with_metric_type(metric)
          .with_m(16)
          .with_ef_construction(200)
          .with_quantizer_param(QuantizerParam(QuantizerType::kFP16))
          .build();
  auto target = IndexFactory::CreateAndInitIndex(*target_param);
  ASSERT_NE(nullptr, target);
  ASSERT_EQ(
      0, target->open(target_path, {StorageOptions::StorageType::kMMAP, true}));
  MergeOptions options;
  options.write_concurrency = 3;
  ASSERT_EQ(0, target->merge({source}, IndexFilter{}, options));
  ASSERT_EQ(kCount, target->get_doc_count());
  auto query_param = HNSWQueryParamBuilder().with_topk(10).build();
  for (int pass = 0; pass < 2; ++pass) {
    for (uint32_t id : {0U, 173U, 1023U, kCount - 1}) {
      SearchResult result;
      ASSERT_EQ(0, target->search(VectorData{DenseVector{vectors[id].data()}},
                                  query_param, &result));
      ASSERT_EQ(10U, result.doc_list_.size());
      EXPECT_EQ(id, result.doc_list_.front().key());
      for (const auto &doc : result.doc_list_) {
        EXPECT_LT(doc.key(), kCount);
        EXPECT_TRUE(std::isfinite(doc.score()));
      }
    }
    ASSERT_EQ(0, target->close());
    if (pass == 0) {
      target = IndexFactory::CreateAndInitIndex(*target_param);
      ASSERT_NE(nullptr, target);
      ASSERT_EQ(0, target->open(target_path,
                                {StorageOptions::StorageType::kMMAP, false}));
    }
  }
  ASSERT_EQ(0, source->close());
  zvec::test_util::RemoveTestFiles(source_path);
  zvec::test_util::RemoveTestFiles(target_path);
}

INSTANTIATE_TEST_SUITE_P(Metrics, HnswParallelMergeTest,
                         testing::Values(MetricType::kL2sq,
                                         MetricType::kInnerProduct,
                                         MetricType::kCosine));
}  // namespace
