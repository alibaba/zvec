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
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <gtest/gtest.h>

#if RABITQ_SUPPORTED
#include <rabitqlib/utils/cpu_features.hpp>
#include "ailego/pattern/scope_guard.h"
#include "algorithm/hnsw_rabitq/hnsw_rabitq_context.h"
#include "algorithm/hnsw_rabitq/hnsw_rabitq_params.h"
#include "algorithm/hnsw_rabitq/rabitq_converter.h"
#include "tests/test_util.h"
#include "zvec/core/framework/index_holder.h"
#include "zvec/core/interface/index.h"
#include "zvec/core/interface/index_param_builders.h"

namespace zvec::core_interface {
namespace {

constexpr uint32_t kDimension = 128;
constexpr uint32_t kTotalBits = 7;
constexpr uint32_t kGraphDocCount =
    core::HnswRabitqEntity::kDefaultBruteForceThreshold + 1;

struct EntryScoreCase {
  MetricType metric;
  uint32_t doc_count;
  const char *name;
};

class TestableHnswRabitqIndex : public HNSWRabitqIndex {
 public:
  explicit TestableHnswRabitqIndex(bool force_graph)
      : force_graph_(force_graph) {}

  using Index::init;

  core::IndexContext::Pointer create_context() const {
    return streamer_->create_context();
  }

 protected:
  int create_and_init_streamer(const BaseIndexParam &param) override {
    if (force_graph_) {
      // A single-node index otherwise falls back to a linear scan.
      proxima_index_params_.set(
          core::PARAM_HNSW_RABITQ_STREAMER_BRUTE_FORCE_THRESHOLD, 0U);
    }
    return HNSWRabitqIndex::create_and_init_streamer(param);
  }

 private:
  bool force_graph_;
};

class HnswRabitqIndexTest : public testing::TestWithParam<EntryScoreCase> {
 protected:
  void SetUp() override {
    if (!rabitqlib::cpu::has_avx512_core() && !rabitqlib::cpu::has_avx2() &&
        !rabitqlib::cpu::has_neon()) {
      GTEST_SKIP() << "CPU does not support AVX2/FMA, AVX512F/BW/DQ, or NEON";
    }
    test_util::RemoveTestFiles(index_path() + "*");
  }

  void TearDown() override {
    test_util::RemoveTestFiles(index_path() + "*");
  }

  std::string index_path() const {
    return std::string("hnsw_rabitq_entry_score_") + GetParam().name;
  }
};

TEST_P(HnswRabitqIndexTest, EntryScoreMatchesFullEstimate) {
  const auto &test_case = GetParam();
  auto holder = std::make_shared<
      core::MultiPassIndexProvider<core::IndexMeta::DataType::DT_FP32>>(
      kDimension);
  std::vector<ailego::NumericalVector<float>> vectors;
  std::mt19937 generator(20260319);
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
  // Keep a nonzero quantization residual even for the single-node index.
  const uint32_t train_count = std::max(2U, test_case.doc_count);
  for (uint32_t i = 0; i < train_count; ++i) {
    ailego::NumericalVector<float> vector(kDimension);
    float norm = 0.0f;
    for (uint32_t j = 0; j < kDimension; ++j) {
      vector[j] = distribution(generator);
      norm += vector[j] * vector[j];
    }
    if (test_case.metric == MetricType::kCosine) {
      norm = std::sqrt(norm);
      for (uint32_t j = 0; j < kDimension; ++j) {
        vector[j] /= norm;
      }
    }
    vectors.push_back(vector);
    ASSERT_TRUE(holder->emplace(i, vector));
  }

  core::IndexMeta meta(DataType::DT_FP32, kDimension);
  meta.set_metric(
      test_case.metric == MetricType::kCosine ? "Cosine" : "SquaredEuclidean",
      0, ailego::Params());
  ailego::Params converter_params;
  converter_params.set(core::PARAM_RABITQ_TOTAL_BITS, kTotalBits);
  converter_params.set(core::PARAM_RABITQ_NUM_CLUSTERS, 1U);
  core::RabitqConverter converter;
  ASSERT_EQ(0, converter.init(meta, converter_params));
  ASSERT_EQ(0, converter.train(holder));
  core::IndexReformer::Pointer reformer;
  ASSERT_EQ(0, converter.to_reformer(&reformer));

  auto param = HNSWRabitqIndexParamBuilder()
                   .with_metric_type(test_case.metric)
                   .with_data_type(DataType::DT_FP32)
                   .with_dimension(kDimension)
                   .with_m(16)
                   .with_ef_construction(100)
                   .with_total_bits(kTotalBits)
                   .with_provider(holder)
                   .with_reformer(reformer)
                   .build();
  TestableHnswRabitqIndex index(test_case.doc_count == 1);
  ASSERT_EQ(0, index.init(*param));
  ASSERT_EQ(
      0, index.open(index_path(), {StorageOptions::StorageType::kMMAP, true}));
  auto close_guard =
      ailego::ScopeGuard::Make([&]() { EXPECT_EQ(0, index.close()); });
  for (uint32_t i = 0; i < test_case.doc_count; ++i) {
    ASSERT_EQ(0, index.add(VectorData{DenseVector{vectors[i].data()}}, i));
  }

  auto context = index.create_context();
  auto *rabitq_context = dynamic_cast<core::HnswRabitqContext *>(context.get());
  ASSERT_NE(nullptr, rabitq_context);
  const auto &entity = rabitq_context->get_entity();
  const auto entry = entity.get_key(entity.entry_point());
  ASSERT_LT(entry, test_case.doc_count);
  const auto &direction = vectors[(entry + 1) % vectors.size()];

  auto query_param =
      HNSWRabitqQueryParamBuilder().with_topk(1).with_ef_search(100).build();
  for (float offset : {0.125f, 0.25f, 0.5f}) {
    SCOPED_TRACE(offset);
    // Keep the entry closest at every level so its score reaches the result.
    std::vector<float> query(kDimension);
    for (uint32_t j = 0; j < kDimension; ++j) {
      query[j] = vectors[entry][j] + offset * direction[j];
    }
    const VectorData query_data{DenseVector{query.data()}};
    SearchResult result;
    query_param->is_linear = false;
    ASSERT_EQ(0, index.search(query_data, query_param, &result));
    ASSERT_EQ(1U, result.doc_list_.size());
    ASSERT_EQ(entry, result.doc_list_[0].key());
    const float graph_score = result.doc_list_[0].score();

    query_param->is_linear = true;
    ASSERT_EQ(0, index.search(query_data, query_param, &result));
    ASSERT_EQ(1U, result.doc_list_.size());
    ASSERT_EQ(entry, result.doc_list_[0].key());
    // Linear search uses all quantization bits; it is still an estimate.
    EXPECT_FLOAT_EQ(result.doc_list_[0].score(), graph_score);
  }
}

INSTANTIATE_TEST_SUITE_P(
    L2AndCosine, HnswRabitqIndexTest,
    testing::Values(EntryScoreCase{MetricType::kL2sq, 1, "L2SingleEntry"},
                    EntryScoreCase{MetricType::kCosine, 1, "CosineSingleEntry"},
                    EntryScoreCase{MetricType::kL2sq, kGraphDocCount,
                                   "L2DefaultGraph"},
                    EntryScoreCase{MetricType::kCosine, kGraphDocCount,
                                   "CosineDefaultGraph"}),
    [](const testing::TestParamInfo<EntryScoreCase> &info) {
      return info.param.name;
    });

}  // namespace
}  // namespace zvec::core_interface
#else
// Keep unsupported builds visible as skipped instead of empty test suites.
TEST(HnswRabitqIndexTest, UnsupportedPlatform) {
  GTEST_SKIP() << "RaBitQ support is disabled for this platform";
}
#endif  // RABITQ_SUPPORTED
