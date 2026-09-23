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

#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/container/params.h>
#include <zvec/ailego/utility/float_helper.h>
#include <zvec/core/framework/index_holder.h>
#include <zvec/core/framework/index_meta.h>
#include "quantizer/common/pq_quantizer/pq_opq.h"
#include "quantizer/common/pq_quantizer/precompute_table_quantizer.h"
#include "quantizer/quantizer.h"

// Inspect retained capacity without exposing a production testing API.
#define private public
#include "quantizer/pq_int8_quantizer/pq_int8_quantizer.h"
#undef private

namespace zvec {
namespace turbo {
namespace {

class PqInt8TrainingMemoryTest
    : public ::testing::TestWithParam<core::IndexMeta::DataType> {
 protected:
  static constexpr uint32_t kDimension = 17;
  static constexpr uint32_t kChunks = 4;
  static constexpr size_t kCount = 512;

  core::IndexMeta input_meta() const {
    core::IndexMeta meta;
    meta.set_meta(GetParam(), kDimension);
    meta.set_metric("SquaredEuclidean", 0, ailego::Params());
    return meta;
  }

  static ailego::Params training_params() {
    ailego::Params params;
    params.set("num_chunk", kChunks);
    params.set("thread_count", 2u);
    params.set("markov_chain_length", 0u);
    return params;
  }

  template <core::IndexMeta::DataType Type, typename T>
  static core::IndexHolder::Pointer make_typed_holder(bool varied) {
    auto holder =
        std::make_shared<core::MultiPassIndexHolder<Type>>(kDimension);
    std::mt19937 generator(42);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    for (size_t i = 0; i < kCount; ++i) {
      ailego::NumericalVector<T> vector(kDimension);
      for (size_t d = 0; d < kDimension; ++d) {
        vector[d] = T(varied ? distribution(generator)
                             : static_cast<float>(d + 1) / 32.0f);
      }
      holder->emplace(i, vector);
    }
    return holder;
  }

  core::IndexHolder::Pointer make_holder(bool varied) const {
    if (GetParam() == core::IndexMeta::DT_FP16) {
      return make_typed_holder<core::IndexMeta::DT_FP16, ailego::Float16>(
          varied);
    }
    return make_typed_holder<core::IndexMeta::DT_FP32, float>(varied);
  }

  static std::vector<uint8_t> encode_all(
      const PqInt8Quantizer &quantizer,
      const core::IndexHolder::Pointer &holder) {
    const size_t length = quantizer.quantized_datapoint_vector_length();
    std::vector<uint8_t> codes(holder->count() * length);
    auto iterator = holder->create_iterator();
    for (size_t i = 0; iterator->is_valid(); iterator->next(), ++i) {
      quantizer.quantize_data(iterator->data(), codes.data() + i * length);
    }
    return codes;
  }
};

TEST_P(PqInt8TrainingMemoryTest, DisabledSdcMatchesDefaultTraining) {
  PqInt8Quantizer with_sdc;
  PqInt8Quantizer without_sdc;
  auto params = training_params();
  ASSERT_EQ(0, with_sdc.init(input_meta(), params));
  params.set("build_sdc_table", false);
  ASSERT_EQ(0, without_sdc.init(input_meta(), params));

  // Identical vectors remove KMC2's random initialization as a confounder:
  // independent trainings must produce byte-identical codebooks and codes.
  auto holder = make_holder(false);
  ASSERT_EQ(0, with_sdc.train(holder));
  ASSERT_EQ(0, without_sdc.train(holder));
  EXPECT_EQ(kChunks * 256u * 256u, with_sdc.dist_table_.size());
  EXPECT_EQ(0u, without_sdc.dist_table_.size());
  EXPECT_EQ(0u, without_sdc.dist_table_.capacity());

  std::string default_blob;
  std::string bounded_blob;
  ASSERT_EQ(0, with_sdc.serialize(&default_blob));
  ASSERT_EQ(0, without_sdc.serialize(&bounded_blob));
  EXPECT_EQ(default_blob, bounded_blob);
  EXPECT_EQ(encode_all(with_sdc, holder), encode_all(without_sdc, holder));
}

TEST_P(PqInt8TrainingMemoryTest, DerivedSdcDoesNotChangeEncodingState) {
  PqInt8Quantizer quantizer;
  auto params = training_params();
  params.set("build_sdc_table", false);
  ASSERT_EQ(0, quantizer.init(input_meta(), params));
  auto holder = make_holder(true);
  ASSERT_EQ(0, quantizer.train(holder));
  ASSERT_EQ(0u, quantizer.dist_table_.capacity());

  std::string before;
  ASSERT_EQ(0, quantizer.serialize(&before));
  auto codes = encode_all(quantizer, holder);
  auto iterator = holder->create_iterator();
  ASSERT_TRUE(iterator->is_valid());
  std::vector<float> lut(quantizer.quantized_query_vector_length() /
                         sizeof(float));
  quantizer.quantize_query(iterator->data(), lut.data());
  const float distance =
      quantizer.calc_distance_dp_query(codes.data(), lut.data());

  // Hold a non-degenerate codebook fixed and add only the skipped derivative.
  quantizer.compute_dist_table();
  ASSERT_EQ(kChunks * 256u * 256u, quantizer.dist_table_.size());
  std::string after;
  ASSERT_EQ(0, quantizer.serialize(&after));
  EXPECT_EQ(before, after);
  EXPECT_EQ(codes, encode_all(quantizer, holder));
  std::vector<float> after_lut(lut.size());
  quantizer.quantize_query(iterator->data(), after_lut.data());
  EXPECT_EQ(lut, after_lut);
  EXPECT_EQ(distance,
            quantizer.calc_distance_dp_query(codes.data(), after_lut.data()));
}

TEST_P(PqInt8TrainingMemoryTest, ReinitReleasesTableAndRestoresDefault) {
  PqInt8Quantizer quantizer;
  auto params = training_params();
  auto holder = make_holder(false);
  ASSERT_EQ(0, quantizer.init(input_meta(), params));
  ASSERT_EQ(0, quantizer.train(holder));
  ASSERT_GT(quantizer.dist_table_.capacity(), 0u);

  params.set("build_sdc_table", false);
  ASSERT_EQ(0, quantizer.init(input_meta(), params));
  EXPECT_EQ(0u, quantizer.dist_table_.capacity());
  ASSERT_EQ(0, quantizer.train(holder));
  EXPECT_EQ(0u, quantizer.dist_table_.capacity());

  ASSERT_EQ(0, quantizer.init(input_meta(), training_params()));
  ASSERT_EQ(0, quantizer.train(holder));
  EXPECT_EQ(kChunks * 256u * 256u, quantizer.dist_table_.size());
}

TEST_P(PqInt8TrainingMemoryTest, RejectsMalformedFlag) {
  PqInt8Quantizer quantizer;
  auto params = training_params();
  params.set("build_sdc_table", std::string("not-a-boolean"));
  EXPECT_NE(0, quantizer.init(input_meta(), params));
  params.set("build_sdc_table", false);
  EXPECT_EQ(0, quantizer.init(input_meta(), params));
}

INSTANTIATE_TEST_SUITE_P(InputPrecisions, PqInt8TrainingMemoryTest,
                         ::testing::Values(core::IndexMeta::DT_FP16,
                                           core::IndexMeta::DT_FP32));

}  // namespace
}  // namespace turbo
}  // namespace zvec
