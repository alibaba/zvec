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
#include <cstring>
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
#include "ailego/pattern/defer.h"
#include "quantizer/common/pq_training_samples.h"
#include "quantizer/pq_fast_quantizer/pq_fast_quantizer.h"
#include "quantizer/pq_int4_quantizer/pq_int4_quantizer.h"
#include "quantizer/pq_int8_quantizer/pq_int8_quantizer.h"
#include "../ailego/buffer/pool_payload_budget.h"

namespace zvec {
namespace turbo {

// Keep the production declaration unchanged in every translation unit. Using
// a private-to-public macro would change MSVC's mangled member-function names.
class PqInt8QuantizerTestAccess {
 public:
  static const std::vector<float> &dist_table(
      const PqInt8Quantizer &quantizer) {
    return quantizer.dist_table_;
  }

  static int compute_dist_table(PqInt8Quantizer &quantizer) {
    return quantizer.compute_dist_table();
  }
  static size_t paged_bytes(const PqInt8Quantizer &quantizer) {
    return quantizer.paged_dist_table_ ? quantizer.paged_dist_table_->size()
                                       : 0;
  }
};

namespace {

template <typename QuantizerType>
void check_small_codebook_budget() {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  pool.init(4 * 1024 * 1024);
  auto reset = ailego::ScopeGuard::Make([&] { pool.init(0); });
  {
    QuantizerType quantizer;
    core::IndexMeta meta;
    meta.set_meta(core::IndexMeta::DT_FP32, 32);
    meta.set_metric("SquaredEuclidean", 0, ailego::Params());
    ailego::Params params;
    params.set("num_chunk", 8u);
    params.set("thread_count", 1u);
    params.set("markov_chain_length", 0u);
    ASSERT_EQ(0, quantizer.init(meta, params));
    const size_t initial = pool.external_used();
    EXPECT_GE(initial, 32u * 16 * sizeof(float));
    auto holder =
        std::make_shared<core::MultiPassIndexHolder<core::IndexMeta::DT_FP32>>(
            32);
    for (size_t i = 0; i < 128; ++i) {
      ailego::NumericalVector<float> vector(32);
      for (size_t d = 0; d < 32; ++d) vector[d] = float((i * 17 + d) % 31);
      holder->emplace(i, vector);
    }
    ASSERT_EQ(0, quantizer.train(holder));
    const auto resident = pool.external_used();
    EXPECT_GT(resident, initial);
    std::string before, after, table;
    ASSERT_EQ(0, quantizer.serialize(&before));
    std::vector<float> center(32, 0.0f);
    ASSERT_EQ(
        0, quantizer.build_centroid_distance_table(center.data(), 1, &table));
    EXPECT_EQ(resident, pool.external_used());
    ASSERT_EQ(0, quantizer.serialize(&after));
    EXPECT_EQ(before, after);
    ASSERT_EQ(0, quantizer.init(meta, params));
    EXPECT_EQ(initial, pool.external_used());
    ASSERT_EQ(0, quantizer.deserialize(before));
    ASSERT_EQ(
        0, quantizer.build_centroid_distance_table(center.data(), 1, &after));
    EXPECT_EQ(table, after);
  }
  EXPECT_EQ(0u, pool.external_used());
  pool.init(64 * 1024);
  {
    QuantizerType quantizer;
    core::IndexMeta meta;
    meta.set_meta(core::IndexMeta::DT_FP32, 2048);
    ailego::Params params;
    params.set("num_chunk", 128u);
    EXPECT_EQ(core::IndexError_NoMemory, quantizer.init(meta, params));
  }
  EXPECT_EQ(0u, pool.external_used());
}

TEST(PqModelMemoryTest, Int4ModelsAreChargedAndReinitReleasesThem) {
  check_small_codebook_budget<PqInt4Quantizer>();
}
TEST(PqModelMemoryTest, FastModelsAreChargedAndReinitReleasesThem) {
  check_small_codebook_budget<PqFastQuantizer>();
}

TEST(PqInt8ModelMemoryTest, CodebookIsChargedAndNormsAreOnlyTemporary) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  pool.init(8 * 1024 * 1024);
  {
    PqInt8Quantizer quantizer;
    core::IndexMeta meta;
    meta.set_meta(core::IndexMeta::DT_FP32, 768);
    meta.set_metric("SquaredEuclidean", 0, ailego::Params());
    ailego::Params params;
    params.set("num_chunk", 384u);
    params.set("build_sdc_table", false);
    ASSERT_EQ(0, quantizer.init(meta, params));
    EXPECT_GE(pool.external_used(), 768 * 256 * sizeof(float));
    std::vector<float> codebook(768 * 256, 0.0f);
    ASSERT_EQ(0, quantizer.import_codebook(codebook.data(),
                                           codebook.size() * sizeof(float)));
    const size_t resident = pool.external_used();
    EXPECT_GE(resident, 768 * 256 * sizeof(float) + 384 * 256 * sizeof(void *));
    EXPECT_LT(resident, 1600 * 1024u);
    std::vector<float> center(768, 0.0f);
    std::string table;
    ASSERT_EQ(
        0, quantizer.build_centroid_distance_table(center.data(), 1, &table));
    EXPECT_EQ(384 * 256 * sizeof(float), table.size());
    EXPECT_EQ(resident, pool.external_used());
  }
  EXPECT_EQ(0u, pool.external_used());
  pool.init(0);
}

TEST(PqInt8ModelMemoryTest, BudgetDenialIsReportedAndDoesNotLeak) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  pool.init(64 * 1024);
  {
    PqInt8Quantizer quantizer;
    core::IndexMeta meta;
    meta.set_meta(core::IndexMeta::DT_FP32, 768);
    ailego::Params params;
    params.set("num_chunk", 384u);
    EXPECT_EQ(core::IndexError_NoMemory, quantizer.init(meta, params));
    EXPECT_EQ(0u, pool.external_used());
  }
  pool.init(0);
}

using PqTestAccess = PqInt8QuantizerTestAccess;

TEST(PqSamplingMemoryTest, OrdinalWorkspaceIsChargedAndReleased) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t previous = pool.capacity();
  ASSERT_EQ(0u, pool.used());
  ASSERT_EQ(0, pool.init(128 * 1024));
  AILEGO_DEFER([&]() {
    EXPECT_EQ(0u, pool.used());
    EXPECT_EQ(0, pool.init(previous));
  });
  auto holder =
      std::make_shared<core::MultiPassIndexHolder<core::IndexMeta::DT_FP32>>(1);
  for (size_t i = 0; i < 256; ++i) {
    ailego::NumericalVector<float> row(1);
    row[0] = static_cast<float>(i);
    holder->emplace(i, row);
  }
  size_t copied = 0;
  auto write = [&](size_t, const void *) {
    ++copied;
    EXPECT_GE(pool.external_used(), 128 * 2 * sizeof(size_t));
    return true;
  };
  EXPECT_TRUE(CollectPqTrainingSamplesTo(holder, 128, sizeof(float), write));
  EXPECT_EQ(128u, copied);
  EXPECT_EQ(0u, pool.external_used());
  {
    ailego::MemoryReservation pressure;
    pressure.add(pool.capacity() - 128);
    const size_t charged = pool.external_used();
    EXPECT_THROW(CollectPqTrainingSamplesTo(holder, 128, sizeof(float), write),
                 std::bad_alloc);
    EXPECT_EQ(128u, copied);
    EXPECT_EQ(charged, pool.external_used());
  }
}

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

  void check_derived_sdc(bool varied);
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
  EXPECT_EQ(kChunks * 256u * 256u, PqTestAccess::dist_table(with_sdc).size());
  EXPECT_EQ(0u, PqTestAccess::dist_table(without_sdc).size());
  EXPECT_EQ(0u, PqTestAccess::dist_table(without_sdc).capacity());

  std::string default_blob;
  std::string bounded_blob;
  ASSERT_EQ(0, with_sdc.serialize(&default_blob));
  ASSERT_EQ(0, without_sdc.serialize(&bounded_blob));
  EXPECT_EQ(default_blob, bounded_blob);
  EXPECT_EQ(encode_all(with_sdc, holder), encode_all(without_sdc, holder));
}

void PqInt8TrainingMemoryTest::check_derived_sdc(bool varied) {
  PqInt8Quantizer quantizer;
  auto params = training_params();
  params.set("build_sdc_table", false);
  ASSERT_EQ(0, quantizer.init(input_meta(), params));
  auto holder = make_holder(varied);
  ASSERT_EQ(0, quantizer.train(holder));
  ASSERT_EQ(0u, PqTestAccess::dist_table(quantizer).capacity());

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
  ASSERT_TRUE(std::isfinite(distance));
  if (!varied) {
    // Identical vectors deterministically leave empty clusters. Their NaN
    // centers exercise the same case that can occur with random training.
    ASSERT_TRUE(std::any_of(lut.begin(), lut.end(),
                            [](float value) { return std::isnan(value); }));
  }

  // Hold the trained codebook fixed and add only the skipped derivative.
  ASSERT_EQ(0, PqTestAccess::compute_dist_table(quantizer));
  ASSERT_EQ(kChunks * 256u * 256u, PqTestAccess::dist_table(quantizer).size());
  std::string after;
  ASSERT_EQ(0, quantizer.serialize(&after));
  EXPECT_EQ(before, after);
  EXPECT_EQ(codes, encode_all(quantizer, holder));
  std::vector<float> after_lut(lut.size());
  quantizer.quantize_query(iterator->data(), after_lut.data());
  // Empty clusters produce NaN LUT entries: NaN != NaN even when every bit
  // is unchanged. Compare representations, retaining a strict invariant for
  // both finite values and NaNs instead of relaxing floating-point tolerance.
  for (size_t i = 0; i < lut.size(); ++i) {
    EXPECT_EQ(0, std::memcmp(&lut[i], &after_lut[i], sizeof(float)))
        << "LUT entry " << i;
  }
  EXPECT_EQ(distance,
            quantizer.calc_distance_dp_query(codes.data(), after_lut.data()));
}

TEST_P(PqInt8TrainingMemoryTest, DerivedSdcDoesNotChangeEncodingState) {
  check_derived_sdc(true);
}

TEST_P(PqInt8TrainingMemoryTest, DerivedSdcPreservesEmptyClusterLutEntries) {
  check_derived_sdc(false);
}

TEST_P(PqInt8TrainingMemoryTest, ReinitReleasesTableAndRestoresDefault) {
  PqInt8Quantizer quantizer;
  auto params = training_params();
  auto holder = make_holder(false);
  ASSERT_EQ(0, quantizer.init(input_meta(), params));
  ASSERT_EQ(0, quantizer.train(holder));
  ASSERT_GT(PqTestAccess::dist_table(quantizer).capacity(), 0u);

  params.set("build_sdc_table", false);
  ASSERT_EQ(0, quantizer.init(input_meta(), params));
  EXPECT_EQ(0u, PqTestAccess::dist_table(quantizer).capacity());
  ASSERT_EQ(0, quantizer.train(holder));
  EXPECT_EQ(0u, PqTestAccess::dist_table(quantizer).capacity());

  ASSERT_EQ(0, quantizer.init(input_meta(), training_params()));
  ASSERT_EQ(0, quantizer.train(holder));
  EXPECT_EQ(kChunks * 256u * 256u, PqTestAccess::dist_table(quantizer).size());
}

TEST_P(PqInt8TrainingMemoryTest, PagedSdcMatchesHeapAndHandleOwnsLifetime) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t previous = pool.capacity();
  auto restore = ailego::ScopeGuard::Make([&] {
    EXPECT_EQ(0u, pool.used());
    EXPECT_EQ(0, pool.init(previous));
  });
  for (const char *metric : {"SquaredEuclidean", "InnerProduct", "Cosine"}) {
    ASSERT_EQ(0u, pool.used());
    ASSERT_EQ(0, pool.init(0));
    PqInt8Quantizer quantizer;
    auto meta = input_meta();
    meta.set_metric(metric, 0, ailego::Params());
    auto params = training_params();
    ASSERT_EQ(0, quantizer.init(meta, params));
    auto holder = make_holder(true);
    ASSERT_EQ(0, quantizer.train(holder));
    auto codes = encode_all(quantizer, holder);
    const size_t stride = quantizer.quantized_datapoint_vector_length();
    std::vector<float> expected(32);
    for (size_t i = 0; i < expected.size(); ++i)
      expected[i] = quantizer.calc_distance_dp_dp(codes.data(),
                                                  codes.data() + i * stride);
    // Account for platform-dependent writable metadata (16 KiB macOS pages),
    // but leave fewer resident pages than the 1 MiB SDC table needs.
    const size_t budget =
        ailego::VecBufferPool::metadata_bytes_for_page_count(256, true) +
        32 * ailego::kVectorPageSize;
    ASSERT_EQ(0, pool.init(budget));
    ASSERT_EQ(0, PqTestAccess::compute_dist_table(quantizer));
    EXPECT_EQ(0u, PqTestAccess::dist_table(quantizer).capacity());
    EXPECT_EQ(kChunks * 256u * 256u * sizeof(float),
              PqTestAccess::paged_bytes(quantizer));
    auto handle = quantizer.sym_distance(codes.data(), core::IndexQueryMeta());
    for (size_t i = 0; i < expected.size(); ++i) {
      const void *code = codes.data() + i * stride;
      EXPECT_NEAR(expected[i],
                  quantizer.calc_distance_dp_dp(codes.data(), code), 1e-5f);
      EXPECT_NEAR(expected[i], handle(code), 1e-5f);
    }
    params.set("build_sdc_table", false);
    ASSERT_EQ(0, quantizer.init(meta, params));
    EXPECT_EQ(0u, PqTestAccess::paged_bytes(quantizer));
    EXPECT_NEAR(expected[1], handle(codes.data() + stride), 1e-5f);
    EXPECT_LE(pool.used(), pool.capacity());
  }
}

TEST_P(PqInt8TrainingMemoryTest, PooledTrainingBuildsSdcAndTinyBudgetFails) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t previous = pool.capacity();
  auto restore = ailego::ScopeGuard::Make([&] {
    EXPECT_EQ(0u, pool.used());
    EXPECT_EQ(0, pool.init(previous));
  });
  ASSERT_EQ(0, pool.init(32 * 1024 * 1024));
  auto holder = make_holder(true);
  std::string codebook;
  {
    PqInt8Quantizer quantizer;
    ASSERT_EQ(0, quantizer.init(input_meta(), training_params()));
    ASSERT_EQ(0, quantizer.train(holder));
    EXPECT_EQ(0u, PqTestAccess::dist_table(quantizer).capacity());
    EXPECT_GT(PqTestAccess::paged_bytes(quantizer), 0u);
    ASSERT_EQ(0, quantizer.serialize(&codebook));
  }
  ASSERT_EQ(0u, pool.used());
  ASSERT_EQ(0, pool.init(1));
  PqInt8Quantizer quantizer;
  EXPECT_EQ(core::IndexError_NoMemory,
            quantizer.init(input_meta(), training_params()));
  // Leave room for the now-budgeted model, then exhaust the remaining budget
  // to exercise SDC allocation failure independently of codebook admission.
  ASSERT_EQ(0, pool.init(1024 * 1024));
  ASSERT_EQ(0, quantizer.init(input_meta(), training_params()));
  ASSERT_EQ(0, quantizer.deserialize(codebook.data(), codebook.size()));
  const size_t blocked = pool.capacity() - pool.used() - 1;
  ASSERT_TRUE(pool.try_charge_external(blocked));
  auto unblock =
      ailego::ScopeGuard::Make([&] { pool.release_external(blocked); });
  EXPECT_EQ(core::IndexError_NoMemory,
            PqTestAccess::compute_dist_table(quantizer));
  EXPECT_EQ(0u, PqTestAccess::dist_table(quantizer).capacity());
  EXPECT_EQ(0u, PqTestAccess::paged_bytes(quantizer));
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
