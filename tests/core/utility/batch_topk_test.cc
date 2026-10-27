// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "utility/batch_topk.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/internal/platform.h>

namespace zvec {
namespace core {

class BatchTopKTestPeer {
 public:
  using Implementation = BatchTopK::Implementation;

  static Implementation select(bool sse2, bool avx2, bool avx512f, bool neon,
                               bool avx_state, bool avx512_state) {
    return BatchTopK::select_implementation(sse2, avx2, avx512f, neon,
                                            avx_state, avx512_state);
  }

  static Implementation runtime() {
    return BatchTopK::implementation_for_runtime();
  }
};

}  // namespace core
}  // namespace zvec

using zvec::core::BatchTopK;
using zvec::core::BatchTopKImpl;
using zvec::core::BatchTopKTestPeer;

namespace {

using ScoredId = std::pair<float, uint32_t>;
using Implementation = BatchTopKTestPeer::Implementation;

#if !defined(ZVEC_BATCH_TOPK_FORCE_SCALAR) && \
    (defined(__SSE2__) || defined(_M_X64) ||  \
     (defined(_M_IX86_FP) && _M_IX86_FP >= 2))
constexpr Implementation kSSE2Fallback = Implementation::kSSE2;
#else
constexpr Implementation kSSE2Fallback = Implementation::kScalar;
#endif

#if !defined(ZVEC_BATCH_TOPK_FORCE_SCALAR) && \
    (defined(__AVX2__) || defined(ZVEC_BATCH_TOPK_HAVE_AVX2))
constexpr Implementation kAVX2Fallback = Implementation::kAVX2;
#else
constexpr Implementation kAVX2Fallback = kSSE2Fallback;
#endif

#if !defined(ZVEC_BATCH_TOPK_FORCE_SCALAR) && \
    (defined(__AVX512F__) || defined(ZVEC_BATCH_TOPK_HAVE_AVX512))
constexpr Implementation kAVX512Fallback = Implementation::kAVX512;
#else
constexpr Implementation kAVX512Fallback = kAVX2Fallback;
#endif

#if !defined(ZVEC_BATCH_TOPK_FORCE_SCALAR) && defined(AILEGO_ARM64_NEON)
constexpr Implementation kNEONFallback = Implementation::kNEON;
#else
constexpr Implementation kNEONFallback = Implementation::kScalar;
#endif

bool is_compiled(Implementation implementation) {
  switch (implementation) {
    case Implementation::kSSE2:
      return kSSE2Fallback == implementation;
    case Implementation::kAVX2:
      return kAVX2Fallback == implementation;
    case Implementation::kAVX512:
      return kAVX512Fallback == implementation;
    case Implementation::kNEON:
      return kNEONFallback == implementation;
    default:
      return true;
  }
}

bool can_execute(Implementation implementation) {
  const auto runtime = BatchTopK::implementation();
  switch (implementation) {
    case Implementation::kSSE2:
      return runtime == Implementation::kSSE2 ||
             runtime == Implementation::kAVX2 ||
             runtime == Implementation::kAVX512;
    case Implementation::kAVX2:
      return runtime == Implementation::kAVX2 ||
             runtime == Implementation::kAVX512;
    case Implementation::kAVX512:
    case Implementation::kNEON:
      return runtime == implementation;
    default:
      return true;
  }
}

template <typename ImplementationTag>
class BatchTopKTypedTest : public ::testing::Test {
 protected:
  using TopK = BatchTopKImpl<ImplementationTag::value>;

  void SetUp() override {
    // An implementation without its SIMD code still has a scalar fallback;
    // exercise that fallback with the same complete correctness suite.
    if (is_compiled(ImplementationTag::value) &&
        !can_execute(ImplementationTag::value)) {
      GTEST_SKIP() << "SIMD implementation is not supported by this CPU/OS";
    }
  }
};

using BatchTopKTypes = ::testing::Types<
    std::integral_constant<Implementation, Implementation::kScalar>,
    std::integral_constant<Implementation, Implementation::kSSE2>,
    std::integral_constant<Implementation, Implementation::kAVX2>,
    std::integral_constant<Implementation, Implementation::kAVX512>,
    std::integral_constant<Implementation, Implementation::kNEON>>;
TYPED_TEST_SUITE(BatchTopKTypedTest, BatchTopKTypes);

// Sort the complete input independently of the collector's batching or
// pruning strategy. A secondary ID order makes comparisons unambiguous for
// the unique-score cases; the collector does not promise an order for ties.
std::vector<ScoredId> reference_topk(const std::vector<float> &distances,
                                     uint32_t first_id, size_t k,
                                     float threshold, bool strict,
                                     float scale) {
  std::vector<ScoredId> result;
  for (size_t i = 0; i < distances.size(); ++i) {
    const float score = distances[i] * scale;
    if (strict ? score < threshold : score <= threshold) {
      result.emplace_back(score, first_id + static_cast<uint32_t>(i));
    }
  }
  std::sort(result.begin(), result.end());
  if (result.size() > k) {
    result.resize(k);
  }
  return result;
}

void expect_matches_reference(const BatchTopK &topk,
                              const std::vector<float> &distances,
                              uint32_t first_id, size_t k, float threshold,
                              bool strict, float scale,
                              bool compare_exact_ids) {
  const auto expected =
      reference_topk(distances, first_id, k, threshold, strict, scale);
  ASSERT_EQ(expected.size(), topk.size());
  std::vector<ScoredId> actual;
  std::unordered_set<uint32_t> seen;
  for (size_t i = 0; i < topk.size(); ++i) {
    const auto &candidate = topk.data()[i];
    EXPECT_FALSE(std::isnan(candidate.score));
    ASSERT_GE(candidate.id, first_id);
    const size_t offset = static_cast<size_t>(candidate.id - first_id);
    ASSERT_LT(offset, distances.size());
    EXPECT_EQ(distances[offset] * scale, candidate.score);
    EXPECT_TRUE(seen.insert(candidate.id).second);
    actual.emplace_back(candidate.score, candidate.id);
  }
  std::sort(actual.begin(), actual.end());
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_EQ(expected[i].first, actual[i].first);
    if (compare_exact_ids) {
      EXPECT_EQ(expected[i].second, actual[i].second);
    }
  }
}

template <typename TopK>
void push_batches(TopK *topk, const std::vector<float> &distances,
                  uint32_t first_id, size_t block_size, float scale) {
  // Use an unaligned start and partial final blocks. The input is not
  // required to have any SIMD alignment or readable trailing padding.
  std::vector<float> unaligned(distances.size() + 1);
  std::copy(distances.begin(), distances.end(), unaligned.begin() + 1);
  for (size_t offset = 0; offset < distances.size(); offset += block_size) {
    topk->push_block(unaligned.data() + 1 + offset,
                     std::min(block_size, distances.size() - offset),
                     first_id + static_cast<uint32_t>(offset), scale);
  }
  topk->finish();
}

TEST(BatchTopK, DispatchRequiresCompiledKernelsCpuFeaturesAndOsState) {
  // These are selector-only checks: simulated CPU features must never execute
  // an instruction set that the machine running this test does not support.
  EXPECT_EQ(Implementation::kScalar,
            BatchTopKTestPeer::select(false, false, false, false, true, true));
  EXPECT_EQ(kSSE2Fallback,
            BatchTopKTestPeer::select(true, false, false, false, false, false));
  // CpuFeatures owns register-state decoding. No AVX implementation may run
  // unless its shared OS state flags permit it, even with every CPU bit set.
  for (bool avx512_state : {false, true}) {
    SCOPED_TRACE(avx512_state);
    EXPECT_EQ(kSSE2Fallback, BatchTopKTestPeer::select(true, true, true, false,
                                                       false, avx512_state));
  }
  EXPECT_EQ(kAVX2Fallback,
            BatchTopKTestPeer::select(true, true, true, false, true, false));
  EXPECT_EQ(kAVX2Fallback,
            BatchTopKTestPeer::select(true, true, false, false, true, true));
  EXPECT_EQ(kSSE2Fallback,
            BatchTopKTestPeer::select(true, false, true, false, true, true));
  EXPECT_EQ(kAVX512Fallback,
            BatchTopKTestPeer::select(true, true, true, false, true, true));
  EXPECT_EQ(kNEONFallback,
            BatchTopKTestPeer::select(false, false, false, true, false, false));
}

TEST(BatchTopK, RuntimeImplementationIsCachedAndExecutable) {
  const auto implementation = BatchTopK::implementation();
  EXPECT_EQ(BatchTopKTestPeer::runtime(), implementation);
  EXPECT_TRUE(is_compiled(implementation));
  for (size_t i = 0; i < 10; ++i) {
    EXPECT_EQ(implementation, BatchTopK::implementation());
  }
#if defined(ZVEC_BATCH_TOPK_FORCE_SCALAR)
  EXPECT_EQ(Implementation::kScalar, implementation);
#elif defined(AILEGO_ARM64_NEON)
  EXPECT_EQ(Implementation::kNEON, implementation);
#elif defined(__x86_64__) || defined(_M_X64)
  // SSE2 is part of the x86_64 baseline; a portable build must still select
  // it when neither AVX variant can execute on the machine running the test.
  EXPECT_NE(Implementation::kScalar, implementation);
  EXPECT_NE(Implementation::kNEON, implementation);
#endif
}

template <Implementation Impl>
void expect_uncompiled_implementation_uses_scalar() {
  if (is_compiled(Impl)) {
    return;
  }
  // Only absent implementations are safe to call without a CPU/OS check:
  // their out-of-line entry points must use the scalar fallback. SIMD variants
  // are exercised by the typed tests only when this machine can execute them.
  SCOPED_TRACE(static_cast<int>(Impl));
  BatchTopKImpl<Impl> topk;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const std::vector<float> distances{9.0f, nan,  4.0f, 8.0f, -3.0f,
                                     8.0f, 2.0f, 1.0f, -8.0f};
  for (float scale : {1.0f, -0.5f}) {
    for (bool strict : {false, true}) {
      topk.reset(3, 2.0f, strict);
      push_batches(&topk, distances, 0x80000025, 7, scale);
      expect_matches_reference(topk, distances, 0x80000025, 3, 2.0f, strict,
                               scale, true);
    }
  }
}

TEST(BatchTopK, MissingImplementationsUseScalarStubs) {
  expect_uncompiled_implementation_uses_scalar<Implementation::kSSE2>();
  expect_uncompiled_implementation_uses_scalar<Implementation::kAVX2>();
  expect_uncompiled_implementation_uses_scalar<Implementation::kAVX512>();
  expect_uncompiled_implementation_uses_scalar<Implementation::kNEON>();
}

TYPED_TEST(BatchTopKTypedTest,
           MatchesSortedReferenceAcrossBatchAndCapacityBoundaries) {
  constexpr uint32_t first_id = 107;
  constexpr size_t count = 4099;
  const float threshold = std::numeric_limits<float>::max();
  std::vector<float> distances(count);
  for (size_t i = 0; i < count; ++i) {
    distances[i] = (static_cast<float>(i) - 2049.0f) * 0.25f;
  }
  std::mt19937 generator(81273);
  std::shuffle(distances.begin(), distances.end(), generator);

  typename TestFixture::TopK topk;
  for (size_t k : {0U, 1U, 7U, 16U, 31U, 32U, 33U, 63U, 64U, 65U, 127U, 128U,
                   129U, 511U, 512U, 513U, 4099U, 5000U}) {
    SCOPED_TRACE(k);
    for (size_t block_size : {1U, 3U, 4U, 7U, 8U, 15U, 16U, 17U, 31U, 32U, 33U,
                              127U, 255U, 511U, 512U, 513U, 4099U}) {
      SCOPED_TRACE(block_size);
      topk.reset(k);
      push_batches(&topk, distances, first_id, block_size, 1.0f);
      expect_matches_reference(topk, distances, first_id, k, threshold, false,
                               1.0f, true);
    }
  }
}

TYPED_TEST(BatchTopKTypedTest,
           ThresholdIsOrderedAndHonorsInclusiveAndStrictBoundaries) {
  const float infinity = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const std::vector<float> pattern{nan,
                                   infinity,
                                   -infinity,
                                   -2.0f,
                                   -1.0f,
                                   -0.0f,
                                   0.0f,
                                   1.0f,
                                   2.0f,
                                   3.0f,
                                   std::numeric_limits<float>::max(),
                                   -infinity,
                                   2.0f,
                                   2.0f,
                                   2.0f,
                                   nan};
  std::vector<float> distances(1031);
  for (size_t i = 0; i < distances.size(); ++i) {
    distances[i] = pattern[i % pattern.size()];
  }

  typename TestFixture::TopK topk;
  for (float threshold : {-infinity, -2.0f, -0.0f, 2.0f,
                          std::numeric_limits<float>::max(), infinity, nan}) {
    SCOPED_TRACE(threshold);
    for (bool strict : {false, true}) {
      SCOPED_TRACE(strict);
      for (size_t k : {1U, 17U, 64U, 2000U}) {
        SCOPED_TRACE(k);
        for (size_t block_size : {3U, 7U, 16U, 31U, 512U}) {
          SCOPED_TRACE(block_size);
          topk.reset(k, threshold, strict);
          push_batches(&topk, distances, 91, block_size, 1.0f);
          expect_matches_reference(topk, distances, 91, k, threshold, strict,
                                   1.0f, false);
        }
      }
    }
  }
}

TYPED_TEST(BatchTopKTypedTest, SelectionAndThresholdUseScaledScores) {
  const float infinity = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  std::vector<float> distances(1031);
  for (size_t i = 0; i < distances.size(); ++i) {
    distances[i] = static_cast<float>((i * 67) % distances.size()) - 515.0f;
  }
  distances[3] = infinity;
  distances[17] = -infinity;
  distances[511] = nan;

  typename TestFixture::TopK topk;
  for (float scale : {0.25f, -2.0f, 0.0f, -0.0f, infinity, nan}) {
    SCOPED_TRACE(scale);
    for (float threshold : {-10.0f, 0.0f, 10.0f, infinity}) {
      SCOPED_TRACE(threshold);
      for (bool strict : {false, true}) {
        SCOPED_TRACE(strict);
        for (size_t k : {1U, 17U, 2000U}) {
          SCOPED_TRACE(k);
          for (size_t block_size : {3U, 16U, 33U, 512U}) {
            SCOPED_TRACE(block_size);
            topk.reset(k, threshold, strict);
            push_batches(&topk, distances, 2007, block_size, scale);
            expect_matches_reference(topk, distances, 2007, k, threshold,
                                     strict, scale, false);
          }
        }
      }
    }
  }
}

TYPED_TEST(BatchTopKTypedTest, PreservesAllIdBitsAndNonzeroBatchOffsets) {
  constexpr size_t count = 1031;
  std::vector<float> distances(count);
  for (size_t i = 0; i < count; ++i) {
    distances[i] = static_cast<float>((i * 67) % count);
  }
  typename TestFixture::TopK topk;
  for (uint32_t first_id :
       {uint32_t{0x80000025}, std::numeric_limits<uint32_t>::max() -
                                  static_cast<uint32_t>(count) + 1}) {
    SCOPED_TRACE(first_id);
    for (size_t k : {1U, 17U, 1031U}) {
      SCOPED_TRACE(k);
      for (size_t block_size : {7U, 17U, 512U, 1031U}) {
        SCOPED_TRACE(block_size);
        topk.reset(k);
        push_batches(&topk, distances, first_id, block_size, 1.0f);
        expect_matches_reference(topk, distances, first_id, k,
                                 std::numeric_limits<float>::max(), false, 1.0f,
                                 true);
      }
    }
  }
}

TYPED_TEST(BatchTopKTypedTest, AcceptsAdditionalBatchesAfterFinish) {
  std::vector<float> distances(2065);
  for (size_t i = 0; i < distances.size(); ++i) {
    distances[i] = static_cast<float>(distances.size() - i) * 0.25f;
  }
  constexpr uint32_t first_id = 20000;
  constexpr float threshold = 300.0f;
  typename TestFixture::TopK topk;
  for (size_t k : {1U, 17U, 65U, 513U, 4096U}) {
    SCOPED_TRACE(k);
    topk.reset(k, threshold);
    std::vector<float> prefix;
    for (size_t offset = 0; offset < distances.size(); offset += 137) {
      SCOPED_TRACE(offset);
      const size_t count = std::min(size_t{137}, distances.size() - offset);
      topk.push_block(distances.data() + offset, count,
                      first_id + static_cast<uint32_t>(offset));
      prefix.insert(prefix.end(), distances.begin() + offset,
                    distances.begin() + offset + count);
      topk.finish();
      expect_matches_reference(topk, prefix, first_id, k, threshold, false,
                               1.0f, true);
    }
  }
}

TYPED_TEST(BatchTopKTypedTest,
           EmptyBlocksAndResetDiscardPreviousCandidatesAndThreshold) {
  typename TestFixture::TopK topk;
  topk.reset(10);
  topk.push_block(nullptr, 0, 0);
  topk.finish();
  EXPECT_EQ(0U, topk.size());

  std::vector<float> descending(2053);
  std::iota(descending.rbegin(), descending.rend(), -1000.0f);
  const std::vector<float> subsequent{50.0f, 20.0f, 40.0f, 10.0f, 30.0f};
  for (size_t previous_k : {1U, 17U, 1024U, 4096U}) {
    SCOPED_TRACE(previous_k);
    topk.reset(previous_k, -100.0f, true);
    push_batches(&topk, descending, 1000, 512, 1.0f);
    expect_matches_reference(topk, descending, 1000, previous_k, -100.0f, true,
                             1.0f, true);

    topk.reset(0);
    topk.push_block(subsequent.data(), subsequent.size(), 100);
    topk.finish();
    EXPECT_EQ(0U, topk.size());

    topk.reset(3);
    EXPECT_EQ(0U, topk.size());
    topk.push_block(nullptr, 0, 0);
    push_batches(&topk, subsequent, 100, 3, 1.0f);
    topk.push_block(nullptr, 0, 1000);
    topk.finish();
    expect_matches_reference(topk, subsequent, 100, 3,
                             std::numeric_limits<float>::max(), false, 1.0f,
                             true);
    // Finishing twice must not discard any additional retained candidate.
    topk.finish();
    expect_matches_reference(topk, subsequent, 100, 3,
                             std::numeric_limits<float>::max(), false, 1.0f,
                             true);
  }
}

TYPED_TEST(BatchTopKTypedTest,
           OverflowResetPreservesThePreviousSelectionState) {
  typename TestFixture::TopK topk;
  const std::vector<float> distances{9.0f, 4.0f, 8.0f, 3.0f, 8.0f, 2.0f};
  constexpr uint32_t first_id = 101;
  topk.reset(4, 8.0f, true);
  topk.push_block(distances.data(), 3, first_id);
  EXPECT_THROW(topk.reset(std::numeric_limits<size_t>::max()),
               std::length_error);
  topk.push_block(distances.data() + 3, distances.size() - 3, first_id + 3);
  topk.finish();
  expect_matches_reference(topk, distances, first_id, 4, 8.0f, true, 1.0f,
                           true);
}

}  // namespace
