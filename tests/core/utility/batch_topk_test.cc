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

using ScoredId = std::pair<float, uint64_t>;
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
                                     uint64_t first_id, size_t k,
                                     float threshold, bool strict) {
  std::vector<ScoredId> result;
  for (size_t i = 0; i < distances.size(); ++i) {
    const float score = distances[i];
    if (strict ? score < threshold : score <= threshold) {
      result.emplace_back(score, first_id + static_cast<uint64_t>(i));
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
                              uint64_t first_id, size_t k, float threshold,
                              bool strict, bool compare_exact_ids) {
  const auto expected =
      reference_topk(distances, first_id, k, threshold, strict);
  ASSERT_EQ(expected.size(), topk.size());
  std::vector<ScoredId> actual;
  std::unordered_set<uint64_t> seen;
  for (size_t i = 0; i < topk.size(); ++i) {
    const auto &candidate = topk.data()[i];
    EXPECT_FALSE(std::isnan(candidate.score));
    ASSERT_GE(candidate.id, first_id);
    const size_t offset = static_cast<size_t>(candidate.id - first_id);
    ASSERT_LT(offset, distances.size());
    EXPECT_EQ(distances[offset], candidate.score);
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
                  uint64_t first_id, size_t block_size) {
  // Use an unaligned start and partial final blocks. The input is not
  // required to have any SIMD alignment or readable trailing padding.
  std::vector<float> unaligned(distances.size() + 1);
  std::copy(distances.begin(), distances.end(), unaligned.begin() + 1);
  std::vector<uint64_t> indices(distances.size() + 1);
  std::iota(indices.begin() + 1, indices.end(), first_id);
  for (size_t offset = 0; offset < distances.size(); offset += block_size) {
    topk->push_block(unaligned.data() + 1 + offset, indices.data() + 1 + offset,
                     std::min(block_size, distances.size() - offset));
  }
  topk->finish();
}

TEST(BatchTopK, CandidateStoresOnlyDistanceAndFullWidthIndex) {
  EXPECT_EQ(8U, sizeof(BatchTopK::Candidate::id));
  EXPECT_TRUE(
      (std::is_same<decltype(BatchTopK::Candidate::id), uint64_t>::value));
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
  for (bool strict : {false, true}) {
    topk.reset(3, 2.0f, strict);
    push_batches(&topk, distances, 0x80000025, 7);
    expect_matches_reference(topk, distances, 0x80000025, 3, 2.0f, strict,
                             true);
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
  constexpr uint64_t first_id = 107;
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
      push_batches(&topk, distances, first_id, block_size);
      expect_matches_reference(topk, distances, first_id, k, threshold, false,
                               true);
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
          push_batches(&topk, distances, 91, block_size);
          expect_matches_reference(topk, distances, 91, k, threshold, strict,
                                   false);
        }
      }
    }
  }
}

TYPED_TEST(BatchTopKTypedTest, PreservesAllIndexBitsAcrossInputSlices) {
  constexpr size_t count = 1031;
  std::vector<float> distances(count);
  for (size_t i = 0; i < count; ++i) {
    distances[i] = static_cast<float>((i * 67) % count);
  }
  typename TestFixture::TopK topk;
  for (uint64_t first_id :
       {uint64_t{0x80000025},
        uint64_t{std::numeric_limits<uint32_t>::max()} - count + 1,
        uint64_t{1} << 40, std::numeric_limits<uint64_t>::max() - count + 1}) {
    SCOPED_TRACE(first_id);
    for (size_t k : {1U, 17U, 1031U}) {
      SCOPED_TRACE(k);
      for (size_t block_size : {7U, 17U, 512U, 1031U}) {
        SCOPED_TRACE(block_size);
        topk.reset(k);
        push_batches(&topk, distances, first_id, block_size);
        expect_matches_reference(topk, distances, first_id, k,
                                 std::numeric_limits<float>::max(), false,
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
  constexpr uint64_t first_id = 20000;
  std::vector<uint64_t> indices(distances.size());
  std::iota(indices.begin(), indices.end(), first_id);
  constexpr float threshold = 300.0f;
  typename TestFixture::TopK topk;
  for (size_t k : {1U, 17U, 65U, 513U, 4096U}) {
    SCOPED_TRACE(k);
    topk.reset(k, threshold);
    std::vector<float> prefix;
    for (size_t offset = 0; offset < distances.size(); offset += 137) {
      SCOPED_TRACE(offset);
      const size_t count = std::min(size_t{137}, distances.size() - offset);
      topk.push_block(distances.data() + offset, indices.data() + offset,
                      count);
      prefix.insert(prefix.end(), distances.begin() + offset,
                    distances.begin() + offset + count);
      topk.finish();
      expect_matches_reference(topk, prefix, first_id, k, threshold, false,
                               true);
    }
  }
}

TYPED_TEST(BatchTopKTypedTest,
           EmptyBlocksAndResetDiscardPreviousCandidatesAndThreshold) {
  typename TestFixture::TopK topk;
  topk.reset(10);
  topk.push_block(nullptr, nullptr, 0);
  topk.finish();
  EXPECT_EQ(0U, topk.size());

  std::vector<float> descending(2053);
  std::iota(descending.rbegin(), descending.rend(), -1000.0f);
  const std::vector<float> subsequent{50.0f, 20.0f, 40.0f, 10.0f, 30.0f};
  const std::vector<uint64_t> subsequent_indices{100, 101, 102, 103, 104};
  for (size_t previous_k : {1U, 17U, 1024U, 4096U}) {
    SCOPED_TRACE(previous_k);
    topk.reset(previous_k, -100.0f, true);
    push_batches(&topk, descending, 1000, 512);
    expect_matches_reference(topk, descending, 1000, previous_k, -100.0f, true,
                             true);

    topk.reset(0);
    topk.push_block(subsequent.data(), subsequent_indices.data(),
                    subsequent.size());
    topk.finish();
    EXPECT_EQ(0U, topk.size());

    topk.reset(3);
    EXPECT_EQ(0U, topk.size());
    topk.push_block(nullptr, nullptr, 0);
    push_batches(&topk, subsequent, 100, 3);
    topk.push_block(nullptr, nullptr, 0);
    topk.finish();
    expect_matches_reference(topk, subsequent, 100, 3,
                             std::numeric_limits<float>::max(), false, true);
    // Finishing twice must not discard any additional retained candidate.
    topk.finish();
    expect_matches_reference(topk, subsequent, 100, 3,
                             std::numeric_limits<float>::max(), false, true);
  }
}

TYPED_TEST(BatchTopKTypedTest,
           OverflowResetPreservesThePreviousSelectionState) {
  typename TestFixture::TopK topk;
  const std::vector<float> distances{9.0f, 4.0f, 8.0f, 3.0f, 8.0f, 2.0f};
  constexpr uint64_t first_id = 101;
  const std::vector<uint64_t> indices{101, 102, 103, 104, 105, 106};
  topk.reset(4, 8.0f, true);
  topk.push_block(distances.data(), indices.data(), 3);
  EXPECT_THROW(topk.reset(std::numeric_limits<size_t>::max()),
               std::length_error);
  topk.push_block(distances.data() + 3, indices.data() + 3,
                  distances.size() - 3);
  topk.finish();
  expect_matches_reference(topk, distances, first_id, 4, 8.0f, true, true);
}


using ScoredIndex = std::pair<float, uint64_t>;

// Independent full-sort oracle: no SIMD masks, compaction or batch boundaries.
std::vector<ScoredIndex> reference_indices(const std::vector<float> &scores,
                                           const std::vector<uint64_t> &indices,
                                           size_t k, float threshold,
                                           bool strict) {
  std::vector<ScoredIndex> result;
  for (size_t i = 0; i < scores.size(); ++i) {
    const float score = scores[i];
    if (strict ? score < threshold : score <= threshold) {
      result.emplace_back(score, indices[i]);
    }
  }
  std::sort(result.begin(), result.end());
  if (result.size() > k) result.resize(k);
  return result;
}

void expect_indices(const BatchTopK &topk,
                    const std::vector<ScoredIndex> &expected) {
  std::vector<ScoredIndex> actual;
  for (size_t i = 0; i < topk.size(); ++i) {
    actual.emplace_back(topk.data()[i].score, topk.data()[i].id);
  }
  std::sort(actual.begin(), actual.end());
  EXPECT_EQ(expected, actual);
}

template <typename TopK>
void check_explicit_indices() {
  constexpr size_t count = 1537;
  const float infinity = std::numeric_limits<float>::infinity();
  std::vector<float> scores(count);
  std::vector<uint64_t> indices(count);
  for (size_t i = 0; i < count; ++i) {
    scores[i] = static_cast<float>((i * 37) % count) - 768.0f;
    // Arbitrary nonmonotonic full-width indices; no batch-relative mapping.
    indices[i] = uint64_t{0xa5f17c09d384b620} ^
                 (static_cast<uint64_t>(i) * uint64_t{104729});
  }
  indices[0] = 0;
  indices[1] = std::numeric_limits<uint64_t>::max();
  indices[2] = uint64_t{1} << 63;
  indices[3] = std::numeric_limits<uint32_t>::max();
  indices[4] = uint64_t{1} << 32;
  // Equal low 32 bits must not make distinct full-width IDs interchangeable.
  indices[5] = 17;
  indices[6] = (uint64_t{1} << 32) | 17;
  indices[11] = (uint64_t{1} << 63) | 17;
  scores[7] = infinity;
  scores[8] = -infinity;
  scores[9] = std::numeric_limits<float>::quiet_NaN();
  TopK topk;
  for (size_t k : {0U, 1U, 10U, 100U, 200U, 4096U}) {
    for (size_t block : {1U, 7U, 32U, 512U, 1537U}) {
      for (float threshold : {0.0f, infinity}) {
        for (bool strict : {false, true}) {
          SCOPED_TRACE(::testing::Message() << k << " " << block << " "
                                            << threshold << " " << strict);
          topk.reset(k, threshold, strict);
          topk.push_block(nullptr, nullptr, 0);
          std::vector<float> distances(block + 1);
          std::vector<uint64_t> ids(block + 1);
          for (size_t offset = 0; offset < count; offset += block) {
            const size_t rows = std::min(block, count - offset);
            std::copy_n(scores.data() + offset, rows, distances.data() + 1);
            std::copy_n(indices.data() + offset, rows, ids.data() + 1);
            topk.push_block(distances.data() + 1, ids.data() + 1, rows);
            // Both arrays are caller-owned scratch, overwritten on every
            // flush. Retained indices must not alias their original slots.
            std::fill(distances.begin(), distances.end(), infinity);
            std::fill(ids.begin(), ids.end(), 0);
          }
          topk.finish();
          expect_indices(
              topk, reference_indices(scores, indices, k, threshold, strict));
        }
      }
    }
  }
}

TYPED_TEST(BatchTopKTypedTest, ExplicitIndicesSurviveBatchesAndScratchReuse) {
  check_explicit_indices<typename TestFixture::TopK>();
}

TEST(BatchTopK, RuntimeDispatchPreservesExplicitIndices) {
  check_explicit_indices<BatchTopK>();
}

template <typename TopK>
void check_full_width_indices_and_reset() {
  TopK topk;
  const std::vector<float> scores{9.0f, -7.0f, 3.0f, -2.0f, 1.0f};
  for (uint64_t first :
       {uint64_t{1} << 40, std::numeric_limits<uint64_t>::max() - 4}) {
    std::vector<uint64_t> indices(scores.size());
    std::iota(indices.begin(), indices.end(), first);
    topk.reset(3);
    topk.push_block(scores.data(), indices.data(), 2);
    topk.push_block(scores.data() + 2, indices.data() + 2, scores.size() - 2);
    topk.finish();
    expect_indices(topk, reference_indices(scores, indices, 3, 100.0f, false));

    // Reset must discard full-width IDs and the tightened previous bound.
    const std::vector<float> next{40.0f, 20.0f, 30.0f};
    const std::vector<uint64_t> ids{17, uint64_t{1} << 63, 17};
    topk.reset(2);
    topk.push_block(next.data(), ids.data(), 1);
    topk.finish();
    topk.push_block(next.data() + 1, ids.data() + 1, 2);
    topk.finish();
    expect_indices(topk, reference_indices(next, ids, 2, 100.0f, false));
  }
}

TYPED_TEST(BatchTopKTypedTest, FullWidthIndicesSurviveResetAndRepeatedFinish) {
  check_full_width_indices_and_reset<typename TestFixture::TopK>();
}

TEST(BatchTopK, RuntimeDispatchSupportsFullWidthIndicesAndReset) {
  check_full_width_indices_and_reset<BatchTopK>();
}

TYPED_TEST(BatchTopKTypedTest, ExplicitIndicesHonorStrictInitialBound) {
  typename TestFixture::TopK topk;
  const std::vector<float> scores{4.0f, 2.0f, 3.0f, 1.0f};
  const std::vector<uint64_t> indices{99, uint64_t{1} << 48, 7, 0};
  for (bool strict : {false, true}) {
    topk.reset(10, 3.0f, strict);
    topk.push_block(scores.data(), indices.data(), 2);
    topk.push_block(scores.data() + 2, indices.data() + 2, 2);
    topk.finish();
    expect_indices(topk, reference_indices(scores, indices, 10, 3.0f, strict));
  }
}

TYPED_TEST(BatchTopKTypedTest, ExplicitInputContinuesAfterFullSelection) {
  typename TestFixture::TopK topk;
  const std::vector<float> scores{1.0f, 5.0f, 8.0f, 8.0f, 7.0f, 0.0f};
  const std::vector<uint64_t> indices{
      uint64_t{1} << 60, 3, 99, 11, 0, std::numeric_limits<uint64_t>::max()};
  topk.reset(3);
  topk.push_block(scores.data(), indices.data(), 3);
  topk.finish();
  ASSERT_EQ(3U, topk.size());
  topk.push_block(scores.data() + 3, indices.data() + 3, 3);
  topk.finish();
  expect_indices(topk, reference_indices(scores, indices, 3, 100.0f, false));
}

}  // namespace
