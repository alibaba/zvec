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
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include "utility/batch_topk_internal.h"

namespace zvec {
namespace core {

// Production consumers cannot select a layout or bypass runtime ISA checks.
class BatchTopKTestAccess {
 public:
  static void Reset(BatchTopK &topk, size_t k, float threshold, bool strict,
                    batch_topk_internal::Isa isa, bool small) {
    topk.reset_impl(k, threshold, strict, isa, small);
  }

  static size_t Capacity(const BatchTopK &topk) {
    return topk.capacity_;
  }
};

}  // namespace core
}  // namespace zvec

namespace {

using zvec::core::BatchTopK;
using zvec::core::BatchTopKTestAccess;
namespace internal = zvec::core::batch_topk_internal;

constexpr float kDefaultBound = std::numeric_limits<float>::max();
constexpr internal::Isa kImplementations[] = {
    internal::Isa::kScalar, internal::Isa::kSSE2, internal::Isa::kAVX2,
    internal::Isa::kAVX512, internal::Isa::kNEON};

static_assert(
    std::is_same<decltype(std::declval<const BatchTopK &>().result(size_t{})),
                 BatchTopK::Candidate>::value,
    "Results are values, not views into an implementation-specific layout");

uint32_t Bits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

float FromBits(uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

struct Input {
  explicit Input(std::vector<float> values) : scores(std::move(values)) {
    const uint64_t special[] = {0,
                                UINT64_MAX,
                                UINT64_C(0x8000000000000000),
                                UINT64_C(0x100000000),
                                UINT32_MAX,
                                1,
                                UINT64_C(0x8000000000000001),
                                UINT64_C(0x7fffffffffffffff)};
    ids.reserve(scores.size());
    for (size_t i = 0; i < scores.size(); ++i) {
      const uint64_t id =
          i < sizeof(special) / sizeof(*special)
              ? special[i]
              : ((UINT64_C(0x80000000) + i) << 32) |
                    (UINT32_C(0x89abcdef) ^ static_cast<uint32_t>(i));
      ids.push_back(id);
      positions.emplace(id, i);
    }
  }

  std::vector<float> scores;
  std::vector<uint64_t> ids;
  std::unordered_map<uint64_t, size_t> positions;
};

void Check(const BatchTopK &topk, const Input &input, size_t seen, size_t k,
           float bound = kDefaultBound, bool strict = false) {
  const auto accepts = [bound, strict](float value) {
    return strict ? value < bound : value <= bound;
  };
  std::vector<float> expected;
  for (size_t i = 0; i < seen; ++i) {
    if (accepts(input.scores[i])) expected.push_back(input.scores[i]);
  }
  // Independent oracle: fully sort input scores, rather than sharing the
  // selector's partition, threshold-update or internal representation code.
  std::sort(expected.begin(), expected.end());
  if (expected.size() > k) expected.resize(k);
  ASSERT_EQ(expected.size(), topk.size());

  std::vector<float> actual;
  std::unordered_set<uint64_t> ids;
  for (size_t i = 0; i < topk.size(); ++i) {
    const auto value = topk.result(i);
    const auto found = input.positions.find(value.id);
    ASSERT_NE(input.positions.end(), found) << "ID " << value.id;
    ASSERT_LT(found->second, seen);
    ASSERT_TRUE(ids.insert(value.id).second) << "Repeated ID " << value.id;
    EXPECT_EQ(Bits(input.scores[found->second]), Bits(value.score))
        << "Score/ID association, including the sign of zero, must survive";
    ASSERT_TRUE(accepts(value.score));
    actual.push_back(value.score);
  }
  std::sort(actual.begin(), actual.end());
  // Equal-score IDs and output order are unspecified; signed zeros are ties.
  EXPECT_EQ(expected, actual);
}

void FeedAndCheck(BatchTopK &topk, const Input &input, size_t k,
                  const std::vector<size_t> &widths,
                  float bound = kDefaultBound, bool strict = false,
                  size_t finish_every = 0) {
  ASSERT_FALSE(widths.empty());
  for (size_t width : widths) ASSERT_NE(0u, width);
  topk.push_block(nullptr, nullptr, 0);
  size_t offset = 0;
  size_t batch = 0;
  while (offset < input.scores.size()) {
    const size_t count =
        std::min(widths[batch % widths.size()], input.scores.size() - offset);
    std::vector<float> scores(input.scores.begin() + offset,
                              input.scores.begin() + offset + count);
    std::vector<uint64_t> ids(input.ids.begin() + offset,
                              input.ids.begin() + offset + count);
    topk.push_block(scores.data(), ids.data(), count);
    // The caller may overwrite/release both arrays immediately after a push.
    std::fill(scores.begin(), scores.end(),
              std::numeric_limits<float>::quiet_NaN());
    std::fill(ids.begin(), ids.end(), UINT64_C(0x5555555555555555));
    offset += count;
    ++batch;
    if (finish_every && batch % finish_every == 0) {
      topk.finish();
      ASSERT_NO_FATAL_FAILURE(Check(topk, input, offset, k, bound, strict));
    }
  }
  topk.finish();
  ASSERT_NO_FATAL_FAILURE(Check(topk, input, offset, k, bound, strict));
  topk.finish();
  topk.push_block(nullptr, nullptr, 0);
  topk.finish();
  ASSERT_NO_FATAL_FAILURE(Check(topk, input, offset, k, bound, strict));
}

std::vector<float> Scores(size_t count, unsigned pattern, uint64_t seed) {
  std::vector<float> scores(count);
  std::mt19937_64 random(seed);
  for (size_t i = 0; i < count; ++i) {
    switch (pattern) {
      case 0:
        scores[i] = static_cast<float>(i) - static_cast<float>(count / 2);
        break;
      case 1:
        scores[i] =
            static_cast<float>(count - i) - static_cast<float>(count / 2);
        break;
      case 2:
        scores[i] = 0.75f;
        break;
      case 3:
        scores[i] = static_cast<float>(static_cast<int>(random() % 17) - 8);
        break;
      default:
        scores[i] =
            static_cast<float>(static_cast<int>(random() % 2000001) - 1000000) /
            1024.0f;
        break;
    }
  }
  return scores;
}

TEST(BatchTopK, MatchesIndependentOracleAcrossLimitsAndTails) {
  BatchTopK topk;
  const size_t counts[] = {0,  1,  3,   31,  32,  33,   63,
                           64, 65, 127, 129, 257, 1027, 4097};
  for (size_t k : {0u, 1u, 10u, 32u, 64u, 65u, 100u, 1000u}) {
    SCOPED_TRACE(k);
    for (size_t count : counts) {
      SCOPED_TRACE(count);
      for (unsigned pattern = 0; pattern < 5; ++pattern) {
        SCOPED_TRACE(pattern);
        const Input input(Scores(count, pattern, UINT64_C(0x11ac31) + count));
        topk.reset(k);
        ASSERT_NO_FATAL_FAILURE(
            FeedAndCheck(topk, input, k, {1, 3, 4, 7, 16, 31, 33, 127, 512}));
      }
    }
  }
}

TEST(BatchTopK, PreservesFloatingPointAndInitialThresholdSemantics) {
  const float infinity = std::numeric_limits<float>::infinity();
  const float tiny = std::numeric_limits<float>::denorm_min();
  const float values[] = {-infinity,
                          -kDefaultBound,
                          -1.0f,
                          -tiny,
                          -0.0f,
                          0.0f,
                          tiny,
                          1.0f,
                          std::nextafter(1.0f, 0.0f),
                          std::nextafter(1.0f, infinity),
                          kDefaultBound,
                          infinity,
                          FromBits(0x7fc01234),
                          FromBits(0xffc05678)};
  std::vector<float> scores;
  for (size_t repeat = 0; repeat < 19; ++repeat) {
    scores.insert(scores.end(), std::begin(values), std::end(values));
  }
  const Input input(std::move(scores));
  BatchTopK topk;
  for (float bound : {kDefaultBound, 0.0f, -0.0f, 1.0f, infinity, -infinity}) {
    SCOPED_TRACE(Bits(bound));
    for (bool strict : {false, true}) {
      SCOPED_TRACE(strict);
      for (size_t k : {0u, 1u, 10u, 64u, 65u, 1000u}) {
        SCOPED_TRACE(k);
        topk.reset(k, bound, strict);
        ASSERT_NO_FATAL_FAILURE(FeedAndCheck(
            topk, input, k, {3, 4, 15, 16, 17, 33}, bound, strict));
      }
    }
  }
}

TEST(BatchTopK, ReusesCapacityAndCanAppendAfterFinish) {
  BatchTopK topk;
  const Input input(Scores(4097, 4, UINT64_C(0xfe3100)));
  for (size_t k : {1000u, 64u, 65u, 1u, 0u, 100u, 32u, 10u}) {
    SCOPED_TRACE(k);
    topk.reset(k, 0.0f, true);
    topk.finish();
    ASSERT_EQ(0u, topk.size());
    ASSERT_NO_FATAL_FAILURE(
        FeedAndCheck(topk, input, k, {3, 17, 512, 1, 33}, 0.0f, true, 3));
  }
  // An unfinished old selection must not leak into a new query or policy.
  for (const auto &limits :
       {std::pair<size_t, size_t>{64, 65}, {65, 64}, {1000, 1}, {1, 1000}}) {
    SCOPED_TRACE(limits.first);
    SCOPED_TRACE(limits.second);
    topk.reset(limits.first);
    topk.push_block(input.scores.data(), input.ids.data(), 103);
    topk.reset(limits.second, -1.0f, true);
    ASSERT_NO_FATAL_FAILURE(
        FeedAndCheck(topk, input, limits.second, {17, 65}, -1.0f, true));
  }
}

TEST(BatchTopK, ResultIsAnOwnedValueIndependentOfLaterMutation) {
  BatchTopK topk;
  const Input input({5.0f, -0.0f, -3.0f});
  topk.reset(1);
  ASSERT_NO_FATAL_FAILURE(FeedAndCheck(topk, input, 1, {3}));
  const auto saved = topk.result(0);
  topk.reset(0);
  topk.push_block(input.scores.data(), input.ids.data(), input.scores.size());
  topk.finish();
  EXPECT_EQ(0u, topk.size());
  EXPECT_EQ(input.ids[2], saved.id);
  EXPECT_EQ(Bits(input.scores[2]), Bits(saved.score));
}

TEST(BatchTopK, OverflowResetPreservesExistingResultsAndContinuation) {
  BatchTopK topk;
  const Input input(Scores(257, 4, UINT64_C(0x7190ad)));
  for (size_t k : {10u, 65u}) {
    SCOPED_TRACE(k);
    topk.reset(k);
    topk.push_block(input.scores.data(), input.ids.data(), 113);
    topk.finish();
    ASSERT_NO_FATAL_FAILURE(Check(topk, input, 113, k));

    // Exercise both capacity arithmetic and allocation-byte overflow guards;
    // neither call should attempt a huge allocation or discard prior results.
    EXPECT_THROW(topk.reset(std::numeric_limits<size_t>::max()),
                 std::length_error);
    ASSERT_NO_FATAL_FAILURE(Check(topk, input, 113, k));
    EXPECT_THROW(
        topk.reset(std::numeric_limits<size_t>::max() / sizeof(uint64_t)),
        std::length_error);
    ASSERT_NO_FATAL_FAILURE(Check(topk, input, 113, k));

    topk.push_block(input.scores.data() + 113, input.ids.data() + 113,
                    input.scores.size() - 113);
    topk.finish();
    ASSERT_NO_FATAL_FAILURE(Check(topk, input, input.scores.size(), k));
  }
}

TEST(BatchTopK, DuplicateIdsRemainIndependentCandidateOccurrences) {
  std::vector<float> scores;
  std::vector<uint64_t> ids;
  const uint64_t tokens[] = {0, UINT64_MAX, UINT64_C(0x8000000000000001), 42};
  for (size_t i = 0; i < 257; ++i) {
    scores.push_back(static_cast<float>(i % 11) - 5.0f);
    ids.push_back(tokens[i % 4]);
  }
  BatchTopK topk;
  for (size_t k : {10u, 65u, 1000u}) {
    SCOPED_TRACE(k);
    topk.reset(k);
    for (size_t i = 0; i < scores.size(); i += 17) {
      topk.push_block(scores.data() + i, ids.data() + i,
                      std::min(size_t{17}, scores.size() - i));
    }
    topk.finish();

    auto expected = scores;
    std::sort(expected.begin(), expected.end());
    expected.resize(std::min(k, expected.size()));
    ASSERT_EQ(expected.size(), topk.size());
    std::map<std::pair<uint32_t, uint64_t>, size_t> available;
    for (size_t i = 0; i < scores.size(); ++i) {
      ++available[{Bits(scores[i]), ids[i]}];
    }
    std::vector<float> actual;
    for (size_t i = 0; i < topk.size(); ++i) {
      const auto value = topk.result(i);
      auto found = available.find({Bits(value.score), value.id});
      ASSERT_NE(available.end(), found);
      ASSERT_GT(found->second, 0u)
          << "A candidate occurrence cannot be returned more often than read";
      --found->second;
      actual.push_back(value.score);
    }
    std::sort(actual.begin(), actual.end());
    EXPECT_EQ(expected, actual);
  }
}

TEST(BatchTopK, AvailableIsaAndBothPoliciesMatchTheSameOracle) {
  EXPECT_TRUE(internal::Supported(internal::Isa::kScalar));
  EXPECT_TRUE(internal::Supported(internal::RuntimeIsa()));
  BatchTopK topk;
  for (auto isa : kImplementations) {
    if (!internal::Supported(isa)) continue;
    SCOPED_TRACE(static_cast<int>(isa));
    for (bool small : {false, true}) {
      SCOPED_TRACE(small);
      for (size_t k : {0u, 1u, 10u, 32u, 64u, 65u, 100u, 1000u}) {
        SCOPED_TRACE(k);
        for (unsigned pattern : {1u, 2u, 4u}) {
          SCOPED_TRACE(pattern);
          const Input input(
              Scores(std::max(size_t{257}, 4 * k + 17), pattern, k + 719));
          BatchTopKTestAccess::Reset(topk, k, kDefaultBound, false, isa, small);
          ASSERT_NO_FATAL_FAILURE(
              FeedAndCheck(topk, input, k, {3, 4, 7, 31, 32, 33, 127, 512}));
        }
      }
      std::vector<float> specials;
      for (size_t i = 0; i < 41; ++i) {
        specials.insert(
            specials.end(),
            {-std::numeric_limits<float>::infinity(), -0.0f, 0.0f, 1.0f,
             std::numeric_limits<float>::infinity(), FromBits(0x7fc01234)});
      }
      const Input special_input(std::move(specials));
      for (size_t k : {65u, 1000u}) {
        const float bound = std::numeric_limits<float>::infinity();
        BatchTopKTestAccess::Reset(topk, k, bound, false, isa, small);
        ASSERT_NO_FATAL_FAILURE(
            FeedAndCheck(topk, special_input, k, {16, 31, 33}, bound));
      }
    }
  }
}

TEST(BatchTopK, CollectionInsideSimdGroupRefreshesItsBoundSafely) {
  BatchTopK topk;
  for (auto isa : kImplementations) {
    if (!internal::Supported(isa)) continue;
    SCOPED_TRACE(static_cast<int>(isa));
    for (bool small : {false, true}) {
      SCOPED_TRACE(small);
      for (size_t k : {1u, 10u, 64u, 65u, 1000u}) {
        SCOPED_TRACE(k);
        BatchTopKTestAccess::Reset(topk, k, kDefaultBound, false, isa, small);
        const size_t capacity = BatchTopKTestAccess::Capacity(topk);
        ASSERT_GT(capacity, 1u);
        auto scores = Scores(capacity - 1, 0, 0);
        for (auto &score : scores) score += 10000.0f;
        // The first lane fills the buffer. Subsequent lanes alternate better
        // and now-out-of-bound values, including positions in 8/16-lane loops.
        const float suffix[] = {
            -1, 1000000, -2, 1000000, -3, 1000000, -4, 1000000, -5, 1000000,
            -6, 1000000, -7, 1000000, -8, 1000000, -9, -10,     -11};
        scores.insert(scores.end(), std::begin(suffix), std::end(suffix));
        const Input input(std::move(scores));
        ASSERT_NO_FATAL_FAILURE(
            FeedAndCheck(topk, input, k, {capacity - 1, 16, 3}));
      }
    }
  }
}

}  // namespace
