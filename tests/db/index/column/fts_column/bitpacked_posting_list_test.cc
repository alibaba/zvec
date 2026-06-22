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

#include "db/index/column/fts_column/posting/bitpacked_posting_list.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>
#include <gtest/gtest.h>
#include "db/index/column/fts_column/bm25_scorer.h"
#include "db/index/column/fts_column/posting/bitpacked_simd_scalar.h"
#if defined(__SSE4_1__)
#include "db/index/column/fts_column/posting/bitpacked_simd_sse41.h"
#endif

using namespace zvec::fts;

// ============================================================
// Helper: create a BM25Scorer with reasonable defaults
// ============================================================

static BM25Scorer make_scorer(uint64_t total_docs = 1000,
                              uint64_t total_tokens = 50000) {
  BM25Scorer scorer;
  scorer.update_stats(total_docs, total_tokens);
  return scorer;
}

// ============================================================
// bits_needed()
// ============================================================

TEST(BitPackedPostingListTest, BitsNeededZero) {
  EXPECT_EQ(BitPackedPostingList::bits_needed(0), 0);
}

TEST(BitPackedPostingListTest, BitsNeededOne) {
  EXPECT_EQ(BitPackedPostingList::bits_needed(1), 1);
}

TEST(BitPackedPostingListTest, BitsNeededPowerOfTwo) {
  EXPECT_EQ(BitPackedPostingList::bits_needed(2), 2);
  EXPECT_EQ(BitPackedPostingList::bits_needed(4), 3);
  EXPECT_EQ(BitPackedPostingList::bits_needed(8), 4);
  EXPECT_EQ(BitPackedPostingList::bits_needed(256), 9);
  EXPECT_EQ(BitPackedPostingList::bits_needed(1024), 11);
}

TEST(BitPackedPostingListTest, BitsNeededNonPowerOfTwo) {
  EXPECT_EQ(BitPackedPostingList::bits_needed(3), 2);
  EXPECT_EQ(BitPackedPostingList::bits_needed(5), 3);
  EXPECT_EQ(BitPackedPostingList::bits_needed(7), 3);
  EXPECT_EQ(BitPackedPostingList::bits_needed(255), 8);
  EXPECT_EQ(BitPackedPostingList::bits_needed(1023), 10);
}

TEST(BitPackedPostingListTest, BitsNeededMaxUint32) {
  EXPECT_EQ(BitPackedPostingList::bits_needed(0xFFFFFFFF), 32);
}

// ============================================================
// pack_uint32 / unpack_uint32 round-trip
// ============================================================

class BitPackingTest : public ::testing::TestWithParam<uint8_t> {};

TEST_P(BitPackingTest, PackUnpackRoundTrip128) {
  const uint8_t bitwidth = GetParam();
  if (bitwidth == 0) return;

  const uint32_t count = 128;
  const uint32_t mask =
      (bitwidth == 32) ? 0xFFFFFFFFu : ((1u << bitwidth) - 1u);

  // Generate test values
  std::vector<uint32_t> original(count);
  for (uint32_t i = 0; i < count; ++i) {
    original[i] = (i * 17 + 3) & mask;  // deterministic pattern
  }

  // Pack
  const size_t packed_size =
      BitPackedPostingList::packed_byte_size(bitwidth, count);
  std::vector<uint8_t> packed(packed_size, 0);
  BitPackedPostingList::pack_uint32(original.data(), bitwidth, count,
                                    packed.data());

  // Unpack
  std::vector<uint32_t> decoded(count, 0);
  BitPackedPostingList::unpack_uint32(packed.data(), bitwidth, count,
                                      decoded.data());

  // Verify
  for (uint32_t i = 0; i < count; ++i) {
    EXPECT_EQ(decoded[i], original[i])
        << "Mismatch at index " << i << " with bitwidth " << (int)bitwidth;
  }
}

TEST_P(BitPackingTest, PackUnpackRoundTripSmall) {
  const uint8_t bitwidth = GetParam();
  if (bitwidth == 0) return;

  // Test with a small count (not a full block)
  const uint32_t count = 7;
  const uint32_t mask =
      (bitwidth == 32) ? 0xFFFFFFFFu : ((1u << bitwidth) - 1u);

  std::vector<uint32_t> original(count);
  for (uint32_t i = 0; i < count; ++i) {
    original[i] = i & mask;
  }

  const size_t packed_size =
      BitPackedPostingList::packed_byte_size(bitwidth, count);
  std::vector<uint8_t> packed(packed_size, 0);
  BitPackedPostingList::pack_uint32(original.data(), bitwidth, count,
                                    packed.data());

  std::vector<uint32_t> decoded(count, 0);
  BitPackedPostingList::unpack_uint32(packed.data(), bitwidth, count,
                                      decoded.data());

  for (uint32_t i = 0; i < count; ++i) {
    EXPECT_EQ(decoded[i], original[i])
        << "Mismatch at index " << i << " with bitwidth " << (int)bitwidth;
  }
}

// The scalar full-block packer must round-trip on its own. This exercises the
// portable (non-SIMD) path directly regardless of the host CPU, so the
// architecture used by ARM / no-SSE builds is always covered even when tests
// run on x86.
TEST_P(BitPackingTest, ScalarFullBlockRoundTrip) {
  const uint8_t bitwidth = GetParam();
  if (bitwidth == 0) return;

  const uint32_t count = 128;
  const uint32_t mask =
      (bitwidth == 32) ? 0xFFFFFFFFu : ((1u << bitwidth) - 1u);

  std::vector<uint32_t> values(count);
  for (uint32_t i = 0; i < count; ++i) {
    values[i] = (i * 2654435761u + 7u) & mask;  // pseudo-random, deterministic
  }

  alignas(16) uint8_t packed[32 * 16];
  simd::scalar_pack_uint32_128(values.data(), bitwidth, packed);

  alignas(16) uint32_t decoded[count];
  simd::scalar_unpack_uint32_128(packed, bitwidth, decoded);

  for (uint32_t i = 0; i < count; ++i) {
    EXPECT_EQ(decoded[i], values[i])
        << "Mismatch at index " << i << " with bitwidth "
        << static_cast<int>(bitwidth);
  }
}

#if defined(__SSE4_1__)
// Cross-architecture portability guard: the scalar full-block packer must
// produce byte-identical output to the SSE4.1 SIMD packer, and each must be
// able to decode the other's output. If the scalar fallback ever diverged from
// the SIMD layout again (the original cross-arch corruption bug), an index
// built on x86 would be silently mis-decoded on ARM/no-SSE and vice versa.
TEST_P(BitPackingTest, ScalarLayoutMatchesSimd) {
  const uint8_t bitwidth = GetParam();
  if (bitwidth == 0) return;

  const uint32_t count = 128;
  const uint32_t mask =
      (bitwidth == 32) ? 0xFFFFFFFFu : ((1u << bitwidth) - 1u);

  std::vector<uint32_t> values(count);
  for (uint32_t i = 0; i < count; ++i) {
    values[i] = (i * 40503u + 12345u) & mask;  // deterministic
  }

  const size_t packed_size = static_cast<size_t>(bitwidth) * 16;
  alignas(16) uint8_t scalar_packed[32 * 16];
  alignas(16) uint8_t simd_packed[32 * 16];
  simd::scalar_pack_uint32_128(values.data(), bitwidth, scalar_packed);
  simd::sse41_pack_uint32_128(values.data(), bitwidth, simd_packed);

  EXPECT_EQ(0, std::memcmp(scalar_packed, simd_packed, packed_size))
      << "Scalar and SIMD packed bytes differ for bitwidth "
      << static_cast<int>(bitwidth) << " — on-disk format is not portable";

  // SIMD decodes scalar-packed bytes.
  alignas(16) uint32_t simd_decoded[count];
  simd::sse41_unpack_uint32_128(scalar_packed, bitwidth, simd_decoded);
  // Scalar decodes SIMD-packed bytes.
  alignas(16) uint32_t scalar_decoded[count];
  simd::scalar_unpack_uint32_128(simd_packed, bitwidth, scalar_decoded);

  for (uint32_t i = 0; i < count; ++i) {
    EXPECT_EQ(simd_decoded[i], values[i]) << "SIMD-decode-of-scalar @" << i;
    EXPECT_EQ(scalar_decoded[i], values[i]) << "scalar-decode-of-SIMD @" << i;
  }
}
#endif  // defined(__SSE4_1__)

// Test all bitwidths from 1 to 32
INSTANTIATE_TEST_SUITE_P(AllBitwidths, BitPackingTest,
                         ::testing::Range(static_cast<uint8_t>(1),
                                          static_cast<uint8_t>(33)));

TEST(BitPackingTest, PackUnpackZeroBitwidth) {
  const uint32_t count = 128;
  std::vector<uint32_t> original(count, 0);
  std::vector<uint32_t> decoded(count, 99);

  // bitwidth 0: all values must be 0
  BitPackedPostingList::unpack_uint32(nullptr, 0, count, decoded.data());
  for (uint32_t i = 0; i < count; ++i) {
    EXPECT_EQ(decoded[i], 0u);
  }
}

// ============================================================
// Encode / Decode: empty posting list
// ============================================================

TEST(BitPackedPostingListTest, EncodeDecodeEmpty) {
  BM25Scorer scorer = make_scorer();
  std::string encoded =
      BitPackedPostingList::encode(nullptr, nullptr, nullptr, 0, 0, scorer);

  EXPECT_TRUE(BitPackedPostingList::is_bitpacked_format(encoded.data(),
                                                        encoded.size()));

  BitPackedPostingIterator iter;
  EXPECT_EQ(iter.open(encoded.data(), encoded.size()), 0);
  EXPECT_EQ(iter.cost(), 0u);
  EXPECT_EQ(iter.next_doc(), BitPackedPostingIterator::NO_MORE_DOCS);
}

// ============================================================
// Encode / Decode: single element
// ============================================================

TEST(BitPackedPostingListTest, EncodeDecodeSingleElement) {
  BM25Scorer scorer = make_scorer();
  uint32_t doc_ids[] = {42};
  uint32_t tfs[] = {3};
  uint32_t doc_lens[] = {100};

  std::string encoded =
      BitPackedPostingList::encode(doc_ids, tfs, doc_lens, 1, 1, scorer);

  BitPackedPostingIterator iter;
  EXPECT_EQ(iter.open(encoded.data(), encoded.size()), 0);
  EXPECT_EQ(iter.cost(), 1u);

  EXPECT_EQ(iter.next_doc(), 42u);
  EXPECT_EQ(iter.doc_id(), 42u);
  EXPECT_EQ(iter.term_freq(), 3u);
  EXPECT_EQ(iter.doc_len(), 100u);

  EXPECT_EQ(iter.next_doc(), BitPackedPostingIterator::NO_MORE_DOCS);
}

// ============================================================
// Encode / Decode: small list (< 128)
// ============================================================

TEST(BitPackedPostingListTest, EncodeDecodeSmallList) {
  BM25Scorer scorer = make_scorer();
  const size_t count = 10;
  std::vector<uint32_t> doc_ids(count);
  std::vector<uint32_t> tfs(count);
  std::vector<uint32_t> doc_lens(count);

  for (size_t i = 0; i < count; ++i) {
    doc_ids[i] = static_cast<uint32_t>(i * 5);
    tfs[i] = static_cast<uint32_t>(i + 1);
    doc_lens[i] = static_cast<uint32_t>(50 + i * 10);
  }

  std::string encoded = BitPackedPostingList::encode(
      doc_ids.data(), tfs.data(), doc_lens.data(), count, count, scorer);

  BitPackedPostingIterator iter;
  EXPECT_EQ(iter.open(encoded.data(), encoded.size()), 0);
  EXPECT_EQ(iter.cost(), count);

  for (size_t i = 0; i < count; ++i) {
    uint32_t doc = iter.next_doc();
    EXPECT_EQ(doc, doc_ids[i]) << "Mismatch at index " << i;
    EXPECT_EQ(iter.term_freq(), tfs[i]) << "TF mismatch at index " << i;
    EXPECT_EQ(iter.doc_len(), doc_lens[i]) << "DocLen mismatch at index " << i;
  }

  EXPECT_EQ(iter.next_doc(), BitPackedPostingIterator::NO_MORE_DOCS);
}

// ============================================================
// Encode / Decode: exactly 128 elements (one full block)
// ============================================================

TEST(BitPackedPostingListTest, EncodeDecodeExact128) {
  BM25Scorer scorer = make_scorer();
  const size_t count = 128;
  std::vector<uint32_t> doc_ids(count);
  std::vector<uint32_t> tfs(count);
  std::vector<uint32_t> doc_lens(count);

  for (size_t i = 0; i < count; ++i) {
    doc_ids[i] = static_cast<uint32_t>(i * 3);
    tfs[i] = static_cast<uint32_t>((i % 10) + 1);
    doc_lens[i] = static_cast<uint32_t>(100 + i);
  }

  std::string encoded = BitPackedPostingList::encode(
      doc_ids.data(), tfs.data(), doc_lens.data(), count, count, scorer);

  BitPackedPostingIterator iter;
  EXPECT_EQ(iter.open(encoded.data(), encoded.size()), 0);
  EXPECT_EQ(iter.cost(), count);

  for (size_t i = 0; i < count; ++i) {
    uint32_t doc = iter.next_doc();
    EXPECT_EQ(doc, doc_ids[i]) << "Mismatch at index " << i;
    EXPECT_EQ(iter.term_freq(), tfs[i]) << "TF mismatch at index " << i;
    EXPECT_EQ(iter.doc_len(), doc_lens[i]) << "DocLen mismatch at index " << i;
  }

  EXPECT_EQ(iter.next_doc(), BitPackedPostingIterator::NO_MORE_DOCS);
}

// ============================================================
// Encode / Decode: 129 elements (two blocks, last block has 1 element)
// ============================================================

TEST(BitPackedPostingListTest, EncodeDecodeCrossBlockBoundary) {
  BM25Scorer scorer = make_scorer();
  const size_t count = 129;
  std::vector<uint32_t> doc_ids(count);
  std::vector<uint32_t> tfs(count);
  std::vector<uint32_t> doc_lens(count);

  for (size_t i = 0; i < count; ++i) {
    doc_ids[i] = static_cast<uint32_t>(i * 2);
    tfs[i] = static_cast<uint32_t>((i % 5) + 1);
    doc_lens[i] = static_cast<uint32_t>(200 + i);
  }

  std::string encoded = BitPackedPostingList::encode(
      doc_ids.data(), tfs.data(), doc_lens.data(), count, count, scorer);

  BitPackedPostingIterator iter;
  EXPECT_EQ(iter.open(encoded.data(), encoded.size()), 0);
  EXPECT_EQ(iter.cost(), count);

  for (size_t i = 0; i < count; ++i) {
    uint32_t doc = iter.next_doc();
    EXPECT_EQ(doc, doc_ids[i]) << "Mismatch at index " << i;
    EXPECT_EQ(iter.term_freq(), tfs[i]) << "TF mismatch at index " << i;
    EXPECT_EQ(iter.doc_len(), doc_lens[i]) << "DocLen mismatch at index " << i;
  }

  EXPECT_EQ(iter.next_doc(), BitPackedPostingIterator::NO_MORE_DOCS);
}

// ============================================================
// Encode / Decode: large list (multiple blocks)
// ============================================================

TEST(BitPackedPostingListTest, EncodeDecodeLargeList) {
  BM25Scorer scorer = make_scorer(10000, 500000);
  const size_t count = 1000;
  std::vector<uint32_t> doc_ids(count);
  std::vector<uint32_t> tfs(count);
  std::vector<uint32_t> doc_lens(count);

  for (size_t i = 0; i < count; ++i) {
    doc_ids[i] = static_cast<uint32_t>(i * 10);
    tfs[i] = static_cast<uint32_t>((i % 20) + 1);
    doc_lens[i] = static_cast<uint32_t>(50 + (i % 200));
  }

  std::string encoded = BitPackedPostingList::encode(
      doc_ids.data(), tfs.data(), doc_lens.data(), count, count, scorer);

  BitPackedPostingIterator iter;
  EXPECT_EQ(iter.open(encoded.data(), encoded.size()), 0);
  EXPECT_EQ(iter.cost(), count);

  for (size_t i = 0; i < count; ++i) {
    uint32_t doc = iter.next_doc();
    EXPECT_EQ(doc, doc_ids[i]) << "Mismatch at index " << i;
    EXPECT_EQ(iter.term_freq(), tfs[i]) << "TF mismatch at index " << i;
    EXPECT_EQ(iter.doc_len(), doc_lens[i]) << "DocLen mismatch at index " << i;
  }

  EXPECT_EQ(iter.next_doc(), BitPackedPostingIterator::NO_MORE_DOCS);
}

// ============================================================
// advance(): basic skip-list functionality
// ============================================================

TEST(BitPackedPostingListTest, AdvanceToExactDocId) {
  BM25Scorer scorer = make_scorer();
  const size_t count = 500;
  std::vector<uint32_t> doc_ids(count);
  std::vector<uint32_t> tfs(count, 1);
  std::vector<uint32_t> doc_lens(count, 100);

  for (size_t i = 0; i < count; ++i) {
    doc_ids[i] = static_cast<uint32_t>(i * 3);
  }

  std::string encoded = BitPackedPostingList::encode(
      doc_ids.data(), tfs.data(), doc_lens.data(), count, count, scorer);

  BitPackedPostingIterator iter;
  EXPECT_EQ(iter.open(encoded.data(), encoded.size()), 0);

  // Advance to exact doc_id
  EXPECT_EQ(iter.advance(300), 300u);
  EXPECT_EQ(iter.doc_id(), 300u);

  // Advance to a doc_id that doesn't exist (should return next >= target)
  EXPECT_EQ(iter.advance(301), 303u);
  EXPECT_EQ(iter.doc_id(), 303u);
}

TEST(BitPackedPostingListTest, AdvanceToFirstDoc) {
  BM25Scorer scorer = make_scorer();
  uint32_t doc_ids[] = {10, 20, 30, 40, 50};
  uint32_t tfs[] = {1, 2, 3, 4, 5};
  uint32_t doc_lens[] = {100, 200, 300, 400, 500};

  std::string encoded =
      BitPackedPostingList::encode(doc_ids, tfs, doc_lens, 5, 5, scorer);

  BitPackedPostingIterator iter;
  EXPECT_EQ(iter.open(encoded.data(), encoded.size()), 0);

  // Advance to 0 should return the first doc (10)
  EXPECT_EQ(iter.advance(0), 10u);
  EXPECT_EQ(iter.term_freq(), 1u);
  EXPECT_EQ(iter.doc_len(), 100u);
}

TEST(BitPackedPostingListTest, AdvanceBeyondLastDoc) {
  BM25Scorer scorer = make_scorer();
  uint32_t doc_ids[] = {10, 20, 30};
  uint32_t tfs[] = {1, 2, 3};
  uint32_t doc_lens[] = {100, 200, 300};

  std::string encoded =
      BitPackedPostingList::encode(doc_ids, tfs, doc_lens, 3, 3, scorer);

  BitPackedPostingIterator iter;
  EXPECT_EQ(iter.open(encoded.data(), encoded.size()), 0);

  EXPECT_EQ(iter.advance(31), BitPackedPostingIterator::NO_MORE_DOCS);
}

TEST(BitPackedPostingListTest, AdvanceAcrossBlocks) {
  BM25Scorer scorer = make_scorer();
  const size_t count = 300;
  std::vector<uint32_t> doc_ids(count);
  std::vector<uint32_t> tfs(count, 2);
  std::vector<uint32_t> doc_lens(count, 50);

  for (size_t i = 0; i < count; ++i) {
    doc_ids[i] = static_cast<uint32_t>(i * 5);
  }

  std::string encoded = BitPackedPostingList::encode(
      doc_ids.data(), tfs.data(), doc_lens.data(), count, count, scorer);

  BitPackedPostingIterator iter;
  EXPECT_EQ(iter.open(encoded.data(), encoded.size()), 0);

  // Advance from start to a doc in the 3rd block (block 2, index 256+)
  // Block 0: doc_ids 0..635 (indices 0..127)
  // Block 1: doc_ids 640..1275 (indices 128..255)
  // Block 2: doc_ids 1280..1495 (indices 256..299)
  EXPECT_EQ(iter.advance(1280), 1280u);
  EXPECT_EQ(iter.doc_id(), 1280u);
  EXPECT_EQ(iter.term_freq(), 2u);

  // Continue with next_doc
  EXPECT_EQ(iter.next_doc(), 1285u);
}

TEST(BitPackedPostingListTest, AdvanceSequentialCalls) {
  BM25Scorer scorer = make_scorer();
  const size_t count = 200;
  std::vector<uint32_t> doc_ids(count);
  std::vector<uint32_t> tfs(count, 1);
  std::vector<uint32_t> doc_lens(count, 100);

  for (size_t i = 0; i < count; ++i) {
    doc_ids[i] = static_cast<uint32_t>(i * 7);
  }

  std::string encoded = BitPackedPostingList::encode(
      doc_ids.data(), tfs.data(), doc_lens.data(), count, count, scorer);

  BitPackedPostingIterator iter;
  EXPECT_EQ(iter.open(encoded.data(), encoded.size()), 0);

  // Multiple sequential advance calls
  EXPECT_EQ(iter.advance(100), 105u);    // 15*7=105
  EXPECT_EQ(iter.advance(500), 504u);    // 72*7=504
  EXPECT_EQ(iter.advance(1000), 1001u);  // 143*7=1001
  EXPECT_EQ(iter.advance(1400), BitPackedPostingIterator::NO_MORE_DOCS);
}

// ============================================================
// advance() after next_doc()
// ============================================================

TEST(BitPackedPostingListTest, AdvanceAfterNextDoc) {
  BM25Scorer scorer = make_scorer();
  const size_t count = 256;
  std::vector<uint32_t> doc_ids(count);
  std::vector<uint32_t> tfs(count, 1);
  std::vector<uint32_t> doc_lens(count, 50);

  for (size_t i = 0; i < count; ++i) {
    doc_ids[i] = static_cast<uint32_t>(i * 4);
  }

  std::string encoded = BitPackedPostingList::encode(
      doc_ids.data(), tfs.data(), doc_lens.data(), count, count, scorer);

  BitPackedPostingIterator iter;
  EXPECT_EQ(iter.open(encoded.data(), encoded.size()), 0);

  // Read a few docs
  EXPECT_EQ(iter.next_doc(), 0u);
  EXPECT_EQ(iter.next_doc(), 4u);
  EXPECT_EQ(iter.next_doc(), 8u);

  // Now advance past the current block
  EXPECT_EQ(iter.advance(600), 600u);  // 150*4=600
  EXPECT_EQ(iter.term_freq(), 1u);

  // Continue with next_doc
  EXPECT_EQ(iter.next_doc(), 604u);
}

// ============================================================
// block_max_score correctness
// ============================================================

TEST(BitPackedPostingListTest, BlockMaxScoreCorrectness) {
  BM25Scorer scorer = make_scorer(100, 5000);
  const size_t count = 256;  // 2 blocks
  std::vector<uint32_t> doc_ids(count);
  std::vector<uint32_t> tfs(count);
  std::vector<uint32_t> doc_lens(count);

  for (size_t i = 0; i < count; ++i) {
    doc_ids[i] = static_cast<uint32_t>(i);
    tfs[i] = static_cast<uint32_t>((i % 10) + 1);
    doc_lens[i] = static_cast<uint32_t>(50 + (i % 50));
  }

  std::string encoded = BitPackedPostingList::encode(
      doc_ids.data(), tfs.data(), doc_lens.data(), count, count, scorer);

  BitPackedPostingIterator iter;
  EXPECT_EQ(iter.open(encoded.data(), encoded.size()), 0);

  // Verify block_max_score for block 0 via block_max_info_for()
  auto info0 = iter.block_max_info_for(0);

  // Manually compute max score for block 0
  float expected_max = 0.0f;
  for (size_t i = 0; i < 128; ++i) {
    float s = scorer.score(count, tfs[i], doc_lens[i]);
    expected_max = std::max(expected_max, s);
  }
  EXPECT_FLOAT_EQ(info0.block_max_score, expected_max);
  EXPECT_EQ(info0.block_last_doc, 127u);

  // Verify block_max_score for block 1 via block_max_info_for()
  auto info1 = iter.block_max_info_for(128);

  expected_max = 0.0f;
  for (size_t i = 128; i < 256; ++i) {
    float s = scorer.score(count, tfs[i], doc_lens[i]);
    expected_max = std::max(expected_max, s);
  }
  EXPECT_FLOAT_EQ(info1.block_max_score, expected_max);
  EXPECT_EQ(info1.block_last_doc, 255u);
}

// ============================================================
// max_score() (global)
// ============================================================

TEST(BitPackedPostingListTest, GlobalMaxScore) {
  BM25Scorer scorer = make_scorer(100, 5000);
  const size_t count = 256;
  std::vector<uint32_t> doc_ids(count);
  std::vector<uint32_t> tfs(count);
  std::vector<uint32_t> doc_lens(count);

  for (size_t i = 0; i < count; ++i) {
    doc_ids[i] = static_cast<uint32_t>(i);
    tfs[i] = static_cast<uint32_t>((i % 10) + 1);
    doc_lens[i] = static_cast<uint32_t>(50 + (i % 50));
  }

  std::string encoded = BitPackedPostingList::encode(
      doc_ids.data(), tfs.data(), doc_lens.data(), count, count, scorer);

  BitPackedPostingIterator iter;
  EXPECT_EQ(iter.open(encoded.data(), encoded.size()), 0);

  // Global max_score should be the maximum of all block_max_scores
  float global_max = 0.0f;
  for (size_t i = 0; i < count; ++i) {
    float s = scorer.score(count, tfs[i], doc_lens[i]);
    global_max = std::max(global_max, s);
  }
  EXPECT_FLOAT_EQ(iter.max_score(), global_max);
}

// ============================================================
// is_bitpacked_format()
// ============================================================

TEST(BitPackedPostingListTest, IsBitpackedFormatTrue) {
  BM25Scorer scorer = make_scorer();
  uint32_t doc_ids[] = {1};
  uint32_t tfs[] = {1};
  uint32_t doc_lens[] = {10};

  std::string encoded =
      BitPackedPostingList::encode(doc_ids, tfs, doc_lens, 1, 1, scorer);
  EXPECT_TRUE(BitPackedPostingList::is_bitpacked_format(encoded.data(),
                                                        encoded.size()));
}

TEST(BitPackedPostingListTest, IsBitpackedFormatFalse) {
  // Random data that doesn't start with the magic number
  std::string random_data = "hello world";
  EXPECT_FALSE(BitPackedPostingList::is_bitpacked_format(random_data.data(),
                                                         random_data.size()));
}

TEST(BitPackedPostingListTest, IsBitpackedFormatTooShort) {
  std::string short_data = "ab";
  EXPECT_FALSE(BitPackedPostingList::is_bitpacked_format(short_data.data(),
                                                         short_data.size()));
}

// ============================================================
// Error handling: open() with invalid data
// ============================================================

TEST(BitPackedPostingListTest, OpenWithNullData) {
  BitPackedPostingIterator iter;
  EXPECT_NE(iter.open(nullptr, 0), 0);
}

TEST(BitPackedPostingListTest, OpenWithTruncatedHeader) {
  BitPackedPostingIterator iter;
  char data[4] = {0};
  EXPECT_NE(iter.open(data, 4), 0);
}

TEST(BitPackedPostingListTest, OpenWithBadMagic) {
  BitPackedPostingIterator iter;
  char data[16] = {0};
  EXPECT_NE(iter.open(data, 16), 0);
}

// ============================================================
// Cross-path encode/decode: verify that posting lists encoded via the
// dispatch path (SIMD on x86) are correctly decodable by the scalar
// fallback.  This guards against the full encode() pipeline drifting
// from the scalar decoder — the low-level ScalarLayoutMatchesSimd test
// only covers the pack/unpack primitives, not the complete block
// layout produced by encode().
// ============================================================

TEST(BitPackedPostingListTest, EncodeDecodeScalarCrossDecode) {
  BM25Scorer scorer = make_scorer(1000, 50000);
  // 300 docs → 2 full blocks (128 each) + 1 tail block (44)
  const size_t count = 300;
  std::vector<uint32_t> doc_ids(count);
  std::vector<uint32_t> tfs(count);
  std::vector<uint32_t> doc_lens(count);

  for (size_t i = 0; i < count; ++i) {
    doc_ids[i] = static_cast<uint32_t>(i * 7 + 3);
    tfs[i] = static_cast<uint32_t>((i % 15) + 1);
    doc_lens[i] = static_cast<uint32_t>(30 + (i % 100));
  }

  std::string encoded = BitPackedPostingList::encode(
      doc_ids.data(), tfs.data(), doc_lens.data(), count, count, scorer);

  // Parse the header
  BitPackedPostingList::Header hdr{};
  std::memcpy(&hdr, encoded.data(), sizeof(hdr));
  ASSERT_EQ(hdr.magic, BitPackedPostingList::MAGIC);
  ASSERT_EQ(hdr.num_docs, count);

  const auto *skip = reinterpret_cast<const BitPackedPostingList::BlockMeta *>(
      encoded.data() + BitPackedPostingList::HEADER_SIZE);

  // Manually decode each block using the scalar path and verify
  std::vector<uint32_t> decoded_doc_ids;
  for (uint32_t b = 0; b < hdr.num_blocks; ++b) {
    const char *block_ptr = encoded.data() + skip[b].block_offset;
    BitPackedPostingList::BlockHeader bhdr{};
    std::memcpy(&bhdr, block_ptr, sizeof(bhdr));

    const uint8_t *packed =
        reinterpret_cast<const uint8_t *>(block_ptr + sizeof(bhdr));
    const bool is_full =
        (bhdr.num_docs == BitPackedPostingList::DOCS_PER_BLOCK);

    // Decode doc_id deltas using explicit scalar path
    alignas(16) uint32_t deltas[BitPackedPostingList::DOCS_PER_BLOCK]{};
    if (is_full) {
      simd::scalar_unpack_uint32_128(packed, bhdr.bitwidth_id, deltas);
    } else {
      BitPackedPostingList::unpack_uint32(packed, bhdr.bitwidth_id,
                                          bhdr.num_docs, deltas);
    }

    // Reconstruct absolute doc_ids via prefix-sum
    uint32_t prev = bhdr.min_doc_id;
    decoded_doc_ids.push_back(prev);
    for (uint32_t i = 1; i < bhdr.num_docs; ++i) {
      prev += deltas[i];
      decoded_doc_ids.push_back(prev);
    }

    const size_t id_bytes =
        is_full ? BitPackedPostingList::simd_packed_byte_size(bhdr.bitwidth_id)
                : BitPackedPostingList::packed_byte_size(bhdr.bitwidth_id,
                                                         bhdr.num_docs);
    const uint8_t *tf_packed = packed + id_bytes;
    alignas(16) uint32_t decoded_tfs[BitPackedPostingList::DOCS_PER_BLOCK]{};
    if (is_full) {
      simd::scalar_unpack_uint32_128(tf_packed, bhdr.bitwidth_tf, decoded_tfs);
    } else {
      BitPackedPostingList::unpack_uint32(tf_packed, bhdr.bitwidth_tf,
                                          bhdr.num_docs, decoded_tfs);
    }

    const size_t tf_bytes =
        is_full ? BitPackedPostingList::simd_packed_byte_size(bhdr.bitwidth_tf)
                : BitPackedPostingList::packed_byte_size(bhdr.bitwidth_tf,
                                                         bhdr.num_docs);
    const uint8_t *dl_packed = tf_packed + tf_bytes;
    alignas(16) uint32_t decoded_dls[BitPackedPostingList::DOCS_PER_BLOCK]{};
    if (is_full) {
      simd::scalar_unpack_uint32_128(dl_packed, bhdr.bitwidth_dl, decoded_dls);
    } else {
      BitPackedPostingList::unpack_uint32(dl_packed, bhdr.bitwidth_dl,
                                          bhdr.num_docs, decoded_dls);
    }

    const size_t start =
        static_cast<size_t>(b) * BitPackedPostingList::DOCS_PER_BLOCK;
    for (uint32_t i = 0; i < bhdr.num_docs; ++i) {
      EXPECT_EQ(decoded_tfs[i], tfs[start + i])
          << "TF mismatch block " << b << " index " << i;
      EXPECT_EQ(decoded_dls[i], doc_lens[start + i])
          << "DocLen mismatch block " << b << " index " << i;
    }
  }

  ASSERT_EQ(decoded_doc_ids.size(), count);
  for (size_t i = 0; i < count; ++i) {
    EXPECT_EQ(decoded_doc_ids[i], doc_ids[i])
        << "DocId mismatch at index " << i;
  }
}

// ============================================================
// Consistency: advance() vs sequential next_doc()
// ============================================================

TEST(BitPackedPostingListTest, AdvanceConsistentWithNextDoc) {
  BM25Scorer scorer = make_scorer();
  const size_t count = 500;
  std::vector<uint32_t> doc_ids(count);
  std::vector<uint32_t> tfs(count);
  std::vector<uint32_t> doc_lens(count);

  std::mt19937 rng(42);
  uint32_t current = 0;
  for (size_t i = 0; i < count; ++i) {
    current += (rng() % 10) + 1;
    doc_ids[i] = current;
    tfs[i] = (rng() % 10) + 1;
    doc_lens[i] = (rng() % 200) + 10;
  }

  std::string encoded = BitPackedPostingList::encode(
      doc_ids.data(), tfs.data(), doc_lens.data(), count, count, scorer);

  // Collect all docs via next_doc
  BitPackedPostingIterator iter1;
  EXPECT_EQ(iter1.open(encoded.data(), encoded.size()), 0);
  std::vector<uint32_t> all_docs;
  std::vector<uint32_t> all_tfs;
  std::vector<uint32_t> all_doc_lens;
  uint32_t doc = iter1.next_doc();
  while (doc != BitPackedPostingIterator::NO_MORE_DOCS) {
    all_docs.push_back(doc);
    all_tfs.push_back(iter1.term_freq());
    all_doc_lens.push_back(iter1.doc_len());
    doc = iter1.next_doc();
  }

  ASSERT_EQ(all_docs.size(), count);

  // Verify advance to various targets matches sequential scan
  BitPackedPostingIterator iter2;
  EXPECT_EQ(iter2.open(encoded.data(), encoded.size()), 0);

  std::vector<uint32_t> targets = {0,
                                   1,
                                   doc_ids[50],
                                   doc_ids[127],
                                   doc_ids[128],
                                   doc_ids[200],
                                   doc_ids[count - 1]};

  for (uint32_t target : targets) {
    BitPackedPostingIterator iter_adv;
    EXPECT_EQ(iter_adv.open(encoded.data(), encoded.size()), 0);
    uint32_t adv_doc = iter_adv.advance(target);

    // Find expected result via linear scan
    auto it = std::lower_bound(all_docs.begin(), all_docs.end(), target);
    if (it == all_docs.end()) {
      EXPECT_EQ(adv_doc, BitPackedPostingIterator::NO_MORE_DOCS)
          << "target=" << target;
    } else {
      size_t idx = it - all_docs.begin();
      EXPECT_EQ(adv_doc, all_docs[idx]) << "target=" << target;
      EXPECT_EQ(iter_adv.term_freq(), all_tfs[idx]) << "target=" << target;
      EXPECT_EQ(iter_adv.doc_len(), all_doc_lens[idx]) << "target=" << target;
    }
  }
}
