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

#include <cstring>
#include <memory>
#include <vector>
#include <ailego/pattern/defer.h>
#include <gtest/gtest.h>
#include "db/index/column/inverted_column/inverted_codec.h"

namespace zvec {
namespace {

using Bitmap =
    std::unique_ptr<roaring_bitmap_t, decltype(&roaring_bitmap_free)>;

Bitmap MakeBitmap() {
  return Bitmap(roaring_bitmap_create(), roaring_bitmap_free);
}

TEST(InvertedCodecMergeTest, ShortListsAllowUnalignedInputAndRepeatedMerges) {
  for (uint32_t count : {1u, uint32_t{INVERT_ID_LIST_SIZE_THRESHOLD}}) {
    auto posting = MakeBitmap();
    for (uint32_t i = 0; i < count; ++i)
      roaring_bitmap_add(posting.get(), i * 65537);
    std::string encoded;
    ASSERT_TRUE(InvertedIndexCodec::Serialize(posting.get(), &encoded).ok());
    ASSERT_EQ(static_cast<unsigned char>(encoded[0]), count);
    for (size_t shift = 0; shift < 8; ++shift) {
      std::vector<uint64_t> storage(encoded.size() / sizeof(uint64_t) + 2);
      char *data = reinterpret_cast<char *>(storage.data()) + shift;
      std::memcpy(data, encoded.data(), encoded.size());
      auto result = MakeBitmap();
      roaring_bitmap_add(result.get(), UINT32_MAX);
      for (int repeat = 0; repeat < 2; ++repeat)
        ASSERT_TRUE(InvertedIndexCodec::merge_or(data, encoded.size(), true,
                                                 result.get())
                        .ok());
      EXPECT_EQ(roaring_bitmap_get_cardinality(result.get()), count + 1);
      EXPECT_TRUE(roaring_bitmap_contains(result.get(), UINT32_MAX));
      for (uint32_t i = 0; i < count; ++i)
        EXPECT_TRUE(roaring_bitmap_contains(result.get(), i * 65537));
      ASSERT_TRUE(
          InvertedIndexCodec::merge_and(data, encoded.size(), result.get())
              .ok());
      EXPECT_EQ(roaring_bitmap_get_cardinality(result.get()), count);
      EXPECT_FALSE(roaring_bitmap_contains(result.get(), UINT32_MAX));
      roaring_bitmap_add(result.get(), UINT32_MAX);
      for (int repeat = 0; repeat < 2; ++repeat)
        ASSERT_TRUE(
            InvertedIndexCodec::merge_andnot(data, encoded.size(), result.get())
                .ok());
      EXPECT_EQ(roaring_bitmap_get_cardinality(result.get()), 1);
      EXPECT_TRUE(roaring_bitmap_contains(result.get(), UINT32_MAX));
    }
  }
}

TEST(InvertedCodecMergeTest, BitmapLayoutsAllowEveryAlignmentAndOutliveInput) {
  for (int shape = 0; shape < 6; ++shape) {
    auto posting = MakeBitmap();
    if (shape == 0) {
      // A 10M-document range bucket spans 153 high-key containers.
      for (uint32_t container = 0; container < 153; ++container)
        for (uint32_t i = 0; i < 96; ++i)
          roaring_bitmap_add(posting.get(), container * 65536 + i * 37);
    } else if (shape == 1) {
      for (uint32_t container = 0; container < 8; ++container)
        for (uint32_t i = 0; i < 20000; ++i)
          roaring_bitmap_add(posting.get(), container * 65536 + i * 2);
    } else if (shape == 2 || shape == 3) {
      for (uint32_t i = 0; i < (shape == 2 ? 2u : 32u); ++i)
        roaring_bitmap_add_range(posting.get(), i * 65536 + 10,
                                 i * 65536 + 110);
      ASSERT_TRUE(roaring_bitmap_run_optimize(posting.get()));
    } else if (shape == 4) {
      // The first array makes the following bitset internally unaligned,
      // even when the portable bitmap header has natural alignment.
      for (uint32_t i = 0; i < 33; ++i)
        roaring_bitmap_add(posting.get(), i * 2);
      for (uint32_t i = 0; i < 10000; ++i)
        roaring_bitmap_add(posting.get(), 65536 + i * 2);
      for (uint32_t container = 2; container < 8; ++container)
        for (uint32_t i = 0; i < 96; ++i)
          roaring_bitmap_add(posting.get(), container * 65536 + i * 37);
    } else {
      // Mixed arrays and bitsets decode into independent aligned storage.
      for (uint32_t container : {0u, 2u})
        for (uint32_t i = 0; i < 10000; ++i)
          roaring_bitmap_add(posting.get(), container * 65536 + i * 2);
      for (uint32_t i = 0; i < 33; ++i)
        roaring_bitmap_add(posting.get(), 65536 + i * 2);
    }
    std::string encoded;
    ASSERT_TRUE(InvertedIndexCodec::Serialize(posting.get(), &encoded).ok());
    ASSERT_EQ(encoded[0], 0);
    for (size_t shift = 0; shift < 8; ++shift) {
      for (int mode : {0, 1, 2, 3}) {
        const bool lazy = mode == 1;
        const bool subtract = mode == 2;
        const bool intersect = mode == 3;
        for (bool nonempty : {false, true}) {
          SCOPED_TRACE(testing::Message()
                       << "shape=" << shape << " shift=" << shift
                       << " mode=" << mode << " nonempty=" << nonempty);
          std::vector<uint64_t> storage(encoded.size() / sizeof(uint64_t) + 2);
          char *data = reinterpret_cast<char *>(storage.data()) + shift;
          std::memcpy(data, encoded.data(), encoded.size());
          auto result = MakeBitmap();
          Bitmap expected(subtract || intersect
                              ? roaring_bitmap_create()
                              : roaring_bitmap_copy(posting.get()),
                          roaring_bitmap_free);
          if (nonempty) {
            roaring_bitmap_add_range(result.get(), 0, 500);
            roaring_bitmap_add_range(expected.get(), 0, 500);
          }
          if (subtract)
            roaring_bitmap_andnot_inplace(expected.get(), posting.get());
          if (intersect)
            roaring_bitmap_and_inplace(expected.get(), posting.get());
          auto status = intersect ? InvertedIndexCodec::merge_and(
                                        data, encoded.size(), result.get())
                        : subtract
                            ? InvertedIndexCodec::merge_andnot(
                                  data, encoded.size(), result.get())
                            : InvertedIndexCodec::merge_or(data, encoded.size(),
                                                           lazy, result.get());
          ASSERT_TRUE(status.ok());
          std::memset(data, 0, encoded.size());
          if (lazy) roaring_bitmap_repair_after_lazy(result.get());
          EXPECT_TRUE(roaring_bitmap_equals(result.get(), expected.get()));
          EXPECT_TRUE(roaring_bitmap_internal_validate(result.get(), nullptr));
        }
      }
    }
  }
}

TEST(InvertedCodecMergeTest, SafeDecodeAlignsUnevenContainerCounts) {
  for (uint32_t count : {1u, 2u, 7u, 8u, 9u, 153u}) {
    auto posting = MakeBitmap();
    for (uint32_t container = 0; container < count; ++container)
      for (uint32_t i = 0; i < 4; ++i)
        roaring_bitmap_add(posting.get(), container * 65536 + i * 2);
    std::string encoded;
    ASSERT_TRUE(InvertedIndexCodec::Serialize(posting.get(), &encoded).ok());
    ASSERT_EQ(encoded[0], 0);
    roaring_bitmap_t *decoded = nullptr;
    ASSERT_TRUE(InvertedIndexCodec::Deserialize(encoded.data(), encoded.size(),
                                                &decoded)
                    .ok());
    Bitmap view(decoded, roaring_bitmap_free);
    ASSERT_NE(view, nullptr);
    for (uint32_t i = 0; i < count; ++i)
      ASSERT_EQ(
          reinterpret_cast<uintptr_t>(view->high_low_container.containers[i]) %
              alignof(void *),
          0u);
    EXPECT_TRUE(roaring_bitmap_equals(view.get(), posting.get()));
  }
}

TEST(InvertedCodecMergeTest, RejectsTruncatedPostingsBeforeChangingResult) {
  auto posting = MakeBitmap();
  for (uint32_t i = 0; i < 10000; ++i) roaring_bitmap_add(posting.get(), i * 2);
  std::string encoded;
  ASSERT_TRUE(InvertedIndexCodec::Serialize(posting.get(), &encoded).ok());
  auto result = MakeBitmap();
  roaring_bitmap_add(result.get(), UINT32_MAX);
  for (size_t length : {size_t{0}, size_t{1}, size_t{2}, size_t{4}, size_t{8},
                        encoded.size() - 1}) {
    EXPECT_FALSE(
        InvertedIndexCodec::merge_or(encoded.data(), length, true, result.get())
            .ok());
    EXPECT_FALSE(
        InvertedIndexCodec::merge_andnot(encoded.data(), length, result.get())
            .ok());
    EXPECT_EQ(roaring_bitmap_get_cardinality(result.get()), 1);
  }
  for (const std::string &invalid :
       {std::string(1, '\x01'), std::string(1, '\x21'),
        std::string("\x02\x00\x00\x00\x00", 5)}) {
    EXPECT_FALSE(InvertedIndexCodec::merge_or(invalid.data(), invalid.size(),
                                              true, result.get())
                     .ok());
    EXPECT_EQ(roaring_bitmap_get_cardinality(result.get()), 1);
  }
}

TEST(InvertedCodecMergeTest, NoncanonicalOffsetsAreReadSequentially) {
  auto posting = MakeBitmap();
  for (uint32_t container = 0; container < 8; ++container)
    for (uint32_t i = 0; i < 96; ++i)
      roaring_bitmap_add(posting.get(), container * 65536 + i * 37);
  std::string encoded;
  ASSERT_TRUE(InvertedIndexCodec::Serialize(posting.get(), &encoded).ok());
  std::vector<uint64_t> storage(encoded.size() / sizeof(uint64_t) + 2);
  char *data = reinterpret_cast<char *>(storage.data()) + 7;
  std::memcpy(data, encoded.data(), encoded.size());
  // Both bounded readers walk containers instead of trusting offsets.
  uint32_t offset = UINT32_MAX;
  std::memcpy(data + 1 + 8 + 8 * 4, &offset, sizeof(offset));
  auto result = MakeBitmap();
  ASSERT_TRUE(
      InvertedIndexCodec::merge_or(data, encoded.size(), true, result.get())
          .ok());
  roaring_bitmap_repair_after_lazy(result.get());
  EXPECT_TRUE(roaring_bitmap_equals(result.get(), posting.get()));
}

TEST(InvertedCodecMergeTest, LazyBitmapsAndShortListsKeepCardinality) {
  auto result = MakeBitmap();
  auto expected = MakeBitmap();
  for (int step = 0; step < 4; ++step) {
    auto posting = MakeBitmap();
    if (step % 2 == 0) {
      for (uint32_t i = 0; i < 10000; ++i)
        roaring_bitmap_add(posting.get(), i * 2 + step / 2);
    } else {
      for (uint32_t id : {1u, 7u, 20001u})
        roaring_bitmap_add(posting.get(), id);
    }
    std::string encoded;
    ASSERT_TRUE(InvertedIndexCodec::Serialize(posting.get(), &encoded).ok());
    ASSERT_TRUE(InvertedIndexCodec::merge_or(encoded.data(), encoded.size(),
                                             true, result.get())
                    .ok());
    roaring_bitmap_or_inplace(expected.get(), posting.get());
  }
  roaring_bitmap_repair_after_lazy(result.get());
  EXPECT_TRUE(roaring_bitmap_equals(result.get(), expected.get()));
  EXPECT_EQ(roaring_bitmap_get_cardinality(result.get()), 20001);
  EXPECT_TRUE(roaring_bitmap_internal_validate(result.get(), nullptr));
}

TEST(InvertedCodecMergeTest, ReusesViewWithoutBorrowingIntoResults) {
  InvertedIndexCodec::MergeScratch scratch;
  for (uint32_t count : {153u, 1u, 8u, 2u, 9u, 7u, 65536u}) {
    auto posting = MakeBitmap();
    for (uint32_t container = 0; container < count; ++container)
      for (uint32_t low = 0; low < 4; ++low)
        roaring_bitmap_add(posting.get(), container * 65536 + low * 2);
    std::string encoded;
    ASSERT_TRUE(InvertedIndexCodec::Serialize(posting.get(), &encoded).ok());
    auto result = MakeBitmap();
    ASSERT_TRUE(InvertedIndexCodec::merge_or(encoded.data(), encoded.size(),
                                             true, result.get(), &scratch)
                    .ok());
    EXPECT_TRUE(roaring_bitmap_equals(result.get(), posting.get()));
    EXPECT_EQ(encoded[0], 0);
    // Reusing scratch and invalidating the old input must leave the result
    // independent of both its source bytes and its temporary descriptors.
    std::fill(encoded.begin(), encoded.end(), '\0');
    auto other = MakeBitmap();
    roaring_bitmap_add_range(other.get(), 100, 10100);
    ASSERT_TRUE(InvertedIndexCodec::Serialize(other.get(), &encoded).ok());
    auto intersection = MakeBitmap();
    ASSERT_TRUE(InvertedIndexCodec::merge_or(encoded.data(), encoded.size(),
                                             false, intersection.get(),
                                             &scratch)
                    .ok());
    ASSERT_TRUE(InvertedIndexCodec::merge_and(encoded.data(), encoded.size(),
                                              intersection.get(), &scratch)
                    .ok());
    ASSERT_TRUE(InvertedIndexCodec::merge_andnot(encoded.data(), encoded.size(),
                                                 intersection.get(), &scratch)
                    .ok());
    EXPECT_TRUE(roaring_bitmap_is_empty(intersection.get()));
    roaring_bitmap_repair_after_lazy(result.get());
    EXPECT_TRUE(roaring_bitmap_equals(result.get(), posting.get()));
    EXPECT_TRUE(roaring_bitmap_internal_validate(result.get(), nullptr));
  }
}

TEST(InvertedCodecMergeTest, ViewValidatesAllTruncationsAndAlignsMetadata) {
  zvec_inverted_bitmap_view scratch{};
  for (bool runs : {false, true}) {
    for (uint32_t count : {1u, 2u, 7u, 8u, 9u, 153u}) {
      auto posting = MakeBitmap();
      for (uint32_t container = 0; container < count; ++container)
        roaring_bitmap_add_range(posting.get(), container * 65536,
                                 container * 65536 + (runs ? 64 : 4));
      if (runs) {
        ASSERT_TRUE(roaring_bitmap_run_optimize(posting.get()));
      }
      std::string encoded;
      ASSERT_TRUE(InvertedIndexCodec::Serialize(posting.get(), &encoded).ok());
      for (size_t shift = 0; shift < 8; ++shift) {
        std::vector<uint64_t> storage(encoded.size() / sizeof(uint64_t) + 2);
        char *data = reinterpret_cast<char *>(storage.data()) + shift;
        std::memcpy(data, encoded.data(), encoded.size());
        const std::string original(data, encoded.size());
        for (size_t length = 0; length < encoded.size() - 1; ++length)
          EXPECT_EQ(zvec_inverted_bitmap_view_load(&scratch, data + 1, length),
                    nullptr);
        const auto *view = zvec_inverted_bitmap_view_load(&scratch, data + 1,
                                                          encoded.size() - 1);
        ASSERT_NE(view, nullptr);
        EXPECT_TRUE(roaring_bitmap_equals(view, posting.get()));
        EXPECT_TRUE(roaring_bitmap_internal_validate(view, nullptr));
        for (int32_t i = 0; i < view->high_low_container.size; ++i)
          EXPECT_EQ(reinterpret_cast<uintptr_t>(
                        view->high_low_container.containers[i]) %
                        alignof(void *),
                    0u);
        EXPECT_EQ(std::string(data, encoded.size()), original);
      }
    }
  }
  zvec_inverted_bitmap_view_clear(&scratch);
}

TEST(InvertedCodecMergeTest, EmptyPortableViewAndInvalidCounts) {
  auto empty = MakeBitmap();
  std::string encoded(1 + roaring_bitmap_portable_size_in_bytes(empty.get()),
                      '\0');
  roaring_bitmap_portable_serialize(empty.get(), encoded.data() + 1);
  InvertedIndexCodec::MergeScratch scratch;
  auto result = MakeBitmap();
  roaring_bitmap_add(result.get(), UINT32_MAX);
  ASSERT_TRUE(InvertedIndexCodec::merge_or(encoded.data(), encoded.size(), true,
                                           result.get(), &scratch)
                  .ok());
  ASSERT_TRUE(InvertedIndexCodec::merge_andnot(encoded.data(), encoded.size(),
                                               result.get(), &scratch)
                  .ok());
  EXPECT_EQ(roaring_bitmap_get_cardinality(result.get()), 1u);
  ASSERT_TRUE(InvertedIndexCodec::merge_and(encoded.data(), encoded.size(),
                                            result.get(), &scratch)
                  .ok());
  EXPECT_TRUE(roaring_bitmap_is_empty(result.get()));
  for (uint32_t count : {65537u, UINT32_MAX}) {
    std::memcpy(encoded.data() + 5, &count, sizeof(count));
    EXPECT_FALSE(InvertedIndexCodec::merge_or(encoded.data(), encoded.size(),
                                              true, result.get(), &scratch)
                     .ok());
    EXPECT_TRUE(roaring_bitmap_is_empty(result.get()));
  }
}

TEST(InvertedCodecMergeTest, DenseMergesReadUnalignedPayloadsDirectly) {
  for (int shape : {0, 1, 2, 4, 5}) {
    const uint32_t count = shape == 0   ? 153
                           : shape == 1 ? 8
                           : shape == 2 ? 9
                                        : 2;
    auto posting = MakeBitmap();
    for (uint32_t high = 0; high < count; ++high) {
      const uint32_t cardinality = shape == 4   ? 4096
                                   : shape == 5 ? 4097
                                   : shape == 0 || (shape == 2 && high % 2)
                                       ? 64
                                       : 10000;
      const uint32_t key = shape >= 4 && high + 1 == count ? 65535 : high;
      for (uint32_t low = 0; low < cardinality; ++low)
        roaring_bitmap_add(posting.get(),
                           (key << 16) + ((low * 673 + high * 73) & 65535));
    }
    std::string encoded;
    ASSERT_TRUE(InvertedIndexCodec::Serialize(posting.get(), &encoded).ok());
    for (bool full : {false, true}) {
      auto initial = MakeBitmap();
      for (uint32_t high = 0; high < count; ++high) {
        const uint32_t key = shape >= 4 && high + 1 == count ? 65535 : high;
        for (uint32_t low = 0; low < 65536; low += full ? 1 : 4)
          roaring_bitmap_add(initial.get(), (key << 16) + low);
      }
      for (size_t shift = 0; shift < 8; ++shift) {
        for (int mode : {0, 1, 2}) {
          SCOPED_TRACE(testing::Message()
                       << "shape=" << shape << " full=" << full
                       << " shift=" << shift << " mode=" << mode);
          const bool subtract = mode == 2;
          const bool lazy = mode == 1;
          std::vector<uint64_t> storage(encoded.size() / sizeof(uint64_t) + 2);
          char *data = reinterpret_cast<char *>(storage.data()) + shift;
          std::memcpy(data, encoded.data(), encoded.size());
          Bitmap result(roaring_bitmap_copy(initial.get()),
                        roaring_bitmap_free);
          Bitmap expected(roaring_bitmap_copy(initial.get()),
                          roaring_bitmap_free);
          Bitmap direct(roaring_bitmap_copy(initial.get()),
                        roaring_bitmap_free);
          ASSERT_EQ(
              zvec_inverted_bitmap_try_merge_dense(
                  direct.get(), data + 1, encoded.size() - 1, subtract, lazy),
              1);
          if (lazy) roaring_bitmap_repair_after_lazy(direct.get());
          if (subtract)
            roaring_bitmap_andnot_inplace(expected.get(), posting.get());
          else
            roaring_bitmap_or_inplace(expected.get(), posting.get());
          EXPECT_TRUE(roaring_bitmap_equals(direct.get(), expected.get()));
          InvertedIndexCodec::MergeScratch scratch;
          for (int repeat = 0; repeat < 2; ++repeat) {
            ASSERT_TRUE(
                (subtract
                     ? InvertedIndexCodec::merge_andnot(data, encoded.size(),
                                                        result.get(), &scratch)
                     : InvertedIndexCodec::merge_or(data, encoded.size(), lazy,
                                                    result.get(), &scratch))
                    .ok());
          }
          if (lazy) roaring_bitmap_repair_after_lazy(result.get());
          EXPECT_TRUE(roaring_bitmap_equals(result.get(), expected.get()));
          EXPECT_TRUE(roaring_bitmap_internal_validate(result.get(), nullptr));
          EXPECT_EQ(std::string(data, encoded.size()), encoded);
        }
      }
    }
  }
}

TEST(InvertedCodecMergeTest, DensePreflightNeverPartiallyChangesResult) {
  auto posting = MakeBitmap();
  auto initial = MakeBitmap();
  for (uint32_t high = 0; high < 2; ++high) {
    for (uint32_t low = 0; low < 10000; ++low)
      roaring_bitmap_add(posting.get(), (high << 16) + low * 2);
    for (uint32_t low = 0; low < 65536; low += 4)
      roaring_bitmap_add(initial.get(), (high << 16) + low);
  }
  std::string encoded;
  ASSERT_TRUE(InvertedIndexCodec::Serialize(posting.get(), &encoded).ok());
  for (bool subtract : {false, true}) {
    Bitmap result(roaring_bitmap_copy(initial.get()), roaring_bitmap_free);
    for (size_t length = 0; length < encoded.size() - 1; ++length)
      ASSERT_EQ(zvec_inverted_bitmap_try_merge_dense(
                    result.get(), encoded.data() + 1, length, subtract, false),
                0);
    EXPECT_TRUE(roaring_bitmap_equals(result.get(), initial.get()));
    EXPECT_FALSE(
        (subtract
             ? InvertedIndexCodec::merge_andnot(
                   encoded.data(), encoded.size() - 1, result.get())
             : InvertedIndexCodec::merge_or(encoded.data(), encoded.size() - 1,
                                            false, result.get()))
            .ok());
    EXPECT_TRUE(roaring_bitmap_equals(result.get(), initial.get()));
  }
  // The final key mismatch must be detected before merging the first key.
  const uint16_t key = 2;
  std::memcpy(encoded.data() + 1 + 8 + 4, &key, sizeof(key));
  EXPECT_EQ(
      zvec_inverted_bitmap_try_merge_dense(initial.get(), encoded.data() + 1,
                                           encoded.size() - 1, true, false),
      0);
  EXPECT_EQ(roaring_bitmap_get_cardinality(initial.get()), 32768u);
  EXPECT_TRUE(roaring_bitmap_contains(initial.get(), 0));
}

TEST(InvertedCodecMergeTest, DenseAllOnesSourceKeepsFullCardinality) {
  auto posting = MakeBitmap();
  auto initial = MakeBitmap();
  for (uint32_t high = 0; high < 2; ++high) {
    for (uint32_t low = 0; low < 65536; ++low)
      roaring_bitmap_add(posting.get(), (high << 16) + low);
    for (uint32_t low = 0; low < 65536; low += 4)
      roaring_bitmap_add(initial.get(), (high << 16) + low);
  }
  std::string encoded;
  ASSERT_TRUE(InvertedIndexCodec::Serialize(posting.get(), &encoded).ok());
  for (bool lazy : {false, true}) {
    Bitmap result(roaring_bitmap_copy(initial.get()), roaring_bitmap_free);
    ASSERT_TRUE(InvertedIndexCodec::merge_or(encoded.data(), encoded.size(),
                                             lazy, result.get())
                    .ok());
    if (lazy) roaring_bitmap_repair_after_lazy(result.get());
    EXPECT_EQ(roaring_bitmap_get_cardinality(result.get()), 131072u);
    EXPECT_TRUE(roaring_bitmap_equals(result.get(), posting.get()));
    EXPECT_TRUE(roaring_bitmap_internal_validate(result.get(), nullptr));
    ASSERT_TRUE(InvertedIndexCodec::merge_andnot(encoded.data(), encoded.size(),
                                                 result.get())
                    .ok());
    EXPECT_TRUE(roaring_bitmap_is_empty(result.get()));
    EXPECT_TRUE(roaring_bitmap_internal_validate(result.get(), nullptr));
  }
}

TEST(InvertedCodecMergeTest, DenseFallbacksPreserveOwnershipAndConversions) {
  auto posting = MakeBitmap();
  for (uint32_t high = 0; high < 2; ++high)
    for (uint32_t low = 0; low < 10000; ++low)
      roaring_bitmap_add(posting.get(), (high << 16) + low * 2);
  std::string encoded;
  ASSERT_TRUE(InvertedIndexCodec::Serialize(posting.get(), &encoded).ok());
  for (int variant : {0, 1, 2, 3}) {
    auto initial = MakeBitmap();
    for (uint32_t high = 0; high < 2; ++high) {
      const uint32_t count = variant == 0                ? 5000
                             : variant == 1 && high == 1 ? 3
                                                         : 16000;
      const uint32_t key = variant == 3 && high == 1 ? 2 : high;
      for (uint32_t low = 0; low < count; ++low)
        roaring_bitmap_add(initial.get(), (key << 16) + low * 2);
    }
    if (variant == 2) roaring_bitmap_set_copy_on_write(initial.get(), true);
    Bitmap original(roaring_bitmap_copy(initial.get()), roaring_bitmap_free);
    Bitmap result(roaring_bitmap_copy(initial.get()), roaring_bitmap_free);
    Bitmap expected(roaring_bitmap_copy(initial.get()), roaring_bitmap_free);
    EXPECT_EQ(
        zvec_inverted_bitmap_try_merge_dense(result.get(), encoded.data() + 1,
                                             encoded.size() - 1, true, false),
        0);
    EXPECT_TRUE(roaring_bitmap_equals(result.get(), original.get()));
    roaring_bitmap_andnot_inplace(expected.get(), posting.get());
    ASSERT_TRUE(InvertedIndexCodec::merge_andnot(encoded.data(), encoded.size(),
                                                 result.get())
                    .ok());
    EXPECT_TRUE(roaring_bitmap_equals(result.get(), expected.get()));
    EXPECT_TRUE(roaring_bitmap_equals(initial.get(), original.get()));
    EXPECT_TRUE(roaring_bitmap_internal_validate(result.get(), nullptr));
    if (variant == 0) {
      EXPECT_TRUE(roaring_bitmap_is_empty(result.get()));
    }
  }
}

TEST(InvertedCodecMergeTest, DeferredSubtractRestoresMixedContainerLayout) {
  constexpr uint32_t kBlock = 1u << 16;
  auto initial = MakeBitmap();
  auto first = MakeBitmap();
  auto second = MakeBitmap();
  auto different_keys = MakeBitmap();
  for (uint32_t i = 0; i < 5000; ++i) {
    const uint32_t value = i * 2;
    roaring_bitmap_add(initial.get(), value);
    roaring_bitmap_add(initial.get(), 2 * kBlock + value);
    roaring_bitmap_add(first.get(), 2 * kBlock + value);
    roaring_bitmap_add(second.get(), kBlock + value);
    roaring_bitmap_add(different_keys.get(), 3 * kBlock + value);
    if (i < 800) roaring_bitmap_add(initial.get(), kBlock + value);
    if (i < 2000) roaring_bitmap_add(first.get(), value);
    if (i < 400) roaring_bitmap_add(first.get(), kBlock + value);
    if (i >= 2000 && i < 2500) roaring_bitmap_add(second.get(), value);
  }
  roaring_bitmap_add(second.get(), 2 * kBlock + 1);
  roaring_bitmap_add(different_keys.get(), 9000);
  roaring_bitmap_add(different_keys.get(), kBlock + 1);
  Bitmap truncated(roaring_bitmap_copy(second.get()), roaring_bitmap_free);
  roaring_bitmap_add(truncated.get(), 9000);
  Bitmap result(roaring_bitmap_copy(initial.get()), roaring_bitmap_free);
  Bitmap expected(roaring_bitmap_copy(initial.get()), roaring_bitmap_free);
  zvec_inverted_bitmap_subtract scratch{};
  AILEGO_DEFER([&]() { roaring_aligned_free(scratch.array_words); });
  std::string first_data, second_data, truncated_data, fallback_data;
  ASSERT_TRUE(InvertedIndexCodec::Serialize(first.get(), &first_data).ok());
  ASSERT_TRUE(InvertedIndexCodec::Serialize(second.get(), &second_data).ok());
  ASSERT_TRUE(
      InvertedIndexCodec::Serialize(truncated.get(), &truncated_data).ok());
  ASSERT_TRUE(
      InvertedIndexCodec::Serialize(different_keys.get(), &fallback_data).ok());
  ASSERT_TRUE(zvec_inverted_bitmap_defer_subtract(
      &scratch, result.get(), first_data.data() + 1, first_data.size() - 1));
  EXPECT_FALSE(zvec_inverted_bitmap_defer_subtract(
      &scratch, result.get(), truncated_data.data() + 1,
      truncated_data.size() - 2));
  ASSERT_TRUE(zvec_inverted_bitmap_defer_subtract(
      &scratch, result.get(), second_data.data() + 1, second_data.size() - 1));
  EXPECT_FALSE(zvec_inverted_bitmap_defer_subtract(
      &scratch, result.get(), fallback_data.data() + 1,
      fallback_data.size() - 1));
  ASSERT_TRUE(zvec_inverted_bitmap_finish_subtract(&scratch, result.get()));
  EXPECT_EQ(scratch.array_words, nullptr);

  roaring_bitmap_andnot_inplace(expected.get(), first.get());
  roaring_bitmap_andnot_inplace(expected.get(), second.get());
  EXPECT_TRUE(roaring_bitmap_equals(result.get(), expected.get()));
  EXPECT_TRUE(roaring_bitmap_internal_validate(result.get(), nullptr));
  EXPECT_EQ(roaring_bitmap_get_cardinality(result.get()), 2500);

  ASSERT_TRUE(InvertedIndexCodec::merge_andnot(fallback_data.data(),
                                               fallback_data.size(),
                                               result.get())
                  .ok());
  roaring_bitmap_andnot_inplace(expected.get(), different_keys.get());
  EXPECT_TRUE(roaring_bitmap_equals(result.get(), expected.get()));
  EXPECT_TRUE(roaring_bitmap_internal_validate(result.get(), nullptr));
}

}  // namespace
}  // namespace zvec
