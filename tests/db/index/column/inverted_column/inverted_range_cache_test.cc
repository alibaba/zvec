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
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include "db/index/column/inverted_column/inverted_column_indexer.h"
#include "db/index/column/inverted_column/inverted_rocksdb_merger.h"
#include "tests/test_util.h"

namespace zvec {
namespace {

constexpr std::array<CompareOp, 4> kRangeOperations = {
    CompareOp::LT, CompareOp::LE, CompareOp::GT, CompareOp::GE};

template <typename T>
std::string RawNumber(T value) {
  return std::string(reinterpret_cast<const char *>(&value), sizeof(value));
}

std::string RawInt32(int32_t value) {
  return RawNumber(value);
}

int32_t DecodeInt32(const std::string &encoded) {
  uint32_t bits = 0;
  for (unsigned char byte : encoded) {
    bits = (bits << 8) | byte;
  }
  bits ^= uint32_t{1} << 31;
  int32_t value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

template <typename T>
bool Matches(const T &value, const T &bound, CompareOp op) {
  switch (op) {
    case CompareOp::LT:
      return value < bound;
    case CompareOp::LE:
      return value <= bound;
    case CompareOp::GT:
      return value > bound;
    case CompareOp::GE:
      return value >= bound;
    default:
      return false;
  }
}

class InvertedRangeCacheTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    ASSERT_EQ(ailego::MemoryLimitPool::get_instance().init(256ULL << 20), 0);
  }

  void SetUp() override {
    path_ = std::string("./inverted_range_cache_") +
            testing::UnitTest::GetInstance()->current_test_info()->name();
    test_util::RemoveTestPath(path_);
    initial_external_bytes_ =
        ailego::MemoryLimitPool::get_instance().external_used();
  }

  void TearDown() override {
    column_.reset();
    context_.close();
    EXPECT_EQ(ailego::MemoryLimitPool::get_instance().external_used(),
              initial_external_bytes_);
    test_util::RemoveTestPath(path_);
  }

  void CreateWriter(const RangePrefixCacheOptions &options = {},
                    DataType type = DataType::INT32,
                    bool enable_ranges = true) {
    field_ =
        FieldSchema("value", type, true,
                    std::make_shared<InvertIndexParams>(enable_ranges, false));
    std::vector<std::string> families{field_.name() + INVERT_SUFFIX_TERMS,
                                      INVERT_CDF};
    if (enable_ranges) {
      families.push_back(field_.name() + INVERT_SUFFIX_RANGES);
    }
    ASSERT_TRUE(context_
                    .create(path_, families,
                            std::make_shared<InvertedRocksdbValueMerger>())
                    .ok());
    column_ = InvertedColumnIndexer::CreateAndOpen("range_cache_test", field_,
                                                   context_, false, options);
    ASSERT_NE(column_, nullptr);
    EXPECT_EQ(column_->range_cache(), nullptr);
  }

  void Reopen(const RangePrefixCacheOptions &options = {}) {
    column_.reset();
    if (!context_.read_only()) {
      ASSERT_TRUE(context_.flush().ok());
    }
    ASSERT_TRUE(context_.close().ok());
    ASSERT_TRUE(
        context_
            .open(path_, true, std::make_shared<InvertedRocksdbValueMerger>())
            .ok());
    column_ = InvertedColumnIndexer::CreateAndOpen("range_cache_test", field_,
                                                   context_, true, options);
    ASSERT_NE(column_, nullptr);
    EXPECT_TRUE(column_->is_sealed());
  }

  void Insert(const std::vector<std::optional<int32_t>> &values) {
    for (uint32_t id = 0; id < values.size(); ++id) {
      ASSERT_TRUE(values[id] ? column_->insert(id, RawInt32(*values[id])).ok()
                             : column_->insert_null(id).ok())
          << "doc ID " << id;
    }
  }

  static std::vector<std::optional<int32_t>> RandomValues(size_t count,
                                                          bool with_null) {
    std::vector<int32_t> order(count);
    for (size_t i = 0; i < count; ++i) {
      order[i] = static_cast<int32_t>(i) - static_cast<int32_t>(count / 2);
    }
    std::mt19937 random(868);
    std::shuffle(order.begin(), order.end(), random);
    std::vector<std::optional<int32_t>> values;
    values.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      values.emplace_back(with_null && i % 97 == 0
                              ? std::optional<int32_t>()
                              : std::optional<int32_t>(order[i]));
    }
    return values;
  }

  std::vector<int32_t> NumericBounds() const {
    std::vector<int32_t> bounds{std::numeric_limits<int32_t>::min(),
                                -20000,
                                -6000,
                                -5999,
                                -4801,
                                -3601,
                                -3001,
                                -2641,
                                -1,
                                0,
                                1,
                                1200,
                                2399,
                                4799,
                                5999,
                                6000,
                                20000,
                                std::numeric_limits<int32_t>::max()};
    if (const auto *cache = column_->range_cache()) {
      for (const auto &prefix : cache->prefixes) {
        const int32_t end =
            DecodeInt32(cache->buckets[prefix.end_bucket - 1].end);
        bounds.push_back(end);
        if (end != std::numeric_limits<int32_t>::min()) {
          bounds.push_back(end - 1);
        }
        if (end != std::numeric_limits<int32_t>::max()) {
          bounds.push_back(end + 1);
        }
      }
    }
    std::sort(bounds.begin(), bounds.end());
    bounds.erase(std::unique(bounds.begin(), bounds.end()), bounds.end());
    return bounds;
  }

  void CheckNumeric(const std::vector<std::optional<int32_t>> &values,
                    const std::vector<int32_t> &bounds) const {
    for (const int32_t bound : bounds) {
      for (const CompareOp op : kRangeOperations) {
        SCOPED_TRACE(testing::Message()
                     << "bound=" << bound << ", op=" << static_cast<int>(op));
        auto result = column_->search(RawInt32(bound), op);
        ASSERT_NE(result, nullptr);
        size_t expected_count = 0;
        for (uint32_t id = 0; id < values.size(); ++id) {
          const bool expected = values[id] && Matches(*values[id], bound, op);
          ASSERT_EQ(result->contains(id), expected) << "doc ID " << id;
          expected_count += expected;
        }
        EXPECT_EQ(result->count(), expected_count);
        EXPECT_FALSE(result->contains(static_cast<uint32_t>(values.size())));
      }
    }
  }

  template <typename T>
  void CheckTypedNumeric(const std::vector<std::optional<T>> &values,
                         const std::vector<T> &bounds) const {
    for (const T bound : bounds) {
      for (const CompareOp op : kRangeOperations) {
        SCOPED_TRACE(testing::Message()
                     << "bound=" << bound << ", op=" << static_cast<int>(op));
        auto result = column_->search(RawNumber(bound), op);
        ASSERT_NE(result, nullptr);
        size_t expected_count = 0;
        for (uint32_t id = 0; id < values.size(); ++id) {
          const bool expected = values[id] && Matches(*values[id], bound, op);
          ASSERT_EQ(result->contains(id), expected) << "doc ID " << id;
          expected_count += expected;
        }
        EXPECT_EQ(result->count(), expected_count);
      }
    }
  }

  template <typename T>
  void CheckTypedSealAndReopen(DataType type,
                               std::vector<std::optional<T>> values,
                               std::vector<T> bounds) {
    CreateWriter({}, type);
    std::mt19937 random(868);
    std::shuffle(values.begin(), values.end(), random);
    for (uint32_t id = 0; id < values.size(); ++id) {
      ASSERT_TRUE(values[id] ? column_->insert(id, RawNumber(*values[id])).ok()
                             : column_->insert_null(id).ok());
    }
    ASSERT_TRUE(column_->seal().ok());
    const auto *cache = column_->range_cache();
    ASSERT_NE(cache, nullptr);
    ASSERT_FALSE(cache->prefixes.empty());
    // Query actual checkpoint values as well as the nearest representable
    // value on each side. Numeric percentile thresholds alone rarely test
    // the inclusive/exclusive switch exactly at a stored bucket boundary.
    for (const auto &prefix : cache->prefixes) {
      const auto &end = cache->buckets[prefix.end_bucket - 1].end;
      const auto found = std::find_if(
          values.begin(), values.end(), [&](const std::optional<T> &value) {
            return value &&
                   InvertedIndexCodec::Encode(RawNumber(*value), type) == end;
          });
      ASSERT_NE(found, values.end());
      const T value = **found;
      bounds.push_back(value);
      if constexpr (std::is_floating_point_v<T>) {
        bounds.push_back(
            std::nextafter(value, -std::numeric_limits<T>::infinity()));
        bounds.push_back(
            std::nextafter(value, std::numeric_limits<T>::infinity()));
      } else {
        if (value > std::numeric_limits<T>::min()) {
          bounds.push_back(value - 1);
        }
        if (value < std::numeric_limits<T>::max()) {
          bounds.push_back(value + 1);
        }
      }
    }
    std::sort(bounds.begin(), bounds.end());
    bounds.erase(std::unique(bounds.begin(), bounds.end()), bounds.end());
    bool used_prefix = false;
    for (const T bound : bounds) {
      for (const auto op : kRangeOperations) {
        const auto path = column_->explain_range(RawNumber(bound), op).path;
        used_prefix |= path == RangeQueryPath::PrefixAdd ||
                       path == RangeQueryPath::PrefixSubtract;
      }
    }
    ASSERT_TRUE(used_prefix);
    CheckTypedNumeric(values, bounds);
    Reopen();
    ASSERT_NE(column_->range_cache(), nullptr);
    CheckTypedNumeric(values, bounds);
  }

  template <typename T>
  void CheckFloatingPoint(DataType type) {
    std::vector<std::optional<T>> values(12000);
    for (size_t i = 0; i < values.size(); ++i) {
      if (i % 97 != 0) {
        values[i] = static_cast<T>(static_cast<int32_t>(i) - 6000) / T{8};
      }
    }
    const std::vector<T> bounds{-std::numeric_limits<T>::infinity(),
                                -std::numeric_limits<T>::max(),
                                -std::numeric_limits<T>::min(),
                                -std::numeric_limits<T>::denorm_min(),
                                T{0},
                                std::numeric_limits<T>::denorm_min(),
                                std::numeric_limits<T>::min(),
                                std::numeric_limits<T>::max(),
                                std::numeric_limits<T>::infinity()};
    for (size_t i = 0; i < bounds.size(); ++i) {
      values[i + 1] = bounds[i];
    }
    CheckTypedSealAndReopen(type, std::move(values), bounds);
  }

  std::vector<std::string> CacheContents() const {
    std::vector<std::string> result;
    if (const auto *cache = column_->range_cache()) {
      for (const auto &prefix : cache->prefixes) {
        std::string encoded(
            roaring_bitmap_portable_size_in_bytes(prefix.bitmap.get()), '\0');
        roaring_bitmap_portable_serialize(prefix.bitmap.get(), encoded.data());
        result.push_back(std::move(encoded));
      }
    }
    return result;
  }

  std::vector<std::pair<std::string, std::string>> RangeRows() {
    std::vector<std::pair<std::string, std::string>> result;
    auto *family = context_.get_cf(field_.name() + INVERT_SUFFIX_RANGES);
    if (!family) {
      return result;
    }
    std::unique_ptr<rocksdb::Iterator> it(
        context_.db_->NewIterator(context_.read_opts_, family));
    for (it->SeekToFirst(); it->Valid(); it->Next()) {
      result.emplace_back(it->key().ToString(), it->value().ToString());
    }
    EXPECT_TRUE(it->status().ok());
    return result;
  }

  std::string path_;
  FieldSchema field_;
  RocksdbContext context_;
  InvertedColumnIndexer::Ptr column_;
  size_t initial_external_bytes_{0};
};

TEST_F(InvertedRangeCacheTest, FreshSealAndReopenPreserveRangeSemantics) {
  CreateWriter();
  const auto values = RandomValues(12000, true);
  Insert(values);
  ASSERT_TRUE(column_->seal().ok());
  const auto *cache = column_->range_cache();
  ASSERT_NE(cache, nullptr);
  ASSERT_EQ(cache->prefixes.size(), 9);
  size_t previous_end = 0;
  for (size_t i = 0; i < cache->prefixes.size(); ++i) {
    const auto &prefix = cache->prefixes[i];
    EXPECT_GT(prefix.end_bucket, previous_end);
    ASSERT_LE(prefix.end_bucket, cache->buckets.size());
    const auto &last_bucket = cache->buckets[prefix.end_bucket - 1];
    const uint64_t target = (cache->total_cardinality * (i + 1) + 9) / 10;
    EXPECT_GE(prefix.cardinality, target);
    EXPECT_LT(prefix.cardinality, target + last_bucket.cardinality);
    EXPECT_EQ(prefix.cardinality,
              roaring_bitmap_get_cardinality(prefix.bitmap.get()));
    previous_end = prefix.end_bucket;
  }
  auto bounds = NumericBounds();
  CheckNumeric(values, bounds);
  const auto range_rows = RangeRows();
  const auto contents = CacheContents();
  Reopen();
  ASSERT_NE(column_->range_cache(), nullptr);
  CheckNumeric(values, bounds);
  EXPECT_EQ(RangeRows(), range_rows);
  EXPECT_EQ(CacheContents(), contents);
}

TEST_F(InvertedRangeCacheTest, OpensUnchangedIndexBuiltWithoutCache) {
  RangePrefixCacheOptions disabled;
  disabled.max_bytes = 0;
  CreateWriter(disabled);
  const auto values = RandomValues(12000, false);
  Insert(values);
  ASSERT_TRUE(column_->seal().ok());
  ASSERT_EQ(column_->range_cache(), nullptr);
  const auto original_rows = RangeRows();
  ASSERT_FALSE(original_rows.empty());
  Reopen();
  ASSERT_NE(column_->range_cache(), nullptr);
  CheckNumeric(values, NumericBounds());
  EXPECT_EQ(RangeRows(), original_rows);
  std::vector<std::string> actual_families;
  for (const auto *family : context_.cf_handles_) {
    actual_families.push_back(family->GetName());
  }
  std::sort(actual_families.begin(), actual_families.end());
  std::vector<std::string> expected_families{INVERT_CDF, "default",
                                             "value$RANGES", "value$TERMS"};
  std::sort(expected_families.begin(), expected_families.end());
  EXPECT_EQ(actual_families, expected_families);
}

TEST_F(InvertedRangeCacheTest, RepeatedValuesAndNullsAtInclusiveBoundaries) {
  CreateWriter();
  std::vector<std::optional<int32_t>> values(15000, 0);
  for (int32_t i = 0; i < 1500; ++i) {
    values[i] = -1500 + i;
    values[13500 + i] = i + 1;
  }
  for (size_t i = 37; i < values.size(); i += 101) {
    values[i].reset();
  }
  std::mt19937 random(868);
  std::shuffle(values.begin(), values.end(), random);
  Insert(values);
  ASSERT_TRUE(column_->seal().ok());
  ASSERT_NE(column_->range_cache(), nullptr);
  EXPECT_LE(column_->range_cache()->prefixes.size(), 9);
  CheckNumeric(values, NumericBounds());
  Reopen();
  CheckNumeric(values, NumericBounds());
}

TEST_F(InvertedRangeCacheTest, OrderedPrefixesUseRunContainers) {
  CreateWriter();
  std::vector<std::optional<int32_t>> values(12000);
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<int32_t>(i) - 6000;
  }
  Insert(values);
  ASSERT_TRUE(column_->seal().ok());
  ASSERT_NE(column_->range_cache(), nullptr);
  ASSERT_EQ(column_->range_cache()->prefixes.size(), 9);
  for (const auto &prefix : column_->range_cache()->prefixes) {
    roaring_statistics_t stats{};
    roaring_bitmap_statistics(prefix.bitmap.get(), &stats);
    // Every prefix is one contiguous interval inside a single high16 block.
    EXPECT_EQ(stats.n_run_containers, 1);
    EXPECT_EQ(stats.n_array_containers, 0);
    EXPECT_EQ(stats.n_bitset_containers, 0);
  }
  CheckNumeric(values, NumericBounds());
}

TEST_F(InvertedRangeCacheTest, LowerOnlyPolicyPreservesAllComparisons) {
  RangePrefixCacheOptions lower_only;
  lower_only.allow_subtraction = false;
  CreateWriter(lower_only);
  const auto values = RandomValues(12000, true);
  Insert(values);
  ASSERT_TRUE(column_->seal().ok());
  ASSERT_NE(column_->range_cache(), nullptr);
  const auto bounds = NumericBounds();
  for (const int32_t bound : bounds) {
    for (const auto op : kRangeOperations) {
      EXPECT_NE(column_->explain_range(RawInt32(bound), op).path,
                RangeQueryPath::PrefixSubtract);
    }
  }
  CheckNumeric(values, bounds);
  Reopen(lower_only);
  CheckNumeric(values, bounds);
}

TEST_F(InvertedRangeCacheTest, BoundaryTermsKeepLegacyPhysicalScanDirection) {
  CreateWriter();
  std::vector<std::optional<int32_t>> values(12000);
  for (size_t i = 0; i < values.size(); ++i) {
    if (i % 97 != 0) {
      values[i] = static_cast<int32_t>(i) - 6000;
    }
  }
  Insert(values);
  ASSERT_TRUE(column_->seal().ok());
  const auto *cache = column_->range_cache();
  ASSERT_NE(cache, nullptr);

  RangePrefixCacheOptions disabled;
  disabled.max_bytes = 0;
  auto legacy = InvertedColumnIndexer::CreateAndOpen("range_cache_test", field_,
                                                     context_, true, disabled);
  ASSERT_NE(legacy, nullptr);
  ASSERT_EQ(legacy->range_cache(), nullptr);

  std::vector<int32_t> bounds;
  bool used_scan_left = false;
  bool used_scan_right = false;
  bool used_add = false;
  bool used_subtract = false;
  for (const size_t percent : {5, 15, 25, 35, 45, 55, 65, 75, 85, 95}) {
    const auto &bucket = cache->buckets[cache->buckets.size() * percent / 100];
    const int32_t begin = DecodeInt32(bucket.begin);
    const int32_t end = DecodeInt32(bucket.end);
    ASSERT_GT(end - begin, 2);
    int32_t bound = begin + (end - begin) / 2;
    // Pick an existing term strictly inside the bucket, so all four operators
    // require exact boundary work and < versus <= has an observable result.
    while (bound < end && !values[static_cast<size_t>(bound + 6000)]) {
      ++bound;
    }
    ASSERT_GT(bound, begin);
    ASSERT_LT(bound, end);
    bounds.push_back(bound);
    for (const auto op : kRangeOperations) {
      SCOPED_TRACE(testing::Message()
                   << "percent=" << percent << ", op=" << static_cast<int>(op));
      uint64_t total = 0;
      uint64_t matching = 0;
      ASSERT_TRUE(
          legacy->evaluate_ratio(RawInt32(bound), op, &total, &matching).ok());
      ASSERT_GT(total, 0);
      const bool query_left = op == CompareOp::LT || op == CompareOp::LE;
      const bool old_complement = 1.0 * matching / total > 0.7;
      const bool expected_boundary_left = query_left != old_complement;
      const auto old_plan = legacy->explain_range(RawInt32(bound), op);
      EXPECT_EQ(old_plan.path, expected_boundary_left
                                   ? RangeQueryPath::ScanLeft
                                   : RangeQueryPath::ScanRight);

      const auto plan = column_->explain_range(RawInt32(bound), op);
      ASSERT_NE(plan.boundary_bucket, SIZE_MAX);
      const bool actual_boundary_left = plan.path == RangeQueryPath::ScanLeft ||
                                        plan.path == RangeQueryPath::PrefixAdd;
      EXPECT_EQ(actual_boundary_left, expected_boundary_left);
      used_scan_left |= plan.path == RangeQueryPath::ScanLeft;
      used_scan_right |= plan.path == RangeQueryPath::ScanRight;
      used_add |= plan.path == RangeQueryPath::PrefixAdd;
      used_subtract |= plan.path == RangeQueryPath::PrefixSubtract;
    }
  }
  EXPECT_TRUE(used_scan_left);
  EXPECT_TRUE(used_scan_right);
  EXPECT_TRUE(used_add);
  EXPECT_TRUE(used_subtract);
  CheckNumeric(values, bounds);
}

TEST_F(InvertedRangeCacheTest, ReadOnlyQueriesDoNotMutateSharedPrefixes) {
  CreateWriter();
  const auto values = RandomValues(12000, true);
  Insert(values);
  ASSERT_TRUE(column_->seal().ok());
  ASSERT_NE(column_->range_cache(), nullptr);
  const auto before = CacheContents();
  const auto bounds = NumericBounds();
  bool used_add = false;
  bool used_subtract = false;
  bool used_flip = false;
  for (int32_t bound = -5900; bound < 5900; bound += 100) {
    for (const auto op : kRangeOperations) {
      const auto plan = column_->explain_range(RawInt32(bound), op);
      used_add |= plan.path == RangeQueryPath::PrefixAdd;
      used_subtract |= plan.path == RangeQueryPath::PrefixSubtract;
      used_flip |= plan.flip;
    }
  }
  EXPECT_TRUE(used_add);
  EXPECT_TRUE(used_subtract);
  EXPECT_TRUE(used_flip);
  std::atomic<bool> correct{true};
  std::vector<std::thread> readers;
  for (size_t worker = 0; worker < 4; ++worker) {
    readers.emplace_back([&, worker]() {
      for (size_t i = worker; i < bounds.size(); i += 4) {
        for (const auto op : kRangeOperations) {
          auto result = column_->search(RawInt32(bounds[i]), op);
          if (!result) {
            correct = false;
            continue;
          }
          size_t count = 0;
          for (uint32_t id = 0; id < values.size(); ++id) {
            const bool expected =
                values[id] && Matches(*values[id], bounds[i], op);
            count += expected;
            if (result->contains(id) != expected) {
              correct = false;
            }
          }
          if (result->count() != count) {
            correct = false;
          }
        }
      }
    });
  }
  for (auto &reader : readers) {
    reader.join();
  }
  EXPECT_TRUE(correct.load());
  EXPECT_EQ(CacheContents(), before);
}

TEST_F(InvertedRangeCacheTest, InsufficientLocalBudgetFallsBack) {
  RangePrefixCacheOptions too_small;
  too_small.max_bytes = 1;
  CreateWriter(too_small);
  const auto values = RandomValues(12000, true);
  Insert(values);
  ASSERT_TRUE(column_->seal().ok());
  EXPECT_EQ(column_->range_cache(), nullptr);
  CheckNumeric(values, NumericBounds());
  Reopen(too_small);
  EXPECT_EQ(column_->range_cache(), nullptr);
  CheckNumeric(values, NumericBounds());
  Reopen();
  ASSERT_NE(column_->range_cache(), nullptr);
  CheckNumeric(values, NumericBounds());
}

TEST_F(InvertedRangeCacheTest,
       GlobalMemoryPressureFallsBackWithoutFailingOpen) {
  RangePrefixCacheOptions disabled;
  disabled.max_bytes = 0;
  CreateWriter(disabled);
  const auto values = RandomValues(12000, false);
  Insert(values);
  ASSERT_TRUE(column_->seal().ok());
  auto &pool = ailego::MemoryLimitPool::get_instance();
  const size_t reservation = pool.capacity() - pool.committed();
  ASSERT_GT(reservation, 0);
  ASSERT_TRUE(pool.try_charge_external(reservation));
  {
    struct ReleaseReservation {
      size_t bytes;
      ~ReleaseReservation() {
        ailego::MemoryLimitPool::get_instance().release_external(bytes);
      }
    } release{reservation};
    Reopen();
    EXPECT_EQ(column_->range_cache(), nullptr);
    CheckNumeric(values, {-10000, -3000, 0, 3000, 10000});
  }
  Reopen();
  EXPECT_NE(column_->range_cache(), nullptr);
}

TEST_F(InvertedRangeCacheTest, EmptyIndexUsesExistingSearchPath) {
  CreateWriter();
  ASSERT_TRUE(column_->seal().ok());
  EXPECT_EQ(column_->range_cache(), nullptr);
  CheckNumeric({}, {-1, 0, 1});
  Reopen();
  EXPECT_EQ(column_->range_cache(), nullptr);
  CheckNumeric({}, {-1, 0, 1});
}

TEST_F(InvertedRangeCacheTest, NullOnlyFieldNeverMatchesRangeQueries) {
  CreateWriter();
  const std::vector<std::optional<int32_t>> values(12000);
  Insert(values);
  ASSERT_TRUE(column_->seal().ok());
  EXPECT_EQ(column_->range_cache(), nullptr);
  CheckNumeric(values, {-1, 0, 1});
  Reopen();
  EXPECT_EQ(column_->range_cache(), nullptr);
  CheckNumeric(values, {-1, 0, 1});
}

TEST_F(InvertedRangeCacheTest, LowCardinalityWithoutRangeBuckets) {
  CreateWriter();
  std::vector<std::optional<int32_t>> values(12000);
  for (size_t i = 0; i < values.size(); ++i) {
    if (i % 19 != 0) {
      values[i] = static_cast<int32_t>(i % 32) - 16;
    }
  }
  Insert(values);
  ASSERT_TRUE(column_->seal().ok());
  EXPECT_EQ(column_->range_cache(), nullptr);
  EXPECT_TRUE(RangeRows().empty());
  CheckNumeric(values, {-100, -17, -16, -1, 0, 1, 15, 16, 100});
  Reopen();
  EXPECT_EQ(column_->range_cache(), nullptr);
  CheckNumeric(values, {-100, -17, -16, -1, 0, 1, 15, 16, 100});
}

TEST_F(InvertedRangeCacheTest, DisabledRangeIndexDoesNotBuildCache) {
  CreateWriter({}, DataType::INT32, false);
  const auto values = RandomValues(1200, true);
  Insert(values);
  ASSERT_TRUE(column_->seal().ok());
  EXPECT_EQ(column_->range_cache(), nullptr);
  CheckNumeric(values, {-1000, -300, 0, 300, 1000});
  Reopen();
  EXPECT_EQ(column_->range_cache(), nullptr);
  CheckNumeric(values, {-1000, -300, 0, 300, 1000});
}

TEST_F(InvertedRangeCacheTest, Uint64HighBitAndExtremeBoundaries) {
  std::vector<std::optional<uint64_t>> values(12000);
  const uint64_t stride = std::numeric_limits<uint64_t>::max() / values.size();
  for (size_t i = 0; i < values.size(); ++i) {
    if (i % 97 != 0) {
      values[i] = stride * i;
    }
  }
  const uint64_t high_bit = uint64_t{1} << 63;
  const std::vector<uint64_t> bounds{0,
                                     1,
                                     high_bit - 1,
                                     high_bit,
                                     high_bit + 1,
                                     std::numeric_limits<uint64_t>::max() - 1,
                                     std::numeric_limits<uint64_t>::max()};
  for (size_t i = 0; i < bounds.size(); ++i) {
    values[i + 1] = bounds[i];
  }
  CheckTypedSealAndReopen(DataType::UINT64, std::move(values), bounds);
}

TEST_F(InvertedRangeCacheTest, FloatRepresentableBoundaryNeighbors) {
  CheckFloatingPoint<float>(DataType::FLOAT);
}

TEST_F(InvertedRangeCacheTest, DoubleRepresentableBoundaryNeighbors) {
  CheckFloatingPoint<double>(DataType::DOUBLE);
}

TEST_F(InvertedRangeCacheTest, SparseValuesHighNullRateAndHotTerms) {
  for (const uint32_t seed : {1, 868, 2026}) {
    SCOPED_TRACE(testing::Message() << "seed=" << seed);
    CreateWriter();
    std::mt19937 random(seed);
    std::vector<std::optional<int32_t>> values(12000);
    for (auto &value : values) {
      if (random() % 10 < 7) {
        continue;
      }
      switch (random() % 4) {
        case 0:
          value = -1000000000;
          break;
        case 1:
          value = 1000000000;
          break;
        default:
          value = (static_cast<int32_t>(random() % 20001) - 10000) * 100000;
          break;
      }
    }
    Insert(values);
    ASSERT_TRUE(column_->seal().ok());
    ASSERT_NE(column_->range_cache(), nullptr);
    auto bounds = NumericBounds();
    bounds.insert(bounds.end(),
                  {-1000000001, -1000000000, -999999999, -500000001, -500000000,
                   -499999999, 499999999, 500000000, 500000001, 999999999,
                   1000000000, 1000000001});
    bool used_prefix = false;
    for (const auto bound : bounds) {
      for (const auto op : kRangeOperations) {
        const auto path = column_->explain_range(RawInt32(bound), op).path;
        used_prefix |= path == RangeQueryPath::PrefixAdd ||
                       path == RangeQueryPath::PrefixSubtract;
      }
    }
    ASSERT_TRUE(used_prefix);
    CheckNumeric(values, bounds);
    RangePrefixCacheOptions disabled;
    disabled.max_bytes = 0;
    Reopen(disabled);
    ASSERT_EQ(column_->range_cache(), nullptr);
    CheckNumeric(values, bounds);
    column_.reset();
    ASSERT_TRUE(context_.close().ok());
    test_util::RemoveTestPath(path_);
  }
}

TEST_F(InvertedRangeCacheTest, VariableLengthStringAndEmbeddedZeroBoundaries) {
  CreateWriter({}, DataType::STRING);
  std::vector<std::optional<std::string>> values(6000);
  for (size_t i = 0; i < values.size(); ++i) {
    if (i % 97 == 0) {
      continue;
    }
    values[i] = (i % 3 == 0 ? std::string("a\0", 2) : std::string("a")) +
                std::to_string(i);
  }
  for (uint32_t id = 0; id < values.size(); ++id) {
    ASSERT_TRUE(values[id] ? column_->insert(id, *values[id]).ok()
                           : column_->insert_null(id).ok());
  }
  ASSERT_TRUE(column_->seal().ok());
  ASSERT_NE(column_->range_cache(), nullptr);
  std::vector<std::string> bounds{"",
                                  "a",
                                  "a100",
                                  "a2000",
                                  "a5999",
                                  "b",
                                  std::string("a\0", 2),
                                  std::string("a\0", 2) + "3000"};
  for (const auto &prefix : column_->range_cache()->prefixes) {
    const auto &end =
        column_->range_cache()->buckets[prefix.end_bucket - 1].end;
    bounds.push_back(end);
    bounds.push_back(end + std::string(1, '\0'));
  }
  for (int pass = 0; pass < 2; ++pass) {
    for (const auto &bound : bounds) {
      for (const auto op : kRangeOperations) {
        auto result = column_->search(bound, op);
        ASSERT_NE(result, nullptr);
        size_t count = 0;
        for (uint32_t id = 0; id < values.size(); ++id) {
          const bool expected = values[id] && Matches(*values[id], bound, op);
          ASSERT_EQ(result->contains(id), expected) << "doc ID " << id;
          count += expected;
        }
        EXPECT_EQ(result->count(), count);
      }
    }
    if (pass == 0) {
      Reopen();
      ASSERT_NE(column_->range_cache(), nullptr);
    }
  }
}

TEST_F(InvertedRangeCacheTest, BinaryStringsPreserveUnsignedLexicalOrder) {
  CreateWriter({}, DataType::STRING);
  std::vector<std::optional<std::string>> values;
  values.reserve(12000);
  for (uint32_t i = 0; i < 4000; ++i) {
    std::string prefix;
    prefix.push_back(static_cast<char>(i >> 4));
    prefix.push_back(static_cast<char>((i & 15) << 4));
    // Adjacent values deliberately have a prefix relationship. Both high-bit
    // bytes and embedded zeroes participate in RocksDB's unsigned ordering.
    values.emplace_back(prefix);
    values.emplace_back(prefix + std::string(1, '\0'));
    values.emplace_back(prefix + std::string(1, static_cast<char>(0xff)));
  }
  std::mt19937 random(868);
  std::shuffle(values.begin(), values.end(), random);
  for (uint32_t id = 0; id < values.size(); ++id) {
    if (id % 97 == 0) {
      values[id].reset();
    }
    ASSERT_TRUE(values[id] ? column_->insert(id, *values[id]).ok()
                           : column_->insert_null(id).ok());
  }
  ASSERT_TRUE(column_->seal().ok());
  ASSERT_NE(column_->range_cache(), nullptr);
  std::vector<std::string> bounds{"", std::string(1, '\0'),
                                  std::string(1, static_cast<char>(0x7f)),
                                  std::string(1, static_cast<char>(0x80)),
                                  std::string(1, static_cast<char>(0xff))};
  for (const auto &prefix : column_->range_cache()->prefixes) {
    const auto &end =
        column_->range_cache()->buckets[prefix.end_bucket - 1].end;
    bounds.push_back(end);
    bounds.push_back(end + std::string(1, '\0'));
    bounds.push_back(end.substr(0, end.size() - 1));
  }
  bool used_prefix = false;
  for (const auto &bound : bounds) {
    for (const auto op : kRangeOperations) {
      const auto path = column_->explain_range(bound, op).path;
      used_prefix |= path == RangeQueryPath::PrefixAdd ||
                     path == RangeQueryPath::PrefixSubtract;
    }
  }
  ASSERT_TRUE(used_prefix);
  for (int pass = 0; pass < 2; ++pass) {
    for (const auto &bound : bounds) {
      for (const auto op : kRangeOperations) {
        auto result = column_->search(bound, op);
        ASSERT_NE(result, nullptr);
        size_t expected_count = 0;
        for (uint32_t id = 0; id < values.size(); ++id) {
          const bool expected = values[id] && Matches(*values[id], bound, op);
          ASSERT_EQ(result->contains(id), expected)
              << "doc ID " << id << ", op=" << static_cast<int>(op);
          expected_count += expected;
        }
        EXPECT_EQ(result->count(), expected_count);
      }
    }
    if (pass == 0) {
      Reopen();
      ASSERT_NE(column_->range_cache(), nullptr);
    }
  }
}

TEST_F(InvertedRangeCacheTest, SparseDocumentIdsDoNotCreatePhantomMatches) {
  CreateWriter();
  const auto values = RandomValues(24000, false);
  for (uint32_t i = 0; i < values.size(); ++i) {
    // These gaps are absent documents, not NULL-valued documents. A cache
    // shortcut must not turn them into matches by complementing [0,max_id].
    ASSERT_TRUE(column_->insert(2 * i, RawInt32(*values[i])).ok());
  }
  ASSERT_TRUE(column_->seal().ok());

  constexpr int32_t bound = 0;
  size_t expected_count = 0;
  for (const auto &value : values) {
    expected_count += *value > bound;
  }
  auto check_result = [&](const InvertedColumnIndexer::Ptr &indexer) {
    const auto plan = indexer->explain_range(RawInt32(bound), CompareOp::GT);
    SCOPED_TRACE(testing::Message() << "path=" << static_cast<int>(plan.path)
                                    << ", flip=" << plan.flip);
    auto result = indexer->search(RawInt32(bound), CompareOp::GT);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->count(), expected_count);
    for (uint32_t id = 0; id < 2 * values.size(); ++id) {
      const bool expected = id % 2 == 0 && *values[id / 2] > bound;
      ASSERT_EQ(result->contains(id), expected) << "doc ID " << id;
    }
  };

  RangePrefixCacheOptions disabled;
  disabled.max_bytes = 0;
  auto legacy = InvertedColumnIndexer::CreateAndOpen("range_cache_test", field_,
                                                     context_, true, disabled);
  ASSERT_NE(legacy, nullptr);
  EXPECT_FALSE(legacy->explain_range(RawInt32(bound), CompareOp::GT).flip);
  check_result(legacy);
  legacy.reset();

  // Correctness is required whether the cache is admitted or safely skipped.
  check_result(column_);
  Reopen();
  check_result(column_);
}

}  // namespace
}  // namespace zvec
