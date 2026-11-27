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
#include <atomic>
#include <cstring>
#include <optional>
#include <random>
#include <thread>
#include <gtest/gtest.h>
#include <rocksdb/utilities/stackable_db.h>
#include <ailego/pattern/defer.h>
#include "db/index/column/inverted_column/inverted_indexer.h"
#include "db/index/column/inverted_column/inverted_rocksdb_merger.h"
#include "tests/test_util.h"

namespace zvec {
namespace {

std::string Raw(int32_t value) {
  return std::string(reinterpret_cast<const char *>(&value), sizeof(value));
}
int32_t Decode(const std::string &value) {
  uint32_t bits = 0;
  for (unsigned char c : value) bits = (bits << 8) | c;
  bits ^= uint32_t{1} << 31;
  int32_t number;
  std::memcpy(&number, &bits, sizeof(number));
  return number;
}

class ReadCounter : public rocksdb::StackableDB {
 public:
  explicit ReadCounter(rocksdb::DB *db)
      : StackableDB(std::shared_ptr<rocksdb::DB>(db, [](rocksdb::DB *) {})) {}
  rocksdb::Status Get(const rocksdb::ReadOptions &options,
                      rocksdb::ColumnFamilyHandle *cf,
                      const rocksdb::Slice &key, std::string *value) override {
    ++reads;
    if (cf && cf->GetName() == "value$TERMS") ++term_gets;
    return StackableDB::Get(options, cf, key, value);
  }
  rocksdb::Iterator *NewIterator(const rocksdb::ReadOptions &options,
                                 rocksdb::ColumnFamilyHandle *cf) override {
    ++reads;
    if (cf && cf->GetName() == "value$TERMS") {
      ++term_iters;
      if (fail_next_term_iterator.exchange(false)) {
        return rocksdb::NewErrorIterator(
            rocksdb::Status::IOError("injected TERM iterator failure"));
      }
    }
    if (cf && cf->GetName() == "value$RANGES") ++range_iters;
    return StackableDB::NewIterator(options, cf);
  }
  rocksdb::Status Put(const rocksdb::WriteOptions &options,
                      rocksdb::ColumnFamilyHandle *cf,
                      const rocksdb::Slice &key,
                      const rocksdb::Slice &value) override {
    if (key == fail_put_key) {
      return rocksdb::Status::IOError("injected metadata write failure");
    }
    return StackableDB::Put(options, cf, key, value);
  }
  rocksdb::Status Write(const rocksdb::WriteOptions &options,
                        rocksdb::WriteBatch *batch) override {
    if (fail_next_write.exchange(false)) {
      return rocksdb::Status::IOError("injected write batch failure");
    }
    return StackableDB::Write(options, batch);
  }
  std::string fail_put_key;
  std::atomic<bool> fail_next_write{false};
  std::atomic<size_t> reads{0};
  std::atomic<size_t> term_gets{0};
  std::atomic<size_t> term_iters{0};
  std::atomic<size_t> range_iters{0};
  std::atomic<bool> fail_next_term_iterator{false};
};

class InvertedRangeCacheTest : public testing::Test {
 protected:
  void SetUp() override {
    path_ = std::string("./range_prefix_") +
            testing::UnitTest::GetInstance()->current_test_info()->name();
    test_util::RemoveTestPath(path_);
  }
  void TearDown() override {
    column_.reset();
    if (context_.db_) context_.close();
    test_util::RemoveTestPath(path_);
  }
  void create(DataType type = DataType::INT32) {
    field_ = FieldSchema("value", type, true,
                         std::make_shared<InvertIndexParams>(true, false));
    ASSERT_TRUE(context_
                    .create(path_, {"value$TERMS", "value$RANGES", INVERT_CDF},
                            std::make_shared<InvertedRocksdbValueMerger>())
                    .ok());
    column_ = InvertedColumnIndexer::CreateAndOpen("prefix_test", field_,
                                                   context_, false);
    ASSERT_NE(column_, nullptr);
  }
  void reopen(bool expect_rebuild = false) {
    column_.reset();
    ASSERT_TRUE(context_.flush().ok());
    ASSERT_TRUE(context_.close().ok());
    ASSERT_TRUE(
        context_
            .open(path_, true, std::make_shared<InvertedRocksdbValueMerger>())
            .ok());
    auto original = std::move(context_.db_);
    auto counter = std::make_unique<ReadCounter>(original.get());
    auto *reader = counter.get();
    context_.db_ = std::move(counter);
    AILEGO_DEFER([&]() { context_.db_ = std::move(original); });
    column_ = InvertedColumnIndexer::CreateAndOpen("prefix_test", field_,
                                                   context_, true);
    ASSERT_NE(column_, nullptr);
    if (expect_rebuild) {
      EXPECT_GT(reader->range_iters.load(), 0);
    } else {
      EXPECT_EQ(reader->range_iters.load(), 0);
      EXPECT_EQ(reader->term_iters.load(), 0);
    }
  }
  std::vector<std::optional<int32_t>> values(bool nulls = true,
                                            bool hot = false,
                                            int32_t count = 8000) {
    std::vector<std::optional<int32_t>> data;
    for (int32_t i = 0; i < count; ++i) {
      const auto value = hot && i > 1000 && i < 7000 ? 0 : i - count / 2;
      data.emplace_back(nulls && i % 97 == 0 ? std::nullopt
                                            : std::optional<int32_t>(value));
    }
    std::mt19937 random(881);
    std::shuffle(data.begin(), data.end(), random);
    for (uint32_t i = 0; i < data.size(); ++i)
      EXPECT_TRUE((data[i] ? column_->insert(i, Raw(*data[i]))
                           : column_->insert_null(i))
                      .ok());
    return data;
  }
  template <class T, class Encode>
  void verify(const std::vector<std::optional<T>> &data,
              const std::vector<T> &bounds, Encode encode) {
    for (const auto &bound : bounds)
      for (CompareOp op :
           {CompareOp::LT, CompareOp::LE, CompareOp::GT, CompareOp::GE}) {
        auto result = column_->search(encode(bound), op);
        ASSERT_NE(result, nullptr);
        auto it = result->create_iterator();
        size_t count = 0;
        for (uint32_t id = 0; id < data.size(); ++id) {
          if (!data[id]) continue;
          bool match = op == CompareOp::LT   ? *data[id] < bound
                       : op == CompareOp::LE ? *data[id] <= bound
                       : op == CompareOp::GT ? *data[id] > bound
                                             : *data[id] >= bound;
          if (!match) continue;
          ASSERT_TRUE(it->valid());
          ASSERT_EQ(it->doc_id(), id);
          it->next();
          ++count;
        }
        EXPECT_FALSE(it->valid());
        EXPECT_EQ(result->count(), count);
      }
  }
  std::string path_;
  FieldSchema field_;
  RocksdbContext context_;
  InvertedColumnIndexer::Ptr column_;
};

TEST_F(InvertedRangeCacheTest, PersistsSixteenthsAndLoadsThemWhenReopenedReadOnly) {
  create();
  auto data = values();
  ASSERT_EQ(column_->range_cache(), nullptr);
  ASSERT_TRUE(column_->seal().ok());
  const auto *cache = column_->range_cache();
  ASSERT_NE(cache, nullptr);
  ASSERT_EQ(cache->prefixes.size(), 15);
  std::vector<int32_t> bounds{-5000, -1, 0, 1, 5000};
  for (const auto &prefix : cache->prefixes) {
    const auto bound = Decode(cache->buckets[prefix.end_bucket - 1].end);
    bounds.insert(bounds.end(), {bound - 1, bound, bound + 1});
    EXPECT_EQ(roaring_bitmap_get_cardinality(prefix.bitmap.get()),
              cache->buckets[prefix.end_bucket - 1].cumulative_cardinality);
    for (uint32_t id = 0; id < data.size(); ++id)
      ASSERT_EQ(roaring_bitmap_contains(prefix.bitmap.get(), id),
                data[id].has_value() && *data[id] <= bound);
  }
  std::string value;
  ASSERT_TRUE(context_.db_->Get(context_.read_opts_,
                                 "value$ZVEC$RANGE_PREFIX$a", &value).ok());
  EXPECT_FALSE(value.empty());
  verify(data, bounds, Raw);
  reopen();
  ASSERT_NE(column_->range_cache(), nullptr);
  ASSERT_EQ(column_->range_cache()->prefixes.size(), 15);
  verify(data, bounds, Raw);
}

TEST_F(InvertedRangeCacheTest, LegacySealRebuildsOnReadOnlyOpen) {
  create();
  const auto data = values();
  ASSERT_TRUE(column_->seal().ok());
  rocksdb::WriteBatch batch;
  ASSERT_TRUE(InvertedRangeCache::Delete(batch, "value").ok());
  ASSERT_TRUE(batch.Put("value" + INVERT_KEY_SEALED, "sealed").ok());
  ASSERT_TRUE(context_.db_->Write(context_.write_opts_, &batch).ok());
  reopen(/*expect_rebuild=*/true);
  ASSERT_NE(column_->range_cache(), nullptr);
  verify(data, {-4001, -1, 0, 1, 4001}, Raw);
  std::string marker;
  ASSERT_TRUE(context_.db_->Get(context_.read_opts_, "value" + INVERT_KEY_SEALED,
                                 &marker).ok());
  EXPECT_EQ(marker, "sealed");
}

TEST_F(InvertedRangeCacheTest, InvalidPrefixFallsBackToRangeScans) {
  create();
  const auto data = values();
  ASSERT_TRUE(column_->seal().ok());
  ASSERT_NE(column_->range_cache(), nullptr);
  ASSERT_TRUE(context_.db_->Put(context_.write_opts_,
                                 "value$ZVEC$RANGE_PREFIX$b", "invalid").ok());
  reopen();
  EXPECT_EQ(column_->range_cache(), nullptr);
  verify(data, {-4001, -1, 0, 1, 4001}, Raw);
}

TEST_F(InvertedRangeCacheTest, FailedPrefixWriteSealsAndRebuildsOnReopen) {
  create();
  auto data = values();
  {
    auto original = std::move(context_.db_);
    auto writer = std::make_unique<ReadCounter>(original.get());
    writer->fail_put_key = "value$ZVEC$RANGE_PREFIX$b";
    context_.db_ = std::move(writer);
    AILEGO_DEFER([&]() { context_.db_ = std::move(original); });
    ASSERT_TRUE(column_->seal().ok());
  }

  std::string value;
  ASSERT_TRUE(
      context_.db_
          ->Get(context_.read_opts_, "value" + INVERT_KEY_SEALED, &value)
          .ok());
  EXPECT_EQ(value, "sealed");
  EXPECT_TRUE(
      context_.db_
          ->Get(context_.read_opts_, "value$ZVEC$RANGE_PREFIX$a", &value)
          .IsNotFound());
  EXPECT_TRUE(column_->is_sealed());
  ASSERT_NE(column_->range_cache(), nullptr);
  verify(data, {-1, 0, 42, 43}, Raw);
  reopen(/*expect_rebuild=*/true);
  ASSERT_NE(column_->range_cache(), nullptr);
  verify(data, {-1, 0, 42, 43}, Raw);
}

TEST_F(InvertedRangeCacheTest, FailedPrefixCleanupDoesNotPublishPartialCache) {
  create();
  auto data = values();
  {
    auto original = std::move(context_.db_);
    auto writer = std::make_unique<ReadCounter>(original.get());
    writer->fail_put_key = "value$ZVEC$RANGE_PREFIX$b";
    writer->fail_next_write = true;
    auto *probe = writer.get();
    context_.db_ = std::move(writer);
    AILEGO_DEFER([&]() { context_.db_ = std::move(original); });
    ASSERT_TRUE(column_->seal().ok());
    EXPECT_FALSE(probe->fail_next_write.load());
  }

  std::string value;
  ASSERT_TRUE(
      context_.db_
          ->Get(context_.read_opts_, "value" + INVERT_KEY_SEALED, &value)
          .ok());
  EXPECT_EQ(value, "sealed");
  EXPECT_TRUE(
      context_.db_
          ->Get(context_.read_opts_, "value$ZVEC$RANGE_PREFIX$a", &value)
          .ok());
  reopen(/*expect_rebuild=*/true);
  ASSERT_NE(column_->range_cache(), nullptr);
  verify(data, {-1, 0, 42, 43}, Raw);
}

TEST_F(InvertedRangeCacheTest, FailedSealMarkerDoesNotPublishAndCanBeRetried) {
  create();
  auto data = values();
  {
    auto original = std::move(context_.db_);
    auto writer = std::make_unique<ReadCounter>(original.get());
    writer->fail_put_key = "value" + INVERT_KEY_SEALED;
    context_.db_ = std::move(writer);
    AILEGO_DEFER([&]() { context_.db_ = std::move(original); });
    EXPECT_FALSE(column_->seal().ok());
  }

  std::string value;
  EXPECT_TRUE(
      context_.db_
          ->Get(context_.read_opts_, "value" + INVERT_KEY_SEALED, &value)
          .IsNotFound());
  ASSERT_TRUE(
      context_.db_
          ->Get(context_.read_opts_, "value$ZVEC$RANGE_PREFIX$a", &value)
          .ok());
  EXPECT_FALSE(column_->is_sealed());
  EXPECT_EQ(column_->range_cache(), nullptr);
  column_.reset();
  column_ = InvertedColumnIndexer::CreateAndOpen("prefix_test", field_,
                                                 context_, false);
  ASSERT_NE(column_, nullptr);
  EXPECT_FALSE(column_->is_sealed());
  EXPECT_EQ(column_->range_cache(), nullptr);
  ASSERT_TRUE(column_->insert(data.size(), Raw(42)).ok());
  data.emplace_back(42);
  ASSERT_TRUE(column_->seal().ok());
  reopen();
  ASSERT_NE(column_->range_cache(), nullptr);
  verify(data, {-1, 0, 42, 43}, Raw);
}

TEST_F(InvertedRangeCacheTest, SealedNullResultsSurviveCloseAndDetachOnWrite) {
  create();
  ASSERT_TRUE(column_->insert_null(0).ok());
  ASSERT_TRUE(column_->insert(1, Raw(1)).ok());
  auto unsealed = column_->search_null();
  ASSERT_NE(unsealed, nullptr);
  ASSERT_TRUE(column_->insert_null(2).ok());
  EXPECT_EQ(unsealed->count(), 1);
  ASSERT_TRUE(column_->seal().ok());
  auto original = column_->search_null();
  auto intersection = column_->search_null();
  auto combination = column_->search_null();
  auto non_null = column_->search_non_null();
  ASSERT_NE(original, nullptr);
  ASSERT_NE(intersection, nullptr);
  ASSERT_NE(combination, nullptr);
  ASSERT_NE(non_null, nullptr);
  auto iterator = original->create_iterator();
  column_.reset();
  ASSERT_TRUE(context_.close().ok());
  EXPECT_EQ(original->count(), 2);
  EXPECT_TRUE(original->contains(0));
  EXPECT_TRUE(original->contains(2));

  intersection->AND(*non_null);
  combination->OR(*non_null);
  EXPECT_EQ(intersection->count(), 0);
  EXPECT_EQ(combination->count(), 3);
  std::vector<uint32_t> ids;
  original->extract_ids(&ids);
  EXPECT_EQ(ids, (std::vector<uint32_t>{0, 2}));
  ASSERT_TRUE(iterator->valid());
  EXPECT_EQ(iterator->doc_id(), 0);
  iterator->next();
  ASSERT_TRUE(iterator->valid());
  EXPECT_EQ(iterator->doc_id(), 2);
}

TEST_F(InvertedRangeCacheTest, ExactPrefixQueriesAvoidRocksdbReads) {
  create();
  auto data = values(false);
  ASSERT_TRUE(column_->seal().ok());
  reopen();
  const auto *cache = column_->range_cache();
  ASSERT_NE(cache, nullptr);
  const auto &prefix = cache->prefixes[3];
  const auto bound = Decode(cache->buckets[prefix.end_bucket - 1].end);
  const auto plan = column_->explain_range(Raw(bound), CompareOp::LE);
  ASSERT_EQ(plan.path, RangeQueryPath::PrefixAdd);
  ASSERT_EQ(plan.range_begin, plan.range_end);
  ASSERT_EQ(plan.boundary_bucket, SIZE_MAX);
  const auto count = roaring_bitmap_get_cardinality(prefix.bitmap.get());
  auto original = std::move(context_.db_);
  context_.db_ = std::make_unique<ReadCounter>(original.get());
  auto *counter = static_cast<ReadCounter *>(context_.db_.get());
  for (unsigned i = 0; i < 10; ++i) {
    auto result = column_->search(Raw(bound), CompareOp::LE);
    EXPECT_NE(result, nullptr);
    if (result) {
      EXPECT_EQ(result->count(), count);
    }
    auto complement = column_->search(Raw(bound), CompareOp::GT);
    EXPECT_NE(complement, nullptr);
    if (complement) {
      EXPECT_EQ(complement->count(), data.size() - count);
    }
  }
  const auto reads = counter->reads.load();
  context_.db_ = std::move(original);
  EXPECT_EQ(reads, 0);
  EXPECT_EQ(roaring_bitmap_get_cardinality(prefix.bitmap.get()), count);
  verify(data, {bound - 1, bound, bound + 1}, Raw);
}

TEST_F(InvertedRangeCacheTest, ExactBucketEndpointsUseOneTermGet) {
  create();
  auto data = values();
  ASSERT_TRUE(column_->seal().ok());
  ASSERT_NE(column_->search(Raw(0), CompareOp::LE), nullptr);
  const auto *cache = column_->range_cache();
  ASSERT_NE(cache, nullptr);
  const size_t end = cache->prefixes[3].end_bucket;
  ASSERT_LT(end, cache->buckets.size());
  const auto upper = Decode(cache->buckets[end - 1].end);
  const auto lower = Decode(cache->buckets[end].begin);
  const auto lt = column_->explain_range(Raw(upper), CompareOp::LT);
  const auto le = column_->explain_range(Raw(lower), CompareOp::LE);
  EXPECT_EQ(lt.path, RangeQueryPath::PrefixSubtract);
  EXPECT_EQ(le.path, RangeQueryPath::PrefixAdd);
  EXPECT_EQ(lt.range_begin, lt.range_end);
  EXPECT_EQ(le.range_begin, le.range_end);
  auto original = std::move(context_.db_);
  context_.db_ = std::make_unique<ReadCounter>(original.get());
  auto *counter = static_cast<ReadCounter *>(context_.db_.get());
  auto left = column_->search(Raw(upper), CompareOp::LT);
  auto right = column_->search(Raw(lower), CompareOp::LE);
  EXPECT_NE(left, nullptr);
  EXPECT_NE(right, nullptr);
  EXPECT_EQ(counter->term_gets.load(), 2);
  EXPECT_EQ(counter->term_iters.load(), 0);
  EXPECT_EQ(counter->range_iters.load(), 0);
  context_.db_ = std::move(original);
  verify(data, {upper - 1, upper, upper + 1, lower - 1, lower, lower + 1}, Raw);
}

TEST_F(InvertedRangeCacheTest, PlannerReducesOwnedRangeDecoding) {
  create();
  auto data = values(false);
  ASSERT_TRUE(column_->seal().ok());
  ASSERT_NE(column_->search(Raw(0), CompareOp::LE), nullptr);
  const auto *cache = column_->range_cache();
  ASSERT_NE(cache, nullptr);
  const size_t boundary = cache->buckets.size() * 9 / 10;
  ASSERT_GT(boundary, 0);
  const auto bound = Decode(cache->buckets[boundary - 1].end);
  const auto plan = column_->explain_range(Raw(bound), CompareOp::GT);
  ASSERT_NE(plan.prefix_index, SIZE_MAX);
  EXPECT_LT(plan.range_end - plan.range_begin,
            cache->buckets.size() - boundary);
  verify(data, {bound - 1, bound, bound + 1}, Raw);
}

TEST_F(InvertedRangeCacheTest, SubtractsFullBucketsAndRestoresBoundaryTerms) {
  create();
  auto data = values(true, false, 2 << 16);
  ASSERT_TRUE(column_->seal().ok());
  ASSERT_NE(column_->search(Raw(0), CompareOp::LE), nullptr);
  const auto *cache = column_->range_cache();
  ASSERT_NE(cache, nullptr);
  ASSERT_GE(cache->prefixes.size(), 2);
  const size_t end = cache->prefixes[1].end_bucket;
  ASSERT_GT(end, cache->prefixes[0].end_bucket + 16);
  const size_t boundary = end - 8;
  const auto &bucket = cache->buckets[boundary];
  const auto first = Decode(bucket.begin);
  const auto last = Decode(bucket.end);
  const auto middle = first + (last - first) / 2;
  ASSERT_LT(first, middle);
  ASSERT_LT(middle, last);

  const auto at_begin = column_->explain_range(Raw(first), CompareOp::GT);
  const auto in_bucket = column_->explain_range(Raw(middle), CompareOp::LT);
  for (const auto &plan : {at_begin, in_bucket}) {
    EXPECT_EQ(plan.path, RangeQueryPath::PrefixSubtractAddBoundary);
    EXPECT_EQ(plan.boundary_bucket, boundary);
    EXPECT_EQ(plan.range_begin, boundary);
    EXPECT_EQ(plan.range_end, end);
  }
  EXPECT_TRUE(at_begin.flip);
  EXPECT_FALSE(in_bucket.flip);
  verify(data, {first, middle, last}, Raw);
  reopen();
  verify(data, {first, middle, last}, Raw);
}

TEST_F(InvertedRangeCacheTest, MixedPrefixKeepsAdditiveBoundaryPlan) {
  create();
  constexpr int32_t kFirstBlock = 1 << 16;
  constexpr int32_t kCount = kFirstBlock + (1 << 14);
  values(true, false, kCount);
  ASSERT_TRUE(column_->seal().ok());
  ASSERT_NE(column_->search(Raw(0), CompareOp::LE), nullptr);
  const auto *cache = column_->range_cache();
  ASSERT_NE(cache, nullptr);
  ASSERT_GE(cache->prefixes.size(), 2);
  const auto &prefix = cache->prefixes[1];
  ASSERT_GT(prefix.end_bucket, cache->prefixes[0].end_bucket + 16);
  EXPECT_GT(roaring_bitmap_range_cardinality(prefix.bitmap.get(), 0,
                                              kFirstBlock),
            4096);
  const auto last_block =
      roaring_bitmap_range_cardinality(prefix.bitmap.get(), kFirstBlock, kCount);
  ASSERT_GT(last_block, 0);
  ASSERT_LE(last_block, 4096);

  const auto first = Decode(cache->buckets[prefix.end_bucket - 8].begin);
  const auto plan = column_->explain_range(Raw(first), CompareOp::GT);
  EXPECT_EQ(plan.path, RangeQueryPath::PrefixAdd);
}

TEST_F(InvertedRangeCacheTest, NearPrefixSubtractUsesBoundedTermProbe) {
  create();
  std::vector<std::optional<int32_t>> data;
  for (int32_t i = 0; i < 160000; ++i) data.emplace_back(i);
  std::mt19937 random(881);
  std::shuffle(data.begin(), data.end(), random);
  for (uint32_t id = 0; id < data.size(); ++id) {
    ASSERT_TRUE(column_->insert(id, Raw(*data[id])).ok());
  }
  ASSERT_TRUE(column_->seal().ok());
  ASSERT_NE(column_->search(Raw(0), CompareOp::LE), nullptr);
  const auto *cache = column_->range_cache();
  ASSERT_NE(cache, nullptr);
  ASSERT_GT(cache->prefixes.size(), 3);
  const auto &bucket = cache->buckets[cache->prefixes[3].end_bucket - 1];
  const auto near_end = Decode(bucket.end) - 2;
  const auto near_begin = Decode(bucket.begin) + 1;
  const auto fast = column_->explain_range(Raw(near_end), CompareOp::GT);
  ASSERT_EQ(fast.path, RangeQueryPath::PrefixAdd);
  ASSERT_EQ(fast.boundary_subtract_prefix, 3);
  ASSERT_GT(fast.boundary_subtract_budget, 0);
  const auto fallback = column_->explain_range(Raw(near_begin), CompareOp::GT);
  ASSERT_EQ(fallback.path, RangeQueryPath::PrefixSubtractAddBoundary);
  ASSERT_EQ(fallback.boundary_subtract_prefix, SIZE_MAX);

  auto original = std::move(context_.db_);
  context_.db_ = std::make_unique<ReadCounter>(original.get());
  auto *counter = static_cast<ReadCounter *>(context_.db_.get());
  auto result = column_->search(Raw(near_end), CompareOp::GT);
  EXPECT_NE(result, nullptr);
  EXPECT_EQ(counter->term_iters.load(), 1);
  EXPECT_EQ(counter->range_iters.load(), 0);

  result = column_->search(Raw(near_begin), CompareOp::GT);
  EXPECT_NE(result, nullptr);
  // The larger suffix is excluded before opening a second TERM iterator.
  EXPECT_EQ(counter->term_iters.load(), 2);
  EXPECT_EQ(counter->range_iters.load(), 1);

  counter->fail_next_term_iterator = true;
  result = column_->search(Raw(near_end), CompareOp::GT);
  EXPECT_NE(result, nullptr);
  if (result) {
    EXPECT_EQ(result->count(), data.size() - near_end - 1);
  }
  EXPECT_EQ(counter->term_iters.load(), 4);
  EXPECT_EQ(counter->range_iters.load(), 2);
  context_.db_ = std::move(original);

  verify(data, {near_end, near_begin}, Raw);
}

TEST_F(InvertedRangeCacheTest, RepeatedValuesAndNullsKeepUniqueBoundaries) {
  create();
  auto data = values(true, true);
  ASSERT_TRUE(column_->seal().ok());
  ASSERT_NE(column_->search(Raw(0), CompareOp::LE), nullptr);
  const auto *cache = column_->range_cache();
  ASSERT_NE(cache, nullptr);
  EXPECT_LT(cache->prefixes.size(), 15);
  size_t previous = 0;
  for (const auto &prefix : cache->prefixes) {
    EXPECT_GT(prefix.end_bucket, previous);
    previous = prefix.end_bucket;
  }
  verify(data, {-5000, -3000, -1, 0, 1, 3000, 5000}, Raw);
  reopen();
  verify(data, {-1, 0, 1}, Raw);
}

TEST_F(InvertedRangeCacheTest, ConcurrentQueriesDoNotMutateSharedPrefixes) {
  create();
  auto data = values();
  ASSERT_TRUE(column_->seal().ok());
  ASSERT_NE(column_->search(Raw(0), CompareOp::LE), nullptr);
  const auto *cache = column_->range_cache();
  ASSERT_NE(cache, nullptr);
  std::vector<std::string> snapshots;
  for (const auto &prefix : cache->prefixes) {
    std::string value;
    ASSERT_TRUE(
        InvertedIndexCodec::Serialize(prefix.bitmap.get(), &value).ok());
    snapshots.push_back(std::move(value));
  }
  std::vector<std::thread> threads;
  for (unsigned i = 0; i < 8; ++i)
    threads.emplace_back([&, i] {
      verify(data, {-3001 + int(i), -1 + int(i), 2001 + int(i)}, Raw);
    });
  for (auto &thread : threads) thread.join();
  for (size_t i = 0; i < snapshots.size(); ++i) {
    std::string value;
    ASSERT_TRUE(
        InvertedIndexCodec::Serialize(cache->prefixes[i].bitmap.get(), &value)
            .ok());
    EXPECT_EQ(value, snapshots[i]);
  }
}

TEST_F(InvertedRangeCacheTest, BinaryStringsKeepRangeSemantics) {
  create(DataType::STRING);
  std::vector<std::optional<std::string>> data;
  for (size_t i = 0; i < 8000; ++i)
    data.emplace_back(std::string(1, char(i % 256)) + std::to_string(i) +
                      std::string("\0end", 4));
  std::mt19937 random(881);
  std::shuffle(data.begin(), data.end(), random);
  for (uint32_t i = 0; i < data.size(); ++i)
    ASSERT_TRUE(column_->insert(i, *data[i]).ok());
  ASSERT_TRUE(column_->seal().ok());
  ASSERT_NE(column_->search("", CompareOp::LE), nullptr);
  ASSERT_NE(column_->range_cache(), nullptr);
  std::vector<std::string> bounds{"", std::string(1, '\0'),
                                  std::string(1, char(255)), *data[23],
                                  *data[5000]};
  auto encode = [](const std::string &value) { return value; };
  verify(data, bounds, encode);
  reopen();
  verify(data, bounds, encode);
}

TEST_F(InvertedRangeCacheTest, SparseDocidsDoNotEnableNewComplementPlans) {
  create();
  for (int32_t id = 0; id < 8000; id += 2)
    ASSERT_TRUE(column_->insert(id, Raw(id)).ok());
  ASSERT_TRUE(column_->seal().ok());
  EXPECT_EQ(column_->range_cache(), nullptr);
  auto result = column_->search(Raw(1000), CompareOp::LT);
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->count(), 500);
  EXPECT_EQ(column_->range_cache(), nullptr);
}

TEST_F(InvertedRangeCacheTest, AllNullValuesUseLegacySearch) {
  create();
  std::vector<std::optional<int32_t>> data(8000);
  for (uint32_t id = 0; id < data.size(); ++id)
    ASSERT_TRUE(column_->insert_null(id).ok());
  ASSERT_TRUE(column_->seal().ok());
  EXPECT_EQ(column_->range_cache(), nullptr);
  verify(data, {-1, 0, 1}, Raw);
  EXPECT_EQ(column_->range_cache(), nullptr);
  reopen();
  verify(data, {-1, 0, 1}, Raw);
}

TEST_F(InvertedRangeCacheTest, RecreatedFieldDoesNotReuseOldCache) {
  FieldSchema field("value", DataType::INT32, true,
                    std::make_shared<InvertIndexParams>(true, false));
  auto index = InvertedIndexer::CreateAndOpen("prefix_test", path_, true,
                                              {field}, false);
  ASSERT_NE(index, nullptr);
  auto column = (*index)["value"];
  for (int32_t i = 0; i < 8000; ++i)
    ASSERT_TRUE(column->insert(i, Raw(i)).ok());
  ASSERT_TRUE(index->seal().ok());
  ASSERT_NE(column->search(Raw(0), CompareOp::LE), nullptr);
  ASSERT_NE(column->range_cache(), nullptr);
  column.reset();
  ASSERT_TRUE(index->remove_column_indexer("value").ok());
  ASSERT_TRUE(index->create_column_indexer(field).ok());
  column = (*index)["value"];
  ASSERT_EQ(column->range_cache(), nullptr);
  ASSERT_TRUE(column->insert(0, Raw(7)).ok());
  ASSERT_TRUE(index->seal().ok());
  EXPECT_EQ(column->range_cache(), nullptr);
  auto result = column->search(Raw(100), CompareOp::LT);
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->count(), 1);
}

TEST_F(InvertedRangeCacheTest, UnsupportedBucketOrderKeepsSealCompatible) {
  create(DataType::STRING);
  for (uint32_t id = 0; id < 8000; ++id) {
    const auto value =
        id < 16 ? "a" + std::string(id, '\0') : "z" + std::to_string(id);
    ASSERT_TRUE(column_->insert(id, value).ok());
  }
  ASSERT_TRUE(column_->seal().ok());
  EXPECT_EQ(column_->range_cache(), nullptr);
  auto range = column_->search("z", CompareOp::LT);
  ASSERT_NE(range, nullptr);
  EXPECT_EQ(range->count(), 16);
  EXPECT_EQ(column_->range_cache(), nullptr);
  reopen();
  auto result = column_->search("a", CompareOp::EQ);
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->count(), 1);
}

}  // namespace
}  // namespace zvec
