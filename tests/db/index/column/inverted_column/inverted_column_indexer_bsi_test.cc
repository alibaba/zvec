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

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <optional>
#include <random>
#include <thread>
#include <type_traits>
#include <ailego/pattern/defer.h>
#include <gtest/gtest.h>
#include <rocksdb/utilities/stackable_db.h>
#include "db/index/column/inverted_column/inverted_indexer.h"
#include "db/index/column/inverted_column/inverted_rocksdb_merger.h"

using namespace zvec;

namespace {

template <typename T>
std::string EncodeValue(T value) {
  return std::string(reinterpret_cast<const char *>(&value), sizeof(value));
}

class BitSliceIndexTest : public testing::TestWithParam<BsiCacheMode> {
 protected:
  void SetUp() override {
    auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path() /
            ("zvec_bsi_" + std::to_string(suffix));
  }

  void TearDown() override {
    std::filesystem::remove_all(path_);
  }

  InvertIndexParams::OPtr Params() {
    return std::make_shared<InvertIndexParams>(InvertIndexLayout::TERM_BSI,
                                               false, GetParam(), 256 * 1024);
  }

  InvertedIndexer::Ptr Open(DataType type, bool read_only = false) {
    FieldSchema field("value", type, true, Params());
    return InvertedIndexer::CreateAndOpen("bsi", path_.string(), true, {field},
                                          read_only);
  }

  template <typename T>
  void Verify(const InvertedColumnIndexer::Ptr &index,
              const std::vector<std::optional<T>> &values,
              const std::vector<T> &bounds) {
    for (T bound : bounds) {
      for (CompareOp op :
           {CompareOp::LT, CompareOp::LE, CompareOp::GT, CompareOp::GE}) {
        auto result = index->search_filter(EncodeValue(bound), op);
        ASSERT_NE(result, nullptr);
        size_t expected_count = 0;
        for (size_t id = 0; id < values.size(); ++id) {
          bool expected = false;
          if (values[id]) {
            T value = *values[id];
            switch (op) {
              case CompareOp::EQ:
                expected = value == bound;
                break;
              case CompareOp::NE:
                expected = value != bound;
                break;
              case CompareOp::LT:
                expected = value < bound;
                break;
              case CompareOp::LE:
                expected = value <= bound;
                break;
              case CompareOp::GT:
                expected = value > bound;
                break;
              case CompareOp::GE:
                expected = value >= bound;
                break;
              default:
                break;
            }
          }
          EXPECT_EQ(result->contains(id), expected) << "docID=" << id;
          expected_count += expected;
        }
        EXPECT_EQ(result->count(), expected_count);
      }
    }
    auto nulls = index->search_null();
    auto non_nulls = index->search_non_null();
    ASSERT_NE(nulls, nullptr);
    ASSERT_NE(non_nulls, nullptr);
    for (size_t id = 0; id < values.size(); ++id) {
      EXPECT_EQ(nulls->contains(id), !values[id].has_value());
      EXPECT_EQ(non_nulls->contains(id), values[id].has_value());
    }
  }

  template <typename T>
  void CheckType(DataType type) {
    std::vector<T> bounds{std::numeric_limits<T>::min(),
                          std::numeric_limits<T>::max(), T{0}, T{1}, T{7}};
    if constexpr (std::is_signed_v<T>) {
      bounds.push_back(T{-1});
      bounds.push_back(T{-7});
    }
    std::mt19937_64 random(19);
    for (int i = 0; i < 24; ++i) {
      bounds.push_back(static_cast<T>(random()));
    }
    std::vector<std::optional<T>> values{std::nullopt, T{0}};
    for (T value : bounds) {
      values.push_back(value);
      values.push_back(value);
    }
    values.push_back(std::nullopt);
    auto owner = Open(type);
    ASSERT_NE(owner, nullptr);
    auto index = (*owner)["value"];
    for (uint32_t id = 0; id < values.size(); ++id) {
      auto status = values[id] ? index->insert(id, EncodeValue(*values[id]))
                               : index->insert_null(id);
      ASSERT_TRUE(status.ok()) << status.message();
    }
    Verify(index, values, bounds);
    ASSERT_TRUE(owner->flush().ok());
    index.reset();
    owner.reset();
    owner = Open(type);
    ASSERT_NE(owner, nullptr);
    index = (*owner)["value"];
    Verify(index, values, bounds);
    ASSERT_TRUE(owner->seal().ok());
    Verify(index, values, bounds);
    index.reset();
    owner.reset();
    owner = Open(type, true);
    ASSERT_NE(owner, nullptr);
    Verify((*owner)["value"], values, bounds);
  }

  std::filesystem::path path_;
};

TEST_P(BitSliceIndexTest, FrozenViewContainersAndResultLifetime) {
  for (bool fourth_container : {false, true}) {
    roaring::Roaring original;
    for (uint32_t id : {1u, 9u, 37u, 91u}) original.add(id);
    for (uint32_t id = 65536; id < 131072; id += 2) original.add(id);
    original.addRange(131172, 141072);
    if (fourth_container) original.add(196609);
    original.runOptimize();
    std::string buffer;
    ASSERT_TRUE(InvertedIndexCodec::Serialize(&original.roaring, &buffer).ok());
    const std::string before = buffer;
    roaring_bitmap_t *raw = nullptr;
    ASSERT_TRUE(InvertedIndexCodec::DeserializeFrozen(buffer.data(),
                                                      buffer.size(), &raw)
                    .ok());
    std::unique_ptr<roaring_bitmap_t, decltype(&roaring_bitmap_free)> view(
        raw, roaring_bitmap_free);
    ASSERT_NE(view, nullptr);
    EXPECT_NE(view->high_low_container.flags & ROARING_FLAG_FROZEN, 0);
    EXPECT_TRUE(roaring_bitmap_equals(view.get(), &original.roaring));
    roaring::Roaring mask;
    mask.addRange(0, 65536);
    auto *difference = roaring_bitmap_andnot(view.get(), &mask.roaring);
    ASSERT_NE(difference, nullptr);
    roaring::Roaring result(difference);
    EXPECT_EQ(buffer, before);
    view.reset();
    std::string{}.swap(buffer);
    EXPECT_EQ(result.cardinality(), original.cardinality() - 4);
    EXPECT_FALSE(result.contains(9));
    EXPECT_TRUE(result.contains(65538));
    EXPECT_TRUE(result.contains(141071));
    EXPECT_EQ(result.contains(196609), fourth_container);
  }
}

TEST_P(BitSliceIndexTest, FrozenViewSmallDocIdList) {
  roaring::Roaring original;
  original.add(0);
  original.add(65536);
  original.add(std::numeric_limits<uint32_t>::max());
  std::string buffer;
  ASSERT_TRUE(InvertedIndexCodec::Serialize(&original.roaring, &buffer).ok());
  ASSERT_NE(buffer[0], 0);
  roaring_bitmap_t *raw = nullptr;
  ASSERT_TRUE(
      InvertedIndexCodec::DeserializeFrozen(buffer.data(), buffer.size(), &raw)
          .ok());
  std::unique_ptr<roaring_bitmap_t, decltype(&roaring_bitmap_free)> view(
      raw, roaring_bitmap_free);
  ASSERT_NE(view, nullptr);
  EXPECT_TRUE(roaring_bitmap_equals(view.get(), &original.roaring));
}

TEST_P(BitSliceIndexTest, FrozenViewRejectsTruncatedValues) {
  roaring::Roaring original;
  for (uint32_t id = 0; id < 10000; id += 2) original.add(id);
  std::string buffer;
  ASSERT_TRUE(InvertedIndexCodec::Serialize(&original.roaring, &buffer).ok());
  for (size_t size : {size_t{0}, size_t{1}, size_t{2}, size_t{8},
                      buffer.size() / 2, buffer.size() - 1}) {
    roaring_bitmap_t *raw = nullptr;
    EXPECT_FALSE(
        InvertedIndexCodec::DeserializeFrozen(buffer.data(), size, &raw).ok());
    EXPECT_EQ(raw, nullptr);
  }
}

TEST_P(BitSliceIndexTest, Int32) {
  CheckType<int32_t>(DataType::INT32);
}
TEST_P(BitSliceIndexTest, Int64) {
  CheckType<int64_t>(DataType::INT64);
}
TEST_P(BitSliceIndexTest, Uint32) {
  CheckType<uint32_t>(DataType::UINT32);
}
TEST_P(BitSliceIndexTest, Uint64) {
  CheckType<uint64_t>(DataType::UINT64);
}

TEST_P(BitSliceIndexTest, EmptyAndAllNull) {
  auto owner = Open(DataType::INT64);
  ASSERT_NE(owner, nullptr);
  auto index = (*owner)["value"];
  Verify<int64_t>(index, {}, {0, 1, -1});
  for (uint32_t id = 0; id < 8; ++id) {
    ASSERT_TRUE(index->insert_null(id).ok());
  }
  Verify<int64_t>(index, std::vector<std::optional<int64_t>>(8), {0, 1, -1});
  ASSERT_TRUE(owner->seal().ok());
  Verify<int64_t>(index, std::vector<std::optional<int64_t>>(8), {0, 1, -1});
}

TEST_P(BitSliceIndexTest, EmptySealed) {
  auto owner = Open(DataType::INT64);
  ASSERT_NE(owner, nullptr);
  ASSERT_TRUE(owner->seal().ok());
  Verify<int64_t>((*owner)["value"], {}, {0, 1, -1});
}

TEST_P(BitSliceIndexTest, OutOfOrderAndIdempotentReplay) {
  auto owner = Open(DataType::INT32);
  ASSERT_NE(owner, nullptr);
  auto index = (*owner)["value"];
  ASSERT_TRUE(index->insert(2, EncodeValue(int32_t{7})).ok());
  // Term recall enumerates only postings; BSI filters may include holes.
  EXPECT_EQ(index->search(EncodeValue(int32_t{7}), CompareOp::EQ)->count(), 1u);
  auto filter = index->search_filter(
      EncodeValue(std::numeric_limits<int32_t>::min()), CompareOp::GE);
  ASSERT_NE(filter, nullptr);
  EXPECT_TRUE(filter->contains(1));
  EXPECT_TRUE(filter->contains(2));
  auto recall = index->search(EncodeValue(std::numeric_limits<int32_t>::min()),
                              CompareOp::GE);
  ASSERT_NE(recall, nullptr);
  EXPECT_FALSE(recall->contains(1));
  EXPECT_TRUE(recall->contains(2));
  ASSERT_TRUE(index->insert_null(0).ok());
  ASSERT_TRUE(owner->flush().ok());
  index.reset();
  owner.reset();
  owner = Open(DataType::INT32);
  ASSERT_NE(owner, nullptr);
  index = (*owner)["value"];
  ASSERT_TRUE(index->insert(1, EncodeValue(int32_t{0})).ok());
  ASSERT_TRUE(index->insert(2, EncodeValue(int32_t{7})).ok());
  ASSERT_TRUE(index->insert_null(0).ok());
  Verify<int32_t>(index, {std::nullopt, 0, 7}, {0, 1, 7, 8});
  ASSERT_TRUE(owner->seal().ok());
}

TEST_P(BitSliceIndexTest, ReplayRepairsSlicesWhenTermAlreadyExists) {
  RocksdbContext context;
  ASSERT_TRUE(context
                  .create(path_.string(),
                          {"value" + INVERT_SUFFIX_TERMS,
                           "value" + INVERT_SUFFIX_BSI, INVERT_CDF},
                          std::make_shared<InvertedRocksdbValueMerger>())
                  .ok());
  FieldSchema field("value", DataType::UINT32, true, Params());
  auto index = InvertedColumnIndexer::CreateAndOpen("replay", field, context);
  ASSERT_NE(index, nullptr);
  const auto value = EncodeValue(uint32_t{7});
  ASSERT_TRUE(index->insert(0, value).ok());
  std::string persisted;
  auto *slice_cf = context.get_cf("value" + INVERT_SUFFIX_BSI);
  EXPECT_TRUE(context.db_
                  ->Get(context.read_opts_, slice_cf, std::string(1, char{31}),
                        &persisted)
                  .IsNotFound());
  EXPECT_EQ(
      index->search_filter(EncodeValue(uint32_t{0}), CompareOp::GT)->count(),
      1u);
  ASSERT_TRUE(index->flush_special_values().ok());
  ASSERT_TRUE(context.flush().ok());
  EXPECT_TRUE(context.db_
                  ->Get(context.read_opts_, slice_cf, std::string(1, char{31}),
                        &persisted)
                  .ok());
  // Model a surviving term CF and a lost BSI CF before WAL replay.
  auto *slices = context.get_cf("value" + INVERT_SUFFIX_BSI);
  for (uint32_t bit = 0; bit < 32; ++bit) {
    ASSERT_TRUE(context.db_
                    ->Delete(context.write_opts_, slices,
                             std::string(1, static_cast<char>(bit)))
                    .ok());
  }
  // Reopen the column to discard its in-memory state, as crash recovery does.
  index.reset();
  index = InvertedColumnIndexer::CreateAndOpen("replay", field, context);
  ASSERT_NE(index, nullptr);
  EXPECT_EQ(index->search(value, CompareOp::EQ)->count(), 1u);
  EXPECT_EQ(
      index->search_filter(EncodeValue(uint32_t{0}), CompareOp::GT)->count(),
      0u);
  ASSERT_TRUE(index->insert(0, value).ok());
  Verify<uint32_t>(index, {7}, {0, 6, 7, 8});

  ASSERT_TRUE(index->flush_special_values().ok());
  // The reverse partial-flush order must also be repaired, even when the
  // persisted document bound already includes this row.
  auto *terms = context.get_cf("value" + INVERT_SUFFIX_TERMS);
  std::unique_ptr<rocksdb::Iterator> iter(
      context.db_->NewIterator(context.read_opts_, terms));
  iter->SeekToFirst();
  ASSERT_TRUE(iter->Valid());
  ASSERT_TRUE(
      context.db_->Delete(context.write_opts_, terms, iter->key()).ok());
  index.reset();
  index = InvertedColumnIndexer::CreateAndOpen("replay", field, context);
  ASSERT_NE(index, nullptr);
  EXPECT_EQ(index->search(value, CompareOp::EQ)->count(), 0u);
  ASSERT_TRUE(index->insert(0, value).ok());
  Verify<uint32_t>(index, {7}, {0, 6, 7, 8});
}

TEST_P(BitSliceIndexTest, SegmentDocIdsAcrossBitmapContainers) {
  auto owner = Open(DataType::INT32);
  ASSERT_NE(owner, nullptr);
  auto index = (*owner)["value"];
  std::vector<std::optional<int32_t>> values;
  for (uint32_t id = 0; id < 70000; ++id) {
    values.emplace_back(
        id % 17 == 0
            ? std::nullopt
            : std::optional<int32_t>(static_cast<int32_t>(id % 257) - 128));
    ASSERT_TRUE((values.back() ? index->insert(id, EncodeValue(*values.back()))
                               : index->insert_null(id))
                    .ok());
  }
  Verify<int32_t>(index, values, {-129, -1, 0, 1, 128, 129});
  ASSERT_TRUE(owner->seal().ok());
  Verify<int32_t>(index, values, {-129, -1, 0, 1, 128, 129});
#ifndef __ANDROID__
  // RocksDB checkpoints require hardlinks, unavailable on the Android emulator.
  const auto snapshot_path = path_ / "checkpoint";
  ASSERT_TRUE(owner->create_snapshot(snapshot_path.string()).ok());
  FieldSchema field("value", DataType::INT32, true, Params());
  auto snapshot = InvertedIndexer::CreateAndOpen(
      "snapshot", snapshot_path.string(), false, {field}, true);
  ASSERT_NE(snapshot, nullptr);
  Verify<int32_t>((*snapshot)["value"], values, {-129, 0, 129});
#endif
}

TEST_P(BitSliceIndexTest, ConcurrentTermQueriesKeepSparsePostings) {
  auto owner = Open(DataType::INT32);
  ASSERT_NE(owner, nullptr);
  auto index = (*owner)["value"];
  const auto value = EncodeValue(int32_t{7});
  for (uint32_t id = 1; id < 1000; id += 2) {
    ASSERT_TRUE(index->insert(id, value).ok());
  }
  EXPECT_EQ(index->search(value, CompareOp::EQ)->count(), 500u);
  std::atomic<bool> done{false};
  std::atomic<bool> valid{true};
  std::thread writer([&]() {
    for (uint32_t id = 0; id < 1000; id += 2) {
      if (!index->insert(id, value).ok()) valid = false;
    }
    done = true;
  });
  do {
    for (auto result : {index->search(value, CompareOp::EQ),
                        index->multi_search({value}, CompareOp::CONTAIN_ANY)}) {
      if (!result) {
        valid = false;
        continue;
      }
      std::vector<uint32_t> ids;
      result->extract_ids(&ids);
      if (ids.size() < 500 || ids.size() > 1000 || !result->contains(999)) {
        valid = false;
      }
      for (auto id : ids) {
        if (id >= 1000) valid = false;
      }
    }
  } while (!done);
  writer.join();
  EXPECT_TRUE(valid);
  EXPECT_EQ(index->search(value, CompareOp::EQ)->count(), 1000u);
}

TEST_P(BitSliceIndexTest, ConcurrentQueriesSeeWholeValues) {
  auto owner = Open(DataType::UINT64);
  ASSERT_NE(owner, nullptr);
  auto index = (*owner)["value"];
  std::atomic<bool> done{false};
  std::atomic<bool> valid{true};
  std::thread writer([&]() {
    for (uint32_t id = 0; id < 400; ++id) {
      auto status =
          id % 3 == 0
              ? index->insert_null(id)
              : index->insert(
                    id,
                    EncodeValue(id % 3 == 1
                                    ? uint64_t{0}
                                    : std::numeric_limits<uint64_t>::max()));
      if (!status.ok()) valid = false;
    }
    done = true;
  });
  do {
    auto result = index->search_filter(EncodeValue(uint64_t{0}), CompareOp::GT);
    if (!result) {
      valid = false;
      break;
    }
    std::vector<uint32_t> ids;
    result->extract_ids(&ids);
    for (uint32_t id : ids) {
      if (id % 3 != 2) valid = false;
    }
  } while (!done);
  writer.join();
  EXPECT_TRUE(valid);
  EXPECT_EQ(
      index->search_filter(EncodeValue(uint64_t{0}), CompareOp::GT)->count(),
      133u);
}

TEST_P(BitSliceIndexTest, DirtySlicesAndIdempotentFlush) {
  auto owner = Open(DataType::UINT64);
  ASSERT_NE(owner, nullptr);
  auto index = (*owner)["value"];
  ASSERT_TRUE(index->insert(0, EncodeValue(uint64_t{1})).ok());
  ASSERT_TRUE(owner->flush().ok());
  EXPECT_EQ(index->bsi_memory_usage().persisted_slices, 1u);
  ASSERT_TRUE(index->insert(0, EncodeValue(uint64_t{1})).ok());
  ASSERT_TRUE(owner->flush().ok());
  EXPECT_EQ(index->bsi_memory_usage().persisted_slices, 1u);
  ASSERT_TRUE(index->insert_null(1).ok());
  ASSERT_TRUE(owner->flush().ok());
  EXPECT_EQ(index->bsi_memory_usage().persisted_slices, 1u);
  ASSERT_TRUE(index->insert(2, EncodeValue(uint64_t{2})).ok());
  ASSERT_TRUE(owner->flush().ok());
  EXPECT_EQ(index->bsi_memory_usage().persisted_slices, 2u);
  auto sample = index->value_sample();
  ASSERT_NE(sample, nullptr);
  ASSERT_EQ(sample->entries.size(), 3u);
  EXPECT_TRUE(sample->entries[1].is_null);
  EXPECT_GT(index->bsi_memory_usage().mutable_bytes, 0u);
  ASSERT_TRUE(owner->seal().ok());
  EXPECT_EQ(index->bsi_memory_usage().mutable_bytes, 0u);
  Verify<uint64_t>(index, {1, std::nullopt, 2}, {0, 1, 2, 3});
  EXPECT_LE(index->bsi_memory_usage().cache_bytes, Params()->bsi_cache_bytes());
  index.reset();
  owner.reset();
  owner = Open(DataType::UINT64, true);
  ASSERT_NE(owner, nullptr);
  index = (*owner)["value"];
  ASSERT_NE(index->value_sample(), nullptr);
  ASSERT_EQ(index->value_sample()->entries.size(), 3u);
  EXPECT_TRUE(index->value_sample()->entries[1].is_null);
  Verify<uint64_t>(index, {1, std::nullopt, 2}, {0, 1, 2, 3});
}

TEST_P(BitSliceIndexTest, SmallCacheAndResultLifetime) {
  FieldSchema field("value", DataType::UINT32, true,
                    std::make_shared<InvertIndexParams>(
                        InvertIndexLayout::TERM_BSI, false, GetParam(), 1024));
  auto owner = InvertedIndexer::CreateAndOpen("budget", path_.string(), true,
                                              {field}, false);
  ASSERT_NE(owner, nullptr);
  auto index = (*owner)["value"];
  std::vector<uint32_t> values;
  for (uint32_t id = 0; id < 10000; ++id) {
    values.push_back((id * 2654435761u) & 4095u);
    ASSERT_TRUE(index->insert(id, EncodeValue(values.back())).ok());
  }
  ASSERT_TRUE(owner->seal().ok());
  auto retained =
      index->search_filter(EncodeValue(uint32_t{2000}), CompareOp::GT);
  ASSERT_NE(retained, nullptr);
  for (uint32_t bound : {1u, 7u, 255u, 1023u, 2048u, 3000u, 4095u}) {
    auto result = index->search_filter(EncodeValue(bound), CompareOp::GT);
    ASSERT_NE(result, nullptr);
    for (uint32_t id = 0; id < values.size(); ++id) {
      EXPECT_EQ(result->contains(id), values[id] > bound);
    }
    const auto usage = index->bsi_memory_usage();
    EXPECT_LE(usage.cache_bytes, 1024u);
    EXPECT_EQ(usage.pinned_cache_bytes, 0u);
  }
  index.reset();
  owner.reset();
  for (uint32_t id = 0; id < values.size(); ++id) {
    EXPECT_EQ(retained->contains(id), values[id] > 2000);
  }
}

TEST_P(BitSliceIndexTest, EmptyRecallAndMaximumDocumentId) {
  auto owner = Open(DataType::UINT32);
  ASSERT_NE(owner, nullptr);
  auto index = (*owner)["value"];
  ASSERT_TRUE(owner->seal().ok());
  EXPECT_EQ(index->search_non_null()->count(), 0u);
  EXPECT_EQ(index->search(EncodeValue(uint32_t{7}), CompareOp::NE)->count(),
            0u);
  index.reset();
  owner.reset();
  owner = Open(DataType::UINT32, true);
  ASSERT_NE(owner, nullptr);
  EXPECT_EQ((*owner)["value"]->search_non_null()->count(), 0u);
  owner.reset();
  std::filesystem::remove_all(path_);
  owner = Open(DataType::UINT32);
  ASSERT_NE(owner, nullptr);
  index = (*owner)["value"];
  ASSERT_TRUE(index->insert(UINT32_MAX, EncodeValue(uint32_t{7})).ok());
  ASSERT_TRUE(owner->seal().ok());
  auto result = index->search_filter(EncodeValue(uint32_t{7}), CompareOp::LE);
  ASSERT_NE(result, nullptr);
  EXPECT_TRUE(result->contains(UINT32_MAX));
  EXPECT_EQ(result->count(), uint64_t{1} << 32);
}

TEST_P(BitSliceIndexTest, QueriesOverlapSealAndDrop) {
  auto owner = Open(DataType::INT32);
  ASSERT_NE(owner, nullptr);
  auto index = (*owner)["value"];
  for (uint32_t id = 0; id < 3000; ++id) {
    ASSERT_TRUE(index->insert(id, EncodeValue(int32_t{7})).ok());
  }
  std::atomic<bool> started{false}, done{false}, valid{true};
  std::thread reader([&] {
    started = true;
    while (!done) {
      auto result =
          index->search_filter(EncodeValue(int32_t{3}), CompareOp::GT);
      if (result && result->count() != 3000) valid = false;
      auto term = index->search(EncodeValue(int32_t{7}), CompareOp::EQ);
      if (term && term->count() != 3000) valid = false;
    }
  });
  while (!started) std::this_thread::yield();
  const auto sealed = owner->seal();
  const auto dropped = index->drop_storage();
  done = true;
  reader.join();
  EXPECT_TRUE(sealed.ok());
  EXPECT_TRUE(dropped.ok());
  EXPECT_TRUE(valid);
  EXPECT_EQ(index->search_filter(EncodeValue(int32_t{3}), CompareOp::GT),
            nullptr);
}

class FailingBsiDB : public rocksdb::StackableDB {
 public:
  explicit FailingBsiDB(rocksdb::DB *db) : StackableDB(db) {}
  bool fail_write{false};
  bool fail_seal{false};
  std::string fail_flush_cf;
  rocksdb::Status Flush(const rocksdb::FlushOptions &options,
                        rocksdb::ColumnFamilyHandle *cf) override {
    if (cf->GetName() == fail_flush_cf) {
      return rocksdb::Status::IOError("Injected CF flush failure");
    }
    return StackableDB::Flush(options, cf);
  }
  rocksdb::Status Write(const rocksdb::WriteOptions &options,
                        rocksdb::WriteBatch *batch) override {
    if (fail_write)
      return rocksdb::Status::IOError("Injected snapshot failure");
    return StackableDB::Write(options, batch);
  }
  rocksdb::Status Put(const rocksdb::WriteOptions &options,
                      rocksdb::ColumnFamilyHandle *cf,
                      const rocksdb::Slice &key,
                      const rocksdb::Slice &value) override {
    if (fail_seal && key.ToString() == "value" + INVERT_KEY_SEALED) {
      return rocksdb::Status::IOError("Injected seal failure");
    }
    return StackableDB::Put(options, cf, key, value);
  }
};

TEST_P(BitSliceIndexTest, FailedSnapshotAndSealCanBeRetried) {
  RocksdbContext context;
  ASSERT_TRUE(context
                  .create(path_.string(),
                          {"value" + INVERT_SUFFIX_TERMS,
                           "value" + INVERT_SUFFIX_BSI, INVERT_CDF},
                          std::make_shared<InvertedRocksdbValueMerger>())
                  .ok());
  auto wrapper = std::make_unique<FailingBsiDB>(context.db_.release());
  auto *fault = wrapper.get();
  context.db_ = std::move(wrapper);
  FieldSchema field("value", DataType::UINT32, true, Params());
  auto index = InvertedColumnIndexer::CreateAndOpen("failure", field, context);
  ASSERT_NE(index, nullptr);
  ASSERT_TRUE(index->insert(0, EncodeValue(uint32_t{1})).ok());
  ASSERT_TRUE(index->insert_null(1).ok());
  fault->fail_write = true;
  EXPECT_FALSE(index->flush_special_values().ok());
  EXPECT_EQ(index->bsi_memory_usage().persisted_slices, 0u);
  Verify<uint32_t>(index, {1, std::nullopt}, {0, 1, 2});
  fault->fail_write = false;
  ASSERT_TRUE(index->flush_special_values().ok());
  EXPECT_EQ(index->bsi_memory_usage().persisted_slices, 1u);
  fault->fail_seal = true;
  EXPECT_FALSE(index->seal().ok());
  EXPECT_FALSE(index->is_sealed());
  ASSERT_TRUE(index->insert(2, EncodeValue(uint32_t{2})).ok());
  fault->fail_seal = false;
  ASSERT_TRUE(index->seal().ok());
  ASSERT_TRUE(context.flush().ok());
  index.reset();
  index = InvertedColumnIndexer::CreateAndOpen("failure", field, context, true);
  ASSERT_NE(index, nullptr);
  Verify<uint32_t>(index, {1, std::nullopt, 2}, {0, 1, 2, 3});
}

TEST_P(BitSliceIndexTest, ProcessExitAfterPartialColumnFamilyFlush) {
  // Re-exec instead of inheriting RocksDB background threads through fork.
  // Keep the parent's path when Google Test replays the test in its child.
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  const char *inherited = std::getenv("ZVEC_BSI_CRASH_TEST_PATH");
  if (inherited)
    path_ = inherited;
  else
    setenv("ZVEC_BSI_CRASH_TEST_PATH", path_.c_str(), 1);
  AILEGO_DEFER([&] {
    if (!inherited) unsetenv("ZVEC_BSI_CRASH_TEST_PATH");
  });
  for (const auto &failed_cf :
       {std::string("default"), "value" + INVERT_SUFFIX_TERMS,
        "value" + INVERT_SUFFIX_BSI}) {
    std::filesystem::remove_all(path_);
    ASSERT_EXIT(
        {
          RocksdbContext context;
          if (!context
                   .create(path_.string(),
                           {"value" + INVERT_SUFFIX_TERMS,
                            "value" + INVERT_SUFFIX_BSI, INVERT_CDF},
                           std::make_shared<InvertedRocksdbValueMerger>())
                   .ok())
            std::_Exit(1);
          auto wrapper = std::make_unique<FailingBsiDB>(context.db_.release());
          auto *fault = wrapper.get();
          context.db_ = std::move(wrapper);
          FieldSchema field("value", DataType::INT32, true, Params());
          auto column =
              InvertedColumnIndexer::CreateAndOpen("partial", field, context);
          if (!column) std::_Exit(2);
          for (uint32_t id = 0; id < 100; ++id) {
            auto status =
                id % 7 == 0 ? column->insert_null(id)
                            : column->insert(id, EncodeValue(int32_t(id) - 50));
            if (!status.ok()) std::_Exit(3);
          }
          if (!column->flush_special_values().ok()) std::_Exit(4);
          fault->fail_flush_cf = failed_cf;
          if (context.flush().ok()) std::_Exit(5);
          // Bypass all destructors: only the CFs preceding the failed flush
          // survive.
          std::_Exit(0);
        },
        ::testing::ExitedWithCode(0), "");
    auto owner = Open(DataType::INT32);
    ASSERT_NE(owner, nullptr) << failed_cf;
    auto index = (*owner)["value"];
    std::vector<std::optional<int32_t>> values;
    // Reapply the same rows, as the segment WAL recovery does. Both surviving
    // and missing terms/slices must converge to one consistent column state.
    for (uint32_t id = 0; id < 100; ++id) {
      values.push_back(id % 7 == 0 ? std::nullopt
                                   : std::optional<int32_t>(int32_t(id) - 50));
      ASSERT_TRUE((values.back()
                       ? index->insert(id, EncodeValue(*values.back()))
                       : index->insert_null(id))
                      .ok());
    }
    Verify<int32_t>(index, values, {-51, -1, 0, 1, 50});
    ASSERT_TRUE(owner->seal().ok());
    Verify<int32_t>(index, values, {-51, -1, 0, 1, 50});
  }
}

INSTANTIATE_TEST_SUITE_P(CacheModes, BitSliceIndexTest,
                         testing::Values(BsiCacheMode::RESIDENT,
                                         BsiCacheMode::ON_DEMAND,
                                         BsiCacheMode::LRU));

}  // namespace
