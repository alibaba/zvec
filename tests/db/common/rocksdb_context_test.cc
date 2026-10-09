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

#include "db/common/rocksdb_context.h"
#include <algorithm>
#include <gtest/gtest.h>
#include <rocksdb/convenience.h>
#include <rocksdb/memtablerep.h>
#include <zvec/db/config.h>
#include "db/common/file_helper.h"
#include "db/common/global_resource.h"

namespace zvec {
namespace {

constexpr char kTestPath[] = "./test_rocksdb_memory_budget";

size_t hash_bucket_count(const rocksdb::Options &options) {
  std::string value;
  const auto status = options.memtable_factory->GetOption(
      rocksdb::ConfigOptions{}, "bucket_count", &value);
  EXPECT_TRUE(status.ok()) << status.ToString();
  return status.ok() ? std::stoull(value) : 0;
}

std::vector<std::string> column_family_names(size_t count) {
  std::vector<std::string> names{rocksdb::kDefaultColumnFamilyName};
  for (size_t i = 1; i < count; ++i) {
    names.push_back("cf" + std::to_string(i));
  }
  return names;
}

class RocksdbContextMemoryTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    GlobalConfig::ConfigData config;
    config.memory_limit_bytes = 128ULL * 1024 * 1024;
    config.query_thread_count = 1;
    config.optimize_thread_count = 1;
    const auto status = GlobalConfig::Instance().initialize(config);
    ASSERT_TRUE(status.ok()) << status.message();
  }

  void SetUp() override {
    FileHelper::RemoveDirectory(kTestPath);
  }

  void TearDown() override {
    FileHelper::RemoveDirectory(kTestPath);
  }
};

TEST_F(RocksdbContextMemoryTest,
       SharesWriteBudgetAndAvoidsReadOnlyFtsHashTable) {
  const std::vector<std::string> column_families = {
      "postings", "positions", "term_freq", "max_tf", "doc_len", "stat"};

  RocksdbContext writer;
  ASSERT_TRUE(writer
                  .create(RocksdbContext::Args{
                      kTestPath, column_families, nullptr, {}, true})
                  .ok());
  EXPECT_EQ(GlobalResource::Instance().rocksdb_write_buffer_manager(),
            writer.create_opts_.write_buffer_manager);
  EXPECT_STREQ("HashSkipListRepFactory",
               writer.create_opts_.memtable_factory->Name());
  ASSERT_TRUE(writer.close().ok());

  RocksdbContext reader;
  ASSERT_TRUE(reader
                  .open(RocksdbContext::Args{kTestPath, {}, nullptr, {}, true},
                        /*read_only=*/true)
                  .ok());
  EXPECT_EQ(GlobalResource::Instance().rocksdb_write_buffer_manager(),
            reader.create_opts_.write_buffer_manager);
  EXPECT_STREQ("SkipListFactory", reader.create_opts_.memtable_factory->Name());
  ASSERT_TRUE(reader.close().ok());
}

TEST_F(RocksdbContextMemoryTest, WritableHashTablesHaveBoundedFixedOverhead) {
  for (size_t cf_count : {1u, 7u, 12u}) {
    SCOPED_TRACE(cf_count);
    RocksdbContext writer;
    ASSERT_TRUE(
        writer
            .create(RocksdbContext::Args{
                kTestPath, column_family_names(cf_count), nullptr, {}, true})
            .ok());
    const uint64_t per_cf_budget =
        GlobalResource::Instance().rocksdb_memory_capacity() / 8 / cf_count;
    const uint64_t expected =
        std::clamp<uint64_t>(per_cf_budget / sizeof(void *), 4096, 65536);
    EXPECT_EQ(expected, hash_bucket_count(writer.create_opts_));
    for (auto *cf : writer.cf_handles_) {
      EXPECT_EQ(expected, hash_bucket_count(writer.db_->GetOptions(cf)));
    }
    // Reducing bucket overhead must not change flush thresholds or opt out
    // of the process-wide memtable budget.
    EXPECT_EQ(128ULL * 1024 * 1024, writer.create_opts_.write_buffer_size);
    EXPECT_EQ(GlobalResource::Instance().rocksdb_write_buffer_manager(),
              writer.create_opts_.write_buffer_manager);
    ASSERT_TRUE(writer.close().ok());
    FileHelper::RemoveDirectory(kTestPath);
  }
}

TEST_F(RocksdbContextMemoryTest, ManyColumnFamiliesKeepBudgetScalingOnReopen) {
  for (size_t cf_count : {33u, 129u}) {
    SCOPED_TRACE(cf_count);
    const uint64_t per_cf_budget =
        GlobalResource::Instance().rocksdb_memory_capacity() / 8 / cf_count;
    const uint64_t expected =
        std::clamp<uint64_t>(per_cf_budget / sizeof(void *), 4096, 65536);
    EXPECT_LT(expected, 65536u);
    RocksdbContext writer;
    ASSERT_TRUE(
        writer
            .create(RocksdbContext::Args{
                kTestPath, column_family_names(cf_count), nullptr, {}, true})
            .ok());
    EXPECT_EQ(expected, hash_bucket_count(writer.create_opts_));
    ASSERT_TRUE(writer.close().ok());

    // With no explicit CF list, sizing must use the persisted CF count,
    // not the initial single-CF estimate used before ListColumnFamilies.
    RocksdbContext reopened;
    ASSERT_TRUE(
        reopened
            .open(RocksdbContext::Args{kTestPath, {}, nullptr, {}, true},
                  /*read_only=*/false)
            .ok());
    EXPECT_EQ(expected, hash_bucket_count(reopened.create_opts_));
    for (auto *cf : reopened.cf_handles_) {
      EXPECT_EQ(expected, hash_bucket_count(reopened.db_->GetOptions(cf)));
    }
    ASSERT_TRUE(reopened.close().ok());
    FileHelper::RemoveDirectory(kTestPath);
  }
}

TEST_F(RocksdbContextMemoryTest, HashWritesSurviveFlushAndReadOnlyReopen) {
  const std::vector<std::string> names{"postings", "positions"};
  RocksdbContext writer;
  ASSERT_TRUE(
      writer.create(RocksdbContext::Args{kTestPath, names, nullptr, {}, true})
          .ok());
  for (const auto &name : names) {
    auto *cf = writer.get_cf(name);
    ASSERT_NE(nullptr, cf);
    for (size_t i = 0; i < 256; ++i) {
      // Every key shares one prefix, exercising collisions within a bucket.
      const auto key = "prefix00" + std::to_string(i);
      ASSERT_TRUE(writer.db_->Put(writer.write_opts_, cf, key, "before").ok());
    }
  }
  ASSERT_TRUE(writer.flush().ok());
  for (const auto &name : names) {
    auto *cf = writer.get_cf(name);
    for (size_t i = 0; i < 256; ++i) {
      const auto key = "prefix00" + std::to_string(i);
      if (i % 2 == 0) {
        ASSERT_TRUE(writer.db_->Delete(writer.write_opts_, cf, key).ok());
      } else {
        ASSERT_TRUE(writer.db_->Put(writer.write_opts_, cf, key, "after").ok());
      }
    }
  }
  ASSERT_TRUE(writer.close().ok());

  RocksdbContext reader;
  ASSERT_TRUE(reader
                  .open(RocksdbContext::Args{kTestPath, {}, nullptr, {}, true},
                        /*read_only=*/true)
                  .ok());
  EXPECT_STREQ("SkipListFactory", reader.create_opts_.memtable_factory->Name());
  for (const auto &name : names) {
    auto *cf = reader.get_cf(name);
    ASSERT_NE(nullptr, cf);
    for (size_t i = 0; i < 256; ++i) {
      const auto key = "prefix00" + std::to_string(i);
      std::string value;
      const auto status = reader.db_->Get(reader.read_opts_, cf, key, &value);
      if (i % 2 == 0) {
        EXPECT_TRUE(status.IsNotFound()) << status.ToString();
      } else {
        ASSERT_TRUE(status.ok()) << status.ToString();
        EXPECT_EQ("after", value);
      }
    }
  }
  ASSERT_TRUE(reader.close().ok());
}

}  // namespace
}  // namespace zvec
