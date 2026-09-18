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
#include <gtest/gtest.h>
#include <rocksdb/memtablerep.h>
#include <zvec/db/config.h>
#include "db/common/file_helper.h"
#include "db/common/global_resource.h"

namespace zvec {
namespace {

constexpr char kTestPath[] = "./test_rocksdb_memory_budget";

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

}  // namespace
}  // namespace zvec
