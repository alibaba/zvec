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
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>
#include <gtest/gtest.h>

using namespace zvec;
namespace fs = std::filesystem;

namespace {

class RocksdbContextTest : public ::testing::Test {
 protected:
  void SetUp() override {
    root_ = fs::temp_directory_path() /
            ("rocksdb_context_test_" + std::to_string(::getpid()));
    fs::remove_all(root_);
    fs::create_directories(root_);
  }
  void TearDown() override {
    fs::remove_all(root_);
  }

  std::string path(const std::string &name) const {
    return (root_ / name).string();
  }

  fs::path root_;
};

std::vector<std::string> HandleNames(const RocksdbContext &ctx) {
  std::vector<std::string> names;
  for (auto *handle : ctx.cf_handles_) {
    names.push_back(handle->GetName());
  }
  return names;
}

// Highest N among the OPTIONS-N files in a RocksDB directory. RocksDB takes a
// new file number for every OPTIONS file it writes.
uint64_t NewestOptionsFileNumber(const std::string &dir) {
  uint64_t newest = 0;
  for (const auto &entry : fs::directory_iterator(dir)) {
    const auto name = entry.path().filename().string();
    if (name.rfind("OPTIONS-", 0) == 0 && name.find('.') == std::string::npos) {
      newest = std::max<uint64_t>(newest, std::stoull(name.substr(8)));
    }
  }
  return newest;
}

}  // namespace

TEST_F(RocksdbContextTest, CreateWritesOptionsFileOnce) {
  RocksdbContext empty;
  ASSERT_TRUE(empty.create(path("empty"), std::vector<std::string>{}).ok());
  ASSERT_TRUE(empty.close().ok());

  std::vector<std::string> families;
  for (int i = 0; i < 24; ++i) {
    families.push_back("cf" + std::to_string(i));
  }
  RocksdbContext many;
  ASSERT_TRUE(many.create(path("many"), families).ok());
  ASSERT_TRUE(many.close().ok());

  // Creating the families takes one OPTIONS file number beyond an instance
  // without families, not one per family.
  const auto base = NewestOptionsFileNumber(path("empty"));
  const auto with_families = NewestOptionsFileNumber(path("many"));
  EXPECT_LE(with_families, base + 2)
      << "base OPTIONS number " << base << ", with 24 families "
      << with_families;
}

TEST_F(RocksdbContextTest, CreateKeepsHandleOrder) {
  const std::vector<std::vector<std::string>> cases = {
      {"a", "default", "b"}, {"default", "a", "b"}, {"a", "b"}, {}};
  const std::vector<std::vector<std::string>> expected = {
      {"a", "default", "b"},
      {"default", "a", "b"},
      {"a", "b", "default"},
      {"default"}};
  for (size_t i = 0; i < cases.size(); ++i) {
    RocksdbContext ctx;
    ASSERT_TRUE(ctx.create(path("order" + std::to_string(i)), cases[i]).ok());
    EXPECT_EQ(HandleNames(ctx), expected[i]) << "case " << i;
    for (const auto &name : cases[i]) {
      ASSERT_NE(ctx.get_cf(name), nullptr) << name;
    }
    ASSERT_TRUE(ctx.close().ok());

    RocksdbContext reopened;
    ASSERT_TRUE(reopened.open(path("order" + std::to_string(i)), false).ok());
    EXPECT_EQ(reopened.cf_handles_.size(), expected[i].size());
    ASSERT_TRUE(reopened.close().ok());
  }
}

TEST_F(RocksdbContextTest, CreateFailureLeavesNoOpenHandles) {
  // The second "a" fails inside family creation, after "a" and "b" exist.
  RocksdbContext ctx;
  auto s = ctx.create(path("dup"), std::vector<std::string>{"a", "b", "a"});
  ASSERT_FALSE(s.ok());
  EXPECT_EQ(ctx.db_, nullptr);
  EXPECT_TRUE(ctx.cf_handles_.empty());

  // The failed instance released its lock: another instance can open it.
  std::vector<std::string> names;
  ASSERT_TRUE(rocksdb::DB::ListColumnFamilies(rocksdb::DBOptions(),
                                              path("dup"), &names)
                  .ok());
  std::vector<rocksdb::ColumnFamilyDescriptor> descriptors;
  for (const auto &name : names) {
    descriptors.emplace_back(name, rocksdb::ColumnFamilyOptions());
  }
  std::vector<rocksdb::ColumnFamilyHandle *> handles;
  rocksdb::DB *db = nullptr;
  ASSERT_TRUE(
      rocksdb::DB::Open(rocksdb::DBOptions(), path("dup"), descriptors,
                        &handles, &db)
          .ok());
  for (auto *handle : handles) {
    db->DestroyColumnFamilyHandle(handle);
  }
  delete db;
}
