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

#include "db/index/common/id_map.h"
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <set>
#include <string>
#include <gtest/gtest.h>
#include "db/common/file_helper.h"

namespace zvec {

class IDMapTest : public ::testing::Test {
 protected:
  void SetUp() override {
    FileHelper::RemoveDirectory(path_);
    id_map_ = IDMap::CreateAndOpen("test", path_, true, false);
    ASSERT_NE(id_map_, nullptr);
  }

  void TearDown() override {
    id_map_.reset();
    FileHelper::RemoveDirectory(path_);
  }

  std::set<std::string> ssts() const {
    std::set<std::string> names;
    for (const auto &entry : std::filesystem::directory_iterator(path_)) {
      if (entry.path().extension() == ".sst") {
        names.insert(entry.path().filename().string());
      }
    }
    return names;
  }

  // Ascending keys, like UUIDv7 / ULID / auto-increment primary keys
  static std::string key(uint64_t i, size_t key_size = 16) {
    char prefix[17];
    snprintf(prefix, sizeof(prefix), "%016llu", (unsigned long long)i);
    std::string k(prefix);
    k.resize(std::max(key_size, k.size()), 'x');
    return k;
  }

  void insert(uint64_t count, size_t key_size = 16) {
    for (uint64_t i = 0; i < count; ++i, ++next_) {
      ASSERT_TRUE(id_map_->upsert(key(next_, key_size), next_).ok());
    }
  }

  void expect_doc(const std::string &k, uint64_t expected) {
    uint64_t doc_id = UINT64_MAX;
    EXPECT_TRUE(id_map_->has(k, &doc_id)) << k;
    EXPECT_EQ(doc_id, expected);
  }

  // level0_file_num_compaction_trigger of the shared RocksDB options
  static constexpr size_t kMaxL0Files = 2;

  const std::string path_{"./test_id_map"};
  IDMap::Ptr id_map_;
  uint64_t next_{0};
};

// Flushing ascending keys used to leave one SST per flush.
TEST_F(IDMapTest, FlushKeepsSstCountBounded) {
  size_t max_ssts = 0;
  for (int round = 0; round < 200; ++round) {
    insert(64);
    ASSERT_TRUE(id_map_->flush().ok());
    max_ssts = std::max(max_ssts, ssts().size());
  }
  EXPECT_LE(max_ssts, 16u);

  // One merged SST, plus the L0 files RocksDB moves down on its own
  ASSERT_TRUE(id_map_->compact().ok());
  EXPECT_LE(ssts().size(), 1u + kMaxL0Files);
  expect_doc(key(0), 0);
  expect_doc(key(next_ - 1), next_ - 1);
}

// Full-size SSTs must never be rewritten to merge small ones behind them.
TEST_F(IDMapTest, CompactLeavesFullSizeSstsAlone) {
  insert(20000, 1024);  // ~20MB, above the small-file threshold
  ASSERT_TRUE(id_map_->flush().ok());
  const auto big = ssts();
  ASSERT_EQ(big.size(), 1u);

  for (int round = 0; round < 50; ++round) {
    insert(64);
    ASSERT_TRUE(id_map_->flush().ok());
    EXPECT_TRUE(ssts().count(*big.begin()));
  }
  ASSERT_TRUE(id_map_->compact().ok());
  auto after = ssts();
  EXPECT_TRUE(after.count(*big.begin()));
  EXPECT_LE(after.size(), 2u + kMaxL0Files);
  expect_doc(key(0, 1024), 0);
  expect_doc(key(next_ - 1), next_ - 1);
}

}  // namespace zvec
