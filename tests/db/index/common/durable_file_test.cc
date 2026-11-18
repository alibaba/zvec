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

#include "db/common/durable_file.h"
#include <array>
#include <gtest/gtest.h>
#include "db/common/file_helper.h"

namespace zvec {
namespace {

TEST(DurableFileTest, FailureBeforeRenamePreservesOldAndAfterIsUncertain) {
  const std::string path = "durable.fixture";
  for (auto failure :
       {DurableFile::Stage::kWrite, DurableFile::Stage::kFileSync,
        DurableFile::Stage::kRename, DurableFile::Stage::kDirectorySync}) {
    auto initial = DurableFile::Replace(path, "old");
    ASSERT_TRUE(initial.status.ok()) << initial.status.message();
    ASSERT_TRUE(initial.renamed);
    unsigned calls = 0;
    auto hook = [&](DurableFile::Stage stage, const std::string &destination) {
      EXPECT_EQ(path, destination);
      ++calls;
      return stage == failure ? Status::InternalError("Injected I/O failure")
                              : Status::OK();
    };
    auto outcome = DurableFile::Replace(path, "new", hook);
    ASSERT_FALSE(outcome.status.ok());
    const bool uncertain = failure == DurableFile::Stage::kDirectorySync;
    EXPECT_EQ(uncertain, outcome.renamed);
    EXPECT_GT(calls, 0u);
    ailego::File file;
    ASSERT_TRUE(file.open(path, true));
    std::array<char, 3> bytes{};
    ASSERT_EQ(bytes.size(), file.read(bytes.data(), bytes.size()));
    EXPECT_EQ(uncertain ? "new" : "old",
              std::string(bytes.data(), bytes.size()));
  }
  EXPECT_TRUE(FileHelper::RemoveFile(path));
  FileHelper::RemoveFile(path + ".pending");
}

}  // namespace
}  // namespace zvec
