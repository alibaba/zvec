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

#include <filesystem>
#include <gtest/gtest.h>
#include <zvec/ailego/io/file.h>
#include <zvec/db/collection.h>
#include "db/common/file_helper.h"
#include "db/index/column/fts_column/fts_file_lifetime.h"
#include "db/index/common/version_manager.h"

namespace zvec {
namespace {

TEST(FtsRecoveryCleanupTest, ReadOnlyPreservesAndWritableRetiresKnownOrphans) {
  const std::string root = "fts_recovery_cleanup.fixture";
  FileHelper::RemoveDirectory(root);
  CollectionSchema schema{"cleanup"};
  schema.add_field(std::make_shared<FieldSchema>(
      "text", DataType::STRING, true,
      std::make_shared<FtsIndexParams>("whitespace")));
  auto collection = Collection::CreateAndOpen(root, schema, {});
  ASSERT_TRUE(collection.has_value()) << collection.error().message();
  ASSERT_TRUE((*collection)->close().ok());
  collection->reset();
  auto manager = VersionManager::Recovery(root);
  ASSERT_TRUE(manager.has_value());
  auto committed = (*manager)->get_current_version();
  ASSERT_TRUE((*manager)->commit(committed).ok());
  const auto segment =
      FileHelper::MakeSegmentPath(root, committed.writing_segment_meta()->id());
  const auto custom = FileHelper::MakeCustomFtsPath(segment, 999, 1, {});
  const std::vector<std::string> files{
      custom, custom + ".pending", root + "/manifest.999999",
      root + "/manifest.999998.pending", root + "/del.999999"};
  const std::vector<std::string> directories{
      custom + ".reserved", segment + "/fts.999.rocksdb",
      segment + "/fts.998.rocksdb.build"};
  const std::vector<std::string> unknown{
      segment + "/fts.notes", root + "/manifest.notes",
      segment + "/fts.0999.rocksdb", root + "/del.0999999"};
  for (const auto &path : files) {
    ailego::File file;
    ASSERT_TRUE(file.create(path, 0));
  }
  for (const auto &path : directories) {
    ASSERT_TRUE(std::filesystem::create_directory(path));
  }
  for (const auto &path : unknown) {
    ASSERT_TRUE(std::filesystem::create_directory(path));
  }
  // A reader from an earlier manager keeps its file alive across reopen.
  auto pinned = (*manager)->fts_retirement()->track(custom, false, false);
  const auto retired_segment = root + "/999";
  ASSERT_TRUE(std::filesystem::create_directory(retired_segment));
  auto directory_pin =
      (*manager)->fts_retirement()->track(retired_segment, true, false);
  auto readonly = Collection::Open(root, CollectionOptions{true, true});
  ASSERT_TRUE(readonly.has_value()) << readonly.error().message();
  ASSERT_TRUE((*readonly)->close().ok());
  readonly->reset();
  for (const auto &path : files) {
    EXPECT_TRUE(std::filesystem::exists(path));
  }
  for (const auto &path : directories) {
    EXPECT_TRUE(std::filesystem::exists(path));
  }
  auto writable = Collection::Open(root, CollectionOptions{});
  ASSERT_TRUE(writable.has_value()) << writable.error().message();
  for (const auto &path : files) {
    EXPECT_EQ(path == custom, std::filesystem::exists(path)) << path;
  }
  for (const auto &path : directories) {
    EXPECT_FALSE(std::filesystem::exists(path)) << path;
  }
  for (const auto &path : unknown) {
    EXPECT_TRUE(std::filesystem::exists(path)) << path;
  }
  EXPECT_TRUE(std::filesystem::exists(retired_segment));
  directory_pin.reset();
  EXPECT_FALSE(std::filesystem::exists(retired_segment));
  pinned.reset();
  EXPECT_FALSE(std::filesystem::exists(custom));
  ASSERT_TRUE((*writable)->close().ok());
  writable->reset();
  FileHelper::RemoveDirectory(root);
}

TEST(FtsRecoveryCleanupTest, MetadataPinsSurviveCommitUntilVersionReleased) {
  const std::string root = "fts_metadata_pins.fixture";
  FileHelper::RemoveDirectory(root);
  ASSERT_TRUE(FileHelper::CreateDirectory(root));
  Version initial;
  initial.set_schema(CollectionSchema{"pins"});
  auto manager = VersionManager::Create(root, initial);
  ASSERT_TRUE(manager.has_value());
  for (const auto &name : {"del.0", "del.1"}) {
    ailego::File file;
    ASSERT_TRUE(file.create(root + "/" + name, 0));
  }
  ASSERT_TRUE((*manager)->flush().ok());
  auto snapshot = (*manager)->get_current_version();
  auto candidate = snapshot.clone();
  candidate.set_delete_snapshot_path_suffix(1);
  ASSERT_TRUE((*manager)->commit(candidate).ok());
  candidate = Version{};
  EXPECT_TRUE(std::filesystem::exists(root + "/manifest.0"));
  EXPECT_TRUE(std::filesystem::exists(root + "/del.0"));
  snapshot = Version{};
  EXPECT_FALSE(std::filesystem::exists(root + "/manifest.0"));
  EXPECT_FALSE(std::filesystem::exists(root + "/del.0"));
  EXPECT_TRUE(std::filesystem::exists(root + "/manifest.1"));
  EXPECT_TRUE(std::filesystem::exists(root + "/del.1"));
  auto recovered = VersionManager::Recovery(root);
  ASSERT_TRUE(recovered.has_value());
  EXPECT_EQ(1,
            (*recovered)->get_current_version().delete_snapshot_path_suffix());
  FileHelper::RemoveDirectory(root);
}

}  // namespace
}  // namespace zvec
