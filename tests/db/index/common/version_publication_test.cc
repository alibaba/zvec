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

#include <gtest/gtest.h>
#include "db/common/file_helper.h"
#include "db/index/common/version_manager.h"

namespace zvec {
namespace {

TEST(VersionPublicationTest, PreparingCandidateCannotMutatePublishedMetadata) {
  Version initial;
  initial.set_schema(CollectionSchema{});
  auto persisted = std::make_shared<SegmentMeta>(1);
  persisted->add_persisted_block({1, BlockType::SCALAR, 0, 9, 10, {"text"}});
  ASSERT_TRUE(initial.add_persisted_segment_meta(persisted).ok());
  auto writing = std::make_shared<SegmentMeta>(2);
  writing->set_writing_forward_block(
      {2, BlockType::SCALAR, 10, 19, 10, {"text"}});
  initial.reset_writing_segment_meta(writing);
  auto created = VersionManager::Create("unused_version_path", initial);
  ASSERT_TRUE(created.has_value());
  auto manager = *created;
  auto candidate = manager->get_current_version();
  candidate.persisted_segment_metas()[0]->persisted_blocks()[0].set_doc_count(
      3);
  candidate.writing_segment_meta()->remove_writing_forward_block();
  EXPECT_EQ(10u, manager->get_current_version()
                     .persisted_segment_metas()[0]
                     ->persisted_blocks()[0]
                     .doc_count());
  EXPECT_TRUE(manager->get_current_version()
                  .writing_segment_meta()
                  ->has_writing_forward_block());
  EXPECT_EQ(10u, persisted->persisted_blocks()[0].doc_count());
  EXPECT_TRUE(writing->has_writing_forward_block());

  ASSERT_TRUE(manager->apply(candidate).ok());
  candidate.persisted_segment_metas()[0]->persisted_blocks()[0].set_doc_count(
      4);
  EXPECT_EQ(3u, manager->get_current_version()
                    .persisted_segment_metas()[0]
                    ->persisted_blocks()[0]
                    .doc_count());
}

TEST(VersionPublicationTest, CommittedAuthorityIgnoresOrphanHigherManifest) {
  const std::string path = "version_publication.fixture";
  FileHelper::RemoveDirectory(path);
  ASSERT_TRUE(FileHelper::CreateDirectory(path));
  Version initial;
  initial.set_schema(CollectionSchema("old"));
  auto created = VersionManager::Create(path, initial);
  ASSERT_TRUE(created.has_value());
  ASSERT_TRUE((*created)->flush().ok());
  auto candidate = (*created)->get_current_version();
  candidate.set_schema(CollectionSchema("committed"));
  auto status = (*created)->commit(candidate);
  ASSERT_TRUE(status.ok()) << status.message();

  Version orphan;
  orphan.set_schema(CollectionSchema("not_committed"));
  ASSERT_TRUE(Version::Save(path + "/manifest.999", orphan).ok());
  auto recovered = VersionManager::Recovery(path);
  ASSERT_TRUE(recovered.has_value()) << recovered.error().message();
  EXPECT_EQ("committed", (*recovered)->get_current_version().schema().name());
  EXPECT_TRUE(FileHelper::RemoveDirectory(path));
}

TEST(VersionPublicationTest,
     SourceCheckpointPublishesMetadataOnlyAfterDurability) {
  const std::string path = "version_checkpoint.fixture";
  for (auto failure :
       {DurableFile::Stage::kWrite, DurableFile::Stage::kFileSync,
        DurableFile::Stage::kRename, DurableFile::Stage::kDirectorySync}) {
    FileHelper::RemoveDirectory(path);
    ASSERT_TRUE(FileHelper::CreateDirectory(path));
    Version initial;
    initial.set_schema(CollectionSchema("old"));
    auto manager = VersionManager::Create(path, initial);
    ASSERT_TRUE(manager.has_value());
    ASSERT_TRUE((*manager)->flush().ok());
    auto candidate = (*manager)->get_current_version();
    candidate.set_schema(CollectionSchema("new"));
    candidate.set_delete_snapshot_path_suffix(1);
    auto status =
        (*manager)->checkpoint(candidate, [&](auto stage, const auto &) {
          return stage == failure ? Status::InternalError("injected")
                                  : Status::OK();
        });
    EXPECT_FALSE(status.ok());
    EXPECT_EQ("old", (*manager)->get_current_version().schema().name());
    EXPECT_EQ(0,
              (*manager)->get_current_version().delete_snapshot_path_suffix());
    EXPECT_FALSE((*manager)->uses_commit_record());
    auto recovered = VersionManager::Recovery(path);
    ASSERT_TRUE(recovered.has_value());
    const bool uncertain = failure == DurableFile::Stage::kDirectorySync;
    EXPECT_EQ(uncertain, (*manager)->requires_recovery());
    EXPECT_EQ(uncertain ? "new" : "old",
              (*recovered)->get_current_version().schema().name());
    if (uncertain) {
      (*manager)->set_delete_snapshot_path_suffix(99);
      EXPECT_EQ(
          0, (*manager)->get_current_version().delete_snapshot_path_suffix());
    } else {
      ASSERT_TRUE((*manager)->checkpoint(candidate).ok());
      EXPECT_EQ("new", (*manager)->get_current_version().schema().name());
    }
    FileHelper::RemoveDirectory(path);
  }
}

TEST(VersionPublicationTest, PublicationFailuresRecoverOnlyCompleteVersions) {
  const std::string path = "version_failure.fixture";
  for (bool fail_authority : {false, true}) {
    for (auto failure :
         {DurableFile::Stage::kWrite, DurableFile::Stage::kFileSync,
          DurableFile::Stage::kRename, DurableFile::Stage::kDirectorySync}) {
      FileHelper::RemoveDirectory(path);
      ASSERT_TRUE(FileHelper::CreateDirectory(path));
      Version initial;
      initial.set_schema(CollectionSchema("old"));
      auto created = VersionManager::Create(path, initial);
      ASSERT_TRUE(created.has_value());
      auto manager = *created;
      ASSERT_TRUE(manager->flush().ok());
      ASSERT_TRUE(manager->commit(initial).ok());
      auto candidate = manager->get_current_version();
      candidate.set_schema(CollectionSchema("new"));
      unsigned injected = 0;
      auto hook = [&](DurableFile::Stage stage, const std::string &target) {
        const bool authority = target == path + "/manifest.current";
        if (stage == failure && authority == fail_authority) {
          ++injected;
          return Status::InternalError("Injected publication failure");
        }
        return Status::OK();
      };
      EXPECT_FALSE(manager->commit(candidate, hook).ok());
      EXPECT_EQ(1u, injected);
      EXPECT_EQ("old", manager->get_current_version().schema().name());
      const bool uncertain =
          fail_authority && failure == DurableFile::Stage::kDirectorySync;
      EXPECT_EQ(uncertain, manager->requires_recovery());
      auto reopened = VersionManager::Recovery(path);
      ASSERT_TRUE(reopened.has_value()) << reopened.error().message();
      EXPECT_EQ(uncertain ? "new" : "old",
                (*reopened)->get_current_version().schema().name());
      if (uncertain) {
        EXPECT_FALSE(manager->flush().ok());
        EXPECT_FALSE(manager->apply(initial).ok());
        EXPECT_FALSE(manager->commit(initial).ok());
      } else {
        // A failed candidate must not prevent a fresh attempt with updated
        // source metadata, nor overwrite the previous prepared file.
        candidate.set_schema(CollectionSchema("retry"));
        EXPECT_TRUE(manager->commit(candidate).ok());
        reopened = VersionManager::Recovery(path);
        ASSERT_TRUE(reopened.has_value());
        EXPECT_EQ("retry", (*reopened)->get_current_version().schema().name());
      }
      EXPECT_TRUE(FileHelper::RemoveDirectory(path));
    }
  }
}

TEST(VersionPublicationTest, CorruptCommittedStateNeverFallsBackToOldManifest) {
  const std::string path = "version_corrupt.fixture";
  for (const auto *damage : {"manifest.current", "manifest.1"}) {
    FileHelper::RemoveDirectory(path);
    ASSERT_TRUE(FileHelper::CreateDirectory(path));
    Version initial;
    initial.set_schema(CollectionSchema("old"));
    auto created = VersionManager::Create(path, initial);
    ASSERT_TRUE(created.has_value());
    ASSERT_TRUE((*created)->flush().ok());
    // Retain an actual metadata snapshot so normal retirement keeps the old
    // manifest available; corruption still must not select it as authority.
    auto old_snapshot = (*created)->get_current_version();
    ASSERT_TRUE((*created)->commit(initial).ok());
    ailego::File file;
    ASSERT_TRUE(file.open(path + "/" + damage, false));
    const char zero = 0;
    ASSERT_EQ(1u, file.write(1, &zero, 1));
    file.close();
    EXPECT_FALSE(VersionManager::Recovery(path).has_value());
    // The valid old manifest remains, but it is not recovery authority.
    Version old;
    EXPECT_TRUE(Version::Load(path + "/manifest.0", &old).ok());
    EXPECT_TRUE(FileHelper::RemoveDirectory(path));
  }
}

TEST(VersionPublicationTest, GatedManifestRequiresAuthorityAndRemainsGated) {
  const std::string path = "version_gate.fixture";
  FileHelper::RemoveDirectory(path);
  ASSERT_TRUE(FileHelper::CreateDirectory(path));
  Version initial;
  initial.set_schema(CollectionSchema("old"));
  auto created = VersionManager::Create(path, initial);
  ASSERT_TRUE(created.has_value());
  ASSERT_TRUE((*created)->flush().ok());
  ASSERT_TRUE((*created)->commit(initial).ok());
  Version committed;
  ASSERT_TRUE(Version::Load(path + "/manifest.1", &committed).ok());
  EXPECT_TRUE(committed.requires_commit_record());
  ASSERT_TRUE((*created)->flush().ok());
  ASSERT_TRUE(Version::Load(path + "/manifest.2", &committed).ok());
  EXPECT_TRUE(committed.requires_commit_record());
  EXPECT_TRUE(FileHelper::RemoveFile(path + "/manifest.current"));
  EXPECT_FALSE(VersionManager::Recovery(path).has_value());
  EXPECT_TRUE(FileHelper::RemoveDirectory(path));
}

TEST(VersionPublicationTest, BootstrapUncertaintyKeepsRecoverableSource) {
  const std::string path = "version_bootstrap.fixture";
  FileHelper::RemoveDirectory(path);
  ASSERT_TRUE(FileHelper::CreateDirectory(path));
  Version initial;
  initial.set_schema(CollectionSchema("source"));
  auto created = VersionManager::Create(path, initial);
  ASSERT_TRUE(created.has_value());
  auto manager = *created;
  EXPECT_FALSE(manager->commit(initial).ok());  // No source checkpoint yet.
  ASSERT_TRUE(manager->flush().ok());
  Version candidate = initial.clone();
  candidate.set_schema(CollectionSchema("candidate"));
  auto hook = [&](DurableFile::Stage stage, const std::string &target) {
    if (stage == DurableFile::Stage::kDirectorySync &&
        target == path + "/manifest.current") {
      return Status::InternalError("Bootstrap directory sync failed");
    }
    return Status::OK();
  };
  EXPECT_FALSE(manager->commit(candidate, hook).ok());
  EXPECT_TRUE(manager->requires_recovery());
  auto reopened = VersionManager::Recovery(path);
  ASSERT_TRUE(reopened.has_value());
  EXPECT_EQ("source", (*reopened)->get_current_version().schema().name());
  EXPECT_TRUE((*reopened)->commit(candidate).ok());
  reopened = VersionManager::Recovery(path);
  ASSERT_TRUE(reopened.has_value());
  EXPECT_EQ("candidate", (*reopened)->get_current_version().schema().name());
  EXPECT_TRUE(FileHelper::RemoveDirectory(path));
}

}  // namespace
}  // namespace zvec
