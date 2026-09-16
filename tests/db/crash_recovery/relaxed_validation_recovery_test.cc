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

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/utility/file_helper.h>
#include <zvec/db/collection.h>
#include <zvec/db/doc.h>
#include <zvec/db/schema.h>
#include "db/index/common/id_map.h"
#include "db/index/common/version_manager.h"
#include "db/index/storage/wal/wal_file.h"

namespace zvec {
namespace {

std::string ReadFileBytes(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(file), {});
}

std::string FindWal(const std::string &path) {
  for (const auto &entry :
       std::filesystem::recursive_directory_iterator(path)) {
    if (entry.path().extension() == ".wal") return entry.path().string();
  }
  return {};
}

std::map<std::string, std::string> ReadManifests(const std::string &path) {
  std::map<std::string, std::string> result;
  for (const auto &entry :
       std::filesystem::recursive_directory_iterator(path)) {
    if (entry.path().filename().string().rfind("manifest", 0) == 0) {
      result.emplace(entry.path().string(),
                     ReadFileBytes(entry.path().string()));
    }
  }
  return result;
}

// Only called in ASSERT_EXIT children. Intentionally skip collection cleanup.
void WriteStringDocsAndExit(const std::string &path,
                            const std::vector<std::string> &ids,
                            const std::vector<std::string> &values) {
  CollectionSchema schema("wal_recovery");
  if (!schema
           .add_field(
               std::make_shared<FieldSchema>("text", DataType::STRING, false))
           .ok()) {
    std::_Exit(1);
  }
  auto created = Collection::CreateAndOpen(path, schema, CollectionOptions{});
  if (!created.has_value()) std::_Exit(2);
  auto collection = std::move(created).value();
  std::vector<Doc> docs;
  for (size_t i = 0; i < ids.size(); ++i) {
    Doc doc;
    doc.set_pk(ids[i]);
    doc.set<std::string>("text", values[i]);
    docs.push_back(std::move(doc));
  }
  auto result = collection->insert(docs);
  if (!result.has_value()) std::_Exit(3);
  for (const auto &status : result.value()) {
    if (!status.ok()) std::_Exit(4);
  }
  std::_Exit(0);
}

std::string FindIdMap(const std::string &path) {
  for (const auto &entry : std::filesystem::directory_iterator(path)) {
    if (entry.is_directory() &&
        entry.path().filename().string().rfind("idmap", 0) == 0) {
      return entry.path().string();
    }
  }
  return {};
}

void WriteUpsertsAndExit(const std::string &path, bool persisted_base,
                         bool append_suffix, bool separate_segment = false) {
  CollectionSchema schema("wal_recovery");
  if (!schema
           .add_field(
               std::make_shared<FieldSchema>("text", DataType::STRING, false))
           .ok()) {
    std::_Exit(1);
  }
  auto created = Collection::CreateAndOpen(path, schema, CollectionOptions{});
  if (!created.has_value()) std::_Exit(2);
  auto collection = std::move(created).value();
  Doc doc;
  doc.set_pk("target");
  if (persisted_base) {
    doc.set<std::string>("text", "original");
    std::vector<Doc> docs{doc};
    auto inserted = collection->insert(docs);
    if (!inserted.has_value() || !inserted.value().front().ok() ||
        !collection->flush().ok())
      std::_Exit(3);
    if (separate_segment && !collection->optimize().ok()) std::_Exit(6);
  }
  for (const auto &value : {"first", "second"}) {
    doc.set<std::string>("text", value);
    std::vector<Doc> docs{doc};
    auto updated = collection->upsert(docs);
    if (!updated.has_value() || !updated.value().front().ok()) std::_Exit(4);
  }
  if (append_suffix) {
    doc.set_pk("broken");
    std::vector<Doc> docs{doc};
    auto inserted = collection->insert(docs);
    if (!inserted.has_value() || !inserted.value().front().ok()) std::_Exit(5);
  }
  std::_Exit(0);
}

void ReadWalDocuments(const std::string &path, std::vector<Doc::Ptr> *docs) {
  const auto wal_path = FindWal(path);
  ASSERT_FALSE(wal_path.empty());
  auto wal = WalFile::Create(wal_path);
  ASSERT_EQ(wal->open(WalOptions{}), 0);
  ASSERT_EQ(wal->prepare_for_read(), 0);
  while (true) {
    auto record = wal->next();
    ASSERT_TRUE(record.has_value()) << record.error().message();
    if (!record.value().has_value()) break;
    const auto &bytes = record.value().value();
    auto doc = Doc::deserialize(reinterpret_cast<const uint8_t *>(bytes.data()),
                                bytes.size());
    ASSERT_NE(doc, nullptr);
    docs->push_back(std::move(doc));
  }
  ASSERT_EQ(wal->close(), 0);
}

// Recreate historical UPSERT records through the serializer and WAL writer so
// the framing and checksums remain valid. New public writes use INSERT/UPDATE.
void RewriteWalAsLegacyUpserts(const std::string &path) {
  std::vector<Doc::Ptr> docs;
  ASSERT_NO_FATAL_FAILURE(ReadWalDocuments(path, &docs));
  ASSERT_FALSE(docs.empty());
  auto wal = WalFile::Create(FindWal(path));
  ASSERT_EQ(wal->remove(), 0);
  WalOptions options;
  options.create_new = true;
  ASSERT_EQ(wal->open(options), 0);
  for (auto &doc : docs) {
    if (doc->pk_ref() == "target") {
      doc->set_operator(Operator::UPSERT);
      // Legacy UPSERT did not record its predecessor's ID.
      doc->set_doc_id(0);
    }
    auto bytes = doc->serialize();
    ASSERT_EQ(wal->append(std::string(bytes.begin(), bytes.end())), 0);
  }
  ASSERT_EQ(wal->flush(), 0);
  ASSERT_EQ(wal->close(), 0);
}

void ExpectOnlyTarget(const Collection::Ptr &collection,
                      const std::string &value) {
  auto fetched = collection->fetch({"target"});
  ASSERT_TRUE(fetched.has_value()) << fetched.error().message();
  ASSERT_NE(fetched.value().at("target"), nullptr);
  EXPECT_EQ(fetched.value().at("target")->get<std::string>("text"), value);

  SearchQuery query;
  query.topk_ = 10;
  query.filter_ = "text != ''";
  auto matches = collection->query(query);
  ASSERT_TRUE(matches.has_value()) << matches.error().message();
  ASSERT_EQ(matches.value().size(), 1u);
  EXPECT_EQ(matches.value().front()->pk_ref(), "target");
  EXPECT_EQ(matches.value().front()->get<std::string>("text"), value);
  auto stats = collection->stats();
  ASSERT_TRUE(stats.has_value()) << stats.error().message();
  EXPECT_EQ(stats.value().doc_count, 1u);
}

class RelaxedValidationDeathTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Re-exec children start with the default style; select threadsafe before
    // InDeathTestChild() interprets their internal death-test flag.
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    // Threadsafe death tests re-exec the fixture. A second crash child must
    // reopen the first child's database instead of deleting it in SetUp.
    if (!::testing::internal::InDeathTestChild()) {
      ailego::FileHelper::RemovePath(path_.c_str());
    }
  }

  void TearDown() override {
    ailego::FileHelper::RemovePath(path_.c_str());
  }

  void CheckLegacyCommittedPredecessor(bool separate_segment) {
    ASSERT_EXIT(WriteUpsertsAndExit(path_, true, false, separate_segment),
                ::testing::ExitedWithCode(0), "");
    if (!::testing::internal::InDeathTestChild()) {
      auto recovered_version = VersionManager::Recovery(path_);
      ASSERT_TRUE(recovered_version.has_value());
      const auto version = recovered_version.value()->get_current_version();
      EXPECT_EQ(version.persisted_segment_metas().empty(), !separate_segment);
      if (!separate_segment) {
        EXPECT_FALSE(
            version.writing_segment_meta()->persisted_blocks().empty());
      }
      ASSERT_EQ(
          version.writing_segment_meta()->writing_forward_block()->min_doc_id(),
          1u);

      // Pin the new writer's format before constructing a historical WAL.
      std::vector<Doc::Ptr> docs;
      ASSERT_NO_FATAL_FAILURE(ReadWalDocuments(path_, &docs));
      ASSERT_EQ(docs.size(), 2u);
      EXPECT_EQ(docs[0]->get_operator(), Operator::UPDATE);
      EXPECT_EQ(docs[0]->doc_id(), 0u);
      EXPECT_EQ(docs[1]->get_operator(), Operator::UPDATE);
      EXPECT_EQ(docs[1]->doc_id(), 1u);
      ASSERT_NO_FATAL_FAILURE(RewriteWalAsLegacyUpserts(path_));

      auto map =
          IDMap::CreateAndOpen("wal_recovery", FindIdMap(path_), false, false);
      ASSERT_NE(map, nullptr);
      // The map can reach disk ahead of the manifest's deletion snapshot.
      // Its latest replay ID no longer identifies committed predecessor ID 0.
      ASSERT_TRUE(map->upsert("target", 2).ok());
      ASSERT_TRUE(map->flush().ok());
    }

    // Neither child flushes its recovered deletion bitmap. Both retries must
    // rediscover ID 0, including when it belongs to another persisted segment.
    for (int attempt = 0; attempt < 2; ++attempt) {
      ASSERT_EXIT(
          {
            auto opened = Collection::Open(path_, CollectionOptions{});
            if (!opened.has_value()) {
              std::cerr << opened.error() << std::endl;
              std::_Exit(1);
            }
            ExpectOnlyTarget(opened.value(), "second");
            std::_Exit(::testing::Test::HasFailure() ? 2 : 0);
          },
          ::testing::ExitedWithCode(0), "");
    }

    {
      auto opened = Collection::Open(path_, CollectionOptions{});
      ASSERT_TRUE(opened.has_value()) << opened.error().message();
      ASSERT_NO_FATAL_FAILURE(ExpectOnlyTarget(opened.value(), "second"));
      ASSERT_TRUE(opened.value()->flush().ok());
    }
    CollectionOptions options;
    options.read_only_ = true;
    auto reopened = Collection::Open(path_, options);
    ASSERT_TRUE(reopened.has_value()) << reopened.error().message();
    ASSERT_NO_FATAL_FAILURE(ExpectOnlyTarget(reopened.value(), "second"));
    auto iterator = reopened.value()->create_iterator();
    ASSERT_TRUE(iterator.has_value()) << iterator.error().message();
    auto first = iterator.value()->next();
    ASSERT_TRUE(first.has_value());
    ASSERT_NE(first.value(), nullptr);
    EXPECT_EQ(first.value()->pk_ref(), "target");
    EXPECT_EQ(first.value()->get<std::string>("text"), "second");
    auto end = iterator.value()->next();
    ASSERT_TRUE(end.has_value());
    EXPECT_EQ(end.value(), nullptr);
  }

  const std::string path_{"relaxed_validation_recovery_db"};
};

TEST_F(RelaxedValidationDeathTest, Utf8AndLongIdsRecoverFromUnflushedWal) {
  const std::vector<std::string> ids{u8"订单:😀",
                                     std::string(1021, 'x') + u8"中", " doc ",
                                     u8"café", u8"cafe\u0301"};
  // Re-exec the child before starting collection threads. Exit without stack
  // unwinding so Collection destruction cannot flush the writing segment.
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  ASSERT_EXIT(
      {
        CollectionSchema schema(u8"恢复 集合");
        if (!schema
                 .add_field(std::make_shared<FieldSchema>(
                     "value", DataType::INT32, false))
                 .ok()) {
          std::_Exit(1);
        }
        auto created =
            Collection::CreateAndOpen(path_, schema, CollectionOptions{});
        if (!created.has_value()) std::_Exit(2);
        auto collection = std::move(created).value();
        std::vector<Doc> docs;
        for (size_t i = 0; i < ids.size(); ++i) {
          Doc doc;
          doc.set_pk(ids[i]);
          doc.set<int32_t>("value", static_cast<int32_t>(i));
          docs.push_back(std::move(doc));
        }
        auto inserted = collection->insert(docs);
        if (!inserted.has_value()) std::_Exit(3);
        for (const auto &status : inserted.value()) {
          if (!status.ok()) std::_Exit(4);
        }
        std::_Exit(0);
      },
      ::testing::ExitedWithCode(0), "");

  auto opened = Collection::Open(path_, CollectionOptions{});
  ASSERT_TRUE(opened.has_value()) << opened.error().message();
  auto collection = std::move(opened).value();
  EXPECT_EQ(collection->schema().value().name(), u8"恢复 集合");
  auto fetched = collection->fetch(ids);
  ASSERT_TRUE(fetched.has_value()) << fetched.error().message();
  ASSERT_EQ(fetched.value().size(), ids.size());
  for (size_t i = 0; i < ids.size(); ++i) {
    const auto found = fetched.value().find(ids[i]);
    ASSERT_NE(found, fetched.value().end());
    ASSERT_NE(found->second, nullptr);
    EXPECT_EQ(found->second->pk_ref(), ids[i]);
    EXPECT_EQ(found->second->get<int32_t>("value"), static_cast<int32_t>(i));
  }
  auto status = collection->flush();
  ASSERT_TRUE(status.ok()) << status.message();
  collection.reset();
  auto reopened = Collection::Open(path_, CollectionOptions{});
  ASSERT_TRUE(reopened.has_value()) << reopened.error().message();
  EXPECT_EQ(reopened.value()->stats().value().doc_count, ids.size());
}

TEST_F(RelaxedValidationDeathTest, LargeStringAndTrailingDocRecoverFromWal) {
  const std::vector<std::string> ids{"prefix", std::string(1024, 'x'),
                                     "suffix"};
  // The same value fits below 4MiB with the old 64-byte ID limit. A 1024-byte
  // ID takes its serialized WAL record over that former reader-only limit.
  const std::vector<std::string> values{"before", std::string(4193700, 'v'),
                                        "after"};
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  ASSERT_EXIT(WriteStringDocsAndExit(path_, ids, values),
              ::testing::ExitedWithCode(0), "");
  for (int reopen = 0; reopen < 2; ++reopen) {
    auto opened = Collection::Open(path_, CollectionOptions{});
    ASSERT_TRUE(opened.has_value()) << opened.error().message();
    auto collection = std::move(opened).value();
    auto fetched = collection->fetch(ids);
    ASSERT_TRUE(fetched.has_value()) << fetched.error().message();
    for (size_t i = 0; i < ids.size(); ++i) {
      auto found = fetched.value().find(ids[i]);
      ASSERT_NE(found, fetched.value().end());
      ASSERT_NE(found->second, nullptr);
      EXPECT_EQ(found->second->pk_ref(), ids[i]);
      EXPECT_EQ(found->second->get<std::string>("text"), values[i]);
    }
    EXPECT_EQ(collection->stats().value().doc_count, ids.size());
    ASSERT_TRUE(collection->flush().ok());
  }
}

TEST_F(RelaxedValidationDeathTest, IncompleteTailAllowsLaterCrashRecovery) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  ASSERT_EXIT(
      WriteStringDocsAndExit(path_, {"prefix", "torn"}, {"before", "tail"}),
      ::testing::ExitedWithCode(0), "");
  if (!::testing::internal::InDeathTestChild()) {
    const auto wal_path = FindWal(path_);
    ASSERT_FALSE(wal_path.empty());
    std::filesystem::resize_file(wal_path,
                                 std::filesystem::file_size(wal_path) - 1);
    const auto tail_bytes = ReadFileBytes(wal_path);
    const auto manifests = ReadManifests(path_);
    ASSERT_FALSE(manifests.empty());
    {
      CollectionOptions options;
      options.read_only_ = true;
      auto opened = Collection::Open(path_, options);
      // Recovery needs writable index stores. A read-only attempt must fail
      // explicitly and leave the WAL and committed manifest untouched.
      ASSERT_FALSE(opened.has_value());
      EXPECT_EQ(opened.error().code(), StatusCode::FAILED_PRECONDITION);
      EXPECT_NE(opened.error().message().find("read-write mode once"),
                std::string::npos);
    }
    EXPECT_EQ(ReadFileBytes(wal_path), tail_bytes);
    EXPECT_EQ(ReadManifests(path_), manifests);
  }

  ASSERT_EXIT(
      {
        auto opened = Collection::Open(path_, CollectionOptions{});
        if (!opened.has_value()) {
          std::cerr << opened.error() << std::endl;
          std::_Exit(1);
        }
        Doc doc;
        doc.set_pk("suffix");
        doc.set<std::string>("text", "after");
        std::vector<Doc> docs{doc};
        auto inserted = opened.value()->insert(docs);
        if (!inserted.has_value() || !inserted.value().front().ok())
          std::_Exit(2);
        std::_Exit(0);
      },
      ::testing::ExitedWithCode(0), "");
  auto opened = Collection::Open(path_, CollectionOptions{});
  ASSERT_TRUE(opened.has_value()) << opened.error().message();
  auto fetched = opened.value()->fetch({"prefix", "torn", "suffix"});
  ASSERT_TRUE(fetched.has_value());
  ASSERT_NE(fetched.value().at("prefix"), nullptr);
  ASSERT_NE(fetched.value().at("suffix"), nullptr);
  EXPECT_TRUE(fetched.value().find("torn") == fetched.value().end() ||
              fetched.value().at("torn") == nullptr);
  EXPECT_EQ(opened.value()->stats().value().doc_count, 2);
}

TEST_F(RelaxedValidationDeathTest, CorruptWalFailsOpenWithoutReplacingFiles) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  ASSERT_EXIT(
      WriteStringDocsAndExit(path_, {"prefix", "broken"}, {"before", "after"}),
      ::testing::ExitedWithCode(0), "");
  const auto wal_path = FindWal(path_);
  ASSERT_FALSE(wal_path.empty());
  {
    std::fstream file(wal_path,
                      std::ios::in | std::ios::out | std::ios::binary);
    ASSERT_TRUE(file.is_open());
    uint32_t first_length;
    file.seekg(64);
    file.read(reinterpret_cast<char *>(&first_length), sizeof(first_length));
    ASSERT_TRUE(file.good());
    // Damage the second record CRC, keeping its complete framing intact.
    file.seekp(64 + 8 + first_length + 4);
    const uint32_t bad_crc = 0;
    file.write(reinterpret_cast<const char *>(&bad_crc), sizeof(bad_crc));
    ASSERT_TRUE(file.good());
  }
  const auto bytes = ReadFileBytes(wal_path);
  const auto manifests = ReadManifests(path_);
  ASSERT_FALSE(manifests.empty());
  for (int attempt = 0; attempt < 2; ++attempt) {
    auto opened = Collection::Open(path_, CollectionOptions{});
    ASSERT_FALSE(opened.has_value());
    EXPECT_NE(opened.error().message().find("CRC mismatch"), std::string::npos);
    EXPECT_EQ(ReadFileBytes(wal_path), bytes);
    EXPECT_EQ(ReadManifests(path_), manifests);
  }
}

TEST_F(RelaxedValidationDeathTest, CorruptTailDoesNotApplyUpsertPrefix) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  ASSERT_EXIT(WriteUpsertsAndExit(path_, true, true),
              ::testing::ExitedWithCode(0), "");
  ASSERT_NO_FATAL_FAILURE(RewriteWalAsLegacyUpserts(path_));
  const auto wal_path = FindWal(path_);
  ASSERT_FALSE(wal_path.empty());
  auto bytes = ReadFileBytes(wal_path);
  size_t prefix_end = 64;
  for (int i = 0; i < 2; ++i) {
    uint32_t length;
    ASSERT_GE(bytes.size() - prefix_end, 8u);
    std::memcpy(&length, bytes.data() + prefix_end, sizeof(length));
    prefix_end += 8 + length;
    ASSERT_LE(prefix_end, bytes.size());
  }
  ASSERT_GE(bytes.size() - prefix_end, 8u);
  bytes[prefix_end + 4] ^= 1;  // Corrupt only the trailing record's CRC.
  {
    std::ofstream file(wal_path, std::ios::binary | std::ios::trunc);
    file.write(bytes.data(), bytes.size());
    ASSERT_TRUE(file.good());
  }
  const auto idmap_path = FindIdMap(path_);
  ASSERT_FALSE(idmap_path.empty());
  {
    auto map = IDMap::CreateAndOpen("wal_recovery", idmap_path, false, false);
    ASSERT_NE(map, nullptr);
    // The original row is committed as ID 0. Make the pre-recovery mapping
    // deterministic even if RocksDB flushed uncommitted writes before exit.
    ASSERT_TRUE(map->upsert("target", 0).ok());
    map->remove("broken");
    ASSERT_TRUE(map->flush().ok());
  }
  const auto manifests = ReadManifests(path_);
  for (int attempt = 0; attempt < 2; ++attempt) {
    auto opened = Collection::Open(path_, CollectionOptions{});
    ASSERT_FALSE(opened.has_value());
    EXPECT_NE(opened.error().message().find("CRC mismatch"), std::string::npos);
    EXPECT_EQ(ReadFileBytes(wal_path), bytes);
    EXPECT_EQ(ReadManifests(path_), manifests);
    auto map = IDMap::CreateAndOpen("wal_recovery", idmap_path, false, true);
    ASSERT_NE(map, nullptr);
    uint64_t original_id;
    ASSERT_TRUE(map->has("target", &original_id));
    EXPECT_EQ(original_id, 0u);
    EXPECT_FALSE(map->has("broken"));
  }

  // Remove the damaged last record and retry the intact UPSERT prefix.
  std::filesystem::resize_file(wal_path, prefix_end);
  {
    auto opened = Collection::Open(path_, CollectionOptions{});
    ASSERT_TRUE(opened.has_value()) << opened.error().message();
    auto fetched = opened.value()->fetch({"target"});
    ASSERT_TRUE(fetched.has_value());
    ASSERT_NE(fetched.value().at("target"), nullptr);
    EXPECT_EQ(fetched.value().at("target")->get<std::string>("text"), "second");
    EXPECT_EQ(opened.value()->stats().value().doc_count, 1);
    Doc update;
    update.set_pk("target");
    update.set<std::string>("text", "third");
    std::vector<Doc> updates{update};
    auto updated = opened.value()->upsert(updates);
    ASSERT_TRUE(updated.has_value());
    ASSERT_TRUE(updated.value().front().ok());
    ASSERT_TRUE(opened.value()->flush().ok());
  }
  auto reopened = Collection::Open(path_, CollectionOptions{});
  ASSERT_TRUE(reopened.has_value());
  auto fetched = reopened.value()->fetch({"target"});
  ASSERT_TRUE(fetched.has_value());
  ASSERT_NE(fetched.value().at("target"), nullptr);
  EXPECT_EQ(fetched.value().at("target")->get<std::string>("text"), "third");
  EXPECT_EQ(reopened.value()->stats().value().doc_count, 1);
}

TEST_F(RelaxedValidationDeathTest, UpsertReplayIgnoresSameAndLaterReplayIds) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  ASSERT_EXIT(WriteUpsertsAndExit(path_, false, false),
              ::testing::ExitedWithCode(0), "");
  if (!::testing::internal::InDeathTestChild()) {
    ASSERT_NO_FATAL_FAILURE(RewriteWalAsLegacyUpserts(path_));
    const auto idmap_path = FindIdMap(path_);
    ASSERT_FALSE(idmap_path.empty());
    auto map = IDMap::CreateAndOpen("wal_recovery", idmap_path, false, false);
    ASSERT_NE(map, nullptr);
    // Simulate a partial previous recovery that persisted the second UPSERT's
    // ID. It is later than the first replay ID and equal to the second.
    ASSERT_TRUE(map->upsert("target", 1).ok());
    ASSERT_TRUE(map->flush().ok());
  }
  ASSERT_EXIT(
      {
        auto opened = Collection::Open(path_, CollectionOptions{});
        if (!opened.has_value()) {
          std::cerr << opened.error() << std::endl;
          std::_Exit(1);
        }
        auto fetched = opened.value()->fetch({"target"});
        if (!fetched.has_value() || !fetched.value().at("target") ||
            fetched.value().at("target")->get<std::string>("text") !=
                "second" ||
            opened.value()->stats().value().doc_count != 1)
          std::_Exit(2);
        Doc update;
        update.set_pk("target");
        update.set<std::string>("text", "third");
        std::vector<Doc> updates{update};
        auto updated = opened.value()->upsert(updates);
        if (!updated.has_value() || !updated.value().front().ok())
          std::_Exit(3);
        std::_Exit(0);
      },
      ::testing::ExitedWithCode(0), "");
  auto reopened = Collection::Open(path_, CollectionOptions{});
  ASSERT_TRUE(reopened.has_value()) << reopened.error().message();
  auto fetched = reopened.value()->fetch({"target"});
  ASSERT_TRUE(fetched.has_value());
  ASSERT_NE(fetched.value().at("target"), nullptr);
  EXPECT_EQ(fetched.value().at("target")->get<std::string>("text"), "third");
  EXPECT_EQ(reopened.value()->stats().value().doc_count, 1);
}

TEST_F(RelaxedValidationDeathTest,
       LegacyUpsertsRecoverCommittedPredecessorInWritingSegment) {
  ASSERT_NO_FATAL_FAILURE(CheckLegacyCommittedPredecessor(false));
}

TEST_F(RelaxedValidationDeathTest,
       LegacyUpsertsRecoverCommittedPredecessorInPersistedSegment) {
  ASSERT_NO_FATAL_FAILURE(CheckLegacyCommittedPredecessor(true));
}

TEST_F(RelaxedValidationDeathTest, UpsertWalRecordsInsertOrUpdatePredecessor) {
  ASSERT_EXIT(WriteUpsertsAndExit(path_, false, false),
              ::testing::ExitedWithCode(0), "");
  std::vector<Doc::Ptr> docs;
  ASSERT_NO_FATAL_FAILURE(ReadWalDocuments(path_, &docs));
  ASSERT_EQ(docs.size(), 2u);
  EXPECT_EQ(docs[0]->get_operator(), Operator::INSERT);
  EXPECT_EQ(docs[0]->pk_ref(), "target");
  EXPECT_EQ(docs[0]->get<std::string>("text"), "first");
  EXPECT_EQ(docs[1]->get_operator(), Operator::UPDATE);
  EXPECT_EQ(docs[1]->doc_id(), 0u);
  EXPECT_EQ(docs[1]->get<std::string>("text"), "second");
}

}  // namespace
}  // namespace zvec
