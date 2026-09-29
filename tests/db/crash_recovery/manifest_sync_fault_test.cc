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

// Manifest publication under sync failures (injected through
// VersionManager::SetSyncFaultForTest) and validation of manifests with
// multi-byte segment ids:
//
// * a failed file sync is a clean failure: nothing is published;
// * a failed directory sync after the rename is an uncertain publish: the new
//   manifest and everything it references stay, and the collection refuses
//   further writes, including queued writers and the rest of the batch that
//   raised the failure, until it is reopened;
// * a writable open makes the loaded manifest durable before it modifies
//   anything, and fails if it cannot;
// * every truncation of a manifest is rejected, also when its ids and
//   next_segment_id take several varint bytes.

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/db/collection.h>
#include <zvec/db/doc.h>
#include <zvec/db/schema.h>
#include "db/common/file_helper.h"
#include "db/index/common/manifest_codec.h"
#include "db/index/common/meta.h"
#include "db/index/common/version_manager.h"

namespace zvec {
namespace {

namespace fs = std::filesystem;

constexpr int kDocs = 64;

std::string ReadFile(const fs::path &path) {
  std::ifstream stream(path, std::ios::binary);
  std::ostringstream buffer;
  buffer << stream.rdbuf();
  return buffer.str();
}

void WriteFile(const fs::path &path, const std::string &data) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream.write(data.data(), static_cast<std::streamsize>(data.size()));
  stream.close();
  ASSERT_TRUE(stream.good()) << path;
}

std::set<std::string> FileNames(const fs::path &dir) {
  std::set<std::string> names;
  for (const auto &entry : fs::directory_iterator(dir)) {
    if (entry.is_regular_file()) {
      names.insert(entry.path().filename().string());
    }
  }
  return names;
}

std::vector<fs::path> Manifests(const fs::path &dir) {
  std::vector<fs::path> manifests;
  for (const auto &entry : fs::directory_iterator(dir)) {
    const auto name = entry.path().filename().string();
    if (name.rfind("manifest.", 0) == 0 &&
        name.find_first_not_of("0123456789", 9) == std::string::npos) {
      manifests.push_back(entry.path());
    }
  }
  return manifests;
}

CollectionSchema MakeSchema() {
  CollectionSchema schema("manifest_sync_fault");
  schema.set_max_doc_count_per_segment(1000);
  schema.add_field(
      std::make_shared<FieldSchema>("title", DataType::STRING, false));
  schema.add_field(std::make_shared<FieldSchema>(
      "vec", DataType::VECTOR_FP32, 4, false,
      std::make_shared<FlatIndexParams>(MetricType::L2)));
  return schema;
}

Doc MakeDoc(int id) {
  Doc doc;
  doc.set_pk("pk_" + std::to_string(id));
  doc.set<std::string>("title", "title " + std::to_string(id));
  doc.set<std::vector<float>>("vec", {float(id), 1.0f, 0.0f, 0.5f});
  return doc;
}

// A version whose segment ids and next_segment_id need 1, 2 and 3 varint
// bytes.
Version MakeVersion(SegmentID writing_id) {
  Version version;
  version.set_schema(MakeSchema());
  SegmentID persisted_ids[] = {5, 300, 70000};
  uint64_t min_doc_id = uint64_t{1} << 20;
  for (SegmentID id : persisted_ids) {
    auto meta = std::make_shared<SegmentMeta>(id);
    meta->add_persisted_block(BlockMeta(id, BlockType::SCALAR, min_doc_id,
                                        min_doc_id + 999, 1000, {"title"}));
    min_doc_id += 1000;
    EXPECT_TRUE(version.add_persisted_segment_meta(meta).ok());
  }
  version.reset_writing_segment_meta(std::make_shared<SegmentMeta>(writing_id));
  version.set_id_map_path_suffix(130);
  version.set_delete_snapshot_path_suffix(20000);
  version.set_next_segment_id(writing_id + 1);
  return version;
}

int g_file_sync_error = 0;
int g_directory_sync_error = 0;
// When set, replaces the two fixed errors above.
std::function<int(bool)> g_sync_fault;

int InjectSyncFault(bool directory) {
  if (g_sync_fault) {
    return g_sync_fault(directory);
  }
  return directory ? g_directory_sync_error : g_file_sync_error;
}

// Every entry under `root` with a digest of its contents ("dir" for
// directories).
std::map<std::string, std::string> Snapshot(const fs::path &root) {
  std::map<std::string, std::string> entries;
  for (const auto &entry : fs::recursive_directory_iterator(root)) {
    const std::string data = entry.is_directory() ? "" : ReadFile(entry.path());
    entries[fs::relative(entry.path(), root).string()] =
        entry.is_directory()
            ? "dir"
            : std::to_string(data.size()) + ":" +
                  std::to_string(std::hash<std::string>()(data));
  }
  return entries;
}

// True when every segment that some manifest in `path` references exists.
bool AllManifestDependenciesExist(const fs::path &path) {
  for (const auto &manifest : Manifests(path)) {
    Version version;
    if (!Version::Load(manifest.string(), &version).ok() ||
        !version.validate().ok()) {
      return false;
    }
    std::vector<SegmentID> ids{version.writing_segment_meta()->id()};
    for (const auto &meta : version.persisted_segment_metas()) {
      ids.push_back(meta->id());
    }
    for (SegmentID id : ids) {
      if (!fs::is_directory(path / std::to_string(id))) {
        return false;
      }
    }
  }
  return true;
}

// Every document whose write returned OK must be readable, unchanged.
void ExpectAcknowledgedDocs(Collection &collection,
                            const std::vector<int> &ids) {
  std::vector<std::string> keys;
  for (int id : ids) keys.push_back("pk_" + std::to_string(id));
  auto fetched = collection.fetch(keys);
  ASSERT_TRUE(fetched.has_value()) << fetched.error().message();
  for (int id : ids) {
    auto found = fetched->find("pk_" + std::to_string(id));
    ASSERT_TRUE(found != fetched->end()) << id;
    ASSERT_TRUE(found->second) << "Acknowledged document lost: pk_" << id;
    EXPECT_EQ(*found->second, MakeDoc(id));
  }
}

class ManifestSyncFaultTest : public ::testing::Test {
 protected:
  void SetUp() override {
    root_ = fs::absolute("manifest_sync_fault_test_db") /
            ::testing::UnitTest::GetInstance()->current_test_info()->name();
    fs::remove_all(root_);
    fs::create_directories(root_);
    g_file_sync_error = 0;
    g_directory_sync_error = 0;
    g_sync_fault = nullptr;
    VersionManager::SetSyncFaultForTest(&InjectSyncFault);
  }

  void TearDown() override {
    VersionManager::SetSyncFaultForTest(nullptr);
    g_sync_fault = nullptr;
    std::error_code ec;
    fs::remove_all(root_, ec);
  }

  fs::path root_;
};

// A failed sync of the temporary manifest publishes nothing: the previous
// manifest stays, no temporary file is left and a later flush succeeds.
TEST_F(ManifestSyncFaultTest, FileSyncFailurePublishesNothing) {
  auto created = VersionManager::Create(root_.string(), MakeVersion(70001));
  ASSERT_TRUE(created.has_value());
  auto manager = created.value();
  ASSERT_TRUE(manager->flush().ok());
  const std::string published = ReadFile(root_ / "manifest.0");

  manager->set_next_segment_id(70005);
  g_file_sync_error = EIO;
  auto status = manager->flush();
  EXPECT_FALSE(status.ok());
  EXPECT_FALSE(manager->fenced());
  EXPECT_EQ(FileNames(root_), std::set<std::string>{"manifest.0"});
  EXPECT_EQ(ReadFile(root_ / "manifest.0"), published);

  g_file_sync_error = 0;
  ASSERT_TRUE(manager->flush().ok());
  EXPECT_EQ(FileNames(root_), std::set<std::string>{"manifest.1"});
  auto recovered = VersionManager::Recovery(root_.string(), true);
  ASSERT_TRUE(recovered.has_value()) << recovered.error().message();
  EXPECT_EQ(recovered.value()->get_current_version().next_segment_id(), 70005u);
}

// A failed directory sync after the rename leaves the outcome unknown: the
// new manifest must stay (unlinking it is not durable either), the previous
// one must stay, and no later publish may happen until reopen.
TEST_F(ManifestSyncFaultTest, DirectorySyncFailureKeepsManifestAndFences) {
  auto created = VersionManager::Create(root_.string(), MakeVersion(70001));
  ASSERT_TRUE(created.has_value());
  auto manager = created.value();
  ASSERT_TRUE(manager->flush().ok());

  manager->set_next_segment_id(70005);
  g_directory_sync_error = EIO;
  auto status = manager->flush();
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(manager->fenced());
  EXPECT_EQ(FileNames(root_),
            (std::set<std::string>{"manifest.0", "manifest.1"}));

  g_directory_sync_error = 0;
  status = manager->flush();
  EXPECT_FALSE(status.ok()) << "flush() published after an uncertain publish";
  EXPECT_EQ(FileNames(root_),
            (std::set<std::string>{"manifest.0", "manifest.1"}));

  // Reopening loads the newest manifest.
  auto recovered = VersionManager::Recovery(root_.string(), true);
  ASSERT_TRUE(recovered.has_value()) << recovered.error().message();
  EXPECT_EQ(recovered.value()->get_current_version().next_segment_id(), 70005u);
  EXPECT_FALSE(recovered.value()->fenced());
}

// An uncertain publish during a schema change must not destroy the new
// writing segment that the new manifest references, and the collection must
// refuse writes until it is reopened. Reopening then loses nothing.
TEST_F(ManifestSyncFaultTest, DirectorySyncFailureDuringCreateIndexKeepsData) {
  const auto path = root_ / "collection";
  {
    auto created = Collection::CreateAndOpen(path.string(), MakeSchema(),
                                             CollectionOptions{});
    ASSERT_TRUE(created.has_value()) << created.error().message();
    auto collection = created.value();
    for (int id = 0; id < kDocs; ++id) {
      std::vector<Doc> docs{MakeDoc(id)};
      auto result = collection->insert(docs);
      ASSERT_TRUE(result.has_value() && result->front().ok());
    }
    // Seal the documents so that create_index publishes a single manifest.
    ASSERT_TRUE(collection->create_iterator().has_value());

    g_directory_sync_error = EIO;
    auto status = collection->create_index(
        "title", std::make_shared<InvertIndexParams>(), CreateIndexOptions{});
    EXPECT_FALSE(status.ok()) << "create_index() ignored the failed sync";
    g_directory_sync_error = 0;

    // Every manifest on disk, which a crash could expose, must reference
    // only segments that still exist.
    for (const auto &manifest : Manifests(path)) {
      Version version;
      ASSERT_TRUE(Version::Load(manifest.string(), &version).ok()) << manifest;
      ASSERT_TRUE(version.validate().ok()) << manifest;
      std::vector<SegmentID> ids{version.writing_segment_meta()->id()};
      for (const auto &meta : version.persisted_segment_metas()) {
        ids.push_back(meta->id());
      }
      for (SegmentID id : ids) {
        EXPECT_TRUE(fs::is_directory(path / std::to_string(id)))
            << manifest << " references removed segment " << id;
      }
    }

    std::vector<Doc> docs{MakeDoc(kDocs)};
    auto result = collection->insert(docs);
    EXPECT_FALSE(result.has_value() && result->front().ok())
        << "Write accepted after an uncertain manifest publish";
    EXPECT_FALSE(collection->flush().ok());
    collection->close();
  }

  // The newest manifest may still not be durable. A writable open whose
  // sync of it fails must not open and must not modify anything.
  {
    const auto before = Snapshot(path);
    g_file_sync_error = EIO;
    auto opened = Collection::Open(path.string(), CollectionOptions{});
    g_file_sync_error = 0;
    EXPECT_FALSE(opened.has_value())
        << "Writable open without a durable manifest";
    EXPECT_EQ(Snapshot(path), before) << "The failed open modified files";
  }

  // A writable open syncs the manifest and its directory first. Until then,
  // what every manifest on disk references must still exist, so that a
  // power loss at any point before the sync recovers either generation.
  int file_syncs = 0;
  int directory_syncs = 0;
  bool dependencies_kept = true;
  g_sync_fault = [&](bool directory) {
    if (file_syncs + directory_syncs == 0) {
      dependencies_kept = AllManifestDependenciesExist(path);
    }
    ++(directory ? directory_syncs : file_syncs);
    return 0;
  };
  auto reopened = Collection::Open(path.string(), CollectionOptions{});
  g_sync_fault = nullptr;
  EXPECT_GE(file_syncs, 1);
  EXPECT_GE(directory_syncs, 1);
  EXPECT_TRUE(dependencies_kept)
      << "Open modified the collection before the manifest was durable";
  ASSERT_TRUE(reopened.has_value()) << reopened.error().message();
  auto stats = reopened.value()->stats();
  ASSERT_TRUE(stats.has_value());
  EXPECT_EQ(stats->doc_count, static_cast<uint64_t>(kDocs));
  std::vector<std::string> keys;
  for (int id = 0; id < kDocs; ++id) keys.push_back("pk_" + std::to_string(id));
  auto fetched = reopened.value()->fetch(keys);
  ASSERT_TRUE(fetched.has_value());
  for (int id = 0; id < kDocs; ++id) {
    auto found = fetched->find("pk_" + std::to_string(id));
    ASSERT_TRUE(found != fetched->end() && found->second) << id;
    EXPECT_EQ(*found->second, MakeDoc(id));
  }
  // Writes work again after the reopen.
  std::vector<Doc> docs{MakeDoc(kDocs)};
  auto result = reopened.value()->insert(docs);
  ASSERT_TRUE(result.has_value() && result->front().ok());
  EXPECT_TRUE(reopened.value()->flush().ok());
}

// An automatic buffer flush inside a batch hits an uncertain publish. The
// writing block has moved on but its WAL has not, so the rest of the batch
// must fail; every document acknowledged before must survive a reopen.
TEST_F(ManifestSyncFaultTest, FenceRaisedMidBatchFailsTheRestOfTheBatch) {
  const auto path = root_ / "collection";
  constexpr int kBatch = 400;
  std::vector<int> acknowledged;
  {
    auto created = Collection::CreateAndOpen(
        path.string(), MakeSchema(), CollectionOptions{false, true, 4096});
    ASSERT_TRUE(created.has_value()) << created.error().message();
    auto collection = created.value();

    std::vector<Doc> docs;
    for (int id = 0; id < kBatch; ++id) docs.push_back(MakeDoc(id));
    g_directory_sync_error = EIO;
    auto results = collection->insert(docs);
    g_directory_sync_error = 0;
    ASSERT_TRUE(results.has_value()) << results.error().message();
    ASSERT_EQ(results->size(), static_cast<size_t>(kBatch));
    int first_failure = -1;
    for (int id = 0; id < kBatch; ++id) {
      if ((*results)[id].ok()) {
        acknowledged.push_back(id);
        EXPECT_EQ(first_failure, -1)
            << "pk_" << id << " acknowledged after a failure in its batch";
      } else if (first_failure < 0) {
        first_failure = id;
      }
    }
    ASSERT_GT(first_failure, 0) << "The buffer never filled";
    collection->close();
  }

  auto reopened = Collection::Open(path.string(), CollectionOptions{});
  ASSERT_TRUE(reopened.has_value()) << reopened.error().message();
  ASSERT_NO_FATAL_FAILURE(
      ExpectAcknowledgedDocs(*reopened.value(), acknowledged));
}

// A writer that is already waiting for the write lock when another batch
// raises the fence must not write either.
TEST_F(ManifestSyncFaultTest, FenceRaisedWhileWriterQueuedFailsTheWriter) {
  const auto path = root_ / "collection";
  constexpr int kBatch = 400;
  constexpr int kQueuedId = 100000;
  std::vector<int> acknowledged;
  {
    auto created = Collection::CreateAndOpen(
        path.string(), MakeSchema(), CollectionOptions{false, true, 4096});
    ASSERT_TRUE(created.has_value()) << created.error().message();
    auto collection = created.value();

    std::thread queued;
    std::atomic<bool> queued_ok{false};
    // The first directory sync happens inside the batch's buffer flush,
    // under the write lock. Start the second writer there, give it time to
    // pass its entry checks and block on the lock, then fail the sync.
    g_sync_fault = [&](bool directory) {
      if (!directory) return 0;
      if (!queued.joinable()) {
        queued = std::thread([&] {
          std::vector<Doc> docs{MakeDoc(kQueuedId)};
          auto result = collection->insert(docs);
          queued_ok = result.has_value() && result->front().ok();
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
      }
      return EIO;
    };
    std::vector<Doc> docs;
    for (int id = 0; id < kBatch; ++id) docs.push_back(MakeDoc(id));
    auto results = collection->insert(docs);
    ASSERT_TRUE(queued.joinable()) << "The buffer never filled";
    queued.join();
    g_sync_fault = nullptr;
    ASSERT_TRUE(results.has_value()) << results.error().message();
    for (int id = 0; id < kBatch; ++id) {
      if ((*results)[id].ok()) acknowledged.push_back(id);
    }
    EXPECT_FALSE(queued_ok) << "A queued writer wrote past the fence";
    if (queued_ok) acknowledged.push_back(kQueuedId);
    collection->close();
  }

  auto reopened = Collection::Open(path.string(), CollectionOptions{});
  ASSERT_TRUE(reopened.has_value()) << reopened.error().message();
  ASSERT_NO_FATAL_FAILURE(
      ExpectAcknowledgedDocs(*reopened.value(), acknowledged));
}

// A collection whose next_segment_id is the largest SegmentID cannot
// allocate another segment. The allocator must refuse instead of wrapping
// to 0, which would publish a manifest that no longer validates.
TEST_F(ManifestSyncFaultTest, SegmentIdExhaustionKeepsCollectionOpenable) {
  const auto path = root_ / "collection";
  {
    auto created = Collection::CreateAndOpen(path.string(), MakeSchema(),
                                             CollectionOptions{});
    ASSERT_TRUE(created.has_value()) << created.error().message();
    ASSERT_TRUE(created.value()->close().ok());
  }
  // Raise next_segment_id to the largest value the manifest can hold.
  {
    auto recovered = VersionManager::Recovery(path.string(), true);
    ASSERT_TRUE(recovered.has_value()) << recovered.error().message();
    recovered.value()->set_next_segment_id(
        std::numeric_limits<SegmentID>::max());
    ASSERT_TRUE(recovered.value()->flush().ok());
  }
  {
    auto opened = Collection::Open(path.string(), CollectionOptions{});
    ASSERT_TRUE(opened.has_value()) << opened.error().message();
    std::vector<Doc> docs{MakeDoc(0)};
    auto result = opened.value()->insert(docs);
    ASSERT_TRUE(result.has_value() && result->front().ok());
    // Sealing the writing segment needs a new segment id.
    auto iterator = opened.value()->create_iterator();
    EXPECT_FALSE(iterator.has_value())
        << "A segment id was allocated past the end of the id space";
    opened.value()->close();
  }
  auto reopened = Collection::Open(path.string(), CollectionOptions{});
  ASSERT_TRUE(reopened.has_value()) << reopened.error().message();
  auto stats = reopened.value()->stats();
  ASSERT_TRUE(stats.has_value());
  EXPECT_EQ(stats->doc_count, 1u);
}

// A file system that cannot sync directories reports EINVAL, ENOTSUP or
// EOPNOTSUPP. That is not an uncertain publish: nothing is fenced, older
// manifests are retired, and writable opens work.
TEST_F(ManifestSyncFaultTest, UnsupportedDirectorySyncIsNotAFailure) {
  for (int error : {EINVAL, ENOTSUP, EOPNOTSUPP}) {
    SCOPED_TRACE(std::strerror(error));
    const auto dir = root_ / ("manifests-" + std::to_string(error));
    fs::create_directories(dir);
    auto created = VersionManager::Create(dir.string(), MakeVersion(70001));
    ASSERT_TRUE(created.has_value());
    auto manager = created.value();
    ASSERT_TRUE(manager->flush().ok());
    manager->set_next_segment_id(70005);
    g_directory_sync_error = error;
    auto status = manager->flush();
    g_directory_sync_error = 0;
    EXPECT_TRUE(status.ok()) << status.message();
    EXPECT_FALSE(manager->fenced());
    EXPECT_EQ(FileNames(dir), std::set<std::string>{"manifest.1"});
  }

  const auto path = root_ / "collection";
  g_directory_sync_error = EINVAL;
  {
    auto created = Collection::CreateAndOpen(path.string(), MakeSchema(),
                                             CollectionOptions{});
    ASSERT_TRUE(created.has_value()) << created.error().message();
    std::vector<Doc> docs{MakeDoc(0)};
    auto result = created.value()->insert(docs);
    ASSERT_TRUE(result.has_value() && result->front().ok());
    EXPECT_TRUE(created.value()->flush().ok());
    EXPECT_TRUE(created.value()->close().ok());
  }
  auto reopened = Collection::Open(path.string(), CollectionOptions{});
  ASSERT_TRUE(reopened.has_value()) << reopened.error().message();
  std::vector<Doc> docs{MakeDoc(1)};
  auto result = reopened.value()->insert(docs);
  EXPECT_TRUE(result.has_value() && result->front().ok());
  EXPECT_TRUE(reopened.value()->flush().ok());
  g_directory_sync_error = 0;
}

// A clean failure (before the rename) of the manifest publish during segment
// rollover leaves the new writing segment unreferenced, and writes continue
// into its WAL. After a crash, a writable open must refuse to hide those
// acknowledged writes: it fails, names the directory and modifies nothing.
TEST_F(ManifestSyncFaultTest, FailedRolloverPublishThenCrashFailsClosed) {
  const auto path = root_ / "collection";
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  // The child exits with 0 after acknowledging pk_0..pk_59 and crashing
  // (no close), 3/5 when a write fails, 4 when the rollover did not fail.
  EXPECT_EXIT(
      {
        auto collection = Collection::CreateAndOpen(path.string(), MakeSchema(),
                                                    CollectionOptions{})
                              .value();
        for (int id = 0; id < 50; ++id) {
          std::vector<Doc> docs{MakeDoc(id)};
          auto result = collection->insert(docs);
          if (!(result.has_value() && result->front().ok())) std::_Exit(3);
        }
        // The first file sync publishes the sealed segment's block, the
        // second the rollover to the new writing segment.
        int file_syncs = 0;
        g_sync_fault = [&](bool directory) {
          return !directory && ++file_syncs == 2 ? ENOSPC : 0;
        };
        auto iterator = collection->create_iterator();
        g_sync_fault = nullptr;
        if (iterator.has_value()) std::_Exit(4);
        for (int id = 50; id < 60; ++id) {
          std::vector<Doc> docs{MakeDoc(id)};
          auto result = collection->insert(docs);
          if (!(result.has_value() && result->front().ok())) std::_Exit(5);
        }
        std::_Exit(0);
      },
      ::testing::ExitedWithCode(0), "");

  const auto before = Snapshot(path);
  auto opened = Collection::Open(path.string(), CollectionOptions{});
  if (opened.has_value()) {
    std::vector<std::string> keys;
    for (int id = 0; id < 60; ++id) keys.push_back("pk_" + std::to_string(id));
    auto fetched = opened.value()->fetch(keys);
    int missing = 0;
    for (const auto &key : keys) {
      missing +=
          !(fetched.has_value() && fetched->count(key) && fetched->at(key));
    }
    FAIL() << "Writable open succeeded with " << missing
           << " of 60 acknowledged documents missing";
  }
  EXPECT_NE(opened.error().message().find("holds WAL records"),
            std::string::npos)
      << opened.error().message();
  EXPECT_EQ(Snapshot(path), before) << "The failed open modified files";
}

// Truncating a manifest at any length must be rejected, including cuts at a
// field boundary of an encoding whose varints take several bytes.
TEST_F(ManifestSyncFaultTest, MultiByteIdsEveryTruncationIsRejected) {
  const auto source = root_ / "source";
  fs::create_directories(source);
  auto created = VersionManager::Create(source.string(), MakeVersion(70001));
  ASSERT_TRUE(created.has_value());
  ASSERT_TRUE(created.value()->flush().ok());
  const std::string published = ReadFile(source / "manifest.0");

  const auto target = root_ / "target";
  fs::create_directories(target);
  size_t decodable = 0;
  for (size_t length = 0; length < published.size(); ++length) {
    SCOPED_TRACE("cut to " + std::to_string(length) + " of " +
                 std::to_string(published.size()) + " bytes");
    const std::string cut = published.substr(0, length);
    ManifestData data;
    if (ManifestCodec::Decode(cut, &data).ok()) {
      ++decodable;
    }
    ASSERT_NO_FATAL_FAILURE(WriteFile(target / "manifest.0", cut));
    EXPECT_FALSE(VersionManager::Recovery(target.string(), true).has_value());
  }
  // The cuts that still decode are the ones only validation can catch.
  EXPECT_GE(decodable, 5u);

  ASSERT_NO_FATAL_FAILURE(WriteFile(target / "manifest.0", published));
  auto recovered = VersionManager::Recovery(target.string(), true);
  ASSERT_TRUE(recovered.has_value()) << recovered.error().message();
  auto version = recovered.value()->get_current_version();
  EXPECT_EQ(version.persisted_segment_metas().size(), 3u);
  EXPECT_EQ(version.writing_segment_meta()->id(), 70001u);
  EXPECT_EQ(version.next_segment_id(), 70002u);
  EXPECT_EQ(version.delete_snapshot_path_suffix(), 20000u);
}

}  // namespace
}  // namespace zvec
