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

// Manifest publication and recovery:
//
// * a damaged manifest (empty, or cut at any length) must never make open
//   delete data: open falls back to a complete previous generation, or fails
//   without modifying anything;
// * leftover temporary manifests are ignored and cleaned up;
// * a short write while publishing a manifest makes flush() fail and keeps
//   the previous generation, so no acknowledged data is lost.

#include <sys/resource.h>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/db/collection.h>
#include <zvec/db/doc.h>
#include <zvec/db/schema.h>
#include "db/common/file_helper.h"
#include "db/index/common/meta.h"
#include "db/index/common/version_manager.h"

namespace zvec {
namespace {

namespace fs = std::filesystem;

constexpr int kSegments = 3;
constexpr int kDocsPerSegment = 32;
constexpr int kDocs = kSegments * kDocsPerSegment;

std::string PrimaryKey(int id) {
  return "pk_" + std::to_string(id);
}

CollectionSchema MakeSchema() {
  CollectionSchema schema("manifest_recovery");
  schema.set_max_doc_count_per_segment(1000);
  schema.add_field(
      std::make_shared<FieldSchema>("generation", DataType::INT32, false,
                                    std::make_shared<InvertIndexParams>()));
  schema.add_field(
      std::make_shared<FieldSchema>("text", DataType::STRING, false));
  schema.add_field(std::make_shared<FieldSchema>(
      "vec", DataType::VECTOR_FP32, 4, false,
      std::make_shared<FlatIndexParams>(MetricType::L2)));
  return schema;
}

Doc MakeDoc(int id) {
  Doc doc;
  doc.set_pk(PrimaryKey(id));
  doc.set<int32_t>("generation", id % 7);
  doc.set<std::string>("text", "text " + std::to_string(id));
  doc.set<std::vector<float>>("vec", {float(id), 1.0f, 0.0f, 0.5f});
  return doc;
}

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

fs::path ManifestPath(const fs::path &collection, uint64_t id) {
  return collection / ("manifest." + std::to_string(id));
}

// Ids of the published manifests ("manifest.<id>") in `collection`.
std::vector<uint64_t> ManifestIds(const fs::path &collection) {
  std::vector<uint64_t> ids;
  for (const auto &entry : fs::directory_iterator(collection)) {
    const auto name = entry.path().filename().string();
    if (name.rfind("manifest.", 0) != 0) continue;
    const auto suffix = name.substr(9);
    if (suffix.empty() ||
        suffix.find_first_not_of("0123456789") != std::string::npos)
      continue;
    ids.push_back(std::stoull(suffix));
  }
  std::sort(ids.begin(), ids.end());
  return ids;
}

// Every entry under `root` with its size (-1 for directories), to show that
// a failed open modified nothing.
std::map<std::string, int64_t> Snapshot(const fs::path &root) {
  std::map<std::string, int64_t> entries;
  for (const auto &entry : fs::recursive_directory_iterator(root)) {
    entries[fs::relative(entry.path(), root).string()] =
        entry.is_directory() ? -1 : static_cast<int64_t>(entry.file_size());
  }
  return entries;
}

void InsertDoc(Collection &collection, int id) {
  std::vector<Doc> docs{MakeDoc(id)};
  auto result = collection.insert(docs);
  ASSERT_TRUE(result.has_value()) << result.error().message();
  ASSERT_EQ(result->size(), 1u);
  ASSERT_TRUE(result->front().ok()) << result->front().message();
}

// Checks that `collection` holds exactly documents [0, count).
void ExpectDocs(Collection &collection, int count) {
  auto stats = collection.stats();
  ASSERT_TRUE(stats.has_value()) << stats.error().message();
  EXPECT_EQ(stats->doc_count, static_cast<uint64_t>(count));
  std::vector<std::string> keys;
  for (int id = 0; id < count; ++id) keys.push_back(PrimaryKey(id));
  auto fetched = collection.fetch(keys);
  ASSERT_TRUE(fetched.has_value()) << fetched.error().message();
  for (int id = 0; id < count; ++id) {
    auto found = fetched->find(PrimaryKey(id));
    ASSERT_NE(found, fetched->end()) << PrimaryKey(id);
    ASSERT_NE(found->second, nullptr) << "Missing document " << PrimaryKey(id);
    EXPECT_EQ(*found->second, MakeDoc(id)) << PrimaryKey(id);
  }
}

CollectionOptions Options(bool read_only) {
  CollectionOptions options;
  options.read_only_ = read_only;
  return options;
}

// Sets RLIMIT_FSIZE while alive, so that writes past `limit` bytes of any
// regular file fail with EFBIG after a short write, as on a full disk.
class FileSizeLimit {
 public:
  explicit FileSizeLimit(rlim_t limit) {
    previous_handler_ = std::signal(SIGXFSZ, SIG_IGN);
    getrlimit(RLIMIT_FSIZE, &previous_);
    struct rlimit limited = previous_;
    limited.rlim_cur = limit;
    ok_ = setrlimit(RLIMIT_FSIZE, &limited) == 0;
  }
  ~FileSizeLimit() {
    setrlimit(RLIMIT_FSIZE, &previous_);
    std::signal(SIGXFSZ, previous_handler_);
  }
  bool ok() const {
    return ok_;
  }

 private:
  struct rlimit previous_{};
  void (*previous_handler_)(int) = nullptr;
  bool ok_ = false;
};

class ManifestRecoveryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    root_ = fs::absolute("manifest_recovery_test_db") /
            ::testing::UnitTest::GetInstance()->current_test_info()->name();
    fs::remove_all(root_);
    fs::create_directories(root_);
  }

  void TearDown() override {
    std::error_code ec;
    fs::remove_all(root_, ec);
  }

  // Creates kSegments segments: all but the last sealed through
  // create_iterator(), the last one a flushed writing segment. When `older`
  // is given, it receives the manifest and WAL files as they were right
  // before the final flush. Together with the flushed collection they form
  // the state of a crash after the final manifest was published and before
  // the previous generation was retired.
  void CreateCollection(const fs::path &path, const fs::path &older = {}) {
    auto created =
        Collection::CreateAndOpen(path.string(), MakeSchema(), Options(false));
    ASSERT_TRUE(created.has_value()) << created.error().message();
    auto collection = created.value();
    for (int begin = 0; begin < kDocs; begin += kDocsPerSegment) {
      for (int id = begin; id < begin + kDocsPerSegment; ++id) {
        ASSERT_NO_FATAL_FAILURE(InsertDoc(*collection, id));
      }
      if (begin + kDocsPerSegment < kDocs) {
        auto iterator = collection->create_iterator();
        ASSERT_TRUE(iterator.has_value()) << iterator.error().message();
      }
    }
    if (!older.empty()) {
      for (const auto &entry : fs::recursive_directory_iterator(path)) {
        const auto name = entry.path().filename().string();
        if (!entry.is_regular_file() || (name.rfind("manifest.", 0) != 0 &&
                                         entry.path().extension() != ".wal"))
          continue;
        const auto target = older / fs::relative(entry.path(), path);
        fs::create_directories(target.parent_path());
        fs::copy_file(entry.path(), target);
      }
    }
    auto status = collection->flush();
    ASSERT_TRUE(status.ok()) << status.message();
    status = collection->close();
    ASSERT_TRUE(status.ok()) << status.message();
  }

  // Opens `path` writable, which is when recovery may delete orphans, and
  // expects every document to be there.
  void ExpectOpensWithAllDocs(const fs::path &path) {
    auto opened = Collection::Open(path.string(), Options(false));
    ASSERT_TRUE(opened.has_value()) << opened.error().message();
    ASSERT_NO_FATAL_FAILURE(ExpectDocs(*opened.value(), kDocs));
    ASSERT_TRUE(opened.value()->close().ok());
  }

  // Expects both a writable and a read-only open of `path` to fail without
  // changing any file.
  void ExpectOpenFailsWithoutChanges(const fs::path &path) {
    const auto before = Snapshot(path);
    {
      auto opened = Collection::Open(path.string(), Options(false));
      EXPECT_FALSE(opened.has_value()) << "Writable open accepted the damage";
    }
    ASSERT_EQ(Snapshot(path), before) << "Writable open modified files";
    {
      auto opened = Collection::Open(path.string(), Options(true));
      EXPECT_FALSE(opened.has_value()) << "Read-only open accepted the damage";
    }
    ASSERT_EQ(Snapshot(path), before) << "Read-only open modified files";
  }

  fs::path root_;
};

// The newest (and only) manifest is empty, as after a power loss before its
// data reached the disk: open must fail and delete nothing.
TEST_F(ManifestRecoveryTest, EmptyManifestFailsClosedWithoutDeleting) {
  const auto path = root_ / "collection";
  ASSERT_NO_FATAL_FAILURE(CreateCollection(path));
  const auto ids = ManifestIds(path);
  ASSERT_EQ(ids.size(), 1u);
  const auto manifest = ManifestPath(path, ids.back());
  const std::string published = ReadFile(manifest);

  fs::resize_file(manifest, 0);
  ASSERT_NO_FATAL_FAILURE(ExpectOpenFailsWithoutChanges(path));

  // Nothing was lost: restoring the manifest restores the collection.
  ASSERT_NO_FATAL_FAILURE(WriteFile(manifest, published));
  ASSERT_NO_FATAL_FAILURE(ExpectOpensWithAllDocs(path));
}

// Every truncation of a real manifest, including cuts at a field boundary
// that still decode, must be rejected without modifying anything.
TEST_F(ManifestRecoveryTest, EveryTruncationFailsClosedWithoutDeleting) {
  const auto path = root_ / "collection";
  ASSERT_NO_FATAL_FAILURE(CreateCollection(path));
  const auto ids = ManifestIds(path);
  ASSERT_EQ(ids.size(), 1u);
  const auto manifest = ManifestPath(path, ids.back());
  const std::string published = ReadFile(manifest);
  ASSERT_GT(published.size(), 0u);

  for (size_t length = 0; length < published.size(); ++length) {
    SCOPED_TRACE("manifest cut to " + std::to_string(length) + " of " +
                 std::to_string(published.size()) + " bytes");
    fs::resize_file(manifest, length);
    ASSERT_NO_FATAL_FAILURE(ExpectOpenFailsWithoutChanges(path));
    ASSERT_NO_FATAL_FAILURE(WriteFile(manifest, published));
  }
  // Control: the untouched manifest opens with every document.
  ASSERT_NO_FATAL_FAILURE(ExpectOpensWithAllDocs(path));
}

// No manifest validates (newest empty, previous cut short): fail closed.
TEST_F(ManifestRecoveryTest, NoValidManifestFailsClosedWithoutDeleting) {
  const auto path = root_ / "collection";
  const auto older = root_ / "older";
  ASSERT_NO_FATAL_FAILURE(CreateCollection(path, older));
  fs::copy(older, path,
           fs::copy_options::recursive | fs::copy_options::skip_existing);
  const auto ids = ManifestIds(path);
  ASSERT_EQ(ids.size(), 2u);

  fs::resize_file(ManifestPath(path, ids[1]), 0);
  const auto previous = ManifestPath(path, ids[0]);
  fs::resize_file(previous, fs::file_size(previous) - 1);
  ASSERT_NO_FATAL_FAILURE(ExpectOpenFailsWithoutChanges(path));
}

// The newest manifest is damaged but the previous generation (manifest and
// WAL) is still on disk: open must recover from it without losing data.
TEST_F(ManifestRecoveryTest, DamagedNewestManifestFallsBackToPrevious) {
  const auto pristine = root_ / "pristine";
  const auto older = root_ / "older";
  ASSERT_NO_FATAL_FAILURE(CreateCollection(pristine, older));
  const auto newest = ManifestIds(pristine);
  ASSERT_EQ(newest.size(), 1u);
  ASSERT_EQ(ManifestIds(older).size(), 1u);
  ASSERT_LT(ManifestIds(older).back(), newest.back());
  const auto manifest_name = ManifestPath(pristine, newest.back()).filename();
  const std::string published = ReadFile(pristine / manifest_name);

  // Control: the previous generation alone recovers every document, so a
  // failure below is a missing fallback and not a bad setup.
  {
    const auto control = root_ / "control";
    fs::copy(pristine, control, fs::copy_options::recursive);
    fs::copy(older, control,
             fs::copy_options::recursive | fs::copy_options::skip_existing);
    fs::remove(control / manifest_name);
    ASSERT_NO_FATAL_FAILURE(ExpectOpensWithAllDocs(control));
  }

  // Empty, cut in the last field (next_segment_id), and one byte short.
  const size_t last_field = published.size() - 2;
  for (size_t length : {size_t{0}, last_field, published.size() - 1}) {
    SCOPED_TRACE("newest manifest cut to " + std::to_string(length) + " of " +
                 std::to_string(published.size()) + " bytes");
    const auto copy = root_ / ("cut-" + std::to_string(length));
    fs::copy(pristine, copy, fs::copy_options::recursive);
    fs::copy(older, copy,
             fs::copy_options::recursive | fs::copy_options::skip_existing);
    fs::resize_file(copy / manifest_name, length);
    ASSERT_NO_FATAL_FAILURE(ExpectOpensWithAllDocs(copy));

    // The next flush publishes a fresh generation and retires the others.
    {
      auto opened = Collection::Open(copy.string(), Options(false));
      ASSERT_TRUE(opened.has_value()) << opened.error().message();
      ASSERT_NO_FATAL_FAILURE(InsertDoc(*opened.value(), kDocs));
      ASSERT_TRUE(opened.value()->flush().ok());
      ASSERT_TRUE(opened.value()->close().ok());
    }
    const auto ids = ManifestIds(copy);
    ASSERT_EQ(ids.size(), 1u);
    EXPECT_GT(ids.back(), newest.back());
    auto reopened = Collection::Open(copy.string(), Options(true));
    ASSERT_TRUE(reopened.has_value()) << reopened.error().message();
    ASSERT_NO_FATAL_FAILURE(ExpectDocs(*reopened.value(), kDocs + 1));
  }
}

// A temporary manifest left by a crash during publication is never loaded,
// and the next flush removes it.
TEST_F(ManifestRecoveryTest, StaleTemporaryManifestIsIgnoredAndRemoved) {
  const auto path = root_ / "collection";
  ASSERT_NO_FATAL_FAILURE(CreateCollection(path));
  const auto ids = ManifestIds(path);
  ASSERT_EQ(ids.size(), 1u);
  const std::string published = ReadFile(ManifestPath(path, ids.back()));
  const auto stale =
      path / ("manifest." + std::to_string(ids.back() + 1) + ".tmp");
  ASSERT_NO_FATAL_FAILURE(
      WriteFile(stale, published.substr(0, published.size() / 2)));

  ASSERT_NO_FATAL_FAILURE(ExpectOpensWithAllDocs(path));
  {
    auto opened = Collection::Open(path.string(), Options(false));
    ASSERT_TRUE(opened.has_value()) << opened.error().message();
    ASSERT_NO_FATAL_FAILURE(InsertDoc(*opened.value(), kDocs));
    ASSERT_TRUE(opened.value()->flush().ok());
    EXPECT_FALSE(fs::exists(stale)) << "flush() left the stale manifest";
    ASSERT_TRUE(opened.value()->close().ok());
  }
  for (const auto &entry : fs::directory_iterator(path)) {
    EXPECT_NE(entry.path().extension(), ".tmp") << entry.path();
  }
  auto reopened = Collection::Open(path.string(), Options(true));
  ASSERT_TRUE(reopened.has_value()) << reopened.error().message();
  ASSERT_NO_FATAL_FAILURE(ExpectDocs(*reopened.value(), kDocs + 1));
}

// A segment directory the manifest does not reference but that holds a WAL
// (for example a new writing segment whose manifest was never published)
// may hold acknowledged writes: open must keep it, and must not hand out its
// id again.
TEST_F(ManifestRecoveryTest, UnreferencedSegmentWithWalIsKept) {
  const auto path = root_ / "collection";
  ASSERT_NO_FATAL_FAILURE(CreateCollection(path));
  SegmentID next_id = 0;
  {
    auto recovered = VersionManager::Recovery(path.string());
    ASSERT_TRUE(recovered.has_value()) << recovered.error().message();
    next_id = recovered.value()->get_current_version().next_segment_id();
  }
  const auto orphan = path / std::to_string(next_id);
  fs::create_directories(orphan);
  ASSERT_NO_FATAL_FAILURE(WriteFile(orphan / "0.wal", "unreplayed records"));

  auto opened = Collection::Open(path.string(), Options(false));
  ASSERT_TRUE(opened.has_value()) << opened.error().message();
  EXPECT_TRUE(fs::exists(orphan / "0.wal")) << "Open deleted a WAL";
  // A new writing segment must not reuse the kept directory.
  ASSERT_NO_FATAL_FAILURE(InsertDoc(*opened.value(), kDocs));
  auto iterator = opened.value()->create_iterator();
  ASSERT_TRUE(iterator.has_value()) << iterator.error().message();
  iterator.value().reset();
  EXPECT_EQ(ReadFile(orphan / "0.wal"), "unreplayed records");
  ASSERT_NO_FATAL_FAILURE(ExpectDocs(*opened.value(), kDocs + 1));
  ASSERT_TRUE(opened.value()->close().ok());
}

// A short write while publishing a manifest (here: the file size limit,
// which fails like a full disk) must make flush() fail and must leave the
// previous generation as the recoverable state.
TEST_F(ManifestRecoveryTest, ShortManifestWriteFailsFlushAndKeepsPrevious) {
  const auto path = root_ / "manifests";
  fs::create_directories(path);

  Version version;
  version.set_schema(MakeSchema());
  auto persisted = std::make_shared<SegmentMeta>(0);
  persisted->add_persisted_block(
      BlockMeta(0, BlockType::SCALAR, 0, 31, 32, {"generation", "text"}));
  ASSERT_TRUE(version.add_persisted_segment_meta(persisted).ok());
  version.reset_writing_segment_meta(std::make_shared<SegmentMeta>(1));
  version.set_next_segment_id(2);
  auto created = VersionManager::Create(path.string(), version);
  ASSERT_TRUE(created.has_value());
  auto manager = created.value();
  ASSERT_TRUE(manager->flush().ok());
  ASSERT_EQ(ManifestIds(path).size(), 1u);
  const auto previous = ManifestPath(path, ManifestIds(path).back());
  const std::string published = ReadFile(previous);

  // The next generation seals segment 1 and opens segment 2.
  auto sealed = std::make_shared<SegmentMeta>(1);
  sealed->add_persisted_block(
      BlockMeta(1, BlockType::SCALAR, 32, 63, 32, {"generation", "text"}));
  ASSERT_TRUE(manager->add_persisted_segment_meta(sealed).ok());
  ASSERT_TRUE(
      manager->reset_writing_segment_meta(std::make_shared<SegmentMeta>(2))
          .ok());
  manager->set_next_segment_id(3);

  Status status;
  {
    FileSizeLimit limit(published.size() / 2);
    ASSERT_TRUE(limit.ok());
    status = manager->flush();
  }
  EXPECT_FALSE(status.ok()) << "flush() reported a short manifest write as OK";
  EXPECT_NE(status.message().find("manifest"), std::string::npos)
      << status.message();

  // The previous generation is intact and is what recovery loads.
  ASSERT_TRUE(fs::exists(previous));
  EXPECT_EQ(ReadFile(previous), published);
  EXPECT_EQ(ManifestIds(path).back(), ManifestIds(path).front());
  auto recovered = VersionManager::Recovery(path.string());
  ASSERT_TRUE(recovered.has_value()) << recovered.error().message();
  auto loaded = recovered.value()->get_current_version();
  EXPECT_EQ(loaded.persisted_segment_metas().size(), 1u);
  ASSERT_NE(loaded.writing_segment_meta(), nullptr);
  EXPECT_EQ(loaded.writing_segment_meta()->id(), 1u);
  EXPECT_EQ(loaded.next_segment_id(), 2u);

  // Once the fault is gone, the same flush publishes the new generation.
  ASSERT_TRUE(manager->flush().ok());
  recovered = VersionManager::Recovery(path.string());
  ASSERT_TRUE(recovered.has_value()) << recovered.error().message();
  EXPECT_EQ(recovered.value()->get_current_version().next_segment_id(), 3u);
  EXPECT_EQ(ManifestIds(path).size(), 1u);
}

}  // namespace
}  // namespace zvec
