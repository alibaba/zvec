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

#include <sys/resource.h>
#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <set>
#include <string>
#include <tuple>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/db/collection.h>
#include <zvec/db/index_params.h>

namespace zvec {
namespace {

constexpr uint32_t kDocs = 1000;

Doc MakeDoc(const std::string &pk, int32_t n = 1) {
  Doc doc;
  doc.set_pk(pk);
  EXPECT_TRUE(doc.set<std::string>("cls", "a"));
  EXPECT_TRUE(doc.set<int32_t>("n", n));
  return doc;
}

// The limit is process-wide, so every case runs in a GoogleTest child process.
// Fill every available descriptor slot, then release four: sealing the already
// flushed segment fits, but opening the next RocksDB scalar index hits EMFILE.
// This exercises real resource exhaustion without a production fault hook.
class FileDescriptorLimit {
 public:
  ~FileDescriptorLimit() {
    restore();
  }

  void exhaust() {
    ASSERT_EQ(getrlimit(RLIMIT_NOFILE, &original_), 0);
    DIR *dir = opendir("/dev/fd");
    ASSERT_NE(dir, nullptr);
    int highest = 2;
    errno = 0;
    while (auto *entry = readdir(dir)) {
      if (entry->d_name[0] != '.') {
        highest = std::max(highest, std::atoi(entry->d_name));
      }
    }
    EXPECT_EQ(errno, 0);
    ASSERT_EQ(closedir(dir), 0);
    struct rlimit limit = original_;
    limit.rlim_cur = highest + 16;
    ASSERT_LE(limit.rlim_cur, original_.rlim_cur);
    ASSERT_EQ(setrlimit(RLIMIT_NOFILE, &limit), 0);
    changed_ = true;
    int fd;
    while ((fd = open("/dev/null", O_RDONLY)) >= 0) {
      descriptors_.push_back(fd);
    }
    ASSERT_EQ(errno, EMFILE);
    ASSERT_GE(descriptors_.size(), 4u);
    for (int i = 0; i < 4; ++i) {
      ASSERT_EQ(::close(descriptors_.back()), 0);
      descriptors_.pop_back();
    }
    std::cout << "EMFILE trigger: four descriptor slots available\n";
  }

  void restore() {
    for (int fd : descriptors_) EXPECT_EQ(::close(fd), 0);
    descriptors_.clear();
    if (changed_) {
      EXPECT_EQ(setrlimit(RLIMIT_NOFILE, &original_), 0);
      changed_ = false;
    }
  }

 private:
  struct rlimit original_{};
  bool changed_ = false;
  std::vector<int> descriptors_;
};

void ExpectDocuments(const Collection::Ptr &collection, uint32_t count) {
  auto stats = collection->stats();
  ASSERT_TRUE(stats.has_value()) << stats.error().message();
  EXPECT_EQ(stats->doc_count, count);
  // An empty filter forces a forward scan even after create_index indexes n.
  // The original failure can put a rejected insert in the forward store but
  // not the scalar index.
  for (const char *filter : {"cls = 'a'", "n >= 0", ""}) {
    SearchQuery query;
    query.filter_ = filter;
    query.topk_ = 2 * kDocs + 10;
    auto rows = collection->query(query);
    ASSERT_TRUE(rows.has_value()) << rows.error().message();
    std::set<std::string> pks;
    for (const auto &doc : rows.value()) {
      pks.insert(doc->pk());
      EXPECT_NE(doc->pk(), "rejected");
    }
    std::cout << filter << ": doc_count=" << stats->doc_count
              << " rows=" << rows->size() << " distinct=" << pks.size() << '\n';
    EXPECT_EQ(rows->size(), count);
    EXPECT_EQ(pks.size(), count);
  }
}

enum class Operation { Insert, Optimize, CreateIndex };

Status Rollover(const Collection::Ptr &collection, Operation operation,
                const std::string &pk) {
  if (operation == Operation::Optimize) return collection->optimize();
  if (operation == Operation::CreateIndex) {
    return collection->create_index("n", std::make_shared<InvertIndexParams>());
  }
  std::vector<Doc> docs{MakeDoc(pk, 7)};
  auto result = collection->insert(docs);
  if (!result) return result.error();
  EXPECT_EQ(result->size(), 1u);
  return result->at(0);
}

void RunCase(Operation operation, bool reopen_before_retry) {
  const std::string path = "failed_rollover_" + std::to_string(getpid());
  ASSERT_FALSE(std::filesystem::exists(path));
  CollectionSchema schema("failed_rollover");
  if (operation == Operation::Insert) {
    schema.set_max_doc_count_per_segment(kDocs);
  }
  ASSERT_TRUE(schema
                  .add_field(std::make_shared<FieldSchema>(
                      "cls", DataType::STRING, false,
                      std::make_shared<InvertIndexParams>()))
                  .ok());
  ASSERT_TRUE(
      schema.add_field(std::make_shared<FieldSchema>("n", DataType::INT32))
          .ok());
  auto opened = Collection::CreateAndOpen(path, schema, CollectionOptions{});
  ASSERT_TRUE(opened.has_value()) << opened.error().message();
  auto collection = std::move(opened.value());
  std::vector<Doc> docs;
  for (uint32_t i = 0; i < kDocs; ++i)
    docs.push_back(MakeDoc(std::to_string(i)));
  auto inserted = collection->insert(docs);
  ASSERT_TRUE(inserted.has_value()) << inserted.error().message();
  for (const auto &s : inserted.value()) ASSERT_TRUE(s.ok()) << s.message();
  // Match the verifier's on-disk starting state and keep the dump cost fixed.
  ASSERT_TRUE(collection->close().ok());
  collection.reset();
  opened = Collection::Open(path, CollectionOptions{});
  ASSERT_TRUE(opened.has_value()) << opened.error().message();
  collection = std::move(opened.value());

  FileDescriptorLimit limit;
  for (int attempt = 1; attempt <= 2; ++attempt) {
    limit.exhaust();
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    auto failed = Rollover(collection, operation, "rejected");
    limit.restore();
    ASSERT_FALSE(failed.ok());
    // A failed scalar-index open identifies the replacement-creation phase,
    // rather than a failure in flushing/sealing the old segment.
    ASSERT_EQ(failed.message(), "Failed to open scalar indexer");
    auto unused_path = path + "/" + std::to_string(attempt);
    std::cout << "failed replacement " << unused_path
              << " exists=" << std::filesystem::exists(unused_path) << '\n';
    EXPECT_FALSE(std::filesystem::exists(unused_path));
    ExpectDocuments(collection, kDocs);
  }

  uint32_t expected = kDocs;
  if (operation != Operation::Insert) {
    // Maintenance rolls over a non-full segment. A plain write must still
    // append to that segment, before another maintenance attempt seals it.
    std::vector<Doc> more{MakeDoc("before_retry")};
    auto written = collection->insert(more);
    ASSERT_TRUE(written.has_value()) << written.error().message();
    ASSERT_TRUE(written->at(0).ok()) << written->at(0).message();
    ++expected;
    ExpectDocuments(collection, expected);
  }

  if (reopen_before_retry) {
    ASSERT_TRUE(collection->close().ok());
    collection.reset();
    opened = Collection::Open(path, CollectionOptions{});
    ASSERT_TRUE(opened.has_value()) << opened.error().message();
    collection = std::move(opened.value());
    ExpectDocuments(collection, expected);
  }

  auto retried = Rollover(collection, operation, "accepted");
  ASSERT_TRUE(retried.ok()) << retried.message();
  if (operation == Operation::Insert) {
    ++expected;
  } else {
    std::vector<Doc> more{MakeDoc("accepted")};
    auto written = collection->insert(more);
    ASSERT_TRUE(written.has_value()) << written.error().message();
    ASSERT_TRUE(written->at(0).ok()) << written->at(0).message();
    ++expected;
  }
  ExpectDocuments(collection, expected);
  ASSERT_TRUE(collection->close().ok());
  collection.reset();
  opened = Collection::Open(path, CollectionOptions{});
  ASSERT_TRUE(opened.has_value()) << opened.error().message();
  collection = std::move(opened.value());
  ExpectDocuments(collection, expected);
  std::vector<Doc> more{MakeDoc("after_restart")};
  auto written = collection->insert(more);
  ASSERT_TRUE(written.has_value()) << written.error().message();
  ASSERT_TRUE(written->at(0).ok()) << written->at(0).message();
  ExpectDocuments(collection, expected + 1);
  ASSERT_TRUE(collection->close().ok());
  collection.reset();
  std::filesystem::remove_all(path);
}

class FailedRollover
    : public ::testing::TestWithParam<std::tuple<Operation, bool>> {};

TEST_P(FailedRollover, EmfileLeavesDocumentsUniqueAndWritesRetryable) {
  // Re-exec instead of inheriting RocksDB threads started by another test.
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  const auto [operation, reopen] = GetParam();
  ASSERT_EXIT(
      {
        RunCase(operation, reopen);
        std::cout.flush();
        std::cerr.flush();
        _exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

INSTANTIATE_TEST_SUITE_P(
    EntryPaths, FailedRollover,
    ::testing::Combine(::testing::Values(Operation::Insert, Operation::Optimize,
                                         Operation::CreateIndex),
                       ::testing::Bool()));

TEST(FailedRolloverBatch, KeepsResultsForTheWrittenPrefix) {
  const std::string path = "failed_rollover_batch_" + std::to_string(getpid());
  ASSERT_FALSE(std::filesystem::exists(path));
  CollectionSchema schema("failed_rollover_batch");
  schema.set_max_doc_count_per_segment(kDocs);
  ASSERT_TRUE(schema
                  .add_field(std::make_shared<FieldSchema>(
                      "cls", DataType::STRING, false,
                      std::make_shared<InvertIndexParams>()))
                  .ok());
  ASSERT_TRUE(
      schema.add_field(std::make_shared<FieldSchema>("n", DataType::INT32))
          .ok());
  auto opened = Collection::CreateAndOpen(path, schema, CollectionOptions{});
  ASSERT_TRUE(opened.has_value()) << opened.error().message();
  auto collection = std::move(opened.value());
  std::vector<Doc> docs;
  for (uint32_t i = 0; i < kDocs - 1; ++i)
    docs.push_back(MakeDoc(std::to_string(i)));
  auto inserted = collection->insert(docs);
  ASSERT_TRUE(inserted.has_value()) << inserted.error().message();
  for (const auto &status : inserted.value()) ASSERT_TRUE(status.ok());

  // The EMFILE cases above cover the real creation failure. A directory
  // conflict here targets the second document's rollover without starving
  // the first document's WAL/index writes of descriptors.
  ASSERT_TRUE(std::filesystem::create_directory(path + "/1"));
  std::vector<Doc> batch{MakeDoc("prefix"), MakeDoc("rejected"),
                         MakeDoc("suffix")};
  auto result = collection->insert(batch);
  ASSERT_TRUE(result.has_value()) << result.error().message();
  ASSERT_EQ(result->size(), batch.size());
  EXPECT_TRUE(result->at(0).ok());
  EXPECT_FALSE(result->at(1).ok());
  EXPECT_FALSE(result->at(2).ok());
  EXPECT_EQ(result->at(1).message(), result->at(2).message());
  // A failed create must not retire a directory it did not create.
  EXPECT_TRUE(std::filesystem::is_directory(path + "/1"));
  ExpectDocuments(collection, kDocs);
  auto fetched = collection->fetch({"prefix", "rejected", "suffix"});
  ASSERT_TRUE(fetched.has_value()) << fetched.error().message();
  ASSERT_NE(fetched->at("prefix"), nullptr);
  EXPECT_EQ(fetched->at("rejected"), nullptr);
  EXPECT_EQ(fetched->at("suffix"), nullptr);
  ASSERT_TRUE(collection->close().ok());
  collection.reset();
  opened = Collection::Open(path, CollectionOptions{});
  ASSERT_TRUE(opened.has_value()) << opened.error().message();
  collection = std::move(opened.value());
  ExpectDocuments(collection, kDocs);
  fetched = collection->fetch({"prefix", "rejected", "suffix"});
  ASSERT_TRUE(fetched.has_value()) << fetched.error().message();
  EXPECT_NE(fetched->at("prefix"), nullptr);
  EXPECT_EQ(fetched->at("rejected"), nullptr);
  EXPECT_EQ(fetched->at("suffix"), nullptr);
  std::vector<Doc> retry{MakeDoc("retry")};
  auto retried = collection->insert(retry);
  ASSERT_TRUE(retried.has_value()) << retried.error().message();
  ASSERT_TRUE(retried->at(0).ok()) << retried->at(0).message();
  ExpectDocuments(collection, kDocs + 1);
  ASSERT_TRUE(collection->close().ok());
  collection.reset();
  std::filesystem::remove_all(path);
}

}  // namespace
}  // namespace zvec
