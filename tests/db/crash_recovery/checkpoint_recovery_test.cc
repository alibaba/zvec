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

#include "checkpoint_recovery.h"
#include <sys/wait.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <thread>
#include <gtest/gtest.h>
#include "utility.h"

namespace zvec::checkpoint_test {
namespace {

using ExpectedDocs = std::map<int, int>;

ExpectedDocs InitialDocs(int count = kInitialDocs) {
  ExpectedDocs docs;
  for (int id = 0; id < count; ++id) {
    docs.emplace(id, 0);
  }
  return docs;
}

void CheckWrite(const Result<WriteResults> &result, size_t count) {
  ASSERT_TRUE(result.has_value()) << result.error().message();
  ASSERT_EQ(result->size(), count);
  for (size_t i = 0; i < result->size(); ++i) {
    ASSERT_TRUE((*result)[i].ok())
        << "document " << i << ": " << (*result)[i].message();
  }
}

class CheckpointRecoveryTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    ASSERT_NO_THROW(worker_ = LocateBinary("checkpoint_recovery_worker"));
    // Each test process owns a unique directory, including parallel CI runs.
    std::string pattern =
        (std::filesystem::current_path() / "zvec_checkpoint_XXXXXX").string();
    char *directory = mkdtemp(pattern.data());
    ASSERT_NE(directory, nullptr);
    directory_ = directory;
    path_ = directory_ + "/collection";
  }

  void TearDown() override {
    if (!directory_.empty()) {
      std::error_code error;
      std::filesystem::remove_all(directory_, error);
      EXPECT_FALSE(error) << error.message();
    }
  }

  void CrashAt(const std::string &checkpoint) {
    SCOPED_TRACE(checkpoint);
    const auto log_path = directory_ + "/worker.log";
    const int log_fd =
        open(log_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    ASSERT_GE(log_fd, 0);
    const char *fts = GetParam() ? "fts" : "plain";
    const char *args[] = {worker_.c_str(), path_.c_str(), checkpoint.c_str(),
                          fts, nullptr};
    const pid_t child = fork();
    if (child == 0) {
      // Only async-signal-safe calls are allowed between fork and exec.
      if (dup2(log_fd, STDOUT_FILENO) < 0 || dup2(log_fd, STDERR_FILENO) < 0) {
        _exit(126);
      }
      close(log_fd);
      execv(args[0], const_cast<char *const *>(args));
      _exit(127);
    }
    close(log_fd);
    ASSERT_GT(child, 0);

    int status = 0;
    pid_t waited = 0;
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(30);
    do {
      waited = waitpid(child, &status, WUNTRACED | WNOHANG);
      if (waited == child || (waited < 0 && errno != EINTR)) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } while (std::chrono::steady_clock::now() < deadline);

    const bool reached_checkpoint =
        waited == child && WIFSTOPPED(status) && WSTOPSIG(status) == SIGSTOP;
    const int checkpoint_status = status;
    int kill_result = 0;
    pid_t reaped = child;
    if (waited != child || WIFSTOPPED(status)) {
      kill_result = kill(child, SIGKILL);
      do {
        reaped = waitpid(child, &status, 0);
      } while (reaped < 0 && errno == EINTR);
    }
    std::ifstream log(log_path);
    std::ostringstream output;
    output << log.rdbuf();
    ASSERT_TRUE(reached_checkpoint)
        << "Worker failed or timed out before checkpoint; waitpid=" << waited
        << ", status=" << checkpoint_status << "\n"
        << output.str();
    ASSERT_EQ(kill_result, 0) << output.str();
    ASSERT_EQ(reaped, child) << output.str();
    ASSERT_TRUE(WIFSIGNALED(status)) << output.str();
    ASSERT_EQ(WTERMSIG(status), SIGKILL) << output.str();
  }

  void CheckQuery(Collection &collection, const SearchQuery &query,
                  const std::set<std::string> &expected) {
    auto result = collection.query(query);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    std::set<std::string> actual;
    for (const auto &doc : result.value()) {
      ASSERT_NE(doc, nullptr);
      EXPECT_TRUE(actual.insert(doc->pk()).second)
          << "Duplicate query result: " << doc->pk();
    }
    EXPECT_EQ(actual, expected);
  }

  void Verify(Collection &collection, const ExpectedDocs &expected,
              int max_id) {
    auto stats = collection.stats();
    ASSERT_TRUE(stats.has_value()) << stats.error().message();
    EXPECT_EQ(stats->doc_count, expected.size());
    std::vector<std::string> keys;
    for (int id = 0; id <= max_id; ++id) {
      keys.push_back(PrimaryKey(id));
    }
    auto fetched = collection.fetch(keys);
    ASSERT_TRUE(fetched.has_value()) << fetched.error().message();
    for (int id = 0; id <= max_id; ++id) {
      SCOPED_TRACE(PrimaryKey(id));
      const auto actual = fetched->find(PrimaryKey(id));
      ASSERT_NE(actual, fetched->end());
      const auto wanted = expected.find(id);
      if (wanted == expected.end()) {
        EXPECT_EQ(actual->second, nullptr);
      } else {
        ASSERT_NE(actual->second, nullptr);
        EXPECT_EQ(*actual->second, MakeDoc(id, wanted->second));
      }
    }

    SearchQuery vector_query;
    vector_query.target_.field_name_ = "vec";
    const std::vector<float> vector{0, 0, 1, 0};
    vector_query.target_.set_vector(
        std::string(reinterpret_cast<const char *>(vector.data()),
                    vector.size() * sizeof(float)));
    vector_query.topk_ = max_id + 2;
    std::set<std::string> all_keys;
    for (const auto &[id, generation] : expected) {
      all_keys.insert(PrimaryKey(id));
    }
    ASSERT_NO_FATAL_FAILURE(CheckQuery(collection, vector_query, all_keys));

    // Exact Flat search avoids approximate-recall tolerances masking lost rows.
    // Query every generation, including generations whose rows were deleted.
    for (int generation = 0; generation <= 3; ++generation) {
      SCOPED_TRACE("generation=" + std::to_string(generation));
      std::set<std::string> generation_keys;
      for (const auto &[id, value] : expected) {
        if (value == generation) {
          generation_keys.insert(PrimaryKey(id));
        }
      }
      vector_query.filter_ = "generation = " + std::to_string(generation);
      ASSERT_NO_FATAL_FAILURE(
          CheckQuery(collection, vector_query, generation_keys));
      if (GetParam()) {
        for (const auto *field : {"text", "title"}) {
          SCOPED_TRACE(field);
          SearchQuery fts_query;
          fts_query.target_.field_name_ = field;
          fts_query.topk_ = max_id + 2;
          FtsClause clause;
          clause.query_string_ = "version" + std::to_string(generation);
          fts_query.target_.clause_ = clause;
          ASSERT_NO_FATAL_FAILURE(
              CheckQuery(collection, fts_query, generation_keys));
        }
      }
    }
  }

  void VerifyAndContinue(ExpectedDocs expected, int max_id) {
    auto opened = Collection::Open(path_, {});
    ASSERT_TRUE(opened.has_value()) << opened.error().message();
    auto collection = std::move(opened.value());
    ASSERT_NO_FATAL_FAILURE(Verify(*collection, expected, max_id));

    // Exercise every mutation API after recovery, then persist and reopen.
    std::vector<Doc> inserted{MakeDoc(max_id + 1, 3)};
    ASSERT_NO_FATAL_FAILURE(CheckWrite(collection->insert(inserted), 1));
    expected[max_id + 1] = 3;
    std::vector<Doc> updated{MakeDoc(max_id + 1, 2)};
    ASSERT_NO_FATAL_FAILURE(CheckWrite(collection->update(updated), 1));
    expected[max_id + 1] = 2;
    ASSERT_NO_FATAL_FAILURE(
        CheckWrite(collection->delete_({PrimaryKey(max_id + 1)}), 1));
    expected.erase(max_id + 1);
    std::vector<Doc> upserted{MakeDoc(max_id + 1, 3)};
    ASSERT_NO_FATAL_FAILURE(CheckWrite(collection->upsert(upserted), 1));
    expected[max_id + 1] = 3;
    ASSERT_NO_FATAL_FAILURE(Verify(*collection, expected, max_id + 1));
    auto status = collection->flush();
    ASSERT_TRUE(status.ok()) << status.message();
    status = collection->optimize();
    ASSERT_TRUE(status.ok()) << status.message();
    ASSERT_NO_FATAL_FAILURE(Verify(*collection, expected, max_id + 1));
    status = collection->close();
    ASSERT_TRUE(status.ok()) << status.message();
    collection.reset();
    auto reopened = Collection::Open(path_, {});
    ASSERT_TRUE(reopened.has_value()) << reopened.error().message();
    ASSERT_NO_FATAL_FAILURE(Verify(*reopened.value(), expected, max_id + 1));
    status = reopened.value()->close();
    ASSERT_TRUE(status.ok()) << status.message();
  }

  std::string worker_;
  std::string directory_;
  std::string path_;
};

TEST_P(CheckpointRecoveryTest, AcknowledgedWritesSurviveProcessKill) {
  ASSERT_NO_FATAL_FAILURE(CrashAt("written"));
  ASSERT_NO_FATAL_FAILURE(VerifyAndContinue(InitialDocs(), kInitialDocs - 1));
}

TEST_P(CheckpointRecoveryTest, FlushedWritesSurviveProcessKill) {
  ASSERT_NO_FATAL_FAILURE(CrashAt("flushed"));
  ASSERT_NO_FATAL_FAILURE(VerifyAndContinue(InitialDocs(), kInitialDocs - 1));
}

TEST_P(CheckpointRecoveryTest, SealedSegmentSurvivesProcessKill) {
  ASSERT_NO_FATAL_FAILURE(CrashAt("sealed"));
  ASSERT_NO_FATAL_FAILURE(VerifyAndContinue(InitialDocs(), kInitialDocs - 1));
}

TEST_P(CheckpointRecoveryTest, OptimizedSegmentSurvivesProcessKill) {
  ASSERT_NO_FATAL_FAILURE(CrashAt("optimized"));
  ASSERT_NO_FATAL_FAILURE(VerifyAndContinue(InitialDocs(), kInitialDocs - 1));
}

TEST_P(CheckpointRecoveryTest, SegmentRolloverSurvivesProcessKill) {
  ASSERT_NO_FATAL_FAILURE(CrashAt("rollover"));
  ASSERT_NO_FATAL_FAILURE(
      VerifyAndContinue(InitialDocs(kSegmentDocs + 1), kSegmentDocs));
}

TEST_P(CheckpointRecoveryTest, RepeatedRecoveryWithoutClosePreservesWal) {
  ASSERT_NO_FATAL_FAILURE(CrashAt("written"));
  // No parent open/close is allowed between kills: it could flush the WAL and
  // conceal replay idempotency or premature WAL deletion bugs.
  for (int attempt = 0; attempt < 3; ++attempt) {
    SCOPED_TRACE(attempt);
    ASSERT_NO_FATAL_FAILURE(CrashAt("recovered"));
  }
  ASSERT_NO_FATAL_FAILURE(VerifyAndContinue(InitialDocs(), kInitialDocs - 1));
}

TEST_P(CheckpointRecoveryTest, MixedMutationsSurviveProcessKill) {
  ASSERT_NO_FATAL_FAILURE(CrashAt("flushed"));
  ASSERT_NO_FATAL_FAILURE(CrashAt("mutated"));
  auto expected = InitialDocs();
  expected[0] = 1;
  expected.erase(1);
  expected[2] = 2;
  expected[kInitialDocs] = 2;
  ASSERT_NO_FATAL_FAILURE(VerifyAndContinue(expected, kInitialDocs));
}

TEST_P(CheckpointRecoveryTest, FlushedMutationsSurviveProcessKill) {
  ASSERT_NO_FATAL_FAILURE(CrashAt("flushed"));
  ASSERT_NO_FATAL_FAILURE(CrashAt("mutated_flushed"));
  auto expected = InitialDocs();
  expected[0] = 1;
  expected.erase(1);
  expected[2] = 2;
  expected[kInitialDocs] = 2;
  ASSERT_NO_FATAL_FAILURE(VerifyAndContinue(expected, kInitialDocs));
}

TEST_P(CheckpointRecoveryTest, DeletedRowsStayDeletedAfterEmptyOptimize) {
  ASSERT_NO_FATAL_FAILURE(CrashAt("flushed"));
  ASSERT_NO_FATAL_FAILURE(CrashAt("deleted"));
  ASSERT_NO_FATAL_FAILURE(VerifyAndContinue({}, kInitialDocs - 1));
}

INSTANTIATE_TEST_SUITE_P(IndexTypes, CheckpointRecoveryTest, ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool> &info) {
                           return info.param ? "Fts" : "Plain";
                         });

}  // namespace
}  // namespace zvec::checkpoint_test
