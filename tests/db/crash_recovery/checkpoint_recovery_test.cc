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
#include "recovery_verifier.h"
#include "utility.h"

namespace zvec::checkpoint_test {
namespace {

class CheckpointRecoveryTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    ASSERT_NO_THROW(worker_ = LocateBinary("checkpoint_recovery_worker"));
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

  void crash_at(const std::string &checkpoint) {
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

  void verify(Collection &collection, const ExpectedDocs &expected, int max_id,
              bool pristine = false) {
    try {
      VerifyState(collection, GetParam(), expected, max_id, pristine);
    } catch (const std::exception &error) {
      FAIL() << error.what();
    }
  }

  void verify_and_continue(ExpectedDocs expected, int max_id) {
    auto opened = Collection::Open(path_, {});
    ASSERT_TRUE(opened.has_value()) << opened.error().message();
    auto collection = std::move(opened.value());
    ASSERT_NO_FATAL_FAILURE(verify(*collection, expected, max_id, true));

    std::vector<Doc> inserted{MakeDoc(max_id + 1, 3)};
    ASSERT_NO_FATAL_FAILURE(CheckWrite(collection->insert(inserted), 1));
    expected[max_id + 1] = 3;
    ASSERT_NO_FATAL_FAILURE(verify(*collection, expected, max_id + 1));
    std::vector<Doc> updated{MakeDoc(max_id + 1, 2)};
    ASSERT_NO_FATAL_FAILURE(CheckWrite(collection->update(updated), 1));
    expected[max_id + 1] = 2;
    ASSERT_NO_FATAL_FAILURE(verify(*collection, expected, max_id + 1));
    ASSERT_NO_FATAL_FAILURE(
        CheckWrite(collection->delete_({PrimaryKey(max_id + 1)}), 1));
    expected.erase(max_id + 1);
    ASSERT_NO_FATAL_FAILURE(verify(*collection, expected, max_id + 1));
    std::vector<Doc> upserted{MakeDoc(max_id + 1, 3)};
    ASSERT_NO_FATAL_FAILURE(CheckWrite(collection->upsert(upserted), 1));
    expected[max_id + 1] = 3;
    ASSERT_NO_FATAL_FAILURE(verify(*collection, expected, max_id + 1));
    if (!expected.empty() && expected.begin()->first <= max_id) {
      int old_id = expected.begin()->first;
      std::vector<Doc> old_update{MakeDoc(old_id, 2)};
      ASSERT_NO_THROW(CheckWrite(collection->update(old_update), 1));
      expected[old_id] = 2;
      ASSERT_NO_FATAL_FAILURE(verify(*collection, expected, max_id + 1));
    }
    auto status = collection->flush();
    ASSERT_TRUE(status.ok()) << status.message();
    status = collection->optimize();
    ASSERT_TRUE(status.ok()) << status.message();
    ASSERT_NO_FATAL_FAILURE(verify(*collection, expected, max_id + 1));
    status = collection->close();
    ASSERT_TRUE(status.ok()) << status.message();
    collection.reset();
    auto reopened = Collection::Open(path_, {});
    ASSERT_TRUE(reopened.has_value()) << reopened.error().message();
    ASSERT_NO_FATAL_FAILURE(verify(*reopened.value(), expected, max_id + 1));
    status = reopened.value()->close();
    ASSERT_TRUE(status.ok()) << status.message();
  }

  std::string worker_;
  std::string directory_;
  std::string path_;
};

TEST_P(CheckpointRecoveryTest, AcknowledgedWritesSurviveProcessKill) {
  ASSERT_NO_FATAL_FAILURE(crash_at("written"));
  ASSERT_NO_FATAL_FAILURE(verify_and_continue(InitialDocs(), kInitialDocs - 1));
}

TEST_P(CheckpointRecoveryTest, FlushedWritesSurviveProcessKill) {
  ASSERT_NO_FATAL_FAILURE(crash_at("flushed"));
  ASSERT_NO_FATAL_FAILURE(verify_and_continue(InitialDocs(), kInitialDocs - 1));
}

TEST_P(CheckpointRecoveryTest, SealedSegmentSurvivesProcessKill) {
  ASSERT_NO_FATAL_FAILURE(crash_at("sealed"));
  ASSERT_NO_FATAL_FAILURE(verify_and_continue(InitialDocs(), kInitialDocs - 1));
}

TEST_P(CheckpointRecoveryTest, OptimizedSegmentSurvivesProcessKill) {
  ASSERT_NO_FATAL_FAILURE(crash_at("optimized"));
  ASSERT_NO_FATAL_FAILURE(verify_and_continue(InitialDocs(), kInitialDocs - 1));
}

TEST_P(CheckpointRecoveryTest, SegmentRolloverSurvivesProcessKill) {
  ASSERT_NO_FATAL_FAILURE(crash_at("rollover"));
  ASSERT_NO_FATAL_FAILURE(
      verify_and_continue(InitialDocs(kSegmentDocs + 1), kSegmentDocs));
}

TEST_P(CheckpointRecoveryTest, RepeatedRecoveryWithoutClosePreservesWal) {
  ASSERT_NO_FATAL_FAILURE(crash_at("written"));
  // No parent open/close is allowed between kills: it could flush the WAL and
  // conceal replay idempotency or premature WAL deletion bugs.
  for (int attempt = 0; attempt < 3; ++attempt) {
    SCOPED_TRACE(attempt);
    ASSERT_NO_FATAL_FAILURE(crash_at("recovered"));
  }
  ASSERT_NO_FATAL_FAILURE(verify_and_continue(InitialDocs(), kInitialDocs - 1));
}

TEST_P(CheckpointRecoveryTest, MixedMutationsSurviveProcessKill) {
  ASSERT_NO_FATAL_FAILURE(crash_at("flushed"));
  ASSERT_NO_FATAL_FAILURE(crash_at("mutated"));
  auto expected = InitialDocs();
  expected[0] = 1;
  expected.erase(1);
  expected[2] = 2;
  expected[kInitialDocs] = 2;
  ASSERT_NO_FATAL_FAILURE(verify_and_continue(expected, kInitialDocs));
}

TEST_P(CheckpointRecoveryTest, FlushedMutationsSurviveProcessKill) {
  ASSERT_NO_FATAL_FAILURE(crash_at("flushed"));
  ASSERT_NO_FATAL_FAILURE(crash_at("mutated_flushed"));
  auto expected = InitialDocs();
  expected[0] = 1;
  expected.erase(1);
  expected[2] = 2;
  expected[kInitialDocs] = 2;
  ASSERT_NO_FATAL_FAILURE(verify_and_continue(expected, kInitialDocs));
}

TEST_P(CheckpointRecoveryTest, DeletedRowsStayDeletedAfterEmptyOptimize) {
  ASSERT_NO_FATAL_FAILURE(crash_at("flushed"));
  ASSERT_NO_FATAL_FAILURE(crash_at("deleted"));
  ASSERT_NO_FATAL_FAILURE(verify_and_continue({}, kInitialDocs - 1));
}

INSTANTIATE_TEST_SUITE_P(IndexTypes, CheckpointRecoveryTest, ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool> &info) {
                           return info.param ? "Fts" : "Plain";
                         });

}  // namespace
}  // namespace zvec::checkpoint_test
