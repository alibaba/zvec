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


#include <csignal>
#include <filesystem>
#include <thread>
#include <gtest/gtest.h>
#include <zvec/db/collection.h>
#include <zvec/db/doc.h>
#include <zvec/db/schema.h>
#include "tests/test_util.h"
#include "utility.h"

#ifdef _WIN32
#include <process.h>
#include <windows.h>
typedef HANDLE pid_t;
#define SIGKILL 9
#define WIFEXITED(status) true
#define WEXITSTATUS(status) (status)
#define WIFSIGNALED(status) ((status) != 0)
#endif


namespace zvec {


static std::string data_generator_bin_;
const std::string collection_name_{"write_recovery_test"};
const std::string dir_path_{"write_recovery_test_db"};
const zvec::CollectionOptions options_{false, true, 256 * 1024};


static std::string LocateDataGenerator() {
  namespace fs = std::filesystem;
  std::cout << "Current path: " << fs::current_path() << std::endl;

  const std::string base_name = "data_generator";
  std::vector<std::string> candidates;
  // Define potential search locations relative to the current working directory
  const std::vector<std::string> search_paths = {"./", "./bin/"};

  for (const auto &p : search_paths) {
    candidates.push_back(p);
  }
#ifdef _WIN32
  for (const auto &p : search_paths) {
    candidates.push_back(p + "Debug/");
    candidates.push_back(p + "Release/");
  }
#endif


  for (auto &p : candidates) {
    p += base_name;
#ifdef _WIN32
    p += ".exe";  // Append .exe suffix for Windows compatibility
#endif
  }

  for (const auto &p : candidates) {
    if (fs::exists(p)) {
      return fs::canonical(p).string();
    }
  }
  throw std::runtime_error("data_generator binary not found");
}


/**
 * Internal helper to execute the data generator process.
 * If 'kill_after_seconds' is greater than 0, the process will be forcibly
 * terminated after the specified duration to simulate a crash.
 */
void ExecuteProcess(const std::string &start, const std::string &end,
                    const std::string &op, const std::string &version,
                    int kill_after_seconds = -1) {
  bool should_crash = kill_after_seconds >= 0;

#ifdef _WIN32
  // 1. Build the command line string with quotes to handle paths with spaces
  std::string cmd_str = data_generator_bin_ + " --path " + dir_path_ +
                        " --start " + start + " --end " + end + " --op " + op +
                        " --version " + version;

  STARTUPINFOA si = {sizeof(si)};
  PROCESS_INFORMATION pi;

  std::cout << cmd_str << std::endl;

  std::vector<char> cmd_buf(cmd_str.begin(), cmd_str.end());
  cmd_buf.push_back('\0');
  if (!CreateProcessA(NULL, cmd_buf.data(), NULL, NULL, FALSE, 0, NULL, NULL,
                      &si, &pi)) {
    FAIL() << "CreateProcess failed (" << GetLastError() << ")";
  }

  if (should_crash) {
    std::this_thread::sleep_for(std::chrono::seconds(kill_after_seconds));
    // Simulate a crash by killing the process
    TerminateProcess(pi.hProcess, 1);
  }

  WaitForSingleObject(pi.hProcess, INFINITE);

  DWORD exit_code;
  GetExitCodeProcess(pi.hProcess, &exit_code);
  CloseHandle(pi.hProcess);
  CloseHandle(pi.hThread);

  if (should_crash) {
    // In crash tests, we expect the process to have been terminated
    ASSERT_NE(exit_code, 0) << "Process was expected to crash/terminate.";
  } else {
    ASSERT_EQ(exit_code, 0) << "Process failed with exit code: " << exit_code;
  }

#else
  // POSIX Implementation
  pid_t pid = fork();
  ASSERT_GE(pid, 0);

  if (pid == 0) {  // Child process
    // Use a vector to manage arguments cleanly
    std::vector<const char *> args = {data_generator_bin_.c_str(),
                                      "--path",
                                      dir_path_.c_str(),
                                      "--start",
                                      start.c_str(),
                                      "--end",
                                      end.c_str(),
                                      "--op",
                                      op.c_str(),
                                      "--version",
                                      version.c_str(),
                                      nullptr};
    execvp(args[0], const_cast<char *const *>(args.data()));
    perror("execvp failed");
    _exit(1);
  }

  int status;
  if (should_crash) {
    std::this_thread::sleep_for(std::chrono::seconds(kill_after_seconds));
    kill(pid, SIGKILL);
    waitpid(pid, &status, 0);
    ASSERT_TRUE(WIFSIGNALED(status)) << "Process did not crash as expected.";
  } else {
    waitpid(pid, &status, 0);
    ASSERT_TRUE(WIFEXITED(status)) << "Process exited abnormally.";
    ASSERT_EQ(WEXITSTATUS(status), 0);
  }
#endif
}

// Public API 1: Normal execution
void RunGenerator(const std::string &start, const std::string &end,
                  const std::string &op, const std::string &version) {
  ExecuteProcess(start, end, op, version);
}

// Public API 2: Execution with simulated crash
void RunGeneratorAndCrash(const std::string &start, const std::string &end,
                          const std::string &op, const std::string &version,
                          int seconds) {
  ExecuteProcess(start, end, op, version, seconds);
}


class CrashRecoveryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    zvec::test_util::RemoveTestPath("./write_recovery_test_db");
    ASSERT_NO_THROW(data_generator_bin_ = LocateDataGenerator());
  }

  void TearDown() override {
    zvec::test_util::RemoveTestPath("./write_recovery_test_db");
  }
};


TEST_F(CrashRecoveryTest, BasicInsertAndReopen) {
  {
    auto schema = CreateTestSchema(collection_name_);
    auto result = Collection::CreateAndOpen(dir_path_, *schema, options_);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    auto collection = result.value();
    collection.reset();
  }

  RunGenerator("0", "5000", "insert", "0");
  auto result = Collection::Open(dir_path_, options_);
  ASSERT_TRUE(result.has_value());
  auto collection = result.value();
  ASSERT_EQ(collection->Stats().value().doc_count, 5000)
      << "Document count mismatch";
}


TEST_F(CrashRecoveryTest, CrashRecoveryDuringInsertion) {
  {
    auto schema = CreateTestSchema(collection_name_);
    auto result = Collection::CreateAndOpen(dir_path_, *schema, options_);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    auto collection = result.value();
    collection.reset();
  }

  RunGeneratorAndCrash("0", "10000", "insert", "0", 3);

  auto result = Collection::Open(dir_path_, options_);
  ASSERT_TRUE(result.has_value()) << "Failed to reopen collection after crash. "
                                     "Recovery mechanism may be broken.";
  auto collection = result.value();
  uint64_t doc_count{collection->Stats().value().doc_count};
  ASSERT_GT(doc_count, 800)
      << "Document count is too low after 3s of insertion and recovery";

  for (uint64_t doc_id = 0; doc_id < doc_count; doc_id++) {
    const auto expected_doc = CreateTestDoc(doc_id, 0);
    std::vector<std::string> pks{};
    pks.emplace_back(expected_doc.pk());
    if (auto res = collection->Fetch(pks); res) {
      auto map = res.value();
      if (map.find(expected_doc.pk()) == map.end()) {
        FAIL() << "Returned map does not contain doc[" << expected_doc.pk()
               << "]";
      }
      const auto actual_doc = map.at(expected_doc.pk());
      ASSERT_EQ(*actual_doc, expected_doc)
          << "Data mismatch for doc[" << expected_doc.pk() << "]";
    } else {
      FAIL() << "Failed to fetch doc[" << expected_doc.pk() << "]";
    }
  }
}


TEST_F(CrashRecoveryTest, CrashRecoveryDuringUpsert) {
  {
    auto schema = CreateTestSchema(collection_name_);
    auto result = Collection::CreateAndOpen(dir_path_, *schema, options_);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    auto collection = result.value();
    collection.reset();
  }

  RunGenerator("0", "5000", "insert", "0");
  {
    auto result = Collection::Open(dir_path_, options_);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    auto collection = result.value();
    ASSERT_EQ(collection->Stats().value().doc_count, 5000)
        << "Document count mismatch";
  }

  RunGeneratorAndCrash("4500", "20000", "upsert", "1", 5);

  auto result = Collection::Open(dir_path_, options_);
  ASSERT_TRUE(result.has_value()) << "Failed to reopen collection after crash. "
                                     "Recovery mechanism may be broken.";
  auto collection = result.value();
  uint64_t doc_count{collection->Stats().value().doc_count};
  ASSERT_GT(doc_count, 6000)
      << "Document count is too low after 5s of insertion and recovery";

  for (uint64_t doc_id = 0; doc_id < doc_count; doc_id++) {
    Doc expected_doc;
    if (doc_id < 4500) {
      expected_doc = CreateTestDoc(doc_id, 0);
    } else {
      expected_doc = CreateTestDoc(doc_id, 1);
    }
    std::vector<std::string> pks{};
    pks.emplace_back(expected_doc.pk());
    if (auto res = collection->Fetch(pks); res) {
      auto map = res.value();
      if (map.find(expected_doc.pk()) == map.end()) {
        FAIL() << "Returned map does not contain doc[" << expected_doc.pk()
               << "]";
      }
      const auto actual_doc = map.at(expected_doc.pk());
      ASSERT_EQ(*actual_doc, expected_doc)
          << "Data mismatch for doc[" << expected_doc.pk() << "]";
    } else {
      FAIL() << "Failed to fetch doc[" << expected_doc.pk() << "]";
    }
  }
}


TEST_F(CrashRecoveryTest, CrashRecoveryDuringUpdate) {
  {
    auto schema = CreateTestSchema(collection_name_);
    auto result = Collection::CreateAndOpen(dir_path_, *schema, options_);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    auto collection = result.value();
    collection.reset();
  }

  RunGenerator("0", "18000", "upsert", "0");
  {
    auto result = Collection::Open(dir_path_, options_);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    auto collection = result.value();
    ASSERT_EQ(collection->Stats().value().doc_count, 18000)
        << "Document count mismatch";
  }

  RunGeneratorAndCrash("3000", "15000", "update", "3", 4);

  auto result = Collection::Open(dir_path_, options_);
  ASSERT_TRUE(result.has_value()) << "Failed to reopen collection after crash. "
                                     "Recovery mechanism may be broken.";
  auto collection = result.value();
  uint64_t doc_count{collection->Stats().value().doc_count};
  ASSERT_EQ(doc_count, 18000) << "Document count mismatch after crash recovery";

  for (int doc_id = 0; doc_id < 3500; doc_id++) {
    Doc expected_doc;
    if (doc_id < 3000) {
      expected_doc = CreateTestDoc(doc_id, 0);
    } else {
      expected_doc = CreateTestDoc(doc_id, 3);
    }
    std::vector<std::string> pks{};
    pks.emplace_back(expected_doc.pk());
    if (auto res = collection->Fetch(pks); res) {
      auto map = res.value();
      if (map.find(expected_doc.pk()) == map.end()) {
        FAIL() << "Returned map does not contain doc[" << expected_doc.pk()
               << "]";
      }
      const auto actual_doc = map.at(expected_doc.pk());
      ASSERT_EQ(*actual_doc, expected_doc)
          << "Data mismatch for doc[" << expected_doc.pk() << "]";
    } else {
      FAIL() << "Failed to fetch doc[" << expected_doc.pk() << "]";
    }
  }
}


TEST_F(CrashRecoveryTest, CrashRecoveryDuringDelete) {
  {
    auto schema = CreateTestSchema(collection_name_);
    auto result = Collection::CreateAndOpen(dir_path_, *schema, options_);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    auto collection = result.value();
    collection.reset();
  }

  RunGenerator("0", "18000", "insert", "0");
  {
    auto result = Collection::Open(dir_path_, options_);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    auto collection = result.value();
    ASSERT_EQ(collection->Stats().value().doc_count, 18000)
        << "Document count mismatch";
  }

  RunGeneratorAndCrash("3000", "15000", "delete", "0", 4);

  auto result = Collection::Open(dir_path_, options_);
  ASSERT_TRUE(result.has_value()) << "Failed to reopen collection after crash. "
                                     "Recovery mechanism may be broken.";
  auto collection = result.value();
  uint64_t doc_count{collection->Stats().value().doc_count};
  ASSERT_LT(doc_count, 18000)
      << "No deletes appear to have been applied before the crash";
  ASSERT_GT(doc_count, 6000)
      << "Too many documents deleted, recovery likely lost data";

  for (int doc_id = 0; doc_id < 3500; doc_id++) {
    auto expected_doc = CreateTestDoc(doc_id, 0);
    std::vector<std::string> pks{};
    pks.emplace_back(expected_doc.pk());
    if (auto res = collection->Fetch(pks); res) {
      auto map = res.value();
      auto it = map.find(expected_doc.pk());
      ASSERT_NE(it, map.end())
          << "Fetch result missing requested pk[" << expected_doc.pk() << "]";
      if (doc_id < 3000) {
        ASSERT_NE(it->second, nullptr)
            << "Existing doc returned as nullptr [" << expected_doc.pk() << "]";
        const auto actual_doc = map.at(expected_doc.pk());
        ASSERT_EQ(*actual_doc, expected_doc)
            << "Data mismatch for doc[" << expected_doc.pk() << "]";
      } else {
        ASSERT_EQ(it->second, nullptr)
            << "Returned doc for deleted pk[" << expected_doc.pk() << "]";
      }
    } else {
      FAIL() << "Failed to fetch doc[" << expected_doc.pk() << "]";
    }
  }
}


}  // namespace zvec
