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
#include "utility.h"


namespace zvec {


static std::string data_generator_bin_;
const std::string collection_name_{"crash_test"};
const std::string dir_path_{"crash_test_db"};
const zvec::CollectionOptions options_{false, true};


static std::string LocateDataGenerator() {
  namespace fs = std::filesystem;
  const std::vector<std::string> candidates{"./data_generator",
                                            "./bin/data_generator"};
  for (const auto &p : candidates) {
    if (fs::exists(p)) {
      return fs::canonical(p).string();
    }
  }
  throw std::runtime_error("data_generator binary not found");
}


void RunGenerator(const std::string &start, const std::string &end,
                  const std::string &op) {
  pid_t pid = fork();
  ASSERT_GE(pid, 0);

  if (pid == 0) {  // Child process
    char arg_path[] = "--path";
    char arg_start[] = "--start";
    char arg_end[] = "--end";
    char arg_op[] = "--op";
    char *args[] = {const_cast<char *>(data_generator_bin_.c_str()),
                    arg_path,
                    const_cast<char *>(dir_path_.c_str()),
                    arg_start,
                    const_cast<char *>(start.c_str()),
                    arg_end,
                    const_cast<char *>(end.c_str()),
                    arg_op,
                    const_cast<char *>(op.c_str()),
                    nullptr};
    execvp(args[0], args);
    perror("execvp failed");
    _exit(1);
  }

  int status;
  waitpid(pid, &status, 0);
  ASSERT_TRUE(WIFEXITED(status))
      << "Child process did not exit normally. Terminated by signal?";
  int exit_code = WEXITSTATUS(status);
  ASSERT_EQ(exit_code, 0) << "data_generator failed with exit code: "
                          << exit_code;
}


void RunGeneratorAndCrash(const std::string &start, const std::string &end,
                          const std::string &op, int seconds) {
  pid_t pid = fork();
  ASSERT_GE(pid, 0);

  if (pid == 0) {  // Child process
    char arg_path[] = "--path";
    char arg_start[] = "--start";
    char arg_end[] = "--end";
    char arg_op[] = "--op";
    char *args[] = {const_cast<char *>(data_generator_bin_.c_str()),
                    arg_path,
                    const_cast<char *>(dir_path_.c_str()),
                    arg_start,
                    const_cast<char *>(start.c_str()),
                    arg_end,
                    const_cast<char *>(end.c_str()),
                    arg_op,
                    const_cast<char *>(op.c_str()),
                    nullptr};
    execvp(args[0], args);
    perror("execvp failed");
    _exit(1);
  }

  std::this_thread::sleep_for(std::chrono::seconds(seconds));
  if (kill(pid, 0) == 0) {
    kill(pid, SIGKILL);
  }
  int status;
  waitpid(pid, &status, 0);
  ASSERT_TRUE(WIFSIGNALED(status))
      << "Child process was not killed by a signal. It exited normally?";
}


class CrashRecoveryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    system("rm -rf ./crash_test_db");
    ASSERT_NO_THROW(data_generator_bin_ = LocateDataGenerator());
  }

  void TearDown() override {
    system("rm -rf ./crash_test_db");
  }
};


TEST_F(CrashRecoveryTest, BasicInsertAndReopen) {
  {
    auto schema = CreateTestSchema(collection_name_);
    auto result = Collection::CreateAndOpen(dir_path_, *schema, options_);
    ASSERT_TRUE(result.has_value());
    auto collection = result.value();
    collection.reset();
  }

  RunGenerator("0", "5000", "insert");
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
    ASSERT_TRUE(result.has_value());
    auto collection = result.value();
    collection.reset();
  }

  RunGeneratorAndCrash("0", "10000", "insert", 3);

  auto result = Collection::Open(dir_path_, options_);
  ASSERT_TRUE(result.has_value()) << "Failed to reopen collection after crash. "
                                     "Recovery mechanism may be broken.";
  auto collection = result.value();
  uint64_t doc_count{collection->Stats().value().doc_count};
  ASSERT_GT(doc_count, 800)
      << "Document count is too low after 3s of insertion and recovery";

  for (int doc_id = 0; doc_id < doc_count; doc_id++) {
    const auto expected_doc = CreateTestDoc(doc_id, false);
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
    ASSERT_TRUE(result.has_value());
    auto collection = result.value();
    collection.reset();
  }

  pid_t pid = fork();
  if (pid == 0) {  // Child process
    char arg_path[] = "--path";
    char arg_start[] = "--start";
    char val_start[] = "0";
    char arg_end[] = "--end";
    char val_end[] = "5000";
    char arg_op[] = "--op";
    char val_op[] = "insert";

    char *args[] = {const_cast<char *>(data_generator_bin_.c_str()),
                    arg_path,
                    const_cast<char *>(dir_path_.c_str()),
                    arg_start,
                    val_start,
                    arg_end,
                    val_end,
                    arg_op,
                    val_op,
                    nullptr};
    execvp(args[0], args);
    perror("execvp failed");
    _exit(1);
  } else {  // Parent process
    int status;
    waitpid(pid, &status, 0);

    if (!WIFEXITED(status)) {
      FAIL() << "Child process did not exit normally. Terminated by signal?";
      return;
    }

    int exit_code = WEXITSTATUS(status);
    if (exit_code != 0) {
      FAIL() << "data_generator failed with exit code: " << exit_code;
      return;
    }
  }

  {
    auto result = Collection::Open(dir_path_, options_);
    ASSERT_TRUE(result.has_value());
    auto collection = result.value();
    ASSERT_EQ(collection->Stats().value().doc_count, 5000)
        << "Document count mismatch";
  }

  pid = fork();
  if (pid == 0) {  // Child process
    char arg_path[] = "--path";
    char arg_start[] = "--start";
    char val_start[] = "3000";
    char arg_end[] = "--end";
    char val_end[] = "20000";
    char arg_op[] = "--op";
    char val_op[] = "upsert";

    char *args[] = {const_cast<char *>(data_generator_bin_.c_str()),
                    arg_path,
                    const_cast<char *>(dir_path_.c_str()),
                    arg_start,
                    val_start,
                    arg_end,
                    val_end,
                    arg_op,
                    val_op,
                    nullptr};
    execvp(args[0], args);
    perror("execvp failed");
    _exit(1);
  } else {  // Parent process
    std::this_thread::sleep_for(std::chrono::seconds(5));
    kill(pid, SIGKILL);

    int status;
    waitpid(pid, &status, 0);

    if (!WIFSIGNALED(status)) {
      FAIL() << "Child process was not killed by a signal. It exited normally?";
      return;
    }
  }

  auto result = Collection::Open(dir_path_, options_);
  ASSERT_TRUE(result.has_value()) << "Failed to reopen collection after crash. "
                                     "Recovery mechanism may be broken.";
  auto collection = result.value();
  uint64_t doc_count{collection->Stats().value().doc_count};
  ASSERT_GT(doc_count, 6500)
      << "Document count is too low after 5s of insertion and recovery";

  for (int doc_id = 0; doc_id < doc_count; doc_id++) {
    const auto expected_doc = CreateTestDoc(doc_id, false);
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


}  // namespace zvec
