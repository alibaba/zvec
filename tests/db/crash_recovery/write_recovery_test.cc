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
#include <thread>
#include <gtest/gtest.h>
#include <zvec/db/collection.h>
#include <zvec/db/doc.h>
#include <zvec/db/schema.h>
#include "zvec/ailego/logger/logger.h"
#include "utility.h"


namespace zvec {


const std::string data_generator_bin_{"./data_generator"};
const std::string collection_name_{"crash_test"};
const std::string dir_path_{"crash_test_db"};
const zvec::CollectionOptions options_{false, true};


class CrashRecoveryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    system("rm -rf ./crash_test_db");
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

  auto result = Collection::Open(dir_path_, options_);
  ASSERT_TRUE(result.has_value());
  auto collection = result.value();
  ASSERT_EQ(collection->Stats().value().doc_count, 5000)
      << "Document count mismatch";
}


TEST_F(CrashRecoveryTest, CrashRecovery_DuringInsertion) {
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
    // The exact count depends on CPU, but 3s should be enough for >1500 docs.
    std::this_thread::sleep_for(std::chrono::seconds(3));
    kill(pid, SIGKILL);

    int status;
    waitpid(pid, &status, 0);

    if (!WIFSIGNALED(status)) {
      FAIL() << "Child process was not killed by a signal. It exited normally?";
      return;
    }

    LOG_INFO("Successfully simulated crash (SIGKILL) during insertion.");
  }

  auto result = Collection::Open(dir_path_, options_);
  ASSERT_TRUE(result.has_value()) << "Failed to reopen collection after crash. "
                                     "Recovery mechanism may be broken.";
  auto collection = result.value();
  uint64_t doc_count{collection->Stats().value().doc_count};
  ASSERT_GT(doc_count, 1500)
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
          << "Data mismatch for doc_id[" << doc_id << "]";
    } else {
      FAIL() << "Failed to fetch doc[" << doc_id << "]";
    }
  }
}


}  // namespace zvec
