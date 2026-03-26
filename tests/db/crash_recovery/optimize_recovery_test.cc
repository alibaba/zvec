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


static std::string optimizer_bin_;
const std::string collection_name_{"optimize_recovery_test"};
const std::string dir_path_{"optimize_recovery_test_db"};
const zvec::CollectionOptions options_{false, true, 256 * 1024};
const int batch_size{50};
const int num_batches{1000};


static std::string LocateOptimizeGenerator() {
  namespace fs = std::filesystem;
  const std::vector<std::string> candidates{"./collection_optimizer",
                                            "./bin/collection_optimizer"};
  for (const auto &p : candidates) {
    if (fs::exists(p)) {
      return fs::canonical(p).string();
    }
  }
  throw std::runtime_error("collection_optimizer binary not found");
}


void RunOptimizer(const std::string &path) {
  pid_t pid = fork();
  ASSERT_GE(pid, 0);

  if (pid == 0) {  // Child process
    char arg_path[] = "--path";
    char *args[] = {const_cast<char *>(optimizer_bin_.c_str()), arg_path,
                    const_cast<char *>(path.c_str()), nullptr};
    execvp(args[0], args);
    perror("execvp failed");
    _exit(1);
  }

  int status;
  waitpid(pid, &status, 0);
  ASSERT_TRUE(WIFEXITED(status))
      << "Child process did not exit normally. Terminated by signal?";
  int exit_code = WEXITSTATUS(status);
  ASSERT_EQ(exit_code, 0) << "optimizer failed with exit code: " << exit_code;
}


void RunOptimizerAndCrash(const std::string &path, int seconds) {
  pid_t pid = fork();
  ASSERT_GE(pid, 0);

  if (pid == 0) {  // Child process
    char arg_path[] = "--path";
    char *args[] = {const_cast<char *>(optimizer_bin_.c_str()), arg_path,
                    const_cast<char *>(path.c_str()), nullptr};
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


class OptimizeRecoveryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    system("rm -rf ./optimize_recovery_test_db");
    ASSERT_NO_THROW(optimizer_bin_ = LocateOptimizeGenerator());
  }

  void TearDown() override {
    system("rm -rf ./optimize_recovery_test_db");
  }
};


TEST_F(OptimizeRecoveryTest, CrashDuringOptimize) {
  {  // Create a collection and insert some documents
    auto schema = CreateTestSchema(collection_name_);
    auto result = Collection::CreateAndOpen(dir_path_, *schema, options_);
    ASSERT_TRUE(result.has_value());
    auto collection = result.value();

    for (int batch = 0; batch < num_batches; batch++) {
      std::vector<Doc> docs;
      for (int i = 0; i < batch_size; i++) {
        docs.push_back(CreateTestDoc(batch * batch_size + i, 0));
      }
      auto write_result = collection->Insert(docs);
      ASSERT_TRUE(write_result);
      for (auto &s : write_result.value()) {
        ASSERT_TRUE(s.ok());
      }
    }

    ASSERT_EQ(collection->Stats()->doc_count, num_batches * batch_size);
    collection.reset();
  }

  RunOptimizerAndCrash(dir_path_, 4);

  {  // Open the collection and verify data integrity
    auto result = Collection::Open(dir_path_, options_);
    ASSERT_TRUE(result.has_value())
        << "Failed to reopen collection after crash. "
           "Recovery mechanism may be broken.";
    auto collection = result.value();
    uint64_t doc_count{collection->Stats().value().doc_count};
    ASSERT_EQ(doc_count, num_batches * batch_size);
    for (uint64_t doc_id = 0; doc_id < doc_count; doc_id++) {
      Doc expected_doc = CreateTestDoc(doc_id, 0);
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

    VectorQuery query;
    query.topk_ = 10;
    std::vector<float> feature(128, 0.0);
    query.query_vector_.assign((const char *)feature.data(),
                               feature.size() * sizeof(float));
    query.field_name_ = "dense_fp32_field";
    auto query_result = collection->Query(query);
    ASSERT_TRUE(query_result);
    auto doc_list = query_result.value();
    ASSERT_EQ(doc_list.size(), 10);
    ASSERT_EQ(doc_list[0]->pk(), "pk_0");

    // Insert some more documents
    for (int batch = num_batches; batch < num_batches + 500; batch++) {
      std::vector<Doc> docs;
      for (int i = 0; i < batch_size; i++) {
        docs.push_back(CreateTestDoc(batch * batch_size + i, 0));
      }
      auto write_result = collection->Insert(docs);
      ASSERT_TRUE(write_result);
      for (auto &s : write_result.value()) {
        ASSERT_TRUE(s.ok());
      }
    }

    collection.reset();
  }

  RunOptimizer(dir_path_);

  // Open the collection and verify data integrity
  auto result = Collection::Open(dir_path_, options_);
  ASSERT_TRUE(result.has_value()) << "Failed to reopen collection after crash. "
                                     "Recovery mechanism may be broken.";
  auto collection = result.value();
  uint64_t doc_count{collection->Stats().value().doc_count};
  ASSERT_EQ(doc_count, (num_batches + 500) * batch_size);
  for (uint64_t doc_id = 0; doc_id < doc_count; doc_id++) {
    Doc expected_doc = CreateTestDoc(doc_id, 0);
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

  VectorQuery query;
  query.topk_ = 10;
  std::vector<float> feature(128, 0.0);
  query.query_vector_.assign((const char *)feature.data(),
                             feature.size() * sizeof(float));
  query.field_name_ = "dense_fp32_field";
  auto query_result = collection->Query(query);
  ASSERT_TRUE(query_result);
  auto doc_list = query_result.value();
  ASSERT_EQ(doc_list.size(), 10);
  ASSERT_EQ(doc_list[0]->pk(), "pk_0");
}


}  // namespace zvec
