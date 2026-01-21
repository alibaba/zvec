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
#include <thread>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/buffer_manager.h>
#include <zvec/ailego/logger/logger.h>

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

using namespace zvec::ailego;


const std::string working_dir{"./buffer_manager_dir/"};
const std::string file_path_forward{working_dir + "test.forward_index"};
const std::string file_path_vector{working_dir + "test.vector_index"};


class BufferManagerTest : public testing::Test {
  /*****  Global initialization and cleanup - Start  *****/
 public:
  static void SetUpTestCase() {
    char cmd_buf[100];
    snprintf(cmd_buf, 100, "rm -rf %s", working_dir.c_str());
    system(cmd_buf);

    if (!File::MakePath(working_dir)) {
      LOG_ERROR("Failed to create working directory.");
      return;
    }

    File file_vector_index;
    size_t file_vector_size = 16 * 1024 * 1024;
    if (!file_vector_index.create(file_path_vector, file_vector_size)) {
      LOG_ERROR("Failed to create vector index file.");
      return;
    }
    // Populate vector file with number series
    for (uint32_t i = 0; i < file_vector_size / sizeof(uint32_t); ++i) {
      file_vector_index.write((void *)&i, sizeof(i));
    }
    file_vector_index.close();

    BufferManager::Instance().init(4 * 1024 * 1024, 1);
  }

  static void TearDownTestCase() {}
  /*****  Global initialization and cleanup - End  *****/
  ;
};


TEST_F(BufferManagerTest, READ_VECTOR_FILE) {
  uint32_t size_4KB = 4 * 1024;

  auto read_and_verify_numbers = [&](uint32_t offset) {
    BufferID id = BufferID::VectorID(file_path_vector, offset, size_4KB);
    auto handle = BufferManager::Instance().acquire(id);
    uint32_t *vector_data = (uint32_t *)handle.pin_vector_data();
    uint32_t num_start = offset / sizeof(uint32_t);
    for (uint32_t i = 0; i < size_4KB / sizeof(uint32_t); i++) {
      ASSERT_EQ(*(vector_data + i), num_start + i);
    }
    handle.unpin_vector_data();
  };

  std::vector<std::thread> threads;

  // Read the same part concurrently
  for (int i = 0; i < 10; ++i) {
    threads.emplace_back(read_and_verify_numbers, 3 * size_4KB);
  }
  for (auto &thread : threads) {
    thread.join();
  }

  {  // Verify the reference count
    BufferID id = BufferID::VectorID(file_path_vector, 3 * size_4KB, size_4KB);
    auto handle = BufferManager::Instance().acquire(id);
    handle.pin_vector_data();
    ASSERT_EQ(handle.references(), 1);
    handle.unpin_vector_data();
    ASSERT_EQ(handle.references(), 0);
  }

  threads.clear();
  // Read different parts concurrently
  for (int i = 0; i < 30; ++i) {
    threads.emplace_back(read_and_verify_numbers, i * size_4KB);
  }
  for (auto &thread : threads) {
    thread.join();
  }
  ASSERT_EQ(BufferManager::Instance().total_size_in_bytes(), 30 * 4 * 1024);

  {  // Read a large chunk so that the buffer is full
    BufferID id =
        BufferID::VectorID(file_path_vector, 4 * 1024 * 1024, 4 * 1024 * 1024);
    auto handle = BufferManager::Instance().acquire(id);
    handle.pin_vector_data();
    handle.unpin_vector_data();
  }

  {  // Trigger eviction
    BufferID id =
        BufferID::VectorID(file_path_vector, 8 * 1024 * 1024, 4 * 1024 * 1024);
    auto handle = BufferManager::Instance().acquire(id);
    handle.pin_vector_data();
    ASSERT_EQ(BufferManager::Instance().total_size_in_bytes(), 4 * 1024 * 1024);
    handle.unpin_vector_data();
    ASSERT_EQ(handle.references(), 0);
  }
}

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif