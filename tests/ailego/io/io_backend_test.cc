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

#include <array>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/io/io_backend.h>

using namespace zvec::ailego;

TEST(IOBackend, ConcurrentProbeReturnsStableType) {
  constexpr size_t kThreadCount = 32;
  std::array<IOBackendType, kThreadCount> results{};
  std::vector<std::thread> threads;
  threads.reserve(kThreadCount);

  for (size_t i = 0; i < kThreadCount; ++i) {
    threads.emplace_back(
        [i, &results]() { results[i] = current_io_backend_type(); });
  }
  for (auto &thread : threads) {
    thread.join();
  }

  for (IOBackendType type : results) {
    EXPECT_EQ(type, results[0]);
  }
  std::string description = current_io_backend_description();
  EXPECT_FALSE(description.empty());
  const char *backend_name = "";
  switch (results[0]) {
    case IOBackendType::kIoUring:
      backend_name = "io_uring";
      break;
    case IOBackendType::kLibAio:
      backend_name = "libaio";
      break;
    case IOBackendType::kPread:
      backend_name = "pread";
      break;
  }
  EXPECT_NE(description.find(backend_name), std::string::npos);
}
