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

#include <ailego/parallel/thread_pool.h>
#include <gtest/gtest.h>
#include <zvec/ailego/pattern/singleton.h>

using namespace zvec::ailego;

struct AAA {
  void run() {
    ++a;
  }
  uint32_t val() {
    return a;
  }
  std::atomic_uint a{0};
};

TEST(Singleton, General) {
  Singleton<int>::Instance() = 15;
  EXPECT_EQ(15, Singleton<int>::Instance());

  Singleton<double>::Instance() = 1.2;
  EXPECT_DOUBLE_EQ(1.2, Singleton<double>::Instance());

  ThreadPool pool1;
  for (int i = 0; i < 1000; ++i) {
    pool1.execute([] { Singleton<AAA>::Instance().run(); });
  }
  pool1.wait_finish();

  EXPECT_EQ(1000u, Singleton<AAA>::Instance().val());
}
