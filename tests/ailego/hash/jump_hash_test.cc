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

#include <random>
#include <set>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/hash/crc32c.h>
#include <zvec/ailego/hash/jump_hash.h>
#include <zvec/ailego/utility/time_helper.h>

using namespace zvec::ailego;

TEST(JumpHash, JumpHash) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint32_t> dist1(25353195, 25358555);
  std::uniform_int_distribution<uint32_t> dist2(1, 10000);
  std::set<uint32_t> result1;
  std::set<uint32_t> result2;

  const int total = 10000;
  for (int i = 0; i < total; ++i) {
    uint32_t ticket = dist1(gen);
    uint32_t signal = dist2(gen);

    uint64_t key = ((uint64_t)signal << 32) | ticket;
    uint32_t hash1 = (JumpHash(key, 32) << 27) | (ticket & 0x7ffffff);

    uint32_t hash2 = (signal << 27) | (ticket & 0x7ffffff);
    result1.insert(hash1);
    result2.insert(hash2);
  }
  printf("Conflict 1: %f\n", (double)(total - result1.size()) / (double)total);
  printf("Conflict 2: %f\n", (double)(total - result2.size()) / (double)total);
}
