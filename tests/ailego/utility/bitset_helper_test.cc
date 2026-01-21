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

#include <cmath>
#include <random>
#include <ailego/utility/bitset_helper.h>
#include <gtest/gtest.h>
#include <zvec/ailego/utility/time_helper.h>

using namespace zvec;

TEST(BitsetHelper, Benchmark) {
  std::mt19937 gen((std::random_device())());
  std::uniform_int_distribution<uint32_t> dist(0, 0xffffffff);
  const size_t batch_size = 1000;
  const size_t dimension = 1024;

  std::vector<uint32_t> vec;
  for (size_t i = 0; i < batch_size; ++i) {
    for (size_t j = 0; j < (dimension >> 5); ++j) {
      vec.push_back(dist(gen));
    }
  }

  ailego::ElapsedTime elapsed_time;
  size_t count = (dimension >> 5);
  size_t total = 0;
  std::cout << "# " << dimension << "d, " << batch_size << std::endl;

  elapsed_time.reset();
  for (size_t i = 0; i < batch_size; ++i) {
    total += ailego::BitsetHelper::Cardinality(&vec[i * count], count);
  }
  printf("* Cardinality (us): \t%zu\n", (size_t)elapsed_time.micro_seconds());
  printf("* Result: \t%zu\n", total);
}
