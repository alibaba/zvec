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

#include <algorithm>
#include <bitset>
#include <iostream>
#include <memory>
#include <set>
#include <vector>
#include <ailego/container/bitmap.h>
#include <ailego/utility/bitset_helper.h>
#include <gtest/gtest.h>
#include <zvec/ailego/utility/time_helper.h>

#if defined(__AVX2__)
#define INTRINSICS_SET "AVX2"
#elif defined(__AVX__)
#define INTRINSICS_SET "AVX"
#elif defined(__SSE4_2__)
#define INTRINSICS_SET "SSE4.2"
#elif defined(__SSE4_1__)
#define INTRINSICS_SET "SSE4.1"
#elif defined(__SSE2__)
#define INTRINSICS_SET "SSE2"
#else
#define INTRINSICS_SET "NONE"
#endif

using namespace zvec::ailego;

TEST(FixedBitset, General) {
  FixedBitset<0> bitset0;
  FixedBitset<32> bitset32;
  FixedBitset<64> bitset64;

  EXPECT_EQ(0u, bitset0.size());
  EXPECT_EQ(32u, bitset32.size());
  EXPECT_EQ(64u, bitset64.size());

  EXPECT_TRUE(bitset32.test_none());
  EXPECT_TRUE(bitset64.test_none());

  bitset32.set(30);
  bitset64.set(60);

  FixedBitset<32> bitset32_2(bitset32);
  FixedBitset<64> bitset64_2(bitset64);

  bitset32.set(28);
  bitset64.set(55);

  EXPECT_TRUE(bitset32_2.test_any());
  EXPECT_TRUE(bitset64_2.test_any());

  EXPECT_FALSE(bitset32_2.test_all());
  EXPECT_FALSE(bitset64_2.test_all());

  EXPECT_EQ(1u, bitset32_2.cardinality());
  EXPECT_EQ(1u, bitset64_2.cardinality());

  bitset32_2 = bitset32;
  bitset64_2 = bitset64;

  EXPECT_EQ(2u, bitset32_2.cardinality());
  EXPECT_EQ(2u, bitset64_2.cardinality());

  bitset32.reset(28);
  bitset64.reset(55);

  bitset32_2 = bitset32;
  bitset64_2 = bitset64;

  EXPECT_EQ(1u, bitset32_2.cardinality());
  EXPECT_EQ(1u, bitset64_2.cardinality());

  bitset32.flip(30);
  bitset64.flip(60);

  EXPECT_EQ(0u, bitset32.cardinality());
  EXPECT_EQ(0u, bitset64.cardinality());
}

TEST(FixedBitset, And) {
  srand((uint32_t)time(NULL));
  srand((uint32_t)rand());

  FixedBitset<3552> bitset1;
  FixedBitset<3552> bitset2;
  FixedBitset<3552> bitset3;
  std::bitset<3552> stl_bitset1;
  std::bitset<3552> stl_bitset2;
  std::bitset<3552> stl_bitset3;

  for (uint32_t i = 0; i < 623; ++i) {
    uint32_t val1 = (uint32_t)(rand() % bitset1.size());
    uint32_t val2 = (uint32_t)(rand() % bitset2.size());

    bitset1.set(val1);
    stl_bitset1.set(val1);

    bitset2.set(val2);
    stl_bitset2.set(val2);
  }
  for (uint32_t i = 0; i < 623; ++i) {
    uint32_t val1 = (uint32_t)(rand() % bitset1.size());
    uint32_t val2 = (uint32_t)(rand() % bitset2.size());

    bitset1.flip(val1);
    stl_bitset1.flip(val1);

    bitset2.flip(val2);
    stl_bitset2.flip(val2);
  }

  bitset3 = bitset1;
  bitset3.bitwise_and(bitset2);
  stl_bitset3 = stl_bitset1 & stl_bitset2;

  for (uint32_t i = 0; i < bitset3.size(); ++i) {
    EXPECT_EQ(bitset3.test(i), stl_bitset3.test(i));
  }
  EXPECT_EQ(stl_bitset3.count(), bitset3.cardinality());

  FixedBitset<512>::Cast((uint32_t *)bitset3.data() + 1)
      ->bitwise_and(*(FixedBitset<512>::Cast((uint32_t *)bitset2.data() + 3)));
}

TEST(FixedBitset, Andnot) {
  srand((uint32_t)time(NULL));
  srand((uint32_t)rand());

  FixedBitset<2528> bitset1;
  FixedBitset<2528> bitset2;
  FixedBitset<2528> bitset3;
  std::bitset<2528> stl_bitset1;
  std::bitset<2528> stl_bitset2;
  std::bitset<2528> stl_bitset3;

  for (uint32_t i = 0; i < 623; ++i) {
    uint32_t val1 = (uint32_t)(rand() % bitset1.size());
    uint32_t val2 = (uint32_t)(rand() % bitset2.size());

    bitset1.set(val1);
    stl_bitset1.set(val1);

    bitset2.set(val2);
    stl_bitset2.set(val2);
  }
  for (uint32_t i = 0; i < 623; ++i) {
    uint32_t val1 = (uint32_t)(rand() % bitset1.size());
    uint32_t val2 = (uint32_t)(rand() % bitset2.size());

    bitset1.flip(val1);
    stl_bitset1.flip(val1);

    bitset2.flip(val2);
    stl_bitset2.flip(val2);
  }

  bitset3 = bitset1;
  bitset3.bitwise_andnot(bitset2);
  stl_bitset3 = stl_bitset1 & (~stl_bitset2);

  for (uint32_t i = 0; i < bitset3.size(); ++i) {
    EXPECT_EQ(bitset3.test(i), stl_bitset3.test(i));
  }
  EXPECT_EQ(stl_bitset3.count(), bitset3.cardinality());

  FixedBitset<512>::Cast((uint32_t *)bitset3.data() + 1)
      ->bitwise_andnot(
          *(FixedBitset<512>::Cast((uint32_t *)bitset2.data() + 3)));
}

TEST(FixedBitset, Or) {
  srand((uint32_t)time(NULL));
  srand((uint32_t)rand());

  FixedBitset<2528> bitset1;
  FixedBitset<2528> bitset2;
  FixedBitset<2528> bitset3;
  std::bitset<2528> stl_bitset1;
  std::bitset<2528> stl_bitset2;
  std::bitset<2528> stl_bitset3;

  for (uint32_t i = 0; i < 623; ++i) {
    uint32_t val1 = (uint32_t)(rand() % bitset1.size());
    uint32_t val2 = (uint32_t)(rand() % bitset2.size());

    bitset1.set(val1);
    stl_bitset1.set(val1);

    bitset2.set(val2);
    stl_bitset2.set(val2);
  }
  for (uint32_t i = 0; i < 623; ++i) {
    uint32_t val1 = (uint32_t)(rand() % bitset1.size());
    uint32_t val2 = (uint32_t)(rand() % bitset2.size());

    bitset1.flip(val1);
    stl_bitset1.flip(val1);

    bitset2.flip(val2);
    stl_bitset2.flip(val2);
  }

  bitset3 = bitset1;
  bitset3.bitwise_or(bitset2);
  stl_bitset3 = stl_bitset1 | stl_bitset2;

  for (uint32_t i = 0; i < bitset3.size(); ++i) {
    EXPECT_EQ(bitset3.test(i), stl_bitset3.test(i));
  }
  EXPECT_EQ(stl_bitset3.count(), bitset3.cardinality());

  FixedBitset<512>::Cast((uint32_t *)bitset3.data() + 1)
      ->bitwise_or(*(FixedBitset<512>::Cast((uint32_t *)bitset2.data() + 3)));
}

TEST(FixedBitset, Xor) {
  srand((uint32_t)time(NULL));
  srand((uint32_t)rand());

  FixedBitset<2528> bitset1;
  FixedBitset<2528> bitset2;
  FixedBitset<2528> bitset3;
  std::bitset<2528> stl_bitset1;
  std::bitset<2528> stl_bitset2;
  std::bitset<2528> stl_bitset3;

  for (uint32_t i = 0; i < 623; ++i) {
    uint32_t val1 = (uint32_t)(rand() % bitset1.size());
    uint32_t val2 = (uint32_t)(rand() % bitset2.size());

    bitset1.set(val1);
    stl_bitset1.set(val1);

    bitset2.set(val2);
    stl_bitset2.set(val2);
  }
  for (uint32_t i = 0; i < 623; ++i) {
    uint32_t val1 = (uint32_t)(rand() % bitset1.size());
    uint32_t val2 = (uint32_t)(rand() % bitset2.size());

    bitset1.flip(val1);
    stl_bitset1.flip(val1);

    bitset2.flip(val2);
    stl_bitset2.flip(val2);
  }

  bitset3 = bitset1;
  bitset3.bitwise_xor(bitset2);
  stl_bitset3 = stl_bitset1 ^ stl_bitset2;

  for (uint32_t i = 0; i < bitset3.size(); ++i) {
    EXPECT_EQ(bitset3.test(i), stl_bitset3.test(i));
  }
  EXPECT_EQ(stl_bitset3.count(), bitset3.cardinality());

  FixedBitset<512>::Cast((uint32_t *)bitset3.data() + 1)
      ->bitwise_xor(*(FixedBitset<512>::Cast((uint32_t *)bitset2.data() + 3)));
}

TEST(FixedBitset, Not) {
  FixedBitset<1504> bitset1;
  EXPECT_FALSE(bitset1.test_all());
  EXPECT_FALSE(bitset1.test_any());
  EXPECT_TRUE(bitset1.test_none());
  EXPECT_EQ(0u, bitset1.cardinality());

  for (uint32_t i = 0; i < bitset1.size(); ++i) {
    bitset1.set(i);
  }
  EXPECT_EQ(bitset1.size(), bitset1.cardinality());
  EXPECT_TRUE(bitset1.test_all());
  EXPECT_TRUE(bitset1.test_any());
  EXPECT_FALSE(bitset1.test_none());

  bitset1.bitwise_not();
  EXPECT_FALSE(bitset1.test_all());
  EXPECT_FALSE(bitset1.test_any());
  EXPECT_TRUE(bitset1.test_none());

  FixedBitset<512> bitset2;
  EXPECT_FALSE(bitset2.test_all());
  EXPECT_FALSE(bitset2.test_any());
  EXPECT_TRUE(bitset2.test_none());

  for (uint32_t i = 0; i < bitset2.size(); ++i) {
    bitset2.set(i);
  }
  EXPECT_TRUE(bitset2.test_all());
  EXPECT_TRUE(bitset2.test_any());
  EXPECT_FALSE(bitset2.test_none());

  bitset2.bitwise_not();
  EXPECT_FALSE(bitset2.test_all());
  EXPECT_FALSE(bitset2.test_any());
  EXPECT_TRUE(bitset2.test_none());

  FixedBitset<512 - 32>::Cast((uint32_t *)bitset2.data() + 1)->bitwise_not();
}

TEST(FixedBitset, TestAll) {
  FixedBitset<1504> bitset;
  EXPECT_FALSE(bitset.test_all());

  for (uint32_t i = 0; i < bitset.size(); ++i) {
    bitset.set(i);
  }
  EXPECT_TRUE(bitset.test_all());

  bitset.reset(999u);
  EXPECT_FALSE(bitset.test_all());
  EXPECT_FALSE(
      FixedBitset<1504 - 32>::Cast((uint32_t *)bitset.data() + 1)->test_all());
}

TEST(FixedBitset, TestAny) {
  FixedBitset<1504> bitset;
  EXPECT_FALSE(bitset.test_any());

  for (uint32_t i = 666; i < 888; ++i) {
    bitset.set(i);
  }
  EXPECT_TRUE(bitset.test_any());

  for (uint32_t i = 666; i < 777; ++i) {
    bitset.reset(i);
  }
  EXPECT_TRUE(bitset.test_any());
  EXPECT_TRUE(
      FixedBitset<1504 - 32>::Cast((uint32_t *)bitset.data() + 1)->test_any());
}

TEST(FixedBitset, TestNone) {
  FixedBitset<1504> bitset;
  EXPECT_TRUE(bitset.test_none());

  for (uint32_t i = 1000; i < 1111; ++i) {
    bitset.set(i);
  }
  EXPECT_FALSE(bitset.test_none());

  for (uint32_t i = 1000; i < 1110; ++i) {
    bitset.flip(i);
  }
  EXPECT_FALSE(bitset.test_none());
  EXPECT_FALSE(
      FixedBitset<1504 - 32>::Cast((uint32_t *)bitset.data() + 1)->test_none());
}

TEST(FixedBitset, Extract) {
  srand((uint32_t)time(NULL));
  srand((uint32_t)rand());

  FixedBitset<2528> bitset1;
  std::vector<size_t> vector1;

  for (uint32_t i = 0; i < 1111; ++i) {
    uint32_t val1 = (uint32_t)(rand() % bitset1.size());

    bitset1.set(val1);
    vector1.push_back(val1);
  }

  std::sort(vector1.begin(), vector1.end());
  vector1.erase(std::unique(vector1.begin(), vector1.end()), vector1.end());

  std::vector<size_t> vector2;
  bitset1.extract(&vector2);

  EXPECT_EQ(vector1.size(), vector2.size());
  EXPECT_TRUE(std::equal(vector1.begin(), vector1.end(), vector2.begin()));
}

TEST(FixedBitset, BitwiseXorCardinality) {
  srand((uint32_t)time(NULL));
  srand((uint32_t)rand());

  FixedBitset<2528> bitset1;
  FixedBitset<2528> bitset2;
  std::bitset<2528> stl_bitset1;
  std::bitset<2528> stl_bitset2;

  for (uint32_t i = 0; i < 600; ++i) {
    uint32_t val1 = (uint32_t)(rand() % bitset1.size());
    uint32_t val2 = (uint32_t)(rand() % bitset2.size());

    bitset1.set(val1);
    stl_bitset1.set(val1);

    bitset2.set(val2);
    stl_bitset2.set(val2);
  }
  for (uint32_t i = 0; i < 600; ++i) {
    uint32_t val1 = (uint32_t)(rand() % bitset1.size());
    uint32_t val2 = (uint32_t)(rand() % bitset2.size());

    bitset1.flip(val1);
    stl_bitset1.flip(val1);

    bitset2.flip(val2);
    stl_bitset2.flip(val2);
  }
  EXPECT_EQ(0u, FixedBitset<2528>::BitwiseXorCardinality(bitset1, bitset1));
  EXPECT_EQ(0u, FixedBitset<2528>::BitwiseXorCardinality(bitset2, bitset2));
  EXPECT_EQ((stl_bitset1 ^ stl_bitset2).count(),
            FixedBitset<2528>::BitwiseXorCardinality(bitset1, bitset2));

  EXPECT_EQ(FixedBitset<2528>::BitwiseAndnotCardinality(bitset1, bitset2) +
                FixedBitset<2528>::BitwiseAndnotCardinality(bitset2, bitset1),
            FixedBitset<2528>::BitwiseXorCardinality(bitset1, bitset2));
}

TEST(FixedBitset, BitwiseOrCardinality) {
  srand((uint32_t)time(NULL));
  srand((uint32_t)rand());

  FixedBitset<2528> bitset1;
  FixedBitset<2528> bitset2;
  std::bitset<2528> stl_bitset1;
  std::bitset<2528> stl_bitset2;

  for (uint32_t i = 0; i < 600; ++i) {
    uint32_t val1 = (uint32_t)(rand() % bitset1.size());
    uint32_t val2 = (uint32_t)(rand() % bitset2.size());

    bitset1.set(val1);
    stl_bitset1.set(val1);

    bitset2.set(val2);
    stl_bitset2.set(val2);
  }
  for (uint32_t i = 0; i < 600; ++i) {
    uint32_t val1 = (uint32_t)(rand() % bitset1.size());
    uint32_t val2 = (uint32_t)(rand() % bitset2.size());

    bitset1.flip(val1);
    stl_bitset1.flip(val1);

    bitset2.flip(val2);
    stl_bitset2.flip(val2);
  }
  EXPECT_EQ((stl_bitset1 | stl_bitset2).count(),
            FixedBitset<2528>::BitwiseOrCardinality(bitset1, bitset2));
}

TEST(FixedBitset, BitwiseAndCardinality) {
  srand((uint32_t)time(NULL));
  srand((uint32_t)rand());

  FixedBitset<2528> bitset1;
  FixedBitset<2528> bitset2;
  std::bitset<2528> stl_bitset1;
  std::bitset<2528> stl_bitset2;

  for (uint32_t i = 0; i < 600; ++i) {
    uint32_t val1 = (uint32_t)(rand() % bitset1.size());
    uint32_t val2 = (uint32_t)(rand() % bitset2.size());

    bitset1.set(val1);
    stl_bitset1.set(val1);

    bitset2.set(val2);
    stl_bitset2.set(val2);
  }
  for (uint32_t i = 0; i < 600; ++i) {
    uint32_t val1 = (uint32_t)(rand() % bitset1.size());
    uint32_t val2 = (uint32_t)(rand() % bitset2.size());

    bitset1.flip(val1);
    stl_bitset1.flip(val1);

    bitset2.flip(val2);
    stl_bitset2.flip(val2);
  }
  EXPECT_EQ((stl_bitset1 & stl_bitset2).count(),
            FixedBitset<2528>::BitwiseAndCardinality(bitset1, bitset2));
}

TEST(FixedBitset, BitwiseAndnotCardinality) {
  srand((uint32_t)time(NULL));
  srand((uint32_t)rand());

  FixedBitset<2528> bitset1;
  FixedBitset<2528> bitset2;
  std::bitset<2528> stl_bitset1;
  std::bitset<2528> stl_bitset2;

  for (uint32_t i = 0; i < 600; ++i) {
    uint32_t val1 = (uint32_t)(rand() % bitset1.size());
    uint32_t val2 = (uint32_t)(rand() % bitset2.size());

    bitset1.set(val1);
    stl_bitset1.set(val1);

    bitset2.set(val2);
    stl_bitset2.set(val2);
  }
  for (uint32_t i = 0; i < 600; ++i) {
    uint32_t val1 = (uint32_t)(rand() % bitset1.size());
    uint32_t val2 = (uint32_t)(rand() % bitset2.size());

    bitset1.flip(val1);
    stl_bitset1.flip(val1);

    bitset2.flip(val2);
    stl_bitset2.flip(val2);
  }
  EXPECT_EQ((stl_bitset1 & ~stl_bitset2).count(),
            FixedBitset<2528>::BitwiseAndnotCardinality(bitset1, bitset2));

  EXPECT_EQ((stl_bitset2 & ~stl_bitset1).count(),
            FixedBitset<2528>::BitwiseAndnotCardinality(bitset2, bitset1));
}

TEST(FixedBitset, Benchmark) {
  srand((uint32_t)time(NULL));
  srand((uint32_t)rand());

  const uint32_t dimension = 2048u;
  const uint32_t test_count = 100000u;

  std::vector<FixedBitset<dimension>> bucket1_vec;
  std::vector<FixedBitset<dimension>> bucket2_vec;

  std::unique_ptr<FixedBitset<dimension>> bucket1(new FixedBitset<dimension>);
  std::unique_ptr<FixedBitset<dimension>> bucket2(new FixedBitset<dimension>);

  for (uint32_t i = 0; i < 2000; ++i) {
    uint32_t val1 = (uint32_t)(rand() % bucket1->size());
    uint32_t val2 = (uint32_t)(rand() % bucket2->size());

    bucket1->set(val1);
    bucket2->set(val2);
  }
  for (uint32_t i = 0; i < 1000; ++i) {
    uint32_t val1 = (uint32_t)(rand() % bucket1->size());
    uint32_t val2 = (uint32_t)(rand() % bucket2->size());

    bucket1->flip(val1);
    bucket2->flip(val2);
  }
  for (uint32_t i = 0; i < 500; ++i) {
    uint32_t val1 = (uint32_t)(rand() % bucket1->size());
    uint32_t val2 = (uint32_t)(rand() % bucket2->size());

    bucket1->reset(val1);
    bucket2->reset(val2);
  }

  bucket1_vec.reserve(test_count);
  bucket2_vec.reserve(test_count);
  for (uint32_t j = 0; j < test_count; ++j) {
    bucket1_vec.push_back(*bucket1);
    bucket2_vec.push_back(*bucket2);
  }

  {
    uint64_t t1 = Monotime::MicroSeconds();
    uint64_t sum = 0;
    for (uint32_t i = 0; i < test_count; ++i) {
      sum += FixedBitset<dimension>::BitwiseAndCardinality(bucket1_vec[i],
                                                           bucket2_vec[i]);
    }
    std::cout << INTRINSICS_SET
              << " BitwiseAndCardinality: " << Monotime::MicroSeconds() - t1
              << " us, sum: " << sum << std::endl;
  }

  {
    uint64_t t1 = Monotime::MicroSeconds();
    uint64_t sum = 0;
    for (uint32_t i = 0; i < test_count; ++i) {
      sum += FixedBitset<dimension>::BitwiseAndnotCardinality(bucket1_vec[i],
                                                              bucket2_vec[i]);
    }
    std::cout << INTRINSICS_SET
              << " BitwiseAndnotCardinality: " << Monotime::MicroSeconds() - t1
              << " us, sum: " << sum << std::endl;
  }

  {
    uint64_t t1 = Monotime::MicroSeconds();
    uint64_t sum = 0;
    for (uint32_t i = 0; i < test_count; ++i) {
      sum += FixedBitset<dimension>::BitwiseXorCardinality(bucket1_vec[i],
                                                           bucket2_vec[i]);
    }
    std::cout << INTRINSICS_SET
              << " BitwiseXorCardinality: " << Monotime::MicroSeconds() - t1
              << " us, sum: " << sum << std::endl;
  }

  {
    uint64_t t1 = Monotime::MicroSeconds();
    uint64_t sum = 0;
    for (uint32_t i = 0; i < test_count; ++i) {
      sum += FixedBitset<dimension>::BitwiseOrCardinality(bucket1_vec[i],
                                                          bucket2_vec[i]);
    }
    std::cout << INTRINSICS_SET
              << " BitwiseOrCardinality: " << Monotime::MicroSeconds() - t1
              << " us, sum: " << sum << std::endl;
  }

  {
    std::unique_ptr<FixedBitset<dimension>> bucket3(new FixedBitset<dimension>);
    *bucket3 = bucket1_vec[0];

    uint64_t t1 = Monotime::MicroSeconds();
    for (uint32_t i = 0; i < test_count; ++i) {
      bucket3->bitwise_and(bucket2_vec[i]);
    }
    std::cout << INTRINSICS_SET << " And: " << Monotime::MicroSeconds() - t1
              << " us" << std::endl;
  }

  {
    std::unique_ptr<FixedBitset<dimension>> bucket3(new FixedBitset<dimension>);
    *bucket3 = bucket1_vec[0];

    uint64_t t1 = Monotime::MicroSeconds();
    for (uint32_t i = 0; i < test_count; ++i) {
      bucket3->bitwise_andnot(bucket2_vec[i]);
    }
    std::cout << INTRINSICS_SET << " Andnot: " << Monotime::MicroSeconds() - t1
              << " us" << std::endl;
  }

  {
    std::unique_ptr<FixedBitset<dimension>> bucket3(new FixedBitset<dimension>);
    *bucket3 = bucket1_vec[0];

    uint64_t t1 = Monotime::MicroSeconds();
    for (uint32_t i = 0; i < test_count; ++i) {
      bucket3->bitwise_or(bucket2_vec[i]);
    }
    std::cout << INTRINSICS_SET << " Or: " << Monotime::MicroSeconds() - t1
              << " us" << std::endl;
  }

  {
    std::unique_ptr<FixedBitset<dimension>> bucket3(new FixedBitset<dimension>);
    *bucket3 = bucket1_vec[0];

    uint64_t t1 = Monotime::MicroSeconds();
    for (uint32_t i = 0; i < test_count; ++i) {
      bucket3->bitwise_xor(bucket2_vec[i]);
    }
    std::cout << INTRINSICS_SET << " Xor: " << Monotime::MicroSeconds() - t1
              << " us" << std::endl;
  }
}

TEST(Bitset, General) {
  Bitset bitset32(31);
  Bitset bitset64(61);

  EXPECT_EQ(32u, bitset32.size());
  EXPECT_EQ(64u, bitset64.size());

  EXPECT_TRUE(bitset32.test_none());
  EXPECT_TRUE(bitset64.test_none());

  bitset32.set(30);
  bitset64.set(60);

  Bitset bitset32_2(bitset32);
  Bitset bitset64_2(bitset64);

  bitset32.set(28);
  bitset64.set(55);

  EXPECT_TRUE(bitset32_2.test_any());
  EXPECT_TRUE(bitset64_2.test_any());

  EXPECT_FALSE(bitset32_2.test_all());
  EXPECT_FALSE(bitset64_2.test_all());

  EXPECT_EQ(1u, bitset32_2.cardinality());
  EXPECT_EQ(1u, bitset64_2.cardinality());

  bitset32_2 = bitset32;
  bitset64_2 = bitset64;

  EXPECT_EQ(2u, bitset32_2.cardinality());
  EXPECT_EQ(2u, bitset64_2.cardinality());

  bitset32.reset(28);
  bitset64.reset(55);

  bitset32_2 = bitset32;
  bitset64_2 = bitset64;

  EXPECT_EQ(1u, bitset32_2.cardinality());
  EXPECT_EQ(1u, bitset64_2.cardinality());

  bitset32.flip(30);
  bitset64.flip(60);

  EXPECT_EQ(0u, bitset32.cardinality());
  EXPECT_EQ(0u, bitset64.cardinality());
}

TEST(Bitset, BitwiseXorCardinality) {
  srand((uint32_t)time(NULL));
  srand((uint32_t)rand());

  Bitset bitset1;
  Bitset bitset2;
  bitset1.resize(500000);
  bitset2.resize(630000);
  std::bitset<638888> stl_bitset1;
  std::bitset<638888> stl_bitset2;

  for (uint32_t i = 0; i < 800; ++i) {
    uint32_t val1 = (uint32_t)(rand() % bitset1.size());
    uint32_t val2 = (uint32_t)(rand() % bitset2.size());

    bitset1.set(val1);
    stl_bitset1.set(val1);

    bitset2.set(val2);
    stl_bitset2.set(val2);
  }
  for (uint32_t i = 0; i < 600; ++i) {
    uint32_t val1 = (uint32_t)(rand() % bitset1.size());
    ;
    uint32_t val2 = (uint32_t)(rand() % bitset2.size());

    bitset1.flip(val1);
    stl_bitset1.flip(val1);

    bitset2.flip(val2);
    stl_bitset2.flip(val2);
  }
  EXPECT_EQ((stl_bitset1 ^ stl_bitset2).count(),
            Bitset::BitwiseXorCardinality(bitset1, bitset2));

  EXPECT_EQ(Bitset::BitwiseAndnotCardinality(bitset1, bitset2) +
                Bitset::BitwiseAndnotCardinality(bitset2, bitset1),
            Bitset::BitwiseXorCardinality(bitset1, bitset2));
  EXPECT_EQ(Bitset::BitwiseXorCardinality(bitset1, bitset2),
            Bitset::BitwiseXorCardinality(bitset2, bitset1));
}

TEST(Bitset, BitwiseOrCardinality) {
  srand((uint32_t)time(NULL));
  srand((uint32_t)rand());

  Bitset bitset1;
  Bitset bitset2;
  bitset1.resize(599999);
  bitset2.resize(500000);

  std::bitset<638888> stl_bitset1;
  std::bitset<638888> stl_bitset2;

  for (uint32_t i = 0; i < 800; ++i) {
    uint32_t val1 = (uint32_t)(rand() % bitset1.size());
    uint32_t val2 = (uint32_t)(rand() % bitset2.size());

    bitset1.set(val1);
    stl_bitset1.set(val1);

    bitset2.set(val2);
    stl_bitset2.set(val2);
  }
  for (uint32_t i = 0; i < 600; ++i) {
    uint32_t val1 = (uint32_t)(rand() % bitset1.size());
    uint32_t val2 = (uint32_t)(rand() % bitset2.size());

    bitset1.flip(val1);
    stl_bitset1.flip(val1);

    bitset2.flip(val2);
    stl_bitset2.flip(val2);
  }
  EXPECT_EQ((stl_bitset1 | stl_bitset2).count(),
            Bitset::BitwiseOrCardinality(bitset1, bitset2));
  EXPECT_EQ(Bitset::BitwiseOrCardinality(bitset1, bitset2),
            Bitset::BitwiseOrCardinality(bitset2, bitset1));
}

TEST(Bitset, BitwiseAndCardinality) {
  srand((uint32_t)time(NULL));
  srand((uint32_t)rand());

  Bitset bitset1;
  Bitset bitset2;
  bitset1.resize(500001);
  bitset2.resize(599999);

  std::bitset<638888> stl_bitset1;
  std::bitset<638888> stl_bitset2;

  for (uint32_t i = 0; i < 800; ++i) {
    uint32_t val1 = (uint32_t)(rand() % bitset1.size());
    uint32_t val2 = (uint32_t)(rand() % bitset2.size());

    bitset1.set(val1);
    stl_bitset1.set(val1);

    bitset2.set(val2);
    stl_bitset2.set(val2);
  }
  for (uint32_t i = 0; i < 600; ++i) {
    uint32_t val1 = (uint32_t)(rand() % bitset1.size());
    uint32_t val2 = (uint32_t)(rand() % bitset2.size());

    bitset1.flip(val1);
    stl_bitset1.flip(val1);

    bitset2.flip(val2);
    stl_bitset2.flip(val2);
  }
  EXPECT_EQ((stl_bitset1 & stl_bitset2).count(),
            Bitset::BitwiseAndCardinality(bitset1, bitset2));
  EXPECT_EQ(Bitset::BitwiseAndCardinality(bitset1, bitset2),
            Bitset::BitwiseAndCardinality(bitset2, bitset1));
}

TEST(Bitset, BitwiseAndnotCardinality) {
  srand((uint32_t)time(NULL));
  srand((uint32_t)rand());

  Bitset bitset1;
  Bitset bitset2;
  bitset1.resize(599997);
  bitset2.resize(500002);

  std::bitset<638888> stl_bitset1;
  std::bitset<638888> stl_bitset2;

  for (uint32_t i = 0; i < 800; ++i) {
    uint32_t val1 = (uint32_t)(rand() % bitset1.size());
    uint32_t val2 = (uint32_t)(rand() % bitset2.size());

    bitset1.set(val1);
    stl_bitset1.set(val1);

    bitset2.set(val2);
    stl_bitset2.set(val2);
  }
  for (uint32_t i = 0; i < 600; ++i) {
    uint32_t val1 = (uint32_t)(rand() % bitset1.size());
    uint32_t val2 = (uint32_t)(rand() % bitset2.size());

    bitset1.flip(val1);
    stl_bitset1.flip(val1);

    bitset2.flip(val2);
    stl_bitset2.flip(val2);
  }
  EXPECT_EQ((stl_bitset1 & ~stl_bitset2).count(),
            Bitset::BitwiseAndnotCardinality(bitset1, bitset2));

  EXPECT_EQ((stl_bitset2 & ~stl_bitset1).count(),
            Bitset::BitwiseAndnotCardinality(bitset2, bitset1));
}

TEST(Bitmap, General) {
  const uint32_t data1[] = {0,     1,      2,      4,      7,    9,
                            31,    65,     77,     100,    1000, 1999,
                            19999, 100000, 188888, 2999999};
  const uint32_t data2[] = {8,     12,    13,    24,     7777,      9999,
                            66666, 88888, 99999, 100002, 0x7fffffff};
  Bitmap bitmap1;

  EXPECT_EQ(0u, bitmap1.cardinality());
  for (size_t i = 0; i < sizeof(data1) / sizeof(data1[0]); ++i) {
    bitmap1.set(data1[i]);
  }

  // Test `Set`
  Bitmap bitmap2(bitmap1);

  EXPECT_NE(0u, bitmap2.cardinality());
  EXPECT_EQ(sizeof(data1) / sizeof(data1[0]), bitmap2.cardinality());
  for (size_t i = 0; i < sizeof(data1) / sizeof(data1[0]); ++i) {
    EXPECT_TRUE(bitmap2.test(data1[i]));
  }

  // Test `Reset`
  for (size_t i = 0; i < sizeof(data2) / sizeof(data2[0]); ++i) {
    bitmap1.reset(data2[i]);
  }
  for (size_t i = 0; i < sizeof(data2) / sizeof(data2[0]); ++i) {
    EXPECT_FALSE(bitmap1.test(data2[i]));
  }

  EXPECT_EQ(sizeof(data1) / sizeof(data1[0]), bitmap1.cardinality());
  for (size_t i = 0; i < sizeof(data1) / sizeof(data1[0]); ++i) {
    bitmap1.reset(data1[i]);
  }
  EXPECT_EQ(0u, bitmap1.cardinality());

  // Test `Flip`
  for (size_t i = 0; i < sizeof(data1) / sizeof(data1[0]); ++i) {
    bitmap1.flip(data1[i]);
  }
  for (size_t i = 0; i < sizeof(data2) / sizeof(data2[0]); ++i) {
    bitmap1.flip(data2[i]);
  }
  EXPECT_EQ(sizeof(data1) / sizeof(data1[0]) + sizeof(data2) / sizeof(data2[0]),
            bitmap1.cardinality());

  bitmap2 = bitmap1;
  for (size_t i = 0; i < sizeof(data1) / sizeof(data1[0]); ++i) {
    EXPECT_TRUE(bitmap2.test(data1[i]));
  }
  for (size_t i = 0; i < sizeof(data2) / sizeof(data2[0]); ++i) {
    EXPECT_TRUE(bitmap2.test(data2[i]));
  }

  // Test `ShrinkToFit`
  bitmap1.shrink_to_fit();
  for (size_t i = 0; i < sizeof(data1) / sizeof(data1[0]); ++i) {
    EXPECT_TRUE(bitmap1.test(data1[i]));
  }
  for (size_t i = 0; i < sizeof(data2) / sizeof(data2[0]); ++i) {
    EXPECT_TRUE(bitmap1.test(data2[i]));
  }

  // Test `Clear`
  EXPECT_NE(0u, bitmap1.cardinality());
  bitmap2 = bitmap1;
  bitmap1.clear();
  EXPECT_EQ(0u, bitmap1.cardinality());
  for (size_t i = 0; i < sizeof(data1) / sizeof(data1[0]); ++i) {
    EXPECT_FALSE(bitmap1.test(data1[i]));
  }
  for (size_t i = 0; i < sizeof(data2) / sizeof(data2[0]); ++i) {
    EXPECT_FALSE(bitmap1.test(data2[i]));
  }
  for (size_t i = 0; i < sizeof(data1) / sizeof(data1[0]); ++i) {
    EXPECT_TRUE(bitmap2.test(data1[i]));
  }
  for (size_t i = 0; i < sizeof(data2) / sizeof(data2[0]); ++i) {
    EXPECT_TRUE(bitmap2.test(data2[i]));
  }
}

TEST(Bitmap, ShrinkToFit) {
  Bitmap bitmap1;
  bitmap1.shrink_to_fit();

  EXPECT_EQ(0u, bitmap1.bucket_size());
  bitmap1.set(2);
  EXPECT_EQ(1u, bitmap1.bucket_size());
  bitmap1.reset(2);
  EXPECT_EQ(1u, bitmap1.bucket_size());
  bitmap1.shrink_to_fit();
  EXPECT_EQ(0u, bitmap1.bucket_size());

  bitmap1.set(100);
  bitmap1.set(100000);
  bitmap1.set(1000000);
  EXPECT_EQ((1000000u + 0xffff) / 0x10000, bitmap1.bucket_size());

  bitmap1.reset(100);
  bitmap1.reset(1000000);
  EXPECT_EQ((1000000u + 0xffff) / 0x10000, bitmap1.bucket_size());
  bitmap1.shrink_to_fit();
  EXPECT_EQ((100000u + 0xffff) / 0x10000, bitmap1.bucket_size());
}

TEST(Bitmap, And) {
  srand((uint32_t)time(NULL));
  srand((uint32_t)rand());

  Bitmap bitmap1, bitmap2;
  std::set<size_t> set1, set2, set3;
  std::vector<size_t> vec1, vec3;

  for (uint32_t i = 0; i < 25000; ++i) {
    uint32_t val1 = rand() % 1000000;
    bitmap1.set(val1);
    set1.insert(val1);
  }

  for (uint32_t i = 0; i < 45000; ++i) {
    uint32_t val2 = rand() % 1000000;
    bitmap2.set(val2);
    set2.insert(val2);
  }
  std::set_intersection(set1.begin(), set1.end(), set2.begin(), set2.end(),
                        std::inserter(set3, set3.begin()));
  bitmap1.bitwise_and(bitmap2);
  bitmap1.extract(&vec1);
  ASSERT_EQ(bitmap1.cardinality(), vec1.size());
  ASSERT_EQ(set3.size(), vec1.size());

  vec3.reserve(set3.size());
  std::copy(set3.begin(), set3.end(), std::back_inserter(vec3));
  EXPECT_TRUE(std::equal(vec1.begin(), vec1.end(), vec3.begin()));
}

TEST(Bitmap, Andnot) {
  srand((uint32_t)time(NULL));
  srand((uint32_t)rand());

  Bitmap bitmap1, bitmap2;
  std::set<size_t> set1, set2, set3;
  std::vector<size_t> vec1, vec3;

  for (uint32_t i = 0; i < 20000; ++i) {
    uint32_t val1 = rand() % 1000000;
    bitmap1.set(val1);
    set1.insert(val1);
  }

  for (uint32_t i = 0; i < 20000; ++i) {
    uint32_t val2 = rand() % 1000000;
    bitmap2.set(val2);
    set2.insert(val2);
  }
  std::set_difference(set1.begin(), set1.end(), set2.begin(), set2.end(),
                      std::inserter(set3, set3.begin()));
  bitmap1.bitwise_andnot(bitmap2);
  bitmap1.extract(&vec1);
  ASSERT_EQ(bitmap1.cardinality(), vec1.size());
  ASSERT_EQ(set3.size(), vec1.size());

  vec3.reserve(set3.size());
  std::copy(set3.begin(), set3.end(), std::back_inserter(vec3));
  EXPECT_TRUE(std::equal(vec1.begin(), vec1.end(), vec3.begin()));
}

TEST(Bitmap, Or) {
  srand((uint32_t)time(NULL));
  srand((uint32_t)rand());

  Bitmap bitmap1, bitmap2;
  std::set<size_t> set1, set2, set3;
  std::vector<size_t> vec1, vec3;

  for (uint32_t i = 0; i < 3000; ++i) {
    uint32_t val1 = rand() % 2000000;
    bitmap1.set(val1);
    set1.insert(val1);
  }

  for (uint32_t i = 0; i < 2000; ++i) {
    uint32_t val2 = rand() % 2000000;
    bitmap2.set(val2);
    set2.insert(val2);
  }
  std::set_union(set1.begin(), set1.end(), set2.begin(), set2.end(),
                 std::inserter(set3, set3.begin()));
  bitmap1.bitwise_or(bitmap2);
  bitmap1.extract(&vec1);
  ASSERT_EQ(bitmap1.cardinality(), vec1.size());
  ASSERT_EQ(set3.size(), vec1.size());

  vec3.reserve(set3.size());
  std::copy(set3.begin(), set3.end(), std::back_inserter(vec3));
  EXPECT_TRUE(std::equal(vec1.begin(), vec1.end(), vec3.begin()));
}

TEST(Bitmap, Xor) {
  srand((uint32_t)time(NULL));
  srand((uint32_t)rand());

  Bitmap bitmap1, bitmap2;
  std::set<size_t> set1, set2, set3;
  std::vector<size_t> vec1, vec3;

  for (uint32_t i = 0; i < 3000; ++i) {
    uint32_t val1 = rand() % 2000000;
    bitmap1.set(val1);
    set1.insert(val1);
  }

  for (uint32_t i = 0; i < 2000; ++i) {
    uint32_t val2 = rand() % 2000000;
    bitmap2.set(val2);
    set2.insert(val2);
  }
  std::set_symmetric_difference(set1.begin(), set1.end(), set2.begin(),
                                set2.end(), std::inserter(set3, set3.begin()));
  bitmap1.bitwise_xor(bitmap2);
  bitmap1.extract(&vec1);
  ASSERT_EQ(bitmap1.cardinality(), vec1.size());
  ASSERT_EQ(set3.size(), vec1.size());

  vec3.reserve(set3.size());
  std::copy(set3.begin(), set3.end(), std::back_inserter(vec3));
  EXPECT_TRUE(std::equal(vec1.begin(), vec1.end(), vec3.begin()));
}

TEST(Bitmap, Not) {
  Bitmap bitmap1, bitmap2, bitmap3;
  std::set<size_t> set1, set2, set3;
  std::vector<size_t> vec1;

  for (uint32_t i = 0; i < 20000; ++i) {
    uint32_t val1 = rand() % 1000000;
    bitmap1.set(val1);
    set1.insert(val1);
  }

  for (uint32_t i = 0; i < 20000; ++i) {
    uint32_t val2 = rand() % 1000000;
    bitmap2.set(val2);
    set2.insert(val2);
  }

  bitmap2.bitwise_not();
  bitmap2.bitwise_not();

  {
    set3.clear();
    vec1.clear();
    std::set_intersection(set1.begin(), set1.end(), set2.begin(), set2.end(),
                          std::inserter(set3, set3.begin()));

    bitmap3 = bitmap1;
    bitmap3.bitwise_and(bitmap2);
    bitmap3.extract(&vec1);
    ASSERT_EQ(bitmap3.cardinality(), vec1.size());
    ASSERT_EQ(set3.size(), vec1.size());

    std::vector<size_t> vec3(set3.begin(), set3.end());
    EXPECT_TRUE(std::equal(vec1.begin(), vec1.end(), vec3.begin()));
  }

  {
    set3.clear();
    vec1.clear();
    std::set_difference(set1.begin(), set1.end(), set2.begin(), set2.end(),
                        std::inserter(set3, set3.begin()));

    bitmap3 = bitmap1;
    bitmap3.bitwise_andnot(bitmap2);
    bitmap3.extract(&vec1);
    ASSERT_EQ(bitmap3.cardinality(), vec1.size());
    ASSERT_EQ(set3.size(), vec1.size());

    std::vector<size_t> vec3(set3.begin(), set3.end());
    EXPECT_TRUE(std::equal(vec1.begin(), vec1.end(), vec3.begin()));
  }

  {
    set3.clear();
    vec1.clear();
    std::set_union(set1.begin(), set1.end(), set2.begin(), set2.end(),
                   std::inserter(set3, set3.begin()));

    bitmap3 = bitmap1;
    bitmap3.bitwise_or(bitmap2);
    bitmap3.extract(&vec1);
    ASSERT_EQ(bitmap3.cardinality(), vec1.size());
    ASSERT_EQ(set3.size(), vec1.size());

    std::vector<size_t> vec3(set3.begin(), set3.end());
    EXPECT_TRUE(std::equal(vec1.begin(), vec1.end(), vec3.begin()));
  }

  {
    set3.clear();
    vec1.clear();
    std::set_symmetric_difference(set1.begin(), set1.end(), set2.begin(),
                                  set2.end(),
                                  std::inserter(set3, set3.begin()));

    bitmap3 = bitmap1;
    bitmap3.bitwise_xor(bitmap2);
    bitmap3.extract(&vec1);
    ASSERT_EQ(bitmap3.cardinality(), vec1.size());
    ASSERT_EQ(set3.size(), vec1.size());

    std::vector<size_t> vec3(set3.begin(), set3.end());
    EXPECT_TRUE(std::equal(vec1.begin(), vec1.end(), vec3.begin()));
  }
}

TEST(Bitmap, TestAll) {
  Bitmap bitmap;
  EXPECT_FALSE(bitmap.test_all());

  for (uint32_t i = 0; i < Bitmap::Bucket::MAX_SIZE * 2; ++i) {
    bitmap.set(i);
  }
  EXPECT_TRUE(bitmap.test_all());

  bitmap.reset(Bitmap::Bucket::MAX_SIZE + 2);
  EXPECT_FALSE(bitmap.test_all());
}

TEST(Bitmap, TestAny) {
  Bitmap bitmap;
  EXPECT_FALSE(bitmap.test_any());

  for (uint32_t i = 69000; i < 70000; ++i) {
    bitmap.set(i);
  }
  EXPECT_TRUE(bitmap.test_any());

  for (uint32_t i = 69888; i < 70111; ++i) {
    bitmap.reset(i);
  }
  EXPECT_TRUE(bitmap.test_any());
}

TEST(Bitmap, TestNone) {
  Bitmap bitmap;
  EXPECT_TRUE(bitmap.test_none());

  for (uint32_t i = 65000; i < 70000; ++i) {
    bitmap.set(i);
  }
  EXPECT_FALSE(bitmap.test_none());

  for (uint32_t i = 65555; i < 70022; ++i) {
    bitmap.flip(i);
  }
  EXPECT_FALSE(bitmap.test_none());
}

TEST(Bitmap, Extract) {
  srand((uint32_t)time(NULL));
  srand((uint32_t)rand());

  Bitmap bitmap1;
  std::vector<size_t> vector1;

  for (uint32_t i = 0; i < 1111; ++i) {
    uint32_t val1 = rand();

    bitmap1.set(val1);
    vector1.push_back(val1);
  }

  std::sort(vector1.begin(), vector1.end());
  vector1.erase(std::unique(vector1.begin(), vector1.end()), vector1.end());

  std::vector<size_t> vector2;
  bitmap1.extract(&vector2);

  EXPECT_EQ(vector1.size(), vector2.size());
  EXPECT_TRUE(std::equal(vector1.begin(), vector1.end(), vector2.begin()));
}

TEST(Bitmap, BitwiseXorCardinality) {
  srand((uint32_t)time(NULL));
  srand((uint32_t)rand());

  Bitmap bitmap1;
  Bitmap bitmap2;
  std::bitset<500000> stl_bitset1;
  std::bitset<500000> stl_bitset2;

  for (uint32_t i = 0; i < 800; ++i) {
    uint32_t val1 = rand() % 500000;
    uint32_t val2 = rand() % 500000;

    bitmap1.set(val1);
    stl_bitset1.set(val1);

    bitmap2.set(val2);
    stl_bitset2.set(val2);
  }
  for (uint32_t i = 0; i < 600; ++i) {
    uint32_t val1 = rand() % 500000;
    uint32_t val2 = rand() % 500000;

    bitmap1.flip(val1);
    stl_bitset1.flip(val1);

    bitmap2.flip(val2);
    stl_bitset2.flip(val2);
  }
  EXPECT_EQ((stl_bitset1 ^ stl_bitset2).count(),
            Bitmap::BitwiseXorCardinality(bitmap1, bitmap2));

  EXPECT_EQ(Bitmap::BitwiseAndnotCardinality(bitmap1, bitmap2) +
                Bitmap::BitwiseAndnotCardinality(bitmap2, bitmap1),
            Bitmap::BitwiseXorCardinality(bitmap1, bitmap2));
  EXPECT_EQ(Bitmap::BitwiseXorCardinality(bitmap2, bitmap1),
            Bitmap::BitwiseXorCardinality(bitmap1, bitmap2));
}

TEST(Bitmap, BitwiseOrCardinality) {
  srand((uint32_t)time(NULL));
  srand((uint32_t)rand());

  Bitmap bitmap1;
  Bitmap bitmap2;
  std::bitset<500000> stl_bitset1;
  std::bitset<500000> stl_bitset2;

  for (uint32_t i = 0; i < 800; ++i) {
    uint32_t val1 = rand() % 500000;
    uint32_t val2 = rand() % 500000;

    bitmap1.set(val1);
    stl_bitset1.set(val1);

    bitmap2.set(val2);
    stl_bitset2.set(val2);
  }
  for (uint32_t i = 0; i < 600; ++i) {
    uint32_t val1 = rand() % 500000;
    uint32_t val2 = rand() % 500000;

    bitmap1.flip(val1);
    stl_bitset1.flip(val1);

    bitmap2.flip(val2);
    stl_bitset2.flip(val2);
  }
  EXPECT_EQ((stl_bitset1 | stl_bitset2).count(),
            Bitmap::BitwiseOrCardinality(bitmap1, bitmap2));
  EXPECT_EQ(Bitmap::BitwiseOrCardinality(bitmap2, bitmap1),
            Bitmap::BitwiseOrCardinality(bitmap1, bitmap2));
}

TEST(Bitmap, BitwiseAndCardinality) {
  srand((uint32_t)time(NULL));
  srand((uint32_t)rand());

  Bitmap bitmap1;
  Bitmap bitmap2;
  std::bitset<500000> stl_bitset1;
  std::bitset<500000> stl_bitset2;

  for (uint32_t i = 0; i < 800; ++i) {
    uint32_t val1 = rand() % 500000;
    uint32_t val2 = rand() % 500000;

    bitmap1.set(val1);
    stl_bitset1.set(val1);

    bitmap2.set(val2);
    stl_bitset2.set(val2);
  }
  for (uint32_t i = 0; i < 600; ++i) {
    uint32_t val1 = rand() % 500000;
    uint32_t val2 = rand() % 500000;

    bitmap1.flip(val1);
    stl_bitset1.flip(val1);

    bitmap2.flip(val2);
    stl_bitset2.flip(val2);
  }
  EXPECT_EQ((stl_bitset1 & stl_bitset2).count(),
            Bitmap::BitwiseAndCardinality(bitmap1, bitmap2));
  EXPECT_EQ(Bitmap::BitwiseAndCardinality(bitmap2, bitmap1),
            Bitmap::BitwiseAndCardinality(bitmap1, bitmap2));
}

TEST(Bitmap, BitwiseAndnotCardinality) {
  srand((uint32_t)time(NULL));
  srand((uint32_t)rand());

  Bitmap bitmap1;
  Bitmap bitmap2;
  std::bitset<500000> stl_bitset1;
  std::bitset<500000> stl_bitset2;

  for (uint32_t i = 0; i < 800; ++i) {
    uint32_t val1 = rand() % 500000;
    uint32_t val2 = rand() % 500000;

    bitmap1.set(val1);
    stl_bitset1.set(val1);

    bitmap2.set(val2);
    stl_bitset2.set(val2);
  }
  for (uint32_t i = 0; i < 600; ++i) {
    uint32_t val1 = rand() % 500000;
    uint32_t val2 = rand() % 500000;

    bitmap1.flip(val1);
    stl_bitset1.flip(val1);

    bitmap2.flip(val2);
    stl_bitset2.flip(val2);
  }
  EXPECT_EQ((stl_bitset1 & ~stl_bitset2).count(),
            Bitmap::BitwiseAndnotCardinality(bitmap1, bitmap2));

  EXPECT_EQ((stl_bitset2 & ~stl_bitset1).count(),
            Bitmap::BitwiseAndnotCardinality(bitmap2, bitmap1));
}

TEST(Bitmap, Benchmark) {
  srand((uint32_t)time(NULL));
  srand((uint32_t)rand());

  Bitmap bitmap1, bitmap2;

  for (uint32_t i = 0; i < 2000; ++i) {
    uint32_t val1 = rand() % 200000000u;
    uint32_t val2 = rand() % 200000000u;

    bitmap1.set(val1);
    bitmap2.set(val2);
  }
  for (uint32_t i = 0; i < 1000; ++i) {
    uint32_t val1 = rand() % 200000000u;
    uint32_t val2 = rand() % 200000000u;

    bitmap1.flip(val1);
    bitmap2.flip(val2);
  }
  for (uint32_t i = 0; i < 500; ++i) {
    uint32_t val1 = rand() % 200000000u;
    uint32_t val2 = rand() % 200000000u;

    bitmap1.reset(val1);
    bitmap2.reset(val2);
  }

  {
    uint64_t t1 = Monotime::MicroSeconds();
    uint64_t sum = 0;
    for (uint32_t i = 0; i < 3; ++i) {
      sum += Bitmap::BitwiseAndCardinality(bitmap1, bitmap2);
    }
    std::cout << INTRINSICS_SET
              << " BitwiseAndCardinality: " << Monotime::MicroSeconds() - t1
              << " us, sum: " << sum << std::endl;
  }

  {
    uint64_t t1 = Monotime::MicroSeconds();
    uint64_t sum = 0;
    for (uint32_t i = 0; i < 3; ++i) {
      sum += Bitmap::BitwiseAndnotCardinality(bitmap1, bitmap2);
    }
    std::cout << INTRINSICS_SET
              << " BitwiseAndnotCardinality: " << Monotime::MicroSeconds() - t1
              << " us, sum: " << sum << std::endl;
  }

  {
    uint64_t t1 = Monotime::MicroSeconds();
    uint64_t sum = 0;
    for (uint32_t i = 0; i < 3; ++i) {
      sum += Bitmap::BitwiseXorCardinality(bitmap1, bitmap2);
    }
    std::cout << INTRINSICS_SET
              << " BitwiseXorCardinality: " << Monotime::MicroSeconds() - t1
              << " us, sum: " << sum << std::endl;
  }

  {
    uint64_t t1 = Monotime::MicroSeconds();
    uint64_t sum = 0;
    for (uint32_t i = 0; i < 3; ++i) {
      sum += Bitmap::BitwiseOrCardinality(bitmap1, bitmap2);
    }
    std::cout << INTRINSICS_SET
              << " BitwiseOrCardinality: " << Monotime::MicroSeconds() - t1
              << " us, sum: " << sum << std::endl;
  }

  {
    Bitmap bitmap3;
    bitmap3 = bitmap1;

    uint64_t t1 = Monotime::MicroSeconds();
    for (uint32_t i = 0; i < 3; ++i) {
      bitmap1.bitwise_and(bitmap2);
    }
    std::cout << INTRINSICS_SET << " And: " << Monotime::MicroSeconds() - t1
              << " us" << std::endl;
  }

  {
    Bitmap bitmap3;
    bitmap3 = bitmap1;

    uint64_t t1 = Monotime::MicroSeconds();
    for (uint32_t i = 0; i < 3; ++i) {
      bitmap1.bitwise_andnot(bitmap2);
    }
    std::cout << INTRINSICS_SET << " Andnot: " << Monotime::MicroSeconds() - t1
              << " us" << std::endl;
  }

  {
    Bitmap bitmap3;
    bitmap3 = bitmap1;

    uint64_t t1 = Monotime::MicroSeconds();
    for (uint32_t i = 0; i < 3; ++i) {
      bitmap1.bitwise_or(bitmap2);
    }
    std::cout << INTRINSICS_SET << " Or: " << Monotime::MicroSeconds() - t1
              << " us" << std::endl;
  }

  {
    Bitmap bitmap3;
    bitmap3 = bitmap1;

    uint64_t t1 = Monotime::MicroSeconds();
    for (uint32_t i = 0; i < 3; ++i) {
      bitmap1.bitwise_xor(bitmap2);
    }
    std::cout << INTRINSICS_SET << " Xor: " << Monotime::MicroSeconds() - t1
              << " us" << std::endl;
  }
}
