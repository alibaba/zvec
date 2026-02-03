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
#include <ailego/container/vector_array.h>
#include <gtest/gtest.h>
#include <zvec/ailego/utility/time_helper.h>

using namespace zvec;

TEST(NumericalVectorArray, General) {
  ailego::NumericalVectorArray<float> arr;
  ailego::NumericalVectorArray<float> &const_arr = arr;
  EXPECT_TRUE(arr.empty());
  EXPECT_EQ(0u, arr.dimension());
  EXPECT_EQ(0u, arr.count());
  EXPECT_EQ(0u, arr.bytes());
  EXPECT_NE(nullptr, arr.data());
  EXPECT_NE(nullptr, const_arr.data());
  arr.shrink_to_fit();
  arr.clear();
  EXPECT_EQ(0u, arr.dimension());
  EXPECT_EQ(0u, arr.count());

  try {
    arr.at(0);
  } catch (const std::out_of_range &oor) {
    std::cerr << "Out of Range error: " << oor.what() << '\n';
  }
  try {
    const_arr.at(0);
  } catch (const std::out_of_range &oor) {
    std::cerr << "Out of Range error: " << oor.what() << '\n';
  }

  ailego::NumericalVector<float> vec1 = {10.0f, 11.0f, 12.0f, 13.0f, 14.0f,
                                         15.0f, 16.0f, 17.0f, 18.0f, 19.0f};
  ailego::NumericalVector<float> vec2 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                                         6.0f, 7.0f, 8.0f, 9.0f, 0.0f};
  arr.reset(10);
  arr.append(vec1);
  arr.append(vec2);
  arr.append(vec1);
  EXPECT_EQ(3u, arr.count());
  EXPECT_EQ(10u, arr.dimension());

  arr.reserve(10);
  EXPECT_EQ(vec1, std::string(reinterpret_cast<const char *>(arr.at(0)),
                              arr.dimension() * sizeof(float)));
  EXPECT_EQ(vec1, std::string(reinterpret_cast<const char *>(arr[2]),
                              arr.dimension() * sizeof(float)));
  arr.replace(2, vec2);
  EXPECT_EQ(vec2, std::string(reinterpret_cast<const char *>(const_arr.at(1)),
                              arr.dimension() * sizeof(float)));
  EXPECT_EQ(vec2, std::string(reinterpret_cast<const char *>(const_arr[2]),
                              arr.dimension() * sizeof(float)));

  arr.clear();
  EXPECT_EQ(10u, arr.dimension());
  EXPECT_EQ(0u, arr.count());

  arr.reset(2);
  arr.append(vec1.data(), 2, 5);
  arr.append(vec2.data(), 2, 5);
  EXPECT_EQ(2u, arr.dimension());
  EXPECT_EQ(10u, arr.count());

  ailego::NumericalVectorArray<float> arr1 = std::move(arr);
  EXPECT_TRUE(arr.empty());
  EXPECT_EQ(2u, arr.dimension());
  EXPECT_EQ(0u, arr.count());
  EXPECT_EQ(2u, arr1.dimension());
  EXPECT_EQ(10u, arr1.count());

  arr1.resize(8u);
  EXPECT_EQ(8u, arr1.count());

  arr1.resize(15u);
  EXPECT_EQ(15u, arr1.count());
}

TEST(NumericalVectorArray, Batch) {
  const size_t DIMENSION = 20;
  const size_t COUNT = 20000u;

  ailego::NumericalVectorArray<float> arr(DIMENSION);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(0.0, 1.0);
  std::string buffer;

  for (size_t i = 0; i < COUNT; ++i) {
    ailego::FixedVector<float, DIMENSION> vec;
    for (size_t j = 0; j < DIMENSION; ++j) {
      vec[j] = dist(gen);
    }
    arr.append(vec.data(), vec.size());
    buffer.append((const char *)vec.data(), sizeof(vec));
  }
  EXPECT_EQ(COUNT, arr.count());
  EXPECT_EQ(buffer, std::string((const char *)arr.data(), arr.bytes()));
}

TEST(BinaryVectorArray, General) {
  ailego::BinaryVectorArray<uint64_t> arr64;
  ailego::BinaryVectorArray<uint64_t> &const_arr64 = arr64;
  EXPECT_TRUE(arr64.empty());
  EXPECT_EQ(0u, arr64.dimension());
  EXPECT_EQ(0u, arr64.count());
  EXPECT_EQ(0u, arr64.bytes());
  EXPECT_NE(nullptr, arr64.data());
  EXPECT_NE(nullptr, const_arr64.data());
  arr64.shrink_to_fit();
  arr64.clear();
  EXPECT_EQ(0u, arr64.dimension());
  EXPECT_EQ(0u, arr64.count());

  try {
    arr64.at(0);
  } catch (const std::out_of_range &oor) {
    std::cerr << "Out of Range error: " << oor.what() << '\n';
  }
  try {
    const_arr64.at(0);
  } catch (const std::out_of_range &oor) {
    std::cerr << "Out of Range error: " << oor.what() << '\n';
  }

  ailego::BinaryVector<uint64_t> vec1 = {true, false, true,  true, false,
                                         true, false, false, true, false};
  ailego::BinaryVector<uint64_t> vec2 = {true,  true,  true,  true,
                                         false, false, false, true,
                                         false, false, true,  false};
  EXPECT_EQ(64u, vec1.dimension());
  EXPECT_EQ(64u, vec2.dimension());
  arr64.reset(10);
  arr64.append(vec1);
  arr64.append(vec2);
  arr64.append(vec1);
  EXPECT_EQ(3u, arr64.count());
  EXPECT_EQ(64u, arr64.dimension());
  EXPECT_EQ(0u, arr64.bytes() % sizeof(uint64_t));

  arr64.reserve(10);
  EXPECT_EQ(vec1, std::string(reinterpret_cast<const char *>(arr64.at(0)),
                              arr64.dimension() >> 3));
  EXPECT_EQ(vec1, std::string(reinterpret_cast<const char *>(arr64[2]),
                              arr64.dimension() >> 3));
  arr64.replace(2, vec2);
  EXPECT_EQ(vec2, std::string(reinterpret_cast<const char *>(const_arr64.at(1)),
                              arr64.dimension() >> 3));
  EXPECT_EQ(vec2, std::string(reinterpret_cast<const char *>(const_arr64[2]),
                              arr64.dimension() >> 3));

  arr64.clear();
  EXPECT_EQ(64u, arr64.dimension());
  EXPECT_EQ(0u, arr64.count());

  ailego::BinaryVectorArray<uint32_t> arr32(1);
  EXPECT_EQ(32u, arr32.dimension());
  arr32.append((const uint32_t *)vec1.data(), 32, 2);
  arr32.append((const uint32_t *)vec2.data(), 32, 2);
  EXPECT_EQ(32u, arr32.dimension());
  EXPECT_EQ(4u, arr32.count());
  EXPECT_EQ(0u, arr64.bytes() % sizeof(uint32_t));

  ailego::BinaryVectorArray<uint32_t> arr1 = std::move(arr32);
  EXPECT_TRUE(arr32.empty());
  EXPECT_EQ(32u, arr32.dimension());
  EXPECT_EQ(0u, arr32.count());
  EXPECT_EQ(32u, arr1.dimension());
  EXPECT_EQ(4u, arr1.count());

  arr1.resize(8u);
  EXPECT_EQ(8u, arr1.count());

  arr1.resize(1u);
  EXPECT_EQ(1u, arr1.count());
}
