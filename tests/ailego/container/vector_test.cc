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
#include <gtest/gtest.h>
#include <zvec/ailego/container/vector.h>
#include <zvec/ailego/utility/time_helper.h>

using namespace zvec;

TEST(FixedVector, General) {
  int aaa[512];
  ailego::FixedVector<int, 512> *v = ailego::FixedVector<int, 512>::Cast(aaa);
  ASSERT_EQ(aaa, v->data());
  EXPECT_EQ(512u, v->size());

  ailego::FixedVector<int, 128> bbb{11, 22, 33};
  EXPECT_EQ(11, bbb[0]);
  EXPECT_EQ(22, bbb[1]);
  EXPECT_EQ(33, bbb[2]);
  EXPECT_EQ(128u, bbb.size());

  bbb = {55, 66, 77};
  EXPECT_EQ(55, bbb[0]);
  EXPECT_EQ(66, bbb[1]);
  EXPECT_EQ(77, bbb[2]);
  EXPECT_EQ(128u, bbb.size());
}

TEST(NumericalVector, General) {
  ailego::NumericalVector<float> vec(10);
  for (size_t i = 0; i < vec.size(); ++i) {
    vec[i] = (float)i;
  }

  {
    size_t index = 0;
    for (auto v : vec) {
      EXPECT_FLOAT_EQ(v, (float)(index++));
    }
  }

  vec.reserve(20);
  EXPECT_EQ(10u, vec.size());

  vec.append(
      {10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f});
  {
    size_t index = 0;
    for (auto v : vec) {
      EXPECT_FLOAT_EQ(v, (float)(index++));
    }
  }
  EXPECT_EQ(20u, vec.size());

  EXPECT_FALSE(vec.empty());
  vec.clear();
  EXPECT_EQ(0u, vec.size());
  EXPECT_TRUE(vec.empty());

  ailego::NumericalVector<float> vec1(10, 1.0f);
  for (auto v : vec1) {
    EXPECT_FLOAT_EQ(1.0f, v);
  }

  vec.swap(vec1);
  for (auto v : vec) {
    EXPECT_FLOAT_EQ(1.0f, v);
  }
  {
    size_t index = 0;
    for (auto v : vec1) {
      EXPECT_FLOAT_EQ(v, (float)(index++));
    }
  }

  ailego::NumericalVector<float> vec2(
      {10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f});
  {
    size_t index = 10;
    for (auto v : vec1) {
      EXPECT_FLOAT_EQ(v, (float)(index++));
    }
  }
}

TEST(NumericalVector, Assign) {
  ailego::NumericalVector<float> vec1;
  EXPECT_TRUE(vec1.data());
  EXPECT_EQ(0u, vec1.size());

  ailego::NumericalVector<size_t> vec2(222u);
  EXPECT_TRUE(!!vec2.data());
  EXPECT_EQ(222u, vec2.size());

  ailego::NumericalVector<size_t> vec3(vec2);
  EXPECT_TRUE(!!vec2.data());
  EXPECT_EQ(222u, vec2.size());
  EXPECT_TRUE(!!vec3.data());
  EXPECT_EQ(222u, vec3.size());

  ailego::NumericalVector<size_t> vec4;
  vec4 = vec3;
  EXPECT_TRUE(!!vec3.data());
  EXPECT_EQ(222u, vec3.size());
  EXPECT_TRUE(!!vec4.data());
  EXPECT_EQ(222u, vec4.size());

  ailego::NumericalVector<size_t> vec5;
  vec5 = std::move(vec4);
  EXPECT_TRUE(vec4.data());
  EXPECT_EQ(0u, vec4.size());
  EXPECT_TRUE(!!vec5.data());
  EXPECT_EQ(222u, vec5.size());

  ailego::NumericalVector<size_t> vec6(std::move(vec5));
  EXPECT_TRUE(vec5.data());
  EXPECT_EQ(0u, vec5.size());
  EXPECT_TRUE(!!vec6.data());
  EXPECT_EQ(222u, vec6.size());

  ailego::NumericalVector<int> vec7 = {1, 2, 3, 4, 5, 6, 7};
  EXPECT_TRUE(!!vec7.data());
  EXPECT_EQ(7u, vec7.size());
}

TEST(BinaryVector, General) {
  ailego::BinaryVector<char> a8({true, false, true, false, true, true});
  EXPECT_EQ(8u, a8.size());
  EXPECT_FALSE(a8.empty());
  EXPECT_FALSE(a8.at(1));
  EXPECT_TRUE(a8[0]);

  for (auto val : a8) {
    std::cout << val << ' ';
  }
  std::cout << std::endl;

  ailego::BinaryVector<int16_t> a16({true, false, true, false, true, true});
  EXPECT_EQ(16u, a16.size());
  EXPECT_FALSE(a16.at(1));
  EXPECT_TRUE(a16[0]);

  for (auto val : a16) {
    std::cout << val << ' ';
  }
  std::cout << std::endl;

  ailego::BinaryVector<uint32_t> a32({true, false, true, false, true, true});
  EXPECT_EQ(32u, a32.size());
  EXPECT_FALSE(a32.at(1));
  EXPECT_TRUE(a32[2]);

  for (auto val : a32) {
    std::cout << val << ' ';
  }
  std::cout << std::endl;

  ailego::BinaryVector<int64_t> a64({true, false, true, false, true, true});
  EXPECT_EQ(64u, a64.size());
  EXPECT_FALSE(a64.at(1));
  EXPECT_TRUE(a64[2]);

  for (auto val : a64) {
    std::cout << val << ' ';
  }
  std::cout << std::endl;

  ailego::BinaryVector<uint64_t> aaa(21, true);
  EXPECT_EQ(64u, aaa.size());
  for (auto val : aaa) {
    EXPECT_TRUE(val);
  }
  for (size_t i = 0; i < aaa.size(); ++i) {
    EXPECT_TRUE(aaa[i]);
    aaa.reset(i);
    EXPECT_FALSE(aaa.at(i));
  }

  ailego::BinaryVector<int32_t> bbb(100);
  EXPECT_EQ(128u, bbb.size());
  for (auto val : bbb) {
    EXPECT_FALSE(val);
  }
  for (size_t i = 0; i < bbb.size(); ++i) {
    EXPECT_FALSE(bbb[i]);
    bbb.set(i);
    EXPECT_TRUE(bbb.at(i));
  }

  ailego::BinaryVector<bool> ccc(100);
  EXPECT_EQ(
      (100u + sizeof(bool) * 8 - 1) / (sizeof(bool) * 8) * (sizeof(bool) * 8),
      ccc.size());
  for (auto val : ccc) {
    EXPECT_FALSE(val);
  }
  for (size_t i = 0; i < ccc.size(); ++i) {
    EXPECT_FALSE(ccc[i]);
    ccc.flip(i);
    EXPECT_TRUE(ccc.at(i));
  }

  ailego::BinaryVector<int32_t> ddd;
  EXPECT_TRUE(ddd.empty());
  EXPECT_FALSE(bbb.empty());
  ddd = std::move(bbb);
  EXPECT_FALSE(ddd.empty());
  EXPECT_TRUE(bbb.empty());

  ailego::BinaryVector<int32_t> eee;
  EXPECT_TRUE(eee.empty());
  eee = ddd;
  EXPECT_FALSE(ddd.empty());
  EXPECT_FALSE(eee.empty());
  ddd.clear();
  bbb.clear();
  EXPECT_TRUE(ddd.empty());
  EXPECT_TRUE(bbb.empty());

  ailego::BinaryVector<int32_t> fff;
  for (auto val : fff) {
    (void)val;
    EXPECT_TRUE(0);
  }

  std::string str;
  ailego::BinaryVector<int32_t> ggg(str);
  ailego::BinaryVector<char> hhh(str);

  str.resize(128);
  ailego::BinaryVector<char> iii(str);
  ailego::BinaryVector<int64_t> jjj(std::move(str));

  jjj.assign({true, true, true, false, true, true, false, true, true, false,
              true, true});
  EXPECT_NE(0u, jjj.capacity());
  EXPECT_TRUE(jjj.front());
  EXPECT_FALSE(jjj.back());

  ailego::BinaryVector<int64_t> mmm;
  EXPECT_TRUE(mmm.data());
  ailego::BinaryVector<int64_t> &nnn = mmm;
  EXPECT_TRUE(nnn.data());

  ailego::BinaryVector<int64_t> ooo;
  ooo.reserve(1111);
  EXPECT_NE(0u, ooo.capacity());
  EXPECT_EQ(0u, ooo.size());
  EXPECT_TRUE(ooo.empty());
  ooo.assign({true});
  EXPECT_EQ(64u, ooo.size());

  ooo.swap(mmm);
  EXPECT_EQ(0u, ooo.size());
}

TEST(BinaryVector, Iterator) {
  std::mt19937 gen((std::random_device())());
  auto dist = std::uniform_int_distribution<size_t>(1, 129);
  size_t dimension = dist(gen) * 32;

  ailego::BinaryVector<uint32_t> bt(dimension);
  std::vector<bool> vec(dimension);

  for (size_t i = 0; i != vec.size(); ++i) {
    bool val = (dist(gen) % 7 == 0);
    vec[i] = val;
    if (val) {
      bt.set(i);
    }
  }

  size_t index = 0;
  for (auto iter = bt.begin(); iter != bt.end(); ++iter) {
    EXPECT_EQ(vec[index], *iter);
    ++index;
  }
}

TEST(BinaryVector, LittleEndian) {
  ailego::BinaryVector<uint8_t> bs8(128 * 4);
  ailego::BinaryVector<uint16_t> bs16(128 * 4);
  ailego::BinaryVector<uint32_t> bs32(128 * 4);
  ailego::BinaryVector<uint64_t> bs64(128 * 4);

  std::mt19937 gen((std::random_device())());
  auto dist = std::uniform_int_distribution<size_t>(0, 128 * 4);

  for (int i = 0; i < 18; ++i) {
    size_t val = dist(gen);
    bs8.set(val);
    bs16.set(val);
    bs32.set(val);
    bs64.set(val);
    EXPECT_TRUE(bs8.at(val));
    EXPECT_TRUE(bs16.at(val));
    EXPECT_TRUE(bs32.at(val));
    EXPECT_TRUE(bs64.at(val));
  }

  EXPECT_TRUE(memcmp(bs8.data(), bs16.data(), bs8.bytes()) == 0);
  EXPECT_TRUE(memcmp(bs8.data(), bs32.data(), bs8.bytes()) == 0);
  EXPECT_TRUE(memcmp(bs8.data(), bs64.data(), bs8.bytes()) == 0);
}

TEST(NibbleVector, General) {
  ailego::NibbleVector<int> nv1(
      {-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7});

  EXPECT_FALSE(nv1.empty());
  EXPECT_EQ(16, nv1.size());
  EXPECT_EQ(16, nv1.dimension());
  EXPECT_EQ(8, nv1.bytes());
  for (int i = -8; i != 8; ++i) {
    EXPECT_EQ(i, nv1.at(i + 8));
  }

  ailego::NibbleVector<uint32_t> nv2(31, 5);
  for (size_t i = 0; i != nv2.size(); ++i) {
    EXPECT_EQ(5u, nv2.at(i));
  }

  ailego::NibbleVector<int32_t> nv3(56);
  nv3.assign({-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7});
  EXPECT_EQ(16u, nv3.size());

  ailego::NibbleVector<uint32_t> nv4(25);
  nv4.assign(88, 6);
  for (size_t i = 0; i != nv4.size(); ++i) {
    EXPECT_EQ(6u, nv4.at(i));
  }
}

TEST(NibbleVector, UnsignedIterator) {
  std::mt19937 gen((std::random_device())());
  auto dist = std::uniform_int_distribution<uint32_t>(0, 15);
  size_t dimension = ((std::uniform_int_distribution<size_t>(1, 63))(gen)) * 2;

  ailego::NibbleVector<uint32_t> nv;
  std::vector<uint32_t> vec;

  for (size_t i = 0; i != dimension; i += 2) {
    uint32_t lo = dist(gen);
    uint32_t hi = dist(gen);
    vec.push_back(lo);
    vec.push_back(hi);
    nv.append(lo, hi);
  }

  size_t index = 0;
  for (auto iter = nv.begin(); iter != nv.end(); ++iter) {
    EXPECT_EQ(vec[index], *iter);
    ++index;
  }
}

TEST(NibbleVector, SignedIterator) {
  std::mt19937 gen((std::random_device())());
  auto dist = std::uniform_int_distribution<int32_t>(-8, 7);
  size_t dimension = ((std::uniform_int_distribution<size_t>(1, 63))(gen)) * 2;

  ailego::NibbleVector<int32_t> nv;
  std::vector<int32_t> vec;
  EXPECT_TRUE(nv.empty());

  for (size_t i = 0; i != dimension; i += 2) {
    int32_t lo = dist(gen);
    int32_t hi = dist(gen);
    vec.push_back(lo);
    vec.push_back(hi);
    nv.append(lo, hi);
  }
  EXPECT_FALSE(nv.empty());
  EXPECT_EQ(vec.size(), nv.size());
  EXPECT_EQ(vec.size(), nv.dimension());
  EXPECT_EQ(vec.size() / 2, nv.bytes());

  size_t index = 0;
  for (auto iter = nv.begin(); iter != nv.end(); ++iter) {
    EXPECT_EQ(vec[index], *iter);
    ++index;
  }

  // Test again
  for (size_t i = 0; i != dimension; i += 2) {
    int32_t lo = dist(gen);
    int32_t hi = dist(gen);
    vec[i + 0] = lo;
    vec[i + 1] = hi;
    nv.set(i + 0, lo);
    nv.set(i + 1, hi);
  }
  index = 0;
  for (auto iter = nv.begin(); iter != nv.end(); ++iter) {
    EXPECT_EQ(vec[index], *iter);
    ++index;
  }
}
