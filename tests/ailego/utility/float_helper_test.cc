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
#include <ailego/math/norm_matrix.h>
#include <gtest/gtest.h>
#include <zvec/ailego/utility/float_helper.h>

using namespace zvec;

TEST(FloatHelper, General) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(0.0f, 0.9f);
  std::uniform_int_distribution<int> dist2(1, 250);

  for (int i = 0; i < 1000; ++i) {
    float fp32 = dist(gen);
    float fp16 = ailego::FloatHelper::ToFP32(
        ailego::FloatHelper::ToFP16(fp32, 1.0f), 1.0f);
    EXPECT_GT(0.00025, std::abs(fp32 - fp16));
  }

  for (int i = 0; i < 1000; ++i) {
    std::vector<float> vec1_fp32, vec2_fp32;
    std::vector<ailego::Float16> vec1_fp16, vec2_fp16;
    int count = dist2(gen);

    vec1_fp32.resize(count);
    vec2_fp32.resize(count);
    vec1_fp16.resize(count);
    vec2_fp16.resize(count);
    for (size_t j = 0; j < vec1_fp32.size(); ++j) {
      vec1_fp32[j] = dist(gen);
    }
    float norm1;
    ailego::Norm2Matrix<float, 1>::Compute(vec1_fp32.data(), vec1_fp32.size(),
                                           &norm1);
    EXPECT_NE(1.0f, norm1);

    // Convert to FP16
    ailego::FloatHelper::ToFP16(vec1_fp32.data(), vec1_fp32.size(),
                                (uint16_t *)vec1_fp16.data());
    ailego::FloatHelper::ToFP16(vec1_fp32.data(), vec1_fp32.size(), norm1,
                                (uint16_t *)vec2_fp16.data());
    for (size_t j = 0; j < vec1_fp32.size(); ++j) {
      EXPECT_GT(0.00025, std::abs(vec1_fp32[j] - vec1_fp16[j]));
      // EXPECT_FLOAT_EQ(vec1_fp32[j], vec1_fp16[j]);
    }

    float norm2;
    ailego::Norm2Matrix<ailego::Float16, 1>::Compute(vec1_fp16.data(),
                                                     vec1_fp16.size(), &norm2);
    EXPECT_NE(1.0f, norm2);

    // Convert to FP32
    ailego::FloatHelper::ToFP32((const uint16_t *)vec1_fp16.data(),
                                vec1_fp16.size(), vec1_fp32.data());
    ailego::FloatHelper::ToFP32((const uint16_t *)vec1_fp16.data(),
                                vec1_fp16.size(), norm2, vec2_fp32.data());
    for (size_t j = 0; j < vec1_fp32.size(); ++j) {
      EXPECT_GT(0.00025, std::abs(vec1_fp32[j] - vec1_fp16[j]));
      // EXPECT_FLOAT_EQ(vec1_fp32[j], vec1_fp16[j]);
    }

    ailego::Norm2Matrix<float, 1>::Compute(vec2_fp32.data(), vec2_fp32.size(),
                                           &norm1);
    ailego::Norm2Matrix<ailego::Float16, 1>::Compute(vec2_fp16.data(),
                                                     vec2_fp16.size(), &norm2);
    // EXPECT_FLOAT_EQ(norm1, norm2);
    // EXPECT_FLOAT_EQ(1.0f, norm1);
    EXPECT_GT(0.001, std::abs(1.0f - norm1));
    // EXPECT_FLOAT_EQ(1.0f, norm2);
    EXPECT_GT(0.001, std::abs(1.0f - norm2));
  }
}

TEST(Float16, General) {
  ailego::Float16 a1;
  EXPECT_FLOAT_EQ(0.0f, a1);

  ailego::Float16 a2 = 0.33f;
  EXPECT_TRUE(0.0f < a2);
  EXPECT_TRUE(0.0f <= a2);
  EXPECT_TRUE(0.5f > a2);
  EXPECT_TRUE(0.5f >= a2);
  EXPECT_TRUE(0.0 < a2);
  EXPECT_TRUE(0.0 <= a2);
  EXPECT_TRUE(0.5 > a2);
  EXPECT_TRUE(0.5 >= a2);
  EXPECT_TRUE((float)a2 != 0.0);
  EXPECT_FALSE((float)a2 == 0.0);

  ailego::Float16 a3 = 0.55;
  EXPECT_TRUE((double)a3 != 0.0);
  EXPECT_FALSE((double)a3 == 0.0);

  EXPECT_TRUE(a1 < a2);
  EXPECT_TRUE(a2 <= a3);
  EXPECT_TRUE(a2 > a1);
  EXPECT_TRUE(a3 >= a1);

  ailego::Float16 a4 = a2 + a3;
  ailego::Float16 a5 = a2 - a3;
  ailego::Float16 a6 = a2 * a3;
  ailego::Float16 a7 = a2 / a3;

  a4 *= 1.0;
  a5 /= 1.0;
  a6 -= 0.0;
  a7 += 0.0;

  EXPECT_TRUE(0.0f != a4);
  EXPECT_TRUE(0.0f != a5);
  EXPECT_TRUE(0.0f != a6);
  EXPECT_TRUE(0.0f != a7);

  ailego::Float16 one = 1.0;
  uint16_t *one_encoded = (uint16_t *)(&one);
  printf("One: %f, 0x%x\n", (float)one, *one_encoded);
}
