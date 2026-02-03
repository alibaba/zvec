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
#include <ailego/container/bitmap.h>
#include <ailego/internal/cpu_features.h>
#include <ailego/math/normalizer.h>
#include <gtest/gtest.h>
#include <zvec/ailego/container/vector.h>
#include <zvec/ailego/utility/time_helper.h>

using namespace zvec::ailego;

TEST(Normalizer, FP32_General) {
  std::mt19937 gen((std::random_device())());
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  for (size_t i = 0; i < 100; ++i) {
    std::vector<float> vec1;
    std::vector<float> vec2;
    for (size_t j = 0; j < 111; ++j) {
      float val = dist(gen);
      vec1.push_back(val);
      vec2.push_back(val);
    }

    Normalizer<float>::Compute(vec1.data(), vec1.size(), 1.1f);
    for (size_t j = 0; j < vec1.size(); ++j) {
      EXPECT_FLOAT_EQ(vec1[j] * 1.1f, vec2[j]);
    }

    float l1 = 0.0f, l2 = 0.0f;
    Normalizer<float>::L1(vec1.data(), vec1.size(), &l1);
    Normalizer<float>::L2(vec2.data(), vec2.size(), &l2);
  }
}

TEST(Normalizer, FP16_General) {
  std::mt19937 gen((std::random_device())());
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  for (size_t i = 0; i < 100; ++i) {
    std::vector<Float16> vec1;
    std::vector<Float16> vec2;
    for (size_t j = 0; j < 111; ++j) {
      float val = dist(gen);
      vec1.push_back(val);
      vec2.push_back(val);
    }

    Normalizer<Float16>::Compute(vec1.data(), vec1.size(), 1.0f);
    for (size_t j = 0; j < vec1.size(); ++j) {
      EXPECT_FLOAT_EQ(vec1[j], vec2[j]);
    }

    float l1 = 0.0f, l2 = 0.0f;
    Normalizer<Float16>::L1(vec1.data(), vec1.size(), &l1);
    Normalizer<Float16>::L2(vec2.data(), vec2.size(), &l2);
  }
}

TEST(Normalizer, FP32_Zero) {
  std::mt19937 gen((std::random_device())());
  std::uniform_int_distribution<uint32_t> dist(1, 128);
  const uint32_t dimension = dist(gen);

  std::vector<float> vec1(dimension, 0.0f);
  std::vector<float> vec2(dimension, 0.0f);

  float norm;
  Normalizer<float>::L1(vec1.data(), vec1.size(), &norm);
  Normalizer<float>::L2(vec2.data(), vec2.size(), &norm);
  for (auto v : vec1) {
    EXPECT_FALSE(std::isnan(v));
  }
  for (auto v : vec2) {
    EXPECT_FALSE(std::isnan(v));
  }
}

TEST(Normalizer, FP16_Zero) {
  std::mt19937 gen((std::random_device())());
  std::uniform_int_distribution<uint32_t> dist(1, 128);
  const uint32_t dimension = dist(gen);

  std::vector<Float16> vec1(dimension, 0.0f);
  std::vector<Float16> vec2(dimension, 0.0f);

  float norm;
  Normalizer<Float16>::L2(vec1.data(), vec1.size(), &norm);
  Normalizer<Float16>::L2(vec2.data(), vec2.size(), &norm);
  for (auto v : vec1) {
    EXPECT_FALSE(std::isnan(v));
  }
  for (auto v : vec2) {
    EXPECT_FALSE(std::isnan(v));
  }
}
