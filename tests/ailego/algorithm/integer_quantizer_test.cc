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

#include <stdlib.h>
#include <random>
#include <ailego/algorithm/integer_quantizer.h>
#include <gtest/gtest.h>

namespace zvec::ailego {

TEST(IntegerQuantizer, INT8_Uniform_Distribution) {
  std::vector<size_t> tests = {1, 100, 1000, 10000, 100000};
  for (auto count : tests) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<float> data;

    std::uniform_real_distribution<float> dist(1.0, 2.0);
    float max = -std::numeric_limits<float>::max();
    float min = std::numeric_limits<float>::max();
    for (size_t i = 0; i < count; ++i) {
      auto v = dist(gen);
      max = std::max(max, v);
      min = std::min(min, v);
      data.emplace_back(v);
    }
    // data.emplace_back(10);  // deviation point
    EntropyInt8Quantizer quantizer;
    quantizer.set_max(max);
    quantizer.set_min(min);
    quantizer.feed(data.data(), data.size());

    ASSERT_TRUE(quantizer.train());

    std::vector<int8_t> qdata(data.size(), 0);
    quantizer.encode(data.data(), qdata.size(), qdata.data());

    std::vector<float> recover_data(data.size(), 0.0f);
    quantizer.decode(qdata.data(), qdata.size(), recover_data.data());

    float var = 0.0f;
    for (size_t i = 0; i < data.size(); ++i) {
      var += (data[i] - recover_data[i]) * (data[i] - recover_data[i]);
    }
    EXPECT_LT(var / count, 0.01);
  }
}

TEST(IntegerQuantizer, INT8_Normal_Distribution) {
  const size_t COUNT = 1000000u;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::vector<float> data;

  std::normal_distribution<float> dist(3, 1.5);
  float max = -std::numeric_limits<float>::max();
  float min = std::numeric_limits<float>::max();
  for (size_t i = 0; i < COUNT; ++i) {
    auto v = dist(gen);
    max = std::max(max, v);
    min = std::min(min, v);
    data.emplace_back(v);
  }
  // data.emplace_back(10);  // deviation point
  EntropyInt8Quantizer quantizer;
  bool non_bias = dist(gen) > 5;
  quantizer.set_non_bias(non_bias);

  quantizer.set_max(max);
  quantizer.set_min(min);
  quantizer.feed(data.data(), data.size());

  ASSERT_TRUE(quantizer.train());
  ASSERT_EQ(quantizer.bias() == 0.0f, non_bias);

  std::vector<int8_t> qdata(data.size(), 0);
  quantizer.encode(data.data(), qdata.size(), qdata.data());

  std::vector<float> recover_data(data.size(), 0.0f);
  quantizer.decode(qdata.data(), qdata.size(), recover_data.data());

  float var = 0.0f;
  for (size_t i = 0; i < data.size(); ++i) {
    var += (data[i] - recover_data[i]) * (data[i] - recover_data[i]);
    // printf("%f %f\n", data[i], recover_data[i]);
  }
#if 0
  printf("max=%f min=%f\n", *std::max_element(data.begin(), data.end()),
         *std::min_element(data.begin(), data.end()));
  printf("recover max=%f min=%f\n",
         *std::max_element(recover_data.begin(), recover_data.end()),
         *std::min_element(recover_data.begin(), recover_data.end()));
  printf("var=%f\n", var);
#endif
  EXPECT_LT(var / COUNT, 0.001);
}

TEST(IntegerQuantizer, INT8_Poisson_Distribution) {
  const size_t COUNT = 100000u;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::vector<float> data;

  std::poisson_distribution<int> dist(10000);
  float max = -std::numeric_limits<float>::min();
  float min = std::numeric_limits<float>::max();
  for (size_t i = 0; i < COUNT; ++i) {
    float v = (float)dist(gen);
    max = std::max(max, v);
    min = std::min(min, v);
    data.emplace_back(v);
  }
  // data.emplace_back(10);  // deviation point
  EntropyInt8Quantizer quantizer;

  quantizer.set_max(max);
  quantizer.set_min(min);
  quantizer.feed(data.data(), data.size());

  ASSERT_TRUE(quantizer.train());

  std::vector<int8_t> qdata(data.size(), 0);
  quantizer.encode(data.data(), qdata.size(), qdata.data());

  std::vector<float> recover_data(data.size(), 0.0f);
  quantizer.decode(qdata.data(), qdata.size(), recover_data.data());

  float var = 0.0f;
  for (size_t i = 0; i < data.size(); ++i) {
    var += (data[i] - recover_data[i]) * (data[i] - recover_data[i]);
  }
  EXPECT_LT(var / COUNT, 100);
}

TEST(IntegerQuantizer, INT4_Uniform_Distribution) {
  std::vector<size_t> tests = {2, 1000, 10000, 100000};
  for (auto count : tests) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<float> data;

    std::uniform_real_distribution<float> dist(1.0, 2.0);
    float max = -std::numeric_limits<float>::min();
    float min = std::numeric_limits<float>::max();
    for (size_t i = 0; i < count; ++i) {
      auto v = dist(gen);
      max = std::max(max, v);
      min = std::min(min, v);
      data.emplace_back(v);
    }
    // data.emplace_back(10);  // deviation point
    EntropyInt4Quantizer quantizer;
    quantizer.set_max(max);
    quantizer.set_min(min);
    quantizer.feed(data.data(), data.size());

    ASSERT_TRUE(quantizer.train());

    std::vector<uint8_t> qdata(data.size() / 2, 0);
    quantizer.encode(data.data(), data.size(), qdata.data());

    std::vector<float> recover_data(data.size(), 0.0f);
    quantizer.decode(qdata.data(), data.size(), recover_data.data());

    float var = 0.0f;
    for (size_t i = 0; i < data.size(); ++i) {
      var += (data[i] - recover_data[i]) * (data[i] - recover_data[i]);
      // printf("%f %f\n", data[i], recover_data[i]);
    }
#if 0
    printf("max=%f min=%f\n", *std::max_element(data.begin(), data.end()),
           *std::min_element(data.begin(), data.end()));
    printf("recover max=%f min=%f\n",
           *std::max_element(recover_data.begin(), recover_data.end()),
           *std::min_element(recover_data.begin(), recover_data.end()));
    printf("var=%f\n", var);
#endif
    EXPECT_LT(var / count, 0.1);
  }
}

TEST(IntegerQuantizer, INT4_Normal_Distribution) {
  const size_t COUNT = 10000u;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::vector<float> data;

  std::normal_distribution<float> avg(-1, 1);
  std::normal_distribution<float> dist(avg(gen), 5);
  float max = -std::numeric_limits<float>::max();
  float min = std::numeric_limits<float>::max();
  for (size_t i = 0; i < COUNT; ++i) {
    auto v = dist(gen);
    max = std::max(max, v);
    min = std::min(min, v);
    data.emplace_back(v);
  }
  // data.emplace_back(10);  // deviation point
  EntropyInt4Quantizer quantizer;
  bool non_bias = avg(gen) > 0;
  quantizer.set_non_bias(non_bias);

  quantizer.set_max(max);
  quantizer.set_min(min);
  quantizer.feed(data.data(), data.size());

  ASSERT_TRUE(quantizer.train());
  ASSERT_EQ(quantizer.bias() == 0.0f, non_bias);

  std::vector<uint8_t> qdata(data.size(), 0);
  quantizer.encode(data.data(), data.size(), qdata.data());

  std::vector<float> recover_data(data.size(), 0.0f);
  quantizer.decode(qdata.data(), data.size(), recover_data.data());

  float var = 0.0f;
  for (size_t i = 0; i < data.size(); ++i) {
    var += (data[i] - recover_data[i]) * (data[i] - recover_data[i]);
    // printf("%f %f\n", data[i], recover_data[i]);
  }
#if 0
  printf("max=%f min=%f\n", *std::max_element(data.begin(), data.end()),
         *std::min_element(data.begin(), data.end()));
  printf("recover max=%f min=%f\n",
         *std::max_element(recover_data.begin(), recover_data.end()),
         *std::min_element(recover_data.begin(), recover_data.end()));
  printf("var=%f\n", var);
#endif
  EXPECT_LT(var / COUNT, 1.0f);
}

TEST(IntegerQuantizer, INT4_Poisson_Distribution) {
  const size_t COUNT = 100000u;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::vector<float> data;

  std::poisson_distribution<int> dist(10000);
  float max = -std::numeric_limits<float>::min();
  float min = std::numeric_limits<float>::max();
  for (size_t i = 0; i < COUNT; ++i) {
    float v = (float)dist(gen);
    max = std::max(max, v);
    min = std::min(min, v);
    data.emplace_back(v);
  }
  // data.emplace_back(10);  // deviation point
  EntropyInt4Quantizer quantizer;

  quantizer.set_max(max);
  quantizer.set_min(min);
  quantizer.feed(data.data(), data.size());

  ASSERT_TRUE(quantizer.train());

  std::vector<uint8_t> qdata(data.size(), 0);
  quantizer.encode(data.data(), data.size(), qdata.data());

  std::vector<float> recover_data(data.size(), 0.0f);
  quantizer.decode(qdata.data(), data.size(), recover_data.data());

  float var = 0.0f;
  for (size_t i = 0; i < data.size(); ++i) {
    var += (data[i] - recover_data[i]) * (data[i] - recover_data[i]);
    // printf("%f %f\n", data[i], recover_data[i]);
  }
#if 0
  printf("max=%f min=%f\n", *std::max_element(data.begin(), data.end()),
         *std::min_element(data.begin(), data.end()));
  printf("recover max=%f min=%f\n",
         *std::max_element(recover_data.begin(), recover_data.end()),
         *std::min_element(recover_data.begin(), recover_data.end()));
  printf("var=%f\n", var);
#endif
  EXPECT_LT(var / COUNT, 500);
}

TEST(IntegerQuantizer, UINT8_Uniform_Distribution) {
  std::vector<size_t> tests = {1, 100, 1000, 10000, 100000};
  for (auto count : tests) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<float> data;

    std::uniform_real_distribution<float> dist(1.0, 2.0);
    float max = -std::numeric_limits<float>::max();
    float min = std::numeric_limits<float>::max();
    for (size_t i = 0; i < count; ++i) {
      auto v = dist(gen);
      max = std::max(max, v);
      min = std::min(min, v);
      data.emplace_back(v);
    }
    // data.emplace_back(10);  // deviation point
    EntropyUInt8Quantizer quantizer;
    quantizer.set_max(max);
    quantizer.set_min(min);
    quantizer.feed(data.data(), data.size());

    ASSERT_TRUE(quantizer.train());

    std::vector<uint8_t> qdata(data.size(), 0);
    quantizer.encode(data.data(), qdata.size(), qdata.data());

    std::vector<float> recover_data(data.size(), 0.0f);
    quantizer.decode(qdata.data(), qdata.size(), recover_data.data());

    float var = 0.0f;
    for (size_t i = 0; i < data.size(); ++i) {
      var += (data[i] - recover_data[i]) * (data[i] - recover_data[i]);
    }
    EXPECT_LT(var / count, 0.01);
  }
}

TEST(IntegerQuantizer, UINT8_Normal_Distribution) {
  const size_t COUNT = 10000u;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::vector<float> data;

  std::normal_distribution<float> dist(5.0f, 1.4f);
  float max = -std::numeric_limits<float>::max();
  float min = std::numeric_limits<float>::max();
  for (size_t i = 0; i < COUNT; ++i) {
    auto v = dist(gen);
    max = std::max(max, v);
    min = std::min(min, v);
    data.emplace_back(v);
  }
  // data.emplace_back(10);  // deviation point
  EntropyUInt8Quantizer quantizer;
  bool non_bias = dist(gen) > 5;
  quantizer.set_non_bias(non_bias);

  quantizer.set_max(max);
  quantizer.set_min(min);
  quantizer.feed(data.data(), data.size());

  ASSERT_TRUE(quantizer.train());
  ASSERT_EQ(quantizer.bias() == 0.0f, non_bias);

  std::vector<uint8_t> qdata(data.size(), 0);
  quantizer.encode(data.data(), qdata.size(), qdata.data());

  std::vector<float> recover_data(data.size(), 0.0f);
  quantizer.decode(qdata.data(), qdata.size(), recover_data.data());

  float var = 0.0f;
  for (size_t i = 0; i < data.size(); ++i) {
    var += (data[i] - recover_data[i]) * (data[i] - recover_data[i]);
    // printf("%f %f\n", data[i], recover_data[i]);
  }
#if 0
  printf("max=%f min=%f\n", *std::max_element(data.begin(), data.end()),
         *std::min_element(data.begin(), data.end()));
  printf("recover max=%f min=%f\n",
         *std::max_element(recover_data.begin(), recover_data.end()),
         *std::min_element(recover_data.begin(), recover_data.end()));
  printf("var=%f\n", var);
#endif
  EXPECT_LT(var / COUNT, 0.01);
}

TEST(IntegerQuantizer, UINT8_Poisson_Distribution) {
  const size_t COUNT = 100000u;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::vector<float> data;

  std::poisson_distribution<int> dist(10000);
  float max = -std::numeric_limits<float>::min();
  float min = std::numeric_limits<float>::max();
  for (size_t i = 0; i < COUNT; ++i) {
    float v = (float)dist(gen);
    max = std::max(max, v);
    min = std::min(min, v);
    data.emplace_back(v);
  }
  // data.emplace_back(10);  // deviation point
  EntropyUInt8Quantizer quantizer;

  quantizer.set_max(max);
  quantizer.set_min(min);
  quantizer.feed(data.data(), data.size());

  ASSERT_TRUE(quantizer.train());

  std::vector<uint8_t> qdata(data.size(), 0);
  quantizer.encode(data.data(), qdata.size(), qdata.data());

  std::vector<float> recover_data(data.size(), 0.0f);
  quantizer.decode(qdata.data(), qdata.size(), recover_data.data());

  float var = 0.0f;
  for (size_t i = 0; i < data.size(); ++i) {
    var += (data[i] - recover_data[i]) * (data[i] - recover_data[i]);
    // printf("%f %f\n", data[i], recover_data[i]);
  }
  EXPECT_LT(var / COUNT, 100);
}

TEST(IntegerQuantizer, UINT4_Uniform_Distribution) {
  std::vector<size_t> tests = {2, 100, 5000, 10000, 100000};
  for (auto count : tests) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<float> data;

    std::uniform_real_distribution<float> dist(1.0, 2.0);
    float max = -std::numeric_limits<float>::min();
    float min = std::numeric_limits<float>::max();
    for (size_t i = 0; i < count; ++i) {
      auto v = dist(gen);
      max = std::max(max, v);
      min = std::min(min, v);
      data.emplace_back(v);
    }
    // data.emplace_back(10);  // deviation point
    EntropyUInt4Quantizer quantizer;
    quantizer.set_max(max);
    quantizer.set_min(min);
    quantizer.feed(data.data(), data.size());

    ASSERT_TRUE(quantizer.train());

    std::vector<uint8_t> qdata(data.size() / 2, 0);
    quantizer.encode(data.data(), data.size(), qdata.data());

    std::vector<float> recover_data(data.size(), 0.0f);
    quantizer.decode(qdata.data(), data.size(), recover_data.data());

    float var = 0.0f;
    for (size_t i = 0; i < data.size(); ++i) {
      var += (data[i] - recover_data[i]) * (data[i] - recover_data[i]);
      // printf("%f %f\n", data[i], recover_data[i]);
    }
#if 0
    printf("max=%f min=%f\n", *std::max_element(data.begin(), data.end()),
           *std::min_element(data.begin(), data.end()));
    printf("recover max=%f min=%f\n",
           *std::max_element(recover_data.begin(), recover_data.end()),
           *std::min_element(recover_data.begin(), recover_data.end()));
    printf("var=%f\n", var);
#endif
    EXPECT_LT(var / count, 0.1);
  }
}

TEST(IntegerQuantizer, UINT4_Normal_Distribution) {
  const size_t COUNT = 100000u;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::vector<float> data;

  std::normal_distribution<float> avg(5, 1.0);
  std::normal_distribution<float> dist(avg(gen), 2);
  float max = -std::numeric_limits<float>::max();
  float min = std::numeric_limits<float>::max();
  for (size_t i = 0; i < COUNT; ++i) {
    auto v = dist(gen);
    max = std::max(max, v);
    min = std::min(min, v);
    data.emplace_back(v);
  }
  // data.emplace_back(10);  // deviation point
  EntropyUInt4Quantizer quantizer;
  bool non_bias = avg(gen) > 5;
  quantizer.set_non_bias(non_bias);

  quantizer.set_max(max);
  quantizer.set_min(min);
  quantizer.feed(data.data(), data.size());

  ASSERT_TRUE(quantizer.train());
  ASSERT_EQ(quantizer.bias() == 0.0f, non_bias);

  std::vector<uint8_t> qdata(data.size(), 0);
  quantizer.encode(data.data(), data.size(), qdata.data());

  std::vector<float> recover_data(data.size(), 0.0f);
  quantizer.decode(qdata.data(), data.size(), recover_data.data());

  float var = 0.0f;
  for (size_t i = 0; i < data.size(); ++i) {
    var += (data[i] - recover_data[i]) * (data[i] - recover_data[i]);
    // printf("%f %f\n", data[i], recover_data[i]);
  }
#if 0
  printf("max=%f min=%f\n", *std::max_element(data.begin(), data.end()),
         *std::min_element(data.begin(), data.end()));
  printf("recover max=%f min=%f\n",
         *std::max_element(recover_data.begin(), recover_data.end()),
         *std::min_element(recover_data.begin(), recover_data.end()));
  printf("var=%f\n", var);
#endif
  EXPECT_LT(var / COUNT, 2.0f);
}

TEST(IntegerQuantizer, UINT4_Poisson_Distribution) {
  const size_t COUNT = 100000u;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::vector<float> data;

  std::poisson_distribution<int> dist(10000);
  float max = -std::numeric_limits<float>::min();
  float min = std::numeric_limits<float>::max();
  for (size_t i = 0; i < COUNT; ++i) {
    float v = (float)dist(gen);
    max = std::max(max, v);
    min = std::min(min, v);
    data.emplace_back(v);
  }
  // data.emplace_back(10);  // deviation point
  EntropyUInt4Quantizer quantizer;

  quantizer.set_max(max);
  quantizer.set_min(min);
  quantizer.feed(data.data(), data.size());

  ASSERT_TRUE(quantizer.train());

  std::vector<uint8_t> qdata(data.size(), 0);
  quantizer.encode(data.data(), data.size(), qdata.data());

  std::vector<float> recover_data(data.size(), 0.0f);
  quantizer.decode(qdata.data(), data.size(), recover_data.data());

  float var = 0.0f;
  for (size_t i = 0; i < data.size(); ++i) {
    var += (data[i] - recover_data[i]) * (data[i] - recover_data[i]);
    // printf("%f %f\n", data[i], recover_data[i]);
  }
#if 0
  printf("max=%f min=%f\n", *std::max_element(data.begin(), data.end()),
         *std::min_element(data.begin(), data.end()));
  printf("recover max=%f min=%f\n",
         *std::max_element(recover_data.begin(), recover_data.end()),
         *std::min_element(recover_data.begin(), recover_data.end()));
  printf("var=%f\n", var);
#endif
  EXPECT_LT(var / COUNT, 350);
}
}  // namespace zvec::ailego