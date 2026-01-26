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
#include <iostream>
#include <ailego/math/norm_matrix.h>
#include <gtest/gtest.h>
#include <zvec/ailego/utility/float_helper.h>
#include "framework/index_factory.h"


using namespace zvec;
using namespace zvec::core;
using namespace zvec::ailego;

static void Norm2(std::vector<Float16> &vec, std::string *out) {
  float norm = 0.0f;

  out->resize(vec.size() * sizeof(Float16) + sizeof(float));

  Norm2Matrix<Float16, 1>::Compute(vec.data(), vec.size(), &norm);

  Float16 *buf = reinterpret_cast<Float16 *>(&(*out)[0]);

  for (uint32_t i = 0; i < vec.size(); ++i) {
    buf[i] = vec[i] / norm;
  }

  float *norm_buf =
      reinterpret_cast<float *>(&(*out)[vec.size() * sizeof(Float16)]);

  memcpy(norm_buf, &norm, sizeof(float));
}

static void Norm2(std::vector<float> &vec, std::string *out) {
  float norm = 0.0f;

  out->resize((vec.size() + 1) * sizeof(float));

  Norm2Matrix<float, 1>::Compute(vec.data(), vec.size(), &norm);

  float *buf = reinterpret_cast<float *>(&(*out)[0]);
  for (uint32_t i = 0; i < vec.size(); ++i) {
    buf[i] = vec[i] / norm;
  }

  buf[vec.size()] = norm;
}

static size_t ExtraDimension(IndexMeta::DataType type) {
  // The extra quantized params storage size to save for each vector
  if (type == IndexMeta::DT_FP32) return 1;
  if (type == IndexMeta::DT_FP16) return 2;

  return 0;
}

TEST(CosineMeasure_General_Test, General) {
  auto measure = IndexFactory::CreateMetric("Cosine");
  EXPECT_TRUE(measure);

  IndexMeta meta;
  meta.set_meta(IndexMeta::DT_INT16, 64);
  ASSERT_NE(0, measure->init(meta, Params()));
  meta.set_meta(IndexMeta::DT_FP16, 64);
  ASSERT_EQ(0, measure->init(meta, Params()));
  meta.set_meta(IndexMeta::DT_FP32, 64);
  ASSERT_EQ(0, measure->init(meta, Params()));
  meta.set_meta(IndexMeta::DT_INT8, 64);
  ASSERT_NE(0, measure->init(meta, Params()));

  meta.set_meta(IndexMeta::DT_BINARY32, 64);
  ASSERT_NE(0, measure->init(meta, Params()));
  meta.set_meta(IndexMeta::DT_BINARY64, 64);
  ASSERT_NE(0, measure->init(meta, Params()));
  meta.set_meta(IndexMeta::DT_INT4, 64);
  ASSERT_NE(0, measure->init(meta, Params()));

  IndexMeta meta2;
  meta2.set_meta(IndexMeta::DT_BINARY32, 64);
  EXPECT_FALSE(measure->is_matched(meta2));
  EXPECT_TRUE(
      measure->is_matched(meta, IndexQueryMeta(IndexMeta::DT_FP32, 64)));
  EXPECT_FALSE(
      measure->is_matched(meta, IndexQueryMeta(IndexMeta::DT_FP32, 63)));

  EXPECT_FALSE(measure->distance_matrix(0, 0));
  EXPECT_FALSE(measure->distance_matrix(3, 5));
  EXPECT_FALSE(measure->distance_matrix(31, 65));
  EXPECT_TRUE(measure->distance_matrix(1, 1));
  EXPECT_FALSE(measure->distance_matrix(2, 1));
  EXPECT_FALSE(measure->distance_matrix(2, 2));
  EXPECT_FALSE(measure->distance_matrix(4, 1));
  EXPECT_FALSE(measure->distance_matrix(4, 2));
  EXPECT_FALSE(measure->distance_matrix(4, 4));
  EXPECT_FALSE(measure->distance_matrix(8, 1));
  EXPECT_FALSE(measure->distance_matrix(8, 2));
  EXPECT_FALSE(measure->distance_matrix(8, 4));
  EXPECT_FALSE(measure->distance_matrix(8, 8));
  EXPECT_FALSE(measure->distance_matrix(16, 1));
  EXPECT_FALSE(measure->distance_matrix(16, 2));
  EXPECT_FALSE(measure->distance_matrix(16, 4));
  EXPECT_FALSE(measure->distance_matrix(16, 8));
  EXPECT_FALSE(measure->distance_matrix(16, 16));
  EXPECT_FALSE(measure->distance_matrix(32, 1));
  EXPECT_FALSE(measure->distance_matrix(32, 2));
  EXPECT_FALSE(measure->distance_matrix(32, 4));
  EXPECT_FALSE(measure->distance_matrix(32, 8));
  EXPECT_FALSE(measure->distance_matrix(32, 16));
  EXPECT_FALSE(measure->distance_matrix(32, 32));

  EXPECT_FALSE(measure->support_normalize());
  float result = 1.0f;
  measure->normalize(&result);
  EXPECT_FLOAT_EQ(1.0f, result);
}

TEST(CosineMeasure_General_Test, TestDistanceFp32) {
  {
    constexpr uint32_t dimension = 2;
    IndexMeta meta;
    meta.set_meta(IndexMeta::DT_FP32, dimension);

    auto measure = IndexFactory::CreateMetric("Cosine");
    ASSERT_TRUE(measure);
    Params params;
    ASSERT_EQ(0, measure->init(meta, params));
    ASSERT_EQ(false, measure->support_train());

    auto distance = measure->distance();
    ASSERT_NE(distance, nullptr);
    auto dist_matrix = measure->distance_matrix(1, 1);
    ASSERT_NE(dist_matrix, nullptr);

    std::vector<float> a = {0.2f, 0.9f};
    std::vector<float> b = {0.3f, 0.5f};

    std::string a_out;
    std::string b_out;

    Norm2(a, &a_out);
    Norm2(b, &b_out);

    float result = 0.0f;
    distance(a_out.data(), b_out.data(),
             dimension + ExtraDimension(IndexMeta::DT_FP32), &result);

    if (measure->support_normalize()) {
      measure->normalize(&result);
    }

    EXPECT_GE(0.00001f, std::abs(result - 0.05131668f));

    dist_matrix(a_out.data(), b_out.data(),
                dimension + ExtraDimension(IndexMeta::DT_FP32), &result);

    if (measure->support_normalize()) {
      measure->normalize(&result);
    }

    EXPECT_GE(0.00001f, std::abs(result - 0.05131668f));
  }

  {
    constexpr uint32_t dimension = 3;
    IndexMeta meta;
    meta.set_meta(IndexMeta::DT_FP32, dimension);

    auto measure = IndexFactory::CreateMetric("Cosine");
    ASSERT_TRUE(measure);
    Params params;
    ASSERT_EQ(0, measure->init(meta, params));
    ASSERT_EQ(false, measure->support_train());

    auto distance = measure->distance();
    ASSERT_NE(distance, nullptr);
    auto dist_matrix = measure->distance_matrix(1, 1);
    ASSERT_NE(dist_matrix, nullptr);

    std::vector<float> a = {0.2f, 0.9f, 0.6f};
    std::vector<float> b = {0.3f, 0.5f, 0.7f};

    std::string a_out;
    std::string b_out;

    Norm2(a, &a_out);
    Norm2(b, &b_out);

    float result = 0.0f;
    distance(a_out.data(), b_out.data(),
             dimension + ExtraDimension(IndexMeta::DT_FP32), &result);

    if (measure->support_normalize()) {
      measure->normalize(&result);
    }

    EXPECT_GE(0.00001f, std::abs(result - 0.07199293f));

    dist_matrix(a_out.data(), b_out.data(),
                dimension + ExtraDimension(IndexMeta::DT_FP32), &result);

    if (measure->support_normalize()) {
      measure->normalize(&result);
    }

    EXPECT_GE(0.00001f, std::abs(result - 0.07199293f));
  }

  {
    constexpr uint32_t dimension = 11;
    IndexMeta meta;
    meta.set_meta(IndexMeta::DT_FP32, dimension);

    auto measure = IndexFactory::CreateMetric("Cosine");
    ASSERT_TRUE(measure);
    Params params;
    ASSERT_EQ(0, measure->init(meta, params));
    ASSERT_EQ(false, measure->support_train());

    auto distance = measure->distance();
    ASSERT_NE(distance, nullptr);
    auto dist_matrix = measure->distance_matrix(1, 1);
    ASSERT_NE(dist_matrix, nullptr);

    std::vector<float> a = {1.0f, 2.0f, 3.0f, 0.2f, 0.3f, 0.1f,
                            5.2f, 2.1f, 7.1f, 6.8f, 1.2f};
    std::vector<float> b = {2.0f, 4.0f, 6.0f, 0.6f, 0.7f, 0.9f,
                            1.0f, 2.3f, 3.4f, 4.5f, 6.4f};


    std::string a_out;
    std::string b_out;

    Norm2(a, &a_out);
    Norm2(b, &b_out);

    float result = 0.0f;
    distance(a_out.data(), b_out.data(),
             dimension + ExtraDimension(IndexMeta::DT_FP32), &result);

    if (measure->support_normalize()) {
      measure->normalize(&result);
    }

    EXPECT_GE(0.00001f, std::abs(result - 0.2803060f));

    dist_matrix(a_out.data(), b_out.data(),
                dimension + ExtraDimension(IndexMeta::DT_FP32), &result);

    if (measure->support_normalize()) {
      measure->normalize(&result);
    }

    EXPECT_GE(0.00001f, std::abs(result - 0.2803060f));
  }
}

TEST(CosineMeasure_General_Test, TestDistanceFp16) {
  {
    constexpr uint32_t dimension = 2;
    IndexMeta meta;
    meta.set_meta(IndexMeta::DT_FP16, dimension);

    auto measure = IndexFactory::CreateMetric("Cosine");
    ASSERT_TRUE(measure);
    Params params;
    ASSERT_EQ(0, measure->init(meta, params));
    ASSERT_EQ(false, measure->support_train());

    auto distance = measure->distance();
    ASSERT_NE(distance, nullptr);
    auto dist_matrix = measure->distance_matrix(1, 1);
    ASSERT_NE(dist_matrix, nullptr);

    std::vector<Float16> a = {0.2f, 0.9f};
    std::vector<Float16> b = {0.3f, 0.5f};

    std::string a_out;
    std::string b_out;

    Norm2(a, &a_out);
    Norm2(b, &b_out);

    float result = 0.0f;
    distance(a_out.data(), b_out.data(),
             dimension + ExtraDimension(IndexMeta::DT_FP16), &result);

    if (measure->support_normalize()) {
      measure->normalize(&result);
    }

    EXPECT_GE(0.001f, std::abs(result - 0.05131668f));

    dist_matrix(a_out.data(), b_out.data(),
                dimension + ExtraDimension(IndexMeta::DT_FP16), &result);

    if (measure->support_normalize()) {
      measure->normalize(&result);
    }

    EXPECT_GE(0.001f, std::abs(result - 0.05131668f));
  }

  {
    constexpr uint32_t dimension = 3;
    IndexMeta meta;
    meta.set_meta(IndexMeta::DT_FP16, dimension);

    auto measure = IndexFactory::CreateMetric("Cosine");
    ASSERT_TRUE(measure);
    Params params;
    ASSERT_EQ(0, measure->init(meta, params));
    ASSERT_EQ(false, measure->support_train());

    auto distance = measure->distance();
    ASSERT_NE(distance, nullptr);
    auto dist_matrix = measure->distance_matrix(1, 1);
    ASSERT_NE(dist_matrix, nullptr);

    std::vector<Float16> a = {0.2f, 0.9f, 0.6f};
    std::vector<Float16> b = {0.3f, 0.5f, 0.7f};

    std::string a_out;
    std::string b_out;

    Norm2(a, &a_out);
    Norm2(b, &b_out);

    float result = 0.0f;
    distance(a_out.data(), b_out.data(),
             dimension + ExtraDimension(IndexMeta::DT_FP16), &result);

    if (measure->support_normalize()) {
      measure->normalize(&result);
    }

    EXPECT_GE(0.001f, std::abs(result - 0.07199293f));

    dist_matrix(a_out.data(), b_out.data(),
                dimension + ExtraDimension(IndexMeta::DT_FP16), &result);

    if (measure->support_normalize()) {
      measure->normalize(&result);
    }

    EXPECT_GE(0.001f, std::abs(result - 0.07199293f));
  }

  {
    constexpr uint32_t dimension = 11;
    IndexMeta meta;
    meta.set_meta(IndexMeta::DT_FP16, dimension);

    auto measure = IndexFactory::CreateMetric("Cosine");
    ASSERT_TRUE(measure);
    Params params;
    ASSERT_EQ(0, measure->init(meta, params));
    ASSERT_EQ(false, measure->support_train());

    auto distance = measure->distance();
    ASSERT_NE(distance, nullptr);
    auto dist_matrix = measure->distance_matrix(1, 1);
    ASSERT_NE(dist_matrix, nullptr);

    std::vector<Float16> a = {1.0f, 2.0f, 3.0f, 0.2f, 0.3f, 0.1f,
                              5.2f, 2.1f, 7.1f, 6.8f, 1.2f};
    std::vector<Float16> b = {2.0f, 4.0f, 6.0f, 0.6f, 0.7f, 0.9f,
                              1.0f, 2.3f, 3.4f, 4.5f, 6.4f};

    std::string a_out;
    std::string b_out;

    Norm2(a, &a_out);
    Norm2(b, &b_out);

    float result = 0.0f;
    dist_matrix(a_out.data(), b_out.data(),
                dimension + ExtraDimension(IndexMeta::DT_FP16), &result);

    if (measure->support_normalize()) {
      measure->normalize(&result);
    }

    EXPECT_GE(0.001f, std::abs(result - 0.2803060f));

    dist_matrix(a_out.data(), b_out.data(),
                dimension + ExtraDimension(IndexMeta::DT_FP16), &result);

    if (measure->support_normalize()) {
      measure->normalize(&result);
    }

    EXPECT_GE(0.001f, std::abs(result - 0.2803060f));
  }
}

TEST(CosineMeasure_General_Test, TestDistanceBatchFp16Simple) {
  {
    constexpr uint32_t dimension = 2;
    IndexMeta meta;
    meta.set_meta(IndexMeta::DT_FP16, dimension);

    auto measure = IndexFactory::CreateMetric("Cosine");
    ASSERT_TRUE(measure);
    Params params;
    ASSERT_EQ(0, measure->init(meta, params));
    ASSERT_EQ(false, measure->support_train());

    auto dist_batch = measure->batch_distance();
    ASSERT_NE(dist_batch, nullptr);

    std::vector<Float16> a = {0.2f, 0.9f};
    std::vector<Float16> b = {0.3f, 0.5f};

    std::string a_out;
    std::string b_out;


    Norm2(a, &a_out);
    Norm2(b, &b_out);

    float results[2] = {0.0f, 0.0f};

    const void *vecs[2];
    vecs[0] = a_out.data();
    vecs[1] = b_out.data();
    dist_batch(vecs, b_out.data(), 2,
               dimension + ExtraDimension(IndexMeta::DT_FP16), results);

    if (measure->support_normalize()) {
      measure->normalize(&results[0]);
      measure->normalize(&results[1]);
    }

    EXPECT_GE(0.001f, std::abs(results[0] - 0.05131668f));
    EXPECT_GE(0.001f, std::abs(results[1] - 0.0f));
  }
}

TEST(CosineMeasure_General_Test, TestDistanceBatchFp32Simple) {
  {
    constexpr uint32_t dimension = 2;
    IndexMeta meta;
    meta.set_meta(IndexMeta::DT_FP32, dimension);

    auto measure = IndexFactory::CreateMetric("Cosine");
    ASSERT_TRUE(measure);
    Params params;
    ASSERT_EQ(0, measure->init(meta, params));
    ASSERT_EQ(false, measure->support_train());

    auto dist_batch = measure->batch_distance();
    ASSERT_NE(dist_batch, nullptr);

    std::vector<float> a = {0.2f, 0.9f};
    std::vector<float> b = {0.3f, 0.5f};

    std::string a_out;
    std::string b_out;

    Norm2(a, &a_out);
    Norm2(b, &b_out);

    float results[2] = {0.0f, 0.0f};

    const void *vecs[2];
    vecs[0] = a_out.data();
    vecs[1] = b_out.data();
    dist_batch(vecs, b_out.data(), 2,
               dimension + ExtraDimension(IndexMeta::DT_FP32), results);

    if (measure->support_normalize()) {
      measure->normalize(&results[0]);
      measure->normalize(&results[1]);
    }

    EXPECT_GE(0.00001f, std::abs(results[0] - 0.05131668f));
    EXPECT_GE(0.00001f, std::abs(results[1] - 0.0f));
  }
}

template <typename T>
void calculate_distance(std::vector<T> &a, std::vector<T> &b, size_t dimension,
                        IndexMeta::DataType data_type, size_t batch_size,
                        float expected_distance, float epsilon = 0.00001f) {
  IndexMeta meta;
  meta.set_meta(data_type, dimension);

  auto measure = IndexFactory::CreateMetric("Cosine");
  ASSERT_TRUE(measure);
  Params params;
  ASSERT_EQ(0, measure->init(meta, params));
  ASSERT_EQ(false, measure->support_train());

  auto dist_batch = measure->batch_distance();
  ASSERT_NE(dist_batch, nullptr);

  std::string a_out;
  std::string b_out;

  Norm2(a, &a_out);
  Norm2(b, &b_out);

  float results[2] = {0.0f, 0.0f};

  const void *vecs[2];
  vecs[0] = a_out.data();
  vecs[1] = b_out.data();
  dist_batch(vecs, b_out.data(), batch_size,
             dimension + ExtraDimension(data_type), results);

  if (measure->support_normalize()) {
    measure->normalize(&results[0]);
    measure->normalize(&results[1]);
  }

  EXPECT_GE(epsilon, std::abs(results[0] - expected_distance));
  EXPECT_GE(epsilon, std::abs(results[1] - 0.0f));
}


TEST(CosineMeasure_General_Test, TestDistanceBatch) {
  {
    constexpr uint32_t dimension = 2;

    {
      std::vector<float> a = {0.2f, 0.9f};
      std::vector<float> b = {0.3f, 0.5f};

      calculate_distance(a, b, dimension, IndexMeta::DT_FP32, 1, 0.05131668f,
                         0.00001f);
      calculate_distance(a, b, dimension, IndexMeta::DT_FP32, 2, 0.05131668f,
                         0.00001f);
    }
    {
      std::vector<Float16> a = {0.2f, 0.9f};
      std::vector<Float16> b = {0.3f, 0.5f};

      calculate_distance(a, b, dimension, IndexMeta::DT_FP16, 1, 0.05131668f,
                         0.001f);
      calculate_distance(a, b, dimension, IndexMeta::DT_FP16, 2, 0.05131668f,
                         0.001f);
    }
  }

  {
    constexpr uint32_t dimension = 3;


    {
      std::vector<float> a = {0.2f, 0.9f, 0.6f};
      std::vector<float> b = {0.3f, 0.5f, 0.7f};

      calculate_distance(a, b, dimension, IndexMeta::DT_FP32, 1, 0.07199293f,
                         0.00001f);
      calculate_distance(a, b, dimension, IndexMeta::DT_FP32, 2, 0.07199293f,
                         0.00001f);
    }
    {
      std::vector<Float16> a = {0.2f, 0.9f, 0.6f};
      std::vector<Float16> b = {0.3f, 0.5f, 0.7f};

      calculate_distance(a, b, dimension, IndexMeta::DT_FP16, 1, 0.07199293f,
                         0.001f);
      calculate_distance(a, b, dimension, IndexMeta::DT_FP16, 2, 0.07199293f,
                         0.001f);
    }
  }

  {
    constexpr uint32_t dimension = 11;

    {
      std::vector<float> a = {1.0f, 2.0f, 3.0f, 0.2f, 0.3f, 0.1f,
                              5.2f, 2.1f, 7.1f, 6.8f, 1.2f};
      std::vector<float> b = {2.0f, 4.0f, 6.0f, 0.6f, 0.7f, 0.9f,
                              1.0f, 2.3f, 3.4f, 4.5f, 6.4f};

      calculate_distance(a, b, dimension, IndexMeta::DT_FP32, 1, 0.2803060f,
                         0.00001f);
      calculate_distance(a, b, dimension, IndexMeta::DT_FP32, 2, 0.2803060f,
                         0.00001f);
    }

    {
      std::vector<Float16> a = {1.0f, 2.0f, 3.0f, 0.2f, 0.3f, 0.1f,
                                5.2f, 2.1f, 7.1f, 6.8f, 1.2f};
      std::vector<Float16> b = {2.0f, 4.0f, 6.0f, 0.6f, 0.7f, 0.9f,
                                1.0f, 2.3f, 3.4f, 4.5f, 6.4f};

      calculate_distance(a, b, dimension, IndexMeta::DT_FP16, 1, 0.2803060f,
                         0.001f);
      calculate_distance(a, b, dimension, IndexMeta::DT_FP16, 2, 0.2803060f,
                         0.001f);
    }
  }
}
