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

#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <ailego/container/bitmap.h>
#include <ailego/internal/cpu_features.h>
#include <ailego/math/distance.h>
#include <gtest/gtest.h>
#include <zvec/ailego/container/vector.h>
#include <zvec/ailego/utility/time_helper.h>

using namespace zvec::ailego;

static inline const char *IntelIntrinsics(void) {
  return internal::CpuFeatures::Intrinsics();
}

static inline void MatrixTranspose(Float16 *dst, const Float16 *src, size_t M,
                                   size_t N) {
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < M; ++j) {
      dst[j * N + i] = src[i * M + j];
    }
  }
}

template <size_t N>
static float InnerProductDistance(const FixedVector<Float16, N> &lhs,
                                  const FixedVector<Float16, N> &rhs) {
  return Distance::InnerProduct(lhs.data(), rhs.data(), lhs.size());
}

template <size_t N>
static float MinusInnerProductDistance(const FixedVector<Float16, N> &lhs,
                                       const FixedVector<Float16, N> &rhs) {
  return Distance::MinusInnerProduct(lhs.data(), rhs.data(), lhs.size());
}

TEST(DistanceMatrix, InnerProduct_General) {
  FixedVector<Float16, 15> x15{5.22f,  0.65f, 0.711f, 7.8f,  8.9f,
                               555.0f, 0.8f,  5.5f,   3.75f, 9.0f,
                               6.6f,   0.1f,  8.8f,   0.2f,  5.6f},
      y15{5.22f, 0.65f, 0.711f, 7.8f, 8.9f, 555.0f, 0.8f, 5.5f,
          3.75f, 9.0f,  6.6f,   0.1f, 8.8f, 0.2f,   0.25f};
  EXPECT_TRUE(MathHelper::IsAlmostEqual(308441.62f,
                                        InnerProductDistance(x15, y15), 1000));

  FixedVector<Float16, 16> x16{5.22f, 0.65f, 0.711f, 7.8f,  8.9f, 555.0f,
                               9.12f, 0.8f,  5.5f,   3.75f, 9.0f, 6.6f,
                               0.1f,  8.8f,  0.2f,   5.6f},
      y16{5.22f, 0.65f, 0.711f, 7.8f, 8.9f, 555.0f, 9.12f, 0.8f,
          5.5f,  3.75f, 9.0f,   6.6f, 0.1f, 8.8f,   0.2f,  0.5f};
  EXPECT_TRUE(MathHelper::IsAlmostEqual(308526.19f,
                                        InnerProductDistance(x16, y16), 1000));

  FixedVector<Float16, 17> x17{3.4f, 4.5f,  5.6f, 1.6f,  3.4f,  8.1f,
                               1.0f, 4.41f, 7.7f, 1.11f, 3.04f, 2.3f,
                               3.4f, 4.5f,  5.6f, 1.6f,  1.3f},
      y17{3.4f,  4.5f,  5.6f, 1.6f, 3.4f, 8.1f, 1.0f, 4.41f, 7.7f,
          1.11f, 3.04f, 2.3f, 3.4f, 4.5f, 5.6f, 1.6f, 2.3f};
  EXPECT_TRUE(MathHelper::IsAlmostEqual(307.1762f,
                                        InnerProductDistance(x17, y17), 1000));

  FixedVector<Float16, 18> x18{1.66f, 2.3f, 1.11f, 3.04f,  8.23f, 1.0f,
                               4.44f, 7.7f, 1.5f,  11.11f, 2.3f,  3.4f,
                               4.5f,  5.6f, 1.6f,  2.3f,   1.11f, 3.04f},
      y18{1.66f,  2.3f, 1.11f, 3.04f, 8.23f, 1.0f, 4.44f, 7.7f,  1.5f,
          11.11f, 2.3f, 3.4f,  4.5f,  5.6f,  1.6f, 2.3f,  1.11f, 3.04f};
  EXPECT_TRUE(MathHelper::IsAlmostEqual(378.67197f,
                                        InnerProductDistance(x18, y18), 1000));

  FixedVector<Float16, 19> x19{1.66f, 2.3f,  1.11f, 3.04f,  8.23f, 1.0f, 1.6f,
                               2.3f,  4.44f, 7.7f,  11.11f, 2.3f,  3.4f, 4.5f,
                               5.6f,  1.6f,  2.3f,  1.11f,  2.3f},
      y19{1.66f,  2.3f, 1.11f, 3.04f, 8.23f, 1.0f, 1.6f, 2.3f,  4.44f, 7.7f,
          11.11f, 2.3f, 3.4f,  4.5f,  5.6f,  1.6f, 2.3f, 1.11f, 2.3f};
  EXPECT_TRUE(MathHelper::IsAlmostEqual(380.33203f,
                                        InnerProductDistance(x19, y19), 1000));

  FixedVector<Float16, 20> x20{1.6f, 2.3f, 1.11f, 2.3f, 3.04f, 8.23f, 1.0f,
                               1.6f, 2.3f, 5.6f,  1.6f, 2.3f,  2.3f,  3.4f,
                               4.5f, 5.6f, 1.6f,  2.3f, 1.11f, 2.3f},
      y20{1.6f, 2.3f, 1.11f, 2.3f, 3.04f, 8.23f, 1.0f, 1.6f, 2.3f,  5.6f,
          1.6f, 2.3f, 2.3f,  3.4f, 4.5f,  5.6f,  1.6f, 2.3f, 1.11f, 2.3f};
  EXPECT_TRUE(MathHelper::IsAlmostEqual(222.23581f,
                                        InnerProductDistance(x20, y20), 1000));

  FixedVector<Float16, 21> x21{0.0f}, y21{0.0f};
  EXPECT_TRUE(
      MathHelper::IsAlmostEqual(0.0f, InnerProductDistance(x21, y21), 1000));
}

TEST(DistanceMatrix, MinusInnerProduct_General) {
  FixedVector<Float16, 1> x1{0.7f}, y1{0.5f};
  EXPECT_TRUE(MathHelper::IsAlmostEqual(
      -0.35009766f, MinusInnerProductDistance(x1, y1), 1000));

  FixedVector<Float16, 2> x2{2.0f, 3.76f}, y2{2.0f, 0.901f};
  EXPECT_TRUE(MathHelper::IsAlmostEqual(
      -7.387093f, MinusInnerProductDistance(x2, y2), 1000));

  FixedVector<Float16, 3> x3{2.0f, 3.0f, 0.7f}, y3{2.0f, 3.0f, 2.0f};
  EXPECT_TRUE(MathHelper::IsAlmostEqual(
      -14.400391f, MinusInnerProductDistance(x3, y3), 1000));

  FixedVector<Float16, 4> x4{7.8f, -8.9f, 9.0f, 5.6f},
      y4{7.8f, 8.9f, -9.0f, -0.1f};
  EXPECT_TRUE(MathHelper::IsAlmostEqual(
      99.89003f, MinusInnerProductDistance(x4, y4), 1000));

  FixedVector<Float16, 5> x5{7.8f, 8.9f, 9.0f, 0.1f, 5.6f},
      y5{7.8f, 8.9f, 9.0f, 0.1f, 0.2f};
  EXPECT_TRUE(MathHelper::IsAlmostEqual(
      -222.16441f, MinusInnerProductDistance(x5, y5), 1000));

  FixedVector<Float16, 6> x6{0.711f, 7.8f, 8.9f, 9.0f, 0.1f, 5.6f},
      y6{0.711f, 7.8f, 8.9f, 9.0f, 0.1f, 0.2f};
  EXPECT_TRUE(MathHelper::IsAlmostEqual(
      -222.66985f, MinusInnerProductDistance(x6, y6), 1000));

  FixedVector<Float16, 7> x7{5.22f, 0.711f, 7.8f, 8.9f, 9.0f, 0.1f, 5.6f},
      y7{5.22f, 0.711f, 7.8f, 8.9f, 9.0f, 0.1f, 0.2f};
  EXPECT_TRUE(MathHelper::IsAlmostEqual(
      -249.9052f, MinusInnerProductDistance(x7, y7), 1000));

  FixedVector<Float16, 8> x8{5.22f, 0.711f, 7.8f, 8.9f, 9.0f, 0.1f, 0.2f, 5.6f},
      y8{5.22f, 0.711f, -7.8f, -8.9f, -9.0f, 0.1f, 0.2f, 0.5f};
  EXPECT_TRUE(MathHelper::IsAlmostEqual(
      190.44284f, MinusInnerProductDistance(x8, y8), 1000));

  FixedVector<Float16, 9> x9{5.22f, 0.711f, 7.8f, 8.9f, 9.0f,
                             6.6f,  0.1f,   0.2f, 5.6f},
      y9{5.22f, 0.711f, 7.8f, 8.9f, 9.0f, 6.6f, 0.1f, 0.2f, 0.5f};
  EXPECT_TRUE(MathHelper::IsAlmostEqual(
      -295.20654f, MinusInnerProductDistance(x9, y9), 1000));

  FixedVector<Float16, 10> x10{5.22f, 0.711f, 7.8f, 8.9f, 5.5f,
                               9.0f,  6.6f,   0.1f, 0.2f, 5.6f},
      y10{5.22f, 0.711f, 7.8f, 8.9f, 5.5f, 9.0f, 6.6f, 0.1f, 0.2f, 0.522f};
  EXPECT_TRUE(MathHelper::IsAlmostEqual(
      -325.57962f, MinusInnerProductDistance(x10, y10), 1000));

  FixedVector<Float16, 11> x11{2.3f,    -1.11f, 3.04f, 8.23f, 1.0f, 7.7f,
                               -11.11f, 2.3f,   3.4f,  4.5f,  5.6f},
      y11{2.3f,    1.11f, 3.04f, 8.23f, -1.0f, 7.7f,
          -11.11f, 2.3f,  3.4f,  4.5f,  0.511f};
  EXPECT_TRUE(MathHelper::IsAlmostEqual(
      -302.63904f, MinusInnerProductDistance(x11, y11), 1000));

  FixedVector<Float16, 12> x12{1.6f, 2.3f,   1.11f, 3.04f, 8.23f, 1.0f,
                               7.7f, 11.11f, 2.3f,  3.4f,  4.5f,  5.6f},
      y12{1.6f, 2.3f,   1.11f, 3.04f, 8.23f, 1.0f,
          7.7f, 11.11f, 2.3f,  3.4f,  4.5f,  0.5f};
  EXPECT_TRUE(MathHelper::IsAlmostEqual(
      -309.60065f, MinusInnerProductDistance(x12, y12), 1000));

  FixedVector<Float16, 13> x13{1.6f, 2.3f,   1.11f, 3.04f, 8.23f, 1.0f, 4.44f,
                               7.7f, 11.11f, 2.3f,  3.4f,  4.5f,  5.6f},
      y13{1.6f, 2.3f,   1.11f, 3.04f, 8.23f, 1.0f, 4.44f,
          7.7f, 11.11f, 2.3f,  3.4f,  4.5f,  3.5f};
  EXPECT_TRUE(MathHelper::IsAlmostEqual(
      -346.13144f, MinusInnerProductDistance(x13, y13), 1000));

  FixedVector<Float16, 14> x14{5.22f, 0.65f, 0.711f, 7.8f, 8.9f, 0.8f, 5.5f,
                               3.75f, 9.0f,  6.6f,   0.1f, 8.8f, 0.2f, 5.6f},
      y14{5.22f, 0.65f, 0.711f, 7.8f, 8.9f, 0.8f, 5.5f,
          3.75f, 9.0f,  6.6f,   0.1f, 8.8f, 0.2f, 0.5f};
  EXPECT_TRUE(MathHelper::IsAlmostEqual(
      -417.96613f, MinusInnerProductDistance(x14, y14), 1000));

  FixedVector<Float16, 15> x15{5.22f,  0.65f, 0.711f, 7.8f,  8.9f,
                               555.0f, 0.8f,  5.5f,   3.75f, 9.0f,
                               6.6f,   0.1f,  8.8f,   0.2f,  5.6f},
      y15{5.22f, 0.65f, 0.711f, 7.8f, 8.9f, 555.0f, 0.8f, 5.5f,
          3.75f, 9.0f,  6.6f,   0.1f, 8.8f, 0.2f,   0.25f};
  EXPECT_TRUE(MathHelper::IsAlmostEqual(
      -308441.62f, MinusInnerProductDistance(x15, y15), 1000));

  FixedVector<Float16, 16> x16{5.22f, 0.65f, 0.711f, 7.8f,  8.9f, 555.0f,
                               9.12f, 0.8f,  5.5f,   3.75f, 9.0f, 6.6f,
                               0.1f,  8.8f,  0.2f,   5.6f},
      y16{5.22f, 0.65f, 0.711f, 7.8f, 8.9f, 555.0f, 9.12f, 0.8f,
          5.5f,  3.75f, 9.0f,   6.6f, 0.1f, 8.8f,   0.2f,  0.5f};
  EXPECT_TRUE(MathHelper::IsAlmostEqual(
      -308526.19f, MinusInnerProductDistance(x16, y16), 1000));

  FixedVector<Float16, 17> x17{3.4f, 4.5f,  5.6f, 1.6f,  3.4f,  8.1f,
                               1.0f, 4.41f, 7.7f, 1.11f, 3.04f, 2.3f,
                               3.4f, 4.5f,  5.6f, 1.6f,  1.3f},
      y17{3.4f,  4.5f,  5.6f, 1.6f, 3.4f, 8.1f, 1.0f, 4.41f, 7.7f,
          1.11f, 3.04f, 2.3f, 3.4f, 4.5f, 5.6f, 1.6f, 2.3f};
  EXPECT_TRUE(MathHelper::IsAlmostEqual(
      -307.17618f, MinusInnerProductDistance(x17, y17), 1000));

  FixedVector<Float16, 18> x18{1.66f, 2.3f, 1.11f, 3.04f,  8.23f, 1.0f,
                               4.44f, 7.7f, 1.5f,  11.11f, 2.3f,  3.4f,
                               4.5f,  5.6f, 1.6f,  2.3f,   1.11f, 3.04f},
      y18{1.66f,  2.3f, 1.11f, 3.04f, 8.23f, 1.0f, 4.44f, 7.7f,  1.5f,
          11.11f, 2.3f, 3.4f,  4.5f,  5.6f,  1.6f, 2.3f,  1.11f, 3.04f};
  EXPECT_TRUE(MathHelper::IsAlmostEqual(
      -378.67197f, MinusInnerProductDistance(x18, y18), 1000));

  FixedVector<Float16, 19> x19{1.66f, 2.3f,  1.11f, 3.04f,  8.23f, 1.0f, 1.6f,
                               2.3f,  4.44f, 7.7f,  11.11f, 2.3f,  3.4f, 4.5f,
                               5.6f,  1.6f,  2.3f,  1.11f,  2.3f},
      y19{1.66f,  2.3f, 1.11f, 3.04f, 8.23f, 1.0f, 1.6f, 2.3f,  4.44f, 7.7f,
          11.11f, 2.3f, 3.4f,  4.5f,  5.6f,  1.6f, 2.3f, 1.11f, 2.3f};
  EXPECT_TRUE(MathHelper::IsAlmostEqual(
      -380.33203f, MinusInnerProductDistance(x19, y19), 1000));

  FixedVector<Float16, 20> x20{1.6f, 2.3f, 1.11f, 2.3f, 3.04f, 8.23f, 1.0f,
                               1.6f, 2.3f, 5.6f,  1.6f, 2.3f,  2.3f,  3.4f,
                               4.5f, 5.6f, 1.6f,  2.3f, 1.11f, 2.3f},
      y20{1.6f, 2.3f, 1.11f, 2.3f, 3.04f, 8.23f, 1.0f, 1.6f, 2.3f,  5.6f,
          1.6f, 2.3f, 2.3f,  3.4f, 4.5f,  5.6f,  1.6f, 2.3f, 1.11f, 2.3f};
  EXPECT_TRUE(MathHelper::IsAlmostEqual(
      -222.23581f, MinusInnerProductDistance(x20, y20), 1000));

  FixedVector<Float16, 21> x21{0.0f}, y21{0.0f};
  EXPECT_TRUE(MathHelper::IsAlmostEqual(
      0.0f, MinusInnerProductDistance(x21, y21), 1000));
}

template <size_t M, size_t N>
void TestInnerProductMatrix(void) {
  std::mt19937 gen((std::random_device())());

  const size_t batch_size = M;
  const size_t query_size = N;
  size_t dimension = (std::uniform_int_distribution<size_t>(1, 65))(gen);
  size_t matrix_size = batch_size * dimension;
  size_t query_matrix_size = query_size * dimension;

  std::vector<Float16> matrix1(matrix_size);
  std::vector<Float16> matrix2(matrix_size);
  std::vector<Float16> query1(query_matrix_size);
  std::vector<Float16> query2(query_matrix_size);
  std::vector<float> result1(batch_size * query_size);
  std::vector<float> result2(batch_size * query_size);

  std::uniform_real_distribution<float> dist(0.0, 1.0);
  for (size_t i = 0; i < matrix_size; ++i) {
    matrix1[i] = dist(gen);
  }
  for (size_t i = 0; i < query_matrix_size; ++i) {
    query1[i] = dist(gen);
  }
  MatrixTranspose(&matrix2[0], matrix1.data(), dimension, batch_size);
  MatrixTranspose(&query2[0], query1.data(), dimension, query_size);

  for (size_t i = 0; i < query_size; ++i) {
    const Float16 *cur_query = &query1[i * dimension];
    float *query_result = &result1[i * batch_size];

    for (size_t j = 0; j < batch_size; ++j) {
      InnerProductMatrix<Float16, 1, 1>::Compute(
          &matrix1[j * dimension], cur_query, dimension, &query_result[j]);
    }
  }
  InnerProductMatrix<Float16, batch_size, query_size>::Compute(
      &matrix2[0], &query2[0], dimension, &result2[0]);

  for (size_t i = 0; i < batch_size * query_size; ++i) {
    EXPECT_TRUE(MathHelper::IsAlmostEqual(result1[i], result2[i], 10000));
  }
}

template <size_t M, size_t N>
void TestMinusInnerProductMatrix(void) {
  std::mt19937 gen((std::random_device())());

  const size_t batch_size = M;
  const size_t query_size = N;
  size_t dimension = (std::uniform_int_distribution<size_t>(1, 65))(gen);
  size_t matrix_size = batch_size * dimension;
  size_t query_matrix_size = query_size * dimension;

  std::vector<Float16> matrix1(matrix_size);
  std::vector<Float16> matrix2(matrix_size);
  std::vector<Float16> query1(query_matrix_size);
  std::vector<Float16> query2(query_matrix_size);
  std::vector<float> result1(batch_size * query_size);
  std::vector<float> result2(batch_size * query_size);

  std::uniform_real_distribution<float> dist(0.0, 1.0);
  for (size_t i = 0; i < matrix_size; ++i) {
    matrix1[i] = dist(gen);
  }
  for (size_t i = 0; i < query_matrix_size; ++i) {
    query1[i] = dist(gen);
  }
  MatrixTranspose(&matrix2[0], matrix1.data(), dimension, batch_size);
  MatrixTranspose(&query2[0], query1.data(), dimension, query_size);

  for (size_t i = 0; i < query_size; ++i) {
    const Float16 *cur_query = &query1[i * dimension];
    float *query_result = &result1[i * batch_size];

    for (size_t j = 0; j < batch_size; ++j) {
      MinusInnerProductMatrix<Float16, 1, 1>::Compute(
          &matrix1[j * dimension], cur_query, dimension, &query_result[j]);
    }
  }
  MinusInnerProductMatrix<Float16, batch_size, query_size>::Compute(
      &matrix2[0], &query2[0], dimension, &result2[0]);

  for (size_t i = 0; i < batch_size * query_size; ++i) {
    EXPECT_TRUE(MathHelper::IsAlmostEqual(result1[i], result2[i], 10000));
  }
}

TEST(DistanceMatrix, InnerProduct_1x1) {
  TestInnerProductMatrix<1, 1>();
}

TEST(DistanceMatrix, InnerProduct_2x1) {
  TestInnerProductMatrix<2, 1>();
}

TEST(DistanceMatrix, InnerProduct_2x2) {
  TestInnerProductMatrix<2, 2>();
}

TEST(DistanceMatrix, InnerProduct_3x3) {
  TestInnerProductMatrix<3, 3>();
}

TEST(DistanceMatrix, InnerProduct_4x1) {
  TestInnerProductMatrix<4, 1>();
}

TEST(DistanceMatrix, InnerProduct_4x2) {
  TestInnerProductMatrix<4, 2>();
}

TEST(DistanceMatrix, InnerProduct_4x4) {
  TestInnerProductMatrix<4, 4>();
}

TEST(DistanceMatrix, InnerProduct_8x1) {
  TestInnerProductMatrix<8, 1>();
}

TEST(DistanceMatrix, InnerProduct_8x2) {
  TestInnerProductMatrix<8, 2>();
}

TEST(DistanceMatrix, InnerProduct_8x4) {
  TestInnerProductMatrix<8, 4>();
}

TEST(DistanceMatrix, InnerProduct_8x8) {
  TestInnerProductMatrix<8, 8>();
}

TEST(DistanceMatrix, InnerProduct_16x1) {
  TestInnerProductMatrix<16, 1>();
}

TEST(DistanceMatrix, InnerProduct_16x2) {
  TestInnerProductMatrix<16, 2>();
}

TEST(DistanceMatrix, InnerProduct_16x4) {
  TestInnerProductMatrix<16, 4>();
}

TEST(DistanceMatrix, InnerProduct_16x8) {
  TestInnerProductMatrix<16, 8>();
}

TEST(DistanceMatrix, InnerProduct_16x16) {
  TestInnerProductMatrix<16, 16>();
}

TEST(DistanceMatrix, InnerProduct_32x1) {
  TestInnerProductMatrix<32, 1>();
}

TEST(DistanceMatrix, InnerProduct_32x2) {
  TestInnerProductMatrix<32, 2>();
}

TEST(DistanceMatrix, InnerProduct_32x4) {
  TestInnerProductMatrix<32, 4>();
}

TEST(DistanceMatrix, InnerProduct_32x8) {
  TestInnerProductMatrix<32, 8>();
}

TEST(DistanceMatrix, InnerProduct_32x16) {
  TestInnerProductMatrix<32, 16>();
}

TEST(DistanceMatrix, InnerProduct_32x32) {
  TestInnerProductMatrix<32, 32>();
}

TEST(DistanceMatrix, InnerProduct_64x1) {
  TestInnerProductMatrix<64, 1>();
}

TEST(DistanceMatrix, InnerProduct_64x2) {
  TestInnerProductMatrix<64, 2>();
}

TEST(DistanceMatrix, InnerProduct_64x4) {
  TestInnerProductMatrix<64, 4>();
}

TEST(DistanceMatrix, InnerProduct_64x8) {
  TestInnerProductMatrix<64, 8>();
}

TEST(DistanceMatrix, InnerProduct_64x16) {
  TestInnerProductMatrix<64, 16>();
}

TEST(DistanceMatrix, InnerProduct_64x32) {
  TestInnerProductMatrix<64, 32>();
}

TEST(DistanceMatrix, InnerProduct_64x64) {
  TestInnerProductMatrix<64, 64>();
}

TEST(DistanceMatrix, InnerProduct_128x1) {
  TestInnerProductMatrix<128, 1>();
}

TEST(DistanceMatrix, InnerProduct_128x2) {
  TestInnerProductMatrix<128, 2>();
}

TEST(DistanceMatrix, InnerProduct_128x4) {
  TestInnerProductMatrix<128, 4>();
}

TEST(DistanceMatrix, InnerProduct_128x8) {
  TestInnerProductMatrix<128, 8>();
}

TEST(DistanceMatrix, InnerProduct_128x16) {
  TestInnerProductMatrix<128, 16>();
}

TEST(DistanceMatrix, InnerProduct_128x32) {
  TestInnerProductMatrix<128, 32>();
}

TEST(DistanceMatrix, InnerProduct_128x64) {
  TestInnerProductMatrix<128, 64>();
}

TEST(DistanceMatrix, InnerProduct_128x128) {
  TestInnerProductMatrix<128, 128>();
}

TEST(DistanceMatrix, MinusInnerProduct_1x1) {
  TestMinusInnerProductMatrix<1, 1>();
}

TEST(DistanceMatrix, MinusInnerProduct_2x1) {
  TestMinusInnerProductMatrix<2, 1>();
}

TEST(DistanceMatrix, MinusInnerProduct_2x2) {
  TestMinusInnerProductMatrix<2, 2>();
}

TEST(DistanceMatrix, MinusInnerProduct_3x3) {
  TestMinusInnerProductMatrix<3, 3>();
}

TEST(DistanceMatrix, MinusInnerProduct_4x1) {
  TestMinusInnerProductMatrix<4, 1>();
}

TEST(DistanceMatrix, MinusInnerProduct_4x2) {
  TestMinusInnerProductMatrix<4, 2>();
}

TEST(DistanceMatrix, MinusInnerProduct_4x4) {
  TestMinusInnerProductMatrix<4, 4>();
}

TEST(DistanceMatrix, MinusInnerProduct_8x1) {
  TestMinusInnerProductMatrix<8, 1>();
}

TEST(DistanceMatrix, MinusInnerProduct_8x2) {
  TestMinusInnerProductMatrix<8, 2>();
}

TEST(DistanceMatrix, MinusInnerProduct_8x4) {
  TestMinusInnerProductMatrix<8, 4>();
}

TEST(DistanceMatrix, MinusInnerProduct_8x8) {
  TestMinusInnerProductMatrix<8, 8>();
}

TEST(DistanceMatrix, MinusInnerProduct_16x1) {
  TestMinusInnerProductMatrix<16, 1>();
}

TEST(DistanceMatrix, MinusInnerProduct_16x2) {
  TestMinusInnerProductMatrix<16, 2>();
}

TEST(DistanceMatrix, MinusInnerProduct_16x4) {
  TestMinusInnerProductMatrix<16, 4>();
}

TEST(DistanceMatrix, MinusInnerProduct_16x8) {
  TestMinusInnerProductMatrix<16, 8>();
}

TEST(DistanceMatrix, MinusInnerProduct_16x16) {
  TestMinusInnerProductMatrix<16, 16>();
}

TEST(DistanceMatrix, MinusInnerProduct_32x1) {
  TestMinusInnerProductMatrix<32, 1>();
}

TEST(DistanceMatrix, MinusInnerProduct_32x2) {
  TestMinusInnerProductMatrix<32, 2>();
}

TEST(DistanceMatrix, MinusInnerProduct_32x4) {
  TestMinusInnerProductMatrix<32, 4>();
}

TEST(DistanceMatrix, MinusInnerProduct_32x8) {
  TestMinusInnerProductMatrix<32, 8>();
}

TEST(DistanceMatrix, MinusInnerProduct_32x16) {
  TestMinusInnerProductMatrix<32, 16>();
}

TEST(DistanceMatrix, MinusInnerProduct_32x32) {
  TestMinusInnerProductMatrix<32, 32>();
}

TEST(DistanceMatrix, MinusInnerProduct_64x1) {
  TestMinusInnerProductMatrix<64, 1>();
}

TEST(DistanceMatrix, MinusInnerProduct_64x2) {
  TestMinusInnerProductMatrix<64, 2>();
}

TEST(DistanceMatrix, MinusInnerProduct_64x4) {
  TestMinusInnerProductMatrix<64, 4>();
}

TEST(DistanceMatrix, MinusInnerProduct_64x8) {
  TestMinusInnerProductMatrix<64, 8>();
}

TEST(DistanceMatrix, MinusInnerProduct_64x16) {
  TestMinusInnerProductMatrix<64, 16>();
}

TEST(DistanceMatrix, MinusInnerProduct_64x32) {
  TestMinusInnerProductMatrix<64, 32>();
}

TEST(DistanceMatrix, MinusInnerProduct_64x64) {
  TestMinusInnerProductMatrix<64, 64>();
}

TEST(DistanceMatrix, MinusInnerProduct_128x1) {
  TestMinusInnerProductMatrix<128, 1>();
}

TEST(DistanceMatrix, MinusInnerProduct_128x2) {
  TestMinusInnerProductMatrix<128, 2>();
}

TEST(DistanceMatrix, MinusInnerProduct_128x4) {
  TestMinusInnerProductMatrix<128, 4>();
}

TEST(DistanceMatrix, MinusInnerProduct_128x8) {
  TestMinusInnerProductMatrix<128, 8>();
}

TEST(DistanceMatrix, MinusInnerProduct_128x16) {
  TestMinusInnerProductMatrix<128, 16>();
}

TEST(DistanceMatrix, MinusInnerProduct_128x32) {
  TestMinusInnerProductMatrix<128, 32>();
}

TEST(DistanceMatrix, MinusInnerProduct_128x64) {
  TestMinusInnerProductMatrix<128, 64>();
}

TEST(DistanceMatrix, MinusInnerProduct_128x128) {
  TestMinusInnerProductMatrix<128, 128>();
}

template <size_t M, size_t N, size_t B, size_t D>
void InnerProductBenchmark(void) {
  const size_t dimension = D;
  const size_t batch_size = M;
  const size_t block_size = B;
  const size_t query_size = N;
  const size_t matrix_size = block_size * batch_size * dimension;
  const size_t query_matrix_size = dimension * query_size;

  std::vector<Float16> matrix1(matrix_size);
  std::vector<Float16> matrix2(matrix_size);
  std::vector<Float16> query1(query_matrix_size);
  std::vector<Float16> query2(query_matrix_size);

  std::mt19937 gen((std::random_device())());
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < matrix_size; ++i) {
    matrix1[i] = dist(gen);
  }
  for (size_t i = 0; i < query_matrix_size; ++i) {
    query1[i] = dist(gen);
  }

  for (size_t i = 0; i < block_size; ++i) {
    size_t start_pos = i * batch_size * dimension;
    MatrixTranspose(&matrix2[start_pos], &matrix1[start_pos], dimension,
                    batch_size);
  }
  MatrixTranspose(&query2[0], query1.data(), dimension, query_size);

  ElapsedTime elapsed_time;
  float results[batch_size * query_size];

  std::cout << "# (" << IntelIntrinsics() << ") FP16 " << dimension << "d, "
            << batch_size << " * " << query_size << " * " << block_size
            << std::endl;

  // 1 Batched InnerProduct
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const Float16 *matrix_batch = &matrix2[i * batch_size * dimension];

    for (size_t j = 0; j < query_size; ++j) {
      const Float16 *current_query = &query1[j * dimension];
      float *current_results = &results[j * batch_size];

      InnerProductMatrix<Float16, batch_size, 1>::Compute(
          matrix_batch, current_query, dimension, current_results);
    }
  }
  std::cout << "* 1 Batched InnerProduct (us) \t"
            << elapsed_time.micro_seconds() << std::endl;

  // N Batched InnerProduct
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const Float16 *matrix_batch = &matrix2[i * batch_size * dimension];

    InnerProductMatrix<Float16, batch_size, query_size>::Compute(
        matrix_batch, &query2[0], dimension, results);
  }
  std::cout << "* N Batched InnerProduct (us) \t"
            << elapsed_time.micro_seconds() << std::endl;

  // Unbatched InnerProduct
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const Float16 *matrix_batch = &matrix1[i * batch_size * dimension];

    for (size_t j = 0; j < query_size; ++j) {
      const Float16 *current_query = &query1[j * dimension];
      float *current_results = &results[j * batch_size];

      for (size_t k = 0; k < batch_size; ++k) {
        InnerProductMatrix<Float16, 1, 1>::Compute(&matrix_batch[k * dimension],
                                                   current_query, dimension,
                                                   &current_results[k]);
      }
    }
  }
  std::cout << "* Unbatched InnerProduct (us) \t"
            << elapsed_time.micro_seconds() << std::endl;
}

template <size_t M, size_t N, size_t B, size_t D>
void MinusInnerProductBenchmark(void) {
  const size_t dimension = D;
  const size_t batch_size = M;
  const size_t block_size = B;
  const size_t query_size = N;
  const size_t matrix_size = block_size * batch_size * dimension;
  const size_t query_matrix_size = dimension * query_size;

  std::vector<Float16> matrix1(matrix_size);
  std::vector<Float16> matrix2(matrix_size);
  std::vector<Float16> query1(query_matrix_size);
  std::vector<Float16> query2(query_matrix_size);

  std::mt19937 gen((std::random_device())());
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < matrix_size; ++i) {
    matrix1[i] = dist(gen);
  }
  for (size_t i = 0; i < query_matrix_size; ++i) {
    query1[i] = dist(gen);
  }

  for (size_t i = 0; i < block_size; ++i) {
    size_t start_pos = i * batch_size * dimension;
    MatrixTranspose(&matrix2[start_pos], &matrix1[start_pos], dimension,
                    batch_size);
  }
  MatrixTranspose(&query2[0], query1.data(), dimension, query_size);

  ElapsedTime elapsed_time;
  float results[batch_size * query_size];

  std::cout << "# (" << IntelIntrinsics() << ") FP16 " << dimension << "d, "
            << batch_size << " * " << query_size << " * " << block_size
            << std::endl;

  // 1 Batched InnerProduct
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const Float16 *matrix_batch = &matrix2[i * batch_size * dimension];

    for (size_t j = 0; j < query_size; ++j) {
      const Float16 *current_query = &query1[j * dimension];
      float *current_results = &results[j * batch_size];

      MinusInnerProductMatrix<Float16, batch_size, 1>::Compute(
          matrix_batch, current_query, dimension, current_results);
    }
  }
  std::cout << "* 1 Batched MinusInnerProduct (us) \t"
            << elapsed_time.micro_seconds() << std::endl;

  // N Batched InnerProduct
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const Float16 *matrix_batch = &matrix2[i * batch_size * dimension];

    MinusInnerProductMatrix<Float16, batch_size, query_size>::Compute(
        matrix_batch, &query2[0], dimension, results);
  }
  std::cout << "* N Batched MinusInnerProduct (us) \t"
            << elapsed_time.micro_seconds() << std::endl;

  // Unbatched InnerProduct
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const Float16 *matrix_batch = &matrix1[i * batch_size * dimension];

    for (size_t j = 0; j < query_size; ++j) {
      const Float16 *current_query = &query1[j * dimension];
      float *current_results = &results[j * batch_size];

      for (size_t k = 0; k < batch_size; ++k) {
        MinusInnerProductMatrix<Float16, 1, 1>::Compute(
            &matrix_batch[k * dimension], current_query, dimension,
            &current_results[k]);
      }
    }
  }
  std::cout << "* Unbatched MinusInnerProduct (us) \t"
            << elapsed_time.micro_seconds() << std::endl;
}

TEST(DistanceMatrix, DISABLED_InnerProduct_Benchmark) {
  InnerProductBenchmark<2, 1, 512, 64>();
  InnerProductBenchmark<2, 2, 512, 64>();
  InnerProductBenchmark<4, 1, 512, 64>();
  InnerProductBenchmark<4, 2, 512, 64>();
  InnerProductBenchmark<4, 4, 512, 64>();
  InnerProductBenchmark<8, 1, 512, 64>();
  InnerProductBenchmark<8, 2, 512, 64>();
  InnerProductBenchmark<8, 4, 512, 64>();
  InnerProductBenchmark<8, 8, 512, 64>();
  InnerProductBenchmark<16, 1, 512, 64>();
  InnerProductBenchmark<16, 2, 512, 64>();
  InnerProductBenchmark<16, 4, 512, 64>();
  InnerProductBenchmark<16, 8, 512, 64>();
  InnerProductBenchmark<16, 16, 512, 64>();
  InnerProductBenchmark<32, 1, 512, 64>();
  InnerProductBenchmark<32, 2, 512, 64>();
  InnerProductBenchmark<32, 4, 512, 64>();
  InnerProductBenchmark<32, 8, 512, 64>();
  InnerProductBenchmark<32, 16, 512, 64>();
  InnerProductBenchmark<32, 32, 512, 64>();
  InnerProductBenchmark<64, 1, 512, 64>();
  InnerProductBenchmark<64, 2, 512, 64>();
  InnerProductBenchmark<64, 4, 512, 64>();
  InnerProductBenchmark<64, 8, 512, 64>();
  InnerProductBenchmark<128, 1, 512, 64>();
  InnerProductBenchmark<1, 1, 1024, 256>();
}

TEST(DistanceMatrix, DISABLED_MinusInnerProduct_Benchmark) {
  MinusInnerProductBenchmark<2, 1, 512, 64>();
  MinusInnerProductBenchmark<2, 2, 512, 64>();
  MinusInnerProductBenchmark<4, 1, 512, 64>();
  MinusInnerProductBenchmark<4, 2, 512, 64>();
  MinusInnerProductBenchmark<4, 4, 512, 64>();
  MinusInnerProductBenchmark<8, 1, 512, 64>();
  MinusInnerProductBenchmark<8, 2, 512, 64>();
  MinusInnerProductBenchmark<8, 4, 512, 64>();
  MinusInnerProductBenchmark<8, 8, 512, 64>();
  MinusInnerProductBenchmark<16, 1, 512, 64>();
  MinusInnerProductBenchmark<16, 2, 512, 64>();
  MinusInnerProductBenchmark<16, 4, 512, 64>();
  MinusInnerProductBenchmark<16, 8, 512, 64>();
  MinusInnerProductBenchmark<16, 16, 512, 64>();
  MinusInnerProductBenchmark<32, 1, 512, 64>();
  MinusInnerProductBenchmark<32, 2, 512, 64>();
  MinusInnerProductBenchmark<32, 4, 512, 64>();
  MinusInnerProductBenchmark<32, 8, 512, 64>();
  MinusInnerProductBenchmark<32, 16, 512, 64>();
  MinusInnerProductBenchmark<32, 32, 512, 64>();
  MinusInnerProductBenchmark<64, 1, 512, 64>();
  MinusInnerProductBenchmark<64, 2, 512, 64>();
  MinusInnerProductBenchmark<64, 4, 512, 64>();
  MinusInnerProductBenchmark<64, 8, 512, 64>();
  MinusInnerProductBenchmark<128, 1, 512, 64>();
  MinusInnerProductBenchmark<1, 1, 1024, 256>();
}

TEST(DistanceMatrix, DISABLED_MinusInnerProduct_BenchmarkSimple) {
  std::mt19937 gen((std::random_device())());

  size_t dimension = 768;
  size_t loop_cnt = 100000000LLU;

  std::vector<Float16> data(dimension);
  std::vector<Float16> query(dimension);

  float result;

  std::uniform_real_distribution<float> dist(0.0, 1.0);
  for (size_t i = 0; i < dimension; ++i) {
    data[i] = dist(gen);
  }
  for (size_t i = 0; i < dimension; ++i) {
    query[i] = dist(gen);
  }

  for (size_t i = 0; i < loop_cnt; ++i) {
    MinusInnerProductMatrix<Float16, 1, 1>::Compute(&data[0], &query[0],
                                                    dimension, &result);
  }
}

static inline float SparseDistanceCommon(uint32_t count1, uint32_t *index1,
                                         Float16 *value1, uint32_t count2,
                                         uint32_t *index2, Float16 *value2) {
  float result{0.0f};

  size_t m = 0;
  size_t q = 0;
  while (m < count1 && q < count2) {
    if (index1[m] == index2[q]) {
      result += value1[m] * value2[q];

      ++m;
      ++q;
    } else if (index1[m] < index2[q]) {
      ++m;
    } else {
      ++q;
    }
  }

  return result;
}

void TestInnerProductSparse(void) {
  // test 1
  const uint32_t sparse_vec_count_0 = 52;
  uint32_t sparse_vec_index_0[] = {
      33,   66,   77,   209,  385,  396,  539,  583,  649,  715,  880,
      935,  968,  1023, 1100, 1111, 1661, 1694, 1749, 2288, 2343, 2453,
      2530, 2772, 2871, 2882, 2948, 3069, 3322, 3333, 3410, 3575, 3608,
      4026, 4037, 4048, 4059, 4070, 4268, 4323, 4741, 4752, 5137, 5170,
      5423, 5555, 5918, 6028, 6094, 6347, 6369, 6468};
  FixedVector<Float16, sparse_vec_count_0> sparse_vec_value_0{
      -0.246404298254, 0.892043114755,  0.163785949199,  -0.680309913534,
      -0.767956138324, -0.410683610329, 0.763314047145,  0.347851184825,
      -0.676969102165, -0.774662820732, 0.274471489215,  -0.131269040962,
      0.206478593023,  0.764082612827,  -0.57678381864,  -0.256053693585,
      0.661507236032,  -0.812832823664, 0.929611593685,  -0.381852499144,
      -0.35890001953,  0.538386710846,  -0.829565442015, 0.384046166409,
      0.623125501212,  0.043215334982,  -0.689536097425, -0.500913794456,
      -0.419818105671, -0.503346955801, -0.99419236655,  -0.414091535679,
      -0.829474457209, -0.103915702521, -0.419445202934, -0.26891898936,
      0.311013521629,  0.172923023003,  -0.818231467063, -0.728015315042,
      0.110116365075,  0.845786117564,  -0.587841450807, 0.533763235805,
      -0.601437402994, -0.117487602176, 0.106103380748,  -0.00151542886833,
      0.189967593506,  0.890365538566,  -0.581876671583, -0.232173604777};

  const uint32_t sparse_vec_count_1 = 43;
  uint32_t sparse_vec_index_1[] = {
      33,   77,   110,  209,  1023, 1111, 1221, 1496, 1661, 1749, 2189,
      2255, 2288, 2420, 2530, 2695, 2772, 2838, 2948, 3179, 3575, 4202,
      4268, 4290, 4433, 4444, 4653, 4697, 4741, 5137, 5192, 5346, 5423,
      5445, 5555, 5588, 5764, 5896, 5918, 6028, 6270, 6347, 6501};
  FixedVector<Float16, sparse_vec_count_1> sparse_vec_value_1{
      -0.847561468192, -0.761580890729,  0.683791378502,  0.729670644228,
      -0.111989702001, -0.3435914518,    -0.806454864134, -0.0243347460596,
      0.497209110076,  0.852745969955,   0.403748558594,  -0.634016410599,
      -0.74513226711,  0.738086689871,   0.364575651925,  0.0867637408004,
      -0.285921174394, -0.321390976616,  -0.971849760722, -0.246041408731,
      -0.110667223833, 0.0744013655781,  0.84846334839,   0.167405689007,
      0.0289923642993, -0.536394124155,  0.518249809298,  -0.695798108647,
      0.0653215071151, -0.0046338401448, 0.644189056747,  -0.52301532328,
      -0.660275328421, 0.643514995264,   0.0333307952838, -0.401825159735,
      -0.188869041499, -0.23065238799,   -0.409416817144, -0.142933941372,
      0.247628793044,  -0.984985692607,  -0.427929860028};

  std::string sparse_query_buffer_0;
  MinusInnerProductSparseMatrix<Float16>::transform_sparse_format(
      sparse_vec_count_0, sparse_vec_index_0, sparse_vec_value_0.data(),
      sparse_query_buffer_0);

  std::string sparse_query_buffer_1;
  MinusInnerProductSparseMatrix<Float16>::transform_sparse_format(
      sparse_vec_count_1, sparse_vec_index_1, sparse_vec_value_1.data(),
      sparse_query_buffer_1);

  float result0{0.0f};
  result0 = SparseDistanceCommon(sparse_vec_count_0, sparse_vec_index_0,
                                 sparse_vec_value_0.data(), sparse_vec_count_1,
                                 sparse_vec_index_1, sparse_vec_value_1.data());

  float result1{0.0f};
  MinusInnerProductSparseMatrix<Float16>::Compute(
      sparse_query_buffer_0.data(), sparse_query_buffer_1.data(), &result1);
  result1 = -result1;

  EXPECT_GE(0.00001, std::abs(result0 - result1));

  // test 2
  constexpr uint32_t sparse_vec_count_2 = 49;
  uint32_t sparse_vec_index_2[] = {
      13200,  20900,  36300,  41800,  50600,  74800,  78100,  81400,  93500,
      99000,  107800, 121000, 127600, 137500, 140800, 143000, 145200, 166100,
      174900, 193600, 194700, 195800, 233200, 261800, 262900, 273900, 277200,
      299200, 302500, 343200, 381700, 387200, 418000, 421300, 436700, 449900,
      480700, 510400, 586300, 596200, 603900, 607200, 612700, 625900, 632500,
      633600, 639100, 642400, 650100};
  FixedVector<Float16, sparse_vec_count_2> sparse_vec_value_2{
      0.167493264953,  0.178347102375,   0.61850792017,    0.707662206696,
      -0.604456492928, 0.898905062153,   -0.971984671516,  -0.337950525868,
      -0.942538751319, -0.115612454156,  0.78433412971,    0.601522288928,
      -0.640321042923, -0.235673191423,  0.00632807223978, 0.629970437467,
      0.966519256786,  -0.279362437157,  0.396153064627,   -0.614592812875,
      -0.642157513141, 0.686723258138,   0.10227967727,    -0.5921196708,
      0.499411577177,  -0.0188556369919, 0.512245212443,   0.424666758023,
      0.299827154891,  -0.615468257454,  -0.0499098903374, -0.54873640329,
      0.899673049133,  -0.873237346565,  0.463117084808,   -0.810200151551,
      0.676836615658,  0.596247430713,   0.946225552468,   0.968425796351,
      -0.821041580744, -0.697734977387,  0.295618053879,   -0.476597945375,
      -0.246035224835, 0.927603570489,   -0.640242995569,  0.610224433234,
      -0.657550506633};

  constexpr uint32_t sparse_vec_count_3 = 58;
  uint32_t sparse_vec_index_3[] = {
      13200,  19800,  37400,  56100,  68200,  78100,  81400,  99000,  103400,
      107800, 108900, 110000, 111100, 125400, 127600, 137500, 141900, 151800,
      154000, 155100, 158400, 163900, 165000, 173800, 198000, 201300, 215600,
      247500, 249700, 264000, 269500, 287100, 291500, 311300, 312400, 336600,
      353100, 354200, 361900, 367400, 390500, 398200, 407000, 414700, 424600,
      510400, 533500, 535700, 551100, 556600, 568700, 576400, 577500, 590700,
      592900, 618200, 631400, 636900};
  FixedVector<Float16, sparse_vec_count_3> sparse_vec_value_3{
      0.175769744964,  -0.198506965419,  0.0842021015107, 0.544957076263,
      0.0856447356878, 0.838582935178,   0.796525374862,  -0.931940801441,
      0.555150441425,  0.957490431546,   -0.422126167235, -0.40903200281,
      0.242643233475,  0.698565387541,   -0.325754491857, 0.540403772154,
      -0.449888493042, 0.349262051644,   -0.612943655195, 0.874112675658,
      0.943939922271,  -0.994946966212,  -0.978705162429, 0.321190597007,
      0.17722019302,   0.6041089417,     -0.353184098327, -0.938569390092,
      -0.92268220981,  -0.268600478592,  -0.598069229627, 0.0720175726713,
      0.426800021137,  0.369250757861,   -0.823348360327, -0.664061107875,
      -0.418342805261, -0.430818720049,  0.0941988181812, 0.0765632945538,
      -0.148533061047, 0.404665036566,   -0.170747760502, -0.206564280292,
      0.311035754032,  0.498520039471,   -0.16255148444,  -0.137950933749,
      -0.234990864629, 0.602901363949,   0.0297103943437, -0.730955584059,
      0.117169059405,  -0.0746546228896, 0.39067258928,   -0.214782717972,
      -0.111009971497, -0.87766242691};

  std::string sparse_query_buffer_2;
  MinusInnerProductSparseMatrix<Float16>::transform_sparse_format(
      sparse_vec_count_2, sparse_vec_index_2, sparse_vec_value_2.data(),
      sparse_query_buffer_2);

  std::string sparse_query_buffer_3;
  MinusInnerProductSparseMatrix<Float16>::transform_sparse_format(
      sparse_vec_count_3, sparse_vec_index_3, sparse_vec_value_3.data(),
      sparse_query_buffer_3);

  float result2{0.0f};
  result2 = SparseDistanceCommon(sparse_vec_count_2, sparse_vec_index_2,
                                 sparse_vec_value_2.data(), sparse_vec_count_3,
                                 sparse_vec_index_3, sparse_vec_value_3.data());

  float result3{0.0f};
  MinusInnerProductSparseMatrix<Float16>::Compute(
      sparse_query_buffer_2.data(), sparse_query_buffer_3.data(), &result3);
  result3 = -result3;

  EXPECT_GE(0.00001, std::abs(result2 - result3));
}

void TestInnerProductSparseMore(void) {
  std::vector<uint32_t> sparse_vec_counts;
  std::vector<uint32_t *> sparse_vec_indices;
  std::vector<Float16 *> sparse_vec_values;

  const uint32_t sparse_vec_count_0 = 173;
  uint32_t sparse_vec_index_0[] = {
      1012,  1996,  2001,  2018,  2020,  2036,  2037,  2056,  2058,  2069,
      2111,  2116,  2138,  2162,  2166,  2245,  2253,  2259,  2306,  2307,
      2318,  2331,  2351,  2359,  2390,  2419,  2426,  2428,  2466,  2470,
      2535,  2554,  2557,  2568,  2590,  2622,  2671,  2739,  2765,  2812,
      2817,  2837,  2913,  2920,  3003,  3092,  3112,  3125,  3144,  3214,
      3241,  3249,  3260,  3268,  3271,  3278,  3280,  3330,  3463,  3478,
      3716,  3739,  3768,  3800,  3908,  3934,  3992,  4028,  4045,  4072,
      4146,  4254,  4301,  4382,  4454,  4471,  4504,  4517,  4598,  4806,
      4807,  4847,  4928,  4988,  5081,  5113,  5177,  5190,  5197,  5201,
      5234,  5456,  5621,  5689,  5792,  5817,  5823,  5875,  5920,  5921,
      5951,  5968,  6033,  6112,  6145,  6215,  6344,  6396,  6429,  6438,
      6529,  6627,  6691,  6731,  6801,  6865,  6950,  7036,  7128,  7155,
      7461,  7551,  7596,  7691,  7784,  7789,  7848,  7857,  8044,  8052,
      8053,  8553,  8573,  8664,  8817,  8826,  9250,  9273,  9593,  9727,
      10013, 10106, 10617, 10639, 10753, 11657, 12108, 13128, 13463, 13702,
      13787, 14152, 14332, 15237, 15313, 15359, 15699, 16724, 17171, 17571,
      17669, 20168, 20805, 20972, 22134, 22229, 22779, 24762, 24823, 25526,
      25699, 26761, 27885};
  FixedVector<Float16, sparse_vec_count_0> sparse_vec_value_0{
      0.36311877,  0.10386213,  0.64821976,   0.26300138,    0.29727572,
      0.047292523, 0.022334402, 0.118793316,  0.7198291,     0.73566943,
      0.19491579,  0.5763569,   0.5245229,    0.022828134,   0.43562022,
      0.6946562,   0.09275672,  0.9687072,    0.1751608,     0.09703954,
      0.18717986,  0.43182945,  0.055112287,  0.0021027816,  0.13972417,
      0.1019873,   0.8679199,   0.26797894,   0.097350314,   0.5125363,
      0.2829703,   0.052232087, 0.3248494,    1.1258097,     0.90756655,
      1.6490538,   0.45066822,  0.004210417,  0.028443621,   0.41171393,
      0.09246816,  0.053040083, 0.052729037,  0.00041907438, 0.32047704,
      0.2290303,   1.3542659,   0.28811434,   1.1722984,     0.4484738,
      0.73670006,  0.22390367,  0.0058781556, 0.48173144,    0.76392287,
      0.32048634,  0.42589885,  0.8624791,    0.0376546,     0.56702816,
      0.002337549, 1.5856861,   0.14177673,   0.22762497,    0.6601752,
      1.0603137,   0.914821,    0.34792075,   1.4387932,     0.035774633,
      0.04391008,  0.7179224,   0.49199906,   0.043692447,   1.1404462,
      0.47572234,  0.22777049,  0.7626374,    0.59730506,    1.4541638,
      1.6540457,   0.089919806, 0.0050144624, 0.15902519,    0.2989032,
      0.121926464, 0.11911,     0.27476037,   1.2774497,     0.42462146,
      0.30179682,  0.18773684,  0.82144237,   1.2033592,     0.07180116,
      0.06378868,  0.029040875, 0.2089903,    0.03591103,    0.94913304,
      0.18240769,  0.9050947,   0.0034226696, 1.2841027,     0.629526,
      0.06401547,  1.0698998,   0.11138009,   0.20497903,    0.017457427,
      0.6316996,   0.12303611,  0.01563728,   0.090583175,   0.23981698,
      0.48518667,  0.6207808,   1.8336427,    2.3282833,     0.8153351,
      0.026216522, 0.6143031,   0.17374748,   0.32929608,    0.33730298,
      1.1497657,   0.1926745,   0.14235665,   1.1076177,     0.945609,
      0.48826388,  0.10458124,  0.19699246,   0.20899634,    0.44853806,
      0.26411146,  0.7495864,   1.3681723,    1.4299264,     0.037516754,
      0.17946614,  0.98060745,  0.055851664,  0.2002921,     0.45136684,
      0.33716172,  0.58752763,  0.34051904,   1.9018586,     0.20597915,
      0.82819384,  0.23866963,  0.4160662,    0.11889692,    0.172538,
      0.005433464, 0.089198045, 0.3896585,    0.74038976,    0.24974349,
      0.044961147, 0.32671204,  0.044312827,  0.25430596,    0.021065181,
      0.071978964, 1.992692,    0.02640776,   1.7344381,     0.09561436,
      0.07097204,  0.2922402,   0.8794989};

  const uint32_t sparse_vec_count_1 = 144;
  uint32_t sparse_vec_index_1[] = {
      1012,  1016,  1059,  1996,  2001,  2020,  2049,  2068,  2076,  2088,
      2109,  2138,  2145,  2149,  2162,  2203,  2220,  2224,  2256,  2259,
      2318,  2373,  2381,  2390,  2393,  2419,  2462,  2466,  2485,  2506,
      2554,  2557,  2580,  2590,  2622,  2633,  2645,  2671,  2716,  2724,
      2900,  2942,  2943,  3003,  3029,  3092,  3112,  3125,  3260,  3271,
      3278,  3283,  3288,  3439,  3466,  3478,  3521,  3578,  3594,  3595,
      3607,  3647,  3690,  3800,  3826,  3896,  3908,  3934,  3947,  3987,
      4045,  4068,  4204,  4254,  4255,  4302,  4329,  4471,  4504,  4517,
      4566,  4736,  4762,  4789,  5081,  5094,  5105,  5195,  5197,  5201,
      5233,  5234,  5584,  5817,  5823,  5832,  5875,  5951,  5968,  6033,
      6035,  6179,  6215,  6245,  6383,  6394,  6396,  6529,  6613,  6691,
      6801,  7091,  7128,  7155,  7240,  7461,  7551,  7596,  7691,  7738,
      7784,  8027,  8144,  8192,  8249,  8309,  8573,  8647,  8826,  9379,
      9593,  9767,  10400, 10461, 10530, 11028, 12799, 13787, 14487, 14670,
      15237, 15523, 20168, 25755};
  FixedVector<Float16, sparse_vec_count_1> sparse_vec_value_1{
      0.3815109,   0.21950184,   0.389138,    0.03037462,  0.738938,
      0.11151163,  0.21257511,   0.008723602, 0.42403504,  0.17748593,
      0.38613674,  0.38208488,   0.49048766,  0.056615792, 1.285813,
      1.1482359,   0.016783785,  0.7362169,   0.21784282,  1.0905122,
      0.37420613,  0.81915,      0.67411584,  0.35778007,  0.80538017,
      0.10094925,  1.2726786,    0.12334787,  0.18297458,  0.13315988,
      0.041079145, 0.2655652,    0.10946682,  0.6782494,   1.7451618,
      0.17126456,  0.17718226,   0.7430134,   0.9090848,   0.31985787,
      0.21779177,  0.13639484,   1.2293936,   0.065131165, 0.03718982,
      0.64121664,  0.46517274,   0.39498892,  0.07401267,  1.2061241,
      0.1276834,   0.059918232,  1.1935436,   0.61886644,  0.32731527,
      0.37830237,  1.0287925,    0.09565632,  0.4313508,   0.03845683,
      0.066990376, 0.10886483,   0.097683005, 0.29624575,  0.48645914,
      0.250733,    0.03274726,   1.205507,    0.048636433, 0.034002367,
      0.83021015,  0.044592205,  0.06007409,  1.1224703,   0.45620173,
      0.16457361,  0.053571727,  0.12527509,  0.1308366,   0.92323685,
      0.7821679,   0.23838642,   0.2558486,   0.09402168,  0.22815736,
      0.51750314,  0.08442147,   0.5565446,   0.3642559,   0.6661639,
      0.73750395,  0.17278494,   0.05865512,  0.013724559, 0.023783961,
      0.04283593,  0.24765956,   0.3991119,   1.5201892,   0.035530984,
      0.049782272, 0.06485597,   0.5367931,   0.15097857,  0.014405596,
      0.14585418,  0.22106051,   0.49575308,  0.08290891,  0.17875223,
      0.21095915,  0.0038430362, 2.3110201,   0.6543391,   0.06421487,
      0.3782336,   0.3514111,    0.5225064,   0.21472597,  0.07987356,
      0.06002587,  1.5242931,    0.081204355, 0.32025364,  0.39068836,
      0.027896391, 0.2872351,    0.50436527,  0.5434884,   1.653683,
      1.444315,    0.988968,     0.024239752, 0.055084217, 0.074782506,
      0.021114044, 0.07288233,   0.822755,    0.10772858,  0.6189507,
      0.29534152,  0.20032129,   0.5609191,   1.2844883};

  const uint32_t sparse_vec_count_2 = 153;
  uint32_t sparse_vec_index_2[] = {
      1012,  1059,  1996,  2001,  2020,  2049,  2052,  2055,  2056,  2081,
      2088,  2124,  2138,  2156,  2158,  2162,  2191,  2231,  2242,  2256,
      2259,  2311,  2318,  2359,  2373,  2381,  2390,  2437,  2458,  2466,
      2477,  2510,  2554,  2580,  2590,  2622,  2640,  2671,  2689,  2825,
      2844,  2881,  2904,  2957,  3029,  3112,  3125,  3144,  3214,  3246,
      3271,  3312,  3330,  3399,  3443,  3478,  3578,  3595,  3647,  3697,
      3740,  3800,  3817,  3818,  3928,  3934,  3987,  4034,  4072,  4079,
      4172,  4204,  4254,  4255,  4302,  4517,  4526,  4695,  4706,  4795,
      4807,  4986,  5081,  5091,  5113,  5195,  5197,  5234,  5253,  5263,
      5623,  5646,  5656,  5817,  5875,  5951,  5954,  5968,  6033,  6061,
      6108,  6119,  6157,  6213,  6215,  6287,  6384,  6396,  6461,  6469,
      6613,  6801,  6842,  7128,  7240,  7305,  7477,  7551,  7596,  7609,
      7624,  7723,  7779,  7857,  7935,  8144,  8238,  8249,  8275,  8547,
      8573,  8647,  8826,  8927,  9036,  9491,  9593,  9767,  10267, 10461,
      10505, 10660, 10721, 11028, 12578, 13787, 14487, 14874, 15523, 20168,
      21565, 24212, 25628};
  FixedVector<Float16, sparse_vec_count_2> sparse_vec_value_2{
      0.19194126,  0.11344757,   0.21317342,  0.6771587,    0.08591107,
      0.006228663, 0.28981656,   0.58056134,  0.064362876,  0.5794717,
      0.4288167,   0.59527594,   0.6106896,   0.23139843,   0.897008,
      0.20689227,  0.28713426,   0.38175523,  0.4028853,    0.08509491,
      1.0562526,   0.1165676,    0.06347306,  0.41331312,   0.16935593,
      0.1626863,   0.29352358,   0.45827967,  0.21193665,   0.39532298,
      0.0789344,   0.026420705,  0.1763078,   0.18424834,   0.7216729,
      1.6683924,   0.06257952,   0.13419773,  0.6851299,    1.2139059,
      0.092483185, 0.10803583,   0.74339646,  0.14461784,   0.2389669,
      0.9306581,   0.5645601,    0.83565444,  0.11930474,   0.22862941,
      0.6214566,   0.0033283439, 0.42018214,  0.15267797,   0.029068783,
      0.24103808,  0.18765616,   0.11574381,  0.31545344,   0.09386852,
      0.038362045, 0.7730324,    0.4456206,   0.20152733,   0.94718367,
      1.1934134,   0.12610391,   0.014013804, 0.47198555,   0.21791361,
      0.05394335,  0.08415188,   0.066486694, 0.47462225,   0.16693182,
      0.9021425,   0.27905586,   0.09939155,  0.12642553,   0.27529165,
      0.024804203, 0.24346212,   0.25561446,  1.4675297,    0.21566682,
      0.5453194,   0.21558505,   0.21294887,  0.2740208,    0.43185237,
      0.2280337,   0.0048945076, 0.26826337,  0.016979327,  0.3338952,
      0.23080347,  0.21200272,   1.3268396,   0.05323057,   0.30005422,
      0.088871606, 0.13259241,   0.04766706,  0.0017769856, 0.2698414,
      0.08068406,  0.38578644,   0.09752118,  0.13972333,   0.0731375,
      0.36664346,  0.12214721,   0.1541759,   2.2295072,    0.22542699,
      0.028530587, 0.022988612,  0.35836184,  0.10530607,   0.53756726,
      0.05818686,  0.044951066,  0.05753079,  0.09009998,   0.24644017,
      0.22693348,  0.0019512648, 0.035316195, 0.057344455,  0.36419895,
      0.1534858,   0.18924302,   0.38702026,  1.2569604,    0.07787755,
      1.7163913,   1.1903315,    0.8173934,   0.13888475,   0.10908335,
      0.35437793,  0.15787303,   0.25039884,  0.130508,     0.09830101,
      0.5841259,   0.22020355,   0.37849018,  0.14054261,   0.5179198,
      1.1891438,   0.44022372,   0.1794719};

  const uint32_t sparse_vec_count_3 = 166;
  uint32_t sparse_vec_index_3[] = {
      1012,  1059,  1996,  1997,  2001,  2020,  2034,  2076,  2086,  2104,
      2138,  2149,  2162,  2170,  2171,  2220,  2231,  2236,  2259,  2311,
      2315,  2318,  2328,  2343,  2344,  2359,  2381,  2390,  2419,  2458,
      2462,  2466,  2472,  2479,  2491,  2510,  2557,  2558,  2565,  2580,
      2590,  2622,  2724,  2764,  2817,  2837,  2881,  2900,  2911,  2933,
      2949,  3003,  3029,  3058,  3092,  3101,  3125,  3188,  3271,  3330,
      3386,  3399,  3434,  3447,  3474,  3478,  3578,  3595,  3607,  3650,
      3690,  3740,  3779,  3800,  3817,  3818,  3826,  3910,  3918,  3934,
      3987,  3992,  4006,  4034,  4068,  4075,  4114,  4146,  4172,  4255,
      4302,  4327,  4503,  4517,  4758,  4883,  4944,  4975,  5036,  5195,
      5205,  5218,  5233,  5234,  5253,  5456,  5623,  5656,  5687,  5817,
      5875,  5951,  5954,  5968,  6059,  6119,  6145,  6157,  6215,  6262,
      6384,  6394,  6613,  6787,  6801,  6842,  6993,  7128,  7156,  7240,
      7305,  7421,  7551,  7596,  7676,  7935,  8547,  8573,  8647,  8773,
      8826,  8886,  8911,  9036,  9274,  9433,  9593,  9767,  9915,  10267,
      10461, 10505, 11028, 11274, 11593, 13058, 13787, 14487, 15237, 17060,
      20168, 21695, 23041, 24363, 25526, 25755};
  FixedVector<Float16, sparse_vec_count_3> sparse_vec_value_3{
      0.17927244,   0.20557176,   0.40560228,   0.32370853,  0.8060634,
      0.21424179,   1.0674698,    0.6046889,    0.21051478,  0.46186206,
      0.24661283,   0.5616991,    1.016811,     0.2618776,   0.9686127,
      0.869671,     0.1458332,    0.60725594,   1.206012,    0.10357225,
      0.4350595,    0.83702874,   0.146196,     0.8644738,   0.15587087,
      0.16456357,   0.36376593,   1.053665,     0.06609649,  0.6504239,
      0.9697015,    0.04947369,   0.43753505,   0.04289205,  0.42075413,
      0.330524,     0.1743388,    0.6540892,    0.012900644, 0.23207273,
      0.2674499,    1.9736407,    0.21540764,   0.63648874,  0.049446102,
      0.3750183,    0.17441651,   0.123951435,  0.015306404, 0.1767618,
      0.24109434,   0.4245122,    0.114403255,  0.91849947,  0.12018716,
      0.01165807,   0.47680765,   0.036503244,  0.5782868,   0.9163635,
      0.27396393,   0.16385026,   0.052631885,  0.72294754,  0.4022935,
      0.06351255,   0.27786675,   0.25394455,   0.08041568,  1.3137422,
      0.5514297,    0.2503315,    0.009040705,  0.40985608,  0.27673048,
      0.14055687,   0.50529444,   0.6049716,    1.0692317,   1.207644,
      0.108388424,  0.9495853,    0.35366973,   0.3762234,   0.19875458,
      0.14685634,   0.0060924664, 1.0126622,    0.034943417, 0.49489433,
      0.34451365,   0.21992311,   0.7039926,    0.9501215,   0.34629604,
      0.20126931,   0.23908958,   0.019030606,  0.12528977,  0.6009518,
      0.056694727,  0.19225678,   0.61745095,   0.26769277,  0.18739952,
      0.10380342,   0.08536158,   0.18679029,   0.040631995, 0.23538794,
      0.081166975,  0.3206779,    0.0018739193, 1.5819491,   0.07052032,
      0.2504746,    0.7514167,    0.06575893,   0.08000714,  0.0012445971,
      0.23989597,   0.12001178,   0.51009554,   0.14469045,  0.12445986,
      0.08644873,   0.5645543,    2.539498,     0.54383165,  0.22437337,
      0.0018195114, 0.11787724,   0.34932667,   0.49611032,  0.24439196,
      0.100613214,  0.2844197,    0.38720158,   0.22204469,  0.078220785,
      0.76444066,   1.7794204,    0.17640579,   0.04227443,  0.28023362,
      0.06434563,   1.320367,     0.9287479,    0.14726646,  0.27983913,
      0.022449814,  0.09246922,   0.22375125,   0.10417365,  0.034148056,
      0.12830476,   0.6065902,    0.16593556,   0.25840235,  0.2596266,
      0.6388732,    1.6666834,    0.030998405,  0.14869562,  0.30502653,
      1.183558};

  const uint32_t sparse_vec_count_4 = 104;
  uint32_t sparse_vec_index_4[] = {
      1012,  1996,  1997,  2001,  2033,  2034,  2080,  2120,  2142,  2149,
      2220,  2231,  2259,  2284,  2318,  2338,  2381,  2405,  2424,  2436,
      2458,  2472,  2533,  2544,  2557,  2580,  2609,  2622,  2627,  2688,
      2800,  2820,  2837,  2862,  2932,  2949,  3029,  3036,  3181,  3390,
      3439,  3690,  3780,  3784,  3818,  3872,  3931,  3934,  4034,  4037,
      4075,  4219,  4348,  4517,  4573,  4617,  4773,  4809,  4822,  4879,
      5234,  5272,  5851,  5968,  6119,  6378,  6396,  6613,  6702,  6728,
      6787,  7128,  7156,  7240,  7479,  7551,  7596,  7692,  7809,  8027,
      8249,  8264,  8299,  8573,  8826,  9123,  9152,  9274,  9445,  9593,
      9915,  11377, 11744, 12935, 13308, 14487, 14947, 15720, 17060, 17669,
      18079, 18629, 19841, 21053};
  FixedVector<Float16, sparse_vec_count_4> sparse_vec_value_4{
      0.2030336,   0.1411735,   0.12635018,  0.45823106,  0.22794029,
      1.4105916,   0.2769118,   0.75515395,  0.07748295,  0.19260094,
      0.12458416,  0.065163694, 0.9765741,   0.07470863,  0.80718166,
      0.12307288,  0.9393725,   0.048733678, 0.17115222,  1.1922649,
      0.03547645,  0.33111426,  0.03772038,  0.46104532,  0.3141086,
      0.25707254,  1.1549219,   1.8509476,   0.98180383,  0.7270674,
      0.91343564,  0.3373339,   0.081498206, 0.01140901,  0.43917242,
      0.072401166, 0.11307132,  0.8945273,   0.10071963,  0.1945517,
      0.7594797,   0.096463405, 0.07759007,  0.11009286,  0.012562437,
      1.1797432,   0.02481144,  1.2393609,   0.50596905,  1.48781,
      0.53125334,  0.9950063,   1.4128636,   1.5830894,   0.93246186,
      0.60709685,  0.40433922,  0.14255294,  0.7125986,   0.021445543,
      0.4104336,   0.14560317,  0.3189296,   0.51019174,  0.041676614,
      0.22844397,  0.18406813,  0.1604107,   1.2178165,   0.46861333,
      0.04899898,  2.4448788,   0.6505235,   0.051029652, 0.7550255,
      0.00625443,  0.5090246,   0.7109037,   0.1125403,   0.05059699,
      0.03856528,  0.4538238,   0.72464395,  0.1360473,   0.5109412,
      2.0780752,   0.049649376, 0.31396037,  0.114775784, 0.9717559,
      0.05478335,  0.12228666,  1.3433831,   1.6574994,   0.053257514,
      0.51201975,  0.029570522, 0.35752434,  0.39366165,  0.25994724,
      1.1072603,   2.0454218,   1.1423918,   0.59795356};

  const uint32_t sparse_vec_count_5 = 147;
  uint32_t sparse_vec_index_5[] = {
      1012,  1996,  2001,  2018,  2020,  2034,  2047,  2081,  2154,  2162,
      2170,  2171,  2207,  2210,  2220,  2233,  2251,  2253,  2257,  2259,
      2287,  2315,  2318,  2328,  2381,  2390,  2458,  2466,  2510,  2557,
      2580,  2609,  2622,  2645,  2688,  2707,  2724,  2762,  2838,  2900,
      2911,  2915,  3047,  3058,  3260,  3282,  3290,  3295,  3297,  3386,
      3390,  3578,  3603,  3607,  3690,  3746,  3826,  3861,  3908,  3910,
      3918,  3934,  3987,  4006,  4045,  4075,  4088,  4110,  4255,  4302,
      4517,  4620,  4761,  4871,  4916,  5195,  5221,  5234,  5246,  5532,
      5700,  5798,  5832,  5855,  5951,  5968,  6033,  6215,  6219,  6302,
      6394,  6396,  6529,  6950,  7008,  7084,  7128,  7155,  7156,  7240,
      7421,  7467,  7551,  7596,  7738,  7760,  8088,  8367,  8372,  8479,
      8573,  8647,  8773,  8826,  9188,  9274,  9290,  9433,  9593,  9767,
      9913,  9919,  9982,  10461, 10815, 11028, 11721, 12416, 12496, 12779,
      13221, 13702, 13787, 14487, 15699, 16164, 18801, 20168, 21650, 24291,
      24321, 25209, 25526, 25755, 28110, 28682, 28858};
  FixedVector<Float16, sparse_vec_count_5> sparse_vec_value_5{
      0.22246745,  0.1639393,    0.6902539,    0.087209724, 0.3150326,
      1.3589038,   0.39210027,   0.06905281,   0.2940129,   0.48745865,
      0.5185849,   0.06468885,   0.33793828,   0.01934533,  0.9160348,
      0.12213709,  0.64625627,   0.05484681,   0.18600157,  0.7439921,
      1.4779477,   0.50866294,   0.9324953,    0.11494038,  0.14815839,
      0.4024814,   0.0025193223, 0.0039419075, 0.04004241,  0.1137441,
      0.100572474, 0.09889997,   1.6465691,    0.45031455,  0.4567774,
      0.7614913,   0.5324026,    0.09957147,   0.21556115,  0.36752453,
      0.13450043,  0.06911261,   0.04267344,   1.2791942,   0.054822505,
      0.06269096,  1.3170663,    0.8852742,    0.37885663,  0.92810893,
      0.12803665,  0.10517517,   0.24920024,   0.16889784,  1.3619378,
      0.59796244,  0.81389725,   0.06489252,   0.020069994, 0.06319,
      0.71297073,  1.2515233,    0.019061586,  0.04731544,  0.3536146,
      0.50835687,  0.56439734,   0.09884678,   1.1007178,   0.1480219,
      1.6361246,   0.3891063,    0.03873499,   0.050479025, 0.5629584,
      1.0016122,   0.16247666,   0.06476003,   0.43833405,  1.3702114,
      0.11968183,  0.29155007,   0.12643526,   0.518913,    0.41796717,
      1.740134,    0.015489911,  0.2183447,    1.5380116,   1.058654,
      0.06226158,  0.270943,     0.91666347,   0.06422295,  0.33474496,
      0.002399514, 2.0762439,    0.8989307,    0.7876583,   0.03783609,
      0.22333156,  0.13323776,   0.27660817,   0.56637865,  0.21507333,
      0.6770579,   0.7013793,    0.7085848,    0.15651116,  0.05219105,
      0.03743524,  0.30775747,   0.073243596,  0.8181374,   0.28133482,
      0.23539418,  0.07533616,   0.2044144,    1.574523,    1.1304078,
      0.24084339,  1.3286508,    0.775562,     0.10096621,  0.197577,
      0.2307252,   1.719028,     0.07254901,   0.13916898,  0.17486195,
      0.8424586,   0.27879223,   0.8650824,    0.35050592,  0.24243252,
      0.31039444,  0.17227773,   0.90619636,   0.63083464,  2.2181685,
      0.20995331,  0.14425081,   0.37305146,   0.5955121,   0.87200415,
      1.028527,    1.0835907};

  const uint32_t sparse_vec_count_6 = 141;
  uint32_t sparse_vec_index_6[] = {
      1012,  1059,  1996,  1997,  1998,  2001,  2012,  2018,  2020,  2021,
      2025,  2055,  2056,  2076,  2077,  2127,  2130,  2134,  2138,  2143,
      2162,  2197,  2203,  2220,  2259,  2318,  2328,  2338,  2345,  2381,
      2390,  2458,  2462,  2466,  2501,  2517,  2580,  2622,  2631,  2645,
      2688,  2707,  2724,  2748,  2764,  2808,  2900,  2911,  2933,  2949,
      3047,  3058,  3074,  3075,  3092,  3101,  3188,  3271,  3283,  3439,
      3478,  3535,  3595,  3607,  3690,  3720,  3740,  3793,  3818,  3826,
      3906,  3908,  3934,  3981,  3986,  4028,  4138,  4469,  4496,  4503,
      4515,  4517,  4566,  4704,  4706,  4761,  4839,  5036,  5175,  5233,
      5234,  5246,  5254,  5263,  5491,  5817,  5823,  5839,  5875,  5968,
      6215,  6254,  6268,  6394,  6407,  6801,  6848,  7128,  7177,  7321,
      7421,  7487,  7551,  7596,  7681,  7940,  8145,  8264,  8321,  8551,
      8573,  8647,  8773,  8826,  8832,  9472,  9593,  9599,  9767,  10530,
      12149, 13787, 14487, 15237, 15523, 17060, 20168, 23633, 24363, 25526,
      25755};
  FixedVector<Float16, sparse_vec_count_6> sparse_vec_value_6{
      0.48692977,  0.23770119,  0.24359323,   0.030566106,  0.121271,
      0.5703241,   0.12787338,  0.037069157,  0.075816214,  0.05305081,
      0.45591223,  0.5893366,   0.01829792,   0.42078727,   0.036012013,
      0.0750098,   0.20031127,  0.033489488,  0.10935432,   0.054307006,
      1.0000131,   0.20630358,  1.1161063,    0.5766484,    0.86030954,
      0.65358734,  0.062234607, 0.8518808,    0.23441537,   0.14816457,
      0.19284223,  0.94708407,  1.0017378,    0.51629704,   0.082293354,
      0.09170858,  0.2138309,   1.533815,     0.0030641577, 0.029126635,
      0.3632337,   0.1761491,   0.34924436,   0.67822266,   0.5976219,
      0.8595736,   0.17943758,  0.038340267,  0.0052374,    0.29047492,
      0.070157826, 0.6779024,   0.75593567,   0.054473646,  0.4906121,
      0.11288958,  0.15934071,  0.3192689,    0.1435216,    0.30725288,
      0.37506026,  0.7213243,   0.18401349,   0.01871983,   0.19455475,
      0.02040177,  0.28111485,  0.043639474,  0.19826981,   0.27416018,
      1.429636,    0.05111553,  1.0482118,    0.98164123,   0.17426124,
      0.10582682,  1.002954,    1.0261939,    0.83377177,   0.6798103,
      0.015373114, 0.8136259,   0.95782644,   0.13387722,   0.40847424,
      0.80647326,  0.28733957,  0.0029352994, 0.30276307,   0.4768307,
      0.32016084,  0.10302183,  0.3044403,    0.040031943,  0.44271877,
      0.061298616, 0.08278493,  0.107188344,  0.5086274,    1.3297924,
      0.050804485, 0.68582493,  0.21776867,   0.027724598,  0.5286007,
      0.1899133,   0.04971613,  2.2401748,    0.09252626,   0.80688274,
      0.014750206, 0.07568165,  0.021886598,  0.23429997,   1.1812011,
      0.6390751,   0.2643012,   0.13720371,   0.10989579,   1.4969206,
      0.2209742,   0.54690766,  0.15685914,   0.47841135,   0.566988,
      0.08368683,  1.2788389,   0.09509155,   1.0241207,    0.07167757,
      0.29240122,  0.5619141,   0.016415644,  0.28731114,   0.035925347,
      0.34043407,  0.60646313,  0.07248792,   0.08602479,   0.10247773,
      1.13258};

  const uint32_t sparse_vec_count_7 = 221;
  uint32_t sparse_vec_index_7[] = {
      1059,  1996,  2001,  2003,  2008,  2010,  2020,  2029,  2034,  2076,
      2080,  2081,  2103,  2104,  2137,  2138,  2142,  2149,  2162,  2163,
      2220,  2231,  2236,  2253,  2256,  2259,  2315,  2318,  2328,  2329,
      2343,  2344,  2350,  2359,  2381,  2390,  2419,  2458,  2462,  2466,
      2470,  2472,  2490,  2510,  2537,  2550,  2554,  2557,  2580,  2590,
      2599,  2608,  2622,  2631,  2640,  2645,  2662,  2710,  2724,  2728,
      2762,  2764,  2817,  2820,  2832,  2837,  2856,  2866,  2881,  2891,
      2957,  2974,  2983,  3003,  3010,  3029,  3050,  3058,  3063,  3068,
      3092,  3101,  3125,  3135,  3257,  3271,  3282,  3330,  3386,  3399,
      3474,  3578,  3595,  3603,  3607,  3650,  3690,  3758,  3800,  3817,
      3826,  3878,  3910,  3918,  3934,  3947,  3965,  3987,  3992,  4006,
      4034,  4045,  4068,  4146,  4172,  4202,  4255,  4302,  4327,  4351,
      4503,  4517,  4637,  4707,  4944,  5025,  5036,  5195,  5201,  5233,
      5234,  5253,  5501,  5584,  5623,  5656,  5687,  5814,  5817,  5911,
      5951,  5954,  5968,  6035,  6108,  6119,  6145,  6157,  6177,  6215,
      6254,  6262,  6384,  6394,  6613,  6728,  6787,  6801,  6842,  6845,
      6922,  6960,  7128,  7155,  7156,  7240,  7421,  7551,  7596,  7609,
      7654,  7676,  7723,  7779,  7935,  8049,  8144,  8151,  8249,  8547,
      8573,  8647,  8773,  8826,  8864,  8886,  9036,  9274,  9290,  9433,
      9593,  9667,  9767,  9915,  10267, 10505, 10544, 10753, 10815, 11028,
      11593, 11837, 12496, 13058, 13308, 13625, 13702, 14487, 15523, 17669,
      18457, 18800, 18826, 20168, 20843, 21695, 24363, 25526, 25755, 26234,
      26911};
  FixedVector<Float16, sparse_vec_count_7> sparse_vec_value_7{
      0.29634815,  0.3303992,    1.0099697,   0.09545747,  0.046319153,
      0.001999375, 0.27222815,   0.107896015, 1.0792782,   0.5411261,
      0.27695096,  0.020715078,  0.021571944, 0.61097443,  0.10560424,
      0.15401895,  0.46480918,   0.6496758,   1.0116925,   0.0040072273,
      0.8931394,   0.2361543,    0.74389607,  0.039703716, 0.020886008,
      1.1108406,   0.09039394,   0.69578373,  0.27737862,  0.3083219,
      0.5698159,   0.31437457,   0.7131746,   0.14947455,  0.33504876,
      1.1611847,   0.8632542,    1.058698,    1.0307701,   0.15223494,
      0.9391413,   0.9473978,    0.3767169,   0.5806728,   0.70086235,
      0.8544429,   0.07839825,   0.46189323,  0.57343185,  0.17151174,
      0.45118546,  0.03416668,   2.037371,    0.1311739,   0.22600843,
      0.061421365, 0.0063685803, 0.9023181,   0.17874505,  1.458104,
      0.09657643,  0.36346155,   0.11396522,  0.2762966,   0.11472289,
      0.16151813,  0.5954224,    0.68847394,  0.6934064,   1.0951325,
      0.008113728, 0.320056,     0.2934685,   0.38948777,  0.64446163,
      0.11539491,  1.4196212,    0.6417532,   0.10939098,  0.115132414,
      0.10055387,  0.15150718,   0.3015885,   0.36512154,  0.85847276,
      0.42005107,  0.06733843,   0.9194887,   0.2446694,   0.3528377,
      0.30540454,  0.0549386,    0.15950806,  0.12754358,  0.22250807,
      1.3793756,   0.01503605,   0.33390692,  0.2052875,   0.32573462,
      0.66194123,  0.03896839,   0.921685,    1.1364039,   1.2451752,
      0.072772495, 0.10148866,   0.2922106,   0.97420144,  0.25800666,
      0.13455145,  0.3459612,    0.16713561,  0.21625288,  0.20754638,
      0.017042752, 1.2139128,    0.38501504,  0.18923776,  0.58807755,
      0.42623222,  1.8636363,    0.15489826,  0.24531981,  0.330716,
      0.6148099,   0.12145276,   0.938947,    0.08298498,  0.5002425,
      0.42643633,  0.3724926,    0.351435,    0.35051146,  0.15093777,
      0.2753887,   0.11030835,   0.05864477,  0.12825343,  0.4938676,
      0.4091608,   0.13155867,   1.362572,    0.26034647,  0.005735014,
      0.25208464,  0.77931124,   0.08418636,  0.2567355,   0.108983725,
      0.04566572,  0.06202907,   0.3991703,   0.2785334,   0.45871663,
      1.584949,    0.099409536,  0.114265166, 0.0603091,   0.71120745,
      0.35286796,  0.03805246,   2.6303916,   0.6235311,   0.6544235,
      0.254192,    0.5172861,    0.46474016,  0.51770395,  0.3868696,
      0.030558605, 0.79667675,   0.1053426,   0.08400551,  0.26797673,
      0.52138245,  0.13453461,   0.070371106, 0.003556521, 0.34309983,
      0.2104394,   0.02274147,   0.19070747,  0.9488226,   0.09138845,
      2.092856,    0.10931594,   0.18929166,  0.113100395, 0.08495193,
      1.124685,    0.08020554,   1.0792019,   0.27422333,  0.31508496,
      0.20671548,  0.05064338,   0.46511328,  0.38314936,  0.52556884,
      0.36894837,  1.4199936,    0.05843645,  0.055732273, 0.26817194,
      0.2876586,   1.0425944,    0.062882155, 0.09840146,  0.1544766,
      0.98742366,  0.20589906,   2.1226256,   0.47266316,  0.33193296,
      2.0077822,   0.23509863,   0.53764015,  1.2505449,   1.719803,
      0.39262286};

  std::vector<std::string> sparse_query_buffers;

  std::string sparse_query_buffer_0;
  MinusInnerProductSparseMatrix<Float16>::transform_sparse_format(
      sparse_vec_count_0, sparse_vec_index_0, sparse_vec_value_0.data(),
      sparse_query_buffer_0);
  sparse_query_buffers.emplace_back(std::move(sparse_query_buffer_0));

  std::string sparse_query_buffer_1;
  MinusInnerProductSparseMatrix<Float16>::transform_sparse_format(
      sparse_vec_count_1, sparse_vec_index_1, sparse_vec_value_1.data(),
      sparse_query_buffer_1);
  sparse_query_buffers.emplace_back(std::move(sparse_query_buffer_1));

  std::string sparse_query_buffer_2;
  MinusInnerProductSparseMatrix<Float16>::transform_sparse_format(
      sparse_vec_count_2, sparse_vec_index_2, sparse_vec_value_2.data(),
      sparse_query_buffer_2);
  sparse_query_buffers.emplace_back(std::move(sparse_query_buffer_2));

  std::string sparse_query_buffer_3;
  MinusInnerProductSparseMatrix<Float16>::transform_sparse_format(
      sparse_vec_count_3, sparse_vec_index_3, sparse_vec_value_3.data(),
      sparse_query_buffer_3);
  sparse_query_buffers.emplace_back(std::move(sparse_query_buffer_3));

  std::string sparse_query_buffer_4;
  MinusInnerProductSparseMatrix<Float16>::transform_sparse_format(
      sparse_vec_count_4, sparse_vec_index_4, sparse_vec_value_4.data(),
      sparse_query_buffer_4);
  sparse_query_buffers.emplace_back(std::move(sparse_query_buffer_4));

  std::string sparse_query_buffer_5;
  MinusInnerProductSparseMatrix<Float16>::transform_sparse_format(
      sparse_vec_count_5, sparse_vec_index_5, sparse_vec_value_5.data(),
      sparse_query_buffer_5);
  sparse_query_buffers.emplace_back(std::move(sparse_query_buffer_5));

  std::string sparse_query_buffer_6;
  MinusInnerProductSparseMatrix<Float16>::transform_sparse_format(
      sparse_vec_count_6, sparse_vec_index_6, sparse_vec_value_6.data(),
      sparse_query_buffer_6);
  sparse_query_buffers.emplace_back(std::move(sparse_query_buffer_6));

  std::string sparse_query_buffer_7;
  MinusInnerProductSparseMatrix<Float16>::transform_sparse_format(
      sparse_vec_count_7, sparse_vec_index_7, sparse_vec_value_7.data(),
      sparse_query_buffer_7);
  sparse_query_buffers.emplace_back(std::move(sparse_query_buffer_7));

  sparse_vec_counts.emplace_back(sparse_vec_count_0);
  sparse_vec_counts.emplace_back(sparse_vec_count_1);
  sparse_vec_counts.emplace_back(sparse_vec_count_2);
  sparse_vec_counts.emplace_back(sparse_vec_count_3);
  sparse_vec_counts.emplace_back(sparse_vec_count_4);
  sparse_vec_counts.emplace_back(sparse_vec_count_5);
  sparse_vec_counts.emplace_back(sparse_vec_count_6);
  sparse_vec_counts.emplace_back(sparse_vec_count_7);

  sparse_vec_indices.emplace_back(sparse_vec_index_0);
  sparse_vec_indices.emplace_back(sparse_vec_index_1);
  sparse_vec_indices.emplace_back(sparse_vec_index_2);
  sparse_vec_indices.emplace_back(sparse_vec_index_3);
  sparse_vec_indices.emplace_back(sparse_vec_index_4);
  sparse_vec_indices.emplace_back(sparse_vec_index_5);
  sparse_vec_indices.emplace_back(sparse_vec_index_6);
  sparse_vec_indices.emplace_back(sparse_vec_index_7);

  sparse_vec_values.emplace_back(sparse_vec_value_0.data());
  sparse_vec_values.emplace_back(sparse_vec_value_1.data());
  sparse_vec_values.emplace_back(sparse_vec_value_2.data());
  sparse_vec_values.emplace_back(sparse_vec_value_3.data());
  sparse_vec_values.emplace_back(sparse_vec_value_4.data());
  sparse_vec_values.emplace_back(sparse_vec_value_5.data());
  sparse_vec_values.emplace_back(sparse_vec_value_6.data());
  sparse_vec_values.emplace_back(sparse_vec_value_7.data());

  for (size_t i = 0; i < sparse_query_buffers.size(); ++i) {
    for (size_t j = 0; j < sparse_query_buffers.size(); ++j) {
      float result0{0.0f};
      result0 = SparseDistanceCommon(
          sparse_vec_counts[i], sparse_vec_indices[i], sparse_vec_values[i],
          sparse_vec_counts[j], sparse_vec_indices[j], sparse_vec_values[j]);

      float result1{0.0f};
      MinusInnerProductSparseMatrix<Float16>::Compute(
          sparse_query_buffers[i].data(), sparse_query_buffers[j].data(),
          &result1);
      result1 = -result1;

      float epsilon = 0.001 * std::max(result0, result1);
      EXPECT_GE(epsilon, std::abs(result0 - result1));
    }
  }
}

TEST(DistanceMatrix, InnerProductSparse) {
  TestInnerProductSparse();
}

TEST(DistanceMatrix, InnerProductSparseMore) {
  TestInnerProductSparseMore();
}

TEST(DistanceMatrix, DISABLED_InnerProductSparse_Benchmark) {
  const uint32_t sparse_vec_count_0 = 52;
  uint32_t sparse_vec_index_0[] = {
      33,   66,   77,   209,  385,  396,  539,  583,  649,  715,  880,
      935,  968,  1023, 1100, 1111, 1661, 1694, 1749, 2288, 2343, 2453,
      2530, 2772, 2871, 2882, 2948, 3069, 3322, 3333, 3410, 3575, 3608,
      4026, 4037, 4048, 4059, 4070, 4268, 4323, 4741, 4752, 5137, 5170,
      5423, 5555, 5918, 6028, 6094, 6347, 6369, 6468};
  FixedVector<Float16, sparse_vec_count_0> sparse_vec_value_0{
      -0.246404298254, 0.892043114755,  0.163785949199,  -0.680309913534,
      -0.767956138324, -0.410683610330, 0.763314047145,  0.347851184825,
      -0.676969102165, -0.774662820732, 0.274471489215,  -0.131269040962,
      0.206478593023,  0.764082612827,  -0.57678381864,  -0.256053693585,
      0.661507236032,  -0.812832823664, 0.929611593685,  -0.381852499144,
      -0.35890001953,  0.538386710846,  -0.829565442015, 0.384046166409,
      0.623125501212,  0.043215334982,  -0.689536097425, -0.500913794456,
      -0.419818105671, -0.503346955801, -0.99419236655,  -0.414091535679,
      -0.829474457209, -0.103915702521, -0.419445202934, -0.26891898936,
      0.311013521629,  0.172923023003,  -0.818231467063, -0.728015315042,
      0.110116365075,  0.845786117564,  -0.587841450807, 0.533763235805,
      -0.601437402994, -0.117487602176, 0.106103380748,  -0.00151542886833,
      0.189967593506,  0.890365538566,  -0.581876671583, -0.232173604777};

  const uint32_t sparse_vec_count_1 = 43;
  uint32_t sparse_vec_index_1[] = {
      33,   77,   110,  209,  1023, 1111, 1221, 1496, 1661, 1749, 2189,
      2255, 2288, 2420, 2530, 2695, 2772, 2838, 2948, 3179, 3575, 4202,
      4268, 4290, 4433, 4444, 4653, 4697, 4741, 5137, 5192, 5346, 5423,
      5445, 5555, 5588, 5764, 5896, 5918, 6028, 6270, 6347, 6501};
  FixedVector<Float16, sparse_vec_count_1> sparse_vec_value_1{
      -0.847561468192, -0.761580890729,  0.683791378502,  0.729670644228,
      -0.111989702001, -0.3435914518,    -0.806454864134, -0.0243347460596,
      0.497209110076,  0.852745969955,   0.403748558594,  -0.634016410599,
      -0.74513226711,  0.738086689871,   0.364575651925,  0.0867637408004,
      -0.285921174394, -0.321390976616,  -0.971849760722, -0.246041408731,
      -0.110667223833, 0.0744013655781,  0.84846334839,   0.167405689007,
      0.0289923642993, -0.536394124155,  0.518249809298,  -0.695798108647,
      0.0653215071151, -0.0046338401448, 0.644189056747,  -0.52301532328,
      -0.660275328421, 0.643514995264,   0.0333307952838, -0.401825159735,
      -0.188869041499, -0.23065238799,   -0.409416817144, -0.142933941372,
      0.247628793044,  -0.984985692607,  -0.427929860028};

  std::string sparse_query_buffer_0;
  MinusInnerProductSparseMatrix<Float16>::transform_sparse_format(
      sparse_vec_count_0, sparse_vec_index_0, sparse_vec_value_0.data(),
      sparse_query_buffer_0);

  std::string sparse_query_buffer_1;
  MinusInnerProductSparseMatrix<Float16>::transform_sparse_format(
      sparse_vec_count_1, sparse_vec_index_1, sparse_vec_value_1.data(),
      sparse_query_buffer_1);

  size_t loop_cnt = 100000000LLU;
  float result[100];

  for (size_t i = 0; i < loop_cnt; ++i) {
    MinusInnerProductSparseMatrix<Float16>::Compute(
        sparse_query_buffer_0.data(), sparse_query_buffer_1.data(),
        result + (i % 100));
  }

  EXPECT_EQ(result[0], result[1]);
}

TEST(DistanceMatrix, TestInnerProductSparseDimWithZero) {
  // test 1
  const uint32_t sparse_vec_count_0 = 10;
  uint32_t sparse_vec_index_0[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  FixedVector<Float16, sparse_vec_count_0> sparse_vec_value_0{
      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};

  const uint32_t sparse_vec_count_1 = 10;
  uint32_t sparse_vec_index_1[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  FixedVector<Float16, sparse_vec_count_1> sparse_vec_value_1{
      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};

  std::string sparse_query_buffer_0;
  MinusInnerProductSparseMatrix<Float16>::transform_sparse_format(
      sparse_vec_count_0, sparse_vec_index_0, sparse_vec_value_0.data(),
      sparse_query_buffer_0);

  std::string sparse_query_buffer_1;
  MinusInnerProductSparseMatrix<Float16>::transform_sparse_format(
      sparse_vec_count_1, sparse_vec_index_1, sparse_vec_value_1.data(),
      sparse_query_buffer_1);

  float result0{0.0f};
  result0 = SparseDistanceCommon(sparse_vec_count_0, sparse_vec_index_0,
                                 sparse_vec_value_0.data(), sparse_vec_count_1,
                                 sparse_vec_index_1, sparse_vec_value_1.data());

  float result1{0.0f};
  MinusInnerProductSparseMatrix<Float16>::Compute(
      sparse_query_buffer_0.data(), sparse_query_buffer_1.data(), &result1);
  result1 = -result1;

  EXPECT_GE(0.00001, std::abs(result0 - result1));
}
