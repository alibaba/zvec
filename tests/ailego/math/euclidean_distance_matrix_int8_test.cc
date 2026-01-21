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

static inline void MatrixTranspose(uint32_t *dst, const uint32_t *src, size_t M,
                                   size_t N) {
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < M; ++j) {
      dst[j * N + i] = src[i * M + j];
    }
  }
}

template <size_t N>
static float EuclideanDistance(const FixedVector<int8_t, N> &lhs,
                               const FixedVector<int8_t, N> &rhs) {
  return Distance::Euclidean(lhs.data(), rhs.data(), lhs.size());
}

template <size_t N>
static float SquaredEuclideanDistance(const FixedVector<int8_t, N> &lhs,
                                      const FixedVector<int8_t, N> &rhs) {
  return Distance::SquaredEuclidean(lhs.data(), rhs.data(), lhs.size());
}

TEST(DistanceMatrix, Euclidean_General) {
  FixedVector<int8_t, 1> a{(int8_t)0}, b{(int8_t)0};
  EXPECT_FLOAT_EQ(0.0f, EuclideanDistance(a, b));

  FixedVector<int8_t, 3> c{(int8_t)1, (int8_t)2, (int8_t)3},
      d{(int8_t)2, (int8_t)4, (int8_t)6};
  EXPECT_FLOAT_EQ(3.741657f, EuclideanDistance(c, d));

  FixedVector<int8_t, 4> e{(int8_t)0, (int8_t)0, (int8_t)127, (int8_t)127},
      f{(int8_t)127, (int8_t)127, (int8_t)0, (int8_t)0};
  EXPECT_FLOAT_EQ(254.0f, EuclideanDistance(e, f));

  FixedVector<int8_t, 5> g{(int8_t)0, (int8_t)0, (int8_t)127, (int8_t)127,
                           (int8_t)-127},
      h{(int8_t)127, (int8_t)127, (int8_t)0, (int8_t)0, (int8_t)127};
  EXPECT_FLOAT_EQ(359.21024f, EuclideanDistance(g, h));

  int8_t a2048[] = {
      59, 46, 36, 99, 49, 61, 45, 68, 79, 86, 2,  8,  73, 14, 45, 85, 5,  63,
      71, 45, 36, 54, 72, 18, 79, 15, 78, 29, 9,  12, 96, 27, 3,  45, 81, 37,
      51, 19, 43, 5,  55, 93, 85, 61, 86, 54, 2,  33, 39, 74, 95, 7,  98, 32,
      25, 30, 83, 45, 2,  7,  76, 95, 58, 52, 80, 85, 71, 56, 92, 41, 67, 98,
      32, 97, 3,  71, 59, 58, 87, 84, 44, 54, 81, 74, 0,  51, 7,  28, 81, 83,
      45, 88, 95, 87, 31, 65, 61, 84, 37, 13, 98, 59, 35, 41, 40, 12, 36, 87,
      4,  84, 15, 96, 97, 15, 19, 7,  67, 87, 13, 40, 56, 80, 86, 3,  85, 99,
      56, 94, 8,  63, 69, 24, 72, 44, 75, 58, 93, 19, 63, 81, 59, 90, 54, 99,
      2,  37, 20, 72, 57, 56, 25, 78, 27, 83, 77, 9,  66, 66, 62, 21, 81, 69,
      8,  13, 29, 95, 8,  75, 20, 48, 76, 53, 5,  97, 8,  26, 93, 76, 63, 48,
      26, 51, 69, 46, 6,  42, 76, 44, 84, 40, 85, 79, 44, 62, 78, 52, 18, 70,
      95, 9,  13, 71, 15, 2,  28, 98, 98, 44, 98, 64, 44, 9,  17, 71, 27, 73,
      24, 54, 24, 64, 68, 38, 90, 20, 89, 4,  79, 20, 56, 33, 92, 65, 64, 83,
      33, 92, 24, 4,  51, 16, 76, 14, 72, 36, 95, 22, 32, 27, 42, 58, 15, 87,
      23, 19, 76, 2,  35, 41, 1,  18, 77, 48, 51, 50, 14, 14, 22, 80, 23, 39,
      53, 69, 61, 63, 45, 91, 67, 75, 51, 9,  40, 42, 16, 3,  61, 18, 28, 58,
      28, 13, 79, 23, 43, 40, 99, 87, 63, 63, 72, 74, 74, 93, 10, 61, 86, 84,
      8,  63, 31, 98, 18, 79, 54, 25, 47, 61, 98, 15, 42, 53, 26, 40, 59, 77,
      80, 62, 73, 53, 22, 21, 6,  38, 31, 3,  80, 91, 53, 77, 36, 25, 28, 64,
      60, 31, 49, 91, 6,  50, 70, 94, 36, 66, 29, 32, 66, 64, 92, 78, 30, 87,
      29, 26, 16, 87, 29, 64, 13, 60, 1,  63, 2,  75, 31, 44, 8,  3,  65, 50,
      48, 26, 94, 44, 7,  45, 9,  94, 56, 57, 25, 95, 5,  8,  92, 71, 10, 83,
      62, 57, 74, 3,  95, 38, 90, 99, 53, 40, 37, 12, 88, 60, 78, 61, 54, 60,
      73, 73, 18, 48, 75, 42, 30, 19, 18, 96, 44, 62, 80, 46, 4,  63, 98, 12,
      44, 34, 83, 42, 36, 95, 84, 2,  48, 7,  68, 68, 47, 71, 74, 93, 78, 5,
      83, 99, 33, 17, 81, 49, 8,  75, 24, 67, 26, 44, 89, 38, 99, 22, 45, 52,
      89, 14, 94, 74, 35, 24, 46, 63, 6,  55, 23, 3,  91, 97, 6,  33, 53, 76,
      13, 68, 35, 9,  24, 43, 86, 5,  24, 4,  64, 86, 85, 12, 24, 60, 2,  9,
      23, 48, 12, 88, 71, 28, 1,  79, 31, 31, 82, 21, 49, 8,  31, 3,  99, 91,
      93, 29, 15, 67, 9,  50, 94, 46, 72, 77, 33, 30, 4,  16, 56, 86, 62, 76,
      78, 77, 77, 25, 30, 88, 29, 17, 38, 2,  80, 86, 73, 8,  8,  38, 3,  50,
      67, 49, 42, 75, 49, 96, 36, 77, 55, 27, 52, 77, 82, 92, 42, 52, 35, 55,
      60, 16, 14, 96, 90, 39, 49, 68, 47, 36, 53, 1,  78, 43, 93, 58, 66, 45,
      80, 56, 51, 23, 64, 49, 56, 28, 77, 99, 18, 21, 98, 46, 88, 97, 98, 21,
      58, 31, 33, 77, 61, 71, 37, 80, 80, 18, 2,  92, 70, 83, 3,  41, 93, 38,
      24, 92, 98, 69, 80, 84, 81, 74, 80, 89, 74, 90, 92, 81, 29, 54, 3,  49,
      52, 24, 78, 99, 33, 35, 54, 98, 36, 90, 66, 71, 67, 39, 79, 55, 68, 68,
      94, 58, 60, 74, 81, 26, 43, 50, 67, 21, 27, 41, 44, 85, 60, 39, 48, 6,
      49, 75, 17, 1,  94, 69, 98, 38, 77, 96, 61, 99, 52, 89, 81, 0,  73, 30,
      68, 90, 4,  77, 93, 16, 56, 92, 58, 49, 22, 49, 60, 4,  66, 33, 54, 49,
      4,  3,  75, 81, 36, 99, 7,  87, 60, 15, 90, 93, 33, 42, 21, 42, 1,  29,
      72, 20, 94, 3,  83, 56, 48, 94, 41, 78, 84, 98, 61, 70, 71, 10, 15, 75,
      86, 6,  57, 53, 7,  38, 94, 2,  94, 38, 79, 21, 53, 69, 89, 84, 53, 59,
      99, 69, 81, 84, 60, 27, 1,  10, 93, 0,  88, 4,  71, 44, 12, 35, 63, 60,
      65, 15, 5,  20, 60, 30, 86, 46, 50, 28, 67, 39, 1,  13, 26, 40, 57, 59,
      87, 46, 5,  51, 14, 62, 78, 92, 11, 42, 61, 59, 4,  70, 24, 54, 34, 54,
      30, 83, 58, 61, 23, 13, 17, 96, 32, 39, 85, 81, 70, 53, 53, 23, 96, 36,
      18, 84, 22, 96, 5,  90, 17, 62, 25, 91, 71, 70, 91, 88, 93, 86, 30, 90,
      25, 79, 86, 12, 28, 87, 46, 64, 82, 19, 82, 91, 79, 40, 83, 38, 83, 86,
      50, 84, 67, 0,  41, 80, 70, 36, 36, 41, 81, 68, 71, 40, 36, 10, 60, 7,
      87, 67, 41, 10, 26, 98, 45, 66, 92, 22, 63, 70, 91, 90, 34, 98, 31, 45,
      70, 34, 69, 55, 55, 32, 0,  81, 78, 7,  13, 50, 95, 69, 13, 76, 90, 85,
      28, 62, 22, 66, 44, 92, 95, 54, 2,  12, 43, 74, 51, 54, 97, 14, 34, 45,
      17, 13, 57, 29, 87, 49, 79, 20, 78, 92, 20, 82, 21, 71, 93, 51, 18, 58,
      12, 55, 70, 97, 60, 51, 94, 65, 64, 76, 27, 57, 76, 2,  32, 64, 9,  56,
      8,  37, 17, 53, 26, 45, 93, 61, 56, 9,  74, 32, 39, 82, 29, 1,  8,  95,
      2,  93, 93, 66, 56, 16, 60, 42, 28, 11, 47, 58, 98, 34, 93, 25, 49, 22,
      95, 6,  1,  78, 78, 11, 19, 13, 6,  80, 90, 20, 82, 26, 48, 51, 16, 84,
      51, 54, 94, 67, 59, 9,  29, 59, 53, 46, 13, 55, 92, 87, 48, 17, 45, 71,
      52, 86, 96, 4,  18, 32, 87, 40, 93, 98, 8,  85, 76, 88, 82, 57, 7,  61,
      5,  72, 99, 37, 45, 42, 15, 70, 8,  5,  41, 14, 28, 50, 20, 2,  77, 48,
      53, 16, 95, 78, 88, 78, 54, 19, 30, 80, 78, 97, 69, 23, 93, 48, 72, 92,
      88, 82, 17, 58, 98, 99, 70, 97, 52, 46, 66, 97, 95, 65, 38, 47, 1,  4,
      18, 31, 99, 16, 64, 84, 44, 40, 46, 2,  46, 32, 8,  47, 64, 28, 87, 70,
      80, 25, 85, 17, 43, 56, 97, 91, 20, 7,  70, 82, 32, 58, 46, 43, 25, 81,
      12, 97, 40, 73, 52, 27, 13, 30, 58, 1,  89, 68, 75, 17, 91, 22, 12, 48,
      41, 98, 81, 44, 60, 93, 54, 81, 3,  8,  43, 16, 11, 62, 33, 81, 1,  49,
      51, 67, 83, 83, 93, 7,  63, 71, 41, 39, 63, 52, 77, 77, 47, 20, 32, 26,
      20, 66, 64, 62, 94, 55, 37, 39, 28, 45, 67, 76, 6,  43, 10, 18, 55, 44,
      35, 41, 29, 33, 96, 90, 72, 70, 87, 75, 97, 43, 36, 14, 79, 8,  10, 83,
      33, 29, 83, 74, 72, 83, 96, 77, 72, 91, 41, 9,  85, 34, 7,  51, 13, 88,
      69, 47, 23, 22, 64, 2,  7,  38, 66, 58, 7,  8,  35, 92, 53, 65, 4,  94,
      79, 29, 88, 23, 81, 72, 55, 22, 44, 78, 75, 80, 74, 28, 54, 16, 8,  16,
      73, 92, 31, 17, 44, 6,  32, 80, 5,  61, 2,  58, 7,  80, 89, 51, 59, 63,
      65, 42, 93, 14, 44, 16, 36, 79, 41, 45, 33, 36, 13, 92, 85, 75, 7,  47,
      31, 62, 98, 66, 5,  20, 55, 26, 21, 93, 50, 62, 44, 3,  66, 43, 11, 15,
      35, 78, 73, 26, 55, 90, 90, 8,  40, 74, 17, 8,  61, 47, 76, 41, 43, 50,
      94, 62, 85, 44, 47, 91, 72, 86, 10, 86, 62, 18, 51, 23, 83, 0,  61, 41,
      99, 24, 15, 72, 42, 56, 19, 34, 54, 63, 5,  14, 3,  64, 26, 6,  1,  21,
      25, 64, 19, 84, 49, 55, 32, 85, 76, 62, 1,  52, 15, 86, 21, 49, 92, 22,
      79, 20, 90, 27, 32, 46, 76, 55, 23, 69, 56, 80, 35, 35, 30, 43, 70, 79,
      73, 12, 60, 20, 22, 80, 83, 72, 66, 56, 41, 68, 4,  8,  94, 97, 41, 76,
      96, 3,  53, 61, 15, 89, 65, 45, 65, 15, 6,  83, 82, 69, 76, 68, 95, 81,
      55, 55, 85, 26, 75, 34, 67, 75, 28, 95, 58, 11, 73, 96, 44, 70, 82, 89,
      72, 40, 17, 89, 51, 87, 69, 85, 45, 59, 2,  53, 82, 87, 24, 33, 41, 53,
      97, 35, 0,  54, 7,  94, 71, 42, 68, 88, 53, 15, 41, 79, 1,  24, 49, 54,
      26, 88, 23, 89, 14, 41, 52, 8,  12, 92, 98, 54, 56, 27, 17, 11, 89, 82,
      34, 81, 78, 15, 63, 18, 17, 18, 40, 85, 41, 57, 68, 21, 7,  34, 44, 97,
      20, 5,  67, 14, 32, 86, 8,  48, 8,  6,  28, 50, 74, 91, 82, 18, 26, 51,
      38, 21, 90, 54, 64, 91, 65, 32, 6,  67, 6,  97, 32, 70, 88, 39, 80, 39,
      86, 13, 72, 81, 6,  93, 10, 67, 41, 32, 32, 8,  60, 95, 94, 11, 63, 45,
      25, 25, 46, 28, 10, 91, 16, 82, 23, 88, 10, 21, 32, 31, 90, 26, 55, 59,
      74, 36, 49, 78, 86, 68, 6,  22, 25, 59, 51, 96, 77, 60, 20, 32, 36, 91,
      56, 52, 85, 42, 26, 30, 17, 31, 5,  18, 74, 42, 75, 45, 31, 40, 81, 65,
      20, 29, 94, 10, 71, 40, 69, 83, 83, 24, 76, 25, 73, 40, 47, 75, 44, 66,
      11, 52, 90, 6,  30, 85, 18, 56, 22, 18, 51, 54, 18, 18, 99, 80, 37, 89,
      83, 8,  83, 74, 18, 48, 39, 3,  45, 47, 70, 59, 14, 15, 94, 84, 39, 62,
      42, 79, 84, 88, 26, 52, 34, 48, 92, 28, 20, 59, 53, 81, 34, 5,  98, 36,
      18, 80, 36, 8,  83, 28, 98, 67, 92, 44, 9,  47, 65, 59, 11, 31, 33, 88,
      77, 2,  20, 22, 0,  24, 12, 45, 88, 11, 38, 75, 43, 99, 30, 71, 66, 47,
      67, 14, 22, 57, 40, 88, 48, 12, 89, 6,  93, 28, 96, 37, 99, 38, 75, 72,
      68, 42, 11, 76, 53, 4,  9,  38, 7,  77, 47, 46, 66, 73, 27, 93, 17, 87,
      9,  72, 77, 78, 1,  74, 97, 54, 87, 44, 43, 64, 70, 34, 62, 82, 74, 48,
      41, 54, 41, 78, 75, 4,  21, 30, 80, 41, 17, 13, 76, 87, 47, 68, 37, 17,
      42, 32, 23, 15, 70, 56, 40, 31, 33, 79, 77, 73, 21, 4,  54, 41, 25, 67,
      18, 6,  26, 42, 36, 44, 33, 87, 94, 22, 41, 79, 15, 16, 5,  84, 29, 30,
      25, 67, 3,  55, 96, 36, 36, 89, 2,  47, 92, 94, 23, 63, 54, 45, 14, 41,
      18, 48, 61, 91, 33, 99, 9,  52, 59, 71, 20, 62, 99, 94, 6,  79, 59, 99,
      94, 3,  9,  16, 53, 74, 55, 43, 44, 62, 89, 2,  17, 97, 47, 99, 87, 31,
      90, 82, 26, 33, 7,  92, 0,  98, 78, 94, 44, 89, 5,  97, 18, 43, 19, 6,
      74, 57, 33, 0,  14, 50, 43, 8,  19, 21, 96, 95, 28, 60, 11, 81, 65, 10,
      20, 51, 45, 45, 54, 16, 22, 26, 35, 30, 79, 51, 16, 91, 25, 40, 25, 75,
      85, 43, 72, 3,  23, 5,  59, 90, 12, 89, 81, 86, 28, 75, 5,  79, 45, 28,
      33, 65, 22, 15, 14, 76, 29, 85, 89, 37, 19, 84, 5,  51};
  int8_t b2048[] = {
      43, 84, 90, 44, 54, 43, 49, 42, 24, 10, 61, 8,  68, 2,  75, 9,  25, 25,
      80, 6,  9,  62, 33, 22, 84, 43, 20, 34, 33, 53, 47, 8,  16, 15, 4,  96,
      3,  73, 75, 61, 75, 68, 37, 6,  25, 48, 40, 0,  67, 89, 98, 92, 37, 72,
      44, 94, 88, 42, 97, 24, 11, 24, 39, 13, 34, 30, 58, 22, 29, 28, 22, 82,
      15, 16, 57, 99, 9,  7,  76, 57, 39, 31, 21, 7,  44, 73, 88, 8,  62, 47,
      45, 65, 11, 78, 82, 89, 72, 18, 9,  24, 59, 75, 17, 0,  70, 1,  62, 52,
      51, 67, 5,  99, 83, 80, 82, 16, 43, 43, 94, 8,  52, 58, 68, 60, 72, 26,
      57, 22, 72, 95, 70, 12, 51, 43, 28, 53, 72, 0,  12, 67, 96, 89, 34, 28,
      9,  96, 5,  82, 19, 52, 28, 8,  8,  45, 60, 34, 66, 60, 54, 41, 87, 13,
      15, 23, 96, 29, 70, 50, 72, 10, 87, 98, 81, 11, 43, 27, 96, 9,  17, 16,
      6,  14, 31, 12, 89, 55, 37, 91, 50, 74, 12, 63, 10, 77, 81, 5,  98, 96,
      22, 9,  3,  48, 96, 1,  36, 87, 54, 40, 91, 51, 35, 38, 56, 78, 84, 4,
      95, 2,  20, 18, 87, 60, 73, 28, 69, 55, 8,  12, 86, 2,  31, 55, 46, 57,
      77, 25, 54, 50, 58, 13, 93, 6,  79, 80, 83, 78, 27, 1,  14, 52, 70, 82,
      87, 81, 82, 63, 86, 24, 37, 12, 66, 22, 63, 93, 21, 11, 86, 92, 22, 47,
      33, 84, 28, 69, 69, 31, 39, 43, 2,  29, 14, 14, 62, 42, 75, 37, 36, 88,
      98, 53, 18, 81, 40, 3,  49, 85, 99, 65, 15, 21, 23, 88, 42, 80, 79, 94,
      46, 2,  46, 91, 80, 4,  13, 90, 3,  52, 23, 65, 30, 1,  37, 86, 71, 64,
      63, 56, 44, 10, 49, 6,  31, 10, 85, 75, 50, 27, 65, 58, 96, 0,  26, 0,
      69, 70, 3,  69, 91, 96, 59, 44, 29, 20, 22, 54, 16, 69, 0,  16, 3,  69,
      64, 68, 55, 9,  71, 62, 38, 84, 6,  27, 21, 50, 42, 1,  27, 14, 49, 16,
      74, 10, 45, 31, 37, 61, 72, 8,  94, 93, 25, 81, 62, 9,  35, 15, 21, 48,
      64, 62, 18, 72, 38, 85, 55, 27, 20, 86, 56, 84, 72, 12, 59, 54, 94, 83,
      21, 25, 34, 11, 82, 32, 59, 90, 97, 81, 29, 18, 38, 16, 5,  53, 96, 85,
      19, 88, 37, 72, 32, 38, 41, 74, 70, 12, 60, 3,  67, 29, 2,  60, 38, 6,
      82, 34, 53, 24, 31, 18, 14, 40, 39, 61, 10, 6,  69, 40, 76, 32, 9,  4,
      47, 65, 13, 45, 60, 35, 59, 53, 67, 88, 74, 71, 3,  32, 97, 4,  77, 55,
      25, 27, 38, 18, 91, 48, 86, 18, 30, 66, 22, 3,  24, 8,  43, 72, 75, 22,
      7,  46, 5,  58, 67, 10, 95, 55, 99, 12, 59, 40, 57, 89, 50, 80, 41, 41,
      36, 28, 35, 87, 66, 94, 9,  11, 24, 19, 94, 51, 3,  34, 21, 44, 33, 71,
      12, 1,  58, 84, 78, 85, 55, 41, 63, 25, 13, 15, 69, 7,  43, 55, 52, 15,
      16, 19, 85, 63, 71, 66, 29, 55, 64, 27, 79, 74, 15, 62, 54, 83, 50, 38,
      54, 2,  40, 29, 94, 65, 32, 50, 41, 72, 5,  68, 15, 8,  4,  50, 74, 37,
      76, 61, 53, 71, 9,  70, 1,  1,  44, 38, 7,  6,  49, 53, 44, 57, 80, 45,
      79, 97, 85, 2,  81, 3,  67, 72, 31, 52, 41, 42, 83, 97, 30, 32, 39, 38,
      71, 32, 17, 96, 12, 34, 52, 64, 25, 20, 60, 2,  53, 66, 1,  38, 10, 75,
      98, 44, 11, 16, 15, 53, 12, 29, 18, 46, 91, 13, 26, 36, 74, 32, 3,  97,
      76, 97, 80, 11, 27, 54, 57, 9,  0,  10, 28, 8,  55, 83, 56, 57, 82, 2,
      70, 42, 2,  64, 84, 97, 1,  34, 2,  7,  42, 54, 20, 55, 39, 77, 79, 58,
      59, 16, 98, 95, 31, 22, 80, 77, 15, 12, 39, 29, 86, 8,  4,  13, 72, 95,
      67, 45, 2,  53, 61, 3,  87, 94, 33, 60, 63, 33, 42, 33, 44, 35, 69, 22,
      96, 69, 73, 33, 28, 0,  79, 23, 54, 23, 80, 87, 99, 32, 56, 0,  51, 40,
      12, 28, 68, 74, 6,  71, 68, 18, 72, 99, 58, 48, 44, 12, 55, 98, 46, 19,
      93, 62, 65, 36, 43, 38, 10, 23, 3,  48, 27, 51, 5,  48, 97, 28, 73, 64,
      43, 77, 10, 52, 36, 5,  1,  44, 18, 20, 58, 21, 30, 14, 12, 35, 66, 90,
      31, 69, 93, 30, 51, 17, 43, 10, 53, 83, 91, 65, 44, 72, 32, 41, 41, 3,
      48, 67, 98, 86, 65, 67, 82, 25, 73, 53, 23, 99, 86, 95, 43, 52, 53, 82,
      65, 79, 59, 64, 69, 89, 71, 13, 60, 28, 61, 97, 88, 39, 31, 65, 90, 40,
      20, 51, 2,  6,  74, 2,  62, 97, 21, 6,  25, 23, 42, 72, 24, 96, 72, 84,
      55, 29, 32, 55, 98, 79, 16, 52, 69, 85, 74, 19, 26, 25, 6,  47, 88, 90,
      40, 63, 58, 45, 64, 59, 65, 83, 27, 62, 15, 65, 23, 68, 23, 95, 13, 35,
      6,  93, 97, 91, 37, 37, 7,  86, 98, 81, 34, 61, 44, 4,  85, 87, 74, 54,
      80, 45, 68, 19, 48, 27, 73, 78, 76, 90, 75, 93, 4,  32, 36, 87, 19, 71,
      47, 37, 83, 83, 99, 58, 83, 2,  34, 25, 18, 25, 74, 8,  12, 96, 83, 93,
      36, 96, 4,  82, 9,  57, 70, 36, 96, 73, 88, 72, 69, 80, 10, 12, 20, 11,
      33, 97, 79, 52, 83, 56, 71, 59, 20, 70, 50, 63, 79, 60, 15, 97, 72, 47,
      53, 60, 89, 53, 98, 24, 86, 40, 74, 9,  39, 27, 15, 59, 11, 84, 41, 68,
      91, 13, 27, 40, 52, 89, 29, 52, 32, 37, 33, 48, 44, 10, 62, 18, 87, 53,
      56, 84, 95, 57, 38, 73, 75, 58, 66, 93, 65, 81, 45, 66, 54, 73, 27, 72,
      46, 46, 19, 46, 53, 53, 5,  77, 88, 3,  19, 99, 67, 16, 89, 93, 68, 37,
      29, 94, 69, 3,  29, 8,  76, 25, 0,  28, 24, 71, 25, 90, 87, 97, 32, 80,
      23, 90, 86, 30, 80, 40, 80, 46, 17, 66, 97, 4,  36, 2,  31, 14, 75, 15,
      34, 84, 56, 76, 61, 15, 93, 87, 52, 69, 26, 2,  18, 39, 60, 37, 31, 79,
      27, 84, 36, 53, 76, 62, 71, 62, 74, 51, 59, 26, 70, 94, 56, 89, 72, 3,
      26, 27, 66, 49, 16, 13, 81, 44, 85, 7,  54, 6,  14, 35, 60, 84, 48, 24,
      11, 29, 57, 15, 0,  76, 23, 72, 11, 50, 69, 90, 20, 5,  32, 64, 4,  23,
      82, 33, 94, 69, 28, 99, 80, 85, 27, 89, 8,  45, 37, 34, 57, 87, 37, 57,
      73, 17, 56, 45, 25, 1,  67, 67, 67, 56, 81, 20, 23, 25, 37, 93, 93, 13,
      5,  58, 62, 93, 16, 61, 69, 43, 52, 66, 59, 20, 65, 89, 84, 67, 98, 98,
      10, 21, 10, 27, 83, 39, 69, 6,  49, 88, 95, 83, 61, 87, 78, 38, 67, 43,
      45, 61, 69, 71, 4,  45, 49, 78, 51, 30, 84, 4,  47, 18, 71, 73, 32, 73,
      24, 56, 76, 82, 99, 40, 39, 42, 71, 24, 57, 83, 31, 68, 6,  38, 38, 2,
      46, 46, 90, 61, 89, 30, 15, 5,  76, 24, 70, 35, 90, 45, 45, 91, 47, 73,
      34, 30, 53, 64, 61, 94, 96, 58, 84, 37, 32, 19, 34, 12, 96, 75, 28, 86,
      66, 91, 55, 93, 93, 6,  69, 51, 44, 92, 40, 85, 22, 1,  42, 10, 38, 86,
      52, 28, 19, 7,  75, 5,  47, 28, 52, 76, 50, 27, 56, 59, 95, 85, 89, 63,
      62, 73, 56, 52, 89, 5,  8,  70, 28, 62, 36, 21, 15, 6,  19, 10, 19, 38,
      4,  61, 27, 87, 71, 54, 34, 5,  27, 48, 8,  26, 48, 29, 4,  76, 52, 29,
      21, 36, 34, 87, 11, 97, 78, 0,  34, 46, 93, 51, 77, 14, 47, 86, 2,  92,
      84, 92, 15, 57, 67, 12, 37, 5,  74, 49, 59, 13, 88, 0,  59, 29, 86, 91,
      19, 20, 2,  19, 4,  51, 68, 5,  77, 26, 36, 88, 73, 13, 68, 5,  77, 18,
      25, 13, 25, 47, 66, 69, 75, 45, 51, 35, 2,  61, 95, 60, 86, 97, 17, 74,
      19, 52, 43, 76, 26, 51, 38, 27, 13, 81, 53, 3,  87, 2,  99, 36, 7,  72,
      44, 42, 10, 8,  78, 87, 20, 75, 9,  36, 25, 0,  56, 37, 20, 13, 41, 80,
      69, 76, 39, 47, 61, 28, 87, 81, 30, 11, 4,  62, 66, 3,  77, 7,  0,  95,
      52, 81, 42, 84, 47, 78, 55, 25, 55, 13, 63, 32, 16, 68, 8,  35, 1,  30,
      66, 75, 79, 63, 71, 63, 65, 70, 92, 74, 68, 92, 61, 97, 36, 86, 61, 3,
      85, 13, 97, 69, 56, 58, 22, 71, 70, 86, 61, 33, 79, 91, 21, 72, 80, 65,
      27, 14, 82, 82, 20, 87, 47, 4,  38, 49, 89, 63, 10, 45, 48, 96, 8,  78,
      95, 67, 3,  6,  2,  64, 44, 89, 13, 31, 18, 83, 95, 92, 11, 80, 35, 87,
      14, 14, 58, 22, 86, 16, 98, 7,  26, 67, 27, 91, 96, 56, 28, 19, 17, 81,
      4,  56, 23, 19, 17, 77, 54, 93, 64, 27, 21, 40, 31, 24, 24, 55, 28, 73,
      13, 33, 76, 47, 38, 48, 66, 95, 72, 84, 23, 77, 65, 5,  28, 55, 32, 0,
      14, 47, 57, 33, 36, 26, 59, 98, 85, 2,  49, 29, 40, 44, 84, 24, 23, 88,
      66, 91, 4,  0,  4,  99, 40, 94, 55, 19, 13, 22, 96, 37, 89, 94, 78, 50,
      0,  37, 48, 79, 69, 16, 15, 57, 91, 52, 85, 92, 18, 38, 56, 55, 11, 10,
      27, 48, 98, 53, 83, 27, 14, 25, 53, 64, 71, 67, 26, 47, 53, 30, 76, 76,
      67, 83, 9,  20, 4,  61, 69, 10, 93, 63, 37, 22, 26, 64, 10, 75, 39, 86,
      34, 44, 42, 44, 4,  42, 37, 85, 3,  95, 49, 43, 84, 44, 73, 7,  59, 33,
      21, 46, 86, 88, 17, 88, 83, 32, 53, 6,  83, 85, 54, 32, 92, 45, 13, 20,
      49, 42, 7,  54, 76, 62, 58, 13, 99, 43, 94, 60, 43, 94, 58, 35, 69, 84,
      23, 57, 22, 81, 97, 97, 49, 91, 76, 65, 71, 82, 72, 39, 53, 92, 58, 77,
      20, 39, 20, 48, 46, 52, 20, 9,  85, 9,  48, 89, 24, 65, 73, 81, 73, 10,
      1,  25, 89, 83, 48, 38, 56, 82, 68, 27, 35, 87, 68, 32, 89, 23, 90, 5,
      99, 19, 55, 97, 83, 41, 34, 29, 69, 58, 8,  2,  90, 54, 66, 66, 37, 27,
      86, 46, 48, 50, 63, 76, 96, 41, 36, 9,  38, 31, 46, 58, 17, 53, 53, 81,
      79, 94, 95, 98, 96, 40, 43, 63, 2,  5,  26, 22, 10, 21, 43, 30, 30, 29,
      80, 49, 51, 74, 41, 64, 86, 50, 23, 81, 48, 41, 48, 98, 55, 38, 61, 40,
      52, 79, 99, 17, 71, 78, 62, 40, 5,  15, 26, 47, 75, 67, 17, 46, 93, 90,
      2,  81, 78, 22, 12, 74, 7,  7,  36, 48, 13, 41, 30, 68, 86, 50, 28, 72,
      40, 45, 82, 92, 38, 95, 68, 48, 42, 23, 4,  40, 82, 9,  59, 81, 58, 33,
      68, 12, 60, 71, 91, 47, 49, 21, 55, 1,  77, 57, 53, 4,  67, 4,  13, 29,
      76, 28, 70, 29, 20, 25, 81, 1,  57, 26, 74, 79, 95, 63, 83, 3,  28, 31,
      49, 30, 87, 84, 29, 60, 47, 49, 45, 16, 37, 68, 13, 19};

  EXPECT_FLOAT_EQ(1844.638672f,
                  EuclideanDistance(*FixedVector<int8_t, 2048>::Cast(a2048),
                                    *FixedVector<int8_t, 2048>::Cast(b2048)));
}

TEST(DistanceMatrix, SquaredEuclidean_General) {
  int8_t a8[] = {127, 0, 1, 2, -127, -127, -127, -127};
  int8_t b8[] = {-127, -127, -127, -127, 1, 2, 1, 127};
  int8_t a16[] = {127, 127, 16,   3,   100,  -127, 1,    2,
                  3,   4,   -127, 100, -127, -127, -127, -127};
  int8_t b16[] = {-127, 123, -127, -127, -127, -127, 127, 127,
                  1,    2,   3,    4,    127,  127,  121, 16};
  int8_t a32[] = {127, 127,  0,    0,   -127, -127, 0,    0,    0,    0, 0,
                  0,   -127, -127, 127, 127,  0,    0,    -127, -127, 0, 0,
                  127, 127,  127,  127, 0,    0,    -127, -127, 0,    0};
  int8_t b32[] = {-127, -127, 0,    0,    127,  127, 0,   0,   0,   0, 0,
                  0,    127,  127,  -127, -127, 0,   0,   127, 127, 0, 0,
                  -127, -127, -127, -127, 0,    0,   127, 127, 0,   0};

  int8_t a47[] = {127, 2, 0,    0,    -127, -127, 0,    0,    0,    0,
                  0,   0, -127, -127, 127,  127,  0,    0,    -127, -127,
                  0,   0, 127,  5,    127,  127,  0,    0,    -127, -127,
                  0,   0, -127, 112,  -127, -127, -127, -127, 127,  127,
                  1,   2, 3,    4,    127,  127,  120};
  int8_t b47[] = {-127, 1, 0,    0,   127,  127,  0,   0,    0,   0,
                  0,    0, 127,  127, -127, -127, 0,   0,    127, 127,
                  0,    0, -127, 3,   -127, -127, 0,   0,    127, 127,
                  0,    0, 127,  127, 80,   111,  122, -127, 1,   2,
                  3,    4, -127, 112, -127, -127, -127};

  EXPECT_FLOAT_EQ(227595.0f,
                  SquaredEuclideanDistance(*FixedVector<int8_t, 8>::Cast(a8),
                                           *FixedVector<int8_t, 8>::Cast(b8)));
  EXPECT_FLOAT_EQ(
      422020.0f, SquaredEuclideanDistance(*FixedVector<int8_t, 16>::Cast(a16),
                                          *FixedVector<int8_t, 16>::Cast(b16)));
  EXPECT_FLOAT_EQ(1032256.0f, SquaredEuclideanDistance(
                                  *FixedVector<int8_t, 32>::Cast(a32),
                                  *FixedVector<int8_t, 32>::Cast(b32)));
  EXPECT_FLOAT_EQ(1379578.0f, SquaredEuclideanDistance(
                                  *FixedVector<int8_t, 47>::Cast(a47),
                                  *FixedVector<int8_t, 47>::Cast(b47)));
}

template <size_t M, size_t N>
void TestSquaredEuclideanMatrix(void) {
  std::mt19937 gen((std::random_device())());

  const size_t batch_size = M;
  const size_t query_size = N;
  size_t dimension = (std::uniform_int_distribution<size_t>(1, 64))(gen) << 2;
  size_t matrix_size = batch_size * dimension;
  size_t query_matrix_size = query_size * dimension;

  std::vector<int8_t> matrix1(matrix_size);
  std::vector<int8_t> matrix2(matrix_size);
  std::vector<int8_t> query1(query_matrix_size);
  std::vector<int8_t> query2(query_matrix_size);
  std::vector<float> result1(batch_size * query_size);
  std::vector<float> result2(batch_size * query_size);

  std::uniform_int_distribution<int> dist(-127, 127);
  for (size_t i = 0; i < matrix_size; ++i) {
    matrix1[i] = (int8_t)dist(gen);
  }
  for (size_t i = 0; i < query_matrix_size; ++i) {
    query1[i] = (int8_t)dist(gen);
  }
  MatrixTranspose((uint32_t *)(&matrix2[0]), (const uint32_t *)matrix1.data(),
                  dimension / 4, batch_size);
  MatrixTranspose((uint32_t *)(&query2[0]), (const uint32_t *)query1.data(),
                  dimension / 4, query_size);

  for (size_t i = 0; i < query_size; ++i) {
    const int8_t *cur_query = &query1[i * dimension];
    float *query_result = &result1[i * batch_size];

    for (size_t j = 0; j < batch_size; ++j) {
      SquaredEuclideanDistanceMatrix<int8_t, 1, 1>::Compute(
          &matrix1[j * dimension], cur_query, dimension, &query_result[j]);
    }
  }
  SquaredEuclideanDistanceMatrix<int8_t, batch_size, query_size>::Compute(
      &matrix2[0], &query2[0], dimension, &result2[0]);

  for (size_t i = 0; i < batch_size * query_size; ++i) {
    EXPECT_FLOAT_EQ(result1[i], result2[i]);
  }
}

template <size_t M, size_t N>
void TestEuclideanMatrix(void) {
  std::mt19937 gen((std::random_device())());

  const size_t batch_size = M;
  const size_t query_size = N;
  size_t dimension = (std::uniform_int_distribution<size_t>(1, 64))(gen) << 2;
  size_t matrix_size = batch_size * dimension;
  size_t query_matrix_size = query_size * dimension;

  std::vector<int8_t> matrix1(matrix_size);
  std::vector<int8_t> matrix2(matrix_size);
  std::vector<int8_t> query1(query_matrix_size);
  std::vector<int8_t> query2(query_matrix_size);
  std::vector<float> result1(batch_size * query_size);
  std::vector<float> result2(batch_size * query_size);

  std::uniform_int_distribution<int> dist(-127, 127);
  for (size_t i = 0; i < matrix_size; ++i) {
    matrix1[i] = (int8_t)dist(gen);
  }
  for (size_t i = 0; i < query_matrix_size; ++i) {
    query1[i] = (int8_t)dist(gen);
  }
  MatrixTranspose((uint32_t *)(&matrix2[0]), (const uint32_t *)matrix1.data(),
                  dimension / 4, batch_size);
  MatrixTranspose((uint32_t *)(&query2[0]), (const uint32_t *)query1.data(),
                  dimension / 4, query_size);

  for (size_t i = 0; i < query_size; ++i) {
    const int8_t *cur_query = &query1[i * dimension];
    float *query_result = &result1[i * batch_size];

    for (size_t j = 0; j < batch_size; ++j) {
      EuclideanDistanceMatrix<int8_t, 1, 1>::Compute(
          &matrix1[j * dimension], cur_query, dimension, &query_result[j]);
    }
  }
  EuclideanDistanceMatrix<int8_t, batch_size, query_size>::Compute(
      &matrix2[0], &query2[0], dimension, &result2[0]);

  for (size_t i = 0; i < batch_size * query_size; ++i) {
    EXPECT_FLOAT_EQ(result1[i], result2[i]);
  }
}

TEST(DistanceMatrix, SquaredEuclidean_1x1) {
  TestSquaredEuclideanMatrix<1, 1>();
}

TEST(DistanceMatrix, SquaredEuclidean_2x1) {
  TestSquaredEuclideanMatrix<2, 1>();
}

TEST(DistanceMatrix, SquaredEuclidean_2x2) {
  TestSquaredEuclideanMatrix<2, 2>();
}

TEST(DistanceMatrix, SquaredEuclidean_3x3) {
  TestSquaredEuclideanMatrix<3, 3>();
}

TEST(DistanceMatrix, SquaredEuclidean_4x1) {
  TestSquaredEuclideanMatrix<4, 1>();
}

TEST(DistanceMatrix, SquaredEuclidean_4x2) {
  TestSquaredEuclideanMatrix<4, 2>();
}

TEST(DistanceMatrix, SquaredEuclidean_4x4) {
  TestSquaredEuclideanMatrix<4, 4>();
}

TEST(DistanceMatrix, SquaredEuclidean_8x1) {
  TestSquaredEuclideanMatrix<8, 1>();
}

TEST(DistanceMatrix, SquaredEuclidean_8x2) {
  TestSquaredEuclideanMatrix<8, 2>();
}

TEST(DistanceMatrix, SquaredEuclidean_8x4) {
  TestSquaredEuclideanMatrix<8, 4>();
}

TEST(DistanceMatrix, SquaredEuclidean_8x8) {
  TestSquaredEuclideanMatrix<8, 8>();
}

TEST(DistanceMatrix, SquaredEuclidean_16x1) {
  TestSquaredEuclideanMatrix<16, 1>();
}

TEST(DistanceMatrix, SquaredEuclidean_16x2) {
  TestSquaredEuclideanMatrix<16, 2>();
}

TEST(DistanceMatrix, SquaredEuclidean_16x4) {
  TestSquaredEuclideanMatrix<16, 4>();
}

TEST(DistanceMatrix, SquaredEuclidean_16x8) {
  TestSquaredEuclideanMatrix<16, 8>();
}

TEST(DistanceMatrix, SquaredEuclidean_16x16) {
  TestSquaredEuclideanMatrix<16, 16>();
}

TEST(DistanceMatrix, SquaredEuclidean_32x1) {
  TestSquaredEuclideanMatrix<32, 1>();
}

TEST(DistanceMatrix, SquaredEuclidean_32x2) {
  TestSquaredEuclideanMatrix<32, 2>();
}

TEST(DistanceMatrix, SquaredEuclidean_32x4) {
  TestSquaredEuclideanMatrix<32, 4>();
}

TEST(DistanceMatrix, SquaredEuclidean_32x8) {
  TestSquaredEuclideanMatrix<32, 8>();
}

TEST(DistanceMatrix, SquaredEuclidean_32x16) {
  TestSquaredEuclideanMatrix<32, 16>();
}

TEST(DistanceMatrix, SquaredEuclidean_32x32) {
  TestSquaredEuclideanMatrix<32, 32>();
}

TEST(DistanceMatrix, SquaredEuclidean_64x1) {
  TestSquaredEuclideanMatrix<64, 1>();
}

TEST(DistanceMatrix, SquaredEuclidean_64x2) {
  TestSquaredEuclideanMatrix<64, 2>();
}

TEST(DistanceMatrix, SquaredEuclidean_64x4) {
  TestSquaredEuclideanMatrix<64, 4>();
}

TEST(DistanceMatrix, SquaredEuclidean_64x8) {
  TestSquaredEuclideanMatrix<64, 8>();
}

TEST(DistanceMatrix, SquaredEuclidean_64x16) {
  TestSquaredEuclideanMatrix<64, 16>();
}

TEST(DistanceMatrix, SquaredEuclidean_64x32) {
  TestSquaredEuclideanMatrix<64, 32>();
}

TEST(DistanceMatrix, SquaredEuclidean_64x64) {
  TestSquaredEuclideanMatrix<64, 128>();
}

TEST(DistanceMatrix, SquaredEuclidean_128x1) {
  TestSquaredEuclideanMatrix<128, 1>();
}

TEST(DistanceMatrix, SquaredEuclidean_128x2) {
  TestSquaredEuclideanMatrix<128, 2>();
}

TEST(DistanceMatrix, SquaredEuclidean_128x4) {
  TestSquaredEuclideanMatrix<128, 4>();
}

TEST(DistanceMatrix, SquaredEuclidean_128x8) {
  TestSquaredEuclideanMatrix<128, 8>();
}

TEST(DistanceMatrix, SquaredEuclidean_128x16) {
  TestSquaredEuclideanMatrix<128, 16>();
}

TEST(DistanceMatrix, SquaredEuclidean_128x32) {
  TestSquaredEuclideanMatrix<128, 32>();
}

TEST(DistanceMatrix, SquaredEuclidean_128x64) {
  TestSquaredEuclideanMatrix<128, 128>();
}

TEST(DistanceMatrix, SquaredEuclidean_128x128) {
  TestSquaredEuclideanMatrix<128, 128>();
}

TEST(DistanceMatrix, Euclidean_1x1) {
  TestEuclideanMatrix<1, 1>();
}

TEST(DistanceMatrix, Euclidean_2x1) {
  TestEuclideanMatrix<2, 1>();
}

TEST(DistanceMatrix, Euclidean_2x2) {
  TestEuclideanMatrix<2, 2>();
}

TEST(DistanceMatrix, Euclidean_3x3) {
  TestEuclideanMatrix<3, 3>();
}

TEST(DistanceMatrix, Euclidean_4x1) {
  TestEuclideanMatrix<4, 1>();
}

TEST(DistanceMatrix, Euclidean_4x2) {
  TestEuclideanMatrix<4, 2>();
}

TEST(DistanceMatrix, Euclidean_4x4) {
  TestEuclideanMatrix<4, 4>();
}

TEST(DistanceMatrix, Euclidean_8x1) {
  TestEuclideanMatrix<8, 1>();
}

TEST(DistanceMatrix, Euclidean_8x2) {
  TestEuclideanMatrix<8, 2>();
}

TEST(DistanceMatrix, Euclidean_8x4) {
  TestEuclideanMatrix<8, 4>();
}

TEST(DistanceMatrix, Euclidean_8x8) {
  TestEuclideanMatrix<8, 8>();
}

TEST(DistanceMatrix, Euclidean_16x1) {
  TestEuclideanMatrix<16, 1>();
}

TEST(DistanceMatrix, Euclidean_16x2) {
  TestEuclideanMatrix<16, 2>();
}

TEST(DistanceMatrix, Euclidean_16x4) {
  TestEuclideanMatrix<16, 4>();
}

TEST(DistanceMatrix, Euclidean_16x8) {
  TestEuclideanMatrix<16, 8>();
}

TEST(DistanceMatrix, Euclidean_16x16) {
  TestEuclideanMatrix<16, 16>();
}

TEST(DistanceMatrix, Euclidean_32x1) {
  TestEuclideanMatrix<32, 1>();
}

TEST(DistanceMatrix, Euclidean_32x2) {
  TestEuclideanMatrix<32, 2>();
}

TEST(DistanceMatrix, Euclidean_32x4) {
  TestEuclideanMatrix<32, 4>();
}

TEST(DistanceMatrix, Euclidean_32x8) {
  TestEuclideanMatrix<32, 8>();
}

TEST(DistanceMatrix, Euclidean_32x16) {
  TestEuclideanMatrix<32, 16>();
}

TEST(DistanceMatrix, Euclidean_32x32) {
  TestEuclideanMatrix<32, 32>();
}

TEST(DistanceMatrix, Euclidean_64x1) {
  TestEuclideanMatrix<64, 1>();
}

TEST(DistanceMatrix, Euclidean_64x2) {
  TestEuclideanMatrix<64, 2>();
}

TEST(DistanceMatrix, Euclidean_64x4) {
  TestEuclideanMatrix<64, 4>();
}

TEST(DistanceMatrix, Euclidean_64x8) {
  TestEuclideanMatrix<64, 8>();
}

TEST(DistanceMatrix, Euclidean_64x16) {
  TestEuclideanMatrix<64, 16>();
}

TEST(DistanceMatrix, Euclidean_64x32) {
  TestEuclideanMatrix<64, 32>();
}

TEST(DistanceMatrix, Euclidean_64x64) {
  TestEuclideanMatrix<64, 128>();
}

TEST(DistanceMatrix, Euclidean_128x1) {
  TestEuclideanMatrix<128, 1>();
}

TEST(DistanceMatrix, Euclidean_128x2) {
  TestEuclideanMatrix<128, 2>();
}

TEST(DistanceMatrix, Euclidean_128x4) {
  TestEuclideanMatrix<128, 4>();
}

TEST(DistanceMatrix, Euclidean_128x8) {
  TestEuclideanMatrix<128, 8>();
}

TEST(DistanceMatrix, Euclidean_128x16) {
  TestEuclideanMatrix<128, 16>();
}

TEST(DistanceMatrix, Euclidean_128x32) {
  TestEuclideanMatrix<128, 32>();
}

TEST(DistanceMatrix, Euclidean_128x64) {
  TestEuclideanMatrix<128, 128>();
}

TEST(DistanceMatrix, Euclidean_128x128) {
  TestEuclideanMatrix<128, 128>();
}

template <size_t M, size_t N, size_t B, size_t D>
void EuclideanBenchmark(void) {
  const size_t dimension = D;
  const size_t batch_size = M;
  const size_t block_size = B;
  const size_t query_size = N;
  const size_t matrix_size = block_size * batch_size * dimension;
  const size_t query_matrix_size = dimension * query_size;

  std::vector<int8_t> matrix1(matrix_size);
  std::vector<int8_t> matrix2(matrix_size);
  std::vector<int8_t> query1(query_matrix_size);
  std::vector<int8_t> query2(query_matrix_size);

  std::mt19937 gen((std::random_device())());
  std::uniform_int_distribution<int> dist(-127, 127);
  for (size_t i = 0; i < matrix_size; ++i) {
    matrix1[i] = (int8_t)dist(gen);
  }
  for (size_t i = 0; i < query_matrix_size; ++i) {
    query1[i] = (int8_t)dist(gen);
  }

  for (size_t i = 0; i < block_size; ++i) {
    size_t start_pos = i * batch_size * dimension;
    MatrixTranspose((uint32_t *)(&matrix2[start_pos]),
                    (const uint32_t *)(&matrix1[start_pos]), dimension / 4,
                    batch_size);
  }
  MatrixTranspose((uint32_t *)(&query2[0]), (const uint32_t *)query1.data(),
                  dimension / 4, query_size);

  ElapsedTime elapsed_time;
  float results[batch_size * query_size];

  std::cout << "# (" << IntelIntrinsics() << ") INT8 " << dimension << "d, "
            << batch_size << " * " << query_size << " * " << block_size
            << std::endl;

  // 1 Batched Euclidean
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const int8_t *matrix_batch = &matrix2[i * batch_size * dimension];

    for (size_t j = 0; j < query_size; ++j) {
      const int8_t *current_query = &query1[j * dimension];
      float *current_results = &results[j * batch_size];

      EuclideanDistanceMatrix<int8_t, batch_size, 1>::Compute(
          matrix_batch, current_query, dimension, current_results);
    }
  }
  std::cout << "* 1 Batched Euclidean (us) \t" << elapsed_time.micro_seconds()
            << std::endl;

  // N Batched Euclidean
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const int8_t *matrix_batch = &matrix2[i * batch_size * dimension];

    EuclideanDistanceMatrix<int8_t, batch_size, query_size>::Compute(
        matrix_batch, &query2[0], dimension, results);
  }
  std::cout << "* N Batched Euclidean (us) \t" << elapsed_time.micro_seconds()
            << std::endl;

  // Unbatched Euclidean
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const int8_t *matrix_batch = &matrix1[i * batch_size * dimension];

    for (size_t j = 0; j < query_size; ++j) {
      const int8_t *current_query = &query1[j * dimension];
      float *current_results = &results[j * batch_size];

      for (size_t k = 0; k < batch_size; ++k) {
        EuclideanDistanceMatrix<int8_t, 1, 1>::Compute(
            &matrix_batch[k * dimension], current_query, dimension,
            &current_results[k]);
      }
    }
  }
  std::cout << "* Unbatched Euclidean (us) \t" << elapsed_time.micro_seconds()
            << std::endl;
}

template <size_t M, size_t N, size_t B, size_t D>
void SquaredEuclideanBenchmark(void) {
  const size_t dimension = D;
  const size_t batch_size = M;
  const size_t block_size = B;
  const size_t query_size = N;
  const size_t matrix_size = block_size * batch_size * dimension;
  const size_t query_matrix_size = dimension * query_size;

  std::vector<int8_t> matrix1(matrix_size);
  std::vector<int8_t> matrix2(matrix_size);
  std::vector<int8_t> query1(query_matrix_size);
  std::vector<int8_t> query2(query_matrix_size);

  std::mt19937 gen((std::random_device())());
  std::uniform_int_distribution<int> dist(-127, 127);
  for (size_t i = 0; i < matrix_size; ++i) {
    matrix1[i] = (int8_t)dist(gen);
  }
  for (size_t i = 0; i < query_matrix_size; ++i) {
    query1[i] = (int8_t)dist(gen);
  }

  for (size_t i = 0; i < block_size; ++i) {
    size_t start_pos = i * batch_size * dimension;
    MatrixTranspose((uint32_t *)(&matrix2[start_pos]),
                    (const uint32_t *)(&matrix1[start_pos]), dimension / 4,
                    batch_size);
  }
  MatrixTranspose((uint32_t *)(&query2[0]), (const uint32_t *)query1.data(),
                  dimension / 4, query_size);

  ElapsedTime elapsed_time;
  float results[batch_size * query_size];

  std::cout << "# (" << IntelIntrinsics() << ") INT8 " << dimension << "d, "
            << batch_size << " * " << query_size << " * " << block_size
            << std::endl;

  // 1 Batched SquaredEuclidean
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const int8_t *matrix_batch = &matrix2[i * batch_size * dimension];

    for (size_t j = 0; j < query_size; ++j) {
      const int8_t *current_query = &query1[j * dimension];
      float *current_results = &results[j * batch_size];

      SquaredEuclideanDistanceMatrix<int8_t, batch_size, 1>::Compute(
          matrix_batch, current_query, dimension, current_results);
    }
  }
  std::cout << "* 1 Batched SquaredEuclidean (us) \t"
            << elapsed_time.micro_seconds() << std::endl;

  // N Batched SquaredEuclidean
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const int8_t *matrix_batch = &matrix2[i * batch_size * dimension];

    SquaredEuclideanDistanceMatrix<int8_t, batch_size, query_size>::Compute(
        matrix_batch, &query2[0], dimension, results);
  }
  std::cout << "* N Batched SquaredEuclidean (us) \t"
            << elapsed_time.micro_seconds() << std::endl;

  // Unbatched SquaredEuclidean
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const int8_t *matrix_batch = &matrix1[i * batch_size * dimension];

    for (size_t j = 0; j < query_size; ++j) {
      const int8_t *current_query = &query1[j * dimension];
      float *current_results = &results[j * batch_size];

      for (size_t k = 0; k < batch_size; ++k) {
        SquaredEuclideanDistanceMatrix<int8_t, 1, 1>::Compute(
            &matrix_batch[k * dimension], current_query, dimension,
            &current_results[k]);
      }
    }
  }
  std::cout << "* Unbatched SquaredEuclidean (us) \t"
            << elapsed_time.micro_seconds() << std::endl;
}

TEST(DistanceMatrix, DISABLED_Euclidean_Benchmark) {
  EuclideanBenchmark<2, 1, 512, 128>();
  EuclideanBenchmark<2, 2, 512, 128>();
  EuclideanBenchmark<4, 1, 512, 128>();
  EuclideanBenchmark<4, 2, 512, 128>();
  EuclideanBenchmark<4, 4, 512, 128>();
  EuclideanBenchmark<8, 1, 512, 128>();
  EuclideanBenchmark<8, 2, 512, 128>();
  EuclideanBenchmark<8, 4, 512, 128>();
  EuclideanBenchmark<8, 8, 512, 128>();
  EuclideanBenchmark<16, 1, 512, 128>();
  EuclideanBenchmark<16, 2, 512, 128>();
  EuclideanBenchmark<16, 4, 512, 128>();
  EuclideanBenchmark<16, 8, 512, 128>();
  EuclideanBenchmark<16, 16, 512, 128>();
  EuclideanBenchmark<32, 1, 512, 128>();
  EuclideanBenchmark<32, 2, 512, 128>();
  EuclideanBenchmark<32, 4, 512, 128>();
  EuclideanBenchmark<32, 8, 512, 128>();
  EuclideanBenchmark<32, 16, 512, 128>();
  EuclideanBenchmark<32, 32, 512, 128>();
  EuclideanBenchmark<64, 1, 512, 128>();
  EuclideanBenchmark<64, 2, 512, 128>();
  EuclideanBenchmark<64, 4, 512, 128>();
  EuclideanBenchmark<64, 8, 512, 128>();
  EuclideanBenchmark<128, 1, 512, 128>();
}

TEST(DistanceMatrix, DISABLED_SquaredEuclidean_Benchmark) {
  SquaredEuclideanBenchmark<2, 1, 512, 128>();
  SquaredEuclideanBenchmark<2, 2, 512, 128>();
  SquaredEuclideanBenchmark<4, 1, 512, 128>();
  SquaredEuclideanBenchmark<4, 2, 512, 128>();
  SquaredEuclideanBenchmark<4, 4, 512, 128>();
  SquaredEuclideanBenchmark<8, 1, 512, 128>();
  SquaredEuclideanBenchmark<8, 2, 512, 128>();
  SquaredEuclideanBenchmark<8, 4, 512, 128>();
  SquaredEuclideanBenchmark<8, 8, 512, 128>();
  SquaredEuclideanBenchmark<16, 1, 512, 128>();
  SquaredEuclideanBenchmark<16, 2, 512, 128>();
  SquaredEuclideanBenchmark<16, 4, 512, 128>();
  SquaredEuclideanBenchmark<16, 8, 512, 128>();
  SquaredEuclideanBenchmark<16, 16, 512, 128>();
  SquaredEuclideanBenchmark<32, 1, 512, 128>();
  SquaredEuclideanBenchmark<32, 2, 512, 128>();
  SquaredEuclideanBenchmark<32, 4, 512, 128>();
  SquaredEuclideanBenchmark<32, 8, 512, 128>();
  SquaredEuclideanBenchmark<32, 16, 512, 128>();
  SquaredEuclideanBenchmark<32, 32, 512, 128>();
  SquaredEuclideanBenchmark<64, 1, 512, 128>();
  SquaredEuclideanBenchmark<64, 2, 512, 128>();
  SquaredEuclideanBenchmark<64, 4, 512, 128>();
  SquaredEuclideanBenchmark<64, 8, 512, 128>();
  SquaredEuclideanBenchmark<128, 1, 512, 128>();
}
