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

#include <bitset>
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

static inline void MatrixTranspose(uint64_t *dst, const uint64_t *src, size_t M,
                                   size_t N) {
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < M; ++j) {
      dst[j * N + i] = src[i * M + j];
    }
  }
}

TEST(DistanceMatrix, Hamming_General) {
  srand((uint32_t)time(NULL));
  srand((uint32_t)rand());

  FixedBitset<63936> bitset1;
  FixedBitset<63936> bitset2;
  std::bitset<63936> stl_bitset1;
  std::bitset<63936> stl_bitset2;

  for (uint32_t i = 0; i < 1333; ++i) {
    uint32_t val1 = (uint32_t)(rand() % bitset1.size());
    uint32_t val2 = (uint32_t)(rand() % bitset2.size());

    bitset1.set(val1);
    stl_bitset1.set(val1);

    bitset2.set(val2);
    stl_bitset2.set(val2);
  }
  for (uint32_t i = 0; i < 1666; ++i) {
    uint32_t val1 = (uint32_t)(rand() % bitset1.size());
    uint32_t val2 = (uint32_t)(rand() % bitset2.size());

    bitset1.flip(val1);
    stl_bitset1.flip(val1);

    bitset2.flip(val2);
    stl_bitset2.flip(val2);
  }

  float result0 = (float)(stl_bitset1 ^ stl_bitset2).count();
  float result1 = Distance::Hamming(bitset1.data(), bitset2.data(),
                                    bitset1.size() / 32 * 32);
  float result2 = Distance::Hamming((const uint64_t *)bitset1.data(),
                                    (const uint64_t *)bitset2.data(),
                                    bitset1.size() / 64 * 64);
  EXPECT_FLOAT_EQ(result0, result1);
  EXPECT_FLOAT_EQ(result0, result2);
}

template <size_t M, size_t N>
void TestHamming32Matrix(void) {
  std::mt19937 gen((std::random_device())());

  const size_t batch_size = M;
  const size_t query_size = N;
  size_t count = (std::uniform_int_distribution<size_t>(1, 8192))(gen);
  size_t matrix_size = batch_size * count;
  size_t query_matrix_size = query_size * count;

  std::vector<uint32_t> matrix1(matrix_size);
  std::vector<uint32_t> matrix2(matrix_size);
  std::vector<uint32_t> query1(query_matrix_size);
  std::vector<uint32_t> query2(query_matrix_size);
  std::vector<float> result1(batch_size * query_size);
  std::vector<float> result2(batch_size * query_size);

  std::uniform_int_distribution<uint32_t> dist(0, 0xfffffffful);
  for (size_t i = 0; i < matrix_size; ++i) {
    matrix1[i] = dist(gen);
  }
  for (size_t i = 0; i < query_matrix_size; ++i) {
    query1[i] = dist(gen);
  }
  MatrixTranspose(&matrix2[0], matrix1.data(), count, batch_size);
  MatrixTranspose(&query2[0], query1.data(), count, query_size);

  for (size_t i = 0; i < query_size; ++i) {
    const uint32_t *cur_query = &query1[i * count];
    float *query_result = &result1[i * batch_size];

    for (size_t j = 0; j < batch_size; ++j) {
      HammingDistanceMatrix<uint32_t, 1, 1>::Compute(
          &matrix1[j * count], cur_query, count * 32, &query_result[j]);
    }
  }
  HammingDistanceMatrix<uint32_t, batch_size, query_size>::Compute(
      &matrix2[0], &query2[0], count * 32, &result2[0]);

  for (size_t i = 0; i < batch_size * query_size; ++i) {
    EXPECT_FLOAT_EQ(result1[i], result2[i]);
  }
}

template <size_t M, size_t N>
void TestHammingSquareRoot32Matrix(void) {
  std::mt19937 gen((std::random_device())());

  const size_t batch_size = M;
  const size_t query_size = N;
  size_t count = (std::uniform_int_distribution<size_t>(1, 8192))(gen);
  size_t matrix_size = batch_size * count;
  size_t query_matrix_size = query_size * count;

  std::vector<uint32_t> matrix1(matrix_size);
  std::vector<uint32_t> matrix2(matrix_size);
  std::vector<uint32_t> query1(query_matrix_size);
  std::vector<uint32_t> query2(query_matrix_size);
  std::vector<float> result1(batch_size * query_size);
  std::vector<float> result2(batch_size * query_size);

  std::uniform_int_distribution<uint32_t> dist(0, 0xfffffffful);
  for (size_t i = 0; i < matrix_size; ++i) {
    matrix1[i] = dist(gen);
  }
  for (size_t i = 0; i < query_matrix_size; ++i) {
    query1[i] = dist(gen);
  }
  MatrixTranspose(&matrix2[0], matrix1.data(), count, batch_size);
  MatrixTranspose(&query2[0], query1.data(), count, query_size);

  for (size_t i = 0; i < query_size; ++i) {
    const uint32_t *cur_query = &query1[i * count];
    float *query_result = &result1[i * batch_size];

    for (size_t j = 0; j < batch_size; ++j) {
      HammingSquareRootDistanceMatrix<uint32_t, 1, 1>::Compute(
          &matrix1[j * count], cur_query, count * 32, &query_result[j]);
    }
  }
  HammingSquareRootDistanceMatrix<uint32_t, batch_size, query_size>::Compute(
      &matrix2[0], &query2[0], count * 32, &result2[0]);

  for (size_t i = 0; i < batch_size * query_size; ++i) {
    EXPECT_FLOAT_EQ(result1[i], result2[i]);
  }
}

TEST(DistanceMatrix, Hamming32_1x1) {
  TestHamming32Matrix<1, 1>();
}

TEST(DistanceMatrix, Hamming32_2x1) {
  TestHamming32Matrix<2, 1>();
}

TEST(DistanceMatrix, Hamming32_2x2) {
  TestHamming32Matrix<2, 2>();
}

TEST(DistanceMatrix, Hamming32_3x3) {
  TestHamming32Matrix<3, 3>();
}

TEST(DistanceMatrix, Hamming32_4x1) {
  TestHamming32Matrix<4, 1>();
}

TEST(DistanceMatrix, Hamming32_4x2) {
  TestHamming32Matrix<4, 2>();
}

TEST(DistanceMatrix, Hamming32_4x4) {
  TestHamming32Matrix<4, 4>();
}

TEST(DistanceMatrix, Hamming32_8x1) {
  TestHamming32Matrix<8, 1>();
}

TEST(DistanceMatrix, Hamming32_8x2) {
  TestHamming32Matrix<8, 2>();
}

TEST(DistanceMatrix, Hamming32_8x4) {
  TestHamming32Matrix<8, 4>();
}

TEST(DistanceMatrix, Hamming32_8x8) {
  TestHamming32Matrix<8, 8>();
}

TEST(DistanceMatrix, Hamming32_16x1) {
  TestHamming32Matrix<16, 1>();
}

TEST(DistanceMatrix, Hamming32_16x2) {
  TestHamming32Matrix<16, 2>();
}

TEST(DistanceMatrix, Hamming32_16x4) {
  TestHamming32Matrix<16, 4>();
}

TEST(DistanceMatrix, Hamming32_16x8) {
  TestHamming32Matrix<16, 8>();
}

TEST(DistanceMatrix, Hamming32_16x16) {
  TestHamming32Matrix<16, 16>();
}

TEST(DistanceMatrix, Hamming32_32x1) {
  TestHamming32Matrix<32, 1>();
}

TEST(DistanceMatrix, Hamming32_32x2) {
  TestHamming32Matrix<32, 2>();
}

TEST(DistanceMatrix, Hamming32_32x4) {
  TestHamming32Matrix<32, 4>();
}

TEST(DistanceMatrix, Hamming32_32x8) {
  TestHamming32Matrix<32, 8>();
}

TEST(DistanceMatrix, Hamming32_32x16) {
  TestHamming32Matrix<32, 16>();
}

TEST(DistanceMatrix, Hamming32_32x32) {
  TestHamming32Matrix<32, 32>();
}

TEST(DistanceMatrix, Hamming32_64x1) {
  TestHamming32Matrix<64, 1>();
}

TEST(DistanceMatrix, Hamming32_64x2) {
  TestHamming32Matrix<64, 2>();
}

TEST(DistanceMatrix, Hamming32_64x4) {
  TestHamming32Matrix<64, 4>();
}

TEST(DistanceMatrix, Hamming32_64x8) {
  TestHamming32Matrix<64, 8>();
}

TEST(DistanceMatrix, Hamming32_64x16) {
  TestHamming32Matrix<64, 16>();
}

TEST(DistanceMatrix, Hamming32_64x32) {
  TestHamming32Matrix<64, 32>();
}

TEST(DistanceMatrix, Hamming32_64x64) {
  TestHamming32Matrix<64, 64>();
}

TEST(DistanceMatrix, Hamming32_128x1) {
  TestHamming32Matrix<128, 1>();
}

TEST(DistanceMatrix, Hamming32_128x2) {
  TestHamming32Matrix<128, 2>();
}

TEST(DistanceMatrix, Hamming32_128x4) {
  TestHamming32Matrix<128, 4>();
}

TEST(DistanceMatrix, Hamming32_128x8) {
  TestHamming32Matrix<128, 8>();
}

TEST(DistanceMatrix, Hamming32_128x16) {
  TestHamming32Matrix<128, 16>();
}

TEST(DistanceMatrix, Hamming32_128x32) {
  TestHamming32Matrix<128, 32>();
}

TEST(DistanceMatrix, Hamming32_128x64) {
  TestHamming32Matrix<128, 64>();
}

TEST(DistanceMatrix, Hamming32_128x128) {
  TestHamming32Matrix<128, 128>();
}

TEST(DistanceMatrix, HammingSquareRoot32_1x1) {
  TestHammingSquareRoot32Matrix<1, 1>();
}

TEST(DistanceMatrix, HammingSquareRoot32_2x1) {
  TestHammingSquareRoot32Matrix<2, 1>();
}

TEST(DistanceMatrix, HammingSquareRoot32_2x2) {
  TestHammingSquareRoot32Matrix<2, 2>();
}

TEST(DistanceMatrix, HammingSquareRoot32_3x3) {
  TestHammingSquareRoot32Matrix<3, 3>();
}

TEST(DistanceMatrix, HammingSquareRoot32_4x1) {
  TestHammingSquareRoot32Matrix<4, 1>();
}

TEST(DistanceMatrix, HammingSquareRoot32_4x2) {
  TestHammingSquareRoot32Matrix<4, 2>();
}

TEST(DistanceMatrix, HammingSquareRoot32_4x4) {
  TestHammingSquareRoot32Matrix<4, 4>();
}

TEST(DistanceMatrix, HammingSquareRoot32_8x1) {
  TestHammingSquareRoot32Matrix<8, 1>();
}

TEST(DistanceMatrix, HammingSquareRoot32_8x2) {
  TestHammingSquareRoot32Matrix<8, 2>();
}

TEST(DistanceMatrix, HammingSquareRoot32_8x4) {
  TestHammingSquareRoot32Matrix<8, 4>();
}

TEST(DistanceMatrix, HammingSquareRoot32_8x8) {
  TestHammingSquareRoot32Matrix<8, 8>();
}

TEST(DistanceMatrix, HammingSquareRoot32_16x1) {
  TestHammingSquareRoot32Matrix<16, 1>();
}

TEST(DistanceMatrix, HammingSquareRoot32_16x2) {
  TestHammingSquareRoot32Matrix<16, 2>();
}

TEST(DistanceMatrix, HammingSquareRoot32_16x4) {
  TestHammingSquareRoot32Matrix<16, 4>();
}

TEST(DistanceMatrix, HammingSquareRoot32_16x8) {
  TestHammingSquareRoot32Matrix<16, 8>();
}

TEST(DistanceMatrix, HammingSquareRoot32_16x16) {
  TestHammingSquareRoot32Matrix<16, 16>();
}

TEST(DistanceMatrix, HammingSquareRoot32_32x1) {
  TestHammingSquareRoot32Matrix<32, 1>();
}

TEST(DistanceMatrix, HammingSquareRoot32_32x2) {
  TestHammingSquareRoot32Matrix<32, 2>();
}

TEST(DistanceMatrix, HammingSquareRoot32_32x4) {
  TestHammingSquareRoot32Matrix<32, 4>();
}

TEST(DistanceMatrix, HammingSquareRoot32_32x8) {
  TestHammingSquareRoot32Matrix<32, 8>();
}

TEST(DistanceMatrix, HammingSquareRoot32_32x16) {
  TestHammingSquareRoot32Matrix<32, 16>();
}

TEST(DistanceMatrix, HammingSquareRoot32_32x32) {
  TestHammingSquareRoot32Matrix<32, 32>();
}

TEST(DistanceMatrix, HammingSquareRoot32_64x1) {
  TestHammingSquareRoot32Matrix<64, 1>();
}

TEST(DistanceMatrix, HammingSquareRoot32_64x2) {
  TestHammingSquareRoot32Matrix<64, 2>();
}

TEST(DistanceMatrix, HammingSquareRoot32_64x4) {
  TestHammingSquareRoot32Matrix<64, 4>();
}

TEST(DistanceMatrix, HammingSquareRoot32_64x8) {
  TestHammingSquareRoot32Matrix<64, 8>();
}

TEST(DistanceMatrix, HammingSquareRoot32_64x16) {
  TestHammingSquareRoot32Matrix<64, 16>();
}

TEST(DistanceMatrix, HammingSquareRoot32_64x32) {
  TestHammingSquareRoot32Matrix<64, 32>();
}

TEST(DistanceMatrix, HammingSquareRoot32_64x64) {
  TestHammingSquareRoot32Matrix<64, 64>();
}

TEST(DistanceMatrix, HammingSquareRoot32_128x1) {
  TestHammingSquareRoot32Matrix<128, 1>();
}

TEST(DistanceMatrix, HammingSquareRoot32_128x2) {
  TestHammingSquareRoot32Matrix<128, 2>();
}

TEST(DistanceMatrix, HammingSquareRoot32_128x4) {
  TestHammingSquareRoot32Matrix<128, 4>();
}

TEST(DistanceMatrix, HammingSquareRoot32_128x8) {
  TestHammingSquareRoot32Matrix<128, 8>();
}

TEST(DistanceMatrix, HammingSquareRoot32_128x16) {
  TestHammingSquareRoot32Matrix<128, 16>();
}

TEST(DistanceMatrix, HammingSquareRoot32_128x32) {
  TestHammingSquareRoot32Matrix<128, 32>();
}

TEST(DistanceMatrix, HammingSquareRoot32_128x64) {
  TestHammingSquareRoot32Matrix<128, 64>();
}

TEST(DistanceMatrix, HammingSquareRoot32_128x128) {
  TestHammingSquareRoot32Matrix<128, 128>();
}

template <size_t M, size_t N, size_t B, size_t D>
void Hamming32Benchmark(void) {
  const size_t count = D;
  const size_t batch_size = M;
  const size_t block_size = B;
  const size_t query_size = N;
  const size_t matrix_size = block_size * batch_size * count;
  const size_t query_matrix_size = count * query_size;

  std::vector<uint32_t> matrix1(matrix_size);
  std::vector<uint32_t> matrix2(matrix_size);
  std::vector<uint32_t> query1(query_matrix_size);
  std::vector<uint32_t> query2(query_matrix_size);

  std::mt19937 gen((std::random_device())());
  std::uniform_int_distribution<uint32_t> dist(0, 0xfffffffful);
  for (size_t i = 0; i < matrix_size; ++i) {
    matrix1[i] = dist(gen);
  }
  for (size_t i = 0; i < query_matrix_size; ++i) {
    query1[i] = dist(gen);
  }

  for (size_t i = 0; i < block_size; ++i) {
    size_t start_pos = i * batch_size * count;
    MatrixTranspose(&matrix2[start_pos], &matrix1[start_pos], count,
                    batch_size);
  }
  MatrixTranspose(&query2[0], query1.data(), count, query_size);

  ElapsedTime elapsed_time;
  float results[batch_size * query_size];

  std::cout << "# (" << IntelIntrinsics() << ") UINT32 " << count << "d, "
            << batch_size << " * " << query_size << " * " << block_size
            << std::endl;

  // 1 Batched Hamming
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const uint32_t *matrix_batch = &matrix2[i * batch_size * count];

    for (size_t j = 0; j < query_size; ++j) {
      const uint32_t *current_query = &query1[j * count];
      float *current_results = &results[j * batch_size];

      HammingDistanceMatrix<uint32_t, batch_size, 1>::Compute(
          matrix_batch, current_query, count * 32, current_results);
    }
  }
  std::cout << "* 1 Batched Hamming (us) \t" << elapsed_time.micro_seconds()
            << std::endl;

  // N Batched Hamming
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const uint32_t *matrix_batch = &matrix2[i * batch_size * count];

    HammingDistanceMatrix<uint32_t, batch_size, query_size>::Compute(
        matrix_batch, &query2[0], count * 32, results);
  }
  std::cout << "* N Batched Hamming (us) \t" << elapsed_time.micro_seconds()
            << std::endl;

  // Unbatched Hamming
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const uint32_t *matrix_batch = &matrix1[i * batch_size * count];

    for (size_t j = 0; j < query_size; ++j) {
      const uint32_t *current_query = &query1[j * count];
      float *current_results = &results[j * batch_size];

      for (size_t k = 0; k < batch_size; ++k) {
        HammingDistanceMatrix<uint32_t, 1, 1>::Compute(
            &matrix_batch[k * count], current_query, count * 32,
            &current_results[k]);
      }
    }
  }
  std::cout << "* Unbatched Hamming (us) \t" << elapsed_time.micro_seconds()
            << std::endl;
}

TEST(DistanceMatrix, DISABLED_Hamming32_Benchmark) {
  Hamming32Benchmark<2, 1, 512, 64>();
  Hamming32Benchmark<2, 2, 512, 64>();
  Hamming32Benchmark<4, 1, 2048, 16>();
  Hamming32Benchmark<4, 2, 2048, 16>();
  Hamming32Benchmark<4, 4, 2048, 16>();
  Hamming32Benchmark<8, 1, 512, 64>();
  Hamming32Benchmark<8, 2, 512, 64>();
  Hamming32Benchmark<8, 4, 512, 64>();
  Hamming32Benchmark<8, 8, 512, 64>();
  Hamming32Benchmark<16, 1, 512, 64>();
  Hamming32Benchmark<16, 2, 512, 64>();
  Hamming32Benchmark<16, 4, 512, 64>();
  Hamming32Benchmark<16, 8, 512, 64>();
  Hamming32Benchmark<16, 16, 512, 64>();
  Hamming32Benchmark<32, 1, 512, 64>();
  Hamming32Benchmark<32, 2, 512, 64>();
  Hamming32Benchmark<32, 4, 512, 64>();
  Hamming32Benchmark<32, 8, 512, 64>();
  Hamming32Benchmark<32, 16, 512, 64>();
  Hamming32Benchmark<32, 32, 512, 64>();
  Hamming32Benchmark<64, 1, 512, 64>();
  Hamming32Benchmark<64, 2, 512, 64>();
  Hamming32Benchmark<64, 4, 512, 64>();
  Hamming32Benchmark<64, 8, 512, 64>();
  Hamming32Benchmark<128, 1, 512, 64>();
}

#if defined(AILEGO_M64)
template <size_t M, size_t N>
void TestHamming64Matrix(void) {
  std::mt19937 gen((std::random_device())());

  const size_t batch_size = M;
  const size_t query_size = N;
  size_t count = (std::uniform_int_distribution<size_t>(1, 512))(gen);
  size_t matrix_size = batch_size * count;
  size_t query_matrix_size = query_size * count;

  std::vector<uint64_t> matrix1(matrix_size);
  std::vector<uint64_t> matrix2(matrix_size);
  std::vector<uint64_t> query1(query_matrix_size);
  std::vector<uint64_t> query2(query_matrix_size);
  std::vector<float> result1(batch_size * query_size);
  std::vector<float> result2(batch_size * query_size);

  std::uniform_int_distribution<uint64_t> dist(0, 0x7fffffffffffffffull);
  for (size_t i = 0; i < matrix_size; ++i) {
    matrix1[i] = dist(gen);
  }
  for (size_t i = 0; i < query_matrix_size; ++i) {
    query1[i] = dist(gen);
  }
  MatrixTranspose(&matrix2[0], matrix1.data(), count, batch_size);
  MatrixTranspose(&query2[0], query1.data(), count, query_size);

  for (size_t i = 0; i < query_size; ++i) {
    const uint64_t *cur_query = &query1[i * count];
    float *query_result = &result1[i * batch_size];

    for (size_t j = 0; j < batch_size; ++j) {
      HammingDistanceMatrix<uint64_t, 1, 1>::Compute(
          &matrix1[j * count], cur_query, count * 64, &query_result[j]);
    }
  }
  HammingDistanceMatrix<uint64_t, batch_size, query_size>::Compute(
      &matrix2[0], &query2[0], count * 64, &result2[0]);

  for (size_t i = 0; i < batch_size * query_size; ++i) {
    EXPECT_FLOAT_EQ(result1[i], result2[i]);
  }
}

template <size_t M, size_t N>
void TestHammingSquareRoot64Matrix(void) {
  std::mt19937 gen((std::random_device())());

  const size_t batch_size = M;
  const size_t query_size = N;
  size_t count = (std::uniform_int_distribution<size_t>(1, 512))(gen);
  size_t matrix_size = batch_size * count;
  size_t query_matrix_size = query_size * count;

  std::vector<uint64_t> matrix1(matrix_size);
  std::vector<uint64_t> matrix2(matrix_size);
  std::vector<uint64_t> query1(query_matrix_size);
  std::vector<uint64_t> query2(query_matrix_size);
  std::vector<float> result1(batch_size * query_size);
  std::vector<float> result2(batch_size * query_size);

  std::uniform_int_distribution<uint64_t> dist(0, 0x7fffffffffffffffull);
  for (size_t i = 0; i < matrix_size; ++i) {
    matrix1[i] = dist(gen);
  }
  for (size_t i = 0; i < query_matrix_size; ++i) {
    query1[i] = dist(gen);
  }
  MatrixTranspose(&matrix2[0], matrix1.data(), count, batch_size);
  MatrixTranspose(&query2[0], query1.data(), count, query_size);

  for (size_t i = 0; i < query_size; ++i) {
    const uint64_t *cur_query = &query1[i * count];
    float *query_result = &result1[i * batch_size];

    for (size_t j = 0; j < batch_size; ++j) {
      HammingSquareRootDistanceMatrix<uint64_t, 1, 1>::Compute(
          &matrix1[j * count], cur_query, count * 64, &query_result[j]);
    }
  }
  HammingSquareRootDistanceMatrix<uint64_t, batch_size, query_size>::Compute(
      &matrix2[0], &query2[0], count * 64, &result2[0]);

  for (size_t i = 0; i < batch_size * query_size; ++i) {
    EXPECT_FLOAT_EQ(result1[i], result2[i]);
  }
}

TEST(DistanceMatrix, Hamming64_1x1) {
  TestHamming64Matrix<1, 1>();
}

TEST(DistanceMatrix, Hamming64_2x1) {
  TestHamming64Matrix<2, 1>();
}

TEST(DistanceMatrix, Hamming64_2x2) {
  TestHamming64Matrix<2, 2>();
}

TEST(DistanceMatrix, Hamming64_3x3) {
  TestHamming64Matrix<3, 3>();
}

TEST(DistanceMatrix, Hamming64_4x1) {
  TestHamming64Matrix<4, 1>();
}

TEST(DistanceMatrix, Hamming64_4x2) {
  TestHamming64Matrix<4, 2>();
}

TEST(DistanceMatrix, Hamming64_4x4) {
  TestHamming64Matrix<4, 4>();
}

TEST(DistanceMatrix, Hamming64_8x1) {
  TestHamming64Matrix<8, 1>();
}

TEST(DistanceMatrix, Hamming64_8x2) {
  TestHamming64Matrix<8, 2>();
}

TEST(DistanceMatrix, Hamming64_8x4) {
  TestHamming64Matrix<8, 4>();
}

TEST(DistanceMatrix, Hamming64_8x8) {
  TestHamming64Matrix<8, 8>();
}

TEST(DistanceMatrix, Hamming64_16x1) {
  TestHamming64Matrix<16, 1>();
}

TEST(DistanceMatrix, Hamming64_16x2) {
  TestHamming64Matrix<16, 2>();
}

TEST(DistanceMatrix, Hamming64_16x4) {
  TestHamming64Matrix<16, 4>();
}

TEST(DistanceMatrix, Hamming64_16x8) {
  TestHamming64Matrix<16, 8>();
}

TEST(DistanceMatrix, Hamming64_16x16) {
  TestHamming64Matrix<16, 16>();
}

TEST(DistanceMatrix, Hamming64_32x1) {
  TestHamming64Matrix<32, 1>();
}

TEST(DistanceMatrix, Hamming64_32x2) {
  TestHamming64Matrix<32, 2>();
}

TEST(DistanceMatrix, Hamming64_32x4) {
  TestHamming64Matrix<32, 4>();
}

TEST(DistanceMatrix, Hamming64_32x8) {
  TestHamming64Matrix<32, 8>();
}

TEST(DistanceMatrix, Hamming64_32x16) {
  TestHamming64Matrix<32, 16>();
}

TEST(DistanceMatrix, Hamming64_32x32) {
  TestHamming64Matrix<32, 32>();
}

TEST(DistanceMatrix, Hamming64_64x1) {
  TestHamming64Matrix<64, 1>();
}

TEST(DistanceMatrix, Hamming64_64x2) {
  TestHamming64Matrix<64, 2>();
}

TEST(DistanceMatrix, Hamming64_64x4) {
  TestHamming64Matrix<64, 4>();
}

TEST(DistanceMatrix, Hamming64_64x8) {
  TestHamming64Matrix<64, 8>();
}

TEST(DistanceMatrix, Hamming64_64x16) {
  TestHamming64Matrix<64, 16>();
}

TEST(DistanceMatrix, Hamming64_64x32) {
  TestHamming64Matrix<64, 32>();
}

TEST(DistanceMatrix, Hamming64_64x64) {
  TestHamming64Matrix<64, 64>();
}

TEST(DistanceMatrix, Hamming64_128x1) {
  TestHamming64Matrix<128, 1>();
}

TEST(DistanceMatrix, Hamming64_128x2) {
  TestHamming64Matrix<128, 2>();
}

TEST(DistanceMatrix, Hamming64_128x4) {
  TestHamming64Matrix<128, 4>();
}

TEST(DistanceMatrix, Hamming64_128x8) {
  TestHamming64Matrix<128, 8>();
}

TEST(DistanceMatrix, Hamming64_128x16) {
  TestHamming64Matrix<128, 16>();
}

TEST(DistanceMatrix, Hamming64_128x32) {
  TestHamming64Matrix<128, 32>();
}

TEST(DistanceMatrix, Hamming64_128x64) {
  TestHamming64Matrix<128, 64>();
}

TEST(DistanceMatrix, Hamming64_128x128) {
  TestHamming64Matrix<128, 128>();
}

TEST(DistanceMatrix, HammingSquareRoot64_1x1) {
  TestHammingSquareRoot64Matrix<1, 1>();
}

TEST(DistanceMatrix, HammingSquareRoot64_2x1) {
  TestHammingSquareRoot64Matrix<2, 1>();
}

TEST(DistanceMatrix, HammingSquareRoot64_2x2) {
  TestHammingSquareRoot64Matrix<2, 2>();
}

TEST(DistanceMatrix, HammingSquareRoot64_3x3) {
  TestHammingSquareRoot64Matrix<3, 3>();
}

TEST(DistanceMatrix, HammingSquareRoot64_4x1) {
  TestHammingSquareRoot64Matrix<4, 1>();
}

TEST(DistanceMatrix, HammingSquareRoot64_4x2) {
  TestHammingSquareRoot64Matrix<4, 2>();
}

TEST(DistanceMatrix, HammingSquareRoot64_4x4) {
  TestHammingSquareRoot64Matrix<4, 4>();
}

TEST(DistanceMatrix, HammingSquareRoot64_8x1) {
  TestHammingSquareRoot64Matrix<8, 1>();
}

TEST(DistanceMatrix, HammingSquareRoot64_8x2) {
  TestHammingSquareRoot64Matrix<8, 2>();
}

TEST(DistanceMatrix, HammingSquareRoot64_8x4) {
  TestHammingSquareRoot64Matrix<8, 4>();
}

TEST(DistanceMatrix, HammingSquareRoot64_8x8) {
  TestHammingSquareRoot64Matrix<8, 8>();
}

TEST(DistanceMatrix, HammingSquareRoot64_16x1) {
  TestHammingSquareRoot64Matrix<16, 1>();
}

TEST(DistanceMatrix, HammingSquareRoot64_16x2) {
  TestHammingSquareRoot64Matrix<16, 2>();
}

TEST(DistanceMatrix, HammingSquareRoot64_16x4) {
  TestHammingSquareRoot64Matrix<16, 4>();
}

TEST(DistanceMatrix, HammingSquareRoot64_16x8) {
  TestHammingSquareRoot64Matrix<16, 8>();
}

TEST(DistanceMatrix, HammingSquareRoot64_16x16) {
  TestHammingSquareRoot64Matrix<16, 16>();
}

TEST(DistanceMatrix, HammingSquareRoot64_32x1) {
  TestHammingSquareRoot64Matrix<32, 1>();
}

TEST(DistanceMatrix, HammingSquareRoot64_32x2) {
  TestHammingSquareRoot64Matrix<32, 2>();
}

TEST(DistanceMatrix, HammingSquareRoot64_32x4) {
  TestHammingSquareRoot64Matrix<32, 4>();
}

TEST(DistanceMatrix, HammingSquareRoot64_32x8) {
  TestHammingSquareRoot64Matrix<32, 8>();
}

TEST(DistanceMatrix, HammingSquareRoot64_32x16) {
  TestHammingSquareRoot64Matrix<32, 16>();
}

TEST(DistanceMatrix, HammingSquareRoot64_32x32) {
  TestHammingSquareRoot64Matrix<32, 32>();
}

TEST(DistanceMatrix, HammingSquareRoot64_64x1) {
  TestHammingSquareRoot64Matrix<64, 1>();
}

TEST(DistanceMatrix, HammingSquareRoot64_64x2) {
  TestHammingSquareRoot64Matrix<64, 2>();
}

TEST(DistanceMatrix, HammingSquareRoot64_64x4) {
  TestHammingSquareRoot64Matrix<64, 4>();
}

TEST(DistanceMatrix, HammingSquareRoot64_64x8) {
  TestHammingSquareRoot64Matrix<64, 8>();
}

TEST(DistanceMatrix, HammingSquareRoot64_64x16) {
  TestHammingSquareRoot64Matrix<64, 16>();
}

TEST(DistanceMatrix, HammingSquareRoot64_64x32) {
  TestHammingSquareRoot64Matrix<64, 32>();
}

TEST(DistanceMatrix, HammingSquareRoot64_64x64) {
  TestHammingSquareRoot64Matrix<64, 64>();
}

TEST(DistanceMatrix, HammingSquareRoot64_128x1) {
  TestHammingSquareRoot64Matrix<128, 1>();
}

TEST(DistanceMatrix, HammingSquareRoot64_128x2) {
  TestHammingSquareRoot64Matrix<128, 2>();
}

TEST(DistanceMatrix, HammingSquareRoot64_128x4) {
  TestHammingSquareRoot64Matrix<128, 4>();
}

TEST(DistanceMatrix, HammingSquareRoot64_128x8) {
  TestHammingSquareRoot64Matrix<128, 8>();
}

TEST(DistanceMatrix, HammingSquareRoot64_128x16) {
  TestHammingSquareRoot64Matrix<128, 16>();
}

TEST(DistanceMatrix, HammingSquareRoot64_128x32) {
  TestHammingSquareRoot64Matrix<128, 32>();
}

TEST(DistanceMatrix, HammingSquareRoot64_128x64) {
  TestHammingSquareRoot64Matrix<128, 64>();
}

TEST(DistanceMatrix, HammingSquareRoot64_128x128) {
  TestHammingSquareRoot64Matrix<128, 128>();
}

template <size_t M, size_t N, size_t B, size_t D>
void Hamming64Benchmark(void) {
  const size_t count = D;
  const size_t batch_size = M;
  const size_t block_size = B;
  const size_t query_size = N;
  const size_t matrix_size = block_size * batch_size * count;
  const size_t query_matrix_size = count * query_size;

  std::vector<uint64_t> matrix1(matrix_size);
  std::vector<uint64_t> matrix2(matrix_size);
  std::vector<uint64_t> query1(query_matrix_size);
  std::vector<uint64_t> query2(query_matrix_size);

  std::mt19937 gen((std::random_device())());
  std::uniform_int_distribution<uint32_t> dist(0, 0x7ffffffful);
  for (size_t i = 0; i < matrix_size; ++i) {
    matrix1[i] = dist(gen);
  }
  for (size_t i = 0; i < query_matrix_size; ++i) {
    query1[i] = dist(gen);
  }

  for (size_t i = 0; i < block_size; ++i) {
    size_t start_pos = i * batch_size * count;
    MatrixTranspose(&matrix2[start_pos], &matrix1[start_pos], count,
                    batch_size);
  }
  MatrixTranspose(&query2[0], query1.data(), count, query_size);

  ElapsedTime elapsed_time;
  float results[batch_size * query_size];

  std::cout << "# (" << IntelIntrinsics() << ") UINT64 " << count << "d, "
            << batch_size << " * " << query_size << " * " << block_size
            << std::endl;

  // 1 Batched Hamming
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const uint64_t *matrix_batch = &matrix2[i * batch_size * count];

    for (size_t j = 0; j < query_size; ++j) {
      const uint64_t *current_query = &query1[j * count];
      float *current_results = &results[j * batch_size];

      HammingDistanceMatrix<uint64_t, batch_size, 1>::Compute(
          matrix_batch, current_query, count * 64, current_results);
    }
  }
  std::cout << "* 1 Batched Hamming (us) \t" << elapsed_time.micro_seconds()
            << std::endl;

  // N Batched Hamming
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const uint64_t *matrix_batch = &matrix2[i * batch_size * count];

    HammingDistanceMatrix<uint64_t, batch_size, query_size>::Compute(
        matrix_batch, &query2[0], count * 64, results);
  }
  std::cout << "* N Batched Hamming (us) \t" << elapsed_time.micro_seconds()
            << std::endl;

  // Unbatched Hamming
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const uint64_t *matrix_batch = &matrix1[i * batch_size * count];

    for (size_t j = 0; j < query_size; ++j) {
      const uint64_t *current_query = &query1[j * count];
      float *current_results = &results[j * batch_size];

      for (size_t k = 0; k < batch_size; ++k) {
        HammingDistanceMatrix<uint64_t, 1, 1>::Compute(
            &matrix_batch[k * count], current_query, count * 64,
            &current_results[k]);
      }
    }
  }
  std::cout << "* Unbatched Hamming (us) \t" << elapsed_time.micro_seconds()
            << std::endl;
}

TEST(DistanceMatrix, DISABLED_Hamming64_Benchmark) {
  Hamming64Benchmark<2, 1, 512, 64>();
  Hamming64Benchmark<2, 2, 512, 64>();
  Hamming64Benchmark<4, 1, 2048, 16>();
  Hamming64Benchmark<4, 2, 2048, 16>();
  Hamming64Benchmark<4, 4, 2048, 16>();
  Hamming64Benchmark<8, 1, 512, 64>();
  Hamming64Benchmark<8, 2, 512, 64>();
  Hamming64Benchmark<8, 4, 512, 64>();
  Hamming64Benchmark<8, 8, 512, 64>();
  Hamming64Benchmark<16, 1, 512, 64>();
  Hamming64Benchmark<16, 2, 512, 64>();
  Hamming64Benchmark<16, 4, 512, 64>();
  Hamming64Benchmark<16, 8, 512, 64>();
  Hamming64Benchmark<16, 16, 512, 64>();
  Hamming64Benchmark<32, 1, 512, 64>();
  Hamming64Benchmark<32, 2, 512, 64>();
  Hamming64Benchmark<32, 4, 512, 64>();
  Hamming64Benchmark<32, 8, 512, 64>();
  Hamming64Benchmark<32, 16, 512, 64>();
  Hamming64Benchmark<32, 32, 512, 64>();
  Hamming64Benchmark<64, 1, 512, 64>();
  Hamming64Benchmark<64, 2, 512, 64>();
  Hamming64Benchmark<64, 4, 512, 64>();
  Hamming64Benchmark<64, 8, 512, 64>();
  Hamming64Benchmark<128, 1, 512, 64>();
}
#endif  // AILEGO_M64
