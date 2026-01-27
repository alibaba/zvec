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
#include <ailego/math/norm_matrix.h>
#include <gtest/gtest.h>
#include <zvec/ailego/container/vector.h>
#include <zvec/ailego/utility/time_helper.h>

using namespace zvec::ailego;

static inline const char *IntelIntrinsics(void) {
  return internal::CpuFeatures::Intrinsics();
}

static inline void MatrixTranspose(uint32_t *dst, const uint32_t *src, size_t M,
                                   size_t N) {
  for (size_t n = 0; n < N * M; n++) {
    size_t i = n / N;
    size_t j = n % N;
    dst[n] = src[M * j + i];
  }
}

static float Norm1(const std::vector<int8_t> &vec) {
  float out = 0.0f;
  Norm1Matrix<int8_t, 1>::Compute(vec.data(), vec.size(), &out);
  return out;
}

static float Norm2(const std::vector<int8_t> &vec) {
  float out = 0.0f;
  Norm2Matrix<int8_t, 1>::Compute(vec.data(), vec.size(), &out);
  return out;
}

TEST(NormMatrix, Norm1_General) {
  std::mt19937 gen((std::random_device())());
  std::uniform_int_distribution<int> dist(-127, 127);

  for (size_t d = 1; d < 100; ++d) {
    std::vector<int8_t> vec;
    float result = 0.0f;
    for (size_t i = 0; i < d; ++i) {
      int8_t val = (int8_t)dist(gen);
      result += std::abs(val);
      vec.push_back(val);
    }
    EXPECT_FLOAT_EQ(result, Norm1(vec));
  }
}

TEST(NormMatrix, Norm2_General) {
  std::mt19937 gen((std::random_device())());
  std::uniform_int_distribution<int> dist(-127, 127);

  for (size_t d = 1; d < 100; ++d) {
    std::vector<int8_t> vec;
    float result = 0.0f;
    for (size_t i = 0; i < d; ++i) {
      int8_t val = (int8_t)dist(gen);
      result += val * val;
      vec.push_back(val);
    }
    EXPECT_FLOAT_EQ(std::sqrt(result), Norm2(vec));
  }
}

template <size_t M>
void TestNorm1Matrix(void) {
  std::mt19937 gen((std::random_device())());

  const size_t batch_size = M;
  size_t dimension = (std::uniform_int_distribution<size_t>(1, 65))(gen) << 2;
  size_t matrix_size = batch_size * dimension;

  std::vector<int8_t> matrix1(matrix_size);
  std::vector<int8_t> matrix2(matrix_size);
  std::vector<float> result1(batch_size);
  std::vector<float> result2(batch_size);

  std::uniform_int_distribution<int> dist(-127, 127);
  for (size_t i = 0; i < matrix_size; ++i) {
    matrix1[i] = (int8_t)dist(gen);
  }
  MatrixTranspose((uint32_t *)&matrix2[0], (const uint32_t *)matrix1.data(),
                  dimension / 4, batch_size);

  for (size_t j = 0; j < batch_size; ++j) {
    Norm1Matrix<int8_t, 1>::Compute(&matrix1[j * dimension], dimension,
                                    &result1[j]);
  }
  Norm1Matrix<int8_t, batch_size>::Compute(&matrix2[0], dimension, &result2[0]);

  for (size_t i = 0; i < batch_size; ++i) {
    EXPECT_FLOAT_EQ(result1[i], result2[i]);
  }
}

template <size_t M>
void TestNorm2Matrix(void) {
  std::mt19937 gen((std::random_device())());

  const size_t batch_size = M;
  size_t dimension = (std::uniform_int_distribution<size_t>(1, 65))(gen) << 2;
  size_t matrix_size = batch_size * dimension;

  std::vector<int8_t> matrix1(matrix_size);
  std::vector<int8_t> matrix2(matrix_size);
  std::vector<float> result1(batch_size);
  std::vector<float> result2(batch_size);

  std::uniform_int_distribution<int> dist(-127, 127);
  for (size_t i = 0; i < matrix_size; ++i) {
    matrix1[i] = (int8_t)dist(gen);
  }
  MatrixTranspose((uint32_t *)&matrix2[0], (const uint32_t *)matrix1.data(),
                  dimension / 4, batch_size);

  for (size_t j = 0; j < batch_size; ++j) {
    Norm2Matrix<int8_t, 1>::Compute(&matrix1[j * dimension], dimension,
                                    &result1[j]);
  }
  Norm2Matrix<int8_t, batch_size>::Compute(&matrix2[0], dimension, &result2[0]);

  for (size_t i = 0; i < batch_size; ++i) {
    EXPECT_FLOAT_EQ(result1[i], result2[i]);
  }
}

TEST(NormMatrix, Norm1_Matrix) {
  TestNorm1Matrix<1>();
  TestNorm1Matrix<3>();
  TestNorm1Matrix<4>();
  TestNorm1Matrix<8>();
  TestNorm1Matrix<10>();
  TestNorm1Matrix<12>();
  TestNorm1Matrix<16>();
  TestNorm1Matrix<29>();
  TestNorm1Matrix<32>();
  TestNorm1Matrix<38>();
  TestNorm1Matrix<40>();
  TestNorm1Matrix<51>();
  TestNorm1Matrix<64>();
  TestNorm1Matrix<65>();
}

TEST(NormMatrix, Norm2_Matrix) {
  TestNorm2Matrix<1>();
  TestNorm2Matrix<3>();
  TestNorm2Matrix<4>();
  TestNorm2Matrix<8>();
  TestNorm2Matrix<10>();
  TestNorm2Matrix<12>();
  TestNorm2Matrix<16>();
  TestNorm2Matrix<29>();
  TestNorm2Matrix<32>();
  TestNorm2Matrix<38>();
  TestNorm2Matrix<40>();
  TestNorm2Matrix<51>();
  TestNorm2Matrix<64>();
  TestNorm2Matrix<65>();
}

template <size_t M, size_t B, size_t D>
void Norm1Benchmark(void) {
  const size_t dimension = D;
  const size_t batch_size = M;
  const size_t block_size = B;
  const size_t matrix_size = block_size * batch_size * dimension;

  std::vector<int8_t> matrix1(matrix_size);
  std::vector<int8_t> matrix2(matrix_size);

  std::mt19937 gen((std::random_device())());
  std::uniform_int_distribution<int> dist(-127, 127);
  for (size_t i = 0; i < matrix_size; ++i) {
    matrix1[i] = (int8_t)dist(gen);
  }

  for (size_t i = 0; i < block_size; ++i) {
    size_t start_pos = i * batch_size * dimension;
    MatrixTranspose((uint32_t *)&matrix2[start_pos],
                    (const uint32_t *)&matrix1[start_pos], dimension / 4,
                    batch_size);
  }

  ElapsedTime elapsed_time;
  float results[batch_size];

  std::cout << "# (" << IntelIntrinsics() << ") INT8 " << dimension << "d, "
            << batch_size << " * " << block_size << std::endl;

  // 1 Batched Norm1
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const int8_t *matrix_batch = &matrix2[i * batch_size * dimension];
    Norm1Matrix<int8_t, batch_size>::Compute(matrix_batch, dimension,
                                             &results[0]);
  }
  std::cout << "* Batched Norm1 (us) \t" << elapsed_time.micro_seconds()
            << std::endl;

  // Unbatched Norm1
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const int8_t *matrix_batch = &matrix1[i * batch_size * dimension];
    for (size_t k = 0; k < batch_size; ++k) {
      Norm1Matrix<int8_t, 1>::Compute(&matrix_batch[k * dimension], dimension,
                                      &results[k]);
    }
  }
  std::cout << "* Unbatched Norm1 (us) \t" << elapsed_time.micro_seconds()
            << std::endl;
}

template <size_t M, size_t B, size_t D>
void Norm2Benchmark(void) {
  const size_t dimension = D;
  const size_t batch_size = M;
  const size_t block_size = B;
  const size_t matrix_size = block_size * batch_size * dimension;

  std::vector<int8_t> matrix1(matrix_size);
  std::vector<int8_t> matrix2(matrix_size);

  std::mt19937 gen((std::random_device())());
  std::uniform_int_distribution<int> dist(-127, 127);
  for (size_t i = 0; i < matrix_size; ++i) {
    matrix1[i] = (int8_t)dist(gen);
  }

  for (size_t i = 0; i < block_size; ++i) {
    size_t start_pos = i * batch_size * dimension;
    MatrixTranspose((uint32_t *)&matrix2[start_pos],
                    (const uint32_t *)&matrix1[start_pos], dimension / 4,
                    batch_size);
  }

  ElapsedTime elapsed_time;
  float results[batch_size];

  std::cout << "# (" << IntelIntrinsics() << ") INT8 " << dimension << "d, "
            << batch_size << " * " << block_size << std::endl;

  // 1 Batched Norm2
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const int8_t *matrix_batch = &matrix2[i * batch_size * dimension];
    Norm2Matrix<int8_t, batch_size>::Compute(matrix_batch, dimension,
                                             &results[0]);
  }
  std::cout << "* Batched Norm2 (us) \t" << elapsed_time.micro_seconds()
            << std::endl;

  // Unbatched Norm2
  elapsed_time.reset();
  for (size_t i = 0; i < block_size; ++i) {
    const int8_t *matrix_batch = &matrix1[i * batch_size * dimension];
    for (size_t k = 0; k < batch_size; ++k) {
      Norm2Matrix<int8_t, 1>::Compute(&matrix_batch[k * dimension], dimension,
                                      &results[k]);
    }
  }
  std::cout << "* Unbatched Norm2 (us) \t" << elapsed_time.micro_seconds()
            << std::endl;
}

TEST(NormMatrix, DISABLED_Norm1_Benchmark) {
  Norm1Benchmark<2, 512, 128>();
  Norm1Benchmark<4, 512, 128>();
  Norm1Benchmark<8, 512, 128>();
  Norm1Benchmark<16, 512, 128>();
  Norm1Benchmark<32, 512, 128>();
  Norm1Benchmark<64, 512, 128>();
}

TEST(NormMatrix, DISABLED_Norm2_Benchmark) {
  Norm2Benchmark<2, 512, 128>();
  Norm2Benchmark<4, 512, 128>();
  Norm2Benchmark<8, 512, 128>();
  Norm2Benchmark<16, 512, 128>();
  Norm2Benchmark<32, 512, 128>();
  Norm2Benchmark<64, 512, 128>();
}
