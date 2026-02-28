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
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <ailego/math/distance.h>
#include <ailego/math/norm_matrix.h>
#include <ailego/math/normalizer.h>
#include <gtest/gtest.h>
#include <zvec/ailego/container/vector.h>
#include <zvec/ailego/utility/float_helper.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_flow.h>
#include "core/quantizer/quantizer_params.h"
#include "zvec/core/framework/index_factory.h"


using namespace zvec;
using namespace zvec::core;
using namespace zvec::ailego;

static IndexHolder::Pointer GetHolder(
    size_t dim, size_t count, std::uniform_real_distribution<float> &dist) {
  std::random_device rd;
  std::mt19937 gen(rd());
  auto holder = std::make_shared<MultiPassIndexHolder<IndexMeta::DT_FP32>>(dim);
  for (size_t i = 0; i < count; ++i) {
    ailego::NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = dist(gen);
    }
    holder->emplace(i + 1, vec);
  }
  return holder;
}

static inline void MatrixTranspose(uint32_t *dst, const uint32_t *src, size_t M,
                                   size_t N) {
  for (size_t n = 0; n < N * M; n++) {
    size_t i = n / N;
    size_t j = n % N;
    dst[n] = src[M * j + i];
  }
}

//! Test whether two floating point numbers are equal
template <class T>
static inline auto IsAlmostEqual(const T &x, const T &y, int ulp) ->
    typename std::enable_if<std::is_floating_point<T>::value, bool>::type {
  // the machine epsilon has to be scaled to the magnitude of the values used
  // and multiplied by the desired precision in ULPs (units in the last place)
  return ((std::fabs(x - y) <=
           std::numeric_limits<T>::epsilon() * std::fabs(x + y) * ulp) ||
          (std::fabs(x - y) < std::numeric_limits<T>::min()));
}

TEST(QuantizedIntegerMetric, General) {
  auto metric = IndexFactory::CreateMetric("MipsSquaredEuclidean");
  ASSERT_TRUE(metric);

  Params params;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0, 1.0);
  const size_t DIMENSION = 21;
  ailego::NumericalVector<float> x(DIMENSION);
  ailego::NumericalVector<float> X(DIMENSION);
  ailego::NumericalVector<float> y(DIMENSION);
  ailego::NumericalVector<float> Y(DIMENSION);
  float xa = dist(gen);
  float xb = dist(gen);
  float ya = dist(gen);
  float yb = dist(gen);
  float x2 = 0, x1 = 0, y2 = 0, y1 = 0;
  float X2 = 0;
  float xx2 = 0;
  for (size_t j = 0; j < DIMENSION; ++j) {
    x[j] = dist(gen);
    printf("%f ", x[j]);
    X[j] = x[j] * xa + xb;
    x1 += x[j];
    X2 += X[j] * X[j];
    xx2 += x[j] * x[j];
  }
  printf("\n");

  for (size_t j = 0; j < DIMENSION; ++j) {
    y[j] = dist(gen);
    Y[j] = y[j] * ya + yb;
    y1 += y[j];
    printf("%f ", y[j]);
  }
  printf("\n");

  auto v1 = ailego::Distance::SquaredEuclidean(X.data(), Y.data(), DIMENSION);
  auto ip = ailego::Distance::InnerProduct(x.data(), y.data(), DIMENSION);
  ailego::SquaredNorm2Matrix<float, 1>::Compute(x.data(), DIMENSION, &x2);
  ailego::SquaredNorm2Matrix<float, 1>::Compute(y.data(), DIMENSION, &y2);
#if 0
  ailego::Norm1Matrix<float, 1>::Compute(x.data(), DIMENSION, &x1);
  ailego::Norm1Matrix<float, 1>::Compute(y.data(), DIMENSION, &y1);
#endif
  auto v2 = xa * xa * x2 + ya * ya * y2 - 2 * xa * ya * ip +
            (xb - yb) * (xb - yb) * DIMENSION +
            2 * (xb - yb) * (xa * x1 - ya * y1);
  auto t1 = (xa * x[0] - ya * y[0]) + (xb - yb);
  auto t2 = (xa * x[1] - ya * y[1]) + (xb - yb);
  auto v3 = t1 * t1 + t2 * t2;
  printf(
      "x=%f y=%f X=%f Y=%f, xa=%f xb=%f ya=%f yb=%f, x2=%f y2=%f x1=%f y1=%f "
      "ip=%f\n",
      x[0], y[0], X[0], Y[0], xa, xb, ya, yb, x2, y2, x1, y1, ip);
  printf("v1=%f v2=%f v3=%f\n", v1, v2, v3);

  auto IP = ailego::Distance::InnerProduct(X.data(), Y.data(), DIMENSION);
  auto v = xa * ya * ip + xb * ya * y1 + xa * yb * x1 + xb * yb * DIMENSION;
  printf("V=%f %f\n", IP, v);

  printf("=========\n");
  float mips;
  ailego::MipsSquaredEuclideanDistanceMatrix<float, 1, 1>::Compute(
      X.data(), Y.data(), DIMENSION, 0.0, &mips);
  printf("u2=%f v2=%f\n", x2, y2);
  float uu2 = xa * xa * x2 + 2 * xa * xb * x1 + xb * xb * DIMENSION;
  float vv2 = ya * ya * y2 + 2 * ya * yb * y1 + yb * yb * DIMENSION;
  float v7 = 2.0 - 2.0 * v / std::max(uu2, vv2);
  printf("mips=%f v7=%f\n", mips, v7);
  printf("X2=%f uu2=%f xx2=%f x2=%f\n", X2, uu2, xx2, x2);
}

TEST(QuantizedIntegerMetric, TestInt8SquaredEuclidean) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0, 2.0);

  const size_t DIMENSION = std::uniform_int_distribution<int>(1, 128)(gen);
  const size_t COUNT = 1000;
  IndexMeta meta;
  meta.set_meta(IndexMeta::DT_FP32, DIMENSION);
  auto converter = IndexFactory::CreateConverter("Int8StreamingConverter");
  ASSERT_TRUE(!!converter);
  ASSERT_EQ(0u, converter->init(meta, Params()));

  auto holder = GetHolder(DIMENSION, COUNT, dist);
  ASSERT_EQ(0u, IndexConverter::TrainAndTransform(converter, holder));
  auto holder2 = converter->result();
  EXPECT_EQ(COUNT, holder2->count());
  EXPECT_EQ(IndexMeta::DT_INT8, holder2->data_type());
  auto &meta2 = converter->meta();

  auto reformer = IndexFactory::CreateReformer(meta2.reformer_name());
  ASSERT_TRUE(reformer);
  ASSERT_EQ(0u, reformer->init(meta2.reformer_params()));

  ailego::NumericalVector<float> vec(DIMENSION);
  for (size_t j = 0; j < DIMENSION; ++j) {
    vec[j] = dist(gen);
  }
  IndexQueryMeta qmeta;
  qmeta.set_meta(IndexMeta::DT_FP32, DIMENSION);
  IndexQueryMeta qmeta2;
  std::string out;
  ASSERT_EQ(0, reformer->transform(vec.data(), qmeta, &out, &qmeta2));
  ASSERT_EQ(qmeta2.dimension(), meta2.dimension());

  auto iter = holder->create_iterator();
  auto iter2 = holder2->create_iterator();
  auto metric = IndexFactory::CreateMetric(meta2.metric_name());
  ASSERT_TRUE(!!metric);
  ASSERT_EQ(0, metric->init(meta2, meta2.metric_params()));
  auto compute = metric->distance();
  ASSERT_TRUE(compute);

  for (; iter->is_valid(); iter->next(), iter2->next()) {
    const float *mf = (const float *)iter->data();
    const int8_t *mi = (const int8_t *)iter2->data();
    const int8_t *qi = reinterpret_cast<const int8_t *>(&out[0]);
    float v1 =
        ailego::Distance::SquaredEuclidean(mf, vec.data(), holder->dimension());
    float v2;
    compute(mi, qi, holder2->dimension(), &v2);
    // printf("%f %f\n", v1, v2);
    ASSERT_NEAR(v1, v2, 0.1 * (DIMENSION + 1));

    std::string out2;
    ASSERT_EQ(0, reformer->convert(iter->data(), qmeta, &out2, &qmeta2));
    ASSERT_EQ(out2.size(), holder2->element_size());
    ASSERT_EQ(0, std::memcmp(out2.data(), iter2->data(), out2.size()));
  }
}

TEST(QuantizedIntegerMetric, TestInt8SquaredEuclideanReformer) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0, 2.0);
  std::uniform_int_distribution<int> dist2(0, 1);

  const size_t DIMENSION = std::uniform_int_distribution<int>(1, 128)(gen);
  const size_t COUNT = std::uniform_int_distribution<int>(1, 128)(gen);
  auto reformer = IndexFactory::CreateReformer("Int8StreamingReformer");
  ASSERT_TRUE(!!reformer);
  ASSERT_EQ(0u, reformer->init(Params()));

  ailego::NumericalVector<float> vecs(DIMENSION * COUNT);
  for (size_t j = 0; j < DIMENSION * COUNT; ++j) {
    vecs[j] = dist(gen);
  }
  IndexQueryMeta qmeta(IndexMeta::DT_FP32, DIMENSION);
  IndexQueryMeta qmeta1;
  std::string out;
  ASSERT_EQ(0,
            dist2(gen)
                ? reformer->transform(vecs.data(), qmeta, COUNT, &out, &qmeta1)
                : reformer->convert(vecs.data(), qmeta, COUNT, &out, &qmeta1));

  std::string out2;
  IndexQueryMeta qmeta2;
  for (size_t i = 0; i < COUNT; ++i) {
    ASSERT_EQ(0,
              reformer->transform(&vecs[i * DIMENSION], qmeta, &out2, &qmeta2));
    ASSERT_EQ(qmeta1.element_size(), qmeta2.element_size());
    ASSERT_EQ(out2.size(), qmeta2.element_size());
    ASSERT_EQ(0, std::memcmp(out2.data(), &out[i * qmeta1.element_size()],
                             out2.size()));

    ASSERT_EQ(0,
              reformer->convert(&vecs[i * DIMENSION], qmeta, &out2, &qmeta2));
    ASSERT_EQ(qmeta1.element_size(), qmeta2.element_size());
    ASSERT_EQ(out2.size(), qmeta2.element_size());
    ASSERT_EQ(0, std::memcmp(out2.data(), &out[i * qmeta1.element_size()],
                             out2.size()));
  }
}

template <size_t M, size_t N>
void TestDistanceMatrixInt8(const std::string &metric_name) {
  std::mt19937 gen((std::random_device())());
  std::uniform_real_distribution<float> dist(-1.0, 2.0);

  const size_t batch_size = M;
  const size_t query_size = N;
  size_t dimension = (std::uniform_int_distribution<size_t>(1, 65))(gen) * 4;
  auto holder = GetHolder(dimension, batch_size, dist);
  IndexMeta meta(IndexMeta::DT_FP32, dimension);
  meta.set_metric(metric_name, 0, Params());
  auto converter = IndexFactory::CreateConverter("Int8StreamingConverter");
  ASSERT_TRUE(!!converter);
  ASSERT_EQ(0u, converter->init(meta, Params()));
  ASSERT_EQ(0u, IndexConverter::TrainAndTransform(converter, holder));
  auto holder2 = converter->result();
  auto &meta2 = converter->meta();
  ASSERT_EQ(dimension + 16, holder2->dimension());
  size_t matrix_size = batch_size * holder2->dimension();
  std::vector<int8_t> matrix1(matrix_size);
  std::vector<int8_t> matrix2(matrix_size);
  auto iter = holder2->create_iterator();
  for (size_t i = 0; i < batch_size; ++i, iter->next()) {
    std::memcpy(&matrix1[i * holder2->dimension()], iter->data(),
                holder2->element_size());
  }
  MatrixTranspose(reinterpret_cast<uint32_t *>(&matrix2[0]),
                  reinterpret_cast<uint32_t *>(matrix1.data()),
                  meta2.dimension() / 4, batch_size);

  auto query_holder = GetHolder(dimension, query_size, dist);
  ASSERT_EQ(0u, IndexConverter::TrainAndTransform(converter, query_holder));
  auto query_holder2 = converter->result();
  ASSERT_EQ(dimension + 16, query_holder2->dimension());
  size_t query_matrix_size = query_size * query_holder2->dimension();
  std::vector<int8_t> query1(query_matrix_size);
  std::vector<int8_t> query2(query_matrix_size);
  auto query_iter = query_holder2->create_iterator();
  for (size_t i = 0; i < query_size; ++i, query_iter->next()) {
    std::memcpy(&query1[i * holder2->dimension()], query_iter->data(),
                query_holder2->element_size());
  }
  MatrixTranspose(reinterpret_cast<uint32_t *>(&query2[0]),
                  reinterpret_cast<uint32_t *>(query1.data()),
                  meta2.dimension() / 4, query_size);

  std::vector<float> result1(batch_size * query_size);
  std::vector<float> result2(batch_size * query_size);

  auto metric = IndexFactory::CreateMetric("QuantizedInteger");
  ASSERT_TRUE(!!metric);
  ASSERT_EQ(0,
            metric->init(converter->meta(), converter->meta().metric_params()));
  auto compute = metric->distance();
  ASSERT_TRUE(compute);
  auto matrix_compute = metric->distance_matrix(M, N);
  ASSERT_TRUE(matrix_compute);

  for (size_t i = 0; i < query_size; ++i) {
    const int8_t *cur_query = &query1[i * meta2.dimension()];
    float *query_result = &result1[i * batch_size];

    for (size_t j = 0; j < batch_size; ++j) {
      compute(&matrix1[j * meta2.dimension()], cur_query, meta2.dimension(),
              &query_result[j]);
    }
  }
  matrix_compute(&matrix2[0], &query2[0], meta2.dimension(), &result2[0]);

  for (size_t i = 0; i < batch_size * query_size; ++i) {
    // EXPECT_FLOAT_EQ(result1[i], result2[i]);
    EXPECT_TRUE(IsAlmostEqual(result1[i], result2[i], 1e4));
  }
}

TEST(QuantizedIntegerMetric, TestInt8SquaredEuclideanMetric) {
  TestDistanceMatrixInt8<1, 1>("SquaredEuclidean");
  TestDistanceMatrixInt8<2, 1>("SquaredEuclidean");
  TestDistanceMatrixInt8<2, 2>("SquaredEuclidean");
  TestDistanceMatrixInt8<4, 1>("SquaredEuclidean");
  TestDistanceMatrixInt8<4, 2>("SquaredEuclidean");
  TestDistanceMatrixInt8<4, 4>("SquaredEuclidean");
  TestDistanceMatrixInt8<8, 1>("SquaredEuclidean");
  TestDistanceMatrixInt8<8, 2>("SquaredEuclidean");
  TestDistanceMatrixInt8<8, 4>("SquaredEuclidean");
  TestDistanceMatrixInt8<8, 8>("SquaredEuclidean");
  TestDistanceMatrixInt8<16, 1>("SquaredEuclidean");
  TestDistanceMatrixInt8<16, 2>("SquaredEuclidean");
  TestDistanceMatrixInt8<16, 4>("SquaredEuclidean");
  TestDistanceMatrixInt8<16, 8>("SquaredEuclidean");
  TestDistanceMatrixInt8<16, 16>("SquaredEuclidean");
  TestDistanceMatrixInt8<32, 1>("SquaredEuclidean");
  TestDistanceMatrixInt8<32, 2>("SquaredEuclidean");
  TestDistanceMatrixInt8<32, 4>("SquaredEuclidean");
  TestDistanceMatrixInt8<32, 8>("SquaredEuclidean");
  TestDistanceMatrixInt8<32, 16>("SquaredEuclidean");
  TestDistanceMatrixInt8<32, 32>("SquaredEuclidean");
}

TEST(QuantizedIntegerMetric, TestInt4SquaredEuclidean) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0, 2.0);

  const size_t DIMENSION = std::uniform_int_distribution<int>(1, 128)(gen) * 2;
  const size_t COUNT = 1000;
  IndexMeta meta;
  meta.set_meta(IndexMeta::DT_FP32, DIMENSION);
  auto converter = IndexFactory::CreateConverter("Int4StreamingConverter");
  ASSERT_TRUE(!!converter);
  ASSERT_EQ(0u, converter->init(meta, Params()));

  auto holder = GetHolder(DIMENSION, COUNT, dist);
  ASSERT_EQ(0u, IndexConverter::TrainAndTransform(converter, holder));
  auto holder2 = converter->result();
  EXPECT_EQ(COUNT, holder2->count());
  EXPECT_EQ(IndexMeta::DT_INT4, holder2->data_type());
  auto &meta2 = converter->meta();

  auto reformer = IndexFactory::CreateReformer(meta2.reformer_name());
  ASSERT_TRUE(reformer);
  ASSERT_EQ(0u, reformer->init(meta2.reformer_params()));

  ailego::NumericalVector<float> vec(DIMENSION);
  for (size_t j = 0; j < DIMENSION; ++j) {
    vec[j] = dist(gen);
  }
  IndexQueryMeta qmeta;
  qmeta.set_meta(IndexMeta::DT_FP32, DIMENSION);
  IndexQueryMeta qmeta2;
  std::string out;
  ASSERT_EQ(0, reformer->transform(vec.data(), qmeta, &out, &qmeta2));
  ASSERT_EQ(qmeta2.dimension(), meta2.dimension());

  auto iter = holder->create_iterator();
  auto iter2 = holder2->create_iterator();
  auto metric = IndexFactory::CreateMetric(meta2.metric_name());
  ASSERT_TRUE(!!metric);
  ASSERT_EQ(0, metric->init(meta2, meta2.metric_params()));
  auto compute = metric->distance();
  ASSERT_TRUE(compute);

  for (; iter->is_valid(); iter->next(), iter2->next()) {
    const float *mf = (const float *)iter->data();
    const int8_t *mi = (const int8_t *)iter2->data();
    const int8_t *qi = reinterpret_cast<const int8_t *>(&out[0]);
    float v1 =
        ailego::Distance::SquaredEuclidean(mf, vec.data(), holder->dimension());
    float v2;
    compute(mi, qi, holder2->dimension(), &v2);
    ASSERT_NEAR(v1, v2, 0.2 * DIMENSION);

    std::string out2;
    ASSERT_EQ(0, reformer->convert(iter->data(), qmeta, &out2, &qmeta2));
    ASSERT_EQ(out2.size(), holder2->element_size());
    ASSERT_EQ(0, std::memcmp(out2.data(), iter2->data(), out2.size()));
  }
}

TEST(QuantizedIntegerMetric, TestInt4SquaredEuclideanReformer) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0, 2.0);
  std::uniform_int_distribution<int> dist2(0, 1);

  const size_t DIMENSION = std::uniform_int_distribution<int>(1, 128)(gen) * 2;
  const size_t COUNT = std::uniform_int_distribution<int>(1, 128)(gen);
  auto reformer = IndexFactory::CreateReformer("Int4StreamingReformer");
  ASSERT_TRUE(!!reformer);
  ASSERT_EQ(0u, reformer->init(Params()));

  ailego::NumericalVector<float> vecs(DIMENSION * COUNT);
  for (size_t j = 0; j < DIMENSION * COUNT; ++j) {
    vecs[j] = dist(gen);
  }
  IndexQueryMeta qmeta(IndexMeta::DT_FP32, DIMENSION);
  IndexQueryMeta qmeta1;
  std::string out;
  ASSERT_EQ(0,
            dist2(gen)
                ? reformer->transform(vecs.data(), qmeta, COUNT, &out, &qmeta1)
                : reformer->convert(vecs.data(), qmeta, COUNT, &out, &qmeta1));

  std::string out2;
  IndexQueryMeta qmeta2;
  for (size_t i = 0; i < COUNT; ++i) {
    ASSERT_EQ(0,
              reformer->transform(&vecs[i * DIMENSION], qmeta, &out2, &qmeta2));
    ASSERT_EQ(qmeta1.element_size(), qmeta2.element_size());
    ASSERT_EQ(out2.size(), qmeta2.element_size());
    ASSERT_EQ(0, std::memcmp(out2.data(), &out[i * qmeta1.element_size()],
                             out2.size()));

    ASSERT_EQ(0,
              reformer->convert(&vecs[i * DIMENSION], qmeta, &out2, &qmeta2));
    ASSERT_EQ(qmeta1.element_size(), qmeta2.element_size());
    ASSERT_EQ(out2.size(), qmeta2.element_size());
    ASSERT_EQ(0, std::memcmp(out2.data(), &out[i * qmeta1.element_size()],
                             out2.size()));
  }
}

template <size_t M, size_t N>
void TestDistanceMatrixInt4(const std::string &metric_name) {
  std::mt19937 gen((std::random_device())());
  std::uniform_real_distribution<float> dist(-1.0, 2.0);

  const size_t batch_size = M;
  const size_t query_size = N;
  size_t dimension = (std::uniform_int_distribution<size_t>(1, 65))(gen) * 8;
  auto holder = GetHolder(dimension, batch_size, dist);
  IndexMeta meta(IndexMeta::DT_FP32, dimension);
  meta.set_metric(metric_name, 0, Params());
  auto converter = IndexFactory::CreateConverter("Int4StreamingConverter");
  ASSERT_TRUE(!!converter);
  ASSERT_EQ(0u, converter->init(meta, Params()));
  ASSERT_EQ(0u, IndexConverter::TrainAndTransform(converter, holder));
  auto holder2 = converter->result();
  auto &meta2 = converter->meta();
  ASSERT_EQ(dimension + 32, holder2->dimension());
  size_t matrix_size = batch_size * holder2->element_size();
  std::vector<uint8_t> matrix1(matrix_size);
  std::vector<uint8_t> matrix2(matrix_size);
  auto iter = holder2->create_iterator();
  for (size_t i = 0; i < batch_size; ++i, iter->next()) {
    std::memcpy(&matrix1[i * holder2->element_size()], iter->data(),
                holder2->element_size());
  }
  MatrixTranspose(reinterpret_cast<uint32_t *>(&matrix2[0]),
                  reinterpret_cast<uint32_t *>(matrix1.data()),
                  meta2.dimension() / 8, batch_size);

  auto query_holder = GetHolder(dimension, query_size, dist);
  ASSERT_EQ(0u, IndexConverter::TrainAndTransform(converter, query_holder));
  auto query_holder2 = converter->result();
  ASSERT_EQ(dimension + 32, query_holder2->dimension());
  size_t query_matrix_size = query_size * query_holder2->element_size();
  std::vector<uint8_t> query1(query_matrix_size);
  std::vector<uint8_t> query2(query_matrix_size);
  auto query_iter = query_holder2->create_iterator();
  for (size_t i = 0; i < query_size; ++i, query_iter->next()) {
    std::memcpy(&query1[i * holder2->element_size()], query_iter->data(),
                query_holder2->element_size());
  }
  MatrixTranspose(reinterpret_cast<uint32_t *>(&query2[0]),
                  reinterpret_cast<uint32_t *>(query1.data()),
                  meta2.dimension() / 8, query_size);

  std::vector<float> result1(batch_size * query_size);
  std::vector<float> result2(batch_size * query_size);

  auto metric = IndexFactory::CreateMetric("QuantizedInteger");
  ASSERT_TRUE(!!metric);
  ASSERT_EQ(0,
            metric->init(converter->meta(), converter->meta().metric_params()));
  auto compute = metric->distance();
  ASSERT_TRUE(compute);
  auto matrix_compute = metric->distance_matrix(M, N);
  ASSERT_TRUE(matrix_compute);

  for (size_t i = 0; i < query_size; ++i) {
    const uint8_t *cur_query = &query1[i * meta2.element_size()];
    float *query_result = &result1[i * batch_size];

    for (size_t j = 0; j < batch_size; ++j) {
      compute(&matrix1[j * meta2.element_size()], cur_query, meta2.dimension(),
              &query_result[j]);
    }
  }
  matrix_compute(&matrix2[0], &query2[0], meta2.dimension(), &result2[0]);

  for (size_t i = 0; i < batch_size * query_size; ++i) {
    EXPECT_NEAR(result1[i], result2[i], 1e-2 * dimension);
    EXPECT_TRUE(IsAlmostEqual(result1[i], result2[i], 1e4));
  }
}

TEST(QuantizedIntegerMetric, TestInt4SquaredEuclideanMetric) {
  TestDistanceMatrixInt4<1, 1>("SquaredEuclidean");
  TestDistanceMatrixInt4<2, 1>("SquaredEuclidean");
  TestDistanceMatrixInt4<2, 2>("SquaredEuclidean");
  TestDistanceMatrixInt4<4, 1>("SquaredEuclidean");
  TestDistanceMatrixInt4<4, 2>("SquaredEuclidean");
  TestDistanceMatrixInt4<4, 4>("SquaredEuclidean");
  TestDistanceMatrixInt4<8, 1>("SquaredEuclidean");
  TestDistanceMatrixInt4<8, 2>("SquaredEuclidean");
  TestDistanceMatrixInt4<8, 4>("SquaredEuclidean");
  TestDistanceMatrixInt4<8, 8>("SquaredEuclidean");
  TestDistanceMatrixInt4<16, 1>("SquaredEuclidean");
  TestDistanceMatrixInt4<16, 2>("SquaredEuclidean");
  TestDistanceMatrixInt4<16, 4>("SquaredEuclidean");
  TestDistanceMatrixInt4<16, 8>("SquaredEuclidean");
  TestDistanceMatrixInt4<16, 16>("SquaredEuclidean");
  TestDistanceMatrixInt4<32, 1>("SquaredEuclidean");
  TestDistanceMatrixInt4<32, 2>("SquaredEuclidean");
  TestDistanceMatrixInt4<32, 4>("SquaredEuclidean");
  TestDistanceMatrixInt4<32, 8>("SquaredEuclidean");
  TestDistanceMatrixInt4<32, 16>("SquaredEuclidean");
  TestDistanceMatrixInt4<32, 32>("SquaredEuclidean");
}

TEST(QuantizedIntegerMetric, TestInt8InnerProduct) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0, 2.0);

  const size_t DIMENSION = std::uniform_int_distribution<int>(1, 128)(gen);
  const size_t COUNT = 1000;
  IndexMeta meta(IndexMeta::DT_FP32, DIMENSION);
  meta.set_metric("InnerProduct", 0, Params());
  auto converter = IndexFactory::CreateConverter("Int8StreamingConverter");
  ASSERT_TRUE(!!converter);
  ASSERT_EQ(0u, converter->init(meta, Params()));

  auto holder = GetHolder(DIMENSION, COUNT, dist);
  ASSERT_EQ(0u, IndexConverter::TrainAndTransform(converter, holder));
  auto holder2 = converter->result();
  EXPECT_EQ(COUNT, holder2->count());
  EXPECT_EQ(IndexMeta::DT_INT8, holder2->data_type());
  auto &meta2 = converter->meta();

  auto reformer = IndexFactory::CreateReformer(meta2.reformer_name());
  ASSERT_TRUE(reformer);
  ASSERT_EQ(0u, reformer->init(meta2.reformer_params()));

  ailego::NumericalVector<float> vec(DIMENSION);
  for (size_t j = 0; j < DIMENSION; ++j) {
    vec[j] = dist(gen);
  }
  IndexQueryMeta qmeta;
  qmeta.set_meta(IndexMeta::DT_FP32, DIMENSION);
  IndexQueryMeta qmeta2;
  std::string out;
  ASSERT_EQ(0, reformer->transform(vec.data(), qmeta, &out, &qmeta2));
  ASSERT_EQ(qmeta2.dimension(), meta2.dimension());

  auto iter = holder->create_iterator();
  auto iter2 = holder2->create_iterator();
  auto metric = IndexFactory::CreateMetric(meta2.metric_name());
  ASSERT_TRUE(!!metric);
  ASSERT_EQ(0, metric->init(meta2, meta2.metric_params()));
  auto compute = metric->distance();
  ASSERT_TRUE(compute);

  for (; iter->is_valid(); iter->next(), iter2->next()) {
    const float *mf = (const float *)iter->data();
    const int8_t *mi = (const int8_t *)iter2->data();
    const int8_t *qi = reinterpret_cast<const int8_t *>(&out[0]);
    float v1 = ailego::Distance::MinusInnerProduct(mf, vec.data(),
                                                   holder->dimension());
    float v2;
    compute(mi, qi, holder2->dimension(), &v2);
    // printf("%f %f\n", v1, v2);
    ASSERT_NEAR(v1, v2, 0.2 * DIMENSION);

    std::string out2;
    ASSERT_EQ(0, reformer->convert(iter->data(), qmeta, &out2, &qmeta2));
    ASSERT_EQ(out2.size(), holder2->element_size());
    ASSERT_EQ(0, std::memcmp(out2.data(), iter2->data(), out2.size()));
  }
}

TEST(QuantizedIntegerMetric, TestInt8InnerProductMetric) {
  TestDistanceMatrixInt8<1, 1>("InnerProduct");
  TestDistanceMatrixInt8<2, 1>("InnerProduct");
  TestDistanceMatrixInt8<2, 2>("InnerProduct");
  TestDistanceMatrixInt8<4, 1>("InnerProduct");
  TestDistanceMatrixInt8<4, 2>("InnerProduct");
  TestDistanceMatrixInt8<4, 4>("InnerProduct");
  TestDistanceMatrixInt8<8, 1>("InnerProduct");
  TestDistanceMatrixInt8<8, 2>("InnerProduct");
  TestDistanceMatrixInt8<8, 4>("InnerProduct");
  TestDistanceMatrixInt8<8, 8>("InnerProduct");
  TestDistanceMatrixInt8<16, 1>("InnerProduct");
  TestDistanceMatrixInt8<16, 2>("InnerProduct");
  TestDistanceMatrixInt8<16, 4>("InnerProduct");
  TestDistanceMatrixInt8<16, 8>("InnerProduct");
  TestDistanceMatrixInt8<16, 16>("InnerProduct");
  TestDistanceMatrixInt8<32, 1>("InnerProduct");
  TestDistanceMatrixInt8<32, 2>("InnerProduct");
  TestDistanceMatrixInt8<32, 4>("InnerProduct");
  TestDistanceMatrixInt8<32, 8>("InnerProduct");
  TestDistanceMatrixInt8<32, 16>("InnerProduct");
  TestDistanceMatrixInt8<32, 32>("InnerProduct");
}

TEST(QuantizedIntegerMetric, TestInt4InnerProduct) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0, 2.0);

  const size_t DIMENSION = std::uniform_int_distribution<int>(1, 128)(gen) * 2;
  const size_t COUNT = 1000;
  IndexMeta meta;
  meta.set_meta(IndexMeta::DT_FP32, DIMENSION);
  meta.set_metric("InnerProduct", 0, Params());
  auto converter = IndexFactory::CreateConverter("Int4StreamingConverter");
  ASSERT_TRUE(!!converter);
  ASSERT_EQ(0u, converter->init(meta, Params()));

  auto holder = GetHolder(DIMENSION, COUNT, dist);
  ASSERT_EQ(0u, IndexConverter::TrainAndTransform(converter, holder));
  auto holder2 = converter->result();
  EXPECT_EQ(COUNT, holder2->count());
  EXPECT_EQ(IndexMeta::DT_INT4, holder2->data_type());
  auto &meta2 = converter->meta();

  auto reformer = IndexFactory::CreateReformer(meta2.reformer_name());
  ASSERT_TRUE(reformer);
  ASSERT_EQ(0u, reformer->init(meta2.reformer_params()));

  ailego::NumericalVector<float> vec(DIMENSION);
  for (size_t j = 0; j < DIMENSION; ++j) {
    vec[j] = dist(gen);
  }
  IndexQueryMeta qmeta;
  qmeta.set_meta(IndexMeta::DT_FP32, DIMENSION);
  IndexQueryMeta qmeta2;
  std::string out;
  ASSERT_EQ(0, reformer->transform(vec.data(), qmeta, &out, &qmeta2));
  ASSERT_EQ(qmeta2.dimension(), meta2.dimension());

  auto iter = holder->create_iterator();
  auto iter2 = holder2->create_iterator();
  auto metric = IndexFactory::CreateMetric(meta2.metric_name());
  ASSERT_TRUE(!!metric);
  ASSERT_EQ(0, metric->init(meta2, meta2.metric_params()));
  auto compute = metric->distance();
  ASSERT_TRUE(compute);

  for (; iter->is_valid(); iter->next(), iter2->next()) {
    const float *mf = (const float *)iter->data();
    const int8_t *mi = (const int8_t *)iter2->data();
    const int8_t *qi = reinterpret_cast<const int8_t *>(&out[0]);
    float v1 = ailego::Distance::MinusInnerProduct(mf, vec.data(),
                                                   holder->dimension());
    float v2;
    compute(mi, qi, holder2->dimension(), &v2);
    ASSERT_NEAR(v1, v2, 0.2 * DIMENSION);

    std::string out2;
    ASSERT_EQ(0, reformer->convert(iter->data(), qmeta, &out2, &qmeta2));
    ASSERT_EQ(out2.size(), holder2->element_size());
    ASSERT_EQ(0, std::memcmp(out2.data(), iter2->data(), out2.size()));
  }
}

TEST(QuantizedIntegerMetric, TestInt4InnerProductMetric) {
  TestDistanceMatrixInt4<1, 1>("InnerProduct");
  TestDistanceMatrixInt4<2, 1>("InnerProduct");
  TestDistanceMatrixInt4<2, 2>("InnerProduct");
  TestDistanceMatrixInt4<4, 1>("InnerProduct");
  TestDistanceMatrixInt4<4, 2>("InnerProduct");
  TestDistanceMatrixInt4<4, 4>("InnerProduct");
  TestDistanceMatrixInt4<8, 1>("InnerProduct");
  TestDistanceMatrixInt4<8, 2>("InnerProduct");
  TestDistanceMatrixInt4<8, 4>("InnerProduct");
  TestDistanceMatrixInt4<8, 8>("InnerProduct");
  TestDistanceMatrixInt4<16, 1>("InnerProduct");
  TestDistanceMatrixInt4<16, 2>("InnerProduct");
  TestDistanceMatrixInt4<16, 4>("InnerProduct");
  TestDistanceMatrixInt4<16, 8>("InnerProduct");
  TestDistanceMatrixInt4<16, 16>("InnerProduct");
  TestDistanceMatrixInt4<32, 1>("InnerProduct");
  TestDistanceMatrixInt4<32, 2>("InnerProduct");
  TestDistanceMatrixInt4<32, 4>("InnerProduct");
  TestDistanceMatrixInt4<32, 8>("InnerProduct");
  TestDistanceMatrixInt4<32, 16>("InnerProduct");
  TestDistanceMatrixInt4<32, 32>("InnerProduct");
}

TEST(QuantizedIntegerMetric, TestInt8MipsSquaredEuclidean) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0, 2.0);

  const size_t DIMENSION = std::uniform_int_distribution<int>(1, 128)(gen);
  const size_t COUNT = 1000;
  IndexMeta meta(IndexMeta::DT_FP32, DIMENSION);
  meta.set_metric("MipsSquaredEuclidean", 0, Params());
  auto converter = IndexFactory::CreateConverter("Int8StreamingConverter");
  ASSERT_TRUE(!!converter);
  ASSERT_EQ(0u, converter->init(meta, Params()));

  auto holder = GetHolder(DIMENSION, COUNT, dist);
  ASSERT_EQ(0u, IndexConverter::TrainAndTransform(converter, holder));
  auto holder2 = converter->result();
  EXPECT_EQ(COUNT, holder2->count());
  EXPECT_EQ(IndexMeta::DT_INT8, holder2->data_type());
  auto &meta2 = converter->meta();

  auto reformer = IndexFactory::CreateReformer(meta2.reformer_name());
  ASSERT_TRUE(reformer);
  ASSERT_EQ(0u, reformer->init(meta2.reformer_params()));

  ailego::NumericalVector<float> vec(DIMENSION);
  for (size_t j = 0; j < DIMENSION; ++j) {
    vec[j] = dist(gen);
  }
  IndexQueryMeta qmeta;
  qmeta.set_meta(IndexMeta::DT_FP32, DIMENSION);
  IndexQueryMeta qmeta2;
  std::string out;
  ASSERT_EQ(0, reformer->transform(vec.data(), qmeta, &out, &qmeta2));
  ASSERT_EQ(qmeta2.dimension(), meta2.dimension());

  auto iter = holder->create_iterator();
  auto iter2 = holder2->create_iterator();
  auto metric = IndexFactory::CreateMetric(meta2.metric_name());
  ASSERT_TRUE(!!metric);
  ASSERT_EQ(0, metric->init(meta2, meta2.metric_params()));
  auto compute = metric->distance();
  ASSERT_TRUE(compute);

  auto query_metric = metric->query_metric();
  ASSERT_TRUE(!!query_metric);
  ASSERT_EQ(query_metric->name(), "QuantizedInteger");

  for (; iter->is_valid(); iter->next(), iter2->next()) {
    const float *mf = (const float *)iter->data();
    const int8_t *mi = (const int8_t *)iter2->data();
    const int8_t *qi = reinterpret_cast<const int8_t *>(&out[0]);
    float v1 = ailego::Distance::MipsSquaredEuclidean(
        mf, vec.data(), holder->dimension(), 0.0f);
    float v2;
    compute(mi, qi, holder2->dimension(), &v2);
    // printf("%f %f\n", v1, v2);
    ASSERT_NEAR(v1, v2, 0.2 * DIMENSION);

    std::string out2;
    ASSERT_EQ(0, reformer->convert(iter->data(), qmeta, &out2, &qmeta2));
    ASSERT_EQ(out2.size(), holder2->element_size());
    ASSERT_EQ(0, std::memcmp(out2.data(), iter2->data(), out2.size()));
  }
}

TEST(QuantizedIntegerMetric, TestInt8MipsSquaredEuclideanMetric) {
  TestDistanceMatrixInt8<1, 1>("MipsSquaredEuclidean");
  TestDistanceMatrixInt8<2, 1>("MipsSquaredEuclidean");
  TestDistanceMatrixInt8<2, 2>("MipsSquaredEuclidean");
  TestDistanceMatrixInt8<4, 1>("MipsSquaredEuclidean");
  TestDistanceMatrixInt8<4, 2>("MipsSquaredEuclidean");
  TestDistanceMatrixInt8<4, 4>("MipsSquaredEuclidean");
  TestDistanceMatrixInt8<8, 1>("MipsSquaredEuclidean");
  TestDistanceMatrixInt8<8, 2>("MipsSquaredEuclidean");
  TestDistanceMatrixInt8<8, 4>("MipsSquaredEuclidean");
  TestDistanceMatrixInt8<8, 8>("MipsSquaredEuclidean");
  TestDistanceMatrixInt8<16, 1>("MipsSquaredEuclidean");
  TestDistanceMatrixInt8<16, 2>("MipsSquaredEuclidean");
  TestDistanceMatrixInt8<16, 4>("MipsSquaredEuclidean");
  TestDistanceMatrixInt8<16, 8>("MipsSquaredEuclidean");
  TestDistanceMatrixInt8<16, 16>("MipsSquaredEuclidean");
  TestDistanceMatrixInt8<32, 1>("MipsSquaredEuclidean");
  TestDistanceMatrixInt8<32, 2>("MipsSquaredEuclidean");
  TestDistanceMatrixInt8<32, 4>("MipsSquaredEuclidean");
  TestDistanceMatrixInt8<32, 8>("MipsSquaredEuclidean");
  TestDistanceMatrixInt8<32, 16>("MipsSquaredEuclidean");
  TestDistanceMatrixInt8<32, 32>("MipsSquaredEuclidean");
}

TEST(QuantizedIntegerMetric, TestInt4MipsSquaredEuclidean) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0, 2.0);

  const size_t DIMENSION = std::uniform_int_distribution<int>(1, 128)(gen) * 2;
  const size_t COUNT = 1000;
  IndexMeta meta;
  meta.set_meta(IndexMeta::DT_FP32, DIMENSION);
  meta.set_metric("MipsSquaredEuclidean", 0, Params());
  auto converter = IndexFactory::CreateConverter("Int4StreamingConverter");
  ASSERT_TRUE(!!converter);
  ASSERT_EQ(0u, converter->init(meta, Params()));

  auto holder = GetHolder(DIMENSION, COUNT, dist);
  ASSERT_EQ(0u, IndexConverter::TrainAndTransform(converter, holder));
  auto holder2 = converter->result();
  EXPECT_EQ(COUNT, holder2->count());
  EXPECT_EQ(IndexMeta::DT_INT4, holder2->data_type());
  auto &meta2 = converter->meta();

  auto reformer = IndexFactory::CreateReformer(meta2.reformer_name());
  ASSERT_TRUE(reformer);
  ASSERT_EQ(0u, reformer->init(meta2.reformer_params()));

  ailego::NumericalVector<float> vec(DIMENSION);
  for (size_t j = 0; j < DIMENSION; ++j) {
    vec[j] = dist(gen);
  }
  IndexQueryMeta qmeta;
  qmeta.set_meta(IndexMeta::DT_FP32, DIMENSION);
  IndexQueryMeta qmeta2;
  std::string out;
  ASSERT_EQ(0, reformer->transform(vec.data(), qmeta, &out, &qmeta2));
  ASSERT_EQ(qmeta2.dimension(), meta2.dimension());

  auto iter = holder->create_iterator();
  auto iter2 = holder2->create_iterator();
  auto metric = IndexFactory::CreateMetric(meta2.metric_name());
  ASSERT_TRUE(!!metric);
  ASSERT_EQ(0, metric->init(meta2, meta2.metric_params()));
  auto compute = metric->distance();
  ASSERT_TRUE(compute);

  for (; iter->is_valid(); iter->next(), iter2->next()) {
    const float *mf = (const float *)iter->data();
    const int8_t *mi = (const int8_t *)iter2->data();
    const int8_t *qi = reinterpret_cast<const int8_t *>(&out[0]);
    float v1 = ailego::Distance::MipsSquaredEuclidean(mf, vec.data(),
                                                      holder->dimension(), 0.0);
    float v2;
    compute(mi, qi, holder2->dimension(), &v2);
    ASSERT_NEAR(v1, v2, 0.2 * DIMENSION);

    std::string out2;
    ASSERT_EQ(0, reformer->convert(iter->data(), qmeta, &out2, &qmeta2));
    ASSERT_EQ(out2.size(), holder2->element_size());
    ASSERT_EQ(0, std::memcmp(out2.data(), iter2->data(), out2.size()));
  }
}

TEST(QuantizedIntegerMetric, TestInt4MipsSquaredEuclideanMetric) {
  TestDistanceMatrixInt4<1, 1>("MipsSquaredEuclidean");
  TestDistanceMatrixInt4<2, 1>("MipsSquaredEuclidean");
  TestDistanceMatrixInt4<2, 2>("MipsSquaredEuclidean");
  TestDistanceMatrixInt4<4, 1>("MipsSquaredEuclidean");
  TestDistanceMatrixInt4<4, 2>("MipsSquaredEuclidean");
  TestDistanceMatrixInt4<4, 4>("MipsSquaredEuclidean");
  TestDistanceMatrixInt4<8, 1>("MipsSquaredEuclidean");
  TestDistanceMatrixInt4<8, 2>("MipsSquaredEuclidean");
  TestDistanceMatrixInt4<8, 4>("MipsSquaredEuclidean");
  TestDistanceMatrixInt4<8, 8>("MipsSquaredEuclidean");
  TestDistanceMatrixInt4<16, 1>("MipsSquaredEuclidean");
  TestDistanceMatrixInt4<16, 2>("MipsSquaredEuclidean");
  TestDistanceMatrixInt4<16, 4>("MipsSquaredEuclidean");
  TestDistanceMatrixInt4<16, 8>("MipsSquaredEuclidean");
  TestDistanceMatrixInt4<16, 16>("MipsSquaredEuclidean");
  TestDistanceMatrixInt4<32, 1>("MipsSquaredEuclidean");
  TestDistanceMatrixInt4<32, 2>("MipsSquaredEuclidean");
  TestDistanceMatrixInt4<32, 4>("MipsSquaredEuclidean");
  TestDistanceMatrixInt4<32, 8>("MipsSquaredEuclidean");
  TestDistanceMatrixInt4<32, 16>("MipsSquaredEuclidean");
  TestDistanceMatrixInt4<32, 32>("MipsSquaredEuclidean");
}

TEST(QuantizedIntegerMetric, TestInt8NormalizedCosine) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0, 2.0);

  const size_t DIMENSION = std::uniform_int_distribution<int>(1, 128)(gen);
  const size_t COUNT = 1000;
  IndexMeta meta(IndexMeta::DT_FP32, DIMENSION);
  meta.set_metric("NormalizedCosine", 0, Params());
  auto converter = IndexFactory::CreateConverter("Int8StreamingConverter");
  ASSERT_TRUE(!!converter);
  Params converter_params;
  converter_params.set(INTEGER_STREAMING_CONVERTER_ENABLE_NORMALIZE, true);
  ASSERT_EQ(0u, converter->init(meta, converter_params));

  auto holder = GetHolder(DIMENSION, COUNT, dist);
  ASSERT_EQ(0u, IndexConverter::TrainAndTransform(converter, holder));
  auto holder2 = converter->result();
  EXPECT_EQ(COUNT, holder2->count());
  EXPECT_EQ(IndexMeta::DT_INT8, holder2->data_type());
  auto &meta2 = converter->meta();

  auto reformer = IndexFactory::CreateReformer(meta2.reformer_name());
  ASSERT_TRUE(reformer);
  ASSERT_EQ(0u, reformer->init(meta2.reformer_params()));

  ailego::NumericalVector<float> vec(DIMENSION);
  for (size_t j = 0; j < DIMENSION; ++j) {
    vec[j] = dist(gen);
  }
  IndexQueryMeta qmeta;
  qmeta.set_meta(IndexMeta::DT_FP32, DIMENSION);
  IndexQueryMeta qmeta2;
  std::string out;
  ASSERT_EQ(0, reformer->transform(vec.data(), qmeta, &out, &qmeta2));
  ASSERT_EQ(qmeta2.dimension(), meta2.dimension());

  auto iter = holder->create_iterator();
  auto iter2 = holder2->create_iterator();
  auto metric = IndexFactory::CreateMetric(meta2.metric_name());
  ASSERT_TRUE(!!metric);
  ASSERT_EQ(0, metric->init(meta2, meta2.metric_params()));
  auto compute = metric->distance();
  ASSERT_TRUE(compute);

  for (; iter->is_valid(); iter->next(), iter2->next()) {
    const float *mf = (const float *)iter->data();
    const int8_t *mi = (const int8_t *)iter2->data();
    const int8_t *qi = reinterpret_cast<const int8_t *>(&out[0]);

    // normalize mf & vec
    std::vector<float> normalized_mf(DIMENSION);
    memcpy(normalized_mf.data(), mf, DIMENSION * sizeof(float));
    float norm_mf = 0.0;
    ailego::Normalizer<float>::L2((float *)normalized_mf.data(), DIMENSION,
                                  &norm_mf);
    std::vector<float> normalized_vec(DIMENSION);
    memcpy(normalized_vec.data(), vec.data(), DIMENSION * sizeof(float));
    float norm_vec = 0.0;
    ailego::Normalizer<float>::L2((float *)normalized_vec.data(), DIMENSION,
                                  &norm_vec);

    float v1 = ailego::Distance::MinusInnerProduct(
        normalized_mf.data(), normalized_vec.data(), holder->dimension());
    float v2;
    compute(mi, qi, holder2->dimension(), &v2);
    // printf("%f %f\n", v1, v2);
    ASSERT_NEAR(v1, v2, 0.2 * DIMENSION);

    std::string out2;
    ASSERT_EQ(0, reformer->convert(iter->data(), qmeta, &out2, &qmeta2));
    ASSERT_EQ(out2.size(), holder2->element_size());
    ASSERT_EQ(0, std::memcmp(out2.data(), iter2->data(), out2.size()));
  }
}

TEST(QuantizedIntegerMetric, TestInt8NormalizedCosineMetric) {
  TestDistanceMatrixInt8<1, 1>("NormalizedCosine");
  TestDistanceMatrixInt8<2, 1>("NormalizedCosine");
  TestDistanceMatrixInt8<2, 2>("NormalizedCosine");
  TestDistanceMatrixInt8<4, 1>("NormalizedCosine");
  TestDistanceMatrixInt8<4, 2>("NormalizedCosine");
  TestDistanceMatrixInt8<4, 4>("NormalizedCosine");
  TestDistanceMatrixInt8<8, 1>("NormalizedCosine");
  TestDistanceMatrixInt8<8, 2>("NormalizedCosine");
  TestDistanceMatrixInt8<8, 4>("NormalizedCosine");
  TestDistanceMatrixInt8<8, 8>("NormalizedCosine");
  TestDistanceMatrixInt8<16, 1>("NormalizedCosine");
  TestDistanceMatrixInt8<16, 2>("NormalizedCosine");
  TestDistanceMatrixInt8<16, 4>("NormalizedCosine");
  TestDistanceMatrixInt8<16, 8>("NormalizedCosine");
  TestDistanceMatrixInt8<16, 16>("NormalizedCosine");
  TestDistanceMatrixInt8<32, 1>("NormalizedCosine");
  TestDistanceMatrixInt8<32, 2>("NormalizedCosine");
  TestDistanceMatrixInt8<32, 4>("NormalizedCosine");
  TestDistanceMatrixInt8<32, 8>("NormalizedCosine");
  TestDistanceMatrixInt8<32, 16>("NormalizedCosine");
  TestDistanceMatrixInt8<32, 32>("NormalizedCosine");
}

TEST(QuantizedIntegerMetric, TestInt8Cosine) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0, 2.0);

  const size_t DIMENSION = std::uniform_int_distribution<int>(1, 128)(gen);
  const size_t COUNT = 1000;
  IndexMeta meta(IndexMeta::DT_FP32, DIMENSION);
  meta.set_metric("Cosine", 0, Params());
  auto converter = IndexFactory::CreateConverter("CosineInt8Converter");
  ASSERT_TRUE(!!converter);
  Params converter_params;
  ASSERT_EQ(0u, converter->init(meta, converter_params));

  auto holder = GetHolder(DIMENSION, COUNT, dist);
  ASSERT_EQ(0u, IndexConverter::TrainAndTransform(converter, holder));
  auto holder2 = converter->result();
  EXPECT_EQ(COUNT, holder2->count());
  EXPECT_EQ(IndexMeta::DT_INT8, holder2->data_type());
  auto &meta2 = converter->meta();

  auto reformer = IndexFactory::CreateReformer(meta2.reformer_name());
  ASSERT_TRUE(reformer);
  ASSERT_EQ(0u, reformer->init(meta2.reformer_params()));

  ailego::NumericalVector<float> vec(DIMENSION);
  for (size_t j = 0; j < DIMENSION; ++j) {
    vec[j] = dist(gen);
  }
  IndexQueryMeta qmeta;
  qmeta.set_meta(IndexMeta::DT_FP32, DIMENSION);
  IndexQueryMeta qmeta2;
  std::string out;
  ASSERT_EQ(0, reformer->transform(vec.data(), qmeta, &out, &qmeta2));
  ASSERT_EQ(qmeta2.dimension(), meta2.dimension());

  auto iter = holder->create_iterator();
  auto iter2 = holder2->create_iterator();
  auto metric = IndexFactory::CreateMetric(meta2.metric_name());
  ASSERT_TRUE(!!metric);
  ASSERT_EQ(0, metric->init(meta2, meta2.metric_params()));
  auto compute_batch = metric->batch_distance();
  ASSERT_TRUE(compute_batch);

  int8_t *qi = reinterpret_cast<int8_t *>(&out[0]);
  if (auto query_preprocess_func = metric->get_query_preprocess_func();
      query_preprocess_func != nullptr) {
    query_preprocess_func(qi, holder2->dimension());
  }

  for (; iter->is_valid(); iter->next(), iter2->next()) {
    const float *mf = (const float *)iter->data();
    const int8_t *mi = (const int8_t *)iter2->data();

    // normalize mf & vec
    std::vector<float> normalized_mf(DIMENSION);
    memcpy(normalized_mf.data(), mf, DIMENSION * sizeof(float));
    float norm_mf = 0.0;
    ailego::Normalizer<float>::L2((float *)normalized_mf.data(), DIMENSION,
                                  &norm_mf);
    std::vector<float> normalized_vec(DIMENSION);
    memcpy(normalized_vec.data(), vec.data(), DIMENSION * sizeof(float));
    float norm_vec = 0.0;
    ailego::Normalizer<float>::L2((float *)normalized_vec.data(), DIMENSION,
                                  &norm_vec);

    float v1 = ailego::Distance::MinusInnerProduct(
        normalized_mf.data(), normalized_vec.data(), holder->dimension());
    float v2;
    compute_batch(reinterpret_cast<const void **>(&mi), qi, 1,
                  holder2->dimension(), &v2);
    // printf("%f %f\n", v1, v2);
    ASSERT_NEAR(v1, v2, 0.2 * DIMENSION);

    std::string out2;
    ASSERT_EQ(0, reformer->convert(iter->data(), qmeta, &out2, &qmeta2));
    ASSERT_EQ(out2.size(), holder2->element_size());
    ASSERT_EQ(0, std::memcmp(out2.data(), iter2->data(), out2.size()));
  }
}

TEST(QuantizedIntegerMetric, TestInt4NormalizedCosine) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0, 2.0);

  const size_t DIMENSION = std::uniform_int_distribution<int>(1, 128)(gen) * 2;
  const size_t COUNT = 1000;
  IndexMeta meta;
  meta.set_meta(IndexMeta::DT_FP32, DIMENSION);
  meta.set_metric("NormalizedCosine", 0, Params());
  auto converter = IndexFactory::CreateConverter("Int4StreamingConverter");
  ASSERT_TRUE(!!converter);
  Params converter_params;
  converter_params.set(INTEGER_STREAMING_CONVERTER_ENABLE_NORMALIZE, true);
  ASSERT_EQ(0u, converter->init(meta, converter_params));

  auto holder = GetHolder(DIMENSION, COUNT, dist);
  ASSERT_EQ(0u, IndexConverter::TrainAndTransform(converter, holder));
  auto holder2 = converter->result();
  EXPECT_EQ(COUNT, holder2->count());
  EXPECT_EQ(IndexMeta::DT_INT4, holder2->data_type());
  auto &meta2 = converter->meta();

  auto reformer = IndexFactory::CreateReformer(meta2.reformer_name());
  ASSERT_TRUE(reformer);
  ASSERT_EQ(0u, reformer->init(meta2.reformer_params()));

  ailego::NumericalVector<float> vec(DIMENSION);
  for (size_t j = 0; j < DIMENSION; ++j) {
    vec[j] = dist(gen);
  }
  IndexQueryMeta qmeta;
  qmeta.set_meta(IndexMeta::DT_FP32, DIMENSION);
  IndexQueryMeta qmeta2;
  std::string out;
  ASSERT_EQ(0, reformer->transform(vec.data(), qmeta, &out, &qmeta2));
  ASSERT_EQ(qmeta2.dimension(), meta2.dimension());

  auto iter = holder->create_iterator();
  auto iter2 = holder2->create_iterator();
  auto metric = IndexFactory::CreateMetric(meta2.metric_name());
  ASSERT_TRUE(!!metric);
  ASSERT_EQ(0, metric->init(meta2, meta2.metric_params()));
  auto compute = metric->distance();
  ASSERT_TRUE(compute);

  for (; iter->is_valid(); iter->next(), iter2->next()) {
    const float *mf = (const float *)iter->data();
    const int8_t *mi = (const int8_t *)iter2->data();
    const int8_t *qi = reinterpret_cast<const int8_t *>(&out[0]);
    // normalize mf & vec
    std::vector<float> normalized_mf(DIMENSION);
    memcpy(normalized_mf.data(), mf, DIMENSION * sizeof(float));
    float norm_mf = 0.0;
    ailego::Normalizer<float>::L2((float *)normalized_mf.data(), DIMENSION,
                                  &norm_mf);
    std::vector<float> normalized_vec(DIMENSION);
    memcpy(normalized_vec.data(), vec.data(), DIMENSION * sizeof(float));
    float norm_vec = 0.0;
    ailego::Normalizer<float>::L2((float *)normalized_vec.data(), DIMENSION,
                                  &norm_vec);

    float v1 = ailego::Distance::MinusInnerProduct(
        normalized_mf.data(), normalized_vec.data(), holder->dimension());
    float v2;
    compute(mi, qi, holder2->dimension(), &v2);
    ASSERT_NEAR(v1, v2, 0.2 * DIMENSION);

    std::string out2;
    ASSERT_EQ(0, reformer->convert(iter->data(), qmeta, &out2, &qmeta2));
    ASSERT_EQ(out2.size(), holder2->element_size());
    ASSERT_EQ(0, std::memcmp(out2.data(), iter2->data(), out2.size()));
  }
}

TEST(QuantizedIntegerMetric, TestInt4NormalizedCosineMetric) {
  TestDistanceMatrixInt4<1, 1>("NormalizedCosine");
  TestDistanceMatrixInt4<2, 1>("NormalizedCosine");
  TestDistanceMatrixInt4<2, 2>("NormalizedCosine");
  TestDistanceMatrixInt4<4, 1>("NormalizedCosine");
  TestDistanceMatrixInt4<4, 2>("NormalizedCosine");
  TestDistanceMatrixInt4<4, 4>("NormalizedCosine");
  TestDistanceMatrixInt4<8, 1>("NormalizedCosine");
  TestDistanceMatrixInt4<8, 2>("NormalizedCosine");
  TestDistanceMatrixInt4<8, 4>("NormalizedCosine");
  TestDistanceMatrixInt4<8, 8>("NormalizedCosine");
  TestDistanceMatrixInt4<16, 1>("NormalizedCosine");
  TestDistanceMatrixInt4<16, 2>("NormalizedCosine");
  TestDistanceMatrixInt4<16, 4>("NormalizedCosine");
  TestDistanceMatrixInt4<16, 8>("NormalizedCosine");
  TestDistanceMatrixInt4<16, 16>("NormalizedCosine");
  TestDistanceMatrixInt4<32, 1>("NormalizedCosine");
  TestDistanceMatrixInt4<32, 2>("NormalizedCosine");
  TestDistanceMatrixInt4<32, 4>("NormalizedCosine");
  TestDistanceMatrixInt4<32, 8>("NormalizedCosine");
  TestDistanceMatrixInt4<32, 16>("NormalizedCosine");
  TestDistanceMatrixInt4<32, 32>("NormalizedCosine");
}
