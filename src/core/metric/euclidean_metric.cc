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
#include <ailego/math/euclidean_distance_matrix.h>
#include <ailego/math/hamming_distance_matrix.h>
#include <ailego/math_batch/distance_batch.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_metric.h>

namespace zvec {
namespace core {

//! Retrieve distance function for index features
static inline IndexMetric::MatrixDistanceHandle
SquaredEuclideanDistanceMatrixFp32(size_t m, size_t n) {
  static const IndexMetric::MatrixDistanceHandle distance_table[6][6] = {
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<float, 1, 1>::Compute),
       nullptr, nullptr, nullptr, nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<float, 2, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<float, 2, 2>::Compute),
       nullptr, nullptr, nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<float, 4, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<float, 4, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<float, 4, 4>::Compute),
       nullptr, nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<float, 8, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<float, 8, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<float, 8, 4>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<float, 8, 8>::Compute),
       nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<float, 16, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<float, 16, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<float, 16, 4>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<float, 16, 8>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<float, 16, 16>::Compute),
       nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<float, 32, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<float, 32, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<float, 32, 4>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<float, 32, 8>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<float, 32, 16>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<float, 32, 32>::Compute)},
  };
  if (m > 32 || n > 32 || ailego_popcount(m) != 1 || ailego_popcount(n) != 1) {
    return nullptr;
  }
  return distance_table[ailego_ctz(m)][ailego_ctz(n)];
}

//! Retrieve distance function for index features
static inline IndexMetric::MatrixDistanceHandle
SquaredEuclideanDistanceMatrixFp16(size_t m, size_t n) {
  static const IndexMetric::MatrixDistanceHandle distance_table[6][6] = {
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<ailego::Float16, 1,
                                                  1>::Compute),
       nullptr, nullptr, nullptr, nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<ailego::Float16, 2,
                                                  1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<ailego::Float16, 2,
                                                  2>::Compute),
       nullptr, nullptr, nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<ailego::Float16, 4,
                                                  1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<ailego::Float16, 4,
                                                  2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<ailego::Float16, 4,
                                                  4>::Compute),
       nullptr, nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<ailego::Float16, 8,
                                                  1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<ailego::Float16, 8,
                                                  2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<ailego::Float16, 8,
                                                  4>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<ailego::Float16, 8,
                                                  8>::Compute),
       nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<ailego::Float16, 16,
                                                  1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<ailego::Float16, 16,
                                                  2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<ailego::Float16, 16,
                                                  4>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<ailego::Float16, 16,
                                                  8>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<ailego::Float16, 16,
                                                  16>::Compute),
       nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<ailego::Float16, 32,
                                                  1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<ailego::Float16, 32,
                                                  2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<ailego::Float16, 32,
                                                  4>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<ailego::Float16, 32,
                                                  8>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<ailego::Float16, 32,
                                                  16>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<ailego::Float16, 32,
                                                  32>::Compute)},
  };
  if (m > 32 || n > 32 || ailego_popcount(m) != 1 || ailego_popcount(n) != 1) {
    return nullptr;
  }
  return distance_table[ailego_ctz(m)][ailego_ctz(n)];
}

static inline IndexMetric::MatrixDistanceHandle
SquaredEuclideanDistanceMatrixInt8(size_t m, size_t n) {
  static const IndexMetric::MatrixDistanceHandle distance_table[6][6] = {
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<int8_t, 1, 1>::Compute),
       nullptr, nullptr, nullptr, nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<int8_t, 2, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<int8_t, 2, 2>::Compute),
       nullptr, nullptr, nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<int8_t, 4, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<int8_t, 4, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<int8_t, 4, 4>::Compute),
       nullptr, nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<int8_t, 8, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<int8_t, 8, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<int8_t, 8, 4>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<int8_t, 8, 8>::Compute),
       nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<int8_t, 16, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<int8_t, 16, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<int8_t, 16, 4>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<int8_t, 16, 8>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<int8_t, 16, 16>::Compute),
       nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<int8_t, 32, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<int8_t, 32, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<int8_t, 32, 4>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<int8_t, 32, 8>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<int8_t, 32, 16>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<int8_t, 32, 32>::Compute)},
  };
  if (m > 32 || n > 32 || ailego_popcount(m) != 1 || ailego_popcount(n) != 1) {
    return nullptr;
  }
  return distance_table[ailego_ctz(m)][ailego_ctz(n)];
}

//! Retrieve distance function for index features in Int4
static inline IndexMetric::MatrixDistanceHandle
SquaredEuclideanDistanceMatrixInt4(size_t m, size_t n) {
  static const IndexMetric::MatrixDistanceHandle distance_table[6][6] = {
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<uint8_t, 1, 1>::Compute),
       nullptr, nullptr, nullptr, nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<uint8_t, 2, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<uint8_t, 2, 2>::Compute),
       nullptr, nullptr, nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<uint8_t, 4, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<uint8_t, 4, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<uint8_t, 4, 4>::Compute),
       nullptr, nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<uint8_t, 8, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<uint8_t, 8, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<uint8_t, 8, 4>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<uint8_t, 8, 8>::Compute),
       nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<uint8_t, 16, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<uint8_t, 16, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<uint8_t, 16, 4>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<uint8_t, 16, 8>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<uint8_t, 16, 16>::Compute),
       nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<uint8_t, 32, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<uint8_t, 32, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<uint8_t, 32, 4>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<uint8_t, 32, 8>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<uint8_t, 32, 16>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::SquaredEuclideanDistanceMatrix<uint8_t, 32, 32>::Compute)},
  };
  if (m > 32 || n > 32 || ailego_popcount(m) != 1 || ailego_popcount(n) != 1) {
    return nullptr;
  }
  return distance_table[ailego_ctz(m)][ailego_ctz(n)];
}

//! Retrieve distance function for index features
static inline IndexMetric::MatrixDistanceHandle EuclideanDistanceMatrixFp32(
    size_t m, size_t n) {
  static const IndexMetric::MatrixDistanceHandle distance_table[6][6] = {
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<float, 1, 1>::Compute),
       nullptr, nullptr, nullptr, nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<float, 2, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<float, 2, 2>::Compute),
       nullptr, nullptr, nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<float, 4, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<float, 4, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<float, 4, 4>::Compute),
       nullptr, nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<float, 8, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<float, 8, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<float, 8, 4>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<float, 8, 8>::Compute),
       nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<float, 16, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<float, 16, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<float, 16, 4>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<float, 16, 8>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<float, 16, 16>::Compute),
       nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<float, 32, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<float, 32, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<float, 32, 4>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<float, 32, 8>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<float, 32, 16>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<float, 32, 32>::Compute)},
  };
  if (m > 32 || n > 32 || ailego_popcount(m) != 1 || ailego_popcount(n) != 1) {
    return nullptr;
  }
  return distance_table[ailego_ctz(m)][ailego_ctz(n)];
}

//! Retrieve distance function for index features
static inline IndexMetric::MatrixDistanceHandle EuclideanDistanceMatrixFp16(
    size_t m, size_t n) {
  static const IndexMetric::MatrixDistanceHandle distance_table[6][6] = {
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<ailego::Float16, 1, 1>::Compute),
       nullptr, nullptr, nullptr, nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<ailego::Float16, 2, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<ailego::Float16, 2, 2>::Compute),
       nullptr, nullptr, nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<ailego::Float16, 4, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<ailego::Float16, 4, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<ailego::Float16, 4, 4>::Compute),
       nullptr, nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<ailego::Float16, 8, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<ailego::Float16, 8, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<ailego::Float16, 8, 4>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<ailego::Float16, 8, 8>::Compute),
       nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<ailego::Float16, 16, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<ailego::Float16, 16, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<ailego::Float16, 16, 4>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<ailego::Float16, 16, 8>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<ailego::Float16, 16, 16>::Compute),
       nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<ailego::Float16, 32, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<ailego::Float16, 32, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<ailego::Float16, 32, 4>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<ailego::Float16, 32, 8>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<ailego::Float16, 32, 16>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<ailego::Float16, 32, 32>::Compute)},
  };
  if (m > 32 || n > 32 || ailego_popcount(m) != 1 || ailego_popcount(n) != 1) {
    return nullptr;
  }
  return distance_table[ailego_ctz(m)][ailego_ctz(n)];
}

static inline IndexMetric::MatrixDistanceHandle EuclideanDistanceMatrixInt8(
    size_t m, size_t n) {
  static const IndexMetric::MatrixDistanceHandle distance_table[6][6] = {
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<int8_t, 1, 1>::Compute),
       nullptr, nullptr, nullptr, nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<int8_t, 2, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<int8_t, 2, 2>::Compute),
       nullptr, nullptr, nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<int8_t, 4, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<int8_t, 4, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<int8_t, 4, 4>::Compute),
       nullptr, nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<int8_t, 8, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<int8_t, 8, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<int8_t, 8, 4>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<int8_t, 8, 8>::Compute),
       nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<int8_t, 16, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<int8_t, 16, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<int8_t, 16, 4>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<int8_t, 16, 8>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<int8_t, 16, 16>::Compute),
       nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<int8_t, 32, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<int8_t, 32, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<int8_t, 32, 4>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<int8_t, 32, 8>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<int8_t, 32, 16>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<int8_t, 32, 32>::Compute)},
  };
  if (m > 32 || n > 32 || ailego_popcount(m) != 1 || ailego_popcount(n) != 1) {
    return nullptr;
  }
  return distance_table[ailego_ctz(m)][ailego_ctz(n)];
}

//! Retrieve distance function for index features in Int4
static inline IndexMetric::MatrixDistanceHandle EuclideanDistanceMatrixInt4(
    size_t m, size_t n) {
  static const IndexMetric::MatrixDistanceHandle distance_table[6][6] = {
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<uint8_t, 1, 1>::Compute),
       nullptr, nullptr, nullptr, nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<uint8_t, 2, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<uint8_t, 2, 2>::Compute),
       nullptr, nullptr, nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<uint8_t, 4, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<uint8_t, 4, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<uint8_t, 4, 4>::Compute),
       nullptr, nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<uint8_t, 8, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<uint8_t, 8, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<uint8_t, 8, 4>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<uint8_t, 8, 8>::Compute),
       nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<uint8_t, 16, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<uint8_t, 16, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<uint8_t, 16, 4>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<uint8_t, 16, 8>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<uint8_t, 16, 16>::Compute),
       nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<uint8_t, 32, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<uint8_t, 32, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<uint8_t, 32, 4>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<uint8_t, 32, 8>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<uint8_t, 32, 16>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::EuclideanDistanceMatrix<uint8_t, 32, 32>::Compute)},
  };
  if (m > 32 || n > 32 || ailego_popcount(m) != 1 || ailego_popcount(n) != 1) {
    return nullptr;
  }
  return distance_table[ailego_ctz(m)][ailego_ctz(n)];
}

//! Retrieve distance function for index features
static inline IndexMetric::MatrixDistanceHandle HammingDistanceMatrix32(
    size_t m, size_t n) {
  static const IndexMetric::MatrixDistanceHandle distance_table[6][6] = {
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint32_t, 1, 1>::Compute),
       nullptr, nullptr, nullptr, nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint32_t, 2, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint32_t, 2, 2>::Compute),
       nullptr, nullptr, nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint32_t, 4, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint32_t, 4, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint32_t, 4, 4>::Compute),
       nullptr, nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint32_t, 8, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint32_t, 8, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint32_t, 8, 4>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint32_t, 8, 8>::Compute),
       nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint32_t, 16, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint32_t, 16, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint32_t, 16, 4>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint32_t, 16, 8>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint32_t, 16, 16>::Compute),
       nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint32_t, 32, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint32_t, 32, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint32_t, 32, 4>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint32_t, 32, 8>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint32_t, 32, 16>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint32_t, 32, 32>::Compute)},
  };
  if (m > 32 || n > 32 || ailego_popcount(m) != 1 || ailego_popcount(n) != 1) {
    return nullptr;
  }
  return distance_table[ailego_ctz(m)][ailego_ctz(n)];
}

#if defined(AILEGO_M64)
static inline IndexMetric::MatrixDistanceHandle HammingDistanceMatrix64(
    size_t m, size_t n) {
  static const IndexMetric::MatrixDistanceHandle distance_table[6][6] = {
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint64_t, 1, 1>::Compute),
       nullptr, nullptr, nullptr, nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint64_t, 2, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint64_t, 2, 2>::Compute),
       nullptr, nullptr, nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint64_t, 4, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint64_t, 4, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint64_t, 4, 4>::Compute),
       nullptr, nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint64_t, 8, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint64_t, 8, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint64_t, 8, 4>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint64_t, 8, 8>::Compute),
       nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint64_t, 16, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint64_t, 16, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint64_t, 16, 4>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint64_t, 16, 8>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint64_t, 16, 16>::Compute),
       nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint64_t, 32, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint64_t, 32, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint64_t, 32, 4>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint64_t, 32, 8>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint64_t, 32, 16>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingDistanceMatrix<uint64_t, 32, 32>::Compute)},
  };
  if (m > 32 || n > 32 || ailego_popcount(m) != 1 || ailego_popcount(n) != 1) {
    return nullptr;
  }
  return distance_table[ailego_ctz(m)][ailego_ctz(n)];
}
#endif  // AILEGO_M64

//! Retrieve distance function for index features
static inline IndexMetric::MatrixDistanceHandle
HammingSquareRootDistanceMatrix32(size_t m, size_t n) {
  static const IndexMetric::MatrixDistanceHandle distance_table[6][6] = {
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingSquareRootDistanceMatrix<uint32_t, 1, 1>::Compute),
       nullptr, nullptr, nullptr, nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingSquareRootDistanceMatrix<uint32_t, 2, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingSquareRootDistanceMatrix<uint32_t, 2, 2>::Compute),
       nullptr, nullptr, nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingSquareRootDistanceMatrix<uint32_t, 4, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingSquareRootDistanceMatrix<uint32_t, 4, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingSquareRootDistanceMatrix<uint32_t, 4, 4>::Compute),
       nullptr, nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingSquareRootDistanceMatrix<uint32_t, 8, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingSquareRootDistanceMatrix<uint32_t, 8, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingSquareRootDistanceMatrix<uint32_t, 8, 4>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingSquareRootDistanceMatrix<uint32_t, 8, 8>::Compute),
       nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingSquareRootDistanceMatrix<uint32_t, 16, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingSquareRootDistanceMatrix<uint32_t, 16, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingSquareRootDistanceMatrix<uint32_t, 16, 4>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingSquareRootDistanceMatrix<uint32_t, 16, 8>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingSquareRootDistanceMatrix<uint32_t, 16, 16>::Compute),
       nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingSquareRootDistanceMatrix<uint32_t, 32, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingSquareRootDistanceMatrix<uint32_t, 32, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingSquareRootDistanceMatrix<uint32_t, 32, 4>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingSquareRootDistanceMatrix<uint32_t, 32, 8>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingSquareRootDistanceMatrix<uint32_t, 32, 16>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingSquareRootDistanceMatrix<uint32_t, 32, 32>::Compute)},
  };
  if (m > 32 || n > 32 || ailego_popcount(m) != 1 || ailego_popcount(n) != 1) {
    return nullptr;
  }
  return distance_table[ailego_ctz(m)][ailego_ctz(n)];
}

#if defined(AILEGO_M64)
static inline IndexMetric::MatrixDistanceHandle
HammingSquareRootDistanceMatrix64(size_t m, size_t n) {
  static const IndexMetric::MatrixDistanceHandle distance_table[6][6] = {
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingSquareRootDistanceMatrix<uint64_t, 1, 1>::Compute),
       nullptr, nullptr, nullptr, nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingSquareRootDistanceMatrix<uint64_t, 2, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingSquareRootDistanceMatrix<uint64_t, 2, 2>::Compute),
       nullptr, nullptr, nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingSquareRootDistanceMatrix<uint64_t, 4, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingSquareRootDistanceMatrix<uint64_t, 4, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingSquareRootDistanceMatrix<uint64_t, 4, 4>::Compute),
       nullptr, nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingSquareRootDistanceMatrix<uint64_t, 8, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingSquareRootDistanceMatrix<uint64_t, 8, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingSquareRootDistanceMatrix<uint64_t, 8, 4>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingSquareRootDistanceMatrix<uint64_t, 8, 8>::Compute),
       nullptr, nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingSquareRootDistanceMatrix<uint64_t, 16, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingSquareRootDistanceMatrix<uint64_t, 16, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingSquareRootDistanceMatrix<uint64_t, 16, 4>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingSquareRootDistanceMatrix<uint64_t, 16, 8>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingSquareRootDistanceMatrix<uint64_t, 16, 16>::Compute),
       nullptr},
      {reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingSquareRootDistanceMatrix<uint64_t, 32, 1>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingSquareRootDistanceMatrix<uint64_t, 32, 2>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingSquareRootDistanceMatrix<uint64_t, 32, 4>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingSquareRootDistanceMatrix<uint64_t, 32, 8>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingSquareRootDistanceMatrix<uint64_t, 32, 16>::Compute),
       reinterpret_cast<IndexMetric::MatrixDistanceHandle>(
           ailego::HammingSquareRootDistanceMatrix<uint64_t, 32, 32>::Compute)},
  };
  if (m > 32 || n > 32 || ailego_popcount(m) != 1 || ailego_popcount(n) != 1) {
    return nullptr;
  }
  return distance_table[ailego_ctz(m)][ailego_ctz(n)];
}
#endif  // AILEGO_M64

/*! Squared Euclidean Distance Metric
 */
class SquaredEuclideanMetric : public IndexMetric {
 public:
  //! Initialize Metric
  int init(const IndexMeta &meta, const ailego::Params &index_params) override {
    IndexMeta::DataType dt = meta.data_type();
    if (dt != IndexMeta::DataType::DT_FP16 &&
        dt != IndexMeta::DataType::DT_FP32 &&
        dt != IndexMeta::DataType::DT_INT8 &&
        dt != IndexMeta::DataType::DT_INT4 &&
        dt != IndexMeta::DataType::DT_BINARY32 &&
        dt != IndexMeta::DataType::DT_BINARY64) {
      return IndexError_Unsupported;
    }
    if (IndexMeta::UnitSizeof(dt) != meta.unit_size()) {
      return IndexError_Unsupported;
    }
    data_type_ = dt;
    params_ = index_params;

    return 0;
  }

  //! Cleanup Metric
  int cleanup(void) override {
    return 0;
  }

  //! Retrieve if it matched
  bool is_matched(const IndexMeta &meta) const override {
    return (meta.data_type() == data_type_ &&
            meta.unit_size() == IndexMeta::UnitSizeof(data_type_));
  }

  //! Retrieve if it matched
  bool is_matched(const IndexMeta &meta,
                  const IndexQueryMeta &qmeta) const override {
    return (qmeta.data_type() == data_type_ &&
            qmeta.unit_size() == IndexMeta::UnitSizeof(data_type_) &&
            qmeta.dimension() == meta.dimension());
  }

  //! Retrieve distance function for query
  MatrixDistance distance(void) const override {
    switch (data_type_) {
      case IndexMeta::DataType::DT_BINARY32:
        return reinterpret_cast<MatrixDistanceHandle>(
            ailego::HammingDistanceMatrix<uint32_t, 1, 1>::Compute);

#if defined(AILEGO_M64)
      case IndexMeta::DataType::DT_BINARY64:
        return reinterpret_cast<MatrixDistanceHandle>(
            ailego::HammingDistanceMatrix<uint64_t, 1, 1>::Compute);
#endif  // AILEGO_M64

      case IndexMeta::DataType::DT_FP16:
        return reinterpret_cast<MatrixDistanceHandle>(
            ailego::SquaredEuclideanDistanceMatrix<ailego::Float16, 1,
                                                   1>::Compute);

      case IndexMeta::DataType::DT_FP32:
        return reinterpret_cast<MatrixDistanceHandle>(
            ailego::SquaredEuclideanDistanceMatrix<float, 1, 1>::Compute);

      case IndexMeta::DataType::DT_INT8:
        return reinterpret_cast<MatrixDistanceHandle>(
            ailego::SquaredEuclideanDistanceMatrix<int8_t, 1, 1>::Compute);

      case IndexMeta::DataType::DT_INT4:
        return reinterpret_cast<MatrixDistanceHandle>(
            ailego::SquaredEuclideanDistanceMatrix<uint8_t, 1, 1>::Compute);

      default:
        return nullptr;
    }
  }

  //! Retrieve sparse distance function for query
  MatrixSparseDistance sparse_distance(void) const override {
    return reinterpret_cast<MatrixSparseDistanceHandle>(
        ailego::SquaredEuclideanSparseDistanceMatrix<float>::Compute);
  }

  //! Retrieve distance function for index features
  MatrixDistance distance_matrix(size_t m, size_t n) const override {
    switch (data_type_) {
      case IndexMeta::DataType::DT_BINARY32:
        return HammingDistanceMatrix32(m, n);

#if defined(AILEGO_M64)
      case IndexMeta::DataType::DT_BINARY64:
        return HammingDistanceMatrix64(m, n);
#endif  // AILEGO_M64

      case IndexMeta::DataType::DT_FP16:
        return SquaredEuclideanDistanceMatrixFp16(m, n);

      case IndexMeta::DataType::DT_FP32:
        return SquaredEuclideanDistanceMatrixFp32(m, n);

      case IndexMeta::DataType::DT_INT8:
        return SquaredEuclideanDistanceMatrixInt8(m, n);

      case IndexMeta::DataType::DT_INT4:
        return SquaredEuclideanDistanceMatrixInt4(m, n);

      default:
        return nullptr;
    }
  }

  //! Retrieve distance function for query
  MatrixBatchDistance batch_distance(void) const override {
    switch (data_type_) {
      case IndexMeta::DataType::DT_BINARY32:
        return reinterpret_cast<IndexMetric::MatrixBatchDistanceHandle>(
            ailego::BaseDistance<ailego::HammingDistanceMatrix, uint32_t, 1,
                                 1>::ComputeBatch);

#if defined(AILEGO_M64)
      case IndexMeta::DataType::DT_BINARY64:
        return reinterpret_cast<IndexMetric::MatrixBatchDistanceHandle>(
            ailego::BaseDistance<ailego::HammingDistanceMatrix, uint64_t, 1,
                                 1>::ComputeBatch);
#endif  // AILEGO_M64

      case IndexMeta::DataType::DT_FP16:
        return reinterpret_cast<IndexMetric::MatrixBatchDistanceHandle>(
            ailego::BaseDistance<ailego::SquaredEuclideanDistanceMatrix,
                                 ailego::Float16, 1, 1>::ComputeBatch);

      case IndexMeta::DataType::DT_FP32:
        return reinterpret_cast<IndexMetric::MatrixBatchDistanceHandle>(
            ailego::BaseDistance<ailego::SquaredEuclideanDistanceMatrix, float,
                                 1, 1>::ComputeBatch);

      case IndexMeta::DataType::DT_INT8:
        return reinterpret_cast<IndexMetric::MatrixBatchDistanceHandle>(
            ailego::BaseDistance<ailego::SquaredEuclideanDistanceMatrix, int8_t,
                                 1, 1>::ComputeBatch);

      case IndexMeta::DataType::DT_INT4:
        return reinterpret_cast<IndexMetric::MatrixBatchDistanceHandle>(
            ailego::BaseDistance<ailego::SquaredEuclideanDistanceMatrix,
                                 uint8_t, 1, 1>::ComputeBatch);

      default:
        return nullptr;
    }
  }

  //! Retrieve params of Metric
  const ailego::Params &params(void) const override {
    return params_;
  }

  //! Retrieve query metric object of this index metric
  Pointer query_metric(void) const override {
    return nullptr;
  }

 private:
  IndexMeta::DataType data_type_{IndexMeta::DataType::DT_FP32};
  ailego::Params params_{};
};

/*! Euclidean Distance Metric
 */
class EuclideanMetric : public IndexMetric {
 public:
  //! Initialize Metric
  int init(const IndexMeta &meta, const ailego::Params &index_params) override {
    IndexMeta::DataType dt = meta.data_type();
    if (dt != IndexMeta::DataType::DT_FP16 &&
        dt != IndexMeta::DataType::DT_FP32 &&
        dt != IndexMeta::DataType::DT_INT8 &&
        dt != IndexMeta::DataType::DT_INT4 &&
        dt != IndexMeta::DataType::DT_BINARY32 &&
        dt != IndexMeta::DataType::DT_BINARY64) {
      return IndexError_Unsupported;
    }
    if (IndexMeta::UnitSizeof(dt) != meta.unit_size()) {
      return IndexError_Unsupported;
    }
    data_type_ = dt;
    params_ = index_params;
    return 0;
  }

  //! Cleanup Metric
  int cleanup(void) override {
    return 0;
  }

  //! Retrieve if it matched
  bool is_matched(const IndexMeta &meta) const override {
    return (meta.data_type() == data_type_ &&
            meta.unit_size() == IndexMeta::UnitSizeof(data_type_));
  }

  //! Retrieve if it matched
  bool is_matched(const IndexMeta &meta,
                  const IndexQueryMeta &qmeta) const override {
    return (qmeta.data_type() == data_type_ &&
            qmeta.unit_size() == IndexMeta::UnitSizeof(data_type_) &&
            qmeta.dimension() == meta.dimension());
  }

  //! Retrieve distance function for query
  MatrixDistance distance(void) const override {
    switch (data_type_) {
      case IndexMeta::DataType::DT_BINARY32:
        return reinterpret_cast<MatrixDistanceHandle>(
            ailego::HammingSquareRootDistanceMatrix<uint32_t, 1, 1>::Compute);

#if defined(AILEGO_M64)
      case IndexMeta::DataType::DT_BINARY64:
        return reinterpret_cast<MatrixDistanceHandle>(
            ailego::HammingSquareRootDistanceMatrix<uint64_t, 1, 1>::Compute);
#endif  // AILEGO_M64

      case IndexMeta::DataType::DT_FP16:
        return reinterpret_cast<MatrixDistanceHandle>(
            ailego::EuclideanDistanceMatrix<ailego::Float16, 1, 1>::Compute);

      case IndexMeta::DataType::DT_FP32:
        return reinterpret_cast<MatrixDistanceHandle>(
            ailego::EuclideanDistanceMatrix<float, 1, 1>::Compute);

      case IndexMeta::DataType::DT_INT8:
        return reinterpret_cast<MatrixDistanceHandle>(
            ailego::EuclideanDistanceMatrix<int8_t, 1, 1>::Compute);

      case IndexMeta::DataType::DT_INT4:
        return reinterpret_cast<MatrixDistanceHandle>(
            ailego::EuclideanDistanceMatrix<uint8_t, 1, 1>::Compute);

      default:
        return nullptr;
    }
  }

  //! Retrieve distance function for index features
  MatrixDistance distance_matrix(size_t m, size_t n) const override {
    switch (data_type_) {
      case IndexMeta::DataType::DT_BINARY32:
        return HammingSquareRootDistanceMatrix32(m, n);

#if defined(AILEGO_M64)
      case IndexMeta::DataType::DT_BINARY64:
        return HammingSquareRootDistanceMatrix64(m, n);
#endif  // AILEGO_M64

      case IndexMeta::DataType::DT_FP16:
        return EuclideanDistanceMatrixFp16(m, n);

      case IndexMeta::DataType::DT_FP32:
        return EuclideanDistanceMatrixFp32(m, n);

      case IndexMeta::DataType::DT_INT8:
        return EuclideanDistanceMatrixInt8(m, n);

      case IndexMeta::DataType::DT_INT4:
        return EuclideanDistanceMatrixInt4(m, n);

      default:
        return nullptr;
    }
  }

  //! Retrieve params of Metric
  const ailego::Params &params(void) const override {
    return params_;
  }

  //! Retrieve query metric object of this index metric
  Pointer query_metric(void) const override {
    return nullptr;
  }

 private:
  IndexMeta::DataType data_type_{IndexMeta::DataType::DT_FP32};
  ailego::Params params_{};
};

/*! Squared Euclidean Sparse Metric
 */
class SquaredEuclideanSparseMetric : public IndexMetric {
 public:
  //! Initialize Metric
  int init(const IndexMeta &meta, const ailego::Params &index_params) override {
    IndexMeta::DataType data_type = meta.data_type();
    if (data_type != IndexMeta::DataType::DT_FP16 &&
        data_type != IndexMeta::DataType::DT_FP32) {
      return IndexError_Unsupported;
    }

    if (IndexMeta::UnitSizeof(data_type) != meta.unit_size()) {
      return IndexError_Unsupported;
    }

    data_type_ = data_type;
    params_ = index_params;

    return 0;
  }

  //! Cleanup Metric
  int cleanup(void) override {
    return 0;
  }

  //! Retrieve if it matched
  bool is_matched(const IndexMeta &meta) const override {
    return (meta.data_type() == data_type_ &&
            meta.unit_size() == IndexMeta::UnitSizeof(data_type_));
  }

  //! Retrieve if it matched
  bool is_matched(const IndexMeta &meta,
                  const IndexQueryMeta &qmeta) const override {
    return (qmeta.data_type() == data_type_ &&
            qmeta.data_type() == meta.data_type() &&
            qmeta.unit_size() == IndexMeta::UnitSizeof(data_type_) &&
            qmeta.unit_size() == meta.unit_size());
  }

  //! Retrieve sparse distance function for query
  MatrixSparseDistance sparse_distance(void) const override {
    return reinterpret_cast<MatrixSparseDistanceHandle>(
        ailego::SquaredEuclideanSparseDistanceMatrix<float>::Compute);
  }

  //! Retrieve params of Metric
  const ailego::Params &params(void) const override {
    return params_;
  }

  //! Retrieve query metric object of this index metric
  Pointer query_metric(void) const override {
    return nullptr;
  }

 private:
  IndexMeta::DataType data_type_{IndexMeta::DataType::DT_FP32};

  ailego::Params params_{};
};

INDEX_FACTORY_REGISTER_METRIC_ALIAS(SquaredEuclidean, SquaredEuclideanMetric);
INDEX_FACTORY_REGISTER_METRIC_ALIAS(Euclidean, EuclideanMetric);

INDEX_FACTORY_REGISTER_METRIC_ALIAS(SquaredEuclideanSparse,
                                    SquaredEuclideanSparseMetric);

}  // namespace core
}  // namespace zvec
