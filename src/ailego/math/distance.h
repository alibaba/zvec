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

#pragma once

#include "distance_matrix.h"

namespace zvec {
namespace ailego {

/*! Distance module
 */
struct Distance {
  //! Compute the squared euclidean distance between two vectors (FP32)
  static float SquaredEuclidean(const float *lhs, const float *rhs,
                                size_t dim) {
    float result;
    SquaredEuclideanDistanceMatrix<float, 1, 1>::Compute(lhs, rhs, dim,
                                                         &result);
    return result;
  }

  //! Compute the squared euclidean distance between two vectors (FP16)
  static float SquaredEuclidean(const Float16 *lhs, const Float16 *rhs,
                                size_t dim) {
    float result;
    SquaredEuclideanDistanceMatrix<Float16, 1, 1>::Compute(lhs, rhs, dim,
                                                           &result);
    return result;
  }

  //! Compute the squared euclidean distance between two vectors (INT8)
  static float SquaredEuclidean(const int8_t *lhs, const int8_t *rhs,
                                size_t dim) {
    float result;
    SquaredEuclideanDistanceMatrix<int8_t, 1, 1>::Compute(lhs, rhs, dim,
                                                          &result);
    return result;
  }

  //! Compute the squared euclidean distance between two vectors (INT4)
  static float SquaredEuclidean(const uint8_t *lhs, const uint8_t *rhs,
                                size_t dim) {
    float result;
    SquaredEuclideanDistanceMatrix<uint8_t, 1, 1>::Compute(lhs, rhs, dim,
                                                           &result);
    return result;
  }

  //! Compute the euclidean distance between two vectors (FP32)
  static float Euclidean(const float *lhs, const float *rhs, size_t dim) {
    float result;
    EuclideanDistanceMatrix<float, 1, 1>::Compute(lhs, rhs, dim, &result);
    return result;
  }

  //! Compute the euclidean distance between two vectors (FP16)
  static float Euclidean(const Float16 *lhs, const Float16 *rhs, size_t dim) {
    float result;
    EuclideanDistanceMatrix<Float16, 1, 1>::Compute(lhs, rhs, dim, &result);
    return result;
  }

  //! Compute the euclidean distance between two vectors (INT8)
  static float Euclidean(const int8_t *lhs, const int8_t *rhs, size_t dim) {
    float result;
    EuclideanDistanceMatrix<int8_t, 1, 1>::Compute(lhs, rhs, dim, &result);
    return result;
  }

  //! Compute the euclidean distance between two vectors (INT4)
  static float Euclidean(const uint8_t *lhs, const uint8_t *rhs, size_t dim) {
    float result;
    EuclideanDistanceMatrix<uint8_t, 1, 1>::Compute(lhs, rhs, dim, &result);
    return result;
  }

  //! Compute the inner product between two vectors (FP32)
  static float InnerProduct(const float *lhs, const float *rhs, size_t dim) {
    float result;
    InnerProductMatrix<float, 1, 1>::Compute(lhs, rhs, dim, &result);
    return result;
  }

  //! Compute the inner product between two vectors (FP16)
  static float InnerProduct(const Float16 *lhs, const Float16 *rhs,
                            size_t dim) {
    float result;
    InnerProductMatrix<Float16, 1, 1>::Compute(lhs, rhs, dim, &result);
    return result;
  }

  //! Compute the inner product between two vectors (INT8)
  static float InnerProduct(const int8_t *lhs, const int8_t *rhs, size_t dim) {
    float result;
    InnerProductMatrix<int8_t, 1, 1>::Compute(lhs, rhs, dim, &result);
    return result;
  }

  //! Compute the minus inner product between two vectors (INT4)
  static float InnerProduct(const uint8_t *lhs, const uint8_t *rhs,
                            size_t dim) {
    float result;
    InnerProductMatrix<uint8_t, 1, 1>::Compute(lhs, rhs, dim, &result);
    return result;
  }

  //! Compute the minus inner product between two vectors (FP32)
  static float MinusInnerProduct(const float *lhs, const float *rhs,
                                 size_t dim) {
    float result;
    MinusInnerProductMatrix<float, 1, 1>::Compute(lhs, rhs, dim, &result);
    return result;
  }

  //! Compute the minus inner product between two vectors (FP16)
  static float MinusInnerProduct(const Float16 *lhs, const Float16 *rhs,
                                 size_t dim) {
    float result;
    MinusInnerProductMatrix<Float16, 1, 1>::Compute(lhs, rhs, dim, &result);
    return result;
  }

  //! Compute the minus inner product between two vectors (INT8)
  static float MinusInnerProduct(const int8_t *lhs, const int8_t *rhs,
                                 size_t dim) {
    float result;
    MinusInnerProductMatrix<int8_t, 1, 1>::Compute(lhs, rhs, dim, &result);
    return result;
  }

  //! Compute the minus inner product between two vectors (INT4)
  static float MinusInnerProduct(const uint8_t *lhs, const uint8_t *rhs,
                                 size_t dim) {
    float result;
    MinusInnerProductMatrix<uint8_t, 1, 1>::Compute(lhs, rhs, dim, &result);
    return result;
  }

  //! Compute the mips squared L2 distance between two vectors
  //! (FP32, RepeatedQuadraticInjection)
  static float MipsSquaredEuclidean(const float *lhs, const float *rhs,
                                    size_t dim, size_t m, float eta) {
    float result;
    MipsSquaredEuclideanDistanceMatrix<float, 1, 1>::Compute(lhs, rhs, dim, m,
                                                             eta, &result);
    return result;
  }

  //! Compute the mips squared L2 distance between two vectors
  //! (FP16, RepeatedQuadraticInjection)
  static float MipsSquaredEuclidean(const Float16 *lhs, const Float16 *rhs,
                                    size_t dim, size_t m, float eta) {
    float result;
    MipsSquaredEuclideanDistanceMatrix<Float16, 1, 1>::Compute(lhs, rhs, dim, m,
                                                               eta, &result);
    return result;
  }

  //! Compute the mips squared L2 distance between two vectors
  //! (INT8, RepeatedQuadraticInjection)
  static float MipsSquaredEuclidean(const int8_t *lhs, const int8_t *rhs,
                                    size_t dim, size_t m, float eta) {
    float result;
    MipsSquaredEuclideanDistanceMatrix<int8_t, 1, 1>::Compute(lhs, rhs, dim, m,
                                                              eta, &result);
    return result;
  }

  //! Compute the mips squared L2 distance between two vectors
  //! (INT4, RepeatedQuadraticInjection)
  static float MipsSquaredEuclidean(const uint8_t *lhs, const uint8_t *rhs,
                                    size_t dim, size_t m, float eta) {
    float result;
    MipsSquaredEuclideanDistanceMatrix<uint8_t, 1, 1>::Compute(lhs, rhs, dim, m,
                                                               eta, &result);
    return result;
  }

  //! Compute the mips squared L2 distance between two vectors
  //! (FP32, SphericalInjection)
  static float MipsSquaredEuclidean(const float *lhs, const float *rhs,
                                    size_t dim, float eta) {
    float result;
    MipsSquaredEuclideanDistanceMatrix<float, 1, 1>::Compute(lhs, rhs, dim, eta,
                                                             &result);
    return result;
  }

  //! Compute the mips squared L2 distance between two vectors
  //! (FP16, SphericalInjection)
  static float MipsSquaredEuclidean(const Float16 *lhs, const Float16 *rhs,
                                    size_t dim, float eta) {
    float result;
    MipsSquaredEuclideanDistanceMatrix<Float16, 1, 1>::Compute(lhs, rhs, dim,
                                                               eta, &result);
    return result;
  }

  //! Compute the mips squared L2 distance between two vectors
  //! (INT8, SphericalInjection)
  static float MipsSquaredEuclidean(const int8_t *lhs, const int8_t *rhs,
                                    size_t dim, float eta) {
    float result;
    MipsSquaredEuclideanDistanceMatrix<int8_t, 1, 1>::Compute(lhs, rhs, dim,
                                                              eta, &result);
    return result;
  }

  //! Compute the mips squared L2 distance between two vectors
  //! (INT4, SphericalInjection)
  static float MipsSquaredEuclidean(const uint8_t *lhs, const uint8_t *rhs,
                                    size_t dim, float eta) {
    float result;
    MipsSquaredEuclideanDistanceMatrix<uint8_t, 1, 1>::Compute(lhs, rhs, dim,
                                                               eta, &result);
    return result;
  }

  //! Compute the cosine distance between two vectors (FP32)
  static float Cosine(const float *lhs, const float *rhs, size_t dim) {
    float result;
    CosineDistanceMatrix<float, 1, 1>::Compute(lhs, rhs, dim, &result);
    return result;
  }

  //! Compute the cosine distance between two vectors (FP16)
  static float Cosine(const Float16 *lhs, const Float16 *rhs, size_t dim) {
    float result;
    CosineDistanceMatrix<Float16, 1, 1>::Compute(lhs, rhs, dim, &result);
    return result;
  }

  //! Compute the cosine distance between two vectors (FP16)
  static float Cosine(const int8_t *lhs, const int8_t *rhs, size_t dim) {
    float result;
    CosineDistanceMatrix<int8_t, 1, 1>::Compute(lhs, rhs, dim, &result);
    return result;
  }
};

}  // namespace ailego
}  // namespace zvec
