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
#include "flat_utility.h"

namespace zvec {
namespace core {

/*! Brute Force Distance Tuple
 */
template <size_t K, typename = void>
class FlatDistanceTuple;

/*! Brute Force Distance Tuple
 */
template <>
class FlatDistanceTuple<1> {
 public:
  //! Retrieve non-zero if all distances are valid.
  bool is_valid() const {
    return !!distance_;
  }

  //! Retrieve non-zero if a distance is valid.
  bool is_valid(size_t m) const {
    return m == 1 && !!distance_;
  }

  //! Initialize the distance tuple
  void initialize(const IndexMetric &measure) {
    distance_ = measure.distance_matrix(1, 1);
  }

  //! Initialize the distance tuple
  void initialize(const IndexMetric &measure, size_t m) {
    distance_ = measure.distance_matrix(m, 1);
  }

  //! Compute the distance between matrix and query
  template <size_t M>
  auto distance(const void *m, const void *q, size_t dim, float *out) const ->
      typename std::enable_if<M == 1>::type {
    distance_(m, q, dim, out);
  }

 private:
  IndexMetric::MatrixDistance distance_{};
};

/*! Brute Force Distance Tuple
 */
template <size_t K>
class FlatDistanceTuple<
    K, typename std::enable_if<IsEqualPowerofTwo<K>::value>::type> {
 public:
  //! Retrieve non-zero if all distances are valid.
  bool is_valid() const {
    return (distance_tuple_.is_valid() && !!distance_);
  }

  //! Retrieve non-zero if a distance is valid.
  bool is_valid(size_t m) const {
    return (m == K ? (!!distance_)
                   : (m < K ? distance_tuple_.is_valid(m) : false));
  }

  //! Initialize the distance tuple
  void initialize(const IndexMetric &measure) {
    distance_tuple_.initialize(measure);
    distance_ = measure.distance_matrix(K, 1);
  }

  //! Initialize the distance tuple
  void initialize(const IndexMetric &measure, size_t m) {
    distance_tuple_.initialize(measure, m);
    distance_ = measure.distance_matrix(m, K);
  }

  //! Compute the distance between matrix and query
  template <size_t M>
  auto distance(const void *m, const void *q, size_t dim, float *out) const ->
      typename std::enable_if<K == M>::type {
    distance_(m, q, dim, out);
  }

  //! Compute the distance between matrix and query
  template <size_t M>
  auto distance(const void *m, const void *q, size_t dim, float *out) const ->
      typename std::enable_if<(K > M) && IsEqualPowerofTwo<M>::value>::type {
    distance_tuple_.template distance<M>(m, q, dim, out);
  }

 private:
  FlatDistanceTuple<(K >> 1)> distance_tuple_{};
  IndexMetric::MatrixDistance distance_{};
};

/*! Brute Force Distance Matrix
 */
template <size_t K, typename = void>
class FlatDistanceMatrix;

/*! Brute Force Distance Matrix
 */
template <>
class FlatDistanceMatrix<1> {
 public:
  //! Retrieve non-zero if all distances are valid.
  bool is_valid() const {
    return (!!distance_);
  }

  //! Initialize the distance tuple
  void initialize(const IndexMetric &measure) {
    distance_ = measure.distance_matrix(1, 1);
  }

  //! Compute the distance between matrix and query
  template <size_t M, size_t N = 1u>
  auto distance(const void *m, const void *q, size_t dim, float *out) const ->
      typename std::enable_if<M == 1u && N == 1u>::type {
    distance_(m, q, dim, out);
  }

 private:
  IndexMetric::MatrixDistance distance_{};
};

/*! Brute Force Distance Matrix
 */
template <size_t K>
class FlatDistanceMatrix<
    K, typename std::enable_if<IsEqualPowerofTwo<K>::value>::type> {
 public:
  //! Retrieve non-zero if all distances are valid.
  bool is_valid() const {
    return (tuple_h_.is_valid() && tuple_v_.is_valid());
  }

  //! Retrieve non-zero if a distance is valid.
  bool is_valid(size_t m, size_t n) const {
    return (m == K ? tuple_h_.is_valid(n)
                   : (m < K && n == 1 ? tuple_v_.is_valid(m) : false));
  }

  //! Initialize the distance tuple
  void initialize(const IndexMetric &measure) {
    tuple_h_.initialize(measure, K);
    tuple_v_.initialize(measure);
  }

  //! Compute the distance between matrix and query
  template <size_t M, size_t N>
  auto distance(const void *m, const void *q, size_t dim, float *out) const ->
      typename std::enable_if<(K == M) && (K >= N)>::type {
    tuple_h_.template distance<N>(m, q, dim, out);
  }

  //! Compute the distance between matrix and query
  template <size_t M, size_t N = 1u>
  auto distance(const void *m, const void *q, size_t dim, float *out) const ->
      typename std::enable_if<(K > M) && (N == 1u)>::type {
    tuple_v_.template distance<M>(m, q, dim, out);
  }

 private:
  FlatDistanceTuple<K> tuple_h_{};
  FlatDistanceTuple<(K >> 1)> tuple_v_{};
};

}  // namespace core
}  // namespace zvec
