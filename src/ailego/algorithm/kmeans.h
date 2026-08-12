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

#include <algorithm>
#include <cmath>
#include <random>
#include <ailego/container/vector_array.h>
#include <ailego/math/euclidean_distance_matrix.h>
#include <ailego/math/inner_product_matrix.h>
#include <ailego/math/norm2_matrix.h>
#include <ailego/math/normalizer.h>
#include <ailego/utility/matrix_helper.h>
#include <zvec/ailego/container/heap.h>
#include <zvec/ailego/internal/platform.h>
#include <zvec/ailego/utility/float_helper.h>
#include <zvec/ailego/utility/type_helper.h>
#include "lloyd_cluster.h"

namespace zvec {
namespace ailego {

/*! K-MC2 Centroids Generator
 */
template <typename T, typename TPool>
class Kmc2CentroidsGenerator {
 public:
  //! Type of values
  using OwnerType = typename std::decay<T>::type;
  using ContainerType = typename OwnerType::ContainerType;
  using ContextType = typename OwnerType::ContextType;
  using ValueType = typename OwnerType::ValueType;
  using StoreType = typename OwnerType::StoreType;
  using ThreadPoolType = TPool;

  //! constexpr variables
  constexpr static size_t BatchCount = OwnerType::BatchCount;
  constexpr static bool NeedsSphericalWeight =
      ContextType::NeedsSphericalKmc2Weight;

  //! Generate centroids
  void operator()(OwnerType *owner, ThreadPoolType &pool) const {
    if (chain_length_ == 0) {
      this->init_centroids_random(owner);
    } else if (!assumption_free_) {
      this->init_centroids_kmc2(owner, pool);
    } else {
      this->init_centroids_afkmc2(owner, pool);
    }
  }

  //! Retrieve the markov chain length
  size_t chain_length(void) const {
    return chain_length_;
  }

  //! Set the mutable markov chain length
  void set_chain_length(size_t len) {
    chain_length_ = len;
  }

  //! Retrieve assumption free option
  bool assumption_free(void) const {
    return assumption_free_;
  }

  //! Set the assumption free option
  void set_assumption_free(bool val) {
    assumption_free_ = val;
  }

 protected:
  //! Initialize centroids randomly
  void init_centroids_random(OwnerType *owner) const {
    RandomSelectBenches(owner->feature_cache(), owner->feature_matrix(),
                        owner->k_value(), owner->mutable_centroids());
  }

  //! Initialize centroids with K-MC2
  void init_centroids_kmc2(OwnerType *owner, ThreadPoolType &pool) const {
    const auto &matrix = owner->feature_matrix();
    const auto &cache = owner->feature_cache();
    auto *centroids = owner->mutable_centroids();

    std::mt19937 mt((std::random_device())());

    std::uniform_real_distribution<float> dist(0.0, 1.0);

    ContainerType benches(cache.dimension());
    std::vector<float> scores;
    std::vector<float> centroid_norms;

    // Sample first center uniformly
    RandomSelectBenches(cache, matrix, 1, centroids);

    // Make a thread group
    auto group = pool.make_group();

    for (size_t i = 1, k = owner->k_value(); i < k; ++i) {
      RandomSelectBenches(cache, matrix, chain_length_, &benches);
      UpdateCentroidNorms(owner, &centroid_norms);

      // Update bench scores
      scores.resize(benches.count());
      for (size_t j = 0; j != scores.size(); ++j) {
        group->submit(Closure::New(&Kmc2CentroidsGenerator::UpdateBenchScores,
                                   owner, benches[j], centroid_norms.data(),
                                   &scores[j]));
      }
      group->wait_finish();

      //! Select the better centroid randomly
      float x = scores[0];
      size_t xj = 0;
      for (size_t j = 1; j != scores.size(); ++j) {
        float y = scores[j];

        if (x == 0.0f || x * dist(mt) < y) {
          x = y;
          xj = j;
        }
      }
      centroids->append(benches[xj], benches.dimension());
    }  // end of for
  }

  //! Initialize centroids with K-MC2
  void init_centroids_afkmc2(OwnerType *owner, ThreadPoolType &pool) const {
    const auto &matrix = owner->feature_matrix();
    const auto &cache = owner->feature_cache();

    // Probability
    std::vector<float> probs(matrix.count() + cache.count());

    // Sample first center uniformly
    RandomSelectBenches(cache, matrix, 1, owner->mutable_centroids());

    // Make a thread group
    auto group = pool.make_group();
    if (!matrix.empty()) {
      size_t n = matrix.count() / BatchCount;
      size_t c = std::max<size_t>(n / pool.count() / 2u, 1u);
      size_t m = n / c * c;

      for (size_t i = 0; i != m; i += c) {
        group->submit(Closure::New(&Kmc2CentroidsGenerator::UpdateMatrixScores,
                                   owner, i, i + c, &probs[0]));
      }
      for (size_t i = m; i != n; i += 1) {
        group->submit(Closure::New(&Kmc2CentroidsGenerator::UpdateMatrixScores,
                                   owner, i, i + 1, &probs[0]));
      }
    }
    if (!cache.empty()) {
      group->submit(Closure::New(&Kmc2CentroidsGenerator::UpdateCacheScores,
                                 owner, &probs[matrix.count()]));
    }
    group->wait_finish();

    // Update probabilities
    NormalizeProbabilities(&probs);

    std::mt19937 mt((std::random_device())());
    std::uniform_real_distribution<float> dist(0.0, 1.0);
    ContainerType benches(cache.dimension());
    std::vector<float> scores;
    std::vector<float> bench_probs;
    std::vector<float> centroid_norms;

    for (size_t i = 1; i < owner->k_value(); ++i) {
      RandomSelectBenches(cache, matrix, chain_length_, probs, &benches,
                          &bench_probs);
      UpdateCentroidNorms(owner, &centroid_norms);

      // Update bench scores
      scores.resize(benches.count());
      for (size_t j = 0; j != scores.size(); ++j) {
        group->submit(Closure::New(&Kmc2CentroidsGenerator::UpdateBenchScores,
                                   owner, benches[j], centroid_norms.data(),
                                   &scores[j]));
      }
      group->wait_finish();

      // Update scores with probabilities
      for (size_t j = 0; j != scores.size(); ++j) {
        scores[j] /= bench_probs[j];
      }

      //! Select the better centroid randomly
      float x = scores[0];
      size_t xj = 0;
      for (size_t j = 1; j != scores.size(); ++j) {
        float y = scores[j];

        if (x == 0.0f || x * dist(mt) < y) {
          x = y;
          xj = j;
        }
      }
      owner->mutable_centroids()->append(benches[xj], benches.dimension());
    }  // end of for
  }

  //! Update matrix score
  static void UpdateMatrixScores(const OwnerType *owner, size_t first,
                                 size_t last, float *out) {
    const auto &matrix = owner->feature_matrix();
    const auto *bench = owner->centroids().data();
    ContainerType rows(matrix.dimension());
    float bench_norm = 0.0f;

    if constexpr (NeedsSphericalWeight) {
      if (owner->spherical()) {
        rows.resize(BatchCount);
        bench_norm = ContextType::Kmc2Norm(bench, matrix.dimension());
      }
    }

    for (size_t i = first * BatchCount; i != last * BatchCount;
         i += BatchCount) {
      ContextType::template BatchDistance<1>(matrix[i], bench,
                                             matrix.dimension(), &out[i]);
      if constexpr (NeedsSphericalWeight) {
        if (owner->spherical()) {
          ContextType::MatrixReverseTranspose(matrix[i], matrix.dimension(),
                                              rows.data());
          for (size_t j = 0; j < BatchCount; ++j) {
            float feature_norm =
                ContextType::Kmc2Norm(rows[j], matrix.dimension());
            out[i + j] =
                ContextType::Kmc2Weight(out[i + j], bench_norm, feature_norm);
          }
        }
      }
    }
  }

  //! Update cache score
  static void UpdateCacheScores(const OwnerType *owner, float *out) {
    const auto &cache = owner->feature_cache();
    const auto *bench = owner->centroids().data();
    float bench_norm = 0.0f;

    if constexpr (NeedsSphericalWeight) {
      if (owner->spherical()) {
        bench_norm = ContextType::Kmc2Norm(bench, cache.dimension());
      }
    }

    for (size_t i = 0, n = cache.count(); i != n; ++i) {
      ContextType::Distance(bench, cache[i], cache.dimension(), &out[i]);
      if constexpr (NeedsSphericalWeight) {
        if (owner->spherical()) {
          float feature_norm =
              ContextType::Kmc2Norm(cache[i], cache.dimension());
          out[i] = ContextType::Kmc2Weight(out[i], bench_norm, feature_norm);
        }
      }
    }
  }

  //! Update centroid norms for spherical K-MC2 weights
  static void UpdateCentroidNorms(const OwnerType *owner,
                                  std::vector<float> *out) {
    out->clear();
    if constexpr (NeedsSphericalWeight) {
      if (owner->spherical()) {
        const auto &centroids = owner->centroids();
        out->resize(centroids.count());
        for (size_t i = 0; i < centroids.count(); ++i) {
          (*out)[i] =
              ContextType::Kmc2Norm(centroids[i], centroids.dimension());
        }
      }
    }
  }

  //! Update bench score
  static void UpdateBenchScores(const OwnerType *owner, const StoreType *feat,
                                const float *centroid_norms, float *out) {
    const auto &benches = owner->centroids();
    float min_score = std::numeric_limits<float>::max();
    float feature_norm = 0.0f;

    if constexpr (NeedsSphericalWeight) {
      if (owner->spherical()) {
        feature_norm = ContextType::Kmc2Norm(feat, benches.dimension());
      }
    }

    for (size_t i = 0, c = benches.count(); i != c; ++i) {
      float new_score;
      ContextType::Distance(benches.at(i), feat, benches.dimension(),
                            &new_score);
      if constexpr (NeedsSphericalWeight) {
        if (owner->spherical()) {
          new_score = ContextType::Kmc2Weight(new_score, centroid_norms[i],
                                              feature_norm);
        }
      }

      if (new_score < min_score) {
        min_score = new_score;
      }
    }
    *out = min_score;
  }

  //! Normalize AFK-MC2 sampling probabilities
  static void NormalizeProbabilities(std::vector<float> *probs) {
    double p_sum = 0.0;
    bool valid = true;
    for (float prob : *probs) {
      p_sum += prob;
      valid = valid && prob >= 0.0f && std::isfinite(prob);
    }

    const double uniform_prob = 1.0 / probs->size();
    if (valid && p_sum > 0.0 && std::isfinite(p_sum)) {
      for (auto &prob : *probs) {
        prob = static_cast<float>((prob / p_sum + uniform_prob) * 0.5);
      }
    } else {
      std::fill(probs->begin(), probs->end(), static_cast<float>(uniform_prob));
    }
  }

  //! Select k benches randomly
  static void RandomSelectBenches(const ContainerType &cache,
                                  const ContainerType &matrix, size_t k,
                                  ContainerType *benches) {
    ContainerType rows(cache.dimension());
    size_t m = matrix.count();
    size_t n = m + cache.count();
    std::mt19937 mt((std::random_device())());

    rows.resize(BatchCount);
    benches->reset(cache.dimension());
    benches->reserve(k);

    for (size_t i = 0; k > 0 && i < n; ++i) {
      if (mt() % (n - i) >= k) {
        continue;
      }
      // Selected a feature
      if (i < m) {
        ContextType::MatrixReverseTranspose(matrix[i / BatchCount * BatchCount],
                                            matrix.dimension(), rows.data());
        benches->append(rows[i & (BatchCount - 1u)], matrix.dimension());
      } else {
        benches->append(cache[i - m], cache.dimension());
      }
      --k;
    }  // end of for
  }

  //! Select k benches randomly
  static void RandomSelectBenches(const ContainerType &cache,
                                  const ContainerType &matrix, size_t k,
                                  const std::vector<float> &probs,
                                  ContainerType *benches,
                                  std::vector<float> *bench_probs) {
    std::mt19937 mt((std::random_device())());
    std::uniform_real_distribution<float> dist(0.0, 1.0);

    // Sample features
    KeyValueHeap<size_t, double, std::greater<double>> samples(k);
    for (size_t i = 0; i < probs.size(); ++i) {
      samples.emplace(i, std::pow(dist(mt), 1.0 / probs[i]));
    }

    ContainerType rows(cache.dimension());
    size_t matrix_count = matrix.count();

    rows.resize(BatchCount);
    benches->reset(cache.dimension());
    benches->reserve(k);
    bench_probs->clear();
    bench_probs->reserve(k);

    for (const auto &it : samples) {
      // Selected a feature
      if (it.first < matrix_count) {
        ContextType::MatrixReverseTranspose(
            matrix[it.first / BatchCount * BatchCount], matrix.dimension(),
            rows.data());
        benches->append(rows[it.first & (BatchCount - 1u)], matrix.dimension());
      } else {
        benches->append(cache[it.first - matrix_count], cache.dimension());
      }
      bench_probs->push_back(probs[it.first]);
    }
  }

 private:
  size_t chain_length_{32};
  bool assumption_free_{false};
};

/*! Numerical K-Means Context
 */
template <typename T, size_t BATCH_COUNT = 32u>
class NumericalKmeansContext {
 public:
  //! constexpr variables
  constexpr static size_t BatchCount = BATCH_COUNT;
  constexpr static bool NeedsSphericalKmc2Weight = false;

  //! Type of values
  using ValueType = typename std::remove_cv<T>::type;
  using StoreType = typename std::remove_cv<T>::type;

  // Check supporting type
  static_assert(IsSignedArithmetic<ValueType>::value,
                "ValueType must be signed arithmetic");

  /*! K-Means Context Cluster
   */
  class Cluster {
   public:
    //! Constructor
    Cluster(size_t dim) : accum_(dim, 0.0) {}

    //! Constructor
    Cluster(const Cluster &rhs)
        : cost_(rhs.cost_), count_(rhs.count_), accum_(rhs.accum_) {}

    //! Constructor
    Cluster(Cluster &&rhs)
        : cost_(rhs.cost_), count_(rhs.count_), accum_(std::move(rhs.accum_)) {}

    //! Assignment
    Cluster &operator=(const Cluster &rhs) {
      cost_ = rhs.cost_;
      count_ = rhs.count_;
      accum_ = rhs.accum_;
      return *this;
    }

    //! Assignment
    Cluster &operator=(Cluster &&rhs) {
      cost_ = rhs.cost_;
      count_ = rhs.count_;
      accum_ = std::move(rhs.accum_);
      return *this;
    }

    //! Append a vector
    void append(const ValueType *vec, size_t dim, float dist) {
      ailego_check_with(dim == accum_.size(), "Unmatched dimension");

      mutex_.lock();
      cost_ += dist;
      count_ += 1;

      for (size_t i = 0; i != dim; ++i) {
        accum_[i] += vec[i];
      }
      mutex_.unlock();
    }

    //! Retrieve the centroid of vectors
    void centroid(ValueType *out, size_t dim) const {
      ailego_check_with(dim == accum_.size(), "Unmatched dimension");

      for (size_t i = 0; i != dim; ++i) {
        out[i] = count_ == 0 ? FloatCast<ValueType>(NAN)
                             : FloatCast<ValueType>(accum_[i] / count_);
      }
    }

    //! Retrieve squared error
    double cost(void) const {
      return cost_;
    }

    //! Retrieve feature count
    size_t count(void) const {
      return count_;
    }

   protected:
    //! Convert float type to another type
    template <typename U>
    static auto FloatCast(const double val) ->
        typename std::enable_if<!std::is_integral<U>::value, U>::type {
      return static_cast<U>(val);
    }

    //! Convert float type to another type
    template <typename U>
    static auto FloatCast(const double val) ->
        typename std::enable_if<std::is_integral<U>::value, U>::type {
      return static_cast<U>(std::round(val));
    }

   private:
    SpinMutex mutex_{};
    double cost_{0.0};
    size_t count_{0u};
    std::vector<double> accum_{};
  };

  //! operator []
  const Cluster &operator[](size_t i) const {
    return clusters_[i];
  }

  //! operator []
  Cluster &operator[](size_t i) {
    return clusters_[i];
  }

  //! Clear the context
  void clear(void) {
    clusters_.clear();
  }

  //! Reset the context
  void reset(size_t k_value, size_t dim) {
    clusters_.clear();
    clusters_.resize(k_value, dim);
  }

  //! Retrieve context of clusters
  const std::vector<Cluster> &clusters(void) const {
    return clusters_;
  }

  //! Compute the distance between matrix and query (batch)
  template <size_t N>
  static void BatchDistance(const ValueType *m, const ValueType *q, size_t dim,
                            float *out) {
    SquaredEuclideanDistanceMatrix<ValueType, BatchCount, N>::Compute(m, q, dim,
                                                                      out);
  }

  //! Compute the distance between matrix and query (single)
  static void Distance(const ValueType *m, const ValueType *q, size_t dim,
                       float *out) {
    SquaredEuclideanDistanceMatrix<ValueType, 1, 1>::Compute(m, q, dim, out);
  }

  //! Transpose a matrix
  template <typename U>
  static auto MatrixTranspose(const U *src, size_t dim, T *dst) ->
      typename std::enable_if<sizeof(U) >= 2>::type {
    MatrixHelper::Transpose<U, BatchCount>(src, dim, dst);
  }

  //! Transpose a matrix
  template <typename U>
  static auto MatrixTranspose(const U *src, size_t dim, U *dst) ->
      typename std::enable_if<sizeof(U) == 1>::type {
    MatrixHelper::Transpose<uint32_t, BatchCount>(src, dim >> 2, dst);
  }

  //! Reverse transpose a matrix
  template <typename U>
  static auto MatrixReverseTranspose(const U *src, size_t dim, U *dst) ->
      typename std::enable_if<sizeof(U) >= 2>::type {
    MatrixHelper::ReverseTranspose<U, BatchCount>(src, dim, dst);
  }

  //! Reverse transpose a matrix
  template <typename U>
  static auto MatrixReverseTranspose(const U *src, size_t dim, U *dst) ->
      typename std::enable_if<sizeof(U) == 1>::type {
    MatrixHelper::ReverseTranspose<uint32_t, BatchCount>(src, dim >> 2, dst);
  }

  //! Normalize with L2 norm for floating-point values, otherwise do nothing
  static void Norm2(ValueType *data, size_t dim, float *norm) {
    if constexpr (IsFloatingPoint<ValueType>::value) {
      Normalizer<ValueType>::L2(data, dim, norm);
    } else {
      (void)data;
      (void)dim;
      *norm = 0.0f;
    }
  }

 private:
  //! Members
  std::vector<Cluster> clusters_{};
};

/*! Nibble K-Means Context (INT4)
 */
template <typename T, size_t BATCH_COUNT = 32u>
class NibbleKmeansContext {
 public:
  //! constexpr variables
  constexpr static size_t BatchCount = BATCH_COUNT;
  constexpr static bool NeedsSphericalKmc2Weight = false;

  //! Type of values
  using ValueType = typename std::remove_cv<T>::type;
  using StoreType = typename std::make_unsigned<ValueType>::type;

  // Check supporting type
  static_assert(std::is_same<ValueType, int32_t>::value ||
                    std::is_same<ValueType, int64_t>::value,
                "ValueType must be int32_t or int64_t");

  /*! K-Means Context Cluster
   */
  class Cluster {
   public:
    //! Constructor
    Cluster(size_t dim) : accum_(dim, 0.0) {}

    //! Constructor
    Cluster(const Cluster &rhs)
        : cost_(rhs.cost_), count_(rhs.count_), accum_(rhs.accum_) {}

    //! Constructor
    Cluster(Cluster &&rhs)
        : cost_(rhs.cost_), count_(rhs.count_), accum_(std::move(rhs.accum_)) {}

    //! Assignment
    Cluster &operator=(const Cluster &rhs) {
      cost_ = rhs.cost_;
      count_ = rhs.count_;
      accum_ = rhs.accum_;
      return *this;
    }

    //! Assignment
    Cluster &operator=(Cluster &&rhs) {
      cost_ = rhs.cost_;
      count_ = rhs.count_;
      accum_ = std::move(rhs.accum_);
      return *this;
    }

    //! Append a vector
    void append(const StoreType *vec, size_t dim, float dist) {
      ailego_check_with(dim == accum_.size(), "Unmatched dimension");

      mutex_.lock();
      cost_ += dist;
      count_ += 1;

      const uint8_t *arr = reinterpret_cast<const uint8_t *>(vec);
      dim = (dim >> 1) << 1;
      for (size_t i = 0; i != dim; i += 2) {
        uint8_t val = arr[i >> 1];
        accum_[i] += ((int8_t)(val << 4) >> 4);
        accum_[i + 1] += ((int8_t)(val) >> 4);
      }
      mutex_.unlock();
    }

    //! Retrieve the centroid of vectors
    void centroid(StoreType *out, size_t dim) const {
      ailego_check_with(dim == accum_.size(), "Unmatched dimension");

      uint8_t *arr = reinterpret_cast<uint8_t *>(out);
      dim = (dim >> 1) << 1;
      for (size_t i = 0; i != dim; i += 2) {
        int lo =
            count_ == 0 ? 0 : static_cast<int>(std::round(accum_[i] / count_));
        int hi = count_ == 0
                     ? 0
                     : static_cast<int>(std::round(accum_[i + 1] / count_));
        arr[i >> 1] = (uint8_t)((hi << 4) & 0xf0) | (uint8_t)(lo & 0xf);
      }
    }

    //! Retrieve squared error
    double cost(void) const {
      return cost_;
    }

    //! Retrieve feature count
    size_t count(void) const {
      return count_;
    }

   private:
    SpinMutex mutex_{};
    double cost_{0.0};
    size_t count_{0u};
    std::vector<double> accum_{};
  };

  //! operator []
  const Cluster &operator[](size_t i) const {
    return clusters_[i];
  }

  //! operator []
  Cluster &operator[](size_t i) {
    return clusters_[i];
  }

  //! Clear the context
  void clear(void) {
    clusters_.clear();
  }

  //! Reset the context
  void reset(size_t k_value, size_t dim) {
    clusters_.clear();
    clusters_.resize(k_value, dim);
  }

  //! Retrieve context of clusters
  const std::vector<Cluster> &clusters(void) const {
    return clusters_;
  }

  //! Compute the distance between matrix and query (batch)
  template <size_t N>
  static void BatchDistance(const StoreType *m, const StoreType *q, size_t dim,
                            float *out) {
    SquaredEuclideanDistanceMatrix<uint8_t, BatchCount, N>::Compute(
        reinterpret_cast<const uint8_t *>(m),
        reinterpret_cast<const uint8_t *>(q), dim, out);
  }

  //! Compute the distance between matrix and query (single)
  static void Distance(const StoreType *m, const StoreType *q, size_t dim,
                       float *out) {
    SquaredEuclideanDistanceMatrix<uint8_t, 1, 1>::Compute(
        reinterpret_cast<const uint8_t *>(m),
        reinterpret_cast<const uint8_t *>(q), dim, out);
  }

  //! Transpose a matrix
  static void MatrixTranspose(const StoreType *src, size_t dim,
                              StoreType *dst) {
    MatrixHelper::Transpose<uint32_t, BatchCount>(src, dim >> 3, dst);
  }

  //! Reverse transpose a matrix
  static void MatrixReverseTranspose(const StoreType *src, size_t dim,
                                     StoreType *dst) {
    MatrixHelper::ReverseTranspose<uint32_t, BatchCount>(src, dim >> 3, dst);
  }

  //! Compute and do norm2
  static void Norm2(StoreType * /*data*/, size_t /*dim*/, float *norm) {
    *norm = 0;
  }

 private:
  //! Members
  std::vector<Cluster> clusters_{};
};

/*! Numerical InnerProduct K-Means Context
 */
template <typename T, size_t BATCH_COUNT = 32u>
class NumericalInnerProductKmeansContext {
 public:
  //! constexpr variables
  constexpr static size_t BatchCount = BATCH_COUNT;
  constexpr static bool NeedsSphericalKmc2Weight =
      IsFloatingPoint<typename std::remove_cv<T>::type>::value;

  //! Type of values
  using ValueType = typename std::remove_cv<T>::type;
  using StoreType = typename std::remove_cv<T>::type;

  // Check supporting type
  static_assert(IsSignedArithmetic<ValueType>::value,
                "ValueType must be signed arithmetic");

  /*! K-Means Context Cluster
   */
  class Cluster {
   public:
    //! Constructor
    Cluster(size_t dim) : accum_(dim, 0.0) {}

    //! Constructor
    Cluster(const Cluster &rhs)
        : cost_(rhs.cost_), count_(rhs.count_), accum_(rhs.accum_) {}

    //! Constructor
    Cluster(Cluster &&rhs)
        : cost_(rhs.cost_), count_(rhs.count_), accum_(std::move(rhs.accum_)) {}

    //! Assignment
    Cluster &operator=(const Cluster &rhs) {
      cost_ = rhs.cost_;
      count_ = rhs.count_;
      accum_ = rhs.accum_;
      return *this;
    }

    //! Assignment
    Cluster &operator=(Cluster &&rhs) {
      cost_ = rhs.cost_;
      count_ = rhs.count_;
      accum_ = std::move(rhs.accum_);
      return *this;
    }

    //! Append a vector
    void append(const ValueType *vec, size_t dim, float dist) {
      ailego_check_with(dim == accum_.size(), "Unmatched dimension");

      mutex_.lock();
      cost_ += dist;
      count_ += 1;

      for (size_t i = 0; i != dim; ++i) {
        accum_[i] += vec[i];
      }
      mutex_.unlock();
    }

    //! Retrieve the centroid of vectors
    void centroid(ValueType *out, size_t dim) const {
      ailego_check_with(dim == accum_.size(), "Unmatched dimension");

      for (size_t i = 0; i != dim; ++i) {
        out[i] = count_ == 0 ? FloatCast<ValueType>(NAN)
                             : FloatCast<ValueType>(accum_[i] / count_);
      }
    }

    //! Retrieve squared error
    double cost(void) const {
      return cost_;
    }

    //! Retrieve feature count
    size_t count(void) const {
      return count_;
    }

   protected:
    //! Convert float type to another type
    template <typename U>
    static auto FloatCast(const double val) ->
        typename std::enable_if<!std::is_integral<U>::value, U>::type {
      return static_cast<U>(val);
    }

    //! Convert float type to another type
    template <typename U>
    static auto FloatCast(const double val) ->
        typename std::enable_if<std::is_integral<U>::value, U>::type {
      return static_cast<U>(std::round(val));
    }

   private:
    SpinMutex mutex_{};
    double cost_{0.0};
    size_t count_{0u};
    std::vector<double> accum_{};
  };

  //! operator []
  const Cluster &operator[](size_t i) const {
    return clusters_[i];
  }

  //! operator []
  Cluster &operator[](size_t i) {
    return clusters_[i];
  }

  //! Clear the context
  void clear(void) {
    clusters_.clear();
  }

  //! Reset the context
  void reset(size_t k_value, size_t dim) {
    clusters_.clear();
    clusters_.resize(k_value, dim);
  }

  //! Retrieve context of clusters
  const std::vector<Cluster> &clusters(void) const {
    return clusters_;
  }

  //! Compute the distance between matrix and query (batch)
  template <size_t N>
  static void BatchDistance(const ValueType *m, const ValueType *q, size_t dim,
                            float *out) {
    MinusInnerProductMatrix<ValueType, BatchCount, N>::Compute(m, q, dim, out);
  }

  //! Compute the distance between matrix and query (single)
  static void Distance(const ValueType *m, const ValueType *q, size_t dim,
                       float *out) {
    MinusInnerProductMatrix<ValueType, 1, 1>::Compute(m, q, dim, out);
  }

  //! Compute the L2 norm used by spherical K-MC2 initialization
  static float Kmc2Norm(const ValueType *vec, size_t dim) {
    float score;
    Distance(vec, vec, dim, &score);
    return std::sqrt(std::max(-score, 0.0f));
  }

  //! Convert -dot to a non-negative spherical K-MC2 sampling weight
  static float Kmc2Weight(float score, float centroid_norm,
                          float feature_norm) {
    // Spherical centroids are unit vectors during Lloyd iterations. K-MC2
    // benches are raw input vectors, so account for their norm here. This is
    // ||x|| - dot(x, c / ||c||), which reduces to cosine distance when both
    // vectors are normalized.
    if (centroid_norm == 0.0f) {
      return feature_norm;
    }
    return std::max(feature_norm + score / centroid_norm, 0.0f);
  }

  //! Transpose a matrix
  template <typename U>
  static auto MatrixTranspose(const U *src, size_t dim, T *dst) ->
      typename std::enable_if<sizeof(U) >= 2>::type {
    MatrixHelper::Transpose<U, BatchCount>(src, dim, dst);
  }

  //! Transpose a matrix
  template <typename U>
  static auto MatrixTranspose(const U *src, size_t dim, U *dst) ->
      typename std::enable_if<sizeof(U) == 1>::type {
    MatrixHelper::Transpose<uint32_t, BatchCount>(src, dim >> 2, dst);
  }

  //! Reverse transpose a matrix
  template <typename U>
  static auto MatrixReverseTranspose(const U *src, size_t dim, U *dst) ->
      typename std::enable_if<sizeof(U) >= 2>::type {
    MatrixHelper::ReverseTranspose<U, BatchCount>(src, dim, dst);
  }

  //! Reverse transpose a matrix
  template <typename U>
  static auto MatrixReverseTranspose(const U *src, size_t dim, U *dst) ->
      typename std::enable_if<sizeof(U) == 1>::type {
    MatrixHelper::ReverseTranspose<uint32_t, BatchCount>(src, dim >> 2, dst);
  }

  //! Normalize with L2 norm for floating-point values, otherwise do nothing
  static void Norm2(ValueType *data, size_t dim, float *norm) {
    if constexpr (IsFloatingPoint<ValueType>::value) {
      Normalizer<ValueType>::L2(data, dim, norm);
    } else {
      (void)data;
      (void)dim;
      *norm = 0.0f;
    }
  }

 private:
  //! Members
  std::vector<Cluster> clusters_{};
};

/*! Nibble InnerProduct K-Means Context (INT4)
 */
template <typename T, size_t BATCH_COUNT = 32u>
class NibbleInnerProductKmeansContext {
 public:
  //! constexpr variables
  constexpr static size_t BatchCount = BATCH_COUNT;
  constexpr static bool NeedsSphericalKmc2Weight = false;

  //! Type of values
  using ValueType = typename std::remove_cv<T>::type;
  using StoreType = typename std::make_unsigned<ValueType>::type;

  // Check supporting type
  static_assert(std::is_same<ValueType, int32_t>::value ||
                    std::is_same<ValueType, int64_t>::value,
                "ValueType must be int32_t or int64_t");

  /*! K-Means Context Cluster
   */
  class Cluster {
   public:
    //! Constructor
    Cluster(size_t dim) : accum_(dim, 0.0) {}

    //! Constructor
    Cluster(const Cluster &rhs)
        : cost_(rhs.cost_), count_(rhs.count_), accum_(rhs.accum_) {}

    //! Constructor
    Cluster(Cluster &&rhs)
        : cost_(rhs.cost_), count_(rhs.count_), accum_(std::move(rhs.accum_)) {}

    //! Assignment
    Cluster &operator=(const Cluster &rhs) {
      cost_ = rhs.cost_;
      count_ = rhs.count_;
      accum_ = rhs.accum_;
      return *this;
    }

    //! Assignment
    Cluster &operator=(Cluster &&rhs) {
      cost_ = rhs.cost_;
      count_ = rhs.count_;
      accum_ = std::move(rhs.accum_);
      return *this;
    }

    //! Append a vector
    void append(const StoreType *vec, size_t dim, float dist) {
      ailego_check_with(dim == accum_.size(), "Unmatched dimension");

      mutex_.lock();
      cost_ += dist;
      count_ += 1;

      const uint8_t *arr = reinterpret_cast<const uint8_t *>(vec);
      dim = (dim >> 1) << 1;
      for (size_t i = 0; i != dim; i += 2) {
        uint8_t val = arr[i >> 1];
        accum_[i] += ((int8_t)(val << 4) >> 4);
        accum_[i + 1] += ((int8_t)(val) >> 4);
      }
      mutex_.unlock();
    }

    //! Retrieve the centroid of vectors
    void centroid(StoreType *out, size_t dim) const {
      ailego_check_with(dim == accum_.size(), "Unmatched dimension");

      uint8_t *arr = reinterpret_cast<uint8_t *>(out);
      dim = (dim >> 1) << 1;
      for (size_t i = 0; i != dim; i += 2) {
        int lo =
            count_ == 0 ? 0 : static_cast<int>(std::round(accum_[i] / count_));
        int hi = count_ == 0
                     ? 0
                     : static_cast<int>(std::round(accum_[i + 1] / count_));
        arr[i >> 1] = (uint8_t)((hi << 4) & 0xf0) | (uint8_t)(lo & 0xf);
      }
    }

    //! Retrieve squared error
    double cost(void) const {
      return cost_;
    }

    //! Retrieve feature count
    size_t count(void) const {
      return count_;
    }

   private:
    SpinMutex mutex_{};
    double cost_{0.0};
    size_t count_{0u};
    std::vector<double> accum_{};
  };

  //! operator []
  const Cluster &operator[](size_t i) const {
    return clusters_[i];
  }

  //! operator []
  Cluster &operator[](size_t i) {
    return clusters_[i];
  }

  //! Clear the context
  void clear(void) {
    clusters_.clear();
  }

  //! Reset the context
  void reset(size_t k_value, size_t dim) {
    clusters_.clear();
    clusters_.resize(k_value, dim);
  }

  //! Retrieve context of clusters
  const std::vector<Cluster> &clusters(void) const {
    return clusters_;
  }

  //! Compute the distance between matrix and query (batch)
  template <size_t N>
  static void BatchDistance(const StoreType *m, const StoreType *q, size_t dim,
                            float *out) {
    MinusInnerProductMatrix<uint8_t, BatchCount, N>::Compute(
        reinterpret_cast<const uint8_t *>(m),
        reinterpret_cast<const uint8_t *>(q), dim, out);
  }

  //! Compute the distance between matrix and query (single)
  static void Distance(const StoreType *m, const StoreType *q, size_t dim,
                       float *out) {
    MinusInnerProductMatrix<uint8_t, 1, 1>::Compute(
        reinterpret_cast<const uint8_t *>(m),
        reinterpret_cast<const uint8_t *>(q), dim, out);
  }

  //! Transpose a matrix
  static void MatrixTranspose(const StoreType *src, size_t dim,
                              StoreType *dst) {
    MatrixHelper::Transpose<uint32_t, BatchCount>(src, dim >> 3, dst);
  }

  //! Reverse transpose a matrix
  static void MatrixReverseTranspose(const StoreType *src, size_t dim,
                                     StoreType *dst) {
    MatrixHelper::ReverseTranspose<uint32_t, BatchCount>(src, dim >> 3, dst);
  }

  //! Compute Norm2
  static void Norm2(StoreType * /*data*/, size_t /*dim*/, float *norm) {
    *norm = 0;
  }

 private:
  //! Members
  std::vector<Cluster> clusters_{};
};

/*! Numerical K-Means cluster algorithm
 */
template <typename T, typename TPool,
          typename TContext = NumericalKmeansContext<T>>
using NumericalKmeans =
    LloydCluster<T, TPool, TContext, NumericalVectorArray<T>>;

/*! Nibble K-Means cluster algorithm
 */
template <typename T, typename TPool,
          typename TContext = NibbleKmeansContext<T>>
using NibbleKmeans = LloydCluster<T, TPool, TContext, NibbleVectorArray<T>>;

/*! Numerical K-Means cluster algorithm
 */
template <typename T, typename TPool,
          typename TContext = NumericalInnerProductKmeansContext<T>>
using NumericalInnerProductKmeans =
    LloydCluster<T, TPool, TContext, NumericalVectorArray<T>>;

/*! Nibble K-Means cluster algorithm
 */
template <typename T, typename TPool,
          typename TContext = NibbleInnerProductKmeansContext<T>>
using NibbleInnerProductKmeans =
    LloydCluster<T, TPool, TContext, NibbleVectorArray<T>>;

}  // namespace ailego
}  // namespace zvec
