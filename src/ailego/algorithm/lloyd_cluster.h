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
#include <random>
#include <ailego/parallel/lock.h>
#include <zvec/ailego/parallel/thread_pool.h>
#include <zvec/ailego/utility/type_helper.h>

namespace zvec {
namespace ailego {

/*! Random Centroids Generator
 */
template <typename T, typename TPool>
struct RandomCentroidsGenerator {
  //! Type of values
  using OwnerType = typename std::decay<T>::type;
  using ContainerType = typename OwnerType::ContainerType;
  using ContextType = typename OwnerType::ContextType;
  using ThreadPoolType = TPool;

  //! constexpr variables
  constexpr static size_t BatchCount = OwnerType::BatchCount;

  //! Generate centroids
  void operator()(OwnerType *owner, ThreadPoolType &) const {
    const auto &matrix = owner->feature_matrix();
    const auto &cache = owner->feature_cache();
    auto *centroids = owner->mutable_centroids();

    ContainerType rows(cache.dimension());
    size_t m = matrix.count();
    size_t n = m + cache.count();
    size_t k = owner->k_value();
    std::mt19937 mt((std::random_device())());

    rows.resize(BatchCount);
    centroids->reset(cache.dimension());
    centroids->reserve(k);

    for (size_t i = 0; k > 0 && i < n; ++i) {
      if (mt() % (n - i) >= k) {
        continue;
      }
      // Selected a feature
      if (i < m) {
        ContextType::MatrixReverseTranspose(matrix[i / BatchCount * BatchCount],
                                            matrix.dimension(), rows.data());
        centroids->append(rows[i & (BatchCount - 1u)], matrix.dimension());
      } else {
        centroids->append(cache[i - m], cache.dimension());
      }
      --k;
    }
  }
};

/*! Lloyd's algorithm cluster
 */
template <typename T, typename TPool, typename TContext, typename TContainer>
class LloydCluster {
 public:
  //! constexpr variables
  constexpr static size_t BatchCount = TContext::BatchCount;

  //! Type of values
  using ThreadPoolType = TPool;
  using ContainerType = TContainer;
  using ContextType = TContext;
  using ValueType = typename TContext::ValueType;
  using StoreType = typename TContext::StoreType;

  //! Constructor
  LloydCluster(size_t k, size_t dim)
      : k_value_(k),
        feature_cache_(dim),
        feature_matrix_(dim),
        centroids_matrix_(dim),
        centroids_(dim) {}

  //! Constructor
  LloydCluster(size_t k, size_t dim, bool spherical)
      : k_value_(k),
        feature_cache_(dim),
        feature_matrix_(dim),
        centroids_matrix_(dim),
        centroids_(dim),
        spherical_{spherical} {}

  //! Constructor
  LloydCluster(void) {}

  //! Destructor
  ~LloydCluster(void) {}

  //! Append a feature
  void append(const StoreType *arr, size_t dim) {
    feature_cache_.append(arr, dim);

    if (feature_cache_.count() == BatchCount) {
      size_t pos = feature_matrix_.count();
      feature_matrix_.resize(pos + BatchCount);
      ContextType::MatrixTranspose(feature_cache_.data(), dim,
                                   feature_matrix_[pos]);
      feature_cache_.clear();
    }
  }

  //! Reset cluster
  void reset(size_t k, size_t dim) {
    k_value_ = k;
    feature_cache_.reset(dim);
    feature_matrix_.reset(dim);
    centroids_.reset(dim);
    centroids_matrix_.reset(dim);
    context_.clear();
  }

  //! Reset cluster
  void reset(size_t k, size_t dim, bool spherical) {
    k_value_ = k;
    feature_cache_.reset(dim);
    feature_matrix_.reset(dim);
    centroids_.reset(dim);
    centroids_matrix_.reset(dim);
    context_.clear();
    spherical_ = spherical;
  }

  //! Initialize centroids
  template <typename G = RandomCentroidsGenerator<LloydCluster, ThreadPoolType>>
  void init_centroids(ThreadPoolType &pool, const G &g = G()) {
    g(this, pool);
  }

  //! Cluster one time
  template <typename ThreadPoolType>
  bool cluster_once(ThreadPoolType &pool, double *cost) {
    if (centroids_.empty()) {
      RandomCentroidsGenerator<LloydCluster, ThreadPoolType> g;
      this->init_centroids(pool, g);
    }
    if (centroids_.count() != k_value_) {
      return false;
    }
    context_.reset(centroids_.count(), centroids_.dimension());

    size_t count = centroids_.count() / BatchCount * BatchCount;
    centroids_matrix_.resize(count);
    for (size_t i = 0; i != count; i += BatchCount) {
      ContextType::MatrixTranspose(centroids_[i], centroids_.dimension(),
                                   centroids_matrix_[i]);
    }
    size_t remain = static_cast<uint32_t>(centroids_.count() - count);
    if (remain > 0) {
      centroids_matrix_.append(centroids_[count], centroids_.dimension(),
                               remain);
    }

    // Using thread pool
    auto group = pool.make_group();
    if (!feature_matrix_.empty()) {
      size_t n = feature_matrix_.count() / BatchCount;
      size_t c = std::max<size_t>(n / pool.count() / 2u, 1u);
      size_t m = n / c * c;

      for (size_t i = 0; i != m; i += c) {
        group->submit(Closure::New(this, &LloydCluster::cluster_matrix_features,
                                   i, i + c));
      }
      for (size_t i = m; i != n; i += 1) {
        group->submit(Closure::New(this, &LloydCluster::cluster_matrix_features,
                                   i, i + 1));
      }
    }
    if (!feature_cache_.empty()) {
      group->submit(Closure::New(this, &LloydCluster::cluster_cache_features));
    }
    group->wait_finish();

    *cost = 0.0;
    for (size_t i = 0, n = centroids_.count(); i != n; ++i) {
      const auto &item = context_[i];
      item.centroid(centroids_[i], centroids_.dimension());
      *cost += item.cost();
    }

    if (spherical_) {
      for (size_t i = 0, n = centroids_.count(); i != n; ++i) {
        float norm;
        ContextType::Norm2(centroids_[i], centroids_.dimension(), &norm);
      }
    }

    return true;
  }

  //! Retrieve the controids
  ContainerType *mutable_centroids(void) {
    return &centroids_;
  }

  //! Retrieve the controids
  const ContainerType &centroids(void) const {
    return centroids_;
  }

  //! Retrieve the K value
  size_t k_value(void) const {
    return k_value_;
  }

  //! Retrieve context
  const ContextType &context(void) const {
    return context_;
  }

  //! Retrieve the feature cache
  const ContainerType &feature_cache(void) const {
    return feature_cache_;
  }

  //! Retrieve the feature matrix
  const ContainerType &feature_matrix(void) const {
    return feature_matrix_;
  }

  //! Reserve the feature matrix
  void feature_matrix_reserve(size_t count) {
    feature_matrix_.reserve(count);
  }

 protected:
  //! Cluster the cache features
  void cluster_cache_features(void) {
    float scores[BatchCount];

    for (size_t i = 0, n = feature_cache_.count(); i != n; ++i) {
      size_t count = centroids_matrix_.count() / BatchCount * BatchCount;
      const StoreType *feature = feature_cache_[i];
      float nearest_score = std::numeric_limits<float>::max();
      size_t nearest_index = 0;

      for (size_t j = 0; j != count; j += BatchCount) {
        ContextType::template BatchDistance<1>(centroids_matrix_[j], feature,
                                               centroids_matrix_.dimension(),
                                               scores);

        for (size_t k = 0; k < BatchCount; ++k) {
          if (scores[k] < nearest_score) {
            nearest_score = scores[k];
            nearest_index = j + k;
          }
        }
      }  // end of for

      for (size_t j = count, total = centroids_matrix_.count(); j != total;
           ++j) {
        ContextType::Distance(centroids_matrix_[j], feature,
                              centroids_matrix_.dimension(), scores);

        if (scores[0] < nearest_score) {
          nearest_score = scores[0];
          nearest_index = j;
        }
      }
      context_[nearest_index].append(feature, feature_cache_.dimension(),
                                     nearest_score);
    }  // end of for
  }

  //! Cluster the matrix features
  void cluster_matrix_features(size_t first, size_t last) {
    std::vector<float> scores(BatchCount * BatchCount, 0);
    ContainerType rows(centroids_matrix_.dimension());

    auto comp = [](float i, float j) {
      if (std::isnan(i)) return false;
      if (std::isnan(j)) return true;

      return i < j;
    };

    float nearest_scores[BatchCount];
    size_t nearest_indexes[BatchCount];

    rows.resize(BatchCount);
    for (size_t i = first * BatchCount; i != last * BatchCount;
         i += BatchCount) {
      size_t count = centroids_matrix_.count() / BatchCount * BatchCount;
      const StoreType *block = feature_matrix_[i];

      std::fill(nearest_indexes, nearest_indexes + BatchCount, 0);
      std::fill(nearest_scores, nearest_scores + BatchCount,
                std::numeric_limits<float>::max());

      for (size_t j = 0; j != count; j += BatchCount) {
        ContextType::template BatchDistance<BatchCount>(
            centroids_matrix_[j], block, centroids_matrix_.dimension(),
            &scores[0]);

        for (size_t k = 0; k < BatchCount; ++k) {
          const float *start = &scores[k * BatchCount];
          const float *result =
              std::min_element(start, start + BatchCount, comp);
          if (*result < nearest_scores[k]) {
            nearest_scores[k] = *result;
            nearest_indexes[k] = j + (result - start);
          }
        }
      }  // end of for

      for (size_t j = count, total = centroids_matrix_.count(); j != total;
           ++j) {
        ContextType::template BatchDistance<1>(block, centroids_matrix_[j],
                                               centroids_matrix_.dimension(),
                                               &scores[0]);

        for (size_t k = 0; k < BatchCount; ++k) {
          float score = scores[k];
          if (score < nearest_scores[k]) {
            nearest_scores[k] = score;
            nearest_indexes[k] = j;
          }
        }
      }  // end of for

      ContextType::MatrixReverseTranspose(block, feature_matrix_.dimension(),
                                          rows.data());
      for (size_t k = 0; k < BatchCount; ++k) {
        context_[nearest_indexes[k]].append(
            rows[k], feature_matrix_.dimension(), nearest_scores[k]);
      }
    }  // end of for
  }

 private:
  //! Members
  size_t k_value_{0u};
  ContainerType feature_cache_{};
  ContainerType feature_matrix_{};
  ContainerType centroids_matrix_{};
  ContainerType centroids_{};
  ContextType context_{};
  bool spherical_{false};
};

}  // namespace ailego
}  // namespace zvec