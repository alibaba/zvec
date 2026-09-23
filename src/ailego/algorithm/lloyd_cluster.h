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
#include <array>
#include <atomic>
#include <memory>
#include <random>
#include <ailego/parallel/lock.h>
#include <zvec/ailego/parallel/thread_pool.h>
#include <zvec/ailego/utility/type_helper.h>

namespace zvec {
namespace ailego {

// Optional backing for the transposed training matrix. The clustering kernel
// owns only batch-sized scratch; implementations preserve their own error
// codes.
class LloydClusterMatrixStorage {
 public:
  virtual ~LloydClusterMatrixStorage() = default;
  virtual int read(size_t offset, void *out, size_t bytes) const = 0;
  virtual int write(size_t offset, const void *data, size_t bytes) = 0;
};

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
    const auto &cache = owner->feature_cache();
    auto *centroids = owner->mutable_centroids();

    ContainerType rows(cache.dimension());
    ContainerType scratch(cache.dimension());
    size_t m = owner->feature_matrix_count();
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
        const auto *block =
            owner->read_feature_matrix(i / BatchCount * BatchCount, &scratch);
        if (!block) return;
        ContextType::MatrixReverseTranspose(block, cache.dimension(),
                                            rows.data());
        centroids->append(rows[i & (BatchCount - 1u)], cache.dimension());
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
  LloydCluster() = default;

  //! Destructor
  ~LloydCluster() = default;

  //! Append a feature
  void append(const StoreType *arr, size_t dim) {
    if (matrix_status() != 0) return;
    feature_cache_.append(arr, dim);

    if (feature_cache_.count() == BatchCount) {
      if (matrix_storage_) {
        matrix_write_buffer_.resize(BatchCount);
        ContextType::MatrixTranspose(feature_cache_.data(), dim,
                                     matrix_write_buffer_.data());
        const size_t bytes = matrix_write_buffer_.bytes();
        const int ret =
            matrix_storage_->write(stored_matrix_count_ / BatchCount * bytes,
                                   matrix_write_buffer_.data(), bytes);
        if (ret != 0) {
          record_matrix_error(ret);
          return;
        }
        stored_matrix_count_ += BatchCount;
      } else {
        size_t pos = feature_matrix_.count();
        feature_matrix_.resize(pos + BatchCount);
        ContextType::MatrixTranspose(feature_cache_.data(), dim,
                                     feature_matrix_[pos]);
      }
      feature_cache_.clear();
    }
  }

  //! Reset cluster
  void reset(size_t k, size_t dim) {
    k_value_ = k;
    feature_cache_.reset(dim);
    feature_matrix_.reset(dim);
    reset_matrix_storage(dim);
    centroids_.reset(dim);
    centroids_matrix_.reset(dim);
    context_.clear();
  }

  //! Reset cluster
  void reset(size_t k, size_t dim, bool spherical) {
    k_value_ = k;
    feature_cache_.reset(dim);
    feature_matrix_.reset(dim);
    reset_matrix_storage(dim);
    centroids_.reset(dim);
    centroids_matrix_.reset(dim);
    context_.clear();
    spherical_ = spherical;
  }

  //! Initialize centroids
  template <typename G = RandomCentroidsGenerator<LloydCluster, ThreadPoolType>>
  void init_centroids(ThreadPoolType &pool, const G &g = G()) {
    if (matrix_status() != 0) return;
    g(this, pool);
  }

  //! Cluster one time
  template <typename ThreadPoolType>
  bool cluster_once(ThreadPoolType &pool, double *cost) {
    if (matrix_status() != 0) return false;
    if (centroids_.empty()) {
      RandomCentroidsGenerator<LloydCluster, ThreadPoolType> g;
      this->init_centroids(pool, g);
    }
    if (matrix_status() != 0 || centroids_.count() != k_value_) {
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
    if (feature_matrix_count() != 0) {
      size_t n = feature_matrix_count() / BatchCount;
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
    if (matrix_status() != 0) return false;

    *cost = 0.0;
    for (size_t i = 0, n = centroids_.count(); i != n; ++i) {
      const auto &item = context_[i];
      item.centroid(centroids_[i], centroids_.dimension());
      *cost += item.cost();
    }

    if (spherical_) {
      for (size_t i = 0, n = centroids_.count(); i != n; ++i) {
        float norm = 0.0f;
        ContextType::Norm2(centroids_[i], centroids_.dimension(), &norm);
      }
    }

    return true;
  }

  //! Retrieve the controids
  ContainerType *mutable_centroids() {
    return &centroids_;
  }

  //! Retrieve the controids
  const ContainerType &centroids() const {
    return centroids_;
  }

  //! Retrieve the K value
  size_t k_value() const {
    return k_value_;
  }

  //! Retrieve spherical option
  bool spherical() const {
    return spherical_;
  }

  //! Retrieve context
  const ContextType &context() const {
    return context_;
  }

  //! Retrieve the feature cache
  const ContainerType &feature_cache() const {
    return feature_cache_;
  }

  //! Retrieve the feature matrix
  const ContainerType &feature_matrix() const {
    return feature_matrix_;
  }

  //! Reserve the feature matrix
  void feature_matrix_reserve(size_t count) {
    if (!matrix_storage_) feature_matrix_.reserve(count);
  }

  // Set before appending any rows. The in-memory path remains the default.
  void set_feature_matrix_storage(
      std::shared_ptr<LloydClusterMatrixStorage> storage) {
    ailego_assert_with(feature_matrix_.empty() && feature_cache_.empty(),
                       "Cannot replace a populated training matrix");
    reset_matrix_storage(feature_cache_.dimension());
    matrix_storage_ = std::move(storage);
  }

  size_t feature_matrix_count() const {
    return matrix_storage_ ? stored_matrix_count_ : feature_matrix_.count();
  }

  int matrix_status() const {
    return matrix_storage_ ? matrix_error_.load(std::memory_order_relaxed) : 0;
  }

  // A returned external block belongs to the caller's scratch and survives
  // concurrent reads/eviction. No pointer to reclaimable storage escapes.
  const StoreType *read_feature_matrix(size_t index,
                                       ContainerType *scratch) const {
    if (!matrix_storage_) return feature_matrix_[index];
    if (matrix_status() != 0) return nullptr;
    scratch->resize(BatchCount);
    const size_t bytes = scratch->bytes();
    const int ret = matrix_storage_->read(index / BatchCount * bytes,
                                          scratch->data(), bytes);
    if (ret != 0) {
      record_matrix_error(ret);
      return nullptr;
    }
    return scratch->data();
  }

 protected:
  //! Cluster the cache features
  void cluster_cache_features() {
    std::array<float, BatchCount> scores;

    for (size_t i = 0, n = feature_cache_.count(); i != n; ++i) {
      size_t count = centroids_matrix_.count() / BatchCount * BatchCount;
      const StoreType *feature = feature_cache_[i];
      float nearest_score = std::numeric_limits<float>::max();
      size_t nearest_index = 0;

      for (size_t j = 0; j != count; j += BatchCount) {
        ContextType::template BatchDistance<1>(centroids_matrix_[j], feature,
                                               centroids_matrix_.dimension(),
                                               scores.data());

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
                              centroids_matrix_.dimension(), scores.data());

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
    std::array<float, BatchCount * BatchCount> scores;
    ContainerType rows(centroids_matrix_.dimension());
    ContainerType scratch(centroids_matrix_.dimension());

    auto comp = [](float i, float j) {
      if (std::isnan(i)) return false;
      if (std::isnan(j)) return true;

      return i < j;
    };

    std::array<float, BatchCount> nearest_scores;
    std::array<size_t, BatchCount> nearest_indexes;

    rows.resize(BatchCount);
    for (size_t i = first * BatchCount; i != last * BatchCount;
         i += BatchCount) {
      size_t count = centroids_matrix_.count() / BatchCount * BatchCount;
      const StoreType *block = read_feature_matrix(i, &scratch);
      if (!block) return;

      std::fill(nearest_indexes.data(), nearest_indexes.data() + BatchCount, 0);
      std::fill(nearest_scores.data(), nearest_scores.data() + BatchCount,
                std::numeric_limits<float>::max());

      for (size_t j = 0; j != count; j += BatchCount) {
        ContextType::template BatchDistance<BatchCount>(
            centroids_matrix_[j], block, centroids_matrix_.dimension(),
            scores.data());

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
                                               scores.data());

        for (size_t k = 0; k < BatchCount; ++k) {
          float score = scores[k];
          if (score < nearest_scores[k]) {
            nearest_scores[k] = score;
            nearest_indexes[k] = j;
          }
        }
      }  // end of for

      ContextType::MatrixReverseTranspose(block, feature_cache_.dimension(),
                                          rows.data());
      for (size_t k = 0; k < BatchCount; ++k) {
        context_[nearest_indexes[k]].append(rows[k], feature_cache_.dimension(),
                                            nearest_scores[k]);
      }
    }  // end of for
  }

 private:
  void record_matrix_error(int error) const {
    int expected = 0;
    matrix_error_.compare_exchange_strong(expected, error,
                                          std::memory_order_relaxed);
  }

  void reset_matrix_storage(size_t dim) {
    matrix_storage_.reset();
    stored_matrix_count_ = 0;
    matrix_error_.store(0, std::memory_order_relaxed);
    matrix_write_buffer_.reset(dim);
  }

  //! Members
  size_t k_value_{0u};
  ContainerType feature_cache_{};
  ContainerType feature_matrix_{};
  ContainerType matrix_write_buffer_{};
  std::shared_ptr<LloydClusterMatrixStorage> matrix_storage_{};
  size_t stored_matrix_count_{0};
  mutable std::atomic<int> matrix_error_{0};
  ContainerType centroids_matrix_{};
  ContainerType centroids_{};
  ContextType context_{};
  bool spherical_{false};
};

}  // namespace ailego
}  // namespace zvec
