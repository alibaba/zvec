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

#include <ailego/algorithm/kmeans.h>
#include <ailego/parallel/multi_thread_list.h>
#include <zvec/ailego/internal/platform.h>
#include <zvec/core/framework/index_cluster.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include "cluster_params.h"

namespace zvec {
namespace core {

//! MultiChunkClusterAlgorithm
class MultiChunkClusterAlgorithm {
 public:
  typedef std::shared_ptr<MultiChunkClusterAlgorithm> Pointer;

  //! Constructor
  MultiChunkClusterAlgorithm() = default;

  //! Destructor
  virtual ~MultiChunkClusterAlgorithm() = default;

  //! Initialize Cluster
  int init(const IndexMeta &meta, const ailego::Params &params);

  //! Cleanup Cluster
  int cleanup();

  //! Reset Cluster
  int reset();

  //! Update Cluster
  int update(const ailego::Params &params);

  //! Suggest dividing to K clusters
  void suggest(uint32_t k);

  //! Mount features
  int mount(IndexFeatures::Pointer feats);

  //! Cluster
  int cluster(IndexThreads::Pointer threads, IndexCluster::CentroidList &cents);

  //! Classify
  int classify(IndexThreads::Pointer threads,
               IndexCluster::CentroidList &cents);

  //! Label
  int label(IndexThreads::Pointer threads,
            const IndexCluster::CentroidList &cents,
            std::vector<uint32_t> *out);


  const std::vector<uint32_t> &chunk_dims() const {
    return chunk_dims_;
  }

  const std::vector<uint32_t> &chunk_dim_offsets() const {
    return chunk_dim_offsets_;
  }

 protected:
  //! Check Centroids
  bool check_centroids(const IndexCluster::CentroidList &cents);

  //! Test if it is valid
  bool is_valid() const;

  //! Do chunk
  int do_chunk();

  //! Update parameters
  int update_params(const ailego::Params &params);

  int init_distance_func();

  //! cluster thread
  virtual void do_cluster(size_t idx, size_t chunk_step,
                          IndexCluster::CentroidList *cents,
                          std::atomic<size_t> *finished) = 0;

  //! label thread
  virtual void do_label(size_t idx, size_t step,
                        const IndexCluster::CentroidList &cents,
                        std::vector<uint32_t> *out,
                        std::atomic<size_t> *finished) = 0;

  //! Initialize Centroids
  void init_centroids(size_t count, IndexCluster::CentroidList *out);

 private:
  constexpr static uint32_t kDefaultLogIntervalSecs = 15U;

 protected:
  uint32_t cluster_count_{0u};
  uint32_t thread_count_{0u};
  uint32_t chunk_count_{0u};
  uint32_t max_iterations_{20u};
  bool assumption_free_{false};
  uint32_t markov_chain_length_{32};
  double epsilon_{std::numeric_limits<float>::epsilon()};

  int errcode_{0};
  std::atomic_bool error_{false};
  uint32_t check_interval_secs_{kDefaultLogIntervalSecs};
  std::mutex mutex_{};
  std::condition_variable cond_{};

  IndexMeta meta_{};
  IndexFeatures::Pointer features_{};

  std::vector<uint32_t> chunk_dims_;
  std::vector<uint32_t> chunk_dim_offsets_;

  IndexMetric::MatrixDistance distance_func_{nullptr};
};

/*! Numerical cluster algorithm
 */
template <typename T>
class MultiChunkNumericalAlgorithm : public MultiChunkClusterAlgorithm {
 public:
  //! Type of value
  using ValueType = typename std::remove_cv<T>::type;

  // Check supporting type
  static_assert(ailego::IsArithmetic<ValueType>::value,
                "ValueType must be arithmetic");

  //! Constructor
  MultiChunkNumericalAlgorithm() = default;

  //! Destructor
  ~MultiChunkNumericalAlgorithm() = default;

 protected:
  //! cluster thread
  void do_cluster(size_t idx, size_t chunk_step,
                  IndexCluster::CentroidList *cents,
                  std::atomic<size_t> *finished);

  //! label thread
  void do_label(size_t idx, size_t step,
                const IndexCluster::CentroidList &cents,
                std::vector<uint32_t> *out, std::atomic<size_t> *finished);
};

//! cluster thread
template <typename T>
void MultiChunkNumericalAlgorithm<T>::do_cluster(
    size_t idx, size_t chunk_step, IndexCluster::CentroidList *cents,
    std::atomic<size_t> *finished) {
  for (size_t chunk = idx; chunk < chunk_count_; chunk += chunk_step) {
    auto chunk_dim = chunk_dims_[chunk];

    ailego::NumericalKmeans<T, IndexThreads> algorithm(cluster_count_,
                                                       chunk_dim);

    // mount features into algorithm
    auto features_count = features_->count();

    algorithm.feature_matrix_reserve(features_count);

    for (size_t i = 0; i < features_count; ++i) {
      auto vec = reinterpret_cast<const T *>(features_->element(i));
      algorithm.append(vec + chunk_dim_offsets_[chunk], chunk_dim);
    }

    IndexThreads::Pointer local_threads =
        std::make_shared<SingleQueueIndexThreads>(1, false);
    if (!local_threads) {
      error_ = IndexError_NoMemory;
      return;
    }

    ailego::Kmc2CentroidsGenerator<ailego::NumericalKmeans<T, IndexThreads>,
                                   IndexThreads>
        cent_gen;
    cent_gen.set_chain_length(markov_chain_length_);
    cent_gen.set_assumption_free(assumption_free_);
    cent_gen(&algorithm, *local_threads);

    double cost = 0.0;

    for (uint32_t i = 0; i < max_iterations_; ++i) {
      double old_cost, new_epsilon;
      old_cost = cost;

      bool result = algorithm.cluster_once(*local_threads, &cost);
      if (result != true) {
        LOG_ERROR("(%u) Failed to cluster.", i + 1);
        errcode_ = -1;

        return;
      }

      new_epsilon = std::abs(cost - old_cost);
      if (new_epsilon < epsilon_) {
        break;
      }
    }

    auto &chunk_cents = algorithm.centroids();

    for (size_t i = 0; i < chunk_cents.count(); ++i) {
      size_t global_cent_idx = chunk * cluster_count_ + i;

      IndexCluster::Centroid *centroid = &(cents->at(global_cent_idx));
      centroid->set_score(algorithm.context().clusters()[i].cost());
      centroid->set_follows(algorithm.context().clusters()[i].count());
      centroid->set_feature(algorithm.centroids()[i],
                            chunk_dim * meta_.unit_size());
    }

    LOG_INFO("(%zu) Chunk Done. Clusters Count: %zu, Features: %zu, Cost: %f",
             chunk, algorithm.centroids().count(), features_->count(), cost);

    (*finished)++;
  }

  return;
}

//! label thread
template <typename T>
void MultiChunkNumericalAlgorithm<T>::do_label(
    size_t idx, size_t step, const IndexCluster::CentroidList &cents,
    std::vector<uint32_t> *out, std::atomic<size_t> *finished) {
  for (size_t id = idx; id < features_->count(); id += step) {
    const T *feat = reinterpret_cast<const T *>(features_->element(id));

    for (size_t chunk = 0; chunk < chunk_count_; ++chunk) {
      size_t chunk_dim_offset = chunk_dim_offsets_[chunk];
      size_t chunk_dim = chunk_dims_[chunk];

      uint32_t sel_index = 0;
      float sel_score = std::numeric_limits<float>::max();

      for (uint32_t cluster = 0; cluster < cluster_count_; ++cluster) {
        float score{0.0};

        distance_func_(cents[chunk * cluster_count_ + cluster].feature(),
                       feat + chunk_dim_offset, chunk_dim, &score);

        if (score < sel_score) {
          sel_score = score;
          sel_index = cluster;
        }
      }

      (*out)[id * chunk_count_ + chunk] = static_cast<uint32_t>(sel_index);
    }

    (*finished)++;
  }
}

/*! Inner Product Cluster Algorithm
 */
template <typename T>
class MultiChunkNumericalInnerProductAlgorithm
    : public MultiChunkClusterAlgorithm {
 public:
  //! Type of value
  using ValueType = typename std::remove_cv<T>::type;

  // Check supporting type
  static_assert(ailego::IsArithmetic<ValueType>::value,
                "ValueType must be arithmetic");

  //! Constructor
  MultiChunkNumericalInnerProductAlgorithm() = default;

  //! Destructor
  ~MultiChunkNumericalInnerProductAlgorithm() = default;

 protected:
  //! cluster thread
  void do_cluster(size_t idx, size_t chunk_step,
                  IndexCluster::CentroidList *cents,
                  std::atomic<size_t> *finished);

  //! label thread
  void do_label(size_t idx, size_t chunk_step,
                const IndexCluster::CentroidList &cents,
                std::vector<uint32_t> *out, std::atomic<size_t> *finished);
};

//! cluster thread
template <typename T>
void MultiChunkNumericalInnerProductAlgorithm<T>::do_cluster(
    size_t idx, size_t chunk_step, IndexCluster::CentroidList *cents,
    std::atomic<size_t> *finished) {
  for (size_t chunk = idx; chunk < chunk_count_; chunk += chunk_step) {
    auto chunk_dim = chunk_dims_[chunk];

    ailego::NumericalInnerProductKmeans<T, IndexThreads> algorithm(
        cluster_count_, chunk_dim);

    // mount features into algorithm
    auto features_count = features_->count();

    algorithm.feature_matrix_reserve(features_count);

    for (size_t i = 0; i < features_count; ++i) {
      auto vec = reinterpret_cast<const T *>(features_->element(i));
      algorithm.append(vec + chunk_dim_offsets_[chunk], chunk_dim);
    }

    IndexThreads::Pointer local_threads =
        std::make_shared<SingleQueueIndexThreads>(1, false);
    if (!local_threads) {
      error_ = IndexError_NoMemory;
      return;
    }

    ailego::Kmc2CentroidsGenerator<
        ailego::NumericalInnerProductKmeans<T, IndexThreads>, IndexThreads>
        cent_gen;
    cent_gen.set_chain_length(markov_chain_length_);
    cent_gen.set_assumption_free(assumption_free_);
    cent_gen(&algorithm, *local_threads);

    double cost = 0.0;

    for (uint32_t i = 0; i < max_iterations_; ++i) {
      double old_cost, new_epsilon;
      old_cost = cost;

      bool result = algorithm.cluster_once(*local_threads, &cost);
      if (result != true) {
        LOG_ERROR("(%zu) Failed to cluster.", (size_t)(i + 1));
        errcode_ = -1;

        return;
      }

      new_epsilon = std::abs(cost - old_cost);
      if (new_epsilon < epsilon_) {
        break;
      }
    }

    auto &chunk_cents = algorithm.centroids();

    for (size_t i = 0; i < chunk_cents.count(); ++i) {
      size_t global_cent_idx = chunk * cluster_count_ + i;

      IndexCluster::Centroid *centroid = &(cents->at(global_cent_idx));
      centroid->set_score(algorithm.context().clusters()[i].cost());
      centroid->set_follows(algorithm.context().clusters()[i].count());
      centroid->set_feature(algorithm.centroids()[i],
                            chunk_dim * meta_.unit_size());
    }

    LOG_INFO("(%zu) Chunk Done. Clusters Count: %zu, Features: %zu, Cost: %f",
             chunk, algorithm.centroids().count(), features_->count(), cost);

    (*finished)++;
  }
}

//! label thread
template <typename T>
void MultiChunkNumericalInnerProductAlgorithm<T>::do_label(
    size_t idx, size_t step, const IndexCluster::CentroidList &cents,
    std::vector<uint32_t> *out, std::atomic<size_t> *finished) {
  for (size_t id = idx; id < features_->count(); id += step) {
    const T *feat = reinterpret_cast<const T *>(features_->element(id));

    for (size_t chunk = 0; chunk < chunk_count_; ++chunk) {
      size_t chunk_dim_offset = chunk_dim_offsets_[chunk];
      size_t chunk_dim = chunk_dims_[chunk];

      uint32_t sel_index = 0;
      float sel_score = std::numeric_limits<float>::max();

      for (uint32_t cluster = 0; cluster < cluster_count_; ++cluster) {
        float score{0.0};

        distance_func_(cents[chunk * cluster_count_ + cluster].feature(),
                       feat + chunk_dim_offset, chunk_dim, &score);

        if (score < sel_score) {
          sel_score = score;
          sel_index = cluster;
        }
      }

      (*out)[id * chunk_count_ + chunk] = sel_index;
    }

    (*finished)++;
  }
}

//! MultiChunkCluster
class MultiChunkCluster {
 public:
  std::shared_ptr<MultiChunkCluster> Pointer;

  //! Constructor
  MultiChunkCluster() = default;

  //! Destructor
  ~MultiChunkCluster() = default;

  //! Initialize Cluster
  int init(const IndexMeta &meta, const ailego::Params &params);

  //! Cleanup Cluster
  int cleanup();

  //! Reset Cluster
  int reset();

  //! Update Cluster
  int update(const ailego::Params &params);

  //! Suggest dividing to K clusters
  void suggest(uint32_t k);

  //! Mount features
  int mount(IndexFeatures::Pointer feats);

  //! Cluster
  int cluster(IndexThreads::Pointer threads, IndexCluster::CentroidList &cents);

  //! Classify
  int classify(IndexThreads::Pointer threads,
               IndexCluster::CentroidList &cents);

  //! Label
  int label(IndexThreads::Pointer threads,
            const IndexCluster::CentroidList &cents,
            std::vector<uint32_t> *out);

  const std::vector<uint32_t> &chunk_dims() const {
    return algorithm_->chunk_dims();
  }

  const std::vector<uint32_t> &chunk_dim_offsets() const {
    return algorithm_->chunk_dim_offsets();
  }

 protected:
  MultiChunkClusterAlgorithm::Pointer algorithm_{};
};

}  // namespace core
}  // namespace zvec
