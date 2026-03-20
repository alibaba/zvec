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
#include <ailego/algorithm/kmeans.h>
#include <ailego/container/reservoir.h>
#include <zvec/core/framework/index_cluster.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include "cluster_params.h"

namespace zvec {
namespace core {

/*! Optimize K-Means cluster algorithm
 */
class OptKmeansAlgorithm : public IndexCluster {
 public:
  //! Constructor
  OptKmeansAlgorithm(void) {}

  //! Destructor
  virtual ~OptKmeansAlgorithm(void) {}

  //! Initialize Cluster
  int init(const IndexMeta &meta, const ailego::Params &params);

  //! Mount features
  virtual int mount(IndexFeatures::Pointer feats);

  //! Suggest dividing to K clusters
  virtual void suggest(uint32_t k);

  //! Classify
  virtual int classify(IndexThreads::Pointer threads,
                       IndexCluster::CentroidList &cents);

  //! Label
  virtual int label(IndexThreads::Pointer threads,
                    const IndexCluster::CentroidList &cents,
                    std::vector<uint32_t> *out);

  //! Cluster
  virtual int cluster(IndexThreads::Pointer threads,
                      IndexCluster::CentroidList &cents) = 0;

  //! Cleanup Cluster
  virtual int cleanup(void);

  //! Reset Cluster
  virtual int reset(void);

  //! Update Cluster
  virtual int update(const ailego::Params &params);

 protected:
  //! Update parameters
  void update_params(const ailego::Params &params);

  //! Init Kmeans Algorithm
  int init_algorithm();

  //! Init Distance function
  int init_distance_func();

  //! Check Centroids
  bool check_centroids(const IndexCluster::CentroidList &cents);

  //! Test if it is valid
  bool is_valid(void) const;

  //! Update Clusters
  void update_clusters(IndexThreads *threads,
                       const IndexCluster::CentroidList &cents);

  //! Update Cluster in Thread
  void update_cluster_thread(size_t index_begin, size_t index_end,
                             const IndexThreads *threads,
                             const IndexCluster::CentroidList &cents);

  //! Initialize Shard Features Containers
  void init_features_containers(size_t shard_count);

  //! Update Clusters' Features
  void update_features(IndexThreads *threads,
                       IndexCluster::CentroidList &cents);

  //! Update Cluster's Features in Thread
  void update_features_thread(size_t column, IndexCluster::CentroidList *out);

  //! Update Labels
  void update_labels(IndexThreads *threads, std::vector<uint32_t> *labels,
                     const IndexCluster::CentroidList &cents);

  //! Update Labels in Thread
  void update_labels_thread(size_t index_begin, size_t index_end,
                            std::vector<uint32_t> *labels,
                            const IndexCluster::CentroidList &cents);

  //! Initialize Centroids
  void init_centroids(size_t count, IndexCluster::CentroidList *out);

 protected:
  uint32_t cluster_count_{0u};
  uint32_t thread_count_{0u};
  uint32_t max_iterations_{20u};
  double epsilon_{std::numeric_limits<float>::epsilon()};
  float shard_factor_{16.0f};
  bool purge_empty_{false};
  bool assumption_free_{false};
  uint32_t markov_chain_length_{32};
  IndexMeta meta_{};
  IndexFeatures::Pointer features_{};
  std::vector<std::vector<const void *>> shard_cluster_features_{};
  IndexMetric::MatrixDistance distance_func_{nullptr};
};

bool OptKmeansAlgorithm::is_valid(void) const {
  if (!features_ || !features_->count()) {
    return false;
  }
  return true;
}

bool OptKmeansAlgorithm::check_centroids(
    const IndexCluster::CentroidList &cents) {
  for (const auto &it : cents) {
    if (it.size() != meta_.element_size()) {
      return false;
    }
  }
  return true;
}

void OptKmeansAlgorithm::update_params(const ailego::Params &params) {
  params.get(GENERAL_THREAD_COUNT, &thread_count_);
  params.get(GENERAL_CLUSTER_COUNT, &cluster_count_);
  params.get(OPTKMEANS_CLUSTER_COUNT, &cluster_count_);
  params.get(OPTKMEANS_CLUSTER_SHARD_FACTOR, &shard_factor_);
  params.get(OPTKMEANS_CLUSTER_EPSILON, &epsilon_);
  params.get(OPTKMEANS_CLUSTER_MAX_ITERATIONS, &max_iterations_);
  params.get(OPTKMEANS_CLUSTER_PURGE_EMPTY, &purge_empty_);
  params.get(OPTKMEANS_CLUSTER_MARKOV_CHAIN_LENGTH, &markov_chain_length_);
  params.get(OPTKMEANS_CLUSTER_ASSUMPTION_FREE, &assumption_free_);
}

int OptKmeansAlgorithm::init_distance_func() {
  IndexMetric::Pointer metric_{};
  metric_ = IndexFactory::CreateMetric(meta_.metric_name());
  if (!metric_) {
    LOG_ERROR("Create Metric %s failed.", meta_.metric_name().c_str());
    return IndexError_Unsupported;
  }
  int ret = metric_->init(meta_, meta_.metric_params());
  if (ret != 0) {
    LOG_ERROR("IndexMetric init failed wit ret %d.", ret);
    return ret;
  }
  distance_func_ = metric_->distance_matrix(1, 1);
  if (!distance_func_) {
    LOG_ERROR("DistanceMatrix function is nullptr.");
    return IndexError_Unsupported;
  }
  return 0;
}

void OptKmeansAlgorithm::update_clusters(
    IndexThreads *threads, const IndexCluster::CentroidList &cents) {
  // Initilize containers
  this->init_features_containers(threads->count() * cents.size());
  auto task_group = threads->make_group();

  size_t features_count = features_->count();
  size_t shard_count = std::max<size_t>(
      static_cast<size_t>(std::ceil(threads->count() * shard_factor_)), 1u);
  size_t fregment_count = (features_count + shard_count - 1) / shard_count;

  for (size_t i = 0, index = 0; (i != shard_count) && (index < features_count);
       ++i) {
    size_t next_index = index + fregment_count;
    if (next_index > features_count) {
      next_index = features_count;
    }

    // Process in work thread·
    task_group->submit(
        ailego::Closure::New(this, &OptKmeansAlgorithm::update_cluster_thread,
                             index, next_index, threads, cents));

    // Next index
    index = next_index;
  }
  task_group->wait_finish();
}

void OptKmeansAlgorithm::update_cluster_thread(
    size_t index_begin, size_t index_end, const IndexThreads *threads,
    const IndexCluster::CentroidList &cents) {
  size_t thread_offset = threads->indexof_this() * cents.size();

  for (size_t i = index_begin; i != index_end; ++i) {
    const void *feat = features_->element(i);
    uint32_t sel_index = 0;
    float sel_score = std::numeric_limits<float>::max();

    // todo: get min distance
    uint32_t total = static_cast<uint32_t>(cents.size());
    for (uint32_t j = 0; j < total; ++j) {
      float score = 0.0f;

      distance_func_(cents[j].feature(), feat, meta_.dimension(), &score);
      if (score < sel_score) {
        sel_score = score;
        sel_index = j;
      }
    }

    size_t sel_column = thread_offset + sel_index;
    shard_cluster_features_[sel_column].emplace_back(feat);
  }
}

void OptKmeansAlgorithm::init_features_containers(size_t shard_count) {
  shard_cluster_features_.resize(shard_count);
  for (auto &features : shard_cluster_features_) {
    features.clear();
  }
}

void OptKmeansAlgorithm::update_features(IndexThreads *threads,
                                         IndexCluster::CentroidList &cents) {
  auto task_group = threads->make_group();
  for (size_t i = 0; i < cents.size(); ++i) {
    // Process in work thread
    task_group->submit(ailego::Closure::New(
        this, &OptKmeansAlgorithm::update_features_thread, i, &cents));
  }
  task_group->wait_finish();
}

void OptKmeansAlgorithm::update_labels(
    IndexThreads *threads, std::vector<uint32_t> *labels,
    const IndexCluster::CentroidList &cents) {
  size_t features_count = features_->count();
  size_t shard_count = std::max<size_t>(
      static_cast<size_t>(std::ceil(threads->count() * shard_factor_)), 1u);
  size_t fregment_count = (features_count + shard_count - 1) / shard_count;
  auto task_group = threads->make_group();

  // Prepare buffer
  labels->resize(features_count);

  for (size_t i = 0, index = 0; (i != shard_count) && (index < features_count);
       ++i) {
    size_t next_index = index + fregment_count;
    if (next_index > features_count) {
      next_index = features_count;
    }

    // Process in work thread
    task_group->submit(
        ailego::Closure::New(this, &OptKmeansAlgorithm::update_labels_thread,
                             index, next_index, labels, cents));

    // Next index
    index = next_index;
  }
  task_group->wait_finish();
}

void OptKmeansAlgorithm::update_labels_thread(
    size_t index_begin, size_t index_end, std::vector<uint32_t> *labels,
    const IndexCluster::CentroidList &cents) {
  for (size_t i = index_begin; i != index_end; ++i) {
    const void *feat = features_->element(i);

    uint32_t sel_index = 0;
    float sel_score = std::numeric_limits<float>::max();

    // todo: get min distance
    uint32_t total = static_cast<uint32_t>(cents.size());
    for (uint32_t j = 0; j < total; ++j) {
      float score = 0.0f;

      distance_func_(cents[j].feature(), feat, meta_.dimension(), &score);
      if (score < sel_score) {
        sel_score = score;
        sel_index = j;
      }
    }

    (*labels)[i] = static_cast<uint32_t>(sel_index);
  }
}

void OptKmeansAlgorithm::init_centroids(size_t count,
                                        IndexCluster::CentroidList *out) {
  // Just resize, because the get random centroid step is done by cluster_once
  out->resize(count);
}

void OptKmeansAlgorithm::update_features_thread(
    size_t column, IndexCluster::CentroidList *out) {
  size_t cluster_count = out->size();
  size_t cluster_follows = 0u;

  // Compute the follows of cluster
  for (size_t i = column; i < shard_cluster_features_.size();
       i += cluster_count) {
    cluster_follows += shard_cluster_features_[i].size();
  }

  // Merge all features in cluster
  std::vector<const void *> &cluster_features =
      *(out->at(column).mutable_similars());
  cluster_features.resize(cluster_follows);

  for (size_t i = column, j = 0; i < shard_cluster_features_.size();
       i += cluster_count) {
    const std::vector<const void *> &it = shard_cluster_features_[i];
    std::memcpy(&cluster_features[j], it.data(), it.size() * sizeof(void *));
    j += it.size();
  }
}

static inline void PurgeCentroids(IndexCluster::CentroidList &cents,
                                  bool cutting) {
  size_t index = 0;
  size_t tamp = cents.size();

  while (index < tamp) {
    if (cents[index].follows() == 0) {
      size_t last_index = tamp - 1;

      if (index != last_index) {
        std::swap(cents[index], cents[last_index]);
      }
      tamp = last_index;
      continue;
    }
    ++index;
  }
  if (cutting) {
    cents.resize(tamp);
  }
}

int OptKmeansAlgorithm::init(const IndexMeta &meta,
                             const ailego::Params &params) {
  meta_ = meta;
  this->update_params(params);

  return init_distance_func();
}

int OptKmeansAlgorithm::mount(IndexFeatures::Pointer feats) {
  if (!feats) {
    return IndexError_InvalidArgument;
  }
  if (!feats->is_matched(meta_)) {
    return IndexError_Mismatch;
  }

  // Check dimension
  auto type_ = meta_.data_type();
  switch (type_) {
    case IndexMeta::DataType::DT_INT4:
      if (feats->dimension() % 8 != 0) {
        LOG_ERROR(
            "Unsupported feature dimension %zu (dimension of int4 "
            "must be an integer multiple of 8).",
            feats->dimension());
        return IndexError_Mismatch;
      }
      break;
    case IndexMeta::DataType::DT_INT8:
      if (feats->dimension() % 4 != 0) {
        LOG_ERROR(
            "Unsupported feature dimension %zu (dimension of int8 "
            "must be an integer multiple of 4).",
            feats->dimension());
        return IndexError_Mismatch;
      }
      break;
    case IndexMeta::DataType::DT_BINARY32:
    case IndexMeta::DataType::DT_BINARY64:
      if (feats->dimension() % 32 != 0) {
        LOG_ERROR(
            "Unsupported feature dimension %zu (dimension of binary "
            "must be an integer multiple of 32).",
            feats->dimension());
        return IndexError_Mismatch;
      }
      break;
    default:
      break;
  }

  features_ = std::move(feats);
  return 0;
}

void OptKmeansAlgorithm::suggest(uint32_t k) {
  cluster_count_ = k;
}

int OptKmeansAlgorithm::classify(IndexThreads::Pointer threads,
                                 IndexCluster::CentroidList &cents) {
  if (!threads) {
    threads = std::make_shared<SingleQueueIndexThreads>(thread_count_, false);
    if (!threads) {
      return IndexError_NoMemory;
    }
  }
  if (cents.empty()) {
    LOG_ERROR("The input centroid's list is empty.");
    return IndexError_InvalidArgument;
  }

  if (!this->check_centroids(cents)) {
    LOG_ERROR("The input centroid's list includes some invalid centroids.");
    return IndexError_InvalidArgument;
  }

  if (!this->is_valid()) {
    LOG_ERROR("The cluster is not ready.");
    return IndexError_NoReady;
  }

  this->update_clusters(threads.get(), cents);
  this->update_features(threads.get(), cents);
  return 0;
}

int OptKmeansAlgorithm::label(IndexThreads::Pointer threads,
                              const IndexCluster::CentroidList &cents,
                              std::vector<uint32_t> *out) {
  if (!threads) {
    threads = std::make_shared<SingleQueueIndexThreads>(thread_count_, false);
    if (!threads) {
      return IndexError_NoMemory;
    }
  }
  if (cents.empty()) {
    LOG_ERROR("The input centroid's list is empty.");
    return IndexError_InvalidArgument;
  }

  if (!this->check_centroids(cents)) {
    LOG_ERROR("The input centroid's list includes some invalid centroids.");
    return IndexError_InvalidArgument;
  }

  if (!this->is_valid()) {
    LOG_ERROR("The cluster is not ready.");
    return IndexError_NoReady;
  }

  this->update_labels(threads.get(), out, cents);
  return 0;
}

int OptKmeansAlgorithm::update(const ailego::Params &params) {
  this->update_params(params);
  // algorithm_->reset(cluster_count_);
  return 0;
}

int OptKmeansAlgorithm::reset(void) {
  features_.reset();
  shard_cluster_features_.clear();

  return 0;
}

int OptKmeansAlgorithm::cleanup(void) {
  features_.reset();
  shard_cluster_features_.clear();

  return 0;
}


/*! Numerical K-Means cluster algorithm
 */
template <typename T>
class NumericalKmeansAlgorithm : public OptKmeansAlgorithm {
 public:
  //! Type of value
  using ValueType = typename std::remove_cv<T>::type;

  // Check supporting type
  static_assert(ailego::IsArithmetic<ValueType>::value,
                "ValueType must be arithmetic");

  //! Constructor
  NumericalKmeansAlgorithm(void) {}

  //! Destructor
  virtual ~NumericalKmeansAlgorithm(void) {}

  //! Cluster
  virtual int cluster(IndexThreads::Pointer threads,
                      IndexCluster::CentroidList &cents);

 protected:
  void update_centroids(
      IndexCluster::CentroidList &cents,
      const ailego::NumericalKmeans<T, IndexThreads> &algorithm);
};

template <typename T>
void NumericalKmeansAlgorithm<T>::update_centroids(
    IndexCluster::CentroidList &cents,
    const ailego::NumericalKmeans<T, IndexThreads> &algorithm) {
  this->init_centroids(algorithm.centroids().count(), &cents);
  for (size_t i = 0; i < cents.size(); ++i) {
    IndexCluster::Centroid *centroid = &(cents.at(i));
    centroid->set_score(algorithm.context().clusters()[i].cost());
    centroid->set_follows(algorithm.context().clusters()[i].count());
    centroid->set_feature(algorithm.centroids()[i],
                          meta_.dimension() * sizeof(T));
  }
}

template <typename T>
int NumericalKmeansAlgorithm<T>::cluster(IndexThreads::Pointer threads,
                                         IndexCluster::CentroidList &cents) {
  ailego::ElapsedTime stamp;

  if (!threads) {
    threads = std::make_shared<SingleQueueIndexThreads>(thread_count_, false);
    if (!threads) {
      return IndexError_NoMemory;
    }
  }
  if (!this->check_centroids(cents)) {
    LOG_ERROR("The input centroid's list includes some invalid centroids.");
    return IndexError_InvalidArgument;
  }

  if (!this->is_valid()) {
    LOG_ERROR("The cluster is not ready.");
    return IndexError_NoReady;
  }

  // get cluster algorithm
  size_t centroid_count =
      cents.empty()
          ? std::min(cluster_count_, static_cast<uint32_t>(features_->count()))
          : cents.size();
  if (centroid_count == 0) {
    LOG_ERROR("The count of cluster is unknown.");
    return IndexError_NoReady;
  }
  ailego::NumericalKmeans<T, IndexThreads> algorithm(centroid_count,
                                                     meta_.dimension());

  // mount features into algorithm
  auto features_count = features_->count();
  auto dim = meta_.dimension();

  algorithm.feature_matrix_reserve(features_count);

  for (size_t i = 0; i < features_count; ++i) {
    auto vec = reinterpret_cast<const T *>(features_->element(i));
    algorithm.append(vec, dim);
  }

  if (!cents.empty()) {
    auto centroids = algorithm.mutable_centroids();
    centroids->reserve(cents.size());
    for (const auto &it : cents) {
      centroids->append(reinterpret_cast<const T *>(it.feature()),
                        meta_.dimension());
    }
  } else {
    ailego::Kmc2CentroidsGenerator<
        ailego::NumericalKmeans<ValueType, IndexThreads>, IndexThreads>
        g;
    g.set_chain_length(markov_chain_length_);
    g.set_assumption_free(assumption_free_);
    algorithm.init_centroids(*threads, g);
  }

  double cost = 0.0;

  for (uint32_t i = 0; i < max_iterations_; ++i) {
    double old_cost, new_epsilon;
    old_cost = cost;

    bool result = algorithm.cluster_once(*threads, &cost);
    if (result != true) {
      LOG_ERROR("(%u) Failed to cluster.", i + 1);
      return -1;
    }

    new_epsilon = std::abs(cost - old_cost);
    LOG_DEBUG("(%u) Updated %zu Clusters, %zu Features: %zu ms, %f -> %f = %f",
              i, algorithm.centroids().count(), features_->count(),
              (size_t)stamp.milli_seconds(), old_cost, cost, new_epsilon);
    stamp.reset();

    if (new_epsilon < epsilon_) {
      break;
    }
  }

  // update_centroids(cents);
  update_centroids(cents, algorithm);

  // Purge the empty centroids
  PurgeCentroids(cents, purge_empty_);
  return 0;
}

/*! Nibble K-Means cluster algorithm
 */
template <typename T>
class NibbleKmeansAlgorithm : public OptKmeansAlgorithm {
 public:
  //! Type of value
  using ValueType = typename std::remove_cv<T>::type;

  // Check supporting type
  static_assert(ailego::IsArithmetic<ValueType>::value,
                "ValueType must be arithmetic");

  //! Constructor
  NibbleKmeansAlgorithm(void) {}

  //! Destructor
  virtual ~NibbleKmeansAlgorithm(void) {}

  //! Cluster
  virtual int cluster(IndexThreads::Pointer threads,
                      IndexCluster::CentroidList &cents);

 protected:
  //! update centroids
  void update_centroids(IndexCluster::CentroidList &cents,
                        const ailego::NibbleKmeans<T, IndexThreads> &algorithm);
};

template <typename T>
void NibbleKmeansAlgorithm<T>::update_centroids(
    IndexCluster::CentroidList &cents,
    const ailego::NibbleKmeans<T, IndexThreads> &algorithm) {
  this->init_centroids(algorithm.centroids().count(), &cents);
  for (size_t i = 0; i < cents.size(); ++i) {
    IndexCluster::Centroid *centroid = &(cents.at(i));
    centroid->set_score(algorithm.context().clusters()[i].cost());
    centroid->set_follows(algorithm.context().clusters()[i].count());
    centroid->set_feature(algorithm.centroids()[i], (meta_.dimension() >> 1));
  }
}

template <typename T>
int NibbleKmeansAlgorithm<T>::cluster(IndexThreads::Pointer threads,
                                      IndexCluster::CentroidList &cents) {
  ailego::ElapsedTime stamp;

  if (!threads) {
    threads = std::make_shared<SingleQueueIndexThreads>(thread_count_, false);
    if (!threads) {
      return IndexError_NoMemory;
    }
  }
  if (!this->check_centroids(cents)) {
    LOG_ERROR("The input centroid's list includes some invalid centroids.");
    return IndexError_InvalidArgument;
  }

  if (!this->is_valid()) {
    LOG_ERROR("The cluster is not ready.");
    return IndexError_NoReady;
  }

  // get cluster algorithm
  size_t centroid_count =
      cents.empty()
          ? std::min(cluster_count_, static_cast<uint32_t>(features_->count()))
          : cents.size();
  if (centroid_count == 0) {
    LOG_ERROR("The count of cluster is unknown.");
    return IndexError_NoReady;
  }
  ailego::NibbleKmeans<T, IndexThreads> algorithm(centroid_count,
                                                  meta_.dimension());

  // mount features into algorithm
  auto features_count = features_->count();
  auto dim = meta_.dimension();
  for (size_t i = 0; i < features_count; ++i) {
    auto vec = reinterpret_cast<const typename std::make_unsigned<T>::type *>(
        features_->element(i));
    algorithm.append(vec, dim);
  }

  if (!cents.empty()) {
    auto centroids = algorithm.mutable_centroids();
    centroids->reserve(cents.size());
    for (const auto &it : cents) {
      centroids->append(
          reinterpret_cast<const typename std::make_unsigned<T>::type *>(
              it.feature()),
          size_t(meta_.dimension()));
    }
  } else {
    ailego::Kmc2CentroidsGenerator<
        ailego::NibbleKmeans<ValueType, IndexThreads>, IndexThreads>
        g;
    g.set_chain_length(markov_chain_length_);
    g.set_assumption_free(assumption_free_);
    algorithm.init_centroids(*threads, g);
  }

  double cost = 0.0;

  for (uint32_t i = 0; i < max_iterations_; ++i) {
    double old_cost, new_epsilon;
    old_cost = cost;

    bool result = algorithm.cluster_once(*threads, &cost);
    if (result != true) {
      LOG_ERROR("(%u) Failed to cluster.", i + 1);
      return -1;
    }

    new_epsilon = std::abs(cost - old_cost);
    LOG_DEBUG(
        "(%u) Updated %zu Clusters, %zu Features: %zu ms, %f -> "
        "%f = %f",
        i, algorithm.centroids().count(), features_->count(),
        (size_t)stamp.milli_seconds(), old_cost, cost, new_epsilon);
    stamp.reset();

    if (new_epsilon < epsilon_) {
      break;
    }
  }

  // update centroids
  update_centroids(cents, algorithm);

  // Purge the empty centroids
  PurgeCentroids(cents, purge_empty_);
  return 0;
}

/*! Binary K-Means cluster algorithm
 */
template <typename T>
class BinaryKmeansAlgorithm : public OptKmeansAlgorithm {
 public:
  //! Type of value
  using ValueType = typename std::remove_cv<T>::type;

  // Check supporting type
  static_assert(ailego::IsArithmetic<ValueType>::value,
                "ValueType must be arithmetic");

  //! Constructor
  BinaryKmeansAlgorithm(void) {}

  //! Destructor
  virtual ~BinaryKmeansAlgorithm(void) {}

  //! Cluster
  virtual int cluster(IndexThreads::Pointer threads,
                      IndexCluster::CentroidList &cents);

 protected:
  //! update centroids
  void update_centroids(IndexCluster::CentroidList &cents,
                        const ailego::BinaryKmeans<T, IndexThreads> &algorithm);
};

template <typename T>
void BinaryKmeansAlgorithm<T>::update_centroids(
    IndexCluster::CentroidList &cents,
    const ailego::BinaryKmeans<T, IndexThreads> &algorithm) {
  this->init_centroids(algorithm.centroids().count(), &cents);
  for (size_t i = 0; i < cents.size(); ++i) {
    IndexCluster::Centroid *centroid = &(cents.at(i));
    centroid->set_score(algorithm.context().clusters()[i].cost());
    centroid->set_follows(algorithm.context().clusters()[i].count());
    centroid->set_feature(algorithm.centroids()[i], (meta_.dimension() >> 3));
  }
}

template <typename T>
int BinaryKmeansAlgorithm<T>::cluster(IndexThreads::Pointer threads,
                                      IndexCluster::CentroidList &cents) {
  ailego::ElapsedTime stamp;

  if (!threads) {
    threads = std::make_shared<SingleQueueIndexThreads>(thread_count_, false);
    if (!threads) {
      return IndexError_NoMemory;
    }
  }
  if (!this->check_centroids(cents)) {
    LOG_ERROR("The input centroid's list includes some invalid centroids.");
    return IndexError_InvalidArgument;
  }

  if (!this->is_valid()) {
    LOG_ERROR("The cluster is not ready.");
    return IndexError_NoReady;
  }

  // get cluster algorithm
  size_t centroid_count =
      cents.empty()
          ? std::min(cluster_count_, static_cast<uint32_t>(features_->count()))
          : cents.size();
  if (centroid_count == 0) {
    LOG_ERROR("The count of cluster is unknown.");
    return IndexError_NoReady;
  }
  ailego::BinaryKmeans<T, IndexThreads> algorithm(centroid_count,
                                                  meta_.dimension());

  // mount features into algorithm
  auto features_count = features_->count();
  auto dim = meta_.dimension();
  for (size_t i = 0; i < features_count; ++i) {
    auto vec = reinterpret_cast<const T *>(features_->element(i));
    algorithm.append(vec, dim);
  }

  if (!cents.empty()) {
    auto centroids = algorithm.mutable_centroids();
    centroids->reserve(cents.size());
    for (const auto &it : cents) {
      centroids->append(reinterpret_cast<const T *>(it.feature()),
                        meta_.dimension());
    }
  } else {
    ailego::Kmc2CentroidsGenerator<
        ailego::BinaryKmeans<ValueType, IndexThreads>, IndexThreads>
        g;
    g.set_chain_length(markov_chain_length_);
    g.set_assumption_free(assumption_free_);
    algorithm.init_centroids(*threads, g);
  }

  double cost = 0.0;

  for (uint32_t i = 0; i < max_iterations_; ++i) {
    double old_cost, new_epsilon;
    old_cost = cost;

    bool result = algorithm.cluster_once(*threads, &cost);
    if (result != true) {
      LOG_ERROR("(%u) Failed to cluster.", i + 1);
      return -1;
    }

    new_epsilon = std::abs(cost - old_cost);
    LOG_DEBUG(
        "(%u) Updated %zu Clusters, %zu Features: %zu ms, %f -> "
        "%f = %f",
        i, algorithm.centroids().count(), features_->count(),
        (size_t)stamp.milli_seconds(), old_cost, cost, new_epsilon);
    stamp.reset();

    if (new_epsilon < epsilon_) {
      break;
    }
  }

  // update centroids
  update_centroids(cents, algorithm);

  // Purge the empty centroids
  PurgeCentroids(cents, purge_empty_);
  return 0;
}

/*! Numerical K-Means cluster algorithm
 */
template <typename T>
class NumericalInnerProductKmeansAlgorithm : public OptKmeansAlgorithm {
 public:
  //! Type of value
  using ValueType = typename std::remove_cv<T>::type;

  // Check supporting type
  static_assert(ailego::IsArithmetic<ValueType>::value,
                "ValueType must be arithmetic");

  //! Constructor
  NumericalInnerProductKmeansAlgorithm(void) {}

  //! Destructor
  virtual ~NumericalInnerProductKmeansAlgorithm(void) {}

  //! Cluster
  virtual int cluster(IndexThreads::Pointer threads,
                      IndexCluster::CentroidList &cents);

 protected:
  void update_centroids(
      IndexCluster::CentroidList &cents,
      const ailego::NumericalInnerProductKmeans<T, IndexThreads> &algorithm);
};

template <typename T>
void NumericalInnerProductKmeansAlgorithm<T>::update_centroids(
    IndexCluster::CentroidList &cents,
    const ailego::NumericalInnerProductKmeans<T, IndexThreads> &algorithm) {
  this->init_centroids(algorithm.centroids().count(), &cents);
  for (size_t i = 0; i < cents.size(); ++i) {
    IndexCluster::Centroid *centroid = &(cents.at(i));
    centroid->set_score(algorithm.context().clusters()[i].cost());
    centroid->set_follows(algorithm.context().clusters()[i].count());
    centroid->set_feature(algorithm.centroids()[i],
                          meta_.dimension() * sizeof(T));
  }
}

template <typename T>
int NumericalInnerProductKmeansAlgorithm<T>::cluster(
    IndexThreads::Pointer threads, IndexCluster::CentroidList &cents) {
  ailego::ElapsedTime stamp;

  if (!threads) {
    threads = std::make_shared<SingleQueueIndexThreads>(thread_count_, false);
    if (!threads) {
      return IndexError_NoMemory;
    }
  }
  if (!this->check_centroids(cents)) {
    LOG_ERROR("The input centroid's list includes some invalid centroids.");
    return IndexError_InvalidArgument;
  }

  if (!this->is_valid()) {
    LOG_ERROR("The cluster is not ready.");
    return IndexError_NoReady;
  }

  // get cluster algorithm
  size_t centroid_count =
      cents.empty()
          ? std::min(cluster_count_, static_cast<uint32_t>(features_->count()))
          : cents.size();
  if (centroid_count == 0) {
    LOG_ERROR("The count of cluster is unknown.");
    return IndexError_NoReady;
  }
  ailego::NumericalInnerProductKmeans<T, IndexThreads> algorithm(
      centroid_count, meta_.dimension(), true);

  // mount features into algorithm
  auto features_count = features_->count();
  auto dim = meta_.dimension();

  algorithm.feature_matrix_reserve(features_count);

  for (size_t i = 0; i < features_count; ++i) {
    auto vec = reinterpret_cast<const T *>(features_->element(i));
    algorithm.append(vec, dim);
  }

  if (!cents.empty()) {
    auto centroids = algorithm.mutable_centroids();
    centroids->reserve(cents.size());
    for (const auto &it : cents) {
      centroids->append(reinterpret_cast<const T *>(it.feature()),
                        meta_.dimension());
    }
  } else {
    ailego::Kmc2CentroidsGenerator<
        ailego::NumericalInnerProductKmeans<ValueType, IndexThreads>,
        IndexThreads>
        g;
    g.set_chain_length(markov_chain_length_);
    g.set_assumption_free(assumption_free_);
    algorithm.init_centroids(*threads, g);
  }

  double cost = 0.0;

  for (uint32_t i = 0; i < max_iterations_; ++i) {
    double old_cost, new_epsilon;
    old_cost = cost;

    bool result = algorithm.cluster_once(*threads, &cost);
    if (result != true) {
      LOG_ERROR("(%u) Failed to cluster.", i + 1);
      return -1;
    }

    new_epsilon = std::abs(cost - old_cost);
    LOG_DEBUG("(%u) Updated %zu Clusters, %zu Features: %zu ms, %f -> %f = %f",
              i, algorithm.centroids().count(), features_->count(),
              (size_t)stamp.milli_seconds(), old_cost, cost, new_epsilon);
    stamp.reset();

    if (new_epsilon < epsilon_) {
      break;
    }
  }

  // update_centroids(cents);
  update_centroids(cents, algorithm);

  // Purge the empty centroids
  PurgeCentroids(cents, purge_empty_);
  return 0;
}

/*! Nibble Inner Product K-Means cluster algorithm
 */
template <typename T>
class NibbleInnerProductKmeansAlgorithm : public OptKmeansAlgorithm {
 public:
  //! Type of value
  using ValueType = typename std::remove_cv<T>::type;

  // Check supporting type
  static_assert(ailego::IsArithmetic<ValueType>::value,
                "ValueType must be arithmetic");

  //! Constructor
  NibbleInnerProductKmeansAlgorithm(void) {}

  //! Destructor
  virtual ~NibbleInnerProductKmeansAlgorithm(void) {}

  //! Cluster
  virtual int cluster(IndexThreads::Pointer threads,
                      IndexCluster::CentroidList &cents);

 protected:
  //! update centroids
  void update_centroids(
      IndexCluster::CentroidList &cents,
      const ailego::NibbleInnerProductKmeans<T, IndexThreads> &algorithm);
};

template <typename T>
void NibbleInnerProductKmeansAlgorithm<T>::update_centroids(
    IndexCluster::CentroidList &cents,
    const ailego::NibbleInnerProductKmeans<T, IndexThreads> &algorithm) {
  this->init_centroids(algorithm.centroids().count(), &cents);
  for (size_t i = 0; i < cents.size(); ++i) {
    IndexCluster::Centroid *centroid = &(cents.at(i));
    centroid->set_score(algorithm.context().clusters()[i].cost());
    centroid->set_follows(algorithm.context().clusters()[i].count());
    centroid->set_feature(algorithm.centroids()[i], (meta_.dimension() >> 1));
  }
}

template <typename T>
int NibbleInnerProductKmeansAlgorithm<T>::cluster(
    IndexThreads::Pointer threads, IndexCluster::CentroidList &cents) {
  ailego::ElapsedTime stamp;

  if (!threads) {
    threads = std::make_shared<SingleQueueIndexThreads>(thread_count_, false);
    if (!threads) {
      return IndexError_NoMemory;
    }
  }
  if (!this->check_centroids(cents)) {
    LOG_ERROR("The input centroid's list includes some invalid centroids.");
    return IndexError_InvalidArgument;
  }

  if (!this->is_valid()) {
    LOG_ERROR("The cluster is not ready.");
    return IndexError_NoReady;
  }

  // get cluster algorithm
  size_t centroid_count =
      cents.empty()
          ? std::min(cluster_count_, static_cast<uint32_t>(features_->count()))
          : cents.size();
  if (centroid_count == 0) {
    LOG_ERROR("The count of cluster is unknown.");
    return IndexError_NoReady;
  }
  ailego::NibbleInnerProductKmeans<T, IndexThreads> algorithm(
      centroid_count, meta_.dimension());

  // mount features into algorithm
  auto features_count = features_->count();
  auto dim = meta_.dimension();
  for (size_t i = 0; i < features_count; ++i) {
    auto vec = reinterpret_cast<const typename std::make_unsigned<T>::type *>(
        features_->element(i));
    algorithm.append(vec, dim);
  }

  if (!cents.empty()) {
    auto centroids = algorithm.mutable_centroids();
    centroids->reserve(cents.size());
    for (const auto &it : cents) {
      centroids->append(
          reinterpret_cast<const typename std::make_unsigned<T>::type *>(
              it.feature()),
          size_t(meta_.dimension()));
    }
  } else {
    ailego::Kmc2CentroidsGenerator<
        ailego::NibbleInnerProductKmeans<ValueType, IndexThreads>, IndexThreads>
        g;
    g.set_chain_length(markov_chain_length_);
    g.set_assumption_free(assumption_free_);
    algorithm.init_centroids(*threads, g);
  }

  double cost = 0.0;

  for (uint32_t i = 0; i < max_iterations_; ++i) {
    double old_cost, new_epsilon;
    old_cost = cost;

    bool result = algorithm.cluster_once(*threads, &cost);
    if (result != true) {
      LOG_ERROR("(%u) Failed to cluster.", i + 1);
      return -1;
    }

    new_epsilon = std::abs(cost - old_cost);
    LOG_DEBUG(
        "(%u) Updated %zu Clusters, %zu Features: %zu ms, %f -> "
        "%f = %f",
        i, algorithm.centroids().count(), features_->count(),
        (size_t)stamp.milli_seconds(), old_cost, cost, new_epsilon);
    stamp.reset();

    if (new_epsilon < epsilon_) {
      break;
    }
  }

  // update centroids
  update_centroids(cents, algorithm);

  // Purge the empty centroids
  PurgeCentroids(cents, purge_empty_);
  return 0;
}

/*! Kmeans Cluster
 */
class OptKmeansCluster : public IndexCluster {
 public:
  //! Constructor
  OptKmeansCluster(void) {}

  //! Destructor
  virtual ~OptKmeansCluster(void) {}

  //! Initialize Cluster
  virtual int init(const IndexMeta &meta, const ailego::Params &params);

  //! Cleanup Cluster
  virtual int cleanup(void);

  //! Reset Cluster
  virtual int reset(void);

  //! Update Cluster
  virtual int update(const ailego::Params &params);

  //! Suggest dividing to K clusters
  virtual void suggest(uint32_t k);

  //! Mount features
  virtual int mount(IndexFeatures::Pointer feats);

  //! Cluster
  virtual int cluster(IndexThreads::Pointer threads,
                      IndexCluster::CentroidList &cents);

  //! Classify
  virtual int classify(IndexThreads::Pointer threads,
                       IndexCluster::CentroidList &cents);

  //! Label
  virtual int label(IndexThreads::Pointer threads,
                    const IndexCluster::CentroidList &cents,
                    std::vector<uint32_t> *out);

 protected:
  //! Members
  IndexCluster::Pointer algorithm_{};
};

//! Cluster
int OptKmeansCluster::cluster(IndexThreads::Pointer threads,
                              IndexCluster::CentroidList &cents) {
  return algorithm_->cluster(std::move(threads), cents);
}

//! Classify
int OptKmeansCluster::classify(IndexThreads::Pointer threads,
                               IndexCluster::CentroidList &cents) {
  return algorithm_->classify(std::move(threads), cents);
}

//! Label
int OptKmeansCluster::label(IndexThreads::Pointer threads,
                            const IndexCluster::CentroidList &cents,
                            std::vector<uint32_t> *out) {
  return algorithm_->label(std::move(threads), cents, out);
}

//! Update Cluster
int OptKmeansCluster::update(const ailego::Params &params) {
  return algorithm_->update(params);
}

//! Reset Cluster
int OptKmeansCluster::reset(void) {
  return algorithm_->reset();
}

//! Cleanup Cluster
int OptKmeansCluster::cleanup(void) {
  return algorithm_->cleanup();
}

//! Suggest dividing to K clusters
void OptKmeansCluster::suggest(uint32_t k) {
  algorithm_->suggest(k);
}

int OptKmeansCluster::mount(IndexFeatures::Pointer feats) {
  return algorithm_->mount(feats);
}

int OptKmeansCluster::init(const IndexMeta &meta,
                           const ailego::Params &params) {
  auto type_ = meta.data_type();

  if (meta.metric_name() == "InnerProduct" || meta.metric_name() == "Cosine") {
    switch (type_) {
      case IndexMeta::DataType::DT_FP16: {
        algorithm_.reset(
            new (std::nothrow)
                NumericalInnerProductKmeansAlgorithm<ailego::Float16>);
        break;
      }
      case IndexMeta::DataType::DT_FP32: {
        algorithm_.reset(new (std::nothrow)
                             NumericalInnerProductKmeansAlgorithm<float>);
        break;
      }
      case IndexMeta::DataType::DT_FP64: {
        algorithm_.reset(new (std::nothrow)
                             NumericalInnerProductKmeansAlgorithm<double>);
        break;
      }
      case IndexMeta::DataType::DT_INT8: {
        algorithm_.reset(new (std::nothrow)
                             NumericalInnerProductKmeansAlgorithm<int8_t>);
        break;
      }
      case IndexMeta::DataType::DT_INT16: {
        algorithm_.reset(new (std::nothrow)
                             NumericalInnerProductKmeansAlgorithm<int16_t>);
        break;
      }
      case IndexMeta::DataType::DT_INT4: {
        algorithm_.reset(new (std::nothrow)
                             NibbleInnerProductKmeansAlgorithm<int32_t>);
        break;
      }
      default: {
        LOG_ERROR("Unsupported feature types %d.", type_);
        return IndexError_Mismatch;
      }
    }
  } else {
    switch (type_) {
      case IndexMeta::DataType::DT_FP16: {
        algorithm_.reset(new (std::nothrow)
                             NumericalKmeansAlgorithm<ailego::Float16>);
        break;
      }
      case IndexMeta::DataType::DT_FP32: {
        algorithm_.reset(new (std::nothrow) NumericalKmeansAlgorithm<float>);
        break;
      }
      case IndexMeta::DataType::DT_FP64: {
        algorithm_.reset(new (std::nothrow) NumericalKmeansAlgorithm<double>);
        break;
      }
      case IndexMeta::DataType::DT_INT8: {
        algorithm_.reset(new (std::nothrow) NumericalKmeansAlgorithm<int8_t>);
        break;
      }
      case IndexMeta::DataType::DT_INT16: {
        algorithm_.reset(new (std::nothrow) NumericalKmeansAlgorithm<int16_t>);
        break;
      }
      case IndexMeta::DataType::DT_INT4: {
        algorithm_.reset(new (std::nothrow) NibbleKmeansAlgorithm<int32_t>);
        break;
      }
      // TODO
      case IndexMeta::DataType::DT_BINARY32: {
        algorithm_.reset(new (std::nothrow) BinaryKmeansAlgorithm<uint32_t>);
        break;
      }

#if defined(AILEGO_M64)
      case IndexMeta::DataType::DT_BINARY64: {
        algorithm_.reset(new (std::nothrow) BinaryKmeansAlgorithm<uint64_t>);
        break;
      }
#endif  // AILEGO_M64

      default: {
        LOG_ERROR("Unsupported feature types %d.", type_);
        return IndexError_Mismatch;
      }
    }
  }

  algorithm_->init(meta, params);

  return 0;
}

INDEX_FACTORY_REGISTER_CLUSTER(OptKmeansCluster);

}  // namespace core
}  // namespace zvec
