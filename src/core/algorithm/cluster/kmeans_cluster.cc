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
#include <ailego/container/reservoir.h>
#include <ailego/utility/time_helper.h>
#include <zvec/ailego/utility/float_helper.h>
#include "framework/index_cluster.h"
#include "framework/index_error.h"
#include "framework/index_factory.h"
#include "cluster_params.h"
#include "linear_seeker.h"
#include "vector_mean.h"

namespace zvec {
namespace core {

/*! Kmeans Cluster
 */
class KmeansCluster : public IndexCluster {
 public:
  //! Constructor
  KmeansCluster(void) {}

  //! Constructor
  KmeansCluster(size_t iters, bool batch)
      : max_iterations_(iters), batch_(batch) {}

  //! Constructor
  KmeansCluster(bool batch) : batch_(batch) {}

  //! Destructor
  virtual ~KmeansCluster(void) {}

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
  //! Test if it is valid
  bool is_valid(void) const;

  //! Cluster once
  int clustering(IndexThreads *threads, IndexCluster::CentroidList &cents,
                 double *cost);

  //! Update parameters
  void update_params(const ailego::Params &params);

  //! Init seeker
  int init_seeker(void);

  //! Build seeker
  int build_seeker(const IndexCluster::CentroidList &cents);

  //! Check Centroids
  bool check_centroids(const IndexCluster::CentroidList &cents);

  //! Initialize Centroids
  void init_centroids(size_t count, IndexCluster::CentroidList *out);

  //! Initialize Shard Containers
  void init_containers(size_t shard_count);

  //! Initialize Shard Features Containers
  void init_features_containers(size_t shard_count);

  //! Split Clusters
  void split_clusters(IndexThreads *threads,
                      const IndexCluster::CentroidList &cents);

  //! Update Centroids
  void update_centroids(IndexThreads *threads,
                        IndexCluster::CentroidList &cents);

  //! Update Clusters
  void update_clusters(IndexThreads *threads,
                       const IndexCluster::CentroidList &cents);

  //! Update Clusters' Features
  void update_features(IndexThreads *threads,
                       IndexCluster::CentroidList &cents);

  //! Update Labels
  void update_labels(IndexThreads *threads, std::vector<uint32_t> *labels);

  //! Split Clusters in Thread
  void split_clusters_thread(size_t index_begin, size_t index_end,
                             const IndexThreads *threads);

  //! Update Centroid in Thread
  void update_centroid_thread(size_t column, IndexCluster::CentroidList *out);

  //! Update Cluster in Thread
  void update_cluster_thread(size_t index_begin, size_t index_end,
                             const IndexThreads *threads);

  //! Update Cluster's Features in Thread
  void update_features_thread(size_t column, IndexCluster::CentroidList *out);

  //! Update Labels in Thread
  void update_labels_thread(size_t index_begin, size_t index_end,
                            std::vector<uint32_t> *labels);

 protected:
  //! Members
  IndexMeta meta_{};
  IndexFeatures::Pointer features_{};
  LinearSeeker::Pointer seeker_{};
  std::vector<double> shard_cluster_scores_{};
  std::vector<std::vector<const void *>> shard_cluster_features_{};
  std::shared_ptr<VectorMeanArray> shard_cluster_means_{};
  std::shared_ptr<VectorMeanArray> batch_means_{};
  std::vector<double> batch_scores_{};
  double epsilon_{std::numeric_limits<float>::epsilon()};
  float shard_factor_{16.0f};
  uint32_t max_iterations_{20u};
  uint32_t cluster_count_{0u};
  uint32_t thread_count_{0u};
  bool batch_{false};
  bool purge_empty_{false};
};

/*! Centroid Features
 */
class KmeansCentroidFeatures : public IndexFeatures {
 public:
  //! Constructor
  KmeansCentroidFeatures(const IndexMeta &meta,
                         const IndexCluster::CentroidList &cents)
      : centroids_(cents),
        feature_size_(meta.element_size()),
        feature_dimension_(meta.dimension()),
        data_type_(meta.data_type()) {}

  virtual size_t count(void) const {
    return centroids_.size();
  }

  virtual size_t dimension(void) const {
    return feature_dimension_;
  }

  virtual const void *element(size_t i) const {
    return centroids_[i].feature();
  }

  virtual IndexMeta::DataType data_type(void) const {
    return data_type_;
  }

  virtual size_t element_size(void) const {
    return feature_size_;
  }

 private:
  const IndexCluster::CentroidList &centroids_;
  size_t feature_size_;
  size_t feature_dimension_;
  IndexMeta::DataType data_type_;
};

static inline std::shared_ptr<VectorMean> NewVectorMean(const IndexMeta &meta) {
  switch (meta.data_type()) {
    case IndexMeta::DataType::DT_FP16:
      return std::make_shared<NumericalVectorMean<ailego::Float16>>(
          meta.dimension());

    case IndexMeta::DataType::DT_FP32:
      return std::make_shared<NumericalVectorMean<float>>(meta.dimension());

    case IndexMeta::DataType::DT_FP64:
      return std::make_shared<NumericalVectorMean<double>>(meta.dimension());

    case IndexMeta::DataType::DT_INT8:
      return std::make_shared<NumericalVectorMean<int8_t>>(meta.dimension());

    case IndexMeta::DataType::DT_INT4:
      return std::make_shared<NibbleVectorMean<uint8_t>>(meta.dimension());

    case IndexMeta::DataType::DT_INT16:
      return std::make_shared<NumericalVectorMean<int16_t>>(meta.dimension());

    default:
      break;
  }
  // As binary default
  return std::make_shared<BinaryVectorMean>(meta.dimension());
}

static inline std::shared_ptr<VectorMeanArray> NewVectorMeanArray(
    const IndexMeta &meta) {
  switch (meta.data_type()) {
    case IndexMeta::DataType::DT_FP16:
      return std::make_shared<
          GeneralVectorMeanArray<NumericalVectorMean<ailego::Float16>>>(
          meta.dimension());

    case IndexMeta::DataType::DT_FP32:
      return std::make_shared<
          GeneralVectorMeanArray<NumericalVectorMean<float>>>(meta.dimension());

    case IndexMeta::DataType::DT_FP64:
      return std::make_shared<
          GeneralVectorMeanArray<NumericalVectorMean<double>>>(
          meta.dimension());

    case IndexMeta::DataType::DT_INT8:
      return std::make_shared<
          GeneralVectorMeanArray<NumericalVectorMean<int8_t>>>(
          meta.dimension());

    case IndexMeta::DataType::DT_INT4:
      return std::make_shared<
          GeneralVectorMeanArray<NibbleVectorMean<uint8_t>>>(meta.dimension());

    case IndexMeta::DataType::DT_INT16:
      return std::make_shared<
          GeneralVectorMeanArray<NumericalVectorMean<int16_t>>>(
          meta.dimension());

    default:
      break;
  }
  // As binary default
  return std::make_shared<GeneralVectorMeanArray<BinaryVectorMean>>(
      meta.dimension());
}

static inline std::shared_ptr<VectorMeanArray> NewVectorMeanArray(
    const IndexMeta &meta, const IndexCluster::CentroidList &cents) {
  switch (meta.data_type()) {
    case IndexMeta::DataType::DT_FP16: {
      auto ptr = std::make_shared<
          GeneralVectorMeanArray<NumericalVectorMean<ailego::Float16>>>(
          meta.dimension());

      for (const auto &it : cents) {
        ptr->emplace(reinterpret_cast<const ailego::Float16 *>(it.feature()),
                     meta.dimension(), it.follows());
      }
      return ptr;
    }

    case IndexMeta::DataType::DT_FP32: {
      auto ptr =
          std::make_shared<GeneralVectorMeanArray<NumericalVectorMean<float>>>(
              meta.dimension());

      for (const auto &it : cents) {
        ptr->emplace(reinterpret_cast<const float *>(it.feature()),
                     meta.dimension(), it.follows());
      }
      return ptr;
    }

    case IndexMeta::DataType::DT_FP64: {
      auto ptr =
          std::make_shared<GeneralVectorMeanArray<NumericalVectorMean<double>>>(
              meta.dimension());

      for (const auto &it : cents) {
        ptr->emplace(reinterpret_cast<const double *>(it.feature()),
                     meta.dimension(), it.follows());
      }
      return ptr;
    }

    case IndexMeta::DataType::DT_INT8: {
      auto ptr =
          std::make_shared<GeneralVectorMeanArray<NumericalVectorMean<int8_t>>>(
              meta.dimension());

      for (const auto &it : cents) {
        ptr->emplace(reinterpret_cast<const int8_t *>(it.feature()),
                     meta.dimension(), it.follows());
      }
      return ptr;
    }

    case IndexMeta::DataType::DT_INT4: {
      auto ptr =
          std::make_shared<GeneralVectorMeanArray<NibbleVectorMean<uint8_t>>>(
              meta.dimension());

      for (const auto &it : cents) {
        ptr->emplace(reinterpret_cast<const uint8_t *>(it.feature()),
                     meta.dimension(), it.follows());
      }
      return ptr;
    }

    case IndexMeta::DataType::DT_INT16: {
      auto ptr = std::make_shared<
          GeneralVectorMeanArray<NumericalVectorMean<int16_t>>>(
          meta.dimension());

      for (const auto &it : cents) {
        ptr->emplace(reinterpret_cast<const int16_t *>(it.feature()),
                     meta.dimension(), it.follows());
      }
      return ptr;
    }

    default:
      break;
  }

  // As binary default
  auto ptr = std::make_shared<GeneralVectorMeanArray<BinaryVectorMean>>(
      meta.dimension());

  for (const auto &it : cents) {
    ptr->emplace(it.feature(), meta.dimension(), it.follows());
  }
  return ptr;
}

static inline double CalculateSSE(const IndexCluster::CentroidList &cents) {
  double accum = 0.0;
  for (const auto &it : cents) {
    accum += it.score();
  }
  return accum;
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

int KmeansCluster::init(const IndexMeta &meta, const ailego::Params &params) {
  meta_ = meta;
  this->update_params(params);

  return this->init_seeker();
}

int KmeansCluster::cleanup(void) {
  features_.reset();
  shard_cluster_scores_.clear();
  shard_cluster_features_.clear();
  shard_cluster_means_.reset();
  batch_means_.reset();
  batch_scores_.clear();
  seeker_->cleanup();
  return 0;
}

int KmeansCluster::reset(void) {
  features_.reset();
  shard_cluster_scores_.clear();
  shard_cluster_features_.clear();
  shard_cluster_means_->clear();
  batch_means_->clear();
  batch_scores_.clear();
  seeker_->reset();
  return 0;
}

int KmeansCluster::update(const ailego::Params &params) {
  this->update_params(params);
  return 0;
}

void KmeansCluster::suggest(uint32_t k) {
  cluster_count_ = k;
}

int KmeansCluster::mount(IndexFeatures::Pointer feats) {
  if (!feats) {
    return IndexError_InvalidArgument;
  }
  if (!feats->is_matched(meta_)) {
    return IndexError_Mismatch;
  }

  // Check dimension
  auto data_type = meta_.data_type();
  switch (data_type) {
    case IndexMeta::DataType::DT_INT4:
      if (feats->dimension() % 2 != 0) {
        LOG_ERROR(
            "Unsupported feature dimension %zu (dimension of int4 "
            "must be an integer multiple of 2).",
            feats->dimension());
        return IndexError_Mismatch;
      }
      break;
    case IndexMeta::DataType::DT_BINARY32:
      if (feats->dimension() % 32 != 0) {
        LOG_ERROR(
            "Unsupported feature dimension %zu (dimension of binary32 "
            "must be an integer multiple of 32).",
            feats->dimension());
        return IndexError_Mismatch;
      }
      break;
    case IndexMeta::DataType::DT_BINARY64:
      if (feats->dimension() % 64 != 0) {
        LOG_ERROR(
            "Unsupported feature dimension %zu (dimension of binary64 "
            "must be an integer multiple of 64).",
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

int KmeansCluster::cluster(IndexThreads::Pointer threads,
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

  if (cents.empty()) {
    if (cluster_count_ == 0) {
      LOG_ERROR("The count of cluster is unknown.");
      return IndexError_NoReady;
    }
    this->init_centroids(cluster_count_, &cents);
  }

  if (batch_) {
    batch_means_ = NewVectorMeanArray(meta_, cents);
    batch_scores_.clear();
    for (const auto &it : cents) {
      batch_scores_.push_back(it.score());
    }
  }

  double cost = 0.0;

  // we need to do clustering and update the centroids' follows, even if
  // cents.size() == 1. Otherwise, the centroid with empty follows will be
  // removed if purge_empty enabled
  for (uint32_t i = 0; (i < max_iterations_) && (cents.size() > 0); ++i) {
    double new_cost, new_epsilon;

    int result = this->clustering(threads.get(), cents, &new_cost);
    if (result != 0) {
      LOG_ERROR("(%u) Failed to cluster.", i + 1);
      return result;
    }

    new_epsilon = new_cost - cost;
    LOG_DEBUG("(%u) Updated %zu Clusters, %zu Features: %zu ms, %f -> %f = %f",
              i, cents.size(), features_->count(),
              (size_t)stamp.milli_seconds(), cost, new_cost, new_epsilon);
    stamp.reset();

    new_epsilon = std::abs(new_epsilon);
    if (new_epsilon < epsilon_) {
      break;
    }
    cost = new_cost;
  }

  // Purge the empty centroids
  PurgeCentroids(cents, purge_empty_);
  return 0;
}

int KmeansCluster::classify(IndexThreads::Pointer threads,
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

  int result = this->build_seeker(cents);
  if (result != 0) {
    LOG_ERROR("Failed to build the seeker.");
    return result;
  }

  this->update_clusters(threads.get(), cents);
  this->update_features(threads.get(), cents);
  return 0;
}

int KmeansCluster::label(IndexThreads::Pointer threads,
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

  int result = this->build_seeker(cents);
  if (result != 0) {
    LOG_ERROR("Failed to build the seeker.");
    return result;
  }

  this->update_labels(threads.get(), out);
  return 0;
}

bool KmeansCluster::is_valid(void) const {
  if (!seeker_ || !features_ || !features_->count()) {
    return false;
  }
  return true;
}

int KmeansCluster::clustering(IndexThreads *threads,
                              IndexCluster::CentroidList &cents, double *cost) {
  int result = this->build_seeker(cents);
  if (result != 0) {
    LOG_ERROR("Failed to build the seeker.");
    return result;
  }

  this->split_clusters(threads, cents);
  this->update_centroids(threads, cents);
  *cost = CalculateSSE(cents);
  return 0;
}

void KmeansCluster::update_params(const ailego::Params &params) {
  params.get(GENERAL_THREAD_COUNT, &thread_count_);
  params.get(GENERAL_CLUSTER_COUNT, &cluster_count_);
  params.get(KMEANS_CLUSTER_COUNT, &cluster_count_);
  params.get(KMEANS_CLUSTER_SHARD_FACTOR, &shard_factor_);
  params.get(KMEANS_CLUSTER_EPSILON, &epsilon_);
  params.get(KMEANS_CLUSTER_MAX_ITERATIONS, &max_iterations_);
  params.get(KMEANS_CLUSTER_BATCH, &batch_);
  params.get(KMEANS_CLUSTER_PURGE_EMPTY, &purge_empty_);
}

int KmeansCluster::init_seeker(void) {
  seeker_.reset(new (std::nothrow) LinearSeeker);
  if (!seeker_) {
    LOG_ERROR("Failed to create linear seeker.");
    return IndexError_NoMemory;
  }

  int result = seeker_->init(meta_);
  if (result != 0) {
    LOG_ERROR("Failed to initialize linear seeker.");
    return result;
  }

  return 0;
}

int KmeansCluster::build_seeker(const IndexCluster::CentroidList &cents) {
  int result =
      seeker_->mount(std::make_shared<KmeansCentroidFeatures>(meta_, cents));
  if (result != 0) {
    LOG_ERROR("Failed to mount features for linear seeker.");
    return result;
  }

  return 0;
}

bool KmeansCluster::check_centroids(const IndexCluster::CentroidList &cents) {
  for (const auto &it : cents) {
    if (it.size() != meta_.element_size()) {
      return false;
    }
  }
  return true;
}

void KmeansCluster::init_centroids(size_t count,
                                   IndexCluster::CentroidList *out) {
  size_t feature_size = features_->element_size();
  size_t features_count = features_->count();
  size_t sample_count = std::min<size_t>(count, features_count);

  ailego::Reservoir<size_t> sampler(sample_count);
  for (size_t i = 0; i < features_count; ++i) {
    sampler.fill(i);
  }

  // Save centroids
  out->reserve(sampler.pool().size());
  for (auto i : sampler.pool()) {
    out->emplace_back(features_->element(i), feature_size);
  }
}

void KmeansCluster::init_containers(size_t shard_count) {
  if (!shard_cluster_means_) {
    shard_cluster_means_ = NewVectorMeanArray(meta_);
  }
  shard_cluster_means_->clear();
  shard_cluster_means_->resize(shard_count);
  shard_cluster_scores_.clear();
  shard_cluster_scores_.resize(shard_count);
}

void KmeansCluster::init_features_containers(size_t shard_count) {
  shard_cluster_features_.resize(shard_count);
  for (auto &features : shard_cluster_features_) {
    features.clear();
  }
}

void KmeansCluster::split_clusters(IndexThreads *threads,
                                   const IndexCluster::CentroidList &cents) {
  // Initilize containers
  this->init_containers(threads->count() * cents.size());
  auto task_group = threads->make_group();

  // Initilize base information
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

    // Process in work thread
    task_group->submit(
        ailego::Closure::New(this, &KmeansCluster::split_clusters_thread, index,
                             next_index, threads));

    // Next index
    index = next_index;
  }
  task_group->wait_finish();
}

void KmeansCluster::update_centroids(IndexThreads *threads,
                                     IndexCluster::CentroidList &cents) {
  auto task_group = threads->make_group();
  for (size_t i = 0; i < cents.size(); ++i) {
    task_group->submit(ailego::Closure::New(
        this, &KmeansCluster::update_centroid_thread, i, &cents));
  }
  task_group->wait_finish();
}

void KmeansCluster::update_clusters(IndexThreads *threads,
                                    const IndexCluster::CentroidList &cents) {
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
    // Process in work thread
    task_group->submit(
        ailego::Closure::New(this, &KmeansCluster::update_cluster_thread, index,
                             next_index, threads));

    // Next index
    index = next_index;
  }
  task_group->wait_finish();
}

void KmeansCluster::update_features(IndexThreads *threads,
                                    IndexCluster::CentroidList &cents) {
  auto task_group = threads->make_group();
  for (size_t i = 0; i < cents.size(); ++i) {
    // Process in work thread
    task_group->submit(ailego::Closure::New(
        this, &KmeansCluster::update_features_thread, i, &cents));
  }
  task_group->wait_finish();
}

void KmeansCluster::update_labels(IndexThreads *threads,
                                  std::vector<uint32_t> *labels) {
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
    task_group->submit(ailego::Closure::New(
        this, &KmeansCluster::update_labels_thread, index, next_index, labels));

    // Next index
    index = next_index;
  }
  task_group->wait_finish();
}

void KmeansCluster::split_clusters_thread(size_t index_begin, size_t index_end,
                                          const IndexThreads *threads) {
  size_t feature_size = features_->element_size();
  size_t thread_offset = threads->indexof_this() * seeker_->original()->count();

  for (size_t i = index_begin; i != index_end; ++i) {
    const void *feat = features_->element(i);
    LinearSeeker::Document result(0, std::numeric_limits<float>::max());

    // ignore error
    seeker_->seek(feat, meta_.element_size(), &result);

    size_t sel_column = thread_offset + result.index;
    shard_cluster_scores_[sel_column] += result.score;
    shard_cluster_means_->at(sel_column).plus(feat, feature_size);
  }
}

void KmeansCluster::update_centroid_thread(size_t column,
                                           IndexCluster::CentroidList *out) {
  size_t cluster_count = out->size();
  double cluster_score = 0.0;

  // Create Accumulator
  std::shared_ptr<VectorMean> accum = NewVectorMean(meta_);
  if (batch_) {
    cluster_score += batch_scores_[column];
    accum->merge(batch_means_->at(column));
  }

  // Compute the score of centroid
  for (size_t i = column; i < shard_cluster_scores_.size();
       i += cluster_count) {
    cluster_score += shard_cluster_scores_[i];
    accum->merge(shard_cluster_means_->at(i));
  }

  // Update centroid
  IndexCluster::Centroid *centroid = &(out->at(column));
  centroid->set_score(cluster_score);
  centroid->set_follows(accum->count());
  accum->mean(centroid->mutable_buffer());
}

void KmeansCluster::update_cluster_thread(size_t index_begin, size_t index_end,
                                          const IndexThreads *threads) {
  size_t thread_offset = threads->indexof_this() * seeker_->original()->count();

  for (size_t i = index_begin; i != index_end; ++i) {
    const void *feat = features_->element(i);
    LinearSeeker::Document result(0, std::numeric_limits<float>::max());

    // ignore error
    seeker_->seek(feat, meta_.element_size(), &result);

    size_t sel_column = thread_offset + result.index;
    shard_cluster_features_[sel_column].emplace_back(feat);
  }
}

void KmeansCluster::update_features_thread(size_t column,
                                           IndexCluster::CentroidList *out) {
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

void KmeansCluster::update_labels_thread(size_t index_begin, size_t index_end,
                                         std::vector<uint32_t> *labels) {
  for (size_t i = index_begin; i != index_end; ++i) {
    const void *feat = features_->element(i);
    LinearSeeker::Document result(0, std::numeric_limits<float>::max());

    // ignore error
    seeker_->seek(feat, meta_.element_size(), &result);
    (*labels)[i] = static_cast<uint32_t>(result.index);
  }
}

INDEX_FACTORY_REGISTER_CLUSTER_ALIAS(KmeansCluster, KmeansCluster, false);
INDEX_FACTORY_REGISTER_CLUSTER_ALIAS(BatchKmeansCluster, KmeansCluster, true);

}  // namespace core
}  // namespace zvec
