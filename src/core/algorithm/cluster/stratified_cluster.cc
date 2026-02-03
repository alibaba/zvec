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
#include <zvec/ailego/container/params.h>
#include <zvec/core/framework/index_cluster.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include "cluster_params.h"

namespace zvec {
namespace core {

/*! Stratified Cluster
 */
class StratifiedCluster : public IndexCluster {
 public:
  //! Constructor
  StratifiedCluster(void) {}

  //! Destructor
  virtual ~StratifiedCluster(void) {}

  //! Initialize Cluster
  virtual int init(const IndexMeta &meta, const ailego::Params &params) {
    meta_ = meta;
    this->update_params(params);
    return 0;
  }

  //! Cleanup Cluster
  virtual int cleanup(void) {
    features_.reset();
    return 0;
  }

  //! Reset Cluster
  virtual int reset(void) {
    features_.reset();
    return 0;
  }

  //! Update Cluster
  virtual int update(const ailego::Params &params) {
    this->update_params(params);
    return 0;
  }

  //! Suggest dividing to K clusters
  virtual void suggest(uint32_t k) {
    cluster_count_ = k;
  }

  //! Mount features
  virtual int mount(IndexFeatures::Pointer feats) {
    if (!feats) {
      return IndexError_InvalidArgument;
    }
    if (!feats->is_matched(meta_)) {
      return IndexError_Mismatch;
    }
    features_ = std::move(feats);
    return 0;
  }

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
  bool is_valid(void) const {
    if (!features_ || !features_->count()) {
      return false;
    }
    return true;
  }

  //! Update parameters
  void update_params(const ailego::Params &params);

  //! Check Centroids
  bool check_centroids(const IndexCluster::CentroidList &cents);

  //! Initialize Sub Clusters
  int init_sub_clusters(IndexCluster::Pointer *first,
                        IndexCluster::Pointer *second);

  //! Initialize First Cluster
  int init_first_cluster(IndexCluster::Pointer *first);

  //! Initialize Second Cluster
  int init_second_cluster(IndexCluster::Pointer *second,
                          IndexFeatures::Pointer features);

 private:
  //! Members
  IndexMeta meta_{};
  IndexFeatures::Pointer features_{};
  uint32_t cluster_count_{0u};
  uint32_t thread_count_{0u};
  uint32_t first_cluster_count_{0u};
  uint32_t second_cluster_count_{0u};
  bool auto_tuning_{false};
  std::string first_cluster_class_{"OptKmeansCluster"};
  std::string second_cluster_class_{"OptKmeansCluster"};
  ailego::Params first_cluster_params_{};
  ailego::Params second_cluster_params_{};

  // TODO: Maybe optimize later
  uint32_t second_threads_count_{10u};  // todo
};

int StratifiedCluster::cluster(IndexThreads::Pointer threads,
                               IndexCluster::CentroidList &cents) {
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

  IndexCluster::Pointer first_cluster;
  int result = init_first_cluster(&first_cluster);
  if (result != 0) {
    LOG_ERROR("Failed to initialize the first cluster.");
    return result;
  }

  if (first_cluster_count_) {
    first_cluster->suggest(first_cluster_count_);
  }

  // The first clustering
  LOG_DEBUG("Clustering with first cluster: %s.", first_cluster_class_.c_str());
  result = first_cluster->cluster(threads, cents);
  if (result != 0) {
    LOG_ERROR("Failed to cluster in first cluster: %s.",
              first_cluster_class_.c_str());
    return result;
  }

  result = first_cluster->classify(threads, cents);
  if (result != 0) {
    LOG_ERROR("Failed to classify in first cluster: %s.",
              first_cluster_class_.c_str());
    return result;
  }

  // Cleanup for saving memory
  first_cluster.reset();

  // Calculate the total cluster count
  uint32_t total_cluster_count = cents.size() * second_cluster_count_;
  if (cluster_count_) {
    total_cluster_count = cluster_count_;
  }

  // Use thread_threads cluster instead
  uint32_t tail_threads = threads->count() % second_threads_count_;
  std::vector<std::shared_ptr<IndexThreads>> threads_cluster;

  // TODO: reusing threads pool?
  // Incase the threads count less than second threads count
  if (threads->count() / second_threads_count_ == 0) {
    for (size_t threads_idx = 0; threads_idx < tail_threads; threads_idx++) {
      std::shared_ptr<IndexThreads> curr_threads =
          std::make_shared<SingleQueueIndexThreads>(1, false);
      threads_cluster.push_back(curr_threads);
    }
  } else {
    for (size_t threads_idx = 0; threads_idx < second_threads_count_;
         threads_idx++) {
      uint32_t curr_threads_count = threads->count() / second_threads_count_;
      if (threads_idx >= second_threads_count_ - tail_threads) {
        curr_threads_count++;
      }
      std::shared_ptr<IndexThreads> curr_threads =
          std::make_shared<SingleQueueIndexThreads>(curr_threads_count, false);
      threads_cluster.push_back(curr_threads);
    }
  }

  auto task_group = threads->make_group();
  // The second clustering
  for (size_t i = 0; i < cents.size(); ++i) {
    if (cents[i].similars().empty()) {
      continue;
    }

    IndexThreads::Pointer &curr_threads =
        threads_cluster[i % (threads_cluster.size())];

    task_group->submit(ailego::Closure::New(
        [this, &curr_threads, &total_cluster_count, &cents](size_t index) {
          auto &it = cents[index];
          IndexCluster::Pointer second_cluster;
          std::shared_ptr<FlexibleIndexFeatures> features =
              std::make_shared<FlexibleIndexFeatures>(
                  meta_, it.similars().data(), it.similars().size());

          int ret = this->init_second_cluster(&second_cluster, features);
          if (ret != 0) {
            LOG_ERROR("Failed to initialize the second cluster.");
            return;
          }

          if (auto_tuning_) {
            if (total_cluster_count) {
              double factor = static_cast<double>(it.similars().size()) /
                              static_cast<double>(this->features_->count());
              second_cluster->suggest(
                  std::max(static_cast<uint32_t>(
                               std::floor(total_cluster_count * factor)),
                           1u));
            }
          } else if (second_cluster_count_) {
            second_cluster->suggest(second_cluster_count_);
          }

          LOG_DEBUG("Clustering with second cluster: %s.",
                    second_cluster_class_.c_str());
          ret = second_cluster->cluster(curr_threads, *(it.mutable_subitems()));
          if (ret != 0) {
            LOG_ERROR("Failed to cluster in second cluster: %s.",
                      second_cluster_class_.c_str());
          }
        },
        i));
  }
  task_group->wait_finish();
  return 0;
}

int StratifiedCluster::classify(IndexThreads::Pointer threads,
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

  IndexCluster::Pointer first_cluster, second_cluster;
  int result = init_sub_clusters(&first_cluster, &second_cluster);
  if (result != 0) {
    LOG_ERROR("Failed to initialize the subclusters.");
    return result;
  }

  // The first classifying
  result = first_cluster->classify(threads, cents);
  if (result != 0) {
    LOG_ERROR("Failed to classify in first cluster: %s.",
              first_cluster_class_.c_str());
    return result;
  }

  // Cleanup for saving memory
  first_cluster.reset();

  std::shared_ptr<FlexibleIndexFeatures> shell =
      std::make_shared<FlexibleIndexFeatures>(meta_);

  // The second classifying
  for (IndexCluster::Centroid &it : cents) {
    const auto &feats = it.similars();

    if (feats.empty()) {
      continue;
    }

    shell->mount(feats.data(), feats.size());
    result = second_cluster->mount(shell);
    if (result != 0) {
      LOG_ERROR("Failed to mount features for second cluster: %s.",
                second_cluster_class_.c_str());
      return result;
    }

    result = second_cluster->classify(threads, *it.mutable_subitems());
    if (result != 0) {
      LOG_ERROR("Failed to classify in second cluster: %s.",
                second_cluster_class_.c_str());
      return result;
    }
  }
  return 0;
}

int StratifiedCluster::label(IndexThreads::Pointer threads,
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

  IndexCluster::Pointer first_cluster;
  int result = init_first_cluster(&first_cluster);
  if (result != 0) {
    LOG_ERROR("Failed to initialize the subclusters.");
    return result;
  }

  result = first_cluster->label(threads, cents, out);
  if (result != 0) {
    LOG_ERROR("Failed to label in first cluster: %s.",
              first_cluster_class_.c_str());
    return result;
  }
  return 0;
}

void StratifiedCluster::update_params(const ailego::Params &params) {
  params.get(GENERAL_THREAD_COUNT, &thread_count_);
  params.get(GENERAL_CLUSTER_COUNT, &cluster_count_);
  params.get(STRATIFIED_CLUSTER_COUNT, &cluster_count_);
  params.get(STRATIFIED_CLUSTER_FIRST_COUNT, &first_cluster_count_);
  params.get(STRATIFIED_CLUSTER_SECOND_COUNT, &second_cluster_count_);
  params.get(STRATIFIED_CLUSTER_FIRST_CLASS, &first_cluster_class_);
  params.get(STRATIFIED_CLUSTER_SECOND_CLASS, &second_cluster_class_);
  params.get(STRATIFIED_CLUSTER_FIRST_PARAMS, &first_cluster_params_);
  params.get(STRATIFIED_CLUSTER_SECOND_PARAMS, &second_cluster_params_);
  params.get(STRATIFIED_CLUSTER_AUTO_TUNING, &auto_tuning_);
  params.get(STRATIFIED_CLUSTER_SECOND_POOL_COUNT, &second_threads_count_);
}

bool StratifiedCluster::check_centroids(
    const IndexCluster::CentroidList &cents) {
  for (const auto &it : cents) {
    if (it.size() != meta_.element_size()) {
      return false;
    }
  }
  return true;
}

int StratifiedCluster::init_sub_clusters(IndexCluster::Pointer *first,
                                         IndexCluster::Pointer *second) {
  IndexCluster::Pointer first_cluster =
      IndexFactory::CreateCluster(first_cluster_class_);

  if (!first_cluster) {
    LOG_ERROR("Failed to create first cluster: %s.",
              first_cluster_class_.c_str());
    return IndexError_NoExist;
  }

  IndexCluster::Pointer second_cluster =
      IndexFactory::CreateCluster(second_cluster_class_);

  if (!second_cluster) {
    LOG_ERROR("Failed to create second cluster: %s.",
              first_cluster_class_.c_str());
    return IndexError_NoExist;
  }

  int result = first_cluster->init(meta_, first_cluster_params_);
  if (result != 0) {
    LOG_ERROR("Failed to initialize first cluster: %s.",
              first_cluster_class_.c_str());
    return result;
  }

  result = second_cluster->init(meta_, second_cluster_params_);
  if (result != 0) {
    LOG_ERROR("Failed to initialize second cluster: %s.",
              second_cluster_class_.c_str());
    return result;
  }

  result = first_cluster->mount(features_);
  if (result != 0) {
    LOG_ERROR("Failed to mount features for first cluster: %s.",
              first_cluster_class_.c_str());
    return result;
  }

  *first = std::move(first_cluster);
  *second = std::move(second_cluster);
  return 0;
}

int StratifiedCluster::init_first_cluster(IndexCluster::Pointer *first) {
  IndexCluster::Pointer first_cluster =
      IndexFactory::CreateCluster(first_cluster_class_);

  if (!first_cluster) {
    LOG_ERROR("Failed to create first cluster: %s.",
              first_cluster_class_.c_str());
    return IndexError_NoExist;
  }

  int result = first_cluster->init(meta_, first_cluster_params_);
  if (result != 0) {
    LOG_ERROR("Failed to initialize first cluster: %s.",
              first_cluster_class_.c_str());
    return result;
  }

  result = first_cluster->mount(features_);
  if (result != 0) {
    LOG_ERROR("Failed to mount features for first cluster: %s.",
              first_cluster_class_.c_str());
    return result;
  }

  *first = std::move(first_cluster);
  return 0;
}

int StratifiedCluster::init_second_cluster(IndexCluster::Pointer *second,
                                           IndexFeatures::Pointer features) {
  IndexCluster::Pointer second_cluster =
      IndexFactory::CreateCluster(second_cluster_class_);

  if (!second_cluster) {
    LOG_ERROR("Failed to create second cluster: %s.",
              second_cluster_class_.c_str());
    return IndexError_NoExist;
  }

  int result = second_cluster->init(meta_, second_cluster_params_);
  if (result != 0) {
    LOG_ERROR("Failed to initialize second cluster: %s.",
              second_cluster_class_.c_str());
    return result;
  }

  result = second_cluster->mount(features);
  if (result != 0) {
    LOG_ERROR("Failed to mount features for second cluster: %s.",
              second_cluster_class_.c_str());
    return result;
  }

  *second = std::move(second_cluster);
  return 0;
}

INDEX_FACTORY_REGISTER_CLUSTER(StratifiedCluster);

}  // namespace core
}  // namespace zvec
