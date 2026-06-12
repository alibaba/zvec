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

#include "multi_chunk_cluster.h"
#include <chrono>

namespace zvec {
namespace core {

bool MultiChunkClusterAlgorithm::is_valid(void) const {
  return features_ && features_->count();
}

bool MultiChunkClusterAlgorithm::check_centroids(
    const IndexCluster::CentroidList & /*cents*/) {
  return true;
}

void MultiChunkClusterAlgorithm::init_centroids(
    size_t count, IndexCluster::CentroidList *out) {
  // Just resize, because the get random centroid step is done by cluster_once
  out->resize(count);
}

int MultiChunkClusterAlgorithm::reset(void) {
  features_.reset();

  return 0;
}

int MultiChunkClusterAlgorithm::cleanup(void) {
  features_.reset();

  return 0;
}

void MultiChunkClusterAlgorithm::suggest(uint32_t k) {
  cluster_count_ = k;
}

int MultiChunkClusterAlgorithm::update(const ailego::Params &params) {
  this->update_params(params);
  return 0;
}

//! MultiChunkCluster
int MultiChunkClusterAlgorithm::update_params(const ailego::Params &params) {
  params.get(GENERAL_THREAD_COUNT, &thread_count_);
  params.get(GENERAL_CLUSTER_COUNT, &cluster_count_);

  params.get(MULTI_CHUNK_CLUSTER_THREAD_COUNT, &thread_count_);
  params.get(MULTI_CHUNK_CLUSTER_COUNT, &cluster_count_);
  params.get(MULTI_CHUNK_CLUSTER_CHUNK_COUNT, &chunk_count_);
  params.get(MULTI_CHUNK_CLUSTER_EPSILON, &epsilon_);
  params.get(MULTI_CHUNK_CLUSTER_MAX_ITERATIONS, &max_iterations_);
  params.get(MULTI_CHUNK_CLUSTER_MARKOV_CHAIN_LENGTH, &markov_chain_length_);

  return 0;
}

int MultiChunkClusterAlgorithm::init_distance_func() {
  IndexMetric::Pointer metric_{};
  metric_ = IndexFactory::CreateMetric(meta_.metric_name());
  if (!metric_) {
    LOG_ERROR("Create metric %s failed.", meta_.metric_name().c_str());
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

int MultiChunkClusterAlgorithm::do_chunk() {
  if (chunk_count_ == 0) {
    LOG_ERROR("Invalid Chunk Count: %u", chunk_count_);
    return IndexError_InvalidArgument;
  }

  size_t large_chunk_count = meta_.dimension() % chunk_count_;
  size_t base_chunk_dim_ = meta_.dimension() / chunk_count_;

  chunk_dims_.clear();

  for (size_t i = 0; i < chunk_count_; ++i) {
    if (i < large_chunk_count) {
      chunk_dims_.push_back(base_chunk_dim_ + 1);
    } else {
      chunk_dims_.push_back(base_chunk_dim_);
    }
  }

  chunk_dim_offsets_.clear();
  chunk_dim_offsets_.push_back(0);
  for (size_t i = 1; i < chunk_count_; ++i) {
    chunk_dim_offsets_.push_back(chunk_dim_offsets_[i - 1] +
                                 chunk_dims_[i - 1]);
  }
  chunk_dim_offsets_.push_back(meta_.dimension());

  return 0;
}

int MultiChunkClusterAlgorithm::init(const IndexMeta &meta,
                                     const ailego::Params &params) {
  meta_ = meta;

  int ret = update_params(params);
  if (ret != 0) {
    return ret;
  }

  ret = do_chunk();
  if (ret != 0) {
    return ret;
  }

  ret = init_distance_func();
  if (ret != 0) {
    return ret;
  }

  return 0;
}

int MultiChunkClusterAlgorithm::mount(IndexFeatures::Pointer features) {
  if (!features) {
    return IndexError_InvalidArgument;
  }

  if (!features->is_matched(meta_)) {
    return IndexError_Mismatch;
  }

  auto data_type = meta_.data_type();
  if (data_type != IndexMeta::DataType::DT_FP32 &&
      data_type != IndexMeta::DataType::DT_FP16) {
    LOG_ERROR("Unsupported meta type %u", data_type);

    return IndexError_Unsupported;
  }

  features_ = std::move(features);

  return 0;
}

//! cluster
int MultiChunkClusterAlgorithm::cluster(IndexThreads::Pointer threads,
                                        IndexCluster::CentroidList &cents) {
  if (chunk_count_ == 0) {
    LOG_ERROR("Invalid Chunk Count: %u", chunk_count_);

    return IndexError_InvalidArgument;
  }

  if (cluster_count_ == 0) {
    LOG_ERROR("Invalid cluster Count: %u", cluster_count_);

    return IndexError_InvalidArgument;
  }

  if (!threads) {
    threads = std::make_shared<SingleQueueIndexThreads>(thread_count_, false);
  }

  auto task_group = threads->make_group();
  if (!task_group) {
    LOG_ERROR("Failed to create task group");
    return IndexError_Runtime;
  }

  cents.clear();
  cents.resize(chunk_count_ * cluster_count_);

  std::atomic<size_t> finished{0};

  for (size_t i = 0; i < threads->count(); ++i) {
    task_group->submit(
        ailego::Closure::New(this, &MultiChunkClusterAlgorithm::do_cluster, i,
                             threads->count(), &cents, &finished));
  }

  while (!task_group->is_finished()) {
    std::unique_lock<std::mutex> lk(mutex_);
    cond_.wait_until(lk, std::chrono::system_clock::now() +
                             std::chrono::seconds(check_interval_secs_));
    if (error_.load(std::memory_order_acquire)) {
      LOG_ERROR("Failed to cluster while waiting finish");
      return errcode_;
    }
    LOG_INFO("Finish Chunk Count %zu, Finished Percent %.3f%%", finished.load(),
             finished.load() * 100.0f / chunk_count_);
  }

  if (error_.load(std::memory_order_acquire)) {
    LOG_ERROR("Failed to cluster while waiting finish");
    return errcode_;
  }

  task_group->wait_finish();

  return 0;
}

//! Classify
int MultiChunkClusterAlgorithm::classify(
    IndexThreads::Pointer /*threads*/, IndexCluster::CentroidList & /*cents*/) {
  return IndexError_Unsupported;
}

//! Label
int MultiChunkClusterAlgorithm::label(IndexThreads::Pointer threads,
                                      const IndexCluster::CentroidList &cents,
                                      std::vector<uint32_t> *out) {
  if (chunk_count_ == 0) {
    LOG_ERROR("Invalid Chunk Count: %u", chunk_count_);

    return IndexError_InvalidArgument;
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

  if (cluster_count_ == 0) {
    LOG_ERROR("Invalid cluster Count: %u", cluster_count_);

    return IndexError_InvalidArgument;
  }

  if (!threads) {
    threads = std::make_shared<SingleQueueIndexThreads>(thread_count_, false);
    if (!threads) {
      return IndexError_NoMemory;
    }
  }

  auto task_group = threads->make_group();
  if (!task_group) {
    LOG_ERROR("Failed to create task group");
    return IndexError_Runtime;
  }

  size_t features_count = features_->count();
  out->resize(features_count * chunk_count_);

  std::atomic<size_t> finished{0};

  for (size_t i = 0; i < threads->count(); ++i) {
    task_group->submit(
        ailego::Closure::New(this, &MultiChunkClusterAlgorithm::do_label, i,
                             threads->count(), cents, out, &finished));
  }

  while (!task_group->is_finished()) {
    std::unique_lock<std::mutex> lk(mutex_);
    cond_.wait_until(lk, std::chrono::system_clock::now() +
                             std::chrono::seconds(check_interval_secs_));
    if (error_.load(std::memory_order_acquire)) {
      LOG_ERROR("Failed to cluster while waiting finish");
      return errcode_;
    }
    LOG_INFO("Finish label cnt %zu, finished percent %.3f%%", finished.load(),
             finished.load() * 100.0f / features_count);
  }

  if (error_.load(std::memory_order_acquire)) {
    LOG_ERROR("Failed to cluster while waiting finish");
    return errcode_;
  }

  task_group->wait_finish();

  return 0;
}

//! Cluster
int MultiChunkCluster::cluster(IndexThreads::Pointer threads,
                               IndexCluster::CentroidList &cents) {
  return algorithm_->cluster(std::move(threads), cents);
}

//! Classify
int MultiChunkCluster::classify(IndexThreads::Pointer threads,
                                IndexCluster::CentroidList &cents) {
  return algorithm_->classify(std::move(threads), cents);
}

//! Label
int MultiChunkCluster::label(IndexThreads::Pointer threads,
                             const IndexCluster::CentroidList &cents,
                             std::vector<uint32_t> *out) {
  return algorithm_->label(std::move(threads), cents, out);
}

//! Update Cluster
int MultiChunkCluster::update(const ailego::Params &params) {
  return algorithm_->update(params);
}

//! Reset Cluster
int MultiChunkCluster::reset(void) {
  return algorithm_->reset();
}

//! Cleanup Cluster
int MultiChunkCluster::cleanup(void) {
  return algorithm_->cleanup();
}

//! Suggest dividing to K clusters
void MultiChunkCluster::suggest(uint32_t k) {
  algorithm_->suggest(k);
}

int MultiChunkCluster::mount(IndexFeatures::Pointer feats) {
  return algorithm_->mount(feats);
}

int MultiChunkCluster::init(const IndexMeta &meta,
                            const ailego::Params &params) {
  IndexMeta new_meta(meta.data_type(), meta.dimension());

  if (meta.metric_name() == "Cosine") {
    new_meta.set_dimension(meta.dimension() - 1);
    new_meta.set_metric("InnerProduct", 0, ailego::Params());
  }

  auto data_type = new_meta.data_type();

  if (new_meta.metric_name() == "InnerProduct") {
    switch (data_type) {
      case IndexMeta::DataType::DT_FP16: {
        algorithm_.reset(
            new (std::nothrow)
                MultiChunkNumericalInnerProductAlgorithm<ailego::Float16>);
        break;
      }
      case IndexMeta::DataType::DT_FP32: {
        algorithm_.reset(new (std::nothrow)
                             MultiChunkNumericalInnerProductAlgorithm<float>);
        break;
      }
      case IndexMeta::DataType::DT_FP64: {
        algorithm_.reset(new (std::nothrow)
                             MultiChunkNumericalInnerProductAlgorithm<double>);
        break;
      }
      case IndexMeta::DataType::DT_INT8: {
        algorithm_.reset(new (std::nothrow)
                             MultiChunkNumericalInnerProductAlgorithm<int8_t>);
        break;
      }
      case IndexMeta::DataType::DT_INT16: {
        algorithm_.reset(new (std::nothrow)
                             MultiChunkNumericalInnerProductAlgorithm<int16_t>);
        break;
      }
      default: {
        LOG_ERROR("Unsupported feature types %d.", data_type);
        return IndexError_Mismatch;
      }
    }
  } else {
    switch (data_type) {
      case IndexMeta::DataType::DT_FP16: {
        algorithm_.reset(new (std::nothrow)
                             MultiChunkNumericalAlgorithm<ailego::Float16>);
        break;
      }
      case IndexMeta::DataType::DT_FP32: {
        algorithm_.reset(new (std::nothrow)
                             MultiChunkNumericalAlgorithm<float>);
        break;
      }
      case IndexMeta::DataType::DT_FP64: {
        algorithm_.reset(new (std::nothrow)
                             MultiChunkNumericalAlgorithm<double>);
        break;
      }
      case IndexMeta::DataType::DT_INT8: {
        algorithm_.reset(new (std::nothrow)
                             MultiChunkNumericalAlgorithm<int8_t>);
        break;
      }
      case IndexMeta::DataType::DT_INT16: {
        algorithm_.reset(new (std::nothrow)
                             MultiChunkNumericalAlgorithm<int16_t>);
        break;
      }
      default: {
        LOG_ERROR("Unsupported feature types %d.", data_type);
        return IndexError_Mismatch;
      }
    }
  }

  algorithm_->init(new_meta, params);

  return 0;
}

}  // namespace core
}  // namespace zvec
