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
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>
#include <zvec/ailego/container/params.h>
#include <zvec/core/framework/index_cluster.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include "ailego/pattern/defer.h"
#include "utility/paged_build_buffer.h"
#include "utility/vector_build_source.h"
#include "cluster_params.h"
#include "holder_cluster.h"

namespace zvec {
namespace core {

namespace {

// A slice of ordinal build input, optionally gathered through a pageable
// permutation. Iterators own one source reader, never pointers into other
// iterators' scratch or a pinned corpus.
class StratifiedInput : public IndexHolder {
 public:
  StratifiedInput(IndexMeta meta, VectorBuildSource::Pointer source,
                  std::shared_ptr<PagedBuildBuffer> order, size_t begin,
                  size_t count)
      : meta_(std::move(meta)),
        source_(std::move(source)),
        order_(std::move(order)),
        begin_(begin),
        count_(count) {}

  size_t count() const override {
    return count_;
  }
  size_t dimension() const override {
    return meta_.dimension();
  }
  size_t element_size() const override {
    return meta_.element_size();
  }
  IndexMeta::DataType data_type() const override {
    return meta_.data_type();
  }
  bool multipass() const override {
    return true;
  }

  Iterator::Pointer create_iterator() override {
    class Cursor : public Iterator {
     public:
      Cursor(VectorBuildSource::Pointer source,
             std::shared_ptr<PagedBuildBuffer> order, size_t begin,
             size_t count)
          : source_(std::move(source)),
            order_(std::move(order)),
            position_(begin),
            end_(begin + count) {
        status_ = source_->create_ordinal_reader(&reader_);
        load();
      }
      const void *data() const override {
        return data_;
      }
      uint64_t key() const override {
        return key_;
      }
      bool is_valid() const override {
        return status_ == 0 && position_ < end_;
      }
      int status() const override {
        return status_;
      }
      void next() override {
        if (is_valid()) ++position_;
        load();
      }

     private:
      void load() {
        data_ = nullptr;
        if (!is_valid()) return;
        uint32_t id = static_cast<uint32_t>(position_);
        if (order_)
          status_ = order_->read(position_ * sizeof(id), &id, sizeof(id));
        if (status_ == 0) status_ = reader_->read(id, &key_, &data_);
        if (status_ == 0 && !data_) status_ = IndexError_ReadData;
      }
      VectorBuildSource::Pointer source_;
      std::shared_ptr<PagedBuildBuffer> order_;
      OrdinalAccessHolder::Reader::Pointer reader_;
      size_t position_;
      size_t end_;
      uint64_t key_{0};
      const void *data_{nullptr};
      int status_{0};
    };
    return std::make_unique<Cursor>(source_, order_, begin_, count_);
  }

 private:
  IndexMeta meta_;
  VectorBuildSource::Pointer source_;
  std::shared_ptr<PagedBuildBuffer> order_;
  size_t begin_;
  size_t count_;
};

// Use the selected cluster's existing label implementation (including custom
// metrics and tie handling), but only mount a bounded batch. Stable partition
// by ordinal replaces classify()'s corpus-sized lists of raw feature pointers.
int PartitionStratifiedInput(const IndexMeta &meta,
                             const VectorBuildSource::Pointer &source,
                             const IndexCluster::Pointer &cluster,
                             const IndexThreads::Pointer &threads,
                             const IndexCluster::CentroidList &cents,
                             PagedBuildBuffer *order,
                             ailego::BudgetedVector<size_t> *offsets) {
  const size_t count = source->count();
  if (count > std::numeric_limits<size_t>::max() / (2 * sizeof(uint32_t)))
    return IndexError_Overflow;
  // Labels and grouped ordinals share one backing file (and one writable
  // page table/writeback workspace), rather than doubling fixed metadata.
  const size_t order_offset = count * sizeof(uint32_t);
  int ret = order->init(2 * order_offset);
  if (ret != 0) return ret;
  OrdinalAccessHolder::Reader::Pointer reader;
  ret = source->create_ordinal_reader(&reader);
  if (ret != 0) return ret;
  const size_t batch_size = std::max<size_t>(
      1, std::min<size_t>(1024, 256 * 1024 / meta.element_size()));
  offsets->assign(cents.size() + 1, 0);
  ailego::MemoryReservation ids_memory;
  ids_memory.add_array(batch_size, sizeof(uint32_t));
  std::vector<uint32_t> ids;
  ids.reserve(batch_size);
  for (size_t begin = 0; begin < count; begin += batch_size) {
    const size_t n = std::min(batch_size, count - begin);
    auto batch = std::make_shared<CompactIndexFeatures>(meta);
    batch->reserve(n);
    for (size_t i = 0; i < n; ++i) {
      uint64_t key = 0;
      const void *data = nullptr;
      ret = reader->read(begin + i, &key, &data);
      if (ret != 0) return ret;
      if (!data) return IndexError_ReadData;
      batch->emplace(data);
    }
    ret = cluster->mount(std::move(batch));
    if (ret != 0) return ret;
    ret = cluster->label(threads, cents, &ids);
    if (ret != 0) return ret;
    if (ids.size() != n) return IndexError_InvalidLength;
    for (const auto id : ids) {
      if (id >= cents.size()) return IndexError_OutOfRange;
      ++(*offsets)[id + 1];
    }
    ret = order->write(begin * sizeof(uint32_t), ids.data(),
                       n * sizeof(uint32_t));
    if (ret != 0) return ret;
  }
  // Release the final mounted batch before second-level training starts.
  ret = cluster->reset();
  if (ret != 0) return ret;
  reader.reset();
  for (size_t i = 1; i < offsets->size(); ++i)
    (*offsets)[i] += (*offsets)[i - 1];
  auto positions = *offsets;
  for (size_t begin = 0; begin < count; begin += batch_size) {
    const size_t n = std::min(batch_size, count - begin);
    ids.resize(n);
    ret =
        order->read(begin * sizeof(uint32_t), ids.data(), n * sizeof(uint32_t));
    if (ret != 0) return ret;
    for (size_t i = 0; i < n; ++i) {
      const uint32_t ordinal = static_cast<uint32_t>(begin + i);
      ret = order->write(order_offset + positions[ids[i]]++ * sizeof(ordinal),
                         &ordinal, sizeof(ordinal));
      if (ret != 0) return ret;
    }
  }
  // Drop labels, compacting the completed permutation into the same file.
  for (size_t begin = 0; begin < count; begin += batch_size) {
    const size_t n = std::min(batch_size, count - begin);
    ids.resize(n);
    ret = order->read(order_offset + begin * sizeof(uint32_t), ids.data(),
                      n * sizeof(uint32_t));
    if (ret == 0)
      ret = order->write(begin * sizeof(uint32_t), ids.data(),
                         n * sizeof(uint32_t));
    if (ret != 0) return ret;
  }
  return order->truncate(order_offset);
}

}  // namespace

/*! Stratified Cluster
 */
class StratifiedCluster : public IndexCluster, public HolderCluster {
 public:
  //! Constructor
  StratifiedCluster() = default;

  //! Destructor
  ~StratifiedCluster() override = default;

  //! Initialize Cluster
  int init(const IndexMeta &meta, const ailego::Params &params) override {
    meta_ = meta;
    this->update_params(params);
    return 0;
  }

  //! Cleanup Cluster
  int cleanup() override {
    features_.reset();
    return 0;
  }

  //! Reset Cluster
  int reset() override {
    features_.reset();
    return 0;
  }

  //! Update Cluster
  int update(const ailego::Params &params) override {
    this->update_params(params);
    return 0;
  }

  //! Suggest dividing to K clusters
  void suggest(uint32_t k) override {
    cluster_count_ = k;
  }

  //! Mount features
  int mount(IndexFeatures::Pointer feats) override {
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
  int cluster(IndexThreads::Pointer threads,
              IndexCluster::CentroidList &cents) override;

  int cluster_holder(IndexThreads::Pointer threads, IndexHolder::Pointer holder,
                     IndexCluster::CentroidList &cents) override;

  //! Classify
  int classify(IndexThreads::Pointer threads,
               IndexCluster::CentroidList &cents) override;

  //! Label
  int label(IndexThreads::Pointer threads,
            const IndexCluster::CentroidList &cents,
            std::vector<uint32_t> *out) override;

 protected:
  //! Test if it is valid
  bool is_valid() const {
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

int StratifiedCluster::cluster_holder(IndexThreads::Pointer threads,
                                      IndexHolder::Pointer holder,
                                      IndexCluster::CentroidList &cents) try {
  if (!holder || !holder->count() || !meta_.element_size())
    return IndexError_InvalidArgument;
  if (!holder->is_matched(meta_) || !check_centroids(cents))
    return IndexError_Mismatch;
  // Keep the legacy path without a pool, and reject unsupported capabilities
  // before touching the input. A later I/O/budget error must not retry on heap.
  if (ailego::MemoryLimitPool::get_instance().capacity() == 0 ||
      holder->count() > std::numeric_limits<uint32_t>::max())
    return IndexError_NotImplemented;
  auto first = IndexFactory::CreateCluster(first_cluster_class_);
  auto second = IndexFactory::CreateCluster(second_cluster_class_);
  if (!first || !second) return IndexError_NoExist;
  auto *first_holder = dynamic_cast<HolderCluster *>(first.get());
  auto *second_holder = dynamic_cast<HolderCluster *>(second.get());
  if (!first_holder || !second_holder) return IndexError_NotImplemented;
  auto init_child = [&](const IndexCluster::Pointer &child,
                        ailego::Params params) {
    if (params.get_as_string(OPTKMEANS_CLUSTER_BUFFERED_STORAGE_PATH).empty()) {
      std::string prefix;
      int ret =
          TemporaryBufferStorage::DefaultPrefix("zvec-stratified", &prefix);
      if (ret != 0) return ret;
      params.set(OPTKMEANS_CLUSTER_BUFFERED_STORAGE_PATH, prefix);
    }
    return child->init(meta_, params);
  };
  int ret = init_child(first, first_cluster_params_);
  if (ret != 0) return ret;
  ret = init_child(second, second_cluster_params_);
  if (ret != 0) return ret;
  if (!threads)
    threads = std::make_shared<SingleQueueIndexThreads>(thread_count_, false);
  auto output =
      centroid_storage_ ? centroid_storage_ : PagedCentroids::Create();
  first_holder->set_centroid_storage(output);
  second_holder->set_centroid_storage(output);
  VectorBuildSource::Pointer source;
  ret = VectorBuildSource::Create(std::move(holder), &source);
  if (ret != 0) return ret;
  auto input = std::make_shared<StratifiedInput>(meta_, source, nullptr, 0,
                                                 source->count());
  CentroidList candidate = cents;
  if (first_cluster_count_) first->suggest(first_cluster_count_);
  ret = first_holder->cluster_holder(threads, input, candidate);
  // The source has already been consumed/spooled: do not signal safe fallback.
  if (ret != 0)
    return ret == IndexError_NotImplemented ? IndexError_Unsupported : ret;
  if (candidate.empty()) return IndexError_InvalidArgument;
  auto order = std::make_shared<PagedBuildBuffer>();
  ailego::BudgetedVector<size_t> offsets;
  ret = PartitionStratifiedInput(meta_, source, first, threads, candidate,
                                 order.get(), &offsets);
  if (ret != 0)
    return ret == IndexError_NotImplemented ? IndexError_Unsupported : ret;
  first.reset();
  const size_t total_clusters =
      cluster_count_
          ? cluster_count_
          : candidate.size() * static_cast<size_t>(second_cluster_count_);
  if (total_clusters > std::numeric_limits<uint32_t>::max())
    return IndexError_Overflow;
  // Reuse the caller's workers, with only one second-level training matrix
  // live at a time. Do not build per-bucket thread pools or mounted copies.
  for (size_t i = 0; i < candidate.size(); ++i) {
    auto &centroid = candidate[i];
    centroid.set_similars(std::vector<const void *>{});
    const size_t n = offsets[i + 1] - offsets[i];
    if (n == 0) continue;
    ret = second->reset();
    if (ret != 0) return ret;
    if (auto_tuning_) {
      if (total_clusters) {
        const double factor = static_cast<double>(n) / source->count();
        second->suggest(std::max(
            static_cast<uint32_t>(std::floor(total_clusters * factor)), 1u));
      }
    } else if (second_cluster_count_) {
      second->suggest(second_cluster_count_);
    }
    auto subset =
        std::make_shared<StratifiedInput>(meta_, source, order, offsets[i], n);
    ret = second_holder->cluster_holder(threads, std::move(subset),
                                        *centroid.mutable_subitems());
    if (ret != 0)
      return ret == IndexError_NotImplemented ? IndexError_Unsupported : ret;
  }
  cents = std::move(candidate);
  return 0;
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
} catch (const std::exception &) {
  return IndexError_Runtime;
}

int StratifiedCluster::cluster(IndexThreads::Pointer threads,
                               IndexCluster::CentroidList &cents) try {
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
  ailego::BudgetedVector<std::shared_ptr<IndexThreads>> threads_cluster;

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

  std::atomic<int> cluster_error{0};
  auto task_group = threads->make_group();
  AILEGO_DEFER([&] { task_group->wait_finish(); });
  // The second clustering
  for (size_t i = 0; i < cents.size(); ++i) {
    if (cents[i].similars().empty()) {
      continue;
    }

    IndexThreads::Pointer &curr_threads =
        threads_cluster[i % (threads_cluster.size())];

    task_group->submit(ailego::Closure::New(
        [this, curr_threads, total_cluster_count, &cents,
         &cluster_error](size_t index) {
          try {
            if (cluster_error.load() != 0) return;
            auto &it = cents[index];
            IndexCluster::Pointer second_cluster;
            std::shared_ptr<FlexibleIndexFeatures> features =
                std::make_shared<FlexibleIndexFeatures>(
                    meta_, it.similars().data(), it.similars().size());

            int ret = this->init_second_cluster(&second_cluster, features);
            if (ret != 0) {
              LOG_ERROR("Failed to initialize the second cluster.");
              int expected = 0;
              cluster_error.compare_exchange_strong(expected, ret);
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
            ret =
                second_cluster->cluster(curr_threads, *(it.mutable_subitems()));
            if (ret != 0) {
              LOG_ERROR("Failed to cluster in second cluster: %s.",
                        second_cluster_class_.c_str());
              int expected = 0;
              cluster_error.compare_exchange_strong(expected, ret);
            }
          } catch (const std::bad_alloc &) {
            int expected = 0;
            cluster_error.compare_exchange_strong(expected,
                                                  IndexError_NoMemory);
          } catch (const std::exception &) {
            int expected = 0;
            cluster_error.compare_exchange_strong(expected,
                                                  IndexError_ReadData);
          }
        },
        i));
  }
  task_group->wait_finish();
  return cluster_error.load();
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
}

int StratifiedCluster::classify(IndexThreads::Pointer threads,
                                IndexCluster::CentroidList &cents) try {
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
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
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
