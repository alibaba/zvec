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
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <ailego/algorithm/kmeans.h>
#include <ailego/container/reservoir.h>
#include <zvec/core/framework/index_cluster.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include "ailego/pattern/defer.h"
#include "utility/temporary_buffer_storage.h"
#include "cluster_params.h"
#include "holder_cluster.h"
#include "turbo_kmeans_context.h"

namespace zvec {
namespace core {

namespace {
class BufferedTrainingMatrix : public ailego::LloydClusterMatrixStorage {
 public:
  explicit BufferedTrainingMatrix(
      std::shared_ptr<TemporaryBufferStorage> storage, size_t base,
      size_t bytes)
      : storage_(std::move(storage)), base_(base), bytes_(bytes) {}

  int read(size_t offset, void *out, size_t bytes) const override {
    if (offset > bytes_ || bytes > bytes_ - offset)
      return IndexError_InvalidArgument;
    return storage_->read(base_ + offset, out, bytes);
  }

  int write(size_t offset, const void *data, size_t bytes) override {
    if (offset > bytes_ || bytes > bytes_ - offset)
      return IndexError_InvalidArgument;
    return storage_->write(base_ + offset, data, bytes);
  }

 private:
  std::shared_ptr<TemporaryBufferStorage> storage_;
  size_t base_;
  size_t bytes_;
};
}  // namespace

/*! Optimize K-Means cluster algorithm
 */
class OptKmeansAlgorithm : public IndexCluster {
 public:
  std::shared_ptr<PagedCentroids> output_storage;

 public:
  //! Constructor
  OptKmeansAlgorithm() = default;

  //! Destructor
  ~OptKmeansAlgorithm() override = default;

  //! Initialize Cluster
  int init(const IndexMeta &meta, const ailego::Params &params) override;

  //! Mount features
  int mount(IndexFeatures::Pointer feats) override;

  //! Suggest dividing to K clusters
  void suggest(uint32_t k) override;

  //! Classify
  int classify(IndexThreads::Pointer threads,
               IndexCluster::CentroidList &cents) override;

  //! Label
  int label(IndexThreads::Pointer threads,
            const IndexCluster::CentroidList &cents,
            std::vector<uint32_t> *out) override;

  //! Cluster
  int cluster(IndexThreads::Pointer threads,
              IndexCluster::CentroidList &cents) override {
    return cluster_impl(std::move(threads), nullptr, cents);
  }

  int cluster_holder(IndexThreads::Pointer threads, IndexHolder::Pointer holder,
                     IndexCluster::CentroidList &cents);

  //! Cleanup Cluster
  int cleanup() override;

  //! Reset Cluster
  int reset() override;

  //! Update Cluster
  int update(const ailego::Params &params) override;

 protected:
  virtual int cluster_impl(IndexThreads::Pointer threads,
                           IndexHolder::Pointer holder,
                           IndexCluster::CentroidList &cents) = 0;

  int check_dimension() const;
  // Preserve centroid-order tie handling, but read each paged centroid once
  // per small input tile rather than once per input vector. Scratch is bounded
  // independently of corpus and centroid counts.
  template <typename Consumer>
  void label_stored(size_t begin, size_t end,
                    const IndexCluster::CentroidList &cents,
                    const Consumer &consume) try {
    constexpr size_t kTile = 64;
    std::array<float, kTile> best{};
    std::array<uint32_t, kTile> ids{};
    std::vector<uint8_t, ailego::BudgetedAllocator<uint8_t>> row(
        meta_.element_size());
    for (size_t base = begin; base < end; base += kTile) {
      const size_t count = std::min(kTile, end - base);
      best.fill(std::numeric_limits<float>::max());
      ids.fill(0);
      for (size_t j = 0; j < cents.size(); ++j) {
        const int ret = cents[j].read_feature(row.data(), row.size());
        if (ret != 0) {
          read_error_.store(ret);
          return;
        }
        for (size_t i = 0; i < count; ++i) {
          float score = 0;
          distance_func_(row.data(), features_->element(base + i),
                         meta_.dimension(), &score);
          if (score < best[i]) {
            best[i] = score;
            ids[i] = static_cast<uint32_t>(j);
          }
        }
      }
      for (size_t i = 0; i < count; ++i) consume(base + i, ids[i]);
    }
  } catch (const std::bad_alloc &) {
    read_error_.store(IndexError_NoMemory);
  } catch (const std::exception &) {
    read_error_.store(IndexError_ReadData);
  }

  template <typename Algorithm>
  int export_centroids(IndexCluster::CentroidList &cents,
                       const Algorithm &algorithm) try {
    const size_t count = algorithm.centroids().count();
    const size_t bytes = meta_.element_size();
    auto storage = output_storage;
    if (!storage && ailego::MemoryLimitPool::get_instance().capacity() != 0)
      storage = PagedCentroids::Create();
    ailego::MemoryReservation directory;
    directory.add_array(count, IndexCluster::Centroid::loaded_metadata_bytes());
    IndexCluster::CentroidList candidate(count);
    for (size_t i = 0; i < count; ++i) {
      candidate[i].adopt_metadata(
          directory.take(IndexCluster::Centroid::loaded_metadata_bytes()));
      const void *row = algorithm.centroids()[i];
      if (algorithm.matrix_status() != 0)
        return algorithm.matrix_status() ==
                       std::decay_t<decltype(algorithm)>::kWorkspaceNoMemory
                   ? IndexError_NoMemory
                   : algorithm.matrix_status();
      if (!row) return IndexError_ReadData;
      int ret = 0;
      if (storage)
        ret = storage->append(row, bytes, &candidate[i]);
      else
        candidate[i].set_feature(row, bytes);
      if (ret != 0) return ret;
      candidate[i].set_score(algorithm.context().clusters()[i].cost());
      candidate[i].set_follows(algorithm.context().clusters()[i].count());
    }
    for (size_t i = 0; i < std::min(count, cents.size()); ++i)
      candidate[i].swap_assignments(cents[i]);
    cents.swap(candidate);
    return 0;
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  } catch (const std::exception &) {
    return IndexError_Runtime;
  }

  // Consume each input before advancing its iterator. The optional backing
  // stores exactly the same transposed matrix without retaining it on the heap.
  template <typename Algorithm>
  int load_features(Algorithm &algorithm, const IndexHolder::Pointer &holder) {
    using StoreType = typename Algorithm::StoreType;
    const size_t count = holder ? holder->count() : features_->count();
    const size_t bytes = meta_.element_size();
    if (bytes == 0 || count > std::numeric_limits<size_t>::max() / bytes) {
      return IndexError_InvalidArgument;
    }
    std::shared_ptr<TemporaryBufferStorage> storage;
    const size_t matrix_count =
        count / Algorithm::BatchCount * Algorithm::BatchCount;
    const size_t center_count =
        algorithm.k_value() / Algorithm::BatchCount * Algorithm::BatchCount;
    if (center_count >
        std::numeric_limits<size_t>::max() / bytes - matrix_count)
      return IndexError_Overflow;
    const size_t matrix_bytes = (matrix_count + center_count) * bytes;
    const size_t dimensions = meta_.dimension();
    if (dimensions > std::numeric_limits<size_t>::max() / sizeof(double))
      return IndexError_Overflow;
    const size_t accumulator_stride = dimensions * sizeof(double);
    if (algorithm.k_value() >
        (std::numeric_limits<size_t>::max() - matrix_bytes) /
            accumulator_stride)
      return IndexError_Overflow;
    const size_t accumulator_bytes = algorithm.k_value() * accumulator_stride;
    if (!buffered_storage_path_.empty() &&
        matrix_bytes + accumulator_bytes != 0) {
      int ret = TemporaryBufferStorage::Create(
          buffered_storage_path_, matrix_bytes + accumulator_bytes, &storage);
      if (ret != 0) return ret;
      algorithm.set_feature_matrix_storage(
          std::make_shared<BufferedTrainingMatrix>(storage, 0,
                                                   matrix_count * bytes));
      if (center_count != 0) {
        algorithm.set_centroid_matrix_storage(
            std::make_shared<BufferedTrainingMatrix>(
                storage, matrix_count * bytes, center_count * bytes));
      }
      algorithm.set_accumulator_storage(
          std::make_shared<BufferedTrainingMatrix>(storage, matrix_bytes,
                                                   accumulator_bytes));
    }
    ailego::BudgetedVector<StoreType> aligned;
    auto append = [&](const void *data) -> int {
      if (!data) {
        return IndexError_InvalidArgument;
      }
      // Providers may return packed or unaligned, reusable storage. At most one
      // row of scratch is needed; preserve the kernel's physical
      // representation.
      if (reinterpret_cast<uintptr_t>(data) % alignof(StoreType) != 0) {
        aligned.resize(bytes / sizeof(StoreType) +
                       (bytes % sizeof(StoreType) != 0));
        std::memcpy(aligned.data(), data, bytes);
        data = aligned.data();
      }
      algorithm.append(reinterpret_cast<const StoreType *>(data),
                       meta_.dimension());
      return algorithm.matrix_status();
    };

    algorithm.feature_matrix_reserve(count);
    if (holder) {
      auto iter = holder->create_iterator();
      if (!iter) {
        return IndexError_Runtime;
      }
      size_t loaded = 0;
      for (; iter->is_valid(); iter->next()) {
        if (loaded == count) {
          return IndexError_InvalidArgument;
        }
        const void *data = iter->data();
        if (iter->status() != 0) return iter->status();
        int ret = append(data);
        if (ret != 0) {
          return ret;
        }
        ++loaded;
      }
      if (iter->status() != 0) return iter->status();
      if (loaded != count) return IndexError_InvalidArgument;
    } else {
      for (size_t i = 0; i < count; ++i) {
        int ret = append(features_->element(i));
        if (ret != 0) {
          return ret;
        }
      }
    }
    return storage ? storage->flush() : 0;
  }

  //! Update parameters
  void update_params(const ailego::Params &params);

  //! Init Kmeans Algorithm
  int init_algorithm();

  //! Init Distance function
  int init_distance_func();

  //! Check Centroids
  bool check_centroids(const IndexCluster::CentroidList &cents);

  //! Test if it is valid
  bool is_valid() const;

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
  std::string buffered_storage_path_;
  std::atomic<int> read_error_{0};
  uint32_t markov_chain_length_{32};
  IndexMeta meta_{};
  IndexFeatures::Pointer features_{};
  ailego::BudgetedVector<ailego::BudgetedVector<const void *>>
      shard_cluster_features_{};
  IndexMetric::MatrixDistance distance_func_{nullptr};
};

bool OptKmeansAlgorithm::is_valid() const {
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
  params.get(OPTKMEANS_CLUSTER_BUFFERED_STORAGE_PATH, &buffered_storage_path_);
}

int OptKmeansAlgorithm::init_distance_func() {
  auto quantizer = CreateDistanceQuantizer(meta_);
  if (quantizer) {
    distance_func_ = [quantizer](const void *m, const void *q, size_t,
                                 float *out) {
      *out = quantizer->calc_distance_dp_dp(m, q);
    };
    return 0;
  }
  IndexMetric::Pointer metric{};
  metric = IndexFactory::CreateMetric(meta_.metric_name());
  if (!metric) {
    LOG_ERROR("Create Metric %s failed.", meta_.metric_name().c_str());
    return IndexError_Unsupported;
  }
  int ret = metric->init(meta_, meta_.metric_params());
  if (ret != 0) {
    LOG_ERROR("IndexMetric init failed wit ret %d.", ret);
    return ret;
  }
  distance_func_ = metric->distance_matrix(1, 1);
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
  AILEGO_DEFER([&] { task_group->wait_finish(); });

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
    const IndexCluster::CentroidList &cents) try {
  ailego::BudgetedVector<uint8_t> row;
  size_t thread_offset = threads->indexof_this() * cents.size();
  if (std::any_of(cents.begin(), cents.end(),
                  [](const auto &c) { return c.feature_stored(); })) {
    label_stored(index_begin, index_end, cents, [&](size_t i, uint32_t id) {
      shard_cluster_features_[thread_offset + id].push_back(
          features_->element(i));
    });
    return;
  }

  for (size_t i = index_begin; i != index_end; ++i) {
    const void *feat = features_->element(i);
    uint32_t sel_index = 0;
    float sel_score = std::numeric_limits<float>::max();

    // todo: get min distance
    uint32_t total = static_cast<uint32_t>(cents.size());
    for (uint32_t j = 0; j < total; ++j) {
      float score = 0.0f;

      const void *center = nullptr;
      if (cents[j].feature_stored()) {
        row.resize(meta_.element_size());
        const int ret = cents[j].read_feature(row.data(), row.size());
        if (ret != 0) {
          read_error_.store(ret);
          return;
        }
        center = row.data();
      } else
        center = cents[j].feature();
      distance_func_(center, feat, meta_.dimension(), &score);
      if (score < sel_score) {
        sel_score = score;
        sel_index = j;
      }
    }

    size_t sel_column = thread_offset + sel_index;
    shard_cluster_features_[sel_column].emplace_back(feat);
  }
} catch (const std::bad_alloc &) {
  read_error_.store(IndexError_NoMemory);
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
  AILEGO_DEFER([&] { task_group->wait_finish(); });
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
  AILEGO_DEFER([&] { task_group->wait_finish(); });

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
    const IndexCluster::CentroidList &cents) try {
  ailego::BudgetedVector<uint8_t> row;
  if (std::any_of(cents.begin(), cents.end(),
                  [](const auto &c) { return c.feature_stored(); })) {
    label_stored(index_begin, index_end, cents,
                 [&](size_t i, uint32_t id) { (*labels)[i] = id; });
    return;
  }
  for (size_t i = index_begin; i != index_end; ++i) {
    const void *feat = features_->element(i);

    uint32_t sel_index = 0;
    float sel_score = std::numeric_limits<float>::max();

    // todo: get min distance
    uint32_t total = static_cast<uint32_t>(cents.size());
    for (uint32_t j = 0; j < total; ++j) {
      float score = 0.0f;

      const void *center = nullptr;
      if (cents[j].feature_stored()) {
        row.resize(meta_.element_size());
        const int ret = cents[j].read_feature(row.data(), row.size());
        if (ret != 0) {
          read_error_.store(ret);
          return;
        }
        center = row.data();
      } else
        center = cents[j].feature();
      distance_func_(center, feat, meta_.dimension(), &score);
      if (score < sel_score) {
        sel_score = score;
        sel_index = j;
      }
    }

    (*labels)[i] = static_cast<uint32_t>(sel_index);
  }
} catch (const std::bad_alloc &) {
  read_error_.store(IndexError_NoMemory);
}

void OptKmeansAlgorithm::init_centroids(size_t count,
                                        IndexCluster::CentroidList *out) {
  // Just resize, because the get random centroid step is done by cluster_once
  out->resize(count);
}

void OptKmeansAlgorithm::update_features_thread(
    size_t column, IndexCluster::CentroidList *out) try {
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
  out->at(column).resize_similars(cluster_follows);

  for (size_t i = column, j = 0; i < shard_cluster_features_.size();
       i += cluster_count) {
    const auto &it = shard_cluster_features_[i];
    if (!it.empty()) {
      std::memcpy(&cluster_features[j], it.data(), it.size() * sizeof(void *));
      j += it.size();
    }
  }
} catch (const std::bad_alloc &) {
  read_error_.store(IndexError_NoMemory);
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
    if (tamp == 0) {
      IndexCluster::CentroidList().swap(cents);
    } else {
      // resize retains the vector capacity; keep its directory bill alive.
      for (size_t i = tamp; i < cents.size(); ++i)
        cents.front().absorb_metadata(cents[i]);
      cents.resize(tamp);
    }
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

  int ret = check_dimension();
  if (ret != 0) {
    return ret;
  }
  features_ = std::move(feats);
  return 0;
}

int OptKmeansAlgorithm::check_dimension() const {
  auto type = meta_.data_type();
  switch (type) {
    case IndexMeta::DataType::DT_INT4:
      if (meta_.dimension() % 8 != 0) {
        LOG_ERROR(
            "Unsupported feature dimension %zu (dimension of int4 "
            "must be an integer multiple of 8).",
            static_cast<size_t>(meta_.dimension()));
        return IndexError_Mismatch;
      }
      break;
    case IndexMeta::DataType::DT_INT8:
      if (meta_.dimension() % 4 != 0) {
        LOG_ERROR(
            "Unsupported feature dimension %zu (dimension of int8 "
            "must be an integer multiple of 4).",
            static_cast<size_t>(meta_.dimension()));
        return IndexError_Mismatch;
      }
      break;
    default:
      break;
  }

  return 0;
}

int OptKmeansAlgorithm::cluster_holder(IndexThreads::Pointer threads,
                                       IndexHolder::Pointer holder,
                                       IndexCluster::CentroidList &cents) {
  if (!holder || !holder->count() || !meta_.dimension()) {
    return IndexError_InvalidArgument;
  }
  if (!holder->is_matched(meta_)) {
    return IndexError_Mismatch;
  }
  int ret = check_dimension();
  if (ret != 0) {
    return ret;
  }
  // Unknown-size or unrepresentable inputs must fall back before consumption.
  if (holder->count() == static_cast<size_t>(-1) ||
      holder->count() > std::numeric_limits<uint32_t>::max()) {
    return IndexError_NotImplemented;
  }
  if (holder->count() >
      std::numeric_limits<size_t>::max() / meta_.element_size()) {
    return IndexError_InvalidArgument;
  }
  return cluster_impl(std::move(threads), std::move(holder), cents);
}

void OptKmeansAlgorithm::suggest(uint32_t k) {
  cluster_count_ = k;
}

int OptKmeansAlgorithm::classify(IndexThreads::Pointer threads,
                                 IndexCluster::CentroidList &cents) try {
  auto release = ailego::ScopeGuard::Make([&] {
    decltype(shard_cluster_features_)().swap(shard_cluster_features_);
  });
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

  read_error_.store(0);
  this->update_clusters(threads.get(), cents);
  if (read_error_.load() != 0) return read_error_.load();
  this->update_features(threads.get(), cents);
  return read_error_.load();
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
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

  read_error_.store(0);
  this->update_labels(threads.get(), out, cents);
  return read_error_.load();
}

int OptKmeansAlgorithm::update(const ailego::Params &params) {
  this->update_params(params);
  // algorithm_->reset(cluster_count_);
  return 0;
}

int OptKmeansAlgorithm::reset() {
  features_.reset();
  shard_cluster_features_.clear();

  return 0;
}

int OptKmeansAlgorithm::cleanup() {
  features_.reset();
  shard_cluster_features_.clear();

  return 0;
}


/*! Numerical K-Means cluster algorithm
 */
template <typename T, typename Context = ailego::NumericalKmeansContext<T>>
class NumericalKmeansAlgorithm : public OptKmeansAlgorithm {
 public:
  //! Type of value
  using ValueType = typename std::remove_cv<T>::type;

  // Check supporting type
  static_assert(ailego::IsArithmetic<ValueType>::value,
                "ValueType must be arithmetic");

  //! Constructor
  NumericalKmeansAlgorithm() = default;

  //! Destructor
  ~NumericalKmeansAlgorithm() override = default;

 protected:
  int cluster_impl(IndexThreads::Pointer threads, IndexHolder::Pointer holder,
                   IndexCluster::CentroidList &cents) override;

  int update_centroids(
      IndexCluster::CentroidList &cents,
      const ailego::NumericalKmeans<T, IndexThreads, Context> &algorithm);
};

template <typename T, typename Context>
int NumericalKmeansAlgorithm<T, Context>::update_centroids(
    IndexCluster::CentroidList &cents,
    const ailego::NumericalKmeans<T, IndexThreads, Context> &algorithm) {
  return export_centroids(cents, algorithm);
}

template <typename T, typename Context>
int NumericalKmeansAlgorithm<T, Context>::cluster_impl(
    IndexThreads::Pointer threads, IndexHolder::Pointer holder,
    IndexCluster::CentroidList &cents) try {
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

  if (!holder && !this->is_valid()) {
    LOG_ERROR("The cluster is not ready.");
    return IndexError_NoReady;
  }

  // get cluster algorithm
  const size_t features_count = holder ? holder->count() : features_->count();
  size_t centroid_count =
      cents.empty()
          ? std::min(cluster_count_, static_cast<uint32_t>(features_count))
          : cents.size();
  if (centroid_count == 0) {
    LOG_ERROR("The count of cluster is unknown.");
    return IndexError_NoReady;
  }
  ailego::NumericalKmeans<T, IndexThreads, Context> algorithm(
      centroid_count, meta_.dimension());

  // Mount directly into the algorithm's matrix.
  int ret = load_features(algorithm, holder);
  if (ret != 0) {
    return ret;
  }
  holder.reset();

  if (!cents.empty()) {
    auto centroids = algorithm.mutable_centroids();
    centroids->reserve(cents.size());
    ailego::BudgetedVector<uint64_t> seed((meta_.element_size() + 7) / 8);
    for (const auto &it : cents) {
      ret = it.read_feature(seed.data(), meta_.element_size());
      if (ret != 0) return ret;
      centroids->append(reinterpret_cast<const T *>(seed.data()),
                        meta_.dimension());
    }
  } else {
    ailego::Kmc2CentroidsGenerator<
        ailego::NumericalKmeans<ValueType, IndexThreads, Context>, IndexThreads>
        g;
    g.set_chain_length(markov_chain_length_);
    g.set_assumption_free(assumption_free_);
    algorithm.init_centroids(*threads, g);
  }

  if (algorithm.matrix_status() != 0)
    return algorithm.matrix_status() ==
                   std::decay_t<decltype(algorithm)>::kWorkspaceNoMemory
               ? IndexError_NoMemory
               : algorithm.matrix_status();

  double cost = 0.0;

  for (uint32_t i = 0; i < max_iterations_; ++i) {
    double old_cost, new_epsilon;
    old_cost = cost;

    bool result = algorithm.cluster_once(*threads, &cost);
    if (result != true) {
      LOG_ERROR("(%u) Failed to cluster.", i + 1);
      return algorithm.matrix_status() ==
                     std::decay_t<decltype(algorithm)>::kWorkspaceNoMemory
                 ? IndexError_NoMemory
                 : (algorithm.matrix_status() != 0 ? algorithm.matrix_status()
                                                   : -1);
    }

    new_epsilon = std::abs(cost - old_cost);
    LOG_DEBUG("(%u) Updated %zu Clusters, %zu Features: %zu ms, %f -> %f = %f",
              i, algorithm.centroids().count(), features_count,
              (size_t)stamp.milli_seconds(), old_cost, cost, new_epsilon);
    stamp.reset();

    if (new_epsilon < epsilon_) {
      break;
    }
  }

  // update_centroids(cents);
  ret = update_centroids(cents, algorithm);
  if (ret != 0) return ret;

  // Purge the empty centroids
  PurgeCentroids(cents, purge_empty_);
  return 0;
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
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
  NibbleKmeansAlgorithm() = default;

  //! Destructor
  ~NibbleKmeansAlgorithm() override = default;

 protected:
  int cluster_impl(IndexThreads::Pointer threads, IndexHolder::Pointer holder,
                   IndexCluster::CentroidList &cents) override;

  //! update centroids
  int update_centroids(IndexCluster::CentroidList &cents,
                       const ailego::NibbleKmeans<T, IndexThreads> &algorithm);
};

template <typename T>
int NibbleKmeansAlgorithm<T>::update_centroids(
    IndexCluster::CentroidList &cents,
    const ailego::NibbleKmeans<T, IndexThreads> &algorithm) {
  return export_centroids(cents, algorithm);
}

template <typename T>
int NibbleKmeansAlgorithm<T>::cluster_impl(
    IndexThreads::Pointer threads, IndexHolder::Pointer holder,
    IndexCluster::CentroidList &cents) try {
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

  if (!holder && !this->is_valid()) {
    LOG_ERROR("The cluster is not ready.");
    return IndexError_NoReady;
  }

  // get cluster algorithm
  const size_t features_count = holder ? holder->count() : features_->count();
  size_t centroid_count =
      cents.empty()
          ? std::min(cluster_count_, static_cast<uint32_t>(features_count))
          : cents.size();
  if (centroid_count == 0) {
    LOG_ERROR("The count of cluster is unknown.");
    return IndexError_NoReady;
  }
  ailego::NibbleKmeans<T, IndexThreads> algorithm(centroid_count,
                                                  meta_.dimension());

  int ret = load_features(algorithm, holder);
  if (ret != 0) {
    return ret;
  }
  holder.reset();

  if (!cents.empty()) {
    auto centroids = algorithm.mutable_centroids();
    centroids->reserve(cents.size());
    ailego::BudgetedVector<uint64_t> seed((meta_.element_size() + 7) / 8);
    for (const auto &it : cents) {
      ret = it.read_feature(seed.data(), meta_.element_size());
      if (ret != 0) return ret;
      centroids->append(
          reinterpret_cast<const typename std::make_unsigned<T>::type *>(
              seed.data()),
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

  if (algorithm.matrix_status() != 0)
    return algorithm.matrix_status() ==
                   std::decay_t<decltype(algorithm)>::kWorkspaceNoMemory
               ? IndexError_NoMemory
               : algorithm.matrix_status();

  double cost = 0.0;

  for (uint32_t i = 0; i < max_iterations_; ++i) {
    double old_cost, new_epsilon;
    old_cost = cost;

    bool result = algorithm.cluster_once(*threads, &cost);
    if (result != true) {
      LOG_ERROR("(%u) Failed to cluster.", i + 1);
      return algorithm.matrix_status() ==
                     std::decay_t<decltype(algorithm)>::kWorkspaceNoMemory
                 ? IndexError_NoMemory
                 : (algorithm.matrix_status() != 0 ? algorithm.matrix_status()
                                                   : -1);
    }

    new_epsilon = std::abs(cost - old_cost);
    LOG_DEBUG(
        "(%u) Updated %zu Clusters, %zu Features: %zu ms, %f -> "
        "%f = %f",
        i, algorithm.centroids().count(), features_count,
        (size_t)stamp.milli_seconds(), old_cost, cost, new_epsilon);
    stamp.reset();

    if (new_epsilon < epsilon_) {
      break;
    }
  }

  // update centroids
  ret = update_centroids(cents, algorithm);
  if (ret != 0) return ret;

  // Purge the empty centroids
  PurgeCentroids(cents, purge_empty_);
  return 0;
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
}

/*! Numerical K-Means cluster algorithm
 */
template <typename T,
          typename Context = ailego::NumericalInnerProductKmeansContext<T>>
class NumericalInnerProductKmeansAlgorithm : public OptKmeansAlgorithm {
 public:
  //! Type of value
  using ValueType = typename std::remove_cv<T>::type;

  // Check supporting type
  static_assert(ailego::IsArithmetic<ValueType>::value,
                "ValueType must be arithmetic");

  //! Constructor
  NumericalInnerProductKmeansAlgorithm() = default;

  //! Destructor
  ~NumericalInnerProductKmeansAlgorithm() override = default;

 protected:
  int cluster_impl(IndexThreads::Pointer threads, IndexHolder::Pointer holder,
                   IndexCluster::CentroidList &cents) override;

  int update_centroids(
      IndexCluster::CentroidList &cents,
      const ailego::NumericalInnerProductKmeans<T, IndexThreads, Context>
          &algorithm);
};

template <typename T, typename Context>
int NumericalInnerProductKmeansAlgorithm<T, Context>::update_centroids(
    IndexCluster::CentroidList &cents,
    const ailego::NumericalInnerProductKmeans<T, IndexThreads, Context>
        &algorithm) {
  return export_centroids(cents, algorithm);
}

template <typename T, typename Context>
int NumericalInnerProductKmeansAlgorithm<T, Context>::cluster_impl(
    IndexThreads::Pointer threads, IndexHolder::Pointer holder,
    IndexCluster::CentroidList &cents) try {
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

  if (!holder && !this->is_valid()) {
    LOG_ERROR("The cluster is not ready.");
    return IndexError_NoReady;
  }

  // get cluster algorithm
  const size_t features_count = holder ? holder->count() : features_->count();
  size_t centroid_count =
      cents.empty()
          ? std::min(cluster_count_, static_cast<uint32_t>(features_count))
          : cents.size();
  if (centroid_count == 0) {
    LOG_ERROR("The count of cluster is unknown.");
    return IndexError_NoReady;
  }
  ailego::NumericalInnerProductKmeans<T, IndexThreads, Context> algorithm(
      centroid_count, meta_.dimension(), true);

  int ret = load_features(algorithm, holder);
  if (ret != 0) {
    return ret;
  }
  holder.reset();

  if (!cents.empty()) {
    auto centroids = algorithm.mutable_centroids();
    centroids->reserve(cents.size());
    ailego::BudgetedVector<uint64_t> seed((meta_.element_size() + 7) / 8);
    for (const auto &it : cents) {
      ret = it.read_feature(seed.data(), meta_.element_size());
      if (ret != 0) return ret;
      centroids->append(reinterpret_cast<const T *>(seed.data()),
                        meta_.dimension());
    }
  } else {
    ailego::Kmc2CentroidsGenerator<
        ailego::NumericalInnerProductKmeans<ValueType, IndexThreads, Context>,
        IndexThreads>
        g;
    g.set_chain_length(markov_chain_length_);
    g.set_assumption_free(assumption_free_);
    algorithm.init_centroids(*threads, g);
  }

  if (algorithm.matrix_status() != 0)
    return algorithm.matrix_status() ==
                   std::decay_t<decltype(algorithm)>::kWorkspaceNoMemory
               ? IndexError_NoMemory
               : algorithm.matrix_status();

  double cost = 0.0;

  for (uint32_t i = 0; i < max_iterations_; ++i) {
    double old_cost, new_epsilon;
    old_cost = cost;

    bool result = algorithm.cluster_once(*threads, &cost);
    if (result != true) {
      LOG_ERROR("(%u) Failed to cluster.", i + 1);
      return algorithm.matrix_status() ==
                     std::decay_t<decltype(algorithm)>::kWorkspaceNoMemory
                 ? IndexError_NoMemory
                 : (algorithm.matrix_status() != 0 ? algorithm.matrix_status()
                                                   : -1);
    }

    new_epsilon = std::abs(cost - old_cost);
    LOG_DEBUG("(%u) Updated %zu Clusters, %zu Features: %zu ms, %f -> %f = %f",
              i, algorithm.centroids().count(), features_count,
              (size_t)stamp.milli_seconds(), old_cost, cost, new_epsilon);
    stamp.reset();

    if (new_epsilon < epsilon_) {
      break;
    }
  }

  // update_centroids(cents);
  ret = update_centroids(cents, algorithm);
  if (ret != 0) return ret;

  // Purge the empty centroids
  PurgeCentroids(cents, purge_empty_);
  return 0;
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
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
  NibbleInnerProductKmeansAlgorithm() = default;

  //! Destructor
  ~NibbleInnerProductKmeansAlgorithm() override = default;

 protected:
  int cluster_impl(IndexThreads::Pointer threads, IndexHolder::Pointer holder,
                   IndexCluster::CentroidList &cents) override;

  //! update centroids
  int update_centroids(
      IndexCluster::CentroidList &cents,
      const ailego::NibbleInnerProductKmeans<T, IndexThreads> &algorithm);
};

template <typename T>
int NibbleInnerProductKmeansAlgorithm<T>::update_centroids(
    IndexCluster::CentroidList &cents,
    const ailego::NibbleInnerProductKmeans<T, IndexThreads> &algorithm) {
  return export_centroids(cents, algorithm);
}

template <typename T>
int NibbleInnerProductKmeansAlgorithm<T>::cluster_impl(
    IndexThreads::Pointer threads, IndexHolder::Pointer holder,
    IndexCluster::CentroidList &cents) try {
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

  if (!holder && !this->is_valid()) {
    LOG_ERROR("The cluster is not ready.");
    return IndexError_NoReady;
  }

  // get cluster algorithm
  const size_t features_count = holder ? holder->count() : features_->count();
  size_t centroid_count =
      cents.empty()
          ? std::min(cluster_count_, static_cast<uint32_t>(features_count))
          : cents.size();
  if (centroid_count == 0) {
    LOG_ERROR("The count of cluster is unknown.");
    return IndexError_NoReady;
  }
  ailego::NibbleInnerProductKmeans<T, IndexThreads> algorithm(
      centroid_count, meta_.dimension());

  int ret = load_features(algorithm, holder);
  if (ret != 0) {
    return ret;
  }
  holder.reset();

  if (!cents.empty()) {
    auto centroids = algorithm.mutable_centroids();
    centroids->reserve(cents.size());
    ailego::BudgetedVector<uint64_t> seed((meta_.element_size() + 7) / 8);
    for (const auto &it : cents) {
      ret = it.read_feature(seed.data(), meta_.element_size());
      if (ret != 0) return ret;
      centroids->append(
          reinterpret_cast<const typename std::make_unsigned<T>::type *>(
              seed.data()),
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

  if (algorithm.matrix_status() != 0)
    return algorithm.matrix_status() ==
                   std::decay_t<decltype(algorithm)>::kWorkspaceNoMemory
               ? IndexError_NoMemory
               : algorithm.matrix_status();

  double cost = 0.0;

  for (uint32_t i = 0; i < max_iterations_; ++i) {
    double old_cost, new_epsilon;
    old_cost = cost;

    bool result = algorithm.cluster_once(*threads, &cost);
    if (result != true) {
      LOG_ERROR("(%u) Failed to cluster.", i + 1);
      return algorithm.matrix_status() ==
                     std::decay_t<decltype(algorithm)>::kWorkspaceNoMemory
                 ? IndexError_NoMemory
                 : (algorithm.matrix_status() != 0 ? algorithm.matrix_status()
                                                   : -1);
    }

    new_epsilon = std::abs(cost - old_cost);
    LOG_DEBUG(
        "(%u) Updated %zu Clusters, %zu Features: %zu ms, %f -> "
        "%f = %f",
        i, algorithm.centroids().count(), features_count,
        (size_t)stamp.milli_seconds(), old_cost, cost, new_epsilon);
    stamp.reset();

    if (new_epsilon < epsilon_) {
      break;
    }
  }

  // update centroids
  ret = update_centroids(cents, algorithm);
  if (ret != 0) return ret;

  // Purge the empty centroids
  PurgeCentroids(cents, purge_empty_);
  return 0;
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
}

/*! Kmeans Cluster
 */
class OptKmeansCluster : public IndexCluster, public HolderCluster {
 public:
  //! Constructor
  OptKmeansCluster() = default;

  //! Destructor
  ~OptKmeansCluster() override = default;

  //! Initialize Cluster
  int init(const IndexMeta &meta, const ailego::Params &params) override;

  //! Cleanup Cluster
  int cleanup() override;

  //! Reset Cluster
  int reset() override;

  //! Update Cluster
  int update(const ailego::Params &params) override;

  //! Suggest dividing to K clusters
  void suggest(uint32_t k) override;

  //! Mount features
  int mount(IndexFeatures::Pointer feats) override;

  //! Cluster
  int cluster(IndexThreads::Pointer threads,
              IndexCluster::CentroidList &cents) override;

  int cluster_holder(IndexThreads::Pointer threads, IndexHolder::Pointer holder,
                     IndexCluster::CentroidList &cents) override {
    algorithm_->output_storage = centroid_storage_;
    return algorithm_->cluster_holder(std::move(threads), std::move(holder),
                                      cents);
  }

  //! Classify
  int classify(IndexThreads::Pointer threads,
               IndexCluster::CentroidList &cents) override;

  //! Label
  int label(IndexThreads::Pointer threads,
            const IndexCluster::CentroidList &cents,
            std::vector<uint32_t> *out) override;

 protected:
  //! Members
  std::shared_ptr<OptKmeansAlgorithm> algorithm_{};
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
int OptKmeansCluster::reset() {
  return algorithm_->reset();
}

//! Cleanup Cluster
int OptKmeansCluster::cleanup() {
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
  auto type = meta.data_type();

  const bool inner_product = meta.metric_name() == "InnerProduct";
  if ((inner_product || meta.metric_name() == "SquaredEuclidean") &&
      (type == IndexMeta::DataType::DT_FP32 ||
       type == IndexMeta::DataType::DT_FP16)) {
    if (type == IndexMeta::DataType::DT_FP32) {
      if (inner_product) {
        algorithm_ = std::make_shared<NumericalInnerProductKmeansAlgorithm<
            float, TurboKmeansContext<float, true>>>();
      } else {
        algorithm_ = std::make_shared<
            NumericalKmeansAlgorithm<float, TurboKmeansContext<float>>>();
      }
    } else {
      if (inner_product) {
        algorithm_ = std::make_shared<NumericalInnerProductKmeansAlgorithm<
            ailego::Float16, TurboKmeansContext<ailego::Float16, true>>>();
      } else {
        algorithm_ = std::make_shared<NumericalKmeansAlgorithm<
            ailego::Float16, TurboKmeansContext<ailego::Float16>>>();
      }
    }
    return algorithm_->init(meta, params);
  }

  if (meta.metric_name() == "InnerProduct") {
    switch (type) {
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
        LOG_ERROR("Unsupported feature types %d.", type);
        return IndexError_Mismatch;
      }
    }
  } else {
    switch (type) {
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
      default: {
        LOG_ERROR("Unsupported feature types %d.", type);
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
