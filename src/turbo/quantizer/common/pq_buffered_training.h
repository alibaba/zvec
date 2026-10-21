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
#include <atomic>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <type_traits>
#include <vector>
#include <ailego/algorithm/kmeans.h>
#include <ailego/math/normalizer.h>
#include <zvec/core/framework/index_threads.h>
#include "core/utility/temporary_buffer_storage.h"
#include "pq_quantizer/pq_opq.h"
#include "turbo/training_workspace.h"
#include "pq_training_samples.h"

namespace zvec::turbo {

// One path for all PQ code widths: sample before copying, then keep the input,
// subspaces and transposed Lloyd matrices in one shared-pool-backed file.
struct PqBufferedTrainingOptions {
  const char *name;
  size_t dimension;
  std::vector<uint32_t> chunk_offsets;
  size_t centroid_count;
  size_t sample_limit;
  uint32_t max_iters;
  uint32_t thread_count;
  uint32_t chain_length;
  double epsilon;
  bool normalize;
  bool zero_mean;
};

class PqMatrixSlice final : public ailego::LloydClusterMatrixStorage {
 public:
  PqMatrixSlice(core::TemporaryBufferStorage::Pointer file, size_t base,
                size_t size)
      : file_(std::move(file)), base_(base), size_(size) {}
  int read(size_t offset, void *out, size_t bytes) const override {
    if (offset > size_ || bytes > size_ - offset)
      return core::IndexError_InvalidArgument;
    return file_->read(base_ + offset, out, bytes);
  }
  int write(size_t offset, const void *data, size_t bytes) override {
    if (offset > size_ || bytes > size_ - offset)
      return core::IndexError_InvalidArgument;
    return file_->write(base_ + offset, data, bytes);
  }

 private:
  core::TemporaryBufferStorage::Pointer file_;
  size_t base_;
  size_t size_;
};

template <class T, class Centroids, class Center>
int TrainPqBuffered(const core::IndexHolder::Pointer &holder, size_t num,
                    const PqBufferedTrainingOptions &options,
                    Centroids &centroids, Center &center, PqOpq &opq,
                    const std::function<void()> &refresh_cache,
                    const PqOpq::ReconstructSum &reconstruct) try {
  const auto &offsets = options.chunk_offsets;
  if (!num || offsets.size() < 2 || offsets.front() != 0 ||
      offsets.back() != options.dimension || !options.centroid_count ||
      num > options.sample_limit)
    return core::IndexError_InvalidArgument;
  for (size_t m = 1; m < offsets.size(); ++m)
    if (offsets[m] <= offsets[m - 1]) return core::IndexError_InvalidArgument;
  auto chunk_dim = [&](size_t m) { return offsets[m + 1] - offsets[m]; };
  const size_t num_chunks = offsets.size() - 1;
  const size_t row_bytes = options.dimension * sizeof(T);
  if (num > std::numeric_limits<size_t>::max() / row_bytes / 3)
    return core::IndexError_Overflow;
  const size_t samples_bytes = num * row_bytes;
  std::string prefix;
  int ret =
      core::TemporaryBufferStorage::DefaultPrefix("zvec-pq-train", &prefix);
  if (ret != 0) return ret;
  core::TemporaryBufferStorage::Pointer file;
  // Three non-overlapping regions: selected rows, contiguous subspaces, and
  // transposed Lloyd matrices. Only one shared pool-backed file is needed.
  ret = core::TemporaryBufferStorage::Create(prefix, 3 * samples_bytes, &file);
  if (ret != 0) return ret;
  int write_error = 0;
  if (!CollectPqTrainingSamplesTo(holder, options.sample_limit, row_bytes,
                                  [&](size_t slot, const void *row) {
                                    write_error = file->write(slot * row_bytes,
                                                              row, row_bytes);
                                    return write_error == 0;
                                  }))
    return write_error != 0 ? write_error : core::IndexError_ReadData;

  using Workspace = std::vector<T, ailego::BudgetedAllocator<T>>;
  Workspace row(options.dimension);
  if (options.zero_mean) center.assign(options.dimension, 0.0f);
  for (size_t i = 0; i < num; ++i) {
    ret = file->read(i * row_bytes, row.data(), row_bytes);
    if (ret != 0) return ret;
    if (options.normalize) {
      float norm = 0.0f;
      ailego::Normalizer<T>::L2(row.data(), options.dimension, &norm);
      ret = file->write(i * row_bytes, row.data(), row_bytes);
      if (ret != 0) return ret;
    }
    // Sum in precisely the selected-sample order, as on the heap path.
    if (options.zero_mean) {
      for (size_t d = 0; d < options.dimension; ++d)
        center[d] += static_cast<float>(row[d]);
    }
  }
  if (options.zero_mean) {
    for (auto &value : center) value /= static_cast<float>(num);
  }
  const size_t tile_rows = TrainingTileRows(num, row_bytes, 2);
  Workspace tile(std::min(num, tile_rows) * options.dimension);
  Workspace subspace(tile.size());
  auto write_subspaces = [&](size_t first, size_t count, const T *rows) -> int {
    subspace.resize(count * options.dimension);
    for (size_t m = 0; m < num_chunks; ++m) {
      const size_t offset = offsets[m];
      const size_t width = chunk_dim(m);
      for (size_t i = 0; i < count; ++i)
        std::memcpy(subspace.data() + i * width,
                    rows + i * options.dimension + offset, width * sizeof(T));
      int status = file->write(
          samples_bytes + (offset * num + first * width) * sizeof(T),
          subspace.data(), count * width * sizeof(T));
      if (status != 0) return status;
    }
    return 0;
  };
  for (size_t first = 0; first < num; first += tile_rows) {
    const size_t count = std::min(tile_rows, num - first);
    ret = file->read(first * row_bytes, tile.data(), count * row_bytes);
    if (ret != 0) return ret;
    if (options.zero_mean) {
      for (size_t i = 0; i < count; ++i)
        for (size_t d = 0; d < options.dimension; ++d)
          tile[i * options.dimension + d] -= center[d];
      ret = file->write(first * row_bytes, tile.data(), count * row_bytes);
      if (ret != 0) return ret;
    }
    ret = write_subspaces(first, count, tile.data());
    if (ret != 0) return ret;
  }
  Workspace().swap(tile);
  Workspace().swap(subspace);
  Workspace().swap(row);

  auto threads = std::make_shared<core::SingleQueueIndexThreads>(
      options.thread_count, false);
  auto train_chunks = [&](uint32_t max_iters) -> int {
    auto group = threads->make_group();
    if (!group) return core::IndexError_Runtime;
    std::atomic<int> error{0};
    auto train_one = [&](size_t m) -> int {
      const size_t width = chunk_dim(m);
      const size_t offset = offsets[m] * num * sizeof(T);
      const size_t bytes = num * width * sizeof(T);
      ailego::NumericalKmeans<T, core::SingleQueueIndexThreads> algorithm(
          options.centroid_count, width);
      algorithm.set_feature_matrix_storage(std::make_shared<PqMatrixSlice>(
          file, 2 * samples_bytes + offset, bytes));
      const size_t batch =
          TrainingTileRows(num, width * sizeof(T), threads->count(), 64 * 1024);
      Workspace rows(std::min(num, batch) * width);
      for (size_t first = 0; first < num; first += batch) {
        const size_t count = std::min(batch, num - first);
        int status =
            file->read(samples_bytes + offset + first * width * sizeof(T),
                       rows.data(), count * width * sizeof(T));
        if (status != 0) return status;
        for (size_t i = 0; i < count; ++i)
          algorithm.append(rows.data() + i * width, width);
        if (algorithm.matrix_status() != 0) return algorithm.matrix_status();
      }
      core::SingleQueueIndexThreads local_threads(1, false);
      ailego::Kmc2CentroidsGenerator<decltype(algorithm),
                                     core::SingleQueueIndexThreads>
          gen;
      gen.set_chain_length(options.chain_length);
      gen.set_assumption_free(false);
      algorithm.init_centroids(local_threads, gen);
      double cost = 0.0;
      for (uint32_t iter = 0; iter < max_iters; ++iter) {
        const double old_cost = cost;
        if (!algorithm.cluster_once(local_threads, &cost)) break;
        if (std::abs(cost - old_cost) < options.epsilon) break;
      }
      if (algorithm.matrix_status() != 0) return algorithm.matrix_status();
      const auto &cents = algorithm.centroids();
      if (cents.count() != options.centroid_count)
        return core::IndexError_Runtime;
      auto *out =
          centroids.data() + offsets[m] * options.centroid_count * sizeof(T);
      for (size_t c = 0; c < cents.count(); ++c)
        std::memcpy(out + c * width * sizeof(T), cents[c], width * sizeof(T));
      return 0;
    };
    for (size_t worker = 0; worker < threads->count(); ++worker) {
      group->submit(ailego::Closure::New([&, worker]() {
        try {
          for (size_t m = worker; m < num_chunks && error.load() == 0;
               m += threads->count()) {
            int status = train_one(m);
            if (status != 0) {
              int expected = 0;
              error.compare_exchange_strong(expected, status);
            }
          }
        } catch (const std::bad_alloc &) {
          error.store(core::IndexError_NoMemory);
        } catch (const std::exception &) {
          error.store(core::IndexError_Runtime);
        }
      }));
    }
    group->wait_finish();
    return error.load();
  };
  if (opq.enabled()) {
    if constexpr (std::is_same<T, float>::value) {
      ret = opq.train_rows(
          num, options.name,
          [&](size_t first, size_t count, float *rows) {
            return file->read(first * row_bytes, rows, count * row_bytes);
          },
          write_subspaces,
          [&](uint32_t max_iters) {
            int status = train_chunks(max_iters);
            if (status == 0) refresh_cache();
            return status;
          },
          reconstruct);
      if (ret != 0) return ret;
    } else {
      return core::IndexError_InvalidArgument;
    }
  }
  return train_chunks(options.max_iters);
} catch (const std::bad_alloc &) {
  return core::IndexError_NoMemory;
} catch (const std::exception &) {
  return core::IndexError_Runtime;
}

}  // namespace zvec::turbo
