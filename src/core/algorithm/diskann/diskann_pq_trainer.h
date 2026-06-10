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

#include <zvec/core/framework/index_framework.h>
#include "diskann_entity.h"
#include "diskann_pq_table.h"
#include "../cluster/multi_chunk_cluster.h"

namespace zvec {
namespace core {

class DiskAnnPqTrainer {
 public:
  typedef std::unique_ptr<DiskAnnPqTrainer> UPointer;

 public:
  DiskAnnPqTrainer(uint32_t max_train_sample_count);
  virtual ~DiskAnnPqTrainer();

 public:
  template <typename T>
  static int prepare_pq_train_data(
      const IndexMeta &meta, size_t num_train, std::string &train_data,
      bool use_zero_mean, std::vector<uint8_t> &centroid,
      std::shared_ptr<CompactIndexFeatures> &train_features);

  template <typename T>
  static int convert_pivot_data(const IndexMeta &meta, uint32_t num_centers,
                                uint32_t pq_chunk_num,
                                const std::vector<uint32_t> &chunk_dims,
                                const std::vector<uint32_t> &chunk_offsets,
                                IndexCluster::CentroidList &centroids,
                                std::vector<uint8_t> &full_pivot_data);

  int gen_random_sample(IndexHolder::Pointer holder, const IndexMeta &meta,
                        std::string &sample_data, size_t &sample_size);

  int generate_quantized_data(IndexThreads::Pointer threads,
                              IndexHolder::Pointer holder,
                              const IndexMeta &meta,
                              std::vector<uint8_t> &pq_centroid,
                              std::vector<uint8_t> &block_compressed_data,
                              size_t num_pq_chunks);

  int generate_pq(IndexThreads::Pointer threads, const IndexMeta &meta,
                  IndexHolder::Pointer holder, uint32_t num_pq_chunks,
                  std::vector<uint8_t> &centroid,
                  std::vector<uint8_t> &block_compressed_data);

  int train_quantized_data(IndexThreads::Pointer threads,
                           IndexHolder::Pointer holder, const IndexMeta &meta,
                           std::vector<uint8_t> &pq_full_pivot_data,
                           std::vector<uint8_t> &pq_centroid,
                           std::vector<uint32_t> &pq_chunk_offsets,
                           size_t num_pq_chunks);

  int train_pq(IndexThreads::Pointer threads, const IndexMeta &meta,
               std::string &train_data, size_t num_train, uint32_t num_centers,
               uint32_t num_pq_chunks, uint32_t max_iterations,
               bool use_zero_mean, std::vector<uint8_t> &full_pivot_data,
               std::vector<uint8_t> &centroid,
               std::vector<uint32_t> &chunk_offsets);

 private:
  static constexpr uint32_t compress_batch_size_{
      DiskAnnEntity::kDefaultCompressBatchSize};

  // pq cluster
  MultiChunkCluster chunk_cluster_;
  IndexCluster::CentroidList cluster_centroids_;
  uint32_t max_train_sample_count_{PQTable::kMaxTrainSampleCount};
};

}  // namespace core
}  // namespace zvec