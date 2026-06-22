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

namespace zvec {
namespace core {

class PQTable {
 public:
  typedef std::shared_ptr<PQTable> Pointer;

 public:
  static constexpr uint32_t kPQBitNum = 8;
  static constexpr uint32_t kPQCentroidNum = 1 << kPQBitNum;
  static constexpr uint32_t kMaxTrainSampleCount = 200000;
  static constexpr double kTrainSampleRatio = 1.0;
  static constexpr uint32_t kMeanIterNum = 12;

 public:
  PQTable(const IndexMeta &meta, uint32_t chunk_num);
  virtual ~PQTable();

  int init(std::vector<uint8_t> &table, std::vector<uint8_t> &centroid,
           std::vector<uint32_t> &chunk_offsets, std::vector<uint8_t> &pq_data);


  template <typename T>
  void compute_distance_table_ip(const T *query_vec, float *dist_vec) {
    memset(dist_vec, 0, kPQCentroidNum * chunk_num_ * sizeof(float));

    const T *transposed_tables_ptr =
        reinterpret_cast<const T *>(transposed_tables_.data());
    // chunk wise distance computation
    for (size_t chunk = 0; chunk < chunk_num_; chunk++) {
      // sum (q-c)^2 for the dimensions associated with this chunk
      float *chunk_dists = dist_vec + (kPQCentroidNum * chunk);

      for (size_t j = chunk_offsets_[chunk]; j < chunk_offsets_[chunk + 1];
           j++) {
        const T *centers_dim_vec = &transposed_tables_ptr[kPQCentroidNum * j];

        for (size_t idx = 0; idx < kPQCentroidNum; idx++) {
          float centor_data = centers_dim_vec[idx];
          float query_data = query_vec[j];
          float dim_score = centor_data * query_data;
          chunk_dists[idx] += -dim_score;
        }
      }
    }
  }

  template <typename T>
  void compute_distance_table(const T *query_vec, float *dist_vec) {
    memset(dist_vec, 0, kPQCentroidNum * chunk_num_ * sizeof(float));

    const T *transposed_tables_ptr =
        reinterpret_cast<const T *>(transposed_tables_.data());
    // chunk wise distance computation
    for (size_t chunk = 0; chunk < chunk_num_; chunk++) {
      // sum (q-c)^2 for the dimensions associated with this chunk
      float *chunk_dists = dist_vec + (kPQCentroidNum * chunk);

      for (size_t j = chunk_offsets_[chunk]; j < chunk_offsets_[chunk + 1];
           j++) {
        const T *centers_dim_vec = &transposed_tables_ptr[kPQCentroidNum * j];

        for (size_t idx = 0; idx < kPQCentroidNum; idx++) {
          float diff = centers_dim_vec[idx] - query_vec[j];
          chunk_dists[idx] += (diff * diff);
        }
      }
    }
  }

  template <typename T>
  void preprocess_query(T *query) {
    const T *centroid_ptr = reinterpret_cast<const T *>(centroid_.data());
    for (size_t i = 0; i < meta_.dimension(); i++) {
      query[i] -= centroid_ptr[i];
    }
  }

  void aggregate_coords(uint32_t id_num, const diskann_id_t *ids,
                        const uint8_t *all_coords, size_t dim, uint8_t *out);

  void pq_dist_lookup(const uint8_t *pq_ids, size_t id_num, size_t pq_nchunks,
                      const float *pq_dist_buffer, float *dists);

  void compute_dists(uint32_t id_num, const diskann_id_t *ids,
                     uint32_t chunk_num, float *pq_dist_buffer,
                     void *coord_buffer, float *dists) {
    uint8_t *coord_buffer_ptr = reinterpret_cast<uint8_t *>(coord_buffer);

    aggregate_coords(id_num, ids, this->pq_data(), chunk_num, coord_buffer_ptr);

    pq_dist_lookup(coord_buffer_ptr, id_num, chunk_num, pq_dist_buffer, dists);

    return;
  }

  int preprocess_pq_dist_table(void *query_rotated, float *dist_buffer) {
    switch (meta_.data_type()) {
      case IndexMeta::DataType::DT_FP32:
        preprocess_query(reinterpret_cast<float *>(query_rotated));
        compute_distance_table(reinterpret_cast<float *>(query_rotated),
                               dist_buffer);

        break;
      case IndexMeta::DataType::DT_FP16:
        preprocess_query(reinterpret_cast<ailego::Float16 *>(query_rotated));
        compute_distance_table(
            reinterpret_cast<ailego::Float16 *>(query_rotated), dist_buffer);
        break;
      default:
        LOG_ERROR("Unsupported Type: %u", meta_.data_type());
        return IndexError_Unsupported;
    }

    return 0;
  }

 public:
  const uint8_t *pq_data() const {
    return pq_data_.data();
  }

 private:
  std::vector<uint8_t> full_pivot_data_;
  std::vector<uint8_t> transposed_tables_;

  std::vector<uint8_t> centroid_;
  std::vector<uint32_t> chunk_offsets_;
  std::vector<uint8_t> pq_data_;

  IndexMeta meta_;
  uint64_t chunk_num_{0};
};

}  // namespace core
}  // namespace zvec
