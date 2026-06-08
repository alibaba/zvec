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

#include "diskann_pq_table.h"
#include "diskann_entity.h"

namespace zvec {
namespace core {

PQTable::PQTable(const IndexMeta &meta, uint32_t chunk_num)
    : chunk_num_(chunk_num) {
  meta_ = meta;

  if (meta.metric_name() == "Cosine") {
    if (meta.data_type() == IndexMeta::DataType::DT_FP32) {
      meta_.set_dimension(meta.dimension() - 1);
    } else {
      meta_.set_dimension(meta.dimension() - 2);
    }
  }
}

PQTable::~PQTable() {}

int PQTable::init(std::vector<uint8_t> &full_pivot_data,
                  std::vector<uint8_t> &centroid,
                  std::vector<uint32_t> &chunk_offsets,
                  std::vector<uint8_t> &pq_data) {
  full_pivot_data_ = std::move(full_pivot_data);
  centroid_ = std::move(centroid);
  chunk_offsets_ = std::move(chunk_offsets);
  pq_data_ = std::move(pq_data);

  // alloc and compute transpose
  transposed_tables_.resize(kPQCentroidNum * meta_.element_size());

  uint32_t dim = meta_.dimension();
  uint32_t type = meta_.data_type();

  switch (type) {
    case IndexMeta::DataType::DT_FP32: {
      float *transposed_tables_ptr =
          reinterpret_cast<float *>(&transposed_tables_[0]);
      float *full_pivot_data_ptr =
          reinterpret_cast<float *>(&full_pivot_data_[0]);
      for (size_t i = 0; i < kPQCentroidNum; i++) {
        for (size_t j = 0; j < dim; j++) {
          transposed_tables_ptr[j * kPQCentroidNum + i] =
              full_pivot_data_ptr[i * dim + j];
        }
      }
      break;
    }
    case IndexMeta::DataType::DT_FP16: {
      ailego::Float16 *transposed_tables_ptr =
          reinterpret_cast<ailego::Float16 *>(&transposed_tables_[0]);
      ailego::Float16 *full_pivot_data_ptr =
          reinterpret_cast<ailego::Float16 *>(&full_pivot_data_[0]);
      for (size_t i = 0; i < kPQCentroidNum; i++) {
        for (size_t j = 0; j < dim; j++) {
          transposed_tables_ptr[j * kPQCentroidNum + i] =
              full_pivot_data_ptr[i * dim + j];
        }
      }
      break;
    }
    default:
      LOG_ERROR("unsupported type, type: %u", type);
      return IndexError_Unsupported;
  }

  return 0;
}

void PQTable::aggregate_coords(uint32_t id_num, const diskann_id_t *ids,
                               const uint8_t *all_coords, size_t dim,
                               uint8_t *out) {
  for (size_t i = 0; i < id_num; i++) {
    memcpy(out + i * dim, all_coords + ids[i] * dim, dim * sizeof(uint8_t));
  }
}

void PQTable::pq_dist_lookup(const uint8_t *pq_ids, size_t id_num,
                             size_t pq_nchunks, const float *pq_dist_buffer,
                             float *dists_out) {
  ailego_prefetch(dists_out);
  ailego_prefetch(pq_ids);
  ailego_prefetch(pq_ids + 64);
  ailego_prefetch(pq_ids + 128);

  memset(dists_out, 0, id_num * sizeof(float));

  for (size_t chunk = 0; chunk < pq_nchunks; chunk++) {
    const float *chunk_dists = pq_dist_buffer + kPQCentroidNum * chunk;
    if (chunk < pq_nchunks - 1) {
      ailego_prefetch(chunk_dists + kPQCentroidNum);
    }
    for (size_t idx = 0; idx < id_num; idx++) {
      uint8_t pq_centerid = pq_ids[pq_nchunks * idx + chunk];
      dists_out[idx] += chunk_dists[pq_centerid];
    }
  }
}

}  // namespace core
}  // namespace zvec
