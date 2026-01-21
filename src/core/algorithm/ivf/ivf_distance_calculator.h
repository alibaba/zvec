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

#include <zvec/ailego/utility/time_helper.h>
#include <zvec/core/framework/index_framework.h>

namespace zvec {
namespace core {

class IVFDistanceCalculator {
 public:
  typedef std::shared_ptr<IVFDistanceCalculator> Pointer;

  //! Constructor
  IVFDistanceCalculator(const IndexMeta &meta,
                        const IndexMetric::Pointer &metric,
                        uint32_t block_vec_cnt);

  virtual ~IVFDistanceCalculator();

 public:
  inline void query_centroids_distance(const void *query, size_t qnum,
                                       const void *feature, size_t fnum,
                                       float *distances);

  inline void query_centroids_distance(const void *query, const void *feature,
                                       size_t fnum, float *distances);

  inline void query_features_distance(const void *query, const void *feature,
                                      size_t fnum, float *distances);

  inline void query_features_distance(const void *query, const void *feature,
                                      bool column_major, size_t fnum,
                                      float *distances);

 protected:
  //! Row Major Distances -> Online
  inline void row_major_distance(const void *query, size_t qnum,
                                 const void *feature, size_t fnum, float *out);

  inline void row_major_distance(const void *query, const void *feature,
                                 size_t fnum, float *out);

  template <size_t Q>
  inline void batch_query_centroids_distance(const void *query,
                                             const void *feature, size_t fnum,
                                             float *distances);

 protected:
  IndexMetric::Pointer metric_ptr_{};
  IndexMetric::MatrixDistance row_distance_{nullptr};
  IndexMetric::MatrixDistance distanceXx1_{nullptr};
  std::vector<IndexMetric::MatrixDistance> distances_{};

  size_t element_size_{0};
  size_t dimension_{0};
  uint32_t block_vec_cnt_{0};
  bool column_major_order_{false};
};

void IVFDistanceCalculator::query_centroids_distance(const void *query,
                                                     size_t qnum,
                                                     const void *feature,
                                                     size_t fnum,
                                                     float *distances) {
  if (column_major_order_) {
    switch (qnum) {
      case 1:
        batch_query_centroids_distance<1>(query, feature, fnum, distances);
        break;
      case 16:
        batch_query_centroids_distance<16>(query, feature, fnum, distances);
        break;
      case 8:
        batch_query_centroids_distance<8>(query, feature, fnum, distances);
        break;
      case 4:
        batch_query_centroids_distance<4>(query, feature, fnum, distances);
        break;
      case 2:
        batch_query_centroids_distance<2>(query, feature, fnum, distances);
        break;
      case 32:
        batch_query_centroids_distance<32>(query, feature, fnum, distances);
        break;
      default:
        LOG_ERROR("Unsupported query num %zu.", qnum);
        break;
    }
  } else {
    const uint8_t *cur_query = reinterpret_cast<const uint8_t *>(query);
    for (size_t q = 0; q < qnum; ++q) {
      this->row_major_distance(cur_query, feature, fnum, distances);
      cur_query += element_size_;
      distances += block_vec_cnt_;
    }
  }
}

void IVFDistanceCalculator::query_centroids_distance(const void *query,
                                                     const void *feature,
                                                     size_t fnum,
                                                     float *distances) {
  this->query_features_distance(query, feature, fnum, distances);
}

void IVFDistanceCalculator::query_features_distance(const void *query,
                                                    const void *feature,
                                                    size_t fnum,
                                                    float *distances) {
  if (column_major_order_) {
    if (fnum == block_vec_cnt_) {
      distanceXx1_(feature, query, dimension_, distances);
    } else {
      this->row_major_distance(query, feature, fnum, distances);
    }
  } else {
    this->row_major_distance(query, feature, fnum, distances);
  }
}

void IVFDistanceCalculator::query_features_distance(const void *query,
                                                    const void *feature,
                                                    bool column_major,
                                                    size_t fnum,
                                                    float *distances) {
  if (column_major) {
    ailego_assert_with(fnum == block_vec_cnt_, "Invalid Block");
    distanceXx1_(feature, query, dimension_, distances);
  } else {
    this->row_major_distance(query, feature, fnum, distances);
  }
}

template <size_t Q>
void IVFDistanceCalculator::batch_query_centroids_distance(const void *query,
                                                           const void *feature,
                                                           size_t fnum,
                                                           float *distances) {
  if (fnum == block_vec_cnt_) {
    distances_[Q](feature, query, dimension_, distances);
  } else {
    row_major_distance(query, Q, feature, fnum, distances);
  }
}

void IVFDistanceCalculator::row_major_distance(const void *query, size_t qnum,
                                               const void *feature, size_t fnum,
                                               float *out) {
  const uint8_t *cur_query = reinterpret_cast<const uint8_t *>(query);
  for (size_t q = 0; q < qnum; ++q) {
    const uint8_t *tmp_feature = reinterpret_cast<const uint8_t *>(feature);
    float *cur_out = out + q * fnum;
    for (size_t f = 0; f < fnum; ++f) {
      row_distance_(cur_query, tmp_feature, dimension_, cur_out + f);
      tmp_feature += element_size_;
    }
    cur_query += element_size_;
  }
}

void IVFDistanceCalculator::row_major_distance(const void *query,
                                               const void *feature, size_t fnum,
                                               float *out) {
  const uint8_t *cur_feature = reinterpret_cast<const uint8_t *>(feature);
  for (size_t f = 0; f < fnum; ++f) {
    row_distance_(query, cur_feature, dimension_, out + f);
    cur_feature += element_size_;
  }
}

}  // namespace core
}  // namespace zvec
