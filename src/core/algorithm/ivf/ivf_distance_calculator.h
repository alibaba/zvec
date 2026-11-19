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

#include <turbo/quantizer/distance.h>
#include <zvec/ailego/utility/time_helper.h>
#include <zvec/core/framework/index_framework.h>

namespace zvec {
namespace core {

class IVFDistanceCalculator {
 public:
  bool column_many_supported() const {
    return column_major_order_ && row_distance_ && column_single_distance_;
  }

  turbo::DistanceImpl column_query(const void *query,
                                   const IndexQueryMeta &qmeta) const {
    return turbo::DistanceImpl(
        row_distance_,
        std::string(static_cast<const char *>(query), qmeta.element_size()),
        dimension_);
  }

  bool joint_column_kernel() const {
    return static_cast<bool>(column_many_distance_);
  }

  void column_queries_distance(const void *const *queries, size_t n,
                               const void *features, size_t count, float *out) {
    if (column_many_distance_ && count == block_vec_cnt_ && n > 1) {
      column_many_distance_(features, queries, n, dimension_, out);
    } else {
      for (size_t q = 0; q < n; ++q) {
        query_features_distance(queries[q], features, count,
                                out + q * block_vec_cnt_);
      }
    }
  }

  typedef std::shared_ptr<IVFDistanceCalculator> Pointer;

  //! Constructor
  IVFDistanceCalculator(const IndexMeta &meta,
                        const IndexMetric::Pointer &metric,
                        uint32_t block_vec_cnt);

  virtual ~IVFDistanceCalculator();

 public:
  //! Gather row-major Turbo codes, including metadata tails and partial blocks.
  void query_features_distance(const turbo::DistanceImpl &distance,
                               const void *feature, size_t fnum,
                               float *out) const {
    const void *candidates[32];
    const auto *rows = static_cast<const char *>(feature);
    for (size_t offset = 0; offset < fnum; offset += 32) {
      const size_t count = std::min(size_t(32), fnum - offset);
      for (size_t i = 0; i < count; ++i) {
        candidates[i] = rows + (offset + i) * element_size_;
      }
      distance.batch(candidates, count, out + offset);
    }
  }

  inline void query_features_distance(const void *query, const void *feature,
                                      size_t fnum, float *distances);

  inline void query_features_distance(const void *query, const void *feature,
                                      bool column_major, size_t fnum,
                                      float *distances);

 protected:
  //! Row Major Distances -> Online
  inline void row_major_distance(const void *query, const void *feature,
                                 size_t fnum, float *out);

 protected:
  IndexMetric::Pointer metric_ptr_{};
  IndexMetric::MatrixDistance row_distance_{nullptr};
  IndexMetric::MatrixDistance column_single_distance_{nullptr};
  turbo::InterleavedManyToManyDistanceFunc column_many_distance_{};

  size_t element_size_{0};
  size_t dimension_{0};
  uint32_t block_vec_cnt_{0};
  bool column_major_order_{false};
};

void IVFDistanceCalculator::query_features_distance(const void *query,
                                                    const void *feature,
                                                    size_t fnum,
                                                    float *distances) {
  if (column_major_order_) {
    if (fnum == block_vec_cnt_) {
      column_single_distance_(feature, query, dimension_, distances);
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
    column_single_distance_(feature, query, dimension_, distances);
  } else {
    this->row_major_distance(query, feature, fnum, distances);
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
