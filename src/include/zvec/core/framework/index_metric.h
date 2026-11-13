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

#include <memory>
#include <zvec/ailego/container/params.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_meta.h>
#include <zvec/core/framework/index_module.h>
#include <zvec/turbo/turbo.h>

namespace zvec {
namespace core {

/*! Index Metric
 */
struct IndexMetric : public IndexModule {
  //! Index Metric Pointer
  typedef std::shared_ptr<IndexMetric> Pointer;

  //! Matrix Distance Function
  typedef void (*MatrixDistanceHandle)(const void *m, const void *q, size_t dim,
                                       float *out);

  //! Matrix Distance Function Object
  using MatrixDistance =
      std::function<void(const void *m, const void *q, size_t dim, float *out)>;

  //! Matrix Sparse Distance Function
  typedef void (*MatrixSparseDistanceHandle)(const void *m_sparse_data,
                                             const void *q_sparse_data,
                                             float *out);

  //! Matrix Sparse Distance Function Object
  using MatrixSparseDistance = std::function<void(
      const void *m_sparse_data, const void *q_sparse_data, float *out)>;


  //! Matrix Batch Distance Function. `extra_values[i]` corresponds to `m[i]`;
  //! callers pass nullptr when the metric has no per-vector extra values.
  typedef void (*MatrixBatchDistanceHandle)(const void **m, const void *q,
                                            size_t num, size_t dim, float *out,
                                            const void **extra_values);

  //! Matrix Batch Distance Function Object
  using MatrixBatchDistance =
      std::function<void(const void **m, const void *q, size_t num, size_t dim,
                         float *out, const void **extra_values)>;

  //! Destructor
  ~IndexMetric() override = default;

  //! Initialize Metric
  virtual int init(const IndexMeta &meta, const ailego::Params &params) = 0;

  //! Cleanup Metric
  virtual int cleanup() = 0;

  //! Retrieve if it matched
  virtual bool is_matched(const IndexMeta &meta) const = 0;

  //! Retrieve if it matched
  virtual bool is_matched(const IndexMeta &meta,
                          const IndexQueryMeta &qmeta) const = 0;

  //! Retrieve distance function for query
  virtual MatrixDistance distance() const {
    return nullptr;
  }

  //! Retrieve hybrid distance function for query
  virtual MatrixSparseDistance sparse_distance() const {
    return nullptr;
  };

  //! Retrieve distance function for query
  virtual MatrixBatchDistance batch_distance() const {
    return nullptr;
  }

  //! Number of trailing bytes per stored record that a storage backend may
  //! split into an extra-values column.
  virtual size_t extra_values_size_per_vector() const {
    return 0;
  }

  //! Retrieve distance function for index features
  virtual MatrixDistance distance_matrix(size_t /*m*/, size_t /*n*/) const {
    return nullptr;
  }

  //! Retrieve params of Metric
  virtual const ailego::Params &params() const = 0;

  //! Retrieve query metric object of this index metric
  virtual Pointer query_metric() const = 0;

  //! Normalize result
  virtual void normalize(float *score) const {
    (void)score;
  }

  //! Denormalize result
  virtual void denormalize(float *score) const {
    (void)score;
  }

  //! Retrieve if it supports normalization
  virtual bool support_normalize() const {
    return false;
  }

  //! Train the metric
  virtual int train(const void *vec, size_t dim) {
    (void)vec;
    (void)dim;
    return 0;
  }

  //! Retrieve if it supports training
  virtual bool support_train() const {
    return false;
  }

  //! Compute the distance between feature and query
  float distance(const void *m, const void *q, size_t dim) const {
    float dist;
    (this->distance())(m, q, dim, &dist);
    return dist;
  }

  using DistanceBatchQueryPreprocessFunc = turbo::QueryPreprocessFunc;

  virtual DistanceBatchQueryPreprocessFunc get_query_preprocess_func() const {
    return nullptr;
  }

  //! Distance offset applied during graph build to make the internal distance
  //! non-negative for ratio-based pruning (e.g. Vamana RobustPrune's
  //! occlude_factor = d(q,c) / d(p,c)). Metrics whose internal distance is a
  //! well-defined non-negative value (SquaredEuclidean, 1-cos, etc.) should
  //! leave the default 0. Metrics that internally store a signed quantity
  //! (e.g. -cos for Cosine / NormalizedCosine on the quantized int8 path)
  //! should override this to return a constant C such that (internal_dist + C)
  //! is always non-negative and preserves the ordering of the original
  //! distance.
  virtual float build_distance_offset() const {
    return 0.0f;
  }
};

}  // namespace core
}  // namespace zvec
