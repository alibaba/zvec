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
#include <vector>
#include "algorithm/hnsw_rabitq/rabitq_params.h"
#include "zvec/core/framework/index_dumper.h"
#include "zvec/core/framework/index_storage.h"
#include "ivf_rabitq_context.h"

namespace zvec {
namespace core {

/*! IVF RaBitQ Reformer
 * Manages centroids, rotator, and quantization for IVF RaBitQ.
 * Provides methods for query transformation, centroid assignment,
 * and batch quantization for cluster building.
 *
 * All rabitqlib types are hidden behind a pimpl to avoid leaking rabitqlib
 * headers to consumers of this class.
 */
class IvfRabitqReformer {
 public:
  typedef std::shared_ptr<IvfRabitqReformer> Pointer;

  IvfRabitqReformer();
  ~IvfRabitqReformer();

  // Non-copyable
  IvfRabitqReformer(const IvfRabitqReformer &) = delete;
  IvfRabitqReformer &operator=(const IvfRabitqReformer &) = delete;

  //! Initialize with metric parameters
  int init(const std::string &metric_name);

  //! Load from storage (reads rabitq.converter segment)
  int load(IndexStorage::Pointer storage);

  //! Dump to storage (writes rabitq.converter segment)
  int dump(const IndexStorage::Pointer &storage);

  //! Dump to dumper (writes rabitq.converter segment via IndexDumper)
  int dump(const IndexDumper::Pointer &dumper);

  //! Create a query state for searching
  int create_query_state(const float *query, IvfRabitqQueryState *state) const;

  //! Prepare query state from a selected probe centroid.
  //! This sets g_add / g_error without recomputing centroid distances.
  int prepare_for_cluster(const IvfRabitqProbeCentroid &centroid,
                          IvfRabitqQueryState *state) const;

  //! Prepare an arbitrary centroid not selected by coarse search.
  int prepare_for_cluster(uint32_t centroid_id,
                          IvfRabitqQueryState *state) const;

  //! Rotate a single vector: input[dimension] -> output[padded_dim]
  int rotate_vector(const float *input, float *output) const;

  //! Quantize a batch of rotated vectors for a specific centroid.
  //! rotated_data: num_points vectors of padded_dim dimension each
  //! centroid_id: which centroid to use as residual reference
  //! num_points: up to fastscan::kBatchSize (32)
  //! batch_data: output batch-packed data
  //! ex_data: output extra-bit data for each vector
  int quantize_batch(const float *rotated_data, uint32_t centroid_id,
                     size_t num_points, char *batch_data, char *ex_data) const;

  //! Find nearest centroid for a vector (using original centroids, brute-force)
  uint32_t find_nearest_centroid(const float *vector) const;

  //! Select top-n probe centroids using the same metric as build assignment.
  int select_probe_centroids(const float *query, size_t nprobe,
                             std::vector<uint32_t> *centroids) const;
  int select_probe_centroids(
      const float *query, size_t nprobe, IvfRabitqQueryState *state,
      std::vector<IvfRabitqProbeCentroid> *centroids) const;

  //! Accessors
  size_t num_clusters() const;
  size_t dimension() const;
  size_t padded_dim() const;
  size_t ex_bits() const;
  RabitqMetricType rabitq_metric_type() const;
  bool loaded() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace core
}  // namespace zvec
