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

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include "db/index/segment/segment.h"

namespace zvec {

/**
 * @brief Result of sampling base vectors, includes both vectors and doc_ids
 */
struct SampledVectors {
  std::vector<std::vector<float>> vectors;
  std::vector<uint64_t>
      doc_ids;  // doc_id of each sampled vector (for exclusion in GT)
};

/**
 * @brief Training query generator for OMEGA model training
 *
 * This class provides utilities to generate training queries by:
 * 1. Sampling base vectors from a persisted segment (held-out approach)
 * 2. Using sampled vectors directly as queries (no noise)
 * 3. Ground truth computation excludes the query vector itself
 */
class TrainingQueryGenerator {
 public:
  /**
   * @brief Sample base vectors from a segment with doc_ids
   *
   * @param segment The segment to sample from (must be persisted)
   * @param field_name The vector field name to sample
   * @param num_samples Number of vectors to sample
   * @param seed Random seed for reproducibility
   * @return SampledVectors with vectors and their doc_ids
   */
  static SampledVectors SampleBaseVectorsWithIds(const Segment::Ptr &segment,
                                                 const std::string &field_name,
                                                 size_t num_samples,
                                                 uint64_t seed = 42);

  /**
   * @brief Generate training queries using held-out approach
   *
   * Samples vectors directly from index and uses them as queries.
   * Returns doc_ids so ground truth can exclude self-matches.
   *
   * @param segment The segment to sample from
   * @param field_name The vector field name
   * @param num_queries Number of training queries to generate
   * @param seed Random seed for reproducibility
   * @return SampledVectors with query vectors and their doc_ids
   */
  static SampledVectors GenerateHeldOutQueries(const Segment::Ptr &segment,
                                               const std::string &field_name,
                                               size_t num_queries,
                                               uint64_t seed = 42);
};

}  // namespace zvec
