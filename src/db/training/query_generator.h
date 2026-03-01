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
 * @brief Training query generator for OMEGA model training
 *
 * This class provides utilities to generate training queries by:
 * 1. Sampling base vectors from a persisted segment
 * 2. Adding Gaussian noise to simulate realistic query variations
 */
class TrainingQueryGenerator {
 public:
  /**
   * @brief Sample base vectors from a segment
   *
   * @param segment The segment to sample from (must be persisted)
   * @param field_name The vector field name to sample
   * @param num_samples Number of vectors to sample
   * @param seed Random seed for reproducibility
   * @return Vector of sampled vectors
   */
  static std::vector<std::vector<float>> SampleBaseVectors(
      const Segment::Ptr& segment,
      const std::string& field_name,
      size_t num_samples,
      uint64_t seed = 42);

  /**
   * @brief Add Gaussian noise to base vectors
   *
   * @param base_vectors Input vectors
   * @param noise_scale Standard deviation of Gaussian noise
   * @param seed Random seed for reproducibility
   * @return Vectors with added noise
   */
  static std::vector<std::vector<float>> AddGaussianNoise(
      const std::vector<std::vector<float>>& base_vectors,
      float noise_scale = 0.01f,
      uint64_t seed = 42);

  /**
   * @brief Generate training queries (sample + noise)
   *
   * Combines sampling and noise addition in one step.
   *
   * @param segment The segment to sample from
   * @param field_name The vector field name
   * @param num_queries Number of training queries to generate
   * @param noise_scale Standard deviation of Gaussian noise
   * @param seed Random seed for reproducibility
   * @return Training query vectors
   */
  static std::vector<std::vector<float>> GenerateTrainingQueries(
      const Segment::Ptr& segment,
      const std::string& field_name,
      size_t num_queries,
      float noise_scale = 0.01f,
      uint64_t seed = 42);
};

}  // namespace zvec
