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

#include "query_generator.h"
#include <random>
#include <zvec/ailego/logger/logger.h>
#include <zvec/db/doc.h>

namespace zvec {

std::vector<std::vector<float>> TrainingQueryGenerator::SampleBaseVectors(
    const Segment::Ptr& segment,
    const std::string& field_name,
    size_t num_samples,
    uint64_t seed) {
  std::vector<std::vector<float>> sampled_vectors;

  // Get total document count
  uint64_t doc_count = segment->doc_count();
  if (doc_count == 0) {
    LOG_WARN("Segment has no documents, cannot sample base vectors");
    return sampled_vectors;
  }

  // Adjust num_samples if it exceeds doc_count
  size_t actual_samples = std::min(num_samples, static_cast<size_t>(doc_count));
  if (actual_samples < num_samples) {
    LOG_WARN("Requested %zu samples but segment only has %zu documents",
             num_samples, static_cast<size_t>(doc_count));
  }

  // Random number generator for sampling indices
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<uint64_t> dist(0, doc_count - 1);

  sampled_vectors.reserve(actual_samples);

  // Sample vectors
  for (size_t i = 0; i < actual_samples; ++i) {
    // Get random document index
    uint64_t doc_idx = dist(rng);

    // Fetch document
    auto doc = segment->Fetch(doc_idx);
    if (!doc) {
      LOG_WARN("Failed to fetch document at index %zu, skipping",
               static_cast<size_t>(doc_idx));
      continue;
    }

    // Extract vector field
    auto vector_opt = doc->get<std::vector<float>>(field_name);
    if (!vector_opt.has_value()) {
      LOG_WARN("Document at index %zu does not have field '%s', skipping",
               static_cast<size_t>(doc_idx), field_name.c_str());
      continue;
    }

    sampled_vectors.push_back(vector_opt.value());
  }

  LOG_INFO("Successfully sampled %zu/%zu vectors from segment",
           sampled_vectors.size(), actual_samples);

  return sampled_vectors;
}

std::vector<std::vector<float>> TrainingQueryGenerator::AddGaussianNoise(
    const std::vector<std::vector<float>>& base_vectors,
    float noise_scale,
    uint64_t seed) {
  if (base_vectors.empty()) {
    LOG_WARN("Input base_vectors is empty, returning empty result");
    return {};
  }

  std::vector<std::vector<float>> noisy_vectors;
  noisy_vectors.reserve(base_vectors.size());

  // Random number generator for Gaussian noise
  std::mt19937 rng(seed);
  std::normal_distribution<float> gaussian(0.0f, noise_scale);

  for (const auto& base_vector : base_vectors) {
    if (base_vector.empty()) {
      LOG_WARN("Encountered empty vector, skipping");
      continue;
    }

    std::vector<float> noisy_vector;
    noisy_vector.reserve(base_vector.size());

    // Add Gaussian noise to each dimension
    for (float base_value : base_vector) {
      float noise = gaussian(rng);
      noisy_vector.push_back(base_value + noise);
    }

    noisy_vectors.push_back(std::move(noisy_vector));
  }

  LOG_INFO("Added Gaussian noise (scale=%.4f) to %zu vectors",
           noise_scale, noisy_vectors.size());

  return noisy_vectors;
}

std::vector<std::vector<float>> TrainingQueryGenerator::GenerateTrainingQueries(
    const Segment::Ptr& segment,
    const std::string& field_name,
    size_t num_queries,
    float noise_scale,
    uint64_t seed) {
  // Step 1: Sample base vectors
  auto base_vectors = SampleBaseVectors(segment, field_name, num_queries, seed);

  if (base_vectors.empty()) {
    LOG_ERROR("Failed to sample base vectors from segment");
    return {};
  }

  // Step 2: Add Gaussian noise
  // Use a different seed for noise generation to avoid correlation
  auto training_queries = AddGaussianNoise(base_vectors, noise_scale, seed + 1);

  LOG_INFO("Generated %zu training queries for field '%s'",
           training_queries.size(), field_name.c_str());

  return training_queries;
}

}  // namespace zvec
