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
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <vector>
#include <turbo/preprocessor/preprocessor.h>
#include <zvec/ailego/container/params.h>

namespace zvec {
namespace turbo {

//! Shared OPQ state and alternating training for all PQ code formats.
//! The caller owns the codebook and performs normalization/centering first.
class PqOpq {
 public:
  using TrainCodebook = std::function<void(uint32_t max_iters)>;
  using Reconstruct =
      std::function<float(const float *rotated, size_t num, float *x_hat)>;

  int init(uint32_t dim, DataType data_type, const ailego::Params &params);

  bool enabled() const {
    return preprocessor_ != nullptr;
  }

  uint8_t rotate_type() const {
    return enabled() ? static_cast<uint8_t>(RotateType::kOpq) : uint8_t{0};
  }

  //! Operates on packed fp32 data, leaving it rotated by the final matrix.
  //! train_codebook must refresh its centroid pointer cache each round.
  //! The caller performs the final full PQ training after this returns.
  void train(void *data, size_t num, const char *name,
             const TrainCodebook &train_codebook,
             const Reconstruct &reconstruct);

  //! Returns input unchanged when disabled, otherwise rotates into buffer.
  const void *apply(const void *input, std::vector<float> *buffer) const;

  //! These two operations require enabled() and non-overlapping buffers.
  void rotate(const float *input, float *output) const;
  void rotate_inverse(const float *input, float *output) const;

  //! Retains the existing rotator blob format; disabled OPQ has an empty blob.
  int serialize(std::string *out) const;
  int deserialize(uint8_t rotate_type, uint32_t dim, const void *data,
                  size_t len);

 private:
  void rotate_batch(const float *input, size_t num, float *output) const;

  static constexpr float kMinImprovement = 1e-4f;
  Preprocessor::Pointer preprocessor_;
  uint32_t opq_iter_{5};
  uint32_t opq_pq_iter_{4};
};

//! Reconstruct packed fp32 training vectors from a PQ codebook. chunk_dim(m)
//! supports both uniform blocks and PQ-int8's unequal blocks. Centroid pointers
//! must already be refreshed by the quantizer after training its codebook.
template <size_t NumCentroids, typename ChunkDim>
float pq_reconstruct_batch(
    const float *rotated, size_t num, size_t dim,
    const std::vector<std::vector<const void *>> &centroid_ptrs,
    const BatchDistanceFunc &l2_batch, ChunkDim chunk_dim, float *x_hat) {
  float dists[NumCentroids];
  double sum_err = 0.0;
  for (size_t i = 0; i < num; ++i) {
    size_t offset = 0;
    for (size_t m = 0; m < centroid_ptrs.size(); ++m) {
      const size_t width = chunk_dim(m);
      const auto &centroids = centroid_ptrs[m];
      l2_batch(const_cast<const void **>(centroids.data()),
               rotated + i * dim + offset, NumCentroids, width, dists, nullptr);
      // Empty clusters can contain NaNs; do not seed the argmin from one.
      float best_dist = std::numeric_limits<float>::infinity();
      size_t best_idx = 0;
      for (size_t j = 0; j < NumCentroids; ++j) {
        if (dists[j] < best_dist) {
          best_dist = dists[j];
          best_idx = j;
        }
      }
      std::memcpy(x_hat + i * dim + offset, centroids[best_idx],
                  width * sizeof(float));
      sum_err += best_dist;
      offset += width;
    }
  }
  return static_cast<float>(sum_err / static_cast<double>(num));
}

}  // namespace turbo
}  // namespace zvec
