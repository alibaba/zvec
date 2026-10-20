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

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
#include <zvec/turbo/turbo.h>

namespace zvec {
namespace turbo {

//! A callable distance handle bound to a quantized query vector.
//!
//! DistanceImpl owns the quantized query bytes and a dispatched
//! DistanceFunc. Invoking `operator()(candidate)` computes the distance
//! between the stored query and the given candidate vector, which is
//! expected to already be in the same quantized layout.
class DistanceImpl {
 public:
  DistanceImpl() = default;

  DistanceImpl(DistanceFunc func, std::string quantized_query, size_t dim)
      : func_(std::move(func)),
        query_storage_(std::move(quantized_query)),
        dim_(dim) {}

  DistanceImpl(DistanceFunc func, BatchDistanceFunc batch_func,
               std::string quantized_query, size_t dim,
               StridedBatchDistanceFunc batch_strided_func = {})
      : func_(std::move(func)),
        batch_func_(std::move(batch_func)),
        batch_strided_func_(std::move(batch_strided_func)),
        query_storage_(std::move(quantized_query)),
        dim_(dim) {}

  //! Whether the handle is ready to compute distances.
  bool valid() const {
    return static_cast<bool>(func_);
  }

  //! Whether a batch distance function is available.
  bool batch_valid() const {
    return static_cast<bool>(batch_func_);
  }

  //! Compute the distance between the stored query and `candidate`.
  float operator()(const void *candidate) const {
    float d = 0.0f;
    func_(candidate, query_storage_.data(), dim_, &d);
    return d;
  }

  //! Compute distances for a batch of `num` candidates against the
  //! stored query. Falls back to the single-pair path when no batch function
  //! is bound.
  void batch(const void **candidates, size_t num, float *out) const {
    if (batch_func_) {
      batch_func_(candidates, query_storage_.data(), num, dim_, out, nullptr);
      return;
    }
    for (size_t i = 0; i < num; ++i) {
      out[i] = 0.0f;
      func_(candidates[i], query_storage_.data(), dim_, out + i);
    }
  }

  //! Distances for row-major codes separated by stride_bytes (including tails
  //! and padding). An empty batch is a no-op, even with null buffers.
  //! Without a native kernel, reuse batch() in bounded, allocation-free chunks.
  void batch_strided(const void *base, size_t stride_bytes, size_t num,
                     float *out) const {
    if (num == 0) {
      return;
    }
    if (batch_strided_func_) {
      batch_strided_func_(base, stride_bytes, query_storage_.data(), num, dim_,
                          out, nullptr);
      return;
    }
    const void *candidates[32];
    const auto *rows = static_cast<const char *>(base);
    for (size_t offset = 0; offset < num;) {
      const size_t count = std::min(size_t(32), num - offset);
      for (size_t i = 0; i < count; ++i) {
        candidates[i] = rows + (offset + i) * stride_bytes;
      }
      batch(candidates, count, out + offset);
      offset += count;
    }
  }

  //! Access the quantized query bytes (for pairwise helpers).
  const std::string &query_storage() const {
    return query_storage_;
  }

  size_t dim() const {
    return dim_;
  }

  //! Raw single-pair distance function (operates on already-quantized
  //! candidates). Useful for pairwise node-vs-node distance where no
  //! stored query is involved.
  const DistanceFunc &func() const {
    return func_;
  }

  //! Raw batch distance function.
  const BatchDistanceFunc &batch_func() const {
    return batch_func_;
  }

 private:
  DistanceFunc func_{};
  BatchDistanceFunc batch_func_{};
  StridedBatchDistanceFunc batch_strided_func_{};
  std::string query_storage_{};
  size_t dim_{0};
};

}  // namespace turbo
}  // namespace zvec
