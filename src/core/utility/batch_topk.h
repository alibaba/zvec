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

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>

namespace zvec {
namespace core {

// Buffered top-k for distance batches with opaque uint64 indices.
// SIMD rejects distances outside the current bound; surviving candidates are
// appended without heap maintenance. A full buffer is partitioned down to k.
// This follows the buffering strategy of ScaNN's FastTopNeighbors, without
// depending on ScaNN or its SIMD/partition libraries.
//
// Storage is O(k), with no allocation for k <= 64. Equal-score IDs and output
// order are unspecified, as with a score-only heap. No ID bits are reserved.
// ISA selection is cached at runtime, checking CPU and OS support.
// BatchTopKImpl can select an ISA once before scanning batches; BatchTopK
// dispatches each batch using the cached runtime choice. Each implementation
// uses SIMD only when enabled by the build; otherwise its method is scalar.
class BatchTopK {
 public:
  enum class Implementation { kScalar, kSSE2, kAVX2, kAVX512, kNEON };

  struct Candidate {
    float score;
    uint64_t id;
  };

  BatchTopK() = default;
  BatchTopK(const BatchTopK &) = delete;
  BatchTopK &operator=(const BatchTopK &) = delete;

  // Return the best compiled implementation supported by the CPU and OS.
  static Implementation implementation();

  // The initial bound is inclusive unless strict_threshold is set (e.g. the
  // kth score from a previous selection). NaN scores are rejected.
  void reset(size_t k, float threshold = std::numeric_limits<float>::max(),
             bool strict_threshold = false);

  // Append without resetting. Indices need not be consecutive or monotonic;
  // all 64 bits are preserved. Passing candidates copy their distance/index,
  // so input buffers can be reused after this call. For a nonempty batch,
  // distances and indices must be non-null.
  void push_block(const float *distances, const uint64_t *indices,
                  size_t count);

  // Retain at most k candidates before consuming data()/size(). Further
  // pushes are allowed; call finish() again before consuming their results.
  void finish();

  size_t size() const {
    return size_;
  }

  const Candidate *data() const {
    return entries_;
  }

 protected:
  void push_block_scalar(const float *distances, const uint64_t *indices,
                         size_t count);
  void push_block_sse2(const float *distances, const uint64_t *indices,
                       size_t count);
  void push_block_avx2(const float *distances, const uint64_t *indices,
                       size_t count);
  void push_block_avx512(const float *distances, const uint64_t *indices,
                         size_t count);
  void push_block_neon(const float *distances, const uint64_t *indices,
                       size_t count);
  void append(float score, uint64_t index);
  void append_tail(const float *distances, const uint64_t *indices,
                   size_t count, size_t offset);

  size_t limit_{0};
  float threshold_{std::numeric_limits<float>::max()};
  bool strict_threshold_{false};

 private:
  friend class BatchTopKTestPeer;

  static Implementation select_implementation(bool sse2, bool avx2,
                                              bool avx512f, bool neon,
                                              bool avx_state,
                                              bool avx512_state);
  static Implementation implementation_for_runtime();

  void compact();

  static constexpr size_t kInlineCapacity = 128;
  std::array<Candidate, kInlineCapacity> inline_entries_;
  std::unique_ptr<Candidate[]> allocated_entries_;
  Candidate *entries_{inline_entries_.data()};
  size_t allocated_capacity_{0};
  size_t buffer_capacity_{0};
  size_t size_{0};
};

// Each fixed implementation owns its SIMD loop, with a scalar fallback when
// that ISA is not compiled. Select Impl through BatchTopK::implementation()
// before entering a scan loop; push_block itself does not dispatch.
template <BatchTopK::Implementation Impl>
class BatchTopKImpl : public BatchTopK {
 public:
  void push_block(const float *distances, const uint64_t *indices,
                  size_t count) {
    if constexpr (Impl == Implementation::kSSE2) {
      push_block_sse2(distances, indices, count);
    } else if constexpr (Impl == Implementation::kAVX2) {
      push_block_avx2(distances, indices, count);
    } else if constexpr (Impl == Implementation::kAVX512) {
      push_block_avx512(distances, indices, count);
    } else if constexpr (Impl == Implementation::kNEON) {
      push_block_neon(distances, indices, count);
    } else {
      push_block_scalar(distances, indices, count);
    }
  }
};

}  // namespace core
}  // namespace zvec
