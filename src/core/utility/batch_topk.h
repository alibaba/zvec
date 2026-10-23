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

// Buffered top-k for contiguous distance batches with consecutive uint32 IDs.
// SIMD rejects distances outside the current bound; surviving candidates are
// appended without heap maintenance. A full buffer is partitioned down to k.
// This follows the buffering strategy of ScaNN's FastTopNeighbors, without
// depending on ScaNN or its SIMD/partition libraries.
//
// Storage is O(k), with no allocation for k <= 64. Equal-score IDs and output
// order are unspecified, as with a score-only heap. No ID bits are reserved.
// ISA selection is cached at runtime, checking CPU and OS support. Select a
// BatchTopKImpl once before scanning batches. Each class uses SIMD only when
// enabled by the build's compiler target; otherwise its method is scalar.
class BatchTopK {
 public:
  enum class Implementation { kScalar, kSSE2, kAVX2, kAVX512, kNEON };

  struct Candidate {
    float score;
    uint32_t id;
  };

  BatchTopK() = default;
  BatchTopK(const BatchTopK &) = delete;
  BatchTopK &operator=(const BatchTopK &) = delete;

  // Return the best compiled implementation supported by the CPU and OS.
  static Implementation implementation();

  // The initial bound is inclusive unless strict_threshold is set (e.g. the
  // worst score in an already-full destination heap). NaN scores are rejected.
  void reset(size_t k, float threshold = std::numeric_limits<float>::max(),
             bool strict_threshold = false);

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
  void push_block_scalar(const float *distances, size_t count,
                         uint32_t first_id, float scale);
  void append(float score, uint32_t id);
  void append_tail(const float *distances, size_t count, uint32_t first_id,
                   float scale, size_t offset);

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
// Scores are multiplied by scale before filtering, matching the destination
// heap's score space. The caller guarantees first_id + count - 1 fits uint32.
template <BatchTopK::Implementation Impl>
class BatchTopKImpl;

template <>
class BatchTopKImpl<BatchTopK::Implementation::kScalar> : public BatchTopK {
 public:
  void push_block(const float *distances, size_t count, uint32_t first_id,
                  float scale = 1.0f);
};

template <>
class BatchTopKImpl<BatchTopK::Implementation::kSSE2> : public BatchTopK {
 public:
  void push_block(const float *distances, size_t count, uint32_t first_id,
                  float scale = 1.0f);
};

template <>
class BatchTopKImpl<BatchTopK::Implementation::kAVX2> : public BatchTopK {
 public:
  void push_block(const float *distances, size_t count, uint32_t first_id,
                  float scale = 1.0f);
};

template <>
class BatchTopKImpl<BatchTopK::Implementation::kAVX512> : public BatchTopK {
 public:
  void push_block(const float *distances, size_t count, uint32_t first_id,
                  float scale = 1.0f);
};

template <>
class BatchTopKImpl<BatchTopK::Implementation::kNEON> : public BatchTopK {
 public:
  void push_block(const float *distances, size_t count, uint32_t first_id,
                  float scale = 1.0f);
};

}  // namespace core
}  // namespace zvec
