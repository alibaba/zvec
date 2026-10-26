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
#include <memory>
#include <optional>
#include <vector>
#include <zvec/export.h>

namespace zvec {
namespace core {

class ZVEC_CORE_API VectorSource {
 public:
  // Borrowed row-major storage. Row i starts at data + i * stride bytes.
  // stride must be at least the vector's byte length. Nonempty batches need
  // valid backing storage; count includes rows excluded by the query filter.
  struct ContiguousBatch {
    const void *data{nullptr};
    uint32_t first_id{0};
    uint32_t count{0};
    size_t stride{0};
    // Optional borrowed array of count IDs. If null, row i has first_id + i.
    // Otherwise first_id is ignored. IDs use the get_vector() namespace.
    const uint32_t *ids{nullptr};
  };

  // Logical row-major vectors in the same ID namespace as get_vector().
  // Pointers need not be contiguous. External Flat currently expects raw FP32.
  struct Batch {
    std::vector<uint32_t> ids;
    // Rows excluded by Flat's registered-ID set or filter may be null.
    // Every row selected for distance computation must have a valid pointer.
    std::vector<const void *> vectors;
    // Optional backing allocation or page pin. Vectors and any borrowed ID
    // array must remain valid until the batch is released, or (for scans)
    // the next next_batch() call.
    std::shared_ptr<const void> lease;
    // Alternative to ids/vectors, which must both be empty when set. Flat
    // adapts this view directly into its reusable distance-computation buffers.
    std::optional<ContiguousBatch> contiguous;

    size_t size() const {
      return contiguous ? contiguous->count : ids.size();
    }

    bool empty() const {
      return size() == 0;
    }

    void clear() {
      vectors.clear();
      ids.clear();
      contiguous.reset();
      lease.reset();
    }
  };

  class Iterator {
   public:
    using Pointer = std::unique_ptr<Iterator>;
    virtual ~Iterator() = default;
    // Return 0 with 1..max_count rows, 0 with an empty batch at EOF, or an
    // IndexError on failure. Enumerate every visible ID exactly once. Either
    // batch representation may include filtered rows; an entirely filtered
    // batch still has a nonzero size and must not signal EOF.
    virtual int next_batch(uint32_t max_count, Batch *out) = 0;
  };

  VectorSource();
  virtual ~VectorSource();

  // Borrowed random-access pointers. Keep all get_vectors() results valid
  // through consumption; fetch_vector results also borrow this memory.
  virtual const void *get_vector(uint32_t node_id) const = 0;

  virtual void get_vectors(const uint32_t *ids, uint32_t count,
                           const void **out) const;

  // Optional full scan. Each call creates an independent cursor. Returning
  // nullptr makes External Flat read registered IDs through get_vectors().
  // The caller binds the source, scan domain and visibility filter to the same
  // stable snapshot for the whole request, including subsequent random reads.
  // A disk source can scan physical blocks behind this interface.
  virtual Iterator::Pointer create_iterator() const;
};

}  // namespace core
}  // namespace zvec
