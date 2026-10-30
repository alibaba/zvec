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
#include <variant>
#include <vector>
#include <zvec/export.h>

namespace zvec {
namespace core {

// Borrowed raw FP32 rows in get_vector()'s ID space. Leases may pin storage.
struct PointerBatch {
  std::vector<uint32_t> ids;
  // Only filtered or unregistered rows may have null pointers.
  std::vector<const void *> vectors;
  std::shared_ptr<const void> lease;
};

// Row i has ID first_id + i; stride is in bytes, at least one vector's size.
struct ContiguousBatch {
  const void *data{nullptr};
  uint32_t first_id{0};
  uint32_t count{0};  // Includes filtered rows.
  size_t stride{0};
  std::shared_ptr<const void> lease;
};

using VectorBatch = std::variant<PointerBatch, ContiguousBatch>;

class ZVEC_CORE_API VectorIterator {
 public:
  using Pointer = std::unique_ptr<VectorIterator>;
  virtual ~VectorIterator() = default;
  // Return 0 with 1..max_count rows, 0 with an empty batch at EOF, or an
  // IndexError on failure. Enumerate every visible ID exactly once.
  virtual int next_batch(uint32_t max_count, VectorBatch *out) = 0;
};

class ZVEC_CORE_API VectorSource {
 public:
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
  virtual VectorIterator::Pointer create_iterator() const;
};

class ZVEC_CORE_API ContiguousVectorSource : public VectorSource {
 public:
  // Iterators retain this view and its lease. The caller keeps storage alive
  // for random reads and any borrowed search results.
  virtual ContiguousBatch get_contiguous_view() const = 0;

  const void *get_vector(uint32_t id) const override;

  void get_vectors(const uint32_t *ids, uint32_t count,
                   const void **out) const override;

  VectorIterator::Pointer create_iterator() const override;
};

}  // namespace core
}  // namespace zvec
