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

#include <cstdint>
#include <memory>
#include <vector>
#include <zvec/export.h>

namespace zvec {
namespace core {

class ZVEC_CORE_API VectorSource {
 public:
  // Logical row-major vectors in the same ID namespace as get_vector().
  // Pointers need not be contiguous. External Flat currently expects raw FP32.
  struct Batch {
    std::vector<uint32_t> ids;
    std::vector<const void *> vectors;
    // Optional backing allocation or page pin. Data must remain valid until
    // the batch is released, or (for scans) the next next_batch() call.
    std::shared_ptr<const void> lease;

    void clear() {
      vectors.clear();
      ids.clear();
      lease.reset();
    }
  };

  class Iterator {
   public:
    using Pointer = std::unique_ptr<Iterator>;
    virtual ~Iterator() = default;
    // Return 0 with 1..max_count rows, 0 with an empty batch at EOF, or an
    // IndexError on failure. Enumerate every visible ID exactly once.
    virtual int next_batch(uint32_t max_count, Batch *out) = 0;
  };

  VectorSource();
  virtual ~VectorSource();

  virtual const void *get_vector(uint32_t node_id) const = 0;

  virtual void get_vectors(const uint32_t *ids, uint32_t count,
                           const void **out) const;

  // Optional full scan. Each call creates an independent cursor. The caller
  // holds a stable source snapshot for the whole request, including subsequent
  // random reads. A disk source can scan physical blocks behind this interface.
  virtual Iterator::Pointer create_iterator() const;

  // Return IDs in input order. The default borrows get_vectors() pointers,
  // which must all remain valid until the caller consumes the batch. Sources
  // with transient buffers should override this method and attach a lease.
  virtual int get_vector_batch(const uint32_t *ids, uint32_t count,
                               Batch *out) const;
};

}  // namespace core
}  // namespace zvec
