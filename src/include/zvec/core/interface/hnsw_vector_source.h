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

namespace zvec {
namespace core {

//! External vector source for the HNSW "external" storage mode. The caller
//! derives from this class and owns a single contiguous block of vector
//! memory, laid out by HNSW internal node_id (i.e. the vector of node_id
//! resides at base + node_id * stride). HNSW only reads vectors through this
//! interface; it never copies or writes the vector data. Used by
//! HnswExternalStreamerEntity so that the index stores only the graph
//! structure (key + neighbors) and reads vectors from the externally-provided
//! memory.
//!
//! This is a self-contained, dependency-free header so it can be shared by
//! both the public interface layer (index.h) and the internal HNSW algorithm
//! layer (hnsw_entity.h) without leaking other internal details.
//!
//! Caller usage pattern:
//! 1. Enable external-vector mode at streamer init by setting param
//!    "proxima.hnsw.streamer.use_external_vector" = true.
//! 2. Implement a vector source whose addressing matches HNSW node_id, e.g.:
//!
//!      class MyVecStore : public zvec::core::HnswVectorSource {
//!       public:
//!        const void *get_vector(uint32_t id) const override {
//!          return base_ + static_cast<size_t>(id) * stride_;
//!        }
//!        const char *base_;  // contiguous buffer owned by the caller
//!        size_t stride_;     // bytes per vector (== dim * element_size)
//!      };
//!
//! 3. Bind it via HNSWIndex::SetVectorSource(&store) before add/search.
//! 4. Vectors are NOT persisted; after re-opening the index, reconstruct the
//!    vector source and bind it again.
//!
//! Lifetime: the vector source object and its memory must outlive any add/search
//! call that uses it. Under concurrent search, each call binds its own source
//! (the binding is per-context, applied before the algorithm runs).
class HnswVectorSource {
 public:
  virtual ~HnswVectorSource() = default;

  //! Return the read-only start address of the vector for node_id; an
  //! invalid id must return nullptr.
  //! Contract: the returned pointer must stay valid for the entire duration
  //! of the add/search call that uses it.
  virtual const void *get_vector(uint32_t node_id) const = 0;

  //! Optional batch hook; defaults to calling get_vector one by one. The
  //! caller may override it to do prefetching / contiguous optimization.
  virtual void get_vectors(const uint32_t *ids, uint32_t count,
                           const void **out) const {
    for (uint32_t i = 0; i < count; ++i) {
      out[i] = get_vector(ids[i]);
    }
  }
};

}  // namespace core
}  // namespace zvec
