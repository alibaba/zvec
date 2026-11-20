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
#include <mutex>
#include <utility>
#include <vector>

namespace zvec {
namespace core {

// A dedicated mapping lets final graph compaction return whole tail pages
// without allocating a second edge buffer or retaining them in malloc.
class DiskAnnNeighborBuffer {
 public:
  DiskAnnNeighborBuffer() = default;
  ~DiskAnnNeighborBuffer();
  DiskAnnNeighborBuffer(const DiskAnnNeighborBuffer &) = delete;
  DiskAnnNeighborBuffer &operator=(const DiskAnnNeighborBuffer &) = delete;

  int allocate(size_t bytes);
  int shrink(size_t bytes);
  void clear();

  char *data() {
    return data_;
  }
  const char *data() const {
    return data_;
  }
  // Page-rounded mapped/committed bytes, excluding a Windows reservation.
  size_t size() const {
    return size_;
  }

 private:
  char *data_{nullptr};
  size_t size_{0};
};

// Mutable rows use capacity classes in dedicated mapped chunks. Callers guard
// each node with the graph's existing striped locks; the pool lock only guards
// slot allocation and reclamation, never distance calculations or traversal.
class DiskAnnNeighborPool {
 public:
  DiskAnnNeighborPool() = default;
  ~DiskAnnNeighborPool();
  DiskAnnNeighborPool(const DiskAnnNeighborPool &) = delete;
  DiskAnnNeighborPool &operator=(const DiskAnnNeighborPool &) = delete;

  int reserve(size_t nodes, uint32_t max_degree);
  std::pair<uint32_t, const uint32_t *> get(uint32_t id) const;
  int set(uint32_t id, const uint32_t *neighbors, size_t count);
  int append(uint32_t id, uint32_t neighbor);
  // All graph writers must have joined. Packs each chunk in place, returns
  // unused tail pages, and freezes rows for subsequent dump/retry reads.
  int compact(uint32_t final_max_degree);
  void clear();
  size_t memory_size() const;

 private:
  struct Chunk;
  struct Row;
  Row *acquire(uint32_t id, uint32_t count);
  void release(Row *row);
  void add_available(Chunk *chunk);
  void remove_available(Chunk *chunk);

  static constexpr uint32_t kCapacityStep = 16;
  std::vector<uint8_t *> rows_;
  std::vector<Chunk *> available_;
  Chunk *chunks_{nullptr};
  mutable std::mutex mutex_;
  uint32_t max_degree_{0};
  size_t chunk_bytes_{0};
  size_t mapped_bytes_{0};
  size_t chunk_count_{0};
  bool compacted_{false};
};

}  // namespace core
}  // namespace zvec
