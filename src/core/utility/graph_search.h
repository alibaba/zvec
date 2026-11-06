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
#include <cstdint>
#include <vector>

namespace zvec::core {

struct GraphSearchScratch {
  std::vector<uint32_t> ids;
  std::vector<float> distances;

  void reserve_batch(size_t capacity) {
    if (ids.size() < capacity) ids.resize(capacity);
    if (distances.size() < capacity) distances.resize(capacity);
  }
};

// Frontier for an exact-distance LinearPool or BlockHeap. The caller seeds the
// pool and marks the entry visited. Its existing termination rule is preserved.
template <class Pool>
struct GraphSearchPoolFrontier {
  Pool &pool;
  bool has_next() const {
    return pool.has_next();
  }
  uint32_t pop() {
    return static_cast<uint32_t>(pool.pop());
  }
  void push_batch(const uint32_t *ids, const float *distances, size_t count) {
    pool.push_block(distances, ids, static_cast<int32_t>(count));
  }
};

// Shared traversal for exact and quantized graph search. All policies are
// statically dispatched. The caller seeds the frontier and initializes visits.
//
// Scan owns/pins the current adjacency via begin(id), exposes neighbor_count(),
// neighbor(i), and batch_size(), and supplies prepare_batch/stage/score.
// score(offset, ids, count, distances) must preserve lane order. It may write
// a full batch, including padded lanes; scratch always reserves batch_size().
// begin and score return zero on success or a caller-defined error code.
//
// Exact scans mark discovery before scoring and submit one compact batch to
// Frontier::push_batch. Quantized scans keep all lanes for encoding alignment,
// mark expansion, and admit candidates individually through accepts()/push().
// This permits improved center-dependent estimates before a node is expanded.
// The frontier owns stopping rules; estimated scores are never compared to an
// exact result threshold by this common loop. Results/filtering/completion are
// policy responsibilities, so excluded nodes can still serve as graph bridges.

template <class Scan, class Frontier, class Visit>
int SearchGraph(Scan &scan, Frontier &frontier, Visit &visit,
                GraphSearchScratch &scratch) {
  while (frontier.has_next()) {
    const uint32_t id = frontier.pop();
    if constexpr (Scan::kVisitOnExpansion) {
      if (visit.visited(id)) continue;
      visit.set_visited(id);
    }
    const int ret = scan.begin(id);
    if (ret != 0) return ret;
    const size_t size = scan.neighbor_count();
    if (size == 0) continue;
    const size_t capacity = scan.batch_size();  // positive for a nonempty row
    scratch.reserve_batch(capacity);
    scan.prepare_batch(capacity);
    for (size_t offset = 0; offset < size; offset += capacity) {
      const size_t lanes = std::min(capacity, size - offset);
      size_t count = 0;
      for (size_t i = 0; i < lanes; ++i) {
        const uint32_t neighbor = scan.neighbor(offset + i);
        if constexpr (!Scan::kVisitOnExpansion) {
          if (visit.visited(neighbor)) {
            scan.on_duplicate();
            continue;
          }
          visit.set_visited(neighbor);
        }
        scratch.ids[count] = neighbor;
        scan.stage(neighbor, offset + i, count);
        ++count;
      }
      if (count == 0) continue;
      const int error = scan.score(offset, scratch.ids.data(), count,
                                   scratch.distances.data());
      if (error != 0) return error;
      if constexpr (Scan::kVisitOnExpansion) {
        for (size_t i = 0; i < count; ++i) {
          const float distance = scratch.distances[i];
          // Admission is sequential: each insertion may tighten the beam.
          if (!frontier.accepts(distance)) continue;
          const uint32_t neighbor = scratch.ids[i];
          if (!visit.visited(neighbor)) frontier.push(neighbor, distance);
        }
      } else {
        frontier.push_batch(scratch.ids.data(), scratch.distances.data(),
                            count);
      }
    }
  }
  return 0;
}

}  // namespace zvec::core
