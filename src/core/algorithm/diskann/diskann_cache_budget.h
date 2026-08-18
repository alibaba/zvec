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
#include <limits>
#include <utility>
#include <zvec/core/framework/index_meta.h>
#include "diskann_entity.h"

namespace zvec {
namespace core {

/*! Converts a nominal DiskANN node-cache budget into a node count.
 *
 * The payload allocations are exact: one original vector and
 * (max_degree + 1) neighbor slots per cached node. The search path also keeps
 * two std::map entries per node. Their allocator-specific node size is not
 * portable, so the estimate includes both value types and four tree pointers
 * per entry. The configured value is therefore a conservative, portable
 * sizing budget rather than an exact RSS limit.
 */
class DiskAnnCacheBudget {
 public:
  using CoordCacheValue = std::pair<const diskann_id_t, void *>;
  using NeighborValue = std::pair<uint32_t, diskann_id_t *>;
  using NeighborCacheValue = std::pair<const diskann_id_t, NeighborValue>;

  static uint64_t PayloadBytesPerNode(const IndexMeta &meta,
                                      uint32_t max_degree) {
    return static_cast<uint64_t>(meta.element_size()) +
           (static_cast<uint64_t>(max_degree) + 1) * sizeof(diskann_id_t);
  }

  static uint64_t EstimatedBytesPerNode(const IndexMeta &meta,
                                        uint32_t max_degree) {
    constexpr uint64_t kTreePointersPerEntry = 4;
    const uint64_t map_bookkeeping = sizeof(CoordCacheValue) +
                                     sizeof(NeighborCacheValue) +
                                     2 * kTreePointersPerEntry * sizeof(void *);
    return PayloadBytesPerNode(meta, max_degree) + map_bookkeeping;
  }

  static uint64_t MaxNodeCount(uint64_t doc_count) {
    if (doc_count == 0) {
      return 0;
    }

    // Preserve the existing static-cache safety cap: at most 10% of the
    // index, rounded to the nearest node (and at least one for a non-empty
    // index). Keep the arithmetic integral so very large document counts do
    // not lose precision through a floating-point conversion.
    const uint64_t rounded_ten_percent =
        doc_count / 10 + (doc_count % 10 >= 5 ? 1 : 0);
    return std::max<uint64_t>(1, rounded_ten_percent);
  }

  static uint32_t ResolveNodeCount(uint64_t budget_bytes, uint64_t doc_count,
                                   uint64_t estimated_bytes_per_node) {
    if (budget_bytes == 0 || doc_count == 0 || estimated_bytes_per_node == 0) {
      return 0;
    }

    const uint64_t budget_nodes = budget_bytes / estimated_bytes_per_node;
    const uint64_t ten_percent = MaxNodeCount(doc_count);
    const uint64_t addressable_nodes =
        static_cast<uint64_t>(std::numeric_limits<size_t>::max()) /
        estimated_bytes_per_node;
    const uint64_t resolved =
        std::min({budget_nodes, doc_count, ten_percent, addressable_nodes,
                  static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())});
    return static_cast<uint32_t>(resolved);
  }
};

}  // namespace core
}  // namespace zvec
