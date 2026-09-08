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
#include <string>
#include <zvec/core/framework/index_meta.h>

namespace zvec {
namespace turbo {

//! Optional capability: precomputed residual distance table protocol
//! (consumed by IVF residual search).  Orthogonal to the Quantizer base
//! contract, so callers discover it via dynamic_cast: a null cast means
//! the quantizer lacks the capability and the caller falls back to its
//! default path.
//!
//! A datapoint code is decomposed as c_i + c_m[j_m] where c_i is the
//! owning centroid and c_m[j_m] is the m-th sub-quantizer centroid picked
//! by the code.  Squared distance to a query becomes
//!   d = ||x - c_i||^2 + table[i] + LUT
//! with the first term per-list (computed by the caller), the second term
//! depending only on (i, code) and the third term only on (query, code):
//! table[i] is produced by build_centroid_distance_table() once per index,
//! LUT by quantize_precomputed_query() once per query, and
//! merge_query_distance_table() fuses them into a per-list scan buffer.
class PrecomputeTableQuantizer {
 public:
  virtual ~PrecomputeTableQuantizer() = default;

  //! Build a query-independent distance table from the coarse centroids.
  //! One row per centroid is produced; the table layout is opaque to the
  //! caller and only consumed by quantize_precomputed_query() and
  //! merge_query_distance_table().  May refuse oversized tables by
  //! returning a nonzero error code so the caller falls back to its
  //! default path.
  virtual int build_centroid_distance_table(const void *centroids,
                                            size_t centroid_num,
                                            std::string *table) const = 0;

  //! Per-query step paired with build_centroid_distance_table(): build the
  //! query-side distance table once per query, independent of the centroid
  //! being scanned.
  virtual int quantize_precomputed_query(const void *query,
                                         const core::IndexQueryMeta &qmeta,
                                         std::string *out,
                                         core::IndexQueryMeta *ometa) const = 0;

  //! Merge the query-side table with the precomputed row of the
  //! centroid_id-th centroid into a buffer compatible with
  //! Quantizer::calc_distance_dp_query_batch().  The produced distances
  //! exclude the query-to-centroid term; the caller must add it back.
  virtual int merge_query_distance_table(const void *query_table,
                                         const std::string &centroid_table,
                                         size_t centroid_id,
                                         std::string *out) const = 0;
};

}  // namespace turbo
}  // namespace zvec
