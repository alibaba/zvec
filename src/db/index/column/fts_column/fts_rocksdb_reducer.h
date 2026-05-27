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

#include <memory>
#include <string>
#include <vector>
#include <roaring.hh>
#include <zvec/db/status.h>
#include "db/common/rocksdb_context.h"
#include "db/index/column/fts_column/bm25_scorer.h"
#include "db/index/column/fts_column/fts_types.h"

namespace zvec::fts {

class FtsRocksdbReducer;
using FtsRocksdbReducerPtr = std::shared_ptr<FtsRocksdbReducer>;

/*! FTS RocksDB segment reducer
 *  Merges FTS index data from multiple source segments into one destination
 *  segment, remapping doc_ids and filtering deleted documents.  Reads only
 *  postings_cf (BitPacked) and positions_cf from each source segment; writes
 *  only postings_cf, positions_cf, and stat_cf on the destination side.
 */
class FtsRocksdbReducer {
 public:
  /*! Initialize the reducer with destination column families.
   *  \param field_name          FTS field name (used for stat_cf keys)
   *  \param dst_postings_cf     Destination postings CF (BitPacked output)
   *  \param dst_positions_cf    Destination positions CF (phrase support)
   *  \param dst_stat_cf         Destination segment-stat CF
   *  \return Result<void> on success, or Status on failure
   */
  Result<void> init(const std::string &field_name, RocksdbContext *ctx,
                    rocksdb::ColumnFamilyHandle *dst_postings_cf,
                    rocksdb::ColumnFamilyHandle *dst_positions_cf,
                    rocksdb::ColumnFamilyHandle *dst_stat_cf);

  /*! Clean up internal state. */
  Result<void> cleanup();

  /*! Feed a source segment to be merged.
   *  Segments must be fed in consecutive doc_id order.
   *  \param segment_stats       Stats of the source segment (min/max doc_id)
   *  \param src_ctx             RocksdbContext owning the source CFs
   *  \param src_postings_cf     Source postings CF (must be BitPacked)
   *  \param src_positions_cf    Source positions CF
   *  \return Result<void> on success, or Status on failure
   */
  Result<void> feed(FtsSegmentStats segment_stats, RocksdbContext *src_ctx,
                    rocksdb::ColumnFamilyHandle *src_postings_cf,
                    rocksdb::ColumnFamilyHandle *src_positions_cf);

  /*! Merge fed segments into the destination: per-term BitPacked postings
   *  to dst_postings_cf, doc_ids remapped to the new segment's dense space,
   *  effective total_docs / total_tokens to dst_stat_cf for BM25.
   *
   *  \param delete_row_id_bitmap  Deleted positions in input scan order,
   *      id space [0, Σ stats.doc_count).  For segment i with
   *      scan_offset = Σ_{j<i} stats_j.doc_count, source (i, local) is
   *      filtered iff the bitmap contains (scan_offset + local).
   *      Same bitmap built by SegmentHelper::FilterRecordBatch — sharing
   *      it avoids materializing a per-doc dense rank table.
   */
  Result<void> reduce(const roaring::Roaring &delete_row_id_bitmap);

  /*! No-op: FTS data is written directly during reduce(). */
  Result<void> dump() {
    return {};
  }

 private:
  // Two-pass streaming merge.  Pass 1: collect effective stats.  Pass 2:
  // multi-way merge by term, encode + put one BitPacked posting per term
  // (peak memory bounded by one term's entries).  Both passes take the
  // shared delete bitmap by reference rather than storing it on the
  // reducer so its lifetime stays scoped to reduce().
  Result<void> reduce_postings(const roaring::Roaring &delete_row_id_bitmap);

  // Pass 1: effective_total_docs_ = Σ stats.doc_count - bitmap.cardinality
  // (counts empty docs too, like the mutable indexer); effective_total_tokens_
  // is summed from inline doc_len payloads of surviving docs.
  Result<void> collect_effective_stats(
      const roaring::Roaring &delete_row_id_bitmap);

  // Pass 2: see reduce_postings.  Dense rank looked up on the fly via
  // the file-local dense_rank helper in the .cc.
  Result<void> merge_and_flush_postings(
      const roaring::Roaring &delete_row_id_bitmap);

  // Per-segment positions CF remap (phrase query support).
  Result<void> reduce_positions(uint32_t segment_index,
                                const roaring::Roaring &delete_row_id_bitmap);

  // Write accumulated stats to destination stat CF.
  Result<void> flush_stat(uint64_t total_docs, uint64_t total_tokens);

 private:
  enum State {
    STATE_UNINITED = 0,
    STATE_INITED = 1,
    STATE_FEED = 2,
    STATE_REDUCE = 3,
  };

  std::string field_name_{};

  // RocksdbContext for CF-level operations (get/put/create_iter)
  RocksdbContext *ctx_{nullptr};

  // Destination column families (only the 3 active ones are tracked here;
  // $TF/$MAX_TF/$DOC_LEN dst CFs exist in the RocksDB schema but the reducer
  // never writes them — they will be empty in the output SST).
  rocksdb::ColumnFamilyHandle *dst_postings_cf_{nullptr};
  rocksdb::ColumnFamilyHandle *dst_positions_cf_{nullptr};
  rocksdb::ColumnFamilyHandle *dst_stat_cf_{nullptr};

  // Per-segment source RocksdbContexts, column families and stats (only
  // postings + positions are needed; the empty $TF/$MAX_TF/$DOC_LEN side CFs
  // are not opened here).
  std::vector<FtsSegmentStats> segment_stats_{};
  std::vector<RocksdbContext *> src_ctxs_{};
  std::vector<rocksdb::ColumnFamilyHandle *> src_postings_cfs_{};
  std::vector<rocksdb::ColumnFamilyHandle *> src_positions_cfs_{};

  uint32_t num_segments_{0};

  // Survivor-only stats; fed into scorer_ for block_max_score and written
  // to dst stat_cf.
  uint64_t effective_total_docs_{0};
  uint64_t effective_total_tokens_{0};

  // Precomputed cumsum: scan_offset_per_seg_[i] = Σ_{j<i} stats_j.doc_count.
  std::vector<uint64_t> scan_offset_per_seg_{};

  // BM25 scorer for computing block_max_score during BitPacked encoding.
  // Initialized inside reduce() once effective stats are known.
  BM25ScorerPtr scorer_;

  State state_{STATE_UNINITED};
};

}  // namespace zvec::fts
