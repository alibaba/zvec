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

  /*! Merge all fed segments into the destination store.
   *  Reads BitPacked posting lists from each source postings_cf, applies
   *  the delete filter, remaps doc_ids, and emits one merged BitPacked
   *  posting list per term to dst_postings_cf.  Also accumulates effective
   *  total_docs / total_tokens from inline doc_len payloads and writes them
   *  to dst_stat_cf for BM25 IDF / avgdl.
   *
   *  \param filter   Returns true for doc_ids that should be filtered out
   *                  (i.e., deleted documents).
   *  \return Result<void> on success, or Status on failure
   */
  Result<void> reduce(const IndexFilter &filter);

  /*! No-op: FTS data is written directly during reduce(). */
  Result<void> dump() {
    return {};
  }

 private:
  // Two-pass streaming merge of postings.  Pass 1 collects effective stats
  // without storing any PostingEntry; Pass 2 does multi-way merge across all
  // source segment iterators by term (lexicographic order), encodes + puts
  // each term's merged BitPacked posting list immediately, keeping peak
  // memory at one term's worth of entries.
  Result<void> reduce_postings(const IndexFilter &filter);

  // Pass 1: collect effective_total_docs_ / effective_total_tokens_ without
  // storing any PostingEntry.
  // - effective_total_docs_ is computed from segment doc_id ranges minus
  //   filtered docs (includes empty docs, matching mutable indexer semantics).
  // - effective_total_tokens_ is accumulated from inline doc_len payloads
  //   of surviving docs seen in postings (empty docs contribute 0).
  Result<void> collect_effective_stats(const IndexFilter &filter);

  // Pass 2: multi-way merge across all source segment iterators by term
  // (lexicographic order), accumulate per-term entries, encode + put as
  // BitPacked into dst_postings_cf_ immediately after each term boundary,
  // keeping peak memory at one term's worth of entries.
  Result<void> merge_and_flush_postings(const IndexFilter &filter);

  // Merge positions CF for one source segment: iterate src positions_cf,
  // drop entries whose doc_id is filtered, remap to the new doc_id space,
  // and put into dst_positions_cf.  Required for phrase query support.
  Result<void> reduce_positions(uint32_t segment_index,
                                const IndexFilter &filter);

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
  uint64_t min_doc_id_{0};

  // Effective per-segment statistics accumulated during reduce_postings()
  // from BitPacked inline doc_len payloads.  Reflect only documents that
  // survive the filter, and are used both as the truth fed into scorer_ for
  // block_max_score computation and as the values written into dst stat_cf.
  uint64_t effective_total_docs_{0};
  uint64_t effective_total_tokens_{0};

  // BM25 scorer for computing block_max_score during BitPacked encoding.
  // Initialized inside reduce() once effective stats are known.
  BM25ScorerPtr scorer_;

  State state_{STATE_UNINITED};
};

}  // namespace zvec::fts
