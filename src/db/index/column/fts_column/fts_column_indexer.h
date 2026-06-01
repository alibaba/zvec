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

#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <zvec/db/schema.h>
#include <zvec/db/status.h>
#include "db/common/rocksdb_context.h"
#include "db/index/column/fts_column/fts_types.h"
#include "iterator/fts_doc_iterator.h"
#include "tokenizer/tokenizer_factory.h"
#include "bm25_scorer.h"
#include "fts_query_ast.h"


namespace zvec::fts {

/*! Single document in FTS query results.
 *
 *  Note: `doc_id` here is the GLOBAL doc_id */
struct FtsResult {
  uint64_t doc_id{0};
  float score{0.0f};

  bool operator>(const FtsResult &other) const {
    return score > other.score;
  }
};

/*! FTS column indexer
 *  Handles both read (search with BM25 + WAND) and write (insert / flush)
 *  operations on a single FTS column backed by RocksDB.
 *  Uses cross-CF WriteBatch to batch all per-document writes into a single
 *  atomic RocksDB Write() call for optimal write throughput.
 */
class FtsColumnIndexer {
 public:
  FtsColumnIndexer() = default;
  ~FtsColumnIndexer();

  // -----------------------------------------------------------------
  // Initialization
  // -----------------------------------------------------------------

  /*! Initialize for read+write (mutable path).
   *  \param field_meta    Field meta describing this FTS field; provides both
   *                       the field name and the tokenizer extra params used
   *                       to acquire/release the shared pipeline.
   *  \param ctx           RocksdbContext pointer
   *  \param postings_cf   postings CF (main CF)
   *  \param positions_cf  $POS CF
   *  \param term_freq_cf  $TF CF
   *  \param max_tf_cf     $MAX_TF CF
   *  \param doc_len_cf    $DOC_LEN CF
   *  \param stat_cf       $SEGMENT_STAT CF
   *  \return Result<void> on success, or Status on failure
   */
  Result<void> open(FieldSchema::Ptr field_meta, RocksdbContext *ctx,
                    rocksdb::ColumnFamilyHandle *postings_cf,
                    rocksdb::ColumnFamilyHandle *positions_cf,
                    rocksdb::ColumnFamilyHandle *term_freq_cf,
                    rocksdb::ColumnFamilyHandle *max_tf_cf,
                    rocksdb::ColumnFamilyHandle *doc_len_cf,
                    rocksdb::ColumnFamilyHandle *stat_cf);

  /*! Initialize for read-only (immutable / standalone reader path).
   *  No tokenizer is acquired; insert() will fail if called.
   *  \param field_name   Field name
   *  \param ctx          RocksdbContext pointer
   *  \param postings_cf  postings CF
   *  \param positions_cf $POS CF
   *  \param term_freq_cf $TF CF  (may be nullptr for immutable)
   *  \param max_tf_cf    $MAX_TF CF (may be nullptr)
   *  \param doc_len_cf   $DOC_LEN CF (may be nullptr)
   *  \param stat_cf      $SEGMENT_STAT CF
   *  \param bm25_params  BM25 parameters (k1, b)
   *  \return Result<void> on success, or Status on failure
   */
  Result<void> open_reader(const std::string &field_name, RocksdbContext *ctx,
                           rocksdb::ColumnFamilyHandle *postings_cf,
                           rocksdb::ColumnFamilyHandle *positions_cf,
                           rocksdb::ColumnFamilyHandle *term_freq_cf,
                           rocksdb::ColumnFamilyHandle *max_tf_cf,
                           rocksdb::ColumnFamilyHandle *doc_len_cf,
                           rocksdb::ColumnFamilyHandle *stat_cf,
                           BM25Params bm25_params = BM25Params{});

  /*! Release all CF pointers and reset internal state.
   *  Must be called before the underlying RocksdbStore is closed.
   *  The caller is responsible for ensuring no concurrent search() or
   *  reset_side_cfs() call is in flight — this method does NOT drain
   *  or wait for them.
   *  \return Result<void> on success, or Status on failure (e.g. already
   *  closed).
   */
  Result<void> close();

  // -----------------------------------------------------------------
  // Query
  // -----------------------------------------------------------------

  /*! Execute FTS query and return result list with BM25 scores
   *  \param ast          Pre-parsed FTS AST (caller owns the parse step)
   *  \param query_params Query parameters (topk, filter, etc.)
   *  \return Result containing sorted results (descending score), or Status
   */
  Result<std::vector<FtsResult>> search(
      const FtsAstNode &ast, const FtsQueryParams &query_params) const;

  /*! Atomically reset $TF/$MAX_TF/$DOC_LEN CF pointers to nullptr.
   *  Called before dropping these CFs so that concurrent search() calls
   *  on the Roaring path gracefully degrade (return default tf=1/doc_len=1).
   */
  void reset_side_cfs();

  // -----------------------------------------------------------------
  // Write
  // -----------------------------------------------------------------

  /*! Insert FTS field content for a document
   *  \param seg_doc_id  Segment-local document ID
   *  \param text        UTF-8 encoded text content
   *  \return Result<void> on success, or Status on failure
   */
  Result<void> insert(uint64_t seg_doc_id, const std::string &text);

  /*! Flush in-memory statistics to RocksDB (called before segment dump)
   *  \return Result<void> on success, or Status on failure
   */
  Result<void> flush();

  /*! Convert all Roaring-format postings in postings_cf to BitPacked format
   *  with inline tf/doc_len/max_score payloads, then DeleteRange-clear the
   *  $TF, $DOC_LEN, and $MAX_TF CFs.
   *
   *  Called by MutableSegment::dump_fts_column_indexers() right before the
   *  SST dump.  After all indexers finish conversion, MutableSegment drops
   *  the $TF/$MAX_TF/$DOC_LEN CFs entirely (via reset_side_cfs() +
   *  RocksdbStore::drop_column_family()), so the dumped immutable segment
   *  no longer contains these CFs at all.
   *
   *  Idempotent: terms whose postings are already in BitPacked format are
   *  skipped, so re-running after a partial-failure dump is safe.
   *
   *  Must be called after flush() so that the BM25 scorer used by encode()
   *  sees the up-to-date segment statistics.
   *
   *  \return Result<void> on success, or Status on failure
   */
  Result<void> convert_postings_to_bitpacked();

  uint64_t total_docs() const {
    return total_docs_.load(std::memory_order_relaxed);
  }
  uint64_t total_tokens() const {
    return total_tokens_.load(std::memory_order_relaxed);
  }

  // Accessors used by the compaction-time FTS reducer to feed source segments
  // (postings + positions) without going through the higher-level search path.
  RocksdbContext *ctx() const {
    return ctx_;
  }
  rocksdb::ColumnFamilyHandle *postings_cf() const {
    return postings_cf_;
  }
  rocksdb::ColumnFamilyHandle *positions_cf() const {
    return positions_cf_;
  }

 private:
  // --- Iterator tree construction (search internals) ---
  Result<DocIteratorPtr> build_iterator(const FtsAstNode &node) const;
  Result<DocIteratorPtr> build_term_iterator(const TermNode &term_node) const;
  Result<DocIteratorPtr> build_phrase_iterator(
      const PhraseNode &phrase_node) const;
  Result<DocIteratorPtr> build_and_iterator(const AndNode &and_node) const;
  Result<DocIteratorPtr> build_or_iterator(const OrNode &or_node) const;
  Result<DocIteratorPtr> create_term_iterator_from_raw(
      const std::string &term, rocksdb::PinnableSlice raw_data,
      float boost = 1.0f) const;
  std::vector<rocksdb::PinnableSlice> batch_get_postings(
      const std::vector<rocksdb::Slice> &terms) const;

  // --- Write helpers ---
  static void encode_varint(uint32_t value, std::string *output);
  static std::string encode_positions(const std::vector<uint32_t> &positions);

  // --- Tokenizer (write path only) ---
  FieldSchema::Ptr field_meta_{};
  TokenizerPipelinePtr tokenizer_pipeline_{nullptr};
  std::shared_ptr<zvec::FtsIndexParams> fts_params_;

  // --- Reader state ---
  std::string field_name_;
  RocksdbContext *ctx_{nullptr};
  BM25ScorerPtr scorer_;

  rocksdb::ColumnFamilyHandle *postings_cf_{nullptr};
  rocksdb::ColumnFamilyHandle *positions_cf_{nullptr};
  std::atomic<rocksdb::ColumnFamilyHandle *> term_freq_cf_{nullptr};
  std::atomic<rocksdb::ColumnFamilyHandle *> max_tf_cf_{nullptr};
  std::atomic<rocksdb::ColumnFamilyHandle *> doc_len_cf_{nullptr};
  mutable std::atomic<int> cf_counter_{0};
  std::atomic<bool> cf_dropped_{false};
  rocksdb::ColumnFamilyHandle *stat_cf_{nullptr};

  // Minimum doc length observed so far. Used as a (loose) lower bound on
  // doc_len when computing the WAND max_score for Roaring-format postings.
  std::atomic<uint32_t> min_doc_len_{std::numeric_limits<uint32_t>::max()};

  std::atomic<bool> opened_{false};

  // --- Write-path statistics ---
  std::atomic<uint64_t> total_docs_{0};
  std::atomic<uint64_t> total_tokens_{0};
};

using FtsColumnIndexerPtr = std::shared_ptr<FtsColumnIndexer>;

}  // namespace zvec::fts
