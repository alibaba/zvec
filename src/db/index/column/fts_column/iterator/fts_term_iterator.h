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
#include <string>
#include <roaring/roaring.h>
#include <rocksdb/slice.h>
#include "db/common/rocksdb_context.h"
#include "fts_doc_iterator.h"
#include "../bm25_scorer.h"
#include "../posting/bitpacked_posting_list.h"

namespace zvec::fts {

/*! Term document iterator
 *  Supports two internal modes:
 *    1. Roaring mode: sorted doc_id array + RocksDB Get for tf/doc_len
 *    2. BitPacked mode: inline payloads, zero RocksDB I/O for score()
 */
class TermDocIterator : public DocIterator {
 public:
  /*! Roaring Bitmap mode constructor.
   *  Takes ownership of the bitmap and iterates lazily via
   *  roaring_uint32_iterator_t — no N×4-byte doc_id array is materialised.
   *
   *  \param term           Processed (tokenized) term string
   *  \param bitmap         Deserialized Roaring bitmap (ownership transferred)
   *  \param df             Document frequency of this term in the segment
   *  \param scorer         BM25 scorer (with segment stats loaded)
   *  \param max_score_val  Precomputed WAND upper bound score for this term
   *                        (caller must NOT pre-multiply by boost — the
   *                        constructor applies boost to both score() output
   *                        and max_score_val_ to keep WAND pivot correct)
   *  \param term_freq_cf   $TF column family for reading per-doc term freq
   *  \param doc_len_cf     $DOC_LEN column family for reading doc length
   *  \param cf_counter     CF reference counter for term_freq_cf and doc_len_cf
   *  \param boost          Per-term boost (1.0 = no boost)
   */
  TermDocIterator(std::string term, roaring_bitmap_t *bitmap, uint64_t df,
                  BM25ScorerPtr scorer, float max_score_val,
                  RocksdbContext *ctx,
                  rocksdb::ColumnFamilyHandle *term_freq_cf,
                  rocksdb::ColumnFamilyHandle *doc_len_cf,
                  std::atomic<int> *cf_counter, float boost = 1.0f);

  ~TermDocIterator() override;

  /*! BitPacked mode constructor.
   *  All payloads (tf, doc_len, per-block max_score, global max_score) are
   *  embedded inline in packed_data, so this iterator is completely
   *  self-contained on the read path:
   *    - score() reads tf/doc_len from bp_iter_ — zero RocksDB I/O.
   *    - block_max_info_for() / max_score() all read from the BitPacked
   *      skip-list / block headers — no $MAX_TF lookup needed.
   *  Construction takes neither $TF, $DOC_LEN, nor $MAX_TF column families:
   *  the immutable segment SST may have these CFs entirely empty (cleared
   *  by FtsColumnIndexer::convert_postings_to_bitpacked at dump time) and
   *  this iterator still works correctly.
   *
   *  df and max_score are read from bp_iter_ after open(); on open failure
   *  cost() returns 0 and callers should treat the iterator as empty.
   *
   *  \param term           Processed (tokenized) term string
   *  \param packed_data    Serialized BitPacked posting list (ownership taken)
   *  \param scorer         BM25 scorer (with segment stats loaded)
   *  \param boost          Per-term boost (1.0 = no boost)
   */
  TermDocIterator(std::string term, rocksdb::PinnableSlice packed_data,
                  BM25ScorerPtr scorer, float boost = 1.0f);

  // Prevent move/copy: bp_iter_ holds a raw pointer into packed_data_'s
  // buffer, so moving would create a dangling pointer.
  TermDocIterator(const TermDocIterator &) = delete;
  TermDocIterator &operator=(const TermDocIterator &) = delete;
  TermDocIterator(TermDocIterator &&) = delete;
  TermDocIterator &operator=(TermDocIterator &&) = delete;

  uint32_t next_doc() override;
  uint32_t advance(uint32_t target) override;
  float score() override;
  uint64_t cost() const override;
  float max_score() const override {
    return max_score_val_;
  }

  // Block-Max WAND support (only effective in BitPacked mode)
  BlockMaxInfo block_max_info_for(uint32_t target) const override;

 private:
  // Read term frequency for the current document (Roaring mode only)
  uint32_t read_term_freq(uint32_t doc_id) const;

  // Read document length for the current document (Roaring mode only)
  uint32_t read_doc_len(uint32_t doc_id) const;

 private:
  enum class Mode { ROARING, BITPACKED };
  Mode mode_;

  std::string term_;
  uint64_t df_;
  BM25ScorerPtr scorer_;
  float max_score_val_;
  float idf_weight_{0.0f};  // Pre-computed IDF to avoid log() per score()
  float boost_{1.0f};       // Per-term boost (collapsed from repeated terms)

  // Roaring mode state (owns the bitmap; iterator is stack-allocated)
  roaring_bitmap_t *bitmap_{nullptr};
  roaring_uint32_iterator_t roaring_iter_{};
  bool roaring_iter_started_{false};  // tracks whether first next_doc called
  RocksdbContext *ctx_{nullptr};
  rocksdb::ColumnFamilyHandle *term_freq_cf_{nullptr};
  rocksdb::ColumnFamilyHandle *doc_len_cf_{nullptr};
  std::atomic<int> *cf_counter_{nullptr};

  // BitPacked mode state
  rocksdb::PinnableSlice packed_data_;  // owns the serialized data (zero-copy)
  BitPackedPostingIterator bp_iter_;    // zero-copy iterator over packed_data_
};

}  // namespace zvec::fts
