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

#include "fts_term_iterator.h"
#include <cstring>
#include <roaring/roaring.h>
#include <zvec/ailego/logger/logger.h>
#include "fts_utils.h"

namespace zvec::fts {

// ============================================================
// Constructors
// ============================================================

// Roaring Bitmap mode — takes ownership of bitmap, iterates lazily.
TermDocIterator::TermDocIterator(std::string term, roaring_bitmap_t *bitmap,
                                 uint64_t df, BM25ScorerPtr scorer,
                                 float max_score_val, RocksdbContext *ctx,
                                 rocksdb::ColumnFamilyHandle *term_freq_cf,
                                 rocksdb::ColumnFamilyHandle *doc_len_cf,
                                 std::atomic<int> *cf_counter)
    : mode_(Mode::ROARING),
      term_(std::move(term)),
      df_(df),
      scorer_(std::move(scorer)),
      max_score_val_(max_score_val),
      bitmap_(bitmap),
      ctx_(ctx),
      term_freq_cf_(term_freq_cf),
      doc_len_cf_(doc_len_cf),
      cf_counter_(cf_counter) {
  roaring_init_iterator(bitmap_, &roaring_iter_);
}

TermDocIterator::~TermDocIterator() {
  if (bitmap_) {
    roaring_bitmap_free(bitmap_);
    bitmap_ = nullptr;
  }
  if (cf_counter_) {
    --*cf_counter_;
  }
}

// BitPacked mode
TermDocIterator::TermDocIterator(std::string term, std::string packed_data,
                                 uint64_t df, BM25ScorerPtr scorer,
                                 float max_score_val)
    : mode_(Mode::BITPACKED),
      term_(std::move(term)),
      df_(df),
      scorer_(std::move(scorer)),
      max_score_val_(max_score_val),
      packed_data_(std::move(packed_data)) {
  // Failure here means the term will produce no docs (next_doc returns
  // NO_MORE_DOCS). bp_iter_.open() already logs the underlying parse error;
  // surface it once more here with the term context for easier triage.
  if (bp_iter_.open(packed_data_.data(), packed_data_.size()) != 0) {
    LOG_ERROR(
        "TermDocIterator: failed to open bitpacked posting for term[%s], "
        "iterator will yield no documents",
        term_.c_str());
  }
}

// ============================================================
// Iterator interface
// ============================================================

uint32_t TermDocIterator::next_doc() {
  if (mode_ == Mode::BITPACKED) {
    current_doc_id_ = bp_iter_.next_doc();
    return current_doc_id_;
  }

  // Roaring mode: stream via roaring_uint32_iterator_t
  if (!roaring_iter_started_) {
    // First call: iterator already points at the first element after
    // roaring_init_iterator in the constructor.
    roaring_iter_started_ = true;
  } else {
    roaring_advance_uint32_iterator(&roaring_iter_);
  }
  if (!roaring_iter_.has_value) {
    current_doc_id_ = NO_MORE_DOCS;
    return NO_MORE_DOCS;
  }
  current_doc_id_ = roaring_iter_.current_value;
  return current_doc_id_;
}

uint32_t TermDocIterator::advance(uint32_t target) {
  if (mode_ == Mode::BITPACKED) {
    current_doc_id_ = bp_iter_.advance(target);
    return current_doc_id_;
  }

  // Roaring mode: skip to the first doc_id >= target
  roaring_iter_started_ = true;
  if (!roaring_move_uint32_iterator_equalorlarger(&roaring_iter_, target)) {
    current_doc_id_ = NO_MORE_DOCS;
    return NO_MORE_DOCS;
  }
  current_doc_id_ = roaring_iter_.current_value;
  return current_doc_id_;
}

float TermDocIterator::score() {
  if (current_doc_id_ == NO_MORE_DOCS) {
    return 0.0f;
  }

  if (mode_ == Mode::BITPACKED) {
    // Fast path: read tf/doc_len from inline payload (zero I/O)
    const uint32_t tf = bp_iter_.term_freq();
    const uint32_t dl = bp_iter_.doc_len();
    return scorer_->score(df_, tf, dl);
  }

  // Roaring mode: read from RocksDB
  const uint32_t tf = read_term_freq(current_doc_id_);
  const uint32_t doc_len = read_doc_len(current_doc_id_);
  return scorer_->score(df_, tf, doc_len);
}

uint64_t TermDocIterator::cost() const {
  if (mode_ == Mode::BITPACKED) {
    return bp_iter_.cost();
  }
  return df_;
}

// ============================================================
// Block-Max WAND support
// ============================================================

float TermDocIterator::current_block_max_score() const {
  if (mode_ == Mode::BITPACKED) {
    return bp_iter_.current_block_max_score();
  }
  // Roaring mode: fall back to global max_score (no block-level info)
  return max_score_val_;
}

uint32_t TermDocIterator::skip_to_next_block() {
  if (mode_ == Mode::BITPACKED) {
    current_doc_id_ = bp_iter_.skip_to_next_block();
    return current_doc_id_;
  }
  // Roaring mode: no block structure, just advance to next doc
  return next_doc();
}

float TermDocIterator::block_max_score_for(uint32_t target) const {
  if (mode_ == Mode::BITPACKED) {
    return bp_iter_.block_max_score_for(target);
  }
  // Roaring mode: fall back to global max_score (no block-level info)
  return max_score_val_;
}

uint32_t TermDocIterator::block_max_last_doc_for(uint32_t target) const {
  if (mode_ == Mode::BITPACKED) {
    return bp_iter_.block_max_last_doc_for(target);
  }
  // Roaring mode: no block structure
  return NO_MORE_DOCS;
}

DocIterator::BlockMaxInfo TermDocIterator::block_max_info_for(
    uint32_t target) const {
  if (mode_ == Mode::BITPACKED) {
    auto info = bp_iter_.block_max_info_for(target);
    return {info.block_max_score, info.block_last_doc};
  }
  // Roaring mode: fall back to global max_score, no block structure
  return {max_score_val_, NO_MORE_DOCS};
}

// ============================================================
// Roaring mode helpers
// ============================================================

uint32_t TermDocIterator::read_term_freq(uint32_t doc_id) const {
  if (!term_freq_cf_) {
    return 1;  // CF dropped after convert_postings_to_bitpacked
  }
  const std::string key = fts::make_doc_term_key(term_, doc_id);
  std::string value;
  if (!ctx_->db_->Get(ctx_->read_opts_, term_freq_cf_, key, &value).ok() ||
      value.size() < sizeof(uint32_t)) {
    return 1;  // Default term frequency is 1
  }
  uint32_t tf = 0;
  std::memcpy(&tf, value.data(), sizeof(uint32_t));
  return tf;
}

uint32_t TermDocIterator::read_doc_len(uint32_t doc_id) const {
  if (!doc_len_cf_) {
    return 1;  // CF dropped after convert_postings_to_bitpacked
  }
  std::string doc_id_key(sizeof(uint32_t), '\0');
  std::memcpy(doc_id_key.data(), &doc_id, sizeof(uint32_t));

  std::string value;
  if (!ctx_->db_->Get(ctx_->read_opts_, doc_len_cf_, doc_id_key, &value).ok() ||
      value.size() < sizeof(uint32_t)) {
    return 1;  // Default document length is 1
  }
  uint32_t doc_len = 0;
  std::memcpy(&doc_len, value.data(), sizeof(uint32_t));
  return doc_len;
}

}  // namespace zvec::fts
