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

#include "fts_rocksdb_reducer.h"
#include <cstring>
#include <vector>
#include <zvec/ailego/logger/logger.h>
#include <zvec/db/status.h>
#include "db/index/column/fts_column/bitpacked_posting_list.h"
#include "db/index/column/fts_column/fts_utils.h"

namespace zvec::fts {

// ============================================================
// Design notes
// ============================================================
//
// Every immutable FTS segment stores its data in three CFs:
//   - postings_cf   : term -> BitPacked posting list (inline
//                     tf / doc_len / per-block max_score)
//   - positions_cf  : term\0doc_id -> varint delta-encoded positions
//                     (needed for phrase queries)
//   - stat_cf       : field_name_total_docs / field_name_total_tokens
//
// The reducer performs a multi-way merge of N source segments into one
// destination segment.  It iterates each source segment's BitPacked
// postings_cf, decodes (doc_id, tf, doc_len) triples directly from the
// inline payloads, applies the delete filter, remaps doc_ids to the new
// segment's local range, and emits a single merged BitPacked posting list
// per term into dst_postings_cf.  positions_cf is merged key-by-key for
// phrase support.  stat_cf is recomputed from the surviving docs.
//
// All input postings_cf values must be in BitPacked format.
//
// doc_id encoding contract (aligned with InvertRocksdbStreamer2):
// every src segment's RocksDB stores LOCAL doc_ids, i.e.
//   local_doc_id = global_doc_id - segment_stats[i].min_doc_id
// so that values fit into uint32_t and reduce_* logic can safely
// reconstruct global_doc_id via
//   global_doc_id = stats.min_doc_id + local_doc_id
// and remap into the dst segment local space via
//   new_local_doc_id = global_doc_id - dst_min_doc_id_.
// FtsColumnIndexer::insert() is responsible for storing local doc_id
// (see start_doc_id_ in FtsColumnIndexer).
//
// Two-pass streaming design:
//
// Pass 1 (collect_effective_stats): iterates all source posting lists to
// compute effective_total_docs_ and effective_total_tokens_ WITHOUT
// storing any PostingEntry.
// - effective_total_docs_ is derived from each segment's
//   [min_doc_id, max_doc_id] range minus filtered docs.
// - effective_total_tokens_ is accumulated from inline doc_len payloads
//   of surviving docs (empty docs contribute 0).
// - Per-segment seen-doc dedup uses vector<bool> instead of
//   unordered_set<uint32_t> (~125KB vs ~40MB per million docs).
//
// Pass 2 (merge_and_flush_postings): opens N RocksDB iterators (one per
// source segment) and performs a multi-way merge by term in lexicographic
// order.  For each term, entries from all segments are aggregated into a
// temporary vector, immediately encoded as BitPacked and put to
// dst_postings_cf, then the vector is cleared.  Peak memory is bounded
// by the single largest term's entries rather than all terms combined.
//
// No Roaring intermediate format is involved, and no $TF/$MAX_TF/$DOC_LEN
// side CF is read or written.

// ============================================================
// Public interface
// ============================================================

Result<void> FtsRocksdbReducer::init(
    const std::string &field_name, RocksdbContext *ctx,
    rocksdb::ColumnFamilyHandle *dst_postings_cf,
    rocksdb::ColumnFamilyHandle *dst_positions_cf,
    rocksdb::ColumnFamilyHandle *dst_stat_cf) {
  if (!dst_postings_cf || !dst_positions_cf || !dst_stat_cf) {
    LOG_ERROR(
        "FtsRocksdbReducer init failed: null destination CF for field[%s]",
        field_name.c_str());
    return tl::make_unexpected(Status::InvalidArgument(
        "FtsRocksdbReducer: null destination CF. field=", field_name));
  }

  field_name_ = field_name;
  ctx_ = ctx;
  dst_postings_cf_ = dst_postings_cf;
  dst_positions_cf_ = dst_positions_cf;
  dst_stat_cf_ = dst_stat_cf;

  state_ = STATE_INITED;
  return {};
}

Result<void> FtsRocksdbReducer::cleanup() {
  segment_stats_.clear();
  src_ctxs_.clear();
  src_postings_cfs_.clear();
  src_positions_cfs_.clear();
  num_segments_ = 0;
  state_ = STATE_UNINITED;
  return {};
}

Result<void> FtsRocksdbReducer::feed(
    FtsSegmentStats segment_stats, RocksdbContext *src_ctx,
    rocksdb::ColumnFamilyHandle *src_postings_cf,
    rocksdb::ColumnFamilyHandle *src_positions_cf) {
  if (state_ != STATE_INITED && state_ != STATE_FEED) {
    LOG_ERROR("FtsRocksdbReducer: call init() before feed()");
    return tl::make_unexpected(Status::InternalError(
        "FtsRocksdbReducer: call init() before feed(). field=", field_name_));
  }

  if (!src_postings_cf || !src_positions_cf) {
    LOG_ERROR("FtsRocksdbReducer feed failed: null source CF for field[%s]",
              field_name_.c_str());
    return tl::make_unexpected(Status::InvalidArgument(
        "FtsRocksdbReducer: null source CF. field=", field_name_));
  }

  // Track global min_doc_id from the first segment; require consecutive
  // doc_id ranges across segments so that downstream remap is safe.
  if (segment_stats_.empty()) {
    min_doc_id_ = segment_stats.min_doc_id;
  } else {
    if (segment_stats.min_doc_id != segment_stats_.back().max_doc_id + 1) {
      LOG_ERROR(
          "FtsRocksdbReducer feed failed: segments must be fed in consecutive "
          "doc_id order. field[%s] expected_min[%zu] got[%zu]",
          field_name_.c_str(), (size_t)(segment_stats_.back().max_doc_id + 1),
          (size_t)segment_stats.min_doc_id);
      return tl::make_unexpected(Status::InternalError(
          "FtsRocksdbReducer: segments not in consecutive doc_id order. field=",
          field_name_));
    }
  }

  segment_stats_.emplace_back(std::move(segment_stats));
  src_ctxs_.emplace_back(src_ctx);
  src_postings_cfs_.emplace_back(src_postings_cf);
  src_positions_cfs_.emplace_back(src_positions_cf);
  ++num_segments_;

  state_ = STATE_FEED;
  return {};
}

Result<void> FtsRocksdbReducer::reduce(const IndexFilter &filter) {
  if (state_ != STATE_FEED || num_segments_ == 0) {
    LOG_ERROR("FtsRocksdbReducer: call feed() before reduce(). field[%s]",
              field_name_.c_str());
    return tl::make_unexpected(Status::InternalError(
        "FtsRocksdbReducer: call feed() before reduce(). field=", field_name_));
  }

  effective_total_docs_ = 0;
  effective_total_tokens_ = 0;

  // Phase 1: Streaming per-term merge across all source segments.  Decodes
  // BitPacked postings inline, applies the filter, remaps doc_ids, and
  // emits one merged BitPacked posting list per term to dst_postings_cf.
  // Also accumulates effective_total_docs_ / effective_total_tokens_ from
  // inline doc_len payloads (each surviving doc counted once across all
  // its terms within a segment).
  auto ret = reduce_postings(filter);
  if (!ret) {
    LOG_ERROR("FtsRocksdbReducer: reduce_postings failed. field[%s]",
              field_name_.c_str());
    return ret;
  }

  // Phase 2: Merge positions CF per segment for phrase query support.
  for (uint32_t segment_index = 0; segment_index < num_segments_;
       ++segment_index) {
    ret = reduce_positions(segment_index, filter);
    if (!ret) {
      LOG_ERROR(
          "FtsRocksdbReducer: reduce_positions failed. segment[%u] field[%s]",
          segment_index, field_name_.c_str());
      return ret;
    }
  }

  // Phase 3: Persist effective stats so search-time IDF / avgdl matches the
  // encode-time block_max_score (single source of truth, derived from the
  // documents that actually survived the filter).
  ret = flush_stat(effective_total_docs_, effective_total_tokens_);
  if (!ret) {
    LOG_ERROR("FtsRocksdbReducer: flush_stat failed. field[%s]",
              field_name_.c_str());
    return ret;
  }

  state_ = STATE_REDUCE;
  LOG_INFO(
      "FtsRocksdbReducer: reduce done. field[%s] segments[%u] "
      "effective_docs[%zu] effective_tokens[%zu]",
      field_name_.c_str(), num_segments_, (size_t)effective_total_docs_,
      (size_t)effective_total_tokens_);
  return {};
}

// ============================================================
// Private: streaming postings merge (single stage, BitPacked in/out)
// ============================================================

Result<void> FtsRocksdbReducer::reduce_postings(const IndexFilter &filter) {
  // Pass 1: collect effective stats (no PostingEntry storage).
  auto ret = collect_effective_stats(filter);
  if (!ret) return ret;

  // Initialize BM25 scorer with final effective stats.
  scorer_ = std::make_shared<BM25Scorer>();
  scorer_->update_stats(effective_total_docs_, effective_total_tokens_);

  // Pass 2: multi-way merge + streaming encode/flush.
  return merge_and_flush_postings(filter);
}

// ============================================================
// Private: Pass 1 — collect effective stats without storing entries
// ============================================================

Result<void> FtsRocksdbReducer::collect_effective_stats(
    const IndexFilter &filter) {
  effective_total_docs_ = 0;
  effective_total_tokens_ = 0;

  for (uint32_t seg = 0; seg < num_segments_; ++seg) {
    const auto &stats = segment_stats_[seg];
    const uint64_t seg_doc_count = stats.max_doc_id - stats.min_doc_id + 1;

    // ---------- effective_total_docs_: from doc_id range - filtered ----------
    // Count how many docs in [min_doc_id, max_doc_id] survive the filter.
    // This includes empty docs (no tokens), matching mutable indexer semantics
    // where total_docs_++ on every insert regardless of doc_len.
    uint64_t seg_filtered = 0;
    for (uint64_t gid = stats.min_doc_id; gid <= stats.max_doc_id; ++gid) {
      if (filter.is_filtered(gid)) {
        ++seg_filtered;
      }
    }
    effective_total_docs_ += (seg_doc_count - seg_filtered);

    // ---------- effective_total_tokens_: from posting inline doc_len
    // ---------- Use vector<bool> for per-segment seen-doc dedup (local_doc_id
    // is a contiguous small integer).  Memory: ~125KB per million docs vs ~40MB
    // for unordered_set<uint32_t>.
    const uint64_t local_range = seg_doc_count;
    std::vector<bool> seen_docs(local_range, false);

    auto *src_cf = src_postings_cfs_[seg];
    auto iter = std::unique_ptr<rocksdb::Iterator>(
        src_ctxs_[seg]->db_->NewIterator(src_ctxs_[seg]->read_opts_, src_cf));
    iter->SeekToFirst();

    while (iter->Valid()) {
      const std::string posting_data = iter->value().ToString();

      if (!BitPackedPostingList::is_bitpacked_format(posting_data.data(),
                                                     posting_data.size())) {
        LOG_ERROR(
            "FtsRocksdbReducer: source postings is not BitPacked. "
            "field[%s] segment[%u]",
            field_name_.c_str(), seg);
        return tl::make_unexpected(Status::InternalError(
            "FtsRocksdbReducer: source postings is not BitPacked. field=",
            field_name_));
      }

      BitPackedPostingIterator bp_iter;
      if (bp_iter.open(posting_data.data(), posting_data.size()) != 0) {
        LOG_ERROR(
            "FtsRocksdbReducer: failed to open bitpacked postings. "
            "field[%s] segment[%u]",
            field_name_.c_str(), seg);
        return tl::make_unexpected(Status::InternalError(
            "FtsRocksdbReducer: failed to open bitpacked postings. field=",
            field_name_));
      }

      uint32_t local_doc_id = bp_iter.next_doc();
      while (local_doc_id != BitPackedPostingIterator::NO_MORE_DOCS) {
        const uint64_t global_doc_id =
            stats.min_doc_id + static_cast<uint64_t>(local_doc_id);
        if (!filter.is_filtered(global_doc_id)) {
          if (local_doc_id < local_range && !seen_docs[local_doc_id]) {
            seen_docs[local_doc_id] = true;
            effective_total_tokens_ += bp_iter.doc_len();
          }
        }
        local_doc_id = bp_iter.next_doc();
      }
      iter->Next();
    }
  }

  LOG_INFO(
      "FtsRocksdbReducer: collect_effective_stats done. field[%s] "
      "effective_docs[%zu] effective_tokens[%zu]",
      field_name_.c_str(), (size_t)effective_total_docs_,
      (size_t)effective_total_tokens_);
  return {};
}

// ============================================================
// Private: Pass 2 — multi-way merge + streaming encode/flush
// ============================================================

Result<void> FtsRocksdbReducer::merge_and_flush_postings(
    const IndexFilter &filter) {
  struct PostingEntry {
    uint32_t doc_id;
    uint32_t tf;
    uint32_t doc_len;
  };

  // Open N iterators, one per source segment.
  struct SegmentCursor {
    uint32_t segment_index;
    std::unique_ptr<rocksdb::Iterator> iter;
    const FtsSegmentStats *stats;
  };
  std::vector<SegmentCursor> cursors;
  cursors.reserve(num_segments_);
  for (uint32_t i = 0; i < num_segments_; ++i) {
    auto it = std::unique_ptr<rocksdb::Iterator>(src_ctxs_[i]->db_->NewIterator(
        src_ctxs_[i]->read_opts_, src_postings_cfs_[i]));
    it->SeekToFirst();
    cursors.push_back(SegmentCursor{i, std::move(it), &segment_stats_[i]});
  }

  // Reusable buffers.
  std::vector<PostingEntry> term_entries;
  std::vector<uint32_t> doc_ids_buf, tfs_buf, doc_lens_buf;

  while (true) {
    // Find the lexicographically smallest current term across all cursors.
    std::string min_term;
    bool found = false;
    for (auto &c : cursors) {
      if (!c.iter->Valid()) {
        continue;
      }
      const std::string t = c.iter->key().ToString();
      if (!found || t < min_term) {
        min_term = t;
        found = true;
      }
    }
    if (!found) {
      break;  // All iterators exhausted.
    }

    // Collect entries for min_term from every cursor that has it.
    // Process cursors in segment order to maintain doc_id ascending order.
    term_entries.clear();
    for (auto &c : cursors) {
      if (!c.iter->Valid()) {
        continue;
      }
      if (c.iter->key().ToString() != min_term) {
        continue;
      }

      const std::string posting_data = c.iter->value().ToString();
      if (!BitPackedPostingList::is_bitpacked_format(posting_data.data(),
                                                     posting_data.size())) {
        LOG_ERROR(
            "FtsRocksdbReducer: source postings is not BitPacked. "
            "field[%s] segment[%u] term[%s]",
            field_name_.c_str(), c.segment_index, min_term.c_str());
        return tl::make_unexpected(Status::InternalError(
            "FtsRocksdbReducer: source postings is not BitPacked. field=",
            field_name_, " term=", min_term));
      }

      BitPackedPostingIterator bp_iter;
      if (bp_iter.open(posting_data.data(), posting_data.size()) != 0) {
        LOG_ERROR(
            "FtsRocksdbReducer: failed to open bitpacked postings. "
            "field[%s] segment[%u] term[%s]",
            field_name_.c_str(), c.segment_index, min_term.c_str());
        return tl::make_unexpected(Status::InternalError(
            "FtsRocksdbReducer: failed to open bitpacked postings. field=",
            field_name_, " term=", min_term));
      }

      term_entries.reserve(term_entries.size() + bp_iter.cost());
      uint32_t local_doc_id = bp_iter.next_doc();
      while (local_doc_id != BitPackedPostingIterator::NO_MORE_DOCS) {
        const uint64_t global_doc_id =
            c.stats->min_doc_id + static_cast<uint64_t>(local_doc_id);
        if (!filter.is_filtered(global_doc_id)) {
          const uint32_t new_doc_id =
              static_cast<uint32_t>(global_doc_id - min_doc_id_);
          term_entries.push_back(
              {new_doc_id, bp_iter.term_freq(), bp_iter.doc_len()});
        }
        local_doc_id = bp_iter.next_doc();
      }
      c.iter->Next();  // Advance past this term in this cursor.
    }

    if (term_entries.empty()) {
      continue;
    }

    // Encode and put immediately — peak memory is one term's entries.
    doc_ids_buf.clear();
    tfs_buf.clear();
    doc_lens_buf.clear();
    doc_ids_buf.reserve(term_entries.size());
    tfs_buf.reserve(term_entries.size());
    doc_lens_buf.reserve(term_entries.size());
    for (const auto &e : term_entries) {
      doc_ids_buf.push_back(e.doc_id);
      tfs_buf.push_back(e.tf);
      doc_lens_buf.push_back(e.doc_len);
    }

    std::string packed = BitPackedPostingList::encode(
        doc_ids_buf.data(), tfs_buf.data(), doc_lens_buf.data(),
        doc_ids_buf.size(), doc_ids_buf.size(), *scorer_);

    if (!ctx_->db_->Put(ctx_->write_opts_, dst_postings_cf_, min_term, packed)
             .ok()) {
      return tl::make_unexpected(Status::InternalError(
          "FtsRocksdbReducer: failed to put bitpacked postings. field=",
          field_name_));
    }
  }

  return {};
}

Result<void> FtsRocksdbReducer::reduce_positions(uint32_t segment_index,
                                                 const IndexFilter &filter) {
  const FtsSegmentStats &stats = segment_stats_[segment_index];
  auto *src_positions_cf = src_positions_cfs_[segment_index];

  auto iter = std::unique_ptr<rocksdb::Iterator>(
      src_ctxs_[segment_index]->db_->NewIterator(
          src_ctxs_[segment_index]->read_opts_, src_positions_cf));
  iter->SeekToFirst();

  for (; iter->Valid(); iter->Next()) {
    const std::string key = iter->key().ToString();

    std::string term;
    uint32_t local_doc_id = 0;
    if (!parse_doc_term_key(key, &term, &local_doc_id)) {
      LOG_WARN(
          "FtsRocksdbReducer::reduce_positions: malformed key, skip. "
          "field[%s] segment[%u] key_size[%zu]",
          field_name_.c_str(), segment_index, key.size());
      continue;
    }

    const uint64_t global_doc_id =
        stats.min_doc_id + static_cast<uint64_t>(local_doc_id);
    if (filter.is_filtered(global_doc_id)) {
      continue;
    }

    const uint32_t new_doc_id =
        static_cast<uint32_t>(global_doc_id - min_doc_id_);
    const std::string new_key = make_doc_term_key(term, new_doc_id);

    if (!ctx_->db_
             ->Put(ctx_->write_opts_, dst_positions_cf_, new_key,
                   iter->value().ToString())
             .ok()) {
      LOG_ERROR(
          "FtsRocksdbReducer: failed to write positions. field[%s] term[%s]",
          field_name_.c_str(), term.c_str());
      return tl::make_unexpected(Status::InternalError(
          "FtsRocksdbReducer: failed to write positions. field=", field_name_));
    }
  }

  return {};
}

Result<void> FtsRocksdbReducer::flush_stat(uint64_t total_docs,
                                           uint64_t total_tokens) {
  if (!ctx_->db_
           ->Put(ctx_->write_opts_, dst_stat_cf_,
                 make_total_docs_key(field_name_),
                 encode_uint64_value(total_docs))
           .ok()) {
    LOG_ERROR("FtsRocksdbReducer: failed to write total_docs. field[%s]",
              field_name_.c_str());
    return tl::make_unexpected(Status::InternalError(
        "FtsRocksdbReducer: failed to write total_docs. field=", field_name_));
  }

  if (!ctx_->db_
           ->Put(ctx_->write_opts_, dst_stat_cf_,
                 make_total_tokens_key(field_name_),
                 encode_uint64_value(total_tokens))
           .ok()) {
    LOG_ERROR("FtsRocksdbReducer: failed to write total_tokens. field[%s]",
              field_name_.c_str());
    return tl::make_unexpected(Status::InternalError(
        "FtsRocksdbReducer: failed to write total_tokens. field=",
        field_name_));
  }

  return {};
}

}  // namespace zvec::fts
