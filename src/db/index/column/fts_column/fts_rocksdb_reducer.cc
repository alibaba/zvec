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
#include "db/index/column/fts_column/fts_utils.h"
#include "db/index/column/fts_column/posting/bitpacked_posting_list.h"

namespace zvec::fts {

namespace {

// Dense survivor index in [0, effective_total_docs), or kFilteredRank if
// scan_pos is in the delete bitmap.  Roaring rank(x) counts elements ≤ x;
// for an alive scan_pos that's exactly the number of deletes strictly
// before it, so `scan_pos - rank(scan_pos)` is its survivor rank.
constexpr uint32_t kFilteredRank = std::numeric_limits<uint32_t>::max();

inline uint32_t dense_rank(uint64_t scan_pos, const roaring::Roaring &bitmap) {
  const uint32_t pos32 = static_cast<uint32_t>(scan_pos);
  if (bitmap.contains(pos32)) {
    return kFilteredRank;
  }
  return static_cast<uint32_t>(scan_pos - bitmap.rank(pos32));
}

}  // namespace

// ============================================================
// Design notes
// ============================================================
//
// Immutable FTS segment CFs:
//   - postings_cf   : term -> BitPacked posting (inline tf/doc_len/max_score)
//   - positions_cf  : term\0doc_id -> varint delta positions (phrase queries)
//   - stat_cf       : field_total_docs / field_total_tokens
//
// Multi-way merge N source segments into one destination, in two passes.
// All input postings must be BitPacked; output is BitPacked too — no
// Roaring intermediate, no side CFs ($TF/$MAX_TF/$DOC_LEN) read or written.
//
// Doc id spaces:
//   SRC LOCAL   ∈ [0, stats.doc_count): value stored in src postings.
//   SCAN POS    ∈ [0, Σ stats.doc_count): feed-order concatenated position;
//               same id space as SegmentHelper::delete_row_id_bitmap.
//               scan_pos = scan_offset_per_seg_[seg] + local
//   DST LOCAL   ∈ [0, effective_total_docs_): dense survivor rank.
//               Equals the row index ReduceScalar writes into the new
//               segment's densified forward storage, so post-merge fetch()
//               needs no translation.
//               dst_local = scan_pos - bitmap.rank(scan_pos)
//
// Pass 1 (collect_effective_stats): no per-doc materialization.
//   - effective_total_docs_   = Σ stats.doc_count - bitmap.cardinality()
//   - effective_total_tokens_ = sum of survivors' inline doc_len
//     (per-segment dedup uses vector<bool>, ~125 KB / 1M docs)
//
// Pass 2 (merge_and_flush_postings): N RocksDB iterators, term-by-term
// multi-way merge in lex order; per-term entries are encoded + put
// immediately so peak memory is one term's entries.  dst_local resolved
// on the fly via dense_rank(scan_pos), sharing the bitmap with the
// vector reducer.

// ============================================================
// Public interface
// ============================================================

Result<void> FtsRocksdbReducer::init(
    const std::string &field_name, RocksdbContext *ctx,
    rocksdb::ColumnFamilyHandle *dst_postings_cf,
    rocksdb::ColumnFamilyHandle *dst_positions_cf,
    rocksdb::ColumnFamilyHandle *dst_stat_cf) {
  if (!dst_postings_cf || !dst_positions_cf || !dst_stat_cf) {
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
  scan_offset_per_seg_.clear();
  num_segments_ = 0;
  state_ = STATE_UNINITED;
  return {};
}

Result<void> FtsRocksdbReducer::feed(
    FtsSegmentStats segment_stats, RocksdbContext *src_ctx,
    rocksdb::ColumnFamilyHandle *src_postings_cf,
    rocksdb::ColumnFamilyHandle *src_positions_cf) {
  if (state_ != STATE_INITED && state_ != STATE_FEED) {
    return tl::make_unexpected(Status::InternalError(
        "FtsRocksdbReducer: call init() before feed(). field=", field_name_));
  }

  if (!src_postings_cf || !src_positions_cf) {
    return tl::make_unexpected(Status::InvalidArgument(
        "FtsRocksdbReducer: null source CF. field=", field_name_));
  }

  // doc_count == 0 segments contribute nothing; mark state and skip so the
  // contiguity check and scan_offset cumsum only see non-empty inputs (the
  // matching FilterRecordBatch / RowIdFilter id space behaves the same way).
  if (segment_stats.doc_count == 0) {
    state_ = STATE_FEED;
    return {};
  }

  // Require consecutive global doc_id ranges between non-empty segments so
  // the shared delete_row_id_bitmap stays aligned with input scan order.
  if (!segment_stats_.empty() &&
      segment_stats.min_doc_id != segment_stats_.back().max_doc_id + 1) {
    return tl::make_unexpected(Status::InternalError(
        "FtsRocksdbReducer: segments not in consecutive doc_id order. field=",
        field_name_));
  }

  segment_stats_.emplace_back(std::move(segment_stats));
  src_ctxs_.emplace_back(src_ctx);
  src_postings_cfs_.emplace_back(src_postings_cf);
  src_positions_cfs_.emplace_back(src_positions_cf);
  ++num_segments_;

  state_ = STATE_FEED;
  return {};
}

Result<void> FtsRocksdbReducer::reduce(
    const roaring::Roaring &delete_row_id_bitmap) {
  if (state_ != STATE_FEED || num_segments_ == 0) {
    return tl::make_unexpected(Status::InternalError(
        "FtsRocksdbReducer: call feed() before reduce(). field=", field_name_));
  }

  effective_total_docs_ = 0;
  effective_total_tokens_ = 0;

  // Precompute scan_offset = cumulative doc_count.  Combined with the
  // bitmap this lets dense_rank() resolve any (seg, local) in
  // O(roaring::rank) without a per-doc table.
  scan_offset_per_seg_.assign(num_segments_, 0);
  uint64_t cumulative = 0;
  for (uint32_t seg = 0; seg < num_segments_; ++seg) {
    scan_offset_per_seg_[seg] = cumulative;
    cumulative += segment_stats_[seg].doc_count;
  }

  // Phase 1: streaming per-term BitPacked merge into dst_postings_cf;
  // accumulates effective_total_docs_ / effective_total_tokens_.
  auto ret = reduce_postings(delete_row_id_bitmap);
  if (!ret) {
    LOG_ERROR("FtsRocksdbReducer: reduce_postings failed. field[%s]",
              field_name_.c_str());
    return ret;
  }

  // Phase 2: per-segment positions CF remap (phrase queries).
  for (uint32_t segment_index = 0; segment_index < num_segments_;
       ++segment_index) {
    ret = reduce_positions(segment_index, delete_row_id_bitmap);
    if (!ret) {
      LOG_ERROR(
          "FtsRocksdbReducer: reduce_positions failed. segment[%u] field[%s]",
          segment_index, field_name_.c_str());
      return ret;
    }
  }

  // Phase 3: persist effective stats — same source of truth used by Phase 1
  // when encoding block_max_score, so search-time IDF/avgdl stays consistent.
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
// Private
// ============================================================

Result<void> FtsRocksdbReducer::reduce_postings(
    const roaring::Roaring &delete_row_id_bitmap) {
  auto ret = collect_effective_stats(delete_row_id_bitmap);
  if (!ret) {
    return ret;
  }
  // Scorer seeded with final effective stats; used by Pass 2 to compute
  // block_max_score consistent with the values flushed to stat_cf.
  scorer_ = std::make_shared<BM25Scorer>();
  scorer_->update_stats(effective_total_docs_, effective_total_tokens_);
  return merge_and_flush_postings(delete_row_id_bitmap);
}

Result<void> FtsRocksdbReducer::collect_effective_stats(
    const roaring::Roaring &delete_row_id_bitmap) {
  effective_total_docs_ = 0;
  effective_total_tokens_ = 0;

  // effective_total_docs = Σ doc_count - |deletes|.  Bitmap covers scan
  // positions [0, Σ doc_count), so cardinality() is the exact filtered
  // count.  Includes empty docs, matching mutable indexer semantics.
  uint64_t total_input_docs = 0;
  for (const auto &s : segment_stats_) {
    total_input_docs += s.doc_count;
  }
  const uint64_t total_deletes = delete_row_id_bitmap.cardinality();
  if (total_deletes > total_input_docs) {
    return tl::make_unexpected(
        Status::InternalError("FtsRocksdbReducer: delete bitmap cardinality[",
                              total_deletes, "] exceeds total input docs[",
                              total_input_docs, "]. field=", field_name_));
  }
  effective_total_docs_ = total_input_docs - total_deletes;

  // effective_total_tokens_: walk every posting, sum doc_len once per
  // surviving local_doc_id.  Per-segment vector<bool> dedup (~125 KB / 1M
  // docs) is required because immutable segments have no per-doc doc_len
  // column to read from directly.
  for (uint32_t seg = 0; seg < num_segments_; ++seg) {
    const uint64_t seg_doc_count = segment_stats_[seg].doc_count;
    const uint64_t scan_offset = scan_offset_per_seg_[seg];
    std::vector<bool> seen_docs(seg_doc_count, false);

    auto *src_cf = src_postings_cfs_[seg];
    auto iter = std::unique_ptr<rocksdb::Iterator>(
        src_ctxs_[seg]->db_->NewIterator(src_ctxs_[seg]->read_opts_, src_cf));
    iter->SeekToFirst();

    while (iter->Valid()) {
      const std::string posting_data = iter->value().ToString();

      if (!BitPackedPostingList::is_bitpacked_format(posting_data.data(),
                                                     posting_data.size())) {
        return tl::make_unexpected(Status::InternalError(
            "FtsRocksdbReducer: source postings is not BitPacked. field=",
            field_name_));
      }

      BitPackedPostingIterator bp_iter;
      if (bp_iter.open(posting_data.data(), posting_data.size()) != 0) {
        return tl::make_unexpected(Status::InternalError(
            "FtsRocksdbReducer: failed to open bitpacked postings. field=",
            field_name_));
      }

      uint32_t local_doc_id = bp_iter.next_doc();
      while (local_doc_id != BitPackedPostingIterator::NO_MORE_DOCS) {
        if (local_doc_id < seg_doc_count && !seen_docs[local_doc_id]) {
          const uint64_t scan_pos = scan_offset + local_doc_id;
          if (!delete_row_id_bitmap.contains(static_cast<uint32_t>(scan_pos))) {
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

Result<void> FtsRocksdbReducer::merge_and_flush_postings(
    const roaring::Roaring &delete_row_id_bitmap) {
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
    // Pick the lex-smallest current term across cursors.
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
      break;
    }

    // Cursors visited in segment order ⇒ dense ranks emerge ascending.
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
        return tl::make_unexpected(Status::InternalError(
            "FtsRocksdbReducer: source postings is not BitPacked. field=",
            field_name_, " term=", min_term));
      }

      BitPackedPostingIterator bp_iter;
      if (bp_iter.open(posting_data.data(), posting_data.size()) != 0) {
        return tl::make_unexpected(Status::InternalError(
            "FtsRocksdbReducer: failed to open bitpacked postings. field=",
            field_name_, " term=", min_term));
      }

      term_entries.reserve(term_entries.size() + bp_iter.cost());
      const uint64_t scan_offset = scan_offset_per_seg_[c.segment_index];
      const uint64_t seg_doc_count = c.stats->doc_count;
      uint32_t local_doc_id = bp_iter.next_doc();
      while (local_doc_id != BitPackedPostingIterator::NO_MORE_DOCS) {
        if (local_doc_id < seg_doc_count) {
          const uint32_t new_doc_id =
              dense_rank(scan_offset + local_doc_id, delete_row_id_bitmap);
          if (new_doc_id != kFilteredRank) {
            term_entries.push_back(
                {new_doc_id, bp_iter.term_freq(), bp_iter.doc_len()});
          }
        }
        local_doc_id = bp_iter.next_doc();
      }
      c.iter->Next();
    }

    if (term_entries.empty()) {
      continue;
    }

    // Encode + put per term ⇒ peak memory is one term's entries.
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

Result<void> FtsRocksdbReducer::reduce_positions(
    uint32_t segment_index, const roaring::Roaring &delete_row_id_bitmap) {
  auto *src_positions_cf = src_positions_cfs_[segment_index];
  const uint64_t scan_offset = scan_offset_per_seg_[segment_index];
  const uint64_t seg_doc_count = segment_stats_[segment_index].doc_count;

  auto iter = std::unique_ptr<rocksdb::Iterator>(
      src_ctxs_[segment_index]->db_->NewIterator(
          src_ctxs_[segment_index]->read_opts_, src_positions_cf));
  iter->SeekToFirst();

  for (; iter->Valid(); iter->Next()) {
    const std::string key = iter->key().ToString();

    std::string term;
    uint32_t local_doc_id = 0;
    if (!parse_doc_term_key(key, &term, &local_doc_id)) {
      return tl::make_unexpected(Status::InternalError(
          "FtsRocksdbReducer: malformed positions key. field=", field_name_));
    }

    if (local_doc_id >= seg_doc_count) {
      continue;
    }
    const uint32_t new_doc_id =
        dense_rank(scan_offset + local_doc_id, delete_row_id_bitmap);
    if (new_doc_id == kFilteredRank) {
      continue;
    }
    const std::string new_key = make_doc_term_key(term, new_doc_id);

    if (!ctx_->db_
             ->Put(ctx_->write_opts_, dst_positions_cf_, new_key,
                   iter->value().ToString())
             .ok()) {
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
    return tl::make_unexpected(Status::InternalError(
        "FtsRocksdbReducer: failed to write total_docs. field=", field_name_));
  }

  if (!ctx_->db_
           ->Put(ctx_->write_opts_, dst_stat_cf_,
                 make_total_tokens_key(field_name_),
                 encode_uint64_value(total_tokens))
           .ok()) {
    return tl::make_unexpected(Status::InternalError(
        "FtsRocksdbReducer: failed to write total_tokens. field=",
        field_name_));
  }

  return {};
}

}  // namespace zvec::fts
