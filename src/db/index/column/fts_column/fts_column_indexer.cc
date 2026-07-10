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

#include "fts_column_indexer.h"
#include <chrono>
#include <cstring>
#include <queue>
#include <thread>
#include <unordered_map>
#include <roaring/roaring.h>
#include <rocksdb/write_batch.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/db/status.h>
#include "db/common/typedef.h"
#include "iterator/fts_candidate_iterator.h"
#include "iterator/fts_conjunction_iterator.h"
#include "iterator/fts_disjunction_iterator.h"
#include "iterator/fts_phrase_iterator.h"
#include "iterator/fts_term_iterator.h"
#include "posting/bitpacked_posting_list.h"
#include "fts_pipeline.h"
#include "fts_utils.h"

namespace zvec::fts {

// ============================================================
// Lifecycle
// ============================================================

FtsColumnIndexer::~FtsColumnIndexer() {
  // Pipeline release is handled by FtsIndexParams destructor via fts_params_.
  if (opened_.load()) {
    (void)close();
  }
}

// ============================================================
// Initialization — shared reader core
// ============================================================

Result<void> FtsColumnIndexer::open_reader(
    const std::string &field_name, RocksdbContext *ctx,
    rocksdb::ColumnFamilyHandle *postings_cf,
    rocksdb::ColumnFamilyHandle *positions_cf,
    rocksdb::ColumnFamilyHandle *term_freq_cf,
    rocksdb::ColumnFamilyHandle *max_tf_cf,
    rocksdb::ColumnFamilyHandle *doc_len_cf,
    rocksdb::ColumnFamilyHandle *stat_cf, BM25Params bm25_params) {
  if (opened_.load()) {
    return tl::make_unexpected(Status::InternalError(
        "FtsColumnIndexer already opened. field=", field_name));
  }

  field_name_ = field_name;
  ctx_ = ctx;
  postings_cf_ = postings_cf;
  positions_cf_ = positions_cf;
  term_freq_cf_ = term_freq_cf;
  max_tf_cf_ = max_tf_cf;
  doc_len_cf_ = doc_len_cf;
  stat_cf_ = stat_cf;

  scorer_ = std::make_shared<BM25Scorer>(bm25_params);

  // doc_len_cf == nullptr → immutable path, load persisted stats.
  // doc_len_cf != nullptr → mutable path, stats maintained in-memory.
  // Mutable reopen also tries persisted stats first: an existing writing
  // segment may already have flushed stats, while a brand-new mutable segment
  // legitimately has no stats yet and falls back to zero below.
  auto load_stats_ret = scorer_->load_segment_stats(field_name, ctx, stat_cf);
  if (load_stats_ret != 0) {
    if (doc_len_cf == nullptr) {
      return tl::make_unexpected(Status::InternalError(
          "FtsColumnIndexer failed to load segment stats. field=", field_name));
    }
    scorer_->update_stats(0, 0);
  }

  const auto stats = scorer_->stats();
  total_docs_.store(stats.total_docs, std::memory_order_release);
  total_tokens_.store(stats.total_tokens, std::memory_order_release);

  opened_.store(true);
  return {};
}

// ============================================================
// Initialization — read+write (mutable)
// ============================================================

Result<void> FtsColumnIndexer::open(FieldSchema::Ptr field_meta,
                                    RocksdbContext *ctx,
                                    rocksdb::ColumnFamilyHandle *postings_cf,
                                    rocksdb::ColumnFamilyHandle *positions_cf,
                                    rocksdb::ColumnFamilyHandle *term_freq_cf,
                                    rocksdb::ColumnFamilyHandle *max_tf_cf,
                                    rocksdb::ColumnFamilyHandle *doc_len_cf,
                                    rocksdb::ColumnFamilyHandle *stat_cf) {
  if (!field_meta || !ctx) {
    return tl::make_unexpected(
        Status::InvalidArgument("FtsColumnIndexer: null field_meta or ctx"));
  }

  // Obtain FtsIndexParams from field_meta's index_params.
  auto index_params = field_meta->index_params();
  auto fts_param =
      std::dynamic_pointer_cast<zvec::FtsIndexParams>(index_params);
  if (!fts_param) {
    return tl::make_unexpected(Status::InvalidArgument(
        "FtsColumnIndexer: field has no FtsIndexParams. field=",
        field_meta->name()));
  }

  auto pipeline_result = zvec::detail::AcquireFtsPipeline(*fts_param);
  if (!pipeline_result.has_value()) {
    return tl::make_unexpected(Status::InternalError(
        "FtsColumnIndexer: failed to create tokenizer pipeline. field=",
        field_meta->name(), " err=", pipeline_result.error().message()));
  }

  field_meta_ = std::move(field_meta);
  tokenizer_pipeline_ = std::move(pipeline_result.value());
  fts_params_ = fts_param;

  return open_reader(field_meta_->name(), ctx, postings_cf, positions_cf,
                     term_freq_cf, max_tf_cf, doc_len_cf, stat_cf);
}

// ============================================================
// Initialization — read-only (immutable / standalone)
// ============================================================

// ============================================================
// Close
// ============================================================

Result<void> FtsColumnIndexer::close() {
  if (!opened_.load()) {
    return tl::make_unexpected(Status::InternalError(
        "FtsColumnIndexer::close: not opened. field=", field_name_));
  }

  ctx_ = nullptr;
  tokenizer_pipeline_.reset();
  postings_cf_ = nullptr;
  positions_cf_ = nullptr;
  term_freq_cf_.store(nullptr, std::memory_order_release);
  max_tf_cf_.store(nullptr, std::memory_order_release);
  doc_len_cf_.store(nullptr, std::memory_order_release);
  stat_cf_ = nullptr;
  scorer_.reset();

  opened_.store(false);
  return {};
}

// ============================================================
// Query entry point
// ============================================================

Result<std::vector<FtsResult>> FtsColumnIndexer::search(
    const FtsAstNode &ast, const FtsQueryParams &query_params) const {
  if (!scorer_) {
    LOG_ERROR("FtsColumnIndexer::search: not opened. field[%s]",
              field_name_.c_str());
    return tl::make_unexpected(Status::InternalError(
        "FtsColumnIndexer::search: not opened. field=", field_name_));
  }

  if (query_params.topk == 0) {
    return std::vector<FtsResult>{};
  }

  if (ast.must_not) {
    LOG_WARN(
        "FtsColumnIndexer::search: must_not on root is not allowed. field[%s]",
        field_name_.c_str());
    return tl::make_unexpected(Status::InvalidArgument(
        "FtsColumnIndexer::search: must_not on root is not allowed. field=",
        field_name_));
  }

  auto iter_result = build_iterator(ast);
  if (!iter_result.has_value()) {
    LOG_ERROR("FtsColumnIndexer::search: build_iterator failed. field[%s] %s",
              field_name_.c_str(), iter_result.error().message().c_str());
    return tl::make_unexpected(iter_result.error());
  }
  DocIteratorPtr root_iter = std::move(iter_result.value());
  if (!root_iter) {
    // No matching terms found — valid empty result, not an error.
    return std::vector<FtsResult>{};
  }

  // Candidate-driven mode: AND a CandidateDocIterator into the root so the
  // small candidate set leads (Conjunction sorts by cost asc), turning the
  // posting walk into per-candidate advance()+matches()+score().
  if (query_params.candidate_ids) {
    if (query_params.candidate_ids->empty()) {
      return std::vector<FtsResult>{};
    }
    std::vector<DocIteratorPtr> musts;
    musts.reserve(2);
    musts.push_back(
        std::make_unique<CandidateDocIterator>(*query_params.candidate_ids));
    musts.push_back(std::move(root_iter));
    root_iter = std::make_unique<ConjunctionIterator>(
        std::move(musts), std::vector<DocIteratorPtr>{});
  }

  const uint32_t topk = query_params.topk;
  const zvec::IndexFilter *filter_ptr = query_params.filter.get();

  using MinHeap = std::priority_queue<FtsResult, std::vector<FtsResult>,
                                      std::greater<FtsResult>>;
  MinHeap min_heap;

  // Filter pushdown: when a filter is present, use the filter-aware next_doc
  // overload so composite iterators skip filtered docs before paying for
  // block-max binary search, do_next alignment, or phase-2 position checks.
  uint32_t doc_id =
      filter_ptr ? root_iter->next_doc(filter_ptr) : root_iter->next_doc();
  while (doc_id != DocIterator::NO_MORE_DOCS) {
    const uint64_t global_doc_id = static_cast<uint64_t>(doc_id);

    if (root_iter->matches()) {
      float s = root_iter->score();
      if (s > 0.0f) {
        if (min_heap.size() < topk) {
          min_heap.push({global_doc_id, s});
          if (min_heap.size() == topk) {
            root_iter->set_min_competitive_score(min_heap.top().score);
          }
        } else if (s > min_heap.top().score) {
          min_heap.pop();
          min_heap.push({global_doc_id, s});
          root_iter->set_min_competitive_score(min_heap.top().score);
        }
      }
    }
    doc_id =
        filter_ptr ? root_iter->next_doc(filter_ptr) : root_iter->next_doc();
  }

  std::vector<FtsResult> results(min_heap.size());
  for (auto it = results.rbegin(); it != results.rend(); ++it) {
    *it = min_heap.top();
    min_heap.pop();
  }

  return results;
}

// ============================================================
// Side CF reset (dump path)
// ============================================================

void FtsColumnIndexer::reset_side_cfs() {
  cf_dropped_.store(true);
  while (cf_counter_.load() > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  term_freq_cf_.store(nullptr, std::memory_order_release);
  max_tf_cf_.store(nullptr, std::memory_order_release);
  doc_len_cf_.store(nullptr, std::memory_order_release);
}

// ============================================================
// Iterator tree construction
// ============================================================

Result<DocIteratorPtr> FtsColumnIndexer::build_iterator(
    const FtsAstNode &node) const {
  switch (node.type()) {
    case FtsNodeType::TERM:
      return build_term_iterator(static_cast<const TermNode &>(node));
    case FtsNodeType::PHRASE:
      return build_phrase_iterator(static_cast<const PhraseNode &>(node));
    case FtsNodeType::AND:
      return build_and_iterator(static_cast<const AndNode &>(node));
    case FtsNodeType::OR:
      return build_or_iterator(static_cast<const OrNode &>(node));
    case FtsNodeType::EMPTY:
      // Null iterator reuses the existing AND/OR/search() null-handling path.
      return DocIteratorPtr{nullptr};
    default:
      return tl::make_unexpected(Status::InternalError(
          "FtsColumnIndexer::build_iterator: unknown node type. field=",
          field_name_));
  }
}

Result<DocIteratorPtr> FtsColumnIndexer::create_term_iterator_from_raw(
    const std::string &term, rocksdb::PinnableSlice raw_data,
    float boost) const {
  if (BitPackedPostingList::is_bitpacked_format(raw_data.data(),
                                                raw_data.size())) {
    auto iter = std::make_unique<TermDocIterator>(term, std::move(raw_data),
                                                  scorer_, boost);
    if (iter->cost() == 0) {
      return DocIteratorPtr{nullptr};
    }
    return iter;
  }

  roaring_bitmap_t *bitmap = roaring_bitmap_portable_deserialize_safe(
      raw_data.data(), raw_data.size());
  if (!bitmap) {
    return tl::make_unexpected(Status::InternalError(
        "FtsColumnIndexer: failed to deserialize roaring bitmap. field=",
        field_name_, " term=", term));
  }

  const uint64_t df = roaring_bitmap_get_cardinality(bitmap);
  if (df == 0) {
    roaring_bitmap_free(bitmap);
    return nullptr;
  }

  ++cf_counter_;
  auto *term_freq_cf = term_freq_cf_.load(std::memory_order_acquire);
  auto *doc_len_cf = doc_len_cf_.load(std::memory_order_acquire);
  auto *max_tf_cf = max_tf_cf_.load(std::memory_order_acquire);
  auto *cf_counter = &cf_counter_;
  if (cf_dropped_) {
    term_freq_cf = nullptr;
    doc_len_cf = nullptr;
    cf_counter = nullptr;
    max_tf_cf = nullptr;
    --cf_counter_;
  }

  // WAND upper bound.  When max_tf_cf is available we compute the tight
  // score(df, max_tf, min_dl).  Otherwise fall back to the formula-derived
  // bound idf*(k1+1), which is still a valid upper bound yet much tighter
  // than +inf, so WAND pruning remains effective.
  float max_score_val = scorer_->max_score_bound(df);
  if (max_tf_cf) {
    WandOptimizer wand;
    if (wand.open(scorer_, ctx_, max_tf_cf, 0) == 0) {
      uint32_t max_tf = wand.read_max_tf(term);
      uint32_t min_dl = min_doc_len_.load(std::memory_order_relaxed);
      if (min_dl == std::numeric_limits<uint32_t>::max()) {
        min_dl = 1;
      }
      max_score_val = scorer_->score(df, max_tf, min_dl);
    }
  }

  return std::make_unique<TermDocIterator>(term, bitmap, df, scorer_,
                                           max_score_val, ctx_, term_freq_cf,
                                           doc_len_cf, cf_counter, boost);
}

Result<DocIteratorPtr> FtsColumnIndexer::build_term_iterator(
    const TermNode &term_node) const {
  const std::string &term = term_node.term;

  rocksdb::PinnableSlice raw_data;
  auto s = ctx_->db_->Get(ctx_->read_opts_, postings_cf_, term, &raw_data);
  if (!s.ok() || raw_data.empty()) {
    return DocIteratorPtr{nullptr};
  }

  return create_term_iterator_from_raw(term, std::move(raw_data),
                                       term_node.boost);
}

std::vector<rocksdb::PinnableSlice> FtsColumnIndexer::batch_get_postings(
    const std::vector<rocksdb::Slice> &terms) const {
  std::vector<rocksdb::PinnableSlice> raw_postings(terms.size());
  if (terms.empty()) {
    return raw_postings;
  }

  std::vector<rocksdb::ColumnFamilyHandle *> cfs(terms.size(), postings_cf_);
  std::vector<rocksdb::Status> statuses(terms.size());
  ctx_->db_->MultiGet(ctx_->read_opts_, terms.size(), cfs.data(), terms.data(),
                      raw_postings.data(), statuses.data());
  // Ignore failed lookups as callers can check via empty()
  return raw_postings;
}

Result<DocIteratorPtr> FtsColumnIndexer::build_phrase_iterator(
    const PhraseNode &phrase_node) const {
  if (phrase_node.terms.empty()) {
    return DocIteratorPtr{nullptr};
  }

  const std::vector<std::string> &terms = phrase_node.terms;
  std::vector<rocksdb::Slice> term_slices;
  term_slices.reserve(terms.size());
  for (const auto &t : terms) {
    term_slices.emplace_back(t);
  }
  auto raw_postings = batch_get_postings(term_slices);

  std::vector<DocIteratorPtr> term_iterators;
  term_iterators.reserve(terms.size());

  // Phrase-level boost is distributed across the internal term iterators.
  // PhraseDocIterator.score() delegates to conjunction.score() which sums the
  // internal contributions, so multiplying each contribution by boost yields
  // boost * (sum) = boost-applied-once at the phrase level.
  for (size_t i = 0; i < terms.size(); ++i) {
    if (raw_postings[i].empty()) {
      return DocIteratorPtr{nullptr};
    }
    auto iter_result = create_term_iterator_from_raw(
        terms[i], std::move(raw_postings[i]), phrase_node.boost);
    if (!iter_result.has_value()) {
      return iter_result;
    }
    if (!iter_result.value()) {
      return DocIteratorPtr{nullptr};
    }
    term_iterators.push_back(std::move(iter_result.value()));
  }

  if (term_iterators.empty()) {
    return DocIteratorPtr{nullptr};
  }

  auto conjunction = std::make_unique<ConjunctionIterator>(
      std::move(term_iterators), std::vector<DocIteratorPtr>{});

  return std::make_unique<PhraseDocIterator>(std::move(conjunction), terms,
                                             ctx_, positions_cf_);
}

Result<DocIteratorPtr> FtsColumnIndexer::build_and_iterator(
    const AndNode &and_node) const {
  if (and_node.children.empty()) {
    return DocIteratorPtr{nullptr};
  }

  std::vector<rocksdb::Slice> term_key_slices;
  std::vector<size_t> term_child_indices;
  term_key_slices.reserve(and_node.children.size());
  term_child_indices.reserve(and_node.children.size());

  for (size_t i = 0; i < and_node.children.size(); ++i) {
    const auto &child = and_node.children[i];
    if (child && child->type() == FtsNodeType::TERM) {
      term_key_slices.emplace_back(static_cast<const TermNode &>(*child).term);
      term_child_indices.push_back(i);
    }
  }

  auto term_raw_postings = batch_get_postings(term_key_slices);

  std::vector<DocIteratorPtr> must_iterators;
  std::vector<DocIteratorPtr> must_not_iterators;
  std::vector<DocIteratorPtr> should_iterators;
  size_t batched_cursor = 0;

  for (size_t i = 0; i < and_node.children.size(); ++i) {
    const auto &child = and_node.children[i];
    const bool is_must_not = child->must_not;
    const bool is_should = child->should;

    DocIteratorPtr iter;
    if (batched_cursor < term_child_indices.size() &&
        term_child_indices[batched_cursor] == i) {
      rocksdb::PinnableSlice &raw = term_raw_postings[batched_cursor];
      const auto &term_node = static_cast<const TermNode &>(*child);
      if (!raw.empty()) {
        auto iter_result = create_term_iterator_from_raw(
            term_node.term, std::move(raw), term_node.boost);
        if (!iter_result.has_value()) {
          return iter_result;
        }
        iter = std::move(iter_result.value());
      }
      ++batched_cursor;
    } else {
      auto iter_result = build_iterator(*child);
      if (!iter_result.has_value()) {
        return iter_result;
      }
      iter = std::move(iter_result.value());
    }

    if (!iter) {
      if (!is_must_not && !is_should) {
        return DocIteratorPtr{nullptr};
      }
      continue;
    }

    if (is_must_not) {
      must_not_iterators.push_back(std::move(iter));
    } else if (is_should) {
      should_iterators.push_back(std::move(iter));
    } else {
      must_iterators.push_back(std::move(iter));
    }
  }

  if (must_iterators.empty()) {
    return DocIteratorPtr{nullptr};
  }

  if (must_iterators.size() == 1 && must_not_iterators.empty() &&
      should_iterators.empty()) {
    return std::move(must_iterators[0]);
  }

  return std::make_unique<ConjunctionIterator>(std::move(must_iterators),
                                               std::move(must_not_iterators),
                                               std::move(should_iterators));
}

Result<DocIteratorPtr> FtsColumnIndexer::build_or_iterator(
    const OrNode &or_node) const {
  if (or_node.children.empty()) {
    return DocIteratorPtr{nullptr};
  }

  std::vector<rocksdb::Slice> term_key_slices;
  std::vector<size_t> term_child_indices;
  term_key_slices.reserve(or_node.children.size());
  term_child_indices.reserve(or_node.children.size());

  for (size_t i = 0; i < or_node.children.size(); ++i) {
    const auto &child = or_node.children[i];
    if (child && child->type() == FtsNodeType::TERM) {
      term_key_slices.emplace_back(static_cast<const TermNode &>(*child).term);
      term_child_indices.push_back(i);
    }
  }

  auto term_raw_postings = batch_get_postings(term_key_slices);

  // Invariant: the AST rewriter (fts::simplify) lifts both must_not and must
  // children out of OrNode into a wrapping AndNode before we get here, so the
  // loop below only ever sees plain positives. A must_not or must child
  // reaching this point indicates a caller that bypassed simplify — bail out
  // loudly rather than silently produce wrong results.
  std::vector<DocIteratorPtr> positive_iterators;
  size_t batched_cursor = 0;

  for (size_t i = 0; i < or_node.children.size(); ++i) {
    const auto &child = or_node.children[i];
    if (child->must_not || child->must) {
      LOG_ERROR(
          "build_or_iterator: must/must_not child reached OR "
          "(rewriter bypassed)");
      return tl::make_unexpected(Status::InternalError(
          "FtsColumnIndexer::build_or_iterator: OR contains must/must_not "
          "child"));
    }

    DocIteratorPtr iter;
    if (batched_cursor < term_child_indices.size() &&
        term_child_indices[batched_cursor] == i) {
      rocksdb::PinnableSlice &raw = term_raw_postings[batched_cursor];
      const auto &term_node = static_cast<const TermNode &>(*child);
      if (!raw.empty()) {
        auto iter_result = create_term_iterator_from_raw(
            term_node.term, std::move(raw), term_node.boost);
        if (!iter_result.has_value()) {
          return iter_result;
        }
        iter = std::move(iter_result.value());
      }
      ++batched_cursor;
    } else {
      auto iter_result = build_iterator(*child);
      if (!iter_result.has_value()) {
        return iter_result;
      }
      iter = std::move(iter_result.value());
    }

    if (iter) {
      positive_iterators.push_back(std::move(iter));
    }
  }

  if (positive_iterators.empty()) {
    return DocIteratorPtr{nullptr};
  }
  if (positive_iterators.size() == 1) {
    return std::move(positive_iterators[0]);
  }
  return std::make_unique<DisjunctionIterator>(std::move(positive_iterators));
}

// ============================================================
// Write operations
// ============================================================

Result<void> FtsColumnIndexer::insert(uint64_t seg_doc_id,
                                      const std::string &text) {
  // safe access check

  if (!tokenizer_pipeline_ || !ctx_) {
    return tl::make_unexpected(Status::InternalError(
        "FtsColumnIndexer::insert: not opened. field=", field_name_));
  }

  // Tokenize
  std::vector<Token> tokens = tokenizer_pipeline_->process(text);
  const uint32_t doc_len = static_cast<uint32_t>(tokens.size());

  // Aggregate position lists by term
  std::unordered_map<std::string, std::vector<uint32_t>> term_positions;
  for (const auto &token : tokens) {
    term_positions[token.text].push_back(token.position);
  }

  // Store seg_doc_id in RocksDB directly, similar to invert indexer
  const uint32_t doc_id_32 = static_cast<uint32_t>(seg_doc_id);

  // Pre-serialize a single-element Roaring Bitmap for this doc_id once,
  // reused across all terms to avoid repeated create/serialize/free overhead.
  roaring_bitmap_t *single_bitmap = roaring_bitmap_create_with_capacity(1);
  roaring_bitmap_add(single_bitmap, doc_id_32);
  size_t bitmap_size = roaring_bitmap_portable_size_in_bytes(single_bitmap);
  std::string bitmap_data(bitmap_size, '\0');
  roaring_bitmap_portable_serialize(single_bitmap, bitmap_data.data());
  roaring_bitmap_free(single_bitmap);

  // Batch all writes for this document into a single cross-CF WriteBatch,
  // reducing 4N+1 individual RocksDB Write() calls to one atomic write.
  rocksdb::WriteBatch batch;

  for (const auto &[term, positions] : term_positions) {
    const uint32_t tf = static_cast<uint32_t>(positions.size());

    // 1. Postings CF: merge doc_id bitmap
    batch.Merge(postings_cf_, term, bitmap_data);

    // 2. Positions CF: term\0doc_id -> delta-varint positions
    const std::string doc_term_key = make_doc_term_key(term, doc_id_32);
    batch.Put(positions_cf_, doc_term_key, encode_positions(positions));

    // 3. Term-freq CF: term\0doc_id -> uint32_t tf
    std::string tf_value(sizeof(uint32_t), '\0');
    std::memcpy(tf_value.data(), &tf, sizeof(uint32_t));
    batch.Put(term_freq_cf_.load(), doc_term_key, tf_value);

    // 4. Max-TF CF: term -> max(tf) via merge
    batch.Merge(max_tf_cf_.load(), term, tf_value);
  }

  // 5. Doc-len CF: doc_id -> uint32_t doc_len
  std::string doc_id_key(sizeof(uint32_t), '\0');
  std::memcpy(doc_id_key.data(), &doc_id_32, sizeof(uint32_t));
  std::string doc_len_value(sizeof(uint32_t), '\0');
  std::memcpy(doc_len_value.data(), &doc_len, sizeof(uint32_t));
  batch.Put(doc_len_cf_.load(), doc_id_key, doc_len_value);

  if (auto s = ctx_->db_->Write(ctx_->write_opts_, &batch); !s.ok()) {
    return tl::make_unexpected(Status::InternalError(
        "FtsColumnIndexer::insert: write batch failed. field=", field_name_,
        " status=", s.ToString()));
  }

  // 6. Update in-memory statistics atomically so concurrent search() calls
  //    see up-to-date values for BM25 scoring.
  const uint64_t new_total_docs =
      total_docs_.fetch_add(1, std::memory_order_relaxed) + 1;
  const uint64_t new_total_tokens =
      total_tokens_.fetch_add(doc_len, std::memory_order_relaxed) + doc_len;

  // Propagate updated stats to the scorer so that search() uses current avgdl.
  if (scorer_) {
    scorer_->update_stats(new_total_docs, new_total_tokens);
  }

  // CAS-update min_doc_len_ only when this document has tokens (doc_len > 0).
  if (doc_len > 0) {
    uint32_t cur = min_doc_len_.load(std::memory_order_relaxed);
    while (doc_len < cur && !min_doc_len_.compare_exchange_weak(
                                cur, doc_len, std::memory_order_relaxed)) {
    }
  }

  return {};
}

Result<void> FtsColumnIndexer::flush() {
  // safe access check

  if (!stat_cf_) {
    return {};
  }

  // Write total_docs and total_tokens to $SEGMENT_STAT CF.
  // Use acquire ordering so we see all inserts that happened before flush().
  const uint64_t snapshot_total_docs =
      total_docs_.load(std::memory_order_acquire);
  const uint64_t snapshot_total_tokens =
      total_tokens_.load(std::memory_order_acquire);

  auto s = ctx_->db_->Put(ctx_->write_opts_, stat_cf_,
                          make_total_docs_key(field_name_),
                          encode_uint64_value(snapshot_total_docs));
  if (!s.ok()) {
    return tl::make_unexpected(Status::InternalError(
        "FtsColumnIndexer::flush: failed to write total_docs. field=",
        field_name_, " status=", s.ToString()));
  }
  s = ctx_->db_->Put(ctx_->write_opts_, stat_cf_,
                     make_total_tokens_key(field_name_),
                     encode_uint64_value(snapshot_total_tokens));
  if (!s.ok()) {
    return tl::make_unexpected(Status::InternalError(
        "FtsColumnIndexer::flush: failed to write total_tokens. field=",
        field_name_, " status=", s.ToString()));
  }

  return {};
}

// ============================================================
// BitPacked conversion (called by MutableSegment::dump_fts_column_indexers)
// ============================================================

Result<void> FtsColumnIndexer::convert_postings_to_bitpacked() {
  // safe access check

  if (!postings_cf_ || !term_freq_cf_ || !doc_len_cf_ || !scorer_) {
    return tl::make_unexpected(Status::InternalError(
        "FtsColumnIndexer::convert_postings_to_bitpacked: not opened. field=",
        field_name_));
  }

  // ---------------------------------------------------------------
  // 1) Load doc_len_cf into an in-memory vector indexed by local doc_id.
  //    Single segment is at most a few MB even for 1M docs (4B per doc),
  //    so a flat vector is by far the cheapest lookup structure.
  // ---------------------------------------------------------------
  std::vector<uint32_t> doc_lens;
  {
    std::unique_ptr<rocksdb::Iterator> iter(
        ctx_->db_->NewIterator(ctx_->read_opts_, doc_len_cf_.load()));
    iter->SeekToFirst();
    while (iter->Valid()) {
      const std::string key = iter->key().ToString();
      const std::string value = iter->value().ToString();
      if (key.size() != sizeof(uint32_t) || value.size() != sizeof(uint32_t)) {
        LOG_WARN(
            "FtsColumnIndexer::convert_postings_to_bitpacked: malformed "
            "doc_len entry. field[%s] key_size[%zu] value_size[%zu]",
            field_name_.c_str(), key.size(), value.size());
        iter->Next();
        continue;
      }
      uint32_t local_doc_id = 0;
      uint32_t doc_len = 0;
      std::memcpy(&local_doc_id, key.data(), sizeof(uint32_t));
      std::memcpy(&doc_len, value.data(), sizeof(uint32_t));
      if (local_doc_id >= doc_lens.size()) {
        // Resize with default 1 to avoid divide-by-zero / log(0) downstream
        // if a stray doc_id ever shows up without a doc_len entry.
        doc_lens.resize(local_doc_id + 1, 1);
      }
      doc_lens[local_doc_id] = doc_len;
      iter->Next();
    }
  }

  // ---------------------------------------------------------------
  // 2) Streaming scan of term_freq_cf, grouped by term.
  //    RocksDB BytewiseComparator + big-endian doc_id encoding guarantees
  //    that within a term, doc_ids appear in ascending order — exactly what
  //    BitPackedPostingList::encode() requires.
  // ---------------------------------------------------------------
  std::string current_term;
  std::vector<uint32_t> doc_ids;
  std::vector<uint32_t> tfs;
  std::vector<uint32_t> term_doc_lens;  // reused buffer

  auto flush_current_term = [&]() -> Result<void> {
    if (current_term.empty() || doc_ids.empty()) {
      return {};
    }
    // Idempotency: skip if this term's postings are already BitPacked.
    // Important for crash-recovery — a re-run of dump after a partial
    // conversion must not double-encode.
    std::string existing;
    auto get_ret =
        ctx_->db_->Get(ctx_->read_opts_, postings_cf_, current_term, &existing);
    if (get_ret.ok() && !existing.empty() &&
        BitPackedPostingList::is_bitpacked_format(existing.data(),
                                                  existing.size())) {
      return {};
    }

    term_doc_lens.assign(doc_ids.size(), 1);
    for (size_t i = 0; i < doc_ids.size(); ++i) {
      const uint32_t did = doc_ids[i];
      if (did < doc_lens.size() && doc_lens[did] > 0) {
        term_doc_lens[i] = doc_lens[did];
      }
    }
    std::string packed = BitPackedPostingList::encode(
        doc_ids.data(), tfs.data(), term_doc_lens.data(), doc_ids.size(),
        /*df=*/doc_ids.size(), *scorer_);
    if (!ctx_->db_->Put(ctx_->write_opts_, postings_cf_, current_term, packed)
             .ok()) {
      return tl::make_unexpected(Status::InternalError(
          "FtsColumnIndexer::convert_postings_to_bitpacked: put failed. field=",
          field_name_, " term=", current_term));
    }
    return {};
  };

  {
    std::unique_ptr<rocksdb::Iterator> iter(
        ctx_->db_->NewIterator(ctx_->read_opts_, term_freq_cf_.load()));
    iter->SeekToFirst();
    while (iter->Valid()) {
      const std::string key = iter->key().ToString();
      const std::string value = iter->value().ToString();
      std::string term;
      uint32_t local_doc_id = 0;
      if (!parse_doc_term_key(key, &term, &local_doc_id) ||
          value.size() != sizeof(uint32_t)) {
        LOG_WARN(
            "FtsColumnIndexer::convert_postings_to_bitpacked: malformed "
            "term_freq entry. field[%s] key_size[%zu] value_size[%zu]",
            field_name_.c_str(), key.size(), value.size());
        iter->Next();
        continue;
      }
      uint32_t tf = 0;
      std::memcpy(&tf, value.data(), sizeof(uint32_t));

      if (term != current_term) {
        auto ret = flush_current_term();
        if (!ret) {
          return ret;
        }
        current_term = std::move(term);
        doc_ids.clear();
        tfs.clear();
      }
      doc_ids.push_back(local_doc_id);
      tfs.push_back(tf);
      iter->Next();
    }
  }
  // Flush the last term.
  auto ret = flush_current_term();
  if (!ret) {
    return ret;
  }

  // ---------------------------------------------------------------
  // 3) Clear $TF / $DOC_LEN / $MAX_TF CFs via DeleteRange.
  //
  // All payloads (tf, doc_len, max_score) have been inlined into the
  // BitPacked postings in step 2.  Wiping them here ensures the SST files
  // are cleaned up during the dump-side compaction, so the dumped immutable
  // segment is significantly smaller.  MutableSegment then drops the CFs
  // entirely after all indexers finish conversion.
  //
  // DeleteRange uses [begin, end) semantics; an empty begin and a 256-byte
  // 0xFF end together cover every possible key in these CFs.
  // ---------------------------------------------------------------
  static const std::string kClearBegin{};
  static const std::string kClearEnd(256, '\xFF');

  const std::pair<const char *, rocksdb::ColumnFamilyHandle *> cfs_to_clear[] =
      {
          {"$TF", term_freq_cf_.load()},
          {"$DOC_LEN", doc_len_cf_.load()},
          {"$MAX_TF", max_tf_cf_.load()},
      };
  for (const auto &[cf_name, cf] : cfs_to_clear) {
    if (cf == nullptr) {
      continue;
    }
    if (!ctx_->db_->DeleteRange(ctx_->write_opts_, cf, kClearBegin, kClearEnd)
             .ok()) {
      return tl::make_unexpected(Status::InternalError(
          "FtsColumnIndexer::convert_postings_to_bitpacked: failed to clear ",
          cf_name, " CF. field=", field_name_));
    }
  }

  return {};
}

// ============================================================
// Private helper methods
// ============================================================

void FtsColumnIndexer::encode_varint(uint32_t value, std::string *output) {
  while (value >= 0x80) {
    output->push_back(static_cast<char>((value & 0x7F) | 0x80));
    value >>= 7;
  }
  output->push_back(static_cast<char>(value));
}

std::string FtsColumnIndexer::encode_positions(
    const std::vector<uint32_t> &positions) {
  std::string result;
  uint32_t prev_position = 0;
  for (uint32_t position : positions) {
    // Delta encoding: store the difference between adjacent positions
    encode_varint(position - prev_position, &result);
    prev_position = position;
  }
  return result;
}

}  // namespace zvec::fts
