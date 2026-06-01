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
#include <memory>
#include <string>
#include "db/common/rocksdb_context.h"

namespace zvec::fts {

/*! BM25 scoring parameters
 */
struct BM25Params {
  // Term frequency saturation parameter, typical value 1.2
  float k1{1.2f};
  // Document length normalization parameter, typical value 0.75
  float b{0.75f};
};

/*! Plain snapshot of per-segment BM25 statistics (non-atomic, for callers)
 */
struct SegmentStatsSnapshot {
  uint64_t total_docs{0};
  uint64_t total_tokens{0};

  float avg_doc_len() const {
    if (total_docs == 0) {
      return 1.0f;
    }
    return static_cast<float>(total_tokens) / static_cast<float>(total_docs);
  }
};

/*! Per-segment BM25 statistics (thread-safe)
 *  Fields are std::atomic so that concurrent insert (writer) and search
 *  (reader) threads do not race on the raw values.
 */
struct SegmentStats {
  // Total number of documents in segment
  std::atomic<uint64_t> total_docs{0};
  // Total number of tokens in all documents in segment (used to calculate
  // average document length)
  std::atomic<uint64_t> total_tokens{0};

  SegmentStats() = default;

  // std::atomic is neither copyable nor movable; provide manual move
  // semantics so that BM25Scorer (which embeds SegmentStats) stays movable.
  // These are only used during single-threaded construction / NRVO and are
  // therefore safe with relaxed ordering.
  SegmentStats(SegmentStats &&other) noexcept
      : total_docs(other.total_docs.load(std::memory_order_relaxed)),
        total_tokens(other.total_tokens.load(std::memory_order_relaxed)) {}

  SegmentStats &operator=(SegmentStats &&other) noexcept {
    total_docs.store(other.total_docs.load(std::memory_order_relaxed),
                     std::memory_order_relaxed);
    total_tokens.store(other.total_tokens.load(std::memory_order_relaxed),
                       std::memory_order_relaxed);
    return *this;
  }

  SegmentStats(const SegmentStats &) = delete;
  SegmentStats &operator=(const SegmentStats &) = delete;

  // Take a consistent snapshot: load total_tokens first (the value that
  // grows together with total_docs) so the pair is *at least* as fresh as
  // the docs count, avoiding avg_doc_len() returning an inflated value.
  SegmentStatsSnapshot snapshot() const {
    const uint64_t tokens = total_tokens.load(std::memory_order_acquire);
    const uint64_t docs = total_docs.load(std::memory_order_acquire);
    return {docs, tokens};
  }

  // Average document length (total_tokens / total_docs)
  float avg_doc_len() const {
    return snapshot().avg_doc_len();
  }
};

/*! BM25 scorer
 *  Encapsulates standard BM25 formula, supports per-segment statistics loading
 * and WAND optimization
 *
 *  BM25 formula:
 *    score(q, d) = Σ IDF(t) * (tf(t,d) * (k1+1)) / (tf(t,d) +
 * k1*(1-b+b*|d|/avgdl)) IDF(t) = ln((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
 */
class BM25Scorer {
 public:
  explicit BM25Scorer(BM25Params params = BM25Params{}) : params_(params) {}

  /*! Load per-segment statistics from $SEGMENT_STAT CF
   *  \param field_name  Field name
   *  \param stat_cf     $SEGMENT_STAT CF
   *  \return 0 for success, non-0 for failure
   */
  int load_segment_stats(const std::string &field_name, RocksdbContext *ctx,
                         rocksdb::ColumnFamilyHandle *stat_cf);

  /*! Calculate BM25 contribution score of a single term for a single document
   *  \param term_doc_freq   Document frequency of this term in segment (df)
   *  \param term_freq       Term frequency of this term in current document
   * (tf) \param doc_len         Length of current document (number of tokens)
   *  \return BM25 score contribution
   */
  float score(uint64_t term_doc_freq, uint32_t term_freq,
              uint32_t doc_len) const;

  /*! Calculate IDF value of a term
   *  \param term_doc_freq  Document frequency of this term in segment (df)
   *  \return IDF value
   */
  float idf(uint64_t term_doc_freq) const;

  /*! Compute a tight WAND upper-bound score for a term without knowing
   *  per-document tf / doc_len.  Uses the identity  lim_{tf→∞} tf_norm = k1+1
   *  so the bound is  idf(df) * (k1 + 1).
   *  \param term_doc_freq  Document frequency of this term in segment (df)
   *  \return upper-bound score (0 when IDF ≤ 0)
   */
  float max_score_bound(uint64_t term_doc_freq) const;

  /*! Calculate BM25 score using a pre-computed IDF value.
   *  Avoids recomputing log() on every call — IDF is constant per term.
   *  \param idf_value      Pre-computed IDF value (from idf())
   *  \param term_freq      Term frequency in current document
   *  \param doc_len        Document length (number of tokens)
   *  \return BM25 score contribution
   */
  float score_with_idf(float idf_value, uint32_t term_freq,
                       uint32_t doc_len) const;

  /*! Calculate BM25 score with a per-term boost multiplier.
   *  Boost > 1 represents a term that appears multiple times in the original
   *  query (collapsed by the AST rewriter) or carries an explicit user weight.
   *  The multiplier is linear so that the post-rewrite score exactly matches
   *  the pre-rewrite "sum of N independent scorers" value.
   *  \param idf_value      Pre-computed IDF value (from idf())
   *  \param term_freq      Term frequency in current document
   *  \param doc_len        Document length (number of tokens)
   *  \param boost          Per-term boost (1.0 = no boost)
   *  \return BM25 score contribution scaled by boost
   */
  float score_with_idf(float idf_value, uint32_t term_freq, uint32_t doc_len,
                       float boost) const;

  /*! Update in-memory segment statistics (called by FtsColumnIndexer after
   *  each insert so that search() uses up-to-date stats for BM25 scoring)
   *  \param total_docs    Current total number of documents
   *  \param total_tokens  Current total number of tokens
   */
  void update_stats(uint64_t total_docs, uint64_t total_tokens) {
    // Store total_docs first so that a concurrent reader calling snapshot()
    // (which loads total_tokens before total_docs) never sees a new docs
    // count paired with a stale tokens count, which would deflate avg_doc_len.
    stats_.total_docs.store(total_docs, std::memory_order_release);
    stats_.total_tokens.store(total_tokens, std::memory_order_release);
  }

  SegmentStatsSnapshot stats() const {
    return stats_.snapshot();
  }
  const BM25Params &params() const {
    return params_;
  }

 private:
  BM25Params params_;
  SegmentStats stats_;
};

using BM25ScorerPtr = std::shared_ptr<BM25Scorer>;

/*! WAND optimizer
 *  Uses $MAX_TF as upper bound for TopK pruning, reduces unnecessary document
 * scoring
 */
class WandOptimizer {
 public:
  /*! Initialize WAND optimizer
   *  \param scorer        BM25 scorer (with segment statistics loaded)
   *  \param max_tf_cf     $MAX_TF CF (stores maximum term frequency for each
   * term) \param topk          Number of TopK results to return
   */
  int open(BM25ScorerPtr scorer, RocksdbContext *ctx,
           rocksdb::ColumnFamilyHandle *max_tf_cf, uint32_t topk);

  /*! Read the maximum term frequency for a term from $MAX_TF CF.
   *  Used by TermDocIterator to precompute WAND upper bound score.
   *  \param term  The term to look up
   *  \return Maximum term frequency, or 1 if not found
   */
  uint32_t read_max_tf(const std::string &term) const;

 private:
  BM25ScorerPtr scorer_;
  RocksdbContext *ctx_{nullptr};
  rocksdb::ColumnFamilyHandle *max_tf_cf_{nullptr};
  uint32_t topk_{10};
};

}  // namespace zvec::fts
