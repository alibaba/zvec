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

#include "bm25_scorer.h"
#include <cmath>
#include <cstring>
#include <zvec/ailego/logger/logger.h>
#include "fts_utils.h"

namespace zvec::fts {

// ============================================================
// BM25Scorer implementation
// ============================================================

int BM25Scorer::load_segment_stats(const std::string &field_name,
                                   RocksdbContext *ctx,
                                   rocksdb::ColumnFamilyHandle *stat_cf) {
  if (!ctx || !stat_cf) {
    LOG_WARN("BM25Scorer::load_segment_stats: null ctx/stat_cf for field[%s]",
             field_name.c_str());
    return -1;
  }

  // Read total_docs
  std::string total_docs_value;
  auto ret = ctx->db_->Get(ctx->read_opts_, stat_cf,
                           make_total_docs_key(field_name), &total_docs_value);
  if (!ret.ok()) {
    LOG_DEBUG(
        "BM25Scorer::load_segment_stats: failed to read total_docs. "
        "field[%s]",
        field_name.c_str());
    return -1;
  }
  if (total_docs_value.size() < sizeof(uint64_t)) {
    LOG_ERROR(
        "BM25Scorer::load_segment_stats: total_docs value too small. "
        "field[%s] value_size[%zu]",
        field_name.c_str(), total_docs_value.size());
    return -1;
  }
  uint64_t total_docs = decode_uint64_value(total_docs_value.data());
  stats_.total_docs.store(total_docs, std::memory_order_release);

  // Read total_tokens
  std::string total_tokens_value;
  auto status =
      ctx->db_->Get(ctx->read_opts_, stat_cf, make_total_tokens_key(field_name),
                    &total_tokens_value);
  if (!status.ok()) {
    LOG_DEBUG(
        "BM25Scorer::load_segment_stats: failed to read total_tokens. "
        "field[%s]",
        field_name.c_str());
    return -1;
  }
  if (total_tokens_value.size() < sizeof(uint64_t)) {
    LOG_ERROR(
        "BM25Scorer::load_segment_stats: total_tokens value too small. "
        "field[%s] value_size[%zu]",
        field_name.c_str(), total_tokens_value.size());
    return -1;
  }
  uint64_t total_tokens = decode_uint64_value(total_tokens_value.data());
  stats_.total_tokens.store(total_tokens, std::memory_order_release);

  return 0;
}

float BM25Scorer::idf(uint64_t term_doc_freq) const {
  const auto snap = stats_.snapshot();
  if (snap.total_docs == 0) {
    return 0.0f;
  }
  // Robertson-Sparck Jones IDF formula (with smoothing):
  // IDF(t) = ln((N - df + 0.5) / (df + 0.5) + 1)
  const float total_docs = static_cast<float>(snap.total_docs);
  const float df = static_cast<float>(term_doc_freq);
  return std::log((total_docs - df + 0.5f) / (df + 0.5f) + 1.0f);
}

float BM25Scorer::max_score_bound(uint64_t term_doc_freq) const {
  const float idf_value = idf(term_doc_freq);
  if (idf_value <= 0.0f) {
    return 0.0f;
  }
  // tf→infinity limit: tf_norm → (k1 + 1), so idf*(k1+1) upper-bounds the
  // score for any (tf, doc_len).
  return idf_value * (params_.k1 + 1.0f);
}

float BM25Scorer::score(uint64_t term_doc_freq, uint32_t term_freq,
                        uint32_t doc_len) const {
  // Take a single snapshot so that IDF and TF normalization use the same
  // consistent values of total_docs / total_tokens.
  const auto snap = stats_.snapshot();
  if (snap.total_docs == 0) {
    return 0.0f;
  }

  // IDF
  const float total_docs = static_cast<float>(snap.total_docs);
  const float df = static_cast<float>(term_doc_freq);
  const float idf_value =
      std::log((total_docs - df + 0.5f) / (df + 0.5f) + 1.0f);
  if (idf_value <= 0.0f) {
    return 0.0f;
  }

  // TF normalization
  const float tf = static_cast<float>(term_freq);
  const float doc_length = static_cast<float>(doc_len);
  const float avg_dl = snap.avg_doc_len();

  // BM25 TF normalization formula:
  // tf_norm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))
  const float tf_norm =
      tf * (params_.k1 + 1.0f) /
      (tf + params_.k1 * (1.0f - params_.b + params_.b * doc_length / avg_dl));

  return idf_value * tf_norm;
}

float BM25Scorer::score_with_idf(float idf_value, uint32_t term_freq,
                                 uint32_t doc_len) const {
  return score_with_idf(idf_value, term_freq, doc_len, 1.0f);
}

float BM25Scorer::score_with_idf(float idf_value, uint32_t term_freq,
                                 uint32_t doc_len, float boost) const {
  if (idf_value <= 0.0f) {
    return 0.0f;
  }
  const auto snap = stats_.snapshot();
  if (snap.total_docs == 0) {
    return 0.0f;
  }

  const float tf = static_cast<float>(term_freq);
  const float doc_length = static_cast<float>(doc_len);
  const float avg_dl = snap.avg_doc_len();

  const float tf_norm =
      tf * (params_.k1 + 1.0f) /
      (tf + params_.k1 * (1.0f - params_.b + params_.b * doc_length / avg_dl));

  return boost * idf_value * tf_norm;
}

// ============================================================
// WandOptimizer implementation
// ============================================================

int WandOptimizer::open(BM25ScorerPtr scorer, RocksdbContext *ctx,
                        rocksdb::ColumnFamilyHandle *max_tf_cf, uint32_t topk) {
  if (!scorer || !ctx || !max_tf_cf) {
    LOG_ERROR(
        "WandOptimizer open failed: null arguments scorer[%p] ctx[%p] "
        "max_tf_cf[%p]",
        (void *)scorer.get(), (void *)ctx, (void *)max_tf_cf);
    return -1;
  }
  scorer_ = std::move(scorer);
  ctx_ = ctx;
  max_tf_cf_ = max_tf_cf;
  topk_ = topk;
  return 0;
}

uint32_t WandOptimizer::read_max_tf(const std::string &term) const {
  if (!max_tf_cf_) {
    return 1;
  }
  std::string max_tf_value;
  if (!ctx_->db_->Get(ctx_->read_opts_, max_tf_cf_, term, &max_tf_value).ok() ||
      max_tf_value.size() < sizeof(uint32_t)) {
    return 1;  // Default max term frequency is 1
  }
  uint32_t max_tf = 0;
  std::memcpy(&max_tf, max_tf_value.data(), sizeof(uint32_t));
  return max_tf;
}

}  // namespace zvec::fts
