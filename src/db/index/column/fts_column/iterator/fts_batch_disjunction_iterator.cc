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

#include "fts_batch_disjunction_iterator.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace zvec::fts {
namespace {

class BatchDisjunction final : public DocIterator {
 public:
  explicit BatchDisjunction(BM25ScorerPtr scorer) : scorer_(std::move(scorer)) {
    cached_max_score_ = std::numeric_limits<float>::infinity();
  }

  Status open(const std::vector<std::pair<FtsReadTerm::Ptr, float>> &terms) {
    clauses_.reserve(terms.size());
    for (const auto &[term, boost] : terms) {
      if (!term || !std::isfinite(boost) || boost < 0) {
        return Status::InvalidArgument("Invalid batched FTS term");
      }
      auto posting = term->postings();
      if (!posting) {
        return posting.error();
      }
      Clause clause;
      clause.posting = std::move(*posting);
      clause.idf = scorer_->idf(term->df());
      clause.boost = boost;
      auto status = refill(clause);
      if (!status.ok()) {
        return status;
      }
      cost_ += std::min(term->df(), UINT64_MAX - cost_);
      clauses_.push_back(std::move(clause));
    }
    return Status::OK();
  }

  uint32_t next_doc() override {
    return next(true);
  }

  uint32_t advance(uint32_t target) override {
    if (started_ && cached_doc_id_ >= target) {
      return cached_doc_id_;
    }
    uint32_t doc;
    do {
      doc = next(false);
    } while (doc < target);
    return doc;
  }

  float score() override {
    return score_;
  }

  uint64_t cost() const override {
    return cost_;
  }

  float max_score() const override {
    return cached_max_score_;
  }

  Status status() const override {
    return status_;
  }

  void set_min_competitive_score(float value) override {
    threshold_ = std::max(0.0f, value);
    competitive_ = true;
  }

 private:
  static constexpr uint32_t kWindowSize = 4096;

  struct Clause {
    FtsReadPosting::Ptr posting;
    FtsPostingBatch batch;
    std::array<float, FtsPostingBatch::kCapacity> scores{};
    float idf{0}, boost{1};
    uint32_t offset{0};
    bool eof{false};
  };

  Status refill(Clause &clause) {
    auto batch = clause.posting->next_batch();
    if (!batch) {
      return batch.error();
    }
    clause.batch = *batch;
    clause.offset = 0;
    if (batch->count == 0) {
      clause.eof = true;
      return Status::OK();
    }
    if (batch->count > FtsPostingBatch::kCapacity || !batch->docs ||
        !batch->tfs || !batch->lengths) {
      return Status::InternalError("Invalid FTS posting batch");
    }
    scorer_->score_batch(clause.idf, batch->tfs, batch->lengths, batch->count,
                         clause.boost, clause.scores.data());
    return Status::OK();
  }

  bool load_window() {
    uint32_t first = NO_MORE_DOCS;
    for (const auto &clause : clauses_) {
      if (!clause.eof) {
        first = std::min(first, clause.batch.docs[clause.offset]);
      }
    }
    if (first == NO_MORE_DOCS) {
      return false;
    }
    base_ = first & ~(kWindowSize - 1);
    const uint64_t end = uint64_t{base_} + kWindowSize;
    scores_.fill(0);
    present_.fill(0);
    // Always add clauses in their original order. Sorting clauses by cost,
    // batching by arrival, or changing the FP32 sum would alter tied TopK.
    for (auto &clause : clauses_) {
      while (!clause.eof && clause.batch.docs[clause.offset] < end) {
        while (clause.offset < clause.batch.count &&
               clause.batch.docs[clause.offset] < end) {
          const uint32_t index = clause.batch.docs[clause.offset] - base_;
          scores_[index] += clause.scores[clause.offset];
          present_[index / 64] |= uint64_t{1} << (index % 64);
          ++clause.offset;
        }
        if (clause.offset == clause.batch.count) {
          status_ = refill(clause);
          if (!status_.ok()) {
            return false;
          }
        }
      }
    }
    position_ = 0;
    return true;
  }

  uint32_t next(bool prune) {
    if (!status_.ok() || eof_) {
      return cached_doc_id_ = NO_MORE_DOCS;
    }
    while (true) {
      while (position_ < kWindowSize) {
        const uint32_t index = position_++;
        // Absent rows have zero scores and the competitive threshold is
        // nonnegative, so an exact competitive score also proves presence.
        // Unpruned positioning must still include matching zero-score rows.
        const bool candidate =
            prune && competitive_
                ? scores_[index] > threshold_
                : (present_[index / 64] & (uint64_t{1} << (index % 64))) != 0;
        if (candidate) {
          score_ = scores_[index];
          started_ = true;
          return cached_doc_id_ = base_ + index;
        }
      }
      if (!load_window()) {
        eof_ = true;
        score_ = 0;
        return cached_doc_id_ = NO_MORE_DOCS;
      }
    }
  }

  BM25ScorerPtr scorer_;
  std::vector<Clause> clauses_;
  std::array<float, kWindowSize> scores_{};
  std::array<uint64_t, kWindowSize / 64> present_{};
  uint32_t base_{0}, position_{kWindowSize};
  float threshold_{0}, score_{0};
  uint64_t cost_{0};
  bool competitive_{false}, started_{false}, eof_{false};
  Status status_;
};

}  // namespace

Result<DocIteratorPtr> CreateBatchDisjunction(
    std::vector<std::pair<FtsReadTerm::Ptr, float>> terms,
    BM25ScorerPtr scorer) {
  if (!scorer) {
    return tl::make_unexpected(Status::InvalidArgument("Missing FTS scorer"));
  }
  if (terms.size() > 1024) {
    return tl::make_unexpected(Status{StatusCode::RESOURCE_EXHAUSTED,
                                      "FTS batch query term budget exceeded"});
  }
  if (terms.empty()) {
    return DocIteratorPtr{};
  }
  auto iterator = std::make_unique<BatchDisjunction>(std::move(scorer));
  auto status = iterator->open(terms);
  if (!status.ok()) {
    return tl::make_unexpected(status);
  }
  return iterator;
}

}  // namespace zvec::fts
