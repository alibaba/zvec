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

#include <cstdint>
#include <map>
#include <memory>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include "db/index/column/fts_column/iterator/fts_conjunction_iterator.h"
#include "db/index/column/fts_column/iterator/fts_disjunction_iterator.h"
#include "db/index/column/fts_column/iterator/fts_doc_iterator.h"

using zvec::fts::ConjunctionIterator;
using zvec::fts::DisjunctionIterator;
using zvec::fts::DocIterator;
using zvec::fts::DocIteratorPtr;

namespace {

constexpr uint32_t kNoMore = DocIterator::NO_MORE_DOCS;

// ---------------------------------------------------------------------------
// FakeDocIterator: controllable posting list with per-doc scores and
//                  per-block max score metadata.
// ---------------------------------------------------------------------------
class FakeDocIterator : public DocIterator {
 public:
  // Each entry: {doc_id, score}
  // block_maxes: {block_last_doc -> block_max_score}
  FakeDocIterator(std::vector<std::pair<uint32_t, float>> entries,
                  float max_score_val,
                  std::map<uint32_t, float> block_maxes = {})
      : entries_(std::move(entries)),
        max_score_val_(max_score_val),
        block_maxes_(std::move(block_maxes)) {
    cached_max_score_ = max_score_val_;
  }

  uint32_t next_doc() override {
    ++pos_;
    if (pos_ >= entries_.size()) {
      cached_doc_id_ = kNoMore;
      return kNoMore;
    }
    cached_doc_id_ = entries_[pos_].first;
    return cached_doc_id_;
  }

  uint32_t advance(uint32_t target) override {
    while (pos_ + 1 < entries_.size() && entries_[pos_ + 1].first < target) {
      ++pos_;
    }
    return next_doc();
  }

  float score() override {
    ++score_call_count_;
    if (pos_ < entries_.size()) {
      return entries_[pos_].second;
    }
    return 0.0f;
  }

  uint64_t cost() const override {
    return entries_.size();
  }

  float max_score() const override {
    return max_score_val_;
  }

  BlockMaxInfo block_max_info_for(uint32_t target) const override {
    if (block_maxes_.empty()) {
      return {max_score_val_, kNoMore};
    }
    // Find the first block whose last_doc >= target
    for (const auto &[last_doc, bm_score] : block_maxes_) {
      if (last_doc >= target) {
        return {bm_score, last_doc};
      }
    }
    return {0.0f, kNoMore};
  }

  int score_call_count() const {
    return score_call_count_;
  }

 private:
  std::vector<std::pair<uint32_t, float>> entries_;
  float max_score_val_;
  std::map<uint32_t, float> block_maxes_;
  size_t pos_{SIZE_MAX};  // before first element
  int score_call_count_{0};
};

// Collect all doc_ids from an iterator
std::vector<uint32_t> collect_docs(DocIterator *iter) {
  std::vector<uint32_t> docs;
  uint32_t doc = iter->next_doc();
  while (doc != kNoMore) {
    if (iter->matches()) {
      docs.push_back(doc);
    }
    doc = iter->next_doc();
  }
  return docs;
}

}  // namespace

// ============================================================
// Optimization 1: Block-Max skip
// ============================================================

// Block 0 [0..127] block_max sum = 1.0 < threshold 2.0 → skipped
// Block 1 [128..255] block_max sum = 10.0 >= 2.0 → kept
TEST(ConjunctionOptTest, BlockMaxSkipNonCompetitiveBlocks) {
  std::vector<std::pair<uint32_t, float>> list1 = {
      {10, 0.3f}, {50, 0.4f}, {130, 4.0f}, {200, 3.5f}};
  std::vector<std::pair<uint32_t, float>> list2 = {
      {10, 0.2f}, {50, 0.3f}, {130, 5.0f}, {200, 4.0f}};

  std::map<uint32_t, float> bm1 = {{127, 0.5f}, {255, 5.0f}};
  std::map<uint32_t, float> bm2 = {{127, 0.5f}, {255, 5.0f}};

  std::vector<DocIteratorPtr> musts;
  musts.push_back(std::make_unique<FakeDocIterator>(list1, 5.0f, bm1));
  musts.push_back(std::make_unique<FakeDocIterator>(list2, 5.0f, bm2));

  ConjunctionIterator conj(std::move(musts), std::vector<DocIteratorPtr>{});
  conj.set_min_competitive_score(2.0f);

  auto docs = collect_docs(&conj);
  ASSERT_EQ(docs.size(), 2u);
  EXPECT_EQ(docs[0], 130u);
  EXPECT_EQ(docs[1], 200u);
}

// 3 blocks, threshold skips block 0 and block 1, only block 2 survives
TEST(ConjunctionOptTest, BlockMaxSkipMultipleBlocks) {
  std::vector<std::pair<uint32_t, float>> list1 = {
      {10, 0.1f}, {130, 1.5f}, {260, 4.0f}};
  std::vector<std::pair<uint32_t, float>> list2 = {
      {10, 0.1f}, {130, 1.0f}, {260, 4.5f}};

  std::map<uint32_t, float> bm1 = {{127, 0.5f}, {255, 2.0f}, {383, 5.0f}};
  std::map<uint32_t, float> bm2 = {{127, 0.5f}, {255, 2.0f}, {383, 5.0f}};

  std::vector<DocIteratorPtr> musts;
  musts.push_back(std::make_unique<FakeDocIterator>(list1, 5.0f, bm1));
  musts.push_back(std::make_unique<FakeDocIterator>(list2, 5.0f, bm2));

  ConjunctionIterator conj(std::move(musts), std::vector<DocIteratorPtr>{});
  // block 0 sum=1.0, block 1 sum=4.0, block 2 sum=10.0; threshold=5.0
  conj.set_min_competitive_score(5.0f);

  auto docs = collect_docs(&conj);
  ASSERT_EQ(docs.size(), 1u);
  EXPECT_EQ(docs[0], 260u);
}

// ============================================================
// Optimization 3: Score early-exit
// ============================================================

// 3 must iterators sorted by cost (ascending). After scoring the first
// (lowest-cost) iterator, accumulated + remaining_max < threshold →
// score() exits early, skipping the remaining two score() calls.
TEST(ConjunctionOptTest, ScoreEarlyExitReducesScoreCalls) {
  // iter0: cost=1, max_score=2.0, actual_score=0.1
  // iter1: cost=2, max_score=2.0, actual_score=1.0
  // iter2: cost=3, max_score=2.0, actual_score=1.5
  // cached_max_score_ = 6.0, threshold = 5.5
  // After iter0: remaining_max=4.0, total=0.1. 0.1+4.0=4.1 < 5.5 → exit
  auto *raw0 = new FakeDocIterator({{1, 0.1f}}, 2.0f);
  auto *raw1 = new FakeDocIterator({{1, 1.0f}}, 2.0f);
  auto *raw2 = new FakeDocIterator({{1, 1.5f}}, 2.0f);

  std::vector<DocIteratorPtr> musts;
  musts.emplace_back(raw0);
  musts.emplace_back(raw1);
  musts.emplace_back(raw2);

  ConjunctionIterator conj(std::move(musts), std::vector<DocIteratorPtr>{});
  conj.set_min_competitive_score(5.5f);

  uint32_t doc = conj.next_doc();
  ASSERT_EQ(doc, 1u);
  conj.score();

  // Only the first iterator's score() should have been called;
  // the other two were skipped by early-exit.
  EXPECT_EQ(raw0->score_call_count(), 1);
  EXPECT_EQ(raw1->score_call_count(), 0);
  EXPECT_EQ(raw2->score_call_count(), 0);
}

// When scores are competitive, all iterators' score() are called
TEST(ConjunctionOptTest, ScoreNoEarlyExitCallsAll) {
  auto *raw0 = new FakeDocIterator({{1, 3.0f}}, 3.0f);
  auto *raw1 = new FakeDocIterator({{1, 3.0f}}, 3.0f);

  std::vector<DocIteratorPtr> musts;
  musts.emplace_back(raw0);
  musts.emplace_back(raw1);

  ConjunctionIterator conj(std::move(musts), std::vector<DocIteratorPtr>{});
  conj.set_min_competitive_score(5.0f);

  uint32_t doc = conj.next_doc();
  ASSERT_EQ(doc, 1u);
  float s = conj.score();

  EXPECT_FLOAT_EQ(s, 6.0f);
  EXPECT_EQ(raw0->score_call_count(), 1);
  EXPECT_EQ(raw1->score_call_count(), 1);
}

// ============================================================
// Optimization 4: Phrase threshold forwarding
// ============================================================

// set_min_competitive_score propagated into inner conjunction triggers
// block-max skip; without forwarding all docs would be returned.
TEST(ConjunctionOptTest, PhraseForwardingBlockMaxSkip) {
  std::vector<std::pair<uint32_t, float>> list1 = {{10, 0.2f}, {130, 4.0f}};
  std::vector<std::pair<uint32_t, float>> list2 = {{10, 0.2f}, {130, 5.0f}};

  std::map<uint32_t, float> bm1 = {{127, 0.3f}, {255, 5.0f}};
  std::map<uint32_t, float> bm2 = {{127, 0.3f}, {255, 5.0f}};

  std::vector<DocIteratorPtr> musts;
  musts.push_back(std::make_unique<FakeDocIterator>(list1, 5.0f, bm1));
  musts.push_back(std::make_unique<FakeDocIterator>(list2, 5.0f, bm2));

  auto inner = std::make_unique<ConjunctionIterator>(
      std::move(musts), std::vector<DocIteratorPtr>{});

  // Forward threshold as PhraseDocIterator would
  inner->set_min_competitive_score(2.0f);

  auto docs = collect_docs(inner.get());
  // Block 0 (sum 0.6 < 2.0) skipped
  ASSERT_EQ(docs.size(), 1u);
  EXPECT_EQ(docs[0], 130u);
}

// ============================================================
// DisjunctionIterator::advance() bypass WAND fix
// ============================================================

// advance() must faithfully return target even with high min_competitive_score.
// Without the fix, advance() delegates to next_doc() which triggers WAND
// pruning and returns NO_MORE_DOCS.
TEST(ConjunctionOptTest, DisjunctionAdvanceBypassesWand) {
  std::vector<DocIteratorPtr> sub_iters;
  sub_iters.push_back(std::make_unique<FakeDocIterator>(
      std::vector<std::pair<uint32_t, float>>{{1, 0.1f}, {5, 0.1f}, {10, 0.1f}},
      1.0f));
  sub_iters.push_back(std::make_unique<FakeDocIterator>(
      std::vector<std::pair<uint32_t, float>>{{3, 0.2f}, {5, 0.2f}, {20, 0.2f}},
      1.0f));

  DisjunctionIterator disj(std::move(sub_iters));
  disj.set_min_competitive_score(100.0f);

  uint32_t doc = disj.advance(5);
  EXPECT_EQ(doc, 5u);
  EXPECT_TRUE(disj.matches());

  doc = disj.advance(10);
  EXPECT_EQ(doc, 10u);
}

TEST(ConjunctionOptTest, ConjunctionAdvanceBypassesCompetitivePruning) {
  std::vector<std::pair<uint32_t, float>> list1 = {{5, 0.1f}, {130, 5.0f}};
  std::vector<std::pair<uint32_t, float>> list2 = {{5, 0.1f}, {130, 5.0f}};

  std::map<uint32_t, float> bm1 = {{127, 0.5f}, {255, 5.0f}};
  std::map<uint32_t, float> bm2 = {{127, 0.5f}, {255, 5.0f}};

  std::vector<DocIteratorPtr> musts;
  musts.push_back(std::make_unique<FakeDocIterator>(list1, 5.0f, bm1));
  musts.push_back(std::make_unique<FakeDocIterator>(list2, 5.0f, bm2));

  ConjunctionIterator conj(std::move(musts), std::vector<DocIteratorPtr>{});
  conj.set_min_competitive_score(2.0f);

  uint32_t doc = conj.advance(5);
  EXPECT_EQ(doc, 5u);
  EXPECT_TRUE(conj.matches());
}
