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

#include "db/index/column/fts_column/iterator/fts_candidate_iterator.h"
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include "db/index/column/fts_column/iterator/fts_doc_iterator.h"

using zvec::fts::CandidateDocIterator;
using zvec::fts::DocIterator;

namespace {

constexpr uint32_t kNoMore = DocIterator::NO_MORE_DOCS;

}  // namespace

TEST(CandidateDocIteratorTest, EmptyVectorYieldsNothing) {
  CandidateDocIterator it({});
  EXPECT_EQ(it.cost(), 0u);
  EXPECT_EQ(it.next_doc(), kNoMore);
  EXPECT_EQ(it.next_doc(), kNoMore);
}

TEST(CandidateDocIteratorTest, NextDocStreamsAscending) {
  CandidateDocIterator it({0, 5, 10, 100});
  EXPECT_EQ(it.cost(), 4u);
  EXPECT_FLOAT_EQ(it.max_score(), 0.0f);
  EXPECT_FLOAT_EQ(it.score(), 0.0f);
  EXPECT_TRUE(it.matches());

  EXPECT_EQ(it.next_doc(), 0u);
  EXPECT_EQ(it.doc_id(), 0u);
  EXPECT_EQ(it.next_doc(), 5u);
  EXPECT_EQ(it.next_doc(), 10u);
  EXPECT_EQ(it.next_doc(), 100u);
  EXPECT_EQ(it.next_doc(), kNoMore);
  EXPECT_EQ(it.next_doc(), kNoMore);
}

TEST(CandidateDocIteratorTest, AdvanceLandsOnExactMatch) {
  CandidateDocIterator it({10, 20, 30, 40, 50});
  EXPECT_EQ(it.advance(20), 20u);
  EXPECT_EQ(it.doc_id(), 20u);
  // Subsequent next_doc continues past the advanced position.
  EXPECT_EQ(it.next_doc(), 30u);
}

TEST(CandidateDocIteratorTest, AdvanceSeeksToNextHigher) {
  CandidateDocIterator it({10, 20, 30, 40, 50});
  EXPECT_EQ(it.advance(25), 30u);
  EXPECT_EQ(it.next_doc(), 40u);
}

TEST(CandidateDocIteratorTest, AdvancePastLastYieldsNoMore) {
  CandidateDocIterator it({10, 20, 30});
  EXPECT_EQ(it.advance(50), kNoMore);
  EXPECT_EQ(it.next_doc(), kNoMore);
}

TEST(CandidateDocIteratorTest, AdvanceBeforeAnyConsumeWorks) {
  CandidateDocIterator it({10, 20, 30});
  EXPECT_EQ(it.advance(0), 10u);
  EXPECT_EQ(it.next_doc(), 20u);
}

TEST(CandidateDocIteratorTest, AdvanceInterleavedWithNext) {
  CandidateDocIterator it({5, 10, 15, 20, 25, 30});
  EXPECT_EQ(it.next_doc(), 5u);
  EXPECT_EQ(it.advance(15), 15u);
  EXPECT_EQ(it.next_doc(), 20u);
  EXPECT_EQ(it.advance(99), kNoMore);
}

TEST(CandidateDocIteratorTest, SingleElement) {
  CandidateDocIterator it({42});
  EXPECT_EQ(it.cost(), 1u);
  EXPECT_EQ(it.advance(42), 42u);
  EXPECT_EQ(it.next_doc(), kNoMore);
}

TEST(CandidateDocIteratorTest, AdvanceCachesDocId) {
  CandidateDocIterator it({1, 2, 3});
  EXPECT_EQ(it.advance(2), 2u);
  EXPECT_EQ(it.doc_id(), 2u);
}
