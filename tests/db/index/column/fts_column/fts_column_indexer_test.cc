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

#include "db/index/column/fts_column/fts_column_indexer.h"
#include <algorithm>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/db/config.h>
#include <zvec/db/index_params.h>
#include "db/common/file_helper.h"
#include "db/index/common/index_filter.h"
// FtsQueryParams defined below
#include "db/index/column/fts_column/fts_ast_rewriter.h"
#include "db/index/column/fts_column/fts_rocksdb_merge.h"
#include "db/index/column/fts_column/parser/fts_query_parser.h"
#include "db/index/column/fts_column/tokenizer/tokenizer_factory.h"
// meta.h not needed in zvec
#include "db/common/constants.h"
#include "db/common/rocksdb_context.h"

using namespace zvec;
using namespace zvec::fts;

namespace {

// Build a transient FieldSchema for FTS unit tests.
// When fts_params is provided, it is attached as the field's index_params
// so that FtsColumnIndexer::open() can retrieve the tokenizer configuration.
FieldSchema::Ptr make_test_field_meta(
    const std::string &field_name,
    std::shared_ptr<zvec::FtsIndexParams> fts_params = nullptr) {
  if (fts_params) {
    return std::make_shared<FieldSchema>(field_name, DataType::STRING, false,
                                         fts_params);
  }
  return std::make_shared<FieldSchema>(field_name, DataType::STRING);
}

}  // namespace

// Build a tokenizer pipeline matching the indexer config used by the tests.
// A standalone helper so tests can pass it to parser.parse() without
// reaching into FtsColumnIndexer internals.
static zvec::fts::TokenizerPipelinePtr make_whitespace_pipeline() {
  zvec::fts::FtsIndexParams params;
  params.tokenizer_name = "whitespace";
  params.filters = {"lowercase"};
  return zvec::fts::TokenizerFactory::create(params);
}

// Helper: parse a query string and call search() on a reader/indexer.
// Terminates the test with ASSERT if parsing fails.
template <typename Reader>
static bool search_ok(Reader &reader, const std::string &query_str,
                      uint32_t topk, std::vector<FtsResult> *results,
                      const zvec::fts::TokenizerPipelinePtr &pipeline =
                          make_whitespace_pipeline()) {
  FtsQueryParser parser;
  auto ast = parser.parse(query_str, pipeline);
  if (!ast) {
    ADD_FAILURE() << "FtsQueryParser failed to parse: " << query_str
                  << " err: " << parser.err_msg();
    return false;
  }
  // Apply the same AST rewrite the production sqlengine path runs so that
  // FtsColumnIndexer::search() sees a canonical AST (no must_not children
  // inside an OrNode, dedup-collapsed siblings, etc.).
  zvec::fts::simplify(ast);
  zvec::fts::FtsQueryParams qp;
  qp.topk = topk;
  auto ret = reader.search(*ast, qp);
  if (!ret.has_value()) {
    return false;
  }
  *results = std::move(ret.value());
  return true;
}

// Helper: parse a query string with a filter and call search().
template <typename Reader>
static bool search_ok_with_filter(Reader &reader, const std::string &query_str,
                                  uint32_t topk, zvec::IndexFilter::Ptr filter,
                                  std::vector<FtsResult> *results,
                                  const zvec::fts::TokenizerPipelinePtr
                                      &pipeline = make_whitespace_pipeline()) {
  FtsQueryParser parser;
  auto ast = parser.parse(query_str, pipeline);
  if (!ast) {
    ADD_FAILURE() << "FtsQueryParser failed to parse: " << query_str
                  << " err: " << parser.err_msg();
    return false;
  }
  zvec::fts::simplify(ast);
  zvec::fts::FtsQueryParams qp;
  qp.topk = topk;
  qp.filter = std::move(filter);
  auto ret = reader.search(*ast, qp);
  if (!ret.has_value()) {
    return false;
  }
  *results = std::move(ret.value());
  return true;
}

// ============================================================
// Test fixture
// ============================================================

static const std::string kDbPath{"./test_fts_db"};

static const std::string kPostingsCf{"fts"};
static const std::string kMaxTfCf{kPostingsCf + zvec::kFtsMaxTfSuffix};
static const std::string kPositionsCf{kPostingsCf + zvec::kFtsPositionsSuffix};
static const std::string kTermFreqCf{kPostingsCf + zvec::kFtsTfSuffix};
static const std::string kDocLenCf{kPostingsCf + zvec::kFtsDocLenSuffix};
static const std::string kStatCf{zvec::kFtsStatCfName};

class FtsColumnIndexerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    zvec::FileHelper::RemoveDirectory(kDbPath);

    // Single RocksDB instance with per-CF merge operators.
    std::vector<std::string> cf_names = {kPostingsCf, kMaxTfCf,  kPositionsCf,
                                         kTermFreqCf, kDocLenCf, kStatCf};
    std::unordered_map<std::string, std::shared_ptr<rocksdb::MergeOperator>>
        per_cf_ops = {
            {kPostingsCf, std::make_shared<FtsPostingsMerge>()},
            {kMaxTfCf, std::make_shared<FtsMaxTfMerge>()},
        };
    ASSERT_TRUE(
        db_.create(RocksdbContext::Args{kDbPath, cf_names, nullptr, per_cf_ops})
            .ok());

    postings_cf_ = db_.get_cf(kPostingsCf);
    max_tf_cf_ = db_.get_cf(kMaxTfCf);
    positions_cf_ = db_.get_cf(kPositionsCf);
    term_freq_cf_ = db_.get_cf(kTermFreqCf);
    doc_len_cf_ = db_.get_cf(kDocLenCf);
    stat_cf_ = db_.get_cf(kStatCf);

    ASSERT_NE(postings_cf_, nullptr);
    ASSERT_NE(max_tf_cf_, nullptr);
    ASSERT_NE(positions_cf_, nullptr);
    ASSERT_NE(term_freq_cf_, nullptr);
    ASSERT_NE(doc_len_cf_, nullptr);
    ASSERT_NE(stat_cf_, nullptr);
  }

  void TearDown() override {
    db_.close();
    zvec::FileHelper::RemoveDirectory(kDbPath);
  }

  // Create and open a fresh indexer with whitespace tokenizer.
  // Returns unique_ptr because FtsColumnIndexer is not copyable (atomic
  // members).
  std::unique_ptr<FtsColumnIndexer> make_indexer(
      const std::string &field_name = "content") {
    auto fts_params = std::make_shared<zvec::FtsIndexParams>("whitespace");
    auto field_meta = make_test_field_meta(field_name, fts_params);
    auto indexer = std::make_unique<FtsColumnIndexer>();
    auto ret = indexer->open(field_meta, &db_, postings_cf_, positions_cf_,
                             term_freq_cf_, max_tf_cf_, doc_len_cf_, stat_cf_);
    EXPECT_TRUE(ret.has_value());
    return indexer;
  }

  RocksdbContext db_;

  rocksdb::ColumnFamilyHandle *postings_cf_{nullptr};
  rocksdb::ColumnFamilyHandle *max_tf_cf_{nullptr};
  rocksdb::ColumnFamilyHandle *positions_cf_{nullptr};
  rocksdb::ColumnFamilyHandle *term_freq_cf_{nullptr};
  rocksdb::ColumnFamilyHandle *doc_len_cf_{nullptr};
  rocksdb::ColumnFamilyHandle *stat_cf_{nullptr};
};
// ============================================================
// open()
// ============================================================

TEST_F(FtsColumnIndexerTest, OpenWithValidTokenizer) {
  auto fts_params = std::make_shared<zvec::FtsIndexParams>("whitespace");
  auto field_meta = make_test_field_meta("content", fts_params);
  FtsColumnIndexer indexer;
  auto ret = indexer.open(field_meta, &db_, postings_cf_, positions_cf_,
                          term_freq_cf_, max_tf_cf_, doc_len_cf_, stat_cf_);
  EXPECT_TRUE(ret.has_value());
  EXPECT_EQ(indexer.total_docs(), 0u);
  EXPECT_EQ(indexer.total_tokens(), 0u);
}

TEST_F(FtsColumnIndexerTest, OpenWithNullFieldMetaFails) {
  FtsColumnIndexer indexer;
  auto ret =
      indexer.open(FieldSchema::Ptr{nullptr}, &db_, postings_cf_, positions_cf_,
                   term_freq_cf_, max_tf_cf_, doc_len_cf_, stat_cf_);
  EXPECT_FALSE(ret.has_value());
}

TEST_F(FtsColumnIndexerTest, OpenWithNullStoreFails) {
  auto fts_params = std::make_shared<zvec::FtsIndexParams>("whitespace");
  auto field_meta = make_test_field_meta("content", fts_params);
  FtsColumnIndexer indexer;
  auto ret =
      indexer.open(field_meta, /*store=*/nullptr, postings_cf_, positions_cf_,
                   term_freq_cf_, max_tf_cf_, doc_len_cf_, stat_cf_);
  EXPECT_FALSE(ret.has_value());
}

// ============================================================
// insert() - statistics update
// ============================================================

TEST_F(FtsColumnIndexerTest, InsertUpdatesTotalDocs) {
  auto indexer = make_indexer();

  EXPECT_TRUE(indexer->insert(0, "hello world").has_value());
  EXPECT_EQ(indexer->total_docs(), 1u);

  EXPECT_TRUE(indexer->insert(1, "foo bar baz").has_value());
  EXPECT_EQ(indexer->total_docs(), 2u);
}

TEST_F(FtsColumnIndexerTest, InsertUpdatesTotalTokens) {
  auto indexer = make_indexer();

  EXPECT_TRUE(indexer->insert(0, "hello world").has_value());
  EXPECT_EQ(indexer->total_tokens(), 2u);  // "hello", "world"

  EXPECT_TRUE(indexer->insert(1, "foo bar baz").has_value());
  EXPECT_EQ(indexer->total_tokens(), 5u);  // 2 + 3
}

TEST_F(FtsColumnIndexerTest, InsertEmptyTextCountsAsZeroTokens) {
  auto indexer = make_indexer();

  EXPECT_TRUE(indexer->insert(0, "").has_value());
  EXPECT_EQ(indexer->total_docs(), 1u);
  EXPECT_EQ(indexer->total_tokens(), 0u);
}

// ============================================================
// flush() - persist stats to RocksDB
// ============================================================

TEST_F(FtsColumnIndexerTest, FlushPersistsStats) {
  auto indexer = make_indexer("content");
  EXPECT_TRUE(indexer->insert(0, "hello world").has_value());
  EXPECT_TRUE(indexer->insert(1, "foo bar").has_value());
  EXPECT_TRUE(indexer->flush().has_value());

  // Verify stats were written to stat_cf by opening a standalone reader.
  // Pass doc_len_cf as nullptr so the reader loads stats from stat_cf.
  FtsColumnIndexer reader;
  auto ret = reader.open_reader("content", &db_, postings_cf_, positions_cf_,
                                term_freq_cf_, max_tf_cf_,
                                /*doc_len_cf=*/nullptr, stat_cf_);
  EXPECT_TRUE(ret.has_value());
  // Reader loads stats from stat_cf on open; search should succeed
  std::vector<FtsResult> results;
  EXPECT_TRUE(search_ok(reader, "hello", 10, &results));
  ASSERT_EQ(results.size(), 1u);
}

// ============================================================
// search() - term query
// ============================================================

TEST_F(FtsColumnIndexerTest, SearchTermFound) {
  auto indexer = make_indexer();
  EXPECT_TRUE(indexer->insert(0, "hello world").has_value());
  EXPECT_TRUE(indexer->insert(1, "hello foo").has_value());
  EXPECT_TRUE(indexer->insert(2, "bar baz").has_value());

  std::vector<FtsResult> results;
  EXPECT_TRUE(search_ok(*indexer, "hello", 10, &results));
  EXPECT_EQ(results.size(), 2u);

  bool found_doc0 = false;
  bool found_doc1 = false;
  for (const auto &result : results) {
    if (result.doc_id == 0) found_doc0 = true;
    if (result.doc_id == 1) found_doc1 = true;
  }
  EXPECT_TRUE(found_doc0);
  EXPECT_TRUE(found_doc1);
}

TEST_F(FtsColumnIndexerTest, SearchTermNotFound) {
  auto indexer = make_indexer();
  EXPECT_TRUE(indexer->insert(0, "hello world").has_value());

  std::vector<FtsResult> results;
  EXPECT_TRUE(search_ok(*indexer, "missing", 10, &results));
  EXPECT_TRUE(results.empty());
}

TEST_F(FtsColumnIndexerTest, SearchResultsSortedByScoreDescending) {
  auto indexer = make_indexer();
  // Doc 0: "hello" appears once
  EXPECT_TRUE(indexer->insert(0, "hello world").has_value());
  // Doc 1: "hello" appears twice (higher TF -> higher BM25 score)
  EXPECT_TRUE(indexer->insert(1, "hello hello").has_value());

  std::vector<FtsResult> results;
  EXPECT_TRUE(search_ok(*indexer, "hello", 10, &results));
  ASSERT_EQ(results.size(), 2u);

  // Results must be in descending score order
  EXPECT_GE(results[0].score, results[1].score);
  // Doc 1 (higher TF) should rank first
  EXPECT_EQ(results[0].doc_id, 1ull);
}

TEST_F(FtsColumnIndexerTest, SearchTopkLimitsResults) {
  auto indexer = make_indexer();
  for (uint64_t doc_id = 0; doc_id < 10; ++doc_id) {
    EXPECT_TRUE(indexer->insert(doc_id, "hello world").has_value());
  }

  std::vector<FtsResult> results;
  EXPECT_TRUE(search_ok(*indexer, "hello", 3, &results));
  EXPECT_LE(results.size(), 3u);
}

// ============================================================
// search() - phrase query
// ============================================================

TEST_F(FtsColumnIndexerTest, SearchPhraseFound) {
  auto indexer = make_indexer();
  EXPECT_TRUE(indexer->insert(0, "machine learning model").has_value());
  EXPECT_TRUE(indexer->insert(1, "learning machine translation").has_value());

  std::vector<FtsResult> results;
  EXPECT_TRUE(search_ok(*indexer, "\"machine learning\"", 10, &results));
  ASSERT_EQ(results.size(), 1u);
  EXPECT_EQ(results[0].doc_id, 0ull);
}

TEST_F(FtsColumnIndexerTest, SearchPhraseNotFound) {
  auto indexer = make_indexer();
  EXPECT_TRUE(indexer->insert(0, "hello world foo").has_value());

  std::vector<FtsResult> results;
  EXPECT_TRUE(search_ok(*indexer, "\"hello foo\"", 10, &results));
  EXPECT_TRUE(results.empty());
}

// Phrase with a repeated term ("a b a") exercises the dedup path in
// PhraseDocIterator::verify_phrase_positions: the two "a" entries must share
// a single MultiGet slot while still validating positions 0 and 2.
TEST_F(FtsColumnIndexerTest, SearchPhraseWithRepeatedTermFound) {
  auto indexer = make_indexer();
  EXPECT_TRUE(indexer->insert(0, "a b a").has_value());  // match
  EXPECT_TRUE(indexer->insert(1, "a b c").has_value());  // a b ✓, trailing a ✗
  EXPECT_TRUE(indexer->insert(2, "b a c").has_value());  // wrong order
  EXPECT_TRUE(indexer->insert(3, "a a b").has_value());  // wrong adjacency

  std::vector<FtsResult> results;
  EXPECT_TRUE(search_ok(*indexer, "\"a b a\"", 10, &results));
  ASSERT_EQ(results.size(), 1u);
  EXPECT_EQ(results[0].doc_id, 0ull);
}

// When the first phrase term is high-frequency in the doc (e.g., "the the the
// the model"), the anchor must be chosen from the rarest position list rather
// than terms_[0]; otherwise the anchor loop iterates many useless candidates.
// This test only asserts correctness — the anchor heuristic is internal — but
// guards against regressions in the shortest-list selection.
TEST_F(FtsColumnIndexerTest, SearchPhraseHighFrequencyLeadingTerm) {
  auto indexer = make_indexer();
  EXPECT_TRUE(indexer->insert(0, "the the the the model").has_value());
  EXPECT_TRUE(indexer->insert(1, "the model the the the").has_value());
  EXPECT_TRUE(
      indexer->insert(2, "the the the the the").has_value());  // no model

  std::vector<FtsResult> results;
  EXPECT_TRUE(search_ok(*indexer, "\"the model\"", 10, &results));
  ASSERT_EQ(results.size(), 2u);
  std::vector<uint64_t> ids{results[0].doc_id, results[1].doc_id};
  std::sort(ids.begin(), ids.end());
  EXPECT_EQ(ids[0], 0ull);
  EXPECT_EQ(ids[1], 1ull);
}

// ============================================================
// search() - boolean query (AND / OR)
// ============================================================

TEST_F(FtsColumnIndexerTest, SearchExplicitAnd) {
  auto indexer = make_indexer();
  EXPECT_TRUE(indexer->insert(0, "hello world").has_value());  // matches both
  EXPECT_TRUE(indexer->insert(1, "hello foo").has_value());    // only hello
  EXPECT_TRUE(indexer->insert(2, "world bar").has_value());    // only world

  std::vector<FtsResult> results;
  EXPECT_TRUE(search_ok(*indexer, "hello AND world", 10, &results));
  ASSERT_EQ(results.size(), 1u);
  EXPECT_EQ(results[0].doc_id, 0ull);
}

TEST_F(FtsColumnIndexerTest, SearchExplicitOr) {
  auto indexer = make_indexer();
  EXPECT_TRUE(indexer->insert(0, "hello world").has_value());
  EXPECT_TRUE(indexer->insert(1, "foo bar").has_value());
  EXPECT_TRUE(indexer->insert(2, "baz qux").has_value());

  std::vector<FtsResult> results;
  EXPECT_TRUE(search_ok(*indexer, "hello OR foo", 10, &results));
  ASSERT_EQ(results.size(), 2u);
}

// Regression: with an unknown WAND upper bound (side CFs dropped while postings
// are still Roaring — the dump-time transient), the per-term max_score must be
// +inf so the disjunction pivot never prunes the term. A 0 bound over-pruned:
// once the top-k threshold rose, higher-scoring docs were dropped.
TEST_F(FtsColumnIndexerTest, SearchRoaringDroppedSideCfsDoesNotOverPrune) {
  auto indexer = make_indexer();
  // doc0..2 match only "alpha"; doc3 also matches the rarer "beta", giving it
  // the strictly highest BM25 score.
  EXPECT_TRUE(indexer->insert(0, "alpha").has_value());
  EXPECT_TRUE(indexer->insert(1, "alpha").has_value());
  EXPECT_TRUE(indexer->insert(2, "alpha").has_value());
  EXPECT_TRUE(indexer->insert(3, "alpha beta").has_value());

  // Drop side CFs before converting postings to BitPacked: term iterators take
  // the Roaring path with an unknown WAND upper bound.
  indexer->reset_side_cfs();

  // topk=1 raises the threshold to score(doc0) after the first hit. A 0 bound
  // would then prune the rest and wrongly return doc0; doc3 must survive.
  std::vector<FtsResult> results;
  EXPECT_TRUE(search_ok(*indexer, "alpha OR beta", 1, &results));
  ASSERT_EQ(results.size(), 1u);
  EXPECT_EQ(results[0].doc_id, 3ull);
}

TEST_F(FtsColumnIndexerTest, SearchImplicitAdjacency) {
  auto indexer = make_indexer();
  EXPECT_TRUE(indexer->insert(0, "hello world").has_value());
  EXPECT_TRUE(indexer->insert(1, "foo bar").has_value());

  // Adjacent terms without operator -> OR semantics (default operator)
  std::vector<FtsResult> results;
  EXPECT_TRUE(search_ok(*indexer, "hello foo", 10, &results));
  EXPECT_EQ(results.size(), 2u);
}

// ============================================================
// search() - EmptyNode (matches zero docs)
// ============================================================

TEST_F(FtsColumnIndexerTest, SearchEmptyNodeReturnsNoResults) {
  auto indexer = make_indexer();
  EXPECT_TRUE(indexer->insert(0, "hello world").has_value());
  EXPECT_TRUE(indexer->insert(1, "foo bar").has_value());

  EmptyNode empty;
  zvec::fts::FtsQueryParams qp;
  qp.topk = 10;
  auto ret = indexer->search(empty, qp);
  ASSERT_TRUE(ret.has_value());
  EXPECT_TRUE(ret.value().empty());
}

TEST_F(FtsColumnIndexerTest, SearchAndWithEmptyChildReturnsNoResults) {
  auto indexer = make_indexer();
  EXPECT_TRUE(indexer->insert(0, "hello world").has_value());
  EXPECT_TRUE(indexer->insert(1, "hello foo").has_value());

  // AND with EmptyNode child → whole conjunction matches nothing.
  AndNode and_node;
  and_node.children.push_back(std::make_unique<EmptyNode>());
  and_node.children.push_back(std::make_unique<TermNode>("hello"));

  zvec::fts::FtsQueryParams qp;
  qp.topk = 10;
  auto ret = indexer->search(and_node, qp);
  ASSERT_TRUE(ret.has_value());
  EXPECT_TRUE(ret.value().empty());
}

TEST_F(FtsColumnIndexerTest, SearchOrWithEmptyChildIgnoresIt) {
  auto indexer = make_indexer();
  EXPECT_TRUE(indexer->insert(0, "hello world").has_value());
  EXPECT_TRUE(indexer->insert(1, "foo bar").has_value());

  // OR with EmptyNode child → empty is skipped, equivalent to OR(hello).
  OrNode or_node;
  or_node.children.push_back(std::make_unique<EmptyNode>());
  or_node.children.push_back(std::make_unique<TermNode>("hello"));

  zvec::fts::FtsQueryParams qp;
  qp.topk = 10;
  auto ret = indexer->search(or_node, qp);
  ASSERT_TRUE(ret.has_value());
  ASSERT_EQ(ret.value().size(), 1u);
  EXPECT_EQ(ret.value()[0].doc_id, 0ull);
}

// ============================================================
// search() - must_not modifier
// ============================================================

TEST_F(FtsColumnIndexerTest, SearchMustNotExcludesDoc) {
  auto indexer = make_indexer();
  EXPECT_TRUE(indexer->insert(0, "hello world").has_value());
  EXPECT_TRUE(indexer->insert(1, "hello foo").has_value());

  // "hello" matches both; "- world" (with space) excludes doc 0
  std::vector<FtsResult> results;
  EXPECT_TRUE(search_ok(*indexer, "hello - world", 10, &results));
  ASSERT_EQ(results.size(), 1u);
  EXPECT_EQ(results[0].doc_id, 1ull);
}

// `a NOT b` is the new binary AND-NOT operator (`a AND NOT b`).
TEST_F(FtsColumnIndexerTest, SearchBinaryNotExcludesDoc) {
  auto indexer = make_indexer();
  EXPECT_TRUE(indexer->insert(0, "hello world").has_value());
  EXPECT_TRUE(indexer->insert(1, "hello foo").has_value());

  std::vector<FtsResult> results;
  EXPECT_TRUE(search_ok(*indexer, "hello NOT world", 10, &results));
  ASSERT_EQ(results.size(), 1u);
  EXPECT_EQ(results[0].doc_id, 1ull);
}

// `a NOT (b OR c)` — must_not on a parenthesised OR sub-expression must
// exclude every doc matching either `b` or `c`.
TEST_F(FtsColumnIndexerTest, SearchMustNotOnGroupedOrExcludesDocs) {
  auto indexer = make_indexer();
  EXPECT_TRUE(
      indexer->insert(0, "hello world").has_value());  // excluded (has world)
  EXPECT_TRUE(
      indexer->insert(1, "hello foo").has_value());  // excluded (has foo)
  EXPECT_TRUE(indexer->insert(2, "hello bar").has_value());  // kept

  std::vector<FtsResult> results;
  EXPECT_TRUE(search_ok(*indexer, "hello NOT (world OR foo)", 10, &results));
  ASSERT_EQ(results.size(), 1u);
  EXPECT_EQ(results[0].doc_id, 2ull);
}

// Top-level `-(...)` produces a must_not root and must be rejected by
// search() (see fts_column_indexer.cc::search early-out).
TEST_F(FtsColumnIndexerTest, SearchTopLevelMustNotIsRejected) {
  auto indexer = make_indexer();
  EXPECT_TRUE(indexer->insert(0, "hello world").has_value());

  // -(hello AND world) => AndNode with must_not=true at the root
  FtsQueryParser parser;
  auto ast = parser.parse("-(hello AND world)", make_whitespace_pipeline());
  ASSERT_NE(ast, nullptr);
  EXPECT_TRUE(ast->must_not);

  std::vector<FtsResult> results;
  FtsQueryParams query_params;
  query_params.topk = 10;
  EXPECT_FALSE(indexer->search(*ast, query_params).has_value());
}

// ============================================================
// search() - must inside OR (should semantics)
// ============================================================

TEST_F(FtsColumnIndexerTest, SearchMustInOrOnlyReturnsDocsMatchingAllMust) {
  auto indexer = make_indexer();
  EXPECT_TRUE(indexer->insert(0, "foo bar baz bay").has_value());
  EXPECT_TRUE(indexer->insert(1, "bar bay").has_value());
  EXPECT_TRUE(indexer->insert(2, "foo baz").has_value());
  EXPECT_TRUE(indexer->insert(3, "foo bar").has_value());

  // "+bar +bay foo baz" — bar and bay must both match
  std::vector<FtsResult> results;
  EXPECT_TRUE(search_ok(*indexer, "+bar +bay foo baz", 10, &results));

  std::unordered_set<uint64_t> doc_ids;
  for (const auto &r : results) {
    doc_ids.insert(r.doc_id);
  }
  // doc 0 (foo bar baz bay) and doc 1 (bar bay) match
  EXPECT_TRUE(doc_ids.count(0));
  EXPECT_TRUE(doc_ids.count(1));
  // doc 2 (foo baz) and doc 3 (foo bar) should NOT match
  EXPECT_FALSE(doc_ids.count(2));
  EXPECT_FALSE(doc_ids.count(3));
}

TEST_F(FtsColumnIndexerTest, SearchMustInOrShouldBoostScore) {
  auto indexer = make_indexer();
  // Both docs contain the must terms (bar, bay)
  // Doc 0 also contains should terms (foo, baz) → should score higher
  EXPECT_TRUE(indexer->insert(0, "foo bar baz bay").has_value());
  EXPECT_TRUE(indexer->insert(1, "bar bay").has_value());

  std::vector<FtsResult> results;
  EXPECT_TRUE(search_ok(*indexer, "+bar +bay foo baz", 10, &results));
  ASSERT_EQ(results.size(), 2u);

  // Doc 0 should score higher than doc 1 due to should-term contributions
  EXPECT_EQ(results[0].doc_id, 0ull);
  EXPECT_GT(results[0].score, results[1].score);
}

TEST_F(FtsColumnIndexerTest, SearchSingleMustInOrFiltersCorrectly) {
  auto indexer = make_indexer();
  EXPECT_TRUE(indexer->insert(0, "hello world").has_value());
  EXPECT_TRUE(indexer->insert(1, "hello foo").has_value());
  EXPECT_TRUE(indexer->insert(2, "world foo").has_value());

  // "+hello world foo" — hello must match
  std::vector<FtsResult> results;
  EXPECT_TRUE(search_ok(*indexer, "+hello world foo", 10, &results));

  std::unordered_set<uint64_t> doc_ids;
  for (const auto &r : results) {
    doc_ids.insert(r.doc_id);
  }
  EXPECT_TRUE(doc_ids.count(0));
  EXPECT_TRUE(doc_ids.count(1));
  // doc 2 does not contain hello → excluded
  EXPECT_FALSE(doc_ids.count(2));
}

TEST_F(FtsColumnIndexerTest, SearchAllMustNoShouldWorksLikeAnd) {
  auto indexer = make_indexer();
  EXPECT_TRUE(indexer->insert(0, "hello world").has_value());
  EXPECT_TRUE(indexer->insert(1, "hello foo").has_value());

  // "+hello +world" — both must match, no should terms
  std::vector<FtsResult> results;
  EXPECT_TRUE(search_ok(*indexer, "+hello +world", 10, &results));
  ASSERT_EQ(results.size(), 1u);
  EXPECT_EQ(results[0].doc_id, 0ull);
}

TEST_F(FtsColumnIndexerTest, SearchWithoutMustUnchanged) {
  auto indexer = make_indexer();
  EXPECT_TRUE(indexer->insert(0, "hello world").has_value());
  EXPECT_TRUE(indexer->insert(1, "hello foo").has_value());
  EXPECT_TRUE(indexer->insert(2, "bar baz").has_value());

  // "hello world" (no must) — pure OR, matches any doc with hello or world
  std::vector<FtsResult> results;
  EXPECT_TRUE(search_ok(*indexer, "hello world", 10, &results));
  EXPECT_EQ(results.size(), 2u);
}

// ============================================================
// BM25 stats are updated in real-time after insert
// ============================================================

TEST_F(FtsColumnIndexerTest, BM25StatsUpdatedAfterInsert) {
  auto indexer = make_indexer();
  EXPECT_EQ(indexer->total_docs(), 0u);
  EXPECT_EQ(indexer->total_tokens(), 0u);

  EXPECT_TRUE(indexer->insert(0, "hello world foo").has_value());
  EXPECT_EQ(indexer->total_docs(), 1u);
  EXPECT_EQ(indexer->total_tokens(), 3u);

  EXPECT_TRUE(indexer->insert(1, "bar baz").has_value());
  EXPECT_EQ(indexer->total_docs(), 2u);
  EXPECT_EQ(indexer->total_tokens(), 5u);
}

TEST_F(FtsColumnIndexerTest, SearchScorePositiveAfterInsert) {
  auto indexer = make_indexer();
  EXPECT_TRUE(indexer->insert(0, "hello world").has_value());

  std::vector<FtsResult> results;
  EXPECT_TRUE(search_ok(*indexer, "hello", 10, &results));
  ASSERT_EQ(results.size(), 1u);
  EXPECT_GT(results[0].score, 0.0f);
}

// ============================================================
// End-to-end: multiple inserts and searches
// ============================================================

TEST_F(FtsColumnIndexerTest, MultipleInsertsAndSearches) {
  auto indexer = make_indexer("content");

  const std::vector<std::string> docs = {
      "the quick brown fox",
      "the lazy dog",
      "quick brown dog",
      "fox and dog",
  };

  for (uint64_t doc_id = 0; doc_id < docs.size(); ++doc_id) {
    EXPECT_TRUE(indexer->insert(doc_id, docs[doc_id]).has_value());
  }

  EXPECT_EQ(indexer->total_docs(), docs.size());

  // "quick" appears in doc 0 and doc 2
  std::vector<FtsResult> results;
  EXPECT_TRUE(search_ok(*indexer, "quick", 10, &results));
  EXPECT_EQ(results.size(), 2u);

  // "the" appears in doc 0 and doc 1
  results.clear();
  EXPECT_TRUE(search_ok(*indexer, "the", 10, &results));
  EXPECT_EQ(results.size(), 2u);

  // "quick AND dog" -> only doc 2
  results.clear();
  EXPECT_TRUE(search_ok(*indexer, "quick AND dog", 10, &results));
  ASSERT_EQ(results.size(), 1u);
  EXPECT_EQ(results[0].doc_id, 2ull);
}

// ============================================================
// Jieba Chinese tokenizer tests
// ============================================================

// JIEBA_DICT_DIR points to thirdparty/cppjieba/.../dict/ (injected by CMake).
#ifndef JIEBA_DICT_DIR
#define JIEBA_DICT_DIR "."
#endif

static const std::string kJiebaDictDir{JIEBA_DICT_DIR};

static bool jieba_dict_available() {
  std::string path = kJiebaDictDir + "/jieba.dict.utf8";
  std::ifstream ifs(path);
  return ifs.good();
}

static std::string make_jieba_extra_params() {
  return std::string(R"({"jieba_dict_dir":")") + kJiebaDictDir + R"("})";
}

class FtsColumnIndexerJiebaTest : public FtsColumnIndexerTest {
 protected:
  void SetUp() override {
    if (!jieba_dict_available()) {
      GTEST_SKIP() << "Jieba dict not available at: " << kJiebaDictDir;
    }
    FtsColumnIndexerTest::SetUp();
  }
  // Create and open a fresh indexer with jieba tokenizer.
  std::unique_ptr<FtsColumnIndexer> make_jieba_indexer(
      const std::string &field_name = "content") {
    auto fts_params = std::make_shared<zvec::FtsIndexParams>(
        "jieba", std::vector<std::string>{"lowercase"},
        make_jieba_extra_params());
    auto field_meta = make_test_field_meta(field_name, fts_params);
    auto indexer = std::make_unique<FtsColumnIndexer>();
    auto ret = indexer->open(field_meta, &db_, postings_cf_, positions_cf_,
                             term_freq_cf_, max_tf_cf_, doc_len_cf_, stat_cf_);
    EXPECT_TRUE(ret.has_value());
    return indexer;
  }
};

// Verify that jieba tokenizer opens successfully with valid dict paths.
TEST_F(FtsColumnIndexerJiebaTest, OpenWithJiebaTokenizerSucceeds) {
  auto fts_params = std::make_shared<zvec::FtsIndexParams>(
      "jieba", std::vector<std::string>{"lowercase"},
      make_jieba_extra_params());
  auto field_meta = make_test_field_meta("content", fts_params);
  FtsColumnIndexer indexer;
  auto ret = indexer.open(field_meta, &db_, postings_cf_, positions_cf_,
                          term_freq_cf_, max_tf_cf_, doc_len_cf_, stat_cf_);
  EXPECT_TRUE(ret.has_value());
}

// Verify that jieba tokenizer fails when no jieba_dict_dir source resolves.
// (cppjieba FATAL-aborts on non-existent dict files, so we test the init-time
// validation in JiebaTokenizer instead.)
// Assumes the ZVEC_JIEBA_DICT_DIR env-var is not set in the test environment.
TEST_F(FtsColumnIndexerJiebaTest, OpenWithJiebaTokenizerFailsWithoutDictDir) {
  zvec::GlobalConfig::Instance().set_default_jieba_dict_dir("");

  fts::FtsIndexParams bad_params;
  bad_params.tokenizer_name = "jieba";
  bad_params.extra_params = "";
  auto pipeline = TokenizerFactory::create(bad_params);
  EXPECT_EQ(pipeline, nullptr);
}

// Insert a Chinese sentence and verify that total_docs and total_tokens are
// updated correctly (jieba should produce at least one token).
TEST_F(FtsColumnIndexerJiebaTest, InsertChineseTextUpdatesStats) {
  auto indexer = make_jieba_indexer();

  // "中文分词测试" should be segmented into multiple tokens by jieba.
  EXPECT_TRUE(indexer->insert(0, "中文分词测试").has_value());
  EXPECT_EQ(indexer->total_docs(), 1u);
  EXPECT_GT(indexer->total_tokens(), 0u);
}

// Insert multiple Chinese documents and verify that a segmented term can be
// found via search(). The dedicated FtsLexer supports UNICODE_TERM so Chinese
// words can be used as bare terms without quoting.
TEST_F(FtsColumnIndexerJiebaTest, SearchChineseTermFound) {
  auto indexer = make_jieba_indexer();

  // doc 0: contains "中文" and "分词"
  EXPECT_TRUE(indexer->insert(0, "中文分词技术").has_value());
  // doc 1: contains "搜索" and "引擎"
  EXPECT_TRUE(indexer->insert(1, "搜索引擎优化").has_value());
  // doc 2: contains "中文" again
  EXPECT_TRUE(indexer->insert(2, "中文搜索").has_value());

  // jieba CutForSearch segments "中文分词技术" → [中文, 分词, 技术, ...] and
  //                             "中文搜索"     → [中文, 搜索], so doc 0 and
  //                             doc 2 should match "中文".
  std::vector<FtsResult> results;
  EXPECT_TRUE(search_ok(*indexer, "中文", 10, &results));
  EXPECT_GE(results.size(), 1u);

  bool found_doc0 = false;
  bool found_doc2 = false;
  for (const auto &result : results) {
    if (result.doc_id == 0) found_doc0 = true;
    if (result.doc_id == 2) found_doc2 = true;
  }
  EXPECT_TRUE(found_doc0);
  EXPECT_TRUE(found_doc2);
}

// Verify that a term not present in any document returns empty results.
TEST_F(FtsColumnIndexerJiebaTest, SearchChineseTermNotFound) {
  auto indexer = make_jieba_indexer();

  EXPECT_TRUE(indexer->insert(0, "中文分词技术").has_value());

  std::vector<FtsResult> results;
  EXPECT_TRUE(search_ok(*indexer, "日语", 10, &results));
  EXPECT_EQ(results.size(), 0u);
}

// Verify BM25 scores are positive after inserting Chinese documents.
TEST_F(FtsColumnIndexerJiebaTest, SearchChineseTermHasPositiveScore) {
  auto indexer = make_jieba_indexer();

  EXPECT_TRUE(indexer->insert(0, "自然语言处理技术").has_value());
  EXPECT_TRUE(indexer->insert(1, "机器学习算法").has_value());

  // Search for a token that jieba should produce from "自然语言处理技术".
  std::vector<FtsResult> results;
  EXPECT_TRUE(search_ok(*indexer, "自然语言", 10, &results));
  if (!results.empty()) {
    EXPECT_GT(results[0].score, 0.0f);
  }
}

// Verify that topk limits the number of results for Chinese queries.
TEST_F(FtsColumnIndexerJiebaTest, SearchChineseTermTopkLimitsResults) {
  auto indexer = make_jieba_indexer();

  // Insert 5 documents all containing "技术"
  for (uint64_t doc_id = 0; doc_id < 5; ++doc_id) {
    EXPECT_TRUE(indexer->insert(doc_id, "人工智能技术发展").has_value());
  }

  std::vector<FtsResult> results;
  EXPECT_TRUE(search_ok(*indexer, "技术", /*topk=*/3, &results));
  EXPECT_LE(results.size(), 3u);
}

// End-to-end: flush and reload with jieba tokenizer.
TEST_F(FtsColumnIndexerJiebaTest, FlushAndReloadWithJiebaTokenizer) {
  auto indexer = make_jieba_indexer("content");

  EXPECT_TRUE(indexer->insert(0, "深度学习模型").has_value());
  EXPECT_TRUE(indexer->insert(1, "神经网络结构").has_value());
  EXPECT_TRUE(indexer->flush().has_value());

  // Reload via a standalone reader (no tokenizer needed for reading).
  // Pass doc_len_cf as nullptr so the reader loads stats from stat_cf.
  FtsColumnIndexer reader;
  auto ret = reader.open_reader("content", &db_, postings_cf_, positions_cf_,
                                term_freq_cf_, max_tf_cf_,
                                /*doc_len_cf=*/nullptr, stat_cf_);
  EXPECT_TRUE(ret.has_value());

  // Search with a term that jieba produces from "深度学习模型":
  // jieba CutForSearch segments it into [深度, 学习, 深度学习, 模型].
  TermNode term_node("模型");
  FtsQueryParams query_params;
  query_params.topk = 10;
  auto search_ret = reader.search(term_node, query_params);
  EXPECT_TRUE(search_ret.has_value());
  EXPECT_GE(search_ret.value().size(), 1u);
}

// Construct a jieba pipeline matching the indexer config so phrase queries
// tokenize the same way the index did.
static zvec::fts::TokenizerPipelinePtr make_jieba_pipeline_for_test() {
  zvec::fts::FtsIndexParams params;
  params.tokenizer_name = "jieba";
  params.filters = {"lowercase"};
  params.extra_params =
      std::string(R"({"jieba_dict_dir":")") + kJiebaDictDir + R"("})";
  return zvec::fts::TokenizerFactory::create(params);
}

// Phrase queries on a jieba-indexed doc must hit when the query goes through
// the same pipeline as the document. Before the parser was pipeline-aware
// the query path split the phrase on ASCII whitespace, so a CJK phrase
// became a single opaque token and failed to match the per-segment tokens
// the index actually stored.
TEST_F(FtsColumnIndexerJiebaTest, PhraseSearchHitsAfterJiebaTokenization) {
  auto indexer = make_jieba_indexer();
  EXPECT_TRUE(indexer->insert(0, "中华人民共和国成立").has_value());
  EXPECT_TRUE(indexer->insert(1, "无关文档").has_value());
  EXPECT_TRUE(indexer->flush().has_value());

  auto pipeline = make_jieba_pipeline_for_test();
  ASSERT_NE(pipeline, nullptr);

  // Phrase covering the full doc text — query and doc tokenize identically
  // so the strict anchor+i adjacency check in PhraseDocIterator succeeds.
  std::vector<FtsResult> results;
  EXPECT_TRUE(
      search_ok(*indexer, "\"中华人民共和国成立\"", 10, &results, pipeline));
  ASSERT_EQ(results.size(), 1u);
  EXPECT_EQ(results[0].doc_id, 0ull);

  // A single-token phrase still works after the position-as-sequence fix:
  // jieba emits "成立" once with a deterministic sequence position, the
  // single-term phrase trivially matches.
  results.clear();
  EXPECT_TRUE(search_ok(*indexer, "\"成立\"", 10, &results, pipeline));
  ASSERT_EQ(results.size(), 1u);
  EXPECT_EQ(results[0].doc_id, 0ull);
}

// JiebaTokenizer.position must be a strictly increasing per-output-token
// sequence number. CutForSearch emits overlapping sub-words for a long
// parent word; using cppjieba's unicode_offset would assign duplicate or
// non-monotonic positions and break PhraseDocIterator's strict adjacency
// check. Sequence numbers are guaranteed contiguous across all emitted
// tokens.
TEST(JiebaTokenizerTest, PositionIsContiguousSequence) {
  if (!jieba_dict_available()) {
    GTEST_SKIP() << "Jieba dict not available at: " << kJiebaDictDir;
  }
  auto pipeline = make_jieba_pipeline_for_test();
  ASSERT_NE(pipeline, nullptr);

  // CutForSearch on this string emits the long parent word followed by its
  // shorter sub-words; the sub-words share a unicode_offset with the parent
  // but get distinct sequence numbers under the new scheme.
  auto tokens = pipeline->process("中华人民共和国");
  ASSERT_FALSE(tokens.empty());
  for (size_t i = 0; i < tokens.size(); ++i) {
    EXPECT_EQ(tokens[i].position, static_cast<uint32_t>(i))
        << "tokens[" << i << "].text=" << tokens[i].text;
  }
}

// ============================================================
// jieba_dict_dir resolution priority chain
// ============================================================
//
// JiebaTokenizer::init resolves jieba_dict_dir in this order:
//   1. extra_params.jieba_dict_dir (per-field)
//   2. ZVEC_JIEBA_DICT_DIR env var
//   3. zvec::GlobalConfig::jieba_dict_dir() (set by SDK or zvec.init)
//
// The cases below assume the ZVEC_JIEBA_DICT_DIR env var is not set in the
// test environment, so they only exercise tiers (1) and (3).

class JiebaDictDirPriorityTest : public FtsColumnIndexerJiebaTest {
 protected:
  void SetUp() override {
    FtsColumnIndexerJiebaTest::SetUp();
    if (IsSkipped()) {
      return;
    }
    saved_global_ = zvec::GlobalConfig::Instance().jieba_dict_dir();
    zvec::GlobalConfig::Instance().set_default_jieba_dict_dir("");
  }

  void TearDown() override {
    zvec::GlobalConfig::Instance().set_default_jieba_dict_dir(saved_global_);
    FtsColumnIndexerJiebaTest::TearDown();
  }

  // Build an indexer with arbitrary extra_params (so individual cases can
  // toggle whether jieba_dict_dir is in the per-field config).
  std::unique_ptr<FtsColumnIndexer> make_indexer_with_extra_params(
      const std::string &extra_params,
      const std::string &field_name = "content") {
    auto fts_params = std::make_shared<zvec::FtsIndexParams>(
        "jieba", std::vector<std::string>{"lowercase"}, extra_params);
    auto field_meta = make_test_field_meta(field_name, fts_params);
    auto indexer = std::make_unique<FtsColumnIndexer>();
    auto ret = indexer->open(field_meta, &db_, postings_cf_, positions_cf_,
                             term_freq_cf_, max_tf_cf_, doc_len_cf_, stat_cf_);
    EXPECT_TRUE(ret.has_value());
    return indexer;
  }

 private:
  std::string saved_global_;
};

// Core scenario: SDK in module-load called set_default_jieba_dict_dir; user
// never called zvec_initialize; per-field extra_params is empty. Jieba must
// still work end-to-end. Validates that SDK auto-registration is decoupled
// from the GlobalConfig::Initialize one-shot lifecycle.
TEST_F(JiebaDictDirPriorityTest, GlobalConfigDefaultUsedWithoutInitialize) {
  zvec::GlobalConfig::Instance().set_default_jieba_dict_dir(kJiebaDictDir);

  auto indexer = make_indexer_with_extra_params("");
  EXPECT_TRUE(indexer->insert(0, "中华人民共和国").has_value());
  EXPECT_TRUE(indexer->flush().has_value());

  std::vector<FtsResult> results;
  auto pipeline = make_jieba_pipeline_for_test();
  EXPECT_TRUE(search_ok(*indexer, "中华", 10, &results, pipeline));
  EXPECT_GE(results.size(), 1u);
}

// per-field extra_params.jieba_dict_dir must beat GlobalConfig even when
// GlobalConfig is bogus.
TEST_F(JiebaDictDirPriorityTest, PerFieldBeatsGlobalConfig) {
  zvec::GlobalConfig::Instance().set_default_jieba_dict_dir(
      "/zvec/intentionally/missing/global");

  auto extra = std::string(R"({"jieba_dict_dir":")") + kJiebaDictDir + R"("})";
  auto indexer = make_indexer_with_extra_params(extra);
  EXPECT_TRUE(indexer->insert(0, "自然语言处理").has_value());
  EXPECT_TRUE(indexer->flush().has_value());

  std::vector<FtsResult> results;
  auto pipeline = make_jieba_pipeline_for_test();
  EXPECT_TRUE(search_ok(*indexer, "自然", 10, &results, pipeline));
  EXPECT_GE(results.size(), 1u);
}

// ============================================================
// convert_postings_to_bitpacked()
// ============================================================
//
// These tests exercise the BitPacked conversion path that is invoked from
// MutableSegment::dump_fts_column_indexers() right before the SST dump.
// They use the BitPackedPostingList::is_bitpacked_format magic-number probe
// to verify that postings have been re-encoded, and iterate $TF / $DOC_LEN
// CFs to verify the DeleteRange tombstones effectively removed all entries.

#include "db/index/column/fts_column/posting/bitpacked_posting_list.h"  // NOLINT: in-test include

namespace {

// Count entries in a CF by iterating from the first key.  Used to verify that
// $TF / $DOC_LEN have been DeleteRange-cleared.
size_t count_cf_entries(RocksdbContext &db, rocksdb::ColumnFamilyHandle *cf) {
  size_t count = 0;
  std::unique_ptr<rocksdb::Iterator> iter(
      db.db_->NewIterator(db.read_opts_, cf));
  for (iter->SeekToFirst(); iter->Valid(); iter->Next()) {
    ++count;
  }
  return count;
}

// Verify every value in postings_cf_ is in BitPacked format.
size_t count_postings_entries_and_check_bitpacked(
    RocksdbContext &db, rocksdb::ColumnFamilyHandle *cf) {
  size_t count = 0;
  std::unique_ptr<rocksdb::Iterator> iter(
      db.db_->NewIterator(db.read_opts_, cf));
  for (iter->SeekToFirst(); iter->Valid(); iter->Next()) {
    const std::string value = iter->value().ToString();
    EXPECT_TRUE(
        BitPackedPostingList::is_bitpacked_format(value.data(), value.size()))
        << "Posting for term[" << iter->key().ToString()
        << "] is not BitPacked";
    ++count;
  }
  return count;
}

}  // namespace

// Insert N docs, run the conversion, and verify:
//   - postings_cf_ values all carry the BitPacked magic
//   - decoded posting iterators yield the original (doc_id, tf, doc_len)
//   - $TF / $DOC_LEN CFs are empty
TEST_F(FtsColumnIndexerTest, ConvertPostingsToBitpackedBasic) {
  auto indexer = make_indexer("content");
  EXPECT_TRUE(indexer->insert(0, "hello world").has_value());
  EXPECT_TRUE(indexer->insert(1, "hello foo bar").has_value());
  EXPECT_TRUE(indexer->insert(2, "hello hello world").has_value());
  EXPECT_TRUE(indexer->flush().has_value());

  EXPECT_TRUE(indexer->convert_postings_to_bitpacked().has_value());

  // All postings must now be BitPacked.
  size_t postings_count =
      count_postings_entries_and_check_bitpacked(db_, postings_cf_);
  EXPECT_GT(postings_count, 0u);

  // Spot-check: decode the "hello" posting and confirm doc_ids/tfs/doc_lens
  // match what we wrote.  Doc 0 -> tf=1, dl=2; Doc 1 -> tf=1, dl=3; Doc 2 ->
  // tf=2, dl=3.
  std::string raw;
  ASSERT_TRUE(db_.db_->Get(db_.read_opts_, postings_cf_, "hello", &raw).ok());
  ASSERT_FALSE(raw.empty());
  ASSERT_TRUE(
      BitPackedPostingList::is_bitpacked_format(raw.data(), raw.size()));

  BitPackedPostingIterator iter;
  ASSERT_EQ(iter.open(raw.data(), raw.size()), 0);

  std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> decoded;
  while (true) {
    uint32_t did = iter.next_doc();
    if (did == BitPackedPostingIterator::NO_MORE_DOCS) break;
    decoded.emplace_back(did, iter.term_freq(), iter.doc_len());
  }
  ASSERT_EQ(decoded.size(), 3u);
  EXPECT_EQ(std::get<0>(decoded[0]), 0u);
  EXPECT_EQ(std::get<1>(decoded[0]), 1u);
  EXPECT_EQ(std::get<2>(decoded[0]), 2u);
  EXPECT_EQ(std::get<0>(decoded[1]), 1u);
  EXPECT_EQ(std::get<1>(decoded[1]), 1u);
  EXPECT_EQ(std::get<2>(decoded[1]), 3u);
  EXPECT_EQ(std::get<0>(decoded[2]), 2u);
  EXPECT_EQ(std::get<1>(decoded[2]), 2u);
  EXPECT_EQ(std::get<2>(decoded[2]), 3u);
}

// After conversion the $TF / $DOC_LEN / $MAX_TF side CFs must be EMPTY: the
// indexer DeleteRange's them once their content has been inlined into the
// BitPacked posting list.  MutableSegment then drops the CFs entirely.
TEST_F(FtsColumnIndexerTest, ConvertPostingsToBitpackedClearsSideCfs) {
  auto indexer = make_indexer("content");
  for (uint64_t doc_id = 0; doc_id < 5; ++doc_id) {
    EXPECT_TRUE(indexer->insert(doc_id, "alpha beta gamma").has_value());
  }
  EXPECT_TRUE(indexer->flush().has_value());

  // Sanity: side CFs are populated before conversion.
  EXPECT_GT(count_cf_entries(db_, term_freq_cf_), 0u);
  EXPECT_GT(count_cf_entries(db_, doc_len_cf_), 0u);
  EXPECT_GT(count_cf_entries(db_, max_tf_cf_), 0u);

  EXPECT_TRUE(indexer->convert_postings_to_bitpacked().has_value());

  // Side CFs must be empty after conversion (DeleteRange'd by the indexer).
  EXPECT_EQ(count_cf_entries(db_, term_freq_cf_), 0u);
  EXPECT_EQ(count_cf_entries(db_, doc_len_cf_), 0u);
  EXPECT_EQ(count_cf_entries(db_, max_tf_cf_), 0u);

  // After reset_side_cfs, search should still work (BitPacked path).
  indexer->reset_side_cfs();
  std::vector<FtsResult> results;
  EXPECT_TRUE(search_ok(*indexer, "alpha", 10, &results));
  EXPECT_EQ(results.size(), 5u);
}

// Conversion must be idempotent: calling it twice should not corrupt postings,
// nor should it re-encode terms that are already BitPacked.
TEST_F(FtsColumnIndexerTest, ConvertPostingsToBitpackedIsIdempotent) {
  auto indexer = make_indexer("content");
  EXPECT_TRUE(indexer->insert(0, "hello world").has_value());
  EXPECT_TRUE(indexer->insert(1, "hello foo").has_value());
  EXPECT_TRUE(indexer->flush().has_value());

  EXPECT_TRUE(indexer->convert_postings_to_bitpacked().has_value());

  // Snapshot the BitPacked posting for "hello" after the first conversion.
  std::string snapshot;
  ASSERT_TRUE(
      db_.db_->Get(db_.read_opts_, postings_cf_, "hello", &snapshot).ok());
  ASSERT_FALSE(snapshot.empty());

  // Second invocation must succeed and leave the posting byte-for-byte
  // identical (the idempotency guard skips re-encoding).
  EXPECT_TRUE(indexer->convert_postings_to_bitpacked().has_value());

  std::string after;
  ASSERT_TRUE(db_.db_->Get(db_.read_opts_, postings_cf_, "hello", &after).ok());
  EXPECT_EQ(snapshot, after);
}

// An indexer with no inserted documents must still allow the conversion to
// succeed (no-op path) — this matches MutableSegment dump-flow expectations
// for FTS fields that received zero writes.
TEST_F(FtsColumnIndexerTest, ConvertPostingsToBitpackedEmptyIndexer) {
  auto indexer = make_indexer("content");
  EXPECT_TRUE(indexer->flush().has_value());
  EXPECT_TRUE(indexer->convert_postings_to_bitpacked().has_value());
  EXPECT_EQ(count_postings_entries_and_check_bitpacked(db_, postings_cf_), 0u);
  // Side CFs were never populated (empty indexer); no special expectation
  // about them here beyond "the conversion did not crash".
}

// After conversion the search() path must keep working — readers fall through
// to the BitPacked branch via is_bitpacked_format(), and no longer require the
// $TF / $DOC_LEN CFs.
TEST_F(FtsColumnIndexerTest, SearchAfterConvertPostingsToBitpacked) {
  auto indexer = make_indexer("content");
  EXPECT_TRUE(indexer->insert(0, "the quick brown fox").has_value());
  EXPECT_TRUE(indexer->insert(1, "the lazy dog").has_value());
  EXPECT_TRUE(indexer->insert(2, "quick brown dog").has_value());
  EXPECT_TRUE(indexer->flush().has_value());

  // Pre-conversion baseline: "quick" hits doc 0 and doc 2.
  std::vector<FtsResult> baseline;
  EXPECT_TRUE(search_ok(*indexer, "quick", 10, &baseline));
  ASSERT_EQ(baseline.size(), 2u);

  EXPECT_TRUE(indexer->convert_postings_to_bitpacked().has_value());

  // Post-conversion via a standalone reader (mirrors immutable segment use).
  // Side CFs are passed as nullptr — immutable segments no longer register
  // them.
  FtsColumnIndexer reader;
  ASSERT_TRUE(reader
                  .open_reader("content", &db_, postings_cf_, positions_cf_,
                               /*term_freq_cf=*/nullptr, /*max_tf_cf=*/nullptr,
                               /*doc_len_cf=*/nullptr, stat_cf_)
                  .has_value());
  std::vector<FtsResult> results;
  EXPECT_TRUE(search_ok(reader, "quick", 10, &results));
  ASSERT_EQ(results.size(), 2u);

  // Same set of doc_ids as the baseline; scores may differ slightly because
  // the reader loaded stats fresh from stat_cf, but both must be positive.
  std::vector<uint64_t> ids;
  for (const auto &r : results) {
    ids.push_back(r.doc_id);
    EXPECT_GT(r.score, 0.0f);
  }
  std::sort(ids.begin(), ids.end());
  EXPECT_EQ(ids[0], 0ull);
  EXPECT_EQ(ids[1], 2ull);
}

// ============================================================
// Multi-column shared RocksDB tests
//
// Mirrors the CF-naming scheme used by SegmentImpl::open_fts_indexers():
//   field_name           -> postings CF
//   field_name_positions -> positions CF
//   field_name_tf        -> term-freq CF
//   field_name_max_tf    -> max-tf CF
//   field_name_doc_len   -> doc-len CF
//   fts_stat             -> shared stat CF
// ============================================================

static const std::string kMultiDbPath{"./test_fts_multi_db"};

class FtsMultiColumnSharedDbTest : public ::testing::Test {
 protected:
  // Two FTS fields sharing the same RocksDB instance.
  static constexpr const char *kFields[] = {"title", "body"};
  static constexpr size_t kNumFields = 2;

  void SetUp() override {
    zvec::FileHelper::RemoveDirectory(kMultiDbPath);

    // Build CF names and per-CF merge operators following the segment pattern.
    std::vector<std::string> cf_names;
    std::unordered_map<std::string, std::shared_ptr<rocksdb::MergeOperator>>
        per_cf_ops;

    for (size_t i = 0; i < kNumFields; ++i) {
      std::string f{kFields[i]};
      cf_names.push_back(f);                        // postings
      cf_names.push_back(f + kFtsPositionsSuffix);  // positions
      cf_names.push_back(f + kFtsTfSuffix);         // term freq
      cf_names.push_back(f + kFtsMaxTfSuffix);      // max tf
      cf_names.push_back(f + kFtsDocLenSuffix);     // doc len

      per_cf_ops[f] = std::make_shared<FtsPostingsMerge>();
      per_cf_ops[f + kFtsMaxTfSuffix] = std::make_shared<FtsMaxTfMerge>();
    }
    cf_names.push_back(zvec::kFtsStatCfName);

    ASSERT_TRUE(db_.create(RocksdbContext::Args{kMultiDbPath, cf_names, nullptr,
                                                per_cf_ops})
                    .ok());

    // Resolve CF handles per field.
    for (size_t i = 0; i < kNumFields; ++i) {
      std::string f{kFields[i]};
      postings_cf_[i] = db_.get_cf(f);
      positions_cf_[i] = db_.get_cf(f + kFtsPositionsSuffix);
      term_freq_cf_[i] = db_.get_cf(f + kFtsTfSuffix);
      max_tf_cf_[i] = db_.get_cf(f + kFtsMaxTfSuffix);
      doc_len_cf_[i] = db_.get_cf(f + kFtsDocLenSuffix);
      ASSERT_NE(postings_cf_[i], nullptr) << "field=" << f;
      ASSERT_NE(positions_cf_[i], nullptr) << "field=" << f;
      ASSERT_NE(term_freq_cf_[i], nullptr) << "field=" << f;
      ASSERT_NE(max_tf_cf_[i], nullptr) << "field=" << f;
      ASSERT_NE(doc_len_cf_[i], nullptr) << "field=" << f;
    }
    stat_cf_ = db_.get_cf(zvec::kFtsStatCfName);
    ASSERT_NE(stat_cf_, nullptr);
  }

  void TearDown() override {
    db_.close();
    zvec::FileHelper::RemoveDirectory(kMultiDbPath);
  }

  // Return the array index for a field name (0 = title, 1 = body).
  size_t field_index(const std::string &field_name) const {
    for (size_t i = 0; i < kNumFields; ++i) {
      if (field_name == kFields[i]) return i;
    }
    ADD_FAILURE() << "Unknown field: " << field_name;
    return 0;
  }

  // Create and open a FtsColumnIndexer bound to the CFs of the given field.
  std::unique_ptr<FtsColumnIndexer> make_indexer(
      const std::string &field_name) {
    size_t idx = field_index(field_name);
    auto fts_params = std::make_shared<zvec::FtsIndexParams>("whitespace");
    auto field_meta = make_test_field_meta(field_name, fts_params);
    auto indexer = std::make_unique<FtsColumnIndexer>();
    auto ret = indexer->open(field_meta, &db_, postings_cf_[idx],
                             positions_cf_[idx], term_freq_cf_[idx],
                             max_tf_cf_[idx], doc_len_cf_[idx], stat_cf_);
    EXPECT_TRUE(ret.has_value());
    return indexer;
  }

  RocksdbContext db_;
  rocksdb::ColumnFamilyHandle *postings_cf_[kNumFields]{};
  rocksdb::ColumnFamilyHandle *positions_cf_[kNumFields]{};
  rocksdb::ColumnFamilyHandle *term_freq_cf_[kNumFields]{};
  rocksdb::ColumnFamilyHandle *max_tf_cf_[kNumFields]{};
  rocksdb::ColumnFamilyHandle *doc_len_cf_[kNumFields]{};
  rocksdb::ColumnFamilyHandle *stat_cf_{nullptr};
};

// Two FTS columns write different documents; search on each column only
// returns hits from that column's data.
TEST_F(FtsMultiColumnSharedDbTest, MultiColumnInsertAndSearchIsolation) {
  auto title_indexer = make_indexer("title");
  auto body_indexer = make_indexer("body");

  // title column: documents about animals
  EXPECT_TRUE(title_indexer->insert(0, "quick brown fox").has_value());
  EXPECT_TRUE(title_indexer->insert(1, "lazy dog").has_value());

  // body column: documents about programming
  EXPECT_TRUE(body_indexer->insert(0, "hello world program").has_value());
  EXPECT_TRUE(body_indexer->insert(1, "quick sort algorithm").has_value());

  // Search "quick" in title -> only doc 0
  {
    std::vector<FtsResult> results;
    EXPECT_TRUE(search_ok(*title_indexer, "quick", 10, &results));
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].doc_id, 0ull);
  }

  // Search "quick" in body -> only doc 1
  {
    std::vector<FtsResult> results;
    EXPECT_TRUE(search_ok(*body_indexer, "quick", 10, &results));
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].doc_id, 1ull);
  }

  // Search "hello" in title -> no results
  {
    std::vector<FtsResult> results;
    EXPECT_TRUE(search_ok(*title_indexer, "hello", 10, &results));
    EXPECT_TRUE(results.empty());
  }

  // Search "fox" in body -> no results
  {
    std::vector<FtsResult> results;
    EXPECT_TRUE(search_ok(*body_indexer, "fox", 10, &results));
    EXPECT_TRUE(results.empty());
  }
}

// Flush both columns, then open read-only readers and verify each column's
// search results survive the reload.
TEST_F(FtsMultiColumnSharedDbTest, MultiColumnFlushAndReload) {
  auto title_indexer = make_indexer("title");
  auto body_indexer = make_indexer("body");

  EXPECT_TRUE(title_indexer->insert(0, "alpha beta gamma").has_value());
  EXPECT_TRUE(body_indexer->insert(0, "delta epsilon").has_value());
  EXPECT_TRUE(body_indexer->insert(1, "alpha zeta").has_value());

  EXPECT_TRUE(title_indexer->flush().has_value());
  EXPECT_TRUE(body_indexer->flush().has_value());

  // Open standalone readers (pass doc_len_cf as nullptr to exercise the
  // stat-CF reload path, matching immutable segment behaviour).
  size_t ti = field_index("title");
  size_t bi = field_index("body");

  FtsColumnIndexer title_reader;
  ASSERT_TRUE(title_reader
                  .open_reader("title", &db_, postings_cf_[ti],
                               positions_cf_[ti], term_freq_cf_[ti],
                               max_tf_cf_[ti], /*doc_len_cf=*/nullptr, stat_cf_)
                  .has_value());

  FtsColumnIndexer body_reader;
  ASSERT_TRUE(body_reader
                  .open_reader("body", &db_, postings_cf_[bi],
                               positions_cf_[bi], term_freq_cf_[bi],
                               max_tf_cf_[bi], /*doc_len_cf=*/nullptr, stat_cf_)
                  .has_value());

  // title reader: "alpha" -> doc 0 only
  {
    std::vector<FtsResult> results;
    EXPECT_TRUE(search_ok(title_reader, "alpha", 10, &results));
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].doc_id, 0ull);
  }

  // body reader: "alpha" -> doc 1 only
  {
    std::vector<FtsResult> results;
    EXPECT_TRUE(search_ok(body_reader, "alpha", 10, &results));
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].doc_id, 1ull);
  }

  // body reader: "delta" -> doc 0 only
  {
    std::vector<FtsResult> results;
    EXPECT_TRUE(search_ok(body_reader, "delta", 10, &results));
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].doc_id, 0ull);
  }
}

// Each column maintains independent total_docs and total_tokens counters.
TEST_F(FtsMultiColumnSharedDbTest, MultiColumnStatsIndependent) {
  auto title_indexer = make_indexer("title");
  auto body_indexer = make_indexer("body");

  // title: 2 docs, 4 tokens
  EXPECT_TRUE(title_indexer->insert(0, "hello world").has_value());
  EXPECT_TRUE(title_indexer->insert(1, "foo bar").has_value());
  EXPECT_EQ(title_indexer->total_docs(), 2u);
  EXPECT_EQ(title_indexer->total_tokens(), 4u);

  // body: 1 doc, 3 tokens
  EXPECT_TRUE(body_indexer->insert(0, "alpha beta gamma").has_value());
  EXPECT_EQ(body_indexer->total_docs(), 1u);
  EXPECT_EQ(body_indexer->total_tokens(), 3u);

  // Inserting into body must not affect title's counters.
  EXPECT_EQ(title_indexer->total_docs(), 2u);
  EXPECT_EQ(title_indexer->total_tokens(), 4u);
}

// ============================================================
// Filter pushdown into FTS iterators (single-term / OR / Phrase)
// ============================================================

namespace {

// Build an IndexFilter that excludes any doc_id present in `blocked`.
zvec::IndexFilter::Ptr make_blocked_filter(
    std::initializer_list<uint64_t> blocked) {
  std::unordered_set<uint64_t> set(blocked);
  return zvec::EasyIndexFilter::Create(
      [set = std::move(set)](uint64_t id) { return set.count(id) > 0; });
}

}  // namespace

// Single-term query path: TermDocIterator inherits the base-class default
// next_doc(filter), which loops over next_doc() and skips filtered docs.
TEST_F(FtsColumnIndexerTest, FilterPushdownExcludesFilteredDocs) {
  auto indexer = make_indexer("content");
  EXPECT_TRUE(indexer->insert(0, "hello world").has_value());
  EXPECT_TRUE(indexer->insert(1, "hello foo").has_value());
  EXPECT_TRUE(indexer->insert(2, "hello world bar").has_value());
  EXPECT_TRUE(indexer->insert(3, "hello baz").has_value());
  EXPECT_TRUE(indexer->flush().has_value());

  // Baseline: no filter — all 4 docs match "hello".
  std::vector<FtsResult> baseline;
  EXPECT_TRUE(search_ok(*indexer, "hello", 10, &baseline));
  EXPECT_EQ(baseline.size(), 4u);

  // Block docs 1 and 3.
  std::vector<FtsResult> filtered;
  EXPECT_TRUE(search_ok_with_filter(*indexer, "hello", 10,
                                    make_blocked_filter({1, 3}), &filtered));
  ASSERT_EQ(filtered.size(), 2u);

  std::vector<uint64_t> ids;
  for (const auto &r : filtered) {
    ids.push_back(r.doc_id);
    EXPECT_GT(r.score, 0.0f);
  }
  std::sort(ids.begin(), ids.end());
  EXPECT_EQ(ids[0], 0ull);
  EXPECT_EQ(ids[1], 2ull);
}

// OR query exercises DisjunctionIterator::next_doc(filter) override —
// pivot_doc is filter-checked before block-max accumulation and resort.
TEST_F(FtsColumnIndexerTest, FilterPushdownWithDisjunction) {
  auto indexer = make_indexer("content");
  EXPECT_TRUE(indexer->insert(0, "alpha beta").has_value());
  EXPECT_TRUE(indexer->insert(1, "alpha gamma").has_value());
  EXPECT_TRUE(indexer->insert(2, "beta gamma").has_value());
  EXPECT_TRUE(indexer->insert(3, "alpha beta gamma").has_value());
  EXPECT_TRUE(indexer->flush().has_value());

  // Baseline: "alpha OR beta" matches all 4 docs.
  std::vector<FtsResult> baseline;
  EXPECT_TRUE(search_ok(*indexer, "alpha beta", 10, &baseline));
  EXPECT_EQ(baseline.size(), 4u);

  std::vector<FtsResult> filtered;
  EXPECT_TRUE(search_ok_with_filter(*indexer, "alpha beta", 10,
                                    make_blocked_filter({0, 2}), &filtered));
  ASSERT_EQ(filtered.size(), 2u);

  std::vector<uint64_t> ids;
  for (const auto &r : filtered) {
    ids.push_back(r.doc_id);
    EXPECT_GT(r.score, 0.0f);
  }
  std::sort(ids.begin(), ids.end());
  EXPECT_EQ(ids[0], 1ull);
  EXPECT_EQ(ids[1], 3ull);
}

// Phrase query exercises PhraseDocIterator::next_doc(filter) -> inner
// ConjunctionIterator::next_doc(filter), ensuring verify_phrase_positions()
// is never executed for filtered docs.
TEST_F(FtsColumnIndexerTest, FilterPushdownWithPhrase) {
  auto indexer = make_indexer("content");
  EXPECT_TRUE(indexer->insert(0, "machine learning model").has_value());
  EXPECT_TRUE(indexer->insert(1, "machine learning notes").has_value());
  EXPECT_TRUE(indexer->insert(2, "learning machine translation").has_value());
  EXPECT_TRUE(indexer->insert(3, "machine learning systems").has_value());
  EXPECT_TRUE(indexer->flush().has_value());

  // Baseline: phrase "machine learning" matches docs 0, 1, 3 (not 2, where
  // the order is reversed).
  std::vector<FtsResult> baseline;
  EXPECT_TRUE(search_ok(*indexer, "\"machine learning\"", 10, &baseline));
  EXPECT_EQ(baseline.size(), 3u);

  // Block docs 1 and 3 — only doc 0 should remain.
  std::vector<FtsResult> filtered;
  EXPECT_TRUE(search_ok_with_filter(*indexer, "\"machine learning\"", 10,
                                    make_blocked_filter({1, 3}), &filtered));
  ASSERT_EQ(filtered.size(), 1u);
  EXPECT_EQ(filtered[0].doc_id, 0ull);
  EXPECT_GT(filtered[0].score, 0.0f);
}

// ============================================================
// Brute-force (candidate-driven) mode via FtsQueryParams.candidate_ids
// ============================================================

namespace {

// Helper: run a query with an explicit candidate id list.
template <typename Reader>
static bool search_ok_with_candidates(Reader &reader,
                                      const std::string &query_str,
                                      uint32_t topk,
                                      std::vector<uint64_t> candidates,
                                      std::vector<FtsResult> *results) {
  FtsQueryParser parser;
  auto ast = parser.parse(query_str, make_whitespace_pipeline());
  if (!ast) {
    ADD_FAILURE() << "FtsQueryParser failed to parse: " << query_str
                  << " err: " << parser.err_msg();
    return false;
  }
  zvec::fts::FtsQueryParams qp;
  qp.topk = topk;
  qp.candidate_ids = std::move(candidates);
  auto ret = reader.search(*ast, qp);
  if (!ret.has_value()) {
    return false;
  }
  *results = std::move(ret.value());
  return true;
}

// Compare two result vectors as (doc_id, score) sets — order independent on
// doc_id, scores compared with FLOAT_EQ. Brute-force and posting-driven
// paths reuse the same TermDocIterator / BM25Scorer so scores must agree.
static void ExpectSameResults(std::vector<FtsResult> a,
                              std::vector<FtsResult> b) {
  ASSERT_EQ(a.size(), b.size());
  auto by_id = [](const FtsResult &x, const FtsResult &y) {
    return x.doc_id < y.doc_id;
  };
  std::sort(a.begin(), a.end(), by_id);
  std::sort(b.begin(), b.end(), by_id);
  for (size_t i = 0; i < a.size(); ++i) {
    EXPECT_EQ(a[i].doc_id, b[i].doc_id) << "i=" << i;
    EXPECT_FLOAT_EQ(a[i].score, b[i].score) << "i=" << i;
  }
}

}  // namespace

// Single-term query: candidate-driven path returns the intersection of the
// term posting and the candidate set, with the same BM25 scores as the
// posting-driven baseline.
TEST_F(FtsColumnIndexerTest, BruteForceTermMatchesPostingDriven) {
  auto indexer = make_indexer("content");
  EXPECT_TRUE(indexer->insert(0, "hello world").has_value());
  EXPECT_TRUE(indexer->insert(1, "hello foo").has_value());
  EXPECT_TRUE(indexer->insert(2, "hello world bar").has_value());
  EXPECT_TRUE(indexer->insert(3, "hello baz").has_value());
  EXPECT_TRUE(indexer->insert(4, "world only").has_value());
  EXPECT_TRUE(indexer->flush().has_value());

  // Baseline: "hello" matches docs 0,1,2,3.
  std::vector<FtsResult> baseline;
  EXPECT_TRUE(search_ok(*indexer, "hello", 10, &baseline));
  ASSERT_EQ(baseline.size(), 4u);

  // Candidate-driven with {1, 2, 4} -> expect {1, 2} (4 is not in posting).
  std::vector<FtsResult> bf;
  EXPECT_TRUE(search_ok_with_candidates(*indexer, "hello", 10,
                                        /*candidates=*/{1, 2, 4}, &bf));

  std::vector<FtsResult> expected;
  for (const auto &r : baseline) {
    if (r.doc_id == 1 || r.doc_id == 2) expected.push_back(r);
  }
  ExpectSameResults(std::move(expected), std::move(bf));
}

// Disjunction (OR) — same BM25 score, only intersected docs returned.
TEST_F(FtsColumnIndexerTest, BruteForceDisjunctionMatchesPostingDriven) {
  auto indexer = make_indexer("content");
  EXPECT_TRUE(indexer->insert(0, "alpha beta").has_value());
  EXPECT_TRUE(indexer->insert(1, "alpha gamma").has_value());
  EXPECT_TRUE(indexer->insert(2, "beta gamma").has_value());
  EXPECT_TRUE(indexer->insert(3, "alpha beta gamma").has_value());
  EXPECT_TRUE(indexer->insert(4, "delta").has_value());
  EXPECT_TRUE(indexer->flush().has_value());

  std::vector<FtsResult> baseline;
  EXPECT_TRUE(search_ok(*indexer, "alpha beta", 10, &baseline));
  ASSERT_EQ(baseline.size(), 4u);  // 0,1,2,3 all match OR

  std::vector<FtsResult> bf;
  EXPECT_TRUE(search_ok_with_candidates(*indexer, "alpha beta", 10,
                                        /*candidates=*/{0, 3, 4}, &bf));

  std::vector<FtsResult> expected;
  for (const auto &r : baseline) {
    if (r.doc_id == 0 || r.doc_id == 3) expected.push_back(r);
  }
  ExpectSameResults(std::move(expected), std::move(bf));
}

// Conjunction (AND) — wrapped AND-of-AND is semantically transparent.
TEST_F(FtsColumnIndexerTest, BruteForceConjunctionMatchesPostingDriven) {
  auto indexer = make_indexer("content");
  EXPECT_TRUE(indexer->insert(0, "alpha beta gamma").has_value());
  EXPECT_TRUE(indexer->insert(1, "alpha gamma").has_value());  // missing beta
  EXPECT_TRUE(indexer->insert(2, "alpha beta").has_value());   // missing gamma
  EXPECT_TRUE(indexer->insert(3, "alpha beta gamma").has_value());
  EXPECT_TRUE(indexer->insert(4, "alpha beta gamma").has_value());
  EXPECT_TRUE(indexer->flush().has_value());

  std::vector<FtsResult> baseline;
  EXPECT_TRUE(search_ok(*indexer, "alpha AND beta AND gamma", 10, &baseline));
  ASSERT_EQ(baseline.size(), 3u);  // 0,3,4

  std::vector<FtsResult> bf;
  EXPECT_TRUE(search_ok_with_candidates(*indexer, "alpha AND beta AND gamma",
                                        10, /*candidates=*/{0, 1, 4}, &bf));

  std::vector<FtsResult> expected;
  for (const auto &r : baseline) {
    if (r.doc_id == 0 || r.doc_id == 4) expected.push_back(r);
  }
  ExpectSameResults(std::move(expected), std::move(bf));
}

// Phrase query — phase-2 position check is preserved in candidate-driven mode.
TEST_F(FtsColumnIndexerTest, BruteForcePhraseMatchesPostingDriven) {
  auto indexer = make_indexer("content");
  EXPECT_TRUE(indexer->insert(0, "machine learning model").has_value());
  EXPECT_TRUE(indexer->insert(1, "machine notes learning").has_value());
  EXPECT_TRUE(indexer->insert(2, "the machine learning jumps").has_value());
  EXPECT_TRUE(indexer->insert(3, "learning machine").has_value());
  EXPECT_TRUE(indexer->flush().has_value());

  std::vector<FtsResult> baseline;
  EXPECT_TRUE(search_ok(*indexer, "\"machine learning\"", 10, &baseline));
  ASSERT_EQ(baseline.size(), 2u);  // 0,2

  // Candidate set = {1, 2, 3}: only 2 is a real phrase match.
  std::vector<FtsResult> bf;
  EXPECT_TRUE(search_ok_with_candidates(*indexer, "\"machine learning\"", 10,
                                        /*candidates=*/{1, 2, 3}, &bf));

  std::vector<FtsResult> expected;
  for (const auto &r : baseline) {
    if (r.doc_id == 2) expected.push_back(r);
  }
  ExpectSameResults(std::move(expected), std::move(bf));
}

// Nested (AND of OR) — root iterator type does not matter; wrap is
// transparent.
TEST_F(FtsColumnIndexerTest, BruteForceNestedMatchesPostingDriven) {
  auto indexer = make_indexer("content");
  EXPECT_TRUE(indexer->insert(0, "alpha").has_value());
  EXPECT_TRUE(indexer->insert(1, "beta").has_value());
  EXPECT_TRUE(indexer->insert(2, "alpha gamma").has_value());  // matches
  EXPECT_TRUE(indexer->insert(3, "beta gamma").has_value());   // matches
  EXPECT_TRUE(indexer->insert(4, "gamma only").has_value());   // no alpha/beta
  EXPECT_TRUE(indexer->flush().has_value());

  // (alpha OR beta) AND gamma -> docs 2, 3
  std::vector<FtsResult> baseline;
  EXPECT_TRUE(search_ok(*indexer, "(alpha OR beta) AND gamma", 10, &baseline));
  ASSERT_EQ(baseline.size(), 2u);

  std::vector<FtsResult> bf;
  EXPECT_TRUE(search_ok_with_candidates(*indexer, "(alpha OR beta) AND gamma",
                                        10, /*candidates=*/{2, 4}, &bf));

  std::vector<FtsResult> expected;
  for (const auto &r : baseline) {
    if (r.doc_id == 2) expected.push_back(r);
  }
  ExpectSameResults(std::move(expected), std::move(bf));
}

// Candidate-driven coexists with the existing filter pushdown:
// candidate_ids narrows the doc set; filter further drops some.
TEST_F(FtsColumnIndexerTest, BruteForceCoexistsWithFilterPushdown) {
  auto indexer = make_indexer("content");
  EXPECT_TRUE(indexer->insert(0, "alpha").has_value());
  EXPECT_TRUE(indexer->insert(1, "alpha").has_value());
  EXPECT_TRUE(indexer->insert(2, "alpha").has_value());
  EXPECT_TRUE(indexer->insert(3, "alpha").has_value());
  EXPECT_TRUE(indexer->flush().has_value());

  FtsQueryParser parser;
  auto ast = parser.parse("alpha", make_whitespace_pipeline());
  ASSERT_NE(ast, nullptr);

  zvec::fts::FtsQueryParams qp;
  qp.topk = 10;
  qp.candidate_ids = {0, 1, 2};          // candidates restrict to {0,1,2}
  qp.filter = make_blocked_filter({1});  // further drop doc 1
  auto ret = indexer->search(*ast, qp);
  ASSERT_TRUE(ret.has_value());
  auto results = std::move(ret.value());
  ASSERT_EQ(results.size(), 2u);

  std::vector<uint64_t> ids;
  for (const auto &r : results) ids.push_back(r.doc_id);
  std::sort(ids.begin(), ids.end());
  EXPECT_EQ(ids[0], 0ull);
  EXPECT_EQ(ids[1], 2ull);
}

// Empty candidate_ids takes the regular posting-driven path (the wrap guard
// requires non-empty), so search still finds all matching docs.
TEST_F(FtsColumnIndexerTest, BruteForceEmptyCandidatesFallsBack) {
  auto indexer = make_indexer("content");
  EXPECT_TRUE(indexer->insert(0, "alpha beta").has_value());
  EXPECT_TRUE(indexer->insert(1, "alpha gamma").has_value());
  EXPECT_TRUE(indexer->flush().has_value());

  std::vector<FtsResult> r;
  EXPECT_TRUE(search_ok_with_candidates(*indexer, "alpha", 10, {}, &r));
  EXPECT_EQ(r.size(), 2u);
}

// Regression guard: a null filter yields the same doc_ids and scores as the
// baseline path (which still uses the no-filter next_doc() overload).
TEST_F(FtsColumnIndexerTest, FilterPushdownNullFilterUnchanged) {
  auto indexer = make_indexer("content");
  EXPECT_TRUE(indexer->insert(0, "quick brown fox").has_value());
  EXPECT_TRUE(indexer->insert(1, "lazy brown dog").has_value());
  EXPECT_TRUE(indexer->flush().has_value());

  std::vector<FtsResult> baseline;
  EXPECT_TRUE(search_ok(*indexer, "brown", 10, &baseline));
  ASSERT_EQ(baseline.size(), 2u);

  std::vector<FtsResult> with_null;
  EXPECT_TRUE(search_ok_with_filter(*indexer, "brown", 10, /*filter=*/nullptr,
                                    &with_null));
  ASSERT_EQ(with_null.size(), 2u);

  auto by_id = [](const FtsResult &a, const FtsResult &b) {
    return a.doc_id < b.doc_id;
  };
  std::sort(baseline.begin(), baseline.end(), by_id);
  std::sort(with_null.begin(), with_null.end(), by_id);
  for (size_t i = 0; i < baseline.size(); ++i) {
    EXPECT_EQ(baseline[i].doc_id, with_null[i].doc_id);
    EXPECT_FLOAT_EQ(baseline[i].score, with_null[i].score);
  }
}
