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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/db/index_params.h>
#include "db/common/file_helper.h"
// FtsQueryParams defined below
#include "db/index/column/fts_column/fts_rocksdb_merge.h"
#include "db/index/column/fts_column/parser/fts_query_parser.h"
#include "db/index/column/fts_column/tokenizer_factory.h"
// meta.h not needed in zvec
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

// Helper: parse a query string and call search() on a reader/indexer.
// Terminates the test with ASSERT if parsing fails.
template <typename Reader>
static bool search_ok(Reader &reader, const std::string &query_str,
                      uint32_t topk, std::vector<FtsResult> *results) {
  FtsQueryParser parser;
  auto ast = parser.parse(query_str);
  if (!ast) {
    ADD_FAILURE() << "FtsQueryParser failed to parse: " << query_str
                  << " err: " << parser.err_msg();
    return false;
  }
  zvec::fts::FtsQueryParams qp;
  qp.topk = topk;
  auto ret = reader.search(*ast, qp, results);
  return ret.has_value();
}

// ============================================================
// Test fixture
// ============================================================

static const std::string kDbPath{"./test_fts_db"};

static const std::string kPostingsCf{"fts_postings"};
static const std::string kMaxTfCf{"fts_max_tf"};
static const std::string kPositionsCf{"fts_positions"};
static const std::string kTermFreqCf{"fts_tf"};
static const std::string kDocLenCf{"fts_doc_len"};
static const std::string kStatCf{"fts_stat"};

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
    ASSERT_TRUE(db_.create(kDbPath, cf_names, nullptr, per_cf_ops).ok());

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
  auto ret =
      reader.open("content", &db_, postings_cf_, positions_cf_, term_freq_cf_,
                  max_tf_cf_, /*doc_len_cf=*/nullptr, stat_cf_);
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
  auto ast = parser.parse("-(hello AND world)");
  ASSERT_NE(ast, nullptr);
  EXPECT_TRUE(ast->must_not);

  std::vector<FtsResult> results;
  FtsQueryParams query_params;
  query_params.topk = 10;
  EXPECT_FALSE(indexer->search(*ast, query_params, &results).has_value());
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

static std::string make_jieba_extra_params() {
  return std::string(R"({"dict_path":")") + kJiebaDictDir +
         R"(/jieba.dict.utf8","model_path":")" + kJiebaDictDir +
         R"(/hmm_model.utf8"})";
}

class FtsColumnIndexerJiebaTest : public FtsColumnIndexerTest {
 protected:
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

// Verify that jieba tokenizer fails to open when required model_path is
// missing.  (Note: cppjieba FATAL-aborts on non-existent dict files, so we
// test the init-time validation in JiebaTokenizer instead.)
TEST_F(FtsColumnIndexerJiebaTest, OpenWithJiebaTokenizerFailsWithoutModelPath) {
  fts::FtsIndexParams bad_params;
  bad_params.tokenizer_name = "jieba";
  // Provide dict_path but omit model_path — JiebaTokenizer::init should fail.
  bad_params.extra_params = std::string(R"({"dict_path":")") + kJiebaDictDir +
                            R"(/jieba.dict.utf8"})";
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
  auto ret =
      reader.open("content", &db_, postings_cf_, positions_cf_, term_freq_cf_,
                  max_tf_cf_, /*doc_len_cf=*/nullptr, stat_cf_);
  EXPECT_TRUE(ret.has_value());

  // Search with a term that jieba produces from "深度学习模型":
  // jieba CutForSearch segments it into [深度, 学习, 深度学习, 模型].
  std::vector<FtsResult> results;
  TermNode term_node("模型");
  FtsQueryParams query_params;
  query_params.topk = 10;
  EXPECT_TRUE(reader.search(term_node, query_params, &results).has_value());
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

#include "db/index/column/fts_column/bitpacked_posting_list.h"  // NOLINT: in-test include

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
                  .open("content", &db_, postings_cf_, positions_cf_,
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
static const std::string kSharedStatCf{"fts_stat"};

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
      cf_names.push_back(f);                 // postings
      cf_names.push_back(f + "_positions");  // positions
      cf_names.push_back(f + "_tf");         // term freq
      cf_names.push_back(f + "_max_tf");     // max tf
      cf_names.push_back(f + "_doc_len");    // doc len

      per_cf_ops[f] = std::make_shared<FtsPostingsMerge>();
      per_cf_ops[f + "_max_tf"] = std::make_shared<FtsMaxTfMerge>();
    }
    cf_names.push_back(kSharedStatCf);

    ASSERT_TRUE(db_.create(kMultiDbPath, cf_names, nullptr, per_cf_ops).ok());

    // Resolve CF handles per field.
    for (size_t i = 0; i < kNumFields; ++i) {
      std::string f{kFields[i]};
      postings_cf_[i] = db_.get_cf(f);
      positions_cf_[i] = db_.get_cf(f + "_positions");
      term_freq_cf_[i] = db_.get_cf(f + "_tf");
      max_tf_cf_[i] = db_.get_cf(f + "_max_tf");
      doc_len_cf_[i] = db_.get_cf(f + "_doc_len");
      ASSERT_NE(postings_cf_[i], nullptr) << "field=" << f;
      ASSERT_NE(positions_cf_[i], nullptr) << "field=" << f;
      ASSERT_NE(term_freq_cf_[i], nullptr) << "field=" << f;
      ASSERT_NE(max_tf_cf_[i], nullptr) << "field=" << f;
      ASSERT_NE(doc_len_cf_[i], nullptr) << "field=" << f;
    }
    stat_cf_ = db_.get_cf(kSharedStatCf);
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
                  .open("title", &db_, postings_cf_[ti], positions_cf_[ti],
                        term_freq_cf_[ti], max_tf_cf_[ti],
                        /*doc_len_cf=*/nullptr, stat_cf_)
                  .has_value());

  FtsColumnIndexer body_reader;
  ASSERT_TRUE(body_reader
                  .open("body", &db_, postings_cf_[bi], positions_cf_[bi],
                        term_freq_cf_[bi], max_tf_cf_[bi],
                        /*doc_len_cf=*/nullptr, stat_cf_)
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
