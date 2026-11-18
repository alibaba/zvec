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

#include <map>
#include <queue>
#include <sstream>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_factory.h>
#include "db/common/file_helper.h"
#include "db/index/column/fts_column/fts_ast_rewriter.h"
#include "db/index/column/fts_column/fts_field.h"
#include "db/index/column/fts_column/fts_indexer.h"
#include "db/index/column/fts_column/parser/fts_query_parser.h"

namespace zvec::fts {
namespace {

class FtsReaderTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(0,
              ailego::MemoryLimitPool::get_instance().init(32 * 1024 * 1024));
    FileHelper::RemoveDirectory(legacy_path_);
    auto params = std::make_shared<zvec::FtsIndexParams>("whitespace");
    auto field =
        std::make_shared<FieldSchema>("text", DataType::STRING, false, params);
    legacy_ = FtsIndexer::CreateAndOpen(legacy_path_, {field}, true);
    ASSERT_TRUE(legacy_);
    FtsIndexParams pipeline_params;
    pipeline_params.tokenizer_name = "whitespace";
    auto pipeline = TokenizerFactory::create(pipeline_params);
    ASSERT_TRUE(pipeline);
    pipeline_ = *pipeline;
  }
  void TearDown() override {
    legacy_.reset();
    FileHelper::RemoveDirectory(legacy_path_);
    std::remove(custom_path_.c_str());
  }

  void build(uint32_t rows = 300) {
    // Independent text input for both formats. The test's small in-memory
    // oracle is not a production conversion path.
    std::map<std::string, std::map<uint32_t, std::vector<uint32_t>>> terms;
    std::vector<std::optional<uint32_t>> lengths(rows);
    uint64_t tokens = 0, present = 0;
    for (uint32_t row = 0; row < lengths.size(); ++row) {
      if (row % 17 == 1) {
        continue;  // Missing field differs from present zero-token text.
      }
      std::string text;
      if (row != 0) {
        const char *patterns[] = {"alpha beta alpha", "beta gamma",
                                  "alpha alpha beta", "gamma alpha",
                                  "alpha beta gamma"};
        text = patterns[row % 5];
        for (uint32_t i = 0; i < row % 7; ++i) {
          text += " delta";
        }
        if (row + 1 == rows) {
          text += " unique";
        }
      }
      ASSERT_TRUE(legacy_->insert("text", row, text).ok());
      std::istringstream input(text);
      std::string word;
      uint32_t position = 0;
      while (input >> word) {
        terms[word][row].push_back(position++);
      }
      lengths[row] = position;
      tokens += position;
      ++present;
    }
    ASSERT_TRUE(legacy_->flush().ok());
    format::FieldMetadata field;
    field.name = "text";
    field.id[0] = 1;
    field.incarnation[0] = 2;
    field.present_count = field.scoring_document_count = present;
    field.total_tokens = tokens;
    auto writer = FtsFieldWriter::Create(lengths.size(), field);
    ASSERT_TRUE(writer);
    for (uint32_t row = 0; row < lengths.size(); ++row) {
      ASSERT_TRUE((*writer)
                      ->add_document(row, lengths[row].has_value(),
                                     lengths[row].value_or(0))
                      .ok());
    }
    ASSERT_TRUE((*writer)->finish_documents().ok());
    for (const auto &[term, documents] : terms) {
      ASSERT_TRUE((*writer)->begin_term(term).ok());
      for (const auto &[doc, positions] : documents) {
        const auto &positions_view = positions;
        size_t offset = 0;
        ASSERT_TRUE((*writer)
                        ->add_posting(doc, positions.size(),
                                      [&]() -> Result<std::optional<uint32_t>> {
                                        if (offset == positions_view.size()) {
                                          return std::optional<uint32_t>{};
                                        }
                                        return std::optional<uint32_t>{
                                            positions_view[offset++]};
                                      })
                        .ok());
      }
      ASSERT_TRUE((*writer)->end_term().ok());
    }
    auto dumper = core::IndexFactory::CreateDumper("FileDumper");
    ASSERT_EQ(0, dumper->create(custom_path_));
    auto container = FtsContainerWriter::Create(dumper);
    ASSERT_TRUE(container);
    auto completed = (*writer)->finish(container->get(), 0);
    ASSERT_TRUE(completed);
    format::Metadata metadata;
    metadata.base_id[0] = 3;
    metadata.base_generation = 1;
    metadata.row_count = lengths.size();
    metadata.fields.push_back(*completed);
    ASSERT_TRUE((*container)->finish(metadata).ok());
  }

  std::string legacy_path_{"fts_reader_legacy"};
  std::string custom_path_{"fts_reader_custom.zfts"};
  FtsIndexer::Ptr legacy_;
  TokenizerPipelinePtr pipeline_;
};

TEST_F(FtsReaderTest, DifferentialQueriesPreserveResultsScoresAndTies) {
  build();
  const std::vector<std::string> queries{"alpha",
                                         "unique",
                                         "absent",
                                         "alpha OR beta",
                                         "alpha OR beta OR gamma OR delta",
                                         "alpha AND beta",
                                         "alpha beta gamma",
                                         "alpha alpha alpha",
                                         "\"alpha beta\"",
                                         "\"alpha alpha\"",
                                         "\"alpha beta alpha\"",
                                         "\"beta alpha\" OR gamma",
                                         "+alpha beta",
                                         "alpha -gamma",
                                         "alpha -\"beta gamma\"",
                                         "(alpha OR beta) AND delta",
                                         "alpha AND (beta OR gamma)",
                                         "+alpha \"beta gamma\""};
  struct Case {
    FtsAstNodePtr ast;
    FtsQueryParams params;
    std::vector<FtsResult> reference;
  };
  std::vector<Case> cases;
  auto legacy = legacy_->get("text");
  for (const auto &query : queries) {
    for (uint32_t topk : {1u, 7u, 500u}) {
      for (int selection = 0; selection < 3; ++selection) {
        FtsQueryParser parser;
        auto ast = parser.parse(query, pipeline_);
        ASSERT_TRUE(ast) << query;
        simplify(ast);
        FtsQueryParams params;
        params.topk = topk;
        if (selection > 0) {
          params.filter = EasyIndexFilter::Create(
              [](uint64_t row) { return row % 3 == 0; });
        }
        if (selection == 2) {
          params.candidate_ids.emplace();
          for (uint64_t row = 0; row < 300; row += 5) {
            params.candidate_ids->push_back(row);
          }
        }
        // More than all 300 rows: the heap never fills, so this reference
        // cannot prune regardless of the term/block bound implementation.
        auto reference_params = params;
        reference_params.topk = 500;
        auto result = legacy->search(*ast, reference_params);
        ASSERT_TRUE(result) << query << " " << result.error();
        // Heap size affects equal-score extraction order. Replay all scored
        // rows in DocID order through the existing score-only TopK contract,
        // instead of truncating the differently-sized reference heap.
        std::sort(
            result->begin(), result->end(),
            [](const auto &a, const auto &b) { return a.doc_id < b.doc_id; });
        std::priority_queue<FtsResult, std::vector<FtsResult>,
                            std::greater<FtsResult>>
            heap;
        for (const auto &scored : *result) {
          if (heap.size() < topk) {
            heap.push(scored);
          } else if (scored.score > heap.top().score) {
            heap.pop();
            heap.push(scored);
          }
        }
        result->resize(heap.size());
        for (auto it = result->rbegin(); it != result->rend(); ++it) {
          *it = heap.top();
          heap.pop();
        }
        cases.push_back({std::move(ast), params, std::move(*result)});
      }
    }
  }
  auto compare = [&](FtsColumnIndexer &reader) {
    for (const auto &test : cases) {
      SCOPED_TRACE(test.ast->text() +
                   " topk=" + std::to_string(test.params.topk));
      auto actual = reader.search(*test.ast, test.params);
      ASSERT_TRUE(actual) << actual.error();
      ASSERT_EQ(test.reference.size(), actual->size());
      for (size_t i = 0; i < actual->size(); ++i) {
        EXPECT_EQ(test.reference[i].doc_id, (*actual)[i].doc_id);
        EXPECT_EQ(test.reference[i].score, (*actual)[i].score);
      }
    }
  };
  ASSERT_TRUE(legacy_->seal_all().ok());
  compare(*legacy);
  for (bool mmap : {false, true}) {
    auto file = FtsContainerReader::Open(custom_path_, mmap);
    ASSERT_TRUE(file);
    auto field = FtsFieldReader::Open(*file, 0);
    ASSERT_TRUE(field);
    FtsColumnIndexer reader;
    ASSERT_TRUE(reader.open_reader(*field));
    compare(reader);  // Reopened: conservative runtime bounds, no full scan.
    ASSERT_TRUE((*field)->verify().ok());
    compare(reader);  // Validated finite bounds: same documents and scores.
  }
}

TEST_F(FtsReaderTest,
       FlatDisjunctionPreservesScoresAndTiesAcrossLargeRowRanges) {
  constexpr uint32_t rows = 8449;
  build(rows);
  ASSERT_TRUE(legacy_->seal_all().ok());
  for (const std::vector<float> &boosts :
       {std::vector<float>{1, 1, 1, 1, 1},
        std::vector<float>{1.0e8f, 1, 3, 1, 0},
        std::vector<float>{0, 0, 0, 0, 0}}) {
    OrNode query;
    const char *names[] = {"alpha", "gamma", "alpha", "absent", "beta"};
    for (size_t i = 0; i < boosts.size(); ++i) {
      auto term = std::make_unique<TermNode>(names[i]);
      term->boost = boosts[i];
      query.children.push_back(std::move(term));
    }
    FtsQueryParams all;
    all.topk = rows + 1;  // Independent legacy OR without competitive pruning.
    auto reference = legacy_->get("text")->search(query, all);
    ASSERT_TRUE(reference);
    std::sort(reference->begin(), reference->end(),
              [](const auto &a, const auto &b) { return a.doc_id < b.doc_id; });
    for (uint32_t topk : {1u, 17u, 200u}) {
      std::priority_queue<FtsResult, std::vector<FtsResult>,
                          std::greater<FtsResult>>
          heap;
      for (const auto &row : *reference) {
        if (heap.size() < topk) {
          heap.push(row);
        } else if (row.score > heap.top().score) {
          heap.pop();
          heap.push(row);
        }
      }
      std::vector<FtsResult> expected(heap.size());
      for (auto it = expected.rbegin(); it != expected.rend(); ++it) {
        *it = heap.top();
        heap.pop();
      }
      for (bool mmap : {false, true}) {
        auto file = FtsContainerReader::Open(custom_path_, mmap);
        ASSERT_TRUE(file);
        auto field = FtsFieldReader::Open(*file, 0);
        ASSERT_TRUE(field);
        FtsColumnIndexer reader;
        ASSERT_TRUE(reader.open_reader(*field));
        FtsQueryParams params;
        params.topk = topk;
        auto actual = reader.search(query, params);
        ASSERT_TRUE(actual) << actual.error();
        ASSERT_EQ(expected.size(), actual->size());
        for (size_t i = 0; i < expected.size(); ++i) {
          EXPECT_EQ(expected[i].doc_id, (*actual)[i].doc_id);
          EXPECT_EQ(expected[i].score, (*actual)[i].score);
        }
      }
    }
  }
}

TEST_F(FtsReaderTest, NegatedPhraseChecksPositionsBeforeExcluding) {
  ASSERT_TRUE(legacy_->insert("text", 0, "alpha beta gamma").ok());
  ASSERT_TRUE(legacy_->insert("text", 1, "alpha beta x gamma").ok());
  ASSERT_TRUE(legacy_->insert("text", 2, "alpha gamma beta").ok());
  ASSERT_TRUE(legacy_->flush().ok());
  FtsQueryParser parser;
  auto ast = parser.parse("alpha -\"beta gamma\"", pipeline_);
  ASSERT_TRUE(ast);
  simplify(ast);
  FtsQueryParams params;
  params.topk = 10;
  for (bool sealed : {false, true}) {
    if (sealed) {
      ASSERT_TRUE(legacy_->seal_all().ok());
    }
    auto result = legacy_->get("text")->search(*ast, params);
    ASSERT_TRUE(result);
    ASSERT_EQ(2u, result->size());
    EXPECT_EQ(2u, (*result)[0].doc_id);
    EXPECT_EQ(1u, (*result)[1].doc_id);
  }
}

}  // namespace
}  // namespace zvec::fts
