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

#include "db/index/column/fts_column/fts_field.h"
#include <atomic>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <set>
#include <thread>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_factory.h>
#include "db/index/column/fts_column/fts_column_indexer.h"
#include "db/index/column/fts_column/fts_merge.h"
#include "db/index/column/fts_column/fts_source.h"
#include "db/index/column/fts_column/iterator/fts_batch_disjunction_iterator.h"

namespace zvec::fts {
namespace {

struct TermInput {
  std::string text;
  uint32_t df;
  uint32_t tf;
};

class FtsFieldTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(0,
              ailego::MemoryLimitPool::get_instance().init(16 * 1024 * 1024));
    path_ = "fts_field_test_" +
            std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()) +
            ".zfts";
  }
  void TearDown() override {
    std::remove(path_.c_str());
  }
  void build(const std::vector<TermInput> &terms, uint32_t fields = 1,
             uint32_t rows = 260) {
    lengths_.assign(rows, 0);
    uint64_t tokens = 0;
    uint64_t present = 1;  // The final row exists but has no tokens.
    for (const auto &term : terms) {
      for (uint32_t i = 0; i < term.df; ++i) {
        lengths_[i * 2] += term.tf;
        tokens += term.tf;
      }
    }
    for (auto length : lengths_) {
      present += length != 0 ? 1 : 0;
    }
    auto dumper = core::IndexFactory::CreateDumper("FileDumper");
    ASSERT_EQ(0, dumper->create(path_));
    auto container = FtsContainerWriter::Create(dumper);
    ASSERT_TRUE(container);
    format::Metadata metadata;
    metadata.base_id[0] = 1;
    metadata.base_generation = 9;
    metadata.row_count = lengths_.size();
    for (uint32_t field = 0; field < fields; ++field) {
      format::FieldMetadata info;
      info.id[0] = static_cast<uint8_t>(field + 2);
      info.incarnation[0] = static_cast<uint8_t>(field + 100);
      info.name = "text_" + std::to_string(field);
      info.present_count = present;
      info.scoring_document_count = lengths_.size();  // Reducer-compatible N.
      info.total_tokens = tokens;
      auto writer = FtsFieldWriter::Create(lengths_.size(), info);
      ASSERT_TRUE(writer);
      for (uint32_t row = 0; row < lengths_.size(); ++row) {
        if (lengths_[row] != 0 || row + 1 == lengths_.size()) {
          ASSERT_TRUE((*writer)->add_document(row, true, lengths_[row]).ok());
        }
      }
      ASSERT_TRUE((*writer)->finish_documents().ok());
      for (const auto &term : terms) {
        ASSERT_TRUE((*writer)->begin_term(term.text).ok());
        for (uint32_t i = 0; i < term.df; ++i) {
          uint32_t emitted = 0;
          ASSERT_TRUE(
              (*writer)
                  ->add_posting(i * 2, term.tf,
                                [&]() -> Result<std::optional<uint32_t>> {
                                  if (emitted == term.tf) {
                                    return std::optional<uint32_t>{};
                                  }
                                  return std::optional<uint32_t>{emitted++ / 2};
                                })
                  .ok());
        }
        ASSERT_TRUE((*writer)->end_term().ok());
      }
      auto completed = (*writer)->finish(container->get(), field);
      ASSERT_TRUE(completed);
      metadata.fields.push_back(*completed);
    }
    ASSERT_TRUE((*container)->finish(metadata).ok());
  }

  void rewrite(const std::string &section_name,
               const std::function<void(std::string *)> &mutate) {
    format::Metadata metadata;
    std::vector<std::pair<std::string, std::string>> sections;
    {
      auto source = FtsContainerReader::Open(path_, false);
      ASSERT_TRUE(source);
      metadata = (*source)->metadata();
      for (const auto &section : metadata.sections) {
        ASSERT_LT(section.length, 65536u);
        auto range = (*source)->read(section.name, 0, section.length);
        ASSERT_TRUE(range);
        sections.emplace_back(section.name, range->view());
        if (section.name == section_name) {
          mutate(&sections.back().second);
        }
      }
    }
    auto dumper = core::IndexFactory::CreateDumper("FileDumper");
    ASSERT_EQ(0, dumper->create(path_));
    auto container = FtsContainerWriter::Create(dumper);
    ASSERT_TRUE(container);
    for (const auto &section : sections) {
      ASSERT_TRUE((*container)->begin_section(section.first).ok());
      ASSERT_TRUE((*container)->append(section.second).ok());
      ASSERT_TRUE((*container)->end_section().ok());
    }
    metadata.sections.clear();
    ASSERT_TRUE((*container)->finish(metadata).ok());
  }

  std::string path_;
  std::vector<uint32_t> lengths_;
};

TEST_F(FtsFieldTest, StreamsOversizedTermWithoutMaterializingItsKey) {
  const uint64_t length = 2 * 1024 * 1024 + 17;
  size_t largest_request = 0;
  auto term = [&](uint64_t offset, size_t count) -> Result<std::string> {
    largest_request = std::max(largest_request, count);
    if (count > format::kDictionaryBlockBytes || offset > length ||
        count > length - offset) {
      return tl::make_unexpected(
          Status::InvalidArgument("Unbounded term read"));
    }
    std::string bytes(count, 'q');
    if (count != 0 && offset + count == length) {
      bytes.back() = 'z';
    }
    return bytes;
  };
  auto dumper = core::IndexFactory::CreateDumper("FileDumper");
  ASSERT_EQ(0, dumper->create(path_));
  auto container = FtsContainerWriter::Create(dumper);
  ASSERT_TRUE(container);
  format::FieldMetadata info;
  info.id[0] = 1;
  info.incarnation[0] = 2;
  info.name = "text";
  info.present_count = info.scoring_document_count = info.total_tokens = 1;
  auto writer = FtsFieldWriter::Create(1, info);
  ASSERT_TRUE(writer);
  ASSERT_TRUE((*writer)->add_document(0, true, 1).ok());
  ASSERT_TRUE((*writer)->finish_documents().ok());
  auto status = (*writer)->begin_term(length, term);
  ASSERT_TRUE(status.ok()) << status.message();
  bool emitted = false;
  ASSERT_TRUE((*writer)
                  ->add_posting(0, 1,
                                [&]() -> Result<std::optional<uint32_t>> {
                                  if (emitted) {
                                    return std::optional<uint32_t>{};
                                  }
                                  emitted = true;
                                  return std::optional<uint32_t>{3};
                                })
                  .ok());
  ASSERT_TRUE((*writer)->end_term().ok());
  auto field = (*writer)->finish(container->get(), 0);
  ASSERT_TRUE(field);
  format::Metadata metadata;
  metadata.base_id[0] = 3;
  metadata.base_generation = 1;
  metadata.row_count = 1;
  metadata.fields.push_back(*field);
  ASSERT_TRUE((*container)->finish(metadata).ok());
  for (bool mmap : {false, true}) {
    auto file = FtsContainerReader::Open(path_, mmap);
    ASSERT_TRUE(file);
    auto reader = FtsFieldReader::Open(*file, 0);
    ASSERT_TRUE(reader);
    ASSERT_TRUE((*reader)->verify().ok());
    auto source = OpenCustomFtsFieldSource(*reader);
    ASSERT_TRUE(source.has_value()) << source.error().message();
    // The reducer compares and rewrites this same oversized key from two
    // sources, then purges the first row without materializing the key.
    const auto merged_path = path_ + (mmap ? ".merge-mmap" : ".merge-buffer");
    {
      auto merged_dumper = core::IndexFactory::CreateDumper("FileDumper");
      ASSERT_EQ(0, merged_dumper->create(merged_path));
      auto merged_container = FtsContainerWriter::Create(merged_dumper);
      ASSERT_TRUE(merged_container.has_value());
      roaring::Roaring deleted;
      deleted.add(0);
      auto merged =
          MergeFtsField({*source, *source}, deleted, FtsMergePolicy::kCompact,
                        info, merged_container->get(), 0);
      ASSERT_TRUE(merged.has_value()) << merged.error().message();
      auto merged_metadata = metadata;
      merged_metadata.fields = {*merged};
      ASSERT_TRUE((*merged_container)->finish(merged_metadata).ok());
      auto merged_file = FtsContainerReader::Open(merged_path, mmap);
      ASSERT_TRUE(merged_file.has_value());
      auto merged_reader = FtsFieldReader::Open(*merged_file, 0);
      ASSERT_TRUE(merged_reader.has_value());
      ASSERT_TRUE((*merged_reader)->verify().ok());
      auto merged_term = (*merged_reader)->term_at(0);
      ASSERT_TRUE(merged_term.has_value());
      EXPECT_EQ(length, merged_term->term_size());
      EXPECT_EQ(1, (*merged_reader)->metadata().scoring_document_count);
      auto merged_postings = (*merged_reader)->postings(*merged_term);
      ASSERT_TRUE(merged_postings.has_value());
      auto merged_doc = (*merged_postings)->next();
      ASSERT_TRUE(merged_doc && merged_doc->has_value());
      EXPECT_EQ(0, (**merged_doc).doc);
      EXPECT_EQ(1, (**merged_doc).tf);
      auto merged_positions = (*merged_postings)->positions();
      ASSERT_TRUE(merged_positions.has_value());
      auto merged_position = (*merged_positions)->next();
      ASSERT_TRUE(merged_position && merged_position->has_value());
      EXPECT_EQ(3, **merged_position);
      merged_positions->reset();
      merged_postings->reset();
      merged_reader->reset();
      merged_file->reset();
    }
    EXPECT_EQ(0, std::remove(merged_path.c_str()));
    EXPECT_EQ(1, (*source)->statistics().rows);
    auto source_terms = (*source)->terms();
    ASSERT_TRUE(source_terms.has_value());
    auto source_term = (*source_terms)->next();
    ASSERT_TRUE(source_term && source_term->has_value());
    EXPECT_EQ(length, (**source_term).length);
    auto last = (**source_term).bytes(length - 1, 1);
    ASSERT_TRUE(last.has_value());
    EXPECT_EQ("z", *last);
    auto postings = (**source_term).reader->postings();
    ASSERT_TRUE(postings.has_value());
    auto doc = (*postings)->next();
    ASSERT_TRUE(doc.has_value());
    EXPECT_EQ(0, *doc);
    auto entry = (*reader)->term_at(0);
    ASSERT_TRUE(entry);
    EXPECT_EQ(length, entry->term_size());
    for (uint64_t offset = 0; offset < length;) {
      const size_t count = std::min<uint64_t>(4096, length - offset);
      auto bytes = entry->read_term(offset, count);
      ASSERT_TRUE(bytes);
      std::string expected(count, 'q');
      if (offset + count == length) {
        expected.back() = 'z';
      }
      EXPECT_EQ(expected, *bytes);
      offset += count;
    }
  }
  EXPECT_GT(largest_request, 0);
  EXPECT_LE(largest_request, format::kDictionaryBlockBytes);
}

TEST_F(FtsFieldTest, ReopenedReaderHasConservativeRuntimeScoreBounds) {
  build({{"common", 128, 1}, {"rare", 1, 2}});
  for (bool mmap : {false, true}) {
    auto file = FtsContainerReader::Open(path_, mmap);
    ASSERT_TRUE(file);
    for (float k1 : {0.0f, 1.2f, 32.0f, 1.0e12f}) {
      for (float b : {0.0f, 0.75f, 1.0f}) {
        auto reader = FtsFieldReader::Open(*file, 0, {k1, b});
        ASSERT_TRUE(reader);
        for (const char *name : {"common", "rare"}) {
          auto term = (*reader)->lookup(name);
          ASSERT_TRUE(term);
          const float bound = (*reader)->max_score(*term);
          ASSERT_TRUE(std::isfinite(bound));
          EXPECT_TRUE(std::isinf((*reader)->max_score(*term, 3)));
          uint32_t random = 0x12345678;
          for (size_t i = 0; i < 2048; ++i) {
            random = random * 1664525u + 1013904223u;
            for (uint32_t tf : {1u, 2u, 3u, std::max(1u, random), UINT32_MAX}) {
              for (uint32_t length : {tf, UINT32_MAX}) {
                EXPECT_GE(bound,
                          (*reader)->score(term->metadata().document_frequency,
                                           tf, length));
              }
            }
          }
          auto posting = (*reader)->postings(*term);
          ASSERT_TRUE(posting);
          auto block = (*posting)->bound(0);
          ASSERT_TRUE(block);
          EXPECT_EQ(bound, block->score);
        }
      }
    }
  }
}

TEST_F(FtsFieldTest, RuntimeBoundsRejectUnsupportedArithmetic) {
  build({{"a", 1, 1}});
  auto file = FtsContainerReader::Open(path_, true);
  ASSERT_TRUE(file);
  auto reader = FtsFieldReader::Open(*file, 0);
  ASSERT_TRUE(reader);
  auto term = (*reader)->lookup("a");
  ASSERT_TRUE(term);
  const int rounding = std::fegetround();
  ASSERT_EQ(0, std::fesetround(FE_DOWNWARD));
  const float bound = (*reader)->max_score(*term);
  EXPECT_EQ(0, std::fesetround(rounding));
  EXPECT_TRUE(std::isinf(bound));

  auto overflow = FtsFieldReader::Open(
      *file, 0, {std::numeric_limits<float>::max(), 0.75f});
  ASSERT_TRUE(overflow);
  auto other = (*overflow)->lookup("a");
  ASSERT_TRUE(other);
  EXPECT_TRUE(std::isinf((*overflow)->max_score(*other)));
}

TEST_F(FtsFieldTest, PostingThresholdsMultifieldAndScoreBounds) {
  const std::vector<TermInput> input{{"a", 1, 1},   {"b", 1, 129},
                                     {"c", 2, 1},   {"d", 127, 1},
                                     {"e", 128, 2}, {"f", 129, 1}};
  build(input, 2);
  for (bool mmap : {false, true}) {
    auto file = FtsContainerReader::Open(path_, mmap);
    ASSERT_TRUE(file);
    auto reader = FtsFieldReader::Open(*file, 0);
    ASSERT_TRUE(reader);
    auto a = (*reader)->lookup("a");
    ASSERT_TRUE(a);
    EXPECT_TRUE(std::isfinite((*reader)->max_score(*a)));
    ASSERT_TRUE((*reader)->verify().ok());
    EXPECT_EQ((*reader)->metadata().scoring_document_count, 260u);
    EXPECT_EQ((*reader)->metadata().present_count, 130u);
    BM25Scorer oracle;
    oracle.update_stats(260, (*reader)->metadata().total_tokens);
    for (const auto &term : input) {
      auto found = (*reader)->lookup(term.text);
      ASSERT_TRUE(found);
      EXPECT_EQ(found->metadata().document_frequency, term.df);
      EXPECT_EQ(found->metadata().total_frequency,
                static_cast<uint64_t>(term.df) * term.tf);
      auto cursor = (*reader)->postings(*found);
      ASSERT_TRUE(cursor);
      float maximum = 0;
      for (uint32_t i = 0; i < term.df; ++i) {
        auto posting = (*cursor)->next();
        ASSERT_TRUE(posting);
        ASSERT_TRUE(posting->has_value());
        EXPECT_EQ((**posting).doc, i * 2);
        EXPECT_EQ((**posting).tf, term.tf);
        auto length = (*cursor)->doc_length();
        ASSERT_TRUE(length);
        EXPECT_EQ(*length, lengths_[i * 2]);
        const float expected = oracle.score(term.df, term.tf, lengths_[i * 2]);
        EXPECT_FLOAT_EQ((*reader)->score(term.df, term.tf, *length), expected);
        EXPECT_GE((*cursor)->block_max_score(), expected);
        EXPECT_TRUE(std::isinf((*cursor)->block_max_score(3)));
        maximum = std::max(maximum, expected);
      }
      EXPECT_FLOAT_EQ((*reader)->max_score(*found), maximum);
      EXPECT_TRUE(std::isinf((*reader)->max_score(*found, 3)));
      auto end = (*cursor)->advance(UINT32_MAX);
      ASSERT_TRUE(end);
      EXPECT_FALSE(end->has_value());
    }
    auto other = FtsFieldReader::Open(*file, 1);
    ASSERT_TRUE(other);
    ASSERT_TRUE((*other)->verify().ok());
    EXPECT_EQ((*other)->postings(*a).error().code(),
              StatusCode::INVALID_ARGUMENT);
    auto different = FtsFieldReader::Open(*file, 0, BM25Params{2.0f, 0.5f});
    ASSERT_TRUE(different);
    ASSERT_TRUE((*different)->verify().ok());
    auto different_term = (*different)->lookup("a");
    ASSERT_TRUE(different_term);
    EXPECT_GE((*different)->max_score(*different_term),
              (*different)->score(1, 1, lengths_[0]));
  }
}

TEST_F(FtsFieldTest, CustomTermQueryPreservesScoresAndPinsGeneration) {
  build({{"a", 129, 2}, {"b", 1, 1}});
  for (bool mmap : {false, true}) {
    auto file = FtsContainerReader::Open(path_, mmap);
    ASSERT_TRUE(file);
    auto field = FtsFieldReader::Open(*file, 0);
    ASSERT_TRUE(field);
    ASSERT_TRUE((*field)->verify().ok());
    FtsColumnIndexer query;
    ASSERT_TRUE(query.open_reader(*field));
    field->reset();
    file->reset();
    FtsQueryParams params;
    params.topk = 200;
    auto results = query.search(TermNode("a"), params);
    ASSERT_TRUE(results);
    ASSERT_EQ(results->size(), 129u);
    BM25Scorer oracle;
    oracle.update_stats(260, 259);
    std::set<uint64_t> ids;
    for (const auto &result : *results) {
      EXPECT_EQ(result.doc_id % 2, 0u);
      EXPECT_LE(result.doc_id, 256u);
      EXPECT_FLOAT_EQ(result.score,
                      oracle.score(129, 2, result.doc_id == 0 ? 3 : 2));
      ids.insert(result.doc_id);
    }
    EXPECT_EQ(ids.size(), 129u);
    auto absent = query.search(TermNode("absent"), params);
    ASSERT_TRUE(absent);
    EXPECT_TRUE(absent->empty());
  }
}

TEST_F(FtsFieldTest, PostingBatchesPreserveRowsPayloadsAndCursorOwnership) {
  build({{"a", 513, 2}, {"b", 129, 3}, {"c", 1, 1}, {"d", 31, 1}}, 1, 1028);
  for (bool mmap : {false, true}) {
    auto file = FtsContainerReader::Open(path_, mmap);
    ASSERT_TRUE(file);
    auto field = FtsFieldReader::Open(*file, 0);
    ASSERT_TRUE(field);
    auto reader = OpenCustomFtsReader(*field);
    ASSERT_TRUE(reader);
    auto term = (*reader)->lookup("a");
    ASSERT_TRUE(term && *term);
    auto first = (*term)->postings();
    auto second = (*term)->postings();
    ASSERT_TRUE(first && second);
    EXPECT_EQ(0u, (*first)->next().value());
    auto batch = (*first)->next_batch();
    ASSERT_TRUE(batch) << batch.error();
    ASSERT_GT(batch->count, 0u);
    ASSERT_LE(batch->count, FtsPostingBatch::kCapacity);
    EXPECT_EQ(2u, batch->docs[0]);
    ASSERT_TRUE((*second)->advance(1000));
    ASSERT_TRUE((*second)->next_batch());
    // The second cursor must not invalidate the first cursor's borrowed view.
    for (uint32_t i = 0; i < batch->count; ++i) {
      EXPECT_EQ((i + 1) * 2, batch->docs[i]);
      EXPECT_EQ(2u, batch->tfs[i]);
      EXPECT_EQ(lengths_[batch->docs[i]], batch->lengths[i]);
    }
    EXPECT_EQ(258u, (*first)->advance(257).value());
    reader->reset();
    field->reset();
    file->reset();
    term->reset();
    second->reset();
    uint32_t expected = 260;
    while (true) {
      batch = (*first)->next_batch();
      ASSERT_TRUE(batch) << batch.error();
      ASSERT_LE(batch->count, FtsPostingBatch::kCapacity);
      if (batch->count == 0) {
        break;
      }
      for (uint32_t i = 0; i < batch->count; ++i) {
        EXPECT_EQ(expected, batch->docs[i]);
        EXPECT_EQ(2u, batch->tfs[i]);
        EXPECT_EQ(lengths_[expected], batch->lengths[i]);
        expected += 2;
      }
      auto payload = (*first)->payload();
      ASSERT_TRUE(payload);
      EXPECT_EQ(2u, payload->tf);
      EXPECT_EQ(lengths_[expected - 2], payload->length);
    }
    EXPECT_EQ(1026u, expected);
    EXPECT_EQ(UINT32_MAX, (*first)->next().value());
    EXPECT_EQ(0u, (*first)->next_batch()->count);
  }
}

TEST_F(FtsFieldTest, PostingBatchesCoverInlineShortAndTailPostings) {
  build({{"a", 1, 1}, {"b", 31, 2}, {"c", 128, 1}, {"d", 129, 3}});
  for (bool mmap : {false, true}) {
    auto file = FtsContainerReader::Open(path_, mmap);
    ASSERT_TRUE(file);
    auto field = FtsFieldReader::Open(*file, 0);
    ASSERT_TRUE(field);
    auto reader = OpenCustomFtsReader(*field);
    ASSERT_TRUE(reader);
    for (const auto &input : std::vector<TermInput>{
             {"a", 1, 1}, {"b", 31, 2}, {"c", 128, 1}, {"d", 129, 3}}) {
      auto term = (*reader)->lookup(input.text);
      ASSERT_TRUE(term && *term);
      auto posting = (*term)->postings();
      ASSERT_TRUE(posting);
      uint32_t count = 0;
      while (true) {
        auto batch = (*posting)->next_batch();
        ASSERT_TRUE(batch);
        ASSERT_LE(batch->count, FtsPostingBatch::kCapacity);
        if (batch->count == 0) {
          break;
        }
        for (uint32_t i = 0; i < batch->count; ++i) {
          EXPECT_EQ(count * 2, batch->docs[i]);
          EXPECT_EQ(input.tf, batch->tfs[i]);
          EXPECT_EQ(lengths_[count * 2], batch->lengths[i]);
          ++count;
        }
      }
      EXPECT_EQ(input.df, count);
    }
  }
}

TEST_F(FtsFieldTest, BatchedAdvanceIncludesZeroScoresDespiteThreshold) {
  build({{"a", 129, 1}, {"b", 1, 1}});
  for (bool mmap : {false, true}) {
    auto file = FtsContainerReader::Open(path_, mmap);
    ASSERT_TRUE(file);
    auto field = FtsFieldReader::Open(*file, 0);
    ASSERT_TRUE(field);
    auto reader = OpenCustomFtsReader(*field);
    ASSERT_TRUE(reader);
    auto a = (*reader)->lookup("a");
    auto b = (*reader)->lookup("b");
    ASSERT_TRUE(a && *a && b && *b);
    auto iterator =
        CreateBatchDisjunction({{*a, 0}, {*b, 0}}, (*reader)->scorer());
    ASSERT_TRUE(iterator && *iterator);
    EXPECT_EQ(0u, (*iterator)->next_doc());
    EXPECT_EQ(0.0f, (*iterator)->score());
    (*iterator)->set_min_competitive_score(
        std::numeric_limits<float>::infinity());
    // Competitive next() may skip zero scores, but advance() is positioning
    // and must retain every logical match, including equal/current targets.
    EXPECT_EQ(2u, (*iterator)->advance(1));
    EXPECT_EQ(2u, (*iterator)->advance(2));
    EXPECT_EQ(256u, (*iterator)->advance(255));
    EXPECT_EQ(0.0f, (*iterator)->score());
    EXPECT_EQ(DocIterator::NO_MORE_DOCS, (*iterator)->advance(UINT32_MAX));
    EXPECT_TRUE((*iterator)->status().ok());
  }
}

TEST_F(FtsFieldTest, BatchQueriesPropagateLateLengthErrors) {
  build({{"a", 4097, 1}, {"b", 1, 1}}, 1, 8196);
  rewrite("field.0.doclen", [](std::string *bytes) {
    // The last matching row is in a later scoring window. Preserve the
    // container checksum and block shape but make its exact length < TF.
    const size_t offset = 64 * (4 + 16 + 128 * 4) + 4 + 1;
    ASSERT_GE(bytes->size(), offset + 4);
    std::fill_n(bytes->begin() + offset, 4, '\0');
  });
  for (bool mmap : {false, true}) {
    auto file = FtsContainerReader::Open(path_, mmap);
    ASSERT_TRUE(file);
    auto field = FtsFieldReader::Open(*file, 0);
    ASSERT_TRUE(field);
    auto reader = OpenCustomFtsReader(*field);
    ASSERT_TRUE(reader);
    auto term = (*reader)->lookup("a");
    ASSERT_TRUE(term && *term);
    auto posting = (*term)->postings();
    ASSERT_TRUE(posting);
    size_t count = 0;
    while (true) {
      auto batch = (*posting)->next_batch();
      if (!batch) {
        EXPECT_EQ(StatusCode::INTERNAL_ERROR, batch.error().code());
        break;
      }
      ASSERT_NE(0u, batch->count);  // Corruption cannot masquerade as EOF.
      count += batch->count;
    }
    EXPECT_EQ(4096u, count);
    EXPECT_EQ(StatusCode::INTERNAL_ERROR,
              (*posting)->next_batch().error().code());
    EXPECT_EQ(StatusCode::INTERNAL_ERROR, (*posting)->next().error().code());
    FtsColumnIndexer query;
    ASSERT_TRUE(query.open_reader(*field));
    OrNode ast;
    ast.children.push_back(std::make_unique<TermNode>("a"));
    ast.children.push_back(std::make_unique<TermNode>("b"));
    FtsQueryParams params;
    params.topk = 1;
    auto result = query.search(ast, params);
    ASSERT_FALSE(result);
    EXPECT_EQ(StatusCode::INTERNAL_ERROR, result.error().code());
  }
}

TEST_F(FtsFieldTest, DocLengthViewsSurviveIndependentCursorPageChanges) {
  build({{"a", 2048, 2}, {"b", 65, 3}}, 1, 4097);
  for (bool mmap : {false, true}) {
    auto file = FtsContainerReader::Open(path_, mmap);
    ASSERT_TRUE(file);
    auto field = FtsFieldReader::Open(*file, 0);
    ASSERT_TRUE(field);
    auto term = (*field)->lookup("a");
    ASSERT_TRUE(term);
    auto first = (*field)->postings(*term);
    auto second = (*field)->postings(*term);
    ASSERT_TRUE(first);
    ASSERT_TRUE(second);
    auto missing = (*field)->document(4095);
    auto empty = (*field)->document(4096);
    ASSERT_TRUE(missing);
    ASSERT_TRUE(empty);
    EXPECT_FALSE(missing->present);
    EXPECT_TRUE(empty->present);
    EXPECT_EQ(0u, missing->length);
    EXPECT_EQ(0u, empty->length);
    // Each cursor must keep its own length page alive after parent release.
    field->reset();
    file->reset();
    ASSERT_TRUE((*first)->advance(0));
    ASSERT_EQ(5u, (*first)->doc_length().value());
    ASSERT_TRUE((*second)->advance(4094));
    EXPECT_EQ(2u, (*second)->doc_length().value());
    EXPECT_EQ(5u, (*first)->doc_length().value());
    second->reset();
    for (uint32_t row : {128u, 130u, 1024u, 4094u}) {
      auto posting = (*first)->advance(row);
      ASSERT_TRUE(posting && posting->has_value());
      EXPECT_EQ(row, (**posting).doc);
      auto length = (*first)->doc_length();
      ASSERT_TRUE(length);
      EXPECT_EQ(row == 128 ? 5u : 2u, *length);
      EXPECT_EQ(*length, (*first)->doc_length().value());
    }
  }
}

TEST_F(FtsFieldTest, NearbyAdvanceDoesNotReadDistantSkipPages) {
  build({{"a", 32768, 1}}, 1, 65536);
  for (bool mmap : {false, true}) {
    auto file = FtsContainerReader::Open(path_, mmap);
    ASSERT_TRUE(file);
    auto reader = FtsFieldReader::Open(*file, 0);
    ASSERT_TRUE(reader);
    auto term = (*reader)->lookup("a");
    ASSERT_TRUE(term);
    auto cursor = (*reader)->postings(*term);
    ASSERT_TRUE(cursor);
    ASSERT_TRUE((*cursor)->next());
    const auto before = (*file)->read_stats();
    // The adjacent posting/skip blocks are in the already pinned pages.
    // Searching the midpoint of the entire remaining list needlessly reads
    // distant pages even though the target is in the very next block.
    auto adjacent = (*cursor)->advance(256);
    ASSERT_TRUE(adjacent);
    ASSERT_TRUE(adjacent->has_value());
    EXPECT_EQ(256u, (**adjacent).doc);
    EXPECT_EQ(before.range_reads, (*file)->read_stats().range_reads);
    auto distant = (*cursor)->advance(60001);
    ASSERT_TRUE(distant);
    ASSERT_TRUE(distant->has_value());
    EXPECT_EQ(60002u, (**distant).doc);
    auto end = (*cursor)->advance(65535);
    ASSERT_TRUE(end);
    EXPECT_FALSE(end->has_value());
  }
}

TEST_F(FtsFieldTest, TargetBlockBoundsDoNotAdvancePostingCursor) {
  build({{"a", 129, 1}});
  for (bool mmap : {false, true}) {
    auto file = FtsContainerReader::Open(path_, mmap);
    ASSERT_TRUE(file);
    auto field = FtsFieldReader::Open(*file, 0);
    ASSERT_TRUE(field);
    ASSERT_TRUE((*field)->verify().ok());
    auto reader = OpenCustomFtsReader(*field);
    ASSERT_TRUE(reader);
    auto term = (*reader)->lookup("a");
    ASSERT_TRUE(term);
    ASSERT_TRUE(*term);
    auto posting = (*term)->postings();
    ASSERT_TRUE(posting);
    auto first = (*posting)->bound(0, 1);
    ASSERT_TRUE(first);
    EXPECT_EQ(254u, first->last_doc);
    EXPECT_GE(first->score, (*field)->score(129, 1, 1));
    ASSERT_EQ(0u, (*posting)->next().value());
    auto tail = (*posting)->bound(255, 1);
    ASSERT_TRUE(tail);
    EXPECT_EQ(256u, tail->last_doc);
    EXPECT_GE(tail->score, (*field)->score(129, 1, 1));
    EXPECT_EQ(2u, (*posting)->next().value());
    auto past_end = (*posting)->bound(UINT32_MAX, 1);
    ASSERT_TRUE(past_end);
    EXPECT_EQ(0, past_end->score);
    EXPECT_EQ(UINT32_MAX, past_end->last_doc);
    auto boosted = (*posting)->bound(0, 3);
    ASSERT_TRUE(boosted);
    EXPECT_TRUE(std::isinf(boosted->score));
    EXPECT_EQ(4u, (*posting)->next().value());
  }
}

TEST_F(FtsFieldTest, MillionPositionsUseBoundedCursorAndPreserveGeneration) {
  build({{"long", 1, 1000003}});
  ASSERT_EQ(0, ailego::MemoryLimitPool::get_instance().init(0));
  for (bool mmap : {false, true}) {
    auto file = FtsContainerReader::Open(path_, mmap, 512 * 1024);
    ASSERT_TRUE(file);
    auto reader = FtsFieldReader::Open(*file, 0);
    ASSERT_TRUE(reader);
    ASSERT_TRUE((*reader)->verify().ok());
    auto term = (*reader)->lookup("long");
    ASSERT_TRUE(term);
    auto cursor = (*reader)->postings(*term);
    ASSERT_TRUE(cursor);
    auto first = (*cursor)->next();
    ASSERT_TRUE(first);
    auto positions = (*cursor)->positions();
    ASSERT_TRUE(positions);
    cursor->reset();
    reader->reset();
    file->reset();
    auto zero = (*positions)->next();
    ASSERT_TRUE(zero);
    ASSERT_TRUE(zero->has_value());
    EXPECT_EQ(**zero, 0u);
    auto duplicate = (*positions)->next();
    ASSERT_TRUE(duplicate);
    ASSERT_TRUE(duplicate->has_value());
    EXPECT_EQ(**duplicate, 0u);
    auto tail = (*positions)->seek_ge(500000);
    ASSERT_TRUE(tail);
    ASSERT_TRUE(tail->has_value());
    EXPECT_EQ(**tail, 500000u);
    auto end = (*positions)->seek_ge(UINT32_MAX);
    ASSERT_TRUE(end);
    EXPECT_FALSE(end->has_value());
  }
}

TEST_F(FtsFieldTest, CorruptPositionIndexIsAnErrorNotEof) {
  build({{"a", 2, 1}});
  rewrite("field.0.position_index", [](std::string *bytes) {
    ASSERT_GT(bytes->size(), 48u);
    (*bytes)[48] = 0;
  });
  for (bool mmap : {false, true}) {
    auto file = FtsContainerReader::Open(path_, mmap);
    ASSERT_TRUE(file);
    EXPECT_TRUE((*file)->verify_all().ok());
    auto reader = FtsFieldReader::Open(*file, 0);
    ASSERT_TRUE(reader);
    auto term = (*reader)->lookup("a");
    ASSERT_TRUE(term);
    auto cursor = (*reader)->postings(*term);
    ASSERT_TRUE(cursor);
    ASSERT_TRUE((*cursor)->next());
    EXPECT_EQ((*cursor)->positions().error().code(),
              StatusCode::INTERNAL_ERROR);
    EXPECT_EQ((*cursor)->next().error().code(), StatusCode::INTERNAL_ERROR);
    EXPECT_EQ((*reader)->verify().code(), StatusCode::INTERNAL_ERROR);
    FtsColumnIndexer query_reader;
    ASSERT_TRUE(query_reader.open_reader(*reader));
    PhraseNode phrase;
    phrase.terms = {"a", "a"};
    FtsQueryParams params;
    params.topk = 10;
    auto result = query_reader.search(phrase, params);
    ASSERT_FALSE(result);
    EXPECT_EQ(StatusCode::INTERNAL_ERROR, result.error().code());
  }
}

TEST_F(FtsFieldTest, RepeatedMillionFrequencyPhraseUsesBoundedCursors) {
  build({{"long", 1, 1000003}});
  ASSERT_EQ(0, ailego::MemoryLimitPool::get_instance().init(0));
  for (bool mmap : {false, true}) {
    auto file = FtsContainerReader::Open(path_, mmap, 1024 * 1024);
    ASSERT_TRUE(file);
    auto field = FtsFieldReader::Open(*file, 0);
    ASSERT_TRUE(field);
    FtsColumnIndexer reader;
    ASSERT_TRUE(reader.open_reader(*field));
    PhraseNode phrase;
    phrase.terms = {"long", "long", "long"};
    FtsQueryParams params;
    params.topk = 10;
    auto result = reader.search(phrase, params);
    ASSERT_TRUE(result) << result.error();
    ASSERT_EQ(1u, result->size());
    EXPECT_EQ(0u, (*result)[0].doc_id);
  }
}

TEST_F(FtsFieldTest, QueryLimitsReturnResourceErrors) {
  build({{"a", 2, 1}});
  for (bool mmap : {false, true}) {
    auto file = FtsContainerReader::Open(path_, mmap);
    ASSERT_TRUE(file);
    auto field = FtsFieldReader::Open(*file, 0);
    ASSERT_TRUE(field);
    FtsColumnIndexer reader;
    ASSERT_TRUE(reader.open_reader(*field));
    FtsQueryParams params;
    params.topk = 10;
    PhraseNode phrase;
    phrase.terms.assign(1025, "a");
    auto terms = reader.search(phrase, params);
    ASSERT_FALSE(terms);
    EXPECT_EQ(StatusCode::RESOURCE_EXHAUSTED, terms.error().code());
    auto bytes =
        reader.search(TermNode(std::string(1024 * 1024 + 1, 'a')), params);
    ASSERT_FALSE(bytes);
    EXPECT_EQ(StatusCode::RESOURCE_EXHAUSTED, bytes.error().code());
    FtsAstNodePtr tree = std::make_unique<TermNode>("a");
    for (size_t depth = 0; depth < 130; ++depth) {
      auto parent = std::make_unique<AndNode>();
      parent->children.push_back(std::move(tree));
      tree = std::move(parent);
    }
    auto depth = reader.search(*tree, params);
    ASSERT_FALSE(depth);
    EXPECT_EQ(StatusCode::RESOURCE_EXHAUSTED, depth.error().code());
  }
}

TEST_F(FtsFieldTest, PrefetchErrorsSurviveEmptyConjunctionAndCandidates) {
  build({{"a", 2, 1}, {"b", 1, 1}});
  rewrite("field.0.postings", [](std::string *bytes) {
    ASSERT_FALSE(bytes->empty());
    (*bytes)[0] = 0;  // Invalid count in the first varint block.
  });
  for (bool mmap : {false, true}) {
    auto file = FtsContainerReader::Open(path_, mmap);
    ASSERT_TRUE(file);
    auto field = FtsFieldReader::Open(*file, 0);
    ASSERT_TRUE(field);
    FtsColumnIndexer reader;
    ASSERT_TRUE(reader.open_reader(*field));
    for (bool missing : {false, true}) {
      for (bool empty_candidates : {false, true}) {
        auto either = std::make_unique<OrNode>();
        either->children.push_back(std::make_unique<TermNode>("a"));
        either->children.push_back(std::make_unique<TermNode>("b"));
        AndNode query;
        query.children.push_back(std::move(either));
        if (missing) {
          query.children.push_back(std::make_unique<TermNode>("missing"));
        }
        FtsQueryParams params;
        params.topk = 10;
        if (empty_candidates) {
          params.candidate_ids.emplace();
        }
        auto result = reader.search(query, params);
        ASSERT_FALSE(result);
        EXPECT_EQ(StatusCode::INTERNAL_ERROR, result.error().code());
      }
    }
  }
}

TEST_F(FtsFieldTest, LatePostingFailureDoesNotReturnPartialTopK) {
  build({{"a", 129, 1}});
  rewrite("field.0.postings", [](std::string *bytes) {
    ASSERT_GE(bytes->size(), 8u);
    // Final block contains one TF=1 posting, hence only its eight-byte header.
    (*bytes)[bytes->size() - 2] = 0;
  });
  for (bool mmap : {false, true}) {
    auto file = FtsContainerReader::Open(path_, mmap);
    ASSERT_TRUE(file);
    auto field = FtsFieldReader::Open(*file, 0);
    ASSERT_TRUE(field);
    FtsColumnIndexer reader;
    ASSERT_TRUE(reader.open_reader(*field));
    FtsQueryParams params;
    params.topk = 7;
    auto result = reader.search(TermNode("a"), params);
    ASSERT_FALSE(result);
    EXPECT_EQ(StatusCode::INTERNAL_ERROR, result.error().code());
  }
}

TEST_F(FtsFieldTest, InsufficientRangeBudgetReturnsResourceError) {
  build({{"a", 1, 129}});
  for (bool mmap : {false, true}) {
    const size_t one = 4 * std::max<size_t>(ailego::kVectorPageSize, 4128);
    auto file = FtsContainerReader::Open(path_, mmap, one);
    ASSERT_TRUE(file);
    auto reader = FtsFieldReader::Open(*file, 0);
    ASSERT_TRUE(reader);
    auto term = (*reader)->lookup("a");
    // Dictionary lookup also needs an index page and a dictionary block.
    ASSERT_FALSE(term);
    EXPECT_EQ(term.error().code(), StatusCode::RESOURCE_EXHAUSTED);
    FtsColumnIndexer query_reader;
    ASSERT_TRUE(query_reader.open_reader(*reader));
    FtsQueryParams params;
    params.topk = 10;
    auto result = query_reader.search(TermNode("a"), params);
    ASSERT_FALSE(result);
    EXPECT_EQ(StatusCode::RESOURCE_EXHAUSTED, result.error().code());
  }
}

TEST_F(FtsFieldTest, ConcurrentReadersUseIndependentCursors) {
  build({{"a", 129, 2}, {"b", 127, 1}});
  for (bool mmap : {false, true}) {
    auto file = FtsContainerReader::Open(path_, mmap, 8 * 1024 * 1024);
    ASSERT_TRUE(file);
    auto reader = FtsFieldReader::Open(*file, 0);
    ASSERT_TRUE(reader);
    ASSERT_TRUE((*reader)->verify().ok());
    std::atomic<bool> success{true};
    std::vector<std::thread> workers;
    for (uint32_t worker = 0; worker < 8; ++worker) {
      workers.emplace_back([&, worker] {
        for (uint32_t i = 0; i < 50; ++i) {
          auto term = (*reader)->lookup("a");
          if (!term) {
            success = false;
            return;
          }
          auto cursor = (*reader)->postings(*term);
          if (!cursor) {
            success = false;
            return;
          }
          const uint32_t target = (i + worker) * 2 + 1;
          auto posting = (*cursor)->advance(target);
          if (!posting || !posting->has_value() ||
              (**posting).doc != target + 1) {
            success = false;
            return;
          }
          auto positions = (*cursor)->positions();
          if (!positions) {
            success = false;
            return;
          }
          auto value = (*positions)->next();
          if (!value || !value->has_value() || **value != 0) {
            success = false;
            return;
          }
        }
      });
    }
    for (auto &worker : workers) {
      worker.join();
    }
    EXPECT_TRUE(success.load());
  }
}

TEST_F(FtsFieldTest, EmptyFieldRetainsMissingAndPresentZeroRows) {
  build({});
  for (bool mmap : {false, true}) {
    auto file = FtsContainerReader::Open(path_, mmap);
    ASSERT_TRUE(file);
    auto reader = FtsFieldReader::Open(*file, 0);
    ASSERT_TRUE(reader);
    ASSERT_TRUE((*reader)->verify().ok());
    EXPECT_EQ((*reader)->metadata().term_count, 0u);
    EXPECT_EQ((*reader)->metadata().present_count, 1u);
    EXPECT_EQ((*reader)->metadata().scoring_document_count, 260u);
    auto missing = (*reader)->document(0);
    auto empty = (*reader)->document(259);
    ASSERT_TRUE(missing);
    ASSERT_TRUE(empty);
    EXPECT_FALSE(missing->present);
    EXPECT_TRUE(empty->present);
    EXPECT_EQ(empty->length, 0u);
    EXPECT_EQ((*reader)->lookup("absent").error().code(),
              StatusCode::NOT_FOUND);
  }
}

TEST_F(FtsFieldTest, UnderestimatedPersistedBoundsUseRuntimeFallback) {
  build({{"a", 128, 1}});
  rewrite("field.0.skips", [](std::string *bytes) {
    ASSERT_EQ(bytes->size(), format::kPostingSkipBytes);
    // A finite but unsafe bound with a valid container checksum must not
    // become a competitive pruning threshold on the current scoring path.
    bytes->replace(8, 4, 4, '\0');
  });
  for (bool mmap : {false, true}) {
    auto file = FtsContainerReader::Open(path_, mmap);
    ASSERT_TRUE(file);
    auto reader = FtsFieldReader::Open(*file, 0);
    ASSERT_TRUE(reader);
    ASSERT_TRUE((*reader)->verify().ok());
    auto term = (*reader)->lookup("a");
    ASSERT_TRUE(term);
    const float fallback = (*reader)->max_score(*term);
    EXPECT_TRUE(std::isfinite(fallback));
    EXPECT_GE(fallback, (*reader)->score(128, 1, 1));
    auto cursor = (*reader)->postings(*term);
    ASSERT_TRUE(cursor);
    ASSERT_TRUE((*cursor)->next());
    EXPECT_EQ(fallback, (*cursor)->block_max_score());
  }
}

TEST_F(FtsFieldTest, PositionSubblockBoundaryAndMalformedDirectory) {
  build({{"a", 1, 128}, {"b", 1, 129}});
  for (bool mmap : {false, true}) {
    auto file = FtsContainerReader::Open(path_, mmap);
    ASSERT_TRUE(file);
    auto reader = FtsFieldReader::Open(*file, 0);
    ASSERT_TRUE(reader);
    ASSERT_TRUE((*reader)->verify().ok());
    auto term = (*reader)->lookup("a");
    ASSERT_TRUE(term);
    auto cursor = (*reader)->postings(*term);
    ASSERT_TRUE(cursor);
    ASSERT_TRUE((*cursor)->next());
    auto positions = (*cursor)->positions();
    ASSERT_TRUE(positions);
    auto last = (*positions)->seek_ge(63);
    ASSERT_TRUE(last);
    ASSERT_TRUE(last->has_value());
    EXPECT_EQ(**last, 63u);
    auto duplicate = (*positions)->next();
    ASSERT_TRUE(duplicate);
    ASSERT_TRUE(duplicate->has_value());
    EXPECT_EQ(**duplicate, 63u);
    auto eof = (*positions)->next();
    ASSERT_TRUE(eof);
    EXPECT_FALSE(eof->has_value());
  }
  rewrite("field.0.position_index", [](std::string *bytes) {
    ASSERT_EQ(bytes->size(), 64u);
    // First long-position subblock claims a count inconsistent with TF.
    bytes->replace(8, 4, 4, '\0');
  });
  for (bool mmap : {false, true}) {
    auto file = FtsContainerReader::Open(path_, mmap);
    ASSERT_TRUE(file);
    auto reader = FtsFieldReader::Open(*file, 0);
    ASSERT_TRUE(reader);
    EXPECT_EQ((*reader)->verify().code(), StatusCode::INTERNAL_ERROR);
  }
}

TEST(FtsFieldWriterTest, SourceErrorsAndInconsistentTotalsPreventFinish) {
  format::FieldMetadata field;
  field.present_count = field.scoring_document_count = 1;
  field.total_tokens = 2;
  auto writer = FtsFieldWriter::Create(1, field);
  ASSERT_TRUE(writer);
  ASSERT_TRUE((*writer)->add_document(0, true, 1).ok());
  EXPECT_EQ((*writer)->finish_documents().code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ((*writer)->begin_term("a").code(), StatusCode::INVALID_ARGUMENT);
  writer = FtsFieldWriter::Create(1, field);
  ASSERT_TRUE(writer);
  ASSERT_TRUE((*writer)->add_document(0, true, 2).ok());
  ASSERT_TRUE((*writer)->finish_documents().ok());
  ASSERT_TRUE((*writer)->begin_term("a").ok());
  EXPECT_EQ((*writer)
                ->add_posting(0, 2,
                              []() -> Result<std::optional<uint32_t>> {
                                return tl::make_unexpected(Status(
                                    StatusCode::UNAVAILABLE, "source failed"));
                              })
                .code(),
            StatusCode::UNAVAILABLE);
  EXPECT_EQ((*writer)->end_term().code(), StatusCode::UNAVAILABLE);
}

}  // namespace
}  // namespace zvec::fts
