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

#include <atomic>
#include <filesystem>
#include <thread>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_factory.h>
#include "db/common/file_helper.h"
#include "db/index/column/fts_column/fts_indexer.h"
#include "db/index/column/fts_column/fts_merge.h"
#include "db/index/column/fts_column/fts_source.h"

namespace zvec {
namespace {

class FtsGenerationTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(0,
              ailego::MemoryLimitPool::get_instance().init(16 * 1024 * 1024));
    FileHelper::RemoveDirectory(path_);
    ASSERT_TRUE(FileHelper::CreateDirectory(path_));
    auto params = std::make_shared<FtsIndexParams>("whitespace");
    for (const char *name : {"title", "body"}) {
      fields_.push_back(
          std::make_shared<FieldSchema>(name, DataType::STRING, true, params));
    }
    source_ = FtsIndexer::CreateAndOpen(path_ + "/mutable", fields_, true);
    ASSERT_NE(nullptr, source_);
    ASSERT_TRUE(source_->insert("title", 0, "alpha alpha beta").ok());
    ASSERT_TRUE(source_->insert("title", 2, "beta").ok());
    ASSERT_TRUE(source_->insert("body", 0, "beta").ok());
    ASSERT_TRUE(source_->insert("body", 1, "").ok());
    ASSERT_TRUE(source_->insert("body", 2, "alpha beta").ok());
    ASSERT_TRUE(source_->flush().ok());
  }

  void TearDown() override {
    source_.reset();
    FileHelper::RemoveDirectory(path_);
  }

  const std::string path_{"fts_generation.fixture"};
  FieldSchemaPtrList fields_;
  FtsIndexer::Ptr source_;

  FtsIndexer::CustomTarget target(bool mmap = false, uint32_t rows = 3) {
    return {
        path_,
        3,
        {7, BlockType::FTS_INDEX, 100, 100 + rows - 1, rows, {"title", "body"}},
        mmap};
  }
};

TEST_F(FtsGenerationTest,
       DirectSealPreservesFieldsScoresAndSourceAcrossReopen) {
  auto retirement = fts::FtsRetirementQueue::Create();
  for (bool mmap : {false, true}) {
    FtsIndexer::CustomTarget target{
        path_,
        3,
        {7, BlockType::FTS_INDEX, 100, 102, 3, {"title", "body"}},
        mmap};
    auto prepared = source_->prepare_custom(target, fields_, retirement);
    ASSERT_TRUE(prepared.has_value()) << prepared.error().message();
    ASSERT_TRUE((*prepared)->custom_block().has_value());
    target.block = *(*prepared)->custom_block();
    (*prepared)->preserve_files();
    auto reopened = FtsIndexer::OpenCustom(target, fields_, retirement);
    ASSERT_TRUE(reopened.has_value()) << reopened.error().message();
    for (const char *field : {"title", "body"}) {
      fts::TermNode query("beta");
      auto expected = source_->get(field)->search(query, {});
      ASSERT_TRUE(expected.has_value());
      for (const auto &generation : {*prepared, *reopened}) {
        auto actual = generation->get(field)->search(query, {});
        ASSERT_TRUE(actual.has_value());
        ASSERT_EQ(expected->size(), actual->size());
        for (size_t i = 0; i < actual->size(); ++i) {
          EXPECT_EQ((*expected)[i].doc_id, (*actual)[i].doc_id);
          EXPECT_EQ((*expected)[i].score, (*actual)[i].score);
        }
      }
    }
    EXPECT_TRUE(source_->get("title"));
    EXPECT_TRUE(source_->flush().ok());
  }
  // Conversion must retain the mutable source's auxiliary CFs for retry.
  EXPECT_TRUE(source_->insert("title", 3, "later").ok());
  EXPECT_TRUE(source_->flush().ok());
}

TEST_F(FtsGenerationTest,
       ReopenedGenerationSupportsConcurrentMultiTermQueries) {
  fts::OrNode query;
  std::string text;
  for (uint32_t i = 0; i < 32; ++i) {
    const auto term = "word" + std::to_string(i);
    query.children.push_back(std::make_unique<fts::TermNode>(term));
    text += term + " ";
  }
  for (uint32_t row = 3; row < 203; ++row) {
    ASSERT_TRUE(source_->insert("title", row, text).ok());
  }
  ASSERT_TRUE(source_->flush().ok());
  const auto expected = source_->get("title")->search(query, {});
  ASSERT_TRUE(expected.has_value());
  ASSERT_FALSE(expected->empty());
  auto retirement = fts::FtsRetirementQueue::Create();
  for (bool mmap : {false, true}) {
    auto destination = target(mmap, 203);
    auto prepared = source_->prepare_custom(destination, fields_, retirement);
    ASSERT_TRUE(prepared.has_value()) << prepared.error().message();
    destination.block = *(*prepared)->custom_block();
    auto reopened = FtsIndexer::OpenCustom(destination, fields_, retirement);
    ASSERT_TRUE(reopened.has_value()) << reopened.error().message();
    auto column = (*reopened)->get("title");
    auto first = column->search(query, {});
    ASSERT_TRUE(first.has_value()) << first.error().message();
    std::atomic<bool> success{true};
    std::atomic<uint32_t> ready{0};
    std::atomic<bool> start{false};
    std::vector<std::thread> workers;
    for (uint32_t worker = 0; worker < 4; ++worker) {
      workers.emplace_back([&] {
        ready.fetch_add(1);
        while (!start.load()) {
          std::this_thread::yield();
        }
        for (uint32_t iteration = 0; iteration < 5; ++iteration) {
          auto found = column->search(query, {});
          if (!found || found->size() != expected->size()) {
            success = false;
            return;
          }
          for (size_t i = 0; i < found->size(); ++i) {
            if ((*found)[i].doc_id != (*expected)[i].doc_id ||
                (*found)[i].score != (*expected)[i].score) {
              success = false;
              return;
            }
          }
        }
      });
    }
    while (ready.load() != workers.size()) {
      std::this_thread::yield();
    }
    start = true;
    for (auto &worker : workers) {
      worker.join();
    }
    EXPECT_TRUE(success.load());
  }
}

TEST_F(FtsGenerationTest, FailedPublicationKeepsSourceWritableAndRetryable) {
  auto retirement = fts::FtsRetirementQueue::Create();
  uint32_t rows = 3;
  for (bool mmap : {false, true}) {
    for (auto stage :
         {DurableFile::Stage::kWrite, DurableFile::Stage::kFileSync,
          DurableFile::Stage::kRename, DurableFile::Stage::kDirectorySync}) {
      std::string output;
      auto failed = source_->prepare_custom(
          target(mmap, rows), fields_, retirement,
          [&](auto step, const std::string &path) {
            output = path;
            return step == stage ? Status::InternalError("injected")
                                 : Status::OK();
          });
      ASSERT_FALSE(failed.has_value());
      ASSERT_FALSE(output.empty());
      for (const auto &suffix : {"", ".pending", ".reserved"}) {
        EXPECT_FALSE(std::filesystem::exists(output + suffix));
      }
      ASSERT_TRUE(source_->insert("title", rows++, "beta").ok());
      ASSERT_TRUE(source_->flush().ok());
      auto retry =
          source_->prepare_custom(target(mmap, rows), fields_, retirement);
      ASSERT_TRUE(retry.has_value()) << retry.error().message();
      auto found = (*retry)->get("title")->search(fts::TermNode("beta"), {});
      ASSERT_TRUE(found.has_value());
      EXPECT_EQ(rows - 1, found->size());
      EXPECT_EQ(source_->get("title")->total_docs(),
                (*retry)->get("title")->total_docs());
      EXPECT_EQ(source_->get("body")->total_tokens(),
                (*retry)->get("body")->total_tokens());
    }
  }
}

TEST_F(FtsGenerationTest, LegacySourceExportsExactLengthsPresenceAndTerms) {
  ASSERT_TRUE(source_->seal_all().ok());
  for (const auto &name : {"title", "body"}) {
    auto indexer = source_->get(name);
    uint32_t row = 0;
    const bool title = std::string(name) == "title";
    auto input = fts::OpenLegacyFtsFieldSource(
        indexer, 3, [&]() -> Result<std::optional<bool>> {
          if (row == 3) {
            return std::optional<bool>{};
          }
          const bool present = !title || row != 1;
          ++row;
          return std::optional<bool>{present};
        });
    ASSERT_TRUE(input.has_value()) << input.error().message();
    indexer.reset();
    if (!title) {
      source_.reset();
    }
    EXPECT_EQ(title ? 2 : 3, (*input)->statistics().scoring_documents);
    EXPECT_EQ(title ? 4 : 3, (*input)->statistics().total_tokens);
    EXPECT_EQ(title ? 2 : 3, (*input)->statistics().present);
    auto empty = (*input)->document(1);
    ASSERT_TRUE(empty.has_value());
    EXPECT_EQ(!title, empty->present);
    EXPECT_EQ(0, empty->length);
    auto terms = (*input)->terms();
    ASSERT_TRUE(terms.has_value());
    auto first = (*terms)->next();
    ASSERT_TRUE(first && first->has_value());
    auto bytes = (**first).bytes(0, 5);
    ASSERT_TRUE(bytes.has_value());
    EXPECT_EQ("alpha", *bytes);
    EXPECT_EQ(1, (**first).reader->df());
    auto postings = (**first).reader->postings();
    ASSERT_TRUE(postings.has_value());
    auto doc = (*postings)->next();
    ASSERT_TRUE(doc.has_value());
    EXPECT_EQ(title ? 0 : 2, *doc);
    auto payload = (*postings)->payload();
    ASSERT_TRUE(payload.has_value());
    EXPECT_EQ(title ? 2 : 1, payload->tf);
    EXPECT_EQ(title ? 3 : 2, payload->length);
    auto positions = (**first).reader->positions(*doc);
    ASSERT_TRUE(positions.has_value());
    auto position = (*positions)->next();
    ASSERT_TRUE(position && position->has_value());
    EXPECT_EQ(0, **position);
    auto second = (*terms)->next();
    ASSERT_TRUE(second && second->has_value());
    auto beta = (**second).bytes(0, 4);
    ASSERT_TRUE(beta.has_value());
    EXPECT_EQ("beta", *beta);
    EXPECT_EQ(2, (**second).reader->df());
    auto end = (*terms)->next();
    ASSERT_TRUE(end.has_value());
    EXPECT_FALSE(end->has_value());
  }
}

TEST_F(FtsGenerationTest, MixedMergeUsesForwardRowRankAndPhysicalScoringCount) {
  auto retirement = fts::FtsRetirementQueue::Create();
  auto custom = source_->prepare_custom(target(), fields_, retirement);
  ASSERT_TRUE(custom.has_value()) << custom.error().message();
  ASSERT_TRUE(source_->seal_all().ok());
  for (bool mmap : {false, true}) {
    uint32_t row = 0;
    auto legacy = fts::OpenLegacyFtsFieldSource(
        source_->get("title"), 3, [&]() -> Result<std::optional<bool>> {
          if (row == 3) {
            return std::optional<bool>{};
          }
          return std::optional<bool>{row++ != 1};
        });
    ASSERT_TRUE(legacy.has_value()) << legacy.error().message();
    auto input = fts::FtsContainerReader::Open((*custom)->working_dir(), mmap,
                                               1024 * 1024, *custom);
    ASSERT_TRUE(input.has_value());
    auto field = fts::FtsFieldReader::Open(*input, 0);
    ASSERT_TRUE(field.has_value());
    auto source = fts::OpenCustomFtsFieldSource(*field);
    ASSERT_TRUE(source.has_value());
    const auto output = path_ + "/merged.zfts";
    auto dumper = core::IndexFactory::CreateDumper("FileDumper");
    ASSERT_EQ(0, dumper->create(output));
    auto writer = fts::FtsContainerWriter::Create(dumper);
    ASSERT_TRUE(writer.has_value());
    roaring::Roaring deleted;
    deleted.add(0);
    deleted.add(4);
    auto result = fts::MergeFtsField({*legacy, *source}, deleted,
                                     fts::FtsMergePolicy::kCompact,
                                     (*field)->metadata(), writer->get(), 0);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(4, result->scoring_document_count);
    EXPECT_EQ(3, result->present_count);
    EXPECT_EQ(5, result->total_tokens);
    fts::format::Metadata metadata;
    metadata.base_id[0] = 42;
    metadata.base_generation = 1;
    metadata.row_count = 4;
    metadata.fields.push_back(*result);
    ASSERT_TRUE((*writer)->finish(metadata).ok());
    auto container = fts::FtsContainerReader::Open(output, mmap);
    ASSERT_TRUE(container.has_value());
    auto merged = fts::FtsFieldReader::Open(*container, 0);
    ASSERT_TRUE(merged.has_value());
    auto verified = (*merged)->verify();
    ASSERT_TRUE(verified.ok()) << verified.message();
    auto missing = (*merged)->document(0);
    ASSERT_TRUE(missing.has_value());
    EXPECT_FALSE(missing->present);
    EXPECT_EQ(0, missing->length);
    auto beta = (*merged)->lookup("beta");
    ASSERT_TRUE(beta.has_value());
    auto postings = (*merged)->postings(*beta);
    ASSERT_TRUE(postings.has_value());
    fts::BM25Scorer scorer;
    scorer.update_stats(4, 5);
    for (uint32_t expected : {1u, 2u, 3u}) {
      auto doc = (*postings)->next();
      ASSERT_TRUE(doc && doc->has_value());
      EXPECT_EQ(expected, (**doc).doc);
      EXPECT_EQ(1, (**doc).tf);
      auto length = (*postings)->doc_length();
      ASSERT_TRUE(length.has_value());
      EXPECT_EQ(expected == 2 ? 3 : 1, *length);
      auto positions = (*postings)->positions();
      ASSERT_TRUE(positions.has_value());
      auto position = (*positions)->next();
      ASSERT_TRUE(position && position->has_value());
      EXPECT_EQ(expected == 2 ? 2 : 0, **position);
      EXPECT_FLOAT_EQ(scorer.score(3, 1, *length),
                      (*merged)->score(3, 1, *length));
    }
    auto end = (*postings)->next();
    ASSERT_TRUE(end.has_value());
    EXPECT_FALSE(end->has_value());
  }
}

TEST_F(FtsGenerationTest, ManySmallSourcesFitTheMergeWorkspace) {
  auto custom = source_->prepare_custom(target(), fields_,
                                        fts::FtsRetirementQueue::Create());
  ASSERT_TRUE(custom.has_value()) << custom.error().message();
  auto immutable = (*custom)->open_source("title", 3);
  ASSERT_TRUE(immutable.has_value());
  ASSERT_TRUE(source_->seal_all().ok());
  uint32_t row = 0;
  auto legacy = fts::OpenLegacyFtsFieldSource(
      source_->get("title"), 3, [&]() -> Result<std::optional<bool>> {
        if (row == 3) {
          return std::optional<bool>{};
        }
        return std::optional<bool>{row++ != 1};
      });
  ASSERT_TRUE(legacy.has_value());
  for (size_t count : {8u, 65u, 129u}) {
    SCOPED_TRACE(count);
    std::vector<fts::FtsFieldSource::Ptr> inputs(
        count, count == 8 ? *legacy : *immutable);
    inputs.front() = *immutable;
    const auto path = path_ + "/many-" + std::to_string(count) + ".zfts";
    auto dumper = core::IndexFactory::CreateDumper("FileDumper");
    ASSERT_EQ(0, dumper->create(path));
    auto writer = fts::FtsContainerWriter::Create(dumper);
    ASSERT_TRUE(writer.has_value());
    auto merged =
        fts::MergeFtsField(inputs, {}, fts::FtsMergePolicy::kCompact,
                           (*immutable)->metadata(), writer->get(), 0);
    if (count == 129) {
      ASSERT_FALSE(merged.has_value());
      EXPECT_EQ(StatusCode::RESOURCE_EXHAUSTED, merged.error().code());
      continue;
    }
    ASSERT_TRUE(merged.has_value()) << merged.error().message();
    EXPECT_EQ(count * 3, merged->scoring_document_count);
    EXPECT_EQ(count * 2, merged->present_count);
    EXPECT_EQ(count * 4, merged->total_tokens);
    fts::format::Metadata metadata;
    metadata.base_id[0] = 42;
    metadata.base_generation = 1;
    metadata.row_count = count * 3;
    metadata.fields.push_back(*merged);
    ASSERT_TRUE((*writer)->finish(metadata).ok());
    auto file = fts::FtsContainerReader::Open(path, true);
    ASSERT_TRUE(file.has_value());
    auto field = fts::FtsFieldReader::Open(*file, 0);
    ASSERT_TRUE(field.has_value());
    ASSERT_TRUE((*field)->verify().ok());
    auto beta = (*field)->lookup("beta");
    ASSERT_TRUE(beta.has_value());
    auto postings = (*field)->postings(*beta);
    ASSERT_TRUE(postings.has_value());
    for (uint32_t segment = 0; segment < count; ++segment) {
      for (uint32_t local : {0u, 2u}) {
        auto doc = (*postings)->next();
        ASSERT_TRUE(doc && doc->has_value());
        EXPECT_EQ(segment * 3 + local, (**doc).doc);
      }
    }
    auto end = (*postings)->next();
    ASSERT_TRUE(end.has_value());
    EXPECT_FALSE(end->has_value());
  }
}

TEST_F(FtsGenerationTest,
       LegacySourceRejectsMissingPresenceAndWorkspaceOverflow) {
  ASSERT_TRUE(source_->seal_all().ok());
  auto field = source_->get("title");
  for (int failure = 0; failure < 4; ++failure) {
    uint32_t row = 0;
    auto input = fts::OpenLegacyFtsFieldSource(
        field, 3,
        [&]() -> Result<std::optional<bool>> {
          if (failure == 2) {
            return tl::make_unexpected(Status{StatusCode::UNAVAILABLE,
                                              "injected forward read failure"});
          }
          if (row == (failure == 0 ? 2u : 3u)) {
            return std::optional<bool>{};
          }
          const bool present = failure != 1 && row != 1;
          ++row;
          return std::optional<bool>{present};
        },
        failure == 3 ? 1 : 8 * 1024 * 1024);
    ASSERT_FALSE(input.has_value());
    EXPECT_EQ(failure == 3 ? StatusCode::RESOURCE_EXHAUSTED
                           : (failure == 2 ? StatusCode::UNAVAILABLE
                                           : StatusCode::INTERNAL_ERROR),
              input.error().code());
  }
}

TEST_F(FtsGenerationTest, PureRewritePreservesMissingRowsAndExactScores) {
  ASSERT_TRUE(source_->seal_all().ok());
  for (const auto &name : {"title", "body"}) {
    uint32_t row = 0;
    const bool title = std::string(name) == "title";
    auto input = fts::OpenLegacyFtsFieldSource(
        source_->get(name), 3, [&]() -> Result<std::optional<bool>> {
          if (row == 3) {
            return std::optional<bool>{};
          }
          return std::optional<bool>{row++ != 1 || !title};
        });
    ASSERT_TRUE(input.has_value());
    const auto output = path_ + "/rewrite.zfts";
    auto dumper = core::IndexFactory::CreateDumper("FileDumper");
    ASSERT_EQ(0, dumper->create(output));
    auto writer = fts::FtsContainerWriter::Create(dumper);
    ASSERT_TRUE(writer.has_value());
    fts::format::FieldMetadata identity;
    identity.id[0] = 1;
    identity.incarnation[0] = 2;
    identity.name = name;
    auto result =
        fts::MergeFtsField({*input}, {}, fts::FtsMergePolicy::kRewrite,
                           identity, writer->get(), 0);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    EXPECT_EQ(title ? 2 : 3, result->scoring_document_count);
    EXPECT_EQ(title ? 4 : 3, result->total_tokens);
    fts::format::Metadata metadata;
    metadata.base_id[0] = 8;
    metadata.base_generation = 1;
    metadata.row_count = 3;
    metadata.fields.push_back(*result);
    ASSERT_TRUE((*writer)->finish(metadata).ok());
    for (bool mmap : {false, true}) {
      auto container = fts::FtsContainerReader::Open(output, mmap);
      ASSERT_TRUE(container.has_value());
      auto field = fts::FtsFieldReader::Open(*container, 0);
      ASSERT_TRUE(field.has_value());
      ASSERT_TRUE((*field)->verify().ok());
      auto empty = (*field)->document(1);
      ASSERT_TRUE(empty.has_value());
      EXPECT_EQ(!title, empty->present);
      EXPECT_EQ(0, empty->length);
      fts::FtsColumnIndexer actual;
      ASSERT_TRUE(actual.open_reader(*field).has_value());
      for (const auto &term : {"alpha", "beta"}) {
        auto expected = source_->get(name)->search(fts::TermNode(term), {});
        auto found = actual.search(fts::TermNode(term), {});
        ASSERT_TRUE(expected && found);
        ASSERT_EQ(expected->size(), found->size());
        for (size_t i = 0; i < expected->size(); ++i) {
          EXPECT_EQ((*expected)[i].doc_id, (*found)[i].doc_id);
          EXPECT_EQ((*expected)[i].score, (*found)[i].score);
        }
      }
    }
  }
}

TEST_F(FtsGenerationTest, RewrittenGenerationPreservesFieldsAndReopens) {
  auto retirement = fts::FtsRetirementQueue::Create();
  auto original = source_->prepare_custom(target(), fields_, retirement);
  ASSERT_TRUE(original.has_value());
  ASSERT_TRUE(source_->seal_all().ok());
  std::vector<std::vector<fts::FtsFieldSource::Ptr>> sources;
  auto title = (*original)->open_source("title", 3);
  ASSERT_TRUE(title.has_value());
  sources.push_back({*title});
  uint32_t row = 0;
  auto body =
      source_->open_source("body", 3, [&]() -> Result<std::optional<bool>> {
        if (row++ < 3) {
          return std::optional<bool>{true};
        }
        return std::optional<bool>{};
      });
  ASSERT_TRUE(body.has_value());
  sources.push_back({*body});
  auto rewritten =
      FtsIndexer::PrepareMerged(target(), fields_, sources, {},
                                fts::FtsMergePolicy::kRewrite, retirement);
  ASSERT_TRUE(rewritten.has_value()) << rewritten.error().message();
  auto reference = target();
  reference.block = *(*rewritten)->custom_block();
  EXPECT_NE((*original)->custom_block()->fts_base_->base_id,
            reference.block.fts_base_->base_id);
  auto reopened = FtsIndexer::OpenCustom(reference, fields_, retirement);
  ASSERT_TRUE(reopened.has_value());
  auto new_title = (*reopened)->open_source("title", 3);
  ASSERT_TRUE(new_title.has_value());
  EXPECT_EQ((*title)->metadata().id, (*new_title)->metadata().id);
  EXPECT_EQ((*title)->metadata().incarnation,
            (*new_title)->metadata().incarnation);
  for (const auto &name : {"title", "body"}) {
    auto expected = source_->get(name)->search(fts::TermNode("beta"), {});
    auto actual = (*reopened)->get(name)->search(fts::TermNode("beta"), {});
    ASSERT_TRUE(expected && actual);
    ASSERT_EQ(expected->size(), actual->size());
    for (size_t i = 0; i < actual->size(); ++i) {
      EXPECT_EQ((*expected)[i].doc_id, (*actual)[i].doc_id);
      EXPECT_EQ((*expected)[i].score, (*actual)[i].score);
    }
  }
  auto mismatch = target(false, 4);
  EXPECT_FALSE(FtsIndexer::PrepareMerged(mismatch, fields_, sources, {},
                                         fts::FtsMergePolicy::kRewrite,
                                         retirement)
                   .has_value());
}

TEST_F(FtsGenerationTest, ImmutableSnapshotSurvivesSourceRetirement) {
  auto retirement = fts::FtsRetirementQueue::Create();
  for (bool mmap : {false, true}) {
    auto generation =
        source_->prepare_custom(target(mmap), fields_, retirement);
    ASSERT_TRUE(generation.has_value());
    auto snapshot = target(mmap);
    snapshot.block = *(*generation)->custom_block();
    snapshot.segment_path =
        path_ + (mmap ? "/snapshot-mmap" : "/snapshot-buffer");
    auto status = (*generation)->create_snapshot(snapshot.segment_path);
    ASSERT_TRUE(status.ok()) << status.message();
    // Existing snapshot destinations are never silently overwritten.
    EXPECT_FALSE((*generation)->create_snapshot(snapshot.segment_path).ok());
    const auto original_path = (*generation)->working_dir();
    (*generation)->retire_files();
    generation->reset();
    EXPECT_FALSE(std::filesystem::exists(original_path));
    auto copy = FtsIndexer::OpenCustom(snapshot, fields_, retirement);
    ASSERT_TRUE(copy.has_value()) << copy.error().message();
    for (const auto &name : {"title", "body"}) {
      auto expected = source_->get(name)->search(fts::TermNode("beta"), {});
      auto actual = (*copy)->get(name)->search(fts::TermNode("beta"), {});
      ASSERT_TRUE(expected && actual);
      ASSERT_EQ(expected->size(), actual->size());
      for (size_t i = 0; i < actual->size(); ++i) {
        EXPECT_EQ((*expected)[i].doc_id, (*actual)[i].doc_id);
        EXPECT_EQ((*expected)[i].score, (*actual)[i].score);
      }
    }
  }
}

TEST_F(FtsGenerationTest, ReopenRequiresExactDescriptorAndAnalysisIdentity) {
  auto retirement = fts::FtsRetirementQueue::Create();
  auto prepared = source_->prepare_custom(target(), fields_, retirement);
  ASSERT_TRUE(prepared.has_value()) << prepared.error().message();
  auto committed = target();
  committed.block = *(*prepared)->custom_block();
  for (int mismatch = 0; mismatch < 5; ++mismatch) {
    auto bad = committed;
    if (mismatch == 0) {
      bad.block.fts_base_->metadata_digest[0] ^= 1;
    } else if (mismatch == 1) {
      ++bad.segment_id;
    } else if (mismatch == 2) {
      ++bad.block.fts_base_->row_count;
    } else if (mismatch == 3) {
      ++bad.block.fts_base_->generation;
    } else {
      bad.block.fts_base_->required_features = 1;
    }
    EXPECT_FALSE(FtsIndexer::OpenCustom(bad, fields_, retirement).has_value());
  }
  auto changed = fields_;
  changed[0] = std::make_shared<FieldSchema>(
      "title", DataType::STRING, true,
      std::make_shared<FtsIndexParams>("standard"));
  EXPECT_FALSE(
      FtsIndexer::OpenCustom(committed, changed, retirement).has_value());
  EXPECT_FALSE(
      source_->prepare_custom(target(), changed, retirement).has_value());
  EXPECT_FALSE((*prepared)->insert("title", 3, "later").ok());
}

TEST_F(FtsGenerationTest, RejectsOversizedConfigurationBeforeCreatingFiles) {
  auto retirement = fts::FtsRetirementQueue::Create();
  for (bool many_filters : {false, true}) {
    for (const auto &field : fields_) {
      auto params =
          std::static_pointer_cast<FtsIndexParams>(field->index_params());
      if (many_filters) {
        params->set_filters(
            std::vector<std::string>(fts::format::kMetadataBudget / 2, ""));
      } else {
        // Each field fits separately; the complete generation must fit too.
        params->set_extra_params(
            std::string(fts::format::kMetadataBudget / 2, 'x'));
      }
    }
    bool created = false;
    auto rejected = source_->prepare_custom(target(), fields_, retirement,
                                            [&](auto, const auto &) {
                                              created = true;
                                              return Status::OK();
                                            });
    EXPECT_FALSE(rejected.has_value());
    if (!rejected) {
      EXPECT_EQ(StatusCode::RESOURCE_EXHAUSTED, rejected.error().code());
    }
    EXPECT_FALSE(created);
    for (const auto &field : fields_) {
      auto params =
          std::static_pointer_cast<FtsIndexParams>(field->index_params());
      params->set_extra_params("");
      params->set_filters({"lowercase"});
    }
  }
  EXPECT_TRUE(
      source_->prepare_custom(target(), fields_, retirement).has_value());
}

TEST_F(FtsGenerationTest, RetirementWaitsForReopenedAndQueryOwners) {
  auto retirement = fts::FtsRetirementQueue::Create();
  for (bool mmap : {false, true}) {
    auto prepared = source_->prepare_custom(target(mmap), fields_, retirement);
    ASSERT_TRUE(prepared.has_value()) << prepared.error().message();
    auto reference = target(mmap);
    reference.block = *(*prepared)->custom_block();
    (*prepared)->preserve_files();
    auto reopened = FtsIndexer::OpenCustom(reference, fields_, retirement);
    ASSERT_TRUE(reopened.has_value());
    auto query_owner = (*reopened)->get("title");
    const auto path = (*prepared)->working_dir();
    (*prepared)->retire_files();
    prepared->reset();
    EXPECT_TRUE(std::filesystem::exists(path));
    reopened->reset();
    EXPECT_TRUE(std::filesystem::exists(path));
    auto found = query_owner->search(fts::TermNode("beta"), {});
    ASSERT_TRUE(found.has_value());
    EXPECT_EQ(2, found->size());
    query_owner.reset();
    EXPECT_FALSE(std::filesystem::exists(path));
  }
}

TEST_F(FtsGenerationTest, LegacyRetirementWaitsForContextHeldByQuery) {
  source_.reset();
  auto retirement = fts::FtsRetirementQueue::Create();
  const auto path = path_ + "/mutable";
  source_ = FtsIndexer::CreateAndOpen(path, fields_, false, false, retirement);
  ASSERT_NE(nullptr, source_);
  auto query_owner = source_->get("title");
  source_->retire_files();
  source_.reset();
  EXPECT_TRUE(std::filesystem::exists(path));
  auto found = query_owner->search(fts::TermNode("beta"), {});
  ASSERT_TRUE(found.has_value());
  EXPECT_EQ(2, found->size());
  query_owner.reset();
  EXPECT_FALSE(std::filesystem::exists(path));
}

}  // namespace
}  // namespace zvec
