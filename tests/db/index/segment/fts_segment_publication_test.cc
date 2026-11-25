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

#include <filesystem>
#include <gtest/gtest.h>
#include <zvec/ailego/io/file.h>
#include <zvec/db/collection.h>
#include "db/common/file_helper.h"
#include "db/index/segment/segment.h"
#include "db/index/segment/segment_helper.h"
#include "db/index/segment/segment_manager.h"
#include "db/index/storage/wal/wal_file.h"

namespace zvec {
namespace {

class FtsSegmentPublicationTest : public testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    ailego::LoggerBroker::SetLevel(ailego::Logger::LEVEL_WARN);
    ASSERT_EQ(0,
              ailego::MemoryLimitPool::get_instance().init(64 * 1024 * 1024));
    FileHelper::RemoveDirectory(path_);
    ASSERT_TRUE(FileHelper::CreateDirectory(path_));
    schema_.add_field(std::make_shared<FieldSchema>(
        "text", DataType::STRING, true,
        std::make_shared<FtsIndexParams>("whitespace")));
    Version version;
    version.set_schema(schema_);
    version.set_enable_mmap(GetParam());
    auto manager = VersionManager::Create(path_, version);
    ASSERT_TRUE(manager.has_value());
    versions_ = *manager;
    ids_ = IDMap::CreateAndOpen(
        "fts", FileHelper::MakeFilePath(path_, FileID::ID_FILE, 0), true,
        false);
    ASSERT_NE(nullptr, ids_);
    deletes_ = std::make_shared<DeleteStore>("fts");
    auto segment = Segment::CreateAndOpen(
        path_, schema_, 0, 100, ids_, deletes_, versions_,
        SegmentOptions{false, GetParam(), 64 * 1024 * 1024});
    ASSERT_TRUE(segment.has_value()) << segment.error().message();
    segment_ = *segment;
    insert("a", "alpha beta");
    insert("b", "beta");
  }
  void TearDown() override {
    segment_.reset();
    ids_.reset();
    versions_.reset();
    FileHelper::RemoveDirectory(path_);
  }
  void insert(const std::string &pk, const std::string &text) {
    Doc doc;
    doc.set_pk(pk);
    doc.set("text", text);
    ASSERT_TRUE(segment_->insert(doc).ok());
  }
  std::string path_{"fts_segment_publication.fixture"};
  CollectionSchema schema_{"fts"};
  VersionManager::Ptr versions_;
  IDMap::Ptr ids_;
  DeleteStore::Ptr deletes_;
  Segment::Ptr segment_;
};

TEST_P(FtsSegmentPublicationTest, FailedCreateRemovesOnlyOwnedDirectory) {
  auto invalid = schema_;
  invalid.get_field("text")->set_index_params(
      std::make_shared<FtsIndexParams>("missing-fts-tokenizer"));
  const auto candidate = FileHelper::MakeSegmentPath(path_, 77);
  auto failed = Segment::CreateAndOpen(
      path_, invalid, 77, 200, ids_, deletes_, versions_,
      SegmentOptions{false, GetParam(), 64 * 1024 * 1024});
  EXPECT_FALSE(failed.has_value());
  EXPECT_FALSE(std::filesystem::exists(candidate));
  ASSERT_TRUE(std::filesystem::create_directory(candidate));
  ASSERT_TRUE(std::filesystem::create_directory(candidate + "/keep"));
  auto existing = Segment::CreateAndOpen(
      path_, schema_, 77, 200, ids_, deletes_, versions_,
      SegmentOptions{false, GetParam(), 64 * 1024 * 1024});
  EXPECT_FALSE(existing.has_value());
  EXPECT_TRUE(std::filesystem::exists(candidate + "/keep"));
}

#ifdef _WIN32
TEST_P(FtsSegmentPublicationTest, UnsupportedSealPreservesReadableCheckpoint) {
  ASSERT_TRUE(segment_->flush().ok());
  auto checkpoint = versions_->get_current_version().writing_segment_meta();
  ASSERT_NE(nullptr, checkpoint);
  auto prepared = segment_->prepare_fts_seal();
  ASSERT_FALSE(prepared);
  EXPECT_EQ(StatusCode::NOT_SUPPORTED, prepared.error().code());
  EXPECT_FALSE(versions_->uses_commit_record());
  EXPECT_TRUE(segment_->meta()->has_writing_forward_block());
  segment_.reset();
  auto reopened =
      Segment::Open(path_, schema_, *checkpoint, ids_, deletes_, versions_,
                    SegmentOptions{false, GetParam(), 64 * 1024 * 1024});
  ASSERT_TRUE(reopened) << reopened.error().message();
  auto result = (*reopened)->fts_search("text", fts::TermNode("beta"), {});
  ASSERT_TRUE(result) << result.error().message();
  EXPECT_EQ(2u, result->size());
}
#else
TEST_P(FtsSegmentPublicationTest,
       PrepareCommitPublishAndReopenPreserveReaders) {
  auto old_reader = segment_->get_fts_indexer("text");
  const auto old_path = old_reader->ctx()->db_->GetName();
  auto prepared = segment_->prepare_fts_seal();
  ASSERT_TRUE(prepared.has_value()) << prepared.error().message();
  EXPECT_TRUE(segment_->meta()->writing_forward_block().has_value());
  EXPECT_EQ(old_reader, segment_->get_fts_indexer("text"));
  auto candidate = versions_->get_current_version();
  ASSERT_TRUE(candidate.add_persisted_segment_meta(prepared->metadata).ok());
  candidate.reset_writing_segment_meta(nullptr);
  ASSERT_TRUE(versions_->commit(candidate).ok());
  auto committed_meta = prepared->metadata;
  segment_->publish_fts_seal(std::move(*prepared));
  EXPECT_FALSE(segment_->meta()->writing_forward_block().has_value());
  auto forward = segment_->prepare_forward_read({"text"}, false);
  ASSERT_TRUE(forward.ok()) << forward.status().ToString();
  EXPECT_EQ(forward->num_rows, 2u);
  EXPECT_EQ(static_cast<bool>(forward->view), GetParam());
  EXPECT_NE(old_reader, segment_->get_fts_indexer("text"));
  EXPECT_TRUE(std::filesystem::exists(old_path));
  auto old_result = old_reader->search(fts::TermNode("beta"), {});
  auto result = segment_->fts_search("text", fts::TermNode("beta"), {});
  ASSERT_TRUE(old_result.has_value());
  ASSERT_TRUE(result.has_value());
  ASSERT_EQ(2, result->size());
  for (size_t i = 0; i < result->size(); ++i) {
    EXPECT_EQ((*old_result)[i].doc_id, (*result)[i].doc_id);
    EXPECT_EQ((*old_result)[i].score, (*result)[i].score);
  }
  old_reader.reset();
  EXPECT_FALSE(std::filesystem::exists(old_path));
  segment_.reset();
  auto recovered = VersionManager::Recovery(path_);
  ASSERT_TRUE(recovered.has_value()) << recovered.error().message();
  auto reopened =
      Segment::Open(path_, schema_, *committed_meta, ids_, deletes_, *recovered,
                    SegmentOptions{true, !GetParam(), 64 * 1024 * 1024});
  ASSERT_TRUE(reopened.has_value()) << reopened.error().message();
  auto after = (*reopened)->fts_search("text", fts::TermNode("beta"), {});
  ASSERT_TRUE(after.has_value());
  EXPECT_EQ(result->size(), after->size());
}

TEST_P(FtsSegmentPublicationTest,
       FailedCandidateAllowsFurtherWritesAndFreshRetry) {
  auto failed = segment_->prepare_fts_seal([](auto stage, const auto &) {
    return stage == DurableFile::Stage::kRename
               ? Status::InternalError("injected")
               : Status::OK();
  });
  ASSERT_FALSE(failed.has_value());
  insert("c", "beta");
  auto retry = segment_->prepare_fts_seal();
  ASSERT_TRUE(retry.has_value()) << retry.error().message();
  EXPECT_EQ(3, retry->metadata->doc_count());
  auto result = retry->indexer->get("text")->search(fts::TermNode("beta"), {});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(3, result->size());
  EXPECT_EQ(3, retry->indexer->get("text")->total_docs());
}

TEST_P(FtsSegmentPublicationTest,
       SingleLegacyRewriteKeepsRowsAndDeletionState) {
  ASSERT_TRUE(segment_->Delete("a").ok());
  ASSERT_TRUE(segment_->dump().ok());
  segment_->remove_writing_forward_block();
  auto legacy_version = versions_->get_current_version();
  ASSERT_TRUE(legacy_version.add_persisted_segment_meta(segment_->meta()).ok());
  legacy_version.reset_writing_segment_meta(nullptr);
  ASSERT_TRUE(versions_->checkpoint(legacy_version).ok());
  auto old_reader = segment_->get_fts_indexer("text");
  auto expected = old_reader->search(fts::TermNode("beta"), {});
  ASSERT_TRUE(expected.has_value());
  auto task = SegmentTask::CreateRewriteFtsTask({segment_, {}});
  auto status = SegmentHelper::Execute(task);
  ASSERT_TRUE(status.ok()) << status.message();
  auto &rewrite = std::get<RewriteFtsTask>(task->task_info());
  ASSERT_TRUE(rewrite.prepared_.has_value());
  EXPECT_EQ(old_reader, segment_->get_fts_indexer("text"));
  auto candidate = versions_->get_current_version();
  ASSERT_TRUE(
      candidate.update_persisted_segment_meta(rewrite.prepared_->metadata)
          .ok());
  ASSERT_TRUE(versions_->commit(candidate).ok());
  segment_->publish_fts_seal(std::move(*rewrite.prepared_));
  EXPECT_EQ(2, segment_->doc_count_snapshot());
  EXPECT_TRUE(segment_->get_filter()->is_filtered(0));
  EXPECT_FALSE(segment_->get_filter()->is_filtered(1));
  auto actual = segment_->fts_search("text", fts::TermNode("beta"), {});
  ASSERT_TRUE(actual.has_value());
  ASSERT_EQ(expected->size(), actual->size());
  for (size_t i = 0; i < actual->size(); ++i) {
    EXPECT_EQ((*expected)[i].doc_id, (*actual)[i].doc_id);
    EXPECT_EQ((*expected)[i].score, (*actual)[i].score);
  }
  auto reopened = Segment::Open(
      path_, schema_, *segment_->meta(), ids_, deletes_, versions_,
      SegmentOptions{true, !GetParam(), 64 * 1024 * 1024});
  ASSERT_TRUE(reopened.has_value()) << reopened.error().message();
  auto source = (*reopened)->open_fts_source("text");
  ASSERT_TRUE(source.has_value()) << source.error().message();
  EXPECT_EQ(2, (*source)->statistics().scoring_documents);
  EXPECT_EQ(3, (*source)->statistics().total_tokens);
}

TEST_P(FtsSegmentPublicationTest, MixedCompactionMatchesForwardWithGlobalGaps) {
  auto prepared = segment_->prepare_fts_seal();
  ASSERT_TRUE(prepared.has_value());
  auto version = versions_->get_current_version();
  ASSERT_TRUE(version.add_persisted_segment_meta(prepared->metadata).ok());
  version.reset_writing_segment_meta(nullptr);
  ASSERT_TRUE(versions_->commit(version).ok());
  segment_->publish_fts_seal(std::move(*prepared));
  auto second = Segment::CreateAndOpen(
      path_, schema_, 1, 200, ids_, deletes_, versions_,
      SegmentOptions{false, GetParam(), 64 * 1024 * 1024});
  ASSERT_TRUE(second.has_value());
  for (const auto &record :
       {std::make_pair("c", "gamma"), std::make_pair("d", "alpha beta")}) {
    Doc doc;
    doc.set_pk(record.first);
    doc.set("text", std::string(record.second));
    ASSERT_TRUE((*second)->insert(doc).ok());
  }
  ASSERT_TRUE((*second)->dump().ok());
  (*second)->remove_writing_forward_block();
  auto filter =
      EasyIndexFilter::Create([](uint64_t doc) { return doc == 100; });
  auto task = SegmentTask::CreateCompactTask(
      {path_,
       std::make_shared<CollectionSchema>(schema_),
       {segment_, *second},
       77,
       filter,
       !GetParam(),
       1});
  std::get<CompactTask>(task->task_info()).published_segment_id_ = 88;
  auto status = SegmentHelper::Execute(task);
  ASSERT_TRUE(status.ok()) << status.message();
  auto &compacted = std::get<CompactTask>(task->task_info());
  ASSERT_NE(nullptr, compacted.output_segment_meta_);
  EXPECT_TRUE(task->requires_commit_record());
  ASSERT_TRUE(
      FileHelper::MoveDirectory(FileHelper::MakeTempSegmentPath(path_, 77),
                                FileHelper::MakeSegmentPath(path_, 88)));
  auto merged = Segment::Open(
      path_, schema_, *compacted.output_segment_meta_, ids_, deletes_,
      versions_, SegmentOptions{true, GetParam(), 64 * 1024 * 1024});
  ASSERT_TRUE(merged.has_value()) << merged.error().message();
  EXPECT_EQ(3, (*merged)->doc_count_snapshot());
  auto result = (*merged)->fts_search("text", fts::TermNode("beta"), {});
  ASSERT_TRUE(result.has_value());
  ASSERT_EQ(2, result->size());
  EXPECT_EQ(0, (*result)[0].doc_id);
  EXPECT_EQ(2, (*result)[1].doc_id);
  auto a = (*merged)->fetch(101);
  auto c = (*merged)->fetch(200);
  auto d = (*merged)->fetch(201);
  ASSERT_NE(nullptr, a);
  ASSERT_NE(nullptr, c);
  ASSERT_NE(nullptr, d);
  EXPECT_EQ("b", a->pk());
  EXPECT_EQ("c", c->pk());
  EXPECT_EQ("d", d->pk());
  auto field = (*merged)->get_fts_indexer("text");
  EXPECT_EQ(3, field->total_docs());
  EXPECT_EQ(4, field->total_tokens());
}

TEST_P(FtsSegmentPublicationTest, CollectionOptimizeCommitsCustomReplacement) {
  // Build a committed custom collection through the internal publication
  // seam, without enabling the production writer merely for a test.
  ailego::File lock;
  ASSERT_TRUE(lock.create((path_ + "/LOCK").c_str(), 0));
  lock.close();
  ASSERT_TRUE(
      deletes_->flush(FileHelper::MakeFilePath(path_, FileID::DELETE_FILE, 0))
          .ok());
  auto manager = std::make_shared<SegmentManager>();
  uint32_t next_id = 1;
  auto switch_writer = [&]() {
    return manager->switch_fts_writer(
        &segment_, versions_,
        [&](uint64_t minimum) -> Result<SegmentManager::FtsSuccessor> {
          auto successor = Segment::CreateAndOpen(
              path_, schema_, next_id++, minimum, ids_, deletes_, versions_,
              SegmentOptions{false, GetParam(), 64 * 1024 * 1024});
          if (!successor) {
            return tl::make_unexpected(successor.error());
          }
          return SegmentManager::FtsSuccessor{*successor, next_id};
        });
  };
  ASSERT_TRUE(switch_writer().ok());
  insert("c", "gamma");
  insert("d", "alpha beta");
  ASSERT_TRUE(switch_writer().ok());
  // Each immutable source has a bounded head reservation. The planner must
  // split this group before the reducer's aggregate workspace is exceeded.
  for (int i = 0; i < 127; ++i) {
    insert("extra-" + std::to_string(i), "beta");
    ASSERT_TRUE(switch_writer().ok());
  }
  segment_.reset();
  manager.reset();
  ids_.reset();
  versions_.reset();
  deletes_.reset();
  auto collection = Collection::Open(path_, CollectionOptions{});
  ASSERT_TRUE(collection.has_value()) << collection.error().message();
  SearchQuery query;
  query.topk_ = 200;
  query.target_.field_name_ = "text";
  FtsClause clause;
  clause.query_string_ = "beta";
  query.target_.clause_ = clause;
  auto before = (*collection)->query(query);
  ASSERT_TRUE(before.has_value()) << before.error().message();
  ASSERT_EQ(130, before->size());
  auto status = (*collection)->optimize();
  ASSERT_TRUE(status.ok()) << status.message();
  auto after = (*collection)->query(query);
  ASSERT_TRUE(after.has_value()) << after.error().message();
  ASSERT_EQ(130, after->size());
  std::set<std::string> pks;
  for (const auto &doc : *after) {
    pks.insert(doc->pk());
  }
  std::set<std::string> expected{"a", "b", "d"};
  for (int i = 0; i < 127; ++i) {
    expected.insert("extra-" + std::to_string(i));
  }
  EXPECT_EQ(expected, pks);
  ASSERT_TRUE((*collection)->close().ok());
  collection->reset();
  auto reopened = Collection::Open(path_, CollectionOptions{});
  ASSERT_TRUE(reopened.has_value()) << reopened.error().message();
  auto again = (*reopened)->query(query);
  ASSERT_TRUE(again.has_value());
  ASSERT_EQ(after->size(), again->size());
  for (size_t i = 0; i < after->size(); ++i) {
    EXPECT_EQ((*after)[i]->pk(), (*again)[i]->pk());
    EXPECT_EQ((*after)[i]->score(), (*again)[i]->score());
  }
  ASSERT_TRUE((*reopened)->close().ok());
}

TEST_P(FtsSegmentPublicationTest, VectorReplacementWaitsForOldSegment) {
  CollectionSchema schema = schema_;
  schema.add_field(std::make_shared<FieldSchema>(
      "vector", DataType::VECTOR_FP32, 4, false,
      std::make_shared<HnswIndexParams>(MetricType::L2)));
  schema.add_field(std::make_shared<FieldSchema>(
      "unchanged", DataType::VECTOR_FP32, 4, false,
      std::make_shared<FlatIndexParams>(MetricType::L2)));
  auto created = Segment::CreateAndOpen(
      path_, schema, 2, 200, ids_, deletes_, versions_,
      SegmentOptions{false, GetParam(), 64 * 1024 * 1024});
  ASSERT_TRUE(created.has_value()) << created.error().message();
  auto old = std::move(*created);
  Doc doc;
  doc.set_pk("vector-row");
  doc.set("text", std::string("beta"));
  doc.set<std::vector<float>>("vector", std::vector<float>(4, 1.0f));
  doc.set<std::vector<float>>("unchanged", std::vector<float>(4, 2.0f));
  ASSERT_TRUE(old->insert(doc).ok());
  auto sealed = old->prepare_fts_seal();
  ASSERT_TRUE(sealed.has_value()) << sealed.error().message();
  auto version = versions_->get_current_version();
  version.set_schema(schema);
  ASSERT_TRUE(version.add_persisted_segment_meta(sealed->metadata).ok());
  ASSERT_TRUE(versions_->commit(version).ok());
  old->publish_fts_seal(std::move(*sealed));
  auto reader =
      Segment::Open(path_, schema, *old->meta(), ids_, deletes_, versions_,
                    SegmentOptions{true, GetParam(), 64 * 1024 * 1024});
  ASSERT_TRUE(reader.has_value()) << reader.error().message();
  SegmentManager manager;
  ASSERT_TRUE(manager.add_segment(old).ok());
  const auto old_path =
      old->get_vector_indexer("vector").front()->index_file_path();
  const auto unchanged_path =
      old->get_vector_indexer("unchanged").front()->index_file_path();
  auto task = SegmentTask::CreateCreateVectorIndexTask({old, "", nullptr, 1});
  ASSERT_TRUE(SegmentHelper::Execute(task).ok());
  auto &built = std::get<CreateVectorIndexTask>(task->task_info());
  auto replacement = Segment::Open(
      path_, schema, *built.output_segment_meta_, ids_, deletes_, versions_,
      SegmentOptions{true, GetParam(), 64 * 1024 * 1024});
  ASSERT_TRUE(replacement.has_value()) << replacement.error().message();
  const auto new_path =
      (*replacement)->get_vector_indexer("vector").front()->index_file_path();
  ASSERT_NE(old_path, new_path);
  {
    auto abandoned = manager.prepare_update({*replacement}, {});
    ASSERT_TRUE(abandoned.has_value());
  }
  EXPECT_TRUE(std::filesystem::exists(old_path));
  auto publication = manager.prepare_update({*replacement}, {});
  ASSERT_TRUE(publication.has_value());
  version = versions_->get_current_version();
  ASSERT_TRUE(
      version.update_persisted_segment_meta(built.output_segment_meta_).ok());
  ASSERT_TRUE(versions_->checkpoint(version).ok());
  manager.publish_update(std::move(*publication));
  task.reset();
  EXPECT_TRUE(std::filesystem::exists(old_path));
  ASSERT_NE(nullptr, old->fetch(200));
  old.reset();
  EXPECT_TRUE(std::filesystem::exists(old_path));
  ASSERT_NE(nullptr, (*reader)->fetch(200));
  reader->reset();
  EXPECT_FALSE(std::filesystem::exists(old_path));
  EXPECT_TRUE(std::filesystem::exists(new_path));
  EXPECT_TRUE(std::filesystem::exists(unchanged_path));
  auto found = (*replacement)->fts_search("text", fts::TermNode("beta"), {});
  ASSERT_TRUE(found.has_value());
  EXPECT_EQ(1, found->size());
}

TEST_P(FtsSegmentPublicationTest,
       FailedVectorCommitDiscardsCandidateAndUncertainCommitPreservesIt) {
  CollectionSchema schema = schema_;
  schema.add_field(std::make_shared<FieldSchema>(
      "vector", DataType::VECTOR_FP32, 4, false,
      std::make_shared<HnswIndexParams>(MetricType::L2)));
  auto created = Segment::CreateAndOpen(
      path_, schema, 2, 200, ids_, deletes_, versions_,
      SegmentOptions{false, GetParam(), 64 * 1024 * 1024});
  ASSERT_TRUE(created);
  auto source = *created;
  Doc doc;
  doc.set_pk("vector-candidate");
  doc.set("text", std::string("beta"));
  doc.set<std::vector<float>>("vector", std::vector<float>(4, 1.0f));
  ASSERT_TRUE(source->insert(doc).ok());
  auto sealed = source->prepare_fts_seal();
  ASSERT_TRUE(sealed);
  auto version = versions_->get_current_version();
  version.set_schema(schema);
  ASSERT_TRUE(version.add_persisted_segment_meta(sealed->metadata).ok());
  ASSERT_TRUE(versions_->commit(version).ok());
  source->publish_fts_seal(std::move(*sealed));
  const auto old_path =
      source->get_vector_indexer("vector").front()->index_file_path();
  for (bool uncertain : {false, true}) {
    auto task = SegmentTask::CreateCreateVectorIndexTask(
        {source, "", nullptr, 1, versions_->fts_retirement()});
    ASSERT_TRUE(SegmentHelper::Execute(task).ok());
    auto &built = std::get<CreateVectorIndexTask>(task->task_info());
    auto candidate = Segment::Open(
        path_, schema, *built.output_segment_meta_, ids_, deletes_, versions_,
        SegmentOptions{true, GetParam(), 64 * 1024 * 1024});
    ASSERT_TRUE(candidate);
    const auto new_path =
        (*candidate)->get_vector_indexer("vector").front()->index_file_path();
    ASSERT_NE(old_path, new_path);
    version = versions_->get_current_version();
    ASSERT_TRUE(
        version.update_persisted_segment_meta(built.output_segment_meta_).ok());
    auto status = versions_->checkpoint(
        version, [&](auto stage, const std::string &path) {
          if (path == path_ + "/manifest.current" &&
              stage == (uncertain ? DurableFile::Stage::kDirectorySync
                                  : DurableFile::Stage::kWrite)) {
            return Status::InternalError("injected vector commit failure");
          }
          return Status::OK();
        });
    ASSERT_FALSE(status.ok());
    ASSERT_EQ(uncertain, versions_->requires_recovery());
    if (uncertain) {
      built.preserve_files();
    }
    task.reset();
    // Preopened candidate readers must pin the file until they are discarded.
    EXPECT_TRUE(std::filesystem::exists(new_path));
    candidate->reset();
    EXPECT_EQ(uncertain, std::filesystem::exists(new_path));
    EXPECT_TRUE(std::filesystem::exists(old_path));
    ASSERT_NE(nullptr, source->fetch(200));
    if (uncertain) {
      auto recovered = VersionManager::Recovery(path_);
      ASSERT_TRUE(recovered);
      auto reopened = Segment::Open(
          path_, schema, *version.persisted_segment_metas().back(), ids_,
          deletes_, *recovered,
          SegmentOptions{true, GetParam(), 64 * 1024 * 1024});
      ASSERT_TRUE(reopened) << reopened.error().message();
      EXPECT_NE(nullptr, (*reopened)->fetch(200));
    }
  }
}

TEST_P(FtsSegmentPublicationTest, MultiFieldDdlKeepsOtherFieldAndOldReader) {
  const auto root = path_ + "/multi";
  ASSERT_TRUE(FileHelper::CreateDirectory(root));
  CollectionSchema schema{"multi"};
  auto params = std::make_shared<FtsIndexParams>("whitespace");
  schema.add_field(
      std::make_shared<FieldSchema>("text", DataType::STRING, true, params));
  schema.add_field(
      std::make_shared<FieldSchema>("title", DataType::STRING, true, params));
  schema.add_field(
      std::make_shared<FieldSchema>("extra", DataType::STRING, true));
  Version initial;
  initial.set_schema(schema);
  initial.set_enable_mmap(GetParam());
  auto versions = VersionManager::Create(root, initial);
  ASSERT_TRUE(versions.has_value());
  auto ids = IDMap::CreateAndOpen(
      "multi", FileHelper::MakeFilePath(root, FileID::ID_FILE, 0), true, false);
  ASSERT_NE(nullptr, ids);
  auto deletes = std::make_shared<DeleteStore>("multi");
  auto segment = Segment::CreateAndOpen(
      root, schema, 0, 100, ids, deletes, *versions,
      SegmentOptions{false, GetParam(), 64 * 1024 * 1024});
  ASSERT_TRUE(segment.has_value());
  for (int i = 0; i < 2; ++i) {
    Doc doc;
    doc.set_pk(std::to_string(i));
    doc.set("text", std::string(i == 0 ? "alpha beta" : ""));
    if (i == 0) {
      doc.set("title", std::string("title"));
    }
    doc.set("extra", std::string("New-Token"));
    ASSERT_TRUE((*segment)->insert(doc).ok());
  }
  auto prepared = (*segment)->prepare_fts_seal();
  ASSERT_TRUE(prepared.has_value());
  const auto old_path = prepared->indexer->working_dir();
  auto candidate = (*versions)->get_current_version();
  ASSERT_TRUE(candidate.add_persisted_segment_meta(prepared->metadata).ok());
  candidate.reset_writing_segment_meta(nullptr);
  ASSERT_TRUE((*versions)->commit(candidate).ok());
  (*segment)->publish_fts_seal(std::move(*prepared));
  auto old_reader = (*segment)->get_fts_indexer("text");
  fts::format::Identity title_identity;
  {
    auto title = (*segment)->open_fts_source("title");
    ASSERT_TRUE(title.has_value());
    title_identity = (*title)->metadata().incarnation;
  }
  SegmentMeta::Ptr changed;
  FtsIndexer::Ptr indexer;
  auto status = (*segment)->drop_fts_index("text", &changed, &indexer);
  ASSERT_TRUE(status.ok()) << status.message();
  schema.get_field("text")->set_index_params(nullptr);
  candidate = (*versions)->get_current_version();
  candidate.set_schema(schema);
  ASSERT_TRUE(candidate.update_persisted_segment_meta(changed).ok());
  ASSERT_TRUE((*versions)->commit(candidate).ok());
  ASSERT_TRUE((*segment)->reload_fts_index(schema, changed, indexer).ok());
  indexer.reset();
  auto title = (*segment)->open_fts_source("title");
  ASSERT_TRUE(title.has_value());
  EXPECT_EQ(title_identity, (*title)->metadata().incarnation);
  EXPECT_EQ(1, (*title)->statistics().scoring_documents);
  EXPECT_EQ(nullptr, (*segment)->get_fts_indexer("text"));
  EXPECT_TRUE(std::filesystem::exists(old_path));
  auto old = old_reader->search(fts::TermNode("beta"), {});
  ASSERT_TRUE(old.has_value());
  EXPECT_EQ(1, old->size());
  old_reader.reset();
  EXPECT_FALSE(std::filesystem::exists(old_path));
  fts::format::Identity extra_identity{};
  for (const auto &tokenizer : {"whitespace", "standard"}) {
    auto new_params = std::make_shared<FtsIndexParams>(tokenizer);
    status =
        (*segment)->create_fts_index("extra", new_params, &changed, &indexer);
    ASSERT_TRUE(status.ok()) << status.message();
    schema.get_field("extra")->set_index_params(new_params);
    candidate = (*versions)->get_current_version();
    candidate.set_schema(schema);
    ASSERT_TRUE(candidate.update_persisted_segment_meta(changed).ok());
    ASSERT_TRUE((*versions)->commit(candidate).ok());
    ASSERT_TRUE((*segment)->reload_fts_index(schema, changed, indexer).ok());
    indexer.reset();
    auto extra = (*segment)->open_fts_source("extra");
    ASSERT_TRUE(extra.has_value());
    EXPECT_NE(extra_identity, (*extra)->metadata().incarnation);
    extra_identity = (*extra)->metadata().incarnation;
    auto unchanged = (*segment)->open_fts_source("title");
    ASSERT_TRUE(unchanged.has_value());
    EXPECT_EQ(title_identity, (*unchanged)->metadata().incarnation);
    EXPECT_EQ(1, (*unchanged)->statistics().scoring_documents);
    auto found = (*segment)->fts_search("extra", fts::TermNode("new"), {});
    ASSERT_TRUE(found.has_value());
    EXPECT_EQ(std::string(tokenizer) == "standard" ? 2 : 0, found->size());
  }
}

TEST_P(FtsSegmentPublicationTest, CollectionDdlAndClosedBackupReopen) {
  ailego::File lock;
  ASSERT_TRUE(lock.create((path_ + "/LOCK").c_str(), 0));
  lock.close();
  ASSERT_TRUE(
      deletes_->flush(FileHelper::MakeFilePath(path_, FileID::DELETE_FILE, 0))
          .ok());
  auto manager = std::make_shared<SegmentManager>();
  auto status = manager->switch_fts_writer(
      &segment_, versions_,
      [&](uint64_t minimum) -> Result<SegmentManager::FtsSuccessor> {
        auto next = Segment::CreateAndOpen(
            path_, schema_, 1, minimum, ids_, deletes_, versions_,
            SegmentOptions{false, GetParam(), 64 * 1024 * 1024});
        if (!next) {
          return tl::make_unexpected(next.error());
        }
        return SegmentManager::FtsSuccessor{*next, 2};
      });
  ASSERT_TRUE(status.ok()) << status.message();
  segment_.reset();
  manager.reset();
  ids_.reset();
  versions_.reset();
  deletes_.reset();
  auto collection = Collection::Open(path_, CollectionOptions{});
  ASSERT_TRUE(collection.has_value()) << collection.error().message();
  SearchQuery query;
  query.topk_ = 10;
  query.target_.field_name_ = "text";
  FtsClause clause;
  clause.query_string_ = "beta";
  query.target_.clause_ = clause;
  auto before = (*collection)->query(query);
  ASSERT_TRUE(before.has_value());
  ASSERT_EQ(2, before->size());
  // Force the authority write to fail after candidates have been built and
  // preopened. The visible schema and old result must remain usable for retry.
  ASSERT_TRUE(
      std::filesystem::create_directory(path_ + "/manifest.current.pending"));
  status =
      (*collection)
          ->create_index("text", std::make_shared<FtsIndexParams>("standard"));
  EXPECT_FALSE(status.ok());
  auto unchanged = (*collection)->schema();
  ASSERT_TRUE(unchanged.has_value());
  EXPECT_EQ("whitespace", std::dynamic_pointer_cast<FtsIndexParams>(
                              unchanged->get_field("text")->index_params())
                              ->tokenizer_name());
  auto still_old = (*collection)->query(query);
  ASSERT_TRUE(still_old.has_value());
  ASSERT_EQ(before->size(), still_old->size());
  for (size_t i = 0; i < before->size(); ++i) {
    EXPECT_EQ((*before)[i]->pk(), (*still_old)[i]->pk());
    EXPECT_EQ((*before)[i]->score(), (*still_old)[i]->score());
  }
  ASSERT_TRUE(std::filesystem::remove(path_ + "/manifest.current.pending"));
  status =
      (*collection)
          ->create_index("text", std::make_shared<FtsIndexParams>("standard"));
  ASSERT_TRUE(status.ok()) << status.message();
  auto after = (*collection)->query(query);
  ASSERT_TRUE(after.has_value()) << after.error().message();
  ASSERT_EQ(before->size(), after->size());
  for (size_t i = 0; i < before->size(); ++i) {
    EXPECT_EQ((*before)[i]->pk(), (*after)[i]->pk());
    EXPECT_EQ((*before)[i]->score(), (*after)[i]->score());
  }
  status = (*collection)
               ->add_column(std::make_shared<FieldSchema>(
                                "number", DataType::INT32, false,
                                std::make_shared<InvertIndexParams>()),
                            "7");
  ASSERT_TRUE(status.ok()) << status.message();
  query.filter_ = "number = 7";
  for (bool drop : {true, false}) {
    ASSERT_TRUE(
        std::filesystem::create_directory(path_ + "/manifest.current.pending"));
    status =
        drop ? (*collection)->drop_column("number")
             : (*collection)
                   ->alter_column("number", "",
                                  std::make_shared<FieldSchema>(
                                      "number", DataType::INT64, false,
                                      std::make_shared<InvertIndexParams>()));
    EXPECT_FALSE(status.ok());
    auto unchanged_schema = (*collection)->schema();
    ASSERT_TRUE(unchanged_schema.has_value());
    ASSERT_NE(nullptr, unchanged_schema->get_field("number"));
    EXPECT_EQ(DataType::INT32,
              unchanged_schema->get_field("number")->data_type());
    ASSERT_TRUE((*collection)->close().ok());
    collection->reset();
    collection = Collection::Open(path_, CollectionOptions{});
    ASSERT_TRUE(collection.has_value()) << collection.error().message();
    auto preserved = (*collection)->query(query);
    ASSERT_TRUE(preserved.has_value()) << preserved.error().message();
    EXPECT_EQ(2, preserved->size());
    ASSERT_TRUE(std::filesystem::remove(path_ + "/manifest.current.pending"));
  }
  status = (*collection)
               ->alter_column("number", "",
                              std::make_shared<FieldSchema>(
                                  "number", DataType::INT64, false,
                                  std::make_shared<InvertIndexParams>()));
  ASSERT_TRUE(status.ok()) << status.message();
  auto altered = (*collection)->query(query);
  ASSERT_TRUE(altered.has_value()) << altered.error().message();
  EXPECT_EQ(2, altered->size());
  ASSERT_TRUE((*collection)->drop_column("number").ok());
  query.filter_.clear();
  ASSERT_TRUE((*collection)->drop_index("text").ok());
  auto missing = (*collection)->query(query);
  EXPECT_FALSE(missing.has_value());
  status = (*collection)
               ->create_index("text",
                              std::make_shared<FtsIndexParams>("whitespace"));
  ASSERT_TRUE(status.ok()) << status.message();
  ASSERT_TRUE((*collection)->close().ok());
  collection->reset();

  // A closed backup includes the fixed authority and its complete file set.
  // It must remain readable after the original collection is removed.
  const auto backup = path_ + ".backup";
  FileHelper::RemoveDirectory(backup);
  std::error_code error;
  std::filesystem::copy(path_, backup, std::filesystem::copy_options::recursive,
                        error);
  ASSERT_FALSE(error) << error.message();
  ASSERT_TRUE(FileHelper::RemoveDirectory(path_));
  auto reopened =
      Collection::Open(backup, CollectionOptions{true, !GetParam()});
  ASSERT_TRUE(reopened.has_value()) << reopened.error().message();
  auto restored = (*reopened)->query(query);
  ASSERT_TRUE(restored.has_value()) << restored.error().message();
  ASSERT_EQ(before->size(), restored->size());
  for (size_t i = 0; i < before->size(); ++i) {
    EXPECT_EQ((*before)[i]->pk(), (*restored)[i]->pk());
    EXPECT_EQ((*before)[i]->score(), (*restored)[i]->score());
  }
  auto recovered = VersionManager::Recovery(backup);
  ASSERT_TRUE(recovered.has_value());
  for (const auto &meta :
       (*recovered)->get_current_version().persisted_segment_metas()) {
    bool custom = false;
    for (const auto &block : meta->persisted_blocks()) {
      custom = custom || block.fts_base_.has_value();
    }
    EXPECT_TRUE(custom);
  }
  ASSERT_TRUE((*reopened)->close().ok());
  reopened->reset();
  FileHelper::RemoveDirectory(backup);
}

TEST_P(FtsSegmentPublicationTest, SourceCheckpointKeepsPinnedDeleteSnapshot) {
  ASSERT_TRUE(
      deletes_->flush(FileHelper::MakeFilePath(path_, FileID::DELETE_FILE, 0))
          .ok());
  ASSERT_TRUE(segment_->checkpoint_for_fts().ok());
  auto snapshot = versions_->get_current_version();
  ASSERT_FALSE(versions_->uses_commit_record());
  const auto old_delete = FileHelper::MakeFilePath(
      path_, FileID::DELETE_FILE, snapshot.delete_snapshot_path_suffix());
  ASSERT_TRUE(segment_->Delete("a").ok());
  auto prepared = segment_->prepare_fts_seal();
  ASSERT_TRUE(prepared.has_value()) << prepared.error().message();
  EXPECT_TRUE(std::filesystem::exists(old_delete));
  snapshot = Version{};
  EXPECT_FALSE(std::filesystem::exists(old_delete));
}

TEST_P(FtsSegmentPublicationTest,
       CustomCheckpointIncludesScalarVectorAndDeletes) {
  const auto root = path_ + "/mixed";
  ASSERT_TRUE(FileHelper::CreateDirectory(root));
  CollectionSchema schema{"mixed"};
  schema.add_field(std::make_shared<FieldSchema>(
      "text", DataType::STRING, false,
      std::make_shared<FtsIndexParams>("whitespace")));
  schema.add_field(std::make_shared<FieldSchema>(
      "number", DataType::INT32, false, std::make_shared<InvertIndexParams>()));
  schema.add_field(std::make_shared<FieldSchema>(
      "vector", DataType::VECTOR_FP32, 4, false,
      std::make_shared<FlatIndexParams>(MetricType::L2)));
  Version initial;
  initial.set_schema(schema);
  initial.set_enable_mmap(GetParam());
  auto versions = VersionManager::Create(root, initial);
  ASSERT_TRUE(versions.has_value());
  auto ids = IDMap::CreateAndOpen(
      "mixed", FileHelper::MakeFilePath(root, FileID::ID_FILE, 0), true, false);
  ASSERT_NE(nullptr, ids);
  auto deletes = std::make_shared<DeleteStore>("mixed");
  ASSERT_TRUE(
      deletes->flush(FileHelper::MakeFilePath(root, FileID::DELETE_FILE, 0))
          .ok());
  ailego::File lock;
  ASSERT_TRUE(lock.create(root + "/LOCK", 0));
  lock.close();
  auto created = Segment::CreateAndOpen(
      root, schema, 0, 100, ids, deletes, *versions,
      SegmentOptions{false, GetParam(), 64 * 1024 * 1024});
  ASSERT_TRUE(created.has_value()) << created.error().message();
  auto writing = std::move(*created);
  for (int i = 0; i < 3; ++i) {
    Doc doc;
    doc.set_pk(std::to_string(i));
    doc.set("text", std::string(i == 0 ? "alpha beta" : "beta"));
    doc.set<int32_t>("number", i);
    doc.set<std::vector<float>>("vector",
                                std::vector<float>(4, static_cast<float>(i)));
    ASSERT_TRUE(writing->insert(doc).ok());
  }
  ASSERT_TRUE(writing->Delete("0").ok());
  auto manager = std::make_shared<SegmentManager>();
  auto status = manager->switch_fts_writer(
      &writing, *versions,
      [&](uint64_t minimum) -> Result<SegmentManager::FtsSuccessor> {
        auto next = Segment::CreateAndOpen(
            root, schema, 1, minimum, ids, deletes, *versions,
            SegmentOptions{false, GetParam(), 64 * 1024 * 1024});
        if (!next) {
          return tl::make_unexpected(next.error());
        }
        return SegmentManager::FtsSuccessor{*next, 2};
      });
  ASSERT_TRUE(status.ok()) << status.message();
  writing.reset();
  manager.reset();
  ids.reset();
  versions->reset();
  deletes.reset();
  for (bool read_only : {false, true}) {
    auto collection =
        Collection::Open(root, CollectionOptions{read_only, !GetParam()});
    ASSERT_TRUE(collection.has_value()) << collection.error().message();
    SearchQuery text;
    text.topk_ = 10;
    text.target_.field_name_ = "text";
    FtsClause clause;
    clause.query_string_ = "beta";
    text.target_.clause_ = clause;
    text.filter_ = "number >= 0";
    auto result = (*collection)->query(text);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    ASSERT_EQ(2, result->size());
    for (const auto &doc : *result) {
      EXPECT_NE("0", doc->pk());
    }
    SearchQuery vector;
    vector.topk_ = 1;
    vector.target_.field_name_ = "vector";
    const std::vector<float> values(4, 1.0f);
    vector.target_.set_vector(
        std::string(reinterpret_cast<const char *>(values.data()),
                    values.size() * sizeof(float)));
    vector.filter_ = "number = 1";
    auto nearest = (*collection)->query(vector);
    ASSERT_TRUE(nearest.has_value()) << nearest.error().message();
    ASSERT_EQ(1, nearest->size());
    EXPECT_EQ("1", (*nearest)[0]->pk());
    if (!read_only) {
      EXPECT_EQ(0, (*nearest)[0]->score());
      auto added = (*collection)
                       ->add_column(std::make_shared<FieldSchema>(
                                        "plain", DataType::INT32, true),
                                    "", AddColumnOptions{});
      ASSERT_TRUE(added.ok()) << added.message();
      auto changed =
          (*collection)
              ->create_index("vector",
                             std::make_shared<HnswIndexParams>(
                                 MetricType::L2, 16, 20, QuantizeType::FP16));
      ASSERT_TRUE(changed.ok()) << changed.message();
      changed = (*collection)->drop_index("vector");
      ASSERT_TRUE(changed.ok()) << changed.message();
    }
    ASSERT_TRUE((*collection)->close().ok());
  }
}

TEST_P(FtsSegmentPublicationTest,
       UncertainCommitFreezesWritesAndKeepsBothFiles) {
  auto prepared = segment_->prepare_fts_seal();
  ASSERT_TRUE(prepared.has_value());
  auto candidate = versions_->get_current_version();
  ASSERT_TRUE(candidate.add_persisted_segment_meta(prepared->metadata).ok());
  candidate.reset_writing_segment_meta(nullptr);
  unsigned authority_syncs = 0;
  auto status =
      versions_->commit(candidate, [&](auto stage, const std::string &path) {
        if (stage == DurableFile::Stage::kDirectorySync &&
            path == path_ + "/manifest.current" && ++authority_syncs == 2) {
          return Status::InternalError("injected uncertain authority");
        }
        return Status::OK();
      });
  ASSERT_FALSE(status.ok());
  ASSERT_TRUE(versions_->requires_recovery());
  prepared->indexer->preserve_files();
  const auto candidate_path = prepared->indexer->working_dir();
  const auto source_path =
      segment_->get_fts_indexer("text")->ctx()->db_->GetName();
  Doc doc;
  doc.set_pk("c");
  doc.set("text", std::string("beta"));
  EXPECT_FALSE(segment_->insert(doc).ok());
  EXPECT_FALSE(segment_->upsert(doc).ok());
  doc.set_pk("a");
  EXPECT_FALSE(segment_->update(doc).ok());
  EXPECT_FALSE(segment_->Delete("a").ok());
  EXPECT_FALSE(segment_->Delete(100).ok());
  EXPECT_FALSE(segment_->flush().ok());
  segment_.reset();
  prepared->indexer.reset();
  EXPECT_TRUE(std::filesystem::exists(candidate_path));
  EXPECT_TRUE(std::filesystem::exists(source_path));
  auto recovered = VersionManager::Recovery(path_);
  ASSERT_TRUE(recovered.has_value());
  EXPECT_EQ(
      1, (*recovered)->get_current_version().persisted_segment_metas().size());
}

TEST_P(FtsSegmentPublicationTest,
       CheckpointSyncFailureRetainsPendingDeletesForRetry) {
  ASSERT_TRUE(segment_->flush().ok());
  std::string forward;
  for (const auto &block : segment_->meta()->persisted_blocks()) {
    if (block.type() == BlockType::SCALAR) {
      forward =
          FileHelper::MakeForwardBlockPath(path_, 0, block.id(), !GetParam());
      break;
    }
  }
  ASSERT_FALSE(forward.empty());
  ASSERT_TRUE(segment_->Delete("a").ok());
  // Keep the native reader open, but make the checkpoint's required file sync
  // fail after the new deletion snapshot has been written.
  std::error_code error;
  std::filesystem::rename(forward, forward + ".held", error);
  ASSERT_FALSE(error);
  auto failed = segment_->prepare_fts_seal();
  std::filesystem::rename(forward + ".held", forward, error);
  ASSERT_FALSE(error);
  ASSERT_FALSE(failed.has_value());
  auto retry = segment_->prepare_fts_seal();
  ASSERT_TRUE(retry.has_value()) << retry.error().message();
  auto version = versions_->get_current_version();
  EXPECT_EQ(1, version.delete_snapshot_path_suffix());
  auto recovered_deletes = DeleteStore::CreateAndLoad(
      "fts", FileHelper::MakeFilePath(path_, FileID::DELETE_FILE,
                                      version.delete_snapshot_path_suffix()));
  ASSERT_NE(nullptr, recovered_deletes);
  EXPECT_TRUE(recovered_deletes->is_deleted(100));
}

TEST_P(FtsSegmentPublicationTest,
       ForwardReopenFailureCanRetryWithoutOverwritingSourceRows) {
  const auto block = *segment_->meta()->writing_forward_block();
  const auto forward =
      FileHelper::MakeForwardBlockPath(path_, 0, block.id(), !GetParam());
  // Let the open writer finish its existing inode, then fail opening the
  // immutable reader. Restore the file before retrying the checkpoint.
  std::error_code error;
  std::filesystem::rename(forward, forward + ".held", error);
  ASSERT_FALSE(error);
  ASSERT_TRUE(std::filesystem::create_directory(forward));
  auto failed = segment_->prepare_fts_seal();
  ASSERT_TRUE(std::filesystem::remove(forward));
  std::filesystem::rename(forward + ".held", forward, error);
  ASSERT_FALSE(error);
  ASSERT_FALSE(failed.has_value());
  auto retry = segment_->prepare_fts_seal();
  ASSERT_TRUE(retry.has_value()) << retry.error().message();
  insert("c", "beta");
  retry = segment_->prepare_fts_seal();
  ASSERT_TRUE(retry.has_value()) << retry.error().message();
  auto candidate = versions_->get_current_version();
  ASSERT_TRUE(candidate.add_persisted_segment_meta(retry->metadata).ok());
  candidate.reset_writing_segment_meta(nullptr);
  ASSERT_TRUE(versions_->commit(candidate).ok());
  auto metadata = retry->metadata;
  segment_->publish_fts_seal(std::move(*retry));
  segment_.reset();
  auto reopened =
      Segment::Open(path_, schema_, *metadata, ids_, deletes_, versions_,
                    SegmentOptions{true, GetParam(), 64 * 1024 * 1024});
  ASSERT_TRUE(reopened.has_value()) << reopened.error().message();
  std::vector<int64_t> ids{0, 1, 2};
  ASSERT_TRUE((*reopened)->get_global_doc_ids(ids).ok());
  EXPECT_EQ((std::vector<int64_t>{100, 101, 102}), ids);
}

TEST_P(FtsSegmentPublicationTest,
       WriterSwitchPublishesPreparedManagerAndSuccessor) {
  SegmentManager segments;
  const auto original = segment_;
  auto old_query = original->get_fts_indexer("text");
  const auto old_path = old_query->ctx()->db_->GetName();
  auto create =
      [&](uint64_t min_doc_id) -> Result<SegmentManager::FtsSuccessor> {
    EXPECT_EQ(original, segment_);
    EXPECT_EQ(0, segments.segment_count());
    EXPECT_TRUE(original->meta()->has_writing_forward_block());
    auto next = Segment::CreateAndOpen(
        path_, schema_, 1, min_doc_id, ids_, deletes_, versions_,
        SegmentOptions{false, GetParam(), 64 * 1024 * 1024});
    if (!next) {
      return tl::make_unexpected(next.error());
    }
    return SegmentManager::FtsSuccessor{*next, 2};
  };
  auto status = segments.switch_fts_writer(&segment_, versions_, create);
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(1, segment_->id());
  EXPECT_EQ(1, segments.segment_count());
  EXPECT_EQ(original, segments.get_segments()[0]);
  EXPECT_FALSE(original->meta()->has_writing_forward_block());
  EXPECT_TRUE(std::filesystem::exists(old_path));
  old_query.reset();
  EXPECT_FALSE(std::filesystem::exists(old_path));
  insert("c", "beta");
  auto result = segment_->fts_search("text", fts::TermNode("beta"), {});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(1, result->size());
  EXPECT_EQ(1, versions_->get_current_version().writing_segment_meta()->id());
  EXPECT_EQ(2, versions_->get_current_version().next_segment_id());
}

TEST_P(FtsSegmentPublicationTest, FailedSwitchDiscardsSuccessorAndCanRetry) {
  SegmentManager segments;
  const auto original = segment_;
  unsigned next_id = 1;
  auto create =
      [&](uint64_t min_doc_id) -> Result<SegmentManager::FtsSuccessor> {
    auto next = Segment::CreateAndOpen(
        path_, schema_, next_id++, min_doc_id, ids_, deletes_, versions_,
        SegmentOptions{false, GetParam(), 64 * 1024 * 1024});
    if (!next) {
      return tl::make_unexpected(next.error());
    }
    return SegmentManager::FtsSuccessor{*next, next_id};
  };
  auto failed = segments.switch_fts_writer(
      &segment_, versions_, create, [&](auto stage, const std::string &path) {
        return stage == DurableFile::Stage::kRename &&
                       path == path_ + "/manifest.current"
                   ? Status::InternalError("injected authority failure")
                   : Status::OK();
      });
  ASSERT_FALSE(failed.ok());
  EXPECT_EQ(original, segment_);
  EXPECT_EQ(0, segments.segment_count());
  EXPECT_FALSE(std::filesystem::exists(path_ + "/1"));
  EXPECT_FALSE(versions_->requires_recovery());
  insert("c", "beta");
  auto retry = segments.switch_fts_writer(&segment_, versions_, create);
  ASSERT_TRUE(retry.ok()) << retry.message();
  EXPECT_EQ(2, segment_->id());
  EXPECT_EQ(3, segments.get_segments()[0]->doc_count_snapshot());
}

TEST_P(FtsSegmentPublicationTest,
       UncertainSwitchRetainsSuccessorAndRecoversCommittedVersion) {
  SegmentManager segments;
  auto original = segment_;
  std::string custom_path;
  unsigned authority_syncs = 0;
  auto status = segments.switch_fts_writer(
      &segment_, versions_,
      [&](uint64_t min_doc_id) -> Result<SegmentManager::FtsSuccessor> {
        auto next = Segment::CreateAndOpen(
            path_, schema_, 1, min_doc_id, ids_, deletes_, versions_,
            SegmentOptions{false, GetParam(), 64 * 1024 * 1024});
        if (!next) {
          return tl::make_unexpected(next.error());
        }
        return SegmentManager::FtsSuccessor{*next, 2};
      },
      [&](auto stage, const std::string &path) {
        if (stage == DurableFile::Stage::kRename &&
            path.find(".zfts") != std::string::npos) {
          custom_path = path;
        }
        if (stage == DurableFile::Stage::kDirectorySync &&
            path == path_ + "/manifest.current" && ++authority_syncs == 2) {
          return Status::InternalError("injected uncertain switch");
        }
        return Status::OK();
      });
  ASSERT_FALSE(status.ok());
  ASSERT_TRUE(versions_->requires_recovery());
  EXPECT_EQ(original, segment_);
  EXPECT_EQ(0, segments.segment_count());
  EXPECT_TRUE(std::filesystem::exists(path_ + "/1"));
  ASSERT_FALSE(custom_path.empty());
  EXPECT_TRUE(std::filesystem::exists(custom_path));
  Doc doc;
  doc.set_pk("c");
  doc.set("text", std::string("beta"));
  EXPECT_FALSE(segment_->insert(doc).ok());
  original.reset();
  segment_.reset();
  auto recovered = VersionManager::Recovery(path_);
  ASSERT_TRUE(recovered.has_value()) << recovered.error().message();
  auto version = (*recovered)->get_current_version();
  ASSERT_EQ(1, version.persisted_segment_metas().size());
  ASSERT_NE(nullptr, version.writing_segment_meta());
  EXPECT_EQ(1, version.writing_segment_meta()->id());
  auto reopened = Segment::Open(
      path_, schema_, *version.persisted_segment_metas()[0], ids_, deletes_,
      *recovered, SegmentOptions{true, GetParam(), 64 * 1024 * 1024});
  ASSERT_TRUE(reopened.has_value()) << reopened.error().message();
  auto found = (*reopened)->fts_search("text", fts::TermNode("beta"), {});
  ASSERT_TRUE(found.has_value());
  EXPECT_EQ(2, found->size());
  auto next = Segment::Open(
      path_, schema_, *version.writing_segment_meta(), ids_, deletes_,
      *recovered, SegmentOptions{false, GetParam(), 64 * 1024 * 1024});
  ASSERT_TRUE(next.has_value()) << next.error().message();
  EXPECT_TRUE((*next)->insert(doc).ok());
}

TEST_P(FtsSegmentPublicationTest,
       AuthoritySwitchWithoutFtsKeepsSourceOnFailure) {
  const auto root = path_ + "/without-fts";
  ASSERT_TRUE(FileHelper::CreateDirectory(root));
  CollectionSchema schema{"without_fts"};
  schema.add_field(std::make_shared<FieldSchema>("number", DataType::INT32));
  Version initial;
  initial.set_schema(schema);
  initial.set_enable_mmap(GetParam());
  auto versions = VersionManager::Create(root, initial);
  ASSERT_TRUE(versions.has_value());
  auto ids = IDMap::CreateAndOpen(
      "without_fts", FileHelper::MakeFilePath(root, FileID::ID_FILE, 0), true,
      false);
  ASSERT_NE(nullptr, ids);
  auto deletes = std::make_shared<DeleteStore>("without_fts");
  auto created = Segment::CreateAndOpen(
      root, schema, 0, 0, ids, deletes, *versions,
      SegmentOptions{false, GetParam(), 64 * 1024 * 1024});
  ASSERT_TRUE(created.has_value());
  auto writing = *created;
  Doc doc;
  doc.set_pk("one");
  doc.set<int32_t>("number", 7);
  ASSERT_TRUE(writing->insert(doc).ok());
  ASSERT_TRUE(writing->checkpoint_for_fts().ok());
  ASSERT_TRUE((*versions)->commit((*versions)->get_current_version()).ok());
  SegmentManager manager;
  SegmentID next_id = 1;
  auto successor =
      [&](uint64_t minimum) -> Result<SegmentManager::FtsSuccessor> {
    auto next = Segment::CreateAndOpen(
        root, schema, next_id++, minimum, ids, deletes, *versions,
        SegmentOptions{false, GetParam(), 64 * 1024 * 1024});
    if (!next) {
      return tl::make_unexpected(next.error());
    }
    return SegmentManager::FtsSuccessor{*next, next_id};
  };
  auto status = manager.switch_fts_writer(
      &writing, *versions, successor, [&](auto stage, const std::string &path) {
        return stage == DurableFile::Stage::kRename &&
                       path == root + "/manifest.current"
                   ? Status::InternalError("injected authority failure")
                   : Status::OK();
      });
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(0, writing->id());
  EXPECT_EQ(0, manager.segment_count());
  EXPECT_FALSE(std::filesystem::exists(root + "/1"));
  status = manager.switch_fts_writer(&writing, *versions, successor);
  ASSERT_TRUE(status.ok()) << status.message();
  ASSERT_EQ(1, manager.segment_count());
  auto recovered = VersionManager::Recovery(root);
  ASSERT_TRUE(recovered.has_value()) << recovered.error().message();
  EXPECT_EQ(writing->id(),
            (*recovered)->get_current_version().writing_segment_meta()->id());
  auto original = (*recovered)->get_current_version().persisted_segment_metas();
  ASSERT_EQ(1, original.size());
  EXPECT_EQ(1, original[0]->doc_count());
}

TEST_P(FtsSegmentPublicationTest, WalReplayRepairsInterruptedCheckpointStats) {
  ASSERT_TRUE(segment_->flush().ok());
  auto baseline = versions_->get_current_version();
  // Reproduce the durable boundary of an interrupted checkpoint: RocksDB
  // has the new row and statistics, while the authority still references the
  // previous forward checkpoint and the new document remains in its WAL.
  auto field = segment_->get_fts_indexer("text");
  ASSERT_TRUE(field->insert(2, "beta").has_value());
  ASSERT_TRUE(field->flush().has_value());
  ASSERT_TRUE(field->ctx()->flush().ok());
  field.reset();
  segment_.reset();
  auto metadata = baseline.writing_segment_meta();
  auto wal_path = FileHelper::MakeWalPath(
      path_, 0, metadata->writing_forward_block()->id());
  WalFilePtr wal;
  ASSERT_EQ(0, WalFile::CreateAndOpen(wal_path, WalOptions{0, true}, &wal));
  Doc doc;
  doc.set_pk("a");
  doc.set("text", std::string("beta"));
  // An older UPSERT WAL has no predecessor ID. Its new ID is already
  // durable, while the authoritative deletion snapshot still retains row 100.
  ASSERT_TRUE(ids_->upsert("a", 102).ok());
  doc.set_operator(Operator::UPSERT);
  auto bytes = doc.serialize();
  ASSERT_EQ(0, wal->append(std::string(bytes.begin(), bytes.end())));
  ASSERT_EQ(0, wal->close());
  wal.reset();
  auto reopened =
      Segment::Open(path_, schema_, *metadata, ids_, deletes_, versions_,
                    SegmentOptions{false, GetParam(), 64 * 1024 * 1024});
  ASSERT_TRUE(reopened.has_value()) << reopened.error().message();
  segment_ = *reopened;
  EXPECT_TRUE(deletes_->is_deleted(100));
  EXPECT_FALSE(deletes_->is_deleted(102));
  ASSERT_NE(nullptr, segment_->fetch(102));
  EXPECT_EQ(3, segment_->get_fts_indexer("text")->total_docs());
  EXPECT_EQ(4, segment_->get_fts_indexer("text")->total_tokens());
  auto prepared = segment_->prepare_fts_seal();
  ASSERT_TRUE(prepared.has_value()) << prepared.error().message();
}

TEST_P(FtsSegmentPublicationTest, DestroyWaitsForIndependentFtsQueryOwner) {
  for (bool custom : {false, true}) {
    if (!segment_) {
      auto next = Segment::CreateAndOpen(
          path_, schema_, 1, 102, ids_, deletes_, versions_,
          SegmentOptions{false, GetParam(), 64 * 1024 * 1024});
      ASSERT_TRUE(next.has_value());
      segment_ = *next;
      insert("c", "beta");
    }
    ASSERT_TRUE(segment_->flush().ok());
    if (custom) {
      auto prepared = segment_->prepare_fts_seal();
      ASSERT_TRUE(prepared.has_value());
      auto candidate = versions_->get_current_version();
      ASSERT_TRUE(
          candidate.add_persisted_segment_meta(prepared->metadata).ok());
      candidate.reset_writing_segment_meta(nullptr);
      ASSERT_TRUE(versions_->commit(candidate).ok());
      segment_->publish_fts_seal(std::move(*prepared));
    }
    auto query = segment_->get_fts_indexer("text");
    auto directory = FileHelper::MakeSegmentPath(path_, segment_->id());
    auto candidate = versions_->get_current_version();
    candidate.reset_writing_segment_meta(nullptr);
    if (custom) {
      ASSERT_TRUE(candidate.remove_persisted_segment_meta(segment_->id()).ok());
    }
    ASSERT_TRUE(versions_->commit(candidate).ok());
    ASSERT_TRUE(segment_->destroy().ok());
    segment_.reset();
    EXPECT_TRUE(std::filesystem::exists(directory));
    auto found = query->search(fts::TermNode("beta"), {});
    ASSERT_TRUE(found.has_value());
    EXPECT_EQ(custom ? 1 : 2, found->size());
    query.reset();
    EXPECT_FALSE(std::filesystem::exists(directory));
  }
}

TEST_P(FtsSegmentPublicationTest, SuccessorFailureAndRepeatedSwitch) {
  SegmentManager segments;
  auto failed = segments.switch_fts_writer(
      &segment_, versions_,
      [](uint64_t) -> Result<SegmentManager::FtsSuccessor> {
        return tl::make_unexpected(Status::InternalError("injected create"));
      });
  ASSERT_FALSE(failed.ok());
  EXPECT_FALSE(versions_->requires_recovery());
  insert("c", "beta");
  unsigned next_id = 1;
  auto create =
      [&](uint64_t min_doc_id) -> Result<SegmentManager::FtsSuccessor> {
    auto next = Segment::CreateAndOpen(
        path_, schema_, next_id++, min_doc_id, ids_, deletes_, versions_,
        SegmentOptions{false, GetParam(), 64 * 1024 * 1024});
    if (!next) {
      return tl::make_unexpected(next.error());
    }
    return SegmentManager::FtsSuccessor{*next, next_id};
  };
  auto status = segments.switch_fts_writer(&segment_, versions_, create);
  ASSERT_TRUE(status.ok()) << status.message();
  insert("d", "beta");
  ASSERT_TRUE(segment_->Delete("a").ok());
  // The authority already covers this immutable file. A second switch must
  // not reopen it for sync. Restore the name before verifying the new version.
  const auto old = segments.get_segments().front();
  std::string forward;
  for (const auto &block : old->meta()->persisted_blocks()) {
    if (block.type() == BlockType::SCALAR) {
      forward = FileHelper::MakeForwardBlockPath(path_, old->id(), block.id(),
                                                 !GetParam());
      break;
    }
  }
  ASSERT_FALSE(forward.empty());
  std::filesystem::rename(forward, forward + ".held");
  status = segments.switch_fts_writer(&segment_, versions_, create);
  std::filesystem::rename(forward + ".held", forward);
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(2, segments.segment_count());
  EXPECT_EQ(2, segment_->id());
  auto version = versions_->get_current_version();
  EXPECT_EQ(2, version.persisted_segment_metas().size());
  EXPECT_EQ(1, version.delete_snapshot_path_suffix());
  auto recovered = VersionManager::Recovery(path_);
  ASSERT_TRUE(recovered.has_value());
  EXPECT_EQ(
      2, (*recovered)->get_current_version().persisted_segment_metas().size());
}

#endif  // _WIN32

INSTANTIATE_TEST_SUITE_P(BothBackends, FtsSegmentPublicationTest,
                         testing::Values(false, true));

#if GTEST_HAS_DEATH_TEST && !defined(_WIN32)
class FtsCheckpointRecoveryDeathTest : public testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    FileHelper::RemoveDirectory(path_);
    schema_.add_field(std::make_shared<FieldSchema>(
        "text", DataType::STRING, true,
        std::make_shared<FtsIndexParams>("whitespace")));
  }
  void TearDown() override {
    FileHelper::RemoveDirectory(path_);
  }
  void check_replayed_upserts(bool fail_manifest);
  std::string path_{"fts_checkpoint_crash.fixture"};
  CollectionSchema schema_{"fts"};
};

void FtsCheckpointRecoveryDeathTest::check_replayed_upserts(
    bool fail_manifest) {
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  ASSERT_EXIT(
      {
        ailego::LoggerBroker::SetLevel(ailego::Logger::LEVEL_FATAL);
        if (ailego::MemoryLimitPool::get_instance().init(64 * 1024 * 1024) !=
                0 ||
            !FileHelper::CreateDirectory(path_)) {
          std::_Exit(1);
        }
        Version version;
        version.set_schema(schema_);
        version.set_enable_mmap(GetParam());
        auto versions = VersionManager::Create(path_, version);
        auto ids = IDMap::CreateAndOpen(
            "fts", FileHelper::MakeFilePath(path_, FileID::ID_FILE, 0), true,
            false);
        auto deletes = std::make_shared<DeleteStore>("fts");
        if (!versions || !ids) {
          std::_Exit(2);
        }
        auto segment = Segment::CreateAndOpen(
            path_, schema_, 0, 100, ids, deletes, *versions,
            SegmentOptions{false, GetParam(), 64 * 1024 * 1024});
        if (!segment) {
          std::_Exit(3);
        }
        auto baseline = (*versions)->get_current_version();
        baseline.reset_writing_segment_meta((*segment)->meta());
        if (!(*versions)->checkpoint(baseline).ok() ||
            (fail_manifest && !(*versions)->commit(baseline).ok())) {
          std::_Exit(4);
        }
        for (const auto &pk : {"a", "b", "a"}) {
          Doc doc;
          doc.set_pk(pk);
          doc.set("text",
                  std::string(pk == std::string("a") ? "alpha beta" : "beta"));
          if (!(*segment)->upsert(doc).ok()) {
            std::_Exit(5);
          }
        }
        if (fail_manifest) {
          const auto pending = path_ + "/manifest.current.pending";
          if (!std::filesystem::create_directory(pending)) {
            std::_Exit(6);
          }
          const auto checkpoint = (*segment)->flush();
          std::_Exit(!checkpoint.ok() && std::filesystem::remove(pending) ? 0
                                                                          : 7);
        }
        auto block = *(*segment)->meta()->writing_forward_block();
        auto file =
            FileHelper::MakeForwardBlockPath(path_, 0, block.id(), !GetParam());
        std::error_code error;
        std::filesystem::rename(file, file + ".held", error);
        if (error || !std::filesystem::create_directory(file)) {
          std::_Exit(6);
        }
        auto failed = (*segment)->prepare_fts_seal();
        if (failed || !std::filesystem::remove(file)) {
          std::_Exit(7);
        }
        std::filesystem::rename(file + ".held", file, error);
        std::_Exit(error ? 8 : 0);
      },
      testing::ExitedWithCode(0), "");
  ASSERT_EQ(0, ailego::MemoryLimitPool::get_instance().init(64 * 1024 * 1024));
  auto versions = VersionManager::Recovery(path_);
  ASSERT_TRUE(versions.has_value()) << versions.error().message();
  auto ids = IDMap::CreateAndOpen(
      "fts", FileHelper::MakeFilePath(path_, FileID::ID_FILE, 0), false, false);
  ASSERT_NE(nullptr, ids);
  auto deletes = std::make_shared<DeleteStore>("fts");
  auto version = (*versions)->get_current_version();
  ASSERT_NE(nullptr, version.writing_segment_meta());
  auto segment = Segment::Open(
      path_, schema_, *version.writing_segment_meta(), ids, deletes, *versions,
      SegmentOptions{false, GetParam(), 64 * 1024 * 1024});
  ASSERT_TRUE(segment.has_value()) << segment.error().message();
  auto field = (*segment)->get_fts_indexer("text");
  EXPECT_TRUE(deletes->is_deleted(100));
  EXPECT_FALSE(deletes->is_deleted(102));
  fts::FtsQueryParams params;
  params.filter = (*segment)->get_filter();
  auto hits = (*segment)->fts_search("text", fts::TermNode("beta"), params);
  ASSERT_TRUE(hits) << hits.error().message();
  ASSERT_EQ(2u, hits->size());
  EXPECT_FALSE(deletes->is_deleted(101));
  EXPECT_EQ(3, field->total_docs());
  EXPECT_EQ(5, field->total_tokens());
  auto prepared = (*segment)->prepare_fts_seal();
  ASSERT_TRUE(prepared.has_value()) << prepared.error().message();
}

TEST_P(FtsCheckpointRecoveryDeathTest,
       SourceCheckpointFailureReplaysWalWithExactFtsStatistics) {
  check_replayed_upserts(false);
}

TEST_P(FtsCheckpointRecoveryDeathTest,
       FailedAuthorityPublicationPreservesUpsertVisibility) {
  check_replayed_upserts(true);
}

INSTANTIATE_TEST_SUITE_P(BothBackends, FtsCheckpointRecoveryDeathTest,
                         testing::Values(false, true));
#endif

}  // namespace
}  // namespace zvec
