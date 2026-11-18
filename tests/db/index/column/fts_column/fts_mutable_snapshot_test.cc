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

#include "db/index/column/fts_column/fts_mutable_snapshot.h"
#include <gtest/gtest.h>
#include <zvec/core/framework/index_factory.h>
#include "db/common/constants.h"
#include "db/common/file_helper.h"
#include "db/index/column/fts_column/fts_column_indexer.h"
#include "db/index/column/fts_column/fts_rocksdb_merge.h"
#include "db/index/column/fts_column/fts_utils.h"

namespace zvec::fts {
namespace {

class FtsMutableSnapshotTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(0,
              ailego::MemoryLimitPool::get_instance().init(16 * 1024 * 1024));
    FileHelper::RemoveDirectory(path_);
    context_ = std::shared_ptr<RocksdbContext>(new RocksdbContext,
                                               [](RocksdbContext *context) {
                                                 context->close();
                                                 delete context;
                                               });
    std::vector<std::string> cfs{"text",         "text$POSITIONS",
                                 "text$TF",      "text$MAX_TF",
                                 "text$DOC_LEN", kFtsStatCfName};
    RocksdbContext::Args args{
        path_,
        cfs,
        nullptr,
        {{"text", std::make_shared<FtsPostingsMerge>()},
         {"text$MAX_TF", std::make_shared<FtsMaxTfMerge>()}}};
    ASSERT_TRUE(context_->create(args).ok());
    auto params = std::make_shared<zvec::FtsIndexParams>("whitespace");
    auto schema =
        std::make_shared<FieldSchema>("text", DataType::STRING, false, params);
    indexer_ = std::make_unique<FtsColumnIndexer>();
    ASSERT_TRUE(indexer_->open(
        schema, context_.get(), context_->get_cf("text"),
        context_->get_cf("text$POSITIONS"), context_->get_cf("text$TF"),
        context_->get_cf("text$MAX_TF"), context_->get_cf("text$DOC_LEN"),
        context_->get_cf(kFtsStatCfName), context_));
    ASSERT_TRUE(indexer_->insert(0, "alpha alpha beta"));
    ASSERT_TRUE(indexer_->insert(2, ""));
    ASSERT_TRUE(indexer_->insert(255, "a a"));
    // This term's legacy TF keys sort between the shorter "a" term's keys.
    ASSERT_TRUE(indexer_->insert(256, std::string("a\0\0 a", 5)));
    ASSERT_TRUE(indexer_->insert(259, "alpha beta"));
    ASSERT_TRUE(indexer_->flush());
    ASSERT_TRUE(context_->flush().ok());
  }
  void TearDown() override {
    indexer_.reset();
    context_.reset();
    FileHelper::RemoveDirectory(path_);
    std::remove("mutable_custom.zfts");
    std::remove("mutable_retry.zfts");
  }
  Result<format::FieldMetadata> write(const FtsMutableSnapshot &snapshot,
                                      const std::string &path) {
    auto dumper = core::IndexFactory::CreateDumper("FileDumper");
    if (dumper->create(path) != 0) {
      return tl::make_unexpected(Status::InternalError("Test create failed"));
    }
    auto container = FtsContainerWriter::Create(dumper);
    if (!container) {
      return tl::make_unexpected(container.error());
    }
    format::FieldMetadata identity;
    identity.name = "text";
    identity.id[0] = 1;
    identity.incarnation[0] = 2;
    identity.analysis_identity = "whitespace+lowercase";
    auto field = snapshot.write_field(identity, {}, container->get(), 0);
    if (!field) {
      return field;
    }
    format::Metadata metadata;
    metadata.base_id[0] = 3;
    metadata.base_generation = 1;
    metadata.row_count = 260;
    metadata.fields.push_back(*field);
    auto status = (*container)->finish(metadata);
    if (!status.ok()) {
      return tl::make_unexpected(status);
    }
    return field;
  }
  std::string path_{"fts_mutable_snapshot"};
  std::shared_ptr<RocksdbContext> context_;
  std::unique_ptr<FtsColumnIndexer> indexer_;
};

TEST_F(FtsMutableSnapshotTest,
       DirectConversionPreservesSnapshotAndBinaryTerms) {
  std::string before;
  ASSERT_TRUE(context_->db_
                  ->Get(context_->read_opts_, context_->get_cf("text"), "alpha",
                        &before)
                  .ok());
  auto snapshot = FtsMutableSnapshot::Capture(context_, 260);
  ASSERT_TRUE(snapshot);
  ASSERT_TRUE(indexer_->insert(257, "later later"));
  ASSERT_TRUE(indexer_->flush());
  auto written = write(**snapshot, "mutable_custom.zfts");
  ASSERT_TRUE(written) << written.error();
  EXPECT_EQ(5u, written->present_count);
  EXPECT_EQ(5u, written->scoring_document_count);
  EXPECT_EQ(9u, written->total_tokens);
  EXPECT_EQ(4u, written->term_count);
  for (bool mmap : {false, true}) {
    auto file = FtsContainerReader::Open("mutable_custom.zfts", mmap);
    ASSERT_TRUE(file);
    auto reader = FtsFieldReader::Open(*file, 0);
    ASSERT_TRUE(reader);
    ASSERT_TRUE((*reader)->verify().ok());
    EXPECT_FALSE((*reader)->document(1)->present);
    EXPECT_TRUE((*reader)->document(2)->present);
    EXPECT_EQ(0u, (*reader)->document(2)->length);
    EXPECT_EQ(2u, (*reader)->document(256)->length);
    EXPECT_FALSE((*reader)->document(257)->present);
    EXPECT_EQ(StatusCode::NOT_FOUND, (*reader)->lookup("later").error().code());
    auto binary = (*reader)->lookup(std::string("a\0\0", 3));
    ASSERT_TRUE(binary);
    auto postings = (*reader)->postings(*binary);
    ASSERT_TRUE(postings);
    auto posting = (*postings)->next();
    ASSERT_TRUE(posting);
    ASSERT_TRUE(*posting);
    EXPECT_EQ(256u, (**posting).doc);
    EXPECT_EQ(1u, (**posting).tf);
    auto positions = (*postings)->positions();
    ASSERT_TRUE(positions);
    EXPECT_EQ(0u, (*positions)->next().value().value());
  }
  std::string after;
  ASSERT_TRUE(
      context_->db_
          ->Get(context_->read_opts_, context_->get_cf("text"), "alpha", &after)
          .ok());
  EXPECT_EQ(before, after);
  EXPECT_NE(nullptr, context_->get_cf("text$TF"));
  auto retry = FtsMutableSnapshot::Capture(context_, 260);
  ASSERT_TRUE(retry);
  auto latest = write(**retry, "mutable_retry.zfts");
  ASSERT_TRUE(latest) << latest.error();
  EXPECT_EQ(6u, latest->scoring_document_count);
  EXPECT_EQ(11u, latest->total_tokens);
}

TEST_F(FtsMutableSnapshotTest, MalformedPositionFailsWithoutChangingSource) {
  auto *positions = context_->get_cf("text$POSITIONS");
  const auto key = make_doc_term_key("alpha", 0);
  std::string original;
  ASSERT_TRUE(
      context_->db_->Get(context_->read_opts_, positions, key, &original).ok());
  ASSERT_TRUE(context_->db_
                  ->Put(context_->write_opts_, positions, key,
                        std::string(1, char{0x7f}))
                  .ok());
  auto snapshot = FtsMutableSnapshot::Capture(context_, 260);
  ASSERT_TRUE(snapshot);
  auto result = write(**snapshot, "mutable_custom.zfts");
  ASSERT_FALSE(result);
  EXPECT_EQ(StatusCode::INTERNAL_ERROR, result.error().code());
  ASSERT_TRUE(
      context_->db_->Put(context_->write_opts_, positions, key, original).ok());
  auto retry = FtsMutableSnapshot::Capture(context_, 260);
  ASSERT_TRUE(retry);
  EXPECT_TRUE(write(**retry, "mutable_retry.zfts"));
  EXPECT_NE(nullptr, context_->get_cf("text$DOC_LEN"));
}

}  // namespace
}  // namespace zvec::fts
