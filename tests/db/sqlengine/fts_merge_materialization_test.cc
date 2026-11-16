// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include "db/common/file_helper.h"
#include "zvec/db/collection.h"
#include "zvec/db/index_params.h"

namespace zvec::sqlengine {
namespace {

TEST(FtsMergeMaterializationTest, GroupedAndInterleavedRanksMatchMmap) {
  ASSERT_EQ(0, ailego::MemoryLimitPool::get_instance().init(2UL << 30));
  const std::string path = "fts_merge_materialization_test_collection";
  std::vector<std::vector<Doc::Ptr>> expected;
  for (const bool mmap : {true, false}) {
    FileHelper::RemoveDirectory(path);
    auto schema = std::make_shared<CollectionSchema>("fts_merge_order");
    schema->set_max_doc_count_per_segment(1000);
    for (const auto &field : {"tied", "content"})
      ASSERT_TRUE(schema
                      ->add_field(std::make_shared<FieldSchema>(
                          field, DataType::STRING, false,
                          std::make_shared<FtsIndexParams>()))
                      .ok());
    ASSERT_TRUE(schema
                    ->add_field(std::make_shared<FieldSchema>(
                        "payload", DataType::STRING, true))
                    .ok());
    auto created = Collection::CreateAndOpen(path, *schema,
                                             CollectionOptions{false, mmap});
    ASSERT_TRUE(created) << created.error().message();
    auto collection = std::move(*created);
    const std::string content[] = {"needle needle needle",
                                   "needle hay hay hay hay hay hay hay hay",
                                   "needle needle", "hay"};
    for (int batch = 0; batch < 2; ++batch) {
      std::vector<Doc> docs;
      // A persisted segment requires at least 1000 rows; only four match.
      for (int row = 0; row < 1000; ++row) {
        const int key = batch * 1000 + row;
        Doc doc;
        doc.set_pk(std::to_string(key));
        doc.set<std::string>("tied", row < 4 ? "same" : "other");
        doc.set<std::string>("content", row < 4 ? content[row] : "hay");
        if (key % 3)
          doc.set<std::string>("payload", "payload" + std::to_string(key));
        docs.push_back(std::move(doc));
      }
      ASSERT_TRUE(collection->insert(docs));
      ASSERT_TRUE(collection->flush().ok());
    }
    ASSERT_TRUE(collection->close().ok());
    collection.reset();
    auto opened = Collection::Open(path, CollectionOptions{true, mmap});
    ASSERT_TRUE(opened) << opened.error().message();
    collection = std::move(*opened);
    for (size_t test = 0; test < 3; ++test) {
      SearchQuery query;
      query.target_.field_name_ = test == 2 ? "content" : "tied";
      FtsClause fts;
      fts.query_string_ = test == 2 ? "needle" : "same";
      query.target_.clause_ = fts;
      query.topk_ = test == 1 ? 1 : 8;
      query.include_vector_ = false;
      query.output_fields_ = std::vector<std::string>{"payload"};
      auto result = collection->query(query);
      ASSERT_TRUE(result) << result.error().message();
      ASSERT_EQ(result->size(), test == 1 ? 1u : test == 2 ? 6u : 8u);
      size_t switches = 0;
      for (size_t row = 1; row < result->size(); ++row) {
        EXPECT_GE((*result)[row - 1]->score(), (*result)[row]->score());
        switches += std::stoi((*result)[row - 1]->pk()) / 1000 !=
                    std::stoi((*result)[row]->pk()) / 1000;
      }
      if (test == 0) EXPECT_EQ(switches, 1u);
      if (test == 2) EXPECT_GT(switches, 1u);
      if (mmap) {
        expected.push_back(std::move(*result));
      } else {
        for (size_t row = 0; row < result->size(); ++row) {
          const auto &actual = (*result)[row];
          const auto &reference = expected[test][row];
          EXPECT_EQ(actual->pk(), reference->pk());
          EXPECT_FLOAT_EQ(actual->score(), reference->score());
          EXPECT_EQ(actual->get<std::string>("payload"),
                    reference->get<std::string>("payload"));
        }
      }
    }
    ASSERT_TRUE(collection->close().ok());
    collection.reset();
  }
  expected.clear();
  FileHelper::RemoveDirectory(path);
}

}  // namespace
}  // namespace zvec::sqlengine
