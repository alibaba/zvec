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

#include "zvec/db/collection.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include <ailego/io/file.h>
#include <ailego/utility/file_helper.h>
#include <gtest/gtest.h>
#include "ailego/logger/logger.h"
#include "db/common/file_helper.h"
#include "db/index/common/type_helper.h"
#include "index/utils/utils.h"
#include "zvec/ailego/utility/float_helper.h"
#include "zvec/db/config.h"
#include "zvec/db/doc.h"
#include "zvec/db/index_params.h"
#include "zvec/db/options.h"
#include "zvec/db/schema.h"
#include "zvec/db/status.h"
#include "zvec/db/type.h"

using namespace zvec;
using namespace zvec::test;

std::string col_path = "test_collection";

class CollectionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    FileHelper::RemoveDirectory(col_path);
  }

  void TearDown() override {}
};

TEST_F(CollectionTest, Feature_CreateAndOpen_General) {
  CollectionOptions options;
  options.read_only_ = false;
  options.enable_mmap_ = true;

  std::string path = "./demo";

  ailego::FileHelper::RemoveDirectory(path.c_str());

  auto schema = TestHelper::CreateNormalSchema();
  auto result = Collection::CreateAndOpen(path, *schema, options);
  if (!result.has_value()) {
    std::cout << result.error().message() << std::endl;
  }
  ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(ailego::FileHelper::IsExist(path.c_str()));

  auto col = result.value();
  ASSERT_EQ(col->Path(), path);
  ASSERT_EQ(col->Schema(), *schema);
  ASSERT_EQ(col->Options(), options);
  auto stats = col->Stats().value();
  ASSERT_TRUE(stats.doc_count == 0);
  ASSERT_EQ(stats.index_completeness["dense_fp32"], 1);
  ASSERT_EQ(stats.index_completeness["dense_fp16"], 1);
  // ASSERT_EQ(stats.index_completeness["dense_fp64"], 1);
  ASSERT_EQ(stats.index_completeness["sparse_fp32"], 1);
  ASSERT_EQ(stats.index_completeness["sparse_fp16"], 1);

  ASSERT_EQ(col->Destroy(), Status::OK());

  // after destroyed, every interface should return error
  std::vector<Doc> empty_docs;
  ASSERT_FALSE(col->Insert(empty_docs).has_value());
  ASSERT_FALSE(col->Update(empty_docs).has_value());
  ASSERT_FALSE(col->Delete({}).has_value());
  ASSERT_FALSE(col->DeleteByFilter("").ok());
  ASSERT_FALSE(col->Fetch({}).has_value());
  ASSERT_FALSE(col->Query({}).has_value());
  ASSERT_FALSE(col->GroupByQuery({}).has_value());
  ASSERT_FALSE(col->CreateIndex("", nullptr).ok());
  ASSERT_FALSE(col->DropIndex("").ok());
  ASSERT_FALSE(col->AddColumn("", nullptr, "").ok());
  ASSERT_FALSE(col->AlterColumn("", "", nullptr).ok());
  ASSERT_FALSE(col->DropColumn("").ok());
  ASSERT_FALSE(col->CreateIndex("", nullptr).ok());
  ASSERT_FALSE(col->Optimize().ok());
  ASSERT_FALSE(col->Flush().ok());
  ASSERT_FALSE(col->Destroy().ok());
  ASSERT_FALSE(col->Options().has_value());
  ASSERT_FALSE(col->Path().has_value());
  ASSERT_FALSE(col->Stats().has_value());
  ASSERT_FALSE(col->Schema().has_value());

  ASSERT_FALSE(ailego::FileHelper::IsExist(path.c_str()));

  // recreate
  result = Collection::CreateAndOpen(path, *schema, options);
  ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(ailego::FileHelper::IsExist(path.c_str()));

  col = std::move(result.value());
  col.reset();
  col = nullptr;

  ASSERT_TRUE(ailego::FileHelper::IsExist(path.c_str()));

  // reopen
  result = Collection::Open(path, options);
  ASSERT_TRUE(result.has_value());
  col = std::move(result.value());
  col.reset();

  // reopen with read-only
  options.read_only_ = true;
  result = Collection::Open(path, options);
  if (!result.has_value()) {
    std::cout << result.error().message() << std::endl;
  }
  ASSERT_TRUE(result.has_value());
  col = result.value();

  ASSERT_EQ(col->Path(), path);
  ASSERT_EQ(col->Schema(), *schema);
  ASSERT_EQ(col->Options(), options);
  stats = col->Stats().value();
  ASSERT_TRUE(stats.doc_count == 0);
  ASSERT_EQ(stats.index_completeness["dense_fp32"], 1);
  ASSERT_EQ(stats.index_completeness["dense_fp16"], 1);
  // ASSERT_EQ(stats.index_completeness["dense_fp64"], 1);
  ASSERT_EQ(stats.index_completeness["sparse_fp32"], 1);
  ASSERT_EQ(stats.index_completeness["sparse_fp16"], 1);

  // when open with read-only, write operation should fail
  ASSERT_FALSE(col->Flush().ok());
  ASSERT_FALSE(col->Destroy().ok());
  ASSERT_FALSE(col->Insert(empty_docs).has_value());
  ASSERT_FALSE(col->Update(empty_docs).has_value());
  ASSERT_FALSE(col->Delete({}).has_value());
  ASSERT_FALSE(col->DeleteByFilter("").ok());
  ASSERT_FALSE(col->CreateIndex("", nullptr).ok());
  ASSERT_FALSE(col->DropIndex("").ok());
  ASSERT_FALSE(col->AddColumn("", nullptr, "").ok());
  ASSERT_FALSE(col->AlterColumn("", "", nullptr).ok());
  ASSERT_FALSE(col->DropColumn("").ok());
  ASSERT_FALSE(col->CreateIndex("", nullptr).ok());
  ASSERT_FALSE(col->Optimize().ok());

  // two threads open with read_only
  result = Collection::Open(path, options);
  if (!result.has_value()) {
    std::cout << result.error().message() << std::endl;
  }
  ASSERT_TRUE(result.has_value());
  col = result.value();

  auto result1 = Collection::Open(path, options);
  if (!result1.has_value()) {
    std::cout << result1.error().message() << std::endl;
  }
  ASSERT_TRUE(result1.has_value());
  auto col1 = result1.value();
}

TEST_F(CollectionTest, Feature_CreateAndOpen_Empty) {
  int doc_count = 0;
  int loop_count = 100;

  // create with normal schema
  auto schema = TestHelper::CreateNormalSchema();
  auto options = CollectionOptions{false, true, 100 * 1024 * 1024};

  // Initial creation and insertion of 1000 docs
  auto collection = TestHelper::CreateCollectionWithDoc(
      col_path, *schema, options, 0, doc_count, false);

  ASSERT_NE(collection, nullptr);

  // Close and reopen, then insert 1 doc - repeat 100 times
  for (int i = 0; i < loop_count; i++) {
    // Close collection
    collection.reset();

    // Reopen collection
    auto result = Collection::Open(col_path, options);
    ASSERT_TRUE(result.has_value())
        << "Failed to reopen collection at iteration " << i;
    collection = std::move(result.value());

    // Verify total doc count
    auto stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, 0);
  }
}

TEST_F(CollectionTest, Feature_CreateAndOpen_PathValidate) {
  CollectionOptions options;
  options.read_only_ = false;
  options.enable_mmap_ = true;
  auto schema = TestHelper::CreateNormalSchema();

  {
    std::vector<std::string> valid_paths = {"abc",
                                            "data123",
                                            "my_collection",
                                            "v1.2_alpha-beta",
                                            ".hidden",
                                            "file.txt",
                                            "/tmp/absolute/path",
                                            "/tmp/a/b/c",
                                            "_",
                                            "-",
                                            "./tmp"};
    for (auto path : valid_paths) {
      ailego::FileHelper::RemoveDirectory(path.c_str());

      auto result = Collection::CreateAndOpen(path, *schema, options);
      if (!result.has_value()) {
        std::cout << result.error().message() << std::endl;
      }
      ASSERT_TRUE(result.has_value());
    }
  }

  {
    std::vector<std::string> inalid_paths = {
        " ",         "",
        "file name",  // space
        "file$name",  // $
        "a&b",        // &
        "a|b",        // |
        "a<b",        // <
        "a>b",        // >
        "a\"b",       // "
        "a'b",        // '
        "a;b",        // ;
        "a?b",        // ?
        "a*b",        // *
        "a[b]",       // []
        "a{b}",       // {}
        "a\\b",       //
        "a~b",        // ~
        "a#b",        // #
        "a\tb",       // tab
        "a\nb",       // newline
        "a\rb",       // carriage return
    };
    for (auto path : inalid_paths) {
      ailego::FileHelper::RemoveDirectory(path.c_str());

      auto result = Collection::CreateAndOpen(path, *schema, options);
      if (!result.has_value()) {
        std::cout << result.error().message() << std::endl;
      }
      ASSERT_FALSE(result.has_value());
    }
  }
}

TEST_F(CollectionTest, Feature_CreateAndOpen_Repeated) {
  int doc_count = 1000;
  int loop_count = 100;

  // create with normal schema
  auto schema = TestHelper::CreateNormalSchema();
  auto options = CollectionOptions{false, true, 100 * 1024 * 1024};

  // Initial creation and insertion of 1000 docs
  auto collection = TestHelper::CreateCollectionWithDoc(
      col_path, *schema, options, 0, doc_count, false);

  ASSERT_NE(collection, nullptr);

  // Close and reopen, then insert 1 doc - repeat 100 times
  for (int i = 0; i < loop_count; i++) {
    // Close collection
    collection.reset();

    // Reopen collection
    auto result = Collection::Open(col_path, options);
    ASSERT_TRUE(result.has_value())
        << "Failed to reopen collection at iteration " << i;
    collection = std::move(result.value());

    // Insert 1 additional doc
    auto s = TestHelper::CollectionInsertDoc(collection, doc_count + i,
                                             doc_count + i + 1, false);
    ASSERT_TRUE(s.ok()) << "Failed to insert doc at iteration " << i;

    // Verify total doc count
    auto stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, doc_count + i + 1)
        << "Document count mismatch at iteration " << i;
  }

  // Final verification - check all docs are present
  for (int i = 0; i < doc_count + loop_count; i++) {
    auto expect_doc = TestHelper::CreateDoc(i, *schema);
    auto result = collection->Fetch({expect_doc.pk()});
    ASSERT_TRUE(result.has_value()) << "Failed to fetch doc " << i;
    ASSERT_EQ(result.value().size(), 1);
    ASSERT_EQ(result.value().count(expect_doc.pk()), 1);
    auto doc = result.value()[expect_doc.pk()];
    if (doc == nullptr) {
      std::cout << "fetch failed, doc_id: " << i << std::endl;
    }
    ASSERT_NE(doc, nullptr);
    if (*doc != expect_doc) {
      std::cout << "       doc:" << doc->to_detail_string() << std::endl;
      std::cout << "expect_doc:" << expect_doc.to_detail_string() << std::endl;
    }
    ASSERT_EQ(*doc, expect_doc);
  }

  // Clean up
  ASSERT_TRUE(collection->Destroy().ok());
}

TEST_F(CollectionTest, Feature_CreateAndOpen_MultiThread) {
  int doc_count = 0;

  // create with normal schema
  auto schema = TestHelper::CreateNormalSchema();
  auto options = CollectionOptions{false, true, 100 * 1024 * 1024};

  // Initial creation and insertion of 1000 docs
  auto collection = TestHelper::CreateCollectionWithDoc(
      col_path, *schema, options, 0, doc_count, false);
  ASSERT_NE(collection, nullptr);
  collection.reset();

  options.read_only_ = true;
  std::atomic<bool> has_error{false};
  auto open_readonly = [&]() {
    auto coll = Collection::Open(col_path, options);
    if (!coll.has_value()) {
      LOG_ERROR("Failed to reopen collection: %s", coll.error().c_str());
      has_error.store(true);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  };
  std::vector<std::thread> threads;
  for (int i = 0; i < 10; i++) {
    threads.emplace_back(open_readonly);
  }
  for (auto &t : threads) {
    t.join();
  }
  ASSERT_FALSE(has_error.load());
}

TEST_F(CollectionTest, Feature_Write_Batch_Validate) {
  FileHelper::RemoveDirectory(col_path);

  // create with normal schema
  auto schema = TestHelper::CreateNormalSchema(false);
  auto options = CollectionOptions{false, true, 100 * 1024 * 1024};
  auto collection = TestHelper::CreateCollectionWithDoc(col_path, *schema,
                                                        options, 0, 0, false);

  auto stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, 0);
  ASSERT_EQ(stats.index_completeness["dense_fp32"], 1);


  // insert batch docs
  auto insert_normal_status =
      TestHelper::CollectionInsertDoc(collection, 0, 1024, false, false, true);
  ASSERT_TRUE(insert_normal_status.ok());

  auto insert_exceed_status =
      TestHelper::CollectionInsertDoc(collection, 0, 1025, false, false, true);
  ASSERT_FALSE(insert_exceed_status.ok());

  // upsert batch docs
  auto upsert_normal_status =
      TestHelper::CollectionUpsertDoc(collection, 0, 1024, false, true);
  ASSERT_TRUE(upsert_normal_status.ok());

  auto upsert_exceed_status =
      TestHelper::CollectionUpsertDoc(collection, 0, 1025, false, true);
  ASSERT_FALSE(upsert_exceed_status.ok());
}

TEST_F(CollectionTest, Feature_Insert_General) {
  auto func = [&](bool schema_nullable, bool doc_nullable,
                  int doc_count = 1000) {
    FileHelper::RemoveDirectory(col_path);

    // create with normal schema
    auto schema = TestHelper::CreateNormalSchema(schema_nullable);
    auto options = CollectionOptions{false, true, 100 * 1024 * 1024};
    auto collection = TestHelper::CreateCollectionWithDoc(
        col_path, *schema, options, 0, doc_count, doc_nullable);


    if (!schema_nullable && doc_nullable) {
      ASSERT_EQ(collection, nullptr);
      return;
    } else {
      ASSERT_NE(collection, nullptr);
    }

    auto stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, doc_count);
    ASSERT_EQ(stats.index_completeness["dense_fp32"], 1);
    ASSERT_EQ(stats.index_completeness["dense_fp16"], 1);
    // ASSERT_EQ(stats.index_completeness["dense_fp64"], 1);
    ASSERT_EQ(stats.index_completeness["sparse_fp32"], 1);
    ASSERT_EQ(stats.index_completeness["sparse_fp16"], 1);

    // validate fetch result
    for (int i = 0; i < doc_count; i++) {
      auto expect_doc = doc_nullable ? TestHelper::CreateDocNull(i, *schema)
                                     : TestHelper::CreateDoc(i, *schema);
      auto result = collection->Fetch({expect_doc.pk()});
      ASSERT_TRUE(result.has_value());
      ASSERT_EQ(result.value().size(), 1);
      ASSERT_EQ(result.value().count(expect_doc.pk()), 1);
      auto doc = result.value()[expect_doc.pk()];
      ASSERT_NE(doc, nullptr);
      if (*doc != expect_doc) {
        std::cout << "       doc:" << doc->to_detail_string() << std::endl;
        std::cout << "expect_doc:" << expect_doc.to_detail_string()
                  << std::endl;
      }
      ASSERT_EQ(*doc, expect_doc);
    }

    ASSERT_TRUE(collection->Flush().ok());

    ASSERT_NE(collection, nullptr);

    collection.reset();
    // Reopen collection
    auto result = Collection::Open(col_path, options);
    ASSERT_TRUE(result.has_value());
    collection = std::move(result.value());

    // insert another 1000 docs
    auto s = TestHelper::CollectionInsertDoc(collection, doc_count,
                                             doc_count * 2, doc_nullable);
    ASSERT_TRUE(s.ok());

    // validate fetch result
    for (int i = 0; i < doc_count * 2; i++) {
      auto expect_doc = doc_nullable ? TestHelper::CreateDocNull(i, *schema)
                                     : TestHelper::CreateDoc(i, *schema);
      auto result = collection->Fetch({expect_doc.pk()});
      ASSERT_TRUE(result.has_value());
      ASSERT_EQ(result.value().size(), 1);
      ASSERT_EQ(result.value().count(expect_doc.pk()), 1);
      auto doc = result.value()[expect_doc.pk()];
      ASSERT_NE(doc, nullptr);
      if (*doc != expect_doc) {
        std::cout << "       doc:" << doc->to_detail_string() << std::endl;
        std::cout << "expect_doc:" << expect_doc.to_detail_string()
                  << std::endl;
      }
      ASSERT_EQ(*doc, expect_doc);
    }

    stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, doc_count * 2);
    ASSERT_EQ(stats.index_completeness["dense_fp32"], 1);

    ASSERT_EQ(stats.index_completeness["dense_fp16"], 1);
    // ASSERT_EQ(stats.index_completeness["dense_fp64"], 1);
    ASSERT_EQ(stats.index_completeness["sparse_fp32"], 1);
    ASSERT_EQ(stats.index_completeness["sparse_fp16"], 1);
  };

  func(false, false);
  func(true, true);
  func(true, false);
  func(false, true);

  func(false, false, 0);
  func(false, false, 1);
  func(false, false, 2);
}

TEST_F(CollectionTest, Feature_Insert_ScalarIndex) {
  auto func = [&](bool nullable, bool enable_optimize, bool doc_nullable) {
    std::cout << "**** TEST INFO: nullable: " << nullable
              << ", enable_optimize: " << enable_optimize
              << ", doc_nullable: " << doc_nullable << std::endl;

    int doc_count = 1000;
    // create with normal schema
    auto schema =
        TestHelper::CreateSchemaWithScalarIndex(nullable, enable_optimize);
    auto options = CollectionOptions{false, true, 100 * 1024 * 1024};
    FileHelper::RemoveDirectory(col_path);
    auto collection = TestHelper::CreateCollectionWithDoc(
        col_path, *schema, options, 0, doc_count, doc_nullable);

    if (!nullable && doc_nullable) {
      ASSERT_EQ(collection, nullptr);
      return;
    } else {
      ASSERT_NE(collection, nullptr);
    }

    for (int i = 0; i < doc_count; i++) {
      auto expect_doc = doc_nullable ? TestHelper::CreateDocNull(i, *schema)
                                     : TestHelper::CreateDoc(i, *schema);
      auto result = collection->Fetch({expect_doc.pk()});
      ASSERT_TRUE(result.has_value());
      ASSERT_EQ(result.value().size(), 1);
      ASSERT_EQ(result.value().count(expect_doc.pk()), 1);
      auto doc = result.value()[expect_doc.pk()];
      ASSERT_NE(doc, nullptr);
      if (*doc != expect_doc) {
        std::cout << "       doc:" << doc->to_detail_string() << std::endl;
        std::cout << "expect_doc:" << expect_doc.to_detail_string()
                  << std::endl;
      }
      ASSERT_EQ(*doc, expect_doc);
    }

    ASSERT_TRUE(collection->Flush().ok());

    ASSERT_NE(collection, nullptr);

    auto stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, doc_count);
    ASSERT_EQ(stats.index_completeness["dense_fp32"], 1);

    // validate fetch result
    for (int i = 0; i < doc_count; i++) {
      auto expect_doc = doc_nullable ? TestHelper::CreateDocNull(i, *schema)
                                     : TestHelper::CreateDoc(i, *schema);
      auto result = collection->Fetch({expect_doc.pk()});
      ASSERT_TRUE(result.has_value());
      ASSERT_EQ(result.value().size(), 1);
      ASSERT_EQ(result.value().count(expect_doc.pk()), 1);
      auto doc = result.value()[expect_doc.pk()];
      ASSERT_NE(doc, nullptr);
      if (*doc != expect_doc) {
        std::cout << "       doc:" << doc->to_detail_string() << std::endl;
        std::cout << "expect_doc:" << expect_doc.to_detail_string()
                  << std::endl;
      }
      ASSERT_EQ(*doc, expect_doc);
    }

    // insert another 1000 docs
    auto s = TestHelper::CollectionInsertDoc(collection, doc_count,
                                             doc_count * 2, doc_nullable);
    ASSERT_TRUE(s.ok());
    ASSERT_TRUE(collection->Flush().ok());

    // validate fetch result
    for (int i = 0; i < doc_count * 2; i++) {
      auto expect_doc = doc_nullable ? TestHelper::CreateDocNull(i, *schema)
                                     : TestHelper::CreateDoc(i, *schema);
      auto result = collection->Fetch({expect_doc.pk()});
      ASSERT_TRUE(result.has_value());
      ASSERT_EQ(result.value().size(), 1);
      ASSERT_EQ(result.value().count(expect_doc.pk()), 1);
      auto doc = result.value()[expect_doc.pk()];
      ASSERT_NE(doc, nullptr);
      if (*doc != expect_doc) {
        std::cout << "       doc:" << doc->to_detail_string() << std::endl;
        std::cout << "expect_doc:" << expect_doc.to_detail_string()
                  << std::endl;
      }
      ASSERT_EQ(*doc, expect_doc);
    }

    stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, doc_count * 2);
    ASSERT_EQ(stats.index_completeness["dense_fp32"], 1);
  };

  func(false, false, false);
  func(false, true, false);
  func(false, false, true);
  func(true, false, true);
  func(true, false, false);
}

TEST_F(CollectionTest, Feature_Insert_VectorIndex) {
  auto func = [&](MetricType metric_type = MetricType::IP,
                  QuantizeType quantize_type = QuantizeType::UNDEFINED) {
    int doc_count = 1000;
    // create with normal schema
    auto schema = TestHelper::CreateSchemaWithVectorIndex(
        false, "demo",
        std::make_shared<HnswIndexParams>(metric_type, 16, 20, quantize_type));
    std::cout << "init schema: " << schema->to_string_formatted() << std::endl;

    auto options = CollectionOptions{false, true, 100 * 1024 * 1024};
    FileHelper::RemoveDirectory(col_path);
    auto collection = TestHelper::CreateCollectionWithDoc(
        col_path, *schema, options, 0, doc_count, false);

    // validate fetch result
    for (int i = 0; i < doc_count; i++) {
      auto expect_doc = TestHelper::CreateDoc(i, *schema);
      auto result = collection->Fetch({expect_doc.pk()});
      ASSERT_TRUE(result.has_value());
      ASSERT_EQ(result.value().size(), 1);
      ASSERT_EQ(result.value().count(expect_doc.pk()), 1);
      auto doc = result.value()[expect_doc.pk()];
      ASSERT_NE(doc, nullptr);
      if (metric_type != MetricType::COSINE) {
        if (*doc != expect_doc) {
          std::cout << "       doc:" << doc->to_detail_string() << std::endl;
          std::cout << "expect_doc:" << expect_doc.to_detail_string()
                    << std::endl;
        }
        ASSERT_EQ(*doc, expect_doc);
      }
    }

    ASSERT_TRUE(collection->Flush().ok());

    ASSERT_NE(collection, nullptr);

    collection.reset();
    // Reopen collection
    auto result = Collection::Open(col_path, options);
    ASSERT_TRUE(result.has_value());
    collection = std::move(result.value());

    auto stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, doc_count);
    ASSERT_EQ(stats.index_completeness["dense_fp32"], 0);

    // validate fetch result
    for (int i = 0; i < doc_count; i++) {
      auto expect_doc = TestHelper::CreateDoc(i, *schema);
      auto result = collection->Fetch({expect_doc.pk()});
      ASSERT_TRUE(result.has_value());
      ASSERT_EQ(result.value().size(), 1);
      ASSERT_EQ(result.value().count(expect_doc.pk()), 1);
      auto doc = result.value()[expect_doc.pk()];
      ASSERT_NE(doc, nullptr);
      if (metric_type != MetricType::COSINE) {
        if (*doc != expect_doc) {
          std::cout << "       doc:" << doc->to_detail_string() << std::endl;
          std::cout << "expect_doc:" << expect_doc.to_detail_string()
                    << std::endl;
        }
        ASSERT_EQ(*doc, expect_doc);
      }
    }

    // insert another 1000 docs
    auto s = TestHelper::CollectionInsertDoc(collection, doc_count,
                                             doc_count * 2, false);
    ASSERT_TRUE(s.ok());
    ASSERT_TRUE(collection->Flush().ok());

    // validate fetch result
    for (int i = 0; i < doc_count * 2; i++) {
      auto expect_doc = TestHelper::CreateDoc(i, *schema);
      auto result = collection->Fetch({expect_doc.pk()});
      ASSERT_TRUE(result.has_value());
      ASSERT_EQ(result.value().size(), 1);
      ASSERT_EQ(result.value().count(expect_doc.pk()), 1);
      auto doc = result.value()[expect_doc.pk()];
      ASSERT_NE(doc, nullptr);
      if (metric_type != MetricType::COSINE) {
        if (*doc != expect_doc) {
          std::cout << "       doc:" << doc->to_detail_string() << std::endl;
          std::cout << "expect_doc:" << expect_doc.to_detail_string()
                    << std::endl;
        }
        ASSERT_EQ(*doc, expect_doc);
      }
    }

    stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, doc_count * 2);
    ASSERT_EQ(stats.index_completeness["dense_fp32"], 0);
  };

  func(MetricType::COSINE);
  func(MetricType::L2);
  func(MetricType::IP);
  func(MetricType::COSINE, QuantizeType::FP16);
  func(MetricType::IP, QuantizeType::FP16);
}

TEST_F(CollectionTest, Feature_Insert_SwitchSegment) {
  auto func = [&](uint64_t segment_doc_count, uint64_t doc_count) {
    std::cout << "**** TEST INFO: segment_doc_count: " << segment_doc_count
              << ", insert_doc_count: " << doc_count << std::endl;

    FileHelper::RemoveDirectory(col_path);

    // create with normal schema
    auto schema = TestHelper::CreateSchemaWithMaxDocCount(segment_doc_count);
    auto options = CollectionOptions{false, true, 100 * 1024 * 1024};
    FileHelper::RemoveDirectory(col_path);
    auto collection = TestHelper::CreateCollectionWithDoc(
        col_path, *schema, options, 0, doc_count);

    ASSERT_TRUE(collection->Flush().ok());

    ASSERT_NE(collection, nullptr);

    collection.reset();
    // Reopen collection
    auto result = Collection::Open(col_path, options);
    ASSERT_TRUE(result.has_value());
    collection = std::move(result.value());

    auto stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, doc_count);
    ASSERT_EQ(stats.index_completeness["dense_fp32"], 1);

    auto check_doc = [&](int total_doc_count) {
      // validate fetch result
      for (int i = 0; i < total_doc_count; i++) {
        auto expect_doc = TestHelper::CreateDoc(i, *schema);
        auto result = collection->Fetch({expect_doc.pk()});
        ASSERT_TRUE(result.has_value());
        ASSERT_EQ(result.value().size(), 1);
        ASSERT_EQ(result.value().count(expect_doc.pk()), 1);
        auto doc = result.value()[expect_doc.pk()];
        ASSERT_NE(doc, nullptr);
        if (*doc != expect_doc) {
          std::cout << "       doc:" << doc->to_detail_string() << std::endl;
          std::cout << "expect_doc:" << expect_doc.to_detail_string()
                    << std::endl;
        }
        ASSERT_EQ(*doc, expect_doc);
      }
    };

    check_doc(doc_count);
    std::cout << "check success 1" << std::endl;

    // insert another 1000 docs
    auto s =
        TestHelper::CollectionInsertDoc(collection, doc_count, doc_count * 2);
    ASSERT_TRUE(s.ok());
    ASSERT_TRUE(collection->Flush().ok());

    // validate fetch result
    check_doc(doc_count * 2);
    std::cout << "check success 2" << std::endl;

    stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, doc_count * 2);
    ASSERT_EQ(stats.index_completeness["dense_fp32"], 1);

    collection.reset();
    // Reopen collection
    result = Collection::Open(col_path, options);
    ASSERT_TRUE(result.has_value());
    collection = std::move(result.value());

    check_doc(doc_count * 2);
    std::cout << "check success 3" << std::endl;
  };

  func(1000, 499);
  func(1000, 500);
  func(1000, 501);
  func(1000, 999);
  func(1000, 1000);
  func(1000, 1001);
}

TEST_F(CollectionTest, Feature_Insert_Duplicate) {
  auto schema = TestHelper::CreateNormalSchema();
  auto options = CollectionOptions{false, true, 100 * 1024 * 1024};
  FileHelper::RemoveDirectory(col_path);

  // insert first
  auto collection =
      TestHelper::CreateCollectionWithDoc(col_path, *schema, options, 0, 100);

  // update all docs then
  Result<WriteResults> s;
  for (int i = 0; i < 100; i++) {
    Doc new_doc = TestHelper::CreateDoc(i, *schema);
    std::vector<Doc> docs = {new_doc};
    s = collection->Insert(docs);
    if (!s.has_value()) {
      std::cout << s.error().message() << std::endl;
    }
    ASSERT_TRUE(s.has_value());
    if (!s.value()[0].ok()) {
      std::cout << "0: " << s.value()[0].message() << std::endl;
    }
    ASSERT_FALSE(s.value()[0].ok());
    ASSERT_EQ(s.value()[0].code(), StatusCode::ALREADY_EXISTS);
  }

  Doc new_doc = TestHelper::CreateDoc(101, *schema);
  std::vector<Doc> docs = {new_doc};
  s = collection->Insert(docs);
  ASSERT_TRUE(s.has_value());
  ASSERT_TRUE(s.value()[0].ok());
}

TEST_F(CollectionTest, Feature_Upsert_General) {
  auto func = [&](bool schema_nullable, bool doc_nullable,
                  int doc_count = 1000) {
    FileHelper::RemoveDirectory(col_path);

    // create with normal schema
    auto schema = TestHelper::CreateNormalSchema(schema_nullable);
    auto options = CollectionOptions{false, true, 100 * 1024 * 1024};
    auto collection = TestHelper::CreateCollectionWithDoc(
        col_path, *schema, options, 0, doc_count, doc_nullable, true);


    if (!schema_nullable && doc_nullable) {
      ASSERT_EQ(collection, nullptr);
      return;
    } else {
      ASSERT_NE(collection, nullptr);
    }

    auto stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, doc_count);
    ASSERT_EQ(stats.index_completeness["dense_fp32"], 1);
    ASSERT_EQ(stats.index_completeness["dense_fp16"], 1);
    // ASSERT_EQ(stats.index_completeness["dense_fp64"], 1);
    ASSERT_EQ(stats.index_completeness["sparse_fp32"], 1);
    ASSERT_EQ(stats.index_completeness["sparse_fp16"], 1);

    // validate fetch result
    for (int i = 0; i < doc_count; i++) {
      auto expect_doc = doc_nullable ? TestHelper::CreateDocNull(i, *schema)
                                     : TestHelper::CreateDoc(i, *schema);
      auto result = collection->Fetch({expect_doc.pk()});
      ASSERT_TRUE(result.has_value());
      ASSERT_EQ(result.value().size(), 1);
      ASSERT_EQ(result.value().count(expect_doc.pk()), 1);
      auto doc = result.value()[expect_doc.pk()];
      ASSERT_NE(doc, nullptr);
      if (*doc != expect_doc) {
        std::cout << "       doc:" << doc->to_detail_string() << std::endl;
        std::cout << "expect_doc:" << expect_doc.to_detail_string()
                  << std::endl;
      }
      ASSERT_EQ(*doc, expect_doc);
    }

    ASSERT_TRUE(collection->Flush().ok());

    ASSERT_NE(collection, nullptr);

    collection.reset();
    // Reopen collection
    auto result = Collection::Open(col_path, options);
    ASSERT_TRUE(result.has_value());
    collection = std::move(result.value());

    // insert another 1000 docs
    auto s = TestHelper::CollectionInsertDoc(collection, doc_count,
                                             doc_count * 2, doc_nullable);
    ASSERT_TRUE(s.ok());

    // validate fetch result
    for (int i = 0; i < doc_count * 2; i++) {
      auto expect_doc = doc_nullable ? TestHelper::CreateDocNull(i, *schema)
                                     : TestHelper::CreateDoc(i, *schema);
      auto result = collection->Fetch({expect_doc.pk()});
      ASSERT_TRUE(result.has_value());
      ASSERT_EQ(result.value().size(), 1);
      ASSERT_EQ(result.value().count(expect_doc.pk()), 1);
      auto doc = result.value()[expect_doc.pk()];
      ASSERT_NE(doc, nullptr);
      if (*doc != expect_doc) {
        std::cout << "       doc:" << doc->to_detail_string() << std::endl;
        std::cout << "expect_doc:" << expect_doc.to_detail_string()
                  << std::endl;
      }
      ASSERT_EQ(*doc, expect_doc);
    }

    stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, doc_count * 2);
    ASSERT_EQ(stats.index_completeness["dense_fp32"], 1);

    ASSERT_EQ(stats.index_completeness["dense_fp16"], 1);
    // ASSERT_EQ(stats.index_completeness["dense_fp64"], 1);
    ASSERT_EQ(stats.index_completeness["sparse_fp32"], 1);
    ASSERT_EQ(stats.index_completeness["sparse_fp16"], 1);
  };

  func(false, false);
  func(true, true);
  func(true, false);
  func(false, true);

  func(false, false, 0);
  func(false, false, 1);
  func(false, false, 2);
}

TEST_F(CollectionTest, Feature_Upsert_Incremental) {
  auto func = [&](bool schema_nullable, bool doc_nullable,
                  int doc_count = 1000) {
    FileHelper::RemoveDirectory(col_path);

    // create with normal schema
    auto schema = TestHelper::CreateNormalSchema(schema_nullable);
    auto options = CollectionOptions{false, true, 100 * 1024 * 1024};
    auto collection = TestHelper::CreateCollectionWithDoc(
        col_path, *schema, options, 0, doc_count, doc_nullable, true);

    if (!schema_nullable && doc_nullable) {
      ASSERT_EQ(collection, nullptr);
      return;
    } else {
      ASSERT_NE(collection, nullptr);
    }

    // validate fetch result
    for (int i = 0; i < doc_count; i++) {
      auto expect_doc = doc_nullable ? TestHelper::CreateDocNull(i, *schema)
                                     : TestHelper::CreateDoc(i, *schema);
      auto result = collection->Fetch({expect_doc.pk()});
      ASSERT_TRUE(result.has_value());
      ASSERT_EQ(result.value().size(), 1);
      ASSERT_EQ(result.value().count(expect_doc.pk()), 1);
      auto doc = result.value()[expect_doc.pk()];
      ASSERT_NE(doc, nullptr);
      if (*doc != expect_doc) {
        std::cout << "       doc:" << doc->to_detail_string() << std::endl;
        std::cout << "expect_doc:" << expect_doc.to_detail_string()
                  << std::endl;
      }
      ASSERT_EQ(*doc, expect_doc);
    }

    ASSERT_TRUE(collection->Flush().ok());

    ASSERT_NE(collection, nullptr);

    collection.reset();
    // Reopen collection
    auto result = Collection::Open(col_path, options);
    ASSERT_TRUE(result.has_value());
    collection = std::move(result.value());

    // upsert 1000 docs
    auto s = TestHelper::CollectionInsertDoc(collection, 0, doc_count,
                                             doc_nullable, true);
    ASSERT_TRUE(s.ok());

    // validate fetch result
    for (int i = 0; i < doc_count; i++) {
      auto expect_doc = doc_nullable ? TestHelper::CreateDocNull(i, *schema)
                                     : TestHelper::CreateDoc(i, *schema);
      auto result = collection->Fetch({expect_doc.pk()});
      ASSERT_TRUE(result.has_value());
      ASSERT_EQ(result.value().size(), 1);
      ASSERT_EQ(result.value().count(expect_doc.pk()), 1);
      auto doc = result.value()[expect_doc.pk()];
      ASSERT_NE(doc, nullptr);
      if (*doc != expect_doc) {
        std::cout << "       doc:" << doc->to_detail_string() << std::endl;
        std::cout << "expect_doc:" << expect_doc.to_detail_string()
                  << std::endl;
      }
      ASSERT_EQ(*doc, expect_doc);
    }
  };

  func(false, false);
  func(true, true);
  func(true, false);
  func(false, true);

  func(false, false, 0);
  func(false, false, 1);
  func(false, false, 2);
}

TEST_F(CollectionTest, Feature_Upsert_Nullable) {
  auto check_doc = [&](const Collection::Ptr &collection, const std::string &pk,
                       const Doc &expected_doc) {
    auto result = collection->Fetch({pk});
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result.value().size(), 1);
    ASSERT_EQ(result.value().count(pk), 1);
    auto doc = result.value()[pk];
    ASSERT_NE(doc, nullptr);
    if (*doc != expected_doc) {
      std::cout << "       doc:" << doc->to_detail_string() << std::endl;
      std::cout << "expect_doc:" << expected_doc.to_detail_string()
                << std::endl;
    }
    ASSERT_EQ(*doc, expected_doc);
  };

  // schema not nulltable
  {
    auto schema = TestHelper::CreateNormalSchema();
    auto options = CollectionOptions{false, true, 100 * 1024 * 1024};
    FileHelper::RemoveDirectory(col_path);

    auto collection =
        TestHelper::CreateCollectionWithDoc(col_path, *schema, options, 0, 0);

    // insert one doc
    auto insert_doc = TestHelper::CreateDoc(0, *schema, TestHelper::MakePK(0));
    std::vector<Doc> docs = {insert_doc};
    auto s = collection->Insert(docs);
    ASSERT_TRUE(s.has_value());

    // update doc
    auto update_doc = TestHelper::CreateDoc(0, *schema, TestHelper::MakePK(0));
    update_doc.remove("int32");
    docs = {update_doc};
    s = collection->Upsert(docs);
    if (!s.has_value()) {
      std::cout << s.error().message() << std::endl;
    }
    ASSERT_FALSE(s.has_value());


    update_doc.set_null("int32");
    docs = {update_doc};
    s = collection->Upsert(docs);
    if (!s.has_value()) {
      std::cout << s.error().message() << std::endl;
    }
    ASSERT_FALSE(s.has_value());

    // check doc
    check_doc(collection, insert_doc.pk(), insert_doc);
  }

  // schema nulltable
  {
    auto schema = TestHelper::CreateNormalSchema(true);
    auto options = CollectionOptions{false, true, 100 * 1024 * 1024};
    FileHelper::RemoveDirectory(col_path);

    auto collection =
        TestHelper::CreateCollectionWithDoc(col_path, *schema, options, 0, 0);

    // insert one doc
    auto insert_doc = TestHelper::CreateDoc(0, *schema, TestHelper::MakePK(0));
    std::vector<Doc> docs = {insert_doc};
    auto s = collection->Insert(docs);
    ASSERT_TRUE(s.has_value());

    // update doc
    auto update_doc = TestHelper::CreateDoc(0, *schema, TestHelper::MakePK(0));
    update_doc.remove("int32");
    docs = {update_doc};
    s = collection->Upsert(docs);
    if (!s.has_value()) {
      std::cout << s.error().message() << std::endl;
    }
    ASSERT_TRUE(s.has_value());
    if (!s.value()[0].ok()) {
      std::cout << s.value()[0].message() << std::endl;
    }
    ASSERT_TRUE(s.value()[0].ok());

    // check doc
    check_doc(collection, insert_doc.pk(), update_doc);

    update_doc.set_null("int32");
    docs = {update_doc};
    s = collection->Update(docs);
    if (!s.has_value()) {
      std::cout << s.error().message() << std::endl;
    }
    ASSERT_TRUE(s.has_value());

    // check doc
    auto pk = insert_doc.pk();
    auto result = collection->Fetch({pk});
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result.value().size(), 1);
    ASSERT_EQ(result.value().count(pk), 1);
    auto doc = result.value()[pk];
    ASSERT_NE(doc, nullptr);
    auto get_result = doc->get_field<int32_t>("int32");
    ASSERT_EQ(get_result.status(), Doc::FieldGetStatus::NOT_FOUND);
  }
}


TEST_F(CollectionTest, Feature_Update_General) {
  auto func = [&](int doc_count) {
    auto schema = TestHelper::CreateNormalSchema();
    auto options = CollectionOptions{false, true, 100 * 1024 * 1024};
    FileHelper::RemoveDirectory(col_path);

    // insert first
    auto collection = TestHelper::CreateCollectionWithDoc(
        col_path, *schema, options, 0, doc_count);

    auto check_doc = [&](int updated_doc_count) {
      for (int i = 0; i < updated_doc_count; i++) {
        auto expect_doc =
            TestHelper::CreateDoc(i + 1, *schema, TestHelper::MakePK(i));
        auto result = collection->Fetch({expect_doc.pk()});
        ASSERT_TRUE(result.has_value());
        ASSERT_EQ(result.value().size(), 1);
        ASSERT_EQ(result.value().count(expect_doc.pk()), 1);
        auto doc = result.value()[expect_doc.pk()];
        ASSERT_NE(doc, nullptr);
        if (*doc != expect_doc) {
          std::cout << "       doc:" << doc->to_detail_string() << std::endl;
          std::cout << "expect_doc:" << expect_doc.to_detail_string()
                    << std::endl;
        }
        ASSERT_EQ(*doc, expect_doc);
      }

      // validate fetch result
      for (int i = updated_doc_count; i < doc_count; i++) {
        auto expect_doc = TestHelper::CreateDoc(i, *schema);
        auto result = collection->Fetch({expect_doc.pk()});
        ASSERT_TRUE(result.has_value());
        ASSERT_EQ(result.value().size(), 1);
        ASSERT_EQ(result.value().count(expect_doc.pk()), 1);
        auto doc = result.value()[expect_doc.pk()];
        ASSERT_NE(doc, nullptr);
        if (*doc != expect_doc) {
          std::cout << "       doc:" << doc->to_detail_string() << std::endl;
          std::cout << "expect_doc:" << expect_doc.to_detail_string()
                    << std::endl;
        }
        ASSERT_EQ(*doc, expect_doc);
      }
    };

    // update all docs then
    Result<WriteResults> s;
    for (int i = 0; i < doc_count; i++) {
      Doc new_doc =
          TestHelper::CreateDoc(i + 1, *schema, TestHelper::MakePK(i));
      std::vector<Doc> docs = {new_doc};
      s = collection->Update(docs);
      if (!s.has_value()) {
        std::cout << s.error().message() << std::endl;
      }
      ASSERT_TRUE(s.has_value());
      if (!s.value()[0].ok()) {
        std::cout << s.value()[0].message() << std::endl;
      }
      ASSERT_TRUE(s.value()[0].ok());

      if (i % 100 == 0 || i == 1) {
        check_doc(i + 1);
        collection.reset();
        auto result = Collection::Open(col_path, options);
        if (!result.has_value()) {
          std::cout << result.error().message() << std::endl;
        }
        collection = std::move(result.value());

        check_doc(i + 1);
      }
    }

    collection.reset();
    auto result = Collection::Open(col_path, options);
    if (!result.has_value()) {
      std::cout << result.error().message() << std::endl;
    }
    collection = std::move(result.value());

    check_doc(doc_count);
  };

  func(99);
  func(100);
  func(101);
  func(1000);
}

TEST_F(CollectionTest, Feature_Update_Incremental) {
  auto func = [&](int doc_count, bool doc_nullable) {
    auto schema = TestHelper::CreateNormalSchema(doc_nullable);
    auto options = CollectionOptions{false, true, 100 * 1024 * 1024};
    FileHelper::RemoveDirectory(col_path);

    // insert first
    auto collection = TestHelper::CreateCollectionWithDoc(
        col_path, *schema, options, 0, doc_count, doc_nullable);

    auto rewrite_doc = [&](Doc &doc) {
      // update int32
      int32_t new_int32 = 9999;
      doc.set("int32", new_int32);

      // update float
      float new_float = 9999.0;
      doc.set("float", new_float);

      // update string
      std::string new_string = "string_value";
      doc.set("string", new_string);
    };

    auto check_doc = [&](int updated_doc_count) {
      for (int i = 0; i < updated_doc_count; i++) {
        auto expect_doc =
            TestHelper::CreateDoc(i + 1, *schema, TestHelper::MakePK(i));
        rewrite_doc(expect_doc);
        auto result = collection->Fetch({expect_doc.pk()});
        ASSERT_TRUE(result.has_value());
        ASSERT_EQ(result.value().size(), 1);
        ASSERT_EQ(result.value().count(expect_doc.pk()), 1);
        auto doc = result.value()[expect_doc.pk()];
        ASSERT_NE(doc, nullptr);
        if (*doc != expect_doc) {
          std::cout << "       doc:" << doc->to_detail_string() << std::endl;
          std::cout << "expect_doc:" << expect_doc.to_detail_string()
                    << std::endl;
        }
        ASSERT_EQ(*doc, expect_doc);
      }

      // validate fetch result
      for (int i = updated_doc_count; i < doc_count; i++) {
        auto expect_doc = doc_nullable ? TestHelper::CreateDocNull(i, *schema)
                                       : TestHelper::CreateDoc(i, *schema);
        auto result = collection->Fetch({expect_doc.pk()});
        ASSERT_TRUE(result.has_value());
        ASSERT_EQ(result.value().size(), 1);
        ASSERT_EQ(result.value().count(expect_doc.pk()), 1);
        auto doc = result.value()[expect_doc.pk()];
        ASSERT_NE(doc, nullptr);
        if (*doc != expect_doc) {
          std::cout << "       doc:" << doc->to_detail_string() << std::endl;
          std::cout << "expect_doc:" << expect_doc.to_detail_string()
                    << std::endl;
        }
        ASSERT_EQ(*doc, expect_doc);
      }
    };

    // update all docs then
    Result<WriteResults> s;
    for (int i = 0; i < doc_count; i++) {
      Doc new_doc =
          TestHelper::CreateDoc(i + 1, *schema, TestHelper::MakePK(i));
      rewrite_doc(new_doc);
      std::vector<Doc> docs = {new_doc};
      s = collection->Update(docs);
      if (!s.has_value()) {
        std::cout << s.error().message() << std::endl;
      }
      ASSERT_TRUE(s.has_value());
      if (!s.value()[0].ok()) {
        std::cout << s.value()[0].message() << std::endl;
      }
      ASSERT_TRUE(s.value()[0].ok());

      if (i % 100 == 0 || i == 1) {
        check_doc(i + 1);
        collection.reset();
        auto result = Collection::Open(col_path, options);
        if (!result.has_value()) {
          std::cout << result.error().message() << std::endl;
        }
        collection = std::move(result.value());

        check_doc(i + 1);
      }
    }

    collection.reset();
    auto result = Collection::Open(col_path, options);
    if (!result.has_value()) {
      std::cout << result.error().message() << std::endl;
    }
    collection = std::move(result.value());

    check_doc(doc_count);
  };

  func(99, false);
  func(99, true);
  func(100, false);
  func(100, true);
  func(101, false);
  func(101, true);
  func(1000, false);
  func(1000, true);
}

TEST_F(CollectionTest, Feature_Update_Nullable) {
  auto check_doc = [&](const Collection::Ptr &collection, const std::string &pk,
                       const Doc &expected_doc) {
    auto result = collection->Fetch({pk});
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result.value().size(), 1);
    ASSERT_EQ(result.value().count(pk), 1);
    auto doc = result.value()[pk];
    ASSERT_NE(doc, nullptr);
    if (*doc != expected_doc) {
      std::cout << "       doc:" << doc->to_detail_string() << std::endl;
      std::cout << "expect_doc:" << expected_doc.to_detail_string()
                << std::endl;
    }
    ASSERT_EQ(*doc, expected_doc);
  };

  // schema not nulltable
  {
    auto schema = TestHelper::CreateNormalSchema();
    auto options = CollectionOptions{false, true, 100 * 1024 * 1024};
    FileHelper::RemoveDirectory(col_path);

    auto collection =
        TestHelper::CreateCollectionWithDoc(col_path, *schema, options, 0, 0);

    // insert one doc
    auto insert_doc = TestHelper::CreateDoc(0, *schema, TestHelper::MakePK(0));
    std::vector<Doc> docs = {insert_doc};
    auto s = collection->Insert(docs);
    ASSERT_TRUE(s.has_value());

    // update doc
    auto update_doc = TestHelper::CreateDoc(0, *schema, TestHelper::MakePK(0));
    update_doc.remove("int32");
    docs = {update_doc};
    s = collection->Update(docs);
    if (!s.has_value()) {
      std::cout << s.error().message() << std::endl;
    }
    ASSERT_TRUE(s.has_value());
    if (!s.value()[0].ok()) {
      std::cout << s.value()[0].message() << std::endl;
    }
    ASSERT_TRUE(s.value()[0].ok());

    update_doc.set_null("int32");
    docs = {update_doc};
    s = collection->Update(docs);
    if (!s.has_value()) {
      std::cout << s.error().message() << std::endl;
    }
    ASSERT_FALSE(s.has_value());

    // check doc
    check_doc(collection, insert_doc.pk(), insert_doc);
  }

  // schema nulltable
  {
    auto schema = TestHelper::CreateNormalSchema(true);
    auto options = CollectionOptions{false, true, 100 * 1024 * 1024};
    FileHelper::RemoveDirectory(col_path);

    auto collection =
        TestHelper::CreateCollectionWithDoc(col_path, *schema, options, 0, 0);

    // insert one doc
    auto insert_doc = TestHelper::CreateDoc(0, *schema, TestHelper::MakePK(0));
    std::vector<Doc> docs = {insert_doc};
    auto s = collection->Insert(docs);
    ASSERT_TRUE(s.has_value());

    // update doc
    auto update_doc = TestHelper::CreateDoc(0, *schema, TestHelper::MakePK(0));
    update_doc.remove("int32");
    docs = {update_doc};
    s = collection->Update(docs);
    if (!s.has_value()) {
      std::cout << s.error().message() << std::endl;
    }
    ASSERT_TRUE(s.has_value());
    if (!s.value()[0].ok()) {
      std::cout << s.value()[0].message() << std::endl;
    }
    ASSERT_TRUE(s.value()[0].ok());

    // check doc
    check_doc(collection, insert_doc.pk(), insert_doc);

    update_doc.set_null("int32");
    docs = {update_doc};
    s = collection->Update(docs);
    if (!s.has_value()) {
      std::cout << s.error().message() << std::endl;
    }
    ASSERT_TRUE(s.has_value());

    // check doc
    auto pk = insert_doc.pk();
    auto result = collection->Fetch({pk});
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result.value().size(), 1);
    ASSERT_EQ(result.value().count(pk), 1);
    auto doc = result.value()[pk];
    ASSERT_NE(doc, nullptr);
    auto get_result = doc->get_field<int32_t>("int32");
    ASSERT_EQ(get_result.status(), Doc::FieldGetStatus::NOT_FOUND);
  }
}

TEST_F(CollectionTest, Feature_Update_Empty) {
  auto schema = TestHelper::CreateNormalSchema();
  auto options = CollectionOptions{false, true, 100 * 1024 * 1024};
  FileHelper::RemoveDirectory(col_path);

  // insert first
  auto collection =
      TestHelper::CreateCollectionWithDoc(col_path, *schema, options, 0, 0);

  // update all docs then
  Result<WriteResults> s;
  for (int i = 0; i < 100; i++) {
    Doc new_doc = TestHelper::CreateDoc(i + 1, *schema, TestHelper::MakePK(i));
    std::vector<Doc> docs = {new_doc};
    s = collection->Update(docs);
    if (!s.has_value()) {
      std::cout << s.error().message() << std::endl;
    }
    ASSERT_TRUE(s.has_value());
    if (!s.value()[0].ok()) {
      std::cout << "0: " << s.value()[0].message() << std::endl;
    }
    ASSERT_FALSE(s.value()[0].ok());
    ASSERT_EQ(s.value()[0].code(), StatusCode::NOT_FOUND);
  }
}

TEST_F(CollectionTest, Feature_Delete_General) {
  auto func = [&](int doc_count) {
    auto schema = TestHelper::CreateNormalSchema();
    auto options = CollectionOptions{false, true, 100 * 1024 * 1024};
    FileHelper::RemoveDirectory(col_path);

    // insert first
    auto collection = TestHelper::CreateCollectionWithDoc(
        col_path, *schema, options, 0, doc_count);

    auto check_doc = [&](int updated_doc_count) {
      for (int i = 0; i < updated_doc_count; i++) {
        auto expect_doc = TestHelper::CreateDoc(i, *schema);
        auto result = collection->Fetch({expect_doc.pk()});
        ASSERT_TRUE(result.has_value());
        ASSERT_EQ(result.value().size(), 1);
        ASSERT_EQ(result.value().count(expect_doc.pk()), 1);
        auto doc = result.value()[expect_doc.pk()];
        ASSERT_EQ(doc, nullptr);
      }

      // validate fetch result
      for (int i = updated_doc_count; i < doc_count; i++) {
        auto expect_doc = TestHelper::CreateDoc(i, *schema);
        auto result = collection->Fetch({expect_doc.pk()});
        ASSERT_TRUE(result.has_value());
        ASSERT_EQ(result.value().size(), 1);
        ASSERT_EQ(result.value().count(expect_doc.pk()), 1);
        auto doc = result.value()[expect_doc.pk()];
        ASSERT_NE(doc, nullptr);
        if (*doc != expect_doc) {
          std::cout << "       doc:" << doc->to_detail_string() << std::endl;
          std::cout << "expect_doc:" << expect_doc.to_detail_string()
                    << std::endl;
        }
        ASSERT_EQ(*doc, expect_doc);
      }
    };

    Result<WriteResults> s;
    for (int i = 0; i < doc_count; i++) {
      s = collection->Delete({TestHelper::MakePK(i)});
      if (!s.has_value()) {
        std::cout << s.error().message() << std::endl;
      }
      ASSERT_TRUE(s.has_value());
      if (!s.value()[0].ok()) {
        std::cout << s.value()[0].message() << std::endl;
      }
      ASSERT_TRUE(s.value()[0].ok());

      if (i % 100 == 0 || i == 0) {
        check_doc(i + 1);
        collection.reset();
        auto result = Collection::Open(col_path, options);
        if (!result.has_value()) {
          std::cout << result.error().message() << std::endl;
        }
        collection = std::move(result.value());

        check_doc(i + 1);

        auto stats = collection->Stats().value();
        ASSERT_EQ(stats.doc_count, doc_count - i - 1);
      }
    }

    collection.reset();
    auto result = Collection::Open(col_path, options);
    if (!result.has_value()) {
      std::cout << result.error().message() << std::endl;
    }
    collection = std::move(result.value());
    auto stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, 0);

    check_doc(doc_count);
  };

  func(99);
  func(100);
  func(101);
  func(1000);
}

TEST_F(CollectionTest, Feature_Delete_Repeated) {
  auto func = [&](int doc_count) {
    auto schema = TestHelper::CreateNormalSchema();
    auto options = CollectionOptions{false, true, 100 * 1024 * 1024};
    FileHelper::RemoveDirectory(col_path);

    // insert first
    auto collection = TestHelper::CreateCollectionWithDoc(
        col_path, *schema, options, 0, doc_count);

    auto check_doc = [&](bool deleted) {
      for (int i = 0; i < doc_count; i++) {
        auto expect_doc = TestHelper::CreateDoc(i, *schema);
        auto result = collection->Fetch({expect_doc.pk()});
        ASSERT_TRUE(result.has_value());
        ASSERT_EQ(result.value().size(), 1);
        ASSERT_EQ(result.value().count(expect_doc.pk()), 1);
        auto doc = result.value()[expect_doc.pk()];
        if (deleted) {
          ASSERT_EQ(doc, nullptr);
        } else {
          ASSERT_EQ(*doc, expect_doc);
        }
      }
    };

    for (int i = 0; i < 10; i++) {
      // delete first
      Result<WriteResults> s;
      for (int i = 0; i < doc_count; i++) {
        s = collection->Delete({TestHelper::MakePK(i)});
        if (!s.has_value()) {
          std::cout << s.error().message() << std::endl;
        }
        ASSERT_TRUE(s.has_value());
        if (!s.value()[0].ok()) {
          std::cout << s.value()[0].message() << std::endl;
        }
        ASSERT_TRUE(s.value()[0].ok());
      }

      check_doc(true);

      // insert then
      auto st = TestHelper::CollectionInsertDoc(collection, 0, doc_count);
      if (!st.ok()) {
        std::cout << st.message() << std::endl;
      }
      ASSERT_TRUE(st.ok());
    }
  };

  func(1);
  func(100);
}

TEST_F(CollectionTest, Feature_DeleteByFilter_General) {
  auto func = [&](int doc_count) {
    auto schema = TestHelper::CreateNormalSchema();
    auto options = CollectionOptions{false, true, 100 * 1024 * 1024};
    FileHelper::RemoveDirectory(col_path);

    // insert first
    auto collection = TestHelper::CreateCollectionWithDoc(
        col_path, *schema, options, 0, doc_count);

    ASSERT_TRUE(collection->Flush().ok());

    auto check_doc = [&](int updated_doc_count) {
      for (int i = 0; i < updated_doc_count; i++) {
        auto expect_doc = TestHelper::CreateDoc(i, *schema);
        auto result = collection->Fetch({expect_doc.pk()});
        ASSERT_TRUE(result.has_value());
        ASSERT_EQ(result.value().size(), 1);
        ASSERT_EQ(result.value().count(expect_doc.pk()), 1);
        auto doc = result.value()[expect_doc.pk()];
        if (doc != nullptr) {
          std::cout << "doc: " << doc->to_detail_string() << std::endl;
        }
        ASSERT_EQ(doc, nullptr);
      }

      // validate fetch result
      for (int i = updated_doc_count; i < doc_count; i++) {
        auto expect_doc = TestHelper::CreateDoc(i, *schema);
        auto result = collection->Fetch({expect_doc.pk()});
        ASSERT_TRUE(result.has_value());
        ASSERT_EQ(result.value().size(), 1);
        ASSERT_EQ(result.value().count(expect_doc.pk()), 1);
        auto doc = result.value()[expect_doc.pk()];
        ASSERT_NE(doc, nullptr);
        if (*doc != expect_doc) {
          std::cout << "       doc:" << doc->to_detail_string() << std::endl;
          std::cout << "expect_doc:" << expect_doc.to_detail_string()
                    << std::endl;
        }
        ASSERT_EQ(*doc, expect_doc);
      }
    };

    Status s;
    for (int i = 0; i < doc_count; i++) {
      s = collection->DeleteByFilter("int32 = " + std::to_string(i));
      if (!s.ok()) {
        std::cout << s.message() << std::endl;
      }
      ASSERT_TRUE(s.ok());

      if (i % 100 == 0 || i == 0) {
        std::cout << "check begin: " << i << std::endl;

        check_doc(i + 1);
        collection.reset();
        auto result = Collection::Open(col_path, options);
        if (!result.has_value()) {
          std::cout << result.error().message() << std::endl;
        }
        collection = std::move(result.value());

        check_doc(i + 1);

        auto stats = collection->Stats().value();
        ASSERT_EQ(stats.doc_count, doc_count - i - 1);
      }
    }

    collection.reset();
    auto result = Collection::Open(col_path, options);
    if (!result.has_value()) {
      std::cout << result.error().message() << std::endl;
    }
    collection = std::move(result.value());
    auto stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, 0);

    check_doc(doc_count);
  };

  func(99);
  func(100);
  func(101);
  func(1000);
}

TEST_F(CollectionTest, Feature_DeleteByFilter_ScalarIndex) {
  auto func = [&](int doc_count) {
    auto schema = TestHelper::CreateNormalSchema(
        false, "demo", std::make_shared<InvertIndexParams>(false));
    auto options = CollectionOptions{false, true, 100 * 1024 * 1024};
    FileHelper::RemoveDirectory(col_path);

    // insert first
    auto collection = TestHelper::CreateCollectionWithDoc(
        col_path, *schema, options, 0, doc_count);

    ASSERT_TRUE(collection->Flush().ok());

    auto check_doc = [&](int updated_doc_count) {
      for (int i = 0; i < updated_doc_count; i++) {
        auto expect_doc = TestHelper::CreateDoc(i, *schema);
        auto result = collection->Fetch({expect_doc.pk()});
        ASSERT_TRUE(result.has_value());
        ASSERT_EQ(result.value().size(), 1);
        ASSERT_EQ(result.value().count(expect_doc.pk()), 1);
        auto doc = result.value()[expect_doc.pk()];
        if (doc != nullptr) {
          std::cout << "doc: " << doc->to_detail_string() << std::endl;
        }
        ASSERT_EQ(doc, nullptr);
      }

      // validate fetch result
      for (int i = updated_doc_count; i < doc_count; i++) {
        auto expect_doc = TestHelper::CreateDoc(i, *schema);
        auto result = collection->Fetch({expect_doc.pk()});
        ASSERT_TRUE(result.has_value());
        ASSERT_EQ(result.value().size(), 1);
        ASSERT_EQ(result.value().count(expect_doc.pk()), 1);
        auto doc = result.value()[expect_doc.pk()];
        ASSERT_NE(doc, nullptr);
        if (*doc != expect_doc) {
          std::cout << "       doc:" << doc->to_detail_string() << std::endl;
          std::cout << "expect_doc:" << expect_doc.to_detail_string()
                    << std::endl;
        }
        ASSERT_EQ(*doc, expect_doc);
      }
    };

    Status s;
    for (int i = 0; i < doc_count; i++) {
      s = collection->DeleteByFilter("int32 = " + std::to_string(i));
      if (!s.ok()) {
        std::cout << s.message() << std::endl;
      }
      ASSERT_TRUE(s.ok());

      if (i % 100 == 0 || i == 0) {
        std::cout << "check begin: " << i << std::endl;

        check_doc(i + 1);
        collection.reset();
        auto result = Collection::Open(col_path, options);
        if (!result.has_value()) {
          std::cout << result.error().message() << std::endl;
        }
        collection = std::move(result.value());

        check_doc(i + 1);

        auto stats = collection->Stats().value();
        ASSERT_EQ(stats.doc_count, doc_count - i - 1);
      }
    }

    collection.reset();
    auto result = Collection::Open(col_path, options);
    if (!result.has_value()) {
      std::cout << result.error().message() << std::endl;
    }
    collection = std::move(result.value());
    auto stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, 0);

    check_doc(doc_count);
  };

  func(1);
  func(100);
  func(101);
  func(1000);
}

TEST_F(CollectionTest, Feature_MixedWrite_General) {
  // case1: insert -> upsert -> update -> delete
  auto schema = TestHelper::CreateNormalSchema();
  auto options = CollectionOptions{false, true, 100 * 1024 * 1024};
  FileHelper::RemoveDirectory(col_path);

  // insert first
  auto collection =
      TestHelper::CreateCollectionWithDoc(col_path, *schema, options, 0, 0);

  for (int i = 0; i < 100; i++) {
    // std::cout << "insert: " << i << std::endl;

    // insert
    auto new_doc = TestHelper::CreateDoc(i, *schema);
    std::vector<Doc> new_docs = {new_doc};
    auto res = collection->Insert(new_docs);
    ASSERT_TRUE(res.has_value());
    ASSERT_TRUE(res.value()[0].ok());

    // fetch
    auto docs = collection->Fetch({TestHelper::MakePK(i)});
    ASSERT_TRUE(docs.has_value());
    ASSERT_EQ(docs.value().size(), 1);
    ASSERT_EQ(docs.value().count(TestHelper::MakePK(i)), 1);
    ASSERT_EQ(new_doc, *docs.value()[TestHelper::MakePK(i)]);

    auto stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, i + 1);

    // upsert
    new_doc = TestHelper::CreateDoc(i + 1, *schema, TestHelper::MakePK(i));
    new_docs = {new_doc};
    res = collection->Upsert(new_docs);
    ASSERT_TRUE(res.has_value());
    ASSERT_TRUE(res.value()[0].ok());

    // fetch
    docs = collection->Fetch({TestHelper::MakePK(i)}).value();
    ASSERT_TRUE(docs.has_value());
    ASSERT_EQ(docs.value().size(), 1);
    ASSERT_EQ(docs.value().count(TestHelper::MakePK(i)), 1);
    ASSERT_EQ(new_doc, *docs.value()[TestHelper::MakePK(i)]);

    stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, i + 1);

    // update
    new_doc = TestHelper::CreateDoc(i + 2, *schema, TestHelper::MakePK(i));
    new_docs = {new_doc};
    res = collection->Update(new_docs);
    ASSERT_TRUE(res.has_value());
    ASSERT_TRUE(res.value()[0].ok());

    // fetch
    docs = collection->Fetch({TestHelper::MakePK(i)}).value();
    ASSERT_TRUE(docs.has_value());
    ASSERT_EQ(docs.value().size(), 1);
    ASSERT_EQ(docs.value().count(TestHelper::MakePK(i)), 1);
    ASSERT_EQ(new_doc, *docs.value()[TestHelper::MakePK(i)]);

    stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, i + 1);

    // delete
    res = collection->Delete({TestHelper::MakePK(i)});
    ASSERT_TRUE(res.has_value());
    ASSERT_TRUE(res.value()[0].ok());

    stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, i);

    // insert again
    new_doc = TestHelper::CreateDoc(i, *schema);
    new_docs = {new_doc};
    res = collection->Insert(new_docs);
    ASSERT_TRUE(res.has_value());
    ASSERT_TRUE(res.value()[0].ok());

    // fetch
    docs = collection->Fetch({TestHelper::MakePK(i)});
    ASSERT_TRUE(docs.has_value());
    ASSERT_EQ(docs.value().size(), 1);
    ASSERT_EQ(docs.value().count(TestHelper::MakePK(i)), 1);
    ASSERT_EQ(new_doc, *docs.value()[TestHelper::MakePK(i)]);

    stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, i + 1);
  }
}

TEST_F(CollectionTest, Feature_CreateIndex_General) {
  // create empty collection
  auto schema = TestHelper::CreateNormalSchema();
  auto options = CollectionOptions{false, true, 64 * 1024 * 1024};
  auto collection = TestHelper::CreateCollectionWithDoc(col_path, *schema,
                                                        options, 0, 0, false);

  ASSERT_TRUE(collection->Flush().ok());
  auto stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, 0);

  auto index_params = std::make_shared<HnswIndexParams>(MetricType::IP);
  auto s = collection->CreateIndex("dense_fp32", index_params);
  if (!s.ok()) {
    std::cout << "status: " << s.message() << std::endl;
    ASSERT_TRUE(false);
  }
  auto new_index_params = std::make_shared<HnswIndexParams>(MetricType::COSINE);
  s = collection->CreateIndex("dense_fp32", index_params);
  if (!s.ok()) {
    std::cout << "status: " << s.message() << std::endl;
    ASSERT_TRUE(false);
  }

  s = collection->CreateIndex("dense_fp32_invalid", index_params);
  ASSERT_FALSE(s.ok());
}

TEST_F(CollectionTest, Feature_CreateIndex_Vector) {
  auto func = [&](std::string field_name,
                  MetricType metric_type = MetricType::IP,
                  QuantizeType quantize_type = QuantizeType::UNDEFINED) {
    std::cout << "**** Test field: " << field_name
              << ", metric: " << MetricTypeCodeBook::AsString(metric_type)
              << ", quantize: " << QuantizeTypeCodeBook::AsString(quantize_type)
              << std::endl;

    FileHelper::RemoveDirectory(col_path);

    int doc_count = 10;

    auto schema = TestHelper::CreateNormalSchema();
    auto options = CollectionOptions{false, true, 64 * 1024 * 1024};
    auto collection = TestHelper::CreateCollectionWithDoc(
        col_path, *schema, options, 0, doc_count, false);

    ASSERT_TRUE(collection->Flush().ok());

    auto stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, doc_count);
    ASSERT_EQ(stats.index_completeness[field_name], 1);

    auto index_params =
        std::make_shared<HnswIndexParams>(metric_type, 16, 200, quantize_type);
    auto s = collection->CreateIndex(field_name, index_params);
    std::cout << "status: " << s.message()
              << ", code: " << GetDefaultMessage(s.code()) << std::endl;
    ASSERT_TRUE(s.ok());

    VectorQuery query;
    query.topk_ = doc_count;
    query.field_name_ = field_name;
    query.include_vector_ = true;
    auto field_scheama = schema->get_vector_field(field_name);
    ASSERT_NE(field_scheama, nullptr);
    ASSERT_TRUE(field_scheama->is_vector_field());

    bool is_dense = field_scheama->is_dense_vector();

    std::vector<float> vector;
    std::vector<ailego::Float16> vector_fp16;
    std::vector<int8_t> vector_int8;
    std::pair<std::vector<uint32_t>, std::vector<float>> sparse_vector;
    std::pair<std::vector<uint32_t>, std::vector<ailego::Float16>>
        sparse_vector_fp16;
    if (is_dense) {
      // std::cout << "vector: " << vector.size() << std::endl;
      if (field_scheama->data_type() == DataType::VECTOR_FP16) {
        vector_fp16 = std::vector<ailego::Float16>(field_scheama->dimension(),
                                                   ailego::Float16(1.0f));
        vector_fp16[0] = 0;
        query.query_vector_.assign(
            (char *)vector_fp16.data(),
            vector_fp16.size() * sizeof(ailego::Float16));
      } else if (field_scheama->data_type() == DataType::VECTOR_FP32) {
        vector = std::vector<float>(field_scheama->dimension(), 1);
        vector[0] = 0;
        query.query_vector_.assign((char *)vector.data(),
                                   vector.size() * sizeof(float));
      } else {
        vector_int8 = std::vector<int8_t>(field_scheama->dimension(), 1);
        vector_int8[0] = 0;
        query.query_vector_.assign((char *)vector_int8.data(),
                                   vector_int8.size() * sizeof(int8_t));
      }
    } else {
      if (field_scheama->data_type() == DataType::SPARSE_VECTOR_FP32) {
        sparse_vector = {{1}, {1}};
        query.query_sparse_indices_.assign(
            (char *)sparse_vector.first.data(),
            sparse_vector.first.size() * sizeof(uint32_t));
        query.query_sparse_values_.assign(
            (char *)sparse_vector.second.data(),
            sparse_vector.second.size() * sizeof(float));
      } else {
        sparse_vector_fp16 = {{1}, {ailego::Float16(1.0f)}};
        query.query_sparse_indices_.assign(
            (char *)sparse_vector_fp16.first.data(),
            sparse_vector_fp16.first.size() * sizeof(uint32_t));
        query.query_sparse_values_.assign(
            (char *)sparse_vector_fp16.second.data(),
            sparse_vector_fp16.second.size() * sizeof(ailego::Float16));
      }
    }
    auto query_result = collection->Query(query);
    if (!query_result.has_value()) {
      std::cout << "status: " << query_result.error().message() << std::endl;
      ASSERT_TRUE(false);
    }
    ASSERT_TRUE(query_result.has_value());
    ASSERT_EQ(query_result.value().size(), doc_count);

    float last_score;
    for (size_t i = 0; i < query_result.value().size(); i++) {
      auto pk = query_result.value()[i]->pk();
      auto score = query_result.value()[i]->score();
      std::cout << "top " << i << ": " << pk << ", score: " << score
                << std::endl;

      auto expect_doc =
          TestHelper::CreateDoc(TestHelper::ExtractDocId(pk), *schema);
      float expect_score;
      if (is_dense) {
        if (field_scheama->data_type() == DataType::VECTOR_FP16) {
          auto query_result_vector =
              expect_doc.get<std::vector<ailego::Float16>>(field_name);
          ASSERT_TRUE(query_result_vector.has_value());
          expect_score = distance_dense(
              vector_fp16, query_result_vector.value(), metric_type);
        } else if (field_scheama->data_type() == DataType::VECTOR_FP32) {
          auto query_result_vector =
              expect_doc.get<std::vector<float>>(field_name);
          ASSERT_TRUE(query_result_vector.has_value());
          expect_score =
              distance_dense(vector, query_result_vector.value(), metric_type);
        } else {
          auto query_result_vector =
              expect_doc.get<std::vector<int8_t>>(field_name);
          ASSERT_TRUE(query_result_vector.has_value());
          expect_score = distance_dense(
              vector_int8, query_result_vector.value(), metric_type);
        }
      } else {
        if (field_scheama->data_type() == DataType::SPARSE_VECTOR_FP32) {
          auto query_result_vector =
              expect_doc
                  .get<std::pair<std::vector<uint32_t>, std::vector<float>>>(
                      field_name);
          ASSERT_TRUE(query_result_vector.has_value());
          expect_score =
              distance_sparse(sparse_vector, query_result_vector.value());
        } else {
          auto query_result_vector = expect_doc.get<
              std::pair<std::vector<uint32_t>, std::vector<ailego::Float16>>>(
              field_name);
          ASSERT_TRUE(query_result_vector.has_value());
          expect_score =
              distance_sparse(sparse_vector_fp16, query_result_vector.value());
        }
      }
      std::cout.precision(8);
      std::cout << "score: " << score << ", expect_score: " << expect_score
                << std::endl;
      // ASSERT_FLOAT_EQ(score, expect_score);
      if (i > 0) {
        if (metric_type == MetricType::L2) {
          ASSERT_GE(score, last_score);
        } else if (metric_type == MetricType::IP) {
          ASSERT_LE(score, last_score);
        }
      }
      last_score = score;
    }

    auto new_schema = std::make_shared<CollectionSchema>(*schema);
    s = new_schema->add_index(field_name, index_params);
    ASSERT_TRUE(s.ok());
    ASSERT_EQ(*new_schema, collection->Schema());


    for (int i = 0; i < doc_count; i++) {
      auto expect_doc = TestHelper::CreateDoc(i, *schema);
      auto result = collection->Fetch({expect_doc.pk()});
      ASSERT_TRUE(result.has_value());
      ASSERT_EQ(result.value().size(), 1);
      ASSERT_EQ(result.value().count(expect_doc.pk()), 1);
      auto doc = result.value()[expect_doc.pk()];
      ASSERT_NE(doc, nullptr);
      if (metric_type != MetricType::COSINE) {
        if (*doc != expect_doc) {
          std::cout << "       doc:" << doc->to_detail_string() << std::endl;
          std::cout << "expect_doc:" << expect_doc.to_detail_string()
                    << std::endl;
        }
        ASSERT_EQ(*doc, expect_doc);
      }
    }

    collection.reset();

    auto result = Collection::Open(col_path, options);
    ASSERT_TRUE(result.has_value());

    collection = result.value();
    stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, doc_count);
    ASSERT_EQ(stats.index_completeness[field_name], 1);

    for (int i = 0; i < doc_count; i++) {
      auto expect_doc = TestHelper::CreateDoc(i, *schema);
      auto result = collection->Fetch({expect_doc.pk()});
      ASSERT_TRUE(result.has_value());
      ASSERT_EQ(result.value().size(), 1);
      ASSERT_EQ(result.value().count(expect_doc.pk()), 1);
      auto doc = result.value()[expect_doc.pk()];
      ASSERT_NE(doc, nullptr);
      if (metric_type != MetricType::COSINE) {
        if (*doc != expect_doc) {
          std::cout << "       doc:" << doc->to_detail_string() << std::endl;
          std::cout << "expect_doc:" << expect_doc.to_detail_string()
                    << std::endl;
        }
        ASSERT_EQ(*doc, expect_doc);
      }
    }

    // insert another 100 docs
    s = TestHelper::CollectionInsertDoc(collection, doc_count, doc_count + 100,
                                        false);
    ASSERT_TRUE(s.ok());
    ASSERT_EQ(collection->Stats().value().doc_count, doc_count + 100);
    ASSERT_FLOAT_EQ(collection->Stats().value().index_completeness[field_name],
                    doc_count * 1.0 / (doc_count + 100));

    s = collection->Flush();
    ASSERT_TRUE(s.ok());

    s = collection->CreateIndex(field_name, index_params);
    ASSERT_TRUE(s.ok());
    ASSERT_EQ(collection->Stats().value().doc_count, doc_count + 100);
    ASSERT_FLOAT_EQ(collection->Stats().value().index_completeness[field_name],
                    doc_count * 1.0 / (doc_count + 100));
  };

  func("dense_fp32", MetricType::L2);
  func("dense_fp32", MetricType::COSINE);
  func("dense_fp32", MetricType::IP);
  func("dense_fp32", MetricType::L2, QuantizeType::FP16);
  func("dense_fp32", MetricType::COSINE, QuantizeType::FP16);
  func("dense_fp32", MetricType::IP, QuantizeType::FP16);
  func("dense_fp16");
  func("dense_int8");
  func("sparse_fp32");
  func("sparse_fp16");
}

TEST_F(CollectionTest, Feature_CreateIndex_Scalar) {
  auto func = [&](std::string field_name, bool enable_optimize,
                  IndexParams::Ptr scalar_index_params = nullptr) {
    FileHelper::RemoveDirectory(col_path);

    int doc_count = 1000;

    auto schema =
        TestHelper::CreateNormalSchema(false, "demo", scalar_index_params);
    auto options = CollectionOptions{false, true, 64 * 1024 * 1024};
    auto collection = TestHelper::CreateCollectionWithDoc(
        col_path, *schema, options, 0, doc_count, false);

    ASSERT_TRUE(collection->Flush().ok());

    auto stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, doc_count);
    ASSERT_EQ(stats.index_completeness["dense_fp32"], 1);

    auto index_params = std::make_shared<InvertIndexParams>(enable_optimize);
    auto s = collection->CreateIndex(field_name, index_params);
    std::cout << "status: " << s.message()
              << ", code: " << GetDefaultMessage(s.code()) << std::endl;
    ASSERT_TRUE(s.ok());

    auto new_schema = std::make_shared<CollectionSchema>(*schema);
    s = new_schema->add_index(field_name, index_params);
    ASSERT_TRUE(s.ok());
    ASSERT_EQ(*new_schema, collection->Schema());

    for (int i = 0; i < doc_count; i++) {
      auto expect_doc = TestHelper::CreateDoc(i, *schema);
      auto result = collection->Fetch({expect_doc.pk()});
      ASSERT_TRUE(result.has_value());
      ASSERT_EQ(result.value().size(), 1);
      ASSERT_EQ(result.value().count(expect_doc.pk()), 1);
      auto doc = result.value()[expect_doc.pk()];
      ASSERT_NE(doc, nullptr);
      if (*doc != expect_doc) {
        std::cout << "       doc:" << doc->to_detail_string() << std::endl;
        std::cout << "expect_doc:" << expect_doc.to_detail_string()
                  << std::endl;
      }
      ASSERT_EQ(*doc, expect_doc);
    }

    collection.reset();

    auto result = Collection::Open(col_path, options);
    ASSERT_TRUE(result.has_value());

    collection = result.value();
    stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, doc_count);
    ASSERT_EQ(stats.index_completeness["dense_fp32"], 1);

    for (int i = 0; i < doc_count; i++) {
      auto expect_doc = TestHelper::CreateDoc(i, *schema);
      auto result = collection->Fetch({expect_doc.pk()});
      ASSERT_TRUE(result.has_value());
      ASSERT_EQ(result.value().size(), 1);
      ASSERT_EQ(result.value().count(expect_doc.pk()), 1);
      auto doc = result.value()[expect_doc.pk()];
      ASSERT_NE(doc, nullptr);
      if (*doc != expect_doc) {
        std::cout << "       doc:" << doc->to_detail_string() << std::endl;
        std::cout << "expect_doc:" << expect_doc.to_detail_string()
                  << std::endl;
      }
      ASSERT_EQ(*doc, expect_doc);
    }

    // insert another 100 docs
    s = TestHelper::CollectionInsertDoc(collection, doc_count, doc_count + 100,
                                        false);
    ASSERT_TRUE(s.ok());
    ASSERT_EQ(collection->Stats().value().doc_count, doc_count + 100);
    ASSERT_FLOAT_EQ(
        collection->Stats().value().index_completeness["dense_fp32"], 1);

    s = collection->Flush();
    ASSERT_TRUE(s.ok());

    s = collection->CreateIndex(field_name, index_params);
    ASSERT_TRUE(s.ok());
    ASSERT_EQ(collection->Stats().value().doc_count, doc_count + 100);
    ASSERT_FLOAT_EQ(
        collection->Stats().value().index_completeness["dense_fp32"], 1);

    for (int i = 0; i < doc_count + 100; i++) {
      auto expect_doc = TestHelper::CreateDoc(i, *schema);
      auto result = collection->Fetch({expect_doc.pk()});
      ASSERT_TRUE(result.has_value());
      ASSERT_EQ(result.value().size(), 1);
      ASSERT_EQ(result.value().count(expect_doc.pk()), 1);
      auto doc = result.value()[expect_doc.pk()];
      ASSERT_NE(doc, nullptr);
      if (*doc != expect_doc) {
        std::cout << "       doc:" << doc->to_detail_string() << std::endl;
        std::cout << "expect_doc:" << expect_doc.to_detail_string()
                  << std::endl;
      }
      ASSERT_EQ(*doc, expect_doc);
    }
  };

  func("int32", true);
  func("int32", false);

  func("int32", false, std::make_shared<InvertIndexParams>(true));
  func("int32", true, std::make_shared<InvertIndexParams>(true));
}

TEST_F(CollectionTest, Feature_DropIndex_General) {
  // create empty collection
  auto schema = TestHelper::CreateSchemaWithVectorIndex();
  auto options = CollectionOptions{false, true, 64 * 1024 * 1204};
  auto collection = TestHelper::CreateCollectionWithDoc(col_path, *schema,
                                                        options, 0, 0, false);

  ASSERT_TRUE(collection->Flush().ok());
  auto stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, 0);
  ASSERT_EQ(stats.index_completeness["dense_fp32"], 1);

  ASSERT_EQ(collection->Schema(), *schema);


  auto s = collection->DropIndex("dense_fp32_invalid");
  ASSERT_FALSE(s.ok());

  s = collection->DropIndex("dense_fp32");
  if (!s.ok()) {
    std::cout << "drop index err: " << s.message() << std::endl;
  }
  ASSERT_TRUE(s.ok());

  s = collection->DropIndex("dense_fp32");
  ASSERT_TRUE(s.ok());

  auto new_schema = std::make_shared<CollectionSchema>(*schema);
  s = new_schema->drop_index("dense_fp32");
  ASSERT_TRUE(s.ok());
  ASSERT_EQ(*new_schema, collection->Schema());

  stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, 0);
  ASSERT_EQ(stats.index_completeness["dense_fp32"], 1);

  ASSERT_EQ(*collection->Schema()
                 .value()
                 .get_vector_field("dense_fp32")
                 ->index_params(),
            DefaultVectorIndexParams);

  s = collection->DropIndex("dense_fp32");
  if (!s.ok()) {
    std::cout << "drop index err: " << s.message() << std::endl;
  }
  ASSERT_TRUE(s.ok());

  auto schema1 = collection->Schema().value();

  collection.reset();

  auto result = Collection::Open(col_path, options);
  ASSERT_TRUE(result.has_value());

  collection = std::move(result.value());
  auto schema2 = collection->Schema().value();

  if (schema1 != schema2) {
    std::cout << "schema1: " << schema1.to_string_formatted() << std::endl;
    std::cout << "schema2: " << schema2.to_string_formatted() << std::endl;
  }
  ASSERT_EQ(schema1, schema2);

  stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, 0);
  ASSERT_EQ(stats.index_completeness["dense_fp32"], 1);
}

TEST_F(CollectionTest, Feature_DropIndex_Vector) {
  auto func = [&](const std::string &field_name, bool add_before_drop = true) {
    FileHelper::RemoveDirectory(col_path);

    int doc_count = 1000;

    // create empty collection
    auto schema = TestHelper::CreateNormalSchema();
    auto options = CollectionOptions{false, true, 64 * 1024 * 1204};
    auto collection = TestHelper::CreateCollectionWithDoc(
        col_path, *schema, options, 0, doc_count, false);

    ASSERT_TRUE(collection->Flush().ok());

    auto stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, doc_count);
    ASSERT_EQ(stats.index_completeness[field_name], 1);
    ASSERT_EQ(collection->Schema(), *schema);

    auto check_doc = [&]() {
      for (int i = 0; i < doc_count; i++) {
        auto expect_doc = TestHelper::CreateDoc(i, *schema);
        auto result = collection->Fetch({expect_doc.pk()});
        ASSERT_TRUE(result.has_value());
        ASSERT_EQ(result.value().size(), 1);
        ASSERT_EQ(result.value().count(expect_doc.pk()), 1);
        auto doc = result.value()[expect_doc.pk()];
        ASSERT_NE(doc, nullptr);
        if (*doc != expect_doc) {
          std::cout << "       doc:" << doc->to_detail_string() << std::endl;
          std::cout << "expect_doc:" << expect_doc.to_detail_string()
                    << std::endl;
        }
        ASSERT_EQ(*doc, expect_doc);
      }
    };

    check_doc();
    std::cout << "check success 1" << std::endl;

    // create index first
    auto index_params = std::make_shared<HnswIndexParams>(MetricType::IP);
    auto s = collection->CreateIndex(field_name, index_params);
    ASSERT_TRUE(s.ok());
    auto new_schema = std::make_shared<CollectionSchema>(*schema);
    s = new_schema->add_index(field_name, index_params);
    ASSERT_TRUE(s.ok());
    ASSERT_EQ(*new_schema, collection->Schema());
    stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, doc_count);
    ASSERT_EQ(stats.index_completeness[field_name], 1);

    check_doc();
    std::cout << "check success 2" << std::endl;

    int new_doc_count = doc_count;
    if (add_before_drop) {
      new_doc_count += doc_count;
      s = TestHelper::CollectionInsertDoc(collection, doc_count, new_doc_count);
      ASSERT_TRUE(s.ok());
    }

    // then drop index field_name
    s = collection->DropIndex(field_name);
    ASSERT_TRUE(s.ok());
    check_doc();
    std::cout << "check success 3" << std::endl;
    s = new_schema->drop_index(field_name);
    ASSERT_TRUE(s.ok());
    ASSERT_EQ(*new_schema, collection->Schema());

    stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, new_doc_count);
    ASSERT_EQ(stats.index_completeness[field_name], 1);

    collection.reset();
    auto result = Collection::Open(col_path, options);
    ASSERT_TRUE(result.has_value());
    collection = std::move(result.value());

    check_doc();
    std::cout << "check success 3" << std::endl;
    stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, new_doc_count);
    ASSERT_EQ(stats.index_completeness[field_name], 1);
  };

  func("dense_fp32", true);
  func("dense_fp32", false);
  func("sparse_fp32");
}

TEST_F(CollectionTest, Feature_DropIndex_Scalar) {
  auto func = [&](std::string field_name, bool enable_optimize) {
    FileHelper::RemoveDirectory(col_path);

    int doc_count = 1000;

    auto schema =
        TestHelper::CreateSchemaWithScalarIndex(false, enable_optimize);
    auto options = CollectionOptions{false, true, 64 * 1024 * 1024};
    auto collection = TestHelper::CreateCollectionWithDoc(
        col_path, *schema, options, 0, doc_count, false);

    ASSERT_TRUE(collection->Flush().ok());

    auto check_doc = [&]() {
      for (int i = 0; i < doc_count; i++) {
        auto expect_doc = TestHelper::CreateDoc(i, *schema);
        auto result = collection->Fetch({expect_doc.pk()});
        ASSERT_TRUE(result.has_value());
        ASSERT_EQ(result.value().size(), 1);
        ASSERT_EQ(result.value().count(expect_doc.pk()), 1);
        auto doc = result.value()[expect_doc.pk()];
        ASSERT_NE(doc, nullptr);
        if (*doc != expect_doc) {
          std::cout << "       doc:" << doc->to_detail_string() << std::endl;
          std::cout << "expect_doc:" << expect_doc.to_detail_string()
                    << std::endl;
        }
        ASSERT_EQ(*doc, expect_doc);
      }
    };

    check_doc();
    std::cout << "check success 1" << std::endl;

    auto stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, doc_count);

    auto s = collection->DropIndex(field_name);
    ASSERT_TRUE(s.ok());

    auto new_schema = std::make_shared<CollectionSchema>(*schema);
    s = new_schema->drop_index(field_name);
    ASSERT_TRUE(s.ok());
    ASSERT_EQ(*new_schema, collection->Schema());

    check_doc();
    std::cout << "check success 2" << std::endl;
    stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, doc_count);

    collection.reset();
    auto result = Collection::Open(col_path, options);
    ASSERT_TRUE(result.has_value());
    collection = std::move(result.value());

    check_doc();
    std::cout << "check success 3" << std::endl;
    stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, doc_count);
  };

  func("int32", true);
  func("int32", false);
}

TEST_F(CollectionTest, Feature_DropIndex_AfterCreate) {
  auto func = [&](std::string field_name, bool enable_optimize) {
    FileHelper::RemoveDirectory(col_path);

    int doc_count = 1000;

    auto schema = TestHelper::CreateNormalSchema();
    auto options = CollectionOptions{false, true, 64 * 1024 * 1024};
    auto collection = TestHelper::CreateCollectionWithDoc(
        col_path, *schema, options, 0, doc_count, false);

    ASSERT_TRUE(collection->Flush().ok());

    auto check_doc = [&]() {
      for (int i = 0; i < doc_count; i++) {
        auto expect_doc = TestHelper::CreateDoc(i, *schema);
        auto result = collection->Fetch({expect_doc.pk()});
        ASSERT_TRUE(result.has_value());
        ASSERT_EQ(result.value().size(), 1);
        ASSERT_EQ(result.value().count(expect_doc.pk()), 1);
        auto doc = result.value()[expect_doc.pk()];
        ASSERT_NE(doc, nullptr);
        if (*doc != expect_doc) {
          std::cout << "       doc:" << doc->to_detail_string() << std::endl;
          std::cout << "expect_doc:" << expect_doc.to_detail_string()
                    << std::endl;
        }
        ASSERT_EQ(*doc, expect_doc);
      }
    };

    check_doc();
    std::cout << "check success 1" << std::endl;

    auto stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, doc_count);

    auto index_params = std::make_shared<InvertIndexParams>(enable_optimize);
    auto s = collection->CreateIndex(field_name, index_params);
    std::cout << "status: " << s.message()
              << ", code: " << GetDefaultMessage(s.code()) << std::endl;
    ASSERT_TRUE(s.ok());

    auto new_schema = std::make_shared<CollectionSchema>(*schema);
    s = new_schema->add_index(field_name, index_params);
    ASSERT_TRUE(s.ok());
    ASSERT_EQ(*new_schema, collection->Schema());

    check_doc();
    std::cout << "check success 2" << std::endl;

    s = collection->DropIndex(field_name);
    ASSERT_TRUE(s.ok());
    check_doc();
    std::cout << "check success 3" << std::endl;
    s = new_schema->drop_index(field_name);
    ASSERT_TRUE(s.ok());
    ASSERT_EQ(*new_schema, collection->Schema());
    stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, doc_count);
  };

  func("int32", true);
  func("int32", false);
}

TEST_F(CollectionTest, Feature_Optimize_General) {
  auto func = [](int concurrency) {
    FileHelper::RemoveDirectory(col_path);

    int doc_count = 1000;

    // create empty collection
    auto schema = TestHelper::CreateSchemaWithVectorIndex();
    auto options = CollectionOptions{false, true, 64 * 1024 * 1024};
    auto collection = TestHelper::CreateCollectionWithDoc(
        col_path, *schema, options, 0, doc_count, false);

    auto check_doc = [&]() {
      for (int i = 0; i < doc_count; i++) {
        auto expect_doc = TestHelper::CreateDoc(i, *schema);
        auto result = collection->Fetch({expect_doc.pk()});
        ASSERT_TRUE(result.has_value());
        ASSERT_EQ(result.value().size(), 1);
        ASSERT_EQ(result.value().count(expect_doc.pk()), 1);
        auto doc = result.value()[expect_doc.pk()];
        ASSERT_NE(doc, nullptr);
        if (*doc != expect_doc) {
          std::cout << "       doc:" << doc->to_detail_string() << std::endl;
          std::cout << "expect_doc:" << expect_doc.to_detail_string()
                    << std::endl;
        }
        ASSERT_EQ(*doc, expect_doc);
      }
    };

    check_doc();
    std::cout << "check success 1" << std::endl;

    ASSERT_TRUE(collection->Flush().ok());
    auto stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, doc_count);
    ASSERT_EQ(stats.index_completeness["dense_fp32"], 0);

    auto s = collection->Optimize(OptimizeOptions{concurrency});
    if (!s.ok()) {
      std::cout << s.message() << std::endl;
    }
    ASSERT_TRUE(s.ok());

    stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, doc_count);
    ASSERT_EQ(stats.index_completeness["dense_fp32"], 1);

    check_doc();
    std::cout << "check success 2" << std::endl;

    collection.reset();
    auto result = Collection::Open(col_path, options);
    ASSERT_TRUE(result.has_value());
    collection = std::move(result.value());

    check_doc();
    std::cout << "check success 3" << std::endl;
  };

  func(0);
  func(4);
}

TEST_F(CollectionTest, Feature_Optimize_Repeated) {
  int doc_count = 1000;

  // create empty collection
  auto schema = TestHelper::CreateSchemaWithVectorIndex();
  auto options = CollectionOptions{false, true, 64 * 1024 * 1024};
  auto collection = TestHelper::CreateCollectionWithDoc(
      col_path, *schema, options, 0, doc_count, false);

  auto check_doc = [&]() {
    for (int i = 0; i < doc_count; i++) {
      auto expect_doc = TestHelper::CreateDoc(i, *schema);
      auto result = collection->Fetch({expect_doc.pk()});
      ASSERT_TRUE(result.has_value());
      ASSERT_EQ(result.value().size(), 1);
      ASSERT_EQ(result.value().count(expect_doc.pk()), 1);
      auto doc = result.value()[expect_doc.pk()];
      if (doc == nullptr) {
        std::cout << "doc is null, pk: " << expect_doc.pk() << std::endl;
      }
      ASSERT_NE(doc, nullptr);
      if (*doc != expect_doc) {
        std::cout << "       doc:" << doc->to_detail_string() << std::endl;
        std::cout << "expect_doc:" << expect_doc.to_detail_string()
                  << std::endl;
      }
      ASSERT_EQ(*doc, expect_doc);
    }
  };

  check_doc();
  std::cout << "check success 1" << std::endl;

  ASSERT_TRUE(collection->Flush().ok());
  auto stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, doc_count);
  ASSERT_EQ(stats.index_completeness["dense_fp32"], 0);

  auto s = collection->Optimize();
  ASSERT_TRUE(s.ok());
  stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, doc_count);
  ASSERT_EQ(stats.index_completeness["dense_fp32"], 1);

  int loop_count = 10;
  uint64_t start_doc_id = doc_count;
  for (int i = 0; i < loop_count; i++) {
    std::cout << "loop: " << i << " begin" << std::endl;

    s = TestHelper::CollectionInsertDoc(collection, start_doc_id,
                                        start_doc_id + 1);
    ASSERT_TRUE(s.ok());

    stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, doc_count + i + 1);
    ASSERT_FLOAT_EQ(stats.index_completeness["dense_fp32"],
                    1.0 * (doc_count + i) / (doc_count + i + 1));


    s = collection->Optimize();
    if (!s.ok()) {
      std::cout << "optimize failed: " << s.message() << std::endl;
    }
    ASSERT_TRUE(s.ok());

    start_doc_id += 1;

    std::cout << "loop: " << i << " end" << std::endl;
  }

  stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, doc_count + loop_count);
  ASSERT_EQ(stats.index_completeness["dense_fp32"], 1);

  doc_count += loop_count;
  check_doc();
  std::cout << "check success 2" << std::endl;
}

TEST_F(CollectionTest, Feature_Optimize_MetricType) {
  auto func = [&](MetricType metric_type,
                  QuantizeType quantize_type = QuantizeType::UNDEFINED) {
    FileHelper::RemoveDirectory(col_path);

    int doc_count = 1000;

    // create empty collection
    auto schema = TestHelper::CreateSchemaWithVectorIndex(
        false, "demo",
        std::make_shared<HnswIndexParams>(metric_type, 16, 200, quantize_type));
    auto options = CollectionOptions{false, true, 64 * 1024 * 1024};
    auto collection = TestHelper::CreateCollectionWithDoc(
        col_path, *schema, options, 0, doc_count, false);

    auto check_doc = [&]() {
      for (int i = 0; i < doc_count; i++) {
        auto expect_doc = TestHelper::CreateDoc(i, *schema);
        auto result = collection->Fetch({expect_doc.pk()});
        ASSERT_TRUE(result.has_value());
        ASSERT_EQ(result.value().size(), 1);
        ASSERT_EQ(result.value().count(expect_doc.pk()), 1);
        auto doc = result.value()[expect_doc.pk()];
        ASSERT_NE(doc, nullptr);
        if (metric_type != MetricType::COSINE) {
          if (*doc != expect_doc) {
            std::cout << "       doc:" << doc->to_detail_string() << std::endl;
            std::cout << "expect_doc:" << expect_doc.to_detail_string()
                      << std::endl;
          }
          ASSERT_EQ(*doc, expect_doc);
        }
      }
    };

    check_doc();
    std::cout << "check success 1" << std::endl;

    ASSERT_TRUE(collection->Flush().ok());
    auto stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, doc_count);
    ASSERT_EQ(stats.index_completeness["dense_fp32"], 0);

    auto s = collection->Optimize();
    ASSERT_TRUE(s.ok());

    stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, doc_count);
    ASSERT_EQ(stats.index_completeness["dense_fp32"], 1);

    check_doc();
    std::cout << "check success 2" << std::endl;

    for (int i = 1; i < 2; i++) {
      auto query_doc = TestHelper::CreateDoc(i, *schema);
      // std::cout << query_doc.to_detail_string() << std::endl;

      VectorQuery query;
      query.topk_ = 10;
      query.include_vector_ = true;
      query.field_name_ = "dense_fp32";

      auto vector = query_doc.get<std::vector<float>>("dense_fp32");
      ASSERT_TRUE(vector.has_value());
      query.query_vector_.assign((char *)vector.value().data(),
                                 vector.value().size() * sizeof(float));


      auto result = collection->Query(query);
      if (!result.has_value()) {
        std::cout << "err: " << result.error().message() << std::endl;
      }
      ASSERT_TRUE(result.has_value());
      ASSERT_EQ(result.value().size(), std::min(query.topk_, doc_count));
    }
  };

  func(MetricType::L2);
  func(MetricType::COSINE);
  func(MetricType::IP);
  func(MetricType::L2, QuantizeType::FP16);
  func(MetricType::COSINE, QuantizeType::FP16);
  func(MetricType::IP, QuantizeType::FP16);
}

TEST_F(CollectionTest, Feature_Optimize_Delete) {
  int doc_count = 1000;

  // create empty collection
  auto schema = TestHelper::CreateSchemaWithVectorIndex();
  auto options = CollectionOptions{false, true, 64 * 1024 * 1024};
  auto collection = TestHelper::CreateCollectionWithDoc(
      col_path, *schema, options, 0, doc_count, false);

  auto check_doc = [&]() {
    for (int i = 0; i < doc_count; i++) {
      auto expect_doc = TestHelper::CreateDoc(i, *schema);
      auto result = collection->Fetch({expect_doc.pk()});
      ASSERT_TRUE(result.has_value());
      ASSERT_EQ(result.value().size(), 1);
      ASSERT_EQ(result.value().count(expect_doc.pk()), 1);
      auto doc = result.value()[expect_doc.pk()];
      ASSERT_NE(doc, nullptr);
      if (*doc != expect_doc) {
        std::cout << "       doc:" << doc->to_detail_string() << std::endl;
        std::cout << "expect_doc:" << expect_doc.to_detail_string()
                  << std::endl;
      }
      ASSERT_EQ(*doc, expect_doc);
    }
  };

  check_doc();
  std::cout << "check success 1" << std::endl;

  ASSERT_TRUE(collection->Flush().ok());
  auto stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, doc_count);
  ASSERT_EQ(stats.index_completeness["dense_fp32"], 0);

  auto s = collection->Optimize();
  if (!s.ok()) {
    std::cout << s.message() << std::endl;
  }
  ASSERT_TRUE(s.ok());

  stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, doc_count);
  ASSERT_EQ(stats.index_completeness["dense_fp32"], 1);

  check_doc();
  std::cout << "check success 2" << std::endl;

  // delete all docs
  std::vector<std::string> pks;
  for (int i = 0; i < doc_count; ++i) {
    pks.push_back(TestHelper::MakePK(i));
  }
  auto res = collection->Delete(pks);
  ASSERT_TRUE(res.has_value());
  for (auto &r : res.value()) {
    ASSERT_TRUE(r.ok());
  }

  stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, 0);
  ASSERT_EQ(stats.index_completeness["dense_fp32"], 1);

  s = collection->Optimize();
  if (!s.ok()) {
    std::cout << s.message() << std::endl;
  }
  ASSERT_TRUE(s.ok());

  collection.reset();
  auto result = Collection::Open(col_path, options);
  ASSERT_TRUE(result.has_value());
  collection = std::move(result.value());

  stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, 0);
  ASSERT_EQ(stats.index_completeness["dense_fp32"], 1);
}

TEST_F(CollectionTest, Feature_Optimize_NormalSchema) {
  int doc_count = 1000;

  // create empty collection
  auto schema = TestHelper::CreateNormalSchema();
  auto options = CollectionOptions{false, true, 64 * 1024 * 1024};
  auto collection = TestHelper::CreateCollectionWithDoc(
      col_path, *schema, options, 0, doc_count, false);

  auto check_doc = [&]() {
    for (int i = 0; i < doc_count; i++) {
      auto expect_doc = TestHelper::CreateDoc(i, *schema);
      auto result = collection->Fetch({expect_doc.pk()});
      ASSERT_TRUE(result.has_value());
      ASSERT_EQ(result.value().size(), 1);
      ASSERT_EQ(result.value().count(expect_doc.pk()), 1);
      auto doc = result.value()[expect_doc.pk()];
      ASSERT_NE(doc, nullptr);
      if (*doc != expect_doc) {
        std::cout << "       doc:" << doc->to_detail_string() << std::endl;
        std::cout << "expect_doc:" << expect_doc.to_detail_string()
                  << std::endl;
      }
      ASSERT_EQ(*doc, expect_doc);
    }
  };

  check_doc();
  std::cout << "check success 1" << std::endl;

  ASSERT_TRUE(collection->Flush().ok());
  auto stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, doc_count);
  ASSERT_EQ(stats.index_completeness["dense_fp32"], 1);

  auto s = collection->Optimize();
  if (!s.ok()) {
    std::cout << s.message() << std::endl;
  }
  ASSERT_TRUE(s.ok());

  stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, doc_count);
  ASSERT_EQ(stats.index_completeness["dense_fp32"], 1);

  check_doc();
  std::cout << "check success 2" << std::endl;

  collection.reset();
  auto result = Collection::Open(col_path, options);
  ASSERT_TRUE(result.has_value());
  collection = std::move(result.value());

  check_doc();
  std::cout << "check success 3" << std::endl;
}

TEST_F(CollectionTest, Feature_Optimize_ExceedMaxDocCount) {
  auto func = [&](std::vector<int> segments_count, bool delete_all = false) {
    FileHelper::RemoveDirectory(col_path);

    int max_doc_per_count = 1000;

    // create empty collection
    auto schema = TestHelper::CreateNormalSchema(
        false, "demo", nullptr,
        std::make_shared<HnswIndexParams>(MetricType::IP), max_doc_per_count);
    auto options = CollectionOptions{false, true, 64 * 1024 * 1024};

    auto collection = TestHelper::CreateCollectionWithDoc(col_path, *schema,
                                                          options, 0, 0, false);

    auto check_doc = [&](int doc_count) {
      for (int i = 0; i < doc_count; i++) {
        auto expect_doc = TestHelper::CreateDoc(i, *schema);
        auto result = collection->Fetch({expect_doc.pk()});
        ASSERT_TRUE(result.has_value());
        ASSERT_EQ(result.value().size(), 1);
        ASSERT_EQ(result.value().count(expect_doc.pk()), 1);
        auto doc = result.value()[expect_doc.pk()];
        ASSERT_NE(doc, nullptr);
        if (*doc != expect_doc) {
          std::cout << "       doc:" << doc->to_detail_string() << std::endl;
          std::cout << "expect_doc:" << expect_doc.to_detail_string()
                    << std::endl;
        }
        ASSERT_EQ(*doc, expect_doc);
      }
    };

    int accu_seg_doc_count = 0;
    for (auto doc_count : segments_count) {
      auto s = TestHelper::CollectionInsertDoc(collection, accu_seg_doc_count,
                                               accu_seg_doc_count + doc_count);

      check_doc(accu_seg_doc_count + doc_count);
      std::cout << "check success 1" << std::endl;

      ASSERT_TRUE(collection->Flush().ok());
      auto stats = collection->Stats().value();
      ASSERT_EQ(stats.doc_count, accu_seg_doc_count + doc_count);
      ASSERT_FLOAT_EQ(
          stats.index_completeness["dense_fp32"],
          accu_seg_doc_count * 1.0 / (accu_seg_doc_count + doc_count));

      s = collection->Optimize();
      if (!s.ok()) {
        std::cout << s.message() << std::endl;
      }
      ASSERT_TRUE(s.ok());

      stats = collection->Stats().value();
      ASSERT_EQ(stats.doc_count, accu_seg_doc_count + doc_count);
      ASSERT_EQ(stats.index_completeness["dense_fp32"], 1);

      check_doc(accu_seg_doc_count + doc_count);
      std::cout << "check success 2" << std::endl;

      collection.reset();
      auto result = Collection::Open(col_path, options);
      ASSERT_TRUE(result.has_value());
      collection = std::move(result.value());

      check_doc(accu_seg_doc_count + doc_count);
      std::cout << "check success 3" << std::endl;

      accu_seg_doc_count += doc_count;
    }

    // delete all docs
    if (delete_all) {
      std::vector<std::string> pks;
      for (int i = 0; i < accu_seg_doc_count; ++i) {
        pks.push_back(TestHelper::MakePK(i));
      }
      auto res = collection->Delete(pks);
      ASSERT_TRUE(res.has_value());
      for (auto &r : res.value()) {
        ASSERT_TRUE(r.ok());
      }
    }

    auto s = collection->Optimize();
    if (!s.ok()) {
      std::cout << s.message() << std::endl;
    }
    ASSERT_TRUE(s.ok());

    if (delete_all) {
      check_doc(0);
    } else {
      check_doc(accu_seg_doc_count);
    }
    std::cout << "check success 3" << std::endl;

    auto stats = collection->Stats().value();
    if (delete_all) {
      ASSERT_EQ(stats.doc_count, 0);
    } else {
      ASSERT_EQ(stats.doc_count, accu_seg_doc_count);
    }
    ASSERT_FLOAT_EQ(stats.index_completeness["dense_fp32"], 1.0);

    collection.reset();
    auto result = Collection::Open(col_path, options);
    ASSERT_TRUE(result.has_value());
    collection = std::move(result.value());

    stats = collection->Stats().value();
    if (delete_all) {
      ASSERT_EQ(stats.doc_count, 0);
    } else {
      ASSERT_EQ(stats.doc_count, accu_seg_doc_count);
    }
    ASSERT_FLOAT_EQ(stats.index_completeness["dense_fp32"], 1.0);
  };

  func({600, 600});
  func({600, 400});
  func({600, 401});

  func({600, 600}, true);
  func({600, 400}, true);
  func({600, 401}, true);

  func(std::vector<int>(100, 1));
  func(std::vector<int>(100, 1), true);
}

TEST_F(CollectionTest, Feature_Optimize_Rebuild) {
  FileHelper::RemoveDirectory(col_path);

  int max_doc_per_count = 1000;

  // create empty collection
  auto schema = TestHelper::CreateNormalSchema(
      false, "demo", nullptr, std::make_shared<HnswIndexParams>(MetricType::IP),
      max_doc_per_count);
  auto options = CollectionOptions{false, true, 64 * 1024 * 1024};

  // create seg1
  auto collection = TestHelper::CreateCollectionWithDoc(
      col_path, *schema, options, 0, max_doc_per_count, false);

  auto check_doc = [&](int doc_count, bool delete_half = false) {
    for (int i = 0; i < doc_count; i++) {
      if (delete_half) {
        if (i % 2 == 0) {
          continue;
        }
      }

      auto expect_doc = TestHelper::CreateDoc(i, *schema);
      auto result = collection->Fetch({expect_doc.pk()});
      ASSERT_TRUE(result.has_value());
      ASSERT_EQ(result.value().size(), 1);
      ASSERT_EQ(result.value().count(expect_doc.pk()), 1);
      auto doc = result.value()[expect_doc.pk()];
      ASSERT_NE(doc, nullptr);
      if (*doc != expect_doc) {
        std::cout << "       doc:" << doc->to_detail_string() << std::endl;
        std::cout << "expect_doc:" << expect_doc.to_detail_string()
                  << std::endl;
      }
      ASSERT_EQ(*doc, expect_doc);
    }
  };

  ASSERT_TRUE(collection->Flush().ok());
  auto stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, max_doc_per_count);
  ASSERT_EQ(stats.index_completeness["dense_fp32"], 0);

  // create seg2
  auto s = TestHelper::CollectionInsertDoc(
      collection, max_doc_per_count, max_doc_per_count + max_doc_per_count);
  ASSERT_TRUE(s.ok());
  stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, max_doc_per_count + max_doc_per_count);
  ASSERT_FLOAT_EQ(stats.index_completeness["dense_fp32"], 0);

  // create seg3
  s = TestHelper::CollectionInsertDoc(collection, max_doc_per_count * 2,
                                      max_doc_per_count * 3);
  ASSERT_TRUE(s.ok());
  stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, max_doc_per_count * 3);
  ASSERT_FLOAT_EQ(stats.index_completeness["dense_fp32"], 0);

  check_doc(max_doc_per_count * 3);
  std::cout << "check success 1" << std::endl;

  // delete half
  std::vector<std::string> pks;
  for (int j = 0; j < 3 * max_doc_per_count; j++) {
    if (j % 2 == 0) {
      pks.push_back(TestHelper::MakePK(j));
    }
  }
  auto res = collection->Delete(pks);
  ASSERT_TRUE(res.has_value());
  for (auto &r : res.value()) {
    ASSERT_TRUE(r.ok());
  }

  s = collection->Optimize();
  if (!s.ok()) {
    std::cout << s.message() << std::endl;
  }
  ASSERT_TRUE(s.ok());

  check_doc(max_doc_per_count * 3, true);
  std::cout << "check success 2" << std::endl;

  stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, max_doc_per_count * 1.5);
  ASSERT_FLOAT_EQ(stats.index_completeness["dense_fp32"], 1);
}

TEST_F(CollectionTest, Feature_Optimize_IndexOperation) {
  FileHelper::RemoveDirectory(col_path);

  int max_doc_per_count = 1000;

  // create empty collection
  auto schema = TestHelper::CreateNormalSchema(
      false, "demo", nullptr, std::make_shared<HnswIndexParams>(MetricType::IP),
      max_doc_per_count);
  auto options = CollectionOptions{false, true, 64 * 1024 * 1024};

  // create seg1
  auto collection = TestHelper::CreateCollectionWithDoc(
      col_path, *schema, options, 0, max_doc_per_count / 2, false);

  auto check_doc = [&](int doc_count) {
    for (int i = 0; i < doc_count; i++) {
      auto expect_doc = TestHelper::CreateDoc(i, *schema);
      auto result = collection->Fetch({expect_doc.pk()});
      ASSERT_TRUE(result.has_value());
      ASSERT_EQ(result.value().size(), 1);
      ASSERT_EQ(result.value().count(expect_doc.pk()), 1);
      auto doc = result.value()[expect_doc.pk()];
      ASSERT_NE(doc, nullptr);
      if (*doc != expect_doc) {
        std::cout << "       doc:" << doc->to_detail_string() << std::endl;
        std::cout << "expect_doc:" << expect_doc.to_detail_string()
                  << std::endl;
      }
      ASSERT_EQ(*doc, expect_doc);
    }
  };

  auto stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, max_doc_per_count / 2);
  ASSERT_EQ(stats.index_completeness["dense_fp32"], 0);
  auto s = collection->DropIndex("dense_fp32");
  ASSERT_TRUE(s.ok());
  stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, max_doc_per_count / 2);
  ASSERT_EQ(stats.index_completeness["dense_fp32"], 1);

  // create seg2
  s = TestHelper::CollectionInsertDoc(collection, max_doc_per_count / 2,
                                      max_doc_per_count);
  ASSERT_TRUE(s.ok());
  s = collection->CreateIndex(
      "dense_fp32", std::make_shared<HnswIndexParams>(MetricType::IP));
  ASSERT_TRUE(s.ok());
  stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, max_doc_per_count);
  ASSERT_EQ(stats.index_completeness["dense_fp32"], 1);

  // create seg3
  s = TestHelper::CollectionInsertDoc(collection, max_doc_per_count,
                                      max_doc_per_count * 3 / 2);
  ASSERT_TRUE(s.ok());
  s = collection->DropIndex("dense_fp32");
  ASSERT_TRUE(s.ok());
  stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, max_doc_per_count * 3 / 2);
  ASSERT_EQ(stats.index_completeness["dense_fp32"], 1);

  check_doc(max_doc_per_count * 3 / 2);
  std::cout << "check success 1" << std::endl;

  s = collection->Optimize();
  if (!s.ok()) {
    std::cout << s.message() << std::endl;
  }
  ASSERT_TRUE(s.ok());

  check_doc(max_doc_per_count * 3 / 2);
  std::cout << "check success 2" << std::endl;

  stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, max_doc_per_count * 3 / 2);
  ASSERT_FLOAT_EQ(stats.index_completeness["dense_fp32"], 1);

  // reset collection
  collection.reset();
  auto result = Collection::Open(col_path, options);
  collection = std::move(result.value());

  check_doc(max_doc_per_count * 3 / 2);
  std::cout << "check success 2" << std::endl;

  stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, max_doc_per_count * 3 / 2);
  ASSERT_FLOAT_EQ(stats.index_completeness["dense_fp32"], 1);
}

TEST_F(CollectionTest, Feature_Optimize_Temp) {
  auto schema = TestHelper::CreateTempSchema();
  auto options = CollectionOptions{false, true, 64 * 1024 * 1024};

  auto collection =
      TestHelper::CreateCollectionWithDoc(col_path, *schema, options, 0, 10);

  auto s = collection->Optimize(OptimizeOptions{1});
  ASSERT_TRUE(s.ok());
}

TEST_F(CollectionTest, Feature_Query_Validate) {
  FileHelper::RemoveDirectory(col_path);

  int doc_count = 1100;
  // create with normal schema
  auto schema = TestHelper::CreateNormalSchema();
  auto options = CollectionOptions{false, true, 100 * 1024 * 1024};
  auto collection = TestHelper::CreateCollectionWithDoc(col_path, *schema,
                                                        options, 0, doc_count);

  ASSERT_NE(collection, nullptr);
  std::string field_name = "dense_fp32";
  auto query_doc = TestHelper::CreateDoc(1, *schema);

  {
    VectorQuery query;
    query.topk_ = 1024;
    query.field_name_ = field_name;

    auto field_scheama = schema->get_vector_field(field_name);
    ASSERT_NE(field_scheama, nullptr);
    ASSERT_TRUE(field_scheama->is_vector_field());

    if (field_scheama->is_dense_vector()) {
      auto vector = query_doc.get<std::vector<float>>(field_name);
      ASSERT_TRUE(vector.has_value());
      query.query_vector_.assign((char *)vector.value().data(),
                                 vector.value().size() * sizeof(float));
    } else {
      auto sparse_vector =
          query_doc.get<std::pair<std::vector<uint32_t>, std::vector<float>>>(
              field_name);
      query.query_sparse_indices_.assign(
          (char *)sparse_vector.value().first.data(),
          sparse_vector.value().first.size() * sizeof(uint32_t));
      query.query_sparse_values_.assign(
          (char *)sparse_vector.value().second.data(),
          sparse_vector.value().second.size() * sizeof(float));
    }
    query.include_vector_ = true;

    auto result = collection->Query(query);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result.value().size(), query.topk_);
  }

  {
    VectorQuery query;
    query.topk_ = 1025;
    query.field_name_ = field_name;

    auto field_scheama = schema->get_vector_field(field_name);
    ASSERT_NE(field_scheama, nullptr);
    ASSERT_TRUE(field_scheama->is_vector_field());

    if (field_scheama->is_dense_vector()) {
      auto vector = query_doc.get<std::vector<float>>(field_name);
      ASSERT_TRUE(vector.has_value());
      query.query_vector_.assign((char *)vector.value().data(),
                                 vector.value().size() * sizeof(float));
    } else {
      auto sparse_vector =
          query_doc.get<std::pair<std::vector<uint32_t>, std::vector<float>>>(
              field_name);
      query.query_sparse_indices_.assign(
          (char *)sparse_vector.value().first.data(),
          sparse_vector.value().first.size() * sizeof(uint32_t));
      query.query_sparse_values_.assign(
          (char *)sparse_vector.value().second.data(),
          sparse_vector.value().second.size() * sizeof(float));
    }
    query.include_vector_ = true;

    auto result = collection->Query(query);
    ASSERT_FALSE(result.has_value());
    std::cout << result.error().message() << std::endl;
  }

  {
    VectorQuery query;
    query.topk_ = 1024;
    query.field_name_ = field_name;
    query.output_fields_ = std::make_optional<std::vector<std::string>>(
        std::vector<std::string>(1025));

    auto field_scheama = schema->get_vector_field(field_name);
    ASSERT_NE(field_scheama, nullptr);
    ASSERT_TRUE(field_scheama->is_vector_field());

    if (field_scheama->is_dense_vector()) {
      auto vector = query_doc.get<std::vector<float>>(field_name);
      ASSERT_TRUE(vector.has_value());
      query.query_vector_.assign((char *)vector.value().data(),
                                 vector.value().size() * sizeof(float));
    } else {
      auto sparse_vector =
          query_doc.get<std::pair<std::vector<uint32_t>, std::vector<float>>>(
              field_name);
      query.query_sparse_indices_.assign(
          (char *)sparse_vector.value().first.data(),
          sparse_vector.value().first.size() * sizeof(uint32_t));
      query.query_sparse_values_.assign(
          (char *)sparse_vector.value().second.data(),
          sparse_vector.value().second.size() * sizeof(float));
    }
    query.include_vector_ = true;

    auto result = collection->Query(query);
    ASSERT_FALSE(result.has_value());
    std::cout << result.error().message() << std::endl;
  }
}

TEST_F(CollectionTest, Feature_Query_General) {
  auto func = [&](std::string field_name) {
    FileHelper::RemoveDirectory(col_path);

    int doc_count = 1000;
    // create with normal schema
    auto schema = TestHelper::CreateNormalSchema();
    auto options = CollectionOptions{false, true, 100 * 1024 * 1024};
    auto collection = TestHelper::CreateCollectionWithDoc(
        col_path, *schema, options, 0, doc_count);

    ASSERT_NE(collection, nullptr);

    auto stats = collection->Stats().value();
    std::cout << stats.to_string_formatted() << std::endl;

    // validate query result
    for (int i = 1; i < 2; i++) {
      auto query_doc = TestHelper::CreateDoc(i, *schema);
      // std::cout << query_doc.to_detail_string() << std::endl;

      VectorQuery query;
      query.topk_ = 10;
      query.field_name_ = field_name;

      auto field_scheama = schema->get_vector_field(field_name);
      ASSERT_NE(field_scheama, nullptr);
      ASSERT_TRUE(field_scheama->is_vector_field());

      if (field_scheama->is_dense_vector()) {
        auto vector = query_doc.get<std::vector<float>>(field_name);
        ASSERT_TRUE(vector.has_value());
        query.query_vector_.assign((char *)vector.value().data(),
                                   vector.value().size() * sizeof(float));
      } else {
        auto sparse_vector =
            query_doc.get<std::pair<std::vector<uint32_t>, std::vector<float>>>(
                field_name);
        query.query_sparse_indices_.assign(
            (char *)sparse_vector.value().first.data(),
            sparse_vector.value().first.size() * sizeof(uint32_t));
        query.query_sparse_values_.assign(
            (char *)sparse_vector.value().second.data(),
            sparse_vector.value().second.size() * sizeof(float));
      }
      query.include_vector_ = true;

      auto result = collection->Query(query);
      if (!result.has_value()) {
        std::cout << "err: " << result.error().message() << std::endl;
      }
      ASSERT_TRUE(result.has_value());
      ASSERT_EQ(result.value().size(), query.topk_);

      for (int j = 0; j < query.topk_; j++) {
        std::cout << "result[" << j
                  << "]:" << result.value()[j]->to_detail_string() << std::endl;
        auto expect_doc = TestHelper::CreateDoc(doc_count - 1 - j, *schema);
        if (*result.value()[j] != expect_doc) {
          std::cout << "       doc:" << result.value()[j]->to_detail_string()
                    << std::endl;
          std::cout << "expect_doc:" << expect_doc.to_detail_string()
                    << std::endl;
        }
        ASSERT_EQ(*result.value()[j], expect_doc);
      }
    }
  };

  func("dense_fp32");
  func("sparse_fp32");
}

TEST_F(CollectionTest, Feature_Query_Empty) {
  auto func = [&](int doc_count, int topk) {
    FileHelper::RemoveDirectory(col_path);
    // create with normal schema
    auto schema = TestHelper::CreateNormalSchema();
    auto options = CollectionOptions{false, true, 100 * 1024 * 1024};
    auto collection = TestHelper::CreateCollectionWithDoc(
        col_path, *schema, options, 0, doc_count);

    ASSERT_NE(collection, nullptr);

    auto stats = collection->Stats().value();
    std::cout << stats.to_string_formatted() << std::endl;

    // validate query result
    for (int i = 1; i < 2; i++) {
      auto query_doc = TestHelper::CreateDoc(i, *schema);
      // std::cout << query_doc.to_detail_string() << std::endl;

      VectorQuery query;
      query.topk_ = topk;
      query.include_vector_ = true;

      auto result = collection->Query(query);
      if (!result.has_value()) {
        std::cout << "err: " << result.error().message() << std::endl;
      }
      ASSERT_TRUE(result.has_value());
      ASSERT_EQ(result.value().size(), std::min(query.topk_, doc_count));

      auto fields_name = schema->all_field_names();
      for (int j = 0; j < std::min(query.topk_, doc_count); j++) {
        auto result_doc = result.value()[j];
        auto doc_fields_names = result_doc->field_names();
        ASSERT_TRUE(vectors_equal_when_sorted(fields_name, doc_fields_names));
      }
    }
  };

  func(1, 1);
  func(1, 2);
  func(1000, 1000);
  func(1000, 1001);
}

TEST_F(CollectionTest, Feature_Query_WithoutVector_CreateScalarIndex) {
  auto func = [&](int doc_count, int topk, std::string field,
                  IndexParams::Ptr index_params, std::string filter,
                  int expected_doc_count) {
    FileHelper::RemoveDirectory(col_path);
    // create with normal schema
    auto schema = TestHelper::CreateNormalSchema();
    auto options = CollectionOptions{false, true, 100 * 1024 * 1024};
    auto collection = TestHelper::CreateCollectionWithDoc(
        col_path, *schema, options, 0, doc_count);

    ASSERT_NE(collection, nullptr);

    auto stats = collection->Stats().value();
    std::cout << stats.to_string_formatted() << std::endl;

    // validate query result
    VectorQuery query;
    query.topk_ = topk;
    query.include_vector_ = true;
    query.filter_ = filter;

    auto result = collection->Query(query);
    if (!result.has_value()) {
      std::cout << "err: " << result.error().message() << std::endl;
    }
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result.value().size(), expected_doc_count);

    // create index
    auto s = collection->CreateIndex(field, index_params);
    ASSERT_TRUE(s.ok());

    auto result2 = collection->Query(query);
    if (!result2.has_value()) {
      std::cout << "err: " << result2.error().message() << std::endl;
    }

    ASSERT_TRUE(result2.has_value());
    ASSERT_EQ(result2.value().size(), expected_doc_count);

    for (int j = 0; j < expected_doc_count; j++) {
      auto result1_doc = result2.value()[j];
      auto result2_doc = result2.value()[j];
      ASSERT_EQ(*result1_doc, *result2_doc);
    }
  };

  func(5, 20, "bool", std::make_shared<InvertIndexParams>(false), "bool=true",
       1);
  func(5, 20, "bool", std::make_shared<InvertIndexParams>(true), "bool =true",
       1);
  func(100, 20, "bool", std::make_shared<InvertIndexParams>(true),
       "bool = true", 10);
  func(100, 20, "int32", std::make_shared<InvertIndexParams>(true), "int32 =1",
       1);
  func(100, 20, "int32", std::make_shared<InvertIndexParams>(true), "int32 <1",
       1);
  func(100, 20, "int32", std::make_shared<InvertIndexParams>(true),
       "int32 >= 1", 20);
  func(100, 20, "string", std::make_shared<InvertIndexParams>(true),
       "string = 'value_1'", 1);
  func(5, 20, "array_bool", std::make_shared<InvertIndexParams>(true),
       "array_bool contain_any (true)", 1);

  func(5, 20, "array_int32", std::make_shared<InvertIndexParams>(true),
       "array_int32 contain_any (1)", 1);
  func(5, 20, "array_int32", std::make_shared<InvertIndexParams>(true),
       "array_int32 contain_any (1,2)", 2);
  func(5, 20, "array_int32", std::make_shared<InvertIndexParams>(true),
       "array_int32 contain_any (0,1,2,3,4)", 5);
  func(5, 20, "array_int32", std::make_shared<InvertIndexParams>(true),
       "array_int32 contain_any (0,4)", 2);
  // func(5, 20, "array_int32", std::make_shared<InvertIndexParams>(true),
  //      "array_int32 contain_any ()", 0);

  func(10000, 20, "array_int32", std::make_shared<InvertIndexParams>(true),
       "array_int32 contain_any (0)", 1);
  func(10000, 20, "array_int32", std::make_shared<InvertIndexParams>(true),
       "array_int32 contain_any (9999)", 1);
  func(10000, 20, "array_int32", std::make_shared<InvertIndexParams>(true),
       "array_int32 contain_any (10000)", 0);
  func(10000, 20, "array_int32", std::make_shared<InvertIndexParams>(true),
       "array_int32 contain_any (-1)", 0);
}

TEST_F(CollectionTest, Feature_Query_WithoutVector_WithScalarIndex) {
  auto func = [&](int doc_count, int topk, std::string field,
                  IndexParams::Ptr index_params, std::string filter,
                  int expected_doc_count) {
    FileHelper::RemoveDirectory(col_path);
    // create with normal schema
    auto schema = TestHelper::CreateNormalSchema(false, "demo", index_params);
    auto options = CollectionOptions{false, true, 100 * 1024 * 1024};
    auto collection = TestHelper::CreateCollectionWithDoc(
        col_path, *schema, options, 0, doc_count);

    ASSERT_NE(collection, nullptr);

    auto stats = collection->Stats().value();
    std::cout << stats.to_string_formatted() << std::endl;

    // validate query result
    VectorQuery query;
    query.topk_ = topk;
    query.include_vector_ = true;
    query.filter_ = filter;

    auto result = collection->Query(query);
    if (!result.has_value()) {
      std::cout << "err: " << result.error().message() << std::endl;
    }
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result.value().size(), expected_doc_count);
  };

  func(5, 20, "bool", std::make_shared<InvertIndexParams>(false), "bool=true",
       1);
  func(5, 20, "bool", std::make_shared<InvertIndexParams>(true), "bool =true",
       1);
  func(100, 20, "bool", std::make_shared<InvertIndexParams>(true),
       "bool = true", 10);
  func(100, 20, "int32", std::make_shared<InvertIndexParams>(true), "int32 =1",
       1);
  func(100, 20, "int32", std::make_shared<InvertIndexParams>(true), "int32 <1",
       1);
  func(100, 20, "int32", std::make_shared<InvertIndexParams>(true),
       "int32 >= 1", 20);
  func(5, 20, "array_bool", std::make_shared<InvertIndexParams>(true),
       "array_bool contain_any (true)", 1);
  func(5, 20, "array_int32", std::make_shared<InvertIndexParams>(true),
       "array_int32 contain_any (1)", 1);
}

TEST_F(CollectionTest, Feature_GroupByQuery) {}

TEST_F(CollectionTest, Feature_AddColumn_General) {
  // create collection
  int doc_count = 1000;
  auto schema = TestHelper::CreateNormalSchema();
  auto options = CollectionOptions{false, true, 64 * 1024 * 1024};
  auto collection = TestHelper::CreateCollectionWithDoc(
      col_path, *schema, options, 0, doc_count, false);

  ASSERT_TRUE(collection->Flush().ok());
  auto stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, doc_count);
  auto field_schema =
      std::make_shared<FieldSchema>("add_int32", DataType::INT32, false);
  auto s = collection->AddColumn("add_int32", field_schema, "int32",
                                 AddColumnOptions());
  if (!s.ok()) {
    std::cout << "status: " << s.message() << std::endl;
    ASSERT_TRUE(false);
  }
  auto new_schema = collection->Schema().value();
  ASSERT_TRUE(new_schema.has_field("add_int32"));

  stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, doc_count);

  auto check_doc = [&](int doc_count) {
    for (int i = 0; i < doc_count; i++) {
      auto expect_doc = TestHelper::CreateDoc(i, new_schema);
      auto result = collection->Fetch({expect_doc.pk()});
      ASSERT_TRUE(result.has_value());
      ASSERT_EQ(result.value().size(), 1);
      ASSERT_EQ(result.value().count(expect_doc.pk()), 1);
      auto doc = result.value()[expect_doc.pk()];
      ASSERT_NE(doc, nullptr);
      if (*doc != expect_doc) {
        std::cout << "       doc:" << doc->to_detail_string() << std::endl;
        std::cout << "expect_doc:" << expect_doc.to_detail_string()
                  << std::endl;
      }
      ASSERT_EQ(*doc, expect_doc);
    }
  };

  check_doc(doc_count);

  // validate query result
  for (int i = 1; i < 2; i++) {
    VectorQuery query;
    query.topk_ = 10;
    query.include_vector_ = true;

    auto result = collection->Query(query);
    if (!result.has_value()) {
      std::cout << "err: " << result.error().message() << std::endl;
    }
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result.value().size(), std::min(query.topk_, doc_count));

    auto fields_name = new_schema.all_field_names();
    for (int j = 0; j < std::min(query.topk_, doc_count); j++) {
      auto result_doc = result.value()[j];
      auto doc_fields_names = result_doc->field_names();
      ASSERT_TRUE(vectors_equal_when_sorted(fields_name, doc_fields_names));
    }
  }
}

TEST_F(CollectionTest, Feature_AddColumn_CornerCase) {
  int doc_count = 1000;
  auto options = CollectionOptions{false, true, 64 * 1024 * 1024};
  {
    // create collection
    auto schema = TestHelper::CreateNormalSchema();
    auto collection = TestHelper::CreateCollectionWithDoc(
        col_path, *schema, options, 0, doc_count, false);

    ASSERT_TRUE(collection->Flush().ok());

    auto stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, doc_count);
  }

  {
    // open collection and add invalid column
    auto result = Collection::Open(col_path, options);
    ASSERT_TRUE(result.has_value());
    auto collection = result.value();

    auto s =
        collection->AddColumn("int32", nullptr, "int32", AddColumnOptions());
    ASSERT_FALSE(s.ok());

    s = collection->AddColumn("add_int32", nullptr, "", AddColumnOptions());
    ASSERT_FALSE(s.ok());

    auto field_schema =
        std::make_shared<FieldSchema>("add_int32", DataType::INT32, false);
    s = collection->AddColumn("", field_schema, "int32", AddColumnOptions());
    ASSERT_FALSE(s.ok());


    s = collection->AddColumn("add_int32_invalid", field_schema, "int32",
                              AddColumnOptions());
    ASSERT_FALSE(s.ok());

    s = collection->AddColumn("add_int32", field_schema, "non_exist_field",
                              AddColumnOptions());
    ASSERT_FALSE(s.ok());
  }

  {
    // open collection and add one column
    auto result = Collection::Open(col_path, options);
    ASSERT_TRUE(result.has_value());
    auto collection = result.value();

    auto field_schema =
        std::make_shared<FieldSchema>("add_int32", DataType::INT32, false);
    auto s = collection->AddColumn("add_int32", field_schema, "int32",
                                   AddColumnOptions());
    if (!s.ok()) {
      std::cout << "status: " << s.message() << std::endl;
      ASSERT_TRUE(false);
    }
    auto new_schema = collection->Schema().value();
    ASSERT_TRUE(new_schema.has_field("add_int32"));
  }

  {
    // open collection and insert more doc
    auto result = Collection::Open(col_path, options);
    ASSERT_TRUE(result.has_value());
    auto collection = result.value();
    auto new_schema = collection->Schema().value();
    ASSERT_TRUE(new_schema.has_field("add_int32"));

    for (int i = doc_count; i < doc_count * 2; i++) {
      auto doc = TestHelper::CreateDoc(i, new_schema);
      std::vector<Doc> docs = {doc};
      auto res = collection->Insert(docs);
      ASSERT_TRUE(res.has_value());
      ASSERT_TRUE(res.value()[0].ok());
    }
    auto stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, doc_count * 2);

    auto check_doc = [&](int doc_count) {
      for (int i = 0; i < doc_count; i++) {
        auto expect_doc = TestHelper::CreateDoc(i, new_schema);
        auto result = collection->Fetch({expect_doc.pk()});
        ASSERT_TRUE(result.has_value());
        ASSERT_EQ(result.value().size(), 1);
        ASSERT_EQ(result.value().count(expect_doc.pk()), 1);
        auto doc = result.value()[expect_doc.pk()];
        ASSERT_NE(doc, nullptr);
        if (*doc != expect_doc) {
          std::cout << "       doc:" << doc->to_detail_string() << std::endl;
          std::cout << "expect_doc:" << expect_doc.to_detail_string()
                    << std::endl;
        }
        ASSERT_EQ(*doc, expect_doc);
      }
    };

    check_doc(doc_count * 2);
  }

  {
    // open collection and add one more column
    auto result = Collection::Open(col_path, options);
    ASSERT_TRUE(result.has_value());
    auto collection = result.value();

    auto field_schema =
        std::make_shared<FieldSchema>("add_int32_dup", DataType::INT32, false);
    auto s = collection->AddColumn("add_int32_dup", field_schema, "add_int32",
                                   AddColumnOptions());
    if (!s.ok()) {
      std::cout << "status: " << s.message() << std::endl;
      ASSERT_TRUE(false);
    }
    auto new_schema = collection->Schema().value();
    ASSERT_TRUE(new_schema.has_field("add_int32_dup"));
  }
}

TEST_F(CollectionTest, Feature_DropColumn_General) {
  // create collection
  int doc_count = 1000;
  auto schema = TestHelper::CreateNormalSchema();
  auto options = CollectionOptions{false, true, 64 * 1024 * 1024};
  auto collection = TestHelper::CreateCollectionWithDoc(
      col_path, *schema, options, 0, doc_count, false);

  ASSERT_TRUE(collection->Flush().ok());
  auto stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, doc_count);

  auto s = collection->DropColumn("int32");
  if (!s.ok()) {
    std::cout << "status: " << s.message() << std::endl;
    ASSERT_TRUE(false);
  }
  auto new_schema = collection->Schema().value();
  ASSERT_TRUE(!new_schema.has_field("int32"));
}

TEST_F(CollectionTest, Feature_AlterColumn_General) {
  // create collection
  int doc_count = 1000;
  auto schema = TestHelper::CreateNormalSchema();
  auto options = CollectionOptions{false, true, 64 * 1024 * 1024};
  auto collection = TestHelper::CreateCollectionWithDoc(
      col_path, *schema, options, 0, doc_count, false);

  ASSERT_TRUE(collection->Flush().ok());
  auto stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, doc_count);

  auto field_schema =
      std::make_shared<FieldSchema>("int32", DataType::INT64, false);
  auto s = collection->AlterColumn("int32", "int32", field_schema,
                                   AlterColumnOptions());
  ASSERT_FALSE(s.ok());

  s = collection->AlterColumn("int32", "", field_schema, AlterColumnOptions());
  ASSERT_TRUE(s.ok());

  auto new_schema = collection->Schema().value();
  ASSERT_TRUE(new_schema.has_field("int32"));
  ASSERT_TRUE(new_schema.get_field("int32")->data_type() == DataType::INT64);

  s = collection->AlterColumn("int32", "rename_in32", nullptr,
                              AlterColumnOptions());
  ASSERT_TRUE(s.ok());
  new_schema = collection->Schema().value();
  ASSERT_FALSE(new_schema.has_field("int32"));
  ASSERT_TRUE(new_schema.has_field("rename_in32"));
  ASSERT_TRUE(new_schema.get_field("rename_in32")->data_type() ==
              DataType::INT64);

  // validate query result
  for (int i = 1; i < 2; i++) {
    VectorQuery query;
    query.topk_ = 10;
    query.include_vector_ = true;

    auto result = collection->Query(query);
    if (!result.has_value()) {
      std::cout << "err: " << result.error().message() << std::endl;
    }
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result.value().size(), std::min(query.topk_, doc_count));

    auto fields_name = new_schema.all_field_names();
    for (int j = 0; j < std::min(query.topk_, doc_count); j++) {
      auto result_doc = result.value()[j];
      auto doc_fields_names = result_doc->field_names();
      ASSERT_TRUE(vectors_equal_when_sorted(fields_name, doc_fields_names));
    }
  }
}

TEST_F(CollectionTest, Feature_AlterColumn_CornerCase) {
  int doc_count = 1000;
  auto options = CollectionOptions{false, true, 64 * 1024 * 1024};

  {
    // create collection
    auto schema = TestHelper::CreateNormalSchema();
    auto collection = TestHelper::CreateCollectionWithDoc(
        col_path, *schema, options, 0, doc_count, false);

    ASSERT_TRUE(collection->Flush().ok());
    auto stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, doc_count);
  }

  {
    // open collection and alter column
    auto result = Collection::Open(col_path, options);
    ASSERT_TRUE(result.has_value());
    auto collection = result.value();

    auto field_schema =
        std::make_shared<FieldSchema>("int32_to_int64", DataType::INT64, false);
    auto s = collection->AlterColumn("int32", "", field_schema,
                                     AlterColumnOptions());
    ASSERT_TRUE(s.ok());

    auto new_schema = collection->Schema().value();
    ASSERT_FALSE(new_schema.has_field("int32"));
    ASSERT_TRUE(new_schema.has_field("int32_to_int64"));
    ASSERT_TRUE(new_schema.get_field("int32_to_int64")->data_type() ==
                DataType::INT64);
  }

  {
    // open collection and insert more doc
    auto result = Collection::Open(col_path, options);
    ASSERT_TRUE(result.has_value());
    auto collection = result.value();

    auto new_schema = collection->Schema().value();

    for (int i = doc_count; i < doc_count * 2; i++) {
      auto doc = TestHelper::CreateDoc(i, new_schema);
      std::vector<Doc> docs = {doc};
      auto res = collection->Insert(docs);
      ASSERT_TRUE(res.has_value());
      ASSERT_TRUE(res.value()[0].ok());
    }
    auto stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, doc_count * 2);

    auto check_doc = [&](int doc_count) {
      for (int i = 0; i < doc_count; i++) {
        auto expect_doc = TestHelper::CreateDoc(i, new_schema);
        auto result = collection->Fetch({expect_doc.pk()});
        ASSERT_TRUE(result.has_value());
        ASSERT_EQ(result.value().size(), 1);
        ASSERT_EQ(result.value().count(expect_doc.pk()), 1);
        auto doc = result.value()[expect_doc.pk()];
        ASSERT_NE(doc, nullptr);
        if (*doc != expect_doc) {
          std::cout << "       doc:" << doc->to_detail_string() << std::endl;
          std::cout << "expect_doc:" << expect_doc.to_detail_string()
                    << std::endl;
        }
        ASSERT_EQ(*doc, expect_doc);
      }
    };

    check_doc(doc_count * 2);

    // validate query result
    for (int i = 1; i < 2; i++) {
      VectorQuery query;
      query.topk_ = 10;
      query.include_vector_ = true;

      auto result = collection->Query(query);
      if (!result.has_value()) {
        std::cout << "err: " << result.error().message() << std::endl;
      }
      ASSERT_TRUE(result.has_value());
      ASSERT_EQ(result.value().size(), std::min(query.topk_, doc_count));

      auto fields_name = new_schema.all_field_names();
      for (int j = 0; j < std::min(query.topk_, doc_count); j++) {
        auto result_doc = result.value()[j];
        auto doc_fields_names = result_doc->field_names();
        ASSERT_TRUE(vectors_equal_when_sorted(fields_name, doc_fields_names));
      }
    }
  }
}

TEST_F(CollectionTest, Feature_Column_MixOperation) {
  int max_doc_per_count = 1000;
  // create empty collection
  auto schema = TestHelper::CreateNormalSchema(
      false, "demo", nullptr, std::make_shared<HnswIndexParams>(MetricType::IP),
      max_doc_per_count);
  auto options = CollectionOptions{false, true, 64 * 1024 * 1024};

  // create seg1
  auto collection = TestHelper::CreateCollectionWithDoc(
      col_path, *schema, options, 0, max_doc_per_count, false);

  // create seg2
  auto s = TestHelper::CollectionInsertDoc(collection, max_doc_per_count,
                                           max_doc_per_count * 3 / 2);

  // add column
  auto field_schema =
      std::make_shared<FieldSchema>("add_int32", DataType::INT32, false);
  s = collection->AddColumn("add_int32", field_schema, "int32",
                            AddColumnOptions());
  if (!s.ok()) {
    std::cout << "status: " << s.message() << std::endl;
    ASSERT_TRUE(false);
  }
  auto new_schema = collection->Schema().value();
  ASSERT_TRUE(new_schema.has_field("add_int32"));

  auto stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, max_doc_per_count * 3 / 2);

  // drop column
  s = collection->DropColumn("uint32");
  if (!s.ok()) {
    std::cout << "status: " << s.message() << std::endl;
    ASSERT_TRUE(false);
  }
  new_schema = collection->Schema().value();
  ASSERT_TRUE(!new_schema.has_field("uint32"));

  stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, max_doc_per_count * 3 / 2);

  // alter column
  s = collection->AlterColumn("int32", "rename_int32", nullptr,
                              AlterColumnOptions());
  if (!s.ok()) {
    std::cout << "status: " << s.message() << std::endl;
    ASSERT_TRUE(false);
  }
  new_schema = collection->Schema().value();
  ASSERT_TRUE(new_schema.has_field("rename_int32"));

  stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, max_doc_per_count * 3 / 2);

  // create seg3
  s = TestHelper::CollectionInsertDoc(collection, max_doc_per_count * 3 / 2,
                                      max_doc_per_count * 5 / 2);

  stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, max_doc_per_count * 5 / 2);

  // drop column
  s = collection->DropColumn("rename_int32");
  if (!s.ok()) {
    std::cout << "status: " << s.message() << std::endl;
    ASSERT_TRUE(false);
  }
  new_schema = collection->Schema().value();
  ASSERT_TRUE(!new_schema.has_field("rename_int32"));


  auto check_doc = [&](int doc_count) {
    for (int i = 0; i < doc_count; i++) {
      auto expect_doc = TestHelper::CreateDoc(i, new_schema);
      auto result = collection->Fetch({expect_doc.pk()});
      ASSERT_TRUE(result.has_value());
      ASSERT_EQ(result.value().size(), 1);
      ASSERT_EQ(result.value().count(expect_doc.pk()), 1);
      auto doc = result.value()[expect_doc.pk()];
      ASSERT_NE(doc, nullptr);
      if (*doc != expect_doc) {
        std::cout << "       doc:" << doc->to_detail_string() << std::endl;
        std::cout << "expect_doc:" << expect_doc.to_detail_string()
                  << std::endl;
      }
      ASSERT_EQ(*doc, expect_doc);
    }
  };

  check_doc(max_doc_per_count * 5 / 2);
}

TEST_F(CollectionTest, Feature_Column_MixOperation_Empty) {
  int doc_count = 0;
  auto options = CollectionOptions{false, true, 64 * 1024 * 1024};
  {
    // create empty collection
    auto schema = TestHelper::CreateNormalSchema();
    auto collection = TestHelper::CreateCollectionWithDoc(
        col_path, *schema, options, 0, doc_count, false);

    ASSERT_TRUE(collection->Flush().ok());

    auto stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, doc_count);
  }

  {
    // open collection and do mix operation
    auto result = Collection::Open(col_path, options);
    ASSERT_TRUE(result.has_value());
    auto collection = result.value();

    // add column
    auto field_schema =
        std::make_shared<FieldSchema>("add_int32", DataType::INT32, false);
    auto s = collection->AddColumn("add_int32", field_schema, "int32",
                                   AddColumnOptions());
    ASSERT_TRUE(s.ok());

    auto new_schema = collection->Schema().value();
    ASSERT_TRUE(new_schema.has_field("add_int32"));

    auto stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, 0);
  }

  {
    // open collection and do mix operation
    auto result = Collection::Open(col_path, options);
    ASSERT_TRUE(result.has_value());
    auto collection = result.value();

    auto new_schema = collection->Schema().value();
    ASSERT_TRUE(new_schema.has_field("add_int32"));

    // alter column
    auto s = collection->AlterColumn("add_int32", "rename_int32", nullptr,
                                     AlterColumnOptions());
    ASSERT_TRUE(s.ok());

    new_schema = collection->Schema().value();
    ASSERT_FALSE(new_schema.has_field("add_int32"));
    ASSERT_TRUE(new_schema.has_field("rename_int32"));

    auto stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, 0);
  }

  {
    // open collection and do mix operation
    auto result = Collection::Open(col_path, options);
    ASSERT_TRUE(result.has_value());
    auto collection = result.value();

    auto new_schema = collection->Schema().value();
    ASSERT_TRUE(new_schema.has_field("rename_int32"));

    // drop column
    auto s = collection->DropColumn("rename_int32");
    ASSERT_TRUE(s.ok());
    new_schema = collection->Schema().value();
    ASSERT_FALSE(new_schema.has_field("rename_int32"));

    auto stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, 0);
  }
}

// **** CORNER CASES **** //
TEST_F(CollectionTest, CornerCase_CreateAndOpen) {
  // Collection::CreateAndOpen
  {
    {
      std::cout << "Collection::CreateAndOpen case 1" << std::endl;
      // create collection with non-exist path with read-only mode
      auto schema = TestHelper::CreateNormalSchema();
      auto result = Collection::CreateAndOpen("non-exist-path", *schema,
                                              CollectionOptions{true, false});
      ASSERT_FALSE(result.has_value());
    }

    {
      std::cout << "Collection::CreateAndOpen case 2" << std::endl;
      // create collection with exist path
      auto schema = TestHelper::CreateNormalSchema();
      FileHelper::CreateDirectory("invalid_path");
      auto result = Collection::CreateAndOpen("invalid_path", *schema,
                                              CollectionOptions{true, true});
      ASSERT_FALSE(result.has_value());
      FileHelper::RemoveDirectory("invalid_path");
    }

    {
      std::cout << "Collection::CreateAndOpen case 3" << std::endl;
      FileHelper::RemoveDirectory("invalid_path");
      // create collection with exist path
      auto schema = TestHelper::CreateNormalSchema();

      auto result = Collection::CreateAndOpen("invalid_path", *schema,
                                              CollectionOptions{false, true});
      if (!result.has_value()) {
        std::cout << result.error().message() << std::endl;
      }
      ASSERT_TRUE(result.has_value());

      std::cout << "Collection::Open again" << std::endl;
      auto new_result = Collection::Open("invalid_path", CollectionOptions{});
      ASSERT_FALSE(new_result.has_value());

      result.value().reset();
      // FileHelper::RemoveDirectory("invalid_path");
    }

    {
      std::cout << "Collection::CreateAndOpen case 4" << std::endl;
      FileHelper::RemoveDirectory(col_path);
      // abnormal schema
      auto schema = TestHelper::CreateNormalSchema(
          false, "demo", std::make_shared<FlatIndexParams>(MetricType::IP));
      auto result = Collection::CreateAndOpen(col_path, *schema,
                                              CollectionOptions{false, true});
      ASSERT_FALSE(result.has_value());
      ASSERT_EQ(result.error().code(), StatusCode::INVALID_ARGUMENT);
      std::cout << result.error().message() << std::endl;
    }

    {
      std::cout << "Collection::CreateAndOpen case 5" << std::endl;
      FileHelper::RemoveDirectory(col_path);
      // abnormal schema
      auto schema = TestHelper::CreateScalarSchema();
      auto result = Collection::CreateAndOpen(col_path, *schema,
                                              CollectionOptions{false, true});
      ASSERT_FALSE(result.has_value());
      ASSERT_EQ(result.error().code(), StatusCode::INVALID_ARGUMENT);
      std::cout << result.error().message() << std::endl;
    }
  }

  {
    std::cout << "Collection::CreateAndOpen case 6" << std::endl;
    FileHelper::RemoveDirectory(col_path);
    auto schema = TestHelper::CreateNormalSchema();

    // start N threas to create_and_open collection
    std::vector<std::thread> threads;
    std::mutex mtx;
    std::vector<Status> statuses;
    for (int i = 0; i < 10; i++) {
      threads.emplace_back([&]() {
        auto result = Collection::CreateAndOpen(col_path, *schema,
                                                CollectionOptions{false, true});
        if (!result.has_value()) {
          std::cout << result.error().message() << std::endl;
          std::lock_guard<std::mutex> lck(mtx);
          statuses.emplace_back(result.error());
        }
      });
    }

    for (auto &t : threads) {
      t.join();
    }

    ASSERT_EQ(statuses.size(), 9);
  }

  // Collection::Open
  {
    {
      std::cout << "Collection::Open case 1" << std::endl;
      // open collection with non-exist path
      auto result = Collection::Open("non-exist-path", CollectionOptions{});
      ASSERT_FALSE(result.has_value());
    }

    {
      std::cout << "Collection::Open case 2" << std::endl;
      // open collection with invalid path which contains no manifest
      FileHelper::RemoveDirectory("invalid_path");
      FileHelper::CreateDirectory("invalid_path");
      auto result = Collection::Open("invalid_path", CollectionOptions{});
      ASSERT_FALSE(result.has_value());
      FileHelper::RemoveDirectory("invalid_path");
    }
  }
}

TEST_F(CollectionTest, CornerCase_CreateIndex) {
  auto schema = TestHelper::CreateNormalSchema();
  auto options = CollectionOptions{false, true, 64 * 1024 * 1024};
  auto collection = TestHelper::CreateCollectionWithDoc(col_path, *schema,
                                                        options, 0, 0, false);

  // create index on non-exist field
  auto s = collection->CreateIndex(
      "non-exist", std::make_shared<FlatIndexParams>(MetricType::IP));
  ASSERT_FALSE(s.ok());
  ASSERT_EQ(s.code(), StatusCode::NOT_FOUND);

  s = collection->DropIndex("non-exist");
  ASSERT_EQ(s.code(), StatusCode::NOT_FOUND);

  // create vector index on scalar field
  s = collection->CreateIndex(
      "uint32", std::make_shared<FlatIndexParams>(MetricType::IP));
  ASSERT_FALSE(s.ok());
  ASSERT_EQ(s.code(), StatusCode::INVALID_ARGUMENT);

  // create scalar index on vector field
  s = collection->CreateIndex("dense_fp32",
                              std::make_shared<InvertIndexParams>(true));
  ASSERT_FALSE(s.ok());
  ASSERT_EQ(s.code(), StatusCode::INVALID_ARGUMENT);

  // create scalar index on sparse vector field
  s = collection->CreateIndex("sparse_fp32",
                              std::make_shared<InvertIndexParams>(true));
  ASSERT_FALSE(s.ok());
  ASSERT_EQ(s.code(), StatusCode::INVALID_ARGUMENT);

  // create Ivf index on vector field
  s = collection->CreateIndex("sparse_fp32",
                              std::make_shared<IVFIndexParams>(MetricType::IP));
  ASSERT_FALSE(s.ok());
  ASSERT_EQ(s.code(), StatusCode::INVALID_ARGUMENT);
}