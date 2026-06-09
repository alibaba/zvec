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
#include <gtest/gtest.h>
#include <magic_enum/magic_enum.hpp>
#include <zvec/ailego/io/file.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/ailego/utility/file_helper.h>
#include "db/common/file_helper.h"
#include "db/index/common/type_helper.h"
#include "index/utils/utils.h"
#include "zvec/ailego/utility/float_helper.h"
#include "zvec/db/doc.h"
#include "zvec/db/index_params.h"
#include "zvec/db/options.h"
#include "zvec/db/reranker.h"
#include "zvec/db/schema.h"
#include "zvec/db/status.h"
#include "zvec/db/type.h"

using namespace zvec;
using namespace zvec::test;

std::string col_path = "test_collection";

class CollectionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    zvec::ailego::MemoryLimitPool::get_instance().init(2 * 1024ll * 1024ll *
                                                       1024ll);
    FileHelper::RemoveDirectory(col_path);
  }

  void TearDown() override {
    FileHelper::RemoveDirectory(col_path);
    ailego::FileHelper::RemoveDirectory("demo");
  }
};

TEST_F(CollectionTest, Feature_CreateAndOpen_General) {
  auto func = [&](bool enable_mmap) {
    CollectionOptions options;
    options.read_only_ = false;
    options.enable_mmap_ = enable_mmap;

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
    ASSERT_FALSE(col->Query(SearchQuery{}).has_value());
    ASSERT_FALSE(col->Query(MultiQuery{}).has_value());
    ASSERT_FALSE(col->GroupByQuery({}).has_value());
    ASSERT_FALSE(col->CreateIndex("", nullptr).ok());
    ASSERT_FALSE(col->DropIndex("").ok());
    ASSERT_FALSE(col->AddColumn(nullptr, "").ok());
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
    ASSERT_FALSE(col->AddColumn(nullptr, "").ok());
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
  };
  func(true);
  func(false);
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
    std::vector<std::string> valid_paths = {"你好",
                                            "data123",
                                            "my_collection",
                                            "v1.2_alpha-beta",
                                            ".hidden",
                                            "file.txt",
                                            "abs_test/nested/path",
                                            "abs test/nested/path",
                                            "nested/a/b/c",
                                            "_",
                                            "-",
                                            "./tmp"};
    for (auto path : valid_paths) {
      ailego::FileHelper::RemoveDirectory(path.c_str());

      auto result = Collection::CreateAndOpen(path, *schema, options);
      if (!result.has_value()) {
        std::cout << result.error().message() << std::endl;
        std::cout << "File error:" << ailego::FileHelper::GetLastErrorString()
                  << std::endl;
      }
      ASSERT_TRUE(result.has_value());

      result.value()->Destroy();
    }
  }

  {
    using std::string_literals::operator""s;
    std::vector<std::string> invalid_paths = {
        "",
        "v\0v"s,  // NUL
#if _WIN32
        "v?v"s,
#endif
    };
    for (auto path : invalid_paths) {
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
  auto func = [&](bool enable_mmap, bool schema_nullable, bool doc_nullable,
                  int doc_count = 1000) {
    FileHelper::RemoveDirectory(col_path);

    // create with normal schema
    auto schema = TestHelper::CreateNormalSchema(schema_nullable);
    auto options = CollectionOptions{false, enable_mmap, 100 * 1024 * 1024};
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

  for (bool enable_mmap : {true, false}) {
    func(enable_mmap, false, false);
    func(enable_mmap, true, true);
    func(enable_mmap, true, false);
    func(enable_mmap, false, true);

    func(enable_mmap, false, false, 0);
    func(enable_mmap, false, false, 1);
    func(enable_mmap, false, false, 2);
  }
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
  auto func = [&](bool enable_mmap, bool schema_nullable, bool doc_nullable,
                  int doc_count = 1000) {
    FileHelper::RemoveDirectory(col_path);

    // create with normal schema
    auto schema = TestHelper::CreateNormalSchema(schema_nullable);
    auto options = CollectionOptions{false, enable_mmap, 100 * 1024 * 1024};
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

  for (bool enable_mmap : {true, false}) {
    func(enable_mmap, false, false);
    func(enable_mmap, true, true);
    func(enable_mmap, true, false);
    func(enable_mmap, false, true);

    func(enable_mmap, false, false, 0);
    func(enable_mmap, false, false, 1);
    func(enable_mmap, false, false, 2);
  }
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
  auto func = [&](bool enable_mmap, int doc_count) {
    auto schema = TestHelper::CreateNormalSchema();
    auto options = CollectionOptions{false, enable_mmap, 100 * 1024 * 1024};
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

  for (bool enable_mmap : {true, false}) {
    func(enable_mmap, 99);
    func(enable_mmap, 100);
    func(enable_mmap, 101);
    func(enable_mmap, 1000);
  }
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
  auto func = [&](bool enable_mmap, int doc_count) {
    auto schema = TestHelper::CreateNormalSchema();
    auto options = CollectionOptions{false, enable_mmap, 100 * 1024 * 1024};
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

  for (bool enable_mmap : {true, false}) {
    func(enable_mmap, 99);
    func(enable_mmap, 100);
    func(enable_mmap, 101);
    func(enable_mmap, 1000);
  }
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
  auto func = [&](bool enable_mmap, int doc_count) {
    auto schema = TestHelper::CreateNormalSchema();
    auto options = CollectionOptions{false, enable_mmap, 100 * 1024 * 1024};
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

  for (bool enable_mmap : {true, false}) {
    func(enable_mmap, 99);
    func(enable_mmap, 100);
    func(enable_mmap, 101);
    func(enable_mmap, 1000);
  }
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
  auto func = [&](bool enable_mmap) {
    // case1: insert -> upsert -> update -> delete
    auto schema = TestHelper::CreateNormalSchema();
    auto options = CollectionOptions{false, enable_mmap, 100 * 1024 * 1024};
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
  };
  func(true);
  func(false);
}

TEST_F(CollectionTest, Feature_CreateIndex_General) {
  auto func = [&](bool enable_mmap) {
    FileHelper::RemoveDirectory(col_path);
    // create empty collection
    auto schema = TestHelper::CreateNormalSchema();
    auto options = CollectionOptions{false, enable_mmap, 64 * 1024 * 1024};
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
    auto new_index_params =
        std::make_shared<HnswIndexParams>(MetricType::COSINE);
    s = collection->CreateIndex("dense_fp32", index_params);
    if (!s.ok()) {
      std::cout << "status: " << s.message() << std::endl;
      ASSERT_TRUE(false);
    }

    s = collection->CreateIndex("dense_fp32_invalid", index_params);
    ASSERT_FALSE(s.ok());
  };
  func(true);
  func(false);
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
    ASSERT_NE(collection, nullptr);

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

    SearchQuery query;
    query.topk_ = doc_count;
    query.target_.field_name_ = field_name;
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
        query.target_.set_vector(
            std::string((char *)vector_fp16.data(),
                        vector_fp16.size() * sizeof(ailego::Float16)));
      } else if (field_scheama->data_type() == DataType::VECTOR_FP32) {
        vector = std::vector<float>(field_scheama->dimension(), 1);
        vector[0] = 0;
        query.target_.set_vector(
            std::string((char *)vector.data(), vector.size() * sizeof(float)));
      } else {
        vector_int8 = std::vector<int8_t>(field_scheama->dimension(), 1);
        vector_int8[0] = 0;
        query.target_.set_vector(std::string(
            (char *)vector_int8.data(), vector_int8.size() * sizeof(int8_t)));
      }
    } else {
      if (field_scheama->data_type() == DataType::SPARSE_VECTOR_FP32) {
        sparse_vector = {{1}, {1}};
        query.target_.set_sparse_vector(
            std::string((char *)sparse_vector.first.data(),
                        sparse_vector.first.size() * sizeof(uint32_t)),
            std::string((char *)sparse_vector.second.data(),
                        sparse_vector.second.size() * sizeof(float)));
      } else {
        sparse_vector_fp16 = {{1}, {ailego::Float16(1.0f)}};
        query.target_.set_sparse_vector(
            std::string((char *)sparse_vector_fp16.first.data(),
                        sparse_vector_fp16.first.size() * sizeof(uint32_t)),
            std::string(
                (char *)sparse_vector_fp16.second.data(),
                sparse_vector_fp16.second.size() * sizeof(ailego::Float16)));
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
#ifdef __ANDROID__
  GTEST_SKIP() << "Skipped on Android: emulator filesystem lacks hardlink "
                  "support (needed by RocksDB checkpoint)";
#endif
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
  auto func = [&](bool enable_mmap) {
    FileHelper::RemoveDirectory(col_path);
    // create empty collection
    auto schema = TestHelper::CreateSchemaWithVectorIndex();
    auto options = CollectionOptions{false, enable_mmap, 64 * 1024 * 1204};
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
  };
  func(true);
  func(false);
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
#ifdef __ANDROID__
  GTEST_SKIP() << "Skipped on Android: emulator filesystem lacks hardlink "
                  "support (needed by RocksDB checkpoint)";
#endif
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
  auto func = [](bool enable_mmap, int concurrency) {
    FileHelper::RemoveDirectory(col_path);

    int doc_count = 1000;

    // create empty collection
    auto schema = TestHelper::CreateSchemaWithVectorIndex();
    auto options = CollectionOptions{false, enable_mmap, 64 * 1024 * 1024};
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

  for (bool enable_mmap : {true, false}) {
    func(enable_mmap, 0);
    func(enable_mmap, 4);
  }
}

TEST_F(CollectionTest, Feature_Optimize_Repeated) {
  auto run_repeated_optimize_test = [&](bool enable_mmap,
                                        IndexParams::Ptr index_params) {
    ASSERT_NE(index_params, nullptr);
    SCOPED_TRACE(testing::Message()
                 << "index_params=" << index_params->to_string());

    FileHelper::RemoveDirectory(col_path);
    int doc_count = 1000;
    auto schema =
        TestHelper::CreateSchemaWithVectorIndex(false, "demo", index_params);
    auto options = CollectionOptions{false, enable_mmap, 64 * 1024 * 1024};
    auto collection = TestHelper::CreateCollectionWithDoc(
        col_path, *schema, options, 0, doc_count, false);

    const bool tracks_completeness = (index_params->type() != IndexType::FLAT);

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

    // Phase 1: docs are inserted but no index is built yet.
    check_doc();

    ASSERT_TRUE(collection->Flush().ok());
    auto stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, doc_count);
    if (tracks_completeness) {
      ASSERT_EQ(stats.index_completeness["dense_fp32"], 0);
    }

    // Phase 2: first full optimize builds the index from scratch.
    auto s = collection->Optimize();
    ASSERT_TRUE(s.ok());
    stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, doc_count);
    if (tracks_completeness) {
      ASSERT_EQ(stats.index_completeness["dense_fp32"], 1);
    }

    // Phase 3: optimize again with no new data; must be a no-op and remain
    // fully built.
    s = collection->Optimize();
    ASSERT_TRUE(s.ok());
    stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, doc_count);
    if (tracks_completeness) {
      ASSERT_EQ(stats.index_completeness["dense_fp32"], 1);
    }

    // Phase 4: repeated single-doc incremental optimize. Each iteration
    // appends one doc and re-optimizes; completeness must shrink to a
    // predictable ratio after insert and return to 1 after optimize.
    int single_loop_count = 10;
    uint64_t next_doc_id = doc_count;
    for (int i = 0; i < single_loop_count; i++) {
      s = TestHelper::CollectionInsertDoc(collection, next_doc_id,
                                          next_doc_id + 1);
      ASSERT_TRUE(s.ok());

      stats = collection->Stats().value();
      ASSERT_EQ(stats.doc_count, doc_count + i + 1);
      if (tracks_completeness) {
        ASSERT_FLOAT_EQ(stats.index_completeness["dense_fp32"],
                        1.0 * (doc_count + i) / (doc_count + i + 1));
      }

      s = collection->Optimize();
      if (!s.ok()) {
        std::cout << "optimize failed: " << s.message() << std::endl;
      }
      ASSERT_TRUE(s.ok());

      stats = collection->Stats().value();
      ASSERT_EQ(stats.doc_count, doc_count + i + 1);
      if (tracks_completeness) {
        ASSERT_EQ(stats.index_completeness["dense_fp32"], 1);
      }

      next_doc_id += 1;
    }
    doc_count += single_loop_count;

    // Phase 5: repeated batch incremental optimize. Each iteration appends
    // a batch of docs and re-optimizes.
    int batch_loop_count = 3;
    int batch_size = 100;
    for (int i = 0; i < batch_loop_count; i++) {
      s = TestHelper::CollectionInsertDoc(collection, next_doc_id,
                                          next_doc_id + batch_size);
      ASSERT_TRUE(s.ok());

      stats = collection->Stats().value();
      ASSERT_EQ(stats.doc_count, doc_count + batch_size);
      if (tracks_completeness) {
        ASSERT_FLOAT_EQ(stats.index_completeness["dense_fp32"],
                        1.0 * doc_count / (doc_count + batch_size));
      }

      s = collection->Optimize();
      if (!s.ok()) {
        std::cout << "optimize failed: " << s.message() << std::endl;
      }
      ASSERT_TRUE(s.ok());

      stats = collection->Stats().value();
      ASSERT_EQ(stats.doc_count, doc_count + batch_size);
      if (tracks_completeness) {
        ASSERT_EQ(stats.index_completeness["dense_fp32"], 1);
      }

      next_doc_id += batch_size;
      doc_count += batch_size;
    }

    // Phase 6: verify all documents survived the repeated optimizes.
    check_doc();

    // Phase 7: reopen the collection and verify the persisted state is
    // still fully built and fetchable.
    collection.reset();
    auto reopen_result = Collection::Open(col_path, options);
    ASSERT_TRUE(reopen_result.has_value());
    collection = std::move(reopen_result.value());

    stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, doc_count);
    if (tracks_completeness) {
      ASSERT_EQ(stats.index_completeness["dense_fp32"], 1);
    }

    check_doc();
  };


  for (bool enable_mmap : {true, false}) {
    run_repeated_optimize_test(enable_mmap,
                               std::make_shared<FlatIndexParams>(
                                   MetricType::IP, QuantizeType::UNDEFINED));
    run_repeated_optimize_test(
        enable_mmap,
        std::make_shared<FlatIndexParams>(MetricType::IP, QuantizeType::FP16));
    run_repeated_optimize_test(
        enable_mmap, std::make_shared<HnswIndexParams>(
                         MetricType::IP, 16, 200, QuantizeType::UNDEFINED));
    run_repeated_optimize_test(
        enable_mmap, std::make_shared<HnswIndexParams>(MetricType::IP, 16, 200,
                                                       QuantizeType::FP16));
    run_repeated_optimize_test(enable_mmap, std::make_shared<IVFIndexParams>(
                                                MetricType::IP, 10, 4, false,
                                                QuantizeType::UNDEFINED));
    run_repeated_optimize_test(
        enable_mmap, std::make_shared<IVFIndexParams>(
                         MetricType::IP, 10, 4, false, QuantizeType::FP16));
#if DISKANN_SUPPORTED
    run_repeated_optimize_test(
        enable_mmap, std::make_shared<DiskAnnIndexParams>(
                         MetricType::IP, 10, 4, 0, QuantizeType::UNDEFINED));
#endif
#if RABITQ_SUPPORTED
    run_repeated_optimize_test(
        enable_mmap, std::make_shared<HnswRabitqIndexParams>(MetricType::IP, 7,
                                                             256, 16, 200, 0));
#endif
  }
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

      SearchQuery query;
      query.topk_ = 10;
      query.include_vector_ = true;
      query.target_.field_name_ = "dense_fp32";

      auto vector = query_doc.get<std::vector<float>>("dense_fp32");
      ASSERT_TRUE(vector.has_value());
      query.target_.set_vector(
          std::string((char *)vector.value().data(),
                      vector.value().size() * sizeof(float)));


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

  // delete by filter
  s = collection->DeleteByFilter("int32 < 10");
  if (!s.ok()) {
    std::cout << s.message() << std::endl;
  }
  ASSERT_TRUE(s.ok());
  stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, doc_count - 10);

  // delete all docs
  std::vector<std::string> pks;
  for (int i = 10; i < doc_count; ++i) {
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
    SearchQuery query;
    query.topk_ = 1024;
    query.target_.field_name_ = field_name;

    auto field_scheama = schema->get_vector_field(field_name);
    ASSERT_NE(field_scheama, nullptr);
    ASSERT_TRUE(field_scheama->is_vector_field());

    if (field_scheama->is_dense_vector()) {
      auto vector = query_doc.get<std::vector<float>>(field_name);
      ASSERT_TRUE(vector.has_value());
      query.target_.set_vector(
          std::string((char *)vector.value().data(),
                      vector.value().size() * sizeof(float)));
    } else {
      auto sparse_vector =
          query_doc.get<std::pair<std::vector<uint32_t>, std::vector<float>>>(
              field_name);
      query.target_.set_sparse_vector(
          std::string((char *)sparse_vector.value().first.data(),
                      sparse_vector.value().first.size() * sizeof(uint32_t)),
          std::string((char *)sparse_vector.value().second.data(),
                      sparse_vector.value().second.size() * sizeof(float)));
    }
    query.include_vector_ = true;

    auto result = collection->Query(query);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result.value().size(), query.topk_);
  }

  {
    SearchQuery query;
    query.topk_ = 100001;
    query.target_.field_name_ = field_name;

    auto field_scheama = schema->get_vector_field(field_name);
    ASSERT_NE(field_scheama, nullptr);
    ASSERT_TRUE(field_scheama->is_vector_field());

    if (field_scheama->is_dense_vector()) {
      auto vector = query_doc.get<std::vector<float>>(field_name);
      ASSERT_TRUE(vector.has_value());
      query.target_.set_vector(
          std::string((char *)vector.value().data(),
                      vector.value().size() * sizeof(float)));
    } else {
      auto sparse_vector =
          query_doc.get<std::pair<std::vector<uint32_t>, std::vector<float>>>(
              field_name);
      query.target_.set_sparse_vector(
          std::string((char *)sparse_vector.value().first.data(),
                      sparse_vector.value().first.size() * sizeof(uint32_t)),
          std::string((char *)sparse_vector.value().second.data(),
                      sparse_vector.value().second.size() * sizeof(float)));
    }
    query.include_vector_ = true;

    auto result = collection->Query(query);
    ASSERT_FALSE(result.has_value());
    std::cout << result.error().message() << std::endl;
  }

  {
    SearchQuery query;
    query.topk_ = 1024;
    query.target_.field_name_ = field_name;
    query.output_fields_ = std::make_optional<std::vector<std::string>>(
        std::vector<std::string>(1025));

    auto field_scheama = schema->get_vector_field(field_name);
    ASSERT_NE(field_scheama, nullptr);
    ASSERT_TRUE(field_scheama->is_vector_field());

    if (field_scheama->is_dense_vector()) {
      auto vector = query_doc.get<std::vector<float>>(field_name);
      ASSERT_TRUE(vector.has_value());
      query.target_.set_vector(
          std::string((char *)vector.value().data(),
                      vector.value().size() * sizeof(float)));
    } else {
      auto sparse_vector =
          query_doc.get<std::pair<std::vector<uint32_t>, std::vector<float>>>(
              field_name);
      query.target_.set_sparse_vector(
          std::string((char *)sparse_vector.value().first.data(),
                      sparse_vector.value().first.size() * sizeof(uint32_t)),
          std::string((char *)sparse_vector.value().second.data(),
                      sparse_vector.value().second.size() * sizeof(float)));
    }
    query.include_vector_ = true;

    auto result = collection->Query(query);
    ASSERT_FALSE(result.has_value());
    std::cout << result.error().message() << std::endl;
  }
}

TEST_F(CollectionTest, Feature_Query_General) {
  auto func = [&](bool enable_mmap, std::string field_name) {
    FileHelper::RemoveDirectory(col_path);

    int doc_count = 1000;
    // create with normal schema
    auto schema = TestHelper::CreateNormalSchema();
    auto options = CollectionOptions{false, enable_mmap, 100 * 1024 * 1024};
    auto collection = TestHelper::CreateCollectionWithDoc(
        col_path, *schema, options, 0, doc_count);

    ASSERT_NE(collection, nullptr);

    auto stats = collection->Stats().value();
    std::cout << stats.to_string_formatted() << std::endl;

    // validate query result
    for (int i = 1; i < 2; i++) {
      auto query_doc = TestHelper::CreateDoc(i, *schema);
      // std::cout << query_doc.to_detail_string() << std::endl;

      SearchQuery query;
      query.topk_ = 10;
      query.target_.field_name_ = field_name;

      auto field_scheama = schema->get_vector_field(field_name);
      ASSERT_NE(field_scheama, nullptr);
      ASSERT_TRUE(field_scheama->is_vector_field());

      if (field_scheama->is_dense_vector()) {
        auto vector = query_doc.get<std::vector<float>>(field_name);
        ASSERT_TRUE(vector.has_value());
        query.target_.set_vector(
            std::string((char *)vector.value().data(),
                        vector.value().size() * sizeof(float)));
      } else {
        auto sparse_vector =
            query_doc.get<std::pair<std::vector<uint32_t>, std::vector<float>>>(
                field_name);
        query.target_.set_sparse_vector(
            std::string((char *)sparse_vector.value().first.data(),
                        sparse_vector.value().first.size() * sizeof(uint32_t)),
            std::string((char *)sparse_vector.value().second.data(),
                        sparse_vector.value().second.size() * sizeof(float)));
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

  for (bool enable_mmap : {true, false}) {
    func(enable_mmap, "dense_fp32");
    func(enable_mmap, "sparse_fp32");
  }
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

      SearchQuery query;
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
    SearchQuery query;
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
    SearchQuery query;
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

// =============================================================================
// MultiQuery Tests
// =============================================================================

TEST_F(CollectionTest, Feature_MultiQuery_Validate) {
  FileHelper::RemoveDirectory(col_path);

  int doc_count = 100;
  auto schema = TestHelper::CreateNormalSchema();
  auto options = CollectionOptions{false, true, 100 * 1024 * 1024};
  auto collection = TestHelper::CreateCollectionWithDoc(col_path, *schema,
                                                        options, 0, doc_count);
  ASSERT_NE(collection, nullptr);

  // Test 1: Empty queries should fail
  {
    MultiQuery mvq;
    mvq.topk = 10;
    mvq.rerank = reranker::RrfParams{60};
    auto result = collection->Query(mvq);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code(), StatusCode::INVALID_ARGUMENT);
  }

  // Test 2: No reranker with multiple queries should fail
  {
    MultiQuery mvq;
    mvq.topk = 10;
    auto query_doc = TestHelper::CreateDoc(1, *schema);

    SubQuery vq1;
    vq1.num_candidates_ = 10;
    vq1.target_.field_name_ = "dense_fp32";
    auto vector = query_doc.get<std::vector<float>>("dense_fp32");
    ASSERT_TRUE(vector.has_value());
    std::get<VectorClause>(vq1.target_.clause_)
        .query_vector_.assign((char *)vector.value().data(),
                              vector.value().size() * sizeof(float));
    mvq.queries.push_back(vq1);

    SubQuery vq2;
    vq2.num_candidates_ = 10;
    vq2.target_.field_name_ = "dense_fp16";
    auto vector2 = query_doc.get<std::vector<float>>("dense_fp32");
    ASSERT_TRUE(vector2.has_value());
    std::get<VectorClause>(vq2.target_.clause_)
        .query_vector_.assign((char *)vector2.value().data(),
                              vector2.value().size() * sizeof(float));
    mvq.queries.push_back(vq2);

    auto result = collection->Query(mvq);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code(), StatusCode::INVALID_ARGUMENT);
  }

  // Test 3: Invalid field name should fail
  {
    MultiQuery mvq;
    mvq.topk = 10;
    mvq.rerank = reranker::RrfParams{60};

    SubQuery vq1;
    vq1.num_candidates_ = 10;
    vq1.target_.field_name_ = "nonexistent_field";
    std::get<VectorClause>(vq1.target_.clause_)
        .query_vector_.assign(128 * sizeof(float), '\0');
    mvq.queries.push_back(vq1);

    SubQuery vq2;
    vq2.num_candidates_ = 10;
    vq2.target_.field_name_ = "dense_fp32";
    std::get<VectorClause>(vq2.target_.clause_)
        .query_vector_.assign(128 * sizeof(float), '\0');
    mvq.queries.push_back(vq2);

    auto result = collection->Query(mvq);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code(), StatusCode::INVALID_ARGUMENT);
  }

  // Test 4: Duplicate field names should succeed (same field, different
  // vectors)
  {
    MultiQuery mvq;
    mvq.topk = 10;
    mvq.rerank = reranker::RrfParams{60};

    SubQuery vq1;
    vq1.num_candidates_ = 10;
    vq1.target_.field_name_ = "dense_fp32";
    std::get<VectorClause>(vq1.target_.clause_)
        .query_vector_.assign(128 * sizeof(float), '\0');
    mvq.queries.push_back(vq1);

    SubQuery vq2;
    vq2.num_candidates_ = 10;
    vq2.target_.field_name_ = "dense_fp32";
    std::get<VectorClause>(vq2.target_.clause_)
        .query_vector_.assign(128 * sizeof(float), '\0');
    mvq.queries.push_back(vq2);

    auto result = collection->Query(mvq);
    ASSERT_TRUE(result.has_value());
  }
}

TEST_F(CollectionTest, Feature_MultiQuery_SingleFieldWithReranker) {
  FileHelper::RemoveDirectory(col_path);

  int doc_count = 100;
  auto schema = TestHelper::CreateNormalSchema();
  auto options = CollectionOptions{false, true, 100 * 1024 * 1024};
  auto collection = TestHelper::CreateCollectionWithDoc(col_path, *schema,
                                                        options, 0, doc_count);
  ASSERT_NE(collection, nullptr);

  // Single query with reranker should fail (requires at least 2 sub-queries)
  auto query_doc = TestHelper::CreateDoc(1, *schema);

  MultiQuery mvq;
  mvq.topk = 10;
  mvq.rerank = reranker::RrfParams{60};

  SubQuery vq;
  vq.num_candidates_ = 10;
  vq.target_.field_name_ = "dense_fp32";
  auto vector = query_doc.get<std::vector<float>>("dense_fp32");
  ASSERT_TRUE(vector.has_value());
  std::get<VectorClause>(vq.target_.clause_)
      .query_vector_.assign((char *)vector.value().data(),
                            vector.value().size() * sizeof(float));
  mvq.queries.push_back(vq);

  auto result = collection->Query(mvq);
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().code(), StatusCode::INVALID_ARGUMENT);
}

TEST_F(CollectionTest, Feature_MultiQuery_MultiFieldRRF) {
  FileHelper::RemoveDirectory(col_path);

  int doc_count = 100;
  auto schema = TestHelper::CreateNormalSchema();
  auto options = CollectionOptions{false, true, 100 * 1024 * 1024};
  auto collection = TestHelper::CreateCollectionWithDoc(col_path, *schema,
                                                        options, 0, doc_count);
  ASSERT_NE(collection, nullptr);

  auto query_doc = TestHelper::CreateDoc(1, *schema);

  MultiQuery mvq;
  mvq.topk = 10;
  mvq.rerank = reranker::RrfParams{60};

  // Query dense_fp32 and dense_fp16 fields with different vectors
  auto vector1 = query_doc.get<std::vector<float>>("dense_fp32");
  ASSERT_TRUE(vector1.has_value());

  {
    SubQuery vq;
    vq.num_candidates_ = 10;
    vq.target_.field_name_ = "dense_fp32";
    std::get<VectorClause>(vq.target_.clause_)
        .query_vector_.assign((char *)vector1.value().data(),
                              vector1.value().size() * sizeof(float));
    mvq.queries.push_back(vq);
  }

  // Query sparse_fp32 field
  auto sparse =
      query_doc.get<std::pair<std::vector<uint32_t>, std::vector<float>>>(
          "sparse_fp32");
  ASSERT_TRUE(sparse.has_value());

  {
    SubQuery vq;
    vq.num_candidates_ = 10;
    vq.target_.field_name_ = "sparse_fp32";
    std::get<VectorClause>(vq.target_.clause_)
        .sparse_indices_.assign((char *)sparse.value().first.data(),
                                sparse.value().first.size() * sizeof(uint32_t));
    std::get<VectorClause>(vq.target_.clause_)
        .sparse_values_.assign((char *)sparse.value().second.data(),
                               sparse.value().second.size() * sizeof(float));
    mvq.queries.push_back(vq);
  }

  auto result = collection->Query(mvq);
  ASSERT_TRUE(result.has_value()) << result.error().message();
  EXPECT_GT(result.value().size(), 0u);
  EXPECT_LE(result.value().size(), 10u);

  // All results should have valid scores (RRF fused)
  for (const auto &doc : result.value()) {
    EXPECT_NE(doc->score(), 0.0f);
  }
}

TEST_F(CollectionTest, Feature_MultiQuery_MultiFieldWeighted) {
  FileHelper::RemoveDirectory(col_path);

  int doc_count = 100;
  auto schema = TestHelper::CreateNormalSchema();
  auto options = CollectionOptions{false, true, 100 * 1024 * 1024};
  auto collection = TestHelper::CreateCollectionWithDoc(col_path, *schema,
                                                        options, 0, doc_count);
  ASSERT_NE(collection, nullptr);

  auto query_doc = TestHelper::CreateDoc(1, *schema);

  MultiQuery mvq;
  mvq.topk = 10;
  // Weights are positional, parallel to the sub-query order below
  // (dense_fp32 first, sparse_fp32 second).
  mvq.rerank = reranker::WeightedParams{{0.7, 0.3}};

  // Query dense_fp32 field
  {
    SubQuery vq;
    vq.num_candidates_ = 10;
    vq.target_.field_name_ = "dense_fp32";
    auto vector = query_doc.get<std::vector<float>>("dense_fp32");
    ASSERT_TRUE(vector.has_value());
    std::get<VectorClause>(vq.target_.clause_)
        .query_vector_.assign((char *)vector.value().data(),
                              vector.value().size() * sizeof(float));
    mvq.queries.push_back(vq);
  }

  // Query sparse_fp32 field
  {
    SubQuery vq;
    vq.num_candidates_ = 10;
    vq.target_.field_name_ = "sparse_fp32";
    auto sparse =
        query_doc.get<std::pair<std::vector<uint32_t>, std::vector<float>>>(
            "sparse_fp32");
    ASSERT_TRUE(sparse.has_value());
    std::get<VectorClause>(vq.target_.clause_)
        .sparse_indices_.assign((char *)sparse.value().first.data(),
                                sparse.value().first.size() * sizeof(uint32_t));
    std::get<VectorClause>(vq.target_.clause_)
        .sparse_values_.assign((char *)sparse.value().second.data(),
                               sparse.value().second.size() * sizeof(float));
    mvq.queries.push_back(vq);
  }

  auto result = collection->Query(mvq);
  ASSERT_TRUE(result.has_value()) << result.error().message();
  EXPECT_GT(result.value().size(), 0u);
  EXPECT_LE(result.value().size(), 10u);
}

TEST_F(CollectionTest, Feature_MultiQuery_WithFilter) {
  FileHelper::RemoveDirectory(col_path);

  int doc_count = 100;
  auto schema = TestHelper::CreateNormalSchema();
  auto options = CollectionOptions{false, true, 100 * 1024 * 1024};
  auto collection = TestHelper::CreateCollectionWithDoc(col_path, *schema,
                                                        options, 0, doc_count);
  ASSERT_NE(collection, nullptr);

  auto query_doc = TestHelper::CreateDoc(1, *schema);

  MultiQuery mvq;
  mvq.topk = 10;
  mvq.filter = "int32 > 50";
  mvq.rerank = reranker::RrfParams{60};

  SubQuery vq1;
  vq1.num_candidates_ = 10;
  vq1.target_.field_name_ = "dense_fp32";
  auto vector = query_doc.get<std::vector<float>>("dense_fp32");
  ASSERT_TRUE(vector.has_value());
  std::get<VectorClause>(vq1.target_.clause_)
      .query_vector_.assign((char *)vector.value().data(),
                            vector.value().size() * sizeof(float));
  mvq.queries.push_back(vq1);

  auto sparse =
      query_doc.get<std::pair<std::vector<uint32_t>, std::vector<float>>>(
          "sparse_fp32");
  ASSERT_TRUE(sparse.has_value());
  SubQuery vq2;
  vq2.num_candidates_ = 10;
  vq2.target_.field_name_ = "sparse_fp32";
  std::get<VectorClause>(vq2.target_.clause_)
      .sparse_indices_.assign((char *)sparse.value().first.data(),
                              sparse.value().first.size() * sizeof(uint32_t));
  std::get<VectorClause>(vq2.target_.clause_)
      .sparse_values_.assign((char *)sparse.value().second.data(),
                             sparse.value().second.size() * sizeof(float));
  mvq.queries.push_back(vq2);

  auto result = collection->Query(mvq);
  ASSERT_TRUE(result.has_value()) << result.error().message();
  EXPECT_GT(result.value().size(), 0u);
  EXPECT_LE(result.value().size(), 10u);
}

TEST_F(CollectionTest, Feature_MultiQuery_WithOutputFields) {
  FileHelper::RemoveDirectory(col_path);

  int doc_count = 100;
  auto schema = TestHelper::CreateNormalSchema();
  auto options = CollectionOptions{false, true, 100 * 1024 * 1024};
  auto collection = TestHelper::CreateCollectionWithDoc(col_path, *schema,
                                                        options, 0, doc_count);
  ASSERT_NE(collection, nullptr);

  auto query_doc = TestHelper::CreateDoc(1, *schema);

  MultiQuery mvq;
  mvq.topk = 5;
  mvq.include_vector = false;
  mvq.output_fields = std::make_optional<std::vector<std::string>>(
      std::vector<std::string>{"int32", "string"});
  mvq.rerank = reranker::RrfParams{60};

  SubQuery vq1;
  vq1.num_candidates_ = 10;
  vq1.target_.field_name_ = "dense_fp32";
  auto vector = query_doc.get<std::vector<float>>("dense_fp32");
  ASSERT_TRUE(vector.has_value());
  std::get<VectorClause>(vq1.target_.clause_)
      .query_vector_.assign((char *)vector.value().data(),
                            vector.value().size() * sizeof(float));
  mvq.queries.push_back(vq1);

  auto sparse =
      query_doc.get<std::pair<std::vector<uint32_t>, std::vector<float>>>(
          "sparse_fp32");
  ASSERT_TRUE(sparse.has_value());
  SubQuery vq2;
  vq2.num_candidates_ = 10;
  vq2.target_.field_name_ = "sparse_fp32";
  std::get<VectorClause>(vq2.target_.clause_)
      .sparse_indices_.assign((char *)sparse.value().first.data(),
                              sparse.value().first.size() * sizeof(uint32_t));
  std::get<VectorClause>(vq2.target_.clause_)
      .sparse_values_.assign((char *)sparse.value().second.data(),
                             sparse.value().second.size() * sizeof(float));
  mvq.queries.push_back(vq2);

  auto result = collection->Query(mvq);
  ASSERT_TRUE(result.has_value()) << result.error().message();
  EXPECT_GT(result.value().size(), 0u);
  EXPECT_LE(result.value().size(), 5u);
}

TEST_F(CollectionTest, Feature_MultiQuery_CallbackReranker) {
  FileHelper::RemoveDirectory(col_path);

  int doc_count = 100;
  auto schema = TestHelper::CreateNormalSchema();
  auto options = CollectionOptions{false, true, 100 * 1024 * 1024};
  auto collection = TestHelper::CreateCollectionWithDoc(col_path, *schema,
                                                        options, 0, doc_count);
  ASSERT_NE(collection, nullptr);

  auto query_doc = TestHelper::CreateDoc(1, *schema);

  // Use a callback rerank strategy with a lambda that merges and sorts by
  // score.
  bool callback_invoked = false;
  auto callback_fn = [&callback_invoked](
                         const std::vector<DocPtrList> &query_results,
                         const std::vector<FieldSchema::Ptr> & /*fields*/,
                         int topn) -> DocPtrList {
    callback_invoked = true;
    DocPtrList all_docs;
    for (const auto &docs : query_results) {
      for (const auto &doc : docs) {
        all_docs.push_back(doc);
      }
    }
    std::sort(all_docs.begin(), all_docs.end(),
              [](const Doc::Ptr &a, const Doc::Ptr &b) {
                return a->score() > b->score();
              });
    if (static_cast<int>(all_docs.size()) > topn) {
      all_docs.resize(topn);
    }
    return all_docs;
  };

  MultiQuery mvq;
  mvq.topk = 10;
  mvq.rerank = reranker::CallbackParams{callback_fn};

  // Query dense_fp32 field
  {
    SubQuery vq;
    vq.num_candidates_ = 10;
    vq.target_.field_name_ = "dense_fp32";
    auto vector = query_doc.get<std::vector<float>>("dense_fp32");
    ASSERT_TRUE(vector.has_value());
    std::get<VectorClause>(vq.target_.clause_)
        .query_vector_.assign((char *)vector.value().data(),
                              vector.value().size() * sizeof(float));
    mvq.queries.push_back(vq);
  }

  // Query sparse_fp32 field
  {
    SubQuery vq;
    vq.num_candidates_ = 10;
    vq.target_.field_name_ = "sparse_fp32";
    auto sparse =
        query_doc.get<std::pair<std::vector<uint32_t>, std::vector<float>>>(
            "sparse_fp32");
    ASSERT_TRUE(sparse.has_value());
    std::get<VectorClause>(vq.target_.clause_)
        .sparse_indices_.assign((char *)sparse.value().first.data(),
                                sparse.value().first.size() * sizeof(uint32_t));
    std::get<VectorClause>(vq.target_.clause_)
        .sparse_values_.assign((char *)sparse.value().second.data(),
                               sparse.value().second.size() * sizeof(float));
    mvq.queries.push_back(vq);
  }

  auto result = collection->Query(mvq);
  ASSERT_TRUE(result.has_value()) << result.error().message();
  EXPECT_TRUE(callback_invoked);
  EXPECT_GT(result.value().size(), 0u);
  EXPECT_LE(result.value().size(), 10u);

  // Verify results are sorted by score descending
  for (size_t i = 1; i < result.value().size(); ++i) {
    EXPECT_GE(result.value()[i - 1]->score(), result.value()[i]->score());
  }
}

TEST_F(CollectionTest, Feature_GroupByQuery) {}

TEST_F(CollectionTest, Feature_AddColumn_General) {
  auto func = [&](bool enable_mmap) {
    FileHelper::RemoveDirectory(col_path);
    // create collection
    int doc_count = 1000;
    auto schema = TestHelper::CreateNormalSchema();
    auto options = CollectionOptions{false, enable_mmap, 64 * 1024 * 1024};
    auto collection = TestHelper::CreateCollectionWithDoc(
        col_path, *schema, options, 0, doc_count, false);

    ASSERT_TRUE(collection->Flush().ok());
    auto stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, doc_count);
    auto field_schema =
        std::make_shared<FieldSchema>("add_int32", DataType::INT32, false);
    auto s = collection->AddColumn(field_schema, "int32", AddColumnOptions());
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
      SearchQuery query;
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
    check_doc(doc_count);

    // validate query result
    for (int i = 1; i < 2; i++) {
      SearchQuery query;
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
  };
  func(true);
  func(false);
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

    auto s = collection->AddColumn(nullptr, "int32", AddColumnOptions());
    ASSERT_FALSE(s.ok());

    s = collection->AddColumn(nullptr, "", AddColumnOptions());
    ASSERT_FALSE(s.ok());

    auto field_schema =
        std::make_shared<FieldSchema>("add_int32", DataType::INT32, false);
    s = collection->AddColumn(field_schema, "non_exist_field",
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
    auto s = collection->AddColumn(field_schema, "int32", AddColumnOptions());
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
    auto s =
        collection->AddColumn(field_schema, "add_int32", AddColumnOptions());
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
    SearchQuery query;
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
      SearchQuery query;
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

TEST_F(CollectionTest, Feature_AddNullableColumn_MultiSegment) {
  int docs_per_segment = 1000;
  int num_segments = 3;
  auto schema = TestHelper::CreateNormalSchema(
      false, "demo", nullptr, std::make_shared<HnswIndexParams>(MetricType::IP),
      docs_per_segment);
  auto options = CollectionOptions{false, true, 64 * 1024 * 1024};

  auto collection = TestHelper::CreateCollectionWithDoc(
      col_path, *schema, options, 0, docs_per_segment, false);
  ASSERT_TRUE(collection->Flush().ok());

  for (int seg = 1; seg < num_segments; seg++) {
    auto s = TestHelper::CollectionInsertDoc(collection, seg * docs_per_segment,
                                             (seg + 1) * docs_per_segment);
    ASSERT_TRUE(s.ok());
    ASSERT_TRUE(collection->Flush().ok());
  }

  int total_docs = docs_per_segment * num_segments;
  auto stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, total_docs);

  // Reopen to ensure segments are persisted
  collection.reset();
  auto result = Collection::Open(col_path, options);
  ASSERT_TRUE(result.has_value());
  collection = result.value();

  // Add nullable columns without expression — this used to crash on
  // multi-segment collections
  std::vector<std::pair<std::string, DataType>> nullable_types = {
      {"add_int32_null", DataType::INT32},
      {"add_int64_null", DataType::INT64},
      {"add_float_null", DataType::FLOAT},
      {"add_double_null", DataType::DOUBLE},
  };
  for (auto &[col_name, data_type] : nullable_types) {
    auto field_schema =
        std::make_shared<FieldSchema>(col_name, data_type, true);
    auto s = collection->AddColumn(field_schema, "", AddColumnOptions());
    ASSERT_TRUE(s.ok()) << "Failed to add nullable column " << col_name << ": "
                        << s.message();
  }

  auto new_schema = collection->Schema().value();
  for (auto &[col_name, _] : nullable_types) {
    ASSERT_TRUE(new_schema.has_field(col_name));
  }

  stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, total_docs);

  // Verify all docs are fetchable and new columns have null values
  for (int i = 0; i < total_docs; i++) {
    auto expect_doc = TestHelper::CreateDoc(i, new_schema);
    auto fetch_result = collection->Fetch({expect_doc.pk()});
    ASSERT_TRUE(fetch_result.has_value());
    ASSERT_EQ(fetch_result.value().size(), 1);
    ASSERT_EQ(fetch_result.value().count(expect_doc.pk()), 1);
    auto doc = fetch_result.value()[expect_doc.pk()];
    ASSERT_NE(doc, nullptr);
  }

  // Insert more docs after adding columns and verify
  auto s = TestHelper::CollectionInsertDoc(collection, total_docs,
                                           total_docs + docs_per_segment);
  ASSERT_TRUE(s.ok());
  stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, total_docs + docs_per_segment);
}

TEST_F(CollectionTest, Feature_AddColumn_MultiSegment_MixedOps) {
  int docs_per_segment = 1000;
  int num_segments = 3;
  auto schema = TestHelper::CreateNormalSchema(
      false, "demo", nullptr, std::make_shared<HnswIndexParams>(MetricType::IP),
      docs_per_segment);
  auto options = CollectionOptions{false, true, 64 * 1024 * 1024};

  auto collection = TestHelper::CreateCollectionWithDoc(
      col_path, *schema, options, 0, docs_per_segment, false);
  ASSERT_TRUE(collection->Flush().ok());

  for (int seg = 1; seg < num_segments; seg++) {
    auto s = TestHelper::CollectionInsertDoc(collection, seg * docs_per_segment,
                                             (seg + 1) * docs_per_segment);
    ASSERT_TRUE(s.ok());
    ASSERT_TRUE(collection->Flush().ok());
  }

  int total_docs = docs_per_segment * num_segments;

  // Reopen
  collection.reset();
  auto result = Collection::Open(col_path, options);
  ASSERT_TRUE(result.has_value());
  collection = result.value();

  // 1) Add column with expression on multi-segment collection
  auto expr_field =
      std::make_shared<FieldSchema>("expr_col", DataType::INT32, false);
  auto s = collection->AddColumn(expr_field, "int32 + 1", AddColumnOptions());
  ASSERT_TRUE(s.ok()) << "AddColumn with expression failed: " << s.message();

  auto stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, total_docs);

  // 2) Add nullable column without expression after expression-based column
  auto null_field =
      std::make_shared<FieldSchema>("null_col", DataType::INT64, true);
  s = collection->AddColumn(null_field, "", AddColumnOptions());
  ASSERT_TRUE(s.ok()) << "AddColumn nullable failed: " << s.message();

  stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, total_docs);

  // 3) Drop the expression-based column, then add another nullable column
  s = collection->DropColumn("expr_col");
  ASSERT_TRUE(s.ok()) << "DropColumn failed: " << s.message();

  auto null_field2 =
      std::make_shared<FieldSchema>("null_col2", DataType::FLOAT, true);
  s = collection->AddColumn(null_field2, "", AddColumnOptions());
  ASSERT_TRUE(s.ok()) << "AddColumn nullable after drop failed: "
                      << s.message();

  stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, total_docs);

  // 4) Verify schema correctness
  auto new_schema = collection->Schema().value();
  ASSERT_FALSE(new_schema.has_field("expr_col"));
  ASSERT_TRUE(new_schema.has_field("null_col"));
  ASSERT_TRUE(new_schema.has_field("null_col2"));

  // 5) Verify all docs are still fetchable
  for (int i = 0; i < total_docs; i++) {
    auto expect_doc = TestHelper::CreateDoc(i, new_schema);
    auto fetch_result = collection->Fetch({expect_doc.pk()});
    ASSERT_TRUE(fetch_result.has_value());
    ASSERT_EQ(fetch_result.value().size(), 1);
  }
}

TEST_F(CollectionTest, Feature_AlterColumn_MultiSegment) {
  int docs_per_segment = 1000;
  int num_segments = 3;
  auto schema = TestHelper::CreateNormalSchema(
      false, "demo", nullptr, std::make_shared<HnswIndexParams>(MetricType::IP),
      docs_per_segment);
  auto options = CollectionOptions{false, true, 64 * 1024 * 1024};

  auto collection = TestHelper::CreateCollectionWithDoc(
      col_path, *schema, options, 0, docs_per_segment, false);
  ASSERT_TRUE(collection->Flush().ok());

  for (int seg = 1; seg < num_segments; seg++) {
    auto s = TestHelper::CollectionInsertDoc(collection, seg * docs_per_segment,
                                             (seg + 1) * docs_per_segment);
    ASSERT_TRUE(s.ok());
    ASSERT_TRUE(collection->Flush().ok());
  }

  int total_docs = docs_per_segment * num_segments;

  // Reopen
  collection.reset();
  auto result = Collection::Open(col_path, options);
  ASSERT_TRUE(result.has_value());
  collection = result.value();

  // Alter type: int32 -> int64
  auto altered_field =
      std::make_shared<FieldSchema>("int32", DataType::INT64, false);
  auto s =
      collection->AlterColumn("int32", "", altered_field, AlterColumnOptions());
  ASSERT_TRUE(s.ok()) << "alter column type failed: " << s.message();

  auto new_schema = collection->Schema().value();
  ASSERT_TRUE(new_schema.get_field("int32")->data_type() == DataType::INT64);

  auto stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, total_docs);

  // Rename column
  s = collection->AlterColumn("uint32", "renamed_uint32", nullptr,
                              AlterColumnOptions());
  ASSERT_TRUE(s.ok()) << "alter column rename failed: " << s.message();

  new_schema = collection->Schema().value();
  ASSERT_FALSE(new_schema.has_field("uint32"));
  ASSERT_TRUE(new_schema.has_field("renamed_uint32"));

  stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, total_docs);

  // Verify all docs are fetchable
  for (int i = 0; i < total_docs; i++) {
    auto expect_doc = TestHelper::CreateDoc(i, new_schema);
    auto fetch_result = collection->Fetch({expect_doc.pk()});
    ASSERT_TRUE(fetch_result.has_value());
    ASSERT_EQ(fetch_result.value().size(), 1);
  }
}

TEST_F(CollectionTest, Feature_DropColumn_MultiSegment) {
  int docs_per_segment = 1000;
  int num_segments = 3;
  auto schema = TestHelper::CreateNormalSchema(
      false, "demo", nullptr, std::make_shared<HnswIndexParams>(MetricType::IP),
      docs_per_segment);
  auto options = CollectionOptions{false, true, 64 * 1024 * 1024};

  auto collection = TestHelper::CreateCollectionWithDoc(
      col_path, *schema, options, 0, docs_per_segment, false);
  ASSERT_TRUE(collection->Flush().ok());

  for (int seg = 1; seg < num_segments; seg++) {
    auto s = TestHelper::CollectionInsertDoc(collection, seg * docs_per_segment,
                                             (seg + 1) * docs_per_segment);
    ASSERT_TRUE(s.ok());
    ASSERT_TRUE(collection->Flush().ok());
  }

  int total_docs = docs_per_segment * num_segments;

  // Reopen
  collection.reset();
  auto result = Collection::Open(col_path, options);
  ASSERT_TRUE(result.has_value());
  collection = result.value();

  // Drop multiple columns
  std::vector<std::string> to_drop = {"int32", "uint32", "float"};
  for (auto &col_name : to_drop) {
    auto s = collection->DropColumn(col_name);
    ASSERT_TRUE(s.ok()) << "drop column " << col_name
                        << " failed: " << s.message();
  }

  auto new_schema = collection->Schema().value();
  for (auto &col_name : to_drop) {
    ASSERT_FALSE(new_schema.has_field(col_name));
  }
  ASSERT_TRUE(new_schema.has_field("int64"));
  ASSERT_TRUE(new_schema.has_field("double"));

  auto stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, total_docs);

  // Insert more docs after dropping columns
  auto s = TestHelper::CollectionInsertDoc(collection, total_docs,
                                           total_docs + docs_per_segment);
  ASSERT_TRUE(s.ok());
  stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, total_docs + docs_per_segment);
}

TEST_F(CollectionTest, Feature_AddNullableColumn_ReopenVerifyNull) {
  int docs_per_segment = 1000;
  auto schema = TestHelper::CreateNormalSchema(
      false, "demo", nullptr, std::make_shared<HnswIndexParams>(MetricType::IP),
      docs_per_segment);
  auto options = CollectionOptions{false, true, 64 * 1024 * 1024};

  auto collection = TestHelper::CreateCollectionWithDoc(
      col_path, *schema, options, 0, docs_per_segment, false);
  ASSERT_TRUE(collection->Flush().ok());

  // Add another segment
  auto s = TestHelper::CollectionInsertDoc(collection, docs_per_segment,
                                           docs_per_segment * 2);
  ASSERT_TRUE(s.ok());
  ASSERT_TRUE(collection->Flush().ok());

  int total_docs = docs_per_segment * 2;

  // Add nullable column
  auto nullable_field =
      std::make_shared<FieldSchema>("null_col", DataType::INT64, true);
  s = collection->AddColumn(nullable_field, "", AddColumnOptions());
  ASSERT_TRUE(s.ok()) << "add nullable column failed: " << s.message();

  // Close and reopen
  collection.reset();
  auto result = Collection::Open(col_path, options);
  ASSERT_TRUE(result.has_value());
  collection = result.value();

  auto new_schema = collection->Schema().value();
  ASSERT_TRUE(new_schema.has_field("null_col"));

  auto stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, total_docs);

  // Verify docs are fetchable after reopen
  for (int i = 0; i < total_docs; i++) {
    auto expect_doc = TestHelper::CreateDoc(i, new_schema);
    auto fetch_result = collection->Fetch({expect_doc.pk()});
    ASSERT_TRUE(fetch_result.has_value());
    ASSERT_EQ(fetch_result.value().size(), 1);
  }

  // Insert new docs with value for the added column and verify
  s = TestHelper::CollectionInsertDoc(collection, total_docs,
                                      total_docs + docs_per_segment);
  ASSERT_TRUE(s.ok());
  stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, total_docs + docs_per_segment);
}

TEST_F(CollectionTest, Feature_AddColumn_WithUnflushedData) {
  int doc_count = 1000;
  auto schema = TestHelper::CreateNormalSchema(
      false, "demo", nullptr, std::make_shared<HnswIndexParams>(MetricType::IP),
      doc_count);
  auto options = CollectionOptions{false, true, 64 * 1024 * 1024};

  // Create collection with flushed data (segment 1)
  auto collection = TestHelper::CreateCollectionWithDoc(
      col_path, *schema, options, 0, doc_count, false);
  ASSERT_TRUE(collection->Flush().ok());

  // Insert more unflushed data (in writing segment)
  auto s =
      TestHelper::CollectionInsertDoc(collection, doc_count, doc_count + 500);
  ASSERT_TRUE(s.ok());

  auto stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, doc_count + 500);

  // AddColumn while writing segment has unflushed data
  auto field_schema =
      std::make_shared<FieldSchema>("new_col", DataType::INT32, false);
  s = collection->AddColumn(field_schema, "int32 + 1", AddColumnOptions());
  ASSERT_TRUE(s.ok()) << "AddColumn with unflushed data failed: "
                      << s.message();

  auto new_schema = collection->Schema().value();
  ASSERT_TRUE(new_schema.has_field("new_col"));

  stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, doc_count + 500);

  // Add nullable column while writing segment has unflushed data
  auto nullable_field =
      std::make_shared<FieldSchema>("null_unflushed", DataType::INT64, true);
  s = collection->AddColumn(nullable_field, "", AddColumnOptions());
  ASSERT_TRUE(s.ok()) << "AddColumn nullable with unflushed data failed: "
                      << s.message();

  stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, doc_count + 500);

  // Insert after add column and flush
  s = TestHelper::CollectionInsertDoc(collection, doc_count + 500,
                                      doc_count + 1000);
  ASSERT_TRUE(s.ok());
  ASSERT_TRUE(collection->Flush().ok());

  stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, doc_count + 1000);
}

TEST_F(CollectionTest, Feature_ColumnDDL_ChainedOps_MultiSegment) {
  int docs_per_segment = 1000;
  int num_segments = 3;
  auto schema = TestHelper::CreateNormalSchema(
      false, "demo", nullptr, std::make_shared<HnswIndexParams>(MetricType::IP),
      docs_per_segment);
  auto options = CollectionOptions{false, true, 64 * 1024 * 1024};

  auto collection = TestHelper::CreateCollectionWithDoc(
      col_path, *schema, options, 0, docs_per_segment, false);
  ASSERT_TRUE(collection->Flush().ok());

  for (int seg = 1; seg < num_segments; seg++) {
    auto s = TestHelper::CollectionInsertDoc(collection, seg * docs_per_segment,
                                             (seg + 1) * docs_per_segment);
    ASSERT_TRUE(s.ok());
    ASSERT_TRUE(collection->Flush().ok());
  }

  int total_docs = docs_per_segment * num_segments;

  // Reopen
  collection.reset();
  auto result = Collection::Open(col_path, options);
  ASSERT_TRUE(result.has_value());
  collection = result.value();

  // Chain 1: add nullable -> alter type -> drop -> add again
  auto field_v1 =
      std::make_shared<FieldSchema>("chain_col", DataType::INT32, true);
  auto s = collection->AddColumn(field_v1, "", AddColumnOptions());
  ASSERT_TRUE(s.ok()) << "chain add v1 failed: " << s.message();

  auto field_v2 =
      std::make_shared<FieldSchema>("chain_col", DataType::INT64, true);
  s = collection->AlterColumn("chain_col", "", field_v2, AlterColumnOptions());
  ASSERT_TRUE(s.ok()) << "chain alter failed: " << s.message();

  auto new_schema = collection->Schema().value();
  ASSERT_TRUE(new_schema.get_field("chain_col")->data_type() ==
              DataType::INT64);

  s = collection->DropColumn("chain_col");
  ASSERT_TRUE(s.ok()) << "chain drop failed: " << s.message();

  new_schema = collection->Schema().value();
  ASSERT_FALSE(new_schema.has_field("chain_col"));

  auto field_v3 =
      std::make_shared<FieldSchema>("chain_col", DataType::FLOAT, false);
  s = collection->AddColumn(field_v3, "float + 1.0", AddColumnOptions());
  ASSERT_TRUE(s.ok()) << "chain re-add failed: " << s.message();

  new_schema = collection->Schema().value();
  ASSERT_TRUE(new_schema.has_field("chain_col"));
  ASSERT_TRUE(new_schema.get_field("chain_col")->data_type() ==
              DataType::FLOAT);

  auto stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, total_docs);

  // Chain 2: add with expr -> rename -> drop
  auto expr_field =
      std::make_shared<FieldSchema>("chain2_col", DataType::DOUBLE, false);
  s = collection->AddColumn(expr_field, "double", AddColumnOptions());
  ASSERT_TRUE(s.ok()) << "chain2 add failed: " << s.message();

  s = collection->AlterColumn("chain2_col", "chain2_renamed", nullptr,
                              AlterColumnOptions());
  ASSERT_TRUE(s.ok()) << "chain2 rename failed: " << s.message();

  new_schema = collection->Schema().value();
  ASSERT_FALSE(new_schema.has_field("chain2_col"));
  ASSERT_TRUE(new_schema.has_field("chain2_renamed"));

  s = collection->DropColumn("chain2_renamed");
  ASSERT_TRUE(s.ok()) << "chain2 drop failed: " << s.message();

  stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, total_docs);

  // Verify all docs still fetchable after all chained operations
  for (int i = 0; i < total_docs; i++) {
    new_schema = collection->Schema().value();
    auto expect_doc = TestHelper::CreateDoc(i, new_schema);
    auto fetch_result = collection->Fetch({expect_doc.pk()});
    ASSERT_TRUE(fetch_result.has_value());
    ASSERT_EQ(fetch_result.value().size(), 1);
  }

  // Insert more docs after all operations
  s = TestHelper::CollectionInsertDoc(collection, total_docs,
                                      total_docs + docs_per_segment);
  ASSERT_TRUE(s.ok());
  stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, total_docs + docs_per_segment);
}

TEST_F(CollectionTest, Feature_AlterColumn_NullableValidation) {
  int doc_count = 1000;
  auto schema = TestHelper::CreateNormalSchema(
      false, "demo", nullptr, std::make_shared<HnswIndexParams>(MetricType::IP),
      doc_count);
  auto options = CollectionOptions{false, true, 64 * 1024 * 1024};

  auto collection = TestHelper::CreateCollectionWithDoc(
      col_path, *schema, options, 0, doc_count, false);
  ASSERT_TRUE(collection->Flush().ok());

  // Add a nullable column
  auto nullable_field =
      std::make_shared<FieldSchema>("nullable_col", DataType::INT32, true);
  auto s = collection->AddColumn(nullable_field, "", AddColumnOptions());
  ASSERT_TRUE(s.ok());

  // Attempt to alter nullable column to non-nullable — should fail
  auto non_nullable_field =
      std::make_shared<FieldSchema>("nullable_col", DataType::INT32, false);
  s = collection->AlterColumn("nullable_col", "", non_nullable_field,
                              AlterColumnOptions());
  ASSERT_FALSE(s.ok()) << "should reject nullable->non-nullable alter";

  // Alter non-nullable to nullable — should succeed
  auto to_nullable =
      std::make_shared<FieldSchema>("int32", DataType::INT32, true);
  s = collection->AlterColumn("int32", "", to_nullable, AlterColumnOptions());
  ASSERT_TRUE(s.ok()) << "non-nullable->nullable alter failed: " << s.message();

  auto new_schema = collection->Schema().value();
  ASSERT_TRUE(new_schema.get_field("int32")->nullable());

  // Alter type and nullable at the same time
  auto type_and_nullable =
      std::make_shared<FieldSchema>("uint32", DataType::INT64, true);
  s = collection->AlterColumn("uint32", "", type_and_nullable,
                              AlterColumnOptions());
  ASSERT_TRUE(s.ok()) << "alter type+nullable failed: " << s.message();

  new_schema = collection->Schema().value();
  ASSERT_TRUE(new_schema.get_field("uint32")->data_type() == DataType::INT64);
  ASSERT_TRUE(new_schema.get_field("uint32")->nullable());

  auto stats = collection->Stats().value();
  ASSERT_EQ(stats.doc_count, doc_count);
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
  s = collection->AddColumn(field_schema, "int32", AddColumnOptions());
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
    auto s = collection->AddColumn(field_schema, "int32", AddColumnOptions());
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

#if RABITQ_SUPPORTED
TEST_F(CollectionTest, Feature_Optimize_HNSW_RABITQ) {
  auto func = [](MetricType metric_type, int concurrency) {
    FileHelper::RemoveDirectory(col_path);

    int doc_count = 1000;

    // create simple schema with only FP32 dense vector for HNSW_RABITQ
    auto schema = std::make_shared<CollectionSchema>("demo");
    schema->set_max_doc_count_per_segment(MAX_DOC_COUNT_PER_SEGMENT);

    auto hnsw_rabitq_params = std::make_shared<HnswRabitqIndexParams>(
        metric_type, 7, 256, 16, 200, 0);
    schema->add_field(std::make_shared<FieldSchema>(
        "dense_fp32", DataType::VECTOR_FP32, 128, false, hnsw_rabitq_params));

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

  func(MetricType::L2, 0);
  func(MetricType::L2, 4);
  func(MetricType::IP, 0);
  func(MetricType::IP, 4);
  // TODO: cosine dense not match, may be accuracy issue
  // func(MetricType::COSINE, 0);
  // func(MetricType::COSINE, 4);
}
#endif

#if DISKANN_SUPPORTED
TEST_F(CollectionTest, Feature_Optimize_DiskAnn) {
  auto func = [](MetricType metric_type, int concurrency) {
    FileHelper::RemoveDirectory(col_path);

    int doc_count = 10000;

    auto schema = std::make_shared<CollectionSchema>("diskann_demo");
    schema->set_max_doc_count_per_segment(MAX_DOC_COUNT_PER_SEGMENT);

    auto diskann_params = std::make_shared<DiskAnnIndexParams>(metric_type);
    schema->add_field(std::make_shared<FieldSchema>(
        "dense_fp32", DataType::VECTOR_FP32, 128, false, diskann_params));

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

    // check_doc();
    std::cout << "check success 2" << std::endl;

    collection.reset();
    auto result = Collection::Open(col_path, options);
    ASSERT_TRUE(result.has_value());
    collection = std::move(result.value());

    // check_doc();
    std::cout << "check success 3" << std::endl;
  };

  func(MetricType::L2, 0);
  func(MetricType::L2, 4);
  func(MetricType::IP, 0);
  func(MetricType::IP, 4);
  // func(MetricType::COSINE, 0);
  // func(MetricType::COSINE, 4);
}
#endif

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

TEST_F(CollectionTest, Feature_Query_NullableFilter_WithoutIndex) {
  auto run_test = [&](bool with_scalar_index) {
    FileHelper::RemoveDirectory(col_path);
    IndexParams::Ptr scalar_idx =
        with_scalar_index ? std::make_shared<InvertIndexParams>(false)
                          : nullptr;
    auto schema =
        TestHelper::CreateNormalSchema(/*nullable=*/true, "demo", scalar_idx);
    CollectionOptions options{false, true, 100 * 1024 * 1024};
    auto result = Collection::CreateAndOpen(col_path, *schema, options);
    ASSERT_TRUE(result.has_value());
    auto collection = result.value();

    int non_null_count = 50;
    int null_count = 50;
    int total = non_null_count + null_count;

    auto s = TestHelper::CollectionInsertDoc(collection, 0, non_null_count,
                                             /*nullable=*/false);
    ASSERT_TRUE(s.ok());
    s = TestHelper::CollectionInsertDoc(collection, non_null_count, total,
                                        /*nullable=*/true);
    ASSERT_TRUE(s.ok());
    collection->Flush();

    auto stats = collection->Stats().value();
    ASSERT_EQ(stats.doc_count, total);

    auto query_doc = TestHelper::CreateDoc(1, *schema);
    SearchQuery query;
    query.topk_ = total;
    query.target_.field_name_ = "dense_fp32";
    auto vec = query_doc.get<std::vector<float>>("dense_fp32");
    ASSERT_TRUE(vec.has_value());
    query.target_.set_vector(std::string((char *)vec.value().data(),
                                         vec.value().size() * sizeof(float)));
    query.filter_ = "int32 > 0";
    query.output_fields_ = std::vector<std::string>{"int32"};

    auto query_result = collection->Query(query);
    ASSERT_TRUE(query_result.has_value());
    for (auto &doc : query_result.value()) {
      auto int32_val = doc->get<int32_t>("int32");
      ASSERT_TRUE(int32_val.has_value())
          << "Null doc leaked through filter: " << doc->pk()
          << " (with_scalar_index=" << with_scalar_index << ")";
      ASSERT_GT(int32_val.value(), 0);
    }
    ASSERT_EQ(query_result.value().size(), non_null_count - 1)
        << "with_scalar_index=" << with_scalar_index;
  };

  run_test(false);
  run_test(true);
}

TEST_F(CollectionTest, Feature_Fetch_OutputFields) {
  FileHelper::RemoveDirectory(col_path);

  auto schema = TestHelper::CreateNormalSchema(false);
  auto options = CollectionOptions{false, true, 100 * 1024 * 1024};
  int doc_count = 10;
  auto collection = TestHelper::CreateCollectionWithDoc(
      col_path, *schema, options, 0, doc_count, false);
  ASSERT_NE(collection, nullptr);

  auto expect_doc = TestHelper::CreateDoc(0, *schema);
  const std::string pk = expect_doc.pk();

  // Case 1: output_fields = nullopt -> all fields returned
  {
    auto result = collection->Fetch({pk}, std::nullopt);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result.value().size(), 1);
    auto doc = result.value()[pk];
    ASSERT_NE(doc, nullptr);
    ASSERT_TRUE(doc->has("int32"));
    ASSERT_TRUE(doc->has("string"));
    ASSERT_TRUE(doc->has("float"));
  }

  // Case 2: output_fields = {"int32", "string"} -> only those fields returned
  {
    auto result =
        collection->Fetch({pk}, std::vector<std::string>{"int32", "string"});
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result.value().size(), 1);
    auto doc = result.value()[pk];
    ASSERT_NE(doc, nullptr);
    // requested fields should be present
    ASSERT_TRUE(doc->has("int32"));
    ASSERT_TRUE(doc->has("string"));
    // unrequested scalar fields should be absent
    ASSERT_FALSE(doc->has("float"));
    ASSERT_FALSE(doc->has("double"));
    ASSERT_FALSE(doc->has("uint32"));
  }

  // Case 3: output_fields = {} (empty vector) -> no scalar fields returned
  {
    auto result = collection->Fetch({pk}, std::vector<std::string>{});
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result.value().size(), 1);
    auto doc = result.value()[pk];
    ASSERT_NE(doc, nullptr);
    // pk should still be set
    ASSERT_EQ(doc->pk(), pk);
    // no scalar fields should be present
    ASSERT_FALSE(doc->has("int32"));
    ASSERT_FALSE(doc->has("string"));
    ASSERT_FALSE(doc->has("float"));
  }

  // Case 4: non-existent pk -> nullptr in map
  {
    auto result = collection->Fetch({"nonexistent_pk"},
                                    std::vector<std::string>{"int32"});
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result.value().size(), 1);
    ASSERT_EQ(result.value()["nonexistent_pk"], nullptr);
  }

  // Case 5: output_fields with non-existent field name -> ignored gracefully
  {
    auto result = collection->Fetch(
        {pk}, std::vector<std::string>{"int32", "nonexistent_field"});
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result.value().size(), 1);
    auto doc = result.value()[pk];
    ASSERT_NE(doc, nullptr);
    ASSERT_TRUE(doc->has("int32"));
    ASSERT_FALSE(doc->has("nonexistent_field"));
  }

  // Case 6: include_vector = false (default) -> no vector fields returned
  {
    auto result = collection->Fetch({pk}, std::nullopt, false);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result.value().size(), 1);
    auto doc = result.value()[pk];
    ASSERT_NE(doc, nullptr);
    ASSERT_TRUE(doc->has("int32"));
    ASSERT_FALSE(doc->has("dense_fp32"));
  }

  // Case 7: include_vector = true -> vector fields returned
  {
    auto result = collection->Fetch({pk}, std::nullopt, true);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result.value().size(), 1);
    auto doc = result.value()[pk];
    ASSERT_NE(doc, nullptr);
    ASSERT_TRUE(doc->has("int32"));
    ASSERT_TRUE(doc->has("dense_fp32"));
  }

  // Case 8: include_vector = true with output_fields
  {
    auto result =
        collection->Fetch({pk}, std::vector<std::string>{"int32"}, true);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result.value().size(), 1);
    auto doc = result.value()[pk];
    ASSERT_NE(doc, nullptr);
    ASSERT_TRUE(doc->has("int32"));
    ASSERT_FALSE(doc->has("string"));
    ASSERT_TRUE(doc->has("dense_fp32"));
  }

  ASSERT_TRUE(collection->Destroy().ok());
}

// FTS-only collection (no vector field).  Covers Create / Insert / FTS Query
// / Delete / Optimize-with-rebuild round trip — the rebuild path exercises
// SegmentHelper::ReduceFts, which is the most invasive consumer of the
// "schema may have zero vector fields" relaxation.
TEST_F(CollectionTest, Feature_NoVectorCollection_FtsLifecycle) {
  FileHelper::RemoveDirectory(col_path);

  auto schema = std::make_shared<CollectionSchema>("fts_only");
  schema->add_field(std::make_shared<FieldSchema>("title", DataType::STRING));
  schema->add_field(std::make_shared<FieldSchema>(
      "content", DataType::STRING, false, std::make_shared<FtsIndexParams>()));

  auto create_res = Collection::CreateAndOpen(col_path, *schema,
                                              CollectionOptions{false, true});
  ASSERT_TRUE(create_res.has_value()) << create_res.error().message();
  auto col = std::move(create_res.value());

  // Insert a corpus where 4 of 5 docs contain "hello".  Doc 4 is the only
  // doc without "hello"; we'll delete it later to verify Optimize correctly
  // rewrites postings + stats.
  auto make_doc = [](uint64_t id, const std::string &title,
                     const std::string &content) {
    Doc d;
    d.set_pk("pk_" + std::to_string(id));
    d.set<std::string>("title", title);
    d.set<std::string>("content", content);
    return d;
  };
  std::vector<Doc> docs;
  docs.push_back(make_doc(0, "intro", "hello world"));
  docs.push_back(make_doc(1, "guide", "hello foo bar"));
  docs.push_back(make_doc(2, "tips", "hello baz"));
  docs.push_back(make_doc(3, "more", "hello hello"));
  docs.push_back(make_doc(4, "other", "nothing relevant"));
  ASSERT_TRUE(col->Insert(docs).has_value());
  ASSERT_EQ(col->Stats().value().doc_count, 5u);

  auto fts_search = [&](const std::string &term) {
    SearchQuery vq;
    vq.target_.field_name_ = "content";
    vq.topk_ = 10;
    FtsClause fts_q;
    fts_q.query_string_ = term;
    vq.target_.clause_ = fts_q;
    auto r = col->Query(vq);
    EXPECT_TRUE(r.has_value()) << r.error().message();
    return r.has_value() ? r.value() : DocPtrList{};
  };

  // Baseline: 4 docs hit "hello".
  ASSERT_EQ(fts_search("hello").size(), 4u);

  // Delete enough to push delete ratio above COMPACT_DELETE_RATIO_THRESHOLD
  // (0.3) so the next Optimize sets rebuild=true and exercises ReduceFts.
  // Drop pk_0 and pk_4: 2/5 = 40% deletes, and pk_0 carries one "hello".
  ASSERT_TRUE(col->Delete({"pk_0", "pk_4"}).has_value());
  ASSERT_EQ(col->Stats().value().doc_count, 3u);

  // Tombstone filter applied at query time — "hello" now returns 3 docs.
  ASSERT_EQ(fts_search("hello").size(), 3u);
  // Doc 4 (only "nothing") is deleted ⇒ no hit for its unique term.
  ASSERT_EQ(fts_search("nothing").size(), 0u);

  // Optimize physically removes tombstones and rebuilds FTS postings via
  // FtsRocksdbReducer.  Same recall expected after rebuild.
  ASSERT_TRUE(col->Optimize().ok());
  ASSERT_EQ(col->Stats().value().doc_count, 3u);
  ASSERT_EQ(fts_search("hello").size(), 3u);
  ASSERT_EQ(fts_search("nothing").size(), 0u);

  // Close and reopen in read-only mode (same as bench query mode).
  col.reset();
  CollectionOptions ro_options;
  ro_options.read_only_ = true;
  auto reopen_res = Collection::Open(col_path, ro_options);
  ASSERT_TRUE(reopen_res.has_value()) << reopen_res.error().message();
  col = std::move(reopen_res.value());

  auto fts_search_ro = [&](const std::string &term) {
    SearchQuery vq;
    vq.target_.field_name_ = "content";
    vq.topk_ = 10;
    FtsClause fts_q;
    fts_q.query_string_ = term;
    vq.target_.clause_ = fts_q;
    auto r = col->Query(vq);
    EXPECT_TRUE(r.has_value()) << r.error().message();
    return r.has_value() ? r.value() : DocPtrList{};
  };

  ASSERT_EQ(fts_search_ro("hello").size(), 3u);
  ASSERT_EQ(fts_search_ro("nothing").size(), 0u);

  col.reset();
  FileHelper::RemoveDirectory(col_path);
}

// Dynamic CreateIndex/DropIndex for FTS: create an FTS index on a STRING column
// that already has data, verify queries hit, then drop the index and verify FTS
// is no longer available. Also covers reopen persistence.
TEST_F(CollectionTest, Feature_CreateOrDropFtsIndex) {
#ifdef __ANDROID__
  GTEST_SKIP() << "Skipped on Android: emulator filesystem lacks hardlink "
                  "support (needed by RocksDB checkpoint)";
#endif
  auto build_schema = [](bool with_fts) {
    auto schema = std::make_shared<CollectionSchema>("fts_dyn");
    schema->add_field(std::make_shared<FieldSchema>("title", DataType::STRING));
    schema->add_field(std::make_shared<FieldSchema>(
        "content", DataType::STRING, false,
        with_fts ? std::make_shared<FtsIndexParams>() : nullptr));
    schema->add_field(std::make_shared<FieldSchema>(
        "vec", DataType::VECTOR_FP32, 4, false,
        std::make_shared<FlatIndexParams>(MetricType::IP)));
    return schema;
  };
  auto make_doc = [](uint64_t id, const std::string &title,
                     const std::string &content) {
    Doc d;
    d.set_pk("pk_" + std::to_string(id));
    d.set<std::string>("title", title);
    d.set<std::string>("content", content);
    d.set<std::vector<float>>("vec", std::vector<float>(4, float(id) + 0.1f));
    return d;
  };
  auto fts_search = [](Collection::Ptr &col, const std::string &term) {
    SearchQuery vq;
    vq.target_.field_name_ = "content";
    vq.topk_ = 10;
    FtsClause fts_q;
    fts_q.query_string_ = term;
    vq.target_.clause_ = fts_q;
    return col->Query(vq);
  };

  // CreateIndex(nullptr) should fail with INVALID_ARGUMENT.
  {
    FileHelper::RemoveDirectory(col_path);
    auto schema = build_schema(false);
    auto col_res = Collection::CreateAndOpen(col_path, *schema,
                                             CollectionOptions{false, true});
    ASSERT_TRUE(col_res.has_value()) << col_res.error().message();
    auto col = std::move(col_res.value());

    auto s_null = col->CreateIndex("content", nullptr);
    ASSERT_FALSE(s_null.ok());
    ASSERT_EQ(s_null.code(), StatusCode::INVALID_ARGUMENT);

    col.reset();
    FileHelper::RemoveDirectory(col_path);
  }

  // Case 1: CreateIndex(FtsIndexParams) on a STRING column without FTS.
  // Insert data first, then create index, verify queries hit, verify reopen.
  {
    FileHelper::RemoveDirectory(col_path);
    auto schema = build_schema(false);
    CollectionOptions options{false, true};
    auto col_res = Collection::CreateAndOpen(col_path, *schema, options);
    ASSERT_TRUE(col_res.has_value()) << col_res.error().message();
    auto col = std::move(col_res.value());

    std::vector<Doc> docs;
    docs.push_back(make_doc(0, "intro", "hello world"));
    docs.push_back(make_doc(1, "guide", "hello foo"));
    docs.push_back(make_doc(2, "more", "nothing here"));
    ASSERT_TRUE(col->Insert(docs).has_value());
    ASSERT_TRUE(col->Flush().ok());

    // FTS query before index creation should fail.
    auto q_before = fts_search(col, "hello");
    ASSERT_FALSE(q_before.has_value());

    // Create FTS index.
    auto s = col->CreateIndex("content", std::make_shared<FtsIndexParams>());
    ASSERT_TRUE(s.ok()) << s.message();

    // FTS query should now succeed.
    auto q_after = fts_search(col, "hello");
    ASSERT_TRUE(q_after.has_value()) << q_after.error().message();
    ASSERT_EQ(q_after.value().size(), 2u);

    // "nothing" appears in doc 2 only.
    auto q_nothing = fts_search(col, "nothing");
    ASSERT_TRUE(q_nothing.has_value()) << q_nothing.error().message();
    ASSERT_EQ(q_nothing.value().size(), 1u);

    // Reopen and verify persistence.
    col.reset();
    auto reopen_res = Collection::Open(col_path, options);
    ASSERT_TRUE(reopen_res.has_value()) << reopen_res.error().message();
    col = reopen_res.value();

    auto q_reopen = fts_search(col, "hello");
    ASSERT_TRUE(q_reopen.has_value()) << q_reopen.error().message();
    ASSERT_EQ(q_reopen.value().size(), 2u);

    col.reset();
    FileHelper::RemoveDirectory(col_path);
  }

  // Case 2: DropIndex on an FTS column removes the FTS index.
  {
    FileHelper::RemoveDirectory(col_path);
    auto schema = build_schema(true);
    CollectionOptions options{false, true};
    auto col_res = Collection::CreateAndOpen(col_path, *schema, options);
    ASSERT_TRUE(col_res.has_value()) << col_res.error().message();
    auto col = std::move(col_res.value());

    std::vector<Doc> docs;
    docs.push_back(make_doc(0, "intro", "hello world"));
    docs.push_back(make_doc(1, "guide", "hello foo"));
    ASSERT_TRUE(col->Insert(docs).has_value());
    ASSERT_TRUE(col->Flush().ok());

    // Baseline: FTS query works.
    auto baseline = fts_search(col, "hello");
    ASSERT_TRUE(baseline.has_value());
    ASSERT_EQ(baseline.value().size(), 2u);

    // Drop FTS index.
    auto s = col->DropIndex("content");
    ASSERT_TRUE(s.ok()) << s.message();

    // FTS query should now fail (field no longer FTS-indexed).
    auto q_after = fts_search(col, "hello");
    ASSERT_FALSE(q_after.has_value());

    // Reopen and verify FTS is still gone.
    col.reset();
    auto reopen_res = Collection::Open(col_path, options);
    ASSERT_TRUE(reopen_res.has_value()) << reopen_res.error().message();
    col = reopen_res.value();

    auto q_reopen = fts_search(col, "hello");
    ASSERT_FALSE(q_reopen.has_value());

    col.reset();
    FileHelper::RemoveDirectory(col_path);
  }

  // Case 3: Create → Drop → Create → Drop cycle on the same column.
  {
    FileHelper::RemoveDirectory(col_path);
    auto schema = build_schema(false);
    CollectionOptions options{false, true};
    auto col_res = Collection::CreateAndOpen(col_path, *schema, options);
    ASSERT_TRUE(col_res.has_value()) << col_res.error().message();
    auto col = std::move(col_res.value());

    std::vector<Doc> docs;
    docs.push_back(make_doc(0, "intro", "hello world"));
    docs.push_back(make_doc(1, "guide", "hello foo"));
    docs.push_back(make_doc(2, "more", "nothing here"));
    ASSERT_TRUE(col->Insert(docs).has_value());
    ASSERT_TRUE(col->Flush().ok());

    // Round 1: Create FTS index.
    auto s = col->CreateIndex("content", std::make_shared<FtsIndexParams>());
    ASSERT_TRUE(s.ok()) << s.message();
    auto q = fts_search(col, "hello");
    ASSERT_TRUE(q.has_value()) << q.error().message();
    ASSERT_EQ(q.value().size(), 2u);

    // Round 1: Drop FTS index.
    s = col->DropIndex("content");
    ASSERT_TRUE(s.ok()) << s.message();
    q = fts_search(col, "hello");
    ASSERT_FALSE(q.has_value());

    // Round 2: Re-create FTS index.
    s = col->CreateIndex("content", std::make_shared<FtsIndexParams>());
    ASSERT_TRUE(s.ok()) << s.message();
    q = fts_search(col, "hello");
    ASSERT_TRUE(q.has_value()) << q.error().message();
    ASSERT_EQ(q.value().size(), 2u);

    // Round 2: Re-drop FTS index.
    s = col->DropIndex("content");
    ASSERT_TRUE(s.ok()) << s.message();
    q = fts_search(col, "hello");
    ASSERT_FALSE(q.has_value());

    // Reopen and verify final state (no FTS).
    col.reset();
    auto reopen_res = Collection::Open(col_path, options);
    ASSERT_TRUE(reopen_res.has_value()) << reopen_res.error().message();
    col = reopen_res.value();

    q = fts_search(col, "hello");
    ASSERT_FALSE(q.has_value());

    col.reset();
    FileHelper::RemoveDirectory(col_path);
  }

  // Case 4: CreateIndex with different FtsIndexParams on a column that already
  // has an FTS index — should remove the old index and rebuild with new params.
  {
    FileHelper::RemoveDirectory(col_path);
    auto schema = build_schema(false);
    CollectionOptions options{false, true};
    auto col_res = Collection::CreateAndOpen(col_path, *schema, options);
    ASSERT_TRUE(col_res.has_value()) << col_res.error().message();
    auto col = std::move(col_res.value());

    std::vector<Doc> docs;
    docs.push_back(make_doc(0, "intro", "hello world"));
    docs.push_back(make_doc(1, "guide", "hello foo"));
    docs.push_back(make_doc(2, "more", "nothing here"));
    ASSERT_TRUE(col->Insert(docs).has_value());
    ASSERT_TRUE(col->Flush().ok());

    // Create FTS index with default params (tokenizer="standard").
    auto params_v1 = std::make_shared<FtsIndexParams>("standard");
    auto s = col->CreateIndex("content", params_v1);
    ASSERT_TRUE(s.ok()) << s.message();
    auto q = fts_search(col, "hello");
    ASSERT_TRUE(q.has_value()) << q.error().message();
    ASSERT_EQ(q.value().size(), 2u);

    // Re-create with different params: no lowercase filter, so indexing
    // preserves original case and case-mismatched queries should miss.
    auto params_v2 = std::make_shared<FtsIndexParams>(
        "standard", std::vector<std::string>{});
    ASSERT_NE(*params_v1, *params_v2);
    s = col->CreateIndex("content", params_v2);
    ASSERT_TRUE(s.ok()) << s.message();

    // Lowercase query should still hit (source text is lowercase).
    q = fts_search(col, "hello");
    ASSERT_TRUE(q.has_value()) << q.error().message();
    ASSERT_EQ(q.value().size(), 2u);

    // Uppercase query should miss — no lowercase filter means case-sensitive.
    q = fts_search(col, "HELLO");
    ASSERT_TRUE(q.has_value());
    ASSERT_EQ(q.value().size(), 0u);

    // Reopen and verify persistence.
    col.reset();
    auto reopen_res = Collection::Open(col_path, options);
    ASSERT_TRUE(reopen_res.has_value()) << reopen_res.error().message();
    col = reopen_res.value();

    q = fts_search(col, "hello");
    ASSERT_TRUE(q.has_value()) << q.error().message();
    ASSERT_EQ(q.value().size(), 2u);

    q = fts_search(col, "HELLO");
    ASSERT_TRUE(q.has_value());
    ASSERT_EQ(q.value().size(), 0u);

    col.reset();
    FileHelper::RemoveDirectory(col_path);
  }
}
