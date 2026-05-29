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

#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/io/file.h>
#include <zvec/ailego/utility/file_helper.h>
#include "db/common/file_helper.h"
#include "zvec/db/collection.h"
#include "zvec/db/doc.h"
#include "zvec/db/index_params.h"
#include "zvec/db/options.h"
#include "zvec/db/schema.h"
#include "zvec/db/status.h"
#include "zvec/db/type.h"

using namespace zvec;

static const std::string kTestPath = "./test_fts_query";

class FtsQueryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    FileHelper::RemoveDirectory(kTestPath);
  }
  void TearDown() override {
    FileHelper::RemoveDirectory(kTestPath);
  }

  // Create a schema with one STRING field (for forward) and one FTS field.
  static CollectionSchema::Ptr CreateFtsSchema() {
    auto schema = std::make_shared<CollectionSchema>("fts_demo");
    // A simple scalar field for forward store
    schema->add_field(std::make_shared<FieldSchema>("title", DataType::STRING));
    // FTS indexed field
    schema->add_field(
        std::make_shared<FieldSchema>("content", DataType::STRING, false,
                                      std::make_shared<FtsIndexParams>()));
    // A vector field is required for Collection to work (segment open expects
    // at least one vector field in the normal schema path).
    schema->add_field(std::make_shared<FieldSchema>(
        "vec", DataType::VECTOR_FP32, 4, false,
        std::make_shared<FlatIndexParams>(MetricType::IP)));
    return schema;
  }

  static Doc MakeDoc(uint64_t id, const std::string &title,
                     const std::string &content) {
    Doc doc;
    doc.set_pk("pk_" + std::to_string(id));
    doc.set<std::string>("title", title);
    doc.set<std::string>("content", content);
    // dummy vector
    doc.set<std::vector<float>>("vec", std::vector<float>(4, float(id + 0.1)));
    return doc;
  }
};

TEST_F(FtsQueryTest, BasicFtsQuery) {
  auto schema = CreateFtsSchema();
  CollectionOptions options;
  options.read_only_ = false;

  auto result = Collection::CreateAndOpen(kTestPath, *schema, options);
  ASSERT_TRUE(result.has_value()) << result.error().message();
  auto col = result.value();

  // Insert documents
  std::vector<Doc> docs;
  docs.push_back(MakeDoc(0, "intro", "hello world from zvec"));
  docs.push_back(MakeDoc(1, "guide", "hello foo bar"));
  docs.push_back(MakeDoc(2, "faq", "baz qux nothing here"));
  docs.push_back(MakeDoc(3, "tips", "hello hello hello world"));

  auto insert_res = col->Insert(docs);
  ASSERT_TRUE(insert_res.has_value()) << insert_res.error().message();

  // FTS query: search for "hello"
  SearchQuery vq;
  vq.target_.field_name_ = "content";
  vq.topk_ = 10;
  FtsClause fts;
  fts.query_string_ = "hello";
  vq.target_.clause_ = fts;

  auto query_res = col->Query(vq);
  ASSERT_TRUE(query_res.has_value()) << query_res.error().message();

  auto &results = query_res.value();
  // Documents 0, 1, 3 contain "hello"; document 2 does not.
  ASSERT_GE(results.size(), 2u);
  ASSERT_LE(results.size(), 3u);
}

TEST_F(FtsQueryTest, FtsQueryEmptyField) {
  auto schema = CreateFtsSchema();
  CollectionOptions options;
  options.read_only_ = false;

  auto result = Collection::CreateAndOpen(kTestPath, *schema, options);
  ASSERT_TRUE(result.has_value());
  auto col = result.value();

  SearchQuery vq;
  vq.target_.field_name_ = "";  // empty
  vq.topk_ = 10;
  FtsClause fts;
  fts.query_string_ = "hello";
  vq.target_.clause_ = fts;

  auto query_res = col->Query(vq);
  ASSERT_FALSE(query_res.has_value());
}

TEST_F(FtsQueryTest, FtsQueryNoMatch) {
  auto schema = CreateFtsSchema();
  CollectionOptions options;
  options.read_only_ = false;

  auto result = Collection::CreateAndOpen(kTestPath, *schema, options);
  ASSERT_TRUE(result.has_value());
  auto col = result.value();

  std::vector<Doc> docs;
  docs.push_back(MakeDoc(0, "intro", "hello world"));
  auto insert_res = col->Insert(docs);
  ASSERT_TRUE(insert_res.has_value());

  SearchQuery vq;
  vq.target_.field_name_ = "content";
  vq.topk_ = 10;
  FtsClause fts;
  fts.query_string_ = "nonexistent_term_xyz";
  vq.target_.clause_ = fts;

  auto query_res = col->Query(vq);
  ASSERT_TRUE(query_res.has_value());
  ASSERT_EQ(query_res.value().size(), 0u);
}

// Verify that FTS fields do NOT support add/alter/drop column operations.
// The schema change validation only allows basic numeric types [INT32..DOUBLE].
TEST_F(FtsQueryTest, FtsFieldUnsupportedAddColumn) {
  auto schema = CreateFtsSchema();
  CollectionOptions options;
  options.read_only_ = false;

  auto result = Collection::CreateAndOpen(kTestPath, *schema, options);
  ASSERT_TRUE(result.has_value());
  auto col = result.value();

  // Insert a document so the collection is non-empty
  std::vector<Doc> docs;
  docs.push_back(MakeDoc(0, "intro", "hello world"));
  auto insert_res = col->Insert(docs);
  ASSERT_TRUE(insert_res.has_value());
  ASSERT_TRUE(col->Flush().ok());

  // Attempt to add a new FTS column — should fail
  auto fts_field = std::make_shared<FieldSchema>(
      "new_fts", DataType::STRING, true, std::make_shared<FtsIndexParams>());
  auto status = col->AddColumn(fts_field, "", AddColumnOptions());
  ASSERT_FALSE(status.ok());
  ASSERT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
}

TEST_F(FtsQueryTest, FtsFieldUnsupportedDropColumn) {
  auto schema = CreateFtsSchema();
  CollectionOptions options;
  options.read_only_ = false;

  auto result = Collection::CreateAndOpen(kTestPath, *schema, options);
  ASSERT_TRUE(result.has_value());
  auto col = result.value();

  // Insert a document so the collection is non-empty
  std::vector<Doc> docs;
  docs.push_back(MakeDoc(0, "intro", "hello world"));
  auto insert_res = col->Insert(docs);
  ASSERT_TRUE(insert_res.has_value());
  ASSERT_TRUE(col->Flush().ok());

  // Attempt to drop an existing FTS column — should fail
  auto status = col->DropColumn("content");
  ASSERT_FALSE(status.ok());
  ASSERT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
}

TEST_F(FtsQueryTest, FtsFieldUnsupportedAlterColumn) {
  auto schema = CreateFtsSchema();
  CollectionOptions options;
  options.read_only_ = false;

  auto result = Collection::CreateAndOpen(kTestPath, *schema, options);
  ASSERT_TRUE(result.has_value());
  auto col = result.value();

  // Insert a document so the collection is non-empty
  std::vector<Doc> docs;
  docs.push_back(MakeDoc(0, "intro", "hello world"));
  auto insert_res = col->Insert(docs);
  ASSERT_TRUE(insert_res.has_value());
  ASSERT_TRUE(col->Flush().ok());

  // Attempt to alter (rename) the FTS column — should fail
  auto status = col->AlterColumn("content", "content_renamed", nullptr,
                                 AlterColumnOptions());
  ASSERT_FALSE(status.ok());
  ASSERT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);

  // Attempt to alter the FTS column with a new schema — should also fail
  auto new_fts_field = std::make_shared<FieldSchema>(
      "content", DataType::STRING, true, std::make_shared<FtsIndexParams>());
  status = col->AlterColumn("content", "", new_fts_field, AlterColumnOptions());
  ASSERT_FALSE(status.ok());
  ASSERT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
}
