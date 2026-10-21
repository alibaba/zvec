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
#include <set>
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

  static CollectionSchema::Ptr CreateNGramFtsSchema() {
    auto schema = std::make_shared<CollectionSchema>("ngram_fts_demo");
    schema->add_field(std::make_shared<FieldSchema>("title", DataType::STRING));
    auto fts_params = std::make_shared<FtsIndexParams>(
        "ngram", std::vector<std::string>{},
        R"({"ngram_min":2,"ngram_max":2,"token_chars":["letter"]})");
    schema->add_field(std::make_shared<FieldSchema>(
        "content", DataType::STRING, false, std::move(fts_params)));
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

  auto insert_res = col->insert(docs);
  ASSERT_TRUE(insert_res.has_value()) << insert_res.error().message();

  // FTS query: search for "hello"
  SearchQuery vq;
  vq.target_.field_name_ = "content";
  vq.topk_ = 10;
  FtsClause fts;
  fts.query_string_ = "hello";
  vq.target_.clause_ = fts;

  auto query_res = col->query(vq);
  ASSERT_TRUE(query_res.has_value()) << query_res.error().message();

  auto &results = query_res.value();
  // Documents 0, 1, 3 contain "hello"; document 2 does not.
  ASSERT_GE(results.size(), 2u);
  ASSERT_LE(results.size(), 3u);
}

TEST_F(FtsQueryTest, NGramTokenizerEndToEnd) {
  auto schema = CreateNGramFtsSchema();
  CollectionOptions options;
  options.read_only_ = false;

  auto result = Collection::CreateAndOpen(kTestPath, *schema, options);
  ASSERT_TRUE(result.has_value()) << result.error().message();
  auto col = result.value();

  std::vector<Doc> docs;
  docs.push_back(MakeDoc(0, "chinese",
                         "\xE4\xB8\xAD\xE6\x96\x87\xE5\x88\x86\xE8\xAF\x8D"));
  docs.push_back(MakeDoc(1, "english", "english tokenizer"));
  auto insert_result = col->insert(docs);
  ASSERT_TRUE(insert_result.has_value()) << insert_result.error().message();

  SearchQuery query;
  query.target_.field_name_ = "content";
  query.topk_ = 10;
  FtsClause fts;
  fts.query_string_ = "\xE4\xB8\xAD\xE6\x96\x87";
  query.target_.clause_ = fts;

  auto query_result = col->query(query);
  ASSERT_TRUE(query_result.has_value()) << query_result.error().message();
  ASSERT_EQ(query_result.value().size(), 1u);
}

TEST_F(FtsQueryTest, NGramConfigErrorIsReturnedByCreateCollection) {
  auto schema = CreateNGramFtsSchema();
  auto field = schema->get_field("content");
  auto params =
      std::dynamic_pointer_cast<FtsIndexParams>(field->index_params());
  ASSERT_NE(params, nullptr);
  params->set_extra_params(R"({"ngram_min":1,"ngram_max":3})");

  CollectionOptions options;
  options.read_only_ = false;
  auto result = Collection::CreateAndOpen(kTestPath, *schema, options);

  ASSERT_FALSE(result.has_value());
  EXPECT_NE(result.error().message().find("ngram_max - ngram_min must be <= 1"),
            std::string::npos);
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

  auto query_res = col->query(vq);
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
  auto insert_res = col->insert(docs);
  ASSERT_TRUE(insert_res.has_value());

  SearchQuery vq;
  vq.target_.field_name_ = "content";
  vq.topk_ = 10;
  FtsClause fts;
  fts.query_string_ = "nonexistent_term_xyz";
  vq.target_.clause_ = fts;

  auto query_res = col->query(vq);
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
  auto insert_res = col->insert(docs);
  ASSERT_TRUE(insert_res.has_value());
  ASSERT_TRUE(col->flush().ok());

  // Attempt to add a new FTS column — should fail
  auto fts_field = std::make_shared<FieldSchema>(
      "new_fts", DataType::STRING, true, std::make_shared<FtsIndexParams>());
  auto status = col->add_column(fts_field, "", AddColumnOptions());
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
  auto insert_res = col->insert(docs);
  ASSERT_TRUE(insert_res.has_value());
  ASSERT_TRUE(col->flush().ok());

  // Attempt to drop an existing FTS column — should fail
  auto status = col->drop_column("content");
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
  auto insert_res = col->insert(docs);
  ASSERT_TRUE(insert_res.has_value());
  ASSERT_TRUE(col->flush().ok());

  // Attempt to alter (rename) the FTS column — should fail
  auto status = col->alter_column("content", "content_renamed", nullptr,
                                  AlterColumnOptions());
  ASSERT_FALSE(status.ok());
  ASSERT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);

  // Attempt to alter the FTS column with a new schema — should also fail
  auto new_fts_field = std::make_shared<FieldSchema>(
      "content", DataType::STRING, true, std::make_shared<FtsIndexParams>());
  status =
      col->alter_column("content", "", new_fts_field, AlterColumnOptions());
  ASSERT_FALSE(status.ok());
  ASSERT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
}


TEST_F(FtsQueryTest, CodeTokenizerRetrievalAndRebuild) {
#ifdef __ANDROID__
  GTEST_SKIP() << "Skipped on Android: emulator filesystem lacks hardlink "
                  "support (needed by RocksDB checkpoint)";
#endif
  auto schema = CreateFtsSchema();
  CollectionOptions options;
  options.read_only_ = false;
  auto opened = Collection::CreateAndOpen(kTestPath, *schema, options);
  ASSERT_TRUE(opened.has_value()) << opened.error().message();
  auto col = std::move(opened.value());
  std::vector<Doc> docs = {MakeDoc(0, "", "getRequestTime"),
                           MakeDoc(1, "", "get_request_time"),
                           MakeDoc(2, "", "get_requestTime"),
                           MakeDoc(3, "", "get"),
                           MakeDoc(4, "", "request time"),
                           MakeDoc(5, "", "getRequestTimeout"),
                           MakeDoc(6, "", "getaway"),
                           MakeDoc(7, "", "HTTPRequest 获取时间")};
  ASSERT_TRUE(col->insert(docs).has_value());
  auto params = std::make_shared<FtsIndexParams>(
      "code", std::vector<std::string>{}, R"({"sub_tokenizer":"standard"})");
  auto status = col->create_index("content", params);
  ASSERT_TRUE(status.ok()) << status.message();

  auto search = [&](const std::string &text, const std::string &op,
                    bool structured) {
    SearchQuery query;
    query.target_.field_name_ = "content";
    query.topk_ = 20;
    FtsClause clause;
    if (structured)
      clause.query_string_ = text;
    else
      clause.match_string_ = text;
    query.target_.clause_ = clause;
    auto qp = std::make_shared<FtsQueryParams>();
    qp->set_default_operator(op);
    query.target_.query_params_ = qp;
    return col->query(query);
  };
  auto check = [&](const std::string &text, const std::string &op,
                   const std::set<std::string> &expected) {
    for (bool structured : {false, true}) {
      auto result = search(text, op, structured);
      ASSERT_TRUE(result.has_value()) << result.error().message();
      std::set<std::string> actual;
      for (const auto &doc : result.value()) actual.insert(doc->pk());
      EXPECT_EQ(actual, expected) << text << " " << op;
    }
  };
  check("request time", "and", {"pk_0", "pk_1", "pk_2", "pk_4"});
  check("getRequestTime", "and", {"pk_0"});
  check("get_request_time", "and", {"pk_1"});
  check("getRequestTime", "or",
        {"pk_0", "pk_1", "pk_2", "pk_3", "pk_4", "pk_5", "pk_7"});
  check("http request", "and", {"pk_7"});
  check("get", "or", {"pk_0", "pk_1", "pk_2", "pk_3", "pk_5"});
  auto phrase = search("\"request time\"", "or", true);
  ASSERT_TRUE(phrase.has_value()) << phrase.error().message();
  std::set<std::string> phrase_pks;
  for (const auto &doc : phrase.value()) phrase_pks.insert(doc->pk());
  EXPECT_EQ(phrase_pks,
            (std::set<std::string>{"pk_0", "pk_1", "pk_2", "pk_4"}));
  ASSERT_TRUE(search("\"request time\"", "and", false).has_value());

  ASSERT_TRUE(col->flush().ok());
  col.reset();
  opened = Collection::Open(kTestPath, options);
  ASSERT_TRUE(opened.has_value()) << opened.error().message();
  col = std::move(opened.value());
  auto persisted_schema = col->schema();
  ASSERT_TRUE(persisted_schema.has_value());
  auto persisted_params = std::dynamic_pointer_cast<FtsIndexParams>(
      persisted_schema.value().get_field("content")->index_params());
  ASSERT_NE(persisted_params, nullptr);
  EXPECT_EQ(persisted_params->tokenizer_name(), "code");
  EXPECT_TRUE(persisted_params->filters().empty());
  EXPECT_EQ(persisted_params->extra_params(), params->extra_params());
  check("request time", "and", {"pk_0", "pk_1", "pk_2", "pk_4"});
  std::vector<Doc> updates{MakeDoc(0, "", "unrelated")};
  ASSERT_TRUE(col->update(updates).has_value());
  ASSERT_TRUE(col->delete_({"pk_1"}).has_value());
  check("request time", "and", {"pk_2", "pk_4"});
  ASSERT_TRUE(col->optimize().ok());
  check("request time", "and", {"pk_2", "pk_4"});
}
