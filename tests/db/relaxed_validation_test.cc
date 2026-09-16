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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/utility/file_helper.h>
#include <zvec/db/collection.h>
#include <zvec/db/doc.h>
#include <zvec/db/index_params.h>
#include <zvec/db/query.h>
#include <zvec/db/schema.h>
#include "db/common/constants.h"

namespace zvec {
namespace {

class RelaxedValidationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ailego::MemoryLimitPool::get_instance().init(2 * 1024ll * 1024ll * 1024ll);
    ailego::FileHelper::RemovePath(path_.c_str());
  }

  void TearDown() override {
    collection_.reset();
    ailego::FileHelper::RemovePath(path_.c_str());
  }

  CollectionSchema MakeSchema(const std::string &name = "x",
                              const std::string &field = "value") {
    CollectionSchema schema(name);
    EXPECT_TRUE(schema
                    .add_field(std::make_shared<FieldSchema>(
                        field, DataType::INT32, false))
                    .ok());
    return schema;
  }

  void Create(const CollectionSchema &schema) {
    auto result = Collection::CreateAndOpen(path_, schema, options_);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    collection_ = std::move(result).value();
  }

  void Reopen() {
    collection_.reset();
    auto result = Collection::Open(path_, options_);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    collection_ = std::move(result).value();
  }

  Doc MakeDoc(const std::string &id, int32_t value,
              const std::string &field = "value") {
    Doc doc;
    doc.set_pk(id);
    EXPECT_TRUE(doc.set<int32_t>(field, value));
    return doc;
  }

  void ExpectWrite(const Result<WriteResults> &result, size_t count) {
    ASSERT_TRUE(result.has_value()) << result.error().message();
    ASSERT_EQ(result.value().size(), count);
    for (const auto &status : result.value()) {
      ASSERT_TRUE(status.ok()) << status.message();
    }
  }

  void ExpectValue(const std::string &id, int32_t expected,
                   const std::string &field = "value") {
    auto result = collection_->fetch({id});
    ASSERT_TRUE(result.has_value()) << result.error().message();
    ASSERT_EQ(result.value().size(), 1u);
    auto found = result.value().find(id);
    ASSERT_NE(found, result.value().end());
    ASSERT_NE(found->second, nullptr);
    EXPECT_EQ(found->second->pk(), id);
    EXPECT_EQ(found->second->get<int32_t>(field), expected);
  }

  const std::string path_{"relaxed_validation_test_db"};
  CollectionOptions options_;
  Collection::Ptr collection_;
};

TEST_F(RelaxedValidationTest, Utf8IdsKeepTheirIdentityAcrossCrudAndReopen) {
  const std::string name = u8"测试 集合/v1";
  ASSERT_NO_FATAL_FAILURE(Create(MakeSchema(name)));
  const std::vector<std::string> ids = {
      "user:123",  "https://example.com/document/42",
      u8"订单-😀", std::string(1021, 'i') + u8"中",
      "doc",       " doc",
      "doc ",      " ",
      u8"café",    u8"cafe\u0301",
      "DOC"};
  std::vector<Doc> docs;
  for (size_t i = 0; i < ids.size(); ++i) {
    docs.push_back(MakeDoc(ids[i], static_cast<int32_t>(i)));
  }
  ASSERT_NO_FATAL_FAILURE(ExpectWrite(collection_->insert(docs), docs.size()));

  std::vector<Doc> updates{MakeDoc(ids[0], 100)};
  ASSERT_NO_FATAL_FAILURE(ExpectWrite(collection_->update(updates), 1));
  std::vector<Doc> upserts{MakeDoc(ids[3], 103), MakeDoc(u8"新增:文档", 200)};
  ASSERT_NO_FATAL_FAILURE(ExpectWrite(collection_->upsert(upserts), 2));
  ASSERT_NO_FATAL_FAILURE(ExpectWrite(collection_->delete_({ids[1]}), 1));

  auto flush_status = collection_->flush();
  ASSERT_TRUE(flush_status.ok()) << flush_status.message();
  ASSERT_NO_FATAL_FAILURE(Reopen());
  auto schema = collection_->schema();
  ASSERT_TRUE(schema.has_value()) << schema.error().message();
  EXPECT_EQ(schema.value().name(), name);
  for (size_t i = 0; i < ids.size(); ++i) {
    if (i == 1) {
      continue;
    }
    int32_t expected = static_cast<int32_t>(i);
    if (i == 0) expected = 100;
    if (i == 3) expected = 103;
    ASSERT_NO_FATAL_FAILURE(ExpectValue(ids[i], expected));
  }
  ASSERT_NO_FATAL_FAILURE(ExpectValue(u8"新增:文档", 200));
  auto deleted = collection_->fetch({ids[1]});
  ASSERT_TRUE(deleted.has_value()) << deleted.error().message();
  ASSERT_EQ(deleted.value().size(), 1u);
  EXPECT_EQ(deleted.value().at(ids[1]), nullptr);
}

TEST_F(RelaxedValidationTest, ShortAndMaximumLengthCollectionNamesPersist) {
  for (const auto &name : std::vector<std::string>{
           "x", "xy", u8"集", std::string(253, 'n') + u8"集"}) {
    SCOPED_TRACE(name);
    ASSERT_NO_FATAL_FAILURE(Create(MakeSchema(name)));
    std::vector<Doc> docs{MakeDoc("id", 1)};
    ASSERT_NO_FATAL_FAILURE(ExpectWrite(collection_->insert(docs), 1));
    auto status = collection_->flush();
    ASSERT_TRUE(status.ok()) << status.message();
    ASSERT_NO_FATAL_FAILURE(Reopen());
    auto schema = collection_->schema();
    ASSERT_TRUE(schema.has_value()) << schema.error().message();
    EXPECT_EQ(schema.value().name(), name);
    ASSERT_NO_FATAL_FAILURE(ExpectValue("id", 1));
    status = collection_->destroy();
    ASSERT_TRUE(status.ok()) << status.message();
    collection_.reset();
  }
}

TEST_F(RelaxedValidationTest, MaximumLengthFieldsSupportIndexesAndFilters) {
  const std::string scalar = "s" + std::string(63, 'a');
  const std::string second_scalar = scalar.substr(0, 63) + "b";
  const std::string vector = "v" + std::string(63, 'b');
  auto schema = MakeSchema("x", scalar);
  ASSERT_TRUE(schema
                  .add_field(std::make_shared<FieldSchema>(
                      second_scalar, DataType::INT32, false))
                  .ok());
  ASSERT_TRUE(schema
                  .add_field(std::make_shared<FieldSchema>(
                      vector, DataType::VECTOR_FP32, 4, false,
                      std::make_shared<FlatIndexParams>(MetricType::L2)))
                  .ok());
  ASSERT_NO_FATAL_FAILURE(Create(schema));
  const std::vector<float> values{1.0f, 2.0f, 3.0f, 4.0f};
  Doc doc = MakeDoc(u8"文档:1", 42, scalar);
  ASSERT_TRUE(doc.set<int32_t>(second_scalar, 7));
  ASSERT_TRUE(doc.set<std::vector<float>>(vector, values));
  std::vector<Doc> docs{doc};
  ASSERT_NO_FATAL_FAILURE(ExpectWrite(collection_->insert(docs), 1));
  auto status = collection_->flush();
  ASSERT_TRUE(status.ok()) << status.message();
  status =
      collection_->create_index(scalar, std::make_shared<InvertIndexParams>());
  ASSERT_TRUE(status.ok()) << status.message();
  status = collection_->create_index(
      vector, std::make_shared<HnswIndexParams>(MetricType::L2));
  ASSERT_TRUE(status.ok()) << status.message();
  status = collection_->create_index(second_scalar,
                                     std::make_shared<InvertIndexParams>());
  ASSERT_TRUE(status.ok()) << status.message();
  ASSERT_NO_FATAL_FAILURE(Reopen());

  // Fetch supplies the query vector, covering lookup by a newly allowed ID.
  auto fetched = collection_->fetch({doc.pk()});
  ASSERT_TRUE(fetched.has_value()) << fetched.error().message();
  ASSERT_EQ(fetched.value().size(), 1u);
  const auto stored_vector =
      fetched.value().at(doc.pk())->get<std::vector<float>>(vector);
  ASSERT_TRUE(stored_vector.has_value());
  EXPECT_EQ(stored_vector.value(), values);
  SearchQuery query;
  query.topk_ = 1;
  query.target_.field_name_ = vector;
  query.target_.set_vector(
      std::string(reinterpret_cast<const char *>(stored_vector->data()),
                  stored_vector->size() * sizeof(float)));
  query.filter_ = scalar + " = 42 AND " + second_scalar + " = 7";
  query.output_fields_ = std::vector<std::string>{scalar, second_scalar};
  auto matches = collection_->query(query);
  ASSERT_TRUE(matches.has_value()) << matches.error().message();
  ASSERT_EQ(matches.value().size(), 1u);
  EXPECT_EQ(matches.value()[0]->pk(), doc.pk());
  EXPECT_EQ(matches.value()[0]->get<int32_t>(scalar), 42);
  EXPECT_EQ(matches.value()[0]->get<int32_t>(second_scalar), 7);
}

TEST_F(RelaxedValidationTest, MaximumLengthFieldsPersistWithBufferedStorage) {
  options_.enable_mmap_ = false;
  const std::string field(64, 'f');
  ASSERT_NO_FATAL_FAILURE(Create(MakeSchema("buffered", field)));
  std::vector<Doc> docs{MakeDoc("doc", 42, field)};
  ASSERT_NO_FATAL_FAILURE(ExpectWrite(collection_->insert(docs), 1));
  const auto status = collection_->flush();
  ASSERT_TRUE(status.ok()) << status.message();
  ASSERT_NO_FATAL_FAILURE(Reopen());
  EXPECT_TRUE(collection_->schema().value().has_field(field));
  ASSERT_NO_FATAL_FAILURE(ExpectValue("doc", 42, field));
}

TEST_F(RelaxedValidationTest, InvalidRenameLeavesSchemaAndDataUnchanged) {
  ASSERT_NO_FATAL_FAILURE(Create(MakeSchema()));
  std::vector<Doc> docs{MakeDoc("id", 42)};
  ASSERT_NO_FATAL_FAILURE(ExpectWrite(collection_->insert(docs), 1));
  const auto before = collection_->schema().value();
  for (const auto &name : std::vector<std::string>{
           "user name", "../value", u8"字段", std::string(65, 'f')}) {
    SCOPED_TRACE(name);
    auto status = collection_->alter_column("value", name);
    ASSERT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(status.message().find("Invalid schema:"), 0u);
    EXPECT_EQ(status.message().find("offset"), std::string::npos);
    EXPECT_EQ(collection_->schema().value(), before);
    ASSERT_NO_FATAL_FAILURE(ExpectValue("id", 42));
  }
  ASSERT_NO_FATAL_FAILURE(Reopen());
  EXPECT_EQ(collection_->schema().value(), before);
  ASSERT_NO_FATAL_FAILURE(ExpectValue("id", 42));

  const std::string renamed(64, 'r');
  auto status = collection_->alter_column("value", renamed);
  ASSERT_TRUE(status.ok()) << status.message();
  ASSERT_NO_FATAL_FAILURE(Reopen());
  EXPECT_FALSE(collection_->schema().value().has_field("value"));
  EXPECT_TRUE(collection_->schema().value().has_field(renamed));
  ASSERT_NO_FATAL_FAILURE(ExpectValue("id", 42, renamed));
}

TEST_F(RelaxedValidationTest, InvalidIdRejectsWholeBatchBeforeWriting) {
  ASSERT_NO_FATAL_FAILURE(Create(MakeSchema()));
  std::vector<Doc> initial{MakeDoc("existing", 1)};
  ASSERT_NO_FATAL_FAILURE(ExpectWrite(collection_->insert(initial), 1));
  for (int operation = 0; operation < 3; ++operation) {
    SCOPED_TRACE(operation);
    const std::string first_id = operation == 0 ? "new:id" : "existing";
    std::vector<Doc> batch{MakeDoc(first_id, 99),
                           MakeDoc(std::string("bad\0id", 6), 100)};
    auto result = operation == 0   ? collection_->insert(batch)
                  : operation == 1 ? collection_->update(batch)
                                   : collection_->upsert(batch);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code(), StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(result.error().message().find("Invalid doc:"), 0u);
    EXPECT_NE(result.error().message().find("null character"),
              std::string::npos);
    EXPECT_NE(result.error().message().find("id[bad\\0id]"), std::string::npos);
    EXPECT_EQ(result.error().message().find("offset"), std::string::npos);
    ASSERT_NO_FATAL_FAILURE(ExpectValue("existing", 1));
    auto missing = collection_->fetch({"new:id"});
    ASSERT_TRUE(missing.has_value()) << missing.error().message();
    ASSERT_EQ(missing.value().size(), 1u);
    EXPECT_EQ(missing.value().at("new:id"), nullptr);
  }
  ASSERT_NO_FATAL_FAILURE(Reopen());
  ASSERT_NO_FATAL_FAILURE(ExpectValue("existing", 1));
  EXPECT_EQ(collection_->stats().value().doc_count, 1u);
}

TEST_F(RelaxedValidationTest, FetchAndDeleteKeepLookupSemantics) {
  ASSERT_NO_FATAL_FAILURE(Create(MakeSchema()));
  // These are invalid for a new document, but lookup must retain its existing
  // missing-key behavior rather than introducing input validation errors.
  const std::vector<std::string> absent_ids{"", std::string("bad\0id", 6),
                                            std::string(1025, 'x'), "\xff"};
  auto fetched = collection_->fetch(absent_ids);
  ASSERT_TRUE(fetched.has_value()) << fetched.error().message();
  ASSERT_EQ(fetched.value().size(), absent_ids.size());
  for (const auto &id : absent_ids) {
    EXPECT_EQ(fetched.value().at(id), nullptr);
  }
  auto deleted = collection_->delete_(absent_ids);
  ASSERT_TRUE(deleted.has_value()) << deleted.error().message();
  ASSERT_EQ(deleted.value().size(), absent_ids.size());
  for (const auto &status : deleted.value()) {
    EXPECT_EQ(status.code(), StatusCode::NOT_FOUND);
  }
}

TEST_F(RelaxedValidationTest, ReservedNamesAndDuplicatesFailBeforeCreation) {
  for (const std::string name :
       {"_zvec_uid_", "_zvec_g_doc_id_", "_zvec_row_id_", "_zvec_score",
        "_zvec_group_id"}) {
    SCOPED_TRACE(name);
    auto result =
        Collection::CreateAndOpen(path_, MakeSchema("x", name), options_);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code(), StatusCode::INVALID_ARGUMENT);
    EXPECT_NE(result.error().message().find("is reserved"), std::string::npos);
    EXPECT_FALSE(ailego::FileHelper::IsExist(path_.c_str()));
  }
  CollectionSchema duplicate(
      "x", {std::make_shared<FieldSchema>("value", DataType::INT32),
            std::make_shared<FieldSchema>("value", DataType::INT64)});
  auto result = Collection::CreateAndOpen(path_, duplicate, options_);
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_NE(result.error().message().find("duplicate field name"),
            std::string::npos);
  EXPECT_FALSE(ailego::FileHelper::IsExist(path_.c_str()));
}

TEST_F(RelaxedValidationTest, ReservedDdlTargetsLeaveTheCollectionUnchanged) {
  ASSERT_NO_FATAL_FAILURE(Create(MakeSchema()));
  std::vector<Doc> docs{MakeDoc("id", 42)};
  ASSERT_NO_FATAL_FAILURE(ExpectWrite(collection_->insert(docs), 1));
  const auto before = collection_->schema().value();
  for (const std::string name :
       {"_zvec_uid_", "_zvec_g_doc_id_", "_zvec_row_id_", "_zvec_score",
        "_zvec_group_id"}) {
    SCOPED_TRACE(name);
    auto status = collection_->add_column(
        std::make_shared<FieldSchema>(name, DataType::INT32, true), "");
    EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
    EXPECT_NE(status.message().find("is reserved"), std::string::npos);
    status = collection_->alter_column("value", name);
    EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
    EXPECT_NE(status.message().find("is reserved"), std::string::npos);
    status = collection_->alter_column(
        "value", "", std::make_shared<FieldSchema>(name, DataType::INT32));
    EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
    EXPECT_NE(status.message().find("is reserved"), std::string::npos);
    EXPECT_EQ(collection_->schema().value(), before);
    ASSERT_NO_FATAL_FAILURE(ExpectValue("id", 42));
  }
  ASSERT_NO_FATAL_FAILURE(Reopen());
  EXPECT_EQ(collection_->schema().value(), before);
  ASSERT_NO_FATAL_FAILURE(ExpectValue("id", 42));
}

TEST_F(RelaxedValidationTest, DdlRetainsFieldCountInvariants) {
  CollectionSchema schema("x");
  for (uint32_t i = 0; i < kMaxScalarFieldSize; ++i) {
    ASSERT_TRUE(schema
                    .add_field(std::make_shared<FieldSchema>(
                        "f" + std::to_string(i), DataType::INT32, true))
                    .ok());
  }
  ASSERT_NO_FATAL_FAILURE(Create(schema));
  auto status = collection_->add_column(
      std::make_shared<FieldSchema>("excess", DataType::INT32, true), "");
  EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_NE(status.message().find("1024 scalar fields"), std::string::npos);
  EXPECT_EQ(collection_->schema().value(), schema);
  EXPECT_FALSE(collection_->schema().value().has_field("excess"));
  status = collection_->destroy();
  ASSERT_TRUE(status.ok()) << status.message();
  collection_.reset();

  ASSERT_NO_FATAL_FAILURE(Create(MakeSchema()));
  std::vector<Doc> docs{MakeDoc("id", 42)};
  ASSERT_NO_FATAL_FAILURE(ExpectWrite(collection_->insert(docs), 1));
  status = collection_->drop_column("value");
  EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_NE(status.message().find("last field"), std::string::npos);
  ASSERT_NO_FATAL_FAILURE(ExpectValue("id", 42));
  ASSERT_NO_FATAL_FAILURE(Reopen());
  EXPECT_TRUE(collection_->schema().value().has_field("value"));
  ASSERT_NO_FATAL_FAILURE(ExpectValue("id", 42));
}

TEST_F(RelaxedValidationTest, DdlSnapshotsCallerOwnedFieldSchemas) {
  ASSERT_NO_FATAL_FAILURE(Create(MakeSchema()));
  std::vector<Doc> initial{MakeDoc("original", 1)};
  ASSERT_NO_FATAL_FAILURE(ExpectWrite(collection_->insert(initial), 1));
  auto added = std::make_shared<FieldSchema>("extra", DataType::INT32, true);
  auto status = collection_->add_column(added, "");
  ASSERT_TRUE(status.ok()) << status.message();
  added->set_name("bad name");
  added->set_data_type(DataType::STRING);
  added->set_nullable(false);

  auto current = collection_->schema().value();
  ASSERT_TRUE(current.has_field("extra"));
  EXPECT_EQ(current.get_field("extra")->data_type(), DataType::INT32);
  EXPECT_TRUE(current.get_field("extra")->nullable());
  EXPECT_FALSE(current.has_field("bad name"));
  Doc doc = MakeDoc("new", 2);
  ASSERT_TRUE(doc.set<int32_t>("extra", 7));
  std::vector<Doc> docs{doc};
  ASSERT_NO_FATAL_FAILURE(ExpectWrite(collection_->insert(docs), 1));

  auto altered = std::make_shared<FieldSchema>("extra", DataType::INT64, true);
  status = collection_->alter_column("extra", "", altered);
  ASSERT_TRUE(status.ok()) << status.message();
  altered->set_name("another bad name");
  altered->set_data_type(DataType::STRING);
  current = collection_->schema().value();
  ASSERT_TRUE(current.has_field("extra"));
  EXPECT_EQ(current.get_field("extra")->data_type(), DataType::INT64);
  EXPECT_FALSE(current.has_field("another bad name"));
  ASSERT_NO_FATAL_FAILURE(Reopen());
  ASSERT_NO_FATAL_FAILURE(ExpectValue("original", 1));
  ASSERT_NO_FATAL_FAILURE(ExpectValue("new", 2));
  auto fetched = collection_->fetch({"new"});
  ASSERT_TRUE(fetched.has_value()) << fetched.error().message();
  ASSERT_NE(fetched.value().at("new"), nullptr);
  EXPECT_EQ(fetched.value().at("new")->get<int64_t>("extra"), 7);
}

}  // namespace
}  // namespace zvec
