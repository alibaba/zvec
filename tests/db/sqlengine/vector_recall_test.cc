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
// limitations under the License

#include <cstdint>
#include <memory>
#include <gtest/gtest.h>
#include "db/sqlengine/sqlengine.h"
#include "recall_base.h"

namespace zvec::sqlengine {

class VectorRecallTest : public RecallTest {};

TEST_F(VectorRecallTest, Basic) {
  VectorQuery query;
  query.output_fields_ = {"id", "name", "age"};
  query.topk_ = 200;
  std::vector<float> feature(4, 0.0);
  query.query_vector_.assign((const char *)feature.data(),
                             feature.size() * sizeof(float));
  query.field_name_ = "dense";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  if (!ret) {
    LOG_ERROR("execute failed: [%s]", ret.error().c_str());
  }
  ASSERT_TRUE(ret.has_value());
  auto docs = ret.value();
  EXPECT_EQ(docs.size(), query.topk_);
  for (int i = 0; i < query.topk_; i++) {
    auto &doc = docs[i];
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(i));
    auto age = doc->get<int32_t>("age");
    EXPECT_EQ(age.value(), i % 100);
    auto name = doc->get<std::string>("name");
    ASSERT_TRUE(name);
    EXPECT_EQ(name.value(), "user_" + std::to_string(i % 100));
    EXPECT_FLOAT_EQ(doc->score(), (float)i * i * 4);
  }
}

TEST_F(VectorRecallTest, HybridInvertFilter) {
  VectorQuery query;
  query.output_fields_ = {"id", "name", "age"};
  query.filter_ = "invert_id >= 1";
  query.topk_ = 200;
  std::vector<float> feature(4, 0.0);
  query.query_vector_.assign((const char *)feature.data(),
                             feature.size() * sizeof(float));
  query.field_name_ = "dense";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  if (!ret) {
    LOG_ERROR("execute failed: [%s]", ret.error().c_str());
  }
  ASSERT_TRUE(ret.has_value());
  auto docs = ret.value();
  EXPECT_EQ(docs.size(), query.topk_);
  for (int j = 0; j < query.topk_; j++) {
    auto &doc = docs[j];
    int i = j + 1;
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(i));
    auto age = doc->get<int32_t>("age");
    EXPECT_EQ(age.value(), i % 100);
    auto name = doc->get<std::string>("name");
    ASSERT_TRUE(name);
    EXPECT_EQ(name.value(), "user_" + std::to_string(i % 100));
    EXPECT_FLOAT_EQ(doc->score(), (float)i * i * 4);
  }
}

TEST_F(VectorRecallTest, HybridInvertFilterBfByKeys) {
  VectorQuery query;
  query.output_fields_ = {"id", "name", "age"};
  query.filter_ = "invert_id < 199";
  query.topk_ = 199;
  std::vector<float> feature(4, 0.0);
  query.query_vector_.assign((const char *)feature.data(),
                             feature.size() * sizeof(float));
  query.field_name_ = "dense";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  if (!ret) {
    LOG_ERROR("execute failed: [%s]", ret.error().c_str());
  }
  ASSERT_TRUE(ret.has_value());
  auto docs = ret.value();
  EXPECT_EQ(docs.size(), query.topk_);
  for (int i = 0; i < query.topk_; i++) {
    auto &doc = docs[i];
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(i));
    auto age = doc->get<int32_t>("age");
    EXPECT_EQ(age.value(), i % 100);
    auto name = doc->get<std::string>("name");
    ASSERT_TRUE(name);
    EXPECT_EQ(name.value(), "user_" + std::to_string(i % 100));
    EXPECT_FLOAT_EQ(doc->score(), (float)i * i * 4);
  }
}

TEST_F(VectorRecallTest, HybridForwardFilter) {
  VectorQuery query;
  query.output_fields_ = {"id", "name", "age"};
  query.filter_ = "id >= 1";
  query.topk_ = 200;
  std::vector<float> feature(4, 0.0);
  query.query_vector_.assign((const char *)feature.data(),
                             feature.size() * sizeof(float));
  query.field_name_ = "dense";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  if (!ret) {
    LOG_ERROR("execute failed: [%s]", ret.error().c_str());
  }
  ASSERT_TRUE(ret.has_value());
  auto docs = ret.value();
  EXPECT_EQ(docs.size(), query.topk_);
  for (int j = 0; j < query.topk_; j++) {
    auto &doc = docs[j];
    int i = j + 1;
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(i));
    auto age = doc->get<int32_t>("age");
    EXPECT_EQ(age.value(), i % 100);
    auto name = doc->get<std::string>("name");
    ASSERT_TRUE(name);
    EXPECT_EQ(name.value(), "user_" + std::to_string(i % 100));
    EXPECT_FLOAT_EQ(doc->score(), (float)i * i * 4);
  }
}

TEST_F(VectorRecallTest, HybridInvertForwardFilter) {
  VectorQuery query;
  query.output_fields_ = {"name", "age"};
  query.filter_ = "invert_id >= 1 and id <= 100";
  query.topk_ = 200;
  std::vector<float> feature(4, 0.0);
  query.query_vector_.assign((const char *)feature.data(),
                             feature.size() * sizeof(float));
  query.field_name_ = "dense";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  if (!ret) {
    LOG_ERROR("execute failed: [%s]", ret.error().c_str());
  }
  ASSERT_TRUE(ret.has_value());
  auto docs = ret.value();
  EXPECT_EQ(docs.size(), 100);
  for (size_t j = 0; j < docs.size(); j++) {
    auto &doc = docs[j];
    int doc_id = j + 1;
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(doc_id));
    auto age = doc->get<int32_t>("age");
    EXPECT_EQ(age.value(), doc_id % 100);
    auto name = doc->get<std::string>("name");
    ASSERT_TRUE(name);
    EXPECT_EQ(name.value(), "user_" + std::to_string(doc_id % 100));
    EXPECT_FLOAT_EQ(doc->score(), (float)doc_id * doc_id * 4);
  }
}

TEST_F(VectorRecallTest, Sparse) {
  VectorQuery query;
  query.output_fields_ = {"id", "name", "age"};
  query.topk_ = 200;
  std::vector<float> feature(4, 1.0);
  std::vector<uint32_t> indices{0, 1, 2, 3};
  query.query_sparse_indices_.assign((const char *)indices.data(),
                                     indices.size() * sizeof(uint32_t));
  query.query_sparse_values_.assign((const char *)feature.data(),
                                    feature.size() * sizeof(float));
  query.field_name_ = "sparse";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  if (!ret) {
    LOG_ERROR("execute failed: [%s]", ret.error().c_str());
  }
  ASSERT_TRUE(ret.has_value());
  auto docs = ret.value();
  EXPECT_EQ(docs.size(), query.topk_);

  int doc_id = 9999;
  for (size_t j = 0; j < docs.size(); j++) {
    auto &doc = docs[j];
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(doc_id));
    auto age = doc->get<int32_t>("age");
    EXPECT_EQ(age.value(), doc_id % 100);
    auto name = doc->get<std::string>("name");
    ASSERT_TRUE(name);
    EXPECT_EQ(name.value(), "user_" + std::to_string(doc_id % 100));
    EXPECT_FLOAT_EQ(doc->score(), (float)doc_id * 4);
    doc_id--;
    while (doc_id % 100 <= 3) {
      doc_id--;
    }
  }
}

}  // namespace zvec::sqlengine