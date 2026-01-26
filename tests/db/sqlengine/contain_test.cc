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
#include <cstdlib>
#include <memory>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <gtest/gtest.h>
#include "db/common/file_helper.h"
#include "db/index/segment/segment.h"
#include "db/sqlengine/sqlengine.h"
#include "zvec/db/index_params.h"
#include "zvec/db/schema.h"
#include "zvec/db/type.h"
#include "test_helper.h"

namespace zvec::sqlengine {

static Doc create_doc(const uint64_t doc_id) {
  Doc new_doc;
  new_doc.set_pk("pk_" + std::to_string(doc_id));
  new_doc.set_doc_id(doc_id);

  auto size = doc_id % 100;
  if (size > 0) {
    std::vector<std::string> str_array;
    std::vector<int32_t> i32_array;
    std::vector<int64_t> i64_array;
    std::vector<uint32_t> u32_array;
    std::vector<uint64_t> u64_array;
    std::vector<float> fp32_array;
    std::vector<double> fp64_array;
    std::vector<bool> bool_array;

    for (uint32_t i = 1; i <= size; i++) {
      i32_array.push_back(i);
      i64_array.push_back(i);
      u32_array.push_back(i);
      u64_array.push_back(i);
      fp32_array.push_back(i);
      fp64_array.push_back(i);
      bool_array.push_back(i % 2 == 0);
      str_array.push_back("name" + std::to_string(i));
    }
    new_doc.set("i32_array", i32_array);
    new_doc.set("i64_array", i64_array);
    new_doc.set("u32_array", u32_array);
    new_doc.set("u64_array", u64_array);
    new_doc.set("fp32_array", fp32_array);
    new_doc.set("fp64_array", fp64_array);
    new_doc.set("bool_array", bool_array);
    new_doc.set("str_array", str_array);
  }
  return new_doc;
}

class ContainTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    FileHelper::RemoveDirectory(seg_path_);
    FileHelper::CreateDirectory(seg_path_);

    auto invert_params = std::make_shared<InvertIndexParams>(true);
    collection_schema_ = std::make_shared<CollectionSchema>(
        "test_collection",
        std::vector<FieldSchema::Ptr>{
            std::make_shared<FieldSchema>("str_array", DataType::ARRAY_STRING,
                                          true, nullptr),
            std::make_shared<FieldSchema>("i32_array", DataType::ARRAY_INT32,
                                          true, nullptr),
            std::make_shared<FieldSchema>("i64_array", DataType::ARRAY_INT64,
                                          true, nullptr),
            std::make_shared<FieldSchema>("u32_array", DataType::ARRAY_UINT32,
                                          true, nullptr),
            std::make_shared<FieldSchema>("u64_array", DataType::ARRAY_UINT64,
                                          true, nullptr),
            std::make_shared<FieldSchema>("fp32_array", DataType::ARRAY_FLOAT,
                                          true, nullptr),
            std::make_shared<FieldSchema>("fp64_array", DataType::ARRAY_DOUBLE,
                                          true, nullptr),
            std::make_shared<FieldSchema>("bool_array", DataType::ARRAY_BOOL,
                                          true, nullptr),

        });

    auto segment = create_segment(seg_path_, *collection_schema_);
    if (segment == nullptr) {
      LOG_ERROR("create segment failed");
      EXPECT_TRUE(segment != nullptr);
      std::exit(EXIT_FAILURE);
    }
    auto status = InsertDoc(segment, 0, 10000, &create_doc);
    if (!status.ok()) {
      LOG_ERROR("insert doc failed: %s", status.c_str());
      EXPECT_TRUE(status.ok());
      std::exit(EXIT_FAILURE);
    }
    segments_.push_back(segment);
  }

  static void TearDownTestSuite() {
    segments_.clear();
    FileHelper::RemoveDirectory(seg_path_);
  }

 protected:
  static inline std::string seg_path_ = "./test_collection";
  static inline CollectionSchema::Ptr collection_schema_;
  static inline std::vector<Segment::Ptr> segments_;
};


TEST_F(ContainTest, ContainAllInt32) {
  VectorQuery query;
  query.output_fields_ = std::vector<std::string>{};
  query.topk_ = 200;
  query.filter_ = "i32_array contain_all (";
  for (int i = 1; i <= 32; i++) {
    query.filter_ += std::to_string(i);
    if (i < 32) {
      query.filter_ += ", ";
    }
  }
  query.filter_ += ")";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error().c_str();
  auto docs = ret.value();
  EXPECT_EQ(docs.size(), 200);
  for (int j = 0, i = 32; j < (int)docs.size(); j++) {
    auto &doc = docs[j];
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(i));

    i += 1;
    while (i % 100 < 32) {
      i += 1;
    }
  }
}

TEST_F(ContainTest, ContainAllInt64) {
  VectorQuery query;
  query.output_fields_ = std::vector<std::string>{};
  query.topk_ = 200;
  query.filter_ = "i64_array contain_all (";
  for (int i = 1; i <= 32; i++) {
    query.filter_ += std::to_string(i);
    if (i < 32) {
      query.filter_ += ", ";
    }
  }
  query.filter_ += ")";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error().c_str();
  auto docs = ret.value();
  EXPECT_EQ(docs.size(), 200);
  for (int j = 0, i = 32; j < (int)docs.size(); j++) {
    auto &doc = docs[j];
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(i));

    i += 1;
    while (i % 100 < 32) {
      i += 1;
    }
  }
}

TEST_F(ContainTest, ContainAllUint32) {
  VectorQuery query;
  query.output_fields_ = std::vector<std::string>{};
  query.topk_ = 200;
  query.filter_ = "u32_array contain_all (";
  for (int i = 1; i <= 32; i++) {
    query.filter_ += std::to_string(i);
    if (i < 32) {
      query.filter_ += ", ";
    }
  }
  query.filter_ += ")";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error().c_str();
  auto docs = ret.value();
  EXPECT_EQ(docs.size(), 200);
  for (int j = 0, i = 32; j < (int)docs.size(); j++) {
    auto &doc = docs[j];
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(i));

    i += 1;
    while (i % 100 < 32) {
      i += 1;
    }
  }
}

TEST_F(ContainTest, ContainAllUint64) {
  VectorQuery query;
  query.output_fields_ = std::vector<std::string>{};
  query.topk_ = 200;
  query.filter_ = "u64_array contain_all (";
  for (int i = 1; i <= 32; i++) {
    query.filter_ += std::to_string(i);
    if (i < 32) {
      query.filter_ += ", ";
    }
  }
  query.filter_ += ")";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error().c_str();
  auto docs = ret.value();
  EXPECT_EQ(docs.size(), 200);
  for (int j = 0, i = 32; j < (int)docs.size(); j++) {
    auto &doc = docs[j];
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(i));

    i += 1;
    while (i % 100 < 32) {
      i += 1;
    }
  }
}

TEST_F(ContainTest, ContainAllFp32) {
  VectorQuery query;
  query.output_fields_ = std::vector<std::string>{};
  query.topk_ = 200;
  query.filter_ = "fp32_array contain_all (";
  for (int i = 1; i <= 32; i++) {
    query.filter_ += std::to_string(i);
    if (i < 32) {
      query.filter_ += ", ";
    }
  }
  query.filter_ += ")";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error().c_str();
  auto docs = ret.value();
  EXPECT_EQ(docs.size(), 200);
  for (int j = 0, i = 32; j < (int)docs.size(); j++) {
    auto &doc = docs[j];
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(i));

    i += 1;
    while (i % 100 < 32) {
      i += 1;
    }
  }
}

TEST_F(ContainTest, ContainAllFp64) {
  VectorQuery query;
  query.output_fields_ = std::vector<std::string>{};
  query.topk_ = 200;
  query.filter_ = "fp64_array contain_all (";
  for (int i = 1; i <= 32; i++) {
    query.filter_ += std::to_string(i);
    if (i < 32) {
      query.filter_ += ", ";
    }
  }
  query.filter_ += ")";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error().c_str();
  auto docs = ret.value();
  EXPECT_EQ(docs.size(), 200);
  for (int j = 0, i = 32; j < (int)docs.size(); j++) {
    auto &doc = docs[j];
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(i));

    i += 1;
    while (i % 100 < 32) {
      i += 1;
    }
  }
}

TEST_F(ContainTest, ContainAllString) {
  VectorQuery query;
  query.output_fields_ = std::vector<std::string>{};
  query.topk_ = 200;
  query.filter_ = "str_array contain_all (";
  for (int i = 1; i <= 32; i++) {
    query.filter_ += "'name" + std::to_string(i) + "'";
    if (i < 32) {
      query.filter_ += ", ";
    }
  }
  query.filter_ += ")";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error().c_str();
  auto docs = ret.value();
  EXPECT_EQ(docs.size(), 200);
  for (int j = 0, i = 32; j < (int)docs.size(); j++) {
    auto &doc = docs[j];
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(i));

    i += 1;
    while (i % 100 < 32) {
      i += 1;
    }
  }
}

TEST_F(ContainTest, ContainAnyInt32) {
  VectorQuery query;
  query.output_fields_ = std::vector<std::string>{};
  query.topk_ = 200;
  query.filter_ = "i32_array contain_any (98,99,100)";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error().c_str();
  auto docs = ret.value();
  EXPECT_EQ(docs.size(), 200);
  for (int j = 0, i = 98; j < (int)docs.size(); j++) {
    auto &doc = docs[j];
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(i));

    i += 1;
    while (i % 100 < 98) {
      i += 1;
    }
  }
}

TEST_F(ContainTest, ContainAnyInt64) {
  VectorQuery query;
  query.output_fields_ = std::vector<std::string>{};
  query.topk_ = 200;
  query.filter_ = "i64_array contain_any (98,99,100)";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error().c_str();
  auto docs = ret.value();
  EXPECT_EQ(docs.size(), 200);
  for (int j = 0, i = 98; j < (int)docs.size(); j++) {
    auto &doc = docs[j];
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(i));

    i += 1;
    while (i % 100 < 98) {
      i += 1;
    }
  }
}

TEST_F(ContainTest, ContainAnyUint32) {
  VectorQuery query;
  query.output_fields_ = std::vector<std::string>{};
  query.topk_ = 200;
  query.filter_ = "u32_array contain_any (98,99,100)";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error().c_str();
  auto docs = ret.value();
  EXPECT_EQ(docs.size(), 200);
  for (int j = 0, i = 98; j < (int)docs.size(); j++) {
    auto &doc = docs[j];
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(i));

    i += 1;
    while (i % 100 < 98) {
      i += 1;
    }
  }
}

TEST_F(ContainTest, ContainAnyUint64) {
  VectorQuery query;
  query.output_fields_ = std::vector<std::string>{};
  query.topk_ = 200;
  query.filter_ = "u64_array contain_any (98,99,100)";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error().c_str();
  auto docs = ret.value();
  EXPECT_EQ(docs.size(), 200);
  for (int j = 0, i = 98; j < (int)docs.size(); j++) {
    auto &doc = docs[j];
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(i));

    i += 1;
    while (i % 100 < 98) {
      i += 1;
    }
  }
}

TEST_F(ContainTest, ContainAnyFp32) {
  VectorQuery query;
  query.output_fields_ = std::vector<std::string>{};
  query.topk_ = 200;
  query.filter_ = "fp32_array contain_any (98,99,100)";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error().c_str();
  auto docs = ret.value();
  EXPECT_EQ(docs.size(), 200);
  for (int j = 0, i = 98; j < (int)docs.size(); j++) {
    auto &doc = docs[j];
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(i));

    i += 1;
    while (i % 100 < 98) {
      i += 1;
    }
  }
}

TEST_F(ContainTest, ContainAnyFp64) {
  VectorQuery query;
  query.output_fields_ = std::vector<std::string>{};
  query.topk_ = 200;
  query.filter_ = "fp64_array contain_any (98,99,100)";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error().c_str();
  auto docs = ret.value();
  EXPECT_EQ(docs.size(), 200);
  for (int j = 0, i = 98; j < (int)docs.size(); j++) {
    auto &doc = docs[j];
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(i));

    i += 1;
    while (i % 100 < 98) {
      i += 1;
    }
  }
}

TEST_F(ContainTest, ContainAnyString) {
  VectorQuery query;
  query.output_fields_ = std::vector<std::string>{};
  query.topk_ = 200;
  query.filter_ = "str_array contain_any ('name98','name99','name100')";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error().c_str();
  auto docs = ret.value();
  EXPECT_EQ(docs.size(), 200);
  for (int j = 0, i = 98; j < (int)docs.size(); j++) {
    auto &doc = docs[j];
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(i));

    i += 1;
    while (i % 100 < 98) {
      i += 1;
    }
  }
}


}  // namespace zvec::sqlengine