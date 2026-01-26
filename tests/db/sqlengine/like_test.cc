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
#include <iostream>
#include <memory>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <gtest/gtest.h>
#include "db/common/file_helper.h"
#include "db/index/common/version_manager.h"
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

  auto name = std::string("user-");
  if (doc_id >= 5000 && doc_id < 8000) {
    name += "%";
  } else if (doc_id >= 8000) {
    name += '_';
  }
  name += std::to_string(doc_id % 100);
  new_doc.set<std::string>("name", name);
  new_doc.set<std::string>("invert_name", name);
  new_doc.set<std::string>("extended_invert_name", name);
  return new_doc;
}

class LikeTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    FileHelper::RemoveDirectory(seg_path_);
    FileHelper::CreateDirectory(seg_path_);

    auto invert_params = std::make_shared<InvertIndexParams>(true);
    collection_schema_ = std::make_shared<CollectionSchema>(
        "test_collection",
        std::vector<FieldSchema::Ptr>{
            std::make_shared<FieldSchema>("name", DataType::STRING, false,
                                          nullptr),
            std::make_shared<FieldSchema>(
                "invert_name", DataType::STRING, false,
                std::make_shared<InvertIndexParams>(false, false)),
            std::make_shared<FieldSchema>(
                "extended_invert_name", DataType::STRING, false,
                std::make_shared<InvertIndexParams>(false, true)),
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


TEST_F(LikeTest, ForwardLikeAll) {
  VectorQuery query;
  query.output_fields_ = {"name"};
  query.topk_ = 200;
  query.filter_ = "name like '%'";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error();
  auto docs = std::move(ret.value());
  for (size_t i = 0; i < docs.size(); i++) {
    auto doc = docs[i];
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(i));
  }
}

TEST_F(LikeTest, InvertLikeAll) {
  VectorQuery query;
  query.output_fields_ = {"name"};
  query.topk_ = 200;
  query.filter_ = "invert_name like '%'";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error();
  auto docs = std::move(ret.value());
  for (size_t i = 0; i < docs.size(); i++) {
    auto doc = docs[i];
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(i));
  }
}

TEST_F(LikeTest, ForwardPrefixLike) {
  VectorQuery query;
  query.output_fields_ = {"name"};
  query.topk_ = 200;
  query.filter_ = "name like 'user-22%'";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error();
  auto docs = std::move(ret.value());
  for (size_t i = 0; i < docs.size(); i++) {
    auto doc = docs[i];
    int doc_id = i * 100 + 22;
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(doc_id));
  }
}

TEST_F(LikeTest, InvertPrefixLike) {
  VectorQuery query;
  query.output_fields_ = {"name"};
  query.topk_ = 200;
  query.filter_ = "invert_name like 'user-22%'";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error();
  auto docs = std::move(ret.value());
  for (size_t i = 0; i < docs.size(); i++) {
    auto doc = docs[i];
    int doc_id = i * 100 + 22;
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(doc_id));
  }
}

TEST_F(LikeTest, ForwardSuffixLike) {
  VectorQuery query;
  query.output_fields_ = {"name"};
  query.topk_ = 200;
  query.filter_ = "name like '%ser-22'";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error();
  auto docs = std::move(ret.value());
  for (size_t i = 0; i < docs.size(); i++) {
    auto doc = docs[i];
    int doc_id = i * 100 + 22;
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(doc_id));
  }
}

TEST_F(LikeTest, NotExtendedInvertSuffixLikeRunAsForward) {
  VectorQuery query;
  query.output_fields_ = {"name"};
  query.topk_ = 200;
  query.filter_ = "invert_name like '%ser-22'";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error();
  auto docs = std::move(ret.value());
  for (size_t i = 0; i < docs.size(); i++) {
    auto doc = docs[i];
    int doc_id = i * 100 + 22;
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(doc_id));
  }
}

TEST_F(LikeTest, ExtendedInvertSuffixLike) {
  VectorQuery query;
  query.output_fields_ = {"name"};
  query.topk_ = 200;
  query.filter_ = "extended_invert_name like '%ser-22'";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error();
  auto docs = std::move(ret.value());
  for (size_t i = 0; i < docs.size(); i++) {
    auto doc = docs[i];
    int doc_id = i * 100 + 22;
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(doc_id));
  }
}

TEST_F(LikeTest, ForwardMiddleLike) {
  VectorQuery query;
  query.output_fields_ = {"name"};
  query.topk_ = 200;
  query.filter_ = "name like 'user%2'";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error();
  auto docs = std::move(ret.value());
  for (size_t i = 0, doc_id = 0; i < docs.size(); i++, doc_id++) {
    auto doc = docs[i];
    while (doc_id % 100 % 10 != 2) {
      doc_id++;
    }
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(doc_id));
  }
}

TEST_F(LikeTest, ExtendedInvertMiddleLike) {
  VectorQuery query;
  query.output_fields_ = {"name"};
  query.topk_ = 200;
  query.filter_ = "extended_invert_name like 'user%2'";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error();
  auto docs = std::move(ret.value());
  for (size_t i = 0, doc_id = 0; i < docs.size(); i++, doc_id++) {
    auto doc = docs[i];
    while (doc_id % 100 % 10 != 2) {
      doc_id++;
    }
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(doc_id));
  }
}

TEST_F(LikeTest, UnderScore) {
  VectorQuery query;
  query.output_fields_ = {"name"};
  query.topk_ = 200;
  query.filter_ = "name like 'user-_2'";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error();
  auto docs = std::move(ret.value());
  for (size_t i = 0, doc_id = 0; i < docs.size(); i++, doc_id++) {
    auto doc = docs[i];
    while (doc_id % 100 % 10 != 2 || doc_id % 100 < 10) {
      doc_id++;
    }
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(doc_id));
  }
}

TEST_F(LikeTest, InvertUnderScoreRunAsForward) {
  VectorQuery query;
  query.output_fields_ = {"name"};
  query.topk_ = 200;
  query.filter_ = "invert_name like 'user-_2'";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error();
  auto docs = std::move(ret.value());
  for (size_t i = 0, doc_id = 0; i < docs.size(); i++, doc_id++) {
    auto doc = docs[i];
    while (doc_id % 100 % 10 != 2 || doc_id % 100 < 10) {
      doc_id++;
    }
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(doc_id));
  }
}

TEST_F(LikeTest, ForwardEscapePercent) {
  VectorQuery query;
  query.output_fields_ = {"name"};
  query.topk_ = 200;
  query.filter_ = R"(name like 'user-\%%')";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error();
  auto docs = std::move(ret.value());
  for (size_t i = 0, doc_id = 5000; i < docs.size(); i++, doc_id++) {
    auto doc = docs[i];
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(doc_id));
  }
}

TEST_F(LikeTest, InvertEscapePercent) {
  VectorQuery query;
  query.output_fields_ = {"name"};
  query.topk_ = 200;
  query.filter_ = R"(invert_name like 'user-\%%')";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error();
  auto docs = std::move(ret.value());
  for (size_t i = 0, doc_id = 5000; i < docs.size(); i++, doc_id++) {
    auto doc = docs[i];
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(doc_id));
  }
}

TEST_F(LikeTest, ForwardEscapeUnderscore) {
  VectorQuery query;
  query.output_fields_ = {"name"};
  query.topk_ = 200;
  query.filter_ = R"(name like 'user-\_%')";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error();
  auto docs = std::move(ret.value());
  for (size_t i = 0, doc_id = 8000; i < docs.size(); i++, doc_id++) {
    auto doc = docs[i];
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(doc_id));
  }
}

TEST_F(LikeTest, InvertEscapeUnderscore) {
  VectorQuery query;
  query.output_fields_ = {"name"};
  query.topk_ = 200;
  query.filter_ = R"(invert_name like 'user-\_%')";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error();
  auto docs = std::move(ret.value());
  for (size_t i = 0, doc_id = 8000; i < docs.size(); i++, doc_id++) {
    auto doc = docs[i];
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(doc_id));
  }
}

TEST_F(LikeTest, NoPercentRunAsEqual) {
  VectorQuery query;
  query.output_fields_ = {"name"};
  query.topk_ = 200;
  query.filter_ = R"(invert_name like 'user-22')";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  EXPECT_TRUE(ret.has_value()) << ret.error();
  auto docs = std::move(ret.value());
  for (size_t i = 0; i < docs.size(); i++) {
    auto doc = docs[i];
    int doc_id = i * 100 + 22;
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(doc_id));
  }
}

}  // namespace zvec::sqlengine