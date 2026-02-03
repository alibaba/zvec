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
#include "zvec/db/schema.h"
#include "recall_base.h"

namespace zvec::sqlengine {

class InvertRecallTest : public RecallTest {};

TEST_F(InvertRecallTest, Eq) {
  VectorQuery query;
  query.output_fields_ = {"id", "name", "age"};
  query.topk_ = 200;
  query.filter_ = "invert_age = 1";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error().c_str();
  auto docs = ret.value();
  EXPECT_EQ(docs.size(), 100);
  for (int j = 0, i = 1; j < (int)docs.size(); j++, i += 100) {
    auto &doc = docs[j];
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(i));
    EXPECT_EQ(i, doc->get<uint64_t>("id"));
    auto age = doc->get<int32_t>("age");
    EXPECT_EQ(age.value(), i % 100);
    auto name = doc->get<std::string>("name");
    ASSERT_TRUE(name);
    EXPECT_EQ(name.value(), "user_" + std::to_string(i % 100));
  }
}

TEST_F(InvertRecallTest, Gt) {
  VectorQuery query;
  query.output_fields_ = {"id", "name", "age"};
  query.topk_ = 200;
  query.filter_ = "invert_id > 1000";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error().c_str();
  auto docs = ret.value();
  EXPECT_EQ(docs.size(), query.topk_);
  for (int j = 0; j < query.topk_; j++) {
    auto &doc = docs[j];
    auto i = j + 1001;
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(i));
    auto age = doc->get<int32_t>("age");
    EXPECT_EQ(age.value(), i % 100);
    auto name = doc->get<std::string>("name");
    ASSERT_TRUE(name);
    EXPECT_EQ(name.value(), "user_" + std::to_string(i % 100));
  }
}

TEST_F(InvertRecallTest, Ge) {
  VectorQuery query;
  query.output_fields_ = {"id", "name", "age"};
  query.topk_ = 200;
  query.filter_ = "invert_id >= 1000";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error().c_str();
  auto docs = ret.value();
  EXPECT_EQ(docs.size(), query.topk_);
  for (int j = 0; j < query.topk_; j++) {
    auto &doc = docs[j];
    auto i = j + 1000;
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(i));
    auto age = doc->get<int32_t>("age");
    EXPECT_EQ(age.value(), i % 100);
    auto name = doc->get<std::string>("name");
    ASSERT_TRUE(name);
    EXPECT_EQ(name.value(), "user_" + std::to_string(i % 100));
  }
}

TEST_F(InvertRecallTest, Lt) {
  VectorQuery query;
  query.output_fields_ = {"id", "name", "age"};
  query.topk_ = 200;
  query.filter_ = "invert_id < 100";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error().c_str();
  auto docs = ret.value();
  ASSERT_EQ(docs.size(), 100);
  for (int j = 0, i = 0; j < (int)docs.size(); j++, i += 1) {
    auto &doc = docs[j];
    EXPECT_EQ(i, doc->get<uint64_t>("id"));
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(i));
    auto age = doc->get<int32_t>("age");
    EXPECT_EQ(age.value(), i % 100);
    auto name = doc->get<std::string>("name");
    ASSERT_TRUE(name);
    EXPECT_EQ(name.value(), "user_" + std::to_string(i % 100));
  }
}

TEST_F(InvertRecallTest, Le) {
  VectorQuery query;
  query.output_fields_ = {"id", "name", "age"};
  query.topk_ = 200;
  query.filter_ = "invert_id <= 100";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error().c_str();
  auto docs = ret.value();
  ASSERT_EQ(docs.size(), 101);
  for (int j = 0, i = 0; j < (int)docs.size(); j++, i += 1) {
    auto &doc = docs[j];
    EXPECT_EQ(i, doc->get<uint64_t>("id"));
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(i));
    auto age = doc->get<int32_t>("age");
    EXPECT_EQ(age.value(), i % 100);
    auto name = doc->get<std::string>("name");
    ASSERT_TRUE(name);
    EXPECT_EQ(name.value(), "user_" + std::to_string(i % 100));
  }
}

TEST_F(InvertRecallTest, And) {
  VectorQuery query;
  query.output_fields_ = {"id", "name", "age"};
  query.topk_ = 200;
  query.filter_ = "invert_id <= 100 and invert_id > 50";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error().c_str();
  auto docs = ret.value();
  ASSERT_EQ(docs.size(), 50);
  for (int j = 0, i = 51; j < (int)docs.size(); j++, i += 1) {
    auto &doc = docs[j];
    EXPECT_EQ(i, doc->get<uint64_t>("id"));
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(i));
    auto age = doc->get<int32_t>("age");
    EXPECT_EQ(age.value(), i % 100);
    auto name = doc->get<std::string>("name");
    ASSERT_TRUE(name);
    EXPECT_EQ(name.value(), "user_" + std::to_string(i % 100));
  }
}

TEST_F(InvertRecallTest, Or) {
  VectorQuery query;
  query.output_fields_ = {"id", "name", "age"};
  query.topk_ = 200;
  query.filter_ = "invert_id < 100 or invert_id > 200";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error().c_str();
  auto docs = ret.value();
  ASSERT_EQ(docs.size(), 200);
  for (int j = 0; j < (int)docs.size(); j++) {
    int i = j < 100 ? j : j + 101;
    auto &doc = docs[j];
    EXPECT_EQ(i, doc->get<uint64_t>("id"));
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(i));
    auto age = doc->get<int32_t>("age");
    EXPECT_EQ(age.value(), i % 100);
    auto name = doc->get<std::string>("name");
    ASSERT_TRUE(name);
    EXPECT_EQ(name.value(), "user_" + std::to_string(i % 100));
  }
}

TEST_F(InvertRecallTest, StrEq) {
  VectorQuery query;
  query.output_fields_ = {"id", "name", "age"};
  query.topk_ = 200;
  query.filter_ = "invert_name = 'user_1'";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error().c_str();
  auto docs = ret.value();
  EXPECT_EQ(docs.size(), 100);
  for (int j = 0, i = 1; j < (int)docs.size(); j++, i += 100) {
    auto &doc = docs[j];
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(i));
    EXPECT_EQ(i, doc->get<uint64_t>("id"));
    auto age = doc->get<int32_t>("age");
    EXPECT_EQ(age.value(), i % 100);
    auto name = doc->get<std::string>("name");
    ASSERT_TRUE(name);
    EXPECT_EQ(name.value(), "user_" + std::to_string(i % 100));
  }
}

TEST_F(InvertRecallTest, StrGe) {
  VectorQuery query;
  query.output_fields_ = {"id", "name", "age"};
  query.topk_ = 200;
  query.filter_ = "invert_name >= 'user_1'";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error().c_str();
  auto docs = ret.value();
  EXPECT_EQ(docs.size(), 200);
  for (int j = 0, i = 0; j < (int)docs.size(); j++, i += 1) {
    if (i % 100 == 0) {
      i += 1;
    }
    auto &doc = docs[j];
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(i));
    EXPECT_EQ(i, doc->get<uint64_t>("id"));
    auto age = doc->get<int32_t>("age");
    EXPECT_EQ(age.value(), i % 100);
    auto name = doc->get<std::string>("name");
    ASSERT_TRUE(name);
    EXPECT_EQ(name.value(), "user_" + std::to_string(i % 100));
  }
}

TEST_F(InvertRecallTest, StrIn) {
  VectorQuery query;
  query.output_fields_ = {"id", "name", "age"};
  query.topk_ = 200;
  query.filter_ = "invert_name IN ('user_1', 'user_2')";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error().c_str();
  auto docs = ret.value();
  EXPECT_EQ(docs.size(), 200);
  for (int j = 0, i = 1; j < (int)docs.size(); j++) {
    auto &doc = docs[j];
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(i));
    EXPECT_EQ(i, doc->get<uint64_t>("id"));
    auto age = doc->get<int32_t>("age");
    EXPECT_EQ(age.value(), i % 100);
    auto name = doc->get<std::string>("name");
    ASSERT_TRUE(name);
    EXPECT_EQ(name.value(), "user_" + std::to_string(i % 100));

    if (i % 100 == 1) {
      i += 1;
    } else if (i % 100 == 2) {
      i += 99;
    }
  }
}

TEST_F(InvertRecallTest, StrNotIn) {
  VectorQuery query;
  query.output_fields_ = {"id", "name", "age"};
  query.topk_ = 200;
  query.filter_ = "invert_name NOT IN ('user_1', 'user_2')";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error().c_str();
  auto docs = ret.value();
  EXPECT_EQ(docs.size(), 200);
  for (int j = 0, i = 0; j < (int)docs.size(); j++) {
    auto &doc = docs[j];
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(i));
    EXPECT_EQ(i, doc->get<uint64_t>("id"));
    auto age = doc->get<int32_t>("age");
    EXPECT_EQ(age.value(), i % 100);
    auto name = doc->get<std::string>("name");
    ASSERT_TRUE(name);
    EXPECT_EQ(name.value(), "user_" + std::to_string(i % 100));

    if (i % 100 == 0) {
      i += 3;
    } else {
      i += 1;
    }
  }
}

TEST_F(InvertRecallTest, StrLike) {
  VectorQuery query;
  query.output_fields_ = {"id", "name", "age"};
  query.topk_ = 200;
  query.filter_ = "invert_name like 'user\\_9%'";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error().c_str();
  auto docs = ret.value();
  EXPECT_EQ(docs.size(), 200);
  for (int j = 0, i = 9; j < (int)docs.size(); j++) {
    auto &doc = docs[j];
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(i));
    EXPECT_EQ(i, doc->get<uint64_t>("id"));
    auto age = doc->get<int32_t>("age");
    EXPECT_EQ(age.value(), i % 100);
    auto name = doc->get<std::string>("name");
    ASSERT_TRUE(name);
    EXPECT_EQ(name.value(), "user_" + std::to_string(i % 100));

    if (i % 100 == 9) {
      i += 81;
    } else if (i % 100 == 99) {
      i += 10;
    } else {
      i += 1;
    }
  }
}

TEST_F(InvertRecallTest, ContainAll) {
  VectorQuery query;
  query.output_fields_ = {"id", "name", "age"};
  query.topk_ = 200;
  query.filter_ = "invert_category_set contain_all (";
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
    EXPECT_EQ(i, doc->get<uint64_t>("id"));
    auto age = doc->get<int32_t>("age");
    EXPECT_EQ(age.value(), i % 100);
    auto name = doc->get<std::string>("name");
    ASSERT_TRUE(name);
    EXPECT_EQ(name.value(), "user_" + std::to_string(i % 100));

    i += 1;
    while (i % 100 < 32) {
      i += 1;
    }
  }
}

TEST_F(InvertRecallTest, NotContainAll) {
  VectorQuery query;
  query.output_fields_ = {"id", "name", "age"};
  query.topk_ = 200;
  query.filter_ = "invert_category_set not contain_all (";
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
  for (int j = 0, i = 1; j < (int)docs.size(); j++) {
    auto &doc = docs[j];
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(i));
    EXPECT_EQ(i, doc->get<uint64_t>("id"));
    auto age = doc->get<int32_t>("age");
    EXPECT_EQ(age.value(), i % 100);
    auto name = doc->get<std::string>("name");
    ASSERT_TRUE(name);
    EXPECT_EQ(name.value(), "user_" + std::to_string(i % 100));

    i += 1;
    // i % 100 == 0 has null category
    while (i % 100 >= 32 || i % 100 == 0) {
      i += 1;
    }
  }
}

TEST_F(InvertRecallTest, ContainAny) {
  VectorQuery query;
  query.output_fields_ = {"id", "name", "age"};
  query.topk_ = 200;
  query.filter_ = "invert_category_set contain_any (98,99,100)";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error().c_str();
  auto docs = ret.value();
  EXPECT_EQ(docs.size(), 200);
  for (int j = 0, i = 98; j < (int)docs.size(); j++) {
    auto &doc = docs[j];
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(i));
    EXPECT_EQ(i, doc->get<uint64_t>("id"));
    auto age = doc->get<int32_t>("age");
    EXPECT_EQ(age.value(), i % 100);
    auto name = doc->get<std::string>("name");
    ASSERT_TRUE(name);
    EXPECT_EQ(name.value(), "user_" + std::to_string(i % 100));

    i += 1;
    while (i % 100 < 98) {
      i += 1;
    }
  }
}

TEST_F(InvertRecallTest, NotContainAny) {
  VectorQuery query;
  query.output_fields_ = {"id", "name", "age"};
  query.topk_ = 200;
  query.filter_ = "invert_category_set not contain_any (98,99,100)";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error().c_str();
  auto docs = ret.value();
  EXPECT_EQ(docs.size(), 200);
  for (int j = 0, i = 1; j < (int)docs.size(); j++) {
    auto &doc = docs[j];
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(i));
    EXPECT_EQ(i, doc->get<uint64_t>("id"));
    auto age = doc->get<int32_t>("age");
    EXPECT_EQ(age.value(), i % 100);
    auto name = doc->get<std::string>("name");
    ASSERT_TRUE(name);
    EXPECT_EQ(name.value(), "user_" + std::to_string(i % 100));

    i += 1;
    // i % 100 == 0 has null category
    while (i % 100 >= 98 || i % 100 == 0) {
      i += 1;
    }
  }
}

TEST_F(InvertRecallTest, BoolContainAll) {
  VectorQuery query;
  query.output_fields_ = {"id", "name", "age"};
  query.topk_ = 200;
  query.filter_ = "invert_bool_array contain_all (true, false)";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error().c_str();
  auto docs = ret.value();
  EXPECT_EQ(docs.size(), 200);
  for (int j = 0, i = 0; j < (int)docs.size(); j++) {
    auto &doc = docs[j];
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(i));
    EXPECT_EQ(i, doc->get<uint64_t>("id"));
    auto age = doc->get<int32_t>("age");
    EXPECT_EQ(age.value(), i % 100);
    auto name = doc->get<std::string>("name");
    ASSERT_TRUE(name);
    EXPECT_EQ(name.value(), "user_" + std::to_string(i % 100));

    i += 3;
  }
}

TEST_F(InvertRecallTest, BoolContainAny) {
  VectorQuery query;
  query.output_fields_ = {"id", "name", "age"};
  query.topk_ = 200;
  query.filter_ = "invert_bool_array contain_any (true)";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error().c_str();
  auto docs = ret.value();
  EXPECT_EQ(docs.size(), 200);
  for (int j = 0, i = 0; j < (int)docs.size(); j++) {
    auto &doc = docs[j];
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(i));
    EXPECT_EQ(i, doc->get<uint64_t>("id"));
    auto age = doc->get<int32_t>("age");
    EXPECT_EQ(age.value(), i % 100);
    auto name = doc->get<std::string>("name");
    ASSERT_TRUE(name);
    EXPECT_EQ(name.value(), "user_" + std::to_string(i % 100));

    i += 1;
    if (i % 3 == 2) {
      i += 1;
    }
  }
}

TEST_F(InvertRecallTest, ContainAllEmptySet) {
  VectorQuery query;
  query.output_fields_ = {"id", "name", "age"};
  query.topk_ = 200;
  query.filter_ = "invert_category_set contain_all ()";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error().c_str();
  auto docs = ret.value();
  EXPECT_EQ(docs.size(), 200);
  for (int j = 0, i = 1; j < (int)docs.size(); j++) {
    auto &doc = docs[j];
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(i));
    EXPECT_EQ(i, doc->get<uint64_t>("id"));
    auto age = doc->get<int32_t>("age");
    EXPECT_EQ(age.value(), i % 100);
    auto name = doc->get<std::string>("name");
    ASSERT_TRUE(name);
    EXPECT_EQ(name.value(), "user_" + std::to_string(i % 100));

    i += 1;
    while (i % 100 == 0) {
      i += 1;
    }
  }
}

TEST_F(InvertRecallTest, NotContainAllEmptySet) {
  VectorQuery query;
  query.output_fields_ = {"id", "name", "age"};
  query.topk_ = 200;
  query.filter_ = "invert_category_set not contain_all ()";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error().c_str();
  auto docs = ret.value();
  ASSERT_EQ(docs.size(), 0);
}

TEST_F(InvertRecallTest, ContainAnyEmptySet) {
  VectorQuery query;
  query.output_fields_ = {"id", "name", "age"};
  query.topk_ = 200;
  query.filter_ = "invert_category_set contain_any ()";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error().c_str();
  auto docs = ret.value();
  ASSERT_EQ(docs.size(), 0);
}

TEST_F(InvertRecallTest, NotContainAnyEmptySet) {
  VectorQuery query;
  query.output_fields_ = {"id", "name", "age"};
  query.topk_ = 200;
  query.filter_ = "invert_category_set not contain_any ()";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error().c_str();
  auto docs = ret.value();
  EXPECT_EQ(docs.size(), 200);
  for (int j = 0, i = 1; j < (int)docs.size(); j++) {
    auto &doc = docs[j];
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(i));
    EXPECT_EQ(i, doc->get<uint64_t>("id"));
    auto age = doc->get<int32_t>("age");
    EXPECT_EQ(age.value(), i % 100);
    auto name = doc->get<std::string>("name");
    ASSERT_TRUE(name);
    EXPECT_EQ(name.value(), "user_" + std::to_string(i % 100));

    i += 1;
    while (i % 100 == 0) {
      i += 1;
    }
  }
}

TEST_F(InvertRecallTest, IsNull) {
  VectorQuery query;
  query.output_fields_ = {"id", "name", "age"};
  query.topk_ = 200;
  query.filter_ = "invert_optional_age is null";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error().c_str();
  auto docs = ret.value();
  EXPECT_EQ(docs.size(), 100);
  for (int j = 0, i = 0; j < (int)docs.size(); j++, i += 100) {
    auto &doc = docs[j];
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(i));
    EXPECT_EQ(i, doc->get<uint64_t>("id"));
    auto age = doc->get<int32_t>("age");
    EXPECT_EQ(age.value(), i % 100);
    auto name = doc->get<std::string>("name");
    ASSERT_TRUE(name);
    EXPECT_EQ(name.value(), "user_" + std::to_string(i % 100));
  }
}

TEST_F(InvertRecallTest, IsNotNull) {
  VectorQuery query;
  query.output_fields_ = {"id", "name", "age"};
  query.topk_ = 200;
  query.filter_ = "invert_optional_age is not null";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error().c_str();
  auto docs = ret.value();
  EXPECT_EQ(docs.size(), 200);
  for (int j = 0, i = 0; j < (int)docs.size(); j++, i += 1) {
    if (i % 100 == 0) {
      i += 1;
    }
    auto &doc = docs[j];
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(i));
    EXPECT_EQ(i, doc->get<uint64_t>("id"));
    auto age = doc->get<int32_t>("age");
    EXPECT_EQ(age.value(), i % 100);
    auto name = doc->get<std::string>("name");
    ASSERT_TRUE(name);
    EXPECT_EQ(name.value(), "user_" + std::to_string(i % 100));
  }
}

TEST_F(InvertRecallTest, BoolEqTrue) {
  VectorQuery query;
  query.output_fields_ = {"id", "name", "age"};
  query.topk_ = 200;
  query.filter_ = "invert_bool = TRuE";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error().c_str();
  auto docs = ret.value();
  EXPECT_EQ(docs.size(), 100);
  for (int j = 0, i = 0; j < (int)docs.size(); j++, i += 100) {
    auto &doc = docs[j];
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(i));
    EXPECT_EQ(i, doc->get<uint64_t>("id"));
    auto age = doc->get<int32_t>("age");
    EXPECT_EQ(age.value(), i % 100);
    auto name = doc->get<std::string>("name");
    ASSERT_TRUE(name);
    EXPECT_EQ(name.value(), "user_" + std::to_string(i % 100));
  }
}

TEST_F(InvertRecallTest, BoolEqFalse) {
  VectorQuery query;
  query.output_fields_ = {"id", "name", "age"};
  query.topk_ = 200;
  query.filter_ = "invert_bool = false";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error().c_str();
  auto docs = ret.value();
  EXPECT_EQ(docs.size(), 200);
  for (int j = 0, i = 1; j < (int)docs.size(); j++) {
    auto &doc = docs[j];
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(i));
    EXPECT_EQ(i, doc->get<uint64_t>("id"));
    auto age = doc->get<int32_t>("age");
    EXPECT_EQ(age.value(), i % 100);
    auto name = doc->get<std::string>("name");
    ASSERT_TRUE(name);
    EXPECT_EQ(name.value(), "user_" + std::to_string(i % 100));

    i += 1;
    if (i % 100 == 0) {
      i += 1;
    }
  }
}

TEST_F(InvertRecallTest, ArrayLengthGe) {
  VectorQuery query;
  query.output_fields_ = {"id", "name", "age"};
  query.topk_ = 200;
  query.filter_ = "array_length(invert_category_set) >= 32";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error().c_str();
  auto docs = ret.value();
  EXPECT_EQ(docs.size(), 200);
  for (int j = 0, i = 32; j < (int)docs.size(); j++) {
    auto &doc = docs[j];
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(i));
    EXPECT_EQ(i, doc->get<uint64_t>("id"));
    auto age = doc->get<int32_t>("age");
    EXPECT_EQ(age.value(), i % 100);
    auto name = doc->get<std::string>("name");
    ASSERT_TRUE(name);
    EXPECT_EQ(name.value(), "user_" + std::to_string(i % 100));

    i += 1;
    while (i % 100 < 32) {
      i += 1;
    }
  }
}

TEST_F(InvertRecallTest, ArrayLengthEq) {
  VectorQuery query;
  query.output_fields_ = {"id", "name", "age"};
  query.topk_ = 200;
  query.filter_ = "array_length(invert_category_set) = 32";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  auto ret = engine->execute(collection_schema_, query, segments_);
  ASSERT_TRUE(ret.has_value()) << ret.error().c_str();
  auto docs = ret.value();
  EXPECT_EQ(docs.size(), 100);
  for (int j = 0, i = 32; j < (int)docs.size(); j++) {
    auto &doc = docs[j];
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(i));
    EXPECT_EQ(i, doc->get<uint64_t>("id"));
    auto age = doc->get<int32_t>("age");
    EXPECT_EQ(age.value(), i % 100);
    auto name = doc->get<std::string>("name");
    ASSERT_TRUE(name);
    EXPECT_EQ(name.value(), "user_" + std::to_string(i % 100));

    i += 100;
  }
}

TEST_F(InvertRecallTest, MultiSegment) {
  VectorQuery query;
  query.output_fields_ = std::vector<std::string>();
  query.topk_ = 200;
  query.include_vector_ = true;
  query.filter_ = "invert_id <= 5000";

  auto engine = SQLEngine::create(std::make_shared<Profiler>());
  std::vector<Segment::Ptr> segments = segments_;
  segments.push_back(segments_[0]);
  auto ret = engine->execute(collection_schema_, query, segments);
  if (!ret) {
    LOG_ERROR("execute failed: [%s]", ret.error().c_str());
  }
  ASSERT_TRUE(ret.has_value()) << ret.error().c_str();
  auto docs = ret.value();
  EXPECT_EQ(docs.size(), query.topk_);
  for (int i = 0; i < query.topk_; i++) {
    auto &doc = docs[i];
    EXPECT_EQ(doc->pk(), "pk_" + std::to_string(i));
    auto dense = doc->get<std::vector<float>>("dense");
    ASSERT_TRUE(dense.has_value());
    EXPECT_EQ(dense.value().size(), 4);
    for (auto v : dense.value()) {
      EXPECT_FLOAT_EQ(v, (float)i);
    }

    auto sparse =
        doc->get<std::pair<std::vector<uint32_t>, std::vector<float>>>(
            "sparse");
    if (i % 100 == 0) {
      // set with empty vector
      ASSERT_FALSE(sparse.has_value());
      continue;
    }

    ASSERT_TRUE(sparse.has_value());
    const auto &[indices, values] = sparse.value();
    EXPECT_EQ(indices.size(), i % 100);
    EXPECT_EQ(values.size(), i % 100);
    for (int j = 0; j < i % 100; j++) {
      EXPECT_EQ(indices[j], j);
      EXPECT_FLOAT_EQ(values[j], (float)i);
    }
  }
}

}  // namespace zvec::sqlengine
