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
#include <cstdlib>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/utility/file_helper.h>
#include <zvec/db/collection.h>
#include <zvec/db/doc.h>
#include <zvec/db/schema.h>

namespace zvec {
namespace {

class RelaxedValidationDeathTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Re-exec children start with the default style; select threadsafe before
    // InDeathTestChild() interprets their internal death-test flag.
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    // Threadsafe death tests re-exec the fixture. A second crash child must
    // reopen the first child's database instead of deleting it in SetUp.
    if (!::testing::internal::InDeathTestChild()) {
      ailego::FileHelper::RemovePath(path_.c_str());
    }
  }

  void TearDown() override {
    ailego::FileHelper::RemovePath(path_.c_str());
  }

  const std::string path_{"relaxed_validation_recovery_db"};
};

TEST_F(RelaxedValidationDeathTest, Utf8AndLongIdsRecoverFromUnflushedWal) {
  const std::vector<std::string> ids{u8"订单:😀",
                                     std::string(1021, 'x') + u8"中", " doc ",
                                     u8"café", u8"cafe\u0301"};
  // Re-exec the child before starting collection threads. Exit without stack
  // unwinding so Collection destruction cannot flush the writing segment.
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  ASSERT_EXIT(
      {
        CollectionSchema schema(u8"恢复 集合");
        if (!schema
                 .add_field(std::make_shared<FieldSchema>(
                     "value", DataType::INT32, false))
                 .ok()) {
          std::_Exit(1);
        }
        auto created =
            Collection::CreateAndOpen(path_, schema, CollectionOptions{});
        if (!created.has_value()) std::_Exit(2);
        auto collection = std::move(created).value();
        std::vector<Doc> docs;
        for (size_t i = 0; i < ids.size(); ++i) {
          Doc doc;
          doc.set_pk(ids[i]);
          doc.set<int32_t>("value", static_cast<int32_t>(i));
          docs.push_back(std::move(doc));
        }
        auto inserted = collection->insert(docs);
        if (!inserted.has_value()) std::_Exit(3);
        for (const auto &status : inserted.value()) {
          if (!status.ok()) std::_Exit(4);
        }
        std::_Exit(0);
      },
      ::testing::ExitedWithCode(0), "");

  auto opened = Collection::Open(path_, CollectionOptions{});
  ASSERT_TRUE(opened.has_value()) << opened.error().message();
  auto collection = std::move(opened).value();
  EXPECT_EQ(collection->schema().value().name(), u8"恢复 集合");
  auto fetched = collection->fetch(ids);
  ASSERT_TRUE(fetched.has_value()) << fetched.error().message();
  ASSERT_EQ(fetched.value().size(), ids.size());
  for (size_t i = 0; i < ids.size(); ++i) {
    const auto found = fetched.value().find(ids[i]);
    ASSERT_NE(found, fetched.value().end());
    ASSERT_NE(found->second, nullptr);
    EXPECT_EQ(found->second->pk_ref(), ids[i]);
    EXPECT_EQ(found->second->get<int32_t>("value"), static_cast<int32_t>(i));
  }
  auto status = collection->flush();
  ASSERT_TRUE(status.ok()) << status.message();
  collection.reset();
  auto reopened = Collection::Open(path_, CollectionOptions{});
  ASSERT_TRUE(reopened.has_value()) << reopened.error().message();
  EXPECT_EQ(reopened.value()->stats().value().doc_count, ids.size());
}

}  // namespace
}  // namespace zvec
