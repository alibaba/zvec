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

#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <gtest/gtest.h>
#include <zvec/ailego/utility/file_helper.h>
#include "db/common/file_helper.h"
#include "index/utils/utils.h"
#include "zvec/db/collection.h"
#include "zvec/db/options.h"
#include "zvec/db/schema.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#undef CreateFile
#undef DELETE
#undef DeleteFile
#undef RemoveDirectory

#ifndef ZVEC_TEST_EXPECTED_ACP
#error "ZVEC_TEST_EXPECTED_ACP must be defined by the build"
#endif

using namespace zvec;
using namespace zvec::test;

namespace {

struct PathCase {
  const char *label;
  const char *utf8_name;
};

// Store path samples as explicit UTF-8 bytes so the test data is independent
// of the encoding used to save this source file.
const PathCase kPathCases[] = {
    {"zh-Hans", "\xe4\xb8\xad\xe6\x96\x87\xe6\xb5\x8b\xe8\xaf\x95"},
    {"ja-ko-issue-665",
     "\xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e-"
     "\xed\x95\x9c\xea\xb5\xad\xec\x96\xb4"},
    {"zh-Hant", "\xe7\xb9\x81\xe9\xab\x94\xe4\xb8\xad\xe6\x96\x87"},
    {"issue-626", "di22222ci\xe3\x80\x82\xe3\x80\x82-cmy"},
};

enum class NarrowConversion {
  kExact,
  kMojibake,
  kThrows,
};

NarrowConversion ClassifyNarrowConversion(const std::string &utf8,
                                          std::string *message) {
  const std::filesystem::path expected = ailego::FileHelper::PathFromUtf8(utf8);
  try {
    // This is the unsafe conversion used by the old WAL existence checks.
    const std::filesystem::path narrow(utf8);
    return narrow.native() == expected.native() ? NarrowConversion::kExact
                                                : NarrowConversion::kMojibake;
  } catch (const std::exception &e) {
    if (message != nullptr) {
      *message = e.what();
    }
    return NarrowConversion::kThrows;
  }
}

class Utf8AcpTest : public ::testing::Test {
 protected:
  void SetUp() override {
    acp_ = ::GetACP();
    if (acp_ != ZVEC_TEST_EXPECTED_ACP) {
      GTEST_SKIP() << "process ACP is " << acp_ << ", expected "
                   << ZVEC_TEST_EXPECTED_ACP;
    }
  }

  static std::string DirFor(const PathCase &path_case) {
    return "utf8_acp" + std::to_string(ZVEC_TEST_EXPECTED_ACP) + "_" +
           path_case.utf8_name;
  }

  static void Remove(const std::string &dir) {
    ailego::FileHelper::RemovePath(dir.c_str());
  }

  unsigned int acp_ = 0;
};

TEST_F(Utf8AcpTest, PathConversionIsAcpIndependent) {
  int hazardous = 0;
  for (const auto &path_case : kPathCases) {
    const std::string name = DirFor(path_case);
    SCOPED_TRACE(std::string(path_case.label) + " -> " + name);

    const std::filesystem::path converted =
        ailego::FileHelper::PathFromUtf8(name);
    EXPECT_EQ(ailego::FileHelper::PathToUtf8(converted), name);

    std::string message;
    switch (ClassifyNarrowConversion(name, &message)) {
      case NarrowConversion::kExact:
        std::cout << "[exact] " << path_case.label << std::endl;
        break;
      case NarrowConversion::kMojibake:
        ++hazardous;
        std::cout << "[mojibake] " << path_case.label << std::endl;
        break;
      case NarrowConversion::kThrows:
        ++hazardous;
        std::cout << "[throws] " << path_case.label << ": " << message
                  << std::endl;
        break;
    }
  }

  if (acp_ == CP_UTF8) {
    EXPECT_EQ(hazardous, 0);
  } else {
    EXPECT_GT(hazardous, 0)
        << "test paths do not expose this code page's narrow-path hazard";
  }
}

TEST_F(Utf8AcpTest, CollectionWorkflowIsAcpIndependent) {
  constexpr int kDocCount = 3;

  for (const auto &path_case : kPathCases) {
    const std::string dir = DirFor(path_case);
    SCOPED_TRACE(std::string(path_case.label) + " -> " + dir);
    Remove(dir);

    CollectionOptions options;
    options.read_only_ = false;
    options.enable_mmap_ = true;
    auto schema = TestHelper::CreateNormalSchema();

    auto created = Collection::CreateAndOpen(dir, *schema, options);
    ASSERT_TRUE(created.has_value()) << created.error().message();
    auto collection = std::move(created).value();

    auto status = TestHelper::CollectionInsertDoc(collection, 0, kDocCount);
    ASSERT_TRUE(status.ok()) << status.message();
    ASSERT_TRUE(collection->Flush().ok());
    collection.reset();

    auto reopened = Collection::Open(dir, options);
    ASSERT_TRUE(reopened.has_value()) << reopened.error().message();
    auto collection2 = std::move(reopened).value();
    ASSERT_EQ(collection2->Stats().value().doc_count, kDocCount);

    status =
        TestHelper::CollectionInsertDoc(collection2, kDocCount, kDocCount * 2);
    ASSERT_TRUE(status.ok()) << status.message();
    ASSERT_TRUE(collection2->Flush().ok());
    ASSERT_EQ(collection2->Stats().value().doc_count, kDocCount * 2);

    auto schema_value = collection2->Schema().value();
    for (int i = 0; i < kDocCount * 2; ++i) {
      auto expected = TestHelper::CreateDoc(i, schema_value);
      auto fetched = collection2->Fetch({expected.pk()});
      ASSERT_TRUE(fetched.has_value()) << fetched.error().message();
      ASSERT_EQ(fetched.value().count(expected.pk()), 1u)
          << "missing doc " << i;
    }

    collection2.reset();
    Remove(dir);
  }
}

TEST_F(Utf8AcpTest, WalExistenceCheckIsAcpIndependent) {
  for (const auto &path_case : kPathCases) {
    const std::string dir = DirFor(path_case);
    SCOPED_TRACE(std::string(path_case.label) + " -> " + dir);
    Remove(dir);

    const std::string wal_path = FileHelper::MakeWalPath(dir, 0, 0);
    const std::filesystem::path native_path =
        ailego::FileHelper::PathFromUtf8(wal_path);

    std::filesystem::create_directories(native_path.parent_path());
    ASSERT_FALSE(std::filesystem::exists(native_path));
    EXPECT_FALSE(FileHelper::FileExists(wal_path));

    {
      std::ofstream output(native_path, std::ios::binary);
      ASSERT_TRUE(output.is_open());
    }

    ASSERT_TRUE(std::filesystem::exists(native_path));
    EXPECT_TRUE(FileHelper::FileExists(wal_path));
    Remove(dir);
  }
}

}  // namespace
