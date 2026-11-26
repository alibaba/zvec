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

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/utility/env_helper.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <zvec/ailego/utility/file_helper.h>
#endif

namespace {

using zvec::ailego::EnvHelper;
using EnvStatus = EnvHelper::Status;

class EnvHelperTest : public ::testing::Test {
 protected:
  void SetUp() override {
#if defined(_WIN32) && defined(ZVEC_TEST_EXPECTED_ACP)
    if (GetACP() != ZVEC_TEST_EXPECTED_ACP) {
      GTEST_SKIP() << "process ACP is " << GetACP() << ", expected "
                   << ZVEC_TEST_EXPECTED_ACP;
    }
#endif
    static std::atomic<unsigned int> sequence{0};
    name_ = "ZVEC_ENV_HELPER_TEST_" +
            std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()) +
            "_" + std::to_string(sequence.fetch_add(1));
  }

  void TearDown() override {
    for (const auto &name : names_) {
      EXPECT_EQ(SetEnvironment(name, nullptr), 0);
    }
  }

  int set(const std::string &name, const char *value) {
    names_.push_back(name);
    return SetEnvironment(name, value);
  }

  std::string name_;

 private:
  static int SetEnvironment(const std::string &name, const char *value) {
#ifdef _WIN32
    const auto wide_name = zvec::ailego::FileHelper::Utf8ToWide(name);
    const auto wide_value =
        zvec::ailego::FileHelper::Utf8ToWide(value ? value : "");
    return SetEnvironmentVariableW(wide_name.c_str(),
                                   value ? wide_value.c_str() : nullptr)
               ? 0
               : -1;
#else
    return value ? setenv(name.c_str(), value, 1) : unsetenv(name.c_str());
#endif
  }

  std::vector<std::string> names_;
};

TEST_F(EnvHelperTest, MissingValueReturnsNotExist) {
  std::string env = "stale";
  EXPECT_EQ(EnvHelper::Get(name_, &env), EnvStatus::kNotExist);
  EXPECT_TRUE(env.empty());
}

TEST_F(EnvHelperTest, EmptyValueReturnsOk) {
  ASSERT_EQ(set(name_, ""), 0);
  std::string env = "stale";
#ifdef _WIN32
  // A successful empty read must not reuse an earlier Win32 error.
  SetLastError(ERROR_ACCESS_DENIED);
#endif
  EXPECT_EQ(EnvHelper::Get(name_, &env), EnvStatus::kOk);
  EXPECT_TRUE(env.empty());
}

TEST_F(EnvHelperTest, Utf8NamesAndValuesArePreserved) {
  const std::string name = name_ + "_中文_日本語_한국어_😀";
  // Environment values are arbitrary text; preserve separators and whitespace.
  const std::string value =
      " 中文 日本語 한국어 😀\nhttps://host//path;a=b|C:\\data\\ ";
  ASSERT_EQ(set(name, value.c_str()), 0);
  std::string env;
  ASSERT_EQ(EnvHelper::Get(name, &env), EnvStatus::kOk);
  EXPECT_EQ(env, value);
}

TEST_F(EnvHelperTest, ReadsCurrentProcessEnvironmentAndReturnsOwnedValue) {
  ASSERT_EQ(set(name_, "first"), 0);
  std::string first;
  ASSERT_EQ(EnvHelper::Get(name_, &first), EnvStatus::kOk);
  ASSERT_EQ(set(name_, "second"), 0);
  std::string current;
  ASSERT_EQ(EnvHelper::Get(name_, &current), EnvStatus::kOk);
  EXPECT_EQ(current, "second");
  EXPECT_EQ(first, "first");
  ASSERT_EQ(set(name_, nullptr), 0);
  EXPECT_EQ(EnvHelper::Get(name_, &current), EnvStatus::kNotExist);
  EXPECT_TRUE(current.empty());
  EXPECT_EQ(first, "first");
}

TEST_F(EnvHelperTest, LongValueIsNotTruncated) {
  const std::string value = std::string(16000, 'x') + "中文😀";
  ASSERT_EQ(set(name_, value.c_str()), 0);
  std::string env;
  ASSERT_EQ(EnvHelper::Get(name_, &env), EnvStatus::kOk);
  EXPECT_EQ(env, value);
}

TEST_F(EnvHelperTest, InvalidNamesReturnError) {
  for (const auto &name :
       {std::string(), std::string("a=b"), std::string("a\0b", 3)}) {
    std::string env = "stale";
    EXPECT_EQ(EnvHelper::Get(name, &env), EnvStatus::kError);
    EXPECT_TRUE(env.empty());
  }
}

TEST_F(EnvHelperTest, NullOutputReturnsError) {
  EXPECT_EQ(EnvHelper::Get(name_, nullptr), EnvStatus::kError);
  ASSERT_EQ(set(name_, "value"), 0);
  EXPECT_EQ(EnvHelper::Get(name_, nullptr), EnvStatus::kError);
}

TEST_F(EnvHelperTest, OutputCanAliasName) {
  ASSERT_EQ(set(name_, "value"), 0);
  std::string env = name_;
  ASSERT_EQ(EnvHelper::Get(env, &env), EnvStatus::kOk);
  EXPECT_EQ(env, "value");
}

#ifdef _WIN32
TEST_F(EnvHelperTest, InvalidUtf8NameReturnsError) {
  std::string env = "stale";
  EXPECT_EQ(EnvHelper::Get(name_ + "\xff", &env), EnvStatus::kError);
  EXPECT_TRUE(env.empty());
}

TEST_F(EnvHelperTest, InvalidUtf16ValueReturnsError) {
  ASSERT_EQ(set(name_, "initial"), 0);
  const auto wide_name = zvec::ailego::FileHelper::Utf8ToWide(name_);
  const wchar_t invalid[] = {0xd800, 0};
  ASSERT_TRUE(SetEnvironmentVariableW(wide_name.c_str(), invalid));
  std::string env = "stale";
  EXPECT_EQ(EnvHelper::Get(name_, &env), EnvStatus::kError);
  EXPECT_TRUE(env.empty());
}
#endif

}  // namespace
