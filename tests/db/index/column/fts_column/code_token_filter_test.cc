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

#include <gtest/gtest.h>
#include "db/index/column/fts_column/tokenizer/tokenizer_factory.h"

namespace zvec::fts {
namespace {

void check_tokens(const std::vector<Token> &actual,
                  const std::vector<Token> &expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    SCOPED_TRACE(i);
    EXPECT_EQ(actual[i].text, expected[i].text);
    EXPECT_EQ(actual[i].offset, expected[i].offset);
    EXPECT_EQ(actual[i].position, i);
  }
}

TEST(CodeTokenFilterTest, PreservesUpstreamTextAndByteOffsets) {
  FtsIndexParams params;
  params.tokenizer_name = "whitespace";
  params.filters = {"code"};
  auto pipeline = TokenizerFactory::create(params);
  ASSERT_TRUE(pipeline.has_value());
  auto tokens = pipeline.value()->process(
      "中文.caféValue..HTTPServer_2fa. .___ __getURLValue__ getValue()");
  check_tokens(tokens, {{"中文", 0},
                        {"caféValue", 7},
                        {"HTTP", 19},
                        {"Server", 23},
                        {"2fa", 30},
                        {"get", 42},
                        {"URL", 45},
                        {"Value", 48},
                        {"getValue()", 56}});
  EXPECT_TRUE(pipeline.value()->process("... __ ._.").empty());
}

TEST(CodeTokenFilterTest, SplitsBeforeLowercasing) {
  FtsIndexParams params;
  params.filters = {"code", "lowercase"};
  auto pipeline = TokenizerFactory::create(params);
  ASSERT_TRUE(pipeline.has_value());
  auto tokens = pipeline.value()->process(
      "get_rpc_request_time abc.xyz.efg getGet base64Encode SHA256 3.14");
  check_tokens(tokens, {{"get", 0},
                        {"rpc", 4},
                        {"request", 8},
                        {"time", 16},
                        {"abc", 21},
                        {"xyz", 25},
                        {"efg", 29},
                        {"get", 33},
                        {"get", 36},
                        {"base64", 40},
                        {"encode", 46},
                        {"sha256", 53},
                        {"3", 60},
                        {"14", 62}});
}

TEST(CodeTokenFilterTest, ExpandsIdentifiersFromJieba) {
  FtsIndexParams params;
  params.tokenizer_name = "jieba";
  params.filters = {"code", "lowercase"};
  auto pipeline = TokenizerFactory::create(params);
  ASSERT_TRUE(pipeline.has_value());
  check_tokens(pipeline.value()->process("获取getValue"),
               {{"获取", 0}, {"get", 6}, {"value", 9}});
}

}  // namespace
}  // namespace zvec::fts
