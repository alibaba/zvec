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

#include "db/index/column/fts_column/tokenizer/code_tokenizer.h"
#include <fstream>
#include <gtest/gtest.h>
#include "db/index/column/fts_column/tokenizer/jieba_tokenizer.h"
#include "db/index/column/fts_column/tokenizer/tokenizer_factory.h"
#include "db/index/column/fts_column/tokenizer/tokenizer_pipeline_manager.h"

namespace zvec::fts {
namespace {
std::vector<std::string> texts(const std::vector<Token> &tokens) {
  std::vector<std::string> result;
  for (const auto &token : tokens) {
    result.push_back(token.text);
  }
  return result;
}

TEST(CodeTokenizerTest, IdentifierRulesAndPositions) {
  CodeTokenizer tokenizer;
  ASSERT_TRUE(tokenizer.init(ailego::JsonObject()).ok());
  const std::vector<std::pair<std::string, std::vector<std::string>>> cases = {
      {"getRequestTime", {"getrequesttime", "get", "request", "time"}},
      {"GetRequestTime", {"getrequesttime", "get", "request", "time"}},
      {"get_request_time", {"get_request_time", "get", "request", "time"}},
      {"get_requestTime", {"get_requesttime", "get", "request", "time"}},
      {"HTTPRequest", {"httprequest", "http", "request"}},
      {"getURLValue", {"geturlvalue", "get", "url", "value"}},
      {"__request_time__", {"__request_time__", "request", "time"}},
      {"__request__", {"__request__", "request"}},
      {"getGet", {"getget", "get", "get"}},
      {"getaway setup request", {"getaway", "setup", "request"}},
      {"base64Encode sha256 utf8Decode",
       {"base64encode", "base64", "encode", "sha256", "utf8decode", "utf8",
        "decode"}},
      {"123 2fa foo__bar", {"123", "2fa", "foo__bar", "foo", "bar"}},
      {"___ ,.! \t\n", {}},
      {"", {}}};
  for (const auto &entry : cases) {
    SCOPED_TRACE(entry.first);
    auto tokens = tokenizer.tokenize(entry.first);
    EXPECT_EQ(texts(tokens), entry.second);
    for (size_t i = 0; i < tokens.size(); ++i) {
      EXPECT_EQ(tokens[i].position, i);
    }
  }
  auto tokens = tokenizer.tokenize("获取getRequestTime");
  ASSERT_EQ(tokens.size(), 6u);
  EXPECT_EQ(texts(tokens),
            (std::vector<std::string>{"获", "取", "getrequesttime", "get",
                                      "request", "time"}));
  EXPECT_EQ(tokens[0].offset, 0u);
  EXPECT_EQ(tokens[1].offset, 3u);
  EXPECT_EQ(tokens[2].offset, 6u);
  EXPECT_EQ(tokens[3].offset, 6u);
  EXPECT_EQ(tokens[4].offset, 9u);
  EXPECT_EQ(tokens[5].offset, 16u);
}

TEST(CodeTokenizerTest, UnicodeBoundaryProtectionAndMalformedInput) {
  CodeTokenizer tokenizer;
  ASSERT_TRUE(tokenizer.init(ailego::JsonObject()).ok());
  EXPECT_EQ(
      texts(tokenizer.tokenize("caféValue naïveParser CAFÉValue")),
      (std::vector<std::string>{"cafévalue", "naïveparser", "cafévalue"}));
  EXPECT_EQ(texts(tokenizer.tokenize("cafe\xCC\x81Value")),
            (std::vector<std::string>{"cafe\xCC\x81value"}));
  EXPECT_EQ(texts(tokenizer.tokenize("get请求Time")),
            (std::vector<std::string>{"get", "请", "求", "time"}));
  auto tokens = tokenizer.tokenize(std::string("foo\xff", 4) + "getValue");
  EXPECT_EQ(texts(tokens),
            (std::vector<std::string>{"foo", "getvalue", "get", "value"}));
  ASSERT_EQ(tokens.size(), 4u);
  EXPECT_EQ(tokens[1].offset, 4u);
  EXPECT_EQ(tokens[3].offset, 7u);
  // Astral Han must be a boundary just like common ideographs.
  EXPECT_EQ(texts(tokenizer.tokenize("𠀀getValue")),
            (std::vector<std::string>{"𠀀", "getvalue", "get", "value"}));
}

TEST(CodeTokenizerTest, IdentifiersAdjacentToKanaAndHangul) {
  for (const char *child_name : {"standard", "jieba"}) {
    ailego::JsonObject config;
    config.set("sub_tokenizer", child_name);
    CodeTokenizer tokenizer;
    ASSERT_TRUE(tokenizer.init(config).ok());
    for (const std::string script :
         {"かな", "カナ", "ｶﾅ", "한국어", "한", "𛀀"}) {
      for (const bool prefix : {false, true}) {
        const std::string input =
            prefix ? script + "getValue" : "getValue" + script;
        SCOPED_TRACE(std::string(child_name) + ": " + input);
        auto expected = tokenizer.tokenize(script);
        const uint32_t offset =
            prefix ? static_cast<uint32_t>(script.size()) : 0;
        std::vector<Token> identifier{{"getvalue", offset, 0},
                                      {"get", offset, 0},
                                      {"value", offset + 3, 0}};
        if (prefix) {
          expected.insert(expected.end(), identifier.begin(), identifier.end());
        } else {
          for (auto &token : expected) {
            token.offset += 8;
          }
          expected.insert(expected.begin(), identifier.begin(),
                          identifier.end());
        }
        auto actual = tokenizer.tokenize(input);
        ASSERT_EQ(actual.size(), expected.size());
        for (size_t i = 0; i < actual.size(); ++i) {
          EXPECT_EQ(actual[i].text, expected[i].text);
          EXPECT_EQ(actual[i].offset, expected[i].offset);
          EXPECT_EQ(actual[i].position, i);
        }
      }
    }
  }
}

TEST(CodeTokenizerTest, ChildConfigurationAndIntrinsicLowercase) {
  FtsIndexParams params;
  params.tokenizer_name = "code";
  params.filters.clear();
  auto result = TokenizerFactory::create(params);
  ASSERT_TRUE(result.has_value());
  auto expected = result.value()->process("ABC getValue CAFÉ");
  EXPECT_EQ(texts(expected), (std::vector<std::string>{"abc", "getvalue", "get",
                                                       "value", "café"}));
  params.filters = {"lowercase"};
  result = TokenizerFactory::create(params);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(texts(result.value()->process("ABC getValue CAFÉ")),
            texts(expected));
  params.extra_params = R"({"sub_tokenizer_params":{"max_token_length":2}})";
  result = TokenizerFactory::create(params);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(texts(result.value()->process("getValue café")),
            (std::vector<std::string>{"getvalue", "get", "value", "ca", "fé"}));
  std::string long_name(8192, 'a');
  EXPECT_EQ(texts(result.value()->process(long_name)),
            std::vector<std::string>{long_name});
  for (
      const std::string &json :
      {R"({"sub_tokenizer":"code"})", R"({"sub_tokenizer":"missing"})",
       R"({"sub_tokenizer":42})", R"({"sub_tokenizer":null})",
       R"({"sub_tokenizer_params":null})", R"({"sub_tokenizer_params":[]})",
       R"({"sub_tokenizer_params":{"max_token_length":0}})",
       R"({"sub_tokenizer":"jieba","sub_tokenizer_params":{"cut_mode":"invalid"}})"}) {
    SCOPED_TRACE(json);
    params.extra_params = json;
    EXPECT_FALSE(TokenizerFactory::create(params).has_value());
  }
}

TEST(CodeTokenizerTest, CacheSeparatesChildParameters) {
  FtsIndexParams normal;
  normal.tokenizer_name = "code";
  normal.filters.clear();
  FtsIndexParams short_words = normal;
  short_words.extra_params =
      R"({"sub_tokenizer_params":{"max_token_length":2}})";
  auto &manager = TokenizerPipelineManager::Instance();
  auto a = manager.acquire(normal);
  auto b = manager.acquire(short_words);
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  EXPECT_NE(a, b);
  EXPECT_EQ(texts(a->process("caféValue")),
            std::vector<std::string>{"cafévalue"});
  EXPECT_EQ(texts(b->process("caféValue")),
            (std::vector<std::string>{"ca", "fé", "va", "lu", "e"}));
  manager.release(normal);
  manager.release(short_words);
}

TEST(CodeTokenizerTest, JiebaChildPreservesWordsAndGlobalOffsets) {
  ailego::JsonObject child_config;
  ailego::JsonObject config;
  config.set("sub_tokenizer", "jieba");
  config.set("sub_tokenizer_params", child_config);
  CodeTokenizer tokenizer;
  ASSERT_TRUE(tokenizer.init(config).ok());
  auto tokens = tokenizer.tokenize("获取请求时间 getRequestTime，调用");
  EXPECT_EQ(texts(tokens),
            (std::vector<std::string>{"获取", "请求", "时间", "getrequesttime",
                                      "get", "request", "time", "调用"}));
  ASSERT_EQ(tokens.size(), 8u);
  EXPECT_EQ(tokens[3].offset, 19u);
  EXPECT_EQ(tokens[5].offset, 22u);
  EXPECT_EQ(tokens[7].offset, 36u);
  for (size_t i = 0; i < tokens.size(); ++i) {
    EXPECT_EQ(tokens[i].position, i);
  }

  JiebaTokenizer child;
  ASSERT_TRUE(child.init(child_config).ok());
  for (const auto &text :
       {"南京市长江大桥", "caféValue", "한국어테스트", "Приветмир"}) {
    EXPECT_EQ(texts(tokenizer.tokenize(text)),
              texts(LowercaseTokenFilter().filter(child.tokenize(text))));
  }
}

TEST(CodeTokenizerTest, JiebaChildSupportsExternalDictionary) {
  const std::string dict_dir{JIEBA_DICT_DIR};
  for (const char *file : {"jieba.dict.utf8", "hmm_model.utf8"}) {
    const std::string path = dict_dir + "/" + file;
    if (!std::ifstream(path).good()) {
      GTEST_SKIP() << "Jieba resource not available at: " << path;
    }
  }
  ailego::JsonObject config;
  config.set("sub_tokenizer", "jieba");
  CodeTokenizer embedded;
  ASSERT_TRUE(embedded.init(config).ok());
  ailego::JsonObject child_config;
  child_config.set("jieba_dict_dir", dict_dir);
  config.set("sub_tokenizer_params", child_config);
  CodeTokenizer external;
  ASSERT_TRUE(external.init(config).ok());
  const std::string input = "获取请求时间getRequestTime，调用";
  const auto expected = embedded.tokenize(input);
  const auto actual = external.tokenize(input);
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_EQ(actual[i].text, expected[i].text);
    EXPECT_EQ(actual[i].offset, expected[i].offset);
    EXPECT_EQ(actual[i].position, expected[i].position);
  }
}

}  // namespace
}  // namespace zvec::fts
