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

#include "db/index/column/fts_column/tokenizer/jieba_tokenizer.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include "db/index/column/fts_column/tokenizer/jieba_embedded_dict.h"

namespace {

using zvec::ailego::JsonObject;
using zvec::fts::EmbeddedJiebaDict;
using zvec::fts::EmbeddedJiebaHmmModel;
using zvec::fts::JiebaTokenizer;
using zvec::fts::Token;

constexpr char kDictEnv[] = "ZVEC_JIEBA_DICT_DIR";
const std::string kDictDir{JIEBA_DICT_DIR};

int set_dict_env(const char *value) {
#ifdef _WIN32
  return _putenv_s(kDictEnv, value ? value : "");
#else
  return value ? setenv(kDictEnv, value, 1) : unsetenv(kDictEnv);
#endif
}

JsonObject make_config(const std::string &mode = "search",
                       const std::string &dict_dir = "",
                       const std::string &user_dict = "") {
  JsonObject config;
  config["cut_mode"] = mode.c_str();
  config["jieba_dict_dir"] = dict_dir.c_str();
  config["user_dict_path"] = user_dict.c_str();
  return config;
}

void expect_same_tokens(const std::vector<Token> &actual,
                        const std::vector<Token> &expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    SCOPED_TRACE(i);
    EXPECT_EQ(actual[i].text, expected[i].text);
    EXPECT_EQ(actual[i].offset, expected[i].offset);
    EXPECT_EQ(actual[i].position, expected[i].position);
  }
}

bool contains_token(const std::vector<Token> &tokens, const std::string &word) {
  return std::any_of(tokens.begin(), tokens.end(),
                     [&](const Token &token) { return token.text == word; });
}

void expect_invalid_init(const JsonObject &config) {
  JiebaTokenizer tokenizer;
  const auto status = tokenizer.init(config);
  EXPECT_EQ(status.code(), zvec::StatusCode::INVALID_ARGUMENT) << status;
  EXPECT_FALSE(tokenizer.is_valid());
  EXPECT_TRUE(tokenizer.tokenize("中文分词技术").empty());
}

class JiebaEmbeddedTokenizerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const char *env = std::getenv(kDictEnv);
    had_env_ = env != nullptr;
    if (had_env_) {
      saved_env_ = env;
    }
    ASSERT_EQ(set_dict_env(nullptr), 0);

    static std::atomic<uint64_t> sequence{0};
    const auto timestamp = std::chrono::steady_clock::now().time_since_epoch();
    std::error_code error;
    const auto temp_root = std::filesystem::temp_directory_path(error);
    ASSERT_FALSE(error) << error.message();
    temp_dir_ =
        temp_root / ("zvec_jieba_test_" + std::to_string(timestamp.count()) +
                     "_" + std::to_string(sequence.fetch_add(1)));
    ASSERT_TRUE(std::filesystem::create_directory(temp_dir_, error))
        << error.message();
  }

  void TearDown() override {
    EXPECT_EQ(set_dict_env(had_env_ ? saved_env_.c_str() : nullptr), 0);
    if (!temp_dir_.empty()) {
      std::error_code error;
      std::filesystem::remove_all(temp_dir_, error);
      EXPECT_FALSE(error) << error.message();
    }
  }

  std::string write_user_dict(const std::string &filename,
                              const std::string &word) {
    const auto path = temp_dir_ / filename;
    std::ofstream output(path, std::ios::binary);
    output << word << " 100000000 nz\n";
    output.close();
    EXPECT_TRUE(output.good());
    return path.string();
  }

  std::filesystem::path temp_dir_;

 private:
  std::string saved_env_;
  bool had_env_{false};
};

// Register this test first so a normal run races the first cache
// initialization.
TEST_F(JiebaEmbeddedTokenizerTest,
       ConcurrentFirstInitializationSharesDefaults) {
  constexpr size_t kThreadCount = 8;
  std::vector<EmbeddedJiebaDict> dictionaries(kThreadCount);
  std::vector<EmbeddedJiebaHmmModel> models(kThreadCount);
  std::atomic<size_t> ready{0};
  std::atomic<bool> start{false};
  std::vector<std::thread> threads;
  for (size_t i = 0; i < kThreadCount; ++i) {
    threads.emplace_back([&, i]() {
      ready.fetch_add(1);
      while (!start.load()) {
        std::this_thread::yield();
      }
      auto dict = zvec::fts::get_embedded_jieba_dict();
      auto model = zvec::fts::get_embedded_jieba_hmm_model();
      if (!dict.has_value() || !model.has_value()) {
        return;
      }
      dictionaries[i] = dict.value();
      models[i] = model.value();
    });
  }
  while (ready.load() != kThreadCount) {
    std::this_thread::yield();
  }
  start.store(true);
  for (auto &thread : threads) {
    thread.join();
  }

  ASSERT_NE(dictionaries[0], nullptr);
  ASSERT_NE(models[0], nullptr);
  for (size_t i = 0; i < kThreadCount; ++i) {
    EXPECT_EQ(dictionaries[i].get(), dictionaries[0].get());
    EXPECT_EQ(models[i].get(), models[0].get());
  }
}

TEST_F(JiebaEmbeddedTokenizerTest, DefaultWorksWithoutExternalDictDir) {
  JiebaTokenizer tokenizer;
  ASSERT_TRUE(tokenizer.init(JsonObject{}).ok());
  ASSERT_TRUE(tokenizer.is_valid());

  const auto tokens = tokenizer.tokenize("中文分词技术");
  const std::vector<std::string> expected{"中文", "分词", "技术"};
  ASSERT_EQ(tokens.size(), expected.size());
  for (size_t i = 0; i < tokens.size(); ++i) {
    EXPECT_EQ(tokens[i].text, expected[i]);
    EXPECT_EQ(tokens[i].offset, i * 6);
    EXPECT_EQ(tokens[i].position, i);
  }
  EXPECT_TRUE(tokenizer.tokenize("").empty());
}

TEST_F(JiebaEmbeddedTokenizerTest, AllModesMatchExternalDictionaryAndModel) {
  ASSERT_TRUE(std::ifstream(kDictDir + "/jieba.dict.utf8").good());
  ASSERT_TRUE(std::ifstream(kDictDir + "/hmm_model.utf8").good());
  const std::vector<std::string> texts{"",
                                       "中文分词技术",
                                       "中华人民共和国成立",
                                       "南京市长江大桥",
                                       "hello abc123 Zvec",
                                       "星河咕噜引擎",
                                       "北京abc123星河咕噜引擎，上海!"};
  for (const std::string mode : {"search", "mix", "full", "hmm"}) {
    SCOPED_TRACE(mode);
    JiebaTokenizer embedded;
    JiebaTokenizer external;
    ASSERT_TRUE(embedded.init(make_config(mode)).ok());
    ASSERT_TRUE(external.init(make_config(mode, kDictDir)).ok());
    for (const auto &text : texts) {
      SCOPED_TRACE(text);
      const auto tokens = embedded.tokenize(text);
      expect_same_tokens(tokens, external.tokenize(text));
      for (size_t i = 0; i < tokens.size(); ++i) {
        EXPECT_EQ(tokens[i].position, i);
        EXPECT_EQ(text.substr(tokens[i].offset, tokens[i].text.size()),
                  tokens[i].text);
      }
    }
  }
}

TEST_F(JiebaEmbeddedTokenizerTest,
       UserDictionariesAreIsolatedBetweenInstances) {
  const std::string first_word = "星河咕噜引擎";
  const std::string second_word = "云端泡泡火车";
  const std::string text = first_word + "和" + second_word;
  std::vector<Token> original_tokens;
  {
    JiebaTokenizer original;
    ASSERT_TRUE(original.init(make_config("mix")).ok());
    original_tokens = original.tokenize(text);
  }
  ASSERT_FALSE(contains_token(original_tokens, first_word));
  ASSERT_FALSE(contains_token(original_tokens, second_word));

  {
    JiebaTokenizer first;
    JiebaTokenizer second;
    ASSERT_TRUE(first
                    .init(make_config(
                        "mix", "", write_user_dict("first.utf8", first_word)))
                    .ok());
    ASSERT_TRUE(second
                    .init(make_config(
                        "mix", "", write_user_dict("second.utf8", second_word)))
                    .ok());
    EXPECT_TRUE(contains_token(first.tokenize(text), first_word));
    EXPECT_FALSE(contains_token(first.tokenize(text), second_word));
    EXPECT_TRUE(contains_token(second.tokenize(text), second_word));
    EXPECT_FALSE(contains_token(second.tokenize(text), first_word));
  }

  // A later instance must also see the original shared base dictionary.
  JiebaTokenizer later;
  ASSERT_TRUE(later.init(make_config("mix")).ok());
  expect_same_tokens(later.tokenize(text), original_tokens);
}

TEST_F(JiebaEmbeddedTokenizerTest, SharedTokenizerSupportsConcurrentTokenize) {
  JiebaTokenizer tokenizer;
  ASSERT_TRUE(tokenizer.init(make_config()).ok());
  const std::string text = "中华人民共和国abc123星河咕噜引擎";
  const auto expected = tokenizer.tokenize(text);
  ASSERT_FALSE(expected.empty());
  constexpr size_t kThreadCount = 8;
  constexpr size_t kRepeatCount = 32;
  std::vector<std::vector<std::vector<Token>>> outputs(kThreadCount);
  std::vector<std::thread> threads;
  for (size_t i = 0; i < kThreadCount; ++i) {
    threads.emplace_back([&, i]() {
      for (size_t repeat = 0; repeat < kRepeatCount; ++repeat) {
        outputs[i].push_back(tokenizer.tokenize(text));
      }
    });
  }
  for (auto &thread : threads) {
    thread.join();
  }
  for (const auto &thread_output : outputs) {
    ASSERT_EQ(thread_output.size(), kRepeatCount);
    for (const auto &tokens : thread_output) {
      expect_same_tokens(tokens, expected);
    }
  }
}

TEST_F(JiebaEmbeddedTokenizerTest, PerFieldBeatsEnvironmentDirectory) {
  ASSERT_EQ(set_dict_env((temp_dir_ / "missing_env_dict").string().c_str()), 0);
  JiebaTokenizer tokenizer;
  ASSERT_TRUE(tokenizer.init(make_config("search", kDictDir)).ok());
  EXPECT_TRUE(contains_token(tokenizer.tokenize("中文分词技术"), "中文"));
}

TEST_F(JiebaEmbeddedTokenizerTest, EnvironmentDirectoryBeatsEmbeddedDefaults) {
  const std::string word = "星河咕噜引擎";
  write_user_dict("jieba.dict.utf8", word);
  ASSERT_EQ(set_dict_env(temp_dir_.string().c_str()), 0);

  JiebaTokenizer tokenizer;
  ASSERT_TRUE(tokenizer.init(make_config("full")).ok());
  const auto tokens = tokenizer.tokenize(word);
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0].text, word);
}

TEST_F(JiebaEmbeddedTokenizerTest, InvalidConfiguredDirectoryDoesNotFallBack) {
  const std::string missing = (temp_dir_ / "missing_dict").string();
  ASSERT_EQ(set_dict_env(kDictDir.c_str()), 0);
  expect_invalid_init(make_config("search", missing));

  ASSERT_EQ(set_dict_env(missing.c_str()), 0);
  expect_invalid_init(make_config());
}

TEST_F(JiebaEmbeddedTokenizerTest, ExternalResourcesAreRequiredOnlyByMode) {
  const auto dict_only = temp_dir_ / "dict_only";
  const auto model_only = temp_dir_ / "model_only";
  std::error_code error;
  ASSERT_TRUE(std::filesystem::create_directory(dict_only, error));
  ASSERT_FALSE(error) << error.message();
  ASSERT_TRUE(std::filesystem::create_directory(model_only, error));
  ASSERT_FALSE(error) << error.message();
  ASSERT_TRUE(std::filesystem::copy_file(
      std::filesystem::u8path(kDictDir) / "jieba.dict.utf8",
      dict_only / "jieba.dict.utf8", error));
  ASSERT_FALSE(error) << error.message();
  ASSERT_TRUE(std::filesystem::copy_file(
      std::filesystem::u8path(kDictDir) / "hmm_model.utf8",
      model_only / "hmm_model.utf8", error));
  ASSERT_FALSE(error) << error.message();

  const std::string text = "中文分词技术星河咕噜引擎";
  JiebaTokenizer full;
  JiebaTokenizer full_default;
  ASSERT_TRUE(full.init(make_config("full", dict_only.string())).ok());
  ASSERT_TRUE(full_default.init(make_config("full")).ok());
  expect_same_tokens(full.tokenize(text), full_default.tokenize(text));

  // HMM mode does not use either a dictionary or a user dictionary.
  JiebaTokenizer hmm;
  JiebaTokenizer hmm_default;
  ASSERT_TRUE(hmm.init(make_config("hmm", model_only.string(),
                                   (temp_dir_ / "missing_user.utf8").string()))
                  .ok());
  ASSERT_TRUE(hmm_default.init(make_config("hmm")).ok());
  expect_same_tokens(hmm.tokenize(text), hmm_default.tokenize(text));

  for (const std::string mode : {"search", "mix"}) {
    SCOPED_TRACE(mode);
    expect_invalid_init(make_config(mode, dict_only.string()));
    expect_invalid_init(make_config(mode, model_only.string()));
  }
}

TEST_F(JiebaEmbeddedTokenizerTest, EmptyExternalResourcesAreRejected) {
  std::ofstream dict(temp_dir_ / "jieba.dict.utf8");
  dict.close();
  ASSERT_TRUE(dict.good());
  std::ofstream model(temp_dir_ / "hmm_model.utf8");
  model.close();
  ASSERT_TRUE(model.good());

  expect_invalid_init(make_config("full", temp_dir_.string()));
  expect_invalid_init(make_config("hmm", temp_dir_.string()));
}

TEST_F(JiebaEmbeddedTokenizerTest, MissingUserDictionaryReturnsError) {
  const std::string missing = (temp_dir_ / "missing_user.utf8").string();
  const std::string valid = write_user_dict("valid.utf8", "星河咕噜引擎");
  expect_invalid_init(make_config("mix", "", missing));
  expect_invalid_init(make_config("mix", kDictDir, missing));
  for (const std::string delimiter : {"|", ";"}) {
    SCOPED_TRACE(delimiter);
    expect_invalid_init(make_config("mix", "", valid + delimiter + missing));
  }
}

TEST_F(JiebaEmbeddedTokenizerTest,
       MultipleAndEmptyUserDictionariesAreSupported) {
  const std::string first_word = "星河咕噜引擎";
  const std::string second_word = "云端泡泡火车";
  const std::string first = write_user_dict("first.utf8", first_word);
  const std::string second = write_user_dict("second.utf8", second_word);
  const auto empty_path = temp_dir_ / "empty.utf8";
  std::ofstream empty(empty_path);
  empty.close();
  ASSERT_TRUE(empty.good());

  for (const std::string delimiter : {"|", ";"}) {
    SCOPED_TRACE(delimiter);
    JiebaTokenizer tokenizer;
    ASSERT_TRUE(tokenizer
                    .init(make_config("mix", "",
                                      first + delimiter + empty_path.string() +
                                          delimiter + second))
                    .ok());
    const auto tokens = tokenizer.tokenize(first_word + "和" + second_word);
    EXPECT_TRUE(contains_token(tokens, first_word));
    EXPECT_TRUE(contains_token(tokens, second_word));
  }
}

TEST_F(JiebaEmbeddedTokenizerTest,
       FailedReinitializationClearsStateAndRecovers) {
  const std::string text = "中文分词技术星河咕噜引擎";
  JiebaTokenizer tokenizer;
  for (const std::string mode : {"search", "mix", "full", "hmm"}) {
    SCOPED_TRACE(mode);
    ASSERT_TRUE(tokenizer.init(make_config(mode)).ok());
    const auto expected = tokenizer.tokenize(text);
    ASSERT_FALSE(expected.empty());

    EXPECT_FALSE(tokenizer.init(make_config("unknown")).ok());
    EXPECT_FALSE(tokenizer.is_valid());
    EXPECT_TRUE(tokenizer.tokenize(text).empty());
    ASSERT_TRUE(tokenizer.init(make_config(mode)).ok());
    expect_same_tokens(tokenizer.tokenize(text), expected);

    EXPECT_FALSE(
        tokenizer.init(make_config(mode, (temp_dir_ / "missing_dict").string()))
            .ok());
    EXPECT_FALSE(tokenizer.is_valid());
    EXPECT_TRUE(tokenizer.tokenize(text).empty());
    ASSERT_TRUE(tokenizer.init(make_config(mode)).ok());
    expect_same_tokens(tokenizer.tokenize(text), expected);
  }
}

TEST_F(JiebaEmbeddedTokenizerTest, UnknownCutModeFails) {
  JiebaTokenizer tokenizer;
  EXPECT_FALSE(tokenizer.init(make_config("unknown")).ok());
  EXPECT_FALSE(tokenizer.is_valid());
  EXPECT_TRUE(tokenizer.tokenize("中文分词技术").empty());
}

}  // namespace
