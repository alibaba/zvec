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

#include "jieba_tokenizer.h"
#include <zvec/ailego/logger/logger.h>
#include "cppjieba/Jieba.hpp"

namespace zvec::fts {

static std::string get_string_or_default(const ailego::JsonObject &config,
                                         const char *key,
                                         const std::string &default_value) {
  auto val = config[key];
  if (val.is_string()) {
    std::string result = val.as_string().c_str();
    if (!result.empty()) {
      return result;
    }
  }
  return default_value;
}

bool JiebaTokenizer::init(const ailego::JsonObject &config) {
  std::string dict_path = get_string_or_default(config, "dict_path", "");
  if (dict_path.empty()) {
    LOG_ERROR("JiebaTokenizer: 'dict_path' is required but not provided");
    return false;
  }
  std::string model_path = get_string_or_default(config, "model_path", "");
  if (model_path.empty()) {
    LOG_ERROR("JiebaTokenizer: 'model_path' is required but not provided");
    return false;
  }
  std::string user_dict_path =
      get_string_or_default(config, "user_dict_path", "");
  std::string idf_path = get_string_or_default(config, "idf_path", "");
  std::string stop_word_path =
      get_string_or_default(config, "stop_word_path", "");

  // Parse cut mode
  std::string mode_str = get_string_or_default(config, "cut_mode", "search");
  if (mode_str == "search") {
    cut_mode_ = CutMode::kSearch;
  } else if (mode_str == "mix") {
    cut_mode_ = CutMode::kMix;
  } else if (mode_str == "full") {
    cut_mode_ = CutMode::kFull;
  } else if (mode_str == "hmm") {
    cut_mode_ = CutMode::kHmm;
  } else {
    LOG_ERROR("JiebaTokenizer: unknown cut_mode '%s'", mode_str.c_str());
    return false;
  }

  // Release any previously initialised handle
  jieba_.reset();

  try {
    jieba_ = std::make_unique<cppjieba::Jieba>(
        dict_path, model_path, user_dict_path, idf_path, stop_word_path);
  } catch (const std::exception &e) {
    LOG_ERROR("JiebaTokenizer init failed: %s", e.what());
    jieba_.reset();
    return false;
  }

  LOG_INFO(
      "JiebaTokenizer init success. dict_path[%s] model_path[%s] "
      "cut_mode[%s]",
      dict_path.c_str(), model_path.c_str(), mode_str.c_str());
  return true;
}

JiebaTokenizer::~JiebaTokenizer() = default;

std::vector<Token> JiebaTokenizer::tokenize(const std::string &text) const {
  std::vector<Token> tokens;
  if (!jieba_ || text.empty()) {
    return tokens;
  }

  std::vector<cppjieba::Word> words;
  switch (cut_mode_) {
    case CutMode::kSearch:
      jieba_->CutForSearch(text, words, true);
      break;
    case CutMode::kMix:
      jieba_->Cut(text, words, true);
      break;
    case CutMode::kFull:
      jieba_->CutAll(text, words);
      break;
    case CutMode::kHmm:
      jieba_->CutHMM(text, words);
      break;
    default:
      LOG_ERROR("JiebaTokenizer: unexpected cut_mode %d",
                static_cast<int>(cut_mode_));
      return tokens;
  }

  tokens.reserve(words.size());
  // CutForSearch (and other cut modes) emit overlapping sub-words right after
  // their long parent word. Using the cppjieba unicode_offset as position
  // breaks PhraseDocIterator's strict anchor+1 adjacency check because
  // overlapping tokens share a unicode_offset and gaps appear between long
  // words. Use the output sequence index instead so doc and query tokenized
  // with the same cut_mode produce contiguous, monotonically increasing
  // positions, which makes phrase matching land on the same subsequence.
  uint32_t seq = 0;
  for (const auto &word : words) {
    if (word.word.empty()) {
      continue;
    }
    Token token;
    token.text = word.word;
    token.offset = word.offset;
    token.position = seq++;
    tokens.push_back(std::move(token));
  }

  return tokens;
}

}  // namespace zvec::fts
