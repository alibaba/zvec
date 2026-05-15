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

#pragma once

#include <string>
#include "tokenizer.h"

namespace cppjieba {
class Jieba;
}  // namespace cppjieba

namespace zvec::fts {

/*! Jieba tokenizer
 *
 *  Wraps cppjieba to provide Chinese (and mixed Chinese/English) word
 *  segmentation.  Uses CutForSearch mode by default which produces finer
 *  granularity suitable for search/indexing scenarios.
 *
 *  The cppjieba::Jieba instance is thread-safe for concurrent Cut* calls
 *  after construction, so tokenize() can be called from multiple threads.
 *
 *  JSON configuration keys (passed to init()):
 *    "dict_path"      – path to jieba.dict.utf8 (optional, has default)
 *    "model_path"     – path to hmm_model.utf8 (optional, has default)
 *    "user_dict_path" – path to user.dict.utf8 (optional, has default)
 *    "idf_path"       – path to idf.utf8 (optional, has default)
 *    "stop_word_path" – path to stop_words.utf8 (optional, has default)
 *    "cut_mode"       – "search" (default) | "mix" | "full" | "hmm"
 */
class JiebaTokenizer : public Tokenizer {
 public:
  JiebaTokenizer() = default;
  ~JiebaTokenizer() override;

  // Non-copyable
  JiebaTokenizer(const JiebaTokenizer &) = delete;
  JiebaTokenizer &operator=(const JiebaTokenizer &) = delete;

  bool init(const ailego::JsonObject &config) override;

  std::vector<Token> tokenize(const std::string &text) const override;

  const char *name() const override {
    return "jieba";
  }

  bool is_valid() const {
    return jieba_ != nullptr;
  }

 private:
  enum class CutMode { kSearch, kMix, kFull, kHmm };

  cppjieba::Jieba *jieba_{nullptr};
  CutMode cut_mode_{CutMode::kSearch};
};

}  // namespace zvec::fts
