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

#include <memory>
#include <string>
#include <cppjieba/QuerySegment.hpp>
#include "tokenizer.h"

namespace zvec::fts {

/*! Jieba tokenizer
 *
 *  Wraps cppjieba's low-level segmenters to provide Chinese (and mixed
 *  Chinese/English) word segmentation. Uses CutForSearch (QuerySegment) by
 *  default, which produces the finer granularity used for indexing/search.
 *
 *  After init(), the active segmenter is thread-safe for concurrent Cut
 *  calls, so tokenize() can be invoked from multiple threads.
 */
class JiebaTokenizer : public Tokenizer {
 public:
  JiebaTokenizer() = default;
  ~JiebaTokenizer() override;

  // Non-copyable
  JiebaTokenizer(const JiebaTokenizer &) = delete;
  JiebaTokenizer &operator=(const JiebaTokenizer &) = delete;

  // JSON config keys:
  //   "jieba_dict_dir" - directory containing jieba.dict.utf8 + hmm_model.utf8
  //   "user_dict_path" - optional user.dict.utf8
  //   "cut_mode"       - "search" (default) | "mix" | "full" | "hmm"
  //
  // jieba_dict_dir resolution: per-field > ZVEC_JIEBA_DICT_DIR >
  // zvec::GlobalConfig::jieba_dict_dir() (set by SDK on import or via init).
  // Stop-word filtering belongs to a TokenFilter, not here.
  bool init(const ailego::JsonObject &config) override;

  std::vector<Token> tokenize(const std::string &text) const override;

  const char *name() const override {
    return "jieba";
  }

  bool is_valid() const {
    return initialized_;
  }

  // Move-only (unique_ptr members)
  JiebaTokenizer(JiebaTokenizer &&) = default;
  JiebaTokenizer &operator=(JiebaTokenizer &&) = default;

 private:
  enum class CutMode { kSearch, kMix, kFull, kHmm };

  // Release segmenters first (they hold raw pointers into dict_trie_ /
  // hmm_model_), then release the underlying dict/model.
  void reset();

  // Declared before segmenters: reverse-order destruction keeps the raw
  // pointers held by segmenters valid until the segmenters die.
  std::unique_ptr<cppjieba::DictTrie> dict_trie_;
  std::unique_ptr<cppjieba::HMMModel> hmm_model_;
  std::unique_ptr<cppjieba::QuerySegment> query_seg_;
  std::unique_ptr<cppjieba::MixSegment> mix_seg_;
  std::unique_ptr<cppjieba::FullSegment> full_seg_;
  std::unique_ptr<cppjieba::HMMSegment> hmm_seg_;

  CutMode cut_mode_{CutMode::kSearch};
  bool initialized_{false};
};

}  // namespace zvec::fts
