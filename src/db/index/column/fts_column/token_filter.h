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
#include <unordered_set>
#include <vector>
#include "tokenizer.h"

namespace zvec::fts {

/*! Token Filter abstract interface
 *  Post-process tokenization results, such as case conversion, stopword
 * filtering, etc.
 */
class TokenFilter {
 public:
  virtual ~TokenFilter() = default;

  /*! 对 token 列表进行过滤/变换
   *  \param tokens  输入 token 列表（可原地修改）
   *  \return        处理后的 token 列表
   */
  virtual std::vector<Token> filter(std::vector<Token> tokens) const = 0;

  /*! Return filter name
   */
  virtual const char *name() const = 0;
};

using TokenFilterPtr = std::shared_ptr<TokenFilter>;

/*! Lowercase Token Filter
 *  Convert all token text to lowercase (only handles ASCII characters)
 */
class LowercaseTokenFilter : public TokenFilter {
 public:
  std::vector<Token> filter(std::vector<Token> tokens) const override;

  const char *name() const override {
    return "lowercase";
  }
};

/*! Stopword Token Filter
 *  Drop tokens whose text matches any entry in the configured stopword set.
 *  The offset and position of remaining tokens are preserved as-is, so that
 *  positional structures (e.g. phrase queries) keep their original gaps.
 *  Matching is byte-wise exact; combine with LowercaseTokenFilter beforehand
 *  if case-insensitive matching is desired.
 */
class StopwordTokenFilter : public TokenFilter {
 public:
  explicit StopwordTokenFilter(std::unordered_set<std::string> stopwords)
      : stopwords_(std::move(stopwords)) {}

  std::vector<Token> filter(std::vector<Token> tokens) const override;

  const char *name() const override {
    return "stopword";
  }

  const std::unordered_set<std::string> &stopwords() const {
    return stopwords_;
  }

 private:
  std::unordered_set<std::string> stopwords_;
};

}  // namespace zvec::fts
