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
#include <vector>
#include "token_filter.h"
#include "tokenizer.h"
#include "../fts_types.h"

namespace zvec::fts {

/*! Tokenizer pipeline: contains one tokenizer and a set of token filters
 *  Execution order: tokenizer → filter[0] → filter[1] → ...
 */
class TokenizerPipeline {
 public:
  TokenizerPipeline(TokenizerPtr tokenizer, std::vector<TokenFilterPtr> filters)
      : tokenizer_(std::move(tokenizer)), filters_(std::move(filters)) {}

  /*! Tokenize text and apply all filters
   */
  std::vector<Token> process(const std::string &text) const;

 private:
  TokenizerPtr tokenizer_;
  std::vector<TokenFilterPtr> filters_;
};

using TokenizerPipelinePtr = std::shared_ptr<TokenizerPipeline>;

/*! Tokenizer factory
 *  Create TokenizerPipeline based on FtsIndexParams configuration.
 */
class TokenizerFactory {
 public:
  /*! Create tokenizer pipeline from FtsIndexParams.
   *  \param params  FTS index parameters containing tokenizer_name, filters,
   *                 and extra_params (JSON string for tokenizer-specific
   *                 configuration).
   *  \return        Tokenizer pipeline, returns nullptr on failure
   */
  static TokenizerPipelinePtr create(const FtsIndexParams &params);

 private:
  static TokenizerPtr create_tokenizer(const std::string &tokenizer_name,
                                       const ailego::JsonObject &extra_json);
  static TokenFilterPtr create_filter(const std::string &filter_name);
};

}  // namespace zvec::fts
