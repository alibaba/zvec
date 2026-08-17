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

#include "tokenizer_factory.h"
#include <zvec/ailego/encoding/json/mod_json_plus.h>
#include <zvec/ailego/logger/logger.h>
#include "ascii_folding_token_filter.h"
#include "jieba_tokenizer.h"
#include "ngram_tokenizer.h"
#include "standard_tokenizer.h"
#include "stemmer_token_filter.h"
#include "whitespace_tokenizer.h"

namespace zvec::fts {

Result<TokenizerPipelinePtr> TokenizerFactory::create(
    const FtsIndexParams &params) {
  // Parse extra_params JSON string into a JsonObject.
  // Empty string is treated as an empty object; malformed JSON fails.
  ailego::JsonObject extra_json;
  if (!params.extra_params.empty()) {
    ailego::JsonValue parsed;
    if (!parsed.parse(params.extra_params.c_str())) {
      return tl::make_unexpected(Status::InvalidArgument(
          "TokenizerFactory: failed to parse extra_params JSON: ",
          params.extra_params));
    }
    if (!parsed.is_object()) {
      return tl::make_unexpected(Status::InvalidArgument(
          "TokenizerFactory: extra_params is not a JSON object: ",
          params.extra_params));
    }
    extra_json = parsed.as_object();
  }

  TokenizerPtr tokenizer;
  auto status = create_tokenizer(params.tokenizer_name, extra_json, &tokenizer);
  if (!status.ok()) {
    return tl::make_unexpected(std::move(status));
  }

  std::vector<TokenFilterPtr> filters;
  for (const auto &filter_name : params.filters) {
    TokenFilterPtr filter = create_filter(filter_name);
    if (!filter) {
      return tl::make_unexpected(Status::InvalidArgument(
          "TokenizerFactory: unknown filter name: ", filter_name));
    }
    if (!filter->init(extra_json)) {
      return tl::make_unexpected(Status::InvalidArgument(
          "TokenizerFactory: failed to init filter: ", filter_name));
    }
    filters.push_back(std::move(filter));
  }

  return std::make_shared<TokenizerPipeline>(std::move(tokenizer),
                                             std::move(filters));
}

std::vector<Token> TokenizerPipeline::process(const std::string &text) const {
  std::vector<Token> tokens = tokenizer_->tokenize(text);
  for (const auto &filter : filters_) {
    tokens = filter->filter(std::move(tokens));
  }
  return tokens;
}

Status TokenizerFactory::create_tokenizer(const std::string &tokenizer_name,
                                          const ailego::JsonObject &extra_json,
                                          TokenizerPtr *tokenizer) {
  if (tokenizer_name.empty() || tokenizer_name == "standard") {
    *tokenizer = std::make_shared<StandardTokenizer>();
  } else if (tokenizer_name == "ngram") {
    *tokenizer = std::make_shared<NGramTokenizer>();
  } else if (tokenizer_name == "jieba") {
    *tokenizer = std::make_shared<JiebaTokenizer>();
  } else if (tokenizer_name == "whitespace") {
    *tokenizer = std::make_shared<WhitespaceTokenizer>();
  } else {
    return Status::InvalidArgument("TokenizerFactory: unknown tokenizer name: ",
                                   tokenizer_name);
  }

  auto status = (*tokenizer)->init(extra_json);
  if (!status.ok()) {
    tokenizer->reset();
    return status;
  }
  return Status::OK();
}

TokenFilterPtr TokenizerFactory::create_filter(const std::string &filter_name) {
  if (filter_name == "lowercase") {
    return std::make_shared<LowercaseTokenFilter>();
  } else if (filter_name == "ascii_folding") {
    return std::make_shared<AsciiFoldingTokenFilter>();
  } else if (filter_name == "stemmer") {
    // The stemmer filter uses Snowball and defaults to "english" unless
    // extra_params overrides stemmer_lang.
    return std::make_shared<StemmerTokenFilter>();
  }
  LOG_ERROR("[TokenizerFactory] unknown filter name: %s", filter_name.c_str());
  return nullptr;
}

}  // namespace zvec::fts
