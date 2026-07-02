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
#include <vector>
#include "token_filter.h"

namespace zvec::fts {

/*! Snowball stemmer token filter.
 *  Configure the language with extra_params key "stemmer_lang". When omitted,
 *  the language is "english".
 */
class StemmerTokenFilter : public TokenFilter {
 public:
  StemmerTokenFilter() = default;
  ~StemmerTokenFilter() override = default;

  StemmerTokenFilter(const StemmerTokenFilter &) = delete;
  StemmerTokenFilter &operator=(const StemmerTokenFilter &) = delete;

  bool init(const ailego::JsonObject &config) override;
  std::vector<Token> filter(std::vector<Token> tokens) const override;

  const char *name() const override {
    return "stemmer";
  }

 private:
  std::string language_{"english"};
};

}  // namespace zvec::fts
