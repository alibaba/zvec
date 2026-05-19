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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <zvec/ailego/encoding/json/mod_json_plus.h>

namespace zvec::fts {

/*! A single token in the tokenization result
 */
struct Token {
  // token text content
  std::string text;
  // start byte offset of token in original text
  uint32_t offset{0};
  // token position in document (which word, starting from 0)
  uint32_t position{0};
};

/*! Abstract tokenizer interface
 *  All tokenizer implementations must inherit from this interface
 */
class Tokenizer {
 public:
  virtual ~Tokenizer() = default;

  /*! Initialise the tokenizer from a JSON configuration object.
   *  Must be called once before tokenize().
   *  \param config  JSON object containing tokenizer-specific parameters.
   *  \return        true on success, false on failure.
   */
  virtual bool init(const ailego::JsonObject &config) = 0;

  /*! Tokenize input text
   *  \param text  UTF-8 encoded input text
   *  \return      Tokenization result list, sorted by position in ascending
   * order
   */
  virtual std::vector<Token> tokenize(const std::string &text) const = 0;

  /*! Return tokenizer name
   */
  virtual const char *name() const = 0;
};

using TokenizerPtr = std::shared_ptr<Tokenizer>;

}  // namespace zvec::fts
