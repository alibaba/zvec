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
#include "BaseErrorListener.h"

namespace zvec::sqlengine {

using namespace antlr4;

#define UNUSED(x) (void)x

class ErrorVerboseListener : BaseErrorListener {
 public:
  ErrorVerboseListener() = default;
  ~ErrorVerboseListener() override = default;

  void syntaxError(Recognizer *recognizer, Token *offending_symbol, size_t line,
                   size_t char_position_in_line, const std::string &msg,
                   std::exception_ptr e) override {
    UNUSED(recognizer);
    UNUSED(offending_symbol);
    UNUSED(e);

    err_msg_ = std::to_string(line) + " " +
               std::to_string(char_position_in_line) + " " + msg;
    return;
  }

  const std::string &err_msg() {
    return err_msg_;
  }

 private:
  std::string err_msg_;
};

}  // namespace zvec::sqlengine
