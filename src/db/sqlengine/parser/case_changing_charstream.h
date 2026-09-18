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

#include <iostream>
#include "CharStream.h"

namespace zvec::sqlengine {

using namespace antlr4;

class CaseChangingCharStream : public CharStream {
 public:
  // Constructs a new CaseChangingCharStream wrapping the given {@link
  // CharStream} forcing all characters to upper_ case or lower case.
  // @param stream_ The stream_ to wrap.
  // @param upper_ If true force each symbol to upper_ case, otherwise force to
  // lower.
  CaseChangingCharStream(CharStream *m_stream, bool m_upper) {
    stream_ = m_stream;
    upper_ = m_upper;
  }

  std::string getText(const misc::Interval &interval) override {
    return stream_->getText(interval);
  }

  void consume() override {
    stream_->consume();
  }

  size_t LA(ssize_t i) override {
    size_t c = stream_->LA(i);
    if (c <= 0) {
      return c;
    }
    if (upper_) {
      return toupper((int)c);
    }
    return tolower((int)c);
  }

  ssize_t mark() override {
    return stream_->mark();
  }

  void release(ssize_t marker) override {
    stream_->release(marker);
  }

  size_t index() override {
    return stream_->index();
  }

  void seek(size_t m_index) override {
    stream_->seek(m_index);
  }

  size_t size() override {
    return stream_->size();
  }

  std::string getSourceName() const override {
    return stream_->getSourceName();
  }

  std::string toString() const override {
    return stream_->toString();
  }

 private:
  CharStream *stream_;
  bool upper_ = true;
};

}  // namespace zvec::sqlengine
