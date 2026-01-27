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

#include <zvec/core/framework/index_framework.h>

namespace zvec {
namespace core {

class Seeker {
 public:
  struct Document {
    uint32_t index;
    float score;

    //! Constructor
    Document(void) : index(0), score(0.0f) {}

    //! Constructor
    Document(uint32_t i, float v) : index(i), score(v) {}

    //! Constructor
    Document(const Document &rhs) : index(rhs.index), score(rhs.score) {}

    //! Assignment
    Document &operator=(const Document &rhs) {
      index = rhs.index;
      score = rhs.score;
      return *this;
    }

    //! Less than
    bool operator<(const Document &rhs) const {
      return (this->score < rhs.score);
    }

    //! Greater than
    bool operator>(const Document &rhs) const {
      return (this->score > rhs.score);
    }
  };

 public:
  //! Destructor
  virtual ~Seeker(void) {}

  virtual int init(const IndexMeta &meta) = 0;

  virtual int cleanup(void) = 0;

  virtual int reset(void) = 0;

  virtual int mount(IndexFeatures::Pointer feats) = 0;

  virtual int seek(const void *query, size_t len, Document *out) = 0;

  virtual IndexFeatures::Pointer original(void) const = 0;
};

}  // namespace core
}  // namespace zvec
