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

#include <map>
#include <string>
#include <zvec/core/framework/index_storage.h>

namespace zvec::core::test {

class TestSegmentStorage : public IndexStorage {
 public:
  int init(const zvec::ailego::Params &) override {
    return 0;
  }
  int cleanup() override {
    return 0;
  }
  int open(const std::string &, bool) override {
    return 0;
  }
  int flush() override {
    return 0;
  }
  int close() override {
    return 0;
  }
  int append(const std::string &, size_t) override {
    return IndexError_NotImplemented;
  }
  void refresh(uint64_t) override {}
  uint64_t check_point() const override {
    return 0;
  }
  Segment::Pointer get(const std::string &id, int = -1) override {
    const auto it = segments.find(id);
    return it == segments.end() ? nullptr : it->second;
  }
  bool has(const std::string &id) const override {
    return segments.find(id) != segments.end();
  }
  uint32_t magic() const override {
    return 0;
  }

  std::map<std::string, Segment::Pointer> segments;
};

}  // namespace zvec::core::test
