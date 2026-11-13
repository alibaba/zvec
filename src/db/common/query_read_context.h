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

#include <algorithm>
#include <memory>
#include <vector>

namespace zvec {

class CollectionImpl;
class Segment;
namespace test {
class ForwardFilterReadTest;
}

// Internal capability, issued only while a read-only collection's query holds
// its schema lock. All execution/readers must finish before that lock is
// released. This value can travel to workers; the thread-affine lock cannot.
// A default context never authorizes a forward view, including direct SQL
// callers.
class QueryReadContext {
 public:
  QueryReadContext() = default;

  bool allows_forward_view(const Segment *segment) const {
    if (!segments_) {
      return false;
    }
    return std::any_of(
        segments_->begin(), segments_->end(),
        [segment](const auto &owned) { return owned.get() == segment; });
  }

 private:
  friend class CollectionImpl;
  friend class test::ForwardFilterReadTest;

  QueryReadContext(bool collection_read_only,
                   const std::vector<std::shared_ptr<Segment>> &segments) {
    if (collection_read_only) {
      segments_ = std::make_shared<const SegmentList>(segments);
    }
  }

  using SegmentList = std::vector<std::shared_ptr<Segment>>;
  std::shared_ptr<const SegmentList> segments_;
};

}  // namespace zvec
