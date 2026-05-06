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
#include <functional>
#include <memory>


namespace zvec {


class IndexFilter {
 public:
  using Ptr = std::shared_ptr<IndexFilter>;

  IndexFilter() = default;

  virtual ~IndexFilter() = default;

  IndexFilter(const IndexFilter &) = delete;

  IndexFilter &operator=(const IndexFilter &) = delete;

  /**
   * @return true if the document is filtered (should be excluded)
   * @return false if the document is not filtered (should be included)
   */
  virtual bool is_filtered(uint64_t id) const = 0;
};

class EasyIndexFilter : public IndexFilter {
 public:
  using FilterFunction = std::function<bool(uint64_t)>;

  /**
   * Create an IndexFilter::Ptr from a lambda expression or function
   * @param filter_func A function that takes a uint64_t id and returns true
   *                    if the document should be filtered (excluded)
   */
  static IndexFilter::Ptr Create(FilterFunction filter_func) {
    return std::make_shared<EasyIndexFilter>(std::move(filter_func));
  }

  /**
   * Constructor that takes a filter function
   * @param filter_func A function that takes a uint64_t id and returns true
   *                    if the document should be filtered (excluded)
   */
  explicit EasyIndexFilter(FilterFunction filter_func)
      : filter_func_(std::move(filter_func)) {}

  /**
   * Check if a document should be filtered
   * @param id The document ID
   * @return true if the document should be filtered (excluded)
   * @return false if the document should not be filtered (included)
   */
  bool is_filtered(uint64_t id) const override {
    return filter_func_(id);
  }

 private:
  FilterFunction filter_func_;
};


}  // namespace zvec