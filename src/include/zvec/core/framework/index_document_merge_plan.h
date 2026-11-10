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
#include <vector>
#include <zvec/ailego/internal/platform.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_filter.h>

namespace zvec::core {
// Complete document ranges are used only during merge. Vector providers keep
// yielding real vectors; rank in these cached decisions gives their output ID.
class IndexDocumentMergePlan {
 public:
  using Pointer = std::shared_ptr<const IndexDocumentMergePlan>;
  int init(const std::vector<uint32_t> &rows, uint32_t output_base,
           const IndexFilter &filter, bool preserve_empty_rows = true) {
    preserve_empty_rows_ = preserve_empty_rows;
    uint64_t input_base = 0, output = output_base;
    for (uint32_t count : rows) {
      Source source;
      source.rows = count;
      source.output_base = output;
      if (!preserve_empty_rows_) {
        sources_.push_back(std::move(source));
        continue;
      }
      source.keep.resize((uint64_t(count) + 63) / 64);
      source.prefix.resize(source.keep.size());
      uint32_t kept = 0;
      for (uint64_t id = 0; id < count; ++id) {
        if (id % 64 == 0) source.prefix[id / 64] = kept;
        if (!filter(input_base + id)) {
          if (output == UINT32_MAX) return IndexError_Overflow;
          source.keep[id / 64] |= uint64_t{1} << (id % 64);
          ++kept;
          ++output;
        }
      }
      input_base += count;
      sources_.push_back(std::move(source));
    }
    output_end_ = output;
    return 0;
  }
  bool preserves_empty_rows() const {
    return preserve_empty_rows_;
  }
  uint32_t rows(size_t source) const {
    return sources_.at(source).rows;
  }
  uint32_t output_end() const {
    return output_end_;
  }
  uint32_t output_id(size_t source, uint64_t id) const {
    if (!preserve_empty_rows_) return UINT32_MAX;
    const auto &s = sources_.at(source);
    if (id >= s.rows) return UINT32_MAX;
    const uint64_t bit = uint64_t{1} << (id % 64);
    const uint64_t word = s.keep[id / 64];
    if (!(word & bit)) return UINT32_MAX;
    return s.output_base + s.prefix[id / 64] +
           ailego_popcount64(word & (bit - 1));
  }

 private:
  struct Source {
    uint32_t rows{0}, output_base{0};
    std::vector<uint64_t> keep;
    std::vector<uint32_t> prefix;
  };
  std::vector<Source> sources_;
  uint32_t output_end_{0};
  bool preserve_empty_rows_{true};
};
}  // namespace zvec::core
