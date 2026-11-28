// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace zvec::sqlengine {

struct RankedDoc {
  size_t segment;
  uint64_t ordinal;
  int32_t row;
  float score;
  size_t group{0};
};

// Only compact row references survive global selection. The query workspace is
// bounded by topk; no result batches or payloads are retained.
class TopKCandidates {
 public:
  struct Better {
    bool descending;
    bool operator()(const RankedDoc &a, const RankedDoc &b) const {
      const bool a_nan = std::isnan(a.score), b_nan = std::isnan(b.score);
      // Match Arrow's default AtEnd placement for NaNs in either direction.
      if (a_nan != b_nan) return !a_nan;
      if (!a_nan && a.score != b.score)
        return descending ? a.score > b.score : a.score < b.score;
      // Stable ties follow SegmentNode's reverse segment traversal, independent
      // of the order in which concurrent segment searches finish.
      if (a.segment != b.segment) return a.segment > b.segment;
      return a.ordinal < b.ordinal;
    }
  };

  TopKCandidates(size_t limit, bool descending)
      : limit_(limit), better_{descending} {}

  void offer(RankedDoc candidate) {
    if (!limit_) return;
    if (rows_.size() < limit_) {
      if (rows_.size() == rows_.capacity()) {
        const size_t capacity = rows_.capacity();
        rows_.reserve(
            capacity > limit_ / 2
                ? limit_
                : std::min(limit_, std::max(size_t{16}, capacity * 2)));
      }
      rows_.push_back(candidate);
      std::push_heap(rows_.begin(), rows_.end(), better_);
    } else if (better_(candidate, rows_.front())) {
      std::pop_heap(rows_.begin(), rows_.end(), better_);
      rows_.back() = candidate;
      std::push_heap(rows_.begin(), rows_.end(), better_);
    }
  }

  void finish() {
    std::sort(rows_.begin(), rows_.end(), better_);
  }

  const auto &rows() const {
    return rows_;
  }

 private:
  size_t limit_;
  Better better_;
  std::vector<RankedDoc> rows_;
};

}  // namespace zvec::sqlengine
