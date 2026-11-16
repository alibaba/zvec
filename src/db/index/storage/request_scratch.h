// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <vector>

namespace zvec::detail {

// Bounded request-local stack storage. Larger requests keep the ordinary
// allocation path. Never retain this storage in a result or cache.
template <typename T, size_t InlineCount>
class RequestScratchArray {
  static_assert(InlineCount > 0 && InlineCount <= 2048 / sizeof(T),
                "request scratch must have a bounded inline footprint");

 public:
  explicit RequestScratchArray(size_t size) : size_(size) {
    if (size > InlineCount) overflow_.resize(size);
  }
  RequestScratchArray(const RequestScratchArray &) = delete;
  RequestScratchArray &operator=(const RequestScratchArray &) = delete;

  size_t size() const noexcept {
    return size_;
  }
  T *data() noexcept {
    return size_ <= InlineCount ? inline_.data() : overflow_.data();
  }
  const T *data() const noexcept {
    return size_ <= InlineCount ? inline_.data() : overflow_.data();
  }
  T &operator[](size_t index) noexcept {
    return data()[index];
  }
  const T &operator[](size_t index) const noexcept {
    return data()[index];
  }
  T *begin() noexcept {
    return data();
  }
  T *end() noexcept {
    return data() + size_;
  }
  const T *begin() const noexcept {
    return data();
  }
  const T *end() const noexcept {
    return data() + size_;
  }

 private:
  const size_t size_;
  std::array<T, InlineCount> inline_{};
  std::vector<T> overflow_;
};

}  // namespace zvec::detail
