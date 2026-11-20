// Copyright 2026 The Stile Authors
// SPDX-License-Identifier: Apache-2.0
// Adapted from Stile v0.1.0 for zvec.

#pragma once

#include <cstring>
#include <string>
#include <string_view>
#include <utility>
#include <arrow/compute/api_scalar.h>
#include <arrow/result.h>

namespace zvec::stile::internal {

// Interpret case-sensitive LIKE patterns that reduce to a substring, prefix
// or suffix search under Arrow's rules. Other patterns are handled by Arrow.
class LikeMatcher {
 public:
  enum class Kind { kSubstring, kPrefix, kSuffix };

  // `is_utf8` selects UTF-8 (string types) or Latin-1 (binary types).
  static arrow::Result<LikeMatcher> Make(
      const arrow::compute::MatchSubstringOptions &options, bool is_utf8);

  bool match(std::string_view value) const {
    const size_t n = literal_.size();
    switch (kind_) {
      case Kind::kSubstring:
        if (n == 1) {
          return value.size() > 0 &&
                 std::memchr(value.data(), literal_[0], value.size());
        }
        return value.find(literal_) != std::string_view::npos;
      case Kind::kPrefix:
        return value.size() >= n && Equal(value.data(), literal_.data(), n);
      case Kind::kSuffix:
        return value.size() >= n &&
               Equal(value.data() + value.size() - n, literal_.data(), n);
    }
    return false;
  }

  Kind kind() const {
    return kind_;
  }

 private:
  LikeMatcher(Kind kind, std::string literal)
      : kind_(kind), literal_(std::move(literal)) {}

  // Short literals compare without a call or early exit.
  static bool Equal(const char *a, const char *b, size_t n) {
    if (n > 16) return std::memcmp(a, b, n) == 0;
    unsigned diff = 0;
    for (size_t i = 0; i < n; ++i) {
      diff |=
          static_cast<unsigned char>(a[i]) ^ static_cast<unsigned char>(b[i]);
    }
    return diff == 0;
  }

  Kind kind_;
  std::string literal_;
};

}  // namespace zvec::stile::internal
