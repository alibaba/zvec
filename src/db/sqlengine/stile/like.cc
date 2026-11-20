// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
//
// The LIKE classification patterns are derived from
// Apache Arrow 21 (cpp/src/arrow/compute/kernels/scalar_string_ascii.cc),
// licensed under the Apache License 2.0. See NOTICE.

#include "db/sqlengine/stile/like.h"
#include <optional>

namespace zvec::stile::internal {
namespace {

// Well-formed UTF-8 (no overlong forms, surrogates or values past U+10FFFF).
bool StrictUtf8(const std::string &text) {
  const auto *p = reinterpret_cast<const unsigned char *>(text.data());
  const auto *end = p + text.size();
  while (p < end) {
    const unsigned char c = *p;
    int extra;
    unsigned char low = 0x80, high = 0xBF;
    if (c < 0x80) {
      ++p;
      continue;
    } else if (c >= 0xC2 && c <= 0xDF) {
      extra = 1;
    } else if (c == 0xE0) {
      extra = 2, low = 0xA0;
    } else if ((c >= 0xE1 && c <= 0xEC) || c == 0xEE || c == 0xEF) {
      extra = 2;
    } else if (c == 0xED) {
      extra = 2, high = 0x9F;
    } else if (c == 0xF0) {
      extra = 3, low = 0x90;
    } else if (c >= 0xF1 && c <= 0xF3) {
      extra = 3;
    } else if (c == 0xF4) {
      extra = 3, high = 0x8F;
    } else {
      return false;
    }
    if (end - p <= extra) return false;
    if (p[1] < low || p[1] > high) return false;
    for (int i = 2; i <= extra; ++i) {
      if (p[i] < 0x80 || p[i] > 0xBF) return false;
    }
    p += extra + 1;
  }
  return true;
}

using Kind = LikeMatcher::Kind;

std::optional<Kind> Classify(const std::string &pattern, bool is_utf8,
                             std::string *literal) {
  // Arrow's three expressions, written out. RE2's classes match code
  // points, so only well-formed UTF-8 is classified byte by byte here;
  // anything else requires Arrow's regex matching.
  literal->clear();
  if (is_utf8 && !StrictUtf8(pattern)) return std::nullopt;
  const size_t n = pattern.size();
  size_t lead = 0;
  while (lead < n && pattern[lead] == '%') ++lead;
  if (n > 0 && lead == n) {  // Only '%': matches every value.
    return n >= 2 ? Kind::kSubstring : Kind::kPrefix;
  }
  size_t trail = 0;
  while (trail < n - lead && pattern[n - 1 - trail] == '%') ++trail;
  const std::string middle = pattern.substr(lead, n - lead - trail);
  const bool plain = middle.find_first_of("%_") == std::string::npos;
  const bool no_trailing_escape = middle.empty() || middle.back() != '\\';
  std::optional<Kind> kind;
  if (plain && lead > 0 && trail > 0 && no_trailing_escape) {
    kind = Kind::kSubstring;
  } else if (plain && lead == 0 && trail > 0 && no_trailing_escape) {
    kind = Kind::kPrefix;
  } else if (plain && lead > 0 && trail == 0) {
    kind = Kind::kSuffix;
  }
  if (kind) *literal = middle;
  return kind;
}

}  // namespace

arrow::Result<LikeMatcher> LikeMatcher::Make(
    const arrow::compute::MatchSubstringOptions &options, bool is_utf8) {
  std::string literal;
  if (!options.ignore_case) {
    if (auto kind = Classify(options.pattern, is_utf8, &literal)) {
      return LikeMatcher(*kind, std::move(literal));
    }
  }
  return arrow::Status::NotImplemented("LIKE pattern needs RE2");
}

}  // namespace zvec::stile::internal
