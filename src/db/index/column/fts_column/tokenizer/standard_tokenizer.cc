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

#include "standard_tokenizer.h"
#include <utf8proc.h>
#include <array>
#include <zvec/ailego/logger/logger.h>

namespace zvec::fts {

namespace {

constexpr uint32_t kDefaultMaxTokenLength = 255;
constexpr uint32_t kMinMaxTokenLength = 1;
constexpr uint32_t kMaxMaxTokenLength = 1048576;
constexpr uint32_t kVariationSelector16 = 0xFE0F;
constexpr uint32_t kKeycap = 0x20E3;
constexpr uint32_t kUnicodeMaxCodepoint = 0x10FFFF;
constexpr uint32_t kUnicodePageShift = 8;
constexpr size_t kUnicodePageCount =
    (kUnicodeMaxCodepoint >> kUnicodePageShift) + 1;
constexpr size_t kCodepointCacheSize = 1024;
constexpr size_t kMaxInitialTokenCapacity = 4096;

enum class WordBreakClass : uint8_t {
  Other,
  CR,
  LF,
  Newline,
  Extend,
  Format,
  ZWJ,
  ALetter,
  HebrewLetter,
  Numeric,
  Katakana,
  ExtendNumLet,
  RegionalIndicator,
  WSegSpace,
  MidLetter,
  MidNum,
  MidNumLet,
  SingleQuote,
  DoubleQuote,
  Hiragana,
  Hangul,
  SoutheastAsian,
  Ideographic,
  ExtendedPictographic,
};

struct Codepoint {
  utf8proc_int32_t cp{0};
  uint32_t start{0};
  uint32_t end{0};
  WordBreakClass cls{WordBreakClass::Other};
  bool extended_pictographic{false};
  bool emoji_modifier_base{false};
  bool emoji_modifier{false};
};

struct CodepointProperties {
  WordBreakClass cls{WordBreakClass::Other};
  bool extended_pictographic{false};
  bool emoji_modifier_base{false};
  bool emoji_modifier{false};
};

struct CodepointCacheEntry {
  uint32_t cp{0};
  CodepointProperties props;
  bool valid{false};
};

struct UnicodeClassRange {
  uint32_t first{0};
  uint32_t last{0};
  WordBreakClass cls{WordBreakClass::Other};
};

struct UnicodeRange {
  uint32_t first{0};
  uint32_t last{0};
};

struct UnicodeRangeIndex {
  uint32_t first{0};
  uint32_t last{0};
};

template <size_t RangeCount>
struct UnicodeRangeIndexTable {
  std::array<UnicodeRangeIndex, kUnicodePageCount> pages;
};

#include "standard_tokenizer_unicode.inc"

std::array<WordBreakClass, 128> build_ascii_word_break_class_table() {
  std::array<WordBreakClass, 128> table{};
  table.fill(WordBreakClass::Other);
  table['\n'] = WordBreakClass::LF;
  table['\r'] = WordBreakClass::CR;
  table[0x0B] = WordBreakClass::Newline;
  table[0x0C] = WordBreakClass::Newline;
  table[' '] = WordBreakClass::WSegSpace;
  table['"'] = WordBreakClass::DoubleQuote;
  table['\''] = WordBreakClass::SingleQuote;
  table[','] = WordBreakClass::MidNum;
  table['.'] = WordBreakClass::MidNumLet;
  table[':'] = WordBreakClass::MidLetter;
  table[';'] = WordBreakClass::MidNum;
  table['_'] = WordBreakClass::ExtendNumLet;
  for (uint32_t cp = '0'; cp <= '9'; ++cp) {
    table[cp] = WordBreakClass::Numeric;
  }
  for (uint32_t cp = 'A'; cp <= 'Z'; ++cp) {
    table[cp] = WordBreakClass::ALetter;
  }
  for (uint32_t cp = 'a'; cp <= 'z'; ++cp) {
    table[cp] = WordBreakClass::ALetter;
  }
  return table;
}

const std::array<WordBreakClass, 128> kAsciiWordBreakClasses =
    build_ascii_word_break_class_table();

size_t estimate_token_capacity(size_t byte_size) {
  size_t capacity = byte_size / 4 + 1;
  if (capacity > kMaxInitialTokenCapacity) {
    return kMaxInitialTokenCapacity;
  }
  return capacity;
}

size_t estimate_codepoint_capacity(size_t byte_size) {
  return byte_size / 2 + 1;
}

template <typename Range, size_t N>
constexpr UnicodeRangeIndexTable<N> build_range_index(
    const Range (&ranges)[N]) {
  UnicodeRangeIndexTable<N> index{};
  size_t first = 0;
  for (size_t page = 0; page < kUnicodePageCount; ++page) {
    uint32_t page_first = static_cast<uint32_t>(page << kUnicodePageShift);
    uint32_t page_last = page_first + ((1u << kUnicodePageShift) - 1);
    while (first < N && ranges[first].last < page_first) {
      ++first;
    }
    size_t last = first;
    while (last < N && ranges[last].first <= page_last) {
      ++last;
    }
    index.pages[page] = {static_cast<uint32_t>(first),
                         static_cast<uint32_t>(last)};
  }
  return index;
}

constexpr auto kWordBreakRangeIndex = build_range_index(kWordBreakRanges);
constexpr auto kScriptClassRangeIndex = build_range_index(kScriptClassRanges);
constexpr auto kExtendedPictographicRangeIndex =
    build_range_index(kExtendedPictographicRanges);
constexpr auto kEmojiModifierBaseRangeIndex =
    build_range_index(kEmojiModifierBaseRanges);
constexpr auto kEmojiModifierRangeIndex =
    build_range_index(kEmojiModifierRanges);
constexpr auto kLineBreakComplexContextRangeIndex =
    build_range_index(kLineBreakComplexContextRanges);

template <typename Range, size_t N>
constexpr size_t range_count(const Range (&)[N]) {
  return N;
}

WordBreakClass lookup_class_range(const UnicodeClassRange *ranges,
                                  size_t range_count,
                                  const UnicodeRangeIndex *index, uint32_t cp) {
  size_t page = cp >> kUnicodePageShift;
  if (page >= kUnicodePageCount) {
    return WordBreakClass::Other;
  }
  size_t left = index[page].first;
  size_t right = index[page].last;
  if (right > range_count) {
    right = range_count;
  }
  if (left > right) {
    left = right;
  }
  while (left < right) {
    size_t mid = left + (right - left) / 2;
    if (cp < ranges[mid].first) {
      right = mid;
    } else if (cp > ranges[mid].last) {
      left = mid + 1;
    } else {
      return ranges[mid].cls;
    }
  }
  return WordBreakClass::Other;
}

bool contains_range(const UnicodeRange *ranges, size_t range_count,
                    const UnicodeRangeIndex *index, uint32_t cp) {
  size_t page = cp >> kUnicodePageShift;
  if (page >= kUnicodePageCount) {
    return false;
  }
  size_t left = index[page].first;
  size_t right = index[page].last;
  if (right > range_count) {
    right = range_count;
  }
  if (left > right) {
    left = right;
  }
  while (left < right) {
    size_t mid = left + (right - left) / 2;
    if (cp < ranges[mid].first) {
      right = mid;
    } else if (cp > ranges[mid].last) {
      left = mid + 1;
    } else {
      return true;
    }
  }
  return false;
}

WordBreakClass lookup_word_break_class(uint32_t cp) {
  return lookup_class_range(kWordBreakRanges, range_count(kWordBreakRanges),
                            kWordBreakRangeIndex.pages.data(), cp);
}

WordBreakClass lookup_script_class(uint32_t cp) {
  return lookup_class_range(kScriptClassRanges, range_count(kScriptClassRanges),
                            kScriptClassRangeIndex.pages.data(), cp);
}

bool is_extended_pictographic(uint32_t cp) {
  return contains_range(kExtendedPictographicRanges,
                        range_count(kExtendedPictographicRanges),
                        kExtendedPictographicRangeIndex.pages.data(), cp);
}

bool is_emoji_modifier_base(uint32_t cp) {
  return contains_range(kEmojiModifierBaseRanges,
                        range_count(kEmojiModifierBaseRanges),
                        kEmojiModifierBaseRangeIndex.pages.data(), cp);
}

bool is_emoji_modifier(uint32_t cp) {
  return contains_range(kEmojiModifierRanges, range_count(kEmojiModifierRanges),
                        kEmojiModifierRangeIndex.pages.data(), cp);
}

bool is_line_break_complex_context(uint32_t cp) {
  return contains_range(kLineBreakComplexContextRanges,
                        range_count(kLineBreakComplexContextRanges),
                        kLineBreakComplexContextRangeIndex.pages.data(), cp);
}

bool is_ahletter(WordBreakClass cls) {
  return cls == WordBreakClass::ALetter || cls == WordBreakClass::HebrewLetter;
}

bool is_ignored(WordBreakClass cls) {
  return cls == WordBreakClass::Extend || cls == WordBreakClass::Format ||
         cls == WordBreakClass::ZWJ;
}

bool is_extend_or_format(WordBreakClass cls) {
  return cls == WordBreakClass::Extend || cls == WordBreakClass::Format;
}

bool previous_is_zwj(const std::vector<Codepoint> &codepoints, size_t index) {
  return index > 0 && codepoints[index - 1].cls == WordBreakClass::ZWJ;
}

bool is_extended_pictographic_codepoint(const Codepoint &codepoint) {
  return codepoint.cls == WordBreakClass::ExtendedPictographic ||
         codepoint.extended_pictographic;
}

bool is_token_start(WordBreakClass cls) {
  return is_ahletter(cls) || cls == WordBreakClass::Numeric ||
         cls == WordBreakClass::Katakana ||
         cls == WordBreakClass::RegionalIndicator ||
         cls == WordBreakClass::Hiragana || cls == WordBreakClass::Hangul ||
         cls == WordBreakClass::SoutheastAsian ||
         cls == WordBreakClass::Ideographic ||
         cls == WordBreakClass::ExtendedPictographic;
}

bool is_connector(WordBreakClass cls) {
  return is_ahletter(cls) || cls == WordBreakClass::Numeric ||
         cls == WordBreakClass::Katakana || cls == WordBreakClass::ExtendNumLet;
}

WordBreakClass lookup_ascii_word_break_class(uint32_t cp) {
  if (cp <= 0x7F) {
    return kAsciiWordBreakClasses[cp];
  }
  return WordBreakClass::Other;
}

bool is_ascii_letter_or_digit(unsigned char ch) {
  return (ch >= '0' && ch <= '9') || (ch >= 'A' && ch <= 'Z') ||
         (ch >= 'a' && ch <= 'z');
}

bool is_ascii_word_body_char(unsigned char ch) {
  return is_ascii_letter_or_digit(ch) || ch == '_';
}

bool is_hangul_syllable(uint32_t codepoint) {
  return codepoint >= 0xAC00 && codepoint <= 0xD7A3;
}

bool is_utf8_continuation(unsigned char ch) {
  return (ch & 0xC0) == 0x80;
}

bool decode_utf8_codepoint(const utf8proc_uint8_t *str, size_t len,
                           size_t index, utf8proc_int32_t *cp, size_t *bytes) {
  unsigned char lead = str[index];
  if (lead < 0x80) {
    *cp = lead;
    *bytes = 1;
    return true;
  }

  if ((lead & 0xE0) == 0xC0) {
    if (index + 1 >= len || !is_utf8_continuation(str[index + 1])) {
      return false;
    }
    uint32_t value = ((lead & 0x1F) << 6) | (str[index + 1] & 0x3F);
    if (value < 0x80) {
      return false;
    }
    *cp = static_cast<utf8proc_int32_t>(value);
    *bytes = 2;
    return true;
  }

  if ((lead & 0xF0) == 0xE0) {
    if (index + 2 >= len || !is_utf8_continuation(str[index + 1]) ||
        !is_utf8_continuation(str[index + 2])) {
      return false;
    }
    uint32_t value = ((lead & 0x0F) << 12) | ((str[index + 1] & 0x3F) << 6) |
                     (str[index + 2] & 0x3F);
    if (value < 0x800 || (value >= 0xD800 && value <= 0xDFFF)) {
      return false;
    }
    *cp = static_cast<utf8proc_int32_t>(value);
    *bytes = 3;
    return true;
  }

  if ((lead & 0xF8) == 0xF0) {
    if (index + 3 >= len || !is_utf8_continuation(str[index + 1]) ||
        !is_utf8_continuation(str[index + 2]) ||
        !is_utf8_continuation(str[index + 3])) {
      return false;
    }
    uint32_t value = ((lead & 0x07) << 18) | ((str[index + 1] & 0x3F) << 12) |
                     ((str[index + 2] & 0x3F) << 6) | (str[index + 3] & 0x3F);
    if (value < 0x10000 || value > kUnicodeMaxCodepoint) {
      return false;
    }
    *cp = static_cast<utf8proc_int32_t>(value);
    *bytes = 4;
    return true;
  }

  return false;
}

WordBreakClass classify_codepoint(utf8proc_int32_t cp) {
  uint32_t codepoint = static_cast<uint32_t>(cp);
  if (codepoint <= 0x7F) {
    return lookup_ascii_word_break_class(codepoint);
  }

  WordBreakClass script_cls = lookup_script_class(codepoint);
  if (script_cls == WordBreakClass::Ideographic ||
      script_cls == WordBreakClass::Hiragana) {
    return script_cls;
  }
  if (is_hangul_syllable(codepoint)) {
    return WordBreakClass::Hangul;
  }

  WordBreakClass word_break_cls = lookup_word_break_class(codepoint);
  if (word_break_cls == WordBreakClass::ALetter ||
      word_break_cls == WordBreakClass::Katakana ||
      word_break_cls == WordBreakClass::Other) {
    if (is_line_break_complex_context(codepoint)) {
      return WordBreakClass::SoutheastAsian;
    }
    if (script_cls != WordBreakClass::Other) {
      if (script_cls == WordBreakClass::Hangul &&
          !is_ahletter(word_break_cls)) {
        return word_break_cls;
      }
      return script_cls;
    }
  }
  if (word_break_cls != WordBreakClass::Other) {
    return word_break_cls;
  }
  if (is_extended_pictographic(codepoint)) {
    return WordBreakClass::ExtendedPictographic;
  }
  return WordBreakClass::Other;
}

CodepointProperties lookup_codepoint_properties(uint32_t codepoint) {
  CodepointProperties props;
  props.cls = classify_codepoint(static_cast<utf8proc_int32_t>(codepoint));
  props.extended_pictographic =
      props.cls == WordBreakClass::ExtendedPictographic;
  if (!props.extended_pictographic && codepoint >= 0x00A9) {
    props.extended_pictographic = is_extended_pictographic(codepoint);
  }
  if (codepoint >= 0x261D) {
    props.emoji_modifier_base = is_emoji_modifier_base(codepoint);
  }
  if (codepoint >= 0x1F3FB) {
    props.emoji_modifier = is_emoji_modifier(codepoint);
  }
  return props;
}

CodepointProperties get_codepoint_properties(
    uint32_t codepoint,
    std::array<CodepointCacheEntry, kCodepointCacheSize> *cache) {
  size_t slot = (codepoint * 2654435761u) & (kCodepointCacheSize - 1);
  CodepointCacheEntry &entry = (*cache)[slot];
  if (entry.valid && entry.cp == codepoint) {
    return entry.props;
  }

  CodepointProperties props = lookup_codepoint_properties(codepoint);
  entry.cp = codepoint;
  entry.props = props;
  entry.valid = true;
  return props;
}

std::array<CodepointCacheEntry, kCodepointCacheSize> *codepoint_cache() {
  thread_local std::array<CodepointCacheEntry, kCodepointCacheSize> cache{};
  return &cache;
}

std::vector<Codepoint> decode_utf8(const std::string &text) {
  std::vector<Codepoint> codepoints;
  codepoints.reserve(estimate_codepoint_capacity(text.size()));
  auto *cache = codepoint_cache();
  const auto *str = reinterpret_cast<const utf8proc_uint8_t *>(text.data());
  size_t len = text.size();
  size_t index = 0;

  while (index < len) {
    utf8proc_int32_t cp;
    size_t bytes = 0;
    if (!decode_utf8_codepoint(str, len, index, &cp, &bytes)) {
      Codepoint item;
      item.start = static_cast<uint32_t>(index);
      item.end = static_cast<uint32_t>(index + 1);
      item.cls = WordBreakClass::Other;
      codepoints.push_back(item);
      ++index;
      continue;
    }

    Codepoint item;
    item.cp = cp;
    item.start = static_cast<uint32_t>(index);
    item.end = static_cast<uint32_t>(index + bytes);
    uint32_t codepoint = static_cast<uint32_t>(cp);
    CodepointProperties props = get_codepoint_properties(codepoint, cache);
    item.cls = props.cls;
    item.extended_pictographic = props.extended_pictographic;
    item.emoji_modifier_base = props.emoji_modifier_base;
    item.emoji_modifier = props.emoji_modifier;
    codepoints.push_back(item);
    index += bytes;
  }
  return codepoints;
}

size_t next_significant(const std::vector<Codepoint> &codepoints,
                        size_t index) {
  while (index < codepoints.size() && is_ignored(codepoints[index].cls)) {
    ++index;
  }
  return index;
}

bool punctuation_connects(WordBreakClass left, WordBreakClass punct,
                          WordBreakClass right) {
  if (is_ahletter(left) && is_ahletter(right) &&
      (punct == WordBreakClass::MidLetter ||
       punct == WordBreakClass::MidNumLet ||
       punct == WordBreakClass::SingleQuote)) {
    return true;
  }
  if (left == WordBreakClass::HebrewLetter &&
      right == WordBreakClass::HebrewLetter &&
      punct == WordBreakClass::DoubleQuote) {
    return true;
  }
  if (left == WordBreakClass::Numeric && right == WordBreakClass::Numeric &&
      (punct == WordBreakClass::MidNum || punct == WordBreakClass::MidNumLet ||
       punct == WordBreakClass::SingleQuote)) {
    return true;
  }
  return false;
}

bool significant_connects(WordBreakClass left, WordBreakClass right) {
  if (is_ahletter(left) && is_ahletter(right)) {
    return true;
  }
  if ((is_ahletter(left) && right == WordBreakClass::Numeric) ||
      (left == WordBreakClass::Numeric && is_ahletter(right))) {
    return true;
  }
  if (left == WordBreakClass::Numeric && right == WordBreakClass::Numeric) {
    return true;
  }
  if (left == WordBreakClass::Katakana && right == WordBreakClass::Katakana) {
    return true;
  }
  if (is_connector(left) && right == WordBreakClass::ExtendNumLet) {
    return true;
  }
  if (left == WordBreakClass::ExtendNumLet && is_connector(right)) {
    return true;
  }
  if (left == WordBreakClass::ExtendNumLet &&
      right == WordBreakClass::ExtendNumLet) {
    return true;
  }
  if (left == WordBreakClass::Hangul && right == WordBreakClass::Hangul) {
    return true;
  }
  if (left == WordBreakClass::SoutheastAsian &&
      right == WordBreakClass::SoutheastAsian) {
    return true;
  }
  return false;
}

bool is_keycap_base(const Codepoint &codepoint) {
  return (codepoint.cp >= '0' && codepoint.cp <= '9') || codepoint.cp == '#' ||
         codepoint.cp == '*';
}

size_t consume_extend_or_format(const std::vector<Codepoint> &codepoints,
                                size_t index) {
  while (index < codepoints.size() &&
         is_extend_or_format(codepoints[index].cls)) {
    ++index;
  }
  return index;
}

size_t consume_extend_format_and_modifier(
    const std::vector<Codepoint> &codepoints, size_t index) {
  while (index < codepoints.size() &&
         is_extend_or_format(codepoints[index].cls)) {
    ++index;
  }
  if (index < codepoints.size() && codepoints[index].emoji_modifier) {
    index = consume_extend_or_format(codepoints, index + 1);
  }
  return index;
}

size_t scan_keycap_token(const std::vector<Codepoint> &codepoints,
                         size_t start) {
  if (!is_keycap_base(codepoints[start])) {
    return start;
  }

  size_t index = start + 1;
  if (index < codepoints.size() &&
      static_cast<uint32_t>(codepoints[index].cp) == kVariationSelector16) {
    ++index;
  }
  if (index >= codepoints.size() ||
      static_cast<uint32_t>(codepoints[index].cp) != kKeycap) {
    return start;
  }

  return consume_extend_or_format(codepoints, index + 1);
}

size_t scan_emoji_modifier_token(const std::vector<Codepoint> &codepoints,
                                 size_t start) {
  if (codepoints[start].emoji_modifier) {
    return consume_extend_or_format(codepoints, start + 1);
  }
  if (!codepoints[start].emoji_modifier_base) {
    return start;
  }

  size_t index = start + 1;
  while (index < codepoints.size() &&
         is_extend_or_format(codepoints[index].cls) &&
         !codepoints[index].emoji_modifier) {
    ++index;
  }
  if (index >= codepoints.size() || !codepoints[index].emoji_modifier) {
    return start;
  }

  return consume_extend_or_format(codepoints, index + 1);
}

size_t scan_emoji_token(const std::vector<Codepoint> &codepoints,
                        size_t start) {
  size_t index = consume_extend_format_and_modifier(codepoints, start + 1);

  while (index < codepoints.size()) {
    if (codepoints[index].cls != WordBreakClass::ZWJ) {
      break;
    }
    ++index;
    while (index < codepoints.size() &&
           is_extend_or_format(codepoints[index].cls)) {
      ++index;
    }
    if (index >= codepoints.size() ||
        !is_extended_pictographic_codepoint(codepoints[index])) {
      break;
    }
    ++index;
    index = consume_extend_format_and_modifier(codepoints, index);
  }
  return index;
}

size_t scan_zwj_ext_pict_token(const std::vector<Codepoint> &codepoints,
                               size_t start) {
  size_t index = start + 1;
  if (index >= codepoints.size() ||
      !is_extended_pictographic_codepoint(codepoints[index])) {
    return start;
  }
  return scan_emoji_token(codepoints, index);
}

size_t scan_regional_indicator_token(const std::vector<Codepoint> &codepoints,
                                     size_t start) {
  size_t index = start + 1;
  while (index < codepoints.size() && is_ignored(codepoints[index].cls)) {
    ++index;
  }
  if (index < codepoints.size() &&
      codepoints[index].cls == WordBreakClass::RegionalIndicator) {
    ++index;
    while (index < codepoints.size() && is_ignored(codepoints[index].cls)) {
      ++index;
    }
  }
  return index;
}

size_t scan_single_token(const std::vector<Codepoint> &codepoints,
                         size_t start) {
  size_t index = start + 1;
  while (index < codepoints.size() && is_ignored(codepoints[index].cls)) {
    ++index;
  }
  return index;
}

size_t scan_word_token(const std::vector<Codepoint> &codepoints, size_t start) {
  size_t end = start + 1;
  size_t last_sig = start;

  while (end < codepoints.size()) {
    WordBreakClass cls = codepoints[end].cls;
    if (is_ignored(cls)) {
      ++end;
      continue;
    }
    if (cls == WordBreakClass::ExtendNumLet) {
      if (!significant_connects(codepoints[last_sig].cls, cls)) {
        break;
      }
      last_sig = end;
      ++end;
      continue;
    }
    if (cls == WordBreakClass::SingleQuote &&
        codepoints[last_sig].cls == WordBreakClass::HebrewLetter) {
      last_sig = end;
      ++end;
      continue;
    }
    if (is_extended_pictographic_codepoint(codepoints[end]) &&
        previous_is_zwj(codepoints, end)) {
      end = scan_emoji_token(codepoints, end);
      last_sig = end - 1;
      continue;
    }
    if (cls == WordBreakClass::Ideographic ||
        cls == WordBreakClass::ExtendedPictographic ||
        cls == WordBreakClass::RegionalIndicator || !is_token_start(cls)) {
      size_t right = next_significant(codepoints, end + 1);
      if (right < codepoints.size() &&
          punctuation_connects(codepoints[last_sig].cls, cls,
                               codepoints[right].cls)) {
        end = right + 1;
        last_sig = right;
        continue;
      }
      break;
    }

    if (!significant_connects(codepoints[last_sig].cls, cls)) {
      break;
    }
    last_sig = end;
    ++end;
  }
  return end;
}

bool span_has_core_token(const std::vector<Codepoint> &codepoints, size_t start,
                         size_t end) {
  for (size_t index = start; index < end; ++index) {
    WordBreakClass cls = codepoints[index].cls;
    if (is_token_start(cls)) {
      return true;
    }
  }
  return false;
}

size_t trim_non_core_suffix(const std::vector<Codepoint> &codepoints,
                            size_t start, size_t end) {
  size_t trimmed = end;
  while (trimmed > start) {
    WordBreakClass cls = codepoints[trimmed - 1].cls;
    if (is_token_start(cls) || is_ignored(cls)) {
      break;
    }
    --trimmed;
  }
  return trimmed;
}

void emit_non_empty_core_span(const std::string &text,
                              const std::vector<Codepoint> &codepoints,
                              size_t start, size_t end, uint32_t *position,
                              std::vector<Token> *tokens) {
  if (start >= end || !span_has_core_token(codepoints, start, end)) {
    return;
  }
  Token token;
  token.text = text.substr(codepoints[start].start,
                           codepoints[end - 1].end - codepoints[start].start);
  token.offset = codepoints[start].start;
  token.position = (*position)++;
  tokens->push_back(std::move(token));
}

void emit_token_span(const std::string &text,
                     const std::vector<Codepoint> &codepoints, size_t start,
                     size_t end, uint32_t max_token_length, uint32_t *position,
                     std::vector<Token> *tokens) {
  if (end - start <= max_token_length) {
    Token token;
    token.text = text.substr(codepoints[start].start,
                             codepoints[end - 1].end - codepoints[start].start);
    token.offset = codepoints[start].start;
    token.position = (*position)++;
    tokens->push_back(std::move(token));
    return;
  }

  size_t token_start = start;
  uint32_t codepoint_count = 0;
  size_t index = start;
  while (index < end) {
    if (codepoint_count >= max_token_length &&
        is_token_start(codepoints[index].cls)) {
      size_t emit_end = trim_non_core_suffix(codepoints, token_start, index);
      emit_non_empty_core_span(text, codepoints, token_start, emit_end,
                               position, tokens);
      token_start = index;
      codepoint_count = 0;
      continue;
    }
    ++codepoint_count;
    ++index;
  }

  size_t token_end = end;
  if (end - token_start > max_token_length) {
    token_end = trim_non_core_suffix(codepoints, token_start, end);
  }
  emit_non_empty_core_span(text, codepoints, token_start, token_end, position,
                           tokens);
}

bool is_ascii_text(const std::string &text) {
  for (unsigned char ch : text) {
    if ((ch & 0x80) != 0) {
      return false;
    }
  }
  return true;
}

size_t scan_ascii_word_token(const std::string &text, size_t start) {
  size_t end = start + 1;
  size_t last_sig = start;

  while (end < text.size()) {
    auto ch = static_cast<unsigned char>(text[end]);
    if (is_ascii_word_body_char(ch)) {
      last_sig = end;
      ++end;
      continue;
    }

    WordBreakClass cls = lookup_ascii_word_break_class(ch);
    if (cls == WordBreakClass::ExtendNumLet) {
      WordBreakClass left = lookup_ascii_word_break_class(
          static_cast<unsigned char>(text[last_sig]));
      if (!significant_connects(left, cls)) {
        break;
      }
      last_sig = end;
      ++end;
      continue;
    }
    if (!is_token_start(cls)) {
      size_t right = end + 1;
      if (right < text.size()) {
        WordBreakClass left = lookup_ascii_word_break_class(
            static_cast<unsigned char>(text[last_sig]));
        WordBreakClass right_cls = lookup_ascii_word_break_class(
            static_cast<unsigned char>(text[right]));
        if (punctuation_connects(left, cls, right_cls)) {
          end = right + 1;
          last_sig = right;
          continue;
        }
      }
      break;
    }

    WordBreakClass left = lookup_ascii_word_break_class(
        static_cast<unsigned char>(text[last_sig]));
    if (!significant_connects(left, cls)) {
      break;
    }
    last_sig = end;
    ++end;
  }
  return end;
}

bool ascii_span_has_core_token(const std::string &text, size_t start,
                               size_t end) {
  for (size_t index = start; index < end; ++index) {
    auto ch = static_cast<unsigned char>(text[index]);
    if (is_ascii_letter_or_digit(ch)) {
      return true;
    }
  }
  return false;
}

size_t trim_ascii_non_core_suffix(const std::string &text, size_t start,
                                  size_t end) {
  size_t trimmed = end;
  while (trimmed > start) {
    auto ch = static_cast<unsigned char>(text[trimmed - 1]);
    if (is_ascii_letter_or_digit(ch)) {
      break;
    }
    --trimmed;
  }
  return trimmed;
}

void emit_non_empty_ascii_core_span(const std::string &text, size_t start,
                                    size_t end, uint32_t *position,
                                    std::vector<Token> *tokens) {
  if (start >= end || !ascii_span_has_core_token(text, start, end)) {
    return;
  }
  Token token;
  token.text = text.substr(start, end - start);
  token.offset = static_cast<uint32_t>(start);
  token.position = (*position)++;
  tokens->push_back(std::move(token));
}

void emit_ascii_token_span(const std::string &text, size_t start, size_t end,
                           uint32_t max_token_length, uint32_t *position,
                           std::vector<Token> *tokens) {
  if (end - start <= max_token_length) {
    Token token;
    token.text = text.substr(start, end - start);
    token.offset = static_cast<uint32_t>(start);
    token.position = (*position)++;
    tokens->push_back(std::move(token));
    return;
  }

  size_t token_start = start;
  uint32_t codepoint_count = 0;
  size_t index = start;
  while (index < end) {
    auto ch = static_cast<unsigned char>(text[index]);
    if (codepoint_count >= max_token_length && is_ascii_letter_or_digit(ch)) {
      size_t emit_end = trim_ascii_non_core_suffix(text, token_start, index);
      emit_non_empty_ascii_core_span(text, token_start, emit_end, position,
                                     tokens);
      token_start = index;
      codepoint_count = 0;
      continue;
    }
    ++codepoint_count;
    ++index;
  }

  size_t token_end = end;
  if (end - token_start > max_token_length) {
    token_end = trim_ascii_non_core_suffix(text, token_start, end);
  }
  emit_non_empty_ascii_core_span(text, token_start, token_end, position,
                                 tokens);
}

std::vector<Token> tokenize_ascii(const std::string &text,
                                  uint32_t max_token_length) {
  std::vector<Token> tokens;
  tokens.reserve(estimate_token_capacity(text.size()));
  uint32_t position = 0;
  size_t index = 0;
  while (index < text.size()) {
    auto ch = static_cast<unsigned char>(text[index]);
    if (is_ascii_letter_or_digit(ch)) {
      size_t end = scan_ascii_word_token(text, index);
      emit_ascii_token_span(text, index, end, max_token_length, &position,
                            &tokens);
      index = end;
      continue;
    }

    WordBreakClass cls = lookup_ascii_word_break_class(ch);
    if (cls == WordBreakClass::ExtendNumLet) {
      size_t end = scan_ascii_word_token(text, index);
      if (ascii_span_has_core_token(text, index, end)) {
        emit_ascii_token_span(text, index, end, max_token_length, &position,
                              &tokens);
      }
      index = end;
      continue;
    }
    ++index;
  }
  return tokens;
}

}  // namespace

bool StandardTokenizer::init(const ailego::JsonObject &config) {
  max_token_length_ = kDefaultMaxTokenLength;
  auto length_val = config["max_token_length"];
  if (!length_val.is_null()) {
    if (!length_val.is_integer()) {
      LOG_ERROR("StandardTokenizer: max_token_length must be integer");
      return false;
    }
    auto configured_length = length_val.as_integer();
    if (configured_length < kMinMaxTokenLength ||
        configured_length > kMaxMaxTokenLength) {
      LOG_ERROR("StandardTokenizer: max_token_length out of range: %zu",
                (size_t)configured_length);
      return false;
    }
    max_token_length_ = static_cast<uint32_t>(configured_length);
  }
  return true;
}

std::vector<Token> StandardTokenizer::tokenize(const std::string &text) const {
  if (is_ascii_text(text)) {
    return tokenize_ascii(text, max_token_length_);
  }

  std::vector<Token> tokens;
  tokens.reserve(estimate_token_capacity(text.size()));
  uint32_t position = 0;
  std::vector<Codepoint> codepoints = decode_utf8(text);

  size_t index = 0;
  while (index < codepoints.size()) {
    WordBreakClass cls = codepoints[index].cls;
    if (cls == WordBreakClass::Ideographic || cls == WordBreakClass::Hiragana) {
      size_t end = scan_single_token(codepoints, index);
      emit_token_span(text, codepoints, index, end, max_token_length_,
                      &position, &tokens);
      index = end;
      continue;
    }
    if (cls == WordBreakClass::Hangul ||
        cls == WordBreakClass::SoutheastAsian) {
      size_t end = scan_word_token(codepoints, index);
      emit_token_span(text, codepoints, index, end, max_token_length_,
                      &position, &tokens);
      index = end;
      continue;
    }
    if (cls == WordBreakClass::RegionalIndicator) {
      size_t end = scan_regional_indicator_token(codepoints, index);
      emit_token_span(text, codepoints, index, end, max_token_length_,
                      &position, &tokens);
      index = end;
      continue;
    }
    size_t end = scan_keycap_token(codepoints, index);
    if (end > index) {
      emit_token_span(text, codepoints, index, end, max_token_length_,
                      &position, &tokens);
      index = end;
      continue;
    }
    if (cls == WordBreakClass::ExtendedPictographic) {
      end = scan_emoji_token(codepoints, index);
      emit_token_span(text, codepoints, index, end, max_token_length_,
                      &position, &tokens);
      index = end;
      continue;
    }
    if (cls == WordBreakClass::ZWJ) {
      end = scan_zwj_ext_pict_token(codepoints, index);
      if (end > index) {
        emit_token_span(text, codepoints, index, end, max_token_length_,
                        &position, &tokens);
        index = end;
        continue;
      }
    }
    end = scan_emoji_modifier_token(codepoints, index);
    if (end > index) {
      emit_token_span(text, codepoints, index, end, max_token_length_,
                      &position, &tokens);
      index = end;
      continue;
    }
    if (cls == WordBreakClass::ExtendNumLet) {
      end = scan_word_token(codepoints, index);
      if (span_has_core_token(codepoints, index, end)) {
        emit_token_span(text, codepoints, index, end, max_token_length_,
                        &position, &tokens);
      }
      index = end;
      continue;
    }
    if (!is_token_start(cls)) {
      ++index;
      continue;
    }

    end = scan_word_token(codepoints, index);
    emit_token_span(text, codepoints, index, end, max_token_length_, &position,
                    &tokens);
    index = end;
  }

  return tokens;
}

}  // namespace zvec::fts
