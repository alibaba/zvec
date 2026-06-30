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

#include "ascii_folding_token_filter.h"
#include <utf8proc.h>
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <memory>
#include <string>

namespace zvec::fts {

namespace {

// Supplementary folding table for codepoints that NFKD+STRIPMARK does not
// reduce to ASCII.  Inspired by Lucene's ASCIIFoldingFilter.
// Each entry maps a single codepoint to a short ASCII string.
struct FoldEntry {
  utf8proc_int32_t codepoint;
  const char *ascii;
};

// clang-format off
// Sorted by codepoint — binary search via std::lower_bound.
static const FoldEntry kExtraFolds[] = {
    {0x00C6, "AE"},  // Æ  LATIN CAPITAL LETTER AE
    {0x00D0, "D"},   // Ð  LATIN CAPITAL LETTER ETH
    {0x00D8, "O"},   // Ø  LATIN CAPITAL LETTER O WITH STROKE
    {0x00DE, "TH"},  // Þ  LATIN CAPITAL LETTER THORN
    {0x00DF, "ss"},  // ß  LATIN SMALL LETTER SHARP S
    {0x00E6, "ae"},  // æ  LATIN SMALL LETTER AE
    {0x00F0, "d"},   // ð  LATIN SMALL LETTER ETH
    {0x00F8, "o"},   // ø  LATIN SMALL LETTER O WITH STROKE
    {0x00FE, "th"},  // þ  LATIN SMALL LETTER THORN
    {0x0110, "D"},   // Đ  LATIN CAPITAL LETTER D WITH STROKE
    {0x0111, "d"},   // đ  LATIN SMALL LETTER D WITH STROKE
    {0x0126, "H"},   // Ħ  LATIN CAPITAL LETTER H WITH STROKE
    {0x0127, "h"},   // ħ  LATIN SMALL LETTER H WITH STROKE
    {0x0131, "i"},   // ı  LATIN SMALL LETTER DOTLESS I
    {0x0132, "IJ"},  // Ĳ  LATIN CAPITAL LIGATURE IJ
    {0x0133, "ij"},  // ĳ  LATIN SMALL LIGATURE IJ
    {0x0138, "k"},   // ĸ  LATIN SMALL LETTER KRA
    {0x0141, "L"},   // Ł  LATIN CAPITAL LETTER L WITH STROKE
    {0x0142, "l"},   // ł  LATIN SMALL LETTER L WITH STROKE
    {0x014A, "N"},   // Ŋ  LATIN CAPITAL LETTER ENG
    {0x014B, "n"},   // ŋ  LATIN SMALL LETTER ENG
    {0x0152, "OE"},  // Œ  LATIN CAPITAL LIGATURE OE
    {0x0153, "oe"},  // œ  LATIN SMALL LIGATURE OE
    {0x0166, "T"},   // Ŧ  LATIN CAPITAL LETTER T WITH STROKE
    {0x0167, "t"},   // ŧ  LATIN SMALL LETTER T WITH STROKE
    {0x0180, "b"},   // ƀ  LATIN SMALL LETTER B WITH STROKE
    {0x0181, "B"},   // Ɓ  LATIN CAPITAL LETTER B WITH HOOK
    {0x0182, "B"},   // Ƃ  LATIN CAPITAL LETTER B WITH TOPBAR
    {0x0183, "b"},   // ƃ  LATIN SMALL LETTER B WITH TOPBAR
    {0x0187, "C"},   // Ƈ  LATIN CAPITAL LETTER C WITH HOOK
    {0x0188, "c"},   // ƈ  LATIN SMALL LETTER C WITH HOOK
    {0x0189, "D"},   // Ɖ  LATIN CAPITAL LETTER AFRICAN D
    {0x018A, "D"},   // Ɗ  LATIN CAPITAL LETTER D WITH HOOK
    {0x018B, "D"},   // Ƌ  LATIN CAPITAL LETTER D WITH TOPBAR
    {0x018C, "d"},   // ƌ  LATIN SMALL LETTER D WITH TOPBAR
    {0x018E, "E"},   // Ǝ  LATIN CAPITAL LETTER REVERSED E
    {0x018F, "A"},   // Ə  LATIN CAPITAL LETTER SCHWA
    {0x0190, "E"},   // Ɛ  LATIN CAPITAL LETTER OPEN E
    {0x0191, "F"},   // Ƒ  LATIN CAPITAL LETTER F WITH HOOK
    {0x0192, "f"},   // ƒ  LATIN SMALL LETTER F WITH HOOK
    {0x0193, "G"},   // Ɠ  LATIN CAPITAL LETTER G WITH HOOK
    {0x0195, "hv"},  // ƕ  LATIN SMALL LETTER HV
    {0x0196, "I"},   // Ɩ  LATIN CAPITAL LETTER IOTA
    {0x0197, "I"},   // Ɨ  LATIN CAPITAL LETTER I WITH STROKE
    {0x0198, "K"},   // Ƙ  LATIN CAPITAL LETTER K WITH HOOK
    {0x0199, "k"},   // ƙ  LATIN SMALL LETTER K WITH HOOK
    {0x019A, "l"},   // ƚ  LATIN SMALL LETTER L WITH BAR
    {0x019D, "N"},   // Ɲ  LATIN CAPITAL LETTER N WITH LEFT HOOK
    {0x019E, "n"},   // ƞ  LATIN SMALL LETTER N WITH LONG RIGHT LEG
    {0x01A0, "O"},   // Ơ  LATIN CAPITAL LETTER O WITH HORN
    {0x01A1, "o"},   // ơ  LATIN SMALL LETTER O WITH HORN
    {0x01A6, "R"},   // Ʀ  LATIN LETTER YR
    {0x01A9, "S"},   // Ʃ  LATIN CAPITAL LETTER ESH
    {0x01AB, "t"},   // ƫ  LATIN SMALL LETTER T WITH PALATAL HOOK
    {0x01AC, "T"},   // Ƭ  LATIN CAPITAL LETTER T WITH HOOK
    {0x01AD, "t"},   // ƭ  LATIN SMALL LETTER T WITH HOOK
    {0x01AE, "T"},   // Ʈ  LATIN CAPITAL LETTER T WITH RETROFLEX HOOK
    {0x01AF, "U"},   // Ư  LATIN CAPITAL LETTER U WITH HORN
    {0x01B0, "u"},   // ư  LATIN SMALL LETTER U WITH HORN
    {0x01B2, "V"},   // Ʋ  LATIN CAPITAL LETTER V WITH HOOK
    {0x01B3, "Y"},   // Ƴ  LATIN CAPITAL LETTER Y WITH HOOK
    {0x01B4, "y"},   // ƴ  LATIN SMALL LETTER Y WITH HOOK
    {0x01B5, "Z"},   // Ƶ  LATIN CAPITAL LETTER Z WITH STROKE
    {0x01B6, "z"},   // ƶ  LATIN SMALL LETTER Z WITH STROKE
    {0x01C4, "DZ"},  // Ǆ  LATIN CAPITAL LETTER DZ WITH CARON
    {0x01C5, "Dz"},  // ǅ  LATIN CAPITAL LETTER D WITH SMALL LETTER Z WITH CARON
    {0x01C6, "dz"},  // ǆ  LATIN SMALL LETTER DZ WITH CARON
    {0x01E4, "G"},   // Ǥ  LATIN CAPITAL LETTER G WITH STROKE
    {0x01E5, "g"},   // ǥ  LATIN SMALL LETTER G WITH STROKE
    {0x0221, "d"},   // ȡ  LATIN SMALL LETTER D WITH CURL
    {0x0222, "OU"},  // Ȣ  LATIN CAPITAL LETTER OU
    {0x0223, "ou"},  // ȣ  LATIN SMALL LETTER OU
    {0x0234, "l"},   // ȴ  LATIN SMALL LETTER L WITH CURL
    {0x0235, "n"},   // ȵ  LATIN SMALL LETTER N WITH CURL
    {0x0236, "t"},   // ȶ  LATIN SMALL LETTER T WITH CURL
    {0x023A, "A"},   // Ⱥ  LATIN CAPITAL LETTER A WITH STROKE
    {0x023B, "C"},   // Ȼ  LATIN CAPITAL LETTER C WITH STROKE
    {0x023C, "c"},   // ȼ  LATIN SMALL LETTER C WITH STROKE
    {0x023E, "T"},   // Ⱦ  LATIN CAPITAL LETTER T WITH DIAGONAL STROKE
    {0x0243, "B"},   // Ƀ  LATIN CAPITAL LETTER B WITH STROKE
    {0x0246, "E"},   // Ɇ  LATIN CAPITAL LETTER E WITH STROKE
    {0x0247, "e"},   // ɇ  LATIN SMALL LETTER E WITH STROKE
    {0x0248, "J"},   // Ɉ  LATIN CAPITAL LETTER J WITH STROKE
    {0x0249, "j"},   // ɉ  LATIN SMALL LETTER J WITH STROKE
    {0x024C, "R"},   // Ɍ  LATIN CAPITAL LETTER R WITH STROKE
    {0x024D, "r"},   // ɍ  LATIN SMALL LETTER R WITH STROKE
    {0x024E, "Y"},   // Ɏ  LATIN CAPITAL LETTER Y WITH STROKE
    {0x024F, "y"},   // ɏ  LATIN SMALL LETTER Y WITH STROKE
    {0x0250, "a"},   // ɐ  LATIN SMALL LETTER TURNED A
    {0x0251, "a"},   // ɑ  LATIN SMALL LETTER ALPHA
    {0x0253, "b"},   // ɓ  LATIN SMALL LETTER B WITH HOOK
    {0x0255, "c"},   // ɕ  LATIN SMALL LETTER C WITH CURL
    {0x0256, "d"},   // ɖ  LATIN SMALL LETTER D WITH TAIL
    {0x0257, "d"},   // ɗ  LATIN SMALL LETTER D WITH HOOK
    {0x0258, "e"},   // ɘ  LATIN SMALL LETTER REVERSED E
    {0x0259, "e"},   // ə  LATIN SMALL LETTER SCHWA
    {0x025B, "e"},   // ɛ  LATIN SMALL LETTER OPEN E
    {0x025C, "e"},   // ɜ  LATIN SMALL LETTER REVERSED OPEN E
    {0x0260, "g"},   // ɠ  LATIN SMALL LETTER G WITH HOOK
    {0x0261, "g"},   // ɡ  LATIN SMALL LETTER SCRIPT G
    {0x0262, "G"},   // ɢ  LATIN LETTER SMALL CAPITAL G
    {0x0265, "h"},   // ɥ  LATIN SMALL LETTER TURNED H
    {0x0266, "h"},   // ɦ  LATIN SMALL LETTER H WITH HOOK
    {0x0268, "i"},   // ɨ  LATIN SMALL LETTER I WITH STROKE
    {0x026A, "I"},   // ɪ  LATIN LETTER SMALL CAPITAL I
    {0x026B, "l"},   // ɫ  LATIN SMALL LETTER L WITH MIDDLE TILDE
    {0x026C, "l"},   // ɬ  LATIN SMALL LETTER L WITH BELT
    {0x026D, "l"},   // ɭ  LATIN SMALL LETTER L WITH RETROFLEX HOOK
    {0x0271, "m"},   // ɱ  LATIN SMALL LETTER M WITH HOOK
    {0x0272, "n"},   // ɲ  LATIN SMALL LETTER N WITH LEFT HOOK
    {0x0273, "n"},   // ɳ  LATIN SMALL LETTER N WITH RETROFLEX HOOK
    {0x0274, "N"},   // ɴ  LATIN LETTER SMALL CAPITAL N
    {0x0275, "o"},   // ɵ  LATIN SMALL LETTER BARRED O
    {0x027D, "r"},   // ɽ  LATIN SMALL LETTER R WITH TAIL
    {0x0282, "s"},   // ʂ  LATIN SMALL LETTER S WITH HOOK
    {0x0283, "s"},   // ʃ  LATIN SMALL LETTER ESH
    {0x0288, "t"},   // ʈ  LATIN SMALL LETTER T WITH RETROFLEX HOOK
    {0x028B, "v"},   // ʋ  LATIN SMALL LETTER V WITH HOOK
    {0x0290, "z"},   // ʐ  LATIN SMALL LETTER Z WITH RETROFLEX HOOK
    {0x0291, "z"},   // ʑ  LATIN SMALL LETTER Z WITH CURL
    {0x0292, "z"},   // ʒ  LATIN SMALL LETTER EZH
    {0x029D, "j"},   // ʝ  LATIN SMALL LETTER J WITH CROSSED-TAIL
    {0x029E, "k"},   // ʞ  LATIN SMALL LETTER TURNED K
    {0x1D6D, "d"},   // ᵭ  LATIN SMALL LETTER D WITH MIDDLE TILDE
    {0x1D6E, "f"},   // ᵮ  LATIN SMALL LETTER F WITH MIDDLE TILDE
    {0x1D6F, "g"},   // ᵯ  LATIN SMALL LETTER G WITH MIDDLE TILDE
    {0x1D70, "r"},   // ᵰ  LATIN SMALL LETTER R WITH MIDDLE TILDE
    {0x1D71, "s"},   // ᵱ  LATIN SMALL LETTER S WITH MIDDLE TILDE
    {0x1D72, "t"},   // ᵲ  LATIN SMALL LETTER T WITH MIDDLE TILDE
    {0x1D7D, "p"},   // ᵽ  LATIN SMALL LETTER P WITH STROKE
    {0x1D85, "l"},   // ᶅ  LATIN SMALL LETTER L WITH PALATAL HOOK
    {0x1D86, "m"},   // ᶆ  LATIN SMALL LETTER M WITH PALATAL HOOK
    {0x1D87, "n"},   // ᶇ  LATIN SMALL LETTER N WITH PALATAL HOOK
    {0x1D88, "p"},   // ᶈ  LATIN SMALL LETTER P WITH PALATAL HOOK
    {0x1D89, "r"},   // ᶉ  LATIN SMALL LETTER R WITH PALATAL HOOK
    {0x1D8A, "s"},   // ᶊ  LATIN SMALL LETTER S WITH PALATAL HOOK
    {0x1D8C, "v"},   // ᶌ  LATIN SMALL LETTER V WITH PALATAL HOOK
    {0x1D8E, "z"},   // ᶎ  LATIN SMALL LETTER Z WITH PALATAL HOOK
    {0x1E9E, "SS"},  // ẞ  LATIN CAPITAL LETTER SHARP S
    {0x2039, "<"},   // ‹  SINGLE LEFT-POINTING ANGLE QUOTATION MARK
    {0x203A, ">"},   // ›  SINGLE RIGHT-POINTING ANGLE QUOTATION MARK
    {0x2190, "<-"},  // ←  LEFTWARDS ARROW
    {0x2192, "->"},  // →  RIGHTWARDS ARROW
};
// clang-format on

#ifndef NDEBUG
struct FoldTableSortChecker {
  FoldTableSortChecker() {
    for (size_t i = 1; i < std::size(kExtraFolds); ++i) {
      assert(kExtraFolds[i - 1].codepoint < kExtraFolds[i].codepoint);
    }
  }
};
static const FoldTableSortChecker kSortChecker;
#endif

const char *lookup_extra_fold(utf8proc_int32_t cp) {
  auto it = std::lower_bound(std::begin(kExtraFolds), std::end(kExtraFolds), cp,
                             [](const FoldEntry &entry, utf8proc_int32_t val) {
                               return entry.codepoint < val;
                             });
  if (it != std::end(kExtraFolds) && it->codepoint == cp) {
    return it->ascii;
  }
  return nullptr;
}

bool fold_codepoint_to_ascii(const utf8proc_uint8_t *data, utf8proc_ssize_t len,
                             std::string *out) {
  utf8proc_uint8_t *mapped_raw = nullptr;
  utf8proc_ssize_t mapped_len = utf8proc_map(
      data, len, &mapped_raw,
      static_cast<utf8proc_option_t>(UTF8PROC_STABLE | UTF8PROC_COMPAT |
                                     UTF8PROC_DECOMPOSE | UTF8PROC_STRIPMARK));
  // RAII guard: utf8proc_map allocates with malloc, free with free().
  std::unique_ptr<utf8proc_uint8_t, decltype(&free)> mapped(mapped_raw, &free);
  if (mapped_len <= 0) {
    return false;
  }
  for (utf8proc_ssize_t i = 0; i < mapped_len; ++i) {
    if (mapped_raw[i] >= 0x80) {
      return false;
    }
  }
  out->assign(reinterpret_cast<const char *>(mapped_raw),
              static_cast<size_t>(mapped_len));
  return true;
}

}  // namespace

std::vector<Token> AsciiFoldingTokenFilter::filter(
    std::vector<Token> tokens) const {
  for (auto &token : tokens) {
    bool all_ascii = true;
    for (unsigned char c : token.text) {
      if (c >= 0x80) {
        all_ascii = false;
        break;
      }
    }
    if (all_ascii) {
      continue;
    }

    std::string result;
    result.reserve(token.text.size());
    const auto *str =
        reinterpret_cast<const utf8proc_uint8_t *>(token.text.data());
    const auto len = static_cast<utf8proc_ssize_t>(token.text.size());
    utf8proc_ssize_t pos = 0;
    while (pos < len) {
      if (str[pos] < 0x80) {
        result.push_back(static_cast<char>(str[pos]));
        ++pos;
        continue;
      }

      utf8proc_int32_t cp;
      utf8proc_ssize_t bytes = utf8proc_iterate(str + pos, len - pos, &cp);
      if (bytes < 1) {
        result.push_back(static_cast<char>(str[pos]));
        ++pos;
        continue;
      }

      const char *fold = lookup_extra_fold(cp);
      if (fold) {
        result.append(fold);
        pos += bytes;
        continue;
      }

      std::string ascii;
      if (fold_codepoint_to_ascii(str + pos, bytes, &ascii)) {
        result.append(ascii);
      } else {
        // Keep the original codepoint when it has no ASCII equivalent.
        result.append(token.text, static_cast<size_t>(pos),
                      static_cast<size_t>(bytes));
      }
      pos += bytes;
    }
    token.text = std::move(result);
  }
  // Folding may leave empty tokens from empty input. Remove them.
  tokens.erase(std::remove_if(tokens.begin(), tokens.end(),
                              [](const Token &t) { return t.text.empty(); }),
               tokens.end());
  return tokens;
}

}  // namespace zvec::fts
