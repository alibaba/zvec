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

#include "db/index/common/name_validation.h"
#include <array>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>

namespace zvec {
namespace {

struct Utf8NameValidator {
  Status (*validate)(std::string_view);
  size_t max_bytes;
  const char *prefix;
};

const std::array<Utf8NameValidator, 2> kUtf8NameValidators{{
    {ValidateDocumentId, kMaxDocumentIdBytes, "Invalid doc: id"},
    {ValidateCollectionName, kMaxCollectionNameBytes,
     "Invalid schema: collection name"},
}};

void ExpectInvalid(const Status &status, const std::string &message) {
  EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(status.message(), message);
  EXPECT_EQ(status.message().find("offset"), std::string::npos);
}

std::string Repeat(std::string_view text, size_t count) {
  std::string result;
  result.reserve(text.size() * count);
  for (size_t i = 0; i < count; ++i) {
    result.append(text.data(), text.size());
  }
  return result;
}

TEST(NameValidationTest, AcceptsUnicodePunctuationAndSpaces) {
  const std::vector<std::string> values{
      "a",
      "A_b-9",
      "https://example.com/docs/1?lang=zh#section",
      "a/b.c:d@e+f",
      "'quoted' \"text\" \\ value",
      " ",
      "   ",
      " leading and trailing ",
      u8"\u00A0\u3000",
      u8"中文",
      u8"€",
      u8"😀",
      u8"👩\u200D💻",
      u8"é",
      u8"e\u0301",
      u8"\U0010FFFF",
  };
  for (const auto &validator : kUtf8NameValidators) {
    SCOPED_TRACE(validator.prefix);
    for (const auto &value : values) {
      EXPECT_TRUE(validator.validate(value).ok());
    }
  }
}

TEST(NameValidationTest, RejectsEmptyNames) {
  for (const auto &validator : kUtf8NameValidators) {
    ExpectInvalid(validator.validate(std::string_view{}),
                  std::string(validator.prefix) + " must not be empty");
  }
  ExpectInvalid(ValidateFieldName(""),
                "Invalid schema: field name must not be empty");
}

TEST(NameValidationTest, MeasuresLimitsInUtf8Bytes) {
  for (const auto &validator : kUtf8NameValidators) {
    SCOPED_TRACE(validator.prefix);
    const auto max_bytes = validator.max_bytes;
    EXPECT_TRUE(validator.validate(std::string(max_bytes, 'a')).ok());
    auto emoji = Repeat(u8"😀", max_bytes / 4);
    ASSERT_EQ(emoji.size(), max_bytes);
    EXPECT_TRUE(validator.validate(emoji).ok());
    EXPECT_TRUE(
        validator.validate(std::string(max_bytes - 3, 'a') + u8"中").ok());

    const auto expected = std::string(validator.prefix) + " exceeds " +
                          std::to_string(max_bytes) + " bytes (got " +
                          std::to_string(max_bytes + 1) + ")";
    ExpectInvalid(validator.validate(std::string(max_bytes + 1, 'a')),
                  expected);
    ExpectInvalid(validator.validate(emoji + "a"), expected);
    ExpectInvalid(validator.validate(std::string(max_bytes - 2, 'a') + u8"中"),
                  expected);
  }
}

TEST(NameValidationTest, RejectsMalformedUtf8) {
  const std::vector<std::string> malformed{
      "\x80",  // Isolated continuation byte.
      "\xBF",
      "\xC2",  // Truncated sequences.
      "\xE4\xB8",
      "\xF0\x9F\x98",
      "\xC2"
      "A",  // Invalid continuation byte.
      "\xE2"
      "A"
      "\xAC",
      "\xC0\x80",          // Overlong NUL.
      "\xC1\xBF",          // Overlong two-byte sequence.
      "\xE0\x80\xAF",      // Overlong three-byte sequence.
      "\xF0\x80\x80\xAF",  // Overlong four-byte sequence.
      "\xED\xA0\x80",      // UTF-16 surrogate U+D800.
      "\xED\xBF\xBF",      // UTF-16 surrogate U+DFFF.
      "\xF4\x90\x80\x80",  // Above U+10FFFF.
      "\xF5\x80\x80\x80",  // Invalid lead byte.
      "\xFE",
      "\xFF",
  };
  for (const auto &validator : kUtf8NameValidators) {
    SCOPED_TRACE(validator.prefix);
    for (const auto &value : malformed) {
      ExpectInvalid(validator.validate(value),
                    std::string(validator.prefix) + " is not valid UTF-8");
      ExpectInvalid(validator.validate("prefix" + value),
                    std::string(validator.prefix) + " is not valid UTF-8");
    }
  }
}

TEST(NameValidationTest, HonorsStringViewLengthAndEmbeddedNulls) {
  const std::string backing = std::string(u8"中文") + "\xFF";
  for (const auto &validator : kUtf8NameValidators) {
    SCOPED_TRACE(validator.prefix);
    EXPECT_TRUE(validator.validate(std::string_view(backing.data(), 6)).ok());
    ExpectInvalid(validator.validate(std::string_view(backing.data(), 5)),
                  std::string(validator.prefix) + " is not valid UTF-8");
    ExpectInvalid(validator.validate(std::string("a\0b", 3)),
                  std::string(validator.prefix) + " contains a null character");
  }
}

TEST(NameValidationTest, RejectsEveryC0AndC1ControlByCodepoint) {
  for (const auto &validator : kUtf8NameValidators) {
    SCOPED_TRACE(validator.prefix);
    for (unsigned int codepoint = 0; codepoint <= 0x9F; ++codepoint) {
      if (codepoint >= 0x20 && codepoint < 0x7F) {
        continue;
      }
      SCOPED_TRACE(codepoint);
      std::string value;
      if (codepoint >= 0x80) {
        value += '\xC2';
      }
      value += static_cast<char>(codepoint);
      std::string reason = "contains a control character";
      if (codepoint == 0) {
        reason = "contains a null character";
      } else if (codepoint == '\n' || codepoint == '\r') {
        reason = "contains a newline";
      } else if (codepoint == '\t') {
        reason = "contains a tab";
      }
      ExpectInvalid(validator.validate("a" + value + "b"),
                    std::string(validator.prefix) + " " + reason);
    }
    // These continuation bytes overlap the C1 byte range, but their decoded
    // codepoints are ordinary letters/symbols and must not be rejected.
    EXPECT_TRUE(validator.validate(u8"中文€😀").ok());
  }
}

TEST(NameValidationTest, DistinguishesUnicodeLineAndParagraphSeparators) {
  for (const auto &validator : kUtf8NameValidators) {
    ExpectInvalid(validator.validate(u8"a\u2028b"),
                  std::string(validator.prefix) + " contains a line separator");
    ExpectInvalid(
        validator.validate(u8"a\u2029b"),
        std::string(validator.prefix) + " contains a paragraph separator");
  }
}

TEST(NameValidationTest, RetainsTheFieldAsciiCharacterSet) {
  for (const std::string name :
       {"a", "Z", "0", "_", "-", "a_b-c1", "123_test", "_zvec_custom"}) {
    EXPECT_TRUE(ValidateFieldName(name).ok());
  }
  EXPECT_TRUE(ValidateFieldName("ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                "abcdefghijklmnopqrstuvwxyz0123456789_-")
                  .ok());
  EXPECT_TRUE(ValidateFieldName(std::string(kMaxFieldNameBytes, 'a')).ok());
  ExpectInvalid(ValidateFieldName(std::string(kMaxFieldNameBytes + 1, 'a')),
                "Invalid schema: field name exceeds 64 bytes (got 65)");
  ExpectInvalid(ValidateFieldName(std::string(10000, 'a')),
                "Invalid schema: field name exceeds 64 bytes (got 10000)");
}

TEST(NameValidationTest, RejectsExactInternalFieldNames) {
  for (const std::string name :
       {"_zvec_row_id_", "_zvec_g_doc_id_", "_zvec_uid_", "_zvec_score",
        "_zvec_group_id"}) {
    SCOPED_TRACE(name);
    ExpectInvalid(ValidateFieldName(name),
                  "Invalid schema: field[" + name +
                      "] is reserved; use a different name");
    // The restriction is an exact match, not a new prefix or case policy.
    EXPECT_TRUE(ValidateFieldName(name + "_custom").ok());
    EXPECT_TRUE(ValidateDocumentId(name).ok());
    EXPECT_TRUE(ValidateCollectionName(name).ok());
  }
  EXPECT_TRUE(ValidateFieldName("_ZVEC_UID_").ok());
  for (const std::string name :
       {"_zvec_vector", "_zvec_sindices", "_zvec_svalues", "_zvec_is_valid"}) {
    EXPECT_TRUE(ValidateFieldName(name).ok());
  }
}

TEST(NameValidationTest, SharedErrorPreviewIsEscapedAndBounded) {
  EXPECT_EQ(FormatNameForError(""), "");
  EXPECT_EQ(FormatNameForError(std::string("a\0\n\r\t[]\\", 8)),
            "a\\0\\n\\r\\t\\[\\]\\\\");
  EXPECT_EQ(FormatNameForError(u8"中"), "\\xE4\\xB8\\xAD");
  EXPECT_EQ(FormatNameForError(std::string(10000, '\xff')),
            Repeat("\\xFF", 32) + "...");
  EXPECT_EQ(FormatNameForError(std::string(10000, 'x')),
            std::string(32, 'x') + "...");
}

TEST(NameValidationTest, DescribesInvalidFieldCharactersWithSafePreviews) {
  const std::string rule =
      "; use letters (A-Z, a-z), digits, underscores (_) or hyphens (-)";
  ExpectInvalid(ValidateFieldName("user name"),
                "Invalid schema: field[user name] contains a space" + rule);
  ExpectInvalid(
      ValidateFieldName("a.b"),
      "Invalid schema: field[a.b] contains an unsupported character" + rule);
  ExpectInvalid(ValidateFieldName(u8"中"),
                "Invalid schema: field[\\xE4\\xB8\\xAD] contains a non-ASCII "
                "character" +
                    rule);
  ExpectInvalid(
      ValidateFieldName("\x80"),
      "Invalid schema: field[\\x80] contains a non-ASCII character" + rule);
  ExpectInvalid(
      ValidateFieldName(std::string("a\0b", 3)),
      "Invalid schema: field[a\\0b] contains a null character" + rule);
  ExpectInvalid(ValidateFieldName("a\nb"),
                "Invalid schema: field[a\\nb] contains a newline" + rule);
  ExpectInvalid(ValidateFieldName("a\tb"),
                "Invalid schema: field[a\\tb] contains a tab" + rule);
  ExpectInvalid(
      ValidateFieldName("a\x1B"
                        "b"),
      "Invalid schema: field[a\\x1Bb] contains a control character" + rule);
  ExpectInvalid(ValidateFieldName("][\\\n"),
                "Invalid schema: field[\\]\\[\\\\\\n] contains an unsupported "
                "character" +
                    rule);
}

TEST(NameValidationTest, BoundsInvalidFieldPreviewsAndNeverEchoesRawBytes) {
  const std::string name(64, '\xFF');
  const auto status = ValidateFieldName(name);
  ExpectInvalid(status,
                "Invalid schema: field[" + Repeat("\\xFF", 32) +
                    "...] contains a non-ASCII character; use letters "
                    "(A-Z, a-z), digits, underscores (_) or hyphens (-)");
  EXPECT_LT(status.message().size(), 256u);
  for (unsigned char byte : status.message()) {
    EXPECT_GE(byte, 0x20);
    EXPECT_LE(byte, 0x7E);
  }
}

}  // namespace
}  // namespace zvec
