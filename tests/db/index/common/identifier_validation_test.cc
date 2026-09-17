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

#include "db/index/common/identifier_validation.h"
#include <array>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#include "db/common/utils.h"

namespace zvec {
namespace {

struct Utf8NameValidator {
  Status (*validate)(std::string_view);
  size_t max_bytes;
  const char *prefix;
};

const std::array<Utf8NameValidator, 2> kUtf8NameValidators{{
    {validate_document_id, kMaxDocumentIdBytes, "Invalid doc: id"},
    {validate_collection_name, kMaxCollectionNameBytes,
     "Invalid schema: collection name"},
}};

void ExpectInvalid(const Status &status, const std::string &message) {
  EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(status.message(), message);
  EXPECT_EQ(status.message().find("offset"), std::string::npos);
}

void ExpectInvalidName(const Utf8NameValidator &validator,
                       std::string_view value, const std::string &reason) {
  ExpectInvalid(
      validator.validate(value),
      std::string(validator.prefix) + "[" + format_name(value) + "] " + reason);
}

std::string Repeat(std::string_view text, size_t count) {
  std::string result;
  result.reserve(text.size() * count);
  for (size_t i = 0; i < count; ++i) {
    result.append(text.data(), text.size());
  }
  return result;
}

TEST(IdentifierValidationTest, AcceptsUnicodePunctuationAndSpaces) {
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

TEST(IdentifierValidationTest, RejectsEmptyNames) {
  for (const auto &validator : kUtf8NameValidators) {
    ExpectInvalid(validator.validate(std::string_view{}),
                  std::string(validator.prefix) + " must not be empty");
  }
  ExpectInvalid(validate_field_name(""),
                "Invalid schema: field name must not be empty");
}

TEST(IdentifierValidationTest, MeasuresLimitsInUtf8Bytes) {
  for (const auto &validator : kUtf8NameValidators) {
    SCOPED_TRACE(validator.prefix);
    const auto max_bytes = validator.max_bytes;
    EXPECT_TRUE(validator.validate(std::string(max_bytes, 'a')).ok());
    auto emoji = Repeat(u8"😀", max_bytes / 4);
    ASSERT_EQ(emoji.size(), max_bytes);
    EXPECT_TRUE(validator.validate(emoji).ok());
    EXPECT_TRUE(
        validator.validate(std::string(max_bytes - 3, 'a') + u8"中").ok());

    const auto reason = "exceeds " + std::to_string(max_bytes) +
                        " bytes (got " + std::to_string(max_bytes + 1) + ")";
    ExpectInvalidName(validator, std::string(max_bytes + 1, 'a'), reason);
    ExpectInvalidName(validator, emoji + "a", reason);
    ExpectInvalidName(validator, std::string(max_bytes - 2, 'a') + u8"中",
                      reason);
  }
}

TEST(IdentifierValidationTest, RejectsMalformedUtf8) {
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
      ExpectInvalidName(validator, value, "is not valid UTF-8");
      ExpectInvalidName(validator, "prefix" + value, "is not valid UTF-8");
    }
  }
}

TEST(IdentifierValidationTest, HonorsStringViewLengthAndEmbeddedNulls) {
  const std::string backing = std::string(u8"中文") + "\xFF";
  for (const auto &validator : kUtf8NameValidators) {
    SCOPED_TRACE(validator.prefix);
    EXPECT_TRUE(validator.validate(std::string_view(backing.data(), 6)).ok());
    ExpectInvalidName(validator, std::string_view(backing.data(), 5),
                      "is not valid UTF-8");
    ExpectInvalidName(validator, std::string("a\0b", 3),
                      "contains a null character");
  }
}

TEST(IdentifierValidationTest, RejectsEveryC0AndC1ControlByCodepoint) {
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
      ExpectInvalidName(validator, "a" + value + "b", reason);
    }
    // These continuation bytes overlap the C1 byte range, but their decoded
    // codepoints are ordinary letters/symbols and must not be rejected.
    EXPECT_TRUE(validator.validate(u8"中文€😀").ok());
  }
}

TEST(IdentifierValidationTest, DistinguishesUnicodeLineAndParagraphSeparators) {
  for (const auto &validator : kUtf8NameValidators) {
    ExpectInvalidName(validator, u8"a\u2028b", "contains a line separator");
    ExpectInvalidName(validator, u8"a\u2029b",
                      "contains a paragraph separator");
  }
}

TEST(IdentifierValidationTest, Utf8ErrorsIncludeEscapedAndBoundedPreviews) {
  for (const auto &validator : kUtf8NameValidators) {
    const std::string prefix = validator.prefix;
    ExpectInvalid(validator.validate("order\n[123]"),
                  prefix + "[order\\n\\[123\\]] contains a newline");
    ExpectInvalid(validator.validate("order\xff"),
                  prefix + "[order\\xFF] is not valid UTF-8");
    ExpectInvalid(
        validator.validate(std::string(40, 'a') + "\n"),
        prefix + "[" + std::string(32, 'a') + "...] contains a newline");
    const auto status = validator.validate(std::string(10000, '\xff'));
    EXPECT_EQ(status.message().find(prefix + "[" + Repeat("\\xFF", 32) +
                                    "...] exceeds "),
              0u);
    EXPECT_LT(status.message().size(), 256u);
    for (unsigned char byte : status.message()) {
      EXPECT_GE(byte, 0x20);
      EXPECT_LE(byte, 0x7E);
    }
  }
}

TEST(IdentifierValidationTest, RetainsTheFieldAsciiCharacterSet) {
  for (const std::string name :
       {"a", "Z", "0", "_", "-", "a_b-c1", "123_test", "_zvec_custom"}) {
    EXPECT_TRUE(validate_field_name(name).ok());
  }
  EXPECT_TRUE(validate_field_name("ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                  "abcdefghijklmnopqrstuvwxyz0123456789_-")
                  .ok());
  EXPECT_TRUE(validate_field_name(std::string(kMaxFieldNameBytes, 'a')).ok());
  ExpectInvalid(validate_field_name(std::string(kMaxFieldNameBytes + 1, 'a')),
                "Invalid schema: field name exceeds 64 bytes (got 65)");
  ExpectInvalid(validate_field_name(std::string(10000, 'a')),
                "Invalid schema: field name exceeds 64 bytes (got 10000)");
}

TEST(IdentifierValidationTest, RejectsExactInternalFieldNames) {
  for (const std::string name :
       {"_zvec_row_id_", "_zvec_g_doc_id_", "_zvec_uid_", "_zvec_score",
        "_zvec_group_id"}) {
    SCOPED_TRACE(name);
    ExpectInvalid(validate_field_name(name),
                  "Invalid schema: field[" + name +
                      "] is reserved; use a different name");
    // The restriction is an exact match, not a new prefix or case policy.
    EXPECT_TRUE(validate_field_name(name + "_custom").ok());
    EXPECT_TRUE(validate_document_id(name).ok());
    EXPECT_TRUE(validate_collection_name(name).ok());
  }
  EXPECT_TRUE(validate_field_name("_ZVEC_UID_").ok());
  EXPECT_TRUE(validate_field_name("_zvec_is_valid").ok());
}

TEST(IdentifierValidationTest, SharedErrorPreviewIsEscapedAndBounded) {
  EXPECT_EQ(format_name(""), "");
  EXPECT_EQ(format_name(std::string("a\0\n\r\t[]\\", 8)),
            "a\\0\\n\\r\\t\\[\\]\\\\");
  EXPECT_EQ(format_name(u8"中"), "\\xE4\\xB8\\xAD");
  EXPECT_EQ(format_name(std::string(10000, '\xff')),
            Repeat("\\xFF", 32) + "...");
  EXPECT_EQ(format_name(std::string(10000, 'x')), std::string(32, 'x') + "...");
}

TEST(IdentifierValidationTest,
     DescribesInvalidFieldCharactersWithSafePreviews) {
  const std::string rule =
      "; use letters (A-Z, a-z), digits, underscores (_) or hyphens (-)";
  ExpectInvalid(validate_field_name("user name"),
                "Invalid schema: field[user name] contains a space" + rule);
  ExpectInvalid(
      validate_field_name("a.b"),
      "Invalid schema: field[a.b] contains an unsupported character" + rule);
  ExpectInvalid(validate_field_name(u8"中"),
                "Invalid schema: field[\\xE4\\xB8\\xAD] contains a non-ASCII "
                "character" +
                    rule);
  ExpectInvalid(
      validate_field_name("\x80"),
      "Invalid schema: field[\\x80] contains a non-ASCII character" + rule);
  ExpectInvalid(
      validate_field_name(std::string("a\0b", 3)),
      "Invalid schema: field[a\\0b] contains a null character" + rule);
  ExpectInvalid(validate_field_name("a\nb"),
                "Invalid schema: field[a\\nb] contains a newline" + rule);
  ExpectInvalid(validate_field_name("a\tb"),
                "Invalid schema: field[a\\tb] contains a tab" + rule);
  ExpectInvalid(
      validate_field_name("a\x1B"
                          "b"),
      "Invalid schema: field[a\\x1Bb] contains a control character" + rule);
  ExpectInvalid(validate_field_name("][\\\n"),
                "Invalid schema: field[\\]\\[\\\\\\n] contains an unsupported "
                "character" +
                    rule);
}

TEST(IdentifierValidationTest,
     BoundsInvalidFieldPreviewsAndNeverEchoesRawBytes) {
  const std::string name(64, '\xFF');
  const auto status = validate_field_name(name);
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
