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

#include "db/index/common/manifest/pb_wire.h"
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <gtest/gtest.h>

namespace zvec {
namespace pbwire {
namespace {

TEST(PbWireWriterTest, VarintRoundTrip) {
  const struct {
    uint64_t value;
    size_t encoded_size;  // tag byte + varint bytes
  } cases[] = {
      {1, 2},
      {127, 2},
      {128, 3},
      {300, 3},
      {std::numeric_limits<uint32_t>::max(), 6},
      {std::numeric_limits<uint64_t>::max(), 11},
  };
  for (const auto &test_case : cases) {
    std::string buf;
    Writer writer(&buf);
    writer.PutVarint(1, test_case.value);
    EXPECT_EQ(buf.size(), test_case.encoded_size);

    Reader reader(buf);
    ASSERT_TRUE(reader.Next());
    EXPECT_EQ(reader.field(), 1u);
    EXPECT_EQ(reader.wire_type(), kVarint);
    EXPECT_EQ(reader.varint(), test_case.value);
    EXPECT_FALSE(reader.Next());
    EXPECT_TRUE(reader.ok());
  }
}

TEST(PbWireWriterTest, VarintByteLayoutMatchesProtobuf) {
  std::string buf;
  Writer writer(&buf);
  writer.PutVarint(1, 300);
  // Tag: field 1 with wire type 0; varint 300 encodes as 0xAC 0x02.
  const std::string expected("\x08\xAC\x02", 3);
  EXPECT_EQ(buf, expected);
}

TEST(PbWireWriterTest, DefaultValuesAreSkipped) {
  std::string buf;
  Writer writer(&buf);
  writer.PutVarint(1, 0);
  writer.PutBool(2, false);
  writer.PutFloat(3, 0.0f);
  writer.PutString(4, "");
  EXPECT_TRUE(buf.empty());
}

TEST(PbWireWriterTest, BoolRoundTrip) {
  std::string buf;
  Writer writer(&buf);
  writer.PutBool(7, true);

  Reader reader(buf);
  ASSERT_TRUE(reader.Next());
  EXPECT_EQ(reader.field(), 7u);
  EXPECT_EQ(reader.wire_type(), kVarint);
  EXPECT_TRUE(reader.bool_value());
  EXPECT_FALSE(reader.Next());
  EXPECT_TRUE(reader.ok());
}

TEST(PbWireWriterTest, FloatRoundTrip) {
  std::string buf;
  Writer writer(&buf);
  writer.PutFloat(5, 3.5f);
  writer.PutFloat(6, -1.25f);

  Reader reader(buf);
  ASSERT_TRUE(reader.Next());
  EXPECT_EQ(reader.field(), 5u);
  EXPECT_EQ(reader.wire_type(), kFixed32);
  EXPECT_FLOAT_EQ(reader.float_value(), 3.5f);
  ASSERT_TRUE(reader.Next());
  EXPECT_EQ(reader.field(), 6u);
  EXPECT_EQ(reader.wire_type(), kFixed32);
  EXPECT_FLOAT_EQ(reader.float_value(), -1.25f);
  EXPECT_FALSE(reader.Next());
  EXPECT_TRUE(reader.ok());
}

TEST(PbWireWriterTest, StringRoundTrip) {
  std::string buf;
  Writer writer(&buf);
  writer.PutString(2, "manifest");

  Reader reader(buf);
  ASSERT_TRUE(reader.Next());
  EXPECT_EQ(reader.field(), 2u);
  EXPECT_EQ(reader.wire_type(), kLenDelim);
  EXPECT_EQ(reader.bytes(), std::string_view("manifest"));
  EXPECT_EQ(reader.string_value(), "manifest");
  EXPECT_FALSE(reader.Next());
  EXPECT_TRUE(reader.ok());
}

TEST(PbWireWriterTest, AddStringWritesEmptyElements) {
  std::string buf;
  Writer writer(&buf);
  writer.AddString(2, "");
  // Tag for field 2 with wire type 2, followed by a zero length.
  const std::string expected("\x12\x00", 2);
  EXPECT_EQ(buf, expected);

  Reader reader(buf);
  ASSERT_TRUE(reader.Next());
  EXPECT_EQ(reader.field(), 2u);
  EXPECT_EQ(reader.wire_type(), kLenDelim);
  EXPECT_TRUE(reader.bytes().empty());
  EXPECT_TRUE(reader.ok());
}

TEST(PbWireWriterTest, PutMessageWritesEmptyPayload) {
  std::string buf;
  Writer writer(&buf);
  writer.PutMessage(3, "");
  // An empty but present sub-message is a zero-length field.
  const std::string expected("\x1A\x00", 2);
  EXPECT_EQ(buf, expected);
}

TEST(PbWireWriterTest, MultipleFieldsInOrder) {
  std::string buf;
  Writer writer(&buf);
  writer.PutVarint(1, 42);
  writer.PutString(2, "zvec");
  writer.PutBool(3, true);

  Reader reader(buf);
  ASSERT_TRUE(reader.Next());
  EXPECT_EQ(reader.field(), 1u);
  EXPECT_EQ(reader.uint32_value(), 42u);
  ASSERT_TRUE(reader.Next());
  EXPECT_EQ(reader.field(), 2u);
  EXPECT_EQ(reader.string_value(), "zvec");
  ASSERT_TRUE(reader.Next());
  EXPECT_EQ(reader.field(), 3u);
  EXPECT_TRUE(reader.bool_value());
  EXPECT_FALSE(reader.Next());
  EXPECT_TRUE(reader.ok());
}

TEST(PbWireWriterTest, Int32AndUint32ValueInterpretation) {
  std::string buf;
  Writer writer(&buf);
  // 0xFFFFFFFF is -1 when interpreted as an int32.
  writer.PutVarint(1, 0xFFFFFFFFULL);
  // Larger than 32 bits: uint32_value keeps the low 32 bits.
  writer.PutVarint(2, 0x1FFFFFFFFULL);

  Reader reader(buf);
  ASSERT_TRUE(reader.Next());
  EXPECT_EQ(reader.int32_value(), -1);
  ASSERT_TRUE(reader.Next());
  EXPECT_EQ(reader.uint32_value(), 0xFFFFFFFFu);
  EXPECT_FALSE(reader.Next());
  EXPECT_TRUE(reader.ok());
}

TEST(PbWireReaderTest, EmptyBufferEndsImmediately) {
  const std::string_view empty;
  Reader reader(empty);
  EXPECT_FALSE(reader.Next());
  EXPECT_TRUE(reader.ok());
}

TEST(PbWireReaderTest, CharSizeConstructorWorks) {
  const std::string buf("\x08\x2A", 2);
  Reader reader(buf.data(), buf.size());
  ASSERT_TRUE(reader.Next());
  EXPECT_EQ(reader.field(), 1u);
  EXPECT_EQ(reader.uint32_value(), 42u);
  EXPECT_FALSE(reader.Next());
  EXPECT_TRUE(reader.ok());
}

TEST(PbWireReaderTest, Fixed64IsAccepted) {
  // Tag for field 1 with wire type 1, followed by 8 little-endian bytes.
  const std::string buf("\x09\x01\x02\x03\x04\x05\x06\x07\x08", 9);
  Reader reader(buf);
  ASSERT_TRUE(reader.Next());
  EXPECT_EQ(reader.field(), 1u);
  EXPECT_EQ(reader.wire_type(), kFixed64);
  EXPECT_EQ(reader.varint(), 0x0807060504030201ULL);
  EXPECT_FALSE(reader.Next());
  EXPECT_TRUE(reader.ok());
}

TEST(PbWireReaderTest, TruncatedVarintFails) {
  // Continuation bit set, then end of buffer.
  const std::string buf("\x08\x80", 2);
  Reader reader(buf);
  EXPECT_FALSE(reader.Next());
  EXPECT_FALSE(reader.ok());
}

TEST(PbWireReaderTest, OverlongVarintFails) {
  // Ten bytes that all carry the continuation bit would need an eleventh.
  const std::string buf(10, '\xFF');
  Reader reader(buf);
  EXPECT_FALSE(reader.Next());
  EXPECT_FALSE(reader.ok());
}

TEST(PbWireReaderTest, TruncatedFixed32Fails) {
  const std::string buf("\x0D\x01\x02", 3);
  Reader reader(buf);
  EXPECT_FALSE(reader.Next());
  EXPECT_FALSE(reader.ok());
}

TEST(PbWireReaderTest, TruncatedFixed64Fails) {
  const std::string buf("\x09\x01\x02\x03", 4);
  Reader reader(buf);
  EXPECT_FALSE(reader.Next());
  EXPECT_FALSE(reader.ok());
}

TEST(PbWireReaderTest, TruncatedLengthDelimitedFails) {
  // Declares 10 payload bytes but provides only 2.
  const std::string buf(
      "\x12\x0A"
      "ab",
      4);
  Reader reader(buf);
  EXPECT_FALSE(reader.Next());
  EXPECT_FALSE(reader.ok());
}

TEST(PbWireReaderTest, TruncatedLengthVarintFails) {
  const std::string buf("\x12\x80", 2);
  Reader reader(buf);
  EXPECT_FALSE(reader.Next());
  EXPECT_FALSE(reader.ok());
}

TEST(PbWireReaderTest, FieldNumberZeroFails) {
  // A tag whose field number is 0 is illegal in protobuf.
  const std::string zero_tag(1, '\x00');
  Reader zero_reader(zero_tag);
  EXPECT_FALSE(zero_reader.Next());
  EXPECT_FALSE(zero_reader.ok());
}

TEST(PbWireReaderTest, GroupAndUnknownWireTypesFail) {
  // Field 1 with wire types 3 (start group), 4 (end group), 6 and 7.
  for (const char tag : {'\x0B', '\x0C', '\x0E', '\x0F'}) {
    const std::string buf(&tag, 1);
    Reader reader(buf);
    EXPECT_FALSE(reader.Next()) << "wire tag " << static_cast<int>(tag);
    EXPECT_FALSE(reader.ok()) << "wire tag " << static_cast<int>(tag);
  }
}

}  // namespace
}  // namespace pbwire
}  // namespace zvec
