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

#include "db/common/binary_codec.h"
#include <gtest/gtest.h>

using namespace zvec;

TEST(BinaryWriterReaderTest, Uint8RoundTrip) {
  BinaryWriter writer;
  writer.PutUint8(0);
  writer.PutUint8(127);
  writer.PutUint8(255);

  BinaryReader reader(writer.data(), writer.size());
  uint8_t val;
  ASSERT_TRUE(reader.GetUint8(&val));
  EXPECT_EQ(val, 0);
  ASSERT_TRUE(reader.GetUint8(&val));
  EXPECT_EQ(val, 127);
  ASSERT_TRUE(reader.GetUint8(&val));
  EXPECT_EQ(val, 255);
  EXPECT_EQ(reader.remaining(), 0u);
}

TEST(BinaryWriterReaderTest, Uint32RoundTrip) {
  BinaryWriter writer;
  writer.PutUint32(0);
  writer.PutUint32(42);
  writer.PutUint32(0xFFFFFFFF);

  BinaryReader reader(writer.data(), writer.size());
  uint32_t val;
  ASSERT_TRUE(reader.GetUint32(&val));
  EXPECT_EQ(val, 0u);
  ASSERT_TRUE(reader.GetUint32(&val));
  EXPECT_EQ(val, 42u);
  ASSERT_TRUE(reader.GetUint32(&val));
  EXPECT_EQ(val, 0xFFFFFFFF);
}

TEST(BinaryWriterReaderTest, Uint64RoundTrip) {
  BinaryWriter writer;
  writer.PutUint64(0);
  writer.PutUint64(123456789012345ULL);
  writer.PutUint64(0xFFFFFFFFFFFFFFFFULL);

  BinaryReader reader(writer.data(), writer.size());
  uint64_t val;
  ASSERT_TRUE(reader.GetUint64(&val));
  EXPECT_EQ(val, 0u);
  ASSERT_TRUE(reader.GetUint64(&val));
  EXPECT_EQ(val, 123456789012345ULL);
  ASSERT_TRUE(reader.GetUint64(&val));
  EXPECT_EQ(val, 0xFFFFFFFFFFFFFFFFULL);
}

TEST(BinaryWriterReaderTest, Int32RoundTrip) {
  BinaryWriter writer;
  writer.PutInt32(0);
  writer.PutInt32(-1);
  writer.PutInt32(2147483647);
  writer.PutInt32(-2147483648);

  BinaryReader reader(writer.data(), writer.size());
  int32_t val;
  ASSERT_TRUE(reader.GetInt32(&val));
  EXPECT_EQ(val, 0);
  ASSERT_TRUE(reader.GetInt32(&val));
  EXPECT_EQ(val, -1);
  ASSERT_TRUE(reader.GetInt32(&val));
  EXPECT_EQ(val, 2147483647);
  ASSERT_TRUE(reader.GetInt32(&val));
  EXPECT_EQ(val, -2147483648);
}

TEST(BinaryWriterReaderTest, BoolRoundTrip) {
  BinaryWriter writer;
  writer.PutBool(true);
  writer.PutBool(false);

  BinaryReader reader(writer.data(), writer.size());
  bool val;
  ASSERT_TRUE(reader.GetBool(&val));
  EXPECT_TRUE(val);
  ASSERT_TRUE(reader.GetBool(&val));
  EXPECT_FALSE(val);
}

TEST(BinaryWriterReaderTest, StringRoundTrip) {
  BinaryWriter writer;
  writer.PutString("");
  writer.PutString("hello");
  writer.PutString("world with spaces and 特殊字符");

  BinaryReader reader(writer.data(), writer.size());
  std::string val;
  ASSERT_TRUE(reader.GetString(&val));
  EXPECT_EQ(val, "");
  ASSERT_TRUE(reader.GetString(&val));
  EXPECT_EQ(val, "hello");
  ASSERT_TRUE(reader.GetString(&val));
  EXPECT_EQ(val, "world with spaces and 特殊字符");
}

TEST(BinaryWriterReaderTest, MixedTypesRoundTrip) {
  BinaryWriter writer;
  writer.PutUint32(42);
  writer.PutString("test");
  writer.PutBool(true);
  writer.PutUint64(99999);
  writer.PutInt32(-5);

  BinaryReader reader(writer.data(), writer.size());
  uint32_t u32;
  std::string str;
  bool b;
  uint64_t u64;
  int32_t i32;

  ASSERT_TRUE(reader.GetUint32(&u32));
  EXPECT_EQ(u32, 42u);
  ASSERT_TRUE(reader.GetString(&str));
  EXPECT_EQ(str, "test");
  ASSERT_TRUE(reader.GetBool(&b));
  EXPECT_TRUE(b);
  ASSERT_TRUE(reader.GetUint64(&u64));
  EXPECT_EQ(u64, 99999u);
  ASSERT_TRUE(reader.GetInt32(&i32));
  EXPECT_EQ(i32, -5);
  EXPECT_EQ(reader.remaining(), 0u);
}

TEST(BinaryWriterReaderTest, ReadBeyondBuffer) {
  BinaryWriter writer;
  writer.PutUint8(1);

  BinaryReader reader(writer.data(), writer.size());
  uint8_t val;
  ASSERT_TRUE(reader.GetUint8(&val));

  // Buffer exhausted, further reads should fail
  EXPECT_FALSE(reader.GetUint8(&val));
  uint32_t u32;
  EXPECT_FALSE(reader.GetUint32(&u32));
  uint64_t u64;
  EXPECT_FALSE(reader.GetUint64(&u64));
  std::string str;
  EXPECT_FALSE(reader.GetString(&str));
}

TEST(BinaryWriterReaderTest, TruncatedString) {
  BinaryWriter writer;
  writer.PutUint32(1000);  // claim string length 1000 but no data follows

  BinaryReader reader(writer.data(), writer.size());
  std::string val;
  EXPECT_FALSE(reader.GetString(&val));
}

TEST(CRC32Test, KnownValues) {
  // CRC32 of empty data
  uint32_t crc_empty = CRC32::Compute(nullptr, 0);
  EXPECT_EQ(crc_empty, 0u);

  // CRC32 of "123456789" is a well-known test vector: 0xCBF43926
  const char *test_data = "123456789";
  uint32_t crc = CRC32::Compute(test_data, 9);
  EXPECT_EQ(crc, 0xCBF43926u);
}

TEST(CRC32Test, DifferentDataProducesDifferentCRC) {
  const char *data1 = "hello";
  const char *data2 = "world";
  uint32_t crc1 = CRC32::Compute(data1, 5);
  uint32_t crc2 = CRC32::Compute(data2, 5);
  EXPECT_NE(crc1, crc2);
}

TEST(CRC32Test, Deterministic) {
  const char *data = "test data for crc";
  uint32_t crc1 = CRC32::Compute(data, strlen(data));
  uint32_t crc2 = CRC32::Compute(data, strlen(data));
  EXPECT_EQ(crc1, crc2);
}
