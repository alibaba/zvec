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

#include <cmath>
#include <limits>
#include <gtest/gtest.h>
#include <zvec/ailego/utility/string_helper.h>

using namespace zvec;

TEST(StringHelper, Split) {
  std::vector<std::string> out;

  ailego::StringHelper::Split("", ",", &out);
  EXPECT_EQ(1u, out.size());
  EXPECT_EQ("", out[0]);

  ailego::StringHelper::Split("", ';', &out);
  EXPECT_EQ(1u, out.size());
  EXPECT_EQ("", out[0]);

  ailego::StringHelper::Split("Hello, world!", "", &out);
  EXPECT_EQ(1u, out.size());
  EXPECT_EQ("Hello, world!", out[0]);

  ailego::StringHelper::Split("Hello, world!", '!', &out);
  EXPECT_EQ(2u, out.size());
  EXPECT_EQ("Hello, world", out[0]);
  EXPECT_EQ("", out[1]);

  ailego::StringHelper::Split("abxycdxyxydefxya", "xyz", &out);
  EXPECT_EQ(1u, out.size());
  EXPECT_EQ("abxycdxyxydefxya", out[0]);

  ailego::StringHelper::Split("abxycdxyxydefxya", 'a', &out);
  EXPECT_EQ(3u, out.size());
  EXPECT_EQ("", out[0]);
  EXPECT_EQ("bxycdxyxydefxy", out[1]);
  EXPECT_EQ("", out[2]);

  ailego::StringHelper::Split("abxycdxy!!xydefxya", "xy", &out);
  EXPECT_EQ(5u, out.size());
  EXPECT_EQ("ab", out[0]);
  EXPECT_EQ("cd", out[1]);
  EXPECT_EQ("!!", out[2]);
  EXPECT_EQ("def", out[3]);
  EXPECT_EQ("a", out[4]);

  ailego::StringHelper::Split("abxycdxy!!xydefxya", '!', &out);
  EXPECT_EQ(3u, out.size());
  EXPECT_EQ("abxycdxy", out[0]);
  EXPECT_EQ("", out[1]);
  EXPECT_EQ("xydefxya", out[2]);

  ailego::StringHelper::Split("abxycdxyxydefxya", "xy", &out);
  EXPECT_EQ(5u, out.size());
  EXPECT_EQ("ab", out[0]);
  EXPECT_EQ("cd", out[1]);
  EXPECT_EQ("", out[2]);
  EXPECT_EQ("def", out[3]);
  EXPECT_EQ("a", out[4]);

  ailego::StringHelper::Split("abxycdxyxydefxya", 'y', &out);
  EXPECT_EQ(5u, out.size());
  EXPECT_EQ("abx", out[0]);
  EXPECT_EQ("cdx", out[1]);
  EXPECT_EQ("x", out[2]);
  EXPECT_EQ("defx", out[3]);
  EXPECT_EQ("a", out[4]);

  ailego::StringHelper::Split("abxycdxyxydefxy", "xy", &out);
  EXPECT_EQ(5u, out.size());
  EXPECT_EQ("ab", out[0]);
  EXPECT_EQ("cd", out[1]);
  EXPECT_EQ("", out[2]);
  EXPECT_EQ("def", out[3]);
  EXPECT_EQ("", out[4]);

  ailego::StringHelper::Split("abxycdxyxydefxy", 'y', &out);
  EXPECT_EQ(5u, out.size());
  EXPECT_EQ("abx", out[0]);
  EXPECT_EQ("cdx", out[1]);
  EXPECT_EQ("x", out[2]);
  EXPECT_EQ("defx", out[3]);
  EXPECT_EQ("", out[4]);

  ailego::StringHelper::Split("xy", "xy", &out);
  EXPECT_EQ(2u, out.size());
  EXPECT_EQ("", out[0]);
  EXPECT_EQ("", out[1]);

  ailego::StringHelper::Split("x", 'x', &out);
  EXPECT_EQ(2u, out.size());
  EXPECT_EQ("", out[0]);
  EXPECT_EQ("", out[1]);
}

TEST(StringHelper, SplitFloat) {
  std::vector<float> out1;
  ailego::StringHelper::Split("1.0, tt, 2,", ',', &out1);
  EXPECT_EQ(4u, out1.size());
  EXPECT_FLOAT_EQ(1.0f, out1[0]);
  EXPECT_FLOAT_EQ(0.0f, out1[1]);
  EXPECT_FLOAT_EQ(2.0f, out1[2]);
  EXPECT_FLOAT_EQ(0.0f, out1[3]);

  std::vector<double> out2;
  ailego::StringHelper::Split("1.0, tt, 2,", ',', &out2);
  EXPECT_EQ(4u, out2.size());
  EXPECT_DOUBLE_EQ(1.0f, out2[0]);
  EXPECT_DOUBLE_EQ(0.0f, out2[1]);
  EXPECT_DOUBLE_EQ(2.0f, out2[2]);
  EXPECT_DOUBLE_EQ(0.0f, out2[3]);
}

TEST(StringHelper, SplitInteger) {
  std::vector<int32_t> out1;
  ailego::StringHelper::Split("-1.0, tt, 2,", ',', &out1);
  EXPECT_EQ(4u, out1.size());
  EXPECT_EQ(-1, out1[0]);
  EXPECT_EQ(0, out1[1]);
  EXPECT_EQ(2, out1[2]);
  EXPECT_EQ(0, out1[3]);

  std::vector<uint32_t> out2;
  ailego::StringHelper::Split("-1.0, tt, 2,", ',', &out2);
  EXPECT_EQ(4u, out2.size());
  EXPECT_EQ(0xffffffffu, out2[0]);
  EXPECT_EQ(0u, out2[1]);
  EXPECT_EQ(2u, out2[2]);
  EXPECT_EQ(0u, out2[3]);

  std::vector<int64_t> out3;
  ailego::StringHelper::Split("-1.0, tt, 2.3,", ',', &out3);
  EXPECT_EQ(4u, out3.size());
  EXPECT_EQ(-1, out3[0]);
  EXPECT_EQ(0, out3[1]);
  EXPECT_EQ(2, out3[2]);
  EXPECT_EQ(0, out3[3]);

  std::vector<uint64_t> out4;
  ailego::StringHelper::Split("-1.0, tt, 2.3,", ',', &out4);
  EXPECT_EQ(4u, out4.size());
  EXPECT_EQ((uint64_t)-1, out4[0]);
  EXPECT_EQ(0u, out4[1]);
  EXPECT_EQ(2u, out4[2]);
  EXPECT_EQ(0u, out4[3]);

  std::vector<int8_t> out5;
  ailego::StringHelper::Split("-1.0, tt, 2,", ',', &out5);
  EXPECT_EQ(4u, out5.size());
  EXPECT_EQ(-1, out5[0]);
  EXPECT_EQ(0, out5[1]);
  EXPECT_EQ(2, out5[2]);
  EXPECT_EQ(0, out5[3]);

  std::vector<uint8_t> out6;
  ailego::StringHelper::Split("-1.0, tt, 2,", ',', &out6);
  EXPECT_EQ(4u, out6.size());
  EXPECT_EQ(255u, out6[0]);
  EXPECT_EQ(0u, out6[1]);
  EXPECT_EQ(2u, out6[2]);
  EXPECT_EQ(0u, out6[3]);

  std::vector<int16_t> out7;
  ailego::StringHelper::Split("-1.0, tt, 2,", ',', &out7);
  EXPECT_EQ(4u, out7.size());
  EXPECT_EQ(-1, out7[0]);
  EXPECT_EQ(0, out7[1]);
  EXPECT_EQ(2, out7[2]);
  EXPECT_EQ(0, out7[3]);

  std::vector<uint16_t> out8;
  ailego::StringHelper::Split("-1.0, tt, 2,", ',', &out8);
  EXPECT_EQ(4u, out8.size());
  EXPECT_EQ(65535u, out8[0]);
  EXPECT_EQ(0u, out8[1]);
  EXPECT_EQ(2u, out8[2]);
  EXPECT_EQ(0u, out8[3]);
}

TEST(StringHelper, SplitWithTValidDelimeter) {
  std::vector<int32_t> out1;
  ailego::StringHelper::Split("12321", '2', &out1);
  EXPECT_EQ(3u, out1.size());
  EXPECT_EQ(1, out1[0]);
  EXPECT_EQ(3, out1[1]);
  EXPECT_EQ(1, out1[2]);

  std::vector<double> out2;
  ailego::StringHelper::Split("300e30e3", 'e', &out2);
  EXPECT_EQ(3u, out2.size());
  EXPECT_DOUBLE_EQ(300.0f, out2[0]);
  EXPECT_DOUBLE_EQ(30.0f, out2[1]);
  EXPECT_DOUBLE_EQ(3.0f, out2[2]);
}

TEST(StringHelper, SplitByString) {
  std::string sep = ",";
  std::vector<int32_t> out1;
  ailego::StringHelper::Split("-1.0, tt, 2,", sep, &out1);
  EXPECT_EQ(4u, out1.size());
  EXPECT_EQ(-1, out1[0]);
  EXPECT_EQ(0, out1[1]);
  EXPECT_EQ(2, out1[2]);
  EXPECT_EQ(0, out1[3]);

  std::vector<uint32_t> out2;
  ailego::StringHelper::Split("-1.0, tt, 2,", sep, &out2);
  EXPECT_EQ(4u, out2.size());
  EXPECT_EQ(0xffffffffu, out2[0]);
  EXPECT_EQ(0u, out2[1]);
  EXPECT_EQ(2u, out2[2]);
  EXPECT_EQ(0u, out2[3]);

  std::vector<int64_t> out3;
  ailego::StringHelper::Split("-1.0, tt, 2.3,", sep, &out3);
  EXPECT_EQ(4u, out3.size());
  EXPECT_EQ(-1, out3[0]);
  EXPECT_EQ(0, out3[1]);
  EXPECT_EQ(2, out3[2]);
  EXPECT_EQ(0, out3[3]);

  std::vector<uint64_t> out4;
  ailego::StringHelper::Split("-1.0, tt, 2.3,", sep, &out4);
  EXPECT_EQ(4u, out4.size());
  EXPECT_EQ((uint64_t)-1, out4[0]);
  EXPECT_EQ(0u, out4[1]);
  EXPECT_EQ(2u, out4[2]);
  EXPECT_EQ(0u, out4[3]);

  std::vector<int8_t> out5;
  ailego::StringHelper::Split("-1.0, tt, 2,", sep, &out5);
  EXPECT_EQ(4u, out5.size());
  EXPECT_EQ(-1, out5[0]);
  EXPECT_EQ(0, out5[1]);
  EXPECT_EQ(2, out5[2]);
  EXPECT_EQ(0, out5[3]);

  std::vector<uint8_t> out6;
  ailego::StringHelper::Split("-1.0, tt, 2,", sep, &out6);
  EXPECT_EQ(4u, out6.size());
  EXPECT_EQ(255u, out6[0]);
  EXPECT_EQ(0u, out6[1]);
  EXPECT_EQ(2u, out6[2]);
  EXPECT_EQ(0u, out6[3]);

  std::vector<int16_t> out7;
  ailego::StringHelper::Split("-1.0, tt, 2,", sep, &out7);
  EXPECT_EQ(4u, out7.size());
  EXPECT_EQ(-1, out7[0]);
  EXPECT_EQ(0, out7[1]);
  EXPECT_EQ(2, out7[2]);
  EXPECT_EQ(0, out7[3]);

  std::vector<uint16_t> out8;
  ailego::StringHelper::Split("-1.0, tt, 2,", sep, &out8);
  EXPECT_EQ(4u, out8.size());
  EXPECT_EQ(65535u, out8[0]);
  EXPECT_EQ(0u, out8[1]);
  EXPECT_EQ(2u, out8[2]);
  EXPECT_EQ(0u, out8[3]);
}

TEST(StringHelper, Trim) {
  std::string aaa = "  \t123 45 67\t\n8\r \n";
  EXPECT_EQ("123 45 67\t\n8\r \n", ailego::StringHelper::CopyLeftTrim(aaa));
  EXPECT_EQ("  \t123 45 67\t\n8", ailego::StringHelper::CopyRightTrim(aaa));
  EXPECT_EQ("123 45 67\t\n8", ailego::StringHelper::CopyTrim(aaa));

  std::string bbb = "  \t123 45 67\t\n8\r \n";
  ailego::StringHelper::LeftTrim(bbb);
  EXPECT_EQ("123 45 67\t\n8\r \n", bbb);

  std::string ccc = "  \t123 45 67\t\n8\r \n";
  ailego::StringHelper::RightTrim(ccc);
  EXPECT_EQ("  \t123 45 67\t\n8", ccc);

  std::string ddd = "  \t123 45 67\t\n8\r \n";
  ailego::StringHelper::Trim(ddd);
  EXPECT_EQ("123 45 67\t\n8", ddd);
}

TEST(StringHelper, CompareIgnoreCase) {
  {
    std::string a = "a b\tc\nd";
    std::string b = "A B\tC\nd";
    EXPECT_TRUE(ailego::StringHelper::CompareIgnoreCase(a, b));
  }
  {
    std::string a = "a d\tc\nd";
    std::string b = "A B\tC\nd";
    EXPECT_FALSE(ailego::StringHelper::CompareIgnoreCase(a, b));
  }
  {
    std::string a = "a d\tc\n";
    std::string b = "A B\tC\nd";
    EXPECT_FALSE(ailego::StringHelper::CompareIgnoreCase(a, b));
  }
  {
    std::string a = "A D\tc\n123456";
    std::string b = "A d\tC\n123456";
    EXPECT_TRUE(ailego::StringHelper::CompareIgnoreCase(a, b));
  }
  {
    std::string a = "A D\tc\n123456";
    std::string b = "";
    EXPECT_FALSE(ailego::StringHelper::CompareIgnoreCase(a, b));
  }
}

namespace zvec::ailego {
namespace testing {

TEST(StringHelperJoinAppend, Integer) {
  short a = -1;
  unsigned short b = 2;
  long c = -3;
  unsigned long d = 4;
  long long e = -5;
  unsigned long long f = 6;
  ssize_t g = -7;
  size_t h = 8;
  auto res = StringHelper::Concat(a, b, c, d, e, f, g, h);
  EXPECT_EQ(res, "-12-34-56-78");
  std::string str = "TEST";
  StringHelper::Append(&str, a, b, c, d, e, f, g, h);
  EXPECT_EQ(str, "TEST-12-34-56-78");
}

TEST(StringHelperJoinAppend, SizedInteger) {
  int8_t a = -1;
  uint8_t b = 2;
  int16_t c = -3;
  uint16_t d = 4;
  int32_t e = -5;
  uint32_t f = 6;
  int64_t g = -7;
  uint64_t h = 8;
  EXPECT_EQ("-12", StringHelper::Concat(a, b));
  EXPECT_EQ("-12-3", StringHelper::Concat(a, b, c));
  EXPECT_EQ("4-5", StringHelper::Concat(d, e));
  EXPECT_EQ("-78", StringHelper::Concat(g, h));

  auto res = StringHelper::Concat(a, b, c, d, e, f, g, h);
  EXPECT_EQ(res, "-12-34-56-78");
  std::string str = "TEST";
  StringHelper::Append(&str, a, b, c, d, e, f, g, h);
  EXPECT_EQ(str, "TEST-12-34-56-78");
}

TEST(StringHelperJoinAppend, MinMax) {
  auto a = StringHelper::Concat(
      std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max(),
      std::numeric_limits<int16_t>::min(), std::numeric_limits<int16_t>::max(),
      std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max(),
      std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max());
  EXPECT_EQ(a,
            "-128127-3276832767-21474836482147483647-"
            "92233720368547758089223372036854775807");
}

TEST(StringHelperJoinAppend, Float) {
  float f = 3.14f;
  double d = 6.28;
  long double ld = 9.42;
  auto a = StringHelper::Concat(
      f, d, ld, NAN, INFINITY, std::numeric_limits<float>::min(),
      std::numeric_limits<float>::max(), std::numeric_limits<double>::min(),
      std::numeric_limits<double>::max());
  EXPECT_EQ(a,
            "3.146.289.42naninf1.17549e-383.40282e+382.22507e-3081.79769e+308");
}

TEST(StringHelperJoinAppend, Enums) {
  enum { kOne = 1, kTen = 10 };
  enum class A : int64_t { kFirst = 100, kLast = 10000 };
  auto a = StringHelper::Concat(kOne, kTen, A::kFirst, A::kLast);
  EXPECT_EQ(a, "11010010000");
}

TEST(StringHelperJoinAppend, String) {
  auto a = StringHelper::Concat("a", std::string{"b"}, "c", std::string{"d"});
  EXPECT_EQ(a, "abcd");
  auto b = StringHelper::Concat("aaaa", std::string{"bbbb"}, "cccc",
                                std::string{"dddd"});
  EXPECT_EQ(b, "aaaabbbbccccdddd");
  auto c = StringHelper::Concat("aaaaaaaa", std::string{"bbbbbbbb"}, "cccccccc",
                                std::string{"dddddddd"});
  EXPECT_EQ(c, "aaaaaaaabbbbbbbbccccccccdddddddd");
}

TEST(StringHelperJoinAppend, ArbitaryNumberOfArguments) {
  auto a = StringHelper::Concat(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "a", "b", "c",
                                "d", "e", "f", "g", "h", "i", "j", "k", "l",
                                "m", "n", "o", "p", "q", "r", "s", "t", "u",
                                "v", "w", "x", "y", "z");
  EXPECT_EQ(a, "0123456789abcdefghijklmnopqrstuvwxyz");

  std::string str = "TEST";
  StringHelper::Append(&str, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "a", "b", "c", "d",
                       "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o",
                       "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z");
  EXPECT_EQ(str, "TEST0123456789abcdefghijklmnopqrstuvwxyz");
}

TEST(StringHelperJoinAppend, Empty) {
  for (const char *t :
       {"", "short string", "a very very very very very long string"}) {
    EXPECT_EQ(t, StringHelper::Concat(t));
    EXPECT_EQ(t, StringHelper::Concat(t, ""));
    EXPECT_EQ(t, StringHelper::Concat(t, "", ""));
    EXPECT_EQ(t, StringHelper::Concat(t, "", "", ""));
    EXPECT_EQ(t, StringHelper::Concat(t, "", "", "", ""));
    EXPECT_EQ(t, StringHelper::Concat(t, "", std::string{}, "", "", ""));
    EXPECT_EQ(t, StringHelper::Concat(t, "", std::string{}, "", std::string{},
                                      "", ""));

    std::string str = t;
    StringHelper::Append(&str);
    EXPECT_EQ(str, t);
    StringHelper::Append(&str, "");
    EXPECT_EQ(str, t);
    StringHelper::Append(&str, "", "");
    EXPECT_EQ(str, t);
    StringHelper::Append(&str, "", "", "");
    EXPECT_EQ(str, t);
    StringHelper::Append(&str, "", "", "", "");
    EXPECT_EQ(str, t);
    StringHelper::Append(&str, "", std::string{}, "", "", "");
    EXPECT_EQ(str, t);
    StringHelper::Append(&str, "", std::string{}, "", std::string{}, "", "");
    EXPECT_EQ(str, t);
  }
}

TEST(StringHelperJoinAppend, StringView) {
  StringView v1 = "hello";
  StringView v2 = v1;
  StringView v3 = nullptr;
  std::string foo = "foo";
  StringView v4 = foo;
  StringView v5 = "bar";
  StringView v6{v1.data() + 2, 2};
  auto s = StringHelper::Concat(v1, v2, v3, v4, v5, v6);
  EXPECT_EQ(s, "hellohellofoobarll");
}

TEST(StringHelper, SplitWithEmptySkipped) {
  std::vector<std::string> out;

  ailego::StringHelper::Split("", ",", &out, true);
  EXPECT_EQ(0u, out.size());

  ailego::StringHelper::Split(";1;", ';', &out, true);
  EXPECT_EQ(1u, out.size());
  EXPECT_EQ("1", out[0]);

  ailego::StringHelper::Split(";;;", ";", &out, true);
  EXPECT_EQ(0u, out.size());

  ailego::StringHelper::Split(";;;1", ';', &out, true);
  EXPECT_EQ(1u, out.size());
  EXPECT_EQ("1", out[0]);
}

}  // namespace testing
}  // namespace zvec::ailego
