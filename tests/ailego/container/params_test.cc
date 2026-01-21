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

#include <gtest/gtest.h>
#include <zvec/ailego/container/params.h>

using namespace zvec;
using namespace zvec::ailego;

TEST(Params, General) {
  Params params;
  Params params1;

  EXPECT_TRUE(params.empty());
  EXPECT_TRUE(params1.empty());
  params1.merge(params);

  char test_string1[] = "test1";
  params.set(test_string1, test_string1);

  const char *test_string2 = "test2";
  params.set(test_string2, test_string2);

  params.set("11111", "11111");
  params.set("22222", params1);
  params.set("true", true);
  params.set("-8", int8_t(-8));
  params.set("-16", int16_t(-16));
  params.set("-32", int32_t(-32));
  params.set("-64", int64_t(-64));
  params.set("8", uint8_t(8));
  params.set("16", uint16_t(16));
  params.set("32", uint32_t(32));
  params.set("64", uint64_t(64));
  params.set("100.0", float(100.0f));
  params.set("1000.0", double(1000.0f));
  params.set(std::string("string"), "string");

  EXPECT_FALSE(params.empty());

  EXPECT_TRUE(params.has("64"));
  EXPECT_TRUE(params.has("32"));
  EXPECT_TRUE(params.has("16"));
  EXPECT_TRUE(params.has("8"));
  EXPECT_TRUE(params.has("-64"));
  EXPECT_TRUE(params.has("-32"));
  EXPECT_TRUE(params.has("-16"));
  EXPECT_TRUE(params.has("-8"));
  EXPECT_TRUE(params.has("true"));
  EXPECT_FALSE(params.has("false"));
  EXPECT_FALSE(params.has(""));
  EXPECT_TRUE(params.has("100.0"));
  EXPECT_TRUE(params.has("1000.0"));
  EXPECT_FALSE(params.has("10000.0"));
  EXPECT_TRUE(params.has("string"));

  EXPECT_EQ("1", params.get_as_string("true"));
  EXPECT_EQ("-8", params.get_as_string("-8"));
  EXPECT_EQ("-16", params.get_as_string("-16"));
  EXPECT_EQ("-32", params.get_as_string("-32"));
  EXPECT_EQ("-64", params.get_as_string("-64"));
  EXPECT_EQ("8", params.get_as_string("8"));
  EXPECT_EQ("16", params.get_as_string("16"));
  EXPECT_EQ("32", params.get_as_string("32"));
  EXPECT_EQ("64", params.get_as_string("64"));
  EXPECT_EQ("100.000000", params.get_as_string("100.0"));
  EXPECT_EQ("1000.000000", params.get_as_string("1000.0"));
  EXPECT_EQ("", params.get_as_string("10000.0"));

  EXPECT_EQ(64u, params.get_as_uint64("64"));
  EXPECT_EQ(32u, params.get_as_uint64("32"));
  EXPECT_EQ(16u, params.get_as_uint64("16"));
  EXPECT_EQ(8u, params.get_as_uint64("8"));
  EXPECT_EQ((uint64_t)(-64), params.get_as_uint64("-64"));
  EXPECT_EQ((uint64_t)(-32), params.get_as_uint64("-32"));
  EXPECT_EQ((uint64_t)(-16), params.get_as_uint64("-16"));
  EXPECT_EQ((uint64_t)(-8), params.get_as_uint64("-8"));
  EXPECT_EQ(1u, params.get_as_uint64("true"));
  EXPECT_EQ(100u, params.get_as_uint64("100.0"));
  EXPECT_EQ(1000u, params.get_as_uint64("1000.0"));
  EXPECT_EQ(0u, params.get_as_uint64("10000.0"));

  EXPECT_EQ(64u, params.get_as_uint32("64"));
  EXPECT_EQ(32u, params.get_as_uint32("32"));
  EXPECT_EQ(16u, params.get_as_uint32("16"));
  EXPECT_EQ(8u, params.get_as_uint32("8"));
  EXPECT_EQ(4294967232u, params.get_as_uint32("-64"));
  EXPECT_EQ((uint32_t)(-32), params.get_as_uint32("-32"));
  EXPECT_EQ((uint32_t)(-16), params.get_as_uint32("-16"));
  EXPECT_EQ((uint32_t)(-8), params.get_as_uint32("-8"));
  EXPECT_EQ(1u, params.get_as_uint32("true"));
  EXPECT_EQ(100u, params.get_as_uint32("100.0"));
  EXPECT_EQ(1000u, params.get_as_uint32("1000.0"));
  EXPECT_EQ(0u, params.get_as_uint32("10000.0"));

  EXPECT_EQ(64u, params.get_as_uint16("64"));
  EXPECT_EQ(32u, params.get_as_uint16("32"));
  EXPECT_EQ(16u, params.get_as_uint16("16"));
  EXPECT_EQ(8u, params.get_as_uint16("8"));
  EXPECT_EQ(65472u, params.get_as_uint16("-64"));
  EXPECT_EQ(65504u, params.get_as_uint16("-32"));
  EXPECT_EQ((uint16_t)(-16), params.get_as_uint16("-16"));
  EXPECT_EQ((uint16_t)(-8), params.get_as_uint16("-8"));
  EXPECT_EQ(1u, params.get_as_uint16("true"));
  EXPECT_EQ(100u, params.get_as_uint16("100.0"));
  EXPECT_EQ(1000u, params.get_as_uint16("1000.0"));
  EXPECT_EQ(0u, params.get_as_uint16("10000.0"));

  EXPECT_EQ(64u, params.get_as_uint8("64"));
  EXPECT_EQ(32u, params.get_as_uint8("32"));
  EXPECT_EQ(32u, params.get_as_uint8("32"));
  EXPECT_EQ(8u, params.get_as_uint8("8"));
  EXPECT_EQ(192u, params.get_as_uint8("-64"));
  EXPECT_EQ(224u, params.get_as_uint8("-32"));
  EXPECT_EQ(240u, params.get_as_uint8("-16"));
  EXPECT_EQ((uint8_t)(-8), params.get_as_uint8("-8"));
  EXPECT_EQ(1u, params.get_as_uint8("true"));
  EXPECT_EQ(100u, params.get_as_uint8("100.0"));
  EXPECT_EQ(232u, params.get_as_uint8("1000.0"));
  EXPECT_EQ(0u, params.get_as_uint8("10000.0"));

  EXPECT_TRUE(params.get_as_bool("64"));
  EXPECT_TRUE(params.get_as_bool("32"));
  EXPECT_TRUE(params.get_as_bool("16"));
  EXPECT_TRUE(params.get_as_bool("8"));
  EXPECT_TRUE(params.get_as_bool("-64"));
  EXPECT_TRUE(params.get_as_bool("-32"));
  EXPECT_TRUE(params.get_as_bool("-16"));
  EXPECT_TRUE(params.get_as_bool("-8"));
  EXPECT_TRUE(params.get_as_bool("true"));
  EXPECT_FALSE(params.get_as_bool("false"));
  EXPECT_FALSE(params.get_as_bool(""));
  EXPECT_TRUE(params.get_as_bool("100.0"));
  EXPECT_TRUE(params.get_as_bool("1000.0"));
  EXPECT_FALSE(params.get_as_bool("10000.0"));
  EXPECT_FALSE(params.get_as_bool("string"));

  EXPECT_EQ(64, params.get_as_int64("64"));
  EXPECT_EQ(32, params.get_as_int64("32"));
  EXPECT_EQ(16, params.get_as_int64("16"));
  EXPECT_EQ(8, params.get_as_int64("8"));
  EXPECT_EQ(-64, params.get_as_int64("-64"));
  EXPECT_EQ(-32, params.get_as_int64("-32"));
  EXPECT_EQ(-16, params.get_as_int64("-16"));
  EXPECT_EQ(-8, params.get_as_int64("-8"));
  EXPECT_EQ(1, params.get_as_int64("true"));
  EXPECT_EQ(100, params.get_as_int64("100.0"));
  EXPECT_EQ(1000, params.get_as_int64("1000.0"));
  EXPECT_EQ(0, params.get_as_int64("10000.0"));

  EXPECT_EQ(64, params.get_as_int32("64"));
  EXPECT_EQ(32, params.get_as_int32("32"));
  EXPECT_EQ(16, params.get_as_int32("16"));
  EXPECT_EQ(8, params.get_as_int32("8"));
  EXPECT_EQ(-64, params.get_as_int32("-64"));
  EXPECT_EQ(-32, params.get_as_int32("-32"));
  EXPECT_EQ(-16, params.get_as_int32("-16"));
  EXPECT_EQ(-8, params.get_as_int32("-8"));
  EXPECT_EQ(1, params.get_as_int32("true"));
  EXPECT_EQ(100, params.get_as_int32("100.0"));
  EXPECT_EQ(1000, params.get_as_int32("1000.0"));
  EXPECT_EQ(0, params.get_as_int32("10000.0"));
  params1.merge(params);

  EXPECT_EQ(64, params.get_as_int16("64"));
  EXPECT_EQ(32, params.get_as_int16("32"));
  EXPECT_EQ(16, params.get_as_int16("16"));
  EXPECT_EQ(8, params.get_as_int16("8"));
  EXPECT_EQ(-64, params.get_as_int16("-64"));
  EXPECT_EQ(-32, params.get_as_int16("-32"));
  EXPECT_EQ(-16, params.get_as_int16("-16"));
  EXPECT_EQ(-8, params.get_as_int16("-8"));
  EXPECT_EQ(1, params.get_as_int16("true"));
  EXPECT_EQ(100, params.get_as_int16("100.0"));
  EXPECT_EQ(1000, params.get_as_int16("1000.0"));
  EXPECT_EQ(0, params.get_as_int16("10000.0"));
  params1.merge(params);

  EXPECT_EQ(64, params.get_as_int8("64"));
  EXPECT_EQ(32, params.get_as_int8("32"));
  EXPECT_EQ(16, params.get_as_int8("16"));
  EXPECT_EQ(8, params.get_as_int8("8"));
  EXPECT_EQ(-64, params.get_as_int8("-64"));
  EXPECT_EQ(-32, params.get_as_int8("-32"));
  EXPECT_EQ(-16, params.get_as_int8("-16"));
  EXPECT_EQ(-8, params.get_as_int8("-8"));
  EXPECT_EQ(1, params.get_as_int8("true"));
  EXPECT_EQ(100, params.get_as_int8("100.0"));
  EXPECT_EQ(-24, params.get_as_int8("1000.0"));
  EXPECT_EQ(0, params.get_as_int8("10000.0"));
  params1.merge(params);

  params.erase("64");
  params.erase("32");
  params.erase("16");
  params.erase("8");
  params.erase("-64");
  params.erase("-32");
  params.erase("-16");
  params.erase("-8");
  params.erase("true");
  params.erase("false");
  params.erase("");
  params.erase("100.0");
  params.erase("1000.0");
  params.erase("10000.0");
  params.erase("string");
  params1.merge(params);
  params.clear();
}

TEST(Params, OverloadedOperator) {
  Params params;
  Params params1;

  char test_string1[] = "test1";
  params[test_string1] = test_string1;

  const char *test_string2 = "test2";
  params[test_string2] = test_string2;

  params["11111"] = "11111";
  params["22222"] = params1;
  params["true"] = true;
  params["-8"] = int8_t(-8);
  params["-16"] = int16_t(-16);
  params["-32"] = int32_t(-32);
  params["-64"] = int64_t(-64);
  params["8"] = uint8_t(8);
  params["16"] = uint16_t(16);
  params["32"] = uint32_t(32);
  params["64"] = uint64_t(64);
  params["100.0"] = float(100.0f);
  params["1000.0"] = double(1000.0f);
  params["size_t"] = size_t(1234);
  params[std::string("string")] = std::string("string");

  EXPECT_EQ(64u, params.get_as_uint64("64"));
  EXPECT_EQ(32u, params.get_as_uint64("32"));
  EXPECT_EQ(16u, params.get_as_uint64("16"));
  EXPECT_EQ(8u, params.get_as_uint64("8"));
  EXPECT_EQ((uint64_t)(-64), params.get_as_uint64("-64"));
  EXPECT_EQ((uint64_t)(-32), params.get_as_uint64("-32"));
  EXPECT_EQ((uint64_t)(-16), params.get_as_uint64("-16"));
  EXPECT_EQ((uint64_t)(-8), params.get_as_uint64("-8"));
  EXPECT_EQ(1u, params.get_as_uint64("true"));
  EXPECT_EQ(100u, params.get_as_uint64("100.0"));
  EXPECT_EQ(1000u, params.get_as_uint64("1000.0"));
  EXPECT_EQ(0u, params.get_as_uint64("10000.0"));
  EXPECT_EQ(1234u, params.get_as_uint64("size_t"));
  EXPECT_EQ(1234u, params.get_as_uint32("size_t"));

  std::cout << "float: " << typeid(float).name() << std::endl;
  std::cout << "double: " << typeid(double).name() << std::endl;
  std::cout << "long double: " << typeid(long double).name() << std::endl;
  std::cout << "char: " << typeid(char).name() << std::endl;
  std::cout << "signed char: " << typeid(signed char).name() << std::endl;
  std::cout << "unsigned char: " << typeid(unsigned char).name() << std::endl;
  std::cout << "short int: " << typeid(short int).name() << std::endl;
  std::cout << "int: " << typeid(int).name() << std::endl;
  std::cout << "long int: " << typeid(long int).name() << std::endl;
  std::cout << "long long int: " << typeid(long long int).name() << std::endl;
  std::cout << "unsigned short int: " << typeid(unsigned short int).name()
            << std::endl;
  std::cout << "unsigned int: " << typeid(unsigned int).name() << std::endl;
  std::cout << "unsigned long int: " << typeid(unsigned long int).name()
            << std::endl;
  std::cout << "unsigned long long int: "
            << typeid(unsigned long long int).name() << std::endl;

  size_t size;
  EXPECT_TRUE(params.get("8", &size));
  EXPECT_TRUE(params.get("16", &size));
  EXPECT_TRUE(params.get("32", &size));
  EXPECT_TRUE(params.get("64", &size));
  EXPECT_TRUE(params.get("-8", &size));
  EXPECT_TRUE(params.get("-16", &size));
  EXPECT_TRUE(params.get("-32", &size));
  EXPECT_TRUE(params.get("-64", &size));
  EXPECT_TRUE(params.get("size_t", &size));
}

TEST(Params, GeneralString) {
  Params params;
  EXPECT_TRUE(params.empty());

  params.set("11111", "11111");
  params.set("22222", "22222");
  params.set("yes", "yes");
  params.set("no", "no");
  params.set("No", "No");
  params.set("Yes", "Yes");
  params.set("true", "true");
  params.set("True", "True");
  params.set("False", "False");
  params.set("false", "false");
  params.set("string", "string");

  EXPECT_TRUE(params.get_as_bool("yes"));
  EXPECT_TRUE(params.get_as_bool("Yes"));
  EXPECT_TRUE(params.get_as_bool("True"));
  EXPECT_TRUE(params.get_as_bool("true"));
  EXPECT_FALSE(params.get_as_bool("No"));
  EXPECT_FALSE(params.get_as_bool("no"));
  EXPECT_FALSE(params.get_as_bool("False"));
  EXPECT_FALSE(params.get_as_bool("false"));
  EXPECT_FALSE(params.get_as_bool("string"));

  EXPECT_TRUE(params.get_as_bool("11111"));
  EXPECT_EQ(103, params.get_as_int8("11111"));
  EXPECT_EQ(11111, params.get_as_int16("11111"));
  EXPECT_EQ(11111, params.get_as_int32("11111"));
  EXPECT_EQ(11111, params.get_as_int64("11111"));
  EXPECT_EQ(103u, params.get_as_uint8("11111"));
  EXPECT_EQ(11111u, params.get_as_uint16("11111"));
  EXPECT_EQ(11111u, params.get_as_uint32("11111"));
  EXPECT_EQ(11111u, params.get_as_uint64("11111"));
  EXPECT_FLOAT_EQ(11111.0, params.get_as_float("11111"));
  EXPECT_FLOAT_EQ(11111.0, params.get_as_double("11111"));

  EXPECT_TRUE(params.get_as_bool("22222"));
  EXPECT_EQ(-50, params.get_as_int8("22222"));
  EXPECT_EQ(22222, params.get_as_int16("22222"));
  EXPECT_EQ(22222, params.get_as_int32("22222"));
  EXPECT_EQ(22222, params.get_as_int64("22222"));
  EXPECT_EQ(206u, params.get_as_uint8("22222"));
  EXPECT_EQ(22222u, params.get_as_uint16("22222"));
  EXPECT_EQ(22222u, params.get_as_uint32("22222"));
  EXPECT_EQ(22222u, params.get_as_uint64("22222"));
  EXPECT_FLOAT_EQ(22222.0, params.get_as_float("22222"));
  EXPECT_FLOAT_EQ(22222.0, params.get_as_double("22222"));
}

TEST(Params, ParseFromEnvironment) {
  Params params;
  Params::ParseFromEnvironment(&params);
  std::cout << params.get_as_string("PATH") << std::endl;
}

TEST(Params, ParseFromBuffer) {
  std::string str =
      "{ -1111: -1111.11, -2222: -2222,  1111: 1111, 2222: "
      "\"2222\", 1: true, \'object\' : {  } }";
  Params params;
  Params::ParseFromBuffer(str, &params);

  ASSERT_FLOAT_EQ(-1111.11, params.get_as_float("-1111"));
  ASSERT_EQ(-2222, params.get_as_int32("-2222"));
  ASSERT_EQ(1111, params.get_as_int32("1111"));
  ASSERT_EQ(true, params.get_as_bool("1"));
  ASSERT_EQ(std::string("2222"), params.get_as_string("2222"));

  ASSERT_TRUE(params.has("object"));

  std::string str1 = "{proxima.general.cluster.count: 4000 }";
  Params::ParseFromBuffer(str1, &params);
  ASSERT_TRUE(params.has("proxima.general.cluster.count"));

  uint32_t count = 0;
  params.get("proxima.general.cluster.count", &count);
  ASSERT_EQ(4000u, count);
}

TEST(Params, SerializeToBuffer) {
  std::string str =
      "{ -1111: -1111.11, -2222: -2222,  1111: 1111, 2222: "
      "\"2222\", 1: true, \'object\' : "
      "{ \"eeee\": false, \'null\':null } }";
  Params params;
  Params::ParseFromBuffer(str, &params);
  params.set("unsupported_string_pointer", &str);
  params.set("supported_string", str);

  std::string str1 = params.debug_string();
  printf("%s\n", str1.c_str());

  Params params1;
  EXPECT_TRUE(Params::ParseFromBuffer(str1, &params1));
  EXPECT_EQ(str1, params1.debug_string());
}
