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

#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/container/cube.h>

using namespace zvec::ailego;

TEST(Cube, TypeInfo) {
  std::cout << "* bool:                " << typeid(bool).name() << std::endl;

  std::cout << "* int8_t:              " << typeid(int8_t).name() << std::endl;
  std::cout << "* char:                " << typeid(char).name() << std::endl;
  std::cout << "* signed char:         " << typeid(signed char).name()
            << std::endl;
  std::cout << "* uint8_t:             " << typeid(uint8_t).name() << std::endl;
  std::cout << "* unsigned char:       " << typeid(unsigned char).name()
            << std::endl;

  std::cout << "* int16_t:             " << typeid(int16_t).name() << std::endl;
  std::cout << "* short:               " << typeid(short).name() << std::endl;
  std::cout << "* signed short:        " << typeid(signed short).name()
            << std::endl;
  std::cout << "* uint16_t:            " << typeid(uint16_t).name()
            << std::endl;
  std::cout << "* unsigned short:      " << typeid(unsigned short).name()
            << std::endl;

  std::cout << "* int32_t:             " << typeid(int32_t).name() << std::endl;
  std::cout << "* int:                 " << typeid(int).name() << std::endl;
  std::cout << "* signed int:          " << typeid(signed int).name()
            << std::endl;
  std::cout << "* uint32_t:            " << typeid(uint32_t).name()
            << std::endl;
  std::cout << "* unsigned int:        " << typeid(unsigned int).name()
            << std::endl;

  std::cout << "* int64_t:             " << typeid(int64_t).name() << std::endl;
  std::cout << "* long:                " << typeid(long).name() << std::endl;
  std::cout << "* signed long:         " << typeid(signed long).name()
            << std::endl;
  std::cout << "* uint64_t:            " << typeid(uint64_t).name()
            << std::endl;
  std::cout << "* unsigned long:       " << typeid(unsigned long).name()
            << std::endl;

  std::cout << "* long long:           " << typeid(long).name() << std::endl;
  std::cout << "* signed long long:    " << typeid(signed long).name()
            << std::endl;
  std::cout << "* unsigned long long:  " << typeid(unsigned long).name()
            << std::endl;
}

TEST(Cube, General) {
  Cube cube1 = 11111;
  EXPECT_EQ(11111, cube1.unsafe_cast<int>());
  EXPECT_EQ(11111, cube1.cast<int>());
  int int1 = cube1;
  EXPECT_EQ(11111, int1);
  EXPECT_TRUE(!cube1.empty());
  EXPECT_EQ(sizeof(11111), cube1.size());

  Cube cube2 = 22222;
  EXPECT_EQ(22222, cube2.unsafe_cast<int>());
  EXPECT_EQ(22222, cube2.cast<int>());
  int int2 = (const int &)cube2;
  EXPECT_EQ(22222, int2);
  EXPECT_TRUE(!cube2.empty());
  EXPECT_EQ(sizeof(22222), cube2.size());

  Cube cube3 = std::vector<int>();
  cube3.unsafe_cast<std::vector<int>>().push_back(1);
  cube3.unsafe_cast<std::vector<int>>().push_back(2);
  cube3.unsafe_cast<std::vector<int>>().push_back(3);
  EXPECT_EQ(3u, cube3.unsafe_cast<std::vector<int>>().size());
  EXPECT_EQ(3u, cube3.cast<std::vector<int>>().size());
  std::vector<int> &vec3 = cube3;
  EXPECT_EQ(3u, vec3.size());
  EXPECT_TRUE(!cube3.empty());
  EXPECT_EQ(sizeof(std::vector<int>), cube3.size());

  std::vector<long> vec4;
  vec4.push_back(1);
  vec4.push_back(2);
  vec4.push_back(3);
  vec4.push_back(4);
  Cube cube4 = vec4;
  EXPECT_EQ(4u, cube4.unsafe_cast<std::vector<long>>().size());
  EXPECT_EQ(4u, cube4.cast<std::vector<long>>().size());
  const std::vector<long> &vec44 = cube4;
  EXPECT_EQ(4u, vec44.size());
  EXPECT_TRUE(!cube4.empty());
  EXPECT_EQ(sizeof(std::vector<long>), cube4.size());

  Cube cube5, cube6;
  EXPECT_TRUE(cube5.empty());
  EXPECT_TRUE(cube6.empty());
  EXPECT_EQ(cube5.type(), cube6.type());
  EXPECT_EQ(0u, cube5.size());
  EXPECT_EQ(0u, cube6.size());

  EXPECT_EQ(cube1.type(), cube2.type());
  EXPECT_NE(cube3.type(), cube4.type());
  EXPECT_NE(cube1.type(), cube3.type());
  EXPECT_NE(cube2.type(), cube4.type());
  EXPECT_NE(cube1.type(), cube5.type());
  EXPECT_NE(cube2.type(), cube5.type());
  EXPECT_NE(cube3.type(), cube5.type());
  EXPECT_NE(cube4.type(), cube5.type());
  EXPECT_TRUE(cube1.compatible(cube2));
  EXPECT_TRUE(cube5.compatible(cube6));
  EXPECT_FALSE(cube1.compatible(cube3));
  EXPECT_FALSE(cube3.compatible(cube5));

  cube1.reset();
  cube3.reset();
  cube5.reset();
  cube6.reset();
  EXPECT_TRUE(cube1.empty());
  EXPECT_TRUE(cube3.empty());
  EXPECT_TRUE(cube5.empty());
  EXPECT_TRUE(cube6.empty());
}

TEST(Cube, LargeObject) {
  std::string str1("1111111");
  std::string str2("2222222");
  std::string str3("3333333");
  std::string str4("4444444");
  std::string str5("5555555");
  std::string str6("6666666");
  std::string str7("7777777");

  Cube cube1(str1);
  Cube cube2;
  cube2 = str2;
  Cube cube3 = str3;

  EXPECT_EQ(str1, cube1.cast<std::string>());
  EXPECT_EQ(str2, cube2.cast<std::string>());
  EXPECT_TRUE(cube1.compatible(cube2));

  cube1 = std::move(cube2);
  EXPECT_EQ(str2, cube1.cast<std::string>());
  EXPECT_TRUE(cube2.empty());
  EXPECT_FALSE(cube1.compatible(cube2));

  EXPECT_EQ(str3, cube3.cast<std::string>());
  cube3 = cube1;
  EXPECT_EQ(str2, cube3.cast<std::string>());
  EXPECT_EQ(str2, cube1.cast<std::string>());

  // Test Constructor Cube(T &&rhs) / Cube(const T &rhs)
  Cube cube41(std::string("444444"));
  Cube cube42(str4);
  EXPECT_NE(std::string(""), str4);
  Cube cube43(std::move(str4));
  EXPECT_EQ(std::string(""), str4);

  const std::string str41 = str4;
  Cube cube44(str41);
  EXPECT_EQ(str41, str4);
  EXPECT_EQ(str4, cube44.cast<std::string>());

  // Test Assignment operator=(T &&rhs) / operator=(const T &rhs)
  Cube cube51, cube52, cube53, cube54;
  cube51 = std::string("55555");
  cube52 = str5;
  EXPECT_NE(std::string(""), str5);
  cube53 = std::move(str5);
  EXPECT_EQ(std::string(""), str5);

  const std::string str51 = str5;
  cube54 = str51;
  EXPECT_EQ(str51, str5);
  EXPECT_EQ(str5, cube54.cast<std::string>());

  // Test Constructor Cube(Cube &&rhs) / Cube(const Cube &rhs)
  Cube cube6(str6);
  Cube cube61(cube6);
  EXPECT_EQ(str6, cube61.cast<std::string>());
  EXPECT_FALSE(cube6.empty());
  Cube cube62(std::move(cube6));
  EXPECT_EQ(str6, cube62.cast<std::string>());
  EXPECT_TRUE(cube6.empty());

  const Cube cube63 = cube62;
  Cube cube64(cube63);
  EXPECT_EQ(str6, cube64.cast<std::string>());
  EXPECT_FALSE(cube63.empty());

  // Test Assignment operator=(Cube &&rhs) / operator=(const Cube &rhs)
  Cube cube7(str7), cube71, cube72;
  cube71 = cube7;
  EXPECT_EQ(str7, cube71.cast<std::string>());
  EXPECT_FALSE(cube7.empty());
  cube72 = std::move(cube7);
  EXPECT_EQ(str7, cube72.cast<std::string>());
  EXPECT_TRUE(cube7.empty());

  const Cube cube73(cube72);
  Cube cube74;
  cube74 = cube73;
  EXPECT_EQ(str7, cube74.cast<std::string>());
  EXPECT_EQ(str7, cube73.cast<std::string>());
  EXPECT_FALSE(cube74.empty());
}

struct SmallObject {
  SmallObject() {
    ++assign_count;
  }

  SmallObject(const SmallObject &) {
    ++clone_count;
  }

  SmallObject(SmallObject &&) {
    ++move_count;
  }

  ~SmallObject() {
    ++cleanup_count;
  }

  int val{0};
  static int assign_count;
  static int clone_count;
  static int move_count;
  static int cleanup_count;
};

int SmallObject::assign_count = 0;
int SmallObject::clone_count = 0;
int SmallObject::move_count = 0;
int SmallObject::cleanup_count = 0;

TEST(Cube, CubePolicy) {
  EXPECT_EQ(0, SmallObject::assign_count);
  EXPECT_EQ(0, SmallObject::clone_count);
  EXPECT_EQ(0, SmallObject::move_count);
  EXPECT_EQ(0, SmallObject::cleanup_count);

  SmallObject obj1, obj2, obj3, obj4, obj5;
  EXPECT_EQ(5, SmallObject::assign_count);
  EXPECT_EQ(0, SmallObject::clone_count);
  EXPECT_EQ(0, SmallObject::move_count);
  EXPECT_EQ(0, SmallObject::cleanup_count);

  Cube cube1(obj1);
  EXPECT_EQ(5, SmallObject::assign_count);
  EXPECT_EQ(1, SmallObject::clone_count);
  EXPECT_EQ(0, SmallObject::move_count);
  EXPECT_EQ(0, SmallObject::cleanup_count);

  Cube cube2(std::move(obj2));
  EXPECT_EQ(5, SmallObject::assign_count);
  EXPECT_EQ(1, SmallObject::clone_count);
  EXPECT_EQ(1, SmallObject::move_count);
  EXPECT_EQ(0, SmallObject::cleanup_count);

  {
    Cube cube3(std::move(obj3));
    EXPECT_EQ(5, SmallObject::assign_count);
    EXPECT_EQ(1, SmallObject::clone_count);
    EXPECT_EQ(2, SmallObject::move_count);
    EXPECT_EQ(0, SmallObject::cleanup_count);
  }

  EXPECT_EQ(5, SmallObject::assign_count);
  EXPECT_EQ(1, SmallObject::clone_count);
  EXPECT_EQ(2, SmallObject::move_count);
  EXPECT_EQ(1, SmallObject::cleanup_count);

  {
    Cube cube4(obj4);
    EXPECT_EQ(5, SmallObject::assign_count);
    EXPECT_EQ(2, SmallObject::clone_count);
    EXPECT_EQ(2, SmallObject::move_count);
    EXPECT_EQ(1, SmallObject::cleanup_count);
  }

  EXPECT_EQ(5, SmallObject::assign_count);
  EXPECT_EQ(2, SmallObject::clone_count);
  EXPECT_EQ(2, SmallObject::move_count);
  EXPECT_EQ(2, SmallObject::cleanup_count);

  {
    Cube cube5(obj5);
    EXPECT_EQ(5, SmallObject::assign_count);
    EXPECT_EQ(3, SmallObject::clone_count);
    EXPECT_EQ(2, SmallObject::move_count);
    EXPECT_EQ(2, SmallObject::cleanup_count);
  }

  EXPECT_EQ(5, SmallObject::assign_count);
  EXPECT_EQ(3, SmallObject::clone_count);
  EXPECT_EQ(2, SmallObject::move_count);
  EXPECT_EQ(3, SmallObject::cleanup_count);
}

TEST(Cube, SmallObject) {
  uint64_t uint1 = 1111111;
  uint64_t uint2 = 2222222;
  uint64_t uint3 = 3333333;
  uint64_t uint4 = 4444444;
  uint64_t uint5 = 5555555;
  uint64_t uint6 = 6666666;
  uint64_t uint7 = 7777777;

  Cube cube1(uint1);
  Cube cube2;
  cube2 = uint2;
  Cube cube3 = uint3;

  EXPECT_EQ(uint1, cube1.cast<uint64_t>());
  EXPECT_EQ(uint2, cube2.cast<uint64_t>());
  EXPECT_TRUE(cube1.compatible(cube2));

  cube1 = std::move(cube2);
  EXPECT_EQ(uint2, cube1.cast<uint64_t>());
  EXPECT_TRUE(cube2.empty());
  EXPECT_FALSE(cube1.compatible(cube2));

  EXPECT_EQ(uint3, cube3.cast<uint64_t>());
  cube3 = cube1;
  EXPECT_EQ(uint2, cube3.cast<uint64_t>());
  EXPECT_EQ(uint2, cube1.cast<uint64_t>());

  // Test Conuintuctor Cube(T &&rhs) / Cube(const T &rhs)
  Cube cube41(uint64_t(444444));
  Cube cube42(uint4);
  EXPECT_NE(uint64_t(0), uint4);
  Cube cube43(std::move(uint4));
  EXPECT_NE(uint64_t(0), uint4);

  const uint64_t uint41 = uint4;
  Cube cube44(uint41);
  EXPECT_EQ(uint41, uint4);
  EXPECT_EQ(uint4, cube44.cast<uint64_t>());

  // Test Assignment operator=(T &&rhs) / operator=(const T &rhs)
  Cube cube51, cube52, cube53, cube54;
  cube51 = uint64_t(55555);
  cube52 = uint5;
  EXPECT_NE(uint64_t(0), uint5);
  cube53 = std::move(uint5);
  EXPECT_NE(uint64_t(0), uint5);

  const uint64_t uint51 = uint5;
  cube54 = uint51;
  EXPECT_EQ(uint51, uint5);
  EXPECT_EQ(uint5, cube54.cast<uint64_t>());

  // Test Conuintuctor Cube(Cube &&rhs) / Cube(const Cube &rhs)
  Cube cube6(uint6);
  Cube cube61(cube6);
  EXPECT_EQ(uint6, cube61.cast<uint64_t>());
  EXPECT_FALSE(cube6.empty());
  Cube cube62(std::move(cube6));
  EXPECT_EQ(uint6, cube62.cast<uint64_t>());
  EXPECT_TRUE(cube6.empty());

  const Cube cube63 = cube62;
  Cube cube64(cube63);
  EXPECT_EQ(uint6, cube64.cast<uint64_t>());
  EXPECT_FALSE(cube63.empty());

  // Test Assignment operator=(Cube &&rhs) / operator=(const Cube &rhs)
  Cube cube7(uint7), cube71, cube72;
  cube71 = cube7;
  EXPECT_EQ(uint7, cube71.cast<uint64_t>());
  EXPECT_FALSE(cube7.empty());
  cube72 = std::move(cube7);
  EXPECT_EQ(uint7, cube72.cast<uint64_t>());
  EXPECT_TRUE(cube7.empty());

  const Cube cube73(cube72);
  Cube cube74;
  cube74 = cube73;
  EXPECT_EQ(uint7, cube74.cast<uint64_t>());
  EXPECT_EQ(uint7, cube73.cast<uint64_t>());
  EXPECT_FALSE(cube74.empty());
}

enum EnumValueType { Unknown, Binary, Float, Double };
enum class EnumClassType { Unknown, RED, GREEN, BLUE };

TEST(Cube, EnumObject) {
  std::cout << "* uint32_t: " << typeid(uint32_t).name() << std::endl;
  std::cout << "* int32_t: " << typeid(int32_t).name() << std::endl;
  std::cout << "* EnumValueType: " << typeid(EnumValueType).name() << std::endl;
  std::cout << "* EnumValueType (underlying_type): "
            << typeid(typename std::underlying_type<EnumValueType>::type).name()
            << std::endl;

  std::cout << "* EnumClassType: " << typeid(EnumClassType).name() << std::endl;
  std::cout << "* EnumClassType (underlying_type): "
            << typeid(typename std::underlying_type<EnumClassType>::type).name()
            << std::endl;

  EnumValueType a(EnumValueType::Binary), c(EnumValueType::Unknown);
  EnumClassType b(EnumClassType::RED), d(EnumClassType::Unknown);

  Cube cubeA(a);
  Cube cubeB(b);

  EXPECT_EQ(a, cubeA.cast<EnumValueType>());
  EXPECT_NE(c, cubeA.cast<EnumValueType>());
  c = cubeA.cast<EnumValueType>();
  EXPECT_EQ(a, c);

  EXPECT_EQ(b, cubeB.cast<EnumClassType>());
  EXPECT_NE(d, cubeB.cast<EnumClassType>());
  d = cubeB.cast<EnumClassType>();
  EXPECT_EQ(b, d);

  Cube cubeC((std::underlying_type<EnumValueType>::type)1);
  Cube cubeD((std::underlying_type<EnumClassType>::type)1);

  std::cout << "* cubeA: " << cubeA.type().name() << std::endl;
  std::cout << "* cubeB: " << cubeB.type().name() << std::endl;
  std::cout << "* cubeC: " << cubeC.type().name() << std::endl;
  std::cout << "* cubeD: " << cubeD.type().name() << std::endl;

  // EXPECT_TRUE(typeid(std::underlying_type<EnumValueType>::type) ==
  //             typeid(uint32_t));
  // EXPECT_TRUE(typeid(std::underlying_type<EnumClassType>::type) ==
  //             typeid(int32_t));

  EXPECT_TRUE(cubeA.compatible<EnumValueType>());
  EXPECT_TRUE(cubeB.compatible<EnumClassType>());
  EXPECT_TRUE(cubeA.compatible<std::underlying_type<EnumValueType>::type>());
  EXPECT_TRUE(cubeB.compatible<std::underlying_type<EnumClassType>::type>());
  EXPECT_TRUE(cubeC.compatible<std::underlying_type<EnumValueType>::type>());
  EXPECT_TRUE(cubeD.compatible<std::underlying_type<EnumClassType>::type>());

  EnumValueType e =
      (EnumValueType)cubeA.cast<std::underlying_type<EnumValueType>::type>();
  EnumClassType f =
      (EnumClassType)cubeB.cast<std::underlying_type<EnumClassType>::type>();
  EXPECT_EQ(a, e);
  EXPECT_EQ(b, f);
}
