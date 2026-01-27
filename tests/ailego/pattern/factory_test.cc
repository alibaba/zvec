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
#include <zvec/ailego/pattern/factory.h>

using namespace zvec;
using namespace zvec::ailego;

struct Base {
  virtual ~Base(void) {}
  virtual void do_something() = 0;
};

struct AAA : public Base {
  AAA(void) {}

  virtual void do_something() {
    printf("do something\n");
  }
};

AILEGO_FACTORY_REGISTER(AAA, Base, AAA);

TEST(Factory, General) {
  EXPECT_TRUE(!ailego::Factory<Base>::MakeShared("BBB"));
  EXPECT_TRUE(!ailego::Factory<Base>::Has("BBB"));

  auto aaa = ailego::Factory<Base>::MakeShared("AAA");
  ASSERT_TRUE(!!aaa);
  aaa->do_something();
  EXPECT_TRUE(!!ailego::Factory<Base>::Has("AAA"));

  auto vec = ailego::Factory<Base>::Classes();
  EXPECT_EQ(1u, vec.size());
  EXPECT_EQ("AAA", std::string(vec[0]));
}
