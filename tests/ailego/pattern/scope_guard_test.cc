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

#include <ailego/pattern/defer.h>
#include <gtest/gtest.h>

using namespace zvec;

TEST(ScopeGuard, Lambda) {
  int count = 0;
  auto a = ailego::ScopeGuard::Make(
      [](int val) { printf("ScopeGuard: Lambda %d\n", val); }, 1);

  auto b = ailego::ScopeGuard::Make([&] {
    printf("ScopeGuard: Lambda 2\n");
    ++count;
  });

  auto c = ailego::ScopeGuard::Make([] {
    printf("ScopeGuard: Lambda 3\n");
    return 0;
  });

  auto d = ailego::ScopeGuard::Make([&] {
    printf("ScopeGuard: Lambda 4\n");
    ++count;
    return false;
  });

  EXPECT_EQ(0, count);
}

struct ClassA {
  static void StaticProcess0() {
    printf("ScopeGuard: Static Function 1\n");
    ++count;
  }

  static int StaticProcess1(int val) {
    printf("ScopeGuard: Static Function %d\n", val);
    ++count;
    return 0;
  }

  static int count;
};

int ClassA::count{0};

TEST(ScopeGuard, StaticFunction) {
  auto a = ailego::ScopeGuard::Make(ClassA::StaticProcess0);
  auto b = ailego::ScopeGuard::Make(ClassA::StaticProcess1, 2);

  EXPECT_EQ(0, ClassA::count);
}

class ClassB {
 public:
  virtual void MemberProcess0() const {
    printf("ScopeGuard: Member Function 0\n");
    ++count;
  }

  virtual void MemberProcess1(int val) {
    printf("ScopeGuard: Member Function %d\n", val);
    ++count;
  }

  virtual void MemberProcess2(long val) const volatile {
    printf("ScopeGuard: Member Function %ld\n", val);
    ++count;
  }

  virtual void MemberProcess3(size_t val) volatile {
    printf("ScopeGuard: Member Function %zu\n", val);
    ++count;
  }

  static int count;
};

int ClassB::count{0};

TEST(ScopeGuard, MemberFunction) {
  ClassB bb;
  auto a = ailego::ScopeGuard::Make(&bb, &ClassB::MemberProcess0);
  auto b = ailego::ScopeGuard::Make(&bb, &ClassB::MemberProcess1, 2);
  AILEGO_DEFER(&bb, &ClassB::MemberProcess2, 3);
  AILEGO_DEFER(&bb, &ClassB::MemberProcess3, 4);
  EXPECT_EQ(0, ClassB::count);
}
