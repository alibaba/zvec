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

#include <functional>
#include <iostream>
#include <gtest/gtest.h>
#include <zvec/ailego/pattern/closure.h>
#include <zvec/ailego/utility/time_helper.h>

using namespace zvec;

void GlobalProcess0(void) {}
void GlobalProcess1(int) {}

void GlobalProcess2(int a1, int *a2) {
  EXPECT_EQ(a1 + 1, *a2);
}

void GlobalProcess3(int a1, int *a2, int &a3) {
  EXPECT_EQ(a1 + 1, *a2);
  EXPECT_EQ(*a2 + 1, a3);
}

void GlobalProcess4(int a1, int *a2, int &a3, const int &a4) {
  EXPECT_EQ(a1 + 1, *a2);
  EXPECT_EQ(*a2 + 1, a3);
  EXPECT_EQ(a3 + 1, a4);
}

void GlobalProcess5(int a1, int *a2, int &a3, const int &a4, volatile int *a5) {
  EXPECT_EQ(a1 + 1, *a2);
  EXPECT_EQ(*a2 + 1, a3);
  EXPECT_EQ(a3 + 1, a4);
  EXPECT_EQ(a4 + 1, *a5);
}

void GlobalProcess6(int a1, int *a2, int &a3, const int &a4, volatile int *a5,
                    int *const volatile a6) {
  EXPECT_EQ(a1 + 1, *a2);
  EXPECT_EQ(*a2 + 1, a3);
  EXPECT_EQ(a3 + 1, a4);
  EXPECT_EQ(a4 + 1, *a5);
  EXPECT_EQ(*a5 + 1, *a6);
}

void GlobalProcess7(int a1, int *a2, int &a3, const int &a4, volatile int *a5,
                    int *const volatile a6, int &&a7) {
  EXPECT_EQ(a1 + 1, *a2);
  EXPECT_EQ(*a2 + 1, a3);
  EXPECT_EQ(a3 + 1, a4);
  EXPECT_EQ(a4 + 1, *a5);
  EXPECT_EQ(*a5 + 1, *a6);
  EXPECT_EQ(*a6 + 1, a7);
}

size_t GlobalFunction0(void) {
  return 0;
}
size_t GlobalFunction1(long) {
  return 1;
}

size_t GlobalFunction2(long a1, long *a2) {
  EXPECT_EQ(a1 + 1, *a2);
  return 2;
}

size_t GlobalFunction3(long a1, long *a2, long &a3) {
  EXPECT_EQ(a1 + 1, *a2);
  EXPECT_EQ(*a2 + 1, a3);
  return 3;
}

size_t GlobalFunction4(long a1, long *a2, long &a3, const long &a4) {
  EXPECT_EQ(a1 + 1, *a2);
  EXPECT_EQ(*a2 + 1, a3);
  EXPECT_EQ(a3 + 1, a4);
  return 4;
}

size_t GlobalFunction5(long a1, long *a2, long &a3, const long &a4,
                       volatile long *a5) {
  EXPECT_EQ(a1 + 1, *a2);
  EXPECT_EQ(*a2 + 1, a3);
  EXPECT_EQ(a3 + 1, a4);
  EXPECT_EQ(a4 + 1, *a5);
  return 5;
}

size_t GlobalFunction6(long a1, long *a2, long &a3, const long &a4,
                       volatile long *a5, long *const volatile a6) {
  EXPECT_EQ(a1 + 1, *a2);
  EXPECT_EQ(*a2 + 1, a3);
  EXPECT_EQ(a3 + 1, a4);
  EXPECT_EQ(a4 + 1, *a5);
  EXPECT_EQ(*a5 + 1, *a6);
  return 6;
}

size_t GlobalFunction7(long a1, long *a2, long &a3, const long &a4,
                       volatile long *a5, long *const volatile a6, long &&a7) {
  EXPECT_EQ(a1 + 1, *a2);
  EXPECT_EQ(*a2 + 1, a3);
  EXPECT_EQ(a3 + 1, a4);
  EXPECT_EQ(a4 + 1, *a5);
  EXPECT_EQ(*a5 + 1, *a6);
  EXPECT_EQ(*a6 + 1, a7);
  return 7;
}

struct WithFunctionCall {
  int operator()(int a) {
    return a + b;
  }
  int do_something(int a) {
    return a + b;
  }
  int b = 11;
};

struct WithoutFunctionCall {
  int do_something(int a) {
    return a + b;
  }
  int b = 11;
};

struct ClassA {
  static void StaticProcess0(void) {}
  static void StaticProcess1(int) {}

  static void StaticProcess2(int a1, int *a2) {
    EXPECT_EQ(a1 + 1, *a2);
  }

  static void StaticProcess3(int a1, int *a2, int &a3) {
    EXPECT_EQ(a1 + 1, *a2);
    EXPECT_EQ(*a2 + 1, a3);
  }

  static void StaticProcess4(int a1, int *a2, int &a3, const int &a4) {
    EXPECT_EQ(a1 + 1, *a2);
    EXPECT_EQ(*a2 + 1, a3);
    EXPECT_EQ(a3 + 1, a4);
  }

  static void StaticProcess5(int a1, int *a2, int &a3, const int &a4,
                             volatile int *a5) {
    EXPECT_EQ(a1 + 1, *a2);
    EXPECT_EQ(*a2 + 1, a3);
    EXPECT_EQ(a3 + 1, a4);
    EXPECT_EQ(a4 + 1, *a5);
  }

  static void StaticProcess6(int a1, int *a2, int &a3, const int &a4,
                             volatile int *a5, int *const volatile a6) {
    EXPECT_EQ(a1 + 1, *a2);
    EXPECT_EQ(*a2 + 1, a3);
    EXPECT_EQ(a3 + 1, a4);
    EXPECT_EQ(a4 + 1, *a5);
    EXPECT_EQ(*a5 + 1, *a6);
  }

  static void StaticProcess7(int a1, int *a2, int &a3, const int &a4,
                             volatile int *a5, int *const volatile a6,
                             int &&a7) {
    EXPECT_EQ(a1 + 1, *a2);
    EXPECT_EQ(*a2 + 1, a3);
    EXPECT_EQ(a3 + 1, a4);
    EXPECT_EQ(a4 + 1, *a5);
    EXPECT_EQ(*a5 + 1, *a6);
    EXPECT_EQ(*a6 + 1, a7);
  }

  static size_t StaticFunction0(void) {
    return 0;
  }
  static size_t StaticFunction1(long) {
    return 1;
  }

  static size_t StaticFunction2(long a1, long *a2) {
    EXPECT_EQ(a1 + 1, *a2);
    return 2;
  }

  static size_t StaticFunction3(long a1, long *a2, long &a3) {
    EXPECT_EQ(a1 + 1, *a2);
    EXPECT_EQ(*a2 + 1, a3);
    return 3;
  }

  static size_t StaticFunction4(long a1, long *a2, long &a3, const long &a4) {
    EXPECT_EQ(a1 + 1, *a2);
    EXPECT_EQ(*a2 + 1, a3);
    EXPECT_EQ(a3 + 1, a4);
    return 4;
  }

  static size_t StaticFunction5(long a1, long *a2, long &a3, const long &a4,
                                volatile long *a5) {
    EXPECT_EQ(a1 + 1, *a2);
    EXPECT_EQ(*a2 + 1, a3);
    EXPECT_EQ(a3 + 1, a4);
    EXPECT_EQ(a4 + 1, *a5);
    return 5;
  }

  static size_t StaticFunction6(long a1, long *a2, long &a3, const long &a4,
                                volatile long *a5, long *const volatile a6) {
    EXPECT_EQ(a1 + 1, *a2);
    EXPECT_EQ(*a2 + 1, a3);
    EXPECT_EQ(a3 + 1, a4);
    EXPECT_EQ(a4 + 1, *a5);
    EXPECT_EQ(*a5 + 1, *a6);
    return 6;
  }

  static size_t StaticFunction7(long a1, long *a2, long &a3, const long &a4,
                                volatile long *a5, long *const volatile a6,
                                long &&a7) {
    EXPECT_EQ(a1 + 1, *a2);
    EXPECT_EQ(*a2 + 1, a3);
    EXPECT_EQ(a3 + 1, a4);
    EXPECT_EQ(a4 + 1, *a5);
    EXPECT_EQ(*a5 + 1, *a6);
    EXPECT_EQ(*a6 + 1, a7);
    return 7;
  }
};

class ClassB {
 public:
  ClassB(int v) : b_(v) {}

  int operator()(int a1) {
    EXPECT_TRUE(0);
    return a1 + b_;
  }

  virtual void MemberProcess0(void) const {}

  virtual void MemberProcess1(int a1) {
    EXPECT_EQ(a1, b_);
  }

  void MemberProcess2(int a1, int *a2) {
    EXPECT_EQ(a1, b_);
    EXPECT_EQ(a1 + 1, *a2);
  }

  void MemberProcess3(int a1, int *a2, int &a3) const {
    EXPECT_EQ(a1, b_);
    EXPECT_EQ(a1 + 1, *a2);
    EXPECT_EQ(*a2 + 1, a3);
  }

  virtual void MemberProcess4(int a1, int *a2, int &a3, const int &a4) {
    EXPECT_EQ(a1, b_);
    EXPECT_EQ(a1 + 1, *a2);
    EXPECT_EQ(*a2 + 1, a3);
    EXPECT_EQ(a3 + 1, a4);
  }

  virtual void MemberProcess5(int a1, int *a2, int &a3, const int &a4,
                              volatile int *a5) const {
    EXPECT_EQ(a1, b_);
    EXPECT_EQ(a1 + 1, *a2);
    EXPECT_EQ(*a2 + 1, a3);
    EXPECT_EQ(a3 + 1, a4);
    EXPECT_EQ(a4 + 1, *a5);
  }

  void MemberProcess6(int a1, int *a2, int &a3, const int &a4, volatile int *a5,
                      int *const volatile a6) {
    EXPECT_EQ(a1, b_);
    EXPECT_EQ(a1 + 1, *a2);
    EXPECT_EQ(*a2 + 1, a3);
    EXPECT_EQ(a3 + 1, a4);
    EXPECT_EQ(a4 + 1, *a5);
    EXPECT_EQ(*a5 + 1, *a6);
  }

  void MemberProcess7(int a1, int *a2, int &a3, const int &a4, volatile int *a5,
                      int *const volatile a6, int &&a7) {
    EXPECT_EQ(a1, b_);
    EXPECT_EQ(a1 + 1, *a2);
    EXPECT_EQ(*a2 + 1, a3);
    EXPECT_EQ(a3 + 1, a4);
    EXPECT_EQ(a4 + 1, *a5);
    EXPECT_EQ(*a5 + 1, *a6);
    EXPECT_EQ(*a6 + 1, a7);
  }

  size_t MemberFunction0(void) {
    return 0;
  }
  size_t MemberFunction1(long a1) {
    EXPECT_EQ(a1, b_);
    return 1;
  }

  size_t MemberFunction2(long a1, long *a2) {
    EXPECT_EQ(a1, b_);
    EXPECT_EQ(a1 + 1, *a2);
    return 2;
  }

  size_t MemberFunction3(long a1, long *a2, long &a3) volatile {
    EXPECT_EQ(a1, b_);
    EXPECT_EQ(a1 + 1, *a2);
    EXPECT_EQ(*a2 + 1, a3);
    return 3;
  }

  size_t MemberFunction4(long a1, long *a2, long &a3, const long &a4) const {
    EXPECT_EQ(a1, b_);
    EXPECT_EQ(a1 + 1, *a2);
    EXPECT_EQ(*a2 + 1, a3);
    EXPECT_EQ(a3 + 1, a4);
    return 4;
  }

  size_t MemberFunction5(long a1, long *a2, long &a3, const long &a4,
                         volatile long *a5) const volatile {
    EXPECT_EQ(a1, b_);
    EXPECT_EQ(a1 + 1, *a2);
    EXPECT_EQ(*a2 + 1, a3);
    EXPECT_EQ(a3 + 1, a4);
    EXPECT_EQ(a4 + 1, *a5);
    return 5;
  }

  size_t MemberFunction6(long a1, long *a2, long &a3, const long &a4,
                         volatile long *a5, long *const volatile a6) const {
    EXPECT_EQ(a1, b_);
    EXPECT_EQ(a1 + 1, *a2);
    EXPECT_EQ(*a2 + 1, a3);
    EXPECT_EQ(a3 + 1, a4);
    EXPECT_EQ(a4 + 1, *a5);
    EXPECT_EQ(*a5 + 1, *a6);
    return 6;
  }

  size_t MemberFunction7(long a1, long *a2, long &a3, const long &a4,
                         volatile long *a5, long *const volatile a6,
                         long &&a7) const volatile {
    EXPECT_EQ(a1, b_);
    EXPECT_EQ(a1 + 1, *a2);
    EXPECT_EQ(*a2 + 1, a3);
    EXPECT_EQ(a3 + 1, a4);
    EXPECT_EQ(a4 + 1, *a5);
    EXPECT_EQ(*a5 + 1, *a6);
    EXPECT_EQ(*a6 + 1, a7);
    return 7;
  }

 private:
  int b_{11};
};

class ClassAB {
 public:
  void Run1(void) const {
    ClassB bbb(1);
    ailego::Closure::New(this, &ClassAB::ConstFunc, &bbb);
  }

  void Run2(void) {
    ClassB bbb(1);
    ailego::Closure::New(this, &ClassAB::ConstFunc, &bbb);
  }

  void Run3(void) {
    ClassB bbb(1);
    ailego::Closure::New(this, &ClassAB::MutableFunc, &bbb);
  }

  void Run4(void) const {
    ClassB bbb(1);
    ailego::Closure::New(this, &ClassAB::VolatileConstFunc, &bbb);
  }

  void Run5(void) {
    ClassB bbb(1);
    ailego::Closure::New(this, &ClassAB::VolatileMutableFunc, &bbb);
  }

  void Run6(void) const volatile {
    ClassB bbb(1);
    ailego::Closure::New(this, &ClassAB::VolatileConstFunc, &bbb);
  }

  void Run7(void) volatile {
    ClassB bbb(1);
    ailego::Closure::New(this, &ClassAB::VolatileConstFunc, &bbb);
  }

  void Run8(void) volatile {
    ClassB bbb(1);
    ailego::Closure::New(this, &ClassAB::VolatileMutableFunc, &bbb);
  }

 protected:
  void ConstFunc(const ClassB *b) const {
    ClassA::StaticFunction0();
    b->MemberProcess0();
  }

  void MutableFunc(const ClassB *b) {
    ClassA::StaticFunction0();
    b->MemberProcess0();
  }

  void VolatileConstFunc(const ClassB *b) const volatile {
    ClassA::StaticFunction0();
    b->MemberProcess0();
  }

  void VolatileMutableFunc(const ClassB *b) volatile {
    ClassA::StaticFunction0();
    b->MemberProcess0();
  }
};

TEST(CallbackValidator, General) {
  EXPECT_FALSE(ailego::CallbackValidator<int>::Value);
  EXPECT_FALSE(ailego::CallbackValidator<long *>::Value);
  EXPECT_FALSE(ailego::CallbackValidator<const void *>::Value);

  EXPECT_TRUE(ailego::CallbackValidator<decltype(GlobalFunction0)>::Value);
  EXPECT_TRUE(ailego::CallbackValidator<decltype(&GlobalFunction0)>::Value);
  EXPECT_TRUE(ailego::CallbackValidator<decltype(GlobalFunction1)>::Value);
  EXPECT_TRUE(ailego::CallbackValidator<decltype(&GlobalFunction1)>::Value);
  EXPECT_TRUE(ailego::CallbackValidator<decltype(GlobalFunction2)>::Value);
  EXPECT_TRUE(ailego::CallbackValidator<decltype(&GlobalFunction2)>::Value);
  EXPECT_TRUE(ailego::CallbackValidator<decltype(GlobalFunction3)>::Value);
  EXPECT_TRUE(ailego::CallbackValidator<decltype(&GlobalFunction3)>::Value);
  EXPECT_TRUE(ailego::CallbackValidator<decltype(GlobalFunction4)>::Value);
  EXPECT_TRUE(ailego::CallbackValidator<decltype(&GlobalFunction4)>::Value);
  EXPECT_TRUE(ailego::CallbackValidator<decltype(GlobalFunction5)>::Value);
  EXPECT_TRUE(ailego::CallbackValidator<decltype(&GlobalFunction5)>::Value);
  EXPECT_TRUE(ailego::CallbackValidator<decltype(GlobalFunction6)>::Value);
  EXPECT_TRUE(ailego::CallbackValidator<decltype(&GlobalFunction6)>::Value);
  EXPECT_TRUE(ailego::CallbackValidator<decltype(GlobalFunction7)>::Value);
  EXPECT_TRUE(ailego::CallbackValidator<decltype(&GlobalFunction7)>::Value);

  EXPECT_TRUE(ailego::CallbackValidator<decltype(GlobalProcess0)>::Value);
  EXPECT_TRUE(ailego::CallbackValidator<decltype(&GlobalProcess0)>::Value);
  EXPECT_TRUE(ailego::CallbackValidator<decltype(GlobalProcess1)>::Value);
  EXPECT_TRUE(ailego::CallbackValidator<decltype(&GlobalProcess1)>::Value);
  EXPECT_TRUE(ailego::CallbackValidator<decltype(GlobalProcess2)>::Value);
  EXPECT_TRUE(ailego::CallbackValidator<decltype(&GlobalProcess2)>::Value);
  EXPECT_TRUE(ailego::CallbackValidator<decltype(GlobalProcess3)>::Value);
  EXPECT_TRUE(ailego::CallbackValidator<decltype(&GlobalProcess3)>::Value);
  EXPECT_TRUE(ailego::CallbackValidator<decltype(GlobalProcess4)>::Value);
  EXPECT_TRUE(ailego::CallbackValidator<decltype(&GlobalProcess4)>::Value);
  EXPECT_TRUE(ailego::CallbackValidator<decltype(GlobalProcess5)>::Value);
  EXPECT_TRUE(ailego::CallbackValidator<decltype(&GlobalProcess5)>::Value);
  EXPECT_TRUE(ailego::CallbackValidator<decltype(GlobalProcess6)>::Value);
  EXPECT_TRUE(ailego::CallbackValidator<decltype(&GlobalProcess6)>::Value);
  EXPECT_TRUE(ailego::CallbackValidator<decltype(GlobalProcess7)>::Value);
  EXPECT_TRUE(ailego::CallbackValidator<decltype(&GlobalProcess7)>::Value);

  EXPECT_TRUE(ailego::CallbackValidator<
              std::function<decltype(GlobalFunction0)>>::Value);
  EXPECT_TRUE(ailego::CallbackValidator<
              std::function<decltype(GlobalFunction1)>>::Value);
  EXPECT_TRUE(ailego::CallbackValidator<
              std::function<decltype(GlobalFunction2)>>::Value);
  EXPECT_TRUE(ailego::CallbackValidator<
              std::function<decltype(GlobalFunction3)>>::Value);
  EXPECT_TRUE(ailego::CallbackValidator<
              std::function<decltype(GlobalFunction4)>>::Value);
  EXPECT_TRUE(ailego::CallbackValidator<
              std::function<decltype(GlobalFunction5)>>::Value);
  EXPECT_TRUE(ailego::CallbackValidator<
              std::function<decltype(GlobalFunction6)>>::Value);
  EXPECT_TRUE(ailego::CallbackValidator<
              std::function<decltype(GlobalFunction7)>>::Value);

  EXPECT_TRUE(ailego::CallbackValidator<WithFunctionCall>::Value);
  EXPECT_TRUE(ailego::CallbackValidator<WithFunctionCall &>::Value);
  EXPECT_TRUE(ailego::CallbackValidator<const WithFunctionCall &>::Value);
  EXPECT_FALSE(ailego::CallbackValidator<WithFunctionCall *>::Value);
  EXPECT_FALSE(ailego::CallbackValidator<const WithFunctionCall *>::Value);
  EXPECT_FALSE(ailego::CallbackValidator<WithoutFunctionCall>::Value);
  EXPECT_FALSE(ailego::CallbackValidator<WithoutFunctionCall &>::Value);
  EXPECT_FALSE(ailego::CallbackValidator<const WithoutFunctionCall &>::Value);
  EXPECT_FALSE(ailego::CallbackValidator<WithoutFunctionCall *>::Value);
  EXPECT_FALSE(ailego::CallbackValidator<const WithoutFunctionCall *>::Value);

  EXPECT_TRUE(
      ailego::CallbackValidator<decltype(ClassA::StaticFunction0)>::Value);
  EXPECT_TRUE(
      ailego::CallbackValidator<decltype(&ClassA::StaticFunction0)>::Value);
  EXPECT_TRUE(
      ailego::CallbackValidator<decltype(ClassA::StaticFunction1)>::Value);
  EXPECT_TRUE(
      ailego::CallbackValidator<decltype(&ClassA::StaticFunction1)>::Value);
  EXPECT_TRUE(
      ailego::CallbackValidator<decltype(ClassA::StaticFunction2)>::Value);
  EXPECT_TRUE(
      ailego::CallbackValidator<decltype(&ClassA::StaticFunction2)>::Value);
  EXPECT_TRUE(
      ailego::CallbackValidator<decltype(ClassA::StaticFunction3)>::Value);
  EXPECT_TRUE(
      ailego::CallbackValidator<decltype(&ClassA::StaticFunction3)>::Value);
  EXPECT_TRUE(
      ailego::CallbackValidator<decltype(ClassA::StaticFunction4)>::Value);
  EXPECT_TRUE(
      ailego::CallbackValidator<decltype(&ClassA::StaticFunction4)>::Value);
  EXPECT_TRUE(
      ailego::CallbackValidator<decltype(ClassA::StaticFunction5)>::Value);
  EXPECT_TRUE(
      ailego::CallbackValidator<decltype(&ClassA::StaticFunction5)>::Value);
  EXPECT_TRUE(
      ailego::CallbackValidator<decltype(ClassA::StaticFunction6)>::Value);
  EXPECT_TRUE(
      ailego::CallbackValidator<decltype(&ClassA::StaticFunction6)>::Value);
  EXPECT_TRUE(
      ailego::CallbackValidator<decltype(ClassA::StaticFunction7)>::Value);
  EXPECT_TRUE(
      ailego::CallbackValidator<decltype(&ClassA::StaticFunction7)>::Value);

  EXPECT_TRUE(
      ailego::CallbackValidator<decltype(ClassA::StaticProcess0)>::Value);
  EXPECT_TRUE(
      ailego::CallbackValidator<decltype(&ClassA::StaticProcess0)>::Value);
  EXPECT_TRUE(
      ailego::CallbackValidator<decltype(ClassA::StaticProcess1)>::Value);
  EXPECT_TRUE(
      ailego::CallbackValidator<decltype(&ClassA::StaticProcess1)>::Value);
  EXPECT_TRUE(
      ailego::CallbackValidator<decltype(ClassA::StaticProcess2)>::Value);
  EXPECT_TRUE(
      ailego::CallbackValidator<decltype(&ClassA::StaticProcess2)>::Value);
  EXPECT_TRUE(
      ailego::CallbackValidator<decltype(ClassA::StaticProcess3)>::Value);
  EXPECT_TRUE(
      ailego::CallbackValidator<decltype(&ClassA::StaticProcess3)>::Value);
  EXPECT_TRUE(
      ailego::CallbackValidator<decltype(ClassA::StaticProcess4)>::Value);
  EXPECT_TRUE(
      ailego::CallbackValidator<decltype(&ClassA::StaticProcess4)>::Value);
  EXPECT_TRUE(
      ailego::CallbackValidator<decltype(ClassA::StaticProcess5)>::Value);
  EXPECT_TRUE(
      ailego::CallbackValidator<decltype(&ClassA::StaticProcess5)>::Value);
  EXPECT_TRUE(
      ailego::CallbackValidator<decltype(ClassA::StaticProcess6)>::Value);
  EXPECT_TRUE(
      ailego::CallbackValidator<decltype(&ClassA::StaticProcess6)>::Value);
  EXPECT_TRUE(
      ailego::CallbackValidator<decltype(ClassA::StaticProcess7)>::Value);
  EXPECT_TRUE(
      ailego::CallbackValidator<decltype(&ClassA::StaticProcess7)>::Value);

  EXPECT_TRUE(
      ailego::CallbackValidator<decltype(&ClassB::MemberFunction0)>::Value);
  EXPECT_TRUE(
      ailego::CallbackValidator<decltype(&ClassB::MemberFunction1)>::Value);
  EXPECT_TRUE(
      ailego::CallbackValidator<decltype(&ClassB::MemberFunction2)>::Value);
  EXPECT_TRUE(
      ailego::CallbackValidator<decltype(&ClassB::MemberFunction3)>::Value);
  EXPECT_TRUE(
      ailego::CallbackValidator<decltype(&ClassB::MemberFunction4)>::Value);
  EXPECT_TRUE(
      ailego::CallbackValidator<decltype(&ClassB::MemberFunction5)>::Value);
  EXPECT_TRUE(
      ailego::CallbackValidator<decltype(&ClassB::MemberFunction6)>::Value);
  EXPECT_TRUE(
      ailego::CallbackValidator<decltype(&ClassB::MemberFunction7)>::Value);
  EXPECT_TRUE(
      ailego::CallbackValidator<decltype(&ClassB::MemberProcess0)>::Value);
  EXPECT_TRUE(
      ailego::CallbackValidator<decltype(&ClassB::MemberProcess1)>::Value);
  EXPECT_TRUE(
      ailego::CallbackValidator<decltype(&ClassB::MemberProcess2)>::Value);
  EXPECT_TRUE(
      ailego::CallbackValidator<decltype(&ClassB::MemberProcess3)>::Value);
  EXPECT_TRUE(
      ailego::CallbackValidator<decltype(&ClassB::MemberProcess4)>::Value);
  EXPECT_TRUE(
      ailego::CallbackValidator<decltype(&ClassB::MemberProcess5)>::Value);
  EXPECT_TRUE(
      ailego::CallbackValidator<decltype(&ClassB::MemberProcess6)>::Value);
  EXPECT_TRUE(
      ailego::CallbackValidator<decltype(&ClassB::MemberProcess7)>::Value);
}

TEST(CallbackTraits, General) {
  EXPECT_EQ(0, ailego::CallbackTraits<decltype(GlobalProcess0)>::Arity);
  EXPECT_EQ(1, ailego::CallbackTraits<decltype(GlobalProcess1)>::Arity);
  EXPECT_EQ(2, ailego::CallbackTraits<decltype(GlobalProcess2)>::Arity);
  EXPECT_EQ(3, ailego::CallbackTraits<decltype(GlobalProcess3)>::Arity);
  EXPECT_EQ(4, ailego::CallbackTraits<decltype(GlobalProcess4)>::Arity);
  EXPECT_EQ(5, ailego::CallbackTraits<decltype(GlobalProcess5)>::Arity);
  EXPECT_EQ(6, ailego::CallbackTraits<decltype(GlobalProcess6)>::Arity);
  EXPECT_EQ(7, ailego::CallbackTraits<decltype(GlobalProcess7)>::Arity);

  EXPECT_EQ(0,
            ailego::CallbackTraits<decltype(&ClassA::StaticProcess0)>::Arity);
  EXPECT_EQ(1,
            ailego::CallbackTraits<decltype(&ClassA::StaticProcess1)>::Arity);
  EXPECT_EQ(2,
            ailego::CallbackTraits<decltype(&ClassA::StaticProcess2)>::Arity);
  EXPECT_EQ(3,
            ailego::CallbackTraits<decltype(&ClassA::StaticProcess3)>::Arity);
  EXPECT_EQ(4,
            ailego::CallbackTraits<decltype(&ClassA::StaticProcess4)>::Arity);
  EXPECT_EQ(5,
            ailego::CallbackTraits<decltype(&ClassA::StaticProcess5)>::Arity);
  EXPECT_EQ(6,
            ailego::CallbackTraits<decltype(&ClassA::StaticProcess6)>::Arity);
  EXPECT_EQ(7,
            ailego::CallbackTraits<decltype(&ClassA::StaticProcess7)>::Arity);

  EXPECT_EQ(0,
            ailego::CallbackTraits<decltype(&ClassB::MemberProcess0)>::Arity);
  EXPECT_EQ(1,
            ailego::CallbackTraits<decltype(&ClassB::MemberProcess1)>::Arity);
  EXPECT_EQ(2,
            ailego::CallbackTraits<decltype(&ClassB::MemberProcess2)>::Arity);
  EXPECT_EQ(3,
            ailego::CallbackTraits<decltype(&ClassB::MemberProcess3)>::Arity);
  EXPECT_EQ(4,
            ailego::CallbackTraits<decltype(&ClassB::MemberProcess4)>::Arity);
  EXPECT_EQ(5,
            ailego::CallbackTraits<decltype(&ClassB::MemberProcess5)>::Arity);
  EXPECT_EQ(6,
            ailego::CallbackTraits<decltype(&ClassB::MemberProcess6)>::Arity);
  EXPECT_EQ(7,
            ailego::CallbackTraits<decltype(&ClassB::MemberProcess7)>::Arity);

  EXPECT_EQ(
      1u, sizeof(ailego::CallbackTraits<decltype(GlobalProcess0)>::TupleType));
  EXPECT_EQ(1u, sizeof(ailego::CallbackTraits<
                       decltype(&ClassA::StaticProcess0)>::TupleType));
  EXPECT_EQ(1u, sizeof(ailego::CallbackTraits<
                       decltype(&ClassB::MemberProcess0)>::TupleType));
}

TEST(Closure, Static) {
  long a[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  int b[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  ailego::Closure::New(GlobalFunction0)->run();
  ailego::Closure::New(&GlobalFunction0)->run();
  ailego::Closure::New(GlobalFunction1, a[1])->run();
  ailego::Closure::New(&GlobalFunction1, 1)->run();
  ailego::Closure::New(GlobalFunction2, 1, &a[2])->run();
  ailego::Closure::New(&GlobalFunction2, a[1], &a[2])->run();
  ailego::Closure::New(GlobalFunction3, a[1], &a[2], a[3])->run();
  ailego::Closure::New(&GlobalFunction3, 1, &a[2], a[3])->run();
  ailego::Closure::New(GlobalFunction4, 1, &a[2], a[3], 4)->run();
  ailego::Closure::New(&GlobalFunction4, a[1], &a[2], a[3], a[4])->run();
  ailego::Closure::New(GlobalFunction5, a[1], &a[2], a[3], a[4], &a[5])->run();
  ailego::Closure::New(&GlobalFunction5, 1, &a[2], a[3], 4, &a[5])->run();
  ailego::Closure::New(GlobalFunction6, 1, &a[2], a[3], 4, &a[5], &a[6])->run();
  ailego::Closure::New(&GlobalFunction6, a[1], &a[2], a[3], a[4], &a[5], &a[6])
      ->run();
  ailego::Closure::New(GlobalFunction7, 1, &a[2], a[3], 4, &a[5], &a[6], a[7])
      ->run();
  ailego::Closure::New(&GlobalFunction7, a[1], &a[2], a[3], a[4], &a[5], &a[6],
                       7)
      ->run();

  ailego::Closure::New(GlobalProcess0)->run();
  ailego::Closure::New(&GlobalProcess0)->run();
  ailego::Closure::New(GlobalProcess1, b[1])->run();
  ailego::Closure::New(&GlobalProcess1, 1)->run();
  ailego::Closure::New(GlobalProcess2, 1, &b[2])->run();
  ailego::Closure::New(&GlobalProcess2, b[1], &b[2])->run();
  ailego::Closure::New(GlobalProcess3, b[1], &b[2], b[3])->run();
  ailego::Closure::New(&GlobalProcess3, 1, &b[2], b[3])->run();
  ailego::Closure::New(GlobalProcess4, 1, &b[2], b[3], 4)->run();
  ailego::Closure::New(&GlobalProcess4, b[1], &b[2], b[3], b[4])->run();
  ailego::Closure::New(GlobalProcess5, b[1], &b[2], b[3], b[4], &b[5])->run();
  ailego::Closure::New(&GlobalProcess5, 1, &b[2], b[3], 4, &b[5])->run();
  ailego::Closure::New(GlobalProcess6, 1, &b[2], b[3], 4, &b[5], &b[6])->run();
  ailego::Closure::New(&GlobalProcess6, b[1], &b[2], b[3], b[4], &b[5], &b[6])
      ->run();
  ailego::Closure::New(GlobalProcess7, 1, &b[2], b[3], 4, &b[5], &b[6], b[7])
      ->run();
  ailego::Closure::New(&GlobalProcess7, b[1], &b[2], b[3], b[4], &b[5], &b[6],
                       7)
      ->run();

  ailego::Closure::New(ClassA::StaticFunction0)->run();
  ailego::Closure::New(&ClassA::StaticFunction0)->run();
  ailego::Closure::New(ClassA::StaticFunction1, a[1])->run();
  ailego::Closure::New(&ClassA::StaticFunction1, 1)->run();
  ailego::Closure::New(ClassA::StaticFunction2, 1, &a[2])->run();
  ailego::Closure::New(&ClassA::StaticFunction2, a[1], &a[2])->run();
  ailego::Closure::New(ClassA::StaticFunction3, a[1], &a[2], a[3])->run();
  ailego::Closure::New(&ClassA::StaticFunction3, 1, &a[2], a[3])->run();
  ailego::Closure::New(ClassA::StaticFunction4, 1, &a[2], a[3], 4)->run();
  ailego::Closure::New(&ClassA::StaticFunction4, a[1], &a[2], a[3], a[4])
      ->run();
  ailego::Closure::New(ClassA::StaticFunction5, a[1], &a[2], a[3], a[4], &a[5])
      ->run();
  ailego::Closure::New(&ClassA::StaticFunction5, 1, &a[2], a[3], 4, &a[5])
      ->run();
  ailego::Closure::New(ClassA::StaticFunction6, 1, &a[2], a[3], 4, &a[5], &a[6])
      ->run();
  ailego::Closure::New(&ClassA::StaticFunction6, a[1], &a[2], a[3], a[4], &a[5],
                       &a[6])
      ->run();
  ailego::Closure::New(ClassA::StaticFunction7, 1, &a[2], a[3], 4, &a[5], &a[6],
                       a[7])
      ->run();
  ailego::Closure::New(&ClassA::StaticFunction7, a[1], &a[2], a[3], a[4], &a[5],
                       &a[6], 7)
      ->run();

  ailego::Closure::New(ClassA::StaticProcess0)->run();
  ailego::Closure::New(&ClassA::StaticProcess0)->run();
  ailego::Closure::New(ClassA::StaticProcess1, b[1])->run();
  ailego::Closure::New(&ClassA::StaticProcess1, 1)->run();
  ailego::Closure::New(ClassA::StaticProcess2, 1, &b[2])->run();
  ailego::Closure::New(&ClassA::StaticProcess2, b[1], &b[2])->run();
  ailego::Closure::New(ClassA::StaticProcess3, b[1], &b[2], b[3])->run();
  ailego::Closure::New(&ClassA::StaticProcess3, 1, &b[2], b[3])->run();
  ailego::Closure::New(ClassA::StaticProcess4, 1, &b[2], b[3], 4)->run();
  ailego::Closure::New(&ClassA::StaticProcess4, b[1], &b[2], b[3], b[4])->run();
  ailego::Closure::New(ClassA::StaticProcess5, b[1], &b[2], b[3], b[4], &b[5])
      ->run();
  ailego::Closure::New(&ClassA::StaticProcess5, 1, &b[2], b[3], 4, &b[5])
      ->run();
  ailego::Closure::New(ClassA::StaticProcess6, 1, &b[2], b[3], 4, &b[5], &b[6])
      ->run();
  ailego::Closure::New(&ClassA::StaticProcess6, b[1], &b[2], b[3], b[4], &b[5],
                       &b[6])
      ->run();
  ailego::Closure::New(ClassA::StaticProcess7, 1, &b[2], b[3], 4, &b[5], &b[6],
                       b[7])
      ->run();
  ailego::Closure::New(&ClassA::StaticProcess7, b[1], &b[2], b[3], b[4], &b[5],
                       &b[6], 7)
      ->run();
}

TEST(Closure, Member) {
  long a[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  int b[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  ClassB bbb(1);

  ailego::Closure::New(&bbb, &ClassB::MemberFunction0)->run();
  ailego::Closure::New(&bbb, &ClassB::MemberFunction1, 1)->run();
  ailego::Closure::New(&bbb, &ClassB::MemberFunction2, a[1], &a[2])->run();
  ailego::Closure::New(&bbb, &ClassB::MemberFunction3, 1, &a[2], a[3])->run();
  ailego::Closure::New(&bbb, &ClassB::MemberFunction4, a[1], &a[2], a[3], a[4])
      ->run();
  ailego::Closure::New(&bbb, &ClassB::MemberFunction5, 1, &a[2], a[3], 4, &a[5])
      ->run();
  ailego::Closure::New(&bbb, &ClassB::MemberFunction6, a[1], &a[2], a[3], a[4],
                       &a[5], &a[6])
      ->run();
  ailego::Closure::New((const ClassB *)(&bbb), &ClassB::MemberFunction7, a[1],
                       &a[2], a[3], a[4], &a[5], &a[6], 7)
      ->run();
  ailego::Closure::New((const volatile ClassB *)(&bbb),
                       &ClassB::MemberFunction7, a[1], &a[2], a[3], a[4], &a[5],
                       &a[6], 7)
      ->run();

  ClassB &&bbc = std::move(bbb);
  ailego::Closure::New(&bbc, &ClassB::MemberProcess0)->run();
  ailego::Closure::New(&bbc, &ClassB::MemberProcess1, 1)->run();
  ailego::Closure::New(&bbc, &ClassB::MemberProcess2, b[1], &b[2])->run();
  ailego::Closure::New(&bbc, &ClassB::MemberProcess3, 1, &b[2], b[3])->run();
  ailego::Closure::New(&bbc, &ClassB::MemberProcess4, b[1], &b[2], b[3], b[4])
      ->run();

  ClassB &bbd = bbb;
  ailego::Closure::New(&bbd, &ClassB::MemberProcess5, 1, &b[2], b[3], 4, &b[5])
      ->run();
  ailego::Closure::New(&bbd, &ClassB::MemberProcess6, b[1], &b[2], b[3], b[4],
                       &b[5], &b[6])
      ->run();
  ailego::Closure::New(&bbd, &ClassB::MemberProcess7, b[1], &b[2], b[3], b[4],
                       &b[5], &b[6], 7)
      ->run();
}

TEST(Closure, Function) {
  long a[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  int b[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  ClassB bbb(1);

  std::function<decltype(GlobalFunction0)> f0 =
      std::bind(&ClassB::MemberFunction0, &bbb);
  ailego::Closure::New(f0)->run();

  std::function<decltype(GlobalFunction1)> f1 =
      std::bind(&ClassB::MemberFunction1, &bbb, std::placeholders::_1);
  ailego::Closure::New(f1, 1)->run();

  std::function<decltype(GlobalFunction2)> f2 =
      std::bind(&ClassB::MemberFunction2, &bbb, std::placeholders::_1,
                std::placeholders::_2);
  ailego::Closure::New(f2, a[1], &a[2])->run();

  std::function<decltype(GlobalFunction3)> f3 =
      std::bind(&ClassB::MemberFunction3, &bbb, std::placeholders::_1,
                std::placeholders::_2, std::placeholders::_3);
  ailego::Closure::New(f3, a[1], &a[2], a[3])->run();

  std::function<decltype(GlobalFunction4)> f4 = std::bind(
      &ClassB::MemberFunction4, &bbb, std::placeholders::_1,
      std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);
  ailego::Closure::New(f4, 1, &a[2], a[3], a[4])->run();

  std::function<decltype(GlobalFunction5)> f5 =
      std::bind(&ClassB::MemberFunction5, &bbb, std::placeholders::_1,
                std::placeholders::_2, std::placeholders::_3,
                std::placeholders::_4, std::placeholders::_5);
  ailego::Closure::New(f5, 1, &a[2], a[3], 4, &a[5])->run();

  std::function<decltype(GlobalFunction6)> f6 = std::bind(
      &ClassB::MemberFunction6, &bbb, std::placeholders::_1,
      std::placeholders::_2, std::placeholders::_3, std::placeholders::_4,
      std::placeholders::_5, std::placeholders::_6);
  ailego::Closure::New(f6, 1, &a[2], a[3], a[4], &a[5], &a[6])->run();

  std::function<decltype(GlobalFunction7)> f7 = std::bind(
      &ClassB::MemberFunction7, &bbb, std::placeholders::_1,
      std::placeholders::_2, std::placeholders::_3, std::placeholders::_4,
      std::placeholders::_5, std::placeholders::_6, std::placeholders::_7);
  ailego::Closure::New(f7, a[1], &a[2], a[3], a[4], &a[5], &a[6], 7)->run();

  std::function<decltype(GlobalProcess0)> p0 =
      std::bind(&ClassB::MemberProcess0, &bbb);
  ailego::Closure::New(p0)->run();

  std::function<decltype(GlobalProcess1)> p1 =
      std::bind(&ClassB::MemberProcess1, &bbb, std::placeholders::_1);
  ailego::Closure::New(p1, 1)->run();

  std::function<decltype(GlobalProcess2)> p2 =
      std::bind(&ClassB::MemberProcess2, &bbb, std::placeholders::_1,
                std::placeholders::_2);
  ailego::Closure::New(p2, b[1], &b[2])->run();

  std::function<decltype(GlobalProcess3)> p3 =
      std::bind(&ClassB::MemberProcess3, &bbb, std::placeholders::_1,
                std::placeholders::_2, std::placeholders::_3);
  ailego::Closure::New(p3, b[1], &b[2], b[3])->run();

  std::function<decltype(GlobalProcess4)> p4 = std::bind(
      &ClassB::MemberProcess4, &bbb, std::placeholders::_1,
      std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);
  ailego::Closure::New(p4, 1, &b[2], b[3], b[4])->run();

  std::function<decltype(GlobalProcess5)> p5 =
      std::bind(&ClassB::MemberProcess5, &bbb, std::placeholders::_1,
                std::placeholders::_2, std::placeholders::_3,
                std::placeholders::_4, std::placeholders::_5);
  ailego::Closure::New(p5, 1, &b[2], b[3], 4, &b[5])->run();

  std::function<decltype(GlobalProcess6)> p6 = std::bind(
      &ClassB::MemberProcess6, &bbb, std::placeholders::_1,
      std::placeholders::_2, std::placeholders::_3, std::placeholders::_4,
      std::placeholders::_5, std::placeholders::_6);
  ailego::Closure::New(p6, 1, &b[2], b[3], b[4], &b[5], &b[6])->run();

  std::function<decltype(GlobalProcess7)> p7 = std::bind(
      &ClassB::MemberProcess7, &bbb, std::placeholders::_1,
      std::placeholders::_2, std::placeholders::_3, std::placeholders::_4,
      std::placeholders::_5, std::placeholders::_6, std::placeholders::_7);
  ailego::Closure::New(p7, b[1], &b[2], b[3], b[4], &b[5], &b[6], 7)->run();
}

TEST(Closure, Lambda) {
  auto lambda0 = []() { return 0; };
  ailego::Closure::New(lambda0)->run();
  ailego::Closure::New([&]() { return 0; })->run();

  long a[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  ClassB bbb(1);

  auto lambda1 = [&](long a1) { return bbb.MemberFunction1(a1); };
  ailego::Closure::New(lambda1, 1)->run();

  auto lambda2 = [&](long a1, long *a2) { return bbb.MemberFunction2(a1, a2); };
  ailego::Closure::New(lambda2, 1, &a[2])->run();

  auto lambda3 = [&](long a1, long *a2, long &a3) {
    return bbb.MemberFunction3(a1, a2, a3);
  };
  ailego::Closure::New(lambda3, 1, &a[2], a[3])->run();

  auto lambda4 = [&](long a1, long *a2, long &a3, const long &a4) {
    return bbb.MemberFunction4(a1, a2, a3, a4);
  };
  ailego::Closure::New(lambda4, a[1], &a[2], a[3], a[4])->run();

  auto lambda5 = [&](long a1, long *a2, long &a3, const long &a4,
                     volatile long *a5) {
    return bbb.MemberFunction5(a1, a2, a3, a4, a5);
  };
  ailego::Closure::New(lambda5, 1, &a[2], a[3], 4, &a[5])->run();

  auto lambda6 = [&](long a1, long *a2, long &a3, const long &a4,
                     volatile long *a5, long *const volatile a6) {
    return bbb.MemberFunction6(a1, a2, a3, a4, a5, a6);
  };
  ailego::Closure::New(lambda6, 1, &a[2], a[3], 4, &a[5], &a[6])->run();

  auto lambda7 = [&](long a1, long *a2, long &a3, const long &a4,
                     volatile long *a5, long *const volatile a6, long &&a7) {
    return bbb.MemberFunction7(a1, a2, a3, a4, a5, a6, std::move(a7));
  };
  ailego::Closure::New(lambda7, a[1], &a[2], a[3], a[4], &a[5], &a[6], 7)
      ->run();
}

TEST(Closure, Return) {
  long a[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  size_t r = 0;
  ailego::Closure::New(&GlobalFunction0)->run(&r);
  EXPECT_EQ(0u, r);
  ailego::Closure::New(&GlobalFunction1, 1)->run(&r);
  EXPECT_EQ(1u, r);
  ailego::Closure::New(&GlobalFunction2, a[1], &a[2])->run(&r);
  EXPECT_EQ(2u, r);
  ailego::Closure::New(&GlobalFunction3, 1, &a[2], a[3])->run(&r);
  EXPECT_EQ(3u, r);
  ailego::Closure::New(&GlobalFunction4, a[1], &a[2], a[3], a[4])->run(&r);
  EXPECT_EQ(4u, r);
  ailego::Closure::New(&GlobalFunction5, 1, &a[2], a[3], 4, &a[5])->run(&r);
  EXPECT_EQ(5u, r);
  ailego::Closure::New(&GlobalFunction6, a[1], &a[2], a[3], a[4], &a[5], &a[6])
      ->run(&r);
  EXPECT_EQ(6u, r);
  ailego::Closure::New(&GlobalFunction7, a[1], &a[2], a[3], a[4], &a[5], &a[6],
                       7)
      ->run(&r);
  EXPECT_EQ(7u, r);

  ClassB bbb(1);
  ailego::Closure::New(&bbb, &ClassB::MemberFunction0)->run(&r);
  EXPECT_EQ(0u, r);
  ailego::Closure::New(&bbb, &ClassB::MemberFunction1, 1)->run(&r);
  EXPECT_EQ(1u, r);
  ailego::Closure::New(&bbb, &ClassB::MemberFunction2, a[1], &a[2])->run(&r);
  EXPECT_EQ(2u, r);
  ailego::Closure::New(&bbb, &ClassB::MemberFunction3, 1, &a[2], a[3])->run(&r);
  EXPECT_EQ(3u, r);
  ailego::Closure::New(&bbb, &ClassB::MemberFunction4, a[1], &a[2], a[3], a[4])
      ->run(&r);
  EXPECT_EQ(4u, r);
  ailego::Closure::New(&bbb, &ClassB::MemberFunction5, 1, &a[2], a[3], 4, &a[5])
      ->run(&r);
  EXPECT_EQ(5u, r);
  ailego::Closure::New(&bbb, &ClassB::MemberFunction6, a[1], &a[2], a[3], a[4],
                       &a[5], &a[6])
      ->run(&r);
  EXPECT_EQ(6u, r);
  ailego::Closure::New(&bbb, &ClassB::MemberFunction7, a[1], &a[2], a[3], a[4],
                       &a[5], &a[6], 7)
      ->run(&r);
  EXPECT_EQ(7u, r);
}

struct LeftValue {
  LeftValue(void) {
    std::cout << "LeftValue(void)" << std::endl;
  }
  LeftValue(const LeftValue &) {
    ++count;
    std::cout << "LeftValue(const LeftValue &)" << std::endl;
  }
  LeftValue(LeftValue &&) {
    std::cout << "LeftValue(LeftValue &&)" << std::endl;
    EXPECT_TRUE(0);
  }
  static int count;
  int val = 1;
};

int LeftValue::count = 0;

struct RightValue {
  RightValue(void) {
    std::cout << "RightValue(void)" << std::endl;
  }
  RightValue(const RightValue &) {
    std::cout << "RightValue(const RightValue &)" << std::endl;
    EXPECT_TRUE(0);
  }
  RightValue(RightValue &&) {
    ++count;
    std::cout << "RightValue(RightValue &&)" << std::endl;
  }
  static int count;
  int val = 2;
};

int RightValue::count = 0;

struct TestLeftRight {
  static int Run(LeftValue &&, const RightValue &) {
    return 0;
  }
  static int RunLeft(LeftValue &&) {
    return 0;
  }
  static int RunRight(const RightValue &) {
    return 0;
  }
};

TEST(Closure, LeftRight) {
  LeftValue lval;
  RightValue rval;

  std::cout << "## Starting 1..." << std::endl;
  ailego::Closure::New(&TestLeftRight::RunLeft, lval)->run();
  EXPECT_EQ(1, LeftValue::count);

  std::cout << "## Starting 2..." << std::endl;
  ailego::Closure::New(&TestLeftRight::RunRight, RightValue())->run();
  EXPECT_EQ(1, RightValue::count);

  std::cout << "## Starting 3..." << std::endl;
  auto call = ailego::Closure::New(&TestLeftRight::Run, std::ref(lval),
                                   std::move(rval));
  (*call)();
  EXPECT_EQ(2, LeftValue::count);
  EXPECT_EQ(2, RightValue::count);
}

void NoinlineFunction(int *a) {
  ++(*a);
}

TEST(Closure, Benchmark) {
  const int count = 10000000;

  ailego::ElapsedTime stamp0;
  int num0 = 0;
  typedef void (*FUNC)(int *);
  volatile FUNC fn0 = NoinlineFunction;
  for (int i = 0; i < count; i++) {
    (*fn0)(&num0);
  }
  std::cout << "Noinline elapsed: " << stamp0.micro_seconds() << " us"
            << std::endl;
  EXPECT_EQ(count, num0);

  ailego::ElapsedTime stamp1;
  int num1 = 0;
  auto fn1 = ailego::Closure::New([](int *a) { ++(*a); }, &num1);
  for (int i = 0; i < count; i++) {
    fn1->run();
  }
  std::cout << "Closure elapsed: " << stamp1.micro_seconds() << " us"
            << std::endl;
  EXPECT_EQ(count, num1);

  ailego::ElapsedTime stamp2;
  int num2 = 0;
  auto fn2 = [](int *a) { ++(*a); };
  for (int i = 0; i < count; i++) {
    fn2(&num2);
  }
  std::cout << "Lambda elapsed: " << stamp2.micro_seconds() << " us"
            << std::endl;
  EXPECT_EQ(count, num2);

  ailego::ElapsedTime stamp3;
  int num3 = 0;
  std::function<void(int *)> fn3 = [](int *a) { ++(*a); };
  for (int i = 0; i < count; i++) {
    fn3(&num3);
  }
  std::cout << "Function elapsed: " << stamp3.micro_seconds() << " us"
            << std::endl;
  EXPECT_EQ(count, num3);
}
