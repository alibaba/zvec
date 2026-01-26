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

#include <iostream>
#include <ailego/utility/type_helper.h>
#include <gtest/gtest.h>
#include <zvec/ailego/utility/float_helper.h>

using namespace zvec;

TEST(TypeHelper, IsArithmetic) {
  EXPECT_TRUE(ailego::IsArithmetic<uintptr_t>::value);
  EXPECT_TRUE(ailego::IsArithmetic<int>::value);
  EXPECT_TRUE(ailego::IsArithmetic<double>::value);
  EXPECT_TRUE(ailego::IsArithmetic<float>::value);
  EXPECT_TRUE(ailego::IsArithmetic<ailego::Float16>::value);
  EXPECT_FALSE(ailego::IsArithmetic<void>::value);
}

TEST(TypeHelper, IsFloatingPoint) {
  EXPECT_FALSE(ailego::IsFloatingPoint<long>::value);
  EXPECT_FALSE(ailego::IsFloatingPoint<int>::value);
  EXPECT_TRUE(ailego::IsFloatingPoint<double>::value);
  EXPECT_TRUE(ailego::IsFloatingPoint<float>::value);
  EXPECT_TRUE(ailego::IsFloatingPoint<ailego::Float16>::value);
  EXPECT_FALSE(ailego::IsFloatingPoint<void>::value);
}

template <typename... TArgs,
          typename = typename std::enable_if<
              ailego::Conjunction<std::is_integral<TArgs>...>::value>::type>
static bool TrueAnd(TArgs...) {
  return true;
}

template <typename... TArgs,
          typename = typename std::enable_if<
              !ailego::Conjunction<std::is_integral<TArgs>...>::value>::type>
static bool FalseAnd(TArgs...) {
  return false;
}

template <typename... TArgs,
          typename = typename std::enable_if<
              ailego::Disjunction<std::is_integral<TArgs>...>::value>::type>
static bool TrueOr(TArgs...) {
  return true;
}

template <typename... TArgs,
          typename = typename std::enable_if<
              !ailego::Disjunction<std::is_integral<TArgs>...>::value>::type>
static bool FalseOr(TArgs...) {
  return false;
}

TEST(TypeHelper, Conjunction) {
  EXPECT_TRUE(TrueAnd(1, 2, 2u, 0u));
  EXPECT_FALSE(FalseAnd(1, 2, 2u, ""));
  EXPECT_FALSE(FalseAnd(1, 2, 2u, 0.0));
}

TEST(TypeHelper, Disjunction) {
  EXPECT_TRUE(TrueOr(1, 2, 2u, ""));
  EXPECT_TRUE(TrueOr(0.0, "", 0u));
  EXPECT_FALSE(FalseOr("", ""));
  EXPECT_FALSE(FalseOr(0.0, ""));
}

struct TriviallyStruct {
  float a;
  uint32_t b;
};

TEST(TypeHelper, IsTriviallyCopyable) {
  EXPECT_TRUE(ailego::IsTriviallyCopyable<ailego::Float16>::value);
  EXPECT_TRUE(ailego::IsTriviallyCopyable<float>::value);
  EXPECT_TRUE(ailego::IsTriviallyCopyable<float>::value);
  EXPECT_TRUE(ailego::IsTriviallyCopyable<uint64_t>::value);
  EXPECT_TRUE(ailego::IsTriviallyCopyable<uint64_t *>::value);
  EXPECT_TRUE(ailego::IsTriviallyCopyable<void *>::value);
  // EXPECT_FALSE(ailego::IsTriviallyCopyable<uint64_t &>::value);
  EXPECT_TRUE(ailego::IsTriviallyCopyable<TriviallyStruct>::value);
}

TEST(TypeHelper, IsInvocable) {
  EXPECT_TRUE(ailego::IsInvocable<int()>::value);

  EXPECT_TRUE(!!(ailego::IsInvocableWithResult<int, int()>::value));
  EXPECT_TRUE(!!(ailego::IsInvocableWithResult<void, void(int), int>::value));
}

static_assert(ailego::IsInvocable<int()>::value, "");
static_assert(ailego::IsInvocableWithResult<int, int()>::value, "");
static_assert(ailego::IsInvocableWithResult<void, void(int), int>::value, "");
