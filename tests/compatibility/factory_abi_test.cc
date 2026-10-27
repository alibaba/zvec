// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0.

#include <gtest/gtest.h>
#include <zvec/ailego/pattern/factory.h>
#include "factory_abi_fixture.h"

namespace zvec {
namespace compatibility {
namespace {

template <int Order>
using CurrentFactory = ailego::Factory<FactoryABIBase<Order>>;

template <int Order>
struct CurrentValue : FactoryABIBase<Order> {
  explicit CurrentValue(int input) : value_(input) {}
  int value() const override {
    return value_;
  }
  int value_;
};

template <int Order>
struct CurrentAccess : CurrentFactory<Order> {
  static const void *Address() {
    return CurrentAccess::Instance();
  }
};

template <int Order>
void CheckBothDirections() {
  EXPECT_EQ(LegacyFactorySize(), sizeof(CurrentFactory<Order>));
  EXPECT_EQ(LegacyFactoryAddress(Order), CurrentAccess<Order>::Address());

  LegacyFactoryRegister(Order, "LegacyEntry", 17);
  auto legacy = CurrentFactory<Order>::MakeUnique("LegacyEntry");
  ASSERT_NE(nullptr, legacy);
  EXPECT_EQ(17, legacy->value());
  EXPECT_TRUE(CurrentFactory<Order>::Has("LegacyEntry"));

  typename CurrentFactory<Order>::template Register<CurrentValue<Order>>
      current("CurrentEntry", 29);
  EXPECT_EQ(29, LegacyFactoryCreate(Order, "CurrentEntry"));
  EXPECT_TRUE(LegacyFactoryHas(Order, "CurrentEntry"));
  EXPECT_EQ(CurrentFactory<Order>::Classes().size(),
            LegacyFactoryClassCount(Order));

  // Each version must be able to destroy and replace the other's map value.
  LegacyFactoryRegister(Order, "CurrentEntry", 41);
  auto replaced = CurrentFactory<Order>::MakeUnique("CurrentEntry");
  ASSERT_NE(nullptr, replaced);
  EXPECT_EQ(41, replaced->value());
  typename CurrentFactory<Order>::template Register<CurrentValue<Order>>
      replace_legacy("LegacyEntry", 53);
  EXPECT_EQ(53, LegacyFactoryCreate(Order, "LegacyEntry"));

  typename CurrentFactory<Order>::template Register<CurrentValue<Order>>
      builtin(typename CurrentFactory<Order>::BuiltinTag{}, "CurrentEntry", 99);
  EXPECT_EQ(41, LegacyFactoryCreate(Order, "CurrentEntry"));
}

void QueryCurrentFactory() {
  CurrentFactory<0>::Has("LegacyReentrantArgument");
  CurrentFactory<0>::Classes();
}

}  // namespace

TEST(FactoryABI, LegacyClientAllocatesSingletonFirst) {
  ASSERT_NE(nullptr, LegacyFactoryAddress(0));
  CheckBothDirections<0>();
}

TEST(FactoryABI, CurrentClientAllocatesSingletonFirst) {
  ASSERT_NE(nullptr, CurrentAccess<1>::Address());
  CheckBothDirections<1>();
}

TEST(FactoryABI, LegacyCallableCanReenterCurrentFactory) {
  LegacyArgumentState state;
  LegacyFactoryRegisterArgument("LegacyReentrantArgument", &state);
  state.on_copy = QueryCurrentFactory;
  state.on_destroy = QueryCurrentFactory;
  const auto copies = state.copies;
  const auto destructions = state.destructions;

  auto result = CurrentFactory<0>::MakeUnique("LegacyReentrantArgument");
  ASSERT_NE(nullptr, result);
  EXPECT_EQ(73, result->value());
  EXPECT_GT(state.copies, copies);
  EXPECT_GT(state.destructions, destructions);

  // Release the legacy callback while its stack-owned test state is alive.
  CurrentFactory<0>::Register<CurrentValue<0>> replacement(
      "LegacyReentrantArgument", 83);
  EXPECT_EQ(83, LegacyFactoryCreate(0, "LegacyReentrantArgument"));
}

}  // namespace compatibility
}  // namespace zvec
