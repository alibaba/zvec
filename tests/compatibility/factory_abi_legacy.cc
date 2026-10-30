// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0.

// This is the unmodified Factory header from ea60a361. Compile this translation
// unit separately from factory_abi_test.cc: it represents an existing client.
#include <utility>
#include "legacy/include/zvec/ailego/pattern/factory.h"
#include "factory_abi_fixture.h"

namespace zvec::compatibility {
namespace {

template <int Order>
using LegacyFactory = ailego::Factory<FactoryABIBase<Order>>;

template <int Order>
struct LegacyAccess : LegacyFactory<Order> {
  static const void *Address() {
    return LegacyAccess::Instance();
  }
};

struct LegacyArgument {
  explicit LegacyArgument(LegacyArgumentState *input) : state(input) {}
  LegacyArgument(const LegacyArgument &other) : state(other.state) {
    ++state->copies;
    if (state->on_copy) {
      state->on_copy();
    }
  }
  ~LegacyArgument() {
    ++state->destructions;
    if (state->on_destroy) {
      state->on_destroy();
    }
  }
  LegacyArgumentState *state;
};

struct LegacyArgumentValue : FactoryABIBase<kReentrantFactoryOrder> {
  explicit LegacyArgumentValue(const LegacyArgument &) {}
  int value() const override {
    return 73;
  }
};

}  // namespace

const void *LegacyFactoryAddress(int order) {
  return order == 0 ? LegacyAccess<0>::Address() : LegacyAccess<1>::Address();
}

size_t LegacyFactorySize() {
  return sizeof(LegacyFactory<0>);
}

void LegacyFactoryRegister(int order, const char *key, int value) {
  // MSVC without /Zc:rvalueCast can deduce int{value} as int& here, leaving
  // the legacy tuple with a reference to this local after registration returns.
  if (order == 0) {
    LegacyFactory<0>::Register<FactoryABIValue<0, true>> registration(
        key, std::move(value));
  } else {
    LegacyFactory<1>::Register<FactoryABIValue<1, true>> registration(
        key, std::move(value));
  }
}

int LegacyFactoryCreate(int order, const char *key) {
  if (order == 0) {
    auto result = LegacyFactory<0>::MakeUnique(key);
    return result ? result->value() : -1;
  }
  auto result = LegacyFactory<1>::MakeUnique(key);
  return result ? result->value() : -1;
}

bool LegacyFactoryHas(int order, const char *key) {
  return order == 0 ? LegacyFactory<0>::Has(key) : LegacyFactory<1>::Has(key);
}

size_t LegacyFactoryClassCount(int order) {
  return order == 0 ? LegacyFactory<0>::Classes().size()
                    : LegacyFactory<1>::Classes().size();
}

void LegacyFactoryRegisterArgument(const char *key,
                                   LegacyArgumentState *state) {
  LegacyFactory<kReentrantFactoryOrder>::Register<LegacyArgumentValue>
      registration(key, LegacyArgument{state});
}

}  // namespace zvec::compatibility
