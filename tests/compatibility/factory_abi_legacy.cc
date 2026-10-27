// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0.

// This is the unmodified Factory header from ea60a361. Compile this translation
// unit separately from factory_abi_test.cc: it represents an existing client.
#include "legacy/include/zvec/ailego/pattern/factory.h"
#include "factory_abi_fixture.h"

namespace zvec {
namespace compatibility {
namespace {

template <int Order>
struct LegacyValue : FactoryABIBase<Order> {
  explicit LegacyValue(int input) : value_(input) {}
  int value() const override {
    return value_;
  }
  int value_;
};

template <int Order>
struct LegacyAccess : ailego::Factory<FactoryABIBase<Order>> {
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

struct LegacyArgumentValue : FactoryABIBase<0> {
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
  return sizeof(ailego::Factory<FactoryABIBase<0>>);
}

void LegacyFactoryRegister(int order, const char *key, int value) {
  if (order == 0) {
    ailego::Factory<FactoryABIBase<0>>::Register<LegacyValue<0>> registration(
        key, int{value});
  } else {
    ailego::Factory<FactoryABIBase<1>>::Register<LegacyValue<1>> registration(
        key, int{value});
  }
}

int LegacyFactoryCreate(int order, const char *key) {
  if (order == 0) {
    auto result = ailego::Factory<FactoryABIBase<0>>::MakeUnique(key);
    return result ? result->value() : -1;
  }
  auto result = ailego::Factory<FactoryABIBase<1>>::MakeUnique(key);
  return result ? result->value() : -1;
}

bool LegacyFactoryHas(int order, const char *key) {
  return order == 0 ? ailego::Factory<FactoryABIBase<0>>::Has(key)
                    : ailego::Factory<FactoryABIBase<1>>::Has(key);
}

size_t LegacyFactoryClassCount(int order) {
  return order == 0 ? ailego::Factory<FactoryABIBase<0>>::Classes().size()
                    : ailego::Factory<FactoryABIBase<1>>::Classes().size();
}

void LegacyFactoryRegisterArgument(const char *key,
                                   LegacyArgumentState *state) {
  ailego::Factory<FactoryABIBase<0>>::Register<LegacyArgumentValue>
      registration(key, LegacyArgument{state});
}

}  // namespace compatibility
}  // namespace zvec
