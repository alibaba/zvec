// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0.

#pragma once

#include <cstddef>

namespace zvec::compatibility {

// Distinct categories exercise either version allocating the shared singleton
// first. Keep these types independent of both versions of the Factory header.
template <int Order>
struct FactoryABIBase {
  virtual ~FactoryABIBase() = default;
  virtual int value() const = 0;
};

// Keep client types distinct so the linker cannot merge their Register code.
template <int Order, bool LegacyClient>
struct FactoryABIValue : FactoryABIBase<Order> {
  explicit FactoryABIValue(int input) : value_(input) {}
  int value() const override {
    return value_;
  }
  int value_;
};

// The legacy client only registers in this category. It must not instantiate
// Make/produce, so the reentrancy test always exercises the current reader.
constexpr int kReentrantFactoryOrder = 2;

struct LegacyArgumentState {
  void (*on_copy)(){nullptr};
  void (*on_destroy)(){nullptr};
  size_t copies{0};
  size_t destructions{0};
};

const void *LegacyFactoryAddress(int order);
size_t LegacyFactorySize();
void LegacyFactoryRegister(int order, const char *key, int value);
int LegacyFactoryCreate(int order, const char *key);
bool LegacyFactoryHas(int order, const char *key);
size_t LegacyFactoryClassCount(int order);
void LegacyFactoryRegisterArgument(const char *key, LegacyArgumentState *state);

}  // namespace zvec::compatibility
