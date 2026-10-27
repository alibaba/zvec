// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0.

#pragma once

#include <cstddef>

namespace zvec {
namespace compatibility {

// Distinct categories exercise either version allocating the shared singleton
// first. Keep these types independent of both versions of the Factory header.
template <int Order>
struct FactoryABIBase {
  virtual ~FactoryABIBase() = default;
  virtual int value() const = 0;
};

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

}  // namespace compatibility
}  // namespace zvec
