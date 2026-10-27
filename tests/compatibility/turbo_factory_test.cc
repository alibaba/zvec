// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0.
#include <turbo/quantizer/quantizer.h>
int main() {
  using Factory = zvec::ailego::Factory<zvec::turbo::Quantizer>;
  if (Factory::Classes().size() != 7) return 1;
  for (const auto &name : Factory::Classes()) {
    if (!Factory::Has(name.c_str()) || !Factory::MakeUnique(name.c_str()))
      return 2;
  }
  return 0;
}
