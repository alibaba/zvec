// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0.
#include <turbo/quantizer/fp32_quantizer/fp32_quantizer.h>
#include <zvec/core/framework/index_factory.h>
extern "C" void RegisterCompatibilityExtension() {
  static zvec::ailego::Factory<zvec::turbo::Quantizer>::Register<
      zvec::turbo::Fp32Quantizer>
      registration("PrebuiltQuantizer");
  (void)registration;
}
