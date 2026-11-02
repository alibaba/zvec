// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0.
#include <zvec/core/framework/index_factory.h>

bool LegacyBuiltinsAvailable();

int RunCompatibilityTest() {
  // Query from old inline code before any call to a modern lazy loader.
  if (!LegacyBuiltinsAvailable()) return 1;
  // Ordinary references retain both SDK catalogs; no forced archive loading.
  if (!zvec::core::IndexFactory::CreateMetric("InnerProduct")) return 2;
  if (!zvec::core::IndexFactory::CreateQuantizer("Fp32Quantizer")) return 3;
  return 0;
}
