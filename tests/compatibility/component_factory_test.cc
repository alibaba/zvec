// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0.
#include <turbo/quantizer/quantizer.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_metric.h>
int main() {
  // Query the generic factory before invoking the facade.
  if (!zvec::ailego::Factory<zvec::core::IndexMetric>::Has("SquaredEuclidean"))
    return 1;
  if (!zvec::ailego::Factory<zvec::core::IndexMetric>::MakeUnique(
          "SquaredEuclidean"))
    return 2;
  if (!zvec::core::IndexFactory::CreateMetric("InnerProduct")) return 3;
  if (!zvec::core::IndexFactory::HasQuantizer("LegacyArchiveQuantizer"))
    return 4;
  if (!zvec::core::IndexFactory::CreateQuantizer("LegacyArchiveQuantizer"))
    return 5;
  auto overridden = zvec::core::IndexFactory::CreateQuantizer("Fp32Quantizer");
  return overridden && overridden->dim() == 73 ? 0 : 6;
}
