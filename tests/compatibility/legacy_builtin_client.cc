// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0.
// This translation unit deliberately has no knowledge of builtin loaders.
#include "legacy/include/zvec/ailego/pattern/factory.h"

namespace zvec::core {
struct IndexMetric;
}
namespace zvec::turbo {
class Quantizer;
}

bool LegacyBuiltinsAvailable() {
  return zvec::ailego::Factory<zvec::core::IndexMetric>::Has("InnerProduct") &&
         zvec::ailego::Factory<zvec::turbo::Quantizer>::Has("Fp32Quantizer");
}
