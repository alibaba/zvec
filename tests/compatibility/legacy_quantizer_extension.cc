// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0.
#include <turbo/quantizer/fp32_quantizer/fp32_quantizer.h>
#include <zvec/core/framework/index_factory.h>
namespace zvec::core {
class LegacyArchiveOverride : public turbo::Fp32Quantizer {
 public:
  int dim() const override {
    return 73;
  }
};
INDEX_FACTORY_REGISTER_QUANTIZER_ALIAS(LegacyArchiveQuantizer,
                                       turbo::Fp32Quantizer);
INDEX_FACTORY_REGISTER_QUANTIZER_ALIAS(Fp32Quantizer, LegacyArchiveOverride);
}  // namespace zvec::core

// Retain this translation unit without changing the legacy registration macros.
extern "C" void RegisterLegacyQuantizerExtension() {}
