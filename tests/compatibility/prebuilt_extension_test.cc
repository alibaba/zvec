// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0.
#include <zvec/core/framework/index_factory.h>
int RunCompatibilityTest() {
  return zvec::core::IndexFactory::CreateQuantizer("PrebuiltQuantizer") ? 0 : 1;
}
