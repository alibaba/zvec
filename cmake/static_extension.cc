// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0.

#ifndef ZVEC_EXTENSION_INITIALIZER
#error "Specify the extension's extern C initialization entry point"
#endif

extern "C" void ZVEC_EXTENSION_INITIALIZER();

namespace {
const bool registered = [] {
  ZVEC_EXTENSION_INITIALIZER();
  return true;
}();
}  // namespace
