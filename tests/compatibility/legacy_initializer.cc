// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0.
extern int legacy_initialized;
namespace {
const bool initialized = [] {
  legacy_initialized = 29;
  return true;
}();
}  // namespace
