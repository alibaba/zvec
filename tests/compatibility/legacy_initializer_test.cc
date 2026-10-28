// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0.
int legacy_initialized = 0;
int RunCompatibilityTest() {
  return legacy_initialized == 29 ? 0 : 1;
}
