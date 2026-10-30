// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0.

#include <gtest/gtest.h>

int RunCompatibilityTest();

// Keep the native CMake link probes small while producing a complete test
// summary for runners such as iOS simctl, which cannot rely on the exit code.
TEST(Compatibility, NativeLinking) {
  EXPECT_EQ(0, RunCompatibilityTest());
}
