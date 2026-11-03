// Custom gmock main for Android cross-compiled tests.
//
// On Android (NDK c++_static), static destructors of glog/gflags/etc. run
// after main() returns, often in unpredictable order. This causes segfaults
// (exit code 139) or aborts (134/135) during teardown even when all tests
// passed.  Using _exit() skips the static destructor phase entirely.

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <unistd.h>
#include "gmock/gmock.h"

GTEST_API_ int main(int argc, char **argv) {
  printf("Running main() from %s\n", __FILE__);
  // InitGoogleMock also initializes GoogleTest internally.
  testing::InitGoogleMock(&argc, argv);
  int ret = RUN_ALL_TESTS();
  // Flush all output so the test runner can parse results
  std::cout.flush();
  std::cerr.flush();
  fflush(stdout);
  fflush(stderr);
  // Use _exit() to skip static destructors and avoid glog/gflags teardown
  // crashes that are common with NDK c++_static builds.
  _exit(ret);
}
