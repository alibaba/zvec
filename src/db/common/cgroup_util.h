// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(__APPLE__)
#define PLATFORM_MACOS 1
#include <mach/mach.h>
#include <sys/sysctl.h>
#elif defined(__linux__)
#define PLATFORM_LINUX 1
#include <sys/sysinfo.h>
#include <unistd.h>
#elif defined(_WIN32) || defined(_WIN64)
#define PLATFORM_WINDOWS 1
#include <windows.h>
#endif

namespace zvec {

class CgroupUtil {
 public:
  // Static methods to get CPU and memory limits
  static int getCpuLimit();
  static uint64_t getMemoryLimit();

  // Static methods to get other resources
  static double getCpuUsage();
  static uint64_t getMemoryUsage();
  static uint64_t getUptime();

 private:
  CgroupUtil() = default;
  ~CgroupUtil() = default;

  // Static member variables to store the computed values
  static int cpu_cores_;
  static uint64_t memory_limit_;
  static bool initialized_;

  // Other member variables for tracking state
  static unsigned long long last_idle_time_;
  static unsigned long long last_total_time_;
  static std::chrono::steady_clock::time_point last_cpu_check_;

  // Static initialization method
  static void initialize();

  // Helper methods (also made static)
  static void updateCpuCores();
  static bool readCpuCgroup();
  static void updateMemoryLimit();
  static bool readMemoryCgroup();
  static void initializeCpuStats();

#if defined(PLATFORM_LINUX)
  static bool readProcStat();
#endif

  static uint64_t getCurrentMemoryUsage();

#if defined(PLATFORM_LINUX)
  static uint64_t readMemoryUsageCgroup();
  static uint64_t readMemoryUsageProc();
#endif

#if defined(PLATFORM_MACOS)
  static uint64_t getMacOSMemoryUsage();
#endif

  static uint64_t extractMemoryValue(const std::string &line);
  static double calculateCpuUsage();

#if defined(PLATFORM_LINUX)
  static double calculateLinuxCpuUsage();
#endif

#if defined(PLATFORM_MACOS)
  static double calculateMacOSCpuUsage();
#endif
};

}  // namespace zvec