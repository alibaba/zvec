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

#include "db/common/cgroup_util.h"

namespace zvec {

// Static member definitions
int CgroupUtil::cpu_cores_ = 0;
uint64_t CgroupUtil::memory_limit_ = 0;
bool CgroupUtil::initialized_ = false;
unsigned long long CgroupUtil::last_idle_time_ = 0;
unsigned long long CgroupUtil::last_total_time_ = 0;
std::chrono::steady_clock::time_point CgroupUtil::last_cpu_check_;

#define ZVEC_CGROUP_MEMORY_UNLIMITED (9223372036854771712ULL)

// Static initialization method
void CgroupUtil::initialize() {
  if (initialized_) {
    return;
  }

  updateCpuCores();
  updateMemoryLimit();
  initializeCpuStats();

  initialized_ = true;
}

int CgroupUtil::getCpuLimit() {
  initialize();
  return cpu_cores_;
}

uint64_t CgroupUtil::getMemoryLimit() {
  initialize();
  return memory_limit_;
}

// Other static methods implementation
double CgroupUtil::getCpuUsage() {
  initialize();
  return calculateCpuUsage();
}

uint64_t CgroupUtil::getMemoryUsage() {
  initialize();
  return getCurrentMemoryUsage();
}

uint64_t CgroupUtil::getUptime() {
#if defined(PLATFORM_LINUX)
  struct sysinfo info;
  if (sysinfo(&info) == 0) {
    return info.uptime;
  }
#elif defined(PLATFORM_MACOS)
  struct timeval boottime;
  size_t len = sizeof(boottime);
  int mib[2] = {CTL_KERN, KERN_BOOTTIME};
  if (sysctl(mib, 2, &boottime, &len, NULL, 0) == 0) {
    time_t bsec = boottime.tv_sec;
    time_t csec = time(NULL);
    return csec - bsec;
  }
#elif defined(PLATFORM_WINDOWS)
  return GetTickCount64() / 1000;
#endif
  return 0;
}

void CgroupUtil::updateCpuCores() {
  if (readCpuCgroup()) {
    return;
  }

#if defined(PLATFORM_MACOS)
  int cores;
  size_t len = sizeof(cores);
  if (sysctlbyname("hw.ncpu", &cores, &len, nullptr, 0) == 0) {
    cpu_cores_ = cores;
  } else {
    cpu_cores_ = 1;
  }
#elif defined(PLATFORM_LINUX)
  cpu_cores_ = sysconf(_SC_NPROCESSORS_ONLN);
  if (cpu_cores_ <= 0) {
    cpu_cores_ = 1;
  }
#elif defined(PLATFORM_WINDOWS)
  SYSTEM_INFO sysinfo;
  GetSystemInfo(&sysinfo);
  cpu_cores_ = static_cast<int>(sysinfo.dwNumberOfProcessors);
  if (cpu_cores_ <= 0) {
    cpu_cores_ = 1;
  }
#endif
}

bool CgroupUtil::readCpuCgroup() {
#if defined(PLATFORM_LINUX)
  // cgroup v2
  std::ifstream file("/sys/fs/cgroup/cpu.max");
  if (file.is_open()) {
    uint64_t quota, period;
    char slash;
    file >> quota >> slash >> period;
    file.close();

    if (quota != std::numeric_limits<uint64_t>::max() && quota != 0 &&
        period > 0) {
      cpu_cores_ =
          static_cast<int>(std::ceil(static_cast<double>(quota) / period));
      return true;
    } else {
      return false;
    }
  }

  // cgroup v1
  std::ifstream quota_file("/sys/fs/cgroup/cpu/cpu.cfs_quota_us");
  std::ifstream period_file("/sys/fs/cgroup/cpu/cpu.cfs_period_us");

  if (quota_file.is_open() && period_file.is_open()) {
    long long quota, period;
    quota_file >> quota;
    period_file >> period;
    quota_file.close();
    period_file.close();

    if (quota > 0 && period > 0) {
      cpu_cores_ =
          static_cast<int>(std::ceil(static_cast<double>(quota) / period));
      return true;
    }
  }
#endif
  return false;
}

void CgroupUtil::updateMemoryLimit() {
  if (readMemoryCgroup()) {
    return;
  }

#if defined(PLATFORM_MACOS)
  uint64_t mem;
  size_t len = sizeof(mem);
  if (sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) == 0) {
    memory_limit_ = mem;
  } else {
    memory_limit_ = 0;
  }
#elif defined(PLATFORM_LINUX)
  long pages = sysconf(_SC_PHYS_PAGES);
  long page_size = sysconf(_SC_PAGE_SIZE);
  if (pages > 0 && page_size > 0) {
    memory_limit_ = static_cast<uint64_t>(pages) * page_size;
  } else {
    memory_limit_ = 0;
  }
#elif defined(PLATFORM_WINDOWS)
  MEMORYSTATUSEX statex;
  statex.dwLength = sizeof(statex);
  if (GlobalMemoryStatusEx(&statex)) {
    memory_limit_ = statex.ullTotalPhys;
  } else {
    memory_limit_ = 0;
  }
#endif
}

bool CgroupUtil::readMemoryCgroup() {
#if defined(PLATFORM_LINUX)
  // cgroup v2
  std::ifstream file("/sys/fs/cgroup/memory.max");
  if (file.is_open()) {
    uint64_t limit;
    file >> limit;
    file.close();

    if (limit != std::numeric_limits<uint64_t>::max() && limit != 0 &&
        limit != ZVEC_CGROUP_MEMORY_UNLIMITED) {
      memory_limit_ = limit;
      return true;
    } else {
      return false;
    }
  }

  // cgroup v1
  std::ifstream v1_file("/sys/fs/cgroup/memory/memory.limit_in_bytes");
  if (v1_file.is_open()) {
    uint64_t limit;
    v1_file >> limit;
    v1_file.close();

    if (limit < std::numeric_limits<uint64_t>::max() &&
        limit != ZVEC_CGROUP_MEMORY_UNLIMITED) {
      memory_limit_ = limit;
      return true;
    }
  }
#endif
  return false;
}

void CgroupUtil::initializeCpuStats() {
  last_cpu_check_ = std::chrono::steady_clock::now();
#if defined(PLATFORM_LINUX)
  readProcStat();
#endif
}

#if defined(PLATFORM_LINUX)
bool CgroupUtil::readProcStat() {
  std::ifstream file("/proc/stat");
  if (!file.is_open()) {
    return false;
  }

  std::string line;
  if (!std::getline(file, line)) {
    return false;
  }

  std::istringstream iss(line);
  std::string cpu_label;
  iss >> cpu_label;

  if (cpu_label != "cpu") {
    return false;
  }

  unsigned long long user, nice, system, idle, iowait, irq, softirq, steal;
  iss >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal;

  unsigned long long idle_time = idle + iowait;
  unsigned long long total_time =
      user + nice + system + irq + softirq + steal + idle_time;

  last_idle_time_ = idle_time;
  last_total_time_ = total_time;

  return true;
}
#endif

uint64_t CgroupUtil::getCurrentMemoryUsage() {
#if defined(PLATFORM_LINUX)
  // cgroup
  uint64_t usage = readMemoryUsageCgroup();
  if (usage > 0) {
    return usage;
  }

  // back to /proc/meminfo
  return readMemoryUsageProc();
#elif defined(PLATFORM_MACOS)
  return getMacOSMemoryUsage();
#elif defined(PLATFORM_WINDOWS)
  MEMORYSTATUSEX statex;
  statex.dwLength = sizeof(statex);
  if (GlobalMemoryStatusEx(&statex)) {
    return statex.ullTotalPhys - statex.ullAvailPhys;
  }
  return 0;
#else
  return 0;
#endif
}

#if defined(PLATFORM_LINUX)
uint64_t CgroupUtil::readMemoryUsageCgroup() {
  // cgroup v2
  std::ifstream file("/sys/fs/cgroup/memory.current");
  if (file.is_open()) {
    uint64_t usage;
    file >> usage;
    file.close();
    return usage;
  }

  // cgroup v1
  std::ifstream v1_file("/sys/fs/cgroup/memory/memory.usage_in_bytes");
  if (v1_file.is_open()) {
    uint64_t usage;
    v1_file >> usage;
    v1_file.close();
    return usage;
  }

  return 0;
}

uint64_t CgroupUtil::readMemoryUsageProc() {
  std::ifstream file("/proc/meminfo");
  if (!file.is_open()) {
    return 0;
  }

  std::string line;
  uint64_t total_mem = 0;
  uint64_t free_mem = 0;
  uint64_t available_mem = 0;
  uint64_t buffers = 0;
  uint64_t cached = 0;

  while (std::getline(file, line)) {
    if (line.find("MemTotal:") == 0) {
      total_mem = extractMemoryValue(line);
    } else if (line.find("MemFree:") == 0) {
      free_mem = extractMemoryValue(line);
    } else if (line.find("MemAvailable:") == 0) {
      available_mem = extractMemoryValue(line);
    } else if (line.find("Buffers:") == 0) {
      buffers = extractMemoryValue(line);
    } else if (line.find("Cached:") == 0) {
      cached = extractMemoryValue(line);
    }
  }

  if (available_mem > 0 && total_mem > available_mem) {
    return total_mem - available_mem;
  }

  if (total_mem > 0 && free_mem > 0) {
    return total_mem - free_mem - buffers - cached;
  }

  return 0;
}
#endif

#if defined(PLATFORM_MACOS)
uint64_t CgroupUtil::getMacOSMemoryUsage() {
  mach_port_t host_port = mach_host_self();
  mach_msg_type_number_t host_size =
      sizeof(vm_statistics64_data_t) / sizeof(integer_t);
  vm_size_t page_size;
  vm_statistics64_data_t vm_stat;

  if (host_page_size(host_port, &page_size) != KERN_SUCCESS) {
    return 0;
  }

  if (host_statistics64(host_port, HOST_VM_INFO64, (host_info64_t)&vm_stat,
                        &host_size) != KERN_SUCCESS) {
    return 0;
  }

  uint64_t used_memory =
      ((vm_stat.active_count + vm_stat.inactive_count + vm_stat.wire_count) *
       page_size);

  return used_memory;
}
#endif

uint64_t CgroupUtil::extractMemoryValue(const std::string &line) {
  size_t colon_pos = line.find(':');
  if (colon_pos == std::string::npos) {
    return 0;
  }

  std::string value_str = line.substr(colon_pos + 1);
  std::istringstream iss(value_str);
  uint64_t value;
  std::string unit;

  iss >> value;
  if (iss >> unit) {
    if (unit == "kB") {
      value *= 1024;
    }
  }

  return value;
}

double CgroupUtil::calculateCpuUsage() {
#if defined(PLATFORM_LINUX)
  return calculateLinuxCpuUsage();
#elif defined(PLATFORM_MACOS)
  return calculateMacOSCpuUsage();
#else
  return 0.0;
#endif
}

#if defined(PLATFORM_LINUX)
double CgroupUtil::calculateLinuxCpuUsage() {
  if (!readProcStat()) {
    return 0.0;
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  std::ifstream file("/proc/stat");
  if (!file.is_open()) {
    return 0.0;
  }

  std::string line;
  if (!std::getline(file, line)) {
    return 0.0;
  }

  std::istringstream iss(line);
  std::string cpu_label;
  iss >> cpu_label;

  if (cpu_label != "cpu") {
    return 0.0;
  }

  unsigned long long user, nice, system, idle, iowait, irq, softirq, steal;
  iss >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal;

  unsigned long long current_idle = idle + iowait;
  unsigned long long current_total =
      user + nice + system + irq + softirq + steal + current_idle;

  unsigned long long idle_delta = current_idle - last_idle_time_;
  unsigned long long total_delta = current_total - last_total_time_;

  last_idle_time_ = current_idle;
  last_total_time_ = current_total;

  if (total_delta == 0) {
    return 0.0;
  }

  double cpu_usage =
      100.0 * (1.0 - static_cast<double>(idle_delta) / total_delta);
  return std::max(0.0, std::min(100.0, cpu_usage));
}
#endif

#if defined(PLATFORM_MACOS)
double CgroupUtil::calculateMacOSCpuUsage() {
  host_cpu_load_info_data_t cpuinfo;
  mach_msg_type_number_t count = HOST_CPU_LOAD_INFO_COUNT;

  if (host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO,
                      (host_info_t)&cpuinfo, &count) != KERN_SUCCESS) {
    return 0.0;
  }

  unsigned long long total_tick =
      cpuinfo.cpu_ticks[CPU_STATE_USER] + cpuinfo.cpu_ticks[CPU_STATE_SYSTEM] +
      cpuinfo.cpu_ticks[CPU_STATE_NICE] + cpuinfo.cpu_ticks[CPU_STATE_IDLE];

  unsigned long long idle_tick = cpuinfo.cpu_ticks[CPU_STATE_IDLE];

  static unsigned long long prev_total = 0;
  static unsigned long long prev_idle = 0;

  if (prev_total == 0) {
    prev_total = total_tick;
    prev_idle = idle_tick;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return calculateMacOSCpuUsage();
  }

  unsigned long long total_delta = total_tick - prev_total;
  unsigned long long idle_delta = idle_tick - prev_idle;

  prev_total = total_tick;
  prev_idle = idle_tick;

  if (total_delta == 0) {
    return 0.0;
  }

  double cpu_usage =
      100.0 * (1.0 - static_cast<double>(idle_delta) / total_delta);
  return std::max(0.0, std::min(100.0, cpu_usage));
}
#endif

}  // namespace zvec