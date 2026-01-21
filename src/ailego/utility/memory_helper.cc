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

#include "memory_helper.h"
#include <cstdio>
#include <cstring>
#include <zvec/ailego/utility/file_helper.h>
#include <zvec/ailego/utility/string_helper.h>

#if defined(_WIN64) || defined(_WIN32)
#include <Windows.h>
#include <psapi.h>
#else
#if defined(__linux__) || defined(__linux)
#include <sys/resource.h>
#elif defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif
#include <unistd.h>
#endif

namespace zvec {
namespace ailego {

#if defined(__linux__) || defined(__linux)
bool MemoryHelper::SelfUsage(size_t *vsz, size_t *rss) {
  FILE *fp = fopen("/proc/self/statm", "r");
  if (!fp) {
    return false;
  }

  if (fscanf(fp, "%zd %zd", vsz, rss) == EOF) {
    fclose(fp);
    return false;
  }
  fclose(fp);

  long pagesz = sysconf(_SC_PAGESIZE);
  *vsz *= (size_t)pagesz;
  *rss *= (size_t)pagesz;
  return true;
}

size_t MemoryHelper::SelfRSS(void) {
  FILE *fp = fopen("/proc/self/statm", "r");
  if (!fp) {
    return 0;
  }

  size_t rss = 0;
  if (fscanf(fp, "%*d %zd %*d", &rss) == EOF) {
    fclose(fp);
    return 0;
  }
  fclose(fp);
  return (rss * sysconf(_SC_PAGESIZE));
}

size_t MemoryHelper::SelfPeakRSS(void) {
  struct rusage rusage;
  getrusage(RUSAGE_SELF, &rusage);
  return (size_t)(rusage.ru_maxrss * 1024);
}

size_t MemoryHelper::TotalRamSize(void) {
  return (sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGESIZE));
}

size_t MemoryHelper::AvailableRamSize(void) {
  FILE *fp = fopen("/proc/meminfo", "r");
  if (!fp) {
    return 0;
  }

  size_t avail = 0;
  char buf[128];
  while (fgets(buf, sizeof(buf), fp)) {
    if (strncmp(buf, "MemAvailable:", 13) == 0) {
      avail = (size_t)strtoull(strchr(buf, ':') + 1, NULL, 10);
      break;
    }
  }

  // No found 'MemAvailable'
  if (avail == 0) {
    fseek(fp, 0L, SEEK_SET);

    size_t count = 0;
    while (fgets(buf, sizeof(buf), fp)) {
      switch (buf[0]) {
        case 'M':
          if (strncmp(buf, "MemFree:", 8) == 0) {
            avail += (size_t)strtoull(strchr(buf, ':') + 1, NULL, 10);
            ++count;
          }
          break;

        case 'B':
          if (strncmp(buf, "Buffers:", 8) == 0) {
            avail += (size_t)strtoull(strchr(buf, ':') + 1, NULL, 10);
            ++count;
          }
          break;

        case 'C':
          if (strncmp(buf, "Cached:", 7) == 0) {
            avail += (size_t)strtoull(strchr(buf, ':') + 1, NULL, 10);
            ++count;
          }
          break;
      }
      // All read
      if (count == 3) {
        break;
      }
    }
  }
  fclose(fp);
  return (avail * 1024);
}

size_t MemoryHelper::UsedRamSize(void) {
  FILE *fp = fopen("/proc/meminfo", "r");
  if (!fp) {
    return 0;
  }

  size_t total = 0, avail = 0, count = 0;
  char buf[128];

  while (fgets(buf, sizeof(buf), fp)) {
    switch (buf[0]) {
      case 'M':
        if (strncmp(buf, "MemTotal:", 9) == 0) {
          total = (size_t)strtoull(strchr(buf, ':') + 1, NULL, 10);
          ++count;
        } else if (strncmp(buf, "MemFree:", 8) == 0) {
          avail += (size_t)strtoull(strchr(buf, ':') + 1, NULL, 10);
          ++count;
        }
        break;

      case 'B':
        if (strncmp(buf, "Buffers:", 8) == 0) {
          avail += (size_t)strtoull(strchr(buf, ':') + 1, NULL, 10);
          ++count;
        }
        break;

      case 'C':
        if (strncmp(buf, "Cached:", 7) == 0) {
          avail += (size_t)strtoull(strchr(buf, ':') + 1, NULL, 10);
          ++count;
        }
        break;

      case 'S':
        if (strncmp(buf, "Slab:", 5) == 0) {
          avail += (size_t)strtoull(strchr(buf, ':') + 1, NULL, 10);
          ++count;
        }
        break;
    }
    // All read
    if (count == 5) {
      break;
    }
  }
  fclose(fp);

  if (total == 0) {
    total = (sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGESIZE)) / 1024;
  }
  return ((total - avail) * 1024);
}

size_t MemoryHelper::ContainerAwareTotalRamSize(void) {
  size_t total_ram_size = TotalRamSize();
  std::string limit_in_bytes = "/sys/fs/cgroup/memory/memory.limit_in_bytes";
  if (FileHelper::IsExist(limit_in_bytes.c_str())) {
    std::ifstream memory_limit_ifs;
    std::string memory_limit_str{""};
    memory_limit_ifs.open(limit_in_bytes, std::ios::in);
    if (memory_limit_ifs.is_open()) {
      uint64_t limit = 0;
      memory_limit_ifs >> memory_limit_str;
      if (memory_limit_str != "-1") {
        // Refer to:
        // https://access.redhat.com/documentation/zh-cn/red_hat_enterprise_linux/7/html/resource_management_guide/sec-memory
        StringHelper::ToUint64(memory_limit_str, &limit);
        if (limit != 0x7FFFFFFFFFFFF000) {
          // Refer to:
          // https://stackoverflow.com/questions/70332396/why-cgroups-file-memory-limit-in-bytes-use-9223372036854771712-as-a-default-valu
          total_ram_size = static_cast<size_t>(limit);
        }
      }
      memory_limit_ifs.close();
    }
  }
  return total_ram_size;
}

#elif defined(__APPLE__) && defined(__MACH__)
bool MemoryHelper::SelfUsage(size_t *vsz, size_t *rss) {
  struct mach_task_basic_info info;
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;

  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info,
                &count) != KERN_SUCCESS) {
    return false;
  }
  *vsz = info.virtual_size;
  *rss = info.resident_size;
  return true;
}

size_t MemoryHelper::SelfRSS(void) {
  struct mach_task_basic_info info;
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;

  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info,
                &count) != KERN_SUCCESS) {
    return 0;
  }
  return info.resident_size;
}

size_t MemoryHelper::SelfPeakRSS(void) {
  struct mach_task_basic_info info;
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;

  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info,
                &count) != KERN_SUCCESS) {
    return 0;
  }
  return info.resident_size_max;
}

size_t MemoryHelper::TotalRamSize(void) {
  int mib[2] = {CTL_HW, HW_MEMSIZE};
  uint64_t size = 0;
  size_t len = sizeof(size);
  if (sysctl(mib, 2, &size, &len, 0, 0) != 0) {
    return 0;
  }
  return (size_t)size;
}

size_t MemoryHelper::AvailableRamSize(void) {
  struct vm_statistics stat;
  mach_msg_type_number_t count = HOST_VM_INFO_COUNT;
  vm_size_t pagesize = 0;

  if (host_page_size(mach_host_self(), &pagesize) != KERN_SUCCESS) {
    return 0;
  }
  if (host_statistics(mach_host_self(), HOST_VM_INFO, (host_info_t)&stat,
                      &count) != KERN_SUCCESS) {
    return 0;
  }
  return ((stat.free_count + stat.inactive_count) * pagesize);
}

size_t MemoryHelper::UsedRamSize(void) {
  struct vm_statistics stat;
  mach_msg_type_number_t count = HOST_VM_INFO_COUNT;
  vm_size_t pagesize = 0;

  if (host_page_size(mach_host_self(), &pagesize) != KERN_SUCCESS) {
    return 0;
  }
  if (host_statistics(mach_host_self(), HOST_VM_INFO, (host_info_t)&stat,
                      &count) != KERN_SUCCESS) {
    return 0;
  }
  return ((stat.active_count + stat.wire_count) * pagesize);
}

size_t MemoryHelper::ContainerAwareTotalRamSize(void) {
  return 0u;
}

#elif defined(_WIN64) || defined(_WIN32)
static inline int getpagesize(void) {
  SYSTEM_INFO info;
  GetSystemInfo(&info);
  return info.dwPageSize;
}

bool MemoryHelper::SelfUsage(size_t *vsz, size_t *rss) {
  PROCESS_MEMORY_COUNTERS info;
  if (!GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info))) {
    return false;
  }
  *vsz = (size_t)info.PagefileUsage;
  *rss = (size_t)info.WorkingSetSize;
  return true;
}

size_t MemoryHelper::SelfRSS(void) {
  PROCESS_MEMORY_COUNTERS info;
  if (!GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info))) {
    return 0u;
  }
  return (size_t)info.WorkingSetSize;
}

size_t MemoryHelper::SelfPeakRSS(void) {
  PROCESS_MEMORY_COUNTERS info;
  GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
  return (size_t)info.PeakWorkingSetSize;
}

size_t MemoryHelper::TotalRamSize(void) {
  MEMORYSTATUSEX status;
  status.dwLength = sizeof(status);
  GlobalMemoryStatusEx(&status);
  return (size_t)status.ullTotalPhys;
}

size_t MemoryHelper::AvailableRamSize(void) {
  MEMORYSTATUSEX status;
  status.dwLength = sizeof(status);
  GlobalMemoryStatusEx(&status);
  return (size_t)status.ullAvailPhys;
}

size_t MemoryHelper::UsedRamSize(void) {
  MEMORYSTATUSEX status;
  status.dwLength = sizeof(status);
  GlobalMemoryStatusEx(&status);
  return (size_t)(status.ullTotalPhys - status.ullAvailPhys);
}

size_t MemoryHelper::ContainerAwareTotalRamSize(void) {
  return 0u;
}

#else
bool MemoryHelper::SelfUsage(size_t *vsz, size_t *rss) {
  *vsz = 0u;
  *rss = 0u;
  return false;
}

size_t MemoryHelper::SelfRSS(void) {
  return 0u;
}

size_t MemoryHelper::SelfPeakRSS(void) {
  return 0u;
}

size_t MemoryHelper::TotalRamSize(void) {
  return 0u;
}

size_t MemoryHelper::AvailableRamSize(void) {
  return 0u;
}

size_t MemoryHelper::UsedRamSize(void) {
  return 0u;
}

size_t MemoryHelper::ContainerAwareTotalRamSize(void) {
  return 0u;
}
#endif

size_t MemoryHelper::PageSize(void) {
  static size_t page_size = static_cast<size_t>(getpagesize());
  return page_size;
}

size_t MemoryHelper::HugePageSize(void) {
  static size_t page_size = static_cast<size_t>(2 * 1024 * 1024);
  return page_size;
}

}  // namespace ailego
}  // namespace zvec