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

#include "concurrency_helper.h"
#include <fstream>
#include <iostream>
#include <thread>
#include <zvec/ailego/utility/file_helper.h>
#include <zvec/ailego/utility/string_helper.h>

namespace zvec {
namespace ailego {

// Refer to:
// https://stackoverflow.com/questions/65551215/get-docker-cpu-memory-limit-inside-container
ConcurrencyHelper::ConcurrencyHelper() {
  std::string cfs_quota_us = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us";
  std::string cfs_period_us = "/sys/fs/cgroup/cpu/cpu.cfs_period_us";

  concurrency_ = std::thread::hardware_concurrency();
  if (FileHelper::IsExist(cfs_quota_us.c_str()) &&
      FileHelper::IsExist(cfs_period_us.c_str())) {
    std::ifstream quota_ifs;
    std::string quota_str{""};
    uint32_t quota_val = 0;
    quota_ifs.open(cfs_quota_us, std::ios::in);
    if (quota_ifs.is_open()) {
      quota_ifs >> quota_str;
      if (quota_str != "-1") {
        StringHelper::ToUint32(quota_str, &quota_val);
      }
      quota_ifs.close();
    }

    if (quota_val > 0) {
      std::ifstream period_ifs;
      std::string period_str{""};
      uint32_t period_val = 0;
      period_ifs.open(cfs_period_us, std::ios::in);
      if (period_ifs.is_open()) {
        period_ifs >> period_str;
        StringHelper::ToUint32(period_str, &period_val);
        period_ifs.close();
      }

      if (period_val > 0) {
        concurrency_ = (quota_val + period_val - 1) / period_val;
      }
    }
  }
}

uint32_t ConcurrencyHelper::container_aware_concurrency() {
  static ConcurrencyHelper concurrency_helper;
  return concurrency_helper.concurrency_;
}

}  // namespace ailego
}  // namespace zvec
