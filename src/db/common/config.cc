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

#include <iostream>
#include <memory>
#include <zvec/db/config.h>
#include <zvec/db/status.h>
#include "db/common/constants.h"
#include "db/common/global_resource.h"
#include "cgroup_util.h"
#include "global_resource.h"
#include "glogger.h"
#include "logger.h"
#include "typedef.h"

namespace zvec {

static void ExitLogHandler() {
  LogUtil::Shutdown();
}

GlobalConfig::ConfigData::ConfigData()
    : memory_limit_bytes(CgroupUtil::getMemoryLimit() *
                         DEFAULT_MEMORY_LIMIT_RATIO),
      log_config(std::make_shared<ConsoleLogConfig>()),
      query_thread_count(CgroupUtil::getCpuLimit()),
      invert_to_forward_scan_ratio(0.9),
      brute_force_by_keys_ratio(0.1),
      optimize_thread_count(CgroupUtil::getCpuLimit()) {}

Status GlobalConfig::Validate(const ConfigData &config) const {
  if (config.memory_limit_bytes < MIN_MEMORY_LIMIT_BYTES) {
    return Status::InvalidArgument("memory_limit_bytes must be greater than ",
                                   MIN_MEMORY_LIMIT_BYTES);
  }

  if (config.memory_limit_bytes > CgroupUtil::getMemoryLimit()) {
    return Status::InvalidArgument("memory_limit_bytes must be less than ",
                                   CgroupUtil::getMemoryLimit());
  }

  // Validate query thread count
  if (config.query_thread_count == 0) {
    return Status::InvalidArgument("query_thread_count must be greater than 0");
  }

  // Validate invert_to_forward_scan_ratio (should be between 0 and 1)
  if (config.invert_to_forward_scan_ratio < 0.0f ||
      config.invert_to_forward_scan_ratio > 1.0f) {
    return Status::InvalidArgument(
        "invert_to_forward_scan_ratio must be between 0 and 1");
  }

  // Validate brute_force_by_keys_ratio (should be between 0 and 1)
  if (config.brute_force_by_keys_ratio < 0.0f ||
      config.brute_force_by_keys_ratio > 1.0f) {
    return Status::InvalidArgument(
        "brute_force_by_keys_ratio must be between 0 and 1");
  }

  // Validate optimize thread count
  if (config.optimize_thread_count == 0) {
    return Status::InvalidArgument(
        "optimize_thread_count must be greater than 0");
  }

  // Validate log configuration
  if (config.log_config->GetLoggerType() == FILE_LOG_TYPE_NAME) {
    auto log_config =
        std::dynamic_pointer_cast<FileLogConfig>(config.log_config);

    // Validate file log specific configurations
    if (log_config->dir.empty()) {
      return Status::InvalidArgument(
          "log_dir cannot be empty when set to FileLogger");
    }

    if (log_config->basename.empty()) {
      return Status::InvalidArgument(
          "log_file basename cannot be empty when set to FileLogger");
    }

    if (log_config->file_size <= MIN_LOG_FILE_SIZE) {
      return Status::InvalidArgument("log file_size must be greater than ",
                                     MIN_LOG_FILE_SIZE,
                                     " when set to FileLogger");
    }

    if (log_config->overdue_days == 0) {
      return Status::InvalidArgument(
          "log_overdue_days must be greater than 0 when set to FileLogger");
    }
  }

  return Status::OK();
}

Status GlobalConfig::Initialize(const ConfigData &config) {
  // Use atomic compare-exchange to ensure only one initialization
  bool expected = false;
  if (!initialized_.compare_exchange_strong(expected, true)) {
    return Status::OK();
  }

  auto s = Validate(config);
  CHECK_RETURN_STATUS(s);

  config_ = config;

  s = LogUtil::Init(log_dir(), log_file_basename(), int(log_level()),
                    log_type(), log_file_size(), log_overdue_days());
  CHECK_RETURN_STATUS(s);

  if (std::atexit(ExitLogHandler) != 0) {
    std::cerr << "Failed to register exit handler" << std::endl;
    return Status::InternalError("Failed to register exit handler");
  }

  GlobalResource::Instance().initialize();
  return Status::OK();
}

uint64_t GlobalConfig::memory_limit_bytes() const noexcept {
  return config_.memory_limit_bytes;
}

FACTORY_REGISTER_LOGGER(AppendLogger);

}  // namespace zvec