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

#pragma once

#include <ailego/io/file.h>
#include <ailego/logger/logger.h>
#include <gflags/gflags.h>
#include <zvec/db/status.h>
#include "ailego/pattern/factory.h"
#include "db/common/constants.h"
#include "error_code.h"

namespace zvec {

class LogUtil {
 public:
  static Status Init(const std::string &log_dir, const std::string &log_file,
                     int log_level, const std::string &logger_type,
                     int log_file_size, int log_overdue_days) {
    if (logger_type == FILE_LOG_TYPE_NAME) {
      if (log_dir.empty() || log_file.empty()) {
        return Status::InvalidArgument("log_dir or log_file is empty");
      }

      if (!ailego::File::IsExist(log_dir)) {
        ailego::File::MakePath(log_dir);
      }
    }

    auto logger =
        ailego::Factory<ailego::Logger>::MakeShared(logger_type.c_str());
    if (!logger) {
      LOG_FATAL("Invalid logger_type[%s]", logger_type.c_str());
      return Status::InvalidArgument("Invalid logger_type: ", logger_type);
    }

    ailego::Params params;
    if (logger_type == FILE_LOG_TYPE_NAME) {
      params.set("proxima.file.logger.log_dir", log_dir);
      params.set("proxima.file.logger.log_file", log_file);
      params.set("proxima.file.logger.path", log_dir + "/" + log_file);
      std::string program_name = ailego::File::BaseName(gflags::GetArgv0());
      params.set("proxima.program.program_name", program_name);
      params.set("proxima.file.logger.file_size", log_file_size);
      params.set("proxima.file.logger.overdue_days", log_overdue_days);
    }

    int ret = logger->init(params);
    if (ret != 0) {
      return Status::InternalError(ErrorCode::What(ret));
    }

    zvec::ailego::LoggerBroker::SetLevel(log_level);
    zvec::ailego::LoggerBroker::Register(logger);
    return Status::OK();
  }

  static void Shutdown() {
    zvec::ailego::LoggerBroker::Unregister();
  }
};

}  // namespace zvec
