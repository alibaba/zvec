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

#include <cstdarg>
#include <memory>
#include <zvec/ailego/container/params.h>
#include <zvec/ailego/logger/logger.h>
#include "index_module.h"

#define ZVEC_LOG_IMPL(level, format, ...)                                   \
  do {                                                                      \
    if (zvec::core::IndexLoggerBroker::IsLevelEnabled(level)) {             \
      zvec::core::IndexLoggerBroker::Log(level, __FILE__, __LINE__, format, \
                                         ##__VA_ARGS__);                    \
    }                                                                       \
  } while (0)

//! Log Debug Message
#ifndef LOG_DEBUG
#define LOG_DEBUG(format, ...) \
  ZVEC_LOG_IMPL(zvec::core::IndexLogger::LEVEL_DEBUG, format, ##__VA_ARGS__)
#endif

//! Log Information Message
#ifndef LOG_INFO
#define LOG_INFO(format, ...) \
  ZVEC_LOG_IMPL(zvec::core::IndexLogger::LEVEL_INFO, format, ##__VA_ARGS__)
#endif

//! Log Warn Message
#ifndef LOG_WARN
#define LOG_WARN(format, ...) \
  ZVEC_LOG_IMPL(zvec::core::IndexLogger::LEVEL_WARN, format, ##__VA_ARGS__)
#endif

//! Log Error Message
#ifndef LOG_ERROR
#define LOG_ERROR(format, ...) \
  ZVEC_LOG_IMPL(zvec::core::IndexLogger::LEVEL_ERROR, format, ##__VA_ARGS__)
#endif

//! Log Fatal Message
#ifndef LOG_FATAL
#define LOG_FATAL(format, ...) \
  ZVEC_LOG_IMPL(zvec::core::IndexLogger::LEVEL_FATAL, format, ##__VA_ARGS__)
#endif

namespace zvec {
namespace core {

/*! Index Logger
 */
struct IndexLogger : public IndexModule {
  //! Index Logger Pointer
  typedef std::shared_ptr<IndexLogger> Pointer;

  static const int LEVEL_DEBUG;
  static const int LEVEL_INFO;
  static const int LEVEL_WARN;
  static const int LEVEL_ERROR;
  static const int LEVEL_FATAL;

  //! Retrieve string of level
  static const char *LevelString(int level) {
    static const char *info[] = {"DEBUG", " INFO", " WARN", "ERROR", "FATAL"};
    if (level < (int)(sizeof(info) / sizeof(info[0]))) {
      return info[level];
    }
    return "";
  }

  //! Retrieve symbol of level
  static char LevelSymbol(int level) {
    static const char info[5] = {'D', 'I', 'W', 'E', 'F'};
    if (level < (int)(sizeof(info) / sizeof(info[0]))) {
      return info[level];
    }
    return ' ';
  }

  //! Destructor
  virtual ~IndexLogger(void) {}

  //! Initialize Logger
  virtual int init(const zvec::ailego::Params &params) = 0;

  //! Cleanup Logger
  virtual int cleanup(void) = 0;

  //! Log Message
  virtual void log(int level, const char *file, int line, const char *format,
                   va_list args) = 0;
};

/*! Index Logger Broker
 */
class IndexLoggerBroker {
 public:
  //! Register Logger
  static IndexLogger::Pointer Register(IndexLogger::Pointer logger) {
    IndexLogger::Pointer ret = std::move(logger_);
    logger_ = std::move(logger);
    return ret;
  }

  //! Register Logger with init params
  static int Register(IndexLogger::Pointer logger,
                      const ailego::Params &params) {
    //! Cleanup the previous, before initizlizing the new one
    if (logger_) {
      logger_->cleanup();
    }
    logger_ = std::move(logger);
    return logger_->init(params);
  }

  //! Unregister Logger
  static void Unregister(void) {
    logger_ = nullptr;
  }

  //! Set Level of Logger
  static void SetLevel(int level) {
    logger_level_ = level;
  }

  //! Check if log level is enabled
  static bool IsLevelEnabled(int level) {
    return logger_level_ <= level && logger_;
  }

  //! Log Message
  __attribute__((format(printf, 4, 5))) static void Log(
      int level, const char *file, int line, const char *format, ...) {
    if (IsLevelEnabled(level)) {
      va_list args;
      va_start(args, format);
      logger_->log(level, file, line, format, args);
      va_end(args);
    }
  }

 private:
  //! Disable them
  IndexLoggerBroker(void) = delete;
  IndexLoggerBroker(const IndexLoggerBroker &) = delete;
  IndexLoggerBroker(IndexLoggerBroker &&) = delete;

  //! Members
  static int logger_level_;
  static IndexLogger::Pointer logger_;
};

}  // namespace core
}  // namespace zvec
