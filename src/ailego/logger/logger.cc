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
#include <sstream>
#include <thread>
#include <zvec/ailego/io/file.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/ailego/utility/time_helper.h>

namespace zvec {
namespace ailego {

const int Logger::LEVEL_DEBUG = 0;
const int Logger::LEVEL_INFO = 1;
const int Logger::LEVEL_WARN = 2;
const int Logger::LEVEL_ERROR = 3;
const int Logger::LEVEL_FATAL = 4;

/*! Console Logger
 */
struct ConsoleLogger : public Logger {
  //! Initialize Logger
  int init(const Params &) override {
    return 0;
  }

  //! Cleanup Logger
  int cleanup(void) override {
    return 0;
  }

  //! Log Message
  void log(int level, const char *file, int line, const char *format,
           va_list args) override {
    char buffer[8192];
    std::ostringstream stream;

    ailego::Realtime::Localtime(buffer, sizeof(buffer));
    stream << '[' << LevelString(level) << ' ' << buffer << ' '
           << std::this_thread::get_id() << ' ' << ailego::File::BaseName(file)
           << ':' << line << "] ";

    vsnprintf(buffer, sizeof(buffer), format, args);
    stream << buffer << '\n';

    if (level <= LEVEL_INFO) {
      std::cout << stream.str() << std::flush;
    } else {
      std::cerr << stream.str() << std::flush;
    }
  }
};

//! Logger Level
int LoggerBroker::logger_level_ = Logger::LEVEL_WARN;

//! Logger
Logger::Pointer LoggerBroker::logger_(new ConsoleLogger);

//! Register Console Logger in Factory
FACTORY_REGISTER_LOGGER(ConsoleLogger);

}  // namespace ailego
}  // namespace zvec