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
#include <zvec/ailego/io/file.h>
#include <zvec/ailego/logger/logger.h>

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include <glog/logging.h>

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

namespace google {
namespace glog_internal_namespace_ {
extern bool IsGoogleLoggingInitialized(void);
extern bool ShutdownGoogleLoggingUtilities(void);
}  // namespace glog_internal_namespace_
}  // namespace google

namespace zvec {

class AppendLogger : public ailego::Logger {
 public:
  AppendLogger() = default;

  ~AppendLogger() {
    this->cleanup();
  }

 public:
  int init(const ailego::Params &params) override {
    if (!google::glog_internal_namespace_::IsGoogleLoggingInitialized()) {
      std::string log_dir = params.get_as_string("proxima.file.logger.log_dir");
      std::string log_file =
          params.get_as_string("proxima.file.logger.log_file");
      uint32_t log_file_size =
          params.get_as_uint32("proxima.file.logger.file_size");
      uint32_t log_overdue_days =
          params.get_as_uint32("proxima.file.logger.overdue_days");

      if (!ailego::File::IsExist(log_dir)) {
        ailego::File::MakePath(log_dir);
      }

      FLAGS_log_dir = log_dir;
      FLAGS_max_log_size = log_file_size;
      FLAGS_logbufsecs = 1;
      // it's really a bad feature for glog
      // logs <= LOG_FATAL will also output to stderr
      // and we can only set FATAL at most
      // and so we should avoid to use LOG_FATAL
      FLAGS_stderrthreshold = google::GLOG_FATAL;

      static std::string new_log_file = log_file;
      google::InitGoogleLogging(new_log_file.c_str());
      google::EnableLogCleaner(log_overdue_days);
    }
    return 0;
  }

  int cleanup() override {
    if (google::glog_internal_namespace_::IsGoogleLoggingInitialized()) {
      google::DisableLogCleaner();
      google::ShutdownGoogleLogging();
    }
    return 0;
  }

  void log(int level, const char *file, int line, const char *format,
           va_list args) override {
    static google::LogSeverity severities[] = {
        google::GLOG_INFO, google::GLOG_INFO, google::GLOG_WARNING,
        google::GLOG_ERROR, google::GLOG_FATAL};
    char buf[2048];
    vsnprintf(buf, sizeof(buf), format, args);
    google::LogMessage(file, line, severities[level]).stream() << buf;
    // NOTE: glog will flush WARN and above immediately, flush INFO every
    // `FLAGS_logbufsecs` or every 1M bytes. FlushLogFiles not needed.
    // google::FlushLogFiles(severities[level]);
  }
};

}  // namespace zvec
