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

#include <string>
#include <zvec/ailego/container/vector.h>

namespace zvec {
namespace ailego {

/*! Monotime
 */
struct Monotime {
  //! Retrieve monotonic time in nanoseconds
  static uint64_t NanoSeconds(void);

  //! Retrieve monotonic time in microseconds
  static uint64_t MicroSeconds(void);

  //! Retrieve monotonic time in milliseconds
  static uint64_t MilliSeconds(void);

  //! Retrieve monotonic time in seconds
  static uint64_t Seconds(void);
};

/*! Realtime
 */
struct Realtime {
  //! Retrieve system time in nanoseconds
  static uint64_t NanoSeconds(void);

  //! Retrieve system time in microseconds
  static uint64_t MicroSeconds(void);

  //! Retrieve system time in milliseconds
  static uint64_t MilliSeconds(void);

  //! Retrieve system time in seconds
  static uint64_t Seconds(void);

  //! Retrieve a timestamp as a specific local time format
  static size_t Localtime(uint64_t stamp, const char *format, char *buf,
                          size_t len);

  //! Retrieve a timestamp as a specific GMT time format
  static size_t Gmtime(uint64_t stamp, const char *format, char *buf,
                       size_t len);

  //! Retrieve local time in string
  static size_t Localtime(const char *format, char *buf, size_t len);

  //! Retrieve GMT time in string
  static size_t Gmtime(const char *format, char *buf, size_t len);

  //! Retrieve local time in string
  static size_t Localtime(char *buf, size_t len) {
    return Localtime("%Y-%m-%d %H:%M:%S", buf, len);
  }

  //! Retrieve GMT time in string
  static size_t Gmtime(char *buf, size_t len) {
    return Gmtime("%Y-%m-%d %H:%M:%S", buf, len);
  }

  //! Retrieve local time in string
  static std::string Localtime(void) {
    char str[32];
    Localtime(str, sizeof(str));
    return std::string(str);
  }

  //! Retrieve GMT time in string
  static std::string Gmtime(void) {
    char str[32];
    Gmtime(str, sizeof(str));
    return std::string(str);
  }

  //! Retrieve a timestamp as a specific local time format
  static size_t Localtime(uint64_t stamp, char *buf, size_t len) {
    return Localtime(stamp, "%Y-%m-%d %H:%M:%S", buf, len);
  }

  //! Retrieve a timestamp as a specific GMT time format
  static size_t Gmtime(uint64_t stamp, char *buf, size_t len) {
    return Gmtime(stamp, "%Y-%m-%d %H:%M:%S", buf, len);
  }

  //! Retrieve a timestamp as a specific local time format
  static std::string Localtime(uint64_t stamp) {
    char str[32];
    Localtime(stamp, str, sizeof(str));
    return std::string(str);
  }

  //! Retrieve a timestamp as a specific GMT time format
  static std::string Gmtime(uint64_t stamp) {
    char str[32];
    Gmtime(stamp, str, sizeof(str));
    return std::string(str);
  }
};

/*! Thread-specific CPU time
 */
struct CPUtime {
  //! Retrieve CPU time in nanoseconds
  static uint64_t NanoSeconds(void);

  //! Retrieve CPU time in microseconds
  static uint64_t MicroSeconds(void);

  //! Retrieve CPU time in milliseconds
  static uint64_t MilliSeconds(void);

  //! Retrieve CPU time in seconds
  static uint64_t Seconds(void);
};

/*! Elapsed Time
 */
class ElapsedTime {
 public:
  //! Constructor
  ElapsedTime(void) : stamp_(Monotime::NanoSeconds()) {}

  //! Retrieve the elapsed time in nanoseconds
  uint64_t nano_seconds(void) const {
    return (Monotime::NanoSeconds() - stamp_);
  }

  //! Retrieve the elapsed time in milliseconds
  uint64_t micro_seconds(void) const {
    return (this->nano_seconds() / 1000u);
  }

  //! Retrieve the elapsed time in milliseconds
  uint64_t milli_seconds(void) const {
    return (this->nano_seconds() / 1000000u);
  }

  //! Retrieve the elapsed time in seconds
  uint64_t seconds(void) const {
    return (this->nano_seconds() / 1000000000u);
  }

  //! Update time stamp
  void reset(void) {
    stamp_ = Monotime::NanoSeconds();
  }

 private:
  uint64_t stamp_;
};

/*! Elapsed CPU Time
 */
class ElapsedCPUTime {
 public:
  //! Constructor
  ElapsedCPUTime(void) : stamp_(CPUtime::NanoSeconds()) {}

  //! Retrieve the elapsed time in nanoseconds
  uint64_t nano_seconds(void) const {
    return (CPUtime::NanoSeconds() - stamp_);
  }

  //! Retrieve the elapsed time in milliseconds
  uint64_t micro_seconds(void) const {
    return (this->nano_seconds() / 1000u);
  }

  //! Retrieve the elapsed time in milliseconds
  uint64_t milli_seconds(void) const {
    return (this->nano_seconds() / 1000000u);
  }

  //! Retrieve the elapsed time in seconds
  uint64_t seconds(void) const {
    return (this->nano_seconds() / 1000000000u);
  }

  //! Update time stamp
  void reset(void) {
    stamp_ = CPUtime::NanoSeconds();
  }

 private:
  uint64_t stamp_;
};

}  // namespace ailego
}  // namespace zvec
