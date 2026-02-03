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

#include <zvec/ailego/utility/time_helper.h>

#if defined(_WIN64) || defined(_WIN32)
#include <Windows.h>
#endif

namespace zvec {
namespace ailego {

#if defined(_WIN64) || defined(_WIN32)
uint64_t Monotime::NanoSeconds(void) {
  LARGE_INTEGER stamp, freq;
  QueryPerformanceFrequency(&freq);
  QueryPerformanceCounter(&stamp);
  return (uint64_t)((double)stamp.QuadPart *
                    (1000000000.0 / (double)freq.QuadPart));
}

uint64_t Monotime::MicroSeconds(void) {
  LARGE_INTEGER stamp, freq;
  QueryPerformanceFrequency(&freq);
  QueryPerformanceCounter(&stamp);
  return (stamp.QuadPart * 1000000u / freq.QuadPart);
}

uint64_t Monotime::MilliSeconds(void) {
  LARGE_INTEGER stamp, freq;
  QueryPerformanceFrequency(&freq);
  QueryPerformanceCounter(&stamp);
  return (stamp.QuadPart * 1000u / freq.QuadPart);
}

uint64_t Monotime::Seconds(void) {
  LARGE_INTEGER stamp, freq;
  QueryPerformanceFrequency(&freq);
  QueryPerformanceCounter(&stamp);
  return (stamp.QuadPart / freq.QuadPart);
}

// January 1, 1970 (start of Unix epoch) in "ticks"
#define UNIX_TIME_START 0x019DB1DED53E8000ull

uint64_t Realtime::NanoSeconds(void) {
  LARGE_INTEGER stamp;
  FILETIME file;
  GetSystemTimeAsFileTime(&file);
  stamp.HighPart = file.dwHighDateTime;
  stamp.LowPart = file.dwLowDateTime;
  return (stamp.QuadPart - UNIX_TIME_START) * 100u;
}

uint64_t Realtime::MicroSeconds(void) {
  LARGE_INTEGER stamp;
  FILETIME file;
  GetSystemTimeAsFileTime(&file);
  stamp.HighPart = file.dwHighDateTime;
  stamp.LowPart = file.dwLowDateTime;
  return (stamp.QuadPart - UNIX_TIME_START) / 10u;
}

uint64_t Realtime::MilliSeconds(void) {
  LARGE_INTEGER stamp;
  FILETIME file;
  GetSystemTimeAsFileTime(&file);
  stamp.HighPart = file.dwHighDateTime;
  stamp.LowPart = file.dwLowDateTime;
  return (stamp.QuadPart - UNIX_TIME_START) / 10000u;
}

uint64_t Realtime::Seconds(void) {
  LARGE_INTEGER stamp;
  FILETIME file;
  GetSystemTimeAsFileTime(&file);
  stamp.HighPart = file.dwHighDateTime;
  stamp.LowPart = file.dwLowDateTime;
  return (stamp.QuadPart - UNIX_TIME_START) / 10000000u;
}

size_t Realtime::Localtime(uint64_t stamp, const char *format, char *buf,
                           size_t len) {
  time_t val = static_cast<time_t>(stamp);
  return strftime(buf, len, format, localtime(&val));
}

size_t Realtime::Gmtime(uint64_t stamp, const char *format, char *buf,
                        size_t len) {
  time_t val = static_cast<time_t>(stamp);
  return strftime(buf, len, format, gmtime(&val));
}

size_t Realtime::Localtime(const char *format, char *buf, size_t len) {
  time_t now = time(0);
  return strftime(buf, len, format, localtime(&now));
}

size_t Realtime::Gmtime(const char *format, char *buf, size_t len) {
  time_t now = time(0);
  return strftime(buf, len, format, gmtime(&now));
}
#else
uint64_t Monotime::NanoSeconds(void) {
  struct timespec tspec;
  clock_gettime(CLOCK_MONOTONIC, &tspec);
  return (tspec.tv_sec * 1000000000u + tspec.tv_nsec);
}

uint64_t Monotime::MicroSeconds(void) {
  struct timespec tspec;
  clock_gettime(CLOCK_MONOTONIC, &tspec);
  return (tspec.tv_sec * 1000000u + tspec.tv_nsec / 1000u);
}

uint64_t Monotime::MilliSeconds(void) {
  struct timespec tspec;
  clock_gettime(CLOCK_MONOTONIC, &tspec);
  return (tspec.tv_sec * 1000u + tspec.tv_nsec / 1000000u);
}

uint64_t Monotime::Seconds(void) {
  struct timespec tspec;
  clock_gettime(CLOCK_MONOTONIC, &tspec);
  return (tspec.tv_sec);
}

uint64_t Realtime::NanoSeconds(void) {
  struct timespec tspec;
  clock_gettime(CLOCK_REALTIME, &tspec);
  return (tspec.tv_sec * 1000000000u + tspec.tv_nsec);
}

uint64_t Realtime::MicroSeconds(void) {
  struct timespec tspec;
  clock_gettime(CLOCK_REALTIME, &tspec);
  return (tspec.tv_sec * 1000000u + tspec.tv_nsec / 1000u);
}

uint64_t Realtime::MilliSeconds(void) {
  struct timespec tspec;
  clock_gettime(CLOCK_REALTIME, &tspec);
  return (tspec.tv_sec * 1000u + tspec.tv_nsec / 1000000u);
}

uint64_t Realtime::Seconds(void) {
  struct timespec tspec;
  clock_gettime(CLOCK_REALTIME, &tspec);
  return (tspec.tv_sec);
}

size_t Realtime::Localtime(uint64_t stamp, const char *format, char *buf,
                           size_t len) {
  struct tm tmbuf;
  time_t val = static_cast<time_t>(stamp);
  return strftime(buf, len, format, localtime_r(&val, &tmbuf));
}

size_t Realtime::Gmtime(uint64_t stamp, const char *format, char *buf,
                        size_t len) {
  struct tm tmbuf;
  time_t val = static_cast<time_t>(stamp);
  return strftime(buf, len, format, gmtime_r(&val, &tmbuf));
}

size_t Realtime::Localtime(const char *format, char *buf, size_t len) {
  struct tm tmbuf;
  time_t now = time(0);
  return strftime(buf, len, format, localtime_r(&now, &tmbuf));
}

size_t Realtime::Gmtime(const char *format, char *buf, size_t len) {
  struct tm tmbuf;
  time_t now = time(0);
  return strftime(buf, len, format, gmtime_r(&now, &tmbuf));
}

uint64_t CPUtime::NanoSeconds(void) {
  struct timespec tspec;
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &tspec);
  return (tspec.tv_sec * 1000000000u + tspec.tv_nsec);
}

uint64_t CPUtime::MicroSeconds(void) {
  struct timespec tspec;
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &tspec);
  return (tspec.tv_sec * 1000000u + tspec.tv_nsec / 1000u);
}

uint64_t CPUtime::MilliSeconds(void) {
  struct timespec tspec;
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &tspec);
  return (tspec.tv_sec * 1000u + tspec.tv_nsec / 1000000u);
}

uint64_t CPUtime::Seconds(void) {
  struct timespec tspec;
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &tspec);
  return (tspec.tv_sec);
}
#endif  // _WIN64 || _WIN32

}  // namespace ailego
}  // namespace zvec