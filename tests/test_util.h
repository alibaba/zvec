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

#include <cstdlib>
#include <string>
#include <zvec/ailego/utility/file_helper.h>

#ifdef __APPLE__
#include <TargetConditionals.h>
#if TARGET_OS_IOS || TARGET_OS_SIMULATOR
#include <glob.h>
#endif
#endif

#ifdef _MSC_VER
#include <fcntl.h>
#include <io.h>
#include <chrono>
#include <thread>
#endif

namespace zvec {
namespace test_util {

inline void RemoveTestPath(const std::string &path) {
  if (!ailego::FileHelper::RemovePath(path.c_str())) {
#ifdef _WIN32
    system(("rmdir /s /q \"" + path + "\" 2>NUL").c_str());
#endif
  }
}

inline void RemoveTestFiles(const std::string &pattern) {
  if (pattern.find('*') != std::string::npos ||
      pattern.find('?') != std::string::npos) {
#ifdef _WIN32
    system(("del /f /q " + pattern + " 2>NUL").c_str());
#elif defined(__APPLE__) && (TARGET_OS_IOS || TARGET_OS_SIMULATOR)
    glob_t globbuf;
    if (glob(pattern.c_str(), 0, nullptr, &globbuf) == 0) {
      for (size_t i = 0; i < globbuf.gl_pathc; ++i) {
        ailego::FileHelper::RemovePath(globbuf.gl_pathv[i]);
      }
      globfree(&globbuf);
    }
#else
    system(("rm -rf " + pattern).c_str());
#endif
  } else {
    ailego::FileHelper::RemovePath(pattern.c_str());
  }
}

}  // namespace test_util
}  // namespace zvec

#ifdef _MSC_VER
inline void sleep(unsigned int seconds) {
  std::this_thread::sleep_for(std::chrono::seconds(seconds));
}

inline void usleep(unsigned int microseconds) {
  std::this_thread::sleep_for(std::chrono::microseconds(microseconds));
}

inline int truncate(const char *path, long length) {
  int fd = _open(path, _O_RDWR);
  if (fd < 0) return -1;
  int ret = _chsize(fd, length);
  _close(fd);
  return ret;
}
#endif
