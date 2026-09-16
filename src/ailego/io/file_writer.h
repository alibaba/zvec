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
#include <ios>
#include "file.h"

namespace zvec {
namespace ailego {

/*! File Stream Writer
 */
class FileWriter {
 public:
  //! Constructor
  FileWriter() {}

  //! Constructor
  FileWriter(FileWriter &&rhs) : file_(std::move(rhs.file_)) {}

  //! Destructor
  ~FileWriter() {}

  //! Assignment
  FileWriter &operator=(FileWriter &&rhs) {
    file_ = std::move(rhs.file_);
    return *this;
  }

  //! Output to writer
  FileWriter &operator<<(const char *str) {
    size_t len = std::strlen(str);
    if (file_.write(str, len) != len) {
      throw std::ios_base::failure("Write error");
    }
    return *this;
  }

  //! Output to writer
  FileWriter &operator<<(const std::string &str) {
    if (file_.write(str.data(), str.size()) != str.size()) {
      throw std::ios_base::failure("Write error");
    }
    return *this;
  }

  //! Output to writer
  FileWriter &operator<<(char c) {
    if (file_.write(&c, 1) != 1) {
      throw std::ios_base::failure("Write error");
    }
    return *this;
  }

  //! Test if the file is valid
  bool is_valid() const {
    return file_.is_valid();
  }

  //! Create a local file
  bool create(const char *path) {
    return file_.create(path, 0, false);
  }

  //! Open a local file
  bool open(const char *path) {
    return file_.open(path, false, false);
  }

  //! Close the local file
  void close() {
    file_.close();
  }

  //! Write data into the file
  size_t write(const void *data, size_t len) {
    return file_.write(data, len);
  }

  //! Synchronize memory with physical storage
  bool flush() {
    return file_.flush();
  }

  //! Output with format
  void print(const char *format, va_list args) {
    char buf[8192];
    std::vsnprintf(buf, sizeof(buf), format, args);
    (*this) << buf;
  }

  //! Output with format
#if defined(__GNUC__)
  void print(const char *format, ...) __attribute__((format(printf, 2, 3))) {
#else
  void print(const char *format, ...) {
#endif
    va_list args;
    va_start(args, format);
    this->print(format, args);
    va_end(args);
  }

 private:
  //! Disable them
  FileWriter(const FileWriter &) = delete;
  FileWriter &operator=(const FileWriter &) = delete;

  //! Members
  File file_;
};

}  // namespace ailego
}  // namespace zvec
