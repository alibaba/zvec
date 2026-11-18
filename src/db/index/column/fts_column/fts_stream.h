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

#include "fts_container.h"

namespace zvec::fts {

// Private candidate workspace. All offsets are checked before platform casts;
// temporary bytes are never a committed database file. I/O failure is sticky.
class FtsSpool {
 public:
  static Result<std::unique_ptr<FtsSpool>> Create();
  ~FtsSpool();
  FtsSpool(const FtsSpool &) = delete;
  FtsSpool &operator=(const FtsSpool &) = delete;
  uint64_t size() const {
    return size_;
  }
  Status append(std::string_view bytes);
  Status overwrite(uint64_t offset, std::string_view bytes);
  Result<std::string> read(uint64_t offset, size_t length);
  Status copy_to(FtsContainerWriter *container, const std::string &section);

 private:
  explicit FtsSpool(std::FILE *file) : file_(file) {}
  Status seek(uint64_t offset);
  std::FILE *file_;
  uint64_t size_{0};
  Status status_;
};

// Cursor-local verified page reuse. Views survive until the next read on this
// object. The cache owns the generation and its charged range lease. Different
// cursors never share mutable state, but can share backend pages.
class FtsSectionReader {
 public:
  static Result<std::unique_ptr<FtsSectionReader>> Open(
      FtsContainerReader::Pointer container, const std::string &section);
  uint64_t size() const {
    return size_;
  }
  Result<std::string_view> read(uint64_t offset, size_t length);

 private:
  FtsSectionReader() = default;
  FtsContainerReader::Pointer owner_;
  std::string name_;
  uint64_t size_{0};
  uint64_t cache_start_{0};
  FtsReadRange cache_;
};

}  // namespace zvec::fts
