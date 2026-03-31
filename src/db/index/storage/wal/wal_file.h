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

#include <memory>
#include <string>


namespace zvec {

class WalFile;

using WalFilePtr = std::shared_ptr<WalFile>;

struct WalOptions {
  uint32_t max_docs_wal_flush{0};
  bool create_new{false};
};

class WalFile {
 public:
  //! Constructor
  WalFile() {}

  //! Destructor
  virtual ~WalFile() {}  // LCOV_EXCL_LINE

  //! Create an instance
  static WalFilePtr Create(const std::string &wal_path);

  //! Crate an instance and open
  static int CreateAndOpen(const std::string &wal_path,
                           const WalOptions &wal_options, WalFilePtr *wal_file);

 public:
  virtual int append(std::string &&data) = 0;
  virtual int prepare_for_read() = 0;
  virtual std::string next() = 0;

 public:
  //! Open and initialize WalFile
  virtual int open(const WalOptions &wal_options) = 0;

  //! Close WalFile
  virtual int close() = 0;

  //! Remove wal disk file
  virtual int remove() = 0;

  //! Flush WalFile's memory to disk file
  virtual int flush() = 0;

  virtual bool has_record() = 0;
};

};  // namespace zvec