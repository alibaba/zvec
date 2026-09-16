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

#include <atomic>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <zvec/ailego/io/file.h>
#include "wal_file.h"

namespace zvec {

/*
 * Wal Header info
 */
struct WalHeader {
  uint64_t wal_version{0U};
  uint64_t reserved_[7];
};

static_assert(sizeof(WalHeader) % 64 == 0,
              "Wal Header must be aligned with 64 bytes");

class WalRecord {
 public:
  uint32_t length_;
  uint32_t crc_;
  std::string content_;
};

class LocalWalFile : public WalFile {
 public:
  LocalWalFile(const LocalWalFile &) = delete;
  LocalWalFile &operator=(const LocalWalFile &) = delete;

  //! Constructor
  LocalWalFile(const std::string &wal_path) : wal_path_(wal_path) {}

  //! Destructor
  ~LocalWalFile() {
    if (opened_) {
      close();
    }
  }

 public:
  int append(std::string &&data) override;
  int prepare_for_read() override;
  std::string next() override;

 public:
  int open(const WalOptions &wal_option) override;

  int close() override;

  int flush() override;

  int remove() override;

  bool has_record() override {
    return file_.size() > sizeof(header_);
  }

 private:
  int write_record(WalRecord &record);
  int read_record(WalRecord &record);

 private:
  ailego::File file_;
  const static int32_t LENGTH_SIZE{4};
  const static int32_t CRC_SIZE{4};

 private:
  std::string wal_path_{};
  std::mutex file_mutex_;
  uint32_t max_docs_wal_flush_{0};
  std::atomic<uint64_t> docs_count_{0UL};
  WalHeader header_;

  bool opened_{false};
};


};  // namespace zvec
