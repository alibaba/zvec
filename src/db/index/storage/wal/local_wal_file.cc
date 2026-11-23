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

#include "local_wal_file.h"
#ifndef _MSC_VER
#include <unistd.h>
#endif
#include <zvec/ailego/hash/crc32c.h>
#include <zvec/ailego/logger/logger.h>
#include "db/common/error_code.h"
#include "db/common/file_helper.h"
#include "db/common/typedef.h"

namespace zvec {

namespace {
constexpr size_t kMaxRecordSize = 4 * 1024 * 1024;
}  // namespace

int LocalWalFile::append(std::string &&data) {
  CHECK_STATUS(read_only_, false);
  if (data.empty() || data.size() > kMaxRecordSize) {
    return -1;
  }
  WalRecord record;
  record.length_ = data.size();
  record.crc_ = ailego::Crc32c::Hash(
      reinterpret_cast<const void *>(data.data()), record.length_, 0);
  record.content_ = std::forward<std::string>(data);

  std::lock_guard<std::mutex> lock(file_mutex_);
  if (truncated_at_) {
    return -1;
  }
  if (write_record(record) < 0) {
    WLOG_ERROR("Wal write record error. record.length_[%zu]",
               (size_t)record.length_);
    return -1;
  }
  // if max_docs_wal_flush_ is 0, no need flush
  if (max_docs_wal_flush_ != 0 && docs_count_ >= max_docs_wal_flush_) {
    if (flush() != 0) {
      WLOG_ERROR("Wal flush error. docs_count_[%zu] max_docs_wal_flush_[%zu]",
                 (size_t)docs_count_, (size_t)max_docs_wal_flush_);
      return -1;
    }
    docs_count_ = 0;
  }
  return 0;
}

std::string LocalWalFile::next() {
  auto result = next_checked();
  if (!result) {
    WLOG_ERROR("Wal read failed: %s", result.error().message().c_str());
    return {};
  }
  return std::move(result->data);
}

int LocalWalFile::open(const WalOptions &wal_option) {
  CHECK_STATUS(opened_, false);
  if (wal_option.read_only && wal_option.create_new) {
    WLOG_ERROR("Cannot create a read-only WAL file");
    return -1;
  }
  if (wal_option.create_new) {
    if (FileHelper::FileExists(wal_path_)) {
      WLOG_ERROR("Wal open error. file already exist create_new[%d]",
                 wal_option.create_new);
      return -1;
    }

    if (!file_.create(wal_path_, false)) {
      WLOG_ERROR("Wal create error. create_new[%d]", wal_option.create_new);
      return -1;
    }

    // write wal header
    int write_size = file_.write((const void *)&header_, sizeof(header_));
    if (write_size != sizeof(header_)) {
      WLOG_ERROR("Wal write header error. create_new[%d]",
                 wal_option.create_new);
      return -1;
    }

  } else {
    if (!FileHelper::FileExists(wal_path_)) {
      WLOG_ERROR("Wal open error. file is not exist create_new[%d]",
                 wal_option.create_new);
      return -1;
    }

    if (!file_.open(wal_path_.c_str(), wal_option.read_only)) {
      WLOG_ERROR("Wal open error. create_new[%d]", wal_option.create_new);
      return -1;
    }

    // Appends start at the end; prepare_for_read() rewinds readers.
    if (!file_.seek(0, ailego::File::Origin::End)) {
      file_.close();
      return -1;
    }
  }

  max_docs_wal_flush_ = wal_option.max_docs_wal_flush;
  docs_count_ = 0;
  truncated_at_.reset();
  read_only_ = wal_option.read_only;
  opened_ = true;

  WLOG_INFO("Wal open success. create_new[%d]", wal_option.create_new);
  return 0;
}

int LocalWalFile::close() {
  CHECK_STATUS(opened_, true);
  file_.close();
  WLOG_INFO("Wal close success");
  opened_ = false;
  return 0;
}

int LocalWalFile::remove() {
  CHECK_STATUS(read_only_, false);
  if (opened_) {
    close();
  }
  if (FileHelper::FileExists(wal_path_)) {
    FileHelper::RemoveFile(wal_path_);
    WLOG_INFO("Wal remove success.");
  }
  return 0;
}

int LocalWalFile::flush() {
  CHECK_STATUS(opened_, true);
  CHECK_STATUS(read_only_, false);
  if (!file_.flush()) {
    WLOG_ERROR("Wal flush error.");
    return -1;
  }
  return 0;
}

int LocalWalFile::prepare_for_read() {
  CHECK_STATUS(opened_, true);
  truncated_at_.reset();
  if (!file_.seek(0, ailego::File::Origin::Begin)) {
    return -1;
  }
  int read_size = file_.read((void *)&header_, sizeof(header_));
  if (read_size != sizeof(header_)) {
    WLOG_ERROR("Wal read header error.");
    return -1;
  }
  if (header_.wal_version != 0UL) {
    WLOG_ERROR("Wal version not support error.");
    return -1;
  }
  return 0;
}

//! Return 1 if success or -1 if write error
int LocalWalFile::write_record(WalRecord &record) {
  CHECK_STATUS(opened_, true);

  int write_size = 0;
  int ret = -1;

  do {
    write_size = file_.write((const void *)&record.length_, LENGTH_SIZE);
    if (write_size != LENGTH_SIZE) {
      WLOG_ERROR("Wal write error. record.length_ error write_size[%d]",
                 write_size);
      break;
    }

    write_size = file_.write((const void *)&record.crc_, CRC_SIZE);
    if (write_size != CRC_SIZE) {
      WLOG_ERROR("Wal write error. record.crc_ error write_size[%d]",
                 write_size);
      break;
    }

    write_size =
        file_.write((const void *)record.content_.data(), record.length_);
    if (write_size != (int)record.length_) {
      WLOG_ERROR("Wal write error. record.content_ error write_size[%d]",
                 write_size);
      break;
    }
    ret = 1;  // write one record success
    docs_count_++;
  } while (false);

  return ret;
}

Result<WalReadResult> LocalWalFile::next_checked() {
  using State = WalReadResult::State;
  if (!opened_) {
    return tl::make_unexpected(Status::FailedPrecondition("WAL is closed"));
  }
  if (truncated_at_) {
    return WalReadResult{State::kTruncatedTail, {}};
  }
  const auto offset = file_.offset();
  const auto size = file_.size();
  if (offset < 0 || static_cast<size_t>(offset) > size) {
    return tl::make_unexpected(Status::InternalError("Invalid WAL offset"));
  }
  const size_t remaining = size - static_cast<size_t>(offset);
  if (remaining == 0) {
    return WalReadResult{State::kEnd, {}};
  }
  auto truncated = [&]() {
    truncated_at_ = static_cast<size_t>(offset);
    return WalReadResult{State::kTruncatedTail, {}};
  };

  // The legacy header stores the native-endian length and CRC, in that order.
  uint32_t header[2];
  if (remaining < sizeof(header)) {
    return truncated();
  }
  if (file_.read(header, sizeof(header)) != sizeof(header)) {
    return tl::make_unexpected(Status::InternalError("Short WAL header read"));
  }
  const auto length = header[0];
  if (length == 0 || length > kMaxRecordSize) {
    return tl::make_unexpected(
        Status::InternalError("Invalid WAL record length"));
  }
  if (length > remaining - sizeof(header)) {
    return truncated();
  }
  std::string data(length, '\0');
  if (file_.read(data.data(), length) != length) {
    return tl::make_unexpected(Status::InternalError("Short WAL record read"));
  }
  if (ailego::Crc32c::Hash(data.data(), length, 0) != header[1]) {
    return tl::make_unexpected(
        Status::InternalError("WAL record checksum mismatch"));
  }
  return WalReadResult{State::kRecord, std::move(data)};
}

Status LocalWalFile::discard_truncated_tail() {
  if (!opened_ || read_only_ || !truncated_at_) {
    return Status::FailedPrecondition("No writable truncated WAL tail");
  }
  if (!file_.truncate(*truncated_at_) || !file_.flush() ||
      !file_.seek(0, ailego::File::Origin::End)) {
    return Status::InternalError("Cannot repair truncated WAL tail");
  }
  truncated_at_.reset();
  return Status::OK();
}

};  // namespace zvec
