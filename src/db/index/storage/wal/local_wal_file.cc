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
#include <limits>
#include <new>
#include <stdexcept>
#ifndef _MSC_VER
#include <unistd.h>
#endif
#include <zvec/ailego/hash/crc32c.h>
#include <zvec/ailego/logger/logger.h>
#include "db/common/error_code.h"
#include "db/common/file_helper.h"
#include "db/common/typedef.h"

namespace zvec {

int LocalWalFile::append(std::string &&data) {
  if (data.empty() || data.size() > std::numeric_limits<uint32_t>::max()) {
    WLOG_ERROR("Wal record length is not representable: %zu", data.size());
    return -1;
  }

  WalRecord record;
  record.length_ = static_cast<uint32_t>(data.size());
  record.crc_ = ailego::Crc32c::Hash(data.data(), data.size(), 0);
  record.content_ = std::move(data);

  std::lock_guard<std::mutex> lock(file_mutex_);
  if (!opened_ || failed_ || read_only_) {
    return -1;
  }
  // Append changes the shared file position. Require a fresh read preparation
  // instead of silently skipping unread records after an intervening write.
  reader_ready_ = false;
  if (incomplete_tail_offset_) {
    if (!file_.truncate(*incomplete_tail_offset_)) {
      WLOG_ERROR("Wal incomplete tail truncation failed");
      failed_ = true;
      return -1;
    }
    incomplete_tail_offset_.reset();
  }
  if (!file_.seek(0, ailego::File::Origin::End)) {
    return -1;
  }
  if (write_record(record) < 0) {
    return -1;
  }
  // Keep the flush counter and flush in the same critical section as writes.
  if (max_docs_wal_flush_ != 0 && docs_count_ >= max_docs_wal_flush_) {
    if (!file_.flush()) {
      WLOG_ERROR("Wal flush error. docs_count_[%zu] max_docs_wal_flush_[%zu]",
                 (size_t)docs_count_, (size_t)max_docs_wal_flush_);
      failed_ = true;
      return -1;
    }
    docs_count_ = 0;
  }
  return 0;
}

Result<std::optional<std::string>> LocalWalFile::next() {
  std::lock_guard<std::mutex> lock(file_mutex_);
  if (!opened_ || failed_ || !reader_ready_) {
    return tl::make_unexpected(
        Status::InternalError("WAL is not open for reading or has failed"));
  }
  WalRecord record;
  auto result = read_record(record);
  if (!result.has_value()) {
    failed_ = true;
    return tl::make_unexpected(result.error());
  }
  if (!result.value()) {
    return std::nullopt;
  }
  const uint32_t crc =
      ailego::Crc32c::Hash(record.content_.data(), record.content_.size(), 0);
  if (crc != record.crc_) {
    failed_ = true;
    return tl::make_unexpected(
        Status::InternalError("WAL record CRC mismatch"));
  }
  return std::optional<std::string>(std::move(record.content_));
}

int LocalWalFile::open(const WalOptions &wal_option) {
  std::lock_guard<std::mutex> lock(file_mutex_);
  CHECK_STATUS(opened_, false);
  if (wal_option.create_new && wal_option.read_only) return -1;
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
    header_ = WalHeader{};
    size_t write_size = file_.write((const void *)&header_, sizeof(header_));
    if (write_size != sizeof(header_)) {
      WLOG_ERROR("Wal write header error. create_new[%d]",
                 wal_option.create_new);
      file_.close();
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

    // open default for write
    if (!file_.seek(0, ailego::File::Origin::End)) {
      file_.close();
      return -1;
    }
  }

  max_docs_wal_flush_ = wal_option.max_docs_wal_flush;
  opened_ = true;
  read_only_ = wal_option.read_only;
  reader_ready_ = false;
  failed_ = false;
  incomplete_tail_offset_.reset();
  docs_count_ = 0;

  WLOG_INFO("Wal open success. create_new[%d]", wal_option.create_new);
  return 0;
}

int LocalWalFile::close() {
  std::lock_guard<std::mutex> lock(file_mutex_);
  CHECK_STATUS(opened_, true);
  file_.close();
  WLOG_INFO("Wal close success");
  opened_ = false;
  reader_ready_ = false;
  return 0;
}

int LocalWalFile::remove() {
  std::lock_guard<std::mutex> lock(file_mutex_);
  if (read_only_) return -1;
  if (opened_) {
    file_.close();
    opened_ = false;
    reader_ready_ = false;
  }
  if (FileHelper::FileExists(wal_path_)) {
    if (!FileHelper::RemoveFile(wal_path_)) {
      WLOG_ERROR("Wal remove failed");
      return -1;
    }
    WLOG_INFO("Wal remove success.");
  }
  return 0;
}

int LocalWalFile::flush() {
  std::lock_guard<std::mutex> lock(file_mutex_);
  CHECK_STATUS(opened_, true);
  if (failed_ || read_only_) return -1;
  if (!file_.flush()) {
    WLOG_ERROR("Wal flush error.");
    failed_ = true;
    return -1;
  }
  docs_count_ = 0;
  return 0;
}

int LocalWalFile::prepare_for_read() {
  std::lock_guard<std::mutex> lock(file_mutex_);
  CHECK_STATUS(opened_, true);
  reader_ready_ = false;
  incomplete_tail_offset_.reset();
  if (failed_ || !file_.seek(0, ailego::File::Origin::Begin)) {
    return -1;
  }
  size_t read_size = file_.read((void *)&header_, sizeof(header_));
  if (read_size != sizeof(header_)) {
    WLOG_ERROR("Wal read header error.");
    failed_ = true;
    return -1;
  }
  if (header_.wal_version != 0UL) {
    WLOG_ERROR("Wal version not support error.");
    failed_ = true;
    return -1;
  }
  reader_ready_ = true;
  return 0;
}

// Caller holds file_mutex_. A failed write must not strand future successful
// appends behind its incomplete record.
int LocalWalFile::write_record(WalRecord &record) {
  const auto start = file_.offset();
  if (start < static_cast<ssize_t>(sizeof(header_))) {
    failed_ = true;
    return -1;
  }
  if (file_.write(&record.length_, LENGTH_SIZE) != LENGTH_SIZE ||
      file_.write(&record.crc_, CRC_SIZE) != CRC_SIZE ||
      file_.write(record.content_.data(), record.content_.size()) !=
          record.content_.size()) {
    WLOG_ERROR("Wal write record failed. record.length_[%zu]",
               record.content_.size());
    if (!file_.truncate(static_cast<size_t>(start)) ||
        !file_.seek(start, ailego::File::Origin::Begin)) {
      failed_ = true;
    }
    return -1;
  }
  ++docs_count_;
  return 1;
}

Result<bool> LocalWalFile::read_record(WalRecord &record) {
  if (incomplete_tail_offset_) {
    return false;
  }
  // File::read reports bytes read for both EOF and I/O failures. Check the
  // physical extent first: a short read within that extent is an I/O error,
  // whereas a final frame that does not fit is a tolerated interrupted write.
  const auto start = file_.offset();
  const size_t file_size = file_.size();
  if (!file_.is_valid() || start < static_cast<ssize_t>(sizeof(header_)) ||
      file_size < sizeof(header_) || static_cast<size_t>(start) > file_size) {
    return tl::make_unexpected(
        Status::InternalError("Failed to determine WAL read position or size"));
  }
  const size_t remaining = file_size - static_cast<size_t>(start);
  if (remaining == 0) {
    return false;
  }
  if (remaining < LENGTH_SIZE + CRC_SIZE) {
    incomplete_tail_offset_ = static_cast<size_t>(start);
    return false;
  }
  if (file_.read(&record.length_, LENGTH_SIZE) != LENGTH_SIZE ||
      file_.read(&record.crc_, CRC_SIZE) != CRC_SIZE) {
    return tl::make_unexpected(
        Status::InternalError("Failed to read WAL record header"));
  }
  if (record.length_ == 0) {
    return tl::make_unexpected(
        Status::InternalError("WAL record has zero length"));
  }
  if (record.length_ > remaining - LENGTH_SIZE - CRC_SIZE) {
    incomplete_tail_offset_ = static_cast<size_t>(start);
    return false;
  }
  try {
    record.content_.resize(record.length_);
  } catch (const std::bad_alloc &) {
    return tl::make_unexpected(Status(StatusCode::RESOURCE_EXHAUSTED,
                                      "Unable to allocate WAL record buffer"));
  } catch (const std::length_error &) {
    return tl::make_unexpected(Status(StatusCode::RESOURCE_EXHAUSTED,
                                      "WAL record exceeds string capacity"));
  }
  if (file_.read(record.content_.data(), record.content_.size()) !=
      record.content_.size()) {
    return tl::make_unexpected(
        Status::InternalError("Failed to read WAL record payload"));
  }
  return true;
}

}  // namespace zvec
