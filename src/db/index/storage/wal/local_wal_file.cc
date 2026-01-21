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
#include <unistd.h>
#include <zvec/ailego/hash/crc32c.h>
#include <zvec/ailego/logger/logger.h>
#include "db/common/error_code.h"
#include "db/common/file_helper.h"
#include "db/common/typedef.h"

#define MAX_RECORD_SIZE 4194304  // 4Mb

namespace zvec {

int LocalWalFile::append(std::string &&data) {
  WalRecord record;
  record.length_ = data.size();
  record.crc_ = ailego::Crc32c::Hash(
      reinterpret_cast<const void *>(data.data()), record.length_, 0);
  record.content_ = std::forward<std::string>(data);

  if (write_record(record) < 0) {
    WLOG_ERROR("Wal write record error. record.length_[%zu]",
               (size_t)record.length_);
    return -1;
  }
  // if max_docs_wal_flush_ is 0, no need flush
  if (max_docs_wal_flush_ != 0 && docs_count_ >= max_docs_wal_flush_) {
    if (!file_.flush()) {
      WLOG_ERROR("Wal flush error. docs_count_[%zu] max_docs_wal_flush_[%zu]",
                 (size_t)docs_count_, (size_t)max_docs_wal_flush_);
    }
    docs_count_ = 0;
  }
  return 0;
}

std::string LocalWalFile::next() {
  WalRecord record;
  if (read_record(record) > 0) {
    uint32_t tmp_crc = ailego::Crc32c::Hash(
        reinterpret_cast<const void *>(record.content_.data()), record.length_,
        0);
    if (tmp_crc == record.crc_) {
      return std::move(record.content_);
    } else {
      WLOG_ERROR(
          "Wal next error. record.length_[%zu] crc_[%zu] != tmp_crc[%zu]",
          (size_t)record.length_, (size_t)record.crc_, (size_t)tmp_crc);
    }
  }
  // end of file or read error
  return std::string();
}

int LocalWalFile::open(const WalOptions &wal_option) {
  CHECK_STATUS(opened_, false);
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

    if (!file_.open(wal_path_.c_str(), false)) {
      WLOG_ERROR("Wal open error. create_new[%d]", wal_option.create_new);
      return -1;
    }

    // open default for write
    file_.seek(0, ailego::File::Origin::End);
  }

  max_docs_wal_flush_ = wal_option.max_docs_wal_flush;
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
  if (!file_.flush()) {
    WLOG_ERROR("Wal flush error.");
    return -1;
  }
  return 0;
}

int LocalWalFile::prepare_for_read() {
  CHECK_STATUS(opened_, true);
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

  std::lock_guard<std::mutex> lock(file_mutex_);
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

//! Return 1 if success or 0 if eof or -1 if read error
int LocalWalFile::read_record(WalRecord &record) {
  CHECK_STATUS(opened_, true);

  int read_size = 0;
  std::string err_msg;
  int ret = -1;

  do {
    read_size =
        file_.read(reinterpret_cast<void *>(&record.length_), LENGTH_SIZE);
    if (read_size == 0) {
      ret = 0;
      WLOG_INFO("Wal read finished. end of file");
      break;
    }

    if (read_size != LENGTH_SIZE) {
      WLOG_ERROR("Wal read error. record.length_ error read_size[%d]",
                 read_size);
      break;
    }

    read_size = file_.read(reinterpret_cast<void *>(&record.crc_), CRC_SIZE);
    if (read_size != CRC_SIZE) {
      WLOG_ERROR("Wal read error. record.crc_ error read_size[%d]", read_size);
      break;
    }

    // resize may crash if record.length_ very large
    if (record.length_ <= 0 || record.length_ > MAX_RECORD_SIZE) {
      WLOG_ERROR("Wal read error. record.length_ value error read_size[%d]",
                 read_size);
      break;
    }

    record.content_.resize(record.length_);
    read_size = file_.read((void *)const_cast<char *>(record.content_.data()),
                           record.length_);
    if (read_size != (int)record.length_) {
      WLOG_ERROR("Wal read error. record.content_ error read_size[%d]",
                 read_size);
      break;
    }
    ret = 1;  // read one record success
  } while (false);

  return ret;
}

};  // namespace zvec