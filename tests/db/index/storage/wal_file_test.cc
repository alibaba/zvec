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

#include "db/index/storage/wal/wal_file.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <stdio.h>
#include <limits>
#include <gtest/gtest.h>
#include <zvec/ailego/io/file.h>
#include <zvec/ailego/parallel/thread_pool.h>
#include <zvec/ailego/utility/string_helper.h>
#include <zvec/ailego/utility/time_helper.h>
#include "db/common/file_helper.h"
#include "tests/test_util.h"

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

using namespace zvec;
using SegmentID = uint32_t;

class WalFileTest : public testing::Test {
 protected:
  void SetUp() override {
    zvec::test_util::RemoveTestFiles("./data.wal.*");
  }

  void TearDown() override {}

  // Legacy success-path tests use empty string as their loop sentinel, but
  // still assert that the new API did not report an error.
  std::string ReadRecord(const WalFilePtr &wal_file) {
    auto result = wal_file->next();
    EXPECT_TRUE(result.has_value())
        << (result.has_value() ? "" : result.error().message());
    if (!result.has_value() || !result.value().has_value()) return {};
    return std::move(result.value().value());
  }
};

TEST_F(WalFileTest, TestGeneral) {
  std::string dir_path = "./";
  SegmentID segment_id = 0;
  std::string wal_file_path =
      FileHelper::MakeFilePath(dir_path, FileID::WAL_FILE, segment_id);
  WalFilePtr wal_file = WalFile::Create(wal_file_path);

  ASSERT_TRUE(wal_file != nullptr);

  WalOptions wal_option;
  wal_option.create_new = true;
  wal_option.max_docs_wal_flush = 0;
  int ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);

  // add 100 same record
  for (size_t i = 0; i < 100; i++) {
    ret = wal_file->append(std::string("hello"));
    ASSERT_EQ(ret, 0);
  }
  ret = wal_file->flush();
  ASSERT_EQ(ret, 0);
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);

  // add 100-200 record
  wal_option.create_new = false;
  wal_option.max_docs_wal_flush = 1;
  ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);
  for (size_t i = 100; i < 200; i++) {
    std::string record = "hello";
    ret = wal_file->append(record + std::to_string(i));
    ASSERT_EQ(ret, 0);
  }
  ret = wal_file->flush();
  ASSERT_EQ(ret, 0);
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);

  // reopen and add next 100 record
  wal_option.create_new = false;
  ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);
  for (size_t i = 200; i < 300; i++) {
    std::string record = "hello";
    ret = wal_file->append(record + std::to_string(i));
    ASSERT_EQ(ret, 0);
  }
  ret = wal_file->flush();
  ASSERT_EQ(ret, 0);
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);

  // reopen and add batch model 100 record
  wal_option.max_docs_wal_flush = 10;
  wal_option.create_new = false;
  ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);
  for (size_t i = 300; i < 400; i++) {
    std::string record = "hello";
    ret = wal_file->append(record + std::to_string(i));
    ASSERT_EQ(ret, 0);
  }
  ret = wal_file->flush();
  ASSERT_EQ(ret, 0);
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);

  // reopen for read
  wal_option.create_new = false;
  ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);

  uint32_t idx = 0;
  ret = wal_file->prepare_for_read();
  ASSERT_EQ(ret, 0);
  std::string record = ReadRecord(wal_file);
  while (!record.empty()) {
    if (idx < 100) {
      ASSERT_EQ(record, "hello");
    } else {
      ASSERT_EQ(record, std::string("hello") + std::to_string(idx));
    }
    record = ReadRecord(wal_file);
    idx++;
  }
  ASSERT_EQ(idx, 400);
  // close
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);
  // remove
  ret = wal_file->remove();
  ASSERT_EQ(ret, 0);
}

void do_append(WalFile *wal_file, size_t number) {
  std::string record = "hello" + std::to_string(number);
  int ret = wal_file->append(std::move(record));
  ASSERT_EQ(ret, 0);
}

TEST_F(WalFileTest, TestMultiThread) {
  std::string dir_path = "./";
  SegmentID segment_id = 0;
  std::string wal_file_path =
      FileHelper::MakeFilePath(dir_path, FileID::WAL_FILE, segment_id);
  WalFilePtr wal_file = WalFile::Create(wal_file_path);
  ASSERT_TRUE(wal_file != nullptr);

  WalOptions wal_option;
  wal_option.create_new = true;
  wal_option.max_docs_wal_flush = 1;
  int ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);

  ailego::ThreadPool pool(10, false);
  for (size_t i = 0; i < 10000; i++) {
    pool.execute(do_append, wal_file.get(), i);
  }
  pool.wait_finish();
  wal_file->flush();
  wal_file->close();

  // reopen for batch model
  wal_option.create_new = false;
  wal_option.max_docs_wal_flush = 1000;
  ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);
  for (size_t i = 10000; i < 20000; i++) {
    pool.execute(do_append, wal_file.get(), i);
  }
  pool.wait_finish();
  wal_file->flush();
  wal_file->close();

  // reopen for batch model
  wal_option.create_new = false;
  wal_option.max_docs_wal_flush = 0;
  ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);
  for (size_t i = 20000; i < 30000; i++) {
    pool.execute(do_append, wal_file.get(), i);
  }
  pool.wait_finish();
  wal_file->flush();
  wal_file->close();

  // reopen for read
  wal_option.create_new = false;
  ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);

  uint32_t idx = 0;
  ret = wal_file->prepare_for_read();
  ASSERT_EQ(ret, 0);
  std::string record = ReadRecord(wal_file);
  while (!record.empty()) {
    record = ReadRecord(wal_file);
    idx++;
  }
  ASSERT_EQ(idx, 30000);
  // close
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);
  // remove
  ret = wal_file->remove();
  ASSERT_EQ(ret, 0);
}


TEST_F(WalFileTest, TestBoundaryCondition) {
  // read empty file
  std::string dir_path = "./";
  SegmentID segment_id = 0;
  std::string wal_file_path =
      FileHelper::MakeFilePath(dir_path, FileID::WAL_FILE, segment_id);
  WalFilePtr wal_file = WalFile::Create(wal_file_path);
  ASSERT_TRUE(wal_file != nullptr);

  WalOptions wal_option;
  wal_option.create_new = true;
  wal_option.max_docs_wal_flush = 1;
  int ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);
  ret = wal_file->flush();
  ASSERT_EQ(ret, 0);
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);

  wal_option.create_new = false;
  ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);
  ASSERT_EQ(wal_file->prepare_for_read(), 0);
  uint32_t idx = 0;
  std::string record = ReadRecord(wal_file);
  while (!record.empty()) {
    record = ReadRecord(wal_file);
    idx++;
  }
  ASSERT_EQ(idx, 0);
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);

  // write and read binary struct
  std::vector<uint8_t> bin_v{0, 1, 2, 3};
  std::string str(bin_v.begin(), bin_v.end());
  wal_option.create_new = false;
  ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);
  ret = wal_file->append(std::move(str));
  ASSERT_EQ(ret, 0);
  ret = wal_file->flush();
  ASSERT_EQ(ret, 0);
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);

  wal_option.create_new = false;
  ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);
  idx = 0;
  ret = wal_file->prepare_for_read();
  ASSERT_EQ(ret, 0);
  record = ReadRecord(wal_file);
  while (!record.empty()) {
    ASSERT_EQ(record.size(), 4);
    for (size_t i = 0; i < 4; i++) {
      ASSERT_EQ(record[i], i);
    }
    record = ReadRecord(wal_file);
    idx++;
  }
  ASSERT_EQ(idx, 1);
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);
  // remove
  ret = wal_file->remove();
  ASSERT_EQ(ret, 0);


  // write very large record 4Mb
  size_t BIG_DATA_SIZE = 4 * 1024 * 1024;
  std::vector<uint8_t> big_data(BIG_DATA_SIZE);
  for (size_t i = 0; i < BIG_DATA_SIZE; i++) {
    big_data[i] = i % 256;
  }
  str.clear();
  str.assign((const char *)big_data.data(), BIG_DATA_SIZE);
  wal_option.create_new = true;
  ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);
  ret = wal_file->append(std::move(str));
  ASSERT_EQ(ret, 0);
  ret = wal_file->flush();
  ASSERT_EQ(ret, 0);
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);

  wal_option.create_new = false;
  ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);
  idx = 0;
  ret = wal_file->prepare_for_read();
  ASSERT_EQ(ret, 0);
  record = ReadRecord(wal_file);
  while (!record.empty()) {
    ASSERT_EQ(record.size(), BIG_DATA_SIZE);
    for (size_t i = 0; i < BIG_DATA_SIZE; i++) {
      ASSERT_EQ((uint8_t)record[i], i % 256);
    }
    record = ReadRecord(wal_file);
    idx++;
  }
  ASSERT_EQ(idx, 1);
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);
  // remove
  ret = wal_file->remove();
  ASSERT_EQ(ret, 0);

  // batch model 100, just add 99 record and close
  wal_option.max_docs_wal_flush = 100;
  wal_option.create_new = true;
  ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);
  for (size_t i = 0; i < 99; i++) {
    std::string record = "hello";
    ret = wal_file->append(record + std::to_string(i));
    ASSERT_EQ(ret, 0);
  }
  ret = wal_file->flush();
  ASSERT_EQ(ret, 0);
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);

  wal_option.create_new = false;
  ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);
  idx = 0;
  ret = wal_file->prepare_for_read();
  ASSERT_EQ(ret, 0);
  record = ReadRecord(wal_file);
  while (!record.empty()) {
    ASSERT_EQ(record, std::string("hello") + std::to_string(idx));
    record = ReadRecord(wal_file);
    idx++;
  }
  ASSERT_EQ(idx, 99);
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);
  // remove
  ret = wal_file->remove();
  ASSERT_EQ(ret, 0);
}

TEST_F(WalFileTest, TestNotExistErrorCase) {
  std::string dir_path = "./";
  SegmentID segment_id = 0;
  std::string wal_file_path =
      FileHelper::MakeFilePath(dir_path, FileID::WAL_FILE, segment_id);
  WalFilePtr wal_file = WalFile::Create(wal_file_path);
  // reopen for read
  WalOptions wal_option;
  wal_option.create_new = false;
  int ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, -1);
}


TEST_F(WalFileTest, TestFirstErrorCase) {
  std::string dir_path = "./";
  SegmentID segment_id = 0;
  std::string wal_file_path =
      FileHelper::MakeFilePath(dir_path, FileID::WAL_FILE, segment_id);
  WalFilePtr wal_file = WalFile::Create(wal_file_path);
  ASSERT_TRUE(wal_file != nullptr);

  WalOptions wal_option;
  wal_option.create_new = true;
  wal_option.max_docs_wal_flush = 1;
  int ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);

  // add 10 same record
  for (size_t i = 0; i < 10; i++) {
    ret = wal_file->append(std::string("hello"));
    ASSERT_EQ(ret, 0);
  }
  ret = wal_file->flush();
  ASSERT_EQ(ret, 0);
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);

  std::string wal_path = ailego::StringHelper::Concat(
      dir_path, "data.wal.", std::to_string(segment_id));
  int wal_fd = open(wal_path.c_str(), O_RDWR, 0644);
  ASSERT_GT(wal_fd, 0);
  // destory first record
  lseek(wal_fd, 64 + 8, SEEK_SET);
  // write err data
  char buf[6] = "nihao";
  write(wal_fd, buf, 5);
  close(wal_fd);

  // reopen for read
  wal_option.create_new = false;
  ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);

  ASSERT_EQ(wal_file->prepare_for_read(), 0);
  for (size_t i = 0; i < 0; ++i) {
    EXPECT_EQ(ReadRecord(wal_file), "hello");
  }
  auto result = wal_file->next();
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().code(), StatusCode::INTERNAL_ERROR);
  EXPECT_NE(result.error().message().find("CRC mismatch"), std::string::npos);
  EXPECT_NE(wal_file->append("after corruption"), 0);
  EXPECT_NE(wal_file->flush(), 0);
  // close
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);

  // remove
  ret = wal_file->remove();
  ASSERT_EQ(ret, 0);
}


TEST_F(WalFileTest, TestMiddleErrorCase) {
  std::string dir_path = "./";
  SegmentID segment_id = 0;
  std::string wal_file_path =
      FileHelper::MakeFilePath(dir_path, FileID::WAL_FILE, segment_id);
  WalFilePtr wal_file = WalFile::Create(wal_file_path);
  ASSERT_TRUE(wal_file != nullptr);

  WalOptions wal_option;
  wal_option.create_new = true;
  wal_option.max_docs_wal_flush = 1;
  int ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);

  // add 10 same record
  for (size_t i = 0; i < 10; i++) {
    ret = wal_file->append(std::string("hello"));
    ASSERT_EQ(ret, 0);
  }
  ret = wal_file->flush();
  ASSERT_EQ(ret, 0);
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);

  std::string wal_path = ailego::StringHelper::Concat(
      dir_path, "data.wal.", std::to_string(segment_id));
  int wal_fd = open(wal_path.c_str(), O_RDWR, 0644);
  ASSERT_GT(wal_fd, 0);
  // destory middle record
  lseek(wal_fd, 64 + 13 * 5 + 8, SEEK_SET);
  // write err data
  char buf[6] = "nihao";
  write(wal_fd, buf, 5);
  close(wal_fd);

  // reopen for read
  wal_option.create_new = false;
  ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);

  ASSERT_EQ(wal_file->prepare_for_read(), 0);
  for (size_t i = 0; i < 5; ++i) {
    EXPECT_EQ(ReadRecord(wal_file), "hello");
  }
  auto result = wal_file->next();
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().code(), StatusCode::INTERNAL_ERROR);
  EXPECT_NE(result.error().message().find("CRC mismatch"), std::string::npos);
  EXPECT_NE(wal_file->append("after corruption"), 0);
  // close
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);

  // remove
  ret = wal_file->remove();
  ASSERT_EQ(ret, 0);
}


TEST_F(WalFileTest, TestLastErrorCase) {
  std::string dir_path = "./";
  SegmentID segment_id = 0;
  std::string wal_file_path =
      FileHelper::MakeFilePath(dir_path, FileID::WAL_FILE, segment_id);
  WalFilePtr wal_file = WalFile::Create(wal_file_path);
  ASSERT_TRUE(wal_file != nullptr);

  WalOptions wal_option;
  wal_option.create_new = true;
  wal_option.max_docs_wal_flush = 1;
  int ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);

  // add 10 same record
  for (size_t i = 0; i < 10; i++) {
    ret = wal_file->append(std::string("hello"));
    ASSERT_EQ(ret, 0);
  }
  ret = wal_file->flush();
  ASSERT_EQ(ret, 0);
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);

  // destory last record
  std::string wal_path = ailego::StringHelper::Concat(
      dir_path, "data.wal.", std::to_string(segment_id));
  int wal_fd = open(wal_path.c_str(), O_RDWR, 0644);
  ASSERT_GT(wal_fd, 0);
  off_t fsize = lseek(wal_fd, 0, SEEK_END);
  close(wal_fd);
  truncate(wal_path.c_str(), (fsize - 4));

  // reopen for read
  wal_option.create_new = false;
  ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);

  uint32_t idx = 0;
  ret = wal_file->prepare_for_read();
  ASSERT_EQ(ret, 0);
  std::string record = ReadRecord(wal_file);
  while (!record.empty()) {
    ASSERT_EQ(record, "hello");
    record = ReadRecord(wal_file);
    idx++;
  }
  ASSERT_EQ(idx, 9);
  // close
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);

  // remove
  ret = wal_file->remove();
  ASSERT_EQ(ret, 0);
}


TEST_F(WalFileTest, TestLengthSmallErrorCase) {
  std::string dir_path = "./";
  SegmentID segment_id = 0;
  std::string wal_file_path =
      FileHelper::MakeFilePath(dir_path, FileID::WAL_FILE, segment_id);
  WalFilePtr wal_file = WalFile::Create(wal_file_path);
  ASSERT_TRUE(wal_file != nullptr);

  WalOptions wal_option;
  wal_option.create_new = true;
  wal_option.max_docs_wal_flush = 1;
  int ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);

  // add 10 same record
  for (size_t i = 0; i < 10; i++) {
    ret = wal_file->append(std::string("hello"));
    ASSERT_EQ(ret, 0);
  }
  ret = wal_file->flush();
  ASSERT_EQ(ret, 0);
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);

  // write error length
  std::string wal_path = ailego::StringHelper::Concat(
      dir_path, "data.wal.", std::to_string(segment_id));
  int wal_fd = open(wal_path.c_str(), O_RDWR, 0644);
  ASSERT_GT(wal_fd, 0);
  uint32_t err_length = 2;
  lseek(wal_fd, 64, SEEK_SET);
  write(wal_fd, (const void *)&err_length, 4);
  close(wal_fd);

  // reopen for read
  wal_option.create_new = false;
  ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);

  ASSERT_EQ(wal_file->prepare_for_read(), 0);
  for (size_t i = 0; i < 0; ++i) {
    EXPECT_EQ(ReadRecord(wal_file), "hello");
  }
  auto result = wal_file->next();
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().code(), StatusCode::INTERNAL_ERROR);
  EXPECT_NE(result.error().message().find("CRC mismatch"), std::string::npos);
  EXPECT_NE(wal_file->append("after corruption"), 0);
  // close
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);

  // remove
  ret = wal_file->remove();
  ASSERT_EQ(ret, 0);
}


TEST_F(WalFileTest, TestLengthBigErrorCase) {
  std::string dir_path = "./";
  SegmentID segment_id = 0;
  std::string wal_file_path =
      FileHelper::MakeFilePath(dir_path, FileID::WAL_FILE, segment_id);
  WalFilePtr wal_file = WalFile::Create(wal_file_path);
  ASSERT_TRUE(wal_file != nullptr);

  WalOptions wal_option;
  wal_option.create_new = true;
  wal_option.max_docs_wal_flush = 1;
  int ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);

  // add 10 same record
  for (size_t i = 0; i < 10; i++) {
    ret = wal_file->append(std::string("hello"));
    ASSERT_EQ(ret, 0);
  }
  ret = wal_file->flush();
  ASSERT_EQ(ret, 0);
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);

  // write error length
  std::string wal_path = ailego::StringHelper::Concat(
      dir_path, "data.wal.", std::to_string(segment_id));
  int wal_fd = open(wal_path.c_str(), O_RDWR, 0644);
  ASSERT_GT(wal_fd, 0);
  uint32_t err_length = std::numeric_limits<uint32_t>::max();

  lseek(wal_fd, 64, SEEK_SET);
  write(wal_fd, (const void *)&err_length, 4);
  close(wal_fd);

  // reopen for read
  wal_option.create_new = false;
  ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);

  uint32_t idx = 0;
  ret = wal_file->prepare_for_read();
  ASSERT_EQ(ret, 0);
  std::string record = ReadRecord(wal_file);
  while (!record.empty()) {
    ASSERT_EQ(record, "hello");
    record = ReadRecord(wal_file);
    idx++;
  }
  ASSERT_EQ(idx, 0);
  // close
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);

  // remove
  ret = wal_file->remove();
  ASSERT_EQ(ret, 0);
}


TEST_F(WalFileTest, TestCRCErrorCase) {
  std::string dir_path = "./";
  SegmentID segment_id = 0;
  std::string wal_file_path =
      FileHelper::MakeFilePath(dir_path, FileID::WAL_FILE, segment_id);
  WalFilePtr wal_file = WalFile::Create(wal_file_path);
  ASSERT_TRUE(wal_file != nullptr);

  WalOptions wal_option;
  wal_option.create_new = true;
  wal_option.max_docs_wal_flush = 1;
  int ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);

  // add 10 same record
  for (size_t i = 0; i < 10; i++) {
    ret = wal_file->append(std::string("hello"));
    ASSERT_EQ(ret, 0);
  }
  ret = wal_file->flush();
  ASSERT_EQ(ret, 0);
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);

  // write error crc
  std::string wal_path = ailego::StringHelper::Concat(
      dir_path, "data.wal.", std::to_string(segment_id));
  int wal_fd = open(wal_path.c_str(), O_RDWR, 0644);
  ASSERT_GT(wal_fd, 0);
  // second record crc 64+(4+4+len(hello))+4
  lseek(wal_fd, 64 + 17, SEEK_SET);
  uint32_t err_crc = 123;
  write(wal_fd, (const void *)&err_crc, 4);
  close(wal_fd);

  // reopen for read
  wal_option.create_new = false;
  ret = wal_file->open(wal_option);
  ASSERT_EQ(ret, 0);

  ASSERT_EQ(wal_file->prepare_for_read(), 0);
  for (size_t i = 0; i < 1; ++i) {
    EXPECT_EQ(ReadRecord(wal_file), "hello");
  }
  auto result = wal_file->next();
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().code(), StatusCode::INTERNAL_ERROR);
  EXPECT_NE(result.error().message().find("CRC mismatch"), std::string::npos);
  EXPECT_NE(wal_file->append("after corruption"), 0);
  // close
  ret = wal_file->close();
  ASSERT_EQ(ret, 0);
  // remove
  ret = wal_file->remove();
  ASSERT_EQ(ret, 0);
}

TEST_F(WalFileTest, RecordLargerThanFourMiBPreservesFollowingRecord) {
  const std::string path = "./data.wal.large";
  auto wal = WalFile::Create(path);
  WalOptions options;
  options.create_new = true;
  ASSERT_EQ(wal->open(options), 0);
  const std::string large(4 * 1024 * 1024 + 1024, 'x');
  ASSERT_EQ(wal->append("prefix"), 0);
  ASSERT_EQ(wal->append(std::string(large)), 0);
  ASSERT_EQ(wal->append("suffix"), 0);
  ASSERT_EQ(wal->close(), 0);
  options.create_new = false;
  ASSERT_EQ(wal->open(options), 0);
  ASSERT_EQ(wal->prepare_for_read(), 0);
  EXPECT_EQ(ReadRecord(wal), "prefix");
  EXPECT_EQ(ReadRecord(wal), large);
  EXPECT_EQ(ReadRecord(wal), "suffix");
  auto end = wal->next();
  ASSERT_TRUE(end.has_value());
  EXPECT_FALSE(end.value().has_value());
}

TEST_F(WalFileTest, IncompleteTailIsRemovedOnlyBeforeAppend) {
  const std::string path = "./data.wal.tail";
  constexpr size_t prefix_end = 64 + 8 + 6;
  // Exercise every partial header and partial payload boundary.
  for (size_t tail_size = 1; tail_size < 8 + 4; ++tail_size) {
    SCOPED_TRACE(tail_size);
    auto wal = WalFile::Create(path);
    WalOptions options;
    options.create_new = true;
    ASSERT_EQ(wal->open(options), 0);
    ASSERT_EQ(wal->append("prefix"), 0);
    ASSERT_EQ(wal->append("torn"), 0);
    ASSERT_EQ(wal->close(), 0);
    ailego::File file;
    ASSERT_TRUE(file.open(path, false));
    ASSERT_TRUE(file.truncate(prefix_end + tail_size));
    file.close();

    options.create_new = false;
    ASSERT_EQ(wal->open(options), 0);
    ASSERT_EQ(wal->prepare_for_read(), 0);
    EXPECT_EQ(ReadRecord(wal), "prefix");
    auto end = wal->next();
    ASSERT_TRUE(end.has_value());
    EXPECT_FALSE(end.value().has_value());
    ASSERT_TRUE(file.open(path, true));
    EXPECT_EQ(file.size(), prefix_end + tail_size);
    file.close();

    ASSERT_EQ(wal->append("suffix"), 0);
    ASSERT_EQ(wal->close(), 0);
    ASSERT_EQ(wal->open(options), 0);
    ASSERT_EQ(wal->prepare_for_read(), 0);
    EXPECT_EQ(ReadRecord(wal), "prefix");
    EXPECT_EQ(ReadRecord(wal), "suffix");
    end = wal->next();
    ASSERT_TRUE(end.has_value());
    EXPECT_FALSE(end.value().has_value());
    ASSERT_EQ(wal->remove(), 0);
  }
}

TEST_F(WalFileTest, ClosedReaderReturnsError) {
  auto wal = WalFile::Create("./data.wal.closed");
  auto result = wal->next();
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error().code(), StatusCode::INTERNAL_ERROR);
}

TEST_F(WalFileTest, ReadOnlyWalCannotBeModified) {
  const std::string path = "./data.wal.readonly";
  auto wal = WalFile::Create(path);
  WalOptions options;
  options.create_new = true;
  ASSERT_EQ(wal->open(options), 0);
  ASSERT_EQ(wal->append("prefix"), 0);
  ASSERT_EQ(wal->close(), 0);
  options.create_new = false;
  options.read_only = true;
  ASSERT_EQ(wal->open(options), 0);
  ASSERT_EQ(wal->prepare_for_read(), 0);
  EXPECT_EQ(ReadRecord(wal), "prefix");
  EXPECT_NE(wal->append("suffix"), 0);
  EXPECT_NE(wal->flush(), 0);
  EXPECT_NE(wal->remove(), 0);
  ASSERT_EQ(wal->close(), 0);
  EXPECT_NE(wal->remove(), 0);
  options.read_only = false;
  ASSERT_EQ(wal->open(options), 0);
  ASSERT_EQ(wal->prepare_for_read(), 0);
  EXPECT_EQ(ReadRecord(wal), "prefix");
  auto end = wal->next();
  ASSERT_TRUE(end.has_value());
  EXPECT_FALSE(end.value().has_value());
  ASSERT_EQ(wal->remove(), 0);
}

TEST_F(WalFileTest, AppendRequiresReadCursorToBePreparedAgain) {
  auto wal = WalFile::Create("./data.wal.cursor");
  WalOptions options;
  options.create_new = true;
  ASSERT_EQ(wal->open(options), 0);
  ASSERT_EQ(wal->append("first"), 0);
  ASSERT_EQ(wal->append("second"), 0);
  ASSERT_EQ(wal->prepare_for_read(), 0);
  EXPECT_EQ(ReadRecord(wal), "first");
  ASSERT_EQ(wal->append("third"), 0);
  EXPECT_FALSE(wal->next().has_value());
  ASSERT_EQ(wal->prepare_for_read(), 0);
  EXPECT_EQ(ReadRecord(wal), "first");
  EXPECT_EQ(ReadRecord(wal), "second");
  EXPECT_EQ(ReadRecord(wal), "third");
}

TEST_F(WalFileTest, RemoveReportsFilesystemFailure) {
  const std::string path = "./data.wal.remove";
  ASSERT_TRUE(FileHelper::CreateDirectory(path));
  auto wal = WalFile::Create(path);
  EXPECT_NE(wal->remove(), 0);
  ASSERT_TRUE(FileHelper::RemoveDirectory(path));
}

TEST_F(WalFileTest, ZeroLengthRecordIsCorruption) {
  const std::string path = "./data.wal.zero";
  auto wal = WalFile::Create(path);
  WalOptions options;
  options.create_new = true;
  ASSERT_EQ(wal->open(options), 0);
  EXPECT_NE(wal->append(""), 0);
  ASSERT_EQ(wal->append("payload"), 0);
  ASSERT_EQ(wal->close(), 0);
  ailego::File file;
  ASSERT_TRUE(file.open(path, false));
  const uint32_t length = 0;
  ASSERT_EQ(file.write(64, &length, sizeof(length)), sizeof(length));
  file.close();
  options.create_new = false;
  ASSERT_EQ(wal->open(options), 0);
  ASSERT_EQ(wal->prepare_for_read(), 0);
  auto result = wal->next();
  ASSERT_FALSE(result.has_value());
  EXPECT_NE(result.error().message().find("zero length"), std::string::npos);
}

TEST_F(WalFileTest, TruncatedFileHeaderReturnsError) {
  const std::string path = "./data.wal.header";
  auto wal = WalFile::Create(path);
  WalOptions options;
  options.create_new = true;
  ASSERT_EQ(wal->open(options), 0);
  ASSERT_EQ(wal->close(), 0);
  ailego::File file;
  ASSERT_TRUE(file.open(path, false));
  ASSERT_TRUE(file.truncate(63));
  file.close();
  options.create_new = false;
  ASSERT_EQ(wal->open(options), 0);
  EXPECT_NE(wal->prepare_for_read(), 0);
}

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif
