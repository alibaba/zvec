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

#include <gtest/gtest.h>
#include <zvec/ailego/io/file.h>
#include <zvec/ailego/io/mmap_file.h>

using namespace zvec::ailego;

TEST(MMapFile, Create) {
  const char *file_path = "mmap_file_create_testing.tmp";
  size_t file_size = 12 * 1022 * 1021;

  File::Delete(file_path);
  EXPECT_FALSE(File::IsRegular(file_path));

  {
    MMapFile file;
    EXPECT_EQ(0u, file.size());
    EXPECT_EQ(0u, file.offset());
    EXPECT_FALSE(file.is_valid());
    EXPECT_TRUE(file.create(file_path, file_size));
    EXPECT_TRUE(file.is_valid());
    EXPECT_TRUE(File::IsRegular(file_path));

    memset(file.region(), 0xff, file.size());
    file.close();
    file.warmup();
    EXPECT_FALSE(file.lock());
    EXPECT_FALSE(file.unlock());
  }
  // create again with exist file
  {
    MMapFile file;
    EXPECT_FALSE(file.is_valid());
    EXPECT_TRUE(file.create(file_path, file_size));
    EXPECT_TRUE(file.is_valid());
    EXPECT_FALSE(file.read_only());
    memset(file.region(), 0xff, file.size());
  }
  File::Delete(file_path);
}

TEST(MMapFile, Open) {
  const char *file_path = "mmap_file_open_testing.tmp";
  const char *file_path2 = "mmap_file_open_testing2.tmp";
  size_t file_size = 23 * 1022 * 1021;
  std::string raw_data;

  File::Delete(file_path);
  raw_data.resize(file_size, 0x74);

  // create a file
  {
    MMapFile file;
    EXPECT_TRUE(file.create(file_path, file_size));
    EXPECT_EQ(file_size, file.size());
    EXPECT_EQ(0u, file.offset());
    EXPECT_TRUE(File::IsRegular(file_path));
    file.warmup();
    file.lock();

    MMapFile file2 = std::move(file);
    memset(file2.region(), 0x74, file2.size());
    EXPECT_EQ(0, memcmp(file2.region(), raw_data.data(), raw_data.size()));
    file.flush();
    file2.lock();
  }

  File::Delete(file_path2);
  ASSERT_TRUE(File::Rename(file_path, file_path2));

  // open a file
  {
    MMapFile file;
    EXPECT_TRUE(File::IsRegular(file_path2));
    EXPECT_TRUE(file.open(file_path2, true));
    EXPECT_TRUE(file.read_only());
    EXPECT_EQ(0, memcmp(file.region(), raw_data.data(), raw_data.size()));
    file.lock();
  }
  {
    MMapFile file;
    MMapFile file2 = std::move(file);
    EXPECT_TRUE(file2.open(file_path2, false));

    EXPECT_FALSE(file.lock());
    EXPECT_FALSE(file.unlock());
    file2.warmup();
    file2.lock();
    file2.unlock();
  }
  // clean up
  File::Delete(file_path2);
}

TEST(MMapFile, ReadAndWrite) {
  const char *file_path = "mmap_file_read_testing.tmp";
  size_t file_size = 11 * 1022 * 1021;

  File::Delete(file_path);
  EXPECT_FALSE(File::IsRegular(file_path));

  MMapFile file;
  EXPECT_EQ(0u, file.size());
  EXPECT_EQ(0u, file.offset());
  EXPECT_FALSE(file.is_valid());
  EXPECT_TRUE(file.create(file_path, file_size));
  EXPECT_EQ(file_size, file.size());
  EXPECT_TRUE(file.is_valid());
  EXPECT_TRUE(File::IsRegular(file_path));

  char buf[] = "abcdefghijklmnopqrstuvwxyz";
  EXPECT_LT(sizeof(buf), file.size());
  EXPECT_EQ(sizeof(buf), file.write(buf, sizeof(buf)));
  EXPECT_EQ(0u, file.write(file_size + 2, buf, sizeof(buf)));

  std::string str;
  str.resize(sizeof(buf) - 1);
  EXPECT_EQ(str.size(), file.read(0, (uint8_t *)str.data(), str.size()));
  EXPECT_EQ(str, std::string(buf));

  EXPECT_EQ(11u, file.write(file_size - 11u, buf, sizeof(buf)));
  const void *p1 = nullptr;
  EXPECT_EQ(11u, file.read(file_size - 11u, &p1, sizeof(buf)));
  EXPECT_TRUE(!!p1);
  EXPECT_EQ(std::string((char *)p1, 11u), std::string(buf, 11u));

  EXPECT_EQ(sizeof(buf), file.offset());
  file.reset();
  EXPECT_EQ(0u, file.offset());

  std::string str2;
  str2.resize(sizeof(buf) - 1);
  EXPECT_EQ(str2.size(), file.read((uint8_t *)str2.data(), str2.size()));
  EXPECT_EQ(str, std::string(buf));

  const void *p2 = nullptr;
  file.reset();
  EXPECT_EQ(0u, file.read(file_size + 11u, &p2, sizeof(buf)));
  const void *p3 = nullptr;
  EXPECT_EQ(sizeof(buf), file.read(&p3, sizeof(buf)));
  EXPECT_EQ(std::string((char *)p3), std::string(buf));

  char dest[64];
  EXPECT_EQ(11u, file.read(file_size - 11u, dest, sizeof(dest)));
  EXPECT_EQ(std::string(dest, 11u), std::string(buf, 11u));

  File::Delete(file_path);
}
