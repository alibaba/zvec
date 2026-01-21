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

#include <ailego/utility/memory_helper.h>
#include <gtest/gtest.h>
#include <zvec/ailego/io/file.h>

using namespace zvec::ailego;

TEST(File, General) {
  EXPECT_TRUE(File::IsDirectory("."));
  EXPECT_TRUE(File::IsDirectory(".."));
  EXPECT_TRUE(File::IsDirectory("../"));
  EXPECT_TRUE(File::IsDirectory("..//"));
  EXPECT_TRUE(File::IsDirectory("..//"));

  EXPECT_FALSE(File::IsSymbolicLink("."));
  EXPECT_FALSE(File::IsSymbolicLink(".."));
  EXPECT_FALSE(File::IsSymbolicLink("../"));
  EXPECT_FALSE(File::IsSymbolicLink("..//"));
  EXPECT_FALSE(File::IsSymbolicLink("..//"));

  EXPECT_FALSE(File::IsRegular("."));
  EXPECT_FALSE(File::IsRegular(".."));
  EXPECT_FALSE(File::IsRegular("../"));
  EXPECT_FALSE(File::IsRegular("..//"));
  EXPECT_FALSE(File::IsRegular("..//"));

  EXPECT_TRUE(File::IsExist("."));
  EXPECT_TRUE(File::IsExist(".."));
  EXPECT_TRUE(File::IsExist("../"));
  EXPECT_TRUE(File::IsExist("..//"));
  EXPECT_TRUE(File::IsExist("..//"));
}

TEST(File, MakePath) {
  EXPECT_TRUE(File::MakePath(""));
  EXPECT_TRUE(File::MakePath("."));
  EXPECT_TRUE(File::MakePath(".."));
  EXPECT_TRUE(File::MakePath("../"));
  EXPECT_TRUE(File::MakePath("..//"));
  EXPECT_TRUE(File::MakePath("..//"));
  EXPECT_TRUE(File::MakePath("/"));

  EXPECT_TRUE(File::MakePath("./file_test_makepath"));
  EXPECT_TRUE(File::MakePath("file_test_makepath"));
  EXPECT_TRUE(File::MakePath("file_test_makepath/1/2/3/"));
  EXPECT_TRUE(File::MakePath("file_test_makepath/1/2/3"));
}

bool TouchFile(const char *path) {
  std::string buf(path);
  char *sp = (char *)strrchr(buf.data(), '/');
  *sp = '\0';
  File::MakePath(buf.data());
  *sp = '/';
  File file;
  return file.create(path, 0);
}

TEST(File, RemoveDirectory) {
  EXPECT_TRUE(File::MakePath("file_test_rmdir/1/2/3"));
  EXPECT_TRUE(File::MakePath("file_test_rmdir/a/b/c/d"));
  EXPECT_TRUE(File::MakePath("file_test_rmdir/1/a/b/c/d"));
  EXPECT_TRUE(File::MakePath("file_test_rmdir/1/2/a/b/c/d"));
  EXPECT_TRUE(File::MakePath("file_test_rmdir/1/2/3/a/b/c/d"));
  EXPECT_TRUE(File::MakePath("file_test_rmdir/a/1/2/3"));
  EXPECT_TRUE(File::MakePath("file_test_rmdir/a/b/1/2/3"));
  EXPECT_TRUE(File::MakePath("file_test_rmdir/a/b/c/1/2/3"));

  EXPECT_TRUE(TouchFile("file_test_rmdir/a/b/c/1/2/3/A"));
  EXPECT_TRUE(TouchFile("file_test_rmdir/a/b/c/1/2/3/B"));
  EXPECT_TRUE(TouchFile("file_test_rmdir/C"));
  EXPECT_TRUE(TouchFile("file_test_rmdir/D"));
  EXPECT_TRUE(TouchFile("file_test_rmdir/1/2/3/E"));
  EXPECT_TRUE(TouchFile("file_test_rmdir/a/b/c/d/F"));
  EXPECT_TRUE(TouchFile("file_test_rmdir/1/a/b/c/d/G"));
  EXPECT_TRUE(TouchFile("file_test_rmdir/1/2/a/b/c/d/H"));
  EXPECT_TRUE(TouchFile("file_test_rmdir/1/2/3/a/b/c/d/I"));
  EXPECT_TRUE(TouchFile("file_test_rmdir/a/1/2/3/J"));
  EXPECT_TRUE(TouchFile("file_test_rmdir/a/b/1/2/3/K"));
  EXPECT_TRUE(TouchFile("file_test_rmdir/1/2/3/M"));
  EXPECT_TRUE(TouchFile("file_test_rmdir/1/2/a/b/c/d/N"));

  EXPECT_FALSE(File::RemoveDirectory("file_test_rmdir/1/2/a/b/c/d/N"));
  EXPECT_FALSE(File::RemoveDirectory("file_test_rmdir/1/2/3/a/b/c/d/I"));
  EXPECT_FALSE(File::RemoveDirectory("file_test_rmdir/C"));
  EXPECT_FALSE(File::RemoveDirectory("file_test_rmdir/D"));

  EXPECT_TRUE(File::IsDirectory("file_test_rmdir/"));
  EXPECT_TRUE(File::IsDirectory("file_test_makepath/"));
  EXPECT_TRUE(File::RemoveDirectory("file_test_rmdir/"));
  EXPECT_TRUE(File::RemoveDirectory("file_test_makepath"));
}

TEST(File, RemovePath) {
  EXPECT_TRUE(File::MakePath("file_test_rmpath/1/2/3"));
  EXPECT_TRUE(File::MakePath("file_test_rmpath/a/b/c/d"));
  EXPECT_TRUE(File::MakePath("file_test_rmpath/1/a/b/c/d"));
  EXPECT_TRUE(File::MakePath("file_test_rmpath/1/2/a/b/c/d"));
  EXPECT_TRUE(File::MakePath("file_test_rmpath/1/2/3/a/b/c/d"));
  EXPECT_TRUE(File::MakePath("file_test_rmpath/a/1/2/3"));
  EXPECT_TRUE(File::MakePath("file_test_rmpath/a/b/1/2/3"));
  EXPECT_TRUE(File::MakePath("file_test_rmpath/a/b/c/1/2/3"));

  EXPECT_TRUE(TouchFile("file_test_rmpath/a/b/c/1/2/3/A"));
  EXPECT_TRUE(TouchFile("file_test_rmpath/a/b/c/1/2/3/B"));
  EXPECT_TRUE(TouchFile("file_test_rmpath/C"));
  EXPECT_TRUE(TouchFile("file_test_rmpath/D"));
  EXPECT_TRUE(TouchFile("file_test_rmpath/1/2/3/E"));
  EXPECT_TRUE(TouchFile("file_test_rmpath/a/b/c/d/F"));
  EXPECT_TRUE(TouchFile("file_test_rmpath/1/a/b/c/d/G"));
  EXPECT_TRUE(TouchFile("file_test_rmpath/1/2/a/b/c/d/H"));
  EXPECT_TRUE(TouchFile("file_test_rmpath/1/2/3/a/b/c/d/I"));
  EXPECT_TRUE(TouchFile("file_test_rmpath/a/1/2/3/J"));
  EXPECT_TRUE(TouchFile("file_test_rmpath/a/b/1/2/3/K"));
  EXPECT_TRUE(TouchFile("file_test_rmpath/1/2/3/M"));
  EXPECT_TRUE(TouchFile("file_test_rmpath/1/2/a/b/c/d/N"));
  EXPECT_TRUE(File::IsExist("file_test_rmpath/1/2/a/b/c/d/N"));

  EXPECT_TRUE(File::IsDirectory("file_test_rmpath/"));
  EXPECT_TRUE(File::RemovePath("file_test_rmpath/"));

  EXPECT_TRUE(File::MakePath("file_test_rmpath/AAA"));
  EXPECT_TRUE(File::MakePath("file_test_rmpath/BBB"));
  EXPECT_TRUE(TouchFile("file_test_rmpath/CCC"));
  EXPECT_TRUE(TouchFile("file_test_rmpath/DDD"));
  EXPECT_TRUE(File::IsExist("file_test_rmpath/BBB"));

  EXPECT_FALSE(File::RemovePath("file_test_rmpath/CCC/"));
  EXPECT_FALSE(File::RemovePath("file_test_rmpath/DDD/"));
  EXPECT_TRUE(File::RemovePath("file_test_rmpath/CCC"));
  EXPECT_TRUE(File::RemovePath("file_test_rmpath/DDD"));
  EXPECT_TRUE(File::RemovePath("file_test_rmpath"));
}

TEST(File, CreateAndOpen) {
  const char *file_path = "file_create_testing.tmp";
  size_t file_size = 12 * 1022 * 1021;

  File::Delete(file_path);
  EXPECT_FALSE(File::IsRegular(file_path));

  {
    File file;
    EXPECT_FALSE(file.is_valid());
    EXPECT_TRUE(file.create(file_path, file_size, true));
    EXPECT_TRUE(file.is_valid());
    EXPECT_TRUE(File::IsRegular(file_path));
    EXPECT_EQ(file_size, file.size());
  }
  // create again with exist file
  {
    File file;
    EXPECT_FALSE(file.is_valid());
    EXPECT_TRUE(file.create(file_path, file_size / 10));
    EXPECT_TRUE(file.is_valid());
    EXPECT_FALSE(file.read_only());
    EXPECT_EQ(file_size / 10, file.size());
  }

  {
    File file;
    EXPECT_FALSE(file.is_valid());
    EXPECT_TRUE(file.create(file_path, file_size * 3, true));
    EXPECT_TRUE(file.is_valid());
    EXPECT_FALSE(file.read_only());
    EXPECT_EQ(file_size * 3, file.size());
  }

  {
    File file;
    EXPECT_TRUE(file.open(file_path, true, true));
    EXPECT_TRUE(file.is_valid());
    EXPECT_TRUE(file.read_only());
    EXPECT_EQ(file_size * 3, file.size());
  }

  {
    File file;
    EXPECT_TRUE(file.open(file_path, false, true));
    EXPECT_TRUE(file.is_valid());
    EXPECT_FALSE(file.read_only());
    EXPECT_EQ(file_size * 3, file.size());
  }
  File::Delete(file_path);
}

TEST(File, ReadAndWrite) {
  const char *file_path = "file_read_testing.tmp";
  size_t file_size = 2u * 1024u * 1024u + 12u * 1024;

  File::Delete(file_path);
  EXPECT_FALSE(File::IsRegular(file_path));

  File file;
  EXPECT_FALSE(file.is_valid());
  EXPECT_TRUE(file.create(file_path, file_size));
  EXPECT_TRUE(File::IsRegular(file_path));

  EXPECT_TRUE(file.is_valid());
  EXPECT_EQ(0, file.offset());
  EXPECT_EQ(file_size, file.size());

  std::string buf;
  buf.resize(file_size, 0x55);
  ASSERT_EQ(file_size, buf.size());
  EXPECT_EQ(file_size, file.write(buf.data(), buf.size()));
  EXPECT_EQ(file_size, file.size());
  EXPECT_EQ((ssize_t)buf.size(), file.offset());
  EXPECT_TRUE(file.flush());

  buf.clear();
  buf.resize(file_size);
  file.reset();
  EXPECT_EQ(file_size, file.read((void *)buf.data(), buf.size()));

  File::Delete(file_path);
}

TEST(File, MemoryMap) {
  const char *file_path = "file_map_testing.tmp";
  size_t file_size = 2u * 1024u * 1024u + 12u * 1024;
  size_t map_offset = MemoryHelper::PageSize() * 16;
  size_t map_size = file_size - MemoryHelper::PageSize();

  File::Delete(file_path);
  EXPECT_FALSE(File::IsRegular(file_path));

  File file;
  EXPECT_FALSE(file.is_valid());
  EXPECT_TRUE(file.create(file_path, file_size));
  EXPECT_TRUE(File::IsRegular(file_path));
  EXPECT_EQ(file_size, file.size());

  void *addr = file.map(map_offset, map_size, 0);
  EXPECT_TRUE(addr != nullptr);
  EXPECT_TRUE(File::MemoryFlush(addr, map_size));
  File::MemoryUnmap(addr, map_size);
  file.close();

  EXPECT_TRUE(file.open(file_path, true));
  EXPECT_EQ(file_size, file.size());
  addr = file.map(map_offset, map_size, 0);
  EXPECT_TRUE(addr != nullptr);
  EXPECT_TRUE(File::MemoryFlush(addr, map_size));
  File::MemoryUnmap(addr, map_size);

  // void *addr1 = file.map(map_offset, map_size, 0);
  // void *addr2 = file.map(map_offset, map_size, 0);
  // EXPECT_EQ(addr1, addr2);
  file.close();

  EXPECT_TRUE(file.open(file_path, true));
  EXPECT_EQ(file_size, file.size());
  addr = file.map(map_offset, map_size, File::MMAP_SHARED);
  EXPECT_TRUE(addr != nullptr);
  EXPECT_TRUE(File::MemoryFlush(addr, map_size));

#if defined(__linux) || defined(__linux__) || defined(__NetBSD__)
  EXPECT_TRUE(File::MemoryRemap(addr, map_size, addr, map_size * 2));
  addr = File::MemoryRemap(addr, map_size, nullptr, map_size * 3);
  EXPECT_TRUE(addr);
#endif

  File::MemoryUnmap(addr, map_size);
  file.close();

#if !defined(_WIN32)
  addr = File::MemoryMap(map_size, 0);
  EXPECT_TRUE(addr != nullptr);
  File::MemoryUnmap(addr, map_size);

  addr = File::MemoryMap(map_size, File::MMAP_SHARED);
  EXPECT_TRUE(addr != nullptr);
  File::MemoryUnmap(addr, map_size);
#endif
}

TEST(File, Append) {
  const char *file_path = "file_append_testing.tmp";
  File file;
  EXPECT_FALSE(file.is_valid());
  EXPECT_TRUE(file.create(file_path, MemoryHelper::PageSize()));
  EXPECT_TRUE(File::IsRegular(file_path));

  std::string padding;
  padding.resize(MemoryHelper::PageSize());
  for (size_t i = 0; i < 10; ++i) {
    EXPECT_EQ(padding.size(),
              file.write(file.size(), padding.data(), padding.size()));
  }
  EXPECT_EQ(padding.size() * 11, file.size());

  file.truncate(padding.size() * 7);
  EXPECT_EQ(padding.size() * 7, file.size());

  file.truncate(padding.size() * 16);
  EXPECT_EQ(padding.size() * 16, file.size());
  file.close();
}

TEST(File, Seek) {
  const char *file_path = "file_seek_testing.tmp";
  File file;
  EXPECT_FALSE(file.is_valid());
  EXPECT_TRUE(file.create(file_path, 0));
  EXPECT_TRUE(File::IsRegular(file_path));

  std::string padding;
  padding.resize(MemoryHelper::PageSize());
  for (size_t i = 0; i < 10; ++i) {
    EXPECT_EQ(padding.size(), file.write(padding.data(), padding.size()));
  }
  EXPECT_EQ(padding.size() * 10, (size_t)file.size());
  EXPECT_EQ(padding.size() * 10, (size_t)file.offset());

  EXPECT_TRUE(file.seek(0, File::Origin::Begin));
  EXPECT_EQ(0, file.offset());

  EXPECT_TRUE(file.seek(-20, File::Origin::End));
  EXPECT_EQ((ssize_t)file.size() - 20, file.offset());

  EXPECT_TRUE(file.seek(20, File::Origin::Current));
  EXPECT_EQ((ssize_t)file.size(), file.offset());
  file.close();
}
