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

#if defined(_WIN32) || defined(_WIN64)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#undef DeleteFile
#undef RemoveDirectory
#undef CreateFile
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <gtest/gtest.h>
#include <zvec/ailego/io/file.h>
#include <zvec/ailego/utility/file_helper.h>
#include <zvec/ailego/utility/string_helper.h>

using namespace zvec::ailego;

// UTF-8 encoded Chinese / Japanese  test strings.
// These contain characters outside the Windows ANSI codepage, so any path
// handling that falls back to narrow Win32 APIs will produce the mojibake.
static const std::string kChinese =
    "\xe4\xb8\xad\xe6\x96\x87\xe8\xb7\xaf\xe5\xbe\x84";  // 中文路径
static const std::string kJapanese =
    "\xe3\x83\x86\xe3\x82\xb9\xe3\x83\x88";          // テスト
static const std::string kMixed = "data_测试_2025";  // data_测试_2025

class Utf8PathTest : public ::testing::Test {
 protected:
  void TearDown() override {
    for (auto &p : cleanup_) {
      FileHelper::RemovePath(p.c_str());
    }
  }
  void ScheduleCleanup(const std::string &path) {
    cleanup_.push_back(path);
  }

 private:
  std::vector<std::string> cleanup_;
};

// ---------------------------------------------------------------------------
// 1. Directory create / query / remove with Chinese path
// ---------------------------------------------------------------------------
TEST_F(Utf8PathTest, MakePath_Chinese) {
  std::string dir = "utf8_test_" + kChinese;
  ScheduleCleanup(dir);

  ASSERT_TRUE(FileHelper::MakePath(dir.c_str()));
  EXPECT_TRUE(FileHelper::IsExist(dir.c_str()));
  EXPECT_TRUE(FileHelper::IsDirectory(dir.c_str()));
  EXPECT_FALSE(FileHelper::IsRegular(dir.c_str()));

  ASSERT_TRUE(FileHelper::RemoveDirectory(dir.c_str()));
  EXPECT_FALSE(FileHelper::IsExist(dir.c_str()));
}

// ---------------------------------------------------------------------------
// 2. Nested directories with mixed UTF-8 segments
// ---------------------------------------------------------------------------
TEST_F(Utf8PathTest, MakePath_Nested) {
  std::string root = "utf8_test_nested_" + kChinese;
  std::string nested = FileHelper::PathJoin(root, kJapanese, kMixed);
  ScheduleCleanup(root);

  ASSERT_TRUE(FileHelper::MakePath(nested.c_str()));
  EXPECT_TRUE(FileHelper::IsDirectory(nested.c_str()));

  ASSERT_TRUE(FileHelper::RemoveDirectory(root.c_str()));
  EXPECT_FALSE(FileHelper::IsExist(root.c_str()));
}

// ---------------------------------------------------------------------------
// 3. File create / read / write through ailego::File with UTF-8 path
// ---------------------------------------------------------------------------
TEST_F(Utf8PathTest, FileCreateAndReadWrite) {
  std::string dir = "utf8_test_file_" + kChinese;
  std::string file_path = FileHelper::PathJoin(dir, kJapanese + ".dat");
  ScheduleCleanup(dir);

  ASSERT_TRUE(FileHelper::MakePath(dir.c_str()));

  const std::string payload =
      "hello_utf8_\xe4\xbd\xa0\xe5\xa5\xbd";  // hello_utf8_你好
  {
    File f;
    ASSERT_TRUE(f.create(file_path.c_str(), 0));
    EXPECT_EQ(payload.size(), f.write(payload.data(), payload.size()));
    EXPECT_TRUE(f.flush());
  }

  EXPECT_TRUE(FileHelper::IsRegular(file_path.c_str()));

  size_t sz = 0;
  ASSERT_TRUE(FileHelper::GetFileSize(file_path.c_str(), &sz));
  EXPECT_EQ(payload.size(), sz);

  {
    File f;
    ASSERT_TRUE(f.open(file_path.c_str(), true));
    std::string buf(sz, '\0');
    EXPECT_EQ(sz, f.read(buf.data(), sz));
    EXPECT_EQ(payload, buf);
  }

  ASSERT_TRUE(FileHelper::DeleteFile(file_path.c_str()));
  EXPECT_FALSE(FileHelper::IsExist(file_path.c_str()));
}

// ---------------------------------------------------------------------------
// 4. FileHelper::RenameFile across UTF-8 paths
// ---------------------------------------------------------------------------
TEST_F(Utf8PathTest, RenameFile) {
  std::string dir = "utf8_test_rename_" + kChinese;
  std::string src = FileHelper::PathJoin(dir, "src_" + kJapanese);
  std::string dst = FileHelper::PathJoin(dir, "dst_" + kMixed);
  ScheduleCleanup(dir);

  ASSERT_TRUE(FileHelper::MakePath(dir.c_str()));
  {
    File f;
    ASSERT_TRUE(f.create(src.c_str(), 0));
  }
  EXPECT_TRUE(FileHelper::IsExist(src.c_str()));

  ASSERT_TRUE(FileHelper::RenameFile(src.c_str(), dst.c_str()));
  EXPECT_FALSE(FileHelper::IsExist(src.c_str()));
  EXPECT_TRUE(FileHelper::IsExist(dst.c_str()));
}

// ---------------------------------------------------------------------------
// 5. OpenIfstream / OpenOfstream with UTF-8 path
// ---------------------------------------------------------------------------
TEST_F(Utf8PathTest, FstreamUtf8) {
  std::string dir = "utf8_test_fstream_" + kChinese;
  std::string file_path = FileHelper::PathJoin(dir, kMixed + ".txt");
  ScheduleCleanup(dir);

  ASSERT_TRUE(FileHelper::MakePath(dir.c_str()));

  const std::string content =
      "line1\nline2_\xe6\xb5\x8b\xe8\xaf\x95\n";  // line2_测试
  {
    std::ofstream ofs;
    FileHelper::OpenOfstream(ofs, file_path, std::ios::binary);
    ASSERT_TRUE(ofs.is_open());
    ofs.write(content.data(), content.size());
  }

  {
    std::ifstream ifs;
    FileHelper::OpenIfstream(ifs, file_path, std::ios::binary);
    ASSERT_TRUE(ifs.is_open());
    std::string buf((std::istreambuf_iterator<char>(ifs)),
                    std::istreambuf_iterator<char>());
    EXPECT_EQ(content, buf);
  }
}

// ---------------------------------------------------------------------------
// 6. PathJoin produces correct separators
// ---------------------------------------------------------------------------
TEST_F(Utf8PathTest, PathJoinSeparator) {
  std::string result = FileHelper::PathJoin("root", "sub", "file.txt");
#if defined(_WIN32) || defined(_WIN64)
  EXPECT_EQ("root\\sub\\file.txt", result);
#else
  EXPECT_EQ("root/sub/file.txt", result);
#endif
}

// ---------------------------------------------------------------------------
// 7. Verify NO mojibake on actual filesystem via "dir" (Windows) / "ls" (POSIX)
//    We create a directory + file with Chinese names, then list the parent
//    using the OS shell and check the expected name appears verbatim.
// ---------------------------------------------------------------------------
#if defined(_WIN32) || defined(_WIN64)
TEST_F(Utf8PathTest, NoGarbledNames_FindFirstFile) {
  std::string parent = "utf8_test_dir_verify";
  std::string child_dir = kChinese;
  std::string child_file = kJapanese + ".dat";
  std::string full_dir = FileHelper::PathJoin(parent, child_dir);
  std::string full_file = FileHelper::PathJoin(parent, child_file);
  ScheduleCleanup(parent);

  ASSERT_TRUE(FileHelper::MakePath(full_dir.c_str()));
  {
    File f;
    ASSERT_TRUE(f.create(full_file.c_str(), 0));
  }

  // Use FindFirstFileW / FindNextFileW to enumerate the parent directory.
  // This is the most reliable way to verify the OS actually stored correct
  // Unicode names (no mojibake), bypassing any console codepage issues.
  std::wstring pattern = FileHelper::Utf8ToWide(parent) + L"\\*";
  WIN32_FIND_DATAW fd;
  HANDLE hFind = FindFirstFileW(pattern.c_str(), &fd);
  ASSERT_NE(INVALID_HANDLE_VALUE, hFind) << "FindFirstFileW failed";

  std::vector<std::wstring> entries;
  do {
    std::wstring name(fd.cFileName);
    if (name != L"." && name != L"..") {
      entries.push_back(name);
    }
  } while (FindNextFileW(hFind, &fd));
  FindClose(hFind);

  std::wstring expected_dir = FileHelper::Utf8ToWide(child_dir);
  std::wstring expected_file = FileHelper::Utf8ToWide(child_file);

  bool found_dir = false, found_file = false;
  std::cout << "[NoGarbledNames] Entries in " << parent << ":" << std::endl;
  for (const auto &e : entries) {
    std::string utf8_name = FileHelper::WideToUtf8(e);
    std::cout << "  " << utf8_name << std::endl;
    if (e == expected_dir) found_dir = true;
    if (e == expected_file) found_file = true;
  }

  EXPECT_TRUE(found_dir) << "Directory '" << child_dir
                         << "' not found (garbled Unicode?)";
  EXPECT_TRUE(found_file) << "File '" << (kJapanese + ".dat")
                          << "' not found (garbled Unicode?)";
}

TEST_F(Utf8PathTest, NoGarbledNames_DirCmd) {
  std::string parent = "utf8_test_dir_cmd";
  std::string child = kChinese;
  std::string full = FileHelper::PathJoin(parent, child);
  std::string tmpfile = "utf8_test_dir_cmd_output.txt";
  ScheduleCleanup(parent);

  ASSERT_TRUE(FileHelper::MakePath(full.c_str()));

  // "cmd /u" makes built-in commands output UTF-16LE when redirected to file.
  std::wstring wparent = FileHelper::Utf8ToWide(parent);
  std::wstring wtmp = FileHelper::Utf8ToWide(tmpfile);
  std::wstring cmd =
      L"cmd /u /c dir /b \"" + wparent + L"\" >\"" + wtmp + L"\"";
  _wsystem(cmd.c_str());

  // Read back the file as raw bytes (UTF-16LE)
  std::ifstream ifs;
  FileHelper::OpenIfstream(ifs, tmpfile, std::ios::binary);
  ASSERT_TRUE(ifs.is_open()) << "Could not read dir output file";
  std::string raw((std::istreambuf_iterator<char>(ifs)),
                  std::istreambuf_iterator<char>());
  ifs.close();
  FileHelper::DeleteFile(tmpfile.c_str());

  // Interpret as UTF-16LE: skip BOM if present, then convert to UTF-8
  const wchar_t *wdata = reinterpret_cast<const wchar_t *>(raw.data());
  size_t wlen = raw.size() / sizeof(wchar_t);
  if (wlen > 0 && wdata[0] == L'\xFEFF') {
    wdata++;
    wlen--;
  }
  std::wstring woutput(wdata, wlen);
  std::string output = FileHelper::WideToUtf8(woutput);

  std::cout << "[NoGarbledNames_DirCmd] dir output: [" << output << "]";

  EXPECT_NE(std::string::npos, output.find(child))
      << "Chinese directory name not found in 'dir' output (garbled?).\n"
      << "Got: [" << output << "]";
}
#endif
