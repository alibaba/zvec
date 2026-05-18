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

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/io/file.h>
#include <zvec/ailego/utility/file_helper.h>
#include "db/common/file_helper.h"
#include "index/utils/utils.h"
#include "zvec/db/collection.h"
#include "zvec/db/doc.h"
#include "zvec/db/options.h"
#include "zvec/db/schema.h"
#include "zvec/db/status.h"

#if defined(_WIN32) || defined(_WIN64)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#undef DeleteFile
#undef RemoveDirectory
#undef CreateFile
#undef DELETE
#endif

using namespace zvec;
using namespace zvec::test;

// UTF-8 test path containing Chinese characters.
// If the path handling is broken, the OS will show garbled names (mojibake).
static const std::string kUtf8Dir =
    "utf8_col_\xe4\xb8\xad\xe6\x96\x87\xe6\xb5\x8b\xe8\xaf\x95";  // utf8_col_中文测试

class Utf8CollectionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ailego::FileHelper::RemovePath(kUtf8Dir.c_str());
  }
  void TearDown() override {
    ailego::FileHelper::RemovePath(kUtf8Dir.c_str());
  }
};

// ---------------------------------------------------------------------------
// 1. Create a collection in a UTF-8 path, insert docs, flush, reopen
// ---------------------------------------------------------------------------
TEST_F(Utf8CollectionTest, CreateInsertFlushReopen) {
  CollectionOptions opts;
  opts.read_only_ = false;
  opts.enable_mmap_ = true;

  auto schema = TestHelper::CreateNormalSchema();
  auto result = Collection::CreateAndOpen(kUtf8Dir, *schema, opts);
  if (!result.has_value()) {
    std::cout << result.error().message() << std::endl;
  }
  ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(ailego::FileHelper::IsExist(kUtf8Dir.c_str()));

  auto col = std::move(result).value();
  ASSERT_EQ(col->Path(), kUtf8Dir);

  const int kDocCount = 10;
  auto s = TestHelper::CollectionInsertDoc(col, 0, kDocCount);
  ASSERT_TRUE(s.ok()) << s.message();

  ASSERT_TRUE(col->Flush().ok());

  auto stats = col->Stats().value();
  ASSERT_EQ(stats.doc_count, kDocCount);

  col.reset();

  // Reopen and verify doc count survives
  auto reopen = Collection::Open(kUtf8Dir, opts);
  ASSERT_TRUE(reopen.has_value()) << reopen.error().message();
  auto col2 = std::move(reopen).value();
  auto stats2 = col2->Stats().value();
  ASSERT_EQ(stats2.doc_count, kDocCount);
}

// ---------------------------------------------------------------------------
// 2. Destroy a collection in a UTF-8 path
// ---------------------------------------------------------------------------
TEST_F(Utf8CollectionTest, CreateAndDestroy) {
  CollectionOptions opts;
  opts.read_only_ = false;
  opts.enable_mmap_ = true;

  auto schema = TestHelper::CreateNormalSchema();
  auto result = Collection::CreateAndOpen(kUtf8Dir, *schema, opts);
  ASSERT_TRUE(result.has_value());

  auto col = std::move(result).value();
  ASSERT_EQ(col->Destroy(), Status::OK());
  ASSERT_FALSE(ailego::FileHelper::IsExist(kUtf8Dir.c_str()));
}

// ---------------------------------------------------------------------------
// 3. Verify the on-disk layout contains correct UTF-8 names (no mojibake)
//    by listing files via the wide-char Windows API or POSIX readdir.
// ---------------------------------------------------------------------------
#if defined(_WIN32) || defined(_WIN64)
TEST_F(Utf8CollectionTest, NoDirGarble) {
  CollectionOptions opts;
  opts.read_only_ = false;
  opts.enable_mmap_ = true;

  auto schema = TestHelper::CreateNormalSchema();
  auto result = Collection::CreateAndOpen(kUtf8Dir, *schema, opts);
  ASSERT_TRUE(result.has_value());
  auto col = std::move(result).value();

  const int kDocCount = 5;
  auto s = TestHelper::CollectionInsertDoc(col, 0, kDocCount);
  ASSERT_TRUE(s.ok()) << s.message();
  ASSERT_TRUE(col->Flush().ok());

  // --- Check parent directory via FindFirstFileW ---
  {
    WIN32_FIND_DATAW fd;
    HANDLE hFind = FindFirstFileW(L".\\*", &fd);
    ASSERT_NE(INVALID_HANDLE_VALUE, hFind);

    std::wstring expected = ailego::FileHelper::Utf8ToWide(kUtf8Dir);
    bool found = false;
    do {
      if (std::wstring(fd.cFileName) == expected) {
        found = true;
        break;
      }
    } while (FindNextFileW(hFind, &fd));
    FindClose(hFind);

    EXPECT_TRUE(found) << "Collection dir '" << kUtf8Dir
                       << "' not found via FindFirstFileW (garbled Unicode?)";
  }

  // --- Check inside collection via "cmd /u /c dir /b" redirected to file ---
  {
    std::wstring wdir = ailego::FileHelper::Utf8ToWide(kUtf8Dir);
    std::string tmpfile = "utf8_col_dir_output.txt";
    std::wstring wtmp = ailego::FileHelper::Utf8ToWide(tmpfile);
    std::wstring cmd = L"cmd /u /c dir /b \"" + wdir + L"\" >\"" + wtmp + L"\"";
    _wsystem(cmd.c_str());

    std::ifstream ifs;
    ailego::FileHelper::OpenIfstream(ifs, tmpfile, std::ios::binary);
    ASSERT_TRUE(ifs.is_open());
    std::string raw((std::istreambuf_iterator<char>(ifs)),
                    std::istreambuf_iterator<char>());
    ifs.close();
    ailego::FileHelper::DeleteFile(tmpfile.c_str());

    const wchar_t *wdata = reinterpret_cast<const wchar_t *>(raw.data());
    size_t wlen = raw.size() / sizeof(wchar_t);
    if (wlen > 0 && wdata[0] == L'\xFEFF') {
      wdata++;
      wlen--;
    }
    std::wstring woutput(wdata, wlen);
    std::string output = ailego::FileHelper::WideToUtf8(woutput);

    EXPECT_FALSE(output.empty()) << "dir listing inside collection is empty";
    std::cout << "[NoDirGarble] Contents of " << kUtf8Dir << ":\n"
              << output << std::endl;
  }
}
#endif

// ---------------------------------------------------------------------------
// 4. MakeWalPath / MakeSegmentPath etc. produce correct UTF-8 results
// ---------------------------------------------------------------------------
TEST_F(Utf8CollectionTest, FileHelperPaths) {
  std::string wal = FileHelper::MakeWalPath(kUtf8Dir, 0, 1);
  std::string seg = FileHelper::MakeSegmentPath(kUtf8Dir, 0);
  std::string fwd = FileHelper::MakeForwardBlockPath(kUtf8Dir, 0, 1, true);
  std::string inv = FileHelper::MakeInvertIndexPath(kUtf8Dir, 0, 1);
  std::string vec = FileHelper::MakeVectorIndexPath(kUtf8Dir, "vec1", 0, 1);

  // All paths should start with our UTF-8 base
  EXPECT_EQ(0u, wal.find(kUtf8Dir));
  EXPECT_EQ(0u, seg.find(kUtf8Dir));
  EXPECT_EQ(0u, fwd.find(kUtf8Dir));
  EXPECT_EQ(0u, inv.find(kUtf8Dir));
  EXPECT_EQ(0u, vec.find(kUtf8Dir));

  // Should not contain forward slash on Windows
#if defined(_WIN32) || defined(_WIN64)
  EXPECT_EQ(std::string::npos, wal.find('/'));
  EXPECT_EQ(std::string::npos, seg.find('/'));
  EXPECT_EQ(std::string::npos, fwd.find('/'));
  EXPECT_EQ(std::string::npos, inv.find('/'));
  EXPECT_EQ(std::string::npos, vec.find('/'));
#endif
}
