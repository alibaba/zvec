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

#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <zvec/ailego/internal/platform.h>
#include <zvec/ailego/utility/string_helper.h>

namespace zvec {
namespace ailego {

/*! File Helper Module
 */
struct FileHelper {
#if defined(_WIN32) || defined(_WIN64)
  //! Native Handle in Windows
  typedef void *NativeHandle;
#else
  //! Native Handle in POSIX
  typedef int NativeHandle;
#endif

  //! Invalid Handle
  static constexpr NativeHandle InvalidHandle = (NativeHandle)(-1);

  //! Retrieve the path of self process
  static bool GetSelfPath(std::string *path);

  //! Retrieve the final path for the specified file
  static bool GetFilePath(NativeHandle handle, std::string *path);

  //! Retrieve current working directory (UTF-8 bytes in \p *path)
  static bool GetWorkingDirectory(std::string *path);

  //! Narrow paths are UTF-8 on all platforms (on Windows, decoded as UTF-8;
  //! on POSIX, native narrow encoding is typically UTF-8).
  static bool GetFileSize(const char *path, size_t *psz);

  //! Delete a name and possibly the file it refers to
  static bool DeleteFile(const char *path);

  //! Change the name or location of a file
  static bool RenameFile(const char *oldpath, const char *newpath);

  //! Make directories' path
  static bool MakePath(const char *path);

  //! Remove a file or a directory (includes files & subdirectories)
  static bool RemovePath(const char *path);

  //! Remove a directory (includes files & subdirectories)
  static bool RemoveDirectory(const char *path);

  //! Retrieve non-zero if the path exists
  static bool IsExist(const char *path);

  //! Retrieve non-zero if the path is a regular file
  static bool IsRegular(const char *path);

  //! Retrieve non-zero if the path is a directory
  static bool IsDirectory(const char *path);

  //! Retrieve non-zero if the path is a symbolic link
  static bool IsSymbolicLink(const char *path);

  //! Retrieve non-zero if two paths are pointing to the same file
  static bool IsSame(const char *path1, const char *path2);

  //! Human-readable error: last std::filesystem failure on this thread if any,
  //! else GetLastError() on Windows or strerror(errno) on POSIX.
  static std::string GetLastErrorString();

  //! Retrieve the size of a file
  static size_t FileSize(const char *path) {
    size_t file_size = 0;
    GetFileSize(path, &file_size);
    return file_size;
  }

  //! Retrieve the base name from a path
  static const char *BaseName(const char *path) {
    const char *output = std::strrchr(path, '/');
    if (!output) {
      output = std::strrchr(path, '\\');
    }
    return (output ? output + 1 : path);
  }

  //! Build std::filesystem::path from a UTF-8 string (handles Windows codepage)
  static std::filesystem::path PathFromUtf8(const std::string &s);
  static std::filesystem::path PathFromUtf8(const char *s);

  //! Convert std::filesystem::path back to a UTF-8 std::string
  static std::string PathToUtf8(const std::filesystem::path &p);

  //! Concatenate path segments with the native separator.
  //! Accepts strings, c-strings, and numbers (anything Alphameric accepts).
  //! E.g. PathJoin(dir, seg_id, "file.ext")
  static std::string PathJoin(const internal::Alphameric &a,
                              const internal::Alphameric &b) {
    return StringHelper::Concat(a, kPathSep, b);
  }
  template <typename... Rest>
  static std::string PathJoin(const internal::Alphameric &a,
                              const internal::Alphameric &b,
                              const Rest &...rest) {
    return PathJoin(StringHelper::Concat(a, kPathSep, b), rest...);
  }

 private:
#if defined(_WIN32) || defined(_WIN64)
  static constexpr const char *kPathSep = "\\";
#else
  static constexpr const char *kPathSep = "/";
#endif
 public:
#if defined(_WIN32) || defined(_WIN64)
  //! UTF-8 narrow string -> wide string (Win32 API ready)
  static std::wstring Utf8ToWide(const std::string &utf8);

  //! Wide string -> UTF-8 narrow string
  static std::string WideToUtf8(const std::wstring &ws);
#endif

  //! Open a std::ifstream from a UTF-8 path
  static void OpenIfstream(std::ifstream &ifs, const std::string &path,
                           std::ios_base::openmode mode = std::ios_base::in) {
    ifs.open(PathFromUtf8(path), mode);
  }

  //! Open a std::ofstream from a UTF-8 path
  static void OpenOfstream(std::ofstream &ofs, const std::string &path,
                           std::ios_base::openmode mode = std::ios_base::out) {
    ofs.open(PathFromUtf8(path), mode);
  }
};

}  // namespace ailego
}  // namespace zvec
