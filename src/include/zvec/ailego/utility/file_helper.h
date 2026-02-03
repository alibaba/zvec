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
#include <string>

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

  //! Retrieve current working directory
  static bool GetWorkingDirectory(std::string *path);

  //! Get the size of a file
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
};

}  // namespace ailego
}  // namespace zvec
