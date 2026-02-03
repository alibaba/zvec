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

#include <zvec/ailego/utility/file_helper.h>

namespace zvec {
namespace ailego {

/*! File Utility
 */
class File {
 public:
  //! Native Handle in OS
  typedef FileHelper::NativeHandle NativeHandle;

  //! Invalid Handle
  static constexpr NativeHandle InvalidHandle = (NativeHandle)(-1);

  //! Specifies the position in a file to use for seeking.
  enum struct Origin { Begin = 0, Current = 1, End = 2 };

  //! Options of memory mapping
  enum {
    MMAP_READONLY = 1,
    MMAP_SHARED = 2,
    MMAP_LOCKED = 4,
    MMAP_WARMUP = 8,
    MMAP_POPULATE = 16,
    MMAP_HUGE_PAGE = 32,
  };

  //! Constructor
  File(void) : native_handle_(File::InvalidHandle), read_only_(false) {}

  //! Constructor
  File(File &&rhs) {
    read_only_ = rhs.read_only_;
    native_handle_ = rhs.native_handle_;
    rhs.read_only_ = false;
    rhs.native_handle_ = File::InvalidHandle;
  }

  //! Destructor
  ~File(void) {
    this->close();
  }

  //! Assignment
  File &operator=(File &&rhs) {
    read_only_ = rhs.read_only_;
    native_handle_ = rhs.native_handle_;
    rhs.read_only_ = false;
    rhs.native_handle_ = File::InvalidHandle;
    return *this;
  }

  //! Test if the file is valid
  bool is_valid(void) const {
    return (native_handle_ != File::InvalidHandle);
  }

  //! Retrieve non-zero if memory region is read only
  bool read_only(void) const {
    return read_only_;
  }

  //! Retrieve native handle
  NativeHandle native_handle(void) const {
    return native_handle_;
  }

  //! Create a local file
  bool create(const char *path, size_t size, bool direct);

  //! Open a local file
  bool open(const char *path, bool rdonly, bool direct);

  //! Close the local file
  void close(void);

  //! Reset the file
  void reset(void);

  //! Write data into the file
  size_t write(const void *data, size_t len);

  //! Write data into the file
  size_t write(ssize_t off, const void *data, size_t len);

  //! Read data from the file
  size_t read(void *buf, size_t len);

  //! Read data from the file
  size_t read(ssize_t off, void *buf, size_t len);

  //! Synchronize memory with physical storage
  bool flush(void);

  //! Sets the current position of the file to the given value
  bool seek(ssize_t off, Origin origin);

  //! Truncate the file to a specified length
  bool truncate(size_t len);

  //! Retrieve size of file
  size_t size(void) const;

  //! Retrieve offset of file
  ssize_t offset(void) const;

  //! Create a local file
  bool create(const char *path, size_t len) {
    return this->create(path, len, false);
  }

  //! Create a local file
  bool create(const std::string &path, size_t len, bool direct) {
    return this->create(path.c_str(), len, direct);
  }

  //! Create a local file
  bool create(const std::string &path, size_t len) {
    return this->create(path.c_str(), len);
  }

  //! Open a local file
  bool open(const char *path, bool rdonly) {
    return this->open(path, rdonly, false);
  }

  //! Open a local file
  bool open(const std::string &path, bool rdonly, bool direct) {
    return this->open(path.c_str(), rdonly, direct);
  }

  //! Open a local file
  bool open(const std::string &path, bool rdonly) {
    return this->open(path.c_str(), rdonly);
  }

  //! Map a region of file into memory
  void *map(ssize_t off, size_t len, int opts) {
    if (read_only_) {
      opts |= File::MMAP_READONLY;
    }
    return File::MemoryMap(native_handle_, off, len, opts);
  }

  //! Map a region of file into memory
  static void *MemoryMap(NativeHandle handle, ssize_t off, size_t len,
                         int opts);

  //! Map an anonymous region into memory
  static void *MemoryMap(size_t len, int opts);

  //! Remap the region into memory
  static void *MemoryRemap(void *oldptr, size_t oldsize, void *newptr,
                           size_t newsize);

  //! Unmap a mapping region
  static void MemoryUnmap(void *addr, size_t len);

  //! Synchronize a memory map
  static bool MemoryFlush(void *addr, size_t len);

  //! Lock the memory region into RAM
  static bool MemoryLock(void *addr, size_t len);

  //! Unlock the memory region in RAM
  static bool MemoryUnlock(void *addr, size_t len);

  //! Warm up a memory region
  static void MemoryWarmup(void *addr, size_t len);

  //! Delete a name and possibly the file it refers to
  static bool Delete(const char *path) {
    return FileHelper::DeleteFile(path);
  }

  //! Delete a name and possibly the file it refers to
  static bool Delete(const std::string &path) {
    return FileHelper::DeleteFile(path.c_str());
  }

  //! Change the name or location of a file
  static bool Rename(const char *oldpath, const char *newpath) {
    return FileHelper::RenameFile(oldpath, newpath);
  }

  //! Change the name or location of a file
  static bool Rename(const std::string &oldpath, const std::string &newpath) {
    return FileHelper::RenameFile(oldpath.c_str(), newpath.c_str());
  }

  //! Retrieve the base name from a path
  static const char *BaseName(const char *path) {
    return FileHelper::BaseName(path);
  }

  //! Retrieve the base name from a path
  static const char *BaseName(const std::string &path) {
    return BaseName(path.c_str());
  }

  //! Make directories' path
  static bool MakePath(const char *path) {
    return FileHelper::MakePath(path);
  }

  //! Make directories' path
  static bool MakePath(const std::string &path) {
    return FileHelper::MakePath(path.c_str());
  }

  //! Remove a file or a directory (includes files & subdirectories)
  static bool RemovePath(const char *path) {
    return FileHelper::RemovePath(path);
  }

  //! Remove a file or a directory (includes files & subdirectories)
  static bool RemovePath(const std::string &path) {
    return FileHelper::RemovePath(path.c_str());
  }

  //! Remove a directory (includes files & subdirectories)
  static bool RemoveDirectory(const char *path) {
    return FileHelper::RemoveDirectory(path);
  }

  //! Remove a directory (includes files & subdirectories)
  static bool RemoveDirectory(const std::string &path) {
    return FileHelper::RemoveDirectory(path.c_str());
  }

  //! Retrieve non-zero if the path exists
  static bool IsExist(const char *path) {
    return FileHelper::IsExist(path);
  }

  //! Retrieve non-zero if the path exists
  static bool IsExist(const std::string &path) {
    return FileHelper::IsExist(path.c_str());
  }

  //! Retrieve non-zero if the path is a regular file
  static bool IsRegular(const char *path) {
    return FileHelper::IsRegular(path);
  }

  //! Retrieve non-zero if the path is a regular file
  static bool IsRegular(const std::string &path) {
    return FileHelper::IsRegular(path.c_str());
  }

  //! Retrieve non-zero if the path is a directory
  static bool IsDirectory(const char *path) {
    return FileHelper::IsDirectory(path);
  }

  //! Retrieve non-zero if the path is a directory
  static bool IsDirectory(const std::string &path) {
    return FileHelper::IsDirectory(path.c_str());
  }

  //! Retrieve non-zero if the path is a symbolic link
  static bool IsSymbolicLink(const char *path) {
    return FileHelper::IsSymbolicLink(path);
  }

  //! Retrieve non-zero if the path is a symbolic link
  static bool IsSymbolicLink(const std::string &path) {
    return FileHelper::IsSymbolicLink(path.c_str());
  }

  //! Retrieve non-zero if two paths are pointing to the same file
  static bool IsSame(const char *path1, const char *path2) {
    return FileHelper::IsSame(path1, path2);
  }

  //! Retrieve non-zero if two paths are pointing to the same file
  static bool IsSame(const std::string &path1, const std::string &path2) {
    return FileHelper::IsSame(path1.c_str(), path2.c_str());
  }

 private:
  //! Disable them
  File(const File &) = delete;
  File &operator=(const File &) = delete;

  //! Members
  NativeHandle native_handle_;
  bool read_only_;
};

}  // namespace ailego
}  // namespace zvec
