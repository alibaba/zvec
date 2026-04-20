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

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <system_error>
#include <zvec/ailego/utility/file_helper.h>

#if defined(_WIN32) || defined(_WIN64)
#include <Windows.h>
#ifdef RemoveDirectory
#undef RemoveDirectory
#endif
#ifdef DeleteFile
#undef DeleteFile
#endif
#ifdef GetFileSize
#undef GetFileSize
#endif
#else
#if defined(__APPLE__) || defined(__MACH__)
#include <mach-o/dyld.h>
#endif
#if defined(__FreeBSD__)
#include <sys/sysctl.h>
#endif
#include <fcntl.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace zvec {
namespace ailego {

namespace {

thread_local std::error_code g_last_fs_error;

void ClearFsError() {
  g_last_fs_error.clear();
}

void SetFsError(std::error_code ec) {
  g_last_fs_error = ec;
}

}  // namespace

// ---------- public UTF-8 / wide helpers ----------

fs::path FileHelper::PathFromUtf8(const char *s) {
#if defined(_WIN32) || defined(_WIN64)
  if (!s || !*s) {
    return fs::path();
  }
  return fs::u8path(s);
#else
  return fs::path(s ? s : "");
#endif
}

fs::path FileHelper::PathFromUtf8(const std::string &s) {
  return PathFromUtf8(s.c_str());
}

std::string FileHelper::PathToUtf8(const fs::path &p) {
  return p.u8string();
}

#if defined(_WIN32) || defined(_WIN64)
std::wstring FileHelper::Utf8ToWide(const std::string &src) {
  if (src.empty()) {
    return {};
  }
  int src_len = static_cast<int>(src.size());
  int dst_len =
      MultiByteToWideChar(CP_UTF8, 0, src.data(), src_len, nullptr, 0);
  if (dst_len <= 0) {
    return {};
  }
  std::wstring dst(static_cast<size_t>(dst_len), L'\0');
  if (MultiByteToWideChar(CP_UTF8, 0, src.data(), src_len, dst.data(),
                          dst_len) != dst_len) {
    return {};
  }
  return dst;
}

std::string FileHelper::WideToUtf8(const std::wstring &src) {
  if (src.empty()) {
    return {};
  }
  int src_len = static_cast<int>(src.size());
  int dst_len = WideCharToMultiByte(CP_UTF8, 0, src.data(), src_len, nullptr, 0,
                                    nullptr, nullptr);
  if (dst_len <= 0) {
    return {};
  }
  std::string dst(static_cast<size_t>(dst_len), '\0');
  if (WideCharToMultiByte(CP_UTF8, 0, src.data(), src_len, dst.data(), dst_len,
                          nullptr, nullptr) != dst_len) {
    return {};
  }
  return dst;
}

#endif

// ---------- internal helpers ----------

namespace {

static bool GetFileSizeImpl(const fs::path &p, size_t *psz) {
  ClearFsError();
  std::error_code ec;
  auto sz = fs::file_size(p, ec);
  if (ec) {
    SetFsError(ec);
    return false;
  }
  *psz = static_cast<size_t>(sz);
  return true;
}

static bool DeleteFileImpl(const fs::path &p) {
  ClearFsError();
  std::error_code ec;
  fs::file_status st = fs::symlink_status(p, ec);
  if (ec) {
    SetFsError(ec);
    return false;
  }
  if (fs::is_directory(st) && !fs::is_symlink(st)) {
    ec = std::make_error_code(std::errc::is_a_directory);
    SetFsError(ec);
    return false;
  }
  if (!fs::remove(p, ec)) {
    SetFsError(ec ? ec
                  : std::make_error_code(std::errc::no_such_file_or_directory));
    return false;
  }
  return true;
}

static bool RenameFileImpl(const fs::path &from, const fs::path &to) {
  ClearFsError();
  std::error_code ec;
  fs::rename(from, to, ec);
  if (ec) {
    SetFsError(ec);
    return false;
  }
  return true;
}

static bool MakePathImpl(const fs::path &p) {
  ClearFsError();
  std::error_code ec;
  fs::create_directories(p, ec);
  if (ec) {
    SetFsError(ec);
    return false;
  }
  return true;
}

static bool RemoveDirectoryImpl(const fs::path &p) {
  ClearFsError();
  std::error_code ec;
  if (!fs::is_directory(p, ec)) {
    if (ec) {
      SetFsError(ec);
    }
    return false;
  }
  std::uintmax_t n = fs::remove_all(p, ec);
  if (ec) {
    SetFsError(ec);
    return false;
  }
  (void)n;
  return true;
}

static bool IsExistImpl(const fs::path &p) {
  std::error_code ec;
  return fs::exists(p, ec);
}

static bool IsRegularImpl(const fs::path &p) {
  std::error_code ec;
  return fs::is_regular_file(p, ec);
}

static bool IsDirectoryImpl(const fs::path &p) {
  std::error_code ec;
  return fs::is_directory(p, ec);
}

static bool IsSymbolicLinkImpl(const fs::path &p) {
  std::error_code ec;
  return fs::is_symlink(p, ec);
}

static bool IsSameImpl(const fs::path &a, const fs::path &b) {
  std::error_code ec;
  return fs::equivalent(a, b, ec);
}

}  // namespace

bool FileHelper::GetSelfPath(std::string *path) {
#if defined(_WIN32) || defined(_WIN64)
  std::wstring wbuf(4096, L'\0');
  DWORD n =
      GetModuleFileNameW(nullptr, wbuf.data(), static_cast<DWORD>(wbuf.size()));
  while (n >= wbuf.size() - 1) {
    if (wbuf.size() > 65536) {
      return false;
    }
    wbuf.resize(wbuf.size() * 2);
    n = GetModuleFileNameW(nullptr, wbuf.data(),
                           static_cast<DWORD>(wbuf.size()));
  }
  if (n == 0) {
    return false;
  }
  wbuf.resize(n);
  *path = WideToUtf8(wbuf);
  return !path->empty();
#elif defined(__APPLE__) || defined(__MACH__)
  char buf[PATH_MAX];
  size_t len = 0;

  char dirty_buf[PATH_MAX];
  uint32_t size = sizeof(dirty_buf);
  if (_NSGetExecutablePath(dirty_buf, &size) == 0) {
    realpath(dirty_buf, buf);
    len = strlen(buf);
  }
#elif defined(__FreeBSD__)
  char buf[PATH_MAX];
  size_t len = PATH_MAX;
  int mib[4] = {CTL_KERN, KERN_PROC, KERN_PROC_PATHNAME, -1};
  if (sysctl(mib, 4, &buf, &len, NULL, 0) != 0) {
    len = 0;
  }
#else
  char buf[PATH_MAX];
  ssize_t len = readlink("/proc/self/exe", buf, PATH_MAX);
#endif

#if !defined(_WIN32) && !defined(_WIN64)
  if (len <= 0) {
    return false;
  }
  path->assign(buf, len);
  return true;
#endif
}

bool FileHelper::GetFilePath(NativeHandle handle, std::string *path) {
#if defined(_WIN32) || defined(_WIN64)
  DWORD need = GetFinalPathNameByHandleW(static_cast<HANDLE>(handle), nullptr,
                                         0, FILE_NAME_OPENED);
  if (need == 0) {
    return false;
  }
  std::wstring wbuf(static_cast<size_t>(need) + 1, L'\0');
  DWORD got = GetFinalPathNameByHandleW(
      static_cast<HANDLE>(handle), wbuf.data(), need + 1, FILE_NAME_OPENED);
  if (got == 0 || got > need) {
    return false;
  }
  wbuf.resize(got);
  *path = WideToUtf8(wbuf);
  return !path->empty();
#elif defined(__linux) || defined(__linux__)
  char buf[PATH_MAX];
  char src[32];
  snprintf(src, sizeof(src), "/proc/self/fd/%d", handle);
  ssize_t len = readlink(src, buf, PATH_MAX);
#else
  char buf[PATH_MAX];
  size_t len = 0;
  if (fcntl(handle, F_GETPATH, buf) != -1) {
    len = strlen(buf);
  }
#endif

#if !defined(_WIN32) && !defined(_WIN64)
  if (len <= 0) {
    return false;
  }
  path->assign(buf, len);
  return true;
#endif
}

bool FileHelper::GetWorkingDirectory(std::string *path) {
  ClearFsError();
  std::error_code ec;
  fs::path cwd = fs::current_path(ec);
  if (ec) {
    SetFsError(ec);
    return false;
  }
  *path = PathToUtf8(cwd);
  return !path->empty();
}

bool FileHelper::GetFileSize(const char *path, size_t *psz) {
  return GetFileSizeImpl(PathFromUtf8(path), psz);
}

bool FileHelper::DeleteFile(const char *path) {
  return DeleteFileImpl(PathFromUtf8(path));
}

bool FileHelper::RenameFile(const char *oldpath, const char *newpath) {
  return RenameFileImpl(PathFromUtf8(oldpath), PathFromUtf8(newpath));
}

bool FileHelper::MakePath(const char *path) {
  return MakePathImpl(PathFromUtf8(path));
}

bool FileHelper::RemoveDirectory(const char *path) {
  if (path == nullptr || *path == '\0') {
    return false;
  }
  return RemoveDirectoryImpl(PathFromUtf8(path));
}

bool FileHelper::IsExist(const char *path) {
  return IsExistImpl(PathFromUtf8(path));
}

bool FileHelper::IsRegular(const char *path) {
  return IsRegularImpl(PathFromUtf8(path));
}

bool FileHelper::IsDirectory(const char *path) {
  return IsDirectoryImpl(PathFromUtf8(path));
}

bool FileHelper::IsSymbolicLink(const char *path) {
  return IsSymbolicLinkImpl(PathFromUtf8(path));
}

bool FileHelper::IsSame(const char *path1, const char *path2) {
  return IsSameImpl(PathFromUtf8(path1), PathFromUtf8(path2));
}

std::string FileHelper::GetLastErrorString() {
  if (g_last_fs_error) {
    return g_last_fs_error.message();
  }
#if defined(_WIN32) || defined(_WIN64)
  DWORD err = GetLastError();
  if (err == 0) {
    return "No error";
  }
  char buf[256];
  DWORD len = FormatMessageA(
      FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, err,
      MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), buf, sizeof(buf), nullptr);
  if (len > 0) {
    while (len > 0 && (buf[len - 1] == '\r' || buf[len - 1] == '\n')) {
      buf[--len] = '\0';
    }
    return std::string(buf, len);
  }
  return "Unknown error " + std::to_string(err);
#else
  return strerror(errno);
#endif
}

bool FileHelper::RemovePath(const char *path) {
  if (FileHelper::IsDirectory(path)) {
    return FileHelper::RemoveDirectory(path);
  }
  return FileHelper::DeleteFile(path);
}

}  // namespace ailego
}  // namespace zvec
