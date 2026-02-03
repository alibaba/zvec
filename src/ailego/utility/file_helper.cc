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

#include <zvec/ailego/utility/file_helper.h>

#if defined(_WIN32) || defined(_WIN64)
#include <Windows.h>
#else
#if defined(__APPLE__) || defined(__MACH__)
#include <mach-o/dyld.h>
#endif
#include <sys/stat.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#endif

namespace zvec {
namespace ailego {

bool FileHelper::GetSelfPath(std::string *path) {
#if defined(_WIN32) || defined(_WIN64)
  char buf[MAX_PATH];
  DWORD len = GetModuleFileNameA(NULL, buf, MAX_PATH);
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

  if (len <= 0) {
    return false;
  }
  path->assign(buf, len);
  return true;
}

bool FileHelper::GetFilePath(NativeHandle handle, std::string *path) {
#if defined(_WIN32) || defined(_WIN64)
  char buf[MAX_PATH];
  DWORD len =
      GetFinalPathNameByHandleA(handle, buf, MAX_PATH, FILE_NAME_OPENED);
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

  if (len <= 0) {
    return false;
  }
  path->assign(buf, len);
  return true;
}

#if !defined(_WIN32) && !defined(_WIN64)

static inline char *JoinFilePath(const char *prefix, const char *suffix) {
  size_t prefix_len = strlen(prefix);
  size_t suffix_len = strlen(suffix);

  char *path = (char *)malloc(prefix_len + suffix_len + 2);
  if (path) {
    memcpy(path, prefix, prefix_len);
    memcpy(path + prefix_len + 1, suffix, suffix_len);
    path[prefix_len] = '/';
    path[prefix_len + suffix_len + 1] = '\0';
  }
  return path;
}

bool FileHelper::GetWorkingDirectory(std::string *path) {
  char buf[PATH_MAX];

  if (!getcwd(buf, PATH_MAX)) {
    return false;
  }
  path->assign(buf);
  return !path->empty();
}

bool FileHelper::GetFileSize(const char *path, size_t *psz) {
  struct stat buf;
  if (stat(path, &buf) != 0) {
    return false;
  }
  *psz = buf.st_size;
  return true;
}

bool FileHelper::DeleteFile(const char *path) {
  // Delete a file by the path
  return (unlink(path) == 0);
}

bool FileHelper::RenameFile(const char *oldpath, const char *newpath) {
  return (rename(oldpath, newpath) == 0);
}

bool FileHelper::MakePath(const char *path) {
  char pathbuf[PATH_MAX];
  char *sp, *pp;

  strncpy(pathbuf, path, sizeof(pathbuf) - 1);
  pathbuf[PATH_MAX - 1] = '\0';

  pp = pathbuf;
  while ((sp = strchr(pp, '/')) != nullptr) {
    // Neither root nor double slash in path
    if (sp != pp) {
      *sp = '\0';
      if (mkdir(pathbuf, 0755) == -1 && errno != EEXIST) {
        return false;
      }
      *sp = '/';
    }
    pp = sp + 1;
  }
  return !(*pp != '\0' && mkdir(pathbuf, 0755) == -1 && errno != EEXIST);
}

bool FileHelper::RemoveDirectory(const char *path) {
  DIR *dir = opendir(path);
  if (!dir) {
    return false;
  }

  struct dirent *dent;
  while ((dent = readdir(dir)) != nullptr) {
    if (!strcmp(dent->d_name, ".") || !strcmp(dent->d_name, "..")) {
      continue;
    }
    char *fullpath = JoinFilePath(path, dent->d_name);
    if (!fullpath) {
      continue;
    }

    if (FileHelper::IsDirectory(fullpath)) {
      FileHelper::RemoveDirectory(fullpath);
    } else {
      FileHelper::DeleteFile(fullpath);
    }
    free(fullpath);
  }
  closedir(dir);
  return (rmdir(path) == 0);
}

bool FileHelper::IsExist(const char *path) {
  return (access(path, F_OK) == 0);
}

bool FileHelper::IsRegular(const char *path) {
  struct stat buf;
  if (stat(path, &buf) != 0) {
    return false;
  }
  return ((buf.st_mode & S_IFREG) != 0);
}

bool FileHelper::IsDirectory(const char *path) {
  struct stat buf;
  if (stat(path, &buf) != 0) {
    return false;
  }
  return ((buf.st_mode & S_IFDIR) != 0);
}

bool FileHelper::IsSymbolicLink(const char *path) {
  struct stat buf;
  if (stat(path, &buf) != 0) {
    return false;
  }
  return ((buf.st_mode & S_IFLNK) != 0);
}

bool FileHelper::IsSame(const char *path1, const char *path2) {
  char real_path1[PATH_MAX];
  char real_path2[PATH_MAX];
  if (!realpath(path1, real_path1)) {
    return false;
  }
  if (!realpath(path2, real_path2)) {
    return false;
  }
  return (!strcmp(real_path1, real_path2));
}

#else
#undef RemoveDirectory
#undef DeleteFile
#undef GetFileSize

static inline char *JoinFilePath(const char *prefix, const char *suffix) {
  size_t prefix_len = strlen(prefix);
  size_t suffix_len = strlen(suffix);

  char *path = (char *)malloc(prefix_len + suffix_len + 2);
  if (path) {
    memcpy(path, prefix, prefix_len);
    memcpy(path + prefix_len + 1, suffix, suffix_len);
    path[prefix_len] = '\\';
    path[prefix_len + suffix_len + 1] = '\0';
  }
  return path;
}

bool FileHelper::GetWorkingDirectory(std::string *path) {
  char buf[MAX_PATH];
  DWORD len = GetCurrentDirectoryA(MAX_PATH, buf);

  if (len <= 0) {
    return false;
  }
  path->assign(buf, len);
  return true;
}

bool FileHelper::GetFileSize(const char *path, size_t *psz) {
  HANDLE handle = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr,
                              OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);

  LARGE_INTEGER file_size;
  if (!GetFileSizeEx(handle, &file_size)) {
    return false;
  }
  *psz = (size_t)file_size.QuadPart;
  return true;
}

bool FileHelper::DeleteFile(const char *path) {
  // Delete a file by the path
  return (DeleteFileA(path));
}

bool FileHelper::RenameFile(const char *oldpath, const char *newpath) {
  return (MoveFileA(oldpath, newpath));
}

bool FileHelper::MakePath(const char *path) {
  char pathbuf[MAX_PATH];
  char *sp, *pp;

  strncpy(pathbuf, path, sizeof(pathbuf) - 1);
  pathbuf[MAX_PATH - 1] = '\0';

  pp = pathbuf;
  while ((sp = strpbrk(pp, "/\\")) != nullptr) {
    // Neither root nor double slash in path
    if (sp != pp) {
      *sp = '\0';
      if (!CreateDirectoryA(pathbuf, nullptr) &&
          GetLastError() != ERROR_ALREADY_EXISTS) {
        return false;
      }
      *sp = '\\';
    }
    pp = sp + 1;
  }
  return !(*pp != '\0' && !CreateDirectoryA(pathbuf, nullptr) &&
           GetLastError() != ERROR_ALREADY_EXISTS);
}

bool FileHelper::RemoveDirectory(const char *path) {
  char *pathbuf = JoinFilePath(path, "*.*");
  ailego_false_if_false(pathbuf);

  WIN32_FIND_DATAA file_info;
  HANDLE file = FindFirstFileA(pathbuf, &file_info);

  ailego_do_if_false(file != INVALID_HANDLE_VALUE) {
    free(pathbuf);
    FindClose(file);
    return false;
  }

  do {
    if (!strcmp(file_info.cFileName, ".") ||
        !strcmp(file_info.cFileName, "..")) {
      continue;
    }

    char *fullpath = JoinFilePath(path, file_info.cFileName);
    if (!fullpath) {
      continue;
    }

    if (file_info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
      FileHelper::RemoveDirectory(fullpath);
    } else {
      FileHelper::DeleteFile(fullpath);
    }
    free(fullpath);
  } while (FindNextFileA(file, &file_info));

  free(pathbuf);
  FindClose(file);
  return (!!RemoveDirectoryA(path));
}

bool FileHelper::IsExist(const char *path) {
  DWORD attr = GetFileAttributesA(path);
  return (attr != INVALID_FILE_ATTRIBUTES);
}

bool FileHelper::IsRegular(const char *path) {
  DWORD attr = GetFileAttributesA(path);
  return (attr != INVALID_FILE_ATTRIBUTES &&
          !(attr & FILE_ATTRIBUTE_DIRECTORY));
}

bool FileHelper::IsDirectory(const char *path) {
  DWORD attr = GetFileAttributesA(path);
  return (attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_DIRECTORY));
}

bool FileHelper::IsSymbolicLink(const char *path) {
  DWORD attr = GetFileAttributesA(path);
  return (attr != INVALID_FILE_ATTRIBUTES &&
          (attr & FILE_ATTRIBUTE_REPARSE_POINT));
}

bool FileHelper::IsSame(const char *path1, const char *path2) {
  char real_path1[MAX_PATH];
  char real_path2[MAX_PATH];
  char **part_path1 = nullptr;
  char **part_path2 = nullptr;
  DWORD path1_size =
      GetFullPathNameA(path1, sizeof(real_path1), real_path1, part_path1);
  DWORD path2_size =
      GetFullPathNameA(path2, sizeof(real_path2), real_path2, part_path2);

  if ((part_path1 && *part_path1 != 0) || (part_path2 && *part_path2 != 0) ||
      (path1_size != path2_size)) {
    return false;
  }
  return (!strcmp(real_path1, real_path2));
}

#endif  // !_WIN32 && !_WIN64

bool FileHelper::RemovePath(const char *path) {
  if (FileHelper::IsDirectory(path)) {
    return FileHelper::RemoveDirectory(path);
  }
  return FileHelper::DeleteFile(path);
}

}  // namespace ailego
}  // namespace zvec