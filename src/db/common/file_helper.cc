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

#include "file_helper.h"
#include <sys/stat.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <algorithm>
#include <cstdio>
#include <ailego/pattern/defer.h>

namespace zvec {


// keep consistent with MANIFEST_BACKUP_FILE
const std::string FileHelper::BACKUP_SUFFIX = ".backup_";
const std::string FileHelper::RECOVER_SUFFIX = ".recovering";

bool FileHelper::CopyFile(const std::string &src_file_path,
                          const std::string &dst_file_path) {
  int src_fd = open(src_file_path.c_str(), O_RDONLY, 0);
  if (src_fd < 0) {
    return false;
  }
  AILEGO_DEFER([src_fd] { close(src_fd); });

  std::string dst_file_path_tmp = dst_file_path + ".tmp";
  int dst_fd =
      open(dst_file_path_tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (dst_fd < 0) {
    return false;
  }
  AILEGO_DEFER([dst_fd] { close(dst_fd); });

  ssize_t size;
  char buf[BUFSIZ];
  while ((size = read(src_fd, buf, BUFSIZ)) > 0) {
    if (size != write(dst_fd, buf, size)) {
      return false;
    }
  }
  return rename(dst_file_path_tmp.c_str(), dst_file_path.c_str()) == 0;
}

bool FileHelper::CopyDirectory(const std::string &src_dir_path,
                               const std::string &dst_dir_path) {
  DIR *dir = opendir(src_dir_path.c_str());
  if (!dir) {
    return false;
  }
  AILEGO_DEFER([dir] { closedir(dir); });

  if (!ailego::FileHelper::IsExist(dst_dir_path.c_str())) {
    if (!ailego::FileHelper::MakePath(dst_dir_path.c_str())) {
      return false;
    }
  }

  struct dirent *dent;
  while ((dent = readdir(dir)) != nullptr) {
    if (!strcmp(dent->d_name, ".") || !strcmp(dent->d_name, "..")) {
      continue;
    }
    std::string src_full_path =
        ailego::StringHelper::Concat(src_dir_path, "/", dent->d_name);
    std::string dst_full_path =
        ailego::StringHelper::Concat(dst_dir_path, "/", dent->d_name);

    if (ailego::FileHelper::IsDirectory(src_full_path.c_str())) {
      if (!CopyDirectory(src_full_path, dst_full_path)) {
        return false;
      }
    } else {
      if (!CopyFile(src_full_path, dst_full_path)) {
        return false;
      }
    }
  }
  return true;
}

void FileHelper::CleanupDirectory(const std::string &backup_dir,
                                  size_t max_backup_count,
                                  const char *prefix_name) {
  if (max_backup_count <= 0) {
    return;
  }

  DIR *dir = opendir(backup_dir.c_str());
  if (!dir) {
    return;
  }

  AILEGO_DEFER([dir] { closedir(dir); });

  size_t prefix_len = strlen(prefix_name);
  std::vector<std::string> candidates;
  struct dirent *dent;
  while ((dent = readdir(dir)) != nullptr) {
    if (strncmp(dent->d_name, prefix_name, prefix_len) == 0) {
      candidates.emplace_back(dent->d_name);
    }
  }
  if (candidates.size() <= max_backup_count) {
    return;
  }
  std::sort(candidates.begin(), candidates.end());
  for (size_t i = 0; i < candidates.size() - max_backup_count; ++i) {
    std::string path =
        ailego::StringHelper::Concat(backup_dir, "/", candidates[i].c_str());
    ailego::FileHelper::RemovePath(path.c_str());
  }
}

}  // namespace zvec