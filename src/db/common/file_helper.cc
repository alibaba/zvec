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
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <vector>


namespace zvec {


// keep consistent with MANIFEST_BACKUP_FILE
const std::string FileHelper::BACKUP_SUFFIX = ".backup_";
const std::string FileHelper::RECOVER_SUFFIX = ".recovering";

bool FileHelper::CopyFile(const std::string &src_file_path,
                          const std::string &dst_file_path) {
  std::string dst_file_path_tmp = dst_file_path + ".tmp";
  std::error_code ec;
  std::filesystem::copy_file(
      ailego::FileHelper::PathFromUtf8(src_file_path),
      ailego::FileHelper::PathFromUtf8(dst_file_path_tmp),
      std::filesystem::copy_options::overwrite_existing, ec);
  if (ec) {
    return false;
  }
  std::filesystem::rename(ailego::FileHelper::PathFromUtf8(dst_file_path_tmp),
                          ailego::FileHelper::PathFromUtf8(dst_file_path), ec);
  return !ec;
}

bool FileHelper::CopyDirectory(const std::string &src_dir_path,
                               const std::string &dst_dir_path) {
  std::error_code ec;
  std::filesystem::copy(ailego::FileHelper::PathFromUtf8(src_dir_path),
                        ailego::FileHelper::PathFromUtf8(dst_dir_path),
                        std::filesystem::copy_options::recursive |
                            std::filesystem::copy_options::overwrite_existing,
                        ec);
  return !ec;
}

void FileHelper::CleanupDirectory(const std::string &backup_dir,
                                  size_t max_backup_count,
                                  const char *prefix_name) {
  if (max_backup_count == 0) {
    return;
  }
  size_t prefix_len = strlen(prefix_name);
  std::vector<std::string> candidates;
  std::error_code ec;
  for (const auto &entry : std::filesystem::directory_iterator(
           ailego::FileHelper::PathFromUtf8(backup_dir), ec)) {
    std::string name = entry.path().filename().u8string();
    if (name.compare(0, prefix_len, prefix_name) == 0) {
      candidates.emplace_back(name);
    }
  }
  if (ec) {
    return;
  }
  if (candidates.size() <= max_backup_count) {
    return;
  }
  std::sort(candidates.begin(), candidates.end());
  for (size_t i = 0; i < candidates.size() - max_backup_count; ++i) {
    std::string path_str =
        ailego::FileHelper::PathJoin(backup_dir, candidates[i]);
    ailego::FileHelper::RemovePath(path_str.c_str());
  }
}

}  // namespace zvec