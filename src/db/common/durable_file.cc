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

#include "durable_file.h"
#include <filesystem>
#include <zvec/ailego/io/file.h>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace zvec {
namespace {

Status IoError(const char *operation, const std::string &path) {
  return {StatusCode::UNAVAILABLE,
          std::string(operation) + " failed for " + path};
}

Status Before(const DurableFile::Hook &hook, DurableFile::Stage stage,
              const std::string &path) {
  return hook ? hook(stage, path) : Status::OK();
}

}  // namespace

DurableFile::Outcome DurableFile::Replace(const std::string &destination,
                                          std::string_view contents,
                                          const Hook &hook) {
  auto status = Before(hook, Stage::kWrite, destination);
  if (!status.ok()) {
    return {status, false};
  }
  const std::string pending = destination + ".pending";
  ailego::File file;
  if (!file.create(pending, 0) ||
      file.write(contents.data(), contents.size()) != contents.size()) {
    return {IoError("Write", pending), false};
  }
  file.close();
  return PublishPrepared(pending, destination, hook);
}

DurableFile::Outcome DurableFile::PublishPrepared(
    const std::string &prepared, const std::string &destination,
    const Hook &hook) {
  const auto source = ailego::FileHelper::PathFromUtf8(prepared);
  const auto target = ailego::FileHelper::PathFromUtf8(destination);
  if (source == target || source.parent_path() != target.parent_path()) {
    return {
        Status::InvalidArgument("Publication requires distinct sibling paths"),
        false};
  }
  auto status = Before(hook, Stage::kFileSync, destination);
  if (status.ok()) {
    status = Sync(prepared);
  }
  if (!status.ok()) {
    return {status, false};
  }
  status = Before(hook, Stage::kRename, destination);
  if (!status.ok()) {
    return {status, false};
  }
#if defined(_WIN32) || defined(_WIN64)
  // No cross-volume copy fallback: publication must be an atomic rename.
  // WRITE_THROUGH supplies the Windows name-persistence boundary; see
  // https://learn.microsoft.com/windows/win32/api/winbase/nf-winbase-movefileexw
  if (!::MoveFileExW(source.c_str(), target.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
    // Conservatively retain both versions when the combined rename/flush
    // operation cannot confirm completion.
    return {IoError("Rename", destination), true};
  }
#else
  std::error_code error;
  std::filesystem::rename(source, target, error);
  if (error) {
    return {IoError("Rename", destination), false};
  }
#endif
  status = Before(hook, Stage::kDirectorySync, destination);
  if (!status.ok()) {
    return {status, true};
  }
#if !defined(_WIN32) && !defined(_WIN64)
  const auto parent = target.has_parent_path() ? target.parent_path()
                                               : std::filesystem::path(".");
  status = SyncDirectory(parent.u8string());
  if (!status.ok()) {
    return {status, true};
  }
#endif
  return {Status::OK(), true};
}

Status DurableFile::SyncDirectory(const std::string &path) {
#if defined(_WIN32) || defined(_WIN64)
  return Status::NotSupported(
      "Full FTS checkpoint directory sync is not verified on Windows");
#else
  const auto directory_path = ailego::FileHelper::PathFromUtf8(path);
  int flags = O_RDONLY;
#ifdef O_DIRECTORY
  flags |= O_DIRECTORY;
#endif
  const int directory = ::open(directory_path.c_str(), flags);
  if (directory < 0) {
    return IoError("Open directory", path);
  }
  const int synced = ::fsync(directory);
  ::close(directory);
  if (synced != 0) {
    return IoError("Directory sync", path);
  }
#endif
  return Status::OK();
}

Status DurableFile::Sync(const std::string &path) {
  ailego::File file;
  if (!file.open(path, false) || !file.flush()) {
    return IoError("File sync", path);
  }
  return Status::OK();
}

}  // namespace zvec
