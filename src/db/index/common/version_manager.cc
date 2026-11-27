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

#include "version_manager.h"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#endif
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>
#include <zvec/ailego/logger/logger.h>
#include <zvec/ailego/pattern/expected.hpp>
#include <zvec/ailego/utility/string_helper.h>
#include <zvec/db/status.h>
#include "db/common/file_helper.h"
#include "db/common/typedef.h"
#include "db/index/common/manifest_codec.h"
#include "db/index/common/type_helper.h"

namespace zvec {

namespace {

namespace fs = std::filesystem;

std::atomic<VersionManager::SyncFaultForTest> g_sync_fault{nullptr};

using SyncStep = VersionManager::SyncStep;

int InjectedFault(SyncStep step) {
  auto fault = g_sync_fault.load();
  return fault ? fault(step) : 0;
}

#ifndef _WIN32
std::string ErrorText(int error) {
  return std::strerror(error);
}

// Flushes a file or directory to stable storage. Returns 0 or an errno value.
int SyncDescriptor(int fd, bool directory) {
  if (int injected = InjectedFault(directory ? SyncStep::kDirectorySync
                                             : SyncStep::kFileSync)) {
    return injected;
  }
#ifdef F_FULLFSYNC
  // On Apple platforms fsync() only hands the data to the drive; F_FULLFSYNC
  // also asks the drive to flush its cache (fcntl(2), fsync(2)). fcntl(2)
  // documents no F_FULLFSYNC-specific errors; an object that does not
  // implement it fails with a "not supported" code (devfs: ENODEV). Only then
  // fall back to fsync(); every other error, EIO included, is reported.
  int result;
  do {
    result = ::fcntl(fd, F_FULLFSYNC);
  } while (result != 0 && errno == EINTR);
  if (result == 0) {
    return 0;
  }
  if (errno != ENOTSUP && errno != EOPNOTSUPP && errno != EINVAL &&
      errno != ENOTTY && errno != ENODEV) {
    return errno;
  }
#endif
  while (::fsync(fd) != 0) {
    if (errno != EINTR) {
      return errno;
    }
  }
  return 0;
}

// Writes `data` to a new file at `path` and syncs it. Returns 0 or an errno
// value. Every write, the sync and the close are checked.
int WriteFileDurably(const std::string &path, const std::string &data) {
  int fd = -1;
  do {
    // 0666 & ~umask, the mode std::ofstream used.
    fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0666);
  } while (fd < 0 && errno == EINTR);
  if (fd < 0) {
    return errno;
  }
  int error = 0;
  size_t written = 0;
  while (written < data.size()) {
    ssize_t n = ::write(fd, data.data() + written, data.size() - written);
    if (n < 0 && errno == EINTR) {
      continue;
    }
    if (n <= 0) {
      error = n < 0 ? errno : EIO;
      break;
    }
    written += static_cast<size_t>(n);
  }
  if (error == 0) {
    error = SyncDescriptor(fd, false);
  }
  int close_error = ::close(fd) != 0 ? errno : 0;
  if (int injected = InjectedFault(SyncStep::kFileClose)) {
    close_error = injected;
  }
  if (error == 0) {
    error = close_error;
  }
  return error;
}

// Makes a rename in `dir` durable. Returns 0 or an errno value.
// A file system that cannot sync directories reports EINVAL (or ENOTSUP /
// EOPNOTSUPP) from the sync. Accepting that is a portability concession:
// the rename's durability is not proven on such file systems. PostgreSQL
// makes the same concession (fsync_fname_ext() in src/backend/storage/file/
// fd.c: "Some OSes don't allow us to fsync directories at all"). Only the
// sync result is normalized; a failed close is always reported.
int SyncDirectory(const std::string &dir) {
  int fd = ::open(dir.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECTORY);
  if (fd < 0) {
    return errno;
  }
  int sync_error = SyncDescriptor(fd, true);
  if (sync_error == EINVAL || sync_error == ENOTSUP ||
      sync_error == EOPNOTSUPP) {
    static std::atomic<bool> logged{false};
    if (!logged.exchange(true)) {
      LOG_WARN("Directory sync is not supported for %s (%s); continuing",
               dir.c_str(), std::strerror(sync_error));
    }
    sync_error = 0;
  }
  int close_error = ::close(fd) != 0 ? errno : 0;
  if (int injected = InjectedFault(SyncStep::kDirectoryClose)) {
    close_error = injected;
  }
  return sync_error != 0 ? sync_error : close_error;
}
#else
std::string ErrorText(DWORD error) {
  return std::system_category().message(static_cast<int>(error));
}

// Writes `data` to a new file at `path` and flushes it. Returns 0 or a
// Win32 error code.
DWORD WriteFileDurably(const std::wstring &path, const std::string &data) {
  HANDLE handle = ::CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr,
                                CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (handle == INVALID_HANDLE_VALUE) {
    return ::GetLastError();
  }
  DWORD error = 0;
  size_t written = 0;
  while (written < data.size()) {
    DWORD chunk = static_cast<DWORD>(
        (std::min)(data.size() - written, static_cast<size_t>(1u << 30)));
    DWORD n = 0;
    if (!::WriteFile(handle, data.data() + written, chunk, &n, nullptr)) {
      error = ::GetLastError();
      break;
    }
    if (n == 0) {
      error = ERROR_WRITE_FAULT;
      break;
    }
    written += n;
  }
  if (error == 0 && InjectedFault(SyncStep::kFileSync) != 0) {
    error = ERROR_WRITE_FAULT;
  }
  if (error == 0 && !::FlushFileBuffers(handle)) {
    error = ::GetLastError();
  }
  if (!::CloseHandle(handle) && error == 0) {
    error = ::GetLastError();
  }
  if (error == 0 && InjectedFault(SyncStep::kFileClose) != 0) {
    error = ERROR_WRITE_FAULT;
  }
  return error;
}
#endif

// Syncs the existing file at `path` and the directory holding it.
Status SyncPublishedFile(const std::string &path) {
  auto dir = ailego::FileHelper::PathFromUtf8(path).parent_path();
  const std::string dir_name =
      dir.empty() ? std::string(".") : ailego::FileHelper::PathToUtf8(dir);
#ifndef _WIN32
  int error = 0;
  int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    error = errno;
  } else {
    error = SyncDescriptor(fd, false);
    int close_error = ::close(fd) != 0 ? errno : 0;
    if (int injected = InjectedFault(SyncStep::kFileClose)) {
      close_error = injected;
    }
    if (error == 0) {
      error = close_error;
    }
  }
  if (error == 0) {
    error = SyncDirectory(dir_name);
  }
  if (error != 0) {
    return Status::InternalError("Failed to sync manifest ", path, ": ",
                                 ErrorText(error));
  }
#else
  const std::wstring wide_path =
      ailego::FileHelper::PathFromUtf8(path).wstring();
  HANDLE handle =
      ::CreateFileW(wide_path.c_str(), GENERIC_WRITE,
                    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                    nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  DWORD error = 0;
  if (handle == INVALID_HANDLE_VALUE) {
    error = ::GetLastError();
  } else {
    if (InjectedFault(SyncStep::kFileSync) != 0) {
      error = ERROR_WRITE_FAULT;
    } else if (!::FlushFileBuffers(handle)) {
      error = ::GetLastError();
    }
    if (!::CloseHandle(handle) && error == 0) {
      error = ::GetLastError();
    }
    if (error == 0 && InjectedFault(SyncStep::kFileClose) != 0) {
      error = ERROR_WRITE_FAULT;
    }
  }
  if (error == 0 && InjectedFault(SyncStep::kDirectorySync) != 0) {
    error = ERROR_WRITE_FAULT;
  }
  if (error != 0) {
    return Status::InternalError("Failed to sync manifest ", path, ": ",
                                 ErrorText(error));
  }
#endif
  (void)dir_name;
  return Status::OK();
}

// Publishes `data` at `path` via `tmp_path`: write and sync the temporary
// file, rename it over `path`, then make the rename durable. `*renamed`
// tells whether the rename happened; on error after it, the outcome of the
// publish is unknown.
Status PublishFile(const std::string &tmp_path, const std::string &path,
                   const std::string &data, bool *renamed) {
  *renamed = false;
#ifndef _WIN32
  int error = WriteFileDurably(tmp_path, data);
  if (error == 0 && ::rename(tmp_path.c_str(), path.c_str()) != 0) {
    error = errno;
  }
  if (error != 0) {
    ::unlink(tmp_path.c_str());
    return Status::InternalError("Failed to write manifest ", path, ": ",
                                 ErrorText(error));
  }
  *renamed = true;
  auto dir = ailego::FileHelper::PathFromUtf8(path).parent_path();
  const std::string dir_name =
      dir.empty() ? std::string(".") : ailego::FileHelper::PathToUtf8(dir);
  error = SyncDirectory(dir_name);
  if (error != 0) {
    return Status::InternalError("Failed to sync directory ", dir_name,
                                 " after publishing manifest ", path, ": ",
                                 ErrorText(error));
  }
  return Status::OK();
#else
  // Untested on Windows. The data is made durable by FlushFileBuffers on the
  // temporary file before the move. MOVEFILE_WRITE_THROUGH makes MoveFileExW
  // wait until the move is done on disk, but Microsoft documents its flush
  // only for moves performed as a copy and delete; a same-volume rename is
  // not covered by that statement, and Windows has no directory sync.
  const std::wstring wide_tmp =
      ailego::FileHelper::PathFromUtf8(tmp_path).wstring();
  const std::wstring wide_path =
      ailego::FileHelper::PathFromUtf8(path).wstring();
  DWORD error = WriteFileDurably(wide_tmp, data);
  if (error == 0 &&
      !::MoveFileExW(wide_tmp.c_str(), wide_path.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
    error = ::GetLastError();
  }
  if (error != 0) {
    ::DeleteFileW(wide_tmp.c_str());
    return Status::InternalError("Failed to write manifest ", path, ": ",
                                 ErrorText(error));
  }
  *renamed = true;
  if (InjectedFault(SyncStep::kDirectorySync) != 0) {
    return Status::InternalError("Failed to sync manifest rename ", path);
  }
  return Status::OK();
#endif
}

// Parses "manifest.<id>" and, with `tmp`, "manifest.<id>.tmp".
bool ParseManifestName(const std::string &name, bool tmp, uint64_t *id) {
  const std::string prefix =
      std::string(GetFileName(FileID::MANIFEST_FILE)) + ".";
  const std::string suffix = tmp ? ".tmp" : "";
  if (name.size() <= prefix.size() + suffix.size() ||
      name.compare(0, prefix.size(), prefix) != 0 ||
      name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0) {
    return false;
  }
  const std::string digits =
      name.substr(prefix.size(), name.size() - prefix.size() - suffix.size());
  if (digits.size() > 19 ||
      digits.find_first_not_of("0123456789") != std::string::npos) {
    return false;
  }
  *id = std::stoull(digits);
  return true;
}

}  // namespace

Status Version::Load(const std::string &path, Version *version) {
  std::ifstream ifs;
  ailego::FileHelper::OpenIfstream(ifs, path, std::ios::binary);
  if (!ifs.is_open()) {
    LOG_ERROR("Failed to open file: %s", path.c_str());
    return Status::InternalError("Failed to open file");
  }

  // Manifests are small (kilobytes), so reading the whole file at once keeps
  // the decoder simple.
  std::ostringstream buffer;
  buffer << ifs.rdbuf();
  const std::string encoded = buffer.str();

  ManifestData manifest;
  auto status = ManifestCodec::Decode(encoded, &manifest);
  if (!status.ok()) {
    LOG_ERROR("Failed to parse manifest from file: %s", path.c_str());
    return Status::InternalError("Failed to parse manifest");
  }

  version->set_schema(*manifest.schema);

  version->set_enable_mmap(manifest.enable_mmap);

  for (auto &meta : manifest.persisted_segment_metas) {
    status = version->add_persisted_segment_meta(meta);
    if (!status.ok()) {
      LOG_ERROR("Duplicate segment %u in manifest file: %s", meta->id(),
                path.c_str());
      return Status::InternalError("Duplicate segment in manifest ", path);
    }
  }

  if (manifest.writing_segment_meta) {
    version->reset_writing_segment_meta(manifest.writing_segment_meta);
  }

  version->set_id_map_path_suffix(manifest.id_map_path_suffix);
  version->set_delete_snapshot_path_suffix(
      manifest.delete_snapshot_path_suffix);

  version->set_next_segment_id(manifest.next_segment_id);

  return Status::OK();
}

Status Version::Save(const std::string &path, const Version &version,
                     bool *uncertain) {
  ManifestData manifest;
  manifest.schema = std::make_shared<CollectionSchema>(version.schema());
  manifest.enable_mmap = version.enable_mmap();
  manifest.persisted_segment_metas = version.persisted_segment_metas();
  manifest.writing_segment_meta = version.writing_segment_meta();
  manifest.id_map_path_suffix = version.id_map_path_suffix();
  manifest.delete_snapshot_path_suffix = version.delete_snapshot_path_suffix();
  manifest.next_segment_id = version.next_segment_id();

  std::string encoded;
  auto status = ManifestCodec::Encode(manifest, &encoded);
  if (!status.ok()) {
    LOG_ERROR("Failed to serialize manifest to file: %s", path.c_str());
    return Status::InternalError("Failed to serialize manifest to file");
  }

  // Readers must only ever see a complete manifest under its final name.
  bool renamed = false;
  status = PublishFile(path + ".tmp", path, encoded, &renamed);
  if (!status.ok()) {
    LOG_ERROR("%s", status.message().c_str());
  }
  if (uncertain != nullptr) {
    *uncertain = !status.ok() && renamed;
  }
  return status;
}

Status Version::validate() const {
  // An absent schema decodes to an empty placeholder (see ManifestCodec).
  if (!schema_ || schema_->fields().empty()) {
    return Status::InternalError("Manifest has no collection schema");
  }
  if (!writing_segment_meta_) {
    return Status::InternalError("Manifest has no writing segment");
  }
  SegmentID max_id = writing_segment_meta_->id();
  for (const auto &[id, meta] : persisted_segment_metas_map_) {
    if (id == writing_segment_meta_->id()) {
      return Status::InternalError("Manifest lists segment ", id,
                                   " as both persisted and writing");
    }
    max_id = (std::max)(max_id, id);
  }
  if (next_segment_id_ <= max_id) {
    return Status::InternalError("Manifest next_segment_id ", next_segment_id_,
                                 " is not above segment id ", max_id);
  }
  return Status::OK();
}

std::string Version::to_string() const {
  std::ostringstream oss;
  oss << "Version{" << "schema:" << (schema_ ? schema_->to_string() : "null")
      << ",persisted_segment_metas:[";

  size_t i = 0;
  for (const auto &pair : persisted_segment_metas_map_) {
    if (i > 0) oss << ",";
    oss << pair.second->to_string();
    ++i;
  }

  oss << "],writing_segment_meta:";
  if (writing_segment_meta_) {
    oss << writing_segment_meta_->to_string();
  } else {
    oss << "null";
  }

  oss << ",id_map_path_suffix:" << id_map_path_suffix_
      << ",delete_snapshot_path_suffix:" << delete_snapshot_path_suffix_
      << ",next_segment_id:" << next_segment_id_
      << ",enable_mmap:" << enable_mmap_ << "}";
  return oss.str();
}

std::string Version::to_string_formatted(int indent_level) const {
  std::ostringstream oss;
  oss << indent(indent_level) << "Version{\n"
      << indent(indent_level + 1) << "schema: ";

  if (schema_) {
    oss << "\n" << schema_->to_string_formatted(indent_level + 2) << "\n";
  } else {
    oss << "null\n";
  }

  oss << indent(indent_level + 1) << "persisted_segment_metas: [\n";

  size_t i = 0;
  for (const auto &pair : persisted_segment_metas_map_) {
    oss << pair.second->to_string_formatted(indent_level + 2);
    if (i < persisted_segment_metas_map_.size() - 1) {
      oss << ",";
    }
    oss << "\n";
    ++i;
  }

  oss << "\n"
      << indent(indent_level + 1) << "],\n"
      << indent(indent_level + 1) << "writing_segment_meta: ";

  if (writing_segment_meta_) {
    oss << "\n"
        << writing_segment_meta_->to_string_formatted(indent_level + 2) << "\n";
  } else {
    oss << "null\n";
  }

  oss << indent(indent_level + 1)
      << "id_map_path_suffix: " << id_map_path_suffix_ << ",\n"
      << indent(indent_level + 1)
      << "delete_snapshot_path_suffix: " << delete_snapshot_path_suffix_
      << ",\n"
      << indent(indent_level + 1) << "next_segment_id: " << next_segment_id_
      << "\n"
      << indent(indent_level + 1) << "enable_mmap: " << enable_mmap_ << "\n"
      << indent(indent_level) << "}";
  return oss.str();
}

Result<VersionManager::Ptr> VersionManager::Recovery(const std::string &path,
                                                     bool validate) {
  auto u8path = ailego::FileHelper::PathFromUtf8(path);
  if (!fs::exists(u8path)) {
    LOG_ERROR("VersionManager::Recovery: path %s does not exist", path.c_str());
    return tl::make_unexpected(
        Status::NotFound("path ", path, " does not exist"));
  }
  if (!fs::is_directory(u8path)) {
    LOG_ERROR("VersionManager::Recovery: path %s is not a directory",
              path.c_str());
    return tl::make_unexpected(
        Status::InvalidArgument("path", path, " is not a directory"));
  }

  // "manifest.<id>.tmp" files are never candidates: they are unpublished and
  // removed by the next flush().
  uint64_t max_id = 0;
  std::string version_path;
  for (const auto &entry : fs::directory_iterator(u8path)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    uint64_t id = 0;
    if (ParseManifestName(
            ailego::FileHelper::PathToUtf8(entry.path().filename()), false,
            &id) &&
        (version_path.empty() || id > max_id)) {
      max_id = id;
      version_path = ailego::FileHelper::PathToUtf8(entry.path());
    }
  }
  if (version_path.empty()) {
    LOG_ERROR("Failed to find the version file in collction_path(%s)",
              path.c_str());
    return tl::make_unexpected(
        Status::NotFound("Failed to find the version file"));
  }

  Version version;
  auto s = Version::Load(version_path, &version);
  if (s.ok() && validate) {
    s = version.validate();
  }
  if (!s.ok()) {
    LOG_ERROR("Unusable manifest %s: %s", version_path.c_str(),
              s.message().c_str());
    return tl::make_unexpected(
        Status::InternalError("Unusable manifest ", version_path, ": ",
                              s.message(), "; nothing was modified"));
  }

  VersionManager::Ptr manager(new VersionManager(path, version, max_id + 1));
  manager->loaded_manifest_path_ = version_path;
  return manager;
}

Result<VersionManager::Ptr> VersionManager::Create(
    const std::string &path, const Version &initial_version) {
  VersionManager::Ptr manager =
      VersionManager::Ptr(new VersionManager(path, initial_version));
  return manager;
}

VersionManager::VersionManager(const std::string &path,
                               const Version &initial_version,
                               uint64_t version_id)
    : path_(path), current_version_(initial_version), version_id_(version_id) {}

Version VersionManager::get_current_version() {
  std::lock_guard lock(mtx_);
  return current_version_;
}

Status VersionManager::apply(const Version &version) {
  std::lock_guard lock(mtx_);
  current_version_ = version;
  return Status::OK();
}

Status VersionManager::reset_writing_segment_meta(SegmentMeta::Ptr meta) {
  std::lock_guard lock(mtx_);
  current_version_.reset_writing_segment_meta(meta);
  return Status::OK();
}

Status VersionManager::add_persisted_segment_meta(SegmentMeta::Ptr meta) {
  std::lock_guard lock(mtx_);
  return current_version_.add_persisted_segment_meta(meta);
}

Status VersionManager::remove_persisted_segment_meta(SegmentID id) {
  std::lock_guard lock(mtx_);
  return current_version_.remove_persisted_segment_meta(id);
}

Status VersionManager::flush() {
  std::lock_guard lock(mtx_);

  if (fenced_.load()) {
    return Status::FailedPrecondition(
        "A previous manifest publish in ", path_,
        " could not be confirmed durable; reopen the collection");
  }

  // version_id_ only advances once the manifest is published, so a failed
  // flush leaves every existing manifest in place.
  bool uncertain = false;
  auto s = Version::Save(
      FileHelper::MakeFilePath(path_, FileID::MANIFEST_FILE, version_id_),
      current_version_, &uncertain);
  if (!s.ok()) {
    if (uncertain) {
      // The new manifest may or may not survive a crash. Keep it and every
      // older generation, never reuse its name, and publish nothing more.
      fenced_.store(true);
      ++version_id_;
    }
    return s;
  }
  const uint64_t published = version_id_++;

  // The new manifest is durable: every older generation and every leftover
  // temporary file is obsolete. A file that cannot be removed only wastes
  // space, since recovery loads the newest manifest.
  std::error_code ec;
  fs::directory_iterator it(ailego::FileHelper::PathFromUtf8(path_), ec);
  std::vector<fs::path> obsolete;
  for (; !ec && it != fs::directory_iterator(); it.increment(ec)) {
    const std::string name =
        ailego::FileHelper::PathToUtf8(it->path().filename());
    uint64_t id = 0;
    if (ParseManifestName(name, true, &id) ||
        (ParseManifestName(name, false, &id) && id < published)) {
      obsolete.push_back(it->path());
    }
  }
  for (const auto &file : obsolete) {
    FileHelper::RemoveFile(ailego::FileHelper::PathToUtf8(file));
  }

  return Status::OK();
}

Status VersionManager::sync_current_manifest() {
  std::lock_guard lock(mtx_);
  if (loaded_manifest_path_.empty()) {
    return Status::OK();
  }
  auto s = SyncPublishedFile(loaded_manifest_path_);
  if (!s.ok()) {
    LOG_ERROR("%s", s.message().c_str());
  }
  return s;
}

void VersionManager::SetSyncFaultForTest(SyncFaultForTest fault) {
  g_sync_fault.store(fault);
}

}  // namespace zvec