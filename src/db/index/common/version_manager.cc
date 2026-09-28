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
#include <zvec/ailego/io/file.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
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

#ifndef _WIN32
// Flushes a file or directory to stable storage. On Apple platforms fsync()
// only hands the data to the drive; F_FULLFSYNC also flushes the drive cache.
int SyncDescriptor(int fd) {
#ifdef F_FULLFSYNC
  if (::fcntl(fd, F_FULLFSYNC) == 0) {
    return 0;
  }
  // Not every file system supports F_FULLFSYNC; fall back to fsync().
#endif
  return ::fsync(fd);
}
#endif

// Writes `data` to a new file at `path` and syncs it. Every write, the sync
// and the close are checked.
Status WriteFileDurably(const std::string &path, const std::string &data) {
#ifdef _WIN32
  ailego::File file;
  if (!file.create(path.c_str(), 0) ||
      file.write(data.data(), data.size()) != data.size() || !file.flush()) {
    return Status::InternalError("Failed to write manifest ", path, ": ",
                                 ailego::FileHelper::GetLastErrorString());
  }
  file.close();
  return Status::OK();
#else
  int fd = -1;
  do {
    fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
  } while (fd < 0 && errno == EINTR);
  if (fd < 0) {
    return Status::InternalError("Failed to create manifest ", path, ": ",
                                 std::strerror(errno));
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
  if (error == 0 && SyncDescriptor(fd) != 0) {
    error = errno;
  }
  if (::close(fd) != 0 && error == 0) {
    error = errno;
  }
  if (error != 0) {
    return Status::InternalError("Failed to write manifest ", path, ": ",
                                 std::strerror(error));
  }
  return Status::OK();
#endif
}

// Makes a rename or unlink in `dir` durable.
Status SyncDirectory(const fs::path &dir) {
#ifdef _WIN32
  // Windows has no directory sync; its metadata journal orders the rename.
  (void)dir;
  return Status::OK();
#else
  const std::string name = ailego::FileHelper::PathToUtf8(dir);
  int fd = ::open(name.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    return Status::InternalError("Failed to open directory ", name, ": ",
                                 std::strerror(errno));
  }
  int error = SyncDescriptor(fd) == 0 ? 0 : errno;
  if (::close(fd) != 0 && error == 0) {
    error = errno;
  }
  if (error != 0) {
    return Status::InternalError("Failed to sync directory ", name, ": ",
                                 std::strerror(error));
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

Status Version::Save(const std::string &path, const Version &version) {
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

  // Readers must only ever see a complete manifest under its final name:
  // write and sync a temporary file, rename it into place, then sync the
  // directory so that the rename itself survives power loss.
  const std::string tmp_path = path + ".tmp";
  status = WriteFileDurably(tmp_path, encoded);
  if (status.ok() &&
      !ailego::FileHelper::RenameFile(tmp_path.c_str(), path.c_str())) {
    status = Status::InternalError("Failed to rename manifest ", tmp_path,
                                   " to ", path, ": ",
                                   ailego::FileHelper::GetLastErrorString());
  }
  if (!status.ok()) {
    LOG_ERROR("%s", status.message().c_str());
    FileHelper::RemoveFile(tmp_path);
    return status;
  }

  auto dir = ailego::FileHelper::PathFromUtf8(path).parent_path();
  status = SyncDirectory(dir.empty() ? fs::path(".") : dir);
  if (!status.ok()) {
    // The caller treats the manifest as unpublished; do not leave it behind
    // for recovery to pick up.
    LOG_ERROR("%s", status.message().c_str());
    FileHelper::RemoveFile(path);
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
    max_id = std::max(max_id, id);
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

  // Candidates, newest first. "manifest.<id>.tmp" files are never
  // candidates: they are unpublished and removed by the next flush().
  std::vector<std::pair<uint64_t, std::string>> candidates;
  for (const auto &entry : fs::directory_iterator(u8path)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    uint64_t id = 0;
    if (ParseManifestName(
            ailego::FileHelper::PathToUtf8(entry.path().filename()), false,
            &id)) {
      candidates.emplace_back(id, ailego::FileHelper::PathToUtf8(entry.path()));
    }
  }
  if (candidates.empty()) {
    LOG_ERROR("Failed to find the version file in collction_path(%s)",
              path.c_str());
    return tl::make_unexpected(
        Status::NotFound("Failed to find the version file"));
  }
  std::sort(
      candidates.begin(), candidates.end(),
      [](const auto &lhs, const auto &rhs) { return lhs.first > rhs.first; });

  for (const auto &[id, version_path] : candidates) {
    Version version;
    auto s = Version::Load(version_path, &version);
    if (s.ok() && validate) {
      s = version.validate();
    }
    if (!s.ok()) {
      LOG_WARN("Skipping unusable manifest %s: %s", version_path.c_str(),
               s.message().c_str());
      continue;
    }
    // New generations are numbered above every manifest on disk, including
    // damaged ones, so that publishing never reuses a name.
    if (id != candidates.front().first) {
      LOG_WARN("Recovered %s from older manifest %s", path.c_str(),
               version_path.c_str());
    }
    return VersionManager::Ptr(
        new VersionManager(path, version, candidates.front().first + 1));
  }

  LOG_ERROR("No usable manifest in collection path %s", path.c_str());
  return tl::make_unexpected(Status::InternalError(
      "No usable manifest in ", path, " (", candidates.size(),
      " candidates failed validation); nothing was modified"));
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

  // version_id_ only advances once the manifest is published, so a failed
  // flush leaves every existing manifest in place.
  auto s = Version::Save(
      FileHelper::MakeFilePath(path_, FileID::MANIFEST_FILE, version_id_),
      current_version_);
  CHECK_RETURN_STATUS(s);
  const uint64_t published = version_id_++;

  // The new manifest is durable: every older generation and every leftover
  // temporary file is obsolete. A file that cannot be removed only wastes
  // space, since recovery prefers the newest manifest that loads.
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


}  // namespace zvec