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
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <regex>
#include <string>
#include <ailego/logger/logger.h>
#include <proto/zvec.pb.h>
#include <zvec/ailego/pattern/expected.hpp>
#include <zvec/ailego/utility/string_helper.h>
#include <zvec/db/status.h>
#include "db/common/file_helper.h"
#include "db/common/typedef.h"
#include "db/index/common/proto_converter.h"
#include "db/index/common/type_helper.h"

namespace zvec {

Status Version::Load(const std::string &path, Version *version) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs.is_open()) {
    LOG_ERROR("Failed to open file: %s", path.c_str());
    return Status::InternalError("Failed to open file");
  }

  proto::Manifest manifest;

  if (!manifest.ParseFromIstream(&ifs)) {
    LOG_ERROR("Failed to parse manifest from file: %s", path.c_str());
    return Status::InternalError("Failed to parse manifest");
  }

  CollectionSchema::Ptr schema = ProtoConverter::FromPb(manifest.schema());
  version->set_schema(*schema);

  version->set_enable_mmap(manifest.enable_mmap());

  for (int i = 0; i < manifest.persisted_segment_metas_size(); ++i) {
    SegmentMeta::Ptr meta =
        ProtoConverter::FromPb(manifest.persisted_segment_metas(i));
    version->add_persisted_segment_meta(meta);
  }

  if (manifest.has_writing_segment_meta()) {
    SegmentMeta::Ptr meta =
        ProtoConverter::FromPb(manifest.writing_segment_meta());
    version->reset_writing_segment_meta(meta);
  }

  version->set_id_map_path_suffix(manifest.id_map_path_suffix());
  version->set_delete_snapshot_path_suffix(
      manifest.delete_snapshot_path_suffix());

  version->set_next_segment_id(manifest.next_segment_id());

  return Status::OK();
}

Status Version::Save(const std::string &path, const Version &version) {
  std::ofstream ofs(path, std::ios::binary);
  if (!ofs.is_open()) {
    LOG_ERROR("Failed to open file: %s, err: %s", path.c_str(),
              strerror(errno));
    return Status::InternalError("Failed to open file: %s", path.c_str());
  }

  proto::Manifest manifest;

  // set schema
  auto schema = ProtoConverter::ToPb(version.schema());
  manifest.mutable_schema()->Swap(&schema);

  manifest.set_enable_mmap(version.enable_mmap());

  // set segments meta
  for (auto &meta : version.persisted_segment_metas()) {
    auto meta_pb = ProtoConverter::ToPb(*meta);
    manifest.add_persisted_segment_metas()->Swap(&meta_pb);
  }

  if (version.writing_segment_meta()) {
    auto meta_pb = ProtoConverter::ToPb(*version.writing_segment_meta());
    manifest.mutable_writing_segment_meta()->Swap(&meta_pb);
  }

  manifest.set_id_map_path_suffix(version.id_map_path_suffix());
  manifest.set_delete_snapshot_path_suffix(
      version.delete_snapshot_path_suffix());
  manifest.set_next_segment_id(version.next_segment_id());

  if (!manifest.SerializeToOstream(&ofs)) {
    LOG_ERROR("Failed to serialize manifest to file: %s", path.c_str());
    return Status::InternalError("Failed to serialize manifest to file");
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

Result<VersionManager::Ptr> VersionManager::Recovery(const std::string &path) {
  namespace fs = std::filesystem;
  if (!fs::exists(path)) {
    LOG_ERROR("VersionManager::Recovery: path %s does not exist", path.c_str());
    return tl::make_unexpected(
        Status::NotFound("path ", path, " does not exist"));
  }
  if (!fs::is_directory(path)) {
    LOG_ERROR("VersionManager::Recovery: path %s is not a directory",
              path.c_str());
    return tl::make_unexpected(
        Status::InvalidArgument("path", path, " is not a directory"));
  }

  std::string prefix = GetFileName(FileID::MANIFEST_FILE);
  std::string manifest_pattern = "^" + prefix + R"(\.(\d+)$)";
  std::regex regex(manifest_pattern);
  std::smatch match;

  uint64_t max_id = UINT64_MAX;
  std::string version_path;

  for (const auto &entry : fs::directory_iterator(path)) {
    if (entry.is_regular_file()) {
      std::string filename = entry.path().filename().string();
      if (std::regex_match(filename, match, regex)) {
        uint64_t id = std::stoull(match[1].str());
        if (id > max_id || max_id == UINT64_MAX) {
          max_id = id;
          version_path = entry.path().string();
        }
      }
    }
  }

  if (max_id == UINT64_MAX) {
    LOG_ERROR("Failed to find the version file in collction_path(%s)",
              path.c_str());
    return tl::make_unexpected(
        Status::NotFound("Failed to find the version file"));
  }

  Version version;
  auto s = Version::Load(version_path, &version);
  CHECK_RETURN_STATUS_EXPECTED(s);

  VersionManager::Ptr manager =
      VersionManager::Ptr(new VersionManager(path, version, max_id + 1));

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

  std::string current_path;
  if (version_id_ != 0) {
    current_path =
        FileHelper::MakeFilePath(path_, FileID::MANIFEST_FILE, version_id_ - 1);
  }

  auto s = Version::Save(
      FileHelper::MakeFilePath(path_, FileID::MANIFEST_FILE, version_id_++),
      current_version_);
  CHECK_RETURN_STATUS(s);

  if (!current_path.empty()) {
    FileHelper::RemoveFile(current_path);
  }

  return Status::OK();
}


}  // namespace zvec