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
#include <charconv>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <zvec/ailego/logger/logger.h>
#include <zvec/ailego/pattern/expected.hpp>
#include <zvec/ailego/utility/string_helper.h>
#include <zvec/db/status.h>
#include "db/common/file_helper.h"
#include "db/common/typedef.h"
#include "db/index/column/fts_column/fts_file_lifetime.h"
#include "db/index/column/fts_column/fts_format.h"
#include "db/index/common/manifest_codec.h"
#include "db/index/common/type_helper.h"

namespace zvec {
namespace {

constexpr size_t kManifestLimit = 64 * 1024 * 1024;
constexpr char kCommitMagic[] = "ZVEC_MANIFEST_CURRENT_V1";
constexpr size_t kCommitBytes = sizeof(kCommitMagic) + 8 + 32 + 32;

std::string AuthorityPath(const std::string &path) {
  return ailego::FileHelper::PathJoin(path, "manifest.current");
}

Result<std::string> ReadBytes(const std::string &path, size_t limit) {
  ailego::File file;
  if (!file.open(path, true)) {
    return tl::make_unexpected(
        Status{StatusCode::UNAVAILABLE, "Cannot read " + path});
  }
  const size_t size = file.size();
  if (size > limit) {
    return tl::make_unexpected(
        Status{StatusCode::RESOURCE_EXHAUSTED, "Manifest exceeds read budget"});
  }
  std::string bytes(size, '\0');
  if (file.read(bytes.data(), bytes.size()) != bytes.size()) {
    return tl::make_unexpected(
        Status{StatusCode::UNAVAILABLE, "Short read of " + path});
  }
  return bytes;
}

Result<std::string> EncodeVersion(const Version &version, bool gated = false) {
  ManifestData manifest;
  manifest.requires_commit_record = gated || version.requires_commit_record();
  manifest.schema = std::make_shared<CollectionSchema>(version.schema());
  manifest.enable_mmap = version.enable_mmap();
  manifest.persisted_segment_metas = version.persisted_segment_metas();
  manifest.writing_segment_meta = version.writing_segment_meta();
  manifest.id_map_path_suffix = version.id_map_path_suffix();
  manifest.delete_snapshot_path_suffix = version.delete_snapshot_path_suffix();
  manifest.next_segment_id = version.next_segment_id();
  std::string bytes;
  auto status = ManifestCodec::Encode(manifest, &bytes);
  if (!status.ok()) {
    return tl::make_unexpected(status);
  }
  if (bytes.size() > kManifestLimit) {
    return tl::make_unexpected(Status{StatusCode::RESOURCE_EXHAUSTED,
                                      "Manifest exceeds write budget"});
  }
  return bytes;
}

fts::format::Digest Digest(std::string_view bytes) {
  // Every caller has already bounded input below the SHA-256 size limit.
  fts::format::Sha256 hash;
  hash.update(bytes);
  return hash.digest();
}

std::string EncodeCommit(uint32_t id, std::string_view manifest) {
  std::string bytes(kCommitMagic, sizeof(kCommitMagic));
  fts::format::AppendU64(id, &bytes);
  auto digest = Digest(manifest);
  bytes.append(reinterpret_cast<const char *>(digest.data()), digest.size());
  digest = Digest(bytes);
  bytes.append(reinterpret_cast<const char *>(digest.data()), digest.size());
  return bytes;
}

struct CommitRecord {
  uint32_t id;
  fts::format::Digest manifest_digest;
};

Result<CommitRecord> DecodeCommit(std::string_view bytes) {
  if (bytes.size() != kCommitBytes ||
      bytes.substr(0, sizeof(kCommitMagic)) !=
          std::string_view(kCommitMagic, sizeof(kCommitMagic))) {
    return tl::make_unexpected(Status::InternalError("Invalid commit record"));
  }
  auto expected = Digest(bytes.substr(0, bytes.size() - 32));
  if (bytes.substr(bytes.size() - 32) !=
      std::string_view(reinterpret_cast<const char *>(expected.data()),
                       expected.size())) {
    return tl::make_unexpected(
        Status::InternalError("Commit checksum mismatch"));
  }
  fts::format::Decoder decoder(bytes.substr(sizeof(kCommitMagic)));
  const uint64_t id = decoder.u64();
  if (id > UINT32_MAX) {
    return tl::make_unexpected(
        Status::NotSupported("Manifest ID exceeds v1 range"));
  }
  CommitRecord record{static_cast<uint32_t>(id), {}};
  auto digest = decoder.take(record.manifest_digest.size());
  std::copy(digest.begin(), digest.end(), record.manifest_digest.begin());
  return record;
}

}  // namespace

Version Version::clone() const {
  Version copy = *this;
  if (schema_) {
    copy.schema_ = std::make_shared<CollectionSchema>(*schema_);
  }
  for (auto &[id, meta] : copy.persisted_segment_metas_map_) {
    meta = std::make_shared<SegmentMeta>(*meta);
  }
  if (writing_segment_meta_) {
    copy.writing_segment_meta_ =
        std::make_shared<SegmentMeta>(*writing_segment_meta_);
  }
  return copy;
}

Status Version::Load(const std::string &path, Version *version) {
  // Manifests are small (usually kilobytes); bound a corrupt file's allocation.
  auto encoded = ReadBytes(path, kManifestLimit);
  if (!encoded) {
    return encoded.error();
  }

  ManifestData manifest;
  auto status = ManifestCodec::Decode(*encoded, &manifest);
  if (!status.ok()) {
    LOG_ERROR("Failed to parse manifest from file: %s", path.c_str());
    return status;
  }

  Version candidate;
  Version *destination = version;
  version = &candidate;
  version->requires_commit_record_ = manifest.requires_commit_record;
  version->set_schema(*manifest.schema);

  version->set_enable_mmap(manifest.enable_mmap);

  for (auto &meta : manifest.persisted_segment_metas) {
    version->add_persisted_segment_meta(meta);
  }

  if (manifest.writing_segment_meta) {
    version->reset_writing_segment_meta(manifest.writing_segment_meta);
  }

  version->set_id_map_path_suffix(manifest.id_map_path_suffix);
  version->set_delete_snapshot_path_suffix(
      manifest.delete_snapshot_path_suffix);

  version->set_next_segment_id(manifest.next_segment_id);
  *destination = std::move(candidate);

  return Status::OK();
}

Status Version::Save(const std::string &path, const Version &version) {
  auto encoded = EncodeVersion(version);
  if (!encoded) {
    return encoded.error();
  }
  return DurableFile::Replace(path, *encoded).status;
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

  std::error_code exists_error;
  const bool has_authority = fs::exists(
      ailego::FileHelper::PathFromUtf8(AuthorityPath(path)), exists_error);
  if (exists_error) {
    return tl::make_unexpected(
        Status::InternalError("Cannot inspect manifest authority"));
  }
  if (has_authority) {
    auto bytes = ReadBytes(AuthorityPath(path), kCommitBytes);
    if (!bytes) {
      return tl::make_unexpected(bytes.error());
    }
    auto record = DecodeCommit(*bytes);
    if (!record) {
      return tl::make_unexpected(record.error());
    }
    const auto manifest_path =
        FileHelper::MakeFilePath(path, FileID::MANIFEST_FILE, record->id);
    auto manifest = ReadBytes(manifest_path, kManifestLimit);
    if (!manifest) {
      return tl::make_unexpected(manifest.error());
    }
    if (Digest(*manifest) != record->manifest_digest) {
      return tl::make_unexpected(
          Status::InternalError("Committed manifest checksum mismatch"));
    }
    Version version;
    auto status = Version::Load(manifest_path, &version);
    CHECK_RETURN_STATUS_EXPECTED(status);
    auto manager =
        Ptr(new VersionManager(path, version, uint64_t(record->id) + 1));
    manager->committed_id_ = record->id;
    manager->explicit_authority_ = true;
    manager->pin_publication(&manager->current_version_, record->id);
    return manager;
  }

  std::string prefix = GetFileName(FileID::MANIFEST_FILE);
  std::string manifest_pattern = "^" + prefix + R"(\.(\d+)$)";
  std::regex regex(manifest_pattern);
  std::smatch match;

  uint64_t max_id = UINT64_MAX;
  std::string version_path;

  for (const auto &entry : fs::directory_iterator(u8path)) {
    if (entry.is_regular_file()) {
      std::string filename = entry.path().filename().u8string();
      if (std::regex_match(filename, match, regex)) {
        const std::string number = match[1].str();
        uint64_t id = 0;
        const auto parsed =
            std::from_chars(number.data(), number.data() + number.size(), id);
        if (parsed.ec != std::errc{} || id > UINT32_MAX) {
          return tl::make_unexpected(
              Status::NotSupported("Manifest ID exceeds v1 range"));
        }
        if (id > max_id || max_id == UINT64_MAX) {
          max_id = id;
          version_path = entry.path().u8string();
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
  if (version.requires_commit_record()) {
    return tl::make_unexpected(Status::FailedPrecondition(
        "Gated manifest has no authoritative commit record"));
  }

  VersionManager::Ptr manager =
      VersionManager::Ptr(new VersionManager(path, version, max_id + 1));
  manager->committed_id_ = static_cast<uint32_t>(max_id);
  manager->pin_publication(&manager->current_version_,
                           static_cast<uint32_t>(max_id));

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
    : path_(path),
      current_version_(initial_version.clone()),
      version_id_(version_id),
      fts_retirement_(fts::FtsRetirementQueue::ForDirectory(path)) {}

void VersionManager::pin_publication(Version *version, uint32_t manifest_id) {
  version->manifest_owner_ = fts_retirement_->track(
      FileHelper::MakeFilePath(path_, FileID::MANIFEST_FILE, manifest_id),
      false, false);
  version->delete_owner_ = fts_retirement_->track(
      FileHelper::MakeFilePath(path_, FileID::DELETE_FILE,
                               version->delete_snapshot_path_suffix()),
      false, false);
}

void VersionManager::publish_version(Version prepared, uint32_t manifest_id) {
  prepared.manifest_owner_->preserve();
  prepared.delete_owner_->preserve();
  if (current_version_.manifest_owner_ &&
      current_version_.manifest_owner_ != prepared.manifest_owner_) {
    current_version_.manifest_owner_->retire();
  }
  if (current_version_.delete_owner_ &&
      current_version_.delete_owner_ != prepared.delete_owner_) {
    current_version_.delete_owner_->retire();
  }
  current_version_ = std::move(prepared);
  committed_id_ = manifest_id;
}

Version VersionManager::get_current_version() {
  std::lock_guard lock(mtx_);
  return current_version_.clone();
}

Status VersionManager::apply(const Version &version) {
  std::lock_guard lock(mtx_);
  if (requires_recovery_) {
    return Status::FailedPrecondition(
        "Uncertain manifest commit; reopen required");
  }
  current_version_ = version.clone();
  return Status::OK();
}

Status VersionManager::reset_writing_segment_meta(SegmentMeta::Ptr meta) {
  std::lock_guard lock(mtx_);
  if (requires_recovery_) {
    return Status::FailedPrecondition(
        "Uncertain manifest commit; reopen required");
  }
  current_version_.reset_writing_segment_meta(
      meta ? std::make_shared<SegmentMeta>(*meta) : nullptr);
  return Status::OK();
}

Status VersionManager::add_persisted_segment_meta(SegmentMeta::Ptr meta) {
  std::lock_guard lock(mtx_);
  if (requires_recovery_) {
    return Status::FailedPrecondition(
        "Uncertain manifest commit; reopen required");
  }
  return current_version_.add_persisted_segment_meta(
      meta ? std::make_shared<SegmentMeta>(*meta) : nullptr);
}

Status VersionManager::remove_persisted_segment_meta(SegmentID id) {
  std::lock_guard lock(mtx_);
  if (requires_recovery_) {
    return Status::FailedPrecondition(
        "Uncertain manifest commit; reopen required");
  }
  return current_version_.remove_persisted_segment_meta(id);
}

Result<uint32_t> VersionManager::reserve_version_id() {
  // An orphan's filename is used only to avoid overwriting immutable files,
  // never to infer commitment. Recovery authority comes from manifest.current.
  while (version_id_ <= UINT32_MAX) {
    const auto id = static_cast<uint32_t>(version_id_++);
    const auto name =
        FileHelper::MakeFilePath(path_, FileID::MANIFEST_FILE, id);
    std::error_code error;
    const bool exists =
        std::filesystem::exists(ailego::FileHelper::PathFromUtf8(name), error);
    if (error) {
      return tl::make_unexpected(
          Status::InternalError("Cannot reserve manifest ID"));
    }
    if (!exists) {
      return id;
    }
  }
  return tl::make_unexpected(
      Status{StatusCode::RESOURCE_EXHAUSTED, "Manifest ID space exhausted"});
}

Status VersionManager::flush() {
  std::lock_guard lock(mtx_);
  return checkpoint_locked(current_version_, {});
}

Status VersionManager::checkpoint(const Version &candidate,
                                  const DurableFile::Hook &hook) {
  std::lock_guard lock(mtx_);
  return checkpoint_locked(candidate, hook);
}

Status VersionManager::checkpoint_locked(const Version &candidate,
                                         const DurableFile::Hook &hook) {
  if (requires_recovery_) {
    return Status::FailedPrecondition(
        "Uncertain manifest commit; reopen required");
  }
  if (explicit_authority_) {
    return commit_locked(candidate, hook);
  }
  Version prepared = candidate.clone();
  auto encoded = EncodeVersion(prepared);
  if (!encoded) {
    return encoded.error();
  }
  if (!encoded->empty() && encoded->front() == '\0') {
    return commit_locked(candidate, hook);
  }
  auto id = reserve_version_id();
  if (!id) {
    return id.error();
  }
  pin_publication(&prepared, *id);
  auto outcome = DurableFile::Replace(
      FileHelper::MakeFilePath(path_, FileID::MANIFEST_FILE, *id), *encoded,
      hook);
  if (!outcome.status.ok()) {
    requires_recovery_ = outcome.renamed;
    return outcome.status;
  }
  publish_version(std::move(prepared), *id);
  return Status::OK();
}

bool VersionManager::requires_recovery() const {
  std::lock_guard lock(mtx_);
  return requires_recovery_;
}

Status VersionManager::commit(const Version &candidate,
                              const DurableFile::Hook &hook) {
  std::lock_guard lock(mtx_);
  return commit_locked(candidate, hook);
}

Status VersionManager::commit_locked(const Version &candidate,
                                     const DurableFile::Hook &hook) {
  if (requires_recovery_) {
    return Status::FailedPrecondition(
        "Uncertain manifest commit; reopen required");
  }
  if (!committed_id_) {
    return Status::FailedPrecondition(
        "A recoverable source checkpoint is required");
  }
  // Allocate independent metadata and encode everything before publishing the
  // durable authority. Only infallible moves/scalar stores follow the commit.
  Version prepared = candidate.clone();
  prepared.requires_commit_record_ = true;
  auto encoded = EncodeVersion(prepared, true);
  if (!encoded) {
    return encoded.error();
  }
  auto id = reserve_version_id();
  if (!id) {
    return id.error();
  }
  const auto manifest_path =
      FileHelper::MakeFilePath(path_, FileID::MANIFEST_FILE, *id);
  const auto authority_path = AuthorityPath(path_);
  const auto record = EncodeCommit(*id, *encoded);
  pin_publication(&prepared, *id);

  if (!explicit_authority_) {
    // Bootstrap authority to the complete legacy checkpoint BEFORE creating
    // any numbered candidate. A crash can then select only the old checkpoint
    // or a fully committed new version, even when a candidate has a higher ID.
    auto baseline = ReadBytes(
        FileHelper::MakeFilePath(path_, FileID::MANIFEST_FILE, *committed_id_),
        kManifestLimit);
    if (!baseline) {
      return baseline.error();
    }
    const auto old_record = EncodeCommit(*committed_id_, *baseline);
    auto outcome = DurableFile::Replace(authority_path, old_record, hook);
    if (!outcome.status.ok()) {
      requires_recovery_ = outcome.renamed;
      return outcome.status;
    }
    explicit_authority_ = true;
  }

  auto outcome = DurableFile::Replace(manifest_path, *encoded, hook);
  if (!outcome.status.ok()) {
    // The old authority is still durable. A prepared file is not a commit.
    return outcome.status;
  }
  outcome = DurableFile::Replace(authority_path, record, hook);
  if (!outcome.status.ok()) {
    requires_recovery_ = outcome.renamed;
    return outcome.status;
  }
  publish_version(std::move(prepared), *id);
  return Status::OK();
}

}  // namespace zvec
