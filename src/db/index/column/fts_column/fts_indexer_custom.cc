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

#include <algorithm>
#include <filesystem>
#include <set>
#include <rocksdb/env.h>
#include <zvec/core/framework/index_factory.h>
#include "db/common/file_helper.h"
#include "db/index/common/manifest_codec.h"
#include "fts_indexer.h"
#include "fts_mutable_snapshot.h"

namespace zvec {
namespace {

using Identity = fts::format::Identity;

// Shared range admission for this container, not an allocation or a process
// RSS limit. Multi-term/concurrent queries retain independent posting, skip
// and doclen pages; 1 MiB admits too few on hosts with 16 KiB native pages.
// Backend cache and verified copies remain charged to the global pool.
constexpr size_t kReaderRangeBudget = 64 * 1024 * 1024;

Result<Identity> NewIdentity() {
  // Reuse RocksDB's portable UUID provider. Hash to fixed-width opaque bytes;
  // the exclusive reservation below also prevents accidental path reuse.
  fts::format::Sha256 hash;
  auto status = hash.update(rocksdb::Env::Default()->GenerateUniqueId());
  if (!status.ok()) {
    return tl::make_unexpected(status);
  }
  Identity identity;
  const auto digest = hash.digest();
  std::copy_n(digest.begin(), identity.size(), identity.begin());
  return identity;
}

std::string AnalysisIdentity(const FieldSchema &field) {
  std::string bytes;
  ManifestCodec::EncodeIndexParams(field.index_params().get(), &bytes);
  return bytes;
}

Status ValidateFields(const FtsIndexer::CustomTarget &target,
                      const FieldSchemaPtrList &fields) {
  if (target.segment_path.empty() ||
      target.block.type() != BlockType::FTS_INDEX || fields.empty() ||
      fields.size() > fts::format::kMaxFields ||
      fields.size() != target.block.columns().size()) {
    return Status::InvalidArgument("Invalid custom FTS target");
  }
  std::set<std::string> names;
  // Preflight all retained identities before serializing or creating files.
  // Reserve fixed base/field/section records, then charge protobuf string
  // framing as well as contents (many empty filters still consume space).
  size_t remaining = fts::format::kMetadataBudget - 256;
  auto charge = [&](size_t bytes) {
    if (bytes > remaining) {
      return false;
    }
    remaining -= bytes;
    return true;
  };
  auto charge_string = [&](const std::string &value) {
    return charge(20) && charge(value.size());
  };
  auto budget_error = []() {
    return Status{StatusCode::RESOURCE_EXHAUSTED,
                  "Custom FTS configuration exceeds metadata budget"};
  };
  for (const auto &field : fields) {
    if (!field || field->index_type() != IndexType::FTS ||
        field->name().empty()) {
      return Status::InvalidArgument("Custom FTS fields disagree with block");
    }
    const auto *params =
        dynamic_cast<const FtsIndexParams *>(field->index_params().get());
    if (!params) {
      return Status::InvalidArgument("Missing FTS analysis configuration");
    }
    if (!charge(1024) || !charge_string(field->name()) ||
        !charge_string(params->tokenizer_name()) ||
        !charge_string(params->extra_params())) {
      return budget_error();
    }
    for (const auto &filter : params->filters()) {
      if (!charge_string(filter)) {
        return budget_error();
      }
    }
    if (!target.block.contain_column(field->name()) ||
        !names.insert(field->name()).second) {
      return Status::InvalidArgument("Custom FTS fields disagree with block");
    }
  }
  return Status::OK();
}

std::string CustomPath(const FtsIndexer::CustomTarget &target) {
  const auto &base = *target.block.fts_base_;
  return FileHelper::MakeCustomFtsPath(target.segment_path, target.block.id(),
                                       base.generation, base.base_id);
}

}  // namespace

Result<FtsIndexer::Ptr> FtsIndexer::prepare_custom(
    const CustomTarget &target, const FieldSchemaPtrList &fields,
    const fts::FtsRetirementQueue::Pointer &retirement,
    const DurableFile::Hook &hook) const {
  auto status = ValidateFields(target, fields);
  if (!status.ok()) {
    return tl::make_unexpected(status);
  }
  if (!retirement || !fts_ctx_ || custom_block_ ||
      fields.size() != indexers_.size() || target.block.fts_base_) {
    return tl::make_unexpected(Status::InvalidArgument(
        "Custom seal requires the complete mutable source"));
  }
  for (const auto &field : fields) {
    auto source = get(field->name());
    if (!source || !source->field_schema() ||
        !source->field_schema()->index_params() ||
        !(*source->field_schema()->index_params() == *field->index_params())) {
      return tl::make_unexpected(
          Status::InvalidArgument("Custom seal source schema changed"));
    }
  }
  auto snapshot =
      fts::FtsMutableSnapshot::Capture(fts_ctx_, target.block.doc_count());
  if (!snapshot) {
    return tl::make_unexpected(snapshot.error());
  }
  return PrepareGeneration(
      target, fields, retirement,
      [&](fts::format::FieldMetadata identity, fts::FtsContainerWriter *writer,
          uint32_t ordinal) {
        return (*snapshot)->write_field(std::move(identity), {}, writer,
                                        ordinal);
      },
      hook);
}

Result<FtsIndexer::Ptr> FtsIndexer::PrepareGeneration(
    const CustomTarget &target, const FieldSchemaPtrList &fields,
    const fts::FtsRetirementQueue::Pointer &retirement,
    const FieldBuilder &build, const DurableFile::Hook &hook) {
  auto status = ValidateFields(target, fields);
  if (!status.ok()) {
    return tl::make_unexpected(status);
  }
  if (!retirement || !build || target.block.fts_base_) {
    return tl::make_unexpected(
        Status::InvalidArgument("Invalid FTS generation target"));
  }
  auto identity = NewIdentity();
  if (!identity) {
    return tl::make_unexpected(identity.error());
  }
  CustomTarget candidate = target;
  FtsBaseDescriptor base;
  base.base_id = *identity;
  base.generation = 1;
  base.row_count = target.block.doc_count();
  candidate.block.fts_base_ = base;
  const auto final_path = CustomPath(candidate);
  const auto pending_path = final_path + ".pending";
  const auto reservation_path = final_path + ".reserved";
  std::error_code error;
  if (!std::filesystem::create_directory(
          ailego::FileHelper::PathFromUtf8(reservation_path), error)) {
    return tl::make_unexpected(
        Status{StatusCode::UNAVAILABLE, "Cannot reserve unique FTS path"});
  }
  auto reservation = retirement->track(reservation_path, true, true);
  for (const auto &path : {final_path, pending_path}) {
    const bool exists =
        std::filesystem::exists(ailego::FileHelper::PathFromUtf8(path), error);
    if (error || exists) {
      return tl::make_unexpected(
          Status{StatusCode::UNAVAILABLE, "FTS candidate path already exists"});
    }
  }
  auto pending_owner = retirement->track(pending_path, false, true);
  auto final_owner = retirement->track(final_path, false, true);
  if (hook) {
    status = hook(DurableFile::Stage::kWrite, final_path);
    if (!status.ok()) {
      return tl::make_unexpected(status);
    }
  }
  // Keep the writer's native handle inside this scope so failure cleanup and
  // pre-opening only run after it closes (including on Windows).
  {
    auto dumper = core::IndexFactory::CreateDumper("FileDumper");
    if (!dumper || dumper->create(pending_path) != 0) {
      return tl::make_unexpected(
          Status{StatusCode::UNAVAILABLE, "Cannot create FTS candidate"});
    }
    auto writer = fts::FtsContainerWriter::Create(dumper);
    if (!writer) {
      return tl::make_unexpected(writer.error());
    }
    fts::format::Metadata metadata;
    metadata.base_id = base.base_id;
    metadata.base_generation = base.generation;
    metadata.block_id = target.block.id();
    metadata.source_segment_id = target.segment_id;
    metadata.row_count = base.row_count;
    for (const auto &field : fields) {
      fts::format::FieldMetadata field_identity;
      auto id = NewIdentity();
      auto incarnation = NewIdentity();
      if (!id || !incarnation) {
        return tl::make_unexpected(!id ? id.error() : incarnation.error());
      }
      field_identity.id = *id;
      field_identity.incarnation = *incarnation;
      field_identity.name = field->name();
      field_identity.analysis_identity = AnalysisIdentity(*field);
      field_identity.dictionary_generation = 1;
      auto result = build(std::move(field_identity), writer->get(),
                          static_cast<uint32_t>(metadata.fields.size()));
      if (!result) {
        return tl::make_unexpected(result.error());
      }
      metadata.fields.push_back(std::move(*result));
    }
    status = (*writer)->finish(metadata);
    if (!status.ok()) {
      return tl::make_unexpected(status);
    }
  }
  // Pre-open every reader and allocate the complete publishable owner before
  // durable publication. File handles remain valid across the sibling rename.
  auto prepared =
      OpenCustomFile(candidate, fields, pending_path, final_owner, true);
  if (!prepared) {
    return prepared;
  }
  (*prepared)->working_dir_ = final_path;
  auto published = DurableFile::PublishPrepared(pending_path, final_path, hook);
  if (!published.status.ok()) {
    // No manifest can reference this private candidate yet. Even a data-file
    // rename with uncertain directory sync is safe to discard here.
    return tl::make_unexpected(published.status);
  }
  return prepared;
}

Result<fts::FtsFieldSource::Ptr> FtsIndexer::open_source(
    const std::string &field, uint64_t rows,
    const fts::FtsPresenceSource &presence) const {
  if (!custom_block_) {
    return fts::OpenLegacyFtsFieldSource(get(field), rows, presence);
  }
  if (rows != container_->metadata().row_count) {
    return tl::make_unexpected(
        Status::InvalidArgument("Custom source row domain mismatch"));
  }
  const auto &fields = container_->metadata().fields;
  for (size_t i = 0; i < fields.size(); ++i) {
    if (fields[i].name == field) {
      auto reader =
          fts::FtsFieldReader::Open(container_, static_cast<uint32_t>(i));
      if (!reader) {
        return tl::make_unexpected(reader.error());
      }
      return fts::OpenCustomFtsFieldSource(std::move(*reader));
    }
  }
  return tl::make_unexpected(Status::NotFound("Missing custom FTS field"));
}

Result<FtsIndexer::Ptr> FtsIndexer::PrepareMerged(
    const CustomTarget &target, const FieldSchemaPtrList &fields,
    const std::vector<std::vector<fts::FtsFieldSource::Ptr>> &sources,
    const roaring::Roaring &deletions, fts::FtsMergePolicy policy,
    const fts::FtsRetirementQueue::Pointer &retirement,
    const DurableFile::Hook &hook) {
  auto status = ValidateFields(target, fields);
  if (!status.ok()) {
    return tl::make_unexpected(status);
  }
  if (fields.size() != sources.size()) {
    return tl::make_unexpected(Status::InvalidArgument("Missing FTS sources"));
  }
  for (size_t i = 0; i < fields.size(); ++i) {
    uint64_t rows = 0;
    const auto analysis = AnalysisIdentity(*fields[i]);
    if (sources[i].empty() || sources[i].size() != sources.front().size() ||
        (policy == fts::FtsMergePolicy::kRewrite &&
         (sources[i].size() != 1 || !deletions.isEmpty()))) {
      return tl::make_unexpected(
          Status::InvalidArgument("Invalid FTS rewrite"));
    }
    for (size_t segment = 0; segment < sources[i].size(); ++segment) {
      const auto &source = sources[i][segment];
      if (!source || source->metadata().name != fields[i]->name() ||
          source->metadata().analysis_identity != analysis ||
          source->statistics().rows !=
              sources.front()[segment]->statistics().rows ||
          !fts::format::ValidRange(rows, source->statistics().rows,
                                   UINT32_MAX)) {
        return tl::make_unexpected(
            Status::InvalidArgument("Incompatible FTS merge source"));
      }
      rows += source->statistics().rows;
    }
    if ((!deletions.isEmpty() && deletions.maximum() >= rows) ||
        rows - deletions.cardinality() != target.block.doc_count()) {
      return tl::make_unexpected(
          Status::InvalidArgument("FTS target disagrees with forward rows"));
    }
  }
  return PrepareGeneration(
      target, fields, retirement,
      [&](fts::format::FieldMetadata identity, fts::FtsContainerWriter *writer,
          uint32_t ordinal) {
        if (policy == fts::FtsMergePolicy::kRewrite) {
          const auto &source = sources[ordinal].front()->metadata();
          // Unchanged custom fields keep their logical incarnation. A
          // legacy field has no persisted identity and gets a fresh one.
          if (source.id != Identity{}) {
            identity.id = source.id;
            identity.incarnation = source.incarnation;
          }
        }
        return fts::MergeFtsField(sources[ordinal], deletions, policy,
                                  std::move(identity), writer, ordinal);
      },
      hook);
}

Result<FtsIndexer::Ptr> FtsIndexer::OpenCustom(
    const CustomTarget &target, const FieldSchemaPtrList &fields,
    const fts::FtsRetirementQueue::Pointer &retirement) {
  if (!retirement || !target.block.fts_base_) {
    return tl::make_unexpected(
        Status::InvalidArgument("Missing custom FTS reference"));
  }
  const auto path = CustomPath(target);
  return OpenCustomFile(target, fields, path,
                        retirement->track(path, false, false), false);
}

Result<FtsIndexer::Ptr> FtsIndexer::OpenCustomFile(
    const CustomTarget &target, const FieldSchemaPtrList &fields,
    const std::string &path, fts::FtsFileOwner::Pointer owner, bool candidate) {
  auto status = ValidateFields(target, fields);
  if (!status.ok() || !target.block.fts_base_) {
    return tl::make_unexpected(
        status.ok() ? Status::InvalidArgument("Missing custom FTS descriptor")
                    : status);
  }
  const auto &base = *target.block.fts_base_;
  if (base.format_version != 1 || base.minimum_reader_version > 1 ||
      base.minimum_reader_version == 0 || base.required_features != 0) {
    return tl::make_unexpected(
        Status::NotSupported("Unsupported custom FTS descriptor"));
  }
  auto container = fts::FtsContainerReader::Open(path, target.enable_mmap,
                                                 kReaderRangeBudget, owner);
  if (!container) {
    return tl::make_unexpected(container.error());
  }
  const auto &metadata = (*container)->metadata();
  if (base.generation == 0 || base.row_count != target.block.doc_count() ||
      metadata.base_id != base.base_id ||
      metadata.base_generation != base.generation ||
      metadata.block_id != target.block.id() ||
      metadata.source_segment_id != target.segment_id ||
      metadata.row_count != base.row_count ||
      metadata.fields.size() != fields.size() ||
      (!candidate && (*container)->metadata_digest() != base.metadata_digest)) {
    return tl::make_unexpected(
        Status::InternalError("Custom FTS identity mismatch"));
  }
  // Candidates have just been encoded by the writer; committed files bind
  // their metadata digest in the descriptor. Both use bounded open checks,
  // without a second complete pass over postings, positions and doc lengths.
  auto result = std::make_shared<FtsIndexer>(path);
  result->file_owner_ = std::move(owner);
  result->container_ = *container;
  result->custom_block_ = target.block;
  result->custom_block_->fts_base_->metadata_digest =
      (*container)->metadata_digest();
  for (size_t ordinal = 0; ordinal < metadata.fields.size(); ++ordinal) {
    const auto &persisted = metadata.fields[ordinal];
    auto field = std::find_if(
        fields.begin(), fields.end(),
        [&](const auto &item) { return item->name() == persisted.name; });
    if (field == fields.end() ||
        AnalysisIdentity(**field) != persisted.analysis_identity) {
      return tl::make_unexpected(
          Status::InternalError("Custom FTS analysis identity mismatch"));
    }
    auto reader = fts::FtsFieldReader::Open(*container, ordinal);
    if (!reader) {
      return tl::make_unexpected(reader.error());
    }
    auto indexer = std::make_shared<fts::FtsColumnIndexer>();
    auto opened = indexer->open_reader(*reader);
    if (!opened) {
      return tl::make_unexpected(opened.error());
    }
    result->indexers_.emplace(persisted.name, std::move(indexer));
  }
  return result;
}

}  // namespace zvec
