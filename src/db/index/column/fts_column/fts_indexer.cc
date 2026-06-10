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

#include "fts_indexer.h"
#include <zvec/ailego/logger/logger.h>
#include "db/common/constants.h"
#include "db/common/file_helper.h"
#include "fts_rocksdb_merge.h"
#include "fts_utils.h"

namespace zvec {

FtsIndexer::~FtsIndexer() {
  close();
}

FtsIndexer::Ptr FtsIndexer::CreateAndOpen(const std::string &working_dir,
                                          const FieldSchemaPtrList &fts_fields,
                                          bool create, bool read_only) {
  auto indexer = std::make_shared<FtsIndexer>(working_dir);
  auto s = indexer->open(fts_fields, create, read_only);
  if (!s.ok()) {
    return nullptr;
  }
  return indexer;
}

Status FtsIndexer::open(const FieldSchemaPtrList &fts_fields, bool create,
                        bool read_only) {
  if (fts_fields.empty()) {
    return Status::OK();
  }

  // Collect CF names and per-CF merge operators.
  std::vector<std::string> cf_names;
  std::unordered_map<std::string, std::shared_ptr<rocksdb::MergeOperator>>
      per_cf_merge_ops;

  for (const auto &field : fts_fields) {
    const auto &name = field->name();
    cf_names.push_back(name);
    cf_names.push_back(name + kFtsPositionsSuffix);

    per_cf_merge_ops[name] = std::make_shared<fts::FtsPostingsMerge>();
    per_cf_merge_ops[name + kFtsMaxTfSuffix] =
        std::make_shared<fts::FtsMaxTfMerge>();

    if (create) {
      cf_names.push_back(name + kFtsTfSuffix);
      cf_names.push_back(name + kFtsMaxTfSuffix);
      cf_names.push_back(name + kFtsDocLenSuffix);
    }
  }
  cf_names.push_back(kFtsStatCfName);

  fts_ctx_ = std::make_shared<RocksdbContext>();
  Status s;

  if (create) {
    s = fts_ctx_->create(RocksdbContext::Args{working_dir_, cf_names, nullptr,
                                              per_cf_merge_ops, true});
  } else {
    // Auto-discover existing CFs via ListColumnFamilies (empty column_names).
    s = fts_ctx_->open(
        RocksdbContext::Args{working_dir_, {}, nullptr, per_cf_merge_ops, true},
        read_only);
  }
  if (!s.ok()) {
    LOG_ERROR("FtsIndexer: failed to %s RocksDB at [%s]: %s",
              create ? "create" : "open", working_dir_.c_str(),
              s.message().c_str());
    fts_ctx_.reset();
    return s;
  }

  auto *stat_cf = fts_ctx_->get_cf(kFtsStatCfName);

  for (const auto &field : fts_fields) {
    const auto &name = field->name();
    auto *postings_cf = fts_ctx_->get_cf(name);
    auto *positions_cf = fts_ctx_->get_cf(name + kFtsPositionsSuffix);
    auto *term_freq_cf = fts_ctx_->get_cf(name + kFtsTfSuffix);
    auto *max_tf_cf = fts_ctx_->get_cf(name + kFtsMaxTfSuffix);
    auto *doc_len_cf = fts_ctx_->get_cf(name + kFtsDocLenSuffix);

    auto indexer = std::make_shared<fts::FtsColumnIndexer>();
    auto ret = indexer->open(field, fts_ctx_.get(), postings_cf, positions_cf,
                             term_freq_cf, max_tf_cf, doc_len_cf, stat_cf);
    if (!ret.has_value()) {
      LOG_ERROR("FtsIndexer: FtsColumnIndexer::open failed for field[%s]: %s",
                name.c_str(), ret.error().message().c_str());
      return Status::InternalError("FtsIndexer: open field failed: ", name, " ",
                                   ret.error().message());
    }

    indexers_[name] = indexer;
  }

  return Status::OK();
}

fts::FtsColumnIndexerPtr FtsIndexer::get(const std::string &field_name) const {
  auto it = indexers_.find(field_name);
  return it != indexers_.end() ? it->second : nullptr;
}

Status FtsIndexer::flush() {
  for (const auto &[name, indexer] : indexers_) {
    auto ret = indexer->flush();
    if (!ret.has_value()) {
      return Status::InternalError("FtsIndexer: flush failed: ", name, " ",
                                   ret.error().message());
    }
  }
  if (fts_ctx_) {
    auto s = fts_ctx_->flush();
    if (!s.ok()) {
      return s;
    }
  }
  return Status::OK();
}

Status FtsIndexer::close() {
  indexers_.clear();
  if (fts_ctx_) {
    auto s = fts_ctx_->close();
    fts_ctx_.reset();
    return s;
  }
  return Status::OK();
}

Status FtsIndexer::create_snapshot(const std::string &snapshot_path) {
  auto s = flush();
  if (!s.ok()) {
    LOG_ERROR("FtsIndexer: flush failed during snapshot");
    return s;
  }
  s = fts_ctx_->create_checkpoint(snapshot_path);
  if (!s.ok()) {
    LOG_ERROR("FtsIndexer: create_checkpoint to [%s] failed: %s",
              snapshot_path.c_str(), s.message().c_str());
  }
  return s;
}

Status FtsIndexer::create_field_indexer(const FieldSchema &field) {
  if (field.index_type() != IndexType::FTS) {
    return Status::InvalidArgument(
        "FtsIndexer::create_field_indexer: not FTS field");
  }

  const auto &name = field.name();
  if (indexers_.find(name) != indexers_.end()) {
    return Status::InvalidArgument(
        "FtsIndexer::create_field_indexer: field already exists: ", name);
  }

  // Register merge operators before creating CFs.
  fts_ctx_->per_cf_merge_ops_[name] = std::make_shared<fts::FtsPostingsMerge>();
  fts_ctx_->per_cf_merge_ops_[name + kFtsMaxTfSuffix] =
      std::make_shared<fts::FtsMaxTfMerge>();

  // Create all CFs for this field.
  std::vector<std::string> cf_names = {
      name, name + kFtsPositionsSuffix, name + kFtsTfSuffix,
      name + kFtsMaxTfSuffix, name + kFtsDocLenSuffix};

  for (const auto &cf_name : cf_names) {
    auto s = fts_ctx_->create_cf(cf_name);
    if (!s.ok()) {
      LOG_ERROR("FtsIndexer::create_field_indexer: create_cf[%s] failed: %s",
                cf_name.c_str(), s.message().c_str());
      return s;
    }
  }

  // Open FtsColumnIndexer on the new CFs.
  auto *postings_cf = fts_ctx_->get_cf(name);
  auto *positions_cf = fts_ctx_->get_cf(name + kFtsPositionsSuffix);
  auto *term_freq_cf = fts_ctx_->get_cf(name + kFtsTfSuffix);
  auto *max_tf_cf = fts_ctx_->get_cf(name + kFtsMaxTfSuffix);
  auto *doc_len_cf = fts_ctx_->get_cf(name + kFtsDocLenSuffix);
  auto *stat_cf = fts_ctx_->get_cf(kFtsStatCfName);

  auto field_schema = std::make_shared<FieldSchema>(field);
  auto indexer = std::make_shared<fts::FtsColumnIndexer>();
  auto ret =
      indexer->open(field_schema, fts_ctx_.get(), postings_cf, positions_cf,
                    term_freq_cf, max_tf_cf, doc_len_cf, stat_cf);
  if (!ret.has_value()) {
    LOG_ERROR(
        "FtsIndexer::create_field_indexer: FtsColumnIndexer::open failed: %s",
        ret.error().message().c_str());
    return Status::InternalError("FtsIndexer::create_field_indexer: open: ",
                                 ret.error().message());
  }

  indexers_[name] = indexer;
  return Status::OK();
}

Status FtsIndexer::remove_field_indexer(const std::string &field_name) {
  auto it = indexers_.find(field_name);
  if (it != indexers_.end()) {
    indexers_.erase(it);
  }

  // Drop all CFs belonging to this field.
  fts_ctx_->drop_cf(field_name);
  fts_ctx_->drop_cf(field_name + kFtsPositionsSuffix);
  fts_ctx_->drop_cf(field_name + kFtsTfSuffix);
  fts_ctx_->drop_cf(field_name + kFtsMaxTfSuffix);
  fts_ctx_->drop_cf(field_name + kFtsDocLenSuffix);

  // Remove per-field stat keys.
  auto *stat_cf = fts_ctx_->get_cf(kFtsStatCfName);
  if (stat_cf) {
    auto rs = fts_ctx_->db_->Delete(fts_ctx_->write_opts_, stat_cf,
                                    fts::make_total_docs_key(field_name));
    if (!rs.ok()) {
      LOG_ERROR(
          "FtsIndexer::remove_field_indexer: delete total_docs key "
          "failed for field[%s]: %s",
          field_name.c_str(), rs.ToString().c_str());
      return Status::InternalError("delete total_docs key failed");
    }
    rs = fts_ctx_->db_->Delete(fts_ctx_->write_opts_, stat_cf,
                               fts::make_total_tokens_key(field_name));
    if (!rs.ok()) {
      LOG_ERROR(
          "FtsIndexer::remove_field_indexer: delete total_tokens key "
          "failed for field[%s]: %s",
          field_name.c_str(), rs.ToString().c_str());
      return Status::InternalError("delete total_tokens key failed");
    }
  }

  return Status::OK();
}

Status FtsIndexer::insert(const std::string &field_name, uint32_t seg_doc_id,
                          const std::string &text) {
  auto it = indexers_.find(field_name);
  if (it == indexers_.end()) {
    return Status::NotFound("FtsIndexer::insert: field not found: ",
                            field_name);
  }
  auto ret = it->second->insert(seg_doc_id, text);
  if (!ret.has_value()) {
    return Status::InternalError("FtsIndexer::insert failed: ", field_name, " ",
                                 ret.error().message());
  }
  return Status::OK();
}

Status FtsIndexer::seal(const std::string &field_name) {
  auto it = indexers_.find(field_name);
  if (it == indexers_.end()) {
    return Status::NotFound("FtsIndexer::seal: field not found: ", field_name);
  }

  auto &indexer = it->second;

  auto ret = indexer->flush();
  if (!ret.has_value()) {
    return Status::InternalError("FtsIndexer::seal flush failed: ", field_name,
                                 " ", ret.error().message());
  }

  ret = indexer->convert_postings_to_bitpacked();
  if (!ret.has_value()) {
    return Status::InternalError("FtsIndexer::seal convert failed: ",
                                 field_name, " ", ret.error().message());
  }

  indexer->reset_side_cfs();
  fts_ctx_->drop_cf(field_name + kFtsTfSuffix);
  fts_ctx_->drop_cf(field_name + kFtsMaxTfSuffix);
  fts_ctx_->drop_cf(field_name + kFtsDocLenSuffix);

  return Status::OK();
}

Status FtsIndexer::seal_all() {
  // Flush all indexers first.
  for (const auto &[name, indexer] : indexers_) {
    auto ret = indexer->flush();
    if (!ret.has_value()) {
      return Status::InternalError("FtsIndexer::seal_all flush failed: ", name,
                                   " ", ret.error().message());
    }
  }

  // Convert all postings to bitpacked format.
  for (const auto &[name, indexer] : indexers_) {
    auto ret = indexer->convert_postings_to_bitpacked();
    if (!ret.has_value()) {
      return Status::InternalError("FtsIndexer::seal_all convert failed: ",
                                   name, " ", ret.error().message());
    }
  }

  // Reset side CFs and drop them.
  for (const auto &[name, indexer] : indexers_) {
    indexer->reset_side_cfs();
  }
  for (const auto &[name, _] : indexers_) {
    fts_ctx_->drop_cf(name + kFtsTfSuffix);
    fts_ctx_->drop_cf(name + kFtsMaxTfSuffix);
    fts_ctx_->drop_cf(name + kFtsDocLenSuffix);
  }

  return Status::OK();
}

}  // namespace zvec
