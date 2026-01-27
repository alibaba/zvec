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


#include <rocksdb/filter_policy.h>
#include <rocksdb/statistics.h>
#include <rocksdb/table.h>
#include <rocksdb/utilities/checkpoint.h>
#include <zvec/ailego/logger/logger.h>
#include "rocksdb_context.h"


namespace zvec {


Status RocksdbContext::create(
    const std::string &db_path,
    std::shared_ptr<rocksdb::MergeOperator> merge_op) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (db_) {
    LOG_ERROR("RocksDB[%s] is already opened", db_path_.c_str());
    return Status::PermissionDenied();
  }

  if (auto s = validate_and_set_db_path(db_path, false); !s.ok()) {
    return s;
  }

  create_opts_.create_if_missing = true;
  prepare_options(merge_op);

  // Open RocksDB
  rocksdb::DB *db;
  if (auto s = rocksdb::DB::Open(create_opts_, db_path, &db); !s.ok()) {
    LOG_ERROR("Failed to create RocksDB[%s], code[%d], reason[%s]",
              db_path.c_str(), s.code(), s.ToString().c_str());
    return Status::InternalError();
  }

  db_.reset(db);
  read_only_ = false;
  write_opts_.disableWAL = true;
  LOG_DEBUG("Created RocksDB[%s]", db_path.c_str());
  return Status::OK();
}


Status RocksdbContext::create(
    const std::string &db_path, const std::vector<std::string> &column_names,
    std::shared_ptr<rocksdb::MergeOperator> merge_op) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (db_) {
    LOG_ERROR("RocksDB[%s] is already opened", db_path_.c_str());
    return Status::PermissionDenied();
  }

  if (auto s = validate_and_set_db_path(db_path, false); !s.ok()) {
    return s;
  }

  create_opts_.create_if_missing = true;
  prepare_options(merge_op);

  // Open RocksDB
  rocksdb::DB *db;
  rocksdb::Status s = rocksdb::DB::Open(create_opts_, db_path, &db);
  if (!s.ok()) {
    LOG_ERROR("Failed to create RocksDB[%s], code[%d], reason[%s]",
              db_path.c_str(), s.code(), s.ToString().c_str());
    return Status::InternalError();
  }
  db_.reset(db);

  // Create column families
  bool has_default = false;
  for (auto const &column_name : column_names) {
    if (column_name == rocksdb::kDefaultColumnFamilyName) {
      cf_handles_.push_back(db->DefaultColumnFamily());
      has_default = true;
      continue;
    }
    rocksdb::ColumnFamilyHandle *cf_handle{nullptr};
    rocksdb::ColumnFamilyOptions cf_options(create_opts_);
    s = db->CreateColumnFamily(cf_options, column_name, &cf_handle);
    if (!s.ok()) {
      LOG_ERROR("Failed to create cf[%s] in RocksDB[%s], code[%d], reason[%s]",
                column_name.c_str(), db_path.c_str(), s.code(),
                s.ToString().c_str());
      delete_cf_handles();
      db->Close();
      db_.reset();
      return Status::InternalError();
    }
    cf_handles_.push_back(cf_handle);
  }
  if (!has_default) {
    cf_handles_.push_back(db->DefaultColumnFamily());
  }

  read_only_ = false;
  write_opts_.disableWAL = true;
  LOG_DEBUG("Created RocksDB[%s]", db_path.c_str());
  return Status::OK();
}


Status RocksdbContext::open(const std::string &db_path, bool read_only,
                            std::shared_ptr<rocksdb::MergeOperator> merge_op) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (db_) {
    LOG_ERROR("RocksDB[%s] is already opened", db_path_.c_str());
    return Status::PermissionDenied();
  }

  if (auto s = validate_and_set_db_path(db_path, true); !s.ok()) {
    return s;
  }

  create_opts_.create_if_missing = false;
  prepare_options(merge_op);

  // Open RocksDB
  rocksdb::DB *db;
  rocksdb::Status s;
  if (read_only) {
    s = rocksdb::DB::OpenForReadOnly(create_opts_, db_path, &db);
  } else {
    s = rocksdb::DB::Open(create_opts_, db_path, &db);
  }
  if (!s.ok()) {
    LOG_ERROR("Failed to open RocksDB[%s], code[%d], reason[%s]",
              db_path.c_str(), s.code(), s.ToString().c_str());
    return Status::InternalError();
  }

  db_.reset(db);
  read_only_ = read_only;
  write_opts_.disableWAL = true;
  LOG_DEBUG("Opened RocksDB[%s]", db_path.c_str());
  return Status::OK();
}


Status RocksdbContext::open(const std::string &db_path,
                            const std::vector<std::string> &column_names,
                            bool read_only,
                            std::shared_ptr<rocksdb::MergeOperator> merge_op) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (db_) {
    LOG_ERROR("RocksDB[%s] is already opened", db_path_.c_str());
    return Status::PermissionDenied();
  }

  if (auto s = validate_and_set_db_path(db_path, true); !s.ok()) {
    return s;
  }

  create_opts_.create_if_missing = false;
  prepare_options(merge_op);

  // Set up column families
  rocksdb::Status s;
  std::vector<std::string> existing_cf_names{};
  std::vector<rocksdb::ColumnFamilyDescriptor> cf_descriptors{};
  s = rocksdb::DB::ListColumnFamilies(create_opts_, db_path,
                                      &existing_cf_names);
  if (!s.ok()) {
    LOG_ERROR("Failed to list cf in RocksDB[%s], code[%d], reason[%s]",
              db_path.c_str(), s.code(), s.ToString().c_str());
    return Status::InternalError();
  }
  rocksdb::ColumnFamilyOptions cf_options(create_opts_);
  if (column_names.empty()) {  // Get all column families from DB
    for (auto const &column_name : existing_cf_names) {
      cf_descriptors.emplace_back(column_name, cf_options);
    }
  } else {
    bool has_default = false;
    for (const auto &column_name : column_names) {
      if (std::find(existing_cf_names.begin(), existing_cf_names.end(),
                    column_name) == existing_cf_names.end()) {
        LOG_ERROR("Column family[%s] does not exist in RocksDB[%s]",
                  column_name.c_str(), db_path.c_str());
        return Status::InvalidArgument();
      }
      if (column_name == rocksdb::kDefaultColumnFamilyName) {
        has_default = true;
      }
    }
    if (read_only) {
      for (const auto &column_name : column_names) {
        cf_descriptors.emplace_back(column_name, cf_options);
      }
      if (!has_default) {
        cf_descriptors.emplace_back(rocksdb::kDefaultColumnFamilyName,
                                    cf_options);
      }
    } else {  // Rocksdb must be opened with all column families in write mode
      for (auto const &column_name : existing_cf_names) {
        cf_descriptors.emplace_back(column_name, cf_options);
      }
    }
  }

  // Open RocksDB
  rocksdb::DB *db;
  if (read_only) {
    s = rocksdb::DB::OpenForReadOnly(create_opts_, db_path, cf_descriptors,
                                     &cf_handles_, &db);
  } else {
    s = rocksdb::DB::Open(create_opts_, db_path, cf_descriptors, &cf_handles_,
                          &db);
  }
  if (!s.ok()) {
    LOG_ERROR("Failed to open RocksDB[%s], code[%d], reason[%s]",
              db_path.c_str(), s.code(), s.ToString().c_str());
    return Status::InternalError();
  }

  db_.reset(db);
  read_only_ = read_only;
  write_opts_.disableWAL = true;
  LOG_DEBUG("Opened RocksDB[%s]", db_path.c_str());
  return Status::OK();
}


Status RocksdbContext::validate_and_set_db_path(const std::string &db_path,
                                                bool should_exist) {
  if (db_path.empty()) {
    LOG_ERROR("RocksDB path cannot be empty");
    return Status::InvalidArgument();
  }

  if (FILE::IsExist(db_path)) {
    if (!should_exist) {
      LOG_ERROR("RocksDB path[%s] already exists", db_path.c_str());
      return Status::InvalidArgument();
    }
    if (!FILE::IsDirectory(db_path)) {
      LOG_ERROR("RocksDB path[%s] is not a directory", db_path.c_str());
      return Status::InvalidArgument();
    }
  } else {
    if (should_exist) {
      LOG_ERROR("RocksDB path[%s] does not exist", db_path.c_str());
      return Status::NotFound();
    }
  }

  db_path_ = db_path;
  return Status::OK();
}


void RocksdbContext::prepare_options(
    std::shared_ptr<rocksdb::MergeOperator> merge_op) {
  // Increase parallelism with default thread count (typically 16)
  create_opts_.IncreaseParallelism();

  // Optimize for level-based compaction style with default setting
  create_opts_.OptimizeLevelStyleCompaction();

  // TODO: enable compression?

  // Setting this to 1 means that when a memtable is full, it will be flushed
  // to disk immediately rather than being merged with other memtables
  create_opts_.min_write_buffer_number_to_merge = 1;

  // Set the block size for the arena memory allocator to 64KB, which controls
  // how much memory is allocated at a time for internal operations
  create_opts_.arena_block_size = 1024 * 64;

  // Do not create LOG.old when reopen
  create_opts_.keep_log_file_num = 1;

  // Warnings and errors only
  create_opts_.info_log_level = rocksdb::WARN_LEVEL;

  rocksdb::BlockBasedTableOptions table_options;

  // Turn on bloom filters
  table_options.filter_policy.reset(rocksdb::NewBloomFilterPolicy(10, false));

  // Merge operator
  if (merge_op) {
    create_opts_.merge_operator = merge_op;
    create_opts_.max_successive_merges = 100;
    create_opts_.write_buffer_size = 8 << 20;
  }

  // Create default cache
  table_options.block_cache = nullptr;

  auto table_factory = NewBlockBasedTableFactory(table_options);
  create_opts_.table_factory.reset(table_factory);

  // Enable statistics
  create_opts_.statistics = rocksdb::CreateDBStatistics();

  // Disable external write buffer manager, let RocksDB manage it
  create_opts_.write_buffer_manager = nullptr;

  // Reduce preallocation size for manifest file to 512KB to save disk space
  create_opts_.manifest_preallocation_size = 512 * 1024;

  // Disable direct reads (use buffered I/O instead)
  create_opts_.use_direct_reads = false;
}


Status RocksdbContext::close() {
  std::lock_guard<std::mutex> lock(mutex_);

  if (db_ == nullptr) {
    LOG_ERROR("RocksDB[%s] is not opened", db_path_.c_str());
    return Status::InternalError();
  }

  if (!read_only_) {
    if (auto s = flush_unlocked(); !s.ok()) {
      LOG_ERROR("Failed to close RocksDB[%s] due to flush failure",
                db_path_.c_str());
      return s;
    }
  }

  delete_cf_handles();

  if (auto s = db_->Close(); s.ok()) {
    LOG_DEBUG("Closed RocksDB[%s]", db_path_.c_str());
    db_.reset();
    return Status::OK();
  } else {
    LOG_ERROR("Failed to close RocksDB[%s], code[%d], reason[%s]",
              db_path_.c_str(), s.code(), s.ToString().c_str());
    return Status::InternalError();
  }
}


Status RocksdbContext::flush_unlocked() {
  if (read_only_) {
    LOG_ERROR("Cannot flush RocksDB[%s] in read-only mode", db_path_.c_str());
    return Status::PermissionDenied();
  }

  for (const auto &cf : cf_handles_) {
    if (auto s = db_->Flush(flush_opts_, cf); !s.ok()) {
      LOG_ERROR("Failed to flush cf[%s] of RocksDB[%s], code[%d], reason[%s]",
                cf->GetName().c_str(), db_path_.c_str(), s.code(),
                s.ToString().c_str());
      return Status::InternalError();
    }
  }

  if (auto s = db_->Flush(flush_opts_); s.ok()) {
    LOG_DEBUG("Flushed RocksDB[%s]", db_path_.c_str());
    return Status::OK();
  } else {
    LOG_ERROR("Failed to flush Rocksdb[%s], code[%d], reason[%s]",
              db_path_.c_str(), s.code(), s.ToString().c_str());
    return Status::InternalError();
  }
}


Status RocksdbContext::flush() {
  std::lock_guard<std::mutex> lock(mutex_);
  return flush_unlocked();
}


Status RocksdbContext::create_checkpoint(const std::string &checkpoint_dir) {
  std::lock_guard<std::mutex> lock(mutex_);

  rocksdb::Checkpoint *cp{nullptr};
  if (auto s = rocksdb::Checkpoint::Create(db_.get(), &cp); !s.ok()) {
    LOG_ERROR(
        "Failed to create a checkpoint object of Rocksdb[%s], code[%d], "
        "reason[%s]",
        db_path_.c_str(), s.code(), s.ToString().c_str());
    return Status::InternalError();
  }

  if (auto s = cp->CreateCheckpoint(checkpoint_dir); s.ok()) {
    LOG_DEBUG("Created a checkpoint of Rocksdb[%s] to [%s]", db_path_.c_str(),
              checkpoint_dir.c_str());
    delete cp;
    return Status::OK();
  } else {
    LOG_ERROR(
        "Failed to create a checkpoint of Rocksdb[%s], code[%d], reason[%s]",
        db_path_.c_str(), s.code(), s.ToString().c_str());
    delete cp;
    return Status::InternalError();
  }
}


rocksdb::ColumnFamilyHandle *RocksdbContext::get_cf(
    const std::string &cf_name) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto cf_handle : cf_handles_) {
    if (cf_handle->GetName() == cf_name) {
      return cf_handle;
    }
  }
  return nullptr;
}


Status RocksdbContext::create_cf(const std::string &cf_name) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (cf_name == rocksdb::kDefaultColumnFamilyName) {
    LOG_ERROR("Forbidden to create default cf in RocksDB[%s]",
              db_path_.c_str());
    return Status::InvalidArgument();
  }

  for (auto cf_handle : cf_handles_) {
    if (cf_handle->GetName() == cf_name) {
      LOG_ERROR("Column family[%s] already exists in RocksDB[%s]",
                cf_name.c_str(), db_path_.c_str());
      return Status::InvalidArgument();
    }
  }

  rocksdb::ColumnFamilyHandle *cf_handle{nullptr};
  auto s = db_->CreateColumnFamily(rocksdb::ColumnFamilyOptions(create_opts_),
                                   cf_name, &cf_handle);
  if (s.ok()) {
    cf_handles_.push_back(cf_handle);
    LOG_DEBUG("Created cf[%s] in RocksDB[%s]", cf_name.c_str(),
              db_path_.c_str());
    return Status::OK();
  } else {
    LOG_ERROR("Failed to create cf[%s] in RocksDB[%s], code[%d], reason[%s]",
              cf_name.c_str(), db_path_.c_str(), s.code(),
              s.ToString().c_str());
    return Status::InternalError();
  }
}


Status RocksdbContext::drop_cf(const std::string &cf_name) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (cf_name == rocksdb::kDefaultColumnFamilyName) {
    LOG_ERROR("Forbidden to drop default cf in RocksDB[%s]", db_path_.c_str());
    return Status::InvalidArgument();
  }

  auto it = std::find_if(cf_handles_.begin(), cf_handles_.end(),
                         [&cf_name](rocksdb::ColumnFamilyHandle *handle) {
                           return handle->GetName() == cf_name;
                         });
  if (it == cf_handles_.end()) {
    LOG_WARN("Failed to find column family[%s] in RocksDB[%s]", cf_name.c_str(),
             db_path_.c_str());
    return Status::OK();
  }

  auto s = db_->DropColumnFamily(*it);
  if (s.ok()) {
    delete *it;
    cf_handles_.erase(it);
    LOG_DEBUG("Dropped cf[%s] in RocksDB[%s]", cf_name.c_str(),
              db_path_.c_str());
    return Status::OK();
  } else {
    LOG_ERROR("Failed to drop cf[%s] in RocksDB[%s], code[%d], reason[%s]",
              cf_name.c_str(), db_path_.c_str(), s.code(),
              s.ToString().c_str());
    return Status::InternalError();
  }
}


Status RocksdbContext::reset_cf(const std::string &cf_name) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (cf_name == rocksdb::kDefaultColumnFamilyName) {
    LOG_ERROR("Forbidden to reset default cf in RocksDB[%s]", db_path_.c_str());
    return Status::InvalidArgument();
  }

  rocksdb::ColumnFamilyHandle *cf_handle{nullptr};
  size_t index;
  for (size_t i = 0; i < cf_handles_.size(); ++i) {
    if (cf_handles_[i]->GetName() == cf_name) {
      cf_handle = cf_handles_[i];
      index = i;
      break;
    }
  }
  if (cf_handle == nullptr) {
    LOG_ERROR("Column family[%s] does not exist in RocksDB[%s]",
              cf_name.c_str(), db_path_.c_str());
    return Status::InvalidArgument();
  }

  auto options = db_->GetOptions(cf_handle);
  auto s = db_->DropColumnFamily(cf_handle);
  if (!s.ok()) {
    LOG_ERROR("Failed to drop cf[%s] in RocksDB[%s], code[%d], reason[%s]",
              cf_name.c_str(), db_path_.c_str(), s.code(),
              s.ToString().c_str());
    return Status::InternalError();
  }
  delete cf_handle;

  rocksdb::ColumnFamilyHandle *new_cf_handle{nullptr};
  s = db_->CreateColumnFamily(options, cf_name, &new_cf_handle);
  if (s.ok()) {
    cf_handles_[index] = new_cf_handle;
    LOG_DEBUG("Reset cf[%s] in RocksDB[%s]", cf_name.c_str(), db_path_.c_str());
    return Status::OK();
  } else {
    LOG_ERROR("Failed to create cf[%s] in RocksDB[%s], code[%d], reason[%s]",
              cf_name.c_str(), db_path_.c_str(), s.code(),
              s.ToString().c_str());
    return Status::InternalError();
  }
}


void RocksdbContext::delete_cf_handles() {
  for (auto cf : cf_handles_) {
    db_->DestroyColumnFamilyHandle(cf);
  }
  cf_handles_.clear();
}


Status RocksdbContext::compact() {
  std::lock_guard<std::mutex> lock(mutex_);

  for (auto cf : cf_handles_) {
    auto s = db_->CompactRange(compact_range_opts_, cf, nullptr, nullptr);
    if (!s.ok()) {
      LOG_ERROR("Failed to compact cf[%s] in RocksDB[%s], code[%d], reason[%s]",
                cf->GetName().c_str(), db_path_.c_str(), s.code(),
                s.ToString().c_str());
    }
  }
  auto s = db_->CompactRange(compact_range_opts_, nullptr, nullptr);
  if (s.ok()) {
    LOG_DEBUG("Compacted RocksDB[%s]", db_path_.c_str());
    return Status::OK();
  } else {
    LOG_ERROR("Failed to compact RocksDB[%s], code[%d], reason[%s]",
              db_path_.c_str(), s.code(), s.ToString().c_str());
    return Status::InternalError();
  }
}


size_t RocksdbContext::sst_file_size() {
  uint64_t int_num = 0;
  if (db_->GetIntProperty("rocksdb.live-sst-files-size", &int_num)) {
    return int_num;
  } else {
    return 0;
  }
}


size_t RocksdbContext::count() {
  uint64_t int_num = 0;
  if (db_->GetIntProperty("rocksdb.estimate-num-keys", &int_num)) {
    return int_num;
  } else {
    return 0;
  }
}


}  // namespace zvec