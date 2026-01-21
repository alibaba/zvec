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

#include "id_map.h"
#include <zvec/ailego/logger/logger.h>
#include "db/common/constants.h"


namespace zvec {


Status IDMap::open(const std::string &working_dir, bool create_if_missing,
                   bool read_only) {
  if (opened_) {
    LOG_ERROR("IDMap is already opened");
    return Status::InternalError();
  }

  Status s;
  if (FILE::IsExist(working_dir)) {
    if (!FILE::IsDirectory(working_dir)) {
      LOG_ERROR("IDMap path[%s] is not a directory", working_dir.c_str());
      return Status::InvalidArgument();
    }
    s = rocksdb_context_.open(working_dir, read_only);
  } else {
    if (!create_if_missing) {
      LOG_ERROR("IDMap path[%s] does not exist", working_dir.c_str());
      return Status::NotFound();
    }
    s = rocksdb_context_.create(working_dir);
  }
  if (s.ok()) {
    LOG_INFO("Opened IDMap[%s]", working_dir.c_str());
    working_dir_ = working_dir;
    opened_ = true;
  } else {
    LOG_ERROR("Failed to open IDMap[%s]", working_dir.c_str());
  }
  return s;
}


IDMap::Ptr IDMap::CreateAndOpen(const std::string &collection_name,
                                const std::string &working_dir,
                                bool create_if_missing, bool read_only) {
  IDMap::Ptr id_map = std::make_shared<IDMap>(collection_name);
  if (id_map->open(working_dir, create_if_missing, read_only).ok()) {
    return id_map;
  } else {
    return nullptr;
  }
}


Status IDMap::close() {
  if (!opened_) {
    return Status::OK();
  }

  Status status = rocksdb_context_.close();
  if (status.ok()) {
    LOG_INFO("Closed IDMap[%s]", working_dir_.c_str());
  } else {
    LOG_ERROR("Failed to close IDMap[%s]", working_dir_.c_str());
  }
  return status;
}


Status IDMap::flush() {
  if (!opened_) {
    return Status::InternalError();
  }

  auto s = rocksdb_context_.flush();
  if (s.ok()) {
    LOG_INFO("Flushed IDMap[%s]", working_dir_.c_str());
  } else {
    LOG_ERROR("Failed to flush IDMap[%s]", working_dir_.c_str());
  }
  return s;
}


Status IDMap::upsert(const std::string &key, uint64_t doc_id) {
  if (!opened_) {
    return Status::InternalError();
  }

  rocksdb::Slice value((const char *)&doc_id, sizeof(uint64_t));
  auto s = rocksdb_context_.db_->Put(rocksdb_context_.write_opts_, key, value);
  if (s.ok()) {
    return Status::OK();
  } else {
    LOG_ERROR("Failed to put [%s, %zu] into IDMap[%s], code[%d], reason[%s]",
              key.c_str(), (size_t)doc_id, working_dir_.c_str(), s.code(),
              s.ToString().c_str());
    return Status::InternalError();
  }
}


void IDMap::remove(const std::string &key) {
  rocksdb_context_.db_->Delete(rocksdb_context_.write_opts_, key);
}


bool IDMap::has(const std::string &key, uint64_t *doc_id) const {
  std::string value;
  auto s = rocksdb_context_.db_->Get(rocksdb_context_.read_opts_, key, &value);
  if (s.ok()) {
    if (doc_id) {
      *doc_id = *(uint64_t *)(value.data());
    }
    return true;
  } else {
    if (doc_id) {
      *doc_id = INVALID_DOC_ID;
    }
    return false;
  }
}


Status IDMap::multi_get(const std::vector<std::string> &keys,
                        std::vector<uint64_t> *doc_ids) const {
  if (keys.empty()) {
    doc_ids->clear();
    return Status::InvalidArgument();
  }

  std::vector<rocksdb::Slice> slice_keys(keys.begin(), keys.end());
  std::vector<rocksdb::PinnableSlice> pinnable_values;
  pinnable_values.resize(keys.size());
  std::vector<rocksdb::Status> statuses;
  statuses.resize(keys.size());

  auto db = rocksdb_context_.db_.get();

  db->MultiGet(rocksdb_context_.read_opts_, db->DefaultColumnFamily(),
               slice_keys.size(), slice_keys.data(), pinnable_values.data(),
               statuses.data(), false);

  doc_ids->resize(keys.size());
  for (size_t i = 0; i < keys.size(); i++) {
    if (statuses[i].ok()) {
      (*doc_ids)[i] = *(uint64_t *)(pinnable_values[i].data());
    } else if (statuses[i].code() == rocksdb::Status::kNotFound) {
      (*doc_ids)[i] = INVALID_DOC_ID;
    } else {
      LOG_ERROR("Failed to get key[%s] from IDMap[%s], code[%d], reason[%s]",
                keys[i].c_str(), working_dir_.c_str(), statuses[i].code(),
                statuses[i].ToString().c_str());
      return Status::InternalError();
    }
  }

  return Status::OK();
}


size_t IDMap::storage_size_in_bytes() {
  return rocksdb_context_.sst_file_size();
}


size_t IDMap::count() {
  return rocksdb_context_.count();
}


}  // namespace zvec
