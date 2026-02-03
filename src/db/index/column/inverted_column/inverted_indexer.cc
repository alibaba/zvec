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


#include "inverted_indexer.h"
#include <zvec/ailego/encoding/json.h>
#include "inverted_rocksdb_merger.h"


namespace zvec {


Status InvertedIndexer::open(bool create_dir_if_missing, bool read_only) {
  std::vector<std::string> cf_names{};  // Column families
  for (const auto &field : fields_) {
    if (field.index_type() != IndexType::INVERT) {
      LOG_ERROR("Field[%s] is not an inverted field", field.name().c_str());
      return Status::InvalidArgument();
    }
    auto params =
        std::dynamic_pointer_cast<InvertIndexParams>(field.index_params());
    cf_names.emplace_back(field.name() + INVERT_SUFFIX_TERMS);
    if (field.is_array_type()) {
      cf_names.emplace_back(field.name() + INVERT_SUFFIX_ARRAY_LEN);
    }
    if (allow_range_optimization(field) &&
        params->enable_range_optimization()) {
      cf_names.emplace_back(field.name() + INVERT_SUFFIX_RANGES);
    }
    if (allow_extended_wildcard(field) && params->enable_extended_wildcard()) {
      cf_names.emplace_back(field.name() + INVERT_SUFFIX_REVERSED_TERMS);
    }
  }
  cf_names.emplace_back(INVERT_CDF);

  Status s;
  if (FILE::IsExist(working_dir_)) {
    if (!FILE::IsDirectory(working_dir_)) {
      LOG_ERROR("InvertedIndexer path[%s] is not a directory",
                working_dir_.c_str());
      return Status::InvalidArgument();
    }
    s = rocksdb_context_.open(working_dir_, cf_names, read_only,
                              std::make_shared<InvertedRocksdbValueMerger>());
  } else {
    if (!create_dir_if_missing) {
      LOG_ERROR("InvertedIndexer path[%s] does not exist",
                working_dir_.c_str());
      return Status::NotFound();
    }
    s = rocksdb_context_.create(working_dir_, cf_names,
                                std::make_shared<InvertedRocksdbValueMerger>());
  }

  if (!s.ok()) {
    LOG_ERROR("Failed to open %s", ID().c_str());
    return s;
  }

  for (const auto &field : fields_) {
    auto column_indexer = InvertedColumnIndexer::CreateAndOpen(
        collection_name_, field, rocksdb_context_, read_only);
    if (column_indexer == nullptr) {
      LOG_ERROR("Failed to create InvertedColumnIndexer[%s]",
                field.name().c_str());
      return Status::InternalError();
    }
    indexers_.emplace(field.name(), std::move(column_indexer));
  }

  LOG_INFO("Opened %s", ID().c_str());
  return s;
}


Status InvertedIndexer::flush() {
  for (auto &[_, indexer] : indexers_) {
    if (indexer->is_sealed()) {
      continue;
    }
    if (!indexer->flush_special_values().ok()) {
      LOG_ERROR("Failed to flush %s", indexer->ID().c_str());
      return Status::InternalError();
    }
  }

  auto s = rocksdb_context_.flush();
  if (s.ok()) {
    LOG_INFO("Flushed %s", ID().c_str());
  } else {
    LOG_ERROR("Failed to flush %s", ID().c_str());
  }
  return s;
}


Status InvertedIndexer::create_snapshot(const std::string &snapshot_dir) {
  Status s;
  if (s = flush(); !s.ok()) {
    LOG_ERROR("Failed to flush %s during creating a snapshot", ID().c_str());
    return s;
  }

  if (s = rocksdb_context_.create_checkpoint(snapshot_dir); s.ok()) {
    LOG_INFO("Created snapshot[%s] of %s", snapshot_dir.c_str(), ID().c_str());
  } else {
    LOG_ERROR("Failed to create snapshot[%s] of %s", snapshot_dir.c_str(),
              ID().c_str());
  }
  return s;
}


Status InvertedIndexer::seal() {
  Status s;
  for (const auto &[_, indexer] : indexers_) {
    if (indexer->is_sealed()) {
      continue;
    }
    if (s = indexer->seal(); !s.ok()) {
      LOG_ERROR("Failed to seal %s", indexer->ID().c_str());
    }
  }

  if (s = flush(); !s.ok()) {
    LOG_ERROR("Failed to flush %s during sealing", ID().c_str());
    return s;
  }

  if (s = rocksdb_context_.compact(); s.ok()) {
    LOG_INFO("Sealed %s", ID().c_str());
  } else {
    LOG_ERROR("Failed to compact %s during sealing", ID().c_str());
  }
  return s;
}


Status InvertedIndexer::create_column_indexer(const FieldSchema &field) {
  if (field.index_type() != IndexType::INVERT) {
    return Status::InvalidArgument();
  }
  auto it = std::find_if(fields_.begin(), fields_.end(),
                         [&field](FieldSchema &cur_field) {
                           return cur_field.name() == field.name();
                         });
  if (it != fields_.end()) {
    LOG_ERROR("InvertedColumnIndexer[%s] already exists in %s",
              field.name().c_str(), ID().c_str());
    return Status::InvalidArgument();
  }
  auto params =
      std::dynamic_pointer_cast<InvertIndexParams>(field.index_params());

  Status s;
  bool cf_terms_created{false};
  bool cf_array_len_created{false};
  bool cf_ranges_created{false};
  bool cf_reversed_terms_created{false};
  AILEGO_DEFER([&]() {
    if (s.ok()) {
      LOG_INFO("Created a new InvertedColumnIndexer[%s] in %s",
               field.name().c_str(), ID().c_str());
    } else {
      if (cf_terms_created) {
        rocksdb_context_.drop_cf(field.name() + INVERT_SUFFIX_TERMS);
      }
      if (cf_array_len_created) {
        rocksdb_context_.drop_cf(field.name() + INVERT_SUFFIX_ARRAY_LEN);
      }
      if (cf_ranges_created) {
        rocksdb_context_.drop_cf(field.name() + INVERT_SUFFIX_RANGES);
      }
      if (cf_reversed_terms_created) {
        rocksdb_context_.drop_cf(field.name() + INVERT_SUFFIX_REVERSED_TERMS);
      }
      LOG_ERROR("Failed to create InvertedColumnIndexer[%s] in %s",
                field.name().c_str(), ID().c_str());
    }
  });

  s = rocksdb_context_.create_cf(field.name() + INVERT_SUFFIX_TERMS);
  if (s.ok()) {
    cf_terms_created = true;
  } else {
    return s;
  }
  if (field.is_array_type()) {
    s = rocksdb_context_.create_cf(field.name() + INVERT_SUFFIX_ARRAY_LEN);
    if (s.ok()) {
      cf_array_len_created = true;
    } else {
      return s;
    }
  }
  if (allow_range_optimization(field) && params->enable_range_optimization()) {
    s = rocksdb_context_.create_cf(field.name() + INVERT_SUFFIX_RANGES);
    if (s.ok()) {
      cf_ranges_created = true;
    } else {
      return s;
    }
  }
  if (allow_extended_wildcard(field) && params->enable_extended_wildcard()) {
    s = rocksdb_context_.create_cf(field.name() + INVERT_SUFFIX_REVERSED_TERMS);
    if (s.ok()) {
      cf_reversed_terms_created = true;
    } else {
      return s;
    }
  }

  auto column_indexer = InvertedColumnIndexer::CreateAndOpen(
      collection_name_, field, rocksdb_context_);
  if (column_indexer) {
    fields_.emplace_back(field);
    indexers_.emplace(field.name(), std::move(column_indexer));
    s = Status::OK();
  } else {
    s = Status::InternalError();
  }
  return s;
}


Status InvertedIndexer::remove_column_indexer(const std::string &field_name) {
  auto it = std::find_if(fields_.begin(), fields_.end(),
                         [&field_name](FieldSchema &cur_field) {
                           return cur_field.name() == field_name;
                         });
  auto column_indexer = (*this)[field_name];
  if (it == fields_.end() && !column_indexer) {
    LOG_ERROR("InvertedColumnIndexer[%s] doesn't exists in %s",
              field_name.c_str(), ID().c_str());
    return Status::NotFound();
  }
  if (it == fields_.end() || !column_indexer) {
    LOG_ERROR("%s is in corrupted state", ID().c_str());
    return Status::InternalError();
  }

  if (auto s = column_indexer->drop_storage(); !s.ok()) {
    LOG_ERROR("Failed to remove InvertedColumnIndexer[%s] in %s",
              field_name.c_str(), ID().c_str());
    return s;
  }

  fields_.erase(it);
  indexers_.erase(field_name);
  LOG_INFO("Removed InvertedColumnIndexer[%s] in %s", field_name.c_str(),
           ID().c_str());
  return Status::OK();
}


}  // namespace zvec