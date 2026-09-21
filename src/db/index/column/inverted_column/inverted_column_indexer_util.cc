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


#include <ailego/pattern/defer.h>
#include "inverted_column_indexer.h"


namespace zvec {


InvertedColumnIndexer::~InvertedColumnIndexer() {
  LOG_INFO("Closed %s", id().c_str());
}


Status InvertedColumnIndexer::open() {
  if (field_.index_type() != IndexType::INVERT) {
    return Status::InvalidArgument();
  }
  auto params =
      std::dynamic_pointer_cast<InvertIndexParams>(field_.index_params());
  enable_range_optimization_ =
      allow_range_optimization(field_) && params->enable_range_optimization();
  enable_extended_wildcard_ =
      allow_extended_wildcard(field_) && params->enable_extended_wildcard();

  rocksdb::Status s;
  std::string value{};

  cf_terms_ = ctx_.get_cf(cf_name_terms());
  if (!cf_terms_) {
    LOG_ERROR("Failed to get cf_terms for %s", id().c_str());
    return Status::InternalError();
  }

  if (field_.is_array_type()) {
    cf_array_len_ = ctx_.get_cf(cf_name_array_len());
    if (!cf_array_len_) {
      LOG_ERROR("Failed to get cf_array_len for %s", id().c_str());
      return Status::InternalError();
    }
  }

  if (enable_range_optimization_) {
    cf_ranges_ = ctx_.get_cf(cf_name_ranges());
    if (!cf_ranges_) {
      LOG_ERROR("Failed to get cf_ranges for %s", id().c_str());
      return Status::InternalError();
    }
    cf_cdf_ = ctx_.get_cf(cf_name_cdf());
    if (!cf_cdf_) {
      LOG_ERROR("Failed to get cf_cdf for %s", id().c_str());
      return Status::InternalError();
    }
    s = ctx_.db_->Get(ctx_.read_opts_, cf_cdf_, field_.name(), &value);
    if (s.ok()) {
      doc_range_stat_ = SegmentDocRangeStat::Create(value);
      if (!doc_range_stat_) {
        LOG_ERROR("Failed to create doc range stats from %s", id().c_str());
        return Status::InternalError();
      }
    } else if (s.code() != rocksdb::Status::kNotFound) {
      LOG_ERROR("Failed to retrieve cdf from %s", id().c_str());
      return Status::InternalError();
    }
  }

  if (enable_extended_wildcard_) {
    cf_reversed_terms_ = ctx_.get_cf(cf_name_reversed_terms());
    if (!cf_reversed_terms_) {
      LOG_ERROR("Failed to get cf_reversed_terms for %s", id().c_str());
      return Status::InternalError();
    }
  }

  // Get max id if exists
  s = ctx_.db_->Get(ctx_.read_opts_, key_max_id(), &value);
  if (s.ok()) {
    try {
      max_id_ = std::stoul(value);
    } catch (const std::exception &e) {
      LOG_ERROR("Failed to parse max id from %s for %s, exception[%s]",
                value.c_str(), id().c_str(), e.what());
      return Status::InternalError();
    }
  } else if (s.code() != rocksdb::Status::kNotFound) {
    LOG_ERROR("Failed to retrieve max id from %s", id().c_str());
    return Status::InternalError();
  }

  // Get null bitmap if exists
  s = ctx_.db_->Get(ctx_.read_opts_, key_null(), &value);
  if (s.ok()) {
    if (auto status = null_bitmap_.deserialize(value); !status.ok()) {
      LOG_ERROR("Failed to deserialize null bitmap from %s", id().c_str());
      return status;
    }
  } else if (s.code() != rocksdb::Status::kNotFound) {
    LOG_ERROR("Failed to retrieve null bitmap from %s", id().c_str());
    return Status::InternalError();
  }

  // Get indexer state
  s = ctx_.db_->Get(ctx_.read_opts_, key_sealed(), &value);
  if (s.ok()) {
    sealed_ = true;
    read_only_ = true;
  } else if (s.code() == rocksdb::Status::kNotFound) {
    sealed_ = false;
  } else {
    LOG_ERROR("Failed to retrieve indexer state from %s", id().c_str());
    return Status::InternalError();
  }

  LOG_INFO("Opened %s", id().c_str());
  return Status::OK();
}


InvertedColumnIndexer::Ptr InvertedColumnIndexer::CreateAndOpen(
    const std::string &collection_name, const FieldSchema &field,
    RocksdbContext &context, bool read_only) {
  auto ptr =
      new InvertedColumnIndexer(collection_name, field, context, read_only);
  auto indexer = std::shared_ptr<InvertedColumnIndexer>(ptr);
  if (indexer->open().ok()) {
    return indexer;
  } else {
    return nullptr;
  }
}


Status InvertedColumnIndexer::drop_storage() {
  Status s = Status::OK();
  rocksdb::Status rs;
  AILEGO_DEFER([&]() {
    if (s.ok()) {
      LOG_INFO("Dropped storage of %s", id().c_str());
    } else {
      LOG_ERROR("Failed to drop storage of %s", id().c_str());
    }
  });

  if (s = ctx_.drop_cf(cf_name_terms()); !s.ok()) {
    return s;
  }
  if (field_.is_array_type()) {
    if (s = ctx_.drop_cf(cf_name_array_len()); !s.ok()) {
      return s;
    }
  }
  if (enable_range_optimization_) {
    if (s = ctx_.drop_cf(cf_name_ranges()); !s.ok()) {
      return s;
    }
    rs = ctx_.db_->Delete(ctx_.write_opts_, cf_cdf_, field_.name());
    if (!rs.ok()) {
      LOG_ERROR("Failed to delete cdf of %s", id().c_str());
      s = Status::InternalError();
      return s;
    }
  }
  if (enable_extended_wildcard_) {
    if (s = ctx_.drop_cf(cf_name_reversed_terms()); !s.ok()) {
      return s;
    }
  }

  rs = ctx_.db_->Delete(ctx_.write_opts_, key_max_id());
  if (!rs.ok()) {
    LOG_ERROR("Failed to delete max_id of %s", id().c_str());
    s = Status::InternalError();
    return s;
  }

  rs = ctx_.db_->Delete(ctx_.write_opts_, key_null());
  if (!rs.ok()) {
    LOG_ERROR("Failed to delete null bitmap of %s", id().c_str());
    s = Status::InternalError();
    return s;
  }

  rs = ctx_.db_->Delete(ctx_.write_opts_, key_sealed());
  if (!rs.ok()) {
    LOG_ERROR("Failed to delete indexer state of %s", id().c_str());
    s = Status::InternalError();
    return s;
  }

  return s;
}


}  // namespace zvec