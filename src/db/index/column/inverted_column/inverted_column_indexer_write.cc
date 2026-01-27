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
#include <zvec/ailego/encoding/json.h>
#include "inverted_codec.h"
#include "inverted_column_indexer.h"


namespace zvec {


Status InvertedColumnIndexer::insert(uint32_t id, const std::string &value) {
  if (read_only_) {
    return Status::PermissionDenied();
  }

  std::string encoded_id = std::string{1}.append(
      reinterpret_cast<const char *>(&id), sizeof(uint32_t));

  rocksdb::Status s;
  AILEGO_DEFER([&]() {
    if (!s.ok()) {
      LOG_ERROR("Failed to insert terms of id[%u] to %s, code[%d], reason[%s]",
                id, ID().c_str(), s.code(), s.ToString().c_str());
    }
  });

  if (field_.is_array_type()) {
    std::vector<std::string> encoded_values = encode_array(value);
    std::sort(encoded_values.begin(), encoded_values.end());
    rocksdb::WriteBatch write_batch;
    for (auto encoded_value : encoded_values) {
      s = write_batch.Merge(cf_terms_, encoded_value, encoded_id);
      if (!s.ok()) {
        return Status::InternalError();
      }
    }
    if (s = ctx_.db_->Write(ctx_.write_opts_, &write_batch); !s.ok()) {
      return Status::InternalError();
    }
    if (s = index_array_len(id, encoded_values.size()); !s.ok()) {
      return Status::InternalError();
    }
  } else {
    std::string encoded_value = encode(value);
    s = ctx_.db_->Merge(ctx_.write_opts_, cf_terms_, encoded_value, encoded_id);
    if (!s.ok()) {
      return Status::InternalError();
    }
    if (cf_reversed_terms_) {
      s = ctx_.db_->Merge(ctx_.write_opts_, cf_reversed_terms_,
                          encode_reversed(value), encoded_id);
      if (!s.ok()) {
        return Status::InternalError();
      }
    }
  }

  update_max_id(id);
  return Status::OK();
}


Status InvertedColumnIndexer::insert(uint32_t id,
                                     const std::vector<std::string> &values) {
  if (read_only_) {
    return Status::PermissionDenied();
  }

  std::string encoded_id = std::string{1}.append(
      reinterpret_cast<const char *>(&id), sizeof(uint32_t));
  auto encoded_values = encode(values);

  rocksdb::Status s;
  AILEGO_DEFER([&]() {
    if (!s.ok()) {
      LOG_ERROR("Failed to insert terms of id[%u] to %s, code[%d], reason[%s]",
                id, ID().c_str(), s.code(), s.ToString().c_str());
    }
  });

  if (s = index_array_len(id, encoded_values.size()); !s.ok()) {
    return Status::InternalError();
  }

  std::sort(encoded_values.begin(), encoded_values.end());
  rocksdb::WriteBatch write_batch;
  for (auto encoded_value : encoded_values) {
    s = write_batch.Merge(cf_terms_, encoded_value, encoded_id);
    if (!s.ok()) {
      return Status::InternalError();
    }
  }
  s = ctx_.db_->Write(ctx_.write_opts_, &write_batch);
  if (s.ok()) {
    update_max_id(id);
    return Status::OK();
  } else {
    return Status::InternalError();
  }
}


Status InvertedColumnIndexer::insert(uint32_t id, bool value) {
  if (read_only_) {
    return Status::PermissionDenied();
  }

  std::string encoded_id = std::string{1}.append(
      reinterpret_cast<const char *>(&id), sizeof(uint32_t));
  std::string encoded_value = encode(value);

  auto s =
      ctx_.db_->Merge(ctx_.write_opts_, cf_terms_, encoded_value, encoded_id);
  if (s.ok()) {
    update_max_id(id);
    return Status::OK();
  } else {
    LOG_ERROR("Failed to insert terms of id[%u] to %s, code[%d], reason[%s]",
              id, ID().c_str(), s.code(), s.ToString().c_str());
    return Status::InternalError();
  }
}


Status InvertedColumnIndexer::insert(uint32_t id,
                                     const std::vector<bool> &values) {
  if (read_only_) {
    return Status::PermissionDenied();
  }

  std::string encoded_id = std::string{1}.append(
      reinterpret_cast<const char *>(&id), sizeof(uint32_t));

  rocksdb::Status rs;
  if (rs = index_array_len(id, values.size()); !rs.ok()) {
    LOG_ERROR("Failed to index array length for %s", ID().c_str());
    return Status::InternalError();
  }

  bool has_true = false;
  bool has_false = false;
  for (bool value : values) {
    if (value) {
      has_true = true;
    } else {
      has_false = true;
    }
  }

  rocksdb::WriteBatch write_batch;
  if (has_true) {
    write_batch.Merge(cf_terms_, encode(true), encoded_id);
  }
  if (has_false) {
    write_batch.Merge(cf_terms_, encode(false), encoded_id);
  }
  rs = ctx_.db_->Write(ctx_.write_opts_, &write_batch);
  if (rs.ok()) {
    update_max_id(id);
    return Status::OK();
  } else {
    LOG_ERROR("Failed to insert terms of id[%u] to %s, code[%d], reason[%s]",
              id, ID().c_str(), rs.code(), rs.ToString().c_str());
    return Status::InternalError();
  }
}


Status InvertedColumnIndexer::insert_null(uint32_t id) {
  if (read_only_) {
    return Status::PermissionDenied();
  }
  null_bitmap_.add(id);
  update_max_id(id);
  return Status::OK();
}


Status InvertedColumnIndexer::flush_special_values() {
  if (read_only_) {
    return Status::PermissionDenied();
  }

  std::string value;
  if (null_bitmap_.cardinality() != 0) {
    if (!null_bitmap_.serialize(&value).ok()) {
      LOG_ERROR("Failed to serialize null bitmap");
      return Status::InternalError();
    }
    auto s = ctx_.db_->Put(ctx_.write_opts_, key_null(), value);
    if (!s.ok()) {
      LOG_ERROR("Failed to insert null bitmap to %s, code[%d], reason[%s]",
                ID().c_str(), s.code(), s.ToString().c_str());
      return Status::InternalError();
    }
  }

  auto s =
      ctx_.db_->Put(ctx_.write_opts_, key_max_id(), std::to_string(max_id_));
  if (s.ok()) {
    LOG_DEBUG("Special values flushed to %s", ID().c_str());
    return Status::OK();
  } else {
    LOG_ERROR("Failed to insert max_id to %s, code[%d], reason[%s]",
              ID().c_str(), s.code(), s.ToString().c_str());
    return Status::InternalError();
  }
}


rocksdb::Status InvertedColumnIndexer::index_array_len(uint32_t id,
                                                       uint32_t len) {
  if (!cf_array_len_) {
    LOG_ERROR("%s doesn't support array length index", ID().c_str());
    return rocksdb::Status::NotSupported();
  }

  std::string encoded_id = std::string{1}.append(
      reinterpret_cast<const char *>(&id), sizeof(uint32_t));
  std::string encoded_len = InvertedIndexCodec::Encode(
      std::string((char *)&len, sizeof(uint32_t)), DataType::UINT32);

  return ctx_.db_->Merge(ctx_.write_opts_, cf_array_len_, encoded_len,
                         encoded_id);
}


Status InvertedColumnIndexer::generate_statistical_indexes() {
  if (read_only_) {
    return Status::PermissionDenied();
  }
  if (!enable_range_optimization_) {
    return Status::PermissionDenied();
  }

  if (!ctx_.reset_cf(cf_name_ranges()).ok()) {
    // Reset the range index in case it is corrupted
    LOG_ERROR("Failed to reset range index");
    return Status::InternalError();
  }
  cf_ranges_ = ctx_.get_cf(cf_name_ranges());
  if (!cf_ranges_) {
    LOG_ERROR("Failed to get column families for %s", ID().c_str());
    return Status::InternalError();
  }

  // TODO: make them configurable
  const uint32_t num_range_slot = 1000;
  const uint32_t num_cdf_slot = 100;

  const uint32_t num_doc_per_range_slot = (max_id_ + 1) / num_range_slot;
  const uint32_t num_doc_per_cdf_slot = (max_id_ + 1) / num_cdf_slot;

  // Iterator for terms in the inverted index
  auto iter_term = ctx_.db_->NewIterator(ctx_.read_opts_, cf_terms_);
  iter_term->SeekToFirst();
  AILEGO_DEFER([&]() { delete iter_term; });

  size_t doc_count = 0, term_count = 0;
  Status s;
  rocksdb::Status rs;

  // Range tracking variables
  std::string range_begin_key{""}, range_end_key{""};
  size_t range_slot_doc_count{0};
  size_t num_range_slot_created = 0;
  roaring_bitmap_t *bitmap_range = roaring_bitmap_create();
  if (bitmap_range == nullptr) {
    LOG_ERROR("Failed to create bitmap");
    return Status::InternalError();
  }
  AILEGO_DEFER([&]() { roaring_bitmap_free(bitmap_range); });

  // Function to create a range slot
  auto create_range_slot = [&]() -> Status {
    std::string range_key = range_begin_key;
    range_key.append(1, '\0');  // Separator byte
    range_key.append(range_end_key.data(), range_end_key.size());
    uint64_t range_key_begin_size = range_begin_key.size();
    range_key.append((char *)&range_key_begin_size, sizeof(uint64_t));
    std::string range_value_str;
    s = InvertedIndexCodec::Serialize(bitmap_range, &range_value_str);
    if (!s.ok()) {
      LOG_ERROR("Failed to serialize bitmap");
      return Status::InternalError();
    }
    rs =
        ctx_.db_->Put(ctx_.write_opts_, cf_ranges_, range_key, range_value_str);
    if (!rs.ok()) {
      LOG_ERROR("Failed to put range slot: %s", rs.ToString().c_str());
      return Status::InternalError();
    }
    num_range_slot_created++;
    return Status::OK();
  };

  // CDF tracking variables
  ailego::JsonArray cdf_json_array;
  size_t cdf_slot_doc_count = 0;

  // Function to create a CDF slot
  auto create_cdf_slot = [&]() {
    ailego::JsonObject json_obj;
    json_obj.set(ailego::JsonString("key").encode(),
                 ailego::JsonString(iter_term->key().ToString()).encode());
    json_obj.set(ailego::JsonString("doc_count").encode(),
                 ailego::JsonValue(doc_count));
    cdf_json_array.push(json_obj);
  };

  // Is the current slot initialized?
  bool range_slot_initialized{false}, cdf_slot_initialized{false};


  // Scan
  roaring_bitmap_t *bitmap_cur{nullptr};
  AILEGO_DEFER([&]() {
    if (bitmap_cur) {
      roaring_bitmap_free(bitmap_cur);
    }
  });

  while (iter_term->Valid()) {
    term_count++;
    s = InvertedIndexCodec::Deserialize(iter_term->value().data(),
                                        iter_term->value().size(), &bitmap_cur);
    if (!s.ok()) {
      LOG_ERROR("Failed to deserialize bitmap for term[%s] from %s",
                iter_term->key().ToString().c_str(), ID().c_str());
      return Status::InternalError();
    }
    // The count of documents for the current term
    auto term_doc_count = roaring_bitmap_get_cardinality(bitmap_cur);
    doc_count += term_doc_count;

    // Range
    if (!range_slot_initialized) {
      range_slot_initialized = true;
      range_slot_doc_count = 0;
      range_begin_key = iter_term->key().ToString();
      roaring_bitmap_clear(bitmap_range);
    }
    range_end_key = iter_term->key().ToString();
    range_slot_doc_count += term_doc_count;
    roaring_bitmap_or_inplace(bitmap_range, bitmap_cur);
    if (range_slot_doc_count >= num_doc_per_range_slot) {
      if (create_range_slot().ok()) {
        range_slot_initialized = false;
      } else {
        return Status::InternalError();
      }
    }

    // CDF
    if (!cdf_slot_initialized) {
      cdf_slot_initialized = true;
      cdf_slot_doc_count = 0;
    }
    cdf_slot_doc_count += term_doc_count;
    if (cdf_slot_doc_count >= num_doc_per_cdf_slot) {
      create_cdf_slot();
      cdf_slot_initialized = false;
    }

    roaring_bitmap_free(bitmap_cur);
    bitmap_cur = nullptr;
    iter_term->Next();
  }


  // Finalize
  if (range_slot_initialized) {
    if (!create_range_slot().ok()) {
      return Status::InternalError();
    }
  }
  if (num_range_slot_created >= term_count) {
    LOG_DEBUG(
        "Drop range index in %s, range_slot_count[%ld] vs term_count[%ld].",
        ID().c_str(), num_range_slot_created, term_count);
    if (!ctx_.reset_cf(cf_name_ranges()).ok()) {
      LOG_ERROR("Failed to drop range index");
      return Status::InternalError();
    }
    cf_ranges_ = ctx_.get_cf(cf_name_ranges());
    if (!cf_ranges_) {
      LOG_ERROR("Failed to get cf_ranges for %s", ID().c_str());
      return Status::InternalError();
    }
  }

  if (cdf_slot_initialized) {
    iter_term->SeekToLast();
    create_cdf_slot();
  }
  ailego::JsonObject cdf_json_obj;
  cdf_json_obj.set("field_value_histogram", cdf_json_array);
  cdf_json_obj.set("total_doc_count", ailego::JsonValue(doc_count));
  ailego::JsonValue cdf_json(std::move(cdf_json_obj));
  rs = ctx_.db_->Put(ctx_.write_opts_, cf_cdf_, field_.name(),
                     cdf_json.as_json_string().as_stl_string());
  if (!rs.ok()) {
    LOG_ERROR("Failed to insert CDF of field[%s] to %s, code[%d], reason[%s]",
              field_.name().c_str(), ID().c_str(), rs.code(),
              rs.ToString().c_str());
    return Status::InternalError();
  }

  doc_range_stat_ =
      SegmentDocRangeStat::Create(cdf_json.as_json_string().as_stl_string());
  if (!doc_range_stat_) {
    LOG_ERROR("Failed to create doc range stats from %s", ID().c_str());
    return Status::InternalError();
  }

  LOG_INFO("Generated statistical indexes in %s", ID().c_str());
  return Status::OK();
}


Status InvertedColumnIndexer::seal() {
  if (read_only_) {
    return Status::PermissionDenied();
  }

  Status status = flush_special_values();
  if (!status.ok()) {
    LOG_ERROR("Failed to flush special values to %s", ID().c_str());
    return status;
  }

  if (enable_range_optimization_) {
    status = generate_statistical_indexes();
    if (!status.ok()) {
      LOG_ERROR("Failed to generate statistical indexes in %s", ID().c_str());
      return status;
    }
  }

  auto rs = ctx_.db_->Put(ctx_.write_opts_, key_sealed(), "sealed");
  if (rs.ok()) {
    sealed_ = true;
    read_only_ = true;
    return Status::OK();
  } else {
    LOG_ERROR("Failed to seal %s", ID().c_str());
    return Status::InternalError();
  }
}


}  // namespace zvec
