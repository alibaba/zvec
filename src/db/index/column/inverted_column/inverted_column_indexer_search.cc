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


#include <optional>
#include <ailego/pattern/defer.h>
#include <zvec/ailego/pattern/expected.hpp>
#include "inverted_codec.h"
#include "inverted_column_indexer.h"


namespace zvec {


Result<roaring_bitmap_t *> InvertedColumnIndexer::get_bitmap_eq(
    const std::string &term) const {
  PinnableSlice bitmap_slice;
  auto s = ctx_.db_->Get(ctx_.read_opts_, cf_terms_, term, &bitmap_slice);
  if (!s.ok()) {
    if (s.code() == rocksdb::Status::kNotFound) {
      return nullptr;
    }
    LOG_ERROR(
        "Failed to retrieve data for term[%s] from %s, code[%d], reason[%s]",
        term.c_str(), ID().c_str(), s.code(), s.ToString().c_str());
    return tl::make_unexpected(Status::InternalError());
  }

  roaring_bitmap_t *bitmap{nullptr};
  Status status = InvertedIndexCodec::Deserialize(bitmap_slice.data(),
                                                  bitmap_slice.size(), &bitmap);
  if (status.ok()) {
    return bitmap;
  } else {
    LOG_ERROR(
        "Failed to deserialize bitmap for term[%s] from %s, bitmap size[%zu]",
        term.c_str(), ID().c_str(), bitmap_slice.size());
    return tl::make_unexpected(Status::InternalError());
  }
}


Result<roaring_bitmap_t *> InvertedColumnIndexer::get_bitmap_contain(
    const std::vector<std::string> &terms, bool is_any) const {
  if (terms.empty()) {
    LOG_ERROR("Terms is empty");
    return tl::make_unexpected(Status::InvalidArgument());
  }

  // Shall we sort the terms here? Does it make any difference in performance?
  std::vector<Slice> slice_terms(terms.begin(), terms.end());
  std::vector<PinnableSlice> bitmap_slices;
  bitmap_slices.resize(terms.size());
  std::vector<rocksdb::Status> statuses;
  statuses.resize(terms.size());
  ctx_.db_->MultiGet(ctx_.read_opts_, cf_terms_, slice_terms.size(),
                     slice_terms.data(), bitmap_slices.data(), statuses.data(),
                     false);

  roaring_bitmap_t *bitmap{nullptr};
  Status s = Status::OK();
  AILEGO_DEFER([&]() {
    if (!s.ok() && bitmap) {
      roaring_bitmap_free(bitmap);
    }
  });

  auto init_or_merge_at_i = [&](size_t i) {
    if (bitmap == nullptr) {
      s = InvertedIndexCodec::Deserialize(bitmap_slices[i].data(),
                                          bitmap_slices[i].size(), &bitmap);
      if (!s.ok()) {
        LOG_ERROR("Failed to deserialize bitmap for term[%s] from %s",
                  terms[i].c_str(), ID().c_str());
      }
      return;
    }

    if (is_any) {
      s = InvertedIndexCodec::Merge_OR(bitmap_slices[i].data(),
                                       bitmap_slices[i].size(), true, bitmap);
    } else {
      s = InvertedIndexCodec::Merge_AND(bitmap_slices[i].data(),
                                        bitmap_slices[i].size(), bitmap);
    }
    if (!s.ok()) {
      LOG_ERROR("Failed to merge bitmap for term[%s] from %s", terms[i].c_str(),
                ID().c_str());
    }
  };

  for (size_t i = 0; i < terms.size(); i++) {
    if (statuses[i].ok()) {
      init_or_merge_at_i(i);
      if (!s.ok()) {
        return tl::make_unexpected(s);
      }
    } else if (statuses[i].code() == rocksdb::Status::kNotFound) {
      if (!is_any) {  // For contain_all, if any term is not found, return empty
        s = Status::NotFound();
        return nullptr;
      }
    } else {
      LOG_ERROR(
          "Failed to retrieve data for term[%s] from %s, code[%d], reason[%s]",
          terms[i].c_str(), ID().c_str(), statuses[i].code(),
          statuses[i].ToString().c_str());
      s = Status::InternalError();
      return tl::make_unexpected(s);
    }
  }

  if (is_any && bitmap) {
    roaring_bitmap_repair_after_lazy(bitmap);
  }
  return bitmap;
}


Result<roaring_bitmap_t *> InvertedColumnIndexer::get_bitmap_ne(
    const std::string &term) const {
  if (sealed_) {
    auto ret = get_bitmap_eq(term);
    if (ret) {
      ret = flip_bitmap(ret.value());
    } else {
      LOG_ERROR("Failed to retrieve bitmap for term[%s] from %s", term.c_str(),
                ID().c_str());
    }
    return ret;
  } else {
    roaring_bitmap_t *bitmap = roaring_bitmap_create();
    if (!bitmap) {
      LOG_ERROR("Failed to create bitmap");
      return tl::make_unexpected(Status::InternalError());
    }
    auto iter = ctx_.db_->NewIterator(ctx_.read_opts_, cf_terms_);
    AILEGO_DEFER([&]() { delete iter; });
    Status s;
    iter->SeekToFirst();
    while (iter->Valid()) {
      if (iter->key() == term) {
        iter->Next();
        continue;
      }
      s = InvertedIndexCodec::Merge_OR(iter->value().data(),
                                       iter->value().size(), true, bitmap);
      if (s.ok()) {
        iter->Next();
      } else {
        roaring_bitmap_free(bitmap);
        LOG_ERROR("Failed to merge bitmap from %s", ID().c_str());
        return tl::make_unexpected(s);
      }
    }
    roaring_bitmap_repair_after_lazy(bitmap);
    return bitmap;
  }
}


Result<roaring_bitmap_t *> InvertedColumnIndexer::get_bitmap_not_contain(
    const std::vector<std::string> &terms, bool is_any) const {
  if (terms.empty()) {
    LOG_ERROR("Terms is empty");
    return tl::make_unexpected(Status::InvalidArgument());
  }

  roaring_bitmap_t *non_null_bitmap{nullptr};
  AILEGO_DEFER([&]() {
    if (non_null_bitmap) {
      roaring_bitmap_free(non_null_bitmap);
    }
  });

  if (sealed_) {
    non_null_bitmap = null_bitmap_.copy();
    roaring_bitmap_flip_inplace(non_null_bitmap, 0, max_id_ + 1);
  } else {
    Status s;
    non_null_bitmap = roaring_bitmap_create();
    if (!non_null_bitmap) {
      LOG_ERROR("Failed to create bitmap");
      return tl::make_unexpected(Status::InternalError());
    }
    auto iter = ctx_.db_->NewIterator(ctx_.read_opts_, cf_terms_);
    AILEGO_DEFER([&]() { delete iter; });
    iter->SeekToFirst();
    while (iter->Valid()) {
      s = InvertedIndexCodec::Merge_OR(
          iter->value().data(), iter->value().size(), true, non_null_bitmap);
      if (s.ok()) {
        iter->Next();
      } else {
        LOG_ERROR("Failed to merge bitmap from %s", ID().c_str());
        return tl::make_unexpected(s);
      }
    }
    roaring_bitmap_repair_after_lazy(non_null_bitmap);
  }

  auto ret = get_bitmap_contain(terms, is_any);
  if (ret) {
    if (ret.value() == nullptr) {
      ret = roaring_bitmap_create();
    }
    roaring_bitmap_flip_inplace(ret.value(), 0, max_id_ + 1);
  } else {
    LOG_ERROR("Failed to retrieve bitmap[%s] from %s, term size[%zu]",
              is_any ? "contain_any" : "contain_all", ID().c_str(),
              terms.size());
    return ret;
  }

  roaring_bitmap_and_inplace(ret.value(), non_null_bitmap);
  return ret;
}


Result<roaring_bitmap_t *> InvertedColumnIndexer::get_bitmap_lt(
    const std::string &term, bool include_eq) const {
  if (field_.element_data_type() == DataType::BOOL) {
    LOG_ERROR("Bool type is not supported for range query");
    return tl::make_unexpected(Status::InternalError());
  }

  // For range queries that match most values, it's more efficient to compute
  // the result by getting the complement and flipping bits.
  if (range_covers_most_values(term, CompareOp::LT)) {
    auto ret = get_bitmap_gt(term, !include_eq);
    if (ret) {
      ret = flip_bitmap(ret.value());
    } else {
      LOG_ERROR("Failed to retrieve range bitmap for term[%s] from %s",
                term.c_str(), ID().c_str());
    }
    return ret;
  }

  Status s = Status::OK();
  roaring_bitmap_t *bitmap = roaring_bitmap_create();
  if (!bitmap) {
    LOG_ERROR("Failed to create bitmap");
    return tl::make_unexpected(Status::InternalError());
  }
  AILEGO_DEFER([&]() {
    if (!s.ok()) {
      roaring_bitmap_free(bitmap);
    }
  });

  rocksdb::Iterator *iter_point, *iter_range;
  iter_point = ctx_.db_->NewIterator(ctx_.read_opts_, cf_terms_);
  iter_point->SeekToFirst();
  AILEGO_DEFER([&]() { delete iter_point; });
  if (sealed_ && cf_ranges_) {
    iter_range = ctx_.db_->NewIterator(ctx_.read_opts_, cf_ranges_);
    iter_range->SeekToFirst();
  } else {
    iter_range = nullptr;
  }
  AILEGO_DEFER([&]() {
    if (iter_range) {
      delete iter_range;
    }
  });

  bool lt;  // True if the current range or term is "less than" the search term

  // Process pre-aggregated range entries and merge matching bitmaps
  if (iter_range && iter_range->Valid()) {
    std::optional<std::string> point_seek_start_pos;
    // 1. Merge ranges where the end boundary is less than the search term
    while (iter_range->Valid()) {
      char *range_begin, *range_end;
      size_t range_begin_key_size, range_end_key_size;
      InvertedIndexCodec::Decode_Range_Key(
          iter_range->key().data(), iter_range->key().size(), &range_begin,
          &range_begin_key_size, &range_end, &range_end_key_size);
      lt = cmp_lt(range_end, range_end_key_size, term.data(), term.length(),
                  include_eq);
      if (!lt) {
        point_seek_start_pos.emplace(range_begin, range_begin_key_size);
        break;
      }
      s = InvertedIndexCodec::Merge_OR(
          iter_range->value().data(), iter_range->value().size(), true, bitmap);
      if (!s.ok()) {
        LOG_ERROR("Failed to merge range bitmap from %s", ID().c_str());
        return tl::make_unexpected(s);
      }
      iter_range->Next();
    }
    // 2. Change the start position of the point iterator
    if (point_seek_start_pos) {
      iter_point->Seek(*point_seek_start_pos);
      if (iter_point->Valid() && iter_point->key() != *point_seek_start_pos) {
        LOG_ERROR(
            "Failed to initialize the point iterator, seek_pos[%s], "
            "first_key_found[%s], term[%s]",
            (*point_seek_start_pos).c_str(),
            iter_point->key().ToStringView().data(), term.c_str());
        s = Status::InternalError();
        return tl::make_unexpected(s);
      }
    } else {
      iter_point->SeekToLast();
    }
  }

  // Process individual point entries
  while (iter_point->Valid()) {
    lt = cmp_lt(iter_point->key().data(), iter_point->key().size(), term.data(),
                term.size(), include_eq);
    if (!lt) {
      break;
    }
    s = InvertedIndexCodec::Merge_OR(iter_point->value().data(),
                                     iter_point->value().size(), true, bitmap);
    if (!s.ok()) {
      LOG_ERROR("Failed to merge range bitmap from %s", ID().c_str());
      return tl::make_unexpected(s);
    }
    iter_point->Next();
  }

  roaring_bitmap_repair_after_lazy(bitmap);
  return bitmap;
}


Result<roaring_bitmap_t *> InvertedColumnIndexer::get_bitmap_gt(
    const std::string &term, bool include_eq) const {
  if (field_.element_data_type() == DataType::BOOL) {
    LOG_ERROR("Bool type is not supported for range query");
    return tl::make_unexpected(Status::InternalError());
  }

  // For range queries that match most values, it's more efficient to compute
  // the result by getting the complement and flipping bits.
  if (range_covers_most_values(term, CompareOp::GT)) {
    auto ret = get_bitmap_lt(term, !include_eq);
    if (ret) {
      ret = flip_bitmap(ret.value());
    } else {
      LOG_ERROR("Failed to retrieve range bitmap for term[%s] from %s",
                term.c_str(), ID().c_str());
    }
    return ret;
  }

  Status s = Status::OK();
  roaring_bitmap_t *bitmap = roaring_bitmap_create();
  if (!bitmap) {
    LOG_ERROR("Failed to create bitmap");
    return tl::make_unexpected(Status::InternalError());
  }
  AILEGO_DEFER([&]() {
    if (!s.ok()) {
      roaring_bitmap_free(bitmap);
    }
  });

  rocksdb::Iterator *iter_point, *iter_range;
  iter_point = ctx_.db_->NewIterator(ctx_.read_opts_, cf_terms_);
  AILEGO_DEFER([&]() { delete iter_point; });
  if (sealed_ && cf_ranges_) {
    iter_range = ctx_.db_->NewIterator(ctx_.read_opts_, cf_ranges_);
  } else {
    iter_range = nullptr;
  }
  AILEGO_DEFER([&]() {
    if (iter_range) {
      delete iter_range;
    }
  });

  std::optional<std::string> point_seek_end_pos;

  // Process pre-aggregated range entries and merge matching bitmaps
  if (iter_range) {
    // 1. Seek to the first range entry that is greater than the search term
    iter_range->Seek(term);
    if (iter_range->Valid()) {
      char *range_begin, *range_end;
      size_t range_begin_key_size, range_end_key_size;
      InvertedIndexCodec::Decode_Range_Key(
          iter_range->key().data(), iter_range->key().size(), &range_begin,
          &range_begin_key_size, &range_end, &range_end_key_size);
      int ret =
          cmp(range_begin, range_begin_key_size, term.data(), term.size());
      if (ret == 0 && !include_eq) {
        iter_range->Next();
        if (iter_range->Valid()) {
          InvertedIndexCodec::Decode_Range_Key(
              iter_range->key().data(), iter_range->key().size(), &range_begin,
              &range_begin_key_size, &range_end, &range_end_key_size);
          point_seek_end_pos.emplace(range_begin, range_begin_key_size);
        }
      } else {
        point_seek_end_pos.emplace(range_begin, range_begin_key_size);
      }
    }
    // 2. Merge ranges where the begin boundary is greater than the search term
    while (iter_range->Valid()) {
      s = InvertedIndexCodec::Merge_OR(
          iter_range->value().data(), iter_range->value().size(), true, bitmap);
      if (!s.ok()) {
        LOG_ERROR("Failed to merge range bitmap from %s", ID().c_str());
        return tl::make_unexpected(s);
      }
      iter_range->Next();
    }
  }

  // Process individual point entries
  iter_point->Seek(term);
  if (!include_eq) {
    if (iter_point->Valid() && iter_point->key() == term) {
      iter_point->Next();
    }
  }
  while (iter_point->Valid()) {
    if (point_seek_end_pos &&
        cmp(iter_point->key().data(), iter_point->key().size(),
            (*point_seek_end_pos).data(), (*point_seek_end_pos).size()) >= 0) {
      break;
    }
    s = InvertedIndexCodec::Merge_OR(iter_point->value().data(),
                                     iter_point->value().size(), true, bitmap);
    if (!s.ok()) {
      LOG_ERROR("Failed to merge range bitmap from %s", ID().c_str());
      return tl::make_unexpected(s);
    }
    iter_point->Next();
  }

  roaring_bitmap_repair_after_lazy(bitmap);
  return bitmap;
}


Result<roaring_bitmap_t *> InvertedColumnIndexer::get_bitmap_array_len_eq(
    uint32_t len) const {
  std::string encoded_len = InvertedIndexCodec::Encode(
      std::string((char *)&len, sizeof(uint32_t)), DataType::UINT32);

  PinnableSlice bitmap_slice;
  auto rs =
      ctx_.db_->Get(ctx_.read_opts_, cf_array_len_, encoded_len, &bitmap_slice);
  if (!rs.ok()) {
    if (rs.code() == rocksdb::Status::kNotFound) {
      return nullptr;
    }
    LOG_ERROR(
        "Failed to retrieve data for len[%u] from %s, code[%d], reason[%s]",
        len, ID().c_str(), rs.code(), rs.ToString().c_str());
    return tl::make_unexpected(Status::InternalError());
  }

  roaring_bitmap_t *bitmap{nullptr};
  Status status = InvertedIndexCodec::Deserialize(bitmap_slice.data(),
                                                  bitmap_slice.size(), &bitmap);
  if (status.ok()) {
    return bitmap;
  } else {
    LOG_ERROR(
        "Failed to deserialize bitmap for len[%u] from %s, bitmap size[%zu]",
        len, ID().c_str(), bitmap_slice.size());
    return tl::make_unexpected(Status::InternalError());
  }
}


Result<roaring_bitmap_t *> InvertedColumnIndexer::get_bitmap_array_len_ne(
    uint32_t len) const {
  std::string encoded_len = InvertedIndexCodec::Encode(
      std::string((char *)&len, sizeof(uint32_t)), DataType::UINT32);

  roaring_bitmap_t *bitmap = roaring_bitmap_create();
  if (!bitmap) {
    LOG_ERROR("Failed to create bitmap");
    return tl::make_unexpected(Status::InternalError());
  }
  auto iter = ctx_.db_->NewIterator(ctx_.read_opts_, cf_array_len_);
  AILEGO_DEFER([&]() { delete iter; });
  Status s;
  iter->SeekToFirst();
  while (iter->Valid()) {
    if (iter->key() == encoded_len) {
      iter->Next();
      continue;
    }
    s = InvertedIndexCodec::Merge_OR(iter->value().data(), iter->value().size(),
                                     true, bitmap);
    if (s.ok()) {
      iter->Next();
    } else {
      roaring_bitmap_free(bitmap);
      LOG_ERROR("Failed to merge bitmap from %s", ID().c_str());
      return tl::make_unexpected(s);
    }
  }
  roaring_bitmap_repair_after_lazy(bitmap);
  return bitmap;
}


Result<roaring_bitmap_t *> InvertedColumnIndexer::get_bitmap_array_len_lt(
    uint32_t len, bool include_eq) const {
  std::string encoded_len = InvertedIndexCodec::Encode(
      std::string((char *)&len, sizeof(uint32_t)), DataType::UINT32);

  roaring_bitmap_t *bitmap = roaring_bitmap_create();
  if (!bitmap) {
    LOG_ERROR("Failed to create bitmap");
    return tl::make_unexpected(Status::InternalError());
  }
  auto iter = ctx_.db_->NewIterator(ctx_.read_opts_, cf_array_len_);
  AILEGO_DEFER([&]() { delete iter; });
  Status s;
  iter->SeekToFirst();
  while (iter->Valid()) {
    bool lt = cmp_lt(iter->key().data(), iter->key().size(), encoded_len.data(),
                     encoded_len.size(), include_eq);
    if (!lt) {
      break;
    }
    s = InvertedIndexCodec::Merge_OR(iter->value().data(), iter->value().size(),
                                     true, bitmap);
    if (s.ok()) {
      iter->Next();
    } else {
      roaring_bitmap_free(bitmap);
      LOG_ERROR("Failed to merge bitmap from %s", ID().c_str());
      return tl::make_unexpected(s);
    }
  }
  roaring_bitmap_repair_after_lazy(bitmap);
  return bitmap;
}


Result<roaring_bitmap_t *> InvertedColumnIndexer::get_bitmap_array_len_gt(
    uint32_t len, bool include_eq) const {
  std::string encoded_len = InvertedIndexCodec::Encode(
      std::string((char *)&len, sizeof(uint32_t)), DataType::UINT32);

  roaring_bitmap_t *bitmap = roaring_bitmap_create();
  if (!bitmap) {
    LOG_ERROR("Failed to create bitmap");
    return tl::make_unexpected(Status::InternalError());
  }
  auto iter = ctx_.db_->NewIterator(ctx_.read_opts_, cf_array_len_);
  AILEGO_DEFER([&]() { delete iter; });
  Status s;
  iter->Seek(encoded_len);
  if (!include_eq) {
    if (iter->Valid() && iter->key() == encoded_len) {
      iter->Next();
    }
  }
  while (iter->Valid()) {
    s = InvertedIndexCodec::Merge_OR(iter->value().data(), iter->value().size(),
                                     true, bitmap);
    if (s.ok()) {
      iter->Next();
    } else {
      roaring_bitmap_free(bitmap);
      LOG_ERROR("Failed to merge bitmap from %s", ID().c_str());
      return tl::make_unexpected(s);
    }
  }
  roaring_bitmap_repair_after_lazy(bitmap);
  return bitmap;
}


Result<roaring_bitmap_t *> InvertedColumnIndexer::get_bitmap_like(
    std::string term) const {
  // convert to `is not null` if `%` is the only character
  if (term == "%") {
    return get_bitmap_non_null();
  }
  size_t percent_loc = std::string::npos;
  size_t size = 0;
  int percent_count = 0;
  // unescape \% and \_, detect % location
  for (size_t i = 0; i < term.size(); i++) {
    if (term[i] == '\\') {
      i++;
      if (i < term.size()) {
        term[size++] = term[i];
      }
      continue;
    }
    if (term[i] == '%') {
      percent_loc = size;
      percent_count += 1;
    }
    term[size++] = term[i];
  }
  term.resize(size);
  // convert to `=` filter if no percent
  if (percent_count == 0) {
    return get_bitmap_eq(term);
  } else if (percent_count != 1) {
    return tl::make_unexpected(Status::InvalidArgument(
        "like should have exactly one percent, unescaped:", term));
  }
  if (percent_loc == 0) {
    return get_bitmap_suffix(term);
  } else if (percent_loc == size - 1) {
    return get_bitmap_prefix(term.substr(0, percent_loc));
  } else {
    std::string prefix = term.substr(0, percent_loc - 1);
    std::string suffix = term.substr(percent_loc + 1, size - percent_loc - 1);
    auto prefix_bitmap = get_bitmap_prefix(prefix);
    if (!prefix_bitmap.has_value()) {
      return tl::make_unexpected(
          Status::InternalError("Get bitmap prefix failed, unescaped:", term));
    }
    auto suffix_bitmap = get_bitmap_suffix(suffix);
    if (!suffix_bitmap.has_value()) {
      return tl::make_unexpected(
          Status::InternalError("Get bitmap suffix failed, unescaped:", term));
    }
    auto *result = prefix_bitmap.value();
    roaring_bitmap_and_inplace(result, suffix_bitmap.value());
    roaring_bitmap_free(suffix_bitmap.value());
    return result;
  }
}


Result<roaring_bitmap_t *> InvertedColumnIndexer::get_bitmap_prefix(
    const std::string &term) const {
  auto iter = ctx_.db_->NewIterator(ctx_.read_opts_, cf_terms_);
  AILEGO_DEFER([&]() { delete iter; });

  roaring_bitmap_t *bitmap = roaring_bitmap_create();
  if (!bitmap) {
    LOG_ERROR("Failed to create bitmap");
    return tl::make_unexpected(Status::InternalError());
  }

  Status s;
  iter->Seek(term);
  while (iter->Valid()) {
    if (!has_prefix(iter->key().data(), iter->key().size(), term.data(),
                    term.size())) {
      break;
    }
    s = InvertedIndexCodec::Merge_OR(iter->value().data(), iter->value().size(),
                                     true, bitmap);
    if (!s.ok()) {
      roaring_bitmap_free(bitmap);
      LOG_ERROR("Failed to merge range bitmap from %s", ID().c_str());
      return tl::make_unexpected(s);
    }
    iter->Next();
  }

  roaring_bitmap_repair_after_lazy(bitmap);
  return bitmap;
}


Result<roaring_bitmap_t *> InvertedColumnIndexer::get_bitmap_suffix(
    const std::string &term) const {
  if (!cf_reversed_terms_) {
    LOG_ERROR("%s doesn't support suffix matching", ID().c_str());
    return tl::make_unexpected(Status::PermissionDenied());
  }

  auto iter = ctx_.db_->NewIterator(ctx_.read_opts_, cf_reversed_terms_);
  AILEGO_DEFER([&]() { delete iter; });

  roaring_bitmap_t *bitmap = roaring_bitmap_create();
  if (!bitmap) {
    LOG_ERROR("Failed to create bitmap");
    return tl::make_unexpected(Status::InternalError());
  }

  Status s;
  auto reversed_term = encode_reversed(term);
  iter->Seek(reversed_term);
  while (iter->Valid()) {
    if (!has_prefix(iter->key().data(), iter->key().size(),
                    reversed_term.data(), reversed_term.size())) {
      break;
    }
    s = InvertedIndexCodec::Merge_OR(iter->value().data(), iter->value().size(),
                                     true, bitmap);
    if (!s.ok()) {
      roaring_bitmap_free(bitmap);
      LOG_ERROR("Failed to merge range bitmap from %s", ID().c_str());
      return tl::make_unexpected(s);
    }
    iter->Next();
  }

  roaring_bitmap_repair_after_lazy(bitmap);
  return bitmap;
}


Result<roaring_bitmap_t *> InvertedColumnIndexer::get_bitmap_null() const {
  return null_bitmap_.copy();
}


Result<roaring_bitmap_t *> InvertedColumnIndexer::get_bitmap_non_null() const {
  if (sealed_) {
    roaring_bitmap_t *bitmap = null_bitmap_.copy();
    roaring_bitmap_flip_inplace(bitmap, 0, max_id_ + 1);
    return bitmap;
  } else {
    Status s = Status::OK();
    auto iter = ctx_.db_->NewIterator(ctx_.read_opts_, cf_terms_);
    AILEGO_DEFER([&]() { delete iter; });
    roaring_bitmap_t *bitmap = roaring_bitmap_create();
    if (!bitmap) {
      LOG_ERROR("Failed to create bitmap");
      return tl::make_unexpected(Status::InternalError());
    }
    iter->SeekToFirst();
    while (iter->Valid()) {
      s = InvertedIndexCodec::Merge_OR(iter->value().data(),
                                       iter->value().size(), true, bitmap);
      if (s.ok()) {
        iter->Next();
      } else {
        roaring_bitmap_free(bitmap);
        LOG_ERROR("Failed to merge bitmap from %s", ID().c_str());
        return tl::make_unexpected(s);
      }
    }
    roaring_bitmap_repair_after_lazy(bitmap);
    return bitmap;
  }
}


InvertedSearchResult::Ptr InvertedColumnIndexer::search(
    const std::string &value, CompareOp op) const {
  if (field_.is_array_type()) {
    LOG_ERROR("%s: array type doesn't support single value search",
              ID().c_str());
    return nullptr;
  }

  std::string encoded_value = encode(value);
  auto search_res = std::make_shared<InvertedSearchResult>();
  Result<roaring_bitmap_t *> bitmap_res;

  switch (op) {
    case CompareOp::EQ: {
      bitmap_res = get_bitmap_eq(encoded_value);
      break;
    }
    case CompareOp::NE: {
      bitmap_res = get_bitmap_ne(encoded_value);
      break;
    }
    case CompareOp::LT: {
      bitmap_res = get_bitmap_lt(encoded_value, false);
      break;
    }
    case CompareOp::LE: {
      bitmap_res = get_bitmap_lt(encoded_value, true);
      break;
    }
    case CompareOp::GT: {
      bitmap_res = get_bitmap_gt(encoded_value, false);
      break;
    }
    case CompareOp::GE: {
      bitmap_res = get_bitmap_gt(encoded_value, true);
      break;
    }
    case CompareOp::LIKE: {
      bitmap_res = get_bitmap_like(std::move(encoded_value));
      break;
    }
    case CompareOp::HAS_PREFIX: {
      bitmap_res = get_bitmap_prefix(std::move(encoded_value));
      break;
    }
    case CompareOp::HAS_SUFFIX: {
      bitmap_res = get_bitmap_suffix(std::move(encoded_value));
      break;
    }
    default:
      LOG_ERROR("%s: unsupported operator[%u]", ID().c_str(),
                static_cast<uint32_t>(op));
      return nullptr;
  }

  if (bitmap_res) {
    search_res->set_and_own_bitmap(bitmap_res.value());
    return search_res;
  } else {
    LOG_ERROR("%s: failed to search, code[%d]", ID().c_str(),
              static_cast<int>(bitmap_res.error().code()));
    return nullptr;
  }
}


InvertedSearchResult::Ptr InvertedColumnIndexer::multi_search(
    const std::vector<std::string> &values, CompareOp op) const {
  auto encoded_values = encode(values);
  auto search_res = std::make_shared<InvertedSearchResult>();
  Result<roaring_bitmap_t *> bitmap_res;

  switch (op) {
    case CompareOp::CONTAIN_ANY: {
      bitmap_res = get_bitmap_contain(encoded_values, true);
      break;
    }
    case CompareOp::CONTAIN_ALL: {
      bitmap_res = get_bitmap_contain(encoded_values, false);
      break;
    }
    case CompareOp::NOT_CONTAIN_ANY: {
      bitmap_res = get_bitmap_not_contain(encoded_values, true);
      break;
    }
    case CompareOp::NOT_CONTAIN_ALL: {
      bitmap_res = get_bitmap_not_contain(encoded_values, false);
      break;
    }
    default:
      LOG_ERROR("%s: unsupported operator[%u]", ID().c_str(),
                static_cast<uint32_t>(op));
      return nullptr;
  }

  if (bitmap_res) {
    search_res->set_and_own_bitmap(bitmap_res.value());
    return search_res;
  } else {
    LOG_ERROR("%s: failed to search, code[%d]", ID().c_str(),
              static_cast<int>(bitmap_res.error().code()));
    return nullptr;
  }
}


InvertedSearchResult::Ptr InvertedColumnIndexer::search_array_len(
    uint32_t len, CompareOp op) const {
  if (!field_.is_array_type()) {
    LOG_ERROR("%s: non-array type doesn't array length search", ID().c_str());
    return nullptr;
  }

  auto search_res = std::make_shared<InvertedSearchResult>();
  Result<roaring_bitmap_t *> bitmap_res;

  switch (op) {
    case CompareOp::EQ: {
      bitmap_res = get_bitmap_array_len_eq(len);
      break;
    }
    case CompareOp::NE: {
      bitmap_res = get_bitmap_array_len_ne(len);
      break;
    }
    case CompareOp::LT: {
      bitmap_res = get_bitmap_array_len_lt(len, false);
      break;
    }
    case CompareOp::LE: {
      bitmap_res = get_bitmap_array_len_lt(len, true);
      break;
    }
    case CompareOp::GT: {
      bitmap_res = get_bitmap_array_len_gt(len, false);
      break;
    }
    case CompareOp::GE: {
      bitmap_res = get_bitmap_array_len_gt(len, true);
      break;
    }
    default:
      LOG_ERROR("%s: unsupported operator[%u]", ID().c_str(),
                static_cast<uint32_t>(op));
      return nullptr;
  }

  if (bitmap_res) {
    search_res->set_and_own_bitmap(bitmap_res.value());
    return search_res;
  } else {
    LOG_ERROR("%s: failed to search, code[%d]", ID().c_str(),
              static_cast<int>(bitmap_res.error().code()));
    return nullptr;
  }
}


InvertedSearchResult::Ptr InvertedColumnIndexer::search_null() const {
  auto search_res = std::make_shared<InvertedSearchResult>();
  auto bitmap_res = get_bitmap_null();
  if (bitmap_res) {
    search_res->set_and_own_bitmap(bitmap_res.value());
    return search_res;
  } else {
    LOG_ERROR("%s: failed to search, code[%d]", ID().c_str(),
              static_cast<int>(bitmap_res.error().code()));
    return nullptr;
  }
}


InvertedSearchResult::Ptr InvertedColumnIndexer::search_non_null() const {
  auto search_res = std::make_shared<InvertedSearchResult>();
  auto bitmap_res = get_bitmap_non_null();
  if (bitmap_res) {
    search_res->set_and_own_bitmap(bitmap_res.value());
    return search_res;
  } else {
    LOG_ERROR("%s: failed to search, code[%d]", ID().c_str(),
              static_cast<int>(bitmap_res.error().code()));
    return nullptr;
  }
}


Status InvertedColumnIndexer::evaluate_ratio(const std::string &value,
                                             CompareOp op, uint64_t *total_size,
                                             uint64_t *range_size) const {
  if (field_.is_array_type()) {
    LOG_ERROR("%s: array type doesn't support ratio evaluation", ID().c_str());
    return Status::PermissionDenied();
  }

  if (sealed_ && doc_range_stat_) {
    std::string encoded_value = encode(value);
    doc_range_stat_->evaluate_ratio(encoded_value, op, total_size, range_size);
  } else {
    *range_size = 0;
    *total_size = 1;
  }
  return Status::OK();
}


inline Status InvertedColumnIndexer::estimate_range_ratio(
    const std::string &term, CompareOp op, uint64_t *total_count,
    uint64_t *matching_count) const {
  if (field_.is_array_type() || field_.element_data_type() == DataType::BOOL) {
    LOG_ERROR("%s: type[%d] doesn't support range ratio estimation",
              ID().c_str(), (int)field_.data_type());
    return Status::PermissionDenied();
  }

  if (sealed_ && doc_range_stat_) {
    doc_range_stat_->evaluate_ratio(term, op, total_count, matching_count);
  } else {
    *matching_count = 0;
    *total_count = 1;
  }
  return Status::OK();
}


inline bool InvertedColumnIndexer::range_covers_most_values(
    const std::string &term, CompareOp op) const {
  constexpr float HIGH_SELECTIVITY_THRESHOLD = 0.7;

  // Estimation is only available for sealed indexes as they have the cumulative
  // distribution index.
  if (!sealed_) {
    return false;
  }

  uint64_t total_cnt{0}, matching_cnt{0};
  if (auto s = estimate_range_ratio(term, op, &total_cnt, &matching_cnt);
      s.ok()) {
    return (total_cnt != 0) &&
           ((1.0f * matching_cnt / total_cnt) > HIGH_SELECTIVITY_THRESHOLD);

  } else {
    return false;
  }
}


inline roaring_bitmap_t *InvertedColumnIndexer::flip_bitmap(
    roaring_bitmap_t *bitmap) const {
  roaring_bitmap_t *ret;
  if (ret = bitmap; ret == nullptr) {
    ret = null_bitmap_.copy();
  } else {
    roaring_bitmap_or_inplace(ret, null_bitmap_.bitmap());
  }
  roaring_bitmap_flip_inplace(ret, 0, max_id_ + 1);
  return ret;
}


}  // namespace zvec