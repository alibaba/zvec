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


#pragma once


#include <memory>
#include <string>
#include <vector>
#include <zvec/ailego/encoding/json/mod_json_plus.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/db/type.h>


namespace zvec {


struct DocRange {
  std::string key_{""};
  size_t doc_count_{0};

  DocRange() {}

  DocRange(const std::string &key, int count) : key_(key), doc_count_(count) {}

  bool operator<(const std::string &key) const {
    return key_ < key;
  }
};


class SegmentDocRangeStat {
 public:
  using Ptr = std::shared_ptr<SegmentDocRangeStat>;


  SegmentDocRangeStat(std::vector<DocRange> &&doc_ranges,
                      uint64_t total_doc_count)
      : doc_ranges_(std::move(doc_ranges)), total_doc_count_(total_doc_count) {
    std::sort(
        doc_ranges_.begin(), doc_ranges_.end(),
        [](const DocRange &a, const DocRange &b) { return a.key_ < b.key_; });
  }


  static Ptr Create(const std::string &stat_json_str) {
    ailego::JsonValue stat_json_value;
    ailego::JsonParser parser;
    if (!parser.parse(stat_json_str.c_str(), &stat_json_value)) {
      LOG_ERROR("Failed to parse json string");
      return nullptr;
    }

    ailego::JsonObject stat_json_obj = stat_json_value.as_object();
    ailego::JsonArray stat_json_array;
    if (!stat_json_obj.get("field_value_histogram", &stat_json_array)) {
      LOG_ERROR("Failed to get histogram");
      return nullptr;
    }

    ailego::JsonValue stat_total_doc_count;
    if (!stat_json_obj.get("total_doc_count", &stat_total_doc_count)) {
      LOG_ERROR("Failed to get total doc count");
      return nullptr;
    }

    std::vector<DocRange> doc_ranges;
    for (auto it = stat_json_array.begin(); it != stat_json_array.end(); ++it) {
      ailego::JsonString stat_key;
      ailego::JsonValue stat_doc_count;
      if (!it->as_object().get("key", &stat_key)) {
        LOG_ERROR("Failed to get key");
        return nullptr;
      }
      if (!it->as_object().get("doc_count", &stat_doc_count)) {
        LOG_ERROR("Failed to get doc count");
        return nullptr;
      }
      doc_ranges.emplace_back(stat_key.decode().as_stl_string(),
                              stat_doc_count.as_integer());
    }

    return std::make_shared<SegmentDocRangeStat>(
        std::move(doc_ranges), stat_total_doc_count.as_integer());
  }


  void evaluate_ratio(const std::string &value, CompareOp op,
                      uint64_t *total_size, uint64_t *range_size) const {
    if (doc_ranges_.size() == 0) {
      *range_size = 0;
      *total_size = total_doc_count_;
    }

    // Is greater than?
    bool is_gt = (op == CompareOp::GT) || (op == CompareOp::GE);

    auto it = std::lower_bound(doc_ranges_.begin(), doc_ranges_.end(), value);

    if (it == doc_ranges_.end()) {
      *range_size = is_gt ? 0 : total_doc_count_;
    } else {
      *range_size = is_gt ? total_doc_count_ - it->doc_count_ : it->doc_count_;
    }
    *total_size = total_doc_count_;
  }


 private:
  std::vector<DocRange> doc_ranges_;
  uint64_t total_doc_count_;
};


}  // namespace zvec