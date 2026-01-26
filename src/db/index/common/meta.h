
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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <vector>
#include "db/common/utils.h"
#include "db/index/common/type_helper.h"

namespace zvec {

using SegmentID = uint32_t;
using BlockID = uint32_t;

class BlockMeta {
 public:
  using Ptr = std::shared_ptr<BlockMeta>;

 public:
  BlockMeta() = default;

  BlockMeta(uint32_t id, BlockType type, uint64_t min_doc_id,
            uint64_t max_doc_id, uint32_t doc_count,
            const std::vector<std::string> &columns)
      : id_(id),
        type_(type),
        min_doc_id_(min_doc_id),
        max_doc_id_(max_doc_id),
        doc_count_(doc_count),
        columns_(columns) {}

  BlockMeta(uint32_t id, BlockType type, uint64_t min_doc_id,
            uint64_t max_doc_id)
      : id_(id),
        type_(type),
        min_doc_id_(min_doc_id),
        max_doc_id_(max_doc_id) {}
  uint32_t id() const {
    return id_;
  }

  void set_id(uint32_t id) {
    id_ = id;
  }

  BlockType type() const {
    return type_;
  }

  void set_type(BlockType type) {
    type_ = type;
  }

  uint64_t min_doc_id() const {
    return min_doc_id_;
  }

  void set_min_doc_id(uint64_t min_doc_id) {
    min_doc_id_ = min_doc_id;
  }

  uint64_t max_doc_id() const {
    return max_doc_id_;
  }

  void set_max_doc_id(uint64_t max_doc_id) {
    max_doc_id_ = max_doc_id;
  }

  uint32_t doc_count() const {
    return doc_count_;
  }

  void set_doc_count(uint32_t doc_count) {
    doc_count_ = doc_count;
  }

  const std::vector<std::string> &columns() const {
    return columns_;
  }

  void set_columns(const std::vector<std::string> &columns) {
    columns_ = columns;
  }

  void add_column(const std::string &column) {
    columns_.push_back(column);
  }

  void del_column(const std::string &column) {
    columns_.erase(std::remove(columns_.begin(), columns_.end(), column),
                   columns_.end());
  }

  bool contain_column(const std::string &column) const {
    return std::find(columns_.begin(), columns_.end(), column) !=
           columns_.end();
  }

 public:
  bool operator==(const BlockMeta &other) const {
    return id_ == other.id_ && type_ == other.type_ &&
           min_doc_id_ == other.min_doc_id_ &&
           max_doc_id_ == other.max_doc_id_ && columns_ == other.columns_ &&
           doc_count_ == other.doc_count_;
  }

  std::string to_string() const {
    std::ostringstream oss;
    oss << "BlockMeta{"
        << "id:" << id_ << ",type:" << BlockTypeCodeBook::AsString(type_)
        << ",min_doc_id:" << min_doc_id_ << ",max_doc_id:" << max_doc_id_
        << ",doc_count:" << doc_count_ << ",columns:[";

    for (size_t i = 0; i < columns_.size(); ++i) {
      if (i > 0) oss << ",";
      oss << "'" << columns_[i] << "'";
    }

    oss << "]}";
    return oss.str();
  }

  std::string to_string_formatted(int indent_level = 0) const {
    std::ostringstream oss;
    oss << indent(indent_level) << "BlockMeta{\n"
        << indent(indent_level + 1) << "id: " << id_ << ",\n"
        << indent(indent_level + 1)
        << "type: " << BlockTypeCodeBook::AsString(type_) << ",\n"
        << indent(indent_level + 1) << "min_doc_id: " << min_doc_id_ << ",\n"
        << indent(indent_level + 1) << "max_doc_id: " << max_doc_id_ << ",\n"
        << indent(indent_level + 1) << "doc_count: " << doc_count_ << ",\n"
        << indent(indent_level + 1) << "columns: [";

    if (!columns_.empty()) {
      oss << "\n";
      for (size_t i = 0; i < columns_.size(); ++i) {
        oss << indent(indent_level + 2) << "'" << columns_[i] << "'";
        if (i < columns_.size() - 1) {
          oss << ",";
        }
        oss << "\n";
      }
      oss << indent(indent_level + 1);
    }

    oss << "]\n" << indent(indent_level) << "}";
    return oss.str();
  }

 public:
  uint32_t id_{0};
  BlockType type_{BlockType::UNDEFINED};
  uint64_t min_doc_id_{0};
  uint64_t max_doc_id_{0};
  uint32_t doc_count_{0};
  std::vector<std::string> columns_{};
};

class SegmentMeta {
 public:
  using Ptr = std::shared_ptr<SegmentMeta>;

 public:
  SegmentMeta() {};

  explicit SegmentMeta(SegmentID id) : id_(id) {}

  void set_id(SegmentID id) {
    id_ = id;
  }

  SegmentID id() const {
    return id_;
  }

  void add_persisted_block(const BlockMeta &block) {
    persisted_blocks_.push_back(block);
  }

  void set_persisted_blocks(const std::vector<BlockMeta> &blocks) {
    persisted_blocks_ = blocks;
  }

  bool remove_block(BlockID block_id) {
    auto it = std::remove_if(
        persisted_blocks_.begin(), persisted_blocks_.end(),
        [block_id](const BlockMeta &block) { return block.id() == block_id; });
    bool found = (it != persisted_blocks_.end());
    persisted_blocks_.erase(it, persisted_blocks_.end());
    return found;
  }

  void remove_vector_persisted_block(const std::string &column, bool quantize) {
    std::vector<BlockMeta> new_persisted_blocks;
    for (auto &b : persisted_blocks_) {
      if (quantize) {
        if (!(b.type() == BlockType::VECTOR_INDEX_QUANTIZE &&
              b.contain_column(column))) {
          new_persisted_blocks.push_back(b);
        }
      } else {
        if (!(b.type() == BlockType::VECTOR_INDEX &&
              b.contain_column(column))) {
          new_persisted_blocks.push_back(b);
        }
      }
    }
    persisted_blocks_ = new_persisted_blocks;
  }

  void remove_vector_persisted_block(const std::string &column) {
    std::vector<BlockMeta> new_persisted_blocks;
    for (auto &b : persisted_blocks_) {
      if (!b.contain_column(column)) {
        new_persisted_blocks.push_back(b);
      }
    }
    persisted_blocks_ = new_persisted_blocks;
  }

  void remove_scalar_index_block() {
    std::vector<BlockMeta> new_persisted_blocks;
    for (auto &b : persisted_blocks_) {
      if (b.type() != BlockType::SCALAR_INDEX) {
        new_persisted_blocks.push_back(b);
      }
    }
    persisted_blocks_ = new_persisted_blocks;
  }

  void set_writing_forward_block(const BlockMeta &writing_forward_block) {
    writing_forward_block_ = writing_forward_block;
  }

  void remove_writing_forward_block() {
    writing_forward_block_ = std::nullopt;
  }

  void update_max_doc_id(uint64_t max_doc_id) {
    if (writing_forward_block_.has_value()) {
      writing_forward_block_->set_max_doc_id(max_doc_id);
    }
  }

  uint64_t min_doc_id() const {
    if (persisted_blocks_.empty()) {
      if (writing_forward_block_.has_value()) {
        return writing_forward_block_->min_doc_id();
      }
      return 0;
    }
    uint64_t min_doc_id{std::numeric_limits<uint64_t>::max()};
    for (const auto &block : persisted_blocks_) {
      if (block.type() == BlockType::SCALAR) {
        min_doc_id = std::min(min_doc_id, block.min_doc_id());
      }
    }
    if (min_doc_id == std::numeric_limits<uint64_t>::max() &&
        writing_forward_block_.has_value()) {
      min_doc_id = writing_forward_block_->min_doc_id();
    }
    return min_doc_id;
  }

  uint64_t max_doc_id() const {
    if (writing_forward_block_.has_value() &&
        writing_forward_block_->doc_count_ != 0) {
      return writing_forward_block_->max_doc_id();
    }
    uint64_t max_doc_id{0};
    for (const auto &block : persisted_blocks_) {
      if (block.type() == BlockType::SCALAR) {
        max_doc_id = std::max(max_doc_id, block.max_doc_id());
      }
    }
    return max_doc_id;
  }

  uint32_t doc_count() const {
    uint32_t count{0};
    if (writing_forward_block_.has_value()) {
      count = writing_forward_block_->doc_count();
    }
    for (const auto &block : persisted_blocks_) {
      if (block.type() == BlockType::SCALAR) {
        count += block.doc_count();
      }
    }
    return count;
  }

  std::vector<BlockMeta> &persisted_blocks() {
    return persisted_blocks_;
  }

  const std::vector<BlockMeta> &persisted_blocks() const {
    return persisted_blocks_;
  }

  std::optional<BlockMeta> &writing_forward_block() {
    return writing_forward_block_;
  }

  const std::optional<BlockMeta> &writing_forward_block() const {
    return writing_forward_block_;
  }

  bool has_writing_forward_block() const {
    return writing_forward_block_.has_value();
  }

  bool vector_indexed(const std::string &field) const {
    return indexed_vector_fields_.count(field) > 0;
  }

  void add_indexed_vector_field(const std::string &field) {
    indexed_vector_fields_.insert(field);
  }

  std::set<std::string> indexed_vector_fields() const {
    return indexed_vector_fields_;
  }

  void set_indexed_vector_fields(const std::set<std::string> &fields) {
    indexed_vector_fields_ = fields;
  }

 public:
  bool operator==(const SegmentMeta &other) const {
    return id_ == other.id_ && persisted_blocks_ == other.persisted_blocks_ &&
           writing_forward_block_ == other.writing_forward_block_ &&
           indexed_vector_fields_ == other.indexed_vector_fields_;
  }

  bool operator!=(const SegmentMeta &other) const {
    return !(*this == other);
  }

  // Add these methods to SegmentMeta class in meta.h

  std::string to_string() const {
    std::ostringstream oss;
    oss << "SegmentMeta{"
        << "id:" << id_ << ",persisted_blocks:[";

    for (size_t i = 0; i < persisted_blocks_.size(); ++i) {
      if (i > 0) oss << ",";
      oss << persisted_blocks_[i].to_string();
    }

    oss << "],writing_forward_block:";
    if (writing_forward_block_.has_value()) {
      oss << writing_forward_block_->to_string();
    } else {
      oss << "null";
    }

    oss << ",indexed_vector_fields:[";

    size_t i = 0;
    for (const auto &field : indexed_vector_fields_) {
      if (i > 0) oss << ",";
      oss << "'" << field << "'";
      ++i;
    }

    oss << "]}";
    return oss.str();
  }

  std::string to_string_formatted(int indent_level = 0) const {
    std::ostringstream oss;
    oss << indent(indent_level) << "SegmentMeta{\n"
        << indent(indent_level + 1) << "id: " << id_ << ",\n"
        << indent(indent_level + 1) << "persisted_blocks: [\n";

    for (size_t i = 0; i < persisted_blocks_.size(); ++i) {
      oss << persisted_blocks_[i].to_string_formatted(indent_level + 2);
      if (i < persisted_blocks_.size() - 1) {
        oss << ",";
      }
      oss << "\n";
    }

    oss << "\n"
        << indent(indent_level + 1) << "],\n"
        << indent(indent_level + 1) << "writing_forward_block: ";

    if (writing_forward_block_.has_value()) {
      oss << "\n"
          << writing_forward_block_->to_string_formatted(indent_level + 2)
          << "\n";
    } else {
      oss << "null\n";
    }

    oss << indent(indent_level + 1) << "indexed_vector_fields: [";

    size_t i = 0;
    for (const auto &field : indexed_vector_fields_) {
      if (i > 0) oss << ",";
      oss << "'" << field << "'";
      ++i;
    }

    oss << "]\n" << indent(indent_level) << "}";
    return oss.str();
  }

 private:
  SegmentID id_{0};
  std::vector<BlockMeta> persisted_blocks_;
  std::optional<BlockMeta> writing_forward_block_;
  std::set<std::string> indexed_vector_fields_;
};

}  // namespace zvec