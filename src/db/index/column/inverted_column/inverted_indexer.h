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


#include <rocksdb/utilities/checkpoint.h>
#include "inverted_column_indexer.h"


namespace zvec {


class InvertedIndexer {
 public:
  using Ptr = std::shared_ptr<InvertedIndexer>;


  explicit InvertedIndexer(const std::string &collection_name,
                           const std::string &working_dir,
                           const std::vector<FieldSchema> &fields)
      : collection_name_(collection_name),
        working_dir_(working_dir),
        fields_(fields) {};


  virtual ~InvertedIndexer() {
    rocksdb_context_.close();
    LOG_INFO("Closed %s", ID().c_str());
  }


  static Ptr CreateAndOpen(const std::string &collection_name,
                           const std::string &working_dir,
                           const bool create_dir_if_missing,
                           const std::vector<FieldSchema> &fields,
                           bool read_only) {
    Ptr indexer =
        std::make_shared<InvertedIndexer>(collection_name, working_dir, fields);
    if (indexer->open(create_dir_if_missing, read_only).ok()) {
      return indexer;
    } else {
      return nullptr;
    }
  }


  InvertedColumnIndexer::Ptr operator[](const std::string &field_name) {
    auto it = indexers_.find(field_name);
    if (it != indexers_.end()) {
      return it->second;
    }
    return nullptr;
  }


  Status flush();

  Status create_snapshot(const std::string &snapshot_dir);

  Status seal();

  Status create_column_indexer(const FieldSchema &field);

  Status remove_column_indexer(const std::string &field_name);

  inline std::string collection() const {
    return collection_name_;
  }

  inline std::string working_dir() const {
    return working_dir_;
  }

  inline const std::string ID() const {
    return "InvertedIndexer[collection:" + collection_name_ + "|path:'" +
           working_dir_ + "']";
  }


 private:
  using FILE = ailego::File;

  Status open(bool create_dir_if_missing, bool read_only);

  inline bool allow_range_optimization(const FieldSchema &field) const {
    bool not_allowed =
        field.is_array_type() || field.data_type() == DataType::BOOL;
    return !not_allowed;
  }

  inline bool allow_extended_wildcard(const FieldSchema &field) const {
    return field.data_type() == DataType::STRING;
  }


 private:
  const std::string collection_name_;
  const std::string working_dir_;
  std::vector<FieldSchema> fields_;

  std::unordered_map<std::string, InvertedColumnIndexer::Ptr> indexers_;
  RocksdbContext rocksdb_context_{};
};


}  // namespace zvec