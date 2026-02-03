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


#include <rocksdb/db.h>
#include <zvec/ailego/io/file.h>
#include <zvec/db/status.h>


namespace zvec {


// A very thin wrapper around RocksDB
struct RocksdbContext {
 public:
  std::unique_ptr<rocksdb::DB> db_{nullptr};
  std::string db_path_;
  bool read_only_;
  std::vector<rocksdb::ColumnFamilyHandle *> cf_handles_;
  rocksdb::Options create_opts_;
  rocksdb::WriteOptions write_opts_;
  rocksdb::ReadOptions read_opts_;
  rocksdb::FlushOptions flush_opts_;
  rocksdb::CompactRangeOptions compact_range_opts_;
  std::mutex mutex_;


 public:
  // Create a Rocksdb instance
  Status create(const std::string &db_path,
                std::shared_ptr<rocksdb::MergeOperator> merge_op = nullptr);


  // Create a Rocksdb instance
  Status create(const std::string &db_path,
                const std::vector<std::string> &column_names,
                std::shared_ptr<rocksdb::MergeOperator> merge_op = nullptr);


  // Open an existing Rocksdb instance
  Status open(const std::string &db_path, bool read_only = false,
              std::shared_ptr<rocksdb::MergeOperator> merge_op = nullptr);


  // Open an existing Rocksdb instance
  Status open(const std::string &db_path,
              const std::vector<std::string> &column_names,
              bool read_only = false,
              std::shared_ptr<rocksdb::MergeOperator> merge_op = nullptr);


  // Close and flush data if needed
  Status close();


  // Flush data
  Status flush();


  // Create a checkpoint
  Status create_checkpoint(const std::string &checkpoint_dir);


  // Get a column family
  rocksdb::ColumnFamilyHandle *get_cf(const std::string &cf_name);


  // Create a column family
  Status create_cf(const std::string &cf_name);


  // Drop a column family
  Status drop_cf(const std::string &cf_name);


  // Reset a column family
  Status reset_cf(const std::string &cf_name);


  // Compact db
  Status compact();


  // Get the size of the SST files
  size_t sst_file_size();


  // Get the estimated number of keys in the database
  size_t count();


 private:
  using FILE = ailego::File;


  Status validate_and_set_db_path(const std::string &db_path,
                                  bool should_exist);

  void prepare_options(std::shared_ptr<rocksdb::MergeOperator> merge_op);

  Status flush_unlocked();

  void delete_cf_handles();
};


}  // namespace zvec