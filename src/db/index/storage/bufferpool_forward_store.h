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
#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/filesystem/filesystem.h>
#include <arrow/io/api.h>
#include <arrow/ipc/reader.h>
#include <arrow/util/async_generator.h>
#include <parquet/arrow/reader.h>
#include <zvec/ailego/buffer/buffer_manager.h>
#include <zvec/db/status.h>
#include "base_forward_store.h"

namespace zvec {

/// BufferPoolForwardStore implements a forward store that uses a buffer pool
/// to efficiently manage data access from parquet files.
class BufferPoolForwardStore
    : public BaseForwardStore,
      public std::enable_shared_from_this<BufferPoolForwardStore> {
 public:
  /// Pointer type for BufferPoolForwardStore instances
  using Ptr = std::shared_ptr<BufferPoolForwardStore>;

  /// Constructor that initializes the store with a file URI
  /// \param uri The URI of the file to be accessed
  explicit BufferPoolForwardStore(const std::string &uri);

  virtual ~BufferPoolForwardStore() = default;

  Status Open() override;

  /// Fetch specific columns and row indices from the data source
  /// \param columns The list of column names to fetch
  /// \param indices The list of row indices to fetch
  /// \return A table containing the requested data or nullptr on failure
  TablePtr fetch(const std::vector<std::string> &columns,
                 const std::vector<int> &indices) override;

  /// Fetch specific columns and row indices from the data source
  /// \param columns The list of column names to fetch
  /// \param index The row index to fetch
  /// \return An ExecBatch containing the requested data or nullptr on failure
  ExecBatchPtr fetch(const std::vector<std::string> &columns,
                     int index) override;

  /// Scan specified columns from the data source
  /// \param columns The list of column names to scan
  /// \return A RecordBatchReader for streaming the data or nullptr on failure
  RecordBatchReaderPtr scan(const std::vector<std::string> &columns) override;

  /// Get the physical schema of the file
  /// \return A shared pointer to the arrow schema representing the physical
  /// structure of the data
  const std::shared_ptr<arrow::Schema> physic_schema() const override {
    return physic_schema_;
  }

  TablePtr get_table() override {
    return nullptr;
  }

 private:
  /// Validate that the requested columns exist in the schema
  /// \param columns The list of column names to validate
  /// \return true if all columns are valid, false otherwise
  bool validate(const std::vector<std::string> &columns) const;

  /// Open a parquet file and initialize metadata
  /// \param file The RandomAccessFile to read from
  /// \return arrow::Status indicating success or failure
  arrow::Status OpenParquet(
      const std::shared_ptr<arrow::io::RandomAccessFile> &file);

  /// Find which row group contains a given row
  /// \param row The row index to locate
  /// \return The row group ID containing the row
  int FindRowGroupForRow(int64_t row);

  /// Get the row offset for a given row group
  /// \param rg_id The row group ID
  /// \return The row offset of the row group, or -1 on error
  int64_t GetRowGroupOffset(int rg_id);

 private:
  /// Physical schema of the file
  std::shared_ptr<arrow::Schema> physic_schema_;

  /// Total number of rows in the file
  int64_t num_rows_ = 0;

  /// Path to the file
  std::string file_path_;

  // Parquet-specific members
  /// The RandomAccessFile for reading data
  std::shared_ptr<arrow::io::RandomAccessFile> file_;

  /// The parquet file reader
  std::unique_ptr<parquet::arrow::FileReader> parquet_reader_;

  /// Number of row groups in the file
  int64_t num_row_groups_ = 0;

  /// Offsets of each row group
  std::vector<int64_t> row_group_offsets_;

  /// Number of rows in each row group
  std::vector<int64_t> row_group_row_nums_;

  /// Buffer manager for caching data
  std::shared_ptr<ailego::BufferManager> buffer_manager_;
};

}  // namespace zvec