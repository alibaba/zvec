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

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <arrow/api.h>
#include <arrow/chunked_array.h>
#include <arrow/compute/api.h>
#include <arrow/dataset/api.h>
#include <arrow/filesystem/api.h>
#include <arrow/io/file.h>
#include <arrow/ipc/reader.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/table.h>
#include <arrow/util/async_generator.h>
#include <parquet/arrow/reader.h>
#include <parquet/column_reader.h>
#include <parquet/exception.h>
#include <zvec/db/status.h>
#include "base_forward_store.h"
#include "store_helper.h"

namespace zvec {

/// MmapForwardStore implements a forward store that uses memory mapping
/// to efficiently access data from parquet or IPC files.
class MmapForwardStore : public BaseForwardStore {
 public:
  /// Pointer type for MmapForwardStore instances
  using Ptr = std::shared_ptr<MmapForwardStore>;

  /// Constructor that initializes the store with a file URI
  /// \param uri The URI of the file to be accessed
  MmapForwardStore(const std::string &uri);
  virtual ~MmapForwardStore() {}

  Status Open() override;

  /// Fetch specific columns and row indices from the data source
  /// \param columns The list of column names to fetch
  /// \param indices The list of row indices to fetch
  /// \return A table containing the requested data or nullptr on failure
  TablePtr fetch(const std::vector<std::string> &columns,
                 const std::vector<int> &indices) override;

  /// Fetch specific columns and a single row index from the data source
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
    return table_;
  }

 private:
  /// Validate that the requested columns exist in the schema
  /// \param columns The list of column names to validate
  /// \return true if all columns are valid, false otherwise
  bool validate(const std::vector<std::string> &columns) const;

 private:
  /// Open a parquet file and initialize metadata
  /// \param file The RandomAccessFile to read from
  /// \return arrow::Status indicating success or failure
  arrow::Status OpenParquet(
      const std::shared_ptr<arrow::io::RandomAccessFile> &file);

  /// Open an IPC file and initialize metadata
  /// \param file The RandomAccessFile to read from
  /// \return arrow::Status indicating success or failure
  arrow::Status OpenIPC(
      const std::shared_ptr<arrow::io::RandomAccessFile> &file);

  /// Fetch data from a parquet file
  /// \param columns The list of column names to fetch
  /// \param indices The list of row indices to fetch
  /// \return A table containing the requested data or nullptr on failure
  TablePtr FetchParquet(const std::vector<std::string> &columns,
                        const std::vector<int> &indices);

  /// Fetch specific columns and a single row index from parquet file
  /// \param columns The list of column names to fetch
  /// \param index The row index to fetch
  /// \return An ExecBatch containing the requested data or nullptr on failure
  ExecBatchPtr FetchParquet(const std::vector<std::string> &columns, int index);

  /// Fetch data from an IPC file
  /// \param columns The list of column names to fetch
  /// \param indices The list of row indices to fetch
  /// \return A table containing the requested data or nullptr on failure
  TablePtr FetchIPC(const std::vector<std::string> &columns,
                    const std::vector<int> &indices);

  /// Fetch specific columns and a single row index from IPC file
  /// \param columns The list of column names to fetch
  /// \param index The row index to fetch
  /// \return An ExecBatch containing the requested data or nullptr on failure
  ExecBatchPtr FetchIPC(const std::vector<std::string> &columns, int index);

  /// Scan data from a parquet file
  /// \param columns The list of column names to scan
  /// \return A RecordBatchReader for streaming the data or nullptr on failure
  RecordBatchReaderPtr ScanParquet(const std::vector<std::string> &columns);

  /// Scan data from an IPC file
  /// \param columns The list of column names to scan
  /// \return A RecordBatchReader for streaming the data or nullptr on failure
  RecordBatchReaderPtr ScanIPC(const std::vector<std::string> &columns);

  /// Find which row group contains a given row
  /// \param row The row index to locate
  /// \return The row group ID containing the row
  int FindRowGroupForRow(int64_t row);

  /// Get the row offset for a given row group
  /// \param rg_id The row group ID
  /// \return The row offset of the row group, or -1 on error
  int64_t GetRowGroupOffset(int rg_id);

  /// Find the chunk that contains a target row index using binary search
  /// \param target_index The row index to locate
  /// \param num_chunks The total number of chunks in the array
  /// \param target_chunk_index Output parameter for the index of the chunk
  /// containing the target
  /// \param offset_in_chunk Output parameter for the offset within the found
  /// chunk
  /// \return true if the target chunk was found, false otherwise
  bool FindTargetChunk(int target_index, int num_chunks,
                       int *target_chunk_index, int64_t *offset_in_chunk);

 private:
  /// Format of the file being accessed
  FileFormat format_;

  /// Physical schema of the file
  std::shared_ptr<arrow::Schema> physic_schema_;

  /// Total number of rows in the file
  int64_t num_rows_{0};

  /// Path to the file
  std::string file_path_;

  // Parquet-specific members
  /// The RandomAccessFile for reading data
  std::shared_ptr<arrow::io::RandomAccessFile> file_;

  /// The parquet file reader
  std::unique_ptr<parquet::arrow::FileReader> parquet_reader_;

  /// Number of row groups in the file
  int64_t num_row_groups_{0};

  /// Offsets of each row group
  std::vector<int64_t> row_group_offsets_;

  /// Number of rows in each row group
  std::vector<int64_t> row_group_row_nums_;

  // IPC-specific members
  /// The IPC file reader
  std::shared_ptr<arrow::ipc::RecordBatchFileReader> ipc_file_reader_;

  std::shared_ptr<arrow::Table> table_;

  std::vector<std::pair<int64_t, int64_t>> chunk_index_map_;

  // For performance tuning
  bool is_fixed_batch_size_{true};
  int64_t fixed_batch_size_{-1};
};

}  // namespace zvec