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
#include <mutex>
#include <string>
#include <vector>
#include <arrow/compute/api.h>
#include <arrow/table_builder.h>
#include <arrow/util/async_generator.h>
#include <zvec/db/doc.h>
#include <zvec/db/status.h>
#include "base_forward_store.h"
#include "chunked_file_writer.h"
#include "store_helper.h"

namespace zvec {

/// MemForwardStore implements a forward store that keeps data in memory
/// and can flush data to disk when needed.
class MemForwardStore : public BaseForwardStore {
 public:
  /// Pointer type for MemForwardStore instances
  using Ptr = std::shared_ptr<MemForwardStore>;

  /// Constructor that initializes the store with collection schema and settings
  /// \param collection_schema The schema for the collection
  /// \param path The path where data will be stored
  /// \param format The file format for persistence
  /// \param max_cache_rows Maximum number of rows to keep in cache
  /// \param max_rows Maximum number of rows allowed in the store
  MemForwardStore(const std::shared_ptr<CollectionSchema> &collection_schema,
                  const std::string &path, const FileFormat format,
                  const uint32_t max_buffer_size = 100 * 1024 * 1024);

  virtual ~MemForwardStore() {
    close();
  }

  /// Check if the store is full
  /// \return true if the store has reached its maximum capacity
  bool is_full() {
    return total_bytes() >= max_buffer_size_;
  }

  /// Open the store
  /// \return 0 on success, non-zero on failure
  Status Open() override;

  /// Insert a document into the store
  /// \param doc The document to insert
  /// \return 0 on success, non-zero on failure
  Status insert(const Doc &doc);

  /// Flush cached data to disk
  /// \return 0 on success, non-zero on failure
  Status flush();

  /// Close the store and flush any remaining data
  /// \return 0 on success, non-zero on failure
  Status close();

 public:
  /// Get the path of the store
  /// \return The path where data is stored
  const std::string path() const {
    return path_;
  }

  /// Get the total bytes of the store
  uint32_t total_bytes() const {
    return total_cache_bytes_ + total_rb_bytes_;
  }

  /// Get the total number of rows in the store
  uint32_t num_rows() const {
    return num_rows_;
  }

 public:
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

  TablePtr get_table() override;

 private:
  /// Create a RecordBatchBuilder for this store
  /// \return A new RecordBatchBuilder
  RecordBatchBuilderPtr createBuilder();

  /// Convert internal data to a RecordBatch
  /// \return A Result containing the RecordBatch or an error status
  arrow::Result<RecordBatchPtr> convertToRecordBatch();

  /// Convert internal data to a Table
  /// \param columns The list of column names to include
  /// \param indices The list of row indices to include
  /// \return A Result containing the Table or an error status
  arrow::Result<TablePtr> convertToTable(
      const std::vector<std::string> &columns, const std::vector<int> &indices);

  /// Convert internal data to a RecordBatchBuilder
  /// \param builder The builder to populate
  /// \return arrow::Status indicating success or failure
  arrow::Status convertToBuilder(RecordBatchBuilderPtr &builder);

  /// Validate that the requested columns exist in the schema
  /// \param columns The list of column names to validate
  /// \return true if all columns are valid, false otherwise
  bool validate(const std::vector<std::string> &columns) const;

 private:
  /// Mutex to protect cache access
  std::mutex cache_mtx_;

  /// Cache of documents waiting to be flushed
  std::vector<Doc> cache_;

  /// Batches of data that have been flushed
  std::vector<RecordBatchPtr> batches_;

  /// Collection schema
  std::shared_ptr<CollectionSchema> schema_;

  /// Physical schema
  std::shared_ptr<arrow::Schema> physic_schema_;

  /// Total RecordBatch bytes in the store
  uint32_t total_rb_bytes_{0};

  /// Total cache doc bytes
  uint32_t total_cache_bytes_{0};

  /// Total number of rows in the store
  uint32_t num_rows_{0};

  /// Path where data is stored
  std::string path_;

  /// File format for persistence
  FileFormat format_;

  /// Number of batches that have been flushed
  uint32_t flushed_batches_{0};

  /// Writer for chunked files
  ChunkedFileWriter::Ptr writer_;


  /// Maximum size of cache, default 1MB
  uint32_t max_cache_size_{1048576};

  /// Maximum size of the buffer, default 100MB
  uint32_t max_buffer_size_{104857600};
};

}  // namespace zvec