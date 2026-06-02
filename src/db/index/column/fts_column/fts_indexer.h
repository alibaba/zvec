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
#include <unordered_map>
#include <vector>
#include <zvec/db/schema.h>
#include <zvec/db/status.h>
#include "db/common/rocksdb_context.h"
#include "fts_column_indexer.h"

namespace zvec {

// Manages the per-segment FTS RocksDB instance and all FtsColumnIndexers.
// Analogous to InvertedIndexer for scalar invert indexes.
class FtsIndexer {
 public:
  using Ptr = std::shared_ptr<FtsIndexer>;

  FtsIndexer(const std::string &working_dir) : working_dir_(working_dir) {}

  ~FtsIndexer();

  // Create a new fts.rocksdb or open an existing one. Returns nullptr on
  // failure.
  static Ptr CreateAndOpen(const std::string &working_dir,
                           const FieldSchemaPtrList &fts_fields, bool create,
                           bool read_only = false);

  // Get FtsColumnIndexer by field name.
  fts::FtsColumnIndexerPtr get(const std::string &field_name) const;

  // Flush all indexers and the underlying RocksDB.
  Status flush();

  // Close all indexers and the underlying RocksDB.
  Status close();

  // Create a RocksDB checkpoint (hard-link snapshot) to snapshot_path.
  Status create_snapshot(const std::string &snapshot_path);

  // Add a new FTS field's CFs and open its FtsColumnIndexer.
  Status create_field_indexer(const FieldSchema &field);

  // Remove a field's CFs and stat keys from the RocksDB.
  Status remove_field_indexer(const std::string &field_name);

  // Insert a document's text into a specific field's indexer.
  Status insert(const std::string &field_name, uint32_t seg_doc_id,
                const std::string &text);

  // Seal a single field: flush + convert_postings_to_bitpacked + drop side CFs.
  Status seal(const std::string &field_name);

  // Seal all fields (used by dump path).
  Status seal_all();

  const std::string &working_dir() const {
    return working_dir_;
  }

  bool empty() const {
    return indexers_.empty();
  }

  bool has_field(const std::string &field_name) const {
    return indexers_.find(field_name) != indexers_.end();
  }

 private:
  Status open(const FieldSchemaPtrList &fts_fields, bool create,
              bool read_only);

  std::string working_dir_;
  std::shared_ptr<RocksdbContext> fts_ctx_;
  std::unordered_map<std::string, fts::FtsColumnIndexerPtr> indexers_;
};

}  // namespace zvec
