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
#include "db/common/durable_file.h"
#include "db/common/rocksdb_context.h"
#include "db/index/common/meta.h"
#include "fts_column_indexer.h"
#include "fts_file_lifetime.h"
#include "fts_merge.h"

namespace zvec {

// Manages the per-segment FTS RocksDB instance and all FtsColumnIndexers.
// Analogous to InvertedIndexer for scalar invert indexes.
class FtsIndexer {
 public:
  using Ptr = std::shared_ptr<FtsIndexer>;

  // Activation follows P4 correctness/recovery and P5 benchmark acceptance.
  // Internal preparation remains available for differential/failure tests.
  static constexpr bool kCustomWritesEnabled = false;

  FtsIndexer(const std::string &working_dir) : working_dir_(working_dir) {}

  ~FtsIndexer();

  // Create a new fts.rocksdb or open an existing one. Returns nullptr on
  // failure.
  static Ptr CreateAndOpen(
      const std::string &working_dir, const FieldSchemaPtrList &fts_fields,
      bool create, bool read_only = false,
      const fts::FtsRetirementQueue::Pointer &retirement = {});

  struct CustomTarget {
    std::string segment_path;
    SegmentID segment_id{0};
    // The complete parent-segment-local row domain is block.doc_count().
    BlockMeta block;
    bool enable_mmap{false};
  };

  // Caller holds the source schema/write exclusion and has completed the
  // full database checkpoint. This prepares a verified immutable generation
  // without mutating source CFs or committing a collection version.
  Result<Ptr> prepare_custom(const CustomTarget &target,
                             const FieldSchemaPtrList &fields,
                             const fts::FtsRetirementQueue::Pointer &retirement,
                             const DurableFile::Hook &hook = {}) const;

  static Result<Ptr> OpenCustom(
      const CustomTarget &target, const FieldSchemaPtrList &fields,
      const fts::FtsRetirementQueue::Pointer &retirement);

  const std::optional<BlockMeta> &custom_block() const {
    return custom_block_;
  }

  // Immutable source export for rewrite/merge. Legacy fields additionally
  // require presence in exact forward scan order; custom fields carry it.
  Result<fts::FtsFieldSource::Ptr> open_source(
      const std::string &field, uint64_t rows,
      const fts::FtsPresenceSource &presence = {}) const;

  // sources[field][segment], with each field using identical forward order.
  // Uses the same verified, durable private-generation preparation as seal.
  static Result<Ptr> PrepareMerged(
      const CustomTarget &target, const FieldSchemaPtrList &fields,
      const std::vector<std::vector<fts::FtsFieldSource::Ptr>> &sources,
      const roaring::Roaring &deletions, fts::FtsMergePolicy policy,
      const fts::FtsRetirementQueue::Pointer &retirement,
      const DurableFile::Hook &hook = {});

  void preserve_files() {
    if (file_owner_) {
      file_owner_->preserve();
    }
  }

  void retire_files() {
    if (file_owner_) {
      file_owner_->retire();
    }
  }

  // Get FtsColumnIndexer by field name.
  fts::FtsColumnIndexerPtr get(const std::string &field_name) const;

  // Flush all indexers and the underlying RocksDB.
  Status flush();

  Status recover_statistics(uint64_t row_count);

  // Close all indexers and the underlying RocksDB.
  Status close();

  // Snapshot this fixed generation into a new directory. Legacy uses a
  // RocksDB checkpoint; custom copies its immutable file under the same name.
  // Caller pins this indexer and records custom_block() with the snapshot's
  // manifest, under the same schema/version exclusion as the source capture.
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
  friend class FtsSealRetryTest;

  Status open(const FieldSchemaPtrList &fts_fields, bool create,
              bool read_only);

  static Result<Ptr> OpenCustomFile(const CustomTarget &target,
                                    const FieldSchemaPtrList &fields,
                                    const std::string &path,
                                    fts::FtsFileOwner::Pointer owner,
                                    bool candidate);

  using FieldBuilder = std::function<Result<fts::format::FieldMetadata>(
      fts::format::FieldMetadata, fts::FtsContainerWriter *, uint32_t)>;
  static Result<Ptr> PrepareGeneration(
      const CustomTarget &target, const FieldSchemaPtrList &fields,
      const fts::FtsRetirementQueue::Pointer &retirement,
      const FieldBuilder &build, const DurableFile::Hook &hook);

  std::string working_dir_;
  std::shared_ptr<RocksdbContext> fts_ctx_;
  std::unordered_map<std::string, fts::FtsColumnIndexerPtr> indexers_;
  std::optional<BlockMeta> custom_block_;
  fts::FtsFileOwner::Pointer file_owner_;
  fts::FtsContainerReader::Pointer container_;
};

}  // namespace zvec
