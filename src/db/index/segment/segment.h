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
#include <arrow/record_batch.h>
#include <zvec/ailego/pattern/expected.hpp>
#include <zvec/db/doc.h>
#include <zvec/db/index_params.h>
#include <zvec/db/options.h>
#include <zvec/db/schema.h>
#include <zvec/db/status.h>
#include "db/index/column/fts_column/fts_column_indexer.h"
#include "db/index/column/fts_column/fts_indexer.h"
#include "db/index/column/inverted_column/inverted_column_indexer.h"
#include "db/index/column/inverted_column/inverted_indexer.h"
#include "db/index/column/vector_column/combined_vector_column_indexer.h"
#include "db/index/column/vector_column/vector_column_indexer.h"
#include "db/index/common/delete_store.h"
#include "db/index/common/id_map.h"
#include "db/index/common/meta.h"
#include "db/index/common/version_manager.h"
#include "db/index/segment/forward_read_view.h"
#include "db/index/storage/base_forward_store.h"

namespace zvec {

class CombinedRecordBatchReader;

class Segment {
 public:
  using Ptr = std::shared_ptr<Segment>;

  // A query-local access decision, not itself an immutable data view. A null
  // view explicitly selects locked Segment::fetch; it never denotes a failed
  // capture. Schema, row boundary and eligibility are captured together.
  struct ForwardReadState {
    std::shared_ptr<arrow::Schema> schema;
    uint64_t num_rows{0};
    ForwardReadView::Ptr view;
    // Sealed IPC only, when requested: the deletion check for this segment,
    // mapping local rows to global doc IDs through the immutable, owned
    // GLOBAL_DOC_ID column instead of a locked segment lookup. Same results
    // as get_filter(); the delete bitmap keeps its own synchronization, so
    // deletes made during the query stay visible. Null selects get_filter().
    IndexFilter::Ptr deletion_filter;
  };

  // Captures how a query reads `columns` per candidate. Implementations that
  // cannot prove buffer ownership return a null view (locked fetch). The
  // segment implementation uses owning views only for sealed IPC segments;
  // writing segments (even immediately after flush) and Parquet keep
  // per-candidate locked fetch. Open and seal prepare immutable columns; DDL
  // and reload replace them under the segment lock, on success or failure.
  // If preparation fails, the segment falls back to locked fetch. Query
  // projections share the published columns and keep their buffers alive
  // independently.
  // `with_deletion` also requests deletion_filter.
  // Callers keep this decision fixed and enforce num_rows only for an owning
  // view. Locked fetch reads live rows, including appends after preparation,
  // and must be protected from close/DDL; an owning view needs no such guard.
  virtual arrow::Result<ForwardReadState> prepare_forward_read(
      const std::vector<std::string> &columns, bool with_deletion) = 0;

  static Result<Ptr> CreateAndOpen(const std::string &path,
                                   const CollectionSchema &schema,
                                   SegmentID segment_id, uint64_t min_doc_id,
                                   const IDMap::Ptr &id_map,
                                   const DeleteStore::Ptr &delete_store,
                                   const VersionManager::Ptr &version_manager,
                                   const SegmentOptions &options);

  static Result<Ptr> Open(const std::string &path,
                          const CollectionSchema &schema,
                          const SegmentMeta &segment_meta,
                          const IDMap::Ptr &id_map,
                          const DeleteStore::Ptr &delete_store,
                          const VersionManager::Ptr &version_manager,
                          const SegmentOptions &options);

  virtual SegmentID id() const = 0;

  virtual SegmentMeta::Ptr meta() const = 0;

  // Doc-id bounds for routing, read under the shared segment lock. Concurrent
  // callers must use this instead of min/max_doc_id() off meta(), which
  // Insert/flush() mutate.
  virtual void doc_id_range(uint64_t *min_id, uint64_t *max_id) const = 0;

  // Like doc_id_range(): total doc count read under the segment lock.
  virtual uint64_t doc_count_snapshot() const = 0;

  // Count documents visible to an optional global-doc-ID filter.
  virtual uint64_t doc_count(const IndexFilter::Ptr filter = nullptr) = 0;

  virtual bool has_record() = 0;

  // ---- Schema and index mutation -----------------------------------------
  virtual Status add_column(FieldSchema::Ptr column_schema,
                            const std::string &expression,
                            const AddColumnOptions &options) = 0;

  virtual Status alter_column(const std::string &column_name,
                              const FieldSchema::Ptr &new_column_schema,
                              const AlterColumnOptions &options) = 0;

  virtual Status drop_column(const std::string &column_name) = 0;

  virtual Status create_all_vector_indexes(
      int concurrency, SegmentMeta::Ptr *new_segment_meta,
      std::unordered_map<std::string, VectorColumnIndexer::Ptr>
          *vector_indexers,
      std::unordered_map<std::string, VectorColumnIndexer::Ptr>
          *quant_vector_indexers) = 0;

  virtual Status create_vector_index(
      const std::string &column, const IndexParams::Ptr &index_params,
      int concurrency, SegmentMeta::Ptr *new_segment_meta,
      std::unordered_map<std::string, VectorColumnIndexer::Ptr>
          *vector_indexers,
      std::unordered_map<std::string, VectorColumnIndexer::Ptr>
          *quant_vector_indexers) = 0;

  virtual Status drop_vector_index(
      const std::string &column, SegmentMeta::Ptr *new_segment_meta,
      std::unordered_map<std::string, VectorColumnIndexer::Ptr>
          *vector_indexers) = 0;

  virtual Status reload_vector_index(
      const CollectionSchema &schema, const SegmentMeta::Ptr &segment_meta,
      const std::unordered_map<std::string, VectorColumnIndexer::Ptr>
          &vector_indexers,
      const std::unordered_map<std::string, VectorColumnIndexer::Ptr>
          &quant_vector_indexers = {}) = 0;

  virtual bool vector_index_ready(
      const std::string &column,
      const IndexParams::Ptr &index_params) const = 0;

  virtual bool all_vector_index_ready() const = 0;

  virtual Status create_scalar_index(
      const std::vector<std::string> &columns,
      const IndexParams::Ptr &index_params, SegmentMeta::Ptr *new_segment_meta,
      InvertedIndexer::Ptr *new_scalar_indexer) = 0;

  virtual Status drop_scalar_index(
      const std::vector<std::string> &columns,
      SegmentMeta::Ptr *new_segment_meta,
      InvertedIndexer::Ptr *new_scalar_indexer) = 0;

  virtual Status reload_scalar_index(
      const CollectionSchema &schema, const SegmentMeta::Ptr &segment_meta,
      const InvertedIndexer::Ptr &scalar_indexer) = 0;

  virtual Status create_fts_index(const std::string &column,
                                  const IndexParams::Ptr &index_params,
                                  SegmentMeta::Ptr *new_segment_meta,
                                  FtsIndexer::Ptr *output_fts_indexer) = 0;

  virtual Status drop_fts_index(const std::string &column,
                                SegmentMeta::Ptr *new_segment_meta,
                                FtsIndexer::Ptr *output_fts_indexer) = 0;

  virtual Status reload_fts_index(const CollectionSchema &schema,
                                  const SegmentMeta::Ptr &segment_meta,
                                  const FtsIndexer::Ptr &new_fts_indexer) = 0;

  // ---- Data operations ----------------------------------------------------
  virtual Status insert(Doc &doc) = 0;

  virtual Status upsert(Doc &doc) = 0;

  virtual Status update(Doc &doc) = 0;

  // NOLINTNEXTLINE(readability-identifier-naming): 'delete' is a keyword
  virtual Status Delete(const std::string &pk) = 0;

  // NOLINTNEXTLINE(readability-identifier-naming): 'delete' is a keyword
  virtual Status Delete(uint64_t g_doc_id) = 0;

  virtual Doc::Ptr fetch(uint64_t g_doc_id,
                         const std::optional<std::vector<std::string>>
                             &output_fields = std::nullopt,
                         bool include_vector = true) = 0;

  virtual TablePtr fetch(const std::vector<std::string> &columns,
                         const std::vector<int> &segment_doc_ids) const = 0;

  virtual ExecBatchPtr fetch(const std::vector<std::string> &columns,
                             int segment_doc_id) const = 0;

  // Valid only while the owning collection is open read-only. Computed by
  // Open after recovery, before queries can run; not maintained for writes.
  virtual bool has_identity_doc_ids() const = 0;

  // Map segment row IDs to insertion ordinals in place; preserve -1 padding.
  virtual Status get_global_doc_ids(std::vector<int64_t> &doc_ids) const = 0;

  // Keep Segment alive while consuming the returned reader.
  virtual RecordBatchReaderPtr scan(
      const std::vector<std::string> &columns) const = 0;

  // ---- Index accessors ----------------------------------------------------
  // Keep Segment alive while using returned indexers.
  virtual CombinedVectorColumnIndexer::Ptr get_combined_vector_indexer(
      const std::string &field_name) const = 0;

  virtual CombinedVectorColumnIndexer::Ptr get_quant_combined_vector_indexer(
      const std::string &field_name) const = 0;

  virtual std::vector<VectorColumnIndexer::Ptr> get_vector_indexer(
      const std::string &field_name) const = 0;

  virtual std::vector<VectorColumnIndexer::Ptr> get_quant_vector_indexer(
      const std::string &field_name) const = 0;

  virtual InvertedColumnIndexer::Ptr get_scalar_indexer(
      const std::string &field_name) const = 0;

  // caller hold segment shared_ptr for segment handle the indexer's lifetime
  virtual fts::FtsColumnIndexerPtr get_fts_indexer(
      const std::string &field_name) const = 0;

  // ---- Index queries and filters -----------------------------------------
  virtual Result<std::vector<fts::FtsResult>> fts_search(
      const std::string &field_name, const fts::FtsAstNode &ast,
      const fts::FtsQueryParams &params) = 0;

  // Returned filter is evaluated with segment-local row IDs. It translates the
  // local row ID to a global doc ID before consulting the delete store.
  virtual const IndexFilter::Ptr get_filter() = 0;

  // ---- Persistence and lifecycle -----------------------------------------
  virtual Status flush() = 0;

  // Rejects a segment that has already been dumped.
  virtual Status dump() = 0;

  // Like dump(), but succeeds if a previous dump already completed. A
  // rollover uses it to retry after a failure later in the switch.
  virtual Status resume_dump() = 0;

  // Prepares the sealed read state and removes the writing block together
  // under the exclusive segment lock. Idempotent once the block is removed.
  // If the read state cannot be prepared, the segment is still sealed and
  // serves forward reads through locked fetch.
  virtual Status remove_writing_forward_block() = 0;

  virtual Status destroy() = 0;
};

}  // namespace zvec
