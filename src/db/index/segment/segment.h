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
#include "db/index/column/inverted_column/inverted_column_indexer.h"
#include "db/index/column/inverted_column/inverted_indexer.h"
#include "db/index/column/vector_column/combined_vector_column_indexer.h"
#include "db/index/column/vector_column/vector_column_indexer.h"
#include "db/index/common/delete_store.h"
#include "db/index/common/id_map.h"
#include "db/index/common/meta.h"
#include "db/index/common/version_manager.h"
#include "db/index/storage/base_forward_store.h"

namespace zvec {

class CombinedRecordBatchReader;

class Segment {
 public:
  using Ptr = std::shared_ptr<Segment>;

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

  virtual uint64_t doc_count(const IndexFilter::Ptr filter = nullptr) = 0;

  // for collection
  virtual Status add_column(FieldSchema::Ptr column_schema,
                            const std::string &expression,
                            const AddColumnOptions &options) = 0;

  virtual Status alter_column(const std::string &column_name,
                              const FieldSchema::Ptr &new_column_schema,
                              const AlterColumnOptions &options) = 0;

  virtual Status drop_column(const std::string &column_name) = 0;

  virtual Status create_all_vector_index(
      int concurrency, SegmentMeta::Ptr *new_segmnet_meta,
      std::unordered_map<std::string, VectorColumnIndexer::Ptr>
          *vector_indexers,
      std::unordered_map<std::string, VectorColumnIndexer::Ptr>
          *quant_vector_indexers) = 0;

  // defined in segment.h cause it needs to access block_id generator
  virtual Status create_vector_index(
      const std::string &column, const IndexParams::Ptr &index_params,
      int concurrency, SegmentMeta::Ptr *new_segmnet_meta,
      std::unordered_map<std::string, VectorColumnIndexer::Ptr>
          *vector_indexers,
      std::unordered_map<std::string, VectorColumnIndexer::Ptr>
          *quant_vector_indexers) = 0;

  virtual Status drop_vector_index(
      const std::string &column, SegmentMeta::Ptr *new_segmnet_meta,
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

  // defined in segment.h cause it needs to access block_id generator
  virtual Status create_scalar_index(
      const std::vector<std::string> &columns,
      const IndexParams::Ptr &index_params, SegmentMeta::Ptr *new_segment_meta,
      InvertedIndexer::Ptr *new_scalar_indexer) = 0;

  // defined in segment.h cause it needs to access block_id generator
  virtual Status drop_scalar_index(
      const std::vector<std::string> &columns,
      SegmentMeta::Ptr *new_segment_meta,
      InvertedIndexer::Ptr *new_scalar_indexer) = 0;

  virtual Status reload_scalar_index(
      const CollectionSchema &schema, const SegmentMeta::Ptr &segment_meta,
      const InvertedIndexer::Ptr &scalar_indexer) = 0;

  virtual Status Insert(Doc &doc) = 0;

  virtual Status Upsert(Doc &doc) = 0;

  virtual Status Update(Doc &doc) = 0;

  virtual Status Delete(const std::string &pk) = 0;

  virtual Status Delete(uint64_t g_doc_id) = 0;

  virtual Doc::Ptr Fetch(uint64_t g_doc_id) = 0;

  // for sqlengine
  virtual TablePtr fetch(const std::vector<std::string> &columns,
                         const std::vector<int> &indices) const = 0;

  virtual ExecBatchPtr fetch(const std::vector<std::string> &columns,
                             int index) const = 0;

  // caller should hold segment shared_ptr for segment handle the indexer's
  // lifetime
  virtual RecordBatchReaderPtr scan(
      const std::vector<std::string> &columns) const = 0;

  // caller hold segment shared_ptr for segment handle the indexer's lifetime
  virtual CombinedVectorColumnIndexer::Ptr get_combined_vector_indexer(
      const std::string &field_name) const = 0;

  // caller hold segment shared_ptr for segment handle the indexer's lifetime
  virtual CombinedVectorColumnIndexer::Ptr get_quant_combined_vector_indexer(
      const std::string &field_name) const = 0;

  // caller hold segment shared_ptr for segment handle the indexer's lifetime
  virtual std::vector<VectorColumnIndexer::Ptr> get_vector_indexer(
      const std::string &field_name) const = 0;

  virtual std::vector<VectorColumnIndexer::Ptr> get_quant_vector_indexer(
      const std::string &field_name) const = 0;

  // caller hold segment shared_ptr for segment handle the indexer's lifetime
  virtual InvertedColumnIndexer::Ptr get_scalar_indexer(
      const std::string &field_name) const = 0;

  virtual const IndexFilter::Ptr get_filter() = 0;

  // for others
  virtual Status flush() = 0;
  virtual Status dump() = 0;

  // only mark need_destroyed
  virtual Status destroy() = 0;
};

}  // namespace zvec