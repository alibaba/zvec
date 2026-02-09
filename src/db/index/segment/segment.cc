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

#include "segment.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <ailego/parallel/multi_thread_list.h>
#include <ailego/pattern/defer.h>
#include <arrow/dataset/dataset.h>
#include <arrow/dataset/scanner.h>
#include <arrow/ipc/reader.h>
#include <arrow/table.h>
#include <arrow/util/iterator.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/ailego/parallel/thread_pool.h>
#include <zvec/ailego/parallel/thread_queue.h>
#include <zvec/db/config.h>
#include <zvec/db/doc.h>
#include <zvec/db/index_params.h>
#include <zvec/db/schema.h>
#include <zvec/db/status.h>
#include <zvec/db/type.h>
#include "db/common/constants.h"
#include "db/common/file_helper.h"
#include "db/common/global_resource.h"
#include "db/common/typedef.h"
#include "db/index/column/inverted_column/inverted_indexer.h"
#include "db/index/column/vector_column/vector_column_indexer.h"
#include "db/index/column/vector_column/vector_column_params.h"
#include "db/index/common/index_filter.h"
#include "db/index/common/meta.h"
#include "db/index/segment/segment_helper.h"
#include "db/index/storage/base_forward_store.h"
#include "db/index/storage/bufferpool_forward_store.h"
#include "db/index/storage/memory_forward_store.h"
#include "db/index/storage/mmap_forward_store.h"
#include "db/index/storage/store_helper.h"
#include "db/index/storage/wal/wal_file.h"
#include "column_merging_reader.h"
#include "sql_expr_parser.h"

namespace zvec {

void global_init() {
  static std::once_flag once;
  // run once
  std::call_once(once, []() {
    auto status = arrow::compute::Initialize();
    if (!status.ok()) {
      LOG_ERROR("arrow compute init failed: [%s]", status.ToString().c_str());
      abort();
    }
  });
}

class SegmentImpl : public Segment,
                    public std::enable_shared_from_this<SegmentImpl> {
 public:
  using Ptr = std::shared_ptr<SegmentImpl>;

  class SegmentIndexFilter : public IndexFilter {
   public:
    SegmentIndexFilter(const DeleteStore::Ptr &delete_store,
                       SegmentImpl::Ptr impl)
        : delete_store_(delete_store), impl_(impl) {}

    bool is_filtered(uint64_t id) const override;

   private:
    DeleteStore::Ptr delete_store_;
    std::weak_ptr<SegmentImpl> impl_;
  };

  SegmentImpl(const std::string &path, const CollectionSchema &schema,
              const SegmentMeta &segment_meta, const IDMap::Ptr &id_map,
              const DeleteStore::Ptr &delete_store,
              const VersionManager::Ptr &version_manager)
      : path_(path),
        collection_schema_(std::make_shared<CollectionSchema>(schema)),
        segment_meta_(std::make_shared<SegmentMeta>(segment_meta)),
        version_manager_(version_manager),
        id_map_(id_map),
        delete_store_(delete_store) {
    seg_path_ = FileHelper::MakeSegmentPath(path_, segment_meta.id());
  }

  virtual ~SegmentImpl() {
    close();
    if (need_destroyed_) {
      cleanup();
    }
  }

  SegmentID id() const override;

  SegmentMeta::Ptr meta() const override;

  uint64_t doc_count(const IndexFilter::Ptr filter = nullptr) override;

  Status Insert(Doc &doc) override;

  Status Update(Doc &doc) override;

  Status Upsert(Doc &doc) override;

  Status Delete(const std::string &pk) override;

  Status Delete(uint64_t g_doc_id) override;

  Doc::Ptr Fetch(uint64_t g_doc_id) override;

  CombinedVectorColumnIndexer::Ptr get_combined_vector_indexer(
      const std::string &field_name) const override;

  CombinedVectorColumnIndexer::Ptr get_quant_combined_vector_indexer(
      const std::string &field_name) const override;

  VectorColumnIndexer::Ptr get_memory_vector_indexer(
      const std::string &field_name);

  VectorColumnIndexer::Ptr get_memory_quant_vector_indexer(
      const std::string &field_name);

  std::vector<VectorColumnIndexer::Ptr> get_vector_indexer(
      const std::string &field_name) const override;

  virtual std::vector<VectorColumnIndexer::Ptr> get_quant_vector_indexer(
      const std::string &field_name) const override;

  InvertedColumnIndexer::Ptr get_scalar_indexer(
      const std::string &field_name) const override;

  const IndexFilter::Ptr get_filter() override;

  Status create_all_vector_index(
      int concurrency, SegmentMeta::Ptr *new_segment_meta,
      std::unordered_map<std::string, VectorColumnIndexer::Ptr>
          *vector_indexers,
      std::unordered_map<std::string, VectorColumnIndexer::Ptr>
          *quant_vector_indexers) override;

  Status create_vector_index(
      const std::string &column, const IndexParams::Ptr &index_params,
      int concurrency, SegmentMeta::Ptr *new_segment_meta,
      std::unordered_map<std::string, VectorColumnIndexer::Ptr>
          *vector_indexers,
      std::unordered_map<std::string, VectorColumnIndexer::Ptr>
          *quant_vector_indexers) override;

  Status drop_vector_index(
      const std::string &column, SegmentMeta::Ptr *new_segment_meta,
      std::unordered_map<std::string, VectorColumnIndexer::Ptr>
          *vector_indexers) override;

  Status reload_vector_index(
      const CollectionSchema &schema, const SegmentMeta::Ptr &new_segment_meta,
      const std::unordered_map<std::string, VectorColumnIndexer::Ptr>
          &vector_indexers,
      const std::unordered_map<std::string, VectorColumnIndexer::Ptr>
          &quant_vector_indexers) override;

  bool vector_index_ready(const std::string &column,
                          const IndexParams::Ptr &index_params) const override;

  bool all_vector_index_ready() const override;

  Status create_scalar_index(const std::vector<std::string> &columns,
                             const IndexParams::Ptr &index_params,
                             SegmentMeta::Ptr *new_segment_meta,
                             InvertedIndexer::Ptr *new_scalar_indexer) override;

  Status drop_scalar_index(const std::vector<std::string> &columns,
                           SegmentMeta::Ptr *new_segment_meta,
                           InvertedIndexer::Ptr *new_scalar_indexer) override;

  Status reload_scalar_index(
      const CollectionSchema &schema, const SegmentMeta::Ptr &segment_meta,
      const InvertedIndexer::Ptr &scalar_indexer) override;

  Status dump() override;

  Status flush() override;

  Status destroy() override;

  TablePtr fetch(const std::vector<std::string> &columns,
                 const std::vector<int> &indices) const override;

  ExecBatchPtr fetch(const std::vector<std::string> &columns,
                     int indice) const override;

  RecordBatchReaderPtr scan(
      const std::vector<std::string> &columns) const override;

  Status add_column(FieldSchema::Ptr column_schema,
                    const std::string &expression,
                    const AddColumnOptions &options) override;

  Status alter_column(const std::string &column_name,
                      const FieldSchema::Ptr &new_column_schema,
                      const AlterColumnOptions &options) override;

  Status drop_column(const std::string &column_name) override;

 public:
  Status Open(const SegmentOptions &options);
  Status Create(const SegmentOptions &options, uint64_t min_doc_id);

 private:
  Status close();
  Status cleanup();
  bool ready_for_dump_block();

  // Helper functions for Open()
  Status load_persist_scalar_blocks();
  Status load_scalar_index_blocks(bool create = false);
  Status load_vector_index_blocks();
  Status init_memory_components();
  Status finish_memory_components();

  void fresh_persist_block_offset();
  void calculate_block_offsets();
  int find_persist_block_id(BlockType type, int segment_doc_id,
                            const std::string &col_name = "",
                            int *out_offset_idx = nullptr) const;
  const std::vector<int> &get_persist_block_offsets(
      BlockType type, const std::string &col_name = "") const;
  const std::vector<BlockMeta> &get_persist_block_metas(
      BlockType type, const std::string &col_name = "") const;

  VectorColumnIndexer::Ptr create_vector_indexer(const std::string &field_name,
                                                 const FieldSchema &field,
                                                 BlockID block_id,
                                                 bool is_quantized = false);

  Result<VectorColumnIndexer::Ptr> merge_vector_indexer(
      const std::string &index_file_path, const std::string &column,
      const FieldSchema &field, int concurrency);

  // Helper functions for Insert/Update/Upsert/Delete
  template <typename ValueType>
  Status InsertScalar(InvertedColumnIndexer::Ptr &indexer, const Doc &doc,
                      const FieldSchema::Ptr &field);
  template <typename ValueType>
  Status InsertVector(VectorColumnIndexer::Ptr &indexer, const Doc &doc,
                      const FieldSchema::Ptr &field);
  Status ConvertVectorDataBufferToDocField(
      const FieldSchema::Ptr &field,
      const vector_column_params::VectorDataBuffer &buf, Doc *doc);

  Status insert_scalar_indexer(Doc &doc);
  Status insert_vector_indexer(Doc &doc);
  Status internal_insert(Doc &doc);
  Status internal_update(Doc &doc);
  Status internal_upsert(Doc &doc);
  Status internal_delete(const Doc &doc);

  Status recover();
  Status open_wal_file();
  Status append_wal(const Doc &doc);
  Status update_version(uint32_t delete_snapshot_path_suffix);

  Result<uint64_t> get_global_doc_id(uint32_t local_id) const;

  BlockID allocate_block_id();

  bool validate(const std::vector<std::string> &columns) const;

  Status reopen_invert_indexer(bool read_only = false);

  Status insert_array_to_invert_indexer(
      const FieldSchema::Ptr &schema,
      const std::shared_ptr<arrow::ChunkedArray> &data,
      InvertedColumnIndexer::Ptr *column_indexer);

  TablePtr fetch_normal(const std::vector<std::string> &columns,
                        const std::shared_ptr<arrow::Schema> &result_schema,
                        const std::vector<int> &indices) const;

  // For performance tuning
  TablePtr fetch_perf(const std::vector<std::string> &columns,
                      const std::shared_ptr<arrow::Schema> &result_schema,
                      const std::vector<int> &indices) const;

  void fresh_persist_chunked_array();

 private:
  // scalar forward
  MemForwardStore::Ptr memory_store_;
  std::vector<BaseForwardStore::Ptr> persist_stores_;

  // scalar index
  InvertedIndexer::Ptr invert_indexers_;

  // vector index
  std::unordered_map<std::string, VectorColumnIndexer::Ptr>
      memory_vector_indexers_;

  std::unordered_map<std::string, BlockID> memory_vector_block_ids_;

  std::unordered_map<std::string, VectorColumnIndexer::Ptr>
      quant_memory_vector_indexers_;

  std::unordered_map<std::string, BlockID> quant_memory_vector_block_ids_;

  std::unordered_map<std::string, std::vector<VectorColumnIndexer::Ptr>>
      vector_indexers_;

  std::unordered_map<std::string, std::vector<VectorColumnIndexer::Ptr>>
      quant_vector_indexers_;

  // index filter
  IndexFilter::Ptr filter_;

  std::string path_;
  std::string seg_path_;
  CollectionSchema::Ptr collection_schema_;
  SegmentMeta::Ptr segment_meta_;
  VersionManager::Ptr version_manager_;
  SegmentOptions options_;

  IDMap::Ptr id_map_;
  DeleteStore::Ptr delete_store_;

  // local_id(index) -> global_doc_id(value)
  std::vector<uint64_t> doc_ids_;

  std::array<std::variant<std::vector<int>,
                          std::unordered_map<std::string, std::vector<int>>>,
             static_cast<size_t>(BlockType::VECTOR_INDEX_QUANTIZE) + 1>
      persist_block_offsets_;
  std::array<
      std::variant<std::vector<BlockMeta>,
                   std::unordered_map<std::string, std::vector<BlockMeta>>>,
      static_cast<size_t>(BlockType::VECTOR_INDEX_QUANTIZE) + 1>
      persist_block_metas_;

  std::atomic<uint64_t> doc_id_allocator_{0};
  std::atomic<BlockID> block_id_allocator_{0};

  // wal
  WalFilePtr wal_file_{nullptr};

  bool sealed_{false};

  mutable std::mutex seg_mtx_;

  // segment column lock
  mutable std::shared_mutex seg_col_mtx_;

  bool need_destroyed_{false};

  // For performance tuning
  std::vector<std::shared_ptr<arrow::ChunkedArray>> persist_chunk_arrays_;
  std::vector<uint64_t> chunk_offsets_;
  std::unordered_map<std::string, int> col_idx_map_;
  bool use_fetch_perf_{false};

  // Inner classes
  class CombinedRecordBatchReader;
};

class SegmentImpl::CombinedRecordBatchReader : public arrow::RecordBatchReader {
 public:
  CombinedRecordBatchReader(
      std::shared_ptr<const SegmentImpl> segment,
      std::vector<std::shared_ptr<arrow::RecordBatchReader>> readers,
      const std::vector<std::string> &columns);

  ~CombinedRecordBatchReader();

  std::shared_ptr<arrow::Schema> schema() const override;

  arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch> *batch) override;

 private:
  std::shared_ptr<const SegmentImpl> segment_;
  std::vector<std::shared_ptr<arrow::RecordBatchReader>> readers_;
  std::vector<uint64_t> offsets_;
  std::shared_ptr<arrow::Schema> projected_schema_;
  bool need_local_doc_id_ = false;
  size_t current_reader_index_;
  size_t local_doc_id_;
  int local_doc_id_col_index_ = -1;
};

////////////////////////////////////////////////////////////////////////////////////
// SegmentImpl implementation
////////////////////////////////////////////////////////////////////////////////////

bool SegmentImpl::SegmentIndexFilter::is_filtered(uint64_t id) const {
  auto impl = impl_.lock();
  if (!impl) return false;
  auto result = impl->get_global_doc_id(id);
  if (!result.has_value()) {
    return false;
  }
  uint64_t doc_id = result.value();
  if (delete_store_ && delete_store_->is_deleted(doc_id)) {
    return true;
  }
  return false;
}

Status SegmentImpl::Open(const SegmentOptions &options) {
  options_ = options;
  options_.enable_mmap_ = version_manager_->get_current_version().enable_mmap();

  filter_ =
      std::make_shared<SegmentIndexFilter>(delete_store_, shared_from_this());

  // load persist forward blocks
  auto s = load_persist_scalar_blocks();
  CHECK_RETURN_STATUS(s);

  // load scalar indexes
  s = load_scalar_index_blocks();
  CHECK_RETURN_STATUS(s);

  // load vector indexes
  s = load_vector_index_blocks();
  CHECK_RETURN_STATUS(s);

  auto writing_block = segment_meta_->writing_forward_block();
  if (!writing_block.has_value() && !options_.read_only_) {
    return Status::InternalError(
        "No writing block found when in writing mode.");
  }

  if (writing_block.has_value()) {
    // init doc_id_allocator and block_id_allocator
    doc_id_allocator_ = writing_block.value().min_doc_id();
    BlockID max_block_id{writing_block.value().id()};
    for (auto &block : segment_meta_->persisted_blocks()) {
      max_block_id = std::max(max_block_id, block.id());
    }
    block_id_allocator_ = max_block_id + 1;

    // recover writing block
    s = recover();
    CHECK_RETURN_STATUS(s);
  } else {
    // Update block_id_allocator_
    BlockID max_block_id{0};
    auto &persist_blocks = segment_meta_->persisted_blocks();
    for (const auto &block : persist_blocks) {
      max_block_id = std::max(max_block_id, block.id());
    }
    block_id_allocator_.store(max_block_id + 1);
  }

  fresh_persist_block_offset();

  fresh_persist_chunked_array();

  return Status::OK();
}

Status SegmentImpl::Create(const SegmentOptions &options, uint64_t min_doc_id) {
  options_ = options;
  filter_ =
      std::make_shared<SegmentIndexFilter>(delete_store_, shared_from_this());

  // init memory forward block
  auto block_id = allocate_block_id();
  std::vector<std::string> columns{GLOBAL_DOC_ID, USER_ID};
  std::vector<std::string> schema_forward_fields =
      collection_schema_->forward_field_names();
  columns.insert(columns.end(), schema_forward_fields.begin(),
                 schema_forward_fields.end());

  segment_meta_->set_writing_forward_block(
      {block_id, BlockType::SCALAR, min_doc_id, min_doc_id, 0, columns});
  auto vector_fields = collection_schema_->vector_fields();
  for (auto &field : vector_fields) {
    if (field->index_params()->type() == IndexType::FLAT) {
      segment_meta_->add_indexed_vector_field(field->name());
    }
  }
  auto s = load_scalar_index_blocks(true);
  CHECK_RETURN_STATUS(s);

  doc_id_allocator_.store(min_doc_id);

  return Status::OK();
}

Status SegmentImpl::close() {
  flush();
  if (invert_indexers_) {
    invert_indexers_.reset();
  }
  for (const auto &[name, indexers] : vector_indexers_) {
    for (auto indexer : indexers) {
      indexer->Close();
    }
  }
  vector_indexers_.clear();
  for (auto [name, indexer] : memory_vector_indexers_) {
    indexer->Close();
  }
  memory_vector_indexers_.clear();

  return Status::OK();
}

SegmentID SegmentImpl::id() const {
  return segment_meta_->id();
}

SegmentMeta::Ptr SegmentImpl::meta() const {
  return segment_meta_;
}

uint64_t SegmentImpl::doc_count(const IndexFilter::Ptr filter) {
  uint64_t doc_count = doc_ids_.size();
  if (filter) {
    for (const auto &doc_id : doc_ids_) {
      if (filter->is_filtered(doc_id)) {
        doc_count--;
      }
    }
  }

  return doc_count;
}

template <typename T>
struct is_vector : std::false_type {};

template <typename T, typename A>
struct is_vector<std::vector<T, A>> : std::true_type {};

template <typename ValueType>
Status SegmentImpl::InsertScalar(InvertedColumnIndexer::Ptr &indexer,
                                 const Doc &doc,
                                 const FieldSchema::Ptr &field) {
  auto value = doc.get<ValueType>(field->name());
  auto segment_doc_id = doc_ids_.size();
  if (value.has_value()) {
    if constexpr (std::is_same_v<ValueType, std::vector<bool>>) {
      return indexer->insert(segment_doc_id, value.value());
    } else if constexpr (std::is_same_v<ValueType, std::vector<std::string>>) {
      return indexer->insert(segment_doc_id, value.value());
    } else if constexpr (is_vector<ValueType>::value) {
      const auto &vec = value.value();
      std::string value_str(
          reinterpret_cast<const char *>(vec.data()),
          vec.size() * sizeof(typename ValueType::value_type));
      return indexer->insert(segment_doc_id, value_str);
    } else if constexpr (std::is_same_v<ValueType, std::string>) {
      const ValueType &val = value.value();
      return indexer->insert(segment_doc_id, val);
    } else if constexpr (std::is_same_v<ValueType, bool>) {
      const ValueType &val = value.value();
      return indexer->insert(segment_doc_id, val);
    } else {
      const ValueType &val = value.value();
      std::string value_str(reinterpret_cast<const char *>(&val),
                            sizeof(ValueType));
      return indexer->insert(segment_doc_id, value_str);
    }
  } else {
    return indexer->insert_null(segment_doc_id);
  }
  return Status::OK();
}

template <typename ValueType>
Status SegmentImpl::InsertVector(VectorColumnIndexer::Ptr &indexer,
                                 const Doc &doc,
                                 const FieldSchema::Ptr &field) {
  auto value = doc.get<ValueType>(field->name());
  if (value.has_value()) {
    vector_column_params::VectorData vector_data;
    if constexpr (std::is_same_v<ValueType,
                                 std::pair<std::vector<uint32_t>,
                                           std::vector<float16_t>>>) {
      const std::vector<uint32_t> &sparse_indices = value.value().first;
      const std::vector<float16_t> &sparse_value = value.value().second;
      vector_data.vector = vector_column_params::SparseVector{
          (uint32_t)sparse_indices.size(), (void *)sparse_indices.data(),
          (void *)sparse_value.data()};
    } else if constexpr (std::is_same_v<ValueType,
                                        std::pair<std::vector<uint32_t>,
                                                  std::vector<float>>>) {
      const std::vector<uint32_t> &sparse_indices = value.value().first;
      const std::vector<float> &sparse_value = value.value().second;
      vector_data.vector = vector_column_params::SparseVector{
          (uint32_t)sparse_indices.size(), (void *)sparse_indices.data(),
          (void *)sparse_value.data()};
    } else {
      vector_data.vector =
          vector_column_params::DenseVector{.data = value.value().data()};
    }

    auto &mem_block_meta = segment_meta_->writing_forward_block().value();
    auto &block_doc_id = mem_block_meta.doc_count_;

    return indexer->Insert(vector_data, block_doc_id);
  } else {
    LOG_WARN("Field %s not found or is null for doc: %s", field->name().c_str(),
             doc.to_detail_string().c_str());
  }
  return Status::OK();
}

Status SegmentImpl::insert_scalar_indexer(Doc &doc) {
  for (const auto &field : collection_schema_->forward_fields()) {
    auto index_type = field->index_type();
    if (index_type != IndexType::INVERT) {
      continue;
    }
    auto indexer = get_scalar_indexer(field->name());
    if (!indexer) {
      return Status::InternalError("Field ", field->name(), " indexer is null");
    }
    Status status;
    auto data_type = field->data_type();
    switch (field->data_type()) {
      case DataType::BINARY: {
        status = InsertScalar<std::string>(indexer, doc, field);
        break;
      }
      case DataType::STRING: {
        status = InsertScalar<std::string>(indexer, doc, field);
        break;
      }
      case DataType::BOOL:
        status = InsertScalar<bool>(indexer, doc, field);
        break;
      case DataType::INT32:
        status = InsertScalar<int32_t>(indexer, doc, field);
        break;
      case DataType::INT64:
        status = InsertScalar<int64_t>(indexer, doc, field);
        break;
      case DataType::UINT32:
        status = InsertScalar<uint32_t>(indexer, doc, field);
        break;
      case DataType::UINT64:
        status = InsertScalar<uint64_t>(indexer, doc, field);
        break;
      case DataType::FLOAT:
        status = InsertScalar<float>(indexer, doc, field);
        break;
      case DataType::DOUBLE:
        status = InsertScalar<double>(indexer, doc, field);
        break;
      case DataType::ARRAY_BINARY:
        status = InsertScalar<std::vector<std::string>>(indexer, doc, field);
        break;
      case DataType::ARRAY_STRING:
        status = InsertScalar<std::vector<std::string>>(indexer, doc, field);
        break;
      case DataType::ARRAY_BOOL:
        status = InsertScalar<std::vector<bool>>(indexer, doc, field);
        break;
      case DataType::ARRAY_INT32:
        status = InsertScalar<std::vector<int32_t>>(indexer, doc, field);
        break;
      case DataType::ARRAY_INT64:
        status = InsertScalar<std::vector<int64_t>>(indexer, doc, field);
        break;
      case DataType::ARRAY_UINT32:
        status = InsertScalar<std::vector<uint32_t>>(indexer, doc, field);
        break;
      case DataType::ARRAY_UINT64:
        status = InsertScalar<std::vector<uint64_t>>(indexer, doc, field);
        break;
      case DataType::ARRAY_FLOAT:
        status = InsertScalar<std::vector<float>>(indexer, doc, field);
        break;
      case DataType::ARRAY_DOUBLE:
        status = InsertScalar<std::vector<double>>(indexer, doc, field);
        break;
      default:
        status = Status::InternalError("unsupport data type ",
                                       DataTypeCodeBook::AsString(data_type));
    }
    if (!status.ok()) {
      LOG_ERROR("insert scalar failed[%s]", status.message().c_str());
      return status;
    }
  }
  return Status::OK();
}

Status SegmentImpl::insert_vector_indexer(Doc &doc) {
  for (const auto &field : collection_schema_->vector_fields()) {
    std::vector<VectorColumnIndexer::Ptr> indexers;
    auto m_indexer = get_memory_vector_indexer(field->name());
    if (!m_indexer) {
      LOG_ERROR("vector indexer not found for field %s", field->name().c_str());
      return Status::InternalError("vector indexer not found for field: ",
                                   field->name());
    }
    indexers.push_back(m_indexer);
    auto vector_index_params =
        std::dynamic_pointer_cast<VectorIndexParams>(field->index_params());
    if (vector_index_params->quantize_type() != QuantizeType::UNDEFINED) {
      m_indexer = get_memory_quant_vector_indexer(field->name());
      if (!m_indexer) {
        LOG_ERROR("quant vector indexer not found for field %s",
                  field->name().c_str());
        return Status::InternalError(
            "quant vector indexer not found for field: ", field->name());
      }
      indexers.push_back(m_indexer);
    }

    for (auto indexer : indexers) {
      Status status;
      auto data_type = field->data_type();
      switch (data_type) {
        case DataType::VECTOR_BINARY32:
          status = InsertVector<std::vector<uint32_t>>(indexer, doc, field);
          break;
        case DataType::VECTOR_BINARY64:
          status = InsertVector<std::vector<uint64_t>>(indexer, doc, field);
          break;
        case DataType::VECTOR_FP16:
          status = InsertVector<std::vector<float16_t>>(indexer, doc, field);
          break;
        case DataType::VECTOR_FP32:
          status = InsertVector<std::vector<float>>(indexer, doc, field);
          break;
        case DataType::VECTOR_FP64:
          status = InsertVector<std::vector<double>>(indexer, doc, field);
          break;
        // case DataType::VECTOR_INT4:
        //   status = InsertVector<std::vector<int8_t>>(indexer, doc, field);
        //   break;
        case DataType::VECTOR_INT8:
          status = InsertVector<std::vector<int8_t>>(indexer, doc, field);
          break;
        case DataType::VECTOR_INT16:
          status = InsertVector<std::vector<int16_t>>(indexer, doc, field);
          break;
        case DataType::SPARSE_VECTOR_FP16:
          status = InsertVector<
              std::pair<std::vector<uint32_t>, std::vector<float16_t>>>(
              indexer, doc, field);
          break;
        case DataType::SPARSE_VECTOR_FP32:
          status = InsertVector<
              std::pair<std::vector<uint32_t>, std::vector<float>>>(indexer,
                                                                    doc, field);
          break;
        default:
          status = Status::InvalidArgument(
              "unsupport data type", DataTypeCodeBook::AsString(data_type));
      }
      if (!status.ok()) {
        LOG_ERROR("insert vector failed[%s]", status.message().c_str());
        return status;
      }
    }
  }
  return Status::OK();
}

Status SegmentImpl::internal_insert(Doc &doc) {
  uint64_t g_doc_id = doc_id_allocator_.fetch_add(1);
  doc.set_doc_id(g_doc_id);

  if (ready_for_dump_block()) {
    auto s = flush();
    CHECK_RETURN_STATUS(s);
  }

  // init writing memory components
  if (!memory_store_) {
    auto s = init_memory_components();
    CHECK_RETURN_STATUS(s);
  }

  // write idmap
  auto s = id_map_->upsert(doc.pk(), g_doc_id);
  CHECK_RETURN_STATUS(s);

  // write forward
  s = memory_store_->insert(doc);
  CHECK_RETURN_STATUS(s);

  // write scalar index
  s = insert_scalar_indexer(doc);
  if (!s.ok() && s.code() != StatusCode::ALREADY_EXISTS) {
    return s;
  }
  // write vector index
  s = insert_vector_indexer(doc);
  if (!s.ok() && s != Status::AlreadyExists()) {
    return s;
  }

  auto &mem_block = segment_meta_->writing_forward_block().value();
  mem_block.max_doc_id_ = g_doc_id;
  mem_block.doc_count_ = mem_block.doc_count_ + 1;

  doc_ids_.push_back(g_doc_id);

  return Status::OK();
}

Status SegmentImpl::internal_update(Doc &doc) {
  delete_store_->mark_deleted(doc.doc_id());
  return internal_insert(doc);
}

Status SegmentImpl::internal_upsert(Doc &doc) {
  uint64_t g_doc_id;
  bool exist = id_map_->has(doc.pk(), &g_doc_id);
  if (exist) {
    delete_store_->mark_deleted(g_doc_id);
  }
  return internal_insert(doc);
}

Status SegmentImpl::internal_delete(const Doc &doc) {
  delete_store_->mark_deleted(doc.doc_id());
  id_map_->remove(doc.pk());
  return Status::OK();
}

Status SegmentImpl::Insert(Doc &doc) {
  std::lock_guard lock(seg_mtx_);

  if (id_map_ && id_map_->has(doc.pk())) {
    return Status::AlreadyExists("insert failed: doc_id[", doc.pk(),
                                 "] already exists in collection");
  }

  doc.set_operator(Operator::INSERT);

  // append wal
  auto s = append_wal(doc);
  CHECK_RETURN_STATUS(s);

  return internal_insert(doc);
}

Status SegmentImpl::Update(Doc &doc) {
  std::lock_guard lock(seg_mtx_);
  uint64_t g_doc_id;
  if (!id_map_->has(doc.pk(), &g_doc_id)) {
    return Status::NotFound("update failed: doc_id[", doc.pk(),
                            "] not found in collection");
  }

  doc.set_doc_id(g_doc_id);
  doc.set_operator(Operator::UPDATE);

  // append wal
  auto s = append_wal(doc);
  CHECK_RETURN_STATUS(s);

  return internal_update(doc);
}

Status SegmentImpl::Upsert(Doc &doc) {
  std::lock_guard lock(seg_mtx_);

  doc.set_operator(Operator::UPSERT);

  // append wal
  auto s = append_wal(doc);
  CHECK_RETURN_STATUS(s);

  return internal_upsert(doc);
}

Status SegmentImpl::Delete(const std::string &pk) {
  std::lock_guard lock(seg_mtx_);

  uint64_t g_doc_id;
  if (!id_map_->has(pk, &g_doc_id)) {
    return Status::NotFound("primary key: ", pk, " not found");
  }
  if (delete_store_->is_deleted(g_doc_id)) {
    return Status::NotFound("primary key: ", pk, " g_doc_id: ", g_doc_id,
                            " already deleted");
  }

  Doc mutable_doc;
  mutable_doc.set_pk(pk);
  mutable_doc.set_doc_id(g_doc_id);
  mutable_doc.set_operator(Operator::DELETE);

  // append wal
  auto s = append_wal(mutable_doc);
  CHECK_RETURN_STATUS(s);

  return internal_delete(mutable_doc);
}

// Note: Here we have no way to determine if g_doc_id is valid
Status SegmentImpl::Delete(uint64_t g_doc_id) {
  std::lock_guard lock(seg_mtx_);
  if (delete_store_->is_deleted(g_doc_id)) {
    return Status::NotFound("g_doc_id:", g_doc_id, " already deleted");
  }

  Doc mutable_doc;
  mutable_doc.set_doc_id(g_doc_id);
  mutable_doc.set_operator(Operator::DELETE);

  // append wal
  auto s = append_wal(mutable_doc);
  CHECK_RETURN_STATUS(s);
  return internal_delete(mutable_doc);
}

template <typename T>
Status DenseVectorDataConverter(
    const FieldSchema::Ptr &field,
    const vector_column_params::DenseVectorBuffer &buffer, Doc *doc) {
  const T *data_ptr = reinterpret_cast<const T *>(buffer.data.data());
  size_t data_size = buffer.data.size() / sizeof(T);
  std::vector<T> vector_data(data_ptr, data_ptr + data_size);
  doc->set(field->name(), vector_data);
  return Status::OK();
}

template <typename IndexType, typename ValueType>
Status SparseVectorDataConverter(
    const FieldSchema::Ptr &field,
    const vector_column_params::SparseVectorBuffer &buffer, Doc *doc) {
  const IndexType *indices_ptr =
      reinterpret_cast<const IndexType *>(buffer.indices.data());
  size_t indices_size = buffer.indices.size() / sizeof(IndexType);
  std::vector<IndexType> indices_vector(indices_ptr,
                                        indices_ptr + indices_size);

  const ValueType *values_ptr =
      reinterpret_cast<const ValueType *>(buffer.values.data());
  size_t values_size = buffer.values.size() / sizeof(ValueType);
  std::vector<ValueType> values_vector(values_ptr, values_ptr + values_size);

  std::pair<std::vector<IndexType>, std::vector<ValueType>> sparse_vector_pair(
      std::move(indices_vector), std::move(values_vector));
  doc->set(field->name(), sparse_vector_pair);
  return Status::OK();
}


Status SegmentImpl::ConvertVectorDataBufferToDocField(
    const FieldSchema::Ptr &field,
    const vector_column_params::VectorDataBuffer &buf, Doc *doc) {
  Status status;
  if (std::holds_alternative<vector_column_params::DenseVectorBuffer>(
          buf.vector_buffer)) {
    const auto &dense_buffer =
        std::get<vector_column_params::DenseVectorBuffer>(buf.vector_buffer);
    switch (field->data_type()) {
      case DataType::VECTOR_BINARY32: {
        status = DenseVectorDataConverter<uint32_t>(field, dense_buffer, doc);
        break;
      }
      case DataType::VECTOR_BINARY64: {
        status = DenseVectorDataConverter<uint64_t>(field, dense_buffer, doc);
        break;
      }
      case DataType::VECTOR_FP16: {
        status = DenseVectorDataConverter<float16_t>(field, dense_buffer, doc);
        break;
      }
      case DataType::VECTOR_FP32: {
        status = DenseVectorDataConverter<float>(field, dense_buffer, doc);
        break;
      }
      case DataType::VECTOR_FP64: {
        status = DenseVectorDataConverter<double>(field, dense_buffer, doc);
        break;
      }
      // case DataType::VECTOR_INT4: {
      //   status = DenseVectorDataConverter<int8_t>(field, dense_buffer, doc);
      //   break;
      // }
      case DataType::VECTOR_INT8: {
        status = DenseVectorDataConverter<int8_t>(field, dense_buffer, doc);
        break;
      }
      case DataType::VECTOR_INT16: {
        status = DenseVectorDataConverter<int16_t>(field, dense_buffer, doc);
        break;
      }
      default:
        return Status::InvalidArgument(
            "Unsupported dense vector element type: ", field->data_type());
    }
  } else if (std::holds_alternative<vector_column_params::SparseVectorBuffer>(
                 buf.vector_buffer)) {
    const auto &sparse_buffer =
        std::get<vector_column_params::SparseVectorBuffer>(buf.vector_buffer);
    switch (field->data_type()) {
      case DataType::SPARSE_VECTOR_FP16: {
        status = SparseVectorDataConverter<uint32_t, float16_t>(
            field, sparse_buffer, doc);
        break;
      }
      case DataType::SPARSE_VECTOR_FP32: {
        status = SparseVectorDataConverter<uint32_t, float>(field,
                                                            sparse_buffer, doc);
        break;
      }
      default:
        return Status::InvalidArgument(
            "Unsupported sparse vector element type: ", field->data_type());
    }
  } else {
    return Status::InvalidArgument("Unsupported vector buffer type");
  }

  return status;
}


Doc::Ptr SegmentImpl::Fetch(uint64_t g_doc_id) {
  std::lock_guard lock(seg_mtx_);

  if (g_doc_id > segment_meta_->max_doc_id()) {
    LOG_ERROR("g_doc_id[%zu] not exist in segment[%d] ", (size_t)g_doc_id,
              id());
    return nullptr;
  }

  int segment_doc_id = 0;
  auto it = std::lower_bound(doc_ids_.begin(), doc_ids_.end(), g_doc_id);
  if (it != doc_ids_.end() && *it == g_doc_id) {
    segment_doc_id = static_cast<int>(std::distance(doc_ids_.begin(), it));
  } else {
    LOG_ERROR(
        "g_doc_id[%zu] not found in doc_ids_[%zu], min_doc_id[%zu] "
        "max_doc_id[%zu], meta[%s]",
        (size_t)g_doc_id, doc_ids_.size(), (size_t)doc_ids_.front(),
        (size_t)doc_ids_.back(), segment_meta_->to_string_formatted().c_str());
    return nullptr;
  }

  std::vector<std::string> forward_columns;
  forward_columns.push_back(GLOBAL_DOC_ID);
  forward_columns.push_back(USER_ID);
  for (const auto &field : collection_schema_->forward_fields()) {
    forward_columns.push_back(field->name());
  }

  // Build result schema
  std::vector<std::shared_ptr<arrow::Field>> fields;
  for (size_t i = 0; i < forward_columns.size(); ++i) {
    const auto &col = forward_columns[i];
    if (col == GLOBAL_DOC_ID) {
      fields.push_back(arrow::field(GLOBAL_DOC_ID, arrow::uint64()));
    } else if (col == USER_ID) {
      fields.push_back(arrow::field(USER_ID, arrow::utf8()));
    } else {
      auto *field = collection_schema_->get_field(col);
      std::shared_ptr<arrow::Field> arrow_field;
      auto status = ConvertFieldSchemaToArrowField(field, &arrow_field);
      if (!status.ok()) {
        LOG_ERROR("Convert field schema failed: %s",
                  field->to_string().c_str());
        return nullptr;
      }
      fields.push_back(std::move(arrow_field));
    }
  }
  auto result_schema = std::make_shared<arrow::Schema>(fields);

  // fetch forward columns
  auto exec_batch = fetch(forward_columns, segment_doc_id);
  if (!exec_batch) {
    LOG_ERROR("Fetch failed, doc_id: %zu", (size_t)g_doc_id);
    return nullptr;
  }
  if (exec_batch->length != 1) {
    LOG_ERROR("Fetch failed, doc_id: %zu, num_rows: %zu != 1", (size_t)g_doc_id,
              (size_t)exec_batch->length);
    return nullptr;
  }

  if (exec_batch->num_values() != (int)forward_columns.size()) {
    LOG_ERROR("table column size error, expect %zu, actual %d",
              forward_columns.size(), exec_batch->num_values());
    return nullptr;
  }

  auto doc = std::make_shared<Doc>();

  // column 0 is the global doc_id
  if (auto doc_id_scalar = std::static_pointer_cast<arrow::Int64Scalar>(
          (*exec_batch)[0].scalar())) {
    doc->set_doc_id(doc_id_scalar->value);
  } else {
    LOG_ERROR("Global doc id scalar is not of int64 type");
    return nullptr;
  }

  // column 1 is the uid(pk)
  if (auto str_scalar = std::dynamic_pointer_cast<arrow::StringScalar>(
          (*exec_batch)[1].scalar())) {
    doc->set_pk(std::string(str_scalar->view()));
  } else {
    LOG_ERROR("Primary key scalar is not of string type");
    return nullptr;
  }

  // other forward columns
  for (int col_idx = 2; col_idx < exec_batch->num_values(); ++col_idx) {
    auto column_name = forward_columns[col_idx];
    auto column = result_schema->GetFieldByName(column_name);
    auto &column_scalar = (*exec_batch)[col_idx].scalar();
    if (column_scalar == nullptr || column_scalar->is_valid == false) {
      continue;
    }
    switch (column->type()->id()) {
      case arrow::Type::STRING: {
        auto str_scalar =
            std::dynamic_pointer_cast<arrow::StringScalar>(column_scalar);
        doc->set(column_name, std::string(str_scalar->view()));
        break;
      }
      case arrow::Type::INT32: {
        auto int32_scalar =
            std::dynamic_pointer_cast<arrow::Int32Scalar>(column_scalar);
        doc->set(column_name, int32_scalar->value);
        break;
      }
      case arrow::Type::INT64: {
        auto int64_scalar =
            std::dynamic_pointer_cast<arrow::Int64Scalar>(column_scalar);
        doc->set(column_name, int64_scalar->value);
        break;
      }
      case arrow::Type::UINT32: {
        auto uint32_scalar =
            std::dynamic_pointer_cast<arrow::UInt32Scalar>(column_scalar);
        doc->set(column_name, uint32_scalar->value);
        break;
      }
      case arrow::Type::UINT64: {
        auto uint64_scalar =
            std::dynamic_pointer_cast<arrow::UInt64Scalar>(column_scalar);
        doc->set(column_name, uint64_scalar->value);
        break;
      }
      case arrow::Type::DOUBLE: {
        auto double_scalar =
            std::dynamic_pointer_cast<arrow::DoubleScalar>(column_scalar);
        doc->set(column_name, double_scalar->value);
        break;
      }
      case arrow::Type::FLOAT: {
        auto float_scalar =
            std::dynamic_pointer_cast<arrow::FloatScalar>(column_scalar);
        doc->set(column_name, float_scalar->value);
        break;
      }
      case arrow::Type::BOOL: {
        auto bool_scalar =
            std::dynamic_pointer_cast<arrow::BooleanScalar>(column_scalar);
        doc->set(column_name, bool_scalar->value);
        break;
      }
      case arrow::Type::BINARY: {
        auto binary_scalar =
            std::dynamic_pointer_cast<arrow::BinaryScalar>(column_scalar);
        doc->set(column_name, std::string(binary_scalar->view()));
        break;
      }
      case arrow::Type::LIST: {
        auto list_scalar =
            std::dynamic_pointer_cast<arrow::ListScalar>(column_scalar);
        if (list_scalar && list_scalar->value) {
          auto list_type =
              std::dynamic_pointer_cast<arrow::ListType>(column->type());
          if (list_type) {
            auto value_type = list_type->value_type();
            switch (value_type->id()) {
              case arrow::Type::BOOL: {
                std::vector<bool> values;
                auto array = std::dynamic_pointer_cast<arrow::BooleanArray>(
                    list_scalar->value);
                if (array) {
                  values.reserve(array->length());
                  for (int64_t i = 0; i < array->length(); ++i) {
                    if (array->IsValid(i)) {
                      values.push_back(array->Value(i));
                    } else {
                      LOG_ERROR(
                          "Invalid arrow::boolean array value at index %zu",
                          (size_t)i);
                      continue;
                    }
                  }
                  doc->set(column_name, values);
                }
                break;
              }
              case arrow::Type::INT32: {
                std::vector<int32_t> values;
                auto array = std::dynamic_pointer_cast<arrow::Int32Array>(
                    list_scalar->value);
                if (array) {
                  values.reserve(array->length());
                  for (int64_t i = 0; i < array->length(); ++i) {
                    if (array->IsValid(i)) {
                      values.push_back(array->Value(i));
                    }
                  }
                  doc->set(column_name, values);
                }
                break;
              }
              case arrow::Type::INT64: {
                std::vector<int64_t> values;
                auto array = std::dynamic_pointer_cast<arrow::Int64Array>(
                    list_scalar->value);
                if (array) {
                  values.reserve(array->length());
                  for (int64_t i = 0; i < array->length(); ++i) {
                    if (array->IsValid(i)) {
                      values.push_back(array->Value(i));
                    }
                  }
                  doc->set(column_name, values);
                }
                break;
              }
              case arrow::Type::UINT32: {
                std::vector<uint32_t> values;
                auto array = std::dynamic_pointer_cast<arrow::UInt32Array>(
                    list_scalar->value);
                if (array) {
                  values.reserve(array->length());
                  for (int64_t i = 0; i < array->length(); ++i) {
                    if (array->IsValid(i)) {
                      values.push_back(array->Value(i));
                    }
                  }
                  doc->set(column_name, values);
                }
                break;
              }
              case arrow::Type::UINT64: {
                std::vector<uint64_t> values;
                auto array = std::dynamic_pointer_cast<arrow::UInt64Array>(
                    list_scalar->value);
                if (array) {
                  values.reserve(array->length());
                  for (int64_t i = 0; i < array->length(); ++i) {
                    if (array->IsValid(i)) {
                      values.push_back(array->Value(i));
                    }
                  }
                  doc->set(column_name, values);
                }
                break;
              }
              case arrow::Type::FLOAT: {
                std::vector<float> values;
                auto array = std::dynamic_pointer_cast<arrow::FloatArray>(
                    list_scalar->value);
                if (array) {
                  values.reserve(array->length());
                  for (int64_t i = 0; i < array->length(); ++i) {
                    if (array->IsValid(i)) {
                      values.push_back(array->Value(i));
                    }
                  }
                  doc->set(column_name, values);
                }
                break;
              }
              case arrow::Type::DOUBLE: {
                std::vector<double> values;
                auto array = std::dynamic_pointer_cast<arrow::DoubleArray>(
                    list_scalar->value);
                if (array) {
                  values.reserve(array->length());
                  for (int64_t i = 0; i < array->length(); ++i) {
                    if (array->IsValid(i)) {
                      values.push_back(array->Value(i));
                    }
                  }
                  doc->set(column_name, values);
                }
                break;
              }
              case arrow::Type::STRING: {
                std::vector<std::string> values;
                auto array = std::dynamic_pointer_cast<arrow::StringArray>(
                    list_scalar->value);
                if (array) {
                  values.reserve(array->length());
                  for (int64_t i = 0; i < array->length(); ++i) {
                    if (array->IsValid(i)) {
                      values.push_back(array->GetString(i));
                    }
                  }
                  doc->set(column_name, values);
                }
                break;
              }
              case arrow::Type::BINARY: {
                std::vector<std::string> values;
                auto array = std::dynamic_pointer_cast<arrow::BinaryArray>(
                    list_scalar->value);
                if (array) {
                  values.reserve(array->length());
                  for (int64_t i = 0; i < array->length(); ++i) {
                    if (array->IsValid(i)) {
                      values.push_back(array->GetString(i));
                    }
                  }
                  doc->set(column_name, values);
                }
                break;
              }
              default:
                LOG_WARN("Unsupported list element type: %s",
                         value_type->ToString().c_str());
                break;
            }
          }
        }
        break;
      }
      default:
        LOG_ERROR("Unsupported type: %s", column_name.c_str());
        break;
    }
  }

  // fetch vector
  for (const auto &field : collection_schema_->vector_fields()) {
    int block_idx = find_persist_block_id(BlockType::VECTOR_INDEX,
                                          segment_doc_id, field->name());
    if (block_idx != -1) {
      const auto &block_offsets =
          get_persist_block_offsets(BlockType::VECTOR_INDEX, field->name());
      auto block_offset = block_offsets[block_idx];
      auto local_row = segment_doc_id - block_offset;

      auto column_name = field->name();
      auto iter = vector_indexers_.find(column_name);
      if (iter != vector_indexers_.end()) {
        const auto &vector_indexers = iter->second;
        if (block_idx >= (int)vector_indexers.size()) {
          LOG_ERROR("block_idx[%d] out of range[%lu]", block_idx,
                    vector_indexers.size());
          continue;
        }
        auto vector_indexer = vector_indexers[block_idx];
        auto fetch_result = vector_indexer->Fetch(local_row);
        if (!fetch_result) {
          LOG_ERROR(
              "vector indexer fetch failed, local_row: %d, block_idx: %d, "
              "segment_doc_id: %d",
              local_row, block_idx, segment_doc_id);
          return nullptr;
        }
        const auto &vector_buffer = fetch_result.value();
        auto status =
            ConvertVectorDataBufferToDocField(field, vector_buffer, doc.get());
        if (!status.ok()) {
          LOG_ERROR("convert vector data buffer to doc field failed %s",
                    status.message().c_str());
        }
      }

    } else {
      if (segment_meta_->has_writing_forward_block()) {
        const auto &p_block_offsets =
            get_persist_block_offsets(BlockType::VECTOR_INDEX, field->name());
        const auto &p_block_metas =
            get_persist_block_metas(BlockType::VECTOR_INDEX, field->name());
        auto mem_block_offset =
            p_block_offsets.empty()
                ? 0
                : p_block_offsets.back() + p_block_metas.back().doc_count_;
        int local_row = segment_doc_id - mem_block_offset;
        auto column_name = field->name();
        auto iter = memory_vector_indexers_.find(column_name);
        if (iter != memory_vector_indexers_.end()) {
          auto vector_indexer = iter->second;
          auto fetch_result = vector_indexer->Fetch(local_row);
          if (!fetch_result.has_value()) {
            LOG_ERROR(
                "vector indexer fetch failed, column: %s, doc_count: %lu, "
                "mem_block_offset: %d, local_row: %d",
                field->name().c_str(), vector_indexer->doc_count(),
                mem_block_offset, local_row);
            continue;
          }
          const auto &vector_buffer = fetch_result.value();
          auto status = ConvertVectorDataBufferToDocField(field, vector_buffer,
                                                          doc.get());
          if (!status.ok()) {
            LOG_ERROR("convert vector data buffer to doc field failed %s",
                      status.message().c_str());
          }
        }
      } else {
        LOG_ERROR("Can't find vector block for g_doc_id: %zu",
                  (size_t)g_doc_id);
      }
    }
  }

  return doc;
}

CombinedVectorColumnIndexer::Ptr SegmentImpl::get_combined_vector_indexer(
    const std::string &field_name) const {
  std::vector<VectorColumnIndexer::Ptr> indexers;
  auto iter = vector_indexers_.find(field_name);
  if (iter != vector_indexers_.end()) {
    indexers = iter->second;
  }
  auto m_iter = memory_vector_indexers_.find(field_name);
  if (m_iter != memory_vector_indexers_.end()) {
    indexers.push_back(m_iter->second);
  }

  auto field = collection_schema_->get_field(field_name);
  auto vector_index_params =
      std::dynamic_pointer_cast<VectorIndexParams>(field->index_params());
  MetricType metric_type = vector_index_params->metric_type();
  auto blocks = get_persist_block_metas(BlockType::VECTOR_INDEX, field_name);

  auto normal_indexers = indexers;
  return std::make_shared<CombinedVectorColumnIndexer>(
      indexers, normal_indexers, *field, *segment_meta_, std::move(blocks),
      metric_type);
}

CombinedVectorColumnIndexer::Ptr SegmentImpl::get_quant_combined_vector_indexer(
    const std::string &field_name) const {
  std::vector<VectorColumnIndexer::Ptr> indexers;
  auto iter = quant_vector_indexers_.find(field_name);
  if (iter != quant_vector_indexers_.end()) {
    indexers = iter->second;
  }
  auto m_iter = quant_memory_vector_indexers_.find(field_name);
  if (m_iter != quant_memory_vector_indexers_.end()) {
    indexers.push_back(m_iter->second);
  }

  std::vector<VectorColumnIndexer::Ptr> normal_indexers;
  iter = vector_indexers_.find(field_name);
  if (iter != vector_indexers_.end()) {
    normal_indexers = iter->second;
  }
  m_iter = memory_vector_indexers_.find(field_name);
  if (m_iter != memory_vector_indexers_.end()) {
    normal_indexers.push_back(m_iter->second);
  }

  auto field = collection_schema_->get_field(field_name);
  auto vector_index_params =
      std::dynamic_pointer_cast<VectorIndexParams>(field->index_params());
  MetricType metric_type = vector_index_params->metric_type();
  auto blocks =
      get_persist_block_metas(BlockType::VECTOR_INDEX_QUANTIZE, field_name);

  return std::make_shared<CombinedVectorColumnIndexer>(
      indexers, normal_indexers, *field, *segment_meta_, std::move(blocks),
      metric_type, true);
}

VectorColumnIndexer::Ptr SegmentImpl::get_memory_vector_indexer(
    const std::string &field_name) {
  auto iter = memory_vector_indexers_.find(field_name);
  if (iter != memory_vector_indexers_.end()) {
    return iter->second;
  }
  return nullptr;
}

VectorColumnIndexer::Ptr SegmentImpl::get_memory_quant_vector_indexer(
    const std::string &field_name) {
  auto iter = quant_memory_vector_indexers_.find(field_name);
  if (iter != quant_memory_vector_indexers_.end()) {
    return iter->second;
  }
  return nullptr;
}

std::vector<VectorColumnIndexer::Ptr> SegmentImpl::get_vector_indexer(
    const std::string &field_name) const {
  auto iter = vector_indexers_.find(field_name);
  if (iter != vector_indexers_.end()) {
    return iter->second;
  }
  return std::vector<VectorColumnIndexer::Ptr>();
}

std::vector<VectorColumnIndexer::Ptr> SegmentImpl::get_quant_vector_indexer(
    const std::string &field_name) const {
  std::vector<VectorColumnIndexer::Ptr> col_indexers;
  auto iter = quant_vector_indexers_.find(field_name);
  if (iter != quant_vector_indexers_.end()) {
    return iter->second;
  }
  return std::vector<VectorColumnIndexer::Ptr>();
}

InvertedColumnIndexer::Ptr SegmentImpl::get_scalar_indexer(
    const std::string &field_name) const {
  if (invert_indexers_) {
    return (*invert_indexers_)[field_name];
  }
  return nullptr;
}

const IndexFilter::Ptr SegmentImpl::get_filter() {
  return delete_store_->empty() ? nullptr : filter_;
}

Status SegmentImpl::create_all_vector_index(
    int concurrency, SegmentMeta::Ptr *segment_meta,
    std::unordered_map<std::string, VectorColumnIndexer::Ptr> *vector_indexers,
    std::unordered_map<std::string, VectorColumnIndexer::Ptr>
        *quant_vector_indexers) {
  const auto &vector_fields = collection_schema_->vector_fields();

  auto new_segment_meta = std::make_shared<SegmentMeta>(*segment_meta_);
  new_segment_meta->remove_writing_forward_block();

  std::set<std::string> vector_field_names;
  for (const auto &field : vector_fields) {
    auto s = create_vector_index(field->name(), field->index_params(),
                                 concurrency, &new_segment_meta,
                                 vector_indexers, quant_vector_indexers);
    CHECK_RETURN_STATUS(s);
    vector_field_names.insert(field->name());
  }

  new_segment_meta->set_indexed_vector_fields(vector_field_names);
  *segment_meta = new_segment_meta;

  return Status::OK();
}

Result<VectorColumnIndexer::Ptr> SegmentImpl::merge_vector_indexer(
    const std::string &index_file_path, const std::string &column,
    const FieldSchema &field, int concurrency) {
  VectorColumnIndexer::Ptr vector_indexer =
      std::make_shared<VectorColumnIndexer>(index_file_path, field);

  vector_column_params::ReadOptions options{options_.enable_mmap_, true};

  auto s = vector_indexer->Open(options);
  CHECK_RETURN_STATUS_EXPECTED(s);
  std::vector<VectorColumnIndexer::Ptr> to_merge_indexers =
      vector_indexers_[column];
  vector_column_params::MergeOptions merge_options;
  if (concurrency == 0) {
    merge_options.pool = GlobalResource::Instance().optimize_thread_pool();
    merge_options.write_concurrency =
        GlobalConfig::Instance().optimize_thread_count();
  } else {
    merge_options.write_concurrency = concurrency;
  }
  s = vector_indexer->Merge(to_merge_indexers, filter_, merge_options);
  CHECK_RETURN_STATUS_EXPECTED(s);
  s = vector_indexer->Flush();
  CHECK_RETURN_STATUS_EXPECTED(s);

  return vector_indexer;
}

Status SegmentImpl::create_vector_index(
    const std::string &column, const IndexParams::Ptr &index_params,
    int concurrency, SegmentMeta::Ptr *segment_meta,
    std::unordered_map<std::string, VectorColumnIndexer::Ptr> *vector_indexers,
    std::unordered_map<std::string, VectorColumnIndexer::Ptr>
        *quant_vector_indexers) {
  auto field = collection_schema_->get_vector_field(column);
  SegmentMeta::Ptr new_segment_meta;
  if (*segment_meta == nullptr) {
    new_segment_meta = std::make_shared<SegmentMeta>(*segment_meta_);
    new_segment_meta->remove_writing_forward_block();
  } else {
    new_segment_meta = *segment_meta;
  }

  if (segment_meta_->vector_indexed(column) &&
      *field->index_params() == *index_params) {
    // if segment is already indexed and index params are same, skip create
    *segment_meta = new_segment_meta;
    return Status::OK();
  }
  new_segment_meta->add_indexed_vector_field(column);

  auto vector_index_params =
      std::dynamic_pointer_cast<VectorIndexParams>(index_params);

  if (vector_index_params->quantize_type() == QuantizeType::UNDEFINED) {
    auto block_id = allocate_block_id();

    auto field_with_new_index_params = std::make_shared<FieldSchema>(*field);
    field_with_new_index_params->set_index_params(index_params);

    std::string index_file_path = FileHelper::MakeVectorIndexPath(
        path_, column, segment_meta_->id(), block_id);
    auto vector_indexer = merge_vector_indexer(
        index_file_path, column, *field_with_new_index_params, concurrency);
    if (!vector_indexer.has_value()) {
      return vector_indexer.error();
    }

    vector_indexers->insert({column, vector_indexer.value()});

    new_segment_meta->remove_vector_persisted_block(column);
    BlockMeta block;
    block.set_id(block_id);
    block.set_type(BlockType::VECTOR_INDEX);
    block.set_columns({column});
    block.set_min_doc_id(doc_ids_.front());
    block.set_max_doc_id(doc_ids_.back());
    block.set_doc_count(doc_ids_.size());
    new_segment_meta->add_persisted_block(block);

    *segment_meta = new_segment_meta;

  } else {
    auto original_index_params =
        std::dynamic_pointer_cast<VectorIndexParams>(field->index_params());

    if (!(vector_index_params->metric_type() ==
              original_index_params->metric_type() &&
          vector_indexers_[column].size() == 1)) {
      BlockID block_id = allocate_block_id();

      auto field_with_flat = std::make_shared<FieldSchema>(*field);
      field_with_flat->set_index_params(
          MakeDefaultVectorIndexParams(vector_index_params->metric_type()));

      std::string index_file_path = FileHelper::MakeVectorIndexPath(
          path_, column, segment_meta_->id(), block_id);
      auto vector_indexer = merge_vector_indexer(index_file_path, column,
                                                 *field_with_flat, concurrency);
      if (!vector_indexer.has_value()) {
        return vector_indexer.error();
      }

      vector_indexers->insert({column, vector_indexer.value()});

      new_segment_meta->remove_vector_persisted_block(column, false);
      BlockMeta block;
      block.set_id(block_id);
      block.set_type(BlockType::VECTOR_INDEX);
      block.set_columns({column});
      block.set_min_doc_id(meta()->min_doc_id());
      block.set_max_doc_id(meta()->max_doc_id());
      block.set_doc_count(meta()->doc_count());
      new_segment_meta->add_persisted_block(block);
    }

    auto quant_block_id = allocate_block_id();
    auto field_with_new_index_params = std::make_shared<FieldSchema>(*field);
    field_with_new_index_params->set_index_params(index_params);

    std::string index_file_path = FileHelper::MakeQuantizeVectorIndexPath(
        path_, column, segment_meta_->id(), quant_block_id);
    auto vector_indexer = merge_vector_indexer(
        index_file_path, column, *field_with_new_index_params, concurrency);
    if (!vector_indexer.has_value()) {
      return vector_indexer.error();
    }

    quant_vector_indexers->insert({column, vector_indexer.value()});

    new_segment_meta->remove_vector_persisted_block(column, true);
    BlockMeta block;
    block.set_id(quant_block_id);
    block.set_type(BlockType::VECTOR_INDEX_QUANTIZE);
    block.set_columns({column});
    block.set_min_doc_id(meta()->min_doc_id());
    block.set_max_doc_id(meta()->max_doc_id());
    block.set_doc_count(meta()->doc_count());
    new_segment_meta->add_persisted_block(block);

    *segment_meta = new_segment_meta;
  }

  return Status::OK();
}

Status SegmentImpl::drop_vector_index(
    const std::string &column, SegmentMeta::Ptr *segment_meta,
    std::unordered_map<std::string, VectorColumnIndexer::Ptr>
        *vector_indexers) {
  auto field = collection_schema_->get_vector_field(column);
  auto new_segment_meta = std::make_shared<SegmentMeta>(*segment_meta_);
  new_segment_meta->remove_writing_forward_block();
  new_segment_meta->add_indexed_vector_field(column);

  if (*field->index_params() == DefaultVectorIndexParams) {
    *segment_meta = new_segment_meta;
    return Status::OK();
  }

  auto vector_index_params =
      std::dynamic_pointer_cast<VectorIndexParams>(field->index_params());

  auto block_id = allocate_block_id();

  auto field_with_default_index = std::make_shared<FieldSchema>(*field);
  field_with_default_index->set_index_params(DefaultVectorIndexParams);

  std::string index_file_path = FileHelper::MakeVectorIndexPath(
      path_, column, segment_meta_->id(), block_id);

  auto new_vector_indexer = std::make_shared<VectorColumnIndexer>(
      index_file_path, *field_with_default_index);
  vector_column_params::ReadOptions options{options_.enable_mmap_, true};

  auto s = new_vector_indexer->Open(options);
  CHECK_RETURN_STATUS(s);
  s = new_vector_indexer->Merge(vector_indexers_[column], nullptr);
  CHECK_RETURN_STATUS(s);
  s = new_vector_indexer->Flush();
  CHECK_RETURN_STATUS(s);

  (*vector_indexers)[column] = new_vector_indexer;
  new_segment_meta->remove_vector_persisted_block(
      column, vector_index_params->quantize_type() != QuantizeType::UNDEFINED);

  BlockMeta block;
  block.set_id(block_id);
  block.set_type(BlockType::VECTOR_INDEX);
  block.set_columns({column});
  block.set_min_doc_id(meta()->min_doc_id());
  block.set_max_doc_id(meta()->max_doc_id());
  block.set_doc_count(meta()->doc_count());
  new_segment_meta->add_persisted_block(block);

  *segment_meta = new_segment_meta;

  return Status::OK();
}

Status SegmentImpl::reload_vector_index(
    const CollectionSchema &schema, const SegmentMeta::Ptr &new_segment_meta,
    const std::unordered_map<std::string, VectorColumnIndexer::Ptr>
        &vector_indexers,
    const std::unordered_map<std::string, VectorColumnIndexer::Ptr>
        &quant_vector_indexers) {
  collection_schema_ = std::make_shared<CollectionSchema>(schema);
  segment_meta_ = new_segment_meta;
  fresh_persist_block_offset();

  auto vector_fields = schema.vector_fields();

  for (auto field : vector_fields) {
    auto vector_index_params =
        std::dynamic_pointer_cast<VectorIndexParams>(field->index_params());

    if (vector_index_params->quantize_type() == QuantizeType::UNDEFINED) {
      auto iter = vector_indexers.find(field->name());
      if (iter != vector_indexers.end()) {
        auto indexers = vector_indexers_[field->name()];
        for (auto indexer : indexers) {
          auto s = indexer->Destroy();
          CHECK_RETURN_STATUS(s);
        }
        vector_indexers_[field->name()] = {iter->second};
      }
      auto q_iter = quant_vector_indexers_.find(field->name());
      if (q_iter != quant_vector_indexers_.end()) {
        auto q_indexers = q_iter->second;
        for (auto q_indexer : q_indexers) {
          auto s = q_indexer->Destroy();
          CHECK_RETURN_STATUS(s);
        }
        quant_vector_indexers_.erase(q_iter);
      }
    } else {
      auto iter = vector_indexers.find(field->name());
      if (iter != vector_indexers.end()) {
        auto indexers = vector_indexers_[field->name()];
        for (auto indexer : indexers) {
          auto s = indexer->Destroy();
          CHECK_RETURN_STATUS(s);
        }
        vector_indexers_[field->name()] = {iter->second};
      }
      auto q_iter = quant_vector_indexers.find(field->name());
      if (q_iter != quant_vector_indexers.end()) {
        auto q_indexers = quant_vector_indexers_[field->name()];
        for (auto q_indexer : q_indexers) {
          auto s = q_indexer->Destroy();
          CHECK_RETURN_STATUS(s);
        }
        quant_vector_indexers_[field->name()] = {q_iter->second};
      }
    }
  }

  return Status::OK();
}

bool SegmentImpl::vector_index_ready(
    const std::string &column, const IndexParams::Ptr &index_params) const {
  auto field = collection_schema_->get_vector_field(column);
  return segment_meta_->vector_indexed(column) &&
         *field->index_params() == *index_params;
}

bool SegmentImpl::all_vector_index_ready() const {
  for (const auto &field : collection_schema_->vector_fields()) {
    if (!segment_meta_->vector_indexed(field->name())) {
      return false;
    }
  }
  return true;
}

Status SegmentImpl::create_scalar_index(const std::vector<std::string> &columns,
                                        const IndexParams::Ptr &index_params,
                                        SegmentMeta::Ptr *segment_meta,
                                        InvertedIndexer::Ptr *scalar_indexer) {
  // validate
  std::vector<FieldSchema> fields;
  std::vector<std::string> field_names;

  for (const auto &column : columns) {
    auto field = collection_schema_->get_field(column);
    if (!field || field->is_vector_field()) {
      return Status::InvalidArgument("Invalid column name");
    }

    if (field->index_params() != nullptr &&
        *field->index_params() == *index_params) {
      // if already indexed, just skip it
      continue;
    }

    auto new_field = std::make_shared<FieldSchema>(*field);
    new_field->set_index_params(index_params);

    fields.push_back(*new_field);
    field_names.push_back(new_field->name());
  }

  auto new_segment_meta = std::make_shared<SegmentMeta>(*segment_meta_);
  if (fields.empty()) {
    *segment_meta = new_segment_meta;
    return Status::OK();
  }

  new_segment_meta->remove_scalar_index_block();

  // create scalar indexer
  // clone original indexer
  auto block_id = allocate_block_id();
  std::string new_invert_index_path =
      FileHelper::MakeInvertIndexPath(path_, id(), block_id);

  Status s;
  InvertedIndexer::Ptr new_scalar_indexer{nullptr};
  if (invert_indexers_) {
    s = invert_indexers_->create_snapshot(new_invert_index_path);
    CHECK_RETURN_STATUS(s);

    auto inverted_fields_ptr = collection_schema_->forward_fields_with_index();
    std::vector<FieldSchema> inverted_fields;
    std::vector<std::string> inverted_field_names;
    for (auto field : inverted_fields_ptr) {
      inverted_fields.push_back(*field);
      inverted_field_names.push_back(field->name());
    }

    new_scalar_indexer = InvertedIndexer::CreateAndOpen(
        collection_schema_->name(), new_invert_index_path, false,
        inverted_fields, false);
    if (!new_scalar_indexer) {
      LOG_ERROR("Failed to create scalar indexer");
      return Status::InternalError("Failed to create scalar indexer");
    }
    for (const auto &field : fields) {
      if (std::find(inverted_field_names.begin(), inverted_field_names.end(),
                    field.name()) != inverted_field_names.end()) {
        s = new_scalar_indexer->remove_column_indexer(field.name());
        CHECK_RETURN_STATUS(s);
      }
      s = new_scalar_indexer->create_column_indexer(field);
      CHECK_RETURN_STATUS(s);
    }
  } else {
    new_scalar_indexer = InvertedIndexer::CreateAndOpen(
        collection_schema_->name(), new_invert_index_path, true, fields, false);
    if (!new_scalar_indexer) {
      LOG_ERROR("Failed to create scalar indexer");
      return Status::InternalError("Failed to create scalar indexer");
    }
  }

  // insert scalar indexer
  auto reader = scan(columns);
  if (reader == nullptr) {
    return Status::InternalError("Failed to create reader");
  }

  int accu_doc_count = 0;
  while (true) {
    auto batch = reader->Next();
    if (!batch.ok()) {
      return Status::InternalError("reader next failed: ",
                                   batch.status().message());
    }

    auto batch_value = batch.ValueOrDie();

    if (!batch_value) {
      break;
    }

    s = SegmentHelper::ReduceScalarIndex(new_scalar_indexer, batch_value,
                                         accu_doc_count);
    if (!s.ok()) {
      LOG_ERROR("Reduce Scalar Index faield, err: %s", s.message().c_str());
    }
    CHECK_RETURN_STATUS(s);

    accu_doc_count += batch_value->num_rows();
  }

  s = new_scalar_indexer->seal();
  CHECK_RETURN_STATUS(s);

  BlockMeta block;
  block.set_id(block_id);
  block.set_type(BlockType::SCALAR_INDEX);
  block.set_columns(field_names);
  new_segment_meta->add_persisted_block(block);

  *segment_meta = new_segment_meta;
  *scalar_indexer = new_scalar_indexer;

  return Status::OK();
}

Status SegmentImpl::drop_scalar_index(const std::vector<std::string> &columns,
                                      SegmentMeta::Ptr *segment_meta,
                                      InvertedIndexer::Ptr *scalar_indexer) {
  // validate
  for (const auto &column : columns) {
    auto field = collection_schema_->get_field(column);
    if (!field || field->is_vector_field()) {
      return Status::InvalidArgument(
          "Invalid column name to drop scalar index");
    }
  }

  std::vector<FieldSchema> fields;
  std::vector<FieldSchema> drop_fields;
  std::vector<FieldSchema> invert_fields;
  std::vector<std::string> field_names;
  for (const auto &field : collection_schema_->forward_fields()) {
    if (field->index_type() == IndexType::INVERT) {
      invert_fields.push_back(*field);
      if (std::find(columns.begin(), columns.end(), field->name()) !=
          columns.end()) {
        drop_fields.push_back(*field);
        continue;
      }
      fields.push_back(*field);
      field_names.push_back(field->name());
    }
  }

  auto new_segment_meta = std::make_shared<SegmentMeta>(*segment_meta_);
  new_segment_meta->remove_scalar_index_block();

  if (fields.empty()) {
    *segment_meta = new_segment_meta;
    *scalar_indexer = nullptr;
    return Status::OK();
  }

  // clone original indexer
  auto block_id = allocate_block_id();
  std::string new_invert_index_path =
      FileHelper::MakeInvertIndexPath(path_, id(), block_id);
  auto s = invert_indexers_->create_snapshot(new_invert_index_path);
  CHECK_RETURN_STATUS(s);

  auto new_scalar_indexer = InvertedIndexer::CreateAndOpen(
      collection_schema_->name(), new_invert_index_path, false, invert_fields,
      options_.read_only_);
  if (!new_scalar_indexer) {
    LOG_ERROR("Failed to create scalar indexer");
    return Status::InternalError("Failed to create scalar indexer");
  }
  for (const auto &field : drop_fields) {
    s = new_scalar_indexer->remove_column_indexer(field.name());
    CHECK_RETURN_STATUS(s);
  }

  s = new_scalar_indexer->seal();
  CHECK_RETURN_STATUS(s);

  BlockMeta block;
  block.set_id(block_id);
  block.set_type(BlockType::SCALAR_INDEX);
  block.set_columns(field_names);

  new_segment_meta->add_persisted_block(block);

  *segment_meta = new_segment_meta;
  *scalar_indexer = new_scalar_indexer;

  return Status::OK();
}

Status SegmentImpl::reload_scalar_index(
    const CollectionSchema &schema, const SegmentMeta::Ptr &segment_meta,
    const InvertedIndexer::Ptr &scalar_indexer) {
  collection_schema_ = std::make_shared<CollectionSchema>(schema);
  segment_meta_ = segment_meta;

  if (!scalar_indexer) {
    // no need to reload inverted indexer
    return Status::OK();
  }

  fresh_persist_block_offset();

  if (invert_indexers_) {
    auto old_dir = invert_indexers_->working_dir();
    invert_indexers_ = scalar_indexer;

    FileHelper::RemoveDirectory(old_dir);
  } else {
    invert_indexers_ = scalar_indexer;
  }

  return Status::OK();
}

Status SegmentImpl::dump() {
  if (sealed_) {
    return Status::NotSupported("Segment has been dumped.");
  }
  auto s = flush();
  CHECK_RETURN_STATUS(s);

  if (invert_indexers_) {
    s = invert_indexers_->seal();
    CHECK_RETURN_STATUS(s);
  }

  sealed_ = true;

  return Status::OK();
}

Status SegmentImpl::flush() {
  CHECK_SEGMENT_READONLY_RETURN_STATUS;

  if (wal_file_ == nullptr || !wal_file_->has_record()) {
    return Status::OK();
  }

  if (wal_file_) {
    if (wal_file_->flush() != 0) {
      return Status::InternalError("Failed to flush wal");
    }
  }

  Status s;

  if (memory_store_) {
    s = memory_store_->flush();
    CHECK_RETURN_STATUS(s);
  }

  // flush scalar indexer
  if (invert_indexers_) {
    s = invert_indexers_->flush();
    CHECK_RETURN_STATUS(s);
  }

  // flush vector indexer
  for (const auto &indexer : memory_vector_indexers_) {
    if (indexer.second) {
      s = indexer.second->Flush();
      CHECK_RETURN_STATUS(s);
    }
  }

  // flush quant vector indexer
  for (const auto &indexer : quant_memory_vector_indexers_) {
    if (indexer.second) {
      s = indexer.second->Flush();
      CHECK_RETURN_STATUS(s);
    }
  }

  if (id_map_) {
    s = id_map_->flush();
    CHECK_RETURN_STATUS(s);
  }

  auto block = segment_meta_->writing_forward_block().value();

  uint32_t delete_snapshot_path_suffix = UINT32_MAX;
  uint32_t delete_snapshot_path_suffix_current = UINT32_MAX;
  if (delete_store_) {
    if (delete_store_->modified_since_last_flush()) {
      delete_snapshot_path_suffix_current =
          version_manager_->delete_snapshot_path_suffix();
      delete_snapshot_path_suffix =
          version_manager_->delete_snapshot_path_suffix() + 1;
      std::string delete_store_path = FileHelper::MakeFilePath(
          path_, FileID::DELETE_FILE, delete_snapshot_path_suffix);
      s = delete_store_->flush(delete_store_path);
      CHECK_RETURN_STATUS(s);
    }
  }

  if (memory_store_) {
    // update segment meta with memory components
    s = finish_memory_components();
    CHECK_RETURN_STATUS(s);

    // set a new mem block
    auto block_id = allocate_block_id();
    segment_meta_->set_writing_forward_block({block_id, BlockType::SCALAR,
                                              block.max_doc_id_ + 1, 0, 0,
                                              block.columns_});
  }

  // update version and flush
  s = update_version(delete_snapshot_path_suffix);
  CHECK_RETURN_STATUS(s);

  // clear wal file
  if (wal_file_) {
    auto ret = wal_file_->remove();
    if (ret != 0) {
      LOG_ERROR("Remove wal file failed.");
      return Status::InternalError("Remove wal file failed");
    }
    wal_file_.reset();
  }

  if (delete_snapshot_path_suffix_current != UINT32_MAX) {
    std::string delete_store_path = FileHelper::MakeFilePath(
        path_, FileID::DELETE_FILE, delete_snapshot_path_suffix_current);
    FileHelper::RemoveFile(delete_store_path);
  }

  return Status::OK();
}

Status SegmentImpl::destroy() {
  if (need_destroyed_) {
    return Status::InvalidArgument("Segment has been marked need destroyed");
  }
  need_destroyed_ = true;
  return Status::OK();
}

Status SegmentImpl::cleanup() {
  auto seg_path = FileHelper::MakeSegmentPath(path_, segment_meta_->id());
  FileHelper::RemoveDirectory(seg_path);
  return Status::OK();
}

bool SegmentImpl::validate(const std::vector<std::string> &columns) const {
  if (columns.empty()) {
    LOG_ERROR("Empty columns");
    return false;
  }
  for (const auto &column : columns) {
    if (column == LOCAL_ROW_ID || column == GLOBAL_DOC_ID ||
        column == USER_ID) {
      continue;
    }
    if (collection_schema_->get_forward_field(column) == nullptr) {
      LOG_ERROR("Validate failed. unknown column: %s", column.c_str());
      return false;
    }
  }
  return true;
}

TablePtr SegmentImpl::fetch_perf(
    const std::vector<std::string> &columns,
    const std::shared_ptr<arrow::Schema> &result_schema,
    const std::vector<int> &indices) const {
  std::vector<std::shared_ptr<arrow::ChunkedArray>> chunk_arrays;
  chunk_arrays.resize(columns.size());

  bool need_local_doc_id = false;
  size_t local_doc_id_col_index = 0;

  for (size_t i = 0; i < columns.size(); ++i) {
    if (columns[i] == LOCAL_ROW_ID) {
      need_local_doc_id = true;
      local_doc_id_col_index = i;
      chunk_arrays[i] = nullptr;
      continue;
    }
    chunk_arrays[i] = persist_chunk_arrays_[col_idx_map_.at(columns[i])];
  }

  std::vector<std::shared_ptr<arrow::Array>> result_arrays(columns.size());

  std::vector<std::pair<int64_t, int64_t>> indices_in_table;
  for (const auto &target_index : indices) {
    auto it = std::upper_bound(chunk_offsets_.begin(), chunk_offsets_.end(),
                               target_index);
    if (it == chunk_offsets_.begin()) {
      LOG_ERROR("Target index %d is out of bounds", target_index);
      return nullptr;
    }
    int chunk_index =
        static_cast<int>(std::distance(chunk_offsets_.begin(), it) - 1);
    int64_t index_in_chunk = target_index - chunk_offsets_[chunk_index];
    indices_in_table.emplace_back(chunk_index, index_in_chunk);
  }

  for (size_t i = 0; i < columns.size(); ++i) {
    if (columns[i] == LOCAL_ROW_ID) {
      continue;
    }
    const auto &source_column = chunk_arrays[i];
    std::shared_ptr<arrow::Array> array;
    auto status =
        BuildArrayFromIndicesWithType(source_column, indices_in_table, &array);
    if (!status.ok()) {
      LOG_ERROR("BuildArrayFromIndices failed: %s", status.ToString().c_str());
      return nullptr;
    }
    result_arrays[i] = array;
  }

  if (need_local_doc_id) {
    std::vector<uint64_t> values;
    values.reserve(indices.size());
    for (const auto idx : indices) {
      values.push_back(idx);
    }

    arrow::UInt64Builder builder;
    auto s = builder.AppendValues(values);
    if (!s.ok()) {
      LOG_ERROR("Failed to append values to builder: %s", s.message().c_str());
      return nullptr;
    }
    std::shared_ptr<arrow::Array> array;
    s = builder.Finish(&array);
    if (!s.ok()) {
      LOG_ERROR("Failed to finish builder: %s", s.message().c_str());
      return nullptr;
    }
    result_arrays[local_doc_id_col_index] = array;
  }

  return arrow::Table::Make(result_schema, result_arrays,
                            static_cast<int64_t>(indices.size()));
}

TablePtr SegmentImpl::fetch_normal(
    const std::vector<std::string> &columns,
    const std::shared_ptr<arrow::Schema> &result_schema,
    const std::vector<int> &indices) const {
  // Store scalars per column: column_index -> (output_row, scalar)
  std::vector<std::vector<std::pair<int, std::shared_ptr<arrow::Scalar>>>>
      column_results(columns.size());

  // Collect local_doc_id values if needed
  std::vector<std::pair<int, uint64_t>> local_doc_id_values;

  // Group fetch requests by block: block_index -> {column -> [(output_row,
  // local_row)]}
  //   block_index >= 0: persisted store
  //   block_index == -1: memory store
  std::map<int, std::map<std::string, std::vector<std::pair<int, int>>>>
      block_request_map;

  std::shared_lock<std::shared_mutex> lock(seg_col_mtx_);

  const auto &block_offsets = get_persist_block_offsets(BlockType::SCALAR);
  const auto &block_metas = get_persist_block_metas(BlockType::SCALAR);

  // Phase 1: Map each (doc_id, column) to its block and local row
  for (int output_row = 0; output_row < static_cast<int>(indices.size());
       ++output_row) {
    int doc_id = indices[output_row];

    for (size_t col_index = 0; col_index < columns.size(); ++col_index) {
      const std::string &col = columns[col_index];
      if (col == LOCAL_ROW_ID) {
        local_doc_id_values.emplace_back(output_row, doc_id);
        continue;
      }
      int offset_idx = -1;
      int block_index =
          find_persist_block_id(BlockType::SCALAR, doc_id, col, &offset_idx);

      int local_row = -1;
      if (block_index != -1 && offset_idx > -1 &&
          offset_idx < static_cast<int>(block_offsets.size())) {
        local_row = doc_id - block_offsets[offset_idx];
        block_request_map[block_index][col].emplace_back(output_row, local_row);
        continue;
      }

      // Check memory store
      if (segment_meta_->has_writing_forward_block()) {
        int mem_offset =
            block_offsets.empty()
                ? 0
                : block_offsets.back() + block_metas.back().doc_count_;
        const auto &mem_block = segment_meta_->writing_forward_block().value();

        if (mem_offset <= doc_id &&
            doc_id < mem_offset + static_cast<int>(mem_block.doc_count_)) {
          local_row = doc_id - mem_offset;
          block_request_map[-1][col].emplace_back(output_row, local_row);
          continue;
        }
      }

      LOG_ERROR("Document ID %d not found in segment %d", doc_id, meta()->id());
      return nullptr;
    }
  }

  // Phase 2: Execute batched fetch per block
  for (const auto &[block_index, col_to_rows] : block_request_map) {
    std::vector<std::string> fetch_columns;
    std::vector<int> fetch_local_rows;
    std::vector<std::pair<int, int>>
        output_to_result_index;  // (output_row, result_pos)

    fetch_columns.reserve(col_to_rows.size());
    for (const auto &kv : col_to_rows) {
      fetch_columns.push_back(kv.first);
    }

    // all column has same output size, here just take first column
    for (const auto &[output_row, local_row] :
         col_to_rows.at(fetch_columns[0])) {
      fetch_local_rows.push_back(local_row);
      output_to_result_index.emplace_back(
          output_row, static_cast<int>(fetch_local_rows.size() - 1));
    }

    std::shared_ptr<arrow::Table> block_table;
    if (block_index >= 0 &&
        block_index < static_cast<int>(persist_stores_.size())) {
      block_table =
          persist_stores_[block_index]->fetch(fetch_columns, fetch_local_rows);
    } else if (block_index == -1 && memory_store_) {
      block_table = memory_store_->fetch(fetch_columns, fetch_local_rows);
    }

    if (!block_table || block_table->num_rows() == 0) {
      continue;
    }

    // Fill results
    for (size_t i = 0; i < fetch_columns.size(); ++i) {
      const std::string &col = fetch_columns[i];
      auto col_it = std::find(columns.begin(), columns.end(), col);
      if (col_it == columns.end()) continue;
      size_t col_index = std::distance(columns.begin(), col_it);

      auto chunked_array = block_table->column(i)->chunks();
      auto flat_array_res =
          arrow::Concatenate(chunked_array, arrow::default_memory_pool());
      if (!flat_array_res.ok()) {
        LOG_ERROR("Concatenate failed: %s",
                  flat_array_res.status().message().c_str());
        return nullptr;
      }
      auto flat_array = flat_array_res.ValueOrDie();

      for (size_t j = 0; j < fetch_local_rows.size(); ++j) {
        auto scalar_result = flat_array->GetScalar(j);
        if (!scalar_result.ok()) continue;
        int output_row = output_to_result_index[j].first;
        column_results[col_index].emplace_back(
            output_row, std::move(scalar_result.ValueOrDie()));
      }
    }
  }

  // Phase 3: Construct result arrays
  std::vector<std::shared_ptr<arrow::Array>> result_arrays(columns.size());

  bool need_local_doc_id = false;
  size_t local_doc_id_col_index = -1;

  for (size_t col_index = 0; col_index < columns.size(); ++col_index) {
    const std::string &col = columns[col_index];
    if (col == LOCAL_ROW_ID) {
      need_local_doc_id = true;
      local_doc_id_col_index = col_index;
      continue;
    }

    auto &result_vec = column_results[col_index];
    std::sort(result_vec.begin(), result_vec.end());

    std::vector<std::shared_ptr<arrow::Scalar>> ordered_scalars;
    for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
      auto it = std::find_if(
          result_vec.begin(), result_vec.end(),
          [i](const std::pair<int, std::shared_ptr<arrow::Scalar>> &p) {
            return p.first == i;
          });
      if (it != result_vec.end()) {
        ordered_scalars.push_back(it->second);
      } else {
        auto field = result_schema->GetFieldByName(col);
        ordered_scalars.push_back(
            arrow::MakeNullScalar(field ? field->type() : arrow::null()));
      }
    }

    auto status = ConvertScalarVectorToArrayByType(ordered_scalars,
                                                   &result_arrays[col_index]);
    if (!status.ok()) {
      LOG_ERROR("Failed to convert scalars to array for column '%s': %s",
                col.c_str(), status.message().c_str());
      return nullptr;
    }
  }

  // Add LOCAL_ROW_ID array if requested
  if (need_local_doc_id) {
    std::sort(local_doc_id_values.begin(), local_doc_id_values.end());
    std::vector<uint64_t> values;
    values.reserve(local_doc_id_values.size());
    for (const auto &[row, id] : local_doc_id_values) {
      values.push_back(id);
    }

    arrow::UInt64Builder builder;
    auto s = builder.AppendValues(values);
    if (!s.ok()) {
      LOG_ERROR("Failed to append values to builder: %s", s.message().c_str());
      return nullptr;
    }
    std::shared_ptr<arrow::Array> array;
    s = builder.Finish(&array);
    if (!s.ok()) {
      LOG_ERROR("Failed to finish builder: %s", s.message().c_str());
      return nullptr;
    }
    result_arrays[local_doc_id_col_index] = std::move(array);
  }

  // Wrap arrays into ChunkedArray and build final table
  std::vector<std::shared_ptr<arrow::ChunkedArray>> result_columns;
  result_columns.reserve(result_arrays.size());
  for (const auto &arr : result_arrays) {
    result_columns.push_back(std::make_shared<arrow::ChunkedArray>(arr));
  }

  return arrow::Table::Make(result_schema, result_columns,
                            static_cast<int64_t>(indices.size()));
}

TablePtr SegmentImpl::fetch(const std::vector<std::string> &columns,
                            const std::vector<int> &indices) const {
  if (!validate(columns)) {
    return nullptr;
  }

  // Build result schema
  std::vector<std::shared_ptr<arrow::Field>> fields;

  for (size_t i = 0; i < columns.size(); ++i) {
    const auto &col = columns[i];
    if (col == LOCAL_ROW_ID) {
      fields.push_back(arrow::field(LOCAL_ROW_ID, arrow::uint64()));
    } else if (col == GLOBAL_DOC_ID) {
      fields.push_back(arrow::field(GLOBAL_DOC_ID, arrow::uint64()));
    } else if (col == USER_ID) {
      fields.push_back(arrow::field(USER_ID, arrow::utf8()));
    } else {
      auto *field = collection_schema_->get_field(col);
      std::shared_ptr<arrow::Field> arrow_field;
      auto status = ConvertFieldSchemaToArrowField(field, &arrow_field);
      if (!status.ok()) {
        LOG_ERROR("Convert field schema failed: %s",
                  field->to_string().c_str());
        return nullptr;
      }
      fields.push_back(std::move(arrow_field));
    }
  }

  auto result_schema = std::make_shared<arrow::Schema>(fields);

  // Early return for empty indices
  if (indices.empty()) {
    arrow::ArrayVector empty_arrays;
    for (const auto &field : fields) {
      empty_arrays.push_back(arrow::MakeEmptyArray(field->type()).ValueOrDie());
    }
    return arrow::Table::Make(result_schema, empty_arrays, 0);
  }

  if (segment_meta_->doc_count() == 0) {
    LOG_ERROR("Segment has no rows");
    return nullptr;
  }

  if (use_fetch_perf_) {
    return fetch_perf(columns, result_schema, indices);
  }
  return fetch_normal(columns, result_schema, indices);
}

ExecBatchPtr SegmentImpl::fetch(const std::vector<std::string> &columns,
                                int doc_id) const {
  if (columns.empty()) {
    LOG_ERROR("Empty columns");
    return nullptr;
  }

  std::shared_lock<std::shared_mutex> lock(seg_col_mtx_);

  const auto &block_offsets = get_persist_block_offsets(BlockType::SCALAR);
  const auto &block_metas = get_persist_block_metas(BlockType::SCALAR);

  bool is_in_single_persist_store = false;
  for (auto &block : block_metas) {
    std::vector<bool> is_column_in_block;
    is_column_in_block.reserve(columns.size());
    for (const auto &column : columns) {
      is_column_in_block.push_back(block.contain_column(column));
    }

    // Count how many columns are in this block
    int count =
        std::count(is_column_in_block.begin(), is_column_in_block.end(), true);

    if (count == 0) {
      // None of the query columns are in this block; continue to the next block
      continue;
    } else if (count == static_cast<int>(columns.size())) {
      // All query columns are present in this block; stop searching
      is_in_single_persist_store = true;
      break;
    } else {
      // Some but not all query columns are in this block (spanning multiple
      // blocks); stop searching
      break;
    }
  }

  if (is_in_single_persist_store) {
    int offset_idx = -1;
    int block_index = find_persist_block_id(BlockType::SCALAR, doc_id,
                                            columns[0], &offset_idx);
    if (block_index != -1 && offset_idx > -1 &&
        offset_idx < static_cast<int>(block_offsets.size())) {
      int local_row = doc_id - block_offsets[offset_idx];
      return persist_stores_[block_index]->fetch(columns, local_row);
    }

    // Check memory store
    if (segment_meta_->has_writing_forward_block()) {
      int mem_offset =
          block_offsets.empty()
              ? 0
              : block_offsets.back() + block_metas.back().doc_count_;
      const auto &mem_block = segment_meta_->writing_forward_block().value();

      if (mem_offset <= doc_id &&
          doc_id < mem_offset + static_cast<int>(mem_block.doc_count_)) {
        int local_row = doc_id - mem_offset;
        return memory_store_->fetch(columns, local_row);
      }
    }
  } else {
    auto table = fetch(columns, std::vector<int>{doc_id});
    if (table) {
      std::vector<arrow::Datum> datums;
      for (const auto &col : table->columns()) {
        datums.emplace_back(col->chunk(0)->GetScalar(0).ValueOrDie());
      }

      arrow::Result<arrow::compute::ExecBatch> exec_batch_result =
          arrow::compute::ExecBatch::Make(datums, table->num_rows());

      if (exec_batch_result.ok()) {
        arrow::compute::ExecBatch exec_batch = exec_batch_result.ValueOrDie();
        return std::make_shared<arrow::compute::ExecBatch>(exec_batch);
      }
    }
  }

  LOG_ERROR("Document ID %d not found in persist segment", doc_id);
  return nullptr;
}

RecordBatchReaderPtr SegmentImpl::scan(
    const std::vector<std::string> &columns) const {
  if (!validate(columns)) {
    return nullptr;
  }

  std::shared_lock<std::shared_mutex> lock(seg_col_mtx_);

  const std::vector<BlockMeta> &scalar_blocks =
      get_persist_block_metas(BlockType::SCALAR);

  std::map<std::pair<int64_t, int64_t>,
           std::vector<std::shared_ptr<arrow::ipc::RecordBatchReader>>>
      block_groups;

  for (size_t i = 0; i < scalar_blocks.size() && i < persist_stores_.size();
       ++i) {
    const auto &block = scalar_blocks[i];
    const auto &store = persist_stores_[i];

    std::vector<std::string> interested_cols;
    for (const auto &col : columns) {
      if (block.contain_column(col)) {
        interested_cols.push_back(col);
      }
    }

    if (interested_cols.empty()) {
      continue;
    }

    auto reader = store->scan(interested_cols);
    if (!reader) {
      continue;
    }

    auto key = std::make_pair(block.min_doc_id(), block.max_doc_id());
    block_groups[key].push_back(std::move(reader));
  }

  if (memory_store_ && memory_store_->num_rows() > 0) {
    auto reader = memory_store_->scan(columns);
    if (reader) {
      auto &mem_block = segment_meta_->writing_forward_block().value();
      auto key = std::make_pair(mem_block.min_doc_id(), mem_block.max_doc_id());
      block_groups[key].push_back(std::move(reader));
    }
  }

  std::vector<std::shared_ptr<arrow::Field>> fields;
  for (const auto &col : columns) {
    if (col == LOCAL_ROW_ID) {
      continue;
    } else if (col == GLOBAL_DOC_ID) {
      fields.push_back(arrow::field(GLOBAL_DOC_ID, arrow::uint64(), false));
    } else if (col == USER_ID) {
      fields.push_back(arrow::field(USER_ID, arrow::utf8(), false));
    } else {
      auto *field = collection_schema_->get_field(col);
      std::shared_ptr<arrow::Field> arrow_field;
      auto s = ConvertFieldSchemaToArrowField(field, &arrow_field);
      if (!s.ok()) {
        LOG_ERROR("convert field schema: %s to arrow field failed",
                  field->to_string().c_str());
        return nullptr;
      }
      fields.push_back(arrow_field);
    }
  }
  auto target_schema = std::make_shared<arrow::Schema>(fields);

  std::vector<std::shared_ptr<arrow::ipc::RecordBatchReader>> merged_readers;
  for (auto &kv : block_groups) {
    auto &&readers = std::move(kv.second);
    auto merging_reader =
        ColumnMergingReader::Make(target_schema, std::move(readers));
    if (merging_reader) {
      merged_readers.push_back(std::move(merging_reader));
    }
  }

  return std::make_shared<CombinedRecordBatchReader>(
      shared_from_this(), std::move(merged_readers), columns);
}


////////////////////////////////////////////////////////////////////////////////////
// CombinedRecordBatchReader implementation
////////////////////////////////////////////////////////////////////////////////////

SegmentImpl::CombinedRecordBatchReader::CombinedRecordBatchReader(
    std::shared_ptr<const SegmentImpl> segment,
    std::vector<std::shared_ptr<arrow::RecordBatchReader>> readers,
    const std::vector<std::string> &columns)
    : segment_(segment),
      readers_(std::move(readers)),
      current_reader_index_(0),
      local_doc_id_(0) {
  if (!readers_.empty()) {
    auto schema = readers_[0]->schema();
    std::vector<std::shared_ptr<arrow::Field>> selected_fields;
    for (size_t i = 0; i < columns.size(); ++i) {
      auto &col_name = columns[i];
      if (col_name == LOCAL_ROW_ID) {
        selected_fields.push_back(
            arrow::field(LOCAL_ROW_ID, arrow::uint64(), false));
        need_local_doc_id_ = true;
        local_doc_id_col_index_ = static_cast<int>(i);
      } else {
        if (auto field = schema->GetFieldByName(col_name); field) {
          selected_fields.push_back(field);
        }
      }
    }

    projected_schema_ = arrow::schema(selected_fields);

    auto segment_meta = segment_->meta();
    const auto &blocks = segment_meta->persisted_blocks();
    for (const auto &block : blocks) {
      if (block.type() != BlockType::SCALAR) continue;
      offsets_.push_back(block.min_doc_id_);
    }
    if (segment_meta->has_writing_forward_block()) {
      const auto &mem_block = segment_meta->writing_forward_block().value();
      offsets_.push_back(mem_block.min_doc_id_);
    }
  }
}

SegmentImpl::CombinedRecordBatchReader::~CombinedRecordBatchReader() {}

std::shared_ptr<arrow::Schema> SegmentImpl::CombinedRecordBatchReader::schema()
    const {
  return projected_schema_;
}

arrow::Status SegmentImpl::CombinedRecordBatchReader::ReadNext(
    std::shared_ptr<arrow::RecordBatch> *batch) {
  *batch = nullptr;
  while (current_reader_index_ < readers_.size()) {
    auto status = readers_[current_reader_index_]->ReadNext(batch);
    if (!status.ok()) {
      return status;
    }

    if (need_local_doc_id_ && *batch) {
      auto num_rows = (*batch)->num_rows();
      arrow::UInt64Builder builder;
      ARROW_RETURN_NOT_OK(builder.Reserve(num_rows));

      for (int64_t i = 0; i < num_rows; ++i) {
        builder.UnsafeAppend(local_doc_id_++);
      }
      std::shared_ptr<arrow::Array> local_id_array;
      ARROW_RETURN_NOT_OK(builder.Finish(&local_id_array));

      auto result =
          (*batch)->AddColumn(local_doc_id_col_index_,
                              projected_schema_->GetFieldByName(LOCAL_ROW_ID),
                              std::move(local_id_array));
      if (result.ok()) {
        *batch = std::move(result.ValueOrDie());
      }
    }

    if (*batch) {
      return arrow::Status::OK();
    }

    current_reader_index_++;
    if (current_reader_index_ < readers_.size()) {
      local_doc_id_ = offsets_[current_reader_index_];
    }
  }

  *batch = nullptr;
  return arrow::Status::OK();
}

bool SegmentImpl::ready_for_dump_block() {
  if (memory_store_) return memory_store_->is_full();
  return false;
}

template <typename ArrayType, typename ValueType>
Status ProcessChunkData(InvertedColumnIndexer::Ptr *column_indexer,
                        const std::shared_ptr<arrow::Array> &chunk,
                        int64_t &doc_count) {
  auto typed_array = std::dynamic_pointer_cast<ArrayType>(chunk);
  if (typed_array) {
    for (int64_t i = 0; i < typed_array->length(); ++i, ++doc_count) {
      if (typed_array->IsNull(i)) {
        auto status = (*column_indexer)->insert_null(doc_count);
        if (!status.ok()) {
          LOG_ERROR("Failed to insert null value to indexer for doc %zu: %s",
                    (size_t)doc_count, status.message().c_str());
          return status;
        }
      } else {
        ValueType value = typed_array->Value(i);
        std::string value_str(reinterpret_cast<const char *>(&value),
                              sizeof(ValueType));
        auto status = (*column_indexer)->insert(doc_count, value_str);
        if (!status.ok()) {
          LOG_ERROR("Failed to insert numeric value to indexer for doc %zu: %s",
                    (size_t)doc_count, status.message().c_str());
          return status;
        }
      }
    }
  }
  return Status::OK();
}

Status SegmentImpl::insert_array_to_invert_indexer(
    const FieldSchema::Ptr &column_schema,
    const std::shared_ptr<arrow::ChunkedArray> &new_column,
    InvertedColumnIndexer::Ptr *column_indexer) {
  // Iterate through the new column data and insert into the indexer
  int64_t doc_count = 0;
  for (int chunk_index = 0; chunk_index < new_column->num_chunks();
       ++chunk_index) {
    auto chunk = new_column->chunk(chunk_index);

    // Handle different data types based on the column schema
    switch (column_schema->data_type()) {
      case DataType::INT32: {
        auto status = ProcessChunkData<arrow::Int32Array, int32_t>(
            column_indexer, chunk, doc_count);
        CHECK_RETURN_STATUS(status);
        break;
      }
      case DataType::INT64: {
        auto status = ProcessChunkData<arrow::Int64Array, int64_t>(
            column_indexer, chunk, doc_count);
        CHECK_RETURN_STATUS(status);
        break;
      }
      case DataType::UINT32: {
        auto status = ProcessChunkData<arrow::UInt32Array, uint32_t>(
            column_indexer, chunk, doc_count);
        CHECK_RETURN_STATUS(status);
        break;
      }
      case DataType::UINT64: {
        auto status = ProcessChunkData<arrow::UInt64Array, uint64_t>(
            column_indexer, chunk, doc_count);
        CHECK_RETURN_STATUS(status);
        break;
      }
      case DataType::FLOAT: {
        auto status = ProcessChunkData<arrow::FloatArray, float>(
            column_indexer, chunk, doc_count);
        CHECK_RETURN_STATUS(status);
        break;
      }
      case DataType::DOUBLE: {
        auto status = ProcessChunkData<arrow::DoubleArray, double>(
            column_indexer, chunk, doc_count);
        CHECK_RETURN_STATUS(status);
        break;
      }
      default:
        LOG_WARN(
            "Unsupported data type for indexing: %s",
            DataTypeCodeBook::AsString(column_schema->data_type()).c_str());
        break;
    }
  }

  return Status::OK();
}


Status SegmentImpl::reopen_invert_indexer(bool read_only) {
  // build invert index path
  uint32_t block_id = 0;
  auto &persist_blocks = segment_meta_->persisted_blocks();
  for (auto &block : persist_blocks) {
    if (block.type() == BlockType::SCALAR_INDEX) {
      block_id = block.id();
      break;
    }
  }
  std::string invert_index_path =
      FileHelper::MakeInvertIndexPath(path_, id(), block_id);

  // build invert index fields
  std::vector<std::string> inverted_field_names;
  auto inverted_fields_ptr = collection_schema_->forward_fields_with_index();
  std::vector<FieldSchema> inverted_fields;
  for (auto field : inverted_fields_ptr) {
    inverted_fields.push_back(*field);
    inverted_field_names.push_back(field->name());
  }

  // reopen invert indexer with read_only false
  invert_indexers_.reset();
  invert_indexers_ = InvertedIndexer::CreateAndOpen(collection_schema_->name(),
                                                    invert_index_path, false,
                                                    inverted_fields, read_only);
  if (!invert_indexers_) {
    LOG_ERROR("Failed to create scalar indexer");
    return Status::InternalError("Failed to create scalar indexer");
  }
  return Status::OK();
}

Status SegmentImpl::add_column(FieldSchema::Ptr column_schema,
                               const std::string &expression,
                               const AddColumnOptions & /*options*/) {
  if (memory_store_) {
    return Status::NotSupported(
        "Add column is not supported for segment with memory store");
  }

  global_init();

  std::vector<std::shared_ptr<arrow::Field>> fields;
  arrow::Status status =
      ConvertCollectionSchemaToArrowFields(collection_schema_, &fields);
  if (!status.ok()) {
    return Status::InvalidArgument(
        "ConvertCollectionSchemaToArrowFields failed:", status.message());
  }
  auto physic_schema = std::make_shared<arrow::Schema>(fields);

  auto &scalar_blocks = get_persist_block_metas(BlockType::SCALAR);
  if (scalar_blocks.empty()) {
    return Status::NotSupported(
        "Add column is not supported for empty scalar segment");
  }

  std::shared_ptr<arrow::Field> arrow_field;
  status = ConvertFieldSchemaToArrowField(column_schema.get(), &arrow_field);
  if (!status.ok()) {
    return Status::InvalidArgument("ConvertFieldSchemaToArrowField failed:",
                                   status.message());
  }

  std::shared_ptr<arrow::ChunkedArray> new_column;
  auto expected_type = arrow_field->type();
  if (expression.empty()) {
    if (!column_schema->nullable()) {
      return Status::InvalidArgument(
          "Add column is not supported for non-nullable column");
    }
    arrow::Result<std::shared_ptr<arrow::Array>> result =
        arrow::MakeArrayOfNull(expected_type, scalar_blocks[0].doc_count_);
    if (!result.ok()) {
      return Status::InternalError("MakeArrayOfNull failed");
    }
    auto array = result.ValueOrDie();
    new_column = std::make_shared<arrow::ChunkedArray>(
        std::vector<std::shared_ptr<arrow::Array>>{array});

  } else {
    // Parse Simple sql expression
    auto p_result = ParseToExpression(expression, physic_schema);
    if (!p_result.ok()) {
      return Status::InvalidArgument("parse expression failed:",
                                     p_result.status().message());
    }
    auto expr = p_result.ValueOrDie();

    auto result = ReadBlocksAsDataset(scalar_blocks, path_, segment_meta_->id(),
                                      !options_.enable_mmap_);
    if (!result.ok()) {
      return Status::InternalError(result.status().message());
    }
    auto dataset = std::move(result).ValueOrDie();
    auto eval_result = EvaluateExpressionWithDataset(
        dataset, column_schema->name(), expr, expected_type);
    if (!eval_result.ok()) {
      return Status::InternalError("evaluate expression failed:",
                                   eval_result.status().message());
    }
    auto result_table = eval_result.ValueOrDie();
    if (result_table->num_columns() != 1) {
      return Status::InvalidArgument(
          "Expression result must have exactly one column");
    }
    new_column = result_table->column(0);
  }

  // write new column
  const std::string &filter_column = scalar_blocks[0].columns()[0];
  std::vector<BlockMeta> filter_column_blocks;
  std::copy_if(scalar_blocks.begin(), scalar_blocks.end(),
               std::back_inserter(filter_column_blocks),
               [&filter_column](const BlockMeta &block) {
                 return block.contain_column(filter_column);
               });

  std::vector<BlockMeta> new_blocks;
  status = WriteColumnInBlocks(
      column_schema->name(), new_column, filter_column_blocks, path_,
      segment_meta_->id(), [this]() { return allocate_block_id(); },
      !options_.enable_mmap_, &new_blocks);
  if (!status.ok()) {
    return Status::InternalError(status.message());
  }

  // create persist scalar indexer
  if (column_schema->has_invert_index()) {
    if (invert_indexers_) {
      auto s = reopen_invert_indexer();
      CHECK_RETURN_STATUS(s);

      s = invert_indexers_->create_column_indexer(*column_schema);
      CHECK_RETURN_STATUS(s);

      // update segment meta
      auto &persist_blocks = segment_meta_->persisted_blocks();
      for (auto &block : persist_blocks) {
        if (block.type() == BlockType::SCALAR_INDEX) {
          block.add_column(column_schema->name());
          break;
        }
      }
    } else {
      auto new_block_id = allocate_block_id();
      std::string new_invert_index_path =
          FileHelper::MakeInvertIndexPath(path_, id(), new_block_id);

      invert_indexers_ = InvertedIndexer::CreateAndOpen(
          collection_schema_->name(), new_invert_index_path, true,
          {*column_schema}, false);
      if (!invert_indexers_) {
        LOG_ERROR("Failed to create scalar indexer");
        return Status::InternalError("Failed to create scalar indexer");
      }

      // update segment meta
      BlockMeta block;
      block.set_id(new_block_id);
      block.set_type(BlockType::SCALAR_INDEX);
      block.set_doc_count(new_column->length());
      block.set_min_doc_id(doc_ids_.front());
      block.set_max_doc_id(doc_ids_.back());
      block.set_columns({column_schema->name()});

      segment_meta_->add_persisted_block(block);
    }

    auto column_indexer = (*invert_indexers_)[column_schema->name()];
    auto s = insert_array_to_invert_indexer(column_schema, new_column,
                                            &column_indexer);
    CHECK_RETURN_STATUS(s);
    column_indexer->seal();
    invert_indexers_->flush();
  }

  std::unique_lock<std::shared_mutex> lock(seg_col_mtx_);
  // create and append persist scalar indexer
  for (auto &block : new_blocks) {
    auto forward_path = FileHelper::MakeForwardBlockPath(
        path_, segment_meta_->id(), block.id_, !options_.enable_mmap_);

    BaseForwardStore::Ptr forward_store;
    if (options_.enable_mmap_) {
      forward_store = std::make_shared<MmapForwardStore>(forward_path);
    } else {
      forward_store = std::make_shared<BufferPoolForwardStore>(forward_path);
    }
    auto s = forward_store->Open();
    CHECK_RETURN_STATUS(s);
    persist_stores_.push_back(forward_store);
    segment_meta_->add_persisted_block(block);
  }

  // collection_schema append new field
  auto s = collection_schema_->add_field(column_schema);
  CHECK_RETURN_STATUS(s);

  fresh_persist_block_offset();

  fresh_persist_chunked_array();

  return Status::OK();
}


Status SegmentImpl::alter_column(const std::string &column_name,
                                 const FieldSchema::Ptr &new_column_schema,
                                 const AlterColumnOptions & /*options*/) {
  if (memory_store_) {
    return Status::NotSupported(
        "Add column is not supported for segment with memory store");
  }

  global_init();

  auto old_field_schema = collection_schema_->get_forward_field(column_name);
  if (!old_field_schema) {
    return Status::NotFound("Column not found: " + column_name);
  }

  std::string new_column_name = new_column_schema->name();
  std::shared_ptr<arrow::Field> new_arrow_field;
  auto as =
      ConvertFieldSchemaToArrowField(new_column_schema.get(), &new_arrow_field);
  if (!as.ok()) {
    return Status::InternalError("ConvertFieldSchemaToArrowField failed: " +
                                 as.ToString());
  }

  auto &scalar_blocks = get_persist_block_metas(BlockType::SCALAR);
  if (scalar_blocks.empty()) {
    return Status::NotSupported(
        "Add column is not supported for empty scalar segment");
  }

  std::vector<BlockMeta> filter_column_blocks;
  for (const auto &block : scalar_blocks) {
    if (block.contain_column(column_name)) {
      filter_column_blocks.push_back(block);
    }
  }

  auto result = ReadBlocksAsDataset(
      filter_column_blocks, path_, segment_meta_->id(), !options_.enable_mmap_);
  if (!result.ok()) {
    return Status::InternalError(result.status().message());
  }
  auto dataset = std::move(result).ValueOrDie();

  arrow::Expression expr = arrow::compute::field_ref(old_field_schema->name());
  auto eval_result = EvaluateExpressionWithDataset(
      dataset, new_column_name, expr, new_arrow_field->type());
  if (!eval_result.ok()) {
    return Status::InternalError("evaluate expression failed:",
                                 eval_result.status().message());
  }
  auto result_table = eval_result.ValueOrDie();
  if (result_table->num_columns() != 1) {
    return Status::InvalidArgument(
        "Expression result must have exactly one column");
  }
  auto new_column = result_table->column(0);

  std::vector<BlockMeta> new_blocks;
  auto status = WriteColumnInBlocks(
      new_column_name, new_column, filter_column_blocks, path_,
      segment_meta_->id(), [this]() { return allocate_block_id(); },
      !options_.enable_mmap_, &new_blocks);
  if (!status.ok()) {
    return Status::InternalError(status.message());
  }

  if (new_column_schema->has_invert_index()) {
    if (invert_indexers_) {
      auto s = reopen_invert_indexer();
      CHECK_RETURN_STATUS(s);

      s = invert_indexers_->remove_column_indexer(column_name);
      CHECK_RETURN_STATUS(s);

      s = invert_indexers_->create_column_indexer(*new_column_schema);
      CHECK_RETURN_STATUS(s);

      // update segment meta
      auto &persist_blocks = segment_meta_->persisted_blocks();
      for (auto &block : persist_blocks) {
        if (block.type() == BlockType::SCALAR_INDEX) {
          block.del_column(old_field_schema->name());
          block.add_column(new_column_schema->name());
          break;
        }
      }
    } else {
      auto new_block_id = allocate_block_id();
      std::string new_invert_index_path =
          FileHelper::MakeInvertIndexPath(path_, id(), new_block_id);

      invert_indexers_ = InvertedIndexer::CreateAndOpen(
          collection_schema_->name(), new_invert_index_path, true,
          {*new_column_schema}, false);
      if (!invert_indexers_) {
        LOG_ERROR("Failed to create scalar indexer");
        return Status::InternalError("Failed to create scalar indexer");
      }

      // update segment meta
      BlockMeta block;
      block.set_id(new_block_id);
      block.set_type(BlockType::SCALAR_INDEX);
      block.set_doc_count(new_column->length());
      block.set_min_doc_id(doc_ids_.front());
      block.set_max_doc_id(doc_ids_.back());
      block.set_columns({new_column_schema->name()});

      segment_meta_->add_persisted_block(block);
    }

    // insert data into new invert indexer
    auto column_indexer = (*invert_indexers_)[new_column_schema->name()];
    auto s = insert_array_to_invert_indexer(new_column_schema, new_column,
                                            &column_indexer);
    CHECK_RETURN_STATUS(s);
    column_indexer->seal();
    invert_indexers_->flush();
  } else if (old_field_schema->has_invert_index()) {
    // drop old invert indexer
    auto s = reopen_invert_indexer();
    CHECK_RETURN_STATUS(s);

    s = invert_indexers_->remove_column_indexer(column_name);
    CHECK_RETURN_STATUS(s);

    auto &persist_blocks = segment_meta_->persisted_blocks();
    for (auto &block : persist_blocks) {
      if (block.type() == BlockType::SCALAR_INDEX) {
        block.del_column(old_field_schema->name());
        if (block.columns().empty()) {
          segment_meta_->remove_block(block.id());
        }
        break;
      }
    }
  }

  std::unique_lock<std::shared_mutex> lock(seg_col_mtx_);
  // update old block, remove column
  std::vector<BlockMeta> &persisted_blocks = segment_meta_->persisted_blocks();
  std::vector<int> will_del_block_idx;
  for (size_t idx = 0; idx < persisted_blocks.size(); idx++) {
    auto &block = persisted_blocks[idx];
    if (block.type() == BlockType::SCALAR) {
      if (block.contain_column(column_name)) {
        if (block.columns_.size() > 1) {
          block.del_column(column_name);
        } else {
          will_del_block_idx.push_back(idx);
        }
      }
    }
  }

  // delete single block
  std::vector<int> will_del_block_ids;
  for (int i = static_cast<int>(will_del_block_idx.size()) - 1; i >= 0; i--) {
    int idx = will_del_block_idx[i];
    auto &block = persisted_blocks[idx];
    will_del_block_ids.push_back(block.id_);
    persisted_blocks.erase(persisted_blocks.begin() + idx);
  }

  std::vector<int> will_del_local_block_idx;
  auto &local_blocks = get_persist_block_metas(BlockType::SCALAR);
  for (size_t idx = 0; idx < local_blocks.size(); idx++) {
    auto &block = local_blocks[idx];
    if (block.contain_column(column_name)) {
      if (block.columns_.size() == 1) {
        will_del_local_block_idx.push_back(idx);
      }
    }
  }

  for (int idx = static_cast<int>(will_del_local_block_idx.size()) - 1;
       idx >= 0; idx--) {
    int local_idx = will_del_local_block_idx[idx];
    persist_stores_.erase(persist_stores_.begin() + local_idx);
  }

  if (!options_.enable_mmap_) {
    ailego::BufferManager::Instance().init(
        GlobalConfig::Instance().memory_limit_bytes(), 1);
  }

  // delete single column store file
  for (auto block_id : will_del_block_ids) {
    // delete forward store file
    std::string filepath = FileHelper::MakeForwardBlockPath(
        path_, meta()->id(), block_id, !options_.enable_mmap_);
    if (!FileHelper::RemoveFile(filepath)) {
      return Status::InternalError("remove ", filepath, " failed");
    } else {
      LOG_INFO("remove scalar store file: %s success", filepath.c_str());
    }
  }

  // create and append persist scalar indexer
  for (auto &block : new_blocks) {
    auto forward_path = FileHelper::MakeForwardBlockPath(
        path_, segment_meta_->id(), block.id_, !options_.enable_mmap_);

    BaseForwardStore::Ptr forward_store;
    if (options_.enable_mmap_) {
      forward_store = std::make_shared<MmapForwardStore>(forward_path);
    } else {
      forward_store = std::make_shared<BufferPoolForwardStore>(forward_path);
    }
    auto s = forward_store->Open();
    CHECK_RETURN_STATUS(s);
    persist_stores_.push_back(forward_store);
    segment_meta_->add_persisted_block(block);
  }

  // collection_schema append new field
  auto alter_status =
      collection_schema_->alter_field(column_name, new_column_schema);
  CHECK_RETURN_STATUS(alter_status);

  fresh_persist_block_offset();

  fresh_persist_chunked_array();

  return Status::OK();
}

Status SegmentImpl::drop_column(const std::string &column_name) {
  if (memory_store_) {
    return Status::NotSupported(
        "Add column is not supported for segment with memory store");
  }

  std::unique_lock<std::shared_mutex> lock(seg_col_mtx_);
  // update old block, remove column
  std::vector<BlockMeta> &persisted_blocks = segment_meta_->persisted_blocks();
  std::vector<int> will_del_block_idx;
  for (size_t idx = 0; idx < persisted_blocks.size(); idx++) {
    auto &block = persisted_blocks[idx];
    if (block.type() == BlockType::SCALAR) {
      if (block.contain_column(column_name)) {
        if (block.columns_.size() > 1) {
          block.del_column(column_name);
        } else {
          will_del_block_idx.push_back(idx);
        }
      }
    }
  }

  // delete single block
  std::vector<int> will_del_block_ids;
  for (int i = static_cast<int>(will_del_block_idx.size()) - 1; i >= 0; i--) {
    int idx = will_del_block_idx[i];
    auto &block = persisted_blocks[idx];
    will_del_block_ids.push_back(block.id_);
    persisted_blocks.erase(persisted_blocks.begin() + idx);
  }

  std::vector<int> will_del_local_block_idx;
  auto &local_blocks = get_persist_block_metas(BlockType::SCALAR);
  for (size_t idx = 0; idx < local_blocks.size(); idx++) {
    auto &block = local_blocks[idx];
    if (block.contain_column(column_name)) {
      if (block.columns_.size() == 1) {
        will_del_local_block_idx.push_back(idx);
      }
    }
  }

  for (int idx = static_cast<int>(will_del_local_block_idx.size()) - 1;
       idx >= 0; idx--) {
    int local_idx = will_del_local_block_idx[idx];
    persist_stores_.erase(persist_stores_.begin() + local_idx);
  }

  if (!options_.enable_mmap_) {
    ailego::BufferManager::Instance().init(
        GlobalConfig::Instance().memory_limit_bytes(), 1);
  }

  // delete single column store file
  for (auto block_id : will_del_block_ids) {
    // delete forward store file
    std::string filepath = FileHelper::MakeForwardBlockPath(
        path_, meta()->id(), block_id, !options_.enable_mmap_);
    if (!FileHelper::RemoveFile(filepath)) {
      return Status::InternalError("remove ", filepath, " failed");
    } else {
      LOG_INFO("remove scalar store file: %s success", filepath.c_str());
    }
  }

  auto old_field_schema = collection_schema_->get_forward_field(column_name);
  if (old_field_schema->has_invert_index()) {
    auto s = reopen_invert_indexer();
    CHECK_RETURN_STATUS(s);

    s = invert_indexers_->remove_column_indexer(old_field_schema->name());
    CHECK_RETURN_STATUS(s);
    invert_indexers_->flush();

    auto &persist_blocks = segment_meta_->persisted_blocks();
    for (auto &block : persist_blocks) {
      if (block.type() == BlockType::SCALAR_INDEX) {
        block.del_column(old_field_schema->name());
        if (block.columns_.empty()) {
          // remove block meta from segment meta
          segment_meta_->remove_block(block.id_);
        }
        break;
      }
    }
  }

  // collection_schema append new field
  auto alter_status = collection_schema_->drop_field(column_name);
  CHECK_RETURN_STATUS(alter_status);

  fresh_persist_block_offset();

  fresh_persist_chunked_array();

  return Status::OK();
}

////////////////////////////////////////////////////////////////////////////////////
// Private methods implementation
////////////////////////////////////////////////////////////////////////////////////


void SegmentImpl::fresh_persist_block_offset() {
  // Clear
  for (size_t i = 0; i <= static_cast<size_t>(BlockType::VECTOR_INDEX_QUANTIZE);
       ++i) {
    if (std::holds_alternative<std::vector<int>>(persist_block_offsets_[i])) {
      std::get<std::vector<int>>(persist_block_offsets_[i]).clear();
    } else if (std::holds_alternative<
                   std::unordered_map<std::string, std::vector<int>>>(
                   persist_block_offsets_[i])) {
      std::get<std::unordered_map<std::string, std::vector<int>>>(
          persist_block_offsets_[i])
          .clear();
    }
    std::visit(
        [](auto &&arg) {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, std::vector<BlockMeta>> ||
                        std::is_same_v<
                            T, std::unordered_map<std::string,
                                                  std::vector<BlockMeta>>>) {
            arg.clear();
          }
        },
        persist_block_metas_[i]);
  }

  for (const auto &block : segment_meta_->persisted_blocks()) {
    size_t type_index = static_cast<size_t>(block.type());
    if (block.type() == BlockType::SCALAR) {
      if (std::holds_alternative<std::vector<BlockMeta>>(
              persist_block_metas_[type_index])) {
        std::get<std::vector<BlockMeta>>(persist_block_metas_[type_index])
            .push_back(block);
      } else {
        persist_block_metas_[type_index] = std::vector<BlockMeta>{block};
        persist_block_offsets_[type_index] = std::vector<int>();
      }
    } else if (block.type() == BlockType::VECTOR_INDEX ||
               block.type() == BlockType::VECTOR_INDEX_QUANTIZE) {
      if (block.columns().size() == 1) {
        auto column_name = block.columns()[0];
        if (std::holds_alternative<
                std::unordered_map<std::string, std::vector<BlockMeta>>>(
                persist_block_metas_[type_index])) {
          auto block_map =
              std::get<std::unordered_map<std::string, std::vector<BlockMeta>>>(
                  persist_block_metas_[type_index]);
          auto iter = block_map.find(column_name);
          if (iter != block_map.end()) {
            auto &block_metas = iter->second;
            block_metas.push_back(block);
          } else {
            block_map.insert(
                std::make_pair(column_name, std::vector<BlockMeta>{block}));
            auto block_offsets_map =
                std::get<std::unordered_map<std::string, std::vector<int>>>(
                    persist_block_offsets_[type_index]);
            block_offsets_map.insert(
                std::make_pair(column_name, std::vector<int>()));
          }

          std::get<std::unordered_map<std::string, std::vector<BlockMeta>>>(
              persist_block_metas_[type_index])[column_name]
              .push_back(block);
        } else {
          std::unordered_map<std::string, std::vector<BlockMeta>> new_map;
          new_map[column_name].push_back(block);
          persist_block_metas_[type_index] = std::move(new_map);
          persist_block_offsets_[type_index] =
              std::unordered_map<std::string, std::vector<int>>();
        }
      } else {
        LOG_ERROR("Add block meta: %s failed, block.columns.size != 1",
                  block.to_string().c_str());
      }
    }
  }

  calculate_block_offsets();
}

void SegmentImpl::fresh_persist_chunked_array() {
  if (options_.enable_mmap_ && options_.read_only_) {
    persist_chunk_arrays_.clear();
    chunk_offsets_.clear();
    col_idx_map_.clear();
    use_fetch_perf_ = false;

    std::vector<std::vector<std::shared_ptr<arrow::ChunkedArray>>> chunk_arrays;
    auto fields = collection_schema_->forward_field_names();
    fields.insert(fields.begin(), USER_ID);
    fields.insert(fields.begin(), GLOBAL_DOC_ID);
    chunk_arrays.resize(fields.size());
    persist_chunk_arrays_.resize(fields.size());

    for (size_t i = 0; i < fields.size(); ++i) {
      col_idx_map_[fields[i]] = i;
    }

    auto &block_metas = get_persist_block_metas(BlockType::SCALAR);
    if (block_metas.empty()) {
      return;
    }

    for (size_t i = 0; i < block_metas.size(); ++i) {
      auto &block_meta = block_metas[i];
      const auto table = persist_stores_[i]->get_table();
      for (size_t j = 0; j < fields.size(); ++j) {
        if (block_meta.contain_column(fields[j])) {
          auto chunked_array = table->GetColumnByName(fields[j]);
          if (chunked_array) {
            chunk_arrays[j].push_back(chunked_array);
          }
        }
      }
    }

    for (size_t i = 0; i < fields.size(); ++i) {
      std::vector<std::shared_ptr<arrow::Array>> all_chunks;
      for (const auto &arr : chunk_arrays[i]) {
        for (int j = 0; j < arr->num_chunks(); ++j) {
          all_chunks.push_back(arr->chunk(j));
        }
      }
      persist_chunk_arrays_[i] =
          std::make_shared<arrow::ChunkedArray>(all_chunks);
    }

    auto &first_chunked_array = persist_chunk_arrays_[0];
    chunk_offsets_.reserve(first_chunked_array->num_chunks() + 1);
    chunk_offsets_.push_back(0);

    for (int chunk_idx = 0; chunk_idx < first_chunked_array->num_chunks();
         ++chunk_idx) {
      chunk_offsets_.push_back(chunk_offsets_.back() +
                               first_chunked_array->chunk(chunk_idx)->length());
    }

    if (persist_chunk_arrays_.size() > 0 && chunk_offsets_.size() > 0) {
      use_fetch_perf_ = true;
    }

    LOG_INFO(
        "fresh_persist_chunked_array persist_chunk_arrays[%zu] "
        "chunk_offset[%zu]",
        persist_chunk_arrays_.size(), chunk_offsets_.size());
  }
}

void SegmentImpl::calculate_block_offsets() {
  for (size_t type_index = 0;
       type_index <= static_cast<size_t>(BlockType::VECTOR_INDEX_QUANTIZE);
       ++type_index) {
    auto &block_offsets = persist_block_offsets_[type_index];
    int current_offset = 0;

    // Visit the appropriate container based on the variant type
    std::visit(
        [&current_offset, &block_offsets](auto &&blocks) {
          using T = std::decay_t<decltype(blocks)>;

          if constexpr (std::is_same_v<T, std::vector<BlockMeta>>) {
            // For SCALAR type - simple vector
            auto &offset_vector = std::get<std::vector<int>>(block_offsets);
            offset_vector.clear();
            offset_vector.reserve(blocks.size());
            if (!blocks.empty()) {
              auto &filter_col_name = blocks[0].columns()[0];
              for (const auto &block : blocks) {
                if (!block.contain_column(filter_col_name)) continue;
                offset_vector.push_back(current_offset);
                current_offset += static_cast<int>(block.doc_count_);
              }
            }
          } else if constexpr (std::is_same_v<T, std::unordered_map<
                                                     std::string,
                                                     std::vector<BlockMeta>>>) {
            // For other types - map with column names
            auto &offset_map =
                std::get<std::unordered_map<std::string, std::vector<int>>>(
                    block_offsets);
            offset_map.clear();

            for (const auto &[column_name, block_list] : blocks) {
              auto &column_offsets = offset_map[column_name];
              column_offsets.reserve(block_list.size());
              int column_offset = 0;

              for (const auto &block : block_list) {
                column_offsets.push_back(column_offset);
                column_offset += static_cast<int>(block.doc_count_);
              }
            }
          }
        },
        persist_block_metas_[type_index]);
  }
}

int SegmentImpl::find_persist_block_id(BlockType type, int segment_doc_id,
                                       const std::string &col_name,
                                       int *out_offset_idx) const {
  size_t type_index = static_cast<size_t>(type);

  auto visitor = [segment_doc_id, col_name,
                  out_offset_idx](const auto &blocks) -> int {
    using T = std::decay_t<decltype(blocks)>;
    int current_offset = 0;

    if constexpr (std::is_same_v<T, std::vector<BlockMeta>>) {
      if (!blocks.empty()) {
        std::string filter_column = col_name;
        if (col_name.empty() || col_name == GLOBAL_DOC_ID ||
            col_name == USER_ID) {
          filter_column = blocks[0].columns()[0];
        }
        int offset_idx = -1;
        for (size_t block_idx = 0; block_idx < blocks.size(); block_idx++) {
          const auto &block = blocks[block_idx];
          if (!block.contain_column(filter_column)) {
            continue;
          }
          offset_idx++;
          if (segment_doc_id >= current_offset &&
              segment_doc_id <
                  current_offset + static_cast<int>(block.doc_count_)) {
            if (out_offset_idx) {
              *out_offset_idx = offset_idx;
            }
            return static_cast<int>(block_idx);
          }
          current_offset += static_cast<int>(block.doc_count_);
        }
      }
    } else if constexpr (std::is_same_v<
                             T, std::unordered_map<std::string,
                                                   std::vector<BlockMeta>>>) {
      for (const auto &[column_name, block_list] : blocks) {
        if (!column_name.empty() && column_name != col_name) {
          continue;
        }

        current_offset = 0;
        for (size_t block_idx = 0; block_idx < block_list.size(); block_idx++) {
          const auto &block = block_list[block_idx];
          if (segment_doc_id >= current_offset &&
              segment_doc_id <
                  current_offset + static_cast<int>(block.doc_count_)) {
            return static_cast<int>(block_idx);
          }
          current_offset += static_cast<int>(block.doc_count_);
        }
      }
    }

    return -1;
  };

  return std::visit(visitor, persist_block_metas_[type_index]);
}

const std::vector<int> &SegmentImpl::get_persist_block_offsets(
    BlockType type, const std::string &col_name) const {
  size_t type_index = static_cast<size_t>(type);

  auto visitor = [&col_name](const auto &offsets) -> const std::vector<int> & {
    using T = std::decay_t<decltype(offsets)>;

    static const std::vector<int> empty_offsets;

    if constexpr (std::is_same_v<T, std::vector<int>>) {
      return offsets;
    } else if constexpr (std::is_same_v<T,
                                        std::unordered_map<std::string,
                                                           std::vector<int>>>) {
      auto it = offsets.find(col_name);
      if (it != offsets.end()) {
        return it->second;
      }
    }

    return empty_offsets;
  };

  return std::visit(visitor, persist_block_offsets_[type_index]);
}

const std::vector<BlockMeta> &SegmentImpl::get_persist_block_metas(
    BlockType type, const std::string &col_name) const {
  size_t type_index = static_cast<size_t>(type);

  auto visitor =
      [&col_name](const auto &metas) -> const std::vector<BlockMeta> & {
    using T = std::decay_t<decltype(metas)>;

    static const std::vector<BlockMeta> empty_metas;

    if constexpr (std::is_same_v<T, std::vector<BlockMeta>>) {
      return metas;
    } else if constexpr (std::is_same_v<
                             T, std::unordered_map<std::string,
                                                   std::vector<BlockMeta>>>) {
      auto it = metas.find(col_name);
      if (it != metas.end()) {
        return it->second;
      }
    }

    return empty_metas;
  };

  return std::visit(visitor, persist_block_metas_[type_index]);
}

Status SegmentImpl::load_persist_scalar_blocks() {
  doc_ids_.reserve(segment_meta_->doc_count());
  for (const auto &block : segment_meta_->persisted_blocks()) {
    if (block.type() == BlockType::SCALAR) {
      auto forward_path = FileHelper::MakeForwardBlockPath(
          path_, segment_meta_->id(), block.id_, !options_.enable_mmap_);

      BaseForwardStore::Ptr forward_store;
      if (options_.enable_mmap_) {
        forward_store = std::make_shared<MmapForwardStore>(forward_path);
      } else {
        forward_store = std::make_shared<BufferPoolForwardStore>(forward_path);
      }
      auto s = forward_store->Open();
      CHECK_RETURN_STATUS(s);
      persist_stores_.push_back(forward_store);

      if (!block.contain_column(GLOBAL_DOC_ID)) {
        continue;
      }
      auto rb_reader = forward_store->scan({GLOBAL_DOC_ID});
      while (true) {
        std::shared_ptr<arrow::RecordBatch> batch;
        auto status = rb_reader->ReadNext(&batch);
        if (!status.ok()) {
          LOG_ERROR("Read batch failed: %s", status.message().c_str());
          return Status::InternalError(status.message());
        }

        if (batch == nullptr) {
          break;
        }

        auto uint64_array =
            std::dynamic_pointer_cast<arrow::UInt64Array>(batch->column(0));
        if (!uint64_array) {
          LOG_ERROR("Failed to cast column to UInt64Array");
          return Status::InternalError("Array type mismatch");
        }
        auto *values = uint64_array->raw_values();
        doc_ids_.insert(doc_ids_.end(), values,
                        values + uint64_array->length());
      }
    }
  }

  return Status::OK();
}

Status SegmentImpl::load_scalar_index_blocks(bool create) {
  std::vector<FieldSchema> fields;
  std::vector<std::string> field_names;
  for (const auto &field : collection_schema_->forward_fields()) {
    if (field->index_type() == IndexType::INVERT) {
      fields.push_back(*field);
      field_names.push_back(field->name());
    }
  }

  if (fields.empty()) {
    LOG_INFO("No scalar index found");
    return Status::OK();
  }

  if (create) {
    auto block_id = allocate_block_id();
    auto invert_path = FileHelper::MakeInvertIndexPath(path_, id(), block_id);
    auto collection_name = collection_schema_->name();
    invert_indexers_ = InvertedIndexer::CreateAndOpen(
        collection_name, invert_path, true, fields, options_.read_only_);
    if (!invert_indexers_) {
      LOG_ERROR("Failed to open scalar indexer");
      return Status::InternalError("Failed to open scalar indexer");
    }

    // scalar index block
    segment_meta_->add_persisted_block(
        BlockMeta{block_id, BlockType::SCALAR_INDEX, 0, 0, 0, field_names});

    return Status::OK();
  } else {
    for (const auto &block : segment_meta_->persisted_blocks()) {
      if (block.type() == BlockType::SCALAR_INDEX) {
        auto block_id = block.id();
        auto invert_path =
            FileHelper::MakeInvertIndexPath(path_, id(), block_id);
        auto collection_name = collection_schema_->name();
        invert_indexers_ = InvertedIndexer::CreateAndOpen(
            collection_name, invert_path, false, fields, options_.read_only_);
        if (!invert_indexers_) {
          LOG_ERROR("Failed to open scalar indexer");
          return Status::InternalError("Failed to open scalar indexer");
        }
        return Status::OK();
      }
    }

    if (invert_indexers_ == nullptr) {
      LOG_ERROR("No scalar index found");
      return Status::NotFound("No scalar index found");
    }
  }
  return Status::OK();
}

Status SegmentImpl::load_vector_index_blocks() {
  for (const auto &block : segment_meta_->persisted_blocks()) {
    if (block.type() == BlockType::VECTOR_INDEX ||
        block.type() == BlockType::VECTOR_INDEX_QUANTIZE) {
      // vector block only contained 1 column
      auto column = block.columns()[0];

      FieldSchema new_field_params =
          *collection_schema_->get_vector_field(column);

      auto vector_index_params = std::dynamic_pointer_cast<VectorIndexParams>(
          new_field_params.index_params());
      if (block.type_ == BlockType::VECTOR_INDEX) {
        if (vector_index_params->quantize_type() != QuantizeType::UNDEFINED ||
            !segment_meta_->vector_indexed(column)) {
          new_field_params.set_index_params(
              MakeDefaultVectorIndexParams(vector_index_params->metric_type()));
        }
      } else {
        if (!segment_meta_->vector_indexed(column)) {
          new_field_params.set_index_params(MakeDefaultQuantVectorIndexParams(
              vector_index_params->metric_type(),
              vector_index_params->quantize_type()));
        }
      }

      std::string index_path;
      if (block.type_ == BlockType::VECTOR_INDEX) {
        index_path = FileHelper::MakeVectorIndexPath(
            path_, column, segment_meta_->id(), block.id_);

      } else {
        index_path = FileHelper::MakeQuantizeVectorIndexPath(
            path_, column, segment_meta_->id(), block.id_);
      }

      auto vector_indexer =
          std::make_shared<VectorColumnIndexer>(index_path, new_field_params);
      auto s = vector_indexer->Open(vector_column_params::ReadOptions{
          options_.enable_mmap_, false, true});
      CHECK_RETURN_STATUS(s);

      if (block.type_ == BlockType::VECTOR_INDEX) {
        auto it = vector_indexers_.find(column);
        if (it == vector_indexers_.end()) {
          std::vector<VectorColumnIndexer::Ptr> vector_indexers;
          vector_indexers.push_back(vector_indexer);
          vector_indexers_.emplace(column, std::move(vector_indexers));
        } else {
          it->second.push_back(vector_indexer);
        }
      } else {
        auto it = quant_vector_indexers_.find(column);
        if (it == quant_vector_indexers_.end()) {
          std::vector<VectorColumnIndexer::Ptr> vector_indexers;
          vector_indexers.push_back(vector_indexer);
          quant_vector_indexers_.emplace(column, std::move(vector_indexers));
        } else {
          it->second.push_back(vector_indexer);
        }
      }
    }
  }
  return Status::OK();
}

VectorColumnIndexer::Ptr SegmentImpl::create_vector_indexer(
    const std::string &field_name, const FieldSchema &field, BlockID block_id,
    bool is_quantized) {
  std::string index_file_path;
  if (is_quantized) {
    index_file_path = FileHelper::MakeQuantizeVectorIndexPath(
        path_, field_name, segment_meta_->id(), block_id);
    quant_memory_vector_block_ids_[field_name] = block_id;
  } else {
    index_file_path = FileHelper::MakeVectorIndexPath(
        path_, field_name, segment_meta_->id(), block_id);
    memory_vector_block_ids_[field_name] = block_id;
  }

  auto vector_indexer =
      std::make_shared<VectorColumnIndexer>(index_file_path, field);
  vector_column_params::ReadOptions options{true, true};
  auto status = vector_indexer->Open(options);
  if (!status.ok()) {
    LOG_ERROR("Failed to open vector indexer for field: %s, err: %s",
              field.to_string().c_str(), status.message().c_str());
    return nullptr;
  }
  return vector_indexer;
}

Status SegmentImpl::init_memory_components() {
  // init memory block id
  auto &mem_block = segment_meta_->writing_forward_block().value();

  // create and open memory forward block
  auto mem_path = FileHelper::MakeForwardBlockPath(seg_path_, mem_block.id_,
                                                   !options_.enable_mmap_);
  memory_store_ = std::make_shared<MemForwardStore>(
      collection_schema_, mem_path,
      options_.enable_mmap_ ? FileFormat::IPC : FileFormat::PARQUET,
      options_.max_buffer_size_);
  auto s = memory_store_->Open();
  CHECK_RETURN_STATUS(s);

  // create and open memory vector indexer
  for (const auto &field : collection_schema_->vector_fields()) {
    auto index_params =
        std::dynamic_pointer_cast<VectorIndexParams>(field->index_params());

    if (index_params->quantize_type() == QuantizeType::UNDEFINED) {
      // create normal vector indexer
      FieldSchema normal_field(*field);
      normal_field.set_index_params(
          MakeDefaultVectorIndexParams(index_params->metric_type()));
      auto block_id = allocate_block_id();
      auto vector_indexer =
          create_vector_indexer(field->name(), normal_field, block_id);
      if (!vector_indexer) {
        return Status::InternalError("Create vector column indexer failed: ",
                                     field->name());
      }
      memory_vector_indexers_.insert({field->name(), vector_indexer});
    } else {
      // first create normal vector indexer
      FieldSchema normal_field(*field);
      normal_field.set_index_params(
          MakeDefaultVectorIndexParams(index_params->metric_type()));
      auto block_id = allocate_block_id();
      auto vector_indexer =
          create_vector_indexer(field->name(), normal_field, block_id);
      if (!vector_indexer) {
        return Status::InternalError("Create vector column indexer failed: ",
                                     field->name());
      }
      memory_vector_indexers_.insert({field->name(), vector_indexer});

      // second create quantize vector indexer
      block_id = allocate_block_id();
      FieldSchema normal_quant_field(*field);
      normal_quant_field.set_index_params(MakeDefaultQuantVectorIndexParams(
          index_params->metric_type(), index_params->quantize_type()));
      auto quant_vector_indexer = create_vector_indexer(
          field->name(), normal_quant_field, block_id, true);

      if (!quant_vector_indexer) {
        return Status::InternalError("Create vector column indexer failed: ",
                                     field->name());
      }
      quant_memory_vector_indexers_.insert(
          {field->name(), quant_vector_indexer});
    }
  }

  return Status::OK();
}

Status SegmentImpl::recover() {
  // recover mem block meta
  auto &mem_block = segment_meta_->writing_forward_block().value();
  doc_id_allocator_.store(mem_block.min_doc_id());

  std::string wal_file_path =
      FileHelper::MakeWalPath(path_, segment_meta_->id(), mem_block.id_);
  if (!std::filesystem::exists(wal_file_path)) {
    LOG_INFO("Recover wal file not exists just return. path: %s",
             wal_file_path.c_str());
    return Status::OK();
  }

  WalFilePtr recover_wal_file;
  WalOptions wal_option;
  wal_option.create_new = false;
  if (WalFile::CreateAndOpen(wal_file_path, wal_option, &recover_wal_file) !=
      0) {
    LOG_WARN("Recover wal file failed. path: %s", wal_file_path.c_str());
    return Status::OK();
  }
  AILEGO_DEFER([&]() { recover_wal_file->close(); });

  std::array<uint64_t, static_cast<size_t>(Operator::DELETE) + 1>
      recovered_doc_count{};
  uint64_t total_recovered_doc_count{0};

  int ret = recover_wal_file->prepare_for_read();
  if (ret != 0) {
    LOG_ERROR("Recover wal file failed. path: %s", wal_file_path.c_str());
    return Status::InternalError("Failed to prepare wal file: ", wal_file_path,
                                 " for read");
  }

  LOG_INFO("Recover start read wal [%s]", wal_file_path.c_str());

  std::lock_guard<std::mutex> lock(seg_mtx_);

  while (true) {
    std::string buf = recover_wal_file->next();
    if (buf.empty()) {
      LOG_INFO("Recover read wal finished");
      break;
    }
    total_recovered_doc_count++;
    auto doc = Doc::deserialize(reinterpret_cast<const uint8_t *>(buf.data()),
                                buf.size());
    if (doc == nullptr) {
      LOG_ERROR("Recover wal failed. doc deserialize failed at %zu",
                (size_t)total_recovered_doc_count);
      continue;
    }

    Status status;
    switch (doc->get_operator()) {
      case Operator::INSERT: {
        internal_insert(*doc);
        break;
      }
      case Operator::UPDATE: {
        internal_update(*doc);
        break;
      }
      case Operator::UPSERT: {
        internal_upsert(*doc);
        break;
      }
      case Operator::DELETE: {
        internal_delete(*doc);
        break;
      }
      default:
        LOG_ERROR("Unknown operator type: %d", (int)doc->get_operator());
        break;
    }

    if (!status.ok()) {
      LOG_ERROR("Recover wal failed. Operation %d failed at %zu: %s",
                static_cast<int>(doc->get_operator()),
                (size_t)total_recovered_doc_count, status.message().c_str());
      continue;
    }

    recovered_doc_count[static_cast<size_t>(doc->get_operator())]++;
  }

  const auto added_docs = recovered_doc_count[0] +  // INSERT
                          recovered_doc_count[1] +  // UPDATE
                          recovered_doc_count[2];   // UPSERT
  mem_block.max_doc_id_ += added_docs;

  LOG_INFO(
      "Recover from wal finished. total_recovered_doc_count[%zu] "
      "insert[%zu] update[%zu] upsert[%zu] "
      "delete[%zu] path[%s]",
      (size_t)total_recovered_doc_count,
      (size_t)recovered_doc_count[0],  // INSERT
      (size_t)recovered_doc_count[1],  // UPDATE
      (size_t)recovered_doc_count[2],  // UPSERT
      (size_t)recovered_doc_count[3],  // DELETE
      wal_file_path.c_str());

  return Status::OK();
}

Status SegmentImpl::open_wal_file() {
  auto mem_block = segment_meta_->writing_forward_block().value();
  std::string wal_file_path =
      FileHelper::MakeWalPath(path_, segment_meta_->id(), mem_block.id_);
  WalOptions wal_option;
  if (std::filesystem::exists(wal_file_path)) {
    wal_option.create_new = false;
  } else {
    wal_option.create_new = true;
  }

  if (WalFile::CreateAndOpen(wal_file_path, wal_option, &wal_file_) != 0) {
    LOG_ERROR("Recover wal file failed. path: %s", wal_file_path.c_str());

    return Status::OK();
  }

  LOG_INFO("Open wal file succ. path: %s", wal_file_path.c_str());
  return Status::OK();
}

Status SegmentImpl::append_wal(const Doc &doc) {
  std::vector<uint8_t> buf = doc.serialize();

  if (!wal_file_) {
    auto s = open_wal_file();
    CHECK_RETURN_STATUS(s);
  }

  auto ret = wal_file_->append(std::string(buf.begin(), buf.end()));
  if (ret != 0) {
    LOG_ERROR("Append wal failed. ret: %d", ret);
    return Status::InternalError("Failed to append wal");
  }

  return Status::OK();
}

Status SegmentImpl::finish_memory_components() {
  auto block = segment_meta_->writing_forward_block().value();

  // close for loading persist block
  auto s = memory_store_->close();
  CHECK_RETURN_STATUS(s);
  memory_store_.reset();

  // load forward store
  auto persist_forward_store_path = FileHelper::MakeForwardBlockPath(
      path_, segment_meta_->id(), block.id_, !options_.enable_mmap_);

  BaseForwardStore::Ptr persist_store;
  if (options_.enable_mmap_) {
    persist_store =
        std::make_shared<MmapForwardStore>(persist_forward_store_path);
  } else {
    persist_store =
        std::make_shared<BufferPoolForwardStore>(persist_forward_store_path);
  }
  s = persist_store->Open();
  CHECK_RETURN_STATUS(s);
  persist_stores_.push_back(persist_store);

  BlockMeta b{block.id_,         block.type_,      block.min_doc_id_,
              block.max_doc_id_, block.doc_count_, block.columns_};
  segment_meta_->add_persisted_block(b);

  // remove indexer from memory to persist
  for (auto &[column_name, indexer] : memory_vector_indexers_) {
    auto block_id = memory_vector_block_ids_[column_name];
    BlockMeta vb =
        BlockMeta{block_id,          BlockType::VECTOR_INDEX, block.min_doc_id_,
                  block.max_doc_id_, block.doc_count_,        {column_name}};
    auto it = vector_indexers_.find(column_name);
    if (it == vector_indexers_.end()) {
      std::vector<VectorColumnIndexer::Ptr> vector_indexers{indexer};
      vector_indexers_.emplace(column_name, std::move(vector_indexers));
    } else {
      it->second.push_back(indexer);
    }
    segment_meta_->add_persisted_block(vb);
  }

  // remove quant indexer from memory to persist
  for (auto &[column_name, indexer] : quant_memory_vector_indexers_) {
    auto block_id = quant_memory_vector_block_ids_[column_name];
    BlockMeta block_meta(block_id, BlockType::VECTOR_INDEX_QUANTIZE,
                         block.min_doc_id_, block.max_doc_id_, block.doc_count_,
                         {column_name});

    auto it = quant_vector_indexers_.find(column_name);
    if (it == quant_vector_indexers_.end()) {
      std::vector<VectorColumnIndexer::Ptr> vector_indexers;
      vector_indexers.push_back(indexer);
      quant_vector_indexers_.emplace(column_name, std::move(vector_indexers));
    } else {
      it->second.push_back(indexer);
    }
    segment_meta_->add_persisted_block(block_meta);
  }

  // clear memory vector indexers
  memory_vector_indexers_.clear();
  quant_memory_vector_indexers_.clear();
  memory_vector_block_ids_.clear();
  quant_memory_vector_block_ids_.clear();

  fresh_persist_block_offset();
  return Status::OK();
}

Status SegmentImpl::update_version(uint32_t delete_snapshot_path_suffix) {
  if (version_manager_) {
    if (delete_snapshot_path_suffix != UINT32_MAX) {
      version_manager_->set_delete_snapshot_path_suffix(
          delete_snapshot_path_suffix);
    }
    auto s = version_manager_->reset_writing_segment_meta(segment_meta_);
    CHECK_RETURN_STATUS(s);
    s = version_manager_->flush();
    CHECK_RETURN_STATUS(s);
  }
  return Status::OK();
}

BlockID SegmentImpl::allocate_block_id() {
  return block_id_allocator_.fetch_add(1);
}

Result<uint64_t> SegmentImpl::get_global_doc_id(uint32_t local_id) const {
  std::lock_guard lock(seg_mtx_);
  if (local_id >= doc_ids_.size()) {
    return tl::make_unexpected(
        Status::InvalidArgument("local_id out of range"));
  }
  // global doc_id
  return doc_ids_[local_id];
}


////////////////////////////////////////////////////////////////////////////////////
// Segment factory methods implementation
////////////////////////////////////////////////////////////////////////////////////

Result<Segment::Ptr> Segment::CreateAndOpen(
    const std::string &path, const CollectionSchema &schema,
    SegmentID segment_id, uint64_t min_doc_id, const IDMap::Ptr &id_map,
    const DeleteStore::Ptr &delete_store,
    const VersionManager::Ptr &version_manager, const SegmentOptions &options) {
  auto segment = std::shared_ptr<SegmentImpl>(
      new SegmentImpl(path, schema, SegmentMeta(segment_id), id_map,
                      delete_store, version_manager));

  auto segment_path = FileHelper::MakeSegmentPath(path, segment_id);
  // check or create path
  if (FileHelper::DirectoryExists(segment_path)) {
    return tl::make_unexpected(Status::InternalError(
        "Segment path is already exists: ", segment_path));
  } else {
    if (!FileHelper::CreateDirectory(segment_path)) {
      return tl::make_unexpected(Status::InternalError(
          "Create segment directory failed: ", segment_path));
    }
  }

  auto s = segment->Create(options, min_doc_id);
  CHECK_RETURN_STATUS_EXPECTED(s);

  return segment;
}

Result<Segment::Ptr> Segment::Open(const std::string &path,
                                   const CollectionSchema &schema,
                                   const SegmentMeta &segment_meta,
                                   const IDMap::Ptr &id_map,
                                   const DeleteStore::Ptr &delete_store,
                                   const VersionManager::Ptr &version_manager,
                                   const SegmentOptions &options) {
  auto segment = std::shared_ptr<SegmentImpl>(new SegmentImpl(
      path, schema, segment_meta, id_map, delete_store, version_manager));

  auto segment_path = FileHelper::MakeSegmentPath(path, segment_meta.id());
  // check path
  if (!FileHelper::DirectoryExists(segment_path)) {
    return tl::make_unexpected(
        Status::InternalError("Segment path is not exist: ", segment_path));
  }

  auto s = segment->Open(options);
  CHECK_RETURN_STATUS_EXPECTED(s);

  return segment;
}

}  // namespace zvec