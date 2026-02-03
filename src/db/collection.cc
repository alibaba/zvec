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

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <variant>
#include <vector>
#include <ailego/io/file_lock.h>
#include <zvec/ailego/io/file.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/ailego/pattern/expected.hpp>
#include <zvec/ailego/utility/file_helper.h>
#include <zvec/ailego/utility/string_helper.h>
#include <zvec/db/collection.h>
#include <zvec/db/doc.h>
#include <zvec/db/options.h>
#include <zvec/db/schema.h>
#include <zvec/db/status.h>
#include "db/common/constants.h"
#include "db/common/file_helper.h"
#include "db/common/profiler.h"
#include "db/common/typedef.h"
#include "db/index/common/delete_store.h"
#include "db/index/common/id_map.h"
#include "db/index/common/index_filter.h"
#include "db/index/common/version_manager.h"
#include "db/index/segment/segment.h"
#include "db/index/segment/segment_helper.h"
#include "db/index/segment/segment_manager.h"
#include "db/sqlengine/sqlengine.h"

namespace zvec {

enum class WriteMode : uint8_t {
  UNDEFINED = 0,
  INSERT,
  UPDATE,
  UPSERT,
};

Collection::~Collection() = default;

class CollectionImpl : public Collection {
  friend class Collection;

 public:
  explicit CollectionImpl(const std::string &path,
                          const CollectionSchema &schema);

  explicit CollectionImpl(const std::string &path);

  ~CollectionImpl() override;

 private:
  Status Open(const CollectionOptions &options);

  Status Close();

 public:
  Status Destroy() override;

  Status Flush() override;

  Result<std::string> Path() const override;

  Result<CollectionStats> Stats() const override;

  Result<CollectionSchema> Schema() const override;

  Result<CollectionOptions> Options() const override;

 public:
  Status CreateIndex(const std::string &column_name,
                     const IndexParams::Ptr &index_params,
                     const CreateIndexOptions &options) override;

  Status DropIndex(const std::string &column_name) override;

  Status Optimize(const OptimizeOptions &options) override;

  Status AddColumn(const FieldSchema::Ptr &column_schema,
                   const std::string &expression,
                   const AddColumnOptions &options) override;

  Status DropColumn(const std::string &column_name) override;

  Status AlterColumn(
      const std::string &column_name, const std::string &rename,
      const FieldSchema::Ptr &new_column_schema = nullptr,
      const AlterColumnOptions &options = AlterColumnOptions()) override;

  Result<WriteResults> Insert(std::vector<Doc> &docs) override;

  Result<WriteResults> Upsert(std::vector<Doc> &docs) override;

  Result<WriteResults> Update(std::vector<Doc> &docs) override;

  Result<WriteResults> Delete(const std::vector<std::string> &pks) override;

  Status DeleteByFilter(const std::string &filter) override;

  Result<DocPtrList> Query(const VectorQuery &query) const override;

  Result<GroupResults> GroupByQuery(
      const GroupByVectorQuery &query) const override;

  Result<DocPtrMap> Fetch(const std::vector<std::string> &pks) const override;

 private:
  void prepare_schema();

  Status close_unsafe();

  Status flush_unsafe();

  Status create();

  Status recovery();

  Status create_idmap_and_delete_store();

  Status recover_idmap_and_delete_store();

  Status acquire_file_lock(bool create = false);

  Status init_version_manager();

  Status init_writing_segment();

  bool need_switch_to_new_segment() const;

  Status switch_to_new_segment_for_writing(
      const CollectionSchema::Ptr &schema = nullptr);

  Result<WriteResults> write_impl(std::vector<Doc> &docs, WriteMode mode);

  std::vector<Segment::Ptr> get_all_segments() const;

  std::vector<Segment::Ptr> get_all_persist_segments() const;

  Segment::Ptr local_segment_by_doc_id(
      uint64_t doc_id, const std::vector<Segment::Ptr> &segments) const;

  SegmentID allocate_segment_id() {
    return segment_id_allocator_.fetch_add(1);
  }

  SegmentID allocate_segment_id_for_tmp_segment() {
    return tmp_segment_id_allocator_.fetch_add(1);
  }

  std::vector<SegmentTask::Ptr> build_compact_task(
      const CollectionSchema::Ptr &schema,
      const std::vector<Segment::Ptr> &segments, int concurrency,
      const IndexFilter::Ptr filter);

  Status execute_compact_task(std::vector<SegmentTask::Ptr> &tasks) const;

  std::vector<SegmentTask::Ptr> build_create_vector_index_task(
      const std::vector<Segment::Ptr> &segments, const std::string &column,
      const IndexParams::Ptr &index_params, int concurrency);

  std::vector<SegmentTask::Ptr> build_create_scalar_index_task(
      const std::vector<Segment::Ptr> &segments, const std::string &column,
      const IndexParams::Ptr &index_params, int concurrency);

  std::vector<SegmentTask::Ptr> build_drop_vector_index_task(
      const std::vector<Segment::Ptr> &segments, const std::string &column);

  std::vector<SegmentTask::Ptr> build_drop_scalar_index_task(
      const std::vector<Segment::Ptr> &segments, const std::string &column);

  Status execute_tasks(std::vector<SegmentTask::Ptr> &tasks) const;

 private:
  Status handle_upsert(Doc &doc);

  Status handle_update(Doc &doc);

  Status handle_insert(Doc &doc);

  Status internal_fetch_by_doc(const Doc &doc, Doc::Ptr *doc_out);

 private:
  // Helper functions for add/alter/drop column
  Status validate(const std::string &column, const FieldSchema::Ptr &schema,
                  const std::string &expression, const std::string &rename,
                  ColumnOp op);

 private:
  std::string path_;

  bool destroyed_{false};

  CollectionSchema::Ptr schema_;

  CollectionOptions options_;

  mutable std::shared_mutex schema_handle_mtx_;
  mutable std::shared_mutex write_mtx_;

  std::atomic<SegmentID> segment_id_allocator_;
  std::atomic<SegmentID> tmp_segment_id_allocator_;

  // writing segment
  Segment::Ptr writing_segment_;
  // non-writing segments, sort by doc_id range
  SegmentManager::Ptr segment_manager_;

  // latest version: std::vector<SegmentMeta>
  VersionManager::Ptr version_manager_;

  // file lock
  ailego::File lock_file_;

  IDMap::Ptr id_map_;
  DeleteStore::Ptr delete_store_;

  sqlengine::SQLEngine::Ptr sql_engine_;
};

Result<Collection::Ptr> Collection::CreateAndOpen(
    const std::string &path, const CollectionSchema &schema,
    const CollectionOptions &options) {
  auto collection = std::make_shared<CollectionImpl>(path, schema);

  auto s = collection->Open(options);
  CHECK_RETURN_STATUS_EXPECTED(s);

  return collection;
}

Result<Collection::Ptr> Collection::Open(const std::string &path,
                                         const CollectionOptions &options) {
  auto collection = std::make_shared<CollectionImpl>(path);

  auto s = collection->Open(options);
  CHECK_RETURN_STATUS_EXPECTED(s);

  return collection;
}

CollectionImpl::CollectionImpl(const std::string &path,
                               const CollectionSchema &schema)
    : path_(path), schema_(std::make_shared<CollectionSchema>(schema)) {
  prepare_schema();
}

void CollectionImpl::prepare_schema() {
  // set default index params for vector fields
  for (auto &field : schema_->fields()) {
    if (field->is_vector_field()) {
      if (field->index_params() == nullptr) {
        field->set_index_params(DefaultVectorIndexParams);
      }
    }
  }
}

CollectionImpl::CollectionImpl(const std::string &path) : path_(path) {}

CollectionImpl::~CollectionImpl() {
  if (!destroyed_) {
    Close();
  }
}

Status CollectionImpl::Open(const CollectionOptions &options) {
  options_ = options;

  if (schema_ != nullptr && options_.read_only_) {
    return Status::InvalidArgument(
        "Unable to create collection with read-only mode.");
  }

  Status s;
  if (schema_ == nullptr) {
    // recovery from disk
    s = recovery();
  } else {
    // create new collection with existing schema
    s = create();
  }

  auto profiler = std::make_shared<Profiler>();
  sql_engine_ = sqlengine::SQLEngine::create(profiler);

  return s;
}

Status CollectionImpl::Close() {
  // only called in deconstructor
  std::lock_guard lock(schema_handle_mtx_);

  CHECK_DESTROY_RETURN_STATUS(destroyed_, false);

  return close_unsafe();
}

Status CollectionImpl::close_unsafe() {
  // flush
  if (!options_.read_only_) {
    auto s = flush_unsafe();
    CHECK_RETURN_STATUS(s);
  }

  // reset
  writing_segment_.reset();
  segment_manager_.reset();
  version_manager_.reset();
  id_map_.reset();
  delete_store_.reset();

  lock_file_.close();

  return Status::OK();
}

Status CollectionImpl::Destroy() {
  CHECK_COLLECTION_READONLY_RETURN_STATUS;

  std::lock_guard lock(schema_handle_mtx_);

  CHECK_DESTROY_RETURN_STATUS(destroyed_, false);

  auto s = close_unsafe();
  CHECK_RETURN_STATUS(s);

  ailego::FileHelper::RemoveDirectory(path_.c_str());

  destroyed_ = true;

  return Status::OK();
}

Status CollectionImpl::Flush() {
  CHECK_COLLECTION_READONLY_RETURN_STATUS;

  std::lock_guard lock(schema_handle_mtx_);
  CHECK_DESTROY_RETURN_STATUS(destroyed_, false);

  return flush_unsafe();
}

Status CollectionImpl::flush_unsafe() {
  if (!writing_segment_) {
    return Status::InternalError(
        "flush writing segment failed because writing segment is nullptr");
  }
  return writing_segment_->flush();
}

Result<std::string> CollectionImpl::Path() const {
  CHECK_DESTROY_RETURN_STATUS_EXPECTED(destroyed_, false);

  return path_;
}

Result<CollectionStats> CollectionImpl::Stats() const {
  std::lock_guard lock(schema_handle_mtx_);

  CHECK_DESTROY_RETURN_STATUS_EXPECTED(destroyed_, false);

  auto segments = get_all_segments();

  CollectionStats stats;
  auto vector_fields = schema_->vector_fields();
  if (segments.empty()) {
    stats.doc_count = 0;
    for (auto &field : vector_fields) {
      stats.index_completeness[field->name()] =
          1;  // if no doc, completeness is 1
    }
    return stats;
  }

  for (auto &segment : segments) {
    stats.doc_count += segment->doc_count(delete_store_->make_filter());
  }

  for (auto &field : vector_fields) {
    if (stats.doc_count == 0) {
      stats.index_completeness[field->name()] = 1;
      continue;
    }

    uint32_t indexed_doc_count{0};
    for (auto &segment : segments) {
      if (segment->meta()->vector_indexed(field->name())) {
        indexed_doc_count += segment->doc_count(delete_store_->make_filter());
      }
    }
    stats.index_completeness[field->name()] =
        indexed_doc_count * 1.0 / stats.doc_count;
  }

  return stats;
}

Result<CollectionSchema> CollectionImpl::Schema() const {
  std::lock_guard lock(schema_handle_mtx_);

  CHECK_DESTROY_RETURN_STATUS_EXPECTED(destroyed_, false);

  return *schema_;
}

Result<CollectionOptions> CollectionImpl::Options() const {
  std::lock_guard lock(schema_handle_mtx_);

  CHECK_DESTROY_RETURN_STATUS_EXPECTED(destroyed_, false);

  return options_;
}

Status CollectionImpl::CreateIndex(const std::string &column_name,
                                   const IndexParams::Ptr &index_params,
                                   const CreateIndexOptions &options) {
  CHECK_COLLECTION_READONLY_RETURN_STATUS;

  std::lock_guard lock(schema_handle_mtx_);

  CHECK_DESTROY_RETURN_STATUS(destroyed_, false);

  auto new_schema = std::make_shared<CollectionSchema>(*schema_);
  auto s = new_schema->add_index(column_name, index_params);
  CHECK_RETURN_STATUS(s);
  s = new_schema->validate();
  CHECK_RETURN_STATUS(s);

  auto field = schema_->get_field(column_name);
  if (field->index_params() != nullptr &&
      *field->index_params() == *index_params) {
    // equal index params
    return Status::OK();
  }

  // forbidden writing until index is ready
  std::lock_guard write_lock(write_mtx_);

  Version new_version = version_manager_->get_current_version();

  if (writing_segment_->doc_count() > 0) {
    s = writing_segment_->dump();
    CHECK_RETURN_STATUS(s);

    s = segment_manager_->add_segment(writing_segment_);
    CHECK_RETURN_STATUS(s);

    auto seg_options =
        SegmentOptions{false, options_.enable_mmap_, options_.max_buffer_size_};
    auto new_segment = Segment::CreateAndOpen(
        path_, *new_schema, allocate_segment_id(),
        writing_segment_->meta()->max_doc_id() + 1, id_map_, delete_store_,
        version_manager_, seg_options);
    if (!new_segment) {
      return new_segment.error();
    }

    s = new_version.add_persisted_segment_meta(writing_segment_->meta());
    CHECK_RETURN_STATUS(s);

    writing_segment_ = new_segment.value();
    new_version.set_next_segment_id(segment_id_allocator_.load());

  } else {
    // TODO: allocate new segment id and clear current writing segment at last
    // recreate writing segment
    s = writing_segment_->destroy();
    CHECK_RETURN_STATUS(s);
    auto id = writing_segment_->id();
    auto min_doc_id = writing_segment_->meta()->min_doc_id();
    writing_segment_.reset();
    SegmentOptions seg_options;
    seg_options.enable_mmap_ = options_.enable_mmap_;
    seg_options.max_buffer_size_ = options_.max_buffer_size_;
    seg_options.read_only_ = options_.read_only_;
    auto writing_segment =
        Segment::CreateAndOpen(path_, *new_schema, id, min_doc_id, id_map_,
                               delete_store_, version_manager_, seg_options);
    if (!writing_segment) {
      return writing_segment.error();
    }
    writing_segment_ = writing_segment.value();
  }
  new_version.reset_writing_segment_meta(writing_segment_->meta());

  // get_all_segment will return writing segment if it has docs
  auto persist_segments = get_all_persist_segments();

  bool is_vector_field = field->is_vector_field();

  std::vector<SegmentTask::Ptr> tasks;
  if (is_vector_field) {
    tasks = build_create_vector_index_task(persist_segments, column_name,
                                           index_params, options.concurrency_);

  } else {
    tasks = build_create_scalar_index_task(persist_segments, column_name,
                                           index_params, options.concurrency_);
  }

  if (tasks.empty()) {
    new_version.set_schema(*new_schema);

    s = version_manager_->apply(new_version);
    CHECK_RETURN_STATUS(s);

    // persist manifest
    s = version_manager_->flush();
    CHECK_RETURN_STATUS(s);

    schema_ = new_schema;
    return Status::OK();
  }

  s = execute_tasks(tasks);
  CHECK_RETURN_STATUS(s);

  new_version.set_schema(*new_schema);

  for (auto &task : tasks) {
    auto task_info = task->task_info();

    if (std::holds_alternative<CreateVectorIndexTask>(task_info)) {
      auto create_index_task = std::get<CreateVectorIndexTask>(task_info);
      s = new_version.update_persisted_segment_meta(
          create_index_task.output_segment_meta_);
    } else if (std::holds_alternative<CreateScalarIndexTask>(task_info)) {
      auto create_index_task = std::get<CreateScalarIndexTask>(task_info);
      s = new_version.update_persisted_segment_meta(
          create_index_task.output_segment_meta_);
    }
    CHECK_RETURN_STATUS(s);
  }

  // 2. update version
  s = version_manager_->apply(new_version);
  CHECK_RETURN_STATUS(s);

  // 3. persist version
  s = version_manager_->flush();
  CHECK_RETURN_STATUS(s);

  // 4. remove old segments or block
  for (auto &task : tasks) {
    auto task_info = task->task_info();

    if (std::holds_alternative<CreateVectorIndexTask>(task_info)) {
      auto create_index_task = std::get<CreateVectorIndexTask>(task_info);
      s = create_index_task.input_segment_->reload_vector_index(
          *new_schema, create_index_task.output_segment_meta_,
          create_index_task.output_vector_indexers_,
          create_index_task.output_quant_vector_indexers_);
    } else if (std::holds_alternative<CreateScalarIndexTask>(task_info)) {
      auto create_index_task = std::get<CreateScalarIndexTask>(task_info);
      s = create_index_task.input_segment_->reload_scalar_index(
          *new_schema, create_index_task.output_segment_meta_,
          create_index_task.output_scalar_indexer_);
    }
    CHECK_RETURN_STATUS(s);
  }

  schema_ = new_schema;

  return Status::OK();
}

std::vector<SegmentTask::Ptr> CollectionImpl::build_create_vector_index_task(
    const std::vector<Segment::Ptr> &segments, const std::string &column,
    const IndexParams::Ptr &index_params, int concurrency) {
  std::vector<SegmentTask::Ptr> tasks;
  for (auto &segment : segments) {
    if (!segment->vector_index_ready(column, index_params)) {
      tasks.push_back(SegmentTask::CreateCreateVectorIndexTask(
          CreateVectorIndexTask{segment, column, index_params, concurrency}));
    }
  }
  return tasks;
}

std::vector<SegmentTask::Ptr> CollectionImpl::build_create_scalar_index_task(
    const std::vector<Segment::Ptr> &segments, const std::string &column,
    const IndexParams::Ptr &index_params, int concurrency) {
  std::vector<SegmentTask::Ptr> tasks;
  for (auto &segment : segments) {
    tasks.push_back(SegmentTask::CreateCreateScalarIndexTask(
        CreateScalarIndexTask{segment, {column}, index_params, concurrency}));
  }
  return tasks;
}

Status CollectionImpl::execute_tasks(
    std::vector<SegmentTask::Ptr> &tasks) const {
  Status s;
  for (auto &task : tasks) {
    s = SegmentHelper::Execute(task);
    if (!s.ok()) {
      return s;
    }
  }

  return Status::OK();
}

Status CollectionImpl::DropIndex(const std::string &column_name) {
  CHECK_COLLECTION_READONLY_RETURN_STATUS;
  CHECK_DESTROY_RETURN_STATUS(destroyed_, false);

  std::lock_guard lock(schema_handle_mtx_);

  auto new_schema = std::make_shared<CollectionSchema>(*schema_);
  auto s = new_schema->drop_index(column_name);
  CHECK_RETURN_STATUS(s);

  auto field = schema_->get_field(column_name);
  if (field->index_params() == nullptr) {
    return Status::OK();  // return ok if not indexed
  }

  if (field->is_vector_field() &&
      *field->index_params() == DefaultVectorIndexParams) {
    return Status::OK();
  }

  // forbidden writing until index is ready
  std::lock_guard write_lock(write_mtx_);

  Version new_version = version_manager_->get_current_version();

  bool is_vector_field = field->is_vector_field();

  if (writing_segment_->doc_count() > 0) {
    s = writing_segment_->dump();
    CHECK_RETURN_STATUS(s);

    s = segment_manager_->add_segment(writing_segment_);
    CHECK_RETURN_STATUS(s);

    auto new_segment =
        Segment::CreateAndOpen(path_, *new_schema, allocate_segment_id(),
                               writing_segment_->meta()->max_doc_id() + 1,
                               id_map_, delete_store_, version_manager_,
                               SegmentOptions{false, options_.enable_mmap_,
                                              options_.max_buffer_size_});
    if (!new_segment) {
      return new_segment.error();
    }

    s = new_version.add_persisted_segment_meta(writing_segment_->meta());
    CHECK_RETURN_STATUS(s);

    writing_segment_ = new_segment.value();
    new_version.set_next_segment_id(segment_id_allocator_.load());

  } else {
    // recreate writing segment
    s = writing_segment_->destroy();
    CHECK_RETURN_STATUS(s);
    auto id = writing_segment_->id();
    auto min_doc_id = writing_segment_->meta()->min_doc_id();
    writing_segment_.reset();
    SegmentOptions seg_options;
    seg_options.enable_mmap_ = options_.enable_mmap_;
    seg_options.max_buffer_size_ = options_.max_buffer_size_;
    seg_options.read_only_ = options_.read_only_;
    auto writing_segment =
        Segment::CreateAndOpen(path_, *new_schema, id, min_doc_id, id_map_,
                               delete_store_, version_manager_, seg_options);
    if (!writing_segment) {
      return writing_segment.error();
    }

    writing_segment_ = writing_segment.value();
  }
  new_version.reset_writing_segment_meta(writing_segment_->meta());

  auto persist_semgents = get_all_persist_segments();

  std::vector<SegmentTask::Ptr> tasks;
  if (is_vector_field) {
    tasks = build_drop_vector_index_task(persist_semgents, column_name);
  } else {
    tasks = build_drop_scalar_index_task(persist_semgents, column_name);
  }

  if (tasks.empty()) {
    new_version.set_schema(*new_schema);

    s = version_manager_->apply(new_version);
    CHECK_RETURN_STATUS(s);

    // persist manifest
    s = version_manager_->flush();
    CHECK_RETURN_STATUS(s);

    schema_ = new_schema;
    return Status::OK();
  }

  s = execute_tasks(tasks);
  CHECK_RETURN_STATUS(s);

  new_version.set_schema(*new_schema);

  for (auto &task : tasks) {
    auto task_info = task->task_info();

    if (std::holds_alternative<DropVectorIndexTask>(task_info)) {
      auto drop_index_task = std::get<DropVectorIndexTask>(task_info);
      s = new_version.update_persisted_segment_meta(
          drop_index_task.output_segment_meta_);
    } else if (std::holds_alternative<DropScalarIndexTask>(task_info)) {
      auto drop_index_task = std::get<DropScalarIndexTask>(task_info);
      s = new_version.update_persisted_segment_meta(
          drop_index_task.output_segment_meta_);
    }
    CHECK_RETURN_STATUS(s);
  }

  s = version_manager_->apply(new_version);
  CHECK_RETURN_STATUS(s);

  // persist manifest
  s = version_manager_->flush();
  CHECK_RETURN_STATUS(s);

  // 4. remove old segments or block
  for (auto &task : tasks) {
    auto task_info = task->task_info();

    if (std::holds_alternative<DropVectorIndexTask>(task_info)) {
      auto drop_index_task = std::get<DropVectorIndexTask>(task_info);
      s = drop_index_task.input_segment_->reload_vector_index(
          *new_schema, drop_index_task.output_segment_meta_,
          drop_index_task.output_vector_indexers_);
    } else if (std::holds_alternative<DropScalarIndexTask>(task_info)) {
      auto drop_index_task = std::get<DropScalarIndexTask>(task_info);
      s = drop_index_task.input_segment_->reload_scalar_index(
          *new_schema, drop_index_task.output_segment_meta_,
          drop_index_task.output_scalar_indexer_);
    }
    CHECK_RETURN_STATUS(s);
  }

  schema_ = new_schema;

  return Status::OK();
}

std::vector<SegmentTask::Ptr> CollectionImpl::build_drop_vector_index_task(
    const std::vector<Segment::Ptr> &segments, const std::string &column) {
  std::vector<SegmentTask::Ptr> tasks;
  for (auto &segment : segments) {
    tasks.emplace_back(SegmentTask::CreateDropVectorIndexTask(
        DropVectorIndexTask{segment, column}));
  }
  return tasks;
}

std::vector<SegmentTask::Ptr> CollectionImpl::build_drop_scalar_index_task(
    const std::vector<Segment::Ptr> &segments, const std::string &column) {
  std::vector<SegmentTask::Ptr> tasks;
  for (auto &segment : segments) {
    tasks.emplace_back(SegmentTask::CreateDropScalarIndexTask(
        DropScalarIndexTask(segment, {column})));
  }
  return tasks;
}

Status CollectionImpl::Optimize(const OptimizeOptions &options) {
  CHECK_COLLECTION_READONLY_RETURN_STATUS;

  std::lock_guard lock(schema_handle_mtx_);
  // when optimizing, schema operations(include another optimize) are not
  // allowed

  CHECK_DESTROY_RETURN_STATUS(destroyed_, false);

  std::vector<Segment::Ptr> persist_segments;

  {
    // forbidden writing for a while
    std::lock_guard write_lock(write_mtx_);

    if (writing_segment_->doc_count() != 0) {
      // flush and create new segment
      auto s = switch_to_new_segment_for_writing();
      if (!s.ok()) {
        return s;
      }
    }

    persist_segments =
        get_all_persist_segments();  // will not return writing segment
    // after leave this scope, writing action is allowed
  }

  if (persist_segments.size() == 0) {
    // no need to optimize
    return Status::OK();
  }

  // build segment compact task
  auto delete_store_clone = delete_store_->clone();
  auto tasks =
      build_compact_task(schema_, persist_segments, options.concurrency_,
                         delete_store_clone->make_filter());

  // execute segment compact task
  auto s = execute_compact_task(tasks);
  CHECK_RETURN_STATUS(s);

  {
    // forbidden writing for updating version
    // writing action may trigger updating version where confict occurs
    std::lock_guard write_lock(write_mtx_);

    Version new_version = version_manager_->get_current_version();

    for (auto &task : tasks) {
      auto task_info = task->task_info();

      if (std::holds_alternative<CompactTask>(task_info)) {
        auto compact_task = std::get<CompactTask>(task_info);

        // 0. check if has output segment meta
        if (compact_task.output_segment_meta_) {
          // 1. rename built tmp segments
          auto tmp_segment_id = compact_task.output_segment_id_;
          auto tmp_segment_path =
              FileHelper::MakeTempSegmentPath(path_, tmp_segment_id);

          auto new_segment_id = allocate_segment_id();
          auto new_segment_path =
              FileHelper::MakeSegmentPath(path_, new_segment_id);

          if (!FileHelper::MoveDirectory(tmp_segment_path, new_segment_path)) {
            return Status::InternalError("move segment directory failed");
          }

          // update output_segment_meta_'s segment id
          compact_task.output_segment_meta_->set_id(new_segment_id);

          s = new_version.add_persisted_segment_meta(
              compact_task.output_segment_meta_);
          CHECK_RETURN_STATUS(s);
          new_version.set_next_segment_id(segment_id_allocator_.load());
        }

        for (auto input_segment : compact_task.input_segments_) {
          s = new_version.remove_persisted_segment_meta(input_segment->id());
          CHECK_RETURN_STATUS(s);
        }
      } else if (std::holds_alternative<CreateVectorIndexTask>(task_info)) {
        auto create_index_task = std::get<CreateVectorIndexTask>(task_info);
        s = new_version.update_persisted_segment_meta(
            create_index_task.output_segment_meta_);
        CHECK_RETURN_STATUS(s);
      }
    }

    // 2. update version
    s = version_manager_->apply(new_version);
    CHECK_RETURN_STATUS(s);

    // 3. persist version
    s = version_manager_->flush();
    CHECK_RETURN_STATUS(s);

    // 4. remove old segments or block
    for (auto &task : tasks) {
      auto task_info = task->task_info();

      if (std::holds_alternative<CompactTask>(task_info)) {
        auto compact_task = std::get<CompactTask>(task_info);

        if (compact_task.output_segment_meta_) {
          auto new_segment =
              Segment::Open(path_, *schema_, *compact_task.output_segment_meta_,
                            id_map_, delete_store_, version_manager_,
                            SegmentOptions{true, options_.enable_mmap_});
          if (!new_segment.has_value()) {
            return new_segment.error();
          }
          s = segment_manager_->add_segment(new_segment.value());
          CHECK_RETURN_STATUS(s);
        }

        for (auto input_segment : compact_task.input_segments_) {
          s = segment_manager_->destroy_segment(input_segment->id());
          CHECK_RETURN_STATUS(s);
        }
      } else if (std::holds_alternative<CreateVectorIndexTask>(task_info)) {
        auto create_index_task = std::get<CreateVectorIndexTask>(task_info);

        s = create_index_task.input_segment_->reload_vector_index(
            *schema_, create_index_task.output_segment_meta_,
            create_index_task.output_vector_indexers_,
            create_index_task.output_quant_vector_indexers_);
        CHECK_RETURN_STATUS(s);
      }
    }
  }

  return Status::OK();
}

std::vector<SegmentTask::Ptr> CollectionImpl::build_compact_task(
    const CollectionSchema::Ptr &schema,
    const std::vector<Segment::Ptr> &segments, int concurrency,
    const IndexFilter::Ptr filter) {
  std::vector<SegmentTask::Ptr> tasks;
  if (segments.empty()) return tasks;

  bool rebuild = false;
  size_t current_doc_count = 0;
  size_t current_actual_doc_count = 0;
  for (auto &segment : segments) {
    current_doc_count += segment->doc_count();
    current_actual_doc_count += segment->doc_count(filter);
  }
  if (current_actual_doc_count <
      current_doc_count * (1 - COMPACT_DELETE_RATIO_THRESHOLD)) {
    // if delete ratio is large enough, rebuild
    rebuild = true;
  }

  auto max_doc_count_per_segment = schema->max_doc_count_per_segment();

  std::vector<Segment::Ptr> current_group;
  current_doc_count = 0;
  current_actual_doc_count = 0;

  for (const auto &seg : segments) {
    uint64_t doc_count = seg->doc_count();
    uint64_t actual_doc_count = seg->doc_count(filter);

    if (!current_group.empty()) {
      SegmentTask::Ptr task;
      bool skip_task{false};
      if (rebuild) {
        if (current_actual_doc_count + actual_doc_count >
            max_doc_count_per_segment) {
          // only create SegmentCompactTask when rebuild=true
          task = SegmentTask::CreateComapctTask(
              CompactTask{path_, schema, current_group,
                          allocate_segment_id_for_tmp_segment(), filter,
                          !options_.enable_mmap_, concurrency});
        }
      } else {
        if (current_doc_count + doc_count > max_doc_count_per_segment) {
          // check current_group size
          if (current_group.size() == 1) {
            task =
                SegmentTask::CreateCreateVectorIndexTask(CreateVectorIndexTask{
                    current_group[0], "", nullptr, concurrency});
            skip_task = current_group[0]->all_vector_index_ready();
          } else {
            task = SegmentTask::CreateComapctTask(
                CompactTask{path_, schema, current_group,
                            allocate_segment_id_for_tmp_segment(), nullptr,
                            !options_.enable_mmap_, concurrency});
          }
        }
      }

      if (task) {
        current_group.clear();
        current_doc_count = 0;
        current_actual_doc_count = 0;
        if (!skip_task) {
          tasks.push_back(task);
        }
      }
    }

    current_group.push_back(seg);
    current_doc_count += doc_count;
    current_actual_doc_count += actual_doc_count;
  }

  if (current_group.size() > 0) {
    SegmentTask::Ptr task;
    if (current_group.size() == 1 && !rebuild) {
      task = SegmentTask::CreateCreateVectorIndexTask(
          CreateVectorIndexTask{current_group[0], "", nullptr, concurrency});
    } else {
      task = SegmentTask::CreateComapctTask(CompactTask{
          path_, schema, current_group, allocate_segment_id_for_tmp_segment(),
          rebuild ? filter : nullptr, !options_.enable_mmap_, concurrency});
    }
    tasks.push_back(task);
  }

  return tasks;
}

Status CollectionImpl::execute_compact_task(
    std::vector<SegmentTask::Ptr> &tasks) const {
  Status s;
  for (auto &task : tasks) {
    s = SegmentHelper::Execute(task);
    if (!s.ok()) {
      return s;
    }
  }

  return Status::OK();
}

Status CollectionImpl::validate(const std::string &column,
                                const FieldSchema::Ptr &schema,
                                const std::string &expression,
                                const std::string &rename, ColumnOp op) {
  auto check_data_type = [&](const FieldSchema *field) -> Status {
    if (field->data_type() < DataType::INT32 ||
        field->data_type() > DataType::DOUBLE) {
      return Status::InvalidArgument(
          "Only support basic numeric data type [int32, int64, uint32, uint64, "
          "float, double]: ",
          field->to_string());
    }
    return Status::OK();
  };

  switch (op) {
    case ColumnOp::ADD: {
      if (schema == nullptr) {
        return Status::InvalidArgument("Column schema is null");
      }

      if (schema->name().empty()) {
        return Status::InvalidArgument("Column name is empty");
      }
      if (schema_->has_field(schema->name())) {
        return Status::InvalidArgument("column already exists");
      }

      auto s = schema->validate();
      CHECK_RETURN_STATUS(s);

      s = check_data_type(schema.get());
      CHECK_RETURN_STATUS(s);

      if (expression.empty() && !schema->nullable()) {
        return Status::InvalidArgument(
            "Add column is not supported for non-nullable column");
      }

      break;
    }
    case ColumnOp::ALTER: {
      if (column.empty()) {
        return Status::InvalidArgument("column name is empty");
      }

      if (!schema_->has_field(column)) {
        return Status::InvalidArgument("column ", column, " not found");
      }

      if (!rename.empty() && schema) {
        return Status::InvalidArgument(
            "cannot specify both rename and new column schema");
      }

      auto *old_field_schema = schema_->get_field(column);
      auto s = check_data_type(old_field_schema);
      CHECK_RETURN_STATUS(s);

      if (!rename.empty()) {
        // rename case
        if (schema_->has_field(rename)) {
          return Status::InvalidArgument("new column name ", rename,
                                         " already exists");
        }
      } else {
        // schema change case
        if (!schema) {
          return Status::InvalidArgument("New column schema is null");
        }

        s = schema->validate();
        CHECK_RETURN_STATUS(s);

        if (schema->name().empty()) {
          return Status::InvalidArgument("new column schema name is empty");
        }

        if (!schema->nullable() && old_field_schema->nullable()) {
          return Status::InvalidArgument(
              "new column schema is not nullable, but old column schema is "
              "nullable");
        }

        if (*old_field_schema == *schema) {
          // equal schema
          return Status::OK();
        }

        s = check_data_type(schema.get());
        CHECK_RETURN_STATUS(s);
      }

      break;
    }
    case ColumnOp::DROP: {
      if (!schema_->has_field(column)) {
        return Status::InvalidArgument("Column not exists: ", column);
      }

      auto *old_field_schema = schema_->get_field(column);
      auto s = check_data_type(old_field_schema);
      CHECK_RETURN_STATUS(s);
      break;
    }
    default:
      break;
  }

  return Status::OK();
}

Status CollectionImpl::AddColumn(const FieldSchema::Ptr &column_schema,
                                 const std::string &expression,
                                 const AddColumnOptions &options) {
  CHECK_COLLECTION_READONLY_RETURN_STATUS;

  std::lock_guard lock(schema_handle_mtx_);

  CHECK_DESTROY_RETURN_STATUS(destroyed_, false);

  // validate
  auto s = validate("", column_schema, expression, "", ColumnOp::ADD);
  CHECK_RETURN_STATUS(s);

  // forbidden writing until index is ready
  std::lock_guard write_lock(write_mtx_);

  auto new_schema = std::make_shared<CollectionSchema>(*schema_);
  s = new_schema->add_field(column_schema);
  CHECK_RETURN_STATUS(s);

  if (writing_segment_->doc_count() > 0) {
    s = switch_to_new_segment_for_writing();
    CHECK_RETURN_STATUS(s);
  }

  Version new_version = version_manager_->get_current_version();

  // add column on segment manager
  s = segment_manager_->add_column(column_schema, expression,
                                   options.concurrency_);
  CHECK_RETURN_STATUS(s);

  // reset writing segment with new schema
  auto id = writing_segment_->id();
  auto min_doc_id = writing_segment_->meta()->min_doc_id();

  s = writing_segment_->destroy();
  CHECK_RETURN_STATUS(s);
  writing_segment_.reset();

  SegmentOptions seg_options;
  seg_options.enable_mmap_ = options_.enable_mmap_;
  seg_options.max_buffer_size_ = options_.max_buffer_size_;
  seg_options.read_only_ = options_.read_only_;
  auto writing_segment =
      Segment::CreateAndOpen(path_, *new_schema, id, min_doc_id, id_map_,
                             delete_store_, version_manager_, seg_options);
  if (!writing_segment) {
    return writing_segment.error();
  }
  writing_segment_ = writing_segment.value();

  // update new version
  new_version.set_schema(*new_schema);
  new_version.reset_writing_segment_meta(writing_segment_->meta());

  auto new_segment_metas = segment_manager_->get_segments_meta();
  for (auto meta : new_segment_metas) {
    s = new_version.update_persisted_segment_meta(meta);
    CHECK_RETURN_STATUS(s);
  }

  s = version_manager_->apply(new_version);
  CHECK_RETURN_STATUS(s);

  // persist manifest
  s = version_manager_->flush();
  CHECK_RETURN_STATUS(s);

  schema_ = new_schema;

  return Status::OK();
}

Status CollectionImpl::DropColumn(const std::string &column_name) {
  CHECK_COLLECTION_READONLY_RETURN_STATUS;

  std::lock_guard lock(schema_handle_mtx_);

  CHECK_DESTROY_RETURN_STATUS(destroyed_, false);

  // validate
  auto s = validate(column_name, nullptr, "", "", ColumnOp::DROP);
  CHECK_RETURN_STATUS(s);

  // forbidden writing until index is ready
  std::lock_guard write_lock(write_mtx_);

  auto new_schema = std::make_shared<CollectionSchema>(*schema_);
  s = new_schema->drop_field(column_name);
  CHECK_RETURN_STATUS(s);

  if (writing_segment_->doc_count() > 0) {
    s = switch_to_new_segment_for_writing();
    CHECK_RETURN_STATUS(s);
  }

  Version new_version = version_manager_->get_current_version();

  // drop column on segment manager
  s = segment_manager_->drop_column(column_name);
  CHECK_RETURN_STATUS(s);

  // reset writing segment with new schema
  auto id = writing_segment_->id();
  auto min_doc_id = writing_segment_->meta()->min_doc_id();

  s = writing_segment_->destroy();
  CHECK_RETURN_STATUS(s);
  writing_segment_.reset();

  SegmentOptions seg_options;
  seg_options.enable_mmap_ = options_.enable_mmap_;
  seg_options.max_buffer_size_ = options_.max_buffer_size_;
  seg_options.read_only_ = options_.read_only_;
  auto writing_segment =
      Segment::CreateAndOpen(path_, *new_schema, id, min_doc_id, id_map_,
                             delete_store_, version_manager_, seg_options);
  if (!writing_segment) {
    return writing_segment.error();
  }
  writing_segment_ = writing_segment.value();

  // update new version
  new_version.set_schema(*new_schema);
  new_version.reset_writing_segment_meta(writing_segment_->meta());

  auto new_segment_metas = segment_manager_->get_segments_meta();
  for (auto meta : new_segment_metas) {
    s = new_version.update_persisted_segment_meta(meta);
    CHECK_RETURN_STATUS(s);
  }

  s = version_manager_->apply(new_version);
  CHECK_RETURN_STATUS(s);

  // persist manifest
  s = version_manager_->flush();
  CHECK_RETURN_STATUS(s);

  schema_ = new_schema;

  return Status::OK();
}

Status CollectionImpl::AlterColumn(const std::string &column_name,
                                   const std::string &rename,
                                   const FieldSchema::Ptr &new_column_schema,
                                   const AlterColumnOptions &options) {
  CHECK_COLLECTION_READONLY_RETURN_STATUS;

  std::lock_guard lock(schema_handle_mtx_);

  CHECK_DESTROY_RETURN_STATUS(destroyed_, false);

  // validate
  auto s =
      validate(column_name, new_column_schema, "", rename, ColumnOp::ALTER);
  CHECK_RETURN_STATUS(s);

  // forbidden writing until index is ready
  std::lock_guard write_lock(write_mtx_);

  std::shared_ptr<FieldSchema> new_field_schema{nullptr};
  if (!rename.empty()) {
    new_field_schema =
        std::make_shared<FieldSchema>(*schema_->get_field(column_name));
    new_field_schema->set_name(rename);
  } else {
    new_field_schema = std::make_shared<FieldSchema>(*new_column_schema);
  }

  auto new_schema = std::make_shared<CollectionSchema>(*schema_);
  s = new_schema->alter_field(column_name, new_field_schema);
  CHECK_RETURN_STATUS(s);

  if (writing_segment_->doc_count() > 0) {
    s = switch_to_new_segment_for_writing();
    CHECK_RETURN_STATUS(s);
  }

  Version new_version = version_manager_->get_current_version();

  // alter column on segment manager
  s = segment_manager_->alter_column(column_name, new_field_schema,
                                     options.concurrency_);
  CHECK_RETURN_STATUS(s);

  // reset writing segment with new schema
  auto id = writing_segment_->id();
  auto min_doc_id = writing_segment_->meta()->min_doc_id();

  s = writing_segment_->destroy();
  CHECK_RETURN_STATUS(s);
  writing_segment_.reset();

  SegmentOptions seg_options;
  seg_options.enable_mmap_ = options_.enable_mmap_;
  seg_options.max_buffer_size_ = options_.max_buffer_size_;
  seg_options.read_only_ = options_.read_only_;
  auto writing_segment =
      Segment::CreateAndOpen(path_, *new_schema, id, min_doc_id, id_map_,
                             delete_store_, version_manager_, seg_options);
  if (!writing_segment) {
    return writing_segment.error();
  }
  writing_segment_ = writing_segment.value();

  // update new version
  new_version.set_schema(*new_schema);
  new_version.reset_writing_segment_meta(writing_segment_->meta());

  auto new_segment_metas = segment_manager_->get_segments_meta();
  for (auto meta : new_segment_metas) {
    s = new_version.update_persisted_segment_meta(meta);
    CHECK_RETURN_STATUS(s);
  }

  s = version_manager_->apply(new_version);
  CHECK_RETURN_STATUS(s);

  // persist manifest
  s = version_manager_->flush();
  CHECK_RETURN_STATUS(s);

  schema_ = new_schema;

  return Status::OK();
}

Result<WriteResults> CollectionImpl::Insert(std::vector<Doc> &docs) {
  return write_impl(docs, WriteMode::INSERT);
}

Result<WriteResults> CollectionImpl::Update(std::vector<Doc> &docs) {
  return write_impl(docs, WriteMode::UPDATE);
}

Result<WriteResults> CollectionImpl::Upsert(std::vector<Doc> &docs) {
  return write_impl(docs, WriteMode::UPSERT);
}

Status CollectionImpl::internal_fetch_by_doc(const Doc &doc,
                                             Doc::Ptr *doc_out) {
  auto segments = get_all_segments();
  uint64_t doc_id;
  bool has = id_map_->has(doc.pk(), &doc_id);
  if (!has) {
    return Status::NotFound("Document not found");
  }
  if (delete_store_->is_deleted(doc_id)) {
    return Status::NotFound("Document already deleted");
  }

  auto segment = local_segment_by_doc_id(doc_id, segments);
  if (!segment) {
    LOG_WARN("doc_id: %zu segment not found", (size_t)doc_id);
    return Status::InternalError("Segment not found");
  }

  auto old_doc = segment->Fetch(doc_id);
  if (!old_doc) {
    LOG_WARN("doc_id: %zu fetch doc failed", (size_t)doc_id);
    return Status::InternalError("Fetch doc failed");
  }
  *doc_out = old_doc;
  return Status::OK();
}

Status CollectionImpl::handle_upsert(Doc &doc) {
  return writing_segment_->Upsert(doc);
}

Status CollectionImpl::handle_update(Doc &doc) {
  Doc::Ptr old_doc{nullptr};
  auto s = internal_fetch_by_doc(doc, &old_doc);
  CHECK_RETURN_STATUS(s);

  old_doc->merge(doc);
  return writing_segment_->Update(*old_doc);
}

Status CollectionImpl::handle_insert(Doc &doc) {
  return writing_segment_->Insert(doc);
}

Result<WriteResults> CollectionImpl::write_impl(std::vector<Doc> &docs,
                                                WriteMode mode) {
  CHECK_READONLY_RETURN_STATUS_EXPECTED();

  std::shared_lock lock(schema_handle_mtx_);

  CHECK_DESTROY_RETURN_STATUS_EXPECTED(destroyed_, false);

  for (auto &&doc : docs) {
    auto validate = doc.validate(schema_, mode == WriteMode::UPDATE);
    CHECK_RETURN_STATUS_EXPECTED(validate);
  }

  // TODO: The granularity of the write_lock is too coarse.
  std::lock_guard write_lock(write_mtx_);

  WriteResults results;
  // validate write batch size
  if (docs.size() > kMaxWriteBatchSize) {
    CHECK_RETURN_STATUS_EXPECTED(Status::InvalidArgument("Too many docs"));
  }

  // validate docs
  for (auto &&doc : docs) {
    if (need_switch_to_new_segment()) {
      auto s = switch_to_new_segment_for_writing();
      CHECK_RETURN_STATUS_EXPECTED(s);
    }

    Status s;

    switch (mode) {
      case WriteMode::UPSERT:
        s = handle_upsert(doc);
        break;
      case WriteMode::UPDATE:
        s = handle_update(doc);
        break;
      case WriteMode::INSERT:
        s = handle_insert(doc);
        break;
      default:
        s = Status::InvalidArgument("Invalid write mode");
    }

    results.push_back(s);
  }

  return results;
}

bool CollectionImpl::need_switch_to_new_segment() const {
  return writing_segment_->doc_count() >= schema_->max_doc_count_per_segment();
}

Status CollectionImpl::switch_to_new_segment_for_writing(
    const CollectionSchema::Ptr &schema) {
  auto s = writing_segment_->dump();
  CHECK_RETURN_STATUS(s);

  s = segment_manager_->add_segment(writing_segment_);
  CHECK_RETURN_STATUS(s);

  // when create new segment, segment meta should create a first new block
  // meta
  auto new_segment = Segment::CreateAndOpen(
      path_, schema == nullptr ? *schema_ : *schema, allocate_segment_id(),
      writing_segment_->meta()->max_doc_id() + 1, id_map_, delete_store_,
      version_manager_,
      SegmentOptions{false, options_.enable_mmap_, options_.max_buffer_size_});
  if (!new_segment) {
    return new_segment.error();
  }

  Version version = version_manager_->get_current_version();
  auto writing_segment_meta = writing_segment_->meta();
  writing_segment_meta->remove_writing_forward_block();
  s = version.add_persisted_segment_meta(writing_segment_meta);
  CHECK_RETURN_STATUS(s);

  writing_segment_ = new_segment.value();
  version.reset_writing_segment_meta(writing_segment_->meta());
  version.set_next_segment_id(segment_id_allocator_.load());

  s = version_manager_->apply(version);
  CHECK_RETURN_STATUS(s);
  s = version_manager_->flush();
  CHECK_RETURN_STATUS(s);

  return Status::OK();
}

Result<WriteResults> CollectionImpl::Delete(
    const std::vector<std::string> &pks) {
  CHECK_READONLY_RETURN_STATUS_EXPECTED();

  std::shared_lock lock(schema_handle_mtx_);

  CHECK_DESTROY_RETURN_STATUS_EXPECTED(destroyed_, false);

  // TODO: The granularity of the write_lock is too coarse.
  std::lock_guard write_lock(write_mtx_);
  WriteResults results;
  for (auto &&pk : pks) {
    Status s = writing_segment_->Delete(pk);
    results.push_back(s);
  }

  return results;
}

Status CollectionImpl::DeleteByFilter(const std::string &filter) {
  CHECK_COLLECTION_READONLY_RETURN_STATUS;

  std::shared_lock lock(schema_handle_mtx_);

  CHECK_DESTROY_RETURN_STATUS(destroyed_, false);

  auto segments = get_all_segments();

  VectorQuery query;
  query.filter_ = filter;
  query.topk_ = INT32_MAX;
  query.output_fields_ = std::vector<std::string>{};
  query.include_doc_id_ = true;

  auto ret = sql_engine_->execute(schema_, query, get_all_segments());
  if (!ret.has_value()) {
    return ret.error();
  }

  // TODO: The granularity of the write_lock is too coarse.
  std::lock_guard write_lock(write_mtx_);
  for (auto &doc : ret.value()) {
    Status s = writing_segment_->Delete(doc->doc_id());
    if (!s.ok()) {
      LOG_ERROR("Delete doc_id failed");
      return s;
    }
  }

  return Status::OK();
}

Result<DocPtrList> CollectionImpl::Query(const VectorQuery &query) const {
  std::shared_lock lock(schema_handle_mtx_);

  CHECK_DESTROY_RETURN_STATUS_EXPECTED(destroyed_, false);

  auto s = query.validate(schema_->get_vector_field(query.field_name_));
  CHECK_RETURN_STATUS_EXPECTED(s);

  auto segments = get_all_segments();
  if (segments.empty()) {
    return DocPtrList();
  }

  return sql_engine_->execute(schema_, query, segments);
}

Result<GroupResults> CollectionImpl::GroupByQuery(
    const GroupByVectorQuery &query) const {
  std::shared_lock lock(schema_handle_mtx_);

  CHECK_DESTROY_RETURN_STATUS_EXPECTED(destroyed_, false);

  auto segments = get_all_segments();
  if (segments.empty()) {
    return GroupResults();
  }

  return sql_engine_->execute_group_by(schema_, query, segments);
}

Result<DocPtrMap> CollectionImpl::Fetch(
    const std::vector<std::string> &pks) const {
  std::shared_lock lock(schema_handle_mtx_);

  CHECK_DESTROY_RETURN_STATUS_EXPECTED(destroyed_, false);

  auto segments = get_all_segments();

  DocPtrMap results;

  for (auto &pk : pks) {
    uint64_t doc_id;
    bool has = id_map_->has(pk, &doc_id);
    if (!has) {
      results.insert({pk, nullptr});
      continue;
    }
    if (delete_store_->is_deleted(doc_id)) {
      results.insert({pk, nullptr});
      continue;
    }
    auto segment = local_segment_by_doc_id(doc_id, segments);
    if (!segment) {
      LOG_WARN("doc_id: %zu segment not found", (size_t)doc_id);
      results.insert({pk, nullptr});
      continue;
    }
    results.insert({pk, segment->Fetch(doc_id)});
  }

  return results;
}

Status CollectionImpl::recovery() {
  if (!FileHelper::DirectoryExists(path_.c_str())) {
    return Status::InvalidArgument("collection path{", path_, "} not exist.");
  }

  // get lock file
  auto s = acquire_file_lock(false);
  CHECK_RETURN_STATUS(s);

  // recovery version first
  auto version_manager = VersionManager::Recovery(path_);
  if (!version_manager.has_value()) {
    return version_manager.error();
  }

  version_manager_ = version_manager.value();
  const auto v = version_manager_->get_current_version();
  schema_ = std::make_shared<CollectionSchema>(v.schema());
  options_.enable_mmap_ = v.enable_mmap();
  s = recover_idmap_and_delete_store();
  CHECK_RETURN_STATUS(s);

  // recover persist segments
  segment_manager_ = std::make_shared<SegmentManager>();

  auto segment_metas = v.persisted_segment_metas();

  SegmentOptions seg_options;
  seg_options.read_only_ = true;
  seg_options.enable_mmap_ = options_.enable_mmap_;
  for (size_t i = 0; i < segment_metas.size(); ++i) {
    auto segment = Segment::Open(path_, *schema_, *segment_metas[i], id_map_,
                                 delete_store_, version_manager_, seg_options);
    if (!segment) {
      return segment.error();
    }

    segment_manager_->add_segment(segment.value());
  }

  seg_options.read_only_ = options_.read_only_;
  seg_options.max_buffer_size_ = options_.max_buffer_size_;

  // recover writing segment
  auto writing_segment =
      Segment::Open(path_, *schema_, *v.writing_segment_meta(), id_map_,
                    delete_store_, version_manager_, seg_options);
  if (!writing_segment) {
    return writing_segment.error();
  }

  writing_segment_ = writing_segment.value();
  segment_id_allocator_.store(v.next_segment_id());

  // recover id map & delete store
  return Status::OK();
}

Status CollectionImpl::recover_idmap_and_delete_store() {
  const auto v = version_manager_->get_current_version();

  // idmap
  std::string idmap_path =
      FileHelper::MakeFilePath(path_, FileID::ID_FILE, v.id_map_path_suffix());
  id_map_ = IDMap::CreateAndOpen(schema_->name(), idmap_path, false,
                                 options_.read_only_);
  if (!id_map_) {
    return Status::InternalError("recovery idmap failed");
  }

  // delete store
  std::string delete_store_path = FileHelper::MakeFilePath(
      path_, FileID::DELETE_FILE, v.delete_snapshot_path_suffix());
  delete_store_ =
      DeleteStore::CreateAndLoad(schema_->name(), delete_store_path);
  if (!delete_store_) {
    return Status::InternalError("recovery delete store failed");
  }

  return Status::OK();
}

Status CollectionImpl::create() {
  // check path
  if (path_.empty()) {
    return Status::InvalidArgument("path validate failed: path is empty");
  }
  if (!std::regex_match(path_, COLLECTION_PATH_REGEX)) {
    return Status::InvalidArgument("path validate failed: path[", path_,
                                   "] cannot pass the regex verification");
  }
  if (ailego::FileHelper::IsExist(path_.c_str())) {
    return Status::InvalidArgument("path validate failed: path[", path_,
                                   "] is existed");
  }

  // check schema
  auto s = schema_->validate();
  CHECK_RETURN_STATUS(s);

  if (!ailego::FileHelper::MakePath(path_.c_str())) {
    return Status::InvalidArgument("create collection path failed: ", path_,
                                   ", error: ", strerror(errno));
  }

  // init lock file
  s = acquire_file_lock(true);
  CHECK_RETURN_STATUS(s);

  // init idmap & delete store
  s = create_idmap_and_delete_store();
  CHECK_RETURN_STATUS(s);

  // init version manager
  s = init_version_manager();
  CHECK_RETURN_STATUS(s);

  // create segment
  s = init_writing_segment();
  CHECK_RETURN_STATUS(s);

  // init version
  Version version;
  version.set_schema(*schema_);
  version.set_enable_mmap(options_.enable_mmap_);
  version.reset_writing_segment_meta(writing_segment_->meta());
  version.set_id_map_path_suffix(0);
  version.set_delete_snapshot_path_suffix(0);
  version.set_next_segment_id(1);

  version_manager_->apply(version);
  s = version_manager_->flush();
  CHECK_RETURN_STATUS(s);

  segment_id_allocator_.store(1);
  segment_manager_ = std::make_unique<SegmentManager>();

  return Status::OK();
}

Status CollectionImpl::create_idmap_and_delete_store() {
  // idmap
  std::string idmap_path = FileHelper::MakeFilePath(path_, FileID::ID_FILE, 0);
  id_map_ = IDMap::CreateAndOpen(schema_->name(), idmap_path, true,
                                 options_.read_only_);
  if (!id_map_) {
    return Status::InternalError("create id map failed");
  }

  std::string delete_store_path =
      FileHelper::MakeFilePath(path_, FileID::DELETE_FILE, 0);
  delete_store_ = std::make_shared<DeleteStore>(schema_->name());
  // when first create collection, delete store will flush a empty snapshot
  delete_store_->flush(delete_store_path);

  return Status::OK();
}

Status CollectionImpl::init_version_manager() {
  // use empty version to init version manager
  auto version_manager = VersionManager::Create(path_, Version{});
  if (!version_manager.has_value()) {
    return version_manager.error();
  }

  version_manager_ = version_manager.value();
  return Status::OK();
}

Status CollectionImpl::init_writing_segment() {
  SegmentOptions options;
  options.enable_mmap_ = options_.enable_mmap_;
  options.max_buffer_size_ = options_.max_buffer_size_;
  options.read_only_ = options_.read_only_;

  auto writing_segment = Segment::CreateAndOpen(
      path_, *schema_, 0, 0, id_map_, delete_store_, version_manager_, options);

  if (!writing_segment) {
    return writing_segment.error();
  }

  writing_segment_ = writing_segment.value();

  return Status::OK();
}

Status CollectionImpl::acquire_file_lock(bool create) {
  std::string lock_file_path = ailego::StringHelper::Concat(path_, "/", "LOCK");

  if (create) {
    if (!lock_file_.create(lock_file_path.c_str(), 0)) {
      return Status::InternalError("Can't create lock file");
    }
  } else {
    if (!lock_file_.open(lock_file_path.c_str(), false)) {
      return Status::InternalError("Can't open lock file");
    }
  }

  if (options_.read_only_) {
    if (!ailego::FileLock::TryLockShared(lock_file_.native_handle())) {
      return Status::InternalError("Can't lock read-only collection");
    }
  } else {
    if (!ailego::FileLock::TryLock(lock_file_.native_handle())) {
      return Status::InternalError("Can't lock read-write collection");
    }
  }

  return Status::OK();
}

Segment::Ptr CollectionImpl::local_segment_by_doc_id(
    uint64_t doc_id, const std::vector<Segment::Ptr> &segments) const {
  size_t left = 0;
  size_t right = segments.size();

  while (left < right) {
    size_t mid = left + (right - left) / 2;
    uint64_t min_id = segments[mid]->meta()->min_doc_id();
    uint64_t max_id = segments[mid]->meta()->max_doc_id();

    if (doc_id < min_id) {
      right = mid;
    } else if (doc_id > max_id) {
      left = mid + 1;
    } else {
      return segments[mid];
    }
  }

  return nullptr;
}

std::vector<Segment::Ptr> CollectionImpl::get_all_segments() const {
  std::vector<Segment::Ptr> segments = get_all_persist_segments();
  if (writing_segment_->doc_count() > 0) {
    segments.push_back(writing_segment_);
  }
  return segments;
}

std::vector<Segment::Ptr> CollectionImpl::get_all_persist_segments() const {
  return segment_manager_->get_segments();
}

}  // namespace zvec