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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <arrow/io/file.h>
#include <arrow/ipc/reader.h>
#include <arrow/result.h>
#include <gtest/gtest.h>
#include "db/common/file_helper.h"
#include "db/index/common/delete_store.h"
#include "db/index/common/id_map.h"
#include "db/index/common/version_manager.h"
#include "utils/utils.h"
#include "zvec/db/options.h"
#include "segment_test_fixture.h"

using namespace zvec;

namespace {

arrow::Result<std::vector<int64_t>> ReadIpcRecordBatchLengths(
    const std::string &path) {
  ARROW_ASSIGN_OR_RAISE(auto input, arrow::io::ReadableFile::Open(path));
  ARROW_ASSIGN_OR_RAISE(auto reader,
                        arrow::ipc::RecordBatchFileReader::Open(input));

  std::vector<int64_t> batch_lengths;
  batch_lengths.reserve(reader->num_record_batches());
  for (int i = 0; i < reader->num_record_batches(); ++i) {
    ARROW_ASSIGN_OR_RAISE(auto batch, reader->ReadRecordBatch(i));
    batch_lengths.push_back(batch->num_rows());
  }
  return batch_lengths;
}

class AddColumnBatchBoundariesTest : public SegmentTest {};

TEST_P(AddColumnBatchBoundariesTest,
       NullableColumnPreservesRowsAcrossDifferentBatchBoundaries) {
  id_map_.reset();
  delete_store_.reset();
  version_manager_.reset();
  FileHelper::RemoveDirectory(col_path_);
  FileHelper::CreateDirectory(col_path_);

  // Keep the schema minimal so crossing the native 4096-row RecordBatch limit
  // remains fast while using the normal insert and dump path.
  schema_ = std::make_shared<CollectionSchema>(col_name_);
  schema_->add_field(
      std::make_shared<FieldSchema>("id", DataType::INT32, false));

  auto id_map_path = FileHelper::MakeFilePath(col_path_, FileID::ID_FILE, 0);
  id_map_ = IDMap::CreateAndOpen(col_name_, id_map_path, true, false);
  ASSERT_TRUE(id_map_ != nullptr);
  delete_store_ = std::make_shared<DeleteStore>(col_name_);

  Version initial_version;
  initial_version.set_schema(*schema_);
  initial_version.set_enable_mmap(true);
  auto version_manager_result =
      VersionManager::Create(col_path_, initial_version);
  ASSERT_TRUE(version_manager_result.has_value());
  version_manager_ = version_manager_result.value();

  options_.max_buffer_size_ = 16 * 1024 * 1024;
  constexpr int doc_count = 4097;
  auto segment = test::TestHelper::CreateSegmentWithDoc(
      col_path_, *schema_, 0, 0, id_map_, delete_store_, version_manager_,
      options_, 0, doc_count);
  ASSERT_TRUE(segment != nullptr);
  ASSERT_TRUE(segment->dump().ok());

  auto writing_segment_meta = segment->meta();
  std::vector<BlockID> base_scalar_block_ids;
  for (const auto &block : writing_segment_meta->persisted_blocks()) {
    if (block.type() == BlockType::SCALAR && block.contain_column("id")) {
      base_scalar_block_ids.push_back(block.id());
    }
  }
  ASSERT_EQ(base_scalar_block_ids.size(), 1);

  Version version = version_manager_->get_current_version();
  writing_segment_meta->remove_writing_forward_block();
  auto status = version.add_persisted_segment_meta(writing_segment_meta);
  ASSERT_TRUE(status.ok());
  status = version_manager_->apply(version);
  ASSERT_TRUE(status.ok());
  status = version_manager_->flush();
  ASSERT_TRUE(status.ok());

  segment.reset();
  version_manager_.reset();
  ASSERT_TRUE(id_map_->flush().ok());
  id_map_.reset();

  std::string delete_store_path =
      FileHelper::MakeFilePath(col_path_, FileID::DELETE_FILE, 0);
  ASSERT_TRUE(delete_store_->flush(delete_store_path).ok());
  delete_store_.reset();

  const auto base_scalar_path = FileHelper::MakeForwardBlockPath(
      col_path_, writing_segment_meta->id(), base_scalar_block_ids[0], false);
  auto base_batch_lengths = ReadIpcRecordBatchLengths(base_scalar_path);
  ASSERT_TRUE(base_batch_lengths.ok())
      << base_batch_lengths.status().ToString();
  // The base field is persisted in multiple batches, while add_column() below
  // writes one full-length nullable array into an independent scalar block.
  ASSERT_GT(base_batch_lengths->size(), 1);
  int64_t base_total_rows = 0;
  for (const auto batch_length : *base_batch_lengths) {
    base_total_rows += batch_length;
  }
  EXPECT_EQ(base_total_rows, doc_count);

  auto recover_version_manager = VersionManager::Recovery(col_path_);
  ASSERT_TRUE(recover_version_manager.has_value());
  auto recover_version_mgr = recover_version_manager.value();
  Version recovered_version = recover_version_mgr->get_current_version();
  const auto &persist_metas = recovered_version.persisted_segment_metas();
  ASSERT_EQ(persist_metas.size(), 1);

  std::string idmap_path = FileHelper::MakeFilePath(
      col_path_, FileID::ID_FILE, recovered_version.id_map_path_suffix());
  IDMap::Ptr recover_id_map = std::make_shared<IDMap>(col_name_);
  status = recover_id_map->open(idmap_path, false, false);
  ASSERT_TRUE(status.ok());

  delete_store_path =
      FileHelper::MakeFilePath(col_path_, FileID::DELETE_FILE,
                               recovered_version.delete_snapshot_path_suffix());
  auto recover_delete_store =
      DeleteStore::CreateAndLoad(col_name_, delete_store_path);
  ASSERT_TRUE(recover_delete_store != nullptr);

  options_.read_only_ = true;
  auto open_result =
      Segment::Open(col_path_, *schema_, *persist_metas[0], recover_id_map,
                    recover_delete_store, recover_version_mgr, options_);
  ASSERT_TRUE(open_result.has_value());
  segment = std::move(open_result).value();
  ASSERT_TRUE(segment != nullptr);

  auto fetched_before_add = segment->fetch({"id"}, {0, doc_count - 1});
  ASSERT_TRUE(fetched_before_add != nullptr);
  ASSERT_EQ(fetched_before_add->num_rows(), 2);

  const std::string added_column = "added_nullable";
  auto added_field =
      std::make_shared<FieldSchema>(added_column, DataType::INT32, true);
  status = segment->add_column(added_field, "", AddColumnOptions());
  ASSERT_TRUE(status.ok()) << status.message();

  std::vector<BlockID> added_scalar_block_ids;
  for (const auto &block : segment->meta()->persisted_blocks()) {
    if (block.type() == BlockType::SCALAR &&
        block.contain_column(added_column)) {
      added_scalar_block_ids.push_back(block.id());
    }
  }
  ASSERT_EQ(added_scalar_block_ids.size(), 1);
  const auto added_scalar_path = FileHelper::MakeForwardBlockPath(
      col_path_, writing_segment_meta->id(), added_scalar_block_ids[0], false);
  auto added_batch_lengths = ReadIpcRecordBatchLengths(added_scalar_path);
  ASSERT_TRUE(added_batch_lengths.ok())
      << added_batch_lengths.status().ToString();
  EXPECT_EQ(added_batch_lengths.ValueOrDie(),
            (std::vector<int64_t>{doc_count}));

  auto reader = segment->scan({"id", added_column});
  ASSERT_TRUE(reader != nullptr);
  std::vector<int64_t> merged_batch_lengths;
  int64_t total_rows = 0;
  while (true) {
    std::shared_ptr<arrow::RecordBatch> batch;
    auto read_status = reader->ReadNext(&batch);
    ASSERT_TRUE(read_status.ok()) << read_status.ToString();
    if (batch == nullptr) {
      break;
    }
    merged_batch_lengths.push_back(batch->num_rows());
    total_rows += batch->num_rows();
    auto added_array =
        batch->column(batch->schema()->GetFieldIndex(added_column));
    EXPECT_EQ(added_array->null_count(), batch->num_rows());
  }
  EXPECT_EQ(merged_batch_lengths, base_batch_lengths.ValueOrDie());
  EXPECT_EQ(total_rows, doc_count);

  const auto first_batch_length = base_batch_lengths.ValueOrDie().front();
  std::vector<int> row_ids = {static_cast<int>(first_batch_length - 1),
                              static_cast<int>(first_batch_length)};
  auto fetched = segment->fetch({"id", added_column}, row_ids);
  ASSERT_TRUE(fetched != nullptr);
  ASSERT_EQ(fetched->num_rows(), row_ids.size());
  auto fetched_added_column = fetched->GetColumnByName(added_column);
  ASSERT_TRUE(fetched_added_column != nullptr);
  EXPECT_EQ(fetched_added_column->null_count(), row_ids.size());

  // Each layout remains independently eligible for the mmap fetch fast path.
  auto fetched_base_only = segment->fetch({"id"}, row_ids);
  ASSERT_TRUE(fetched_base_only != nullptr);
  ASSERT_EQ(fetched_base_only->num_rows(), row_ids.size());
  auto fetched_added_only = segment->fetch({added_column}, row_ids);
  ASSERT_TRUE(fetched_added_only != nullptr);
  ASSERT_EQ(fetched_added_only->num_rows(), row_ids.size());
  auto fetched_added_only_column =
      fetched_added_only->GetColumnByName(added_column);
  ASSERT_TRUE(fetched_added_only_column != nullptr);
  EXPECT_EQ(fetched_added_only_column->null_count(), row_ids.size());
}

INSTANTIATE_TEST_SUITE_P(MMap, AddColumnBatchBoundariesTest,
                         testing::Values(true));

}  // namespace
