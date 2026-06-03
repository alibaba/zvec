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
#include <stdexcept>
#include <string>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include "db/common/file_helper.h"
#include "db/index/common/delete_store.h"
#include "db/index/common/id_map.h"
#include "db/index/common/version_manager.h"
#include "utils/utils.h"
#include "zvec/db/options.h"


namespace zvec {


class SegmentTest : public testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    ailego::LoggerBroker::SetLevel(ailego::Logger::LEVEL_WARN);
    zvec::ailego::MemoryLimitPool::get_instance().init(MIN_MEMORY_LIMIT_BYTES);

    FileHelper::RemoveDirectory(col_path_);
    FileHelper::CreateDirectory(col_path_);

    auto id_map_path = FileHelper::MakeFilePath(col_path_, FileID::ID_FILE, 0);
    id_map_ = IDMap::CreateAndOpen(col_name_, id_map_path, true, false);
    if (id_map_ == nullptr) {
      throw std::runtime_error("Failed to create id map");
    }

    delete_store_ = std::make_shared<DeleteStore>(col_name_);

    schema_ =
        test::TestHelper::CreateSchemaWithScalarIndex(false, false, col_name_);
    schema_->add_field(
        std::make_shared<FieldSchema>("id", DataType::INT32, false));
    schema_->add_field(
        std::make_shared<FieldSchema>("name", DataType::STRING, false));
    schema_->add_field(
        std::make_shared<FieldSchema>("age", DataType::UINT32, false));
    schema_->add_field(
        std::make_shared<FieldSchema>("binary", DataType::BINARY, false));
    schema_->add_field(std::make_shared<FieldSchema>(
        "array_binary", DataType::ARRAY_BINARY, false));

    bool enable_mmap = GetParam();

    Version version;
    version.set_schema(*schema_);
    version.set_enable_mmap(enable_mmap);
    auto version_manager_tmp = VersionManager::Create(col_path_, version);
    if (!version_manager_tmp.has_value()) {
      throw std::runtime_error("Failed to create version manager");
    }

    version_manager_ = version_manager_tmp.value();

    options_.read_only_ = false;
    options_.enable_mmap_ = enable_mmap;
    options_.max_buffer_size_ = 64 * 1024 * 1024;
  }

  void TearDown() override {
    id_map_.reset();
    delete_store_.reset();
    version_manager_.reset();

    FileHelper::RemoveDirectory(col_path_);
  }

 public:
  std::string GetColPath() {
    return col_path_;
  }

 protected:
  std::string col_name_ = "test_segment";
  std::string col_path_ = "./test_collection";
  IDMap::Ptr id_map_;
  DeleteStore::Ptr delete_store_;
  VersionManager::Ptr version_manager_;
  CollectionSchema::Ptr schema_;
  SegmentOptions options_;
};


}  // namespace zvec
